"""
speech_engine.py — Dynamic Vocabulary Steerer and Speech Postprocessor for Rin (V10.1)

Performs dynamic logit steering using transformers.AutoTokenizer on Llama/GGUF/GPT models.
Implements O(1) fast precompiled steerings and filters prompt leakage artifacts.
"""

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict
from transformers import AutoTokenizer

from openai import OpenAI

logger = logging.getLogger("speech_engine")

# ── Dynamic steering constants ───────────────────────────
COOLDOWN_LIMIT = 50   # Steering vocabulary token limit per tactic

# Steered word arrays (influence relationship warmth and tone)
STEERING_WORDS_COLD = [
    "away", "okay", "whatever", "...", "sigh", "why", "stop", "no",
    "uninteresting", "leave", "silly", "nothing", "fine", "gray"
]
STEERING_WORDS_WARM = [
    "hello", "thanks", "quiet", "probably", "too", "listening", "why",
    "tell", "how", "little", "smiling", "together", "want"
]

# Engine global initialization states
_ENGINE_INITIALIZED = False
_TOKENIZER = None
_PRECOMPUTED_COLD_BIAS: Dict[str, float] = {}
_PRECOMPUTED_WARM_BIAS: Dict[str, float] = {}

# Prompt leakage markers (detected interior meta-motives to block)
_LEAKAGE_MARKERS = [
    "<thinking", "</thinking", "<response", "</response",
    "[ref:", "[observation:", "[archive", "my tactic", "selected tactic"
]

# Tactic length constraints (tokens)
TACTIC_LENGTH = {
    "short reaction":  15,
    "dry response":       25,
    "soft listening":   55,
    "ignoring":     1,
    "response tactic":    40
}

# Emotion examples for speech steering
EMOTION_EXAMPLES = {
    "warmth": ("are you okay?", "... yes. just a bit tired."),
    "coldness": ("do you want to talk?", "no. leave me alone."),
    "neutral": ("what are you doing?", "nothing special."),
}
_DEFAULT_EXAMPLE = ("hello", "...")

# ════════════════════════════════════════════════════════
#  Speech Engine Initializer (Startup Tokenizer Precomputation)
# ════════════════════════════════════════════════════════

def init_speech_engine(model_repo: str = "NousResearch/Hermes-3-Llama-3.1-8B"):
    """
    Loads model tokenizer from HuggingFace to compile O(1) logit steering index.
    Gracefully falls back to unsteered model operations if offline.
    """
    global _ENGINE_INITIALIZED, _TOKENIZER, _PRECOMPUTED_COLD_BIAS, _PRECOMPUTED_WARM_BIAS
    
    if _ENGINE_INITIALIZED:
        return True
        
    try:
        logger.info(f"⏳ [SPEECH] Loading dynamic tokenizer for vocabulary steering: '{model_repo}'...")
        
        # Load local or remote tokenizer configuration
        _TOKENIZER = AutoTokenizer.from_pretrained(model_repo, use_fast=True)
        
        # Helper function to map words to token IDs
        def get_token_ids(word: str) -> list[int]:
            # Encode with and without leading space to support both variations
            ids_with_space = _TOKENIZER.encode(" " + word, add_special_tokens=False)
            ids_no_space   = _TOKENIZER.encode(word, add_special_tokens=False)
            return list(set(ids_with_space + ids_no_space))

        # Precompute O(1) steering weights for Cold Vocabulary
        for word in STEERING_WORDS_COLD:
            for t_id in get_token_ids(word):
                _PRECOMPUTED_COLD_BIAS[str(t_id)] = 2.5  # moderate cold boost

        # Precompute O(1) steering weights for Warm Vocabulary
        for word in STEERING_WORDS_WARM:
            for t_id in get_token_ids(word):
                _PRECOMPUTED_WARM_BIAS[str(t_id)] = 1.8  # gentle warmth boost

        _ENGINE_INITIALIZED = True
        logger.info(f"✅ [SPEECH] dynamic logit steerings initialized. Steered tokens: cold={len(_PRECOMPUTED_COLD_BIAS)}, warm={len(_PRECOMPUTED_WARM_BIAS)}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ [SPEECH] Tokenizer could not be loaded ({e}). Dynamic steering disabled; falling back to standard LLM generation.")
        _ENGINE_INITIALIZED = False
        return False

# ════════════════════════════════════════════════════════
#  Logit Bias Constructor
# ════════════════════════════════════════════════════════

def build_generation_logit_bias(warmth: float) -> dict[str, float]:
    """
    Retrieves the compiled O(1) logit bias configuration based on user warmth level.
    Clamps and filters steering arrays to keep models aligned with core character rules.
    """
    if not _ENGINE_INITIALIZED:
        return {}

    # Extreme cold (warmth < 0) -> Steer cold words
    if warmth < 0:
        return _PRECOMPUTED_COLD_BIAS

    # Genuine warmth (warmth > 0.5) -> Steer warm words
    elif warmth > 0.5:
        return _PRECOMPUTED_WARM_BIAS

    # Neutral relationships (0.0 to 0.5) -> No active steering
    return {}

# ════════════════════════════════════════════════════════
#  SpeechGraph — Decoupled Speech Generation
# ════════════════════════════════════════════════════════

class SpeechGraph:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model  = model

    def generate(
        self,
        signal:       Optional[object],
        user_text:    str,
        history:      list[dict],
        persona_block: str = "",
        tool_result:   str = "",
        warmth:       float = 0.0,
    ) -> str:
        if signal is not None:
            if not getattr(signal, "should_speak", True):
                return ""
            tactic_id = getattr(signal, "tactic_id", "response tactic")
            max_tokens = TACTIC_LENGTH.get(tactic_id, 40)
            if max_tokens == 0:
                return ""
            emotion_id = getattr(signal, "emotion_id", "neutral")
            ex_user, ex_rin = EMOTION_EXAMPLES.get(emotion_id, _DEFAULT_EXAMPLE)
            warmth_val = getattr(signal, "warmth", warmth)
        else:
            max_tokens = 80
            ex_user, ex_rin = _DEFAULT_EXAMPLE
            warmth_val = warmth

        system = persona_block if persona_block else "Respond as Rin."
        clean_history = [m for m in history if m["role"] != "system"][-10:]

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user",      "content": ex_user})
        messages.append({"role": "assistant", "content": ex_rin})
        messages.extend(clean_history[:-1])

        if tool_result:
            messages.append({"role": "system", "content": f"[Reference: {tool_result}]"})

        messages.append({"role": "user", "content": f"<user_message>{user_text}</user_message>"})

        # O(1) logit bias lookup
        logit_bias = build_generation_logit_bias(warmth_val)

        temperature = 0.4 + warmth_val * 0.4
        temperature = max(0.3, min(0.9, temperature))

        try:
            kwargs = dict(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=0.5,
                presence_penalty=0.2,
                stop=["\n\n", "User:", "Rin:"],
            )
            if logit_bias:
                kwargs["logit_bias"] = logit_bias

            res = self.client.chat.completions.create(**kwargs)
            raw = res.choices[0].message.content.strip()

            return _clean_output(raw)

        except Exception as e:
            logger.error(f"❌ [SPEECH] Generation error: {e}")
            return ""

    async def generate_async(
        self,
        signal:       Optional[object],
        user_text:    str,
        history:      list[dict],
        persona_block: str = "",
        tool_result:   str = "",
        warmth:       float = 0.0,
    ) -> str:
        loop = asyncio.get_event_loop()
        fn = functools.partial(
            self.generate, signal, user_text, history, persona_block, tool_result, warmth
        )
        return await loop.run_in_executor(None, fn)

# ════════════════════════════════════════════════════════
#  Post-Processing Filters (Anti-Leakage Engine)
# ════════════════════════════════════════════════════════

def _clean_output(text: str) -> str:
    """Removes structured prompt leakage markers and reasoning tags from responses."""
    import re
    
    # Strip <thinking>...</thinking> blocks
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
    
    # Extract interior from <response>...</response> blocks if present
    m = re.match(r'<response>(.*?)</response>', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Detect prompt leaks and truncate sentence to prevent leaks
    for marker in _LEAKAGE_MARKERS:
        if marker.lower() in text.lower():
            logger.warning(f"⚠️ [SPEECH] Prompt leak detected in assistant text. Triggering truncation filter for: '{marker}'")
            first_sentence = text.split('.')[0].strip()
            return first_sentence if len(first_sentence) > 1 else "."

    return text.strip() or "."
