"""
think_engine.py — Rin Cognitive Thinking Module (V10: Structured Outputs + Pydantic + Native Tools)

Architectural Inspiration:
  • OpenAI Structured Outputs → Native JSON parsing via Pydantic.
  • LangGraph (LC)             → Adaptive branching: System 1 (Fast) vs System 2 (Deep).
"""

import asyncio
import functools
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("think_engine")

# ════════════════════════════════════════════════════════
#  Data Contracts
# ════════════════════════════════════════════════════════

@dataclass
class ThinkSignal:
    emotion_id: str
    tactic_id: str
    warmth: float
    should_speak: bool
    needs_tool: bool
    tool_name: Optional[str]
    tool_args: Optional[dict]
    hidden_intent: str
    rin_attitude: str
    debug_log: str


class ThinkState:
    def __init__(self, user_text: str, **kwargs):
        self.user_text: str = user_text
        self.user_role: str = kwargs.get("user_role", "stranger")
        self.current_user_name: str = kwargs.get("user_name", "User")
        self.base_attitude: str = kwargs.get("base_attitude", "neutral")
        self.warmth: float = kwargs.get("warmth", 0.0)
        self.history_summary: str = kwargs.get("history_summary", "")
        self.memories_summary: str = kwargs.get("memories_summary", "")
        
        # Time constraints
        self.time_passed_str: str = ""
        self.current_time_str: str = ""
        
        # Analysis results
        self.fact_analysis: str = ""
        self.hidden_intent: str = ""
        self.rin_inner_conflict: str = ""
        self.rin_emotion: str = "boredom"
        self.rin_attitude: str = "neutral"
        self.response_tactic: str = "short response"
        
        # Flags
        self.should_ignore: bool = False
        self.needs_tool: bool = False
        self.tool_name: Optional[str] = None
        self.tool_args: Optional[dict] = None
        self.confidence: float = 1.0
        self.error: Optional[str] = None

# ════════════════════════════════════════════════════════
#  Valid Parameter Lists (Enums aligned with fine-tune dataset)
# ════════════════════════════════════════════════════════

VALID_EMOTIONS = {
    "задумчивость", "скука", "растерянность", "сухой сарказм",
    "тихое презрение", "раздражение", "равнодушие", "отстраненность",
    "усталая нежность", "грусть", "щемящая пустота", "тихое тепло"
}

VALID_ATTITUDES = {
    "отстраненное", "настороженное", "нейтральное", "заинтересованное",
    "тёплое и доверительное"
}

VALID_TACTICS = {
    "короткая реакция", "сухой ответ", "мягкое слушание", "игнорирование"
}

TACTIC_LENGTH = {
    "короткая реакция": 15,
    "сухой ответ": 25,
    "мягкое слушание": 55,
    "игнорирование": 1
}

# ════════════════════════════════════════════════════════
#  Native Tool Declarations (Function Schemas)
# ════════════════════════════════════════════════════════

NATIVE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the exact current time and day of the week.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for objective facts (who is, what is).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run a complex calculation or execute sandboxed Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_core_memory",
            "description": "Query long-term memories for dynamic user profiles details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Information topic query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_fact_to_memory",
            "description": "Save a verified factual preference or event to user's memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "Fact statement"}
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_real_world_context",
            "description": "Load real-world environments like current weather or forecast stats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "info_type": {"type": "string", "enum": ["weather"], "description": "Context type"}
                },
                "required": ["city"]
            }
        }
    }
]

# ════════════════════════════════════════════════════════
#  System 2 Schemas (Structured Pydantic Model)
# ════════════════════════════════════════════════════════

class LLMThinkOutput(BaseModel):
    """Rin Structured Cognitive Architecture (System 2)."""
    fact_analysis: str = Field(..., description="Objective breakdown of the user's input message")
    hidden_intent: str = Field(..., description="Perceived hidden motive or goal of the user")
    rin_inner_conflict: str = Field(..., description="Rin's internal reflection (FACT -> ANALYSIS -> REACTION)")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analysis confidence level")
    rin_emotion: str = Field(..., description="Rin's current mapped emotion from valid list")
    rin_attitude: str = Field(..., description="Rin's current relationship stance towards the user")
    response_tactic: str = Field(..., description="Chosen conversational response tactic")

    @field_validator('rin_emotion')
    def validate_emotion(cls, v):
        if v not in VALID_EMOTIONS:
            logger.warning(f"⚠️ [THINK] Unknown emotion '{v}' fallback to 'отстраненность'")
            return "отстраненность"
        return v

    @field_validator('rin_attitude')
    def validate_attitude(cls, v):
        if v not in VALID_ATTITUDES:
            logger.warning(f"⚠️ [THINK] Unknown attitude '{v}' fallback to 'нейтральное'")
            return "нейтральное"
        return v

    @field_validator('response_tactic')
    def validate_tactic(cls, v):
        if v not in VALID_TACTICS:
            logger.warning(f"⚠️ [THINK] Unknown tactic '{v}' fallback to 'короткая реакция'")
            return "короткая реакция"
        return v

# ════════════════════════════════════════════════════════
#  Utilities
# ════════════════════════════════════════════════════════

_VULNERABILITY_SIGNALS = ["грустно", "плохо", "устал", "одиноко", "больно", "😢", "😭", "😔", "😞", ":(", "тяжело", "не могу"]

def _build_persona_block(warmth: float, base_attitude: str, user_name: str,
                         core_memory: str, persona_narrative: str) -> str:
    """Assembles the dynamic persona context block for speech generation."""
    base = (
        f"Твое имя Rin. Тебе 20 лет. Внутри всегда ощущается легкая пустота и серость. "
        f"Ты не любишь тратить слова впустую. Редко пишешь больше одного предложения.\n"
        f"Текущий собеседник: {user_name}. Базовое отношение: {base_attitude}.\n"
    )
    if core_memory:
         base += f"[Факты из долгосрочной памяти]: {core_memory}\n"
    if persona_narrative:
         base += f"[История отношений]: {persona_narrative}\n"
         
    # Warmth gradients steering tone narrative
    if warmth < 0:
        style = "ДИНАМИКА: Собеседник неприятен. Сухо, отстраненно. Не пиши лишних букв."
    elif warmth <= 0.5:
        style = "ДИНАМИКА: Знакомый. Дистанция, краткость. Редко пишешь больше одного предложения."
    else:
        style = "ДИНАМИКА: Близкий человек. Можно быть чуть теплее. Одно-два предложения с искренним теплом."
        
    anti_bleed = (
        "ВАЖНО: Запрещено выдавать свои мысли или мета-параметры. Пиши только саму фразу.\n"
        "Отвечай строго на языке собеседника (русском)."
    )
    return base + style + "\n\n" + anti_bleed

def _check_vulnerability(user_text: str, warmth: float, rin_emotion: str) -> str:
    """Emotion Gradient: overrides aggressive tones if user exhibits vulnerability."""
    if warmth <= 0: return rin_emotion
    text_lower = user_text.lower()
    if any(sig in text_lower for sig in _VULNERABILITY_SIGNALS):
        if rin_emotion in ["сухой сарказм", "тихое презрение", "раздражённая скука", "раздражение"]:
            logger.info(f"🧬 [THINK] Emotion Gradient Override: {rin_emotion} → равнодушие")
            return "равнодушие"
    return rin_emotion

# ════════════════════════════════════════════════════════
#  Graph Nodes (LC Core Layout)
# ════════════════════════════════════════════════════════

def _node_router(state: ThinkState) -> str:
    """Graph routing node: branches System 1 (fast matching) or System 2 (deep analysis)."""
    fast_triggers = ["привет", "как дела", "ясно", "понятно", "ок", "угу", "мм", "спокойной ночи", "доброе утро"]
    clean_text = state.user_text.strip().lower().rstrip("!?.,")
    
    if len(clean_text) < 25 and clean_text in fast_triggers:
        return "system_1"
    return "system_2"


def _node_system_1_fast_track(state: ThinkState) -> ThinkState:
    """System 1: Fast conversational path."""
    state.fact_analysis = "Simple interaction."
    state.hidden_intent = "Maintaining contact."
    state.rin_inner_conflict = "FACT: Small talk. ANALYSIS: Deep cognition not required. REACTION: Standard dry tone."
    state.confidence = 1.0
    state.rin_emotion = "скука"
    state.response_tactic = "короткая реакция"
    return state


def _node_build_system_2_prompt(state: ThinkState) -> tuple[str, str]:
    """System 2: Prompt assembly."""
    emotion_list = ", ".join(sorted(VALID_EMOTIONS))
    user_context = f"User: {state.current_user_name} ({state.user_role}). Base stance: {state.base_attitude}."
    time_info = f"Context: {state.current_time_str}. {state.time_passed_str}"

    system_msg = (
        "You are Rin's cognitive analysis engine (20yo, reserved, cold, slightly depressive).\n"
        f"{user_context}\n{time_info}\n\n"
        "Analyze the incoming user message and output structured JSON matching LLMThinkOutput schema.\n"
        "Fill 'rin_inner_conflict' strictly following: FACT -> ANALYSIS -> REACTION.\n"
        f"Available Emotions: {emotion_list}\n"
        f"Available Attitudes: {', '.join(VALID_ATTITUDES)}\n"
        f"Available Tactics: {', '.join(VALID_TACTICS)}\n"
    )
    parts = []
    if state.history_summary: parts.append(f"History context: {state.history_summary}")
    if state.memories_summary: parts.append(f"Retrieved memories: {state.memories_summary}")
    parts.append(f"Current User Message: \"{state.user_text}\"")

    return system_msg, "\n".join(parts)


def _node_system_2_deep_thought(state: ThinkState, client: OpenAI, model: str) -> ThinkState:
    """System 2: Structured LLM Parsing and Dispatch."""
    try:
        sys_msg, user_msg = _node_build_system_2_prompt(state)
        
        # Parallel execution: OpenAI Native Tool calling mapping
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ],
            tools=NATIVE_TOOLS,
            response_format=LLMThinkOutput
        )
        
        message = completion.choices[0].message
        
        # 1. Parse Tool Calls
        if message.tool_calls:
            tc = message.tool_calls[0].function
            state.needs_tool = True
            state.tool_name = tc.name
            try:
                import json
                state.tool_args = json.loads(tc.arguments)
            except:
                state.tool_args = {}
                
        # 2. Extract Structured Schema Stance
        if message.parsed:
            out: LLMThinkOutput = message.parsed
            state.fact_analysis = out.fact_analysis
            state.hidden_intent = out.hidden_intent
            state.rin_inner_conflict = out.rin_inner_conflict
            state.confidence = out.confidence
            state.rin_emotion = out.rin_emotion
            state.rin_attitude = out.rin_attitude
            state.response_tactic = out.response_tactic
            
    except Exception as e:
        logger.error(f"❌ [THINK] System 2 deep thought failure: {e}")
        state.error = str(e)
        state.rin_inner_conflict = f"Error in cognitive cycle: {e}"
        # safe fallback
        state.rin_emotion = "отстраненность"
        state.response_tactic = "короткая реакция"
        
    return state

# ════════════════════════════════════════════════════════
#  Public Interface Classes
# ════════════════════════════════════════════════════════

class ThinkGraph:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def run(self, user_text: str, warmth: float, **kwargs) -> ThinkSignal:
        """Runs the cognitive thinking graph sequentially synchronously."""
        state = ThinkState(user_text, warmth=warmth, **kwargs)
        
        state.time_passed_str = kwargs.get("time_passed_str", "")
        state.current_time_str = kwargs.get("current_time_str", "")
        
        # Load dialogue summaries
        history = kwargs.get("history", [])
        state.history_summary = self._summarize_history(history)
        state.memories_summary = kwargs.get("memories_summary", "")

        # Route Branching
        branch = _node_router(state)
        if branch == "system_1":
            state = _node_system_1_fast_track(state)
        else:
            state = _node_system_2_deep_thought(state, self.client, self.model)
            
        # Apply Emotion Gradient filter
        state.rin_emotion = _check_vulnerability(user_text, warmth, state.rin_emotion)
        
        # Ignores triggers
        if state.response_tactic == "игнорирование":
            state.should_ignore = True
            
        max_len = TACTIC_LENGTH.get(state.response_tactic, 40)
        
        ctx = (
            f"<thinking>\nANALYSIS: {state.fact_analysis}\n"
            f"INTENT: {state.hidden_intent}\n"
            f"THOUGHTS: {state.rin_inner_conflict}\n"
            f"EMOTION: {state.rin_emotion} | TACTIC: {state.response_tactic}\n"
            f"LENGTH: ~{max_len}\n</thinking>"
        )

        debug = [
            "┌─── 🧠 THINK ENGINE V10 ───",
            f"│ 📥 Input: {user_text[:50]}",
            f"│ 🎯 Intent: {state.hidden_intent[:45]}",
            f"│ 💜 Emotion: {state.rin_emotion}",
            f"│ 🧊 Attitude: {state.rin_attitude}",
            f"│ 🗣  Tactic: {state.response_tactic}",
            f"│ 🔧 Tool: {state.tool_name or 'none'}",
            "└───────────────────────────"
        ]

        return ThinkSignal(
            emotion_id=state.rin_emotion, tactic_id=state.response_tactic,
            warmth=warmth, should_speak=not state.should_ignore,
            needs_tool=state.needs_tool, tool_name=state.tool_name, tool_args=state.tool_args,
            hidden_intent=state.hidden_intent, rin_attitude=state.rin_attitude,
            debug_log="\n".join(debug)
        )

    async def run_async(self, *args, **kwargs) -> ThinkSignal:
        return await asyncio.get_event_loop().run_in_executor(None, functools.partial(self.run, *args, **kwargs))

    @staticmethod
    def _summarize_history(history: list) -> str:
        relevant = [m for m in history if m["role"] != "system"][-4:]
        return " | ".join([f"{'User' if m['role']=='user' else 'Rin'}: {m['content'][:40]}" for m in relevant])


class IdleGraph:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def run(self, user_name: str, current_attitude: str, think_logs: list[dict], warmth: float, **kwargs) -> dict:
        if not think_logs: return {"attitude": current_attitude, "narrative": "empty", "initiative_text": None}
        digest = "\n".join([f"• {l.get('user_text','')[:40]} -> {l.get('rin_emotion','')}" for l in think_logs[-20:]])
        
        sys = (
            f"You are IdleGraph (Rin). Analyze user logs of {user_name}.\n"
            f"Current warmth: {warmth:.2f}. Day's cognitive log digest:\n{digest}\n"
            "Respond in JSON matching: { 'new_attitude': str, 'narrative': str, 'initiative_text': str|null }"
        )

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": sys}],
                response_format={"type": "json_object"}
            )
            import json
            data = json.loads(completion.choices[0].message.content)
            return {
                "attitude": data.get("new_attitude", current_attitude),
                "narrative": data.get("narrative", ""),
                "initiative_text": data.get("initiative_text"),
                "reasoning": "Analysis complete."
            }
        except:
            return {"attitude": current_attitude, "narrative": "error", "initiative_text": None}

    async def run_async(self, *args, **kwargs) -> dict:
        return await asyncio.get_event_loop().run_in_executor(None, functools.partial(self.run, *args, **kwargs))
