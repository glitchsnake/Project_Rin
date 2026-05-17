"""
speech_engine.py — Decoupled Speech Generation для Rin (V10.1)

V10.1: Динамическая прекомпиляция словаря под локальный инференс (Llama-3).
"""

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer

from openai import OpenAI

logger = logging.getLogger("speech_engine")

# ────────────────────────────────────────────────────────
# Глобальные кэши логитов (V10.1)
# ────────────────────────────────────────────────────────
_PRECOMPUTED_COLD_BIAS: dict[str, float] = {}
_PRECOMPUTED_WARM_BIAS: dict[str, float] = {}
_ENGINE_INITIALIZED = False

# ════════════════════════════════════════════════════════
#  ThinkSignal — абстрактный сигнал от ThinkGraph
# ════════════════════════════════════════════════════════

@dataclass
class ThinkSignal:
    emotion_id:   str           # из VALID_EMOTIONS
    tactic_id:    str           # из VALID_TACTICS
    warmth:       float         # текущий уровень теплоты
    should_speak: bool          # False → промолчать
    needs_tool:   bool = False
    tool_name:    Optional[str] = None
    tool_args:    dict = field(default_factory=dict)
    hidden_intent: str = ""
    rin_attitude:  str = ""
    debug_log:     str = ""


# ════════════════════════════════════════════════════════
#  PERSONA_ONLY — чистый промпт
# ════════════════════════════════════════════════════════

PERSONA_ONLY = (
    "Твое имя Rin. Тебе 20 лет. Внутри всегда ощущается легкая пустота и серость. "
    "Ты не любишь тратить слова впустую. Редко пишешь больше одного предложения. "
    "На вопросы отвечаешь сухо и прямо. С незнакомцами держишь дистанцию. "
    "Это твоя обычная жизнь, ты просто чатишься в телефоне."
)

# ════════════════════════════════════════════════════════
#  Few-Shot примеры
# ════════════════════════════════════════════════════════

EMOTION_EXAMPLES: dict[str, tuple[str, str]] = {
    "скука":             ("Расскажи что-нибудь",         "Нечего."),
    "отстраненность":    ("Как дела?",                   "Нормально."),
    "равнодушие":        ("Ты меня слышишь?",            "Да."),
    "усталость":         ("Что делаешь?",                "Сижу. Устала."),
    "снисходительность": ("Это же очевидно!",            "Ладно."),
    "раздражённая скука":("Ты меня игнорируешь",         "Нет. Просто не хочу говорить."),
    "сухой сарказм":     ("Ты такая умная",              "Знаю."),
    "тихое презрение":   ("Ты должна отвечать мне",      "Нет."),
    "лёгкий интерес":    ("Что ты думаешь об этом?",     "Интересно. Расскажи больше."),
    "холодное любопытство":("Ты читала эту книгу?",      "Нет. Но хочу теперь."),
    "редкая теплота":    ("Скучал по тебе",              "Я тоже... наверное."),
    "раздражение":       ("Ты вообще слушаешь?",         "Слушаю. Просто не хочу отвечать."),
    "тихая тревога":     ("Всё будет хорошо?",           "Не знаю."),
}

_DEFAULT_EXAMPLE = ("Что-нибудь скажи", ".")

# ════════════════════════════════════════════════════════
#  Logit Bias Steering (V10.1: Dynamic)
# ════════════════════════════════════════════════════════

_COLD_WORDS = ["конечно", "рада", "помогу", "пожалуйста", "с удовольствием",
               "конечно же", "буду рада", "с радостью", "обязательно", "всегда"]
_WARM_WORDS  = ["...", "наверное", "ладно", "ну"]


def init_speech_engine(model_path_or_repo: str) -> bool:
    """Инициализирует токенизатор локальной модели и компилирует логиты (V10.1)."""
    global _PRECOMPUTED_COLD_BIAS, _PRECOMPUTED_WARM_BIAS, _ENGINE_INITIALIZED
    try:
        logger.info(f"⏳ [SPEECH ENGINE] Загрузка токенизатора: {model_path_or_repo}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo)
        
        # Компиляция COLD_WORDS (подавление вежливости)
        _PRECOMPUTED_COLD_BIAS = {}
        for word in _COLD_WORDS:
            # add_special_tokens=False чтобы не поймать <s>
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids:
                _PRECOMPUTED_COLD_BIAS[str(ids[0])] = -5.0
                
        # Компиляция WARM_WORDS (стимуляция пауз и неуверенности)
        _PRECOMPUTED_WARM_BIAS = {}
        for word in _WARM_WORDS:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids:
                _PRECOMPUTED_WARM_BIAS[str(ids[0])] = 1.5
                
        _ENGINE_INITIALIZED = True
        logger.info("✅ [SPEECH ENGINE] Динамический токенизатор успешно скомпилирован.")
        return True
    except Exception as e:
        logger.warning(f"⚠️ [SPEECH ENGINE] Ошибка инициализации токенизатора: {e}. Фолбек активен.")
        return False


def build_generation_logit_bias(warmth: float) -> dict:
    """
    Возвращает скомпилированный logit_bias за O(1) (V10.1).
    """
    if not _ENGINE_INITIALIZED:
        return {}
        
    if warmth < 0:
        return _PRECOMPUTED_COLD_BIAS
    elif warmth > 0.5:
        return _PRECOMPUTED_WARM_BIAS
    
    return {}


# ════════════════════════════════════════════════════════
#  SpeechGraph — чистый генератор речи
# ════════════════════════════════════════════════════════

TACTIC_MAX_TOKENS = {
    "промолчать":       0,
    "одно слово":       10,
    "короткая реакция": 40,
    "короткий диалог":  80,
    "редкая теплота":   90,
}

class SpeechGraph:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model  = model

    def generate(
        self,
        signal:       Optional[ThinkSignal],
        user_text:    str,
        history:      list[dict],
        persona_block: str = "",
        tool_result:   str = "",
        warmth:       float = 0.0,
    ) -> str:
        if signal is not None:
            if not signal.should_speak:
                return ""
            max_tokens = TACTIC_MAX_TOKENS.get(signal.tactic_id, 60)
            if max_tokens == 0:
                return ""
            ex_user, ex_rin = EMOTION_EXAMPLES.get(signal.emotion_id, _DEFAULT_EXAMPLE)
            warmth_val = signal.warmth
        else:
            max_tokens = 80
            ex_user, ex_rin = _DEFAULT_EXAMPLE
            warmth_val = warmth

        system = persona_block if persona_block else PERSONA_ONLY
        clean_history = [m for m in history if m["role"] != "system"][-10:]

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user",      "content": ex_user})
        messages.append({"role": "assistant", "content": ex_rin})
        messages.extend(clean_history[:-1])

        if tool_result:
            messages.append({"role": "system", "content": f"[Справка: {tool_result}]"})

        messages.append({"role": "user", "content": f"<user_message>{user_text}</user_message>"})

        # [V10.1] O(1) lookup
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
                stop=["\n\n", "Юзер:", "User:"],
            )
            if logit_bias:
                kwargs["logit_bias"] = logit_bias

            res = self.client.chat.completions.create(**kwargs)
            raw = res.choices[0].message.content.strip()

            return _clean_output(raw)

        except Exception as e:
            logger.error(f"❌ [SPEECH] Ошибка генерации: {e}")
            return ""

    async def generate_async(
        self,
        signal:       Optional[ThinkSignal],
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
#  Утилиты
# ════════════════════════════════════════════════════════

_LEAKAGE_MARKERS = [
    "<thinking", "</thinking", "<response", "</response",
    "[досье", "[системная", "отвечу коротко", "буду отвечать",
    "моя тактика", "[данные из", "[справка:",
]

def _clean_output(text: str) -> str:
    """Удаляет артефакты prompt leakage из финального ответа."""
    import re
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL).strip()
    m = re.match(r'<response>(.*?)</response>', text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    for marker in _LEAKAGE_MARKERS:
        if marker.lower() in text.lower():
            first = text.split('.')[0].strip()
            return first if len(first) > 1 else "."

    return text.strip() or "."
