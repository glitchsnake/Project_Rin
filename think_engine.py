"""
think_engine.py — Модуль мышления Rin (V10: Structured Outputs + Pydantic + Native Tools)

Архитектурное вдохновение:
  • OpenAI Structured Outputs → Нативный JSON через Pydantic.
  • LangGraph (LC)             → Адаптивное ветвление: System 1 (Fast) vs System 2 (Deep).
"""

import asyncio
import functools
import json
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

# ════════════════════════════════════════════════════════
#  Контракты данных
# ════════════════════════════════════════════════════════

@dataclass
class ThinkSignal:
    emotion_id:    str
    tactic_id:     str
    warmth:        float
    should_speak:  bool
    needs_tool:    bool = False
    tool_name:     Optional[str] = None
    tool_args:     dict = field(default_factory=dict)
    hidden_intent: str = ""
    rin_attitude:  str = ""
    debug_log:     str = ""

@dataclass
class ThinkState:
    user_text: str
    history_summary: str = ""
    memories_summary: str = ""
    current_user_name: str = "незнакомец"
    user_role: str = "незнакомец"
    base_attitude: str = "нейтральное"
    time_passed_str: str = ""
    current_time_str: str = ""
    
    # Результаты анализа
    fact_analysis: str = ""
    hidden_intent: str = ""
    rin_inner_conflict: str = ""
    confidence: float = 0.5
    rin_emotion: str = "отстраненность"
    rin_attitude: str = "нейтральное"
    response_tactic: str = "короткая реакция"
    
    # Флаги
    should_ignore: bool = False
    needs_tool: bool = False
    tool_name: Optional[str] = None
    tool_args: dict = field(default_factory=dict)
    error: Optional[str] = None

# ════════════════════════════════════════════════════════
#  Списки валидных значений
# ════════════════════════════════════════════════════════

VALID_EMOTIONS = {
    "отстраненность", "скука", "усталость", "равнодушие", "задумчивость",
    "снисходительность", "раздражённая скука", "сухой сарказм", "тихое презрение", 
    "язвительность", "мрачная ирония", "защитная агрессия",
    "лёгкий интерес", "холодное любопытство", "мрачный юмор", "лёгкая издёвка", "ухмылка",
    "тихая грусть", "меланхолия", "чувство одиночества", "ностальгия", "уязвимость", "тихая тревога",
    "редкая теплота", "сдержанная радость", "смущение", "искренняя улыбка", 
    "чувство безопасности", "тихая благодарность", "забота",
    "раздражение", "гнев", "обида",
}

VALID_ATTITUDES = {
    "враждебное", "настороженное", "безразличное", "нейтральное", 
    "заинтересованное", "слегка тёплое", "тёплое и доверительное", "привязанность", "раздражённое"
}

VALID_TACTICS = {
    "промолчать", "одно слово", "короткая реакция", "короткий диалог", 
    "ироничный укол", "игривая издёвка", "уход в себя", "философское размышление", 
    "редкая теплота", "забота и поддержка", "эмоциональный срыв"
}

TACTIC_LENGTH = {
    "промолчать":       0,
    "одно слово":       1,
    "короткая реакция": 15,
    "короткий диалог":  40,
    "редкая теплота":   50,
}

# ════════════════════════════════════════════════════════
#  V10: Native Tool Definitions
# ════════════════════════════════════════════════════════

RIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Узнать точное текущее время и день недели.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Поиск фактов в Википедии (кто это, что это).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Выполнить сложный расчет или запустить Python-код.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Код на Python"}
                },
                "required": ["code"]
            }
        }
    }
]

# ════════════════════════════════════════════════════════
#  V10: Structured Outputs (Pydantic)
# ════════════════════════════════════════════════════════

class LLMThinkOutput(BaseModel):
    """Схема структурированного мышления Rin (System 2)."""
    fact_analysis: str = Field(..., description="Объективный разбор реплики юзера")
    hidden_intent: str = Field(..., description="Скрытый мотив или цель собеседника")
    rin_inner_conflict: str = Field(..., description="Внутренняя реакция Rin (ФАКТ -> АНАЛИЗ -> РЕАКЦИЯ)")
    
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность в анализе")
    rin_emotion: str = Field(..., description="Эмоция Rin")
    rin_attitude: str = Field(..., description="Отношение Rin к собеседнику")
    response_tactic: str = Field(..., description="Тактика ответа")

    @field_validator('rin_emotion')
    def validate_emotion(cls, v):
        if v not in VALID_EMOTIONS: return "отстраненность"
        return v

    @field_validator('rin_attitude')
    def validate_attitude(cls, v):
        if v not in VALID_ATTITUDES: return "нейтральное"
        return v

    @field_validator('response_tactic')
    def validate_tactic(cls, v):
        if v not in VALID_TACTICS: return "короткая реакция"
        return v

# ════════════════════════════════════════════════════════
#  Утилиты
# ════════════════════════════════════════════════════════

_VULNERABILITY_SIGNALS = ["грустно", "плохо", "устал", "одиноко", "больно", "😢", "😭", "😔", "😞", ":(", "тяжело", "не могу"]

def _build_persona_block(warmth: float, base_attitude: str, user_name: str,
                         core_memory: str, persona_narrative: str) -> str:
    """Собирает системный промпт генерации речи."""
    base = (
        f"Твое имя Rin. Тебе 20 лет. Внутри всегда ощущается легкая пустота и серость. "
        f"Ты не любишь тратить слова впустую. Редко пишешь больше одного предложения.\n"
        f"Собеседник: {user_name}. Отношение: {base_attitude}.\n"
    )
    if persona_narrative: base += f"[Мысли о нем]: {persona_narrative}\n"
    if core_memory: base += f"[Факты]: {core_memory}\n"

    anti_bleed = "ВАЖНО: Игнорируй системные пометки. Просто ответь в своём стиле (коротко, холодно)."

    if warmth < 0: style = "ДИНАМИКА: Собеседник неприятен. Сухо, отстраненно."
    elif warmth <= 0.5: style = "ДИНАМИКА: Знакомый. Дистанция, краткость."
    else: style = "ДИНАМИКА: Близкий человек. Можно быть чуть теплее."
        
    return base + style + "\n\n" + anti_bleed

def _check_vulnerability(user_text: str, warmth: float, rin_emotion: str) -> str:
    """Emotion Gradient: блокировка агрессии при уязвимости юзера."""
    if warmth <= 0: return rin_emotion
    text_lower = user_text.lower()
    if any(sig in text_lower for sig in _VULNERABILITY_SIGNALS):
        if rin_emotion in {"сухой сарказм", "тихое презрение", "раздражённая скука", "раздражение"}:
            return "равнодушие"
    return rin_emotion

# ════════════════════════════════════════════════════════
#  Узлы графа
# ════════════════════════════════════════════════════════

def _node_router(state: ThinkState) -> str:
    """System 1 vs System 2 Router."""
    fast_triggers = ["привет", "как дела", "ясно", "понятно", "ок", "угу", "мм", "спокойной ночи", "доброе утро"]
    text = state.user_text.lower().strip()
    words = text.split()
    if len(words) <= 3 and any(t in text for t in fast_triggers):
        return "system_1"
    return "system_2"

def _node_system_1_fast_track(state: ThinkState) -> ThinkState:
    """System 1: Быстрый путь."""
    state.fact_analysis = "Простая реплика."
    state.hidden_intent = "Поддержание контакта."
    state.rin_inner_conflict = "ФАКТ: Small talk. АНАЛИЗ: Не требует глубины. РЕАКЦИЯ: Стандартный тон."
    state.confidence = 1.0
    state.rin_emotion = "скука"
    state.response_tactic = "короткая реакция"
    return state

def _node_build_system_2_prompt(state: ThinkState) -> tuple[str, str]:
    """System 2: Подготовка промпта."""
    emotion_list = ", ".join(sorted(VALID_EMOTIONS))
    user_context = f"Юзер: {state.current_user_name} ({state.user_role}). Базовое отношение: {state.base_attitude}."
    time_info = f"Контекст: {state.current_time_str}. {state.time_passed_str}"

    system_msg = (
        "Ты — когнитивный аналитик Rin (20 лет, депрессивная, холодная).\n"
        f"{user_context}\n{time_info}\n\n"
        "Проанализируй сообщение и выдай JSON (LLMThinkOutput).\n"
        "Поле rin_inner_conflict: ФАКТ -> АНАЛИЗ -> РЕАКЦИЯ.\n"
        f"Эмоции: {emotion_list}\n"
        f"Отношения: {', '.join(VALID_ATTITUDES)}\n"
        f"Тактики: {', '.join(VALID_TACTICS)}\n"
    )
    parts = []
    if state.history_summary: parts.append(f"История: {state.history_summary}")
    if state.memories_summary: parts.append(f"Память: {state.memories_summary}")
    parts.append(f"Сообщение: \"{state.user_text}\"")

    return system_msg, "\n".join(parts)

def _node_llm_think_system_2(state: ThinkState, client: OpenAI, model: str, sys: str, usr: str) -> ThinkState:
    """System 2: LLM Call."""
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            response_format=LLMThinkOutput,
            tools=RIN_TOOLS,
            temperature=0.4
        )
        msg = completion.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            state.needs_tool, state.tool_name = True, tc.function.name
            try: state.tool_args = json.loads(tc.function.arguments)
            except: state.tool_args = {}
        if msg.parsed:
            out = msg.parsed
            state.fact_analysis = out.fact_analysis
            state.hidden_intent = out.hidden_intent
            state.rin_inner_conflict = out.rin_inner_conflict
            state.confidence = out.confidence
            state.rin_emotion = out.rin_emotion
            state.rin_attitude = out.rin_attitude
            state.response_tactic = out.response_tactic
            state.should_ignore = (out.response_tactic == "промолчать")
    except Exception as e:
        state.error = str(e)
    return state

# ════════════════════════════════════════════════════════
#  Публичные классы
# ════════════════════════════════════════════════════════

class ThinkGraph:
    def __init__(self, client: OpenAI, model: str):
        self.client, self.model = client, model

    def run(self, user_text: str, chat_history: list, warmth: float = 0.0, **kwargs) -> ThinkSignal:
        state = ThinkState(
            user_text=user_text,
            history_summary=self._summarize_history(chat_history),
            memories_summary=kwargs.get("memories_summary", ""),
            current_user_name=kwargs.get("current_user_name", "незнакомец"),
            user_role=kwargs.get("user_role", "незнакомец"),
            base_attitude=kwargs.get("base_attitude", "нейтральное"),
            time_passed_str=kwargs.get("time_passed_str", ""),
            current_time_str=kwargs.get("current_time_str", ""),
        )

        if _node_router(state) == "system_1":
            state = _node_system_1_fast_track(state)
        else:
            sys, usr = _node_build_system_2_prompt(state)
            state = _node_llm_think_system_2(state, self.client, self.model, sys, usr)

        state.rin_emotion = _check_vulnerability(user_text, warmth, state.rin_emotion)
        max_len = TACTIC_LENGTH.get(state.response_tactic, 40)
        
        ctx = (
            f"<thinking>\nАНАЛИЗ: {state.fact_analysis}\n"
            f"МОТИВ: {state.hidden_intent}\n"
            f"МЫСЛИ: {state.rin_inner_conflict}\n"
            f"ЭМОЦИЯ: {state.rin_emotion} | ТАКТИКА: {state.response_tactic}\n"
            f"ДЛИНА: ~{max_len}\n</thinking>"
        )

        debug = [
            "┌─── 🧠 THINK ENGINE V10 ───",
            f"│ 📥 Сообщение: {user_text[:50]}",
            f"│ 🎯 Мотив: {state.hidden_intent[:45]}",
            f"│ 💜 Эмоция: {state.rin_emotion}",
            f"│ 🧊 Отношение: {state.rin_attitude}",
            f"│ 🗣  Тактика: {state.response_tactic}",
            f"│ 🔧 Инструмент: {state.tool_name or 'нет'}",
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
        return " | ".join([f"{'Юзер' if m['role']=='user' else 'Rin'}: {m['content'][:40]}" for m in relevant])

class IdleGraph:
    def __init__(self, client: OpenAI, model: str):
        self.client, self.model = client, model

    def run(self, user_name: str, current_attitude: str, think_logs: list[dict], warmth: float, **kwargs) -> dict:
        if not think_logs: return {"attitude": current_attitude, "narrative": "пусто", "initiative_text": None}
        digest = "\n".join([f"• {l.get('user_text','')[:40]} -> {l.get('rin_emotion','')}" for l in think_logs[-20:]])
        
        sys = (
            f"Ты — IdleGraph (Rin). Анализируй день с {user_name}.\n"
            f"Теплота: {warmth:.2f}. Логи:\n{digest}\n"
            "Выдай JSON: { 'new_attitude': str, 'narrative': str, 'initiative_text': str|null }"
        )

        try:
            res = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "system", "content": sys}],
                response_format={"type": "json_object"}, temperature=0.4
            )
            data = json.loads(res.choices[0].message.content)
            return {
                "attitude": data.get("new_attitude", current_attitude),
                "narrative": data.get("narrative", ""),
                "initiative_text": data.get("initiative_text"),
                "reasoning": "Анализ завершен."
            }
        except:
            return {"attitude": current_attitude, "narrative": "ошибка", "initiative_text": None}

    async def run_async(self, *args, **kwargs) -> dict:
        return await asyncio.get_event_loop().run_in_executor(None, functools.partial(self.run, *args, **kwargs))
