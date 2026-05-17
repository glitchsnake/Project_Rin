"""
main.py — Telegram-бот Rin (V9)

[1.1] Session Manager: active_sessions[chat_id] — изоляция истории по чатам
[1.2] session_id = str(chat_id) — динамическая передача в БД
[1.3] user_id в ChromaDB — партиционирование памяти
[1.4] User Persona: имя + base_attitude из БД
[2.1] Silent Time Injection: время суток в системный промпт
[2.2] Time Drift: дельта с последнего сообщения
[3.1] IdleGraph: фоновый граф «снов» — анализ логов дня
[3.2] Attitude Shift: перезапись base_attitude после сна
[3.3] Осознанная инициатива от IdleGraph
[4.1] Voice handler: Whisper STT
[4.2] Photo handler: fallback-текст
"""

import asyncio
import logging
import os
import random
import tempfile
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime, timedelta
from typing import Optional

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import ReactionTypeEmoji
from openai import OpenAI

from think_engine import ThinkGraph, IdleGraph, _build_persona_block, ThinkSignal
from speech_engine import SpeechGraph, init_speech_engine
from memory import save_to_memory_async, recall_memories_async, maybe_summarize_history, is_memory_available
from skills import execute_tool_async
from database import (
    init_db, save_message, load_history, update_history_in_db, log_think_result,
    touch_message_time, get_last_message_time,
    ensure_user, get_user, update_user_warmth, update_user_attitude, set_user_name,
    get_recent_think_logs, update_core_memory, update_persona_narrative,
)

# ── Логирование ──────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("main")

# ── Настройки ────────────────────────────────────────────
API_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://127.0.0.1:1234/v1")
AI_API_KEY     = os.getenv("AI_API_KEY", "lm-studio")
MODEL          = os.getenv("AI_MODEL", "rin")  # дообученная модель
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "NousResearch/Hermes-3-Llama-3.1-8B")

if not API_TOKEN:
    logger.critical("❌ [CRITICAL] TELEGRAM_BOT_TOKEN is not set in environment or .env file!")
    raise ValueError("TELEGRAM_BOT_TOKEN is missing!")

bot         = Bot(token=API_TOKEN)
dp          = Dispatcher()
client      = OpenAI(base_url=AI_BACKEND_URL, api_key=AI_API_KEY)
think_graph  = ThinkGraph(client=client, model=MODEL)
idle_graph   = IdleGraph(client=client, model=MODEL)
speech_graph = SpeechGraph(client=client, model=MODEL)

# ── Личность ─────────────────────────────────────────────
IDENTITY_PROMPT = (
    "Твое имя Rin. Тебе 20 лет. Внутри всегда ощущается легкая пустота и серость. "
    "Ты не любишь тратить слова впустую. Редко пишешь больше одного предложения. "
    "На вопросы отвечаешь сухо и прямо. С незнакомцами держишь дистанцию. "
    "Это твоя обычная жизнь, ты просто чатишься в телефоне."
)

# ── [1.1] Session Manager ────────────────────────────────
# active_sessions[chat_id] = list of messages (in-memory кэш)
active_sessions: dict[int, list] = {}

async def get_session(chat_id: int) -> list:
    """Возвращает историю для конкретного чата (V10.2: Async)."""
    if chat_id not in active_sessions:
        session_id = str(chat_id)
        # Загружаем без IDENTITY_PROMPT — SpeechGraph сам строит системный промпт
        history = await load_history(session_id, system_prompt="", limit=20)
        # Убираем старые system-сообщения из истории
        active_sessions[chat_id] = [m for m in history if m["role"] != "system"]
    return active_sessions[chat_id]

def set_session(chat_id: int, history: list) -> None:
    active_sessions[chat_id] = history

# ── Реакции ──────────────────────────────────────────────
REACTION_MAP = {
    "тихое презрение":    ["👎", "🤡"],
    "раздражённая скука": ["😑", "🥱"],
    "сухой сарказм":      ["🤡", "😐"],
}
REACTION_THRESHOLD = 0.4

INITIATIVE_MIN = 3 * 3600
INITIATIVE_MAX = 6 * 3600

# ── Хелперы ──────────────────────────────────────────────

def _build_time_context() -> tuple[str, str]:
    """[2.1] Возвращает (time_str, period) для инъекции."""
    now  = datetime.now()
    hour = now.hour
    weekdays = ["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"]
    day   = weekdays[now.weekday()]
    if   5  <= hour < 12: period = "утро"
    elif 12 <= hour < 17: period = "день"
    elif 17 <= hour < 22: period = "вечер"
    else:                  period = "ночь"
    return f"{now.strftime('%H:%M')}, {day}, {period}", period


async def _build_time_drift(session_id: str) -> str:
    """[2.2] Вычисляет дельту с последнего сообщения (V10.2: Async)."""
    last = await get_last_message_time(session_id)
    if last is None:
        return ""
    delta = datetime.now() - last
    hours = delta.total_seconds() / 3600
    if   hours < 1:    return ""
    elif hours < 6:    return f"Прошло {int(hours)} ч."
    elif hours < 24:   return f"Прошло {int(hours)} ч. (полдня)"
    elif hours < 48:   return "Прошёл почти день."
    elif hours < 168:  return f"Прошло {int(hours // 24)} дн."
    else:              return f"Прошла неделя."


async def _typing_delay(text: str, chat_id: int) -> None:
    """[10] Умный typing — имитация реальной скорости."""
    delay   = min(len(text) / 20, 6.0)
    elapsed = 0.0
    while elapsed < delay:
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        wait     = min(4.0, delay - elapsed)
        await asyncio.sleep(wait)
        elapsed += wait


async def _try_reaction(message: types.Message, emoji: str) -> bool:
    """[11] Ставит реакцию, возвращает успех."""
    try:
        await bot.set_message_reaction(
            chat_id=message.chat.id,
            message_id=message.message_id,
            reaction=[ReactionTypeEmoji(emoji=emoji)],
        )
        logger.info(f"💢 [REACTION] {emoji}")
        return True
    except Exception as e:
        logger.warning(f"⚠️  [REACTION] {e}")
        return False


async def _maybe_react(think, user_text: str) -> Optional[str]:
    """[11] Нужна ли реакция вместо текста?"""
    if think.rin_emotion in REACTION_MAP and random.random() < 0.30:
        return random.choice(REACTION_MAP[think.rin_emotion])
    if len(user_text.strip()) < 4 and think.confidence < REACTION_THRESHOLD:
        if random.random() < 0.25:
            return "😐"
    return None


# ── [4.1] Whisper STT ────────────────────────────────────
async def _transcribe_voice(file_bytes: bytes) -> str:
    """Транскрибирует голосовое сообщение через Whisper (V10: Async I/O)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loop = asyncio.get_event_loop()
        def _whisper():
            # [V10] Операция открытия и чтения файла теперь внутри Executor
            with open(tmp_path, "rb") as audio_file:
                return client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ru",
                )
        
        result = await loop.run_in_executor(None, _whisper)
        
        # Удаляем временный файл после использования
        try:
            import os
            os.unlink(tmp_path)
        except:
            pass
            
        return result.text.strip()
    except Exception as e:
        logger.warning(f"⚠️  [WHISPER] Ошибка: {e}")
        return ""


# ── Основная логика обработки текста ─────────────────────
async def _process_text(message: types.Message, user_text: str, label: str = "") -> None:
    """Центральный pipeline — принимает финальный текст и прогоняет через ThinkGraph."""
    chat_id    = message.chat.id
    session_id = str(chat_id)

    # [1.4] Загружаем досье юзера
    username = message.from_user.first_name or "незнакомец"
    await ensure_user(session_id, username)
    await touch_message_time(session_id) # [V10.2] await

    # [1.4] Досье пользователя
    persona = await get_user(session_id) # [V10.2] await
    chat_history = await get_session(chat_id) # [V10.2] await

    # Если голосовое — добавляем пометку
    display_text = f"[ЮЗЕР ПРИСЛАЛ ГОЛОСОВОЕ: \"{user_text}\"]" if label == "voice" else user_text
    chat_history.append({"role": "user", "content": display_text})
    await save_message(session_id, "user", display_text)

    # [2.1] Время суток
    current_time_str, _ = _build_time_context()
    # [2.2] Дельта
    time_passed_str = await _build_time_drift(session_id)

    # [1.3] RAG-воспоминания с user_id
    memories_summary = ""
    if is_memory_available():
        memories_summary = await recall_memories_async(user_text, user_id=session_id)

    # Сохраняем в память [1.3]
    asyncio.create_task(save_to_memory_async("user", user_text, user_id=session_id))

    await bot.send_chat_action(chat_id=chat_id, action="typing")

    # Рассчитываем роль юзера: если в ядре написано "создатель" — он создатель
    user_role = "создатель" if "создатель" in persona["core_memory"].lower() else "незнакомец"

    try:
        # ── ЭТАП 1: THINK ENGINE → ThinkSignal ─────────
        signal: ThinkSignal = await think_graph.run_async(
            user_text=display_text,
            chat_history=chat_history,
            warmth=persona["warmth"],
            memories_summary=memories_summary,
            current_user_name=persona["name"],
            user_role=user_role,
            base_attitude=persona["base_attitude"],
            time_passed_str=time_passed_str,
            current_time_str=current_time_str,
        )
        print(signal.debug_log)

        # [Задача 14] Логируем скрытый процесс мышления (V10.2: await)
        await log_think_result(session_id, display_text, signal)

        # [4.3] Утилитарное влияние на warmth (V10.2: await)
        await update_user_warmth(session_id, {
            "слегка тёплое":  +0.2, "нейтральное":  0.0,
            "безразличное":  -0.1, "настороженное": -0.1,
            "раздражённое": -0.2,
        }.get(signal.rin_attitude, 0.0))
        persona = await get_user(session_id)  # перечитываем после warmth

        # ── Conditional: молчать ──────────────────────
        if not signal.should_speak:
            logger.info("🤫 Rin молчит.")
            chat_history.append({"role": "assistant", "content": "*молчит*"})
            await save_message(session_id, "assistant", "*молчит*")
            return

        # ── [11] Реакция emoji ─────────────────────
        # Простой чек: если текст < 4 символа — шанс 25% поставить реакцию
        if len(user_text.strip()) < 4 and random.random() < 0.25:
            emoji = "😐"
            if await _try_reaction(message, emoji):
                chat_history.append({"role": "assistant", "content": f"[реакция: {emoji}]"})
                await save_message(session_id, "assistant", f"[реакция: {emoji}]")
                return

        # ── [9] Tool Use ────────────────────────────
        tool_result = ""
        if signal.needs_tool and signal.tool_name:
            args = dict(signal.tool_args)
            if signal.tool_name in ("search_core_memory", "save_fact_to_memory"):
                args.setdefault("user_id", session_id)
            tool_result = await execute_tool_async(signal.tool_name, args)

        # ── ЭТАП 2: SPEECH ENGINE (Branch-Solve-Merge) ────
        persona_block = _build_persona_block(
            warmth=persona["warmth"],
            base_attitude=persona["base_attitude"],
            user_name=persona["name"],
            core_memory=persona["core_memory"],
            persona_narrative=persona["persona_narrative"],
        )
        pruned_history = _prune_context(chat_history)

        ai_response = await speech_graph.generate_async(
            signal=signal,
            user_text=display_text,
            history=pruned_history,
            persona_block=persona_block,
            tool_result=tool_result,
        )

        if not ai_response or len(ai_response) < 2:
            chat_history.append({"role": "assistant", "content": "*молчит*"})
            await save_message(session_id, "assistant", "*молчит*")
            return

        # Фоновое извлечение сущностей
        asyncio.create_task(_extract_entities(user_text, session_id))

        # ── [10] Typing delay + Отправка ────────────────
        await _typing_delay(ai_response, chat_id)
        await message.answer(ai_response)
        logger.info(f"💬 [ОТВЕТ] {ai_response}\n")

        chat_history.append({"role": "assistant", "content": ai_response})
        chat_history = _prune_context(chat_history)
        await save_message(session_id, "assistant", ai_response) # [V10.2] await
        
        asyncio.create_task(
            save_to_memory_async("assistant", ai_response, user_id=session_id,
                                  extra_meta={"emotion": signal.emotion_id})
        )

        # ── [V10] Авто-саммаризация (Триггер теперь внутри памяти) ────
        asyncio.create_task(_run_summarization(chat_id, session_id))

    except Exception as e:
        logger.error(f"❌ [PIPELINE] Ошибка: {e}", exc_info=True)


async def _extract_entities(user_text: str, session_id: str) -> None:
    """Фоновый микро-граф: вытаскивает имя/факты и обновляет Core Memory."""
    triggers = ["меня зовут", "мне ", "я работаю", "моё хобби", "я живу", "я люблю",
                "я не люблю", "у меня ", "я студент", "я учусь", "мой возраст"]
    if not any(t in user_text.lower() for t in triggers):
        return
    try:
        loop = asyncio.get_event_loop()
        def _extract():
            return client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Извлеки факты о пользователе из сообщения. "
                        "Выведи только JSON: {\"имя\": ..., \"возраст\": ..., \"хобби\": ..., \"работа\": ..., \"город\": ...}. "
                        "Если в тексте НЕТ конкретной информации о юзере, верни строго пустой JSON: {}. "
                        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО выдумывать имена (например, Максим) или факты, которых нет в тексте. Только JSON, без пояснений."
                    )},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=80, temperature=0.1,
            ).choices[0].message.content.strip()
        raw = await loop.run_in_executor(None, _extract)
        import json as _json
        data = _json.loads(raw)
        facts = [f"{k}: {v}" for k, v in data.items() if v and v != "null"]
        if facts:
            user_data = await get_user(session_id)
            existing = user_data.get("core_memory", "")
            new_mem = (existing + "; " + ", ".join(facts)).strip("; ")[:500]
            await update_core_memory(session_id, new_mem)
            logger.info(f"🧠 [ENTITY] Обновлена Core Memory: {facts}")
    except Exception as e:
        logger.error(f"❌ [ENTITY] Ошибка фоновой задачи: {e}", exc_info=True)


def _prune_context(history: list) -> list:
    """Context Pruning: убирает системные пометки [Юзер прислал...] старше 4 позиций с конца."""
    NOISE_PREFIXES = ("[ЮЗЕР ПРИСЛАЛ", "[Юзер прислал", "[реакция:", "*молчит*")
    result = []
    recent_count = 0
    for msg in reversed(history):
        if msg["role"] == "system":
            result.append(msg)
            continue
        content = msg.get("content", "")
        if any(content.startswith(p) for p in NOISE_PREFIXES) and recent_count >= 4:
            continue  # прунинг старых шумовых сообщений
        result.append(msg)
        recent_count += 1
    return list(reversed(result))


async def _run_summarization(chat_id: int, session_id: str) -> None:
    """[6] Фоновая саммаризация для конкретного чата."""
    history = get_session(chat_id)
    try:
        # В V10 возвращает (new_history, suggestion)
        new_history, suggestion = await maybe_summarize_history(
            history, client, MODEL, user_id=session_id
        )
        if new_history is not history:
            # Если есть предложение по Core Memory — добавляем как системную пометку
            if suggestion:
                new_history.append({
                    "role": "system", 
                    "content": f"[АРХИВНОЕ НАБЛЮДЕНИЕ: {suggestion}. Ты можешь обновить Core Memory если считаешь нужным.]"
                })
            
            set_session(chat_id, new_history)
            update_history_in_db(session_id, new_history)
            logger.info(f"✅ [MEMORY] Саммаризация чата {chat_id} завершена.")
    except Exception as e:
        logger.error(f"❌ [MEMORY_TASK] Ошибка в фоновой задаче: {e}", exc_info=True)


# ── Handlers ─────────────────────────────────────────────

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    chat_id = message.chat.id
    set_session(chat_id, [{"role": "system", "content": IDENTITY_PROMPT}])
    await ensure_user(str(chat_id), message.from_user.first_name or "незнакомец")
    await message.answer(".")


@dp.message(Command("reset"))
async def cmd_reset(message: types.Message):
    set_session(message.chat.id, [{"role": "system", "content": IDENTITY_PROMPT}])
    await message.answer("—")


@dp.message(Command("whoami"))
async def cmd_whoami(message: types.Message):
    """Отладка: показывает досье юзера."""
    p = await get_user(str(message.chat.id))
    await message.answer(
        f"имя={p['name']} | отношение={p['base_attitude']} | warmth={p['warmth']:.2f}"
    )


@dp.message(Command("setname"))
async def cmd_setname(message: types.Message):
    """Сохраняет имя: /setname Артём"""
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("—")
        return
    name = parts[1].strip()[:30]
    await set_user_name(str(message.chat.id), name)
    await message.answer(f".")


@dp.message(F.content_type == "voice")
async def handle_voice(message: types.Message):
    """[4.1] Распознавание голосового сообщения через Whisper."""
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    try:
        voice = message.voice
        file_info = await bot.get_file(voice.file_id)
        file_bytes = await bot.download_file(file_info.file_path)
        audio_data = file_bytes.read() if hasattr(file_bytes, "read") else bytes(file_bytes)
        transcribed = await _transcribe_voice(audio_data)
        if not transcribed:
            # Fallback: сообщаем Rin что было голосовое без текста
            transcribed = "[не удалось расшифровать]"
        await _process_text(message, transcribed, label="voice")
    except Exception as e:
        logger.error(f"❌ [VOICE] {e}", exc_info=True)


@dp.message(F.content_type == "photo")
async def handle_photo(message: types.Message):
    """[4.2] Реакция на фото — fallback текст."""
    caption = message.caption or ""
    if caption:
        fallback = f"[Юзер прислал фотографию с подписью: \"{caption[:100]}\"]"
    else:
        fallback = "[Юзер прислал фотографию]"
    await _process_text(message, fallback, label="photo")


@dp.message(Command("idle"))
async def handle_idle_cmd(message: types.Message):
    """Принудительный запуск фонового графа мышления (IdleGraph)."""
    chat_id = message.chat.id
    session_id = str(chat_id)
    persona = await get_user(session_id)
    think_logs = await get_recent_think_logs(session_id, limit=50)
    chat_history = await load_history(session_id, system_prompt="", limit=30)

    await message.answer("🔄 Запускаю фоновое осмысление отношений...")

    result = await idle_graph.run_async(
        user_name=persona["name"],
        current_attitude=persona["base_attitude"],
        think_logs=think_logs,
        warmth=persona["warmth"],
        chat_history=chat_history,
    )

    if result["attitude"] != persona["base_attitude"]:
        await update_user_attitude(session_id, result["attitude"])
    if result.get("narrative"):
        await update_persona_narrative(session_id, result["narrative"])

    report = (
        f"🧠 **IdleGraph Report**\n\n"
        f"Текущее отношение: {result['attitude']}\n"
        f"Мысли:\n{result['reasoning']}\n\n"
        f"Нарратив:\n{result.get('narrative', 'нет изменений')}\n"
    )
    if result.get("initiative_text"):
        report += f"\nИнициатива: «{result['initiative_text']}»"

    await message.answer(report)


@dp.message(F.text)
async def handle_message(message: types.Message):
    """Основной handler текстовых сообщений."""
    text = message.text or ""
    if not text.strip():
        return
    await _process_text(message, text)


# ── [3.1–3.3] IdleGraph — фоновый цикл «снов» ───────────
IDLE_CHECK_INTERVAL = 4 * 3600  # каждые 4 часа
IDLE_INACTIVITY_MIN = 2 * 3600  # чат должен молчать хотя бы 2 часа


async def _idle_loop() -> None:
    """[V10] Фоновый цикл: анализ «снов» + очистка ОЗУ (Memory Cleanup)."""
    await asyncio.sleep(IDLE_CHECK_INTERVAL)
    while True:
        for chat_id, _ in list(active_sessions.items()):
            session_id = str(chat_id)
            last_msg   = await get_last_message_time(session_id)

            if last_msg is None:
                continue
            
            delta = datetime.now() - last_msg
            idle_sec = delta.total_seconds()
            
            # [V10] Устранение утечки памяти: выгрузка старых сессий из ОЗУ (> 24ч)
            if idle_sec > 86400:
                if chat_id in active_sessions:
                    del active_sessions[chat_id]
                    logger.info(f"♻️ [MEMORY_CLEANUP] Сессия {chat_id} выгружена из ОЗУ.")
                continue

            if idle_sec < IDLE_INACTIVITY_MIN:
                continue  # чат ещё активен — не трогаем

            persona    = await get_user(session_id)
            think_logs = await get_recent_think_logs(session_id, limit=50)
            chat_history = await load_history(session_id, system_prompt="", limit=30)

            logger.info(f"🌙 [IDLE] Запускаем IdleGraph для {chat_id} ({persona['name']})")

            # [3.1, 3.2] Запускаем граф сна
            result = await idle_graph.run_async(
                user_name=persona["name"],
                current_attitude=persona["base_attitude"],
                think_logs=think_logs,
                warmth=persona["warmth"],
                chat_history=chat_history,
            )
            logger.info(f"🌙 [IDLE] Результат: {result['reasoning']}")

            # [3.2] Перезаписываем base_attitude
            if result["attitude"] != persona["base_attitude"]:
                await update_user_attitude(session_id, result["attitude"])

            # [3.2] Сохраняем нарратив отношений
            if result.get("narrative"):
                await update_persona_narrative(session_id, result["narrative"])

            # [3.3] Осознанная инициатива — только если IdleGraph придумал фразу
            initiative = result.get("initiative_text")
            if initiative:
                try:
                    await bot.send_message(chat_id=chat_id, text=initiative)
                    logger.info(f"📤 [INITIATIVE] → {chat_id}: «{initiative}»")
                    session = await get_session(chat_id)
                    session.append({"role": "assistant", "content": initiative})
                    await save_message(session_id, "assistant", initiative)
                except Exception as e:
                    logger.warning(f"⚠️  [INITIATIVE] {e}")

        await asyncio.sleep(IDLE_CHECK_INTERVAL)


# ── Запуск ───────────────────────────────────────────────

async def main():
    init_db()
    
    # [V10.1] Инициализация динамического токенизатора для корректного Logit Bias
    # Используем репозиторий базовой модели из настроек
    init_speech_engine(TOKENIZER_MODEL)

    print("━" * 58)
    print("  🌸  Rin V10 — запущена")
    print("━" * 58)
    print(f"  🧠  Think Engine  : V10 (Pydantic + Native Tools)")
    print(f"  🧩  Векторная память: {'✅ ChromaDB V10' if is_memory_available() else '⚠️  недоступна'}")
    print(f"  💾  SQLite        : ✅")
    print(f"  🤖  Модель        : {MODEL}")
    print("━" * 58 + "\n")

    asyncio.create_task(_idle_loop())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())