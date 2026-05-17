import os
import re
import uuid
import logging
import asyncio
import tempfile
import random
from datetime import datetime, timedelta
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Инициализация логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("main")

# Настройки ИИ и Токенизатора
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://127.0.0.1:1234/v1")
AI_API_KEY     = os.getenv("AI_API_KEY", "lm-studio")
MODEL          = os.getenv("AI_MODEL", "rin")  # дообученная модель
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "NousResearch/Hermes-3-Llama-3.1-8B")

# Инициализация OpenAI-клиента
client = OpenAI(base_url=AI_BACKEND_URL, api_key=AI_API_KEY)

# Импорты ядра Rin (импортируются лениво или напрямую)
from database import (
    init_db, get_user, ensure_user, save_message, load_history,
    update_user_warmth, update_user_attitude, update_persona_narrative,
    get_recent_think_logs, get_last_message_time, update_history_in_db
)
from memory import (
    save_to_memory_async, is_memory_available, count_history_tokens,
    maybe_summarize_history
)
from think_engine import ThinkGraph, IdleGraph, ThinkState
from speech_engine import SpeechGraph, init_speech_engine
from semantic_cache import get_semantic_cache_async, save_semantic_cache_async
from semantic_router_engine import route_message, init_router
from skills import execute_tool_async

# Графы мышления
think_graph  = ThinkGraph(client=client, model=MODEL)
idle_graph   = IdleGraph(client=client, model=MODEL)
speech_graph = SpeechGraph(client=client, model=MODEL)

# Личность Rin
IDENTITY_PROMPT = (
    "Твое имя Rin. Тебе 20 лет. Внутри всегда ощущается легкая пустота и серость. "
    "Ты не любишь тратить слова впустую. Редко пишешь больше одного предложения. "
    "На вопросы отвечаешь сухо и прямо. С незнакомцами держишь дистанцию. "
    "Это твоя обычная жизнь, ты просто чатишься в телефоне."
)

# ── [1.1] Session Manager ────────────────────────────────
# active_sessions[session_id] = list of messages (in-memory кэш)
active_sessions: dict[str, list] = {}

async def get_session(session_id: str) -> list:
    """Возвращает историю для конкретной сессии (V10.2: Async)."""
    if session_id not in active_sessions:
        # Загружаем без IDENTITY_PROMPT — SpeechGraph сам строит системный промпт
        history = await load_history(session_id, system_prompt="", limit=20)
        # Убираем старые system-сообщения из истории
        active_sessions[session_id] = [m for m in history if m["role"] != "system"]
    return active_sessions[session_id]


def set_session(session_id: str, history: list) -> None:
    """Устанавливает историю сессии в ОЗУ."""
    active_sessions[session_id] = [m for m in history if m["role"] != "system"]


# ── [Rate Limiter] ───────────────────────────────────────
_user_message_times: dict[int, list[datetime]] = {}
RATE_LIMIT_SECONDS = 3.0

def _check_rate_limit(chat_id: int) -> bool:
    """Ограничивает частоту запросов (не чаще 1 раза в 3 секунды для одного чата)."""
    now = datetime.now()
    if chat_id in _user_message_times:
        # Очищаем старые записи
        times = _user_message_times[chat_id]
        times = [t for t in times if now - t < timedelta(seconds=RATE_LIMIT_SECONDS)]
        _user_message_times[chat_id] = times
        if len(times) >= 1:
            return False
    else:
        _user_message_times[chat_id] = []
    
    _user_message_times[chat_id].append(now)
    return True


# ── [4.1] Whisper STT (Async) ────────────────────────────
async def _transcribe_voice(file_bytes: bytes) -> str:
    """Транскрибирует голосовое сообщение через Whisper (V10: Async I/O)."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loop = asyncio.get_event_loop()
        def _whisper():
            with open(tmp_path, "rb") as audio_file:
                return client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ru",
                )
        
        result = await loop.run_in_executor(None, _whisper)
        
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return result.text.strip()
    except Exception as e:
        logger.error(f"❌ [WHISPER] Ошибка распознавания голоса: {e}")
        return ""


# ── [TTS] Генерация речи (для голосовых каналов) ──────────
async def _generate_speech(text: str) -> Optional[bytes]:
    """Генерирует аудио из текста с помощью OpenAI-совместимого TTS API."""
    try:
        loop = asyncio.get_event_loop()
        def _tts():
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            return response.content
        return await loop.run_in_executor(None, _tts)
    except Exception as e:
        logger.error(f"❌ [TTS] Ошибка генерации речи: {e}")
        return None


# ── Попытка поставить реакцию на сообщение ────────────────
async def _try_reaction(message, emoji: str) -> bool:
    """Пытается поставить реакцию в Discord."""
    try:
        await message.add_reaction(emoji)
        return True
    except Exception as e:
        logger.warning(f"⚠️ [REACTION] Не удалось добавить реакцию: {e}")
        return False


# Имитация задержки печати/говорения
async def _typing_delay(text: str, channel) -> None:
    """Имитирует задержку набора ответа (зависит от длины строки)."""
    word_count = len(text.split())
    delay = min(max(word_count * 0.12, 0.5), 2.5)
    await asyncio.sleep(delay)


# ── Импорт Discord с резервным моком (для тестов в песочнице) ──
try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DISCORD_AVAILABLE = False
    # Заглушки классов для обеспечения работы юнит-тестов без установленного discord.py
    class Dummy:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return Dummy
    discord = Dummy()
    discord.Intents = Dummy()
    discord.Intents.default = lambda: Dummy()
    commands = Dummy()
    commands.Bot = Dummy


# ── [Центральный Пайплайн обработки текста] ────────────────
async def _process_text(message, user_text: str, label: str = "") -> None:
    """Центральный pipeline — принимает финальный текст и прогоняет через ThinkGraph."""
    # Определение уникального ID сессии (личка vs серверные каналы)
    if isinstance(message.channel, discord.DMChannel):
        session_id = f"dm_{message.author.id}"
        chat_id = message.author.id
    else:
        session_id = f"channel_{message.channel.id}"
        chat_id = message.channel.id

    # Проверка лимитов (Rate Limiter)
    if not _check_rate_limit(chat_id):
        logger.warning(f"⚠️ [RATE LIMIT] Запрос от {chat_id} проигнорирован (слишком часто)")
        return

    # Загружаем досье юзера
    username = message.author.global_name or message.author.name or "незнакомец"
    await ensure_user(session_id, username)

    persona = await get_user(session_id)
    chat_history = await get_session(session_id)

    # ── [Семантический кэш] ──────────────────────────────
    warmth = persona.get("warmth", 0.0)
    attitude = persona.get("base_attitude", "нейтральное")
    if warmth < 0:
        warmth_tier = "cold"
    elif warmth <= 0.5:
        warmth_tier = "neutral"
    else:
        warmth_tier = "warm"

    cached_response = await get_semantic_cache_async(user_text, attitude, warmth_tier)
    if cached_response:
        async with message.channel.typing():
            await _typing_delay(cached_response, message.channel)
        
        # Отвечаем в текстовом канале
        await message.reply(cached_response)
        logger.info(f"⚡ [SEMANTIC CACHE] Ответ выдан из кэша: {cached_response}\n")
        
        # Если бот находится в голосовом канале на том же сервере — озвучиваем ответ!
        if getattr(message.guild, "voice_client", None):
            asyncio.create_task(_speak_in_voice_channel(message.guild, cached_response))
        
        # Обновляем историю и БД
        display_text = f"[ЮЗЕР ПРИСЛАЛ ГОЛОСОВОЕ: \"{user_text}\"]" if label == "voice" else user_text
        chat_history.append({"role": "user", "content": display_text})
        await save_message(session_id, "user", display_text)
        
        chat_history.append({"role": "assistant", "content": cached_response})
        await save_message(session_id, "assistant", cached_response)
        
        # Фоновое сохранение в векторную память
        asyncio.create_task(save_to_memory_async("user", user_text, user_id=session_id))
        asyncio.create_task(save_to_memory_async("assistant", cached_response, user_id=session_id))
        return

    # Добавляем в историю
    display_text = f"[ЮЗЕР ПРИСЛАЛ ГОЛОСОВОЕ: \"{user_text}\"]" if label == "voice" else user_text
    chat_history.append({"role": "user", "content": display_text})
    await save_message(session_id, "user", display_text)

    # Текущее время
    current_time_str = datetime.now().strftime("%H:%M")
    
    # ── [Семантический роутер (Stage 1.2)] ─────────────────
    route = route_message(user_text)
    logger.info(f"🛣️ [ROUTER] Маршрут сообщения: {route}")

    # Если роут "general" (приветствие, простые фразы) — обходим тяжелый ThinkGraph!
    if route == "general":
        logger.info("⚡ [ROUTER] Быстрый путь: пропуск ThinkGraph.")
        
        # Быстрый блок личности
        persona_block = (
            f"Companion core state: {persona['base_attitude']}. "
            f"Companion warmth tier: {warmth_tier}. "
            f"Companion narrative/memory: {persona['persona_narrative'] or 'Empty'}. "
            f"User name: {username}."
        )

        async with message.channel.typing():
            ai_response = await speech_graph.generate_async(
                signal=None,  # нет мета-сигнала
                user_text=user_text,
                history=chat_history,
                persona_block=persona_block,
                tool_result="",
                warmth=warmth
            )
            
            if not ai_response or len(ai_response) < 2:
                ai_response = "*молчит*"
                
            await _typing_delay(ai_response, message.channel)
            await message.reply(ai_response)
            logger.info(f"💬 [ОТВЕТ (FAST)] {ai_response}\n")

        chat_history.append({"role": "assistant", "content": ai_response})
        await save_message(session_id, "assistant", ai_response)
        
        # Озвучиваем если в голосовом канале
        if getattr(message.guild, "voice_client", None):
            asyncio.create_task(_speak_in_voice_channel(message.guild, ai_response))
            
        asyncio.create_task(save_to_memory_async("user", user_text, user_id=session_id))
        asyncio.create_task(save_to_memory_async("assistant", ai_response, user_id=session_id))
        asyncio.create_task(save_semantic_cache_async(user_text, ai_response, attitude, warmth_tier))
        return

    # ── ЭТАП 1: THINK ENGINE (System 2 Thinking) ───────
    try:
        # Извлекаем контекстные воспоминания из ChromaDB
        memories_summary = ""
        if is_memory_available():
            from memory import recall_memories
            memories_summary = recall_memories(user_text, user_id=session_id, n_results=3)

        recent_think_logs = await get_recent_think_logs(session_id, limit=6)

        state = ThinkState(
            user_text=user_text,
            memories_summary=memories_summary,
            current_user_name=persona["name"],
            user_role="companion",  # стандартная роль
            base_attitude=persona["base_attitude"],
            time_passed_str="1m",    # дефолт
            current_time_str=current_time_str,
            recent_think_logs=recent_think_logs,
        )

        async with message.channel.typing():
            # Запуск размышления
            signal = await think_graph.run_async(state)
            
            logger.info(f"🧠 [THOUGHT] Emotion: {signal.rin_emotion}")
            logger.info(f"🧠 [THOUGHT] Tactic : {signal.rin_tactic}")
            logger.info(f"🧠 [THOUGHT] Attitude: {signal.rin_attitude}")
            logger.info(f"🧠 [THOUGHT] Plan   : {signal.internal_plan}")

            # Влияние на теплоту
            await update_user_warmth(session_id, {
                "слегка тёплое":  +0.2, "нейтральное":  0.0,
                "безразличное":  -0.1, "настороженное": -0.1,
                "раздражённое": -0.2,
            }.get(signal.rin_attitude, 0.0))
            persona = await get_user(session_id)  # перечитываем

            # Если бот решает молчать
            if not signal.should_speak:
                logger.info("🤫 Rin молчит.")
                chat_history.append({"role": "assistant", "content": "*молчит*"})
                await save_message(session_id, "assistant", "*молчит*")
                await message.reply("*молчит*")
                return

            # Выполнение инструмента (навыка) при необходимости
            tool_result = ""
            if signal.needs_tool and signal.tool_name:
                args = dict(signal.tool_args)
                if signal.tool_name in ("search_core_memory", "save_fact_to_memory"):
                    args.setdefault("user_id", session_id)
                tool_result = await execute_tool_async(signal.tool_name, args)

            # ЭТАП 2: SPEECH ENGINE
            persona_block = (
                f"Companion core state: {persona['base_attitude']}. "
                f"Companion warmth tier: {warmth_tier}. "
                f"Companion narrative/memory: {persona['persona_narrative'] or 'Empty'}. "
                f"User name: {username}."
            )
            
            ai_response = await speech_graph.generate_async(
                signal=signal,
                user_text=display_text,
                history=chat_history,
                persona_block=persona_block,
                tool_result=tool_result,
            )

            if not ai_response or len(ai_response) < 2:
                ai_response = "*молчит*"

            await _typing_delay(ai_response, message.channel)
            await message.reply(ai_response)
            logger.info(f"💬 [ОТВЕТ] {ai_response}\n")

        chat_history.append({"role": "assistant", "content": ai_response})
        await save_message(session_id, "assistant", ai_response)
        
        # Озвучивание в голосовом канале
        if getattr(message.guild, "voice_client", None):
            asyncio.create_task(_speak_in_voice_channel(message.guild, ai_response))
        
        # Сохранение в векторную память
        asyncio.create_task(
            save_to_memory_async("assistant", ai_response, user_id=session_id,
                                  extra_meta={"emotion": signal.rin_emotion})
        )
        
        # Сохранение в семантический кэш
        final_warmth = persona.get("warmth", 0.0)
        final_attitude = persona.get("base_attitude", "нейтральное")
        if final_warmth < 0:
            final_warmth_tier = "cold"
        elif final_warmth <= 0.5:
            final_warmth_tier = "neutral"
        else:
            final_warmth_tier = "warm"
        asyncio.create_task(save_semantic_cache_async(user_text, ai_response, final_attitude, final_warmth_tier))

        # Фоновая саммаризация при превышении лимита токенов
        asyncio.create_task(_run_summarization(chat_id, session_id))

    except Exception as e:
        logger.error(f"❌ [PIPELINE] Ошибка: {e}", exc_info=True)


async def _run_summarization(chat_id: int, session_id: str) -> None:
    """Фоновая саммаризация по токенам (V10)."""
    try:
        history = await get_session(session_id)
        # maybe_summarize_history проверяет лимиты внутри
        new_history, suggestion = await maybe_summarize_history(
            chat_history=history,
            client=client,
            model=MODEL,
            user_id=session_id
        )
        
        if new_history is not history:
            if suggestion:
                new_history.append({
                    "role": "system", 
                    "content": f"[АРХИВНОЕ НАБЛЮДЕНИЕ: {suggestion}. Ты можешь обновить Core Memory если считаешь нужным.]"
                })
            
            set_session(session_id, new_history)
            await update_history_in_db(session_id, new_history)
            logger.info(f"✅ [MEMORY] Саммаризация сессии {session_id} завершена.")
    except Exception as e:
        logger.error(f"❌ [MEMORY_TASK] Ошибка в фоновой задаче саммаризации: {e}", exc_info=True)


# ── [TTS] Воспроизведение речи в голосовом канале ──────────
async def _speak_in_voice_channel(guild, text: str):
    """Генерирует речь и воспроизводит ее в голосовом канале сервера."""
    if not DISCORD_AVAILABLE or not guild.voice_client:
        return
    
    audio_bytes = await _generate_speech(text)
    if not audio_bytes:
        return
        
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        vc = guild.voice_client
        if vc.is_playing():
            vc.stop()
            
        # Воспроизведение MP3 файла с помощью FFmpeg
        vc.play(
            discord.FFmpegPCMAudio(tmp_path), 
            after=lambda e: os.unlink(tmp_path)
        )
    except Exception as e:
        logger.error(f"❌ [VOICE_PLAY] Ошибка воспроизведения: {e}")
        try:
            os.unlink(tmp_path)
        except:
            pass


# ── Проверка соединения с ИИ-бэкендом ──────────────────────
async def _check_ai_connection():
    """Тестирует связь с OpenAI-совместимым API LM Studio."""
    logger.info(f"🧠 [AI CONNECTION] Проверка связи с AI бэкендом: {AI_BACKEND_URL} ...")
    try:
        loop = asyncio.get_event_loop()
        def _call():
            return client.models.list()
        
        models = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=5.0)
        available_models = [m.id for m in models.data]
        logger.info(f"✅ [AI CONNECTION] Подключение успешно! Доступные модели: {', '.join(available_models)}")
        if MODEL in available_models:
            logger.info(f"🎯 [AI CONNECTION] Заданная модель '{MODEL}' полностью доступна!")
        else:
            logger.warning(f"⚠️ [AI CONNECTION] Заданная модель '{MODEL}' отсутствует в списке. Бэкенд будет использовать модель по умолчанию.")
        return True
    except asyncio.TimeoutError:
        logger.error(f"❌ [AI CONNECTION] Превышено время ожидания ответа от бэкенда {AI_BACKEND_URL} (таймаут 5 сек)!")
        return False
    except Exception as e:
        logger.error(f"❌ [AI CONNECTION] Ошибка соединения с AI бэкендом {AI_BACKEND_URL}: {e}")
        return False


# ── Настройка Discord Бота (если библиотека доступна) ──────
if DISCORD_AVAILABLE:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.guilds = True
    intents.members = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        logger.info(f"🌸 Бот {bot.user} успешно авторизован в Discord!")
        logger.info(f"🆔 Discord ID бота: {bot.user.id}")
        logger.info(f"🏡 Серверы бота ({len(bot.guilds)}): {', '.join([g.name for g in bot.guilds]) if bot.guilds else 'только личные сообщения (DM)'}")
        
        try:
            await bot.change_presence(status=discord.Status.dnd)
            logger.info("🌙 Статус присутствия успешно изменен на 'Не беспокоить' (DND)!")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сменить статус на DND: {e}")

        # Фоновая проверка связи с ИИ при запуске
        asyncio.create_task(_check_ai_connection())

    # ── Команды ───────────────────────────────────────────
    @bot.command(name="start")
    async def cmd_start(ctx):
        session_id = f"dm_{ctx.author.id}" if isinstance(ctx.channel, discord.DMChannel) else f"channel_{ctx.channel.id}"
        set_session(session_id, [{"role": "system", "content": IDENTITY_PROMPT}])
        await ensure_user(session_id, ctx.author.global_name or ctx.author.name)
        await ctx.reply("...")

    @bot.command(name="reset")
    async def cmd_reset(ctx):
        session_id = f"dm_{ctx.author.id}" if isinstance(ctx.channel, discord.DMChannel) else f"channel_{ctx.channel.id}"
        set_session(session_id, [{"role": "system", "content": IDENTITY_PROMPT}])
        await ctx.reply("—")

    @bot.command(name="whoami")
    async def cmd_whoami(ctx):
        session_id = f"dm_{ctx.author.id}" if isinstance(ctx.channel, discord.DMChannel) else f"channel_{ctx.channel.id}"
        p = await get_user(session_id)
        await ctx.reply(
            f"имя={p['name']} | отношение={p['base_attitude']} | warmth={p['warmth']:.2f}"
        )

    @bot.command(name="join")
    async def cmd_join(ctx):
        """Подключается к голосовому каналу пользователя."""
        if not ctx.author.voice:
            await ctx.reply("Ты должен находиться в голосовом канале!")
            return
        
        channel = ctx.author.voice.channel
        if ctx.voice_client:
            await ctx.voice_client.move_to(channel)
        else:
            await channel.connect()
        await ctx.reply("👋 Привет в голосовом канале.")

    @bot.command(name="leave")
    async def cmd_leave(ctx):
        """Отключается от голосового канала."""
        if ctx.voice_client:
            await ctx.voice_client.disconnect()
            await ctx.reply("🚪 Вышла.")
        else:
            await ctx.reply("Я не нахожусь в голосовом канале.")

    # ── Главный обработчик входящих событий ───────────────
    @bot.event
    async def on_message(message: discord.Message):
        # Логируем для отладки
        logger.info(f"📨 [MESSAGE] Получено сообщение от {message.author}: content='{message.content}', attachments={len(message.attachments)}")

        # Пропускаем сообщения самого бота
        if message.author == bot.user:
            return

        # Позволяем командам работать
        await bot.process_commands(message)

        # Если это команда (начинается с приставки) — пропускаем дальнейший анализ текста
        if message.content.startswith("!"):
            return

        user_text = message.content or ""
        
        # ── [4.1] Обработка аудио вложений как голосовых сообщений ──
        voice_data = None
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.ogg', '.mp3', '.wav', '.m4a', '.mp4', '.3gp')):
                async with message.channel.typing():
                    # Скачиваем файл в ОЗУ
                    audio_bytes = await attachment.read()
                    voice_data = audio_bytes
                    logger.info(f"🎙️ [VOICE] Обнаружен аудиофайл: {attachment.filename}")
                    break

        if voice_data:
            transcribed = await _transcribe_voice(voice_data)
            if not transcribed:
                transcribed = "[не удалось расшифровать голосовое]"
            await _process_text(message, transcribed, label="voice")
        elif user_text.strip():
            await _process_text(message, user_text)


# ── [3.1–3.3] IdleGraph — фоновый цикл «снов» ───────────
IDLE_CHECK_INTERVAL = 4 * 3600  # каждые 4 часа
IDLE_INACTIVITY_MIN = 2 * 3600  # чат должен молчать хотя бы 2 часа

async def _idle_loop() -> None:
    """Фоновый цикл: анализ снов + очистка ОЗУ (Memory Cleanup)."""
    await asyncio.sleep(IDLE_CHECK_INTERVAL)
    while True:
        for session_id, _ in list(active_sessions.items()):
            last_msg = await get_last_message_time(session_id)
            if last_msg is None:
                continue
            
            delta = datetime.now() - last_msg
            idle_sec = delta.total_seconds()
            
            # Устранение утечки памяти: выгрузка старых сессий из ОЗУ (> 24ч)
            if idle_sec > 86400:
                if session_id in active_sessions:
                    del active_sessions[session_id]
                    logger.info(f"♻️ [MEMORY_CLEANUP] Сессия {session_id} выгружена из ОЗУ.")
                continue

            if idle_sec < IDLE_INACTIVITY_MIN:
                continue

            persona    = await get_user(session_id)
            think_logs = await get_recent_think_logs(session_id, limit=50)
            chat_history = await load_history(session_id, system_prompt="", limit=30)

            logger.info(f"🌙 [IDLE] Запускаем IdleGraph для {session_id} ({persona['name']})")

            result = await idle_graph.run_async(
                user_name=persona["name"],
                current_attitude=persona["base_attitude"],
                think_logs=think_logs,
                warmth=persona["warmth"],
                chat_history=chat_history,
            )
            logger.info(f"🌙 [IDLE] Результат: {result['reasoning']}")

            if result["attitude"] != persona["base_attitude"]:
                await update_user_attitude(session_id, result["attitude"])

            if result.get("narrative"):
                await update_persona_narrative(session_id, result["narrative"])

            # Осознанная инициатива (Бот пишет первым в канал/личку)
            initiative = result.get("initiative_text")
            if initiative and DISCORD_AVAILABLE:
                try:
                    # Извлекаем ID получателя/канала из session_id
                    target_id = int(session_id.split("_")[1])
                    if session_id.startswith("dm_"):
                        user = await bot.fetch_user(target_id)
                        await user.send(initiative)
                    else:
                        channel = await bot.fetch_channel(target_id)
                        await channel.send(initiative)
                        
                    logger.info(f"📤 [INITIATIVE] → {session_id}: «{initiative}»")
                    session = await get_session(session_id)
                    session.append({"role": "assistant", "content": initiative})
                    await save_message(session_id, "assistant", initiative)
                except Exception as e:
                    logger.warning(f"⚠️  [INITIATIVE] {e}")

        await asyncio.sleep(IDLE_CHECK_INTERVAL)


# ── Главная точка входа ───────────────────────────────────
async def main():
    init_db()
    init_speech_engine(TOKENIZER_MODEL)
    init_router()

    print("━" * 58)
    print("  🌸  Rin V10 Discord Bot — запущена")
    print("━" * 58)
    print(f"  🧠  Think Engine  : V10 (Pydantic + Native Tools)")
    print(f"  🧩  Векторная память: {'✅ ChromaDB V10' if is_memory_available() else '⚠️  недоступна'}")
    print(f"  💾  SQLite        : ✅")
    print(f"  🤖  Модель        : {MODEL}")
    print("━" * 58 + "\n")

    # Запускаем фоновый цикл снов
    asyncio.create_task(_idle_loop())
    
    if DISCORD_AVAILABLE:
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            logger.critical("❌ [CRITICAL] DISCORD_BOT_TOKEN is not set in environment or .env file!")
            return
        await bot.start(token)
    else:
        logger.warning("⚠️  [START] Бот запущен в режиме тестирования (discord.py недоступен в песочнице).")
        # Держим процесс запущенным для тестов
        while True:
            await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Завершение работы бота.")