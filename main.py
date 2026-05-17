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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("main")

# AI and Tokenizer settings
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://127.0.0.1:1234/v1")
AI_API_KEY     = os.getenv("AI_API_KEY", "lm-studio")
MODEL          = os.getenv("AI_MODEL", "rin")  # fine-tuned model
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "NousResearch/Hermes-3-Llama-3.1-8B")

# Initialize OpenAI client
client = OpenAI(base_url=AI_BACKEND_URL, api_key=AI_API_KEY)

# Core modules imports
from database import (
    init_db, ensure_user, get_user, update_user_warmth,
    update_user_attitude, update_core_memory, update_persona_narrative,
    save_message, load_history, touch_message_time, get_last_message_time,
    append_dashboard_log
)
from think_engine import ThinkGraph, IdleGraph, ThinkSignal, _build_persona_block
from memory import save_to_memory_async, recall_memories_async, summarize_if_needed
from speech_engine import (
    init_speech_engine, build_generation_logit_bias, _clean_output, TACTIC_LENGTH
)
from skills import execute_tool
from semantic_cache import get_semantic_cache_async, save_semantic_cache_async
from semantic_router_engine import route_message_async

# AI Engine setup
think_graph  = ThinkGraph(client=client, model=MODEL)
idle_graph   = IdleGraph(client=client, model=MODEL)

# RAM Leak Prevention: in-memory user cache
active_sessions = {}
session_locks = {}

# ── [Rate Limiter] ───────────────────────────────────────
_user_message_times: dict[int, list[datetime]] = {}
RATE_LIMIT_SECONDS = 3.0

def _check_rate_limit(chat_id: int) -> bool:
    """Limits the request frequency (no more than 1 request per 3 seconds per channel/DM)."""
    now = datetime.now()
    if chat_id in _user_message_times:
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
    """Transcribes audio file using Whisper STT via OpenAI-compatible endpoint."""
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
                    language="en",
                )
        
        result = await loop.run_in_executor(None, _whisper)
        
        try:
            os.unlink(tmp_path)
        except:
            pass
            
        return result.text.strip()
    except Exception as e:
        logger.error(f"❌ [WHISPER] Failed to transcribe voice: {e}")
        return ""


# ── [TTS] Speech Generation (for voice channels) ──────────
async def _generate_speech(text: str) -> Optional[bytes]:
    """Generates speech audio bytes using OpenAI-compatible TTS API."""
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
        logger.error(f"❌ [TTS] Speech generation failed: {e}")
        return None


# ── Try Reaction on Message ──────────────────────────────
async def _try_reaction(message, emoji: str) -> bool:
    try:
        await message.add_reaction(emoji)
        return True
    except Exception as e:
        logger.warning(f"⚠️ [REACTION] Could not add reaction: {e}")
        return False


# Simulated typing delay
async def _typing_delay(text: str, channel) -> None:
    word_count = len(text.split())
    delay = min(max(word_count * 0.12, 0.5), 2.5)
    await asyncio.sleep(delay)


# ── Discord Import with Sandbox Safe Fallback ───────────
try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DISCORD_AVAILABLE = False
    class Dummy:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return Dummy
    discord = Dummy()
    discord.Intents = Dummy()
    discord.Intents.default = lambda: Dummy()
    commands = Dummy()
    commands.Bot = Dummy


# ── [Central Text Pipeline] ──────────────────────────────
async def _process_text(message, user_text: str, label: str = "") -> None:
    """Core text processing pipeline."""
    if isinstance(message.channel, discord.DMChannel):
        session_id = f"dm_{message.author.id}"
        chat_id = message.author.id
    else:
        session_id = f"channel_{message.channel.id}"
        chat_id = message.channel.id

    if not _check_rate_limit(chat_id):
        logger.warning(f"⚠️ [RATE LIMIT] Rate limit triggered for {chat_id}")
        return

    # Ensure user exists in database
    username = message.author.global_name or message.author.name or "stranger"
    await ensure_user(session_id, username)
    await touch_message_time(session_id)

    # Load session and lock
    if session_id not in active_sessions:
        db_user = await get_user(session_id)
        history = await load_history(session_id, system_prompt="", limit=20)
        active_sessions[session_id] = {
            "history": history,
            "warmth": db_user["warmth"],
            "base_attitude": db_user["base_attitude"],
            "core_memory": db_user["core_memory"],
            "persona_narrative": db_user["persona_narrative"],
            "last_active_time": datetime.now()
        }
    
    session = active_sessions[session_id]
    session["last_active_time"] = datetime.now()

    if session_id not in session_locks:
        session_locks[session_id] = asyncio.Lock()

    async with session_locks[session_id]:
        # ── [Semantic Cache Check] ──────────────────────────
        warmth = session["warmth"]
        attitude = session["base_attitude"]
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
            
            await message.reply(cached_response)
            logger.info(f"⚡ [SEMANTIC CACHE] Response served from cache: {cached_response}")
            
            # Speak in voice channel if connected
            if getattr(message.guild, "voice_client", None):
                asyncio.create_task(_speak_in_voice_channel(message.guild, cached_response))
            
            # Save to database and history cache
            await save_message(session_id, "user", user_text)
            session["history"].append({"role": "user", "content": user_text})
            
            await save_message(session_id, "assistant", cached_response)
            session["history"].append({"role": "assistant", "content": cached_response})
            
            asyncio.create_task(save_to_memory_async("user", user_text, session_id))
            asyncio.create_task(save_to_memory_async("assistant", cached_response, session_id))
            return

        # ── [Semantic Routing] ──────────────────────────────
        route = await route_message_async(user_text)
        logger.info(f"🛣️ [ROUTER] Message classified into: {route}")

        if route == "general":
            logger.info("⚡ [ROUTER] Fast-path routing triggered. Bypassing ThinkGraph.")
            
            # Inject fast-path user text
            session["history"].append({"role": "user", "content": user_text})
            await save_message(session_id, "user", user_text)
            
            persona_block = _build_persona_block(
                warmth=session["warmth"],
                base_attitude=session["base_attitude"],
                user_name=username,
                core_memory=session["core_memory"],
                persona_narrative=session["persona_narrative"]
            )
            
            async with message.channel.typing():
                from speech_engine import SpeechGraph
                speech_graph = SpeechGraph(client=client, model=MODEL)
                response_text = await speech_graph.generate_async(
                    signal=None,
                    user_text=user_text,
                    history=session["history"],
                    persona_block=persona_block,
                    tool_result="",
                    warmth=session["warmth"]
                )
                
                if not response_text or len(response_text) < 2:
                    response_text = "*remains silent*"
                    
                await _typing_delay(response_text, message.channel)
                await message.reply(response_text)
                logger.info(f"💬 [RESPONSE (FAST)] {response_text}\n")

            await save_message(session_id, "assistant", response_text)
            session["history"].append({"role": "assistant", "content": response_text})
            
            if getattr(message.guild, "voice_client", None):
                asyncio.create_task(_speak_in_voice_channel(message.guild, response_text))
                
            asyncio.create_task(save_to_memory_async("user", user_text, session_id))
            asyncio.create_task(save_to_memory_async("assistant", response_text, session_id))
            asyncio.create_task(save_semantic_cache_async(user_text, response_text, attitude, warmth_tier))
            return

        # ── 1. Context Injection & Log Retrieval ─────────────
        memories_summary = await recall_memories_async(user_text, session_id, n_results=3)
        recent_think_logs = await get_recent_think_logs(session_id, limit=6)

        # ── 2. Think Graph System 2 Reasoning ────────────────
        state = ThinkState(
            user_text=user_text,
            memories_summary=memories_summary,
            current_user_name=username,
            user_role="stranger",  # default
            base_attitude=session["base_attitude"],
            time_passed_str="1m",
            current_time_str=datetime.now().strftime("%H:%M"),
            recent_think_logs=recent_think_logs
        )

        async with message.channel.typing():
            output: ThinkSignal = await think_graph.run_async(state)
            
            logger.info(f"🧠 [THOUGHT] Emotion: {output.emotion_id}")
            logger.info(f"🧠 [THOUGHT] Tactic : {output.tactic_id}")
            logger.info(f"🧠 [THOUGHT] Attitude: {output.attitude_id}")

            # Append user message turn
            session["history"].append({"role": "user", "content": user_text})
            await save_message(session_id, "user", user_text)

            # Speak check
            if not output.should_speak:
                logger.info("🤫 Rin decided to remain silent.")
                await save_message(session_id, "assistant", "*remains silent*")
                session["history"].append({"role": "assistant", "content": "*remains silent*"})
                await message.reply("*remains silent*")
                return

            # Emoji reaction chance
            if len(user_text.strip()) < 4 and random.random() < 0.25:
                emoji = "😐"
                if await _try_reaction(message, emoji):
                    session["history"].append({"role": "assistant", "content": f"[reaction: {emoji}]"})
                    await save_message(session_id, "assistant", f"[reaction: {emoji}]")
                    return

            # Tool Execution
            tool_result_str = ""
            if output.needs_tool and output.tool_name:
                tool_args = dict(output.tool_args)
                if output.tool_name in ("search_core_memory", "save_fact_to_memory"):
                    tool_args.setdefault("user_id", session_id)
                tool_result_str = await execute_tool(output.tool_name, tool_args)

            # ── 7. Speech Engine ───────────────────────────
            persona_block = (
                f"Companion core state: {session['base_attitude']}. "
                f"Companion warmth tier: {warmth_tier}. "
                f"Companion narrative/memory: {session['persona_narrative'] or 'Empty'}. "
                f"User name: {username}."
            )

            from speech_engine import SpeechGraph
            speech_graph = SpeechGraph(client=client, model=MODEL)
            response_text = await speech_graph.generate_async(
                signal=output,
                user_text=user_text,
                history=session["history"],
                persona_block=persona_block,
                tool_result=tool_result_str,
            )

            if not response_text or len(response_text) < 2:
                response_text = "*remains silent*"

            await _typing_delay(response_text, message.channel)
            await message.reply(response_text)
            logger.info(f"💬 [RESPONSE] {response_text}\n")

        # Save turns
        await save_message(session_id, "assistant", response_text)
        session["history"].append({"role": "assistant", "content": response_text})
        
        # Play in voice channel
        if getattr(message.guild, "voice_client", None):
            asyncio.create_task(_speak_in_voice_channel(message.guild, response_text))

        asyncio.create_task(save_to_memory_async("assistant", response_text, session_id))

        # Save to semantic cache
        final_warmth = session["warmth"]
        final_attitude = session["base_attitude"]
        if final_warmth < 0:
            final_warmth_tier = "cold"
        elif final_warmth <= 0.5:
            final_warmth_tier = "neutral"
        else:
            final_warmth_tier = "warm"
        asyncio.create_task(save_semantic_cache_async(user_text, response_text, final_attitude, final_warmth_tier))

        # ── 9. Warmth & Relationship Adjustments ─────────────
        warmth_delta = 0.0
        if output.emotion_id in ["tired tenderness", "quiet warmth"]:
            warmth_delta = +0.15
        elif output.emotion_id in ["guarded coldness", "silent irritation"]:
            warmth_delta = -0.15

        if warmth_delta != 0.0:
            new_warmth = await update_user_warmth(session_id, warmth_delta)
            session["warmth"] = new_warmth
            logger.info(f"📈 [WARMTH] Updated warmth for {session_id}: {new_warmth:+.2f}")

        # ── 10. Memory Summarization & Pruning ────────────────
        asyncio.create_task(_run_summarization(session_id))


async def _run_summarization(session_id: str) -> None:
    """Asynchronous background memory summarization trigger."""
    try:
        session = active_sessions[session_id]
        pruned_history, summary_result = await summarize_if_needed(
            history=session["history"],
            user_id=session_id,
            client=client,
            model=MODEL
        )
        
        if pruned_history is not session["history"]:
            session["history"] = pruned_history
            await update_history_in_db(session_id, pruned_history)
            
            if summary_result:
                if getattr(summary_result, "core_memory_update", None):
                    await update_core_memory(session_id, summary_result.core_memory_update)
                    session["core_memory"] = summary_result.core_memory_update
                    
                await append_dashboard_log(
                    session_id,
                    f"Summarization completed. Core memory updated: {bool(summary_result.core_memory_update)}"
                )
    except Exception as e:
        logger.error(f"❌ [SUMMARY] In background execution: {e}")


# ── [TTS] Voice Playing ───────────────────────────────
async def _speak_in_voice_channel(guild, text: str):
    """Speaks output text inside active guild voice channel."""
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
            
        vc.play(
            discord.FFmpegPCMAudio(tmp_path), 
            after=lambda e: os.unlink(tmp_path)
        )
    except Exception as e:
        logger.error(f"❌ [VOICE_PLAY] Failed to play speech audio: {e}")
        try:
            os.unlink(tmp_path)
        except:
            pass


# ── AI Connection Check ───────────────────────────────
async def _check_ai_connection():
    """Tests the connection to OpenAI-compatible AI backend."""
    logger.info(f"🧠 [AI CONNECTION] Testing connection to AI backend: {AI_BACKEND_URL} ...")
    try:
        loop = asyncio.get_event_loop()
        def _call():
            return client.models.list()
        
        models = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=5.0)
        available_models = [m.id for m in models.data]
        logger.info(f"✅ [AI CONNECTION] Connection successful! Available models: {', '.join(available_models)}")
        if MODEL in available_models:
            logger.info(f"🎯 [AI CONNECTION] Target model '{MODEL}' is available!")
        else:
            logger.warning(f"⚠️ [AI CONNECTION] Target model '{MODEL}' was not found. Backend default model will be used.")
        return True
    except asyncio.TimeoutError:
        logger.error(f"❌ [AI CONNECTION] Connection timed out for {AI_BACKEND_URL} (5s timeout)!")
        return False
    except Exception as e:
        logger.error(f"❌ [AI CONNECTION] Failed to connect to AI backend {AI_BACKEND_URL}: {e}")
        return False


# ── Discord Bot Setup ─────────────────────────────────
if DISCORD_AVAILABLE:
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.guilds = True
    intents.members = True

    bot = commands.Bot(command_prefix="!", intents=intents)

    @bot.event
    async def on_ready():
        logger.info(f"🌸 Bot {bot.user} successfully connected to Discord!")
        logger.info(f"🆔 Bot Discord ID: {bot.user.id}")
        logger.info(f"🏡 Bot active guilds ({len(bot.guilds)}): {', '.join([g.name for g in bot.guilds]) if bot.guilds else 'direct messages only (DM)'}")
        
        try:
            await bot.change_presence(status=discord.Status.dnd)
            logger.info("🌙 Bot status presence successfully set to 'Do Not Disturb' (DND)!")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set status to DND: {e}")

        # Async background AI connection test on ready
        asyncio.create_task(_check_ai_connection())

    @bot.command(name="start")
    async def cmd_start(ctx):
        session_id = f"dm_{ctx.author.id}" if isinstance(ctx.channel, discord.DMChannel) else f"channel_{ctx.channel.id}"
        await ensure_user(session_id, ctx.author.global_name or ctx.author.name)
        await ctx.reply("...")

    @bot.command(name="reset")
    async def cmd_reset(ctx):
        session_id = f"dm_{ctx.author.id}" if isinstance(ctx.channel, discord.DMChannel) else f"channel_{ctx.channel.id}"
        if session_id in active_sessions:
            del active_sessions[session_id]
        await ctx.reply("—")

    @bot.command(name="whoami")
    async def cmd_whoami(ctx):
        session_id = f"dm_{ctx.author.id}" if isinstance(ctx.channel, discord.DMChannel) else f"channel_{ctx.channel.id}"
        p = await get_user(session_id)
        await ctx.reply(
            f"name={p['name']} | attitude={p['base_attitude']} | warmth={p['warmth']:.2f}"
        )

    @bot.command(name="join")
    async def cmd_join(ctx):
        """Joins user's current voice channel."""
        if not ctx.author.voice:
            await ctx.reply("You must be in a voice channel!")
            return
        
        channel = ctx.author.voice.channel
        if ctx.voice_client:
            await ctx.voice_client.move_to(channel)
        else:
            await channel.connect()
        await ctx.reply("👋 Connected to your voice channel.")

    @bot.command(name="leave")
    async def cmd_leave(ctx):
        """Disconnects from active voice channel."""
        if ctx.voice_client:
            await ctx.voice_client.disconnect()
            await ctx.reply("🚪 Left voice channel.")
        else:
            await ctx.reply("I am not in a voice channel.")

    @bot.event
    async def on_message(message: discord.Message):
        # Log for debugging
        logger.info(f"📨 [MESSAGE] Received message from {message.author}: content='{message.content}', attachments={len(message.attachments)}")

        if message.author == bot.user:
            return

        await bot.process_commands(message)

        if message.content.startswith("!"):
            return

        user_text = message.content or ""
        
        # Audio attachments check (voice messages)
        voice_data = None
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.ogg', '.mp3', '.wav', '.m4a', '.mp4', '.3gp')):
                async with message.channel.typing():
                    audio_bytes = await attachment.read()
                    voice_data = audio_bytes
                    logger.info(f"🎙️ [VOICE] Detected audio attachment: {attachment.filename}")
                    break

        if voice_data:
            transcribed = await _transcribe_voice(voice_data)
            if not transcribed:
                transcribed = "[failed to transcribe voice]"
            await _process_text(message, transcribed, label="voice")
        elif user_text.strip():
            await _process_text(message, user_text)


# ── IdleGraph Sleep Loop & RAM Pruning ─────────────────
async def _idle_loop():
    """Runs a periodic loop to clean up RAM memory leaks + run IdleGraph dreams."""
    while True:
        try:
            await asyncio.sleep(3600)  # check every hour
            now = datetime.now()
            expired_keys = []
            
            # RAM Pruning
            for key, session in list(active_sessions.items()):
                last_active = session.get("last_active_time", now)
                if now - last_active > timedelta(hours=24):
                    expired_keys.append(key)
                    
            for key in expired_keys:
                if key in active_sessions:
                    del active_sessions[key]
                    logger.info(f"♻️ [MEMORY_CLEANUP] Evicted inactive session {key} from RAM cache.")

            # IdleGraph analysis cycle (every 4 hours of inactivity)
            for session_id, session in list(active_sessions.items()):
                last_msg = await get_last_message_time(session_id)
                if last_msg is None:
                    continue
                
                delta = now - last_msg
                if delta.total_seconds() > 14400:  # 4 hours
                    logger.info(f"🌙 [IDLE] Activating IdleGraph dreams for {session_id}")
                    
                    persona = await get_user(session_id)
                    think_logs = await get_recent_think_logs(session_id, limit=50)
                    chat_history = await load_history(session_id, system_prompt="", limit=30)

                    result = await idle_graph.run_async(
                        user_name=persona["name"],
                        current_attitude=persona["base_attitude"],
                        think_logs=think_logs,
                        warmth=persona["warmth"],
                        chat_history=chat_history
                    )

                    if result.get("attitude") and result["attitude"] != persona["base_attitude"]:
                        await update_user_attitude(session_id, result["attitude"])
                        
                    if result.get("narrative"):
                        await update_persona_narrative(session_id, result["narrative"])

                    # Direct proactive output
                    initiative = result.get("initiative_text")
                    if initiative and DISCORD_AVAILABLE:
                        try:
                            target_id = int(session_id.split("_")[1])
                            if session_id.startswith("dm_"):
                                user = await bot.fetch_user(target_id)
                                await user.send(initiative)
                            else:
                                channel = await bot.fetch_channel(target_id)
                                await channel.send(initiative)
                            
                            logger.info(f"📤 [INITIATIVE] → {session_id}: «{initiative}»")
                            session["history"].append({"role": "assistant", "content": initiative})
                            await save_message(session_id, "assistant", initiative)
                        except Exception as e:
                            logger.warning(f"⚠️ [INITIATIVE] Proactive send failed: {e}")

        except Exception as e:
            logger.error(f"❌ [IDLE_LOOP] In background dream cycle: {e}")


# ── Main Entrypoint ────────────────────────────────────
async def main():
    init_db()
    init_speech_engine(TOKENIZER_MODEL)

    print("━" * 58)
    print("  🌸  Rin V10 Discord Bot (English) — Online")
    print("━" * 58)
    print(f"  🧠  Think Engine  : V10 (Pydantic + Native Tools)")
    print(f"  🧩  Vector Memory : {'✅ ChromaDB V10' if is_memory_available() else '⚠️  Unavailable'}")
    print(f"  💾  SQLite        : ✅")
    print(f"  🤖  Model         : {MODEL}")
    print("━" * 58 + "\n")

    asyncio.create_task(_idle_loop())
    
    if DISCORD_AVAILABLE:
        token = os.getenv("DISCORD_BOT_TOKEN")
        if not token:
            logger.critical("❌ [CRITICAL] DISCORD_BOT_TOKEN is missing!")
            return
        await bot.start(token)
    else:
        logger.warning("⚠️  [START] Running in simulation mode (discord.py is offline).")
        while True:
            await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutting down Rin.")