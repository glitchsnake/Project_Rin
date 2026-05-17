"""
main.py — Telegram bot Rin (V10.3)

Key Features:
  - 100% Async / Non-blocking architecture.
  - Periodic RAM leak protection (removes active sessions inactive for >24h).
  - Dynamic speech compilation vocabulary steering using HuggingFace tokenizers.
  - Non-blocking SQLite persistence queries via aiosqlite and async vector memory checks.
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

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from openai import OpenAI

# Core modules imports
from database import (
    init_db,
    ensure_user,
    get_user,
    update_user_warmth,
    update_user_attitude,
    update_core_memory,
    update_persona_narrative,
    save_message,
    load_history,
    touch_message_time,
    get_last_message_time,
    append_dashboard_log
)
from think_engine import ThinkGraph, IdleGraph, ThinkSignal, _build_persona_block
from memory import save_to_memory_async, recall_memories_async, summarize_if_needed
from speech_engine import (
    init_speech_engine,
    build_generation_logit_bias,
    _clean_output,
    TACTIC_LENGTH
)
from skills import execute_tool

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# ── Configurations ────────────────────────────────────────
API_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://127.0.0.1:1234/v1")
AI_API_KEY     = os.getenv("AI_API_KEY", "lm-studio")
MODEL          = os.getenv("AI_MODEL", "rin")  # fine-tuned model name
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "NousResearch/Hermes-3-Llama-3.1-8B")

if not API_TOKEN:
    logger.critical("❌ [CRITICAL] TELEGRAM_BOT_TOKEN is not set in environment or .env file!")
    raise ValueError("TELEGRAM_BOT_TOKEN is missing!")

bot         = Bot(token=API_TOKEN)
dp          = Dispatcher()
client      = OpenAI(base_url=AI_BACKEND_URL, api_key=AI_API_KEY)
think_graph  = ThinkGraph(client=client, model=MODEL)
idle_graph   = IdleGraph(client=client, model=MODEL)

# RAM Leak Prevention: in-memory user cache
active_sessions = {}
session_locks = {}

# ════════════════════════════════════════════════════════
#  RAM Protection: Periodic Session Pruning
# ════════════════════════════════════════════════════════

async def _session_cleaner_loop():
    """Runs a periodic loop to clean up active_sessions to prevent RAM memory leaks."""
    while True:
        try:
            await asyncio.sleep(3600)  # check every hour
            now = datetime.now()
            expired_keys = []
            
            for key, session in list(active_sessions.items()):
                last_active = session.get("last_active_time", now)
                # Keep active sessions in RAM for max 24 hours of inactivity
                if now - last_active > timedelta(hours=24):
                    expired_keys.append(key)
                    
            for key in expired_keys:
                if key in active_sessions:
                    del active_sessions[key]
                    logger.info(f"🧹 [RAM CLEANER] Session {key} purged due to 24h inactivity.")
        except Exception as e:
            logger.error(f"❌ [RAM CLEANER] Error in cleaner loop: {e}")

# ════════════════════════════════════════════════════════
#  Speech Graph (Async V10)
# ════════════════════════════════════════════════════════

class SpeechGraph:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def run(self, prompt: str, logit_bias: dict, max_tokens: int) -> str:
        """Runs the OpenAI text completion using dynamic logit steering constraints."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
                logit_bias=logit_bias
            )
            raw = completion.choices[0].message.content
            return _clean_output(raw)
        except Exception as e:
            logger.error(f"❌ [SPEECH] Speech generation failure: {e}")
            return "..."

    async def run_async(self, *args, **kwargs) -> str:
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.run(*args, **kwargs)
        )

speech_graph = SpeechGraph(client=client, model=MODEL)

# ════════════════════════════════════════════════════════
#  Background Task Runners (Safe Error Handling)
# ════════════════════════════════════════════════════════

async def _safe_run_summarization(history: list[dict], user_id: str):
    """Executes long-term memory summarization in a safe background task."""
    try:
        updated_history, summary_output = await summarize_if_needed(history, user_id, client, MODEL)
        
        # Inject core memory suggestions back into the active sessions cycle
        if summary_output and summary_output.summary:
            if user_id in active_sessions:
                active_sessions[user_id]["history"] = updated_history
                
                # Signal to the agent to dynamically update its Core Memory profile
                if summary_output.core_memory_update:
                    observation_msg = (
                        f"[ARCHIVE OBSERVATION]: Dialogue analysis revealed new facts. "
                        f"Suggested Core Memory update: '{summary_output.core_memory_update}'. "
                        f"You can call update_core_memory tool to save this if it is important."
                    )
                    active_sessions[user_id]["history"].append({"role": "system", "content": observation_msg})
                    logger.info(f"💡 [MEMORY] Suggestion injected for user {user_id}: {summary_output.core_memory_update}")
    except Exception as e:
         logger.error(f"❌ [BACKGROUND TASK] Summarization failed: {e}", exc_info=True)


async def _safe_save_to_memory(role: str, content: str, user_id: str):
    """Saves conversation turns to vector store in a background task."""
    try:
        await save_to_memory_async(role, content, user_id)
    except Exception as e:
         logger.error(f"❌ [BACKGROUND TASK] Save memory failed: {e}", exc_info=True)

# ════════════════════════════════════════════════════════
#  Voice Processing (STT Async ThreadPoolExecutor)
# ════════════════════════════════════════════════════════

async def _transcribe_voice(file_path: str) -> Optional[str]:
    """Sends voice message file to OpenAI whisper API asynchronously."""
    def worker():
        try:
            with open(file_path, "rb") as audio:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio
                )
                return transcript.text
        except Exception as e:
            logger.error(f"❌ [STT] Transcription error: {e}")
            return None
            
    return await asyncio.get_event_loop().run_in_executor(None, worker)

# ════════════════════════════════════════════════════════
#  Core Orchestration Loop: Message Handlers
# ════════════════════════════════════════════════════════

@dp.message(F.content_type.in_({"text", "voice"}))
async def handle_any_message(message: Message):
    user_id   = str(message.from_user.id)
    user_name = message.from_user.first_name or "Friend"
    
    # Initialize lock for this user to prevent race conditions in fast double clicks
    if user_id not in session_locks:
        session_locks[user_id] = asyncio.Lock()
        
    async with session_locks[user_id]:
        await ensure_user(user_id, user_name)
        
        # Load active session parameters from RAM or SQLite
        if user_id not in active_sessions:
            logger.info(f"⏳ [SESSION] Loading session {user_id} into RAM cache...")
            db_user = await get_user(user_id)
            
            # Formulate the initial dynamic persona block
            system_prompt = _build_persona_block(
                warmth=db_user["warmth"],
                base_attitude=db_user["base_attitude"],
                user_name=db_user["name"],
                core_memory=db_user["core_memory"],
                persona_narrative=db_user["persona_narrative"]
            )
            
            history = await load_history(user_id, system_prompt=system_prompt, limit=15)
            active_sessions[user_id] = {
                "history": history,
                "warmth": db_user["warmth"],
                "base_attitude": db_user["base_attitude"],
                "core_memory": db_user["core_memory"],
                "persona_narrative": db_user["persona_narrative"],
                "last_active_time": datetime.now()
            }
        else:
            active_sessions[user_id]["last_active_time"] = datetime.now()
            
        session = active_sessions[user_id]
        
        # ── 1. Load User Message ──────────────────────────
        user_text = ""
        voice_temp_path = None
        
        if message.text:
            user_text = message.text
        elif message.voice:
            # Voice loading
            try:
                voice = message.voice
                # Create secure temporary file
                fd, voice_temp_path = tempfile.mkstemp(suffix=".ogg")
                os.close(fd)
                
                await bot.download(voice, destination=voice_temp_path)
                logger.info(f"🎙️ [STT] Audio file downloaded: {voice_temp_path}")
                
                user_text = await _transcribe_voice(voice_temp_path)
                if not user_text:
                    await message.reply("I couldn't understand the voice...")
                    return
                logger.info(f"🎙️ [STT] Result: \"{user_text}\"")
            except Exception as e:
                logger.error(f"❌ [VOICE] Failed to process voice: {e}")
                await message.reply("I was unable to listen to your voice recording.")
                return
            finally:
                if voice_temp_path and os.path.exists(voice_temp_path):
                    try:
                        os.remove(voice_temp_path)
                    except:
                        pass

        if not user_text.strip():
             return

        # Save user dialogue turn to database
        await save_message(user_id, "user", user_text)
        session["history"].append({"role": "user", "content": user_text})
        
        # Async background saving of message to vector store
        asyncio.create_task(_safe_save_to_memory("user", user_text, user_id))
        
        # ── 2. Run Memory Summarization check (Background Task) ────
        asyncio.create_task(_safe_run_summarization(session["history"], user_id))
        
        # ── 3. Recall Semantically Relevant memories from ChromaDB ─
        memories = await recall_memories_async(user_text, user_id=user_id, n_results=3)

        # ── 4. Retrieve Time passed parameters ───────────────
        last_time = await get_last_message_time(user_id)
        await touch_message_time(user_id)
        
        time_passed_str = ""
        if last_time:
            delta = datetime.now() - last_time
            hours = delta.total_seconds() / 3600
            time_passed_str = f"Time since last interaction: {hours:.2f} hours."
            
        current_time_str = f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # ── 5. System 2: Thinking Graph Cycle ────────────────
        logger.info(f"🧠 [ORCHESTRATOR] running Cognitive Thinking Graph for {user_name}...")
        
        output: ThinkSignal = await think_graph.run_async(
            user_text=user_text,
            warmth=session["warmth"],
            user_role="creator" if user_name == "Loki" else "stranger",
            user_name=user_name,
            base_attitude=session["base_attitude"],
            memories_summary=memories,
            time_passed_str=time_passed_str,
            current_time_str=current_time_str,
            history=session["history"]
        )
        
        # Render debug terminal logs
        print(output.debug_log)
        
        # Dashboard async logging
        append_dashboard_log({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_text": user_text,
            "rin_emotion": output.emotion_id,
            "rin_attitude": output.rin_attitude,
            "response_tactic": output.tactic_id,
            "tool_called": output.tool_name
        })

        # Check for Silence Tactic
        if not output.should_speak:
            logger.info("🗣️ [SPEECH] Tactic 'ignoring' activated. Rin is silent.")
            return

        # ── 6. Tool / Function Dispatching ───────────────────
        tool_result_str = ""
        if output.needs_tool:
            logger.info(f"🔧 [TOOL CALL] Requesting tool execution: {output.tool_name} with arguments {output.tool_args}")
            # Dynamically pass user context variables if required
            args = output.tool_args or {}
            if output.tool_name in ["search_core_memory", "save_fact_to_memory"]:
                args["user_id"] = user_id
                
            tool_res = await execute_tool(output.tool_name, args)
            tool_result_str = f"\n[Factual system context from tool {output.tool_name}]: {tool_res}"
            logger.info(f"🔧 [TOOL CALL] Done. Result: {tool_res[:80]}...")

        # ── 7. Generate Speech Generation Prompt ──────────────
        speech_prompt = _build_persona_block(
            warmth=session["warmth"],
            base_attitude=output.rin_attitude,
            user_name=user_name,
            core_memory=session["core_memory"],
            persona_narrative=session["persona_narrative"]
        )
        
        # Construct dynamic logit steering bias
        logit_bias = build_generation_logit_bias(session["warmth"])
        max_tokens = TACTIC_LENGTH.get(output.tactic_id, 40)

        # Inject thinking context and tool results to steer final output speech
        steered_instruction = (
            f"{speech_prompt}\n"
            f"<thinking>\nEmotion: {output.emotion_id}. Tactic: {output.tactic_id}.\n"
            f"Hidden intent of companion: {output.hidden_intent}\n</thinking>\n"
            f"Current message of companion: \"{user_text}\""
        )
        if tool_result_str:
            steered_instruction += f"\n{tool_result_str}"

        # ── 8. Speech Graph Completion Generation ────────────
        response_text = await speech_graph.run_async(
            prompt=steered_instruction,
            logit_bias=logit_bias,
            max_tokens=max_tokens
        )
        
        # Save assistant dialogue turn to database
        await save_message(user_id, "assistant", response_text)
        session["history"].append({"role": "assistant", "content": response_text})
        
        # Async background saving of assistant response to vector store
        asyncio.create_task(_safe_save_to_memory("assistant", response_text, user_id))

        # Send response back to Telegram
        await message.reply(response_text)

        # ── 9. Warmth & Relationship Adjustments ─────────────
        # Cold responses slowly decrease warmth, neutral keeps same, warm acts differently
        warmth_delta = 0.0
        if output.emotion_id in ["tired tenderness", "quiet warmth"]:
             warmth_delta = 0.05
        elif output.emotion_id in ["dry sarcasm", "irritation", "quiet contempt"]:
             warmth_delta = -0.05
             
        if warmth_delta != 0.0:
            await update_user_warmth(user_id, warmth_delta)
            session["warmth"] = max(-1.0, min(1.0, session["warmth"] + warmth_delta))
            
        if output.rin_attitude != session["base_attitude"]:
            await update_user_attitude(user_id, output.rin_attitude)
            session["base_attitude"] = output.rin_attitude

# ════════════════════════════════════════════════════════
#  Main Startup Thread
# ════════════════════════════════════════════════════════

async def main():
    init_db()
    
    # Initialize tokenizer dynamic compiler
    init_speech_engine(TOKENIZER_MODEL)

    print("━" * 58)
    print("  🌸  Rin V10 — launched successfully")
    print("━" * 58)
    
    # Start cleaner loop task and bot polling
    asyncio.create_task(_session_cleaner_loop())
    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")