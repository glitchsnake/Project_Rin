"""
database.py — Asynchronous SQLite Database Layer for Rin (V10.2)

Uses aiosqlite to prevent blocking the Main Thread loop.
Contains user state persistence, relationship warmth adjustments, and session logs.
"""

import sqlite3
import aiosqlite
import logging
import asyncio
import functools
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("database")

DB_PATH = Path("./rin_sessions.db")

# ════════════════════════════════════════════════════════
#  Synchronous Table Initialization (Runs at Startup)
# ════════════════════════════════════════════════════════

def init_db():
    """Initializes the database schema synchronously at startup."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # User Profiles: holds social parameters, attitude, and core memory
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                warmth REAL DEFAULT 0.0,
                base_attitude TEXT DEFAULT 'нейтральное',
                core_memory TEXT DEFAULT '',
                persona_narrative TEXT DEFAULT ''
            )
        """)
        
        # Session Logs: saves historical logs of all conversations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES users(session_id)
            )
        """)
        
        # Session Metadata: tracks timestamps and operational stats
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT PRIMARY KEY,
                last_message_time TEXT,
                FOREIGN KEY (session_id) REFERENCES users(session_id)
            )
        """)
        
        conn.commit()
    logger.info(f"✅ [DB] Initialized at: {DB_PATH}")

# ════════════════════════════════════════════════════════
#  Asynchronous CRUD Queries (Async V10.2)
# ════════════════════════════════════════════════════════

async def ensure_user(session_id: str, name: str):
    """Ensures a user record exists, creates one with default state if not found."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO users (session_id, name) VALUES (?, ?)",
            (session_id, name)
        )
        await db.commit()


async def get_user(session_id: str) -> dict:
    """Retrieves user profile information as a structured dictionary."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE session_id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
            return None


async def update_user_warmth(session_id: str, delta: float):
    """Increments or decrements the user's relationship warmth level (clamped to [-1, 1])."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT warmth FROM users WHERE session_id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                current_warmth = row[0]
                new_warmth = max(-1.0, min(1.0, current_warmth + delta))
                await db.execute(
                    "UPDATE users SET warmth = ? WHERE session_id = ?",
                    (new_warmth, session_id)
                )
                await db.commit()
                logger.info(f"🧊 [DB] Updated warmth for {session_id}: {current_warmth:.2f} → {new_warmth:.2f}")


async def update_user_attitude(session_id: str, attitude: str):
    """Updates the dynamic social attitude parameter of the user."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET base_attitude = ? WHERE session_id = ?",
            (attitude, session_id)
        )
        await db.commit()
        logger.info(f"🧊 [DB] Updated attitude for {session_id} to: {attitude}")


async def update_core_memory(session_id: str, memory_text: str):
    """Updates the user's compressed long-term profile facts summary."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET core_memory = ? WHERE session_id = ?",
            (memory_text, session_id)
        )
        await db.commit()
        logger.info(f"💾 [DB] Updated core memory for {session_id}: {memory_text[:60]}")


async def update_persona_narrative(session_id: str, narrative_text: str):
    """Updates the customized relationship narrative descriptor."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET persona_narrative = ? WHERE session_id = ?",
            (narrative_text, session_id)
        )
        await db.commit()
        logger.info(f"📝 [DB] Updated persona narrative for {session_id}")


async def save_message(session_id: str, role: str, content: str):
    """Persists a dialogue exchange turn asynchronously."""
    async with aiosqlite.connect(DB_PATH) as db:
        now = datetime.now().isoformat()
        await db.execute(
            "INSERT INTO session_logs (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        await db.commit()


async def load_history(session_id: str, system_prompt: str, limit: int = 15) -> list[dict]:
    """Loads dialogue log history, formatting it with a system prompt context."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT role, content FROM session_logs WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            
            # Format and reverse to maintain chronological order
            history = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
            return [{"role": "system", "content": system_prompt}] + history


async def touch_message_time(session_id: str):
    """Touches last active conversation timestamp."""
    async with aiosqlite.connect(DB_PATH) as db:
        now = datetime.now().isoformat()
        await db.execute(
            "INSERT OR REPLACE INTO session_metadata (session_id, last_message_time) VALUES (?, ?)",
            (session_id, now)
        )
        await db.commit()


async def get_last_message_time(session_id: str) -> datetime:
    """Retrieves last active message timestamp as a datetime object."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT last_message_time FROM session_metadata WHERE session_id = ?",
            (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return datetime.fromisoformat(row[0])
            return None

# ════════════════════════════════════════════════════════
#  Dashboard Logging (Non-blocking file append)
# ════════════════════════════════════════════════════════

def append_dashboard_log(log_data: dict):
    """Appends structural reasoning logs to the dashboard in a non-blocking background thread."""
    import json
    def worker():
        try:
            with open("rin_think_dashboard.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"❌ [DB] Dashboard logging error: {e}")
            
    asyncio.get_event_loop().run_in_executor(None, worker)
