"""
database.py — Персистентная БД сессий Rin (V10.2)

V10.2: Асинхронный aiosqlite драйвер, Non-blocking Event Loop.
"""

import asyncio
import json
import logging
import sqlite3  # для синхронной инициализации/миграции
import aiosqlite
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("database")

DB_PATH        = Path("./rin_sessions.db")
THINK_LOG_PATH = Path("./rin_think_dashboard.jsonl")


# ════════════════════════════════════════════════════════
#  Инициализация БД (Синхронно при старте)
# ════════════════════════════════════════════════════════

def _migrate_db() -> None:
    """Безопасная синхронная миграция при старте."""
    migrations = [
        "ALTER TABLE sessions ADD COLUMN last_message_at TEXT",
        """CREATE TABLE IF NOT EXISTS users (
            chat_id          TEXT PRIMARY KEY,
            name             TEXT NOT NULL DEFAULT 'незнакомец',
            base_attitude    TEXT NOT NULL DEFAULT 'нейтральное',
            warmth           REAL NOT NULL DEFAULT 0.0,
            first_seen       TEXT NOT NULL,
            last_seen        TEXT NOT NULL
        )""",
        "ALTER TABLE users ADD COLUMN core_memory TEXT DEFAULT ''",
        "ALTER TABLE users ADD COLUMN persona_narrative TEXT DEFAULT ''",
        "CREATE INDEX IF NOT EXISTS idx_thinklogs_session ON think_logs(session_id, timestamp)",
    ]
    with sqlite3.connect(DB_PATH) as conn:
        for sql in migrations:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass
    logger.info("✅ [DB] Миграция выполнена.")


def init_db() -> None:
    """Создаёт таблицы (синхронно)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id               TEXT PRIMARY KEY,
                created_at       TEXT NOT NULL,
                last_active      TEXT NOT NULL,
                last_message_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS users (
                chat_id           TEXT PRIMARY KEY,
                name              TEXT NOT NULL DEFAULT 'незнакомец',
                base_attitude     TEXT NOT NULL DEFAULT 'нейтральное',
                warmth            REAL NOT NULL DEFAULT 0.0,
                core_memory       TEXT DEFAULT '',
                persona_narrative TEXT DEFAULT '',
                first_seen        TEXT NOT NULL,
                last_seen         TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS think_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                think_json  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_thinklogs_session ON think_logs(session_id, timestamp);
        """)
    logger.info(f"✅ [DB] Инициализирована: {DB_PATH}")
    _migrate_db()


# ════════════════════════════════════════════════════════
#  Сессии (Async)
# ════════════════════════════════════════════════════════

async def ensure_session(session_id: str) -> None:
    """Создаёт сессию если не существует, обновляет last_active (V10.2)."""
    now = datetime.now().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR IGNORE INTO sessions(id, created_at, last_active) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        await db.execute(
            "UPDATE sessions SET last_active=? WHERE id=?",
            (now, session_id),
        )
        await db.commit()


async def touch_message_time(session_id: str) -> None:
    """Обновляет last_message_at при каждом входящем сообщении (V10.2)."""
    await ensure_session(session_id)
    now = datetime.now().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET last_message_at=? WHERE id=?",
            (now, session_id),
        )
        await db.commit()


async def get_last_message_time(session_id: str) -> Optional[datetime]:
    """Возвращает datetime последнего сообщения или None (V10.2)."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT last_message_at FROM sessions WHERE id=?", (session_id,)) as cursor:
            row = await cursor.fetchone()
    if row and row[0]:
        try:
            return datetime.fromisoformat(row[0])
        except ValueError:
            return None
    return None


# ════════════════════════════════════════════════════════
#  Досье на пользователей (Async)
# ════════════════════════════════════════════════════════

async def ensure_user(chat_id: str, name: str = "незнакомец") -> None:
    """Создаёт запись юзера при первом контакте (V10.2)."""
    now = datetime.now().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR IGNORE INTO users(chat_id, name, base_attitude, warmth, first_seen, last_seen)
               VALUES (?, ?, 'нейтральное', 0.0, ?, ?)""",
            (chat_id, name, now, now),
        )
        # Обновляем last_seen и имя если изменилось
        await db.execute(
            "UPDATE users SET last_seen=?, name=? WHERE chat_id=? AND name='незнакомец'",
            (now, name, chat_id),
        )
        await db.execute(
            "UPDATE users SET last_seen=? WHERE chat_id=?",
            (now, chat_id),
        )
        await db.commit()


async def get_user(chat_id: str) -> dict:
    """Возвращает досье юзера (V10.2)."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT name, base_attitude, warmth, core_memory, persona_narrative FROM users WHERE chat_id=?",
            (chat_id,)
        ) as cursor:
            row = await cursor.fetchone()
    if row:
        return {
            "name":              row[0],
            "base_attitude":     row[1],
            "warmth":            row[2],
            "core_memory":       row[3] or "",
            "persona_narrative": row[4] or "",
        }
    return {"name": "незнакомец", "base_attitude": "нейтральное",
            "warmth": 0.0, "core_memory": "", "persona_narrative": ""}


async def update_user_warmth(chat_id: str, delta: float) -> float:
    """Обновляет warmth юзера в БД (V10.2)."""
    await ensure_user(chat_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET warmth = MAX(-1.0, MIN(1.0, warmth + ?)) WHERE chat_id=?",
            (delta, chat_id),
        )
        await db.commit()
        async with db.execute("SELECT warmth FROM users WHERE chat_id=?", (chat_id,)) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else 0.0


async def update_user_attitude(chat_id: str, new_attitude: str) -> None:
    """Перезаписывает base_attitude после IdleGraph 'сна' (V10.2)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET base_attitude=? WHERE chat_id=?",
            (new_attitude, chat_id),
        )
        await db.commit()
    logger.info(f"🌙 [DB] Attitude юзера {chat_id} → {new_attitude}")


async def update_core_memory(chat_id: str, core_memory: str) -> None:
    """Обновляет Core Memory юзера (V10.2)."""
    await ensure_user(chat_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET core_memory=? WHERE chat_id=?",
            (core_memory[:500], chat_id),
        )
        await db.commit()


async def update_persona_narrative(chat_id: str, narrative: str) -> None:
    """Обновляет нарратив отношений (V10.2)."""
    await ensure_user(chat_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET persona_narrative=? WHERE chat_id=?",
            (narrative[:600], chat_id),
        )
        await db.commit()
    logger.info(f"🌙 [DB] Нарратив {chat_id} обновлён.")


async def set_user_name(chat_id: str, name: str) -> None:
    """Сохраняет имя юзера (V10.2)."""
    await ensure_user(chat_id, name)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET name=? WHERE chat_id=?",
            (name, chat_id),
        )
        await db.commit()


# ════════════════════════════════════════════════════════
#  Сообщения (история) (Async)
# ════════════════════════════════════════════════════════

async def save_message(session_id: str, role: str, content: str) -> None:
    """Сохраняет сообщение в БД (V10.2)."""
    await ensure_session(session_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )
        await db.commit()


async def load_history(
    session_id: str,
    system_prompt: str,
    limit: int = 20,
) -> list[dict]:
    """Загружает chat_history из БД (V10.2)."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?""",
            (session_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()

    history = [{"role": "system", "content": system_prompt}]
    for role, content in reversed(rows):
        if role != "system":
            history.append({"role": role, "content": content})
    return history


async def update_history_in_db(session_id: str, messages: list[dict]) -> None:
    """Синхронизирует in-memory историю с БД (V10.2)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        now = datetime.now().isoformat()
        for m in messages:
            if m["role"] == "system":
                continue
            await db.execute(
                "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, m["role"], m["content"], now),
            )
        await db.commit()


async def get_recent_think_logs(session_id: str, limit: int = 50) -> list[dict]:
    """Возвращает последние N ThinkResult для IdleGraph (V10.2)."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT think_json FROM think_logs WHERE session_id=? ORDER BY id DESC LIMIT ?""",
            (session_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
    results = []
    for (raw,) in rows:
        try:
            results.append(json.loads(raw))
        except Exception:
            pass
    return list(reversed(results))


# ════════════════════════════════════════════════════════
#  Дашборд ThinkResult (Async)
# ════════════════════════════════════════════════════════

async def log_think_result(session_id: str, user_text: str, think_result) -> None:
    """Записывает ThinkResult в SQLite + JSONL-файл (V10.2)."""
    now = datetime.now().isoformat()
    think_dict = {
        "timestamp":       now,
        "session_id":      session_id,
        "user_text":       user_text,
        "hidden_intent":   getattr(think_result, "hidden_intent", ""),
        "rin_emotion":     getattr(think_result, "emotion_id", getattr(think_result, "rin_emotion", "")),
        "rin_attitude":    getattr(think_result, "rin_attitude", ""),
        "response_tactic": getattr(think_result, "tactic_id", getattr(think_result, "response_tactic", "")),
        "should_ignore":   not getattr(think_result, "should_speak", not getattr(think_result, "should_ignore", False)),
        "needs_tool":      getattr(think_result, "needs_tool", False),
        "tool_name":       getattr(think_result, "tool_name", None),
    }
    think_json = json.dumps(think_dict, ensure_ascii=False)

    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO think_logs(session_id, timestamp, think_json) VALUES (?, ?, ?)",
                (session_id, now, think_json),
            )
            await db.commit()
    except Exception as e:
        logger.warning(f"⚠️ [DB] Ошибка записи think_log: {e}")

    # Запись в JSONL через Executor для избежания блокировок
    def _write_jsonl():
        try:
            with open(THINK_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(think_json + "\n")
        except Exception as e:
            logger.warning(f"⚠️ [DB] Ошибка записи JSONL: {e}")
            
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_jsonl)
