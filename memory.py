"""
memory.py — Долгосрочная векторная память Rin (V10)

[V10] Архитектура:
      - Иерархическая память (Core, Episodic, Semantic)
      - Токенизированный контроль буфера (tiktoken, 3500 tokens)
      - Семантическая дедупликация (cosine distance < 0.08)
      - GraphRAG Emulation (Extraction of entities/tags)
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Optional, List
import tiktoken
from pydantic import BaseModel, Field

logger = logging.getLogger("memory")

# ────────────────────────────────────────────────────────
# Модели данных
# ────────────────────────────────────────────────────────

class DialogueSummaryOutput(BaseModel):
    """Схема структурированного сжатия истории (V10)."""
    summary_text: str = Field(..., description="Сжатый текст диалога от 3-го лица")
    extracted_entities: List[str] = Field(..., description="Список ключевых сущностей и тем (технологии, имена, события)")
    core_memory_update: str = Field(..., description="Рекомендации по обновлению Core Memory, если обнаружены важные изменения")

# ────────────────────────────────────────────────────────
# Ленивая инициализация
# ────────────────────────────────────────────────────────

_chroma_client    = None
_collection       = None
_embedder         = None
_memory_available = False


def _init_memory() -> bool:
    """Инициализация ChromaDB + sentence-transformers (ленивая)."""
    global _chroma_client, _collection, _embedder, _memory_available
    if _memory_available:
        return True
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        chroma_host = os.getenv("CHROMA_HOST")
        if chroma_host:
            chroma_port = int(os.getenv("CHROMA_PORT", 8000))
            logger.info(f"🔌 [MEMORY] Подключение к удаленному ChromaDB серверу: {chroma_host}:{chroma_port}")
            _chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            _chroma_client = chromadb.PersistentClient(path="./rin_memory_db")

        _collection = _chroma_client.get_or_create_collection(
            name="rin_memories",
            metadata={"hnsw:space": "cosine"},
        )
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _memory_available = True
        logger.info("✅ [MEMORY] ChromaDB + эмбеддер инициализированы.")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  [MEMORY] Библиотеки не установлены: {e}. Память отключена.")
        return False
    except Exception as e:
        logger.warning(f"⚠️  [MEMORY] Ошибка инициализации: {e}. Память отключена.")
        return False


def is_memory_available() -> bool:
    return _memory_available or _init_memory()


def _embed(text: str) -> list[float]:
    return _embedder.encode(text, normalize_embeddings=True).tolist()


def _is_important(role: str, content: str) -> bool:
    if role == "system":
        return False
    return len(content.strip()) > 15


# ────────────────────────────────────────────────────────
# Утилиты токенизации
# ────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Подсчет токенов с использованием tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_history_tokens(history: list, model: str = "gpt-4o") -> int:
    """Подсчет токенов во всей истории сообщений."""
    total = 0
    for m in history:
        total += count_tokens(m.get("content", ""), model)
        total += 4  # Примерный оверхед на структуру сообщения
    return total


# ════════════════════════════════════════════════════════
#  V10: Сохранение с семантической дедупликацией
# ════════════════════════════════════════════════════════

DUPLICATE_THRESHOLD = 0.08  # 1 - 0.92 = 0.08 (косинусное расстояние)

def save_to_memory(
    role: str,
    content: str,
    user_id: str = "global",
    extra_meta: Optional[dict] = None,
) -> None:
    """
    Сохраняет сообщение в ChromaDB с проверкой на дубликаты (V10).
    """
    if not is_memory_available():
        return
        
    # Ограничение размера входного текста (защита от гигантских мусорных текстов)
    content = content[:1000]
    
    if not _is_important(role, content):
        return

    try:
        # Квота памяти: не более 500 записей на одного пользователя (FIFO вытеснение)
        try:
            user_docs = _collection.get(where={"user_id": user_id}, include=["metadatas"])
            if user_docs and user_docs["ids"] and len(user_docs["ids"]) >= 500:
                oldest_id = None
                oldest_time = None
                for doc_id, meta in zip(user_docs["ids"], user_docs["metadatas"]):
                    ts_str = meta.get("timestamp") or meta.get("last_seen_timestamp")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if oldest_time is None or ts < oldest_time:
                                oldest_time = ts
                                oldest_id = doc_id
                        except Exception:
                            oldest_id = doc_id
                            break
                    else:
                        oldest_id = doc_id
                        break
                if oldest_id:
                    _collection.delete(ids=[oldest_id])
                    logger.info(f"🗑️ [MEMORY] Превышена квота (500) для {user_id}. Удалена старая запись: {oldest_id}")
        except Exception as quota_err:
            logger.warning(f"⚠️ [MEMORY] Ошибка проверки квоты: {quota_err}")

        embedding = _embed(content)
        
        # Семантическая проверка на дубликаты (V10)
        existing = _collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"user_id": user_id}
        )

        if (existing["distances"] and existing["distances"][0] and 
            existing["distances"][0][0] < DUPLICATE_THRESHOLD):
            
            doc_id = existing["ids"][0][0]
            meta = existing["metadatas"][0][0]
            meta["last_seen_timestamp"] = datetime.now().isoformat()
            meta["frequency"] = meta.get("frequency", 1) + 1
            if extra_meta:
                meta.update(extra_meta)
            
            _collection.update(ids=[doc_id], metadatas=[meta])
            logger.debug(f"♻️  [MEMORY] [{user_id}] Дедупликация: {content[:30]}...")
            return

        # Стандартная запись нового эмбеддинга
        doc_id = f"{user_id}_{role}_{datetime.now().isoformat()}_{hash(content) & 0xFFFFFF}"
        metadata: dict = {
            "role":            role,
            "user_id":         user_id,
            "timestamp":       datetime.now().isoformat(),
            "content_preview": content[:100],
            "frequency":       1
        }
        if extra_meta:
            metadata.update(extra_meta)

        _collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )
        logger.debug(f"💾 [MEMORY] [{user_id}] [{role}] {content[:50]}...")
    except Exception as e:
        logger.warning(f"⚠️  [MEMORY] Ошибка сохранения: {e}")


async def save_to_memory_async(
    role: str,
    content: str,
    user_id: str = "global",
    extra_meta: Optional[dict] = None,
) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_to_memory, role, content, user_id, extra_meta)


# ════════════════════════════════════════════════════════
#  Поиск воспоминаний (V10: Гибридный поиск)
# ════════════════════════════════════════════════════════

def recall_memories(
    query: str,
    user_id: str = "global",
    tags: Optional[List[str]] = None,
    n_results: int = 3,
) -> str:
    """
    Семантический поиск + опциональная фильтрация по тегам.
    """
    if not is_memory_available():
        return ""
    try:
        count = _collection.count()
        if count == 0:
            return ""

        where_clause: dict = {"user_id": user_id}
        if tags and len(tags) > 0:
            # ChromaDB v0.4+ поддерживает фильтрацию по метаданным
            # Если теги хранятся как строка, можно использовать только точное совпадение
            # или если мы перейдем на хранение списков (но ChromaDB метаданные ограничены примитивами).
            # Для простоты: ищем по первому тегу через $contains если поддерживается, 
            # либо просто логируем намерение.
            where_clause["tags"] = {"$contains": tags[0]}

        embedding = _embed(query)
        results = _collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas"],
            where=where_clause,
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            role_label = "Юзер сказал" if meta.get("role") == "user" else "Rin ответила"
            ts = meta.get("timestamp", "")[:10]
            # Добавляем теги в вывод если они есть
            tags_info = f" [теги: {meta.get('tags')}]" if meta.get("tags") else ""
            parts.append(f"{role_label} ({ts}){tags_info}: «{doc[:80]}»")

        return "; ".join(parts) if parts else ""

    except Exception as e:
        logger.warning(f"⚠️  [MEMORY] Ошибка поиска: {e}")
        return ""


async def recall_memories_async(
    query: str,
    user_id: str = "global",
    tags: Optional[List[str]] = None,
    n_results: int = 3,
) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, recall_memories, query, user_id, tags, n_results)


# ════════════════════════════════════════════════════════
#  [V10] Менеджер саммаризации (Token-based)
# ════════════════════════════════════════════════════════

TOKEN_LIMIT = 3500

async def maybe_summarize_history(
    chat_history: list,
    client,
    model: str,
    user_id: str = "global",
) -> tuple[list, Optional[str]]:
    """
    Фоновая саммаризация по лимиту токенов (V10).
    Возвращает (новый_список_истории, предложение_по_core_memory).
    """
    total_tokens = count_history_tokens(chat_history, model)
    
    if total_tokens < TOKEN_LIMIT:
        return chat_history, None

    logger.info(f"📝 [MEMORY] Лимит токенов превышен ({total_tokens}/{TOKEN_LIMIT}). Саммаризация для {user_id}...")

    # Оставляем последние 5 сообщений для контекста
    non_system = [m for m in chat_history if m["role"] != "system"]
    to_summarize = non_system[:-5]
    to_keep      = non_system[-5:]

    if not to_summarize:
        return chat_history, None

    dialog_text = "\n".join(
        f"{'Юзер' if m['role'] == 'user' else 'Rin'}: {m['content']}"
        for m in to_summarize
    )

    core_update_suggestion = None

    try:
        # Используем Structured Outputs (parse)
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — архивариус Rin. Сожми фрагмент диалога в один абзац (от 3-го лица). "
                        "Извлеки ключевые сущности и предложи обновления для Core Memory."
                    ),
                },
                {"role": "user", "content": dialog_text},
            ],
            response_format=DialogueSummaryOutput,
        )
        
        output = completion.choices[0].message.parsed
        
        # Сохраняем в ChromaDB с тегами (GraphRAG Emulation)
        tags_str = ", ".join(output.extracted_entities)
        await save_to_memory_async(
            "summary", output.summary_text,
            user_id=user_id,
            extra_meta={
                "type": "dialog_summary",
                "tags": tags_str,
                "entities_count": len(output.extracted_entities)
            },
        )
        
        if output.core_memory_update:
            core_update_suggestion = output.core_memory_update
            logger.info(f"💡 [MEMORY] Предложение для Core Memory ({user_id}): {core_update_suggestion}")

        logger.info(f"✅ [MEMORY] Саммаризация завершена. Извлечено сущностей: {len(output.extracted_entities)}")

    except Exception as e:
        logger.warning(f"⚠️  [MEMORY] Ошибка саммаризации: {e}")
        return chat_history, None

    system_msgs = [m for m in chat_history if m["role"] == "system"]
    return system_msgs + to_keep, core_update_suggestion
