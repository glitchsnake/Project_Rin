import logging
from typing import Optional
from datetime import datetime
import asyncio

logger = logging.getLogger("semantic_cache")

_cache_collection = None

def _init_cache_collection():
    global _cache_collection
    if _cache_collection is not None:
        return _cache_collection
    
    import memory
    if not memory.is_memory_available():
        return None
        
    try:
        client = memory._chroma_client
        _cache_collection = client.get_or_create_collection(
            name="rin_semantic_cache",
            metadata={"hnsw:space": "cosine"}
        )
        return _cache_collection
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC CACHE] Ошибка инициализации коллекции кэша: {e}")
        return None


def get_semantic_cache(user_text: str, attitude: str, warmth_tier: str) -> Optional[str]:
    """Ищет семантически похожий ответ в кэше с учетом настроения (attitude) и теплоты (warmth_tier)."""
    # Не кэшируем команды
    if user_text.strip().startswith("/"):
        return None
        
    collection = _init_cache_collection()
    if not collection:
        return None
        
    try:
        import memory
        if collection.count() == 0:
            return None
            
        embedding = memory._embed(user_text)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
            where={"$and": [{"attitude": attitude}, {"warmth_tier": warmth_tier}]}
        )
        
        if not results["documents"] or not results["documents"][0]:
            return None
            
        distance = results["distances"][0][0]
        # Порог косинусного расстояния: 0.08 означает >92% сходства
        if distance < 0.08:
            metadata = results["metadatas"][0][0]
            ai_response = metadata.get("ai_response")
            logger.info(f"⚡ [SEMANTIC CACHE] Попадание! Сходство: {1 - distance:.4f}, возвращаем кэшированный ответ.")
            return ai_response
            
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC CACHE] Ошибка чтения из кэша: {e}")
        
    return None


async def get_semantic_cache_async(user_text: str, attitude: str, warmth_tier: str) -> Optional[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_semantic_cache, user_text, attitude, warmth_tier)


def save_semantic_cache(user_text: str, ai_response: str, attitude: str, warmth_tier: str) -> None:
    """Сохраняет пару запрос-ответ в семантический кэш с учетом настроения (attitude) и теплоты (warmth_tier)."""
    if user_text.strip().startswith("/") or not ai_response:
        return
        
    collection = _init_cache_collection()
    if not collection:
        return
        
    try:
        import memory
        import uuid
        embedding = memory._embed(user_text)
        doc_id = str(uuid.uuid4())
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[user_text],
            metadatas=[{
                "ai_response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "attitude": attitude,
                "warmth_tier": warmth_tier
            }]
        )
        logger.info("💾 [SEMANTIC CACHE] Ответ сохранен в кэш.")
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC CACHE] Ошибка сохранения в кэш: {e}")


async def save_semantic_cache_async(user_text: str, ai_response: str, attitude: str, warmth_tier: str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_semantic_cache, user_text, ai_response, attitude, warmth_tier)
