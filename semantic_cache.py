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
        client = memory._client
        _cache_collection = client.get_or_create_collection(
            name="rin_semantic_cache",
            metadata={"hnsw:space": "cosine"}
        )
        return _cache_collection
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC CACHE] Error initializing cache collection: {e}")
        return None


def get_semantic_cache(user_text: str, attitude: str, warmth_tier: str) -> Optional[str]:
    """Looks up a semantically similar response in cache with attitude and warmth_tier filtering."""
    # Do not cache slash commands
    if user_text.strip().startswith("/"):
        return None
        
    collection = _init_cache_collection()
    if not collection:
        return None
        
    try:
        import memory
        if collection.count() == 0:
            return None
            
        embedding = memory._embedder.encode(user_text).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
            where={"$and": [{"attitude": attitude}, {"warmth_tier": warmth_tier}]}
        )
        
        if not results["documents"] or not results["documents"][0]:
            return None
            
        distance = results["distances"][0][0]
        # Cosine distance threshold: 0.08 means >92% similarity
        if distance < 0.08:
            metadata = results["metadatas"][0][0]
            ai_response = metadata.get("ai_response")
            logger.info(f"⚡ [SEMANTIC CACHE] Hit! Similarity: {1 - distance:.4f}, returning cached response.")
            return ai_response
            
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC CACHE] Error reading from cache: {e}")
        
    return None


async def get_semantic_cache_async(user_text: str, attitude: str, warmth_tier: str) -> Optional[str]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_semantic_cache, user_text, attitude, warmth_tier)


def save_semantic_cache(user_text: str, ai_response: str, attitude: str, warmth_tier: str) -> None:
    """Saves prompt-response pair to semantic cache with attitude and warmth_tier metadata."""
    if user_text.strip().startswith("/") or not ai_response:
        return
        
    collection = _init_cache_collection()
    if not collection:
        return
        
    try:
        import memory
        import uuid
        embedding = memory._embedder.encode(user_text).tolist()
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
        logger.info("💾 [SEMANTIC CACHE] Response saved to cache.")
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC CACHE] Error saving to cache: {e}")


async def save_semantic_cache_async(user_text: str, ai_response: str, attitude: str, warmth_tier: str) -> None:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_semantic_cache, user_text, ai_response, attitude, warmth_tier)
