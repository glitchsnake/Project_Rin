import logging
import numpy as np
import asyncio
from typing import Optional

logger = logging.getLogger("semantic_router")

# Anchor sentences for classification (English)
ROUTE_ANCHORS = {
    "tools": [
        "execute python code",
        "run python script",
        "calculate math",
        "search wikipedia",
        "what is the weather in",
        "search facts about",
        "calculate the value of",
        "write a python program",
        "run calculation",
    ],
    "deep_thought": [
        "tell me about your depression",
        "what is the meaning of life",
        "why is everything gray",
        "who are you really",
        "let's talk about feelings",
        "why are you sad",
        "what do you think about death",
        "I feel lonely talk to me",
        "how do you feel about me",
        "explain your philosophy",
    ]
}

_anchor_embeddings = {}
_initialized = False

def init_router():
    global _anchor_embeddings, _initialized
    if _initialized:
        return
        
    import memory
    if not memory.is_memory_available():
        return
        
    try:
        for route_name, anchors in ROUTE_ANCHORS.items():
            embeddings = [memory._embedder.encode(anchor).tolist() for anchor in anchors]
            _anchor_embeddings[route_name] = embeddings
        _initialized = True
        logger.info("✅ [SEMANTIC ROUTER] Semantic router initialized successfully.")
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC ROUTER] Error initializing router: {e}")


def route_message(user_text: str) -> str:
    """Determines message route by embedding similarity."""
    init_router()
    if not _initialized:
        return "general"
        
    import memory
    try:
        user_emb = np.array(memory._embedder.encode(user_text).tolist())
        user_emb_norm = np.linalg.norm(user_emb)
        if user_emb_norm == 0:
            return "general"
        user_emb = user_emb / user_emb_norm
        
        best_route = "general"
        best_score = 0.0
        
        for route_name, embeddings in _anchor_embeddings.items():
            for emb in embeddings:
                emb_arr = np.array(emb)
                emb_norm = np.linalg.norm(emb_arr)
                if emb_norm == 0:
                    continue
                emb_arr = emb_arr / emb_norm
                similarity = np.dot(user_emb, emb_arr)
                if similarity > best_score:
                    best_score = similarity
                    best_route = route_name
                    
        # Similarity threshold: 0.80 (>80% similarity)
        if best_score > 0.80:
            logger.info(f"🧭 [SEMANTIC ROUTER] Route: {best_route} (confidence: {best_score:.4f})")
            return best_route
            
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC ROUTER] Error routing message: {e}")
        
    return "general"


async def route_message_async(user_text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, route_message, user_text)
