import logging
import numpy as np
import asyncio
from typing import Optional

logger = logging.getLogger("semantic_router")

# Вспомогательные предложения для классификации (Russian)
ROUTE_ANCHORS = {
    "tools": [
        "выполни код на python",
        "запусти скрипт",
        "посчитай математику",
        "найди в википедии",
        "какая погода",
        "поиск фактов о",
        "вычисли значение",
        "напиши программу",
        "выполни вычисления",
    ],
    "deep_thought": [
        "расскажи о своей депрессии",
        "в чем смысл жизни",
        "почему все серое",
        "кто ты на самом деле",
        "давай поговорим о чувствах",
        "почему тебе грустно",
        "что ты думаешь о смерти",
        "мне одиноко поговори со мной",
        "какие у тебя чувства ко мне",
        "расскажи философию",
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
            embeddings = [memory._embed(anchor) for anchor in anchors]
            _anchor_embeddings[route_name] = embeddings
        _initialized = True
        logger.info("✅ [SEMANTIC ROUTER] Семантический роутер успешно инициализирован.")
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC ROUTER] Ошибка инициализации роутера: {e}")


def route_message(user_text: str) -> str:
    """Определяет маршрут сообщения по сходству эмбеддингов."""
    init_router()
    if not _initialized:
        return "general"
        
    import memory
    try:
        user_emb = np.array(memory._embed(user_text))
        # Нормализация
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
                    
        # Порог сходства для маршрутизации: 0.80 (более 80% совпадения)
        if best_score > 0.80:
            logger.info(f"🧭 [SEMANTIC ROUTER] Маршрут: {best_route} (уверенность: {best_score:.4f})")
            return best_route
            
    except Exception as e:
        logger.warning(f"⚠️  [SEMANTIC ROUTER] Ошибка маршрутизации: {e}")
        
    return "general"


async def route_message_async(user_text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, route_message, user_text)
