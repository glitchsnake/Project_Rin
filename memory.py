"""
memory.py — Token-Based Long-Term Memory & Context Manager for Rin (V10)

Key Features:
  - Token-Based Context Triggering (tiktoken, summarization triggered at >3500 tokens).
  - Cosine Similarity-based Semantic Deduplication (using sentence-transformers, prevents vector garbage).
  - Pydantic structured output mapping for GraphRAG entity extractions and Core Memory recommendations.
"""

import os
import uuid
import logging
import asyncio
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
import tiktoken

# Lazy Loading imports for performance acceleration
chromadb = None
SentenceTransformer = None

logger = logging.getLogger("memory")

# Configuration thresholds
TOKEN_THRESHOLD     = 3500     # Summarization limit (tokens)
TOKEN_PRUNE_TARGET  = 1200     # Pruning retention target (tokens)
DUPLICATE_THRESHOLD = 0.08     # 1 - Cosine Similarity threshold (equivalent to similarity > 0.92)

# Global memory engine references
_client = None
_collection = None
_embedder = None
_memory_available = False

# ════════════════════════════════════════════════════════
#  Data Schemas (Pydantic Structured Output)
# ════════════════════════════════════════════════════════

class DialogSummaryOutput(BaseModel):
    """Dialogue summary contract with entity extraction and core memory signals."""
    summary: str = Field(..., description="Chronological, highly compressed summary of new facts and interactions")
    entities: List[str] = Field(..., description="Key extracted entity names, topics, or hashtags")
    core_memory_update: Optional[str] = Field(None, description="Suggested direct modifications or updates to User's Core Memory")

# ════════════════════════════════════════════════════════
#  Token Measurement Utilities
# ════════════════════════════════════════════════════════

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """Calculates precisely the number of tokens in a string using tiktoken."""
    if not text:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_history_tokens(history: list[dict], model_name: str = "gpt-4o") -> int:
    """Calculates precisely the number of tokens in the chat history list."""
    num_tokens = 0
    for message in history:
        num_tokens += 4  # overhead per message
        for key, value in message.items():
            num_tokens += count_tokens(str(value), model_name)
            if key == "name":
                num_tokens += 1
    num_tokens += 2  # overhead per reply
    return num_tokens

# ════════════════════════════════════════════════════════
#  Long-Term Vector Memory Engine (Lazy ChromaDB)
# ════════════════════════════════════════════════════════

def _init_memory() -> bool:
    """Lazy initialization of ChromaDB and embedding models to accelerate startup times."""
    global chromadb, SentenceTransformer, _client, _collection, _embedder, _memory_available
    if _memory_available:
        return True
        
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        # Setup persistent ChromaDB store
        _client = chromadb.PersistentClient(path="rin_memory_db")
        _collection = _client.get_or_create_collection(
            name="rin_dialog_memory",
            metadata={"hnsw:space": "cosine"}  # standard cosine distance
        )
        
        # Load local embedding model
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _memory_available = True
        logger.info("✅ [MEMORY] ChromaDB and sentence-transformers initialized successfully.")
        return True
    except Exception as e:
        logger.warning(f"⚠️ [MEMORY] Failed to load long-term memory engine dependencies (using memory fallbacks): {e}")
        _memory_available = False
        return False


def is_memory_available() -> bool:
    """Returns True if the long-term memory engines are loaded and ready."""
    return _memory_available or _init_memory()


def _is_important(role: str, content: str) -> bool:
    """Checks whether the statement contains enough information to warrant long-term storage."""
    if role == "system":
        return False
    return len(content.strip()) > 15

# ════════════════════════════════════════════════════════
#  Semantic Vector Memory Operations (Async V10)
# ════════════════════════════════════════════════════════

def save_to_memory(role: str, content: str, user_id: str, extra_meta: Optional[dict] = None):
    """
    Saves a message to vector memory.
    Performs cosine similarity deduplication to prevent vector garbage.
    If similarity > 0.92, increments frequency instead of adding a new document.
    """
    if not is_memory_available() or not _is_important(role, content):
        return
        
    try:
        # Precompute text embeddings
        emb = _embedder.encode(content).tolist()
        now_str = datetime.now().isoformat()
        
        # Query closest existing document in user's index space
        results = _collection.query(
            query_embeddings=[emb],
            n_results=1,
            where={"user_id": user_id}
        )
        
        # Check for semantic duplicate (distance < DUPLICATE_THRESHOLD)
        if results and results["ids"] and results["ids"][0]:
            dist = results["distances"][0][0]
            if dist < DUPLICATE_THRESHOLD:
                dup_id = results["ids"][0][0]
                meta = results["metadatas"][0][0]
                freq = meta.get("frequency", 1) + 1
                
                # Update existing duplicate document frequency
                meta["frequency"] = freq
                meta["last_seen_timestamp"] = now_str
                if extra_meta:
                    meta.update(extra_meta)
                    
                _collection.update(
                    ids=[dup_id],
                    metadatas=[meta]
                )
                logger.info(f"💾 [MEMORY] Semantic duplicate detected (dist={dist:.3f}). Incremented frequency to {freq} for {dup_id}")
                return

        # No duplicate found: insert new document
        doc_id = str(uuid.uuid4())
        metadata = {
            "user_id": user_id,
            "role": role,
            "frequency": 1,
            "created_timestamp": now_str,
            "last_seen_timestamp": now_str
        }
        if extra_meta:
            metadata.update(extra_meta)
            
        _collection.add(
            ids=[doc_id],
            documents=[content],
            embeddings=[emb],
            metadatas=[metadata]
        )
        logger.info(f"💾 [MEMORY] Saved new factual document {doc_id} to ChromaDB")
    except Exception as e:
        logger.error(f"❌ [MEMORY] Save operation failed: {e}")


def recall_memories(query: str, user_id: str, n_results: int = 3) -> str:
    """Queries top relevant facts from long-term memory matched using semantic search."""
    if not is_memory_available() or not query:
        return ""
        
    try:
        emb = _embedder.encode(query).tolist()
        results = _collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            where={"user_id": user_id}
        )
        
        if not results or not results["documents"] or not results["documents"][0]:
            return ""
            
        docs = results["documents"][0]
        logger.info(f"💾 [MEMORY] Recalled {len(docs)} relevant memories for query '{query[:30]}'")
        return " | ".join(docs)
    except Exception as e:
        logger.error(f"❌ [MEMORY] Recall operation failed: {e}")
        return ""

# ════════════════════════════════════════════════════════
#  Thread-Safe Non-blocking Async Wrappers
# ════════════════════════════════════════════════════════

async def save_to_memory_async(*args, **kwargs):
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: save_to_memory(*args, **kwargs)
    )

async def recall_memories_async(*args, **kwargs) -> str:
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: recall_memories(*args, **kwargs)
    )

# ════════════════════════════════════════════════════════
#  Token-Bound Adaptive Summarizer
# ════════════════════════════════════════════════════════

async def summarize_if_needed(history: list[dict], user_id: str, client, model: str) -> tuple[list[dict], Optional[DialogSummaryOutput]]:
    """
    Monitors history token size.
    If it exceeds TOKEN_THRESHOLD (3500 tokens), compiles a compressed structured summary 
    using the LLM Pydantic parser, updates history, and returns recommendations.
    """
    current_tokens = count_history_tokens(history)
    if current_tokens < TOKEN_THRESHOLD:
        return history, None
        
    logger.info(f"🧠 [MEMORY] Token boundary exceeded ({current_tokens} > {TOKEN_THRESHOLD}). Initiating summarization...")
    
    # Isolate messages excluding system instruction rules
    prune_candidates = [m for m in history if m["role"] != "system"]
    if not prune_candidates:
        return history, None
        
    # Build summarization context blocks
    content_to_summarize = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in prune_candidates])
    
    sys_instruction = (
        "You are the long-term memory engine for Rin.\n"
        "Analyze the dialogue context and map structured JSON matching DialogSummaryOutput schema.\n"
        "Compile a chronologically accurate, highly compressed narrative summary of new user facts, events, and interactions.\n"
        "Extract key entities (user habits, context details, names) as tags.\n"
        "Generate concrete User Core Memory updates if new preferences, schedules, or relations are discovered."
    )
    
    try:
        # LLM parsing with strict Pydantic schemas
        completion = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": sys_instruction},
                    {"role": "user", "content": f"Dialogue history to compress:\n{content_to_summarize}"}
                ],
                response_format=DialogSummaryOutput
            )
        )
        
        summary_result: DialogSummaryOutput = completion.choices[0].message.parsed
        logger.info(f"🧠 [MEMORY] Dialogue summarized. Key entities: {summary_result.entities}")
        
        # Save summary to vector memory to make it semantically searchable
        extra_meta = {"type": "history_summary", "entities": ",".join(summary_result.entities)}
        await save_to_memory_async("system", summary_result.summary, user_id, extra_meta)
        
        # Pruning: retain the system prompt and the most recent dialogue turns (clamped to TOKEN_PRUNE_TARGET)
        sys_prompts = [m for m in history if m["role"] == "system"]
        
        retained = []
        accumulated_tokens = 0
        
        # Traverse backwards to keep the most recent messages intact
        for m in reversed(prune_candidates):
            m_tokens = count_tokens(m["content"])
            if accumulated_tokens + m_tokens > TOKEN_PRUNE_TARGET and len(retained) >= 4:
                break
            retained.insert(0, m)
            accumulated_tokens += m_tokens
            
        pruned_history = sys_prompts + retained
        logger.info(f"🧠 [MEMORY] Pruning complete. History token size optimized: {current_tokens} → {count_history_tokens(pruned_history)} tokens.")
        
        return pruned_history, summary_result
        
    except Exception as e:
        logger.error(f"❌ [MEMORY] Summarization process failed: {e}")
        return history, None
