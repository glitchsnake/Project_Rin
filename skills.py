"""
skills.py — Tool and Function Dispatcher for Rin (V10.3)

Features:
  - 100% Async / Non-blocking (using aiohttp and asyncio.subprocess)
  - Isolated Python execution sandbox (V2) with execution timeouts and safety blacklist
  - Automatic dynamic loading of local modules
"""

import os
import sys
import logging
import asyncio
import aiohttp
from datetime import datetime

logger = logging.getLogger("skills")

# ════════════════════════════════════════════════════════
#  Individual Tools / Functions (Async V10.3)
# ════════════════════════════════════════════════════════

async def get_current_time() -> str:
    """Returns the exact current local time and day of the week."""
    days = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
        4: "Friday", 5: "Saturday", 6: "Sunday"
    }
    now = datetime.now()
    day_name = days.get(now.weekday(), "Unknown Day")
    return f"Current Time: {now.strftime('%H:%M:%S')}, Day of the Week: {day_name}"


async def search_wikipedia(query: str = "") -> str:
    """Async Wikipedia search via aiohttp API."""
    if not query:
        return "[Wikipedia search query is empty]"
        
    url = "https://ru.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return f"[Wikipedia Error: HTTP {resp.status}]"
                data = await resp.json()
                results = data.get("query", {}).get("search", [])
                if not results:
                    return f"Nothing found in Wikipedia for '{query}'"
                
                # Combine top-2 results snippets
                output = []
                for idx, r in enumerate(results[:2]):
                    clean_snippet = r['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
                    output.append(f"• {r['title']}: {clean_snippet}...")
                return "\n".join(output)
    except Exception as e:
        logger.warning(f"⚠️ [SKILLS] Wikipedia search error: {e}")
        return f"[Wikipedia Error: {e}]"


async def execute_python(code: str) -> str:
    """
    Executes Python calculation in an isolated secure sandbox (V2) asynchronously.
    Uses create_subprocess_exec to prevent blocking the Main Thread loop.
    """
    # Expanded Security Blacklist (protects local host from exploitation)
    BLOCKED = [
        "os.", "sys.", "subprocess", "open", "eval", "exec",
        "__import__", "importlib", "shutil", "urllib", "requests", "aiohttp"
    ]
    
    for b in BLOCKED:
        if b in code:
            return f"[SECURITY BLOCKED] Unsafe module/function usage: {b}"
            
    try:
        # Launch async subprocess execution
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            # Wait for execution with strict 5.0 seconds timeout
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            
            output = stdout.decode().strip()
            error  = stderr.decode().strip()
            
            if error:
                return f"[Python Error]: {error[:300]}"
            
            return output[:500] if output else "[Code executed successfully, no stdout]"
            
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except:
                pass
            return "[Python Error: Execution timeout limit exceeded (5.0s)]"
            
    except Exception as e:
        logger.error(f"❌ [SKILLS] Python execution crash: {e}")
        return f"[Python Critical Error]: {e}"


async def search_core_memory(query: str = "", user_id: str = "") -> str:
    """Proactive async retrieval from long-term memory (V10.3)."""
    if not query:
        return "[Search query is empty]"
    try:
        from memory import recall_memories_async
        result = await recall_memories_async(query, user_id=user_id, n_results=3)
        return result if result else "[No memories found matching the query]"
    except Exception as e:
        return f"[Memory Search Error: {e}]"


async def save_fact_to_memory(fact: str = "", user_id: str = "") -> str:
    """Conscious async fact persistence to memory (V10.3)."""
    if not fact:
        return "[Fact description is empty]"
    try:
        from memory import save_to_memory_async
        await save_to_memory_async("fact", fact, user_id=user_id, extra_meta={"type": "explicit_fact"})
        logger.info(f"💾 [SKILLS] Fact saved for {user_id}: {fact[:60]}")
        return "[Got it. I committed it to my memory]"
    except Exception as e:
        return f"[Memory Save Error: {e}]"


async def get_real_world_context(city: str, info_type: str = "weather") -> str:
    """Async weather context loader via wttr.in (V10.3)."""
    if info_type != "weather":
        return "[Only 'weather' type is currently supported]"
        
    url = f"https://wttr.in/{city}"
    params = {"format": 3, "lang": "en"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return f"[Weather Error: HTTP {resp.status}]"
                data = await resp.text()
                return data.strip() if data else f"[Weather for {city}: no data]"
    except Exception as e:
        logger.warning(f"⚠️ [SKILLS] Weather error: {e}")
        return f"[Weather Error: {e}]"


# ════════════════════════════════════════════════════════
#  Tool Dispatcher (Async V10.3)
# ════════════════════════════════════════════════════════

TOOL_MAP = {
    "get_current_time":      get_current_time,
    "search_wikipedia":       search_wikipedia,
    "execute_python":         execute_python,
    "search_core_memory":     search_core_memory,
    "save_fact_to_memory":    save_fact_to_memory,
    "get_real_world_context": get_real_world_context,
}


async def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Primary asynchronous skills and functions dispatcher (V10.3)."""
    if tool_name not in TOOL_MAP:
        return f"[Tool {tool_name!r} not found]"
        
    try:
        if tool_args:
            result = await TOOL_MAP[tool_name](**tool_args)
        else:
            result = await TOOL_MAP[tool_name]()
            
        logger.info(f"🔧 [SKILLS] {tool_name} → {str(result)[:60]}")
        return result
    except Exception as e:
        logger.warning(f"⚠️ [SKILLS] {tool_name}: {e}")
        return f"[Execution Error in {tool_name}: {e}]"


async def execute_tool_async(tool_name: str, tool_args: dict) -> str:
    """Legacy wrapper for backward compatibility (V10.3)."""
    return await execute_tool(tool_name, tool_args)
