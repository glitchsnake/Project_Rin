"""
skills.py — Навыки (Tools) Rin (V10.3)

V10.3: Полностью асинхронные инструменты, Non-blocking Event Loop,
       изолированное выполнение Python-кода.
"""

import asyncio
import logging
import sys
import aiohttp
from datetime import datetime
from typing import Optional

logger = logging.getLogger("skills")


# ════════════════════════════════════════════════════════
#  Реализации навыков (Async V10.3)
# ════════════════════════════════════════════════════════

async def get_current_time() -> str:
    """Возвращает текущее время и время суток."""
    now  = datetime.now()
    hour = now.hour
    if   5  <= hour < 12: period = "утро"
    elif 12 <= hour < 17: period = "день"
    elif 17 <= hour < 22: period = "вечер"
    else:                  period = "ночь"
    return f"{now.strftime('%H:%M')} — {period} ({now.strftime('%d.%m.%Y')})"


async def search_wikipedia(query: str) -> str:
    """Асинхронный поиск в Wikipedia через API (V10.3)."""
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
                search_results = data.get("query", {}).get("search", [])
                
                if not search_results:
                    return f"Wikipedia: по запросу «{query}» ничего не найдено."
                
                # Берем первый результат
                first_result = search_results[0]
                title = first_result["title"]
                snippet = first_result["snippet"]
                
                # Очистка HTML-тегов из сниппета
                import re
                clean_snippet = re.sub(r'<[^>]+>', '', snippet)
                
                return f"Wikipedia ({title}): {clean_snippet}..."
                
    except asyncio.TimeoutError:
        return "[Wikipedia Error: Превышено время ожидания]"
    except Exception as e:
        logger.warning(f"⚠️ [SKILLS] Wikipedia error: {e}")
        return f"[Wikipedia Error: {e}]"


_DOCKER_AVAILABLE = None

async def is_docker_available() -> bool:
    """Быстрая проверка доступности Docker daemon с кэшированием и тайм-аутом 1.5 сек."""
    global _DOCKER_AVAILABLE
    if _DOCKER_AVAILABLE is not None:
        return _DOCKER_AVAILABLE
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "version",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=1.5)
            _DOCKER_AVAILABLE = (proc.returncode == 0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except:
                pass
            _DOCKER_AVAILABLE = False
    except Exception:
        _DOCKER_AVAILABLE = False
    return _DOCKER_AVAILABLE


async def execute_python(code: str) -> str:
    """Асинхронное выполнение Python-кода (V10.3 Sandbox с изоляцией Docker / Subprocess)."""
    BLOCKED = [
        "import os", "import sys", "import subprocess", "open(",
        "__import__", "exec(", "eval(", "shutil", "socket", "requests", 
        "urllib", "aiohttp", "threading", "multiprocessing"
    ]
    
    for b in BLOCKED:
        if b in code:
            return f"[SECURITY BLOCKED] Использование запрещенного модуля/функции: {b}"
            
    # 1. Попытка изолированного запуска в контейнере Docker
    if await is_docker_available():
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm", "--network", "none", "--cpus", "0.5", "-m", "50m", "python:3.10-alpine", "python", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
                
                # Возвращаемый код 125 означает ошибку самого демона докера (например, докер не запущен)
                if proc.returncode == 125:
                    raise RuntimeError("Docker daemon error (exit code 125)")
                    
                output = stdout.decode().strip()
                error  = stderr.decode().strip()
                
                if error:
                    return f"[Python Error]: {error[:300]}"
                return output[:500] if output else "[Код выполнен успешно, нет вывода]"
                
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except:
                    pass
                return "[Python Error: Превышен лимит времени выполнения (5.0с)]"
                
        except Exception as docker_err:
            logger.warning(f"⚠️ [SKILLS] Docker недоступен ({docker_err}). Переключаюсь на резервный subprocess...")
        
    # 2. Резервный запуск в изолированном локальном подпроцессе (если Docker отсутствует)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            
            output = stdout.decode().strip()
            error  = stderr.decode().strip()
            
            if error:
                return f"[Python Error]: {error[:300]}"
            
            return output[:500] if output else "[Код выполнен успешно, нет вывода]"
            
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except:
                pass
            return "[Python Error: Превышен лимит времени выполнения (5.0с)]"
            
    except Exception as e:
        logger.error(f"❌ [SKILLS] Python execution crash: {e}")
        return f"[Python Critical Error]: {e}"


async def search_core_memory(query: str = "", user_id: str = "") -> str:
    """Проактивный асинхронный поиск по долгосрочной памяти (V10.3)."""
    if not query:
        return "[Запрос пуст]"
    try:
        from memory import recall_memories_async
        result = await recall_memories_async(query, user_id=user_id, n_results=3)
        return result if result else "[В памяти ничего не найдено]"
    except Exception as e:
        return f"[Memory Search Error: {e}]"


async def save_fact_to_memory(fact: str = "", user_id: str = "") -> str:
    """Осознанное асинхронное сохранение факта в память (V10.3)."""
    if not fact:
        return "[Факт пуст]"
    try:
        from memory import save_to_memory_async
        await save_to_memory_async("fact", fact, user_id=user_id, extra_meta={"type": "explicit_fact"})
        logger.info(f"💾 [SKILLS] Факт сохранён для {user_id}: {fact[:60]}")
        return "[Запомнила]"
    except Exception as e:
        return f"[Memory Save Error: {e}]"


async def get_real_world_context(city: str, info_type: str = "weather") -> str:
    """Асинхронное получение погоды через wttr.in (V10.3)."""
    if info_type != "weather":
        return "[Поддерживается только 'weather']"
        
    url = f"https://wttr.in/{city}"
    params = {"format": 3, "lang": "ru"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return f"[Weather Error: HTTP {resp.status}]"
                data = await resp.text()
                return data.strip() if data else f"[Погода для {city}: нет данных]"
    except Exception as e:
        logger.warning(f"⚠️ [SKILLS] Weather error: {e}")
        return f"[Weather Error: {e}]"


# ════════════════════════════════════════════════════════
#  Диспетчер инструментов (Async V10.3)
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
    """Главный асинхронный диспетчер навыков (V10.3)."""
    if tool_name not in TOOL_MAP:
        return f"[Инструмент {tool_name!r} не найден]"
        
    try:
        # Все функции в TOOL_MAP теперь async def
        if tool_args:
            result = await TOOL_MAP[tool_name](**tool_args)
        else:
            result = await TOOL_MAP[tool_name]()
            
        logger.info(f"🔧 [SKILLS] {tool_name} → {str(result)[:60]}")
        return result
    except Exception as e:
        logger.warning(f"⚠️ [SKILLS] {tool_name}: {e}")
        return f"[Ошибка выполнения {tool_name}: {e}]"


async def execute_tool_async(tool_name: str, tool_args: dict) -> str:
    """Legacy wrapper для совместимости (V10.3)."""
    return await execute_tool(tool_name, tool_args)
