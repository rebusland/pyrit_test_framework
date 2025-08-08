import asyncio
import time
from functools import wraps
import logging
from typing import Any, Callable

from logging_handler import logger

def run_only_if_log_level_debug():
    return run_if_log_level_at_most(logging.DEBUG)

def run_if_log_level_at_most(level: int):
    """
    Decorator to run the function only if the current logger level is <= level.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger.isEnabledFor(level):
                return func(*args, **kwargs)
            return None
        return wrapper
    return decorator

def _format_duration(seconds: float) -> str:
    """Format seconds into H:MM:SS.sss."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = (seconds - int(seconds)) * 1000
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{int(millis):03d}"

def log_execution_time(return_time: bool = False, level: int = logging.INFO):
    """
    Decorator that logs (and optionally returns) the execution time of a function.
    Formats time as H:MM:SS.mmm.
    Works for both sync and async functions.
    """
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.log(level, f"{func.__name__} took {_format_duration(elapsed)}")
                return (result, elapsed) if return_time else result
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.log(level, f"{func.__name__} took {_format_duration(elapsed)}")
                return (result, elapsed) if return_time else result
            return wrapper
    return decorator