import time
import asyncio
from functools import wraps
import logging
from typing import Any, Callable, Sequence

from config_handler import load_test_config
from utils import format_duration

# TODO get severity level from config file
LOG_LEVEL = getattr(logging, load_test_config()['logging']['level'].upper(), logging.INFO)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug(f"Logger level set to {logging.getLevelName(LOG_LEVEL)}")

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

@run_only_if_log_level_debug()
def peek_iterable(
    *,
    iterable: Sequence[Any],
    header: str,
    element_description: str,
    stringifyier = lambda el : str(el)
) -> None:
    logger.debug(f"\n\n\n ****** {header} ******\n\n")
    for el in iterable:
        logger.debug(f"{element_description}: {stringifyier(el)}")

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
                logger.log(level, f"{func.__name__} took {format_duration(elapsed)}")
                return (result, elapsed) if return_time else result
            return wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.log(level, f"{func.__name__} took {format_duration(elapsed)}")
                return (result, elapsed) if return_time else result
            return wrapper
    return decorator
