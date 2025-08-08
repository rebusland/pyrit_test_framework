from functools import wraps
import logging

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