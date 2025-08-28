from functools import wraps
from typing import Any, Callable
import tiktoken

def format_duration(seconds: float) -> str:
    """Format seconds into H:MM:SS.sss."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = (seconds - int(seconds)) * 1000
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{int(millis):03d}"

def num_tokens_for_model(text: str, model_name: str) -> int:
    # GPT-4o should use o200k_base encoding
    try:
        model_name = model_name or 'cl100k_base' # fallback encoding if nothing provided
        enc = tiktoken.encoding_for_model(model_name=model_name)
    except KeyError:
        from logging_handler import logger
        logger.warning(f"Cannot find enconding for given model {model_name}, falling back to a default encoding")
        enc = tiktoken.get_encoding("cl100k_base")  # fallback encoding
    return len(enc.encode(text))
