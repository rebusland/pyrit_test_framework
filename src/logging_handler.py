import logging
from typing import Any, Sequence

from config_handler import load_test_config

# TODO get severity level from config file
LOG_LEVEL = getattr(logging, load_test_config()['logging']['level'].upper(), logging.INFO)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug(f"Logger level set to {logging.getLevelName(LOG_LEVEL)}")

from utils import run_only_if_log_level_debug
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
