import logging
from typing import Any, Sequence

from utils import run_only_if_log_level_debug

# TODO get severity level from config file
LOG_LEVEL = logging.DEBUG

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

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
