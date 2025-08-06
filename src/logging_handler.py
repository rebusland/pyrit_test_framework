import logging

# TODO get severity level from config file
LOG_LEVEL = logging.DEBUG

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
