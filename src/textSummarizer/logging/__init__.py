import os
import sys
import logging

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "running_logs.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logger = logging.getLogger("textSummarizerLogger")
logger.setLevel(logging.INFO)
logger.propagate = False   # IMPORTANT

# Avoid duplicate handlers
if not logger.handlers:

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter(logging_str))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(logging_str))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
