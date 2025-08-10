import logging

# Default log filename
DEFAULT_LOG_FILENAME = "training_log.txt"

# Configure logger once
logger = logging.getLogger()  # Get root logger
logger.setLevel(logging.INFO)  # Set root logger level

logfmt_str = "%(asctime)s %(name)s:%(lineno)03d: %(message)s"
formatter = logging.Formatter(logfmt_str, datefmt="%Y-%m-%d %H:%M:%S")

handler = logging.FileHandler(DEFAULT_LOG_FILENAME, mode="w")
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger.addHandler(handler)
