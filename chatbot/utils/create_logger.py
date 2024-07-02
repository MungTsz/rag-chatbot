import logging
from datetime import datetime


def get_logger(name, log_file_path=None):
    """
    Creates a logger with the specified name, log level, and optional log file.

    Args:
        name (str): The name of the logger.
        level (str, optional): The log level. Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'. Defaults to 'INFO'.
        log_file (str, optional): The path to the log file. If not provided, logs will be printed to the console.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create handlers
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
