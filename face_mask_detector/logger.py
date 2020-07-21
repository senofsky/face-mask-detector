"""Provides logging capabilities
"""
import logging


def _generate_log_handler(log_level: int, log_format: str) -> logging.Handler:
    """Generates a log handler given the log level and log format
    """
    log_handler = logging.StreamHandler()
    log_handler.setLevel(log_level)

    formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(formatter)

    return log_handler


def generate_logger(logger_name, verbosity: int) -> logging.Logger:
    """Configures the log levels and log formats given the verbosity
    """
    if verbosity == 0:
        log_level = logging.WARNING
        log_format = "%(levelname)s:%(message)s"

    elif verbosity == 1:
        log_level = logging.INFO
        log_format = "%(levelname)s:%(message)s"

    else:
        log_level = logging.DEBUG
        log_format = "%(asctime)s:%(levelname)s:%(module)s:%(funcName)s%(message)s"

    log_handler = _generate_log_handler(log_level, log_format)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(log_handler)

    return logger
