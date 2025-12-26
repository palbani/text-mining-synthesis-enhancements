"""Logging utilities for synthesis enhancements."""

import logging
import sys
from typing import Optional
from pathlib import Path

_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "synthesis_enhancements",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up and configure a logger.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging output
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "synthesis_enhancements") -> logging.Logger:
    """Get or create a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]
