"""Lightweight logger (rich if available, else stdlib)."""
from __future__ import annotations

import logging


def get_logger(name: str = "star", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    try:
        from rich.logging import RichHandler

        handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
