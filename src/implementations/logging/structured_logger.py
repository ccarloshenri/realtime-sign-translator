"""
Structured logger adapter.

Exposes the ILogger contract on top of stdlib `logging`. Extra kwargs passed
to each method are serialized as `key=value` pairs appended to the message,
which keeps the output greppable without pulling in a heavier logging
framework.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from src.implementations.config.yaml_configuration import LoggingConfig

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def build_logger(name: str, config: LoggingConfig) -> "StructuredLogger":
    logger = logging.getLogger(name)
    logger.setLevel(config.level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(_FORMAT)

    if config.console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.file:
        path = Path(config.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return StructuredLogger(logger)


class StructuredLogger:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(self._format(msg, kwargs))

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(self._format(msg, kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(self._format(msg, kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(self._format(msg, kwargs))

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._logger.exception(self._format(msg, kwargs))

    @staticmethod
    def _format(msg: str, kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return msg
        pairs = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
        return f"{msg} | {pairs}"
