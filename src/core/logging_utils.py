"""
Unified logging configuration for TraceGen.

Goal:
- Make Loguru + stdlib `logging` output consistent.
- Ensure modules that still use stdlib logging are formatted the same way.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from loguru import logger


UNIFIED_LOGURU_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except Exception:
            level = record.levelno

        # Find the caller frame so Loguru reports correct origin.
        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(
    *,
    verbose: bool,
    stream: Optional[object] = None,
) -> None:
    """
    Configure logging for both Loguru and stdlib logging.

    Call this once early in the entrypoint (after Hydra config is available).
    """
    logger.remove()
    logger.add(
        stream or sys.stderr,
        level="INFO" if verbose else "WARNING",
        format=UNIFIED_LOGURU_FORMAT,
    )

    # Route stdlib logging to Loguru (Hydra + third-party libs included).
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(_InterceptHandler())
    root.setLevel(logging.INFO if verbose else logging.WARNING)

    # Avoid overly chatty third-party logs.
    for name in ("httpx", "openai", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

