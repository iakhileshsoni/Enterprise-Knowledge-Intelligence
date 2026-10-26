from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.config import get_settings

settings = get_settings()


def configure_logging() -> None:
    """Configure application-wide structured logging.

    Uses a simple text format in development and a JSON-friendly format in
    production.  Passwords, tokens, and document content must never appear in
    log messages — that responsibility lies with callers.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Remove any existing handlers to avoid duplicates when called multiple times
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if settings.is_production:
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "faiss", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Usage::

        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Document ingested", extra={"document_id": str(doc_id)})
    """
    return logging.getLogger(name)
