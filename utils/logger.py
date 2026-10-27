"""Structured logging with a rotating file handler.

Every module logs through :func:`get_logger` so the whole pipeline can be
tailed from ``logs/app.log`` during an incident or a support session.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.config import settings

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("resume_parser")
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    root.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if settings.log_to_file:
        try:
            Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                settings.log_dir / "app.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:  # read-only filesystem, e.g. some serverless hosts
            root.warning("Could not open log file; continuing with stderr only")

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring handlers on first use."""
    _configure_root()
    return logging.getLogger(f"resume_parser.{name}")
