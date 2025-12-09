"""
tb_logging.py

Shared logging utilities for the TB project.

Goals:
- Provide a single helper to get a logger that logs
  both to a file (per-run) and to stdout.
- Avoid duplicate handlers when called multiple times.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def _find_file_handler_for(logger: logging.Logger, log_file: Path) -> Optional[logging.FileHandler]:
    """Return an existing FileHandler for log_file if present on logger."""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                handler_path = Path(handler.baseFilename)
            except Exception:
                continue
            if handler_path.resolve() == log_file.resolve():
                return handler
    return None


def _find_stdout_stream_handler(logger: logging.Logger) -> Optional[logging.StreamHandler]:
    """
    Return an existing StreamHandler that we previously marked as stdout handler.
    We mark it via a custom attribute _tb_is_stdout to avoid guessing.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "_tb_is_stdout", False):
            return handler
    return None


def get_tb_logger(name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Get a TB logger with a file + stdout handler.

    Args:
        name: Logger name (e.g. "tb.run", "tb.backtest").
        log_file: Path to the log file (per-run file, e.g. D:\\TB_DATA\\runs\\<run_id>\\tb.log).
        level: Logging level (default: logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # do not bubble up to root logger

    # Ensure directory for log_file exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # --- File handler (per-run) ---
    file_handler = _find_file_handler_for(logger, log_file)
    if file_handler is None:
        file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        logger.addHandler(file_handler)

    # --- Stream handler (stdout) ---
    stream_handler = _find_stdout_stream_handler(logger)
    if stream_handler is None:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        # Mark this handler as the dedicated stdout handler for TB
        stream_handler._tb_is_stdout = True  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)

    # --- Common formatter ---
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    return logger
