"""Structured logging setup."""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler


_configured = False


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    global _configured
    if _configured:
        return

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=verbose)],
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
