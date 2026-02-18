"""Performance timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Generator

from .logging import get_logger

logger = get_logger(__name__)


@contextmanager
def timed(label: str) -> Generator[None, None, None]:
    """Context manager that logs elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{label}: {elapsed:.2f}s")


def timed_func(func: Callable) -> Callable:
    """Decorator that logs function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__}: {elapsed:.2f}s")
        return result

    return wrapper
