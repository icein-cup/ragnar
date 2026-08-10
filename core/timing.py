"""Small timing utilities for measuring code paths during development."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_elapsed(label: str, elapsed: float, **extra) -> None:
    """Log a wall-clock elapsed duration with optional key/value context."""
    parts = " ".join(f"{k}={v!r}" for k, v in extra.items()) if extra else ""
    suffix = f" ({parts})" if parts else ""
    logger.info("%s took %.2fs%s", label, elapsed, suffix)