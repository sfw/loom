"""Low-overhead latency diagnostics helpers."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

_TRUTHY = {"1", "true", "yes", "on"}


def diagnostics_enabled() -> bool:
    """Return whether latency diagnostics are enabled for this process."""
    raw = os.environ.get("LOOM_LATENCY_DIAGNOSTICS", "")
    return raw.strip().lower() in _TRUTHY


def log_latency_event(
    logger: logging.Logger,
    *,
    event: str,
    duration_seconds: float,
    fields: dict[str, Any] | None = None,
) -> None:
    """Emit one structured-ish latency line when diagnostics are enabled."""
    if not diagnostics_enabled():
        return
    duration_ms = max(0.0, float(duration_seconds)) * 1000.0
    payload = ""
    if fields:
        parts = [f"{key}={value}" for key, value in fields.items()]
        payload = " " + " ".join(parts)
    logger.info("latency event=%s duration_ms=%.2f%s", event, duration_ms, payload)


@contextmanager
def timed_block(
    logger: logging.Logger,
    *,
    event: str,
    fields: dict[str, Any] | None = None,
    sink: Callable[[float], None] | None = None,
):
    """Context manager that logs elapsed duration for an operation."""
    started = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        if sink is not None:
            sink(elapsed)
        log_latency_event(
            logger,
            event=event,
            duration_seconds=elapsed,
            fields=fields,
        )
