"""Shared retry primitives for model invocations."""

from __future__ import annotations

import asyncio
import random
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from loom.config import ExecutionConfig
from loom.models.base import ModelConnectionError

T = TypeVar("T")


@dataclass(frozen=True)
class ModelRetryPolicy:
    """Retry policy for model invocations."""

    max_attempts: int = 5
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.25

    @classmethod
    def from_execution_config(cls, execution: ExecutionConfig) -> ModelRetryPolicy:
        max_attempts = int(getattr(execution, "model_call_max_attempts", 5) or 5)
        base_delay = float(
            getattr(execution, "model_call_retry_base_delay_seconds", 0.5) or 0.5
        )
        max_delay = float(
            getattr(execution, "model_call_retry_max_delay_seconds", 8.0) or 8.0
        )
        jitter = float(
            getattr(execution, "model_call_retry_jitter_seconds", 0.25) or 0.25
        )
        max_attempts = max(1, min(10, max_attempts))
        base_delay = max(0.0, base_delay)
        max_delay = max(base_delay, max(0.0, max_delay))
        jitter = max(0.0, jitter)
        return cls(
            max_attempts=max_attempts,
            base_delay_seconds=base_delay,
            max_delay_seconds=max_delay,
            jitter_seconds=jitter,
        )


def is_retryable_model_error(error: BaseException) -> bool:
    """Return True when a model invocation error is likely transient."""
    if isinstance(error, ModelConnectionError):
        return True

    text = str(error or "").strip().lower()
    if not text:
        return False

    retry_markers = (
        "http ",
        "https://",
        "connection",
        "connect",
        "timeout",
        "timed out",
        "rate limit",
        "too many requests",
        "temporar",
        "unavailable",
    )
    return any(marker in text for marker in retry_markers)


async def call_with_model_retry(
    invoke: Callable[[], Awaitable[T]],
    *,
    policy: ModelRetryPolicy,
    should_retry: Callable[[BaseException], bool] | None = None,
    on_failure: Callable[[int, int, BaseException, int], None] | None = None,
) -> T:
    """Invoke an async model call with a queued retry policy."""
    decider = should_retry or _retry_all_failures
    attempts = deque(range(1, policy.max_attempts + 1))
    last_error: BaseException | None = None

    while attempts:
        attempt = attempts.popleft()
        try:
            return await invoke()
        except Exception as error:  # pragma: no cover - exercised by callers
            last_error = error
            remaining = len(attempts)
            retryable = decider(error)
            if on_failure is not None:
                on_failure(attempt, policy.max_attempts, error, remaining)
            if not retryable or remaining <= 0:
                raise
            delay = min(
                policy.max_delay_seconds,
                policy.base_delay_seconds * (2 ** (attempt - 1)),
            )
            if policy.jitter_seconds > 0:
                delay += random.uniform(0.0, policy.jitter_seconds)
            if delay > 0:
                await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("model retry queue exhausted without attempts")


def _retry_all_failures(error: BaseException) -> bool:
    """Default retry policy: retry any model-call failure except cancellations."""
    return not isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))


async def stream_with_model_retry(
    invoke_stream: Callable[[], AsyncGenerator[T, None]],
    *,
    policy: ModelRetryPolicy,
    should_retry: Callable[[BaseException], bool] | None = None,
    on_failure: Callable[[int, int, BaseException, int], None] | None = None,
) -> AsyncGenerator[T, None]:
    """Invoke an async model stream with queued retries on stream failures.

    Retries are attempted only when the stream fails before yielding any chunks.
    Once chunks are yielded, a retry would duplicate visible output.
    """
    decider = should_retry or _retry_all_failures
    attempts = deque(range(1, policy.max_attempts + 1))
    last_error: BaseException | None = None

    while attempts:
        attempt = attempts.popleft()
        yielded_chunk = False
        try:
            async for chunk in invoke_stream():
                yielded_chunk = True
                yield chunk
            return
        except Exception as error:  # pragma: no cover - exercised by callers
            last_error = error
            remaining = len(attempts)
            retryable = decider(error)
            if on_failure is not None:
                on_failure(attempt, policy.max_attempts, error, remaining)
            if yielded_chunk or not retryable or remaining <= 0:
                raise
            delay = min(
                policy.max_delay_seconds,
                policy.base_delay_seconds * (2 ** (attempt - 1)),
            )
            if policy.jitter_seconds > 0:
                delay += random.uniform(0.0, policy.jitter_seconds)
            if delay > 0:
                await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("model stream retry queue exhausted without attempts")
