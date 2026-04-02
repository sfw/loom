"""Shared retry primitives for model invocations."""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeVar

from loom.config import ExecutionConfig
from loom.models.base import ModelConnectionError

T = TypeVar("T")

_HTTP_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b", flags=re.IGNORECASE)
_JSON_ERROR_CODE_RE = re.compile(
    r'"(?:type|code)"\s*:\s*"([^"]+)"',
    flags=re.IGNORECASE,
)
_RETRY_AFTER_RE = re.compile(
    r"retry[- ]after[:=)\s]*([0-9]+(?:\.[0-9]+)?)",
    flags=re.IGNORECASE,
)
_MODEL_BACKPRESSURE_STATUS_CODES = frozenset({429, 503, 529})
_MODEL_BACKPRESSURE_MARKERS = (
    "rate limit",
    "rate-limited",
    "too many requests",
    "overloaded",
    "overload",
    "throttled",
    "throttle",
    "try again later",
)
_MODEL_OVERLOAD_MARKERS = (
    "engine_overloaded_error",
    "currently overloaded",
    "engine is currently overloaded",
    "server overloaded",
    "temporarily overloaded",
)
_MODEL_BACKPRESSURE_RETRY_BASE_DELAY_SECONDS = 2.0
_MODEL_BACKPRESSURE_MIN_ATTEMPTS = 8


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


def extract_model_error_status_code(error: BaseException) -> int | None:
    """Extract an HTTP status code from a wrapped model error when available."""
    response = _model_error_response(error)
    if response is not None:
        status = getattr(response, "status_code", None)
        try:
            if status is not None:
                return int(status)
        except (TypeError, ValueError):
            pass

    match = _HTTP_STATUS_RE.search(str(error or ""))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def extract_model_retry_after_seconds(error: BaseException) -> float | None:
    """Extract a retry-after hint from wrapped model errors when present."""
    response = _model_error_response(error)
    if response is not None:
        headers = getattr(response, "headers", None)
        if hasattr(headers, "get"):
            parsed = _parse_retry_after_seconds(headers.get("retry-after", ""))
            if parsed is not None:
                return parsed

    return _parse_retry_after_seconds(str(error or ""))


def extract_model_error_code(error: BaseException) -> str:
    """Extract a provider error code/type from wrapped model errors when possible."""
    text = str(error or "").strip()
    if not text:
        return ""

    for payload in _error_payloads_from_text(text):
        raw = payload.get("error", payload)
        if not isinstance(raw, dict):
            continue
        for key in ("type", "code", "error_type"):
            code = str(raw.get(key, "") or "").strip()
            if code:
                return code

    match = _JSON_ERROR_CODE_RE.search(text)
    if match is None:
        return ""
    return str(match.group(1) or "").strip()


def is_model_backpressure_error(error: BaseException) -> bool:
    """Return True when the provider is asking us to slow down and retry."""
    status = extract_model_error_status_code(error)
    if status in _MODEL_BACKPRESSURE_STATUS_CODES:
        return True

    error_code = extract_model_error_code(error).strip().lower()
    if error_code and any(marker in error_code for marker in (
        "overload",
        "rate_limit",
        "rate-limit",
        "too_many_requests",
        "too-many-requests",
        "throttle",
    )):
        return True

    text = str(error or "").strip().lower()
    return any(marker in text for marker in _MODEL_BACKPRESSURE_MARKERS)


def is_model_overloaded_error(error: BaseException) -> bool:
    """Return True when the provider explicitly reports an overloaded engine."""
    error_code = extract_model_error_code(error).strip().lower()
    if error_code and "overload" in error_code:
        return True
    text = str(error or "").strip().lower()
    return any(marker in text for marker in _MODEL_OVERLOAD_MARKERS)


def build_model_retry_event_payload(
    error: BaseException,
    *,
    delay_seconds: float,
) -> dict[str, object]:
    """Build structured retry metadata for telemetry/UI surfaces."""
    safe_delay = max(0.0, float(delay_seconds))
    payload: dict[str, object] = {
        "retry_scheduled": True,
        "retry_delay_seconds": round(safe_delay, 3),
        "retry_resume_at_ms": int(
            (datetime.now(UTC).timestamp() + safe_delay) * 1000,
        ),
    }
    status = extract_model_error_status_code(error)
    if status is not None:
        payload["http_status"] = status
    retry_after = extract_model_retry_after_seconds(error)
    if retry_after is not None:
        payload["retry_after_seconds"] = round(retry_after, 3)
    error_code = extract_model_error_code(error)
    if error_code:
        payload["model_error_code"] = error_code
    if is_model_backpressure_error(error):
        payload["backpressure_error"] = True
    if is_model_overloaded_error(error):
        payload["overloaded_error"] = True
    return payload


def compute_model_retry_delay_seconds(
    error: BaseException,
    *,
    attempt: int,
    policy: ModelRetryPolicy,
) -> float:
    """Compute the retry delay for a failed model invocation."""
    delay = min(
        policy.max_delay_seconds,
        policy.base_delay_seconds * (2 ** (attempt - 1)),
    )
    retry_after = extract_model_retry_after_seconds(error)
    if retry_after is not None:
        delay = max(delay, retry_after)
    elif is_model_backpressure_error(error):
        delay = max(
            delay,
            min(
                _MODEL_BACKPRESSURE_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
                max(
                    policy.max_delay_seconds,
                    _MODEL_BACKPRESSURE_RETRY_BASE_DELAY_SECONDS,
                ),
            ),
        )
    if policy.jitter_seconds > 0:
        delay += random.uniform(0.0, policy.jitter_seconds)
    return max(0.0, delay)


async def call_with_model_retry(
    invoke: Callable[[], Awaitable[T]],
    *,
    policy: ModelRetryPolicy,
    should_retry: Callable[[BaseException], bool] | None = None,
    on_failure: Callable[[int, int, BaseException, int], None] | None = None,
    on_retry_scheduled: (
        Callable[[int, int, BaseException, int, float], None] | None
    ) = None,
) -> T:
    """Invoke an async model call with a queued retry policy."""
    decider = should_retry or _retry_all_failures
    last_error: BaseException | None = None
    max_attempts = int(max(1, policy.max_attempts))
    attempt = 0
    backpressure_retry_extended = False

    while attempt < max_attempts:
        attempt += 1
        try:
            return await invoke()
        except Exception as error:  # pragma: no cover - exercised by callers
            last_error = error
            if (
                is_model_backpressure_error(error)
                and not backpressure_retry_extended
            ):
                max_attempts = max(
                    max_attempts,
                    _MODEL_BACKPRESSURE_MIN_ATTEMPTS,
                )
                backpressure_retry_extended = True
            remaining = max(0, max_attempts - attempt)
            retryable = decider(error)
            if on_failure is not None:
                on_failure(attempt, max_attempts, error, remaining)
            if not retryable or remaining <= 0:
                raise
            delay = compute_model_retry_delay_seconds(
                error,
                attempt=attempt,
                policy=policy,
            )
            if on_retry_scheduled is not None:
                on_retry_scheduled(attempt, max_attempts, error, remaining, delay)
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
    on_retry_scheduled: (
        Callable[[int, int, BaseException, int, float], None] | None
    ) = None,
) -> AsyncGenerator[T, None]:
    """Invoke an async model stream with queued retries on stream failures.

    Retries are attempted only when the stream fails before yielding any chunks.
    Once chunks are yielded, a retry would duplicate visible output.
    """
    decider = should_retry or _retry_all_failures
    last_error: BaseException | None = None
    max_attempts = int(max(1, policy.max_attempts))
    attempt = 0
    backpressure_retry_extended = False

    while attempt < max_attempts:
        attempt += 1
        yielded_chunk = False
        try:
            async for chunk in invoke_stream():
                yielded_chunk = True
                yield chunk
            return
        except Exception as error:  # pragma: no cover - exercised by callers
            last_error = error
            if (
                is_model_backpressure_error(error)
                and not backpressure_retry_extended
            ):
                max_attempts = max(
                    max_attempts,
                    _MODEL_BACKPRESSURE_MIN_ATTEMPTS,
                )
                backpressure_retry_extended = True
            remaining = max(0, max_attempts - attempt)
            retryable = decider(error)
            if on_failure is not None:
                on_failure(attempt, max_attempts, error, remaining)
            if yielded_chunk or not retryable or remaining <= 0:
                raise
            delay = compute_model_retry_delay_seconds(
                error,
                attempt=attempt,
                policy=policy,
            )
            if on_retry_scheduled is not None:
                on_retry_scheduled(attempt, max_attempts, error, remaining, delay)
            if delay > 0:
                await asyncio.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("model stream retry queue exhausted without attempts")


def _model_error_response(error: BaseException) -> object | None:
    if not isinstance(error, ModelConnectionError):
        return None
    original = getattr(error, "original", None)
    return getattr(original, "response", None)


def _parse_retry_after_seconds(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        match = _RETRY_AFTER_RE.search(text)
        if match is None:
            return None
        try:
            parsed = float(match.group(1))
        except (TypeError, ValueError):
            return None
    if parsed < 0:
        return None
    return parsed


def _error_payloads_from_text(text: str) -> list[dict]:
    payloads: list[dict] = []
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads
