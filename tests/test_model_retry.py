"""Tests for shared model invocation retry utilities."""

from __future__ import annotations

import pytest

from loom.models.base import ModelConnectionError
from loom.models.retry import (
    ModelRetryPolicy,
    build_model_retry_event_payload,
    call_with_model_retry,
    stream_with_model_retry,
)


class TestModelRetry:
    @pytest.mark.asyncio
    async def test_retries_failures_until_success(self):
        calls = {"count": 0}

        async def invoke():
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("synthetic failure")
            return "ok"

        result = await call_with_model_retry(
            invoke,
            policy=ModelRetryPolicy(
                max_attempts=5,
                base_delay_seconds=0.0,
                max_delay_seconds=0.0,
                jitter_seconds=0.0,
            ),
        )

        assert result == "ok"
        assert calls["count"] == 3

    @pytest.mark.asyncio
    async def test_exhausts_retry_queue_and_raises_last_error(self):
        calls = {"count": 0}

        async def invoke():
            calls["count"] += 1
            raise ValueError("always fails")

        with pytest.raises(ValueError, match="always fails"):
            await call_with_model_retry(
                invoke,
                policy=ModelRetryPolicy(
                    max_attempts=4,
                    base_delay_seconds=0.0,
                    max_delay_seconds=0.0,
                    jitter_seconds=0.0,
                ),
            )

        assert calls["count"] == 4

    @pytest.mark.asyncio
    async def test_custom_should_retry_can_stop_immediately(self):
        calls = {"count": 0}

        async def invoke():
            calls["count"] += 1
            raise RuntimeError("do not retry")

        with pytest.raises(RuntimeError, match="do not retry"):
            await call_with_model_retry(
                invoke,
                policy=ModelRetryPolicy(
                    max_attempts=6,
                    base_delay_seconds=0.0,
                    max_delay_seconds=0.0,
                    jitter_seconds=0.0,
                ),
                should_retry=lambda _error: False,
            )

        assert calls["count"] == 1

    @pytest.mark.asyncio
    async def test_stream_retries_when_failure_happens_before_first_chunk(self):
        calls = {"count": 0}

        async def invoke_stream():
            calls["count"] += 1
            if calls["count"] < 3:
                if False:  # pragma: no cover - keeps this an async generator
                    yield "never"
                raise RuntimeError("stream setup failed")
            yield "alpha"
            yield "beta"

        chunks: list[str] = []
        async for chunk in stream_with_model_retry(
            invoke_stream,
            policy=ModelRetryPolicy(
                max_attempts=5,
                base_delay_seconds=0.0,
                max_delay_seconds=0.0,
                jitter_seconds=0.0,
            ),
        ):
            chunks.append(chunk)

        assert chunks == ["alpha", "beta"]
        assert calls["count"] == 3

    @pytest.mark.asyncio
    async def test_stream_does_not_retry_after_partial_output(self):
        calls = {"count": 0}

        async def invoke_stream():
            calls["count"] += 1
            yield "partial"
            raise RuntimeError("stream failed mid-response")

        chunks: list[str] = []
        with pytest.raises(RuntimeError, match="stream failed mid-response"):
            async for chunk in stream_with_model_retry(
                invoke_stream,
                policy=ModelRetryPolicy(
                    max_attempts=4,
                    base_delay_seconds=0.0,
                    max_delay_seconds=0.0,
                    jitter_seconds=0.0,
                ),
            ):
                chunks.append(chunk)

        assert chunks == ["partial"]
        assert calls["count"] == 1

    @pytest.mark.asyncio
    async def test_backpressure_errors_extend_retry_budget_and_emit_schedule(
        self,
        monkeypatch,
    ):
        calls = {"count": 0}
        scheduled: list[tuple[int, int, int, float]] = []

        async def fake_sleep(_delay: float):
            return None

        monkeypatch.setattr("loom.models.retry.asyncio.sleep", fake_sleep)

        async def invoke():
            calls["count"] += 1
            if calls["count"] < 6:
                raise ModelConnectionError(
                    (
                        "Model server returned HTTP 429: "
                        '{"error":{"message":"The engine is currently overloaded, '
                        'please try again later","type":"engine_overloaded_error"}}'
                    ),
                )
            return "ok"

        def on_retry(
            attempt: int,
            max_attempts: int,
            _error: BaseException,
            remaining: int,
            delay_seconds: float,
        ) -> None:
            scheduled.append((attempt, max_attempts, remaining, delay_seconds))

        result = await call_with_model_retry(
            invoke,
            policy=ModelRetryPolicy(
                max_attempts=5,
                base_delay_seconds=0.0,
                max_delay_seconds=0.0,
                jitter_seconds=0.0,
            ),
            on_retry_scheduled=on_retry,
        )

        assert result == "ok"
        assert calls["count"] == 6
        assert scheduled[0] == (1, 8, 7, 2.0)

    def test_build_model_retry_event_payload_extracts_backpressure_fields(self):
        error = ModelConnectionError(
            (
                "Model server returned HTTP 429: "
                '{"error":{"message":"The engine is currently overloaded, '
                'please try again later","type":"engine_overloaded_error"}}'
            ),
        )

        payload = build_model_retry_event_payload(error, delay_seconds=8.0)

        assert payload["retry_scheduled"] is True
        assert payload["retry_delay_seconds"] == 8.0
        assert payload["retry_resume_at_ms"] > 0
        assert payload["http_status"] == 429
        assert payload["model_error_code"] == "engine_overloaded_error"
        assert payload["backpressure_error"] is True
        assert payload["overloaded_error"] is True
