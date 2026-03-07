"""Event payload contract tests for semantic compactor model invocation."""

from __future__ import annotations

from typing import Any

import pytest

from loom.engine.semantic_compactor import model as compactor_model
from loom.models.base import ModelResponse


class _SuccessfulModel:
    name = "successful-compactor"

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        return ModelResponse(text='{"compressed_text":"ok"}')


class _FailingModel:
    name = "failing-compactor"

    async def complete(
        self,
        messages,
        tools=None,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        del messages, tools, temperature, max_tokens, response_format
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_invoke_compactor_model_emits_start_and_done_payload_fields() -> None:
    events: list[dict[str, Any]] = []

    def emit_model_event(*, model_name: str, phase: str, details: dict[str, Any]) -> None:
        events.append({"model_name": model_name, "phase": phase, **details})

    response = await compactor_model.invoke_compactor_model(
        model=_SuccessfulModel(),  # type: ignore[arg-type]
        system="compact",
        user="text",
        requested_max_chars=240,
        target_chars=180,
        hard_limit=220,
        max_tokens=512,
        temperature=0.4,
        label="contract-check",
        strict=True,
        response_format={"type": "json_object"},
        validation_attempt=2,
        emit_model_event=emit_model_event,
        should_retry=lambda _: False,
    )

    assert isinstance(response, ModelResponse)
    phases = [event["phase"] for event in events]
    assert phases == ["start", "done"]

    start_event = events[0]
    assert start_event["model_name"] == "successful-compactor"
    assert start_event["compactor_label"] == "contract-check"
    assert start_event["compactor_requested_max_chars"] == 240
    assert start_event["compactor_target_chars"] == 180
    assert start_event["compactor_hard_limit_chars"] == 220
    assert start_event["compactor_max_tokens"] == 512
    assert start_event["compactor_validation_attempt"] == 2
    assert "request_bytes" in start_event
    assert "request_est_tokens" in start_event

    done_event = events[1]
    assert done_event["model_name"] == "successful-compactor"
    assert done_event["compactor_label"] == "contract-check"
    assert done_event["compactor_response_chars"] > 0
    assert "response_preview" in done_event
    assert "response_finish_reason" in done_event


@pytest.mark.asyncio
async def test_invoke_compactor_model_emits_terminal_error_payload() -> None:
    events: list[dict[str, Any]] = []

    def emit_model_event(*, model_name: str, phase: str, details: dict[str, Any]) -> None:
        events.append({"model_name": model_name, "phase": phase, **details})

    result = await compactor_model.invoke_compactor_model(
        model=_FailingModel(),  # type: ignore[arg-type]
        system="compact",
        user="text",
        requested_max_chars=120,
        target_chars=90,
        hard_limit=110,
        max_tokens=256,
        temperature=0.2,
        label="error-check",
        strict=False,
        response_format=None,
        validation_attempt=1,
        emit_model_event=emit_model_event,
        should_retry=lambda _: False,
    )

    assert isinstance(result, RuntimeError)
    phases = [event["phase"] for event in events]
    assert phases == ["start", "done"]
    done_event = events[-1]
    assert done_event["error_type"] == "RuntimeError"
    assert "provider unavailable" in done_event["error"]
    assert done_event["compactor_response_chars"] == 0
