"""Focused tests for extracted runner memory helpers."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from loom.engine.runner import memory as runner_memory
from loom.engine.runner.types import SubtaskResult, ToolCallRecord
from loom.models.base import ModelResponse
from loom.tools.registry import ToolResult


def test_parse_memory_entries_accepts_json_array_and_normalizes_type() -> None:
    runner = SimpleNamespace(
        _validator=SimpleNamespace(
            validate_json_response=lambda *args, **kwargs: SimpleNamespace(
                valid=False,
                parsed=None,
            ),
        ),
    )
    response = ModelResponse(
        text=(
            '[{"type":"decision","summary":"s1","detail":"d1","tags":"t1"},'
            '{"type":"unknown","summary":"s2","detail":"d2","tags":"t2"}]'
        ),
    )

    entries = runner_memory.parse_memory_entries(
        runner,
        response,
        task_id="task-1",
        subtask_id="subtask-1",
    )

    assert len(entries) == 2
    assert entries[0].entry_type == "decision"
    assert entries[1].entry_type == "discovery"


def test_parse_memory_entries_falls_back_to_validator_entries() -> None:
    runner = SimpleNamespace(
        _validator=SimpleNamespace(
            validate_json_response=lambda *args, **kwargs: SimpleNamespace(
                valid=True,
                parsed={
                    "entries": [
                        {"type": "artifact", "summary": "found", "detail": "detail", "tags": "a,b"},
                    ],
                },
            ),
        ),
    )
    response = ModelResponse(text="not json")

    entries = runner_memory.parse_memory_entries(
        runner,
        response,
        task_id="task-1",
        subtask_id="subtask-1",
    )

    assert len(entries) == 1
    assert entries[0].entry_type == "artifact"
    assert entries[0].summary == "found"


def test_spawn_memory_extraction_skips_when_timeout_guard_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = SimpleNamespace(
        EXTRACTOR_TIMEOUT_GUARD_SECONDS=20,
        _extractor_timeout_guard_seconds=10,
        _remaining_subtask_seconds=lambda: 5.0,
        _extract_memory=AsyncMock(),
    )

    def _unexpected_running_loop():
        raise AssertionError("get_running_loop should not be called when guard blocks")

    monkeypatch.setattr(runner_memory.asyncio, "get_running_loop", _unexpected_running_loop)

    runner_memory.spawn_memory_extraction(
        runner,
        task_id="task-1",
        subtask_id="subtask-1",
        result=SubtaskResult(),
        logger=logging.getLogger("test-runner-memory"),
    )

    assert runner._extract_memory.call_count == 0


@pytest.mark.asyncio
async def test_spawn_memory_extraction_schedules_background_task() -> None:
    runner = SimpleNamespace(
        EXTRACTOR_TIMEOUT_GUARD_SECONDS=20,
        _extractor_timeout_guard_seconds=10,
        _remaining_subtask_seconds=lambda: 120.0,
        _extract_memory=AsyncMock(return_value=None),
    )

    runner_memory.spawn_memory_extraction(
        runner,
        task_id="task-1",
        subtask_id="subtask-1",
        result=SubtaskResult(),
        logger=logging.getLogger("test-runner-memory"),
    )
    await asyncio.sleep(0)

    runner._extract_memory.assert_awaited_once_with("task-1", "subtask-1", SubtaskResult())


@pytest.mark.asyncio
async def test_extract_memory_tightens_prompt_and_emits_compaction_details() -> None:
    class _ExtractorModel:
        name = "extractor-model"

        def __init__(self) -> None:
            self.messages: list[list[dict]] = []

        async def complete(self, messages):
            self.messages.append(messages)
            return ModelResponse(text="[]")

    class _RunnerStub:
        EXTRACTOR_TOOL_ARGS_MAX_CHARS = 260
        EXTRACTOR_TOOL_TRACE_MAX_CHARS = 3600
        EXTRACTOR_PROMPT_MAX_CHARS = 9000

        def __init__(self) -> None:
            self._extractor_tool_args_max_chars = 260
            self._extractor_tool_trace_max_chars = 3600
            self._extractor_prompt_max_chars = 260
            self._router = SimpleNamespace(select=lambda **kwargs: extractor_model)
            self._prompts = SimpleNamespace(
                build_extractor_prompt=lambda subtask_id, tool_calls_formatted, model_output: (
                    f"{'H' * 300}\n{subtask_id}\n"
                    f"TOOLS:{tool_calls_formatted}\nOUTPUT:{model_output}"
                ),
            )
            self._config = SimpleNamespace(execution=SimpleNamespace())
            self._memory = SimpleNamespace(store_many=AsyncMock())
            self._events: list[dict] = []
            self.compact_calls: list[str] = []

        async def _summarize_tool_call_arguments(self, args, *, max_chars, label):
            del max_chars, label
            return {"path": args.get("path", ""), "content": "<compacted>"}

        async def _compact_text(self, text, *, max_chars, label):
            del text, max_chars
            self.compact_calls.append(label)
            return f"[compacted:{label}]"

        def _estimate_message_tokens(self, messages):
            return len(messages[0]["content"]) // 10

        def _emit_model_event(self, **payload):
            self._events.append(payload)

        def _parse_memory_entries(self, response, task_id, subtask_id):
            del response, task_id, subtask_id
            return []

    extractor_model = _ExtractorModel()
    runner = _RunnerStub()
    result = SubtaskResult(
        summary="X" * 4000,
        tool_calls=[
            ToolCallRecord(
                tool="document_write",
                args={"path": "report.md", "content": "A" * 4000},
                result=ToolResult.ok("ok"),
            ),
        ],
    )

    await runner_memory.extract_memory(
        runner,
        task_id="task-1",
        subtask_id="subtask-1",
        result=result,
        logger=logging.getLogger("test-runner-memory"),
    )

    assert extractor_model.messages
    prompt = extractor_model.messages[0][0]["content"]
    assert "[compacted:memory extractor model output]" in prompt
    assert "[compacted:memory extractor tool trace strict]" in prompt
    assert "memory extractor model output" in runner.compact_calls
    assert "memory extractor tool trace strict" in runner.compact_calls
    assert runner._events
    start_event = runner._events[0]
    assert start_event["phase"] == "start"
    compacted_fields = start_event["details"]["extractor_compacted_fields"]
    assert "model_output" in compacted_fields
    assert "tool_trace" in compacted_fields
    assert "tool_args" in compacted_fields
