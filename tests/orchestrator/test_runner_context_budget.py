"""Orchestrator subtask runner context budget tests."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config
from loom.engine.orchestrator import SubtaskResult, ToolCallRecord
from loom.events.bus import EventBus
from loom.events.types import (
    ARTIFACT_CONFINEMENT_VIOLATION,
    ARTIFACT_INGEST_CLASSIFIED,
    ARTIFACT_INGEST_COMPLETED,
    ARTIFACT_READ_COMPLETED,
    ARTIFACT_RETENTION_PRUNED,
    COMPACTION_POLICY_DECISION,
    OVERFLOW_FALLBACK_APPLIED,
    TOOL_CALL_COMPLETED,
)
from loom.models.base import ModelResponse, TokenUsage
from loom.state.task_state import Subtask
from loom.tools.registry import ToolResult
from tests.orchestrator.conftest import _make_task


class TestSubtaskRunnerContextBudget:
    class _FakeCompactor:
        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            value = str(text or "")
            if len(value) <= max_chars:
                return value
            words = value.split()
            if not words:
                return value
            if len(words) == 1 and len(words[0]) > max_chars:
                return f"[compacted {len(value)} chars]"
            compacted = ""
            for word in words:
                candidate = f"{compacted} {word}".strip()
                if compacted and len(candidate) > max_chars:
                    break
                compacted = candidate
            return compacted or value

    class _RecordingCompactor:
        def __init__(self):
            self.calls: list[tuple[str, int, int]] = []

        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            value = str(text or "")
            self.calls.append((str(label), int(max_chars), len(value)))
            if len(value) <= max_chars:
                return value
            words = value.split()
            if not words:
                return value
            compacted = ""
            for word in words:
                candidate = f"{compacted} {word}".strip()
                if compacted and len(candidate) > max_chars:
                    break
                compacted = candidate
            return compacted or value

    @staticmethod
    def _make_runner_for_compaction():
        from loom.engine.runner import SubtaskRunner

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._compactor = TestSubtaskRunnerContextBudget._FakeCompactor()
        runner._runner_compaction_policy_mode = "legacy"
        runner._max_model_context_tokens = SubtaskRunner.MAX_MODEL_CONTEXT_TOKENS
        return runner

    @staticmethod
    def _make_runner_for_tiered_compaction(*, context_budget: int = 2500):
        from loom.engine.runner import SubtaskRunner

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._compactor = TestSubtaskRunnerContextBudget._RecordingCompactor()
        runner._runner_compaction_policy_mode = "tiered"
        runner._max_model_context_tokens = context_budget
        runner._compaction_pressure_ratio_soft = 0.70
        runner._compaction_pressure_ratio_hard = 0.92
        runner._preserve_recent_critical_messages = 4
        runner._compact_tool_call_argument_chars = 160
        runner._compact_tool_result_output_chars = 180
        runner._compact_text_output_chars = 220
        runner._minimal_text_output_chars = 120
        runner._compaction_timeout_guard_seconds = 25
        runner._compaction_no_gain_min_delta_chars = 4
        runner._compaction_no_gain_attempt_limit = 2
        runner._compaction_churn_warning_calls = 100
        runner._extractor_tool_args_max_chars = 180
        runner._extractor_tool_trace_max_chars = 1800
        runner._extractor_prompt_max_chars = 2600
        runner._enable_model_overflow_fallback = True
        runner._overflow_fallback_tool_message_min_chars = 500
        runner._overflow_fallback_tool_output_excerpt_chars = 220
        return runner

    @staticmethod
    def _make_runner_for_telemetry():
        from loom.engine.runner import SubtaskRunner

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._event_bus = EventBus()
        runner._enable_artifact_telemetry_events = True
        runner._artifact_telemetry_max_metadata_chars = 120
        runner._runner_compaction_policy_mode = "tiered"
        runner._max_model_context_tokens = 2400
        runner._last_compaction_diagnostics = {
            "compaction_policy_mode": "tiered",
            "compaction_pressure_ratio": 0.91,
            "compaction_stage": "stage_2_tool_outputs",
            "compaction_skipped_reason": "",
        }
        runner._active_subtask_telemetry_counters = (
            SubtaskRunner._new_subtask_telemetry_counters()
        )
        return runner

    class _NoopCompactor:
        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            return str(text or "")

    def test_detects_model_overflow_error_markers(self):
        from loom.engine.runner import SubtaskRunner

        assert SubtaskRunner._is_model_request_overflow_error(
            "Invalid request: total message size 123 exceeds limit 99",
        )
        assert SubtaskRunner._is_model_request_overflow_error(
            "Invalid request: Your request exceeded model token limit: 262144",
        )
        assert not SubtaskRunner._is_model_request_overflow_error(
            "Model server returned HTTP 522: upstream timeout",
        )

    def test_overflow_fallback_rewrites_older_tool_messages_and_preserves_latest(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=1200)
        tool_payload = json.dumps({
            "success": True,
            "output": "A" * 12_000,
            "error": None,
            "files_changed": [],
            "data": {
                "content_kind": "pdf",
                "artifact_ref": "af_123",
                "size_bytes": 2_000_000,
                "url": "https://example.com/report.pdf",
            },
        })
        latest_payload = json.dumps({
            "success": True,
            "output": "latest short output",
            "error": None,
            "files_changed": [],
            "data": {"content_kind": "pdf", "artifact_ref": "af_latest"},
        })
        messages = [
            {"role": "user", "content": "Goal: analyze report."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://example.com/a.pdf\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_old", "content": tool_payload},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_latest",
                    "type": "function",
                    "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://example.com/b.pdf\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_latest", "content": latest_payload},
        ]

        rewritten, report = runner._apply_model_overflow_fallback(messages)

        assert report["overflow_fallback_applied"] is True
        assert report["overflow_fallback_rewritten_messages"] == 1
        assert report["overflow_fallback_chars_reduced"] > 0
        rewritten_old = json.loads(rewritten[2]["content"])
        assert "overflow fallback applied" in rewritten_old["output"]
        # Latest tool result is preserved verbatim.
        assert rewritten[4]["content"] == latest_payload

    @pytest.mark.asyncio
    async def test_serialize_tool_result_for_model_compacts_output_and_data(self):
        runner = self._make_runner_for_compaction()

        result = ToolResult.ok(
            "x" * 20_000,
            data={
                "url": "https://example.com/really/long/path",
                "nested": {"a": 1, "b": 2},
                "results": [1, 2, 3],
            },
            files_changed=["report.md"],
        )

        payload = await runner._serialize_tool_result_for_model("web_fetch", result)
        parsed = json.loads(payload)

        assert parsed["success"] is True
        assert len(parsed["output"]) < len(result.output)
        assert parsed["files_changed"] == ["report.md"]
        assert parsed["data"]["url"].startswith("https://example.com/")
        assert "a" in parsed["data"]["nested"]
        assert "1" in parsed["data"]["results"]

    @pytest.mark.asyncio
    async def test_compact_messages_for_model_keeps_structure_and_reduces_tokens(self):
        runner = self._make_runner_for_compaction()

        messages = [{"role": "user", "content": "Goal: perform market research"}]
        for idx in range(24):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": "web_fetch",
                        "arguments": "{\"url\": \"https://example.com\"}",
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{idx}",
                "content": json.dumps({
                    "success": True,
                    "output": "A" * 8_000,
                    "error": None,
                    "files_changed": [],
                }),
            })
            messages.append({
                "role": "user",
                "content": (
                    "CURRENT TASK STATE:\n"
                    "Goal: market research\n"
                    "Current subtask: analyze\n"
                    "Do NOT move to the next subtask."
                ),
            })

        before = runner._estimate_message_tokens(messages)
        compacted = await runner._compact_messages_for_model(messages)
        after = runner._estimate_message_tokens(compacted)

        assert after < before
        assert after <= runner.MAX_MODEL_CONTEXT_TOKENS
        assert compacted[0]["role"] == "user"
        assert any(
            isinstance(msg, dict) and msg.get("role") == "tool"
            and msg.get("tool_call_id") == "call_23"
            for msg in compacted
        )

    @pytest.mark.asyncio
    async def test_summarize_model_output_uses_semantic_compaction(self):
        runner = self._make_runner_for_compaction()

        text = (
            "First sentence is complete. "
            "Second sentence is also complete. "
            "Third sentence should be cut near boundary."
        )
        summary = await runner._summarize_model_output(
            text,
            max_chars=60,
            label="test summary",
        )

        assert "First sentence is complete." in summary
        assert len(summary) <= 60

    @pytest.mark.asyncio
    async def test_compact_text_keeps_oversize_compactor_output(self):
        runner = self._make_runner_for_compaction()
        runner._compactor = self._NoopCompactor()

        value = "A" * 5000
        compacted = await runner._compact_text(
            value,
            max_chars=120,
            label="oversize guard",
        )

        assert compacted == value

    @pytest.mark.asyncio
    async def test_compacts_recent_assistant_tool_call_arguments(self):
        runner = self._make_runner_for_compaction()
        huge_args = json.dumps({
            "path": "report.md",
            "content": "A" * 400_000,
        })
        messages = [
            {"role": "user", "content": "Goal: write report"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_recent",
                    "type": "function",
                    "function": {
                        "name": "document_write",
                        "arguments": huge_args,
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_recent",
                "content": json.dumps({
                    "success": True,
                    "output": "ok",
                    "error": None,
                    "files_changed": ["report.md"],
                }),
            },
            {
                "role": "user",
                "content": (
                    "CURRENT TASK STATE:\nGoal: report\nCurrent subtask: s1\n"
                    "Do NOT move to next subtask"
                ),
            },
        ]

        compacted = await runner._compact_messages_for_model(messages)
        assistant = next(
            msg for msg in compacted
            if isinstance(msg, dict) and msg.get("role") == "assistant"
        )
        args_text = (
            assistant.get("tool_calls", [{}])[0]
            .get("function", {})
            .get("arguments", "")
        )
        assert len(args_text) <= 500
        assert "A" * 200 not in args_text

    @pytest.mark.asyncio
    async def test_no_compaction_when_under_budget(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=12_000)
        messages = [
            {"role": "user", "content": "Goal: summarize file structure."},
            {"role": "assistant", "content": "I will inspect the workspace."},
            {"role": "user", "content": "Focus on src and tests only."},
        ]

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=240)

        assert compacted == messages
        assert runner._compactor.calls == []
        assert runner._last_compaction_diagnostics["compaction_skipped_reason"] == "no_pressure"

    @pytest.mark.asyncio
    async def test_compaction_policy_mode_off_disables_runner_compaction(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=1200)
        runner._runner_compaction_policy_mode = "off"

        result = ToolResult.ok("X" * 5000)
        payload = await runner._serialize_tool_result_for_model(
            "web_fetch",
            result,
            max_output_chars=120,
        )
        parsed = json.loads(payload)

        messages = [
            {"role": "user", "content": "Goal: draft deliverable."},
            {"role": "assistant", "content": "Y " * 1500},
        ]
        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=120)

        assert parsed["output"] == result.output
        assert compacted == messages
        assert runner._compactor.calls == []
        assert runner._last_compaction_diagnostics["compaction_policy_mode"] == "off"
        assert runner._last_compaction_diagnostics["compaction_skipped_reason"] == "policy_disabled"

    @pytest.mark.asyncio
    async def test_compaction_order_tool_trace_before_critical_context(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=1200)
        huge_args_older = json.dumps({"path": "report-old.md", "content": "A" * 14000})
        huge_args_latest = json.dumps({"path": "report.md", "content": "D" * 9000})
        messages = [
            {"role": "user", "content": "Goal: build report."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_older",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args_older},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_older",
                "content": json.dumps({
                    "success": True,
                    "output": "B" * 10000,
                    "error": None,
                    "files_changed": ["report-old.md"],
                }),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_latest",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args_latest},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_latest",
                "content": json.dumps({
                    "success": True,
                    "output": "short ok",
                    "error": None,
                    "files_changed": ["report.md"],
                }),
            },
            {"role": "user", "content": "Historical context " + ("C " * 1400)},
            {
                "role": "assistant",
                "content": "LATEST CRITICAL: keep acceptance criteria unchanged.",
            },
            {"role": "user", "content": "LATEST USER STEER: preserve bullet ordering exactly."},
        ]

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=240)
        labels = [label for label, _max, _size in runner._compactor.calls]
        arg_idx = next(i for i, label in enumerate(labels) if "assistant tool-call args" in label)
        context_indices = [
            idx for idx, label in enumerate(labels)
            if label.endswith("context")
        ]
        if context_indices:
            assert arg_idx < min(context_indices)
        assert any(
            msg.get("content") == "LATEST USER STEER: preserve bullet ordering exactly."
            for msg in compacted
            if isinstance(msg, dict)
        )

    @pytest.mark.asyncio
    async def test_preserve_latest_critical_turns_under_pressure(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=1500)
        messages = [{"role": "user", "content": "Goal: analyze telemetry."}]
        for idx in range(10):
            messages.append({"role": "user", "content": f"Old context {idx}: " + ("x " * 600)})
            messages.append({
                "role": "assistant",
                "content": f"Old assistant note {idx}: " + ("y " * 400),
            })
        latest_assistant = "LATEST CRITICAL ASSISTANT: keep file names exact."
        latest_user = "LATEST CRITICAL USER: do not drop failed-subtask IDs."
        messages.extend([
            {"role": "assistant", "content": latest_assistant},
            {"role": "user", "content": latest_user},
        ])

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=240)
        contents = [
            msg.get("content", "")
            for msg in compacted
            if isinstance(msg, dict) and isinstance(msg.get("content"), str)
        ]
        assert latest_assistant in contents
        assert latest_user in contents

    @pytest.mark.asyncio
    async def test_critical_tier_old_context_merge_without_latest_instruction_loss(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=600)
        huge_args = json.dumps({"path": "output.md", "content": "Z" * 18000})
        messages = [{"role": "user", "content": "Goal: execute and verify all subtasks."}]
        for idx in range(6):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{idx}",
                "content": json.dumps({
                    "success": True,
                    "output": "Q" * 8000,
                    "error": None,
                    "files_changed": ["output.md"],
                }),
            })
            messages.append({"role": "user", "content": f"Older narrative {idx}: " + ("r " * 500)})

        latest_instruction = "LATEST INSTRUCTION: preserve the rubric schema exactly."
        messages.extend([
            {"role": "assistant", "content": latest_instruction},
            {"role": "user", "content": "LATEST USER: keep unresolved evidence IDs in output."},
        ])

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=240)
        contents = [
            msg.get("content", "")
            for msg in compacted
            if isinstance(msg, dict) and isinstance(msg.get("content"), str)
        ]
        assert any(content.startswith("Prior compacted context:\n") for content in contents)
        assert latest_instruction in contents
        assert runner._last_compaction_diagnostics["compaction_pressure_tier"] == "critical"

    @pytest.mark.asyncio
    async def test_memory_extractor_compacts_large_tool_args(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=2000)
        runner._config = Config()
        runner._subtask_deadline_monotonic = time.monotonic() + 120.0
        runner._memory = AsyncMock()
        runner._memory.store_many = AsyncMock()
        runner._event_bus = None
        runner._validator = MagicMock()
        runner._validator.validate_json_response = MagicMock(
            return_value=MagicMock(valid=False, parsed=None),
        )

        class _ExtractorModel:
            name = "mock-extractor"
            roles = ["extractor"]

            def __init__(self):
                self.messages = []

            async def complete(self, messages, **kwargs):
                del kwargs
                self.messages.append(messages)
                return ModelResponse(
                    text="[]",
                    usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
                )

        extractor_model = _ExtractorModel()
        runner._router = MagicMock()
        runner._router.select = MagicMock(return_value=extractor_model)
        runner._prompts = MagicMock()
        runner._prompts.build_extractor_prompt = MagicMock(
            side_effect=lambda subtask_id, tool_calls_formatted, model_output: (
                f"SUBTASK {subtask_id}\nTOOLS\n{tool_calls_formatted}\nOUTPUT\n{model_output}"
            ),
        )

        result = SubtaskResult(
            summary="Execution completed with output artifacts.",
            tool_calls=[
                ToolCallRecord(
                    tool="document_write",
                    args={"path": "report.md", "content": "A" * 200_000},
                    result=ToolResult.ok("ok"),
                ),
            ],
        )

        await runner._extract_memory("task-1", "subtask-1", result)

        assert extractor_model.messages
        prompt = extractor_model.messages[0][0]["content"]
        assert "document_write(" in prompt
        assert "A" * 500 not in prompt
        runner._memory.store_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_near_skips_nonessential_compaction(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=800)
        huge_args_old = json.dumps({"path": "report-old.md", "content": "A" * 12000})
        huge_args_latest = json.dumps({"path": "report.md", "content": "D" * 8000})
        messages = [
            {"role": "user", "content": "Goal: finish report."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args_old},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": json.dumps({
                    "success": True,
                    "output": "B" * 9000,
                    "error": None,
                    "files_changed": ["report-old.md"],
                }),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_latest",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args_latest},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_latest",
                "content": json.dumps({
                    "success": True,
                    "output": "done",
                    "error": None,
                    "files_changed": ["report.md"],
                }),
            },
            {"role": "user", "content": "Old narrative " + ("x " * 800)},
            {"role": "assistant", "content": "LATEST CRITICAL: keep final file name unchanged."},
        ]

        await runner._compact_messages_for_model(messages, remaining_seconds=5)
        labels = [label for label, _max, _size in runner._compactor.calls]
        assert any("assistant tool-call args" in label for label in labels)
        assert all("tool message output" not in label for label in labels)
        assert runner._last_compaction_diagnostics["compaction_skipped_reason"] == "timeout_guard"

    @pytest.mark.asyncio
    async def test_no_hard_truncation_marker_inserted_by_runner_compaction_path(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=700)
        messages = [{"role": "user", "content": "Goal: reduce context safely."}]
        for idx in range(8):
            messages.append({"role": "user", "content": f"Context {idx}: " + ("data " * 1000)})
        messages.append({"role": "assistant", "content": "LATEST CRITICAL: keep all file paths."})

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=180)
        serialized = json.dumps(compacted, ensure_ascii=False, default=str)
        assert "...[truncated]..." not in serialized

    def test_emit_tool_event_emits_artifact_confinement_violation(self):
        from loom.engine.runner import SubtaskRunner

        bus = EventBus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._event_bus = bus

        runner._emit_tool_event(
            TOOL_CALL_COMPLETED,
            "task-1",
            "subtask-1",
            "write_file",
            {"path": "../outside.md"},
            result=ToolResult.fail(
                "Safety violation: Path '/tmp/outside.md' escapes workspace '/tmp/run'."
            ),
            workspace=Path("/tmp/run"),
        )

        event_types = [event.event_type for event in events]
        assert TOOL_CALL_COMPLETED in event_types
        assert ARTIFACT_CONFINEMENT_VIOLATION in event_types
        violation = next(
            event for event in events
            if event.event_type == ARTIFACT_CONFINEMENT_VIOLATION
        )
        assert violation.data["attempted_path"] == "../outside.md"

    def test_tool_call_completed_payload_contract_additive(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        runner._emit_tool_event(
            TOOL_CALL_COMPLETED,
            "task-1",
            "subtask-1",
            "web_fetch",
            {"url": "https://example.com/report.pdf"},
            result=ToolResult.ok("ok"),
        )

        tool_event = next(event for event in events if event.event_type == TOOL_CALL_COMPLETED)
        expected_fields = {
            "subtask_id": "subtask-1",
            "tool": "web_fetch",
            "args": {"url": "https://example.com/report.pdf"},
            "success": True,
            "error": "",
            "files_changed": [],
            "files_changed_paths": [],
        }
        for key, value in expected_fields.items():
            assert tool_event.data.get(key) == value

        # EventBus normalization is additive; required business fields must remain stable.
        assert tool_event.data.get("task_id") == "task-1"
        assert str(tool_event.data.get("timestamp", "")).strip()
        assert str(tool_event.data.get("event_id", "")).strip()
        assert str(tool_event.data.get("correlation_id", "")).strip().startswith("task:")
        assert int(tool_event.data.get("sequence", 0)) >= 1
        assert int(tool_event.data.get("schema_version", 0)) >= 1
        assert str(tool_event.data.get("source_component", "")).strip()

    def test_artifact_ingest_telemetry_required_fields_and_redaction(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        result = ToolResult.ok(
            "Fetched PDF artifact",
            data={
                "url": "https://example.com/report.pdf?token=secret#fragment",
                "content_kind": "pdf",
                "content_type": "application/pdf",
                "artifact_ref": "af_1234abcd",
                "artifact_workspace_relpath": ".loom_artifacts/fetched/s1/af_1234abcd.pdf",
                "size_bytes": 4096,
                "declared_size_bytes": 5000,
                "handler": "pdf_handler",
                "extracted_chars": 1800,
                "extraction_truncated": True,
                "handler_metadata": {"details": "x" * 800},
            },
        )

        runner._emit_artifact_ingest_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="web_fetch",
            tool_args={"url": "https://example.com/report.pdf?token=secret#fragment"},
            result=result,
        )

        classified = next(
            event for event in events
            if event.event_type == ARTIFACT_INGEST_CLASSIFIED
        )
        completed = next(
            event for event in events
            if event.event_type == ARTIFACT_INGEST_COMPLETED
        )
        for event in (classified, completed):
            payload = event.data
            assert payload["subtask_id"] == "subtask-1"
            assert payload["tool"] == "web_fetch"
            assert payload["status"] == "ok"
            assert payload["url"] == "https://example.com/report.pdf"
            assert "token=secret" not in payload["url"]
            assert payload["content_kind"] == "pdf"
            assert payload["content_type"] == "application/pdf"
            assert payload["artifact_ref"] == "af_1234abcd"
            assert payload["artifact_workspace_relpath"].startswith(".loom_artifacts/")
            assert payload["size_bytes"] == 4096
            assert payload["declared_size_bytes"] == 5000
            assert payload["handler"] == "pdf_handler"
            assert payload["extracted_chars"] == 1800
            assert payload["extraction_truncated"] is True
            metadata_payload = payload.get("handler_metadata")
            assert isinstance(metadata_payload, dict)
            assert metadata_payload.get("_loom_meta") == "metadata_omitted"
            assert metadata_payload.get("original_type") == "dict"
            assert isinstance(metadata_payload.get("sha1"), str)
            assert metadata_payload["sha1"]
            assert "truncated" not in json.dumps(metadata_payload, ensure_ascii=False)
        assert runner._active_subtask_telemetry_counters["artifact_ingests"] == 1

    def test_artifact_retention_event_emitted_only_when_deletions_occur(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        no_delete = ToolResult.ok(
            "ok",
            data={
                "url": "https://example.com/report.pdf",
                "content_kind": "pdf",
                "content_type": "application/pdf",
                "artifact_ref": "af_no_delete",
                "artifact_workspace_relpath": ".loom_artifacts/fetched/s1/af_no_delete.pdf",
                "artifact_retention": {
                    "scopes_scanned": 1,
                    "files_deleted": 0,
                    "bytes_deleted": 0,
                },
            },
        )
        runner._emit_artifact_ingest_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="web_fetch",
            tool_args={"url": "https://example.com/report.pdf"},
            result=no_delete,
        )
        assert ARTIFACT_RETENTION_PRUNED not in [event.event_type for event in events]

        with_delete = ToolResult.ok(
            "ok",
            data={
                "url": "https://example.com/report.pdf",
                "content_kind": "pdf",
                "content_type": "application/pdf",
                "artifact_ref": "af_deleted",
                "artifact_workspace_relpath": ".loom_artifacts/fetched/s1/af_deleted.pdf",
                "artifact_retention": {
                    "scopes_scanned": 2,
                    "files_deleted": 3,
                    "bytes_deleted": 9000,
                },
            },
        )
        runner._emit_artifact_ingest_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="web_fetch",
            tool_args={"url": "https://example.com/report.pdf"},
            result=with_delete,
        )
        retention = next(
            event for event in events
            if event.event_type == ARTIFACT_RETENTION_PRUNED
        )
        assert retention.data["files_deleted"] == 3
        assert retention.data["bytes_deleted"] == 9000
        assert runner._active_subtask_telemetry_counters["artifact_retention_deletes"] == 3

    def test_artifact_read_completed_emits_success_and_failure(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        success_result = ToolResult.ok(
            "ok",
            data={
                "source_url": "https://example.com/report.pdf?sig=hidden",
                "content_kind": "pdf",
                "content_type": "application/pdf",
                "artifact_ref": "af_read_ok",
                "artifact_workspace_relpath": ".loom_artifacts/fetched/s1/af_read_ok.pdf",
                "handler": "pdf_handler",
                "extracted_chars": 1200,
                "extraction_truncated": False,
            },
        )
        runner._emit_artifact_read_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="read_artifact",
            tool_args={"artifact_ref": "af_read_ok"},
            result=success_result,
        )

        failed_result = ToolResult.fail("Artifact not found")
        runner._emit_artifact_read_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="read_artifact",
            tool_args={"artifact_ref": "af_missing"},
            result=failed_result,
        )

        read_events = [
            event for event in events
            if event.event_type == ARTIFACT_READ_COMPLETED
        ]
        assert len(read_events) == 2
        assert read_events[0].data["status"] == "ok"
        assert read_events[0].data["url"] == "https://example.com/report.pdf"
        assert read_events[1].data["status"] == "error"
        assert read_events[1].data["artifact_ref"] == "af_missing"
        assert runner._active_subtask_telemetry_counters["artifact_reads"] == 2

    def test_compaction_and_overflow_telemetry_events(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        runner._emit_compaction_policy_decision_from_diagnostics(
            task_id="task-1",
            subtask_id="subtask-1",
        )
        decision_event = next(
            event for event in events
            if event.event_type == COMPACTION_POLICY_DECISION
        )
        assert decision_event.data["decision"] == "compact_tool"
        assert decision_event.data["reason"] == "tool_output_compacted"

        runner._emit_overflow_fallback_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            report={
                "overflow_fallback_applied": True,
                "overflow_fallback_rewritten_messages": 2,
                "overflow_fallback_chars_reduced": 6400,
                "overflow_fallback_preserved_recent_messages": 1,
            },
        )
        overflow_events = [
            event for event in events
            if event.event_type == OVERFLOW_FALLBACK_APPLIED
        ]
        assert len(overflow_events) == 1
        overflow_payload = overflow_events[0].data
        assert overflow_payload["decision"] == "fallback_rewrite"
        assert overflow_payload["rewritten_messages"] == 2
        assert overflow_payload["chars_reduced"] == 6400
        assert overflow_payload["preserved_recent_messages"] == 1

        runner._emit_overflow_fallback_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            report={"overflow_fallback_applied": False},
        )
        overflow_events = [
            event for event in events
            if event.event_type == OVERFLOW_FALLBACK_APPLIED
        ]
        assert len(overflow_events) == 1
        assert runner._active_subtask_telemetry_counters["overflow_fallback_count"] == 1

    def test_tool_iteration_budget_uses_global_limit(self):
        from loom.engine.runner import SubtaskRunner

        research_subtask = Subtask(
            id="collect-evidence",
            description="Research and collect supporting evidence.",
        )
        verify_subtask = Subtask(
            id="verify-findings",
            description="Run verification checks on outputs.",
        )
        final_subtask = Subtask(
            id="evaluate-select-twelve",
            description=(
                "Apply selection rubric to longlist to select exactly 12 final cases."
            ),
        )
        research_budget = SubtaskRunner._tool_iteration_budget(
            subtask=research_subtask,
            retry_strategy="",
            has_expected_deliverables=False,
        )
        verify_budget = SubtaskRunner._tool_iteration_budget(
            subtask=verify_subtask,
            retry_strategy="",
            has_expected_deliverables=False,
        )
        remediation_budget = SubtaskRunner._tool_iteration_budget(
            subtask=research_subtask,
            retry_strategy="evidence_gap",
            has_expected_deliverables=True,
        )
        final_budget = SubtaskRunner._tool_iteration_budget(
            subtask=final_subtask,
            retry_strategy="",
            has_expected_deliverables=False,
        )
        custom_budget = SubtaskRunner._tool_iteration_budget(
            subtask=final_subtask,
            retry_strategy="rate_limit",
            has_expected_deliverables=True,
            base_budget=37,
        )

        assert research_budget == SubtaskRunner.MAX_TOOL_ITERATIONS
        assert verify_budget == SubtaskRunner.MAX_TOOL_ITERATIONS
        assert final_budget == SubtaskRunner.MAX_TOOL_ITERATIONS
        assert remediation_budget == SubtaskRunner.MAX_TOOL_ITERATIONS
        assert custom_budget == 37

    def test_deliverable_policy_blocks_variant_and_noncanonical_retry_paths(self, tmp_path):
        from loom.engine.runner import SubtaskRunner

        variant_error = SubtaskRunner._validate_deliverable_write_policy(
            tool_name="write_file",
            tool_args={"path": "analysis-v2.md"},
            workspace=tmp_path,
            expected_deliverables=["analysis.md"],
            forbidden_deliverables=[],
            allowed_output_prefixes=[],
            enforce_deliverable_paths=False,
            edit_existing_only=False,
        )
        assert variant_error is not None
        assert "analysis.md" in variant_error
        assert "reason_code=forbidden_output_path" in variant_error

        noncanonical_error = SubtaskRunner._validate_deliverable_write_policy(
            tool_name="write_file",
            tool_args={"path": "scratch-notes.md"},
            workspace=tmp_path,
            expected_deliverables=["analysis.md"],
            forbidden_deliverables=[],
            allowed_output_prefixes=[],
            enforce_deliverable_paths=True,
            edit_existing_only=True,
        )
        assert noncanonical_error is not None
        assert "Unexpected target(s)" in noncanonical_error
        assert "reason_code=forbidden_output_path" in noncanonical_error

    def test_deliverable_policy_blocks_forbidden_canonical_worker_writes(self, tmp_path):
        from loom.engine.runner import SubtaskRunner

        forbidden_error = SubtaskRunner._validate_deliverable_write_policy(
            tool_name="write_file",
            tool_args={"path": "analysis.md"},
            workspace=tmp_path,
            expected_deliverables=[],
            forbidden_deliverables=["analysis.md"],
            allowed_output_prefixes=[],
            enforce_deliverable_paths=False,
            edit_existing_only=False,
        )
        assert forbidden_error is not None
        assert "reserved for a phase finalizer" in forbidden_error
        assert "reason_code=forbidden_output_path" in forbidden_error

    def test_deliverable_policy_enforces_worker_intermediate_prefix(self, tmp_path):
        from loom.engine.runner import SubtaskRunner

        violation = SubtaskRunner._validate_deliverable_write_policy(
            tool_name="write_file",
            tool_args={"path": "analysis.md"},
            workspace=tmp_path,
            expected_deliverables=[],
            forbidden_deliverables=[],
            allowed_output_prefixes=[".loom/phase-artifacts/run-1/phase-a/worker-a"],
            enforce_deliverable_paths=False,
            edit_existing_only=False,
        )
        assert violation is not None
        assert "Fan-in worker output path violation" in violation
        assert "reason_code=forbidden_output_path" in violation

        allowed = SubtaskRunner._validate_deliverable_write_policy(
            tool_name="write_file",
            tool_args={"path": ".loom/phase-artifacts/run-1/phase-a/worker-a/out.md"},
            workspace=tmp_path,
            expected_deliverables=[],
            forbidden_deliverables=[],
            allowed_output_prefixes=[".loom/phase-artifacts/run-1/phase-a/worker-a"],
            enforce_deliverable_paths=False,
            edit_existing_only=False,
        )
        assert allowed is None

        workspace_named_loom = tmp_path / "loom"
        workspace_named_loom.mkdir()
        allowed_when_workspace_name_matches_prefix = (
            SubtaskRunner._validate_deliverable_write_policy(
                tool_name="write_file",
                tool_args={"path": ".loom/phase-artifacts/run-1/phase-a/worker-a/out.md"},
                workspace=workspace_named_loom,
                expected_deliverables=[],
                forbidden_deliverables=[],
                allowed_output_prefixes=["loom/phase-artifacts/run-1/phase-a/worker-a"],
                enforce_deliverable_paths=False,
                edit_existing_only=False,
            )
        )
        assert allowed_when_workspace_name_matches_prefix is None

    def test_completion_contract_mutation_mismatch_detection(self, tmp_path):
        runner = self._make_runner_for_telemetry()

        tool_calls = [
            ToolCallRecord(
                tool="write_file",
                args={"path": "analysis.md"},
                result=ToolResult.ok("ok", files_changed=["analysis.md"]),
            ),
        ]
        mismatch = runner._completion_contract_mutation_mismatch(
            response_text=json.dumps({
                "status": "success",
                "deliverables_touched": ["report.md"],
                "verification_notes": "done",
            }),
            tool_calls=tool_calls,
            workspace=tmp_path,
        )
        assert "Completion contract mismatch" in mismatch
        assert "Declared: report.md" in mismatch
        assert "Actual: analysis.md" in mismatch

        no_mismatch = runner._completion_contract_mutation_mismatch(
            response_text=json.dumps({
                "status": "success",
                "deliverables_touched": ["analysis.md"],
                "verification_notes": "done",
            }),
            tool_calls=tool_calls,
            workspace=tmp_path,
        )
        assert no_mismatch == ""

    def test_forbidden_output_path_error_detector(self):
        from loom.engine.runner import SubtaskRunner

        assert SubtaskRunner._is_forbidden_output_path_error(
            "reason_code=forbidden_output_path; Canonical deliverable policy violation.",
        )
        assert not SubtaskRunner._is_forbidden_output_path_error(
            "Permission denied",
        )

    def test_sealed_artifact_policy_blocks_edit_without_post_seal_evidence(self, tmp_path):
        from loom.engine.runner import SubtaskRunner

        task = _make_task(goal="Seal enforcement", workspace=str(tmp_path))
        task.metadata["artifact_seals"] = {
            "analysis.md": {
                "path": "analysis.md",
                "sha256": hashlib.sha256(b"sealed").hexdigest(),
                "subtask_id": "s1",
                "sealed_at": "2026-03-05T10:00:00",
            },
        }
        task.metadata["validity_scorecard"] = {
            "subtask_metrics": {
                "s1": {"verification_outcome": "pass"},
            },
        }
        prior_calls = [
            ToolCallRecord(
                tool="read_file",
                args={"path": "analysis.md"},
                result=ToolResult.ok("old evidence"),
                timestamp="2026-03-05T09:59:59",
            ),
        ]

        error = SubtaskRunner._validate_sealed_artifact_mutation_policy(
            task=task,
            tool_name="edit_file",
            tool_args={"path": "analysis.md"},
            workspace=tmp_path,
            prior_successful_tool_calls=prior_calls,
            current_tool_calls=[],
        )

        assert error is not None
        assert "analysis.md" in error
        assert "blocked" in error.lower()

    def test_sealed_artifact_policy_allows_edit_with_post_seal_evidence(self, tmp_path):
        from loom.engine.runner import SubtaskRunner

        task = _make_task(goal="Seal enforcement", workspace=str(tmp_path))
        task.metadata["artifact_seals"] = {
            "analysis.md": {
                "path": "analysis.md",
                "sha256": hashlib.sha256(b"sealed").hexdigest(),
                "subtask_id": "s1",
                "sealed_at": "2026-03-05T10:00:00",
            },
        }
        task.metadata["validity_scorecard"] = {
            "subtask_metrics": {
                "s1": {"verification_outcome": "pass"},
            },
        }
        prior_calls = [
            ToolCallRecord(
                tool="web_fetch",
                args={"url": "https://example.com"},
                result=ToolResult.ok("new evidence"),
                timestamp="2026-03-05T10:10:00",
            ),
        ]

        error = SubtaskRunner._validate_sealed_artifact_mutation_policy(
            task=task,
            tool_name="edit_file",
            tool_args={"path": "analysis.md"},
            workspace=tmp_path,
            prior_successful_tool_calls=prior_calls,
            current_tool_calls=[],
        )

        assert error is None

    def test_reseal_updates_sha_for_tracked_verified_artifact(self, tmp_path):
        from loom.engine.runner import SubtaskRunner

        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        artifact = workspace / "analysis.md"
        artifact.write_text("sealed", encoding="utf-8")

        task = _make_task(goal="Reseal", workspace=str(workspace))
        task.metadata["artifact_seals"] = {
            "analysis.md": {
                "path": "analysis.md",
                "sha256": hashlib.sha256(b"sealed").hexdigest(),
                "subtask_id": "s1",
                "sealed_at": "2026-03-05T10:00:00",
            },
        }
        task.metadata["validity_scorecard"] = {
            "subtask_metrics": {
                "s1": {"verification_outcome": "pass"},
            },
        }

        artifact.write_text("updated with evidence", encoding="utf-8")
        updated = SubtaskRunner._reseal_tracked_artifacts_after_mutation(
            task=task,
            workspace=workspace,
            tool_name="edit_file",
            tool_args={"path": "analysis.md"},
            tool_result=ToolResult.ok("edited", files_changed=["analysis.md"]),
            subtask_id="s2",
            tool_call_id="call-1",
        )

        assert updated == 1
        seal = task.metadata.get("artifact_seals", {}).get("analysis.md", {})
        assert seal.get("sha256") == hashlib.sha256(b"updated with evidence").hexdigest()
        assert seal.get("subtask_id") == "s2"
        assert seal.get("verified_origin") is True

    def test_reseal_moves_verified_seal_to_destination(self, tmp_path):
        from loom.engine.runner import SubtaskRunner

        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        source = workspace / "analysis.md"
        source.write_text("sealed", encoding="utf-8")
        destination = workspace / "analysis-v2.md"
        source.rename(destination)

        task = _make_task(goal="Reseal move", workspace=str(workspace))
        task.metadata["artifact_seals"] = {
            "analysis.md": {
                "path": "analysis.md",
                "sha256": hashlib.sha256(b"sealed").hexdigest(),
                "subtask_id": "s1",
                "sealed_at": "2026-03-05T10:00:00",
            },
        }
        task.metadata["validity_scorecard"] = {
            "subtask_metrics": {
                "s1": {"verification_outcome": "pass"},
            },
        }

        updated = SubtaskRunner._reseal_tracked_artifacts_after_mutation(
            task=task,
            workspace=workspace,
            tool_name="move_file",
            tool_args={"source": "analysis.md", "destination": "analysis-v2.md"},
            tool_result=ToolResult.ok(
                "moved",
                files_changed=["analysis.md", "analysis-v2.md"],
            ),
            subtask_id="s2",
            tool_call_id="call-2",
        )

        assert updated == 2
        seals = task.metadata.get("artifact_seals", {})
        assert "analysis.md" not in seals
        moved = seals.get("analysis-v2.md", {})
        assert moved.get("sha256") == hashlib.sha256(b"sealed").hexdigest()
        assert moved.get("verified_origin") is True
