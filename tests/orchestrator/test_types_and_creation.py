"""Orchestrator type and task creation tests."""

from __future__ import annotations

from loom.engine.orchestrator import SubtaskResult, ToolCallRecord, create_task
from loom.tools.registry import ToolResult


class TestCreateTask:
    def test_creates_with_unique_id(self):
        t1 = create_task("goal 1")
        t2 = create_task("goal 2")
        assert t1.id != t2.id

    def test_has_required_fields(self):
        t = create_task("my goal", workspace="/tmp/w", approval_mode="confirm")
        assert t.goal == "my goal"
        assert t.workspace == "/tmp/w"
        assert t.approval_mode == "confirm"
        assert t.created_at != ""

    def test_defaults(self):
        t = create_task("g")
        assert t.workspace == ""
        assert t.approval_mode == "auto"
        assert t.context == {}


# --- ToolCallRecord ---


class TestToolCallRecord:
    def test_auto_timestamp(self):
        r = ToolCallRecord(tool="read", args={}, result=ToolResult.ok("ok"))
        assert r.timestamp != ""

    def test_explicit_timestamp(self):
        r = ToolCallRecord(
            tool="read", args={}, result=ToolResult.ok("ok"),
            timestamp="2025-01-01",
        )
        assert r.timestamp == "2025-01-01"


# --- SubtaskResult ---


class TestSubtaskResult:
    def test_defaults(self):
        r = SubtaskResult(status="success", summary="done")
        assert r.tool_calls == []
        assert r.duration_seconds == 0.0
        assert r.tokens_used == 0


# --- Orchestrator ---
