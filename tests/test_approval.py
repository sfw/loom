"""Tests for the approval manager."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from loom.events.bus import EventBus
from loom.recovery.approval import (
    ApprovalDecision,
    ApprovalManager,
    ApprovalRequest,
)
from loom.tools.registry import ToolResult

# --- Helpers ---


@dataclass
class MockToolCallRecord:
    tool: str
    args: dict
    result: ToolResult


@dataclass
class MockSubtaskResult:
    status: str = "success"
    summary: str = "Done"
    tool_calls: list = field(default_factory=list)


# --- Tests ---


class TestApprovalDecisions:
    def test_manual_mode_always_waits(self):
        mgr = ApprovalManager(EventBus())
        decision = mgr.check_approval(
            approval_mode="manual",
            confidence=1.0,
            result=MockSubtaskResult(),
        )
        assert decision == ApprovalDecision.WAIT

    def test_auto_mode_proceeds_at_high_confidence(self):
        mgr = ApprovalManager(EventBus())
        decision = mgr.check_approval(
            approval_mode="auto",
            confidence=0.9,
            result=MockSubtaskResult(),
        )
        assert decision == ApprovalDecision.PROCEED

    def test_auto_mode_waits_with_timeout_at_medium_confidence(self):
        mgr = ApprovalManager(EventBus())
        decision = mgr.check_approval(
            approval_mode="auto",
            confidence=0.6,
            result=MockSubtaskResult(),
        )
        assert decision == ApprovalDecision.WAIT_WITH_TIMEOUT

    def test_auto_mode_waits_at_low_confidence(self):
        mgr = ApprovalManager(EventBus())
        decision = mgr.check_approval(
            approval_mode="auto",
            confidence=0.3,
            result=MockSubtaskResult(),
        )
        assert decision == ApprovalDecision.WAIT

    def test_auto_mode_aborts_at_zero_confidence(self):
        mgr = ApprovalManager(EventBus())
        decision = mgr.check_approval(
            approval_mode="auto",
            confidence=0.1,
            result=MockSubtaskResult(),
        )
        assert decision == ApprovalDecision.ABORT

    def test_auto_mode_gates_destructive_operations(self):
        mgr = ApprovalManager(EventBus())
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="shell_execute",
                args={"command": "rm -rf /tmp/data"},
                result=ToolResult.ok("ok"),
            )],
        )
        decision = mgr.check_approval(
            approval_mode="auto",
            confidence=0.95,  # High confidence, but destructive
            result=result,
        )
        assert decision == ApprovalDecision.WAIT

    def test_confidence_threshold_mode_proceeds_above_threshold(self):
        mgr = ApprovalManager(EventBus())
        decision = mgr.check_approval(
            approval_mode="confidence_threshold",
            confidence=0.85,
            result=MockSubtaskResult(),
            confidence_threshold=0.8,
        )
        assert decision == ApprovalDecision.PROCEED

    def test_confidence_threshold_mode_waits_below_threshold(self):
        mgr = ApprovalManager(EventBus())
        decision = mgr.check_approval(
            approval_mode="confidence_threshold",
            confidence=0.7,
            result=MockSubtaskResult(),
            confidence_threshold=0.8,
        )
        assert decision == ApprovalDecision.WAIT

    def test_confidence_threshold_mode_gates_destructive(self):
        mgr = ApprovalManager(EventBus())
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="shell_execute",
                args={"command": "drop table users"},
                result=ToolResult.ok("ok"),
            )],
        )
        decision = mgr.check_approval(
            approval_mode="confidence_threshold",
            confidence=0.95,
            result=result,
            confidence_threshold=0.8,
        )
        assert decision == ApprovalDecision.WAIT

    def test_unknown_mode_defaults_to_proceed_at_high_confidence(self):
        mgr = ApprovalManager(EventBus())
        decision = mgr.check_approval(
            approval_mode="unknown_mode",
            confidence=0.9,
            result=MockSubtaskResult(),
        )
        assert decision == ApprovalDecision.PROCEED


class TestAlwaysGated:
    def test_detects_rm_rf(self):
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="shell_execute",
                args={"command": "rm -rf /tmp"},
                result=ToolResult.ok("ok"),
            )],
        )
        assert ApprovalManager._is_always_gated(result)

    def test_detects_sudo(self):
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="shell_execute",
                args={"command": "sudo apt install nginx"},
                result=ToolResult.ok("ok"),
            )],
        )
        assert ApprovalManager._is_always_gated(result)

    def test_detects_env_file_write(self):
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="write_file",
                args={"path": ".env.production"},
                result=ToolResult.ok("ok"),
            )],
        )
        assert ApprovalManager._is_always_gated(result)

    def test_safe_operations_not_gated(self):
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="write_file",
                args={"path": "main.py", "content": "print('hello')"},
                result=ToolResult.ok("ok"),
            )],
        )
        assert not ApprovalManager._is_always_gated(result)


class TestApprovalWorkflow:
    @pytest.mark.asyncio
    async def test_resolve_approval_sets_event(self):
        bus = EventBus()
        mgr = ApprovalManager(bus)

        # Start a request in the background
        request = ApprovalRequest(
            task_id="t1",
            subtask_id="s1",
            reason="Test",
            proposed_action="Write file",
            risk_level="medium",
        )

        async def resolve_after_delay():
            await asyncio.sleep(0.05)
            mgr.resolve_approval("t1", "s1", approved=True)

        asyncio.create_task(resolve_after_delay())
        result = await mgr.request_approval(request)
        assert result is True

    @pytest.mark.asyncio
    async def test_resolve_rejection(self):
        bus = EventBus()
        mgr = ApprovalManager(bus)

        request = ApprovalRequest(
            task_id="t1",
            subtask_id="s1",
            reason="Test",
            proposed_action="Write file",
            risk_level="medium",
        )

        async def reject_after_delay():
            await asyncio.sleep(0.05)
            mgr.resolve_approval("t1", "s1", approved=False)

        asyncio.create_task(reject_after_delay())
        result = await mgr.request_approval(request)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_approve_on_timeout(self):
        bus = EventBus()
        mgr = ApprovalManager(bus)

        request = ApprovalRequest(
            task_id="t1",
            subtask_id="s1",
            reason="Test",
            proposed_action="Write file",
            risk_level="low",
            auto_approve_timeout=1,  # 1 second
        )

        result = await mgr.request_approval(request)
        assert result is True

    def test_resolve_nonexistent_approval(self):
        bus = EventBus()
        mgr = ApprovalManager(bus)

        resolved = mgr.resolve_approval("t1", "s1", approved=True)
        assert resolved is False
