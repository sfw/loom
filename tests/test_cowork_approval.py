"""Tests for the cowork tool approval system."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from loom.cowork.approval import (
    AUTO_APPROVED_TOOLS,
    ApprovalDecision,
    ToolApprover,
    _format_args_preview,
)
from loom.cowork.session import CoworkSession, CoworkTurn, ToolCallEvent
from loom.models.base import ModelProvider, ModelResponse, TokenUsage, ToolCall
from loom.tools import create_default_registry


# --- Fixtures ---


class MockProvider(ModelProvider):
    """Mock model for testing."""

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, messages, tools=None, **kwargs):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return ModelResponse(text="Done.", usage=TokenUsage())

    async def health_check(self):
        return True

    @property
    def name(self):
        return "mock"

    @property
    def tier(self):
        return 1

    @property
    def roles(self):
        return ["executor"]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "test.py").write_text("print('hello')\n")
    return tmp_path


@pytest.fixture
def tools():
    return create_default_registry()


# --- ToolApprover unit tests ---


class TestAutoApprovedTools:
    def test_read_only_tools(self):
        for t in ("read_file", "search_files", "list_directory",
                   "glob_find", "ripgrep_search", "analyze_code",
                   "ask_user", "web_search", "web_fetch"):
            assert t in AUTO_APPROVED_TOOLS

    def test_write_tools_excluded(self):
        for t in ("write_file", "edit_file", "delete_file",
                   "move_file", "shell_execute", "git_command"):
            assert t not in AUTO_APPROVED_TOOLS


class TestToolApprover:
    async def test_auto_approved_skips_callback(self):
        cb = AsyncMock(return_value=ApprovalDecision.DENY)
        approver = ToolApprover(prompt_callback=cb)
        decision = await approver.check("read_file", {"path": "x"})
        assert decision == ApprovalDecision.APPROVE
        cb.assert_not_called()

    async def test_no_callback_permissive(self):
        approver = ToolApprover()
        decision = await approver.check("shell_execute", {"command": "ls"})
        assert decision == ApprovalDecision.APPROVE

    async def test_callback_approve(self):
        cb = AsyncMock(return_value=ApprovalDecision.APPROVE)
        approver = ToolApprover(prompt_callback=cb)
        decision = await approver.check("shell_execute", {"command": "ls"})
        assert decision == ApprovalDecision.APPROVE
        cb.assert_called_once_with("shell_execute", {"command": "ls"})

    async def test_callback_deny(self):
        cb = AsyncMock(return_value=ApprovalDecision.DENY)
        approver = ToolApprover(prompt_callback=cb)
        decision = await approver.check("shell_execute", {"command": "rm -rf /"})
        assert decision == ApprovalDecision.DENY

    async def test_approve_all_remembers_per_tool(self):
        cb = AsyncMock(return_value=ApprovalDecision.APPROVE_ALL)
        approver = ToolApprover(prompt_callback=cb)

        # First call prompts
        d1 = await approver.check("shell_execute", {"command": "ls"})
        assert d1 == ApprovalDecision.APPROVE
        assert cb.call_count == 1

        # Second call for same tool: no prompt
        d2 = await approver.check("shell_execute", {"command": "pwd"})
        assert d2 == ApprovalDecision.APPROVE
        assert cb.call_count == 1

        # Different tool still prompts
        await approver.check("git_command", {"args": ["status"]})
        assert cb.call_count == 2

    async def test_always_approved_property(self):
        cb = AsyncMock(return_value=ApprovalDecision.APPROVE_ALL)
        approver = ToolApprover(prompt_callback=cb)
        assert len(approver.always_approved_tools) == 0

        await approver.check("shell_execute", {"command": "ls"})
        assert "shell_execute" in approver.always_approved_tools

    async def test_manual_approve_always(self):
        cb = AsyncMock(return_value=ApprovalDecision.DENY)
        approver = ToolApprover(prompt_callback=cb)
        approver.approve_tool_always("shell_execute")

        decision = await approver.check("shell_execute", {"command": "ls"})
        assert decision == ApprovalDecision.APPROVE
        cb.assert_not_called()


class TestFormatArgsPreview:
    def test_shell(self):
        assert _format_args_preview("shell_execute", {"command": "ls -la"}) == "ls -la"

    def test_shell_truncated(self):
        long_cmd = "a" * 100
        preview = _format_args_preview("shell_execute", {"command": long_cmd})
        assert len(preview) <= 83

    def test_file_tool(self):
        assert _format_args_preview("write_file", {"path": "foo.py"}) == "foo.py"

    def test_git(self):
        assert _format_args_preview("git_command", {"args": ["push", "origin"]}) == "push origin"

    def test_generic(self):
        assert _format_args_preview("unknown_tool", {"x": "hello"}) == "hello"


# --- Integration: approval in CoworkSession ---


class TestSessionApproval:
    async def test_denied_tool_returns_error_result(self, workspace, tools):
        """Denied tools get an error result, model sees the denial."""
        cb = AsyncMock(return_value=ApprovalDecision.DENY)
        approver = ToolApprover(prompt_callback=cb)

        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tc1", name="shell_execute",
                    arguments={"command": "ls"},
                )],
                usage=TokenUsage(total_tokens=10),
            ),
            ModelResponse(
                text="Shell was denied.",
                usage=TokenUsage(total_tokens=5),
            ),
        ])

        session = CoworkSession(
            model=provider, tools=tools,
            workspace=workspace, approver=approver,
        )

        events = []
        async for event in session.send("Run ls"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        # The tool call is emitted twice (start + complete with denial)
        assert len(tool_events) == 2
        assert tool_events[1].result is not None
        assert not tool_events[1].result.success
        assert "denied" in tool_events[1].result.error.lower()

    async def test_approved_tool_executes_normally(self, workspace, tools):
        """Approved tools execute as usual."""
        cb = AsyncMock(return_value=ApprovalDecision.APPROVE)
        approver = ToolApprover(prompt_callback=cb)

        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tc1", name="read_file",
                    arguments={"path": "test.py"},
                )],
                usage=TokenUsage(total_tokens=10),
            ),
            ModelResponse(
                text="The file has a print statement.",
                usage=TokenUsage(total_tokens=5),
            ),
        ])

        session = CoworkSession(
            model=provider, tools=tools,
            workspace=workspace, approver=approver,
        )

        events = []
        async for event in session.send("Read test.py"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        # Each tool emits twice: start (no result) + complete (with result)
        completed = [e for e in tool_events if e.result is not None]
        # Both yields reference the same event object, so deduplicate
        unique_completed = list({id(e): e for e in completed}.values())
        assert len(unique_completed) == 1
        assert unique_completed[0].result.success

        # read_file is auto-approved, so callback should NOT be called
        cb.assert_not_called()

    async def test_no_approver_permissive(self, workspace, tools):
        """Without an approver, everything runs."""
        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tc1", name="read_file",
                    arguments={"path": "test.py"},
                )],
                usage=TokenUsage(total_tokens=10),
            ),
            ModelResponse(
                text="Done.",
                usage=TokenUsage(total_tokens=5),
            ),
        ])

        session = CoworkSession(
            model=provider, tools=tools,
            workspace=workspace,  # no approver
        )

        events = []
        async for event in session.send("Read file"):
            events.append(event)

        completed = [e for e in events
                     if isinstance(e, ToolCallEvent) and e.result is not None]
        unique_completed = list({id(e): e for e in completed}.values())
        assert len(unique_completed) == 1
        assert unique_completed[0].result.success
