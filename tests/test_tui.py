"""Tests for the TUI app and its components."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from loom.tui.api_client import LoomAPIClient
from loom.tui.app import (
    AskUserScreen,
    ToolApprovalScreen,
    _tool_args_preview,
    _tool_output_preview,
    _trunc,
)

# --- LoomAPIClient tests (unchanged from before) ---


class TestLoomAPIClient:
    def test_init_default_url(self):
        client = LoomAPIClient()
        assert client._base_url == "http://localhost:9000"

    def test_init_custom_url(self):
        client = LoomAPIClient("http://myhost:8080/")
        assert client._base_url == "http://myhost:8080"

    async def test_close_when_no_client(self):
        api = LoomAPIClient()
        await api.close()  # Should not raise

    async def test_list_tasks(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [{"task_id": "t1", "status": "running"}]
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        api._client = mock_client

        tasks = await api.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "t1"
        mock_client.get.assert_called_once_with("/tasks")

    async def test_create_task(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"task_id": "t1", "status": "pending"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.create_task("Build a CLI", workspace="/tmp/proj")
        assert result["task_id"] == "t1"

    async def test_health(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.health()
        assert result["status"] == "ok"


# --- Display helper tests ---


class TestToolArgsPreview:
    def test_file_tool(self):
        assert _tool_args_preview("read_file", {"path": "foo.py"}) == "foo.py"

    def test_shell(self):
        assert _tool_args_preview("shell_execute", {"command": "ls -la"}) == "ls -la"

    def test_git(self):
        assert _tool_args_preview("git_command", {"args": ["push", "origin"]}) == "push origin"

    def test_ripgrep(self):
        assert _tool_args_preview("ripgrep_search", {"pattern": "TODO"}) == "/TODO/"

    def test_glob(self):
        assert _tool_args_preview("glob_find", {"pattern": "**/*.py"}) == "**/*.py"

    def test_web_search(self):
        assert _tool_args_preview("web_search", {"query": "python docs"}) == "python docs"

    def test_web_fetch(self):
        assert _tool_args_preview("web_fetch", {"url": "https://example.com"}) == "https://example.com"

    def test_task_tracker(self):
        result = _tool_args_preview("task_tracker", {"action": "add", "content": "Fix bug"})
        assert result == "add: Fix bug"

    def test_task_tracker_no_content(self):
        assert _tool_args_preview("task_tracker", {"action": "list"}) == "list"

    def test_ask_user(self):
        assert _tool_args_preview("ask_user", {"question": "Which DB?"}) == "Which DB?"

    def test_analyze_code(self):
        assert _tool_args_preview("analyze_code", {"path": "src/"}) == "src/"

    def test_generic_fallback(self):
        assert _tool_args_preview("unknown", {"x": "hello"}) == "hello"

    def test_empty(self):
        assert _tool_args_preview("unknown", {}) == ""


class TestToolOutputPreview:
    def test_empty(self):
        assert _tool_output_preview("read_file", "") == ""

    def test_read_file(self):
        assert _tool_output_preview("read_file", "line1\nline2\nline3\n") == "3 lines"

    def test_search_no_matches(self):
        assert _tool_output_preview("ripgrep_search", "No matches found.") == "No matches found."

    def test_search_results(self):
        output = "file1.py:10:match\nfile2.py:20:match"
        assert _tool_output_preview("ripgrep_search", output) == "2 results"

    def test_shell(self):
        assert _tool_output_preview("shell_execute", "hello world\nmore output") == "hello world"

    def test_web_search(self):
        output = "1. Result one\n   url\n2. Result two\n   url\n3. Result three\n   url"
        assert "3 results" in _tool_output_preview("web_search", output)

    def test_unknown_tool(self):
        assert _tool_output_preview("unknown", "whatever") == ""


class TestTrunc:
    def test_short(self):
        assert _trunc("hello", 10) == "hello"

    def test_exact(self):
        assert _trunc("hello", 5) == "hello"

    def test_long(self):
        assert _trunc("hello world", 8) == "hello..."


# --- Screen class tests (unit, no Textual app runner) ---


class TestToolApprovalScreen:
    def test_init(self):
        screen = ToolApprovalScreen("shell_execute", "ls -la")
        assert screen._tool_name == "shell_execute"
        assert screen._args_preview == "ls -la"


class TestAskUserScreen:
    def test_init_no_options(self):
        screen = AskUserScreen("What language?")
        assert screen._question == "What language?"
        assert screen._options == []

    def test_init_with_options(self):
        screen = AskUserScreen("Pick one:", ["Python", "Rust"])
        assert screen._question == "Pick one:"
        assert screen._options == ["Python", "Rust"]
