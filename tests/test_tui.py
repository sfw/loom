"""Tests for the TUI app and its components."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from loom.tui.api_client import LoomAPIClient
from loom.tui.screens.approval import ToolApprovalScreen
from loom.tui.screens.ask_user import AskUserScreen
from loom.tui.widgets.tool_call import (
    _escape,
    _style_diff_output,
    _trunc,
    tool_args_preview,
    tool_output_preview,
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
        assert api._client is None

    async def test_list_tasks(self):
        api = LoomAPIClient()
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"task_id": "t1", "status": "running"},
        ]
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
        mock_response.json.return_value = {
            "task_id": "t1", "status": "pending",
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        api._client = mock_client

        result = await api.create_task(
            "Build a CLI", workspace="/tmp/proj",
        )
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
        assert tool_args_preview("read_file", {"path": "foo.py"}) == "foo.py"

    def test_shell(self):
        result = tool_args_preview(
            "shell_execute", {"command": "ls -la"},
        )
        assert result == "ls -la"

    def test_git(self):
        result = tool_args_preview(
            "git_command", {"args": ["push", "origin"]},
        )
        assert result == "push origin"

    def test_ripgrep(self):
        result = tool_args_preview(
            "ripgrep_search", {"pattern": "TODO"},
        )
        assert result == "/TODO/"

    def test_glob(self):
        result = tool_args_preview(
            "glob_find", {"pattern": "**/*.py"},
        )
        assert result == "**/*.py"

    def test_web_search(self):
        result = tool_args_preview(
            "web_search", {"query": "python docs"},
        )
        assert result == "python docs"

    def test_web_fetch(self):
        result = tool_args_preview(
            "web_fetch", {"url": "https://example.com"},
        )
        assert result == "https://example.com"

    def test_web_fetch_html(self):
        result = tool_args_preview(
            "web_fetch_html", {"url": "https://example.com"},
        )
        assert result == "https://example.com"

    def test_task_tracker(self):
        result = tool_args_preview(
            "task_tracker", {"action": "add", "content": "Fix bug"},
        )
        assert result == "add: Fix bug"

    def test_task_tracker_no_content(self):
        result = tool_args_preview("task_tracker", {"action": "list"})
        assert result == "list"

    def test_ask_user(self):
        result = tool_args_preview(
            "ask_user", {"question": "Which DB?"},
        )
        assert result == "Which DB?"

    def test_analyze_code(self):
        result = tool_args_preview("analyze_code", {"path": "src/"})
        assert result == "src/"

    def test_generic_fallback(self):
        assert tool_args_preview("unknown", {"x": "hello"}) == "hello"

    def test_empty(self):
        assert tool_args_preview("unknown", {}) == ""


class TestToolOutputPreview:
    def test_empty(self):
        assert tool_output_preview("read_file", "") == ""

    def test_read_file(self):
        result = tool_output_preview(
            "read_file", "line1\nline2\nline3\n",
        )
        assert result == "3 lines"

    def test_search_no_matches(self):
        result = tool_output_preview(
            "ripgrep_search", "No matches found.",
        )
        assert result == "No matches found."

    def test_search_results(self):
        output = "file1.py:10:match\nfile2.py:20:match"
        assert tool_output_preview("ripgrep_search", output) == "2 results"

    def test_shell(self):
        result = tool_output_preview(
            "shell_execute", "hello world\nmore output",
        )
        assert result == "hello world"

    def test_web_search(self):
        output = (
            "1. Result one\n   url\n"
            "2. Result two\n   url\n"
            "3. Result three\n   url"
        )
        assert "3 results" in tool_output_preview("web_search", output)

    def test_edit_file_summary(self):
        output = "Edited foo.py: replaced 2 lines with 3 lines\n\n--- a/foo.py\n+++ b/foo.py"
        result = tool_output_preview("edit_file", output)
        assert "Edited foo.py" in result
        # Should NOT contain diff markers
        assert "---" not in result

    def test_unknown_tool(self):
        assert tool_output_preview("unknown", "whatever") == ""


class TestTrunc:
    def test_short(self):
        assert _trunc("hello", 10) == "hello"

    def test_exact(self):
        assert _trunc("hello", 5) == "hello"

    def test_long(self):
        assert _trunc("hello world", 8) == "hello..."


class TestEscape:
    def test_escapes_brackets(self):
        assert _escape("list[int]") == "list\\[int]"

    def test_no_brackets(self):
        assert _escape("hello world") == "hello world"


class TestStyleDiffOutput:
    def test_summary_line_is_dim(self):
        output = "Edited foo.py: replaced 1 lines with 1 lines"
        styled = _style_diff_output(output)
        assert "[dim]" in styled

    def test_additions_are_green(self):
        output = "summary\n\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"
        styled = _style_diff_output(output)
        assert "[#9ece6a]" in styled  # green for additions
        assert "[#f7768e]" in styled  # red for removals
        assert "[#7dcfff]" in styled  # cyan for hunk headers

    def test_headers_are_bold(self):
        output = "--- a/foo.py\n+++ b/foo.py"
        styled = _style_diff_output(output)
        assert "[bold]" in styled

    def test_brackets_in_code_escaped(self):
        output = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x: list[int]\n+x: list[str]"
        styled = _style_diff_output(output)
        # Brackets should be escaped to prevent Rich markup interpretation
        assert "\\[int]" in styled
        assert "\\[str]" in styled


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


class TestExitConfirmScreen:
    def test_init(self):
        from loom.tui.screens.confirm_exit import ExitConfirmScreen

        screen = ExitConfirmScreen()
        assert screen is not None

    def test_on_key_confirm_paths(self):
        from loom.tui.screens.confirm_exit import ExitConfirmScreen

        screen = ExitConfirmScreen()
        dismissed = []
        screen.dismiss = lambda value: dismissed.append(value)

        for key in ("y", "enter", "ctrl+c"):
            event = MagicMock()
            event.key = key
            screen.on_key(event)
            assert dismissed[-1] is True
            event.stop.assert_called_once()
            event.prevent_default.assert_called_once()

    def test_on_key_cancel_paths(self):
        from loom.tui.screens.confirm_exit import ExitConfirmScreen

        screen = ExitConfirmScreen()
        dismissed = []
        screen.dismiss = lambda value: dismissed.append(value)

        for key in ("n", "escape"):
            event = MagicMock()
            event.key = key
            screen.on_key(event)
            assert dismissed[-1] is False
            event.stop.assert_called_once()
            event.prevent_default.assert_called_once()


# --- Theme tests ---


class TestTheme:
    def test_loom_dark_theme(self):
        from loom.tui.theme import LOOM_DARK
        assert LOOM_DARK.name == "loom-dark"
        assert LOOM_DARK.dark is True
        assert LOOM_DARK.primary == "#7dcfff"

    def test_color_constants(self):
        from loom.tui.theme import (
            ACCENT_CYAN,
            TOOL_ERR,
            TOOL_OK,
            USER_MSG,
        )
        assert USER_MSG == "#73daca"
        assert TOOL_OK == "#9ece6a"
        assert TOOL_ERR == "#f7768e"
        assert ACCENT_CYAN == "#7dcfff"


# --- Widget tests ---


class TestStatusBar:
    def test_default_state(self):
        from loom.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        assert bar.state == "Ready"
        assert bar.total_tokens == 0

    def test_render_includes_process_name(self):
        from loom.tui.widgets.status_bar import StatusBar

        bar = StatusBar()
        bar.workspace_name = "loom"
        bar.model_name = "primary"
        bar.process_name = "marketing-strategy"
        bar.total_tokens = 12

        rendered = bar.render()
        assert "process:marketing-strategy" in rendered
        assert "12 tokens" in rendered


class TestTaskProgressPanel:
    def test_render_empty(self):
        from loom.tui.widgets.sidebar import TaskProgressPanel
        panel = TaskProgressPanel()
        assert "No tasks tracked" in panel.render()

    def test_render_with_tasks(self):
        from loom.tui.widgets.sidebar import TaskProgressPanel
        panel = TaskProgressPanel()
        panel.tasks = [
            {"content": "Read file", "status": "completed"},
            {"content": "Fix bug", "status": "in_progress"},
            {"content": "Run tests", "status": "pending"},
            {"content": "Handle failure", "status": "failed"},
            {"content": "Skip optional step", "status": "skipped"},
        ]
        rendered = panel.render()
        assert "Read file" in rendered
        assert "Fix bug" in rendered
        assert "Run tests" in rendered
        assert "Handle failure" in rendered
        assert "Skip optional step" in rendered

    def test_task_update_triggers_scroll_hook(self):
        from loom.tui.widgets.sidebar import TaskProgressPanel

        panel = TaskProgressPanel(auto_follow=True)
        panel._scroll_to_latest = MagicMock()
        panel.tasks = [{"content": "Read file", "status": "completed"}]
        assert panel._scroll_to_latest.call_count >= 1

    def test_empty_message_update_triggers_scroll_hook(self):
        from loom.tui.widgets.sidebar import TaskProgressPanel

        panel = TaskProgressPanel(auto_follow=True)
        panel._scroll_to_latest = MagicMock()
        panel.empty_message = "No outputs yet"
        assert panel._scroll_to_latest.call_count >= 1


class TestProcessRunPane:
    def test_process_run_panels_enable_auto_follow(self):
        from loom.tui.app import ProcessRunPane

        pane = ProcessRunPane(
            run_id="abc123",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
        )
        assert pane._progress._auto_follow is True
        assert pane._progress._follow_mode == "active"
        assert pane._outputs._auto_follow is True
        assert pane._outputs._follow_mode == "active"

    def test_restart_button_shown_only_for_failed_run_with_task(self):
        from loom.tui.app import ProcessRunPane

        pane = ProcessRunPane(
            run_id="abc123",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
        )
        pane.set_status_header(status="running", elapsed="0:01", task_id="cowork-1")
        assert pane._actions.display is False
        assert pane._restart_button.display is False
        assert pane._restart_button.disabled is True

        pane.set_status_header(status="failed", elapsed="0:42", task_id="")
        assert pane._actions.display is False
        assert pane._restart_button.display is False
        assert pane._restart_button.disabled is True

        pane.set_status_header(status="failed", elapsed="0:42", task_id="cowork-1")
        assert pane._actions.display is True
        assert pane._restart_button.display is True
        assert pane._restart_button.disabled is False

    def test_process_run_rows_render_with_fold_overflow(self):
        from rich.text import Text

        from loom.tui.app import ProcessRunList

        panel = ProcessRunList(empty_message="No progress yet")
        panel._rows = [{
            "status": "completed",
            "content": (
                "Generate a high-volume longlist of slogan/tagline options across all "
                "territories and devices before filtering."
            ),
        }]

        rendered = panel._render_rows()
        assert isinstance(rendered, Text)
        assert rendered.no_wrap is False
        assert rendered.overflow == "fold"
        assert "longlist of slogan/tagline options" in rendered.plain

    def test_process_run_rows_escape_markup_like_content(self):
        from loom.tui.app import ProcessRunList

        panel = ProcessRunList(empty_message="No progress yet")
        panel._rows = [{
            "status": "completed",
            "content": "Verifier note ...[truncated]... follow-up",
        }]

        rendered = panel._render_rows()
        assert "[truncated]" in rendered.plain

    def test_process_run_result_coerces_rich_text_to_plain(self):
        from rich.text import Text

        from loom.tui.app import ProcessRunPane

        pane = ProcessRunPane(
            run_id="abc123",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
        )
        pane._log = MagicMock()

        pane.add_result(Text.from_markup("[dim]ok[/dim]"), success=True)

        assert pane._pending_results == [("ok", True)]

    def test_process_run_active_follow_waits_for_started_rows(self):
        from unittest.mock import PropertyMock, patch

        from loom.tui.app import ProcessRunList

        panel = ProcessRunList(
            empty_message="No outputs yet",
            auto_follow=True,
            follow_mode="active",
        )
        panel._rows = [
            {"status": "pending", "content": "slogan-longlist.csv"},
            {"status": "pending", "content": "shortlist-scorecard.csv"},
        ]
        panel.call_after_refresh = MagicMock()

        with patch.object(
            ProcessRunList,
            "is_attached",
            new_callable=PropertyMock,
            return_value=True,
        ):
            panel._scroll_to_latest()

        panel.call_after_refresh.assert_not_called()

    def test_process_run_active_follow_targets_current_output(self):
        from unittest.mock import PropertyMock, patch

        from loom.tui.app import ProcessRunList

        panel = ProcessRunList(
            empty_message="No outputs yet",
            auto_follow=True,
            follow_mode="active",
        )
        panel._rows = [
            {"status": "pending", "content": "brief-normalized.md"},
            {"status": "completed", "content": "brief-assumptions.md"},
            {"status": "completed", "content": "tension-map.csv"},
            {"status": "in_progress", "content": "insight-angles.md"},
            {"status": "pending", "content": "signal-board.md"},
        ]
        panel.scroll_to = MagicMock()

        def _run_immediately(callback, *_args, **_kwargs):
            callback()

        panel.call_after_refresh = MagicMock(side_effect=_run_immediately)

        with patch.object(
            ProcessRunList,
            "is_attached",
            new_callable=PropertyMock,
            return_value=True,
        ):
            panel._scroll_to_latest()

        panel.scroll_to.assert_called_once_with(y=1, animate=False, force=True)


class TestSidebarWidget:
    def test_refresh_workspace_tree_calls_reload(self):
        from loom.tui.widgets.sidebar import Sidebar

        sidebar = Sidebar(workspace=Path("/tmp"))
        tree = MagicMock()
        sidebar.query_one = MagicMock(return_value=tree)

        sidebar.refresh_workspace_tree()

        tree.reload.assert_called_once()


class TestChatLogStreaming:
    def test_flush_stream_buffer_uses_internal_text(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        widget = MagicMock()
        log._stream_widget = widget
        log._stream_text = "hello"
        log._stream_buffer = [" ", "world"]

        log._flush_stream_buffer()

        widget.update.assert_called_once_with("hello world")
        assert log._stream_text == "hello world"
        assert log._stream_buffer == []

    def test_flush_and_reset_stream_clears_state(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        widget = MagicMock()
        log._stream_widget = widget
        log._stream_text = "chunk"
        log._stream_buffer = ["!"]

        log._flush_and_reset_stream()

        widget.update.assert_called_once_with("chunk!")
        assert log._stream_widget is None
        assert log._stream_text == ""
        assert log._stream_buffer == []

    def test_static_messages_expand_to_available_width(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_user_message("hello")
        log.add_model_text("world")
        log.add_info("info")
        log.add_turn_separator(tool_count=1, tokens=42, model="test-model")

        assert mounted
        assert all(getattr(widget, "expand", False) for widget in mounted)

    def test_streaming_widget_expands_to_available_width(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_streaming_text("hello")

        assert log._stream_widget is not None
        assert log._stream_widget.expand is True
        assert mounted == [log._stream_widget]


# --- CoworkSession total_tokens tests ---


class TestCoworkSessionTokens:
    def test_initial_total_tokens(self):
        from unittest.mock import MagicMock

        from loom.cowork.session import CoworkSession

        model = MagicMock()
        model.name = "test-model"
        tools = MagicMock()
        tools.all_schemas.return_value = []

        session = CoworkSession(model=model, tools=tools)
        assert session.total_tokens == 0


# --- Sad-path: _init_persistence -------------------------------------------


class TestInitPersistence:
    """Test graceful degradation when database init fails."""

    def test_returns_none_on_db_error(self, tmp_path):
        """If the db path is unwritable, return (None, None)."""
        from loom.__main__ import _init_persistence
        from loom.config import Config, MemoryConfig

        # Point database_path at a path with a null byte (invalid)
        cfg = Config(
            memory=MemoryConfig(
                database_path=str(tmp_path / "no" / "such" / "parent" / "\0invalid" / "loom.db"),
            ),
        )
        db, store = _init_persistence(cfg)
        assert db is None
        assert store is None

    def test_returns_valid_on_success(self, tmp_path):
        """Sanity: happy path still works."""
        from loom.__main__ import _init_persistence
        from loom.config import Config, MemoryConfig

        cfg = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "loom.db")),
        )
        db, store = _init_persistence(cfg)
        assert db is not None
        assert store is not None


# --- Sad-path: delegate_task unbound ----------------------------------------


class TestDelegateTaskUnbound:
    """delegate_task returns a clean error when not bound."""

    @pytest.mark.asyncio
    async def test_unbound_returns_not_available(self):
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        tool = DelegateTaskTool()
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({"goal": "do something"}, ctx)
        assert not result.success
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_missing_goal(self):
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        async def _factory():
            return MagicMock()

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({}, ctx)
        assert not result.success
        assert "goal" in result.error

    @pytest.mark.asyncio
    async def test_success_includes_subtask_progress_payload(self):
        from loom.state.task_state import (
            Plan,
            Subtask,
            SubtaskStatus,
            Task,
            TaskStatus,
        )
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        task = Task(
            id="task-123",
            goal="Analyze Tesla",
            workspace="/tmp",
            status=TaskStatus.COMPLETED,
        )
        task.plan = Plan(subtasks=[
            Subtask(
                id="company-screening",
                description="Company screening",
                status=SubtaskStatus.COMPLETED,
            ),
            Subtask(
                id="financial-analysis",
                description="Financial analysis",
                status=SubtaskStatus.RUNNING,
            ),
            Subtask(
                id="memo",
                description="Memo",
                status=SubtaskStatus.SKIPPED,
            ),
        ])

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(return_value=task)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert isinstance(result.data, dict)
        assert result.data["task_id"] == "task-123"
        assert result.data["tasks"][0]["status"] == "completed"
        assert result.data["tasks"][1]["status"] == "in_progress"
        assert result.data["tasks"][2]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_failed_task_returns_failure_and_task_failed_progress(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        async def _execute(task):
            task.status = TaskStatus.FAILED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        progress: list[dict] = []
        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {"goal": "Analyze Tesla", "_progress_callback": progress.append},
            ctx,
        )

        assert result.success is False
        assert result.error == "Task execution failed."
        assert isinstance(result.data, dict)
        assert result.data.get("status") == "failed"
        assert any(p.get("event_type") == "task_failed" for p in progress)

    @pytest.mark.asyncio
    async def test_delegate_logs_detailed_event_stream_when_configured(self, tmp_path):
        from loom.config import Config, LoggingConfig
        from loom.events.bus import Event, EventBus
        from loom.events.types import TASK_EXECUTING
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        bus = EventBus()

        async def _execute(task):
            task.status = TaskStatus.EXECUTING
            bus.emit(Event(
                event_type=TASK_EXECUTING,
                task_id=task.id,
                data={"note": "running"},
            ))
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._events = bus
            orchestrator._config = Config(
                logging=LoggingConfig(
                    event_log_path=str(tmp_path / "loom-logs"),
                )
            )
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert isinstance(result.data, dict)
        event_log_path = str(result.data.get("event_log_path", "")).strip()
        assert event_log_path
        log_path = Path(event_log_path)
        assert log_path.exists()

        records = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(record.get("kind") == "meta" for record in records)
        assert any(record.get("kind") == "event_bus" for record in records)
        assert any(record.get("kind") == "progress_snapshot" for record in records)

    @pytest.mark.asyncio
    async def test_delegate_ignores_non_pathlike_log_config(self, tmp_path, monkeypatch):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        monkeypatch.chdir(tmp_path)

        async def _execute(task):
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._config = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert isinstance(result.data, dict)
        assert "event_log_path" not in result.data
        assert not list(tmp_path.glob("<MagicMock*"))

    @pytest.mark.asyncio
    async def test_progress_callback_includes_event_metadata(self):
        from loom.events.bus import Event, EventBus
        from loom.events.types import SUBTASK_STARTED
        from loom.state.task_state import Plan, Subtask, SubtaskStatus, TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        bus = EventBus()

        async def _execute(_task):
            _task.status = TaskStatus.EXECUTING
            _task.plan = Plan(subtasks=[
                Subtask(
                    id="company-screening",
                    description="Company screening",
                    status=SubtaskStatus.RUNNING,
                ),
            ])
            bus.emit(Event(
                event_type=SUBTASK_STARTED,
                task_id=_task.id,
                data={"subtask_id": "company-screening"},
            ))
            _task.plan.subtasks[0].status = SubtaskStatus.COMPLETED
            _task.status = TaskStatus.COMPLETED
            return _task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._events = bus
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        progress: list[dict] = []
        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))

        result = await tool.execute(
            {"goal": "Analyze Tesla", "_progress_callback": progress.append},
            ctx,
        )

        assert result.success is True
        events = [p for p in progress if p.get("event_type") == SUBTASK_STARTED]
        assert events, "Expected at least one streamed progress event"
        payload = events[0]
        assert payload["event_data"]["subtask_id"] == "company-screening"

    @pytest.mark.asyncio
    async def test_delegate_creates_fresh_orchestrator_per_execute(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        created: list[MagicMock] = []

        async def _execute(task):
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            created.append(orchestrator)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))

        first = await tool.execute({"goal": "first"}, ctx)
        second = await tool.execute({"goal": "second"}, ctx)

        assert first.success is True
        assert second.success is True
        assert len(created) == 2
        assert created[0] is not created[1]
        created[0].execute_task.assert_awaited_once()
        created[1].execute_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delegate_applies_internal_approval_mode_override(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        observed_modes: list[str] = []

        async def _execute(task):
            observed_modes.append(task.approval_mode)
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))

        result = await tool.execute(
            {"goal": "Analyze Tesla", "_approval_mode": "disabled"},
            ctx,
        )

        assert result.success is True
        assert observed_modes == ["disabled"]

    @pytest.mark.asyncio
    async def test_delegate_resume_reuses_plan_and_resets_incomplete_subtasks(self):
        from loom.state.task_state import Plan, Subtask, SubtaskStatus, Task, TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        task = Task(
            id="cowork-1234",
            goal="Original goal",
            workspace="/tmp",
            status=TaskStatus.FAILED,
            context={"existing": "ctx"},
            plan=Plan(subtasks=[
                Subtask(
                    id="done",
                    description="already done",
                    status=SubtaskStatus.COMPLETED,
                ),
                Subtask(
                    id="retry-me",
                    description="retry this",
                    status=SubtaskStatus.FAILED,
                    summary="failed before",
                    active_issue="timeout",
                    retry_count=2,
                ),
                Subtask(
                    id="blocked",
                    description="blocked by failure",
                    status=SubtaskStatus.SKIPPED,
                    summary="Skipped: blocked",
                ),
            ]),
        )

        observed: dict = {}

        async def _execute(resume_task, **kwargs):
            observed["reuse_existing_plan"] = kwargs.get("reuse_existing_plan")
            observed["statuses"] = [subtask.status for subtask in resume_task.plan.subtasks]
            resume_task.status = TaskStatus.COMPLETED
            return resume_task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._state = MagicMock()
            orchestrator._state.load = MagicMock(return_value=task)
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {
                "goal": "Updated goal",
                "context": {"new": "value"},
                "_resume_task_id": "cowork-1234",
            },
            ctx,
        )

        assert result.success is True
        assert observed["reuse_existing_plan"] is True
        assert observed["statuses"] == [
            SubtaskStatus.COMPLETED,
            SubtaskStatus.PENDING,
            SubtaskStatus.PENDING,
        ]
        assert task.goal == "Updated goal"
        assert task.context["existing"] == "ctx"
        assert task.context["new"] == "value"

    @pytest.mark.asyncio
    async def test_delegate_resume_rejects_completed_task(self):
        from loom.state.task_state import Task, TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        task = Task(
            id="cowork-done",
            goal="Completed",
            workspace="/tmp",
            status=TaskStatus.COMPLETED,
        )

        async def _factory():
            orchestrator = MagicMock()
            orchestrator._state = MagicMock()
            orchestrator._state.load = MagicMock(return_value=task)
            orchestrator.execute_task = AsyncMock()
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {"goal": "Completed", "_resume_task_id": "cowork-done"},
            ctx,
        )

        assert result.success is False
        assert "already completed" in (result.error or "").lower()

    def test_delegate_timeout_default(self, monkeypatch):
        from loom.tools.delegate_task import DelegateTaskTool

        monkeypatch.delenv("LOOM_DELEGATE_TIMEOUT_SECONDS", raising=False)
        tool = DelegateTaskTool()
        assert tool.timeout_seconds == 3600

    def test_delegate_timeout_env_override(self, monkeypatch):
        from loom.tools.delegate_task import DelegateTaskTool

        monkeypatch.setenv("LOOM_DELEGATE_TIMEOUT_SECONDS", "7200")
        tool = DelegateTaskTool()
        assert tool.timeout_seconds == 7200

    def test_delegate_timeout_constructor_override(self, monkeypatch):
        from loom.tools.delegate_task import DelegateTaskTool

        monkeypatch.delenv("LOOM_DELEGATE_TIMEOUT_SECONDS", raising=False)
        tool = DelegateTaskTool(timeout_seconds=5400)
        assert tool.timeout_seconds == 5400


# --- Sad-path: session resume failures --------------------------------------


class TestSessionResumeFallback:
    """Verify CoworkSession.resume raises on bad IDs."""

    @pytest.mark.asyncio
    async def test_resume_nonexistent_session(self, tmp_path):
        """Resuming a session that doesn't exist should raise."""
        from loom.cowork.session import CoworkSession
        from loom.state.conversation_store import ConversationStore
        from loom.state.memory import Database

        db = Database(tmp_path / "test.db")
        await db.initialize()
        store = ConversationStore(db)

        model = MagicMock()
        model.name = "test"
        tools = MagicMock()
        tools.all_schemas.return_value = []

        session = CoworkSession(
            model=model, tools=tools, store=store,
        )
        with pytest.raises(Exception):
            await session.resume("nonexistent-session-id")

    @pytest.mark.asyncio
    async def test_resume_without_store_raises(self):
        """Resuming with no store should raise."""
        from loom.cowork.session import CoworkSession

        model = MagicMock()
        model.name = "test"
        tools = MagicMock()
        tools.all_schemas.return_value = []

        session = CoworkSession(model=model, tools=tools)
        with pytest.raises(Exception):
            await session.resume("any-id")


# --- Sad-path: CLI --resume with no DB -------------------------------------


class TestCLIResumeNoDB:
    """--resume should fail cleanly when persistence is unavailable."""

    def test_resume_flag_without_db(self, monkeypatch):
        """If _init_persistence returns None, --resume should exit 1."""
        from click.testing import CliRunner

        import loom.__main__ as main_mod
        from loom.__main__ import cli

        monkeypatch.setattr(main_mod, "_init_persistence", lambda cfg: (None, None))
        # Mock model resolution so we don't need a real config
        monkeypatch.setattr(
            main_mod, "_resolve_model", lambda cfg, name: MagicMock(name="mock"),
        )

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--resume", "abc123"],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        assert "requires database" in result.output


# --- P0-2: Input event guard ------------------------------------------------


class TestInputSubmitGuard:
    """P0-2: App should only handle Input.Submitted from #user-input."""

    def test_on_user_submit_uses_selector(self):
        """Verify the handler is decorated with the '#user-input' selector."""
        from loom.tui.app import LoomApp

        bindings = getattr(LoomApp.on_user_submit, "_textual_on", [])
        assert bindings, "Expected textual on-handler metadata"
        _event_cls, selector_spec = bindings[0]
        control = selector_spec.get("control", ())
        assert control
        assert "#user-input" in str(control[0])

    def test_setup_screen_stops_input_submitted(self):
        """SetupScreen.on_input_submitted should call event.stop()."""
        from unittest.mock import MagicMock

        from loom.tui.screens.setup import _STEP_DETAILS, SetupScreen

        screen = SetupScreen()
        # Mock _show_step to prevent reactive watcher from doing DOM ops
        screen._show_step = MagicMock()
        screen._step = _STEP_DETAILS
        screen._provider_key = "ollama"

        event = MagicMock()
        event.input.id = "input-model"

        # Mock query_one for the model input focus
        screen.query_one = MagicMock()

        screen.on_input_submitted(event)
        event.stop.assert_called_once()


# --- P1-5: Session/model mismatch on re-setup --------------------------------


class TestFinalizeSetupResetsSession:
    """P1-5: Re-running /setup should invalidate the old session."""

    @pytest.mark.asyncio
    async def test_finalize_setup_clears_session(self, monkeypatch):
        """_finalize_setup should invalidate old session and reinitialize."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=None,
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._initialize_session = AsyncMock()

        fake_config = MagicMock()
        fake_model = MagicMock()
        fake_router = MagicMock()
        fake_router.select.return_value = fake_model

        monkeypatch.setattr("loom.config.load_config", lambda: fake_config)
        monkeypatch.setattr(
            "loom.models.router.ModelRouter.from_config",
            lambda cfg: fake_router,
        )

        # Call the undecorated coroutine behind @work.
        await LoomApp._finalize_setup.__wrapped__(app)

        assert app._config is fake_config
        assert app._model is fake_model
        app._store.update_session.assert_awaited_once_with(
            "old-session", is_active=False,
        )
        app._initialize_session.assert_awaited_once()


class TestFilesPanelReset:
    def test_clear_files_panel(self):
        """Session changes should reset the files panel and clear the diff."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=None,
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        panel = MagicMock()
        app.query_one = MagicMock(return_value=panel)

        app._clear_files_panel()

        panel.clear_files.assert_called_once()
        panel.show_diff.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_new_session_clears_files_panel(self):
        """Creating a new session should clear stale file history."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._model = MagicMock()
        app._model.name = "test-model"
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._store.create_session = AsyncMock(return_value="new-session")
        app._bind_session_tools = MagicMock()
        panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return panel
            return chat

        app.query_one = MagicMock(side_effect=_query_one)

        await app._new_session()

        panel.clear_files.assert_called_once()
        panel.show_diff.assert_called_once_with("")


class TestWorkspaceRefresh:
    def test_update_files_panel_refreshes_sidebar_tree(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        files_panel = MagicMock()
        sidebar = MagicMock()
        app.notify = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return files_panel
            if selector == "#sidebar":
                return sidebar
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        turn = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    name="write_file",
                    args={"path": "new-file.md"},
                    result=SimpleNamespace(success=True, output="ok"),
                ),
            ],
        )

        app._update_files_panel(turn)

        files_panel.update_files.assert_called_once()
        sidebar.refresh_workspace_tree.assert_called_once()

    def test_update_files_panel_refreshes_tree_for_document_write(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        files_panel = MagicMock()
        sidebar = MagicMock()
        app.notify = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return files_panel
            if selector == "#sidebar":
                return sidebar
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        turn = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    name="document_write",
                    args={"path": "memo.md"},
                    result=SimpleNamespace(success=True, output="ok"),
                ),
            ],
        )

        app._update_files_panel(turn)

        files_panel.update_files.assert_not_called()
        sidebar.refresh_workspace_tree.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_session_clears_files_panel(self, monkeypatch):
        """Switching sessions should clear stale file history."""
        from loom.tui.app import LoomApp

        class FakeSession:
            def __init__(self, *args, **kwargs):
                self.session_id = ""
                self.session_state = SimpleNamespace(turn_count=3)
                self.total_tokens = 99
                self.workspace = Path("/tmp")

            async def resume(self, session_id: str):
                self.session_id = session_id

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._model = MagicMock()
        app._model.name = "test-model"
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._bind_session_tools = MagicMock()
        panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return panel
            return chat

        app.query_one = MagicMock(side_effect=_query_one)

        await app._switch_to_session("new-session")

        panel.clear_files.assert_called_once()
        panel.show_diff.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_new_session_uses_process_system_prompt(self):
        """Creating a new session should preserve active process prompt text."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._model = MagicMock()
        app._model.name = "test-model"
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._store.create_session = AsyncMock(return_value="new-session")
        app._bind_session_tools = MagicMock()
        app._process_defn = SimpleNamespace(
            persona="You are a process specialist.",
            tool_guidance="Use process tool guidance.",
        )
        panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return panel
            return chat

        app.query_one = MagicMock(side_effect=_query_one)

        await app._new_session()

        create_kwargs = app._store.create_session.await_args.kwargs
        prompt = create_kwargs["system_prompt"]
        assert "DOMAIN ROLE" in prompt
        assert "process specialist" in prompt
        assert "DOMAIN TOOL GUIDANCE" in prompt

    @pytest.mark.asyncio
    async def test_switch_session_uses_process_system_prompt(self, monkeypatch):
        """Switching sessions should keep process prompt extensions."""
        from loom.tui.app import LoomApp

        captured_prompt: dict[str, str] = {}

        class FakeSession:
            def __init__(self, *args, **kwargs):
                captured_prompt["value"] = kwargs["system_prompt"]
                self.session_id = ""
                self.session_state = SimpleNamespace(turn_count=3)
                self.total_tokens = 99
                self.workspace = Path("/tmp")

            async def resume(self, session_id: str):
                self.session_id = session_id

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._model = MagicMock()
        app._model.name = "test-model"
        app._session = SimpleNamespace(session_id="old-session")
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._bind_session_tools = MagicMock()
        app._process_defn = SimpleNamespace(
            persona="You are a process specialist.",
            tool_guidance="Use process tool guidance.",
        )
        panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#files-panel":
                return panel
            return chat

        app.query_one = MagicMock(side_effect=_query_one)

        await app._switch_to_session("new-session")

        prompt = captured_prompt["value"]
        assert "DOMAIN ROLE" in prompt
        assert "process specialist" in prompt
        assert "DOMAIN TOOL GUIDANCE" in prompt


class TestDelegateBindingProcess:
    @pytest.mark.asyncio
    async def test_delegate_orchestrator_factory_includes_process(self, monkeypatch):
        """delegate_task should spawn process-aware orchestrators when active."""
        from loom.tools.conversation_recall import ConversationRecallTool
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolRegistry
        from loom.tui.app import LoomApp

        captured: dict[str, object] = {}

        class FakeOrchestrator:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("loom.engine.orchestrator.Orchestrator", FakeOrchestrator)
        monkeypatch.setattr("loom.events.bus.EventBus", lambda: MagicMock())
        monkeypatch.setattr(
            "loom.models.router.ModelRouter.from_config",
            lambda _cfg: MagicMock(),
        )
        monkeypatch.setattr("loom.prompts.assembler.PromptAssembler", lambda: MagicMock())
        monkeypatch.setattr("loom.state.memory.MemoryManager", lambda _db: MagicMock())
        monkeypatch.setattr("loom.state.task_state.TaskStateManager", lambda _dir: MagicMock())
        monkeypatch.setattr(
            "loom.tools.create_default_registry",
            lambda _config=None: MagicMock(),
        )

        registry = ToolRegistry()
        registered_delegate = DelegateTaskTool()
        registry.register(ConversationRecallTool())
        registry.register(registered_delegate)

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=registry,
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._ensure_persistence_tools()
        app._session = SimpleNamespace(
            session_id="session-id",
            session_state=SimpleNamespace(),
        )
        app._config = SimpleNamespace(
            workspace=SimpleNamespace(scratch_dir="/tmp"),
        )
        app._db = MagicMock()
        app._process_defn = SimpleNamespace(
            name="marketing-strategy",
            tools=SimpleNamespace(excluded=[]),
        )

        assert app._delegate_tool is registered_delegate
        app._bind_session_tools()

        assert app._delegate_tool._factory is not None
        await app._delegate_tool._factory()
        assert captured["process"] is app._process_defn

    def test_apply_process_tool_policy_reports_missing_required_tools(self):
        """Missing required process tools should produce a visible warning."""
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        registry = MagicMock()
        registry.has.return_value = False
        app._tools = registry
        app._process_defn = SimpleNamespace(
            tools=SimpleNamespace(excluded=[], required=["missing-tool"]),
        )
        chat = MagicMock()

        app._apply_process_tool_policy(chat)

        chat.add_info.assert_called_once()
        assert "missing-tool" in chat.add_info.call_args.args[0]


class TestStartupSessionResume:
    @pytest.mark.asyncio
    async def test_resolve_startup_resume_target_prefers_explicit(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            resume_session="explicit-session",
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock()

        session_id, auto = await app._resolve_startup_resume_target()

        assert session_id == "explicit-session"
        assert auto is False
        app._store.list_sessions.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_startup_resume_target_uses_latest_workspace(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[
            {"id": "latest-session"},
            {"id": "older-session"},
        ])

        session_id, auto = await app._resolve_startup_resume_target()

        assert session_id == "latest-session"
        assert auto is True
        app._store.list_sessions.assert_awaited_once_with(workspace="/tmp")

    @pytest.mark.asyncio
    async def test_resolve_startup_resume_target_disabled_after_initial_start(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[
            {"id": "latest-session"},
        ])
        app._auto_resume_workspace_on_init = False

        session_id, auto = await app._resolve_startup_resume_target()

        assert session_id is None
        assert auto is False
        app._store.list_sessions.assert_not_called()


class TestQuitConfirmation:
    @pytest.mark.asyncio
    async def test_action_quit_confirmed_persists_and_exits(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._confirm_exit = AsyncMock(return_value=True)
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._session = SimpleNamespace(session_id="sess-123")
        app.exit = MagicMock()

        await app.action_quit()

        app._confirm_exit.assert_awaited_once()
        app._store.update_session.assert_awaited_once_with(
            "sess-123", is_active=False,
        )
        app.exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_quit_cancelled_does_not_exit(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._confirm_exit = AsyncMock(return_value=False)
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._session = SimpleNamespace(session_id="sess-123")
        app.exit = MagicMock()

        await app.action_quit()

        app._confirm_exit.assert_awaited_once()
        app._store.update_session.assert_not_called()
        app.exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_slash_quit_routes_through_request_exit(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.action_request_quit = MagicMock()
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/quit")

        assert handled is True
        app.action_request_quit.assert_called_once()

    def test_action_request_quit_starts_worker(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        captured: dict = {}

        def fake_run_worker(coro, **kwargs):
            captured["kwargs"] = kwargs
            coro.close()
            return MagicMock()

        app.run_worker = fake_run_worker
        app.action_request_quit()

        assert captured["kwargs"]["group"] == "exit-flow"
        assert captured["kwargs"]["exclusive"] is True

    @pytest.mark.asyncio
    async def test_ctrl_c_modal_accepts_y_and_exits(self):
        from loom.tui.app import LoomApp
        from loom.tui.screens.confirm_exit import ExitConfirmScreen

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()
        app.exit = MagicMock()

        async with app.run_test() as pilot:
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], ExitConfirmScreen)

            await pilot.press("y")
            await pilot.pause()
            assert app.exit.called

    @pytest.mark.asyncio
    async def test_confirm_exit_reentrant_uses_single_modal(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        callbacks = []
        app.push_screen = lambda _screen, callback: callbacks.append(callback)

        first = asyncio.create_task(app._confirm_exit())
        await asyncio.sleep(0)
        second = asyncio.create_task(app._confirm_exit())
        await asyncio.sleep(0)

        assert len(callbacks) == 1
        callbacks[0](True)

        assert await first is True
        assert await second is True


class TestSlashCommandHints:
    def test_root_slash_shows_catalog(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/")
        assert "Slash commands:" in hint
        assert "/quit" in hint
        assert "/mcp" in hint
        assert "/setup" in hint

    def test_root_slash_includes_dynamic_process_commands(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[
                {"name": "investment-analysis", "version": "1.0"},
                {"name": "marketing-strategy", "version": "1.2"},
            ])
        ))

        hint = app._render_slash_hint("/")

        assert "/investment-analysis <goal>" in hint
        assert "/marketing-strategy <goal>" in hint

    def test_prefix_filters_matches(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/res")
        assert "Matching /res:" in hint
        assert "/resume" in hint
        assert "/setup" not in hint

    def test_prefix_h_matches_help(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/h")
        assert "Matching /h:" in hint
        assert "/help" in hint
        assert "/new" not in hint

    def test_prefix_n_matches_new(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/n")
        assert "Matching /n:" in hint
        assert "/new" in hint

    def test_prefix_p_matches_process(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/p")
        assert "Matching /p:" in hint
        assert "/process" in hint

    def test_process_use_hint_shows_available_processes(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[
                {"name": "investment-analysis", "version": "1.0"},
                {"name": "marketing-strategy", "version": "1.2"},
            ])
        ))
        hint = app._render_slash_hint("/process use")
        assert "Available processes for /process use:" in hint
        assert "investment-analysis" in hint
        assert "marketing-strategy" in hint

    def test_process_use_hint_filters_by_prefix(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[
                {"name": "investment-analysis", "version": "1.0"},
                {"name": "marketing-strategy", "version": "1.2"},
            ])
        ))
        hint = app._render_slash_hint("/process use inv")
        assert "Process matches 'inv':" in hint
        assert "investment-analysis" in hint
        assert "marketing-strategy" not in hint

    def test_prefix_r_matches_resume_and_run(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/r")
        assert "Matching /r:" in hint
        assert "/resume" in hint
        assert "/run" in hint

    def test_prefix_l_matches_learned(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/l")
        assert "Matching /l:" in hint
        assert "/learned" in hint

    def test_alias_prefix_matches_canonical(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/e")
        assert "Matching /e:" in hint
        assert "/quit" in hint
        assert "/exit" in hint

    def test_no_match_shows_help(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/zzz")
        assert "No command matches" in hint
        assert "/help" in hint

    def test_input_changed_uses_live_widget_value(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        user_input = SimpleNamespace(value="/h")
        app.query_one = MagicMock(return_value=user_input)

        captured = []
        app._set_slash_hint = lambda text: captured.append(text)

        stale_event = SimpleNamespace(value="/")
        app.on_user_input_changed(stale_event)

        assert captured
        assert "/help" in captured[-1]

    def test_set_slash_hint_sets_height_from_line_count(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        fake_hint = SimpleNamespace(
            display=False,
            styles=SimpleNamespace(height=None),
            update=MagicMock(),
            scroll_home=MagicMock(),
        )
        fake_footer = SimpleNamespace(display=True)
        fake_status = SimpleNamespace(display=True)

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#slash-hint":
                return fake_hint
            if selector == "#status-bar":
                return fake_status
            return fake_footer

        app.query_one = MagicMock(side_effect=_query_one)

        app._set_slash_hint("a\nb\nc")

        assert fake_hint.display is True
        assert fake_hint.styles.height == 3
        fake_hint.scroll_home.assert_called_once_with(animate=False)
        # Slash hints no longer toggle footer/status visibility.
        assert fake_footer.display is True
        assert fake_status.display is True

    def test_set_slash_hint_empty_resets_auto_height(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        fake_hint = SimpleNamespace(
            display=True,
            styles=SimpleNamespace(height=5),
            update=MagicMock(),
            scroll_home=MagicMock(),
        )
        fake_footer = SimpleNamespace(display=False)
        fake_status = SimpleNamespace(display=False)

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#slash-hint":
                return fake_hint
            if selector == "#status-bar":
                return fake_status
            return fake_footer

        app.query_one = MagicMock(side_effect=_query_one)

        app._set_slash_hint("")

        assert fake_hint.display is False
        assert fake_hint.styles.height == "auto"
        # Slash hints no longer toggle footer/status visibility.
        assert fake_footer.display is False
        assert fake_status.display is False

    def test_slash_completion_candidates_prefix(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[])
        ))
        assert app._slash_completion_candidates("/s") == [
            "/setup",
            "/session",
            "/sessions",
        ]
        assert app._slash_completion_candidates("/h") == ["/help"]
        assert app._slash_completion_candidates("/m") == ["/model", "/mcp"]
        assert app._slash_completion_candidates("/t") == ["/tools", "/tokens"]
        assert app._slash_completion_candidates("/p") == ["/process", "/processes"]
        assert app._slash_completion_candidates("/r") == ["/resume", "/run"]

    def test_slash_completion_candidates_include_dynamic_process_commands(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[
                {"name": "investment-analysis", "version": "1.0"},
                {"name": "marketing-strategy", "version": "1.2"},
            ])
        ))

        assert app._slash_completion_candidates("/i") == ["/investment-analysis"]
        assert app._slash_completion_candidates("/m") == [
            "/model",
            "/mcp",
            "/marketing-strategy",
        ]


class TestProcessCatalogRendering:
    def test_process_catalog_renders_full_description_without_truncation(self):
        from loom.tui.app import LoomApp

        description = (
            "End-to-end market research workflow across geography, demand, "
            "competitor mapping, environmental risk, and synthesis deliverables."
        )
        assert len(description) > 80

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[
                {
                    "name": "market-research",
                    "version": "1.0",
                    "description": description,
                },
            ])
        ))

        rendered = app._render_process_catalog()

        assert "End-to-end market research workflow across geography" in rendered
        assert "and synthesis deliverables." in rendered


class TestProcessSlashCommands:
    @pytest.mark.asyncio
    async def test_process_without_args_shows_catalog(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = SimpleNamespace(name="marketing-strategy")
        app._render_process_catalog = MagicMock(return_value="catalog")
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/process")

        assert handled is True
        app._render_process_catalog.assert_called_once()
        chat.add_info.assert_called_once_with("catalog")

    @pytest.mark.asyncio
    async def test_process_list_uses_catalog_renderer(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._render_process_catalog = MagicMock(return_value="catalog")

        handled = await app._handle_slash_command("/process list")

        assert handled is True
        app._render_process_catalog.assert_called_once()
        chat.add_info.assert_called_once_with("catalog")

    @pytest.mark.asyncio
    async def test_processes_alias_uses_catalog_renderer(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._render_process_catalog = MagicMock(return_value="catalog")

        handled = await app._handle_slash_command("/processes")

        assert handled is True
        app._render_process_catalog.assert_called_once()
        chat.add_info.assert_called_once_with("catalog")

    @pytest.mark.asyncio
    async def test_process_use_requires_argument(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/process use")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Usage" in message
        assert "/process use" in message
        assert "<name-or-path>" in message

    @pytest.mark.asyncio
    async def test_process_use_load_error(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        loader = MagicMock()
        loader.load.side_effect = ValueError("boom")
        app._create_process_loader = MagicMock(return_value=loader)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/process use bad-proc")

        assert handled is True
        chat.add_info.assert_called_once()
        assert "Failed to load process 'bad-proc'" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_process_use_sets_process_and_reloads_session(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        loader = MagicMock()
        loader.load.return_value = SimpleNamespace(
            name="marketing-strategy",
            version="1.0",
        )
        app._create_process_loader = MagicMock(return_value=loader)
        app._reload_session_for_process_change = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command(
            "/process use marketing-strategy",
        )

        assert handled is True
        assert app._process_name == "marketing-strategy"
        app._reload_session_for_process_change.assert_awaited_once()
        message = chat.add_info.call_args.args[0]
        assert "Active Process Updated" in message
        assert "Name:[/] [bold]marketing-strategy[/bold]" in message
        assert "Version:[/] [dim]v1.0[/dim]" in message

    @pytest.mark.asyncio
    async def test_process_use_blocks_reserved_command_name(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        loader = MagicMock()
        loader.list_available.return_value = []
        loader.load.return_value = SimpleNamespace(name="run", version="1.0")
        app._create_process_loader = MagicMock(return_value=loader)
        app._reload_session_for_process_change = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/process use run")

        assert handled is True
        app._reload_session_for_process_change.assert_not_awaited()
        chat.add_info.assert_called_once()
        assert "conflicts with a built-in slash command" in (
            chat.add_info.call_args.args[0]
        )

    @pytest.mark.asyncio
    async def test_process_off_with_no_active_process(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_name = None
        app._process_defn = None
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/process off")

        assert handled is True
        chat.add_info.assert_called_once_with("No active process.")

    @pytest.mark.asyncio
    async def test_process_off_reloads_session(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_name = "marketing-strategy"
        app._process_defn = SimpleNamespace(name="marketing-strategy")
        app._reload_session_for_process_change = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/process off")

        assert handled is True
        assert app._process_name is None
        assert app._process_defn is None
        app._reload_session_for_process_change.assert_awaited_once()
        assert "Name:[/] none" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_run_requires_goal(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/run")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Usage" in message
        assert "/run" in message
        assert "<goal>" in message

    @pytest.mark.asyncio
    async def test_run_without_active_process_synthesizes_adhoc(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = None
        app._config = SimpleNamespace(process=SimpleNamespace(default=""))
        adhoc_process = SimpleNamespace(name="adhoc-report", phases=[])
        app._get_or_create_adhoc_process = AsyncMock(return_value=(
            SimpleNamespace(
                process_defn=adhoc_process,
                recommended_tools=["web_search"],
            ),
            False,
        ))
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/run analyze tesla")

        assert handled is True
        app._get_or_create_adhoc_process.assert_awaited_once_with(
            "analyze tesla",
            fresh=False,
        )
        app._start_process_run.assert_awaited_once()
        assert app._start_process_run.await_args.args == ("analyze tesla",)
        kwargs = app._start_process_run.await_args.kwargs
        assert kwargs["process_defn"] is adhoc_process
        assert kwargs["is_adhoc"] is True
        assert kwargs["recommended_tools"] == ["web_search"]
        assert isinstance(kwargs.get("adhoc_synthesis_notes"), list)
        assert kwargs.get("adhoc_synthesis_notes")

    @pytest.mark.asyncio
    async def test_run_fresh_flag_bypasses_cache_lookup(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = None
        app._config = SimpleNamespace(process=SimpleNamespace(default=""))
        adhoc_process = SimpleNamespace(name="adhoc-report", phases=[])
        app._get_or_create_adhoc_process = AsyncMock(return_value=(
            SimpleNamespace(
                process_defn=adhoc_process,
                recommended_tools=[],
            ),
            False,
        ))
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/run --fresh analyze tesla")

        assert handled is True
        app._get_or_create_adhoc_process.assert_awaited_once_with(
            "analyze tesla",
            fresh=True,
        )
        app._start_process_run.assert_awaited_once()
        assert app._start_process_run.await_args.args == ("analyze tesla",)
        kwargs = app._start_process_run.await_args.kwargs
        assert kwargs["process_defn"] is adhoc_process
        assert kwargs["is_adhoc"] is True
        assert kwargs["recommended_tools"] == []
        assert isinstance(kwargs.get("adhoc_synthesis_notes"), list)
        assert kwargs.get("adhoc_synthesis_notes")

    @pytest.mark.asyncio
    async def test_get_or_create_adhoc_process_persists_cache_file(self, tmp_path, monkeypatch):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        goal = "research wasted utility history"
        key = app._adhoc_cache_key(goal)
        spec = {
            "intent": "research",
            "name": "book-research-adhoc",
            "description": "Ad hoc book research process",
            "persona": "Investigative researcher",
            "phase_mode": "guided",
            "tool_guidance": "Gather and synthesize evidence.",
            "required_tools": ["read_file", "write_file"],
            "recommended_tools": ["spreadsheet"],
            "phases": [
                {
                    "id": "scope",
                    "description": "Define scope",
                    "depends_on": [],
                    "acceptance_criteria": "Scope documented.",
                    "deliverables": ["scope.md"],
                },
                {
                    "id": "research",
                    "description": "Collect cases",
                    "depends_on": ["scope"],
                    "acceptance_criteria": "Cases validated.",
                    "deliverables": ["cases.md"],
                },
            ],
        }
        generated_entry = app._build_adhoc_cache_entry(key=key, goal=goal, spec=spec)
        app._synthesize_adhoc_process = AsyncMock(return_value=generated_entry)

        resolved, from_cache = await app._get_or_create_adhoc_process(goal)

        assert from_cache is False
        assert resolved.process_defn.name == "book-research-adhoc"
        cache_path = tmp_path / ".loom" / "cache" / "adhoc-processes" / f"{key}.yaml"
        assert cache_path.exists()
        payload = yaml.safe_load(cache_path.read_text(encoding="utf-8"))
        assert payload["key"] == key
        assert payload["goal"] == goal
        assert payload["spec"]["name"] == "book-research-adhoc"

    @pytest.mark.asyncio
    async def test_get_or_create_adhoc_process_uses_disk_cache(self, tmp_path, monkeypatch):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file"]

        goal = "draft incident retrospective"
        key = app._adhoc_cache_key(goal)
        spec = {
            "intent": "writing",
            "source": "model_generated",
            "name": "retro-adhoc",
            "description": "Incident retrospective process",
            "persona": "Incident analyst",
            "phase_mode": "guided",
            "tool_guidance": "Collect timeline, analyze causes, summarize.",
            "required_tools": ["read_file"],
            "recommended_tools": [],
            "phases": [
                {
                    "id": "timeline",
                    "description": "Build timeline",
                    "depends_on": [],
                    "acceptance_criteria": "Timeline captured.",
                    "deliverables": ["timeline.md"],
                },
                {
                    "id": "identify-root-cause",
                    "description": "Analyze root causes.",
                    "depends_on": ["timeline"],
                    "acceptance_criteria": "Root causes documented.",
                    "deliverables": ["root-cause.md"],
                },
                {
                    "id": "define-remediations",
                    "description": "Define corrective actions.",
                    "depends_on": ["identify-root-cause"],
                    "acceptance_criteria": "Remediation plan defined.",
                    "deliverables": ["remediations.md"],
                },
                {
                    "id": "publish-retro",
                    "description": "Publish final retrospective.",
                    "depends_on": ["define-remediations"],
                    "acceptance_criteria": "Final retrospective approved.",
                    "deliverables": ["retro.md"],
                },
            ],
        }
        generated_entry = app._build_adhoc_cache_entry(key=key, goal=goal, spec=spec)
        app._synthesize_adhoc_process = AsyncMock(return_value=generated_entry)

        first, first_cached = await app._get_or_create_adhoc_process(goal)
        assert first_cached is False
        assert first.process_defn.name == "retro-adhoc"

        app._adhoc_process_cache.clear()
        app._synthesize_adhoc_process = AsyncMock(
            side_effect=AssertionError("synthesis should not run when disk cache exists"),
        )

        second, second_cached = await app._get_or_create_adhoc_process(goal)

        assert second_cached is True
        assert second.process_defn.name == "retro-adhoc"
        app._synthesize_adhoc_process.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_or_create_adhoc_process_fresh_bypasses_memory_and_disk_cache(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        goal = "research wasted utility history"
        key = app._adhoc_cache_key(goal)

        cached_spec = {
            "intent": "research",
            "source": "model_generated",
            "name": "cached-process-adhoc",
            "description": "cached",
            "persona": "analyst",
            "phase_mode": "strict",
            "tool_guidance": "cached guidance",
            "required_tools": ["read_file"],
            "recommended_tools": [],
            "phases": [
                {
                    "id": "cached-a",
                    "description": "a",
                    "depends_on": [],
                    "acceptance_criteria": "a",
                    "deliverables": ["a.md"],
                },
                {
                    "id": "cached-b",
                    "description": "b",
                    "depends_on": ["cached-a"],
                    "acceptance_criteria": "b",
                    "deliverables": ["b.md"],
                },
                {
                    "id": "cached-c",
                    "description": "c",
                    "depends_on": ["cached-b"],
                    "acceptance_criteria": "c",
                    "deliverables": ["c.md"],
                },
                {
                    "id": "cached-d",
                    "description": "d",
                    "depends_on": ["cached-c"],
                    "acceptance_criteria": "d",
                    "deliverables": ["d.md"],
                },
            ],
        }
        cached_entry = app._build_adhoc_cache_entry(key=key, goal=goal, spec=cached_spec)
        app._adhoc_process_cache[key] = cached_entry
        disk_path = tmp_path / ".loom" / "cache" / "adhoc-processes" / f"{key}.yaml"
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_text(
            yaml.safe_dump(
                {
                    "key": key,
                    "goal": goal,
                    "generated_at_monotonic": 1.0,
                    "saved_at": "2026-02-21T00:00:00+00:00",
                    "spec": cached_spec,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        fresh_spec = {
            "intent": "research",
            "source": "model_generated",
            "name": "fresh-process-adhoc",
            "description": "fresh",
            "persona": "analyst",
            "phase_mode": "strict",
            "tool_guidance": "fresh guidance",
            "required_tools": ["read_file"],
            "recommended_tools": [],
            "phases": [
                {
                    "id": "fresh-a",
                    "description": "a",
                    "depends_on": [],
                    "acceptance_criteria": "a",
                    "deliverables": ["a.md"],
                },
                {
                    "id": "fresh-b",
                    "description": "b",
                    "depends_on": ["fresh-a"],
                    "acceptance_criteria": "b",
                    "deliverables": ["b.md"],
                },
                {
                    "id": "fresh-c",
                    "description": "c",
                    "depends_on": ["fresh-b"],
                    "acceptance_criteria": "c",
                    "deliverables": ["c.md"],
                },
                {
                    "id": "fresh-d",
                    "description": "d",
                    "depends_on": ["fresh-c"],
                    "acceptance_criteria": "d",
                    "deliverables": ["d.md"],
                },
            ],
        }
        fresh_entry = app._build_adhoc_cache_entry(key=key, goal=goal, spec=fresh_spec)
        app._synthesize_adhoc_process = AsyncMock(return_value=fresh_entry)

        resolved, from_cache = await app._get_or_create_adhoc_process(goal, fresh=True)

        assert from_cache is False
        assert resolved.process_defn.name == "fresh-process-adhoc"
        app._synthesize_adhoc_process.assert_awaited_once()

    def test_adhoc_synthesis_activity_lines_include_diagnostics(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        entry = SimpleNamespace(
            spec={
                "source": "model_generated",
                "intent": "research",
                "phases": [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
                "required_tools": ["read_file"],
                "recommended_tools": ["web_search"],
                "_synthesis": {
                    "initial_parse_ok": False,
                    "repair_attempted": True,
                    "repair_parse_ok": True,
                    "initial_response_chars": 1450,
                    "fallback_reason": "",
                    "artifact_dir": "/tmp/adhoc-synthesis/run-1",
                    "log_path": "/tmp/adhoc-synthesis.jsonl",
                },
            },
        )

        lines = app._adhoc_synthesis_activity_lines(
            entry,  # type: ignore[arg-type]
            from_cache=False,
            fresh=True,
        )

        assert any("Ad hoc definition summary" in line for line in lines)
        assert any("cache decision: miss (fresh=True)" in line for line in lines)
        assert any("parse diagnostics: initial=failed, repair=ok" in line for line in lines)
        assert any("Ad hoc synthesis artifacts:" in line for line in lines)
        assert any("Ad hoc synthesis log:" in line for line in lines)

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_persists_diagnostics_log(self, tmp_path, monkeypatch):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]
        app._model.complete = AsyncMock(return_value=SimpleNamespace(
            text=json.dumps({
                "intent": "research",
                "name": "history-research",
                "description": "custom research flow",
                "persona": "historian",
                "phase_mode": "strict",
                "tool_guidance": "use primary sources",
                "required_tools": ["read_file"],
                "recommended_tools": [],
                "phases": [
                    {
                        "id": "scope-hypothesis",
                        "description": "define scope",
                        "depends_on": [],
                        "acceptance_criteria": "scope done",
                        "deliverables": ["scope.md"],
                    },
                    {
                        "id": "collect-cases",
                        "description": "collect cases",
                        "depends_on": ["scope-hypothesis"],
                        "acceptance_criteria": "cases done",
                        "deliverables": ["cases.md"],
                    },
                    {
                        "id": "quantify-loss",
                        "description": "quantify loss",
                        "depends_on": ["collect-cases"],
                        "acceptance_criteria": "quantified",
                        "deliverables": ["quant.md"],
                    },
                    {
                        "id": "select-twelve",
                        "description": "finalize twelve",
                        "depends_on": ["quantify-loss"],
                        "acceptance_criteria": "selected",
                        "deliverables": ["selected.md"],
                    },
                    {
                        "id": "deliver-manuscript-notes",
                        "description": "deliver notes",
                        "depends_on": ["select-twelve"],
                        "acceptance_criteria": "delivered",
                        "deliverables": ["notes.md"],
                    },
                ],
            }),
        ))

        entry = await app._synthesize_adhoc_process("research failures", key="a1b2c3d4")

        synthesis = entry.spec.get("_synthesis", {})
        assert isinstance(synthesis, dict)
        assert synthesis.get("initial_parse_ok") is True
        assert synthesis.get("resolved_source") == "model_generated"
        artifact_dir = Path(str(synthesis.get("artifact_dir", "")))
        assert artifact_dir.exists()
        assert (artifact_dir / "01-initial-prompt.txt").exists()
        assert (artifact_dir / "02-initial-response.txt").exists()
        assert (artifact_dir / "10-normalized-spec.yaml").exists()
        assert (artifact_dir / "11-diagnostics.json").exists()
        log_path = Path(str(synthesis.get("log_path", "")))
        assert log_path.exists()
        log_lines = [
            line.strip()
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert log_lines
        latest = json.loads(log_lines[-1])
        assert latest.get("event") == "adhoc_synthesis"
        assert latest.get("cache_key") == "a1b2c3d4"

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_respects_repair_source_limit(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.config import Config, LimitsConfig
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        class RepairCapModel:
            name = "repair-cap-model"
            configured_temperature = 1.0

            def __init__(self) -> None:
                self.calls = 0
                self.prompts: list[str] = []

            async def complete(
                self,
                messages,
                tools=None,
                temperature=None,
                max_tokens=None,
                response_format=None,
            ):
                del tools, temperature, max_tokens, response_format
                self.calls += 1
                prompt_text = str(messages[0].get("content", "") if messages else "")
                self.prompts.append(prompt_text)
                if self.calls == 1:
                    return SimpleNamespace(text="not json " + ("x" * 2000))
                return SimpleNamespace(text=json.dumps({
                    "intent": "research",
                    "name": "repair-cap-ok",
                    "description": "custom process",
                    "persona": "analyst",
                    "phase_mode": "strict",
                    "tool_guidance": "use tools",
                    "required_tools": ["read_file"],
                    "recommended_tools": [],
                    "phases": [
                        {
                            "id": "scope",
                            "description": "scope",
                            "depends_on": [],
                            "acceptance_criteria": "ok",
                            "deliverables": ["scope.md"],
                        },
                        {
                            "id": "collect",
                            "description": "collect",
                            "depends_on": ["scope"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["collect.md"],
                        },
                        {
                            "id": "analyze",
                            "description": "analyze",
                            "depends_on": ["collect"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["analyze.md"],
                        },
                        {
                            "id": "verify",
                            "description": "verify",
                            "depends_on": ["analyze"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["verify.md"],
                        },
                        {
                            "id": "deliver",
                            "description": "deliver",
                            "depends_on": ["verify"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["deliver.md"],
                        },
                    ],
                }))

        model = RepairCapModel()
        app = LoomApp(
            model=model,  # type: ignore[arg-type]
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(limits=LimitsConfig(adhoc_repair_source_max_chars=120)),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process(
            "research failure modes",
            key="repaircap",
        )

        synthesis = entry.spec.get("_synthesis", {})
        assert entry.process_defn.name == "repair-cap-ok-adhoc"
        assert synthesis.get("repair_source_truncated") is True
        assert synthesis.get("repair_source_chars") == 120
        assert model.calls >= 2
        assert len(model.prompts) >= 2
        assert "<<<BEGIN_SOURCE>>>" in model.prompts[1]

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_prefers_planner_role_model(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        planner_model = MagicMock(name="planner-model")
        planner_model.name = "planner-model"
        planner_model.configured_temperature = 0.3
        planner_model.complete = AsyncMock(return_value=SimpleNamespace(
            text=json.dumps({
                "intent": "research",
                "name": "planner-selected",
                "description": "planned",
                "persona": "planner persona",
                "phase_mode": "strict",
                "tool_guidance": "use tools",
                "required_tools": ["read_file"],
                "recommended_tools": [],
                "phases": [
                    {
                        "id": "scope",
                        "description": "scope",
                        "depends_on": [],
                        "acceptance_criteria": "ok",
                        "deliverables": ["scope.md"],
                    },
                    {
                        "id": "collect",
                        "description": "collect",
                        "depends_on": ["scope"],
                        "acceptance_criteria": "ok",
                        "deliverables": ["collect.md"],
                    },
                    {
                        "id": "analyze",
                        "description": "analyze",
                        "depends_on": ["collect"],
                        "acceptance_criteria": "ok",
                        "deliverables": ["analyze.md"],
                    },
                    {
                        "id": "deliver",
                        "description": "deliver",
                        "depends_on": ["analyze"],
                        "acceptance_criteria": "ok",
                        "deliverables": ["deliver.md"],
                    },
                ],
            }),
        ))

        active_model = MagicMock(name="active-executor-model")
        active_model.complete = AsyncMock(side_effect=AssertionError(
            "active cowork model should not be used for ad hoc synthesis",
        ))

        fake_router = MagicMock()

        def _select(*, tier=1, role="executor"):
            assert role == "planner"
            assert tier == 2
            return planner_model

        fake_router.select = MagicMock(side_effect=_select)
        fake_router.close = AsyncMock()
        monkeypatch.setattr(
            "loom.models.router.ModelRouter.from_config",
            lambda cfg: fake_router,
        )

        app = LoomApp(
            model=active_model,
            tools=MagicMock(),
            workspace=tmp_path,
            config=Config(models={
                "primary": ModelConfig(
                    provider="ollama",
                    base_url="http://localhost:11434",
                    model="executor-model",
                    roles=["executor"],
                ),
                "planner": ModelConfig(
                    provider="ollama",
                    base_url="http://localhost:11434",
                    model="planner-model",
                    roles=["planner"],
                ),
            }),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process("research failures", key="planrole")

        assert entry.process_defn.name == "planner-selected-adhoc"
        assert planner_model.complete.await_count == 1
        assert active_model.complete.await_count == 0
        assert fake_router.select.call_count == 1
        fake_router.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_uses_planner_token_limit(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.config import Config, LimitsConfig, ModelConfig
        from loom.tui.app import LoomApp

        planner_calls: list[int | None] = []
        planner_model = MagicMock(name="planner-model")
        planner_model.name = "planner-model"
        planner_model.configured_temperature = 0.3
        planner_model.configured_max_tokens = 8192

        async def _planner_complete(messages, **kwargs):
            del messages
            planner_calls.append(kwargs.get("max_tokens"))
            return SimpleNamespace(text=json.dumps({
                "intent": "research",
                "name": "planner-limit",
                "description": "planned",
                "persona": "planner persona",
                "phase_mode": "strict",
                "tool_guidance": "use tools",
                "required_tools": ["read_file"],
                "recommended_tools": [],
                "phases": [
                    {
                        "id": "scope",
                        "description": "scope",
                        "depends_on": [],
                        "acceptance_criteria": "ok",
                        "deliverables": ["scope.md"],
                    },
                    {
                        "id": "collect",
                        "description": "collect",
                        "depends_on": ["scope"],
                        "acceptance_criteria": "ok",
                        "deliverables": ["collect.md"],
                    },
                    {
                        "id": "analyze",
                        "description": "analyze",
                        "depends_on": ["collect"],
                        "acceptance_criteria": "ok",
                        "deliverables": ["analyze.md"],
                    },
                ],
            }))

        planner_model.complete = AsyncMock(side_effect=_planner_complete)

        active_model = MagicMock(name="active-executor-model")
        active_model.complete = AsyncMock(side_effect=AssertionError(
            "active cowork model should not be used for ad hoc synthesis",
        ))

        fake_router = MagicMock()
        fake_router.select = MagicMock(return_value=planner_model)
        fake_router.close = AsyncMock()
        monkeypatch.setattr(
            "loom.models.router.ModelRouter.from_config",
            lambda cfg: fake_router,
        )

        app = LoomApp(
            model=active_model,
            tools=MagicMock(),
            workspace=tmp_path,
            config=Config(
                models={
                    "primary": ModelConfig(
                        provider="ollama",
                        base_url="http://localhost:11434",
                        model="executor-model",
                        roles=["executor"],
                    ),
                    "planner": ModelConfig(
                        provider="ollama",
                        base_url="http://localhost:11434",
                        model="planner-model",
                        roles=["planner"],
                        max_tokens=8192,
                    ),
                },
                limits=LimitsConfig(planning_response_max_tokens=16384),
            ),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process("research failures", key="planlimit")

        assert entry.process_defn.name == "planner-limit-adhoc"
        assert planner_calls
        assert planner_calls[0] == 16384
        assert active_model.complete.await_count == 0
        fake_router.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_uses_model_configured_temperature(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.models.base import ModelConnectionError
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        class TempOneModel:
            name = "temp-one-primary"
            configured_temperature = 1.0

            def __init__(self) -> None:
                self.temps: list[float | None] = []

            async def complete(
                self,
                messages,
                tools=None,
                temperature=None,
                max_tokens=None,
                response_format=None,
            ):
                del messages, tools, max_tokens, response_format
                self.temps.append(temperature)
                if temperature != 1.0:
                    raise ModelConnectionError(
                        "Model server returned HTTP 400: "
                        '{"error":{"message":"invalid temperature: '
                        'only 1 is allowed for this model"}}'
                    )
                return SimpleNamespace(
                    text=json.dumps({
                        "intent": "research",
                        "name": "temp-recovered",
                        "description": "custom process",
                        "persona": "researcher",
                        "phase_mode": "strict",
                        "tool_guidance": "use tools",
                        "required_tools": ["read_file"],
                        "recommended_tools": [],
                        "phases": [
                            {
                                "id": "scope",
                                "description": "scope",
                                "depends_on": [],
                                "acceptance_criteria": "ok",
                                "deliverables": ["scope.md"],
                            },
                            {
                                "id": "collect",
                                "description": "collect",
                                "depends_on": ["scope"],
                                "acceptance_criteria": "ok",
                                "deliverables": ["collect.md"],
                            },
                            {
                                "id": "analyze",
                                "description": "analyze",
                                "depends_on": ["collect"],
                                "acceptance_criteria": "ok",
                                "deliverables": ["analyze.md"],
                            },
                            {
                                "id": "deliver",
                                "description": "deliver",
                                "depends_on": ["analyze"],
                                "acceptance_criteria": "ok",
                                "deliverables": ["deliver.md"],
                            },
                        ],
                    }),
                )

        model = TempOneModel()
        app = LoomApp(
            model=model,  # type: ignore[arg-type]
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process(
            "research big failures",
            key="temp1111",
        )

        assert entry.process_defn.name == "temp-recovered-adhoc"
        assert model.temps and model.temps[0] == 1.0
        synthesis = entry.spec.get("_synthesis", {})
        assert synthesis.get("initial_temperature") == 1.0
        assert synthesis.get("initial_parse_ok") is True
        assert synthesis.get("resolved_source") == "model_generated"

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_marks_temperature_config_mismatch(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.models.base import ModelConnectionError
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        class TempMismatchModel:
            name = "temp-mismatch-primary"
            configured_temperature = 0.1

            async def complete(
                self,
                messages,
                tools=None,
                temperature=None,
                max_tokens=None,
                response_format=None,
            ):
                del messages, tools, temperature, max_tokens, response_format
                raise ModelConnectionError(
                    "Model server returned HTTP 400: "
                    '{"error":{"message":"invalid temperature: '
                    'only 1 is allowed for this model"}}'
                )

        app = LoomApp(
            model=TempMismatchModel(),  # type: ignore[arg-type]
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process(
            "research big failures",
            key="temp2222",
        )

        synthesis = entry.spec.get("_synthesis", {})
        assert synthesis.get("resolved_source") == "fallback_template"
        assert synthesis.get("fallback_reason") == "temperature_config_mismatch"
        assert "invalid temperature" in str(synthesis.get("initial_error", "")).lower()

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_marks_empty_model_response(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        class EmptyResponseModel:
            name = "empty-response-primary"
            configured_temperature = 1.0

            def __init__(self) -> None:
                self.calls = 0

            async def complete(
                self,
                messages,
                tools=None,
                temperature=None,
                max_tokens=None,
                response_format=None,
            ):
                del messages, tools, temperature, max_tokens, response_format
                self.calls += 1
                return SimpleNamespace(text="")

        model = EmptyResponseModel()
        app = LoomApp(
            model=model,  # type: ignore[arg-type]
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process(
            "research big failures",
            key="temp3333",
        )

        synthesis = entry.spec.get("_synthesis", {})
        assert synthesis.get("resolved_source") == "fallback_template"
        assert synthesis.get("fallback_reason") == "empty_model_response"
        assert synthesis.get("empty_response_retry_attempted") is True
        assert model.calls >= 2

    def test_normalize_adhoc_spec_preserves_phase_mode_and_minimum_phases(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        raw = {
            "name": "book-research",
            "description": "Research workflow",
            "persona": "Research analyst",
            "phase_mode": "guided",
            "tool_guidance": "Use tools",
            "required_tools": [],
            "recommended_tools": [],
            "phases": [
                {
                    "id": "only-step",
                    "description": "Do everything",
                    "depends_on": [],
                    "acceptance_criteria": "Done",
                    "deliverables": ["out.md"],
                },
            ],
        }

        normalized = app._normalize_adhoc_spec(
            raw,
            goal="research overinvestment cases",
            key="abc12345deadbeef",
            available_tools=[],
        )

        assert normalized["phase_mode"] == "guided"
        assert len(normalized["phases"]) >= 3
        ids = [phase["id"] for phase in normalized["phases"]]
        assert "verify-quality" in ids
        assert normalized["source"] == "fallback_template"

    def test_extract_json_payload_handles_wrapped_text(self):
        from loom.tui.app import LoomApp

        raw = """
        The process spec is below.
        {
          "intent": "research",
          "name": "custom-adhoc",
          "description": "custom",
          "persona": "analyst",
          "phase_mode": "strict",
          "tool_guidance": "guidance",
          "required_tools": [],
          "recommended_tools": [],
          "phases": []
        }
        """
        parsed = LoomApp._extract_json_payload(
            raw,
            expected_keys=("intent", "name", "phases"),
        )
        assert isinstance(parsed, dict)
        assert parsed.get("name") == "custom-adhoc"

    def test_extract_json_payload_accepts_yaml_like_fallback(self):
        from loom.tui.app import LoomApp

        raw = (
            "intent: research\n"
            "name: yaml-adhoc\n"
            "description: custom\n"
            "persona: analyst\n"
            "phase_mode: strict\n"
            "tool_guidance: use tools\n"
            "required_tools: []\n"
            "recommended_tools: []\n"
            "phases: []\n"
        )
        parsed = LoomApp._extract_json_payload(
            raw,
            expected_keys=("intent", "name", "phases"),
        )
        assert isinstance(parsed, dict)
        assert parsed.get("name") == "yaml-adhoc"

    def test_extract_json_payload_parses_fenced_block_with_preamble(self):
        from loom.tui.app import LoomApp

        raw = (
            "I planned this process for you.\n"
            "```json\n"
            "{\n"
            '  "intent": "research",\n'
            '  "name": "fenced-adhoc",\n'
            '  "description": "custom",\n'
            '  "persona": "analyst",\n'
            '  "phase_mode": "strict",\n'
            '  "tool_guidance": "use tools",\n'
            '  "required_tools": [],\n'
            '  "recommended_tools": [],\n'
            '  "phases": []\n'
            "}\n"
            "```\n"
            "Done."
        )
        parsed = LoomApp._extract_json_payload(
            raw,
            expected_keys=("intent", "name", "phases"),
        )
        assert isinstance(parsed, dict)
        assert parsed.get("name") == "fenced-adhoc"

    def test_extract_json_payload_parses_yaml_after_markdown_heading(self):
        from loom.tui.app import LoomApp

        raw = (
            "### Proposed process\n"
            "- notes: draft\n\n"
            "intent: research\n"
            "name: markdown-yaml-adhoc\n"
            "description: custom\n"
            "persona: analyst\n"
            "phase_mode: strict\n"
            "tool_guidance: use tools\n"
            "required_tools: []\n"
            "recommended_tools: []\n"
            "phases: []\n"
        )
        parsed = LoomApp._extract_json_payload(
            raw,
            expected_keys=("intent", "name", "phases"),
        )
        assert isinstance(parsed, dict)
        assert parsed.get("name") == "markdown-yaml-adhoc"

    def test_extract_json_payload_rejects_nested_partial_when_schema_expected(self):
        from loom.tui.app import LoomApp

        raw = (
            '{"intent":"research","name":"x","phases":[{"id":"discovery","description":"d"}'
        )
        parsed = LoomApp._extract_json_payload(
            raw,
            expected_keys=("intent", "name", "phases"),
        )
        assert parsed is None

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_uses_minimal_retry_when_initial_and_repair_non_parseable(  # noqa: E501
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        class ThreeStageModel:
            name = "three-stage-primary"
            configured_temperature = 1.0

            def __init__(self) -> None:
                self.calls = 0

            async def complete(
                self,
                messages,
                tools=None,
                temperature=None,
                max_tokens=None,
                response_format=None,
            ):
                del messages, tools, temperature, max_tokens, response_format
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(text=(
                        "Here is your process plan with reasoning and prose. "
                        "Phase 1 scope, phase 2 evidence, phase 3 analyze."
                    ))
                if self.calls == 2:
                    return SimpleNamespace(text=(
                        "Still prose; not JSON. I recommend six phases and strict mode."
                    ))
                return SimpleNamespace(text=json.dumps({
                    "intent": "research",
                    "name": "minimal-retry-win",
                    "description": "custom process",
                    "persona": "research analyst",
                    "phase_mode": "strict",
                    "tool_guidance": "use tools",
                    "required_tools": ["read_file"],
                    "recommended_tools": [],
                    "phases": [
                        {
                            "id": "scope",
                            "description": "scope",
                            "depends_on": [],
                            "acceptance_criteria": "ok",
                            "deliverables": ["scope.md"],
                        },
                        {
                            "id": "source-plan",
                            "description": "source planning",
                            "depends_on": ["scope"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["plan.md"],
                        },
                        {
                            "id": "collect-evidence",
                            "description": "collect evidence",
                            "depends_on": ["source-plan"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["evidence.md"],
                        },
                        {
                            "id": "analyze-findings",
                            "description": "analyze",
                            "depends_on": ["collect-evidence"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["analysis.md"],
                        },
                        {
                            "id": "deliver-report",
                            "description": "deliver",
                            "depends_on": ["analyze-findings"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["report.md"],
                        },
                    ],
                }))

        model = ThreeStageModel()
        app = LoomApp(
            model=model,  # type: ignore[arg-type]
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process(
            "research big failures",
            key="temp4444",
        )

        synthesis = entry.spec.get("_synthesis", {})
        assert entry.process_defn.name == "minimal-retry-win-adhoc"
        assert synthesis.get("resolved_source") == "model_generated"
        assert synthesis.get("minimal_retry_attempted") is True
        assert synthesis.get("minimal_retry_parse_ok") is True
        assert model.calls >= 3

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_uses_minimal_retry_when_parsed_spec_incomplete(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        class IncompleteParsedModel:
            name = "incomplete-primary"
            configured_temperature = 1.0

            def __init__(self) -> None:
                self.calls = 0

            async def complete(
                self,
                messages,
                tools=None,
                temperature=None,
                max_tokens=None,
                response_format=None,
            ):
                del messages, tools, temperature, max_tokens, response_format
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(text=json.dumps({
                        "intent": "research",
                        "name": "too-thin",
                        "description": "incomplete process",
                        "persona": "analyst",
                        "phase_mode": "strict",
                        "tool_guidance": "use tools",
                        "required_tools": ["read_file"],
                        "recommended_tools": [],
                        "phases": [{
                            "id": "scope",
                            "description": "scope",
                            "depends_on": [],
                            "acceptance_criteria": "ok",
                            "deliverables": ["scope.md"],
                        }],
                    }))
                return SimpleNamespace(text=json.dumps({
                    "intent": "research",
                    "name": "recovered-thin",
                    "description": "full process",
                    "persona": "analyst",
                    "phase_mode": "strict",
                    "tool_guidance": "use tools",
                    "required_tools": ["read_file"],
                    "recommended_tools": [],
                    "phases": [
                        {
                            "id": "scope",
                            "description": "scope",
                            "depends_on": [],
                            "acceptance_criteria": "ok",
                            "deliverables": ["scope.md"],
                        },
                        {
                            "id": "source-plan",
                            "description": "source plan",
                            "depends_on": ["scope"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["plan.md"],
                        },
                        {
                            "id": "collect-evidence",
                            "description": "collect",
                            "depends_on": ["source-plan"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["evidence.md"],
                        },
                        {
                            "id": "analyze-findings",
                            "description": "analyze",
                            "depends_on": ["collect-evidence"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["analysis.md"],
                        },
                        {
                            "id": "deliver-report",
                            "description": "deliver",
                            "depends_on": ["analyze-findings"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["report.md"],
                        },
                    ],
                }))

        model = IncompleteParsedModel()
        app = LoomApp(
            model=model,  # type: ignore[arg-type]
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process(
            "research big failures",
            key="temp5555",
        )

        synthesis = entry.spec.get("_synthesis", {})
        assert entry.process_defn.name == "recovered-thin-adhoc"
        assert synthesis.get("parsed_raw_incomplete") is True
        assert synthesis.get("minimal_retry_attempted") is True
        assert synthesis.get("minimal_retry_parse_ok") is True
        assert model.calls >= 2

    @pytest.mark.asyncio
    async def test_synthesize_adhoc_process_does_not_accept_truncated_nested_json(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        class TruncatedThenValidModel:
            name = "truncated-valid-primary"
            configured_temperature = 1.0

            def __init__(self) -> None:
                self.calls = 0

            async def complete(
                self,
                messages,
                tools=None,
                temperature=None,
                max_tokens=None,
                response_format=None,
            ):
                del messages, tools, temperature, max_tokens, response_format
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(text=(
                        '{"intent":"research","name":"x","description":"d","persona":"p",'
                        '"phase_mode":"strict","tool_guidance":"g","required_tools":[],"recommended_tools":[],"phases":[{"id":"a","description":"b"'
                    ))
                if self.calls == 2:
                    return SimpleNamespace(text="not json")
                return SimpleNamespace(text=json.dumps({
                    "intent": "research",
                    "name": "final-valid",
                    "description": "valid",
                    "persona": "analyst",
                    "phase_mode": "strict",
                    "tool_guidance": "use tools",
                    "required_tools": ["read_file"],
                    "recommended_tools": [],
                    "phases": [
                        {
                            "id": "scope",
                            "description": "scope",
                            "depends_on": [],
                            "acceptance_criteria": "ok",
                            "deliverables": ["scope.md"],
                        },
                        {
                            "id": "source-plan",
                            "description": "plan",
                            "depends_on": ["scope"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["plan.md"],
                        },
                        {
                            "id": "collect-evidence",
                            "description": "collect",
                            "depends_on": ["source-plan"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["evidence.md"],
                        },
                        {
                            "id": "analyze-findings",
                            "description": "analyze",
                            "depends_on": ["collect-evidence"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["analysis.md"],
                        },
                        {
                            "id": "verify-quality",
                            "description": "verify",
                            "depends_on": ["analyze-findings"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["verify.md"],
                        },
                        {
                            "id": "deliver-report",
                            "description": "deliver",
                            "depends_on": ["verify-quality"],
                            "acceptance_criteria": "ok",
                            "deliverables": ["report.md"],
                        },
                    ],
                }))

        model = TruncatedThenValidModel()
        app = LoomApp(
            model=model,  # type: ignore[arg-type]
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        entry = await app._synthesize_adhoc_process(
            "research big failures",
            key="temp6666",
        )

        synthesis = entry.spec.get("_synthesis", {})
        assert entry.process_defn.name == "final-valid-adhoc"
        assert synthesis.get("resolved_source") == "model_generated"
        assert synthesis.get("initial_parse_ok") is False
        assert synthesis.get("minimal_retry_parse_ok") is True
        assert model.calls >= 3

    def test_normalize_adhoc_spec_build_requires_implementation_phase(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        raw = {
            "name": "feature-work",
            "description": "Build a feature",
            "persona": "Software engineer",
            "phase_mode": "strict",
            "tool_guidance": "Use tools",
            "required_tools": [],
            "recommended_tools": [],
            "phases": [
                {
                    "id": "scope",
                    "description": "Define scope",
                    "depends_on": [],
                    "acceptance_criteria": "Scope documented",
                    "deliverables": ["scope.md"],
                },
                {
                    "id": "collect-notes",
                    "description": "Collect notes",
                    "depends_on": ["scope"],
                    "acceptance_criteria": "Notes collected",
                    "deliverables": ["notes.md"],
                },
                {
                    "id": "analyze",
                    "description": "Analyze notes",
                    "depends_on": ["collect-notes"],
                    "acceptance_criteria": "Analysis done",
                    "deliverables": ["analysis.md"],
                },
                {
                    "id": "deliver",
                    "description": "Deliver summary",
                    "depends_on": ["analyze"],
                    "acceptance_criteria": "Delivered",
                    "deliverables": ["final.md"],
                },
            ][:3],
        }

        normalized = app._normalize_adhoc_spec(
            raw,
            goal="build a new workflow engine",
            key="feedbeef12345678",
            available_tools=[],
            intent="build",
        )

        assert normalized["phase_mode"] == "strict"
        ids = [phase["id"] for phase in normalized["phases"]]
        assert "implement-solution" in ids
        assert "test-and-verify" in ids

    def test_normalize_adhoc_spec_uses_model_intent_over_goal_text(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        raw = {
            "intent": "writing",
            "name": "book-flow",
            "description": "Authoring workflow",
            "persona": "Author",
            "phase_mode": "strict",
            "tool_guidance": "Write cleanly.",
            "required_tools": [],
            "recommended_tools": [],
            "phases": [
                {
                    "id": "scope",
                    "description": "Define scope",
                    "depends_on": [],
                    "acceptance_criteria": "Scope documented",
                    "deliverables": ["scope.md"],
                },
            ],
        }

        normalized = app._normalize_adhoc_spec(
            raw,
            goal="build a distributed workflow engine",
            key="aabbccddeeff0011",
            available_tools=[],
        )

        ids = [phase["id"] for phase in normalized["phases"]]
        assert normalized["intent"] == "writing"
        assert "draft-content" in ids
        assert "implement-solution" not in ids
        assert normalized["source"] == "fallback_template"

    def test_normalize_adhoc_spec_preserves_model_custom_phases_when_valid(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        raw = {
            "intent": "research",
            "name": "wasted-utility-custom",
            "description": "Custom process",
            "persona": "Historian",
            "phase_mode": "strict",
            "tool_guidance": "Use primary sources.",
            "required_tools": [],
            "recommended_tools": [],
            "phases": [
                {
                    "id": "scope-hypothesis",
                    "description": "Define hypothesis and inclusion rules.",
                    "depends_on": [],
                    "acceptance_criteria": "Rules are explicit.",
                    "deliverables": ["rules.md"],
                },
                {
                    "id": "assemble-candidates",
                    "description": "Build candidate pool from diverse eras.",
                    "depends_on": ["scope-hypothesis"],
                    "acceptance_criteria": "Candidate pool complete.",
                    "deliverables": ["candidates.md"],
                },
                {
                    "id": "quantify-investment",
                    "description": "Estimate inflation-adjusted investment and opportunity cost.",
                    "depends_on": ["assemble-candidates"],
                    "acceptance_criteria": "Quantification complete.",
                    "deliverables": ["investment-model.md"],
                },
                {
                    "id": "select-twelve",
                    "description": "Finalize twelve cases with justification.",
                    "depends_on": ["quantify-investment"],
                    "acceptance_criteria": "Selection justified.",
                    "deliverables": ["twelve-cases.md"],
                },
                {
                    "id": "synthesize-manuscript-notes",
                    "description": "Produce manuscript-ready notes.",
                    "depends_on": ["select-twelve"],
                    "acceptance_criteria": "Notes ready for book drafting.",
                    "deliverables": ["manuscript-notes.md"],
                },
            ],
        }

        normalized = app._normalize_adhoc_spec(
            raw,
            goal="research overinvestment patterns for a history book",
            key="1122334455667788",
            available_tools=[],
        )

        ids = [phase["id"] for phase in normalized["phases"]]
        assert normalized["source"] == "model_generated"
        assert ids[0] == "scope-hypothesis"
        assert "quantify-investment" in ids

    @pytest.mark.asyncio
    async def test_get_or_create_adhoc_resynthesizes_fallback_template_cache(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.tui.app import LoomApp

        monkeypatch.setenv("HOME", str(tmp_path))

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools.list_tools.return_value = ["read_file", "write_file", "web_search"]

        goal = "research wasted utility history"
        key = app._adhoc_cache_key(goal)
        fallback_spec = app._fallback_adhoc_spec(
            goal,
            available_tools=app._tools.list_tools.return_value,
            intent="research",
        )
        cache_path = tmp_path / ".loom" / "cache" / "adhoc-processes" / f"{key}.yaml"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            yaml.safe_dump(
                {
                    "key": key,
                    "goal": goal,
                    "generated_at_monotonic": 1.0,
                    "saved_at": "2026-02-21T00:00:00+00:00",
                    "spec": fallback_spec,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

        custom_spec = {
            "intent": "research",
            "source": "model_generated",
            "name": "custom-research-adhoc",
            "description": "custom",
            "persona": "analyst",
            "phase_mode": "strict",
            "tool_guidance": "custom",
            "required_tools": ["read_file"],
            "recommended_tools": [],
            "phases": [
                {
                    "id": "custom-a",
                    "description": "a",
                    "depends_on": [],
                    "acceptance_criteria": "a",
                    "deliverables": ["a.md"],
                },
                {
                    "id": "custom-b",
                    "description": "b",
                    "depends_on": ["custom-a"],
                    "acceptance_criteria": "b",
                    "deliverables": ["b.md"],
                },
                {
                    "id": "custom-c",
                    "description": "c",
                    "depends_on": ["custom-b"],
                    "acceptance_criteria": "c",
                    "deliverables": ["c.md"],
                },
                {
                    "id": "custom-d",
                    "description": "d",
                    "depends_on": ["custom-c"],
                    "acceptance_criteria": "d",
                    "deliverables": ["d.md"],
                },
            ],
        }
        generated_entry = app._build_adhoc_cache_entry(key=key, goal=goal, spec=custom_spec)
        app._synthesize_adhoc_process = AsyncMock(return_value=generated_entry)

        resolved, from_cache = await app._get_or_create_adhoc_process(goal)

        assert from_cache is False
        assert resolved.process_defn.name == "custom-research-adhoc"
        app._synthesize_adhoc_process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_starts_process_run_for_active_process(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = SimpleNamespace(name="investment-analysis")
        app._start_process_run = AsyncMock()
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command(
            '/run "Analyze Tesla for investment"',
        )

        assert handled is True
        app._start_process_run.assert_awaited_once_with(
            "Analyze Tesla for investment",
            process_defn=app._process_defn,
            is_adhoc=False,
            recommended_tools=[],
        )

    @pytest.mark.asyncio
    async def test_run_close_routes_to_close_handler(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._close_process_run_from_target = AsyncMock(return_value=True)
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/run close #abc123")

        assert handled is True
        app._close_process_run_from_target.assert_awaited_once_with("#abc123")

    @pytest.mark.asyncio
    async def test_run_resume_routes_to_resume_handler(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._resume_process_run_from_target = AsyncMock(return_value=True)
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/run resume current")

        assert handled is True
        app._resume_process_run_from_target.assert_awaited_once_with("current")

    def test_process_run_restart_button_dispatches_restart_in_place(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._restart_process_run_in_place = AsyncMock(return_value=True)
        app.run_worker = MagicMock()
        event = SimpleNamespace(
            button=SimpleNamespace(id="process-run-restart-abc123"),
            stop=MagicMock(),
            prevent_default=MagicMock(),
        )

        app.on_process_run_restart_pressed(event)

        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()
        app.run_worker.assert_called_once()
        assert app.run_worker.call_args.kwargs["name"] == "process-run-restart-abc123"
        assert app.run_worker.call_args.kwargs["group"] == "process-run-restart-abc123"
        assert app.run_worker.call_args.kwargs["exclusive"] is False
        worker_coro = app.run_worker.call_args.args[0]
        assert asyncio.iscoroutine(worker_coro)
        worker_coro.close()

    @pytest.mark.asyncio
    async def test_restart_process_run_in_place_reuses_same_run_id(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            run_id="abc123",
            process_name="market-research",
            process_defn=None,
            goal="Analyze EPCOR",
            status="failed",
            task_id="cowork-9",
            run_workspace=Path("/tmp/process-run"),
            pane_id="tab-run-abc123",
            pane=MagicMock(),
            tasks=[{"id": "old", "status": "failed", "content": "old"}],
            task_labels={"old": "old"},
            last_progress_message="old",
            last_progress_at=12.0,
            worker=None,
            closed=False,
        )
        app._process_runs = {"abc123": run}
        app._update_process_run_visuals = MagicMock()
        app._refresh_process_run_outputs = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._persist_process_run_ui_state = AsyncMock()
        chat = MagicMock()
        events_panel = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        captured = {}

        def fake_run_worker(coro, **kwargs):
            captured["kwargs"] = kwargs
            coro.close()
            return MagicMock()

        app.run_worker = fake_run_worker

        restarted = await app._restart_process_run_in_place("abc123")

        assert restarted is True
        assert run.status == "queued"
        assert run.tasks == [{"id": "old", "status": "pending", "content": "old"}]
        assert run.task_labels == {"old": "old"}
        assert run.worker is not None
        assert captured["kwargs"]["name"] == "process-run-abc123"
        assert captured["kwargs"]["group"] == "process-run-abc123"
        run.pane.set_tasks.assert_called_once_with(
            [{"id": "old", "status": "pending", "content": "old"}]
        )
        app._persist_process_run_ui_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resume_process_run_continues_in_place(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            run_id="abc123",
            process_name="market-research",
            process_defn=None,
            goal="Analyze EPCOR",
            status="failed",
            task_id="cowork-9",
            run_workspace=Path("/tmp/process-run"),
            is_adhoc=True,
            recommended_tools=["ripgrep_search"],
            tasks=[
                {"id": "phase-1", "status": "failed", "content": "Scope companies"},
                {"id": "phase-2", "status": "completed", "content": "Analyze comps"},
            ],
            task_labels={
                "phase-1": "Scope companies",
                "phase-2": "Analyze comps",
                "stale": "Ignore",
            },
        )
        app._resolve_process_run_target = MagicMock(return_value=(run, None))
        app._restart_process_run_in_place = AsyncMock(return_value=True)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        resumed = await app._resume_process_run_from_target("abc123")

        assert resumed is True
        chat.add_user_message.assert_called_once_with("/run resume abc123")
        app._restart_process_run_in_place.assert_awaited_once_with(
            "abc123",
            mode="resume",
        )

    @pytest.mark.asyncio
    async def test_dynamic_process_slash_command_runs_process_directly(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()
        app._process_command_map = {"/investment-analysis": "investment-analysis"}
        loader = MagicMock()
        process_defn = SimpleNamespace(name="investment-analysis", version="1.0")
        loader.load.return_value = process_defn
        app._create_process_loader = MagicMock(return_value=loader)
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command(
            '/investment-analysis "Analyze Tesla for investment"',
        )

        assert handled is True
        loader.load.assert_called_once_with("investment-analysis")
        app._start_process_run.assert_awaited_once_with(
            "Analyze Tesla for investment",
            process_defn=process_defn,
            command_prefix="/investment-analysis",
        )

    @pytest.mark.asyncio
    async def test_start_process_run_reports_missing_delegate_tool(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = SimpleNamespace(name="investment-analysis")
        app._tools.has.return_value = False

        chat = MagicMock()
        events_panel = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        await app._start_process_run("analyze tesla")

        chat.add_info.assert_called_once()
        assert "Process orchestration is unavailable" in (
            chat.add_info.call_args.args[0]
        )

    @pytest.mark.asyncio
    async def test_start_process_run_adds_tab_and_worker(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = SimpleNamespace(name="investment-analysis")
        app._tools.has.return_value = True
        chat = MagicMock()
        events_panel = MagicMock()
        tabs = MagicMock()
        tabs.add_pane = AsyncMock()
        tabs.active = "tab-chat"

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            if selector == "#tabs":
                return tabs
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        captured = {}
        app._prepare_process_run_workspace = AsyncMock(
            return_value=Path("/tmp/process-run"),
        )

        def fake_run_worker(coro, **kwargs):
            captured["kwargs"] = kwargs
            coro.close()
            return MagicMock()

        app.run_worker = fake_run_worker

        await app._start_process_run("Analyze Tesla")

        tabs.add_pane.assert_awaited_once()
        assert captured["kwargs"]["exclusive"] is False
        assert captured["kwargs"]["group"].startswith("process-run-")
        assert len(app._process_runs) == 1
        run = next(iter(app._process_runs.values()))
        assert run.status == "queued"
        assert run.run_workspace == Path("/tmp/process-run")
        assert tabs.active == run.pane_id
        chat.add_user_message.assert_called_once_with("/run Analyze Tesla")

    @pytest.mark.asyncio
    async def test_start_process_run_respects_concurrency_limit(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = SimpleNamespace(name="investment-analysis")
        app._tools.has.return_value = True
        app._process_runs = {
            f"run-{i}": SimpleNamespace(status="running") for i in range(4)
        }

        chat = MagicMock()
        events_panel = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        app._prepare_process_run_workspace = AsyncMock(
            side_effect=[Path("/tmp/process-run-a"), Path("/tmp/process-run-b")],
        )
        app.run_worker = MagicMock()

        await app._start_process_run("Analyze Tesla")

        chat.add_info.assert_called_once()
        assert "Too many active process runs" in chat.add_info.call_args.args[0]
        app.run_worker.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_process_run_supports_multiple_concurrent_runs(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = SimpleNamespace(name="investment-analysis")
        app._tools.has.return_value = True
        chat = MagicMock()
        events_panel = MagicMock()
        tabs = MagicMock()
        tabs.add_pane = AsyncMock()
        tabs.active = "tab-chat"

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            if selector == "#tabs":
                return tabs
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        def fake_run_worker(coro, **_kwargs):
            coro.close()
            return MagicMock()

        app.run_worker = fake_run_worker

        await app._start_process_run("Analyze Tesla")
        await app._start_process_run("Analyze Nvidia")

        assert len(app._process_runs) == 2
        assert tabs.add_pane.await_count == 2
        assert chat.add_user_message.call_count == 2

    @pytest.mark.asyncio
    async def test_prepare_process_run_workspace_uses_collision_suffix(self, tmp_path):
        from loom.tui.app import LoomApp

        existing = tmp_path / "investment-analysis-analyze-tesla"
        existing.mkdir()

        app = LoomApp(
            model=None,
            tools=MagicMock(),
            workspace=tmp_path,
        )
        path = await app._prepare_process_run_workspace(
            "investment-analysis",
            "Analyze Tesla",
        )

        assert path.parent == tmp_path
        assert path.name == "investment-analysis-analyze-tesla-2"
        assert path.exists()

    @pytest.mark.asyncio
    async def test_prepare_process_run_workspace_respects_disable_flag(self, tmp_path):
        from loom.config import Config, ProcessConfig
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=None,
            tools=MagicMock(),
            workspace=tmp_path,
            config=Config(
                process=ProcessConfig(
                    tui_run_scoped_workspace_enabled=False,
                ),
            ),
        )
        path = await app._prepare_process_run_workspace(
            "investment-analysis",
            "Analyze Tesla",
        )

        assert path == tmp_path

    @pytest.mark.asyncio
    async def test_llm_process_run_folder_name_retries_transient_failure(self, tmp_path):
        from loom.config import Config, ExecutionConfig
        from loom.tui.app import LoomApp

        model = MagicMock(name="model")
        model.complete = AsyncMock(side_effect=[
            RuntimeError("temporary model error"),
            SimpleNamespace(text="market-scan-encor"),
        ])

        app = LoomApp(
            model=model,
            tools=MagicMock(),
            workspace=tmp_path,
            config=Config(
                execution=ExecutionConfig(
                    model_call_max_attempts=3,
                    model_call_retry_base_delay_seconds=0.0,
                    model_call_retry_max_delay_seconds=0.0,
                    model_call_retry_jitter_seconds=0.0,
                ),
            ),
        )

        slug = await app._llm_process_run_folder_name(
            "market-research",
            "Analyze Encor",
        )

        assert slug == "market-scan-encor"
        assert model.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_llm_process_run_folder_name_prefers_extractor_role_model(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        extractor_model = MagicMock(name="extractor-model")
        extractor_model.name = "extractor-model"
        extractor_model.configured_temperature = 0.0
        extractor_model.complete = AsyncMock(return_value=SimpleNamespace(
            text="market-scan-encor",
        ))

        active_model = MagicMock(name="active-executor-model")
        active_model.complete = AsyncMock(side_effect=AssertionError(
            "active cowork model should not be used for folder naming",
        ))

        fake_router = MagicMock()

        def _select(*, tier=1, role="executor"):
            assert role == "extractor"
            assert tier == 1
            return extractor_model

        fake_router.select = MagicMock(side_effect=_select)
        fake_router.close = AsyncMock()
        monkeypatch.setattr(
            "loom.models.router.ModelRouter.from_config",
            lambda cfg: fake_router,
        )

        app = LoomApp(
            model=active_model,
            tools=MagicMock(),
            workspace=tmp_path,
            config=Config(models={
                "primary": ModelConfig(
                    provider="ollama",
                    base_url="http://localhost:11434",
                    model="executor-model",
                    roles=["executor"],
                ),
                "extractor": ModelConfig(
                    provider="ollama",
                    base_url="http://localhost:11434",
                    model="extractor-model",
                    roles=["extractor"],
                ),
            }),
        )

        slug = await app._llm_process_run_folder_name(
            "market-research",
            "Analyze Encor",
        )

        assert slug == "market-scan-encor"
        assert extractor_model.complete.await_count == 1
        assert active_model.complete.await_count == 0
        assert fake_router.select.call_count == 1
        fake_router.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prepare_process_run_workspace_falls_back_when_llm_name_invalid(self, tmp_path):
        from loom.tui.app import LoomApp

        model = MagicMock(name="model")
        model.complete = AsyncMock(return_value=SimpleNamespace(text="///"))

        app = LoomApp(
            model=model,
            tools=MagicMock(),
            workspace=tmp_path,
        )
        path = await app._prepare_process_run_workspace(
            "market-research",
            "Analyze Encor",
        )

        assert path.parent == tmp_path
        assert path.name == "market-research-analyze-encor"
        assert path.exists()

    @pytest.mark.asyncio
    async def test_llm_process_run_folder_name_rejects_prompt_echo(self, tmp_path):
        from loom.tui.app import LoomApp

        model = MagicMock(name="model")
        model.complete = AsyncMock(return_value=SimpleNamespace(
            text="the-user-wants-a-kebab-case-folder-name-for-a-pr",
        ))
        app = LoomApp(
            model=model,
            tools=MagicMock(),
            workspace=tmp_path,
        )

        slug = await app._llm_process_run_folder_name(
            "research-process",
            "I need you to research and summarize coastal flood risk in Miami.",
        )

        assert slug == ""

    @pytest.mark.asyncio
    async def test_llm_process_run_folder_name_extracts_slug_from_noisy_response(
        self,
        tmp_path,
    ):
        from loom.tui.app import LoomApp

        model = MagicMock(name="model")
        model.complete = AsyncMock(return_value=SimpleNamespace(
            text="Slug: miami-flood-risk-summary (concise)",
        ))
        app = LoomApp(
            model=model,
            tools=MagicMock(),
            workspace=tmp_path,
        )

        slug = await app._llm_process_run_folder_name(
            "research-process",
            "Research and summarize coastal flood risk in Miami.",
        )

        assert slug == "miami-flood-risk-summary"

    @pytest.mark.asyncio
    async def test_execute_process_run_passes_workspace_and_process_override(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        process_defn = SimpleNamespace(name="investment-analysis")
        pane = MagicMock()
        app._process_runs = {
            "abc123": SimpleNamespace(
                run_id="abc123",
                process_name="investment-analysis",
                goal="Analyze Tesla",
                run_workspace=Path("/tmp/process-run"),
                process_defn=process_defn,
                pane_id="tab-run-abc123",
                pane=pane,
                status="queued",
                task_id="",
                started_at=0.0,
                ended_at=None,
                tasks=[],
                last_progress_message="",
                last_progress_at=0.0,
                worker=None,
                closed=False,
            ),
        }
        app._tools.execute = AsyncMock(return_value=SimpleNamespace(
            success=True,
            output="ok",
            error=None,
            data={"task_id": "cowork-1", "tasks": []},
        ))
        app._run_auth_profile_overrides = {"notion": "notion_marketing"}
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._refresh_workspace_tree = MagicMock()
        app.notify = MagicMock()
        chat = MagicMock()
        events_panel = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        await app._execute_process_run("abc123")

        app._tools.execute.assert_awaited_once()
        execute_call = app._tools.execute.await_args
        assert execute_call.args[0] == "delegate_task"
        payload = execute_call.args[1]
        assert payload["_process_override"] is process_defn
        assert payload["_approval_mode"] == "disabled"
        assert payload["_read_roots"] == [str(Path("/tmp").resolve())]
        assert payload["_auth_profile_overrides"] == {
            "notion": "notion_marketing"
        }
        assert payload["_resume_task_id"] == ""
        assert payload["context"]["workspace"] == "/tmp/process-run"
        assert payload["context"]["source_workspace_root"] == str(Path("/tmp").resolve())
        assert payload["context"]["requested_goal"] == "Analyze Tesla"
        assert execute_call.kwargs["workspace"] == Path("/tmp/process-run")

    @pytest.mark.asyncio
    async def test_execute_process_run_treats_failed_delegate_status_as_failure(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        process_defn = SimpleNamespace(name="market-research")
        pane = MagicMock()
        app._process_runs = {
            "abc123": SimpleNamespace(
                run_id="abc123",
                process_name="market-research",
                goal="Analyze market",
                run_workspace=Path("/tmp/process-run"),
                process_defn=process_defn,
                pane_id="tab-run-abc123",
                pane=pane,
                status="queued",
                task_id="",
                started_at=0.0,
                ended_at=None,
                tasks=[],
                task_labels={},
                last_progress_message="",
                last_progress_at=0.0,
                worker=None,
                closed=False,
            ),
        }
        app._tools.execute = AsyncMock(return_value=SimpleNamespace(
            success=True,
            output='Task failed: "Analyze market"',
            error=None,
            data={"task_id": "cowork-1", "status": "failed", "tasks": []},
        ))
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._refresh_workspace_tree = MagicMock()
        app.notify = MagicMock()
        chat = MagicMock()
        events_panel = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        await app._execute_process_run("abc123")

        run = app._process_runs["abc123"]
        assert run.status == "failed"
        chat.add_info.assert_any_call(
            "[bold #f7768e]Process run abc123 failed:[/] Process run failed."
        )

    def test_build_process_run_context_includes_recent_cowork_messages(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(
            session_state=SimpleNamespace(
                turn_count=7,
                current_focus="investment thesis",
                key_decisions=["Use DCF", "Track catalysts"],
            ),
            messages=[
                {"role": "user", "content": "Using this information, analyze Tesla."},
                {"role": "assistant", "content": "I will use the prior context."},
                {"role": "tool", "content": "ignored tool output"},
            ],
        )

        context = app._build_process_run_context(
            "Analyze Tesla",
            workspace=Path("/tmp"),
        )

        assert context["workspace"] == "/tmp"
        assert context["source_workspace_root"] == str(Path("/tmp").resolve())
        assert context["requested_goal"] == "Analyze Tesla"
        assert context["cowork"]["turn_count"] == 7
        assert context["cowork"]["recent_messages"] == [
            {"role": "user", "content": "Using this information, analyze Tesla."},
            {"role": "assistant", "content": "I will use the prior context."},
        ]

    @pytest.mark.asyncio
    async def test_close_process_run_cancelled_by_user_keeps_tab(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = SimpleNamespace(
            run_id="abc123",
            process_name="investment-analysis",
            pane_id="tab-run-abc123",
            pane=pane,
            status="running",
            started_at=0.0,
            ended_at=None,
            closed=False,
            worker=MagicMock(),
        )
        app._process_runs = {"abc123": run}
        app._confirm_close_process_run = AsyncMock(return_value=False)
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        chat = MagicMock()
        events_panel = MagicMock()
        tabs = MagicMock()
        tabs.remove_pane = AsyncMock()
        tabs.active = run.pane_id

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            if selector == "#tabs":
                return tabs
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        closed = await app._close_process_run(run)

        assert closed is False
        assert run.closed is False
        tabs.remove_pane.assert_not_awaited()
        assert "abc123" in app._process_runs
        run.worker.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_process_run_running_marks_failed_and_cancels_worker(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        worker = MagicMock()
        run = SimpleNamespace(
            run_id="abc123",
            process_name="investment-analysis",
            pane_id="tab-run-abc123",
            pane=pane,
            status="running",
            started_at=0.0,
            ended_at=None,
            closed=False,
            worker=worker,
        )
        app._process_runs = {"abc123": run}
        app._confirm_close_process_run = AsyncMock(return_value=True)
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        chat = MagicMock()
        events_panel = MagicMock()
        tabs = MagicMock()
        tabs.remove_pane = AsyncMock()
        tabs.active = "tab-chat"

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            if selector == "#tabs":
                return tabs
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        closed = await app._close_process_run(run)

        assert closed is True
        assert run.closed is True
        assert run.status == "failed"
        assert run.ended_at is not None
        pane.add_activity.assert_called_once()
        pane.add_result.assert_called_once()
        worker.cancel.assert_called_once()
        tabs.remove_pane.assert_awaited_once_with("tab-run-abc123")
        assert "abc123" not in app._process_runs
        chat.add_info.assert_called_once()
        assert "cancelled" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_persist_process_run_ui_state_serializes_run_tabs(self):
        from loom.cowork.session_state import SessionState
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        state = SessionState(session_id="sess-123", workspace="/tmp", model_name="m")
        app._session = SimpleNamespace(session_id="sess-123", session_state=state)
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._process_runs = {
            "abc123": SimpleNamespace(
                run_id="abc123",
                process_name="market-research",
                goal="Analyze EPCOR",
                status="completed",
                task_id="cowork-1",
                started_at=0.0,
                ended_at=12.0,
                tasks=[{"id": "scope", "status": "completed", "content": "Scope"}],
                task_labels={"scope": "Scope"},
                activity_log=["Run started.", "Completed scope."],
                result_log=[{"text": "done", "success": True}],
                closed=False,
                pane_id="tab-run-abc123",
                pane=MagicMock(),
            ),
        }
        app.query_one = MagicMock(return_value=SimpleNamespace(active="tab-run-abc123"))

        await app._persist_process_run_ui_state()

        app._store.update_session.assert_awaited_once()
        payload = app._store.update_session.await_args.kwargs["session_state"]
        tabs = payload["ui_state"]["process_tabs"]
        assert tabs["active_run_id"] == "abc123"
        assert tabs["runs"][0]["process_name"] == "market-research"
        assert tabs["runs"][0]["goal"] == "Analyze EPCOR"

    @pytest.mark.asyncio
    async def test_restore_process_run_tabs_from_session_state(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(
            session_state=SimpleNamespace(
                ui_state={
                    "process_tabs": {
                        "active_run_id": "abc123",
                        "runs": [
                            {
                                "run_id": "abc123",
                                "process_name": "market-research",
                                "goal": "Analyze EPCOR",
                                "status": "running",
                                "task_id": "cowork-9",
                                "elapsed_seconds": 42.0,
                                "tasks": [
                                    {
                                        "id": "scope",
                                        "status": "in_progress",
                                        "content": "Scope companies",
                                    },
                                ],
                                "task_labels": {"scope": "Scope companies"},
                                "activity_log": ["Run started."],
                                "result_log": [],
                            },
                        ],
                    },
                },
            ),
        )
        app._process_runs = {}
        app._refresh_sidebar_progress_summary = MagicMock()
        app._refresh_process_run_outputs = MagicMock()
        app._update_process_run_visuals = MagicMock()
        loader = MagicMock()
        loader.load.return_value = SimpleNamespace(
            name="market-research",
            phases=[],
            get_deliverables=lambda: {},
        )
        app._create_process_loader = MagicMock(return_value=loader)

        tabs = SimpleNamespace(active="tab-chat", add_pane=AsyncMock(), remove_pane=AsyncMock())
        app.query_one = MagicMock(return_value=tabs)
        chat = MagicMock()

        await app._restore_process_run_tabs(chat)

        assert len(app._process_runs) == 1
        run = app._process_runs["abc123"]
        assert run.process_name == "market-research"
        assert run.status == "failed"  # interrupted running runs cannot be resumed
        tabs.add_pane.assert_awaited_once()
        assert tabs.active == run.pane_id
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Restored Process Tabs" in message
        assert "Count:[/] 1" in message

    def test_process_progress_event_scoped_to_run_tab(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = SimpleNamespace(
            run_id="abc123",
            process_name="investment-analysis",
            goal="Analyze Tesla",
            pane_id="tab-run-abc123",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
        )
        app._process_runs = {"abc123": run}
        app._update_process_run_visuals = MagicMock()

        sidebar = MagicMock()
        chat = MagicMock()
        events_panel = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#sidebar":
                return sidebar
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        payload = {
            "event_type": "subtask_started",
            "event_data": {"subtask_id": "company-screening"},
            "tasks": [
                {
                    "id": "company-screening",
                    "status": "in_progress",
                    "content": "Build company overview",
                },
            ],
        }
        app._on_process_progress_event(payload, run_id="abc123")

        pane.set_tasks.assert_called_once_with(payload["tasks"])
        pane.add_activity.assert_called_once()
        sidebar.update_tasks.assert_called_once()
        summary_rows = sidebar.update_tasks.call_args.args[0]
        assert summary_rows[0]["id"] == "process-run-abc123"
        assert summary_rows[0]["status"] == "in_progress"
        assert "investment-analysis #abc123 Running" in summary_rows[0]["content"]
        chat.add_info.assert_not_called()
        events_panel.add_event.assert_called_once()

    def test_process_progress_keeps_stable_phase_labels(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        process_defn = SimpleNamespace(
            phases=[
                SimpleNamespace(
                    id="scope-companies",
                    description=(
                        "Interpret requested company name(s), normalize legal/entity "
                        "naming, and define product/service lines."
                    ),
                ),
            ],
            get_deliverables=lambda: {},
        )
        run = SimpleNamespace(
            run_id="abc123",
            process_name="market-research",
            goal="Analyze market",
            process_defn=process_defn,
            pane_id="tab-run-abc123",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            task_labels={},
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
        )
        app._process_runs = {"abc123": run}
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()

        sidebar = MagicMock()
        events_panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#sidebar":
                return sidebar
            if selector == "#events-panel":
                return events_panel
            if selector == "#chat-log":
                return chat
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        app._on_process_progress_event(
            {
                "event_type": "subtask_started",
                "event_data": {"subtask_id": "scope-companies"},
                "tasks": [
                    {
                        "id": "scope-companies",
                        "status": "in_progress",
                        "content": "Short working label",
                    },
                ],
            },
            run_id="abc123",
        )
        app._on_process_progress_event(
            {
                "event_type": "subtask_completed",
                "event_data": {"subtask_id": "scope-companies"},
                "tasks": [
                    {
                        "id": "scope-companies",
                        "status": "completed",
                        "content": (
                            "**Subtask Complete** Created research-scope.md and "
                            "company-service-map.csv with long narrative summary."
                        ),
                    },
                ],
            },
            run_id="abc123",
        )

        final_tasks = pane.set_tasks.call_args.args[0]
        assert final_tasks[0]["id"] == "scope-companies"
        assert final_tasks[0]["status"] == "completed"
        assert "normalize legal/entity naming" in final_tasks[0]["content"]
        assert "Subtask Complete" not in final_tasks[0]["content"]

    def test_process_progress_updates_outputs_panel(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        (tmp_path / "research-scope.md").write_text("ok")

        pane = MagicMock()
        process_defn = SimpleNamespace(
            phases=[
                SimpleNamespace(id="scope-companies", description="Scope companies"),
                SimpleNamespace(id="map-geographies", description="Map geographies"),
            ],
            get_deliverables=lambda: {
                "scope-companies": ["research-scope.md"],
                "map-geographies": ["geography-footprint.csv"],
            },
        )
        run = SimpleNamespace(
            run_id="abc123",
            process_name="market-research",
            goal="Analyze market",
            process_defn=process_defn,
            pane_id="tab-run-abc123",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            task_labels={},
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
        )
        app._process_runs = {"abc123": run}
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()

        sidebar = MagicMock()
        events_panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#sidebar":
                return sidebar
            if selector == "#events-panel":
                return events_panel
            if selector == "#chat-log":
                return chat
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        app._on_process_progress_event(
            {
                "event_type": "subtask_completed",
                "event_data": {"subtask_id": "map-geographies"},
                "tasks": [
                    {
                        "id": "scope-companies",
                        "status": "completed",
                        "content": "Scope companies",
                    },
                    {
                        "id": "map-geographies",
                        "status": "completed",
                        "content": "Map geographies",
                    },
                ],
            },
            run_id="abc123",
        )

        output_rows = pane.set_outputs.call_args.args[0]
        by_content = {row["content"]: row for row in output_rows}
        assert "research-scope.md (scope-companies)" in by_content
        assert by_content["research-scope.md (scope-companies)"]["status"] == (
            "completed"
        )
        assert "geography-footprint.csv (map-geographies) (missing)" in by_content
        assert by_content["geography-footprint.csv (map-geographies) (missing)"]["status"] == (
            "failed"
        )

    def test_process_progress_outputs_use_run_workspace(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        run_workspace = tmp_path / "market-research-run"
        run_workspace.mkdir(parents=True, exist_ok=True)
        (run_workspace / "market-trends.md").write_text("ok")

        pane = MagicMock()
        process_defn = SimpleNamespace(
            phases=[
                SimpleNamespace(
                    id="analyze-market-trends",
                    description="Analyze market trends",
                ),
            ],
            get_deliverables=lambda: {
                "analyze-market-trends": [
                    "market-trends.md",
                    "trend-dataset.csv",
                ],
            },
        )
        run = SimpleNamespace(
            run_id="abc124",
            process_name="market-research",
            goal="Analyze trends",
            run_workspace=run_workspace,
            process_defn=process_defn,
            pane_id="tab-run-abc124",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            task_labels={},
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
        )
        app._process_runs = {"abc124": run}
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()

        sidebar = MagicMock()
        events_panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#sidebar":
                return sidebar
            if selector == "#events-panel":
                return events_panel
            if selector == "#chat-log":
                return chat
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        app._on_process_progress_event(
            {
                "event_type": "subtask_completed",
                "event_data": {"subtask_id": "analyze-market-trends"},
                "tasks": [
                    {
                        "id": "analyze-market-trends",
                        "status": "completed",
                        "content": "Analyze market trends",
                    },
                ],
            },
            run_id="abc124",
        )

        output_rows = pane.set_outputs.call_args.args[0]
        by_content = {row["content"]: row for row in output_rows}
        assert "market-trends.md (analyze-market-trends)" in by_content
        assert by_content["market-trends.md (analyze-market-trends)"]["status"] == "completed"
        assert "trend-dataset.csv (analyze-market-trends) (missing)" in by_content
        assert by_content["trend-dataset.csv (analyze-market-trends) (missing)"]["status"] == (
            "failed"
        )

    def test_adhoc_outputs_keep_declared_deliverables_after_plan_ready(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )

        pane = MagicMock()
        process_defn = SimpleNamespace(
            phases=[
                SimpleNamespace(id="inspect-pitch", description="Inspect pitch and constraints"),
                SimpleNamespace(id="map-eras", description="Map historical eras"),
            ],
            get_deliverables=lambda: {
                "inspect-pitch": ["pitch-analysis.md"],
                "map-eras": ["era-coverage-matrix.csv"],
            },
        )
        run = SimpleNamespace(
            run_id="abc126",
            process_name="wasted-utility-research-adhoc",
            goal="Research historical over-investment cases",
            run_workspace=tmp_path,
            process_defn=process_defn,
            pane_id="tab-run-abc126",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            task_labels={},
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
            is_adhoc=True,
        )
        app._process_runs = {"abc126": run}
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()

        sidebar = MagicMock()
        events_panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#sidebar":
                return sidebar
            if selector == "#events-panel":
                return events_panel
            if selector == "#chat-log":
                return chat
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        app._on_process_progress_event(
            {
                "event_type": "task_plan_ready",
                "tasks": [
                    {
                        "id": "plan-step-1",
                        "status": "in_progress",
                        "content": "Read and analyze pitch.md",
                    },
                    {
                        "id": "plan-step-2",
                        "status": "pending",
                        "content": "Establish historical coverage matrix",
                    },
                ],
            },
            run_id="abc126",
        )

        output_rows = pane.set_outputs.call_args.args[0]
        by_content = {row["content"]: row for row in output_rows}
        assert "pitch-analysis.md (inspect-pitch) (planned)" in by_content
        assert "era-coverage-matrix.csv (map-eras) (planned)" in by_content
        assert all("(expected output)" not in row["content"] for row in output_rows)

    def test_adhoc_outputs_fallback_to_task_rows_without_deliverables(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        pane = MagicMock()
        process_defn = SimpleNamespace(
            phases=[],
            get_deliverables=lambda: {},
        )
        run = SimpleNamespace(
            run_id="abc127",
            process_name="wasted-utility-research-adhoc",
            goal="Research historical over-investment cases",
            process_defn=process_defn,
            pane_id="tab-run-abc127",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            task_labels={},
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
            is_adhoc=True,
        )
        app._process_runs = {"abc127": run}
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()

        sidebar = MagicMock()
        events_panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#sidebar":
                return sidebar
            if selector == "#events-panel":
                return events_panel
            if selector == "#chat-log":
                return chat
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        app._on_process_progress_event(
            {
                "event_type": "task_plan_ready",
                "tasks": [
                    {
                        "id": "plan-step-1",
                        "status": "in_progress",
                        "content": "Read and analyze pitch.md",
                    },
                ],
            },
            run_id="abc127",
        )

        output_rows = pane.set_outputs.call_args.args[0]
        assert output_rows == [
            {
                "id": "adhoc-output-1",
                "status": "in_progress",
                "content": "Read and analyze pitch.md (expected output)",
            },
        ]

    def test_process_progress_outputs_detect_existing_file_when_task_id_drifts(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        run_workspace = tmp_path / "campaign-slogans-run"
        run_workspace.mkdir(parents=True, exist_ok=True)
        (run_workspace / "slogan-longlist.csv").write_text("ok")

        pane = MagicMock()
        process_defn = SimpleNamespace(
            phases=[
                SimpleNamespace(
                    id="slogan-divergence",
                    description=(
                        "Generate a high-volume longlist of slogan/tagline options "
                        "across all territories and devices before filtering."
                    ),
                ),
            ],
            get_deliverables=lambda: {
                "slogan-divergence": ["slogan-longlist.csv"],
            },
        )
        run = SimpleNamespace(
            run_id="abc125",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
            run_workspace=run_workspace,
            process_defn=process_defn,
            pane_id="tab-run-abc125",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            task_labels={},
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
        )
        app._process_runs = {"abc125": run}
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()

        sidebar = MagicMock()
        events_panel = MagicMock()
        chat = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#sidebar":
                return sidebar
            if selector == "#events-panel":
                return events_panel
            if selector == "#chat-log":
                return chat
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        app._on_process_progress_event(
            {
                "event_type": "subtask_started",
                "event_data": {"subtask_id": "generate-slogan-longlist"},
                "tasks": [
                    {
                        "id": "generate-slogan-longlist",
                        "status": "in_progress",
                        "content": (
                            "Generate a high-volume longlist of slogan/tagline options "
                            "across all territories and devices before filtering."
                        ),
                    },
                ],
            },
            run_id="abc125",
        )

        output_rows = pane.set_outputs.call_args.args[0]
        by_content = {row["content"]: row for row in output_rows}
        assert "slogan-longlist.csv (slogan-divergence)" in by_content
        assert by_content["slogan-longlist.csv (slogan-divergence)"]["status"] == (
            "completed"
        )

    def test_process_progress_event_refreshes_tree_on_subtask_completion(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        sidebar = MagicMock()
        app.query_one = MagicMock(return_value=sidebar)

        app._on_process_progress_event({
            "event_type": "subtask_completed",
            "tasks": [
                {"id": "company-screening", "status": "completed", "content": "Done"},
            ],
        })

        sidebar.update_tasks.assert_called_once()
        sidebar.refresh_workspace_tree.assert_called_once()

    def test_process_progress_event_refreshes_tree_on_document_write_tool(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = SimpleNamespace(
            run_id="abc123",
            process_name="investment-analysis",
            goal="Analyze Tesla",
            pane_id="tab-run-abc123",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
        )
        app._process_runs = {"abc123": run}
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._refresh_workspace_tree = MagicMock()

        app._on_process_progress_event(
            {
                "event_type": "tool_call_completed",
                "event_data": {
                    "tool": "document_write",
                    "subtask_id": "investment-memo",
                    "success": True,
                },
            },
            run_id="abc123",
        )

        app._refresh_workspace_tree.assert_called_once()

    def test_process_progress_event_streams_to_chat(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        sidebar = MagicMock()
        chat = MagicMock()
        events_panel = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#sidebar":
                return sidebar
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        app._on_process_progress_event({
            "event_type": "subtask_started",
            "event_data": {"subtask_id": "company-screening"},
            "tasks": [
                {
                    "id": "company-screening",
                    "status": "in_progress",
                    "content": "Build company overview",
                },
            ],
        })

        sidebar.update_tasks.assert_called_once()
        chat.add_info.assert_called_once()
        assert "Started company-screening" in chat.add_info.call_args.args[0]
        events_panel.add_event.assert_called_once()

    def test_process_progress_event_uses_stable_run_label_for_messages(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            run_id="abc123",
            process_name="market-research",
            goal="Analyze market",
            process_defn=None,
            pane_id="tab-run-abc123",
            pane=MagicMock(),
            status="running",
            task_id="task-1",
            started_at=0.0,
            ended_at=None,
            tasks=[
                {
                    "id": "environmental-scan",
                    "status": "in_progress",
                    "content": "Perform environmental scans for priority markets",
                },
            ],
            task_labels={
                "environmental-scan": "Perform environmental scans for priority markets",
            },
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
        )

        message = app._format_process_progress_event(
            {
                "event_type": "tool_call_started",
                "event_data": {
                    "subtask_id": "environmental-scan",
                    "tool": "web_search",
                },
                "tasks": [
                    {
                        "id": "environmental-scan",
                        "status": "in_progress",
                        "content": (
                            "Verification inconclusive: could not parse verifier output."
                        ),
                    },
                ],
            },
            run=run,
        )

        assert message is not None
        assert "Verification inconclusive" not in message
        assert "environmental-scan - Perform environmental scans" in message

    def test_mark_process_run_failed_shows_timeout_guidance(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        sidebar = MagicMock()
        app.query_one = MagicMock(return_value=sidebar)

        app._mark_process_run_failed("Tool 'delegate_task' timed out after 600s")

        sidebar.update_tasks.assert_called_once()
        payload = sidebar.update_tasks.call_args.args[0]
        assert payload[0]["status"] == "failed"
        assert "delegate_task_timeout_seconds" in payload[0]["content"]
        assert "LOOM_DELE" in payload[0]["content"]


class TestMCPSlashCommands:
    @pytest.mark.asyncio
    async def test_mcp_without_args_opens_manager(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._open_mcp_manager_screen = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp")

        assert handled is True
        app._open_mcp_manager_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_list_uses_manager(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        merged = MagicMock()
        merged.as_views.return_value = [
            SimpleNamespace(
                alias="demo",
                server=SimpleNamespace(enabled=True),
                source="user",
            )
        ]
        manager.load.return_value = merged
        app._mcp_manager = MagicMock(return_value=manager)
        app._render_mcp_list = MagicMock(return_value="mcp-catalog")
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp list")

        assert handled is True
        manager.load.assert_called_once()
        merged.as_views.assert_called_once()
        app._render_mcp_list.assert_called_once()
        chat.add_info.assert_called_once_with("mcp-catalog")

    @pytest.mark.asyncio
    async def test_mcp_list_shows_legacy_migration_hint(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        merged = MagicMock()
        merged.as_views.return_value = [
            SimpleNamespace(
                alias="legacy_demo",
                server=SimpleNamespace(enabled=True),
                source="legacy",
            )
        ]
        manager.load.return_value = merged
        app._mcp_manager = MagicMock(return_value=manager)
        app._render_mcp_list = MagicMock(return_value="mcp-catalog")
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp list")

        assert handled is True
        chat.add_info.assert_called_once()
        assert "loom mcp migrate" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_show_requires_alias(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._mcp_manager = MagicMock(return_value=MagicMock())
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp show")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Usage" in message
        assert "/mcp show" in message
        assert "<alias>" in message

    @pytest.mark.asyncio
    async def test_mcp_show_missing_alias(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        manager.get_view.return_value = None
        app._mcp_manager = MagicMock(return_value=manager)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp show missing")

        assert handled is True
        manager.get_view.assert_called_once_with("missing")
        chat.add_info.assert_called_once()
        assert "MCP server not found" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_enable_edits_and_reloads(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        manager.edit_server.return_value = (Path("/tmp/mcp.toml"), MagicMock())
        app._mcp_manager = MagicMock(return_value=manager)
        app._reload_mcp_runtime = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp enable demo")

        assert handled is True
        manager.edit_server.assert_called_once()
        app._reload_mcp_runtime.assert_awaited_once()
        chat.add_info.assert_called_once()
        assert "enabled" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_remove_edits_and_reloads(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        manager.remove_server.return_value = Path("/tmp/mcp.toml")
        app._mcp_manager = MagicMock(return_value=manager)
        app._reload_mcp_runtime = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp remove demo")

        assert handled is True
        manager.remove_server.assert_called_once_with("demo")
        app._reload_mcp_runtime.assert_awaited_once()
        chat.add_info.assert_called_once()
        assert "removed" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_test_reports_discovered_tools(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        view = SimpleNamespace(alias="demo")
        manager.probe_server.return_value = (
            view,
            [{"name": "echo"}, {"name": "ping"}],
        )
        app._mcp_manager = MagicMock(return_value=manager)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp test demo")

        assert handled is True
        manager.probe_server.assert_called_once_with("demo")
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Tools discovered: 2" in message
        assert "echo" in message
        assert "ping" in message

    @pytest.mark.asyncio
    async def test_mcp_manage_routes_to_manager_screen(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._open_mcp_manager_screen = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp manage")

        assert handled is True
        app._open_mcp_manager_screen.assert_called_once()

    def test_mcp_manager_inherits_explicit_config_paths(self, monkeypatch):
        from loom.tui.app import LoomApp

        captured: dict[str, object] = {}

        class DummyManager:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("loom.mcp.config.MCPConfigManager", DummyManager)

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=MagicMock(),
            explicit_mcp_path=Path("/tmp/override-mcp.toml"),
            legacy_config_path=Path("/tmp/loom.toml"),
        )

        manager = app._mcp_manager()

        assert isinstance(manager, DummyManager)
        assert captured["explicit_path"] == Path("/tmp/override-mcp.toml")
        assert captured["legacy_config_path"] == Path("/tmp/loom.toml")

    @pytest.mark.asyncio
    async def test_mcp_add_parses_flags_and_adds_server(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        manager.add_server.return_value = Path("/tmp/mcp.toml")
        app._mcp_manager = MagicMock(return_value=manager)
        app._reload_mcp_runtime = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command(
            "/mcp add demo --command python --arg -m --arg demo.server "
            "--env-ref TOKEN=REAL_TOKEN --timeout 45",
        )

        assert handled is True
        manager.add_server.assert_called_once()
        app._reload_mcp_runtime.assert_awaited_once()
        chat.add_info.assert_called_once()
        assert "added" in chat.add_info.call_args.args[0]


class TestAuthSlashCommands:
    @pytest.mark.asyncio
    async def test_auth_without_args_opens_manager(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()
        app._open_auth_manager_screen = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/auth")

        assert handled is True
        app._open_auth_manager_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_auth_use_sets_run_override(self, monkeypatch):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()

        merged = SimpleNamespace(
            config=SimpleNamespace(
                profiles={
                    "notion_marketing": SimpleNamespace(provider="notion"),
                },
                defaults={},
                mcp_alias_profiles={},
            ),
            workspace_defaults={},
            user_path=Path("/tmp/user-auth.toml"),
            explicit_path=None,
            workspace_defaults_path=Path("/tmp/.loom/auth.defaults.toml"),
        )
        monkeypatch.setattr(
            "loom.auth.config.load_merged_auth_config",
            lambda **_kwargs: merged,
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/auth use notion=notion_marketing")

        assert handled is True
        assert app._run_auth_profile_overrides == {
            "notion": "notion_marketing"
        }
        chat.add_info.assert_called_once()
        assert "notion -> notion_marketing" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_auth_add_and_remove_profile(self, monkeypatch):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        monkeypatch.setattr(
            "loom.auth.config.resolve_auth_write_path",
            lambda explicit_path=None: Path("/tmp/auth.toml"),
        )
        monkeypatch.setattr(
            "loom.auth.config.upsert_auth_profile",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "loom.auth.config.remove_auth_profile",
            lambda *args, **kwargs: None,
        )

        handled_add = await app._handle_slash_command(
            "/auth add notion_marketing --provider notion --mode oauth2_pkce "
            "--token-ref keychain://loom/notion/notion_marketing/tokens",
        )
        assert handled_add is True
        assert "Added auth profile" in chat.add_info.call_args.args[0]

        handled_remove = await app._handle_slash_command(
            "/auth remove notion_marketing",
        )
        assert handled_remove is True
        assert "Removed auth profile" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_auth_manage_routes_to_manager_screen(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._open_auth_manager_screen = MagicMock()
        app._refresh_process_command_index = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/auth manage")

        assert handled is True
        app._open_auth_manager_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_auth_routes_subcommand_is_no_longer_supported(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/auth routes")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Usage" in message
        assert "/auth" in message
        assert "routes" not in message

    @pytest.mark.asyncio
    async def test_auth_edit_uses_explicit_auth_path(self, monkeypatch):
        from loom.tui.app import LoomApp

        explicit_path = Path("/tmp/override-auth.toml")
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            explicit_auth_path=explicit_path,
        )
        app._refresh_process_command_index = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        captured: dict[str, object] = {}
        current = SimpleNamespace(
            profile_id="notion_marketing",
            provider="notion",
            mode="oauth2_pkce",
            account_label="",
            secret_ref="",
            token_ref="",
            scopes=[],
            env={},
            command="",
            auth_check=[],
            metadata={},
        )

        def _fake_load_merged_auth_config(*, workspace, explicit_path=None):
            captured["load_explicit_path"] = explicit_path
            return SimpleNamespace(
                config=SimpleNamespace(profiles={"notion_marketing": current}),
            )

        def _fake_resolve_auth_write_path(*, explicit_path=None):
            captured["write_explicit_path"] = explicit_path
            return Path("/tmp/auth.toml")

        monkeypatch.setattr(
            "loom.auth.config.load_merged_auth_config",
            _fake_load_merged_auth_config,
        )
        monkeypatch.setattr(
            "loom.auth.config.resolve_auth_write_path",
            _fake_resolve_auth_write_path,
        )
        monkeypatch.setattr(
            "loom.auth.config.upsert_auth_profile",
            lambda *args, **kwargs: None,
        )

        handled = await app._handle_slash_command(
            "/auth edit notion_marketing --label Marketing",
        )

        assert handled is True
        assert captured["load_explicit_path"] == explicit_path
        assert captured["write_explicit_path"] == explicit_path

    @pytest.mark.asyncio
    async def test_auth_use_rejects_mcp_selector(self, monkeypatch):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        def _fake_load_merged_auth_config(**_kwargs):
            return SimpleNamespace(
                config=SimpleNamespace(
                    profiles={
                        "notion_marketing": SimpleNamespace(provider="notion"),
                    },
                    defaults={},
                    mcp_alias_profiles={},
                ),
                workspace_defaults={},
                user_path=Path("/tmp/user-auth.toml"),
                explicit_path=None,
                workspace_defaults_path=Path("/tmp/.loom/auth.defaults.toml"),
            )

        monkeypatch.setattr(
            "loom.auth.config.load_merged_auth_config",
            _fake_load_merged_auth_config,
        )

        handled = await app._handle_slash_command(
            "/auth use mcp.notion=notion_marketing",
        )

        assert handled is True
        chat.add_info.assert_called_once()
        assert "MCP selectors are no longer supported" in chat.add_info.call_args.args[0]


class TestCommandPaletteProcessActions:
    def test_ctrl_r_binding_registered(self):
        from loom.tui.app import LoomApp

        keys = {binding.key for binding in LoomApp.BINDINGS}
        assert "ctrl+r" in keys

    @pytest.mark.asyncio
    async def test_reload_workspace_action_calls_refresh(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_workspace_tree = MagicMock()
        app.notify = MagicMock()

        await app.action_loom_command("reload_workspace")

        app._refresh_workspace_tree.assert_called_once()
        app.notify.assert_called_once_with("Workspace reloaded", timeout=2)

    @pytest.mark.asyncio
    async def test_show_learned_patterns_queries_behavioral(self, monkeypatch):
        from loom.learning.manager import LearnedPattern
        from loom.tui.app import LoomApp
        from loom.tui.screens.learned import LearnedScreen

        pattern = LearnedPattern(
            pattern_type="behavioral_gap",
            pattern_key="run-tests",
            data={"description": "Run tests before reporting completion."},
        )

        class FakeLearningManager:
            def __init__(self, db):
                self._db = db

            async def query_behavioral(self, limit=15):
                assert limit == 50
                return [pattern]

        monkeypatch.setattr(
            "loom.learning.manager.LearningManager", FakeLearningManager,
        )

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._db = MagicMock()

        pushed: dict[str, object] = {}

        def _push(screen, callback=None):
            pushed["screen"] = screen
            pushed["callback"] = callback

        app.push_screen = _push

        await app._show_learned_patterns()

        assert isinstance(pushed.get("screen"), LearnedScreen)
        assert pushed["screen"]._patterns == [pattern]
        assert callable(pushed.get("callback"))

    @pytest.mark.asyncio
    async def test_process_info_action(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = SimpleNamespace(name="marketing-strategy")
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        await app.action_loom_command("process_info")

        chat.add_info.assert_called_once_with("Active process: marketing-strategy")

    @pytest.mark.asyncio
    async def test_process_list_action(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._render_process_catalog = MagicMock(return_value="catalog")
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        await app.action_loom_command("process_list")

        app._render_process_catalog.assert_called_once()
        chat.add_info.assert_called_once_with("catalog")

    @pytest.mark.asyncio
    async def test_process_off_action_no_active(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_name = None
        app._process_defn = None
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        await app.action_loom_command("process_off")

        chat.add_info.assert_called_once_with("No active process.")

    @pytest.mark.asyncio
    async def test_process_off_action_reloads(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_name = "marketing-strategy"
        app._process_defn = SimpleNamespace(name="marketing-strategy")
        app._reload_session_for_process_change = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        await app.action_loom_command("process_off")

        assert app._process_name is None
        assert app._process_defn is None
        app._reload_session_for_process_change.assert_awaited_once()
        assert "Active process: none" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_palette_slash_backed_actions_delegate_to_slash_handler(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._handle_slash_command = AsyncMock(return_value=True)

        await app.action_loom_command("setup")
        await app.action_loom_command("session_info")
        await app.action_loom_command("new_session")
        await app.action_loom_command("sessions_list")
        await app.action_loom_command("mcp_list")
        await app.action_loom_command("learned_patterns")

        calls = [c.args[0] for c in app._handle_slash_command.await_args_list]
        assert calls == [
            "/setup",
            "/session",
            "/new",
            "/sessions",
            "/mcp list",
            "/learned",
        ]

    @pytest.mark.asyncio
    async def test_palette_prompt_actions_prefill_input(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._prefill_user_input = MagicMock()

        await app.action_loom_command("process_use_prompt")
        await app.action_loom_command("run_prompt")
        await app.action_loom_command("resume_prompt")

        calls = [c.args[0] for c in app._prefill_user_input.call_args_list]
        assert calls == ["/process use ", "/run ", "/resume "]

    @pytest.mark.asyncio
    async def test_palette_dynamic_process_prompt_action_prefills_input(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._prefill_user_input = MagicMock()

        await app.action_loom_command("process_run_prompt:investment-analysis")

        app._prefill_user_input.assert_called_once_with(
            "/investment-analysis "
        )

    def test_dynamic_process_palette_entries_filter_reserved_names(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        loader = MagicMock()
        loader.list_available.return_value = [
            {"name": "run", "version": "1.0", "description": "bad collision"},
            {
                "name": "investment-analysis",
                "version": "1.0",
                "description": "valid process",
            },
        ]
        app._create_process_loader = MagicMock(return_value=loader)

        entries = app.iter_dynamic_process_palette_entries()

        assert entries == [(
            "Run investment-analysis…",
            "process_run_prompt:investment-analysis",
            "Prefill /investment-analysis for direct process execution",
        )]

    def test_prefill_user_input_updates_value_and_hint(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        input_widget = SimpleNamespace(value="", cursor_position=0, focus=MagicMock())
        app.query_one = MagicMock(return_value=input_widget)
        app._render_slash_hint = MagicMock(return_value="hint text")
        app._set_slash_hint = MagicMock()

        app._prefill_user_input("/process use ")

        assert input_widget.value == "/process use "
        assert input_widget.cursor_position == len("/process use ")
        input_widget.focus.assert_called_once()
        app._render_slash_hint.assert_called_once_with("/process use ")
        app._set_slash_hint.assert_called_once_with("hint text")

    def test_slash_tab_completion_cycles_forward(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        input_widget = SimpleNamespace(value="/s", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/setup"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/session"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/sessions"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/setup"

    def test_slash_tab_completion_cycles_backward(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        input_widget = SimpleNamespace(value="/s", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)

        assert app._apply_slash_tab_completion(reverse=True) is True
        assert input_widget.value == "/sessions"

    def test_slash_tab_completion_ignores_non_slash_or_args(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        input_widget = SimpleNamespace(value="hello", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)
        assert app._apply_slash_tab_completion(reverse=False) is False

        input_widget.value = "/resume abc"
        assert app._apply_slash_tab_completion(reverse=False) is False

    def test_slash_tab_completion_process_use_prefix(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[
                {"name": "investment-analysis"},
                {"name": "marketing-strategy"},
            ])
        ))
        input_widget = SimpleNamespace(value="/process use inv", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/process use investment-analysis"

    def test_slash_tab_completion_process_use_cycles(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[
                {"name": "marketing-strategy"},
                {"name": "market-research"},
                {"name": "investment-analysis"},
            ])
        ))
        input_widget = SimpleNamespace(value="/process use m", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/process use marketing-strategy"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/process use market-research"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/process use marketing-strategy"

    def test_slash_tab_completion_process_use_no_matches(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[
                {"name": "investment-analysis"},
            ])
        ))
        input_widget = SimpleNamespace(value="/process use zz", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)

        assert app._apply_slash_tab_completion(reverse=False) is False

    @pytest.mark.asyncio
    async def test_tab_key_is_captured_for_slash_autocomplete(self):
        from textual.widgets import Input

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test() as pilot:
            await pilot.press("/")
            await pilot.press("s")
            await pilot.pause()

            input_widget = app.query_one("#user-input", Input)
            assert input_widget.value == "/s"

            await pilot.press("tab")
            await pilot.pause()
            assert input_widget.value == "/setup"

            await pilot.press("tab")
            await pilot.pause()
            assert input_widget.value == "/session"

    @pytest.mark.asyncio
    async def test_ctrl_w_closes_process_tab_from_input_focus(self):
        from textual.widgets import Input

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()
        called = {"count": 0}
        app.action_close_process_tab = lambda: called.__setitem__(  # noqa: B023
            "count", called["count"] + 1,
        )

        async with app.run_test() as pilot:
            input_widget = app.query_one("#user-input", Input)
            input_widget.focus()
            await pilot.pause()
            await pilot.press("ctrl+w")
            await pilot.pause()

        assert called["count"] == 1

    @pytest.mark.asyncio
    async def test_action_close_process_tab_starts_nonexclusive_worker(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._close_process_run_from_target = AsyncMock(return_value=True)
        captured: dict = {}

        def _capture_worker(coro, **kwargs):
            captured["coro"] = coro
            captured["kwargs"] = kwargs
            return MagicMock()

        app.run_worker = MagicMock(side_effect=_capture_worker)

        app.action_close_process_tab()

        assert app.run_worker.call_count == 1
        assert captured["kwargs"]["group"] == "close-process-tab"
        assert captured["kwargs"]["exclusive"] is False
        assert app._close_process_tab_inflight is True

        await captured["coro"]
        app._close_process_run_from_target.assert_awaited_once_with("current")
        assert app._close_process_tab_inflight is False

    def test_action_close_process_tab_ignores_reentry(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._close_process_run_from_target = AsyncMock(return_value=True)
        first_coro: dict = {}

        def _capture_worker(coro, **_kwargs):
            first_coro["coro"] = coro
            return MagicMock()

        app.run_worker = MagicMock(side_effect=_capture_worker)

        app.action_close_process_tab()
        app.action_close_process_tab()

        assert app.run_worker.call_count == 1
        # Close pending coroutine to avoid unawaited coroutine warnings.
        first_coro["coro"].close()


class TestSlashHelp:
    def test_help_lines_include_resume_and_aliases(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        rendered = "\n".join(app._help_lines())
        assert "/resume <session-id-prefix>" in rendered
        assert "/run <goal|close" in rendered
        assert "resume <run-id-prefix|current>" in rendered
        assert "run-id-prefix" in rendered
        assert "/quit (aliases: /exit, /q)" in rendered
        assert "/setup" in rendered
        assert "Ctrl+R reload workspace" in rendered
        assert "Ctrl+W close run tab" in rendered

    @pytest.mark.asyncio
    async def test_resume_without_arg_shows_usage(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/resume")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Usage" in message
        assert "/resume" in message
        assert "<session-id-prefix>" in message


class TestFileViewer:
    def test_renderer_registry_supports_common_types(self):
        from loom.tui.screens.file_viewer import resolve_file_renderer

        assert resolve_file_renderer(Path("README.md")) is not None
        assert resolve_file_renderer(Path("README.markdown")) is not None
        assert resolve_file_renderer(Path("src/main.ts")) is not None
        assert resolve_file_renderer(Path("styles/site.css")) is not None
        assert resolve_file_renderer(Path("data.json")) is not None
        assert resolve_file_renderer(Path("report.csv")) is not None
        assert resolve_file_renderer(Path("slides.pptx")) is not None
        assert resolve_file_renderer(Path("paper.pdf")) is not None
        assert resolve_file_renderer(Path("image.png")) is not None
        assert resolve_file_renderer(Path("Dockerfile")) is not None
        assert resolve_file_renderer(Path("README.foobar")) is None

    def test_file_viewer_loads_markdown_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "notes.md"
        doc.write_text("# Hello\n\nThis is a markdown preview.\n", encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_file_viewer_loads_json_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "data.json"
        doc.write_text('{"b":2,"a":1}', encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_file_viewer_loads_csv_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "table.csv"
        doc.write_text("name,value\nfoo,1\nbar,2\n", encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_file_viewer_loads_html_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "index.html"
        doc.write_text(
            "<html><body><h1>Title</h1><p>Hello world</p></body></html>",
            encoding="utf-8",
        )

        screen = FileViewerScreen(doc, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_render_pdf_missing_dependency_shows_sync_hint(self, tmp_path, monkeypatch):
        import builtins

        from loom.tui.screens import file_viewer

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pypdf":
                raise ImportError("No module named 'pypdf'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        widget = file_viewer._render_pdf(pdf, None)
        rendered = str(widget.render())
        assert "PDF preview unavailable" in rendered
        assert "uv sync" in rendered

    def test_file_viewer_image_metadata_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        image = workspace / "pixel.png"
        image.write_bytes(
            bytes.fromhex(
                "89504E470D0A1A0A"
                "0000000D49484452"
                "0000000100000001"
                "08060000001F15C489"
                "0000000A49444154"
                "789C6360000000020001E221BC33"
                "0000000049454E44AE426082"
            ),
        )

        screen = FileViewerScreen(image, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_file_viewer_unsupported_extension_sets_error(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        unknown = workspace / "data.foobar"
        unknown.write_text("hello", encoding="utf-8")

        screen = FileViewerScreen(unknown, workspace)

        assert screen._viewer is None
        assert screen._error is not None
        assert "No viewer renderer registered" in screen._error

    def test_file_viewer_click_outside_dismisses(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "notes.md"
        doc.write_text("# Hello\n", encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)
        screen.dismiss = MagicMock()
        dialog = MagicMock()
        dialog.region.contains.return_value = False
        screen.query_one = MagicMock(return_value=dialog)

        event = MagicMock()
        event.screen_x = 0
        event.screen_y = 0

        screen.on_mouse_down(event)

        screen.dismiss.assert_called_once_with(None)
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()

    def test_file_viewer_click_inside_does_not_dismiss(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "notes.md"
        doc.write_text("# Hello\n", encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)
        screen.dismiss = MagicMock()
        dialog = MagicMock()
        dialog.region.contains.return_value = True
        screen.query_one = MagicMock(return_value=dialog)

        event = MagicMock()
        event.screen_x = 1
        event.screen_y = 1

        screen.on_mouse_down(event)

        screen.dismiss.assert_not_called()
        event.stop.assert_not_called()
        event.prevent_default.assert_not_called()

    def test_workspace_file_selected_opens_viewer_modal(self, tmp_path):
        from textual.widgets import DirectoryTree

        from loom.tui.app import LoomApp
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "notes.md"
        doc.write_text("# Hello\n", encoding="utf-8")

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=workspace,
        )
        app.push_screen = MagicMock()
        app.notify = MagicMock()

        event = DirectoryTree.FileSelected(MagicMock(), doc)
        app.on_workspace_file_selected(event)

        app.push_screen.assert_called_once()
        screen = app.push_screen.call_args.args[0]
        assert isinstance(screen, FileViewerScreen)
        app.notify.assert_not_called()

    def test_workspace_file_selected_rejects_paths_outside_workspace(self, tmp_path):
        from textual.widgets import DirectoryTree

        from loom.tui.app import LoomApp

        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("# Outside\n", encoding="utf-8")

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=workspace,
        )
        app.push_screen = MagicMock()
        app.notify = MagicMock()

        event = DirectoryTree.FileSelected(MagicMock(), outside)
        app.on_workspace_file_selected(event)

        app.push_screen.assert_not_called()
        app.notify.assert_called_once_with(
            "Cannot open files outside the workspace.",
            severity="error",
            timeout=4,
        )
