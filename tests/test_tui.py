"""Tests for the TUI app and its components."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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
        ]
        rendered = panel.render()
        assert "Read file" in rendered
        assert "Fix bug" in rendered
        assert "Run tests" in rendered


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
        from loom.tools.delegate_task import DelegateTaskTool
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
        monkeypatch.setattr("loom.tools.create_default_registry", lambda: MagicMock())

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(
            session_id="session-id",
            session_state=SimpleNamespace(),
        )
        app._delegate_tool = DelegateTaskTool()
        app._config = SimpleNamespace(
            workspace=SimpleNamespace(scratch_dir="/tmp"),
        )
        app._db = MagicMock()
        app._process_defn = SimpleNamespace(
            name="marketing-strategy",
            tools=SimpleNamespace(excluded=[]),
        )

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
        assert "/setup" in hint

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
        assert fake_footer.display is False
        assert fake_status.display is False

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
        assert fake_footer.display is True
        assert fake_status.display is True

    def test_slash_completion_candidates_prefix(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        assert app._slash_completion_candidates("/s") == [
            "/setup",
            "/session",
            "/sessions",
        ]
        assert app._slash_completion_candidates("/h") == ["/help"]
        assert app._slash_completion_candidates("/t") == ["/tools", "/tokens"]

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
        assert "/quit (/exit, /q)" in rendered
        assert "/setup" in rendered

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
        chat.add_info.assert_called_once_with("Usage: /resume <session-id-prefix>")
