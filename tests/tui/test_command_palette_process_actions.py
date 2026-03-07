"""TUI command palette process action tests."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Button

from loom.tui.screens.approval import ToolApprovalScreen


class TestCommandPaletteProcessActions:
    def test_ctrl_r_binding_registered(self):
        from loom.tui.app import LoomApp

        keys = {binding.key for binding in LoomApp.BINDINGS}
        assert "ctrl+r" in keys
        assert "ctrl+p" in keys
        assert "ctrl+a" in keys
        assert "ctrl+m" in keys

    def test_auth_mcp_binding_actions_registered(self):
        from loom.tui.app import LoomApp

        bindings = {binding.key: binding for binding in LoomApp.BINDINGS}
        key_to_action = {key: binding.action for key, binding in bindings.items()}
        assert key_to_action["ctrl+a"] == "open_auth_tab"
        assert key_to_action["ctrl+m"] == "open_mcp_tab"
        assert key_to_action["ctrl+p"] == "command_palette"
        assert bindings["ctrl+p"].description == "Commands"
        assert bindings["ctrl+p"].show is False
        assert bindings["ctrl+a"].priority is True
        assert bindings["ctrl+m"].priority is True
        assert bindings["ctrl+a"].show is False
        assert bindings["ctrl+m"].show is False

    @pytest.mark.asyncio
    async def test_action_command_palette_opens_custom_grouped_screen(self):
        from textual.widgets import OptionList

        from loom.tui.app import LoomApp
        from loom.tui.screens.command_palette import LoomCommandPaletteScreen

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            app.action_command_palette()
            await pilot.pause()
            assert isinstance(app.screen, LoomCommandPaletteScreen)
            assert len(list(app.screen.query("SearchIcon"))) == 0
            command_list = app.screen.query_one("#loom-command-list", OptionList)
            first = command_list.get_option_at_index(0)
            assert first.disabled is True
            assert "Suggested" in str(first.prompt)

    def test_command_palette_click_outside_dismisses(self):
        from loom.tui.screens.command_palette import LoomCommandPaletteScreen

        screen = LoomCommandPaletteScreen()
        screen.dismiss = MagicMock()
        card = MagicMock()
        card.region.contains.return_value = False
        screen.query_one = MagicMock(return_value=card)

        event = MagicMock()
        event.screen_x = 0
        event.screen_y = 0

        screen.on_mouse_down(event)

        screen.dismiss.assert_called_once_with()
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()

    def test_command_palette_click_inside_does_not_dismiss(self):
        from loom.tui.screens.command_palette import LoomCommandPaletteScreen

        screen = LoomCommandPaletteScreen()
        screen.dismiss = MagicMock()
        card = MagicMock()
        card.region.contains.return_value = True
        screen.query_one = MagicMock(return_value=card)

        event = MagicMock()
        event.screen_x = 1
        event.screen_y = 1

        screen.on_mouse_down(event)

        screen.dismiss.assert_not_called()
        event.stop.assert_not_called()
        event.prevent_default.assert_not_called()

    @pytest.mark.asyncio
    async def test_command_palette_escape_closes_palette_not_landing(self):
        from loom.tui.app import LoomApp
        from loom.tui.screens.command_palette import LoomCommandPaletteScreen

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                tui=SimpleNamespace(
                    startup_landing_enabled=True,
                    always_open_chat_directly=False,
                ),
            ),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[])
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            assert app._startup_landing_active is True
            app.action_command_palette()
            await pilot.pause()
            assert isinstance(app.screen, LoomCommandPaletteScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert app._startup_landing_active is True
            assert not isinstance(app.screen, LoomCommandPaletteScreen)

    @pytest.mark.asyncio
    async def test_command_palette_ctrl_p_closes_palette(self):
        from loom.tui.app import LoomApp
        from loom.tui.screens.command_palette import LoomCommandPaletteScreen

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            app.action_command_palette()
            await pilot.pause()
            assert isinstance(app.screen, LoomCommandPaletteScreen)
            await pilot.press("ctrl+p")
            await pilot.pause()
            assert not isinstance(app.screen, LoomCommandPaletteScreen)

    @pytest.mark.asyncio
    async def test_command_palette_ctrl_a_does_not_trigger_global_auth_shortcut(self):
        from loom.tui.app import LoomApp
        from loom.tui.screens.command_palette import LoomCommandPaletteScreen

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()
        app._open_auth_manager_screen = MagicMock()

        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause()
            app.action_command_palette()
            await pilot.pause()
            assert isinstance(app.screen, LoomCommandPaletteScreen)
            await pilot.press("ctrl+a")
            await pilot.pause()
            assert isinstance(app.screen, LoomCommandPaletteScreen)

        app._open_auth_manager_screen.assert_not_called()

    def test_footer_manager_shortcut_buttons_trigger_actions(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.action_open_auth_tab = MagicMock()
        app.action_open_mcp_tab = MagicMock()

        auth_event = SimpleNamespace(
            button=SimpleNamespace(id="footer-auth-shortcut"),
            stop=MagicMock(),
            prevent_default=MagicMock(),
        )
        app._on_footer_manager_shortcut_pressed(auth_event)
        app.action_open_auth_tab.assert_called_once()
        auth_event.stop.assert_called_once()
        auth_event.prevent_default.assert_called_once()

        mcp_event = SimpleNamespace(
            button=SimpleNamespace(id="footer-mcp-shortcut"),
            stop=MagicMock(),
            prevent_default=MagicMock(),
        )
        app._on_footer_manager_shortcut_pressed(mcp_event)
        app.action_open_mcp_tab.assert_called_once()
        mcp_event.stop.assert_called_once()
        mcp_event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_footer_manager_shortcuts_render_on_right_row(self):

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            auth_label = app.query_one("#footer-auth-shortcut", Button).label.plain.lower()
            mcp_label = app.query_one("#footer-mcp-shortcut", Button).label.plain.lower()
            assert "ctrl + a" in auth_label and "auth" in auth_label
            assert "ctrl + m" in mcp_label and "mcp" in mcp_label
            dividers = app.query("#footer-shortcuts .footer-shortcut-divider")
            assert len(list(dividers)) == 2

    @pytest.mark.asyncio
    async def test_header_activity_indicator_is_left_of_clock(self):
        from textual.widgets import Header

        from loom.tui.app import LoomApp
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            header = app.query_one("#app-header", Header)
            indicator = app.query_one("#header-activity-indicator", ActivityIndicator)
            clock = next(
                child for child in header.children
                if type(child).__name__ == "HeaderClock"
            )
            # Ensure indicator is rendered on the active header row and sits
            # directly left of the clock cluster.
            assert indicator.region.y == clock.region.y
            assert indicator.region.x + indicator.region.width <= clock.region.x

    @pytest.mark.asyncio
    async def test_header_activity_indicator_scales_with_tall_header(self):
        from textual.widgets import Header

        from loom.tui.app import LoomApp
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            header = app.query_one("#app-header", Header)
            indicator = app.query_one("#header-activity-indicator", ActivityIndicator)
            clock = next(
                child for child in header.children
                if type(child).__name__ == "HeaderClock"
            )
            header.tall = True
            await pilot.pause()
            assert indicator.region.height == clock.region.height
            assert indicator.region.y == clock.region.y
            rendered = indicator.render()
            assert rendered.count("\n") == max(0, indicator.region.height - 1)

    def test_background_work_active_predicate(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        app._chat_busy = False
        app._process_runs = {}
        app._active_delegate_streams = {}
        assert app._is_background_work_active() is False

        app._chat_busy = True
        assert app._is_background_work_active() is True

        app._chat_busy = False
        app._process_runs = {"run_1": SimpleNamespace(status="running")}
        assert app._is_background_work_active() is True

        app._process_runs = {}
        app._active_delegate_streams = {"call_1": {"finalized": False}}
        assert app._is_background_work_active() is True

        app._active_delegate_streams = {"call_1": {"finalized": True}}
        assert app._is_background_work_active() is False

    def test_open_auth_tab_action_opens_manager(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._open_auth_manager_screen = MagicMock()

        app.action_open_auth_tab()

        app._open_auth_manager_screen.assert_called_once()

    def test_open_mcp_tab_action_opens_manager(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._open_mcp_manager_screen = MagicMock()

        app.action_open_mcp_tab()

        app._open_mcp_manager_screen.assert_called_once()

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
        app.notify.assert_called_once_with(
            "Workspace reloaded.",
            severity="information",
            timeout=2,
        )

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

        chat.add_info.assert_called_once()
        assert "Process Modes" in chat.add_info.call_args.args[0]

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
    async def test_models_info_action(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._render_models_catalog = MagicMock(return_value="models-catalog")
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        await app.action_loom_command("models_info")

        app._render_models_catalog.assert_called_once()
        chat.add_info.assert_called_once_with("models-catalog")

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
    async def test_palette_stop_chat_action_triggers_stop(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.action_stop_chat = MagicMock()

        await app.action_loom_command("stop_chat")

        app.action_stop_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_palette_close_tab_action_uses_unified_close_flow(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.action_close_process_tab = MagicMock()
        app._close_process_run_from_target = AsyncMock(return_value=True)

        await app.action_loom_command("close_process_tab")

        app.action_close_process_tab.assert_called_once()
        app._close_process_run_from_target.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_palette_prompt_actions_prefill_input(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._prefill_user_input = MagicMock()

        await app.action_loom_command("run_prompt")
        await app.action_loom_command("resume_prompt")

        calls = [c.args[0] for c in app._prefill_user_input.call_args_list]
        assert calls == ["/run ", "/resume "]

    @pytest.mark.asyncio
    async def test_palette_steering_actions_dispatch(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._handle_slash_command = AsyncMock(return_value=True)
        app._prefill_user_input = MagicMock()

        await app.action_loom_command("pause_chat")
        await app.action_loom_command("resume_chat")
        await app.action_loom_command("inject_prompt")
        await app.action_loom_command("redirect_prompt")
        await app.action_loom_command("steer_queue")
        await app.action_loom_command("steer_clear")

        slash_calls = [c.args[0] for c in app._handle_slash_command.await_args_list]
        assert slash_calls == [
            "/pause",
            "/steer resume",
            "/steer queue",
            "/steer clear",
        ]
        prefill_calls = [c.args[0] for c in app._prefill_user_input.call_args_list]
        assert prefill_calls == ["/inject ", "/redirect "]

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

        app._prefill_user_input("/processes")

        assert input_widget.value == "/processes"
        assert input_widget.cursor_position == len("/processes")
        input_widget.focus.assert_called_once()
        app._render_slash_hint.assert_called_once_with("/processes")
        app._set_slash_hint.assert_called_once_with("hint text")

    def test_hydrate_input_history_from_session_filters_non_user_messages(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "  /setup  "},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "   "},
            {"role": "user", "content": "final request"},
        ])

        app._hydrate_input_history_from_session()

        assert app._input_history == ["/setup", "final request"]

    def test_hydrate_input_history_prefers_ui_state_payload(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(
            session_state=SimpleNamespace(
                ui_state={
                    "input_history": {
                        "version": 1,
                        "items": ["/clear", "plan next steps"],
                    },
                },
            ),
            messages=[
                {"role": "user", "content": "this should not win"},
            ],
        )

        app._hydrate_input_history_from_session()

        assert app._input_history == ["/clear", "plan next steps"]

    @pytest.mark.asyncio
    async def test_up_down_keys_navigate_input_history_and_restore_draft(self):
        from textual.widgets import Input

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()
        app._input_history = ["first command", "second command"]

        async with app.run_test() as pilot:
            input_widget = app.query_one("#user-input", Input)
            input_widget.focus()
            input_widget.value = "draft message"
            input_widget.cursor_position = len("draft message")
            await pilot.pause()

            await pilot.press("up")
            await pilot.pause()
            assert input_widget.value == "second command"

            await pilot.press("up")
            await pilot.pause()
            assert input_widget.value == "first command"

            await pilot.press("down")
            await pilot.pause()
            assert input_widget.value == "second command"

            await pilot.press("down")
            await pilot.pause()
            assert input_widget.value == "draft message"

    @pytest.mark.asyncio
    async def test_user_submit_records_history_for_slash_commands(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.query_one = MagicMock(return_value=SimpleNamespace(value=""))
        app._handle_slash_command = AsyncMock(return_value=True)
        app._persist_process_run_ui_state = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._run_turn = MagicMock()

        event = SimpleNamespace(value="/help")
        await app.on_user_submit(event)

        assert app._input_history == ["/help"]
        app._run_turn.assert_not_called()
        app._persist_process_run_ui_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_user_submit_records_history_for_messages(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.query_one = MagicMock(return_value=SimpleNamespace(value=""))
        app._handle_slash_command = AsyncMock(return_value=False)
        app._set_slash_hint = MagicMock()
        worker = object()
        app._run_turn = MagicMock(return_value=worker)

        event = SimpleNamespace(value="hello loom")
        await app.on_user_submit(event)

        assert app._input_history == ["hello loom"]
        app._run_turn.assert_called_once_with("hello loom")
        assert app._chat_turn_worker is worker

    @pytest.mark.asyncio
    async def test_user_submit_busy_enter_defaults_to_inject(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.query_one = MagicMock(return_value=SimpleNamespace(value=""))
        app._handle_slash_command = AsyncMock(return_value=False)
        app._is_cowork_stop_visible = MagicMock(return_value=True)
        app._queue_chat_inject_instruction = AsyncMock(return_value=True)
        app._persist_process_run_ui_state = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._run_turn = MagicMock()

        event = SimpleNamespace(value="inject this")
        await app.on_user_submit(event)

        app._queue_chat_inject_instruction.assert_awaited_once_with(
            "inject this",
            source="enter",
        )
        app._run_turn.assert_not_called()
        assert app._input_history == ["inject this"]
        app._persist_process_run_ui_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_user_submit_busy_queue_failure_restores_input(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.query_one = MagicMock(return_value=SimpleNamespace(value=""))
        app._handle_slash_command = AsyncMock(return_value=False)
        app._is_cowork_stop_visible = MagicMock(side_effect=[True, True])
        app._queue_chat_inject_instruction = AsyncMock(return_value=False)
        app._persist_process_run_ui_state = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._set_user_input_text = MagicMock()
        app._run_turn = MagicMock()

        event = SimpleNamespace(value="keep me")
        await app.on_user_submit(event)

        app._queue_chat_inject_instruction.assert_awaited_once_with(
            "keep me",
            source="enter",
        )
        app._set_user_input_text.assert_called_once_with("keep me")
        app._run_turn.assert_not_called()
        app._persist_process_run_ui_state.assert_not_awaited()
        assert app._input_history == []

    @pytest.mark.asyncio
    async def test_user_submit_busy_queue_failure_falls_back_to_turn_if_idle(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.query_one = MagicMock(return_value=SimpleNamespace(value=""))
        app._handle_slash_command = AsyncMock(return_value=False)
        app._is_cowork_stop_visible = MagicMock(side_effect=[True, False])
        app._queue_chat_inject_instruction = AsyncMock(return_value=False)
        app._persist_process_run_ui_state = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._set_user_input_text = MagicMock()
        worker = object()
        app._run_turn = MagicMock(return_value=worker)

        event = SimpleNamespace(value="fallback send")
        await app.on_user_submit(event)

        app._queue_chat_inject_instruction.assert_awaited_once_with(
            "fallback send",
            source="enter",
        )
        app._set_user_input_text.assert_not_called()
        app._run_turn.assert_called_once_with("fallback send")
        assert app._chat_turn_worker is worker
        assert app._input_history == ["fallback send"]

    @pytest.mark.asyncio
    async def test_user_submit_focused_process_run_routes_to_run_inject(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(run_id="abc123", status="running")
        app.query_one = MagicMock(return_value=SimpleNamespace(value=""))
        app._handle_slash_command = AsyncMock(return_value=False)
        app._current_process_run = MagicMock(return_value=run)
        app._inject_process_run = AsyncMock(return_value=True)
        app._persist_process_run_ui_state = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._run_turn = MagicMock()

        event = SimpleNamespace(value="steer this run")
        await app.on_user_submit(event)

        app._inject_process_run.assert_awaited_once_with(
            run,
            "steer this run",
            source="enter",
            queue_if_unavailable=True,
        )
        app._run_turn.assert_not_called()
        assert app._input_history == ["steer this run"]
        app._persist_process_run_ui_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_user_submit_focused_process_run_inject_failure_restores_input(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(run_id="abc123", status="running")
        app.query_one = MagicMock(return_value=SimpleNamespace(value=""))
        app._handle_slash_command = AsyncMock(return_value=False)
        app._current_process_run = MagicMock(return_value=run)
        app._inject_process_run = AsyncMock(return_value=False)
        app._set_user_input_text = MagicMock()
        app._persist_process_run_ui_state = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._run_turn = MagicMock()

        event = SimpleNamespace(value="keep this text")
        await app.on_user_submit(event)

        app._inject_process_run.assert_awaited_once_with(
            run,
            "keep this text",
            source="enter",
            queue_if_unavailable=True,
        )
        app._set_user_input_text.assert_called_once_with("keep this text")
        app._run_turn.assert_not_called()
        app._persist_process_run_ui_state.assert_not_awaited()
        assert app._input_history == []

    @pytest.mark.asyncio
    async def test_request_chat_stop_clears_state_when_only_delegate_streams_active(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        status = SimpleNamespace(state="Ready")

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        app._chat_busy = False
        app._active_delegate_streams["call_1"] = {"finalized": False}
        app._session = SimpleNamespace(
            request_stop=MagicMock(),
            clear_stop_request=MagicMock(),
            session_id="session-1",
            _turn_counter=1,
        )
        app._sync_activity_indicator = MagicMock()
        app._handle_interrupted_chat_turn = AsyncMock()

        await app._request_chat_stop()

        app._handle_interrupted_chat_turn.assert_awaited_once_with(
            path="cooperative",
            reason="user_requested",
            stage="delegate_stream_cleanup",
        )
        assert app._chat_stop_requested is False
        assert status.state == "Ready"
        app._session.request_stop.assert_called_once_with("user_requested")
        assert app._session.clear_stop_request.call_count >= 1

    def test_sync_chat_stop_control_visibility_and_disabled_state(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        stop_btn = SimpleNamespace(display=False, disabled=False, label="Stop")
        inject_btn = SimpleNamespace(display=False, disabled=False)
        redirect_btn = SimpleNamespace(display=False, disabled=False)
        input_widget = SimpleNamespace(value="queued steer")

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-stop-btn":
                return stop_btn
            if selector == "#chat-inject-btn":
                return inject_btn
            if selector == "#chat-redirect-btn":
                return redirect_btn
            if selector == "#user-input":
                return input_widget
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        app._chat_busy = True
        app._chat_stop_requested = False
        app._sync_chat_stop_control()
        assert stop_btn.display is True
        assert stop_btn.disabled is False
        assert stop_btn.label == "■"
        assert inject_btn.display is True
        assert redirect_btn.display is True

        input_widget.value = "   "
        app._sync_chat_stop_control()
        assert stop_btn.display is True
        assert inject_btn.display is False
        assert redirect_btn.display is False

        app._chat_stop_requested = True
        input_widget.value = "queued steer"
        app._sync_chat_stop_control()
        assert stop_btn.display is True
        assert stop_btn.disabled is True
        assert stop_btn.label == "■"

        app._chat_busy = False
        app._chat_stop_requested = False
        app._sync_chat_stop_control()
        assert stop_btn.display is False
        assert inject_btn.display is False
        assert redirect_btn.display is False

    def test_set_slash_hint_prefers_steering_queue_popup_when_pending(self):
        from loom.tui.app import LoomApp, SteeringDirective

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._chat_busy = True
        app._pending_inject_directives = [
            SteeringDirective(
                id="d1",
                kind="inject",
                text="focus on tests first",
                source="slash",
            ),
        ]
        fake_hint = SimpleNamespace(
            display=False,
            scroll_home=MagicMock(),
            styles=SimpleNamespace(height=None, max_height=None, overflow_y=None),
        )
        fake_hint_body = SimpleNamespace(update=MagicMock(), display=True)
        fake_list = SimpleNamespace(display=False)
        fake_grid = SimpleNamespace(
            display=False,
            styles=SimpleNamespace(background=None),
        )

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#slash-hint":
                return fake_hint
            if selector == "#slash-hint-body":
                return fake_hint_body
            if selector == "#steer-queue-grid":
                return fake_grid
            if selector == "#steer-queue-list":
                return fake_list
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        app._render_steer_queue_rows = MagicMock()

        app._set_slash_hint("slash hint")

        assert fake_hint.display is True
        assert fake_list.display is True
        assert fake_grid.display is True
        assert fake_hint.styles.height == 3
        assert fake_hint.styles.max_height == 3
        assert fake_hint.styles.overflow_y == "hidden"
        assert fake_grid.styles.background == "#3a4465"
        app._render_steer_queue_rows.assert_called_once()
        fake_hint_body.update.assert_called_once_with("")
        assert fake_hint_body.display is False

    def test_set_slash_hint_queue_popup_skips_rerender_when_unchanged(self):
        from loom.tui.app import LoomApp, SteeringDirective

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._chat_busy = True
        app._pending_inject_directives = [
            SteeringDirective(
                id="d1",
                kind="inject",
                text="focus on tests first",
                source="slash",
            ),
        ]
        fake_hint = SimpleNamespace(
            display=False,
            scroll_home=MagicMock(),
            styles=SimpleNamespace(height=None, max_height=None, overflow_y=None),
        )
        fake_hint_body = SimpleNamespace(update=MagicMock(), display=True)
        fake_list = SimpleNamespace(display=False)
        fake_grid = SimpleNamespace(
            display=False,
            styles=SimpleNamespace(background=None),
        )

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#slash-hint":
                return fake_hint
            if selector == "#slash-hint-body":
                return fake_hint_body
            if selector == "#steer-queue-grid":
                return fake_grid
            if selector == "#steer-queue-list":
                return fake_list
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        def _mark_rows_rendered() -> None:
            app._last_rendered_steer_queue_signature = app._steer_queue_signature()

        app._render_steer_queue_rows = MagicMock(side_effect=_mark_rows_rendered)

        app._set_slash_hint("slash hint")
        app._set_slash_hint("slash hint")

        app._render_steer_queue_rows.assert_called_once()

    @pytest.mark.asyncio
    async def test_steer_queue_popup_renders_three_buttons_per_item(self):
        from loom.tui.app import LoomApp, SteeringDirective

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()
        app._chat_busy = True
        app._pending_inject_directives = [
            SteeringDirective(id="d1", kind="inject", text="first", source="slash"),
            SteeringDirective(id="d2", kind="inject", text="second", source="slash"),
        ]

        async with app.run_test() as pilot:
            app._set_slash_hint("")
            await pilot.pause()

            button_ids = {
                str(widget.id)
                for widget in app.query(".steer-queue-item-btn")
            }

        assert button_ids == {
            "steer-queue-edit-d1",
            "steer-queue-redirect-d1",
            "steer-queue-dismiss-d1",
            "steer-queue-edit-d2",
            "steer-queue-redirect-d2",
            "steer-queue-dismiss-d2",
        }

    @pytest.mark.asyncio
    async def test_request_chat_redirect_requires_confirm_only_for_mutating_tool(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(
            pause_requested=False,
            request_resume=MagicMock(),
            clear_pending_inject_instruction=MagicMock(),
            _turn_counter=2,
        )
        app._record_steering_event = AsyncMock()
        app._request_chat_stop = AsyncMock()
        app._is_cowork_stop_visible = MagicMock(return_value=False)
        app._run_turn = MagicMock(return_value=object())
        app._sync_activity_indicator = MagicMock()
        app._confirm_redirect_with_mutating_tool = AsyncMock(return_value=True)

        app._cowork_inflight_tool_counts = {"read_file": 1}
        app._is_mutating_tool = MagicMock(side_effect=lambda name: name == "write_file")

        ok = await app._request_chat_redirect("new objective", source="slash")

        assert ok is True
        app._confirm_redirect_with_mutating_tool.assert_not_awaited()

        app._cowork_inflight_tool_counts = {"write_file": 1}
        app._confirm_redirect_with_mutating_tool = AsyncMock(return_value=False)
        app._run_turn.reset_mock()
        app._chat_turn_worker = None

        ok = await app._request_chat_redirect("another objective", source="slash")

        assert ok is False
        app._confirm_redirect_with_mutating_tool.assert_awaited_once_with("write_file")
        app._run_turn.assert_not_called()

    def test_action_steer_queue_edit_restores_input(self):
        from loom.tui.app import LoomApp, SteeringDirective

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._pending_inject_directives = [SteeringDirective(
            id="d1",
            kind="inject",
            text="restore me",
            source="slash",
        )]
        app._session = SimpleNamespace(clear_pending_inject_instruction=MagicMock())
        app._set_user_input_text = MagicMock()
        app._sync_chat_stop_control = MagicMock()
        app._refresh_hint_panel = MagicMock()
        app.query_one = MagicMock(return_value=SimpleNamespace(focus=MagicMock()))
        captured: dict[str, object] = {}
        app.run_worker = MagicMock(
            side_effect=lambda coro, **kwargs: captured.update(coro=coro, kwargs=kwargs)
        )

        app.action_steer_queue_edit()

        app._set_user_input_text.assert_called_once_with("restore me")
        assert app._pending_inject_directives == []
        app._session.clear_pending_inject_instruction.assert_called_once()
        coro = captured.get("coro")
        if coro is not None:
            coro.close()

    def test_action_steer_queue_dismiss_clears_pending(self):
        from loom.tui.app import LoomApp, SteeringDirective

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._pending_inject_directives = [SteeringDirective(
            id="d1",
            kind="inject",
            text="dismiss me",
            source="slash",
        )]
        app._session = SimpleNamespace(clear_pending_inject_instruction=MagicMock())
        app._sync_chat_stop_control = MagicMock()
        app._refresh_hint_panel = MagicMock()
        captured: dict[str, object] = {}
        app.run_worker = MagicMock(
            side_effect=lambda coro, **kwargs: captured.update(coro=coro, kwargs=kwargs)
        )

        app.action_steer_queue_dismiss()

        assert app._pending_inject_directives == []
        app._session.clear_pending_inject_instruction.assert_called_once()
        coro = captured.get("coro")
        if coro is not None:
            coro.close()

    @pytest.mark.asyncio
    async def test_action_steer_queue_redirect_uses_queued_text(self):
        from loom.tui.app import LoomApp, SteeringDirective

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._pending_inject_directives = [SteeringDirective(
            id="d1",
            kind="inject",
            text="switch objective",
            source="slash",
        )]
        app._session = SimpleNamespace(clear_pending_inject_instruction=MagicMock())
        app._sync_chat_stop_control = MagicMock()
        app._refresh_hint_panel = MagicMock()
        app._request_chat_redirect = AsyncMock(return_value=True)
        app._record_steering_event = AsyncMock()
        captured: dict[str, object] = {}
        app.run_worker = MagicMock(
            side_effect=lambda coro, **kwargs: captured.update(coro=coro, kwargs=kwargs)
        )

        app.action_steer_queue_redirect()
        await captured["coro"]

        app._request_chat_redirect.assert_awaited_once_with(
            "switch objective",
            source="queue_popup",
        )
        assert app._pending_inject_directives == []
        app._session.clear_pending_inject_instruction.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_chat_stop_when_idle_is_safe_noop(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        status = SimpleNamespace(state="Ready")

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        app._chat_busy = False
        app._session = None

        await app._request_chat_stop()

        chat.add_info.assert_called_once_with("No active cowork chat execution to stop.")

    def test_chat_stop_button_invokes_stop_action(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.action_stop_chat = MagicMock()
        event = MagicMock()

        app._on_chat_stop_pressed(event)

        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()
        app.action_stop_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_stop_chat_reports_failure_and_resets_state(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        status = SimpleNamespace(state="Stopping...")

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        app._request_chat_stop = AsyncMock(side_effect=RuntimeError("boom"))
        app._session = SimpleNamespace(
            clear_stop_request=MagicMock(),
            session_id="session-1",
        )
        app._chat_stop_requested = True
        app._chat_stop_requested_at = time.monotonic()
        app._sync_activity_indicator = MagicMock()
        captured: dict[str, object] = {}
        app.run_worker = MagicMock(
            side_effect=lambda coro, **kwargs: captured.update(coro=coro, kwargs=kwargs)
        )

        app.action_stop_chat()
        assert app._chat_stop_inflight is True

        await captured["coro"]

        assert app._chat_stop_inflight is False
        assert app._chat_stop_requested is False
        app._session.clear_stop_request.assert_called_once()
        assert status.state == "Ready"
        chat.add_info.assert_called_once()
        assert "Stop failed" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_run_turn_resets_state_after_cooperative_stop(self):
        from loom.cowork.session import CoworkStopRequestedError
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        status = SimpleNamespace(state="Idle")

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        app._session = SimpleNamespace(
            clear_stop_request=MagicMock(),
            stop_requested=False,
        )
        app._append_chat_replay_event = AsyncMock()
        app._run_interaction = AsyncMock(side_effect=CoworkStopRequestedError(
            reason="user_requested",
            stage="model_response",
            path="cooperative",
        ))
        app._handle_interrupted_chat_turn = AsyncMock()
        app._sync_activity_indicator = MagicMock()
        app._chat_busy = False
        app._chat_turn_worker = object()

        await LoomApp._run_turn.__wrapped__(app, "hello")

        app._handle_interrupted_chat_turn.assert_awaited_once_with(
            path="cooperative",
            reason="user_requested",
            stage="model_response",
        )
        assert app._chat_busy is False
        assert app._chat_stop_requested is False
        assert app._chat_turn_worker is None
        assert status.state == "Ready"
        assert app._session.clear_stop_request.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_turn_dispatches_enter_queued_followup_after_completion(self):
        from loom.tui.app import LoomApp, SteeringDirective

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        status = SimpleNamespace(state="Idle", total_tokens=0)

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)
        app._session = SimpleNamespace(
            clear_stop_request=MagicMock(),
            stop_requested=False,
            pending_inject_instruction_count=1,
            has_pending_inject_instruction=True,
            clear_pending_inject_instruction=MagicMock(),
            queue_inject_instruction=MagicMock(),
            _turn_counter=1,
        )
        app._pending_inject_directives = [
            SteeringDirective(
                id="d1",
                kind="inject",
                text="Write me another.",
                source="enter",
            ),
        ]
        app._append_chat_replay_event = AsyncMock()
        app._record_steering_event = AsyncMock()
        app._run_interaction = AsyncMock()
        app._sync_activity_indicator = MagicMock()
        app._start_queued_followup_turn = MagicMock()
        app.call_after_refresh = MagicMock(side_effect=lambda callback: callback())

        await LoomApp._run_turn.__wrapped__(app, "write me a story")

        assert app._pending_inject_directives == []
        app._session.clear_pending_inject_instruction.assert_called_once()
        app._start_queued_followup_turn.assert_called_once_with("Write me another.")
        assert app._record_steering_event.await_count >= 1
        assert app._record_steering_event.await_args_list[-1].args[0] == "steer_inject_applied"

    @pytest.mark.asyncio
    async def test_chat_stop_button_renders_when_visible(self):

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(120, 40)) as pilot:
            app._chat_busy = True
            app._chat_stop_requested = False
            app._sync_chat_stop_control()
            await pilot.pause()

            stop_btn = app.query_one("#chat-stop-btn", Button)
            assert stop_btn.display is True
            assert str(stop_btn.label) == "■"

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
        assert input_widget.value == "/sessions"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/session"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/stop"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/steer"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/setup"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/sessions"

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
        assert input_widget.value == "/setup"

    def test_slash_tab_completion_cycles_tool_names_for_tool_command(self):
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = ["read_file", "redirect_path", "write_file"]
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )
        input_widget = SimpleNamespace(value="/tool r", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/tool read_file"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/tool redirect_path"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/tool read_file"

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

        input_widget.value = '/tool read_file {"path":"README.md"}'
        assert app._apply_slash_tab_completion(reverse=False) is False

    def test_slash_tab_completion_process_use_prefix_not_supported(self):
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

        assert app._apply_slash_tab_completion(reverse=False) is False

    def test_slash_tab_completion_process_use_cycles_not_supported(self):
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

        assert app._apply_slash_tab_completion(reverse=False) is False

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
            assert input_widget.value == "/sessions"

            await pilot.press("tab")
            await pilot.pause()
            assert input_widget.value == "/session"

    @pytest.mark.asyncio
    async def test_tab_key_is_captured_for_slash_autocomplete_on_landing_input(self):
        from textual.widgets import Input

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                tui=SimpleNamespace(
                    startup_landing_enabled=True,
                    always_open_chat_directly=False,
                ),
            ),
        )
        app._store = MagicMock()
        app._store.list_sessions = AsyncMock(return_value=[])
        app._initialize_session = AsyncMock()

        async with app.run_test() as pilot:
            await pilot.pause()
            landing_input = app.query_one("#landing-input", Input)
            assert app._startup_landing_active is True
            assert landing_input.value == ""

            await pilot.press("/")
            await pilot.press("s")
            await pilot.pause()
            assert landing_input.value == "/s"

            await pilot.press("tab")
            await pilot.pause()
            assert landing_input.value == "/sessions"

            await pilot.press("tab")
            await pilot.pause()
            assert landing_input.value == "/session"

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

    def test_ctrl_w_non_chat_input_focus_is_not_intercepted(self):
        from textual.widgets import Input

        from loom.tui.app import LoomApp

        class _FocusedHarness(LoomApp):
            @property
            def focused(self):
                return self._focused_widget

        app = _FocusedHarness(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._focused_widget = Input(id="mcp-alias")
        event = MagicMock()
        event.key = "ctrl+w"

        app.on_key(event)

        event.stop.assert_not_called()
        event.prevent_default.assert_not_called()

    def test_on_key_dismisses_active_tool_approval_screen(self):
        from loom.tui.app import LoomApp

        class _ScreenHarness(LoomApp):
            @property
            def screen(self):  # pragma: no cover - property shim
                return self._screen_for_test

        app = _ScreenHarness(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        screen = ToolApprovalScreen("write_file", "test.md")
        screen.dismiss = MagicMock()
        app._screen_for_test = screen
        event = MagicMock()
        event.key = "y"

        app.on_key(event)

        screen.dismiss.assert_called_once_with("approve")
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_close_process_tab_starts_nonexclusive_worker(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._close_process_run_from_target = AsyncMock(return_value=True)
        app._process_runs = {"abc": MagicMock()}
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
        assert "current" in app._close_process_tab_inflight

        await captured["coro"]
        app._close_process_run_from_target.assert_awaited_once_with("current")
        assert "current" not in app._close_process_tab_inflight

    @pytest.mark.asyncio
    async def test_action_close_process_tab_closes_active_manager_tab(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._close_process_run_from_target = AsyncMock(return_value=True)
        app._remove_tab_if_present = AsyncMock(return_value=None)
        app.query_one = MagicMock(return_value=SimpleNamespace(active=app._MCP_MANAGER_TAB_ID))
        captured: dict = {}

        def _capture_worker(coro, **kwargs):
            captured["coro"] = coro
            captured["kwargs"] = kwargs
            return MagicMock()

        app.run_worker = MagicMock(side_effect=_capture_worker)

        app.action_close_process_tab()

        assert app.run_worker.call_count == 1
        assert captured["kwargs"]["group"] == "close-process-tab"
        await captured["coro"]
        app._remove_tab_if_present.assert_awaited_once_with(app._MCP_MANAGER_TAB_ID)
        app._close_process_run_from_target.assert_not_awaited()
        assert app._MCP_MANAGER_TAB_ID not in app._close_process_tab_inflight

    @pytest.mark.asyncio
    async def test_action_close_process_tab_requests_manager_guarded_close_when_supported(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._close_process_run_from_target = AsyncMock(return_value=True)
        app._remove_tab_if_present = AsyncMock(return_value=None)
        tabs = SimpleNamespace(active=app._MCP_MANAGER_TAB_ID)
        manager_widget = SimpleNamespace(action_request_close=AsyncMock(return_value=None))
        captured: dict = {}

        def _query_one(selector, *_args):
            if selector == "#tabs":
                return tabs
            if selector == f"#{app._MCP_MANAGER_TAB_ID} MCPManagerScreen":
                return manager_widget
            return SimpleNamespace()

        def _capture_worker(coro, **kwargs):
            captured["coro"] = coro
            captured["kwargs"] = kwargs
            return MagicMock()

        app.query_one = MagicMock(side_effect=_query_one)
        app.run_worker = MagicMock(side_effect=_capture_worker)

        app.action_close_process_tab()

        assert app.run_worker.call_count == 1
        assert captured["kwargs"]["group"] == "close-process-tab"
        await captured["coro"]
        manager_widget.action_request_close.assert_awaited_once()
        app._remove_tab_if_present.assert_not_awaited()
        app._close_process_run_from_target.assert_not_awaited()
        assert app._MCP_MANAGER_TAB_ID not in app._close_process_tab_inflight

    @pytest.mark.asyncio
    async def test_action_close_process_tab_with_no_closable_tabs_shows_tab_scoped_message(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._close_process_run_from_target = AsyncMock(return_value=True)
        chat = MagicMock()
        tabs = SimpleNamespace(active="tab-chat")
        captured: dict = {}

        def _query_one(selector, *_args):
            if selector == "#tabs":
                return tabs
            if selector == "#chat-log":
                return chat
            return MagicMock()

        def _capture_worker(coro, **kwargs):
            captured["coro"] = coro
            captured["kwargs"] = kwargs
            return MagicMock()

        app.query_one = MagicMock(side_effect=_query_one)
        app.run_worker = MagicMock(side_effect=_capture_worker)

        app.action_close_process_tab()

        assert app.run_worker.call_count == 1
        await captured["coro"]
        app._close_process_run_from_target.assert_not_awaited()
        chat.add_info.assert_called_once()
        assert "No closable tabs are open" in chat.add_info.call_args.args[0]

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
        assert "current" in app._close_process_tab_inflight
        # Close pending coroutine to avoid unawaited coroutine warnings.
        first_coro["coro"].close()
