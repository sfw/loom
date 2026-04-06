"""TUI process slash command tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
import yaml

from loom.tui.screens.approval import ToolApprovalScreen


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
        assert "/models" in hint
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

    def test_root_slash_uses_priority_ordering(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/")
        assert hint.index("/new") < hint.index("/run")
        assert hint.index("/run") < hint.index("/mcp")
        assert hint.index("/model") < hint.index("/models")
        assert hint.index("/mcp") < hint.index("/help")

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

    def test_prefix_p_matches_processes(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/p")
        assert "Matching /p:" in hint
        assert "/processes" in hint

    def test_prefix_process_matches_processes(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/process")
        assert "Matching /process:" in hint
        assert "/processes" in hint

    def test_process_use_hint_is_not_rendered(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/process use")
        assert hint == ""

    def test_process_use_hint_with_prefix_is_not_rendered(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        hint = app._render_slash_hint("/process use inv")
        assert hint == ""

    def test_process_use_hint_large_match_sets_not_rendered(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        hint = app._render_slash_hint("/process use process-")

        assert hint == ""

    def test_tool_hint_lists_matching_tool_names(self):
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = ["read_file", "write_file", "web_search"]
        tools.get.return_value = None
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )

        hint = app._render_slash_hint("/tool w")

        assert "Matching tools for w:" in hint
        assert "write_file" in hint
        assert "web_search" in hint
        assert "read_file" not in hint

    def test_tool_hint_root_lists_all_tools_without_pagination(self):
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = [f"tool_{i:02d}" for i in range(12)]
        tools.get.return_value = None
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )

        hint = app._render_slash_hint("/tool")

        for i in range(12):
            assert f"tool_{i:02d}" in hint
        assert "more" not in hint
        assert "\\[key=value ... | json-object-args\\]" in hint

    def test_tool_hint_shows_argument_details_for_exact_match(self):
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = ["read_file"]
        tools.get.return_value = SimpleNamespace(
            description="Read a file from workspace",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "encoding": {"type": "string"},
                },
                "required": ["path"],
            },
        )
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )

        hint = app._render_slash_hint("/tool read_file")

        assert "Tool: read_file" in hint
        assert "Required:" in hint
        assert "path" in hint
        assert "Optional:" in hint
        assert "encoding" in hint
        assert "Example:" in hint

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

    def test_landing_input_changed_uses_live_widget_value(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        landing_input = SimpleNamespace(value="/h")
        app.query_one = MagicMock(return_value=landing_input)

        captured = []
        app._set_slash_hint = lambda text: captured.append(text)

        stale_event = SimpleNamespace(value="/")
        app.on_landing_input_changed(stale_event)

        assert captured

    @pytest.mark.asyncio
    async def test_tool_hint_markup_renders_without_error(self):
        from textual.widgets import Static

        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = [f"tool_{i}" for i in range(20)]
        tools.get.return_value = None
        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=tools,
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test() as pilot:
            hint_text = app._render_slash_hint("/tool")
            app._set_slash_hint(hint_text)
            await pilot.pause()
            hint_body = app.query_one("#slash-hint-body", Static)
            assert hint_body.display is True

    @pytest.mark.asyncio
    async def test_slash_hint_container_supports_overflow_scrolling(self):
        from textual.containers import VerticalScroll

        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(100, 14)) as pilot:
            await pilot.press("/")
            await pilot.pause()
            hint = app.query_one("#slash-hint", VerticalScroll)
            assert hint.display is True
            assert hint.max_scroll_y > 0

    def test_set_slash_hint_updates_scroll_container(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        fake_hint = SimpleNamespace(
            display=False,
            scroll_home=MagicMock(),
        )
        fake_hint_body = SimpleNamespace(update=MagicMock())
        fake_footer = SimpleNamespace(display=True)
        fake_status = SimpleNamespace(display=True)
        fake_list = SimpleNamespace(display=True)
        fake_grid = SimpleNamespace(display=False)

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#slash-hint":
                return fake_hint
            if selector == "#slash-hint-body":
                return fake_hint_body
            if selector == "#steer-queue-grid":
                return fake_grid
            if selector == "#steer-queue-list":
                return fake_list
            if selector == "#status-bar":
                return fake_status
            return fake_footer

        app.query_one = MagicMock(side_effect=_query_one)

        app._set_slash_hint("a\nb\nc")

        assert fake_hint.display is True
        fake_hint_body.update.assert_called_once_with("a\nb\nc")
        fake_hint.scroll_home.assert_called_once_with(animate=False)
        # Slash hints no longer toggle footer/status visibility.
        assert fake_footer.display is True
        assert fake_status.display is True
        assert fake_list.display is False
        assert fake_grid.display is True

    def test_set_slash_hint_empty_hides_container_and_clears_body(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        fake_hint = SimpleNamespace(
            display=True,
            scroll_home=MagicMock(),
        )
        fake_hint_body = SimpleNamespace(update=MagicMock())
        fake_footer = SimpleNamespace(display=False)
        fake_status = SimpleNamespace(display=False)
        fake_list = SimpleNamespace(display=True)
        fake_grid = SimpleNamespace(display=True)

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#slash-hint":
                return fake_hint
            if selector == "#slash-hint-body":
                return fake_hint_body
            if selector == "#steer-queue-grid":
                return fake_grid
            if selector == "#steer-queue-list":
                return fake_list
            if selector == "#status-bar":
                return fake_status
            return fake_footer

        app.query_one = MagicMock(side_effect=_query_one)

        app._set_slash_hint("")

        assert fake_hint.display is False
        fake_hint_body.update.assert_called_once_with("")
        fake_hint.scroll_home.assert_called_once_with(animate=False)
        # Slash hints no longer toggle footer/status visibility.
        assert fake_footer.display is False
        assert fake_status.display is False
        assert fake_list.display is False
        assert fake_grid.display is False

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
            "/sessions",
            "/session",
            "/stop",
            "/steer",
            "/setup",
        ]
        assert app._slash_completion_candidates("/h") == ["/history", "/help"]
        assert app._slash_completion_candidates("/m") == ["/mcp", "/model", "/models"]
        assert app._slash_completion_candidates("/t") == [
            "/tools",
            "/tool",
            "/tokens",
            "/telemetry",
        ]
        assert app._slash_completion_candidates("/p") == ["/pause", "/processes"]
        assert app._slash_completion_candidates("/r") == ["/resume", "/run", "/redirect"]

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

        assert app._slash_completion_candidates("/i") == ["/inject", "/investment-analysis"]
        assert app._slash_completion_candidates("/m") == [
            "/mcp",
            "/model",
            "/models",
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
    async def test_process_command_is_removed(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/process")

        assert handled is False
        chat.add_info.assert_not_called()

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
    async def test_tool_command_executes_via_run_tool(self):
        from loom.tools.registry import ToolResult
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = ["read_file"]
        tools.execute = AsyncMock(return_value=ToolResult.ok("ok"))
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._session = SimpleNamespace(_auth_context=None)
        app._ingest_files_panel_from_paths = MagicMock(return_value=0)
        app._is_mutating_tool = MagicMock(return_value=False)
        app._request_workspace_refresh = MagicMock()

        handled = await app._handle_slash_command('/tool read_file {"path":"README.md"}')

        assert handled is True
        tools.execute.assert_awaited_once()
        execute_call = tools.execute.await_args
        assert execute_call.args[0] == "run_tool"
        assert execute_call.args[1] == {
            "name": "read_file",
            "arguments": {"path": "README.md"},
        }
        chat.add_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_command_uses_worker_when_app_running(self):
        from loom.tui.app import LoomApp

        class RunningHarness(LoomApp):
            @property
            def is_running(self) -> bool:  # pragma: no cover - property shim
                return True

        tools = MagicMock()
        tools.list_tools.return_value = ["read_file"]
        tools.execute = AsyncMock()
        app = RunningHarness(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._session = SimpleNamespace(_auth_context=None)
        app._execute_slash_tool_command = AsyncMock()
        captured: dict[str, object] = {}
        app.run_worker = MagicMock(
            side_effect=lambda coro, **kwargs: captured.update(coro=coro, kwargs=kwargs)
        )

        handled = await app._handle_slash_command("/tool read_file path=README.md")

        assert handled is True
        app._execute_slash_tool_command.assert_called_once_with(
            "read_file",
            {"path": "README.md"},
        )
        app._execute_slash_tool_command.assert_not_awaited()
        tools.execute.assert_not_awaited()
        assert app.run_worker.call_count == 1
        assert captured["kwargs"]["exclusive"] is False
        assert str(captured["kwargs"]["group"]).startswith("slash-tool-command-")

        await captured["coro"]
        app._execute_slash_tool_command.assert_awaited_once_with(
            "read_file",
            {"path": "README.md"},
        )

    @pytest.mark.asyncio
    async def test_tool_command_modal_from_input_submit_accepts_keys(self):
        from textual.widgets import Input

        from loom.cowork.approval import ApprovalDecision
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = ["write_file"]
        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=tools,
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()
        app._tool_name_inventory = MagicMock(return_value=["write_file"])

        decision: dict[str, ApprovalDecision | None] = {"value": None}

        async def _fake_execute(tool_name: str, tool_args: dict[str, object]) -> None:
            decision["value"] = await app._approval_callback(
                tool_name,
                {"path": tool_args.get("path", "")},
            )

        app._execute_slash_tool_command = AsyncMock(side_effect=_fake_execute)

        async with app.run_test() as pilot:
            input_widget = app.query_one("#user-input", Input)
            input_widget.value = "/tool write_file path=test.md"
            input_widget.cursor_position = len(input_widget.value)

            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], ToolApprovalScreen)

            await pilot.press("y")
            for _ in range(10):
                if decision["value"] is not None:
                    break
                await pilot.pause()

            assert all(
                not isinstance(screen, ToolApprovalScreen)
                for screen in app.screen_stack
            )
            assert decision["value"] == ApprovalDecision.APPROVE

    @pytest.mark.asyncio
    async def test_tool_command_executes_via_kv_pairs(self):
        from loom.tools.registry import ToolResult
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = ["ripgrep_search"]
        tools.execute = AsyncMock(return_value=ToolResult.ok("ok"))
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._session = SimpleNamespace(_auth_context=None)
        app._ingest_files_panel_from_paths = MagicMock(return_value=0)
        app._is_mutating_tool = MagicMock(return_value=False)
        app._request_workspace_refresh = MagicMock()

        handled = await app._handle_slash_command(
            "/tool ripgrep_search pattern=TODO max_results=25 case_sensitive=true",
        )

        assert handled is True
        tools.execute.assert_awaited_once()
        execute_call = tools.execute.await_args
        assert execute_call.args[0] == "run_tool"
        assert execute_call.args[1] == {
            "name": "ripgrep_search",
            "arguments": {
                "pattern": "TODO",
                "max_results": 25,
                "case_sensitive": True,
            },
        }
        chat.add_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_command_rejects_invalid_json(self):
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = ["read_file"]
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command('/tool read_file {"path": }')

        assert handled is True
        chat.add_info.assert_called_once()
        assert "Invalid /tool JSON arguments" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_tool_command_rejects_invalid_kv_argument(self):
        from loom.tui.app import LoomApp

        tools = MagicMock()
        tools.list_tools.return_value = ["read_file"]
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=tools,
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/tool read_file README.md")

        assert handled is True
        chat.add_info.assert_called_once()
        assert "Use key=value pairs or a JSON object" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_stop_requests_active_chat_turn(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app.action_stop_chat = MagicMock()

        handled = await app._handle_slash_command("/stop")

        assert handled is True
        app.action_stop_chat.assert_called_once()
        chat.add_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_rejects_arguments(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app.action_stop_chat = MagicMock()

        handled = await app._handle_slash_command("/stop now")

        assert handled is True
        app.action_stop_chat.assert_not_called()
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Usage" in message
        assert "/stop" in message

    @pytest.mark.asyncio
    async def test_pause_requests_active_chat_pause(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._request_chat_pause = AsyncMock(return_value=True)

        handled = await app._handle_slash_command("/pause")

        assert handled is True
        app._request_chat_pause.assert_awaited_once_with(source="slash")

    @pytest.mark.asyncio
    async def test_inject_requires_text(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._queue_chat_inject_instruction = AsyncMock(return_value=True)

        handled = await app._handle_slash_command("/inject")

        assert handled is True
        app._queue_chat_inject_instruction.assert_not_awaited()
        chat.add_info.assert_called_once()
        assert "/inject" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_redirect_requires_text(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._request_chat_redirect = AsyncMock(return_value=True)

        handled = await app._handle_slash_command("/redirect")

        assert handled is True
        app._request_chat_redirect.assert_not_awaited()
        chat.add_info.assert_called_once()
        assert "/redirect" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_steer_subcommands_dispatch(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._request_chat_pause = AsyncMock(return_value=True)
        app._request_chat_resume = AsyncMock(return_value=True)
        app._clear_chat_steering = AsyncMock(return_value=True)
        app._render_steer_queue_status = MagicMock(return_value="queue")

        assert await app._handle_slash_command("/steer pause") is True
        app._request_chat_pause.assert_awaited_once_with(source="slash")

        assert await app._handle_slash_command("/steer resume") is True
        app._request_chat_resume.assert_awaited_once_with(source="slash")

        assert await app._handle_slash_command("/steer queue") is True
        chat.add_info.assert_called_with("queue")

        assert await app._handle_slash_command("/steer clear") is True
        app._clear_chat_steering.assert_awaited_once_with(source="slash")

    @pytest.mark.asyncio
    async def test_steer_rejects_unknown_subcommand(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/steer nope")

        assert handled is True
        chat.add_info.assert_called_once()
        assert "/steer" in chat.add_info.call_args.args[0]

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
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/run analyze tesla")

        assert handled is True
        app._start_process_run.assert_awaited_once()
        assert app._start_process_run.await_args.args == ("analyze tesla",)
        kwargs = app._start_process_run.await_args.kwargs
        assert kwargs["process_defn"] is None
        assert kwargs["process_name_override"] is None
        assert kwargs["is_adhoc"] is True
        assert kwargs["synthesis_goal"] == "analyze tesla"
        assert kwargs["force_fresh"] is False

    @pytest.mark.asyncio
    async def test_run_goal_with_apostrophe_synthesizes_adhoc(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = None
        app._config = SimpleNamespace(process=SimpleNamespace(default=""))
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/run I don't know")

        assert handled is True
        app._start_process_run.assert_awaited_once()
        assert app._start_process_run.await_args.args == ("I don't know",)
        kwargs = app._start_process_run.await_args.kwargs
        assert kwargs["process_defn"] is None
        assert kwargs["process_name_override"] is None
        assert kwargs["is_adhoc"] is True
        assert kwargs["synthesis_goal"] == "I don't know"
        assert kwargs["force_fresh"] is False

    @pytest.mark.asyncio
    async def test_run_inject_accepts_apostrophe_text(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._inject_process_run_from_target = AsyncMock(return_value=True)
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/run inject current don't lose scope")

        assert handled is True
        app._inject_process_run_from_target.assert_awaited_once_with(
            "current",
            "don't lose scope",
            source="slash",
        )

    @pytest.mark.asyncio
    async def test_run_goal_with_embedded_double_quote_synthesizes_adhoc(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = None
        app._config = SimpleNamespace(process=SimpleNamespace(default=""))
        app._start_process_run = AsyncMock()
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/run Need a 6\" monitor review")

        assert handled is True
        app._start_process_run.assert_awaited_once()
        assert app._start_process_run.await_args.args == ('Need a 6" monitor review',)

    @pytest.mark.asyncio
    async def test_run_goal_with_unclosed_leading_quote_is_tolerated(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_defn = None
        app._config = SimpleNamespace(process=SimpleNamespace(default=""))
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command('/run "Need monitor review')

        assert handled is True
        app._start_process_run.assert_awaited_once()
        assert app._start_process_run.await_args.args == ('"Need monitor review',)
        chat.add_info.assert_not_called()

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
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/run --fresh analyze tesla")

        assert handled is True
        app._start_process_run.assert_awaited_once()
        assert app._start_process_run.await_args.args == ("analyze tesla",)
        kwargs = app._start_process_run.await_args.kwargs
        assert kwargs["process_defn"] is None
        assert kwargs["process_name_override"] is None
        assert kwargs["is_adhoc"] is True
        assert kwargs["synthesis_goal"] == "analyze tesla"
        assert kwargs["force_fresh"] is True

    @pytest.mark.asyncio
    async def test_run_process_flag_is_not_supported_in_tui(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command(
            "/run --process investment-analysis analyze tesla",
        )

        assert handled is True
        app._start_process_run.assert_not_awaited()
        chat.add_info.assert_called_once()
        assert "/run --process is not supported in TUI" in chat.add_info.call_args.args[0]
        assert "/investment-analysis analyze tesla" not in chat.add_info.call_args.args[0]
        assert "/<process-name> <goal>" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_run_file_goal_loads_content_for_adhoc(self, tmp_path):
        from loom.tui.app import LoomApp

        (tmp_path / "problem.md").write_text(
            "# Problem\nImplement a deterministic CSV export.",
            encoding="utf-8",
        )
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        app._process_defn = None
        app._config = SimpleNamespace(process=SimpleNamespace(default=""))
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/run problem.md")

        assert handled is True
        app._start_process_run.assert_awaited_once()
        assert app._start_process_run.await_args.args == ("problem.md",)
        kwargs = app._start_process_run.await_args.kwargs
        assert kwargs["process_defn"] is None
        assert kwargs["process_name_override"] is None
        assert kwargs["is_adhoc"] is True
        assert "problem.md" in kwargs["synthesis_goal"]
        assert "Implement a deterministic CSV export." in kwargs["synthesis_goal"]
        assert kwargs["goal_context_overrides"]["run_goal_file_input"]["path"] == "problem.md"
        assert kwargs["goal_context_overrides"]["run_goal_file_input"]["truncated"] is False
        assert "Implement a deterministic CSV export." in (
            kwargs["goal_context_overrides"]["run_goal_file_input"]["content"]
        )

    @pytest.mark.asyncio
    async def test_run_at_file_prefix_supports_inline_goal_override(self, tmp_path):
        from loom.tui.app import LoomApp

        (tmp_path / "problem.md").write_text(
            "Primary acceptance criteria for release packaging.",
            encoding="utf-8",
        )
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        app._process_defn = SimpleNamespace(name="investment-analysis")
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command(
            "/run @problem.md focus only on risk items",
        )

        assert handled is True
        app._start_process_run.assert_awaited_once_with(
            "focus only on risk items",
            process_defn=None,
            process_name_override=None,
            is_adhoc=True,
            synthesis_goal=ANY,
            force_fresh=False,
            goal_context_overrides={
                "run_goal_file_input": {
                    "path": "problem.md",
                    "content": "Primary acceptance criteria for release packaging.",
                    "truncated": False,
                    "max_chars": 32000,
                },
            },
        )

    @pytest.mark.asyncio
    async def test_run_at_file_prefix_requires_existing_workspace_file(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/run @missing-problem.md")

        assert handled is True
        app._start_process_run.assert_not_awaited()
        chat.add_info.assert_called()
        assert "Run goal file not found" in chat.add_info.call_args.args[0]

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
        assert (
            "Do not add phases whose main purpose is creating folder schemas"
            in model.prompts[0]
        )
        assert "<<<BEGIN_SOURCE>>>" in model.prompts[1]
        assert "Prefer root-level deliverable filenames" in model.prompts[1]

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

    def test_normalize_adhoc_spec_injects_default_validity_contract_by_intent(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        normalized = app._normalize_adhoc_spec(
            {
                "intent": "research",
                "name": "research-adhoc",
                "description": "Research flow",
                "persona": "Research analyst",
                "phase_mode": "strict",
                "tool_guidance": "Use evidence.",
                "required_tools": [],
                "recommended_tools": [],
                "phases": [],
            },
            goal="research major claims",
            key="aa11bb22cc33dd44",
            available_tools=[],
        )
        contract = normalized.get("validity_contract", {})

        assert isinstance(contract, dict)
        assert contract.get("enabled") is True
        assert contract.get("claim_extraction", {}).get("enabled") is True
        assert contract.get("require_fact_checker_for_synthesis") is False
        assert contract.get("final_gate", {}).get("enforce_verified_context_only") is True
        assert contract.get("final_gate", {}).get("synthesis_min_verification_tier") == 2

    def test_normalize_adhoc_spec_merges_model_validity_contract_with_default_floor(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        normalized = app._normalize_adhoc_spec(
            {
                "intent": "writing",
                "name": "writing-adhoc",
                "description": "Writing flow",
                "persona": "Writer",
                "phase_mode": "strict",
                "tool_guidance": "Write carefully.",
                "required_tools": [],
                "recommended_tools": [],
                "validity_contract": {
                    "min_supported_ratio": 0.9,
                    "max_unverified_ratio": 0.1,
                    "final_gate": {"synthesis_min_verification_tier": 3},
                },
                "phases": [],
            },
            goal="draft a high-risk recommendation memo",
            key="ee55ff66778899aa",
            available_tools=[],
        )
        contract = normalized.get("validity_contract", {})

        assert contract.get("enabled") is True
        assert contract.get("claim_extraction", {}).get("enabled") is True
        assert contract.get("min_supported_ratio") == 0.9
        assert contract.get("max_unverified_ratio") == 0.1
        assert contract.get("final_gate", {}).get("synthesis_min_verification_tier") == 3
        assert contract.get("require_fact_checker_for_synthesis") is False

    def test_normalize_adhoc_spec_injects_default_verification_policy(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        normalized = app._normalize_adhoc_spec(
            {
                "intent": "research",
                "name": "research-adhoc",
                "description": "Research flow",
                "persona": "Research analyst",
                "phase_mode": "strict",
                "tool_guidance": "Use evidence.",
                "required_tools": [],
                "recommended_tools": [],
                "phases": [],
            },
            goal="research major claims",
            key="aa11bb22cc33dd44",
            available_tools=[],
        )
        policy = normalized.get("verification_policy", {})

        assert policy.get("mode") == "llm_first"
        assert (
            policy.get("static_checks", {}).get("tool_success_policy")
            == "method_resilient"
        )
        assert policy.get("semantic_checks") == []
        assert policy.get("output_contract") == {}
        assert policy.get("outcome_policy") == {}

    def test_normalize_adhoc_spec_injects_build_verification_policy(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        normalized = app._normalize_adhoc_spec(
            {
                "intent": "build",
                "name": "build-adhoc",
                "description": "Build flow",
                "persona": "Software engineer",
                "phase_mode": "guided",
                "tool_guidance": "Implement and verify.",
                "required_tools": [],
                "recommended_tools": [],
                "phases": [],
            },
            goal="Fix the broken build and verify the web app",
            key="cc33dd44ee55ff66",
            available_tools=[],
        )
        policy = normalized.get("verification_policy", {})

        assert policy.get("mode") == "static_first"
        assert (
            policy.get("static_checks", {}).get("tool_success_policy")
            == "development_balanced"
        )
        assert policy.get("output_contract", {}).get("required_fields") == [
            "passed",
            "outcome",
            "reason_code",
            "severity_class",
        ]
        assert policy.get("outcome_policy", {}).get("treat_verifier_infra_as_warning") is True

    def test_fallback_adhoc_spec_prefers_verification_helper_for_build_intent(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        spec = app._fallback_adhoc_spec(
            "build a microsite from this dataset",
            available_tools=[
                "search_files",
                "read_file",
                "write_file",
                "verification_helper",
                "shell_execute",
                "ripgrep_search",
                "document_write",
            ],
            intent="build",
        )

        assert "verification_helper" in spec["required_tools"]
        assert spec["required_tools"].index("verification_helper") < spec["required_tools"].index(
            "shell_execute"
        )
        assert "verification_helper" in spec["tool_guidance"]

    def test_normalize_adhoc_spec_merges_model_verification_policy(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        normalized = app._normalize_adhoc_spec(
            {
                "intent": "research",
                "name": "research-adhoc",
                "description": "Research flow",
                "persona": "Research analyst",
                "phase_mode": "strict",
                "tool_guidance": "Use evidence.",
                "required_tools": [],
                "recommended_tools": [],
                "verification_policy": {
                    "mode": "static_first",
                    "static_checks": {"phase_scope": "global"},
                    "semantic_checks": [{"name": "coverage"}, "ignore-me"],
                    "output_contract": {"required_fields": ["passed"]},
                    "outcome_policy": {"allow_partial_verified": True},
                },
                "phases": [],
            },
            goal="research major claims",
            key="bb22cc33dd44ee55",
            available_tools=[],
        )
        policy = normalized.get("verification_policy", {})

        assert policy.get("mode") == "static_first"
        assert (
            policy.get("static_checks", {}).get("tool_success_policy")
            == "method_resilient"
        )
        assert policy.get("static_checks", {}).get("phase_scope") == "global"
        assert policy.get("semantic_checks") == [{"name": "coverage"}]
        assert policy.get("output_contract") == {"required_fields": ["passed"]}
        assert policy.get("outcome_policy") == {"allow_partial_verified": True}

    def test_build_adhoc_cache_entry_preserves_verification_policy(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        spec = app._fallback_adhoc_spec(
            "research major claims",
            available_tools=[],
            intent="research",
        )
        spec["name"] = "policy-roundtrip-adhoc"
        spec["verification_policy"] = {
            "mode": "static_first",
            "static_checks": {
                "tool_success_policy": "all_tools_hard",
                "phase_scope": "global",
            },
            "semantic_checks": [{"name": "coverage"}, "ignore-me"],
            "output_contract": {"required_fields": ["passed"]},
            "outcome_policy": {"allow_partial_verified": True},
        }

        entry = app._build_adhoc_cache_entry(
            key="11aa22bb33cc44dd",
            goal="research major claims",
            spec=spec,
        )
        policy = entry.process_defn.verification_policy

        assert policy.mode == "static_first"
        assert policy.static_checks.get("tool_success_policy") == "all_tools_hard"
        assert policy.static_checks.get("phase_scope") == "global"
        assert policy.semantic_checks == [{"name": "coverage"}]
        assert policy.output_contract == {"required_fields": ["passed"]}
        assert policy.outcome_policy == {"allow_partial_verified": True}

    def test_normalize_adhoc_spec_infers_high_risk_level_from_goal(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        normalized = app._normalize_adhoc_spec(
            {
                "intent": "build",
                "name": "finance-build-adhoc",
                "description": "Build workflow",
                "persona": "Engineer",
                "phase_mode": "strict",
                "tool_guidance": "Use tools.",
                "required_tools": [],
                "recommended_tools": [],
                "phases": [],
            },
            goal="build an investment recommendation engine",
            key="cc22dd33ee44ff55",
            available_tools=[],
        )

        assert normalized.get("risk_level") == "high"
        contract = normalized.get("validity_contract", {})
        assert contract.get("require_fact_checker_for_synthesis") is True
        assert contract.get("max_unverified_ratio") <= 0.2
        temporal = contract.get("final_gate", {}).get("temporal_consistency", {})
        assert temporal.get("enabled") is True

    def test_normalize_adhoc_spec_respects_explicit_model_risk_level(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        normalized = app._normalize_adhoc_spec(
            {
                "intent": "build",
                "risk_level": "low",
                "name": "safe-build-adhoc",
                "description": "Build workflow",
                "persona": "Engineer",
                "phase_mode": "strict",
                "tool_guidance": "Use tools.",
                "required_tools": [],
                "recommended_tools": [],
                "phases": [],
            },
            goal="build an investment recommendation engine",
            key="dd33ee44ff556677",
            available_tools=[],
        )

        assert normalized.get("risk_level") == "low"
        contract = normalized.get("validity_contract", {})
        assert contract.get("require_fact_checker_for_synthesis") is False

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
    async def test_run_defaults_to_adhoc_even_when_process_is_loaded(self):
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
            process_defn=None,
            process_name_override=None,
            is_adhoc=True,
            synthesis_goal="Analyze Tesla for investment",
            force_fresh=False,
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

    @pytest.mark.asyncio
    async def test_run_pause_routes_to_pause_handler(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._pause_process_run_from_target = AsyncMock(return_value=True)
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/run pause current")

        assert handled is True
        app._pause_process_run_from_target.assert_awaited_once_with("current")

    @pytest.mark.asyncio
    async def test_run_play_routes_to_play_handler(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._play_process_run_from_target = AsyncMock(return_value=True)
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/run play current")

        assert handled is True
        app._play_process_run_from_target.assert_awaited_once_with("current")

    @pytest.mark.asyncio
    async def test_run_stop_routes_to_stop_handler(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._stop_process_run_from_target = AsyncMock(return_value=True)
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/run stop current")

        assert handled is True
        app._stop_process_run_from_target.assert_awaited_once_with("current")

    @pytest.mark.asyncio
    async def test_run_inject_routes_to_inject_handler(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._inject_process_run_from_target = AsyncMock(return_value=True)
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command('/run inject current "stay on task"')

        assert handled is True
        app._inject_process_run_from_target.assert_awaited_once_with(
            "current",
            "stay on task",
            source="slash",
        )

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

    def test_process_run_toggle_button_dispatches_pause_when_running(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(run_id="abc123", closed=False, status="running")
        app._process_runs = {"abc123": run}
        app._pause_process_run = AsyncMock(return_value=True)
        app._play_process_run = AsyncMock(return_value=True)
        app.run_worker = MagicMock()
        event = SimpleNamespace(
            button=SimpleNamespace(id="process-run-toggle-abc123"),
            stop=MagicMock(),
            prevent_default=MagicMock(),
        )

        app.on_process_run_control_pressed(event)

        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()
        app.run_worker.assert_called_once()
        assert app.run_worker.call_args.kwargs["name"] == "process-run-pause-abc123"
        worker_coro = app.run_worker.call_args.args[0]
        assert asyncio.iscoroutine(worker_coro)
        worker_coro.close()

    def test_process_run_toggle_button_dispatches_play_when_paused(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(run_id="abc123", closed=False, status="paused")
        app._process_runs = {"abc123": run}
        app._pause_process_run = AsyncMock(return_value=True)
        app._play_process_run = AsyncMock(return_value=True)
        app.run_worker = MagicMock()
        event = SimpleNamespace(
            button=SimpleNamespace(id="process-run-toggle-abc123"),
            stop=MagicMock(),
            prevent_default=MagicMock(),
        )

        app.on_process_run_control_pressed(event)

        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()
        app.run_worker.assert_called_once()
        assert app.run_worker.call_args.kwargs["name"] == "process-run-play-abc123"
        worker_coro = app.run_worker.call_args.args[0]
        assert asyncio.iscoroutine(worker_coro)
        worker_coro.close()

    def test_process_run_toggle_button_ignores_non_actionable_status(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(run_id="abc123", closed=False, status="queued")
        app._process_runs = {"abc123": run}
        app._pause_process_run = AsyncMock(return_value=True)
        app._play_process_run = AsyncMock(return_value=True)
        app.run_worker = MagicMock()
        event = SimpleNamespace(
            button=SimpleNamespace(id="process-run-toggle-abc123"),
            stop=MagicMock(),
            prevent_default=MagicMock(),
        )

        app.on_process_run_control_pressed(event)

        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()
        app.run_worker.assert_not_called()

    def test_process_run_stop_button_dispatches_confirmed_stop(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(run_id="abc123", closed=False, status="running")
        app._process_runs = {"abc123": run}
        app._stop_process_run = AsyncMock(return_value=True)
        app.run_worker = MagicMock()
        event = SimpleNamespace(
            button=SimpleNamespace(id="process-run-stop-abc123"),
            stop=MagicMock(),
            prevent_default=MagicMock(),
        )

        app.on_process_run_control_pressed(event)

        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()
        app._stop_process_run.assert_called_once_with(run, confirm=True)
        app.run_worker.assert_called_once()
        assert app.run_worker.call_args.kwargs["name"] == "process-run-stop-abc123"
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
        task_rows = run.pane.set_tasks.call_args.args[0]
        assert task_rows[0]["id"] == "stage:summary"
        assert "Queueing delegate" in task_rows[0]["content"]
        assert task_rows[1:] == [{"id": "old", "status": "pending", "content": "old"}]
        app._persist_process_run_ui_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_restart_process_run_in_place_without_task_id_retries_goal(self):
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
            task_id="",
            run_workspace=Path("/tmp/process-run"),
            pane_id="tab-run-abc123",
            pane=MagicMock(),
            is_adhoc=True,
            recommended_tools=["ripgrep_search"],
            goal_context_overrides={"priority": "high"},
            tasks=[{"id": "old", "status": "failed", "content": "old"}],
            task_labels={"old": "old"},
            subtask_phase_ids={"old": "phase-1"},
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
        app._prepare_and_execute_process_run = AsyncMock(return_value=None)
        app._execute_process_run = AsyncMock(return_value=None)
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
        assert run.tasks == []
        assert run.task_labels == {}
        assert run.subtask_phase_ids == {}
        assert run.worker is not None
        assert captured["kwargs"]["name"] == "process-run-abc123"
        assert captured["kwargs"]["group"] == "process-run-abc123"
        task_rows = run.pane.set_tasks.call_args.args[0]
        assert task_rows[0]["id"] in {"stage:summary", "stage:accepted"}
        assert (
            "Queueing delegate" in task_rows[0]["content"]
            or "Accepted" in task_rows[0]["content"]
        )
        app._prepare_and_execute_process_run.assert_called_once()
        app._execute_process_run.assert_not_called()
        app._persist_process_run_ui_state.assert_awaited_once()

    def test_ensure_delegate_task_ready_for_run_detects_bound_delegate(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(),
            db=MagicMock(),
        )
        app._session = SimpleNamespace(session_id="s1")
        app._ensure_persistence_tools = MagicMock()
        app._bind_session_tools = MagicMock()
        app._tools.get = MagicMock(return_value=SimpleNamespace(_factory=lambda: None))

        ready, reason = app._ensure_delegate_task_ready_for_run()

        assert ready is True
        assert reason == ""
        app._ensure_persistence_tools.assert_called_once()
        app._bind_session_tools.assert_called_once()
        app._tools.get.assert_called_once_with("delegate_task")

    def test_ensure_delegate_task_ready_for_run_reports_unbound_delegate(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(),
            db=MagicMock(),
        )
        app._session = SimpleNamespace(session_id="s1")
        app._ensure_persistence_tools = MagicMock()
        app._bind_session_tools = MagicMock()
        app._tools.get = MagicMock(return_value=SimpleNamespace(_factory=None))

        ready, reason = app._ensure_delegate_task_ready_for_run()

        assert ready is False
        assert "unbound" in reason.lower()

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
    async def test_play_process_run_falls_back_to_resume_when_restored_paused(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            run_id="abc123",
            process_name="market-research",
            goal="Analyze EPCOR",
            status="paused",
            task_id="cowork-9",
            worker=None,
            closed=False,
        )
        app._request_process_run_play = AsyncMock(return_value={
            "requested": False,
            "path": "none",
            "error": "Play is unavailable until delegate task control is ready.",
            "status": "paused",
        })
        app._restart_process_run_in_place = AsyncMock(return_value=True)
        chat = MagicMock()
        events_panel = MagicMock()
        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: {
            "#chat-log": chat,
            "#events-panel": events_panel,
        }[selector])

        played = await app._play_process_run(run)

        assert played is True
        app._restart_process_run_in_place.assert_awaited_once_with("abc123", mode="resume")

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
        app._start_process_run = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command(
            '/investment-analysis "Analyze Tesla for investment"',
        )

        assert handled is True
        app._start_process_run.assert_awaited_once_with(
            "Analyze Tesla for investment",
            process_name_override="investment-analysis",
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
        assert run.run_workspace == Path("/tmp")
        assert run.launch_stage == "accepted"
        assert tabs.active == run.pane_id
        chat.add_user_message.assert_called_once_with("/run Analyze Tesla")

    @pytest.mark.asyncio
    async def test_start_process_run_uses_inline_preflight_when_disabled(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(
                tui=SimpleNamespace(run_preflight_async_enabled=False),
            ),
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
        app._prepare_process_run_with_timeout = AsyncMock(return_value=True)
        captured: dict[str, object] = {}

        def fake_run_worker(coro, **kwargs):
            captured["kwargs"] = kwargs
            captured["coro_name"] = coro.cr_code.co_name
            coro.close()
            return MagicMock()

        app.run_worker = fake_run_worker

        await app._start_process_run("Analyze Tesla")

        app._prepare_process_run_with_timeout.assert_awaited_once()
        assert captured["coro_name"] == "_execute_process_run"
        assert captured["kwargs"]["group"].startswith("process-run-")

    @pytest.mark.asyncio
    async def test_prepare_process_run_launch_refreshes_workspace_immediately(self, tmp_path):
        from loom.tui.app import LoomApp, ProcessRunLaunchRequest, ProcessRunState

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        app._request_workspace_refresh = MagicMock()
        app._ingest_files_panel_from_paths = MagicMock(return_value=1)
        app._resolve_auth_overrides_for_run_start = AsyncMock(return_value=({}, []))
        app._choose_process_run_workspace = AsyncMock(
            return_value=tmp_path / "runs" / "market-research-run",
        )
        pane = MagicMock()
        run = ProcessRunState(
            run_id="abc123",
            process_name="market-research",
            goal="Analyze EPCOR",
            run_workspace=tmp_path,
            process_defn=SimpleNamespace(name="market-research"),
            pane_id="tab-run-abc123",
            pane=pane,
            status="queued",
        )
        app._process_runs = {"abc123": run}

        prepared = await app._prepare_process_run_launch(
            "abc123",
            ProcessRunLaunchRequest(
                goal="Analyze EPCOR",
                process_defn=SimpleNamespace(name="market-research"),
            ),
        )

        assert prepared is True
        app._request_workspace_refresh.assert_called_once_with(
            "run-workspace-created",
            immediate=True,
        )
        app._ingest_files_panel_from_paths.assert_called_once_with(
            ["runs/market-research-run"],
            operation_hint="create",
        )

    @pytest.mark.asyncio
    async def test_prepare_process_run_launch_cancels_when_workspace_selection_cancelled(
        self,
        tmp_path,
    ):
        from loom.tui.app import LoomApp, ProcessRunLaunchRequest, ProcessRunState

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        app._request_workspace_refresh = MagicMock()
        app._ingest_files_panel_from_paths = MagicMock(return_value=0)
        app._resolve_auth_overrides_for_run_start = AsyncMock(return_value=({}, []))
        app._choose_process_run_workspace = AsyncMock(return_value=None)
        pane = MagicMock()
        run = ProcessRunState(
            run_id="abc123",
            process_name="market-research",
            goal="Analyze EPCOR",
            run_workspace=tmp_path,
            process_defn=SimpleNamespace(name="market-research"),
            pane_id="tab-run-abc123",
            pane=pane,
            status="queued",
        )
        app._process_runs = {"abc123": run}

        prepared = await app._prepare_process_run_launch(
            "abc123",
            ProcessRunLaunchRequest(
                goal="Analyze EPCOR",
                process_defn=SimpleNamespace(name="market-research"),
            ),
        )

        assert prepared is False
        assert run.status == "failed"
        assert "working folder selection cancelled" in run.launch_error.lower()
        app._request_workspace_refresh.assert_not_called()
        app._ingest_files_panel_from_paths.assert_not_called()

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

    def test_cowork_compactor_model_prefers_compactor_role_model(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        compactor_model = MagicMock(name="compactor-model")
        compactor_model.name = "compactor-model"

        active_model = MagicMock(name="active-executor-model")

        fake_router = MagicMock()

        def _select(*, tier=1, role="executor"):
            assert role == "compactor"
            assert tier == 1
            return compactor_model

        fake_router.select = MagicMock(side_effect=_select)
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
                "compactor": ModelConfig(
                    provider="ollama",
                    base_url="http://localhost:11434",
                    model="compactor-model",
                    roles=["compactor"],
                ),
            }),
        )

        selected = app._cowork_compactor_model()

        assert selected is compactor_model
        assert fake_router.select.call_count == 1

    def test_cowork_memory_indexer_model_prefers_compactor_role(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        compactor_model = MagicMock(name="compactor-memory-model")
        compactor_model.name = "compactor-memory-model"
        active_model = MagicMock(name="active-model")

        fake_router = MagicMock()

        def _select(*, tier=1, role="executor"):
            if role == "compactor":
                return compactor_model
            raise RuntimeError("unexpected role")

        fake_router.select = MagicMock(side_effect=_select)
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
                "compactor": ModelConfig(
                    provider="ollama",
                    base_url="http://localhost:11434",
                    model="compactor-model",
                    roles=["compactor"],
                ),
            }),
        )

        model, role = app._cowork_memory_indexer_model()
        assert model is compactor_model
        assert role == "compactor"

    def test_cowork_memory_indexer_model_falls_back_to_extractor(
        self,
        tmp_path,
        monkeypatch,
    ):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        extractor_model = MagicMock(name="extractor-memory-model")
        extractor_model.name = "extractor-memory-model"
        active_model = MagicMock(name="active-model")

        fake_router = MagicMock()

        def _select(*, tier=1, role="executor"):
            if role == "compactor":
                raise RuntimeError("missing compactor")
            if role == "extractor":
                return extractor_model
            raise RuntimeError("unexpected role")

        fake_router.select = MagicMock(side_effect=_select)
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

        model, role = app._cowork_memory_indexer_model()
        assert model is extractor_model
        assert role == "extractor"

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
    async def test_choose_process_run_workspace_use_root_selection(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        app._llm_process_run_folder_name = AsyncMock(
            return_value="market-research-analyze-encor",
        )
        app._prompt_process_run_workspace_choice = AsyncMock(return_value="")
        chat = MagicMock()

        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: (
            chat if selector == "#chat-log" else None
        ))

        selected = await app._choose_process_run_workspace(
            "market-research",
            "Analyze Encor",
        )

        assert selected == tmp_path.resolve()
        app._prompt_process_run_workspace_choice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_choose_process_run_workspace_use_folder_selection(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        app._llm_process_run_folder_name = AsyncMock(
            return_value="market-research-analyze-encor",
        )
        app._prompt_process_run_workspace_choice = AsyncMock(return_value="run-workdir")
        chat = MagicMock()

        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: (
            chat if selector == "#chat-log" else None
        ))

        selected = await app._choose_process_run_workspace(
            "market-research",
            "Analyze Encor",
        )

        assert selected == (tmp_path / "run-workdir")
        assert selected.exists()
        app._prompt_process_run_workspace_choice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_choose_process_run_workspace_reprompts_on_invalid_selection(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        app._llm_process_run_folder_name = AsyncMock(return_value="market-research")
        app._prompt_process_run_workspace_choice = AsyncMock(
            side_effect=["../outside", "runs/research-scan"],
        )
        chat = MagicMock()

        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: (
            chat if selector == "#chat-log" else None
        ))

        selected = await app._choose_process_run_workspace(
            "market-research",
            "Analyze Encor",
        )

        assert selected == (tmp_path / "runs" / "research-scan")
        assert selected.exists()
        assert app._prompt_process_run_workspace_choice.await_count == 2
        chat.add_info.assert_called_once()
        assert "Invalid working folder" in chat.add_info.call_args.args[0]

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

    @pytest.mark.asyncio
    async def test_execute_process_run_treats_cancelled_delegate_status_as_cancelled(self):
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
            success=False,
            output='Task cancelled: "Analyze market"',
            error="Task execution cancelled.",
            data={"task_id": "cowork-1", "status": "cancelled", "tasks": []},
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
        assert run.status == "cancelled"
        chat.add_info.assert_any_call("[bold #f7768e]Process run abc123 cancelled.[/]")

    @pytest.mark.asyncio
    async def test_execute_process_run_cancellation_preserves_paused_state(self):
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
                launch_error="",
            ),
        }

        async def _execute(_tool_name, payload, **_kwargs):
            progress_cb = payload.get("_progress_callback")
            if callable(progress_cb):
                progress_cb(
                    {
                        "status": "paused",
                        "event_type": "task_paused",
                        "event_data": {"requested": True, "status": "paused"},
                        "tasks": [],
                    },
                )
            raise asyncio.CancelledError

        app._tools.execute = AsyncMock(side_effect=_execute)
        app._update_process_run_visuals = MagicMock()
        app._refresh_process_run_progress = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._refresh_workspace_tree = MagicMock()
        app._persist_process_run_ui_state = AsyncMock()
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

        with pytest.raises(asyncio.CancelledError):
            await app._execute_process_run("abc123")

        run = app._process_runs["abc123"]
        assert run.status == "paused"
        assert run.ended_at is None
        assert run.launch_error == ""
        cancelled_msgs = [
            call.args[0]
            for call in chat.add_info.call_args_list
            if call.args
        ]
        assert not any("cancelled" in msg.lower() for msg in cancelled_msgs)

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
    async def test_close_process_run_nonrunning_cancelled_by_user_keeps_tab(self):
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
            status="completed",
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
        app._confirm_close_process_run.assert_awaited_once_with(run)
        run.worker.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_process_run_running_requests_cancel_and_closes_after_settle(self):
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
            goal="Analyze Tesla",
            started_at=0.0,
            ended_at=None,
            closed=False,
            worker=worker,
            task_id="cowork-1",
        )
        app._process_runs = {"abc123": run}
        app._confirm_close_process_run = AsyncMock(return_value=True)
        app._request_process_run_cancellation = AsyncMock(return_value={
            "requested": True,
            "path": "orchestrator",
            "error": "",
            "timeout": False,
        })
        app._wait_for_process_run_terminal_state = AsyncMock(return_value=True)
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
        assert run.status == "cancel_requested"
        app._confirm_close_process_run.assert_not_awaited()
        pane.add_activity.assert_called()
        pane.add_result.assert_not_called()
        worker.cancel.assert_not_called()
        app._request_process_run_cancellation.assert_awaited_once_with(run)
        app._wait_for_process_run_terminal_state.assert_awaited_once()
        tabs.remove_pane.assert_awaited_once_with("tab-run-abc123")
        assert "abc123" not in app._process_runs
        messages = [call.args[0] for call in chat.add_info.call_args_list]
        assert any("Cancel requested" in msg for msg in messages)
        assert any("after cancellation settled" in msg for msg in messages)

    @pytest.mark.asyncio
    async def test_close_process_run_queued_bypasses_confirm_and_closes_via_worker_fallback(self):
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
            status="queued",
            goal="Analyze Tesla",
            started_at=0.0,
            ended_at=None,
            closed=False,
            worker=worker,
            task_id="",
        )
        app._process_runs = {"abc123": run}
        app._confirm_close_process_run = AsyncMock(return_value=False)
        app._request_process_run_cancellation = AsyncMock(return_value={
            "requested": True,
            "path": "worker_fallback",
            "error": "",
            "timeout": False,
        })
        app._wait_for_process_run_terminal_state = AsyncMock(return_value=False)
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
        assert run.status == "cancelled"
        app._confirm_close_process_run.assert_not_awaited()
        app._request_process_run_cancellation.assert_awaited_once_with(run)
        app._wait_for_process_run_terminal_state.assert_not_awaited()
        tabs.remove_pane.assert_awaited_once_with("tab-run-abc123")
        assert "abc123" not in app._process_runs
        worker.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirm_stop_process_run_copy_warns_terminal(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(run_id="abc123", process_name="investment-analysis")
        captured_screen: dict[str, object] = {}

        def _push_screen(screen, callback):
            captured_screen["screen"] = screen
            callback(False)

        app.push_screen = MagicMock(side_effect=_push_screen)

        confirmed = await app._confirm_stop_process_run(run)

        assert confirmed is False
        screen = captured_screen["screen"]
        assert "Stop process investment-analysis #abc123?" in screen._prompt_override
        assert "can't be revived" in screen._detail_override
        assert screen._confirm_label == "Stop Process"
        assert screen._cancel_label == "Keep Running"

    @pytest.mark.asyncio
    async def test_process_run_close_screen_mounts_when_running_true(self):
        from loom.tui.app import LoomApp
        from loom.tui.screens.process_run_close import ProcessRunCloseScreen

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test(size=(120, 40)) as pilot:
            app.push_screen(
                ProcessRunCloseScreen(
                    run_label="investment-analysis #abc123",
                    running=True,
                ),
            )
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, ProcessRunCloseScreen)
            assert screen.is_mounted is True
            assert len(list(screen.children)) == 1
            screen.query_one("#process-close-dialog")

    @pytest.mark.asyncio
    async def test_stop_process_run_respects_confirmation_decline(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            run_id="abc123",
            process_name="investment-analysis",
            status="running",
            closed=False,
            cancel_requested_at=0.0,
        )
        app._confirm_stop_process_run = AsyncMock(return_value=False)
        app._request_process_run_cancellation = AsyncMock()
        app._persist_process_run_ui_state = AsyncMock()
        app._update_process_run_visuals = MagicMock()
        app._refresh_process_run_progress = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        chat = MagicMock()
        events_panel = MagicMock()

        def _query_one(selector, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#events-panel":
                return events_panel
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        stopped = await app._stop_process_run(run, confirm=True)

        assert stopped is False
        assert run.status == "running"
        app._confirm_stop_process_run.assert_awaited_once_with(run)
        app._request_process_run_cancellation.assert_not_awaited()
        app._persist_process_run_ui_state.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_request_process_run_cancellation_bridge_uses_immediate_ack_wait(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        worker = MagicMock()
        run = SimpleNamespace(
            run_id="abc123",
            status="running",
            worker=worker,
            closed=False,
            ended_at=None,
        )
        app._process_runs = {"abc123": run}
        observed: dict[str, float] = {}

        async def _cancel_handler(**kwargs):
            observed["wait_timeout_seconds"] = float(
                kwargs.get("wait_timeout_seconds", -1.0),
            )
            return {
                "requested": True,
                "path": "orchestrator",
                "error": "",
                "timeout": False,
                "status": "cancelled",
            }

        app._process_run_cancel_handlers["abc123"] = _cancel_handler

        result = await app._request_process_run_cancellation(run)

        assert result["requested"] is True
        assert result["path"] == "orchestrator"
        assert observed["wait_timeout_seconds"] == 0.0
        assert run.status == "cancelled"
        assert run.ended_at is not None
        worker.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_process_run_cancel_timeout_keeps_tab_when_force_close_declined(self):
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
            goal="Analyze Tesla",
            started_at=0.0,
            ended_at=None,
            closed=False,
            worker=worker,
            task_id="cowork-1",
        )
        app._process_runs = {"abc123": run}
        app._confirm_close_process_run = AsyncMock(return_value=True)
        app._request_process_run_cancellation = AsyncMock(return_value={
            "requested": True,
            "path": "orchestrator",
            "error": "",
            "timeout": True,
        })
        app._confirm_force_close_process_run = AsyncMock(return_value=False)
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

        assert closed is False
        assert run.closed is False
        assert run.status == "cancel_failed"
        worker.cancel.assert_not_called()
        tabs.remove_pane.assert_not_awaited()
        assert "abc123" in app._process_runs
        app._confirm_force_close_process_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_process_run_cancel_timeout_force_closes_with_worker_cancel(self):
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
            goal="Analyze Tesla",
            started_at=0.0,
            ended_at=None,
            closed=False,
            worker=worker,
            task_id="cowork-1",
        )
        app._process_runs = {"abc123": run}
        app._confirm_close_process_run = AsyncMock(return_value=True)
        app._request_process_run_cancellation = AsyncMock(return_value={
            "requested": True,
            "path": "orchestrator",
            "error": "",
            "timeout": True,
        })
        app._confirm_force_close_process_run = AsyncMock(return_value=True)
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
        assert run.status == "force_closed"
        worker.cancel.assert_called_once()
        tabs.remove_pane.assert_awaited_once_with("tab-run-abc123")
        assert "abc123" not in app._process_runs

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
        app._store.patch_session_state_metadata = AsyncMock()
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

        app._store.patch_session_state_metadata.assert_awaited_once()
        payload = app._store.patch_session_state_metadata.await_args.kwargs["ui_state"]
        tabs = payload["process_tabs"]
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

    @pytest.mark.asyncio
    async def test_restore_process_run_tabs_keeps_paused_run_resumable(self, monkeypatch):
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
                                "status": "paused",
                                "task_id": "cowork-9",
                                "elapsed_seconds": 42.0,
                                "tasks": [],
                                "task_labels": {},
                                "activity_log": ["Run paused."],
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

        now = {"value": 100.0}
        monkeypatch.setattr("loom.tui.app.time.monotonic", lambda: now["value"])

        await app._restore_process_run_tabs(chat)

        run = app._process_runs["abc123"]
        assert run.status == "paused"
        assert run.ended_at is None
        assert run.paused_started_at == 100.0
        assert app._elapsed_seconds_for_run(run) == pytest.approx(42.0)

        now["value"] = 130.0
        assert app._elapsed_seconds_for_run(run) == pytest.approx(42.0)
        message = chat.add_info.call_args.args[0]
        assert "interrupted run(s)" not in message.lower()

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

    @pytest.mark.asyncio
    async def test_prompt_process_run_question_submits_answer(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            run_id="abc123",
            closed=False,
            status="running",
        )
        app._process_runs = {"abc123": run}
        app._append_process_run_activity = MagicMock()
        app._request_process_run_question_answer = AsyncMock(return_value={"requested": True})

        def _push_screen(_screen, callback):
            callback({
                "response_type": "single_choice",
                "selected_option_ids": ["python"],
                "selected_labels": ["Python"],
                "custom_response": "",
                "source": "tui",
            })

        app.push_screen = _push_screen

        await app._prompt_process_run_question(
            run_id="abc123",
            question_payload={
                "question_id": "q1",
                "subtask_id": "s1",
                "question": "Pick language",
                "question_type": "single_choice",
                "options": [
                    {"id": "python", "label": "Python", "description": ""},
                    {"id": "rust", "label": "Rust", "description": ""},
                ],
            },
        )

        app._request_process_run_question_answer.assert_awaited_once()
        _, kwargs = app._request_process_run_question_answer.await_args
        assert kwargs["question_id"] == "q1"
        assert kwargs["answer_payload"]["selected_option_ids"] == ["python"]
        assert kwargs["answer_payload"]["source"] == "tui"

    def test_process_progress_event_ask_user_requested_spawns_worker(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            run_id="abc123",
            process_name="investment-analysis",
            goal="Analyze Tesla",
            pane_id="tab-run-abc123",
            pane=MagicMock(),
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
            launch_last_progress_at=0.0,
            launch_last_heartbeat_at=0.0,
            launch_silent_warning_emitted=False,
            launch_error="",
            progress_ui_last_refresh_at=0.0,
        )
        app._process_runs = {"abc123": run}
        app._refresh_process_run_progress = MagicMock()
        app._refresh_process_run_outputs = MagicMock()
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._append_process_run_activity = MagicMock()
        app._format_process_progress_event = MagicMock(return_value="Clarification requested.")
        app.run_worker = MagicMock()
        app.query_one = MagicMock(return_value=MagicMock())

        app._on_process_progress_event(
            {
                "event_type": "ask_user_requested",
                "event_data": {
                    "question_id": "q-1",
                    "subtask_id": "s1",
                    "question": "Pick runtime",
                    "question_type": "single_choice",
                    "options": [
                        {"id": "python", "label": "Python", "description": ""},
                        {"id": "rust", "label": "Rust", "description": ""},
                    ],
                },
                "status": "executing",
                "tasks": [],
            },
            run_id="abc123",
        )

        assert app.run_worker.call_count == 1
        assert "q-1" in app._process_run_seen_questions["abc123"]
        coro = app.run_worker.call_args.args[0]
        coro.close()

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
        assert any(
            key.startswith("pitch-analysis.md (inspect-pitch)")
            for key in by_content
        )
        assert any(
            key.startswith("era-coverage-matrix.csv (map-eras)")
            for key in by_content
        )
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

    def test_process_progress_outputs_use_phase_id_from_task_row(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )

        pane = MagicMock()
        process_defn = SimpleNamespace(
            phases=[
                SimpleNamespace(
                    id="slogan-divergence",
                    description="Generate slogan longlist",
                ),
            ],
            get_deliverables=lambda: {"slogan-divergence": ["slogan-longlist.csv"]},
        )
        run = SimpleNamespace(
            run_id="abc129",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
            run_workspace=tmp_path,
            process_defn=process_defn,
            pane_id="tab-run-abc129",
            pane=pane,
            status="running",
            task_id="",
            started_at=0.0,
            ended_at=None,
            tasks=[],
            task_labels={},
            subtask_phase_ids={},
            last_progress_message="",
            last_progress_at=0.0,
            worker=None,
            closed=False,
        )
        app._process_runs = {"abc129": run}
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
                "event_data": {"subtask_id": "generate-slogan-longlist"},
                "tasks": [
                    {
                        "id": "generate-slogan-longlist",
                        "status": "completed",
                        "phase_id": "slogan-divergence",
                        "content": "Generated longlist.",
                    },
                ],
            },
            run_id="abc129",
        )

        output_rows = pane.set_outputs.call_args.args[0]
        by_content = {row["content"]: row for row in output_rows}
        assert "slogan-longlist.csv (slogan-divergence) (missing)" in by_content
        assert by_content["slogan-longlist.csv (slogan-divergence) (missing)"]["status"] == (
            "failed"
        )

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

    def test_process_progress_event_formats_plan_normalized(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        message = app._format_process_progress_event({
            "event_type": "task_plan_normalized",
            "event_data": {
                "context": "planner",
                "normalized_subtasks": [
                    {"subtask_id": "intermediate-synth"},
                ],
            },
        })

        assert message is not None
        assert "Normalized plan (planner)" in message
        assert "intermediate-synth" in message

    def test_process_progress_event_formats_stalled_reason(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        message = app._format_process_progress_event({
            "event_type": "task_stalled",
            "event_data": {
                "attempt": 1,
                "blocked_subtasks": [
                    {
                        "subtask_id": "downstream",
                        "reasons": ["dependency_unmet:upstream=failed"],
                    },
                ],
            },
        })

        assert message is not None
        assert "Execution stalled (attempt 1)" in message
        assert "downstream blocked" in message
        assert "dependency_unmet:upstream=failed" in message

    def test_process_progress_event_formats_stall_recovery(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        success = app._format_process_progress_event({
            "event_type": "task_stalled_recovery_attempted",
            "event_data": {
                "attempt": 1,
                "recovery_mode": "normalize",
                "recovery_success": True,
            },
        })
        failure = app._format_process_progress_event({
            "event_type": "task_stalled_recovery_attempted",
            "event_data": {
                "attempt": 2,
                "recovery_mode": "replan",
                "recovery_success": False,
                "reason": "strict_phase_mode_disallows_normalization",
            },
        })

        assert success == "Stall recovery via normalize succeeded (attempt 1)."
        assert (
            failure
            == "Stall recovery via replan failed (attempt 2): "
            "strict_phase_mode_disallows_normalization"
        )

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
