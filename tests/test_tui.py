"""Tests for the TUI app and its components."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

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
        assert screen._risk_info is None

    def test_init_with_risk_info(self):
        screen = ToolApprovalScreen(
            "shell_execute",
            "wp db reset --yes",
            risk_info={"risk_level": "high"},
        )
        assert screen._risk_info == {"risk_level": "high"}

    def test_on_key_approve(self):
        screen = ToolApprovalScreen("shell_execute", "ls -la")
        dismissed = []
        screen.dismiss = lambda value: dismissed.append(value)
        event = MagicMock()
        event.key = "y"

        screen.on_key(event)

        assert dismissed == ["approve"]
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()

    def test_on_key_approve_all(self):
        screen = ToolApprovalScreen("shell_execute", "ls -la")
        dismissed = []
        screen.dismiss = lambda value: dismissed.append(value)
        event = MagicMock()
        event.key = "a"

        screen.on_key(event)

        assert dismissed == ["approve_all"]
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()

    def test_on_key_deny_paths(self):
        screen = ToolApprovalScreen("shell_execute", "ls -la")
        dismissed = []
        screen.dismiss = lambda value: dismissed.append(value)

        for key in ("n", "escape", "ctrl+c", "ctrl+z"):
            event = MagicMock()
            event.key = key
            screen.on_key(event)
            assert dismissed[-1] == "deny"
            event.stop.assert_called_once()
            event.prevent_default.assert_called_once()

    @pytest.mark.asyncio
    async def test_modal_renders_markup_sensitive_preview_without_crashing(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()

        async with app.run_test() as pilot:
            app.push_screen(
                ToolApprovalScreen(
                    "run_tool[unsafe]",
                    'name=read_file args={"path":"note[1].md"}',
                    risk_info={
                        "risk_level": "high[1]",
                        "action_class": "destructive[action]",
                        "impact_preview": "touches [all] files",
                        "consequences": "possible [loss]",
                    },
                )
            )
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()


class TestApprovalCallback:
    @pytest.mark.asyncio
    async def test_approval_callback_events_are_not_clobbered_by_overlap(self):
        from loom.cowork.approval import ApprovalDecision
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        callbacks: list = []

        def _push_screen(_screen, callback):
            callbacks.append(callback)

        app.push_screen = _push_screen

        first = asyncio.create_task(app._approval_callback("shell_execute", {"command": "echo 1"}))
        await asyncio.sleep(0)
        second = asyncio.create_task(app._approval_callback("shell_execute", {"command": "echo 2"}))
        await asyncio.sleep(0)

        callbacks[0]("approve")
        result_first = await asyncio.wait_for(first, timeout=1)
        callbacks[1]("deny")
        result_second = await asyncio.wait_for(second, timeout=1)

        assert result_first == ApprovalDecision.APPROVE
        assert result_second == ApprovalDecision.DENY


class TestAskUserScreen:
    def test_init_no_options(self):
        screen = AskUserScreen("What language?")
        assert screen._question == "What language?"
        assert screen._options == []

    def test_init_with_options(self):
        screen = AskUserScreen("Pick one:", ["Python", "Rust"])
        assert screen._question == "Pick one:"
        assert screen._options == ["Python", "Rust"]

    def test_payload_single_choice_maps_numeric_selection(self):
        screen = AskUserScreen(
            "Pick one:",
            question_type="single_choice",
            option_items=[
                {"id": "python", "label": "Python", "description": ""},
                {"id": "rust", "label": "Rust", "description": ""},
            ],
            return_payload=True,
        )
        payload = screen._payload_answer("2")
        assert isinstance(payload, dict)
        assert payload["response_type"] == "single_choice"
        assert payload["selected_option_ids"] == ["rust"]
        assert payload["selected_labels"] == ["Rust"]

    def test_payload_multi_choice_enforces_min_max(self):
        screen = AskUserScreen(
            "Pick two:",
            question_type="multi_choice",
            option_items=[
                {"id": "a", "label": "A", "description": ""},
                {"id": "b", "label": "B", "description": ""},
                {"id": "c", "label": "C", "description": ""},
            ],
            min_selections=2,
            max_selections=2,
            return_payload=True,
        )
        assert screen._payload_answer("1") is None
        payload = screen._payload_answer("1,2")
        assert isinstance(payload, dict)
        assert payload["response_type"] == "multi_choice"
        assert payload["selected_option_ids"] == ["a", "b"]

    def test_input_hidden_when_options_no_custom_response(self):
        screen = AskUserScreen(
            "Pick one:",
            question_type="single_choice",
            option_items=[
                {"id": "python", "label": "Python", "description": ""},
                {"id": "rust", "label": "Rust", "description": ""},
            ],
            allow_custom_response=False,
            return_payload=True,
        )
        assert screen._show_input is False

    def test_input_shown_when_options_allow_custom_response(self):
        screen = AskUserScreen(
            "Pick one:",
            question_type="single_choice",
            option_items=[
                {"id": "python", "label": "Python", "description": ""},
                {"id": "rust", "label": "Rust", "description": ""},
            ],
            allow_custom_response=True,
            return_payload=True,
        )
        assert screen._show_input is True

    def test_payload_from_selected_options_single_choice(self):
        screen = AskUserScreen(
            "Pick one:",
            question_type="single_choice",
            option_items=[
                {"id": "python", "label": "Python", "description": ""},
                {"id": "rust", "label": "Rust", "description": ""},
            ],
            allow_custom_response=False,
            return_payload=True,
        )
        screen._selected_option_ids = ["rust"]
        payload = screen._payload_from_selected_options()
        assert isinstance(payload, dict)
        assert payload["response_type"] == "single_choice"
        assert payload["selected_option_ids"] == ["rust"]
        assert payload["selected_labels"] == ["Rust"]

    def test_payload_from_selected_options_multi_choice(self):
        screen = AskUserScreen(
            "Pick two:",
            question_type="multi_choice",
            option_items=[
                {"id": "a", "label": "A", "description": ""},
                {"id": "b", "label": "B", "description": ""},
                {"id": "c", "label": "C", "description": ""},
            ],
            min_selections=2,
            max_selections=2,
            allow_custom_response=False,
            return_payload=True,
        )
        screen._selected_option_ids = ["a", "b"]
        payload = screen._payload_from_selected_options()
        assert isinstance(payload, dict)
        assert payload["response_type"] == "multi_choice"
        assert payload["selected_option_ids"] == ["a", "b"]


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


class TestAuthAndMCPManagerScreens:
    def test_auth_manager_mode_encoding_round_trip(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        assert AuthManagerScreen._encode_mode_value("") == "__mode_unset__"
        assert AuthManagerScreen._encode_mode_value(" OAUTH2_PKCE ") == "oauth2_pkce"
        assert AuthManagerScreen._decode_mode_value("__mode_unset__") == ""
        assert AuthManagerScreen._decode_mode_value("api_key") == "api_key"

    def test_auth_manager_ctrl_w_binding_registered(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        bindings = {binding.key: binding for binding in AuthManagerScreen.BINDINGS}
        assert bindings["ctrl+w"].action == "request_close"
        assert bindings["ctrl+w"].priority is True

    def test_auth_manager_on_key_ctrl_w_closes(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen.action_request_close = AsyncMock()
        captured: dict[str, object] = {}
        screen.run_worker = MagicMock(
            side_effect=lambda coro, **kwargs: captured.update(coro=coro, kwargs=kwargs)
        )
        event = MagicMock()
        event.key = "ctrl+w"

        screen.on_key(event)

        screen.action_request_close.assert_called_once()
        assert captured["kwargs"]["group"] == "auth-manager-close-request"
        assert captured["kwargs"]["exclusive"] is True
        captured["coro"].close()
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()

    def test_auth_manager_mode_options_include_supported_modes(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))

        options = screen._mode_options()
        values = [value for _label, value in options]

        assert "__mode_unset__" in values
        assert "api_key" in values
        assert "oauth2_pkce" in values
        assert "oauth2_device" in values
        assert "cli_passthrough" in values
        assert "env_passthrough" in values

    def test_auth_manager_summary_css_has_border(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        css = AuthManagerScreen.DEFAULT_CSS
        block = css.split("#auth-manager-summary {", 1)[1].split("}", 1)[0]
        assert "border: round $surface-lighten-1;" in block

    def test_auth_manager_oauth_settings_css_uses_auto_height(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        css = AuthManagerScreen.DEFAULT_CSS
        block = css.split("#auth-oauth-settings {", 1)[1].split("}", 1)[0]
        assert "height: auto;" in block
        assert "layout: vertical;" in block
        shared_block = css.split("#auth-secret-ref-section,", 1)[1].split("}", 1)[0]
        assert "#auth-token-ref-section" in shared_block
        assert "#auth-scopes-section" in shared_block
        assert "#auth-env-section" in shared_block
        assert "#auth-command-section" in shared_block

    @pytest.mark.asyncio
    async def test_auth_manager_refresh_summary_context_is_single_line(self, monkeypatch):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        merged = SimpleNamespace(
            config=SimpleNamespace(
                profiles={
                    "draft_notion": SimpleNamespace(
                        profile_id="draft_notion",
                    ),
                },
            ),
            workspace_defaults={},
            user_path=Path("/tmp/auth.toml"),
            explicit_path=Path("/tmp/auth.override.toml"),
            workspace_defaults_path=Path("/tmp/auth.defaults.toml"),
        )
        resources_store = SimpleNamespace(resources={}, workspace_defaults={})
        monkeypatch.setattr(
            "loom.tui.screens.auth_manager.load_merged_auth_config",
            lambda workspace, explicit_path=None: merged,
        )
        monkeypatch.setattr(
            "loom.tui.screens.auth_manager.default_workspace_auth_resources_path",
            lambda workspace: Path("/tmp/auth.resources.toml"),
        )
        monkeypatch.setattr(
            "loom.tui.screens.auth_manager.load_workspace_auth_resources",
            lambda path: resources_store,
        )
        monkeypatch.setattr(
            "loom.tui.screens.auth_manager.profile_bindings_map",
            lambda store: {},
        )

        screen = AuthManagerScreen(
            workspace=Path("/tmp"),
            mcp_manager=SimpleNamespace(
                list_views=lambda: [SimpleNamespace(alias="notion"), SimpleNamespace(alias="")],
            ),
            process_defs=[SimpleNamespace(name="research"), SimpleNamespace(name="")],
        )
        context_widget = SimpleNamespace(update=MagicMock())

        def _query_one(selector, _cls=None):
            assert selector == "#auth-manager-context"
            return context_widget

        screen.query_one = _query_one
        screen._set_mcp_server_select_value = lambda alias: None
        screen._render_summary = MagicMock()
        screen.notify = MagicMock()

        await screen._refresh_summary()

        context = context_widget.update.call_args.args[0]
        assert context == (
            "profiles=1 source=explicit discovery=1 process(es), 1 mcp alias(es)"
        )
        assert "\n" not in context

    def test_auth_manager_mcp_target_encoding_round_trip(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        assert AuthManagerScreen._encode_mcp_server_value("") == "__none__"
        assert AuthManagerScreen._encode_mcp_server_value(" notion ") == "notion"
        assert AuthManagerScreen._decode_mcp_server_value("__none__") == ""
        assert AuthManagerScreen._decode_mcp_server_value("notion") == "notion"

    def test_auth_manager_mcp_target_options_include_none_sorted_aliases(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._mcp_aliases = ["prod", "dev"]

        options = screen._mcp_target_options()

        assert options == [
            ("None (provider-wide)", "__none__"),
            ("MCP: dev", "dev"),
            ("MCP: prod", "prod"),
        ]

    def test_auth_manager_mcp_target_options_include_active_alias(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._mcp_aliases = ["dev"]

        options = screen._mcp_target_options(include_alias="staging")

        assert ("MCP: dev", "dev") in options
        assert ("MCP: staging", "staging") in options

    def test_auth_manager_workspace_default_detection(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._workspace_defaults = {"notion": "notion_dev"}

        assert screen._is_workspace_default(
            provider="notion",
            profile_id="notion_dev",
        )
        assert not screen._is_workspace_default(
            provider="notion",
            profile_id="notion_prod",
        )

    def test_auth_manager_discovery_scope_text_without_loaded_processes(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))

        assert "no process contracts loaded" in screen._discovery_scope_text()

    def test_auth_manager_discovery_scope_text_with_workspace_processes(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._discovery_process_names = ["youtube-draft-descriptions", "research-report"]

        text = screen._discovery_scope_text()
        assert "all workspace processes (2)" in text

    def test_auth_manager_resource_id_for_profile_requires_binding(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._resource_binding_by_profile = {}
        profile = SimpleNamespace(profile_id="draft_mcp_notion", mcp_server="notion")

        assert screen._resource_id_for_profile(profile) == ""

    def test_auth_manager_render_summary_populates_rows(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        table = MagicMock()
        table.move_cursor = MagicMock()
        screen.query_one = MagicMock(return_value=table)
        screen._profiles = {
            "notion_dev": SimpleNamespace(
                profile_id="notion_dev",
                provider="notion",
                mode="oauth2_pkce",
                mcp_server="",
                account_label="Dev",
            ),
        }
        screen._active_profile_id = "notion_dev"

        screen._render_summary()

        table.clear.assert_called_once()
        table.add_row.assert_called_once()
        table.move_cursor.assert_called_once()

    def test_auth_manager_render_summary_marks_unbound_mcp_profile(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        table = MagicMock()
        table.move_cursor = MagicMock()
        screen.query_one = MagicMock(return_value=table)
        screen._profiles = {
            "draft_mcp_notion": SimpleNamespace(
                profile_id="draft_mcp_notion",
                provider="notion",
                mode="api_key",
                mcp_server="notion",
                account_label="MCP: notion (Draft)",
            ),
        }
        screen._resource_binding_by_profile = {}
        screen._resources_by_id = {}
        screen._active_profile_id = "draft_mcp_notion"

        screen._render_summary()

        table.add_row.assert_called_once()
        assert table.add_row.call_args.args[1] == "Unbound (mcp:notion)"

    def test_auth_manager_set_form_values_sets_resource_before_mode(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        order: list[str] = []
        profile_input = SimpleNamespace(value="")
        default_checkbox = SimpleNamespace(value=False)
        label_input = SimpleNamespace(value="")
        secret_input = SimpleNamespace(value="")
        token_input = SimpleNamespace(value="")
        scopes_input = SimpleNamespace(value="")
        env_input = SimpleNamespace(value="")
        command_input = SimpleNamespace(value="")
        auth_check_input = SimpleNamespace(value="")
        oauth_authorize_input = SimpleNamespace(value="")
        oauth_token_input = SimpleNamespace(value="")
        oauth_client_id_input = SimpleNamespace(value="")
        oauth_client_secret_input = SimpleNamespace(value="")
        oauth_scope_input = SimpleNamespace(value="")
        meta_input = SimpleNamespace(value="")
        oauth_settings = SimpleNamespace(display=False)
        provider_static = SimpleNamespace(update=lambda _value: None)

        def _query_one(selector, _cls=None):
            mapping = {
                "#auth-profile-id": profile_input,
                "#auth-default-provider": default_checkbox,
                "#auth-label": label_input,
                "#auth-secret-ref": secret_input,
                "#auth-token-ref": token_input,
                "#auth-scopes": scopes_input,
                "#auth-env": env_input,
                "#auth-command": command_input,
                "#auth-auth-check": auth_check_input,
                "#auth-oauth-authorize-url": oauth_authorize_input,
                "#auth-oauth-token-url": oauth_token_input,
                "#auth-oauth-client-id": oauth_client_id_input,
                "#auth-oauth-client-secret": oauth_client_secret_input,
                "#auth-oauth-scope": oauth_scope_input,
                "#auth-meta": meta_input,
                "#auth-oauth-settings": oauth_settings,
                "#auth-provider-derived": provider_static,
            }
            return mapping[selector]

        screen.query_one = _query_one
        screen._set_mcp_server_select_value = lambda _alias: order.append("resource")
        screen._set_mode_select_value = lambda _mode: order.append("mode")
        screen._refresh_oauth_settings_visibility = MagicMock()

        screen._set_form_values(
            profile_id="draft_api_integration_youtube_data_api",
            provider="youtube_data_api",
            mode="oauth2_pkce",
            set_default=False,
            label="API: youtube_data_api (Draft)",
            resource_id="res-youtube",
            secret_ref="",
            token_ref="env://TODO_YOUTUBE_DATA_API_TOKEN",
            scopes="",
            env="",
            command="",
            auth_check="",
            metadata="generated=true",
        )

        assert order == ["resource", "mode"]

    def test_auth_manager_default_oauth_token_ref_uses_keychain_template(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))

        token_ref = screen._default_oauth_token_ref(
            provider="YouTube Data API",
            key_name="Primary Account",
        )

        assert token_ref == "keychain://loom/youtube_data_api/primary_account/tokens"

    def test_auth_manager_oauth_settings_visibility_tracks_mode(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        oauth_settings = SimpleNamespace(display=False)
        secret_section = SimpleNamespace(display=True)
        token_section = SimpleNamespace(display=False)
        scopes_section = SimpleNamespace(display=False)
        env_section = SimpleNamespace(display=False)
        command_section = SimpleNamespace(display=False)

        def _query_one(selector, _cls=None):
            mapping = {
                "#auth-oauth-settings": oauth_settings,
                "#auth-secret-ref-section": secret_section,
                "#auth-token-ref-section": token_section,
                "#auth-scopes-section": scopes_section,
                "#auth-env-section": env_section,
                "#auth-command-section": command_section,
            }
            return mapping[selector]

        screen.query_one = _query_one

        screen._selected_mode = lambda: "oauth2_pkce"
        screen._refresh_oauth_settings_visibility()
        assert oauth_settings.display is True
        assert secret_section.display is False
        assert token_section.display is True
        assert scopes_section.display is True
        assert env_section.display is False
        assert command_section.display is False

        screen._selected_mode = lambda: "api_key"
        screen._refresh_oauth_settings_visibility()
        assert oauth_settings.display is False
        assert secret_section.display is True
        assert token_section.display is False
        assert scopes_section.display is False
        assert env_section.display is True
        assert command_section.display is False

        screen._selected_mode = lambda: "env_passthrough"
        screen._refresh_oauth_settings_visibility()
        assert oauth_settings.display is False
        assert secret_section.display is False
        assert token_section.display is False
        assert scopes_section.display is False
        assert env_section.display is True
        assert command_section.display is False

        screen._selected_mode = lambda: "cli_passthrough"
        screen._refresh_oauth_settings_visibility()
        assert oauth_settings.display is False
        assert secret_section.display is False
        assert token_section.display is False
        assert scopes_section.display is False
        assert env_section.display is False
        assert command_section.display is True

    def test_auth_manager_maybe_set_default_oauth_token_ref_only_when_empty(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        token_input = SimpleNamespace(value="")

        def _query_one(selector, _cls=None):
            assert selector == "#auth-token-ref"
            return token_input

        screen.query_one = _query_one
        screen._selected_mode = lambda: "oauth2_pkce"
        screen._default_oauth_token_ref = lambda provider="", key_name="": (
            f"keychain://loom/{provider}/{key_name}/tokens"
        )

        screen._maybe_set_default_oauth_token_ref(
            provider="youtube_data_api",
            key_name="api_integration_youtube_data_api",
        )
        assert (
            token_input.value
            == "keychain://loom/youtube_data_api/api_integration_youtube_data_api/tokens"
        )

        token_input.value = "keychain://loom/custom/path/tokens"
        screen._maybe_set_default_oauth_token_ref(
            provider="youtube_data_api",
            key_name="new_profile",
        )
        assert token_input.value == "keychain://loom/custom/path/tokens"

    def test_auth_manager_missing_required_oauth_metadata(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        missing = AuthManagerScreen._missing_required_oauth_metadata(
            {
                "oauth_client_id": "client-123",
            }
        )

        assert missing == ("OAuth Authorization URL", "OAuth Token URL")

    @pytest.mark.asyncio
    async def test_auth_manager_save_profile_requires_oauth_metadata(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._profiles = {}
        screen._workspace_defaults = {}
        screen._workspace_resource_defaults = {}
        screen._resources_by_id = {
            "res-youtube": SimpleNamespace(
                provider="youtube_data_api",
                resource_kind="api_integration",
                resource_key="youtube_data_api",
            ),
        }
        screen._selected_resource_id = lambda: "res-youtube"
        screen._selected_mode = lambda: "oauth2_pkce"
        screen._profile_id = lambda: "youtube_profile"
        screen._default_provider_selected = lambda: False
        screen.notify = MagicMock()

        fields = {
            "#auth-label": SimpleNamespace(value="API: youtube_data_api"),
            "#auth-secret-ref": SimpleNamespace(value=""),
            "#auth-token-ref": SimpleNamespace(value=""),
            "#auth-scopes": SimpleNamespace(value=""),
            "#auth-env": SimpleNamespace(value=""),
            "#auth-command": SimpleNamespace(value=""),
            "#auth-auth-check": SimpleNamespace(value=""),
            "#auth-meta": SimpleNamespace(value=""),
            "#auth-oauth-authorize-url": SimpleNamespace(value=""),
            "#auth-oauth-token-url": SimpleNamespace(value=""),
            "#auth-oauth-client-id": SimpleNamespace(value=""),
            "#auth-oauth-client-secret": SimpleNamespace(value=""),
            "#auth-oauth-scope": SimpleNamespace(value=""),
        }
        screen.query_one = lambda selector, _cls=None: fields[selector]

        saved = await screen._save_profile()

        assert saved is False
        assert any(
            "OAuth mode requires:" in str(call.args[0]) for call in screen.notify.call_args_list
        )

    @pytest.mark.asyncio
    async def test_auth_manager_save_profile_requires_keychain_token_ref_for_oauth(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._profiles = {}
        screen._workspace_defaults = {}
        screen._workspace_resource_defaults = {}
        screen._resources_by_id = {
            "res-youtube": SimpleNamespace(
                provider="youtube_data_api",
                resource_kind="api_integration",
                resource_key="youtube_data_api",
            ),
        }
        screen._selected_resource_id = lambda: "res-youtube"
        screen._selected_mode = lambda: "oauth2_pkce"
        screen._profile_id = lambda: "youtube_profile"
        screen._default_provider_selected = lambda: False
        screen.notify = MagicMock()

        fields = {
            "#auth-label": SimpleNamespace(value="API: youtube_data_api"),
            "#auth-secret-ref": SimpleNamespace(value=""),
            "#auth-token-ref": SimpleNamespace(value="env://YOUTUBE_TOKEN"),
            "#auth-scopes": SimpleNamespace(value=""),
            "#auth-env": SimpleNamespace(value=""),
            "#auth-command": SimpleNamespace(value=""),
            "#auth-auth-check": SimpleNamespace(value=""),
            "#auth-meta": SimpleNamespace(value=""),
            "#auth-oauth-authorize-url": SimpleNamespace(
                value="https://accounts.google.com/o/oauth2/v2/auth"
            ),
            "#auth-oauth-token-url": SimpleNamespace(value="https://oauth2.googleapis.com/token"),
            "#auth-oauth-client-id": SimpleNamespace(value="client-123"),
            "#auth-oauth-client-secret": SimpleNamespace(value=""),
            "#auth-oauth-scope": SimpleNamespace(value=""),
        }
        screen.query_one = lambda selector, _cls=None: fields[selector]

        saved = await screen._save_profile()

        assert saved is False
        assert any(
            "OAuth token_ref must use keychain://... storage." in str(call.args[0])
            for call in screen.notify.call_args_list
        )

    def test_auth_manager_ignores_programmatic_select_changes_when_suppressed(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._suppress_dirty_tracking = True
        screen._sync_provider_display = MagicMock()
        screen._refresh_mode_select = MagicMock()
        screen._update_form_dirty = MagicMock()

        event = SimpleNamespace(select=SimpleNamespace(id="auth-resource-target"))
        screen._on_form_select_changed(event)

        screen._sync_provider_display.assert_not_called()
        screen._refresh_mode_select.assert_not_called()
        screen._update_form_dirty.assert_not_called()

    @pytest.mark.asyncio
    async def test_auth_manager_oauth_buttons_dispatch_to_oauth_actions(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._oauth_login = AsyncMock()
        screen._oauth_status = AsyncMock()
        screen._oauth_logout = AsyncMock()
        screen._oauth_refresh = AsyncMock()

        await screen.on_button_pressed(
            SimpleNamespace(button=SimpleNamespace(id="auth-btn-oauth-login"))
        )
        await screen.on_button_pressed(
            SimpleNamespace(button=SimpleNamespace(id="auth-btn-oauth-status"))
        )
        await screen.on_button_pressed(
            SimpleNamespace(button=SimpleNamespace(id="auth-btn-oauth-logout"))
        )
        await screen.on_button_pressed(
            SimpleNamespace(button=SimpleNamespace(id="auth-btn-oauth-refresh"))
        )

        screen._oauth_login.assert_awaited_once()
        screen._oauth_status.assert_awaited_once()
        screen._oauth_logout.assert_awaited_once()
        screen._oauth_refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auth_manager_oauth_login_passes_callback_prompt(self, monkeypatch):
        from loom.auth.oauth_profiles import OAuthProfileError
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._oauth_target_profile = AsyncMock(return_value=SimpleNamespace(profile_id="youtube"))
        screen.notify = MagicMock()

        observed: dict[str, object] = {}

        def _fake_login(profile, **kwargs):
            observed["profile"] = profile
            observed["callback_prompt"] = kwargs.get("callback_prompt")
            raise OAuthProfileError(
                "callback_missing",
                "Callback URL/code required for manual OAuth completion.",
            )

        monkeypatch.setattr("loom.tui.screens.auth_manager.login_oauth_profile", _fake_login)

        await screen._oauth_login()

        assert observed["profile"].profile_id == "youtube"
        assert callable(observed["callback_prompt"])
        assert any(
            "OAuth callback input was not provided; login canceled." in str(call.args[0])
            for call in screen.notify.call_args_list
        )

    @pytest.mark.asyncio
    async def test_auth_manager_oauth_target_profile_autosaves_dirty_form(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._form_dirty = True
        screen._save_profile = AsyncMock(return_value=True)
        screen._profiles = {
            "youtube": SimpleNamespace(profile_id="youtube", mode="oauth2_pkce")
        }
        screen._active_profile_id = "youtube"
        screen._profile_id = lambda: "youtube"
        screen.notify = MagicMock()

        profile = await screen._oauth_target_profile()

        assert profile is not None
        assert profile.profile_id == "youtube"
        screen._save_profile.assert_awaited_once_with(notify_success=False)

    @pytest.mark.asyncio
    async def test_auth_manager_load_profile_keeps_mode_selected_after_events(self):
        from textual.app import App
        from textual.containers import Vertical
        from textual.widgets import Input, Select

        from loom.tui.screens.auth_manager import AuthManagerScreen

        class _DummyMCPManager:
            def list_views(self):
                return []

        class _AuthManagerHarnessApp(App):
            CSS_PATH = None

            def __init__(self):
                super().__init__()
                self.screen_under_test = AuthManagerScreen(
                    workspace=Path("/tmp"),
                    mcp_manager=_DummyMCPManager(),
                    process_defs=[],
                )

            def compose(self):
                yield self.screen_under_test

        app = _AuthManagerHarnessApp()
        screen = app.screen_under_test
        screen._sync_missing_drafts = AsyncMock()
        screen._refresh_summary = AsyncMock()
        screen.notify = MagicMock()

        resource_id = "res-youtube"
        profile_id = "draft_api_integration_youtube_data_api"
        profile = SimpleNamespace(
            profile_id=profile_id,
            provider="youtube_data_api",
            mode="oauth2_pkce",
            account_label="API: youtube_data_api (Draft)",
            secret_ref="",
            token_ref="env://TODO_YOUTUBE_DATA_API_TOKEN",
            scopes=[],
            env={},
            command="",
            auth_check=[],
            metadata={
                "generated": "true",
                "oauth_authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
                "oauth_token_endpoint": "https://oauth2.googleapis.com/token",
                "oauth_client_id": "client-123",
                "oauth_scope": "https://www.googleapis.com/auth/youtube.readonly",
            },
        )
        screen._profiles = {profile_id: profile}
        screen._resources_by_id = {
            resource_id: SimpleNamespace(
                display_name="API: youtube_data_api",
                provider="youtube_data_api",
                resource_kind="api_integration",
                resource_key="youtube_data_api",
                modes=("oauth2_pkce", "oauth2_device", "env_passthrough"),
            ),
        }
        screen._resource_binding_by_profile = {profile_id: resource_id}
        screen._workspace_defaults = {}
        screen._workspace_resource_defaults = {}

        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            loaded = await screen._load_profile_into_form(profile_id=profile_id)
            await pilot.pause()

            mode_select = screen.query_one("#auth-mode", Select)
            oauth_authorize = screen.query_one("#auth-oauth-authorize-url", Input)
            oauth_token = screen.query_one("#auth-oauth-token-url", Input)
            oauth_client_id = screen.query_one("#auth-oauth-client-id", Input)
            meta_input = screen.query_one("#auth-meta", Input)
            oauth_settings = screen.query_one("#auth-oauth-settings", Vertical)
            assert loaded is True
            assert mode_select.value == "oauth2_pkce"
            assert screen._selected_mode() == "oauth2_pkce"
            assert oauth_authorize.value == "https://accounts.google.com/o/oauth2/v2/auth"
            assert oauth_token.value == "https://oauth2.googleapis.com/token"
            assert oauth_client_id.value == "client-123"
            assert meta_input.value == "generated=true"
            assert oauth_settings.display is True

    @pytest.mark.asyncio
    async def test_auth_manager_on_mount_does_not_sync_implicitly(self):
        from textual.app import App

        from loom.tui.screens.auth_manager import AuthManagerScreen

        class _DummyMCPManager:
            def list_views(self):
                return []

        class _AuthManagerHarnessApp(App):
            CSS_PATH = None

            def __init__(self):
                super().__init__()
                self.screen_under_test = AuthManagerScreen(
                    workspace=Path("/tmp"),
                    mcp_manager=_DummyMCPManager(),
                    process_defs=[],
                )

            def compose(self):
                yield self.screen_under_test

        app = _AuthManagerHarnessApp()
        screen = app.screen_under_test
        screen._sync_missing_drafts = AsyncMock()
        screen._refresh_summary = AsyncMock()

        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()

        screen._sync_missing_drafts.assert_not_awaited()
        screen._refresh_summary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auth_manager_refresh_and_sync_controls_have_distinct_behavior(self):
        from textual.app import App

        from loom.tui.screens.auth_manager import AuthManagerScreen

        class _DummyMCPManager:
            def list_views(self):
                return []

        class _AuthManagerHarnessApp(App):
            CSS_PATH = None

            def __init__(self):
                super().__init__()
                self.screen_under_test = AuthManagerScreen(
                    workspace=Path("/tmp"),
                    mcp_manager=_DummyMCPManager(),
                    process_defs=[],
                )

            def compose(self):
                yield self.screen_under_test

        app = _AuthManagerHarnessApp()
        screen = app.screen_under_test
        screen._sync_missing_drafts = AsyncMock()
        screen._refresh_summary = AsyncMock()

        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            screen._sync_missing_drafts.reset_mock()
            screen._refresh_summary.reset_mock()

            await screen.action_refresh()
            assert screen._sync_missing_drafts.await_count == 0
            assert screen._refresh_summary.await_count == 1

            screen._sync_missing_drafts.reset_mock()
            screen._refresh_summary.reset_mock()
            await pilot.click("#auth-btn-refresh")
            await pilot.pause()
            assert screen._sync_missing_drafts.await_count == 0
            assert screen._refresh_summary.await_count == 1

            screen._sync_missing_drafts.reset_mock()
            screen._refresh_summary.reset_mock()
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert screen._sync_missing_drafts.await_count == 0
            assert screen._refresh_summary.await_count == 1

            screen._sync_missing_drafts.reset_mock()
            screen._refresh_summary.reset_mock()
            await pilot.click("#auth-btn-sync")
            await pilot.pause()
            assert screen._sync_missing_drafts.await_count == 1
            assert screen._refresh_summary.await_count == 1

    def test_auth_manager_next_duplicate_profile_id(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._profiles = {
            "notion_dev": SimpleNamespace(),
            "notion_dev_copy": SimpleNamespace(),
        }
        assert screen._next_duplicate_profile_id("notion_dev") == "notion_dev_copy2"

    def test_mcp_manager_render_summary_populates_rows(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        table = MagicMock()
        table.move_cursor = MagicMock()
        screen.query_one = MagicMock(return_value=table)
        screen._views = [
            SimpleNamespace(
                alias="notion",
                server=SimpleNamespace(
                    enabled=True,
                    command="python",
                    args=["-m", "notion_server"],
                ),
                source="workspace",
            ),
        ]
        screen._active_alias = "notion"

        screen._render_summary()

        table.clear.assert_called_once()
        table.add_row.assert_called_once()
        table.move_cursor.assert_called_once()
        added = table.add_row.call_args.args
        assert added[0] == "notion"
        assert added[1] == "local"
        assert "python" in added[3]

    def test_mcp_manager_ctrl_w_binding_registered(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        bindings = {binding.key: binding for binding in MCPManagerScreen.BINDINGS}
        assert bindings["ctrl+w"].action == "request_close"
        assert bindings["ctrl+w"].priority is True

    def test_mcp_manager_on_key_ctrl_w_closes(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        screen.action_request_close = AsyncMock()
        captured: dict[str, object] = {}
        screen.run_worker = MagicMock(
            side_effect=lambda coro, **kwargs: captured.update(coro=coro, kwargs=kwargs)
        )
        event = MagicMock()
        event.key = "ctrl+w"

        screen.on_key(event)

        screen.action_request_close.assert_called_once()
        assert captured["kwargs"]["group"] == "mcp-manager-close-request"
        assert captured["kwargs"]["exclusive"] is True
        captured["coro"].close()
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()

    def test_mcp_manager_summary_css_has_border(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        css = MCPManagerScreen.DEFAULT_CSS
        block = css.split("#mcp-manager-summary {", 1)[1].split("}", 1)[0]
        assert "border: round $surface-lighten-1;" in block

    @pytest.mark.asyncio
    async def test_mcp_manager_actions_render_above_summary(self):
        from textual.app import App
        from textual.containers import Horizontal, Vertical

        from loom.tui.screens.mcp_manager import MCPManagerScreen

        class _Manager:
            def list_views(self):
                return []

        class _Harness(App):
            CSS_PATH = None

            def __init__(self):
                super().__init__()
                self.screen_under_test = MCPManagerScreen(manager=_Manager())

            def compose(self):
                yield self.screen_under_test

        app = _Harness()
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            dialog = app.screen_under_test.query_one("#mcp-manager-dialog", Vertical)
            ordered_ids = [child.id for child in dialog.children if child.id]
            assert ordered_ids.index("mcp-actions-primary") < ordered_ids.index(
                "mcp-manager-summary"
            )
            primary_actions = app.screen_under_test.query_one(
                "#mcp-actions-primary",
                Horizontal,
            )
            action_ids = [child.id for child in primary_actions.children if child.id]
            assert "mcp-btn-new" in action_ids

    def test_mcp_manager_render_summary_remote_target(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        table = MagicMock()
        table.move_cursor = MagicMock()
        screen.query_one = MagicMock(return_value=table)
        screen._views = [
            SimpleNamespace(
                alias="notion_remote",
                server=SimpleNamespace(
                    enabled=True,
                    type="remote",
                    url="https://mcp.notion.com/mcp",
                ),
                source="workspace",
            ),
        ]

        screen._render_summary()

        assert table.add_row.call_count == 1
        added = table.add_row.call_args.args
        assert added[0] == "notion_remote"
        assert added[1] == "remote"
        assert added[3] == "https://mcp.notion.com/mcp"

    def test_mcp_manager_parse_csv_map(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        parsed = MCPManagerScreen._parse_csv_map(
            "X-Team=alpha, Authorization=Bearer ${TOKEN}",
            option_name="Headers",
        )
        assert parsed == {
            "X-Team": "alpha",
            "Authorization": "Bearer ${TOKEN}",
        }

        with pytest.raises(ValueError, match="KEY=VALUE"):
            MCPManagerScreen._parse_csv_map("invalid", option_name="Headers")

    def test_mcp_manager_stores_explicit_auth_path(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        auth_path = Path("/tmp/auth.toml")
        screen = MCPManagerScreen(
            manager=MagicMock(),
            explicit_auth_path=auth_path,
        )
        assert screen._explicit_auth_path == auth_path

    def test_mcp_manager_sync_transport_fields_toggles_oauth_buttons(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        widgets: dict[str, object] = {}

        class _Widget:
            def __init__(self, value: object = "") -> None:
                self.disabled = False
                self.value = value
                self.display = True

            def update(self, _value: str) -> None:
                return

        for field_id in (
            "mcp-label-command",
            "mcp-help-command",
            "mcp-label-args",
            "mcp-help-args",
            "mcp-label-cwd",
            "mcp-help-cwd",
            "mcp-label-env",
            "mcp-help-env",
            "mcp-label-env-ref",
            "mcp-help-env-ref",
            "mcp-label-url",
            "mcp-help-url",
            "mcp-label-fallback-sse-url",
            "mcp-help-fallback-sse-url",
            "mcp-manager-remote-advanced",
            "mcp-actions-oauth",
            "mcp-command",
            "mcp-args",
            "mcp-cwd",
            "mcp-env",
            "mcp-env-ref",
            "mcp-url",
            "mcp-fallback-sse-url",
            "mcp-headers",
            "mcp-oauth-scopes",
            "mcp-oauth-access-token",
            "mcp-oauth-refresh-token",
            "mcp-oauth-expires-in",
            "mcp-oauth-enabled",
            "mcp-allow-insecure-http",
            "mcp-allow-private-network",
            "mcp-btn-oauth-login",
            "mcp-btn-oauth-copy-url",
            "mcp-btn-oauth-enter-code",
            "mcp-btn-oauth-logout",
            "mcp-btn-oauth-save",
            "mcp-oauth-status",
        ):
            widgets[f"#{field_id}"] = _Widget()

        screen.query_one = MagicMock(side_effect=lambda selector, *_args: widgets[selector])
        screen._selected_server_type = MagicMock(return_value=screen._TYPE_LOCAL_VALUE)
        screen._sync_transport_fields()
        assert widgets["#mcp-label-command"].display is True
        assert widgets["#mcp-help-command"].display is True
        assert widgets["#mcp-label-url"].display is False
        assert widgets["#mcp-help-url"].display is False
        assert widgets["#mcp-actions-oauth"].display is False
        assert widgets["#mcp-btn-oauth-login"].disabled is True
        assert widgets["#mcp-btn-oauth-copy-url"].disabled is True
        assert widgets["#mcp-btn-oauth-enter-code"].disabled is True
        assert widgets["#mcp-btn-oauth-logout"].disabled is True
        assert widgets["#mcp-btn-oauth-save"].disabled is True

        screen._selected_server_type = MagicMock(return_value=screen._TYPE_REMOTE_VALUE)
        screen._sync_transport_fields()
        assert widgets["#mcp-label-command"].display is False
        assert widgets["#mcp-help-command"].display is False
        assert widgets["#mcp-label-url"].display is True
        assert widgets["#mcp-help-url"].display is True
        assert widgets["#mcp-actions-oauth"].display is True
        assert widgets["#mcp-btn-oauth-login"].disabled is False
        assert widgets["#mcp-btn-oauth-copy-url"].disabled is False
        assert widgets["#mcp-btn-oauth-enter-code"].disabled is False
        assert widgets["#mcp-btn-oauth-logout"].disabled is False
        assert widgets["#mcp-btn-oauth-save"].disabled is False

    def test_mcp_manager_sync_transport_fields_hides_browser_actions_when_disabled(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(
            manager=MagicMock(),
            oauth_browser_login_enabled=False,
        )
        widgets: dict[str, object] = {}

        class _Widget:
            def __init__(self, value: object = "") -> None:
                self.disabled = False
                self.value = value
                self.display = True

            def update(self, _value: str) -> None:
                return

        for field_id in (
            "mcp-label-command",
            "mcp-help-command",
            "mcp-label-args",
            "mcp-help-args",
            "mcp-label-cwd",
            "mcp-help-cwd",
            "mcp-label-env",
            "mcp-help-env",
            "mcp-label-env-ref",
            "mcp-help-env-ref",
            "mcp-label-url",
            "mcp-help-url",
            "mcp-label-fallback-sse-url",
            "mcp-help-fallback-sse-url",
            "mcp-manager-remote-advanced",
            "mcp-actions-oauth",
            "mcp-command",
            "mcp-args",
            "mcp-cwd",
            "mcp-env",
            "mcp-env-ref",
            "mcp-url",
            "mcp-fallback-sse-url",
            "mcp-headers",
            "mcp-oauth-scopes",
            "mcp-oauth-access-token",
            "mcp-oauth-refresh-token",
            "mcp-oauth-expires-in",
            "mcp-oauth-enabled",
            "mcp-allow-insecure-http",
            "mcp-allow-private-network",
            "mcp-btn-oauth-login",
            "mcp-btn-oauth-copy-url",
            "mcp-btn-oauth-enter-code",
            "mcp-btn-oauth-logout",
            "mcp-btn-oauth-save",
            "mcp-oauth-status",
        ):
            widgets[f"#{field_id}"] = _Widget()

        status_text = {"value": ""}

        def _set_status(value: str):
            status_text["value"] = value

        widgets["#mcp-oauth-status"].update = _set_status

        screen.query_one = MagicMock(side_effect=lambda selector, *_args: widgets[selector])
        screen._selected_server_type = MagicMock(return_value=screen._TYPE_REMOTE_VALUE)
        screen._sync_transport_fields()

        assert widgets["#mcp-label-command"].display is False
        assert widgets["#mcp-label-url"].display is True
        assert widgets["#mcp-actions-oauth"].display is True
        assert widgets["#mcp-btn-oauth-login"].disabled is True
        assert widgets["#mcp-btn-oauth-copy-url"].disabled is True
        assert widgets["#mcp-btn-oauth-enter-code"].disabled is True
        assert widgets["#mcp-btn-oauth-save"].disabled is False
        assert "disabled by config" in status_text["value"]

    def test_mcp_manager_select_remote_defaults_oauth_enabled_for_new_alias(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        screen._active_alias = ""
        screen._selected_server_type = MagicMock(return_value=screen._TYPE_REMOTE_VALUE)
        screen._sync_transport_fields = MagicMock()
        screen._update_form_dirty = MagicMock()

        checkbox = SimpleNamespace(value=False)
        screen.query_one = MagicMock(return_value=checkbox)
        event = SimpleNamespace(select=SimpleNamespace(id="mcp-type"))

        screen._on_form_select_changed(event)

        assert checkbox.value is True
        screen._sync_transport_fields.assert_called_once()
        screen._update_form_dirty.assert_called_once()

    def test_mcp_manager_select_remote_keeps_existing_oauth_value_for_loaded_alias(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        screen._active_alias = "remote_demo"
        screen._selected_server_type = MagicMock(return_value=screen._TYPE_REMOTE_VALUE)
        screen._sync_transport_fields = MagicMock()
        screen._update_form_dirty = MagicMock()

        checkbox = SimpleNamespace(value=False)
        screen.query_one = MagicMock(return_value=checkbox)
        event = SimpleNamespace(select=SimpleNamespace(id="mcp-type"))

        screen._on_form_select_changed(event)

        assert checkbox.value is False
        screen._sync_transport_fields.assert_called_once()
        screen._update_form_dirty.assert_called_once()

    def test_mcp_manager_oauth_copy_url_warns_when_missing(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        screen.notify = MagicMock()
        screen._oauth_pending_url = ""

        screen._oauth_copy_login_url()

        screen.notify.assert_called_once()
        assert "No OAuth login URL is pending" in screen.notify.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_manager_oauth_login_refuses_when_pending(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        screen.notify = MagicMock()
        screen._oauth_pending_alias = "remote_demo"
        screen._oauth_pending_state = "state-1"

        await screen._oauth_start_browser_login()

        screen.notify.assert_called_once()
        assert "already in progress" in screen.notify.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_manager_load_button_prefers_selected_row(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        screen._request_alias_switch = AsyncMock()
        screen._selected_summary_alias = MagicMock(return_value="selected_alias")
        screen._current_alias = MagicMock(return_value="typed_alias")

        event = SimpleNamespace(button=SimpleNamespace(id="mcp-btn-load"))
        await screen.on_button_pressed(event)

        screen._request_alias_switch.assert_awaited_once_with("selected_alias")

    @pytest.mark.asyncio
    async def test_mcp_manager_oauth_login_respects_browser_login_flag(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(
            manager=MagicMock(),
            oauth_browser_login_enabled=False,
        )
        screen.notify = MagicMock()

        await screen._oauth_start_browser_login()

        screen.notify.assert_called_once()
        assert "disabled by config" in screen.notify.call_args.args[0]

    @pytest.mark.asyncio
    async def test_auth_manager_load_button_prefers_selected_row(self):
        from loom.tui.screens.auth_manager import AuthManagerScreen

        screen = AuthManagerScreen(workspace=Path("/tmp"))
        screen._request_profile_switch = AsyncMock()
        screen._selected_summary_profile_id = MagicMock(return_value="selected_profile")
        screen._profile_id = MagicMock(return_value="typed_profile")

        event = SimpleNamespace(button=SimpleNamespace(id="auth-btn-load"))
        await screen.on_button_pressed(event)

        screen._request_profile_switch.assert_awaited_once_with("selected_profile")

    @pytest.mark.asyncio
    async def test_mcp_manager_start_new_alias_when_clean_resets_form(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        screen._form_dirty = False
        screen._set_blank_form = MagicMock()
        screen.notify = MagicMock()
        alias_input = MagicMock()
        alias_input.focus = MagicMock()
        screen.query_one = MagicMock(return_value=alias_input)

        await screen._start_new_alias()

        screen._set_blank_form.assert_called_once()
        alias_input.focus.assert_called_once()
        screen.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_manager_start_new_alias_when_dirty_prompts_switch_dialog(self):
        from textual.app import App

        from loom.tui.screens.mcp_manager import MCPManagerScreen

        class _Merged:
            def as_views(self):
                return []

        class _Mgr:
            def load(self):
                return _Merged()

        class _HarnessApp(App):
            CSS_PATH = None

            def __init__(self):
                super().__init__()
                self.screen_under_test = MCPManagerScreen(manager=_Mgr())

            def compose(self):
                yield self.screen_under_test

        app = _HarnessApp()
        screen = app.screen_under_test
        screen.notify = MagicMock()
        push_calls: list[tuple[object, object]] = []

        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            screen._form_dirty = True
            screen._active_alias = "notion"
            app.push_screen = MagicMock(
                side_effect=lambda modal, callback=None: push_calls.append((modal, callback)),
            )

            await screen._start_new_alias()
            await pilot.pause()

        assert len(push_calls) == 1
        assert push_calls[0][1] is not None
        screen.notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_manager_oauth_complete_clears_pending_on_missing_access_token(self):
        from loom.oauth.engine import OAuthProviderConfig
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        screen = MCPManagerScreen(manager=MagicMock())
        screen.notify = MagicMock()
        screen._oauth_show_status = AsyncMock()
        screen._oauth_pending_alias = "remote_demo"
        screen._oauth_pending_state = "state-1"
        screen._oauth_pending_provider = OAuthProviderConfig(
            authorization_endpoint="https://auth.example.com/authorize",
            token_endpoint="https://auth.example.com/token",
            client_id="loom-cli",
        )
        screen._oauth_engine.await_callback = MagicMock(return_value=MagicMock())
        screen._oauth_engine.finish_auth = MagicMock(return_value={"token_type": "Bearer"})

        await screen._oauth_complete_pending_login("remote_demo")

        assert screen._oauth_pending_alias == ""
        assert screen._oauth_pending_state == ""
        assert screen._oauth_pending_url == ""
        assert screen._oauth_pending_provider is None
        assert "missing access_token" in screen.notify.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_manager_remove_alias_cancels_pending_oauth(self):
        from loom.tui.screens.mcp_manager import MCPManagerScreen

        manager = MagicMock()
        manager.remove_server = MagicMock()
        screen = MCPManagerScreen(manager=manager)
        screen.notify = MagicMock()
        screen._refresh_summary = AsyncMock()
        screen._manager_workspace = MagicMock(return_value=None)
        screen._oauth_pending_alias = "remote_demo"
        screen._oauth_pending_state = "state-1"
        screen._oauth_engine.cancel_auth = MagicMock()

        await screen._remove_alias_confirmed("remote_demo", impact=None)

        screen._oauth_engine.cancel_auth.assert_called_once_with(state="state-1")
        assert screen._oauth_pending_alias == ""
        assert screen._oauth_pending_state == ""


# --- Theme tests ---


class TestTheme:
    def test_loom_dark_theme(self):
        from loom.tui.theme import LOOM_DARK
        assert LOOM_DARK.name == "loom-dark"
        assert LOOM_DARK.dark is True
        assert LOOM_DARK.primary == "#7dcfff"

    def test_markdown_rich_theme_overrides_magenta_defaults(self):
        from loom.tui.theme import LOOM_MARKDOWN_RICH_THEME

        styles = LOOM_MARKDOWN_RICH_THEME.styles
        assert "markdown.h2" in styles
        assert str(styles["markdown.h2"]) == "underline #7dcfff"
        assert str(styles["markdown.block_quote"]) == "#9aa5ce"

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


class TestActivityIndicator:
    def test_idle_render_uses_dim_static_dots(self):
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        indicator = ActivityIndicator(dot_count=8, idle_hold_ms=0)
        rendered = indicator.render()
        assert rendered.count(indicator._GLYPH_IDLE) == 8
        assert indicator._GLYPH_HEAD not in rendered

    def test_active_frame_progression_ping_pongs(self):
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        indicator = ActivityIndicator(dot_count=4, idle_hold_ms=0)
        indicator.set_active(True)
        sequence = [indicator._frame_index]
        for _ in range(7):
            indicator._advance_frame()
            sequence.append(indicator._frame_index)
        assert sequence == [0, 1, 2, 3, 2, 1, 0, 1]

    def test_inactive_resets_to_dim_strip(self):
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        indicator = ActivityIndicator(dot_count=4, idle_hold_ms=0)
        indicator.set_active(True)
        indicator._advance_frame()
        indicator.set_active(False)
        rendered = indicator.render()
        assert indicator._frame_index == 0
        assert rendered.count(indicator._GLYPH_IDLE) == 4
        assert indicator._GLYPH_HEAD not in rendered

    def test_repeated_idle_sync_does_not_restart_hold_window(self):
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        indicator = ActivityIndicator(dot_count=8, idle_hold_ms=300)
        indicator.set_active(True)
        indicator.set_active(False)
        first_hold_until = indicator._hold_until
        indicator.set_active(False)
        assert indicator._hold_until == first_hold_until


class TestTaskProgressPanel:
    def test_render_empty(self):
        from rich.text import Text

        from loom.tui.widgets.sidebar import TaskProgressPanel
        panel = TaskProgressPanel()
        rendered = panel.render()
        assert isinstance(rendered, Text)
        assert "No tasks tracked" in rendered.plain

    def test_render_with_tasks(self):
        from rich.console import Console

        from loom.tui.widgets.sidebar import TaskProgressPanel
        panel = TaskProgressPanel()
        panel.tasks = [
            {"content": "Read file", "status": "completed"},
            {
                "content": "crypto-externalities-article-adhoc #2f3f27 Running 29:46",
                "status": "in_progress",
            },
            {"content": "Run tests", "status": "pending"},
            {"content": "Handle failure", "status": "failed"},
            {"content": "Skip optional step", "status": "skipped"},
        ]
        rendered = panel.render()
        console = Console(width=34, record=True)
        console.print(rendered)
        plain = console.export_text(styles=False)

        assert "Read file" in plain
        assert "crypto-externalities-article-adh" in plain
        assert "oc #2f3f27 Running 29:46" in plain
        assert "Run tests" in plain
        assert "Handle failure" in plain
        assert "Skip optional step" in plain
        assert "\n◉\n" not in plain

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

    def test_process_run_controls_and_restart_visibility(self):
        from loom.tui.app import ProcessRunPane

        pane = ProcessRunPane(
            run_id="abc123",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
        )
        pane.set_status_header(
            status="queued",
            elapsed="0:00",
            task_id="",
            working_folder="(workspace root)",
        )
        meta_text = str(getattr(pane._meta, "_Static__content", ""))
        assert "Working folder:" in meta_text
        assert "(workspace root)" in meta_text
        assert pane._actions.display is True
        assert pane._toggle_pause_button.display is False
        assert pane._stop_button.display is True
        assert pane._stop_button.disabled is False

        pane.set_status_header(status="running", elapsed="0:01", task_id="cowork-1")
        assert pane._actions.display is True
        assert pane._toggle_pause_button.display is True
        assert pane._toggle_pause_button.disabled is False
        assert pane._toggle_pause_button.label == "\u23f8"
        assert pane._stop_button.display is True
        assert pane._stop_button.disabled is False
        assert pane._restart_button.display is False
        assert pane._restart_button.disabled is True

        pane.set_status_header(status="paused", elapsed="0:11", task_id="cowork-1")
        assert pane._toggle_pause_button.disabled is False
        assert pane._toggle_pause_button.label == "\u25b6"

        pane.set_status_header(status="failed", elapsed="0:42", task_id="")
        assert pane._actions.display is False
        assert pane._toggle_pause_button.display is False
        assert pane._restart_button.display is False
        assert pane._restart_button.disabled is True

        pane.set_status_header(status="failed", elapsed="0:42", task_id="cowork-1")
        assert pane._actions.display is True
        assert pane._toggle_pause_button.display is False
        assert pane._restart_button.display is True
        assert pane._restart_button.disabled is False

    def test_elapsed_seconds_freezes_while_paused(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            started_at=10.0,
            ended_at=None,
            paused_started_at=30.0,
            paused_accumulated_seconds=4.0,
        )

        with patch("loom.tui.app.time.monotonic", return_value=40.0):
            first = app._elapsed_seconds_for_run(run)
        with patch("loom.tui.app.time.monotonic", return_value=120.0):
            second = app._elapsed_seconds_for_run(run)

        assert first == second
        assert first == 16.0

    def test_has_active_process_runs_ignores_paused_status(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_runs = {
            "p1": SimpleNamespace(status="paused"),
            "p2": SimpleNamespace(status="completed"),
        }
        assert app._has_active_process_runs() is False

        app._process_runs["p3"] = SimpleNamespace(status="running")
        assert app._has_active_process_runs() is True

    def test_process_run_working_folder_label_formats_root_and_relative(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        run = SimpleNamespace(run_workspace=tmp_path)
        assert app._process_run_working_folder_label(run) == "(workspace root)"

        run.run_workspace = tmp_path / "runs" / "research-report"
        assert app._process_run_working_folder_label(run) == "runs/research-report"

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

    def test_process_run_stage_rows_render_when_tasks_empty(self):
        from loom.tui.app import LoomApp, ProcessRunState

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = ProcessRunState(
            run_id="abc123",
            process_name="market-research",
            goal="Analyze EPCOR",
            run_workspace=Path("/tmp"),
            process_defn=None,
            pane_id="tab-run-abc123",
            pane=pane,
            status="queued",
            launch_stage="auth_preflight",
            tasks=[],
        )

        rows = app._process_run_stage_rows(run)

        assert rows
        assert any(row["content"] == "Auth preflight" for row in rows)
        assert any(row["status"] == "in_progress" for row in rows)

    def test_process_run_heartbeat_emits_liveness_line(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            closed=False,
            status="queued",
            launch_stage="auth_preflight",
            launch_last_progress_at=0.0,
            launch_last_heartbeat_at=0.0,
            launch_silent_warning_emitted=False,
            launch_stage_heartbeat_stage="",
            launch_stage_heartbeat_dots=0,
            activity_log=[],
            launch_stage_activity_indices={},
            pane=MagicMock(),
        )

        app._maybe_emit_process_run_heartbeat(run)

        assert run.activity_log
        assert run.activity_log[-1].startswith("Auth preflight.")
        assert "Still working" not in run.activity_log[-1]
        assert run.launch_last_heartbeat_at > 0.0

    def test_process_run_heartbeat_updates_same_stage_line_with_more_dots(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            closed=False,
            status="queued",
            launch_stage="resolving_process",
            launch_last_progress_at=0.0,
            launch_last_heartbeat_at=0.0,
            launch_silent_warning_emitted=False,
            launch_stage_heartbeat_stage="",
            launch_stage_heartbeat_dots=0,
            activity_log=[],
            launch_stage_activity_indices={},
            pane=MagicMock(),
        )

        app._maybe_emit_process_run_heartbeat(run)
        first = run.activity_log[-1]
        run.launch_last_progress_at -= 7.0
        run.launch_last_heartbeat_at -= 7.0
        app._maybe_emit_process_run_heartbeat(run)
        second = run.activity_log[-1]

        assert len(run.activity_log) == 1
        assert first == "Resolving process."
        assert second == "Resolving process.."

    def test_process_run_heartbeat_updates_running_stage_line(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            closed=False,
            status="running",
            launch_stage="running",
            launch_last_progress_at=0.0,
            launch_last_heartbeat_at=0.0,
            launch_silent_warning_emitted=False,
            launch_stage_heartbeat_stage="",
            launch_stage_heartbeat_dots=0,
            activity_log=[],
            launch_stage_activity_indices={},
            pane=MagicMock(),
        )

        app._maybe_emit_process_run_heartbeat(run)
        first = run.activity_log[-1]
        run.launch_last_progress_at -= 7.0
        run.launch_last_heartbeat_at -= 7.0
        app._maybe_emit_process_run_heartbeat(run)
        second = run.activity_log[-1]

        assert len(run.activity_log) == 1
        assert first == "Running."
        assert second == "Running.."
        run.pane.upsert_activity.assert_called()

    def test_set_process_run_launch_stage_finalizes_previous_phase_with_elapsed(self):
        from loom.tui.app import LoomApp, ProcessRunState

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = ProcessRunState(
            run_id="abc777",
            process_name="market-research",
            goal="Analyze EPCOR",
            run_workspace=Path("/tmp"),
            process_defn=None,
            pane_id="tab-run-abc777",
            pane=pane,
            status="queued",
            launch_stage="resolving_process",
            launch_stage_started_at=time.monotonic() - 12.0,
        )

        app._set_process_run_launch_stage(run, "provisioning_workspace", note="")

        assert any("Resolving process." in line and "00:12" in line for line in run.activity_log)
        assert any(line.startswith("Provisioning workspace.") for line in run.activity_log)

    def test_process_run_progress_keeps_stage_summary_when_tasks_ready(self):
        from loom.tui.app import LoomApp, ProcessRunState

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = ProcessRunState(
            run_id="abc125",
            process_name="market-research",
            goal="Analyze EPCOR",
            run_workspace=Path("/tmp"),
            process_defn=None,
            pane_id="tab-run-abc125",
            pane=pane,
            status="running",
            launch_stage="auth_preflight",
            tasks=[
                {
                    "id": "scope-companies",
                    "status": "in_progress",
                    "content": "Scope requested companies",
                },
            ],
        )

        app._refresh_process_run_progress(run)

        rows = pane.set_tasks.call_args.args[0]
        assert rows[0]["id"] == "stage:summary"
        assert rows[0]["status"] == "in_progress"
        assert "Auth preflight" in rows[0]["content"]
        assert rows[1]["id"] == "scope-companies"


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
        from rich.markdown import Markdown as RichMarkdown

        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        widget = MagicMock()
        log._stream_widget = widget
        log._stream_text = "chunk"
        log._stream_buffer = ["!"]

        log._flush_and_reset_stream()

        assert widget.update.call_count == 2
        assert widget.update.call_args_list[0].args[0] == "chunk!"
        assert isinstance(widget.update.call_args_list[1].args[0], RichMarkdown)
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

    def test_turn_separator_renders_latency_and_throughput(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_turn_separator(
            tool_count=2,
            tokens=42,
            model="test-model",
            tokens_per_second=21.0,
            latency_ms=450,
            total_time_ms=2200,
        )

        rendered = str(mounted[-1].render())
        assert "2 tools" in rendered
        assert "42 tokens" in rendered
        assert "21.0 tok/s" in rendered
        assert "450ms latency" in rendered
        assert "2.2s total" in rendered
        assert "test-model" in rendered

    def test_turn_separator_renders_context_telemetry(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_turn_separator(
            tool_count=0,
            tokens=12,
            model="test-model",
            context_tokens=19221,
            context_messages=45,
            omitted_messages=57,
            recall_index_used=True,
        )

        rendered = str(mounted[-1].render())
        assert "ctx 19,221 tok" in rendered
        assert "45 ctx msg" in rendered
        assert "57 archived" in rendered
        assert "recall-index" in rendered

    def test_model_text_uses_markdown_renderer(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_model_text("## Heading\n\n- a\n- b\n\n`code`")

        assert mounted
        rendered = mounted[-1].render()
        assert type(rendered).__name__ == "RichVisual"

    def test_model_text_markup_mode_keeps_rich_markup(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_model_text("[bold]Error[/]", markup=True)

        assert mounted
        rendered = mounted[-1].render()
        assert str(rendered) == "Error"

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

    def test_streaming_scrolls_on_mount_and_flush_only(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        log.mount = lambda *_args, **_kwargs: None
        log._schedule_stream_flush = lambda: None
        log._scroll_to_end = MagicMock()

        # First chunk mounts stream widget, so one scroll.
        log.add_streaming_text("a")
        # Next three chunks are buffered only, no additional scroll.
        log.add_streaming_text("b")
        log.add_streaming_text("c")
        log.add_streaming_text("d")
        # Fifth buffered chunk flushes, so one more scroll.
        log.add_streaming_text("e")

        assert log._scroll_to_end.call_count == 2

    def test_scroll_to_end_is_coalesced_per_refresh(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        log._auto_scroll = True
        log._scroll_end_pending = False
        scheduled: list = []
        log.call_after_refresh = lambda callback, *_a, **_k: scheduled.append(callback)
        log.scroll_end = MagicMock()

        log._scroll_to_end()
        log._scroll_to_end()

        assert len(scheduled) == 1
        scheduled[0]()
        log.scroll_end.assert_called_once_with(animate=False, immediate=True)
        assert log._scroll_end_pending is False

    def test_watch_scroll_y_toggles_auto_follow_by_position(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        calls: list[tuple[float, float]] = []

        # Ensure parent watch handler is still invoked.
        log.show_vertical_scrollbar = False
        log._refresh_scroll = lambda: calls.append((1.0, 2.0))

        log._auto_scroll = True
        log._is_near_bottom = lambda **_kwargs: False
        log.watch_scroll_y(1.0, 2.0)
        assert log._auto_scroll is False

        log._is_near_bottom = lambda **_kwargs: True
        log.watch_scroll_y(2.0, 3.0)
        assert log._auto_scroll is True
        assert calls

    def test_link_aware_widget_opens_url_on_click(self):
        from types import SimpleNamespace

        from rich.style import Style

        from loom.tui.widgets.chat_log import LinkAwareStatic

        opened: list[str] = []
        stopped: list[bool] = []
        widget = LinkAwareStatic("")
        widget._open_link = lambda href: opened.append(href)  # type: ignore[method-assign]

        widget.on_click(
            SimpleNamespace(
                style=Style(link="https://example.com"),
                stop=lambda: stopped.append(True),
            )
        )

        assert opened == ["https://example.com"]
        assert stopped == [True]

    def test_link_aware_widget_sets_tooltip_from_hovered_link(self):
        from types import SimpleNamespace

        from rich.style import Style

        from loom.tui.widgets.chat_log import LinkAwareStatic

        widget = LinkAwareStatic("")
        widget.on_mouse_move(SimpleNamespace(style=Style(link="https://example.com")))
        assert widget.tooltip == "https://example.com"

        widget.on_mouse_move(SimpleNamespace(style=Style(link="#heading")))
        assert widget.tooltip is None

    def test_delegate_progress_section_lifecycle(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_delegate_progress_section("call_1", title="Delegated progress")
        assert log.has_delegate_progress_section("call_1") is True
        assert log.append_delegate_progress_line("call_1", "Started subtask.") is True
        assert log.finalize_delegate_progress_section(
            "call_1",
            success=True,
            elapsed_ms=1250,
        ) is True

        log.reset_runtime_state()
        assert log.has_delegate_progress_section("call_1") is False


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
                phase_id="company-screening",
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
        assert result.data["tasks"][0]["phase_id"] == "company-screening"
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
    async def test_progress_callback_forwards_ask_user_events(self):
        from loom.events.bus import Event, EventBus
        from loom.events.types import ASK_USER_REQUESTED
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        bus = EventBus()

        async def _execute(task):
            task.status = TaskStatus.EXECUTING
            bus.emit(Event(
                event_type=ASK_USER_REQUESTED,
                task_id=task.id,
                data={
                    "subtask_id": "s1",
                    "question_id": "q-1",
                    "question": "Pick runtime",
                    "question_type": "single_choice",
                    "options": [
                        {"id": "python", "label": "Python", "description": ""},
                        {"id": "rust", "label": "Rust", "description": ""},
                    ],
                },
            ))
            task.status = TaskStatus.COMPLETED
            return task

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
        events = [p for p in progress if p.get("event_type") == ASK_USER_REQUESTED]
        assert events
        assert events[0]["event_data"]["question_id"] == "q-1"

    @pytest.mark.asyncio
    async def test_delegate_registers_and_clears_cancel_handler(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        captured: dict[str, object] = {}
        cleared: list[bool] = []

        async def _execute(task):
            task.status = TaskStatus.COMPLETED
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.cancel_task = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        def _register(payload):
            captured.update(payload)

        def _clear():
            cleared.append(True)

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        result = await tool.execute(
            {
                "goal": "Analyze Tesla",
                "_register_cancel_handler": _register,
                "_clear_cancel_handler": _clear,
            },
            ctx,
        )

        assert result.success is True
        assert isinstance(result.data, dict)
        assert captured.get("task_id") == result.data.get("task_id")
        assert callable(captured.get("cancel"))
        assert callable(captured.get("pause"))
        assert callable(captured.get("resume"))
        assert callable(captured.get("inject"))
        assert callable(captured.get("answer_question"))
        assert cleared == [True]

    @pytest.mark.asyncio
    async def test_delegate_cancel_handler_requests_orchestrator_cancel(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        gate = asyncio.Event()
        cancel_holder: dict[str, object] = {}
        registered = asyncio.Event()

        async def _execute(task):
            await gate.wait()
            return task

        def _cancel_task(task):
            task.status = TaskStatus.CANCELLED
            gate.set()

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.cancel_task = MagicMock(side_effect=_cancel_task)
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            return orchestrator

        def _register(payload):
            cancel_holder.update(payload)
            registered.set()

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        pending = asyncio.create_task(tool.execute(
            {
                "goal": "Analyze Tesla",
                "_register_cancel_handler": _register,
            },
            ctx,
        ))
        await asyncio.wait_for(registered.wait(), timeout=1.0)

        cancel_fn = cancel_holder.get("cancel")
        assert callable(cancel_fn)
        cancel_response = await cancel_fn(wait_timeout_seconds=0.2)
        assert cancel_response["requested"] is True
        assert cancel_response["path"] == "orchestrator"
        assert cancel_response["timeout"] is False

        result = await pending
        assert result.success is False
        assert result.error == "Task execution cancelled."
        assert isinstance(result.data, dict)
        assert result.data.get("status") == "cancelled"

    @pytest.mark.asyncio
    async def test_delegate_cancellation_does_not_cancel_paused_task(self):
        from loom.state.task_state import TaskStatus
        from loom.tools.delegate_task import DelegateTaskTool
        from loom.tools.registry import ToolContext

        orchestrator_holder: dict[str, object] = {}

        async def _execute(task):
            task.status = TaskStatus.PAUSED
            await asyncio.Future()
            return task

        async def _factory():
            orchestrator = MagicMock()
            orchestrator.cancel_task = MagicMock()
            orchestrator.execute_task = AsyncMock(side_effect=_execute)
            orchestrator_holder["value"] = orchestrator
            return orchestrator

        tool = DelegateTaskTool(orchestrator_factory=_factory)
        ctx = ToolContext(workspace=Path("/tmp"))
        pending = asyncio.create_task(tool.execute({"goal": "Analyze Tesla"}, ctx))
        await asyncio.sleep(0.05)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

        orchestrator = orchestrator_holder["value"]
        orchestrator.cancel_task.assert_not_called()

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
    async def test_delegate_defaults_to_disabled_approval_in_tui_surface(self):
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

        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert observed_modes == ["disabled"]

    @pytest.mark.asyncio
    async def test_delegate_keeps_confidence_threshold_default_for_api_surface(self):
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
        ctx = ToolContext(workspace=Path("/tmp"), execution_surface="api")

        result = await tool.execute({"goal": "Analyze Tesla"}, ctx)

        assert result.success is True
        assert observed_modes == ["confidence_threshold"]

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
    def test_ingest_files_panel_falls_back_to_summary_markers(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        panel = MagicMock()
        panel._all_entries = []
        app.query_one = MagicMock(return_value=panel)

        count = app._ingest_files_panel_from_paths(
            ["(2 files created)", "(1 files modified)"],
            operation_hint="modify",
        )

        assert count == 2
        entries = panel.update_files.call_args.args[0]
        assert entries[0]["operation"] == "create"
        assert entries[0]["path"] == "(2 files created)"
        assert entries[1]["operation"] == "modify"
        assert entries[1]["path"] == "(1 files modified)"

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
    async def test_cowork_delegate_progress_updates_chat_section(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        chat.append_delegate_progress_line.return_value = True
        sidebar = MagicMock()
        events_panel = MagicMock()
        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: {
            "#chat-log": chat,
            "#sidebar": sidebar,
            "#events-panel": events_panel,
        }[selector])
        app._append_chat_replay_event = AsyncMock()
        app._request_workspace_refresh = MagicMock()
        app._ingest_files_panel_from_paths = MagicMock(return_value=1)

        await app._on_cowork_delegate_progress_event({
            "tool_call_id": "call_1",
            "caller_tool_name": "delegate_task",
            "event_type": "tool_call_completed",
            "event_data": {
                "tool": "write_file",
                "subtask_id": "scope",
                "files_changed": ["notes.md"],
            },
            "tasks": [
                {"id": "scope", "status": "completed", "content": "Scope work"},
            ],
        })

        chat.add_delegate_progress_section.assert_called()
        chat.append_delegate_progress_line.assert_called_once()
        replay_events = [call.args[0] for call in app._append_chat_replay_event.await_args_list]
        assert replay_events == ["delegate_progress_started"]
        app._request_workspace_refresh.assert_called_once()
        app._ingest_files_panel_from_paths.assert_called_once()

    @pytest.mark.asyncio
    async def test_cowork_delegate_progress_ignores_late_events_after_finalize(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        sidebar = MagicMock()
        events_panel = MagicMock()
        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: {
            "#chat-log": chat,
            "#sidebar": sidebar,
            "#events-panel": events_panel,
        }[selector])
        app._append_chat_replay_event = AsyncMock()
        app._request_workspace_refresh = MagicMock()
        app._ingest_files_panel_from_paths = MagicMock(return_value=0)
        app._active_delegate_streams["call_done"] = {
            "tool_call_id": "call_done",
            "title": "Delegated progress",
            "status": "completed",
            "elapsed_ms": 1200,
            "lines": ["Delegated task completed."],
            "started_at": 1.0,
            "finalized": True,
        }

        await app._on_cowork_delegate_progress_event({
            "tool_call_id": "call_done",
            "caller_tool_name": "delegate_task",
            "event_type": "subtask_started",
            "event_data": {
                "subtask_id": "late",
            },
        })

        chat.add_delegate_progress_section.assert_not_called()
        chat.append_delegate_progress_line.assert_not_called()
        app._append_chat_replay_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cowork_delegate_progress_skips_token_streamed_lines(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        chat.append_delegate_progress_line.return_value = True
        sidebar = MagicMock()
        events_panel = MagicMock()
        app.query_one = MagicMock(side_effect=lambda selector, *_args, **_kwargs: {
            "#chat-log": chat,
            "#sidebar": sidebar,
            "#events-panel": events_panel,
        }[selector])
        app._append_chat_replay_event = AsyncMock()
        app._request_workspace_refresh = MagicMock()
        app._ingest_files_panel_from_paths = MagicMock(return_value=0)

        await app._on_cowork_delegate_progress_event({
            "tool_call_id": "call_tokens",
            "caller_tool_name": "delegate_task",
            "event_type": "token_streamed",
            "event_data": {
                "subtask_id": "scope",
                "token_count": 12,
            },
        })

        chat.add_delegate_progress_section.assert_called_once()
        chat.append_delegate_progress_line.assert_not_called()
        replay_events = [call.args[0] for call in app._append_chat_replay_event.await_args_list]
        assert replay_events == ["delegate_progress_started"]

    def test_format_process_progress_event_uses_cowork_delegate_terminal_phrasing(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        assert (
            app._format_process_progress_event(
                {"event_type": "task_completed", "event_data": {}},
                context="cowork_delegate",
            )
            == "Delegated task completed."
        )
        assert (
            app._format_process_progress_event(
                {
                    "event_type": "task_failed",
                    "event_data": {"reason": "rate limited"},
                },
                context="cowork_delegate",
            )
            == "Delegated task failed: rate limited"
        )

    def test_render_chat_event_delegate_progress_rehydrates(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        chat.append_delegate_progress_line.return_value = True
        app.query_one = MagicMock(return_value=chat)

        assert app._render_chat_event({
            "event_type": "delegate_progress_started",
            "payload": {
                "tool_call_id": "call_2",
                "title": "Delegated progress",
            },
        })
        assert app._render_chat_event({
            "event_type": "delegate_progress_line",
            "payload": {
                "tool_call_id": "call_2",
                "line": "Started scope.",
            },
        })
        assert app._render_chat_event({
            "event_type": "delegate_progress_finalized",
            "payload": {
                "tool_call_id": "call_2",
                "title": "Delegated progress",
                "status": "completed",
                "elapsed_ms": 320,
                "lines": ["Started scope.", "Completed scope."],
            },
        })

        assert chat.add_delegate_progress_section.call_count >= 2
        chat.append_delegate_progress_line.assert_called_once()
        chat.finalize_delegate_progress_section.assert_called_once()
        stream = app._active_delegate_streams.get("call_2")
        assert isinstance(stream, dict)
        assert stream.get("status") == "completed"

    def test_render_chat_event_turn_interrupted_rehydrates(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        assert app._render_chat_event({
            "event_type": "turn_interrupted",
            "payload": {
                "message": "Stopped current chat execution.",
                "markup": True,
            },
        })
        chat.add_info.assert_called_once_with("Stopped current chat execution.", markup=True)

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
        created_registry_args: dict[str, object] = {}

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
            lambda _config=None: (
                created_registry_args.update({"config": _config}) or MagicMock()
            ),
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
        assert created_registry_args.get("config") is app._config

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

    @pytest.mark.asyncio
    async def test_initialize_session_auto_resume_restores_input_history(self, monkeypatch):
        from loom.tui.app import LoomApp

        class FakeSession:
            def __init__(self, **_kwargs):
                self.total_tokens = 12
                self.session_id = "latest-session"
                self.session_state = SimpleNamespace(
                    turn_count=4,
                    ui_state={
                        "input_history": {
                            "version": 1,
                            "items": ["/help", "summarize the doc"],
                        },
                    },
                )
                self.messages = [{"role": "system", "content": "system"}]

            async def resume(self, _session_id: str) -> None:
                self.messages = [
                    {"role": "system", "content": "system"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "summarize the doc"},
                ]

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._resolve_startup_resume_target = AsyncMock(return_value=("latest-session", True))
        app._refresh_tool_registry = MagicMock()
        app._refresh_process_command_index = MagicMock()
        app._ensure_persistence_tools = MagicMock()
        app._apply_process_tool_policy = MagicMock()
        app._build_system_prompt = MagicMock(return_value="system")
        app._bind_session_tools = MagicMock()
        app._restore_process_run_tabs = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._tools.list_tools = MagicMock(return_value=[])

        chat = MagicMock()
        status = SimpleNamespace(workspace_name="", model_name="", process_name="")
        input_widget = SimpleNamespace(focus=MagicMock())

        def _query_one(selector: str, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            if selector == "#user-input":
                return input_widget
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        await app._initialize_session()

        assert app._input_history == ["/help", "summarize the doc"]

    @pytest.mark.asyncio
    async def test_initialize_session_auto_resume_replays_chat_history(self, monkeypatch):
        from loom.tui.app import LoomApp

        class FakeSession:
            def __init__(self, **_kwargs):
                self.total_tokens = 12
                self.session_id = "latest-session"
                self.session_state = SimpleNamespace(turn_count=4, ui_state={})
                self.messages = [{"role": "system", "content": "system"}]

            async def resume(self, _session_id: str) -> None:
                self.messages = [{"role": "system", "content": "system"}]

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = MagicMock()
        app._store.get_chat_events = AsyncMock(return_value=[
            {
                "seq": 1,
                "event_type": "user_message",
                "payload": {"text": "hello"},
            },
            {
                "seq": 2,
                "event_type": "assistant_text",
                "payload": {"text": "world", "markup": False},
            },
        ])
        app._resolve_startup_resume_target = AsyncMock(return_value=("latest-session", True))
        app._refresh_tool_registry = MagicMock()
        app._refresh_process_command_index = MagicMock()
        app._ensure_persistence_tools = MagicMock()
        app._apply_process_tool_policy = MagicMock()
        app._build_system_prompt = MagicMock(return_value="system")
        app._bind_session_tools = MagicMock()
        app._restore_process_run_tabs = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._tools.list_tools = MagicMock(return_value=[])

        chat = MagicMock()
        chat.children = []
        status = SimpleNamespace(workspace_name="", model_name="", process_name="")
        input_widget = SimpleNamespace(focus=MagicMock())

        def _query_one(selector: str, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            if selector == "#user-input":
                return input_widget
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        await app._initialize_session()

        chat.add_user_message.assert_any_call("hello")
        chat.add_model_text.assert_any_call("world", markup=False)

    @pytest.mark.asyncio
    async def test_initialize_session_refreshes_process_index_in_background(self, monkeypatch):
        from loom.tui.app import LoomApp

        class FakeSession:
            def __init__(self, **_kwargs):
                self.total_tokens = 0
                self.session_id = "s1"
                self.session_state = SimpleNamespace(turn_count=0, ui_state={})
                self.messages = []

        monkeypatch.setattr("loom.tui.app.CoworkSession", FakeSession)

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._store = None
        app._resolve_startup_resume_target = AsyncMock(return_value=(None, False))
        app._refresh_tool_registry = MagicMock()
        app._refresh_process_command_index = MagicMock()
        app._ensure_persistence_tools = MagicMock()
        app._apply_process_tool_policy = MagicMock()
        app._build_system_prompt = MagicMock(return_value="system")
        app._bind_session_tools = MagicMock()
        app._hydrate_chat_history_for_active_session = AsyncMock()
        app._restore_process_run_tabs = AsyncMock()
        app._set_slash_hint = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        app._tools.list_tools = MagicMock(return_value=[])
        app._reset_cowork_steering_state = MagicMock()
        app._hydrate_input_history_from_session = MagicMock()

        chat = MagicMock()
        status = SimpleNamespace(workspace_name="", model_name="", process_name="")
        input_widget = SimpleNamespace(focus=MagicMock())

        def _query_one(selector: str, *_args, **_kwargs):
            if selector == "#chat-log":
                return chat
            if selector == "#status-bar":
                return status
            if selector == "#user-input":
                return input_widget
            raise AssertionError(f"Unexpected selector: {selector}")

        app.query_one = MagicMock(side_effect=_query_one)

        to_thread_calls: list[tuple] = []

        async def _fake_to_thread(func, *args, **kwargs):
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr("loom.tui.app.asyncio.to_thread", _fake_to_thread)

        await app._initialize_session()

        app._refresh_tool_registry.assert_called_once()
        app._refresh_process_command_index.assert_called_once_with(
            chat=chat,
            notify_conflicts=True,
            background=True,
            force=True,
        )
        assert to_thread_calls

    def test_start_workspace_watch_does_not_scan_synchronously(self):
        from loom.tui.app import LoomApp

        class RunningHarness(LoomApp):
            @property
            def is_running(self) -> bool:  # pragma: no cover - property shim
                return True

        app = RunningHarness(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._stop_workspace_watch = MagicMock()
        app._tui_realtime_refresh_enabled = MagicMock(return_value=True)
        app._tui_workspace_watch_backend = MagicMock(return_value="poll")
        app._tui_workspace_poll_interval_seconds = MagicMock(return_value=2.0)
        app._compute_workspace_signature = MagicMock(
            side_effect=AssertionError("workspace signature scan should not run inline"),
        )
        timer = MagicMock()
        app.set_interval = MagicMock(return_value=timer)

        app._start_workspace_watch()

        app._compute_workspace_signature.assert_not_called()
        app.set_interval.assert_called_once_with(
            2.0,
            app._on_workspace_poll_tick,
        )
        assert app._workspace_poll_timer is timer
        assert app._workspace_signature is None


class TestChatReplayHydration:
    class _PerfChat:
        def __init__(self):
            self.children = []

        def add_user_message(self, _text):
            return None

        def add_model_text(self, _text, *, markup=False):
            _ = markup
            return None

        def add_tool_call(
            self,
            _tool_name,
            _args,
            *,
            success=None,
            elapsed_ms=0,
            output="",
            error="",
        ):
            _ = (success, elapsed_ms, output, error)
            return None

        def add_content_indicator(self, _blocks):
            return None

        def add_turn_separator(
            self,
            _tool_count,
            _tokens,
            _model,
            *,
            tokens_per_second=0.0,
            latency_ms=0,
            total_time_ms=0,
            context_tokens=0,
            context_messages=0,
            omitted_messages=0,
            recall_index_used=False,
        ):
            _ = (
                tokens_per_second,
                latency_ms,
                total_time_ms,
                context_tokens,
                context_messages,
                omitted_messages,
                recall_index_used,
            )
            return None

        def add_info(self, _text, *, markup=True):
            _ = markup
            return None

    @pytest.mark.asyncio
    async def test_hydrate_chat_history_skips_bad_row_and_keeps_rendering(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(session_id="session-1")
        app._store = MagicMock()
        app._store.get_chat_events = AsyncMock(return_value=[
            {"seq": 1, "event_type": "user_message", "payload": {"text": "boom"}},
            {
                "seq": 2,
                "event_type": "assistant_text",
                "payload": {"text": "still renders", "markup": False},
            },
        ])
        app._store.synthesize_chat_events_from_turns = AsyncMock(return_value=[])

        chat = MagicMock()
        chat.children = []
        chat.add_user_message.side_effect = RuntimeError("bad row")
        app.query_one = MagicMock(return_value=chat)

        await app._hydrate_chat_history_for_active_session()

        chat.add_model_text.assert_called_once_with("still renders", markup=False)

    @pytest.mark.asyncio
    async def test_append_chat_replay_event_trims_with_continuity_sentinel(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._chat_resume_max_rendered_rows = MagicMock(return_value=2)

        chat = MagicMock()
        chat.children = []
        app.query_one = MagicMock(return_value=chat)

        await app._append_chat_replay_event(
            "user_message",
            {"text": "one"},
            persist=False,
        )
        await app._append_chat_replay_event(
            "user_message",
            {"text": "two"},
            persist=False,
        )
        await app._append_chat_replay_event(
            "user_message",
            {"text": "three"},
            persist=False,
        )

        assert [row["payload"]["text"] for row in app._chat_replay_events] == ["two", "three"]
        sentinel = chat.add_info.call_args_list[-1].args[0]
        assert "Transcript window truncated" in sentinel

    @pytest.mark.asyncio
    async def test_append_chat_replay_event_does_not_render_by_default(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        chat.children = []
        app.query_one = MagicMock(return_value=chat)

        await app._append_chat_replay_event(
            "user_message",
            {"text": "hello"},
            persist=False,
        )

        assert app._chat_replay_events[-1]["event_type"] == "user_message"
        chat.add_user_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_hydrate_perf_latest_300_rows_under_target(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(session_id="session-1")
        app._store = MagicMock()
        app._store.get_chat_events = AsyncMock(return_value=[
            {
                "seq": idx + 1,
                "event_type": "user_message",
                "payload": {"text": f"row {idx + 1}"},
            }
            for idx in range(300)
        ])
        app._store.synthesize_chat_events_from_turns = AsyncMock(return_value=[])

        chat = self._PerfChat()
        app.query_one = MagicMock(return_value=chat)

        started = asyncio.get_running_loop().time()
        await app._hydrate_chat_history_for_active_session()
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 0.8

    @pytest.mark.asyncio
    async def test_chat_hydrate_perf_older_200_rows_under_target(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._session = SimpleNamespace(session_id="session-1")
        app._store = MagicMock()

        latest = [
            {
                "seq": seq,
                "event_type": "user_message",
                "payload": {"text": f"latest {seq}"},
            }
            for seq in range(201, 401)
        ]
        older = [
            {
                "seq": seq,
                "event_type": "user_message",
                "payload": {"text": f"older {seq}"},
            }
            for seq in range(1, 201)
        ]

        async def _get_chat_events(
            _session_id: str,
            *,
            before_seq: int | None = None,
            after_seq: int | None = None,
            limit: int = 200,
        ):
            _ = (after_seq, limit)
            if before_seq is None:
                return latest
            return older if before_seq == 201 else []

        app._store.get_chat_events = AsyncMock(side_effect=_get_chat_events)
        app._store.synthesize_chat_events_from_turns = AsyncMock(return_value=[])

        chat = self._PerfChat()
        app.query_one = MagicMock(return_value=chat)
        await app._hydrate_chat_history_for_active_session()

        started = asyncio.get_running_loop().time()
        loaded = await app._load_older_chat_history()
        elapsed = asyncio.get_running_loop().time() - started

        assert loaded is True
        assert elapsed < 0.35


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
            workspace=Path("/Users/sfw/Development/loom"),
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
        assert app._slash_completion_candidates("/t") == ["/tools", "/tool", "/tokens"]
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


class TestModelSlashCommands:
    @pytest.mark.asyncio
    async def test_model_without_args_renders_detailed_active_model(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        active_model = SimpleNamespace(
            name="primary",
            model="gpt-4o-mini",
            tier=2,
            roles=["executor", "planner"],
            configured_temperature=0.15,
            configured_max_tokens=16000,
            _config=SimpleNamespace(reasoning_effort="medium"),
            _capabilities=SimpleNamespace(
                vision=True,
                native_pdf=False,
                thinking=False,
                citations=False,
                audio_input=False,
                audio_output=False,
            ),
        )

        app = LoomApp(
            model=active_model,
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "primary": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    max_tokens=16000,
                    temperature=0.15,
                    roles=["executor", "planner"],
                    api_key="sk-test-secret",
                    reasoning_effort="medium",
                )
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/model")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Active Model" in rendered
        assert "primary" in rendered
        assert "openai-chat-completions" in rendered
        assert "https://api.example.com/v1" in rendered
        assert "gpt-4o-mini" in rendered
        assert "sk-test-secret" not in rendered

    @pytest.mark.asyncio
    async def test_model_with_args_shows_usage(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/model secondary")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Usage" in rendered
        assert "/model" in rendered
        assert "Runtime model switching is not supported yet." in rendered

    @pytest.mark.asyncio
    async def test_models_without_args_renders_catalog_and_redacts(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        active_model = SimpleNamespace(
            name="primary",
            model="gpt-4o-mini",
            tier=2,
            roles=["executor"],
            configured_temperature=0.1,
            configured_max_tokens=8192,
        )
        app = LoomApp(
            model=active_model,
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "primary": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1?token=leak",
                    model="gpt-4o-mini",
                    api_key="sk-primary-secret",
                    roles=["executor"],
                ),
                "secondary": ModelConfig(
                    provider="anthropic",
                    base_url="",
                    model="claude-3-5-sonnet-20241022",
                    api_key="sk-secondary-secret",
                    roles=["planner"],
                ),
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/models")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Configured Models" in rendered
        assert "primary" in rendered
        assert "secondary" in rendered
        assert rendered.count("active:[/] yes") == 1
        assert "https://api.example.com/v1" in rendered
        assert "https://api.anthropic.com" in rendered
        assert "sk-primary-secret" not in rendered
        assert "sk-secondary-secret" not in rendered
        assert "token=leak" not in rendered

    @pytest.mark.asyncio
    async def test_models_with_args_shows_usage(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/models anything")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Usage" in rendered
        assert "/models" in rendered

    @pytest.mark.asyncio
    async def test_model_invalid_endpoint_never_echoes_raw_value(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary", model="gpt-4o-mini", tier=1, roles=["executor"]),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "primary": ModelConfig(
                    provider="openai_compatible",
                    base_url="not-a-url?token=secret",
                    model="gpt-4o-mini",
                    roles=["executor"],
                )
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/model")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "(invalid-configured-url)" in rendered
        assert "token=secret" not in rendered

    @pytest.mark.asyncio
    async def test_models_catalog_orders_active_alias_first(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="zzz", model="gpt-4o-mini", tier=1, roles=["executor"]),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "aaa": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    roles=["executor"],
                ),
                "zzz": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    roles=["executor"],
                ),
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/models")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert rendered.index("[bold]zzz[/bold]") < rendered.index("[bold]aaa[/bold]")

    @pytest.mark.asyncio
    async def test_models_with_no_config_shows_runtime_only(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(
                name="ephemeral",
                model="qwen3:8b",
                tier=1,
                roles=["executor"],
                configured_temperature=0.2,
                configured_max_tokens=4096,
                _config=SimpleNamespace(provider="ollama", base_url="http://localhost:11434"),
            ),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/models")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "No configured models." in rendered
        assert "(runtime-only)" in rendered

    @pytest.mark.asyncio
    async def test_model_marks_alias_as_ambiguous(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        runtime_model = SimpleNamespace(
            name="runtime",
            model="gpt-4o-mini",
            tier=2,
            roles=["executor"],
            configured_temperature=0.1,
            configured_max_tokens=8192,
            _config=SimpleNamespace(
                provider="openai_compatible",
                base_url="https://api.example.com/v1",
            ),
        )
        app = LoomApp(
            model=runtime_model,
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "alpha": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    roles=["executor"],
                ),
                "beta": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    roles=["executor"],
                ),
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/model")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "active_alias:[/] ambiguous" in rendered
        assert "candidates:[/] alpha, beta" in rendered


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
        assert captured["config"] is None
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
    async def test_auth_non_manager_subcommand_shows_cli_handoff(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/auth list")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "manager-first" in message
        assert "/auth manage" in message
        assert "loom auth" in message


class TestManagerTabs:
    @pytest.mark.asyncio
    async def test_open_mcp_manager_tab_adds_embedded_view(self, monkeypatch):
        from textual.widgets import Static

        from loom.tui.app import LoomApp

        captured_kwargs: list[dict[str, object]] = []

        class _DummyMCPManagerScreen(Static):
            def __init__(self, *args, **kwargs) -> None:
                captured_kwargs.append(dict(kwargs))
                super().__init__("mcp")

        monkeypatch.setattr("loom.tui.app.MCPManagerScreen", _DummyMCPManagerScreen)

        class _Tabs:
            def __init__(self) -> None:
                self.active = "tab-chat"
                self._panes: dict[str, object] = {}
                self.add_calls = 0

            def get_tab(self, pane_id: str):
                if pane_id not in self._panes:
                    raise LookupError(pane_id)
                return self._panes[pane_id]

            async def add_pane(self, pane, *, after: str | None = None):
                self.add_calls += 1
                self._panes[pane.id] = pane
                _ = after
                return pane

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=SimpleNamespace(mcp=SimpleNamespace(oauth_browser_login=True)),
        )
        tabs = _Tabs()
        app._mcp_manager = MagicMock(return_value=MagicMock())
        app.query_one = MagicMock(return_value=tabs)

        await app._open_mcp_manager_tab()
        await app._open_mcp_manager_tab()

        assert tabs.active == app._MCP_MANAGER_TAB_ID
        assert tabs.add_calls == 1
        assert captured_kwargs
        assert captured_kwargs[0]["embedded"] is True
        assert callable(captured_kwargs[0]["on_close"])

    @pytest.mark.asyncio
    async def test_open_auth_manager_tab_adds_embedded_view(self, monkeypatch):
        from textual.widgets import Static

        from loom.tui.app import LoomApp

        captured_kwargs: list[dict[str, object]] = []

        class _DummyAuthManagerScreen(Static):
            def __init__(self, *args, **kwargs) -> None:
                captured_kwargs.append(dict(kwargs))
                super().__init__("auth")

        monkeypatch.setattr("loom.tui.app.AuthManagerScreen", _DummyAuthManagerScreen)

        class _Tabs:
            def __init__(self) -> None:
                self.active = "tab-chat"
                self._panes: dict[str, object] = {}
                self.add_calls = 0

            def get_tab(self, pane_id: str):
                if pane_id not in self._panes:
                    raise LookupError(pane_id)
                return self._panes[pane_id]

            async def add_pane(self, pane, *, after: str | None = None):
                self.add_calls += 1
                self._panes[pane.id] = pane
                _ = after
                return pane

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        tabs = _Tabs()
        app._auth_discovery_process_defs = MagicMock(return_value=[])
        app._mcp_manager = MagicMock(return_value=MagicMock())
        app.query_one = MagicMock(return_value=tabs)

        await app._open_auth_manager_tab()
        await app._open_auth_manager_tab()

        assert tabs.active == app._AUTH_MANAGER_TAB_ID
        assert tabs.add_calls == 1
        assert captured_kwargs
        assert captured_kwargs[0]["embedded"] is True
        assert callable(captured_kwargs[0]["on_close"])


class TestRunStartAuthResolution:
    def test_collect_required_auth_resources_uses_required_tool_set_when_declared(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._tools = SimpleNamespace(
            list_tools=lambda: ["tool_required", "tool_non_required", "tool_excluded"],
            get=lambda name: {
                "tool_required": SimpleNamespace(
                    auth_requirements=[
                        {"provider": "ga_provider", "source": "api"},
                    ]
                ),
                "tool_non_required": SimpleNamespace(
                    auth_requirements=[
                        {"provider": "non_required_provider", "source": "api"},
                    ]
                ),
                "tool_excluded": SimpleNamespace(
                    auth_requirements=[
                        {"provider": "excluded_provider", "source": "api"},
                    ]
                ),
            }.get(name),
        )
        process_defn = SimpleNamespace(
            auth=SimpleNamespace(required=[{"provider": "notion", "source": "mcp"}]),
            tools=SimpleNamespace(
                required=["tool_required"],
                excluded=["tool_excluded"],
            ),
        )

        required = app._collect_required_auth_resources_for_process(process_defn)
        selectors = {
            (item["provider"], item.get("source", "api"))
            for item in required
        }
        assert ("notion", "mcp") in selectors
        assert ("ga_provider", "api") in selectors
        assert ("non_required_provider", "api") not in selectors
        assert ("excluded_provider", "api") not in selectors

    @pytest.mark.asyncio
    async def test_blocking_unresolved_auth_opens_manager_and_retries(self, monkeypatch):
        from loom.auth.runtime import (
            AuthResourceRequirement,
            UnresolvedAuthResource,
            UnresolvedAuthResourcesError,
        )
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._collect_required_auth_resources_for_process = MagicMock(return_value=[
            {"provider": "notion", "source": "api"},
        ])
        app._prompt_auth_choice = AsyncMock(return_value="Open Auth Manager")
        app._open_auth_manager_for_run_start = AsyncMock(return_value=True)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)
        app._config = SimpleNamespace(mcp=SimpleNamespace(servers={}))

        unresolved_error = UnresolvedAuthResourcesError(
            "blocked",
            unresolved=[
                UnresolvedAuthResource(
                    provider="notion",
                    source="api",
                    reason="auth_missing",
                    message="token missing",
                )
            ],
            defaults_user={},
            defaults_workspace={},
            explicit_overrides={},
            required_resources=[AuthResourceRequirement(provider="notion", source="api")],
        )
        fake_context = SimpleNamespace(
            profile_for_provider=lambda provider: (
                SimpleNamespace(profile_id="notion_dev")
                if provider == "notion" else None
            )
        )
        call_count = {"value": 0}

        def _fake_build_run_auth_context(**_kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                raise unresolved_error
            return fake_context

        monkeypatch.setattr(
            "loom.auth.runtime.build_run_auth_context",
            _fake_build_run_auth_context,
        )

        overrides, required_resources = await app._resolve_auth_overrides_for_run_start(
            process_defn=SimpleNamespace(),
            base_overrides={},
        )

        assert required_resources == [{"provider": "notion", "source": "api"}]
        assert overrides == {"notion": "notion_dev"}
        app._open_auth_manager_for_run_start.assert_awaited_once()
        assert call_count["value"] == 2


class TestCommandPaletteProcessActions:
    def test_ctrl_r_binding_registered(self):
        from loom.tui.app import LoomApp

        keys = {binding.key for binding in LoomApp.BINDINGS}
        assert "ctrl+r" in keys
        assert "ctrl+a" in keys
        assert "ctrl+m" in keys

    def test_auth_mcp_binding_actions_registered(self):
        from loom.tui.app import LoomApp

        bindings = {binding.key: binding for binding in LoomApp.BINDINGS}
        key_to_action = {key: binding.action for key, binding in bindings.items()}
        assert key_to_action["ctrl+a"] == "open_auth_tab"
        assert key_to_action["ctrl+m"] == "open_mcp_tab"
        assert bindings["ctrl+a"].priority is True
        assert bindings["ctrl+m"].priority is True
        assert bindings["ctrl+a"].show is False
        assert bindings["ctrl+m"].show is False

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
        from textual.widgets import Button

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
            assert "^a" in auth_label and "auth" in auth_label
            assert "^m" in mcp_label and "mcp" in mcp_label
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
        from textual.widgets import Button

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
        assert "/history" in rendered
        assert "/run <goal|close" in rendered
        assert "resume <run-id-prefix|current>" in rendered
        assert "run-id-prefix" in rendered
        assert "/quit (aliases: /exit, /q)" in rendered
        assert "/setup" in rendered
        assert "Ctrl+R reload workspace" in rendered
        assert "Ctrl+W close tab" in rendered
        assert "Ctrl+A auth" in rendered
        assert "Ctrl+M mcp" in rendered

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

    @pytest.mark.asyncio
    async def test_resume_while_busy_is_blocked(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._chat_busy = True
        app._store = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/resume abc")

        assert handled is True
        message = chat.add_info.call_args.args[0]
        assert "Cannot create/switch sessions" in message

    @pytest.mark.asyncio
    async def test_new_while_busy_is_blocked(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._chat_busy = True
        app._store = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/new")

        assert handled is True
        message = chat.add_info.call_args.args[0]
        assert "Cannot create/switch sessions" in message

    @pytest.mark.asyncio
    async def test_history_older_reports_result(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._load_older_chat_history = AsyncMock(return_value=True)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/history older")

        assert handled is True
        app._load_older_chat_history.assert_awaited_once()
        chat.add_info.assert_called_once_with("Loaded older chat history.")


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
