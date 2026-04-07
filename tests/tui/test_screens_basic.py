"""Basic TUI screen behavior tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Button

from loom.tui.screens.approval import ToolApprovalScreen
from loom.tui.screens.ask_user import AskUserScreen


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
    async def test_approval_callback_surfaces_prompt_in_chat(self):
        from loom.cowork.approval import ApprovalDecision
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        callbacks: list = []

        def _push_screen(_screen, callback):
            callbacks.append(callback)

        def _query_one(selector, *_args, **_kwargs):
            assert selector == "#chat-log"
            return chat

        app.push_screen = _push_screen
        app.query_one = _query_one

        pending = asyncio.create_task(
            app._approval_callback("shell_execute", {"command": "echo 1"}),
        )
        await asyncio.sleep(0)

        chat.add_approval_prompt.assert_called_once()
        prompt_id, tool_name, preview = chat.add_approval_prompt.call_args.args
        assert prompt_id.startswith("approval:")
        assert tool_name == "shell_execute"
        assert preview == "echo 1"

        callbacks[0]("approve")
        result = await asyncio.wait_for(pending, timeout=1)

        assert result == ApprovalDecision.APPROVE
        chat.clear_info_line.assert_called_once_with(prompt_id)

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

    def test_set_button_selected_uses_selector_only_query_signature(self):
        screen = AskUserScreen(
            "Pick one:",
            question_type="single_choice",
            option_items=[
                {"id": "k12", "label": "K-12", "description": ""},
                {"id": "higher_ed", "label": "Higher Ed", "description": ""},
            ],
            allow_custom_response=False,
            return_payload=True,
        )
        screen._button_to_option_id = {
            "ask-user-option-1": "k12",
            "ask-user-option-2": "higher_ed",
        }

        button1 = Button("K-12", id="ask-user-option-1")
        button2 = Button("Higher Ed", id="ask-user-option-2")
        calls: list[object] = []

        def _query(selector):
            calls.append(selector)
            return [button1, button2]

        screen.query = _query
        screen._set_button_selected("k12", True)

        assert calls == [".ask-user-option-btn"]
        assert button1.has_class("selected")
        assert not button2.has_class("selected")

    def test_prunes_redundant_custom_option_when_custom_input_enabled(self):
        screen = AskUserScreen(
            "Pick one:",
            question_type="single_choice",
            option_items=[
                {"id": "k12", "label": "K-12", "description": ""},
                {"id": "custom", "label": "Custom/Other", "description": ""},
            ],
            allow_custom_response=True,
            return_payload=True,
        )
        option_ids = [str(item.get("id", "")) for item in screen._option_items]
        assert option_ids == ["k12"]
        assert screen._show_input is True

    def test_keeps_non_redundant_custom_prefixed_option(self):
        screen = AskUserScreen(
            "Pick one:",
            question_type="single_choice",
            option_items=[
                {"id": "open_source", "label": "Open-source", "description": ""},
                {"id": "custom_ml", "label": "Custom ML models", "description": ""},
            ],
            allow_custom_response=True,
            return_payload=True,
        )
        option_ids = [str(item.get("id", "")) for item in screen._option_items]
        assert option_ids == ["open_source", "custom_ml"]

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
