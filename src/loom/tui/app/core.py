"""Loom TUI — the unified interactive cowork interface.

Full-featured command center with sidebar, tabbed content area, rich
tool call rendering, session persistence (SQLite), conversation recall,
task delegation, and a polished dark theme. No server required —
runs CoworkSession directly.

This is the default interface launched by ``loom`` with no subcommand.

Layout:
  +-----+----------------------------------------------+
  | S   | [Chat]  [Files Changed]  [Events]            |
  | I   |                                              |
  | D   |  > user message                              |
  | E   |  tool_call  args       ok 12ms 45 lines      |
  | B   |  Model response text ...                     |
  | A   |                                              |
  | R   |  --- 3 tools | 1,247 tokens | model ---      |
  +-----+----------------------------------------------+
  | [>] Input bar                 Ready | ws | 3.2k    |
  +-----+----------------------------------------------+
  | ctrl + b Sidebar  ctrl + l Clear  ctrl + p Commands  ctrl + c Quit |
  +-----+----------------------------------------------+
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
import time  # noqa: F401 - compatibility for tests monkeypatching loom.tui.app.time
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import (
    Button,
    DirectoryTree,
    Header,
    Input,
    TabbedContent,
)

from loom.cowork.approval import ApprovalDecision, ToolApprover
from loom.cowork.session import (
    CoworkSession,
    CoworkTurn,
    ToolCallEvent,
)
from loom.events.types import TelemetryMode
from loom.events.verbosity import DEFAULT_TELEMETRY_MODE, normalize_telemetry_mode
from loom.models.base import ModelProvider
from loom.models.retry import ModelRetryPolicy
from loom.tools.registry import ToolRegistry
from loom.tui.commands import LoomCommands
from loom.tui.screens import (
    AuthManagerScreen,
    FileViewerScreen,
    MCPManagerScreen,
)
from loom.tui.widgets import (
    ActivityIndicator,
    ChatLog,
    FilesChangedPanel,
)

from . import actions as app_actions
from . import command_palette as app_command_palette
from . import files_panel as app_files_panel
from . import info_models as app_info_models
from . import info_views as app_info_views
from . import lifecycle as app_lifecycle
from . import manager_tabs as app_manager_tabs
from . import model_roles as app_model_roles
from . import runtime_config as app_runtime_config
from . import state_init as app_state_init
from . import tool_binding as app_tool_binding
from . import ui_events as app_ui_events
from . import workspace_watch as app_workspace_watch
from .chat import approval as chat_approval
from .chat import delegate_progress as chat_delegate_progress
from .chat import history as chat_history
from .chat import input_submission as chat_input_submission
from .chat import learned as chat_learned
from .chat import session as chat_session
from .chat import steering as chat_steering
from .chat import turns as chat_turns
from .constants import *  # noqa: F403
from .constants import _INFO_WRAP_WIDTH
from .models import (
    ProcessRunLaunchRequest,
    ProcessRunState,
    SlashCommandSpec,
    SteeringDirective,
)
from .process_runs import adhoc as process_run_adhoc
from .process_runs import auth as process_run_auth
from .process_runs import controls as process_run_controls
from .process_runs import definition as process_run_definition
from .process_runs import events as process_run_events
from .process_runs import launch as process_run_launch
from .process_runs import lifecycle as process_run_lifecycle
from .process_runs import questions as process_run_questions
from .process_runs import rendering as process_run_rendering
from .process_runs import ui_state as process_run_ui_state
from .process_runs import workspace as process_run_workspace
from .slash import completion as slash_completion
from .slash import handlers as slash_handlers
from .slash import hints as slash_hints
from .slash import input_history as slash_input_history
from .slash import parsing as slash_parsing
from .slash import process_catalog as slash_process_catalog
from .slash import tooling as slash_tooling

if TYPE_CHECKING:
    from loom.config import Config
    from loom.processes.schema import ProcessDefinition
    from loom.state.conversation_store import ConversationStore
    from loom.state.memory import Database

logger = logging.getLogger(__name__)


class LoomApp(App):
    """Loom TUI — the unified interactive cowork interface."""

    TITLE = "Loom"
    COMMANDS = {LoomCommands}
    _MCP_MANAGER_TAB_ID = "tab-mcp-manager"
    _AUTH_MANAGER_TAB_ID = "tab-auth-manager"

    CSS_PATH = str(Path(__file__).with_name("app.css"))

    BINDINGS = [
        Binding(
            "ctrl+c",
            "request_quit",
            "Quit",
            show=True,
            key_display="ctrl + c",
            priority=True,
        ),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", show=True, key_display="ctrl + b"),
        Binding("ctrl+l", "clear_chat", "Clear", show=True, key_display="ctrl + l"),
        Binding("ctrl+r", "reload_workspace", "Reload", show=True, key_display="ctrl + r"),
        Binding("ctrl+p", "command_palette", "Commands", show=False, key_display="ctrl + p"),
        Binding("ctrl+w", "close_process_tab", "Close Tab", show=True, key_display="ctrl + w"),
        Binding("ctrl+1", "tab_chat", "Chat"),
        Binding("ctrl+2", "tab_files", "Files"),
        Binding("ctrl+3", "tab_events", "Events"),
        Binding("ctrl+a", "open_auth_tab", "Auth", show=False, priority=True),
        Binding("ctrl+m", "open_mcp_tab", "MCP", show=False, priority=True),
    ]

    def __init__(
        self,
        model: ModelProvider | None,
        tools: ToolRegistry,
        workspace: Path,
        *,
        config: Config | None = None,
        db: Database | None = None,
        store: ConversationStore | None = None,
        resume_session: str | None = None,
        explicit_mcp_path: Path | None = None,
        legacy_config_path: Path | None = None,
        explicit_auth_path: Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        app_state_init.initialize_app_state(
            self,
            model=model,
            tools=tools,
            workspace=workspace,
            config=config,
            db=db,
            store=store,
            resume_session=resume_session,
            explicit_mcp_path=explicit_mcp_path,
            legacy_config_path=legacy_config_path,
            explicit_auth_path=explicit_auth_path,
        )

    def _mount_header_activity_indicator(self) -> None:
        """Mount the header activity indicator immediately left of the clock."""
        try:
            header = self.query_one("#app-header", Header)
        except Exception:
            return
        try:
            header.query_one("#header-activity-indicator")
            return
        except Exception:
            pass

        indicator = ActivityIndicator(id="header-activity-indicator")
        clock_widget = None
        for child in list(header.children):
            if type(child).__name__ in {"HeaderClock", "HeaderClockSpace"}:
                clock_widget = child
                break
        try:
            if clock_widget is not None:
                header.mount(indicator, before=clock_widget)
            else:
                header.mount(indicator)
        except Exception:
            return

    def _has_unfinalized_delegate_streams(self) -> bool:
        return chat_steering.has_unfinalized_delegate_streams(self)

    def _is_background_work_active(self) -> bool:
        return chat_steering.is_background_work_active(self)

    def _is_cowork_stop_visible(self) -> bool:
        return chat_steering.is_cowork_stop_visible(self)

    def _has_input_text(self) -> bool:
        return chat_steering.has_input_text(self)

    def _has_pending_inject(self) -> bool:
        return chat_steering.has_pending_inject(self)

    def _pending_inject_age_seconds(self) -> float:
        return chat_steering.pending_inject_age_seconds(self)

    def _pending_inject_count(self) -> int:
        return chat_steering.pending_inject_count(self)

    def _steer_queue_signature(self) -> tuple[tuple[str, str, str], ...]:
        return chat_steering.steer_queue_signature(self)

    def _current_steering_hint_text(self) -> str:
        return chat_steering.current_steering_hint_text(self)

    def _should_show_steer_queue_popup(self) -> bool:
        return chat_steering.should_show_steer_queue_popup(self)

    def _render_steer_queue_popup(self) -> str:
        return chat_steering.render_steer_queue_popup(self)

    def _render_steer_queue_rows(self) -> None:
        chat_steering.render_steer_queue_rows(self)

    def _refresh_hint_panel(self) -> None:
        chat_steering.refresh_hint_panel(self)

    def _sync_chat_stop_control(self) -> None:
        chat_steering.sync_chat_stop_control(self)

    def _sync_activity_indicator(self) -> None:
        """Sync header activity indicator animation state."""
        try:
            indicator = self.query_one(
                "#header-activity-indicator",
                ActivityIndicator,
            )
            indicator.set_active(self._is_background_work_active())
        except Exception:
            pass
        self._sync_chat_stop_control()
        self._refresh_hint_panel()

    def _tui_startup_landing_enabled(self) -> bool:
        tui_cfg = getattr(self._config, "tui", None)
        if tui_cfg is None:
            return True
        try:
            value = getattr(tui_cfg, "startup_landing_enabled", True)
            if isinstance(value, bool):
                return value
            return True
        except Exception:
            return True

    def _tui_always_open_chat_directly(self) -> bool:
        tui_cfg = getattr(self._config, "tui", None)
        if tui_cfg is None:
            return False
        try:
            value = getattr(tui_cfg, "always_open_chat_directly", False)
            if isinstance(value, bool):
                return value
            return False
        except Exception:
            return False

    def _should_show_startup_landing(self, *, resume_target: str | None) -> bool:
        """Return True when startup should show the landing surface."""
        if self._store is None:
            return False
        if self._tui_always_open_chat_directly():
            return False
        if not self._tui_startup_landing_enabled():
            return False
        return not bool(str(resume_target or "").strip())

    def _landing_model_display_name(self) -> str:
        return app_lifecycle.landing_model_display_name(self)

    def _sync_landing_surface(self) -> None:
        app_lifecycle.sync_landing_surface(self)

    def _set_startup_surface(self, *, show_landing: bool) -> None:
        app_lifecycle.set_startup_surface(self, show_landing=show_landing)

    async def _enter_workspace_surface(self, *, ensure_session: bool) -> None:
        return await app_lifecycle.enter_workspace_surface(
            self,
            ensure_session=ensure_session,
        )

    def compose(self) -> ComposeResult:
        yield from app_lifecycle.compose(self)

    async def on_mount(self) -> None:
        return await app_lifecycle.on_mount(self)

    async def _monitor_event_loop_lag(self) -> None:
        return await app_lifecycle.monitor_event_loop_lag(self)

    def on_unmount(self) -> None:
        """Cleanup runtime timers when app unmounts."""
        self._stop_workspace_watch()
        timer = self._process_elapsed_timer
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._process_elapsed_timer = None

    def _on_setup_complete(self, result: list[dict] | None) -> None:
        app_lifecycle.on_setup_complete(self, result)

    @work
    async def _finalize_setup(self) -> None:
        return await app_lifecycle.finalize_setup(self)

    def _refresh_tool_registry(self) -> None:
        """Reset registry to discovered tools."""
        from loom.tools import create_default_registry

        self._tools = create_default_registry(
            self._config,
            mcp_startup_mode="background",
        )

    def _create_process_loader(self):
        return process_run_definition.create_process_loader(self)

    def _active_process_name(self) -> str:
        return process_run_definition.active_process_name(self)

    def _mcp_manager(self):
        """Build MCP config manager scoped to current app workspace."""
        from loom.mcp.config import MCPConfigManager

        return MCPConfigManager(
            # Use file-backed MCP layers to avoid stale in-memory alias snapshots.
            config=None,
            workspace=self._workspace,
            explicit_path=self._explicit_mcp_path,
            legacy_config_path=self._legacy_config_path,
        )

    def _tab_exists(self, tabs: TabbedContent, pane_id: str) -> bool:
        return app_manager_tabs.tab_exists(self, tabs, pane_id)

    async def _remove_tab_if_present(
        self,
        pane_id: str,
        *,
        fallback_active: str = "tab-chat",
    ) -> None:
        return await app_manager_tabs.remove_tab_if_present(
            self,
            pane_id,
            fallback_active=fallback_active,
        )

    def _announce_user_feedback(
        self,
        message: str,
        *,
        chat_line: bool = False,
        severity: str = "information",
        timeout: int = 3,
    ) -> None:
        app_manager_tabs.announce_user_feedback(
            self,
            message,
            chat_line=chat_line,
            severity=severity,
            timeout=timeout,
        )

    def _handle_mcp_manager_tab_close(self, result: dict[str, object]) -> None:
        app_manager_tabs.handle_mcp_manager_tab_close(self, result)

    async def _open_mcp_manager_tab(self) -> None:
        return await app_manager_tabs.open_mcp_manager_tab(
            self,
            mcp_manager_screen_cls=MCPManagerScreen,
        )

    def _open_mcp_manager_screen(self) -> None:
        app_manager_tabs.open_mcp_manager_screen(self)

    def _handle_auth_manager_tab_close(self, result: dict[str, object]) -> None:
        app_manager_tabs.handle_auth_manager_tab_close(self, result)

    async def _open_auth_manager_tab(self) -> None:
        return await app_manager_tabs.open_auth_manager_tab(
            self,
            auth_manager_screen_cls=AuthManagerScreen,
        )

    def _open_auth_manager_screen(self) -> None:
        app_manager_tabs.open_auth_manager_screen(self)

    def _auth_discovery_process_defs(self) -> list[ProcessDefinition]:
        return app_manager_tabs.auth_discovery_process_defs(self)

    def _auth_defaults_path(self) -> Path:
        return app_manager_tabs.auth_defaults_path(self)

    @staticmethod
    def _split_slash_args(raw: str) -> list[str]:
        """Split slash-command argument string using shell-like quoting."""
        return slash_parsing.split_slash_args(raw)

    @staticmethod
    def _split_slash_args_forgiving(raw: str) -> list[str]:
        """Best-effort tokenizer that never fails on unmatched quote punctuation."""
        return slash_parsing.split_slash_args_forgiving(raw)

    @staticmethod
    def _split_tool_slash_args(raw: str) -> tuple[str, str]:
        """Split `/tool` args into (tool_name, raw_json_args)."""
        return slash_parsing.split_tool_slash_args(raw)

    @staticmethod
    def _parse_tool_kv_value(raw_value: str) -> Any:
        """Parse one `/tool` key=value literal into a typed JSON-compatible value."""
        return slash_parsing.parse_tool_kv_value(raw_value)

    def _parse_tool_slash_arguments(self, raw: str) -> tuple[dict[str, Any] | None, str]:
        """Parse `/tool` arguments from either JSON object or key=value pairs."""
        return slash_parsing.parse_tool_slash_arguments(raw)

    def _tool_name_inventory(self) -> list[str]:
        return slash_tooling.tool_name_inventory(self)

    def _tool_description(self, tool_name: str) -> str:
        return slash_tooling.tool_description(self, tool_name)

    def _tool_parameters_schema(self, tool_name: str) -> dict[str, Any]:
        return slash_tooling.tool_parameters_schema(self, tool_name)

    @staticmethod
    def _tool_argument_lists(parameters: dict[str, Any]) -> tuple[list[str], list[str]]:
        return slash_tooling.tool_argument_lists(parameters)

    @staticmethod
    def _tool_argument_placeholder(schema: dict[str, Any]) -> str:
        return slash_tooling.tool_argument_placeholder(schema)

    def _tool_argument_example(self, tool_name: str, *, max_fields: int = 4) -> str:
        return slash_tooling.tool_argument_example(
            self,
            tool_name,
            max_fields=max_fields,
        )

    def _tool_argument_summary(self, tool_name: str) -> tuple[str, str]:
        return slash_tooling.tool_argument_summary(self, tool_name)

    async def _execute_slash_tool_command(
        self,
        resolved_tool_name: str,
        tool_args: dict[str, Any],
    ) -> None:
        return await slash_tooling.execute_slash_tool_command(
            self,
            resolved_tool_name,
            tool_args,
        )

    @staticmethod
    def _truncate_run_goal_file_content(content: str) -> tuple[str, bool]:
        return process_run_launch.truncate_run_goal_file_content(content)

    def _resolve_run_goal_file_path(self, raw_path: str) -> Path | None:
        return process_run_launch.resolve_run_goal_file_path(self, raw_path)

    def _expand_run_goal_file_input(
        self,
        goal_tokens: list[str],
    ) -> tuple[str, str, dict[str, Any], str | None]:
        return process_run_launch.expand_run_goal_file_input(self, goal_tokens)

    @staticmethod
    def _parse_kv_assignments(
        values: list[str],
        *,
        option_name: str,
        env_keys: bool = False,
    ) -> dict[str, str]:
        """Parse repeated KEY=VALUE assignments."""
        result: dict[str, str] = {}
        if env_keys:
            from loom.mcp.config import ensure_valid_env_key

        for value in values:
            raw = str(value or "").strip()
            if "=" not in raw:
                raise ValueError(f"{option_name} expects KEY=VALUE entries.")
            key, item = raw.split("=", 1)
            clean_key = key.strip()
            if env_keys:
                clean_key = ensure_valid_env_key(clean_key)
            if not clean_key:
                raise ValueError(f"{option_name} key cannot be empty.")
            result[clean_key] = item
        return result

    async def _reload_mcp_runtime(self) -> None:
        """Reload merged MCP config and reconcile MCP tools in registry."""
        if self._config is None:
            return

        from loom.integrations.mcp_tools import register_mcp_tools

        manager = self._mcp_manager()
        merged = await asyncio.to_thread(manager.load)
        self._config = replace(self._config, mcp=merged.config)
        await asyncio.to_thread(
            register_mcp_tools,
            self._tools,
            mcp_config=merged.config,
        )

    @staticmethod
    def _adhoc_cache_key(*args, **kwargs):
        return process_run_adhoc.adhoc_cache_key(*args, **kwargs)

    @staticmethod
    def _adhoc_cache_dir(*args, **kwargs):
        return process_run_adhoc.adhoc_cache_dir(*args, **kwargs)

    def _adhoc_synthesis_log_path(self, *args, **kwargs):
        return process_run_adhoc.adhoc_synthesis_log_path(self, *args, **kwargs)

    def _adhoc_synthesis_artifact_root(self, *args, **kwargs):
        return process_run_adhoc.adhoc_synthesis_artifact_root(self, *args, **kwargs)

    def _create_adhoc_synthesis_artifact_dir(self, *args, **kwargs):
        return process_run_adhoc.create_adhoc_synthesis_artifact_dir(self, *args, **kwargs)

    @staticmethod
    def _write_adhoc_synthesis_artifact_text(*args, **kwargs):
        return process_run_adhoc.write_adhoc_synthesis_artifact_text(*args, **kwargs)

    @staticmethod
    def _write_adhoc_synthesis_artifact_yaml(*args, **kwargs):
        return process_run_adhoc.write_adhoc_synthesis_artifact_yaml(*args, **kwargs)

    def _append_adhoc_synthesis_log(self, *args, **kwargs):
        return process_run_adhoc.append_adhoc_synthesis_log(self, *args, **kwargs)

    def _adhoc_cache_path(self, *args, **kwargs):
        return process_run_adhoc.adhoc_cache_path(self, *args, **kwargs)

    def _adhoc_legacy_cache_path(self, *args, **kwargs):
        return process_run_adhoc.adhoc_legacy_cache_path(self, *args, **kwargs)

    @classmethod
    def _spec_from_process_defn(
        cls,
        process_defn: ProcessDefinition,
        *,
        recommended_tools: list[str],
    ) -> dict[str, Any]:
        return process_run_adhoc.spec_from_process_defn(
            cls,
            process_defn,
            recommended_tools=recommended_tools,
        )

    def _persist_adhoc_cache_entry(self, *args, **kwargs):
        return process_run_adhoc.persist_adhoc_cache_entry(self, *args, **kwargs)

    def _load_adhoc_cache_entry_from_disk(self, *args, **kwargs):
        return process_run_adhoc.load_adhoc_cache_entry_from_disk(self, *args, **kwargs)

    @staticmethod
    def _sanitize_synthesis_trace(*args, **kwargs):
        return process_run_adhoc.sanitize_synthesis_trace(*args, **kwargs)

    @staticmethod
    def _sanitize_kebab_token(value: str, *, fallback: str, max_len: int = 48) -> str:
        return process_run_adhoc.sanitize_kebab_token(
            value,
            fallback=fallback,
            max_len=max_len,
        )

    @staticmethod
    def _sanitize_deliverable_name(value: str, *, fallback: str) -> str:
        return process_run_adhoc.sanitize_deliverable_name(
            value,
            fallback=fallback,
        )

    def _available_tool_names(self) -> list[str]:
        return process_run_adhoc.available_tool_names(self)

    def _adhoc_package_contract_hint(self, *args, **kwargs):
        return process_run_adhoc.adhoc_package_contract_hint(self, *args, **kwargs)

    @staticmethod
    def _extract_json_payload(
        raw_text: str,
        *,
        expected_keys: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        return process_run_adhoc.extract_json_payload(
            raw_text,
            expected_keys=expected_keys,
        )

    @staticmethod
    def _synthesis_preview(*args, **kwargs):
        return process_run_adhoc.synthesis_preview(*args, **kwargs)

    @staticmethod
    def _raw_adhoc_spec_needs_minimal_retry(*args, **kwargs):
        return process_run_adhoc.raw_adhoc_spec_needs_minimal_retry(*args, **kwargs)

    @staticmethod
    def _normalize_adhoc_intent(*args, **kwargs):
        return process_run_adhoc.normalize_adhoc_intent(*args, **kwargs)

    @staticmethod
    def _normalize_adhoc_risk_level(*args, **kwargs):
        return process_run_adhoc.normalize_adhoc_risk_level(*args, **kwargs)

    @classmethod
    def _infer_adhoc_intent_from_phases(cls, *args, **kwargs):
        return process_run_adhoc.infer_adhoc_intent_from_phases(cls, *args, **kwargs)

    @classmethod
    def _resolve_adhoc_intent(cls, *args, **kwargs):
        return process_run_adhoc.resolve_adhoc_intent(cls, *args, **kwargs)

    @classmethod
    def _resolve_adhoc_risk_level(cls, *args, **kwargs):
        return process_run_adhoc.resolve_adhoc_risk_level(cls, *args, **kwargs)

    @staticmethod
    def _adhoc_intent_progression(*args, **kwargs):
        return process_run_adhoc.adhoc_intent_progression(*args, **kwargs)

    @staticmethod
    def _adhoc_default_validity_contract(*args, **kwargs):
        return process_run_adhoc.adhoc_default_validity_contract(*args, **kwargs)

    @staticmethod
    def _adhoc_default_verification_policy(*args, **kwargs):
        return process_run_adhoc.adhoc_default_verification_policy(*args, **kwargs)

    @staticmethod
    def _merge_adhoc_verification_policy(*args, **kwargs):
        return process_run_adhoc.merge_adhoc_verification_policy(*args, **kwargs)

    @staticmethod
    def _adhoc_intent_phase_blueprint(*args, **kwargs):
        return process_run_adhoc.adhoc_intent_phase_blueprint(*args, **kwargs)

    @staticmethod
    def _phases_satisfy_intent(phases: list[dict[str, Any]], intent: str) -> bool:
        return process_run_adhoc.phases_satisfy_intent(phases, intent)

    def _fallback_adhoc_spec(self, *args, **kwargs):
        return process_run_adhoc.fallback_adhoc_spec(self, *args, **kwargs)

    def _normalize_adhoc_spec(self, *args, **kwargs):
        return process_run_adhoc.normalize_adhoc_spec(self, *args, **kwargs)

    def _build_adhoc_cache_entry(self, *args, **kwargs):
        return process_run_adhoc.build_adhoc_cache_entry(self, *args, **kwargs)

    def _is_template_like_adhoc_spec(self, *args, **kwargs):
        return process_run_adhoc.is_template_like_adhoc_spec(self, *args, **kwargs)

    def _adhoc_synthesis_activity_lines(self, *args, **kwargs):
        return process_run_adhoc.adhoc_synthesis_activity_lines(self, *args, **kwargs)

    @staticmethod
    def _is_temperature_one_only_error(value: object) -> bool:
        return process_run_adhoc.is_temperature_one_only_error(value)

    @staticmethod
    def _configured_model_temperature(model: ModelProvider | None) -> float | None:
        return app_model_roles.configured_model_temperature(model)

    @staticmethod
    def _configured_model_max_tokens(model: ModelProvider | None) -> int | None:
        return app_model_roles.configured_model_max_tokens(model)

    def _planning_response_max_tokens_limit(self) -> int | None:
        return app_model_roles.planning_response_max_tokens_limit(self)

    def _has_configured_role_model(self, role: str) -> bool:
        return app_model_roles.has_configured_role_model(self, role)

    def _cowork_compactor_model(self) -> ModelProvider | None:
        return app_model_roles.cowork_compactor_model(self)

    def _cowork_memory_indexer_model(self) -> tuple[ModelProvider | None, str]:
        return app_model_roles.cowork_memory_indexer_model(self)

    def _select_helper_model_for_role(
        self,
        *,
        role: str,
        tier: int,
    ) -> tuple[ModelProvider | None, object | None]:
        return app_model_roles.select_helper_model_for_role(
            self,
            role=role,
            tier=tier,
        )

    async def _invoke_helper_role_completion(
        self,
        *,
        role: str,
        tier: int,
        prompt: str,
        max_tokens: int | None,
        temperature: float | None = None,
    ) -> tuple[object, str, float | None, int | None]:
        return await app_model_roles.invoke_helper_role_completion(
            self,
            role=role,
            tier=tier,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _should_resynthesize_cached_adhoc(self, *args, **kwargs):
        return process_run_adhoc.should_resynthesize_cached_adhoc(self, *args, **kwargs)

    async def _synthesize_adhoc_process(self, *args, **kwargs):
        return await process_run_adhoc.synthesize_adhoc_process(self, *args, **kwargs)

    async def _get_or_create_adhoc_process(self, *args, **kwargs):
        return await process_run_adhoc.get_or_create_adhoc_process(self, *args, **kwargs)

    @staticmethod
    def _serialize_process_for_package(process_defn: ProcessDefinition) -> dict[str, Any]:
        return process_run_adhoc.serialize_process_for_package(process_defn)

    def _save_adhoc_process_package(self, *args, **kwargs):
        return process_run_adhoc.save_adhoc_process_package(self, *args, **kwargs)

    def _has_active_process_runs(self) -> bool:
        """Return True when at least one run is still active."""
        return any(
            self._is_process_run_busy_status(run.status)
            for run in self._process_runs.values()
        )

    @staticmethod
    def _reserved_slash_command_names() -> set[str]:
        return slash_process_catalog.reserved_slash_command_names()

    def _is_reserved_process_name(self, name: str) -> bool:
        return slash_process_catalog.is_reserved_process_name(self, name)

    def _compute_process_command_index(
        self,
    ) -> tuple[list[dict[str, str]], dict[str, str], list[str]]:
        return slash_process_catalog.compute_process_command_index(self)

    def _refresh_process_command_index(
        self,
        *,
        chat: ChatLog | None = None,
        notify_conflicts: bool = False,
        background: bool = False,
        force: bool = False,
    ) -> None:
        slash_process_catalog.refresh_process_command_index(
            self,
            chat=chat,
            notify_conflicts=notify_conflicts,
            background=background,
            force=force,
        )

    @staticmethod
    def _escape_markup(value: object | None) -> str:
        """Escape Rich markup control chars in user/content-provided text."""
        if value is None:
            return ""
        return str(value).replace("[", "\\[")

    @staticmethod
    def _wrap_info_text(
        text: str,
        *,
        initial_indent: str = "",
        subsequent_indent: str = "",
    ) -> str:
        """Wrap long informational text for chat readability."""
        if not text:
            return ""
        return textwrap.fill(
            " ".join(text.split()),
            width=_INFO_WRAP_WIDTH,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
        )

    def _render_process_usage(self) -> str:
        return app_info_views.render_process_usage()

    def _render_tools_catalog(self) -> str:
        return app_info_views.render_tools_catalog(self)

    def _configured_models(self) -> dict[str, Any]:
        return app_info_models.configured_models(self)

    @staticmethod
    def _normalize_provider_name(provider: object | None) -> str:
        return app_info_models.normalize_provider_name(provider)

    @staticmethod
    def _protocol_for_provider(provider: object | None) -> str:
        return app_info_models.protocol_for_provider(provider)

    @staticmethod
    def _sanitize_endpoint_url(raw_url: object | None) -> str:
        return app_info_models.sanitize_endpoint_url(raw_url)

    def _endpoint_for_config(self, provider: object | None, base_url: object | None) -> str:
        return app_info_models.endpoint_for_config(self, provider, base_url)

    def _runtime_model_provider(self, model: ModelProvider | None) -> str:
        return app_info_models.runtime_model_provider(self, model)

    def _runtime_model_id(self, model: ModelProvider | None) -> str:
        return app_info_models.runtime_model_id(model)

    def _runtime_model_roles(self, model: ModelProvider | None) -> list[str]:
        return app_info_models.runtime_model_roles(model)

    def _runtime_model_endpoint(self, model: ModelProvider | None) -> str:
        return app_info_models.runtime_model_endpoint(self, model)

    @staticmethod
    def _runtime_model_tier(model: ModelProvider | None) -> int | None:
        return app_info_models.runtime_model_tier(model)

    def _infer_tier_from_config(
        self,
        provider: object | None,
        model_id: object | None,
    ) -> int | None:
        return app_info_models.infer_tier_from_config(self, provider, model_id)

    @staticmethod
    def _format_tier_label(explicit_tier: object | None, inferred_tier: int | None) -> str:
        return app_info_models.format_tier_label(explicit_tier, inferred_tier)

    @staticmethod
    def _format_temperature(value: object | None) -> str:
        return app_info_models.format_temperature(value)

    @staticmethod
    def _format_max_tokens(value: object | None) -> str:
        return app_info_models.format_max_tokens(value)

    @staticmethod
    def _format_model_roles(roles: object | None) -> str:
        return app_info_models.format_model_roles(roles)

    @staticmethod
    def _format_capabilities(capabilities: object | None) -> str:
        return app_info_models.format_capabilities(capabilities)

    def _resolve_active_model_alias(self) -> tuple[str | None, list[str]]:
        return app_info_views.resolve_active_model_alias(self)

    def _render_model_block(
        self,
        *,
        alias: str,
        active: bool,
        provider: str,
        endpoint: str,
        model_id: str,
        roles: object | None,
        tier_label: str,
        temperature: object | None,
        max_tokens: object | None,
        reasoning_effort: object | None,
        capabilities: object | None,
    ) -> str:
        return app_info_views.render_model_block(
            self,
            alias=alias,
            active=active,
            provider=provider,
            endpoint=endpoint,
            model_id=model_id,
            roles=roles,
            tier_label=tier_label,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            capabilities=capabilities,
        )

    def _render_configured_model_block(
        self,
        alias: str,
        cfg: Any,
        *,
        active: bool,
        runtime_model: ModelProvider | None = None,
    ) -> str:
        return app_info_views.render_configured_model_block(
            self,
            alias,
            cfg,
            active=active,
            runtime_model=runtime_model,
        )

    def _render_runtime_model_block(
        self,
        model: ModelProvider,
        *,
        alias_override: str,
        active: bool,
    ) -> str:
        return app_info_views.render_runtime_model_block(
            self,
            model,
            alias_override=alias_override,
            active=active,
        )

    def _render_active_model_info(self) -> str:
        return app_info_views.render_active_model_info(self)

    def _render_models_catalog(self) -> str:
        return app_info_views.render_models_catalog(self)

    def _render_session_info(self, state) -> str:
        return app_info_views.render_session_info(self, state)

    def _render_sessions_list(self, sessions: list[dict]) -> str:
        return app_info_views.render_sessions_list(self, sessions)

    def _render_startup_summary(self, *, tool_count: int, persisted: str) -> str:
        return app_info_views.render_startup_summary(
            self,
            tool_count=tool_count,
            persisted=persisted,
        )

    def _render_process_catalog(self) -> str:
        return app_info_views.render_process_catalog(self)

    @staticmethod
    def _render_mcp_list(views: list) -> str:
        return app_info_views.render_mcp_list(views)

    @staticmethod
    def _render_mcp_view(view) -> str:
        return app_info_views.render_mcp_view(view)

    def _apply_process_tool_policy(self, chat: ChatLog) -> None:
        process_run_definition.apply_process_tool_policy(self, chat)

    def _build_system_prompt(self) -> str:
        return process_run_definition.build_system_prompt(self)

    def _model_retry_policy(self) -> ModelRetryPolicy:
        return app_runtime_config.model_retry_policy(self)

    def _cowork_tool_exposure_mode(self) -> str:
        return app_runtime_config.cowork_tool_exposure_mode(self)

    def _cowork_memory_index_enabled(self) -> bool:
        return app_runtime_config.cowork_memory_index_enabled(self)

    def _cowork_memory_index_v2_actions_enabled(self) -> bool:
        return app_runtime_config.cowork_memory_index_v2_actions_enabled(self)

    def _cowork_memory_index_force_fts(self) -> bool:
        return app_runtime_config.cowork_memory_index_force_fts(self)

    def _cowork_indexer_model_role_strict(self) -> bool:
        return app_runtime_config.cowork_indexer_model_role_strict(self)

    def _cowork_memory_index_llm_extraction_enabled(self) -> bool:
        return app_runtime_config.cowork_memory_index_llm_extraction_enabled(self)

    def _cowork_memory_index_queue_max_batches(self) -> int:
        return app_runtime_config.cowork_memory_index_queue_max_batches(self)

    def _cowork_memory_index_section_limit(self) -> int:
        return app_runtime_config.cowork_memory_index_section_limit(self)

    def _cowork_recall_index_max_chars(self) -> int:
        return app_runtime_config.cowork_recall_index_max_chars(self)

    def _cowork_max_context_tokens(self) -> int:
        return app_runtime_config.cowork_max_context_tokens(self)

    def _cowork_scratch_dir(self) -> Path | None:
        return app_runtime_config.cowork_scratch_dir(self)

    def _cowork_enable_filetype_ingest_router(self) -> bool:
        return app_runtime_config.cowork_enable_filetype_ingest_router(self)

    def _cowork_ingest_artifact_retention_max_age_days(self) -> int:
        return app_runtime_config.cowork_ingest_artifact_retention_max_age_days(self)

    def _cowork_ingest_artifact_retention_max_files_per_scope(self) -> int:
        return app_runtime_config.cowork_ingest_artifact_retention_max_files_per_scope(self)

    def _cowork_ingest_artifact_retention_max_bytes_per_scope(self) -> int:
        return app_runtime_config.cowork_ingest_artifact_retention_max_bytes_per_scope(self)

    def _chat_resume_page_size(self) -> int:
        return app_runtime_config.chat_resume_page_size(self)

    def _chat_resume_max_rendered_rows(self) -> int:
        return app_runtime_config.chat_resume_max_rendered_rows(self)

    def _chat_resume_use_event_journal(self) -> bool:
        return app_runtime_config.chat_resume_use_event_journal(self)

    def _chat_resume_enable_legacy_fallback(self) -> bool:
        return app_runtime_config.chat_resume_enable_legacy_fallback(self)

    def _tui_progress_auto_follow(self) -> bool:
        return app_runtime_config.tui_progress_auto_follow(self)

    def _tui_realtime_refresh_enabled(self) -> bool:
        return app_runtime_config.tui_realtime_refresh_enabled(self)

    def _tui_workspace_watch_backend(self) -> str:
        return app_runtime_config.tui_workspace_watch_backend(self)

    def _tui_workspace_poll_interval_seconds(self) -> float:
        return app_runtime_config.tui_workspace_poll_interval_seconds(self)

    def _tui_workspace_refresh_debounce_seconds(self) -> float:
        return app_runtime_config.tui_workspace_refresh_debounce_seconds(self)

    def _tui_workspace_refresh_max_wait_seconds(self) -> float:
        return app_runtime_config.tui_workspace_refresh_max_wait_seconds(self)

    def _tui_workspace_scan_max_entries(self) -> int:
        return app_runtime_config.tui_workspace_scan_max_entries(self)

    def _tui_chat_stream_flush_interval_ms(self) -> int:
        return app_runtime_config.tui_chat_stream_flush_interval_ms(self)

    def _tui_files_panel_max_rows(self) -> int:
        return app_runtime_config.tui_files_panel_max_rows(self)

    def _tui_delegate_progress_max_lines(self) -> int:
        return app_runtime_config.tui_delegate_progress_max_lines(self)

    def _tui_run_launch_heartbeat_interval_seconds(self) -> float:
        return app_runtime_config.tui_run_launch_heartbeat_interval_seconds(self)

    def _tui_run_launch_timeout_seconds(self) -> float:
        return app_runtime_config.tui_run_launch_timeout_seconds(self)

    def _tui_run_close_modal_timeout_seconds(self) -> float:
        return app_runtime_config.tui_run_close_modal_timeout_seconds(self)

    def _tui_run_cancel_wait_timeout_seconds(self) -> float:
        return app_runtime_config.tui_run_cancel_wait_timeout_seconds(self)

    def _tui_run_progress_refresh_interval_seconds(self) -> float:
        return app_runtime_config.tui_run_progress_refresh_interval_seconds(self)

    def _tui_run_preflight_async_enabled(self) -> bool:
        return app_runtime_config.tui_run_preflight_async_enabled(self)

    def _is_mutating_tool(self, tool_name: str) -> bool:
        return app_runtime_config.is_mutating_tool(self, tool_name)

    async def _initialize_session(
        self,
        *,
        startup_resume: tuple[str | None, bool] | None = None,
        allow_auto_resume: bool = True,
        emit_info_messages: bool = True,
    ) -> None:
        return await chat_session.initialize_session(
            self,
            startup_resume=startup_resume,
            allow_auto_resume=allow_auto_resume,
            emit_info_messages=emit_info_messages,
            session_cls=CoworkSession,
            approver_cls=ToolApprover,
        )

    async def _resolve_startup_resume_target(self) -> tuple[str | None, bool]:
        """Resolve resume session for startup: explicit first, then workspace latest."""
        if self._store is None:
            return None, False
        if self._resume_session:
            return self._resume_session, False
        if not self._auto_resume_workspace_on_init:
            return None, False
        try:
            sessions = await self._store.list_sessions(workspace=str(self._workspace))
        except Exception:
            return None, False
        if not sessions:
            return None, False
        session_id = str(sessions[0].get("id", "")).strip()
        if not session_id:
            return None, False
        return session_id, True

    def _new_process_run_id(self) -> str:
        return process_run_ui_state._new_process_run_id(self)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        return process_run_ui_state._format_elapsed(seconds)

    def _process_run_user_input_paused_seconds(
        self,
        run_id: str,
        *,
        now: float | None = None,
    ) -> float:
        return process_run_ui_state._process_run_user_input_paused_seconds(
            self,
            run_id,
            now=now,
        )

    def _begin_process_run_user_input_pause(
        self,
        run_id: str,
        *,
        now: float | None = None,
    ) -> None:
        process_run_ui_state._begin_process_run_user_input_pause(
            self,
            run_id,
            now=now,
        )

    def _end_process_run_user_input_pause(
        self,
        run_id: str,
        *,
        now: float | None = None,
    ) -> None:
        process_run_ui_state._end_process_run_user_input_pause(
            self,
            run_id,
            now=now,
        )

    def _clear_process_run_user_input_pause(self, run_id: str) -> None:
        process_run_ui_state._clear_process_run_user_input_pause(self, run_id)

    def _elapsed_seconds_for_run(self, run: ProcessRunState) -> float:
        return process_run_ui_state._elapsed_seconds_for_run(self, run)

    @staticmethod
    def _is_process_run_busy_status(status: str) -> bool:
        return process_run_ui_state._is_process_run_busy_status(status)

    def _set_process_run_status(self, run: ProcessRunState, status: str) -> None:
        process_run_ui_state._set_process_run_status(self, run, status)

    def _append_process_run_activity(
        self,
        run: ProcessRunState,
        message: str,
    ) -> None:
        process_run_ui_state._append_process_run_activity(self, run, message)

    def _trim_process_run_activity_log(self, run: ProcessRunState) -> None:
        process_run_ui_state._trim_process_run_activity_log(self, run)

    @staticmethod
    def _process_run_stage_activity_key(stage: str) -> str:
        return process_run_ui_state._process_run_stage_activity_key(stage)

    def _upsert_process_run_stage_activity(
        self,
        run: ProcessRunState,
        *,
        stage: str,
        text: str,
    ) -> None:
        process_run_ui_state._upsert_process_run_stage_activity(
            self,
            run,
            stage=stage,
            text=text,
        )

    def _render_process_run_stage_activity_text(
        self,
        stage: str,
        *,
        dots: int,
        duration_seconds: float | None = None,
    ) -> str:
        return process_run_ui_state._render_process_run_stage_activity_text(
            self,
            stage,
            dots=dots,
            duration_seconds=duration_seconds,
        )

    def _start_process_run_stage_activity_line(self, run: ProcessRunState, stage: str) -> None:
        process_run_ui_state._start_process_run_stage_activity_line(self, run, stage)

    def _finalize_process_run_stage_activity_line(
        self,
        run: ProcessRunState,
        *,
        stage: str,
        duration_seconds: float,
    ) -> None:
        process_run_ui_state._finalize_process_run_stage_activity_line(
            self,
            run,
            stage=stage,
            duration_seconds=duration_seconds,
        )

    def _append_process_run_result(
        self,
        run: ProcessRunState,
        text: str,
        *,
        success: bool,
    ) -> None:
        process_run_ui_state._append_process_run_result(
            self,
            run,
            text,
            success=success,
        )

    @staticmethod
    def _process_run_launch_stage_label(stage: str) -> str:
        return process_run_ui_state._process_run_launch_stage_label(stage)

    def _process_run_stage_rows(self, run: ProcessRunState) -> list[dict]:
        return process_run_ui_state._process_run_stage_rows(self, run)

    def _process_run_stage_summary_row(self, run: ProcessRunState) -> dict | None:
        return process_run_ui_state._process_run_stage_summary_row(self, run)

    def _refresh_process_run_progress(self, run: ProcessRunState) -> None:
        process_run_ui_state._refresh_process_run_progress(self, run)

    def _set_process_run_launch_stage(
        self,
        run: ProcessRunState,
        stage: str,
        *,
        note: str = "",
    ) -> None:
        process_run_ui_state._set_process_run_launch_stage(
            self,
            run,
            stage,
            note=note,
        )

    def _fail_process_run_launch(self, run: ProcessRunState, message: str) -> None:
        process_run_ui_state._fail_process_run_launch(self, run, message)

    def _maybe_emit_process_run_heartbeat(self, run: ProcessRunState) -> None:
        process_run_ui_state._maybe_emit_process_run_heartbeat(self, run)

    def _log_terminal_stage_duration(
        self,
        run: ProcessRunState,
        *,
        terminal_state: str,
    ) -> None:
        process_run_ui_state._log_terminal_stage_duration(
            self,
            run,
            terminal_state=terminal_state,
        )

    def _serialize_process_run_state(self, run: ProcessRunState) -> dict:
        return process_run_ui_state._serialize_process_run_state(self, run)

    def _sync_process_runs_into_session_state(self) -> None:
        process_run_ui_state._sync_process_runs_into_session_state(self)

    def _sync_input_history_into_session_state(self) -> None:
        process_run_ui_state._sync_input_history_into_session_state(self)

    async def _persist_process_run_ui_state(
        self,
        *,
        is_active: bool | None = None,
    ) -> None:
        return await process_run_ui_state._persist_process_run_ui_state(
            self,
            is_active=is_active,
        )

    def _persisted_process_tabs_payload(self) -> tuple[list[dict], str]:
        return process_run_ui_state._persisted_process_tabs_payload(self)

    async def _drop_process_run_tabs(self) -> None:
        return await process_run_ui_state._drop_process_run_tabs(self)

    async def _restore_process_run_tabs(self, chat: ChatLog | None = None) -> None:
        return await process_run_ui_state._restore_process_run_tabs(self, chat=chat)

    def _format_process_run_tab_title(self, run: ProcessRunState) -> str:
        return process_run_ui_state._format_process_run_tab_title(self, run)

    def _update_process_run_visuals(self, run: ProcessRunState) -> None:
        process_run_ui_state._update_process_run_visuals(self, run)

    def _process_run_working_folder_label(self, run: ProcessRunState) -> str:
        return process_run_ui_state._process_run_working_folder_label(self, run)

    def _tick_process_run_elapsed(self) -> None:
        process_run_ui_state._tick_process_run_elapsed(self)

    def _build_process_run_context(self, goal: str, *, workspace: Path) -> dict:
        return process_run_workspace.build_process_run_context(
            self,
            goal,
            workspace=workspace,
        )

    @staticmethod
    def _slugify_process_run_folder(value: str, *, max_len: int = 48) -> str:
        return process_run_launch.slugify_process_run_folder(value, max_len=max_len)

    @staticmethod
    def _run_goal_for_folder_name(goal: str) -> str:
        return process_run_launch.run_goal_for_folder_name(goal)

    @classmethod
    def _extract_run_folder_slug(cls, response_text: str) -> str:
        return process_run_launch.extract_run_folder_slug(response_text)

    @staticmethod
    def _is_low_quality_run_folder_slug(slug: str) -> bool:
        return process_run_launch.is_low_quality_run_folder_slug(slug)

    def _fallback_process_run_folder_name(self, process_name: str, goal: str) -> str:
        return process_run_launch.fallback_process_run_folder_name(process_name, goal)

    async def _llm_process_run_folder_name(self, process_name: str, goal: str) -> str:
        return await process_run_workspace.llm_process_run_folder_name(
            self,
            process_name,
            goal,
        )

    async def _prepare_process_run_workspace(
        self,
        process_name: str,
        goal: str,
    ) -> Path:
        return await process_run_workspace.prepare_process_run_workspace(
            self,
            process_name,
            goal,
        )

    def _next_available_process_run_folder_name(self, base_slug: str) -> str:
        return process_run_workspace.next_available_process_run_folder_name(
            self,
            base_slug,
        )

    @staticmethod
    def _normalize_process_run_workspace_selection(raw_value: str) -> str:
        """Normalize a user-selected run workspace relative path."""
        return process_run_launch.normalize_process_run_workspace_selection(raw_value)

    def _materialize_process_run_workspace_selection(self, relative_path: str) -> Path:
        return process_run_workspace.materialize_process_run_workspace_selection(
            self,
            relative_path,
        )

    async def _prompt_process_run_workspace_choice(
        self,
        *,
        run_id: str = "",
        process_name: str,
        suggested_folder: str,
    ) -> str | None:
        return await process_run_workspace.prompt_process_run_workspace_choice(
            self,
            run_id=run_id,
            process_name=process_name,
            suggested_folder=suggested_folder,
        )

    async def _choose_process_run_workspace(
        self,
        run_id: str,
        process_name: str,
        goal: str,
    ) -> Path | None:
        return await process_run_workspace.choose_process_run_workspace(
            self,
            run_id,
            process_name,
            goal,
        )
    def _current_process_run(self) -> ProcessRunState | None:
        return process_run_controls._current_process_run(self)

    @staticmethod
    def _is_process_run_active_status(status: str) -> bool:
        return process_run_controls._is_process_run_active_status(status)

    def _resolve_process_run_target(
        self,
        target: str,
    ) -> tuple[ProcessRunState | None, str | None]:
        return process_run_controls._resolve_process_run_target(self, target)

    async def _confirm_close_process_run(self, run: ProcessRunState) -> bool:
        return await process_run_controls._confirm_close_process_run(self, run)

    async def _confirm_force_close_process_run(
        self,
        run: ProcessRunState,
        *,
        timeout_seconds: float,
    ) -> bool:
        return await process_run_controls._confirm_force_close_process_run(
            self,
            run,
            timeout_seconds=timeout_seconds,
        )

    async def _confirm_stop_process_run(self, run: ProcessRunState) -> bool:
        return await process_run_controls._confirm_stop_process_run(self, run)

    def _register_process_run_cancel_handler(self, run_id: str, payload: object) -> None:
        process_run_controls._register_process_run_cancel_handler(self, run_id, payload)

    def _clear_process_run_cancel_handler(self, run_id: str) -> None:
        process_run_controls._clear_process_run_cancel_handler(self, run_id)

    def _persist_process_run_conversation_link(self, run: ProcessRunState) -> None:
        process_run_controls._persist_process_run_conversation_link(self, run)

    @staticmethod
    def _normalize_process_run_status(raw_status: object | None) -> str:
        return process_run_controls._normalize_process_run_status(raw_status)

    async def _request_process_run_cancellation(self, run: ProcessRunState) -> dict:
        return await process_run_controls._request_process_run_cancellation(self, run)

    async def _request_process_run_pause(self, run: ProcessRunState) -> dict:
        return await process_run_controls._request_process_run_pause(self, run)

    async def _request_process_run_play(self, run: ProcessRunState) -> dict:
        return await process_run_controls._request_process_run_play(self, run)

    async def _request_process_run_inject(self, run: ProcessRunState, text: str) -> dict:
        return await process_run_controls._request_process_run_inject(self, run, text)

    async def _request_process_run_question_answer(
        self,
        run: ProcessRunState,
        *,
        question_id: str,
        answer_payload: dict[str, Any],
    ) -> dict[str, Any]:
        return await process_run_controls._request_process_run_question_answer(
            self,
            run,
            question_id=question_id,
            answer_payload=answer_payload,
        )

    async def _flush_pending_process_run_inject(self, run_id: str) -> None:
        return await process_run_controls._flush_pending_process_run_inject(self, run_id)

    async def _wait_for_process_run_terminal_state(
        self,
        run_id: str,
        *,
        timeout_seconds: float,
    ) -> bool:
        return await process_run_controls._wait_for_process_run_terminal_state(
            self,
            run_id,
            timeout_seconds=timeout_seconds,
        )

    async def _finalize_process_run_tab_close(
        self,
        run: ProcessRunState,
        *,
        tabs: TabbedContent,
        cancel_worker: bool = False,
    ) -> bool:
        return await process_run_controls._finalize_process_run_tab_close(
            self,
            run,
            tabs=tabs,
            cancel_worker=cancel_worker,
        )

    async def _close_process_run(self, run: ProcessRunState) -> bool:
        return await process_run_controls._close_process_run(self, run)

    async def _close_process_run_from_target(self, target: str) -> bool:
        return await process_run_controls._close_process_run_from_target(self, target)

    async def _pause_process_run(self, run: ProcessRunState) -> bool:
        return await process_run_controls._pause_process_run(self, run)

    async def _play_process_run(self, run: ProcessRunState) -> bool:
        return await process_run_controls._play_process_run(self, run)

    async def _inject_process_run(
        self,
        run: ProcessRunState,
        text: str,
        *,
        source: str = "slash",
        queue_if_unavailable: bool = True,
    ) -> bool:
        return await process_run_controls._inject_process_run(
            self,
            run,
            text,
            source=source,
            queue_if_unavailable=queue_if_unavailable,
        )

    async def _stop_process_run(self, run: ProcessRunState, *, confirm: bool = False) -> bool:
        return await process_run_controls._stop_process_run(
            self,
            run,
            confirm=confirm,
        )

    async def _pause_process_run_from_target(self, target: str) -> bool:
        return await process_run_controls._pause_process_run_from_target(self, target)

    async def _play_process_run_from_target(self, target: str) -> bool:
        return await process_run_controls._play_process_run_from_target(self, target)

    async def _stop_process_run_from_target(self, target: str) -> bool:
        return await process_run_controls._stop_process_run_from_target(self, target)

    async def _inject_process_run_from_target(
        self,
        target: str,
        text: str,
        *,
        source: str = "slash",
    ) -> bool:
        return await process_run_controls._inject_process_run_from_target(
            self,
            target,
            text,
            source=source,
        )

    async def _resume_process_run_from_target(self, target: str) -> bool:
        return await process_run_controls._resume_process_run_from_target(self, target)

    @staticmethod
    def _resume_seed_task_rows(run: ProcessRunState) -> tuple[list[dict], dict[str, str]]:
        return process_run_controls._resume_seed_task_rows(run)

    async def _restart_process_run_in_place(
        self,
        run_id: str,
        *,
        mode: str = "restart",
    ) -> bool:
        return await process_run_controls._restart_process_run_in_place(
            self,
            run_id,
            mode=mode,
        )

    @staticmethod
    def _format_auth_profile_option(profile: Any) -> str:
        return process_run_auth.format_auth_profile_option(profile)

    async def _prompt_auth_choice(
        self,
        question: str,
        options: list[str],
        *,
        run_id: str = "",
    ) -> str:
        return await process_run_auth.prompt_auth_choice(
            self,
            question,
            options,
            run_id=run_id,
        )

    async def _open_auth_manager_for_run_start(
        self,
        *,
        process_def: ProcessDefinition | None = None,
        run_id: str = "",
    ) -> bool:
        return await process_run_auth.open_auth_manager_for_run_start(
            self,
            process_def=process_def,
            run_id=run_id,
        )

    def _collect_required_auth_resources_for_process(
        self,
        process_defn: ProcessDefinition | None,
    ) -> list[dict[str, Any]]:
        return process_run_auth.collect_required_auth_resources_for_process(
            self,
            process_defn,
        )

    async def _resolve_auth_overrides_for_run_start(
        self,
        *,
        process_defn: ProcessDefinition | None,
        base_overrides: dict[str, str],
        run_id: str = "",
    ) -> tuple[dict[str, str] | None, list[dict[str, Any]]]:
        return await process_run_auth.resolve_auth_overrides_for_run_start(
            self,
            process_defn=process_defn,
            base_overrides=base_overrides,
            run_id=run_id,
        )

    async def _start_process_run(
        self,
        goal: str,
        *,
        process_defn: ProcessDefinition | None = None,
        process_name_override: str | None = None,
        command_prefix: str = "/run",
        is_adhoc: bool = False,
        recommended_tools: list[str] | None = None,
        adhoc_synthesis_notes: list[str] | None = None,
        goal_context_overrides: dict[str, Any] | None = None,
        resume_task_id: str = "",
        run_workspace_override: Path | None = None,
        synthesis_goal: str = "",
        force_fresh: bool = False,
    ) -> None:
        return await process_run_lifecycle.start_process_run(
            self,
            goal,
            process_defn=process_defn,
            process_name_override=process_name_override,
            command_prefix=command_prefix,
            is_adhoc=is_adhoc,
            recommended_tools=recommended_tools,
            adhoc_synthesis_notes=adhoc_synthesis_notes,
            goal_context_overrides=goal_context_overrides,
            resume_task_id=resume_task_id,
            run_workspace_override=run_workspace_override,
            synthesis_goal=synthesis_goal,
            force_fresh=force_fresh,
        )

    async def _prepare_process_run_with_timeout(
        self,
        run_id: str,
        launch_request: ProcessRunLaunchRequest,
    ) -> bool:
        return await process_run_lifecycle.prepare_process_run_with_timeout(
            self,
            run_id,
            launch_request,
        )

    async def _prepare_and_execute_process_run(
        self,
        run_id: str,
        launch_request: ProcessRunLaunchRequest,
    ) -> None:
        return await process_run_lifecycle.prepare_and_execute_process_run(
            self,
            run_id,
            launch_request,
        )

    async def _prepare_process_run_launch(
        self,
        run_id: str,
        launch_request: ProcessRunLaunchRequest,
    ) -> bool:
        return await process_run_lifecycle.prepare_process_run_launch(
            self,
            run_id,
            launch_request,
        )

    async def _execute_process_run(self, run_id: str) -> None:
        return await process_run_lifecycle.execute_process_run(self, run_id)

    def _ensure_persistence_tools(self) -> None:
        app_tool_binding.ensure_persistence_tools(self)

    def _ensure_delegate_task_ready_for_run(self) -> tuple[bool, str]:
        return app_tool_binding.ensure_delegate_task_ready_for_run(self)

    def _slash_command_catalog(self) -> list[tuple[str, str]]:
        return slash_hints.slash_command_catalog(self)

    def _render_root_slash_hint(self) -> str:
        return slash_hints.render_root_slash_hint(self)

    def _config_source_file(self) -> Path | None:
        return getattr(self, "_config_source_path", None)

    def _sync_effective_runtime_config(self) -> None:
        self._config = self._config_runtime_store.effective_config()

    def _refresh_runtime_config_bindings(self) -> None:
        self._sync_effective_runtime_config()
        self._ensure_persistence_tools()
        if self._session is not None and self._db is not None:
            self._bind_session_tools()

    def _config_snapshot(self, path: str) -> dict[str, Any]:
        return self._config_runtime_store.snapshot(path)

    def _set_runtime_config_value(
        self,
        *,
        path: str,
        raw_value: object,
    ) -> tuple[Any, Any, dict[str, Any]]:
        entry, parsed = self._config_runtime_store.set_runtime_value(path, raw_value)
        self._sync_effective_runtime_config()
        return entry, parsed, self._config_runtime_store.snapshot(entry.path)

    def _persist_config_value(
        self,
        *,
        path: str,
        raw_value: object,
    ) -> tuple[Any, Any, dict[str, Any]]:
        entry, parsed = self._config_runtime_store.persist_value(path, raw_value)
        self._config_source_path = self._config_runtime_store.source_path()
        self._sync_effective_runtime_config()
        return entry, parsed, self._config_runtime_store.snapshot(entry.path)

    def _clear_runtime_config_value(self, *, path: str) -> dict[str, Any]:
        entry = self._config_runtime_store.clear_runtime_value(path)
        self._sync_effective_runtime_config()
        return self._config_runtime_store.snapshot(entry.path)

    def _reset_persisted_config_value(self, *, path: str) -> dict[str, Any]:
        entry = self._config_runtime_store.reset_persisted_value(path)
        self._config_source_path = self._config_runtime_store.source_path()
        self._sync_effective_runtime_config()
        return self._config_runtime_store.snapshot(entry.path)

    def _configured_telemetry_mode(self) -> TelemetryMode:
        try:
            configured = self._config_runtime_store.snapshot("telemetry.mode")["configured"]
        except Exception:
            telemetry_cfg = getattr(getattr(self, "_config", None), "telemetry", None)
            configured = getattr(telemetry_cfg, "mode", DEFAULT_TELEMETRY_MODE)
        return normalize_telemetry_mode(configured, default=DEFAULT_TELEMETRY_MODE).mode

    def _runtime_telemetry_mode(self) -> TelemetryMode | None:
        try:
            mode = self._config_runtime_store.snapshot("telemetry.mode")["runtime_override"]
        except Exception:
            mode = self._telemetry_runtime_override_mode
        if mode is None:
            return None
        return normalize_telemetry_mode(
            mode,
            default=DEFAULT_TELEMETRY_MODE,
        ).mode

    def _effective_telemetry_mode(self) -> TelemetryMode:
        runtime_mode = self._runtime_telemetry_mode()
        return runtime_mode or self._configured_telemetry_mode()

    def _render_telemetry_mode_status(self) -> str:
        snapshot = self._config_runtime_store.snapshot("telemetry.mode")
        configured_mode = snapshot["configured_display"]
        runtime_display = snapshot["runtime_display"]
        effective_mode = snapshot["effective_display"]
        updated_at = str(snapshot["updated_at"] or "").strip()
        lines = [
            "[bold #7dcfff]Telemetry Mode[/]",
            f"configured: [bold]{self._escape_markup(configured_mode)}[/bold]",
            f"runtime override: [bold]{self._escape_markup(runtime_display)}[/bold]",
            f"effective: [bold]{self._escape_markup(effective_mode)}[/bold]",
            "scope: [bold]process_local[/bold]",
        ]
        if updated_at:
            lines.append(f"updated_at: [dim]{self._escape_markup(updated_at)}[/dim]")
        return "\n".join(lines)

    @staticmethod
    def _slash_spec_sort_key(spec: SlashCommandSpec) -> tuple[int, str]:
        return slash_hints.slash_spec_sort_key(spec)

    def _ordered_slash_specs(self) -> list[SlashCommandSpec]:
        return slash_hints.ordered_slash_specs()

    @staticmethod
    def _slash_match_keys(spec: SlashCommandSpec) -> tuple[str, ...]:
        return slash_hints.slash_match_keys(spec)

    def _help_lines(self) -> list[str]:
        return slash_hints.help_lines(self)

    def _render_slash_command_usage(self, command: str, usage: str) -> str:
        return slash_hints.render_slash_command_usage(self, command, usage)

    def _render_tool_slash_hint(self, raw_input: str) -> str | None:
        return slash_hints.render_tool_slash_hint(self, raw_input)

    def _tool_name_completion_candidates(
        self,
        raw_input: str,
    ) -> tuple[str, list[str]] | None:
        return slash_completion.tool_name_completion_candidates(self, raw_input)

    def _matching_slash_commands(
        self,
        raw_input: str,
    ) -> tuple[str, list[tuple[str, str]]]:
        return slash_hints.matching_slash_commands(self, raw_input)

    def _reset_slash_tab_cycle(self) -> None:
        slash_completion.reset_slash_tab_cycle(self)

    def _reset_input_history_navigation(self) -> None:
        slash_input_history.reset_input_history_navigation(self)

    def _clear_input_history(self) -> None:
        slash_input_history.clear_input_history(self)

    def _append_input_history(self, value: str) -> None:
        slash_input_history.append_input_history(self, value)

    def _hydrate_input_history_from_session(self) -> None:
        slash_input_history.hydrate_input_history_from_session(self)

    def _set_user_input_text(self, value: str, *, from_history_navigation: bool = False) -> None:
        slash_input_history.set_user_input_text(
            self,
            value,
            from_history_navigation=from_history_navigation,
        )

    def _apply_input_history_navigation(self, *, older: bool) -> bool:
        return slash_input_history.apply_input_history_navigation(self, older=older)

    def _slash_completion_candidates(self, token: str) -> list[str]:
        return slash_completion.slash_completion_candidates(self, token)

    def _apply_slash_tab_completion(
        self,
        *,
        reverse: bool = False,
        input_widget: Input | None = None,
    ) -> bool:
        return slash_completion.apply_slash_tab_completion(
            self,
            reverse=reverse,
            input_widget=input_widget,
        )

    def _render_slash_hint(self, raw_input: str) -> str:
        return slash_hints.render_slash_hint(self, raw_input)

    @staticmethod
    def _strip_wrapping_quotes(value: str) -> str:
        return slash_completion.strip_wrapping_quotes(value)

    def _set_landing_slash_hint(self, hint_text: str) -> None:
        slash_hints.set_landing_slash_hint(self, hint_text)

    def _set_slash_hint(self, hint_text: str) -> None:
        slash_hints.set_slash_hint(self, hint_text)

    def _new_steering_directive(
        self,
        *,
        kind: str,
        text: str,
        source: str,
    ) -> SteeringDirective:
        return chat_steering.new_steering_directive(
            kind=kind,
            text=text,
            source=source,
            id_factory=lambda: uuid.uuid4().hex[:12],
        )

    def _pop_pending_inject_directive(
        self,
        *,
        clear_session: bool = True,
    ) -> SteeringDirective | None:
        return chat_steering.pop_pending_inject_directive(
            self,
            clear_session=clear_session,
        )

    def _clear_pending_inject_directives(
        self,
        *,
        clear_session: bool = True,
    ) -> list[SteeringDirective]:
        return chat_steering.clear_pending_inject_directives(
            self,
            clear_session=clear_session,
        )

    def _pending_inject_directive_index(self, directive_id: str) -> int:
        return chat_steering.pending_inject_directive_index(
            self,
            directive_id,
        )

    def _sync_session_pending_inject_queue(self) -> None:
        chat_steering.sync_session_pending_inject_queue(self)

    def _remove_pending_inject_directive_at(
        self,
        index: int,
        *,
        clear_session: bool = True,
    ) -> SteeringDirective | None:
        return chat_steering.remove_pending_inject_directive_at(
            self,
            index,
            clear_session=clear_session,
        )

    def _pop_next_queued_followup_directive(self) -> SteeringDirective | None:
        return chat_steering.pop_next_queued_followup_directive(self)

    def _start_queued_followup_turn(self, message: str) -> None:
        chat_steering.start_queued_followup_turn(self, message)

    def _take_input_text_for_steering(self) -> str:
        return chat_steering.take_input_text_for_steering(self)

    def _render_steer_queue_status(self) -> str:
        return chat_steering.render_steer_queue_status(self)

    def _mark_cowork_tool_inflight(self, tool_name: str) -> None:
        chat_steering.mark_cowork_tool_inflight(self, tool_name)

    def _clear_cowork_tool_inflight(self, tool_name: str) -> None:
        chat_steering.clear_cowork_tool_inflight(self, tool_name)

    def _cowork_inflight_mutating_tool_name(self) -> str:
        return chat_steering.cowork_inflight_mutating_tool_name(self)

    async def _record_steering_event(
        self,
        event_type: str,
        *,
        message: str = "",
        directive: SteeringDirective | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        return await chat_steering.record_steering_event(
            self,
            event_type,
            message=message,
            directive=directive,
            extra=extra,
        )

    async def _queue_chat_inject_instruction(
        self,
        text: str,
        *,
        source: str,
    ) -> bool:
        return await chat_steering.queue_chat_inject_instruction(
            self,
            text,
            source=source,
        )

    async def _sync_pending_inject_apply_state(self) -> None:
        return await chat_steering.sync_pending_inject_apply_state(self)

    async def _request_chat_pause(self, *, source: str) -> bool:
        return await chat_steering.request_chat_pause(
            self,
            source=source,
        )

    async def _request_chat_resume(self, *, source: str) -> bool:
        return await chat_steering.request_chat_resume(
            self,
            source=source,
        )

    async def _clear_chat_steering(self, *, source: str) -> bool:
        return await chat_steering.clear_chat_steering(
            self,
            source=source,
        )

    async def _confirm_redirect_with_mutating_tool(self, tool_name: str) -> bool:
        return await chat_steering.confirm_redirect_with_mutating_tool(
            self,
            tool_name,
        )

    async def _request_chat_redirect(
        self,
        text: str,
        *,
        source: str,
    ) -> bool:
        return await chat_steering.request_chat_redirect(
            self,
            text,
            source=source,
        )

    def _reset_cowork_steering_state(self, *, clear_session: bool = True) -> None:
        chat_steering.reset_cowork_steering_state(
            self,
            clear_session=clear_session,
        )

    def _bind_session_tools(self) -> None:
        app_tool_binding.bind_session_tools(self)

    # ------------------------------------------------------------------
    # Session management helpers
    # ------------------------------------------------------------------

    def _clear_files_panel(self) -> None:
        """Reset file history/diff when changing sessions."""
        panel = self.query_one("#files-panel", FilesChangedPanel)
        panel.clear_files()
        panel.show_diff("")
        self._files_panel_recent_ops.clear()

    def _clear_chat_widgets(self) -> None:
        """Drop all currently mounted chat widgets."""
        chat = self.query_one("#chat-log", ChatLog)
        for child in list(chat.children):
            child.remove()
        reset_runtime_state = getattr(chat, "reset_runtime_state", None)
        if callable(reset_runtime_state):
            reset_runtime_state()
        self._active_delegate_streams = {}
        self._sync_activity_indicator()

    def _reset_chat_history_state(self) -> None:
        """Reset in-memory replay state for the active chat transcript."""
        self._chat_replay_events = []
        self._chat_history_source = ""
        self._chat_history_oldest_seq = None
        self._chat_history_oldest_turn = None
        self._chat_render_window_start = 0
        self._chat_follow_latest = True
        self._chat_hidden_older_count = 0
        self._chat_hidden_newer_count = 0
        self._chat_transcript_mode = False
        self._chat_transcript_show_thinking = False
        self._chat_search_query = ""
        self._chat_search_match_positions = []
        self._chat_search_match_current = -1

    def _active_session_id(self) -> str:
        return chat_session.active_session_id(self._session)

    def _chat_event_cursor_turn(self, event: dict) -> int | None:
        return chat_session.chat_event_cursor_turn(event)

    @staticmethod
    def _coerce_int(value: object, *, default: int = 0) -> int:
        return chat_session.coerce_int(value, default=default)

    @staticmethod
    def _coerce_float(value: object, *, default: float = 0.0) -> float:
        return chat_session.coerce_float(value, default=default)

    @staticmethod
    def _coerce_bool(value: object, *, default: bool = False) -> bool:
        return chat_session.coerce_bool(value, default=default)

    def _apply_chat_render_cap(self, *, mode: str = "append") -> bool:
        """Update the visible transcript window. Returns True if rerender is needed."""
        (
            rerender,
            render_window_start,
            follow_latest,
            hidden_older,
            hidden_newer,
        ) = chat_history.update_chat_render_window(
            total_rows=len(self._chat_replay_events),
            max_rows=self._chat_resume_max_rendered_rows(),
            current_start=getattr(self, "_chat_render_window_start", 0),
            follow_latest=getattr(self, "_chat_follow_latest", True),
            mode=mode,
        )
        self._chat_render_window_start = render_window_start
        self._chat_follow_latest = follow_latest
        self._chat_hidden_older_count = hidden_older
        self._chat_hidden_newer_count = hidden_newer
        return rerender

    def _render_chat_event(self, *args, **kwargs):
        return chat_history.render_chat_event(self, *args, **kwargs)

    def _rerender_chat_from_replay_events(self, *args, **kwargs):
        return chat_history.rerender_chat_from_replay_events(self, *args, **kwargs)

    async def _append_chat_replay_event(self, *args, **kwargs):
        return await chat_history.append_chat_replay_event(self, *args, **kwargs)

    async def _hydrate_chat_history_for_active_session(self, *args, **kwargs):
        return await chat_history.hydrate_chat_history_for_active_session(self, *args, **kwargs)

    async def _load_older_chat_history(self, *args, **kwargs):
        return await chat_history.load_older_chat_history(self, *args, **kwargs)

    def _set_chat_transcript_mode(self, *args, **kwargs):
        return chat_history.set_chat_transcript_mode(self, *args, **kwargs)

    def _set_chat_transcript_show_thinking(self, *args, **kwargs):
        return chat_history.set_chat_transcript_show_thinking(self, *args, **kwargs)

    def _clear_chat_search(self, *args, **kwargs):
        return chat_history.clear_chat_search(self, *args, **kwargs)

    def _search_chat_history(self, *args, **kwargs):
        return chat_history.search_chat_history(self, *args, **kwargs)

    def _step_chat_history_search(self, *args, **kwargs):
        return chat_history.step_chat_history_search(self, *args, **kwargs)

    def _jump_chat_history_latest(self, *args, **kwargs):
        return chat_history.jump_chat_history_latest(self, *args, **kwargs)

    async def _new_session(self, *args, **kwargs):
        return await chat_history.new_session(self, *args, **kwargs)

    async def _switch_to_session(self, *args, **kwargs):
        return await chat_history.switch_to_session(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Learned patterns
    # ------------------------------------------------------------------

    async def _show_learned_patterns(self) -> None:
        return await chat_learned.show_learned_patterns(self)

    @work
    async def _delete_learned_patterns(self, deleted_ids_csv: str) -> None:
        return await chat_learned.delete_learned_patterns(self, deleted_ids_csv)

    # ------------------------------------------------------------------
    # Approval callback
    # ------------------------------------------------------------------

    async def _approval_callback(
        self, tool_name: str, args: dict,
    ) -> ApprovalDecision:
        return await chat_approval.approval_callback(self, tool_name, args)

    # ------------------------------------------------------------------
    # User input
    # ------------------------------------------------------------------

    async def _submit_user_text(self, text: str, *, source: str) -> None:
        await chat_input_submission.submit_user_text(self, text, source=source)

    @on(Input.Submitted, "#user-input")
    async def on_user_submit(self, event: Input.Submitted) -> None:
        await self._submit_user_text(event.value, source="chat")

    @on(Input.Submitted, "#landing-input")
    async def on_landing_submit(self, event: Input.Submitted) -> None:
        """Handle initial startup prompt submission from landing composer."""
        await self._submit_user_text(event.value, source="landing")

    @on(events.Click, "#landing-close-btn")
    async def on_landing_close_pressed(self, event: events.Click) -> None:
        await app_ui_events.on_landing_close_pressed(self, event)

    @on(Input.Changed, "#user-input")
    def on_user_input_changed(self, _event: Input.Changed) -> None:
        app_ui_events.on_user_input_changed(self, _event)

    @on(Input.Changed, "#landing-input")
    def on_landing_input_changed(self, _event: Input.Changed) -> None:
        app_ui_events.on_landing_input_changed(self, _event)

    @on(Button.Pressed, ".process-run-restart-btn")
    def on_process_run_restart_pressed(self, event: Button.Pressed) -> None:
        app_ui_events.on_process_run_restart_pressed(self, event)

    @on(Button.Pressed, ".process-run-control-btn")
    def on_process_run_control_pressed(self, event: Button.Pressed) -> None:
        app_ui_events.on_process_run_control_pressed(self, event)

    @on(Button.Pressed, "#chat-stop-btn")
    def _on_chat_stop_pressed(self, event: Button.Pressed) -> None:
        app_ui_events.on_chat_stop_pressed(self, event)

    @on(Button.Pressed, "#chat-inject-btn")
    def _on_chat_inject_pressed(self, event: Button.Pressed) -> None:
        app_ui_events.on_chat_inject_pressed(self, event)

    @on(Button.Pressed, "#chat-redirect-btn")
    def _on_chat_redirect_pressed(self, event: Button.Pressed) -> None:
        app_ui_events.on_chat_redirect_pressed(self, event)

    @on(Button.Pressed)
    def _on_dynamic_steer_queue_button_pressed(self, event: Button.Pressed) -> None:
        app_ui_events.on_dynamic_steer_queue_button_pressed(self, event)

    @on(Button.Pressed, "#footer-auth-shortcut, #footer-mcp-shortcut")
    def _on_footer_manager_shortcut_pressed(self, event: Button.Pressed) -> None:
        app_ui_events.on_footer_manager_shortcut_pressed(self, event)

    def on_key(self, event: events.Key) -> None:
        app_lifecycle.on_key(self, event)

    @on(TabbedContent.TabActivated, "#tabs")
    def on_tabs_tab_activated(self, _event: TabbedContent.TabActivated) -> None:
        """Keep sidebar summary in sync as tabs change."""
        self._refresh_sidebar_progress_summary()

    @on(DirectoryTree.FileSelected, "#workspace-tree")
    def on_workspace_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Open a modal preview when selecting a file in the workspace tree."""
        selected = self._resolve_workspace_file(Path(event.path))
        if selected is None:
            self.notify(
                "Cannot open files outside the workspace.",
                severity="error",
                timeout=4,
            )
            return
        event.stop()
        event.prevent_default()
        self.push_screen(
            FileViewerScreen(selected, self._workspace, defer_heavy_load=True)
        )

    async def _handle_slash_command(self, text: str) -> bool:
        """Compatibility wrapper that delegates slash handling to the slash package."""
        return await slash_handlers.handle_slash_command(self, text)

    async def _handle_slash_command_core(self, text: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        return await slash_handlers.handle_slash_command_core(self, text)

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    @staticmethod
    def _chat_stop_cooperative_wait_seconds() -> float:
        return chat_steering.chat_stop_cooperative_wait_seconds()

    @staticmethod
    def _chat_stop_settle_timeout_seconds() -> float:
        return chat_steering.chat_stop_settle_timeout_seconds()

    async def _wait_for_chat_turn_settle(self, *, timeout_seconds: float) -> bool:
        return await chat_steering.wait_for_chat_turn_settle(
            self,
            timeout_seconds=timeout_seconds,
        )

    async def _finalize_unsettled_delegate_streams_for_stop(self) -> None:
        return await chat_steering.finalize_unsettled_delegate_streams_for_stop(self)

    async def _handle_interrupted_chat_turn(
        self,
        *,
        path: str,
        reason: str = "",
        stage: str = "",
    ) -> None:
        return await chat_steering.handle_interrupted_chat_turn(
            self,
            path=path,
            reason=reason,
            stage=stage,
        )

    async def _request_chat_stop(self) -> None:
        return await chat_steering.request_chat_stop(self)

    def action_stop_chat(self) -> None:
        chat_steering.action_stop_chat(self)

    def action_inject_chat(self) -> None:
        chat_steering.action_inject_chat(self)

    def action_redirect_chat(self) -> None:
        chat_steering.action_redirect_chat(self)

    def action_steer_queue_edit(self, directive_id: str = "") -> None:
        chat_steering.action_steer_queue_edit(
            self,
            directive_id=directive_id,
        )

    def action_steer_queue_dismiss(self, directive_id: str = "") -> None:
        chat_steering.action_steer_queue_dismiss(
            self,
            directive_id=directive_id,
        )

    def action_steer_queue_redirect(self, directive_id: str = "") -> None:
        chat_steering.action_steer_queue_redirect(
            self,
            directive_id=directive_id,
        )

    @work(group="chat-turn", exclusive=False)
    async def _run_turn(self, *args, **kwargs):
        return await chat_turns.run_turn(self, *args, **kwargs)

    async def _run_followup(self, message: str) -> None:
        """Run a follow-up turn (e.g. after ask_user answer)."""
        if self._session is None:
            return
        try:
            await self._run_interaction(message)
        except Exception as e:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_model_text(f"[bold #f7768e]Error:[/] {e}", markup=True)
            self.notify(str(e), severity="error", timeout=5)

    @staticmethod
    def _delegate_target_for_tool_call(tool_name: str, args: dict | None) -> str:
        """Return delegated target tool name when this call wraps another tool."""
        return chat_turns.delegate_target_for_tool_call(tool_name, args)

    @staticmethod
    def _delegate_progress_title(caller_tool_name: str) -> str:
        return chat_turns.delegate_progress_title(caller_tool_name)

    def _ensure_delegate_progress_widget(
        self,
        *,
        tool_call_id: str,
        title: str,
        status: str = "running",
        elapsed_ms: int = 0,
        lines: list[str] | None = None,
    ) -> bool:
        return chat_delegate_progress.ensure_delegate_progress_widget(
            self,
            tool_call_id=tool_call_id,
            title=title,
            status=status,
            elapsed_ms=elapsed_ms,
            lines=lines,
        )

    def _append_delegate_progress_widget_line(self, tool_call_id: str, line: str) -> bool:
        return chat_delegate_progress.append_delegate_progress_widget_line(
            self,
            tool_call_id,
            line,
        )

    async def _start_delegate_progress_stream(
        self,
        *,
        tool_call_id: str,
        caller_tool_name: str,
        persist: bool = True,
    ) -> None:
        return await chat_delegate_progress.start_delegate_progress_stream(
            self,
            tool_call_id=tool_call_id,
            caller_tool_name=caller_tool_name,
            persist=persist,
        )

    async def _finalize_delegate_progress_stream(
        self,
        *,
        tool_call_id: str,
        success: bool,
        elapsed_ms: int = 0,
        persist: bool = True,
    ) -> None:
        return await chat_delegate_progress.finalize_delegate_progress_stream(
            self,
            tool_call_id=tool_call_id,
            success=success,
            elapsed_ms=elapsed_ms,
            persist=persist,
        )

    async def _on_cowork_delegate_progress_event(self, payload: dict[str, Any]) -> None:
        return await chat_delegate_progress.on_cowork_delegate_progress_event(
            self,
            payload,
        )

    async def _run_interaction(self, *args, **kwargs):
        return await chat_turns.run_interaction(self, *args, **kwargs)

    async def _run_process_goal(self, goal: str) -> None:
        """Compatibility wrapper for callers that still use the old method."""
        await self._start_process_run(goal)

    def _on_process_progress_event(
        self,
        data: dict,
        *,
        run_id: str | None = None,
    ) -> None:
        process_run_events.on_process_progress_event(self, data, run_id=run_id)

    @staticmethod
    def _one_line(text: object | None, max_len: int | None = 180) -> str:
        return process_run_rendering._one_line(text, max_len=max_len)

    def _normalize_process_run_tasks(
        self,
        run: ProcessRunState,
        tasks: list[dict],
    ) -> list[dict]:
        return process_run_rendering._normalize_process_run_tasks(self, run, tasks)

    def _process_run_output_rows(self, run: ProcessRunState) -> list[dict]:
        return process_run_rendering._process_run_output_rows(self, run)

    def _process_run_phase_map(self, run: ProcessRunState) -> dict[str, str]:
        return process_run_rendering._process_run_phase_map(self, run)

    @staticmethod
    def _aggregate_phase_state(statuses: list[str]) -> str:
        return process_run_rendering._aggregate_phase_state(statuses)

    def _infer_process_run_task_phase_id(
        self,
        run: ProcessRunState,
        *,
        row: dict,
        phase_ids: list[str],
        phase_labels: dict[str, str],
        deliverables_by_phase: dict[str, list[str]],
        phase_map: dict[str, str],
    ) -> str:
        return process_run_rendering._infer_process_run_task_phase_id(
            self,
            run,
            row=row,
            phase_ids=phase_ids,
            phase_labels=phase_labels,
            deliverables_by_phase=deliverables_by_phase,
            phase_map=phase_map,
        )

    def _refresh_process_run_outputs(self, run: ProcessRunState) -> None:
        process_run_rendering._refresh_process_run_outputs(self, run)

    @staticmethod
    def _subtask_content(
        data: dict,
        subtask_id: str,
        run: ProcessRunState | None = None,
    ) -> str:
        return process_run_rendering._subtask_content(data, subtask_id, run=run)

    def _format_process_progress_event(
        self,
        data: dict,
        *,
        run: ProcessRunState | None = None,
        context: str = "process_run",
    ) -> str | None:
        return process_run_rendering._format_process_progress_event(
            self,
            data,
            run=run,
            context=context,
        )

    def _mark_process_run_failed(self, error: str) -> None:
        process_run_rendering._mark_process_run_failed(self, error)

    async def _prompt_process_run_question(
        self,
        *,
        run_id: str,
        question_payload: dict[str, Any],
    ) -> None:
        return await process_run_questions._prompt_process_run_question(
            self,
            run_id=run_id,
            question_payload=question_payload,
        )

    async def _handle_ask_user(self, event: ToolCallEvent) -> str:
        return await process_run_questions._handle_ask_user(self, event)

    def _update_sidebar_tasks(self, data: dict) -> None:
        process_run_rendering._update_sidebar_tasks(self, data)

    def _summarize_cowork_tasks(self) -> list[dict]:
        return process_run_rendering._summarize_cowork_tasks(self)

    def _refresh_sidebar_progress_summary(self) -> None:
        process_run_rendering._refresh_sidebar_progress_summary(self)

    def _compute_workspace_signature(
        self,
    ) -> tuple[tuple[int, int, int] | None, bool]:
        return app_workspace_watch._compute_workspace_signature(self)

    def _start_workspace_watch(self) -> None:
        app_workspace_watch._start_workspace_watch(self)

    def _stop_workspace_watch(self) -> None:
        app_workspace_watch._stop_workspace_watch(self)

    def _on_workspace_poll_tick(self) -> None:
        app_workspace_watch._on_workspace_poll_tick(self)

    def _cancel_workspace_refresh_timer(self) -> None:
        app_workspace_watch._cancel_workspace_refresh_timer(self)

    def _request_workspace_refresh(
        self,
        reason: str,
        *,
        immediate: bool = False,
    ) -> None:
        app_workspace_watch._request_workspace_refresh(
            self,
            reason,
            immediate=immediate,
        )

    def _flush_workspace_refresh_requests(self) -> None:
        app_workspace_watch._flush_workspace_refresh_requests(self)

    def _refresh_workspace_tree(self) -> None:
        app_workspace_watch._refresh_workspace_tree(self)

    def _resolve_workspace_file(self, path: Path) -> Path | None:
        return app_files_panel._resolve_workspace_file(self, path)

    def _normalize_files_changed_paths(self, raw_paths: object) -> list[str]:
        return app_files_panel._normalize_files_changed_paths(self, raw_paths)

    @staticmethod
    def _summary_files_changed_markers(raw_paths: object) -> list[tuple[str, str]]:
        return app_files_panel._summary_files_changed_markers(raw_paths)

    @staticmethod
    def _operation_hint_for_tool(tool_name: str) -> str:
        return app_files_panel._operation_hint_for_tool(tool_name)

    def _ingest_files_panel_from_paths(
        self,
        raw_paths: object,
        *,
        operation_hint: str = "modify",
    ) -> int:
        return app_files_panel._ingest_files_panel_from_paths(
            self,
            raw_paths,
            operation_hint=operation_hint,
        )

    def _ingest_files_panel_from_tool_call_event(self, event: ToolCallEvent) -> int:
        return app_files_panel._ingest_files_panel_from_tool_call_event(self, event)

    def _update_files_panel(self, turn: CoworkTurn) -> None:
        app_files_panel._update_files_panel(self, turn)

    def action_toggle_sidebar(self) -> None:
        app_actions.action_toggle_sidebar(self)

    def action_clear_chat(self) -> None:
        app_actions.action_clear_chat(self)

    def action_reload_workspace(self) -> None:
        app_actions.action_reload_workspace(self)

    async def _request_manager_tab_close(self, pane_id: str) -> bool:
        return await app_actions.request_manager_tab_close(self, pane_id)

    def _command_palette_active(self) -> bool:
        return app_actions.command_palette_is_active(self)

    def action_close_process_tab(self) -> None:
        app_actions.action_close_process_tab(self)

    def action_tab_chat(self) -> None:
        app_actions.action_tab_chat(self)

    def action_tab_files(self) -> None:
        app_actions.action_tab_files(self)

    def action_tab_events(self) -> None:
        app_actions.action_tab_events(self)

    def action_command_palette(self) -> None:
        app_actions.action_command_palette(self)

    def action_open_auth_tab(self) -> None:
        app_actions.action_open_auth_tab(self)

    def action_open_mcp_tab(self) -> None:
        app_actions.action_open_mcp_tab(self)

    async def action_quit(self) -> None:
        await app_actions.action_quit(self)

    def action_request_quit(self) -> None:
        app_actions.action_request_quit(self)

    async def _confirm_exit(self) -> bool:
        return await app_actions.confirm_exit(self)

    async def _request_exit(self) -> None:
        await app_actions.request_exit(self)

    async def action_loom_command(self, command: str) -> None:
        await app_actions.action_loom_command(self, command)

    def _prefill_user_input(self, text: str) -> None:
        app_command_palette.prefill_user_input(self, text)

    def _show_tools(self) -> None:
        app_command_palette.show_tools(self)

    def _show_model_info(self) -> None:
        app_command_palette.show_model_info(self)

    def _show_models_info(self) -> None:
        app_command_palette.show_models_info(self)

    def _show_process_info(self) -> None:
        app_command_palette.show_process_info(self)

    def _show_process_list(self) -> None:
        app_command_palette.show_process_list(self)

    def iter_dynamic_process_palette_entries(self) -> list[tuple[str, str, str]]:
        return app_command_palette.iter_dynamic_process_palette_entries(self)

    def _show_token_info(self) -> None:
        app_command_palette.show_token_info(self)

    def _show_help(self) -> None:
        app_command_palette.show_help(self)

    async def _palette_quit(self) -> None:
        await app_command_palette.palette_quit(self)


def _now_str() -> str:
    """Return current time as HH:MM:SS string."""
    return datetime.now().strftime("%H:%M:%S")
