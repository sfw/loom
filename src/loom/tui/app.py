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
  | ^B Sidebar  ^L Clear  ^P Commands   ^C Quit        |
  +-----+----------------------------------------------+
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shlex
import textwrap
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Static,
    TabbedContent,
    TabPane,
)

from loom.cowork.approval import ApprovalDecision, ToolApprover
from loom.cowork.session import (
    CoworkSession,
    CoworkTurn,
    ToolCallEvent,
    build_cowork_system_prompt,
)
from loom.models.base import ModelProvider
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.tools.registry import ToolRegistry
from loom.tui.commands import LoomCommands
from loom.tui.screens import (
    AskUserScreen,
    AuthManagerScreen,
    ExitConfirmScreen,
    FileViewerScreen,
    LearnedScreen,
    MCPManagerScreen,
    ProcessRunCloseScreen,
    SetupScreen,
    ToolApprovalScreen,
)
from loom.tui.theme import LOOM_DARK
from loom.tui.widgets import (
    ChatLog,
    EventPanel,
    FilesChangedPanel,
    Sidebar,
    StatusBar,
)
from loom.tui.widgets.tool_call import tool_args_preview

if TYPE_CHECKING:
    from loom.config import Config
    from loom.processes.schema import ProcessDefinition
    from loom.state.conversation_store import ConversationStore
    from loom.state.memory import Database

logger = logging.getLogger(__name__)


def _plain_text(value: object | None) -> str:
    """Coerce rich/plain values to a user-facing plain string."""
    if value is None:
        return ""
    if isinstance(value, Text):
        return value.plain
    return str(value)


def _escape_markup_text(value: object | None) -> str:
    """Escape Rich markup control chars in dynamic text."""
    return _plain_text(value).replace("[", "\\[")


@dataclass(frozen=True)
class SlashCommandSpec:
    """Definition for a slash command shown in help and autocomplete."""

    canonical: str
    description: str
    aliases: tuple[str, ...] = ()
    usage: str = ""


_SLASH_COMMANDS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec(
        canonical="/quit",
        aliases=("/exit", "/q"),
        description="exit Loom",
    ),
    SlashCommandSpec(canonical="/clear", description="clear chat log"),
    SlashCommandSpec(canonical="/help", description="show command help"),
    SlashCommandSpec(canonical="/setup", description="run setup wizard"),
    SlashCommandSpec(canonical="/model", description="show active model"),
    SlashCommandSpec(
        canonical="/mcp",
        usage=(
            "[manage|list|show <alias>|test <alias>|add <alias> ...|"
            "edit <alias> ...|enable <alias>|disable <alias>|remove <alias>]"
        ),
        description="inspect/manage MCP server config",
    ),
    SlashCommandSpec(
        canonical="/auth",
        usage=(
            "[manage|list|show <profile-id>|check|use <selector=profile>|"
            "clear [selector]|select <selector=profile>|unset <selector>|"
            "add <profile-id> ...|edit <profile-id> ...|remove <profile-id>]"
        ),
        description="inspect/manage run auth profile selection",
    ),
    SlashCommandSpec(canonical="/tools", description="list available tools"),
    SlashCommandSpec(canonical="/tokens", description="show session token usage"),
    SlashCommandSpec(canonical="/session", description="show current session info"),
    SlashCommandSpec(canonical="/new", description="start a new session"),
    SlashCommandSpec(canonical="/sessions", description="list recent sessions"),
    SlashCommandSpec(
        canonical="/resume",
        usage="<session-id-prefix>",
        description="resume session by ID prefix",
    ),
    SlashCommandSpec(
        canonical="/learned",
        description="review/delete learned patterns",
    ),
    SlashCommandSpec(
        canonical="/process",
        usage="[list|use <name>|off]",
        description="legacy process controls (prefer /processes)",
    ),
    SlashCommandSpec(
        canonical="/processes",
        description="list available process definitions",
    ),
    SlashCommandSpec(
        canonical="/run",
        usage=(
            "<goal|close [run-id-prefix]|resume <run-id-prefix|current>|"
            "save <run-id-prefix|current> <name>>"
        ),
        description="run goal via process orchestrator (auto-ad-hoc when needed)",
    ),
)
_MAX_SLASH_HINT_LINES = 24
_WORKSPACE_REFRESH_TOOLS = {"document_write", "document_create", "humanize_writing"}
_PROCESS_STATUS_ICON = {
    "queued": "\u25cb",
    "running": "\u25c9",
    "completed": "\u2713",
    "failed": "\u2717",
    "cancelled": "\u25a0",
}
_PROCESS_STATUS_LABEL = {
    "queued": "Queued",
    "running": "Running",
    "completed": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
}
_MAX_CONCURRENT_PROCESS_RUNS = 4
_MAX_PERSISTED_PROCESS_RUNS = 12
_MAX_PERSISTED_PROCESS_ACTIVITY = 300
_MAX_PERSISTED_PROCESS_RESULTS = 120
_INFO_WRAP_WIDTH = 108
_RUN_GOAL_FILE_CONTENT_MAX_CHARS = 32_000
_MAX_INPUT_HISTORY = 500


class ProcessRunList(VerticalScroll):
    """Scrollable checklist list used by process-run Progress/Outputs panes."""

    def __init__(
        self,
        *,
        empty_message: str,
        auto_follow: bool = True,
        follow_mode: str = "tail",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._auto_follow = bool(auto_follow)
        self._empty_message = str(empty_message or "No items")
        self._follow_mode = (
            str(follow_mode).strip().lower()
            if str(follow_mode).strip().lower() in {"tail", "active"}
            else "tail"
        )
        self._rows: list[dict] = []
        self._pending_rows: list[dict] | None = None
        self._body = Static("", classes="process-run-list-body", expand=True)

    def compose(self) -> ComposeResult:
        yield self._body

    def set_rows(self, rows: list[dict]) -> None:
        """Replace rendered rows and keep the newest rows visible."""
        normalized = [row for row in rows if isinstance(row, dict)]
        if not self.is_attached:
            self._pending_rows = normalized
            return
        self._rows = normalized
        self._body.update(self._render_rows())
        self._scroll_to_latest()

    def on_mount(self) -> None:
        """Flush pre-mount row updates once the widget is attached."""
        if self._pending_rows is not None:
            pending = list(self._pending_rows)
            self._pending_rows = None
            self.set_rows(pending)
            return
        self._body.update(self._render_rows())

    def _render_rows(self) -> Text:
        if not self._rows:
            empty = Text.from_markup(f"[dim]{self._empty_message}[/dim]")
            empty.no_wrap = False
            empty.overflow = "fold"
            return empty

        lines: list[str] = []
        for row in self._rows:
            status = str(row.get("status", "pending")).strip()
            content = _escape_markup_text(row.get("content", "?")).strip() or "?"
            if status == "completed":
                icon = "[#9ece6a]\u2713[/]"
            elif status == "in_progress":
                icon = "[#7dcfff]\u25c9[/]"
            elif status == "failed":
                icon = "[#f7768e]\u2717[/]"
            elif status == "skipped":
                icon = "[dim]-[/dim]"
            else:
                icon = "[dim]\u25cb[/dim]"
            lines.append(f"{icon} {content}")
        rendered = Text.from_markup("\n".join(lines))
        rendered.no_wrap = False
        rendered.overflow = "fold"
        return rendered

    def _scroll_to_latest(self) -> None:
        if not self._auto_follow or not self.is_attached:
            return
        if self._follow_mode == "active":
            focus_index = self._active_focus_index()
            if focus_index is None:
                return

            # Keep the most relevant in-flight / recently-finished rows in view.
            target_line = max(focus_index - 2, 0)

            def _focus() -> None:
                self.scroll_to(y=target_line, animate=False, force=True)

            self.call_after_refresh(_focus)
            return
        self.call_after_refresh(self.scroll_end, animate=False)

    def _active_focus_index(self) -> int | None:
        """Return the output row index that should stay in view for active-follow."""
        active_idx: int | None = None
        complete_idx: int | None = None
        terminal_idx: int | None = None
        for idx, row in enumerate(self._rows):
            status = str(row.get("status", "pending")).strip()
            if status == "in_progress":
                active_idx = idx
            elif status == "completed":
                complete_idx = idx
            elif status in {"failed", "skipped"}:
                terminal_idx = idx
        if active_idx is not None:
            return active_idx
        if complete_idx is not None:
            return complete_idx
        return terminal_idx


class ProcessRunPane(Vertical):
    """A per-run process pane with status, progress rows, and run log."""

    DEFAULT_CSS = """
    ProcessRunPane {
        height: 1fr;
        padding: 0 1;
        overflow: hidden;
    }
    ProcessRunPane .process-run-header {
        color: $text;
        text-style: bold;
        margin: 0 0 1 0;
    }
    ProcessRunPane .process-run-meta {
        color: $text-muted;
        margin: 0 0 1 0;
    }
    ProcessRunPane .process-run-actions {
        margin: 0 0 1 0;
        height: auto;
    }
    ProcessRunPane .process-run-restart-btn {
        width: auto;
        min-width: 22;
    }
    ProcessRunPane .process-run-section {
        color: $text-muted;
        text-style: bold;
        margin: 1 0 0 0;
    }
    ProcessRunPane .process-run-list {
        border: round $surface-lighten-1;
        margin: 0;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    ProcessRunPane .process-run-list-body {
        width: 100%;
    }
    ProcessRunPane #process-run-progress {
        height: 11;
        min-height: 7;
        max-height: 16;
    }
    ProcessRunPane #process-run-outputs {
        height: 9;
        min-height: 5;
        max-height: 14;
    }
    ProcessRunPane ChatLog {
        height: 1fr;
        border: round $surface-lighten-1;
        margin: 1 0 0 0;
    }
    """

    def __init__(self, *, run_id: str, process_name: str, goal: str) -> None:
        super().__init__()
        self._run_id = run_id
        self._process_name = process_name
        self._goal = goal
        self._pending_tasks: list[dict] | None = None
        self._pending_outputs: list[dict] | None = None
        self._pending_activity: list[str] = []
        self._pending_results: list[tuple[str, bool]] = []
        self._header = Static(classes="process-run-header")
        self._meta = Static(classes="process-run-meta")
        self._actions = Horizontal(classes="process-run-actions")
        self._actions.display = False
        self._restart_button = Button(
            "Restart Failed Run",
            id=f"process-run-restart-{run_id}",
            classes="process-run-restart-btn",
            variant="primary",
        )
        self._restart_button.display = False
        self._restart_button.disabled = True
        self._progress_label = Static("Progress", classes="process-run-section")
        self._progress = ProcessRunList(
            id="process-run-progress",
            classes="process-run-list",
            auto_follow=True,
            follow_mode="active",
            empty_message="No progress yet",
        )
        self._outputs_label = Static("Outputs", classes="process-run-section")
        self._outputs = ProcessRunList(
            id="process-run-outputs",
            classes="process-run-list",
            auto_follow=True,
            follow_mode="active",
            empty_message="No outputs yet",
        )
        self._log_label = Static("Activity", classes="process-run-section")
        self._log = ChatLog()

    def compose(self) -> ComposeResult:
        yield self._header
        yield self._meta
        with self._actions:
            yield self._restart_button
        yield self._progress_label
        yield self._progress
        yield self._outputs_label
        yield self._outputs
        yield self._log_label
        yield self._log

    def set_status_header(
        self,
        *,
        status: str,
        elapsed: str,
        task_id: str = "",
    ) -> None:
        """Update run header metadata."""
        label = _PROCESS_STATUS_LABEL.get(status, status.title())
        self._header.update(
            f"[bold]{self._process_name}[/bold] [dim]#{self._run_id}[/dim]  "
            f"[{self._status_color(status)}]{label}[/]  [dim]{elapsed}[/dim]"
        )
        meta = f"[dim]Goal:[/] {self._goal}"
        if task_id:
            meta += f"\n[dim]Task:[/] {task_id}"
        meta += (
            "\n[dim]Close: Ctrl+W | /run close [run-id-prefix] | "
            "Ctrl+P: Close process run tab[/dim]"
        )
        self._meta.update(meta)
        can_restart = status == "failed" and bool(task_id.strip())
        self._actions.display = can_restart
        self._restart_button.display = can_restart
        self._restart_button.disabled = not can_restart

    def set_tasks(self, tasks: list[dict]) -> None:
        """Replace task rows shown in the progress section."""
        if not self.is_attached:
            self._pending_tasks = list(tasks)
            return
        self._progress.set_rows(tasks)

    def set_outputs(self, outputs: list[dict]) -> None:
        """Replace output rows shown in the outputs section."""
        if not self.is_attached:
            self._pending_outputs = list(outputs)
            return
        self._outputs.set_rows(outputs)

    def add_activity(self, text: str) -> None:
        """Append informational activity text."""
        safe_text = _escape_markup_text(text)
        if not self.is_attached:
            self._pending_activity.append(safe_text)
            return
        self._log.add_info(safe_text)

    def add_result(self, text: str, *, success: bool) -> None:
        """Append final result text."""
        safe_text = _escape_markup_text(text)
        if not self.is_attached:
            self._pending_results.append((safe_text, success))
            return
        if success:
            self._log.add_model_text(safe_text)
            return
        self._log.add_model_text(
            f"[bold #f7768e]Error:[/] {safe_text}",
            markup=True,
        )

    def on_mount(self) -> None:
        """Flush updates queued before the pane was attached to the DOM."""
        if self._pending_tasks is not None:
            self._progress.set_rows(self._pending_tasks)
            self._pending_tasks = None
        if self._pending_outputs is not None:
            self._outputs.set_rows(self._pending_outputs)
            self._pending_outputs = None
        if self._pending_activity:
            for line in self._pending_activity:
                self._log.add_info(line)
            self._pending_activity.clear()
        if self._pending_results:
            for text, success in self._pending_results:
                if success:
                    self._log.add_model_text(text)
                else:
                    self._log.add_model_text(
                        f"[bold #f7768e]Error:[/] {text}",
                        markup=True,
                    )
            self._pending_results.clear()

    @staticmethod
    def _status_color(status: str) -> str:
        if status == "completed":
            return "#9ece6a"
        if status == "failed":
            return "#f7768e"
        if status == "running":
            return "#7dcfff"
        return "#a9b1d6"


@dataclass
class ProcessRunState:
    """In-memory state for a single process run tab."""

    run_id: str
    process_name: str
    goal: str
    run_workspace: Path
    process_defn: ProcessDefinition | None
    pane_id: str
    pane: ProcessRunPane
    status: str = "queued"
    task_id: str = ""
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    tasks: list[dict] = field(default_factory=list)
    task_labels: dict[str, str] = field(default_factory=dict)
    last_progress_message: str = ""
    last_progress_at: float = 0.0
    activity_log: list[str] = field(default_factory=list)
    result_log: list[dict] = field(default_factory=list)
    worker: object | None = None
    closed: bool = False
    is_adhoc: bool = False
    recommended_tools: list[str] = field(default_factory=list)
    goal_context_overrides: dict[str, Any] = field(default_factory=dict)
    auth_profile_overrides: dict[str, str] = field(default_factory=dict)
    auth_required_resources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AdhocProcessCacheEntry:
    """Cached ad hoc process synthesized for `/run` goals."""

    key: str
    goal: str
    process_defn: ProcessDefinition
    recommended_tools: list[str] = field(default_factory=list)
    spec: dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.monotonic)


class LoomApp(App):
    """Loom TUI — the unified interactive cowork interface."""

    TITLE = "Loom"
    COMMANDS = {LoomCommands}

    CSS = """
    #main-layout {
        height: 1fr;
    }
    #main-area {
        width: 1fr;
    }
    #bottom-stack {
        dock: bottom;
        height: 4;
        background: $panel;
    }
    #input-top-rule {
        height: 1;
        border: none;
        border-top: solid $primary-darken-1;
        border-left: solid $primary-darken-2;
        border-right: solid $primary-darken-2;
        background: $panel;
    }
    #user-input {
        margin: 0;
        height: 2;
        padding: 0 1;
        background: $panel;
        border: none;
        border-left: solid $primary-darken-2;
        border-right: solid $primary-darken-2;
        border-bottom: solid $primary-darken-1;
        color: $text;
    }
    #user-input:focus {
        border: none;
        border-left: solid $primary-darken-1;
        border-right: solid $primary-darken-1;
        border-bottom: solid $primary;
        background: $surface;
    }
    #status-bar {
        display: none;
    }
    #app-footer {
        dock: none;
        height: 1;
        border: none;
    }
    #slash-hint {
        dock: bottom;
        display: none;
        margin: 0 0 4 0;
        min-height: 1;
        max-height: 14;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "request_quit", "Quit", show=True, priority=True),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("ctrl+r", "reload_workspace", "Reload", show=True),
        Binding("ctrl+w", "close_process_tab", "Close Run", show=True),
        Binding("ctrl+1", "tab_chat", "Chat"),
        Binding("ctrl+2", "tab_files", "Files"),
        Binding("ctrl+3", "tab_events", "Events"),
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
        process_name: str | None = None,
        explicit_mcp_path: Path | None = None,
        legacy_config_path: Path | None = None,
        explicit_auth_path: Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._tools = tools
        self._workspace = workspace
        self._config = config
        self._db = db
        self._store = store
        self._resume_session = resume_session
        self._process_name = process_name
        self._explicit_mcp_path = explicit_mcp_path
        self._legacy_config_path = legacy_config_path
        self._explicit_auth_path = explicit_auth_path
        self._process_defn: ProcessDefinition | None = None
        self._session: CoworkSession | None = None
        self._chat_busy = False
        self._total_tokens = 0

        # Approval state — resolved via Textual modal
        self._approval_event: asyncio.Event | None = None
        self._approval_result: ApprovalDecision = ApprovalDecision.DENY

        # Tools that need late-binding to session
        self._recall_tool = None
        self._delegate_tool = None
        self._confirm_exit_waiter: asyncio.Future[bool] | None = None
        self._slash_cycle_seed: str = ""
        self._slash_cycle_candidates: list[str] = []
        self._applying_slash_tab_completion = False
        self._skip_slash_cycle_reset_once = False
        self._input_history: list[str] = []
        self._input_history_nav_index: int | None = None
        self._input_history_nav_draft: str = ""
        self._applying_input_history_navigation = False
        self._skip_input_history_reset_once = False
        self._process_runs: dict[str, ProcessRunState] = {}
        self._process_elapsed_timer = None
        self._process_command_map: dict[str, str] = {}
        self._blocked_process_commands: list[str] = []
        self._cached_process_catalog: list[dict[str, str]] = []
        self._adhoc_process_cache: dict[str, AdhocProcessCacheEntry] = {}
        self._adhoc_package_doc_cache: str | None = None
        self._sidebar_cowork_tasks: list[dict] = []
        self._process_close_hint_shown = False
        self._close_process_tab_inflight = False
        self._auto_resume_workspace_on_init = True
        self._run_auth_profile_overrides: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-layout"):
            yield Sidebar(self._workspace, id="sidebar")
            with Vertical(id="main-area"):
                with TabbedContent(id="tabs"):
                    with TabPane("Chat", id="tab-chat"):
                        yield ChatLog(id="chat-log")
                    with TabPane("Files", id="tab-files"):
                        yield FilesChangedPanel(id="files-panel")
                    with TabPane("Events", id="tab-events"):
                        yield EventPanel(id="events-panel")
        yield Static("", id="slash-hint")
        yield StatusBar(id="status-bar")
        with Vertical(id="bottom-stack"):
            yield Static("", id="input-top-rule")
            yield Input(
                placeholder="Type a message... (Enter to send)",
                id="user-input",
            )
            yield Footer(id="app-footer")

    async def on_mount(self) -> None:
        # Register and activate theme
        self.register_theme(LOOM_DARK)
        self.theme = "loom-dark"
        self._process_elapsed_timer = self.set_interval(
            1.0,
            self._tick_process_run_elapsed,
        )

        if self._model is None:
            # No model configured — launch the setup wizard
            self.push_screen(
                SetupScreen(), callback=self._on_setup_complete,
            )
            return

        await self._initialize_session()
        # Keep input focus deterministic even when _initialize_session is mocked
        # in tests or returns early in partial startup paths.
        try:
            self.query_one("#user-input", Input).focus()
        except Exception:
            pass

    def _on_setup_complete(self, result: list[dict] | None) -> None:
        """Handle setup wizard dismissal."""
        if result is None:
            self.exit()
            return
        self._finalize_setup()

    @work
    async def _finalize_setup(self) -> None:
        """Reload config and initialize after setup wizard completes."""
        from loom.config import Config, load_config
        from loom.mcp.config import apply_mcp_overrides
        from loom.models.router import ModelRouter

        loaded = load_config()
        if isinstance(loaded, Config):
            self._config = apply_mcp_overrides(
                loaded,
                workspace=self._workspace,
            )
        else:
            # Defensive fallback for mocked/non-standard config objects.
            self._config = loaded
        router = ModelRouter.from_config(self._config)
        try:
            self._model = router.select(role="executor")
        except Exception as e:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info(
                f"[bold #f7768e]Setup error: {e}[/]\n"
                f"Edit ~/.loom/loom.toml or run /setup to try again."
            )
            return

        # P1-5: If re-running /setup during an active session, invalidate
        # the old session so _initialize_session creates a fresh one with
        # the new model and system prompt.
        if self._session is not None:
            if self._store and self._session.session_id:
                await self._store.update_session(
                    self._session.session_id, is_active=False,
                )
            self._session = None

        await self._initialize_session()

    def _refresh_tool_registry(self) -> None:
        """Reset registry to discovered tools."""
        from loom.tools import create_default_registry

        self._tools = create_default_registry(self._config)

    def _create_process_loader(self):
        """Create a process loader for the current workspace/config."""
        from loom.processes.schema import ProcessLoader

        extra: list[Path] = []
        if self._config:
            extra = [Path(p) for p in self._config.process.search_paths]
        return ProcessLoader(
            workspace=self._workspace,
            extra_search_paths=extra,
            require_rule_scope_metadata=bool(
                getattr(
                    getattr(self._config, "process", None),
                    "require_rule_scope_metadata",
                    False,
                ),
            ),
            require_v2_contract=bool(
                getattr(
                    getattr(self._config, "process", None),
                    "require_v2_contract",
                    False,
                ),
            ),
        )

    def _active_process_name(self) -> str:
        """Return active process display name."""
        if self._process_defn:
            return self._process_defn.name
        return "none"

    def _mcp_manager(self):
        """Build MCP config manager scoped to current app workspace."""
        from loom.mcp.config import MCPConfigManager

        return MCPConfigManager(
            config=self._config,
            workspace=self._workspace,
            explicit_path=self._explicit_mcp_path,
            legacy_config_path=self._legacy_config_path,
        )

    def _open_mcp_manager_screen(self) -> None:
        """Open modal MCP manager and reload runtime when changed."""
        manager = self._mcp_manager()

        def _handle_result(result: dict[str, object] | None) -> None:
            if not isinstance(result, dict):
                return
            if not result.get("changed"):
                return
            self.run_worker(
                self._reload_mcp_runtime(),
                group="mcp-manager-refresh",
                exclusive=True,
            )
            try:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_info("MCP configuration updated.")
            except Exception:
                pass

        self.push_screen(
            MCPManagerScreen(
                manager,
                explicit_auth_path=self._explicit_auth_path,
            ),
            callback=_handle_result,
        )

    def _open_auth_manager_screen(self) -> None:
        """Open modal auth manager."""

        def _handle_result(result: dict[str, object] | None) -> None:
            if not isinstance(result, dict):
                return
            if not result.get("changed"):
                return
            try:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_info("Auth configuration updated.")
            except Exception:
                pass

        self.push_screen(
            AuthManagerScreen(
                workspace=self._workspace,
                explicit_auth_path=self._explicit_auth_path,
                mcp_manager=self._mcp_manager(),
                process_def=self._process_defn,
                tool_registry=self._tools,
            ),
            callback=_handle_result,
        )

    def _auth_defaults_path(self) -> Path:
        from loom.auth.config import default_workspace_auth_defaults_path

        return default_workspace_auth_defaults_path(self._workspace)

    @staticmethod
    def _split_slash_args(raw: str) -> list[str]:
        """Split slash-command argument string using shell-like quoting."""
        try:
            return shlex.split(raw)
        except ValueError as e:
            raise ValueError(f"Invalid quoted argument syntax: {e}") from e

    @staticmethod
    def _truncate_run_goal_file_content(content: str) -> tuple[str, bool]:
        """Bound file-input size for `/run` goal context and synthesis prompts."""
        if len(content) <= _RUN_GOAL_FILE_CONTENT_MAX_CHARS:
            return content, False
        return content[:_RUN_GOAL_FILE_CONTENT_MAX_CHARS].rstrip(), True

    def _resolve_run_goal_file_path(self, raw_path: str) -> Path | None:
        """Resolve a `/run` file-input token to a workspace-local file path."""
        token = str(raw_path or "").strip()
        if not token:
            return None

        candidate = Path(token).expanduser()
        if not candidate.is_absolute():
            candidate = self._workspace / candidate
        try:
            resolved = candidate.resolve()
            workspace_root = self._workspace.resolve()
        except OSError:
            return None
        if not resolved.is_file():
            return None
        try:
            resolved.relative_to(workspace_root)
        except ValueError:
            return None
        return resolved

    def _expand_run_goal_file_input(
        self,
        goal_tokens: list[str],
    ) -> tuple[str, str, dict[str, Any], str | None]:
        """Expand optional `/run` file shorthand into goal/context payloads.

        Returns:
            (execution_goal, synthesis_goal, context_overrides, error_message)
        """
        goal_text = " ".join(str(token or "").strip() for token in goal_tokens).strip()
        if not goal_tokens:
            return goal_text, goal_text, {}, None

        first = str(goal_tokens[0] or "").strip()
        first_path_token = first[1:] if first.startswith("@") else first
        should_treat_as_file = bool(first.startswith("@") or len(goal_tokens) == 1)
        if not should_treat_as_file:
            return goal_text, goal_text, {}, None

        resolved = self._resolve_run_goal_file_path(first_path_token)
        if resolved is None:
            if first.startswith("@"):
                return (
                    goal_text,
                    goal_text,
                    {},
                    f"Run goal file not found (or outside workspace): {first_path_token}",
                )
            return goal_text, goal_text, {}, None

        try:
            raw_content = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return (
                goal_text,
                goal_text,
                {},
                f"Failed reading run goal file '{first_path_token}': {e}",
            )

        content, truncated = self._truncate_run_goal_file_content(raw_content)
        workspace_root = self._workspace.resolve()
        try:
            file_label = str(resolved.relative_to(workspace_root))
        except ValueError:
            file_label = str(resolved)

        user_goal = ""
        if first.startswith("@") and len(goal_tokens) > 1:
            user_goal = " ".join(
                str(token or "").strip() for token in goal_tokens[1:]
            ).strip()

        if user_goal:
            execution_goal = user_goal
            preface = (
                f"{user_goal}\n\n"
                f"Supplemental task specification file: {file_label}\n"
                "Use the file content below as authoritative detail."
            )
        else:
            execution_goal = file_label
            preface = (
                f"Use the following task specification from file "
                f"'{file_label}' as the primary goal."
            )

        truncated_note = ""
        if truncated:
            truncated_note = (
                f"\n\n[File content truncated to first "
                f"{_RUN_GOAL_FILE_CONTENT_MAX_CHARS} characters.]"
            )
        synthesis_goal = (
            f"{preface}\n\n"
            f"--- BEGIN FILE: {file_label} ---\n"
            f"{content}\n"
            f"--- END FILE: {file_label} ---"
            f"{truncated_note}"
        )
        return (
            execution_goal,
            synthesis_goal,
            {
                "run_goal_file_input": {
                    "path": file_label,
                    "content": content,
                    "truncated": truncated,
                    "max_chars": _RUN_GOAL_FILE_CONTENT_MAX_CHARS,
                },
            },
            None,
        )

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

    def _load_process_definition(self, chat: ChatLog) -> None:
        """Load active process definition and import bundled tools."""
        self._process_defn = None
        if not self._process_name:
            return
        if self._is_reserved_process_name(self._process_name):
            chat.add_info(
                f"[bold #f7768e]Process '{self._process_name}' conflicts with "
                "a built-in slash command and was not loaded.[/]"
            )
            self._process_name = None
            return

        loader = self._create_process_loader()
        try:
            self._process_defn = loader.load(self._process_name)
            if self._is_reserved_process_name(self._process_defn.name):
                chat.add_info(
                    f"[bold #f7768e]Process '{self._process_defn.name}' conflicts "
                    "with a built-in slash command and was not loaded.[/]"
                )
                self._process_defn = None
                self._process_name = None
                return
            # Process load may import bundled tool modules; rebuild to include
            # them in the active registry.
            self._refresh_tool_registry()
            chat.add_info(
                f"Process: [bold]{self._process_defn.name}[/bold] "
                f"v{self._process_defn.version}"
            )
        except Exception as e:
            chat.add_info(
                f"[bold #f7768e]Failed to load process "
                f"'{self._process_name}': {e}[/]"
            )

    @staticmethod
    def _adhoc_cache_key(goal: str) -> str:
        """Build a stable cache key for a run goal."""
        normalized = " ".join(str(goal or "").strip().lower().split())
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        return digest[:16]

    @staticmethod
    def _adhoc_cache_dir() -> Path:
        """Return on-disk cache directory for ad hoc process specs."""
        return Path.home() / ".loom" / "cache" / "adhoc-processes"

    def _adhoc_synthesis_log_path(self) -> Path:
        """Return diagnostics log path for ad hoc synthesis internals."""
        configured = getattr(getattr(self._config, "logging", None), "event_log_path", "")
        root = Path(str(configured).strip()).expanduser() if str(configured).strip() else (
            Path.home() / ".loom" / "logs"
        )
        return root / "adhoc-synthesis.jsonl"

    def _adhoc_synthesis_artifact_root(self) -> Path:
        """Return directory root for per-run ad hoc synthesis artifacts."""
        return self._adhoc_synthesis_log_path().parent / "adhoc-synthesis"

    def _create_adhoc_synthesis_artifact_dir(self, *, key: str, goal: str) -> Path | None:
        """Create a per-run artifact directory for ad hoc synthesis."""
        try:
            root = self._adhoc_synthesis_artifact_root()
            stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            goal_slug = self._sanitize_kebab_token(
                goal,
                fallback="goal",
                max_len=24,
            )
            run_id = uuid.uuid4().hex[:6]
            run_dir = root / f"{stamp}-{key}-{goal_slug}-{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        except Exception as e:
            logger.warning("Failed creating ad hoc synthesis artifact dir: %s", e)
            return None

    @staticmethod
    def _write_adhoc_synthesis_artifact_text(
        artifact_dir: Path | None,
        filename: str,
        content: str,
    ) -> None:
        """Write a text artifact into the synthesis run directory."""
        if artifact_dir is None:
            return
        try:
            path = artifact_dir / filename
            path.write_text(str(content or ""), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed writing ad hoc synthesis artifact %s: %s", filename, e)

    @staticmethod
    def _write_adhoc_synthesis_artifact_yaml(
        artifact_dir: Path | None,
        filename: str,
        payload: dict[str, Any] | None,
    ) -> None:
        """Write a YAML artifact into the synthesis run directory."""
        if artifact_dir is None or not isinstance(payload, dict):
            return
        try:
            import yaml

            path = artifact_dir / filename
            path.write_text(
                yaml.safe_dump(
                    payload,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed writing ad hoc synthesis YAML artifact %s: %s", filename, e)

    def _append_adhoc_synthesis_log(self, payload: dict[str, Any]) -> Path | None:
        """Append one ad hoc synthesis diagnostic entry to disk."""
        if not isinstance(payload, dict) or not payload:
            return None
        path = self._adhoc_synthesis_log_path()
        record = {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            **payload,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return path
        except Exception as e:
            logger.warning("Failed writing ad hoc synthesis log %s: %s", path, e)
            return None

    def _adhoc_cache_path(self, key: str) -> Path:
        """Return cache file path for an ad hoc cache key."""
        safe_key = self._sanitize_kebab_token(
            str(key or ""),
            fallback="adhoc",
            max_len=64,
        ).replace("-", "")
        if not safe_key:
            safe_key = "adhoc"
        return self._adhoc_cache_dir() / f"{safe_key}.yaml"

    def _adhoc_legacy_cache_path(self, key: str) -> Path:
        """Return legacy JSON cache file path for ad hoc cache key."""
        safe_key = self._sanitize_kebab_token(
            str(key or ""),
            fallback="adhoc",
            max_len=64,
        ).replace("-", "")
        if not safe_key:
            safe_key = "adhoc"
        return self._adhoc_cache_dir() / f"{safe_key}.json"

    @classmethod
    def _spec_from_process_defn(
        cls,
        process_defn: ProcessDefinition,
        *,
        recommended_tools: list[str],
    ) -> dict[str, Any]:
        """Serialize a ProcessDefinition into ad hoc spec-shaped payload."""
        phases = [
            {
                "id": str(phase.id or "").strip(),
                "description": str(phase.description or "").strip(),
                "depends_on": [
                    str(dep).strip()
                    for dep in list(phase.depends_on)
                    if str(dep).strip()
                ],
                "acceptance_criteria": str(phase.acceptance_criteria or "").strip(),
                "deliverables": [
                    str(item).strip()
                    for item in list(phase.deliverables)
                    if str(item).strip()
                ],
            }
            for phase in list(process_defn.phases)
        ]
        return {
            "intent": cls._infer_adhoc_intent_from_phases(phases),
            "name": str(process_defn.name or "").strip(),
            "description": str(process_defn.description or "").strip(),
            "persona": str(process_defn.persona or "").strip(),
            "phase_mode": str(process_defn.phase_mode or "guided").strip(),
            "tool_guidance": str(process_defn.tool_guidance or "").strip(),
            "required_tools": [
                str(item).strip()
                for item in list(getattr(process_defn.tools, "required", []) or [])
                if str(item).strip()
            ],
            "recommended_tools": [
                str(item).strip()
                for item in recommended_tools
                if str(item).strip()
            ],
            "phases": phases,
        }

    def _persist_adhoc_cache_entry(self, entry: AdhocProcessCacheEntry) -> Path:
        """Persist synthesized ad hoc process definition to ~/.loom/cache."""
        import yaml

        cache_path = self._adhoc_cache_path(entry.key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        spec_payload = entry.spec or self._spec_from_process_defn(
            entry.process_defn,
            recommended_tools=entry.recommended_tools,
        )
        payload: dict[str, Any] = {
            "key": entry.key,
            "goal": entry.goal,
            "generated_at_monotonic": float(entry.generated_at),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "spec": spec_payload,
        }
        cache_path.write_text(
            yaml.safe_dump(
                payload,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return cache_path

    def _load_adhoc_cache_entry_from_disk(self, key: str) -> AdhocProcessCacheEntry | None:
        """Load ad hoc process cache entry from ~/.loom/cache when present."""
        import yaml

        cache_path = self._adhoc_cache_path(key)
        legacy_path = self._adhoc_legacy_cache_path(key)
        read_path: Path | None = None
        if cache_path.exists():
            read_path = cache_path
        elif legacy_path.exists():
            read_path = legacy_path
        else:
            return None

        try:
            raw_text = read_path.read_text(encoding="utf-8")
            if read_path.suffix.lower() == ".json":
                payload = json.loads(raw_text)
            else:
                payload = yaml.safe_load(raw_text)
        except Exception as e:
            logger.warning("Failed to read ad hoc cache '%s': %s", read_path, e)
            return None
        if not isinstance(payload, dict):
            return None

        goal = str(payload.get("goal", "")).strip()
        raw_spec = payload.get("spec")
        if not goal or not isinstance(raw_spec, dict):
            return None

        raw_intent = self._normalize_adhoc_intent(
            str(raw_spec.get("intent", "")),
            default="",
        )
        if not raw_intent:
            # Legacy cache entries (pre-intent) are treated as stale so /run
            # re-synthesizes with LLM-selected intent.
            return None

        normalized = self._normalize_adhoc_spec(
            raw_spec,
            goal=goal,
            key=key,
            available_tools=self._available_tool_names(),
            intent=raw_intent,
        )
        entry = self._build_adhoc_cache_entry(
            key=key,
            goal=goal,
            spec=normalized,
        )
        generated_at = payload.get("generated_at_monotonic")
        if isinstance(generated_at, (int, float)):
            entry.generated_at = float(generated_at)
        if read_path.suffix.lower() == ".json":
            try:
                self._persist_adhoc_cache_entry(entry)
            except Exception:
                pass
        return entry

    @staticmethod
    def _sanitize_synthesis_trace(raw: dict[str, Any] | None) -> dict[str, Any]:
        """Preserve only simple scalar fields from synthesis diagnostics."""
        if not isinstance(raw, dict):
            return {}
        clean: dict[str, Any] = {}
        for key, value in raw.items():
            name = str(key or "").strip()
            if not name:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean[name] = value
        return clean

    @staticmethod
    def _sanitize_kebab_token(value: str, *, fallback: str, max_len: int = 48) -> str:
        """Normalize free-form text into a kebab-case token."""
        lowered = str(value or "").strip().lower()
        token = re.sub(r"[^a-z0-9-]+", "-", lowered)
        token = re.sub(r"-{2,}", "-", token).strip("-")
        if not token:
            token = fallback
        if len(token) > max_len:
            token = token[:max_len].strip("-")
        if not token:
            token = fallback
        return token

    @staticmethod
    def _sanitize_deliverable_name(value: str, *, fallback: str) -> str:
        """Normalize deliverable path names for generated ad hoc processes."""
        raw = str(value or "").strip()
        if not raw:
            return fallback
        # Strip optional "filename — description" suffixes.
        raw = raw.split("—")[0].split(" - ")[0].strip()
        raw = raw.replace("\\", "/").lstrip("/")
        parts = [p for p in raw.split("/") if p and p not in {".", ".."}]
        if not parts:
            return fallback
        safe_parts: list[str] = []
        for part in parts:
            safe = re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip("-")
            if safe:
                safe_parts.append(safe)
        if not safe_parts:
            return fallback
        candidate = "/".join(safe_parts)
        if "." not in Path(candidate).name:
            candidate += ".md"
        return candidate

    def _available_tool_names(self) -> list[str]:
        """Return sorted available tool names from the active registry."""
        try:
            tools = self._tools.list_tools()
        except Exception:
            return []
        names = sorted({
            str(name or "").strip()
            for name in tools
            if str(name or "").strip()
        })
        return names

    def _adhoc_package_contract_hint(self) -> str:
        """Load full package authoring reference doc for ad hoc synthesis."""
        cached = self._adhoc_package_doc_cache
        if isinstance(cached, str) and cached:
            return cached

        candidates = [
            Path(__file__).resolve().parents[3] / "docs" / "creating-packages.md",
            self._workspace / "docs" / "creating-packages.md",
            Path.cwd() / "docs" / "creating-packages.md",
        ]
        for path in candidates:
            try:
                if path.exists() and path.is_file():
                    doc = path.read_text(encoding="utf-8")
                    text = (
                        f"Reference document path: {path}\n"
                        "Use this full reference when designing the ad hoc process "
                        "package contract.\n\n"
                        f"{doc}"
                    )
                    self._adhoc_package_doc_cache = text
                    return text
            except Exception:
                continue

        fallback = (
            "Reference document unavailable: docs/creating-packages.md.\n"
            "Use Loom package conventions: kebab-case name, schema_version: 2, "
            "phases with deliverables and acceptance criteria, strict|guided|suggestive "
            "phase_mode, and tools.required drawn from available tools."
        )
        self._adhoc_package_doc_cache = fallback
        return fallback

    @staticmethod
    def _extract_json_payload(
        raw_text: str,
        *,
        expected_keys: tuple[str, ...] = (),
    ) -> dict[str, Any] | None:
        """Best-effort structured payload extraction from model output."""
        text = str(raw_text or "").strip()
        if not text:
            return None
        # Normalize typographic quotes that commonly break JSON parsing.
        text = (
            text.replace("“", '"')
            .replace("”", '"')
            .replace("’", "'")
            .replace("‘", "'")
        )

        def _strip_wrapping_fence(value: str) -> str:
            candidate = value.strip()
            if not candidate.startswith("```"):
                return candidate
            lines = candidate.splitlines()
            if lines:
                lines = lines[1:]
            while lines and not lines[-1].strip():
                lines.pop()
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()

        def _parse_blob(blob: str) -> dict[str, Any] | None:
            source = _strip_wrapping_fence(blob)
            if not source:
                return None

            decoder = json.JSONDecoder()
            candidates: list[dict[str, Any]] = []
            try:
                parsed = decoder.decode(source)
                if isinstance(parsed, dict):
                    candidates.append(parsed)
            except Exception:
                pass

            for idx, ch in enumerate(source):
                if ch != "{":
                    continue
                try:
                    parsed, _end = decoder.raw_decode(source[idx:])
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    candidates.append(parsed)

            if expected_keys:
                for payload in candidates:
                    if all(key in payload for key in expected_keys):
                        return payload
            if candidates:
                return candidates[0]

            # Some models drift into YAML-like output despite JSON instructions.
            try:
                import yaml

                parsed_yaml = yaml.safe_load(source)
                if isinstance(parsed_yaml, dict):
                    return parsed_yaml
            except Exception:
                pass
            return None

        payloads: list[dict[str, Any]] = []

        direct = _parse_blob(text)
        if direct is not None:
            payloads.append(direct)

        # Try fenced code blocks anywhere in the response, not just full-body fences.
        for match in re.finditer(r"```(?:json|yaml|yml)?\s*([\s\S]*?)```", text, re.IGNORECASE):
            block = str(match.group(1) or "").strip()
            if not block:
                continue
            parsed = _parse_blob(block)
            if parsed is not None:
                payloads.append(parsed)

        # Try to recover YAML objects from markdown/prose preambles.
        if expected_keys:
            lines = text.splitlines()
            lowered_key_prefixes = tuple(
                f"{str(key).strip().lower()}:"
                for key in expected_keys
                if str(key).strip()
            )
            for idx, line in enumerate(lines):
                lower = str(line).strip().lower()
                if not lowered_key_prefixes:
                    break
                if not any(lower.startswith(prefix) for prefix in lowered_key_prefixes):
                    continue
                snippet = "\n".join(lines[idx:]).strip()
                if not snippet:
                    continue
                parsed = _parse_blob(snippet)
                if parsed is not None:
                    payloads.append(parsed)
                    break

        if expected_keys:
            for payload in payloads:
                if all(key in payload for key in expected_keys):
                    return payload
            # When an explicit schema is expected, avoid returning nested/partial
            # dicts (e.g., truncated JSON where only one phase object parses).
            return None

        return payloads[0] if payloads else None

    @staticmethod
    def _synthesis_preview(text: str, *, max_chars: int = 480) -> str:
        """Compact a model response preview for diagnostics logs."""
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip() + "..."

    @staticmethod
    def _raw_adhoc_spec_needs_minimal_retry(raw: dict[str, Any] | None) -> bool:
        """Return True when parsed raw spec is too incomplete to trust directly."""
        if not isinstance(raw, dict):
            return True
        phases_raw = raw.get("phases", [])
        if not isinstance(phases_raw, list):
            return True
        phase_rows = [item for item in phases_raw if isinstance(item, dict)]
        # Keep this permissive: simple goals can legitimately need ~3 phases.
        if len(phase_rows) < 3:
            return True
        for phase in phase_rows:
            if not str(phase.get("description", "")).strip():
                return True
            raw_deliverables = phase.get("deliverables", [])
            if isinstance(raw_deliverables, str):
                raw_deliverables = [raw_deliverables]
            if not isinstance(raw_deliverables, list):
                return True
            if not any(str(item or "").strip() for item in raw_deliverables):
                return True
        return False

    @staticmethod
    def _normalize_adhoc_intent(intent: str, *, default: str = "research") -> str:
        """Normalize ad hoc intent labels to a supported enum-like value."""
        text = str(intent or "").strip().lower()
        if text in {"research", "writing", "build"}:
            return text
        fallback = str(default or "").strip().lower()
        if not fallback:
            return ""
        if fallback in {"research", "writing", "build"}:
            return fallback
        return "research"

    @classmethod
    def _infer_adhoc_intent_from_phases(cls, phases: list[dict[str, Any]]) -> str:
        """Infer intent from phase semantics when explicit intent is unavailable."""
        if cls._phases_satisfy_intent(phases, "build"):
            return "build"
        if cls._phases_satisfy_intent(phases, "writing"):
            return "writing"
        if cls._phases_satisfy_intent(phases, "research"):
            return "research"
        return "research"

    @classmethod
    def _resolve_adhoc_intent(
        cls,
        raw: dict[str, Any] | None,
        *,
        intent_hint: str | None = None,
    ) -> str:
        """Resolve intent from model-provided fields, then phase semantics."""
        normalized_hint = cls._normalize_adhoc_intent(
            str(intent_hint or ""),
            default="",
        )
        if normalized_hint:
            return normalized_hint
        if not isinstance(raw, dict):
            return "research"
        for key in ("intent", "goal_intent", "request_intent", "goal_type"):
            value = cls._normalize_adhoc_intent(str(raw.get(key, "")), default="")
            if value:
                return value
        raw_phases = raw.get("phases", [])
        if isinstance(raw_phases, list):
            phase_rows = [item for item in raw_phases if isinstance(item, dict)]
            if phase_rows:
                return cls._infer_adhoc_intent_from_phases(phase_rows)
        return "research"

    @staticmethod
    def _adhoc_intent_progression(intent: str) -> str:
        """Return phase progression guidance for the inferred intent."""
        if intent == "build":
            return (
                "scope -> implementation plan/design -> implement/build -> "
                "test/verify -> package/handoff -> final delivery"
            )
        if intent == "writing":
            return (
                "scope -> outline -> draft -> revise/edit -> "
                "editorial verification -> final delivery"
            )
        return (
            "scope -> source planning -> evidence collection -> "
            "analysis/synthesis -> verification -> final delivery"
        )

    @staticmethod
    def _adhoc_intent_phase_blueprint(intent: str, slug: str) -> list[dict[str, Any]]:
        """Return deterministic phase blueprint for a goal intent."""
        if intent == "build":
            return [
                {
                    "id": "scope-and-constraints",
                    "description": (
                        "Clarify functional goals, constraints, and acceptance criteria."
                    ),
                    "depends_on": [],
                    "acceptance_criteria": (
                        "Requirements and success criteria are explicit and testable."
                    ),
                    "deliverables": [f"{slug}-requirements.md"],
                },
                {
                    "id": "implementation-plan",
                    "description": "Design implementation approach and execution plan.",
                    "depends_on": ["scope-and-constraints"],
                    "acceptance_criteria": (
                        "Plan maps required changes, affected files, and sequencing."
                    ),
                    "deliverables": [f"{slug}-implementation-plan.md"],
                },
                {
                    "id": "implement-solution",
                    "description": "Execute implementation changes to satisfy requirements.",
                    "depends_on": ["implementation-plan"],
                    "acceptance_criteria": (
                        "Requested solution is implemented and integrated without regressions."
                    ),
                    "deliverables": [f"{slug}-implementation-summary.md"],
                },
                {
                    "id": "test-and-verify",
                    "description": "Run validations and verify behavior against requirements.",
                    "depends_on": ["implement-solution"],
                    "acceptance_criteria": (
                        "Verification evidence confirms behavior, edge cases, and quality."
                    ),
                    "deliverables": [f"{slug}-verification.md"],
                },
                {
                    "id": "package-and-handoff",
                    "description": "Prepare final artifacts, notes, and operational guidance.",
                    "depends_on": ["test-and-verify"],
                    "acceptance_criteria": (
                        "Artifacts and notes are complete, actionable, and ready for handoff."
                    ),
                    "deliverables": [f"{slug}-handoff.md"],
                },
                {
                    "id": "deliver-final",
                    "description": "Deliver final outcome with summary of what changed and why.",
                    "depends_on": ["package-and-handoff"],
                    "acceptance_criteria": (
                        "Final output is complete, validated, and aligned with the goal."
                    ),
                    "deliverables": [f"{slug}-final.md"],
                },
            ]
        if intent == "writing":
            return [
                {
                    "id": "scope-and-constraints",
                    "description": (
                        "Clarify audience, objective, constraints, tone, and output format."
                    ),
                    "depends_on": [],
                    "acceptance_criteria": (
                        "Writing brief defines objective, audience, and quality bar."
                    ),
                    "deliverables": [f"{slug}-brief.md"],
                },
                {
                    "id": "outline-and-sources",
                    "description": (
                        "Create outline and gather source material or supporting points."
                    ),
                    "depends_on": ["scope-and-constraints"],
                    "acceptance_criteria": (
                        "Outline and supporting material are sufficient for a full draft."
                    ),
                    "deliverables": [f"{slug}-outline.md"],
                },
                {
                    "id": "draft-content",
                    "description": "Write the first complete draft.",
                    "depends_on": ["outline-and-sources"],
                    "acceptance_criteria": (
                        "Draft covers required content and major claims end-to-end."
                    ),
                    "deliverables": [f"{slug}-draft.md"],
                },
                {
                    "id": "revise-and-edit",
                    "description": "Revise structure, clarity, and argument quality.",
                    "depends_on": ["draft-content"],
                    "acceptance_criteria": (
                        "Revisions improve coherence, clarity, and reader usefulness."
                    ),
                    "deliverables": [f"{slug}-revised.md"],
                },
                {
                    "id": "verify-quality",
                    "description": "Perform editorial and factual quality checks.",
                    "depends_on": ["revise-and-edit"],
                    "acceptance_criteria": (
                        "Claims, consistency, and style pass defined quality checks."
                    ),
                    "deliverables": [f"{slug}-verification.md"],
                },
                {
                    "id": "deliver-final",
                    "description": "Deliver final polished piece and rationale for key choices.",
                    "depends_on": ["verify-quality"],
                    "acceptance_criteria": (
                        "Final output is publication-ready for the stated objective."
                    ),
                    "deliverables": [f"{slug}-final.md"],
                },
            ]
        return [
            {
                "id": "scope-and-constraints",
                "description": (
                    "Clarify objective, constraints, assumptions, and success criteria."
                ),
                "depends_on": [],
                "acceptance_criteria": (
                    "Scope, constraints, and success criteria are explicit and actionable."
                ),
                "deliverables": [f"{slug}-brief.md"],
            },
            {
                "id": "source-plan",
                "description": (
                    "Define research strategy, source classes, and evidence standards."
                ),
                "depends_on": ["scope-and-constraints"],
                "acceptance_criteria": (
                    "Research plan lists source priorities, collection method, and "
                    "quality checks."
                ),
                "deliverables": [f"{slug}-source-plan.md"],
            },
            {
                "id": "collect-evidence",
                "description": "Gather and verify relevant evidence and source data.",
                "depends_on": ["source-plan"],
                "acceptance_criteria": (
                    "Evidence log captures sufficient, credible, and attributable sources."
                ),
                "deliverables": [f"{slug}-evidence.md"],
            },
            {
                "id": "analyze-findings",
                "description": (
                    "Analyze evidence, compare alternatives, and synthesize conclusions."
                ),
                "depends_on": ["collect-evidence"],
                "acceptance_criteria": (
                    "Analysis explains tradeoffs, uncertainty, and rationale for conclusions."
                ),
                "deliverables": [f"{slug}-analysis.md"],
            },
            {
                "id": "verify-quality",
                "description": "Validate completeness, consistency, and evidentiary support.",
                "depends_on": ["analyze-findings"],
                "acceptance_criteria": (
                    "Claims are checked against evidence and key gaps/risks are documented."
                ),
                "deliverables": [f"{slug}-verification.md"],
            },
            {
                "id": "deliver-report",
                "description": "Produce final deliverable with recommendations and citations.",
                "depends_on": ["verify-quality"],
                "acceptance_criteria": (
                    "Final output meets goal, includes references, and is ready to share."
                ),
                "deliverables": [f"{slug}-report.md"],
            },
        ]

    @staticmethod
    def _phases_satisfy_intent(phases: list[dict[str, Any]], intent: str) -> bool:
        """Check whether phase set includes intent-critical steps."""
        phase_texts = [
            (
                f"{str(phase.get('id', '')).strip()} "
                f"{str(phase.get('description', '')).strip()}"
            ).lower()
            for phase in phases
            if isinstance(phase, dict)
        ]
        if not phase_texts:
            return False

        def _has_any(markers: tuple[str, ...]) -> bool:
            return any(any(marker in text for marker in markers) for text in phase_texts)

        if intent == "build":
            return _has_any(("implement", "build", "execute", "develop", "code")) and _has_any(
                ("test", "verify", "validate", "qa"),
            )
        if intent == "writing":
            return _has_any(("draft", "write", "compose")) and _has_any(
                ("revise", "edit", "review", "verify"),
            )
        return _has_any(("collect", "evidence", "research", "source")) and _has_any(
            ("analy", "synth", "compare", "evaluate"),
        )

    def _fallback_adhoc_spec(
        self,
        goal: str,
        *,
        available_tools: list[str],
        intent: str | None = None,
    ) -> dict[str, Any]:
        """Return deterministic fallback spec when model synthesis fails."""
        resolved_intent = self._normalize_adhoc_intent(
            str(intent or ""),
            default="research",
        )
        slug = self._sanitize_kebab_token(goal, fallback="adhoc-process", max_len=26)
        available = set(available_tools)
        preferred_by_intent: dict[str, list[str]] = {
            "build": [
                "search_files",
                "read_file",
                "write_file",
                "shell_execute",
                "ripgrep_search",
                "document_write",
            ],
            "writing": [
                "read_file",
                "write_file",
                "document_write",
                "search_files",
                "web_search",
            ],
            "research": [
                "search_files",
                "read_file",
                "write_file",
                "document_write",
                "web_search",
                "web_fetch",
                "spreadsheet",
            ],
        }
        preferred = preferred_by_intent.get(
            resolved_intent,
            preferred_by_intent["research"],
        )
        required_tools = [name for name in preferred if name in available][:5]
        if not required_tools and available_tools:
            required_tools = available_tools[: min(5, len(available_tools))]
        recommended_by_intent: dict[str, list[str]] = {
            "build": ["shell_execute", "ripgrep_search", "web_search"],
            "writing": ["web_search", "document_write"],
            "research": ["web_search", "web_fetch", "spreadsheet", "calculator"],
        }
        recommended = [
            name
            for name in recommended_by_intent.get(resolved_intent, [])
            if name not in available
        ]
        return {
            "source": "fallback_template",
            "intent": resolved_intent,
            "name": f"{slug}-adhoc",
            "description": f"Ad hoc process synthesized for goal: {goal.strip()}",
            "persona": (
                "You are a pragmatic analyst. Build a concrete plan, produce useful "
                "artifacts, and state evidence and assumptions."
            ),
            # Guided keeps the planner in control while still providing structure.
            "phase_mode": "guided",
            "tool_guidance": (
                "Use available tools aggressively for evidence gathering, verification, "
                "and artifact production. Prefer primary sources, maintain traceability, "
                "and keep outputs concise and decision-oriented."
            ),
            "required_tools": required_tools,
            "recommended_tools": recommended,
            "phases": self._adhoc_intent_phase_blueprint(resolved_intent, slug),
        }

    def _normalize_adhoc_spec(
        self,
        raw: dict[str, Any] | None,
        *,
        goal: str,
        key: str,
        available_tools: list[str],
        intent: str | None = None,
    ) -> dict[str, Any]:
        """Normalize model-produced ad hoc process spec into safe structure."""
        resolved_intent = self._resolve_adhoc_intent(raw, intent_hint=intent)
        fallback = self._fallback_adhoc_spec(
            goal,
            available_tools=available_tools,
            intent=resolved_intent,
        )
        if not isinstance(raw, dict):
            return fallback

        raw_synthesis = self._sanitize_synthesis_trace(
            raw.get("_synthesis") if isinstance(raw.get("_synthesis"), dict) else None
        )
        available_set = set(available_tools)
        proposed_name = (
            str(raw.get("name") or raw.get("name_hint") or "")
            .strip()
            .lower()
        )
        name = self._sanitize_kebab_token(
            proposed_name,
            fallback=f"adhoc-{key[:8]}",
            max_len=40,
        )
        if not name.endswith("-adhoc"):
            name = f"{name}-adhoc"

        description = str(raw.get("description", "")).strip() or fallback["description"]
        persona = str(raw.get("persona", "")).strip() or fallback["persona"]
        tool_guidance = str(raw.get("tool_guidance", "")).strip() or fallback["tool_guidance"]
        valid_phase_modes = {"strict", "guided", "suggestive"}
        raw_phase_mode = str(raw.get("phase_mode", "")).strip().lower()
        fallback_phase_mode = str(fallback.get("phase_mode", "guided")).strip().lower()
        if fallback_phase_mode not in valid_phase_modes:
            fallback_phase_mode = "guided"
        phase_mode = raw_phase_mode if raw_phase_mode in valid_phase_modes else fallback_phase_mode

        required_tools: list[str] = []
        for item in raw.get("required_tools", []):
            tool_name = str(item or "").strip()
            if not tool_name or tool_name not in available_set:
                continue
            if tool_name not in required_tools:
                required_tools.append(tool_name)
        if not required_tools:
            required_tools = list(fallback["required_tools"])

        recommended_tools: list[str] = []
        for item in raw.get("recommended_tools", []):
            tool_name = str(item or "").strip()
            if not tool_name or tool_name in available_set:
                continue
            if tool_name not in recommended_tools:
                recommended_tools.append(tool_name)
        for item in fallback["recommended_tools"]:
            if item not in recommended_tools and item not in available_set:
                recommended_tools.append(item)

        seen_phase_ids: set[str] = set()
        phases: list[dict[str, Any]] = []
        raw_phases = raw.get("phases", [])
        if not isinstance(raw_phases, list):
            raw_phases = []
        for idx, phase in enumerate(raw_phases, start=1):
            if not isinstance(phase, dict):
                continue
            phase_id = self._sanitize_kebab_token(
                str(phase.get("id", "")),
                fallback=f"phase-{idx}",
                max_len=36,
            )
            if phase_id in seen_phase_ids:
                phase_id = f"{phase_id}-{idx}"
            seen_phase_ids.add(phase_id)
            desc = str(phase.get("description", "")).strip()
            if not desc:
                desc = f"Execute {phase_id.replace('-', ' ')}."

            deliverables: list[str] = []
            raw_deliverables = phase.get("deliverables", [])
            if isinstance(raw_deliverables, str):
                raw_deliverables = [raw_deliverables]
            if isinstance(raw_deliverables, list):
                for didx, item in enumerate(raw_deliverables, start=1):
                    clean = self._sanitize_deliverable_name(
                        str(item or ""),
                        fallback=f"{phase_id}-{didx}.md",
                    )
                    if clean not in deliverables:
                        deliverables.append(clean)
            if not deliverables:
                deliverables = [f"{phase_id}.md"]

            depends_on: list[str] = []
            raw_depends = phase.get("depends_on", [])
            if isinstance(raw_depends, str):
                raw_depends = [raw_depends]
            if isinstance(raw_depends, list):
                for dep in raw_depends:
                    dep_id = self._sanitize_kebab_token(
                        str(dep or ""),
                        fallback="",
                        max_len=36,
                    )
                    if not dep_id or dep_id == phase_id:
                        continue
                    if dep_id in seen_phase_ids and dep_id not in depends_on:
                        depends_on.append(dep_id)

            phases.append({
                "id": phase_id,
                "description": desc,
                "depends_on": depends_on,
                "deliverables": deliverables,
                "acceptance_criteria": str(
                    phase.get("acceptance_criteria", ""),
                ).strip(),
            })

        used_template_phases = False
        enforce_intent_shape = resolved_intent in {"build", "writing"}
        if (
            not phases
            or len(phases) < 3
            or (
                enforce_intent_shape
                and not self._phases_satisfy_intent(phases, resolved_intent)
            )
        ):
            phases = list(fallback["phases"])
            used_template_phases = True

        if recommended_tools:
            recommended = ", ".join(recommended_tools)
            tool_guidance = (
                f"{tool_guidance}\n\nRecommended additional tools for better outcomes: "
                f"{recommended}"
            )

        source = "fallback_template" if used_template_phases else "model_generated"
        raw_source = str(raw.get("source", "")).strip().lower()
        if raw_source in {"fallback_template", "model_generated"}:
            source = raw_source
        elif not used_template_phases:
            # Preserve template provenance when a cache entry omits source but
            # still exactly matches the fallback phase blueprint.
            if self._is_template_like_adhoc_spec(
                {"intent": resolved_intent, "phases": phases},
                goal=goal,
            ):
                source = "fallback_template"

        return {
            "source": source,
            "intent": resolved_intent,
            "name": name,
            "description": description,
            "persona": persona,
            "phase_mode": phase_mode,
            "tool_guidance": tool_guidance,
            "required_tools": required_tools,
            "recommended_tools": recommended_tools,
            "phases": phases,
            "_synthesis": raw_synthesis,
        }

    def _build_adhoc_cache_entry(
        self,
        *,
        key: str,
        goal: str,
        spec: dict[str, Any],
    ) -> AdhocProcessCacheEntry:
        """Build a cached ad hoc process entry from normalized spec."""
        from loom.processes.schema import PhaseTemplate, ProcessDefinition, ToolRequirements

        phases = [
            PhaseTemplate(
                id=str(phase.get("id", "")).strip(),
                description=str(phase.get("description", "")).strip(),
                depends_on=[
                    str(dep).strip()
                    for dep in phase.get("depends_on", [])
                    if str(dep).strip()
                ],
                acceptance_criteria=str(
                    phase.get("acceptance_criteria", ""),
                ).strip(),
                deliverables=[
                    str(item).strip()
                    for item in phase.get("deliverables", [])
                    if str(item).strip()
                ],
            )
            for phase in spec.get("phases", [])
            if isinstance(phase, dict)
        ]
        process_defn = ProcessDefinition(
            name=str(spec.get("name", "")).strip() or f"adhoc-{key[:8]}",
            version="adhoc-1",
            description=str(spec.get("description", "")).strip(),
            persona=str(spec.get("persona", "")).strip(),
            tool_guidance=str(spec.get("tool_guidance", "")).strip(),
            tools=ToolRequirements(
                required=[
                    str(item).strip()
                    for item in spec.get("required_tools", [])
                    if str(item).strip()
                ],
            ),
            phase_mode=str(spec.get("phase_mode", "guided")).strip() or "guided",
            phases=phases,
            tags=["adhoc", "generated"],
        )
        try:
            spec_snapshot = json.loads(json.dumps(spec, ensure_ascii=False))
        except Exception:
            spec_snapshot = self._spec_from_process_defn(
                process_defn,
                recommended_tools=[
                    str(item).strip()
                    for item in spec.get("recommended_tools", [])
                    if str(item).strip()
                ],
            )
        return AdhocProcessCacheEntry(
            key=key,
            goal=goal,
            process_defn=process_defn,
            recommended_tools=[
                str(item).strip()
                for item in spec.get("recommended_tools", [])
                if str(item).strip()
            ],
            spec=spec_snapshot if isinstance(spec_snapshot, dict) else {},
        )

    def _is_template_like_adhoc_spec(self, spec: dict[str, Any], *, goal: str) -> bool:
        """Return True when a spec appears to be the fallback template."""
        if not isinstance(spec, dict):
            return False
        source = str(spec.get("source", "")).strip().lower()
        if source == "fallback_template":
            return True
        if source == "model_generated":
            return False

        intent = self._normalize_adhoc_intent(str(spec.get("intent", "")), default="research")
        slug = self._sanitize_kebab_token(goal, fallback="adhoc-process", max_len=26)
        expected_ids = [
            str(item.get("id", "")).strip()
            for item in self._adhoc_intent_phase_blueprint(intent, slug)
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]
        raw_phases = spec.get("phases", [])
        if not isinstance(raw_phases, list) or not raw_phases:
            return False
        observed_ids = [
            str(item.get("id", "")).strip()
            for item in raw_phases
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        ]
        return bool(expected_ids and observed_ids == expected_ids)

    def _adhoc_synthesis_activity_lines(
        self,
        entry: AdhocProcessCacheEntry,
        *,
        from_cache: bool,
        fresh: bool,
    ) -> list[str]:
        """Render concise diagnostics for Activity pane visibility."""
        raw_spec = getattr(entry, "spec", None)
        spec = raw_spec if isinstance(raw_spec, dict) else {}
        source = str(spec.get("source", "")).strip() or "unknown"
        intent = self._normalize_adhoc_intent(str(spec.get("intent", "")), default="research")
        phases = spec.get("phases", [])
        phase_count = len(phases) if isinstance(phases, list) else 0
        required = spec.get("required_tools", [])
        recommended = spec.get("recommended_tools", [])
        required_count = len(required) if isinstance(required, list) else 0
        recommended_count = len(recommended) if isinstance(recommended, list) else 0

        lines = [
            (
                "Ad hoc definition summary: "
                f"source={source}, intent={intent}, phases={phase_count}, "
                f"required_tools={required_count}, recommended_tools={recommended_count}."
            ),
            f"Ad hoc cache decision: {'hit' if from_cache else 'miss'} (fresh={fresh}).",
        ]
        synthesis = self._sanitize_synthesis_trace(
            spec.get("_synthesis") if isinstance(spec.get("_synthesis"), dict) else None
        )
        if synthesis:
            initial_ok = bool(synthesis.get("initial_parse_ok"))
            repair_attempted = bool(synthesis.get("repair_attempted"))
            repair_ok = bool(synthesis.get("repair_parse_ok"))
            minimal_attempted = bool(synthesis.get("minimal_retry_attempted"))
            minimal_ok = bool(synthesis.get("minimal_retry_parse_ok"))
            repair_state = "skipped"
            if repair_attempted:
                repair_state = "ok" if repair_ok else "failed"
            minimal_state = "skipped"
            if minimal_attempted:
                minimal_state = "ok" if minimal_ok else "failed"
            response_chars = int(synthesis.get("initial_response_chars") or 0)
            if repair_attempted:
                response_chars = int(synthesis.get("repair_response_chars") or response_chars)
            if minimal_attempted:
                response_chars = int(synthesis.get("minimal_retry_chars") or response_chars)
            lines.append(
                "Ad hoc parse diagnostics: "
                f"initial={'ok' if initial_ok else 'failed'}, "
                f"repair={repair_state}, minimal={minimal_state}, "
                f"response_chars={response_chars}."
            )
            if bool(synthesis.get("empty_response_retry_attempted")):
                retry_chars = int(synthesis.get("empty_response_retry_chars") or 0)
                lines.append(
                    f"Ad hoc empty-response retry: attempted, response_chars={retry_chars}.",
                )
            fallback_reason = str(synthesis.get("fallback_reason", "")).strip()
            if fallback_reason:
                lines.append(f"Ad hoc fallback reason: {fallback_reason}.")
            initial_error = str(synthesis.get("initial_error", "")).strip()
            if initial_error:
                lines.append(f"Ad hoc initial error: {initial_error}.")
            repair_error = str(synthesis.get("repair_error", "")).strip()
            if repair_error:
                lines.append(f"Ad hoc repair error: {repair_error}.")
            minimal_error = str(synthesis.get("minimal_retry_error", "")).strip()
            if minimal_error:
                lines.append(f"Ad hoc minimal retry error: {minimal_error}.")
            artifact_dir = str(synthesis.get("artifact_dir", "")).strip()
            if artifact_dir:
                lines.append(f"Ad hoc synthesis artifacts: {artifact_dir}")
            log_path = str(synthesis.get("log_path", "")).strip()
            if log_path:
                lines.append(f"Ad hoc synthesis log: {log_path}")
        return lines

    @staticmethod
    def _is_temperature_one_only_error(value: object) -> bool:
        text = str(value or "").lower()
        return "invalid temperature" in text and "only 1 is allowed" in text

    @staticmethod
    def _configured_model_temperature(model: ModelProvider | None) -> float | None:
        """Return the selected model's configured temperature from loom.toml."""
        if model is None:
            return None
        value = getattr(model, "configured_temperature", None)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _configured_model_max_tokens(model: ModelProvider | None) -> int | None:
        """Return the selected model's configured max_tokens from loom.toml."""
        if model is None:
            return None
        value = getattr(model, "configured_max_tokens", None)
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _planning_response_max_tokens_limit(self) -> int | None:
        """Return planner-only max_tokens override from limits config."""
        value = getattr(
            getattr(self._config, "limits", None),
            "planning_response_max_tokens",
            None,
        )
        if isinstance(value, int) and value > 0:
            return value
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _has_configured_role_model(self, role: str) -> bool:
        """Return True when config declares at least one model for the role.

        If no model config is loaded (tests/ephemeral contexts), fall back to
        whether an active session model exists.
        """
        models = getattr(getattr(self, "_config", None), "models", None)
        if isinstance(models, dict) and models:
            return any(
                role in list(getattr(model_cfg, "roles", []) or [])
                for model_cfg in models.values()
            )
        return self._model is not None

    def _select_helper_model_for_role(
        self,
        *,
        role: str,
        tier: int,
    ) -> tuple[ModelProvider | None, object | None]:
        """Select a helper model for a role and return (model, router_or_none).

        When full model config is available, selection is role-routed through
        ModelRouter. Without configured models (common in unit tests), this
        falls back to the active cowork model.
        """
        models = getattr(getattr(self, "_config", None), "models", None)
        if isinstance(models, dict) and models:
            from loom.models.router import ModelRouter

            try:
                router = ModelRouter.from_config(self._config)
            except Exception as e:
                logger.debug("Failed to build model router for role %s: %s", role, e)
                return None, None
            try:
                model = router.select(tier=tier, role=role)
            except Exception as e:
                logger.debug("No helper model configured for role %s: %s", role, e)
                return None, router
            return model, router
        return self._model, None

    async def _invoke_helper_role_completion(
        self,
        *,
        role: str,
        tier: int,
        prompt: str,
        max_tokens: int | None,
        temperature: float | None = None,
    ) -> tuple[object, str, float | None, int | None]:
        """Invoke a role-routed helper completion and close temporary routers."""
        model, router = self._select_helper_model_for_role(role=role, tier=tier)
        try:
            if model is None:
                raise RuntimeError(f"No model configured for role: {role}")
            resolved_temperature = (
                temperature
                if temperature is not None
                else self._configured_model_temperature(model)
            )
            planner_max_tokens = (
                self._planning_response_max_tokens_limit()
                if role == "planner"
                else None
            )
            resolved_max_tokens = (
                max_tokens
                if isinstance(max_tokens, int) and max_tokens > 0
                else planner_max_tokens
                if isinstance(planner_max_tokens, int) and planner_max_tokens > 0
                else self._configured_model_max_tokens(model)
            )
            response = await call_with_model_retry(
                lambda: model.complete(
                    [{"role": "user", "content": prompt}],
                    temperature=resolved_temperature,
                    max_tokens=resolved_max_tokens,
                ),
                policy=self._model_retry_policy(),
            )
            return (
                response,
                str(getattr(model, "name", "") or ""),
                resolved_temperature,
                resolved_max_tokens,
            )
        finally:
            if router is not None:
                close = getattr(router, "close", None)
                if close is not None:
                    try:
                        await close()
                    except Exception as e:
                        logger.debug("Failed closing helper role router: %s", e)

    def _should_resynthesize_cached_adhoc(self, entry: AdhocProcessCacheEntry) -> bool:
        """Return True when cache should be refreshed from model synthesis."""
        if not self._has_configured_role_model("planner"):
            return False
        spec = entry.spec if isinstance(entry.spec, dict) else {}
        return self._is_template_like_adhoc_spec(spec, goal=entry.goal)

    async def _synthesize_adhoc_process(self, goal: str, *, key: str) -> AdhocProcessCacheEntry:
        """Generate an ad hoc process definition from a free-form run goal."""
        available_tools = self._available_tool_names()
        fallback_spec = self._fallback_adhoc_spec(
            goal,
            available_tools=available_tools,
            intent="research",
        )
        diagnostics: dict[str, Any] = {
            "cache_key": key,
            "model_name": "",
            "available_tool_count": len(available_tools),
            "goal_chars": len(str(goal or "")),
            "initial_max_tokens": None,
            "repair_max_tokens": None,
            "minimal_max_tokens": None,
            "prompt_chars": 0,
            "initial_response_chars": 0,
            "initial_output_tokens": 0,
            "initial_parse_ok": False,
            "repair_attempted": False,
            "repair_response_chars": 0,
            "repair_output_tokens": 0,
            "repair_parse_ok": False,
            "empty_response_retry_attempted": False,
            "empty_response_retry_chars": 0,
            "minimal_retry_attempted": False,
            "minimal_retry_chars": 0,
            "minimal_output_tokens": 0,
            "minimal_retry_parse_ok": False,
            "parsed_raw_incomplete": False,
            "minimal_retry_kept_prior_raw": False,
            "fallback_reason": "",
            "initial_error": "",
            "repair_error": "",
            "minimal_retry_error": "",
            "initial_preview": "",
            "repair_preview": "",
            "minimal_preview": "",
            "repair_source_chars": 0,
            "repair_source_truncated": False,
            "initial_temperature": None,
            "repair_temperature": None,
            "resolved_source": "fallback_template",
            "resolved_intent": "research",
            "phase_count": len(fallback_spec.get("phases", [])),
            "required_tool_count": len(fallback_spec.get("required_tools", [])),
            "recommended_tool_count": len(fallback_spec.get("recommended_tools", [])),
            "log_path": str(self._adhoc_synthesis_log_path()),
            "artifact_dir": "",
        }
        artifact_dir = self._create_adhoc_synthesis_artifact_dir(key=key, goal=goal)
        if artifact_dir is not None:
            diagnostics["artifact_dir"] = str(artifact_dir)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "00-goal.txt",
                str(goal or "").strip(),
            )
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "00-available-tools.txt",
                "\n".join(available_tools),
            )
        if not self._has_configured_role_model("planner"):
            diagnostics["fallback_reason"] = "model_unavailable"
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "11-diagnostics.json",
                json.dumps(
                    self._sanitize_synthesis_trace(diagnostics),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "12-rejection-reason.txt",
                "fallback_reason=model_unavailable\nresolved_source=fallback_template\n",
            )
            fallback_spec["_synthesis"] = self._sanitize_synthesis_trace(diagnostics)
            self._append_adhoc_synthesis_log({
                "event": "adhoc_synthesis",
                "goal": goal,
                **self._sanitize_synthesis_trace(diagnostics),
            })
            logger.warning(
                "Ad hoc synthesis[%s] used fallback: model unavailable",
                key,
            )
            return self._build_adhoc_cache_entry(key=key, goal=goal, spec=fallback_spec)

        tool_list = ", ".join(available_tools) if available_tools else "(none)"
        prompt = (
            "The user wants to do this:\n"
            f"{goal}\n\n"
            "Let's design an abstract Loom process for this request using "
            "docs/creating-packages.md as the contract reference.\n"
            "The process must be reusable, well-structured, and outcome-driven.\n\n"
            "Return ONLY valid JSON with keys:\n"
            "intent, name, description, persona, phase_mode, tool_guidance, required_tools, "
            "recommended_tools, phases.\n"
            "phases must be a list of objects with keys:\n"
            "id, description, depends_on, acceptance_criteria, deliverables.\n\n"
            "Hard requirements:\n"
            "- Determine intent from the user goal and set `intent` to exactly one of: "
            "research, writing, build.\n"
            "- Use lowercase kebab-case for name and phase ids.\n"
            "- phase_mode MUST be one of: strict, guided, suggestive.\n"
            "- Prefer phase_mode=\"guided\" unless strict ordering is explicitly needed.\n"
            "- Use as many phases as needed for the goal (typically 3-20), not a fixed count.\n"
            "- Keep each phase small enough to finish within one subtask wall-clock budget.\n"
            "- Use `depends_on` to model independence so parallelizable phases "
            "can run concurrently.\n"
            "- If intent=build, include an implementation/build phase and a test/verify phase.\n"
            "- If intent=writing, include draft/write and revision/edit phases.\n"
            "- If intent=research, include source/evidence collection and analysis/synthesis "
            "phases.\n"
            "- Every phase must include measurable acceptance_criteria.\n"
            "- Every phase must include concrete deliverable filenames (.md/.csv/etc).\n"
            "- Deliverables should default to root-level filenames (e.g., report.md), "
            "not numbered phase folders, unless the goal explicitly requires "
            "subdirectories.\n"
            "- Do not add phases whose main purpose is creating folder schemas, "
            "workspace schema docs, or numbered directories unless the user "
            "explicitly asks for that structure.\n"
            "- If the goal mentions local files or directories, include a phase that "
            "inspects them.\n"
            "- required_tools must be selected ONLY from available tools.\n"
            "- recommended_tools should list useful missing tools not currently available.\n"
            "- Keep the process reusable and focused on outcomes, not implementation trivia.\n"
            "- Do not wrap JSON in markdown fences.\n\n"
            "Full package authoring reference:\n"
            "<<<BEGIN_CREATING_PACKAGES_MD>>>\n"
            + self._adhoc_package_contract_hint()
            + "\n<<<END_CREATING_PACKAGES_MD>>>\n\n"
            f"Available tools: {tool_list}\n"
        )
        diagnostics["prompt_chars"] = len(prompt)
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "01-initial-prompt.txt",
            prompt,
        )
        configured_temperature: float | None = None

        expected_json_keys = (
            "intent",
            "name",
            "description",
            "persona",
            "phase_mode",
            "tool_guidance",
            "required_tools",
            "recommended_tools",
            "phases",
        )
        raw: dict[str, Any] | None = None
        raw_text = ""
        try:
            response, model_name, configured_temperature, initial_max_tokens = (
                await self._invoke_helper_role_completion(
                    role="planner",
                    tier=2,
                    prompt=prompt,
                    max_tokens=None,
                )
            )
            diagnostics["model_name"] = model_name
            diagnostics["initial_temperature"] = configured_temperature
            diagnostics["repair_temperature"] = configured_temperature
            diagnostics["initial_max_tokens"] = initial_max_tokens
            raw_text = str(getattr(response, "text", "") or "")
            diagnostics["initial_response_chars"] = len(raw_text)
            diagnostics["initial_preview"] = self._synthesis_preview(raw_text)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "02-initial-response.txt",
                raw_text,
            )
            raw = self._extract_json_payload(
                raw_text,
                expected_keys=expected_json_keys,
            )
            diagnostics["initial_parse_ok"] = isinstance(raw, dict)
        except Exception as e:
            diagnostics["initial_error"] = str(e)
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "02-initial-error.txt",
                str(e),
            )
            logger.warning("Ad hoc process synthesis failed: %s", e)

        if raw is None and not raw_text.strip() and not str(
            diagnostics.get("initial_error", ""),
        ).strip():
            diagnostics["repair_attempted"] = True
            diagnostics["empty_response_retry_attempted"] = True
            retry_prompt = (
                "Your previous response was empty.\n"
                "Return a non-empty strict JSON object only.\n"
                "Required keys: intent, name, description, persona, phase_mode, "
                "tool_guidance, required_tools, recommended_tools, phases.\n"
                "Each phase object keys: id, description, depends_on, "
                "acceptance_criteria, deliverables.\n\n"
                "Deliverables should default to root-level filenames (e.g., "
                "report.md) unless the user explicitly requires subdirectories.\n"
                "Do not include folder-scaffolding-only phases unless explicitly requested.\n\n"
                "Use this same task request:\n"
                f"{goal}\n\n"
                "Do not include markdown fences."
            )
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "03-empty-retry-prompt.txt",
                retry_prompt,
            )
            try:
                retry_response, retry_model_name, retry_temperature, retry_max_tokens = (
                    await self._invoke_helper_role_completion(
                        role="planner",
                        tier=2,
                        prompt=retry_prompt,
                        max_tokens=None,
                        temperature=configured_temperature,
                    )
                )
                if not diagnostics["model_name"]:
                    diagnostics["model_name"] = retry_model_name
                if configured_temperature is None:
                    configured_temperature = retry_temperature
                    diagnostics["repair_temperature"] = retry_temperature
                diagnostics["repair_max_tokens"] = retry_max_tokens
                retry_text = str(getattr(retry_response, "text", "") or "")
                diagnostics["empty_response_retry_chars"] = len(retry_text)
                diagnostics["repair_response_chars"] = len(retry_text)
                diagnostics["repair_preview"] = self._synthesis_preview(retry_text)
                self._write_adhoc_synthesis_artifact_text(
                    artifact_dir,
                    "04-empty-retry-response.txt",
                    retry_text,
                )
                if retry_text.strip():
                    raw_text = retry_text
                    raw = self._extract_json_payload(
                        retry_text,
                        expected_keys=expected_json_keys,
                    )
                    diagnostics["repair_parse_ok"] = isinstance(raw, dict)
            except Exception as e:
                diagnostics["repair_error"] = str(e)
                self._write_adhoc_synthesis_artifact_text(
                    artifact_dir,
                    "04-empty-retry-error.txt",
                    str(e),
                )
                logger.warning("Ad hoc process empty-response retry failed: %s", e)

        if raw is None and raw_text.strip():
            diagnostics["repair_attempted"] = True
            repair_source_cap = int(
                getattr(
                    getattr(self._config, "limits", None),
                    "adhoc_repair_source_max_chars",
                    0,
                )
                or 0,
            )
            repair_source_text = raw_text
            if repair_source_cap > 0 and len(repair_source_text) > repair_source_cap:
                repair_source_text = repair_source_text[:repair_source_cap]
            diagnostics["repair_source_chars"] = len(repair_source_text)
            diagnostics["repair_source_truncated"] = len(repair_source_text) < len(raw_text)
            repair_prompt = (
                "You will receive model output that should describe a Loom process.\n"
                "Convert it into STRICT JSON only.\n"
                "Return exactly one JSON object with keys:\n"
                "intent, name, description, persona, phase_mode, tool_guidance, "
                "required_tools, recommended_tools, phases.\n"
                "Each phase object must contain: id, description, depends_on, "
                "acceptance_criteria, deliverables.\n"
                "Prefer root-level deliverable filenames unless the user explicitly "
                "requires subdirectories.\n"
                "Do not include folder-scaffolding-only phases unless explicitly requested.\n"
                "Do not include markdown fences.\n\n"
                "SOURCE OUTPUT:\n"
                "<<<BEGIN_SOURCE>>>\n"
                f"{repair_source_text}\n"
                "<<<END_SOURCE>>>"
            )
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "05-repair-prompt.txt",
                repair_prompt,
            )
            try:
                repaired, repaired_model_name, repaired_temperature, repair_max_tokens = (
                    await self._invoke_helper_role_completion(
                        role="planner",
                        tier=2,
                        prompt=repair_prompt,
                        max_tokens=None,
                        temperature=configured_temperature,
                    )
                )
                if not diagnostics["model_name"]:
                    diagnostics["model_name"] = repaired_model_name
                if configured_temperature is None:
                    configured_temperature = repaired_temperature
                    diagnostics["repair_temperature"] = repaired_temperature
                diagnostics["repair_max_tokens"] = repair_max_tokens
                repaired_text = str(getattr(repaired, "text", "") or "")
                diagnostics["repair_response_chars"] = len(repaired_text)
                diagnostics["repair_preview"] = self._synthesis_preview(repaired_text)
                self._write_adhoc_synthesis_artifact_text(
                    artifact_dir,
                    "06-repair-response.txt",
                    repaired_text,
                )
                raw = self._extract_json_payload(
                    repaired_text,
                    expected_keys=expected_json_keys,
                )
                diagnostics["repair_parse_ok"] = isinstance(raw, dict)
            except Exception as e:
                diagnostics["repair_error"] = str(e)
                self._write_adhoc_synthesis_artifact_text(
                    artifact_dir,
                    "06-repair-error.txt",
                    str(e),
                )
                logger.warning("Ad hoc process JSON repair failed: %s", e)

        parsed_raw = raw if isinstance(raw, dict) else None
        needs_minimal_retry = (
            raw is None or self._raw_adhoc_spec_needs_minimal_retry(parsed_raw)
        )
        diagnostics["parsed_raw_incomplete"] = bool(
            isinstance(parsed_raw, dict)
            and self._raw_adhoc_spec_needs_minimal_retry(parsed_raw),
        )
        if needs_minimal_retry:
            diagnostics["minimal_retry_attempted"] = True
            minimal_prompt = (
                "Return exactly one STRICT JSON object and nothing else.\n"
                "Required top-level keys:\n"
                "intent, name, description, persona, phase_mode, tool_guidance, "
                "required_tools, recommended_tools, phases.\n"
                "Constraints:\n"
                "- intent: research | writing | build\n"
                "- phase_mode: strict | guided | suggestive (prefer guided)\n"
                "- phases length: 3-20\n"
                "- each phase object keys: id, description, depends_on, "
                "acceptance_criteria, deliverables\n"
                "- required_tools must be from available tools only\n"
                "- description, persona, tool_guidance must be concise (<= 180 chars each)\n"
                "- phase descriptions must be concise (<= 120 chars)\n"
                "- acceptance_criteria must be a short STRING (not array)\n"
                "- deliverables must be filename strings like report.md\n"
                "- prefer root-level deliverables; avoid numbered phase folders unless "
                "explicitly required by the goal\n"
                "- do not include folder-scaffolding-only phases unless explicitly requested\n"
                "- do not include markdown/code fences/prose\n\n"
                f"Goal:\n{goal}\n\n"
                f"Available tools: {tool_list}\n"
            )
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "07-minimal-retry-prompt.txt",
                minimal_prompt,
            )
            try:
                minimal, minimal_model_name, minimal_temperature, minimal_max_tokens = (
                    await self._invoke_helper_role_completion(
                        role="planner",
                        tier=2,
                        prompt=minimal_prompt,
                        max_tokens=None,
                        temperature=configured_temperature,
                    )
                )
                if not diagnostics["model_name"]:
                    diagnostics["model_name"] = minimal_model_name
                if configured_temperature is None:
                    configured_temperature = minimal_temperature
                    diagnostics["repair_temperature"] = minimal_temperature
                diagnostics["minimal_max_tokens"] = minimal_max_tokens
                minimal_text = str(getattr(minimal, "text", "") or "")
                diagnostics["minimal_retry_chars"] = len(minimal_text)
                diagnostics["minimal_preview"] = self._synthesis_preview(minimal_text)
                self._write_adhoc_synthesis_artifact_text(
                    artifact_dir,
                    "08-minimal-retry-response.txt",
                    minimal_text,
                )
                minimal_parsed = self._extract_json_payload(
                    minimal_text,
                    expected_keys=expected_json_keys,
                )
                diagnostics["minimal_retry_parse_ok"] = isinstance(minimal_parsed, dict)
                if isinstance(minimal_parsed, dict):
                    raw = minimal_parsed
                elif isinstance(parsed_raw, dict):
                    # Preserve previously parsed content when minimal retry fails.
                    diagnostics["minimal_retry_kept_prior_raw"] = True
                    raw = parsed_raw
            except Exception as e:
                diagnostics["minimal_retry_error"] = str(e)
                self._write_adhoc_synthesis_artifact_text(
                    artifact_dir,
                    "08-minimal-retry-error.txt",
                    str(e),
                )
                if isinstance(parsed_raw, dict):
                    diagnostics["minimal_retry_kept_prior_raw"] = True
                    raw = parsed_raw
                logger.warning("Ad hoc process minimal retry failed: %s", e)

        if raw is None:
            if self._is_temperature_one_only_error(diagnostics.get("initial_error", "")):
                diagnostics["fallback_reason"] = "temperature_config_mismatch"
            elif self._is_temperature_one_only_error(diagnostics.get("repair_error", "")):
                diagnostics["fallback_reason"] = "temperature_config_mismatch"
            elif not raw_text.strip():
                diagnostics["fallback_reason"] = "empty_model_response"
            elif str(diagnostics.get("initial_error", "")).strip():
                diagnostics["fallback_reason"] = "model_completion_error"
            elif (
                diagnostics.get("repair_attempted")
                and not diagnostics.get("repair_parse_ok")
                and diagnostics.get("minimal_retry_attempted")
                and not diagnostics.get("minimal_retry_parse_ok")
            ):
                diagnostics["fallback_reason"] = "schema_parse_failed"
            else:
                diagnostics["fallback_reason"] = "non_parseable_response"
            preview = re.sub(r"\s+", " ", raw_text).strip()
            if len(preview) > 280:
                preview = preview[:280].rstrip() + "..."
            logger.warning(
                "Ad hoc process synthesis returned non-parseable payload; using fallback."
                " preview=%r",
                preview,
            )

        normalized = self._normalize_adhoc_spec(
            raw,
            goal=goal,
            key=key,
            available_tools=available_tools,
        )
        source = str(normalized.get("source", "")).strip() or "unknown"
        diagnostics["resolved_source"] = source
        diagnostics["resolved_intent"] = str(normalized.get("intent", "")).strip() or "research"
        diagnostics["phase_count"] = len(normalized.get("phases", []) or [])
        diagnostics["required_tool_count"] = len(normalized.get("required_tools", []) or [])
        diagnostics["recommended_tool_count"] = len(normalized.get("recommended_tools", []) or [])
        if (
            source == "fallback_template"
            and not str(diagnostics.get("fallback_reason", "")).strip()
        ):
            diagnostics["fallback_reason"] = "normalization_template_substitution"

        self._write_adhoc_synthesis_artifact_yaml(
            artifact_dir,
            "09-parsed-raw.yaml",
            raw if isinstance(raw, dict) else None,
        )
        self._write_adhoc_synthesis_artifact_yaml(
            artifact_dir,
            "10-normalized-spec.yaml",
            normalized,
        )
        self._write_adhoc_synthesis_artifact_text(
            artifact_dir,
            "11-diagnostics.json",
            json.dumps(
                self._sanitize_synthesis_trace(diagnostics),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        if str(diagnostics.get("fallback_reason", "")).strip():
            self._write_adhoc_synthesis_artifact_text(
                artifact_dir,
                "12-rejection-reason.txt",
                (
                    "fallback_reason="
                    f"{str(diagnostics.get('fallback_reason', '')).strip()}\n"
                    "resolved_source="
                    f"{str(diagnostics.get('resolved_source', '')).strip()}\n"
                    "parsed_raw_incomplete="
                    f"{bool(diagnostics.get('parsed_raw_incomplete'))}\n"
                ),
            )

        normalized["_synthesis"] = self._sanitize_synthesis_trace(diagnostics)
        self._append_adhoc_synthesis_log({
            "event": "adhoc_synthesis",
            "goal": goal,
            **self._sanitize_synthesis_trace(diagnostics),
        })
        logger.warning(
            "Ad hoc synthesis[%s]: source=%s intent=%s phases=%s parse(initial=%s repair=%s)",
            key,
            diagnostics["resolved_source"],
            diagnostics["resolved_intent"],
            diagnostics["phase_count"],
            diagnostics["initial_parse_ok"],
            diagnostics["repair_parse_ok"] if diagnostics["repair_attempted"] else "skipped",
        )
        return self._build_adhoc_cache_entry(key=key, goal=goal, spec=normalized)

    async def _get_or_create_adhoc_process(
        self,
        goal: str,
        *,
        fresh: bool = False,
    ) -> tuple[AdhocProcessCacheEntry, bool]:
        """Fetch cached ad hoc process for goal, or synthesize and cache one."""
        key = self._adhoc_cache_key(goal)
        if fresh:
            # Explicit --fresh requests should always bypass memory + disk cache.
            self._adhoc_process_cache.pop(key, None)
            generated = await self._synthesize_adhoc_process(goal, key=key)
            self._adhoc_process_cache[key] = generated
            try:
                self._persist_adhoc_cache_entry(generated)
            except Exception as e:
                logger.warning("Failed to persist ad hoc process cache for %s: %s", key, e)
            self._append_adhoc_synthesis_log({
                "event": "adhoc_cache_decision",
                "cache_key": key,
                "goal": goal,
                "decision": "fresh_synthesis",
            })
            return generated, False

        cached = self._adhoc_process_cache.get(key)
        if cached is not None and not self._should_resynthesize_cached_adhoc(cached):
            self._append_adhoc_synthesis_log({
                "event": "adhoc_cache_decision",
                "cache_key": key,
                "goal": goal,
                "decision": "memory_hit",
            })
            return cached, True
        if cached is not None:
            self._adhoc_process_cache.pop(key, None)
        disk_cached = self._load_adhoc_cache_entry_from_disk(key)
        if disk_cached is not None and not self._should_resynthesize_cached_adhoc(disk_cached):
            self._adhoc_process_cache[key] = disk_cached
            self._append_adhoc_synthesis_log({
                "event": "adhoc_cache_decision",
                "cache_key": key,
                "goal": goal,
                "decision": "disk_hit",
            })
            return disk_cached, True
        generated = await self._synthesize_adhoc_process(goal, key=key)
        self._adhoc_process_cache[key] = generated
        try:
            self._persist_adhoc_cache_entry(generated)
        except Exception as e:
            logger.warning("Failed to persist ad hoc process cache for %s: %s", key, e)
        self._append_adhoc_synthesis_log({
            "event": "adhoc_cache_decision",
            "cache_key": key,
            "goal": goal,
            "decision": "synthesized",
        })
        return generated, False

    @staticmethod
    def _serialize_process_for_package(process_defn: ProcessDefinition) -> dict[str, Any]:
        """Convert a process definition into a process.yaml payload."""
        payload: dict[str, Any] = {
            "name": process_defn.name,
            "schema_version": int(process_defn.schema_version or 1),
            "version": "0.1.0",
            "description": process_defn.description,
            "persona": process_defn.persona,
            "tool_guidance": process_defn.tool_guidance,
            "phase_mode": process_defn.phase_mode or "guided",
            "tools": {
                "guidance": process_defn.tools.guidance,
                "required": list(process_defn.tools.required),
                "excluded": list(process_defn.tools.excluded),
            },
            "phases": [
                {
                    "id": phase.id,
                    "description": phase.description,
                    "depends_on": list(phase.depends_on),
                    "acceptance_criteria": phase.acceptance_criteria,
                    "deliverables": list(phase.deliverables),
                }
                for phase in process_defn.phases
            ],
            "tags": list(dict.fromkeys([*process_defn.tags, "adhoc", "generated"])),
            "author": process_defn.author or "loom-adhoc",
        }
        auth_required = process_defn.auth_required_resources()
        if auth_required:
            payload["auth"] = {"required": auth_required}
        return payload

    def _save_adhoc_process_package(
        self,
        *,
        process_defn: ProcessDefinition,
        package_name: str,
        recommended_tools: list[str],
    ) -> Path:
        """Persist an ad hoc process as a workspace-local package."""
        import yaml

        safe_name = self._sanitize_kebab_token(
            package_name,
            fallback="adhoc-process",
            max_len=40,
        )
        package_dir = self._workspace / ".loom" / "processes" / safe_name
        if package_dir.exists():
            raise ValueError(f"Package already exists: {package_dir}")
        package_dir.mkdir(parents=True, exist_ok=False)

        spec = self._serialize_process_for_package(process_defn)
        spec["name"] = safe_name
        process_yaml = package_dir / "process.yaml"
        process_yaml.write_text(
            yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        notes = [
            f"# {safe_name}",
            "",
            "Generated from an ad hoc `/run` synthesis in Loom TUI.",
        ]
        if recommended_tools:
            notes.append("")
            notes.append("## Recommended Additional Tools")
            for tool_name in recommended_tools:
                notes.append(f"- {tool_name}")
        (package_dir / "README.md").write_text(
            "\n".join(notes).rstrip() + "\n",
            encoding="utf-8",
        )
        return package_dir

    async def _reload_session_for_process_change(self, chat: ChatLog) -> None:
        """Rebuild session after changing active process."""
        if self._chat_busy or self._has_active_process_runs():
            chat.add_info(
                "[bold #f7768e]Cannot switch process while work is running.[/]"
            )
            return

        if self._session is not None:
            if self._store and self._session.session_id:
                await self._store.update_session(
                    self._session.session_id, is_active=False,
                )
            self._session = None

        self._resume_session = None
        await self._initialize_session()

    def _has_active_process_runs(self) -> bool:
        """Return True when at least one run is queued/running."""
        return any(
            run.status in {"queued", "running"}
            for run in self._process_runs.values()
        )

    @staticmethod
    def _reserved_slash_command_names() -> set[str]:
        """Return reserved slash command names (without leading slash)."""
        reserved: set[str] = set()
        for spec in _SLASH_COMMANDS:
            reserved.add(spec.canonical.lstrip("/").lower())
            for alias in spec.aliases:
                reserved.add(alias.lstrip("/").lower())
        return reserved

    def _is_reserved_process_name(self, name: str) -> bool:
        """Return True when process name collides with built-in slash command."""
        return name.strip().lower() in self._reserved_slash_command_names()

    def _refresh_process_command_index(
        self,
        *,
        chat: ChatLog | None = None,
        notify_conflicts: bool = False,
    ) -> None:
        """Refresh process catalog and dynamic slash-command map."""
        try:
            loader = self._create_process_loader()
            available = loader.list_available()
        except Exception:
            self._cached_process_catalog = []
            self._process_command_map = {}
            self._blocked_process_commands = []
            return

        selectable: list[dict[str, str]] = []
        command_map: dict[str, str] = {}
        blocked: set[str] = set()
        reserved = self._reserved_slash_command_names()

        for proc in available:
            name = str(proc.get("name", "")).strip()
            if not name:
                continue
            lowered = name.lower()
            if lowered in reserved:
                blocked.add(name)
                continue
            selectable.append(proc)
            command_map[f"/{lowered}"] = name

        self._cached_process_catalog = selectable
        self._process_command_map = command_map
        self._blocked_process_commands = sorted(blocked, key=str.lower)

        if notify_conflicts and chat and self._blocked_process_commands:
            blocked_cmds = ", ".join(f"/{name}" for name in self._blocked_process_commands)
            chat.add_info(
                "[bold #f7768e]Process command name collision:[/] "
                f"{blocked_cmds}\n"
                "[dim]These process names collide with built-in slash commands "
                "and were skipped in TUI.[/dim]"
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
        """Render `/process` usage with clear hierarchy."""
        active = self._escape_markup(self._active_process_name())
        return "\n".join([
            "[bold #7dcfff]Process Controls[/bold #7dcfff]",
            f"  [bold]Active:[/] {active}",
            "  [bold]Commands:[/]",
            "    /processes",
            "    /process list",
            "    /process use <name-or-path>",
            "    /process off",
            "    /<process-name> <goal>",
            "    /run <goal>",
        ])

    def _render_tools_catalog(self) -> str:
        """Render `/tools` output as a wrapped catalog."""
        tools = self._tools.list_tools()
        lines = [f"[bold #7dcfff]Tools[/bold #7dcfff] [dim]({len(tools)})[/dim]"]
        if not tools:
            lines.append("  [dim](none)[/dim]")
            return "\n".join(lines)
        joined = ", ".join(self._escape_markup(tool) for tool in tools)
        lines.append(
            self._wrap_info_text(
                joined,
                initial_indent="  ",
                subsequent_indent="  ",
            )
        )
        return "\n".join(lines)

    def _render_session_info(self, state) -> str:
        """Render `/session` output with compact sections."""
        session_id = self._escape_markup(self._session.session_id if self._session else "?")
        workspace = self._escape_markup(self._session.workspace if self._session else "?")
        process_name = self._escape_markup(self._active_process_name())
        focus = self._escape_markup(state.current_focus or "(none)")

        lines = [
            "[bold #7dcfff]Current Session[/bold #7dcfff]",
            f"  [bold]ID:[/] [dim]{session_id}[/dim]",
            f"  [bold]Workspace:[/] [dim]{workspace}[/dim]",
            f"  [bold]Process:[/] {process_name}",
            f"  [bold]Turns:[/] {state.turn_count}",
            f"  [bold]Tokens:[/] {state.total_tokens:,}",
            f"  [bold]Focus:[/] {focus}",
        ]
        if state.key_decisions:
            lines.append("  [bold]Recent Decisions:[/]")
            for decision in state.key_decisions[-5:]:
                lines.append(
                    self._wrap_info_text(
                        self._escape_markup(decision),
                        initial_indent="    - ",
                        subsequent_indent="      ",
                    )
                )
        return "\n".join(lines)

    def _render_sessions_list(self, sessions: list[dict]) -> str:
        """Render `/sessions` output with readable per-session rows."""
        lines = [
            "[bold #7dcfff]Recent Sessions[/bold #7dcfff]",
            "[dim]Use /resume <session-id-prefix> to switch[/dim]",
        ]
        for row in sessions[:10]:
            sid = self._escape_markup(str(row.get("id", "")))
            turns = int(row.get("turn_count", 0) or 0)
            started = self._escape_markup(str(row.get("started_at", "?"))[:16])
            ws = self._escape_markup(row.get("workspace_path", "?"))

            badges: list[str] = []
            if row.get("is_active"):
                badges.append("[green]active[/green]")
            if self._session and sid == self._session.session_id:
                badges.append("[cyan]current[/cyan]")
            badge_text = f" [dim]({' | '.join(badges)})[/dim]" if badges else ""

            lines.append(
                f"  [bold]{sid[:12]}...[/bold]  [dim]{started}[/dim]  "
                f"{turns} turns{badge_text}"
            )
            lines.append(
                self._wrap_info_text(
                    ws,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )
        return "\n".join(lines)

    def _render_startup_summary(self, *, tool_count: int, persisted: str) -> str:
        """Render startup summary block with workspace/session details."""
        workspace = self._escape_markup(self._workspace)
        process_name = self._escape_markup(self._active_process_name())
        lines = [
            f"[bold #7dcfff]Loom[/bold #7dcfff]  [dim]({self._model.name})[/dim]",
            f"  [bold]Workspace:[/] [dim]{workspace}[/dim]",
            f"  [bold]Tools:[/] {tool_count}",
            f"  [bold]Session Mode:[/] {persisted}",
            f"  [bold]Process:[/] {process_name}",
        ]
        if self._session and self._session.session_id:
            lines.append(
                f"  [bold]Session ID:[/] [dim]{self._escape_markup(self._session.session_id)}[/dim]"
            )
        return "\n".join(lines)

    def _render_process_catalog(self) -> str:
        """Build a human-readable process list."""
        self._refresh_process_command_index()
        available = self._cached_process_catalog
        if not available:
            if self._blocked_process_commands:
                blocked = ", ".join(
                    self._escape_markup(name) for name in self._blocked_process_commands
                )
                return (
                    "[bold #7dcfff]Available Processes[/bold #7dcfff]\n"
                    "  [dim]No selectable process definitions found.[/dim]\n"
                    f"  [#f7768e]Blocked (name collisions): {blocked}[/]"
                )
            return (
                "[bold #7dcfff]Available Processes[/bold #7dcfff]\n"
                "  [dim]No process definitions found.[/dim]"
            )

        active = self._process_defn.name if self._process_defn else ""
        lines = [
            "[bold #7dcfff]Available Processes[/bold #7dcfff]",
            "[dim]Run directly with /<process-name> <goal> or /run <goal>[/dim]",
        ]
        for proc in available:
            name = str(proc.get("name", "")).strip()
            ver = str(proc.get("version", "")).strip()
            desc = str(proc.get("description", "")).strip().split("\n")[0]
            marker = " [cyan](active)[/cyan]" if name == active else ""
            safe_name = self._escape_markup(name)
            safe_ver = self._escape_markup(ver)
            lines.append(f"  [bold]{safe_name}[/bold] [dim]v{safe_ver}[/dim]{marker}")
            if desc:
                lines.append(
                    self._wrap_info_text(
                        self._escape_markup(desc),
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                )
        if self._blocked_process_commands:
            blocked = ", ".join(
                f"/{self._escape_markup(name)}" for name in self._blocked_process_commands
            )
            lines.append(f"  [#f7768e]Blocked (name collisions): {blocked}[/]")
        return "\n".join(lines)

    @staticmethod
    def _render_mcp_list(views: list) -> str:
        """Build a readable MCP server list."""
        if not views:
            return "No MCP servers configured."
        lines = ["[bold]MCP servers:[/bold]"]
        for view in views:
            status = "enabled" if view.server.enabled else "disabled"
            lines.append(
                f"  {view.alias:16s} {status:8s} [dim]source={view.source}[/dim]"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_mcp_view(view) -> str:
        """Build a readable MCP server details block."""
        from loom.mcp.config import redact_server_env

        env = redact_server_env(view.server)
        lines = [
            f"[bold]{view.alias}[/bold]",
            f"  source: {view.source}",
            f"  source_path: {view.source_path or '-'}",
            f"  enabled: {view.server.enabled}",
            f"  command: {view.server.command}",
            f"  args: {' '.join(view.server.args) if view.server.args else '-'}",
            f"  cwd: {view.server.cwd or '-'}",
            f"  timeout_seconds: {view.server.timeout_seconds}",
            "  env:",
        ]
        if env:
            for key, value in env.items():
                lines.append(f"    {key}={value}")
        else:
            lines.append("    (none)")
        return "\n".join(lines)

    def _apply_process_tool_policy(self, chat: ChatLog) -> None:
        """Apply process tool exclusions to the active registry."""
        if not self._process_defn:
            return
        if self._process_defn.tools.excluded:
            for tool_name in self._process_defn.tools.excluded:
                self._tools.exclude(tool_name)
        if self._process_defn.tools.required:
            missing = [
                tool_name
                for tool_name in self._process_defn.tools.required
                if not self._tools.has(tool_name)
            ]
            if missing:
                joined = ", ".join(sorted(missing))
                chat.add_info(
                    f"[bold #f7768e]Process requires missing tool(s): "
                    f"{joined}[/]"
                )

    def _build_system_prompt(self) -> str:
        """Build cowork system prompt with optional process extensions."""
        system_prompt = build_cowork_system_prompt(self._workspace)
        if self._process_defn:
            if self._process_defn.persona:
                system_prompt += (
                    f"\n\nDOMAIN ROLE:\n{self._process_defn.persona.strip()}"
                )
            if self._process_defn.tool_guidance:
                system_prompt += (
                    f"\n\nDOMAIN TOOL GUIDANCE:\n"
                    f"{self._process_defn.tool_guidance.strip()}"
                )
        return system_prompt

    def _model_retry_policy(self) -> ModelRetryPolicy:
        if self._config is None:
            return ModelRetryPolicy()
        return ModelRetryPolicy.from_execution_config(self._config.execution)

    def _cowork_scratch_dir(self) -> Path | None:
        if self._config is None:
            return None
        try:
            return self._config.scratch_path
        except Exception:
            return None

    def _cowork_enable_filetype_ingest_router(self) -> bool:
        runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
        if runner_limits is None:
            return True
        return bool(getattr(runner_limits, "enable_filetype_ingest_router", True))

    def _cowork_ingest_artifact_retention_max_age_days(self) -> int:
        runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
        if runner_limits is None:
            return 14
        return max(0, int(getattr(runner_limits, "ingest_artifact_retention_max_age_days", 14)))

    def _cowork_ingest_artifact_retention_max_files_per_scope(self) -> int:
        runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
        if runner_limits is None:
            return 96
        return max(
            1,
            int(getattr(runner_limits, "ingest_artifact_retention_max_files_per_scope", 96)),
        )

    def _cowork_ingest_artifact_retention_max_bytes_per_scope(self) -> int:
        runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
        if runner_limits is None:
            return 268_435_456
        return max(
            1024,
            int(getattr(
                runner_limits,
                "ingest_artifact_retention_max_bytes_per_scope",
                268_435_456,
            )),
        )

    async def _initialize_session(self) -> None:
        """Initialize tools, session, and welcome message.

        Called from on_mount (normal start) or _finalize_setup (post-wizard).
        Requires self._model to be set.
        """
        chat = self.query_one("#chat-log", ChatLog)
        self._total_tokens = 0
        self._sidebar_cowork_tasks = []

        # Start from a clean registry each initialization to avoid stale
        # excludes/bindings when setup or process configuration changes.
        self._refresh_tool_registry()

        # Load process definition (imports bundled tools, then refreshes).
        self._load_process_definition(chat)
        self._refresh_process_command_index(chat=chat, notify_conflicts=True)

        # Ensure persistence-dependent tools are present and tracked.
        self._ensure_persistence_tools()

        # Build system prompt
        self._apply_process_tool_policy(chat)
        system_prompt = self._build_system_prompt()

        # Build approver
        approver = ToolApprover(prompt_callback=self._approval_callback)

        resume_target, auto_resume = await self._resolve_startup_resume_target()

        # Create or resume session
        if self._store is not None and resume_target:
            # Resume existing session
            self._session = CoworkSession(
                model=self._model,
                tools=self._tools,
                workspace=self._workspace,
                scratch_dir=self._cowork_scratch_dir(),
                system_prompt=system_prompt,
                approver=approver,
                store=self._store,
                model_retry_policy=self._model_retry_policy(),
                enable_filetype_ingest_router=self._cowork_enable_filetype_ingest_router(),
                ingest_artifact_retention_max_age_days=self._cowork_ingest_artifact_retention_max_age_days(),
                ingest_artifact_retention_max_files_per_scope=self._cowork_ingest_artifact_retention_max_files_per_scope(),
                ingest_artifact_retention_max_bytes_per_scope=self._cowork_ingest_artifact_retention_max_bytes_per_scope(),
            )
            try:
                await self._session.resume(resume_target)
                self._total_tokens = self._session.total_tokens
                resume_label = (
                    "Resumed latest workspace session"
                    if auto_resume
                    else "Resumed session"
                )
                chat.add_info(
                    f"[bold #7dcfff]{resume_label}[/bold #7dcfff]\n"
                    f"  [bold]Session ID:[/] [dim]{self._escape_markup(resume_target)}[/dim]\n"
                    f"  [bold]Turns:[/] {self._session.session_state.turn_count}"
                )
            except Exception as e:
                chat.add_info(
                    f"[bold #f7768e]Resume failed: {e}[/] "
                    f"Starting fresh session."
                )
                self._session = None

        if self._session is not None:
            # Successfully resumed — keep it
            pass
        elif self._store is not None:
            # New persisted session
            session_id = await self._store.create_session(
                workspace=str(self._workspace),
                model_name=self._model.name,
                system_prompt=system_prompt,
            )
            self._session = CoworkSession(
                model=self._model,
                tools=self._tools,
                workspace=self._workspace,
                scratch_dir=self._cowork_scratch_dir(),
                system_prompt=system_prompt,
                approver=approver,
                store=self._store,
                session_id=session_id,
                model_retry_policy=self._model_retry_policy(),
                enable_filetype_ingest_router=self._cowork_enable_filetype_ingest_router(),
                ingest_artifact_retention_max_age_days=self._cowork_ingest_artifact_retention_max_age_days(),
                ingest_artifact_retention_max_files_per_scope=self._cowork_ingest_artifact_retention_max_files_per_scope(),
                ingest_artifact_retention_max_bytes_per_scope=self._cowork_ingest_artifact_retention_max_bytes_per_scope(),
            )
        else:
            # Ephemeral session (no database)
            self._session = CoworkSession(
                model=self._model,
                tools=self._tools,
                workspace=self._workspace,
                scratch_dir=self._cowork_scratch_dir(),
                system_prompt=system_prompt,
                approver=approver,
                model_retry_policy=self._model_retry_policy(),
                enable_filetype_ingest_router=self._cowork_enable_filetype_ingest_router(),
                ingest_artifact_retention_max_age_days=self._cowork_ingest_artifact_retention_max_age_days(),
                ingest_artifact_retention_max_files_per_scope=self._cowork_ingest_artifact_retention_max_files_per_scope(),
                ingest_artifact_retention_max_bytes_per_scope=self._cowork_ingest_artifact_retention_max_bytes_per_scope(),
            )

        self._hydrate_input_history_from_session()

        # Bind session-dependent tools
        self._bind_session_tools()

        # Configure status bar
        status = self.query_one("#status-bar", StatusBar)
        status.workspace_name = self._workspace.name
        status.model_name = self._model.name
        status.process_name = self._active_process_name()

        # Welcome message
        tool_count = len(self._tools.list_tools())
        persisted = "persisted" if self._store else "ephemeral"
        chat.add_info(self._render_startup_summary(tool_count=tool_count, persisted=persisted))
        await self._restore_process_run_tabs(chat)
        self._process_close_hint_shown = bool(self._process_runs)

        # Resume is a one-shot startup hint; subsequent reinitializations
        # should not keep trying to reopen the same prior session.
        self._resume_session = None
        self._auto_resume_workspace_on_init = False

        self.query_one("#user-input", Input).focus()
        # Ensure command/footer bars are visible after any prior slash-hint state.
        self._set_slash_hint("")
        self._refresh_sidebar_progress_summary()

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
        """Create a short unique run ID for display and routing."""
        while True:
            run_id = uuid.uuid4().hex[:6]
            if run_id not in self._process_runs:
                return run_id

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed seconds as MM:SS or H:MM:SS."""
        total = max(0, int(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _elapsed_seconds_for_run(self, run: ProcessRunState) -> float:
        """Return elapsed seconds for a run (live or finalized)."""
        end = run.ended_at if run.ended_at is not None else time.monotonic()
        return max(0.0, end - run.started_at)

    def _append_process_run_activity(
        self, run: ProcessRunState, message: str,
    ) -> None:
        """Record and render one process-run activity line."""
        text = self._one_line(message, 1200)
        if not text:
            return
        log = getattr(run, "activity_log", None)
        if not isinstance(log, list):
            try:
                run.activity_log = []
                log = run.activity_log
            except Exception:
                log = None
        if isinstance(log, list):
            log.append(text)
            if len(log) > _MAX_PERSISTED_PROCESS_ACTIVITY:
                del log[:-_MAX_PERSISTED_PROCESS_ACTIVITY]
        try:
            run.pane.add_activity(text)
        except Exception:
            pass

    def _append_process_run_result(
        self, run: ProcessRunState, text: str, *, success: bool,
    ) -> None:
        """Record and render one process-run final result line."""
        message = _plain_text(text).strip()
        if not message:
            message = "Process run completed." if success else "Process run failed."
        records = getattr(run, "result_log", None)
        if not isinstance(records, list):
            try:
                run.result_log = []
                records = run.result_log
            except Exception:
                records = None
        if isinstance(records, list):
            records.append({"text": message, "success": bool(success)})
            if len(records) > _MAX_PERSISTED_PROCESS_RESULTS:
                del records[:-_MAX_PERSISTED_PROCESS_RESULTS]
        try:
            run.pane.add_result(message, success=success)
        except Exception:
            pass

    def _serialize_process_run_state(self, run: ProcessRunState) -> dict:
        """Serialize one in-memory process run for session UI persistence."""
        tasks = [
            dict(row)
            for row in getattr(run, "tasks", [])
            if isinstance(row, dict)
        ][-_MAX_PERSISTED_PROCESS_ACTIVITY:]
        labels = getattr(run, "task_labels", {})
        if not isinstance(labels, dict):
            labels = {}
        activity = [
            self._one_line(line, 1200)
            for line in getattr(run, "activity_log", [])
            if self._one_line(line, 1200)
        ][-_MAX_PERSISTED_PROCESS_ACTIVITY:]
        result_log: list[dict] = []
        for item in getattr(run, "result_log", []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            result_log.append({"text": text, "success": bool(item.get("success", False))})
        result_log = result_log[-_MAX_PERSISTED_PROCESS_RESULTS:]
        status = str(getattr(run, "status", "completed")).strip()
        if status not in _PROCESS_STATUS_LABEL:
            status = "completed"
        return {
            "run_id": str(getattr(run, "run_id", "")).strip(),
            "process_name": str(getattr(run, "process_name", "")).strip(),
            "goal": str(getattr(run, "goal", "")).strip(),
            "run_workspace": str(getattr(run, "run_workspace", self._workspace)),
            "status": status,
            "task_id": str(getattr(run, "task_id", "")).strip(),
            "elapsed_seconds": float(self._elapsed_seconds_for_run(run)),
            "tasks": tasks,
            "task_labels": {str(k): str(v) for k, v in labels.items()},
            "activity_log": activity,
            "result_log": result_log,
            "auth_profile_overrides": {
                str(k): str(v)
                for k, v in getattr(run, "auth_profile_overrides", {}).items()
                if str(k).strip() and str(v).strip()
            },
            "auth_required_resources": [
                dict(item)
                for item in getattr(run, "auth_required_resources", [])
                if isinstance(item, dict)
            ],
        }

    def _sync_process_runs_into_session_state(self) -> None:
        """Mirror process-run tab state into SessionState.ui_state."""
        session = self._session
        if session is None:
            return
        state = getattr(session, "session_state", None)
        if state is None:
            return
        ui_state = getattr(state, "ui_state", None)
        if not isinstance(ui_state, dict):
            ui_state = {}
            try:
                state.ui_state = ui_state
            except Exception:
                return

        serialized_runs = [
            self._serialize_process_run_state(run)
            for run in sorted(self._process_runs.values(), key=lambda r: r.started_at)
            if not getattr(run, "closed", False)
        ][-_MAX_PERSISTED_PROCESS_RUNS:]

        active_run_id = ""
        try:
            tabs = self.query_one("#tabs", TabbedContent)
            active_tab = str(getattr(tabs, "active", "") or "")
            for run in self._process_runs.values():
                if getattr(run, "pane_id", "") == active_tab:
                    active_run_id = str(getattr(run, "run_id", "")).strip()
                    break
        except Exception:
            pass

        ui_state["process_tabs"] = {
            "version": 1,
            "active_run_id": active_run_id,
            "runs": serialized_runs,
        }

    def _sync_input_history_into_session_state(self) -> None:
        """Mirror input history into SessionState.ui_state."""
        session = self._session
        if session is None:
            return
        state = getattr(session, "session_state", None)
        if state is None:
            return
        ui_state = getattr(state, "ui_state", None)
        if not isinstance(ui_state, dict):
            ui_state = {}
            try:
                state.ui_state = ui_state
            except Exception:
                return
        ui_state["input_history"] = {
            "version": 1,
            "items": list(self._input_history[-_MAX_INPUT_HISTORY:]),
        }

    async def _persist_process_run_ui_state(
        self, *, is_active: bool | None = None,
    ) -> None:
        """Persist SessionState (including process-tab UI state) to storage."""
        session = self._session
        store = self._store
        if session is None or store is None:
            return
        session_id = str(getattr(session, "session_id", "") or "").strip()
        if not session_id:
            return

        self._sync_process_runs_into_session_state()
        self._sync_input_history_into_session_state()
        payload: dict = {}
        state = getattr(session, "session_state", None)
        to_dict = getattr(state, "to_dict", None)
        if callable(to_dict):
            try:
                payload["session_state"] = to_dict()
            except Exception:
                pass
        if is_active is not None:
            payload["is_active"] = is_active
        if not payload:
            return
        try:
            await store.update_session(session_id, **payload)
        except Exception as e:
            logger.debug("Failed to persist process UI state: %s", e)

    def _persisted_process_tabs_payload(self) -> tuple[list[dict], str]:
        """Return persisted process-tab payload from SessionState.ui_state."""
        session = self._session
        if session is None:
            return [], ""
        state = getattr(session, "session_state", None)
        if state is None:
            return [], ""
        ui_state = getattr(state, "ui_state", None)
        if not isinstance(ui_state, dict):
            return [], ""

        payload = ui_state.get("process_tabs")
        if isinstance(payload, dict):
            runs = payload.get("runs", [])
            active = str(payload.get("active_run_id", "")).strip()
            if isinstance(runs, list):
                return runs, active
        legacy_runs = ui_state.get("process_runs")
        if isinstance(legacy_runs, list):
            return legacy_runs, ""
        return [], ""

    async def _drop_process_run_tabs(self) -> None:
        """Remove all process run panes from the UI and clear in-memory state."""
        if not self._process_runs:
            return
        tabs = None
        try:
            tabs = self.query_one("#tabs", TabbedContent)
        except Exception:
            tabs = None

        for run in list(self._process_runs.values()):
            worker = getattr(run, "worker", None)
            if worker is not None and hasattr(worker, "cancel"):
                try:
                    worker.cancel()
                except Exception:
                    pass
            pane_id = str(getattr(run, "pane_id", "") or "").strip()
            if tabs is not None and pane_id:
                try:
                    await tabs.remove_pane(pane_id)
                except Exception:
                    pass
        self._process_runs.clear()
        if tabs is not None:
            try:
                if not tabs.active:
                    tabs.active = "tab-chat"
            except Exception:
                pass
        self._refresh_sidebar_progress_summary()

    async def _restore_process_run_tabs(self, chat: ChatLog | None = None) -> None:
        """Restore process run tabs for the current resumed session."""
        runs_payload, active_run_id = self._persisted_process_tabs_payload()
        await self._drop_process_run_tabs()
        if not runs_payload:
            self._refresh_sidebar_progress_summary()
            return
        try:
            tabs = self.query_one("#tabs", TabbedContent)
        except Exception:
            self._refresh_sidebar_progress_summary()
            return

        loader = None
        try:
            loader = self._create_process_loader()
        except Exception:
            loader = None

        restored = 0
        interrupted = 0
        seen_ids: set[str] = set()

        for raw in runs_payload[:_MAX_PERSISTED_PROCESS_RUNS]:
            if not isinstance(raw, dict):
                continue

            run_id = str(raw.get("run_id", "")).strip()[:12]
            if not run_id:
                run_id = self._new_process_run_id()
            while run_id in seen_ids or run_id in self._process_runs:
                run_id = self._new_process_run_id()
            seen_ids.add(run_id)

            process_name = str(raw.get("process_name", "")).strip() or "process"
            goal = str(raw.get("goal", "")).strip() or "(restored run)"
            status = str(raw.get("status", "completed")).strip()
            if status not in _PROCESS_STATUS_LABEL:
                status = "completed"
            task_id = str(raw.get("task_id", "")).strip()
            try:
                elapsed_seconds = max(0.0, float(raw.get("elapsed_seconds", 0.0)))
            except (TypeError, ValueError):
                elapsed_seconds = 0.0
            started_at = time.monotonic() - elapsed_seconds
            ended_at = (
                None
                if status in {"queued", "running"}
                else time.monotonic()
            )

            tasks = [
                dict(row)
                for row in raw.get("tasks", [])
                if isinstance(row, dict)
            ]
            labels_raw = raw.get("task_labels", {})
            task_labels = (
                {str(k): str(v) for k, v in labels_raw.items()}
                if isinstance(labels_raw, dict)
                else {}
            )
            activity_log = [
                self._one_line(line, 1200)
                for line in raw.get("activity_log", [])
                if self._one_line(line, 1200)
            ][-_MAX_PERSISTED_PROCESS_ACTIVITY:]
            result_log: list[dict] = []
            for item in raw.get("result_log", []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                result_log.append({"text": text, "success": bool(item.get("success", False))})
            result_log = result_log[-_MAX_PERSISTED_PROCESS_RESULTS:]
            auth_profile_overrides_raw = raw.get("auth_profile_overrides", {})
            auth_profile_overrides = {}
            if isinstance(auth_profile_overrides_raw, dict):
                for key, value in auth_profile_overrides_raw.items():
                    selector = str(key or "").strip()
                    profile_id = str(value or "").strip()
                    if selector and profile_id:
                        auth_profile_overrides[selector] = profile_id
            auth_required_resources = [
                dict(item)
                for item in raw.get("auth_required_resources", [])
                if isinstance(item, dict)
            ]

            process_defn = None
            if loader is not None:
                try:
                    process_defn = loader.load(process_name)
                except Exception:
                    process_defn = None

            pane_id = f"tab-run-{run_id}"
            pane = ProcessRunPane(
                run_id=run_id,
                process_name=process_name,
                goal=goal,
            )
            run_workspace_raw = str(raw.get("run_workspace", "")).strip()
            run_workspace = Path(run_workspace_raw or str(self._workspace)).expanduser()
            try:
                run_workspace.resolve().relative_to(self._workspace.resolve())
            except Exception:
                run_workspace = self._workspace
            run = ProcessRunState(
                run_id=run_id,
                process_name=process_name,
                goal=goal,
                run_workspace=run_workspace,
                process_defn=process_defn,
                pane_id=pane_id,
                pane=pane,
                status=status,
                task_id=task_id,
                started_at=started_at,
                ended_at=ended_at,
                tasks=tasks,
                task_labels=task_labels,
                activity_log=activity_log,
                result_log=result_log,
                auth_profile_overrides=auth_profile_overrides,
                auth_required_resources=auth_required_resources,
            )

            if run.status in {"queued", "running"}:
                interrupted += 1
                run.status = "failed"
                run.ended_at = time.monotonic()
                note = "Run interrupted before session resume; marked failed."
                run.activity_log.append(note)
                run.result_log.append({"text": note, "success": False})

            self._process_runs[run_id] = run
            await tabs.add_pane(
                TabPane(
                    self._format_process_run_tab_title(run),
                    pane,
                    id=pane_id,
                ),
                after="tab-events",
            )
            run.pane.set_tasks(run.tasks)
            self._refresh_process_run_outputs(run)
            for line in run.activity_log:
                run.pane.add_activity(line)
            for item in run.result_log:
                run.pane.add_result(
                    str(item.get("text", "")),
                    success=bool(item.get("success", False)),
                )
            if run.activity_log:
                run.last_progress_message = run.activity_log[-1]
                run.last_progress_at = time.monotonic()
            self._update_process_run_visuals(run)
            restored += 1

        if active_run_id and active_run_id in self._process_runs:
            tabs.active = self._process_runs[active_run_id].pane_id
        self._refresh_sidebar_progress_summary()

        if restored and chat is not None:
            info = (
                "[bold #7dcfff]Restored Process Tabs[/bold #7dcfff]\n"
                f"  [bold]Count:[/] {restored}"
            )
            if interrupted:
                info += (
                    "\n"
                    f"  [#f7768e]{interrupted} interrupted run(s) were marked failed. "
                    "Use /run resume <run-id-prefix> to continue.[/]"
                )
            chat.add_info(info)

    def _format_process_run_tab_title(self, run: ProcessRunState) -> str:
        """Build tab title with status indicator and elapsed timer."""
        icon = _PROCESS_STATUS_ICON.get(run.status, "\u25cb")
        elapsed = self._format_elapsed(self._elapsed_seconds_for_run(run))
        name = run.process_name
        if len(name) > 16:
            name = f"{name[:15]}\u2026"
        return f"{icon} {name} #{run.run_id} {elapsed}"

    def _update_process_run_visuals(self, run: ProcessRunState) -> None:
        """Update pane header and tab title from current run state."""
        elapsed = self._format_elapsed(self._elapsed_seconds_for_run(run))
        run.pane.set_status_header(
            status=run.status,
            elapsed=elapsed,
            task_id=run.task_id,
        )
        try:
            tabs = self.query_one("#tabs", TabbedContent)
            tab = tabs.get_tab(run.pane_id)
            tab.label = self._format_process_run_tab_title(run)
        except Exception:
            pass

    def _tick_process_run_elapsed(self) -> None:
        """Periodic timer to refresh elapsed text for active process runs."""
        active = False
        for run in self._process_runs.values():
            if run.status in {"queued", "running"}:
                active = True
                self._update_process_run_visuals(run)
        if active:
            self._refresh_sidebar_progress_summary()

    def _build_process_run_context(self, goal: str, *, workspace: Path) -> dict:
        """Build compact cowork context to pass into delegated process runs."""
        if self._session is None:
            return {
                "requested_goal": goal,
                "workspace": str(workspace),
                "source_workspace_root": str(self._workspace.resolve()),
            }
        state = self._session.session_state
        context: dict = {
            "requested_goal": goal,
            "workspace": str(workspace),
            "source_workspace_root": str(self._workspace.resolve()),
            "cowork": {
                "turn_count": state.turn_count,
                "current_focus": state.current_focus,
                "key_decisions": state.key_decisions[-8:],
            },
        }
        recent_messages: list[dict[str, str]] = []
        for message in reversed(self._session.messages):
            role = str(message.get("role", "")).strip()
            if role not in {"user", "assistant"}:
                continue
            content = self._one_line(message.get("content", ""), 500)
            if not content:
                continue
            recent_messages.append({"role": role, "content": content})
            if len(recent_messages) >= 6:
                break
        if recent_messages:
            recent_messages.reverse()
            context["cowork"]["recent_messages"] = recent_messages
        return context

    @staticmethod
    def _slugify_process_run_folder(value: str, *, max_len: int = 48) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
        slug = slug.strip("-")
        if not slug:
            return ""
        if len(slug) > max_len:
            slug = slug[:max_len].strip("-")
        return slug

    @staticmethod
    def _run_goal_for_folder_name(goal: str) -> str:
        text = " ".join(str(goal or "").split()).strip()
        if not text:
            return ""
        prefixes = (
            r"^(?:please\s+)?i\s+need\s+you\s+to\s+",
            r"^(?:please\s+)?can\s+you\s+",
            r"^(?:please\s+)?could\s+you\s+",
            r"^(?:please\s+)?help\s+me\s+(?:to\s+)?",
            r"^the\s+user\s+wants(?:\s+me)?\s+to\s+",
        )
        for pattern in prefixes:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        return text[:240]

    @classmethod
    def _extract_run_folder_slug(cls, response_text: str) -> str:
        text = str(response_text or "").strip()
        if not text:
            return ""
        first_line = text.splitlines()[0].strip().strip("`").strip("\"'")
        match = re.search(r"[a-z0-9]+(?:-[a-z0-9]+)+", first_line.lower())
        candidate = match.group(0) if match else first_line
        return cls._slugify_process_run_folder(candidate, max_len=48)

    @staticmethod
    def _is_low_quality_run_folder_slug(slug: str) -> bool:
        value = str(slug or "").strip().lower()
        if not value:
            return True
        tokens = [part for part in value.split("-") if part]
        if len(tokens) < 2:
            return True
        if tokens[:3] == ["the", "user", "wants"]:
            return True
        if tokens[:4] == ["i", "need", "you", "to"]:
            return True

        banned_tokens = {
            "the",
            "user",
            "wants",
            "i",
            "need",
            "you",
            "to",
            "folder",
            "name",
            "kebab",
            "case",
            "run",
            "process",
            "task",
            "request",
            "prompt",
            "query",
            "for",
            "a",
            "pr",
        }
        banned_hits = sum(1 for token in tokens if token in banned_tokens)
        return banned_hits >= max(2, (len(tokens) + 1) // 2)

    def _fallback_process_run_folder_name(self, process_name: str, goal: str) -> str:
        merged = f"{process_name} {goal}".strip().lower()
        tokens = re.findall(r"[a-z0-9]+", merged)
        base = "-".join(tokens[:6])
        slug = self._slugify_process_run_folder(base)
        return slug or "process-run"

    async def _llm_process_run_folder_name(self, process_name: str, goal: str) -> str:
        process_cfg = getattr(self._config, "process", None)
        if not bool(getattr(process_cfg, "llm_run_folder_naming_enabled", True)):
            return ""
        if not self._has_configured_role_model("extractor"):
            return ""
        goal_seed = self._run_goal_for_folder_name(goal)
        prompt = (
            "Return exactly one kebab-case folder name for this process run.\n"
            "Requirements:\n"
            "- Output ONLY the slug (single line, no quotes, no backticks, no explanation).\n"
            "- 2-6 words, lowercase letters/numbers/hyphens only.\n"
            "- Use concrete topic words from the goal.\n"
            "- Do NOT echo prompt scaffolding or meta wording such as "
            "\"the user wants\", \"i need you to\", \"folder\", \"name\", or \"for a pr\".\n"
            f"Process label: {process_name}\n"
            f"Goal: {goal_seed or goal}\n"
            "Slug:"
        )
        try:
            response, _, _, _ = await self._invoke_helper_role_completion(
                role="extractor",
                tier=1,
                prompt=prompt,
                max_tokens=20,
                temperature=0.2,
            )
        except Exception as e:
            logger.warning("LLM run-folder naming failed: %s", e)
            return ""
        text = str(getattr(response, "text", "") or "")
        slug = self._extract_run_folder_slug(text)
        if self._is_low_quality_run_folder_slug(slug):
            logger.debug(
                "Discarding low-quality LLM run-folder name '%s' for goal '%s'",
                slug,
                goal_seed or goal,
            )
            return ""
        return slug

    async def _prepare_process_run_workspace(
        self,
        process_name: str,
        goal: str,
    ) -> Path:
        process_cfg = getattr(self._config, "process", None)
        if not bool(getattr(process_cfg, "tui_run_scoped_workspace_enabled", True)):
            return self._workspace

        root = self._workspace.resolve()
        slug = await self._llm_process_run_folder_name(process_name, goal)
        if not slug:
            slug = self._fallback_process_run_folder_name(process_name, goal)

        for suffix in range(1, 1000):
            candidate_name = slug if suffix == 1 else f"{slug}-{suffix}"
            candidate = root / candidate_name
            try:
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            except FileExistsError:
                continue
            except OSError as e:
                logger.warning("Failed to create run workspace %s: %s", candidate, e)
                break

        return self._workspace

    def _current_process_run(self) -> ProcessRunState | None:
        """Return run associated with currently active tab, if any."""
        try:
            tabs = self.query_one("#tabs", TabbedContent)
            active = tabs.active
        except Exception:
            return None
        if not active:
            return None
        return next(
            (r for r in self._process_runs.values() if r.pane_id == active),
            None,
        )

    def _resolve_process_run_target(
        self, target: str,
    ) -> tuple[ProcessRunState | None, str | None]:
        """Resolve a process run by target selector or current tab."""
        token = target.strip().lstrip("#")
        if not token or token.lower() in {"current", "this"}:
            current = self._current_process_run()
            if current is not None:
                return current, None
            if len(self._process_runs) == 1:
                return next(iter(self._process_runs.values())), None
            if not self._process_runs:
                return None, "No process run tabs are open."
            return None, "Multiple runs open. Use /run close <run-id-prefix>."

        matches = [
            run
            for run in self._process_runs.values()
            if run.run_id.startswith(token)
        ]
        if not matches:
            return None, f"No run found matching '{token}'."
        if len(matches) > 1:
            return None, f"Ambiguous run prefix '{token}'."
        return matches[0], None

    async def _confirm_close_process_run(self, run: ProcessRunState) -> bool:
        """Prompt before closing a process run tab."""
        waiter: asyncio.Future[bool] = asyncio.Future()
        screen = None

        def handle_result(confirmed: bool) -> None:
            if not waiter.done():
                waiter.set_result(bool(confirmed))

        running = run.status in {"queued", "running"}
        screen = ProcessRunCloseScreen(
            run_label=f"{run.process_name} #{run.run_id}",
            running=running,
        )
        self.push_screen(screen, callback=handle_result)
        try:
            return await waiter
        except asyncio.CancelledError:
            # If the close-flow worker is cancelled, dismiss the modal so the
            # UI doesn't end up blocked behind an orphaned confirmation screen.
            try:
                if screen is not None:
                    screen.dismiss(False)
            except Exception:
                pass
            raise

    async def _close_process_run(self, run: ProcessRunState) -> bool:
        """Close a process run tab; active runs are marked failed/cancelled."""
        if run.closed:
            return False
        chat = self.query_one("#chat-log", ChatLog)
        events_panel = self.query_one("#events-panel", EventPanel)
        tabs = self.query_one("#tabs", TabbedContent)

        if not await self._confirm_close_process_run(run):
            return False

        was_running = run.status in {"queued", "running"}
        run.closed = True
        if was_running:
            run.status = "failed"
            run.ended_at = time.monotonic()
            self._append_process_run_activity(run, "Run cancelled: tab closed by user.")
            self._append_process_run_result(
                run, "Run cancelled: tab closed by user.", success=False,
            )
            self._update_process_run_visuals(run)
            events_panel.add_event(
                _now_str(),
                "process_err",
                f"{run.process_name} #{run.run_id} cancelled",
            )
            chat.add_info(
                f"[bold #f7768e]Process run {run.run_id} cancelled:[/] tab closed."
            )
            worker = run.worker
            if worker is not None and hasattr(worker, "cancel"):
                try:
                    worker.cancel()
                except Exception:
                    pass
        else:
            chat.add_info(f"Closed process run tab [dim]{run.run_id}[/dim].")

        try:
            await tabs.remove_pane(run.pane_id)
        except Exception:
            pass
        self._process_runs.pop(run.run_id, None)
        self._refresh_sidebar_progress_summary()
        await self._persist_process_run_ui_state()
        if not tabs.active:
            tabs.active = "tab-chat"
        return True

    async def _close_process_run_from_target(self, target: str) -> bool:
        """Resolve and close a process run from /run close target syntax."""
        chat = self.query_one("#chat-log", ChatLog)
        run, error = self._resolve_process_run_target(target)
        if run is None:
            if error:
                chat.add_info(error)
            return False
        return await self._close_process_run(run)

    async def _resume_process_run_from_target(self, target: str) -> bool:
        """Resolve and resume a failed/cancelled process run from /run resume."""
        chat = self.query_one("#chat-log", ChatLog)
        run, error = self._resolve_process_run_target(target)
        if run is None:
            if error:
                if error == "Multiple runs open. Use /run close <run-id-prefix>.":
                    error = "Multiple runs open. Use /run resume <run-id-prefix>."
                chat.add_info(error)
            return False

        chat.add_user_message(f"/run resume {target}")
        return await self._restart_process_run_in_place(run.run_id, mode="resume")

    @staticmethod
    def _resume_seed_task_rows(run: ProcessRunState) -> tuple[list[dict], dict[str, str]]:
        """Clone prior task rows for resume; keep completed rows and reset the rest."""
        rows: list[dict] = []
        row_ids: set[str] = set()
        for item in getattr(run, "tasks", []):
            if not isinstance(item, dict):
                continue
            cloned = dict(item)
            status = str(cloned.get("status", "pending")).strip()
            if status != "completed":
                status = "pending"
            cloned["status"] = status
            subtask_id = str(cloned.get("id", "")).strip()
            if subtask_id:
                row_ids.add(subtask_id)
            rows.append(cloned)

        labels: dict[str, str] = {}
        raw_labels = getattr(run, "task_labels", {})
        if isinstance(raw_labels, dict):
            for key, value in raw_labels.items():
                subtask_id = str(key).strip()
                if not subtask_id:
                    continue
                if row_ids and subtask_id not in row_ids:
                    continue
                labels[subtask_id] = str(value)
        return rows, labels

    async def _restart_process_run_in_place(
        self,
        run_id: str,
        *,
        mode: str = "restart",
    ) -> bool:
        """Restart one failed/cancelled run in the same tab."""
        run = self._process_runs.get(run_id)
        if run is None or run.closed:
            return False

        normalized_mode = str(mode or "").strip().lower()
        is_resume = normalized_mode == "resume"
        verb_denied = "resumed" if is_resume else "restarted"
        verb_ongoing = "Resuming" if is_resume else "Restarting"
        verb_done = "Resumed" if is_resume else "Restarted"
        event_action = "resumed" if is_resume else "restarted"

        chat = self.query_one("#chat-log", ChatLog)
        events_panel = self.query_one("#events-panel", EventPanel)
        if run.status in {"queued", "running"}:
            chat.add_info(
                f"Run [dim]{run.run_id}[/dim] is already active and cannot be {verb_denied}."
            )
            return False
        if not run.task_id:
            chat.add_info(
                f"Run [dim]{run.run_id}[/dim] has no task ID, so it cannot be {verb_denied}."
            )
            return False

        run.status = "queued"
        run.started_at = time.monotonic()
        run.ended_at = None
        run.tasks, run.task_labels = self._resume_seed_task_rows(run)
        run.last_progress_message = ""
        run.last_progress_at = 0.0
        self._update_process_run_visuals(run)
        run.pane.set_tasks(run.tasks)
        self._refresh_process_run_outputs(run)
        self._append_process_run_activity(
            run,
            f"{verb_ongoing} in place from task state {run.task_id}.",
        )
        run.worker = self.run_worker(
            self._execute_process_run(run_id),
            name=f"process-run-{run_id}",
            group=f"process-run-{run_id}",
            exclusive=False,
        )
        self._refresh_sidebar_progress_summary()
        chat.add_info(
            f"{verb_done} process run [dim]{run.run_id}[/dim] in place."
        )
        events_panel.add_event(
            _now_str(),
            "process_run",
            f"{run.process_name} #{run.run_id} {event_action}",
        )
        await self._persist_process_run_ui_state()
        return True

    @staticmethod
    def _format_auth_profile_option(profile: Any) -> str:
        """Render one concise auth profile choice label."""
        profile_id = str(getattr(profile, "profile_id", "") or "").strip()
        label = str(getattr(profile, "account_label", "") or "").strip()
        mcp_server = str(getattr(profile, "mcp_server", "") or "").strip()
        parts = [profile_id]
        if label:
            parts.append(f"label={label}")
        if mcp_server:
            parts.append(f"mcp_server={mcp_server}")
        return " | ".join(parts)

    async def _prompt_auth_choice(self, question: str, options: list[str]) -> str:
        """Prompt for one auth selection via modal and return chosen option text."""
        answer_event = asyncio.Event()
        selected: list[str] = []

        def _handle(answer: str) -> None:
            selected.append(str(answer or "").strip())
            answer_event.set()

        self.push_screen(AskUserScreen(question, options), callback=_handle)
        await answer_event.wait()
        return selected[0] if selected else ""

    async def _open_auth_manager_for_run_start(
        self,
        *,
        process_def: ProcessDefinition | None = None,
    ) -> bool:
        """Open auth manager during run-start flow and wait for completion."""
        done = asyncio.Event()
        changed = {"value": False}

        def _handle(result: dict[str, object] | None) -> None:
            changed["value"] = bool(
                isinstance(result, dict) and result.get("changed")
            )
            done.set()

        self.push_screen(
            AuthManagerScreen(
                workspace=self._workspace,
                explicit_auth_path=self._explicit_auth_path,
                mcp_manager=self._mcp_manager(),
                process_def=process_def or self._process_defn,
                tool_registry=self._tools,
            ),
            callback=_handle,
        )
        await done.wait()
        return bool(changed["value"])

    def _collect_required_auth_resources_for_process(
        self,
        process_defn: ProcessDefinition | None,
    ) -> list[dict[str, Any]]:
        """Collect process + allowed-tool auth requirements in metadata shape."""
        if process_defn is None:
            return []

        from loom.auth.runtime import (
            coerce_auth_requirements,
            serialize_auth_requirements,
        )

        raw_items: list[object] = []
        auth_block = getattr(process_defn, "auth", None)
        process_required = getattr(auth_block, "required", [])
        if isinstance(process_required, list):
            raw_items.extend(process_required)

        tools_cfg = getattr(process_defn, "tools", None)
        excluded = {
            str(item).strip()
            for item in (getattr(tools_cfg, "excluded", []) or [])
            if str(item).strip()
        }
        for tool_name in sorted(
            {
                str(name).strip()
                for name in self._tools.list_tools()
                if str(name).strip() and str(name).strip() not in excluded
            }
        ):
            tool = self._tools.get(tool_name)
            if tool is None:
                continue
            declared = getattr(tool, "auth_requirements", [])
            if isinstance(declared, list):
                raw_items.extend(declared)

        return serialize_auth_requirements(coerce_auth_requirements(raw_items))

    async def _resolve_auth_overrides_for_run_start(
        self,
        *,
        process_defn: ProcessDefinition | None,
        base_overrides: dict[str, str],
    ) -> tuple[dict[str, str] | None, list[dict[str, Any]]]:
        """Resolve run-start auth, prompting for ambiguous resources when needed."""
        from loom.auth.config import (
            load_merged_auth_config,
            set_workspace_auth_default,
        )
        from loom.auth.runtime import (
            UnresolvedAuthResourcesError,
            build_run_auth_context,
            coerce_auth_requirements,
        )

        required_resources = self._collect_required_auth_resources_for_process(process_defn)
        if not required_resources:
            return dict(base_overrides), required_resources

        overrides = dict(base_overrides)
        while True:
            metadata: dict[str, Any] = {
                "auth_workspace": str(self._workspace.resolve()),
                "auth_required_resources": required_resources,
                "auth_profile_overrides": dict(overrides),
            }
            if self._explicit_auth_path is not None:
                metadata["auth_config_path"] = str(self._explicit_auth_path.resolve())

            try:
                auth_context = await asyncio.to_thread(
                    build_run_auth_context,
                    workspace=self._workspace,
                    metadata=metadata,
                    required_resources=coerce_auth_requirements(required_resources),
                    available_mcp_aliases=set(self._config.mcp.servers.keys()),
                )
            except UnresolvedAuthResourcesError as e:
                unresolved = list(e.unresolved)
                blocking = [
                    item
                    for item in unresolved
                    if str(getattr(item, "reason", "")).strip()
                    not in {"ambiguous", "blocked_ambiguous_binding"}
                ]
                if blocking:
                    chat = self.query_one("#chat-log", ChatLog)
                    lines = [
                        "[bold #f7768e]Run auth preflight failed.[/]",
                    ]
                    for item in blocking:
                        lines.append(
                            f"  - provider={item.provider} source={item.source} "
                            f"reason={item.reason}"
                        )
                        message = str(item.message or "").strip()
                        if message:
                            lines.append(f"    {message}")
                    chat.add_info("\n".join(lines))
                    choice = await self._prompt_auth_choice(
                        "Open Auth Manager to fix auth and retry this run?",
                        ["Open Auth Manager", "Cancel run"],
                    )
                    if choice != "Open Auth Manager":
                        chat.add_info("Run cancelled: unresolved auth requirements.")
                        return None, required_resources
                    changed = await self._open_auth_manager_for_run_start(
                        process_def=process_defn,
                    )
                    if changed:
                        chat.add_info("Auth configuration updated. Retrying preflight.")
                    else:
                        chat.add_info(
                            "Auth Manager closed without changes. Retrying preflight."
                        )
                    continue

                made_selection = False
                for item in unresolved:
                    candidates = list(item.candidates)
                    if not candidates:
                        continue
                    merged = await asyncio.to_thread(
                        load_merged_auth_config,
                        workspace=self._workspace,
                        explicit_path=self._explicit_auth_path,
                    )
                    options: list[str] = []
                    option_to_profile: dict[str, str] = {}
                    for candidate_id in candidates:
                        profile = merged.config.profiles.get(candidate_id)
                        if profile is None:
                            label = str(candidate_id)
                        else:
                            label = self._format_auth_profile_option(profile)
                        option_to_profile[label] = candidate_id
                        options.append(label)
                    question = (
                        "Select auth profile for "
                        f"provider={item.provider} source={item.source}"
                    )
                    answer = await self._prompt_auth_choice(question, options)
                    profile_id = option_to_profile.get(answer)
                    if not profile_id:
                        chat = self.query_one("#chat-log", ChatLog)
                        chat.add_info("Run cancelled: auth selection was not completed.")
                        return None, required_resources
                    selector = (
                        str(getattr(item, "resource_id", "")).strip()
                        or str(getattr(item, "resource_ref", "")).strip()
                        or str(getattr(item, "provider", "")).strip()
                    )
                    if not selector:
                        chat = self.query_one("#chat-log", ChatLog)
                        chat.add_info(
                            "Run cancelled: unresolved auth item has no selector."
                        )
                        return None, required_resources
                    overrides[selector] = profile_id
                    save_default = await self._prompt_auth_choice(
                        (
                            "Save this as workspace default for "
                            f"{selector}?"
                        ),
                        ["No", "Yes"],
                    )
                    if save_default == "Yes":
                        try:
                            if str(getattr(item, "resource_id", "")).strip():
                                from loom.auth.resources import (
                                    default_workspace_auth_resources_path,
                                    set_workspace_resource_default,
                                )

                                await asyncio.to_thread(
                                    set_workspace_resource_default,
                                    default_workspace_auth_resources_path(
                                        self._workspace.resolve()
                                    ),
                                    resource_id=str(item.resource_id).strip(),
                                    profile_id=profile_id,
                                )
                            else:
                                await asyncio.to_thread(
                                    set_workspace_auth_default,
                                    self._auth_defaults_path(),
                                    selector=selector,
                                    profile_id=profile_id,
                                )
                        except Exception as e:
                            chat = self.query_one("#chat-log", ChatLog)
                            chat.add_info(
                                f"[bold #f7768e]Failed to save workspace auth default: {e}[/]"
                            )
                    made_selection = True
                if not made_selection:
                    chat = self.query_one("#chat-log", ChatLog)
                    chat.add_info("Run cancelled: auth selection was not completed.")
                    return None, required_resources
                continue
            except Exception as e:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_info(f"[bold #f7768e]Auth preflight failed: {e}[/]")
                return None, required_resources

            for req in coerce_auth_requirements(required_resources):
                selector = (
                    str(getattr(req, "resource_id", "")).strip()
                    or str(getattr(req, "resource_ref", "")).strip()
                    or str(getattr(req, "provider", "")).strip()
                )
                profile_for_selector = getattr(
                    auth_context,
                    "profile_for_selector",
                    None,
                )
                profile = None
                if callable(profile_for_selector):
                    profile = profile_for_selector(selector)
                if profile is None:
                    profile = auth_context.profile_for_provider(req.provider)
                if profile is None:
                    continue
                if selector:
                    overrides[selector] = profile.profile_id
            return overrides, required_resources

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
    ) -> None:
        """Create a run tab and launch process execution in a background worker."""
        chat = self.query_one("#chat-log", ChatLog)
        events_panel = self.query_one("#events-panel", EventPanel)

        selected_process = process_defn or self._process_defn
        if selected_process is None and not resume_task_id:
            chat.add_info(
                "[bold #f7768e]No active process. Use /process use "
                "<name-or-path> first.[/]"
            )
            return

        if not self._tools.has("delegate_task"):
            chat.add_info(
                "[bold #f7768e]Process orchestration is unavailable in this "
                "session.[/]",
            )
            return

        active_runs = sum(
            1 for run in self._process_runs.values()
            if run.status in {"queued", "running"}
        )
        if active_runs >= _MAX_CONCURRENT_PROCESS_RUNS:
            chat.add_info(
                f"[bold #f7768e]Too many active process runs "
                f"({_MAX_CONCURRENT_PROCESS_RUNS} max).[/]"
            )
            return

        process_name = str(process_name_override or "").strip()
        if not process_name and selected_process is not None:
            process_name = selected_process.name
        if not process_name:
            process_name = "resumed-process-run"
        if run_workspace_override is not None:
            run_workspace = Path(run_workspace_override).expanduser()
        else:
            run_workspace = await self._prepare_process_run_workspace(process_name, goal)
        run_auth_overrides = dict(self._run_auth_profile_overrides)
        resolved_auth_overrides, required_auth_resources = (
            await self._resolve_auth_overrides_for_run_start(
                process_defn=selected_process,
                base_overrides=run_auth_overrides,
            )
        )
        if resolved_auth_overrides is None:
            return
        run_auth_overrides = resolved_auth_overrides
        run_id = self._new_process_run_id()
        pane_id = f"tab-run-{run_id}"
        pane = ProcessRunPane(
            run_id=run_id,
            process_name=process_name,
            goal=goal,
        )
        run = ProcessRunState(
            run_id=run_id,
            process_name=process_name,
            goal=goal,
            run_workspace=run_workspace,
            process_defn=selected_process,
            pane_id=pane_id,
            pane=pane,
            status="queued",
            task_id=str(resume_task_id or "").strip(),
            started_at=time.monotonic(),
            is_adhoc=bool(is_adhoc),
            recommended_tools=list(recommended_tools or []),
            goal_context_overrides=dict(goal_context_overrides or {}),
            auth_profile_overrides=dict(run_auth_overrides),
            auth_required_resources=list(required_auth_resources),
        )
        self._process_runs[run_id] = run

        tabs = self.query_one("#tabs", TabbedContent)
        await tabs.add_pane(
            TabPane(
                self._format_process_run_tab_title(run),
                pane,
                id=pane_id,
            ),
            after="tab-events",
        )
        tabs.active = pane_id
        self._update_process_run_visuals(run)
        run.pane.set_tasks(run.tasks)
        self._refresh_process_run_outputs(run)
        self._refresh_sidebar_progress_summary()

        chat.add_user_message(f"{command_prefix} {goal}")
        chat.add_info(
            f"Started process run [dim]{run_id}[/dim] "
            f"([bold]{process_name}[/bold])."
        )
        self._append_process_run_activity(
            run,
            f"Run workspace: {run_workspace}",
        )
        if bool(getattr(run, "is_adhoc", False)):
            self._append_process_run_activity(run, "Run mode: synthesized ad hoc process.")
            for note in list(adhoc_synthesis_notes or []):
                self._append_process_run_activity(run, note)
            if run.recommended_tools:
                self._append_process_run_activity(
                    run,
                    "Recommended extra tools: " + ", ".join(sorted(run.recommended_tools)),
                )
        if run.auth_profile_overrides:
            rendered = ", ".join(
                f"{selector}={profile_id}"
                for selector, profile_id in sorted(run.auth_profile_overrides.items())
            )
            self._append_process_run_activity(run, f"Run auth overrides: {rendered}")
        if run.task_id:
            self._append_process_run_activity(
                run,
                f"Resuming task state: {run.task_id}",
            )
        self._append_process_run_activity(run, "Queued process run.")
        if not self._process_close_hint_shown:
            chat.add_info(
                "[dim]Tip: close process tabs with Ctrl+W, /run close "
                "[run-id-prefix], or Ctrl+P -> Close process run tab.[/]"
            )
            self._process_close_hint_shown = True
        events_panel.add_event(
            _now_str(),
            "process_run",
            f"{process_name} #{run_id}: {goal[:120]}",
        )

        run.worker = self.run_worker(
            self._execute_process_run(run_id),
            name=f"process-run-{run_id}",
            group=f"process-run-{run_id}",
            exclusive=False,
        )
        await self._persist_process_run_ui_state()

    async def _execute_process_run(self, run_id: str) -> None:
        """Execute one process run and stream updates into its dedicated tab."""
        run = self._process_runs.get(run_id)
        if run is None:
            return
        if run.closed:
            return
        chat = self.query_one("#chat-log", ChatLog)
        events_panel = self.query_one("#events-panel", EventPanel)

        run.status = "running"
        run.started_at = time.monotonic()
        run.ended_at = None
        self._update_process_run_visuals(run)
        self._refresh_sidebar_progress_summary()
        self._append_process_run_activity(run, "Run started.")

        try:
            run_context = self._build_process_run_context(
                run.goal,
                workspace=run.run_workspace,
            )
            extra_context = getattr(run, "goal_context_overrides", {})
            if isinstance(extra_context, dict) and extra_context:
                run_context.update(extra_context)
            result = await self._tools.execute(
                "delegate_task",
                {
                    "goal": run.goal,
                    "context": run_context,
                    "wait": True,
                    "_approval_mode": "disabled",
                    "_process_override": run.process_defn,
                    "_read_roots": [str(self._workspace.resolve())],
                    "_auth_profile_overrides": dict(
                        getattr(run, "auth_profile_overrides", self._run_auth_profile_overrides),
                    ),
                    "_auth_required_resources": list(
                        getattr(run, "auth_required_resources", []),
                    ),
                    "_auth_workspace": str(self._workspace.resolve()),
                    "_resume_task_id": str(run.task_id or "").strip(),
                    "_progress_callback": (
                        lambda data, rid=run_id: self._on_process_progress_event(
                            data, run_id=rid,
                        )
                    ),
                },
                workspace=run.run_workspace,
            )
            data = getattr(result, "data", None)
            if run.closed:
                return
            if isinstance(data, dict):
                run.task_id = str(data.get("task_id", "") or run.task_id)
                event_log_path = str(data.get("event_log_path", "")).strip()
                if event_log_path:
                    self._append_process_run_activity(
                        run,
                        f"Detailed log: {event_log_path}",
                    )
                tasks = data.get("tasks", [])
                if isinstance(tasks, list):
                    normalized = self._normalize_process_run_tasks(run, tasks)
                    run.tasks = normalized
                    run.pane.set_tasks(normalized)
                    self._refresh_process_run_outputs(run)
            delegated_status = ""
            if isinstance(data, dict):
                delegated_status = str(data.get("status", "")).strip().lower()
            failed_terminal_status = delegated_status in {"failed", "cancelled"}
            run_succeeded = bool(result.success) and not failed_terminal_status

            if run_succeeded:
                output = result.output or "Process run completed."
                self._append_process_run_result(run, output, success=True)
                run.status = "completed"
                run.ended_at = time.monotonic()
                self._update_process_run_visuals(run)
                self._refresh_sidebar_progress_summary()
                events_panel.add_event(
                    _now_str(), "process_ok", f"{run.process_name} #{run.run_id}",
                )
                chat.add_info(f"Process run [dim]{run.run_id}[/dim] completed.")
            else:
                if result.success and failed_terminal_status:
                    detail = result.output or f"Process run {delegated_status}."
                    error = f"Process run {delegated_status}."
                    self._append_process_run_result(run, detail, success=False)
                else:
                    error = result.error or result.output or "Process run failed."
                    self._append_process_run_result(run, error, success=False)
                run.status = "failed"
                run.ended_at = time.monotonic()
                self._update_process_run_visuals(run)
                self._refresh_sidebar_progress_summary()
                events_panel.add_event(
                    _now_str(), "process_err", f"{run.process_name} #{run.run_id}",
                )
                chat.add_info(
                    f"[bold #f7768e]Process run {run.run_id} failed:[/] {error}"
                )
                self.notify(error, severity="error", timeout=5)
            self._refresh_workspace_tree()
        except asyncio.CancelledError:
            if run.closed:
                return
            run.status = "cancelled"
            run.ended_at = time.monotonic()
            self._append_process_run_result(run, "Process run cancelled.", success=False)
            self._update_process_run_visuals(run)
            self._refresh_sidebar_progress_summary()
            events_panel.add_event(
                _now_str(), "process_err", f"{run.process_name} #{run.run_id}",
            )
            chat.add_info(f"[bold #f7768e]Process run {run.run_id} cancelled.[/]")
            raise
        except Exception as e:  # pragma: no cover - defensive guard
            self._append_process_run_result(run, str(e), success=False)
            run.status = "failed"
            run.ended_at = time.monotonic()
            self._update_process_run_visuals(run)
            self._refresh_sidebar_progress_summary()
            events_panel.add_event(
                _now_str(), "process_err", f"{run.process_name} #{run.run_id}",
            )
            chat.add_info(f"[bold #f7768e]Process run {run.run_id} failed:[/] {e}")
            self.notify(str(e), severity="error", timeout=5)
        finally:
            run.worker = None
            await self._persist_process_run_ui_state()

    def _ensure_persistence_tools(self) -> None:
        """Ensure recall/delegate tools are registered and tracked.

        Tool auto-discovery may pre-register these tool names. We must bind
        and use the *registered* instances, otherwise /run can hit an
        unbound delegate_task instance ("no orchestrator configured").
        """
        self._recall_tool = None
        self._delegate_tool = None
        if self._store is None:
            return

        from loom.tools.conversation_recall import ConversationRecallTool
        from loom.tools.delegate_task import DelegateTaskTool

        recall = self._tools.get("conversation_recall")
        if recall is not None and not isinstance(recall, ConversationRecallTool):
            logger.warning(
                "Replacing unexpected conversation_recall tool type: %s",
                type(recall).__name__,
            )
            self._tools.exclude("conversation_recall")
            recall = None
        if recall is None:
            recall = ConversationRecallTool()
            self._tools.register(recall)
        self._recall_tool = recall

        delegate = self._tools.get("delegate_task")
        if delegate is not None and not isinstance(delegate, DelegateTaskTool):
            logger.warning(
                "Replacing unexpected delegate_task tool type: %s",
                type(delegate).__name__,
            )
            self._tools.exclude("delegate_task")
            delegate = None
        if delegate is None:
            delegate = DelegateTaskTool()
            self._tools.register(delegate)
        self._delegate_tool = delegate

    def _slash_command_catalog(self) -> list[tuple[str, str]]:
        """Return canonical slash commands with optional alias annotations."""
        entries: list[tuple[str, str]] = []
        for spec in _SLASH_COMMANDS:
            label = spec.canonical
            if spec.usage:
                label = f"{label} {spec.usage}"
            desc = spec.description
            if spec.aliases:
                desc = f"{desc} ({', '.join(spec.aliases)})"
            entries.append((label, desc))
        for token, process_name in sorted(self._process_command_map.items()):
            entries.append((f"{token} <goal>", f"run goal via process '{process_name}'"))
        return entries

    @staticmethod
    def _slash_match_keys(spec: SlashCommandSpec) -> tuple[str, ...]:
        """Return normalized command tokens used for prefix matching."""
        return (spec.canonical.lower(), *(alias.lower() for alias in spec.aliases))

    def _help_lines(self) -> list[str]:
        """Build slash help lines from the shared command registry."""
        lines = ["[bold #7dcfff]Slash Commands[/bold #7dcfff]"]
        for spec in _SLASH_COMMANDS:
            label = spec.canonical
            if spec.usage:
                label = f"{label} {spec.usage}"
            if spec.aliases:
                alias_str = ", ".join(spec.aliases)
                label = f"{label} (aliases: {alias_str})"
            lines.append(f"  [bold]{self._escape_markup(label)}[/bold]")
            lines.append(
                self._wrap_info_text(
                    self._escape_markup(spec.description),
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )
        if self._process_command_map:
            lines.append("")
            lines.append("[bold #7dcfff]Process Slash Commands[/bold #7dcfff]")
            for token, process_name in sorted(self._process_command_map.items()):
                lines.append(f"  [bold]{self._escape_markup(token)} <goal>[/bold]")
                lines.append(
                    self._wrap_info_text(
                        self._escape_markup(
                            f"Run goal via process '{process_name}'"
                        ),
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                )
        if self._blocked_process_commands:
            blocked = ", ".join(f"/{name}" for name in self._blocked_process_commands)
            lines.append("")
            lines.append(
                "[#f7768e]Blocked process commands (name collisions): "
                f"{self._escape_markup(blocked)}[/]"
            )
        lines.extend([
            "",
            "[bold #7dcfff]Keys[/bold #7dcfff]",
            self._wrap_info_text(
                "Ctrl+B sidebar, Ctrl+L clear, Ctrl+R reload workspace, "
                "Ctrl+W close run tab, Ctrl+P palette, Ctrl+1/2/3 tabs",
                initial_indent="  ",
                subsequent_indent="  ",
            ),
        ])
        return lines

    def _render_slash_command_usage(self, command: str, usage: str) -> str:
        """Render a usage error block for slash commands."""
        return (
            f"[bold #7dcfff]Usage[/bold #7dcfff]\n"
            f"  [bold]{self._escape_markup(command)}[/bold] {self._escape_markup(usage)}"
        )

    def _matching_slash_commands(
        self,
        raw_input: str,
    ) -> tuple[str, list[tuple[str, str]]]:
        """Return current slash token and matching commands."""
        text = raw_input.strip()
        if not text.startswith("/"):
            return "", []
        # Keep dynamic process slash commands in sync as the user types.
        self._refresh_process_command_index()
        token = text.split()[0].lower()
        if token == "/":
            return token, self._slash_command_catalog()

        matches: list[tuple[str, str]] = []
        fallback_matches: list[tuple[str, str]] = []
        for spec in _SLASH_COMMANDS:
            keys = self._slash_match_keys(spec)
            label = spec.canonical
            if spec.usage:
                label = f"{label} {spec.usage}"
            desc = spec.description
            if spec.aliases:
                desc = f"{desc} ({', '.join(spec.aliases)})"
            entry = (label, desc)
            if any(key.startswith(token) for key in keys):
                matches.append(entry)
            elif any(token in key for key in keys):
                fallback_matches.append(entry)
        for dynamic_token, process_name in sorted(self._process_command_map.items()):
            entry = (
                f"{dynamic_token} <goal>",
                f"run goal via process '{process_name}'",
            )
            if dynamic_token.startswith(token):
                matches.append(entry)
            elif token in dynamic_token:
                fallback_matches.append(entry)
        if matches:
            return token, matches
        return token, fallback_matches

    def _reset_slash_tab_cycle(self) -> None:
        """Clear slash tab-completion cycle state."""
        self._slash_cycle_seed = ""
        self._slash_cycle_candidates = []

    def _reset_input_history_navigation(self) -> None:
        """Clear active input-history navigation state."""
        self._input_history_nav_index = None
        self._input_history_nav_draft = ""

    def _clear_input_history(self) -> None:
        """Drop in-memory input history and reset navigation."""
        self._input_history = []
        self._reset_input_history_navigation()

    def _append_input_history(self, value: str) -> None:
        """Record one executed user input in bounded history."""
        entry = str(value or "").strip()
        if not entry:
            return
        self._input_history.append(entry)
        if len(self._input_history) > _MAX_INPUT_HISTORY:
            del self._input_history[:-_MAX_INPUT_HISTORY]
        self._sync_input_history_into_session_state()
        self._reset_input_history_navigation()

    def _hydrate_input_history_from_session(self) -> None:
        """Populate input history from the active session's user messages."""
        self._clear_input_history()
        if self._session is None:
            return
        state = getattr(self._session, "session_state", None)
        ui_state = getattr(state, "ui_state", None) if state is not None else None
        if isinstance(ui_state, dict):
            payload = ui_state.get("input_history")
            items: list[object] | None = None
            if isinstance(payload, dict):
                raw_items = payload.get("items")
                if isinstance(raw_items, list):
                    items = raw_items
            elif isinstance(payload, list):
                items = payload
            if items is not None:
                self._input_history = [
                    str(item).strip() for item in items
                    if str(item).strip()
                ][-_MAX_INPUT_HISTORY:]
                return
        restored: list[str] = []
        messages = getattr(self._session, "messages", [])
        if not isinstance(messages, list):
            messages = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if role != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            text = content.strip()
            if text:
                restored.append(text)
        if len(restored) > _MAX_INPUT_HISTORY:
            restored = restored[-_MAX_INPUT_HISTORY:]
        self._input_history = restored

    def _set_user_input_text(self, value: str, *, from_history_navigation: bool = False) -> None:
        """Update the main input box value and keep the cursor at the end."""
        input_widget = self.query_one("#user-input", Input)
        if from_history_navigation:
            self._applying_input_history_navigation = True
            self._skip_input_history_reset_once = True
        try:
            input_widget.value = value
            input_widget.cursor_position = len(value)
        finally:
            if from_history_navigation:
                self._applying_input_history_navigation = False

    def _apply_input_history_navigation(self, *, older: bool) -> bool:
        """Move through recorded input history like a shell."""
        if not self._input_history:
            return False
        input_widget = self.query_one("#user-input", Input)
        if self._input_history_nav_index is None:
            self._input_history_nav_index = len(self._input_history)
            self._input_history_nav_draft = input_widget.value
        index = self._input_history_nav_index
        if index is None:
            return False

        if older:
            next_index = max(0, index - 1)
            self._input_history_nav_index = next_index
            text = self._input_history[next_index]
        else:
            if index >= len(self._input_history):
                return False
            next_index = index + 1
            if next_index >= len(self._input_history):
                self._input_history_nav_index = len(self._input_history)
                text = self._input_history_nav_draft
            else:
                self._input_history_nav_index = next_index
                text = self._input_history[next_index]

        self._set_user_input_text(text, from_history_navigation=True)
        return True

    def _slash_completion_candidates(self, token: str) -> list[str]:
        """Return slash command completions for a token prefix."""
        self._refresh_process_command_index()
        if token == "/":
            builtins = [spec.canonical for spec in _SLASH_COMMANDS]
            dynamic = sorted(self._process_command_map)
            return builtins + dynamic

        candidates: list[str] = []
        seen: set[str] = set()
        for spec in _SLASH_COMMANDS:
            for key in (spec.canonical, *spec.aliases):
                if key.startswith(token) and key not in seen:
                    candidates.append(key)
                    seen.add(key)
        for key in sorted(self._process_command_map):
            if key.startswith(token) and key not in seen:
                candidates.append(key)
                seen.add(key)
        return candidates

    def _process_use_completion_candidates(
        self,
        raw_input: str,
    ) -> tuple[str, list[str]] | None:
        """Return `/process use` tab-completion seed and candidates."""
        text = raw_input.strip()
        match = re.fullmatch(
            r"/process\s+use(?:\s+(?P<prefix>\S*))?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None

        prefix = (match.group("prefix") or "").strip()
        base = "/process use"
        seed = f"{base} {prefix}" if prefix else base

        self._refresh_process_command_index()
        available = self._cached_process_catalog

        candidates: list[str] = []
        seen: set[str] = set()
        prefix_lower = prefix.lower()
        for proc in available:
            name = str(proc.get("name", "")).strip()
            if not name or name in seen:
                continue
            if prefix and not name.lower().startswith(prefix_lower):
                continue
            candidates.append(f"{base} {name}")
            seen.add(name)
        return seed, candidates

    def _render_process_use_hint(self, raw_input: str) -> str | None:
        """Render contextual hint rows for `/process use ...`."""
        text = raw_input.strip()
        match = re.fullmatch(
            r"/process\s+use(?:\s+(?P<prefix>\S*))?",
            text,
            re.IGNORECASE,
        )
        if not match:
            return None

        prefix = (match.group("prefix") or "").strip()
        self._refresh_process_command_index()
        available = self._cached_process_catalog

        if not available:
            if self._blocked_process_commands:
                blocked = ", ".join(self._blocked_process_commands)
                return (
                    "[#f7768e]No selectable process definitions found.[/]  "
                    f"[dim]Blocked: {blocked}[/]"
                )
            return (
                "[#f7768e]No process definitions found.[/]  "
                "[dim]Try /process list[/]"
            )

        active = self._process_defn.name if self._process_defn else ""
        prefix_lower = prefix.lower()
        rows: list[tuple[str, str, str]] = []
        for proc in available:
            name = str(proc.get("name", "")).strip()
            if not name:
                continue
            if prefix and not name.lower().startswith(prefix_lower):
                continue
            version = str(proc.get("version", "?")).strip() or "?"
            marker = " [cyan]<< active[/cyan]" if active and name == active else ""
            rows.append((name, version, marker))

        if not rows:
            return (
                f"[#f7768e]No processes match '{prefix}'.[/]  "
                "[dim]Try /process list[/]"
            )

        title = (
            "Available processes for /process use:"
            if not prefix
            else f"Process matches '{prefix}':"
        )
        lines = [f"[bold #7dcfff]{title}[/]"]
        max_rows = 12
        for name, version, marker in rows[:max_rows]:
            lines.append(f"  [#73daca]{name:<30}[/] [dim]v{version}[/]{marker}")
        remaining = len(rows) - max_rows
        if remaining > 0:
            lines.append(f"  [dim]... and {remaining} more[/dim]")
        lines.append("[dim]Press Tab to autocomplete[/dim]")
        return "\n".join(lines)

    def _apply_slash_tab_completion(self, *, reverse: bool = False) -> bool:
        """Apply slash tab completion (forward/backward)."""
        input_widget = self.query_one("#user-input", Input)
        current = input_widget.value.strip()
        if not current.startswith("/"):
            self._reset_slash_tab_cycle()
            return False

        completion_scope = "slash"
        token = current
        process_completion = self._process_use_completion_candidates(current)
        if process_completion is not None:
            completion_scope = "process_use"
            token, scoped_candidates = process_completion
        elif " " in current:
            self._reset_slash_tab_cycle()
            return False

        if current in self._slash_cycle_candidates:
            candidates = self._slash_cycle_candidates
            current_index = candidates.index(current)
            next_index = (
                (current_index - 1) if reverse else (current_index + 1)
            ) % len(candidates)
        else:
            if completion_scope == "process_use":
                candidates = scoped_candidates
            else:
                candidates = self._slash_completion_candidates(token)
            if not candidates:
                self._reset_slash_tab_cycle()
                return False
            self._slash_cycle_seed = token
            self._slash_cycle_candidates = candidates
            next_index = len(candidates) - 1 if reverse else 0

        completion = candidates[next_index]
        self._applying_slash_tab_completion = True
        self._skip_slash_cycle_reset_once = True
        try:
            input_widget.value = completion
            input_widget.cursor_position = len(completion)
        finally:
            self._applying_slash_tab_completion = False
        return True

    def _render_slash_hint(self, raw_input: str) -> str:
        """Build slash-command hint text for the current input."""
        process_hint = self._render_process_use_hint(raw_input)
        if process_hint is not None:
            return process_hint
        if " " in raw_input.strip():
            return ""
        token, matches = self._matching_slash_commands(raw_input)
        if not token:
            return ""

        if not matches:
            return (
                f"[#f7768e]No command matches '{token}'[/]  "
                "[dim]Try /help[/]"
            )

        title = "Slash commands:" if token == "/" else f"Matching {token}:"
        lines = [f"[bold #7dcfff]{title}[/]"]
        for cmd, desc in matches:
            safe_cmd = self._escape_markup(cmd)
            safe_desc = self._escape_markup(desc)
            lines.append(f"  [#73daca]{safe_cmd:<10}[/] {safe_desc}")
        return "\n".join(lines)

    @staticmethod
    def _strip_wrapping_quotes(value: str) -> str:
        """Remove matching wrapping quotes from a command argument."""
        text = value.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            return text[1:-1].strip()
        return text

    def _set_slash_hint(self, hint_text: str) -> None:
        """Show or hide the slash-command hint panel."""
        hint = self.query_one("#slash-hint", Static)
        if hint_text:
            hint.update(hint_text)
            hint.display = True
            line_count = max(1, len(hint_text.splitlines()))
            hint.styles.height = min(line_count, _MAX_SLASH_HINT_LINES)
            hint.scroll_home(animate=False)
        else:
            hint.display = False
            hint.styles.height = "auto"
            hint.update("")

    def _bind_session_tools(self) -> None:
        """Bind tools that hold a reference to the active session."""
        if self._session is None:
            return
        if self._recall_tool and self._store:
            self._recall_tool.bind(
                store=self._store,
                session_id=self._session.session_id,
                session_state=self._session.session_state,
                compactor=getattr(self._session, "compactor", None),
            )
        if self._delegate_tool and self._config and self._db:
            try:
                from loom.engine.orchestrator import Orchestrator
                from loom.events.bus import EventBus
                from loom.models.router import ModelRouter
                from loom.prompts.assembler import PromptAssembler
                from loom.state.memory import MemoryManager
                from loom.state.task_state import TaskStateManager
                from loom.tools import create_default_registry as _create_tools

                config = self._config
                db = self._db

                if hasattr(config, "workspace"):
                    data_dir = Path(
                        config.workspace.scratch_dir,
                    ).expanduser()
                else:
                    data_dir = Path.home() / ".loom"

                router = ModelRouter.from_config(config)

                async def _orchestrator_factory(
                    process_override: ProcessDefinition | None = None,
                ):
                    return Orchestrator(
                        model_router=router,
                        tool_registry=_create_tools(),
                        memory_manager=MemoryManager(db),
                        prompt_assembler=PromptAssembler(),
                        state_manager=TaskStateManager(data_dir),
                        event_bus=EventBus(),
                        config=config,
                        process=process_override or self._process_defn,
                    )

                self._delegate_tool.bind(_orchestrator_factory)
            except Exception as e:
                logger.warning("Failed to bind delegate_task tool: %s", e)

    # ------------------------------------------------------------------
    # Session management helpers
    # ------------------------------------------------------------------

    def _clear_files_panel(self) -> None:
        """Reset file history/diff when changing sessions."""
        panel = self.query_one("#files-panel", FilesChangedPanel)
        panel.clear_files()
        panel.show_diff("")

    async def _new_session(self) -> None:
        """Create a fresh session, replacing the current one."""
        if self._store is None or self._session is None or self._model is None:
            return

        # Persist any UI state for the old session before rotating.
        await self._persist_process_run_ui_state(is_active=False)

        system_prompt = self._build_system_prompt()
        approver = ToolApprover(prompt_callback=self._approval_callback)
        session_id = await self._store.create_session(
            workspace=str(self._workspace),
            model_name=self._model.name,
            system_prompt=system_prompt,
        )
        self._session = CoworkSession(
            model=self._model,
            tools=self._tools,
            workspace=self._workspace,
            scratch_dir=self._cowork_scratch_dir(),
            system_prompt=system_prompt,
            approver=approver,
            store=self._store,
            session_id=session_id,
            model_retry_policy=self._model_retry_policy(),
            enable_filetype_ingest_router=self._cowork_enable_filetype_ingest_router(),
            ingest_artifact_retention_max_age_days=self._cowork_ingest_artifact_retention_max_age_days(),
            ingest_artifact_retention_max_files_per_scope=self._cowork_ingest_artifact_retention_max_files_per_scope(),
            ingest_artifact_retention_max_bytes_per_scope=self._cowork_ingest_artifact_retention_max_bytes_per_scope(),
        )
        self._total_tokens = 0
        self._bind_session_tools()
        self._hydrate_input_history_from_session()
        self._clear_files_panel()
        chat = self.query_one("#chat-log", ChatLog)
        await self._restore_process_run_tabs(chat)
        self._process_close_hint_shown = bool(self._process_runs)
        chat.add_info(
            "[bold #7dcfff]New Session Created[/bold #7dcfff]\n"
            f"  [bold]Session ID:[/] [dim]{self._escape_markup(session_id)}[/dim]"
        )

    async def _switch_to_session(self, session_id: str) -> None:
        """Resume a different session by ID."""
        if self._store is None or self._session is None or self._model is None:
            return

        system_prompt = self._build_system_prompt()
        approver = ToolApprover(prompt_callback=self._approval_callback)

        # Persist outgoing session UI state before switching.
        await self._persist_process_run_ui_state(is_active=False)

        new_session = CoworkSession(
            model=self._model,
            tools=self._tools,
            workspace=self._workspace,
            scratch_dir=self._cowork_scratch_dir(),
            system_prompt=system_prompt,
            approver=approver,
            store=self._store,
            model_retry_policy=self._model_retry_policy(),
            enable_filetype_ingest_router=self._cowork_enable_filetype_ingest_router(),
            ingest_artifact_retention_max_age_days=self._cowork_ingest_artifact_retention_max_age_days(),
            ingest_artifact_retention_max_files_per_scope=self._cowork_ingest_artifact_retention_max_files_per_scope(),
            ingest_artifact_retention_max_bytes_per_scope=self._cowork_ingest_artifact_retention_max_bytes_per_scope(),
        )
        await new_session.resume(session_id)

        self._session = new_session
        self._total_tokens = new_session.total_tokens
        self._bind_session_tools()
        self._hydrate_input_history_from_session()
        self._clear_files_panel()
        chat = self.query_one("#chat-log", ChatLog)
        await self._restore_process_run_tabs(chat)
        self._process_close_hint_shown = bool(self._process_runs)
        chat.add_info(
            "[bold #7dcfff]Switched Session[/bold #7dcfff]\n"
            f"  [bold]Session ID:[/] [dim]{self._escape_markup(session_id)}[/dim]\n"
            f"  [bold]Turns:[/] {new_session.session_state.turn_count}"
        )

    # ------------------------------------------------------------------
    # Learned patterns
    # ------------------------------------------------------------------

    async def _show_learned_patterns(self) -> None:
        """Show the learned behavioral patterns review modal."""
        from loom.learning.manager import LearningManager

        mgr = LearningManager(self._db)
        patterns = await mgr.query_behavioral(limit=50)

        def handle_result(result: str) -> None:
            if result:
                self._delete_learned_patterns(result)

        self.push_screen(LearnedScreen(patterns), callback=handle_result)

    @work
    async def _delete_learned_patterns(self, deleted_ids_csv: str) -> None:
        """Delete patterns whose IDs were selected in the review screen."""
        from loom.learning.manager import LearningManager

        if not deleted_ids_csv or not self._db:
            return

        mgr = LearningManager(self._db)
        chat = self.query_one("#chat-log", ChatLog)
        count = 0

        for raw_id in deleted_ids_csv.split(","):
            raw_id = raw_id.strip()
            if not raw_id:
                continue
            try:
                pid = int(raw_id)
                if await mgr.delete_pattern(pid):
                    count += 1
            except (ValueError, Exception):
                pass

        if count:
            chat.add_info(f"Deleted {count} learned pattern(s).")

    # ------------------------------------------------------------------
    # Approval callback
    # ------------------------------------------------------------------

    async def _approval_callback(
        self, tool_name: str, args: dict,
    ) -> ApprovalDecision:
        """Show approval modal and wait for result."""
        preview = tool_args_preview(tool_name, args)

        self._approval_event = asyncio.Event()
        self._approval_result = ApprovalDecision.DENY

        def handle_result(result: str) -> None:
            if result == "approve":
                self._approval_result = ApprovalDecision.APPROVE
            elif result == "approve_all":
                self._approval_result = ApprovalDecision.APPROVE_ALL
            else:
                self._approval_result = ApprovalDecision.DENY
            if self._approval_event:
                self._approval_event.set()

        self.push_screen(
            ToolApprovalScreen(tool_name, preview),
            callback=handle_result,
        )

        await self._approval_event.wait()
        self._approval_event = None
        return self._approval_result

    # ------------------------------------------------------------------
    # User input
    # ------------------------------------------------------------------

    @on(Input.Submitted, "#user-input")
    async def on_user_submit(self, event: Input.Submitted) -> None:
        # P0-2: Only handle submissions from the main chat input
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        self._reset_slash_tab_cycle()
        self._reset_input_history_navigation()
        self._set_slash_hint("")

        # Handle slash commands
        if text.startswith("/"):
            handled = await self._handle_slash_command(text)
            if handled:
                self._append_input_history(text)
                await self._persist_process_run_ui_state()
                return

        if self._chat_busy:
            return

        self._append_input_history(text)
        self._run_turn(text)

    @on(Input.Changed, "#user-input")
    def on_user_input_changed(self, _event: Input.Changed) -> None:
        """Show slash-command hints as the user types."""
        if self._skip_input_history_reset_once:
            self._skip_input_history_reset_once = False
        elif not self._applying_input_history_navigation:
            self._reset_input_history_navigation()
        if self._skip_slash_cycle_reset_once:
            self._skip_slash_cycle_reset_once = False
        elif not self._applying_slash_tab_completion:
            self._reset_slash_tab_cycle()
        # Use the widget's current value rather than event payload to avoid
        # stale-value edge cases that can appear one keypress behind.
        current = self.query_one("#user-input", Input).value
        self._set_slash_hint(self._render_slash_hint(current))

    @on(Button.Pressed, ".process-run-restart-btn")
    def on_process_run_restart_pressed(self, event: Button.Pressed) -> None:
        """Restart a failed process run directly from its tab button."""
        button_id = str(getattr(event.button, "id", "") or "").strip()
        if not button_id.startswith("process-run-restart-"):
            return
        run_id = button_id.removeprefix("process-run-restart-").strip()
        if not run_id:
            return
        event.stop()
        event.prevent_default()
        self.run_worker(
            self._restart_process_run_in_place(run_id),
            name=f"process-run-restart-{run_id}",
            group=f"process-run-restart-{run_id}",
            exclusive=False,
        )

    def on_key(self, event: events.Key) -> None:
        """Handle user-input key captures (autocomplete + close-run shortcut)."""
        if event.key == "ctrl+w":
            focused = self.focused
            if isinstance(focused, Input) and focused.id == "user-input":
                event.stop()
                event.prevent_default()
                self.action_close_process_tab()
            return

        if event.key in ("up", "down"):
            focused = self.focused
            if isinstance(focused, Input) and focused.id == "user-input":
                if self._apply_input_history_navigation(older=event.key == "up"):
                    event.stop()
                    event.prevent_default()
            return

        if event.key not in ("tab", "shift+tab"):
            return
        focused = self.focused
        if not isinstance(focused, Input) or focused.id != "user-input":
            return
        if self._apply_slash_tab_completion(reverse=event.key == "shift+tab"):
            event.stop()
            event.prevent_default()

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
        self.push_screen(FileViewerScreen(selected, self._workspace))

    async def _handle_slash_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        raw = text.strip()
        parts = raw.split(None, 1)
        token = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        chat = self.query_one("#chat-log", ChatLog)
        self._refresh_process_command_index()

        if token in ("/quit", "/exit", "/q"):
            self.action_request_quit()
            return True
        if token == "/clear":
            self.action_clear_chat()
            return True
        if token == "/help":
            self._show_help()
            return True
        if token == "/model":
            name = self._model.name if self._model else "(not configured)"
            chat.add_info(f"Model: {name}")
            return True
        if token == "/mcp":
            from loom.mcp.config import (
                MCPConfigManagerError,
                ensure_valid_alias,
                merge_server_edits,
                parse_mcp_server_from_flags,
            )

            manager = self._mcp_manager()
            if not arg:
                self._open_mcp_manager_screen()
                return True

            subparts = arg.split(None, 1)
            subcmd = subparts[0].lower()
            rest = subparts[1].strip() if len(subparts) > 1 else ""

            if subcmd == "manage":
                self._open_mcp_manager_screen()
                return True

            if subcmd == "list":
                try:
                    merged = await asyncio.to_thread(manager.load)
                    views = merged.as_views()
                    output = self._render_mcp_list(views)
                    if any(view.source == "legacy" for view in views):
                        output += (
                            "\n[dim]Legacy MCP config detected in loom.toml. "
                            "Run `loom mcp migrate` from CLI.[/dim]"
                        )
                    chat.add_info(output)
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]MCP list failed: {e}[/]")
                return True

            if subcmd == "show":
                if not rest:
                    chat.add_info(self._render_slash_command_usage("/mcp show", "<alias>"))
                    return True
                try:
                    alias = ensure_valid_alias(rest)
                    view = await asyncio.to_thread(manager.get_view, alias)
                except MCPConfigManagerError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                if view is None:
                    chat.add_info(
                        f"[bold #f7768e]MCP server not found: {alias}[/]"
                    )
                    return True
                output = self._render_mcp_view(view)
                if view.source == "legacy":
                    output += (
                        "\n[dim]This alias comes from legacy loom.toml. "
                        "Run `loom mcp migrate` from CLI to move it.[/dim]"
                    )
                chat.add_info(output)
                return True

            if subcmd == "test":
                if not rest:
                    chat.add_info(self._render_slash_command_usage("/mcp test", "<alias>"))
                    return True
                try:
                    alias = ensure_valid_alias(rest)
                    view, tools = await asyncio.to_thread(
                        manager.probe_server,
                        alias,
                    )
                except Exception as e:
                    chat.add_info(
                        f"[bold #f7768e]MCP probe failed for '{rest}': {e}[/]"
                    )
                    return True
                names = [str(tool.get("name", "")) for tool in tools]
                lines = [
                    f"MCP probe succeeded for [bold]{view.alias}[/bold].",
                    f"Tools discovered: {len(names)}",
                ]
                for name in names:
                    lines.append(f"  - {name}")
                chat.add_info("\n".join(lines))
                return True

            if subcmd == "add":
                if not rest:
                    chat.add_info(
                        self._render_slash_command_usage(
                            "/mcp add",
                            (
                                "<alias> --command <cmd> [--arg <value>] "
                                "[--env KEY=VALUE] [--env-ref KEY=ENV] "
                                "[--cwd <path>] [--timeout <seconds>] [--disabled]"
                            ),
                        )
                    )
                    return True
                try:
                    tokens = self._split_slash_args(rest)
                except ValueError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                if not tokens:
                    chat.add_info(
                        self._render_slash_command_usage(
                            "/mcp add",
                            "<alias> --command <cmd> [...]",
                        )
                    )
                    return True
                alias_token = tokens[0]
                args_tokens = tokens[1:]
                command = ""
                cmd_args: list[str] = []
                env_pairs: list[str] = []
                env_refs: list[str] = []
                cwd = ""
                timeout = 30
                disabled = False
                index = 0
                while index < len(args_tokens):
                    item = args_tokens[index]
                    if item == "--disabled":
                        disabled = True
                        index += 1
                        continue
                    if item in {
                        "--command",
                        "--arg",
                        "--env",
                        "--env-ref",
                        "--cwd",
                        "--timeout",
                    }:
                        if index + 1 >= len(args_tokens):
                            chat.add_info(
                                f"[bold #f7768e]Missing value for {item}.[/]"
                            )
                            return True
                        value = args_tokens[index + 1]
                        if item == "--command":
                            command = value
                        elif item == "--arg":
                            cmd_args.append(value)
                        elif item == "--env":
                            env_pairs.append(value)
                        elif item == "--env-ref":
                            env_refs.append(value)
                        elif item == "--cwd":
                            cwd = value
                        elif item == "--timeout":
                            try:
                                timeout = int(value)
                            except ValueError:
                                chat.add_info(
                                    "[bold #f7768e]--timeout must be an integer.[/]"
                                )
                                return True
                        index += 2
                        continue
                    chat.add_info(
                        f"[bold #f7768e]Unknown /mcp add option: {item}[/]"
                    )
                    return True
                try:
                    alias = ensure_valid_alias(alias_token)
                    server = parse_mcp_server_from_flags(
                        command=command,
                        args=tuple(cmd_args),
                        env_pairs=tuple(env_pairs),
                        env_refs=tuple(env_refs),
                        cwd=cwd,
                        timeout=timeout,
                        disabled=disabled,
                    )
                    await asyncio.to_thread(manager.add_server, alias, server)
                    await self._reload_mcp_runtime()
                    chat.add_info(f"MCP server '{alias}' added.")
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                return True

            if subcmd == "edit":
                if not rest:
                    chat.add_info(
                        self._render_slash_command_usage(
                            "/mcp edit",
                            (
                                "<alias> [--command <cmd>] [--arg <value>] "
                                "[--env KEY=VALUE] [--env-ref KEY=ENV] "
                                "[--cwd <path>] [--timeout <seconds>] "
                                "[--enable|--disabled]"
                            ),
                        )
                    )
                    return True
                try:
                    tokens = self._split_slash_args(rest)
                except ValueError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                if not tokens:
                    chat.add_info(
                        self._render_slash_command_usage("/mcp edit", "<alias> [...]")
                    )
                    return True
                alias_token = tokens[0]
                args_tokens = tokens[1:]
                command: str | None = None
                cmd_args: list[str] = []
                env_pairs: list[str] = []
                env_refs: list[str] = []
                cwd: str | None = None
                timeout: int | None = None
                enabled_toggle: bool | None = None
                index = 0
                while index < len(args_tokens):
                    item = args_tokens[index]
                    if item == "--enable":
                        enabled_toggle = True
                        index += 1
                        continue
                    if item == "--disabled":
                        enabled_toggle = False
                        index += 1
                        continue
                    if item in {
                        "--command",
                        "--arg",
                        "--env",
                        "--env-ref",
                        "--cwd",
                        "--timeout",
                    }:
                        if index + 1 >= len(args_tokens):
                            chat.add_info(
                                f"[bold #f7768e]Missing value for {item}.[/]"
                            )
                            return True
                        value = args_tokens[index + 1]
                        if item == "--command":
                            command = value
                        elif item == "--arg":
                            cmd_args.append(value)
                        elif item == "--env":
                            env_pairs.append(value)
                        elif item == "--env-ref":
                            env_refs.append(value)
                        elif item == "--cwd":
                            cwd = value
                        elif item == "--timeout":
                            try:
                                timeout = int(value)
                            except ValueError:
                                chat.add_info(
                                    "[bold #f7768e]--timeout must be an integer.[/]"
                                )
                                return True
                        index += 2
                        continue
                    chat.add_info(
                        f"[bold #f7768e]Unknown /mcp edit option: {item}[/]"
                    )
                    return True

                if (
                    command is None
                    and not cmd_args
                    and not env_pairs
                    and not env_refs
                    and cwd is None
                    and timeout is None
                    and enabled_toggle is None
                ):
                    chat.add_info(
                        "[bold #f7768e]/mcp edit requires at least one change flag.[/]"
                    )
                    return True

                try:
                    alias = ensure_valid_alias(alias_token)

                    def _mutate(current):
                        merged = merge_server_edits(
                            current=current,
                            command=command,
                            args=tuple(cmd_args),
                            env_pairs=tuple(env_pairs),
                            env_refs=tuple(env_refs),
                            cwd=cwd,
                            timeout=timeout,
                            disabled=(enabled_toggle is False),
                        )
                        if enabled_toggle is True:
                            return replace(merged, enabled=True)
                        return merged

                    await asyncio.to_thread(manager.edit_server, alias, _mutate)
                    await self._reload_mcp_runtime()
                    chat.add_info(f"MCP server '{alias}' updated.")
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                return True

            if subcmd in {"enable", "disable"}:
                if not rest:
                    chat.add_info(self._render_slash_command_usage(f"/mcp {subcmd}", "<alias>"))
                    return True
                try:
                    alias = ensure_valid_alias(rest)
                    enabled = subcmd == "enable"
                    await asyncio.to_thread(
                        manager.edit_server,
                        alias,
                        lambda current: replace(current, enabled=enabled),
                    )
                    await self._reload_mcp_runtime()
                    chat.add_info(
                        f"MCP server '{alias}' "
                        f"{'enabled' if enabled else 'disabled'}."
                    )
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                return True

            if subcmd == "remove":
                if not rest:
                    chat.add_info(self._render_slash_command_usage("/mcp remove", "<alias>"))
                    return True
                try:
                    alias = ensure_valid_alias(rest)
                    from loom.auth.resources import (
                        cleanup_deleted_resource,
                        resource_delete_impact,
                    )

                    impact = await asyncio.to_thread(
                        resource_delete_impact,
                        workspace=self._workspace,
                        resource_kind="mcp",
                        resource_key=alias,
                    )
                    await asyncio.to_thread(manager.remove_server, alias)
                    await asyncio.to_thread(
                        cleanup_deleted_resource,
                        workspace=self._workspace,
                        explicit_auth_path=self._explicit_auth_path,
                        resource_kind="mcp",
                        resource_key=alias,
                    )
                    await self._reload_mcp_runtime()
                    impact_text = ""
                    if impact.resource_id:
                        impact_text = (
                            " "
                            f"(auth cleanup: {len(impact.active_profile_ids)} profile(s), "
                            f"{len(impact.active_binding_ids)} binding(s), "
                            f"default={'yes' if impact.workspace_default_profile_id else 'no'})"
                        )
                    chat.add_info(f"MCP server '{alias}' removed.{impact_text}")
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                return True

            chat.add_info(
                self._render_slash_command_usage(
                    "/mcp",
                    (
                        "[manage|list|show <alias>|test <alias>|add <alias> ...|"
                        "edit <alias> ...|enable <alias>|disable <alias>|remove <alias>]"
                    ),
                )
            )
            return True
        if token == "/auth":
            from loom.auth.config import (
                AuthConfigError,
                AuthProfile,
                load_merged_auth_config,
                remove_auth_profile,
                resolve_auth_write_path,
                set_workspace_auth_default,
                upsert_auth_profile,
            )
            from loom.auth.resources import (
                bind_resource_to_profile,
                default_workspace_auth_resources_path,
                has_active_binding,
                load_workspace_auth_resources,
                resolve_resource,
                set_workspace_resource_default,
            )
            from loom.auth.runtime import AuthResolutionError, parse_auth_profile_overrides

            if not arg:
                self._open_auth_manager_screen()
                return True

            subparts = arg.split(None, 1)
            subcmd = subparts[0].lower()
            rest = subparts[1].strip() if len(subparts) > 1 else ""

            if subcmd == "manage":
                self._open_auth_manager_screen()
                return True

            def _effective_defaults(merged_auth) -> dict[str, str]:
                defaults = dict(merged_auth.config.defaults)
                defaults.update(merged_auth.workspace_defaults)
                return defaults

            if subcmd == "list":
                try:
                    merged_auth = await asyncio.to_thread(
                        load_merged_auth_config,
                        workspace=self._workspace,
                        explicit_path=self._explicit_auth_path,
                    )
                except AuthConfigError as e:
                    chat.add_info(f"[bold #f7768e]Auth config error: {e}[/]")
                    return True

                lines = [
                    "[bold #7dcfff]Auth Profiles[/bold #7dcfff]",
                    f"  user: {merged_auth.user_path}",
                    f"  explicit: {merged_auth.explicit_path or '-'}",
                    f"  workspace defaults: {merged_auth.workspace_defaults_path or '-'}",
                ]
                if not merged_auth.config.profiles:
                    lines.append("  (none)")
                else:
                    for profile_id in sorted(merged_auth.config.profiles):
                        profile = merged_auth.config.profiles[profile_id]
                        label = f" label={profile.account_label}" if profile.account_label else ""
                        lines.append(
                            f"  {profile.profile_id}: provider={profile.provider} "
                            f"mode={profile.mode}{label}"
                        )
                        if profile.mcp_server:
                            lines[-1] = lines[-1] + f" mcp_server={profile.mcp_server}"

                defaults = {
                    selector: profile_id
                    for selector, profile_id in _effective_defaults(merged_auth).items()
                    if not selector.startswith("mcp.")
                }
                if defaults:
                    lines.append("")
                    lines.append("Defaults:")
                    for selector, profile_id in sorted(defaults.items()):
                        lines.append(f"  {selector} -> {profile_id}")

                run_overrides = {
                    selector: profile_id
                    for selector, profile_id in self._run_auth_profile_overrides.items()
                    if not selector.startswith("mcp.")
                }
                if run_overrides:
                    lines.append("")
                    lines.append("Run overrides:")
                    for selector, profile_id in sorted(run_overrides.items()):
                        lines.append(f"  {selector} -> {profile_id}")

                chat.add_info("\n".join(lines))
                return True

            if subcmd == "show":
                if not rest:
                    chat.add_info(
                        self._render_slash_command_usage("/auth show", "<profile-id>")
                    )
                    return True
                try:
                    merged_auth = await asyncio.to_thread(
                        load_merged_auth_config,
                        workspace=self._workspace,
                        explicit_path=self._explicit_auth_path,
                    )
                except AuthConfigError as e:
                    chat.add_info(f"[bold #f7768e]Auth config error: {e}[/]")
                    return True
                profile = merged_auth.config.profiles.get(rest)
                if profile is None:
                    chat.add_info(f"[bold #f7768e]Auth profile not found: {rest}[/]")
                    return True
                env_keys = ", ".join(sorted(profile.env.keys())) or "-"
                lines = [
                    f"[bold #7dcfff]Profile[/bold #7dcfff] {profile.profile_id}",
                    f"  provider: {profile.provider}",
                    f"  mode: {profile.mode}",
                    f"  label: {profile.account_label or '-'}",
                    f"  mcp_server: {profile.mcp_server or '-'}",
                    f"  secret_ref: {profile.secret_ref or '-'}",
                    f"  token_ref: {profile.token_ref or '-'}",
                    f"  env_keys: {env_keys}",
                    f"  command: {profile.command or '-'}",
                ]
                if profile.scopes:
                    lines.append(f"  scopes: {', '.join(profile.scopes)}")
                chat.add_info("\n".join(lines))
                return True

            if subcmd == "check":
                try:
                    merged_auth = await asyncio.to_thread(
                        load_merged_auth_config,
                        workspace=self._workspace,
                        explicit_path=self._explicit_auth_path,
                    )
                except AuthConfigError as e:
                    chat.add_info(f"[bold #f7768e]Auth config error: {e}[/]")
                    return True

                profiles = merged_auth.config.profiles
                defaults = _effective_defaults(merged_auth)
                errors: list[str] = []

                for selector, profile_id in sorted(defaults.items()):
                    if profile_id not in profiles:
                        errors.append(
                            f"default {selector!r} references unknown profile {profile_id!r}"
                        )
                        continue
                    profile = profiles[profile_id]
                    if selector.startswith("mcp."):
                        continue
                    if selector != profile.provider:
                        errors.append(
                            f"default {selector!r} must match provider {profile.provider!r}"
                        )
                mcp_aliases = set(self._config.mcp.servers.keys())
                for profile in profiles.values():
                    mcp_server = str(getattr(profile, "mcp_server", "") or "").strip()
                    if not mcp_server:
                        continue
                    if mcp_server not in mcp_aliases:
                        errors.append(
                            f"profile {profile.profile_id!r} references unknown "
                            f"mcp_server {mcp_server!r}"
                        )
                for selector, profile_id in sorted(self._run_auth_profile_overrides.items()):
                    profile = profiles.get(profile_id)
                    if profile is None:
                        errors.append(
                            f"run override {selector!r} references unknown profile {profile_id!r}"
                        )
                        continue
                    if selector.startswith("mcp."):
                        errors.append(
                            f"run override {selector!r} is no longer supported; "
                            "MCP account selection is managed via MCP aliases."
                        )
                        continue
                    if selector != profile.provider:
                        errors.append(
                            f"run override {selector!r} must match provider {profile.provider!r}"
                        )

                if errors:
                    chat.add_info(
                        "[bold #f7768e]Auth config validation failed:[/]\n"
                        + "\n".join(f"  - {err}" for err in errors)
                    )
                    return True
                chat.add_info("Auth config is valid.")
                return True

            if subcmd == "use":
                if not rest:
                    chat.add_info(
                        self._render_slash_command_usage("/auth use", "<selector=profile>")
                    )
                    return True
                try:
                    parsed = parse_auth_profile_overrides((rest,))
                except AuthResolutionError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                selector, profile_id = next(iter(parsed.items()))
                try:
                    merged_auth = await asyncio.to_thread(
                        load_merged_auth_config,
                        workspace=self._workspace,
                        explicit_path=self._explicit_auth_path,
                    )
                except AuthConfigError as e:
                    chat.add_info(f"[bold #f7768e]Auth config error: {e}[/]")
                    return True
                profile = merged_auth.config.profiles.get(profile_id)
                if profile is None:
                    chat.add_info(
                        f"[bold #f7768e]Unknown auth profile: {profile_id}[/]"
                    )
                    return True
                if selector.startswith("mcp."):
                    chat.add_info(
                        "[bold #f7768e]MCP selectors are no longer supported in /auth use. "
                        "Select MCP accounts via MCP aliases in /mcp.[/]"
                    )
                    return True
                resource_store = await asyncio.to_thread(
                    load_workspace_auth_resources,
                    default_workspace_auth_resources_path(self._workspace.resolve()),
                )
                resolved_resource = None
                if ":" in selector:
                    resolved_resource = resolve_resource(
                        resource_store,
                        resource_ref=selector,
                    )
                elif selector in resource_store.resources:
                    resolved_resource = resolve_resource(
                        resource_store,
                        resource_id=selector,
                    )
                if resolved_resource is None and selector != profile.provider:
                    chat.add_info(
                        "[bold #f7768e]Selector must match profile provider "
                        "or a known resource selector.[/]"
                    )
                    return True
                if (
                    resolved_resource is not None
                    and profile.provider != resolved_resource.provider
                ):
                    chat.add_info(
                        "[bold #f7768e]Profile provider does not match resource provider: "
                        f"{profile.provider!r} != {resolved_resource.provider!r}.[/]"
                    )
                    return True
                self._run_auth_profile_overrides[selector] = profile_id
                chat.add_info(f"Run auth override set: {selector} -> {profile_id}")
                return True

            if subcmd == "clear":
                if rest:
                    removed = self._run_auth_profile_overrides.pop(rest, None)
                    if removed is None:
                        chat.add_info(f"No run auth override set for {rest}.")
                        return True
                    chat.add_info(f"Removed run auth override: {rest}")
                    return True
                self._run_auth_profile_overrides.clear()
                chat.add_info("Cleared all run auth overrides.")
                return True

            if subcmd == "select":
                if not rest:
                    chat.add_info(
                        self._render_slash_command_usage("/auth select", "<selector=profile>")
                    )
                    return True
                try:
                    parsed = parse_auth_profile_overrides((rest,))
                except AuthResolutionError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                selector, profile_id = next(iter(parsed.items()))
                try:
                    merged_auth = await asyncio.to_thread(
                        load_merged_auth_config,
                        workspace=self._workspace,
                        explicit_path=self._explicit_auth_path,
                    )
                except AuthConfigError as e:
                    chat.add_info(f"[bold #f7768e]Auth config error: {e}[/]")
                    return True
                profile = merged_auth.config.profiles.get(profile_id)
                if profile is None:
                    chat.add_info(
                        f"[bold #f7768e]Unknown auth profile: {profile_id}[/]"
                    )
                    return True
                if selector.startswith("mcp."):
                    chat.add_info(
                        "[bold #f7768e]MCP selectors are no longer supported in /auth select. "
                        "Select MCP accounts via MCP aliases in /mcp.[/]"
                    )
                    return True
                resources_path = default_workspace_auth_resources_path(
                    self._workspace.resolve()
                )
                resource_store = await asyncio.to_thread(
                    load_workspace_auth_resources,
                    resources_path,
                )
                resolved_resource = None
                if ":" in selector:
                    resolved_resource = resolve_resource(
                        resource_store,
                        resource_ref=selector,
                    )
                elif selector in resource_store.resources:
                    resolved_resource = resolve_resource(
                        resource_store,
                        resource_id=selector,
                    )

                if resolved_resource is not None:
                    if profile.provider != resolved_resource.provider:
                        chat.add_info(
                            "[bold #f7768e]Profile provider does not match resource provider: "
                            f"{profile.provider!r} != {resolved_resource.provider!r}.[/]"
                        )
                        return True
                    try:
                        if not has_active_binding(
                            resource_store,
                            resource_id=resolved_resource.resource_id,
                            profile_id=profile_id,
                        ):
                            await asyncio.to_thread(
                                bind_resource_to_profile,
                                resources_path,
                                resource_id=resolved_resource.resource_id,
                                profile_id=profile_id,
                                generated_from=f"slash:auth-select:{selector}",
                                priority=0,
                            )
                        await asyncio.to_thread(
                            set_workspace_resource_default,
                            resources_path,
                            resource_id=resolved_resource.resource_id,
                            profile_id=profile_id,
                        )
                    except Exception as e:
                        chat.add_info(f"[bold #f7768e]{e}[/]")
                        return True
                    chat.add_info(
                        "Workspace resource default set: "
                        f"{resolved_resource.resource_ref} -> {profile_id}\n"
                        f"[dim]{resources_path}[/dim]"
                    )
                    return True

                if selector != profile.provider:
                    chat.add_info(
                        "[bold #f7768e]Selector must match profile provider "
                        "or a known resource selector.[/]"
                    )
                    return True

                defaults_path = self._auth_defaults_path()
                try:
                    await asyncio.to_thread(
                        set_workspace_auth_default,
                        defaults_path,
                        selector=selector,
                        profile_id=profile_id,
                    )
                except AuthConfigError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                chat.add_info(
                    f"Workspace auth default set: {selector} -> {profile_id}\n"
                    f"[dim]{defaults_path}[/dim]"
                )
                return True

            if subcmd == "unset":
                clean_selector = rest.strip()
                if not clean_selector:
                    chat.add_info(
                        self._render_slash_command_usage("/auth unset", "<selector>")
                    )
                    return True
                resources_path = default_workspace_auth_resources_path(
                    self._workspace.resolve()
                )
                resource_store = await asyncio.to_thread(
                    load_workspace_auth_resources,
                    resources_path,
                )
                resolved_resource = None
                if ":" in clean_selector:
                    resolved_resource = resolve_resource(
                        resource_store,
                        resource_ref=clean_selector,
                    )
                elif clean_selector in resource_store.resources:
                    resolved_resource = resolve_resource(
                        resource_store,
                        resource_id=clean_selector,
                    )
                if resolved_resource is not None:
                    try:
                        await asyncio.to_thread(
                            set_workspace_resource_default,
                            resources_path,
                            resource_id=resolved_resource.resource_id,
                            profile_id=None,
                        )
                    except Exception as e:
                        chat.add_info(f"[bold #f7768e]{e}[/]")
                        return True
                    chat.add_info(
                        "Workspace resource default removed: "
                        f"{resolved_resource.resource_ref}\n"
                        f"[dim]{resources_path}[/dim]"
                    )
                    return True
                defaults_path = self._auth_defaults_path()
                try:
                    await asyncio.to_thread(
                        set_workspace_auth_default,
                        defaults_path,
                        selector=clean_selector,
                        profile_id=None,
                    )
                except AuthConfigError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                chat.add_info(
                    f"Workspace auth default removed: {clean_selector}\n"
                    f"[dim]{defaults_path}[/dim]"
                )
                return True

            if subcmd == "add":
                if not rest:
                    chat.add_info(
                        self._render_slash_command_usage(
                            "/auth add",
                            (
                                "<profile-id> --provider <provider> --mode <mode> "
                                "[--label <text>] [--secret-ref <ref>] "
                                "[--mcp-server <alias>] "
                                "[--token-ref <ref>] [--scope <scope>] "
                                "[--env KEY=VALUE] [--command <cmd>] "
                                "[--auth-check <token>] [--meta KEY=VALUE]"
                            ),
                        )
                    )
                    return True
                try:
                    tokens = self._split_slash_args(rest)
                except ValueError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                if not tokens:
                    chat.add_info(
                        self._render_slash_command_usage("/auth add", "<profile-id> ...")
                    )
                    return True
                profile_id = str(tokens[0]).strip()
                args_tokens = tokens[1:]
                provider = ""
                mode = ""
                label = ""
                mcp_server = ""
                secret_ref = ""
                token_ref = ""
                scopes: list[str] = []
                env_values: list[str] = []
                command = ""
                auth_check: list[str] = []
                meta_values: list[str] = []
                index = 0
                while index < len(args_tokens):
                    item = args_tokens[index]
                    if item in {
                        "--provider",
                        "--mode",
                        "--label",
                        "--mcp-server",
                        "--secret-ref",
                        "--token-ref",
                        "--scope",
                        "--env",
                        "--command",
                        "--auth-check",
                        "--meta",
                    }:
                        if index + 1 >= len(args_tokens):
                            chat.add_info(
                                f"[bold #f7768e]Missing value for {item}.[/]"
                            )
                            return True
                        value = args_tokens[index + 1]
                        if item == "--provider":
                            provider = value
                        elif item == "--mode":
                            mode = value
                        elif item == "--label":
                            label = value
                        elif item == "--mcp-server":
                            mcp_server = value
                        elif item == "--secret-ref":
                            secret_ref = value
                        elif item == "--token-ref":
                            token_ref = value
                        elif item == "--scope":
                            scopes.append(value)
                        elif item == "--env":
                            env_values.append(value)
                        elif item == "--command":
                            command = value
                        elif item == "--auth-check":
                            auth_check.append(value)
                        elif item == "--meta":
                            meta_values.append(value)
                        index += 2
                        continue
                    chat.add_info(
                        f"[bold #f7768e]Unknown /auth add option: {item}[/]"
                    )
                    return True
                if not profile_id:
                    chat.add_info("[bold #f7768e]Profile id cannot be empty.[/]")
                    return True
                if not provider or not mode:
                    chat.add_info(
                        "[bold #f7768e]/auth add requires --provider and --mode.[/]"
                    )
                    return True
                try:
                    env = self._parse_kv_assignments(
                        env_values,
                        option_name="--env",
                        env_keys=True,
                    )
                    metadata = self._parse_kv_assignments(
                        meta_values,
                        option_name="--meta",
                    )
                    profile = AuthProfile(
                        profile_id=profile_id,
                        provider=provider.strip(),
                        mode=mode.strip(),
                        account_label=label.strip(),
                        mcp_server=mcp_server.strip(),
                        secret_ref=secret_ref.strip(),
                        token_ref=token_ref.strip(),
                        scopes=[str(scope).strip() for scope in scopes if str(scope).strip()],
                        env=env,
                        command=command.strip(),
                        auth_check=[
                            str(item).strip()
                            for item in auth_check
                            if str(item).strip()
                        ],
                        metadata=metadata,
                    )
                    target = resolve_auth_write_path(
                        explicit_path=self._explicit_auth_path,
                    )
                    await asyncio.to_thread(
                        upsert_auth_profile,
                        target,
                        profile,
                        must_exist=False,
                    )
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                chat.add_info(
                    f"Added auth profile '{profile_id}'.\n[dim]{target}[/dim]"
                )
                return True

            if subcmd == "edit":
                if not rest:
                    chat.add_info(
                        self._render_slash_command_usage(
                            "/auth edit",
                            (
                                "<profile-id> [--provider <provider>] [--mode <mode>] "
                                "[--label <text>] [--mcp-server <alias>] "
                                "[--clear-mcp-server] "
                                "[--secret-ref <ref>] [--token-ref <ref>] "
                                "[--scope <scope>] [--clear-scopes] "
                                "[--env KEY=VALUE] [--clear-env] "
                                "[--command <cmd>] [--auth-check <token>] "
                                "[--clear-auth-check] [--meta KEY=VALUE] [--clear-meta]"
                            ),
                        )
                    )
                    return True
                try:
                    tokens = self._split_slash_args(rest)
                except ValueError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                if not tokens:
                    chat.add_info(
                        self._render_slash_command_usage("/auth edit", "<profile-id> ...")
                    )
                    return True
                profile_id = str(tokens[0]).strip()
                args_tokens = tokens[1:]
                provider: str | None = None
                mode: str | None = None
                label: str | None = None
                mcp_server: str | None = None
                clear_mcp_server = False
                secret_ref: str | None = None
                token_ref: str | None = None
                scopes: list[str] = []
                clear_scopes = False
                env_values: list[str] = []
                clear_env = False
                command: str | None = None
                auth_check_values: list[str] = []
                clear_auth_check = False
                meta_values: list[str] = []
                clear_meta = False
                index = 0
                while index < len(args_tokens):
                    item = args_tokens[index]
                    if item in {
                        "--clear-scopes",
                        "--clear-mcp-server",
                        "--clear-env",
                        "--clear-auth-check",
                        "--clear-meta",
                    }:
                        if item == "--clear-scopes":
                            clear_scopes = True
                        elif item == "--clear-mcp-server":
                            clear_mcp_server = True
                        elif item == "--clear-env":
                            clear_env = True
                        elif item == "--clear-auth-check":
                            clear_auth_check = True
                        elif item == "--clear-meta":
                            clear_meta = True
                        index += 1
                        continue
                    if item in {
                        "--provider",
                        "--mode",
                        "--label",
                        "--mcp-server",
                        "--secret-ref",
                        "--token-ref",
                        "--scope",
                        "--env",
                        "--command",
                        "--auth-check",
                        "--meta",
                    }:
                        if index + 1 >= len(args_tokens):
                            chat.add_info(
                                f"[bold #f7768e]Missing value for {item}.[/]"
                            )
                            return True
                        value = args_tokens[index + 1]
                        if item == "--provider":
                            provider = value
                        elif item == "--mode":
                            mode = value
                        elif item == "--label":
                            label = value
                        elif item == "--mcp-server":
                            mcp_server = value
                        elif item == "--secret-ref":
                            secret_ref = value
                        elif item == "--token-ref":
                            token_ref = value
                        elif item == "--scope":
                            scopes.append(value)
                        elif item == "--env":
                            env_values.append(value)
                        elif item == "--command":
                            command = value
                        elif item == "--auth-check":
                            auth_check_values.append(value)
                        elif item == "--meta":
                            meta_values.append(value)
                        index += 2
                        continue
                    chat.add_info(
                        f"[bold #f7768e]Unknown /auth edit option: {item}[/]"
                    )
                    return True
                if not profile_id:
                    chat.add_info("[bold #f7768e]Profile id cannot be empty.[/]")
                    return True
                if (
                    provider is None
                    and mode is None
                    and label is None
                    and mcp_server is None
                    and not clear_mcp_server
                    and secret_ref is None
                    and token_ref is None
                    and not scopes
                    and not clear_scopes
                    and not env_values
                    and not clear_env
                    and command is None
                    and not auth_check_values
                    and not clear_auth_check
                    and not meta_values
                    and not clear_meta
                ):
                    chat.add_info(
                        "[bold #f7768e]/auth edit requires at least one change flag.[/]"
                    )
                    return True
                try:
                    merged_auth = await asyncio.to_thread(
                        load_merged_auth_config,
                        workspace=self._workspace,
                        explicit_path=self._explicit_auth_path,
                    )
                    current = merged_auth.config.profiles.get(profile_id)
                    if current is None:
                        chat.add_info(
                            f"[bold #f7768e]Auth profile not found: {profile_id}[/]"
                        )
                        return True
                    env_updates = self._parse_kv_assignments(
                        env_values,
                        option_name="--env",
                        env_keys=True,
                    )
                    meta_updates = self._parse_kv_assignments(
                        meta_values,
                        option_name="--meta",
                    )
                    next_scopes = [] if clear_scopes else list(current.scopes)
                    if scopes:
                        next_scopes = [
                            str(scope).strip()
                            for scope in scopes
                            if str(scope).strip()
                        ]
                    next_mcp_server = (
                        ""
                        if clear_mcp_server
                        else current.mcp_server
                    )
                    if mcp_server is not None:
                        next_mcp_server = mcp_server.strip()
                    next_env = {} if clear_env else dict(current.env)
                    next_env.update(env_updates)
                    next_auth_check = (
                        []
                        if clear_auth_check
                        else list(current.auth_check)
                    )
                    if auth_check_values:
                        next_auth_check = [
                            str(item).strip()
                            for item in auth_check_values
                            if str(item).strip()
                        ]
                    next_metadata = {} if clear_meta else dict(current.metadata)
                    next_metadata.update(meta_updates)
                    updated = AuthProfile(
                        profile_id=current.profile_id,
                        provider=current.provider if provider is None else provider.strip(),
                        mode=current.mode if mode is None else mode.strip(),
                        account_label=(
                            current.account_label
                            if label is None else label.strip()
                        ),
                        mcp_server=next_mcp_server,
                        secret_ref=(
                            current.secret_ref
                            if secret_ref is None else secret_ref.strip()
                        ),
                        token_ref=current.token_ref if token_ref is None else token_ref.strip(),
                        scopes=next_scopes,
                        env=next_env,
                        command=current.command if command is None else command.strip(),
                        auth_check=next_auth_check,
                        metadata=next_metadata,
                    )
                    if not updated.provider or not updated.mode:
                        chat.add_info(
                            "[bold #f7768e]Provider and mode must be non-empty.[/]"
                        )
                        return True
                    target = resolve_auth_write_path(
                        explicit_path=self._explicit_auth_path,
                    )
                    await asyncio.to_thread(
                        upsert_auth_profile,
                        target,
                        updated,
                        must_exist=True,
                    )
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                chat.add_info(
                    f"Updated auth profile '{profile_id}'.\n[dim]{target}[/dim]"
                )
                return True

            if subcmd == "remove":
                profile_id = rest.strip()
                if not profile_id:
                    chat.add_info(
                        self._render_slash_command_usage("/auth remove", "<profile-id>")
                    )
                    return True
                target = resolve_auth_write_path(
                    explicit_path=self._explicit_auth_path,
                )
                try:
                    await asyncio.to_thread(
                        remove_auth_profile,
                        target,
                        profile_id,
                    )
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                chat.add_info(
                    f"Removed auth profile '{profile_id}'.\n[dim]{target}[/dim]"
                )
                return True

            chat.add_info(
                self._render_slash_command_usage(
                    "/auth",
                    (
                        "[manage|list|show <profile-id>|check|use <selector=profile>|"
                        "clear [selector]|select <selector=profile>|unset <selector>|"
                        "add <profile-id> ...|edit <profile-id> ...|remove <profile-id>]"
                    ),
                )
            )
            return True
        if token == "/setup":
            self.push_screen(
                SetupScreen(), callback=self._on_setup_complete,
            )
            return True
        if token == "/tools":
            chat.add_info(self._render_tools_catalog())
            return True
        if token == "/tokens":
            chat.add_info(f"Session tokens: {self._total_tokens:,}")
            return True
        if token in {"/process", "/processes"}:
            self._refresh_process_command_index()
            if token == "/processes":
                if arg:
                    chat.add_info(self._render_slash_command_usage("/processes", ""))
                    return True
                chat.add_info(self._render_process_catalog())
                return True
            if not arg:
                chat.add_info(self._render_process_catalog())
                return True

            subparts = arg.split(None, 1)
            subcmd = subparts[0].lower()
            rest = subparts[1].strip() if len(subparts) > 1 else ""

            if subcmd == "list":
                chat.add_info(self._render_process_catalog())
                return True

            if subcmd == "use":
                if not rest:
                    chat.add_info(
                        self._render_slash_command_usage(
                            "/process use",
                            "<name-or-path>",
                        )
                    )
                    return True
                loader = self._create_process_loader()
                try:
                    loaded = loader.load(rest)
                except Exception as e:
                    chat.add_info(
                        f"[bold #f7768e]Failed to load process "
                        f"'{rest}': {e}[/]"
                    )
                    return True
                if self._is_reserved_process_name(loaded.name):
                    chat.add_info(
                        f"[bold #f7768e]Process '{loaded.name}' conflicts with a "
                        "built-in slash command and cannot be loaded in TUI.[/]"
                    )
                    return True

                self._process_name = loaded.name
                await self._reload_session_for_process_change(chat)
                self._refresh_process_command_index(chat=chat, notify_conflicts=True)
                chat.add_info(
                    "[bold #7dcfff]Active Process Updated[/bold #7dcfff]\n"
                    f"  [bold]Name:[/] [bold]{self._escape_markup(loaded.name)}[/bold]\n"
                    f"  [bold]Version:[/] [dim]v{self._escape_markup(loaded.version)}[/dim]"
                )
                return True

            if subcmd in {"off", "none", "clear"}:
                if not self._process_name and self._process_defn is None:
                    chat.add_info("No active process.")
                    return True
                self._process_name = None
                self._process_defn = None
                await self._reload_session_for_process_change(chat)
                chat.add_info(
                    "[bold #7dcfff]Active Process Updated[/bold #7dcfff]\n"
                    "  [bold]Name:[/] none"
                )
                return True

            chat.add_info(
                self._render_slash_command_usage(
                    "/process",
                    "[list|use <name-or-path>|off]",
                )
            )
            return True

        if token == "/run":
            if arg:
                try:
                    tokens = self._split_slash_args(arg)
                except ValueError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                if tokens:
                    subcmd = tokens[0].lower()
                    if subcmd == "close":
                        target = " ".join(tokens[1:]).strip()
                        await self._close_process_run_from_target(target)
                        return True
                    if subcmd == "resume":
                        if len(tokens) < 2:
                            chat.add_info(
                                self._render_slash_command_usage(
                                    "/run resume",
                                    "<run-id-prefix|current>",
                                )
                            )
                            return True
                        target = tokens[1]
                        await self._resume_process_run_from_target(target)
                        return True
                    if subcmd == "save":
                        if len(tokens) < 3:
                            chat.add_info(
                                self._render_slash_command_usage(
                                    "/run save",
                                    "<run-id-prefix|current> <name>",
                                )
                            )
                            return True
                        target = tokens[1]
                        package_name = tokens[2]
                        run, error = self._resolve_process_run_target(target)
                        if run is None:
                            chat.add_info(error or "Run not found.")
                            return True
                        if run.process_defn is None or not bool(
                            getattr(run, "is_adhoc", False),
                        ):
                            chat.add_info(
                                "[bold #f7768e]Only ad hoc /run processes can be saved.[/]"
                            )
                            return True
                        try:
                            saved_dir = await asyncio.to_thread(
                                self._save_adhoc_process_package,
                                process_defn=run.process_defn,
                                package_name=package_name,
                                recommended_tools=run.recommended_tools,
                            )
                        except Exception as e:
                            chat.add_info(
                                f"[bold #f7768e]Failed to save process package: {e}[/]"
                            )
                            return True
                        self._refresh_process_command_index()
                        safe_name = self._sanitize_kebab_token(
                            package_name,
                            fallback="adhoc-process",
                            max_len=40,
                        )
                        chat.add_info(
                            "Saved ad hoc process package:\n"
                            f"  [bold]{safe_name}[/bold]\n"
                            f"  [dim]{saved_dir}[/dim]\n"
                            f"Run it with [bold]/{safe_name} <goal>[/bold]."
                        )
                        return True

            run_process_name = ""
            goal = ""
            goal_tokens: list[str] = []
            force_fresh = False
            if arg:
                try:
                    tokens = self._split_slash_args(arg)
                except ValueError as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                    return True
                idx = 0
                while idx < len(tokens):
                    item = str(tokens[idx] or "").strip()
                    if item in {"--fresh", "-f"}:
                        force_fresh = True
                        idx += 1
                        continue
                    if item in {"--process", "-p"}:
                        if idx + 1 >= len(tokens):
                            chat.add_info(
                                self._render_slash_command_usage(
                                    "/run",
                                    "--process <name> <goal>",
                                )
                            )
                            return True
                        run_process_name = tokens[idx + 1].strip()
                        idx += 2
                        continue
                    if item.startswith("-"):
                        chat.add_info(
                            self._render_slash_command_usage(
                                "/run",
                                "[--fresh] [--process <name>] <goal>",
                            )
                        )
                        return True
                    break
                goal_tokens = [
                    str(item or "").strip()
                    for item in tokens[idx:]
                    if str(item or "").strip()
                ]
                goal = " ".join(goal_tokens).strip()
            else:
                goal = ""
            if not goal:
                chat.add_info(self._render_slash_command_usage("/run", "<goal>"))
                return True

            execution_goal = goal
            synthesis_goal = goal
            goal_context_overrides: dict[str, Any] = {}
            if goal_tokens:
                (
                    execution_goal,
                    synthesis_goal,
                    goal_context_overrides,
                    file_goal_error,
                ) = self._expand_run_goal_file_input(goal_tokens)
                if file_goal_error:
                    chat.add_info(
                        f"[bold #f7768e]{self._escape_markup(file_goal_error)}[/]"
                    )
                    return True
                file_context = goal_context_overrides.get("run_goal_file_input", {})
                if isinstance(file_context, dict) and file_context:
                    file_label = self._escape_markup(str(file_context.get("path", "")).strip())
                    truncated = bool(file_context.get("truncated", False))
                    max_chars = int(
                        file_context.get("max_chars", _RUN_GOAL_FILE_CONTENT_MAX_CHARS),
                    )
                    trunc_note = (
                        f" (truncated to {max_chars:,} chars)"
                        if truncated
                        else ""
                    )
                    chat.add_info(
                        f"Loaded /run goal file [bold]{file_label}[/bold]{trunc_note}."
                    )

            process_defn = None
            recommended_tools: list[str] = []
            is_adhoc = False
            adhoc_synthesis_notes: list[str] = []

            if run_process_name:
                loader = self._create_process_loader()
                try:
                    process_defn = loader.load(run_process_name)
                except Exception as e:
                    chat.add_info(
                        f"[bold #f7768e]Failed to load process "
                        f"'{run_process_name}': {e}[/]"
                    )
                    return True
            elif self._process_defn is not None:
                process_defn = self._process_defn
            else:
                default_process = str(
                    getattr(getattr(self._config, "process", None), "default", "") or "",
                ).strip()
                if default_process:
                    loader = self._create_process_loader()
                    try:
                        process_defn = loader.load(default_process)
                    except Exception as e:
                        chat.add_info(
                            f"[bold #f7768e]Failed to load default process "
                            f"'{default_process}': {e}[/]"
                        )
                        return True
                else:
                    chat.add_info("Synthesizing ad hoc process for /run goal...")
                    entry, from_cache = await self._get_or_create_adhoc_process(
                        synthesis_goal,
                        fresh=force_fresh,
                    )
                    process_defn = entry.process_defn
                    recommended_tools = list(entry.recommended_tools)
                    is_adhoc = True
                    if from_cache:
                        chat.add_info(
                            f"Using cached ad hoc process [bold]{process_defn.name}[/bold]."
                        )
                    else:
                        chat.add_info(
                            f"Synthesized ad hoc process [bold]{process_defn.name}[/bold] "
                            f"with {len(process_defn.phases)} phases."
                        )
                    if recommended_tools:
                        chat.add_info(
                            "Recommended additional tools: "
                            + ", ".join(sorted(recommended_tools))
                        )
                    cache_key = str(
                        getattr(entry, "key", "") or self._adhoc_cache_key(synthesis_goal),
                    )
                    chat.add_info(
                        "Ad hoc process cache: "
                        f"[dim]{self._adhoc_cache_path(cache_key)}[/dim]"
                    )
                    adhoc_synthesis_notes = self._adhoc_synthesis_activity_lines(
                        entry,
                        from_cache=from_cache,
                        fresh=force_fresh,
                    )

            if process_defn is None:
                chat.add_info(
                    "[bold #f7768e]Unable to resolve process for /run.[/]"
                )
                return True
            start_kwargs: dict[str, Any] = {
                "process_defn": process_defn,
                "is_adhoc": is_adhoc,
                "recommended_tools": recommended_tools,
            }
            if adhoc_synthesis_notes:
                start_kwargs["adhoc_synthesis_notes"] = adhoc_synthesis_notes
            if goal_context_overrides:
                start_kwargs["goal_context_overrides"] = goal_context_overrides
            await self._start_process_run(execution_goal, **start_kwargs)
            return True

        process_name = self._process_command_map.get(token)
        if process_name:
            goal = self._strip_wrapping_quotes(arg)
            if not goal:
                chat.add_info(
                    self._render_slash_command_usage(f"/{process_name}", "<goal>")
                )
                return True
            loader = self._create_process_loader()
            try:
                process_defn = loader.load(process_name)
            except Exception as e:
                chat.add_info(
                    f"[bold #f7768e]Failed to load process "
                    f"'{process_name}': {e}[/]"
                )
                return True
            if self._is_reserved_process_name(process_defn.name):
                chat.add_info(
                    f"[bold #f7768e]Process '{process_defn.name}' conflicts with "
                    "a built-in slash command and cannot be run from TUI.[/]"
                )
                return True
            await self._start_process_run(
                goal,
                process_defn=process_defn,
                command_prefix=f"/{process_name}",
            )
            return True

        # Persistence-dependent commands
        if token == "/session":
            if not self._session:
                chat.add_info("No active session.")
                return True
            state = self._session.session_state
            chat.add_info(self._render_session_info(state))
            return True

        if token == "/new":
            if self._store:
                await self._new_session()
            else:
                chat.add_info("No database — sessions are ephemeral.")
            return True

        if token == "/sessions":
            if not self._store:
                chat.add_info("No database — sessions are ephemeral.")
                return True
            all_sessions = await self._store.list_sessions()
            if not all_sessions:
                chat.add_info("No previous sessions.")
                return True
            chat.add_info(self._render_sessions_list(all_sessions))
            return True

        if token == "/resume":
            if not arg:
                chat.add_info(
                    self._render_slash_command_usage("/resume", "<session-id-prefix>")
                )
                return True
            prefix = arg.lower()
            if not self._store:
                chat.add_info("No database — sessions are ephemeral.")
                return True
            all_sessions = await self._store.list_sessions()
            match = None
            for s in all_sessions:
                if s["id"].lower().startswith(prefix):
                    match = s
                    break
            if match:
                try:
                    await self._switch_to_session(match["id"])
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]Resume failed: {e}[/]")
            else:
                chat.add_info(f"No session found matching '{prefix}'.")
            return True

        if token == "/learned":
            if not self._db:
                chat.add_info("No database — learned patterns unavailable.")
                return True
            await self._show_learned_patterns()
            return True

        return False

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _run_turn(self, user_message: str) -> None:
        if self._session is None:
            return

        self._chat_busy = True
        chat = self.query_one("#chat-log", ChatLog)
        status = self.query_one("#status-bar", StatusBar)

        chat.add_user_message(user_message)
        status.state = "Thinking..."

        try:
            await self._run_interaction(user_message)
        except Exception as e:
            chat.add_model_text(f"[bold #f7768e]Error:[/] {e}", markup=True)
            self.notify(str(e), severity="error", timeout=5)

        self._chat_busy = False
        status.state = "Ready"

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

    async def _run_interaction(self, message: str) -> None:
        """Execute a turn interaction with the model.

        Shared implementation for both initial turns and follow-ups.
        """
        chat = self.query_one("#chat-log", ChatLog)
        status = self.query_one("#status-bar", StatusBar)
        events_panel = self.query_one("#events-panel", EventPanel)

        streamed_text = False

        async for event in self._session.send_streaming(message):
            if isinstance(event, str):
                if not streamed_text:
                    streamed_text = True
                chat.add_streaming_text(event)

            elif isinstance(event, ToolCallEvent):
                if event.result is None:
                    # Tool starting
                    chat.add_tool_call(event.name, event.args)
                    status.state = f"Running {event.name}..."
                    events_panel.add_event(
                        _now_str(), "tool_start",
                        (
                            f"{event.name} "
                            f"{tool_args_preview(event.name, event.args)}"
                        ),
                    )
                else:
                    # Tool completed
                    output = ""
                    if event.result.success:
                        output = event.result.output
                    error = event.result.error or ""
                    chat.add_tool_call(
                        event.name, event.args,
                        success=event.result.success,
                        elapsed_ms=event.elapsed_ms,
                        output=output,
                        error=error,
                    )

                    # Show multimodal content indicators
                    if (
                        event.result.content_blocks
                        and event.result.success
                    ):
                        chat.add_content_indicator(
                            event.result.content_blocks,
                        )

                    etype = (
                        "tool_ok"
                        if event.result.success
                        else "tool_err"
                    )
                    events_panel.add_event(
                        _now_str(), etype,
                        f"{event.name} {event.elapsed_ms}ms",
                    )
                    if event.result.success and event.name in _WORKSPACE_REFRESH_TOOLS:
                        self._refresh_workspace_tree()

                    if (
                        event.name in {"task_tracker", "delegate_task"}
                        and event.result.data
                    ):
                        self._update_sidebar_tasks(event.result.data)

                    # Handle ask_user
                    if (
                        event.name == "ask_user"
                        and event.result
                        and event.result.success
                    ):
                        answer = await self._handle_ask_user(event)
                        if answer:
                            await self._run_followup(answer)

            elif isinstance(event, CoworkTurn):
                if event.text and not streamed_text:
                    chat.add_model_text(event.text)

                self._total_tokens += event.tokens_used
                status = self.query_one("#status-bar", StatusBar)
                status.total_tokens = self._total_tokens

                chat.add_turn_separator(
                    len(event.tool_calls),
                    event.tokens_used,
                    event.model,
                )
                events_panel.add_event(
                    _now_str(), "turn",
                    f"{event.tokens_used} tokens",
                )
                events_panel.record_turn_tokens(event.tokens_used)

                # Update files panel
                self._update_files_panel(event)

    async def _run_process_goal(self, goal: str) -> None:
        """Compatibility wrapper for callers that still use the old method."""
        await self._start_process_run(goal)

    def _on_process_progress_event(
        self,
        data: dict,
        *,
        run_id: str | None = None,
    ) -> None:
        """Handle incremental delegate_task progress events in /run flows."""
        if not isinstance(data, dict):
            return
        run = self._process_runs.get(run_id) if run_id else None
        if run_id is not None and run is None:
            return
        if run is not None:
            if run.closed:
                return
            tasks = data.get("tasks", [])
            if isinstance(tasks, list):
                normalized = self._normalize_process_run_tasks(run, tasks)
                run.tasks = normalized
                run.pane.set_tasks(normalized)
                self._refresh_process_run_outputs(run)

            task_id = str(data.get("task_id", "")).strip()
            if task_id:
                run.task_id = task_id

            event_type = str(data.get("event_type") or "")
            if event_type == "tool_call_completed":
                tool_name = str(data.get("event_data", {}).get("tool", "")).strip()
                if tool_name in _WORKSPACE_REFRESH_TOOLS:
                    self._refresh_workspace_tree()
            if event_type in {
                "subtask_completed",
                "subtask_failed",
                "task_completed",
                "task_failed",
            }:
                self._refresh_workspace_tree()

            self._update_process_run_visuals(run)
            self._refresh_sidebar_progress_summary()
            message = self._format_process_progress_event(data, run=run)
            if not message:
                return
            now = time.monotonic()
            if (
                message == run.last_progress_message
                and (now - run.last_progress_at) < 2.0
            ):
                return
            run.last_progress_message = message
            run.last_progress_at = now
            self._append_process_run_activity(run, message)
            try:
                events_panel = self.query_one("#events-panel", EventPanel)
                if event_type != "token_streamed":
                    events_panel.add_event(
                        _now_str(),
                        "process",
                        f"{run.run_id}: {message[:132]}",
                    )
            except Exception:
                pass
            return

        self._update_sidebar_tasks(data)
        event_type = str(data.get("event_type") or "")
        if event_type == "tool_call_completed":
            tool_name = str(data.get("event_data", {}).get("tool", "")).strip()
            if tool_name in _WORKSPACE_REFRESH_TOOLS:
                self._refresh_workspace_tree()
        if event_type in {
            "subtask_completed",
            "subtask_failed",
            "task_completed",
            "task_failed",
        }:
            self._refresh_workspace_tree()
        message = self._format_process_progress_event(data)
        if not message:
            return
        try:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info(message)
        except Exception:
            pass
        try:
            events_panel = self.query_one("#events-panel", EventPanel)
            if event_type != "token_streamed":
                events_panel.add_event(_now_str(), "process", message[:140])
        except Exception:
            pass

    @staticmethod
    def _one_line(text: object | None, max_len: int | None = 180) -> str:
        """Normalize whitespace and cap a string for concise progress rows."""
        if text is None:
            return ""
        compact = " ".join(_plain_text(text).split())
        if max_len is None or max_len <= 0:
            return compact
        if len(compact) <= max_len:
            return compact
        return f"{compact[:max_len - 1].rstrip()}…"

    def _normalize_process_run_tasks(
        self, run: ProcessRunState, tasks: list[dict]
    ) -> list[dict]:
        """Keep process task rows stable and focused on the original plan labels."""
        phase_labels: dict[str, str] = {}
        process = getattr(run, "process_defn", None)
        if process is not None:
            for phase in getattr(process, "phases", []):
                phase_id = str(getattr(phase, "id", "")).strip()
                if not phase_id:
                    continue
                phase_desc = self._one_line(
                    getattr(phase, "description", ""),
                    max_len=None,
                )
                phase_labels[phase_id] = phase_desc or phase_id

        task_labels = getattr(run, "task_labels", None)
        if not isinstance(task_labels, dict):
            task_labels = {}
            try:
                run.task_labels = task_labels
            except Exception:
                pass

        normalized: list[dict] = []
        for row in tasks:
            if not isinstance(row, dict):
                continue
            subtask_id = str(row.get("id", "")).strip()
            raw_status = str(row.get("status", "pending")).strip()
            status = raw_status if raw_status in {
                "pending", "in_progress", "completed", "failed", "skipped",
            } else "pending"
            candidate = self._one_line(row.get("content", ""), max_len=None)
            if not candidate:
                candidate = subtask_id or "subtask"

            if subtask_id in phase_labels:
                label = phase_labels[subtask_id]
                task_labels[subtask_id] = label
            else:
                if subtask_id:
                    existing = str(task_labels.get(subtask_id, "")).strip()
                    # Only improve labels while task is active; don't let
                    # completion summaries replace the original checklist label.
                    if (
                        status in {"pending", "in_progress"}
                        and candidate
                        and (not existing or existing == subtask_id)
                    ):
                        task_labels[subtask_id] = candidate
                    label = str(task_labels.get(subtask_id, "")).strip() or candidate
                else:
                    label = candidate

            normalized.append({
                "id": subtask_id or candidate,
                "status": status,
                "content": label,
            })
        return normalized

    def _process_run_output_rows(self, run: ProcessRunState) -> list[dict]:
        """Build per-deliverable output status rows for the process run pane."""
        process = getattr(run, "process_defn", None)
        if process is None or not hasattr(process, "get_deliverables"):
            return []

        try:
            deliverables_by_phase = process.get_deliverables()
        except Exception:
            deliverables_by_phase = {}

        has_deliverables = (
            isinstance(deliverables_by_phase, dict)
            and bool(deliverables_by_phase)
        )

        if bool(getattr(run, "is_adhoc", False)) and not has_deliverables:
            rows: list[dict] = []
            for idx, task in enumerate(run.tasks, start=1):
                if not isinstance(task, dict):
                    continue
                status = str(task.get("status", "pending")).strip()
                if status not in {
                    "pending", "in_progress", "completed", "failed", "skipped",
                }:
                    status = "pending"
                label = self._one_line(task.get("content", ""), max_len=None)
                if not label:
                    label = str(task.get("id", "")).strip() or f"step-{idx}"
                rows.append({
                    "id": f"adhoc-output-{idx}",
                    "status": status,
                    "content": f"{label} (expected output)",
                })
            return rows

        if not has_deliverables:
            return []

        subtask_status: dict[str, str] = {}
        content_status: dict[str, str] = {}
        for row in getattr(run, "tasks", []):
            if not isinstance(row, dict):
                continue
            subtask_id = str(row.get("id", "")).strip()
            status = str(row.get("status", "pending")).strip()
            if subtask_id:
                subtask_status[subtask_id] = status
            content_text = self._one_line(row.get("content", ""), max_len=None)
            if content_text:
                content_status[content_text] = status

        ordered_phase_ids: list[str] = []
        phase_labels: dict[str, str] = {}
        for phase in getattr(process, "phases", []):
            phase_id = str(getattr(phase, "id", "")).strip()
            if phase_id:
                ordered_phase_ids.append(phase_id)
                phase_label = self._one_line(
                    getattr(phase, "description", ""),
                    max_len=None,
                )
                if phase_label:
                    phase_labels[phase_id] = phase_label
        for phase_id in deliverables_by_phase.keys():
            if phase_id not in ordered_phase_ids:
                ordered_phase_ids.append(phase_id)

        run_workspace = getattr(run, "run_workspace", None)
        workspace_root = Path(run_workspace) if run_workspace else self._workspace

        rows: list[dict] = []
        for phase_id in ordered_phase_ids:
            phase_deliverables = deliverables_by_phase.get(phase_id) or []
            if not isinstance(phase_deliverables, list):
                continue
            phase_state = subtask_status.get(phase_id, "").strip()
            if not phase_state:
                phase_label = phase_labels.get(phase_id, "")
                if phase_label:
                    phase_state = content_status.get(phase_label, "").strip()
            if phase_state not in {"pending", "in_progress", "completed", "failed", "skipped"}:
                phase_state = "pending"
            for path in phase_deliverables:
                rel_path = str(path).strip()
                if not rel_path:
                    continue
                exists = (workspace_root / rel_path).exists()
                if exists:
                    status = "completed"
                    suffix = ""
                elif phase_state == "in_progress":
                    status = "in_progress"
                    suffix = " (pending)"
                elif phase_state == "completed":
                    status = "failed"
                    suffix = " (missing)"
                elif phase_state == "failed":
                    status = "failed"
                    suffix = " (not produced)"
                elif phase_state == "skipped":
                    status = "skipped"
                    suffix = " (skipped)"
                else:
                    status = "pending"
                    suffix = " (planned)"
                rows.append({
                    "id": f"{phase_id}:{rel_path}",
                    "status": status,
                    "content": f"{rel_path} ({phase_id}){suffix}",
                })
        return rows

    def _refresh_process_run_outputs(self, run: ProcessRunState) -> None:
        """Refresh per-run output rows in the process pane."""
        if not hasattr(run, "pane") or run.pane is None:
            return
        try:
            rows = self._process_run_output_rows(run)
            run.pane.set_outputs(rows)
        except Exception:
            return

    @staticmethod
    def _subtask_content(
        data: dict,
        subtask_id: str,
        run: ProcessRunState | None = None,
    ) -> str:
        """Lookup subtask label, preferring stable run-normalized labels."""
        if not subtask_id:
            return ""

        if run is not None:
            labels = getattr(run, "task_labels", {})
            if isinstance(labels, dict):
                label = str(labels.get(subtask_id, "")).strip()
                if label and label != subtask_id:
                    return label

            run_tasks = getattr(run, "tasks", [])
            if isinstance(run_tasks, list):
                for row in run_tasks:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("id", "")) != subtask_id:
                        continue
                    content = _plain_text(row.get("content", "")).strip()
                    if content and content != subtask_id:
                        return content
                    break

        tasks = data.get("tasks")
        if isinstance(tasks, list):
            for row in tasks:
                if not isinstance(row, dict):
                    continue
                if str(row.get("id", "")) != subtask_id:
                    continue
                content = _plain_text(row.get("content", "")).strip()
                if content and content != subtask_id:
                    return content
                break
        return ""

    def _format_process_progress_event(
        self,
        data: dict,
        *,
        run: ProcessRunState | None = None,
    ) -> str | None:
        """Format orchestrator progress events into concise chat messages."""
        event_type = str(data.get("event_type") or "")
        event_data = data.get("event_data")
        if not event_type:
            return None
        if not isinstance(event_data, dict):
            event_data = {}

        subtask_id = str(event_data.get("subtask_id", "")).strip()
        subtask_content = self._subtask_content(data, subtask_id, run)
        subtask_label = subtask_id or "subtask"
        if subtask_content:
            subtask_label = f"{subtask_label} - {self._one_line(subtask_content, 90)}"

        if event_type == "task_planning":
            return "Planning process run..."
        if event_type == "task_plan_ready":
            count = len(data.get("tasks", [])) if isinstance(data.get("tasks"), list) else 0
            return f"Plan ready: {count} subtasks."
        if event_type == "task_executing":
            return "Executing subtasks..."
        if event_type == "model_invocation":
            phase = str(event_data.get("phase", "")).strip()
            model_name = str(event_data.get("model", "")).strip()
            label = subtask_label

            def _int_value(value: object) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return 0

            request_bytes = _int_value(event_data.get("request_bytes"))
            request_tokens = _int_value(event_data.get("request_est_tokens"))
            message_count = _int_value(event_data.get("message_count"))
            assistant_tool_calls = _int_value(event_data.get("assistant_tool_calls"))
            origin = str(event_data.get("origin", "")).strip()

            details: list[str] = []
            if request_tokens > 0:
                details.append(f"{request_tokens:,} est tokens")
            if request_bytes > 0:
                if request_bytes >= 1024 * 1024:
                    details.append(f"{request_bytes / (1024 * 1024):.2f} MB")
                else:
                    details.append(f"{request_bytes / 1024:.0f} KB")
            if message_count > 0:
                details.append(f"{message_count} msgs")
            if assistant_tool_calls > 0:
                details.append(f"{assistant_tool_calls} tool calls in ctx")
            if origin and not origin.startswith("runner.execute_subtask."):
                details.append(origin)

            if phase == "start":
                if request_bytes >= 3_500_000:
                    size_text = (
                        f"{request_bytes / (1024 * 1024):.2f} MB"
                        if request_bytes > 0
                        else "oversize"
                    )
                    token_text = (
                        f"{request_tokens:,} est tokens"
                        if request_tokens > 0
                        else "estimated tokens unavailable"
                    )
                    return (
                        f"Request-size risk for {label}: {size_text}, {token_text}. "
                        "Compaction/plumbing needed."
                    )
                if model_name:
                    if details:
                        return (
                            f"Thinking on {label} with {model_name} "
                            f"({', '.join(details)})..."
                        )
                    return f"Thinking on {label} with {model_name}..."
                if details:
                    return f"Thinking on {label} ({', '.join(details)})..."
                return f"Thinking on {label}..."
            if phase == "done":
                return None
            return None
        if event_type == "token_streamed":
            count = event_data.get("token_count")
            try:
                token_count = int(count)
            except (TypeError, ValueError):
                token_count = 0
            if token_count > 0:
                return f"Working on {subtask_label}... ({token_count} streamed chunks)"
            return None
        if event_type == "tool_call_started":
            tool = str(event_data.get("tool", "")).strip() or "tool"
            return f"Using {tool} for {subtask_label}."
        if event_type == "tool_call_completed":
            tool = str(event_data.get("tool", "")).strip() or "tool"
            success = event_data.get("success")
            if success is True:
                return f"Finished {tool} for {subtask_label}."
            error = self._one_line(event_data.get("error", ""), 120)
            if error:
                return f"{tool} failed for {subtask_label}: {error}"
            return f"{tool} failed for {subtask_label}."
        if event_type == "subtask_started":
            return f"Started {subtask_label}."
        if event_type == "subtask_retrying":
            attempt = event_data.get("attempt")
            tier = event_data.get("escalated_tier")
            reason = self._one_line(event_data.get("feedback", ""), 120)
            msg = f"Retrying {subtask_label}"
            if attempt:
                msg += f" (attempt {attempt})"
            if tier:
                msg += f", tier {tier}"
            if reason:
                msg += f": {reason}"
            return f"{msg}."
        if event_type == "subtask_completed":
            return f"Completed {subtask_label}."
        if event_type == "subtask_failed":
            reason = self._one_line(
                event_data.get("feedback")
                or event_data.get("reason")
                or event_data.get("error")
                or "",
                140,
            )
            if reason:
                return f"Failed {subtask_label}: {reason}"
            return f"Failed {subtask_label}."
        if event_type == "task_replanning":
            reason = self._one_line(event_data.get("reason", ""), 140)
            if reason:
                return f"Replanning task: {reason}"
            return "Replanning task..."
        if event_type == "task_plan_normalized":
            normalized = event_data.get("normalized_subtasks", [])
            if not isinstance(normalized, list):
                normalized = []
            changed_ids: list[str] = []
            for item in normalized:
                if not isinstance(item, dict):
                    continue
                subtask_id = str(item.get("subtask_id", "")).strip()
                if subtask_id:
                    changed_ids.append(subtask_id)
            context = str(event_data.get("context", "")).strip()
            if changed_ids:
                joined = ", ".join(changed_ids[:3])
                if len(changed_ids) > 3:
                    joined += ", ..."
                if context:
                    return (
                        f"Normalized plan ({context}): demoted non-terminal synthesis "
                        f"for {joined}."
                    )
                return (
                    "Normalized plan: demoted non-terminal synthesis for "
                    f"{joined}."
                )
            return "Normalized plan topology."
        if event_type == "task_stalled":
            blocked = event_data.get("blocked_subtasks", [])
            if not isinstance(blocked, list):
                blocked = []
            attempt = event_data.get("attempt")
            attempt_text = ""
            try:
                attempt_num = int(attempt)
                if attempt_num > 0:
                    attempt_text = f" (attempt {attempt_num})"
            except (TypeError, ValueError):
                attempt_text = ""
            if blocked and isinstance(blocked[0], dict):
                first = blocked[0]
                first_id = str(first.get("subtask_id", "")).strip() or "subtask"
                reasons = first.get("reasons", [])
                if isinstance(reasons, list):
                    reason_text = self._one_line(
                        ", ".join(
                            str(reason).strip()
                            for reason in reasons
                            if str(reason).strip()
                        ),
                        120,
                    )
                else:
                    reason_text = self._one_line(reasons, 120)
                if reason_text:
                    return (
                        f"Execution stalled{attempt_text}: "
                        f"{first_id} blocked ({reason_text})."
                    )
                return f"Execution stalled{attempt_text}: {first_id} blocked."
            return f"Execution stalled{attempt_text}: no runnable subtasks."
        if event_type == "task_stalled_recovery_attempted":
            mode = str(event_data.get("recovery_mode", "")).strip().lower()
            success = event_data.get("recovery_success")
            attempt = event_data.get("attempt")
            attempt_suffix = ""
            try:
                attempt_num = int(attempt)
                if attempt_num > 0:
                    attempt_suffix = f" (attempt {attempt_num})"
            except (TypeError, ValueError):
                attempt_suffix = ""
            mode_label = mode or "recovery"
            if success is True:
                return (
                    f"Stall recovery via {mode_label} succeeded"
                    f"{attempt_suffix}."
                )
            reason = self._one_line(event_data.get("reason", ""), 120)
            if reason:
                return (
                    f"Stall recovery via {mode_label} failed"
                    f"{attempt_suffix}: {reason}"
                )
            return f"Stall recovery via {mode_label} failed{attempt_suffix}."
        if event_type == "task_completed":
            return "Process run completed."
        if event_type == "task_failed":
            reason = self._one_line(
                event_data.get("reason")
                or event_data.get("error")
                or "",
                140,
            )
            if reason:
                return f"Process run failed: {reason}"
            return "Process run failed."
        return None

    def _mark_process_run_failed(self, error: str) -> None:
        """Reflect a failed /run execution in the progress panel."""
        message = error.strip() or "Process run failed."
        if "timed out" in message.lower():
            message = (
                f"{message} Increase \\[execution].delegate_task_timeout_seconds "
                "(or LOOM_DELEGATE_TIMEOUT_SECONDS) for longer runs."
            )
        self._sidebar_cowork_tasks = [
            {
                "id": "process-run",
                "status": "failed",
                "content": f"/run failed: {message}",
            },
        ]
        self._refresh_sidebar_progress_summary()

    async def _handle_ask_user(self, event: ToolCallEvent) -> str:
        """Show an ask_user modal and return the answer."""
        question = event.args.get("question", "")
        options = event.args.get("options", [])

        answer_event = asyncio.Event()
        answer_holder: list[str] = []

        def handle_answer(answer: str) -> None:
            answer_holder.append(answer)
            answer_event.set()

        self.push_screen(
            AskUserScreen(question, options),
            callback=handle_answer,
        )

        await answer_event.wait()
        answer = answer_holder[0] if answer_holder else ""

        if answer:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_user_message(answer)

        return answer

    # ------------------------------------------------------------------
    # Data panel updates
    # ------------------------------------------------------------------

    def _update_sidebar_tasks(self, data: dict) -> None:
        """Update sidebar task progress from a tool result payload."""
        if not data:
            return
        if not isinstance(data, dict):
            return
        tasks = data.get("tasks", [])
        if not tasks and "id" in data:
            tasks = [data]
        normalized: list[dict] = []
        if isinstance(tasks, list):
            for row in tasks:
                if isinstance(row, dict):
                    normalized.append(row)
        self._sidebar_cowork_tasks = normalized
        self._refresh_sidebar_progress_summary()

    def _summarize_cowork_tasks(self) -> list[dict]:
        """Return a compact one-row summary for cowork delegated tasks."""
        if not self._sidebar_cowork_tasks:
            return []
        total = len(self._sidebar_cowork_tasks)
        in_progress = sum(
            1
            for t in self._sidebar_cowork_tasks
            if str(t.get("status", "")) == "in_progress"
        )
        failed = sum(
            1
            for t in self._sidebar_cowork_tasks
            if str(t.get("status", "")) == "failed"
        )
        completed = sum(
            1
            for t in self._sidebar_cowork_tasks
            if str(t.get("status", "")) == "completed"
        )
        primary = next(
            (
                t
                for t in self._sidebar_cowork_tasks
                if str(t.get("status", "")) == "in_progress"
            ),
            self._sidebar_cowork_tasks[0],
        )
        focus = self._one_line(primary.get("content", ""), 180)
        status = "pending"
        if in_progress:
            status = "in_progress"
        elif failed:
            status = "failed"
        elif completed and completed == total:
            status = "completed"
        content = (
            f"Cowork delegated: {total} task(s) "
            f"({in_progress} active, {failed} failed)"
        )
        if focus:
            content += f" - {focus}"
        return [{
            "id": "cowork-delegated",
            "status": status,
            "content": content,
        }]

    def _refresh_sidebar_progress_summary(self) -> None:
        """Render concise sidebar progress: one row per run + cowork summary."""
        try:
            sidebar = self.query_one("#sidebar", Sidebar)
        except Exception:
            return
        rows: list[dict] = []
        for run in sorted(self._process_runs.values(), key=lambda r: r.started_at):
            state = run.status
            row_status = {
                "queued": "pending",
                "running": "in_progress",
                "completed": "completed",
                "failed": "failed",
                "cancelled": "skipped",
            }.get(state, "pending")
            elapsed = self._format_elapsed(self._elapsed_seconds_for_run(run))
            label = _PROCESS_STATUS_LABEL.get(state, state.title())
            rows.append({
                "id": f"process-run-{run.run_id}",
                "status": row_status,
                "content": f"{run.process_name} #{run.run_id} {label} {elapsed}",
            })
        rows.extend(self._summarize_cowork_tasks())
        sidebar.update_tasks(rows)
        self._sync_process_runs_into_session_state()

    def _refresh_workspace_tree(self) -> None:
        """Reload sidebar workspace tree to pick up new files."""
        try:
            sidebar = self.query_one("#sidebar", Sidebar)
        except Exception:
            return
        sidebar.refresh_workspace_tree()

    def _resolve_workspace_file(self, path: Path) -> Path | None:
        """Resolve a selected file and ensure it remains inside workspace."""
        try:
            resolved = path.resolve()
            workspace_root = self._workspace.resolve()
        except OSError:
            return None
        if not resolved.is_file():
            return None
        try:
            resolved.relative_to(workspace_root)
        except ValueError:
            return None
        return resolved

    def _update_files_panel(self, turn: CoworkTurn) -> None:
        """Update the Files Changed panel from tool call events."""
        file_entries: list[dict] = []
        last_diff = ""
        refresh_workspace = False
        for tc in turn.tool_calls:
            if not tc.result or not tc.result.success:
                continue
            if tc.name in _WORKSPACE_REFRESH_TOOLS:
                refresh_workspace = True
            path = tc.args.get(
                "path", tc.args.get("file_path", "?"),
            )
            now = _now_str()
            if tc.name == "write_file":
                file_entries.append({
                    "operation": "create",
                    "path": path,
                    "timestamp": now,
                })
            elif tc.name == "edit_file":
                file_entries.append({
                    "operation": "modify",
                    "path": path,
                    "timestamp": now,
                })
                # Extract diff from edit output for the diff viewer
                output = tc.result.output or ""
                marker = "--- a/"
                idx = output.find(marker)
                if idx != -1:
                    last_diff = output[idx:]
            elif tc.name == "delete_file":
                file_entries.append({
                    "operation": "delete",
                    "path": path,
                    "timestamp": now,
                })
            elif tc.name == "move_file":
                src = tc.args.get("source", "?")
                dst = tc.args.get("destination", "?")
                file_entries.append({
                    "operation": "rename",
                    "path": f"{src} -> {dst}",
                    "timestamp": now,
                })
        if file_entries:
            panel = self.query_one("#files-panel", FilesChangedPanel)
            panel.update_files(file_entries)
            if last_diff:
                panel.show_diff(last_diff)
            self._refresh_workspace_tree()
            count = len(file_entries)
            s = "s" if count != 1 else ""
            self.notify(
                f"{count} file{s} changed", timeout=3,
            )
            return
        if refresh_workspace:
            self._refresh_workspace_tree()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_toggle_sidebar(self) -> None:
        self.query_one("#sidebar", Sidebar).toggle()

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        for child in list(chat.children):
            child.remove()

    def action_reload_workspace(self) -> None:
        """Reload sidebar workspace tree to show external file changes."""
        self._refresh_workspace_tree()
        self._refresh_process_command_index()
        self.notify("Workspace reloaded", timeout=2)

    def action_close_process_tab(self) -> None:
        """Close current process run tab with confirmation."""
        if self._close_process_tab_inflight:
            return
        self._close_process_tab_inflight = True

        async def _close_current_tab() -> None:
            try:
                await self._close_process_run_from_target("current")
            finally:
                self._close_process_tab_inflight = False

        try:
            self.run_worker(
                _close_current_tab(),
                group="close-process-tab",
                exclusive=False,
            )
        except Exception:
            self._close_process_tab_inflight = False
            raise

    def action_tab_chat(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab-chat"

    def action_tab_files(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab-files"

    def action_tab_events(self) -> None:
        tabs = self.query_one("#tabs", TabbedContent)
        tabs.active = "tab-events"

    async def action_quit(self) -> None:
        """Compatibility action that runs the exit flow inline."""
        await self._request_exit()

    def action_request_quit(self) -> None:
        """Start the exit flow without blocking key/event dispatch."""
        self.run_worker(
            self._request_exit(),
            group="exit-flow",
            exclusive=True,
        )

    async def _confirm_exit(self) -> bool:
        """Show exit confirmation modal and return True when confirmed."""
        if self._confirm_exit_waiter is not None:
            return await self._confirm_exit_waiter

        result_waiter: asyncio.Future[bool] = asyncio.Future()
        self._confirm_exit_waiter = result_waiter

        def handle_result(confirmed: bool) -> None:
            if not result_waiter.done():
                result_waiter.set_result(bool(confirmed))

        self.push_screen(ExitConfirmScreen(), callback=handle_result)
        try:
            return await result_waiter
        finally:
            self._confirm_exit_waiter = None

    async def _request_exit(self) -> None:
        """Prompt for exit confirmation, then persist and quit when approved."""
        if not await self._confirm_exit():
            return
        await self._persist_process_run_ui_state(is_active=False)
        self.exit()

    async def action_loom_command(self, command: str) -> None:
        """Dispatch command palette actions."""
        if command.startswith("process_run_prompt:"):
            process_name = command.partition(":")[2].strip()
            if process_name:
                self._prefill_user_input(f"/{process_name} ")
            return
        if command == "quit":
            self.action_request_quit()
            return
        if command == "setup":
            await self._handle_slash_command("/setup")
            return
        if command == "session_info":
            await self._handle_slash_command("/session")
            return
        if command == "new_session":
            await self._handle_slash_command("/new")
            return
        if command == "sessions_list":
            await self._handle_slash_command("/sessions")
            return
        if command == "mcp_list":
            await self._handle_slash_command("/mcp list")
            return
        if command == "mcp_manage":
            await self._handle_slash_command("/mcp manage")
            return
        if command == "mcp_add_prompt":
            self._prefill_user_input(
                "/mcp add <alias> --command <cmd> --arg <value> "
            )
            return
        if command == "auth_list":
            await self._handle_slash_command("/auth list")
            return
        if command == "auth_manage":
            await self._handle_slash_command("/auth manage")
            return
        if command == "auth_add_prompt":
            self._prefill_user_input(
                "/auth add <profile-id> --provider <provider> --mode <mode> --mcp-server <alias> "
            )
            return
        if command == "learned_patterns":
            await self._handle_slash_command("/learned")
            return
        if command == "process_use_prompt":
            self._prefill_user_input("/process use ")
            return
        if command == "run_prompt":
            self._prefill_user_input("/run ")
            return
        if command == "resume_prompt":
            self._prefill_user_input("/resume ")
            return
        if command == "close_process_tab":
            await self._close_process_run_from_target("current")
            return
        if command == "process_off":
            chat = self.query_one("#chat-log", ChatLog)
            if not self._process_name and self._process_defn is None:
                chat.add_info("No active process.")
                return
            self._process_name = None
            self._process_defn = None
            await self._reload_session_for_process_change(chat)
            chat.add_info("Active process: none")
            return
        actions = {
            "clear_chat": self.action_clear_chat,
            "toggle_sidebar": self.action_toggle_sidebar,
            "reload_workspace": self.action_reload_workspace,
            "tab_chat": self.action_tab_chat,
            "tab_files": self.action_tab_files,
            "tab_events": self.action_tab_events,
            "list_tools": self._show_tools,
            "model_info": self._show_model_info,
            "process_info": self._show_process_info,
            "process_list": self._show_process_list,
            "token_info": self._show_token_info,
            "help": self._show_help,
        }
        action_fn = actions.get(command)
        if action_fn:
            action_fn()

    def _prefill_user_input(self, text: str) -> None:
        """Seed the chat input with a command template and focus it."""
        self._reset_input_history_navigation()
        input_widget = self.query_one("#user-input", Input)
        self._set_user_input_text(text)
        input_widget.focus()
        self._set_slash_hint(self._render_slash_hint(text))

    def _show_tools(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(self._render_tools_catalog())

    def _show_model_info(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        name = self._model.name if self._model else "(not configured)"
        chat.add_info(f"Model: {name}")

    def _show_process_info(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(f"Active process: {self._active_process_name()}")

    def _show_process_list(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(self._render_process_catalog())

    def iter_dynamic_process_palette_entries(self) -> list[tuple[str, str, str]]:
        """Return palette entries for dynamically discovered process commands."""
        self._refresh_process_command_index()
        entries: list[tuple[str, str, str]] = []
        for token, process_name in sorted(self._process_command_map.items()):
            entries.append((
                f"Run {process_name}…",
                f"process_run_prompt:{process_name}",
                f"Prefill {token} for direct process execution",
            ))
        return entries

    def _show_token_info(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(f"Session tokens: {self._total_tokens:,}")

    def _show_help(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info("\n".join(self._help_lines()))

    async def _palette_quit(self) -> None:
        await self._request_exit()


def _now_str() -> str:
    """Return current time as HH:MM:SS string."""
    return datetime.now().strftime("%H:%M:%S")
