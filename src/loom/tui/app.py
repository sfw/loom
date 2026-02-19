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
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
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
    ExitConfirmScreen,
    FileViewerScreen,
    LearnedScreen,
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
from loom.tui.widgets.sidebar import TaskProgressPanel
from loom.tui.widgets.tool_call import tool_args_preview

if TYPE_CHECKING:
    from loom.config import Config
    from loom.processes.schema import ProcessDefinition
    from loom.state.conversation_store import ConversationStore
    from loom.state.memory import Database

logger = logging.getLogger(__name__)


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
        usage="[list|show <alias>|test <alias>|enable <alias>|disable <alias>|remove <alias>]",
        description="inspect/manage MCP server config",
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
        description="inspect/switch active process",
    ),
    SlashCommandSpec(
        canonical="/run",
        usage="<goal|close [run-id-prefix]>",
        description="run goal via active process orchestrator",
    ),
)
_MAX_SLASH_HINT_LINES = 24
_WORKSPACE_REFRESH_TOOLS = {"document_write", "document_create"}
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


class ProcessRunPane(Vertical):
    """A per-run process pane with status, progress rows, and run log."""

    DEFAULT_CSS = """
    ProcessRunPane {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
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
    ProcessRunPane .process-run-section {
        color: $text-muted;
        text-style: bold;
        margin: 1 0 0 0;
    }
    ProcessRunPane #process-run-progress {
        max-height: 12;
        height: auto;
        overflow-y: auto;
    }
    ProcessRunPane #process-run-outputs {
        max-height: 8;
        height: auto;
        overflow-y: auto;
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
        self._progress_label = Static("Progress", classes="process-run-section")
        self._progress = TaskProgressPanel(id="process-run-progress")
        self._outputs_label = Static("Outputs", classes="process-run-section")
        self._outputs = TaskProgressPanel(id="process-run-outputs")
        self._outputs.empty_message = "No outputs yet"
        self._log_label = Static("Activity", classes="process-run-section")
        self._log = ChatLog()

    def compose(self) -> ComposeResult:
        yield self._header
        yield self._meta
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
        goal_preview = self._goal
        if len(goal_preview) > 140:
            goal_preview = f"{goal_preview[:139].rstrip()}\u2026"
        meta = f"[dim]Goal:[/] {goal_preview}"
        if task_id:
            meta += f"\n[dim]Task:[/] {task_id}"
        meta += (
            "\n[dim]Close: Ctrl+W | /run close [run-id-prefix] | "
            "Ctrl+P: Close process run tab[/dim]"
        )
        self._meta.update(meta)

    def set_tasks(self, tasks: list[dict]) -> None:
        """Replace task rows shown in the progress section."""
        if not self.is_attached:
            self._pending_tasks = list(tasks)
            return
        self._progress.tasks = tasks

    def set_outputs(self, outputs: list[dict]) -> None:
        """Replace output rows shown in the outputs section."""
        if not self.is_attached:
            self._pending_outputs = list(outputs)
            return
        self._outputs.tasks = outputs

    def add_activity(self, text: str) -> None:
        """Append informational activity text."""
        if not self.is_attached:
            self._pending_activity.append(text)
            return
        self._log.add_info(text)

    def add_result(self, text: str, *, success: bool) -> None:
        """Append final result text."""
        if not self.is_attached:
            self._pending_results.append((text, success))
            return
        if success:
            self._log.add_model_text(text)
            return
        self._log.add_model_text(f"[bold #f7768e]Error:[/] {text}")

    def on_mount(self) -> None:
        """Flush updates queued before the pane was attached to the DOM."""
        if self._pending_tasks is not None:
            self._progress.tasks = self._pending_tasks
            self._pending_tasks = None
        if self._pending_outputs is not None:
            self._outputs.tasks = self._pending_outputs
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
                    self._log.add_model_text(f"[bold #f7768e]Error:[/] {text}")
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
        self._process_runs: dict[str, ProcessRunState] = {}
        self._process_elapsed_timer = None
        self._process_command_map: dict[str, str] = {}
        self._blocked_process_commands: list[str] = []
        self._cached_process_catalog: list[dict[str, str]] = []
        self._sidebar_cowork_tasks: list[dict] = []
        self._process_close_hint_shown = False
        self._auto_resume_workspace_on_init = True

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
        )

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

    def _render_process_catalog(self) -> str:
        """Build a human-readable process list."""
        self._refresh_process_command_index()
        available = self._cached_process_catalog
        if not available:
            if self._blocked_process_commands:
                blocked = ", ".join(self._blocked_process_commands)
                return (
                    "No selectable process definitions found.\n"
                    f"[dim]Blocked (name collisions): {blocked}[/dim]"
                )
            return "No process definitions found."

        active = self._process_defn.name if self._process_defn else ""
        lines = ["[bold]Available processes:[/bold]"]
        for proc in available:
            name = proc["name"]
            ver = proc["version"]
            desc = proc.get("description", "").strip().split("\n")[0]
            marker = " [cyan]<< active[/cyan]" if name == active else ""
            lines.append(f"  {name:30s} v{ver:6s}{marker}")
            if desc:
                lines.append(f"    [dim]{desc[:80]}[/dim]")
        if self._blocked_process_commands:
            blocked = ", ".join(self._blocked_process_commands)
            lines.append(f"[dim]Blocked (name collisions): {blocked}[/dim]")
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
                system_prompt=system_prompt,
                approver=approver,
                store=self._store,
                model_retry_policy=self._model_retry_policy(),
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
                    f"{resume_label} [dim]{resume_target}[/dim] "
                    f"({self._session.session_state.turn_count} turns)"
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
                system_prompt=system_prompt,
                approver=approver,
                store=self._store,
                session_id=session_id,
                model_retry_policy=self._model_retry_policy(),
            )
        else:
            # Ephemeral session (no database)
            self._session = CoworkSession(
                model=self._model,
                tools=self._tools,
                workspace=self._workspace,
                system_prompt=system_prompt,
                approver=approver,
                model_retry_policy=self._model_retry_policy(),
            )

        # Bind session-dependent tools
        self._bind_session_tools()

        # Configure status bar
        status = self.query_one("#status-bar", StatusBar)
        status.workspace_name = self._workspace.name
        status.model_name = self._model.name
        status.process_name = self._active_process_name()

        # Welcome message
        chat.add_info(
            f"[bold]Loom[/bold]  [dim]({self._model.name})[/dim]"
        )
        chat.add_info(f"workspace: {self._workspace}")
        tool_count = len(self._tools.list_tools())
        persisted = "persisted" if self._store else "ephemeral"
        chat.add_info(f"{tool_count} tools loaded. Session: {persisted}.")
        if self._session and self._session.session_id:
            chat.add_info(
                f"[dim]session: {self._session.session_id}[/dim]"
            )
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
        message = str(text or "").strip()
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
            info = f"Restored {restored} process run tab(s) for this session."
            if interrupted:
                info += (
                    f" {interrupted} interrupted run(s) were marked failed "
                    "because execution cannot resume after restart."
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
            }
        state = self._session.session_state
        context: dict = {
            "requested_goal": goal,
            "workspace": str(workspace),
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
        if self._model is None:
            return ""
        prompt = (
            "Return one concise kebab-case folder name for this process run. "
            "Use 2-6 words, only lowercase letters/numbers/hyphens, no slashes, "
            "no punctuation, no explanation.\n"
            f"Process: {process_name}\n"
            f"Goal: {goal}\n"
            "Folder name:"
        )
        try:
            response = await call_with_model_retry(
                lambda: self._model.complete(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=24,
                ),
                policy=self._model_retry_policy(),
            )
        except Exception as e:
            logger.warning("LLM run-folder naming failed: %s", e)
            return ""
        text = str(getattr(response, "text", "") or "")
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        first_line = first_line.strip().strip("`")
        return self._slugify_process_run_folder(first_line)

    async def _prepare_process_run_workspace(self, process_name: str, goal: str) -> Path:
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

        def handle_result(confirmed: bool) -> None:
            if not waiter.done():
                waiter.set_result(bool(confirmed))

        running = run.status in {"queued", "running"}
        self.push_screen(
            ProcessRunCloseScreen(
                run_label=f"{run.process_name} #{run.run_id}",
                running=running,
            ),
            callback=handle_result,
        )
        return await waiter

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

    async def _start_process_run(
        self,
        goal: str,
        *,
        process_defn: ProcessDefinition | None = None,
        command_prefix: str = "/run",
    ) -> None:
        """Create a run tab and launch process execution in a background worker."""
        chat = self.query_one("#chat-log", ChatLog)
        events_panel = self.query_one("#events-panel", EventPanel)

        selected_process = process_defn or self._process_defn
        if selected_process is None:
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

        process_name = selected_process.name
        run_workspace = await self._prepare_process_run_workspace(process_name, goal)
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
            started_at=time.monotonic(),
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
            result = await self._tools.execute(
                "delegate_task",
                {
                    "goal": run.goal,
                    "context": self._build_process_run_context(
                        run.goal,
                        workspace=run.run_workspace,
                    ),
                    "wait": True,
                    "_approval_mode": "disabled",
                    "_process_override": run.process_defn,
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
        lines = ["Slash commands:"]
        for spec in _SLASH_COMMANDS:
            label = spec.canonical
            if spec.usage:
                label = f"{label} {spec.usage}"
            if spec.aliases:
                alias_str = ", ".join(spec.aliases)
                label = f"{label} ({alias_str})"
            lines.append(f"  {label:<34} {spec.description}")
        if self._process_command_map:
            lines.append("")
            lines.append("Process slash commands:")
            for token, process_name in sorted(self._process_command_map.items()):
                lines.append(
                    f"  {f'{token} <goal>':<34} run goal via process '{process_name}'"
                )
        if self._blocked_process_commands:
            blocked = ", ".join(f"/{name}" for name in self._blocked_process_commands)
            lines.append("")
            lines.append(
                f"[#f7768e]Blocked process commands (name collisions): {blocked}[/]"
            )
        lines.append(
            "Keys: Ctrl+B sidebar, Ctrl+L clear, Ctrl+R reload workspace, "
            "Ctrl+W close run tab, Ctrl+P palette, Ctrl+1/2/3 tabs",
        )
        return lines

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
            lines.append(f"  [#73daca]{cmd:<10}[/] {desc}")
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
            system_prompt=system_prompt,
            approver=approver,
            store=self._store,
            session_id=session_id,
            model_retry_policy=self._model_retry_policy(),
        )
        self._total_tokens = 0
        self._bind_session_tools()
        self._clear_files_panel()
        chat = self.query_one("#chat-log", ChatLog)
        await self._restore_process_run_tabs(chat)
        self._process_close_hint_shown = bool(self._process_runs)
        chat.add_info(f"New session: [dim]{session_id}[/dim]")

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
            system_prompt=system_prompt,
            approver=approver,
            store=self._store,
            model_retry_policy=self._model_retry_policy(),
        )
        await new_session.resume(session_id)

        self._session = new_session
        self._total_tokens = new_session.total_tokens
        self._bind_session_tools()
        self._clear_files_panel()
        chat = self.query_one("#chat-log", ChatLog)
        await self._restore_process_run_tabs(chat)
        self._process_close_hint_shown = bool(self._process_runs)
        chat.add_info(
            f"Switched to session [dim]{session_id}[/dim] "
            f"({new_session.session_state.turn_count} turns)"
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
        self._set_slash_hint("")

        # Handle slash commands
        if text.startswith("/"):
            handled = await self._handle_slash_command(text)
            if handled:
                return

        if self._chat_busy:
            return

        self._run_turn(text)

    @on(Input.Changed, "#user-input")
    def on_user_input_changed(self, _event: Input.Changed) -> None:
        """Show slash-command hints as the user types."""
        if self._skip_slash_cycle_reset_once:
            self._skip_slash_cycle_reset_once = False
        elif not self._applying_slash_tab_completion:
            self._reset_slash_tab_cycle()
        # Use the widget's current value rather than event payload to avoid
        # stale-value edge cases that can appear one keypress behind.
        current = self.query_one("#user-input", Input).value
        self._set_slash_hint(self._render_slash_hint(current))

    def on_key(self, event: events.Key) -> None:
        """Handle user-input key captures (autocomplete + close-run shortcut)."""
        if event.key == "ctrl+w":
            focused = self.focused
            if isinstance(focused, Input) and focused.id == "user-input":
                event.stop()
                event.prevent_default()
                self.action_close_process_tab()
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
            from loom.mcp.config import MCPConfigManagerError, ensure_valid_alias

            manager = self._mcp_manager()
            if not arg:
                chat.add_info(
                    "Usage:\n"
                    "  /mcp list\n"
                    "  /mcp show <alias>\n"
                    "  /mcp test <alias>\n"
                    "  /mcp enable <alias>\n"
                    "  /mcp disable <alias>\n"
                    "  /mcp remove <alias>"
                )
                return True

            subparts = arg.split(None, 1)
            subcmd = subparts[0].lower()
            rest = subparts[1].strip() if len(subparts) > 1 else ""

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
                    chat.add_info("Usage: /mcp show <alias>")
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
                    chat.add_info("Usage: /mcp test <alias>")
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

            if subcmd in {"enable", "disable"}:
                if not rest:
                    chat.add_info(f"Usage: /mcp {subcmd} <alias>")
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
                    chat.add_info("Usage: /mcp remove <alias>")
                    return True
                try:
                    alias = ensure_valid_alias(rest)
                    await asyncio.to_thread(manager.remove_server, alias)
                    await self._reload_mcp_runtime()
                    chat.add_info(f"MCP server '{alias}' removed.")
                except Exception as e:
                    chat.add_info(f"[bold #f7768e]{e}[/]")
                return True

            chat.add_info(
                "Usage: /mcp [list|show <alias>|test <alias>|enable <alias>|"
                "disable <alias>|remove <alias>]"
            )
            return True
        if token == "/setup":
            self.push_screen(
                SetupScreen(), callback=self._on_setup_complete,
            )
            return True
        if token == "/tools":
            tools = self._tools.list_tools()
            chat.add_info(
                f"{len(tools)} tools: " + ", ".join(tools)
            )
            return True
        if token == "/tokens":
            chat.add_info(f"Session tokens: {self._total_tokens:,}")
            return True
        if token == "/process":
            self._refresh_process_command_index()
            if not arg:
                active = self._active_process_name()
                chat.add_info(
                    f"Active process: {active}\n"
                    "Usage:\n"
                    "  /process list\n"
                    "  /process use <name-or-path>\n"
                    "  /process off"
                )
                return True

            subparts = arg.split(None, 1)
            subcmd = subparts[0].lower()
            rest = subparts[1].strip() if len(subparts) > 1 else ""

            if subcmd == "list":
                chat.add_info(self._render_process_catalog())
                return True

            if subcmd == "use":
                if not rest:
                    chat.add_info("Usage: /process use <name-or-path>")
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
                    f"Active process: [bold]{loaded.name}[/bold] v{loaded.version}"
                )
                return True

            if subcmd in {"off", "none", "clear"}:
                if not self._process_name and self._process_defn is None:
                    chat.add_info("No active process.")
                    return True
                self._process_name = None
                self._process_defn = None
                await self._reload_session_for_process_change(chat)
                chat.add_info("Active process: none")
                return True

            chat.add_info(
                "Usage: /process [list|use <name-or-path>|off]"
            )
            return True

        if token == "/run":
            if arg:
                subcmd, _, subrest = arg.partition(" ")
                if subcmd.lower() == "close":
                    await self._close_process_run_from_target(subrest)
                    return True
            goal = self._strip_wrapping_quotes(arg)
            if not goal:
                chat.add_info("Usage: /run <goal>")
                return True
            if self._process_defn is None:
                chat.add_info(
                    "No active process. Use /process use <name-or-path> first.",
                )
                return True
            await self._start_process_run(goal, process_defn=self._process_defn)
            return True

        process_name = self._process_command_map.get(token)
        if process_name:
            goal = self._strip_wrapping_quotes(arg)
            if not goal:
                chat.add_info(f"Usage: /{process_name} <goal>")
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
            info = (
                f"Session: {self._session.session_id}\n"
                f"Workspace: {self._session.workspace}\n"
                f"Process: {self._active_process_name()}\n"
                f"Turns: {state.turn_count}\n"
                f"Tokens: {state.total_tokens}\n"
                f"Focus: {state.current_focus or '(none)'}"
            )
            if state.key_decisions:
                info += "\nDecisions:"
                for d in state.key_decisions[-5:]:
                    info += f"\n  - {d}"
            chat.add_info(info)
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
            lines = ["[bold]Sessions:[/bold]"]
            for s in all_sessions[:10]:
                sid = s["id"]
                turns = s.get("turn_count", 0)
                started = s.get("started_at", "?")[:16]
                ws = s.get("workspace_path", "?")
                active = " [green](active)[/green]" if s.get("is_active") else ""
                current = " [cyan]<< current[/cyan]" if (
                    self._session and sid == self._session.session_id
                ) else ""
                lines.append(
                    f"  [dim]{started}[/dim] {turns} turns "
                    f"[dim]{sid[:12]}...[/dim]{active}{current}"
                )
                lines.append(f"    [dim]{ws}[/dim]")
            chat.add_info("\n".join(lines))
            chat.add_info(
                "[dim]To switch: type the session ID prefix after /resume[/dim]"
            )
            return True

        if token == "/resume":
            if not arg:
                chat.add_info("Usage: /resume <session-id-prefix>")
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
            chat.add_model_text(f"[bold #f7768e]Error:[/] {e}")
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
            chat.add_model_text(f"[bold #f7768e]Error:[/] {e}")
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
    def _one_line(text: object | None, max_len: int = 180) -> str:
        """Normalize whitespace and cap a string for concise progress rows."""
        if text is None:
            return ""
        compact = " ".join(str(text).split())
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
                phase_desc = self._one_line(getattr(phase, "description", ""), 110)
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
            candidate = self._one_line(row.get("content", ""), 140)
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
            return []
        if not isinstance(deliverables_by_phase, dict) or not deliverables_by_phase:
            return []

        subtask_status: dict[str, str] = {}
        for row in getattr(run, "tasks", []):
            if not isinstance(row, dict):
                continue
            subtask_id = str(row.get("id", "")).strip()
            if not subtask_id:
                continue
            subtask_status[subtask_id] = str(row.get("status", "pending")).strip()

        ordered_phase_ids: list[str] = []
        for phase in getattr(process, "phases", []):
            phase_id = str(getattr(phase, "id", "")).strip()
            if phase_id:
                ordered_phase_ids.append(phase_id)
        for phase_id in deliverables_by_phase.keys():
            if phase_id not in ordered_phase_ids:
                ordered_phase_ids.append(phase_id)

        rows: list[dict] = []
        for phase_id in ordered_phase_ids:
            phase_deliverables = deliverables_by_phase.get(phase_id) or []
            if not isinstance(phase_deliverables, list):
                continue
            phase_state = subtask_status.get(phase_id, "pending")
            if phase_state == "pending":
                continue
            run_workspace = getattr(run, "run_workspace", None)
            workspace_root = (
                Path(run_workspace) if run_workspace else self._workspace
            )
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
                    suffix = " [dim](pending)[/]"
                elif phase_state == "completed":
                    status = "failed"
                    suffix = " [#f7768e](missing)[/]"
                elif phase_state == "failed":
                    status = "failed"
                    suffix = " [#f7768e](not produced)[/]"
                elif phase_state == "skipped":
                    status = "skipped"
                    suffix = " [dim](skipped)[/]"
                else:
                    status = "pending"
                    suffix = ""
                rows.append({
                    "id": f"{phase_id}:{rel_path}",
                    "status": status,
                    "content": f"{rel_path} [dim]({phase_id})[/]{suffix}",
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
                    content = str(row.get("content", "")).strip()
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
                content = str(row.get("content", "")).strip()
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
                f"{message} Increase [execution].delegate_task_timeout_seconds "
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
        focus = self._one_line(primary.get("content", ""), 120)
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
        self.run_worker(
            self._close_process_run_from_target("current"),
            group="close-process-tab",
            exclusive=True,
        )

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
        input_widget = self.query_one("#user-input", Input)
        input_widget.value = text
        input_widget.cursor_position = len(text)
        input_widget.focus()
        self._set_slash_hint(self._render_slash_hint(text))

    def _show_tools(self) -> None:
        tools = self._tools.list_tools()
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(f"{len(tools)} tools: " + ", ".join(tools))

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
