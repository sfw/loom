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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
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
from loom.tools.registry import ToolRegistry
from loom.tui.commands import LoomCommands
from loom.tui.screens import (
    AskUserScreen,
    ExitConfirmScreen,
    LearnedScreen,
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
)
_MAX_SLASH_HINT_LINES = 14


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
    #user-input {
        dock: bottom;
        margin: 0;
    }
    #slash-hint {
        dock: bottom;
        display: none;
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
        self._busy = False
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
        yield Input(
            placeholder="Type a message... (Enter to send)",
            id="user-input",
        )
        yield Static("", id="slash-hint")
        yield StatusBar(id="status-bar")
        yield Footer()

    async def on_mount(self) -> None:
        # Register and activate theme
        self.register_theme(LOOM_DARK)
        self.theme = "loom-dark"

        if self._model is None:
            # No model configured — launch the setup wizard
            self.push_screen(
                SetupScreen(), callback=self._on_setup_complete,
            )
            return

        await self._initialize_session()

    def _on_setup_complete(self, result: list[dict] | None) -> None:
        """Handle setup wizard dismissal."""
        if result is None:
            self.exit()
            return
        self._finalize_setup()

    @work
    async def _finalize_setup(self) -> None:
        """Reload config and initialize after setup wizard completes."""
        from loom.config import load_config
        from loom.models.router import ModelRouter

        self._config = load_config()
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

        self._tools = create_default_registry()

    def _load_process_definition(self, chat: ChatLog) -> None:
        """Load active process definition and import bundled tools."""
        self._process_defn = None
        if not self._process_name or not self._config:
            return

        from loom.processes.schema import ProcessLoader

        extra = [Path(p) for p in self._config.process.search_paths]
        loader = ProcessLoader(
            workspace=self._workspace,
            extra_search_paths=extra,
        )
        try:
            self._process_defn = loader.load(self._process_name)
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

    async def _initialize_session(self) -> None:
        """Initialize tools, session, and welcome message.

        Called from on_mount (normal start) or _finalize_setup (post-wizard).
        Requires self._model to be set.
        """
        chat = self.query_one("#chat-log", ChatLog)

        # Start from a clean registry each initialization to avoid stale
        # excludes/bindings when setup or process configuration changes.
        self._refresh_tool_registry()

        # Load process definition (imports bundled tools, then refreshes).
        self._load_process_definition(chat)

        # Register extra tools if persistence is available
        if self._store is not None:
            from loom.tools.conversation_recall import ConversationRecallTool
            from loom.tools.delegate_task import DelegateTaskTool

            if self._recall_tool is None:
                self._recall_tool = ConversationRecallTool()
            if not self._tools.has(self._recall_tool.name):
                self._tools.register(self._recall_tool)

            if self._delegate_tool is None:
                self._delegate_tool = DelegateTaskTool()
            if not self._tools.has(self._delegate_tool.name):
                self._tools.register(self._delegate_tool)

        # Build system prompt
        self._apply_process_tool_policy(chat)
        system_prompt = self._build_system_prompt()

        # Build approver
        approver = ToolApprover(prompt_callback=self._approval_callback)

        # Create or resume session
        if self._store is not None and self._resume_session:
            # Resume existing session
            self._session = CoworkSession(
                model=self._model,
                tools=self._tools,
                workspace=self._workspace,
                system_prompt=system_prompt,
                approver=approver,
                store=self._store,
            )
            try:
                await self._session.resume(self._resume_session)
                self._total_tokens = self._session.total_tokens
                chat.add_info(
                    f"Resumed session [dim]{self._resume_session}[/dim] "
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
            )
        else:
            # Ephemeral session (no database)
            self._session = CoworkSession(
                model=self._model,
                tools=self._tools,
                workspace=self._workspace,
                system_prompt=system_prompt,
                approver=approver,
            )

        # Bind session-dependent tools
        self._bind_session_tools()

        # Configure status bar
        status = self.query_one("#status-bar", StatusBar)
        status.workspace_name = self._workspace.name
        status.model_name = self._model.name

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

        self.query_one("#user-input", Input).focus()

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
        lines.append(
            "Keys: Ctrl+B sidebar, Ctrl+L clear, Ctrl+P palette, Ctrl+1/2/3 tabs",
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
        if matches:
            return token, matches
        return token, fallback_matches

    def _reset_slash_tab_cycle(self) -> None:
        """Clear slash tab-completion cycle state."""
        self._slash_cycle_seed = ""
        self._slash_cycle_candidates = []

    def _slash_completion_candidates(self, token: str) -> list[str]:
        """Return slash command completions for a token prefix."""
        if token == "/":
            return [spec.canonical for spec in _SLASH_COMMANDS]

        candidates: list[str] = []
        seen: set[str] = set()
        for spec in _SLASH_COMMANDS:
            for key in (spec.canonical, *spec.aliases):
                if key.startswith(token) and key not in seen:
                    candidates.append(key)
                    seen.add(key)
        return candidates

    def _apply_slash_tab_completion(self, *, reverse: bool = False) -> bool:
        """Apply slash tab completion (forward/backward)."""
        input_widget = self.query_one("#user-input", Input)
        token = input_widget.value.strip()
        if not token.startswith("/") or " " in token:
            self._reset_slash_tab_cycle()
            return False

        if token in self._slash_cycle_candidates:
            candidates = self._slash_cycle_candidates
            current_index = candidates.index(token)
            next_index = (
                (current_index - 1) if reverse else (current_index + 1)
            ) % len(candidates)
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

    def _set_slash_hint(self, hint_text: str) -> None:
        """Show or hide the slash-command hint panel."""
        hint = self.query_one("#slash-hint", Static)
        footer: Footer | None = None
        status: StatusBar | None = None
        try:
            footer = self.query_one(Footer)
        except Exception:
            footer = None
        try:
            status = self.query_one("#status-bar", StatusBar)
        except Exception:
            status = None
        if hint_text:
            hint.update(hint_text)
            hint.display = True
            line_count = max(1, len(hint_text.splitlines()))
            hint.styles.height = min(line_count, _MAX_SLASH_HINT_LINES)
            hint.scroll_home(animate=False)
            if footer is not None:
                footer.display = False
            if status is not None:
                status.display = False
        else:
            hint.display = False
            hint.styles.height = "auto"
            hint.update("")
            if footer is not None:
                footer.display = True
            if status is not None:
                status.display = True

    def _bind_session_tools(self) -> None:
        """Bind tools that hold a reference to the active session."""
        if self._session is None:
            return
        if self._recall_tool and self._store:
            self._recall_tool.bind(
                store=self._store,
                session_id=self._session.session_id,
                session_state=self._session.session_state,
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

                async def _orchestrator_factory():
                    return Orchestrator(
                        model_router=router,
                        tool_registry=_create_tools(),
                        memory_manager=MemoryManager(db),
                        prompt_assembler=PromptAssembler(),
                        state_manager=TaskStateManager(data_dir),
                        event_bus=EventBus(),
                        config=config,
                        process=self._process_defn,
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

        # Mark old session inactive
        await self._store.update_session(
            self._session.session_id, is_active=False,
        )

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
        )
        self._total_tokens = 0
        self._bind_session_tools()
        self._clear_files_panel()

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(f"New session: [dim]{session_id}[/dim]")

    async def _switch_to_session(self, session_id: str) -> None:
        """Resume a different session by ID."""
        if self._store is None or self._session is None or self._model is None:
            return

        old_id = self._session.session_id
        system_prompt = self._build_system_prompt()
        approver = ToolApprover(prompt_callback=self._approval_callback)

        new_session = CoworkSession(
            model=self._model,
            tools=self._tools,
            workspace=self._workspace,
            system_prompt=system_prompt,
            approver=approver,
            store=self._store,
        )
        await new_session.resume(session_id)

        # Mark old session inactive
        await self._store.update_session(old_id, is_active=False)

        self._session = new_session
        self._total_tokens = new_session.total_tokens
        self._bind_session_tools()
        self._clear_files_panel()

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(
            f"Switched to session [dim]{session_id}[/dim] "
            f"({new_session.session_state.turn_count} turns)"
        )

    # ------------------------------------------------------------------
    # Learned patterns
    # ------------------------------------------------------------------

    async def _show_learned_patterns(self) -> None:
        """Show the learned patterns review modal."""
        from loom.learning.manager import LearningManager

        mgr = LearningManager(self._db)
        patterns = await mgr.query_all(limit=50)

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

        if self._busy:
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
        """Handle slash tab-completion from the user input."""
        if event.key not in ("tab", "shift+tab"):
            return
        focused = self.focused
        if not isinstance(focused, Input) or focused.id != "user-input":
            return
        if self._apply_slash_tab_completion(reverse=event.key == "shift+tab"):
            event.stop()
            event.prevent_default()

    async def _handle_slash_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        raw = text.strip()
        parts = raw.split(None, 1)
        token = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        chat = self.query_one("#chat-log", ChatLog)

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

        # Persistence-dependent commands
        if token == "/session":
            if not self._session:
                chat.add_info("No active session.")
                return True
            state = self._session.session_state
            info = (
                f"Session: {self._session.session_id}\n"
                f"Workspace: {self._session.workspace}\n"
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

        self._busy = True
        chat = self.query_one("#chat-log", ChatLog)
        status = self.query_one("#status-bar", StatusBar)

        chat.add_user_message(user_message)
        status.state = "Thinking..."

        try:
            await self._run_interaction(user_message)
        except Exception as e:
            chat.add_model_text(f"[bold #f7768e]Error:[/] {e}")
            self.notify(str(e), severity="error", timeout=5)

        self._busy = False
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

                    # Update sidebar tasks if task_tracker
                    if (
                        event.name == "task_tracker"
                        and event.result.data
                    ):
                        self._update_sidebar_tasks(event)

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

    def _update_sidebar_tasks(self, event: ToolCallEvent) -> None:
        """Update sidebar task progress from task_tracker result."""
        if not event.result or not event.result.data:
            return
        data = event.result.data
        tasks = data.get("tasks", [])
        if not tasks and "id" in data:
            tasks = [data]
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.update_tasks(tasks)

    def _update_files_panel(self, turn: CoworkTurn) -> None:
        """Update the Files Changed panel from tool call events."""
        file_entries: list[dict] = []
        last_diff = ""
        for tc in turn.tool_calls:
            if not tc.result or not tc.result.success:
                continue
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
            count = len(file_entries)
            s = "s" if count != 1 else ""
            self.notify(
                f"{count} file{s} changed", timeout=3,
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_toggle_sidebar(self) -> None:
        self.query_one("#sidebar", Sidebar).toggle()

    def action_clear_chat(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        for child in list(chat.children):
            child.remove()

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
        if self._store and self._session and self._session.session_id:
            await self._store.update_session(
                self._session.session_id, is_active=False,
            )
        self.exit()

    async def action_loom_command(self, command: str) -> None:
        """Dispatch command palette actions."""
        if command == "quit":
            self.action_request_quit()
            return
        actions = {
            "clear_chat": self.action_clear_chat,
            "toggle_sidebar": self.action_toggle_sidebar,
            "tab_chat": self.action_tab_chat,
            "tab_files": self.action_tab_files,
            "tab_events": self.action_tab_events,
            "list_tools": self._show_tools,
            "model_info": self._show_model_info,
            "token_info": self._show_token_info,
            "help": self._show_help,
        }
        action_fn = actions.get(command)
        if action_fn:
            action_fn()

    def _show_tools(self) -> None:
        tools = self._tools.list_tools()
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(f"{len(tools)} tools: " + ", ".join(tools))

    def _show_model_info(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        name = self._model.name if self._model else "(not configured)"
        chat.add_info(f"Model: {name}")

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
