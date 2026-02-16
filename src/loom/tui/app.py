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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Input,
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
    from loom.state.conversation_store import ConversationStore
    from loom.state.memory import Database


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
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
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
        self._session: CoworkSession | None = None
        self._busy = False
        self._total_tokens = 0

        # Approval state — resolved via Textual modal
        self._approval_event: asyncio.Event | None = None
        self._approval_result: ApprovalDecision = ApprovalDecision.DENY

        # Tools that need late-binding to session
        self._recall_tool = None
        self._delegate_tool = None

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

        await self._initialize_session()

    async def _initialize_session(self) -> None:
        """Initialize tools, session, and welcome message.

        Called from on_mount (normal start) or _finalize_setup (post-wizard).
        Requires self._model to be set.
        """
        chat = self.query_one("#chat-log", ChatLog)

        # Register extra tools if persistence is available
        if self._store is not None and self._recall_tool is None:
            from loom.tools.conversation_recall import ConversationRecallTool
            from loom.tools.delegate_task import DelegateTaskTool

            self._recall_tool = ConversationRecallTool()
            self._tools.register(self._recall_tool)

            self._delegate_tool = DelegateTaskTool()
            self._tools.register(self._delegate_tool)

        # Load process definition if specified
        process_defn = None
        if self._process_name and self._config:
            from loom.processes.schema import ProcessLoader

            extra = [Path(p) for p in self._config.process.search_paths]
            loader = ProcessLoader(
                workspace=self._workspace, extra_search_paths=extra,
            )
            try:
                process_defn = loader.load(self._process_name)
                chat.add_info(
                    f"Process: [bold]{process_defn.name}[/bold] "
                    f"v{process_defn.version}"
                )
                if process_defn.tools.excluded:
                    for tool_name in process_defn.tools.excluded:
                        if tool_name in self._tools._tools:
                            del self._tools._tools[tool_name]
            except Exception as e:
                chat.add_info(
                    f"[bold #f7768e]Failed to load process "
                    f"'{self._process_name}': {e}[/]"
                )

        # Build system prompt
        system_prompt = build_cowork_system_prompt(self._workspace)
        if process_defn:
            if process_defn.persona:
                system_prompt += (
                    f"\n\nDOMAIN ROLE:\n{process_defn.persona.strip()}"
                )
            if process_defn.tool_guidance:
                system_prompt += (
                    f"\n\nDOMAIN TOOL GUIDANCE:\n"
                    f"{process_defn.tool_guidance.strip()}"
                )

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
                    )

                self._delegate_tool.bind(_orchestrator_factory)
            except Exception:
                # delegate_task remains unbound; it will return a
                # "not available" message if the model tries to use it.
                pass

    # ------------------------------------------------------------------
    # Session management helpers
    # ------------------------------------------------------------------

    async def _new_session(self) -> None:
        """Create a fresh session, replacing the current one."""
        if self._store is None or self._session is None or self._model is None:
            return

        # Mark old session inactive
        await self._store.update_session(
            self._session.session_id, is_active=False,
        )

        system_prompt = build_cowork_system_prompt(self._workspace)
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

        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(f"New session: [dim]{session_id}[/dim]")

    async def _switch_to_session(self, session_id: str) -> None:
        """Resume a different session by ID."""
        if self._store is None or self._session is None or self._model is None:
            return

        old_id = self._session.session_id
        system_prompt = build_cowork_system_prompt(self._workspace)
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

    @on(Input.Submitted)
    async def on_user_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""

        # Handle slash commands
        if text.startswith("/"):
            handled = await self._handle_slash_command(text)
            if handled:
                return

        if self._busy:
            return

        self._run_turn(text)

    async def _handle_slash_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = text.strip().lower()
        chat = self.query_one("#chat-log", ChatLog)

        if cmd in ("/quit", "/exit", "/q"):
            if self._store and self._session and self._session.session_id:
                await self._store.update_session(
                    self._session.session_id, is_active=False,
                )
            self.exit()
            return True
        if cmd == "/clear":
            self.action_clear_chat()
            return True
        if cmd == "/help":
            lines = [
                "Commands: /quit, /clear, /model, /tools, /tokens, "
                "/learned, /setup, /help",
                "Keys: Ctrl+B sidebar, Ctrl+L clear, Ctrl+P palette, "
                "Ctrl+1/2/3 tabs",
            ]
            if self._store:
                lines.insert(1, "  /sessions — list and switch sessions")
                lines.insert(2, "  /new — start a new session")
                lines.insert(3, "  /session — current session info")
                lines.insert(4, "  /learned — review/delete learned patterns")
            chat.add_info("\n".join(lines))
            return True
        if cmd == "/model":
            name = self._model.name if self._model else "(not configured)"
            chat.add_info(f"Model: {name}")
            return True
        if cmd == "/setup":
            self.push_screen(
                SetupScreen(), callback=self._on_setup_complete,
            )
            return True
        if cmd == "/tools":
            tools = self._tools.list_tools()
            chat.add_info(
                f"{len(tools)} tools: " + ", ".join(tools)
            )
            return True
        if cmd == "/tokens":
            chat.add_info(f"Session tokens: {self._total_tokens:,}")
            return True

        # Persistence-dependent commands
        if cmd == "/session":
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

        if cmd == "/new":
            if self._store:
                await self._new_session()
            else:
                chat.add_info("No database — sessions are ephemeral.")
            return True

        if cmd == "/sessions":
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

        if cmd.startswith("/resume "):
            prefix = cmd.split(None, 1)[1].strip()
            if not self._store:
                chat.add_info("No database — sessions are ephemeral.")
                return True
            all_sessions = await self._store.list_sessions()
            match = None
            for s in all_sessions:
                if s["id"].startswith(prefix):
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

        if cmd == "/learned":
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
        events_panel = self.query_one("#events-panel", EventPanel)

        chat.add_user_message(user_message)
        status.state = "Thinking..."

        streamed_text = False

        try:
            async for event in self._session.send_streaming(
                user_message,
            ):
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

        except Exception as e:
            chat.add_model_text(f"[bold #f7768e]Error:[/] {e}")
            self.notify(str(e), severity="error", timeout=5)

        self._busy = False
        status.state = "Ready"

    async def _run_followup(self, message: str) -> None:
        """Run a follow-up turn (e.g. after ask_user answer)."""
        if self._session is None:
            return

        chat = self.query_one("#chat-log", ChatLog)
        events_panel = self.query_one("#events-panel", EventPanel)
        streamed_text = False

        try:
            async for event in self._session.send_streaming(message):
                if isinstance(event, str):
                    if not streamed_text:
                        streamed_text = True
                    chat.add_streaming_text(event)

                elif isinstance(event, ToolCallEvent):
                    if event.result is None:
                        chat.add_tool_call(event.name, event.args)
                        events_panel.add_event(
                            _now_str(), "tool_start",
                            (
                                f"{event.name} "
                                f"{tool_args_preview(event.name, event.args)}"
                            ),
                        )
                    else:
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
                        if (
                            event.name == "task_tracker"
                            and event.result.data
                        ):
                            self._update_sidebar_tasks(event)

                        # Handle ask_user in followup turns too
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
                    events_panel.record_turn_tokens(event.tokens_used)
                    self._update_files_panel(event)
        except Exception as e:
            chat.add_model_text(f"[bold #f7768e]Error:[/] {e}")
            self.notify(str(e), severity="error", timeout=5)

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

    def action_loom_command(self, command: str) -> None:
        """Dispatch command palette actions."""
        actions = {
            "clear_chat": self.action_clear_chat,
            "toggle_sidebar": self.action_toggle_sidebar,
            "tab_chat": self.action_tab_chat,
            "tab_files": self.action_tab_files,
            "tab_events": self.action_tab_events,
            "list_tools": self._show_tools,
            "model_info": self._show_model_info,
            "token_info": self._show_token_info,
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


def _now_str() -> str:
    """Return current time as HH:MM:SS string."""
    return datetime.now().strftime("%H:%M:%S")
