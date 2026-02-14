"""Loom TUI — interactive cowork session in a Textual app.

Multi-panel command center with sidebar, tabbed content area, rich
tool call rendering, and a polished dark theme. No server required —
runs CoworkSession directly.

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
from loom.tui.screens import AskUserScreen, ToolApprovalScreen
from loom.tui.theme import LOOM_DARK
from loom.tui.widgets import (
    ChatLog,
    EventPanel,
    FilesChangedPanel,
    Sidebar,
    StatusBar,
)
from loom.tui.widgets.tool_call import tool_args_preview


class LoomApp(App):
    """Loom TUI — interactive cowork session as a command center."""

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
        model: ModelProvider,
        tools: ToolRegistry,
        workspace: Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._tools = tools
        self._workspace = workspace
        self._session: CoworkSession | None = None
        self._busy = False
        self._total_tokens = 0

        # Approval state — resolved via Textual modal
        self._approval_event: asyncio.Event | None = None
        self._approval_result: ApprovalDecision = ApprovalDecision.DENY

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

        # Build session
        approver = ToolApprover(prompt_callback=self._approval_callback)
        system_prompt = build_cowork_system_prompt(self._workspace)
        self._session = CoworkSession(
            model=self._model,
            tools=self._tools,
            workspace=self._workspace,
            system_prompt=system_prompt,
            approver=approver,
        )

        # Configure status bar
        status = self.query_one("#status-bar", StatusBar)
        status.workspace_name = self._workspace.name
        status.model_name = self._model.name

        # Welcome message
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(
            f"[bold]Loom Cowork[/bold]  [dim]({self._model.name})[/dim]"
        )
        chat.add_info(f"workspace: {self._workspace}")
        tool_count = len(self._tools.list_tools())
        chat.add_info(f"{tool_count} tools loaded. Type your request.")

        self.query_one("#user-input", Input).focus()

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
            handled = self._handle_slash_command(text)
            if handled:
                return

        if self._busy:
            return

        self._run_turn(text)

    def _handle_slash_command(self, text: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        cmd = text.strip().lower()
        chat = self.query_one("#chat-log", ChatLog)

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
            return True
        if cmd == "/clear":
            self.action_clear_chat()
            return True
        if cmd == "/help":
            chat.add_info(
                "Commands: /quit, /clear, /model, /tools, /tokens, /help\n"
                "Keys: Ctrl+B sidebar, Ctrl+L clear, Ctrl+P palette, "
                "Ctrl+1/2/3 tabs"
            )
            return True
        if cmd == "/model":
            chat.add_info(f"Model: {self._model.name}")
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
        for tc in turn.tool_calls:
            if not tc.result or not tc.result.success:
                continue
            path = tc.args.get(
                "path", tc.args.get("file_path", "?"),
            )
            if tc.name == "write_file":
                file_entries.append({
                    "operation": "create",
                    "path": path,
                    "timestamp": _now_str(),
                })
            elif tc.name == "edit_file":
                file_entries.append({
                    "operation": "modify",
                    "path": path,
                    "timestamp": _now_str(),
                })
            elif tc.name == "delete_file":
                file_entries.append({
                    "operation": "delete",
                    "path": path,
                    "timestamp": _now_str(),
                })
        if file_entries:
            panel = self.query_one("#files-panel", FilesChangedPanel)
            panel.update_files(file_entries)
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
        chat.add_info(f"Model: {self._model.name}")

    def _show_token_info(self) -> None:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(f"Session tokens: {self._total_tokens:,}")


def _now_str() -> str:
    """Return current time as HH:MM:SS string."""
    return datetime.now().strftime("%H:%M:%S")
