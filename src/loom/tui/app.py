"""Loom TUI — interactive cowork session in a Textual app.

Replaces the old task-dashboard TUI with a conversation-first interface
that uses CoworkSession directly. No server required.

Layout:
  ┌──────────────────────────────────────────┐
  │ Header                                   │
  ├──────────────────────────────────────────┤
  │ Chat log (RichLog)                       │
  │   streaming text, tool calls, results    │
  │                                          │
  ├──────────────────────────────────────────┤
  │ [>] Input bar                            │
  ├──────────────────────────────────────────┤
  │ Footer (keybindings)                     │
  └──────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, RichLog, Static

from loom.cowork.approval import ApprovalDecision, ToolApprover
from loom.cowork.session import (
    CoworkSession,
    CoworkTurn,
    ToolCallEvent,
    build_cowork_system_prompt,
)
from loom.models.base import ModelProvider
from loom.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class ToolApprovalScreen(ModalScreen[str]):
    """Prompt the user to approve a tool call.

    Returns: "approve", "approve_all", or "deny"
    """

    BINDINGS = [
        Binding("y", "approve", "Yes"),
        Binding("a", "approve_all", "Always"),
        Binding("n", "deny", "No"),
        Binding("escape", "deny", "Cancel"),
    ]

    CSS = """
    #approval-dialog {
        align: center middle;
        width: 70;
        height: auto;
        max-height: 12;
        border: solid $warning;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self, tool_name: str, args_preview: str) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args_preview = args_preview

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("[bold yellow]Approve tool call?[/bold yellow]"),
            Label(f"[bold cyan]{self._tool_name}[/bold cyan]  [dim]{self._args_preview}[/dim]"),
            Label(""),
            Label("[y] Yes  [a] Always allow this tool  [n] No  [Esc] Cancel"),
            id="approval-dialog",
        )

    def action_approve(self) -> None:
        self.dismiss("approve")

    def action_approve_all(self) -> None:
        self.dismiss("approve_all")

    def action_deny(self) -> None:
        self.dismiss("deny")


class AskUserScreen(ModalScreen[str]):
    """Display a question from the model and collect the user's answer."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    CSS = """
    #ask-user-dialog {
        align: center middle;
        width: 70;
        height: auto;
        max-height: 16;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #ask-user-input { margin-top: 1; }
    """

    def __init__(self, question: str, options: list[str] | None = None) -> None:
        super().__init__()
        self._question = question
        self._options = options or []

    def compose(self) -> ComposeResult:
        children = [
            Label(f"[bold yellow]Question:[/bold yellow] {self._question}"),
        ]
        for i, opt in enumerate(self._options, 1):
            children.append(Label(f"  [cyan]{i}.[/cyan] {opt}"))
        if self._options:
            children.append(Label("[dim]Enter a number or type your answer[/dim]"))
        children.append(Input(placeholder="Your answer...", id="ask-user-input"))

        yield Vertical(*children, id="ask-user-dialog")

    def on_mount(self) -> None:
        self.query_one("#ask-user-input", Input).focus()

    @on(Input.Submitted)
    def on_submit(self, event: Input.Submitted) -> None:
        answer = event.value.strip()
        if not answer:
            return
        # Map number to option
        if self._options and answer.isdigit():
            idx = int(answer) - 1
            if 0 <= idx < len(self._options):
                answer = self._options[idx]
        self.dismiss(answer)

    def action_cancel(self) -> None:
        self.dismiss("")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class LoomApp(App):
    """Loom TUI — interactive cowork session."""

    TITLE = "Loom"
    CSS = """
    #chat-log {
        height: 1fr;
        border: solid $primary;
        scrollbar-size: 1 1;
    }
    #status-bar {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    #user-input {
        dock: bottom;
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True, priority=True),
        Binding("ctrl+l", "clear_log", "Clear"),
    ]

    def __init__(
        self,
        model: ModelProvider,
        tools: ToolRegistry,
        workspace: Path,
        *,
        server_url: str | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._tools = tools
        self._workspace = workspace
        self._session: CoworkSession | None = None
        self._busy = False

        # Approval state — resolved via Textual modal
        self._approval_event: asyncio.Event | None = None
        self._approval_result: ApprovalDecision = ApprovalDecision.DENY

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
        yield Input(placeholder="Type a message... (Enter to send)", id="user-input")
        yield Static("", id="status-bar")
        yield Footer()

    async def on_mount(self) -> None:
        approver = ToolApprover(prompt_callback=self._approval_callback)
        system_prompt = build_cowork_system_prompt(self._workspace)
        self._session = CoworkSession(
            model=self._model,
            tools=self._tools,
            workspace=self._workspace,
            system_prompt=system_prompt,
            approver=approver,
        )

        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold]Loom Cowork[/bold] [dim]({self._model.name})[/dim]")
        log.write(f"[dim]workspace: {self._workspace}[/dim]")
        log.write("[dim]16 tools loaded. Type your request.[/dim]")
        log.write("")

        self._update_status("Ready")
        self.query_one("#user-input", Input).focus()

    # ------------------------------------------------------------------
    # Approval callback (bridged into Textual modal)
    # ------------------------------------------------------------------

    async def _approval_callback(self, tool_name: str, args: dict) -> ApprovalDecision:
        """Called by ToolApprover when a tool needs user permission.

        Shows a modal and waits for the user's response.
        """
        from loom.cowork.approval import _format_args_preview

        preview = _format_args_preview(tool_name, args)

        # Set up an event to wait on
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

        # Wait for the modal to be dismissed
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

        if text.lower() in ("/quit", "/exit", "/q"):
            self.exit()
            return

        if text.lower() == "/clear":
            self.query_one("#chat-log", RichLog).clear()
            return

        if self._busy:
            return

        self._run_turn(text)

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    @work(exclusive=True)
    async def _run_turn(self, user_message: str) -> None:
        if self._session is None:
            return

        self._busy = True
        log = self.query_one("#chat-log", RichLog)

        # Show user message
        log.write(f"[bold green]> {user_message}[/bold green]")
        log.write("")

        self._update_status("Thinking...")

        streamed_text = False

        try:
            async for event in self._session.send_streaming(user_message):
                if isinstance(event, str):
                    # Streamed text token
                    if not streamed_text:
                        streamed_text = True
                    log.write(event, shrink=False, scroll_end=True)

                elif isinstance(event, ToolCallEvent):
                    if event.result is None:
                        # Tool starting
                        args_preview = _tool_args_preview(event.name, event.args)
                        log.write(
                            f"  [dim]{event.name}[/dim] [dim]{args_preview}[/dim]",
                        )
                        self._update_status(f"Running {event.name}...")

                        # Handle ask_user specially
                        if event.name == "ask_user":
                            pass  # will handle on completion
                    else:
                        # Tool completed
                        if event.result.success:
                            elapsed = f"{event.elapsed_ms}ms" if event.elapsed_ms else ""
                            preview = _tool_output_preview(event.name, event.result.output)
                            log.write(
                                f"  [green]ok[/green] [dim]{elapsed} {preview}[/dim]",
                            )
                        else:
                            error = event.result.error or ""
                            log.write(f"  [red]err[/red] [dim]{error[:80]}[/dim]")

                        # If ask_user completed, show question and get answer
                        if event.name == "ask_user" and event.result and event.result.success:
                            answer = await self._handle_ask_user(event)
                            if answer:
                                # Send answer as a follow-up turn
                                await self._run_followup(answer)

                elif isinstance(event, CoworkTurn):
                    # Turn complete
                    if event.text and not streamed_text:
                        log.write(f"\n{event.text}")

                    n = len(event.tool_calls)
                    if n:
                        log.write(
                            f"\n[dim][{n} tool call{'s' if n != 1 else ''}"
                            f" | {event.tokens_used} tokens"
                            f" | {event.model}][/dim]"
                        )
                    log.write("")

        except Exception as e:
            log.write(f"[bold red]Error:[/bold red] {e}")

        self._busy = False
        self._update_status("Ready")

    async def _run_followup(self, message: str) -> None:
        """Run a follow-up turn (e.g. after ask_user answer)."""
        if self._session is None:
            return

        log = self.query_one("#chat-log", RichLog)
        streamed_text = False

        async for event in self._session.send_streaming(message):
            if isinstance(event, str):
                if not streamed_text:
                    streamed_text = True
                log.write(event, shrink=False, scroll_end=True)
            elif isinstance(event, ToolCallEvent):
                if event.result is None:
                    args_preview = _tool_args_preview(event.name, event.args)
                    log.write(f"  [dim]{event.name}[/dim] [dim]{args_preview}[/dim]")
                else:
                    if event.result.success:
                        elapsed = f"{event.elapsed_ms}ms" if event.elapsed_ms else ""
                        preview = _tool_output_preview(event.name, event.result.output)
                        log.write(f"  [green]ok[/green] [dim]{elapsed} {preview}[/dim]")
                    else:
                        error = event.result.error or ""
                        log.write(f"  [red]err[/red] [dim]{error[:80]}[/dim]")
            elif isinstance(event, CoworkTurn):
                if event.text and not streamed_text:
                    log.write(f"\n{event.text}")
                if event.tool_calls:
                    n = len(event.tool_calls)
                    log.write(
                        f"\n[dim][{n} tool call{'s' if n != 1 else ''}"
                        f" | {event.tokens_used} tokens][/dim]"
                    )
                log.write("")

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
            log = self.query_one("#chat-log", RichLog)
            log.write(f"[bold green]> {answer}[/bold green]")

        return answer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_status(self, text: str) -> None:
        bar = self.query_one("#status-bar", Static)
        bar.update(f"[dim]{self._workspace}  |  {text}[/dim]")

    def action_clear_log(self) -> None:
        self.query_one("#chat-log", RichLog).clear()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _tool_args_preview(tool_name: str, args: dict) -> str:
    """Short preview of tool arguments for the chat log."""
    if tool_name in ("read_file", "write_file", "edit_file", "delete_file"):
        return _trunc(args.get("path", args.get("file_path", "")), 60)
    if tool_name == "shell_execute":
        return _trunc(args.get("command", ""), 80)
    if tool_name == "git_command":
        return _trunc(" ".join(args.get("args", [])), 60)
    if tool_name in ("ripgrep_search", "search_files"):
        return _trunc(f"/{args.get('pattern', '')}/", 60)
    if tool_name == "glob_find":
        return _trunc(args.get("pattern", ""), 60)
    if tool_name in ("web_fetch", "web_search"):
        return _trunc(args.get("url", args.get("query", "")), 60)
    if tool_name == "task_tracker":
        action = args.get("action", "")
        content = args.get("content", "")
        return _trunc(f"{action}: {content}" if content else action, 60)
    if tool_name == "ask_user":
        return _trunc(args.get("question", ""), 60)
    if tool_name == "analyze_code":
        return _trunc(args.get("path", ""), 60)
    for v in args.values():
        if isinstance(v, str) and v:
            return _trunc(v, 50)
    return ""


def _tool_output_preview(tool_name: str, output: str) -> str:
    """Short preview of tool output for the chat log."""
    if not output:
        return ""
    if tool_name in ("ripgrep_search", "search_files", "glob_find"):
        lines = output.strip().split("\n")
        if lines and ("No matches" in lines[0] or "No files" in lines[0]):
            return lines[0]
        return f"{len(lines)} results"
    if tool_name == "read_file":
        return f"{len(output.splitlines())} lines"
    if tool_name == "shell_execute":
        return _trunc(output.strip().split("\n")[0], 60)
    if tool_name == "web_search":
        lines = [x for x in output.strip().split("\n") if x.startswith(("1.", "2.", "3."))]
        return f"{len(lines)} results" if lines else ""
    return ""


def _trunc(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
