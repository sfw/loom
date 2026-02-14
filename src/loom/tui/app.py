"""Loom TUI application built with Textual.

A task monitoring dashboard with live event streaming,
token display, file viewer, memory inspector, and conversation mode.
"""

from __future__ import annotations

import asyncio

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    TextArea,
    Tree,
)

from loom.tui.api_client import LoomAPIClient

# --- Custom Messages ---


class TaskEventMessage(Message):
    """SSE event received from the server."""

    def __init__(self, event: dict) -> None:
        super().__init__()
        self.event = event


# --- Screens ---


class SteerScreen(ModalScreen[str | None]):
    """Modal for entering steering instructions."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Enter instruction for the running task:", id="steer-label"),
            Input(placeholder="e.g., Use TypeScript instead of JavaScript", id="steer-input"),
            Label("Press Enter to send, Escape to cancel", id="steer-hint"),
            id="steer-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#steer-input", Input).focus()

    @on(Input.Submitted)
    def on_submit(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class NewTaskScreen(ModalScreen[dict | None]):
    """Multi-field form for creating a new task."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "submit", "Submit"),
    ]

    CSS = """
    #new-task-dialog {
        align: center middle;
        width: 70;
        height: auto;
        max-height: 24;
        border: solid $primary;
        padding: 1 2;
    }
    #goal-input { margin-bottom: 1; }
    #workspace-input { margin-bottom: 1; }
    #context-input { height: 4; margin-bottom: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("[bold]Create New Task[/bold]", id="new-task-title"),
            Label("Goal (required):", id="goal-label"),
            Input(placeholder="What should be accomplished?", id="goal-input"),
            Label("Workspace path (optional):", id="workspace-label"),
            Input(placeholder="/path/to/project", id="workspace-input"),
            Label("Approval mode:", id="mode-label"),
            Select(
                [(label, val) for label, val in [
                    ("Auto", "auto"), ("Gate all", "gate_all"),
                    ("Gate critical", "gate_critical"),
                ]],
                value="auto",
                id="mode-select",
            ),
            Label("Additional context (optional):", id="context-label"),
            TextArea(id="context-input"),
            Label("[Ctrl+S] Submit  [Esc] Cancel", id="new-task-hint"),
            id="new-task-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#goal-input", Input).focus()

    def action_submit(self) -> None:
        goal = self.query_one("#goal-input", Input).value.strip()
        if not goal:
            return
        workspace = self.query_one("#workspace-input", Input).value.strip() or None
        mode = self.query_one("#mode-select", Select).value
        context_text = self.query_one("#context-input", TextArea).text.strip()
        context = {"user_notes": context_text} if context_text else {}

        self.dismiss({
            "goal": goal,
            "workspace": workspace,
            "approval_mode": mode,
            "context": context,
        })

    def action_cancel(self) -> None:
        self.dismiss(None)


class ApprovalScreen(ModalScreen[bool]):
    """Modal for approving or rejecting a subtask."""

    BINDINGS = [
        Binding("y", "approve", "Approve"),
        Binding("n", "reject", "Reject"),
        Binding("escape", "reject", "Cancel"),
    ]

    def __init__(self, task_id: str, subtask_id: str, reason: str, risk_level: str):
        super().__init__()
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.reason = reason
        self.risk_level = risk_level

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(f"Approval Required — Risk: {self.risk_level}", id="approval-title"),
            Label(f"Task: {self.task_id}", id="approval-task"),
            Label(f"Subtask: {self.subtask_id}", id="approval-subtask"),
            Label(f"Reason: {self.reason}", id="approval-reason"),
            Label("[y] Approve  [n] Reject  [Esc] Cancel", id="approval-hint"),
            id="approval-dialog",
        )

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)


class FeedbackScreen(ModalScreen[dict | None]):
    """Modal for submitting feedback after task completion."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "submit", "Submit"),
    ]

    CSS = """
    #feedback-dialog {
        align: center middle;
        width: 60;
        height: auto;
        border: solid $success;
        padding: 1 2;
    }
    #feedback-input { height: 5; margin-bottom: 1; }
    """

    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(f"[bold]Feedback for {self.task_id}[/bold]"),
            Label("Rating (1-5):"),
            Select(
                [(str(i), i) for i in range(1, 6)],
                value=3,
                id="rating-select",
            ),
            Label("Comments:"),
            TextArea(id="feedback-input"),
            Label("[Ctrl+S] Submit  [Esc] Cancel"),
            id="feedback-dialog",
        )

    def action_submit(self) -> None:
        rating = self.query_one("#rating-select", Select).value
        comment = self.query_one("#feedback-input", TextArea).text.strip()
        self.dismiss({"rating": rating, "comment": comment, "task_id": self.task_id})

    def action_cancel(self) -> None:
        self.dismiss(None)


class MemoryInspectorScreen(ModalScreen[None]):
    """Screen for inspecting task memory entries."""

    BINDINGS = [Binding("escape", "dismiss_screen", "Close")]

    CSS = """
    #memory-dialog {
        align: center middle;
        width: 80;
        height: 20;
        border: solid $accent;
        padding: 1 2;
    }
    #memory-log {
        height: 1fr;
    }
    """

    def __init__(self, entries: list[dict]):
        super().__init__()
        self._entries = entries

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("[bold]Memory Inspector[/bold]"),
            RichLog(id="memory-log", highlight=True, markup=True),
            Label("[Esc] Close"),
            id="memory-dialog",
        )

    def on_mount(self) -> None:
        log = self.query_one("#memory-log", RichLog)
        if not self._entries:
            log.write("[dim]No memory entries found.[/dim]")
            return
        for entry in self._entries:
            entry_type = entry.get("entry_type", "?")
            summary = entry.get("summary", "")
            detail = entry.get("detail", "")
            tags = entry.get("tags", "")
            ts = entry.get("timestamp", "")[:19]
            log.write(f"[bold cyan][{entry_type}][/bold cyan] {summary}")
            if detail and detail != summary:
                log.write(f"  [dim]{detail[:200]}[/dim]")
            if tags:
                log.write(f"  [dim]tags: {tags}[/dim]")
            if ts:
                log.write(f"  [dim]{ts}[/dim]")
            log.write("")

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)


class ConversationScreen(ModalScreen[str | None]):
    """Chat-like interface for interactive conversation with a running task."""

    BINDINGS = [Binding("escape", "cancel", "Close")]

    CSS = """
    #conversation-dialog {
        align: center middle;
        width: 80;
        height: 22;
        border: solid $primary;
        padding: 1 2;
    }
    #conversation-log {
        height: 1fr;
        margin-bottom: 1;
    }
    """

    def __init__(self, task_id: str, history: list[dict]):
        super().__init__()
        self.task_id = task_id
        self._history = history

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label(f"[bold]Conversation — {self.task_id}[/bold]"),
            RichLog(id="conversation-log", highlight=True, markup=True),
            Input(placeholder="Type a message... (Enter to send)", id="conversation-input"),
            Label("[Enter] Send  [Esc] Close"),
            id="conversation-dialog",
        )

    def on_mount(self) -> None:
        log = self.query_one("#conversation-log", RichLog)
        for entry in self._history:
            msg = entry.get("message", "")
            tags = entry.get("tags", "")
            ts = entry.get("timestamp", "")[:19]
            if "conversation" in tags or "steer" in tags:
                label = "[bold yellow]You:[/bold yellow]"
            else:
                label = "[bold cyan]System:[/bold cyan]"
            log.write(f"{label} {msg}  [dim]{ts}[/dim]")
        self.query_one("#conversation-input", Input).focus()

    @on(Input.Submitted)
    def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text:
            self.dismiss(text)

    def action_cancel(self) -> None:
        self.dismiss(None)


# --- Main Application ---


class LoomApp(App):
    """Loom TUI — task monitoring dashboard."""

    CSS = """
    #task-table {
        height: 1fr;
        min-height: 8;
    }
    #detail-container {
        height: 3fr;
    }
    #plan-tree {
        width: 1fr;
        min-width: 30;
        border: solid $primary;
    }
    #live-output {
        width: 2fr;
        border: solid $secondary;
    }
    #files-changed {
        height: auto;
        max-height: 6;
        border: solid $accent;
    }
    #steer-dialog {
        align: center middle;
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }
    #approval-dialog {
        align: center middle;
        width: 60;
        height: auto;
        border: solid $warning;
        padding: 1 2;
    }
    #status-bar {
        height: 1;
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "steer", "Steer"),
        Binding("a", "approve_action", "Approve"),
        Binding("c", "cancel_task", "Cancel"),
        Binding("n", "new_task", "New Task"),
        Binding("m", "show_memory", "Memory"),
        Binding("f", "show_feedback", "Feedback"),
        Binding("t", "show_conversation", "Chat"),
    ]

    def __init__(self, server_url: str = "http://localhost:9000"):
        super().__init__()
        self.server_url = server_url
        self.api = LoomAPIClient(server_url)
        self.selected_task_id: str | None = None
        self._pending_approval: dict | None = None
        self._streaming_task: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            DataTable(id="task-table"),
            Horizontal(
                Tree("Plan", id="plan-tree"),
                RichLog(id="live-output", highlight=True, markup=True),
                id="detail-container",
            ),
            Static("", id="files-changed"),
            Static(f"Connected to {self.server_url}", id="status-bar"),
        )
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the task table and start background refresh."""
        table = self.query_one("#task-table", DataTable)
        table.add_columns("ID", "Status", "Goal", "Progress")
        table.cursor_type = "row"
        await self._refresh_tasks()

    @work(exclusive=True)
    async def _refresh_tasks(self) -> None:
        """Fetch tasks from the API and update the table."""
        try:
            tasks = await self.api.list_tasks()
            table = self.query_one("#task-table", DataTable)
            table.clear()
            for t in tasks:
                status_icon = {
                    "running": "●",
                    "executing": "●",
                    "completed": "✓",
                    "failed": "✗",
                    "waiting_approval": "⏸",
                    "cancelled": "○",
                    "pending": "○",
                    "planning": "…",
                }.get(t.get("status", ""), "?")
                goal = t.get("goal", "")[:50]
                progress = ""
                if "progress" in t:
                    p = t["progress"]
                    pct = p.get("percent_complete", 0)
                    progress = f"{p.get('completed', 0)}/{p.get('total_subtasks', 0)} {pct:.0f}%"
                table.add_row(
                    t.get("task_id", ""),
                    f"{status_icon} {t.get('status', '')}",
                    goal,
                    progress,
                    key=t.get("task_id", ""),
                )
            self.query_one("#status-bar", Static).update(
                f"Connected to {self.server_url} — {len(tasks)} tasks"
            )
        except Exception as e:
            self.query_one("#status-bar", Static).update(f"Error: {e}")

    @on(DataTable.RowSelected)
    async def on_row_selected(self, event: DataTable.RowSelected) -> None:
        """When a task row is selected, show its details."""
        if event.row_key and event.row_key.value:
            self.selected_task_id = str(event.row_key.value)
            await self._show_task_detail(self.selected_task_id)
            self._start_event_stream(self.selected_task_id)

    @work(exclusive=True)
    async def _show_task_detail(self, task_id: str) -> None:
        """Fetch and display task detail: plan tree, live output."""
        try:
            task = await self.api.get_task(task_id)
            subtasks = await self.api.get_subtasks(task_id)
        except Exception:
            return

        # Update plan tree
        tree = self.query_one("#plan-tree", Tree)
        tree.clear()
        tree.root.label = f"Plan — {task.get('goal', '')[:40]}"
        for s in subtasks:
            icon = {
                "completed": "✓",
                "running": "→",
                "failed": "✗",
                "pending": "○",
                "blocked": "⏸",
                "skipped": "–",
            }.get(s.get("status", ""), "?")
            desc = s.get('description', '')[:40]
            sid = s.get('id', '')
            status = s.get('status', '')
            tree.root.add_leaf(f"{icon} [{status}] {sid}: {desc}")
        tree.root.expand()

        # Update files changed
        progress = task.get("progress", {})
        done = progress.get('completed', 0)
        total = progress.get('total_subtasks', 0)
        files_info = f"Subtasks: {done}/{total} completed"
        if progress.get("failed", 0):
            files_info += f" | {progress['failed']} failed"
        self.query_one("#files-changed", Static).update(files_info)

    # --- Live SSE Event Streaming ---

    @work(exclusive=True, group="event-stream")
    async def _start_event_stream(self, task_id: str) -> None:
        """Background worker that subscribes to task SSE events."""
        self._streaming_task = task_id
        log = self.query_one("#live-output", RichLog)
        log.clear()
        log.write(f"[dim]Streaming events for task {task_id}...[/dim]")

        try:
            async for event_data in self.api.stream_task_events(task_id):
                event_type = event_data.get("event_type", event_data.get("type", ""))
                await self._handle_sse_event(event_type, event_data)

                # Auto-refresh on terminal events
                if event_type in ("task_completed", "task_failed", "task_cancelled"):
                    await self._refresh_tasks()
                    break
        except Exception as e:
            log.write(f"[dim]Stream ended: {e}[/dim]")

    async def _handle_sse_event(self, event_type: str, data: dict) -> None:
        """Route SSE events to appropriate TUI updates."""
        log = self.query_one("#live-output", RichLog)

        if event_type == "token_streamed":
            # Display streaming tokens inline
            token = data.get("token", "")
            if token:
                log.write(token, shrink=False, scroll_end=True)
            return

        if event_type == "subtask_started":
            sid = data.get("subtask_id", "")
            log.write(f"\n[bold blue]▶ Started:[/bold blue] {sid}")
        elif event_type == "subtask_completed":
            sid = data.get("subtask_id", "")
            summary = data.get("summary", "")[:80]
            log.write(f"[bold green]✓ Completed:[/bold green] {sid} — {summary}")
        elif event_type == "subtask_failed":
            sid = data.get("subtask_id", "")
            feedback = data.get("feedback", "")[:80]
            log.write(f"[bold red]✗ Failed:[/bold red] {sid} — {feedback}")
        elif event_type == "subtask_retrying":
            sid = data.get("subtask_id", "")
            attempt = data.get("attempt", "?")
            log.write(f"[bold yellow]↻ Retrying:[/bold yellow] {sid} (attempt {attempt})")
        elif event_type == "task_completed":
            log.write("\n[bold green]Task completed successfully.[/bold green]")
        elif event_type == "task_failed":
            error = data.get("error", "Unknown error")
            log.write(f"\n[bold red]Task failed:[/bold red] {error}")
        elif event_type == "task_planning":
            log.write("[dim]Planning...[/dim]")
        elif event_type == "task_plan_ready":
            count = data.get("subtask_count", "?")
            log.write(f"[bold]Plan ready:[/bold] {count} subtasks")
        elif event_type == "task_replanning":
            log.write("[bold yellow]Re-planning...[/bold yellow]")
        elif event_type == "approval_requested":
            self.show_approval_modal(data)
        elif event_type == "conversation_message":
            role = data.get("role", "user")
            msg = data.get("message", "")[:100]
            log.write(f"[bold yellow]{role}:[/bold yellow] {msg}")
        elif event_type == "tool_call_started":
            tool = data.get("tool", "")
            log.write(f"  [dim]→ {tool}[/dim]")
        elif event_type == "tool_call_completed":
            tool = data.get("tool", "")
            success = data.get("success", True)
            icon = "✓" if success else "✗"
            log.write(f"  [dim]{icon} {tool}[/dim]")

        # Refresh plan tree on subtask state changes
        if event_type in ("subtask_started", "subtask_completed", "subtask_failed"):
            if self.selected_task_id:
                await self._show_task_detail(self.selected_task_id)

    # --- Actions ---

    async def action_refresh(self) -> None:
        await self._refresh_tasks()

    async def action_steer(self) -> None:
        if not self.selected_task_id:
            return

        def handle_steer(instruction: str | None) -> None:
            if instruction:
                self._send_steer(self.selected_task_id, instruction)

        self.push_screen(SteerScreen(), callback=handle_steer)

    @work
    async def _send_steer(self, task_id: str, instruction: str) -> None:
        try:
            await self.api.steer(task_id, instruction)
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold yellow]Steer:[/] {instruction}")
        except Exception as e:
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Steer failed:[/] {e}")

    async def action_approve_action(self) -> None:
        if self._pending_approval:
            await self._handle_approval(True)

    async def _handle_approval(self, approved: bool) -> None:
        if not self._pending_approval:
            return
        info = self._pending_approval
        try:
            await self.api.approve(info["task_id"], info["subtask_id"], approved)
            log = self.query_one("#live-output", RichLog)
            verdict = 'Approved' if approved else 'Rejected'
            log.write(f"[bold green]{verdict}[/] {info['subtask_id']}")
        except Exception as e:
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Approval failed:[/] {e}")
        self._pending_approval = None

    async def action_cancel_task(self) -> None:
        if not self.selected_task_id:
            return
        try:
            await self.api.cancel_task(self.selected_task_id)
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Cancelled[/] task {self.selected_task_id}")
            await self._refresh_tasks()
        except Exception as e:
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Cancel failed:[/] {e}")

    async def action_new_task(self) -> None:
        def handle_result(result: dict | None) -> None:
            if result:
                self._create_task_from_form(result)

        self.push_screen(NewTaskScreen(), callback=handle_result)

    @work
    async def _create_task_from_form(self, form_data: dict) -> None:
        try:
            result = await self.api.create_task(
                goal=form_data["goal"],
                workspace=form_data.get("workspace"),
                approval_mode=form_data.get("approval_mode", "auto"),
            )
            log = self.query_one("#live-output", RichLog)
            tid = result.get("task_id", "")
            log.write(f"[bold green]Created:[/] {tid}")
            await self._refresh_tasks()
            # Auto-select and start streaming
            self.selected_task_id = tid
            self._start_event_stream(tid)
        except Exception as e:
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Create failed:[/] {e}")

    async def action_show_memory(self) -> None:
        """Open memory inspector for the selected task."""
        if not self.selected_task_id:
            return
        self._open_memory_inspector(self.selected_task_id)

    @work
    async def _open_memory_inspector(self, task_id: str) -> None:
        try:
            entries = await self.api.get_memory(task_id)
            self.app.push_screen(MemoryInspectorScreen(entries))
        except Exception as e:
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Memory load failed:[/] {e}")

    async def action_show_feedback(self) -> None:
        """Open feedback modal for the selected task."""
        if not self.selected_task_id:
            return

        def handle_feedback(result: dict | None) -> None:
            if result:
                self._submit_feedback(result)

        self.push_screen(
            FeedbackScreen(self.selected_task_id),
            callback=handle_feedback,
        )

    @work
    async def _submit_feedback(self, feedback_data: dict) -> None:
        try:
            comment = f"Rating: {feedback_data['rating']}/5\n{feedback_data['comment']}"
            await self.api.submit_feedback(feedback_data["task_id"], comment)
            log = self.query_one("#live-output", RichLog)
            log.write("[bold green]Feedback submitted.[/]")
        except Exception as e:
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Feedback failed:[/] {e}")

    async def action_show_conversation(self) -> None:
        """Open conversation mode for the selected task."""
        if not self.selected_task_id:
            return
        self._open_conversation(self.selected_task_id)

    @work
    async def _open_conversation(self, task_id: str) -> None:
        try:
            history = await self.api.get_conversation_history(task_id)
        except Exception:
            history = []

        def handle_message(message: str | None) -> None:
            if message:
                self._send_conversation_message(task_id, message)

        self.app.push_screen(
            ConversationScreen(task_id, history),
            callback=handle_message,
        )

    @work
    async def _send_conversation_message(self, task_id: str, message: str) -> None:
        try:
            await self.api.send_message(task_id, message)
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold yellow]You:[/] {message}")
        except Exception as e:
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Message failed:[/] {e}")

    def show_approval_modal(self, data: dict) -> None:
        """Show an approval modal from an SSE event."""
        self._pending_approval = data

        def handle_result(approved: bool) -> None:
            self._resolve_approval(data, approved)

        self.push_screen(
            ApprovalScreen(
                task_id=data.get("task_id", ""),
                subtask_id=data.get("subtask_id", ""),
                reason=data.get("reason", ""),
                risk_level=data.get("risk_level", "unknown"),
            ),
            callback=handle_result,
        )

    @work
    async def _resolve_approval(self, data: dict, approved: bool) -> None:
        try:
            await self.api.approve(data["task_id"], data["subtask_id"], approved)
        except Exception:
            pass
        self._pending_approval = None

    async def on_unmount(self) -> None:
        await self.api.close()
