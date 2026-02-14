"""Loom TUI application built with Textual.

A task monitoring dashboard — NOT a chat interface.
Connects to the API server as a client.
"""

from __future__ import annotations

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
    Static,
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
    ]

    def __init__(self, server_url: str = "http://localhost:9000"):
        super().__init__()
        self.server_url = server_url
        self.api = LoomAPIClient(server_url)
        self.selected_task_id: str | None = None
        self._pending_approval: dict | None = None

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
        def handle_goal(goal: str | None) -> None:
            if goal:
                self._create_task(goal)

        self.push_screen(SteerScreen(), callback=handle_goal)

    @work
    async def _create_task(self, goal: str) -> None:
        try:
            result = await self.api.create_task(goal)
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold green]Created:[/] {result.get('task_id', '')}")
            await self._refresh_tasks()
        except Exception as e:
            log = self.query_one("#live-output", RichLog)
            log.write(f"[bold red]Create failed:[/] {e}")

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
