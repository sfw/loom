"""Process run tab close confirmation modal screen."""

from __future__ import annotations

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ProcessRunCloseScreen(ModalScreen[bool]):
    """Prompt the user to confirm closing a process run tab."""

    # Avoid inheriting app-level bindings like Ctrl+C -> quit while modal is open.
    _inherit_bindings = False

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("enter", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ProcessRunCloseScreen {
        align: center middle;
    }
    #process-close-dialog {
        width: 72;
        height: auto;
        border: solid $warning;
        padding: 1 2;
        background: $surface;
    }
    #process-close-actions {
        width: 100%;
        height: auto;
        align-horizontal: right;
        margin: 1 0 0 0;
    }
    #process-close-cancel {
        margin-right: 1;
    }
    """

    def __init__(self, *, run_label: str, running: bool) -> None:
        super().__init__()
        self._run_label = run_label
        self._running = running

    def compose(self) -> ComposeResult:
        prompt = (
            f"[bold #e0af68]Close running tab {self._run_label}?[/]"
            if self._running
            else f"[bold #e0af68]Close tab {self._run_label}?[/]"
        )
        detail = (
            "This will cancel the process run and mark it failed."
            if self._running
            else "You can still inspect logs in events history."
        )
        yield Vertical(
            Label(prompt),
            Label(detail),
            Label(
                "Press [#9ece6a]Y[/]/[dim]Enter[/] to confirm or "
                "[#f7768e]N[/]/[dim]Esc[/] to keep it open.",
            ),
            Horizontal(
                Button("Keep Open", id="process-close-cancel"),
                Button("Close Tab", variant="error", id="process-close-confirm"),
                id="process-close-actions",
            ),
            id="process-close-dialog",
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#process-close-confirm")
    def _on_confirm_button(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#process-close-cancel")
    def _on_cancel_button(self) -> None:
        self.dismiss(False)

    def on_key(self, event: events.Key) -> None:
        """Fallback key handling for terminals with flaky modal bindings."""
        key = event.key.lower()
        if key in {"y", "enter"}:
            self.dismiss(True)
            event.stop()
            event.prevent_default()
        elif key in {"n", "escape"}:
            self.dismiss(False)
            event.stop()
            event.prevent_default()
