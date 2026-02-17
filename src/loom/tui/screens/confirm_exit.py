"""Exit confirmation modal screen."""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label


class ExitConfirmScreen(ModalScreen[bool]):
    """Prompt the user to confirm quitting the TUI."""

    # Avoid inheriting app-level bindings like Ctrl+C -> quit, which can
    # recursively open additional confirm modals while this one is active.
    _inherit_bindings = False

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("enter", "confirm", "Yes"),
        Binding("ctrl+c", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    ExitConfirmScreen {
        align: center middle;
    }
    #exit-confirm-dialog {
        width: 56;
        height: auto;
        border: solid $warning;
        padding: 1 2;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("[bold #e0af68]Exit Loom?[/]"),
            Label("Press [#9ece6a]Y[/] or [#9ece6a]Enter[/] to quit."),
            Label("Press [#f7768e]N[/] or [dim]Esc[/dim] to stay."),
            id="exit-confirm-dialog",
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_key(self, event: events.Key) -> None:
        """Fallback key handling for terminals where bindings are unreliable."""
        key = event.key.lower()
        if key in ("y", "enter", "ctrl+c"):
            self.dismiss(True)
            event.stop()
            event.prevent_default()
        elif key in ("n", "escape"):
            self.dismiss(False)
            event.stop()
            event.prevent_default()
