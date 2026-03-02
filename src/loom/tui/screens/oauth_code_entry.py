"""Shared OAuth callback input modal used by TUI auth flows."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class OAuthCodeEntryScreen(ModalScreen[str | None]):
    """Prompt for callback URL/code when auto-callback is unavailable."""

    _inherit_bindings = False

    BINDINGS = [
        Binding("enter", "submit", "Submit"),
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    OAuthCodeEntryScreen {
        align: center middle;
    }
    #oauth-code-dialog {
        width: 84;
        height: auto;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #oauth-code-input {
        margin-top: 1;
    }
    #oauth-code-actions {
        margin-top: 1;
        height: auto;
    }
    #oauth-code-actions Button {
        margin-right: 1;
    }
    """

    def __init__(
        self,
        *,
        title_text: str = "Enter OAuth Callback",
        prompt_text: str = "Paste full callback URL or raw authorization code.",
    ) -> None:
        super().__init__()
        self._title_text = str(title_text or "Enter OAuth Callback")
        self._prompt_text = str(
            prompt_text or "Paste full callback URL or raw authorization code."
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="oauth-code-dialog"):
            yield Label(f"[bold #7dcfff]{self._title_text}[/bold #7dcfff]")
            yield Label(self._prompt_text)
            yield Input(id="oauth-code-input")
            with Horizontal(id="oauth-code-actions"):
                yield Button("Submit", id="oauth-code-submit", variant="primary")
                yield Button("Cancel", id="oauth-code-cancel")

    def on_mount(self) -> None:
        self.query_one("#oauth-code-input", Input).focus()

    def action_submit(self) -> None:
        raw = self.query_one("#oauth-code-input", Input).value.strip()
        self.dismiss(raw or None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#oauth-code-submit")
    def _on_submit(self) -> None:
        self.action_submit()

    @on(Button.Pressed, "#oauth-code-cancel")
    def _on_cancel(self) -> None:
        self.action_cancel()
