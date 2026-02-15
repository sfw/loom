"""Enhanced ask-user modal screen."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class AskUserScreen(ModalScreen[str]):
    """Display a question from the model and collect the user's answer."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    CSS = """
    AskUserScreen {
        align: center middle;
    }
    #ask-user-dialog {
        width: 72;
        height: auto;
        max-height: 20;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #ask-user-input {
        margin-top: 1;
    }
    """

    def __init__(
        self, question: str, options: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._question = question
        self._options = options or []

    def compose(self) -> ComposeResult:
        children = [
            Label(
                f"[bold #e0af68]Question:[/] {self._question}"
            ),
        ]
        for i, opt in enumerate(self._options, 1):
            children.append(Label(f"  [#7dcfff]{i}.[/] {opt}"))
        if self._options:
            children.append(
                Label("[dim]Enter a number or type your answer[/dim]")
            )
        children.append(
            Input(
                placeholder="Your answer...", id="ask-user-input",
            )
        )

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
