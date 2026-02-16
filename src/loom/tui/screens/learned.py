"""Learned patterns review screen — review, approve, and delete patterns."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from loom.learning.manager import LearnedPattern


class PatternRow(Static):
    """A single pattern row with delete button."""

    CSS = """
    PatternRow {
        layout: horizontal;
        height: auto;
        min-height: 3;
        padding: 0 1;
        margin: 0 0 1 0;
        background: $surface;
    }
    PatternRow .pattern-info {
        width: 1fr;
    }
    PatternRow .pattern-delete {
        width: 10;
        min-width: 10;
        margin-left: 1;
    }
    """

    def __init__(self, pattern: LearnedPattern) -> None:
        super().__init__()
        self.pattern = pattern

    def compose(self) -> ComposeResult:
        p = self.pattern
        desc = p.data.get("description", p.pattern_key)
        ptype = p.pattern_type.replace("behavioral_", "").replace("_", " ")
        freq = f"  [dim](observed {p.frequency}x)[/dim]" if p.frequency > 1 else ""
        last = p.last_seen[:10] if p.last_seen else ""

        yield Label(
            f"[bold #7dcfff]{ptype}[/bold #7dcfff]: {desc}{freq}"
            f"\n[dim]key: {p.pattern_key}  |  last: {last}  |  id: {p.id}[/dim]",
            classes="pattern-info",
        )
        yield Button("Delete", variant="error", classes="pattern-delete",
                      id=f"del-{p.id}")


class LearnedScreen(ModalScreen[str]):
    """Review, approve, and delete learned behavioral patterns."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("d", "delete_prompt", "Delete by ID"),
    ]

    CSS = """
    LearnedScreen {
        align: center middle;
    }
    #learned-dialog {
        width: 90;
        height: 80%;
        max-height: 40;
        border: solid $primary;
        padding: 1 2;
        background: $surface;
    }
    #learned-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    #learned-scroll {
        height: 1fr;
    }
    #learned-footer {
        dock: bottom;
        height: 3;
        padding: 1 0 0 0;
    }
    #delete-input {
        display: none;
        margin-top: 1;
    }
    """

    def __init__(self, patterns: list[LearnedPattern]) -> None:
        super().__init__()
        self._patterns = patterns
        self._deleted_ids: list[int] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="learned-dialog"):
            yield Label(
                "[bold #e0af68]Learned Patterns[/bold #e0af68]",
                id="learned-title",
            )

            with VerticalScroll(id="learned-scroll"):
                if not self._patterns:
                    yield Label(
                        "[dim]No learned patterns yet. "
                        "Patterns are extracted automatically from your "
                        "interactions.[/dim]"
                    )
                else:
                    for p in self._patterns:
                        yield PatternRow(p)

            yield Label(
                "[dim]Esc: close  |  Click Delete to remove a pattern[/dim]",
                id="learned-footer",
            )
            yield Input(
                placeholder="Enter pattern ID to delete...",
                id="delete-input",
            )

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("del-"):
            try:
                pattern_id = int(btn_id[4:])
                self._deleted_ids.append(pattern_id)
                # Remove the row from the UI
                row = event.button.parent
                if row is not None:
                    row.remove()
                self.notify(f"Pattern {pattern_id} marked for deletion.")
            except ValueError:
                pass

    @on(Input.Submitted, "#delete-input")
    def on_delete_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        try:
            pattern_id = int(text)
            self._deleted_ids.append(pattern_id)
            # Try to remove matching row
            for child in self.query("PatternRow"):
                if isinstance(child, PatternRow) and child.pattern.id == pattern_id:
                    child.remove()
                    break
            self.notify(f"Pattern {pattern_id} marked for deletion.")
        except ValueError:
            self.notify("Enter a valid numeric pattern ID.", severity="error")
        inp = self.query_one("#delete-input", Input)
        inp.value = ""

    def action_close(self) -> None:
        # Return comma-separated list of deleted IDs (or empty)
        result = ",".join(str(i) for i in self._deleted_ids)
        self.dismiss(result)

    def action_delete_prompt(self) -> None:
        inp = self.query_one("#delete-input", Input)
        inp.display = True
        inp.focus()
