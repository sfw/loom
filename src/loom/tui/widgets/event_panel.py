"""Events panel — live log of tool calls and session events."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Sparkline, Static


class EventPanel(Vertical):
    """Live event log + token usage sparkline."""

    DEFAULT_CSS = """
    EventPanel {
        height: 1fr;
        padding: 0 1;
    }
    EventPanel DataTable {
        height: 1fr;
    }
    EventPanel #token-sparkline-container {
        height: 5;
        padding: 0;
        margin-top: 1;
    }
    EventPanel #token-sparkline-label {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._token_history: list[float] = []

    def compose(self) -> ComposeResult:
        table = DataTable(id="events-table")
        table.add_columns("Time", "Type", "Detail")
        table.cursor_type = "row"
        yield table
        yield Static("Tokens per turn", id="token-sparkline-label")
        yield Sparkline([], id="token-sparkline")

    def add_event(
        self, time_str: str, event_type: str, detail: str,
    ) -> None:
        """Add an event row to the log."""
        table = self.query_one("#events-table", DataTable)
        type_colors = {
            "tool_start": "#7dcfff",
            "tool_ok": "#9ece6a",
            "tool_err": "#f7768e",
            "text": "#c0caf5",
            "turn": "#bb9af7",
        }
        color = type_colors.get(event_type, "")
        label = event_type
        if color:
            label = f"[{color}]{event_type}[/]"
        table.add_row(time_str, label, detail)

    def record_turn_tokens(self, tokens: int) -> None:
        """Record token usage for a completed turn."""
        self._token_history.append(float(tokens))
        sparkline = self.query_one("#token-sparkline", Sparkline)
        sparkline.data = list(self._token_history)
