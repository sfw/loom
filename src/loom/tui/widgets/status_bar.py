"""Status bar widget for the bottom of the TUI."""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Structured status line showing state, workspace, model, and tokens."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    state: reactive[str] = reactive("Ready")
    workspace_name: reactive[str] = reactive("")
    model_name: reactive[str] = reactive("")
    total_tokens: reactive[int] = reactive(0)

    def render(self) -> str:
        parts: list[str] = [self.state]
        if self.workspace_name:
            parts.append(self.workspace_name)
        if self.model_name:
            parts.append(self.model_name)
        if self.total_tokens:
            parts.append(f"{self.total_tokens:,} tokens")
        return "[dim]" + " | ".join(parts) + "[/dim]"
