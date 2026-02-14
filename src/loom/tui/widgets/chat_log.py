"""Chat log widget — scrollable container for messages and tool calls."""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static

from loom.tui.widgets.tool_call import ToolCallWidget


class ChatLog(VerticalScroll):
    """Scrollable chat log that holds Static messages and ToolCallWidgets.

    Unlike a plain RichLog, this supports embedded interactive widgets
    (Collapsible tool calls, etc.) alongside static text.
    """

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        padding: 0 1;
        scrollbar-size: 1 1;
    }
    ChatLog .user-msg {
        margin: 1 0 0 0;
        padding: 0;
    }
    ChatLog .model-text {
        margin: 0;
        padding: 0;
    }
    ChatLog .turn-separator {
        margin: 1 0;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._auto_scroll = True

    def add_user_message(self, text: str) -> None:
        """Append a user message to the chat."""
        self.mount(
            Static(
                f"[bold #73daca]> {text}[/]",
                classes="user-msg",
            )
        )
        self._scroll_to_end()

    def add_model_text(self, text: str) -> None:
        """Append model response text."""
        self.mount(Static(text, classes="model-text"))
        self._scroll_to_end()

    def add_streaming_text(self, text: str) -> None:
        """Append a streamed text chunk.

        Tries to append to the last model-text widget if one exists.
        Otherwise creates a new one.
        """
        children = list(self.children)
        if children and isinstance(children[-1], Static):
            last = children[-1]
            if "model-text" in last.classes:
                current = str(last.renderable)
                last.update(current + text)
                self._scroll_to_end()
                return
        self.mount(Static(text, classes="model-text"))
        self._scroll_to_end()

    def add_tool_call(
        self,
        tool_name: str,
        args: dict,
        *,
        success: bool | None = None,
        elapsed_ms: int = 0,
        output: str = "",
        error: str = "",
    ) -> None:
        """Append a tool call widget."""
        self.mount(ToolCallWidget(
            tool_name,
            args,
            success=success,
            elapsed_ms=elapsed_ms,
            output=output,
            error=error,
        ))
        self._scroll_to_end()

    def add_turn_separator(
        self,
        tool_count: int,
        tokens: int,
        model: str,
    ) -> None:
        """Add a turn separator line with stats."""
        parts: list[str] = []
        if tool_count:
            s = "s" if tool_count != 1 else ""
            parts.append(f"{tool_count} tool{s}")
        parts.append(f"{tokens:,} tokens")
        if model:
            parts.append(model)
        detail = " | ".join(parts)
        line = f"[dim]\u2500\u2500\u2500 {detail} \u2500\u2500\u2500[/dim]"
        self.mount(Static(line, classes="turn-separator"))
        self._scroll_to_end()

    def add_info(self, text: str) -> None:
        """Add an informational message (e.g. welcome text)."""
        self.mount(Static(f"[dim]{text}[/dim]", classes="model-text"))
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        if self._auto_scroll:
            self.scroll_end(animate=False)
