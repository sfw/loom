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
        self._stream_buffer: list[str] = []
        self._stream_widget: Static | None = None
        self._stream_text: str = ""

    def add_user_message(self, text: str) -> None:
        """Append a user message to the chat."""
        self._flush_and_reset_stream()
        self.mount(
            Static(
                f"[bold #73daca]> {text}[/]",
                classes="user-msg",
            )
        )
        self._scroll_to_end()

    def add_model_text(self, text: str) -> None:
        """Append model response text."""
        self._flush_and_reset_stream()
        self.mount(Static(text, classes="model-text"))
        self._scroll_to_end()

    def add_streaming_text(self, text: str) -> None:
        """Append a streamed text chunk.

        Buffers chunks and flushes to the widget periodically to avoid
        O(n^2) string concatenation on every chunk.
        """
        self._stream_buffer.append(text)

        if self._stream_widget is None:
            self._stream_text = ""
            self._stream_widget = Static("", classes="model-text")
            self.mount(self._stream_widget)

        # Flush every 5 chunks or when text is large
        if len(self._stream_buffer) >= 5 or len(text) > 100:
            self._flush_stream_buffer()

        self._scroll_to_end()

    def _flush_stream_buffer(self) -> None:
        """Flush buffered streaming chunks to the widget."""
        if not self._stream_buffer or self._stream_widget is None:
            return
        self._stream_text += "".join(self._stream_buffer)
        self._stream_widget.update(self._stream_text)
        self._stream_buffer.clear()

    def _flush_and_reset_stream(self) -> None:
        """Flush any buffered stream data and reset stream state."""
        self._flush_stream_buffer()
        self._stream_widget = None
        self._stream_text = ""

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
        self._flush_and_reset_stream()
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
        self._flush_and_reset_stream()
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
        self._flush_and_reset_stream()
        self.mount(Static(f"[dim]{text}[/dim]", classes="model-text"))
        self._scroll_to_end()

    def add_content_indicator(self, content_blocks: list) -> None:
        """Display inline indicators for multimodal content blocks.

        Shows styled placeholders for images and documents that the model
        is processing, since terminals cannot display actual images.
        """
        self._flush_and_reset_stream()
        from loom.content import DocumentBlock, ImageBlock

        def _esc(text: str) -> str:
            """Escape Rich markup in user-controlled text."""
            return text.replace("[", "\\[")

        for block in content_blocks:
            if isinstance(block, ImageBlock):
                dims = f"{block.width}x{block.height}" if block.width else ""
                size = f"{block.size_bytes:,} bytes" if block.size_bytes else ""
                name = _esc(block.source_path.rsplit("/", 1)[-1]) if block.source_path else ""
                parts = [p for p in [name, dims, size] if p]
                label = ", ".join(parts)
                self.mount(Static(
                    f"  [#bb9af7]\\[image: {label}][/]",
                    classes="model-text",
                ))
            elif isinstance(block, DocumentBlock):
                name = _esc(block.source_path.rsplit("/", 1)[-1]) if block.source_path else ""
                pr = ""
                if block.page_range:
                    pr = f" pages {block.page_range[0] + 1}-{block.page_range[1]}"
                total = f" of {block.page_count}" if block.page_count else ""
                label = f"{name}{pr}{total}"
                self.mount(Static(
                    f"  [#7dcfff]\\[document: {label}][/]",
                    classes="model-text",
                ))
        if content_blocks:
            self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        if self._auto_scroll:
            self.scroll_end(animate=False)
