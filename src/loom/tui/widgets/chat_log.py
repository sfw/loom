"""Chat log widget — scrollable container for messages and tool calls."""

from __future__ import annotations

from rich.markdown import Markdown as RichMarkdown
from textual.containers import VerticalScroll
from textual.widgets import Static

from loom.tui.widgets.tool_call import ToolCallWidget


class ChatLog(VerticalScroll):
    """Scrollable chat log that holds Markdown/text messages and tool widgets.

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
        width: 100%;
    }
    ChatLog .model-text {
        margin: 0;
        padding: 0;
        width: 100%;
    }
    ChatLog .turn-separator {
        margin: 1 0;
        color: $text-muted;
        width: 100%;
    }
    ChatLog .info-msg {
        margin: 0;
        padding: 0;
        width: 100%;
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
                expand=True,
            )
        )
        self._scroll_to_end()

    def add_model_text(self, text: str, *, markup: bool = False) -> None:
        """Append model response text.

        `markup=True` keeps Rich markup behavior for system/error lines.
        Normal model output is rendered as Markdown.
        """
        self._flush_and_reset_stream()
        if markup:
            widget = Static(text, classes="model-text", expand=True, markup=True)
        else:
            widget = Static(
                RichMarkdown(text or ""),
                classes="model-text",
                expand=True,
            )
        self.mount(widget)
        self._scroll_to_end()

    def add_streaming_text(self, text: str) -> None:
        """Append a streamed text chunk.

        Buffers chunks and flushes to the widget periodically to avoid
        O(n^2) string concatenation on every chunk.
        """
        self._stream_buffer.append(text)

        if self._stream_widget is None:
            self._stream_text = ""
            self._stream_widget = Static("", classes="model-text", expand=True)
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
        # Convert streamed plain text into Markdown once the message segment ends.
        if self._stream_widget is not None and self._stream_text:
            self._stream_widget.update(RichMarkdown(self._stream_text))
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
        *,
        tokens_per_second: float = 0.0,
        latency_ms: int = 0,
        total_time_ms: int = 0,
        context_tokens: int = 0,
        context_messages: int = 0,
        omitted_messages: int = 0,
        recall_index_used: bool = False,
    ) -> None:
        """Add a turn separator line with stats."""
        self._flush_and_reset_stream()
        parts: list[str] = []
        if tool_count:
            s = "s" if tool_count != 1 else ""
            parts.append(f"{tool_count} tool{s}")
        parts.append(f"{tokens:,} tokens")
        if tokens_per_second > 0:
            parts.append(f"{tokens_per_second:.1f} tok/s")
        if latency_ms > 0:
            parts.append(f"{self._format_ms(latency_ms)} latency")
        if total_time_ms > 0:
            parts.append(f"{self._format_ms(total_time_ms)} total")
        if context_tokens > 0:
            parts.append(f"ctx {context_tokens:,} tok")
        if context_messages > 0:
            parts.append(f"{context_messages} ctx msg")
        if omitted_messages > 0:
            parts.append(f"{omitted_messages} archived")
        if recall_index_used:
            parts.append("recall-index")
        if model:
            parts.append(model)
        detail = " | ".join(parts)
        line = f"[dim]\u2500\u2500\u2500 {detail} \u2500\u2500\u2500[/dim]"
        self.mount(Static(line, classes="turn-separator", expand=True))
        self._scroll_to_end()

    @staticmethod
    def _format_ms(duration_ms: int) -> str:
        if duration_ms >= 1000:
            return f"{duration_ms / 1000.0:.1f}s"
        return f"{duration_ms}ms"

    def add_info(self, text: str, *, markup: bool = True) -> None:
        """Add an informational message (e.g. welcome text)."""
        self._flush_and_reset_stream()
        self.mount(
            Static(text, classes="info-msg", expand=True, markup=markup),
        )
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
                    expand=True,
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
                    expand=True,
                ))
        if content_blocks:
            self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        if self._auto_scroll:
            self.scroll_end(animate=False)
