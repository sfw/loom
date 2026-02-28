"""Tool call display widget for the chat log."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.widgets import Collapsible, Static


def _trunc(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def tool_args_preview(tool_name: str, args: dict) -> str:
    """Short preview of tool arguments for the chat log."""
    if tool_name in (
        "read_file", "write_file", "edit_file", "delete_file",
    ):
        return _trunc(args.get("path", args.get("file_path", "")), 60)
    if tool_name == "shell_execute":
        return _trunc(args.get("command", ""), 80)
    if tool_name == "git_command":
        return _trunc(" ".join(args.get("args", [])), 60)
    if tool_name in ("ripgrep_search", "search_files"):
        return _trunc(f"/{args.get('pattern', '')}/", 60)
    if tool_name == "glob_find":
        return _trunc(args.get("pattern", ""), 60)
    if tool_name in ("web_fetch", "web_fetch_html", "web_search"):
        return _trunc(args.get("url", args.get("query", "")), 60)
    if tool_name == "task_tracker":
        action = args.get("action", "")
        content = args.get("content", "")
        return _trunc(f"{action}: {content}" if content else action, 60)
    if tool_name == "ask_user":
        return _trunc(args.get("question", ""), 60)
    if tool_name == "analyze_code":
        return _trunc(args.get("path", ""), 60)
    for v in args.values():
        if isinstance(v, str) and v:
            return _trunc(v, 50)
    return ""


def tool_output_preview(tool_name: str, output: str) -> str:
    """Short preview of tool output for the chat log."""
    if not output:
        return ""
    if tool_name in ("ripgrep_search", "search_files", "glob_find"):
        lines = output.strip().split("\n")
        if lines and ("No matches" in lines[0] or "No files" in lines[0]):
            return lines[0]
        return f"{len(lines)} results"
    if tool_name == "read_file":
        # Detect multimodal content from output format
        stripped = output.strip()
        if stripped.startswith("[Image:") or stripped.startswith("[Image too large"):
            return _trunc(stripped.strip("[]"), 60)
        if stripped.startswith("[PDF:"):
            return _trunc(stripped.strip("[]"), 60)
        if stripped.startswith("--- Page"):
            # PDF with extracted text — show page info
            page_lines = [ln for ln in stripped.split("\n") if ln.startswith("--- Page")]
            return f"{len(page_lines)} pages"
        return f"{len(output.splitlines())} lines"
    if tool_name == "shell_execute":
        return _trunc(output.strip().split("\n")[0], 60)
    if tool_name == "edit_file":
        # Show summary line only, not the diff
        return _trunc(output.split("\n")[0], 80)
    if tool_name == "web_search":
        hits = [
            x for x in output.strip().split("\n")
            if x.startswith(("1.", "2.", "3."))
        ]
        return f"{len(hits)} results" if hits else ""
    return ""


def _is_multimodal_output(output: str) -> bool:
    """Check if tool output represents multimodal content (image/PDF)."""
    stripped = output.strip()
    return stripped.startswith(("[Image:", "[Image too large", "[PDF:"))


def _style_diff_output(output: str) -> str:
    """Apply Rich markup to diff output for syntax highlighting.

    Colors diff lines: green for additions, red for removals,
    cyan for hunk headers. Summary lines stay dim.
    """
    lines = output.splitlines()
    styled_lines = []
    in_diff = False

    for line in lines:
        if line.startswith("--- a/"):
            in_diff = True
            styled_lines.append(f"[bold]{_escape(line)}[/bold]")
        elif line.startswith("+++ b/"):
            styled_lines.append(f"[bold]{_escape(line)}[/bold]")
        elif line.startswith("@@") and in_diff:
            styled_lines.append(f"[#7dcfff]{_escape(line)}[/]")
        elif line.startswith("+") and in_diff:
            styled_lines.append(f"[#9ece6a]{_escape(line)}[/]")
        elif line.startswith("-") and in_diff:
            styled_lines.append(f"[#f7768e]{_escape(line)}[/]")
        else:
            styled_lines.append(f"[dim]{_escape(line)}[/dim]")

    return "\n".join(styled_lines)


def _style_multimodal_output(output: str) -> str:
    """Style multimodal content indicators with distinct colors."""
    escaped = _escape(output)
    # Image indicators in magenta
    if "Image:" in output or "Image too large" in output:
        return f"[#bb9af7]{escaped}[/]"
    # PDF/document indicators in blue
    if "PDF:" in output or "Page " in output:
        return f"[#7dcfff]{escaped}[/]"
    return f"[dim]{escaped}[/dim]"


def _escape(text: str) -> str:
    """Escape Rich markup characters in text."""
    return text.replace("[", "\\[")


class ToolCallWidget(Static):
    """Renders a single tool call as a collapsible block in the chat."""

    DEFAULT_CSS = """
    ToolCallWidget {
        height: auto;
        margin: 0 0;
        padding: 0 1;
    }
    ToolCallWidget Collapsible {
        padding: 0;
        margin: 0;
        border: none;
    }
    """

    def __init__(
        self,
        tool_name: str,
        args: dict,
        *,
        success: bool | None = None,
        elapsed_ms: int = 0,
        output: str = "",
        error: str = "",
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args
        self._success = success
        self._elapsed_ms = elapsed_ms
        self._output = output
        self._error = error

    def compose(self) -> ComposeResult:
        preview = tool_args_preview(self._tool_name, self._args)
        elapsed = f"{self._elapsed_ms}ms" if self._elapsed_ms else ""

        if self._success is None:
            # Tool still running
            title = f"[dim]{self._tool_name}[/dim] [dim]{preview}[/dim]"
            yield Static(f"  {title}")
        elif self._success:
            out_preview = tool_output_preview(self._tool_name, self._output)
            status = "[#9ece6a]ok[/]"
            title = (
                f"  [dim]{self._tool_name}[/dim] [dim]{preview}[/dim]"
                f"  {status} [dim]{elapsed} {out_preview}[/dim]"
            )
            if self._output.strip():
                # Show output snippet in collapsible
                snippet = self._output[:2000]
                if len(self._output) > 2000:
                    snippet += "\n..."
                # Apply diff highlighting for edit_file output
                if self._tool_name == "edit_file":
                    styled = _style_diff_output(snippet)
                elif _is_multimodal_output(self._output):
                    styled = _style_multimodal_output(snippet)
                else:
                    styled = f"[dim]{_escape(snippet)}[/dim]"
                yield Collapsible(
                    Static(styled),
                    title=title,
                    collapsed=True,
                )
            else:
                yield Static(title)
        else:
            err_msg = _trunc(self._error or "failed", 80)
            title = (
                f"  [#f7768e]err[/] [dim]{self._tool_name}[/dim]"
                f" [dim]{preview}[/dim] [dim]{elapsed}[/dim]"
            )
            if self._error:
                yield Collapsible(
                    Static(f"[#f7768e]{_escape(err_msg)}[/]"),
                    title=title,
                    collapsed=True,
                )
            else:
                yield Static(title)


class DelegateProgressWidget(Static):
    """Collapsed live stream of delegated subagent progress."""

    DEFAULT_CSS = """
    DelegateProgressWidget {
        height: auto;
        margin: 0 0;
        padding: 0 1;
    }
    DelegateProgressWidget Collapsible {
        padding: 0;
        margin: 0;
        border: none;
    }
    """

    def __init__(
        self,
        *,
        tool_call_id: str,
        title: str = "Delegated progress",
        max_lines: int = 150,
    ) -> None:
        super().__init__()
        self._tool_call_id = str(tool_call_id or "").strip()
        self._base_title = str(title or "Delegated progress").strip()
        self._max_lines = max(20, int(max_lines))
        self._lines: list[str] = []
        self._trimmed_count = 0
        self._status: str = "running"  # running | completed | failed
        self._elapsed_ms = 0
        self._last_line = ""
        self._last_line_at = 0.0
        self._line_dedupe_window_seconds = 1.5

    def compose(self) -> ComposeResult:
        yield Collapsible(
            Static("", id="delegate-progress-body"),
            title=self._render_title(),
            collapsed=True,
            id="delegate-progress-collapsible",
        )

    def append_line(self, line: str) -> bool:
        """Append one line; returns True when it was accepted."""
        text = " ".join(str(line or "").split())
        if not text:
            return False
        now = time.monotonic()
        if text == self._last_line and (now - self._last_line_at) < self._line_dedupe_window_seconds:
            return False
        self._last_line = text
        self._last_line_at = now
        self._lines.append(text)
        if len(self._lines) > self._max_lines:
            overflow = len(self._lines) - self._max_lines
            if overflow > 0:
                self._lines = self._lines[overflow:]
                self._trimmed_count += overflow
        self._render_body()
        return True

    def finalize(self, *, success: bool, elapsed_ms: int = 0) -> None:
        self._status = "completed" if success else "failed"
        self._elapsed_ms = max(0, int(elapsed_ms))
        self._render_title_only()

    def restore_state(
        self,
        *,
        status: str = "running",
        elapsed_ms: int = 0,
        lines: list[str] | None = None,
    ) -> None:
        """Restore widget state from replay hydration."""
        normalized = str(status or "running").strip().lower()
        if normalized not in {"running", "completed", "failed"}:
            normalized = "running"
        self._status = normalized
        self._elapsed_ms = max(0, int(elapsed_ms))
        restored = [str(item or "").strip() for item in (lines or [])]
        self._lines = [item for item in restored if item]
        if len(self._lines) > self._max_lines:
            overflow = len(self._lines) - self._max_lines
            self._lines = self._lines[overflow:]
            self._trimmed_count += overflow
        self._render_body()
        self._render_title_only()

    def _render_title(self) -> str:
        elapsed = ""
        if self._elapsed_ms > 0:
            elapsed = (
                f"{self._elapsed_ms}ms"
                if self._elapsed_ms < 1000
                else f"{self._elapsed_ms / 1000.0:.1f}s"
            )
        if self._status == "running":
            return f"  [dim]{_escape(self._base_title)}[/dim] [dim]running[/dim]"
        if self._status == "completed":
            suffix = f" [dim]{elapsed}[/dim]" if elapsed else ""
            return f"  [#9ece6a]ok[/] [dim]{_escape(self._base_title)}[/dim]{suffix}"
        suffix = f" [dim]{elapsed}[/dim]" if elapsed else ""
        return f"  [#f7768e]err[/] [dim]{_escape(self._base_title)}[/dim]{suffix}"

    def _render_title_only(self) -> None:
        try:
            collapsible = self.query_one("#delegate-progress-collapsible", Collapsible)
            collapsible.title = self._render_title()
        except Exception:
            pass

    def _render_body(self) -> None:
        try:
            body = self.query_one("#delegate-progress-body", Static)
        except Exception:
            return
        rendered = [f"[dim]{_escape(line)}[/dim]" for line in self._lines]
        if self._trimmed_count > 0:
            rendered.append(
                f"[dim]... {self._trimmed_count} older line(s) omitted ...[/dim]"
            )
        body.update("\n".join(rendered))
