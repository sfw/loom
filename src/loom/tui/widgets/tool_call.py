"""Tool call display widget for the chat log."""

from __future__ import annotations

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
    if tool_name in ("web_fetch", "web_search"):
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
        return f"{len(output.splitlines())} lines"
    if tool_name == "shell_execute":
        return _trunc(output.strip().split("\n")[0], 60)
    if tool_name == "web_search":
        hits = [
            x for x in output.strip().split("\n")
            if x.startswith(("1.", "2.", "3."))
        ]
        return f"{len(hits)} results" if hits else ""
    return ""


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
                yield Collapsible(
                    Static(f"[dim]{snippet}[/dim]"),
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
                    Static(f"[#f7768e]{err_msg}[/]"),
                    title=title,
                    collapsed=True,
                )
            else:
                yield Static(title)
