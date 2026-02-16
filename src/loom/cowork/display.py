"""Terminal display for cowork mode.

Renders tool calls, streaming text, and status information in the terminal
with colors and formatting for a clean interactive experience.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loom.cowork.session import CoworkTurn, ToolCallEvent


# ANSI color codes
class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"


def display_tool_start(event: ToolCallEvent) -> None:
    """Display a tool call starting."""
    args_summary = _summarize_args(event.name, event.args)
    sys.stdout.write(
        f"  {_C.GRAY}  {event.name}{_C.RESET}"
        f" {_C.DIM}{args_summary}{_C.RESET}\n"
    )
    sys.stdout.flush()


def display_tool_complete(event: ToolCallEvent) -> None:
    """Display a tool call result."""
    if event.result is None:
        return

    elapsed = f"{event.elapsed_ms}ms" if event.elapsed_ms else ""

    if event.result.success:
        icon = f"{_C.GREEN}ok{_C.RESET}"
    else:
        icon = f"{_C.RED}err{_C.RESET}"

    # Show a preview of the output
    preview = _output_preview(event.name, event.result.output)
    if preview:
        sys.stdout.write(
            f"  {_C.GRAY}  {icon} {_C.DIM}{elapsed}{_C.RESET}"
            f" {_C.DIM}{preview}{_C.RESET}\n"
        )
    else:
        sys.stdout.write(
            f"  {_C.GRAY}  {icon} {_C.DIM}{elapsed}{_C.RESET}\n"
        )

    # Show multimodal content indicators
    if event.result.content_blocks and event.result.success:
        _display_content_indicators(event.result.content_blocks)

    sys.stdout.flush()


def display_text_chunk(text: str) -> None:
    """Display a streaming text chunk."""
    sys.stdout.write(text)
    sys.stdout.flush()


def display_turn_summary(turn: CoworkTurn) -> None:
    """Display summary after a complete turn."""
    if turn.tool_calls:
        n = len(turn.tool_calls)
        sys.stdout.write(
            f"\n{_C.DIM}[{n} tool call{'s' if n != 1 else ''}"
            f" | {turn.tokens_used} tokens"
            f" | {turn.model}]{_C.RESET}\n"
        )
        sys.stdout.flush()


def display_ask_user(event: ToolCallEvent) -> str:
    """Display a question from the model and get user input."""
    question = event.args.get("question", "")
    options = event.args.get("options", [])

    sys.stdout.write(f"\n{_C.YELLOW}{_C.BOLD}Question:{_C.RESET} {question}\n")

    if options:
        for i, opt in enumerate(options, 1):
            sys.stdout.write(f"  {_C.CYAN}{i}.{_C.RESET} {opt}\n")
        sys.stdout.write(f"{_C.DIM}(Enter a number or type your answer){_C.RESET}\n")

    sys.stdout.flush()

    try:
        answer = input(f"{_C.GREEN}> {_C.RESET}")
    except (EOFError, KeyboardInterrupt):
        answer = ""

    # If they entered a number and options exist, map it
    if options and answer.strip().isdigit():
        idx = int(answer.strip()) - 1
        if 0 <= idx < len(options):
            answer = options[idx]

    return answer


def display_welcome(workspace: Path | None, model_name: str) -> None:
    """Display welcome message when starting a cowork session."""
    sys.stdout.write(f"\n{_C.BOLD}Loom Cowork{_C.RESET}")
    sys.stdout.write(f" {_C.DIM}({model_name}){_C.RESET}\n")
    if workspace:
        sys.stdout.write(f"{_C.DIM}workspace: {workspace}{_C.RESET}\n")
    sys.stdout.write(f"{_C.DIM}Type your request. Ctrl+C to exit.{_C.RESET}\n\n")
    sys.stdout.flush()


def display_error(message: str) -> None:
    """Display an error message."""
    sys.stdout.write(f"{_C.RED}Error:{_C.RESET} {message}\n")
    sys.stdout.flush()


def display_goodbye() -> None:
    """Display exit message."""
    sys.stdout.write(f"\n{_C.DIM}Goodbye.{_C.RESET}\n")
    sys.stdout.flush()


def _summarize_args(tool_name: str, args: dict) -> str:
    """Create a short summary of tool arguments."""
    if tool_name in ("read_file", "write_file", "edit_file", "delete_file"):
        path = args.get("path", args.get("file_path", ""))
        if path:
            return _truncate(path, 60)

    if tool_name == "shell_execute":
        cmd = args.get("command", "")
        return _truncate(cmd, 80)

    if tool_name == "git_command":
        git_args = args.get("args", [])
        return _truncate(" ".join(git_args), 60)

    if tool_name in ("ripgrep_search", "search_files"):
        pattern = args.get("pattern", "")
        return _truncate(f"/{pattern}/", 60)

    if tool_name == "glob_find":
        pattern = args.get("pattern", "")
        return _truncate(pattern, 60)

    if tool_name == "web_fetch":
        url = args.get("url", "")
        return _truncate(url, 60)

    if tool_name == "web_search":
        query = args.get("query", "")
        return _truncate(query, 60)

    if tool_name == "task_tracker":
        action = args.get("action", "")
        content = args.get("content", "")
        if content:
            return _truncate(f"{action}: {content}", 60)
        return action

    if tool_name == "ask_user":
        question = args.get("question", "")
        return _truncate(question, 60)

    if tool_name == "analyze_code":
        path = args.get("path", "")
        return _truncate(path, 60)

    # Generic: show first string value
    for v in args.values():
        if isinstance(v, str) and v:
            return _truncate(v, 50)

    return ""


def display_file_diff(event: ToolCallEvent) -> None:
    """Display an inline diff when a file-modifying tool completes."""
    if event.result is None or not event.result.success:
        return

    # Only show diffs for file-modifying tools
    if event.name not in ("edit_file", "write_file", "delete_file", "move_file"):
        return

    output = event.result.output or ""

    # For edit_file, the diff is already embedded in the output
    if event.name == "edit_file":
        diff_text = _extract_diff(output)
        if diff_text:
            _render_diff(diff_text)
        return

    # For write/delete, show a short summary of what changed
    files = event.result.files_changed
    if files:
        for f in files:
            if event.name == "write_file":
                sys.stdout.write(
                    f"  {_C.GRAY}    {_C.GREEN}M{_C.RESET} {_C.DIM}{f}{_C.RESET}\n"
                )
            elif event.name == "delete_file":
                sys.stdout.write(
                    f"  {_C.GRAY}    {_C.RED}D{_C.RESET} {_C.DIM}{f}{_C.RESET}\n"
                )
            elif event.name == "move_file":
                sys.stdout.write(
                    f"  {_C.GRAY}    {_C.YELLOW}R{_C.RESET} {_C.DIM}{f}{_C.RESET}\n"
                )
        sys.stdout.flush()


def _extract_diff(output: str) -> str:
    """Extract the diff portion from edit_file output."""
    # The diff starts with --- a/ line
    marker = "--- a/"
    idx = output.find(marker)
    if idx == -1:
        return ""
    return output[idx:]


def _render_diff(diff_text: str) -> None:
    """Render a unified diff with colors."""
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            sys.stdout.write(f"  {_C.GRAY}    {_C.BOLD}{line}{_C.RESET}\n")
        elif line.startswith("@@"):
            sys.stdout.write(f"  {_C.GRAY}    {_C.CYAN}{line}{_C.RESET}\n")
        elif line.startswith("+"):
            sys.stdout.write(f"  {_C.GRAY}    {_C.GREEN}{line}{_C.RESET}\n")
        elif line.startswith("-"):
            sys.stdout.write(f"  {_C.GRAY}    {_C.RED}{line}{_C.RESET}\n")
        else:
            sys.stdout.write(f"  {_C.GRAY}    {_C.DIM}{line}{_C.RESET}\n")
    sys.stdout.flush()


def _output_preview(tool_name: str, output: str) -> str:
    """Create a short preview of tool output."""
    if not output:
        return ""

    # For search tools, show match count
    if tool_name in ("ripgrep_search", "search_files", "glob_find"):
        lines = output.strip().split("\n")
        if lines and (lines[0] == "No matches found." or lines[0] == "No files matched."):
            return lines[0]
        return f"{len(lines)} results"

    # For file reads, show line count or multimodal info
    if tool_name == "read_file":
        stripped = output.strip()
        if stripped.startswith("[Image:") or stripped.startswith("[Image too large"):
            return _truncate(stripped.strip("[]"), 60)
        if stripped.startswith("[PDF:"):
            return _truncate(stripped.strip("[]"), 60)
        if stripped.startswith("--- Page"):
            page_lines = [ln for ln in stripped.split("\n") if ln.startswith("--- Page")]
            return f"{len(page_lines)} pages"
        lines = output.split("\n")
        return f"{len(lines)} lines"

    # For shell, show first line
    if tool_name == "shell_execute":
        first = output.strip().split("\n")[0]
        return _truncate(first, 60)

    # For edit_file, show the summary line only (not the diff)
    if tool_name == "edit_file":
        first = output.split("\n")[0]
        return _truncate(first, 80)

    return ""


def _display_content_indicators(content_blocks: list) -> None:
    """Show inline indicators for multimodal content blocks in terminal."""
    from loom.content import DocumentBlock, ImageBlock

    def _sanitize(text: str) -> str:
        """Strip ANSI escape sequences from user-controlled text."""
        import re
        return re.sub(r'\033\[[0-9;]*[a-zA-Z]', '', text)

    for block in content_blocks:
        if isinstance(block, ImageBlock):
            dims = f"{block.width}x{block.height}" if block.width else ""
            size = f"{block.size_bytes:,} bytes" if block.size_bytes else ""
            name = _sanitize(block.source_path.rsplit("/", 1)[-1]) if block.source_path else ""
            parts = [p for p in [name, dims, size] if p]
            label = ", ".join(parts)
            sys.stdout.write(
                f"  {_C.GRAY}    {_C.MAGENTA}[image: {label}]{_C.RESET}\n"
            )
        elif isinstance(block, DocumentBlock):
            name = _sanitize(block.source_path.rsplit("/", 1)[-1]) if block.source_path else ""
            pr = ""
            if block.page_range:
                pr = f" pages {block.page_range[0] + 1}-{block.page_range[1]}"
            total = f" of {block.page_count}" if block.page_count else ""
            label = f"{name}{pr}{total}"
            sys.stdout.write(
                f"  {_C.GRAY}    {_C.CYAN}[document: {label}]{_C.RESET}\n"
            )


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."
