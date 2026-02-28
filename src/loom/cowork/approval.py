"""Tool approval system for interactive cowork mode.

Controls which tool calls require user approval before execution.
Tools are categorized into auto-approved (read-only/safe) and
needs-approval (write/execute/destructive). Users can approve
individual calls or approve all future calls for a given tool.
"""

from __future__ import annotations

import enum
import sys
from collections.abc import Awaitable, Callable


class ApprovalDecision(enum.Enum):
    """Result of an approval request."""
    APPROVE = "approve"
    APPROVE_ALL = "approve_all"
    DENY = "deny"


# Tools that are safe to run without asking — read-only or non-destructive.
AUTO_APPROVED_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "search_files",
    "list_directory",
    "glob_find",
    "ripgrep_search",
    "analyze_code",
    "ask_user",
    "web_search",
    "web_fetch",
    "web_fetch_html",
    "task_tracker",
    "conversation_recall",
    "list_tools",
    "run_tool",
})


# Callback type: given tool name and args, return an ApprovalDecision.
ApprovalCallback = Callable[[str, dict], Awaitable[ApprovalDecision]]


class ToolApprover:
    """Tracks per-tool approval state and gates tool execution.

    Usage:
        approver = ToolApprover()
        decision = await approver.check("shell_execute", {"command": "rm -rf /"}, prompt_fn)
        if decision == ApprovalDecision.DENY:
            # skip execution, return denial result
    """

    def __init__(
        self,
        auto_approved: frozenset[str] = AUTO_APPROVED_TOOLS,
        prompt_callback: ApprovalCallback | None = None,
    ) -> None:
        self._auto_approved = auto_approved
        self._always_approved: set[str] = set()
        self._prompt_callback = prompt_callback

    @property
    def always_approved_tools(self) -> frozenset[str]:
        """Tools the user has permanently approved this session."""
        return frozenset(self._always_approved)

    def set_prompt_callback(self, callback: ApprovalCallback) -> None:
        """Set the callback used to prompt the user."""
        self._prompt_callback = callback

    def approve_tool_always(self, tool_name: str) -> None:
        """Mark a tool as always-approved for this session."""
        self._always_approved.add(tool_name)

    async def check(self, tool_name: str, args: dict) -> ApprovalDecision:
        """Check whether a tool call is approved.

        Returns ApprovalDecision.APPROVE if auto-approved or previously
        approved-all. Otherwise prompts the user via the callback.
        If no callback is set, auto-approves everything.
        """
        # Auto-approved tools never need permission
        if tool_name in self._auto_approved:
            return ApprovalDecision.APPROVE

        # User already said "always approve" for this tool
        if tool_name in self._always_approved:
            return ApprovalDecision.APPROVE

        # No callback = deny by default (safe fallback for headless/API deployments)
        if self._prompt_callback is None:
            return ApprovalDecision.DENY

        decision = await self._prompt_callback(tool_name, args)

        if decision == ApprovalDecision.APPROVE_ALL:
            self._always_approved.add(tool_name)
            return ApprovalDecision.APPROVE

        return decision


def terminal_approval_prompt(tool_name: str, args: dict) -> ApprovalDecision:
    """Synchronous terminal prompt for tool approval.

    Shows the tool name and a summary of arguments, then asks:
    [y]es / [n]o / [a]lways allow <tool>
    """
    # ANSI codes
    bold = "\033[1m"
    yellow = "\033[33m"
    cyan = "\033[36m"
    dim = "\033[2m"
    reset = "\033[0m"

    args_preview = _format_args_preview(tool_name, args)

    sys.stdout.write(
        f"\n{yellow}{bold}Approve?{reset} "
        f"{cyan}{tool_name}{reset}"
    )
    if args_preview:
        sys.stdout.write(f" {dim}{args_preview}{reset}")
    sys.stdout.write(
        f"\n  {dim}[y]es  [n]o  [a]lways allow {tool_name}{reset}\n"
    )
    sys.stdout.flush()

    try:
        answer = input(f"{yellow}? {reset}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return ApprovalDecision.DENY

    if answer in ("y", "yes", ""):
        return ApprovalDecision.APPROVE
    elif answer in ("a", "always"):
        return ApprovalDecision.APPROVE_ALL
    else:
        return ApprovalDecision.DENY


async def async_terminal_approval_prompt(tool_name: str, args: dict) -> ApprovalDecision:
    """Async wrapper around terminal_approval_prompt."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, terminal_approval_prompt, tool_name, args)


def _format_args_preview(tool_name: str, args: dict) -> str:
    """Create a short preview of args for the approval prompt."""
    if tool_name in ("write_file", "edit_file", "delete_file", "move_file"):
        path = args.get("path", args.get("file_path", ""))
        return path if path else ""

    if tool_name == "shell_execute":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            return cmd[:77] + "..."
        return cmd

    if tool_name == "git_command":
        git_args = args.get("args", [])
        return " ".join(git_args) if git_args else ""

    # Generic: show first string value
    for v in args.values():
        if isinstance(v, str) and v:
            preview = v if len(v) <= 60 else v[:57] + "..."
            return preview
    return ""
