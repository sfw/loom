"""Git command tool with safety checks."""

from __future__ import annotations

import asyncio
import re

from loom.tools.registry import Tool, ToolContext, ToolResult, ToolSafetyError

# Allowed git subcommands (whitelist approach)
ALLOWED_SUBCOMMANDS = frozenset({
    "status", "diff", "log", "show", "blame",
    "add", "commit", "reset",
    "branch", "checkout", "switch", "merge", "rebase",
    "stash", "tag",
    "fetch", "pull", "push", "remote",
    "ls-files", "rev-parse",
})

# Patterns that are destructive even within allowed subcommands
BLOCKED_PATTERNS = [
    r"push\s+.*--force",          # force push
    r"reset\s+--hard",            # hard reset
    r"clean\s+-[a-zA-Z]*f",       # git clean -f
    r"branch\s+-[a-zA-Z]*D",      # force delete branch
    r"checkout\s+\.\s*$",         # checkout . (discard all changes)
]

BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


def check_git_safety(args_str: str) -> str | None:
    """Check if a git command matches any blocked patterns.

    Returns the matched pattern description if blocked, None if safe.
    """
    for pattern in BLOCKED_RE:
        if pattern.search(args_str):
            return f"Blocked dangerous git pattern: {pattern.pattern}"
    return None


def parse_subcommand(args: list[str]) -> str | None:
    """Extract the git subcommand from args list."""
    for arg in args:
        if not arg.startswith("-"):
            return arg.lower()
    return None


class GitCommandTool(Tool):
    @property
    def name(self) -> str:
        return "git_command"

    @property
    def description(self) -> str:
        return (
            "Execute a git command in the workspace. "
            "Supports: status, diff, log, show, blame, add, commit, "
            "branch, checkout, switch, merge, rebase, stash, tag, "
            "fetch, pull, push, remote, ls-files, rev-parse. "
            "Note: force push is blocked for safety."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Git arguments (without 'git' prefix). "
                        "Example: ['status'] or ['commit', '-m', 'fix bug']"
                    ),
                },
            },
            "required": ["args"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 60

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        git_args: list[str] = args.get("args", [])
        if not git_args:
            return ToolResult.fail("No git arguments provided")

        # Validate subcommand is in allowlist
        subcommand = parse_subcommand(git_args)
        if subcommand is None:
            return ToolResult.fail("Could not determine git subcommand")
        if subcommand not in ALLOWED_SUBCOMMANDS:
            return ToolResult.fail(
                f"Git subcommand '{subcommand}' is not allowed. "
                f"Allowed: {', '.join(sorted(ALLOWED_SUBCOMMANDS))}"
            )

        # Safety check on full args string
        args_str = " ".join(git_args)
        violation = check_git_safety(args_str)
        if violation:
            raise ToolSafetyError(violation)

        cwd = str(ctx.workspace) if ctx.workspace else None
        command = ["git"] + git_args

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await process.communicate()
        except OSError as e:
            return ToolResult.fail(f"Failed to execute git: {e}")

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        output_parts = []
        if stdout_text:
            output_parts.append(stdout_text)
        if stderr_text:
            output_parts.append(f"[stderr]\n{stderr_text}")

        output = "\n".join(output_parts)

        max_size = ToolResult.MAX_OUTPUT_SIZE
        if len(output) > max_size:
            output = output[:max_size] + "\n... (output truncated)"

        return ToolResult(
            success=process.returncode == 0,
            output=output,
            error=f"Git exit code: {process.returncode}" if process.returncode != 0 else None,
            data={"exit_code": process.returncode, "subcommand": subcommand},
        )
