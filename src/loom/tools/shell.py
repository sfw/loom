"""Shell command execution tool with safety checks."""

from __future__ import annotations

import asyncio
import re

from loom.tools.registry import Tool, ToolContext, ToolResult, ToolSafetyError

# Patterns that are obviously destructive
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/(?:\s|$)",       # rm -rf /
    r"rm\s+-rf\s+~",               # rm -rf ~
    r"rm\s+-rf\s+/\*",             # rm -rf /*
    r"\bmkfs\b",                    # mkfs
    r"\bdd\s+if=",                  # dd if=
    r">\s*/dev/",                   # > /dev/
    r"chmod\s+-R\s+777\s+/(?:\s|$)",  # chmod -R 777 /
    r"curl\s+.*\|\s*(?:ba)?sh",    # curl | sh
    r"wget\s+.*\|\s*(?:ba)?sh",    # wget | bash
]

BLOCKED_RE = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]


def check_command_safety(command: str) -> str | None:
    """Check if a shell command matches any blocked patterns.

    Returns the matched pattern description if blocked, None if safe.
    """
    for pattern in BLOCKED_RE:
        if pattern.search(command):
            return f"Blocked dangerous command pattern: {pattern.pattern}"
    return None


class ShellExecuteTool(Tool):
    @property
    def name(self) -> str:
        return "shell_execute"

    @property
    def description(self) -> str:
        return "Execute a shell command in the workspace directory. Captures stdout and stderr."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 60

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        command = args.get("command", "")
        if not command.strip():
            return ToolResult.fail("Empty command")

        # Safety check
        violation = check_command_safety(command)
        if violation:
            raise ToolSafetyError(violation)

        cwd = str(ctx.workspace) if ctx.workspace else None

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await process.communicate()
        except OSError as e:
            return ToolResult.fail(f"Failed to execute: {e}")

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        output_parts = []
        if stdout_text:
            output_parts.append(stdout_text)
        if stderr_text:
            output_parts.append(f"[stderr]\n{stderr_text}")

        output = "\n".join(output_parts)

        # Truncate to 10KB
        max_size = ToolResult.MAX_OUTPUT_SIZE
        if len(output) > max_size:
            output = output[:max_size] + "\n... (output truncated)"

        return ToolResult(
            success=process.returncode == 0,
            output=output,
            error=f"Exit code: {process.returncode}" if process.returncode != 0 else None,
            data={"exit_code": process.returncode},
        )
