"""Shell command execution tool with safety checks."""

from __future__ import annotations

import asyncio
import re

from loom.tools.registry import Tool, ToolContext, ToolResult, ToolSafetyError

# Max bytes to buffer from process stdout/stderr before killing it.
# Prevents OOM from commands like `yes` or `cat /dev/urandom`.
_MAX_OUTPUT_BUFFER = 1_048_576  # 1 MB

# Patterns that are obviously destructive — covers flag reordering and quoting bypasses
BLOCKED_PATTERNS = [
    r"\brm\b.*\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(?:\s|$)",   # rm -rf /, rm -r -f /, etc.
    r"\brm\b.*\s+-[a-zA-Z]*r[a-zA-Z]*\s+~",            # rm -rf ~
    r"\brm\b.*\s+-[a-zA-Z]*r[a-zA-Z]*\s+/\*",          # rm -rf /*
    r"\brm\b.*--recursive",                              # rm --recursive
    r"\bmkfs\b",                                         # mkfs
    r"\bdd\s+if=",                                       # dd if=
    r">\s*/dev/",                                        # > /dev/
    r"chmod\s+-R\s+777\s+/(?:\s|$)",                     # chmod -R 777 /
    r"curl\s+.*\|\s*(?:ba)?sh",                          # curl | sh
    r"wget\s+.*\|\s*(?:ba)?sh",                          # wget | bash
    r"\$\(.*\brm\b",                                     # $(rm ...) command substitution
    r"`.*\brm\b",                                        # `rm ...` backtick substitution
    r"\bpython[23]?\s+-c\s",                             # python -c (arbitrary code)
    r"\bperl\s+-e\s",                                    # perl -e (arbitrary code)
    r"\bruby\s+-e\s",                                    # ruby -e (arbitrary code)
    r"\bsudo\s",                                         # sudo anything
    r"\bchown\s+-R\s.*\s+/(?:\s|$)",                     # chown -R ... /
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


async def _read_limited(stream: asyncio.StreamReader | None, limit: int) -> bytes:
    """Read from stream up to *limit* bytes, then discard the rest."""
    if stream is None:
        return b""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        remaining = limit - total
        if remaining <= 0:
            continue  # drain remaining output without storing
        chunks.append(chunk[:remaining])
        total += len(chunk[:remaining])
    return b"".join(chunks)


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
        return 120

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        command = args.get("command", "")
        if not command.strip():
            return ToolResult.fail("Empty command")

        # Safety check
        violation = check_command_safety(command)
        if violation:
            raise ToolSafetyError(violation)

        cwd = str(ctx.workspace) if ctx.workspace else None

        process = None
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            # Read with bounded buffer to prevent OOM
            stdout_bytes, stderr_bytes = await asyncio.gather(
                _read_limited(process.stdout, _MAX_OUTPUT_BUFFER),
                _read_limited(process.stderr, _MAX_OUTPUT_BUFFER),
            )
            await process.wait()
        except OSError as e:
            return ToolResult.fail(f"Failed to execute: {e}")
        except (asyncio.CancelledError, TimeoutError):
            # Ensure subprocess is terminated on cancellation/timeout
            if process is not None:
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
            raise

        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        output_parts = []
        if stdout_text:
            output_parts.append(stdout_text)
        if stderr_text:
            output_parts.append(f"[stderr]\n{stderr_text}")

        output = "\n".join(output_parts)

        # Truncate for model context
        max_size = ToolResult.MAX_OUTPUT_SIZE
        if len(output) > max_size:
            output = output[:max_size] + "\n... (output truncated)"

        return ToolResult(
            success=process.returncode == 0,
            output=output,
            error=f"Exit code: {process.returncode}" if process.returncode != 0 else None,
            data={"exit_code": process.returncode},
        )
