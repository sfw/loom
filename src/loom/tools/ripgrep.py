"""Ripgrep-powered content search.

Shells out to `rg` (ripgrep) for fast, gitignore-aware content search.
Falls back to a Python implementation if ripgrep is not installed.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from loom.tools.registry import Tool, ToolContext, ToolResult

MAX_OUTPUT_BYTES = 30_000
DEFAULT_MAX_MATCHES = 200


class RipgrepSearchTool(Tool):
    """Search file contents using ripgrep."""

    name = "ripgrep_search"
    description = (
        "Search file contents using ripgrep (rg). Much faster than search_files. "
        "Supports regex patterns, file type filtering, context lines, and "
        "gitignore-aware searching. "
        "Examples: pattern='def main', type='py', context=2"
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for in file contents.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in. Defaults to workspace root.",
            },
            "glob": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. '*.py', '*.{ts,tsx}').",
            },
            "type": {
                "type": "string",
                "description": (
                    "File type filter (e.g. 'py', 'js', 'rust', 'go')."
                    " Efficient type filtering."
                ),
            },
            "context": {
                "type": "integer",
                "description": "Number of context lines before and after each match. Default 0.",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case insensitive search. Default false.",
            },
            "files_only": {
                "type": "boolean",
                "description": "Only return file paths that match, not the matching lines.",
            },
            "max_matches": {
                "type": "integer",
                "description": (
                    "Maximum matches to return."
                    f" Default {DEFAULT_MAX_MATCHES}."
                ),
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        pattern = args.get("pattern", "")
        if not pattern:
            return ToolResult(success=False, output="No search pattern provided.")

        search_path = Path(args["path"]) if args.get("path") else ctx.workspace
        if not search_path:
            return ToolResult(success=False, output="No workspace or path specified.")

        search_path = Path(search_path).expanduser().resolve()
        if not search_path.exists():
            return ToolResult(success=False, output=f"Path not found: {search_path}")

        # Workspace containment check
        if ctx.workspace:
            try:
                search_path.relative_to(ctx.workspace.resolve())
            except ValueError:
                return ToolResult(
                    success=False,
                    output=f"Path '{search_path}' is outside workspace '{ctx.workspace}'.",
                )

        # Try ripgrep first, fall back to grep
        rg_path = shutil.which("rg")
        if rg_path:
            return await self._run_ripgrep(rg_path, pattern, search_path, args)

        grep_path = shutil.which("grep")
        if grep_path:
            return await self._run_grep(grep_path, pattern, search_path, args)

        return await self._python_fallback(pattern, search_path, args)

    async def _run_ripgrep(
        self, rg_path: str, pattern: str, search_path: Path, args: dict,
    ) -> ToolResult:
        """Run ripgrep subprocess."""
        max_matches = args.get("max_matches", DEFAULT_MAX_MATCHES)

        cmd = [rg_path, "--color=never", f"--max-count={max_matches}"]

        if args.get("case_insensitive"):
            cmd.append("-i")
        if args.get("files_only"):
            cmd.append("-l")
        else:
            cmd.extend(["-n"])  # line numbers

        ctx_lines = args.get("context", 0)
        if ctx_lines and ctx_lines > 0:
            cmd.extend([f"-C{ctx_lines}"])

        if args.get("glob"):
            cmd.extend(["--glob", args["glob"]])
        if args.get("type"):
            cmd.extend(["--type", args["type"]])

        cmd.extend(["--", pattern, str(search_path)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            output = stdout.decode("utf-8", errors="replace")

            # Truncate if too large
            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n\n(output truncated)"

            if proc.returncode == 0:
                match_count = output.count("\n")
                return ToolResult(
                    success=True,
                    output=output or "Matches found (see above).",
                    data={"match_count": match_count},
                )
            elif proc.returncode == 1:
                return ToolResult(success=True, output="No matches found.")
            else:
                err = stderr.decode("utf-8", errors="replace")
                return ToolResult(success=False, output=f"ripgrep error: {err}")

        except TimeoutError:
            # Kill the subprocess on timeout to avoid resource leak
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return ToolResult(success=False, output="Search timed out after 30 seconds.")
        except Exception as e:
            return ToolResult(success=False, output=f"ripgrep error: {e}")

    async def _run_grep(
        self, grep_path: str, pattern: str, search_path: Path, args: dict,
    ) -> ToolResult:
        """Fallback: use grep -r."""
        max_matches = args.get("max_matches", DEFAULT_MAX_MATCHES)

        cmd = [grep_path, "-rn", "--color=never", f"--max-count={max_matches}"]

        if args.get("case_insensitive"):
            cmd.append("-i")
        if args.get("files_only"):
            cmd.append("-l")

        ctx_lines = args.get("context", 0)
        if ctx_lines and ctx_lines > 0:
            cmd.extend([f"-C{ctx_lines}"])

        if args.get("glob"):
            cmd.extend(["--include", args["glob"]])

        cmd.extend(["--", pattern, str(search_path)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")

            if len(output) > MAX_OUTPUT_BYTES:
                output = output[:MAX_OUTPUT_BYTES] + "\n\n(output truncated)"

            if proc.returncode == 0:
                return ToolResult(success=True, output=output or "Matches found.")
            elif proc.returncode == 1:
                return ToolResult(success=True, output="No matches found.")
            else:
                err = stderr.decode("utf-8", errors="replace")
                return ToolResult(success=False, output=f"grep error: {err}")

        except TimeoutError:
            return ToolResult(success=False, output="Search timed out after 30 seconds.")
        except Exception as e:
            return ToolResult(success=False, output=f"grep error: {e}")

    async def _python_fallback(
        self, pattern: str, search_path: Path, args: dict,
    ) -> ToolResult:
        """Pure Python fallback when neither rg nor grep is available."""
        import os
        import re

        try:
            regex = re.compile(pattern, re.IGNORECASE if args.get("case_insensitive") else 0)
        except re.error as e:
            return ToolResult(success=False, output=f"Invalid regex: {e}")

        skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
        max_matches = args.get("max_matches", DEFAULT_MAX_MATCHES)
        matches: list[str] = []

        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fname in files:
                if len(matches) >= max_matches:
                    break
                fpath = Path(root) / fname
                try:
                    text = fpath.read_text(errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if regex.search(line):
                            rel = fpath.relative_to(search_path)
                            matches.append(f"{rel}:{i}:{line.rstrip()}")
                            if len(matches) >= max_matches:
                                break
                except (OSError, UnicodeDecodeError):
                    continue

        if not matches:
            return ToolResult(success=True, output="No matches found.")

        output = "\n".join(matches)
        if len(output) > MAX_OUTPUT_BYTES:
            output = output[:MAX_OUTPUT_BYTES] + "\n\n(output truncated)"

        return ToolResult(
            success=True,
            output=output,
            data={"match_count": len(matches)},
        )
