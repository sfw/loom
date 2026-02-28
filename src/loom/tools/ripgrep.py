"""Ripgrep-powered content search.

Shells out to `rg` (ripgrep) for fast, gitignore-aware content search.
Falls back to a Python implementation if ripgrep is not installed.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

from loom.tools.registry import Tool, ToolContext, ToolResult

DEFAULT_MAX_MATCHES = 200


class RipgrepSearchTool(Tool):
    """Search file contents using ripgrep."""

    name = "ripgrep_search"
    description = (
        "Search file contents using ripgrep (rg). Much faster than search_files. "
        "Supports regex patterns, file type filtering, context lines, and "
        "gitignore-aware searching. "
        "Returns structured counts in data (match_count, file_count). "
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

        raw_path = args.get("path")
        if raw_path:
            if ctx.workspace is None:
                return ToolResult(success=False, output="No workspace or path specified.")
            search_path = self._resolve_read_path(
                str(raw_path),
                ctx.workspace,
                ctx.read_roots,
            )
        else:
            search_path = ctx.workspace
        if not search_path:
            return ToolResult(success=False, output="No workspace or path specified.")

        search_path = Path(search_path).resolve()
        if not search_path.exists():
            return ToolResult(success=False, output=f"Path not found: {search_path}")

        # Workspace/read-root containment check
        if ctx.workspace:
            try:
                self._verify_within_allowed_roots(
                    search_path,
                    ctx.workspace,
                    ctx.read_roots,
                )
            except Exception:
                return ToolResult(
                    success=False,
                    output=(
                        f"Path '{search_path}' is outside workspace/read roots. "
                        f"Workspace: '{ctx.workspace}'."
                    ),
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
        cmd.append("--stats")

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

            if proc.returncode == 0:
                err = stderr.decode("utf-8", errors="replace")
                stats = self._parse_rg_stats(err)
                match_count, file_count = self._infer_counts_from_output(
                    output,
                    files_only=bool(args.get("files_only")),
                )
                if stats.get("match_count") is not None:
                    match_count = int(stats["match_count"])
                if stats.get("file_count") is not None:
                    file_count = int(stats["file_count"])
                return ToolResult(
                    success=True,
                    output=output or "Matches found (see above).",
                    data={
                        "match_count": match_count,
                        "file_count": file_count,
                    },
                )
            elif proc.returncode == 1:
                return ToolResult(
                    success=True,
                    output="No matches found.",
                    data={"match_count": 0, "file_count": 0},
                )
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

            if proc.returncode == 0:
                match_count, file_count = self._infer_counts_from_output(
                    output,
                    files_only=bool(args.get("files_only")),
                )
                return ToolResult(
                    success=True,
                    output=output or "Matches found.",
                    data={
                        "match_count": match_count,
                        "file_count": file_count,
                    },
                )
            elif proc.returncode == 1:
                return ToolResult(
                    success=True,
                    output="No matches found.",
                    data={"match_count": 0, "file_count": 0},
                )
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
            return ToolResult(
                success=True,
                output="No matches found.",
                data={"match_count": 0, "file_count": 0},
            )

        output = "\n".join(matches)
        file_paths = {
            line.split(":", 1)[0]
            for line in matches
            if ":" in line
        }
        return ToolResult(
            success=True,
            output=output,
            data={
                "match_count": len(matches),
                "file_count": len(file_paths),
            },
        )

    @staticmethod
    def _parse_rg_stats(stderr_text: str) -> dict[str, int | None]:
        """Parse ripgrep --stats summary from stderr."""
        text = str(stderr_text or "")
        file_count_match = re.search(
            r"(?m)^\s*(\d+)\s+files?\s+contained\s+matches?\s*$",
            text,
        )
        match_count_match = re.search(
            r"(?m)^\s*(\d+)\s+matches?\s*$",
            text,
        )
        matched_lines_match = re.search(
            r"(?m)^\s*(\d+)\s+matched\s+lines?\s*$",
            text,
        )
        match_count: int | None = None
        if match_count_match:
            match_count = int(match_count_match.group(1))
        elif matched_lines_match:
            match_count = int(matched_lines_match.group(1))

        file_count: int | None = None
        if file_count_match:
            file_count = int(file_count_match.group(1))
        return {
            "match_count": match_count,
            "file_count": file_count,
        }

    @staticmethod
    def _infer_counts_from_output(output: str, *, files_only: bool) -> tuple[int | None, int]:
        """Infer counts from CLI output without re-running search."""
        lines = [line for line in str(output or "").splitlines() if line.strip()]
        if not lines:
            return 0, 0

        if files_only:
            files = {line.strip() for line in lines if line.strip()}
            # In files_only mode, exact match_count is not always present unless
            # tool-specific stats are available.
            return None, len(files)

        match_line_re = re.compile(r"^(?P<path>.+?):\d+:.+$")
        context_line_re = re.compile(r"^(?P<path>.+?)-\d+-.+$")

        match_count = 0
        files: set[str] = set()
        for line in lines:
            if line == "--":
                continue
            match_hit = match_line_re.match(line)
            if match_hit:
                match_count += 1
                files.add(match_hit.group("path"))
                continue
            context_hit = context_line_re.match(line)
            if context_hit:
                files.add(context_hit.group("path"))

        if match_count == 0:
            # Conservative fallback for unusual output shapes.
            match_count = len(lines)

        return match_count, len(files)
