"""Glob file finder: fast file discovery by pattern.

Uses Python's pathlib.glob for portable, fast file pattern matching.
Respects common ignore patterns (.git, node_modules, __pycache__, etc.).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from loom.tools.registry import Tool, ToolContext, ToolResult

# Directories to always skip
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs", "*.egg-info",
})

MAX_RESULTS = 500


class GlobFindTool(Tool):
    """Find files matching a glob pattern."""

    name = "glob_find"
    description = (
        "Find files matching a glob pattern. Fast file discovery. "
        "Supports patterns like '**/*.py', 'src/**/*.ts', '*.yaml'. "
        "Returns matching file paths sorted by modification time (newest first). "
        "Automatically skips .git, node_modules, __pycache__, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Glob pattern to match files against. "
                    "Examples: '**/*.py', 'src/**/*.tsx', '*.yaml', '**/test_*.py'"
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory to search in. Defaults to the workspace root. "
                    "Can be a relative path within the workspace."
                ),
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        pattern = args.get("pattern", "")
        if not pattern:
            return ToolResult(success=False, output="No pattern provided.")

        raw_path = args.get("path")
        if raw_path:
            p = Path(raw_path)
            if not p.is_absolute() and ctx.workspace:
                search_path = (ctx.workspace / p).resolve()
            else:
                search_path = p.expanduser().resolve()
        else:
            search_path = ctx.workspace
        if not search_path:
            return ToolResult(success=False, output="No workspace or path specified.")

        search_path = Path(search_path).resolve()
        if not search_path.is_dir():
            return ToolResult(success=False, output=f"Not a directory: {search_path}")

        # Workspace containment check
        if ctx.workspace:
            try:
                search_path.relative_to(ctx.workspace.resolve())
            except ValueError:
                return ToolResult(
                    success=False,
                    output=f"Path '{search_path}' is outside workspace '{ctx.workspace}'.",
                )

        try:
            matches = []
            for p in search_path.glob(pattern):
                # Skip ignored directories
                if any(part in _SKIP_DIRS for part in p.parts):
                    continue
                # Also skip via fnmatch for wildcard dir patterns
                if any(fnmatch.fnmatch(part, skip) for part in p.parts for skip in _SKIP_DIRS):
                    continue
                if p.is_file():
                    matches.append(p)

            # Sort by modification time (newest first)
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            if len(matches) > MAX_RESULTS:
                truncated = True
                matches = matches[:MAX_RESULTS]
            else:
                truncated = False

            # Format as relative paths from search_path
            lines = []
            for m in matches:
                try:
                    rel = m.relative_to(search_path)
                except ValueError:
                    rel = m
                lines.append(str(rel))

            output = "\n".join(lines) if lines else "No files matched."
            if truncated:
                output += f"\n\n(Showing first {MAX_RESULTS} of more results)"

            return ToolResult(
                success=True,
                output=output,
                data={"count": len(lines), "truncated": truncated},
            )

        except Exception as e:
            return ToolResult(success=False, output=f"Glob error: {e}")
