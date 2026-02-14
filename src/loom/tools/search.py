"""Search and listing tools for workspace exploration."""

from __future__ import annotations

import os
import re

from loom.tools.registry import Tool, ToolContext, ToolResult

# Directories to always exclude from listings
EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache"}


class SearchFilesTool(Tool):
    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return "Search for a regex pattern across files in the workspace."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search in (relative to workspace)",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for files (e.g., '*.py')",
                },
            },
            "required": ["pattern"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 30

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        pattern = args["pattern"]
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult.fail(f"Invalid regex: {e}")

        search_path = ctx.workspace
        if "path" in args and args["path"]:
            search_path = self._resolve_path(args["path"], ctx.workspace)

        if not search_path.exists():
            return ToolResult.fail(f"Path not found: {args.get('path', '')}")

        file_pattern = args.get("file_pattern", "")
        matches = []
        max_matches = 200

        for root, dirs, files in os.walk(search_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for filename in files:
                if file_pattern and not _glob_match(filename, file_pattern):
                    continue

                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, encoding="utf-8", errors="replace") as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel = os.path.relpath(filepath, ctx.workspace)
                                matches.append(f"{rel}:{line_num}: {line.rstrip()}")
                                if len(matches) >= max_matches:
                                    break
                except (OSError, UnicodeDecodeError):
                    continue

                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break

        if not matches:
            return ToolResult.ok(f"No matches found for pattern: {pattern}")

        output = "\n".join(matches)
        if len(matches) >= max_matches:
            output += f"\n... ({max_matches} matches shown, more may exist)"

        return ToolResult.ok(output)


class ListDirectoryTool(Tool):
    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and directories up to 2 levels deep."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to workspace (default: root)",
                },
            },
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        target = ctx.workspace
        if "path" in args and args["path"]:
            target = self._resolve_path(args["path"], ctx.workspace)

        if not target.exists():
            return ToolResult.fail(f"Path not found: {args.get('path', '')}")
        if not target.is_dir():
            return ToolResult.fail(f"Not a directory: {args.get('path', '')}")

        lines = []
        _build_tree(target, lines, prefix="", max_depth=2, current_depth=0)

        if not lines:
            return ToolResult.ok("(empty directory)")

        return ToolResult.ok("\n".join(lines))


def _build_tree(
    path: os.PathLike,
    lines: list[str],
    prefix: str,
    max_depth: int,
    current_depth: int,
) -> None:
    """Build a tree-style directory listing."""
    from pathlib import Path

    p = Path(path)
    if current_depth >= max_depth:
        return

    try:
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return

    dirs = [e for e in entries if e.is_dir() and e.name not in EXCLUDED_DIRS]
    files = [e for e in entries if e.is_file()]

    for d in dirs:
        lines.append(f"{prefix}{d.name}/")
        _build_tree(d, lines, prefix=prefix + "  ", max_depth=max_depth,
                     current_depth=current_depth + 1)

    for f in files:
        lines.append(f"{prefix}{f.name}")


def _glob_match(filename: str, pattern: str) -> bool:
    """Glob matching for file patterns like '*.py', 'test_*.py', '*.{js,ts}'."""
    import fnmatch
    return fnmatch.fnmatch(filename, pattern)
