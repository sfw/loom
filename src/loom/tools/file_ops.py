"""File operation tools: read, write, edit, delete, move files."""

from __future__ import annotations

from loom.tools.registry import Tool, ToolContext, ToolResult, ToolSafetyError


class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Optionally specify a line range."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "line_start": {"type": "integer", "description": "Start line (1-based, optional)"},
                "line_end": {"type": "integer", "description": "End line (inclusive, optional)"},
            },
            "required": ["path"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path = self._resolve_path(args["path"], ctx.workspace)
        if not path.exists():
            return ToolResult.fail(f"File not found: {args['path']}")
        if not path.is_file():
            return ToolResult.fail(f"Not a file: {args['path']}")

        content = path.read_text(encoding="utf-8", errors="replace")

        line_start = args.get("line_start")
        line_end = args.get("line_end")
        if line_start is not None or line_end is not None:
            lines = content.splitlines(keepends=True)
            start = (line_start or 1) - 1
            end = line_end or len(lines)
            content = "".join(lines[start:end])

        return ToolResult.ok(content)


class WriteFileTool(Tool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates parent directories if needed."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path = self._resolve_path(args["path"], ctx.workspace)
        rel_path = str(path.relative_to(ctx.workspace.resolve()))

        # Record in changelog before writing
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(rel_path, subtask_id=ctx.subtask_id)

        path.parent.mkdir(parents=True, exist_ok=True)

        content = args["content"]
        path.write_text(content, encoding="utf-8")

        return ToolResult.ok(
            f"Wrote {len(content)} bytes to {rel_path}",
            files_changed=[rel_path],
        )


class DeleteFileTool(Tool):
    @property
    def name(self) -> str:
        return "delete_file"

    @property
    def description(self) -> str:
        return "Delete a file or empty directory. Cannot delete workspace root or .git/."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
            },
            "required": ["path"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path = self._resolve_path(args["path"], ctx.workspace)
        rel_path = str(path.relative_to(ctx.workspace.resolve()))

        # Safety: refuse to delete workspace root
        if path.resolve() == ctx.workspace.resolve():
            return ToolResult.fail("Cannot delete workspace root")

        # Safety: refuse to delete .git directory
        if ".git" in path.parts:
            return ToolResult.fail("Cannot delete .git directory or its contents")

        if not path.exists():
            return ToolResult.fail(f"Path not found: {args['path']}")

        # Record in changelog before deleting
        if ctx.changelog is not None and path.is_file():
            ctx.changelog.record_delete(rel_path, subtask_id=ctx.subtask_id)

        if path.is_file():
            path.unlink()
            return ToolResult.ok(f"Deleted file: {rel_path}", files_changed=[rel_path])
        elif path.is_dir():
            try:
                path.rmdir()  # Only removes empty directories
                return ToolResult.ok(f"Deleted empty directory: {rel_path}")
            except OSError:
                return ToolResult.fail(
                    f"Directory not empty: {rel_path}. Remove contents first."
                )
        else:
            return ToolResult.fail(f"Unsupported path type: {rel_path}")


class MoveFileTool(Tool):
    @property
    def name(self) -> str:
        return "move_file"

    @property
    def description(self) -> str:
        return "Move or rename a file or directory within the workspace."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path relative to workspace"},
                "destination": {"type": "string", "description": "Destination path relative to workspace"},
            },
            "required": ["source", "destination"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        source = self._resolve_path(args["source"], ctx.workspace)
        destination = self._resolve_path(args["destination"], ctx.workspace)

        if not source.exists():
            return ToolResult.fail(f"Source not found: {args['source']}")

        if destination.exists():
            return ToolResult.fail(f"Destination already exists: {args['destination']}")

        rel_source = str(source.relative_to(ctx.workspace.resolve()))
        rel_dest = str(destination.relative_to(ctx.workspace.resolve()))

        # Record in changelog before moving
        if ctx.changelog is not None:
            ctx.changelog.record_rename(rel_source, rel_dest, subtask_id=ctx.subtask_id)

        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

        return ToolResult.ok(
            f"Moved {rel_source} -> {rel_dest}",
            files_changed=[rel_source, rel_dest],
        )


class EditFileTool(Tool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing a unique string. "
            "old_str must appear exactly once in the file."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace"},
                "old_str": {
                    "type": "string",
                    "description": "Exact string to find (must be unique)",
                },
                "new_str": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_str", "new_str"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 10

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        path = self._resolve_path(args["path"], ctx.workspace)
        if not path.exists():
            return ToolResult.fail(f"File not found: {args['path']}")

        content = path.read_text(encoding="utf-8")
        old_str = args["old_str"]
        new_str = args["new_str"]

        count = content.count(old_str)
        if count == 0:
            return ToolResult.fail(f"old_str not found in {args['path']}")
        if count > 1:
            return ToolResult.fail(
                f"old_str appears {count} times in {args['path']}. Must be unique."
            )

        rel_path = str(path.relative_to(ctx.workspace.resolve()))

        # Record in changelog before writing
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(rel_path, subtask_id=ctx.subtask_id)

        new_content = content.replace(old_str, new_str, 1)
        path.write_text(new_content, encoding="utf-8")

        lines_old = old_str.count("\n") + 1
        lines_new = new_str.count("\n") + 1
        return ToolResult.ok(
            f"Edited {rel_path}: replaced {lines_old} lines with {lines_new} lines",
            files_changed=[rel_path],
        )
