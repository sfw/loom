"""Spreadsheet tool — CSV create/read/edit operations.

Provides structured data manipulation for CSV files:
create, read, add rows, add columns, update cells, and filter.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from loom.tools.registry import Tool, ToolContext, ToolResult


class SpreadsheetTool(Tool):
    """Create and manipulate CSV spreadsheet files."""

    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

    @property
    def name(self) -> str:
        return "spreadsheet"

    @property
    def description(self) -> str:
        return (
            "Create and manipulate CSV spreadsheets. Operations: "
            "'create' (new CSV with headers and optional rows), "
            "'read' (read CSV contents), "
            "'add_rows' (append rows to existing CSV), "
            "'add_column' (add a new column with optional default value), "
            "'update_cell' (update a specific cell by row/column), "
            "'summary' (get row count, column names, sample data)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "create", "read", "add_rows",
                        "add_column", "update_cell", "summary",
                    ],
                    "description": "The operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the CSV file (relative to workspace).",
                },
                "headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column headers (for 'create').",
                },
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": (
                        "Data rows as arrays of strings "
                        "(for 'create' and 'add_rows')."
                    ),
                },
                "column_name": {
                    "type": "string",
                    "description": (
                        "Column name (for 'add_column' or 'update_cell')."
                    ),
                },
                "default_value": {
                    "type": "string",
                    "description": (
                        "Default value for new column (for 'add_column')."
                    ),
                },
                "row_index": {
                    "type": "integer",
                    "description": (
                        "0-based row index (for 'update_cell'). "
                        "Does not count header row."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "New cell value (for 'update_cell').",
                },
            },
            "required": ["operation", "path"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if ctx.workspace is None:
            return ToolResult.fail("No workspace set")

        operation = args.get("operation", "")
        raw_path = args.get("path", "")
        if not raw_path:
            return ToolResult.fail("No path provided")

        try:
            filepath = self._resolve_path(raw_path, ctx.workspace)
        except Exception as e:
            return ToolResult.fail(str(e))

        if operation == "create":
            return await self._create(filepath, args, ctx)
        elif operation == "read":
            return await self._read(filepath)
        elif operation == "add_rows":
            return await self._add_rows(filepath, args, ctx)
        elif operation == "add_column":
            return await self._add_column(filepath, args, ctx)
        elif operation == "update_cell":
            return await self._update_cell(filepath, args, ctx)
        elif operation == "summary":
            return await self._summary(filepath)
        else:
            return ToolResult.fail(f"Unknown operation: {operation}")

    async def _create(
        self, filepath: Path, args: dict, ctx: ToolContext,
    ) -> ToolResult:
        headers = args.get("headers", [])
        rows = args.get("rows", [])
        if not headers:
            return ToolResult.fail("'create' requires 'headers'")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(
                str(filepath), subtask_id=ctx.subtask_id,
            )

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in rows:
                writer.writerow(row)

        assert ctx.workspace is not None
        rel = filepath.relative_to(ctx.workspace)
        return ToolResult.ok(
            f"Created {rel} with {len(headers)} columns and {len(rows)} rows.",
            files_changed=[str(rel)],
        )

    async def _read(self, filepath: Path) -> ToolResult:
        if not filepath.exists():
            return ToolResult.fail(f"File not found: {filepath.name}")
        size_err = self._check_file_size(filepath)
        if size_err:
            return ToolResult.fail(size_err)

        content = filepath.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return ToolResult.ok("Empty CSV file.")

        # Format as readable table
        lines = []
        for i, row in enumerate(rows[:101]):  # Cap at 100 data rows + header
            lines.append(" | ".join(row))
            if i == 0:
                lines.append("-" * len(lines[0]))

        output = "\n".join(lines)
        if len(rows) > 101:
            output += f"\n... ({len(rows) - 1} total rows, showing first 100)"
        return ToolResult.ok(output)

    async def _add_rows(
        self, filepath: Path, args: dict, ctx: ToolContext,
    ) -> ToolResult:
        if not filepath.exists():
            return ToolResult.fail(f"File not found: {filepath.name}")
        rows = args.get("rows", [])
        if not rows:
            return ToolResult.fail("'add_rows' requires 'rows'")

        if ctx.changelog is not None:
            ctx.changelog.record_before_write(
                str(filepath), subtask_id=ctx.subtask_id,
            )

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        assert ctx.workspace is not None
        rel = filepath.relative_to(ctx.workspace)
        return ToolResult.ok(
            f"Added {len(rows)} rows to {rel}.",
            files_changed=[str(rel)],
        )

    def _check_file_size(self, filepath: Path) -> str | None:
        """Return error message if file exceeds size limit, else None."""
        if filepath.stat().st_size > self.MAX_FILE_SIZE:
            return "File too large (max 5 MB)"
        return None

    async def _add_column(
        self, filepath: Path, args: dict, ctx: ToolContext,
    ) -> ToolResult:
        if not filepath.exists():
            return ToolResult.fail(f"File not found: {filepath.name}")
        size_err = self._check_file_size(filepath)
        if size_err:
            return ToolResult.fail(size_err)
        col_name = args.get("column_name", "")
        if not col_name:
            return ToolResult.fail("'add_column' requires 'column_name'")
        default = args.get("default_value", "")

        content = filepath.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return ToolResult.fail("CSV file is empty")

        # Add column
        rows[0].append(col_name)
        for row in rows[1:]:
            row.append(default)

        if ctx.changelog is not None:
            ctx.changelog.record_before_write(
                str(filepath), subtask_id=ctx.subtask_id,
            )

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        assert ctx.workspace is not None
        rel = filepath.relative_to(ctx.workspace)
        return ToolResult.ok(
            f"Added column '{col_name}' to {rel}.",
            files_changed=[str(rel)],
        )

    async def _update_cell(
        self, filepath: Path, args: dict, ctx: ToolContext,
    ) -> ToolResult:
        if not filepath.exists():
            return ToolResult.fail(f"File not found: {filepath.name}")
        size_err = self._check_file_size(filepath)
        if size_err:
            return ToolResult.fail(size_err)
        col_name = args.get("column_name", "")
        row_idx = args.get("row_index")
        value = args.get("value", "")
        if not col_name:
            return ToolResult.fail("'update_cell' requires 'column_name'")
        if row_idx is None:
            return ToolResult.fail("'update_cell' requires 'row_index'")

        content = filepath.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if len(rows) < 2:
            return ToolResult.fail("CSV has no data rows")

        headers = rows[0]
        if col_name not in headers:
            return ToolResult.fail(
                f"Column '{col_name}' not found. "
                f"Available: {', '.join(headers)}",
            )
        col_idx = headers.index(col_name)
        data_row = row_idx + 1  # Skip header
        if data_row < 1 or data_row >= len(rows):
            return ToolResult.fail(
                f"Row index {row_idx} out of range "
                f"(0-{len(rows) - 2})",
            )

        old_value = rows[data_row][col_idx]
        rows[data_row][col_idx] = value

        if ctx.changelog is not None:
            ctx.changelog.record_before_write(
                str(filepath), subtask_id=ctx.subtask_id,
            )

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        assert ctx.workspace is not None
        rel = filepath.relative_to(ctx.workspace)
        return ToolResult.ok(
            f"Updated {rel}[{row_idx}, {col_name}]: "
            f"'{old_value}' -> '{value}'.",
            files_changed=[str(rel)],
        )

    async def _summary(self, filepath: Path) -> ToolResult:
        if not filepath.exists():
            return ToolResult.fail(f"File not found: {filepath.name}")

        content = filepath.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return ToolResult.ok("Empty CSV file.")

        headers = rows[0]
        data_rows = rows[1:]
        lines = [
            f"File: {filepath.name}",
            f"Columns ({len(headers)}): {', '.join(headers)}",
            f"Rows: {len(data_rows)}",
        ]

        # Sample first 3 data rows
        if data_rows:
            lines.append("\nSample data (first 3 rows):")
            lines.append(" | ".join(headers))
            lines.append("-" * 40)
            for row in data_rows[:3]:
                lines.append(" | ".join(row))

        return ToolResult.ok("\n".join(lines))
