"""Files Changed panel — DataTable of changed files with diff viewer."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static


class FilesChangedPanel(Vertical):
    """Shows a table of files changed during the session, with diffs."""

    DEFAULT_CSS = """
    FilesChangedPanel {
        height: 1fr;
        padding: 0 1;
    }
    FilesChangedPanel DataTable {
        height: 1fr;
    }
    FilesChangedPanel #diff-viewer {
        height: auto;
        max-height: 50%;
        background: $surface;
        padding: 1;
        display: none;
    }
    FilesChangedPanel #diff-viewer.visible {
        display: block;
    }
    """

    def compose(self) -> ComposeResult:
        table = DataTable(id="files-table")
        table.add_columns("Status", "File", "Time")
        table.cursor_type = "row"
        yield table
        yield Static("", id="diff-viewer")

    def update_files(self, entries: list[dict]) -> None:
        """Refresh the file table with changelog entries.

        Each entry should have: operation, path, timestamp
        """
        table = self.query_one("#files-table", DataTable)
        table.clear()

        status_colors = {
            "create": "#9ece6a",
            "modify": "#e0af68",
            "delete": "#f7768e",
            "rename": "#7dcfff",
        }

        for entry in entries:
            op = entry.get("operation", "?")
            path = entry.get("path", "?")
            ts = entry.get("timestamp", "")
            # Show only time portion if it's an ISO timestamp
            if "T" in ts:
                ts = ts.split("T")[1][:8]

            color = status_colors.get(op, "")
            status_label = op.capitalize()
            if color:
                status_label = f"[{color}]{status_label}[/]"

            table.add_row(status_label, path, ts)

    def show_diff(self, diff_text: str) -> None:
        """Display a diff in the viewer panel."""
        from loom.tui.widgets.tool_call import _style_diff_output

        viewer = self.query_one("#diff-viewer", Static)
        if diff_text:
            viewer.update(_style_diff_output(diff_text))
            viewer.add_class("visible")
        else:
            viewer.update("")
            viewer.remove_class("visible")
