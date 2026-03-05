"""Sidebar widget: workspace file tree + task progress panel."""

from __future__ import annotations

from pathlib import Path

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import DirectoryTree, Label, Static

# Directories to hide from the file tree.
_HIDDEN_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox",
    "dist", "build", "*.egg-info",
})


class FilteredTree(DirectoryTree):
    """DirectoryTree that filters out common junk directories."""

    def filter_paths(self, paths: list[Path]) -> list[Path]:  # type: ignore[override]
        return [
            p for p in paths
            if p.name not in _HIDDEN_DIRS and not p.name.endswith(".egg-info")
        ]


class TaskProgressPanel(Static):
    """Displays task_tracker progress in the sidebar."""

    tasks: reactive[list[dict]] = reactive(list, layout=True)
    empty_message: reactive[str] = reactive("No tasks tracked")

    def __init__(self, *, auto_follow: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._auto_follow = bool(auto_follow)

    @staticmethod
    def _status_icon(status: str) -> tuple[str, str]:
        if status == "completed":
            return "\u2713", "#9ece6a"
        if status == "in_progress":
            return "\u25c9", "#7dcfff"
        if status == "failed":
            return "\u2717", "#f7768e"
        if status == "skipped":
            return "-", "dim"
        return "\u25cb", "dim"

    def render(self) -> RenderableType:
        if not self.tasks:
            return Text(self.empty_message, style="dim")

        grid = Table.grid(padding=(0, 1), expand=True)
        grid.add_column("status", width=1, no_wrap=True)
        grid.add_column("content", ratio=1, no_wrap=False, overflow="fold")
        for t in self.tasks:
            status = str(t.get("status", "pending")).strip()
            raw_content = t.get("content", "?")
            if isinstance(raw_content, Text):
                content = raw_content.plain
            else:
                content = str(raw_content)
            content = content.strip() or "?"
            icon, style = self._status_icon(status)
            grid.add_row(
                Text(icon, style=style, no_wrap=True),
                Text(content, no_wrap=False, overflow="fold"),
            )
        return grid

    def watch_tasks(self, _tasks: list[dict]) -> None:
        self._scroll_to_latest()

    def watch_empty_message(self, _message: str) -> None:
        self._scroll_to_latest()

    def _scroll_to_latest(self) -> None:
        """Keep the newest rows visible for streaming process-run updates."""
        if not self._auto_follow or not self.is_attached:
            return
        self.call_after_refresh(self.scroll_end, animate=False)


class Sidebar(Vertical):
    """Left sidebar: workspace tree + task progress."""

    DEFAULT_CSS = """
    Sidebar {
        width: 30;
        dock: left;
        background: $panel;
        border-right: solid $surface-lighten-2;
        overflow-y: auto;
    }
    Sidebar.compact {
        width: 24;
    }
    Sidebar.hidden {
        display: none;
    }
    Sidebar #sidebar-label {
        padding: 0 1;
        color: $text-muted;
        text-style: bold;
    }
    Sidebar #sidebar-divider {
        padding: 0 1;
        color: $text-muted;
    }
    Sidebar #progress-label {
        padding: 0 1;
        color: $text-muted;
        text-style: bold;
    }
    Sidebar TaskProgressPanel {
        padding: 0 1;
        height: auto;
        max-height: 14;
        overflow-y: auto;
    }
    Sidebar FilteredTree {
        height: 1fr;
    }
    """

    def __init__(
        self,
        workspace: Path,
        *,
        progress_auto_follow: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._workspace = workspace
        self._progress_auto_follow = bool(progress_auto_follow)

    def compose(self) -> ComposeResult:
        yield Label("Workspace", id="sidebar-label")
        yield FilteredTree(str(self._workspace), id="workspace-tree")
        divider = "\u2500" * 10
        yield Label(divider, id="sidebar-divider")
        yield Label("Progress", id="progress-label")
        yield TaskProgressPanel(
            id="task-progress",
            auto_follow=self._progress_auto_follow,
        )

    def toggle(self) -> None:
        """Toggle visibility."""
        self.toggle_class("hidden")

    def update_tasks(self, tasks: list[dict]) -> None:
        """Update the task progress panel with new task data."""
        panel = self.query_one("#task-progress", TaskProgressPanel)
        label = self.query_one("#progress-label", Label)
        panel.tasks = tasks
        has_rows = bool(tasks)
        label.display = has_rows
        panel.display = has_rows
        if has_rows:
            self.remove_class("compact")
        else:
            self.add_class("compact")

    def refresh_workspace_tree(self) -> None:
        """Reload workspace tree so newly-created files become visible."""
        try:
            tree = self.query_one("#workspace-tree", FilteredTree)
        except Exception:
            return
        tree.reload()

    def on_mount(self) -> None:
        """Start with compact layout until task progress exists."""
        self.add_class("compact")
        try:
            self.query_one("#progress-label", Label).display = False
            self.query_one("#task-progress", TaskProgressPanel).display = False
        except Exception:
            return
