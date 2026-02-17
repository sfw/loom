"""Sidebar widget: workspace file tree + task progress panel."""

from __future__ import annotations

from pathlib import Path

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

    def render(self) -> str:
        if not self.tasks:
            return "[dim]No tasks tracked[/dim]"

        lines: list[str] = []
        for t in self.tasks:
            status = t.get("status", "pending")
            content = t.get("content", "?")
            if status == "completed":
                icon = "[#9ece6a]\u2713[/]"
            elif status == "in_progress":
                icon = "[#7dcfff]\u25c9[/]"
            elif status == "failed":
                icon = "[#f7768e]\u2717[/]"
            elif status == "skipped":
                icon = "[dim]-[/dim]"
            else:
                icon = "[dim]\u25cb[/dim]"
            lines.append(f"{icon} {content}")
        return "\n".join(lines)


class Sidebar(Vertical):
    """Left sidebar: workspace tree + task progress."""

    DEFAULT_CSS = """
    Sidebar {
        width: 32;
        dock: left;
        background: $panel;
        border-right: solid $surface-lighten-2;
        overflow-y: auto;
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

    def __init__(self, workspace: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._workspace = workspace

    def compose(self) -> ComposeResult:
        yield Label("Workspace", id="sidebar-label")
        yield FilteredTree(str(self._workspace), id="workspace-tree")
        divider = "\u2500" * 10
        yield Label(divider, id="sidebar-divider")
        yield Label("Progress", id="progress-label")
        yield TaskProgressPanel(id="task-progress")

    def toggle(self) -> None:
        """Toggle visibility."""
        self.toggle_class("hidden")

    def update_tasks(self, tasks: list[dict]) -> None:
        """Update the task progress panel with new task data."""
        panel = self.query_one("#task-progress", TaskProgressPanel)
        panel.tasks = tasks

    def refresh_workspace_tree(self) -> None:
        """Reload workspace tree so newly-created files become visible."""
        try:
            tree = self.query_one("#workspace-tree", FilteredTree)
        except Exception:
            return
        tree.reload()
