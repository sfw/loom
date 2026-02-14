"""Workspace management: changelog, snapshots, diff, and revert.

Every file-modifying operation records an entry BEFORE the modification.
This enables full undo capability at per-file, per-subtask, or whole-task level.
"""

from __future__ import annotations

import difflib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ChangeEntry:
    """A single entry in the workspace changelog."""

    id: int
    timestamp: str
    operation: str  # create, modify, delete, rename
    path: str  # Relative to workspace
    subtask_id: str = ""
    before_snapshot: str | None = None  # Absolute path to snapshot
    new_path: str | None = None  # For rename operations


class ChangeLog:
    """Tracks every file modification in the workspace for undo capability."""

    def __init__(self, task_id: str, workspace: Path, data_dir: Path):
        self._task_id = task_id
        self._workspace = workspace.resolve()
        self._data_dir = data_dir
        self._log_path = data_dir / "changelog.json"
        self._snapshots_dir = data_dir / "snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._entries: list[ChangeEntry] = []
        self._load()

    def _load(self) -> None:
        """Load existing changelog from disk."""
        if self._log_path.exists():
            data = json.loads(self._log_path.read_text())
            self._entries = [
                ChangeEntry(
                    id=e["id"],
                    timestamp=e["timestamp"],
                    operation=e["operation"],
                    path=e["path"],
                    subtask_id=e.get("subtask_id", ""),
                    before_snapshot=e.get("before_snapshot"),
                    new_path=e.get("new_path"),
                )
                for e in data.get("entries", [])
            ]

    def _save(self) -> None:
        """Persist changelog to disk."""
        data = {
            "task_id": self._task_id,
            "workspace": str(self._workspace),
            "entries": [
                {
                    "id": e.id,
                    "timestamp": e.timestamp,
                    "operation": e.operation,
                    "path": e.path,
                    "subtask_id": e.subtask_id,
                    "before_snapshot": e.before_snapshot,
                    "new_path": e.new_path,
                }
                for e in self._entries
            ],
        }
        self._log_path.write_text(json.dumps(data, indent=2))

    def _next_id(self) -> int:
        if not self._entries:
            return 1
        return self._entries[-1].id + 1

    def record_before_write(
        self, path: str, subtask_id: str = ""
    ) -> None:
        """Record a change BEFORE writing to a file.

        Snapshots the current content if the file exists (modify),
        records as create if it doesn't.
        """
        abs_path = (self._workspace / path).resolve()
        entry_id = self._next_id()

        if abs_path.exists():
            snapshot_name = f"{entry_id}_{abs_path.name}"
            snapshot_path = self._snapshots_dir / snapshot_name
            shutil.copy2(abs_path, snapshot_path)
            operation = "modify"
            before = str(snapshot_path)
        else:
            operation = "create"
            before = None

        self._entries.append(ChangeEntry(
            id=entry_id,
            timestamp=datetime.now().isoformat(),
            operation=operation,
            path=path,
            subtask_id=subtask_id,
            before_snapshot=before,
        ))
        self._save()

    def record_delete(self, path: str, subtask_id: str = "") -> None:
        """Record BEFORE deleting a file. Snapshots the content."""
        abs_path = (self._workspace / path).resolve()
        if not abs_path.exists():
            return

        entry_id = self._next_id()
        snapshot_name = f"{entry_id}_{abs_path.name}"
        snapshot_path = self._snapshots_dir / snapshot_name
        shutil.copy2(abs_path, snapshot_path)

        self._entries.append(ChangeEntry(
            id=entry_id,
            timestamp=datetime.now().isoformat(),
            operation="delete",
            path=path,
            subtask_id=subtask_id,
            before_snapshot=str(snapshot_path),
        ))
        self._save()

    def record_rename(
        self, old_path: str, new_path: str, subtask_id: str = ""
    ) -> None:
        """Record BEFORE renaming a file."""
        self._entries.append(ChangeEntry(
            id=self._next_id(),
            timestamp=datetime.now().isoformat(),
            operation="rename",
            path=old_path,
            subtask_id=subtask_id,
            new_path=new_path,
        ))
        self._save()

    def get_entries(self) -> list[ChangeEntry]:
        """Return all changelog entries."""
        return list(self._entries)

    def get_summary(self) -> dict[str, list[str]]:
        """Return a summary for the TUI files panel.

        Groups files by operation type, deduplicating.
        """
        created: set[str] = set()
        modified: set[str] = set()
        deleted: set[str] = set()
        renamed: list[str] = []

        for e in self._entries:
            if e.operation == "create":
                created.add(e.path)
            elif e.operation == "modify":
                modified.add(e.path)
            elif e.operation == "delete":
                deleted.add(e.path)
            elif e.operation == "rename":
                renamed.append(f"{e.path} -> {e.new_path}")

        return {
            "created": sorted(created),
            "modified": sorted(modified),
            "deleted": sorted(deleted),
            "renamed": renamed,
        }

    def _verify_in_workspace(self, path: Path) -> None:
        """Ensure resolved path stays within workspace."""
        try:
            path.resolve().relative_to(self._workspace)
        except ValueError:
            raise ValueError(f"Path '{path}' escapes workspace '{self._workspace}'")

    def revert_entry(self, entry_id: int) -> None:
        """Revert a single changelog entry."""
        entry = self._find_entry(entry_id)
        if entry is None:
            raise ValueError(f"Changelog entry not found: {entry_id}")

        target = (self._workspace / entry.path).resolve()
        self._verify_in_workspace(target)

        if entry.operation == "create":
            target.unlink(missing_ok=True)
        elif entry.operation in ("modify", "delete"):
            if entry.before_snapshot:
                snapshot = Path(entry.before_snapshot).resolve()
                # Snapshot must be within our data dir
                try:
                    snapshot.relative_to(self._data_dir.resolve())
                except ValueError:
                    raise ValueError(f"Snapshot path '{snapshot}' is not in data dir")
                if snapshot.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(snapshot, target)
        elif entry.operation == "rename":
            if entry.new_path:
                current = (self._workspace / entry.new_path).resolve()
                self._verify_in_workspace(current)
                if current.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    current.rename(target)

    def revert_all(self) -> int:
        """Revert all changes in reverse order. Returns count of reverted entries."""
        count = 0
        for entry in reversed(self._entries):
            self.revert_entry(entry.id)
            count += 1
        return count

    def revert_subtask(self, subtask_id: str) -> int:
        """Revert all changes made by a specific subtask."""
        entries = [e for e in self._entries if e.subtask_id == subtask_id]
        count = 0
        for entry in reversed(entries):
            self.revert_entry(entry.id)
            count += 1
        return count

    def _find_entry(self, entry_id: int) -> ChangeEntry | None:
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None


class DiffGenerator:
    """Generate unified diffs between snapshots and current files."""

    def __init__(self, workspace: Path):
        self._workspace = workspace.resolve()

    def generate(self, changelog: ChangeLog, file_path: str) -> str:
        """Generate unified diff for a changed file.

        Uses the before-snapshot from changelog and current file content.
        """
        entries = [e for e in changelog.get_entries() if e.path == file_path]
        if not entries:
            return ""

        # Find the first snapshot (original state)
        first_with_snapshot = None
        for e in entries:
            if e.before_snapshot:
                first_with_snapshot = e
                break

        if first_with_snapshot and first_with_snapshot.before_snapshot:
            snapshot_path = Path(first_with_snapshot.before_snapshot)
            if snapshot_path.exists():
                before = snapshot_path.read_text(errors="replace").splitlines()
            else:
                before = []
        else:
            before = []  # New file — diff against empty

        current_path = self._workspace / file_path
        if current_path.exists():
            after = current_path.read_text(errors="replace").splitlines()
        else:
            after = []  # File was deleted

        return "\n".join(difflib.unified_diff(
            before,
            after,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))


def validate_workspace(path: str | Path) -> tuple[bool, str]:
    """Validate that a workspace path is usable.

    Returns (is_valid, message).
    """
    p = Path(path)
    if not p.exists():
        return False, f"Path does not exist: {path}"
    if not p.is_dir():
        return False, f"Path is not a directory: {path}"
    if not os.access(p, os.R_OK):
        return False, f"Path is not readable: {path}"
    if not os.access(p, os.W_OK):
        return True, f"Warning: path is not writable: {path}"
    return True, "OK"


# Avoid import at module level for os.access
import os  # noqa: E402
