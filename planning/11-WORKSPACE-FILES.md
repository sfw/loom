# Spec 11: Workspace and File Management

## Overview

When a task is created, the user mounts a working directory. Loom reads from and writes to this directory, tracks every change in a changelog, and provides full undo capability. The workspace is the bridge between Loom's execution and the user's real project.

## Workspace Mounting

### On Task Creation

```python
POST /tasks {
    "goal": "Add comprehensive test coverage",
    "workspace": "/Users/scott/projects/myapp",
    ...
}
```

The engine validates:
1. Path exists and is a directory
2. Path is readable
3. Path is writable (warn if not — some tasks may only need to read)

### Workspace Structure

```
/Users/scott/projects/myapp/          ← user's project (mounted read-write)

~/.loom/tasks/{task_id}/
    ├── state.yaml                    ← Layer 1 task state (Spec 03)
    ├── changelog.json                ← every file operation tracked
    ├── scratch/                      ← temp files, intermediate work
    │   ├── temp_script.py
    │   └── test_output.log
    └── artifacts/                    ← task outputs (reports, generated files)
```

The user's project is never polluted with Loom metadata. All Loom-specific data lives in `~/.loom/tasks/{task_id}/`.

## Changelog

Every file-modifying operation records an entry before the modification happens. This enables full undo.

### Changelog Format

```json
{
  "task_id": "a1b2c3d4",
  "workspace": "/Users/scott/projects/myapp",
  "entries": [
    {
      "id": 1,
      "timestamp": "2026-02-13T14:30:01",
      "subtask_id": "install-deps",
      "operation": "modify",
      "path": "package.json",
      "before_snapshot": "/Users/scott/.loom/tasks/a1b2c3d4/snapshots/1_package.json",
      "after_hash": "sha256:abc123..."
    },
    {
      "id": 2,
      "timestamp": "2026-02-13T14:30:05",
      "subtask_id": "add-tsconfig",
      "operation": "create",
      "path": "tsconfig.json",
      "before_snapshot": null,
      "after_hash": "sha256:def456..."
    },
    {
      "id": 3,
      "timestamp": "2026-02-13T14:31:12",
      "subtask_id": "rename-files",
      "operation": "rename",
      "path": "src/app.js",
      "new_path": "src/app.ts",
      "before_snapshot": null
    },
    {
      "id": 4,
      "timestamp": "2026-02-13T14:32:00",
      "subtask_id": "add-types",
      "operation": "modify",
      "path": "src/app.ts",
      "before_snapshot": "/Users/scott/.loom/tasks/a1b2c3d4/snapshots/4_app.ts",
      "after_hash": "sha256:ghi789..."
    }
  ]
}
```

### Operations

| Operation | Before Snapshot | Description |
|-----------|----------------|-------------|
| `create` | null | New file created |
| `modify` | Original file saved | File content changed |
| `delete` | Original file saved | File removed |
| `rename` | null | File moved/renamed |

### Changelog Manager

```python
class ChangeLog:
    def __init__(self, task_id: str, workspace: Path, data_dir: Path):
        self._task_id = task_id
        self._workspace = workspace
        self._log_path = data_dir / "changelog.json"
        self._snapshots_dir = data_dir / "snapshots"
        self._snapshots_dir.mkdir(exist_ok=True)

    async def record_before_write(self, path: Path) -> None:
        """
        Call BEFORE writing to a file. Snapshots the current content
        for undo capability.
        """
        relative = path.relative_to(self._workspace)
        entry_id = self._next_id()

        if path.exists():
            # Snapshot existing file
            snapshot_path = self._snapshots_dir / f"{entry_id}_{path.name}"
            shutil.copy2(path, snapshot_path)
            operation = "modify"
        else:
            snapshot_path = None
            operation = "create"

        self._append_entry({
            "id": entry_id,
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "path": str(relative),
            "before_snapshot": str(snapshot_path) if snapshot_path else None,
        })

    async def record_delete(self, path: Path) -> None:
        """Call BEFORE deleting a file."""
        snapshot_path = self._snapshots_dir / f"{self._next_id()}_{path.name}"
        shutil.copy2(path, snapshot_path)
        self._append_entry({
            "operation": "delete",
            "path": str(path.relative_to(self._workspace)),
            "before_snapshot": str(snapshot_path),
        })

    async def record_rename(self, old_path: Path, new_path: Path) -> None:
        """Call BEFORE renaming a file."""
        self._append_entry({
            "operation": "rename",
            "path": str(old_path.relative_to(self._workspace)),
            "new_path": str(new_path.relative_to(self._workspace)),
        })

    def get_changes(self) -> list[dict]:
        """Return all changelog entries."""
        ...

    def get_changes_summary(self) -> dict:
        """
        Return a summary suitable for the TUI files panel:
        {"created": ["tsconfig.json"], "modified": ["package.json", "src/app.ts"], ...}
        """
        ...

    async def revert_entry(self, entry_id: int) -> None:
        """Revert a single changelog entry."""
        entry = self._get_entry(entry_id)
        target = self._workspace / entry["path"]

        if entry["operation"] == "create":
            target.unlink(missing_ok=True)
        elif entry["operation"] in ("modify", "delete"):
            shutil.copy2(entry["before_snapshot"], target)
        elif entry["operation"] == "rename":
            current = self._workspace / entry["new_path"]
            current.rename(target)

    async def revert_all(self) -> None:
        """Revert all changes in reverse order."""
        entries = self.get_changes()
        for entry in reversed(entries):
            await self.revert_entry(entry["id"])

    async def revert_subtask(self, subtask_id: str) -> None:
        """Revert all changes made by a specific subtask."""
        entries = [e for e in self.get_changes() if e.get("subtask_id") == subtask_id]
        for entry in reversed(entries):
            await self.revert_entry(entry["id"])
```

## Diff Generation

For the TUI and API, generate diffs between before-snapshot and current file:

```python
class DiffGenerator:
    def generate(self, changelog: ChangeLog, file_path: str) -> str:
        """
        Generate unified diff for a changed file.
        Uses the before-snapshot from changelog and current file content.
        Returns standard unified diff format.
        """
        entry = changelog.find_latest_entry(file_path)
        if not entry or not entry.get("before_snapshot"):
            # New file — diff against empty
            current = (workspace / file_path).read_text().splitlines()
            return "\n".join(f"+{line}" for line in current)

        before = Path(entry["before_snapshot"]).read_text().splitlines()
        after = (workspace / file_path).read_text().splitlines()

        return "\n".join(difflib.unified_diff(
            before, after,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        ))
```

## Scratch Directory

Each task gets a scratch directory for intermediate work that shouldn't pollute the user's project:

- Temp scripts the model needs to run
- Test output logs
- Intermediate file transformations
- Downloaded resources

Tools can write to scratch via a `scratch://` path prefix:
```json
{"path": "scratch://temp_analysis.py"}
```

Resolves to `~/.loom/tasks/{task_id}/scratch/temp_analysis.py`.

## API Endpoints for Workspace

```
GET  /tasks/{task_id}/files                      List changed files with summary
GET  /tasks/{task_id}/files/{path}/diff           Get diff for a specific file
POST /tasks/{task_id}/files/{path}/revert         Revert a specific file
POST /tasks/{task_id}/revert                      Revert all changes
POST /tasks/{task_id}/subtasks/{sid}/revert       Revert changes from one subtask
```

## Acceptance Criteria

- [ ] Workspace path is validated on task creation (exists, is directory)
- [ ] All file tools resolve paths relative to workspace
- [ ] Every file modification records a changelog entry BEFORE the change
- [ ] Snapshots capture original file content accurately
- [ ] Single-file revert restores the original content
- [ ] Full revert restores workspace to pre-task state
- [ ] Per-subtask revert only undoes that subtask's changes
- [ ] Diff generation produces valid unified diff format
- [ ] Scratch directory is created per-task and isolated
- [ ] Changed files summary is available for TUI display
- [ ] Changelog survives engine restart (persisted to disk)
- [ ] New file creation records as "create" with no before-snapshot
- [ ] Renaming is tracked and revertible
