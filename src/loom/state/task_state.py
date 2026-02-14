"""Layer 1: Always-in-context task state.

The YAML state object is included in every prompt. It must stay compact
(~500-1500 tokens) and always reflect the current truth.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml


class TaskStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubtaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    id: str
    description: str
    status: SubtaskStatus = SubtaskStatus.PENDING
    summary: str = ""
    active_issue: str = ""
    depends_on: list[str] = field(default_factory=list)
    model_tier: int = 1
    verification_tier: int = 1
    is_critical_path: bool = False
    acceptance_criteria: str = ""
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ErrorRecord:
    subtask: str
    error: str
    resolution: str | None = None


@dataclass
class WorkspaceChanges:
    files_created: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    last_change: str = ""


@dataclass
class Plan:
    subtasks: list[Subtask] = field(default_factory=list)
    version: int = 1
    last_replanned: str = ""


@dataclass
class Task:
    id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    workspace: str = ""
    plan: Plan = field(default_factory=Plan)
    decisions_log: list[str] = field(default_factory=list)
    errors_encountered: list[ErrorRecord] = field(default_factory=list)
    workspace_changes: WorkspaceChanges = field(default_factory=WorkspaceChanges)
    approval_mode: str = "auto"
    callback_url: str = ""
    context: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    completed_at: str = ""
    metadata: dict = field(default_factory=dict)

    # Size budget limits
    MAX_DECISIONS: int = 10
    MAX_ERRORS: int = 5
    MAX_SUBTASKS_DISPLAY: int = 15

    def add_decision(self, decision: str) -> None:
        self.decisions_log.append(decision)
        if len(self.decisions_log) > self.MAX_DECISIONS:
            self.decisions_log = self.decisions_log[-self.MAX_DECISIONS :]

    def add_error(self, subtask_id: str, error: str, resolution: str | None = None) -> None:
        self.errors_encountered.append(
            ErrorRecord(subtask=subtask_id, error=error, resolution=resolution)
        )
        if len(self.errors_encountered) > self.MAX_ERRORS:
            self.errors_encountered = self.errors_encountered[-self.MAX_ERRORS :]

    def get_subtask(self, subtask_id: str) -> Subtask | None:
        for s in self.plan.subtasks:
            if s.id == subtask_id:
                return s
        return None

    def update_subtask(self, subtask_id: str, **updates) -> None:
        subtask = self.get_subtask(subtask_id)
        if subtask is None:
            raise ValueError(f"Subtask not found: {subtask_id}")
        for key, value in updates.items():
            if hasattr(subtask, key):
                setattr(subtask, key, value)
        self.updated_at = datetime.now().isoformat()

    @property
    def progress(self) -> tuple[int, int]:
        """Return (completed, total) subtask counts."""
        total = len(self.plan.subtasks)
        completed = sum(1 for s in self.plan.subtasks if s.status == SubtaskStatus.COMPLETED)
        return completed, total


class TaskStateManager:
    """Manages Layer 1 state files on disk."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def _task_dir(self, task_id: str) -> Path:
        d = self._data_dir / "tasks" / task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _state_path(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "state.yaml"

    def create(self, task: Task) -> None:
        """Create initial state file for a new task."""
        if not task.created_at:
            task.created_at = datetime.now().isoformat()
        if not task.updated_at:
            task.updated_at = task.created_at
        self.save(task)

    def save(self, task: Task) -> None:
        """Atomic write: write to temp file, then rename."""
        task.updated_at = datetime.now().isoformat()
        state_path = self._state_path(task.id)
        yaml_content = self.to_yaml(task)

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(dir=state_path.parent, suffix=".yaml.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(yaml_content)
            os.replace(tmp_path, state_path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def load(self, task_id: str) -> Task:
        """Load task state from disk."""
        state_path = self._state_path(task_id)
        if not state_path.exists():
            raise FileNotFoundError(f"No state file for task: {task_id}")

        data = yaml.safe_load(state_path.read_text())
        return self._from_dict(data)

    def exists(self, task_id: str) -> bool:
        return self._state_path(task_id).exists()

    def to_yaml(self, task: Task) -> str:
        """Render task state as YAML for prompt injection."""
        data = self._to_dict(task)
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def to_compact_yaml(self, task: Task) -> str:
        """Render a compact version for prompt injection.

        If >15 subtasks, only show completed count + current + next 3 pending.
        """
        data = self._to_dict(task)

        subtasks = data.get("subtasks", [])
        if len(subtasks) > task.MAX_SUBTASKS_DISPLAY:
            completed = [s for s in subtasks if s["status"] == "completed"]
            current = [s for s in subtasks if s["status"] in ("running", "blocked")]
            pending = [s for s in subtasks if s["status"] == "pending"][:3]

            data["subtasks"] = (
                [{"_completed_count": len(completed)}] + current + pending
            )

        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _to_dict(self, task: Task) -> dict:
        """Convert Task dataclass to a dict for YAML serialization."""
        subtasks_data = []
        for s in task.plan.subtasks:
            entry: dict = {"id": s.id, "status": s.status.value}
            if s.summary:
                entry["summary"] = s.summary[:100]
            if s.active_issue:
                entry["active_issue"] = s.active_issue
            if s.depends_on:
                entry["depends_on"] = s.depends_on
            if s.description:
                entry["description"] = s.description
            subtasks_data.append(entry)

        errors_data = []
        for e in task.errors_encountered:
            entry = {"subtask": e.subtask, "error": e.error}
            if e.resolution:
                entry["resolution"] = e.resolution
            errors_data.append(entry)

        result = {
            "task": {
                "id": task.id,
                "goal": task.goal,
                "status": task.status.value,
                "workspace": task.workspace,
            },
            "plan": {
                "version": task.plan.version,
                "last_replanned": task.plan.last_replanned,
            },
            "subtasks": subtasks_data,
        }

        if task.decisions_log:
            result["decisions_log"] = task.decisions_log

        if task.errors_encountered:
            result["errors_encountered"] = errors_data

        changes = task.workspace_changes
        if changes.files_created or changes.files_modified or changes.files_deleted:
            result["workspace_changes"] = {
                "files_created": changes.files_created,
                "files_modified": changes.files_modified,
                "files_deleted": changes.files_deleted,
                "last_change": changes.last_change,
            }

        return result

    def _from_dict(self, data: dict) -> Task:
        """Reconstruct a Task from a dict loaded from YAML."""
        task_data = data.get("task", {})
        plan_data = data.get("plan", {})

        subtasks = []
        for s in data.get("subtasks", []):
            if "_completed_count" in s:
                continue
            subtasks.append(Subtask(
                id=s["id"],
                description=s.get("description", ""),
                status=SubtaskStatus(s.get("status", "pending")),
                summary=s.get("summary", ""),
                active_issue=s.get("active_issue", ""),
                depends_on=s.get("depends_on", []),
            ))

        errors = []
        for e in data.get("errors_encountered", []):
            errors.append(ErrorRecord(
                subtask=e["subtask"],
                error=e["error"],
                resolution=e.get("resolution"),
            ))

        changes_data = data.get("workspace_changes", {})
        changes = WorkspaceChanges(
            files_created=changes_data.get("files_created", 0),
            files_modified=changes_data.get("files_modified", 0),
            files_deleted=changes_data.get("files_deleted", 0),
            last_change=changes_data.get("last_change", ""),
        )

        return Task(
            id=task_data.get("id", ""),
            goal=task_data.get("goal", ""),
            status=TaskStatus(task_data.get("status", "pending")),
            workspace=task_data.get("workspace", ""),
            plan=Plan(
                subtasks=subtasks,
                version=plan_data.get("version", 1),
                last_replanned=plan_data.get("last_replanned", ""),
            ),
            decisions_log=data.get("decisions_log", []),
            errors_encountered=errors,
            workspace_changes=changes,
        )
