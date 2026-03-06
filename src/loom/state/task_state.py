"""Layer 1: Always-in-context task state.

The YAML state object is included in every prompt. It must stay compact
(~500-1500 tokens) and always reflect the current truth.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml

from loom.state.evidence import merge_evidence_records


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
    phase_id: str = ""
    model_tier: int = 1
    verification_tier: int = 1
    is_critical_path: bool = False
    is_synthesis: bool = False
    acceptance_criteria: str = ""
    validity_contract_snapshot: dict[str, object] = field(default_factory=dict)
    validity_contract_hash: str = ""
    retry_count: int = 0
    max_retries: int = 3
    iteration_attempt: int = 0
    iteration_runner_invocations: int = 0
    iteration_max_attempts: int = 0
    iteration_no_improvement_count: int = 0
    iteration_best_score: float | None = None
    iteration_terminal_reason: str = ""
    iteration_loop_run_id: str = ""
    iteration_replan_count: int = 0
    iteration_last_gate_summary: str = ""


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

    def _evidence_path(self, task_id: str) -> Path:
        return self._task_dir(task_id) / "evidence-ledger.json"

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

    def load_evidence_records(self, task_id: str) -> list[dict]:
        """Load persisted evidence records for a task.

        Returns an empty list when no evidence ledger exists.
        """
        evidence_path = self._evidence_path(task_id)
        if not evidence_path.exists():
            return []
        try:
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def save_evidence_records(self, task_id: str, records: list[dict]) -> None:
        """Persist evidence records atomically."""
        evidence_path = self._evidence_path(task_id)
        safe_records = [item for item in records if isinstance(item, dict)]
        content = json.dumps(
            safe_records,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        fd, tmp_path = tempfile.mkstemp(
            dir=evidence_path.parent,
            suffix=".json.tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, evidence_path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def append_evidence_records(self, task_id: str, records: list[dict]) -> list[dict]:
        """Merge and persist additional evidence records for a task."""
        existing = self.load_evidence_records(task_id)
        merged = merge_evidence_records(existing, records)
        self.save_evidence_records(task_id, merged)
        return merged

    def to_yaml(self, task: Task) -> str:
        """Render task state as YAML for prompt injection."""
        data = self._to_dict(task)
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def to_compact_yaml(self, task: Task) -> str:
        """Render YAML for prompt injection without lossy hard compaction."""
        data = self._to_dict(task)
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _to_dict(self, task: Task) -> dict:
        """Convert Task dataclass to a dict for YAML serialization."""
        subtasks_data = []
        for s in task.plan.subtasks:
            entry: dict = {"id": s.id, "status": s.status.value}
            entry["model_tier"] = int(s.model_tier or 1)
            entry["verification_tier"] = int(s.verification_tier or 1)
            if s.summary:
                entry["summary"] = s.summary
            if s.active_issue:
                entry["active_issue"] = s.active_issue
            if s.depends_on:
                entry["depends_on"] = s.depends_on
            if s.phase_id:
                entry["phase_id"] = s.phase_id
            if s.description:
                entry["description"] = s.description
            if s.is_critical_path:
                entry["is_critical_path"] = True
            if s.is_synthesis:
                entry["is_synthesis"] = True
            if s.acceptance_criteria:
                entry["acceptance_criteria"] = s.acceptance_criteria
            if s.validity_contract_snapshot:
                entry["validity_contract_snapshot"] = s.validity_contract_snapshot
            if s.validity_contract_hash:
                entry["validity_contract_hash"] = s.validity_contract_hash
            if s.retry_count:
                entry["retry_count"] = int(s.retry_count)
            if s.max_retries != 3:
                entry["max_retries"] = int(s.max_retries)
            if s.iteration_attempt:
                entry["iteration_attempt"] = int(s.iteration_attempt)
            if s.iteration_runner_invocations:
                entry["iteration_runner_invocations"] = int(
                    s.iteration_runner_invocations,
                )
            if s.iteration_max_attempts:
                entry["iteration_max_attempts"] = int(s.iteration_max_attempts)
            if s.iteration_no_improvement_count:
                entry["iteration_no_improvement_count"] = int(
                    s.iteration_no_improvement_count,
                )
            if s.iteration_best_score is not None:
                entry["iteration_best_score"] = float(s.iteration_best_score)
            if s.iteration_terminal_reason:
                entry["iteration_terminal_reason"] = s.iteration_terminal_reason
            if s.iteration_loop_run_id:
                entry["iteration_loop_run_id"] = s.iteration_loop_run_id
            if s.iteration_replan_count:
                entry["iteration_replan_count"] = int(s.iteration_replan_count)
            if s.iteration_last_gate_summary:
                entry["iteration_last_gate_summary"] = s.iteration_last_gate_summary
            subtasks_data.append(entry)

        errors_data = []
        for e in task.errors_encountered:
            entry = {"subtask": e.subtask, "error": e.error}
            if e.resolution:
                entry["resolution"] = e.resolution
            errors_data.append(entry)

        task_dict: dict = {
            "id": task.id,
            "goal": task.goal,
            "status": task.status.value,
            "workspace": task.workspace,
            "approval_mode": task.approval_mode,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        }
        if task.callback_url:
            task_dict["callback_url"] = task.callback_url
        if task.context:
            task_dict["context"] = task.context
        if task.metadata:
            task_dict["metadata"] = task.metadata
        if task.completed_at:
            task_dict["completed_at"] = task.completed_at

        result = {
            "task": task_dict,
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
            validity_snapshot_raw = s.get("validity_contract_snapshot", {})
            if not isinstance(validity_snapshot_raw, dict):
                validity_snapshot_raw = {}
            best_score_raw = s.get("iteration_best_score")
            try:
                best_score = (
                    float(best_score_raw) if best_score_raw is not None else None
                )
            except (TypeError, ValueError):
                best_score = None
            subtasks.append(Subtask(
                id=s["id"],
                description=s.get("description", ""),
                status=SubtaskStatus(s.get("status", "pending")),
                summary=s.get("summary", ""),
                active_issue=s.get("active_issue", ""),
                depends_on=s.get("depends_on", []),
                phase_id=s.get("phase_id", ""),
                model_tier=int(s.get("model_tier", 1) or 1),
                verification_tier=int(s.get("verification_tier", 1) or 1),
                is_critical_path=bool(s.get("is_critical_path", False)),
                is_synthesis=bool(s.get("is_synthesis", False)),
                acceptance_criteria=str(s.get("acceptance_criteria", "") or ""),
                validity_contract_snapshot=dict(validity_snapshot_raw),
                validity_contract_hash=str(s.get("validity_contract_hash", "") or ""),
                retry_count=int(s.get("retry_count", 0) or 0),
                max_retries=int(s.get("max_retries", 3) or 3),
                iteration_attempt=int(s.get("iteration_attempt", 0) or 0),
                iteration_runner_invocations=int(
                    s.get("iteration_runner_invocations", 0) or 0,
                ),
                iteration_max_attempts=int(s.get("iteration_max_attempts", 0) or 0),
                iteration_no_improvement_count=int(
                    s.get("iteration_no_improvement_count", 0) or 0,
                ),
                iteration_best_score=best_score,
                iteration_terminal_reason=str(
                    s.get("iteration_terminal_reason", "") or "",
                ),
                iteration_loop_run_id=str(
                    s.get("iteration_loop_run_id", "") or "",
                ),
                iteration_replan_count=int(s.get("iteration_replan_count", 0) or 0),
                iteration_last_gate_summary=str(
                    s.get("iteration_last_gate_summary", "") or "",
                ),
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
            approval_mode=task_data.get("approval_mode", "auto"),
            callback_url=task_data.get("callback_url", ""),
            context=task_data.get("context", {}),
            metadata=task_data.get("metadata", {}),
            created_at=task_data.get("created_at", ""),
            updated_at=task_data.get("updated_at", ""),
            completed_at=task_data.get("completed_at", ""),
            plan=Plan(
                subtasks=subtasks,
                version=plan_data.get("version", 1),
                last_replanned=plan_data.get("last_replanned", ""),
            ),
            decisions_log=data.get("decisions_log", []),
            errors_encountered=errors,
            workspace_changes=changes,
        )
