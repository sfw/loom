"""Tests for Layer 1: Task state management."""

from __future__ import annotations

from pathlib import Path

import yaml

from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStateManager,
    TaskStatus,
    WorkspaceChanges,
)


def _make_task(task_id: str = "test-123", goal: str = "Test goal") -> Task:
    """Create a minimal task for testing."""
    return Task(
        id=task_id,
        goal=goal,
        status=TaskStatus.PENDING,
        workspace="/tmp/test-workspace",
    )


def _make_task_with_plan() -> Task:
    task = _make_task()
    task.plan = Plan(
        subtasks=[
            Subtask(id="step-1", description="First step", status=SubtaskStatus.COMPLETED,
                    summary="Done first"),
            Subtask(id="step-2", description="Second step", status=SubtaskStatus.RUNNING,
                    summary="Working on it"),
            Subtask(id="step-3", description="Third step", status=SubtaskStatus.PENDING,
                    depends_on=["step-2"]),
        ],
        version=1,
    )
    return task


class TestTask:
    """Test Task dataclass behavior."""

    def test_create_minimal(self):
        task = _make_task()
        assert task.id == "test-123"
        assert task.goal == "Test goal"
        assert task.status == TaskStatus.PENDING

    def test_add_decision(self):
        task = _make_task()
        task.add_decision("Using PostgreSQL")
        assert "Using PostgreSQL" in task.decisions_log

    def test_decision_pruning(self):
        task = _make_task()
        for i in range(15):
            task.add_decision(f"Decision {i}")
        assert len(task.decisions_log) == task.MAX_DECISIONS
        assert task.decisions_log[0] == "Decision 5"  # Oldest pruned

    def test_add_error(self):
        task = _make_task()
        task.add_error("step-1", "Something failed", "Fixed it")
        assert len(task.errors_encountered) == 1
        assert task.errors_encountered[0].error == "Something failed"
        assert task.errors_encountered[0].resolution == "Fixed it"

    def test_error_pruning(self):
        task = _make_task()
        for i in range(8):
            task.add_error(f"step-{i}", f"Error {i}")
        assert len(task.errors_encountered) == task.MAX_ERRORS

    def test_get_subtask(self):
        task = _make_task_with_plan()
        s = task.get_subtask("step-2")
        assert s is not None
        assert s.description == "Second step"

    def test_get_subtask_missing(self):
        task = _make_task_with_plan()
        assert task.get_subtask("nonexistent") is None

    def test_update_subtask(self):
        task = _make_task_with_plan()
        task.update_subtask("step-2", status=SubtaskStatus.COMPLETED, summary="Done!")
        s = task.get_subtask("step-2")
        assert s.status == SubtaskStatus.COMPLETED
        assert s.summary == "Done!"

    def test_update_subtask_missing_raises(self):
        task = _make_task()
        try:
            task.update_subtask("nonexistent", status=SubtaskStatus.COMPLETED)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_progress(self):
        task = _make_task_with_plan()
        completed, total = task.progress
        assert completed == 1
        assert total == 3


class TestTaskStateManager:
    """Test state file persistence."""

    def test_create_and_load(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        task = _make_task_with_plan()
        mgr.create(task)

        loaded = mgr.load("test-123")
        assert loaded.id == "test-123"
        assert loaded.goal == "Test goal"
        assert len(loaded.plan.subtasks) == 3

    def test_save_and_load(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        task = _make_task_with_plan()
        mgr.create(task)

        task.status = TaskStatus.EXECUTING
        task.add_decision("Using REST API")
        mgr.save(task)

        loaded = mgr.load("test-123")
        assert loaded.status == TaskStatus.EXECUTING
        assert "Using REST API" in loaded.decisions_log

    def test_exists(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        assert not mgr.exists("nonexistent")

        task = _make_task()
        mgr.create(task)
        assert mgr.exists("test-123")

    def test_load_missing_raises(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        try:
            mgr.load("nonexistent")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_to_yaml_valid(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        task = _make_task_with_plan()
        task.add_decision("Important choice")
        task.add_error("step-1", "Oops", "Fixed")

        yaml_str = mgr.to_yaml(task)
        data = yaml.safe_load(yaml_str)

        assert data["task"]["id"] == "test-123"
        assert data["task"]["goal"] == "Test goal"
        assert len(data["subtasks"]) == 3
        assert "Important choice" in data["decisions_log"]
        assert data["errors_encountered"][0]["error"] == "Oops"

    def test_to_compact_yaml_small_plan(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        task = _make_task_with_plan()
        compact = mgr.to_compact_yaml(task)
        data = yaml.safe_load(compact)
        # Small plan should not be compacted
        assert len(data["subtasks"]) == 3

    def test_to_compact_yaml_large_plan(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(id=f"step-{i}", description=f"Step {i}",
                        status=SubtaskStatus.COMPLETED if i < 12
                        else SubtaskStatus.RUNNING if i == 12
                        else SubtaskStatus.PENDING)
                for i in range(20)
            ]
        )
        compact = mgr.to_compact_yaml(task)
        data = yaml.safe_load(compact)
        # Should be compacted: 1 completed count + 1 running + 3 pending = 5
        assert len(data["subtasks"]) < 20

    def test_roundtrip_preserves_status(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        task = _make_task_with_plan()
        task.status = TaskStatus.EXECUTING
        mgr.create(task)

        loaded = mgr.load("test-123")
        assert loaded.status == TaskStatus.EXECUTING
        assert loaded.plan.subtasks[0].status == SubtaskStatus.COMPLETED
        assert loaded.plan.subtasks[1].status == SubtaskStatus.RUNNING
        assert loaded.plan.subtasks[2].status == SubtaskStatus.PENDING

    def test_workspace_changes_roundtrip(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        task = _make_task()
        task.workspace_changes = WorkspaceChanges(
            files_created=2, files_modified=5, last_change="2026-02-13T14:00:00"
        )
        mgr.create(task)

        loaded = mgr.load("test-123")
        assert loaded.workspace_changes.files_created == 2
        assert loaded.workspace_changes.files_modified == 5

    def test_summary_truncation_in_yaml(self, tmp_path: Path):
        mgr = TaskStateManager(tmp_path)
        task = _make_task()
        task.plan = Plan(subtasks=[
            Subtask(id="s1", description="test", summary="x" * 300),
        ])
        yaml_str = mgr.to_yaml(task)
        data = yaml.safe_load(yaml_str)
        # Truncated to 200 chars + "..."
        assert len(data["subtasks"][0]["summary"]) == 203
        assert data["subtasks"][0]["summary"].endswith("...")

    def test_roundtrip_preserves_metadata_fields(self, tmp_path: Path):
        """All externally visible fields survive YAML round-trip."""
        mgr = TaskStateManager(tmp_path)
        task = _make_task()
        task.approval_mode = "manual"
        task.callback_url = "https://example.com/hook"
        task.context = {"env": "test"}
        task.metadata = {"source": "api"}
        task.created_at = "2026-01-01T00:00:00"
        task.updated_at = "2026-01-01T00:00:01"
        task.completed_at = "2026-01-01T01:00:00"
        mgr.create(task)

        loaded = mgr.load("test-123")
        assert loaded.approval_mode == "manual"
        assert loaded.callback_url == "https://example.com/hook"
        assert loaded.context == {"env": "test"}
        assert loaded.metadata == {"source": "api"}
        assert loaded.created_at == "2026-01-01T00:00:00"
        assert loaded.completed_at == "2026-01-01T01:00:00"
