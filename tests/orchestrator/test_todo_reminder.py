"""Orchestrator todo reminder tests."""

from __future__ import annotations

from loom.state.task_state import Subtask
from tests.orchestrator.conftest import _make_task


class TestOrchestratorTodoReminder:
    def test_build_todo_reminder(self):
        task = _make_task(goal="Build a CLI tool")
        subtask = Subtask(id="step-1", description="Create main.py")

        from loom.engine.runner import SubtaskRunner
        reminder = SubtaskRunner._build_todo_reminder(task, subtask)

        assert "Build a CLI tool" in reminder
        assert "step-1" in reminder
        assert "Create main.py" in reminder
        assert "Do NOT move to the next subtask" in reminder
