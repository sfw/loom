"""Tests for the subtask scheduler."""

from __future__ import annotations

from loom.engine.scheduler import Scheduler
from loom.state.task_state import Plan, Subtask, SubtaskStatus


def _plan(*subtasks: Subtask) -> Plan:
    """Create a plan from subtasks."""
    return Plan(subtasks=list(subtasks), version=1)


def _subtask(id: str, depends_on: list[str] | None = None, status: str = "pending") -> Subtask:
    """Create a subtask with defaults."""
    return Subtask(
        id=id,
        description=f"Do {id}",
        status=SubtaskStatus(status),
        depends_on=depends_on or [],
    )


class TestSchedulerNextSubtask:
    def test_simple_linear(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a"),
            _subtask("b", depends_on=["a"]),
            _subtask("c", depends_on=["b"]),
        )

        # First runnable should be 'a' (no deps)
        assert scheduler.next_subtask(plan).id == "a"

    def test_returns_none_when_all_completed(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="completed"),
            _subtask("b", status="completed"),
        )
        assert scheduler.next_subtask(plan) is None

    def test_returns_none_when_blocked(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="running"),
            _subtask("b", depends_on=["a"]),
        )
        assert scheduler.next_subtask(plan) is None

    def test_skips_running(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="running"),
            _subtask("b"),
        )
        # 'a' is running, 'b' has no deps so it's runnable
        assert scheduler.next_subtask(plan).id == "b"

    def test_dep_completed_unlocks(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="completed"),
            _subtask("b", depends_on=["a"]),
        )
        assert scheduler.next_subtask(plan).id == "b"

    def test_multiple_deps_must_all_complete(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="completed"),
            _subtask("b", status="pending"),
            _subtask("c", depends_on=["a", "b"]),
        )
        # 'c' depends on both a and b; b is not done
        # But 'b' itself is runnable
        result = scheduler.next_subtask(plan)
        assert result.id == "b"

    def test_empty_plan(self):
        scheduler = Scheduler()
        plan = _plan()
        assert scheduler.next_subtask(plan) is None


class TestSchedulerRunnableSubtasks:
    def test_multiple_runnable(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a"),
            _subtask("b"),
            _subtask("c", depends_on=["a"]),
        )
        runnable = scheduler.runnable_subtasks(plan)
        ids = [s.id for s in runnable]
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids

    def test_no_runnable(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="running"),
            _subtask("b", depends_on=["a"]),
        )
        assert scheduler.runnable_subtasks(plan) == []


class TestSchedulerHasPending:
    def test_has_pending_true(self):
        scheduler = Scheduler()
        plan = _plan(_subtask("a"))
        assert scheduler.has_pending(plan) is True

    def test_has_pending_running(self):
        scheduler = Scheduler()
        plan = _plan(_subtask("a", status="running"))
        assert scheduler.has_pending(plan) is True

    def test_has_pending_false(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="completed"),
            _subtask("b", status="failed"),
        )
        assert scheduler.has_pending(plan) is False


class TestSchedulerIsBlocked:
    def test_blocked_when_deps_unmet(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="failed"),
            _subtask("b", depends_on=["a"]),
        )
        assert scheduler.is_blocked(plan) is True

    def test_not_blocked_when_runnable(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a"),
            _subtask("b", depends_on=["a"]),
        )
        assert scheduler.is_blocked(plan) is False

    def test_not_blocked_when_nothing_pending(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="completed"),
        )
        assert scheduler.is_blocked(plan) is False

    def test_blocked_circular_deps(self):
        """Subtasks that depend on each other should be blocked."""
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", depends_on=["b"]),
            _subtask("b", depends_on=["a"]),
        )
        assert scheduler.is_blocked(plan) is True
