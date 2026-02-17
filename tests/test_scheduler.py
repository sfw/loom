"""Tests for the subtask scheduler."""

from __future__ import annotations

from loom.engine.scheduler import Scheduler
from loom.state.task_state import Plan, Subtask, SubtaskStatus


def _plan(*subtasks: Subtask) -> Plan:
    """Create a plan from subtasks."""
    return Plan(subtasks=list(subtasks), version=1)


def _subtask(
    id: str,
    depends_on: list[str] | None = None,
    status: str = "pending",
    is_synthesis: bool = False,
) -> Subtask:
    """Create a subtask with defaults."""
    return Subtask(
        id=id,
        description=f"Do {id}",
        status=SubtaskStatus(status),
        depends_on=depends_on or [],
        is_synthesis=is_synthesis,
    )


class TestSchedulerNextRunnable:
    def test_simple_linear(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a"),
            _subtask("b", depends_on=["a"]),
            _subtask("c", depends_on=["b"]),
        )

        # First runnable should be 'a' (no deps)
        runnable = scheduler.runnable_subtasks(plan)
        assert len(runnable) == 1
        assert runnable[0].id == "a"

    def test_returns_empty_when_all_completed(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="completed"),
            _subtask("b", status="completed"),
        )
        assert scheduler.runnable_subtasks(plan) == []

    def test_returns_empty_when_blocked(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="running"),
            _subtask("b", depends_on=["a"]),
        )
        assert scheduler.runnable_subtasks(plan) == []

    def test_skips_running(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="running"),
            _subtask("b"),
        )
        # 'a' is running, 'b' has no deps so it's runnable
        runnable = scheduler.runnable_subtasks(plan)
        assert len(runnable) == 1
        assert runnable[0].id == "b"

    def test_dep_completed_unlocks(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="completed"),
            _subtask("b", depends_on=["a"]),
        )
        runnable = scheduler.runnable_subtasks(plan)
        assert len(runnable) == 1
        assert runnable[0].id == "b"

    def test_multiple_deps_must_all_complete(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("a", status="completed"),
            _subtask("b", status="pending"),
            _subtask("c", depends_on=["a", "b"]),
        )
        # 'c' depends on both a and b; b is not done
        # But 'b' itself is runnable
        runnable = scheduler.runnable_subtasks(plan)
        assert len(runnable) == 1
        assert runnable[0].id == "b"

    def test_empty_plan(self):
        scheduler = Scheduler()
        plan = _plan()
        assert scheduler.runnable_subtasks(plan) == []


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

    def test_synthesis_waits_for_all_non_synthesis_subtasks(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("prep-a", status="completed"),
            _subtask("prep-b", status="pending"),
            _subtask("synthesize", depends_on=["prep-a"], is_synthesis=True),
        )
        runnable = scheduler.runnable_subtasks(plan)
        assert [s.id for s in runnable] == ["prep-b"]

    def test_synthesis_runs_when_all_non_synthesis_complete(self):
        scheduler = Scheduler()
        plan = _plan(
            _subtask("prep-a", status="completed"),
            _subtask("prep-b", status="completed"),
            _subtask("synthesize", is_synthesis=True),
        )
        runnable = scheduler.runnable_subtasks(plan)
        assert [s.id for s in runnable] == ["synthesize"]


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
