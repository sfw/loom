"""Subtask scheduler: dependency resolution and ordering."""

from __future__ import annotations

from loom.state.task_state import Plan, Subtask, SubtaskStatus


class Scheduler:
    """Determines which subtask to execute next based on the dependency graph."""

    def next_subtask(self, plan: Plan) -> Subtask | None:
        """Return the next runnable subtask, or None if all are blocked/completed.

        A subtask is runnable if:
        1. Status is 'pending'
        2. All depends_on subtasks are 'completed'
        """
        for subtask in plan.subtasks:
            if subtask.status != SubtaskStatus.PENDING:
                continue
            if self._deps_met(plan, subtask):
                return subtask
        return None

    def runnable_subtasks(self, plan: Plan) -> list[Subtask]:
        """Return ALL runnable subtasks (for potential parallel execution)."""
        return [
            s for s in plan.subtasks
            if s.status == SubtaskStatus.PENDING and self._deps_met(plan, s)
        ]

    def has_pending(self, plan: Plan) -> bool:
        """Check if there are any pending or running subtasks."""
        return any(
            s.status in (SubtaskStatus.PENDING, SubtaskStatus.RUNNING)
            for s in plan.subtasks
        )

    def is_blocked(self, plan: Plan) -> bool:
        """Check if all remaining subtasks are blocked (no progress possible)."""
        if not self.has_pending(plan):
            return False
        return self.next_subtask(plan) is None

    @staticmethod
    def _deps_met(plan: Plan, subtask: Subtask) -> bool:
        """Check if all dependencies are completed."""
        if not subtask.depends_on:
            return True
        for dep_id in subtask.depends_on:
            dep = None
            for s in plan.subtasks:
                if s.id == dep_id:
                    dep = s
                    break
            if dep is None or dep.status != SubtaskStatus.COMPLETED:
                return False
        return True
