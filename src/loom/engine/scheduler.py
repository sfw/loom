"""Subtask scheduler: dependency resolution and ordering."""

from __future__ import annotations

from loom.state.task_state import Plan, Subtask, SubtaskStatus


class Scheduler:
    """Determines which subtask to execute next based on the dependency graph."""

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
        return len(self.runnable_subtasks(plan)) == 0

    @staticmethod
    def _is_terminal_synthesis(plan: Plan, subtask: Subtask) -> bool:
        """Return True when a synthesis subtask has no dependents.

        Non-terminal synthesis subtasks can create a deadlock when paired with the
        global synthesis gate. We only apply that gate to terminal sinks.
        """
        if not subtask.is_synthesis:
            return False
        for candidate in plan.subtasks:
            if candidate.id == subtask.id:
                continue
            if subtask.id in candidate.depends_on:
                return False
        return True

    @staticmethod
    def _deps_met(plan: Plan, subtask: Subtask) -> bool:
        """Check if all dependencies are completed."""
        if not subtask.depends_on:
            deps_satisfied = True
        else:
            deps_satisfied = True
            for dep_id in subtask.depends_on:
                dep = None
                for s in plan.subtasks:
                    if s.id == dep_id:
                        dep = s
                        break
                if dep is None or dep.status != SubtaskStatus.COMPLETED:
                    deps_satisfied = False
                    break

        if not deps_satisfied:
            return False

        # Synthesis subtasks are "final integration" stages and should only
        # run after all non-synthesis work is complete.
        # Apply this gate only for terminal synthesis sinks.
        if Scheduler._is_terminal_synthesis(plan, subtask):
            for candidate in plan.subtasks:
                if candidate.id == subtask.id or candidate.is_synthesis:
                    continue
                if candidate.status != SubtaskStatus.COMPLETED:
                    return False

        return True
