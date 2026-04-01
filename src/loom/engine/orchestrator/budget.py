"""Run-budget tracking helper for orchestrator task execution."""

from __future__ import annotations

import time
from datetime import datetime

from loom.config import Config
from loom.engine.runner import SubtaskResult
from loom.events.types import TASK_BUDGET_EXHAUSTED
from loom.state.task_state import SubtaskStatus, Task, TaskStatus


class _RunBudget:
    """Tracks task-level resource usage and limit enforcement."""

    def __init__(self, config: Config):
        execution = getattr(config, "execution", None)
        self.enabled = bool(getattr(execution, "enable_global_run_budget", False))
        self.wall_clock_seconds_limit = int(
            max(0, getattr(execution, "max_task_wall_clock_seconds", 0) or 0),
        )
        self.total_tokens_limit = int(
            max(0, getattr(execution, "max_task_total_tokens", 0) or 0),
        )
        self.model_invocations_limit = int(
            max(0, getattr(execution, "max_task_model_invocations", 0) or 0),
        )
        self.tool_calls_limit = int(
            max(0, getattr(execution, "max_task_tool_calls", 0) or 0),
        )
        self.mutating_tool_calls_limit = int(
            max(0, getattr(execution, "max_task_mutating_tool_calls", 0) or 0),
        )
        self.replans_limit = int(
            max(0, getattr(execution, "max_task_replans", 0) or 0),
        )
        self.remediation_attempts_limit = int(
            max(0, getattr(execution, "max_task_remediation_attempts", 0) or 0),
        )

        self.started_monotonic = time.monotonic()
        self.total_tokens = 0
        self.model_invocations = 0
        self.tool_calls = 0
        self.mutating_tool_calls = 0
        self.replans = 0
        self.remediation_attempts = 0

    def observe_result(self, result: SubtaskResult) -> None:
        self.total_tokens += max(0, int(getattr(result, "tokens_used", 0) or 0))
        counters = getattr(result, "telemetry_counters", None)
        if isinstance(counters, dict):
            self.model_invocations += max(0, int(counters.get("model_invocations", 0) or 0))
            self.tool_calls += max(0, int(counters.get("tool_calls", 0) or 0))
            self.mutating_tool_calls += max(
                0,
                int(counters.get("mutating_tool_calls", 0) or 0),
            )

    def observe_replan(self) -> None:
        self.replans += 1

    def observe_remediation_attempt(self) -> None:
        self.remediation_attempts += 1

    def snapshot(self) -> dict[str, int | float]:
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        return {
            "elapsed_seconds": round(elapsed, 3),
            "total_tokens": int(self.total_tokens),
            "model_invocations": int(self.model_invocations),
            "tool_calls": int(self.tool_calls),
            "mutating_tool_calls": int(self.mutating_tool_calls),
            "replans": int(self.replans),
            "remediation_attempts": int(self.remediation_attempts),
        }

    def exceeded(self) -> tuple[bool, str, int | float, int]:
        if not self.enabled:
            return False, "", 0, 0
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        checks: tuple[tuple[str, float, int], ...] = (
            ("max_task_wall_clock_seconds", elapsed, self.wall_clock_seconds_limit),
            ("max_task_total_tokens", float(self.total_tokens), self.total_tokens_limit),
            (
                "max_task_model_invocations",
                float(self.model_invocations),
                self.model_invocations_limit,
            ),
            ("max_task_tool_calls", float(self.tool_calls), self.tool_calls_limit),
            (
                "max_task_mutating_tool_calls",
                float(self.mutating_tool_calls),
                self.mutating_tool_calls_limit,
            ),
            ("max_task_replans", float(self.replans), self.replans_limit),
            (
                "max_task_remediation_attempts",
                float(self.remediation_attempts),
                self.remediation_attempts_limit,
            ),
        )
        for key, current, limit in checks:
            if limit > 0 and current > float(limit):
                return True, key, current, limit
        return False, "", 0, 0


# Extracted budget enforcement orchestration helpers

def _apply_budget_metadata(
    self,
    task: Task,
    budget_name: str,
    observed: int | float,
    limit: int,
) -> None:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["budget_exhausted"] = {
        "name": budget_name,
        "observed": observed,
        "limit": limit,
        "snapshot": self._run_budget.snapshot(),
        "at": datetime.now().isoformat(),
        "run_id": self._task_run_id(task),
    }
    task.metadata = metadata

async def _enforce_global_budget(self, task: Task) -> bool:
    exceeded, budget_name, observed, limit = self._run_budget.exceeded()
    if not exceeded:
        return False
    self._apply_budget_metadata(task, budget_name, observed, limit)
    for candidate in task.plan.subtasks:
        if candidate.status == SubtaskStatus.PENDING:
            candidate.status = SubtaskStatus.SKIPPED
            candidate.summary = (
                "Skipped: global run budget exhausted "
                f"({budget_name})."
            )
    task.status = TaskStatus.FAILED
    task.add_error(
        "budget",
        f"Global run budget exhausted: {budget_name} "
        f"(observed={observed}, limit={limit})",
    )
    await self._save_task_state(task)
    self._emit(TASK_BUDGET_EXHAUSTED, task.id, {
        "run_id": self._task_run_id(task),
        "budget_name": budget_name,
        "observed": observed,
        "limit": limit,
        "snapshot": self._run_budget.snapshot(),
    })
    return True
