"""Telemetry rollup and event-count helpers for orchestrator."""

from __future__ import annotations

import logging

from loom.engine.runner import SubtaskResult
from loom.events.types import (
    APPROVAL_RECEIVED,
    APPROVAL_REQUESTED,
    ASK_USER_ANSWERED,
    ASK_USER_CANCELLED,
    ASK_USER_REQUESTED,
    ASK_USER_TIMEOUT,
    FORBIDDEN_CANONICAL_WRITE_BLOCKED,
    REMEDIATION_ATTEMPT,
    REMEDIATION_EXPIRED,
    REMEDIATION_FAILED,
    REMEDIATION_QUEUED,
    REMEDIATION_RESOLVED,
    REMEDIATION_STARTED,
    REMEDIATION_TERMINAL,
    STEER_INSTRUCTION,
    SUBTASK_BLOCKED,
    SUBTASK_OUTPUT_CONFLICT_DEFERRED,
    SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING,
    TASK_CANCEL_ACK,
    TASK_CANCEL_REQUESTED,
    TASK_CANCEL_TIMEOUT,
    TASK_INJECTED,
    TASK_PAUSED,
    TASK_PLAN_DEGRADED,
    TASK_REPLANNING,
    TASK_RESUMED,
    TASK_STALLED,
    TELEMETRY_RUN_SUMMARY,
    VERIFICATION_FAILED,
    VERIFICATION_OUTCOME,
    VERIFICATION_PASSED,
    VERIFICATION_STARTED,
)
from loom.state.task_state import Task

logger = logging.getLogger(__name__)


def new_telemetry_rollup() -> dict[str, int]:
    """Return empty per-run telemetry counter aggregates."""
    return {
        "model_invocations": 0,
        "tool_calls": 0,
        "mutating_tool_calls": 0,
        "artifact_ingests": 0,
        "artifact_reads": 0,
        "artifact_retention_deletes": 0,
        "compaction_policy_decisions": 0,
        "overflow_fallback_count": 0,
        "compactor_warning_count": 0,
    }


def accumulate_subtask_telemetry(orchestrator, result: SubtaskResult) -> None:
    """Accumulate runner subtask telemetry into orchestrator run-level counters."""
    counters = getattr(result, "telemetry_counters", None)
    if not isinstance(counters, dict):
        return
    rollup = getattr(orchestrator, "_telemetry_rollup", None)
    if not isinstance(rollup, dict):
        orchestrator._telemetry_rollup = new_telemetry_rollup()
        rollup = orchestrator._telemetry_rollup
    for key in (
        "model_invocations",
        "tool_calls",
        "mutating_tool_calls",
        "artifact_ingests",
        "artifact_reads",
        "artifact_retention_deletes",
        "compaction_policy_decisions",
        "overflow_fallback_count",
        "compactor_warning_count",
    ):
        try:
            increment = int(counters.get(key, 0))
        except (TypeError, ValueError):
            increment = 0
        if increment <= 0:
            continue
        rollup[key] = int(rollup.get(key, 0)) + increment


def task_event_counts(event_bus, task_id: str) -> dict[str, int]:
    """Count recent event types emitted for a task."""
    history_limit = max(1000, int(getattr(event_bus, "_max_history", 1000) or 1000))
    counters: dict[str, int] = {}
    for event in event_bus.recent_events(limit=history_limit):
        if str(getattr(event, "task_id", "") or "") != task_id:
            continue
        event_type = str(getattr(event, "event_type", "") or "").strip()
        if not event_type:
            continue
        counters[event_type] = int(counters.get(event_type, 0)) + 1
    return counters


def verification_reason_counts(
    *,
    event_bus,
    task_id: str,
    verification_outcome_event_type: str,
) -> dict[str, int]:
    """Count verification reason codes from outcome events."""
    history_limit = max(1000, int(getattr(event_bus, "_max_history", 1000) or 1000))
    reasons: dict[str, int] = {}
    for event in event_bus.recent_events(limit=history_limit):
        if str(getattr(event, "task_id", "") or "") != task_id:
            continue
        if str(getattr(event, "event_type", "") or "").strip() != verification_outcome_event_type:
            continue
        payload = getattr(event, "data", None)
        if not isinstance(payload, dict):
            continue
        reason = str(payload.get("reason_code", "") or "").strip().lower()
        if not reason:
            reason = "unspecified"
        reasons[reason] = int(reasons.get(reason, 0)) + 1
    return reasons


# Extracted telemetry summary emitter

def _emit_telemetry_run_summary(self, task: Task) -> None:
    run_key = self._task_run_id(task) or task.id
    if run_key in self._emitted_telemetry_summary_runs:
        return
    rollup = getattr(self, "_telemetry_rollup", None)
    if not isinstance(rollup, dict):
        rollup = self._new_telemetry_rollup()
    validity_summary = {}
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if isinstance(metadata, dict):
        scorecard = metadata.get("validity_scorecard", {})
        if isinstance(scorecard, dict):
            run_summary = scorecard.get("run", {})
            if isinstance(run_summary, dict):
                validity_summary = run_summary
    event_counts = self._task_event_counts(task.id)
    verification_reason_counts = self._verification_reason_counts(task.id)
    verification_lifecycle_counts = {
        "started": int(event_counts.get(VERIFICATION_STARTED, 0)),
        "passed": int(event_counts.get(VERIFICATION_PASSED, 0)),
        "failed": int(event_counts.get(VERIFICATION_FAILED, 0)),
        "outcome": int(event_counts.get(VERIFICATION_OUTCOME, 0)),
    }
    remediation_lifecycle_counts = {
        "queued": int(event_counts.get(REMEDIATION_QUEUED, 0)),
        "started": int(event_counts.get(REMEDIATION_STARTED, 0)),
        "attempt": int(event_counts.get(REMEDIATION_ATTEMPT, 0)),
        "resolved": int(event_counts.get(REMEDIATION_RESOLVED, 0)),
        "failed": int(event_counts.get(REMEDIATION_FAILED, 0)),
        "expired": int(event_counts.get(REMEDIATION_EXPIRED, 0)),
        "terminal": int(event_counts.get(REMEDIATION_TERMINAL, 0)),
    }
    human_loop_counts = {
        "approval_requested": int(event_counts.get(APPROVAL_REQUESTED, 0)),
        "approval_received": int(event_counts.get(APPROVAL_RECEIVED, 0)),
        "ask_user_requested": int(event_counts.get(ASK_USER_REQUESTED, 0)),
        "ask_user_answered": int(event_counts.get(ASK_USER_ANSWERED, 0)),
        "ask_user_timeout": int(event_counts.get(ASK_USER_TIMEOUT, 0)),
        "ask_user_cancelled": int(event_counts.get(ASK_USER_CANCELLED, 0)),
        "steer_instruction": int(event_counts.get(STEER_INSTRUCTION, 0)),
    }
    control_plane_counts = {
        "paused": int(event_counts.get(TASK_PAUSED, 0)),
        "resumed": int(event_counts.get(TASK_RESUMED, 0)),
        "injected": int(event_counts.get(TASK_INJECTED, 0)),
        "cancel_requested": int(event_counts.get(TASK_CANCEL_REQUESTED, 0)),
        "cancel_ack": int(event_counts.get(TASK_CANCEL_ACK, 0)),
        "cancel_timeout": int(event_counts.get(TASK_CANCEL_TIMEOUT, 0)),
    }
    output_conflict_counts = {
        "deferred": int(event_counts.get(SUBTASK_OUTPUT_CONFLICT_DEFERRED, 0)),
        "starvation_warning": int(
            event_counts.get(SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING, 0),
        ),
        "forbidden_canonical_write_blocked": int(
            event_counts.get(FORBIDDEN_CANONICAL_WRITE_BLOCKED, 0),
        ),
    }
    self._emit(TELEMETRY_RUN_SUMMARY, task.id, {
        "run_id": self._task_run_id(task),
        "model_invocations": int(rollup.get("model_invocations", 0)),
        "tool_calls": int(rollup.get("tool_calls", 0)),
        "mutating_tool_calls": int(rollup.get("mutating_tool_calls", 0)),
        "artifact_ingests": int(rollup.get("artifact_ingests", 0)),
        "artifact_reads": int(rollup.get("artifact_reads", 0)),
        "artifact_retention_deletes": int(rollup.get("artifact_retention_deletes", 0)),
        "compaction_policy_decisions": int(rollup.get("compaction_policy_decisions", 0)),
        "overflow_fallback_count": int(rollup.get("overflow_fallback_count", 0)),
        "compactor_warning_count": int(rollup.get("compactor_warning_count", 0)),
        "verification_lifecycle_counts": verification_lifecycle_counts,
        "verification_reason_counts": verification_reason_counts,
        "remediation_lifecycle_counts": remediation_lifecycle_counts,
        "human_loop_counts": human_loop_counts,
        "control_plane_counts": control_plane_counts,
        "output_conflict_counts": output_conflict_counts,
        "blocked_indicator": bool(event_counts.get(SUBTASK_BLOCKED, 0) > 0),
        "degraded_indicator": bool(event_counts.get(TASK_PLAN_DEGRADED, 0) > 0),
        "replanned_count": int(event_counts.get(TASK_REPLANNING, 0)),
        "stalled_count": int(event_counts.get(TASK_STALLED, 0)),
        "budget_snapshot": self._run_budget.snapshot(),
        "validity_summary": validity_summary,
    })
    self._emitted_telemetry_summary_runs.add(run_key)


# Extracted run-id + learning helpers

async def _learn_from_task(self, task: Task) -> None:
    """Run post-task learning extraction (best-effort)."""
    if self._learning is None:
        return
    try:
        await self._learning.learn_from_task(task)
    except Exception as e:
        logger.warning("Post-task learning failed for %s: %s", task.id, e)
