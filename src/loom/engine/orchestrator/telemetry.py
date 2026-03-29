"""Telemetry rollup and event-count helpers for orchestrator."""

from __future__ import annotations

import logging

from loom.engine.runner import SubtaskResult
from loom.engine.verification.types import severity_class_for_reason_code
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
    VERIFICATION_FALSE_NEGATIVE_CANDIDATE,
    VERIFICATION_OUTCOME,
    VERIFICATION_PASSED,
    VERIFICATION_SHADOW_DIFF,
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
        "sealed_policy_preflight_blocked": 0,
        "sealed_reseal_applied": 0,
        "sealed_unexpected_mutation_detected": 0,
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
        "sealed_policy_preflight_blocked",
        "sealed_reseal_applied",
        "sealed_unexpected_mutation_detected",
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


def development_verification_summary_counts(
    *,
    event_bus,
    task_id: str,
    verification_outcome_event_type: str,
) -> dict[str, int]:
    """Aggregate development verification summary fields from outcome events."""
    history_limit = max(1000, int(getattr(event_bus, "_max_history", 1000) or 1000))
    counts = {
        "optional_warning_outcomes": 0,
        "report_mismatch_warning_outcomes": 0,
        "product_failure_count": 0,
        "infra_failure_count": 0,
        "inconclusive_failure_count": 0,
    }
    for event in event_bus.recent_events(limit=history_limit):
        if str(getattr(event, "task_id", "") or "") != task_id:
            continue
        if str(getattr(event, "event_type", "") or "").strip() != verification_outcome_event_type:
            continue
        payload = getattr(event, "data", None)
        if not isinstance(payload, dict):
            continue
        summary = payload.get("dev_verification_summary", {})
        if not isinstance(summary, dict):
            continue
        if bool(summary.get("has_optional_verifier_warnings", False)):
            counts["optional_warning_outcomes"] = int(
                counts["optional_warning_outcomes"],
            ) + 1
        if bool(summary.get("has_report_mismatch_warning", False)):
            counts["report_mismatch_warning_outcomes"] = int(
                counts["report_mismatch_warning_outcomes"],
            ) + 1
        for key in (
            "product_failure_count",
            "infra_failure_count",
            "inconclusive_failure_count",
        ):
            try:
                increment = int(summary.get(key, 0) or 0)
            except (TypeError, ValueError):
                increment = 0
            if increment <= 0:
                continue
            counts[key] = int(counts[key]) + increment
    return counts


def verification_severity_counts(
    reason_counts: dict[str, int] | None,
) -> dict[str, int]:
    """Aggregate verification reasons by severity class."""
    counts = {
        "hard_invariant": 0,
        "semantic": 0,
        "inconclusive": 0,
        "infra": 0,
        "unknown": 0,
    }
    if not isinstance(reason_counts, dict):
        return counts
    for reason, value in reason_counts.items():
        try:
            increment = int(value)
        except (TypeError, ValueError):
            increment = 0
        if increment <= 0:
            continue
        severity = severity_class_for_reason_code(reason) or "unknown"
        counts[severity] = int(counts.get(severity, 0)) + increment
    return counts


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
    development_summary_counts = development_verification_summary_counts(
        event_bus=self._events,
        task_id=task.id,
        verification_outcome_event_type=VERIFICATION_OUTCOME,
    )
    verification_severity_counts_map = verification_severity_counts(
        verification_reason_counts,
    )
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
    total_verification_outcomes = max(
        1,
        int(verification_lifecycle_counts.get("outcome", 0)),
    )
    verifier_terminal_failures = int(
        verification_lifecycle_counts.get("failed", 0),
    )
    inconclusive_outcomes = int(
        verification_reason_counts.get("parse_inconclusive", 0)
        + verification_reason_counts.get("claim_inconclusive", 0)
        + verification_reason_counts.get("infra_verifier_error", 0),
    )
    remediation_resolved = int(remediation_lifecycle_counts.get("resolved", 0))
    remediation_attempted = int(remediation_lifecycle_counts.get("attempt", 0))
    remediation_total_terminal = max(
        1,
        remediation_resolved
        + int(remediation_lifecycle_counts.get("failed", 0))
        + int(remediation_lifecycle_counts.get("expired", 0)),
    )
    shadow_events = int(event_counts.get(VERIFICATION_SHADOW_DIFF, 0))
    reliability_metrics = {
        "verifier_terminal_failure_rate": round(
            float(verifier_terminal_failures) / float(total_verification_outcomes),
            4,
        ),
        "inconclusive_outcome_rate": round(
            float(inconclusive_outcomes) / float(total_verification_outcomes),
            4,
        ),
        "inconclusive_rescue_rate": round(
            float(remediation_resolved) / float(max(1, remediation_attempted)),
            4,
        ),
        "false_block_audit_rate": round(
            float(event_counts.get(VERIFICATION_FALSE_NEGATIVE_CANDIDATE, 0))
            / float(max(1, shadow_events)),
            4,
        ),
        "remediation_terminal_resolution_rate": round(
            float(remediation_resolved) / float(remediation_total_terminal),
            4,
        ),
    }
    development_verification_health = {
        "product_failure_reasons": int(verification_reason_counts.get("dev_test_failed", 0))
        + int(verification_reason_counts.get("dev_build_failed", 0))
        + int(verification_reason_counts.get("dev_contract_failed", 0))
        + int(verification_reason_counts.get("dev_browser_check_failed", 0)),
        "verifier_infra_reasons": int(
            verification_reason_counts.get("dev_verifier_timeout", 0),
        )
        + int(verification_reason_counts.get("dev_verifier_capability_unavailable", 0))
        + int(verification_reason_counts.get("dev_report_contract_violation", 0)),
        "report_contract_violation_reasons": int(
            verification_reason_counts.get("dev_report_contract_violation", 0),
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
        "sealed_policy_preflight_blocked": int(
            rollup.get("sealed_policy_preflight_blocked", 0),
        ),
        "sealed_reseal_applied": int(rollup.get("sealed_reseal_applied", 0)),
        "sealed_unexpected_mutation_detected": int(
            rollup.get("sealed_unexpected_mutation_detected", 0),
        ),
        "verification_lifecycle_counts": verification_lifecycle_counts,
        "verification_reason_counts": verification_reason_counts,
        "verification_severity_counts": verification_severity_counts_map,
        "development_verification_summary_counts": development_summary_counts,
        "development_verification_health": development_verification_health,
        "remediation_lifecycle_counts": remediation_lifecycle_counts,
        "human_loop_counts": human_loop_counts,
        "control_plane_counts": control_plane_counts,
        "output_conflict_counts": output_conflict_counts,
        "blocked_indicator": bool(event_counts.get(SUBTASK_BLOCKED, 0) > 0),
        "degraded_indicator": bool(event_counts.get(TASK_PLAN_DEGRADED, 0) > 0),
        "replanned_count": int(event_counts.get(TASK_REPLANNING, 0)),
        "stalled_count": int(event_counts.get(TASK_STALLED, 0)),
        "reliability_metrics": reliability_metrics,
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
