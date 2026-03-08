"""Remediation queue helpers for orchestrator extraction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from loom.engine.runner import SubtaskResult, SubtaskResultStatus, ToolCallRecord
from loom.engine.verification import VerificationResult
from loom.events.types import (
    MODEL_INVOCATION,
    PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED,
    PLACEHOLDER_FILLED,
    PLACEHOLDER_PRUNED,
    PLACEHOLDER_REMEDIATION_UNRESOLVED,
    REMEDIATION_ATTEMPT,
    REMEDIATION_EXPIRED,
    REMEDIATION_FAILED,
    REMEDIATION_QUEUED,
    REMEDIATION_RESOLVED,
    REMEDIATION_STARTED,
    REMEDIATION_TERMINAL,
    SEALED_RESEAL_APPLIED,
    SUBTASK_RETRYING,
    UNCONFIRMED_DATA_QUEUED,
)
from loom.models.base import ModelResponse
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.recovery.errors import ErrorCategory, categorize_error
from loom.recovery.retry import AttemptRecord, RetryStrategy
from loom.state.evidence import merge_evidence_records
from loom.state.task_state import Subtask, Task
from loom.utils.concurrency import run_blocking_io

from . import validity as orchestrator_validity

logger = logging.getLogger(__name__)

_REMEDIATION_TERMINAL_STATES = frozenset({"resolved", "failed", "expired"})
_VALID_CRITICAL_PATH_BEHAVIORS = frozenset({
    "block",
    "confirm_or_prune_then_queue",
    "queue_follow_up",
})
_PLACEHOLDER_UNCONFIRMED_REASON_CODES = frozenset({
    "incomplete_deliverable_placeholder",
    "incomplete_deliverable_content",
})
_PLACEHOLDER_PREPASS_MODE = "deterministic_placeholder_prepass"
_FAILURE_RESOLUTION_METADATA_KEYS = (
    "remediation_required",
    "remediation_mode",
    "failure_class",
    "missing_targets",
    "placeholder_finding_count",
    "placeholder_findings",
    "unverified_claim_count",
    "verified_claim_count",
    "supporting_ratio",
    "coverage_sufficient",
    "coverage_insufficient_reason",
    "contradiction_detected",
    "contradiction_downgraded",
    "contradicted_reason_code",
    "parser_stage",
    "issues",
)


def remediation_queue_limits(orchestrator) -> tuple[int, float, float]:
    """Resolve remediation retry/backoff limits from config + process contract."""
    max_attempts = int(
        getattr(orchestrator._config.verification, "remediation_queue_max_attempts", 3) or 3,
    )
    process = orchestrator._process
    if process is not None:
        retry_budget = getattr(
            getattr(process, "verification_remediation", None),
            "retry_budget",
            {},
        )
        if isinstance(retry_budget, dict):
            raw_max_attempts = orchestrator._to_int_or_none(retry_budget.get("max_attempts"))
            if raw_max_attempts is not None and raw_max_attempts > 0:
                max_attempts = raw_max_attempts
    if max_attempts <= 0:
        max_attempts = int(
            getattr(orchestrator._config.verification, "confirm_or_prune_max_attempts", 2) or 2,
        )
    max_attempts = max(1, max_attempts)

    base_backoff = float(
        getattr(orchestrator._config.verification, "remediation_queue_backoff_seconds", 2.0) or 0.0,
    )
    if base_backoff < 0:
        base_backoff = 0.0

    max_backoff = float(
        getattr(
            orchestrator._config.verification,
            "remediation_queue_max_backoff_seconds",
            30.0,
        ) or 0.0,
    )
    if max_backoff <= 0:
        max_backoff = max(base_backoff, 0.0)
    if max_backoff < base_backoff:
        max_backoff = base_backoff
    return max_attempts, base_backoff, max_backoff


def bounded_remediation_backoff_seconds(
    *,
    base_backoff_seconds: float,
    max_backoff_seconds: float,
    attempt_count: int,
) -> float:
    """Compute exponential remediation backoff with an upper bound."""
    base = max(0.0, float(base_backoff_seconds or 0.0))
    if base <= 0:
        return 0.0
    ceiling = max(base, float(max_backoff_seconds or 0.0))
    exponent = max(0, int(attempt_count) - 1)
    computed = base * (2**exponent)
    return min(ceiling, computed)


def parse_iso_datetime(raw: object) -> datetime | None:
    """Parse an ISO datetime string."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def remediation_item_due(item: dict, now: datetime) -> bool:
    """Check if remediation item is due for execution."""
    due_at = parse_iso_datetime(item.get("next_attempt_at"))
    if due_at is None:
        return True
    return now >= due_at


def remediation_item_expired(item: dict, now: datetime) -> bool:
    """Check if remediation item has expired by TTL."""
    ttl_at = parse_iso_datetime(item.get("ttl_at"))
    if ttl_at is None:
        return False
    return now >= ttl_at

async def queue_remediation_work_item(
    orchestrator,
    *,
    task: Task,
    subtask: Subtask,
    verification: VerificationResult,
    strategy: RetryStrategy,
    blocking: bool,
) -> None:
    queue = orchestrator._remediation_queue(task)
    reason_code = str(verification.reason_code or "").strip().lower()
    strategy_value = strategy.value
    now = datetime.now()
    max_attempts, base_backoff_seconds, max_backoff_seconds = (
        orchestrator._remediation_queue_limits()
    )
    uncertainty = orchestrator._extract_unconfirmed_metadata(verification)
    missing_targets = orchestrator._normalize_missing_targets(
        uncertainty.get("missing_targets"),
    )

    for item in queue:
        if not isinstance(item, dict):
            continue
        if str(item.get("subtask_id", "")).strip() != subtask.id:
            continue
        if str(item.get("strategy", "")).strip() != strategy_value:
            continue
        if str(item.get("reason_code", "")).strip().lower() != reason_code:
            continue
        state = str(item.get("state", "queued")).strip().lower()
        if state in _REMEDIATION_TERMINAL_STATES:
            continue

        if blocking and not bool(item.get("blocking", False)):
            item["blocking"] = True
        existing_targets = orchestrator._normalize_missing_targets(
            item.get("missing_targets"),
        )
        merged_targets = existing_targets + [
            target for target in missing_targets
            if target not in existing_targets
        ]
        if merged_targets:
            item["missing_targets"] = merged_targets
        for key, value in uncertainty.items():
            if key == "missing_targets":
                continue
            item[key] = value
        item["feedback"] = verification.feedback or str(item.get("feedback", ""))
        item["verification_outcome"] = verification.outcome
        existing_max_attempts = orchestrator._to_int_or_none(item.get("max_attempts"))
        if existing_max_attempts is None:
            existing_max_attempts = max_attempts
        existing_base_backoff = orchestrator._to_float_or_none(
            item.get("base_backoff_seconds"),
        )
        if existing_base_backoff is None:
            try:
                existing_base_backoff = float(
                    item.get("base_backoff_seconds", base_backoff_seconds) or 0.0,
                )
            except (TypeError, ValueError):
                existing_base_backoff = base_backoff_seconds
        existing_max_backoff = orchestrator._to_float_or_none(
            item.get("max_backoff_seconds"),
        )
        if existing_max_backoff is None:
            try:
                existing_max_backoff = float(
                    item.get("max_backoff_seconds", max_backoff_seconds) or 0.0,
                )
            except (TypeError, ValueError):
                existing_max_backoff = max_backoff_seconds
        item["max_attempts"] = max(
            existing_max_attempts,
            max_attempts,
        )
        item["base_backoff_seconds"] = max(
            max(0.0, float(existing_base_backoff or 0.0)),
            base_backoff_seconds,
        )
        item["max_backoff_seconds"] = max(
            max(0.0, float(existing_max_backoff or 0.0)),
            max_backoff_seconds,
        )
        item["updated_at"] = now.isoformat()
        async with orchestrator._state_lock:
            orchestrator._state.save(task)
        orchestrator._emit(UNCONFIRMED_DATA_QUEUED, task.id, {
            "remediation_id": str(item.get("id", "")).strip(),
            "subtask_id": subtask.id,
            "strategy": strategy_value,
            "reason_code": reason_code,
            "blocking": bool(item.get("blocking", False)),
            "critical_path": bool(subtask.is_critical_path),
            "deduped": True,
        })
        return

    item = {
        "id": f"rem-{uuid.uuid4().hex[:10]}",
        "task_id": task.id,
        "subtask_id": subtask.id,
        "strategy": strategy_value,
        "reason_code": reason_code,
        "verification_outcome": verification.outcome,
        "feedback": verification.feedback or "",
        "blocking": bool(blocking),
        "state": "queued",
        "attempt_count": 0,
        "last_error": "",
        "terminal_reason": "",
        "critical_path": bool(subtask.is_critical_path),
        "max_attempts": max_attempts,
        "base_backoff_seconds": base_backoff_seconds,
        "max_backoff_seconds": max_backoff_seconds,
        "missing_targets": missing_targets,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "next_attempt_at": now.isoformat(),
        "ttl_at": (now + timedelta(hours=6)).isoformat(),
    }
    item.update({
        key: value
        for key, value in uncertainty.items()
        if key != "missing_targets"
    })
    queue.append(item)
    async with orchestrator._state_lock:
        task.add_decision(
            "Queued remediation for "
            f"{subtask.id} ({strategy.value}, blocking={blocking})."
        )
        orchestrator._state.save(task)
    orchestrator._emit(REMEDIATION_QUEUED, task.id, {
        "remediation_id": item["id"],
        "subtask_id": subtask.id,
        "strategy": strategy_value,
        "reason_code": reason_code,
        "blocking": bool(blocking),
    })
    orchestrator._emit(UNCONFIRMED_DATA_QUEUED, task.id, {
        "remediation_id": item["id"],
        "subtask_id": subtask.id,
        "strategy": strategy_value,
        "reason_code": reason_code,
        "blocking": bool(blocking),
        "critical_path": bool(subtask.is_critical_path),
        "deduped": False,
        "missing_targets": list(missing_targets),
    })

async def process_remediation_queue(
    orchestrator,
    *,
    task: Task,
    attempts_by_subtask: dict[str, list[AttemptRecord]],
    finalizing: bool,
) -> None:
    queue = task.metadata.get("remediation_queue")
    if not isinstance(queue, list) or not queue:
        return

    default_max_attempts, default_base_backoff, default_max_backoff = (
        orchestrator._remediation_queue_limits()
    )
    changed = False
    now = datetime.now()

    for item in queue:
        if not isinstance(item, dict):
            continue
        state = str(item.get("state", "queued")).strip().lower()
        if state in _REMEDIATION_TERMINAL_STATES:
            continue

        remediation_id = str(item.get("id", "")).strip() or f"rem-{uuid.uuid4().hex[:10]}"
        item["id"] = remediation_id
        subtask_id = str(item.get("subtask_id", "")).strip()

        if orchestrator._remediation_item_expired(item, now):
            item["state"] = "expired"
            item["updated_at"] = now.isoformat()
            item["last_error"] = (
                str(item.get("last_error", "")).strip()
                or "remediation ttl exceeded"
            )
            item["terminal_reason"] = "ttl_expired"
            changed = True
            orchestrator._emit(REMEDIATION_EXPIRED, task.id, {
                "remediation_id": remediation_id,
                "subtask_id": subtask_id,
                "strategy": str(item.get("strategy", "")),
            })
            orchestrator._emit(REMEDIATION_TERMINAL, task.id, {
                "remediation_id": remediation_id,
                "subtask_id": subtask_id,
                "strategy": str(item.get("strategy", "")),
                "state": "expired",
                "reason": "ttl_expired",
            })
            continue

        if not orchestrator._remediation_item_due(item, now):
            continue

        item["state"] = "running"
        item["updated_at"] = now.isoformat()
        changed = True
        orchestrator._emit(REMEDIATION_STARTED, task.id, {
            "remediation_id": remediation_id,
            "subtask_id": subtask_id,
            "strategy": str(item.get("strategy", "")),
            "blocking": bool(item.get("blocking", False)),
        })

        resolved, error = await orchestrator._execute_remediation_item(
            task=task,
            item=item,
            attempts_by_subtask=attempts_by_subtask,
        )
        if resolved:
            item["state"] = "resolved"
            item["updated_at"] = datetime.now().isoformat()
            item["last_error"] = ""
            item["terminal_reason"] = "resolved"
            changed = True
            orchestrator._emit(REMEDIATION_RESOLVED, task.id, {
                "remediation_id": remediation_id,
                "subtask_id": subtask_id,
                "strategy": str(item.get("strategy", "")),
            })
            orchestrator._emit(REMEDIATION_TERMINAL, task.id, {
                "remediation_id": remediation_id,
                "subtask_id": subtask_id,
                "strategy": str(item.get("strategy", "")),
                "state": "resolved",
                "reason": "resolved",
            })
            continue

        attempt_count = int(item.get("attempt_count", 0) or 0) + 1
        item["attempt_count"] = attempt_count
        item["last_error"] = str(error or "").strip()
        item["updated_at"] = datetime.now().isoformat()
        parsed_max_attempts = orchestrator._to_int_or_none(item.get("max_attempts"))
        if parsed_max_attempts is None:
            parsed_max_attempts = default_max_attempts
        max_attempts = max(1, parsed_max_attempts)

        parsed_base_backoff = orchestrator._to_float_or_none(
            item.get("base_backoff_seconds"),
        )
        if parsed_base_backoff is None:
            parsed_base_backoff = default_base_backoff
        base_backoff_seconds = max(0.0, parsed_base_backoff)

        parsed_max_backoff = orchestrator._to_float_or_none(
            item.get("max_backoff_seconds"),
        )
        if parsed_max_backoff is None:
            parsed_max_backoff = default_max_backoff
        max_backoff_seconds = max(
            base_backoff_seconds,
            parsed_max_backoff,
        )
        exhausted = attempt_count >= max_attempts
        if exhausted or (finalizing and bool(item.get("blocking", False))):
            item["state"] = "failed"
            item["terminal_reason"] = (
                "max_attempts_exhausted"
                if exhausted else "blocking_unresolved_at_finalization"
            )
            orchestrator._emit(REMEDIATION_FAILED, task.id, {
                "remediation_id": remediation_id,
                "subtask_id": subtask_id,
                "strategy": str(item.get("strategy", "")),
                "attempt_count": attempt_count,
                "error": item["last_error"],
            })
            orchestrator._emit(REMEDIATION_TERMINAL, task.id, {
                "remediation_id": remediation_id,
                "subtask_id": subtask_id,
                "strategy": str(item.get("strategy", "")),
                "state": "failed",
                "reason": str(item.get("terminal_reason", "")),
                "attempt_count": attempt_count,
            })
        else:
            item["state"] = "queued"
            delay_seconds = orchestrator._bounded_remediation_backoff_seconds(
                base_backoff_seconds=base_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                attempt_count=attempt_count,
            )
            item["next_attempt_at"] = (
                datetime.now() + timedelta(seconds=delay_seconds)
            ).isoformat()
        changed = True

    if finalizing:
        for item in queue:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("blocking", False)):
                continue
            state = str(item.get("state", "queued")).strip().lower()
            if state in _REMEDIATION_TERMINAL_STATES:
                continue
            item["state"] = "failed"
            item["updated_at"] = datetime.now().isoformat()
            item["last_error"] = (
                str(item.get("last_error", "")).strip()
                or "blocking remediation unresolved at task finalization"
            )
            item["terminal_reason"] = "blocking_unresolved_at_finalization"
            changed = True
            orchestrator._emit(REMEDIATION_FAILED, task.id, {
                "remediation_id": str(item.get("id", "")).strip(),
                "subtask_id": str(item.get("subtask_id", "")).strip(),
                "strategy": str(item.get("strategy", "")),
                "attempt_count": int(item.get("attempt_count", 0) or 0),
                "error": item["last_error"],
            })
            orchestrator._emit(REMEDIATION_TERMINAL, task.id, {
                "remediation_id": str(item.get("id", "")).strip(),
                "subtask_id": str(item.get("subtask_id", "")).strip(),
                "strategy": str(item.get("strategy", "")),
                "state": "failed",
                "reason": "blocking_unresolved_at_finalization",
                "attempt_count": int(item.get("attempt_count", 0) or 0),
            })

    if changed:
        async with orchestrator._state_lock:
            orchestrator._state.save(task)

async def execute_remediation_item(
    orchestrator,
    *,
    task: Task,
    item: dict,
    attempts_by_subtask: dict[str, list[AttemptRecord]],
) -> tuple[bool, str]:
    strategy_value = str(item.get("strategy", "")).strip()
    subtask_id = str(item.get("subtask_id", "")).strip()
    subtask = task.get_subtask(subtask_id)
    if subtask is None:
        return False, f"subtask not found: {subtask_id}"

    if strategy_value != RetryStrategy.UNCONFIRMED_DATA.value:
        return False, f"unsupported remediation strategy: {strategy_value}"

    attempts = attempts_by_subtask.setdefault(subtask.id, [])
    return await orchestrator._run_confirm_or_prune_remediation(
        task=task,
        subtask=subtask,
        attempts=attempts,
        remediation_id=str(item.get("id", "")).strip() or None,
        placeholder_metadata=item if isinstance(item, dict) else None,
    )

async def run_confirm_or_prune_remediation(
    orchestrator,
    *,
    task: Task,
    subtask: Subtask,
    attempts: list[AttemptRecord],
    remediation_id: str | None = None,
    verification: VerificationResult | None = None,
    placeholder_metadata: dict[str, object] | None = None,
) -> tuple[bool, str]:
    if not orchestrator._config.verification.auto_confirm_prune_critical_path:
        return False, "auto_confirm_prune_critical_path disabled"

    max_attempts = max(
        1,
        int(
            getattr(
                orchestrator._config.verification,
                "confirm_or_prune_max_attempts",
                2,
            ) or 2
        ),
    )
    backoff_seconds = max(
        0.0,
        float(
            getattr(
                orchestrator._config.verification,
                "confirm_or_prune_backoff_seconds",
                2.0,
            ) or 0.0
        ),
    )
    retry_on_transient = bool(
        getattr(
            orchestrator._config.verification,
            "confirm_or_prune_retry_on_transient",
            True,
        ),
    )
    last_failure = "process remediation failed"
    placeholder_unconfirmed = orchestrator._is_placeholder_unconfirmed_failure(
        verification=verification,
        placeholder_metadata=placeholder_metadata,
    )
    deterministic_note = ""
    deterministic_details: dict[str, object] = {}
    placeholder_reason_code = (
        str(getattr(verification, "reason_code", "") or "").strip().lower()
    )
    placeholder_findings: list[dict[str, object]] = []
    if placeholder_unconfirmed:
        placeholder_context: dict[str, object] = {}
        if verification is not None and isinstance(verification.metadata, dict):
            placeholder_context.update(verification.metadata)
        if isinstance(placeholder_metadata, dict):
            placeholder_context.update(placeholder_metadata)
        placeholder_findings = orchestrator._normalize_placeholder_findings(
            placeholder_context.get("placeholder_findings"),
        )
    workspace_path: Path | None = None
    workspace_text = str(getattr(task, "workspace", "") or "").strip()
    if workspace_text:
        candidate_workspace = Path(workspace_text)
        if candidate_workspace.exists() and candidate_workspace.is_dir():
            workspace_path = candidate_workspace

    for attempt_number in range(1, max_attempts + 1):
        orchestrator._run_budget.observe_remediation_attempt()
        orchestrator._emit(SUBTASK_RETRYING, task.id, {
            "subtask_id": subtask.id,
            "mode": "confirm_or_prune",
            "reason": "critical_unconfirmed_data",
            "attempt": attempt_number,
            "max_attempts": max_attempts,
        })
        orchestrator._emit(REMEDIATION_ATTEMPT, task.id, {
            "remediation_id": remediation_id or "",
            "subtask_id": subtask.id,
            "attempt": attempt_number,
            "max_attempts": max_attempts,
            "phase": "start",
        })
        await orchestrator._persist_remediation_attempt(
            task=task,
            remediation_id=remediation_id or "",
            subtask_id=subtask.id,
            attempt=attempt_number,
            max_attempts=max_attempts,
            phase="start",
        )

        if attempt_number == 1 and placeholder_unconfirmed:
            (
                _resolved_deterministically,
                deterministic_note,
                deterministic_details,
            ) = await orchestrator._run_deterministic_placeholder_prepass(
                task=task,
                subtask=subtask,
                verification=verification,
                placeholder_metadata=placeholder_metadata,
                origin="confirm_or_prune_remediation",
                attempt_number=attempt_number,
                max_attempts=max_attempts,
            )
            if deterministic_note:
                last_failure = deterministic_note

        prior_successful_tool_calls: list[ToolCallRecord] = []
        prior_evidence_records = orchestrator._evidence_for_subtask(task.id, subtask.id)
        for attempt in attempts:
            raw_calls = getattr(attempt, "successful_tool_calls", [])
            if isinstance(raw_calls, list):
                for call in raw_calls:
                    if isinstance(call, ToolCallRecord):
                        prior_successful_tool_calls.append(call)
            raw_evidence = getattr(attempt, "evidence_records", [])
            if isinstance(raw_evidence, list):
                prior_evidence_records = merge_evidence_records(
                    prior_evidence_records,
                    [item for item in raw_evidence if isinstance(item, dict)],
                )

        remediation_context = (
            orchestrator._build_remediation_retry_context(
                strategy=RetryStrategy.UNCONFIRMED_DATA,
                reason_code=placeholder_reason_code,
            )
        )
        output_policy = orchestrator._output_write_policy_for_subtask(subtask=subtask)
        expected_deliverables = list(output_policy.get("expected_deliverables", []))
        forbidden_deliverables = list(output_policy.get("forbidden_deliverables", []))
        allowed_output_prefixes = orchestrator._fan_in_worker_output_prefixes(
            task=task,
            subtask=subtask,
        )
        stage_plan = orchestrator._finalizer_stage_publish_plan(
            task=task,
            subtask=subtask,
            canonical_deliverables=expected_deliverables,
            attempt_index=len(attempts) + 1,
        )
        runner_expected_deliverables = list(expected_deliverables)
        runner_forbidden_deliverables = list(forbidden_deliverables)
        if bool(stage_plan.get("enabled", False)):
            runner_expected_deliverables = list(stage_plan.get("stage_deliverables", []))
            runner_forbidden_deliverables = orchestrator._merge_unique_paths(
                runner_forbidden_deliverables,
                expected_deliverables,
            )
        manifest_requirements = orchestrator._evaluate_finalizer_manifest_requirements(
            task=task,
            subtask=subtask,
        )
        remediation_context = orchestrator._augment_retry_context_for_outputs(
            subtask=subtask,
            attempts=attempts,
            strategy=RetryStrategy.UNCONFIRMED_DATA,
            expected_deliverables=expected_deliverables,
            forbidden_deliverables=runner_forbidden_deliverables,
            base_context=remediation_context,
        )
        remediation_context = orchestrator._augment_retry_context_for_stage_publish(
            base_context=remediation_context,
            stage_plan=stage_plan,
        )
        remediation_context = orchestrator._augment_retry_context_with_phase_artifacts(
            task=task,
            subtask=subtask,
            base_context=remediation_context,
        )
        if bool(manifest_requirements.get("enabled", False)):
            policy = str(manifest_requirements.get("policy", "") or "").strip().lower()
            missing_worker_ids = list(manifest_requirements.get("missing_worker_ids", []))
            if missing_worker_ids:
                remediation_context = (
                    f"{remediation_context}\n\n"
                    "FINALIZER INPUT POLICY STATUS:\n"
                    f"- policy: {policy}\n"
                    "- workers missing manifest artifacts: "
                    f"{', '.join(missing_worker_ids)}"
                ).strip()
            if missing_worker_ids and policy == "require_all_workers":
                message = (
                    "Finalizer blocked: missing worker artifacts for phase policy "
                    f"'require_all_workers': {', '.join(missing_worker_ids)}"
                )
                last_failure = message
                orchestrator._emit(REMEDIATION_ATTEMPT, task.id, {
                    "remediation_id": remediation_id or "",
                    "subtask_id": subtask.id,
                    "attempt": attempt_number,
                    "max_attempts": max_attempts,
                    "phase": "done",
                    "outcome": "failed",
                    "error": message,
                })
                await orchestrator._persist_remediation_attempt(
                    task=task,
                    remediation_id=remediation_id or "",
                    subtask_id=subtask.id,
                    attempt=attempt_number,
                    max_attempts=max_attempts,
                    phase="done",
                    outcome="failed",
                    error=message,
                )
                continue
        if deterministic_note:
            remediation_context = (
                f"{remediation_context}\n\n"
                "DETERMINISTIC PLACEHOLDER REMEDIATION RESULT:\n"
                f"- {deterministic_note}"
            ).strip()
        if deterministic_details:
            remediation_context = (
                f"{remediation_context}\n"
                "DETERMINISTIC PLACEHOLDER PREPASS DETAILS:\n"
                f"{json.dumps(deterministic_details, ensure_ascii=True)}"
            ).strip()
        escalated_tier = orchestrator._retry.get_escalation_tier(
            attempt=len(attempts),
            original_tier=subtask.model_tier,
        )
        changelog = orchestrator._get_changelog(task)
        pre_resolution_state: dict[str, int] | None = None
        if placeholder_unconfirmed and placeholder_findings and workspace_path is not None:
            raw_pre_state = await run_blocking_io(
                orchestrator._summarize_placeholder_resolution_state,
                workspace=workspace_path,
                findings=placeholder_findings,
            )
            if isinstance(raw_pre_state, dict):
                pre_resolution_state = {
                    "tracked_findings": int(raw_pre_state.get("tracked_findings", 0) or 0),
                    "remaining_findings": int(raw_pre_state.get("remaining_findings", 0) or 0),
                    "replacement_token_occurrences": int(
                        raw_pre_state.get("replacement_token_occurrences", 0) or 0,
                    ),
                }
        remediation_result, remediation_verification = await orchestrator._runner.run(
            task,
            subtask,
            model_tier=escalated_tier,
            retry_context=remediation_context,
            changelog=changelog,
            prior_successful_tool_calls=prior_successful_tool_calls,
            prior_evidence_records=prior_evidence_records,
            expected_deliverables=runner_expected_deliverables,
            forbidden_deliverables=runner_forbidden_deliverables,
            allowed_output_prefixes=allowed_output_prefixes,
            enforce_deliverable_paths=bool(runner_expected_deliverables),
            edit_existing_only=bool(runner_expected_deliverables),
            retry_strategy=RetryStrategy.UNCONFIRMED_DATA.value,
        )
        if bool(manifest_requirements.get("enabled", False)):
            allowed_manifest_paths = list(
                manifest_requirements.get("allowed_manifest_paths", []),
            )
            allowed_stage_prefixes = list(stage_plan.get("stage_prefixes", []))
            violations = orchestrator._manifest_only_input_violations(
                task=task,
                subtask=subtask,
                tool_calls=remediation_result.tool_calls,
                allowed_manifest_paths=allowed_manifest_paths,
                allowed_extra_prefixes=allowed_stage_prefixes,
            )
            if violations:
                message = (
                    "Finalizer input policy violation: read access to intermediate "
                    "artifacts outside latest worker manifest entries: "
                    + ", ".join(violations)
                )
                remediation_result.status = SubtaskResultStatus.FAILED
                remediation_verification = VerificationResult(
                    tier=max(1, int(subtask.verification_tier or 1)),
                    passed=False,
                    confidence=0.0,
                    feedback=message,
                    outcome="fail",
                    reason_code="manifest_input_policy_violation",
                    severity_class="semantic",
                )
        if (
            bool(stage_plan.get("enabled", False))
            and remediation_result.status != SubtaskResultStatus.FAILED
            and remediation_verification.passed
        ):
            commit_ok, commit_error = orchestrator._commit_finalizer_stage_publish(
                task=task,
                subtask=subtask,
                stage_plan=stage_plan,
            )
            if not commit_ok:
                message = commit_error or "Transactional stage+commit publish failed."
                remediation_result.status = SubtaskResultStatus.FAILED
                remediation_verification = VerificationResult(
                    tier=max(1, int(subtask.verification_tier or 1)),
                    passed=False,
                    confidence=0.0,
                    feedback=message,
                    outcome="fail",
                    reason_code="output_publish_commit_failed",
                    severity_class="semantic",
                )
                remediation_result.summary = (
                    f"{remediation_result.summary}\n{message}".strip()
                    if remediation_result.summary
                    else message
                )
        orchestrator._persist_subtask_evidence(
            task.id,
            subtask.id,
            remediation_result.evidence_records,
            tool_calls=remediation_result.tool_calls,
            workspace=task.workspace,
        )

        if remediation_verification.passed:
            await orchestrator._handle_success(
                task,
                subtask,
                remediation_result,
                remediation_verification,
            )
            if (
                placeholder_unconfirmed
                and placeholder_findings
                and workspace_path is not None
                and pre_resolution_state is not None
            ):
                raw_post_state = await run_blocking_io(
                    orchestrator._summarize_placeholder_resolution_state,
                    workspace=workspace_path,
                    findings=placeholder_findings,
                )
                if isinstance(raw_post_state, dict):
                    pre_remaining = int(
                        pre_resolution_state.get("remaining_findings", 0) or 0,
                    )
                    post_remaining = int(raw_post_state.get("remaining_findings", 0) or 0)
                    pre_replacement_count = int(
                        pre_resolution_state.get("replacement_token_occurrences", 0) or 0,
                    )
                    post_replacement_count = int(
                        raw_post_state.get("replacement_token_occurrences", 0) or 0,
                    )
                    resolved_count = max(0, pre_remaining - post_remaining)
                    replacement_delta = max(0, post_replacement_count - pre_replacement_count)
                    if post_remaining > 0:
                        orchestrator._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
                            "subtask_id": subtask.id,
                            "reason_code": placeholder_reason_code,
                            "mode": "confirm_or_prune_remediation",
                            "attempt": attempt_number,
                            "max_attempts": max_attempts,
                            "stage": "post_model_success",
                            "outcome": "placeholders_remaining",
                            "finding_count": len(placeholder_findings),
                            "remaining_count": post_remaining,
                        })
                    elif resolved_count > 0:
                        if replacement_delta > 0:
                            orchestrator._emit(PLACEHOLDER_PRUNED, task.id, {
                                "subtask_id": subtask.id,
                                "reason_code": placeholder_reason_code,
                                "mode": "confirm_or_prune_remediation",
                                "attempt": attempt_number,
                                "max_attempts": max_attempts,
                                "stage": "model_retry",
                                "finding_count": len(placeholder_findings),
                                "resolved_count": resolved_count,
                                "replacement_delta": replacement_delta,
                            })
                        else:
                            orchestrator._emit(PLACEHOLDER_FILLED, task.id, {
                                "subtask_id": subtask.id,
                                "reason_code": placeholder_reason_code,
                                "mode": "confirm_or_prune_remediation",
                                "attempt": attempt_number,
                                "max_attempts": max_attempts,
                                "stage": "model_retry",
                                "finding_count": len(placeholder_findings),
                                "filled_count": resolved_count,
                            })
            orchestrator._emit(REMEDIATION_ATTEMPT, task.id, {
                "remediation_id": remediation_id or "",
                "subtask_id": subtask.id,
                "attempt": attempt_number,
                "max_attempts": max_attempts,
                "phase": "done",
                "outcome": "resolved",
            })
            await orchestrator._persist_remediation_attempt(
                task=task,
                remediation_id=remediation_id or "",
                subtask_id=subtask.id,
                attempt=attempt_number,
                max_attempts=max_attempts,
                phase="done",
                outcome="resolved",
                retry_strategy="resolved",
                reason_code=str(remediation_verification.reason_code or ""),
            )
            orchestrator._record_confirm_or_prune_attempt(
                task=task,
                subtask_id=subtask.id,
                status="resolved",
                attempt=attempt_number,
                max_attempts=max_attempts,
                transient=False,
                reason_code=remediation_verification.reason_code,
                retry_strategy="resolved",
                error="",
            )
            return True, ""

        remediation_strategy, missing_targets = orchestrator._retry.classify_failure(
            verification_feedback=remediation_verification.feedback,
            execution_error=remediation_result.summary,
            verification=remediation_verification,
        )
        combined_error = " | ".join(
            part
            for part in [
                remediation_verification.feedback,
                remediation_result.summary,
            ]
            if part
        ) or "process remediation failed"
        categorized = categorize_error(combined_error)
        transient = (
            remediation_strategy == RetryStrategy.RATE_LIMIT
            or categorized.category in {
                ErrorCategory.MODEL_ERROR,
                ErrorCategory.TIMEOUT,
            }
        )
        attempts.append(AttemptRecord(
            attempt=len(attempts) + 1,
            tier=escalated_tier,
            feedback=remediation_verification.feedback or None,
            error=combined_error,
            successful_tool_calls=[
                call for call in remediation_result.tool_calls
                if getattr(getattr(call, "result", None), "success", False)
            ],
            evidence_records=[
                item for item in remediation_result.evidence_records
                if isinstance(item, dict)
            ],
            retry_strategy=remediation_strategy,
            missing_targets=missing_targets,
            error_category=categorized.category,
        ))
        orchestrator._record_confirm_or_prune_attempt(
            task=task,
            subtask_id=subtask.id,
            status="failed",
            attempt=attempt_number,
            max_attempts=max_attempts,
            transient=transient,
            reason_code=remediation_verification.reason_code,
            retry_strategy=remediation_strategy.value,
            error=combined_error,
        )
        orchestrator._emit(REMEDIATION_ATTEMPT, task.id, {
            "remediation_id": remediation_id or "",
            "subtask_id": subtask.id,
            "attempt": attempt_number,
            "max_attempts": max_attempts,
            "phase": "done",
            "outcome": "failed",
            "retry_strategy": remediation_strategy.value,
            "transient": transient,
        })
        await orchestrator._persist_remediation_attempt(
            task=task,
            remediation_id=remediation_id or "",
            subtask_id=subtask.id,
            attempt=attempt_number,
            max_attempts=max_attempts,
            phase="done",
            outcome="failed",
            retry_strategy=remediation_strategy.value,
            transient=transient,
            reason_code=str(remediation_verification.reason_code or ""),
            error=combined_error,
        )
        last_failure = combined_error

        more_attempts_remain = attempt_number < max_attempts
        if remediation_strategy == RetryStrategy.RATE_LIMIT and more_attempts_remain:
            if backoff_seconds > 0:
                await asyncio.sleep(backoff_seconds)
            continue
        if retry_on_transient and transient and more_attempts_remain:
            if backoff_seconds > 0:
                await asyncio.sleep(backoff_seconds)
            continue
        break

    if placeholder_unconfirmed:
        orchestrator._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
            "subtask_id": subtask.id,
            "reason_code": placeholder_reason_code,
            "mode": "confirm_or_prune_remediation",
            "attempt": max_attempts,
            "max_attempts": max_attempts,
            "stage": "terminal",
            "outcome": "max_attempts_exhausted",
            "finding_count": len(placeholder_findings),
            "error": last_failure,
        })
    return False, last_failure


# Extracted remediation/placeholder resolution orchestration helpers

def _critical_path_behavior(self) -> str:
    process = self._process
    if process is None:
        return "block"
    getter = getattr(process, "remediation_critical_path_behavior", None)
    if callable(getter):
        value = str(getter() or "").strip().lower()
    else:
        remediation = getattr(process, "verification_remediation", None)
        value = str(
            getattr(remediation, "critical_path_behavior", "") or "",
        ).strip().lower()
    if value in _VALID_CRITICAL_PATH_BEHAVIORS:
        return value
    return "block"

def _is_hard_invariant_failure(verification: VerificationResult | None) -> bool:
    if verification is None:
        return False
    reason_code = str(verification.reason_code or "").strip().lower()
    severity = str(verification.severity_class or "").strip().lower()
    return (
        reason_code == "hard_invariant_failed"
        or severity == "hard_invariant"
    )

def _extract_unconfirmed_metadata(
    cls,
    verification: VerificationResult,
) -> dict[str, object]:
    metadata = verification.metadata if isinstance(verification.metadata, dict) else {}
    extracted: dict[str, object] = {}

    remediation_required = metadata.get("remediation_required", False)
    if isinstance(remediation_required, bool):
        extracted["remediation_required"] = remediation_required
    else:
        text = str(remediation_required or "").strip().lower()
        extracted["remediation_required"] = text in {"1", "true", "yes", "on"}

    remediation_mode = str(metadata.get("remediation_mode", "") or "").strip().lower()
    if remediation_mode:
        extracted["remediation_mode"] = remediation_mode

    failure_class = str(metadata.get("failure_class", "") or "").strip().lower()
    if failure_class:
        extracted["failure_class"] = failure_class

    missing_targets = cls._normalize_missing_targets(metadata.get("missing_targets"))
    if missing_targets:
        extracted["missing_targets"] = missing_targets

    placeholder_findings = metadata.get("placeholder_findings", [])
    if isinstance(placeholder_findings, list) and placeholder_findings:
        compacted: list[dict[str, object]] = []
        for raw in placeholder_findings:
            if not isinstance(raw, dict):
                continue
            compacted.append({
                "rule_name": str(raw.get("rule_name", "") or "").strip(),
                "pattern": str(raw.get("pattern", "") or ""),
                "source": str(raw.get("source", "") or ""),
                "file_path": str(raw.get("file_path", "") or "").strip(),
                "line": cls._to_int_or_none(raw.get("line")) or 0,
                "column": cls._to_int_or_none(raw.get("column")) or 0,
                "token": str(raw.get("token", "") or ""),
                "context": str(raw.get("context", "") or ""),
            })
            if len(compacted) >= 120:
                break
        if compacted:
            extracted["placeholder_findings"] = compacted
            extracted["placeholder_finding_count"] = len(compacted)
    else:
        finding_count = cls._to_int_or_none(metadata.get("placeholder_finding_count"))
        if finding_count is not None and finding_count > 0:
            extracted["placeholder_finding_count"] = int(finding_count)

    unverified_claim_count = cls._to_int_or_none(
        metadata.get("unverified_claim_count"),
    )
    if unverified_claim_count is not None:
        extracted["unverified_claim_count"] = max(0, unverified_claim_count)

    verified_claim_count = cls._to_int_or_none(
        metadata.get("verified_claim_count"),
    )
    if verified_claim_count is not None:
        extracted["verified_claim_count"] = max(0, verified_claim_count)

    supporting_ratio = cls._to_ratio_or_none(metadata.get("supporting_ratio"))
    if supporting_ratio is not None:
        extracted["supporting_ratio"] = supporting_ratio

    return extracted

def _is_placeholder_unconfirmed_failure(
    *,
    verification: VerificationResult | None,
    placeholder_metadata: dict[str, object] | None = None,
) -> bool:
    reason_code = ""
    metadata: dict[str, object] = {}
    if verification is not None:
        reason_code = str(verification.reason_code or "").strip().lower()
        if isinstance(verification.metadata, dict):
            metadata = dict(verification.metadata)
    if isinstance(placeholder_metadata, dict):
        metadata = {**metadata, **placeholder_metadata}
    if reason_code in _PLACEHOLDER_UNCONFIRMED_REASON_CODES:
        return True
    failure_class = str(metadata.get("failure_class", "") or "").strip().lower()
    remediation_mode = str(metadata.get("remediation_mode", "") or "").strip().lower()
    findings = metadata.get("placeholder_findings")
    has_findings = isinstance(findings, list) and bool(findings)
    if not has_findings:
        try:
            has_findings = int(metadata.get("placeholder_finding_count", 0) or 0) > 0
        except (TypeError, ValueError):
            has_findings = False
    return (
        has_findings
        and (
            failure_class == "recoverable_placeholder"
            or remediation_mode == "confirm_or_prune"
        )
    )

def _apply_deterministic_placeholder_prune_actions(
    *,
    workspace: Path,
    findings: list[dict[str, object]],
    replacement_token: str = "UNSUPPORTED_NO_EVIDENCE",
) -> dict[str, object]:
    root = workspace.resolve(strict=False)
    grouped: dict[str, list[dict[str, object]]] = {}
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        rel_path = orchestrator_validity.normalize_workspace_relpath(
            root,
            str(finding.get("file_path", "") or ""),
        )
        if not rel_path:
            continue
        grouped.setdefault(rel_path, []).append(finding)

    actions: list[dict[str, object]] = []
    files_modified: list[str] = []
    remaining_count = 0
    applied_count = 0

    for rel_path, rows in grouped.items():
        file_path = root / rel_path
        try:
            if not file_path.exists() or not file_path.is_file():
                continue
            original = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = original.splitlines(keepends=True)
        changed = False

        for row in rows:
            token = str(row.get("token", "") or "")
            if not token:
                continue
            line_no = 0
            try:
                line_no = int(row.get("line", 0) or 0)
            except (TypeError, ValueError):
                line_no = 0
            replaced = False
            if 1 <= line_no <= len(lines):
                line_text = lines[line_no - 1]
                if token in line_text:
                    lines[line_no - 1] = line_text.replace(
                        token,
                        replacement_token,
                        1,
                    )
                    replaced = True
            if not replaced:
                current_text = "".join(lines)
                if token in current_text:
                    lines = current_text.replace(token, replacement_token, 1).splitlines(
                        keepends=True,
                    )
                    replaced = True
            if replaced:
                changed = True
                applied_count += 1
                actions.append({
                    "file_path": rel_path,
                    "line": max(0, line_no),
                    "token": token,
                    "action": "pruned_token",
                    "replacement": replacement_token,
                })

        updated = "".join(lines)
        if changed and updated != original:
            try:
                file_path.write_text(updated, encoding="utf-8")
                files_modified.append(rel_path)
            except OSError:
                continue

    for finding in findings:
        if not isinstance(finding, dict):
            continue
        token = str(finding.get("token", "") or "")
        if not token:
            continue
        rel_path = orchestrator_validity.normalize_workspace_relpath(
            root,
            str(finding.get("file_path", "") or ""),
        )
        if not rel_path:
            continue
        file_path = root / rel_path
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            remaining_count += 1
            continue
        if token in text:
            remaining_count += 1

    return {
        "applied_count": int(applied_count),
        "remaining_count": int(remaining_count),
        "files_modified": files_modified,
        "actions": actions,
    }

def _summarize_placeholder_resolution_state(
    *,
    workspace: Path,
    findings: list[dict[str, object]],
    replacement_token: str = "UNSUPPORTED_NO_EVIDENCE",
) -> dict[str, int]:
    root = workspace.resolve(strict=False)
    tracked_findings = 0
    remaining_findings = 0
    replacement_token_occurrences = 0
    scanned_file_count = 0

    grouped_tokens: dict[str, set[str]] = {}
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        token = str(finding.get("token", "") or "")
        if not token:
            continue
        rel_path = orchestrator_validity.normalize_workspace_relpath(
            root,
            str(finding.get("file_path", "") or ""),
        )
        if not rel_path:
            continue
        tracked_findings += 1
        grouped_tokens.setdefault(rel_path, set()).add(token)

    for rel_path, tokens in grouped_tokens.items():
        file_path = root / rel_path
        try:
            if not file_path.exists() or not file_path.is_file():
                remaining_findings += len(tokens)
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            remaining_findings += len(tokens)
            continue
        scanned_file_count += 1
        replacement_token_occurrences += text.count(replacement_token)
        for token in tokens:
            if token in text:
                remaining_findings += 1

    return {
        "tracked_findings": int(tracked_findings),
        "remaining_findings": int(remaining_findings),
        "replacement_token_occurrences": int(replacement_token_occurrences),
        "scanned_file_count": int(scanned_file_count),
    }


def _reseal_placeholder_prepass_mutations(
    orchestrator,
    *,
    task: Task,
    subtask: Subtask,
    workspace: Path,
    files_modified: list[str],
) -> dict[str, object]:
    if not files_modified:
        return {"resealed_count": 0, "resealed_paths": []}

    seals = orchestrator._artifact_seal_registry(task)
    if not isinstance(seals, dict):
        return {"resealed_count": 0, "resealed_paths": []}

    run_id = ""
    try:
        run_id = str(orchestrator._task_run_id(task) or "")
    except Exception:
        run_id = ""
    sealed_at = datetime.now().isoformat()
    resealed_paths: list[str] = []
    seen: set[str] = set()

    for raw in files_modified:
        rel_path = orchestrator_validity.normalize_workspace_relpath(
            workspace,
            str(raw or ""),
        )
        if not rel_path or rel_path in seen:
            continue
        seen.add(rel_path)
        if orchestrator._is_intermediate_artifact_path(task=task, relpath=rel_path):
            continue
        path = (workspace / rel_path).resolve(strict=False)
        if not path.exists() or not path.is_file():
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        sha256 = hashlib.sha256(payload).hexdigest()
        previous = seals.get(rel_path)
        previous_dict = previous if isinstance(previous, dict) else {}
        previous_sha = str(previous_dict.get("sha256", "") or "").strip()

        next_seal = dict(previous_dict)
        next_seal.update({
            "path": rel_path,
            "sha256": sha256,
            "size_bytes": len(payload),
            "tool": "deterministic_placeholder_prepass",
            "tool_call_id": "",
            "subtask_id": subtask.id,
            "run_id": run_id,
            "sealed_at": sealed_at,
            "resealed_after_mutation": True,
            "resealed_reason": "deterministic_placeholder_prepass",
        })
        if previous_sha and previous_sha != sha256:
            next_seal["previous_sha256"] = previous_sha
        else:
            next_seal.pop("previous_sha256", None)
        seals[rel_path] = next_seal
        resealed_paths.append(rel_path)

    if resealed_paths:
        task.metadata["artifact_seals"] = seals
    return {
        "resealed_count": len(resealed_paths),
        "resealed_paths": resealed_paths,
    }

async def _run_deterministic_placeholder_prepass(
    self,
    *,
    task: Task,
    subtask: Subtask,
    verification: VerificationResult | None = None,
    placeholder_metadata: dict[str, object] | None = None,
    origin: str = "unconfirmed_data_retry",
    attempt_number: int | None = None,
    max_attempts: int | None = None,
) -> tuple[bool, str, dict[str, object]]:
    metadata: dict[str, object] = {}
    if verification is not None and isinstance(verification.metadata, dict):
        metadata.update(verification.metadata)
    if isinstance(placeholder_metadata, dict):
        metadata.update(placeholder_metadata)
    findings = self._normalize_placeholder_findings(metadata.get("placeholder_findings"))
    reason_code = str(getattr(verification, "reason_code", "") or "").strip().lower()
    parent_mode = str(origin or "unconfirmed_data_retry").strip().lower()
    source_remediation_mode = str(metadata.get("remediation_mode", "") or "").strip().lower()
    failure_class = str(metadata.get("failure_class", "") or "").strip().lower()
    finding_count = len(findings)
    self._emit(PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED, task.id, {
        "subtask_id": subtask.id,
        "reason_code": reason_code,
        "mode": _PLACEHOLDER_PREPASS_MODE,
        "parent_mode": parent_mode,
        "attempt": int(attempt_number or 0),
        "max_attempts": int(max_attempts or 0),
        "remediation_mode": _PLACEHOLDER_PREPASS_MODE,
        "source_remediation_mode": source_remediation_mode,
        "failure_class": failure_class,
        "finding_count": finding_count,
    })

    workspace_text = str(getattr(task, "workspace", "") or "").strip()
    if not workspace_text:
        self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
            "subtask_id": subtask.id,
            "reason_code": reason_code,
            "mode": _PLACEHOLDER_PREPASS_MODE,
            "parent_mode": parent_mode,
            "attempt": int(attempt_number or 0),
            "max_attempts": int(max_attempts or 0),
            "stage": "deterministic_prepass",
            "outcome": "workspace_unavailable",
            "finding_count": finding_count,
        })
        return False, "deterministic placeholder remediation skipped: workspace unavailable", {}
    workspace = Path(workspace_text)
    if not workspace.exists() or not workspace.is_dir():
        self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
            "subtask_id": subtask.id,
            "reason_code": reason_code,
            "mode": _PLACEHOLDER_PREPASS_MODE,
            "parent_mode": parent_mode,
            "attempt": int(attempt_number or 0),
            "max_attempts": int(max_attempts or 0),
            "stage": "deterministic_prepass",
            "outcome": "workspace_path_missing",
            "finding_count": finding_count,
        })
        return (
            False,
            "deterministic placeholder remediation skipped: workspace path missing",
            {},
        )

    if not findings:
        self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
            "subtask_id": subtask.id,
            "reason_code": reason_code,
            "mode": _PLACEHOLDER_PREPASS_MODE,
            "parent_mode": parent_mode,
            "attempt": int(attempt_number or 0),
            "max_attempts": int(max_attempts or 0),
            "stage": "deterministic_prepass",
            "outcome": "no_structured_findings",
            "finding_count": finding_count,
        })
        return (
            False,
            "deterministic placeholder remediation skipped: no structured findings",
            {},
        )

    outcome = await run_blocking_io(
        self._apply_deterministic_placeholder_prune_actions,
        workspace=workspace,
        findings=findings,
    )
    if not isinstance(outcome, dict):
        self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
            "subtask_id": subtask.id,
            "reason_code": reason_code,
            "mode": _PLACEHOLDER_PREPASS_MODE,
            "parent_mode": parent_mode,
            "attempt": int(attempt_number or 0),
            "max_attempts": int(max_attempts or 0),
            "stage": "deterministic_prepass",
            "outcome": "invalid_outcome",
            "finding_count": finding_count,
        })
        return False, "deterministic placeholder remediation failed: invalid outcome", {}

    applied_count = int(outcome.get("applied_count", 0) or 0)
    remaining_count = int(outcome.get("remaining_count", 0) or 0)
    files_modified = outcome.get("files_modified", [])
    if not isinstance(files_modified, list):
        files_modified = []
    summary = (
        "Deterministic placeholder prepass "
        f"applied={applied_count} remaining={remaining_count} "
        f"files={len(files_modified)}."
    )
    details = {
        "mode": _PLACEHOLDER_PREPASS_MODE,
        "parent_mode": parent_mode,
        "applied_count": applied_count,
        "remaining_count": remaining_count,
        "files_modified": files_modified[:40],
        "actions": outcome.get("actions", [])[:120],
    }
    if applied_count > 0 and files_modified:
        reseal_result = _reseal_placeholder_prepass_mutations(
            self,
            task=task,
            subtask=subtask,
            workspace=workspace,
            files_modified=files_modified,
        )
        resealed_count = int(reseal_result.get("resealed_count", 0) or 0)
        resealed_paths = reseal_result.get("resealed_paths", [])
        if not isinstance(resealed_paths, list):
            resealed_paths = []
        details["resealed_count"] = resealed_count
        details["resealed_paths"] = resealed_paths[:40]
        if resealed_count > 0:
            self._emit(SEALED_RESEAL_APPLIED, task.id, {
                "subtask_id": subtask.id,
                "tool": "deterministic_placeholder_prepass",
                "tool_call_id": "",
                "path_count": resealed_count,
            })
    if applied_count > 0:
        self._emit(PLACEHOLDER_PRUNED, task.id, {
            "subtask_id": subtask.id,
            "reason_code": reason_code,
            "mode": _PLACEHOLDER_PREPASS_MODE,
            "parent_mode": parent_mode,
            "attempt": int(attempt_number or 0),
            "max_attempts": int(max_attempts or 0),
            "stage": "deterministic_prepass",
            "finding_count": finding_count,
            "applied_count": applied_count,
            "remaining_count": remaining_count,
            "files_modified_count": len(files_modified),
            "files_modified": files_modified[:40],
        })
    if remaining_count > 0 or applied_count <= 0:
        unresolved_outcome = (
            "placeholders_remaining"
            if remaining_count > 0
            else "no_mutations_applied"
        )
        self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
            "subtask_id": subtask.id,
            "reason_code": reason_code,
            "mode": _PLACEHOLDER_PREPASS_MODE,
            "parent_mode": parent_mode,
            "attempt": int(attempt_number or 0),
            "max_attempts": int(max_attempts or 0),
            "stage": "deterministic_prepass",
            "outcome": unresolved_outcome,
            "finding_count": finding_count,
            "applied_count": applied_count,
            "remaining_count": remaining_count,
            "files_modified_count": len(files_modified),
        })
    return (applied_count > 0 and remaining_count == 0), summary, details

def _apply_unconfirmed_follow_up_success(
    self,
    *,
    result: SubtaskResult,
    verification: VerificationResult,
    note: str,
    default_reason_code: str,
) -> None:
    verification.passed = True
    if verification.outcome == "fail":
        verification.outcome = (
            "partial_verified"
            if self._config.verification.allow_partial_verified
            else "pass_with_warnings"
        )
    elif verification.outcome == "pass":
        verification.outcome = "pass_with_warnings"
    if not verification.reason_code:
        verification.reason_code = default_reason_code
    if verification.severity_class == "hard_invariant":
        verification.severity_class = "semantic"
    verification.feedback = (
        f"{verification.feedback}\n{note}" if verification.feedback else note
    )
    result.status = SubtaskResultStatus.SUCCESS

def _resolution_plan_items(raw: object, *, max_items: int = 8) -> list[str]:
    if isinstance(raw, str):
        parts = [
            item.strip(" -\t")
            for item in raw.splitlines()
            if item.strip()
        ]
    elif isinstance(raw, list):
        parts = [str(item or "").strip() for item in raw]
    else:
        return []
    normalized: list[str] = []
    for item in parts:
        text = str(item or "").strip()
        if not text or text in normalized:
            continue
        normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized

def _compact_failure_resolution_metadata_value(
    value: object,
    *,
    depth: int = 0,
    max_depth: int = 3,
    max_list_items: int = 8,
    max_dict_items: int = 12,
    max_text_chars: int = 220,
) -> object:
    return orchestrator_validity.compact_failure_resolution_metadata_value(
        value,
        depth=depth,
        max_depth=max_depth,
        max_list_items=max_list_items,
        max_dict_items=max_dict_items,
        max_text_chars=max_text_chars,
    )

def _summarize_failure_resolution_metadata(
    cls,
    metadata: dict[str, object],
) -> dict[str, object]:
    del cls  # retained for compatibility during extraction
    return orchestrator_validity.summarize_failure_resolution_metadata(
        metadata,
        keys=_FAILURE_RESOLUTION_METADATA_KEYS,
    )

def _failure_resolution_metadata_char_budget(self) -> int:
    raw = getattr(
        getattr(self._config, "limits", None),
        "evidence_context_text_max_chars",
        4000,
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1400
    return max(600, min(value, 2200))

def _format_failure_resolution_plan(self, response: ModelResponse) -> str:
    validation = self._validator.validate_json_response(
        response,
        expected_keys=["diagnosis", "actions"],
    )
    if validation.valid and isinstance(validation.parsed, dict):
        payload = validation.parsed
        diagnosis = str(payload.get("diagnosis", "") or "").strip()
        actions = self._resolution_plan_items(payload.get("actions"))
        guardrails = self._resolution_plan_items(payload.get("guardrails"))
        success = self._resolution_plan_items(
            payload.get("success_criteria"),
        )
        lines: list[str] = []
        if diagnosis:
            lines.append(f"Diagnosis: {diagnosis}")
        if actions:
            lines.append("Actions:")
            for index, action in enumerate(actions, start=1):
                lines.append(f"{index}. {action}")
        if guardrails:
            lines.append("Guardrails:")
            for item in guardrails:
                lines.append(f"- {item}")
        if success:
            lines.append("Success criteria:")
            for item in success:
                lines.append(f"- {item}")
        rendered = "\n".join(lines).strip()
        if rendered:
            return rendered[:2400]
    fallback = str(response.text or "").strip()
    if not fallback:
        return ""
    return fallback[:2400]

def _build_failure_resolution_prompt(
    self,
    *,
    subtask: Subtask,
    result: SubtaskResult,
    verification: VerificationResult,
    strategy: RetryStrategy,
    missing_targets: list[str],
    prior_attempts: list[AttemptRecord],
) -> str:
    raw_metadata = (
        verification.metadata
        if isinstance(verification.metadata, dict)
        else {}
    )
    metadata = self._summarize_failure_resolution_metadata(raw_metadata)
    metadata_json = json.dumps(metadata, indent=2, sort_keys=True, default=str)
    metadata_budget = self._failure_resolution_metadata_char_budget()
    if len(metadata_json) > metadata_budget:
        metadata_json = json.dumps(
            {
                "_loom_meta": "metadata_truncated",
                "total_chars": len(metadata_json),
                "included_keys": list(metadata.keys())[:12],
            },
            indent=2,
            sort_keys=True,
        )
    output_policy = self._output_write_policy_for_subtask(subtask=subtask)
    expected_deliverables = list(output_policy.get("expected_deliverables", []))
    forbidden_deliverables = list(output_policy.get("forbidden_deliverables", []))
    changed_files = self._files_from_attempts(prior_attempts)
    current_changed = self._files_from_tool_calls(result.tool_calls)
    for path in current_changed:
        if path not in changed_files:
            changed_files.append(path)
    lines = [
        "You are planning a targeted remediation for a failed subtask.",
        "Return strict JSON with keys:",
        (
            '{"diagnosis": "...", "actions": ["..."], '
            '"guardrails": ["..."], "success_criteria": ["..."]}'
        ),
        "Keep the plan short and actionable.",
        "",
        "Rules:",
        "- Handle the observed failure generically; do not hardcode one-off logic.",
        "- Preserve validated outputs and evidence.",
        "- Prefer minimal edits over broad reruns.",
        (
            "- If outputs must align to canonical deliverables, explain how to "
            "reconcile file paths safely."
        ),
        "",
        f"Subtask ID: {subtask.id}",
        f"Subtask description: {subtask.description}",
        f"Acceptance criteria: {subtask.acceptance_criteria}",
        f"Retry strategy: {strategy.value}",
        f"Verification tier: {verification.tier}",
        f"Verification outcome: {verification.outcome}",
        f"Reason code: {verification.reason_code}",
        f"Severity class: {verification.severity_class}",
        f"Verification feedback: {verification.feedback or ''}",
        f"Execution summary: {result.summary or ''}",
    ]
    if missing_targets:
        lines.append(f"Missing targets: {', '.join(missing_targets)}")
    if expected_deliverables:
        lines.append(
            "Expected deliverables: " + ", ".join(expected_deliverables),
        )
    if forbidden_deliverables:
        lines.append(
            "Forbidden canonical deliverables for this subtask: "
            + ", ".join(forbidden_deliverables),
        )
    if changed_files:
        lines.append("Touched files: " + ", ".join(changed_files))
    if metadata_json != "{}":
        lines.extend([
            "Verification metadata:",
            metadata_json,
        ])
    return "\n".join(lines)

async def _plan_failure_resolution(
    self,
    *,
    task: Task,
    subtask: Subtask,
    result: SubtaskResult,
    verification: VerificationResult,
    strategy: RetryStrategy,
    missing_targets: list[str],
    prior_attempts: list[AttemptRecord],
) -> str:
    prompt = self._build_failure_resolution_prompt(
        subtask=subtask,
        result=result,
        verification=verification,
        strategy=strategy,
        missing_targets=missing_targets,
        prior_attempts=prior_attempts,
    )
    if not prompt.strip():
        return ""

    model = self._router.select(tier=2, role="planner")
    request_messages = [{"role": "user", "content": prompt}]
    policy = ModelRetryPolicy.from_execution_config(self._config.execution)
    invocation_attempt = 0
    request_diag = None
    max_tokens = self._planning_response_max_tokens()
    if max_tokens is None:
        max_tokens = 900
    max_tokens = max(256, min(max_tokens, 1500))

    async def _invoke_model():
        nonlocal invocation_attempt, request_diag
        invocation_attempt += 1
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin="orchestrator.failure_resolution.complete",
        )
        self._emit(MODEL_INVOCATION, task.id, {
            "subtask_id": subtask.id,
            "model": model.name,
            "phase": "start",
            "operation": "failure_resolution_plan",
            "invocation_attempt": invocation_attempt,
            "invocation_max_attempts": policy.max_attempts,
            "retry_strategy": strategy.value,
            **request_diag.to_event_payload(),
        })
        return await model.complete(
            request_messages,
            max_tokens=max_tokens,
        )

    def _on_failure(
        attempt: int,
        max_attempts: int,
        error: BaseException,
        remaining: int,
    ) -> None:
        self._emit(MODEL_INVOCATION, task.id, {
            "subtask_id": subtask.id,
            "model": model.name,
            "phase": "done",
            "operation": "failure_resolution_plan",
            "invocation_attempt": attempt,
            "invocation_max_attempts": max_attempts,
            "retry_queue_remaining": remaining,
            "origin": request_diag.origin if request_diag else "",
            "error_type": type(error).__name__,
            "error": str(error),
            "retry_strategy": strategy.value,
        })

    try:
        response = await call_with_model_retry(
            _invoke_model,
            policy=policy,
            on_failure=_on_failure,
        )
    except Exception as e:
        logger.debug(
            "Failure-resolution planner call failed for %s/%s: %s",
            task.id,
            subtask.id,
            e,
            exc_info=True,
        )
        return ""

    self._emit(MODEL_INVOCATION, task.id, {
        "subtask_id": subtask.id,
        "model": model.name,
        "phase": "done",
        "operation": "failure_resolution_plan",
        "invocation_attempt": invocation_attempt,
        "invocation_max_attempts": policy.max_attempts,
        "origin": request_diag.origin if request_diag else "",
        "retry_strategy": strategy.value,
        **collect_response_diagnostics(response).to_event_payload(),
    })
    return self._format_failure_resolution_plan(response)

def _build_remediation_retry_context(
    self,
    *,
    strategy: RetryStrategy,
    reason_code: str = "",
) -> str:
    lines = [
        "TARGETED REMEDIATION:",
        "- Keep already validated work; avoid redoing solved sections.",
        "- Resolve only failing verification findings and missing evidence links.",
        "- Use explicit evidence to confirm uncertain claims; otherwise relabel "
        "or remove unsupported claims per process policy.",
        "- Make the smallest safe edits needed to satisfy acceptance criteria.",
    ]
    normalized_reason = str(reason_code or "").strip().lower()
    if normalized_reason == "missing_precedent_transactions":
        lines.extend([
            "REASON-SPECIFIC REMEDIATION:",
            "- Add explicit structured precedent transaction evidence in canonical "
            "deliverables.",
            "- Include deal identifiers, multiple basis, assumptions, and implied "
            "valuation bridge.",
            "- Ensure precedent evidence is not only narrative prose.",
        ])
    elif normalized_reason == "csv_schema_mismatch":
        lines.extend([
            "REASON-SPECIFIC REMEDIATION:",
            "- Repair CSV row-width mismatches so every non-empty row matches the header.",
            "- Re-check appended rows and delimiter/quoting consistency before retry.",
        ])
    process = self._process
    if process is not None:
        instructions = ""
        if normalized_reason:
            instructions = process.prompt_remediation_instructions(normalized_reason)
        if not instructions:
            instructions = process.prompt_remediation_instructions(strategy.value)
        if instructions:
            lines.append("PROCESS REMEDIATION INSTRUCTIONS:")
            lines.append(instructions)
    return "\n".join(lines)

def _record_confirm_or_prune_attempt(
    self,
    *,
    task: Task,
    subtask_id: str,
    status: str,
    attempt: int,
    max_attempts: int,
    transient: bool,
    reason_code: str,
    retry_strategy: str,
    error: str,
) -> None:
    attempts_map = task.metadata.get("confirm_or_prune_attempts")
    if not isinstance(attempts_map, dict):
        attempts_map = {}
        task.metadata["confirm_or_prune_attempts"] = attempts_map
    rows = attempts_map.get(subtask_id)
    if not isinstance(rows, list):
        rows = []
        attempts_map[subtask_id] = rows
    rows.append({
        "attempt": int(attempt),
        "max_attempts": int(max_attempts),
        "status": str(status),
        "transient": bool(transient),
        "reason_code": str(reason_code or "").strip().lower(),
        "retry_strategy": str(retry_strategy or "").strip().lower(),
        "error": str(error or ""),
        "at": datetime.now().isoformat(),
    })
    if len(rows) > 25:
        del rows[:-25]

async def _persist_subtask_attempt_record(
    self,
    *,
    task: Task,
    subtask: Subtask,
    subtask_id: str,
    attempt_record: AttemptRecord,
    verification: VerificationResult,
) -> None:
    if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
        return
    verification_metadata = (
        dict(verification.metadata)
        if isinstance(verification.metadata, dict)
        else {}
    )
    attempt_metadata: dict[str, object] = {
        "model_tier": int(getattr(subtask, "model_tier", 1) or 1),
        "verification_tier": int(getattr(subtask, "verification_tier", 1) or 1),
        "acceptance_criteria": str(getattr(subtask, "acceptance_criteria", "") or ""),
        "validity_contract_hash": str(
            getattr(subtask, "validity_contract_hash", "") or "",
        ),
        "validity_contract_snapshot": (
            dict(getattr(subtask, "validity_contract_snapshot", {}))
            if isinstance(getattr(subtask, "validity_contract_snapshot", {}), dict)
            else {}
        ),
    }
    if "claim_status_counts" in verification_metadata:
        attempt_metadata["claim_status_counts"] = verification_metadata.get(
            "claim_status_counts",
        )
    if "claim_reason_codes" in verification_metadata:
        attempt_metadata["claim_reason_codes"] = verification_metadata.get("claim_reason_codes")
    if "supported_ratio" in verification_metadata:
        attempt_metadata["supported_ratio"] = verification_metadata.get("supported_ratio")
    if "unverified_ratio" in verification_metadata:
        attempt_metadata["unverified_ratio"] = verification_metadata.get("unverified_ratio")
    if "critical_support_ratio" in verification_metadata:
        attempt_metadata["critical_support_ratio"] = verification_metadata.get(
            "critical_support_ratio",
        )
    try:
        await self._memory.insert_subtask_attempt(
            task_id=task.id,
            run_id=self._task_run_id(task),
            subtask_id=subtask_id,
            attempt=int(getattr(attempt_record, "attempt", 0) or 0),
            tier=int(getattr(attempt_record, "tier", 1) or 1),
            retry_strategy=str(
                getattr(getattr(attempt_record, "retry_strategy", None), "value", "")
                or getattr(attempt_record, "retry_strategy", "")
                or RetryStrategy.GENERIC.value
            ),
            reason_code=str(getattr(verification, "reason_code", "") or ""),
            feedback=str(getattr(attempt_record, "feedback", "") or ""),
            error=str(getattr(attempt_record, "error", "") or ""),
            missing_targets=list(getattr(attempt_record, "missing_targets", []) or []),
            error_category=str(
                getattr(
                    getattr(attempt_record, "error_category", None),
                    "value",
                    "",
                ) or ""
            ),
            metadata=attempt_metadata,
        )
    except Exception:
        logger.debug(
            "Failed persisting subtask attempt for %s/%s",
            task.id,
            subtask_id,
            exc_info=True,
        )

async def _sync_remediation_queue_to_db(self, task: Task) -> None:
    if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
        return
    queue = task.metadata.get("remediation_queue")
    if not isinstance(queue, list):
        return
    for item in queue:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        payload["task_id"] = task.id
        payload.setdefault("run_id", self._task_run_id(task))
        try:
            await self._memory.upsert_remediation_item(payload)
        except Exception:
            logger.debug(
                "Failed syncing remediation item %s for task %s",
                str(item.get("id", "")),
                task.id,
                exc_info=True,
            )

async def _hydrate_remediation_queue_from_db(self, task: Task) -> None:
    if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
        return
    try:
        db_items = await self._memory.list_remediation_items(task_id=task.id)
    except Exception:
        logger.debug("Failed loading remediation queue from db for %s", task.id, exc_info=True)
        return
    if not db_items:
        return
    queue = self._remediation_queue(task)
    existing_ids = {
        str(item.get("id", "")).strip()
        for item in queue
        if isinstance(item, dict)
    }
    for item in db_items:
        item_id = str(item.get("id", "")).strip()
        if not item_id or item_id in existing_ids:
            continue
        queue.append(item)
    self._state.save(task)

async def _persist_remediation_attempt(
    self,
    *,
    task: Task,
    remediation_id: str,
    subtask_id: str,
    attempt: int,
    max_attempts: int,
    phase: str,
    outcome: str = "",
    retry_strategy: str = "",
    transient: bool = False,
    reason_code: str = "",
    error: str = "",
) -> None:
    if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
        return
    try:
        await self._memory.insert_remediation_attempt(
            remediation_id=remediation_id,
            task_id=task.id,
            run_id=self._task_run_id(task),
            subtask_id=subtask_id,
            attempt=attempt,
            max_attempts=max_attempts,
            phase=phase,
            outcome=outcome,
            retry_strategy=retry_strategy,
            transient=transient,
            reason_code=reason_code,
            error=error,
        )
    except Exception:
        logger.debug(
            "Failed persisting remediation attempt for %s/%s",
            task.id,
            remediation_id,
            exc_info=True,
        )

def _remediation_queue(self, task: Task) -> list[dict]:
    queue = task.metadata.get("remediation_queue")
    if not isinstance(queue, list):
        queue = []
        task.metadata["remediation_queue"] = queue
    queue[:] = [item for item in queue if isinstance(item, dict)]
    return queue
