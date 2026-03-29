"""Telemetry event payload contracts and redaction utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from loom.events import types

_EVENT_FAMILIES: dict[str, str] = {}


def _register(events: tuple[str, ...], family: str) -> None:
    for event_type in events:
        _EVENT_FAMILIES[event_type] = family


_register(
    (
        types.TASK_CREATED,
        types.TASK_PLANNING,
        types.TASK_PLAN_READY,
        types.TASK_PLAN_NORMALIZED,
        types.TASK_EXECUTING,
        types.TASK_REPLANNING,
        types.TASK_REPLAN_REJECTED,
        types.TASK_STALLED,
        types.TASK_STALLED_RECOVERY_ATTEMPTED,
        types.TASK_BUDGET_EXHAUSTED,
        types.TASK_PLAN_DEGRADED,
        types.TASK_COMPLETED,
        types.TASK_FAILED,
        types.TASK_CANCELLED,
        types.TASK_RESTARTED,
        types.TASK_PAUSED,
        types.TASK_RESUMED,
        types.TASK_INJECTED,
        types.TASK_CANCEL_REQUESTED,
        types.TASK_CANCEL_ACK,
        types.TASK_CANCEL_TIMEOUT,
        types.TASK_RUN_ACQUIRED,
        types.TASK_RUN_HEARTBEAT,
        types.TASK_RUN_RECOVERED,
        types.TELEMETRY_RUN_SUMMARY,
        types.RUN_VALIDITY_SCORECARD,
        types.TELEMETRY_MODE_CHANGED,
        types.TELEMETRY_SETTINGS_WARNING,
    ),
    "task",
)

_register(
    (
        types.SUBTASK_STARTED,
        types.SUBTASK_COMPLETED,
        types.SUBTASK_FAILED,
        types.SUBTASK_OUTCOME_STALE,
        types.SUBTASK_BLOCKED,
        types.SUBTASK_OUTPUT_CONFLICT_DEFERRED,
        types.SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING,
        types.SUBTASK_POLICY_RECONCILED,
        types.SUBTASK_RETRYING,
    ),
    "subtask",
)

_register(
    (
        types.TOOL_CALL_STARTED,
        types.TOOL_CALL_COMPLETED,
        types.TOOL_CALL_DEDUPLICATED,
        types.FORBIDDEN_CANONICAL_WRITE_BLOCKED,
        types.ARTIFACT_CONFINEMENT_VIOLATION,
        types.ARTIFACT_INGEST_CLASSIFIED,
        types.ARTIFACT_INGEST_COMPLETED,
        types.ARTIFACT_RETENTION_PRUNED,
        types.ARTIFACT_READ_COMPLETED,
        types.COMPACTION_POLICY_DECISION,
        types.OVERFLOW_FALLBACK_APPLIED,
    ),
    "tool",
)

_register(
    (
        types.VERIFICATION_STARTED,
        types.VERIFICATION_PASSED,
        types.VERIFICATION_FAILED,
        types.VERIFICATION_RULE_APPLIED,
        types.VERIFICATION_RULE_SKIPPED,
        types.VERIFICATION_OUTCOME,
        types.VERIFICATION_SHADOW_DIFF,
        types.VERIFICATION_FALSE_NEGATIVE_CANDIDATE,
        types.VERIFICATION_INCONCLUSIVE_RATE,
        types.VERIFICATION_RULE_FAILURE_BY_TYPE,
        types.VERIFICATION_DETERMINISTIC_BLOCK_RATE,
        types.VERIFICATION_CONTRADICTION_DETECTED,
        types.CLAIM_VERIFICATION_SUMMARY,
        types.CLAIMS_PRUNED,
        types.SYNTHESIS_INPUT_GATE_DECISION,
        types.ARTIFACT_SEAL_VALIDATION,
    ),
    "verification",
)

_register(
    (
        types.APPROVAL_REQUESTED,
        types.APPROVAL_RECEIVED,
        types.APPROVAL_REJECTED,
        types.APPROVAL_TIMED_OUT,
        types.ASK_USER_REQUESTED,
        types.ASK_USER_ANSWERED,
        types.ASK_USER_TIMEOUT,
        types.ASK_USER_CANCELLED,
        types.STEER_INSTRUCTION,
    ),
    "human_loop",
)

_register(
    (
        types.REMEDIATION_QUEUED,
        types.REMEDIATION_STARTED,
        types.REMEDIATION_ATTEMPT,
        types.REMEDIATION_RESOLVED,
        types.REMEDIATION_FAILED,
        types.REMEDIATION_EXPIRED,
        types.REMEDIATION_TERMINAL,
        types.UNCONFIRMED_DATA_QUEUED,
        types.PLACEHOLDER_FINDINGS_EXTRACTED,
        types.PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED,
        types.PLACEHOLDER_FILLED,
        types.PLACEHOLDER_PRUNED,
        types.PLACEHOLDER_REMEDIATION_UNRESOLVED,
    ),
    "remediation",
)

_register(
    (
        types.ITERATION_STARTED,
        types.ITERATION_GATE_FAILED,
        types.ITERATION_RETRYING,
        types.ITERATION_COMPLETED,
        types.ITERATION_TERMINAL,
        types.ITERATION_STATE_RECONCILED,
    ),
    "iteration",
)

_register(
    (
        types.WEBHOOK_DELIVERY_ATTEMPTED,
        types.WEBHOOK_DELIVERY_SUCCEEDED,
        types.WEBHOOK_DELIVERY_FAILED,
        types.WEBHOOK_DELIVERY_DROPPED,
    ),
    "webhook",
)

_register(
    (
        types.MODEL_INVOCATION,
        types.DB_MIGRATION_START,
        types.DB_MIGRATION_APPLIED,
        types.DB_MIGRATION_VERIFY_FAILED,
        types.DB_MIGRATION_FAILED,
        types.DB_SCHEMA_READY,
        types.TELEMETRY_DIAGNOSTIC,
        types.TOKEN_STREAMED,
        types.CONVERSATION_MESSAGE,
    ),
    "internal",
)


_FAMILY_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "task": (),
    "subtask": ("subtask_id",),
    "tool": ("subtask_id",),
    "verification": ("subtask_id",),
    "human_loop": (),
    "remediation": ("subtask_id",),
    "iteration": ("subtask_id",),
    "webhook": ("delivery_target_host",),
    "internal": (),
}


_EVENT_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    types.TASK_CREATED: ("goal",),
    types.SUBTASK_BLOCKED: ("subtask_id", "reasons"),
    types.VERIFICATION_STARTED: ("subtask_id", "target_tier"),
    types.VERIFICATION_PASSED: ("subtask_id", "tier", "outcome", "reason_code"),
    types.VERIFICATION_FAILED: ("subtask_id", "tier", "outcome", "reason_code"),
    types.STEER_INSTRUCTION: ("instruction_chars",),
    types.TASK_CANCEL_ACK: ("path",),
    types.TASK_CANCEL_TIMEOUT: ("path",),
    types.TELEMETRY_MODE_CHANGED: (
        "configured_mode",
        "runtime_override_mode",
        "effective_mode",
        "scope",
        "actor",
        "source",
    ),
    types.TELEMETRY_SETTINGS_WARNING: (
        "warning_code",
        "input_value",
        "normalized_mode",
        "source",
    ),
    types.TELEMETRY_DIAGNOSTIC: ("diagnostic_type",),
    types.WEBHOOK_DELIVERY_ATTEMPTED: ("delivery_target_host", "attempt", "max_retries"),
    types.WEBHOOK_DELIVERY_SUCCEEDED: ("delivery_target_host", "attempt"),
    types.WEBHOOK_DELIVERY_FAILED: ("delivery_target_host", "attempts"),
    types.WEBHOOK_DELIVERY_DROPPED: ("delivery_target_host", "reason"),
}


_REDACTED_TEXT = "[REDACTED]"
_REDACTION_MARKERS = (
    "auth",
    "secret",
    "token",
    "password",
    "api_key",
    "credential",
    "bearer",
)


def event_family(event_type: str) -> str:
    """Return the configured event family."""
    normalized = str(event_type or "").strip()
    return _EVENT_FAMILIES.get(normalized, "task")


def required_payload_keys(event_type: str) -> tuple[str, ...]:
    """Return merged family + event-specific required payload keys."""
    normalized = str(event_type or "").strip()
    family = event_family(normalized)
    family_required = _FAMILY_REQUIRED_KEYS.get(family, ())
    specific = _EVENT_REQUIRED_KEYS.get(normalized, ())
    return tuple(dict.fromkeys([*family_required, *specific]))


def validate_payload_shape(event_type: str, payload: dict[str, Any]) -> list[str]:
    """Return contract violations for the given payload."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload is not a dict"]
    for key in required_payload_keys(event_type):
        if key not in payload:
            errors.append(f"missing required key: {key}")
            continue
        value = payload.get(key)
        if value is None:
            errors.append(f"required key is null: {key}")
            continue
        if isinstance(value, str) and not value.strip():
            errors.append(f"required key is empty: {key}")
    return errors


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied payload with sensitive values redacted."""
    cloned = deepcopy(payload) if isinstance(payload, dict) else {}
    return _redact_value(cloned, parent_key="")


def _redact_value(value: Any, parent_key: str) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key or "")
            if _should_redact_key(key_text):
                redacted[key_text] = _REDACTED_TEXT
                continue
            if _looks_like_url_key(key_text) and isinstance(item, str):
                redacted[key_text] = _sanitize_url(item)
                continue
            redacted[key_text] = _redact_value(item, parent_key=key_text)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item, parent_key=parent_key) for item in value]
    if _should_redact_key(parent_key) and isinstance(value, str):
        return _REDACTED_TEXT
    return value


def _should_redact_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    if not normalized:
        return False
    if normalized == "token":
        return False
    return any(marker in normalized for marker in _REDACTION_MARKERS)


def _looks_like_url_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    return normalized.endswith("url") or normalized.endswith("uri")


def _sanitize_url(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
