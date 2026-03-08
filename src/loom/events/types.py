"""Typed telemetry event names and lifecycle metadata for Loom."""

from __future__ import annotations

from typing import Literal

EventLifecycle = Literal["active", "deprecated", "internal_only"]
TelemetryMode = Literal["off", "active", "all_typed", "debug"]

# Task lifecycle events
TASK_CREATED = "task_created"
TASK_PLANNING = "task_planning"
TASK_PLAN_READY = "task_plan_ready"
TASK_PLAN_NORMALIZED = "task_plan_normalized"
TASK_EXECUTING = "task_executing"
TASK_REPLANNING = "task_replanning"
TASK_REPLAN_REJECTED = "task_replan_rejected"
TASK_STALLED = "task_stalled"
TASK_STALLED_RECOVERY_ATTEMPTED = "task_stalled_recovery_attempted"
TASK_BUDGET_EXHAUSTED = "task_budget_exhausted"
TASK_PLAN_DEGRADED = "task_plan_degraded"
TASK_COMPLETED = "task_completed"
TASK_FAILED = "task_failed"
TASK_CANCELLED = "task_cancelled"

# Control-plane lifecycle events
TASK_PAUSED = "task_paused"
TASK_RESUMED = "task_resumed"
TASK_INJECTED = "task_injected"
TASK_CANCEL_REQUESTED = "task_cancel_requested"
TASK_CANCEL_ACK = "task_cancel_ack"
TASK_CANCEL_TIMEOUT = "task_cancel_timeout"

# Subtask lifecycle events
SUBTASK_STARTED = "subtask_started"
SUBTASK_COMPLETED = "subtask_completed"
SUBTASK_FAILED = "subtask_failed"
SUBTASK_OUTCOME_STALE = "subtask_outcome_stale"
SUBTASK_BLOCKED = "subtask_blocked"
SUBTASK_OUTPUT_CONFLICT_DEFERRED = "subtask_output_conflict_deferred"
SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING = "subtask_output_conflict_starvation_warning"
SUBTASK_POLICY_RECONCILED = "subtask_policy_reconciled"

# Tool events
TOOL_CALL_STARTED = "tool_call_started"
TOOL_CALL_COMPLETED = "tool_call_completed"
TOOL_CALL_DEDUPLICATED = "tool_call_deduplicated"
FORBIDDEN_CANONICAL_WRITE_BLOCKED = "forbidden_canonical_write_blocked"
SEALED_POLICY_PREFLIGHT_BLOCKED = "sealed_policy_preflight_blocked"
SEALED_RESEAL_APPLIED = "sealed_reseal_applied"
SEALED_UNEXPECTED_MUTATION_DETECTED = "sealed_unexpected_mutation_detected"
ARTIFACT_CONFINEMENT_VIOLATION = "artifact_confinement_violation"
ARTIFACT_INGEST_CLASSIFIED = "artifact_ingest_classified"
ARTIFACT_INGEST_COMPLETED = "artifact_ingest_completed"
ARTIFACT_RETENTION_PRUNED = "artifact_retention_pruned"
ARTIFACT_READ_COMPLETED = "artifact_read_completed"
COMPACTION_POLICY_DECISION = "compaction_policy_decision"
OVERFLOW_FALLBACK_APPLIED = "overflow_fallback_applied"
TELEMETRY_RUN_SUMMARY = "telemetry_run_summary"
RUN_VALIDITY_SCORECARD = "run_validity_scorecard"

# Model events
MODEL_INVOCATION = "model_invocation"

# Verification events
VERIFICATION_STARTED = "verification_started"
VERIFICATION_PASSED = "verification_passed"
VERIFICATION_FAILED = "verification_failed"
VERIFICATION_RULE_APPLIED = "verification_rule_applied"
VERIFICATION_RULE_SKIPPED = "verification_rule_skipped"
VERIFICATION_OUTCOME = "verification_outcome"
VERIFICATION_SHADOW_DIFF = "verification_shadow_diff"
VERIFICATION_FALSE_NEGATIVE_CANDIDATE = "verification_false_negative_candidate"
VERIFICATION_INCONCLUSIVE_RATE = "verification_inconclusive_rate"
VERIFICATION_RULE_FAILURE_BY_TYPE = "rule_failure_by_type"
VERIFICATION_DETERMINISTIC_BLOCK_RATE = "deterministic_block_rate"
VERIFICATION_CONTRADICTION_DETECTED = "verification_contradiction_detected"
CLAIM_VERIFICATION_SUMMARY = "claim_verification_summary"
CLAIMS_PRUNED = "claims_pruned"
SYNTHESIS_INPUT_GATE_DECISION = "synthesis_input_gate_decision"
ARTIFACT_SEAL_VALIDATION = "artifact_seal_validation"

# Human interaction events
APPROVAL_REQUESTED = "approval_requested"
APPROVAL_RECEIVED = "approval_received"
ASK_USER_REQUESTED = "ask_user_requested"
ASK_USER_ANSWERED = "ask_user_answered"
ASK_USER_TIMEOUT = "ask_user_timeout"
ASK_USER_CANCELLED = "ask_user_cancelled"
STEER_INSTRUCTION = "steer_instruction"

# Retry and remediation events
SUBTASK_RETRYING = "subtask_retrying"
REMEDIATION_QUEUED = "remediation_queued"
REMEDIATION_STARTED = "remediation_started"
REMEDIATION_ATTEMPT = "remediation_attempt"
REMEDIATION_RESOLVED = "remediation_resolved"
REMEDIATION_FAILED = "remediation_failed"
REMEDIATION_EXPIRED = "remediation_expired"
REMEDIATION_TERMINAL = "remediation_terminal"
UNCONFIRMED_DATA_QUEUED = "unconfirmed_data_queued"
PLACEHOLDER_FINDINGS_EXTRACTED = "placeholder_findings_extracted"
PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED = "placeholder_confirm_or_prune_started"
PLACEHOLDER_FILLED = "placeholder_filled"
PLACEHOLDER_PRUNED = "placeholder_pruned"
PLACEHOLDER_REMEDIATION_UNRESOLVED = "placeholder_remediation_unresolved"

# Iteration loop events
ITERATION_STARTED = "iteration_started"
ITERATION_GATE_FAILED = "iteration_gate_failed"
ITERATION_RETRYING = "iteration_retrying"
ITERATION_COMPLETED = "iteration_completed"
ITERATION_TERMINAL = "iteration_terminal"
ITERATION_STATE_RECONCILED = "iteration_state_reconciled"

# Durable run lifecycle
TASK_RUN_ACQUIRED = "task_run_acquired"
TASK_RUN_HEARTBEAT = "task_run_heartbeat"
TASK_RUN_RECOVERED = "task_run_recovered"

# Database migration diagnostics
DB_MIGRATION_START = "db_migration_start"
DB_MIGRATION_APPLIED = "db_migration_applied"
DB_MIGRATION_VERIFY_FAILED = "db_migration_verify_failed"
DB_MIGRATION_FAILED = "db_migration_failed"
DB_SCHEMA_READY = "db_schema_ready"

# Telemetry runtime/settings events
TELEMETRY_MODE_CHANGED = "telemetry_mode_changed"
TELEMETRY_SETTINGS_WARNING = "telemetry_settings_warning"
TELEMETRY_DIAGNOSTIC = "telemetry_diagnostic"

# Streaming events
TOKEN_STREAMED = "token_streamed"

# Conversation events
CONVERSATION_MESSAGE = "conversation_message"

# Webhook delivery events
WEBHOOK_DELIVERY_ATTEMPTED = "webhook_delivery_attempted"
WEBHOOK_DELIVERY_SUCCEEDED = "webhook_delivery_succeeded"
WEBHOOK_DELIVERY_FAILED = "webhook_delivery_failed"
WEBHOOK_DELIVERY_DROPPED = "webhook_delivery_dropped"


def _event_constants() -> dict[str, str]:
    constants: dict[str, str] = {}
    for name, value in globals().items():
        if not name.isupper():
            continue
        if not isinstance(value, str):
            continue
        if name.startswith("_"):
            continue
        constants[name] = value
    return constants


EVENT_NAME_TO_TYPE = _event_constants()


EVENT_LIFECYCLE: dict[str, EventLifecycle] = {
    # Task lifecycle
    TASK_CREATED: "active",
    TASK_PLANNING: "active",
    TASK_PLAN_READY: "active",
    TASK_PLAN_NORMALIZED: "active",
    TASK_EXECUTING: "active",
    TASK_REPLANNING: "active",
    TASK_REPLAN_REJECTED: "active",
    TASK_STALLED: "active",
    TASK_STALLED_RECOVERY_ATTEMPTED: "active",
    TASK_BUDGET_EXHAUSTED: "active",
    TASK_PLAN_DEGRADED: "active",
    TASK_COMPLETED: "active",
    TASK_FAILED: "active",
    TASK_CANCELLED: "active",
    TASK_PAUSED: "active",
    TASK_RESUMED: "active",
    TASK_INJECTED: "active",
    TASK_CANCEL_REQUESTED: "active",
    TASK_CANCEL_ACK: "active",
    TASK_CANCEL_TIMEOUT: "active",
    # Subtask lifecycle
    SUBTASK_STARTED: "active",
    SUBTASK_COMPLETED: "active",
    SUBTASK_FAILED: "active",
    SUBTASK_OUTCOME_STALE: "active",
    SUBTASK_BLOCKED: "active",
    SUBTASK_OUTPUT_CONFLICT_DEFERRED: "active",
    SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING: "active",
    SUBTASK_POLICY_RECONCILED: "active",
    # Tool events
    TOOL_CALL_STARTED: "active",
    TOOL_CALL_COMPLETED: "active",
    TOOL_CALL_DEDUPLICATED: "active",
    FORBIDDEN_CANONICAL_WRITE_BLOCKED: "active",
    SEALED_POLICY_PREFLIGHT_BLOCKED: "active",
    SEALED_RESEAL_APPLIED: "active",
    SEALED_UNEXPECTED_MUTATION_DETECTED: "active",
    ARTIFACT_CONFINEMENT_VIOLATION: "active",
    ARTIFACT_INGEST_CLASSIFIED: "active",
    ARTIFACT_INGEST_COMPLETED: "active",
    ARTIFACT_RETENTION_PRUNED: "active",
    ARTIFACT_READ_COMPLETED: "active",
    COMPACTION_POLICY_DECISION: "active",
    OVERFLOW_FALLBACK_APPLIED: "active",
    TELEMETRY_RUN_SUMMARY: "active",
    RUN_VALIDITY_SCORECARD: "active",
    # Model events
    MODEL_INVOCATION: "internal_only",
    # Verification events
    VERIFICATION_STARTED: "active",
    VERIFICATION_PASSED: "active",
    VERIFICATION_FAILED: "active",
    VERIFICATION_RULE_APPLIED: "active",
    VERIFICATION_RULE_SKIPPED: "active",
    VERIFICATION_OUTCOME: "active",
    VERIFICATION_SHADOW_DIFF: "internal_only",
    VERIFICATION_FALSE_NEGATIVE_CANDIDATE: "internal_only",
    VERIFICATION_INCONCLUSIVE_RATE: "active",
    VERIFICATION_RULE_FAILURE_BY_TYPE: "active",
    VERIFICATION_DETERMINISTIC_BLOCK_RATE: "active",
    VERIFICATION_CONTRADICTION_DETECTED: "active",
    CLAIM_VERIFICATION_SUMMARY: "active",
    CLAIMS_PRUNED: "active",
    SYNTHESIS_INPUT_GATE_DECISION: "active",
    ARTIFACT_SEAL_VALIDATION: "active",
    # Human events
    APPROVAL_REQUESTED: "active",
    APPROVAL_RECEIVED: "active",
    ASK_USER_REQUESTED: "active",
    ASK_USER_ANSWERED: "active",
    ASK_USER_TIMEOUT: "active",
    ASK_USER_CANCELLED: "active",
    STEER_INSTRUCTION: "active",
    # Retry / remediation events
    SUBTASK_RETRYING: "active",
    REMEDIATION_QUEUED: "active",
    REMEDIATION_STARTED: "active",
    REMEDIATION_ATTEMPT: "active",
    REMEDIATION_RESOLVED: "active",
    REMEDIATION_FAILED: "active",
    REMEDIATION_EXPIRED: "active",
    REMEDIATION_TERMINAL: "active",
    UNCONFIRMED_DATA_QUEUED: "active",
    PLACEHOLDER_FINDINGS_EXTRACTED: "active",
    PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED: "active",
    PLACEHOLDER_FILLED: "active",
    PLACEHOLDER_PRUNED: "active",
    PLACEHOLDER_REMEDIATION_UNRESOLVED: "active",
    # Iteration events
    ITERATION_STARTED: "active",
    ITERATION_GATE_FAILED: "active",
    ITERATION_RETRYING: "active",
    ITERATION_COMPLETED: "active",
    ITERATION_TERMINAL: "active",
    ITERATION_STATE_RECONCILED: "active",
    # Durable run lifecycle
    TASK_RUN_ACQUIRED: "active",
    TASK_RUN_HEARTBEAT: "active",
    TASK_RUN_RECOVERED: "active",
    # Database migration diagnostics
    DB_MIGRATION_START: "internal_only",
    DB_MIGRATION_APPLIED: "internal_only",
    DB_MIGRATION_VERIFY_FAILED: "internal_only",
    DB_MIGRATION_FAILED: "internal_only",
    DB_SCHEMA_READY: "internal_only",
    # Telemetry runtime/settings
    TELEMETRY_MODE_CHANGED: "active",
    TELEMETRY_SETTINGS_WARNING: "active",
    TELEMETRY_DIAGNOSTIC: "internal_only",
    # Streaming / conversation
    TOKEN_STREAMED: "internal_only",
    CONVERSATION_MESSAGE: "internal_only",
    # Webhook delivery
    WEBHOOK_DELIVERY_ATTEMPTED: "active",
    WEBHOOK_DELIVERY_SUCCEEDED: "active",
    WEBHOOK_DELIVERY_FAILED: "active",
    WEBHOOK_DELIVERY_DROPPED: "active",
}


def _validate_catalog() -> None:
    declared_values = set(EVENT_NAME_TO_TYPE.values())
    catalog_values = set(EVENT_LIFECYCLE.keys())
    missing = sorted(declared_values - catalog_values)
    extra = sorted(catalog_values - declared_values)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing lifecycle entries: {missing}")
        if extra:
            details.append(f"unknown lifecycle entries: {extra}")
        raise RuntimeError("Invalid event catalog: " + "; ".join(details))


_validate_catalog()


ALL_EVENT_TYPES = frozenset(EVENT_LIFECYCLE.keys())
ACTIVE_EVENT_TYPES = frozenset(
    event_type
    for event_type, lifecycle in EVENT_LIFECYCLE.items()
    if lifecycle == "active"
)
DEPRECATED_EVENT_TYPES = frozenset(
    event_type
    for event_type, lifecycle in EVENT_LIFECYCLE.items()
    if lifecycle == "deprecated"
)
INTERNAL_ONLY_EVENT_TYPES = frozenset(
    event_type
    for event_type, lifecycle in EVENT_LIFECYCLE.items()
    if lifecycle == "internal_only"
)


def event_lifecycle(event_type: str) -> EventLifecycle | None:
    """Return lifecycle for an event type, or ``None`` if unknown."""
    return EVENT_LIFECYCLE.get(str(event_type or "").strip())


def is_known_event_type(event_type: str) -> bool:
    return str(event_type or "").strip() in ALL_EVENT_TYPES


def is_active_event_type(event_type: str) -> bool:
    return str(event_type or "").strip() in ACTIVE_EVENT_TYPES
