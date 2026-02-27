"""Event type constants for Loom."""

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

# Subtask lifecycle events
SUBTASK_STARTED = "subtask_started"
SUBTASK_COMPLETED = "subtask_completed"
SUBTASK_FAILED = "subtask_failed"
SUBTASK_OUTCOME_STALE = "subtask_outcome_stale"
SUBTASK_BLOCKED = "subtask_blocked"

# Tool events
TOOL_CALL_STARTED = "tool_call_started"
TOOL_CALL_COMPLETED = "tool_call_completed"
TOOL_CALL_DEDUPLICATED = "tool_call_deduplicated"
ARTIFACT_CONFINEMENT_VIOLATION = "artifact_confinement_violation"
ARTIFACT_INGEST_CLASSIFIED = "artifact_ingest_classified"
ARTIFACT_INGEST_COMPLETED = "artifact_ingest_completed"
ARTIFACT_RETENTION_PRUNED = "artifact_retention_pruned"
ARTIFACT_READ_COMPLETED = "artifact_read_completed"
COMPACTION_POLICY_DECISION = "compaction_policy_decision"
OVERFLOW_FALLBACK_APPLIED = "overflow_fallback_applied"
TELEMETRY_RUN_SUMMARY = "telemetry_run_summary"

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

# Human interaction events
APPROVAL_REQUESTED = "approval_requested"
APPROVAL_RECEIVED = "approval_received"
STEER_INSTRUCTION = "steer_instruction"

# Retry events
SUBTASK_RETRYING = "subtask_retrying"
REMEDIATION_QUEUED = "remediation_queued"
REMEDIATION_STARTED = "remediation_started"
REMEDIATION_ATTEMPT = "remediation_attempt"
REMEDIATION_RESOLVED = "remediation_resolved"
REMEDIATION_FAILED = "remediation_failed"
REMEDIATION_EXPIRED = "remediation_expired"
REMEDIATION_TERMINAL = "remediation_terminal"
UNCONFIRMED_DATA_QUEUED = "unconfirmed_data_queued"

# Durable run lifecycle
TASK_RUN_ACQUIRED = "task_run_acquired"
TASK_RUN_HEARTBEAT = "task_run_heartbeat"
TASK_RUN_RECOVERED = "task_run_recovered"

# Streaming events
TOKEN_STREAMED = "token_streamed"

# Conversation events
CONVERSATION_MESSAGE = "conversation_message"
