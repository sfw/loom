"""Event type constants for Loom."""

# Task lifecycle events
TASK_CREATED = "task_created"
TASK_PLANNING = "task_planning"
TASK_PLAN_READY = "task_plan_ready"
TASK_EXECUTING = "task_executing"
TASK_REPLANNING = "task_replanning"
TASK_COMPLETED = "task_completed"
TASK_FAILED = "task_failed"
TASK_CANCELLED = "task_cancelled"

# Subtask lifecycle events
SUBTASK_STARTED = "subtask_started"
SUBTASK_COMPLETED = "subtask_completed"
SUBTASK_FAILED = "subtask_failed"
SUBTASK_BLOCKED = "subtask_blocked"

# Tool events
TOOL_CALL_STARTED = "tool_call_started"
TOOL_CALL_COMPLETED = "tool_call_completed"

# Model events
MODEL_INVOCATION = "model_invocation"

# Verification events
VERIFICATION_STARTED = "verification_started"
VERIFICATION_PASSED = "verification_passed"
VERIFICATION_FAILED = "verification_failed"

# Human interaction events
APPROVAL_REQUESTED = "approval_requested"
APPROVAL_RECEIVED = "approval_received"
STEER_INSTRUCTION = "steer_instruction"

# Retry events
SUBTASK_RETRYING = "subtask_retrying"
