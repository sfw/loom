# Spec 14: Human-in-the-Loop

## Overview

Loom uses a confidence-based interruption model rather than binary approve/reject at permission boundaries. The system determines when to proceed automatically, when to pause for review, and when to hard-stop for approval, based on verification confidence scores, task criticality, and learned patterns.

## Confidence-Based Interruption Model

```
┌─────────────────────────────────────────────────────────┐
│ HIGH CONFIDENCE (≥ 0.8)                                 │
│ Deterministic validation passes, output matches schema, │
│ matches prior similar tasks.                            │
│ Action: Proceed automatically. Log for audit.           │
├─────────────────────────────────────────────────────────┤
│ MEDIUM CONFIDENCE (0.5 - 0.8)                           │
│ Validation passes but output seems unusual, or it's a   │
│ new task type without prior history.                     │
│ Action: Show summary in TUI. Auto-proceed after 10s     │
│ unless human intervenes.                                │
├─────────────────────────────────────────────────────────┤
│ LOW CONFIDENCE (0.2 - 0.5)                              │
│ Verification failed, model expressed uncertainty, or     │
│ operation is destructive/critical.                       │
│ Action: Hard stop. Require explicit approval.            │
├─────────────────────────────────────────────────────────┤
│ ZERO CONFIDENCE (< 0.2)                                 │
│ Repeated failures, incoherent output, or model admits    │
│ it cannot complete the task.                             │
│ Action: Abort subtask. Present diagnostics. Wait for     │
│ human decision.                                         │
└─────────────────────────────────────────────────────────┘
```

## Confidence Scoring

```python
class ConfidenceScorer:
    def score(
        self,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
    ) -> float:
        """
        Compute confidence score (0.0 - 1.0) for a completed subtask.
        """
        score = 0.0
        weights_total = 0.0

        # Verification tier 1 passed (weight: 0.3)
        if verification.tier >= 1:
            t1_checks = [c for c in verification.checks if c.passed]
            t1_score = len(t1_checks) / max(len(verification.checks), 1)
            score += t1_score * 0.3
            weights_total += 0.3

        # Verification tier 2 confidence (weight: 0.3)
        if verification.tier >= 2:
            score += verification.confidence * 0.3
            weights_total += 0.3

        # No retries needed (weight: 0.2)
        retry_penalty = min(subtask.retry_count / subtask.max_retries, 1.0)
        score += (1.0 - retry_penalty) * 0.2
        weights_total += 0.2

        # Not a destructive operation (weight: 0.1)
        if not self._is_destructive(result):
            score += 0.1
        weights_total += 0.1

        # All tool calls succeeded (weight: 0.1)
        if result.tool_calls:
            tool_success = sum(1 for tc in result.tool_calls if tc.result.success)
            score += (tool_success / len(result.tool_calls)) * 0.1
        else:
            score += 0.1
        weights_total += 0.1

        return score / weights_total if weights_total > 0 else 0.0

    def _is_destructive(self, result: SubtaskResult) -> bool:
        """Check if the subtask performed destructive operations."""
        destructive_tools = {"shell_execute"}
        destructive_patterns = ["rm ", "drop ", "delete ", "truncate "]
        for tc in result.tool_calls:
            if tc.tool in destructive_tools:
                cmd = tc.args.get("command", "")
                if any(p in cmd.lower() for p in destructive_patterns):
                    return True
        return False
```

## Approval Gates

### Automatic Gating Rules

Some operations always require approval regardless of confidence:

```python
ALWAYS_GATE = [
    "Deleting files",
    "Dropping database tables",
    "Modifying production configuration",
    "Installing system packages",
    "Running destructive shell commands",
    "Modifying .env or secret files",
]
```

### Approval Request

When the engine needs human approval:

```python
@dataclass
class ApprovalRequest:
    task_id: str
    subtask_id: str
    reason: str                      # Why approval is needed
    proposed_action: str             # What the engine wants to do
    risk_level: str                  # "low", "medium", "high", "critical"
    details: dict                    # Supporting info (affected files, command, etc.)
    auto_approve_timeout: int | None # Seconds before auto-approve (None = no auto)
```

The request is:
1. Emitted as an `approval_requested` event
2. Shown in the TUI as a modal prompt
3. Available via `GET /tasks/{id}` (status shows "waiting_approval")
4. Delivered to webhook if registered

### Approval Modes

Set per-task at creation:

| Mode | Behavior |
|------|----------|
| `auto` | Proceed automatically for high confidence. Gate only destructive ops and low confidence. |
| `manual` | Gate every subtask completion. Human must approve each step. |
| `confidence_threshold` | Auto-proceed above threshold (configurable). Gate below. |

```python
class ApprovalManager:
    async def check_approval(
        self,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        confidence: float,
    ) -> ApprovalDecision:
        """
        Determine whether to proceed, wait, or abort.
        """
        if task.approval_mode == "manual":
            return ApprovalDecision.WAIT

        if task.approval_mode == "auto":
            if self._is_always_gated(result):
                return ApprovalDecision.WAIT
            if confidence >= 0.8:
                return ApprovalDecision.PROCEED
            if confidence >= 0.5:
                return ApprovalDecision.WAIT_WITH_TIMEOUT  # Auto-proceed after 10s
            return ApprovalDecision.WAIT

        if task.approval_mode == "confidence_threshold":
            threshold = task.metadata.get("confidence_threshold", 0.8)
            if confidence >= threshold and not self._is_always_gated(result):
                return ApprovalDecision.PROCEED
            return ApprovalDecision.WAIT

    async def wait_for_approval(
        self,
        task_id: str,
        subtask_id: str,
        timeout: int | None = None,
    ) -> bool:
        """
        Block until human approves/rejects, or timeout expires.
        Returns True if approved, False if rejected.
        """
        # Implemented via asyncio.Event set by API endpoint
        ...
```

## Steering

Humans can inject instructions into a running task at any time:

```python
# Via TUI: press 's', type instruction
# Via API: PATCH /tasks/{id} {"instruction": "Use PostgreSQL instead"}

async def inject_steering(self, task_id: str, instruction: str):
    """
    Inject a mid-task instruction.
    1. Store as user_instruction memory entry
    2. If a subtask is currently running, inject into next model call
    3. Emit steer_instruction event
    """
    await self._memory.store(MemoryEntry(
        task_id=task_id,
        entry_type="user_instruction",
        summary=instruction[:150],
        detail=instruction,
        tags="user,steering",
    ))
    self._event_bus.emit(SteerInstruction(task_id, instruction))
```

## Per-Tool-Call Approval (Cowork Mode)

In addition to the confidence-based model above (used in autonomous task mode), Loom
has a **per-tool-call approval system** for interactive cowork/TUI sessions
(`cowork/approval.py`):

| Category | Tools | Approval |
|----------|-------|----------|
| Auto-approved | `read_file`, `search_files`, `list_directory`, `glob_find`, `ripgrep_search`, `analyze_code`, `ask_user`, `web_search`, `web_fetch`, `task_tracker` | Never prompted |
| Needs approval | `write_file`, `edit_file`, `delete_file`, `move_file`, `shell_execute`, `git_command` | `[y]es / [a]lways / [n]o` |

"Always" remembers per-tool for the session. In the CLI: `terminal_approval_prompt()`.
In the TUI: `ToolApprovalScreen` modal.

## Acceptance Criteria

- [x] Confidence scoring produces values between 0.0 and 1.0
- [x] Destructive operations are detected and flagged
- [x] Auto mode proceeds at high confidence, gates at low
- [x] Manual mode gates every subtask
- [x] Confidence threshold mode respects configured threshold
- [x] Approval requests appear in TUI and are available via API
- [x] Approved subtasks resume execution
- [x] Rejected subtasks are marked failed with reason
- [ ] Timed approvals auto-proceed after timeout
- [x] Steering instructions are injected into subsequent prompts
- [x] Always-gated operations require approval regardless of confidence
- [x] Per-tool-call approval in cowork mode with auto/approve/always/deny
- [x] TUI shows approval modal with [y]/[a]/[n] keybindings
