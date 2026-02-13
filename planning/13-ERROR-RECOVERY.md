# Spec 13: Error Recovery

## Overview

Recovery capability — not initial correctness — best predicts overall success in agentic systems. Loom implements a structured retry and escalation ladder that handles transient failures, model errors, and fundamental approach failures differently.

## Four Failure Archetypes

Research identifies four recurring failure patterns in local LLM agents. Loom has explicit defenses for each:

| Archetype | Description | Defense |
|-----------|-------------|---------|
| Premature action | Model guesses state instead of inspecting | Grounding step required: planner always starts with a "read/inspect" subtask |
| Over-helpfulness | Model substitutes missing data instead of reporting gaps | Explicit constraint: "If data missing, report gap. Do NOT substitute." |
| Context pollution | Prior outputs contaminate current reasoning | Stateless subtasks with fresh context per execution |
| Fragile execution | Malformed JSON, dropped decisions mid-task | Structured output validation at harness level before accepting |

## Escalation Ladder

When a subtask fails, the system escalates through increasingly expensive recovery strategies:

```
Attempt 1: Execute with assigned model (tier from plan)
  → Tier 1 verification fails?

Attempt 2: Retry SAME model with:
  - Verification feedback injected as context
  - Rephrased prompt emphasizing failure point
  → Still fails?

Attempt 3: Escalate to NEXT TIER model
  - Higher capability model
  - Fresh context (no polluted history from failed attempts)
  - Verification feedback from prior attempts
  → Still fails?

Attempt 4: Escalate to HIGHEST TIER with extended thinking
  - Thinking mode enabled
  - All prior error context provided
  - Extended token budget
  → Still fails?

Attempt 5: FLAG FOR HUMAN REVIEW
  - Park the subtask
  - Continue with independent subtasks if any
  - Present diagnostic info to human via TUI/API
  - Emit approval_requested event
```

## Retry Logic

```python
class RetryManager:
    def __init__(self, model_router: ModelRouter, config: Config):
        self._router = model_router
        self._max_retries = config.max_subtask_retries

    async def execute_with_retry(
        self,
        subtask: Subtask,
        execute_fn: Callable,
        task: Task,
    ) -> SubtaskResult:
        """
        Execute a subtask with retry and escalation.

        execute_fn: async function that runs the subtask at a given tier
        """
        attempts = []
        current_tier = subtask.model_tier

        for attempt in range(self._max_retries + 1):
            # Build retry context from prior attempts
            retry_context = self._build_retry_context(attempts) if attempts else None

            # Execute
            result = await execute_fn(
                subtask=subtask,
                model_tier=current_tier,
                retry_context=retry_context,
            )

            attempts.append(AttemptRecord(
                attempt=attempt + 1,
                tier=current_tier,
                result=result,
                timestamp=datetime.now(),
            ))

            if result.verification.passed:
                return result

            # Decide escalation strategy
            if attempt == 0:
                # First retry: same tier, add feedback
                pass
            elif attempt == 1:
                # Second retry: escalate tier
                current_tier = min(current_tier + 1, 3)
            elif attempt == 2:
                # Third retry: max tier with thinking
                current_tier = 3

        # All retries exhausted — flag for human
        return SubtaskResult(
            status="failed",
            summary=f"Failed after {len(attempts)} attempts. Needs human review.",
            verification=VerificationResult(passed=False, feedback=self._summarize_failures(attempts)),
            attempts=attempts,
        )

    def _build_retry_context(self, attempts: list[AttemptRecord]) -> str:
        """
        Build context from prior failed attempts.
        Includes what went wrong and verification feedback.
        """
        lines = ["PREVIOUS ATTEMPTS (all failed):"]
        for a in attempts:
            lines.append(f"\nAttempt {a.attempt} (model tier {a.tier}):")
            lines.append(f"  Verification feedback: {a.result.verification.feedback}")
            if a.result.error:
                lines.append(f"  Error: {a.result.error}")
        lines.append("\nFix the issues identified above. Take a different approach if needed.")
        return "\n".join(lines)
```

## Structured Output Validation

Before accepting any model response, validate at the harness level:

```python
class OutputValidator:
    def validate_tool_call(self, raw_response: str, available_tools: list[dict]) -> ValidationResult:
        """
        Validate that tool calls in the response are well-formed:
        1. Parse as valid JSON
        2. Tool name exists in available tools
        3. Arguments match expected schema
        4. No hallucinated tool names
        """
        ...

    def validate_json_response(self, raw_response: str, expected_schema: dict = None) -> ValidationResult:
        """
        For responses expected to be JSON (plans, extraction):
        1. Strip markdown fences if present
        2. Parse as valid JSON
        3. Validate against schema if provided
        """
        # Strip common LLM artifacts
        cleaned = raw_response.strip()
        cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                error=f"Invalid JSON: {e}",
                suggestion="Respond with ONLY valid JSON, no markdown or explanation.",
            )

        if expected_schema:
            # jsonschema validation
            ...

        return ValidationResult(valid=True, parsed=parsed)
```

## Blocked Subtask Handling

When a subtask cannot proceed (missing dependencies, unresolvable errors):

```python
async def _handle_blocked_subtask(self, task: Task, subtask: Subtask, reason: str):
    """
    When a subtask is blocked:
    1. Mark it as blocked
    2. Check if any independent subtasks can still proceed
    3. If yes, continue with those
    4. If no, pause task and request human input
    """
    subtask.status = SubtaskStatus.BLOCKED
    subtask.result = SubtaskResult(status="blocked", summary=reason)

    # Find independent subtasks that don't depend on the blocked one
    independent = [
        s for s in task.plan.subtasks
        if s.status == SubtaskStatus.PENDING
        and subtask.id not in s.depends_on
    ]

    if independent:
        self._event_bus.emit(SubtaskBlocked(task.id, subtask.id, reason,
            message=f"Continuing with {len(independent)} independent subtasks"))
        # Orchestrator continues with independent subtasks
    else:
        self._event_bus.emit(ApprovalRequested(task.id, subtask.id,
            reason=f"Subtask blocked: {reason}. No independent work remaining.",
            details={"blocked_subtask": subtask.id, "all_attempts": ...}))
```

## Acceptance Criteria

- [ ] First failure triggers retry with feedback, not immediate escalation
- [ ] Model tier escalates on repeated failures
- [ ] Retry context includes verification feedback from all prior attempts
- [ ] Maximum retry count is respected (configurable per-subtask and globally)
- [ ] Blocked subtasks allow independent work to continue
- [ ] Human review is flagged after all retries exhausted
- [ ] Malformed JSON responses trigger immediate retry with format reminder
- [ ] Invalid tool calls are caught before execution
- [ ] Retry history is logged for debugging and learning
- [ ] Escalation ladder is configurable
