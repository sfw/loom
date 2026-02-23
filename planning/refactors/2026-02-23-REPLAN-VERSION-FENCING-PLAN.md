# Replan Version Fencing Plan (2026-02-23)

## Objective
Prevent stale subtask updates from crashing runs after replanning, while preserving agile replanning.

This plan enforces deterministic orchestration rules:
1. Replans happen only at batch boundaries.
2. Outcome application is version-fenced.
3. Replan outputs must preserve unfinished subtask IDs exactly.
4. No ID remapping, no heuristic/static matching, no LLM-assisted reconciliation.

## Non-Negotiable Constraints
1. No subtask remap layer.
2. No alias map.
3. No fuzzy/static/semantic comparison of old vs new subtasks.
4. Identity continuity is exact string equality on subtask IDs.
5. Any stale outcome must be non-fatal.

## Failure Class Being Addressed
Observed failure sequence:
1. A subtask failure triggers replan.
2. Replan replaces the task plan.
3. Another in-flight outcome attempts to update an ID that no longer exists.
4. `Task.update_subtask(...)` raises `ValueError: Subtask not found`.
5. Orchestrator converts this into fatal task failure.

Relevant code paths:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py:223`
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py:577`
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py:1827`
- `/Users/sfw/Development/loom/src/loom/state/task_state.py:122`

## Target End State
1. Replan cannot occur mid-outcome-processing for a batch.
2. Every subtask result carries dispatch plan version metadata.
3. Outcome writes are ignored when dispatch version is stale.
4. Replan is rejected when unfinished IDs are removed or renamed.
5. Orchestrator never fatally fails because a stale subtask ID is missing.

## Scope
Core runtime:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/state/task_state.py`
- `/Users/sfw/Development/loom/src/loom/events/types.py`

Tests:
- `/Users/sfw/Development/loom/tests/test_orchestrator.py`
- `/Users/sfw/Development/loom/tests/test_task_state.py`

## Design

### D1: Batch-Boundary Replan Scheduling
Current behavior may replan during per-outcome processing.

Refactor:
1. `_handle_failure(...)` returns an action object instead of calling `_replan_task(...)` directly.
2. Actions include:
- `none`
- `retry`
- `abort`
- `request_replan` with reason payload
3. Execute at most one replan after all outcomes in current batch are processed.
4. If multiple `request_replan` actions exist in one batch, choose deterministic winner:
- first by batch order
- tie-break by subtask ID lexical order

Result:
- No plan mutation while processing the same batch results.

### D2: Version-Fenced Outcome Application
Refactor outcome envelopes to include:
- `dispatch_plan_version`
- `subtask_id`
- result + verification

Rules:
1. Before mutating state for an outcome, check:
- `task.plan.version == dispatch_plan_version`
- `task.get_subtask(subtask_id) is not None`
2. If either check fails:
- emit `stale_outcome_ignored` event
- record a non-fatal decision/error note
- skip mutation and continue

Result:
- Old outcomes can never crash the run.

### D3: Strict Replan Contract Validator (No Remap)
Add deterministic validation in `_replan_task(...)` before assigning `task.plan = new_plan`.

Inputs:
- old plan
- new plan
- old subtask statuses

Rules:
1. Let `required_ids = {old subtask IDs where status != completed}`.
2. Let `new_ids = {new plan subtask IDs}`.
3. Require `required_ids` subset of `new_ids`.
4. Require all IDs in `new_plan` unique.
5. No translation step is allowed.

If invalid:
1. reject replan
2. emit `task_replan_rejected` with missing IDs
3. retry replanner prompt (bounded attempts)
4. if still invalid, keep existing plan unchanged and continue with fallback behavior

Result:
- Replanning stays agile, but live identity is immutable.

### D4: Safe Subtask Update in Orchestrator Path
Keep strict `Task.update_subtask(...)` for generic callers.

For orchestrator:
1. introduce helper: `_safe_update_subtask(task, subtask_id, **updates) -> bool`
2. helper does `get_subtask` check first
3. if missing, emit stale event and return `False`
4. orchestrator paths use helper where stale IDs may appear

Result:
- strict API remains available
- orchestration path becomes crash-resistant

### D5: Eventing and Observability
Add events:
1. `SUBTASK_OUTCOME_STALE`
2. `TASK_REPLAN_REJECTED`
3. optional: `TASK_REPLAN_DEFERRED` (for batch-boundary traceability)

Suggested payload fields:
- `subtask_id`
- `dispatch_plan_version`
- `current_plan_version`
- `reason` (`version_mismatch|missing_subtask`)
- `missing_ids` (for replan rejection)

## Implementation Workstreams

### W1: Action-Driven Failure Handling
Files:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`

Tasks:
1. Define `FailureAction` dataclass/typed dict.
2. Change `_handle_failure` signature to return `FailureAction`.
3. In execute loop, collect actions per batch.
4. Run replan once at batch boundary based on collected action.

### W2: Outcome Envelope + Version Fence
Files:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`

Tasks:
1. Update dispatch return shape to include `dispatch_plan_version`.
2. Add stale guard in outcome processing.
3. Route stale path to event + continue.

### W3: Replan Contract Validator
Files:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`

Tasks:
1. Add `_validate_replan_contract(old_plan, new_plan)`.
2. Add bounded semantic retry loop for invalid replans (validation-only retries).
3. Reject invalid replans without mutating active plan.

### W4: Safe Update Helper
Files:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/state/task_state.py` (if helper centralized there)

Tasks:
1. Add orchestrator-scoped safe update helper.
2. Replace vulnerable `task.update_subtask(...)` usage in stale-prone paths.
3. Preserve strict exception behavior in base Task method.

### W5: Events and TUI Surfacing
Files:
- `/Users/sfw/Development/loom/src/loom/events/types.py`
- `/Users/sfw/Development/loom/src/loom/tui/app.py`

Tasks:
1. Add new event constants.
2. Emit new events from orchestrator.
3. Add concise TUI status text for stale/replan-rejected events.

## Test Plan

### Unit Tests
1. `test_replan_deferred_to_batch_boundary`
- two outcomes in one batch, first requests replan
- assert replan called after outcomes loop, not inline

2. `test_stale_outcome_is_ignored_nonfatal`
- dispatch with plan version N
- mutate plan to N+1 before applying outcome
- assert no exception, task continues

3. `test_missing_subtask_outcome_is_ignored_nonfatal`
- outcome references removed/missing subtask ID
- assert stale event emitted and no fatal failure

4. `test_replan_contract_rejects_removed_unfinished_ids`
- new plan missing unfinished old IDs
- assert rejection + no plan swap

5. `test_replan_contract_accepts_superset_ids`
- new plan contains all unfinished IDs plus new subtasks
- assert accepted

6. `test_no_remap_attempt_occurs`
- ensure code path does not construct alias/remap structures

### Integration/Behavior Tests
1. `test_parallel_failure_replan_does_not_raise_subtask_not_found`
- reproduce old race structure
- assert no `ValueError` fatal

2. `test_task_survives_invalid_replan_then_continues_existing_plan`
- invalid replans exhausted
- existing plan remains active and runnable

### Regression Guard
Add assertion in tests that fatal `TASK_FAILED` with `error_type=ValueError` and message `Subtask not found` is not produced by stale outcomes.

## Rollout
1. Implement behind feature flag:
- `execution.strict_replan_version_fencing` (default false initially)
2. Run CI with flag on in dedicated test matrix.
3. Flip default to true after green soak.
4. Remove flag after stabilization.

## Risk and Mitigation
1. Risk: deferred replan slightly delays corrective planning.
- Mitigation: only one batch latency; maintain max parallel cap.

2. Risk: strict ID continuity can reject useful replans.
- Mitigation: replanner prompt updates to preserve IDs while allowing added subtasks.

3. Risk: stale outcomes may hide useful work.
- Mitigation: explicit stale events + counters for observability.

## Acceptance Criteria
1. No run fails due to `ValueError: Subtask not found` from stale outcomes.
2. Replans never mutate plan mid-batch.
3. Invalid replans that drop unfinished IDs are rejected deterministically.
4. No remapping/aliasing code exists in orchestrator.
5. Existing retry/replan behavior remains functional for valid plans.

## PR Sequence
1. PR1: failure-action refactor + batch-boundary replan
2. PR2: version-fenced outcomes + stale eventing
3. PR3: strict replan contract validation + invalid-replan retry loop
4. PR4: safe update helper adoption + final regression tests

