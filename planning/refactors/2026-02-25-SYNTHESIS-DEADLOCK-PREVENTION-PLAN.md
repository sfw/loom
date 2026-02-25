# Synthesis Deadlock Prevention Plan (2026-02-25)

## Objective
Eliminate scheduler deadlocks caused by invalid `is_synthesis` usage while preserving run quality, especially the preparation of inputs needed for final outputs.

## Failure Class Being Addressed
Observed run: `cowork-52700a58`.

Key facts:
1. No replanning occurred (`task_replanning` events: 0).
2. Run ended `failed` with `completed=6`, `total=11`, `failed_subtasks=[]`.
3. Remaining plan had 5 pending subtasks, but none runnable.

Root-cause shape:
1. `synthesize-opportunity-scores` was marked `is_synthesis: true`.
2. Scheduler blocks synthesis subtasks until all non-synthesis subtasks are complete.
3. A non-synthesis subtask (`validate-liquidity-risk`) depended on a synthesis-gated path (`construct-concentrated-portfolio` -> `synthesize-opportunity-scores`).
4. This creates a deterministic deadlock with pending work and zero runnable subtasks.

Relevant code paths:
- `/Users/sfw/Development/loom/src/loom/engine/scheduler.py`
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/prompts/templates/planner.yaml`

## Design Principles
1. `is_synthesis` must mean terminal integration, not "important" or "late-stage".
2. Preparation quality is enforced via explicit dependencies and verification, not scheduler side effects.
3. Plan-shape defects must be caught before execution.
4. Deadlocks must be recoverable and observable, not silent.
5. Determinism over heuristics: apply stable graph rules and bounded retries.

## Non-Goals
1. Replacing the dependency scheduler with a new execution model.
2. Introducing fuzzy subtask remapping.
3. Relaxing acceptance criteria to improve completion metrics.

## Target End State
1. Invalid synthesis topology is rejected or normalized before execution.
2. Scheduler cannot deadlock solely due to synthesis gating.
3. Execution loop emits explicit stalled diagnostics and attempts recovery.
4. Final synthesis subtasks still consume prepared upstream outputs with verifier-backed checks.

## Scope
Core runtime:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/engine/scheduler.py`
- `/Users/sfw/Development/loom/src/loom/events/types.py`

Planning contract:
- `/Users/sfw/Development/loom/src/loom/prompts/templates/planner.yaml`
- `/Users/sfw/Development/loom/src/loom/prompts/templates/replanner.yaml`

Verification and UX transparency:
- `/Users/sfw/Development/loom/src/loom/engine/verification.py`
- `/Users/sfw/Development/loom/src/loom/tui/app.py`

Tests:
- `/Users/sfw/Development/loom/tests/test_scheduler.py`
- `/Users/sfw/Development/loom/tests/test_orchestrator.py`
- `/Users/sfw/Development/loom/tests/test_prompts.py`
- `/Users/sfw/Development/loom/tests/test_verification.py`

## Design

### D1: Plan Graph Validation + Normalization at Ingest
Add a plan graph validation step after planning and after replanning, before assigning/continuing execution.

Rules:
1. Dependencies must reference existing subtask IDs.
2. Dependency cycles are invalid.
3. Any subtask with `is_synthesis=true` must be a terminal sink (no dependents).
4. No non-synthesis subtask may depend on a synthesis subtask.

Policy by mode:
1. `strict`: reject invalid plan and retry planner/replanner (bounded).
2. `guided`/`suggestive`: normalize by demoting invalid `is_synthesis` flags to `false` and emit a normalization event.

Notes:
1. Normalization changes only flags, not IDs/dependencies/criteria.
2. This preserves preparation flow and avoids altering substantive plan intent.

### D2: Scheduler Hardening (Safety Net)
Update synthesis gating logic to apply only to synthesis sink nodes.

Current behavior is global for all synthesis nodes. New behavior:
1. If `is_synthesis=true` and subtask is terminal sink, enforce global "all non-synthesis complete" gate.
2. Otherwise, evaluate only declared `depends_on`.

This remains a defense-in-depth fallback; D1 should already prevent invalid synthesis placements.

### D3: Explicit Deadlock Detection and Recovery
In orchestrator loop, when `has_pending(plan)` is true and `runnable_subtasks(plan)` is empty:
1. Emit `task_stalled` with blocked reasons per pending subtask.
2. Attempt one in-memory graph normalization pass (idempotent).
3. If still blocked, request replan with reason `scheduler_deadlock`.
4. If recovery fails, fail task with explicit `blocked_subtasks` payload.

Outcome:
1. No more opaque `task_failed` with incomplete counts and no failed subtasks context.

### D4: Planner/Replanner Contract Tightening
Update planner constraints so model behavior aligns with runtime semantics:
1. `is_synthesis=true` only for terminal integration/reporting steps.
2. Steps needed by downstream execution must not be marked synthesis.

Apply same wording to replanner template to avoid regression during recovery.

### D5: Quality Guardrails for Final Output Preparation
Address quality concern directly by adding synthesis input integrity checks:
1. For terminal synthesis subtasks, verifier checks output references or incorporates required upstream deliverables/metrics.
2. If upstream preparation exists but is absent from synthesis output, fail with actionable feedback.

This ensures deadlock fixes do not reduce preparation quality.

## Eventing and Observability
Add event types:
1. `task_plan_normalized`
2. `task_stalled`
3. `task_stalled_recovery_attempted`

Payload recommendations:
1. `plan_version`, `normalized_subtasks`, `reason`
2. `pending_subtasks`, `blocked_reasons`, `runnable_count`
3. `recovery_mode` (`normalize|replan`), `recovery_success`

TUI behavior:
1. Surface concise stalled reason in activity stream.
2. Keep status understandable without opening raw event logs.

## Implementation Workstreams

### W1: Plan-Shape Validator and Normalizer
Files:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/engine/scheduler.py`

Tasks:
1. Add helper to compute dependents/out-degree and validate synthesis rules.
2. Add normalization path for non-strict modes.
3. Wire into initial planning and replanning flows.

### W2: Scheduler Gate Refinement
Files:
- `/Users/sfw/Development/loom/src/loom/engine/scheduler.py`

Tasks:
1. Gate only sink synthesis nodes with global non-synthesis completion rule.
2. Keep dependency check behavior unchanged for all other nodes.

### W3: Deadlock Recovery Path
Files:
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/events/types.py`

Tasks:
1. Add blocked-reason computation.
2. Emit stalled/recovery telemetry.
3. Add bounded recovery flow before final failure.

### W4: Planner Contract Update
Files:
- `/Users/sfw/Development/loom/src/loom/prompts/templates/planner.yaml`
- `/Users/sfw/Development/loom/src/loom/prompts/templates/replanner.yaml`

Tasks:
1. Tighten `is_synthesis` guidance to terminal-only semantics.
2. Keep wording explicit and machine-checkable.

### W5: Synthesis Input Integrity Verification
Files:
- `/Users/sfw/Development/loom/src/loom/engine/verification.py`

Tasks:
1. Add/extend check for terminal synthesis coverage of upstream outputs.
2. Produce concise actionable failure reasons for missing upstream integration.

### W6: TUI Transparency
Files:
- `/Users/sfw/Development/loom/src/loom/tui/app.py`

Tasks:
1. Render stalled/recovery events in activity pane.
2. Avoid noisy duplication with existing progress snapshot updates.

## Test Plan

### Scheduler Unit Tests
1. `test_synthesis_sink_global_gate_applies`.
2. `test_non_sink_synthesis_does_not_trigger_global_gate`.
3. `test_blocked_reason_reports_unmet_dependency_and_synthesis_gate`.

### Orchestrator Unit/Integration Tests
1. `test_invalid_synthesis_topology_normalized_in_guided_mode`.
2. `test_invalid_synthesis_topology_rejected_in_strict_mode`.
3. `test_stalled_loop_emits_task_stalled_and_attempts_recovery`.
4. `test_deadlock_recovery_replans_and_continues_when_possible`.
5. `test_deadlock_failure_reports_blocked_subtasks_when_unrecoverable`.

### Prompt Contract Tests
1. Planner template includes terminal-only synthesis instruction.
2. Replanner template includes same instruction.

### Verification Tests
1. Terminal synthesis fails when required upstream outputs are omitted.
2. Terminal synthesis passes when upstream preparation is integrated.

## Rollout Strategy

### Phase 1: Safety + Diagnostics
1. Implement D1, D2, D3 with tests.
2. Add events and TUI surfacing.
3. Ship behind config flag if needed for quick rollback.

### Phase 2: Prompt Contract
1. Update planner/replanner templates.
2. Confirm reduced invalid synthesis topology frequency in run logs.

### Phase 3: Quality Guardrails
1. Add synthesis input integrity verifier checks.
2. Track pass/fail and false-positive rate for one release cycle.

## Success Metrics
1. Zero runs failing with `completed < total` and `failed_subtasks=[]` due to scheduler deadlock.
2. Reduction in `task_stalled` events that remain unrecovered.
3. No degradation in final synthesis verification pass rate.
4. Stable or improved rerun completion rate for process-heavy goals.

## Risks and Mitigations
1. Risk: Over-normalizing synthesis flags may hide planner quality issues.
   Mitigation: emit `task_plan_normalized` with details and track frequency.
2. Risk: New verifier checks could be noisy.
   Mitigation: start with narrow terminal-synthesis scope and add golden tests.
3. Risk: Replan-on-stall could increase latency.
   Mitigation: perform one cheap normalization pass before replanning.

## Open Decisions
1. Should strict mode reject immediately or allow one auto-normalization pass with explicit warning?
2. Should `task_stalled` include full dependency graph snapshot or just blocked reasons?
3. Should synthesis integrity checks be always-on or process-configurable initially?
