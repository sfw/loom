# Dynamic Utility-Aware Compaction Plan (2026-02-23)

## Objective
Implement a runtime compaction controller that preserves output quality under dynamic workload pressure by combining:
1. Deterministic safety and ordering rules.
2. Dynamic utility scoring based on live execution context.
3. Optional model advisory ranking for ambiguous decisions.

This plan explicitly avoids hard-cap truncation in runner model-context paths and avoids static/offline analysis workflows.

## Why the Current Approach Still Fails Under Load
In heavy drafting runs, context pressure can cross critical thresholds while the highest-value content is still active (for example, drafting the 12 profiles that feed final deliverables). A pressure-only tier decision can then over-target compaction on content that should remain high-fidelity, while repeated overshoot retries consume wall-clock budget.

The issue is not just "more compaction needed"; it is "compaction target selection and aggressiveness are under-informed."

## Design Principles
1. Quality-first for active deliverables: active writing context is protected unless failure would otherwise be certain.
2. Deterministic core policy: hard invariants and ordering remain code-driven and reproducible.
3. Dynamic runtime signals, not static rulesets: adapt from live pressure, task phase, and recent compactor outcomes.
4. Model advisory, not model authority: models rank ambiguous candidates, but policy guardrails make final decisions.
5. No hard truncation in runner model-context compaction path.
6. Anti-thrash by default: bounded retries, cooldowns, and skip reasons are first-class behavior.

## Runtime Context Model
Each candidate context unit (message/tool payload/context summary) gets runtime metadata:
1. `lane`: `instruction`, `active_write`, `source_evidence`, `tool_trace`, `historical_chat`.
2. `age`: turns since last direct use.
3. `reusability`: how recoverable it is from files/tools (`high|medium|low`).
4. `task_relevance`: overlap with current subtask goal and recent assistant intent.
5. `compression_history`: previous attempts, realized ratio, and failure mode.

This metadata is computed online each iteration from live task state and message/tool history.

## Decision Model: Pressure + Utility + Feasibility
### 1) Pressure
Use current token pressure ratio as a trigger, not as the only selector.
1. `normal`: no compaction.
2. `pressure`: compact lower-value lanes first.
3. `critical`: broader actions allowed, but write-lane invariants still apply.

### 2) Utility
Score each candidate by expected harm if compacted:
1. Higher harm for `instruction` and `active_write`.
2. Lower harm for `tool_trace` and stale `historical_chat`.
3. Adjust with recency and direct dependency on current drafting goal.

### 3) Feasibility
Before invoking compactor, check whether requested reduction is realistic based on recent runtime outcomes:
1. If required ratio is implausible, skip direct compaction and choose alternate strategy.
2. Alternate strategy preference: pointer/reference summary over repeated aggressive re-compaction.

## Operation Set (No Hard Cap)
The planner chooses among bounded operations:
1. `keep_raw`: do not compact.
2. `compact_tool_args`: compact tool call arguments first.
3. `compact_tool_output`: compact tool outputs second.
4. `compact_historical`: compact stale narrative context third.
5. `replace_with_pointer`: preserve reference to artifact/path + concise digest.
6. `checkpoint_merge`: incremental historical checkpointing (delta-based), never full re-merge each turn.

## Model Advisory Role (Targeted and Bounded)
Use model advisory only for borderline cases where deterministic scores are close.
1. Trigger: top candidate harm scores within epsilon, or uncertain lane conflict.
2. Scope: rank at most top `K` candidates once per iteration.
3. Output schema: strict JSON ranking with `candidate_id`, `priority`, `rationale_tag`.
4. Enforcement: deterministic guardrails can overrule advisory ranking.

Guardrails that advisory cannot violate:
1. Preserve latest critical instructions and active write context.
2. Compact tool traces before touching protected narrative lanes.
3. Respect per-iteration compactor-call and retry budgets.

## Anti-Thrash Controls
1. Per-iteration compactor call budget.
2. Per-label retry cap with cooldown after repeated overshoot.
3. "No-gain" skip when reduction is below minimum delta.
4. Feasibility gate to avoid impossible targets.
5. Reuse cached compaction for unchanged payload + target + label.
6. Stage-4 merge converted to incremental checkpoint updates (no full historical recompress each turn).

## Write-Lane Protection for Drafting Tasks
When subtask intent is drafting/finalization:
1. Classify current draft buffers and selected profile/evidence inputs as `active_write` or `source_evidence`.
2. Keep these lanes raw whenever possible.
3. If pressure is critical, prefer compacting old tool traces and stale historical chat first.
4. Use pointer summaries for recoverable large sources before compacting active narrative text.

This directly addresses document-generation tasks where quality loss compounds downstream.

## Proposed Implementation Workstreams
### Workstream 1: Runtime Lane Classifier and Candidate Graph
### Files
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`

### Changes
1. Add lane classification (`instruction`, `active_write`, `source_evidence`, `tool_trace`, `historical_chat`).
2. Build candidate records with runtime metadata and stable IDs.
3. Detect drafting/finalization mode from subtask intent + recent actions.

### Acceptance Criteria
1. Candidate set deterministically generated for same runtime state.
2. Draft-related context consistently classified into protected lanes.

### Workstream 2: Utility + Feasibility Planner
### Files
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`

### Changes
1. Add utility scoring function (small, runtime-only signal set).
2. Add feasibility gate using recent realized compression outcomes.
3. Replace strict stage-only ordering with lane-aware operation planning while preserving priority constraints.

### Acceptance Criteria
1. Under pressure, planner picks tool traces before active writing context.
2. Implausible compression requests are skipped with explicit reason.

### Workstream 3: Advisory Ranker Integration
### Files
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`
2. `/Users/sfw/Development/loom/src/loom/config.py`

### Changes
1. Add optional advisory mode flag and limits (`enabled`, `max_candidates`, `epsilon`).
2. Add one bounded advisor invocation in borderline cases.
3. Enforce deterministic invariants after advisor output.

### Acceptance Criteria
1. Advisory is never required for correctness.
2. Runs proceed safely when advisor is disabled/unavailable.

### Workstream 4: Incremental Checkpointing Instead of Full Stage-4 Merge
### Files
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`

### Changes
1. Replace full "prior conversation context" recompression with incremental checkpoint updates.
2. Merge only newly stale context deltas.
3. Preserve recent protected lanes outside checkpoint body.

### Acceptance Criteria
1. No repeated full-history recompress on every iteration.
2. Critical-tier pressure relief still works without eroding active draft context.

### Workstream 5: Compactor Runtime Guardrails
### Files
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`
2. `/Users/sfw/Development/loom/src/loom/engine/semantic_compactor.py`
3. `/Users/sfw/Development/loom/src/loom/config.py`

### Changes
1. Add per-iteration call budgets and per-label cooldown.
2. Add overshoot streak tracking and skip behavior.
3. Emit explicit skip reasons (`unachievable_ratio`, `budget_exhausted`, `cooldown`, `protected_lane`).

### Acceptance Criteria
1. Bounded compactor churn in long drafting runs.
2. Retry storms no longer consume most of remaining subtask time.

### Workstream 6: Memory Extraction Alignment
### Files
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`

### Changes
1. Apply same lane/utility principles to extractor prompt formation.
2. Prefer compact references for large recovered artifacts.
3. Respect timeout guard before expensive extraction compaction.

### Acceptance Criteria
1. Extractor no longer injects document-scale payloads by default.
2. Extraction remains best-effort and does not starve execution path.

### Workstream 7: Telemetry, Rollout, and Validation
### Files
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`
2. `/Users/sfw/Development/loom/src/loom/events/types.py` (only if needed)
3. `/Users/sfw/Development/loom/tests/test_orchestrator.py`
4. `/Users/sfw/Development/loom/tests/test_semantic_compactor.py`
5. `/Users/sfw/Development/loom/tests/test_verification.py`

### Changes
1. Add observability for lane decisions, feasibility skips, advisory usage, and budget exhaustion.
2. Expand tests for protected write-lane behavior and anti-thrash boundaries.
3. Roll out behind config mode:
   - `tiered` (current deterministic baseline)
   - `tiered_adaptive` (new planner + optional advisory)

### Acceptance Criteria
1. Clear event-log trace of why compaction was or was not applied.
2. Draft-heavy workloads show reduced compactor churn and improved completion reliability.

## Minimal Algorithm (Per Runner Iteration)
1. Build candidate graph with lane + utility metadata.
2. Compute pressure ratio and required token delta.
3. If no pressure, return raw context.
4. Generate compaction operations ordered by lowest estimated harm per token saved.
5. If top candidates are near-tied, request advisory ranking once.
6. Apply operations incrementally with feasibility checks and budgets.
7. Stop as soon as target pressure is satisfied.
8. Emit diagnostics and carry forward runtime stats.

## Configuration Additions
Add lean runtime knobs; avoid large policy surface.
1. `runner_compaction_policy_mode = "tiered" | "tiered_adaptive" | "legacy"`
2. `compaction_advisory_enabled: bool`
3. `compaction_advisory_candidate_limit: int`
4. `compaction_advisory_tie_epsilon: float`
5. `compaction_iteration_call_budget: int`
6. `compaction_label_cooldown_turns: int`
7. `compaction_feasibility_min_expected_ratio: float`

Defaults should keep behavior safe without requiring hand tuning.

## Validation Strategy
1. Replay heavy drafting runs and compare:
   - completion rate,
   - compactor calls per iteration,
   - wall-clock spent in compaction,
   - output completeness/quality proxy checks.
2. Assert no hard truncation markers inserted by runner compaction path.
3. Verify protected write-lane survives pressure and critical tiers.
4. Verify graceful fallback when advisory model is unavailable.

## Risks and Mitigations
1. Risk: advisory latency increases turn time.
   Mitigation: one-shot advisory, capped candidates, strict timeout.
2. Risk: utility scoring misclassifies edge cases.
   Mitigation: deterministic lane invariants + telemetry-driven tuning.
3. Risk: extra config complexity.
   Mitigation: minimal knobs and strong defaults.

## Non-Goals
1. No offline static analysis dependency.
2. No full ML training loop for policy.
3. No hard-cap truncation fallback in runner prompt assembly.

## Exit Criteria
1. Adaptive mode is stable in drafting/document-heavy runs.
2. Compaction churn is bounded and explainable in logs.
3. Final deliverables preserve high-fidelity active writing context.
4. Subtasks complete more reliably under long-context pressure.
