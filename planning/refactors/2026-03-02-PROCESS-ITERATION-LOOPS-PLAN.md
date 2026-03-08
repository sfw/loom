# Process Iteration Loops Plan (2026-03-02)

## Objective
Add first-class, bounded, gate-driven iteration to process contracts so Loom can execute Ralph-style cycles safely:
1. Execute.
2. Evaluate against explicit gates.
3. Apply targeted remediation.
4. Repeat until gates pass or bounded budgets are exhausted.

## Why This Matters
This feature establishes a general loop system for process execution where one pass is often insufficient.

Examples (not special cases):
1. Writing: iterate until quality gates pass.
2. Coding: implement -> test -> fix -> repeat until gates pass.
3. Planning/analysis: critique -> patch -> re-evaluate until gates pass.
4. Data workflows: collect -> validate -> remediate until schema/coverage gates pass.

Current behavior already has retries and replanning, but not a declarative process-level loop contract tied to explicit convergence gates across arbitrary domains.

## Existing Capabilities We Should Reuse
1. Subtask retry loop and failure handling:
   - `<repo-root>/src/loom/engine/orchestrator.py` (`_handle_failure`).
2. Global run budget and budget exhaustion safeguards:
   - `<repo-root>/src/loom/engine/orchestrator.py` (`_enforce_global_budget`).
   - `<repo-root>/src/loom/config.py` (`ExecutionConfig` limits).
3. Tool loop iteration budget:
   - `<repo-root>/src/loom/engine/runner.py`.
4. Verification contracts and reason codes:
   - `<repo-root>/src/loom/engine/verification.py`.
5. Existing scoring signals for writing:
   - `<repo-root>/src/loom/tools/humanize_writing.py`.

## Core Problem
Process spec has no explicit loop semantics (what to repeat, when to stop, how to detect convergence). We currently rely on generic retry/replan behavior, which is:
1. Not explicit to process authors.
2. Not reliably metric-driven.
3. Hard to reason about for product outcomes.

## Non-Negotiable Invariants
1. No unbounded loops.
2. Loop progression must be observable and explainable.
3. Global budgets always override loop policies.
4. Deterministic gates are preferred for blocking decisions.
5. Loop must degrade gracefully (clear terminal state with reason).
6. Avoid multiplicative retry explosions (`iteration x retry x replan`) via explicit layered caps.
7. Gate-evaluation uncertainty must converge deterministically (bounded retries, then explicit replan/fail path).

## Draft Proposal (v0)

### 1) Process Contract Extension
Add optional `iteration` policy to phases (phase-level only in MVP).

Example:

```yaml
phases:
  - id: rewrite
    description: "Improve draft quality."
    depends_on: [draft]
    model_tier: 2
    verification_tier: 2
    acceptance_criteria: "Score >= 80 and no placeholder text."
    iteration:
      enabled: true
      max_attempts: 4
      strategy: targeted_remediation   # targeted_remediation | full_rerun
      stop_on_no_improvement_attempts: 2
      gates:
        - id: humanization-score
          type: tool_metric            # tool_metric | command_exit | artifact_regex | verifier_field
          blocking: true
          tool: humanize_writing
          metric_path: report.humanization_score
          operator: gte                # gte | lte | eq | contains | not_contains
          value: 80
        - id: no-placeholders
          type: artifact_regex
          blocking: true
          target: deliverables
          pattern: "\\[TBD\\]|\\[TODO\\]|\\[PLACEHOLDER\\]"
          expect_match: false
```

### 2) Runtime Semantics (v0)
1. Phase executes normally.
2. Gate evaluator runs after verification.
3. If any blocking gate fails and attempts remain:
   - increment loop attempt.
   - build focused retry context with failed gate deltas.
   - rerun same subtask/phase.
4. Exit when:
   - all blocking gates pass, or
   - loop attempts exhausted, or
   - global budget exhausted.

### 2.2) Gate-Evaluation Failure Semantics (Locked)
If a gate cannot be evaluated (timeout, missing metric path, parse/read error, command policy rejection):
1. Retry gate evaluation once within the same iteration attempt.
2. If still unevaluable, mark gate as failed with reason `gate_unevaluable`.
3. Consume the iteration attempt and continue loop logic.
4. If loop exhausts on this failure pattern, request replan (bounded by replan cap).

### 2.3) Layered Boundedness and Counter Separation (Locked)
Counters are separate but jointly bounded:
1. `iteration_attempt` (phase-loop counter) is distinct from `subtask.retry_count`.
2. Subtask retries remain for execution/transient failures.
3. Loop retries remain for gate failures.
4. Replans after loop exhaustion are bounded to `2` attempts by default.

Hard safety guards:
1. `iteration.max_attempts` default `4`.
2. `iteration.max_total_runner_invocations` required to cap combined loop+retry churn.
3. Existing global budgets remain ultimate stop condition.

### 2.4) General Loop System (Domain-Agnostic)
The loop engine is generic and should not encode writing/coding-specific behavior:
1. Loop controller executes a phase repeatedly under declared budgets.
2. Gate registry evaluates typed gates and returns normalized outcomes.
3. Convergence policy decides whether to continue, replan, or terminate.
4. Process contracts supply gate definitions and stop behavior.

Domain behavior enters only through process YAML gate declarations, not runtime hardcoding.

### 3) Gate Types (MVP)
1. `tool_metric`: evaluate known tool output paths from structured tool result data.
2. `command_exit`: run a command and compare exit code.
3. `artifact_regex`: regex include/exclude checks over declared deliverables.
4. `verifier_field`: gate on structured verifier outputs (`outcome`, `reason_code`, metadata counters).

MVP enforcement rules:
1. Blocking gates must be deterministic (`tool_metric`, `command_exit`, `artifact_regex`).
2. `verifier_field` is advisory-only in MVP (cannot be sole blocker).
3. `command_exit` is sandbox-only in MVP and must pass allowlist validation.

### 4) Data Model
Persist per-loop metadata on subtask:
1. `iteration_attempt`.
2. `iteration_history` entries (failed gates, deltas, summary).
3. `iteration_terminal_reason` (`passed`, `max_attempts_exhausted`, `no_improvement`, `budget_exhausted`).

Persist to SQLite (first-class) in MVP:
1. `iteration_runs` (task_id, subtask_id/phase_id, run_id, policy snapshot, terminal reason).
2. `iteration_attempts` (attempt index, gate outcomes, counters, summary deltas).
3. `iteration_gate_results` (gate_id, status, measured value, threshold, reason code).

Source of truth and reconciliation:
1. SQLite is the source of truth when iteration feature flag is enabled.
2. Task metadata may mirror compact loop status for prompt/context use, but is non-authoritative.
3. Startup/hydration reads from SQLite first and repopulates in-memory/task metadata mirrors.
4. On mismatch, SQLite wins and emits a reconciliation event.

## Implementation Plan (v0)

### W0: SQLite Schema and Migration (MVP Foundation)
Files:
1. `<repo-root>/src/loom/state/schema.sql`
2. `<repo-root>/src/loom/state/memory.py`
3. `<repo-root>/src/loom/config.py`

Tasks:
1. Add `iteration_runs`, `iteration_attempts`, and `iteration_gate_results` tables.
2. Add memory-manager APIs for write/read/reconcile of iteration records.
3. Define feature flag and source-of-truth rules (SQLite authoritative when enabled).
4. Add reconciliation event on divergence between metadata mirror and SQLite.

### W1: Schema and Validation
Files:
1. `<repo-root>/src/loom/processes/schema.py`

Tasks:
1. Add dataclasses for `IterationPolicy` and `IterationGate`.
2. Parse and validate new fields.
3. Fail fast on invalid gate definitions.
4. Enforce install/runtime lint rules for risky configs:
   - reject excessive attempt caps.
   - require stop criteria for score-like loops.
   - require command allowlist compliance for `command_exit`.

### W2: Orchestrator Loop Integration
Files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. `<repo-root>/src/loom/state/task_state.py`

Tasks:
1. Attach loop state to subtask attempts.
2. Introduce gate evaluation step after successful execution/verification.
3. Requeue subtask when iteration policy requires rerun.
4. On loop exhaustion, request replan by default.
5. Keep `iteration_attempt` separate from `subtask.retry_count`.
6. Enforce `iteration.max_total_runner_invocations` to prevent layered retry explosions.

### W3: Gate Evaluator
Files:
1. `<repo-root>/src/loom/engine/verification.py` (or new module `engine/iteration_gates.py`)

Tasks:
1. Implement deterministic evaluators (`tool_metric`, `command_exit`, `artifact_regex`).
2. Implement verifier-field evaluator.
3. Return normalized gate results with reason codes.
4. Implement gate-evaluation retry-once semantics, then `gate_unevaluable` failure.
5. Enforce `command_exit` security policy:
   - sandbox-only execution.
   - argument-vector execution (no shell metacharacter expansion).
   - allowlisted command prefixes only.

### W4: Retry Context and Convergence
Files:
1. `<repo-root>/src/loom/recovery/retry.py`
2. `<repo-root>/src/loom/engine/orchestrator.py`

Tasks:
1. Encode failed-gate summary in retry context.
2. Add no-improvement detection and stop rules.
3. Keep best-known outcome metadata for explainability.
4. Add loop exhaustion fingerprinting and bounded replan behavior:
   - default max replans after loop exhaustion: `2`.
   - fail terminally on repeated equivalent exhaustion fingerprint.

### W5: Observability and UX
Files:
1. `<repo-root>/src/loom/events/types.py`
2. `<repo-root>/src/loom/engine/orchestrator.py`
3. `<repo-root>/src/loom/tui/*` (display path)

Tasks:
1. Emit `iteration_started`, `iteration_gate_failed`, `iteration_retrying`, `iteration_completed`, `iteration_terminal`.
2. Surface current loop attempt and failing gate in progress views.
3. MVP control plane:
   - stop and pause process-run loops.
   - inject operator instruction into current loop context.
4. Keep UX parity with cowork as implementation reference while limiting MVP to loop-control essentials above.

### W6: Tests and Docs
Files:
1. `<repo-root>/tests/test_processes.py`
2. `<repo-root>/tests/test_orchestrator.py`
3. `<repo-root>/tests/test_verification.py`
4. `<repo-root>/docs/creating-packages.md`
5. `<repo-root>/docs/CONFIG.md`

Tasks:
1. Add loop convergence tests for writing and coding examples.
2. Add budget precedence tests.
3. Add process authoring examples with anti-pattern warnings.
4. Add tests for:
   - gate unevaluable retry then fail behavior.
   - bounded replan-after-loop-exhaustion (`max=2`) and fingerprint terminal fail.
   - `command_exit` allowlist + sandbox enforcement.
   - SQLite as source-of-truth reconciliation.

## Risks (v0)
1. Runaway cost/time from non-convergent loops.
2. Oscillation (fix one gate, regress another).
3. Flaky pass/fail with non-deterministic gates.
4. Process-author misuse (too many strict gates, unrealistic thresholds).
5. Safety risk from repeated mutating actions.

## Critique Round 1 (Cost/Freeze Risk)
Weaknesses in v0:
1. Loop stop conditions are present but not strongly coupled to global budgets.
2. No explicit per-loop budget partition.
3. No hard cap on gate evaluator command runtime.

Strengthening changes (v1):
1. Add `iteration_budget` fields:
   - `max_wall_clock_seconds`
   - `max_tokens`
   - `max_tool_calls`
2. Enforce `min(global_budget_remaining, iteration_budget_remaining)` at each attempt.
3. Add per-gate command timeout and safe command allowlist policy.
4. Add terminal reason `iteration_budget_exhausted` distinct from global budget exhaustion.

## Critique Round 2 (Correctness/Flakiness Risk)
Weaknesses in v1:
1. `verifier_field` gates can still be non-deterministic for hard blocking.
2. No mandatory distinction between hard vs advisory gate classes.
3. Missing anti-flake mechanics for transient test failures.

Strengthening changes (v2):
1. Gate class taxonomy:
   - `hard_deterministic` (blocking, default for exit decisions).
   - `semantic_advisory` (cannot be sole blocker unless explicitly opted in).
2. Add optional `confirmations_required` for flaky gates.
3. Add transient classification and one-shot retry for known transient failures.
4. Require process authors to mark each blocking gate with `failure_policy`:
   - `retry`
   - `queue_follow_up`
   - `abort_phase`

## Critique Round 3 (Product/Usability Risk)
Weaknesses in v2:
1. Contract is powerful but verbose for process authors.
2. No simple defaults for common scenarios.
3. Hard to reason about loop interactions with existing retry/replan.

Strengthening changes (v3, recommended):
1. Add presets:
   - `preset: writing_quality`
   - `preset: test_fix_loop`
   - `preset: critique_hardening`
2. Keep advanced fields optional; presets compile to full gate configs.
3. Define strict precedence:
   - global budget -> iteration budget -> loop gates -> retry -> replan.
4. Scope MVP to phase-level loops only (no nested loops, no cross-phase loops).
5. Add dry-run/lint command for loop policy simulation before runtime.
6. Keep presets optional convenience only; core system remains a fully generic gate-driven loop engine.

## Final Recommended MVP
1. Phase-level `iteration` policy only.
2. Deterministic blocking gates only in MVP (`tool_metric`, `command_exit`, `artifact_regex`).
3. `verifier_field` allowed as advisory only in MVP.
4. Hard attempt cap required (`max_attempts` <= 6).
5. `iteration_budget` required when `iteration.enabled=true`:
   - `max_wall_clock_seconds`
   - `max_tokens`
   - `max_tool_calls`
6. `command_exit` requires sandbox-only + allowlisted prefixes + per-gate timeout.
7. No-improvement stop required for score-driven loops.
8. Budget precedence strictly enforced with explicit terminal reasons.
9. Loop exhaustion requests replan by default.
10. Replan-after-loop-exhaustion hard cap: `2` by default, with exhaustion fingerprint termination.
11. Persist iteration state in SQLite in MVP.
12. Add MVP stop/pause/inject controls for process-run loops.
13. Start with two first-party examples for validation coverage only:
   - Writing: `humanization_score >= 80`.
   - Coding: `pytest` exit code 0.

## Rollout
1. Phase 1 (dark launch):
   - schema parse + validation only, no runtime behavior.
2. Phase 2 (flagged runtime):
   - runtime iteration with deterministic gates behind config flag.
3. Phase 3 (process adoption):
   - enable on one built-in process and one coding package.
4. Phase 4 (broader adoption):
   - publish presets and authoring docs.

## MVP Exit Criteria (Release Gate)
1. No infinite loop path is possible under any valid config.
2. Combined loop/retry/replan execution is bounded by explicit caps and emits terminal reason codes.
3. SQLite iteration records are authoritative and recoverable across restart.
4. `command_exit` cannot execute outside sandbox and rejects non-allowlisted commands.
5. Gate-evaluation errors follow retry-once then deterministic resolution (`gate_unevaluable` -> bounded replan/fail path).
6. Stop/pause/inject controls operate during live process-run loops.

## Product Scope Status
All previously open product questions for MVP are now resolved and captured as locked decisions below.

## Product Decisions Captured (2026-03-02)
1. Scope: phase-level loops in MVP.
2. Loop exhaustion behavior: replan by default.
3. `command_exit` policy: sandbox-only by default.
4. Semantic/LLM blocking gates in MVP: not allowed (advisory only).
5. Defaults: both process-defined and global caps (global safety override).
6. Loop history persistence: SQLite in MVP.
7. Default attempts: 4.
8. System framing: generic loop system; writing/coding are examples only.
9. Follow-up remediation success path: yes, for non-critical paths.
10. UX controls: include MVP stop/pause/inject for process-run loops (cowork parity is reference, not hard dependency).
11. Telemetry day one: no per-gate cost accounting requirement.
12. Linting: fail risky configs.
13. Counter model: `iteration_attempt` is separate from `subtask.retry_count`.
14. Replan-after-loop-exhaustion cap: default `2`.
15. Gate-evaluation failure behavior: retry once, then mark `gate_unevaluable`; on loop exhaustion, follow bounded replan path.
