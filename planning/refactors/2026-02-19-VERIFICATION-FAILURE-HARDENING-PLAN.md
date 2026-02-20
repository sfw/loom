# Verification Failure Hardening Plan (2026-02-19)

## Objective
Eliminate remaining verification failure modes after the robustness refactor, with emphasis on process continuity, deterministic safeguards, and recoverability.

This plan targets the seven follow-on improvements:
1. remediation queue execution lifecycle
2. structured retry routing (not free-text parsing)
3. rule-scope governance/linting
4. deterministic synthesis partition checks
5. confirm-or-prune retry budget with transient handling
6. golden-corpus regression and shadow quality gates
7. TUI run-scoped workspace folder isolation with LLM naming

## Why This Is Needed
Current system quality is materially improved, but still has residual risk:
- queued remediations can be recorded but not operationally resolved
- retry strategy classification still depends on feedback text matching
- legacy rules without scope metadata can still leak across phases
- synthesis integrity partially depends on verifier self-report fields
- critical confirm-or-prune remediation is currently single-attempt
- no persistent golden corpus from real failures to prevent regressions
- process artifacts from TUI runs are not guaranteed to be isolated in a dedicated run folder

## Target End State
Verification and remediation become contract-driven, observable, and resilient:
- every non-pass maps to structured outcome + reason code + remediation action
- remediation items run to terminal state (`resolved`, `failed`, `expired`)
- retries are routed by machine-readable classification, not text heuristics
- phase-scoped rules are mandatory for process definitions (with migration mode)
- synthesis structure and recommendation integrity have deterministic checks
- real historical failures are continuously replayed as golden regression tests
- each TUI-started process run gets a dedicated workspace subfolder (LLM-named with fallback), and all run artifacts are confined to it

## Scope
Core runtime:
- `/Users/sfw/Development/loom/src/loom/engine/verification.py`
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/engine/runner.py`
- `/Users/sfw/Development/loom/src/loom/recovery/retry.py`
- `/Users/sfw/Development/loom/src/loom/recovery/errors.py`
- `/Users/sfw/Development/loom/src/loom/state/task_state.py`
- `/Users/sfw/Development/loom/src/loom/state/evidence.py`
- `/Users/sfw/Development/loom/src/loom/events/types.py`
- `/Users/sfw/Development/loom/src/loom/tui/app.py`
- `/Users/sfw/Development/loom/src/loom/cowork/session.py`

Process/schema governance:
- `/Users/sfw/Development/loom/src/loom/processes/schema.py`
- `/Users/sfw/Development/loom/src/loom/processes/builtin/*.yaml`
- `/Users/sfw/Development/loom/packages/*/process.yaml`

Prompting:
- `/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`

Tests and fixtures:
- `/Users/sfw/Development/loom/tests/test_verification.py`
- `/Users/sfw/Development/loom/tests/test_orchestrator.py`
- `/Users/sfw/Development/loom/tests/test_retry.py`
- `/Users/sfw/Development/loom/tests/test_processes.py`
- `/Users/sfw/Development/loom/tests/test_tui.py`
- `/Users/sfw/Development/loom/tests/test_workspace.py`
- `/Users/sfw/Development/loom/tests/golden/verification/*`

## Workstreams

### W1: Structured Verification Failure Contract
1. Introduce a canonical verification contract:
- `outcome`: `pass|pass_with_warnings|partial_verified|fail`
- `reason_code`: controlled set (enum-style constants)
- `remediation_mode`: `none|queue_follow_up|confirm_or_prune`
- `severity_class`: `hard_invariant|semantic|inconclusive|infra`
2. Make orchestrator and retry routing consume the structured contract first.
3. Keep text-based fallback routing only as temporary compatibility path.

Exit criteria:
- Retry routing does not depend on free-text in >95% of non-pass cases.
- All terminal failures include known `reason_code`.

### W2: Remediation Queue Execution Engine
1. Add explicit remediation item model persisted with task state:
- `id, task_id, subtask_id, strategy, blocking, state, attempt_count`
- `last_error, created_at, updated_at, next_attempt_at, ttl_at`
2. Build execution path for queued remediation in orchestrator loop:
- runs opportunistically for non-blocking items
- must run before terminal abort for blocking items
3. Add lifecycle states:
- `queued -> running -> resolved|failed|expired`
4. Add idempotency guard for duplicate queue entries per `(task, subtask, reason_code)`.

Exit criteria:
- `remediation_queue` items are actively processed, not passive metadata.
- No blocking remediation remains in `queued` at task terminal state.

### W3: Confirm-or-Prune Retry Budget + Transient-Aware Handling
1. Add dedicated config:
- `verification.confirm_or_prune_max_attempts`
- `verification.confirm_or_prune_backoff_seconds`
- `verification.confirm_or_prune_retry_on_transient`
2. Allow multiple confirm-or-prune attempts before critical abort.
3. Use error categorization (`timeout`, `model_error`, `rate_limit`) to decide retry vs terminal fail.
4. Emit remediation attempt metrics/events per attempt and final outcome.

Exit criteria:
- Critical-path remediation does not fail terminally after a single transient error.
- Attempt history is visible in state and events.

### W4: Rule Scope Governance and Linting
1. Add process validation/lint mode requiring explicit rule scope metadata:
- each rule must declare `applies_to_phases` or explicit `scope`
2. Migration mode:
- warn-only first (`process.require_rule_scope_metadata = false`)
- enforce later (`true`)
3. Add process-lint tests and package-level checks in CI.

Exit criteria:
- New/updated process definitions cannot merge with ambiguous rule scope.
- Cross-phase leakage from unscope legacy rules is eliminated in enforce mode.

### W5: Deterministic Synthesis Integrity Checks
1. Add deterministic checks for synthesis artifacts:
- if unconfirmed supporting claims exist, verify section headers:
  `Verified Findings` and `Unconfirmed Appendix`
2. Add deterministic recommendation integrity check:
- recommendation section must not include unconfirmed/unsupported markers
3. Cross-check deterministic findings against verifier-reported metadata and fail on contradiction.

Exit criteria:
- Synthesis partition is validated from artifact content, not just LLM self-report.
- Recommendation section remains zero-unconfirmed under deterministic guard.

### W6: Golden Corpus + Shadow Evaluation
1. Build golden corpus from real failure logs/SQL snapshots:
- parse verifier outputs, tool-call patterns, phase/rule context, expected disposition
2. Add replay harness:
- old vs new policy comparison
- diff classes: `old_fail_new_pass`, `old_pass_new_fail`, `reason_diff`
3. Add quality gates based on corpus + canary runs.

Exit criteria:
- corpus replay is part of CI/verification pipeline
- false-fail reduction and bad-pass control are measured before cutover decisions

### W7: TUI Run-Scoped Workspace Foldering
1. At TUI process start, generate a succinct run folder slug via LLM from process context (goal/process name).
2. Add deterministic naming fallback when LLM naming is unavailable, invalid, or times out.
3. Create run folder under selected workspace root before first process step:
- sanitize slug for filesystem safety
- enforce uniqueness with collision suffixing (`-2`, `-3`, ...)
4. Persist run folder path in run/task state and pass it through orchestrator/runner as effective workspace.
5. Enforce artifact confinement:
- all generated output files for the run must resolve inside the run folder
- reject/normalize path escapes and emit explicit violation event if encountered
6. Surface run folder path in TUI status/log output for operator visibility.

Exit criteria:
- every TUI-started run creates a dedicated run folder before task execution
- integration tests confirm generated artifacts stay under that run folder
- fallback folder naming path is covered and reliable under model/transient failures

## Rollout Strategy
Phase rollout:
1. Observe: instrumentation + shadow compare only
2. Soft enforce: remediation queue active, scope lint warnings
3. Enforce: structured routing mandatory, scope lint hard fail, synthesis deterministic guards hard

Feature flags:
- `verification.shadow_compare_enabled`
- `verification.policy_engine_enabled`
- `verification.strict_output_protocol`
- `verification.allow_partial_verified`
- `verification.auto_confirm_prune_critical_path`
- `verification.confirm_or_prune_max_attempts`
- `verification.confirm_or_prune_retry_on_transient`
- `process.require_rule_scope_metadata`
- `process.tui_run_scoped_workspace_enabled`
- `process.llm_run_folder_naming_enabled`

Rollback:
- fallback to legacy aggregation/routing via flags while preserving instrumentation.

## PR Sequence
### PR-H1: Structured Contract Consolidation
- normalize `reason_code` + remediation metadata contract
- structured routing in retry manager with compatibility fallback

### PR-H2: Remediation Queue Model + Persistence
- explicit queue item schema + state manager helpers
- idempotency and lifecycle state transitions

### PR-H3: Remediation Executor
- orchestrator execution path for queued items (blocking + non-blocking)
- terminal-state guarantees for blocking remediations

### PR-H4: Confirm-or-Prune Retry Budget
- multi-attempt remediation with transient-aware backoff
- critical-path abort only after budget exhaustion

### PR-H5: Rule Scope Lint Enforcement
- schema validation flag + linter
- CI checks for process packages

### PR-H6: Deterministic Synthesis Guards
- section and recommendation integrity parsing checks
- contradiction detection against verifier metadata

### PR-H7: Golden Corpus Harness
- fixtures, replay runner, diff metrics, and acceptance thresholds

### PR-H8: Cutover + Cleanup
- deprecate brittle text-routing paths
- promote enforce-mode defaults after canary thresholds

### PR-H9: TUI Run-Scoped Workspace Isolation
- LLM run-folder naming with deterministic fallback and sanitization
- run-folder creation + persistence + orchestrator/runner workspace wiring
- artifact-confinement guardrails and TUI visibility updates

## Acceptance Criteria
1. Routing quality:
- >=95% of retry decisions driven by structured `reason_code`/metadata.

2. Remediation continuity:
- >=95% of queued remediations reach terminal state (`resolved|failed|expired`) within SLA.

3. Critical-path resilience:
- confirm-or-prune salvage rate improves materially (target: +30% vs single-attempt baseline).

4. Scope correctness:
- 0 known cross-phase false failures from unscope rules in enforce mode.

5. Synthesis integrity:
- 0 recommendation sections with unconfirmed claims in canary + corpus replay.

6. Regression safety:
- no statistically significant increase in bad-pass defects in shadow/canary review.

7. Run artifact isolation:
- 100% of TUI-started process runs create and use a dedicated run folder.
- 0 generated process artifacts written outside the run folder in integration coverage.

## Test Strategy
1. Unit tests
- structured routing precedence over free-text heuristics
- remediation item lifecycle transitions
- deterministic synthesis section parsing + recommendation checks
- scope lint enforcement and migration behavior
- run-folder slug sanitization, collision suffixing, and deterministic fallback behavior

2. Integration tests
- end-to-end non-critical queue -> remediation -> completion
- critical-path confirm-or-prune with transient failure then recovery
- blocking remediation unresolved -> controlled terminal failure
- TUI process start creates run-scoped folder and passes it as effective workspace
- all process outputs for a run are written inside that run folder

3. Golden replay tests
- historical log+SQL failures mapped to expected final disposition
- policy diff classification assertions

## Risks and Mitigations
1. Risk: over-enforcement increases failures during migration.
- Mitigation: staged warn/enforce flags, shadow mode, process-level exemptions with expiry.

2. Risk: remediation loop increases token/runtime cost.
- Mitigation: bounded retry budgets, targeted context, idempotent queue dedupe.

3. Risk: deterministic synthesis parser is too brittle for format variation.
- Mitigation: strict but minimal required markers + compatibility parser tests.

4. Risk: LLM-generated folder names may be invalid, low quality, or collide frequently.
- Mitigation: strict sanitization, bounded-length slug rules, deterministic fallback, and collision-safe suffixing.

## Definition of Done
This hardening is done when:
1. verification failures are structurally routed and observable
2. remediations are executed to terminal state with bounded retries
3. critical-path unconfirmed failures no longer terminally fail on first remediation miss
4. phase-rule scope is enforced for maintained process definitions
5. synthesis integrity is guaranteed by both deterministic and semantic checks
6. golden-corpus replay prevents recurrence of known failure classes
7. TUI-started process artifacts are isolated into a dedicated run-scoped workspace folder
