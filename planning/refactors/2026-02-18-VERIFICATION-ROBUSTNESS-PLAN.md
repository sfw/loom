# Verification Robustness Refactor Plan (2026-02-18)

## Objective
Make process verification resilient to loosely/dynamically structured outputs while preserving strict quality standards and keeping deterministic checks only where they are truly safety-critical.

## Why This Refactor Is Needed
Current verification still has deterministic shape-sensitive failure paths that can block valid executions:
- Tier 1 deterministic checks still gate all runs before LLM verification.
- Process regex rules can hard-fail on text-pattern matches.
- Some process/YAML parsing paths still assume strict object shapes.
- Verifier parse failures can still end in inconclusive hard-fail outcomes.
- Domain LLM verification rules are not phase-scoped, so rules can leak across subtasks and cause cross-phase false failures.

This creates false negatives when outputs are semantically correct but structurally non-uniform.

## Design Principles
1. Semantic correctness over rigid shape assumptions.
2. Deterministic checks only for hard invariants (safety/integrity), never domain semantics.
3. Policy-driven verdicts: every failure must map to an explicit rule and severity.
4. Parse/infrastructure uncertainty should trigger verification retry, not immediate execution failure.
5. Observability-first rollout: ship with shadow comparison before hard cutover.
6. Verification rules should be phase-scoped by default, with explicit opt-in for global rules.
7. Unconfirmed data should be quarantined and remediated, not used to discard all prior work by default.

## Target End State
Verification should produce a final verdict from three categories of signals:
1. Hard deterministic invariants
- Tool execution fatality (excluding explicitly advisory transient failures).
- Deliverable existence only when explicitly required by process contract.
- Syntax/parse validity only for machine-readable artifacts where parsing is required for downstream use.

2. LLM semantic assessment (primary)
- Process-level acceptance criteria.
- Domain-specific verification rules.
- Evidence traceability and unsupported-claim labeling.
- Dynamic schema interpretation from actual headers/sections/content.

3. Advisory diagnostics (non-blocking by default)
- Placeholder pattern matches.
- Weak or incomplete structure hints.
- Soft inconsistencies not violating explicit acceptance criteria.

## Verification Outcome Policy (New)
Verification should return one of four outcomes:
- `pass`: acceptance criteria satisfied with no blocking findings.
- `pass_with_warnings`: acceptance criteria satisfied; only advisory findings remain.
- `partial_verified`: core requirements met, but unresolved unconfirmed content exceeds supporting threshold or is isolated to non-critical sections.
- `fail`: hard invariant or hard semantic requirement not met.

Decision policy for unconfirmed data:
- Unconfirmed claims must never appear in final executive recommendations.
- Supporting analysis may include unconfirmed claims up to a bounded threshold if explicitly labeled.
- Unconfirmed claims above threshold should trigger remediation and downgrade verdict to `partial_verified` unless hard requirements are violated.

Path-specific handling:
- Critical-path subtask: auto-run `confirm-or-prune` remediation before declaring terminal failure.
- Critical-path subtask: if remediation cannot confirm, prune unconfirmed claims and retry semantic verification once.
- Non-critical subtask: continue run with `pass_with_warnings` or `partial_verified`.
- Non-critical subtask: queue remediation as follow-up work item.
- Final synthesis subtask: always separate `verified synthesis` from `unconfirmed appendix`.
- Final synthesis subtask: recommendations section must be 100% confirmed.
- Advisory/optional paths: never block process completion solely due to unconfirmed supporting claims.
- Advisory/optional paths: emit metrics and warnings for downstream review.

## Scope
Core runtime:
- `/Users/sfw/Development/loom/src/loom/engine/verification.py`
- `/Users/sfw/Development/loom/src/loom/engine/runner.py`
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/recovery/retry.py`
- `/Users/sfw/Development/loom/src/loom/models/router.py`

Process schema/config:
- `/Users/sfw/Development/loom/src/loom/processes/schema.py`
- `/Users/sfw/Development/loom/src/loom/config.py`
- `/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`
- `/Users/sfw/Development/loom/src/loom/processes/builtin/*.yaml`
- `/Users/sfw/Development/loom/packages/google-analytics/process.yaml`

Tests:
- `/Users/sfw/Development/loom/tests/test_verification.py`
- `/Users/sfw/Development/loom/tests/test_processes.py`
- `/Users/sfw/Development/loom/tests/test_process_testing.py`
- `/Users/sfw/Development/loom/tests/test_orchestrator.py`
- `/Users/sfw/Development/loom/tests/test_retry.py`

## Phase Plan

### Phase 0: Baseline + Guardrails (No Behavior Change)
1. Add verification metrics/events:
- `verification_false_negative_candidate`
- `verification_inconclusive_rate`
- `rule_failure_by_type`
- `deterministic_block_rate`

2. Add structured reason taxonomy for every fail:
- `hard_invariant_failed`
- `llm_semantic_failed`
- `parse_inconclusive`
- `infra_verifier_error`

Exit criteria:
- Every failed verification has machine-readable reason code.
- Dashboard-able counts per reason code.

### Phase 1: Policy Engine for Verification Outcomes
1. Introduce rule policy model:
- `enforcement`: `hard` | `advisory`
- `source`: `deterministic` | `llm`
- `scope`: `output` | `deliverables` | `evidence`

2. Refactor verifier aggregation:
- Hard-fail only on `enforcement=hard`.
- Attach advisory findings to feedback without failing.

3. Keep current behavior behind feature flag:
- `verification.policy_engine_enabled`.

Exit criteria:
- Policy engine can emulate current behavior.
- Policy engine can run with advisory-only regex mode.

### Phase 2: Robust Verifier Output Protocol
1. Make verifier output contract strict and self-repairing:
- Primary: JSON object with schema.
- First fallback: schema-repair prompt (same model).
- Second fallback: alternate verifier model.
- Only then classify as `parse_inconclusive`.

2. Change retry semantics:
- `parse_inconclusive` and verifier transport errors should trigger verification-only retries, never full execution rerun.

3. Add parser hardening tests:
- JSON wrapped in prose.
- YAML-like responses.
- Partial/incorrect keys.
- Ambiguous verdict wording.

Exit criteria:
- Inconclusive parse rate < 1% in deterministic test corpus.
- Verification-only retry path covers all parse/transport failure classes.

### Phase 3: Process Rule Migration (Static -> Policy + LLM)
1. Migrate regex rules to advisory by default:
- Existing `type: regex` remains supported but non-blocking unless explicitly marked hard.

2. Add process schema fields for explicit strictness:
- `enforcement: hard|advisory`
- `requires_exact_cardinality: true|false`
- `min_count` where cardinality is required.
- `applies_to_phases: [phase-id, ...] | ["*"]`
- `scope: phase|global` (compat alias for `applies_to_phases`)

3. Update built-in process YAMLs:
- Keep placeholder checks as advisory diagnostics.
- Encode true hard requirements explicitly in semantic LLM rules.
- Mark rule applicability per phase, with explicit `["*"]` only for true cross-phase checks.

Exit criteria:
- Built-in processes no longer hard-fail from placeholder regex alone.
- Hard failures correspond only to explicit contract violations.
- Rules do not apply outside their declared phase scope.

### Phase 3.5: Phase-Scoped Rule Execution
1. Change verifier prompt assembly to include only rules applicable to current subtask phase.
2. Set default applicability policy:
- Legacy rules with no scope metadata default to `current_phase` during migration mode.
- Temporary compatibility flag allows fallback to legacy global behavior.
3. Add rule applicability tracing:
- Emit `rule_applied` and `rule_skipped` events with `rule_id`, `phase_id`, `reason`.

Exit criteria:
- No cross-phase rule leakage in integration tests.
- Shadow diff shows reduced false fails with no rise in escaped defects.

### Phase 4: Deterministic Check Tightening
1. Restrict deterministic checks to truly invariant checks:
- Existence checks only for declared deliverables.
- Syntax checks only for changed/declared machine-readable files.
- No domain semantic assertions in deterministic layer.

2. Add deterministic fail-safe behavior:
- Missing data structure fields in tool outputs should degrade gracefully to advisory warnings, not runtime exceptions.

Exit criteria:
- Deterministic layer has zero domain-specific semantic logic.
- No crash/fail from non-dict/non-list tool payload variants.

### Phase 4.5: Unconfirmed Data Governance + Progressive Completion
1. Add explicit claim-confidence handling:
- Require claim labels (`confirmed`, `inferred`, `unsupported`) in synthesis pipeline.
- Treat `unsupported` as quarantined unless promoted by remediation.
2. Enforce recommendation integrity:
- Executive recommendations must contain 0 unconfirmed claims.
- If any recommendation claim is unconfirmed, run `confirm-or-prune` before final verdict.
3. Set supporting-content threshold:
- Default supporting threshold: 30% max unconfirmed claims in non-recommendation sections.
- Above threshold -> `partial_verified` + mandatory remediation task.
4. Add path-specific remediation routing:
- Critical-path failures trigger immediate remediation subtask.
- Non-critical failures create queued remediation work without blocking completion.
- Final synthesis produces two sections: `Verified Findings` and `Unconfirmed Appendix`.

Exit criteria:
- Recommendation sections are always fully confirmed.
- Runs with non-critical unconfirmed claims complete without full abort.
- Threshold-driven `partial_verified` behavior is deterministic and test-covered.

### Phase 5: Shadow Mode + Cutover
1. Run dual verification (old vs new) in shadow mode for process runs.
2. Capture diff classes:
- old-fail/new-pass
- old-pass/new-fail
- both-fail with different reasons

3. Review threshold gates before cutover:
- False negative reduction target met.
- No increase in verified-bad outputs in sampled QA review.

4. Remove/deprecate old hard-coded static paths after cutover.

Exit criteria:
- New system enabled by default.
- Legacy mode available only as temporary rollback flag.

## Acceptance Criteria (Program-Level)
1. Verification reliability
- `parse_inconclusive` < 1% of verifier invocations.
- Verification-only retry resolves >= 90% of inconclusive cases.

2. Quality protection
- No statistically significant increase in post-verification defect escapes in canary process runs.

3. Reduced false failures
- At least 50% reduction in structurally-induced false fails for market/research/report style processes.

4. Explainability
- 100% of fail verdicts include explicit rule IDs and enforcement level.

5. Phase correctness
- 100% of rule evaluations include phase applicability metadata.
- 0 known cross-phase false failures in golden corpus.

6. Process continuity
- Non-critical unconfirmed claims no longer cause full-process failure.
- >= 90% of prior "structural/unconfirmed-only" failures complete as `pass_with_warnings` or `partial_verified`.

7. Recommendation integrity
- 0 unconfirmed claims in executive recommendation sections across canary runs.

## Test Strategy
1. Unit tests
- Policy aggregation behavior (`hard` vs `advisory`).
- Verifier parsing fallbacks and retry routing.
- Schema loader robustness on malformed/heterogeneous YAML nodes.
- Rule applicability resolution (`current_phase` vs `global`).
- Threshold classification for `pass_with_warnings` vs `partial_verified`.

2. Integration tests
- Full subtask run with dynamic CSV/markdown shapes.
- Processes with single-company but plural phrasing constraints.
- Advisory regex hits with successful semantic verification.
- Cross-phase guard: risk-only rules must not fail earlier competition subtasks.
- Final synthesis output partition: verified section + unconfirmed appendix.
- Critical-path remediation flow (`confirm-or-prune`) before terminal fail.

3. Fuzz/property tests
- Randomized verifier text wrappers around valid/invalid JSON.
- Randomized tool result shapes (`dict`, `list`, scalar, null, mixed).

4. Golden corpus
- Archive historical failure examples from market-research/research-report and assert new verifier behavior.

## Rollout Controls
Feature flags:
- `verification.policy_engine_enabled`
- `verification.regex_default_advisory`
- `verification.strict_output_protocol`
- `verification.shadow_compare_enabled`
- `verification.phase_scope_default` (`current_phase|global`)
- `verification.allow_partial_verified`
- `verification.unconfirmed_supporting_threshold`
- `verification.auto_confirm_prune_critical_path`

Rollback:
- Single config toggle to return to legacy verification aggregation while keeping instrumentation.

## Risks and Mitigations
1. Risk: Over-relaxing checks lets low-quality outputs pass.
- Mitigation: hard contract fields + canary QA sampling + shadow diff review before cutover.

2. Risk: LLM verifier variance.
- Mitigation: strict output protocol, retry+repair, optional tier-3 voting for high-stakes subtasks.

3. Risk: Migration complexity across process YAMLs.
- Mitigation: schema compatibility layer and incremental per-process migration.

## Execution Order (Hard Priority)
1. Phase 0 (instrumentation)
2. Phase 1 (policy engine)
3. Phase 2 (protocol + retry hardening)
4. Phase 3 (rule migration)
5. Phase 3.5 (phase-scoped rule execution)
6. Phase 4 (deterministic tightening)
7. Phase 4.5 (unconfirmed data governance + progressive completion)
8. Phase 5 (shadow + cutover)

## Implementation Backlog (PR Sequence)
### PR-1: Instrumentation + Reason Codes
- Files:
`/Users/sfw/Development/loom/src/loom/engine/verification.py`
`/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
`/Users/sfw/Development/loom/src/loom/events/types.py`
`/Users/sfw/Development/loom/tests/test_verification.py`
- Must ship:
- Structured fail reason code attached to all non-pass outcomes.
- Event emission for reason code and enforcement type.
- Merge gate:
- Existing verification tests pass + new reason-code coverage tests.

### PR-2: Verification Policy Engine
- Files:
`/Users/sfw/Development/loom/src/loom/engine/verification.py`
`/Users/sfw/Development/loom/src/loom/config.py`
`/Users/sfw/Development/loom/tests/test_verification.py`
- Must ship:
- `hard` vs `advisory` aggregation in one policy layer.
- Config flag `verification.policy_engine_enabled`.
- Merge gate:
- Policy-engine-off preserves current behavior exactly.

### PR-3: Verifier Output Protocol Hardening
- Files:
`/Users/sfw/Development/loom/src/loom/engine/verification.py`
`/Users/sfw/Development/loom/src/loom/models/router.py`
`/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`
`/Users/sfw/Development/loom/tests/test_verification.py`
- Must ship:
- Multi-step parse recovery pipeline.
- Alternate-model verifier retry fallback.
- Merge gate:
- Inconclusive parse tests pass across malformed response corpus.

### PR-4: Retry Routing (Verification-only for Parse/Infra)
- Files:
`/Users/sfw/Development/loom/src/loom/recovery/retry.py`
`/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
`/Users/sfw/Development/loom/tests/test_orchestrator.py`
`/Users/sfw/Development/loom/tests/test_retry.py`
- Must ship:
- Parse/transport verifier failures never rerun execution path.
- Merge gate:
- Orchestrator tests prove execution is skipped for verifier-only retries.

### PR-5: Process Rule Schema Evolution
- Files:
`/Users/sfw/Development/loom/src/loom/processes/schema.py`
`/Users/sfw/Development/loom/tests/test_processes.py`
- Must ship:
- Rule-level enforcement metadata (`hard`/`advisory`).
- Rule applicability metadata (`applies_to_phases` / scope).
- Backward compatibility for existing YAML definitions.
- Merge gate:
- Legacy process YAMLs still load without edits.

### PR-5A: Phase-Scoped Rule Injection
- Files:
`/Users/sfw/Development/loom/src/loom/prompts/assembler.py`
`/Users/sfw/Development/loom/src/loom/engine/verification.py`
`/Users/sfw/Development/loom/src/loom/config.py`
`/Users/sfw/Development/loom/tests/test_verification.py`
- Must ship:
- Only current-phase + global rules are injected for verifier.
- Compatibility flag for legacy global injection mode.
- Rule apply/skip telemetry emitted.
- Merge gate:
- Integration tests prove no cross-phase rule leakage.

### PR-6: Built-in Process Rule Migration
- Files:
`/Users/sfw/Development/loom/src/loom/processes/builtin/*.yaml`
`/Users/sfw/Development/loom/packages/google-analytics/process.yaml`
`/Users/sfw/Development/loom/tests/test_processes.py`
`/Users/sfw/Development/loom/tests/test_process_testing.py`
- Must ship:
- Placeholder regex converted to advisory behavior by policy.
- Hard requirements encoded in semantic rules.
- Rule applicability declared for each verification rule.
- Merge gate:
- Process contract tests updated and passing.

### PR-6A: Unconfirmed Data Policy + Synthesis Partitioning
- Files:
`/Users/sfw/Development/loom/src/loom/engine/verification.py`
`/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
`/Users/sfw/Development/loom/src/loom/processes/schema.py`
`/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`
`/Users/sfw/Development/loom/tests/test_orchestrator.py`
`/Users/sfw/Development/loom/tests/test_verification.py`
- Must ship:
- Outcome model supports `pass_with_warnings` and `partial_verified`.
- Recommendation section hard-check enforces 0 unconfirmed claims.
- Supporting threshold default (`30%`) enforced for synthesis verdict.
- Merge gate:
- Golden tests show runs continue with quarantined unconfirmed sections while preserving recommendation integrity.

### PR-6B: Confirm-or-Prune Remediation Flow
- Files:
`/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
`/Users/sfw/Development/loom/src/loom/recovery/retry.py`
`/Users/sfw/Development/loom/src/loom/processes/builtin/*.yaml`
`/Users/sfw/Development/loom/tests/test_retry.py`
`/Users/sfw/Development/loom/tests/test_orchestrator.py`
- Must ship:
- Critical-path unconfirmed failures trigger targeted remediation subtask before terminal abort.
- Non-critical unconfirmed failures create queued remediation work item without blocking completion.
- Merge gate:
- Critical and non-critical path routing behavior verified end-to-end.

### PR-7: Deterministic Layer Tightening
- Files:
`/Users/sfw/Development/loom/src/loom/engine/verification.py`
`/Users/sfw/Development/loom/src/loom/engine/runner.py`
`/Users/sfw/Development/loom/src/loom/state/evidence.py`
`/Users/sfw/Development/loom/tests/test_verification.py`
- Must ship:
- Deterministic checks limited to hard invariants.
- Non-uniform data-shape handling degrades to advisory, never crash.
- Merge gate:
- Fuzz tests for mixed payload shapes pass.

### PR-8: Shadow Compare + Cutover Controls
- Files:
`/Users/sfw/Development/loom/src/loom/engine/verification.py`
`/Users/sfw/Development/loom/src/loom/config.py`
`/Users/sfw/Development/loom/tests/test_verification.py`
- Must ship:
- Dual-run comparison mode (old/new) with diff classification.
- Default-on new engine behind rollback toggle.
- Merge gate:
- Canary metrics meet acceptance thresholds before removing legacy path.

## Definition of Done
This refactor is done when:
1. Verification failures are policy-explicit, explainable, and mostly semantic.
2. Static structural assumptions no longer cause major false negatives.
3. Verification rules apply only where intended (phase-scoped by default).
4. Unconfirmed data is quarantined and remediated without unnecessary full-process aborts.
5. Executive recommendations are always fully confirmed.
6. Process outputs remain high-quality with measurable, sustained verification confidence.
