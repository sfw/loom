# Verification Resilience Implementation Plan (2026-03-09)

## 1) Goal
Reduce verification brittleness so terminal failures become rare, while keeping hard quality protections against garbage-in/garbage-out.

This plan combines:
1. Deterministic integrity gates for non-negotiables.
2. LLM-backed semantic verification for dynamic/non-deterministic content.
3. Reason-code-driven intelligent remediation before terminal failure.

## 2) Review Summary Against Current System

### What already works
1. Tiered verification exists with deterministic Tier1 and LLM Tier2/Tier3 orchestration:
   - `src/loom/engine/verification/gates.py:448`
2. Inconclusive Tier2 fallback to Tier1 warning is already implemented:
   - `src/loom/engine/verification/policy.py:137`
3. LLM verifier already has parse-repair attempts and retry:
   - `src/loom/engine/verification/tier2.py:1458`
4. Retry manager already classifies failures by reason/severity and supports targeted strategies:
   - `src/loom/recovery/retry.py:220`
5. Validity/synthesis gates already support inconclusive soft-pass in some claim cases:
   - `src/loom/engine/orchestrator/validity.py:1319`
   - `src/loom/engine/orchestrator/validity.py:1679`
6. Telemetry already captures verification reason-code counts:
   - `src/loom/engine/orchestrator/telemetry.py:107`

### Proven brittle points (code + incident substantiation)
1. Deterministic path policy can block useful verifier side-effects when tools write non-canonical outputs:
   - Output path keys are treated as mutation targets:
   - `src/loom/engine/runner/policy.py:24`
   - Enforcement path:
   - `src/loom/engine/runner/policy.py:233`
2. Incident evidence: first fact-check attempt failed on forbidden output path:
   - Log sequence 1207: `reason_code=forbidden_output_path` on `validity-scorecard.json`
   - `/Users/sfw/.loom/logs/20260308-154806-cowork-c0ea39ce.events.jsonl`
3. Incident evidence: fact_checker retry succeeded but all 150 claims remained unsupported:
   - Log sequence 1229: `extracted=150, supported=0, insufficient_evidence=150`
   - `/Users/sfw/.loom/logs/20260308-154806-cowork-c0ea39ce.events.jsonl`
4. Incident evidence: synthesis then hard-failed on coverage gate:
   - Log sequence 1241: `coverage_below_threshold` with “no supported claims”
   - `/Users/sfw/.loom/logs/20260308-154806-cowork-c0ea39ce.events.jsonl`
5. Incident summary reported mixed semantic/infra reason codes:
   - Log sequence 1253: reason_codes include `coverage_below_threshold`, `infra_verifier_error`
   - `/Users/sfw/.loom/logs/20260308-154806-cowork-c0ea39ce.events.jsonl`
6. Fact-check retrieval is still mostly lexical + overlap scoring before optional LLM override:
   - `src/loom/tools/fact_checker.py:600`
   - `src/loom/tools/fact_checker.py:654`
7. Placeholder scanning is improved but still candidate-dependent and can regress if fallback behavior broadens:
   - `src/loom/engine/verification/tier1.py:451`
8. Synthesis input gate still has a hard block branch when no supported claims are available:
   - `src/loom/engine/orchestrator/dispatch.py:310`

## 3) Target Design

### A) Keep hard deterministic checks, but narrow them
Hard-fail only for:
1. Contract/integrity/security violations.
2. Non-recoverable deterministic invariants after bounded remediation.
3. High-confidence contradiction/safety policy violations.

Everything else should route to:
1. Intelligent remediation retry, or
2. Pass with warnings + explicit uncertainty metadata.

### B) Use a typed assertion envelope (not a universal claim CSV)
Do not force one `claim,evidence` schema for every subtask.

Use a shared envelope:
1. `assertion_id`
2. `assertion_type` (`fact`, `behavior`, `format`, `safety`, `policy`, etc.)
3. `verdict` (`supported`, `partially_supported`, `contradicted`, `inconclusive`, `failed_contract`)
4. `confidence`
5. `reason_code`
6. `evidence_refs`
7. `remediation_hint`

Task-specific adapters:
1. Research/reporting: claim/evidence snippets.
2. Coding: test results, static checks, command outputs, stack traces, diff provenance.
3. Data tasks: schema checks, reconciliation deltas, row-level constraints.

### C) Add profile-based verification policy with auto-classification
Profiles:
1. `research`
2. `coding`
3. `data_ops`
4. `hybrid`

Auto-classify from:
1. Process metadata/tags.
2. Tool mix.
3. Deliverable types.
4. Prompt/acceptance criteria language.

If profile confidence is low, run `hybrid` policy rather than hard-committing.

### D) Make retries intelligent and bounded
For each failure class:
1. Deterministic-fixable (`forbidden_output_path`, schema mismatches): attempt targeted deterministic repair + model-guided patching.
2. Semantic inconclusive: retrieve additional targeted evidence + rerun verifier.
3. Contradiction high-confidence: block unless explicit corrective rewrite removes contradiction.

Stop conditions:
1. Max attempts reached.
2. No delta across two consecutive attempts.
3. Remaining risk above policy threshold.

## 4) Implementation Plan

## Phase 0: Baseline and Guardrails (P0)
1. Add a run-level reliability dashboard snapshot from existing telemetry:
   - Use `verification_reason_counts` and remediation lifecycle counts.
   - Source: `src/loom/engine/orchestrator/telemetry.py:107`
2. Introduce explicit SLOs:
   - Verifier-caused terminal failure rate.
   - Inconclusive-to-success rescue rate.
   - False-block audit rate.

Deliverable:
1. New reliability metrics section in telemetry run summary payload.

## Phase 1: Policy Refactor to Severity Lanes (P0)
1. Add a central decision matrix mapping `(severity_class, reason_code, profile)` -> action:
   - `block`, `retry_targeted`, `retry_semantic`, `pass_with_warnings`.
2. Wire matrix into:
   - Verification aggregation/fallback:
   - `src/loom/engine/verification/policy.py`
   - Retry classification:
   - `src/loom/recovery/retry.py`
   - Synthesis input gate dispatch behavior:
   - `src/loom/engine/orchestrator/dispatch.py`
3. Keep hard deterministic gates unchanged for security/integrity classes.

Deliverable:
1. Deterministic hard-fail scope is explicit, tested, and minimal.

## Phase 2: Assertion Envelope + Adapters (P1)
1. Add typed assertion model in verification types:
   - `src/loom/engine/verification/types.py`
2. Add adapter emitters:
   - Fact checker assertions from claim verdicts:
   - `src/loom/engine/verification/claims.py`
   - Coding/data adapters from tool outputs and diagnostics:
   - `src/loom/engine/orchestrator/validity.py`
3. Store assertion summaries in verification metadata for downstream gates.

Deliverable:
1. Synthesis and retry logic consume assertions, not tool-specific ad hoc structures.

## Phase 3: Profile Auto-Classification (P1)
1. Add `verification_profile` resolver:
   - New helper under `src/loom/engine/orchestrator/` (or `verification/`).
2. Inputs:
   - Process tags, subtask description, acceptance criteria, tool usage.
3. Output:
   - Profile + confidence + fallback profile.
4. Apply to:
   - Gate thresholds.
   - Retry strategy selection.
   - Synthesis gate strictness.

Deliverable:
1. Ad hoc tasks are handled via confidence-aware hybrid mode rather than brittle assumptions.

## Phase 4: Intelligent Remediation Expansion (P1)
1. Extend remediation playbooks by reason family:
   - `forbidden_output_path`: targeted canonical-path rewrite or scoped adapter behavior.
   - `evidence_gap`/`claim_inconclusive`: targeted evidence retrieval prompts.
2. Ensure retries produce explicit delta plans and stop if no progress.
3. Preserve existing bounded retry caps in config:
   - `src/loom/config.py:287`

Deliverable:
1. Deterministic gate failures attempt intelligent correction before terminal fail where safe.

## Phase 5: Synthesis Gate Robustness by Profile (P1)
1. For research profile:
   - Keep claim coverage thresholds, but treat pure inconclusive bundles as warning-first when contradiction-free.
2. For coding/data profiles:
   - Permit synthesis based on behavior assertions (tests/checks) even with sparse factual claims.
3. Keep explicit uncertainty annotations in final outputs when proceeding under inconclusive status.

Deliverable:
1. `coverage_below_threshold` is no longer a common terminal endpoint for non-research synthesis tasks.

## Phase 6: Rollout Strategy (P0/P1)
1. Shadow mode first:
   - Run old and new decision matrix in parallel, emit diff classification.
2. Gradual enablement:
   - Process allowlist.
   - Then profile-based defaults.
3. Kill switch:
   - Config toggle to revert to current behavior quickly.

Deliverable:
1. Safe migration with measurable quality and reliability impact.

## 5) Concrete File Touch Plan
1. `src/loom/engine/verification/policy.py`
   - Add severity/profile decision matrix and aggregation behavior.
2. `src/loom/recovery/retry.py`
   - Align retry strategy classification with severity/profile matrix and progress-stop rules.
3. `src/loom/engine/orchestrator/dispatch.py`
   - Convert synthesis input gate block/warn behavior to profile-aware policy action.
4. `src/loom/engine/orchestrator/validity.py`
   - Emit and consume typed assertions, add profile-aware synthesis gating thresholds.
5. `src/loom/engine/verification/types.py`
   - Add assertion envelope dataclasses and normalized verdict enums.
6. `src/loom/engine/verification/claims.py`
   - Adapter from fact-check verdicts to assertion envelope.
7. `src/loom/engine/verification/tier1.py`
   - Keep deterministic integrity checks strict; avoid semantic overreach.
8. `src/loom/engine/verification/tier2.py`
   - Preserve model-router path; improve inconclusive metadata for retry targeting.
9. `src/loom/engine/orchestrator/telemetry.py`
   - Add resilience metrics (rescued inconclusive rate, verifier-caused terminal rate).
10. `src/loom/config.py`
   - Add optional profile policy knobs and rollout flags.

## 6) Test Plan
1. `tests/test_verification_policy.py`
   - Decision matrix action coverage by reason/severity/profile.
2. `tests/test_retry.py`
   - Retry strategy + stop conditions + no-progress termination.
3. `tests/test_verification_claims.py`
   - Assertion mapping for `partially_supported` and inconclusive reasons.
4. `tests/test_verification.py`
   - Placeholder false-positive protections remain intact.
   - Existing anchor: `tests/test_verification.py:1424`
5. `tests/orchestrator/test_validity_policy.py`
   - Profile-aware synthesis gate outcomes.
   - Existing inconclusive soft-pass anchor: `tests/orchestrator/test_validity_policy.py:627`
6. `tests/test_research_tools.py`
   - Fact checker opt-in write behavior and paraphrase support remain valid.
   - Anchors: `tests/test_research_tools.py:218`, `tests/test_research_tools.py:239`
7. New profile classification tests:
   - Ad hoc subtasks with mixed tool signatures resolve to `hybrid` when uncertain.

## 7) Tradeoffs
1. More policy complexity in exchange for fewer brittle terminal failures.
2. Slightly higher runtime cost from additional targeted retries and profile logic.
3. Better explainability and auditability because outcomes remain reason-code-driven.

## 8) Risks and Mitigations
1. Risk: Over-permissive soft passes degrade quality.
   - Mitigation: contradiction/safety remain hard blocks; uncertainty must be explicit.
2. Risk: Retry thrash loops.
   - Mitigation: bounded attempts + no-progress cutoff + delta requirements.
3. Risk: Profile misclassification.
   - Mitigation: classifier confidence + automatic hybrid fallback.

## 9) Definition of Done
1. Verifier-caused terminal failures materially drop in shadow + staged rollout.
2. No increase in audited quality regressions.
3. `coverage_below_threshold` and `infra_verifier_error` become uncommon terminal causes.
4. Ad hoc mixed tasks complete with warning/remediation paths rather than brittle hard fails.

