# Process Validity and Evidence Rigor Refactor Plan (2026-03-05)

## Executive Summary
The current system can pass structurally valid outputs that are factually invalid. This plan introduces a unified validity architecture for both ad hoc and defined processes so downstream synthesis can only consume verified claims, while missing evidence causes targeted pruning and uncertainty labeling rather than blanket hard-fail behavior.

Primary outcomes:
1. Unsupported or contradicted claims are removed before they reach synthesis inputs.
2. Final outputs cannot pass when critical claims are unverified or contradicted.
3. Resume/retry behavior preserves verification strictness and contracts.
4. Every claim in final outputs is auditable to source evidence.

## Incident Anchor (What This Plan Must Fix)
The `cowork-3df9d4dd` run exposed four failure classes:
1. Verification strictness regressed on resumed execution.
2. Final synthesis passed without claim-level grounding checks.
3. Required fact-check capability existed but was not enforced.
4. Post-run mutation and write-path evidence gaps reduced auditability.

This plan treats these as symptoms of a broader design gap: verification is phase-level and structural, not claim-level and lineage-enforced.

## Goals
1. Enforce claim-level validity end-to-end, not only deliverable presence and format.
2. Make missing/uncertain evidence non-fatal for intermediate steps via pruning semantics.
3. Block final outputs only when critical validity thresholds are not met.
4. Apply the same rigor to ad hoc and defined process execution paths.
5. Make validity decisions observable, reproducible, and queryable.

## Non-Goals
1. Guarantee objective truth for every claim in open-world domains.
2. Replace domain-specific judgment with rigid deterministic rules.
3. Fully redesign process authoring UX in this iteration.

## System Invariants (Non-Negotiable)
1. No contradicted claim can appear in final output.
2. No unverified critical claim can appear in final output.
3. Unverified non-critical claims must be pruned or explicitly labeled uncertain.
4. Resume/retry must preserve effective verification policy and tier.
5. Every final claim must map to at least one evidence record with provenance metadata.
6. Verification outcomes must be persisted and recoverable after restart.

## Design Principles
1. `Garbage in` is handled early through claim extraction and pruning.
2. Missing confirmation degrades confidence and scope, not always task status.
3. Final synthesis consumes a sanitized claim graph, not raw narrative artifacts.
4. Process contracts define required rigor; runtime enforces minimum policy floors.

## Target Architecture

## 1) Claim-Centric Validity Layer
Introduce claim objects as first-class runtime entities across all workflows.

Claim lifecycle states:
1. `extracted`
2. `supported`
3. `contradicted`
4. `insufficient_evidence`
5. `stale`
6. `pruned`
7. `waived` (explicit, audited exception only)

Each claim carries:
1. canonical text
2. claim type (`numeric`, `date`, `entity_fact`, `forecast_assumption`, `qualitative`)
3. criticality (`critical`, `important`, `optional`)
4. source time scope (`as_of`, `period`, `forecast_horizon`)
5. linked evidence and verification outcomes

## 2) Non-Fatal Pruning and Sanitization
For intermediate phases:
1. `supported` claims flow downstream.
2. `insufficient_evidence` claims are removed from downstream context or rewritten as explicit uncertainty.
3. `contradicted` claims are removed and logged as remediation items.
4. Phase can still pass if required coverage thresholds are met post-prune.

For synthesis/final phases:
1. Critical-claim support threshold is strict (default 100%).
2. Contradicted-claim count must be zero.
3. Optional claims may be dropped without failing the run.

## 3) Synthesis Input Gate
Before any final recommendation/report generation:
1. Build a `verified context bundle` from supported claims only.
2. Exclude raw artifacts that contain unresolved claims.
3. Fail synthesis if minimum critical coverage is not met.

## 4) Validity Policy Engine
Add a unified policy evaluator that runs for ad hoc and defined processes:
1. policy floors by intent/domain (`investment`, `medical`, `legal`, etc.)
2. per-phase thresholds
3. claim criticality rules
4. hard-fail criteria limited to high-risk cases

## Contract Changes (Ad Hoc and Defined)

## 1) Process Schema Extensions
Extend process definitions with a `validity_contract` (or equivalent nested contract under `prompt_contracts`):
1. `enabled`
2. `claim_extraction.enabled`
3. `critical_claim_types`
4. `min_supported_ratio`
5. `max_unverified_ratio`
6. `max_contradicted_count`
7. `prune_mode` (`drop`, `rewrite_uncertainty`)
8. `require_fact_checker_for_synthesis`
9. `final_gate.enforce_verified_context_only`

## 2) Default Policy Injection
1. Ad hoc generation path auto-injects baseline validity contracts by intent.
2. Defined process loader applies default policy floors if contract is missing.
3. Built-in process YAMLs are migrated to explicit validity contracts.

## 3) Contract Linting
Add install/runtime lint checks:
1. reject final synthesis phases with no validity gate
2. reject high-risk intents with tier below minimum floor
3. reject contracts that allow contradicted claims in final outputs

## Runtime and Persistence Refactor

## 1) Resume/Retry Correctness (P0)
Persist and hydrate these subtask fields in task state:
1. `model_tier`
2. `verification_tier`
3. `acceptance_criteria`
4. validity contract snapshot/hash

On resume:
1. reconcile subtask policy with process phase by `phase_id`
2. if missing/mismatch, backfill from process definition and emit reconciliation event
3. enforce minimum synthesis verification tier at execution time

## 2) Evidence Capture Completeness
Include write-path evidence in both state evidence and verifier excerpts:
1. `write_file`
2. `document_write`
3. transformed/edited artifacts used in final synthesis

This closes current blind spots where final narrative mutations are not auditable.

## 3) Claim and Verification Storage
Add SQLite entities:
1. `artifact_claims`
2. `claim_evidence_links`
3. `claim_verification_results`
4. `artifact_validity_summaries`

Extend existing rows/metadata for:
1. `subtask_attempts` with policy snapshot and validity summary
2. remediation tables with claim references

## 4) Enforcement of Required Tools
For phases/contracts that require fact grounding:
1. `fact_checker` execution must be observed before phase pass
2. absence of required verification tools yields `verification_outcome=fail` for that phase

## Verification Logic Refactor

## 1) Multi-Stage Verification Pipeline
For each narrative artifact:
1. claim extraction
2. evidence retrieval/linking
3. claim-level verification
4. pruning/sanitization
5. phase-level gate evaluation

## 2) Reason Codes and Deterministic Outcomes
Standardize reason codes:
1. `claim_supported`
2. `claim_contradicted`
3. `claim_insufficient_evidence`
4. `claim_stale_source`
5. `required_verifier_missing`
6. `coverage_below_threshold`

## 3) Final Gate Logic
Default high-rigor gate for recommendation outputs:
1. contradicted critical claims = 0
2. unsupported critical claims = 0
3. supported all-claims ratio >= configured floor
4. explicit evidence appendix generated

## Ad Hoc and Defined Process Parity
Use one execution/verification path for both:
1. same claim engine
2. same policy evaluator
3. same persistence schema
4. same observability events

Differences should only come from contract values, not runtime semantics.

## Observability and Auditability

## 1) New Events
Emit events for:
1. claim extraction completed
2. claim verification summary
3. prune actions
4. synthesis input gate decision
5. policy reconciliation on resume

## 2) Run-Level Validity Scorecard
Generate machine-readable and human-readable summaries:
1. claim counts by status
2. critical claim coverage
3. contradicted claim count
4. pruned claim list with reason
5. final trust score

## 3) Output Provenance Footer
Final outputs include:
1. analysis timestamp
2. source time windows
3. validity summary metrics
4. link/path to verification report

## Rollout Plan

## Wave P0 (Immediate Correctness and Safety)
Scope:
1. persist/hydrate verification fields on resume
2. synthesis min-tier enforcement
3. enforce required fact-check run for synthesis when contract says so
4. include write-file evidence capture

Exit criteria:
1. resumed runs cannot silently downgrade verification strictness
2. unsupported critical claim test fixture fails final gate

## Wave P1 (Claim Engine and Non-Fatal Pruning)
Scope:
1. claim extraction + claim verification data model
2. prune/uncertainty rewrite for intermediate artifacts
3. verified-context bundle for synthesis
4. standardized reason codes and scorecards

Exit criteria:
1. intermediate phases can continue with pruned outputs
2. final synthesis reads only verified context

## Wave P2 (Contract Parity and Authoring Enforcement)
Scope:
1. validity contract schema and linting
2. ad hoc default contract injection by intent
3. defined process migration to explicit validity policies

Exit criteria:
1. all built-in high-risk processes declare validity contracts
2. ad hoc and defined runs enforce identical policy semantics

## Wave P3 (Hardening and Governance)
Scope:
1. dashboards/alerts on contradiction escapes
2. regression corpus with intentionally wrong claims
3. policy tuning via measured false-positive/false-negative tradeoffs

Exit criteria:
1. contradiction escape rate below SLO
2. reproducible postmortem pack available for every failed gate

## Wave P4 (Integrity and High-Risk Controls)
Scope:
1. immutable artifact sealing (hash/signature per phase artifact) and synthesis-time seal validation
2. strict temporal-consistency gates (`as_of` alignment, stale-source policy, cross-artifact date conflict detection)
3. numeric lineage enforcement (every final numeric claim maps to source row + transform chain)
4. adversarial disconfirmation pass for top critical claims before final gate
5. high-risk approval mode requiring explicit human review for investment/medical/legal outputs when configured

Exit criteria:
1. post-verification artifact mutation is detected and blocks final synthesis
2. temporal conflict fixtures fail with deterministic reason codes
3. sampled final outputs show 100% numeric lineage coverage for critical metrics
4. disconfirmation pass runs on every high-risk final output and emits audit artifacts
5. high-risk approval policy is enforceable by configuration without code changes

## Testing Strategy

## Unit Tests
1. task state round-trip preserves verification and validity contract fields
2. claim status transitions and prune semantics are deterministic
3. required-verifier enforcement triggers failures correctly
4. artifact seal generation/validation is stable across resume/retry
5. temporal-consistency checker detects stale/conflicting dates
6. numeric lineage validator rejects orphaned or opaque computed values

## Integration Tests
1. resume run preserves tier-2+ verification behavior
2. recommendation phase fails when unsupported critical numeric claim is injected
3. missing evidence causes pruning, not full run failure, in intermediate phases
4. final synthesis uses verified-context bundle only
5. sealed artifacts mutated after verification are blocked at synthesis gate
6. disconfirmation pass flags contradicted critical claims and prevents final pass

## End-to-End Regression Tests
1. replay `cowork-3df9d4dd`-style fixture with contradictory claims and date drift
2. ensure final gate blocks invalid output and emits actionable remediation reasons
3. verify write-path evidence exists for final artifacts
4. verify high-risk approval mode pauses/blocks release until explicit decision is recorded

## Success Metrics (SLOs)
1. `critical_claim_escape_rate` < 0.1%
2. `contradicted_claim_escape_rate` = 0 for high-risk intents
3. `resume_policy_drift_incidents` = 0
4. `auditable_final_output_rate` > 99%
5. `nonfatal_prune_recovery_rate` measured and trending upward
6. `post_verification_mutation_block_rate` = 100%
7. `critical_numeric_lineage_coverage` = 100%
8. `temporal_conflict_escape_rate` = 0 for high-risk intents

## Implementation Map (Code Areas)
1. `src/loom/state/task_state.py` for persistence/hydration correctness
2. `src/loom/engine/orchestrator.py` for synthesis tier floors and gate orchestration
3. `src/loom/engine/runner.py` for required verifier enforcement
4. `src/loom/engine/verification.py` for claim-level verification and reason codes
5. `src/loom/state/evidence.py` for write-path evidence capture
6. `src/loom/state/schema.sql` and `src/loom/state/memory.py` for claim tables and APIs
7. `src/loom/processes/schema.py` for validity contract schema/lint
8. `src/loom/tui/app.py` and ad hoc synthesis normalization for default contract injection
9. `src/loom/processes/builtin/*.yaml` for defined-process policy parity

## Risks and Mitigations
1. Risk: over-pruning removes useful context.
2. Mitigation: claim criticality levels plus uncertainty rewrite mode.
3. Risk: higher latency/cost from claim verification.
4. Mitigation: cache claim fingerprints and incremental verification.
5. Risk: false contradictions from weak extraction.
6. Mitigation: confidence thresholds and human-review escape hatch for ambiguous claims.

## Definition of Done
1. Final recommendation/report outputs are provably traceable to supported claims.
2. Invalid claims are pruned upstream and do not leak into synthesis inputs.
3. Resume/retry never weakens verification enforcement.
4. Ad hoc and defined processes share the same validity guarantees.
