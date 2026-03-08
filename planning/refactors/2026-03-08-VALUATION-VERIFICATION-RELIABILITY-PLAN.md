# Valuation Verification Reliability Refactor Plan (2026-03-08)

## Executive Summary
This revision intentionally narrows scope to avoid broad systemic churn for a single failure class.

Default implementation is **process/tool-local**:
1. strengthen valuation tool capability for precedent transactions,
2. tighten investment process acceptance + remediation behavior,
3. add one low-risk generic safeguard (CSV row-width integrity).

Broader orchestration changes are moved to an **optional future track** and are explicitly not required for this incident.

## Incident Anchor
Failure observed:
1. `reason_code=missing_precedent_transactions` in valuation subtask.
2. DCF/comps present, but precedent transaction evidence not accepted as structured proof.
3. CSV appended rows had schema drift (header/row field count mismatch), reducing verifiability.

## Scope Guardrail (Non-Negotiable)
For this initiative:
1. prioritize fixes in `process + tool` first,
2. allow only minimal verifier hardening with clear cross-process benefit,
3. defer platform-wide recovery-controller redesign unless repeated cross-domain evidence appears.

## Goals
1. Reduce recurrence of `missing_precedent_transactions`.
2. Increase in-run repair success before hard failure.
3. Preserve high quality standards (no criteria downgrades by default).
4. Avoid large orchestrator/planner refactors in the first pass.

## Core Track (Scoped, Recommended)

## A) Add Structured Precedent Modeling to Valuation Tool (P0)
Target: `src/loom/tools/valuation_engine.py`

Changes:
1. Add operation `precedent_transaction_range`.
2. Accept structured transaction comp inputs (deal set, multiple basis, premium assumptions, normalization choices).
3. Return structured outputs:
4. implied EV/equity range,
5. per-share range,
6. assumptions and sensitivity notes,
7. confidence and caveats.
8. Optional report output compatible with existing artifact evidence capture.

Why this helps:
1. Directly closes the capability gap that led to missing evidence.
2. Produces machine-checkable precedent output instead of narrative-only prose.

Knock-ons:
1. Added tool logic and tests.
2. Slight increase in runtime/tool-call complexity in valuation flows.

## B) Tighten Investment Process Contract + Remediation (P0)
Target: `src/loom/processes/builtin/investment-analysis.yaml`

Changes:
1. For `valuation-model` phase, require explicit structured precedent evidence when precedent method is requested.
2. Add/clarify deliverable contract language so precedent evidence must be represented in structured artifact(s), not only narrative.
3. Add reason-code-aligned remediation guidance for:
4. `missing_precedent_transactions`,
5. `csv_schema_mismatch`.
6. Keep quality bar strict; no automatic requirement downgrade in this process.

Why this helps:
1. Makes verifier expectations explicit within the process definition.
2. Improves deterministic remediation for this exact failure class.

Knock-ons:
1. Process spec becomes slightly more detailed.
2. Requires test fixture updates for process-level expectations.

## C) Minimal Generic Safeguard: CSV Row-Width Integrity (P0)
Target: verifier static path in `src/loom/engine/verification/tier2.py` (or shared static-check module)

Changes:
1. For changed CSV files, enforce row field count equals header field count.
2. Emit deterministic reason code `csv_schema_mismatch` with row index hints.
3. Run before semantic LLM verification.

Why this helps:
1. Catches malformed CSV artifacts immediately.
2. Benefits all processes with negligible policy surface expansion.

Knock-ons:
1. Some runs fail earlier but with clearer repair target.

## D) Targeted Remediation Routing Only for Relevant Codes (P1)
Target: `src/loom/engine/orchestrator/remediation.py`

Changes:
1. Add focused playbooks for:
2. `missing_precedent_transactions` -> run precedent modeling + inject structured evidence artifact update.
3. `csv_schema_mismatch` -> repair CSV shape, then re-verify.
4. Bound attempts (for example max 2 per reason code) with no generic retry redesign.

Why this helps:
1. Improves in-process fixing while keeping orchestration changes small.

Knock-ons:
1. Small increase in reason-code routing logic/tests.

## Optional Track (Deferred, Systemic)
These are intentionally deferred and out-of-scope for initial delivery:
1. Global dynamic recovery-controller redesign across all reasons/processes.
2. Global planner capability-feasibility engine for all ad-hoc tasks.
3. Broad verifier evidence sampling overhaul beyond CSV row-shape and current valuation needs.

Trigger to activate optional track:
1. repeated similar failures across multiple process families after Core Track ships.

## Rollout Plan

## Wave 1 (Core P0)
1. Implement `precedent_transaction_range` operation.
2. Update investment-analysis process contract/remediation guidance.
3. Add CSV row-width static check.

Exit Criteria:
1. valuation phase can produce structured precedent evidence.
2. malformed CSV rows are deterministically blocked with `csv_schema_mismatch`.
3. no regression in existing investment-analysis deliverable generation.

## Wave 2 (Core P1)
1. Add targeted remediation routing for `missing_precedent_transactions` and `csv_schema_mismatch`.
2. Verify bounded in-run retries converge or fail with precise terminal diagnostics.

Exit Criteria:
1. increased same-run recovery rate for the two targeted reason codes.
2. reduced terminal failures for valuation subtasks.

## Test Plan
1. `tests/test_investment_tools.py`:
2. add coverage for `precedent_transaction_range` success/failure and edge assumptions.
3. `tests/test_verification.py`:
4. add CSV row-width mismatch detection and reason-code assertion.
5. process/remediation tests:
6. assert targeted playbook routing for `missing_precedent_transactions` and `csv_schema_mismatch`.

## Metrics
1. `missing_precedent_transactions` frequency (pre vs post).
2. `csv_schema_mismatch` detection count and remediation success rate.
3. valuation-subtask hard-fail rate after at least one remediation attempt.
4. overall investment-analysis completion rate.

## Risks and Mitigations
1. Risk: precedent operation introduces weak assumptions.
2. Mitigation: require explicit assumptions in output + strict tests around input validation.
3. Risk: process contract changes become too rigid.
4. Mitigation: keep requirements method-specific and phase-scoped only.
5. Risk: CSV check creates early failures.
6. Mitigation: targeted remediation path to auto-repair when possible.

## Decision Log
1. Chose scoped approach first to avoid overengineering.
2. Kept only one generic verifier change with broad safety value.
3. Deferred platform-wide recovery/planner refactors pending cross-process evidence.
