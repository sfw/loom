# Placeholder Hard-Invariant Recovery Refactor Plan (2026-03-06)

## Executive Summary
The current placeholder failure path is too blunt: placeholder-like tokens (for example `N/A` in evidence cells) can trigger a hard invariant failure that blocks critical-path progress, even when the issue is recoverable through targeted remediation.

This refactor separates `structural/safety hard failures` from `recoverable content completeness failures`, and upgrades failure-resolution from "planner-only hints" to a deterministic confirm-or-prune workflow for placeholder findings.

Primary outcomes:
1. Placeholder-linked gaps route to `UNCONFIRMED_DATA` remediation instead of immediate hard abort.
2. System either confirms missing placeholder data from evidence or removes the unverifiable claim/row/field.
3. Critical-path runs continue under bounded remediation policy (`confirm_or_prune_then_queue` / `queue_follow_up`) rather than dying on recoverable content defects.
4. Hard failures remain hard for true integrity/safety violations.

## Incident Anchor
Run: `cowork-ab1006f1`  
Log: `~/.loom/logs/<timestamp>-cowork-<id>.events.jsonl`

Observed behavior:
1. `no-placeholders` rule is hard and includes `N/A`.
2. Deliverables contained `N/A` in evidence/status columns:
   - `/path/to/workspace/test-strategy.md:266`
   - `/path/to/workspace/security-and-compliance-plan.md:104`
3. Tier-1 set `reason_code=hard_invariant_failed`, causing retry strategy to become `generic`.
4. `failure_resolution_plan` ran multiple times, but remediation attempts remained `0` and run terminated on critical path.

## Why Current Behavior Fails

## 1) Classification Collapses Recoverable and Non-Recoverable Failures
Current Tier-1 behavior marks any deterministic failure as hard invariant:
1. deterministic checks aggregate into `hard_failures`.
2. any `hard_failures` -> `reason_code=hard_invariant_failed`, `severity_class=hard_invariant`.

Effect: recoverable placeholder content defects are treated like missing deliverables, permission denials, or safety violations.

## 2) Retry Strategy Never Reaches Confirm-or-Prune
`RetryManager.classify_failure(...)` maps `hard_invariant_failed` to `RetryStrategy.GENERIC`.  
Confirm-or-prune flow is activated for `UNCONFIRMED_DATA`, not for hard invariants.

## 3) Failure Resolution Is Planner-Only
`_plan_failure_resolution(...)` returns a textual plan saved into attempt context, but no deterministic "execute plan items" stage exists. Success depends on subsequent model behavior.

## Goals
1. Preserve strict hard-fail semantics for true safety/integrity failures.
2. Reclassify placeholder/content incompleteness as recoverable (`UNCONFIRMED_DATA`) when policy permits.
3. Make placeholder remediation deterministic and auditable:
   - locate exact placeholder instances,
   - attempt evidence-backed fill,
   - prune unverifiable data when fill fails.
4. Keep critical-path progress moving under bounded remediation policy.
5. Avoid reintroducing garbage by requiring post-remediation deterministic recheck.

## Non-Goals
1. Eliminate all process-level hard rules.
2. Infer objective truth for unsupported claims.
3. Redesign all verification tiers in one pass.

## Invariants (Post-Refactor)
1. Safety/integrity violations remain hard and blocking.
2. Placeholder content defects must end in one of:
   - evidence-backed concrete value, or
   - explicit removal/pruning of unverifiable content.
3. No placeholder token accepted in final deliverables.
4. Remediation remains bounded by retry/iteration budgets.
5. Every prune/fill action is logged with file/line and reason.

## Proposed Design

## 1) Failure Taxonomy Split
Introduce two deterministic failure classes:
1. `hard_integrity` (non-recoverable in-loop):
   - tool safety/integrity failures (permission denied, path escape, blocked host),
   - syntax invalid output,
   - missing required deliverable files,
   - synthesis input integrity preconditions (when strict).
2. `recoverable_placeholder` (confirm-or-prune):
   - regex placeholder hits in content fields,
   - placeholder-like tokens in required semantic fields (configurable),
   - incomplete deliverable markers tied to missing data.

Implementation direction:
1. Extend rule metadata interpretation to support `remediation_mode` and `failure_class`.
2. Keep `enforcement: hard` available, but allow `hard+recoverable` semantics for specific rule types.

## 2) Placeholder-Aware Rule Semantics
For `type=regex` rules targeting deliverables:
1. Add optional rule config:
   - `remediation_mode: confirm_or_prune`
   - `placeholder_policy`:
     - token list (`TBD`, `TODO`, `N/A`, etc.)
     - allowed columns/contexts (for example allow `N/A` only in explicitly nullable fields)
     - disallowed semantic fields (evidence, target, owner, acceptance, thresholds)
2. On match:
   - produce structured metadata with precise matches:
     - file path
     - line number
     - matched token
     - row/column hint for table/csv
   - set reason code to a recoverable unconfirmed code (for example `incomplete_deliverable_placeholder`) when eligible.

## 3) Deterministic Confirm-or-Prune Engine
Add a remediation executor stage for placeholder findings:
1. Input: structured placeholder findings + expected deliverables + evidence sources.
2. For each finding:
   - attempt deterministic fill:
     - cross-reference known evidence artifacts and prior verified claims.
   - if no support, prune:
     - remove cell/field/row/claim depending on format and policy.
3. Emit remediation action log:
   - `filled` with source evidence id(s), or
   - `pruned` with reason (`unsupported_placeholder_data`).
4. Re-run Tier-1 deterministic verification after edits.

Notes:
1. This engine is narrow and deterministic; it does not perform broad content generation.
2. Model-based retries still exist, but deterministic sanitization runs first for placeholder class.

## 4) Failure-Resolution Integration Changes
Current flow:
1. classify
2. generate planner text
3. retry model execution

New flow for placeholder-unconfirmed class:
1. classify as `UNCONFIRMED_DATA`.
2. call deterministic confirm-or-prune executor first.
3. if resolved, continue phase.
4. if unresolved, escalate to bounded model remediation with structured findings embedded.
5. on exhaustion, apply critical-path behavior policy (`confirm_or_prune_then_queue` or `queue_follow_up`) instead of unconditional block, when failure class is recoverable placeholder.

## 5) Critical-Path Policy Refinement
Process-level policy today may be `critical_path_behavior: block`.

Refactor:
1. Keep `block` default for `hard_integrity`.
2. Permit per-failure-class override for `recoverable_placeholder`:
   - default recommended: `confirm_or_prune_then_queue`.
3. Ensure final synthesis gate still blocks if unresolved placeholder artifacts remain.

## Schema and Contract Changes

## 1) Verification Rule Extensions
Add optional fields to `VerificationRule`:
1. `failure_class`: `hard_integrity | recoverable_placeholder | semantic`
2. `remediation_mode`: `confirm_or_prune | targeted_remediation | none`
3. `context_constraints`:
   - allowed/disallowed column names
   - file globs
   - markdown-table semantic hints

Backward compatibility:
1. missing fields preserve existing behavior.

## 2) VerificationResult Metadata Contract
When placeholder class fires, metadata must include:
1. `remediation_required: true`
2. `remediation_mode: confirm_or_prune`
3. `missing_targets`
4. `placeholder_findings`: list of structured match entries
5. `candidate_fill_sources`

## Runtime Refactor (Phased)

## W0: Classification and Metadata Wiring
Files:
1. `<repo-root>/src/loom/engine/verification.py`
2. `<repo-root>/src/loom/recovery/retry.py`
3. `<repo-root>/src/loom/processes/schema.py`

Tasks:
1. Emit recoverable placeholder reason codes instead of `hard_invariant_failed` when configured.
2. Populate structured placeholder metadata.
3. Ensure retry classifier maps these to `UNCONFIRMED_DATA`.

## W1: Deterministic Placeholder Scanner and Findings
Files:
1. `<repo-root>/src/loom/engine/verification.py`
2. New module: `<repo-root>/src/loom/engine/placeholder_remediation.py`

Tasks:
1. Build tokenizer/matcher with file+line localization.
2. Add table/csv field context extraction where possible.
3. Produce stable findings object for remediation and logs.

## W2: Confirm-or-Prune Executor
Files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. New module: `<repo-root>/src/loom/engine/placeholder_remediation.py`

Tasks:
1. Execute deterministic fill-or-prune before model retry for placeholder-unconfirmed failures.
2. Record action outcomes and attach to attempt metadata.
3. Re-run deterministic verification immediately.

## W3: Critical-Path Behavior by Failure Class
Files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. `<repo-root>/src/loom/processes/schema.py`
3. Process YAMLs using strict placeholder rules.

Tasks:
1. Add behavior branch keyed by failure class.
2. Keep hard integrity as blocking.
3. Route recoverable placeholder failures through queue/follow-up policy.

## W4: Process Rule Cleanup (`N/A` and Context)
Files:
1. Relevant process YAMLs under `~/.loom/processes/*` and/or builtin process definitions.

Tasks:
1. Replace broad token matching with context-aware placeholder policy.
2. Decide explicit policy for `N/A`:
   - either disallow globally, or
   - allow in explicitly nullable non-critical columns.
3. Document policy examples for process authors.

## Test Plan

## Unit Tests
1. Placeholder in disallowed field -> recoverable unconfirmed classification.
2. Placeholder in allowed nullable context -> no failure.
3. Hard integrity failures still map to hard invariant.
4. Retry classifier maps recoverable placeholder reason codes to `UNCONFIRMED_DATA`.

## Integration Tests
1. Reproduce `N/A` in evidence column case:
   - deterministic scanner finds exact line,
   - confirm-or-prune removes or fills,
   - run proceeds without critical-path abort.
2. Confirm unresolved placeholders after budget exhaustion:
   - remediation queued/follow-up behavior matches policy.
3. Final synthesis gate rejects unresolved placeholders.

## Regression Tests
1. Existing hard safety/integrity checks remain blocking.
2. No increase in false passes with placeholder tokens present.

## Observability
Add events:
1. `placeholder_findings_extracted`
2. `placeholder_confirm_or_prune_started`
3. `placeholder_filled`
4. `placeholder_pruned`
5. `placeholder_remediation_unresolved`

Run summary additions:
1. count of placeholder findings
2. fills vs prunes
3. unresolved placeholder count

## Risks and Mitigations
1. Risk: Over-pruning useful data.
   - Mitigation: prune at minimal unit (cell/field first), preserve audit trail.
2. Risk: Misclassification of hard failures as recoverable.
   - Mitigation: explicit failure-class matrix and safety-first defaults.
3. Risk: Policy complexity for process authors.
   - Mitigation: provide sensible defaults and cookbook examples.

## Rollout Strategy
1. Behind feature flag:
   - `verification.placeholder_confirm_or_prune_enabled`.
2. Start with one process family (`prd-software-design`) as canary.
3. Collect metrics on:
   - reduced critical-path aborts from placeholder-only issues,
   - no increase in invalid final outputs.
4. Expand to other built-in process packages.

## Definition of Done
1. Placeholder-only defects no longer hard-abort critical path by default.
2. Every placeholder finding ends in deterministic fill or prune action.
3. `failure_resolution` integrates with executable remediation rather than guidance-only for this class.
4. Existing hard safety/integrity behavior remains unchanged.
5. Incident scenario equivalent to `cowork-ab1006f1` completes without garbage placeholder residue.
