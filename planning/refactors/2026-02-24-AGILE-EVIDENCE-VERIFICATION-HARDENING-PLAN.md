# Agile Evidence Verification Hardening Plan (2026-02-24)

## Objective
Harden process execution so Loom is:
1. Agile when data is unavailable or only partially confirmable.
2. Persistent in evidence gathering/remediation when data may be available but currently unverified.
3. Strict only on true hard invariants, not brittle verifier contradictions or stale context artifacts.

## Problem Summary
Observed in `marketing-strategy` run `#200963`:
1. The run terminated on a critical-path verification fail in `customer-segmentation`.
2. The verifier reported `[MISSING]` placeholder presence while execution evidence indicated no match in canonical files.
3. The same phase also contained explicit unsupported claims (`UNSUPPORTED_NO_EVIDENCE`) that should trigger targeted remediation, not indiscriminate full-run failure.
4. Current critical-path handling can over-block when failures are unconfirmed/contradictory.

## Target End State
1. Verification outcomes are routed by structured semantics:
- `hard_invariant` -> block/fail.
- `semantic_unconfirmed` -> targeted remediation, continue when policy allows.
- `inconclusive/infra` -> verification retry or downgrade to warning when deterministic checks pass.
2. Critical-path behavior is process-contract-driven (`block`, `confirm_or_prune_then_queue`, `queue_follow_up`).
3. Verifier contradiction checks prevent false hard-fails when deterministic evidence disagrees.
4. Unavailable data is explicitly represented as uncertainty and tracked via remediation queue.
5. Remediation queue executes persistently with bounded retries/backoff and clear terminal states.

## Scope
Core runtime:
1. `/Users/sfw/Development/loom/src/loom/engine/verification.py`
2. `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
3. `/Users/sfw/Development/loom/src/loom/recovery/retry.py`
4. `/Users/sfw/Development/loom/src/loom/config.py`

Process/schema and prompts:
1. `/Users/sfw/Development/loom/src/loom/processes/schema.py`
2. `/Users/sfw/Development/loom/src/loom/prompts/templates/verifier.yaml`
3. `/Users/sfw/Development/loom/src/loom/processes/builtin/marketing-strategy.yaml`

State/evidence context:
1. `/Users/sfw/Development/loom/src/loom/state/task_state.py`
2. `/Users/sfw/Development/loom/src/loom/state/evidence.py`

Tests:
1. `/Users/sfw/Development/loom/tests/test_verification.py`
2. `/Users/sfw/Development/loom/tests/test_orchestrator.py`
3. `/Users/sfw/Development/loom/tests/test_retry.py`
4. `/Users/sfw/Development/loom/tests/test_processes.py`

## Workstreams

### W1: Failure Taxonomy and Routing Hardening
1. Extend structured routing coverage in `RetryManager.classify_failure(...)` for high-frequency semantic failures:
- `incomplete_deliverable_placeholder`
- `incomplete_deliverable_content`
- `unsupported_claims_and_incomplete_evidence`
- `insufficient_evidence`
- `recommendation_unconfirmed`
2. Route these to `RetryStrategy.UNCONFIRMED_DATA` by default instead of generic retry.
3. Preserve strict routing for `hard_invariant_failed`.

Exit criteria:
1. These reason codes map deterministically to targeted remediation strategies.
2. Generic retries are reduced for semantic-evidence failures.

### W2: Verifier Contradiction Guard
1. Add a contradiction detector in verification flow:
- if verifier asserts placeholder/TODO-style failure, run deterministic canonical scan over changed deliverables/current phase files.
2. If deterministic scan contradicts verifier claim:
- return `outcome=fail` only if a deterministic hard check confirms it.
- otherwise downgrade to `inconclusive` or `pass_with_warnings` with explicit metadata (`contradiction_detected=true`), then retry verification-only.
3. Emit instrumentation event for contradiction path to enable regression tracking.

Exit criteria:
1. Placeholder false-positives cannot hard-fail a run without deterministic confirmation.
2. Contradictions are observable via event telemetry.

### W3: Critical-Path Behavior from Process Contract
1. Use `verification.remediation.critical_path_behavior` from process definitions in orchestrator logic.
2. Support values:
- `block` (current default)
- `confirm_or_prune_then_queue`
- `queue_follow_up`
3. Implement handling:
- `confirm_or_prune_then_queue`: run confirm/prune; if still unresolved and non-hard-invariant, queue blocking remediation and continue according to policy.
- `queue_follow_up`: never immediate hard-block on unconfirmed semantic failures; queue and continue.
4. Keep `hard_invariant` as unconditional block.

Exit criteria:
1. Critical-path semantic uncertainty is policy-controlled rather than globally hard-blocked.
2. Hard invariant failures still terminate deterministically.

### W4: First-Class Uncertainty Contract
1. Extend verifier output metadata usage:
- `remediation_required`
- `remediation_mode`
- `missing_targets`
- `unverified_claim_count`
- `verified_claim_count`
- `supporting_ratio`
2. Ensure prompt and parser preserve/normalize these fields.
3. Codify policy:
- explicit `UNSUPPORTED_NO_EVIDENCE` is allowed in supporting sections when thresholds are respected.
- recommendation/executive sections remain fully confirmed.

Exit criteria:
1. Unavailable data can be represented safely without immediate process abortion.
2. Missing targets are surfaced for persistent remediation.

### W5: Persistent Remediation Execution
1. Reuse existing remediation queue but tighten SLA behavior:
- stronger dedupe by `(subtask_id, reason_code, strategy)`
- bounded retry/backoff behavior
- explicit expiration behavior for stale queue items
2. Ensure queued work can be processed during execution and at finalization boundaries.
3. Ensure unresolved blocking remediation is the only remediation-based terminal failure.

Exit criteria:
1. Unconfirmed data produces actionable queue items that are retried persistently.
2. Task failure due to remediation is explicit and attributable.

### W6: Process Rule Updates (Marketing Strategy)
1. Update `marketing-strategy` placeholder regex coverage to include `[MISSING]` explicitly.
2. Confirm metadata fields are declared for verifier output contract.
3. Keep sources-cited and no-placeholder checks, but align enforcement with semantic/targeted remediation behavior.

Exit criteria:
1. Placeholder detection is deterministic and not dependent on free-text verifier interpretation.
2. Process contract supports robust remediation metadata.

### W7: Observability and Regression Protection
1. Add/extend events for:
- contradiction detected
- unconfirmed data queued (critical vs non-critical)
- remediation terminal state
2. Add targeted tests and golden fixtures for:
- contradictory placeholder claim
- unsupported-but-labeled claims
- critical-path queue/confirm-or-prune behaviors

Exit criteria:
1. Regressions in false-fail vs bad-pass tradeoff are measurable.
2. Historical failure mode is covered by automated tests.

## Rollout Strategy

### Phase 1: Safe Internal Refactor
1. Implement routing + contradiction guard + metadata normalization.
2. Keep default critical-path behavior as `block` unless process overrides.

### Phase 2: Process Policy Activation
1. Enable `confirm_or_prune_then_queue` (or chosen value) for selected built-in processes.
2. Run shadow checks and compare outcome diffs.

### Phase 3: Broader Adoption
1. Apply policy pattern to other built-ins.
2. Promote defaults after telemetry validates reduced false-fails without bad-pass increase.

## Acceptance Criteria
1. Historical failure class:
- runs like `#200963` do not fail hard solely on contradicted placeholder claims.
2. Safety:
- hard invariant failures still block immediately.
3. Agility:
- unsupported-but-labeled supporting content can pass as `partial_verified`/`pass_with_warnings` with queued remediation when policy allows.
4. Persistence:
- remediation queue records and executes missing-target follow-ups with bounded retries.
5. Test coverage:
- new tests cover contradiction guard, reason-code routing, critical-path behavior modes, and queue outcomes.

## Risks and Mitigations
1. Risk: overly permissive downgrade could hide real failures.
- Mitigation: never downgrade deterministic hard invariant failures; require deterministic contradiction proof before downgrade.
2. Risk: policy complexity in critical path.
- Mitigation: explicit defaults, strict schema validation, focused tests per mode.
3. Risk: verifier metadata quality variance.
- Mitigation: parser normalization + deterministic fallbacks + verification-only retries on inconclusive outputs.

