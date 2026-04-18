# Tool Method Failure Resilience Plan (2026-04-04)

## Objective
Broaden Loom's resilience model so ordinary tool failures are treated as failed methods to route around, not objective-level failures. Hard-blocking outcomes should remain reserved for safety, policy, integrity, and true exhaustion of viable alternatives.

This plan follows the earlier runtime-availability refactor and addresses the next gap:
1. missing-binary and capability failures now replan correctly
2. many non-capability tool failures still fall through as blocking verifier failures
3. some ordinary write/runtime faults are over-classified as hard safety/integrity failures

## Desired Behavior
For process runs, Loom should prefer this order of interpretation:
1. if the tool failure is a safety/policy/integrity fault, block
2. else if the tool failure is an ordinary method/runtime failure, retry or replan around it
3. only stop when alternative methods are exhausted or process policy explicitly forbids fallback

Examples that should usually be recoverable:
1. target website unavailable
2. transient fetch/network failure
3. parser/CLI command failed
4. document/file write failed for non-policy reasons
5. local service/tool timed out

Examples that should remain hard/blocking:
1. sandbox escape / forbidden path
2. blocked internal-host access / SSRF policy
3. destructive action without required approval
4. artifact integrity or sealed-output violations
5. objective-level impossibility after fallback methods are exhausted

## Current Gaps
1. `DeterministicVerifier._classify_tool_failure(...)` still defaults many tool failures to non-advisory blocking failures.
2. `_is_hard_safety_or_integrity_failure(...)` currently treats broad write/runtime strings like `permission denied` too aggressively.
3. `safety_integrity_only` often downgrades tool failures to advisory/pass, which avoids hard failure but does not actively drive replanning around the failed method.
4. retry routing has special treatment for capability-unavailable failures, but not a broader family of recoverable method failures.

## Design

### 1) Introduce explicit recoverable method-failure reason codes
Add structured reason codes for ordinary tool-path failures, for example:
1. `tool_method_failed`
2. `tool_transient_failure`
3. `tool_upstream_unavailable`
4. `tool_write_retryable`
5. `tool_runtime_retryable`

These should not imply that the subtask objective is impossible.

### 2) Narrow the hard safety/policy/integrity classifier
`_is_hard_safety_or_integrity_failure(...)` should focus on true policy/sandbox/integrity faults, not generic runtime issues.

Keep hard:
1. blocked hosts / SSRF / internal-network denial
2. workspace escape / forbidden path / sealed-artifact violations
3. explicit safety violation markers

Do not hard-classify by default:
1. `permission denied`
2. `operation not permitted`
3. `read-only file system`
4. generic timeouts
5. upstream 4xx/5xx/transport failures

### 3) Add a resilient verifier tool-success policy
Introduce a new tool-success policy:
1. `method_resilient`

Semantics:
1. hard safety/policy/integrity failures still block
2. recoverable method failures remain failing verification outcomes, but carry structured reason codes that route into targeted retry/replan instead of broad hard failure
3. success is not granted just because a tool failed; the system should retry/replan the subtask objective

This differs from:
1. `all_tools_hard`: too brittle
2. `safety_integrity_only`: too permissive because many failures become advisory/pass
3. `development_balanced`: specialized for coding/build verification helpers

### 4) Make ad hoc/process defaults more resilient
Update the process-schema fallback for ad hoc processes to prefer `method_resilient` instead of `safety_integrity_only`.

Rationale:
1. ad hoc/process runs should try alternate methods instead of passing over failed attempts
2. preserving failure as recoverable is better than silently accepting missing work

### 5) Route recoverable method failures into replan-focused retry handling
Retry/recovery should treat the new reason-code family similarly to capability-unavailable failures:
1. preserve the subtask objective
2. avoid reusing the failing method blindly
3. instruct the model to try alternate tools, sources, or write strategies

### 6) Keep API/run failure analysis accurate
Run failure analysis should distinguish:
1. hard invariant / contract failure
2. recoverable method failure that exhausted retries

This avoids misleading operators into thinking the first failed tool proved the task impossible.

## Implementation Workstreams

### Workstream A: Verification classification
Primary files:
1. `src/loom/engine/verification/tier1.py`
2. `src/loom/engine/verification/types.py`
3. `src/loom/engine/verification/policy.py`

Tasks:
1. add recoverable method-failure reason codes
2. classify common runtime/write/network failures into that family
3. narrow hard safety/integrity markers
4. add `method_resilient` policy handling

### Workstream B: Process policy defaults
Primary files:
1. `src/loom/processes/schema.py`
2. prompt/policy consumers if needed

Tasks:
1. allow `method_resilient` as a valid tool-success policy
2. use it as the ad hoc fallback default
3. preserve explicit `all_tools_hard`, `development_balanced`, and `safety_integrity_only`

### Workstream C: Retry/replan behavior
Primary files:
1. `src/loom/recovery/retry.py`
2. `src/loom/engine/orchestrator/dispatch.py`
3. `src/loom/api/routes.py`

Tasks:
1. route recoverable method-failure reason codes into targeted retry/replan handling
2. strengthen retry-context messaging to explicitly avoid repeating the same failed method without adjustment
3. keep hard-blocking behavior for true policy/integrity faults

## Acceptance Criteria
1. missing website / transient network / generic tool runtime failures do not become hard invariant failures
2. non-policy write failures do not get classified as hard safety/integrity failures
3. ad hoc/process runs prefer retry/replan over advisory-pass or hard-stop for ordinary tool failures
4. sandbox/policy/integrity failures remain blocking
5. full test suite passes with explicit coverage for the new policy and reason-code routing

## Test Plan
1. verifier classifies generic network/runtime failures as recoverable method failures
2. verifier classifies non-policy write failures as recoverable method failures
3. verifier still hard-classifies true path-escape / blocked-host / safety failures
4. ad hoc processes default to `method_resilient`
5. explicit `all_tools_hard` still preserves strict behavior
6. retry manager routes new reason codes into replan-focused retry handling
7. run failure analysis does not describe these as hard invariant failures unless they truly are
