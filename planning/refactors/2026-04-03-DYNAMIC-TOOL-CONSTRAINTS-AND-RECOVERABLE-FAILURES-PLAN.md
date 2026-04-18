# Dynamic Tool Constraints and Recoverable Failures Plan (2026-04-03)

## Objective
Refactor Loom so tool availability is treated as dynamic runtime state, not a static registration/configuration assumption, and so tool-path failures do not automatically collapse into terminal hard-invariant task failure when the subtask objective remains achievable by other means.

This plan addresses two coupled problems:
1. tools can be exposed and selected even when their required binaries, auth, or runtime preconditions are not currently satisfiable
2. execution failures such as `Binary not found: codex` are currently over-interpreted as objective-level failure rather than method-level failure

## Decision
Loom should adopt a capability-aware execution model with four rules:
1. tool constraints are evaluated at every backend launch and stored as runtime state
2. tool selection surfaces should distinguish `configured` from `runnable now`
3. provider/tool execution should support explicit binary path configuration in addition to `PATH` lookup
4. missing-capability and missing-tool-path failures should route to recoverable replanning whenever the subtask objective is still satisfiable with other tools or strategies

## Why This Is Needed
Recent failure analysis exposed a brittle behavior pattern:
1. a run selected `openai_codex` for an intermediate parsing step
2. the packaged desktop environment could not resolve `codex` via `PATH`
3. the provider tool returned `binary_not_found`
4. verification/retry logic escalated the failure into `hard_invariant_failed`
5. the run stopped even though "parse the delegate list into structured data" was still feasible via other methods

The important takeaway is not "Codex is bad" or "desktop PATH is wrong." The deeper problem is that Loom currently treats one failed execution path as if it proved the subtask objective itself was impossible. That operating model is too brittle for a multi-tool system.

## Goals
1. Re-evaluate tool constraints on every launch because system conditions are dynamic.
2. Make tool availability a first-class runtime state with machine-readable reasons.
3. Support CLI tools through both `PATH` lookup and explicit configured binary paths.
4. Allow tool providers to express required binaries, auth, execution surfaces, and other preconditions in a reusable contract.
5. Prevent unavailable tools from being presented to planners/executors as if they were currently runnable.
6. Convert missing-binary and similar capability failures into recoverable replanning signals where the subtask objective remains possible.
7. Preserve true hard-failure semantics for safety, policy, integrity, and objective-level impossibility.
8. Surface tool readiness clearly in desktop/API/doctor views so operators can understand why a tool is unavailable.

## Non-Goals
1. Solving all environment portability issues in one pass.
2. Removing all runtime execution checks; execution-time validation still remains necessary.
3. Replacing the existing tool registry from scratch.
4. Designing provider-specific shell login semantics as the primary solution.
5. Treating every tool failure as recoverable; some failures should still block.

## Current State

### A) Tool exposure is mostly registration/config based
The tool registry and tool catalog currently answer "is this tool registered and allowed on this surface?" more than "is this tool runnable right now?"

Consequences:
1. planners can see tools that are configured but operationally unavailable
2. runs can build plans around tools that are known-bad for the current environment

### B) Provider agent tools resolve binaries only at call time
Provider-backed tools such as `openai_codex` call `shutil.which(spec.binary)` during execution.

Consequences:
1. availability problems are only discovered after planning and tool selection
2. packaged-app environment differences become runtime surprises

### C) Failure taxonomy is too coarse for tool-path failures
A missing provider binary currently becomes a tool failure which can later collapse into `hard_invariant_failed`.

Consequences:
1. method failure is conflated with objective failure
2. remediation and retry logic become too conservative
3. the system cannot easily replan around a missing provider even when alternatives exist

### D) Runtime status does not currently expose tool readiness
The runtime status endpoint reports broad runtime basics, but not a first-class tool-capability snapshot.

Consequences:
1. desktop UI cannot warn about unavailable tool families before a run starts
2. operator diagnosis requires log/code inspection rather than direct runtime introspection

## Design Principles

### P1: Dynamic by default
Tool constraints must be re-evaluated at every backend launch. System conditions are inherently dynamic:
1. `PATH` differs across launch surfaces
2. binaries appear/disappear
3. auth material changes
4. permissions, mounts, network, and service availability change

### P2: Separate configuration from current capability
A tool can be:
1. configured
2. allowed
3. exposed on a surface
4. unavailable right now

These states must not be collapsed into one boolean.

### P3: Distinguish method failure from objective failure
"This tool path failed" does not imply "this subtask cannot be completed."

### P4: Launch-time visibility plus execution-time defense
Launch-time preflight should catch expected unavailability early, but every tool execution should still perform final validation in case the environment changed after launch.

### P5: Replanning should be capability-aware
When a tool path fails due to missing capability, Loom should remove that path from the immediate candidate set and try another route before concluding the objective is blocked.

### P6: Hard invariants must stay narrow
Hard invariants should remain reserved for:
1. safety and policy violations
2. integrity breaches
3. forbidden actions
4. objective-level impossibility or non-negotiable contract failure

## Target End State
Loom should behave like this:
1. backend launch computes a tool capability snapshot
2. each tool/provider has a runtime status such as `available`, `degraded`, or `unavailable`
3. each status includes structured reasons such as:
   - `binary_not_found`
   - `binary_not_executable`
   - `version_unsupported`
   - `auth_unavailable`
   - `surface_not_supported`
   - `feature_disabled`
4. planners and `list_tools` consume a filtered "runnable now" inventory by default
5. desktop/settings/doctor views can show both runnable and unavailable tools with reasons
6. provider execution supports configured absolute binary paths in addition to `PATH`
7. execution-time capability failures route to recoverable reason codes when the objective remains possible
8. remediation/retry logic can replan around unavailable methods instead of hard-aborting the whole objective

## Proposed Architecture

### 1) Introduce a tool capability contract
Add a reusable runtime capability contract for tools, especially external-tool wrappers.

Suggested shape:
1. `ToolAvailabilityStatus`
   - `state`: `available | degraded | unavailable`
   - `reasons`: list of structured reason codes/messages
   - `checked_at`
   - `metadata`
2. `ToolRequirement`
   - binaries
   - minimum versions
   - required auth/env
   - supported execution surfaces
   - optional capabilities

Implementation direction:
1. extend `Tool` with an optional capability/preflight hook
2. provide shared helpers for external binary tools
3. let provider-agent tools and WordPress tools adopt the same contract

### 2) Add launch-time capability evaluation
At backend startup:
1. evaluate all runtime-sensitive tools
2. store results in a runtime capability snapshot
3. expose the snapshot through API/runtime status and desktop diagnostics

Refresh triggers:
1. every backend launch
2. relevant runtime config changes
3. explicit doctor/refresh actions
4. optional lazy refresh when cached results are stale

### 3) Split tool inventories into multiple views
Loom should maintain distinct inventories:
1. `registered_tools`
2. `configured_tools`
3. `runnable_tools`
4. `unavailable_tools`

Behavior:
1. model-facing planning/tool catalogs should use `runnable_tools` by default
2. operator-facing diagnostics should show both runnable and unavailable tools with reasons
3. explicit advanced views can still display configured-but-unavailable tools for debugging

### 4) Support explicit binary path configuration
External tools should support configured binary resolution, with a provider-specific override.

Suggested resolution order:
1. explicit configured absolute path
2. explicit configured command name/path fragment
3. `PATH` lookup
4. optional future shell/PTY bridge if intentionally enabled

This should be implemented as a reusable external-binary resolver rather than one-off provider hacks.

### 5) Add capability-aware tool preconditions to planning and run launch
Before a run starts:
1. process/tool preflight should verify required tools are currently runnable
2. if a required tool is unavailable, preflight should:
   - fail early with actionable reasons, or
   - trigger a pre-execution replan that removes the unavailable tool path

For ad hoc runs:
1. recommendations and required-tool scaffolding should draw from runnable tools, not merely registered tools

### 6) Introduce recoverable reason codes for tool-path failures
Missing binary and related capability failures should no longer default to `hard_invariant_failed`.

Suggested reason-code family:
1. `tool_capability_unavailable`
2. `provider_binary_not_found`
3. `provider_binary_unsupported`
4. `provider_auth_unavailable`
5. `tool_runtime_capability_unavailable`

These should map to retry/replan strategies that:
1. mark the specific tool path unavailable for the current attempt
2. preserve the subtask objective
3. ask the planner/executor to choose another method

### 7) Preserve hard-failure routing for real hard cases
Still terminal/blocking:
1. policy-denied tool use
2. forbidden output path / sandbox escape
3. destructive action without approval
4. explicit process contract requiring one unavailable capability with no allowed fallback
5. objective proven impossible after alternative paths are exhausted

## Workstreams

### Workstream A: Capability Model and Registry Integration
Primary files:
1. `src/loom/tools/registry.py`
2. `src/loom/tools/tooling_common/provider_agent_tool.py`
3. `src/loom/tools/wp_cli.py`
4. `src/loom/tools/wp_env.py`
5. `src/loom/tools/wp_quality_gate.py`
6. new shared runtime capability module under `src/loom/runtime/` or `src/loom/tools/tooling_common/`

Tasks:
1. define shared capability status models and reason codes
2. add optional capability hook to tools
3. add registry support for capability snapshots and filtered runnable inventory
4. preserve backward compatibility for tools that do not implement the new hook

Acceptance:
1. registry can return both all registered tools and runnable-now tools
2. capability status can be queried without executing the tool

### Workstream B: External Binary Resolution Contract
Primary files:
1. `src/loom/tools/tooling_common/provider_agent_tool.py`
2. `src/loom/tools/tooling_common/version_matrix.py`
3. `src/loom/config.py`
4. `src/loom/config_runtime/registry.py`
5. `docs/CONFIG.md`

Tasks:
1. add config for provider binary overrides
2. refactor binary lookup into a shared resolver
3. distinguish "configured path invalid" from "PATH lookup failed"
4. preserve existing defaults for users who rely on PATH

Acceptance:
1. provider-backed tools can run with explicit absolute binary paths
2. capability checks and execution share the same resolver logic

### Workstream C: Runtime Snapshot and Desktop/API Visibility
Primary files:
1. `src/loom/api/engine.py`
2. `src/loom/api/schemas.py`
3. `src/loom/api/routes.py`
4. `src/loom/cli/commands/root.py`
5. desktop UI files under `apps/desktop/src/`

Tasks:
1. add tool capability state to runtime snapshot
2. extend runtime doctor to report per-tool availability
3. surface unavailable tools and reasons in desktop settings/diagnostics
4. provide an explicit refresh path for operator-triggered re-checks

Acceptance:
1. a packaged desktop launch shows unavailable provider tools before a run starts
2. doctor output makes the exact missing precondition obvious

### Workstream D: Planner and Preflight Filtering
Primary files:
1. `src/loom/tui/app/process_runs/adhoc.py`
2. `src/loom/cowork/session.py`
3. `src/loom/tools/list_tools.py`
4. `src/loom/engine/runner/execution.py`
5. `src/loom/tui/app/process_runs/lifecycle.py`

Tasks:
1. default model-facing tool catalogs to runnable tools only
2. ensure ad hoc planning scaffolds use runnable tool inventories
3. add launch/preflight checks for required-but-unavailable tools
4. allow operator-facing views to optionally inspect unavailable tools too

Acceptance:
1. planners stop selecting tools that the current runtime already knows are unavailable
2. runs fail fast or replan before execution when a required tool path is unavailable

### Workstream E: Failure Taxonomy and Recovery Routing
Primary files:
1. `src/loom/engine/verification/tier1.py`
2. `src/loom/recovery/retry.py`
3. `src/loom/engine/orchestrator/remediation.py`
4. `src/loom/api/routes.py`
5. `src/loom/engine/verification/development.py`

Tasks:
1. add recoverable capability-unavailable reason codes
2. stop collapsing these failures into broad `hard_invariant_failed`
3. mark the failed tool path unavailable for the current remediation cycle
4. replan around the objective using the reduced runnable-tool set
5. only escalate to terminal failure when:
   - no valid alternative methods remain, or
   - the process contract explicitly forbids fallback

Acceptance:
1. `Binary not found: codex` does not terminate an otherwise feasible parsing subtask
2. the run can continue with another tool/strategy when one exists

## Failure Classification Matrix

### Route to recoverable replanning
Examples:
1. provider binary missing
2. provider version too old
3. auth token missing for one optional method
4. browser runtime unavailable when browser checks are optional
5. one helper tool unavailable but alternate tools exist

Recommended outcome:
1. reason code in the capability-unavailable family
2. retry strategy focused on replanning or alternate method selection
3. no hard-invariant escalation by default

### Route to hard/blocking
Examples:
1. sandbox/policy denies the requested operation
2. required deliverable path is forbidden
3. process explicitly requires a capability and declares no fallback
4. all viable methods are exhausted and the objective remains unmet

Recommended outcome:
1. blocking policy decision
2. hard-failure reason only after objective-level impossibility is established

## Rollout Strategy

### Phase 0: Observability First
1. implement runtime capability snapshot
2. expose it in doctor/API/desktop
3. keep planning behavior unchanged initially

### Phase 1: Planner Filtering
1. default model-facing tool catalogs to runnable-only
2. collect telemetry on filtered tools and avoided failures

### Phase 2: Recoverable Failure Routing
1. introduce new reason codes and retry routing
2. shadow-compare old `hard_invariant` behavior versus new recoverable behavior

### Phase 3: Preflight Enforcement
1. enforce required-tool preconditions at run launch
2. add targeted preflight replan when unavailable required tools have obvious substitutes

## Telemetry and Diagnostics
Add telemetry for:
1. tool capability snapshot counts by state
2. launch-time unavailable tools
3. tool-path failures converted into replanning
4. runs saved by alternate-method recovery
5. hard failures caused by true no-fallback exhaustion

This is needed both for correctness and to prove the refactor reduces brittleness instead of hiding failures.

## Test Plan

### Unit Tests
1. provider tool with explicit binary path override resolves successfully
2. invalid explicit binary path reports structured unavailability
3. packaged-like PATH without `/opt/homebrew/bin` marks `codex` unavailable
4. `list_tools` defaults to runnable inventory while diagnostics can still show unavailable tools
5. tool capability snapshot refreshes on launch and config changes
6. capability-unavailable failures map to replanning rather than `hard_invariant_failed`

### Integration Tests
1. desktop/runtime snapshot reports provider unavailability before a run starts
2. ad hoc run planning excludes unavailable provider tools
3. a subtask that first chooses `openai_codex` can recover via alternate method when `codex` is unavailable
4. a process that explicitly requires an unavailable tool fails early with a clear precondition error

### Regression Tests
1. true policy/sandbox failures remain hard-blocking
2. existing registered tools without capability hooks still behave correctly
3. CLI-launched environments and packaged-desktop environments both produce accurate capability snapshots

## Open Questions
1. Should capability snapshots be global per backend instance, workspace-scoped, or hybrid?
2. Which failures should be treated as `degraded` versus `unavailable`?
3. Should model-facing `list_tools` ever include unavailable tools as advisory entries, or should those stay fully operator-only?
4. How aggressively should execution attempts cache negative capability results during a run?
5. Do we want a process/schema way to declare fallback groups such as "any one of these providers satisfies this requirement"?

## Recommended Initial Scope
For the first implementation wave:
1. cover provider-agent tools (`openai_codex`, `claude_code`, `opencode`)
2. cover WordPress/system-binary tools that already perform `which(...)` checks
3. publish runtime capability status in doctor and desktop runtime status
4. filter model-facing tool inventories to runnable tools
5. reroute missing-binary failures into recoverable replanning

This yields the largest brittleness reduction with the smallest architecture risk.

## Success Criteria
This refactor is successful when:
1. a packaged desktop launch can immediately explain why a tool is unavailable
2. planners stop selecting tools the runtime already knows are unusable
3. missing-provider failures no longer terminate feasible objectives by default
4. hard invariants become narrower and more trustworthy
5. users can distinguish "the chosen method failed" from "the task was impossible"
