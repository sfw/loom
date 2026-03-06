# Telemetry Runtime Verbosity Modes Plan (2026-03-06)

## Executive Summary
Loom has typed telemetry lifecycle states (`active`, `internal_only`, `deprecated`) but no hardened runtime verbosity control that operators can safely change while the system is running. This plan introduces runtime-configurable telemetry verbosity with explicit sink contracts, security boundaries, and liveness guarantees.

Target runtime modes:
1. `off`
2. `active`
3. `all_typed`
4. `debug`

Primary outcomes:
1. Telemetry verbosity is configurable in `loom.toml` and adjustable at runtime.
2. Runtime control flow remains safe under all modes (no stuck runs, blocked approvals, or dangling streams).
3. Compliance-grade persistence behavior is explicit and protected from accidental operator suppression.
4. Operator surfaces (SSE, TUI, delegate progress/event logs) use one deterministic filtering policy.
5. Mode changes are auditable and access-controlled.

## Current State Snapshot (Baseline)
1. Event lifecycle metadata exists in `src/loom/events/types.py` (`ACTIVE_EVENT_TYPES`, `INTERNAL_ONLY_EVENT_TYPES`).
2. Event payloads are normalized in `src/loom/events/bus.py` and persisted via `EventPersister`.
3. No first-class runtime settings API controls telemetry verbosity.
4. Existing knobs are partial (`enable_artifact_telemetry_events`, `shadow_compare_enabled`, logging level).
5. Sink behavior is not governed by a unified filtering contract.

## Goals
1. Add a first-class telemetry mode with strict, tested semantics.
2. Support both static config (`loom.toml`) and runtime override.
3. Define scope/precedence of overrides explicitly.
4. Keep orchestration internals independent of operator verbosity choices.
5. Preserve backward compatibility with default `active` behavior.

## Non-Goals
1. Replacing event taxonomy or lifecycle labels.
2. Disabling internal runtime signaling required for task execution.
3. Building distributed tracing.
4. Redesigning unrelated logging architecture.

## Critical Decisions

## 1) Sink Classes
Define two sink classes with different guarantees:
1. Compliance sinks:
   1. SQLite `events` persistence (`EventPersister`)
2. Operator sinks:
   1. API SSE (`/tasks/{id}/stream`, `/tasks/{id}/tokens` terminal control path)
   2. Delegate progress/event-log forwarding
   3. TUI process progress rendering

## 2) Persistence Contract
Phase A contract:
1. Compliance sinks are **not suppressed by runtime mode**.
2. Compliance sinks persist at least all `active` events + liveness-critical passthrough events.
3. Operator mode changes only affect operator sinks.

Rationale:
1. Prevents silent loss of audit evidence under `off` mode.
2. Keeps forensic reconstruction and run-summary reconciliation viable.

Optional later phase:
1. Add explicit persistence filtering mode only behind separate opt-in config.
2. Require prominent operator warning + docs + migration notes before enabling.

## 3) Runtime Override Scope
Phase A scope:
1. Runtime override is process-local (applies to current engine instance only).
2. No implicit cross-process propagation between API and TUI processes.

Future phase:
1. Workspace-scoped persisted override with explicit opt-in and atomic config writes.

## 4) Mode Naming
To avoid ambiguity:
1. Runtime mode name is `all_typed` (means `active + internal_only`).
2. Event lifecycle label remains `internal_only` in taxonomy.
3. Accept `internal_only` as a deprecated input alias for one release cycle, normalize to `all_typed`.

## Proposed Mode Semantics

## Canonical Enum
Add `TelemetryMode = Literal["off", "active", "all_typed", "debug"]`.

## Effective Mode Resolution
1. Config default from `loom.toml` (`[telemetry].mode`).
2. Optional runtime override from settings store (if enabled).
3. Effective mode = runtime override when present, else config default.
4. Invalid value normalization:
   1. normalize to `active`
   2. emit typed settings-warning telemetry event

## Event-Class Delivery Rules (Operator Sinks)
1. `off`
   1. suppress non-essential operator telemetry
   2. always deliver liveness/control/human-gate passthrough events
2. `active`
   1. deliver `ACTIVE_EVENT_TYPES` + passthrough
3. `all_typed`
   1. deliver `ACTIVE_EVENT_TYPES ∪ INTERNAL_ONLY_EVENT_TYPES` + passthrough
4. `debug`
   1. deliver everything from `all_typed`
   2. include debug diagnostics channel (typed, rate-limited)

## Liveness/Control/Human-Gate Passthrough Set
Always delivered to operator sinks in every mode, including `off`:
1. `task_created`
2. `task_run_acquired`
3. `task_run_heartbeat`
4. `task_run_recovered`
5. `task_cancel_requested`
6. `task_cancel_ack`
7. `task_cancel_timeout`
8. `task_paused`
9. `task_resumed`
10. `task_injected`
11. `approval_requested`
12. `ask_user_requested`
13. `task_completed`
14. `task_failed`
15. `task_cancelled`

Rationale:
1. Prevent run/delegate/TUI liveness regressions.
2. Prevent human-loop deadlocks caused by hidden gating events.

## Security and Access Control
Runtime mode mutation must be restricted:
1. Settings mutation endpoint disabled by default (`telemetry.runtime_override_api_enabled=false`).
2. When enabled:
   1. require local origin guard (loopback-only), and
   2. require admin token/header (or equivalent auth boundary).
3. Emit audit event on every mode change with actor/source metadata.
4. Reject mode mutation from untrusted callers with explicit 403.

## Configuration Changes

## New `[telemetry]` Block in `loom.toml`
Proposed keys:
1. `mode = "active"` (`off|active|all_typed|debug`)
2. `runtime_override_enabled = true`
3. `runtime_override_api_enabled = false`
4. `persist_runtime_override = false` (phase-gated)
5. `debug_diagnostics_rate_per_minute = 120`
6. `debug_diagnostics_burst = 30`

Files:
1. `src/loom/config.py`
2. `src/loom/setup.py`
3. `docs/CONFIG.md`

## Runtime Settings Surface

## API
Add settings endpoints:
1. `GET /settings/telemetry`
2. `PATCH /settings/telemetry`

PATCH request:
```json
{
  "mode": "debug",
  "persist": false
}
```

Response fields:
1. configured mode
2. runtime override mode
3. effective mode
4. scope (`process_local` for phase A)
5. `updated_at`

## TUI
Expose mode switch in settings UX (or command palette):
1. select mode
2. show effective mode badge and scope
3. show whether change is persisted or process-local

## Debug Diagnostics Channel Hardening
1. Add typed diagnostics event(s) marked `internal_only`.
2. Diagnostics include:
   1. unknown event type emitted
   2. payload contract violation
   3. persistence failure count snapshot
3. Prevent recursion/log storms:
   1. diagnostic events do not self-trigger diagnostics
   2. per-diagnostic-type rate limiting and burst cap
   3. bounded payload fields only

## Sink Consistency Contract
Define expected cross-sink consistency explicitly:
1. Operator sinks must agree on filter decision for the same `(event_type, effective_mode)`.
2. Compliance sink follows persistence contract and is intentionally not suppressed by operator mode.
3. Documentation must include a sink behavior matrix to prevent ambiguity.

## Implementation Architecture

## Workstream 1: Domain Model + Policy
Files:
1. `src/loom/events/types.py`
2. `src/loom/events/` (new policy module)

Tasks:
1. Add `TelemetryMode` and normalization with alias handling.
2. Add shared policy functions:
   1. `should_deliver_operator(event_type, mode)`
   2. `should_persist_compliance(event_type)`
3. Add explicit passthrough registry.

Acceptance:
1. Deterministic policy, unit-tested.
2. Alias normalization (`internal_only` -> `all_typed`) covered by tests.

## Workstream 2: Config + Runtime Store
Files:
1. `src/loom/config.py`
2. `src/loom/api/engine.py`

Tasks:
1. Add `TelemetryConfig` dataclass.
2. Parse `[telemetry]` with validation.
3. Add lock-protected runtime override holder in `Engine`.
4. Expose `effective_telemetry_mode()` and scope metadata.

Acceptance:
1. Config default loads correctly.
2. Runtime override updates mode immediately within process.

## Workstream 3: Security + Settings API
Files:
1. `src/loom/api/routes.py`
2. `src/loom/api/schemas.py`

Tasks:
1. Implement `GET/PATCH /settings/telemetry`.
2. Enforce endpoint gate + caller authorization.
3. Emit mode-change audit telemetry event.

Acceptance:
1. Unauthorized mutation blocked.
2. Authorized mutation reflected immediately.

## Workstream 4: Sink Integration
Files:
1. `src/loom/events/bus.py`
2. `src/loom/api/routes.py`
3. `src/loom/tools/delegate_task.py`
4. `src/loom/tui/app.py`

Tasks:
1. Integrate shared operator filtering policy in SSE/delegate/TUI paths.
2. Keep terminal and human-gate passthrough behavior in `off` mode.
3. Keep compliance persistence contract unchanged in phase A.

Acceptance:
1. `off` mode preserves liveness and interactive gating.
2. No operator sink hangs due to missing terminal/gating events.

## Workstream 5: Persisted Override (Phase-Gated)
Files:
1. `src/loom/config.py`
2. `src/loom/setup.py` and config writer paths

Tasks:
1. Implement atomic persisted override writes only when opt-in enabled.
2. Use file lock + temp file + rename semantics.
3. Conflict policy:
   1. mtime/version check before write
   2. reject with actionable error on conflict

Acceptance:
1. No torn config writes.
2. Concurrent updates are deterministic and safe.

## Workstream 6: Test Matrix
Files:
1. `tests/test_events.py`
2. `tests/test_api.py`
3. `tests/test_orchestrator.py`
4. `tests/test_tui.py`
5. `tests/test_webhook.py`

Tasks:
1. Mode policy unit tests for all event classes and alias inputs.
2. Security tests for settings mutation authorization.
3. SSE/delegate/TUI liveness tests under `off`.
4. Compliance persistence tests proving audit rows remain present in `off`.
5. Debug diagnostics recursion/rate-limit tests.
6. Sink-consistency tests for shared filter behavior.

Acceptance:
1. All mode semantics and guards are test-covered.
2. No regressions in terminal lifecycle, webhook, or human-gate behavior.

## Workstream 7: Docs and Operator Guidance
Files:
1. `docs/CONFIG.md`
2. `docs/telemetry-catalog.md`
3. `docs/agent-integration.md`
4. `docs/tutorial.html`

Tasks:
1. Document modes and lifecycle terminology distinction.
2. Add sink behavior matrix (compliance vs operator sinks).
3. Document scope/precedence of runtime overrides.
4. Add security/authorization requirements for settings mutation.
5. Add troubleshooting playbook for missing events by mode/sink.

Acceptance:
1. Docs match runtime behavior and tests.
2. Operators can reason about visibility deterministically.

## Compatibility and Migration
1. Default mode remains `active`.
2. Missing `[telemetry]` maps to safe defaults.
3. Runtime alias `internal_only` accepted temporarily and normalized to `all_typed`.
4. Existing event lifecycle labels remain unchanged.
5. Persisted override remains disabled by default until phase-gate completion.

## Risks and Mitigations
1. Risk: `off` causes hidden liveness and gating deadlocks.
   1. Mitigation: explicit passthrough set including run/control/human-gate events.
2. Risk: loss of audit evidence from persistence suppression.
   1. Mitigation: compliance persistence not mode-suppressed in phase A.
3. Risk: inconsistent behavior across sinks/processes.
   1. Mitigation: shared policy + scope metadata + sink consistency tests.
4. Risk: runtime mode mutation abuse.
   1. Mitigation: endpoint disabled by default + auth checks + audit event.
5. Risk: debug log storms or recursive diagnostics.
   1. Mitigation: recursion guard + rate limiting + bounded payloads.
6. Risk: persisted override corruption/races.
   1. Mitigation: atomic writes, lock, conflict checks.

## Implementation Sequence
1. Add telemetry config model and mode normalization.
2. Add shared policy module and passthrough registry.
3. Add runtime override store and effective mode resolver in engine.
4. Integrate operator sink filtering (SSE/delegate/TUI).
5. Add settings API with security gating and audit events.
6. Add debug diagnostics with recursion/rate controls.
7. Add full test matrix (including security and liveness).
8. Update docs and operator matrix.
9. Phase-gate persisted override implementation.

## Rollout Strategy
1. Phase A: ship `off`/`active`/`all_typed` with process-local override and compliance persistence lock.
2. Phase B: ship `debug` diagnostics with rate-limit guards.
3. Phase C: ship persisted override (opt-in) with atomic writer.
4. Phase D: tighten CI checks for mode/sink contract coverage.

## Acceptance Criteria (Plan Done)
1. Telemetry mode is configurable in `loom.toml` and exposed in runtime settings.
2. Runtime mode changes apply without restart within process scope.
3. Modes `off`, `active`, `all_typed`, and `debug` have tested and documented semantics.
4. Operator sink filtering is consistent and deterministic.
5. `off` mode does not break stream completion, control-plane acknowledgements, or human-gate prompts.
6. Compliance persistence remains audit-safe in phase A.
7. Settings mutation path is access-controlled and auditable.
8. Docs include sink matrix, scope/precedence rules, and troubleshooting.

## Deliverables
1. Hardened telemetry mode config/runtime contract.
2. Shared sink policy implementation.
3. Secure runtime settings API + TUI controls.
4. Debug diagnostics channel with recursion/rate protections.
5. Test coverage for mode semantics, security, and liveness.
6. Updated telemetry/operator documentation.
