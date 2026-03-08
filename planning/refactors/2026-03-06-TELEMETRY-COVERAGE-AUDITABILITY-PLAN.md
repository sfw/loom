# Telemetry Coverage and Auditability Plan (2026-03-06)

## Executive Summary
Telemetry in Loom is useful but not complete enough for reliable debugging and audits. This plan closes event coverage gaps, standardizes payload contracts, and adds CI checks so future changes cannot silently reduce observability.

Primary outcomes:
1. Every declared event type has an explicit lifecycle state (`active` or `deprecated`) and a clear runtime status (emitted or intentionally removed).
2. Every active event type has deterministic payload requirements and at least one verification test.
3. Run timelines become fully traceable with correlation identifiers and consistent phase/subtask metadata.
4. Auditors can answer "what happened, why, and with what evidence" from persisted telemetry alone.
5. Operator-facing telemetry surfaces (delegate event logs, TUI progress UI, SSE/docs) stay in sync with the active event taxonomy.

## Coverage Snapshot (Baseline, 2026-03-06)
Static audit of `src/loom/events/types.py` and runtime emit paths found:
1. 81 declared event constants.
2. 74 with runtime emission paths.
3. 7 declared but not emitted.
4. 30 emitted with no direct test assertions on event presence/payload.
5. Control-plane progress events (`task_paused`, `task_resumed`, `task_injected`, `task_cancel_*`, `steer_*`) are used in delegate/TUI flows but are not part of the typed core event catalog.
6. Event persistence currently generates a random `correlation_id` per row and stamps persistence time, reducing causal trace quality.
7. Delegate/TUI consumers subscribe/format only a subset of active events, so some telemetry is emitted but not surfaced to operators.

Declared but not emitted:
1. `TASK_CREATED`
2. `SUBTASK_BLOCKED`
3. `VERIFICATION_STARTED`
4. `VERIFICATION_PASSED`
5. `VERIFICATION_FAILED`
6. `STEER_INSTRUCTION`
7. `VERIFICATION_CONTRADICTION_DETECTED` is emitted by string alias, not the exported constant.

## Goals
1. Complete and enforce telemetry coverage for all active event types.
2. Make event payloads stable and machine-validated.
3. Improve per-run forensic debugging (ordering, causality, and scope).
4. Improve audit readiness (lineage, safety-relevant decisions, human-in-the-loop actions).
5. Prevent regressions with automation, not manual review.

## Non-Goals
1. Introducing a distributed tracing platform.
2. Replacing all logs with telemetry events.
3. Capturing sensitive raw content in event payloads.

## Debug and Audit Requirements
For any run, operators must be able to answer:
1. Which lifecycle transitions occurred and in what order.
2. Which model/tool decisions were made and what triggered them.
3. Which human-gate events occurred (`approval`, `ask_user`, steering) and outcomes.
4. Why verification accepted/rejected output, including contradiction and remediation flow.
5. Which files/artifacts were read/written and how policy gates reacted.

## Design Principles
1. Event type constants are the source of truth; no unowned string-only event types.
2. Every active event has a contract: required keys, optional keys, and redaction policy.
3. Emissions happen at orchestration boundaries where context is available.
4. Event volume stays high-signal; avoid chatty internal-only diagnostics.
5. CI enforces completeness (declaration -> emission -> test -> documentation).

## Proposed Event Contract (v1)
All active events should include:
1. `task_id`
2. `timestamp` (event object timestamp)
3. `run_id` when available
4. `subtask_id` when available
5. `phase_id` when available
6. `attempt` or retry index when available
7. `source_component` (`orchestrator`, `runner`, `verification`, `api`, `recovery`, etc.)
8. `schema_version` (event payload schema version)
9. `event_id` (stable unique ID per emitted event)
10. `correlation_id` (causal chain identifier)
11. `sequence` (monotonic ordering key at least within task/run scope)

Event-specific payload remains allowed, but required common keys should be normalized with helpers.

## Workstreams

## Workstream 1: Event Catalog Hardening
Files:
1. `<repo-root>/src/loom/events/types.py`
2. `<repo-root>/docs/` (new telemetry catalog doc)

Tasks:
1. Add lifecycle annotations for each event (`active`, `deprecated`, `internal_only` if needed).
2. Resolve alias mismatch: emit `VERIFICATION_CONTRADICTION_DETECTED` via exported constant.
3. Decide for each non-emitted declaration: implement or deprecate.

Acceptance:
1. No active event remains declared-but-unemitted without explicit rationale.
2. No runtime event type is emitted only by free-form string when a constant exists.

## Workstream 2: Runtime Gap Closure
Files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. `<repo-root>/src/loom/engine/verification.py`
3. `<repo-root>/src/loom/recovery/` (and steering path)
4. `<repo-root>/src/loom/api/routes.py` (task creation path if needed)

Tasks:
1. Implement missing active lifecycle emissions (`TASK_CREATED`, `SUBTASK_BLOCKED`, verification lifecycle trio, steering).
2. Ensure emit location reflects real state transitions (not synthetic-only events).
3. Ensure terminal lifecycle events (`task_completed`, `task_failed`, `task_cancelled`) always include reason/outcome metadata.

Acceptance:
1. Each active event has at least one runtime emission path reachable in tests.
2. Lifecycle transitions are represented once, at the correct boundary.

## Workstream 3: Payload Schema Validation
Files:
1. `<repo-root>/src/loom/events/bus.py`
2. `<repo-root>/src/loom/engine/*` emit helpers
3. `<repo-root>/tests/` schema tests

Tasks:
1. Add lightweight event payload validation helper used by emit wrappers.
2. Define required keys per event family (task/subtask/tool/verification/human-loop/remediation).
3. Add redaction guards for sensitive fields (`auth`, secrets, raw private content).

Acceptance:
1. Invalid payload shape fails tests.
2. Redaction tests prove sensitive keys are excluded/sanitized.

## Workstream 4: Completeness Test Matrix
Files:
1. `<repo-root>/tests/test_events.py`
2. `<repo-root>/tests/test_orchestrator.py`
3. `<repo-root>/tests/test_verification.py`
4. `<repo-root>/tests/test_approval.py`
5. `<repo-root>/tests/test_questions.py`
6. `<repo-root>/tests/test_api.py`
7. `<repo-root>/tests/test_tui.py` (as needed)

Tasks:
1. Add/expand tests so every active event type is asserted at least once.
2. Add payload assertions for audit-critical fields (reason codes, counters, IDs).
3. Add event ordering tests for key sequences (plan -> execute -> verify -> terminal).

Acceptance:
1. CI check reports zero active events without test coverage.
2. Key run flows have deterministic event order assertions.

## Workstream 5: CI Guardrails and Drift Prevention
Files:
1. `<repo-root>/scripts/` (new telemetry audit script)
2. `<repo-root>/.github/workflows/ci.yml`

Tasks:
1. Add script to compare:
   1. declared event constants
   2. known emission sites
   3. test assertions
2. Fail CI when an active event is missing emission or missing tests.
3. Generate artifact report in CI for visibility.

Acceptance:
1. Any future event drift fails CI before merge.
2. Engineers get a clear "missing event coverage" report.

## Workstream 6: Audit-Focused Run Summaries
Files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. `<repo-root>/src/loom/events/` summary utilities
3. `<repo-root>/docs/`

Tasks:
1. Expand run-summary payloads to include:
   1. verification outcome counts by reason
   2. remediation lifecycle counters
   3. human-loop decision counts
   4. blocked/degraded/replanned indicators
2. Ensure summaries are emitted exactly once per run and persisted.

Acceptance:
1. A single summary event can answer top-level run audit questions.
2. Summary counters reconcile with per-event totals in integration tests.

## Workstream 7: Transport and Persistence Integrity
Files:
1. `<repo-root>/src/loom/events/bus.py`
2. `<repo-root>/src/loom/state/memory.py`
3. `<repo-root>/src/loom/state/schema.sql`
4. `<repo-root>/src/loom/api/engine.py`

Tasks:
1. Persist emitted event timestamp (UTC) rather than generating a second persistence timestamp.
2. Replace random per-row correlation IDs with causal correlation propagation.
3. Add deterministic sequence semantics for replay ordering (`events.id` and/or per-task sequence field usage contract).
4. Add telemetry health coverage for persistence failure paths (drop counters or failure events/log records).
5. Define and enforce timezone policy for all telemetry timestamps (UTC canonical).

Acceptance:
1. Replay ordering and causal chains are deterministic for audit reconstruction.
2. Persisted event record preserves original emitted timestamp.
3. Persistence failures are observable and test-covered.

## Workstream 8: Consumer Surface Parity
Files:
1. `<repo-root>/src/loom/tools/delegate_task.py`
2. `<repo-root>/src/loom/tui/app.py`
3. `<repo-root>/docs/agent-integration.md`
4. `<repo-root>/docs/tutorial.html`
5. `<repo-root>/docs/CONFIG.md`

Tasks:
1. Expand delegate event subscription/forwarding coverage to include the active event catalog (or a documented subset policy).
2. Ensure TUI progress formatting has explicit handling or generic fallback for audit-critical events (`run_validity_scorecard`, remediation, verification-rate events, task-run lifecycle).
3. Prevent silent drops: unknown-but-valid events should still surface in operator logs/panels.
4. Preserve run-tab launch/cancel heartbeat diagnostics and delegate event-log path visibility for stuck `/run` debugging.
5. Align public docs/event tables with the real event taxonomy.

Acceptance:
1. Operator surfaces expose all audit-critical active events.
2. Documentation event reference matches runtime behavior and tests.

## Workstream 9: Control-Plane and Cowork Telemetry Normalization
Files:
1. `<repo-root>/src/loom/events/types.py`
2. `<repo-root>/src/loom/api/routes.py`
3. `<repo-root>/src/loom/tools/delegate_task.py`
4. `<repo-root>/src/loom/tui/app.py`
5. `<repo-root>/src/loom/state/conversation_store.py`

Tasks:
1. Decide canonical status for control events currently outside typed catalog (`task_paused`, `task_resumed`, `task_injected`, `task_cancel_requested`, `task_cancel_ack`, `task_cancel_timeout`, `steer_*`).
2. Either promote these events into typed telemetry with contracts or explicitly mark them as UI-local only.
3. Emit `STEER_INSTRUCTION` where steering actually enters execution context.
4. Define a compatibility boundary between task-run telemetry and cowork chat replay telemetry (`cowork_chat_events`).
5. Include cowork context-window telemetry (`context_tokens`, `context_messages`, `omitted_messages`, `recall_index_used`) in the telemetry catalog and replay expectations.

Acceptance:
1. No ambiguous "semi-telemetry" event names remain.
2. Steering and control-plane actions are auditable with clear ownership and scope.

## Workstream 10: Webhook Delivery Telemetry
Files:
1. `<repo-root>/src/loom/events/webhook.py`
2. `<repo-root>/src/loom/events/types.py`
3. `<repo-root>/tests/test_webhook.py`

Tasks:
1. Add webhook delivery lifecycle telemetry (`attempted`, `succeeded`, `failed`, `dropped/unregistered`) with bounded metadata.
2. Ensure callback URL telemetry is safe (host/hash only, no secrets/query params).
3. Add tests for webhook lifecycle telemetry across retry/success/failure paths.

Acceptance:
1. Webhook delivery behavior is auditable from event logs.
2. Delivery retries and terminal failure conditions are reconstructible post-run.

## Implementation Sequence
1. Finalize event catalog status (`active/deprecated`) and naming cleanups.
2. Normalize control-plane/cowork telemetry boundaries and naming.
3. Implement missing active emissions and constant-alias cleanup.
4. Add payload contract helper and event-family schema tests.
5. Harden persistence semantics (timestamp, correlation, ordering, failure visibility).
6. Expand delegate/TUI/documentation parity for active telemetry.
7. Add completeness matrix tests and CI drift checker.
8. Expand audit summary payload, webhook delivery telemetry, and reconciliation tests.
9. Update docs and release notes.

## Rollout Strategy
1. Phase A (safe additive): add emissions + tests, no behavior changes.
2. Phase B (strictness): enable CI gate for active-event completeness.
3. Phase C (enforcement): reject new event additions without schema/test entries.
4. Phase D (operator parity): require TUI/delegate/docs coverage for audit-critical events before release.

## Risks and Mitigations
1. Risk: Event volume/noise increases.
   1. Mitigation: keep event set milestone-based; avoid low-level chatter.
2. Risk: Test brittleness from strict order assertions.
   1. Mitigation: assert partial order only for critical transitions.
3. Risk: Backward compatibility for downstream consumers.
   1. Mitigation: additive payload changes, schema versioning, deprecation window.

## Acceptance Criteria (Plan Done)
1. 0 active events declared-but-unemitted.
2. 0 active events emitted-without-tests.
3. Event constants are used consistently (no shadow string aliases for exported events).
4. Persisted telemetry preserves emitted timestamp, ordering semantics, and causal correlation.
5. Control-plane and steering telemetry are cataloged and auditable by policy.
6. Webhook delivery lifecycle is observable.
7. Run-summary reconciliation tests pass.
8. CI blocks telemetry drift automatically.
9. Documentation includes event catalog, payload contracts, and query examples.

## Deliverables
1. Updated event catalog and lifecycle status.
2. Completed runtime instrumentation for active event set.
3. Payload schema validation tests.
4. Persistence/correlation/ordering hardening for replay-grade auditability.
5. Delegate/TUI/docs telemetry parity update.
6. Webhook lifecycle telemetry coverage.
7. CI telemetry completeness check.
8. Updated audit/debug documentation and operator query guide.
