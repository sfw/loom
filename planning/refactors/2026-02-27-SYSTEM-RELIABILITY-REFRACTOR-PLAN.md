# System Reliability Refactor Plan (2026-02-27)

## Objective
Deliver a coordinated refactor program for:
1. Global run budget manager.
2. Explicit executor completion contract.
3. Policy-gated planner fallback behavior.
4. First-class SQLite persistence for remediation queue and retry history.
5. Durable background execution with crash-safe resume.
6. Idempotency keys for mutating tool calls.
7. SLO-driven observability.

This plan is implementation-focused and designed for phased rollout with rollback safety.

## Scope and Non-Goals
1. Scope: runtime orchestration, runner loop, persistence schema, API lifecycle, and event telemetry.
2. Scope: backward-compatible rollout paths, feature flags, and migration strategy.
3. Non-goal: replacing the core planner/replanner architecture.
4. Non-goal: distributed multi-node task execution in this phase.
5. Non-goal: changing process package contracts unless required by completion protocol.

## Current State Summary (Why This Refactor Is Needed)
1. Task bounds are local (loop/tool/subtask), but no global task-level spend cap exists.
2. Runner treats first text-only model response as completion, which can allow false-success narratives.
3. Planner fallback silently degrades to `execute-goal` in multiple paths, reducing controllability.
4. Remediation queue and retry lineage are primarily metadata-driven, limiting queryability and recovery robustness.
5. API background execution relies on `asyncio.create_task(...)` with no durable lease/heartbeat and no startup resume.
6. Tool retries can replay mutating actions without durable idempotency controls.
7. Telemetry is rich but not yet SLO-oriented with run-level correlation and alert surfaces.

## Program Principles
1. Preserve existing behavior by default where possible, then tighten via explicit flags.
2. Make every critical decision observable with stable event payload fields.
3. Prefer additive schema changes and dual-write migrations before source-of-truth switchovers.
4. Ensure failure modes terminate deterministically with explicit reason codes.
5. Keep task state reconcilable between YAML (`state.yaml`) and SQLite.

## Delivery Strategy
1. Phase 0: Foundation and migration scaffolding.
2. Phase 1: Runtime correctness controls (`#1`, `#2`, `#3`).
3. Phase 2: Persistence hardening (`#4`).
4. Phase 3: Durable execution semantics (`#5`).
5. Phase 4: Mutation safety (`#6`).
6. Phase 5: SLO and operations surface (`#10`).

## Cross-Cutting Foundation (Phase 0)

### F0.1 Introduce Run Identity
Add a stable `run_id` generated per execution attempt and propagated through:
1. task metadata.
2. event payloads.
3. retry/remediation records.
4. future task-run table rows.

Primary touchpoints:
1. `src/loom/engine/orchestrator.py`
2. `src/loom/events/bus.py`
3. `src/loom/state/task_state.py`
4. `src/loom/state/memory.py`
5. `src/loom/api/routes.py`

### F0.2 Feature Flag Matrix
Add config gates for each workstream to allow progressive activation:
1. `execution.enable_global_run_budget`
2. `execution.executor_completion_contract_mode` (`off|warn|enforce`)
3. `execution.planner_degraded_mode` (`allow|require_approval|deny`)
4. `execution.enable_sqlite_remediation_queue`
5. `execution.enable_durable_task_runner`
6. `execution.enable_mutation_idempotency`
7. `execution.enable_slo_metrics`

Primary touchpoints:
1. `src/loom/config.py`
2. `loom.toml.example`
3. `docs/CONFIG.md`
4. `tests/test_config.py`

### F0.3 Event Field Standardization
Standardize optional event keys:
1. `run_id`
2. `subtask_id`
3. `attempt`
4. `reason_code`
5. `duration_ms`

Primary touchpoints:
1. `src/loom/events/types.py`
2. `src/loom/engine/orchestrator.py`
3. `src/loom/engine/runner.py`
4. `tests/test_events.py`

## Workstream A: Global Run Budget Manager (`#1`)

### A1 Design
Implement a task-level budget controller with hard and soft limits:
1. wall-clock limit per task run.
2. total model tokens.
3. total model invocations.
4. total tool calls.
5. total mutating tool calls.
6. replan and remediation-attempt caps.

Budget checks run at:
1. start of each orchestrator loop iteration.
2. before dispatching each subtask.
3. after each subtask result aggregation.
4. before executing each tool call in runner.

### A2 Data/Config Changes
Add execution config fields:
1. `max_task_wall_clock_seconds`
2. `max_task_total_tokens`
3. `max_task_model_invocations`
4. `max_task_tool_calls`
5. `max_task_mutating_tool_calls`
6. `max_task_replans`
7. `max_task_remediation_attempts`

### A3 Runtime Behavior
On limit breach:
1. emit `task_budget_exhausted`.
2. record budget snapshot in task metadata and SQLite.
3. mark remaining pending subtasks as skipped due to budget policy.
4. terminate with explicit failure reason.

### A4 Touchpoints
1. `src/loom/config.py`
2. `src/loom/engine/orchestrator.py`
3. `src/loom/engine/runner.py`
4. `src/loom/events/types.py`
5. `tests/test_orchestrator.py`
6. `tests/test_runner.py` (new if needed)

### A5 Acceptance Criteria
1. A task exceeding any configured global budget fails deterministically.
2. Failure payload includes exhausted budget name, configured limit, observed value.
3. Budget counters remain consistent across retries/replans/remediation paths.

## Workstream B: Explicit Executor Completion Contract (`#2`)

### B1 Design
Replace implicit completion ("text-only response means done") with explicit completion protocol:
1. executor final response must include a machine-parseable completion block.
2. completion block must declare `status`, `deliverables_touched`, and `verification_notes`.
3. runner only exits successfully when completion block is valid.

### B2 Contract Modes
1. `off`: legacy behavior.
2. `warn`: allow completion without contract but emit warning event.
3. `enforce`: completion contract required for success.

### B3 Prompt and Validator Changes
1. extend executor prompt template instructions with completion schema.
2. add parser in `ResponseValidator`.
3. if invalid/missing in enforce mode, append deterministic correction system message and continue loop.

### B4 Touchpoints
1. `src/loom/prompts/assembler.py`
2. `src/loom/prompts/templates/*.yaml` (executor templates)
3. `src/loom/models/router.py` (or validator location)
4. `src/loom/engine/runner.py`
5. `src/loom/engine/verification.py`
6. `tests/test_prompts.py`
7. `tests/test_runner.py` (new if needed)
8. `tests/test_verification.py`

### B5 Acceptance Criteria
1. In `enforce` mode, text-only responses without completion block do not terminate subtask successfully.
2. Completion block data appears in subtask summary and verification context.
3. Regression tests confirm legacy behavior still works in `off` mode.

## Workstream C: Policy-Gated Planner Fallback (`#3`)

### C1 Design
Centralize all planner degradation paths under policy:
1. planner call failure fallback.
2. planner JSON parse failure fallback.
3. replanner parse failure fallback.

Replace silent degradation with explicit policy behavior:
1. `allow`: create degraded single-subtask plan and emit `task_plan_degraded`.
2. `require_approval`: pause for approval before degraded plan activation.
3. `deny`: fail planning with explicit error.

### C2 Safety Behavior
For degraded plans:
1. mark plan metadata `degraded=true` and include `degrade_reason`.
2. reduce parallelism to `1` for degraded mode by default.
3. include degradation context in telemetry summary.

### C3 Touchpoints
1. `src/loom/engine/orchestrator.py`
2. `src/loom/recovery/approval.py`
3. `src/loom/events/types.py`
4. `src/loom/config.py`
5. `tests/test_orchestrator.py`
6. `tests/test_approval.py`

### C4 Acceptance Criteria
1. Every degraded planning path emits a dedicated event with reason code.
2. `deny` mode never executes fallback plans.
3. `require_approval` mode waits for operator decision before degraded execution.

## Workstream D: SQLite-Backed Remediation and Retry Lineage (`#4`)

### D1 Design
Move remediation queue state and attempt lineage into first-class tables.

Proposed tables:
1. `subtask_attempts`
2. `remediation_items`
3. `remediation_attempts`

Keep YAML metadata mirror temporarily for prompt compactness during migration.

### D2 Migration Strategy
1. Add tables with `CREATE TABLE IF NOT EXISTS`.
2. Start dual-write from orchestrator for one release.
3. Backfill unresolved metadata queue items into SQLite at task load.
4. Switch read-path source-of-truth to SQLite behind flag.
5. Keep metadata as derived cache after cutover.

### D3 Touchpoints
1. `src/loom/state/schema.sql`
2. `src/loom/state/memory.py`
3. `src/loom/engine/orchestrator.py`
4. `src/loom/state/task_state.py`
5. `tests/test_memory.py`
6. `tests/test_orchestrator.py`

### D4 Acceptance Criteria
1. Remediation item lifecycle survives process restart without loss.
2. Attempt history is queryable by `task_id`, `subtask_id`, and strategy.
3. No behavior regression in finalization semantics for blocking remediation.

## Workstream E: Durable Background Execution and Resume (`#5`)

### E1 Design
Replace ad-hoc in-memory background task launch with durable run leasing.

Introduce run management table:
1. `task_runs` with fields for `run_id`, `task_id`, `status`, `lease_owner`, `lease_expires_at`, `heartbeat_at`, `attempt`, `started_at`, `ended_at`, `last_error`.

Execution manager behavior:
1. enqueue run on task creation.
2. worker acquires lease before execution.
3. heartbeat while running.
4. on startup, reclaim expired leases and resume unfinished runs.
5. on shutdown, stop intake and release/expire leases safely.

### E2 Resume Semantics
On resume:
1. load `state.yaml`.
2. normalize any `RUNNING` subtasks to `PENDING` with resume note.
3. restart orchestrator with `reuse_existing_plan=True`.
4. preserve attempt counters and remediation queue state.

### E3 API/Engine Changes
1. move `asyncio.create_task(...)` orchestration launch behind execution manager.
2. expose run state in task response and events.
3. add internal startup reconciliation in server lifespan.

### E4 Touchpoints
1. `src/loom/api/routes.py`
2. `src/loom/api/engine.py`
3. `src/loom/api/server.py`
4. `src/loom/state/schema.sql`
5. `src/loom/state/memory.py`
6. `src/loom/engine/orchestrator.py`
7. `tests/test_api.py`
8. `tests/test_orchestrator.py`
9. `tests/test_full_integration.py`

### E5 Acceptance Criteria
1. Server restart during execution does not permanently orphan in-flight tasks.
2. Leases prevent duplicate concurrent execution of the same run.
3. Resume path is deterministic and leaves clear audit events.

## Workstream F: Idempotency for Mutating Tool Calls (`#6`)

### F1 Design
Add mutation idempotency ledger keyed by deterministic execution signature.

Proposed table:
1. `tool_mutation_ledger` with `idempotency_key`, `task_id`, `subtask_id`, `tool_name`, `args_hash`, `result_json`, `status`, `created_at`.

Idempotency key seed:
1. `run_id`
2. `subtask_id`
3. `tool_name`
4. canonicalized mutation target (path/resource)
5. normalized args hash.

### F2 Behavior
For mutating tools:
1. check ledger before execution.
2. if prior success with same key, return cached `ToolResult` and emit dedupe event.
3. if prior failure, honor retry policy (configurable).
4. record all first executions atomically.

### F3 Tool Classification
Introduce explicit mutability metadata on tool definitions:
1. `is_mutating` property on `Tool` (default `False`).
2. mark file-write/edit/delete/move and any external state mutation tools as `True`.

### F4 Touchpoints
1. `src/loom/tools/registry.py`
2. `src/loom/tools/file_ops.py`
3. `src/loom/tools/document_write.py`
4. `src/loom/engine/runner.py`
5. `src/loom/state/schema.sql`
6. `src/loom/state/memory.py`
7. `src/loom/events/types.py`
8. `tests/test_tools.py`
9. `tests/test_orchestrator.py`

### F5 Acceptance Criteria
1. Duplicate mutating tool invocations in retries/remediation are deduplicated deterministically.
2. Deduped invocations preserve downstream behavior (verification and memory extraction).
3. Ledger corruption or lookup failure fails closed with explicit safety error or falls back per configured mode.

## Workstream G: SLO-Driven Observability (`#10`)

### G1 SLO Definition Set (Initial)
1. Task success rate by process and overall.
2. Task p50/p95 duration.
3. Subtask retry rate and remediation queue exhaustion rate.
4. Budget exhaustion rate.
5. Planner degradation rate.
6. Resume recovery success rate.
7. Mutating-call dedupe rate.

### G2 Telemetry Enhancements
1. ensure every event carries `run_id` once available.
2. add explicit events:
   - `task_budget_exhausted`
   - `task_plan_degraded`
   - `task_run_acquired`
   - `task_run_heartbeat`
   - `task_run_recovered`
   - `tool_call_deduplicated`
3. add periodic SLO rollup job from SQLite event/task tables.

### G3 Surface Area
1. add API endpoint(s) for SLO snapshot and recent breaches.
2. add structured run summary payload with reason-code counts.
3. optional webhook for SLO breach notifications.

### G4 Touchpoints
1. `src/loom/events/types.py`
2. `src/loom/events/bus.py`
3. `src/loom/engine/orchestrator.py`
4. `src/loom/state/memory.py`
5. `src/loom/api/routes.py`
6. `tests/test_events.py`
7. `tests/test_api.py`
8. `docs/CONFIG.md`

### G5 Acceptance Criteria
1. Operators can answer core SLO questions from API and persisted telemetry without manual log spelunking.
2. SLO computations are reproducible from persisted data.
3. Alerting path is rate-limited and avoids duplicate flood events.

## Implementation Order and Dependencies
1. F0 foundation (`run_id`, flags, event field normalization).
2. D schema expansion and data access methods (needed by E and F).
3. A global budget manager.
4. B completion contract.
5. C planner fallback policy.
6. E durable execution and resume.
7. F idempotency ledger.
8. G SLO surfaces and alerting.

Dependency notes:
1. `#5` depends on `#4` for durable run state.
2. `#6` depends on `#4` for ledger persistence.
3. `#10` depends on `#1/#3/#5/#6` for meaningful indicators.

## Testing Strategy

### Unit Tests
1. Budget counters and breach detection.
2. Completion contract parser and enforce-mode control flow.
3. Planner fallback policy matrix.
4. DB CRUD for remediation/retry/run/idempotency tables.
5. Idempotency key generation and dedupe behavior.

### Integration Tests
1. Task crash/resume with preserved state and no duplicate side effects.
2. Replan/remediation execution across process restart.
3. End-to-end degraded planning with approval gating.
4. SLO rollup correctness from synthetic event streams.

### Regression Tests
1. Legacy execution path with all new flags off.
2. Existing process contract behavior unchanged.
3. Existing API responses remain backward compatible (except additive fields).

## Rollout and Risk Control
1. Ship all features behind flags default-off except additive telemetry fields.
2. Enable in this order: `#4` dual-write, `#1`, `#2 warn`, `#3 allow`, `#5`, `#6`, `#10`, then tighten policies.
3. Keep one-release dual-write period for remediation/retry state before read-path cutover.
4. Add rollback playbook per flag in `docs/CONFIG.md`.
5. Gate strict modes (`completion enforce`, `planner deny`) only after baseline stability metrics hold for one release cycle.

## Operational Risks and Mitigations
1. Risk: schema drift across upgrades.
   Mitigation: additive DDL only, startup schema check event, compatibility tests.
2. Risk: duplicate task execution during transition.
   Mitigation: lease acquisition and heartbeat with atomic status transitions.
3. Risk: idempotency false positives on semantically different calls.
   Mitigation: canonical key includes normalized target and args hash; emit debug payload on dedupe.
4. Risk: stricter completion protocol increases loop iterations.
   Mitigation: `warn` mode first; monitor budget pressure and adjust prompt contract.
5. Risk: SLO rollup overhead.
   Mitigation: bounded-window aggregation, indexed query paths, optional background cadence.

## Definition of Done
1. All seven requested areas are implemented behind controlled flags.
2. Crash/restart recovery test passes in CI.
3. Mutation dedupe test passes for representative mutating tools.
4. SLO endpoint returns stable, validated metrics.
5. Documentation updated for configuration, rollout, and rollback.
