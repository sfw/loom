# 2026-03-07 Engine Orchestrator Split Plan

## File

- `src/loom/engine/orchestrator.py` (~9,955 LOC, 197 methods)

## Objective

- Break `Orchestrator` into cohesive subsystems without changing runtime behavior, event ordering, or public import paths.
- Preserve existing external API:
  - `from loom.engine.orchestrator import Orchestrator, create_task`
  - Re-exported runner types (`SubtaskResult`, `ToolCallRecord`)

## Current Hotspots

- Task lifecycle and loop control:
  - `execute_task` (~318 LOC)
  - `_dispatch_subtask` (~378 LOC)
  - `_handle_failure` (~386 LOC)
- Planning and replanning:
  - `_plan_task`, `_replan_task`, `_align_plan_output_coordination`
- Remediation + queueing:
  - `_run_confirm_or_prune_remediation` (~514 LOC)
  - `_process_remediation_queue`
- Output coordination and finalizer logic:
  - `_select_conflict_safe_batch`, `_commit_finalizer_stage_publish`
- Validity, claims, evidence, telemetry:
  - temporal/synthesis gates, artifact seals, scorecards, CSV export

## Initial Split Plan

### Target Package Layout

- `src/loom/engine/orchestrator/`
  - Package replacing monolithic `orchestrator.py` module.
- `src/loom/engine/orchestrator/__init__.py`
  - Compatibility facade containing `Orchestrator`, `create_task`, and re-exported runner types.
- `src/loom/engine/orchestrator/core.py`
  - Lifecycle shell: `__init__`, `execute_task`, pause/resume/cancel entry points.
- `src/loom/engine/orchestrator/planning.py`
  - Plan parsing/normalization, topology checks, phase coordination, replanning contract.
- `src/loom/engine/orchestrator/dispatch.py`
  - Subtask dispatch, iteration-aware success/failure handoff, stale-outcome handling.
- `src/loom/engine/orchestrator/remediation.py`
  - Remediation queue lifecycle, confirm-or-prune mode, queue persistence/hydration.
- `src/loom/engine/orchestrator/output.py`
  - Output conflict scheduling, phase finalizer staging/publish rules, manifest checks.
- `src/loom/engine/orchestrator/validity.py`
  - Claim graph updates, temporal/synthesis gates, required fact-checker enforcement.
- `src/loom/engine/orchestrator/evidence.py`
  - Artifact seals, subtask evidence persistence, validity scorecards, evidence ledger CSV.
- `src/loom/engine/orchestrator/telemetry.py`
  - Run-level telemetry rollups and event payload normalization.
- `src/loom/engine/orchestrator/budget.py`
  - `_RunBudget` extracted as dedicated class.

### Refactor Pattern

- Keep a single `Orchestrator` state owner.
- Move behavior into helper collaborators that receive a narrow context object:
  - `OrchestratorContext(state, config, events, runner, verification, retry, question, process, scheduler, ...)`
- Keep private method wrappers on `Orchestrator` during transition so tests touching private methods do not break immediately.

## Incremental Execution Plan

### Phase 1: Establish Seams (No Behavior Change)

- Convert `orchestrator.py` into `orchestrator/` package with `__init__.py` facade.
- Create extracted modules with interfaces and pass-through wrappers.
- Move `_RunBudget` to `orchestrator/budget.py`.
- Add smoke tests ensuring `Orchestrator` imports and basic `execute_task` path still work.

### Phase 2: Extract Pure/Low-Risk Utilities

- Move normalization/conversion helpers (`_to_bool`, `_to_ratio`, path normalization, metadata compaction).
- Move telemetry aggregation helpers.
- Move evidence CSV formatting helpers.
- Add unit tests for moved pure helpers with fixture parity from current behavior.

### Phase 3: Extract Output Coordination Subsystem

- Move conflict-safe batch selection and finalizer staging/publish methods.
- Add focused tests for:
  - conflict deferral/starvation signaling
  - finalizer manifest-only input violations
  - stage publish sealing behavior

### Phase 4: Extract Remediation Subsystem

- Move remediation queue state machine and DB sync/hydration.
- Add tests for queue due/expired semantics, bounded backoff, and retry context shaping.

### Phase 5: Extract Planning/Replanning Subsystem

- Move plan building/validation/replan contract enforcement.
- Add tests for topology detection, degraded planning fallback, and phase-mode application.

### Phase 6: Extract Validity/Evidence Subsystem

- Move claim lifecycle, temporal consistency, synthesis gate, artifact/evidence scorecard logic.
- Add tests for:
  - reason-code normalization
  - temporal contradiction detection
  - claim lifecycle attachment and pruning
  - artifact seal validation/backfill

### Phase 7: Shrink Facade and Stabilize

- Keep only compatibility exports in `orchestrator/__init__.py`.
- Retain compatibility wrappers for high-churn private APIs until downstream tests no longer require them.
- Run full `tests/test_orchestrator.py` and integration suites; fix ordering regressions.

## Execution Status

- Phase 1a complete (2026-03-06 local): atomic module->package conversion (`orchestrator.py` -> `orchestrator/__init__.py`) with orchestrator/import-contract parity tests.
- Phase 1 complete (2026-03-06 local): extracted `_RunBudget` to `orchestrator/budget.py` and added seam scaffolding modules (`core`, `planning`, `dispatch`, `remediation`, `output`, `validity`, `evidence`, `telemetry`) with import-smoke parity tests.
- Phase 2 complete (2026-03-06 local): extracted pure validity/metadata-compaction helpers to `orchestrator/validity.py`, evidence CSV formatting helpers to `orchestrator/evidence.py`, telemetry rollup/event-count helpers to `orchestrator/telemetry.py`, and output-coordination policy resolvers to `orchestrator/output.py` with focused parity tests.
- Phase 3 complete (2026-03-06 local): moved output conflict-prioritization/selection, canonical deliverable-path normalization, and finalizer fan-in staging/publish + manifest-input helpers to `orchestrator/output.py` with output/fan-in parity tests.
- Phase 3a complete (2026-03-07 local): extracted iteration dispatch helpers (`_iteration_retry_mode`, invocation/runtime observers) to `orchestrator/dispatch.py` with compatibility wrappers on `Orchestrator`.
- Phase 1b complete (2026-03-07 local): moved task factory `create_task` into `orchestrator/core.py` and re-exported it through the package facade.
- Phase 4a complete (2026-03-07 local): extracted remediation queue pure helpers (`_remediation_queue_limits`, bounded backoff, ISO datetime parse, due/expiry checks) into `orchestrator/remediation.py` with compatibility wrappers retained on `Orchestrator`.
- Phase 5a complete (2026-03-07 local): extracted planning-mode helpers (`_phase_mode`, `_topology_retry_attempts`, `_planner_degraded_mode`) into `orchestrator/planning.py` with compatibility wrappers retained on `Orchestrator`.
- Program status correction (2026-03-07 local): orchestrator split is still partial. Large lifecycle/remediation/planning methods remain in `orchestrator/__init__.py`, and facade-thinning exit criteria are not yet met.
- Phase 4
- Phase 5
- Phase 6
- Phase 7

## Critique of Initial Plan

- Risk: Too many modules can recreate complexity in import wiring and hidden coupling.
- Risk: Private method extraction can break tests that call `Orchestrator._foo` directly.
- Risk: Event ordering is part of behavioral contract; extraction may reorder emissions.
- Risk: Shared mutable task metadata can drift if multiple collaborators mutate in different phases.
- Risk: Static calls to `SubtaskRunner` helpers from orchestrator output logic can cause awkward cross-module coupling.

## Hardened Plan

- Collapse module count to 6 operational collaborators (planning, dispatch, remediation, output, validity/evidence, telemetry) plus budget.
- Use one explicit `TaskMutationJournal` helper to centralize metadata writes in deterministic order.
- Introduce event-sequence golden tests before extraction for:
  - normal success flow
  - failure + remediation flow
  - replan flow
  - paused/cancelled flow
- Keep compatibility wrappers for private methods with deprecation comments; remove only after test migration.
- Add `OrchestratorInvariants` checks in debug/test mode:
  - plan version monotonicity
  - subtask status transition legality
  - run_id consistency across events

## Missing Items Added

- Package-conversion mechanics:
  - Perform an atomic move `git mv src/loom/engine/orchestrator.py src/loom/engine/orchestrator/__init__.py` before internal extractions so `loom.engine.orchestrator` import path remains stable.
  - Keep `__all__` parity in `orchestrator/__init__.py` (`Orchestrator`, `create_task`, `SubtaskResult`, `ToolCallRecord`).
- Cross-plan sequencing:
  - Execute orchestrator extraction after runner/verification package conversions stabilize to reduce concurrent import-churn in core control flow.
- Circular import guardrails:
  - Ban importing from `orchestrator/__init__.py` inside orchestrator submodules; import concrete siblings (`from .dispatch import ...`) only.
  - Add one CI check that imports each orchestrator submodule in isolation.
- PR slicing and rollback:
  - Split into small PRs per phase (max one subsystem extraction + tests).
  - Keep each phase revertable without DB/state migration steps.
- Observability parity:
  - Snapshot event payload keys for orchestrator terminal events and run summary events before refactor; require parity after each phase.

## Test Strategy

- Existing suites:
  - `tests/test_orchestrator.py`
  - `tests/test_full_integration.py`
  - `tests/test_integrations.py`
  - `tests/test_iteration_gates.py`
- New targeted suites:
  - `tests/test_orchestrator_output.py`
  - `tests/test_orchestrator_remediation.py`
  - `tests/test_orchestrator_validity.py`
  - `tests/test_orchestrator_event_ordering.py`
  - `tests/test_engine_import_contracts.py` (import/re-export parity for `loom.engine.orchestrator`)

## Exit Criteria

- `src/loom/engine/orchestrator/__init__.py` remains thin facade (<= 200 LOC).
- `src/loom/engine/orchestrator/core.py` reduced to <= 2,500 LOC.
- No regression in event ordering golden tests.
- No public import-path breakage.
- All orchestrator/integration tests green.
