# 2026-03-07 Test Orchestrator Split Plan

## File
- `tests/test_orchestrator.py` (~6,801 LOC)

## Goal
- Split the monolithic orchestrator test file into coherent, non-overlapping modules under a dedicated package.
- Preserve test behavior and make execution ergonomics explicit: run the full suite via `tests/orchestrator/`.
- Sweep and update legacy references to `tests/test_orchestrator.py` across the repo.

## Current Shape
- 13 classes in one file, with major heavy blocks:
  - `TestOrchestratorExecution` (~2,396 LOC, 42 tests)
  - `TestSubtaskRunnerContextBudget` (~1,261 LOC, 31 tests)
  - `TestOrchestratorValidityPolicy` (~1,085 LOC, 17 tests)
- Several factory helpers are top-level, shared across all classes.

## Target Test Package Layout
- `tests/orchestrator/conftest.py`
  - shared fixtures/factories now in top-level helpers (`_make_*`).
- `tests/orchestrator/test_types_and_creation.py`
  - `TestToolCallRecord`, `TestSubtaskResult`, `TestCreateTask`.
- `tests/orchestrator/test_planning.py`
  - `TestOrchestratorPlan`.
- `tests/orchestrator/test_phase_mode.py`
  - `TestOrchestratorProcessPhaseMode`.
- `tests/orchestrator/test_critical_path_behavior.py`
  - `TestOrchestratorCriticalPathBehavior`.
- `tests/orchestrator/test_execution_loop.py`
  - `TestOrchestratorExecution`.
- `tests/orchestrator/test_finalize.py`
  - `TestOrchestratorFinalize`.
- `tests/orchestrator/test_workspace_scan.py`
  - `TestWorkspaceDocumentScan`.
- `tests/orchestrator/test_runner_context_budget.py`
  - `TestSubtaskRunnerContextBudget`.
- `tests/orchestrator/test_iteration_loops.py`
  - `TestIterationLoops`.
- `tests/orchestrator/test_validity_policy.py`
  - `TestOrchestratorValidityPolicy`.
- `tests/orchestrator/test_todo_reminder.py`
  - `TestOrchestratorTodoReminder`.

## Naming and Boundary Rules

1. All orchestrator tests live under `tests/orchestrator/`.
2. No duplicate coverage with other suites (`tests/test_verification.py`, `tests/test_iteration_gates.py`, etc.).
3. Shared fixture logic moves to `tests/orchestrator/conftest.py`; domain-specific helpers remain local.
4. Keep module names domain-based, not method-count-based.

## Incremental Plan

### Phase 1: Package Scaffolding
- Create `tests/orchestrator/` and `conftest.py`.
- Move low-risk type/create/planning tests first.

### Phase 2: Core Execution and Finalization Blocks
- Move `TestOrchestratorExecution` and `TestOrchestratorFinalize`.
- Ensure fixtures remain stable and deterministic.

### Phase 3: Policy/Iteration/Runner-Coupling Blocks
- Move validity policy, iteration loop, and runner context budget classes.
- Keep private helper access compatibility where tests currently rely on it.

### Phase 4: Legacy Path Sweep and Command Migration
- Replace references to `tests/test_orchestrator.py` with `tests/orchestrator` in:
  - `.github/workflows/`
  - `scripts/`
  - developer docs/runbooks (`README`, `docs/`, active plans/checklists)
  - local helper scripts/aliases where present.
- Standardize invocation guidance to:
  - `uv run pytest tests/orchestrator`
- Keep a short-lived compatibility shim `tests/test_orchestrator.py` only if needed during migration window, then remove.

### Phase 5: Cleanup and Enforcement
- Remove/reduce legacy monolithic file.
- Add a guard check (script or test) that fails on new references to `tests/test_orchestrator.py` outside approved archival docs/changelog.
- Confirm `pytest --collect-only tests/orchestrator` has no duplicate nodeids.

## Legacy Reference Sweep Specification

1. Detection command:
   - `rg -n "tests/test_orchestrator\\.py|test_orchestrator\\.py" . --glob '!planning/**'`
2. Update policy:
   - Runtime/CI/doc references should point to `tests/orchestrator` (module directory), not old file path.
3. Optional archival allowlist:
   - historical changelog entries and immutable archived notes can keep old path text.

## Risks and Hardening
- Risk: Fixture behavior drift after helper extraction.
  - Hardening: move-only commit first, fixture-centralization commit second.
- Risk: Partial migration leaves split + monolith both active.
  - Hardening: collect-only gate and duplicate-nodeid check after each phase.
- Risk: CI/scripts still execute old path.
  - Hardening: explicit reference sweep phase + enforcement check.

## Test Strategy
- Per-module during migration:
  - `uv run pytest tests/orchestrator/<module>.py`
- Full orchestrator package:
  - `uv run pytest tests/orchestrator`
- Integration sanity:
  - `uv run pytest tests/test_iteration_gates.py tests/test_verification.py`
- Collection uniqueness:
  - `uv run pytest --collect-only tests/orchestrator`

## Exit Criteria
- `tests/test_orchestrator.py` removed or reduced to temporary shim only.
- Full orchestrator suite is runnable as:
  - `uv run pytest tests/orchestrator`
- Legacy references to `tests/test_orchestrator.py` updated in runtime/CI/docs.
- No duplicate nodeids and no behavior regressions.
