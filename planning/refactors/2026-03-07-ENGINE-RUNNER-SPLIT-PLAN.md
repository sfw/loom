# 2026-03-07 Engine Runner Split Plan

## File
- `src/loom/engine/runner.py` (~4,814 LOC, 98 `SubtaskRunner` methods)

## Objective
- Split `SubtaskRunner` into focused components while preserving tool-loop behavior, ask-user semantics, artifact policy enforcement, and compaction outcomes.
- Keep public imports stable:
  - `ToolCallRecord`
  - `SubtaskResultStatus`
  - `SubtaskResult`
  - `SubtaskRunner`

## Current Hotspots
- Constructor/config hydration:
  - `__init__` (~452 LOC)
- Main execution loop:
  - `run` (~880 LOC)
- Compaction/overflow handling:
  - `_compact_messages_for_model_tiered` (~286 LOC)
  - `_compact_messages_for_model_legacy` (~211 LOC)
  - `_rewrite_tool_payload_for_overflow` (~127 LOC)
- Policy and mutation safety:
  - `_validate_deliverable_write_policy`
  - `_validate_sealed_artifact_mutation_policy`
  - `_reseal_tracked_artifacts_after_mutation`
- Memory extraction:
  - `_extract_memory` (~150 LOC)
- Telemetry/event logic spread across runner

## Initial Split Plan

### Target Package Layout
- `src/loom/engine/runner/`
  - Package replacing monolithic `runner.py` module.
- `src/loom/engine/runner/__init__.py`
  - Compatibility facade; exports `SubtaskRunner` and runner result types.
- `src/loom/engine/runner/types.py`
  - `ToolCallRecord`, `SubtaskResultStatus`, `SubtaskResult`, compaction enums/plan dataclass.
- `src/loom/engine/runner/settings.py`
  - `RunnerSettings.from_config(config)` to replace giant constructor parsing.
- `src/loom/engine/runner/session.py`
  - Per-subtask mutable execution state (`messages`, counters, deadlines, evidence buffers, loop flags).
- `src/loom/engine/runner/execution.py`
  - Core tool-calling loop and response handling.
- `src/loom/engine/runner/policy.py`
  - Deliverable path policy, sealed artifact mutation policy, resealing.
- `src/loom/engine/runner/compaction.py`
  - Compaction plan construction, tiered/legacy compaction, overflow fallback.
- `src/loom/engine/runner/memory.py`
  - Memory extraction and parsing pipeline.
- `src/loom/engine/runner/telemetry.py`
  - Model/tool/artifact telemetry emission and payload normalization.

### Refactor Pattern
- Keep `SubtaskRunner` as orchestration shell delegating to collaborators.
- Collaborators receive explicit dependencies (`tools`, `router`, `memory`, settings), not the full runner.
- Keep selected static helper wrappers on `SubtaskRunner` for existing tests that call private methods directly.

## Incremental Execution Plan

### Phase 1: Stabilize Interfaces
- Convert `runner.py` into `runner/` package with `__init__.py` facade.
- Extract `runner/types.py`; re-export from `runner/__init__.py`.
- Add `RunnerSettings` and swap constructor internals to settings object.
- No behavior change; parity tests only.

### Phase 2: Extract Policy Layer
- Move deliverable/forbidden path logic, mutation idempotency key helpers, and reseal policy checks.
- Add targeted tests:
  - canonical path normalization
  - forbidden path detection
  - sealed artifact mutation allow/deny behavior

### Phase 3: Extract Compaction Layer
- Move compaction plan classification and tiered/legacy compaction routines.
- Move overflow fallback rewrite into the same module.
- Add tests around:
  - compaction-stage selection
  - cache/no-gain behavior
  - overflow fallback correctness for tool message payloads

### Phase 4: Extract Telemetry Layer
- Move telemetry event builders and artifact telemetry emitters.
- Add event payload contract tests (keys and normalized values).

### Phase 5: Extract Memory Layer
- Move memory extraction, tool trace summarization, and parsing utilities.
- Add tests for extractor prompt shaping, timeout guard behavior, and parser resilience.

### Phase 6: Introduce `RunnerSession` and Shrink `run`
- Replace ad hoc local variables with `RunnerSession`.
- Keep tool-loop behavior identical (same iteration order, same interruption checks).
- Ensure contextvar and deadline cleanup in `finally` path via dedicated session guard.

### Phase 7: Final Facade Cleanup
- Keep `SubtaskRunner` as integration shell + compatibility wrappers.
- Remove redundant utility duplication between runner and collaborator modules.

## Execution Status

- [x] Phase 1a complete (2026-03-06 local): atomic module->package conversion (`runner.py` -> `runner/__init__.py`) with import-contract and runner-consumer parity tests.
- [x] Phase 1b complete (2026-03-06 local): extracted `runner/types.py` and `runner/settings.py`; constructor config hydration now delegates to `RunnerSettings.from_config(...)` with parity tests.
- [x] Phase 2 complete (2026-03-06 local): policy helpers extracted to `runner/policy.py` with deliverable/forbidden/sealed mutation parity tests.
- [x] Phase 3 complete (2026-03-06 local): extracted compaction helpers/routines to `runner/compaction.py` (overflow rewrite + tiered/legacy orchestration) with parity coverage.
- [x] Phase 4 complete (2026-03-06 local): telemetry helpers extracted to `runner/telemetry.py` with payload contract parity tests.
- [x] Phase 5 complete (2026-03-06 local): memory extraction/parsing extracted to `runner/memory.py` with timeout-guard, prompt-shaping, and parser resilience parity tests.
- [x] Phase 6 complete (2026-03-06 local): introduced `runner/session.py`, rewired `run()` mutable loop state to `RunnerSession`, and enforced deadline/telemetry/compactor context cleanup via single `try/finally` guard.
- [x] Phase 6b complete (2026-03-07 local): extracted the execution loop into `runner/execution.py`; `SubtaskRunner.run()` in `runner/__init__.py` now delegates through a thin compatibility wrapper while preserving monkeypatch seam compatibility for `build_run_auth_context`.
- [x] Phase 7 complete (2026-03-06 local): finalized facade cleanup by moving remaining path/target/variant policy utilities into `runner/policy.py` and keeping `SubtaskRunner` wrappers for compatibility with focused policy parity tests.

## Critique of Initial Plan
- Risk: Over-abstracting async loop internals can hide control flow and introduce subtle cancellation regressions.
- Risk: Tests currently reach deep private helpers on `SubtaskRunner`; naive extraction will break them.
- Risk: `run` mixes state from many concerns (tool calls, evidence ids, ask-user counters, compaction stats); splitting can desynchronize state.
- Risk: Compaction and overflow fallback are tightly coupled to model invocation retries; splitting may break retry semantics.
- Risk: telemetry currently piggybacks on runtime-local fields; moving it can drop counters or context fields.

## Hardened Plan
- Keep execution loop in one module (`runner/execution.py`) and avoid splitting it across multiple services.
- Introduce `RunnerSession` first, before moving loop logic, to make state transitions explicit.
- Preserve `SubtaskRunner` method names as thin wrappers for compatibility during migration.
- Add characterization tests before extraction for:
  - ask-user timeout/limit behavior
  - model overflow fallback flow
  - interruption paths (cancel/pause/timeout)
  - mutation idempotency behavior
- Enforce cleanup invariant with a single `try/finally` block for:
  - `_subtask_deadline_monotonic`
  - `_active_subtask_telemetry_counters`
  - compactor contextvar reset

## Missing Items Added
- Package-conversion mechanics:
  - Perform `git mv src/loom/engine/runner.py src/loom/engine/runner/__init__.py` first to preserve `loom.engine.runner` imports.
  - Preserve exported symbols and `__all__` semantics in `runner/__init__.py`.
- Shared helper decoupling:
  - Extract path/policy helpers currently used cross-module (for example by orchestrator) into a neutral module (`runner/path_policy.py` or `engine/path_policy.py`) to avoid deep static method reach-ins on `SubtaskRunner`.
- Cross-plan sequencing:
  - Land runner package conversion before orchestrator extraction phases that depend on runner internals.
  - Coordinate with semantic compactor split so runner compaction hooks do not change in the same PR as execution-loop changes.
- Circular import guardrails:
  - Disallow internal imports from `runner/__init__.py`; submodules import siblings directly.
  - Add import-cycle smoke test for all runner submodules.
- Runtime safety:
  - Add per-phase latency and token-usage regression check on representative orchestrator integration tests.

## Test Strategy
- Existing suites:
  - `tests/test_orchestrator.py` (runner helpers referenced directly)
  - `tests/test_questions.py`
  - `tests/test_multimodal_integration.py`
  - `tests/test_iteration_gates.py`
- New focused suites:
  - `tests/test_runner_execution.py`
  - `tests/test_runner_policy.py`
  - `tests/test_runner_compaction.py`
  - `tests/test_runner_telemetry.py`
  - `tests/test_runner_memory.py`
  - `tests/test_engine_import_contracts.py` (import/re-export parity for `loom.engine.runner`)

## Exit Criteria
- `src/loom/engine/runner/__init__.py` remains thin facade (<= 200 LOC).
- `src/loom/engine/runner/execution.py` reduced to <= 1,800 LOC.
- `run` reduced to <= 300 LOC shell orchestrating collaborators.
- No regression in ask-user, overflow fallback, artifact policy, or telemetry payload tests.
- Public type imports unchanged.
