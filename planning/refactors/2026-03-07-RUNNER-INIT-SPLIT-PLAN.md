# 2026-03-07 Runner Init Split Plan

## File
- `src/loom/engine/runner/__init__.py` (~3,068 LOC)

## Goal
- Reduce `runner/__init__.py` to a thin API facade while preserving:
  - `from loom.engine.runner import SubtaskRunner, SubtaskResult, ToolCallRecord`
  - current subtask execution behavior
  - ask-user runtime behavior
  - artifact/policy guard behavior

## Current Shape
- Single `SubtaskRunner` class with ~98 methods.
- Existing helper modules already present: `compaction.py`, `memory.py`, `policy.py`, `session.py`, `telemetry.py`, `settings.py`, `types.py`.
- Largest hotspot remains `run()` (~879 LOC), with heavy inline orchestration.

## Target Layout
- `src/loom/engine/runner/__init__.py`
  - Facade exports only.
- `src/loom/engine/runner/core.py`
  - `SubtaskRunner` shell and dependency wiring.
- `src/loom/engine/runner/execution/`
  - `loop.py` (main run loop flow)
  - `model_io.py` (model invocation/retry/stream)
  - `tool_calls.py` (tool call dispatch + ask_user branch)
- `src/loom/engine/runner/guards/`
  - `deliverables.py` (path/deliverable write policy)
  - `artifact_seals.py` (seal mutation validation + reseal flow)
- `src/loom/engine/runner/messages/`
  - `serialize.py` (tool-call/result serialization)
  - `summaries.py` (summary compaction helpers)
- Keep existing modules:
  - `compaction.py`, `memory.py`, `policy.py`, `session.py`, `telemetry.py`, `settings.py`, `types.py`

## Incremental Plan

### Phase 1: Core Facade Cut
- Create `core.py` and move class definition there.
- Leave `runner/__init__.py` as import facade.
 - Preserve `__all__` and top-level export behavior in `runner/__init__.py`.

### Phase 2: Execution Loop Extraction
- Move run-loop orchestration into `execution/loop.py`.
- Keep `SubtaskRunner.run()` as a thin delegator.

### Phase 3: Tool and Model Flow Extraction
- Move model invocation/streaming to `execution/model_io.py`.
- Move tool execution branches (including ask_user limits) to `execution/tool_calls.py`.

### Phase 4: Policy/Seal Guard Extraction
- Move deliverable/seal logic into `guards/` modules.
- Keep static wrappers on `SubtaskRunner` during transition for tests.

### Phase 5: Message/Summary Extraction
- Move serialization and summary helper methods to `messages/`.
- Keep compaction routing consistent with current `compaction.py`.

### Phase 6: Final Shrink
- Minimize `SubtaskRunner` to coordination and dependency injection.
- Remove temporary wrappers once tests no longer reference private methods directly.

## Risks and Hardening
- Risk: Async cancellation and cleanup regressions.
  - Hardening: enforce one `try/finally` cleanup path for deadline/contextvars.
- Risk: Behavior drift in ask_user runtime branch.
  - Hardening: isolate ask_user logic and add characterization tests before moving.
- Risk: Orchestrator tests calling private helpers break.
  - Hardening: keep compatibility wrappers until tests are migrated.

## Adherence to Prior Plan Standards
- Facade import rule:
  - New runner internals must not import from `loom.engine.runner` (`__init__.py`) to avoid circular imports.
- Shared engine compatibility contract:
  - Keep central import contract checks in `tests/test_engine_import_contracts.py` for runner exports.
- Revertable slices:
  - One PR per extraction stage (core cut, execution split, guards split, message split).
- Observability parity:
  - Snapshot key runner model/tool telemetry payload fields before/after extraction.
- Module conversion note:
  - `runner` is already a package; no `git mv <module>.py -> <module>/__init__.py` step required.

## Test Strategy
- Existing:
  - `tests/test_orchestrator.py`
  - `tests/test_questions.py`
  - `tests/test_multimodal_integration.py`
  - `tests/test_iteration_gates.py`
- New (non-overlapping package):
  - `tests/test_engine_import_contracts.py` (shared engine export checks)
  - `tests/engine/runner/test_import_contracts.py`
  - `tests/engine/runner/test_import_cycles.py`
  - `tests/engine/runner/test_execution_loop.py`
  - `tests/engine/runner/test_tool_calls.py`
  - `tests/engine/runner/test_guards.py`

## Exit Criteria
- `runner/__init__.py` <= 200 LOC.
- `SubtaskRunner.run()` <= 250 LOC orchestration shell.
- Public import contracts unchanged.
- Existing runner-related tests pass.
