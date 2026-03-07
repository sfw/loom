# 2026-03-07 TUI App Split Plan

## File
- `src/loom/tui/app.py` (~16,189 LOC)

## Goal
- Break the monolithic TUI app into a package with feature-focused modules while preserving:
  - `from loom.tui.app import LoomApp`
  - exported helper types (`ProcessRunList`, `ProcessRunPane`, `ProcessRunState`, etc.)
  - UI behavior, keyboard bindings, slash command behavior, and process-run flow.

## Current Shape
- `LoomApp` alone spans ~15k lines with ~446 methods.
- Large hotspot areas:
  - slash command routing/handling (`_handle_slash_command`, ~933 LOC)
  - ad hoc process synthesis (`_synthesize_adhoc_process`, ~515 LOC)
  - process-run launch/execution lifecycle
  - chat history hydration/replay
  - steering/stop/redirect controls
  - rendering/panel refresh logic

## Target Layout (Package-First)
- `src/loom/tui/app/__init__.py`
  - Public facade and stable exports.
- `src/loom/tui/app/constants.py`
  - slash specs, CSS/defaults, process status dictionaries, limits.
- `src/loom/tui/app/models.py`
  - dataclasses and small value objects (`ProcessRunState`, `SteeringDirective`, etc.).
- `src/loom/tui/app/widgets.py`
  - `ProcessRunList`, `ProcessRunPane` (or move to `loom.tui.widgets/` with re-export shim).
- `src/loom/tui/app/core.py`
  - `LoomApp` class shell, composition/mount/key hooks, high-level orchestration.
- `src/loom/tui/app/slash/`
  - `parsing.py` (argument splitting/parsing)
  - `registry.py` (command index/spec lookup)
  - `handlers.py` (dispatch entrypoint + per-command handler routing)
- `src/loom/tui/app/process_runs/`
  - `state.py` (run-state mutation helpers)
  - `launch.py` (launch preflight/auth/workspace preparation)
  - `events.py` (progress/event formatting and mapping)
- `src/loom/tui/app/chat/`
  - `session.py` (init/new/resume/session persistence)
  - `turns.py` (run-turn loop, interaction execution)
  - `history.py` (history hydration and pagination)
  - `steering.py` (pause/inject/redirect/stop controls)
- `src/loom/tui/app/rendering.py`
  - chat/event/process row rendering and panel updates.
- `src/loom/tui/app/actions.py`
  - `action_*` handlers and key-driven action routing.

## Incremental Plan

### Phase 1: Package Scaffolding and Pure Extractions
- Perform atomic move: `git mv src/loom/tui/app.py src/loom/tui/app/__init__.py`.
- Convert `loom.tui.app` into a package with `app/__init__.py` facade.
- Move constants/dataclasses/pure formatting helpers first.

### Phase 2: Widget + Process-Run State Extraction
- Move `ProcessRunList` and `ProcessRunPane`.
- Move process-run state mutation helpers into `process_runs/state.py`.

### Phase 3: Slash Command Subsystem
- Move slash parsing, registry/index refresh, and command handler dispatch into `slash/`.
- Keep one compatibility method in `LoomApp` delegating to slash handler entrypoint.

### Phase 4: Chat and Session Subsystem
- Move session/bootstrap/history helpers into `chat/session.py` and `chat/history.py`.
- Move turn execution/interaction logic into `chat/turns.py`.

### Phase 5: Process-Run Launch and Event Flows
- Move launch preflight and progress-event formatting to `process_runs/launch.py` + `events.py`.

### Phase 6: Rendering and Actions
- Move rendering methods and `action_*` methods into dedicated modules.
- Keep `LoomApp` as orchestration shell.

### Phase 7: Final Cleanup
- Remove temporary wrappers once tests target new module seams.
- Keep `app/__init__.py` export compatibility stable.

## Risks and Hardening
- Risk: UI behavior drift from event ordering/state transitions.
  - Hardening: add snapshot-style tests for key rendered outputs and process progress rows.
- Risk: Slash command regressions.
  - Hardening: move parser/dispatch as pure functions first; add exhaustive command tests.
- Risk: Circular imports due heavy cross-feature references.
  - Hardening: enforce strict dependency direction (constants/models -> helpers -> handlers -> core).

## Adherence to Prior Plan Standards
- Import contract:
  - Preserve `from loom.tui.app import LoomApp, ProcessRunPane, ProcessRunList, ProcessRunState, ProcessRunLaunchRequest, SteeringDirective`.
- Facade import rule:
  - Internal submodules must not import through `loom.tui.app` facade; import concrete sibling modules directly.
- Revertable slices:
  - One subsystem extraction per PR (slash, chat, process runs, rendering/actions).
- Import-cycle guardrails:
  - Add `tests/tui/app/test_import_cycles.py` for new package boundaries.
- Behavior parity guardrails:
  - Snapshot slash command help/hints and process progress row rendering before/after.

## Test Strategy
- Existing:
  - `tests/test_tui.py`
  - `tests/test_setup.py` (imports `LoomApp`)
- New (non-overlapping package):
  - `tests/tui/app/test_import_contracts.py`
  - `tests/tui/app/test_import_cycles.py`
  - `tests/tui/app/test_slash_parsing.py`
  - `tests/tui/app/test_process_runs_state.py`
  - `tests/tui/app/test_chat_session_flow.py`
  - `tests/tui/app/test_rendering_rows.py`

## Exit Criteria
- `src/loom/tui/app/__init__.py` <= 250 LOC.
- `LoomApp` shell in `core.py` <= 2,500 LOC.
- Slash commands and process-run flows extracted to dedicated subpackages.
- Existing TUI tests still pass with no behavior regressions.
