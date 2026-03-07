# 2026-03-07 Test TUI Split Plan

## File
- `tests/test_tui.py` (~14,778 LOC)

## Goal
- Split the monolithic TUI test file into coherent, non-overlapping test modules.
- Keep outcomes intuitive: each test file should map to one product surface.
- Avoid overlap with existing test files outside TUI scope (`tests/test_cli.py`, `tests/test_api.py`, etc.).

## Current Shape
- 43 test classes in one file.
- Major heavy blocks:
  - `TestProcessSlashCommands` (~5,837 LOC)
  - `TestCommandPaletteProcessActions` (~1,938 LOC)
  - `TestAuthAndMCPManagerScreens` (~1,345 LOC)
- Many repeated `from loom.tui.app import LoomApp` imports inside methods.

## Target Test Package Layout
- `tests/tui/conftest.py`
  - shared fixtures and common app/mocks factories for TUI-only tests.
- `tests/tui/test_api_client.py`
  - `TestLoomAPIClient`.
- `tests/tui/test_tool_call_previews.py`
  - preview/formatting helpers (`TestToolArgsPreview`, `TestToolOutputPreview`, `TestTrunc`, `TestEscape`, `TestStyleDiffOutput`).
- `tests/tui/test_screens_basic.py`
  - approval/ask-user/exit and smaller modal/screen unit tests.
- `tests/tui/test_app_core_panels.py`
  - theme/status/activity/task panel/process pane/sidebar/chat replay foundational behavior.
- `tests/tui/test_process_runs.py`
  - process run lifecycle tests, startup resume, run start auth resolution.
- `tests/tui/test_process_slash_commands.py`
  - split from `TestProcessSlashCommands`.
- `tests/tui/test_command_palette_process_actions.py`
  - split from `TestCommandPaletteProcessActions`.
- `tests/tui/test_slash_commands_model.py`
  - `TestModelSlashCommands`.
- `tests/tui/test_slash_commands_telemetry.py`
  - `TestTelemetrySlashCommands`.
- `tests/tui/test_slash_commands_mcp.py`
  - `TestMCPSlashCommands`.
- `tests/tui/test_slash_commands_auth.py`
  - `TestAuthSlashCommands`.
- `tests/tui/test_slash_help_and_file_viewer.py`
  - `TestSlashHelp`, `TestFileViewer`, and related help/hints tests.
- `tests/tui/app/`
  - reserve for direct tests of newly modularized `loom.tui.app` internals from the TUI app split plan (prevents overlap with broader behavior suites).

## Naming and Overlap Rules

1. New TUI tests live only under `tests/tui/`.
2. Do not duplicate existing non-TUI suites:
   - keep CLI behavior tests in `tests/test_cli.py`
   - keep API behavior tests in their existing files
3. Each new module owns one domain boundary (no mixed CLI/API/TUI concerns).
4. Keep class names descriptive, but shorten when file scope already conveys context.
5. If `tests/tui/app/` exists, keep architecture/module-boundary tests there and keep user-flow behavior tests in `tests/tui/` root modules.

## Incremental Plan

### Phase 1: Create `tests/tui/` Package and Shared Fixtures
- Add `tests/tui/conftest.py`.
- Move low-risk helper and API-client tests first.

### Phase 2: Split Screen and Core Panel Tests
- Move modal/screen tests and foundational app panel tests.
- Keep imports stable (`loom.tui.app` facade path unchanged).

### Phase 3: Split Process-Run and Slash Test Families
- Move process-run lifecycle tests.
- Split giant slash command classes into focused files.

### Phase 4: Split Command Palette and Help/File Viewer
- Move command-palette process action tests to dedicated module.
- Move slash help + file viewer tests into one UI-assistance module.

### Phase 5: Cleanup and Deterministic Collection
- Remove old monolithic `tests/test_tui.py`.
- Ensure deterministic pytest collection ordering and no duplicate test IDs.
 - Validate `pytest --collect-only tests/tui` has no duplicate nodeids and expected module ownership.

## Risks and Hardening
- Risk: fixture drift when moving tests.
  - Hardening: centralize only truly shared fixtures; keep domain fixtures local to module.
- Risk: duplicated coverage creeping across modules.
  - Hardening: enforce module ownership comments at top of each new file.
- Risk: difficult bisect due huge move-only diffs.
  - Hardening: do move-only commits first, then fixture refactor commits.

## Adherence to Prior Plan Standards
- Revertable slices:
  - Move tests in domain batches (helpers/screens, process runs, slash families, command palette, help/viewer).
- Compatibility during transition:
  - Keep a short-lived `tests/test_tui.py` shim that imports migrated modules until full move completes.
- Non-overlap with existing plans:
  - Do not duplicate import-cycle/import-contract checks already planned under `tests/tui/app/`; reference those modules instead.
- Fixture discipline:
  - Shared fixtures only in `tests/tui/conftest.py`; domain-specific fixtures stay in local modules to avoid global coupling.

## Test Strategy
- Validate per-split:
  - `uv run pytest tests/tui/<new-module>.py`
- Validate full TUI package:
  - `uv run pytest tests/tui`
- Validate integration with existing suites:
  - `uv run pytest tests/test_setup.py tests/test_cli.py`
 - Validate collection and uniqueness:
   - `uv run pytest --collect-only tests/tui`

## Exit Criteria
- `tests/test_tui.py` removed or reduced to a transitional shim.
- All TUI tests live under `tests/tui/` with clear domain boundaries.
- No overlap with non-TUI test packages.
- Pytest run time and failure diagnostics improve versus monolith.
