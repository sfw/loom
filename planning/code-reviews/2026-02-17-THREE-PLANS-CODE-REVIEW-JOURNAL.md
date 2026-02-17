# 2026-02-17 Code Review Journal (Code-Review Plan + TUI Review + Tree-sitter)

## Scope
Reviewed implementation changes from:
- `4d488ad` (`2026-02-16-CODE-REVIEW-PLAN.md` execution)
- `9e6a8cf` (`2026-02-16-TUI-REVIEW.md` execution)
- `0e727be` + `b80f32a` (tree-sitter plan execution and follow-up fixes)

Also performed a full-system readiness pass.

## Verification Run
- `PYTHONPATH=src .venv/bin/pytest -q` -> **1234 passed, 44 skipped**
- `.venv/bin/ruff check src tests` -> **All checks passed**
- TUI-focused regression run: `PYTHONPATH=src .venv/bin/pytest -q tests/test_setup.py tests/test_tui.py` -> **83 passed**

## Findings (ordered by severity)

### P0 - API process state leaks across tasks (cross-task collision)
- **Files:** `src/loom/api/routes.py:113`, `src/loom/api/routes.py:114`, `src/loom/api/routes.py:115`
- **What is wrong:** `_execute_in_background()` mutates shared orchestrator state (`engine.orchestrator._process` and `engine.orchestrator._prompts.process`) before running a task.
- **Why this is a bug:** The orchestrator instance is shared for all API tasks. This creates process collisions between concurrent tasks and also leaks process state into later tasks (including tasks with no process).
- **Repro evidence:** calling `_execute_in_background(..., process_def='PROC_A')` followed by `_execute_in_background(..., process_def=None)` leaves `_process == 'PROC_A'`.
- **Impact:** Wrong process persona/rules can be applied to the wrong task; behavior becomes nondeterministic under concurrency.
- **Recommended fix:** Remove shared mutation; pass process context per task (new orchestrator instance per task, or make `execute_task(..., process=...)` fully task-local and immutable).

### P1 - API process behavior is only partially wired (verification/exclusions mismatch)
- **Files:** `src/loom/api/routes.py:113`, `src/loom/api/routes.py:114`, `src/loom/api/routes.py:115`, `src/loom/engine/orchestrator.py:114`, `src/loom/engine/orchestrator.py:118`, `src/loom/engine/orchestrator.py:132`
- **What is wrong:** API path only sets `_process` and prompt process. But process-dependent verification and tool exclusions are initialized in `Orchestrator.__init__`.
- **Why this is a bug:** A task started via API with `process` can miss process regex checks and tool exclusions, so API behavior can diverge from TUI behavior.
- **Impact:** Policy bypass risk and inconsistent outputs for the same process definition.
- **Recommended fix:** Build process-aware execution context once (constructor-level), not via private field mutation mid-flight.

### P1 - API process discovery ignores configured `process.search_paths`
- **File:** `src/loom/api/routes.py:62`
- **What is wrong:** `ProcessLoader` in API create-task route is built without `extra_search_paths` from config.
- **Why this is a bug:** `loom.toml` process search paths are honored in TUI/CLI paths but not in this API route.
- **Impact:** Processes that resolve in TUI/CLI can fail in API with "not found".
- **Recommended fix:** Pass `extra_search_paths=[Path(p) for p in engine.config.process.search_paths]` when constructing `ProcessLoader`.

### P1 - Files Changed panel is never reset on session switch/new
- **Files:** `src/loom/tui/widgets/file_panel.py:54`, `src/loom/tui/app.py:401`, `src/loom/tui/app.py:433`
- **What is wrong:** `FilesChangedPanel.clear_files()` exists, but neither `_new_session()` nor `_switch_to_session()` calls it.
- **Why this is a bug:** File history now accumulates forever across sessions after P1-6.
- **Impact:** Misleading demo UX; users see stale changes from prior sessions.
- **Recommended fix:** Clear `#files-panel` (and diff viewer) on new/switch session.

### P2 - Tree-sitter path is currently unvalidated in default test environment
- **File:** `tests/test_treesitter.py:20`
- **What is wrong:** Entire tree-sitter suite is skipped when optional dependency is absent (currently `44 skipped`).
- **Why this matters:** This is a readiness risk for a demo that depends on tree-sitter features.
- **Impact:** Regressions in structural matching/extraction can ship undetected.
- **Recommended fix:** Add a CI/test target with `.[dev,treesitter]` and require `tests/test_treesitter.py` there.

## Demo Readiness Verdict
**Not demo-ready yet** due to one blocker:
- P0 cross-task process collision in API execution path.

### Ready after these are addressed
1. Fix P0 process state isolation in API execution.
2. Align API process wiring with constructor-level process behavior (verification + excluded tools).
3. Honor configured process search paths in API route.
4. Reset Files panel on session changes.
5. Run tree-sitter tests in an environment with the optional dependency installed.
