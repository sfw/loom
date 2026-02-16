# Code Review Refactor Plan (2026-02-16)

## Baseline
- Test run: `PYTHONPATH=src .venv/bin/pytest -q` -> `1208 passed, 89 warnings`.
- Lint run: `.venv/bin/ruff check src tests` -> clean.
- Review focus: latent production defects, integration mismatches, edge-case safety gaps, and collision risks.

## Priority 0 (Fix first)

### 1) Task persistence split-brain (state files vs SQLite tasks table)
- Problem: Task creation writes only YAML state, while `/tasks` list reads only SQLite `tasks` rows.
- Evidence: `src/loom/api/routes.py:65`, `src/loom/api/routes.py:101`, `src/loom/state/memory.py:105`.
- Impact: `/tasks` can return empty/misleading data even when tasks exist; DB task metadata (`approval_mode`, `callback_url`, `metadata`) is effectively unused.
- Refactor plan:
  - Define one source of truth for task metadata (recommended: SQLite) and one for execution snapshot (YAML).
  - On create: insert into DB and create YAML state in one transaction-like flow.
  - On status/plan updates: update both DB task status fields and YAML state.
  - Add recovery reconciliation for orphaned YAML tasks/DB rows on startup.
- Tests to add:
  - API integration test: create task -> list task -> verify status transitions in `/tasks`.
  - Recovery test: stale YAML + missing DB row and stale DB row + missing YAML.

### 2) `loom cancel` CLI command calls non-existent API route
- Problem: CLI sends `POST /tasks/{id}/cancel`, API exposes `DELETE /tasks/{id}`.
- Evidence: `src/loom/__main__.py:357`, `src/loom/api/routes.py:300`.
- Impact: CLI cancel fails at runtime (404/405) despite task existing.
- Refactor plan:
  - Switch CLI to `DELETE /tasks/{id}`.
  - Optionally add compatibility route alias if backward compatibility is needed.
- Tests to add:
  - CLI integration test with mocked HTTP client verifying `DELETE` is used.

### 3) `--process` is silently ignored in `loom run` API flow
- Problem: CLI sends `process` in payload, but API request schema has no process field and route ignores it.
- Evidence: `src/loom/__main__.py:233`, `src/loom/__main__.py:277`, `src/loom/api/schemas.py:10`, `src/loom/api/routes.py:57`.
- Impact: Users think a process is applied when it is not (silent behavior mismatch).
- Refactor plan:
  - Add `process` to `TaskCreateRequest`.
  - Resolve/load process during task creation and inject into orchestrator execution path.
  - Validate unknown process names with a 4xx response.
- Tests to add:
  - API test asserting process selection affects prompt/process behavior.
  - Negative test for invalid process name.

### 4) Process listing executes bundled tool code (collision and safety risk)
- Problem: `list_available()` calls `_load_from_path()`, which imports bundled tools as a side effect.
- Evidence: `src/loom/processes/schema.py:252`, `src/loom/processes/schema.py:257`, `src/loom/processes/schema.py:289`, `src/loom/processes/schema.py:587`, `src/loom/tools/registry.py:109`.
- Impact:
  - Listing metadata can execute arbitrary package code.
  - Loaded tool classes persist globally via `Tool._registered_classes`, causing cross-process tool collisions.
- Refactor plan:
  - Split metadata parsing from runtime tool registration.
  - Ensure `list_available()` never imports tool modules.
  - Register bundled tools only when an explicit process is selected/executed.
  - Add namespacing/conflict policy for bundled tool names.
- Tests to add:
  - Listing processes does not import any `tools/*.py` modules.
  - Conflict test for bundled tool name matching built-in tool name.

## Priority 1

### 5) Learning manager is instantiated but never wired into orchestrator
- Problem: `LearningManager` is created but not passed to `Orchestrator`.
- Evidence: `src/loom/api/engine.py:114`, `src/loom/api/engine.py:117`.
- Impact: Post-task learning extraction silently never runs in API mode.
- Refactor plan:
  - Pass `learning_manager=learning_manager` into orchestrator construction.
  - Add logging/metrics to confirm learning runs and failures.
- Tests to add:
  - Integration test asserting `learn_from_task()` is invoked after completion/failure.

### 6) Task state serialization drops key fields
- Problem: YAML state round-trip excludes `approval_mode`, `callback_url`, `context`, `metadata`, and timestamps.
- Evidence: `src/loom/state/task_state.py:235`, `src/loom/state/task_state.py:299`.
- Impact: Reloaded tasks can lose intended approval behavior and metadata fidelity.
- Refactor plan:
  - Persist full task metadata in DB (preferred) and load response-facing fields from DB.
  - If YAML remains authoritative for any field, include explicit serialization for required fields.
- Tests to add:
  - Round-trip tests for all externally visible task fields.

### 7) Relative-path handling bug in `ripgrep_search` and `glob_find`
- Problem: Provided `path` is resolved relative to process CWD, not workspace.
- Evidence: `src/loom/tools/ripgrep.py:79`, `src/loom/tools/ripgrep.py:83`, `src/loom/tools/glob_find.py:60`, `src/loom/tools/glob_find.py:64`.
- Impact: Valid workspace-relative paths can be rejected as outside workspace or search wrong location.
- Refactor plan:
  - Use `_resolve_path()` for non-absolute paths when workspace exists.
  - Keep current explicit rejection for paths escaping workspace.
- Tests to add:
  - Relative path case (`path="src"`) with workspace set for both tools.

### 8) Web fetch SSRF guard is hostname-pattern only
- Problem: Safety check does not resolve DNS/IP literals comprehensively.
- Evidence: `src/loom/tools/web.py:16`, `src/loom/tools/web.py:26`, `src/loom/tools/web.py:39`.
- Impact: Hostnames resolving to private/internal addresses can bypass current blocklist.
- Refactor plan:
  - Resolve hostnames and block private, loopback, link-local, multicast, and reserved ranges via `ipaddress` checks.
  - Validate each redirect target after resolution too.
- Tests to add:
  - DNS resolution and encoded-IP edge cases mapped to blocked private targets.

### 9) Timeout cancellation can leave shell subprocesses running
- Problem: Registry timeout cancels coroutine without guaranteed subprocess teardown.
- Evidence: `src/loom/tools/registry.py:212`, `src/loom/tools/shell.py:103`.
- Impact: Long-running commands can leak orphan processes after timeout.
- Refactor plan:
  - Add explicit cancellation handling in shell/git/ripgrep subprocess tools (`try/finally`, `kill`, `wait`).
  - Consider moving timeout ownership into tool implementation where process handles are known.
- Tests to add:
  - Timeout test that asserts child process is terminated.

### 10) CORS origin config uses unsupported wildcard port format
- Problem: `allow_origins=["http://localhost:*", ...]` uses literal strings, not regex.
- Evidence: `src/loom/api/server.py:50`.
- Impact: Browser clients on `localhost:<port>` may fail CORS unexpectedly.
- Refactor plan:
  - Use `allow_origin_regex` for localhost ports or explicit known origins.
- Tests to add:
  - CORS preflight tests for representative localhost ports.

### 11) Event bus uses deprecated coroutine detection API
- Problem: `asyncio.iscoroutinefunction` is deprecated for Python 3.16.
- Evidence: `src/loom/events/bus.py:89`.
- Impact: Future compatibility break; currently emits deprecation warnings in tests.
- Refactor plan:
  - Replace with `inspect.iscoroutinefunction` and support async callable objects.
- Tests to add:
  - Event handler tests for async function and async `__call__` object handlers.

## Priority 2

### 12) Dead/undocumented API client endpoint reference
- Problem: `LoomAPIClient.stream_all_events()` calls `/events/stream`, but no matching route exists.
- Evidence: `src/loom/tui/api_client.py:152`.
- Impact: Method is broken if called; API surface is inconsistent.
- Refactor plan:
  - Either implement `/events/stream` route or remove/rename method.
- Tests to add:
  - Contract test matching client methods to existing server routes.

### 13) `execution.enable_streaming` is defined but not loaded from config
- Problem: Config dataclass has field, loader never maps TOML value.
- Evidence: `src/loom/config.py:112`, `src/loom/config.py:243`.
- Impact: Users cannot enable streaming via config file.
- Refactor plan:
  - Parse and pass through `enable_streaming` in `load_config`.
- Tests to add:
  - Config parse test for explicit `enable_streaming=true`.

### 14) TUI persistence path diverges from configured `memory.database_path`
- Problem: TUI persistence initializes DB at `workspace.scratch_dir/loom.db` instead of `memory.database_path`.
- Evidence: `src/loom/__main__.py:146`, `src/loom/__main__.py:150`.
- Impact: TUI and API may write to different databases by default; operational data appears inconsistent across interfaces.
- Refactor plan:
  - Decide canonical DB path policy and align TUI/API.
  - If split DBs are intentional, make it explicit in config and docs.
- Tests to add:
  - Startup test verifying TUI uses configured memory DB path.

## Execution order
1. Unify task persistence + fix CLI/API contract mismatches (`/tasks`, cancel, process field).
2. Remove process-list side effects and tool collision risk.
3. Wire learning and restore field fidelity for task metadata.
4. Fix path resolution + subprocess timeout cleanup + SSRF hardening.
5. Clean compatibility issues (CORS, deprecations, dead endpoints, config gaps).

## Coverage gaps found during review
- No test validates `/tasks` list after `POST /tasks`.
- No CLI behavior test covers `cancel` endpoint verb/path.
- No tests for workspace-relative `path` in `ripgrep_search` or `glob_find`.
- No test that `--process` actually affects API task execution.
