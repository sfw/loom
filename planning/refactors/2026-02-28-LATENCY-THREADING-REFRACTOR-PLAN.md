# Latency and Threading Refactor Plan (2026-02-28)

## Objective
Reduce user-visible latency in TUI interaction and startup by moving blocking discovery/IO work off the main event loop and UI hot paths, while preserving correctness, auth isolation, and safety guarantees.

## Reported Symptoms
1. Keypresses in TUI approval modals feel delayed by multiple seconds.
2. Startup/system load shows 3-5+ second delays.
3. MCP refresh/discovery behavior contributes to blocking during interactive flows.

## Baseline (Current Measurements and Findings)
1. Process discovery call cost in this workspace:
   - `ProcessLoader.list_available()` average: about 80.94 ms per call.
   - This is currently called from slash-hint/input-completion paths on typing.
2. Setup model discovery path is synchronous HTTP in UI path:
   - Unreachable endpoint test produced about 4007.88 ms delay with discovery timeout=2 and endpoint fallback behavior.
3. Startup registry creation still performs synchronous MCP discovery:
   - `create_default_registry(...)` calls `register_mcp_tools(...)` -> `refresh(force=True)` synchronously.
4. Auth-scoped MCP discovery can still run synchronously in async execution paths via `list_tools`/`all_schemas`.
5. Verification/planning and multiple async tools still run blocking filesystem scans in async context.
6. TUI startup can build tool registry more than once when an active process is loaded (`_initialize_session` + `_load_process_definition` path).
7. File viewer preview currently performs synchronous file reads/parsing in modal construction path.

## Scope
1. TUI input/hint path latency.
2. TUI setup screen discovery responsiveness.
3. MCP startup and auth-scoped refresh/discovery execution model.
4. Verification/planning scan offloading.
5. Blocking async tool internals.
6. API request-path blocking setup calls.
7. Startup registry rebuild deduplication and tool discovery caching.
8. TUI file preview loading responsiveness.
9. Threadpool governance and cancellation semantics for background IO.

## Non-Goals
1. Full architectural rewrite of ToolRegistry or orchestration.
2. Distributed execution/multi-process scheduler changes.
3. Broad behavior changes to approval policy, auth semantics, or tool contracts.

## Design Principles
1. Keep UI/event-loop paths non-blocking.
2. Use stale-while-refresh where possible for discovery catalogs.
3. Preserve strict sync fallback for correctness-critical "execute missing tool" cases.
4. Keep all UI mutation on the Textual thread; background work only computes/fetches.
5. Bound concurrency and queue growth to avoid thread storms.
6. Add telemetry before and during rollout to prove p50/p95 improvement.
7. Centralize blocking-IO offload behind bounded executors instead of ad hoc thread creation.

## Workstream 0: Event-Loop Stall Observability

### Problem
Current timing instrumentation is useful but does not directly capture event-loop stalls and lock contention windows causing perceived keypress lag.

### Plan
1. Add lightweight event-loop lag telemetry (for example, periodic scheduler drift measurement in TUI and API runtime).
2. Add per-path timing envelopes around:
   - approval modal open-to-key-ack latency
   - slash-hint render
   - process index refresh
   - MCP discovery/refresh (global and auth-scoped)
3. Emit lock wait-time diagnostics for ToolRegistry/MCP synchronizer hot locks where feasible.
4. Keep instrumentation low-overhead and gated for debug/diagnostic modes.

### Primary Files
1. `src/loom/tui/app.py`
2. `src/loom/tools/registry.py`
3. `src/loom/integrations/mcp_tools.py`
4. `src/loom/api/engine.py` (if runtime metrics hook is shared)

### Tests
1. Unit tests for telemetry payload fields and guard behavior.
2. No-regression checks ensuring instrumentation does not alter control flow.

### Acceptance
1. We can attribute latency spikes to specific paths (UI, MCP, scan, tool IO) before and after each phase.

## Workstream 1: Remove Keystroke-Path Process Discovery Blocking

### Problem
`_refresh_process_command_index()` is called in input-change and completion/hint paths, and currently performs synchronous `loader.list_available()` each time.

### Plan
1. Add in-memory TTL cache for process catalog and command map.
2. Make slash hint/completion paths read cache only.
3. Schedule catalog refresh via background worker/debounced task.
4. Trigger immediate refresh only on explicit commands that can tolerate it (`/process list`, `/process use` entry points), but execute via worker where feasible.
5. Add "refresh in progress" guard to avoid duplicate concurrent scans.

### Primary Files
1. `src/loom/tui/app.py`
2. `src/loom/processes/schema.py` (optional lightweight metadata caching helper)

### Tests
1. Slash-hint/input-changed tests proving no synchronous `list_available()` call on every keystroke.
2. Process command map update tests for stale-to-fresh transition.
3. Conflict-notification behavior tests remain intact.

### Acceptance
1. Typing `/` and `/process ...` remains responsive under slow filesystems.
2. Process command map converges to latest state without blocking keystrokes.

## Workstream 2: Async Setup Model Discovery UX

### Problem
Setup screen `_discover_models()` calls synchronous `discover_models(...)` directly in UI flow.

### Plan
1. Move discovery calls to `run_worker` + `asyncio.to_thread` or native async discovery helper.
2. Add cancellation or "last request wins" semantics for rapid field edits.
3. Add explicit "discovering..." UI state and disable duplicate submit while in-flight.
4. Keep manual model entry available immediately without waiting on discovery.
5. Revisit endpoint probing behavior (`/models` then `/v1/models`) so timeout compounding is bounded and explicit.

### Primary Files
1. `src/loom/tui/screens/setup.py`
2. `src/loom/setup.py` (introduce async variant or thread-safe wrapper)

### Tests
1. Setup screen interaction tests for in-flight discovery state.
2. Regression tests for Enter-key path and auto-select behavior.
3. No-blocking smoke test with delayed discovery stub.

### Acceptance
1. No visible UI freeze while discovery request is in progress.
2. Final selected model behavior remains unchanged for successful discovery.

## Workstream 3: MCP Discovery/Refresh Execution Model Hardening

### Problem
Some MCP refresh/discovery flows remain synchronous on startup and auth-scoped schema requests.

### Plan
1. Add startup mode that registers MCP synchronizer and returns immediately with empty/stale MCP tool set, then warms in background.
2. Keep force-sync refresh only when an explicitly requested MCP tool is missing at execute time.
3. For auth-scoped `list_tools`/`all_schemas`, return cached auth view immediately and trigger refresh in background when expired.
4. Bound MCP refresh threads and add single-flight guards per registry/auth fingerprint.
5. Add observability around refresh duration, cache hit ratio, and forced-sync fallbacks.
6. Remove unsynchronized direct access patterns to registry internals in synchronizer paths and enforce lock-order discipline.
7. Add cache hygiene policy for auth-view cache (TTL + bounded size) to avoid unbounded growth from many auth fingerprints.

### Primary Files
1. `src/loom/tools/__init__.py`
2. `src/loom/tools/registry.py`
3. `src/loom/integrations/mcp_tools.py`
4. `src/loom/tui/app.py` (startup summary state messaging if needed)

### Tests
1. Startup test proving no synchronous MCP discovery block in default path.
2. Auth-scoped isolation tests: no tool leakage across contexts.
3. Force-refresh correctness test for missing MCP tool execution.
4. Background refresh thread tests with lock/race coverage.

### Acceptance
1. TUI startup no longer waits on MCP tool-list roundtrips by default.
2. Correctness preserved for explicit MCP tool execution.
3. Auth-scoped isolation guarantees remain intact.

## Workstream 4: Verification and Planner Scan Offloading

### Problem
Verification and workspace analysis perform blocking filesystem traversal/reads in async workflows.

### Plan
1. Move heavy scan methods to `to_thread` boundaries.
2. Keep deterministic caps and scan-order logic unchanged.
3. Add timeout/cancellation-safe wrappers to prevent runaway scan latency.
4. Instrument scan duration and cap-exhaustion telemetry.

### Primary Files
1. `src/loom/engine/verification.py`
2. `src/loom/engine/orchestrator.py`

### Tests
1. Verification guard behavior parity tests (same pass/fail semantics).
2. Cancellation and timeout tests for scan wrapper.
3. Performance smoke tests in larger fixture workspace.

### Acceptance
1. No behavior regressions in contradiction guard outcomes.
2. Reduced event-loop blocking during verification/planning phases.

## Workstream 5: Blocking Async Tool Internals

### Problem
Several `async def execute(...)` tools do blocking filesystem/parse work directly.

### Plan
1. Audit and annotate tools into categories:
   - Safe synchronous (tiny work).
   - Candidate for `to_thread` wrapper.
   - Already non-blocking.
2. Prioritize highest-cost tools first:
   - `search_files`, `glob_find`, `read_file` heavy formats, `analyze_code`.
3. Add shared helper for bounded tool threadpool dispatch to avoid unbounded `to_thread` fanout.
4. Preserve existing timeout semantics and error messages.
5. Explicitly document cancellation semantics for thread-offloaded work and ensure timeout failures cannot leave unbounded orphan work.

### Primary Files
1. `src/loom/tools/search.py`
2. `src/loom/tools/glob_find.py`
3. `src/loom/tools/file_ops.py`
4. `src/loom/tools/code_analysis.py`
5. `src/loom/tools/registry.py` (shared dispatch helper if centralized)

### Tests
1. Behavior parity tests for tool output and error format.
2. Timeout/cancellation tests around thread-wrapped execution.
3. Concurrency tests for bounded pool behavior.

### Acceptance
1. Tool correctness unchanged.
2. Reduced event-loop stalls during expensive local IO tools.

## Workstream 6: Startup Registry Rebuild Deduplication

### Problem
Startup/session initialization can do redundant registry construction and repeated tool-module scanning.

### Plan
1. Remove duplicate registry rebuild paths during `_initialize_session` + process load flows.
2. Add safe caching for discovered built-in tool classes (`discover_tools`) after first import scan.
3. Preserve explicit rebuild path only when process-bundled tools actually change registration set.
4. Keep MCP startup behavior consistent with Workstream 3.

### Primary Files
1. `src/loom/tui/app.py`
2. `src/loom/tools/__init__.py`

### Tests
1. Startup/session tests proving correct tool inventory when process is present/absent.
2. Regression test for process-bundled tool registration updates.

### Acceptance
1. Startup does not perform unnecessary registry rebuilds.
2. Tool inventory correctness remains unchanged.

## Workstream 7: File Viewer Preview Offload

### Problem
File preview loading currently happens synchronously in modal construction and can block UI on large/expensive formats.

### Plan
1. Defer preview load to background worker (`run_worker`/`to_thread`) with loading state.
2. Keep rendering/UI updates on main thread only.
3. Add clear timeout/error states for slow or failed preview extraction.
4. Preserve preview truncation and format behavior.

### Primary Files
1. `src/loom/tui/screens/file_viewer.py`
2. `src/loom/content_utils.py` (if helper boundaries are needed)

### Tests
1. File viewer screen tests for loading state, success state, and failure state.
2. Regression tests for preview truncation and supported file types.

### Acceptance
1. Opening file preview remains responsive under heavy files.
2. Existing preview behavior/output parity remains intact.

## Workstream 8: API Request-Path Blocking Calls

### Problem
`POST /tasks` path performs synchronous process load/registry creation inside async route.

### Plan
1. Move heavy sync calls behind `asyncio.to_thread` in route handlers.
2. Consider cache reuse for required-tool preflight within request scope.
3. Keep same error payloads and status codes.

### Primary Files
1. `src/loom/api/routes.py`
2. `src/loom/api/engine.py` (if helper extraction needed)

### Tests
1. API regression tests for preflight error semantics.
2. Concurrency test with multiple task-create requests.

### Acceptance
1. Async route responsiveness improves under concurrent request load.
2. No API contract changes.

## Rollout Strategy

### Phase 0: Instrumentation and Guardrails
1. Deliver Workstream 0.
2. Add timing metrics for:
   - slash hint rendering path
   - process index refresh
   - setup discovery latency
   - MCP refresh/discovery (sync vs background)
   - verification scan duration
3. Add feature flags for each workstream where risk is medium/high.

### Phase 1: TUI Hot Path Fixes
1. Workstream 1.
2. Workstream 2.
3. Workstream 7.

### Phase 2: MCP Refresh Model
1. Workstream 3.
2. Workstream 6.

### Phase 3: Engine and Tool Offloading
1. Workstream 4.
2. Workstream 5.

### Phase 4: API Request Path
1. Workstream 8.

## PR Plan
1. PR1: Event-loop stall instrumentation and latency diagnostics (Workstream 0).
2. PR2: TUI process-index cache + debounced background refresh (Workstream 1).
3. PR3: Setup screen async model discovery + timeout compounding controls (Workstream 2).
4. PR4: MCP startup/background refresh, auth-view stale-while-refresh, and cache/lock hardening (Workstream 3).
5. PR5: Startup registry dedupe + tool discovery caching (Workstream 6).
6. PR6: Verification/planner scan offload (Workstream 4).
7. PR7: Async tool blocking-IO offload with bounded executor policy (Workstream 5).
8. PR8: File viewer preview offload (Workstream 7).
9. PR9: API preflight offload and request-path cleanup (Workstream 8).

## Validation Matrix
1. Manual TUI latency checks:
   - keypress echo in approval screen
   - slash hint responsiveness under rapid typing
   - startup ready-time with MCP configured
2. Automated suites:
   - `uv run pytest tests/test_tui.py tests/test_tools.py tests/test_mcp.py tests/test_cowork_approval.py`
   - `uv run pytest tests/test_api.py tests/test_orchestrator.py tests/test_verification.py`
3. Static checks:
   - `uv run ruff check src tests`
4. Optional local benchmark script in `scripts/` for before/after timing snapshots.

## Risks and Mitigations
1. Risk: Thread safety regressions in registry/tool maps.
   - Mitigation: explicit lock scope, single-flight refresh, race-focused tests.
2. Risk: Stale catalogs confuse tool availability.
   - Mitigation: force-sync fallback on execute-missing path and visible refresh indicators where useful.
3. Risk: Too many background tasks/threads.
   - Mitigation: bounded pool, dedupe gates, per-feature in-flight flags, and centralized executor ownership.
4. Risk: Behavior drift from async wrappers.
   - Mitigation: output parity tests and golden response fixtures for key tools.
5. Risk: Cancellation mismatch for thread-offloaded operations.
   - Mitigation: explicit timeout failure semantics, bounded queueing, and cooperative cancellation notes in tool contracts.

## Definition of Done
1. Reported UI keypress and startup lag is materially reduced in real workflows.
2. MCP correctness and auth isolation behavior remain intact under concurrency.
3. Verification/planning outputs remain semantically identical on existing regression tests.
4. API and TUI regression suites pass.
5. Timing metrics show improvement in p50/p95 for targeted paths.
