# /run Launch Liveness Hardening Plan (2026-02-28)

## Objective
Eliminate the perceived TUI freeze during `/run`, guarantee visible progress from the first 100ms, and keep workspace/file views coherent as run workspaces and artifacts are created.

## Reported UX Failures
1. `/run` appears to freeze with no status messaging until a run tab appears.
2. New run workspace folders are created but workspace tree refresh is not reliably visible at the moment of creation.
3. Long planning/orchestration windows show little or no signal, and the process tab progress area can look blank.
4. Users cannot distinguish "working normally but slow" from "hung."

## Baseline (Repo-Accurate)
1. `on_user_submit()` awaits `_handle_slash_command()` directly for slash commands in `src/loom/tui/app.py`.
2. `/run` path does expensive work before tab creation:
3. Ad hoc synthesis can run in `_get_or_create_adhoc_process()` before `_start_process_run()`.
4. `_start_process_run()` awaits `_prepare_process_run_workspace()` and `_resolve_auth_overrides_for_run_start()` before creating the run tab.
5. `_prepare_process_run_workspace()` may call `_llm_process_run_folder_name()` (model roundtrip) before mkdir.
6. Process tab progress rows are driven by `run.tasks`; these are often empty until `task_plan_ready`.
7. `delegate_task` emits an initial `_emit_progress()` snapshot, but this can still yield empty task rows while planning.
8. Workspace refresh is coalesced via `_request_workspace_refresh()`, but run-workspace creation itself is not treated as a first-class immediate refresh event.
9. `delegate_task` result `files_changed` currently summarizes counts (e.g. `"(N files created)"`), so Files panel path-level fidelity is weak for delegated runs.

## UX/SLO Targets
1. `/run` acknowledgment visible in chat within 100ms p95.
2. Run tab visible within 250ms p95 from command submit.
3. Run tab never shows a blank progress surface for >1s while status is `queued|running`.
4. Newly created run workspace folder visible in workspace tree within 1s p95.
5. If no meaningful progress events arrive for 6s, animate the active launch/running stage line in place (dot suffix) without appending duplicate rows.
6. Terminal state always includes explicit outcome and next action (`Resume`, `View log`, or `Close`).

## Non-Goals
1. Replacing delegate/orchestrator planning semantics.
2. Rewriting process package synthesis logic.
3. Building a new backend service for run orchestration.

## Design Decisions

### 1) Split `/run` into Immediate Shell + Async Preflight
1. Create a run shell tab immediately after command validation (goal parsed, command recognized), before synthesis/workspace/auth preflight.
2. Show deterministic stage states in tab header + activity log:
3. `Accepted -> Resolving process -> Provisioning workspace -> Auth preflight -> Queueing delegate -> Running -> Completed|Failed|Cancelled`.
4. Move expensive pre-run work off the slash-command hot path into a per-run worker.

### 2) Add a First-Class Launch State Machine
1. Extend `ProcessRunState` with `launch_stage`, `launch_started_at`, `launch_last_heartbeat_at`, `launch_error`.
2. Make launch-stage transitions explicit, idempotent, and journaled for resume.
3. Enforce no silent transition gaps: each stage transition must write a visible activity line.

### 3) Make Progress Surface Non-Blank by Design
1. Keep existing subtask checklist (`run.tasks`) but add stage rows independent of subtasks.
2. While no subtasks exist, progress panel shows stage checklist entries (planning/orchestrating/spawning).
3. On `task_plan_ready`, switch to plan/subtask checklist while preserving stage summary at top.
4. Add heartbeat row update if event stream is quiet for configured interval.

### 4) Treat Run Workspace Creation as Immediate UI Event
1. As soon as run workspace directory is created, request immediate refresh (`_request_workspace_refresh(..., immediate=True)`).
2. Add explicit activity line with absolute workspace path.
3. Ingest run-workspace folder creation into Files panel as a synthetic `create` entry for visibility.

### 5) Improve Delegated File Change Fidelity
1. Preserve backward compatibility for existing `files_changed` summaries.
2. Add additive path-level payload fields where available (e.g. `files_changed_paths`, capped and sanitized workspace-relative).
3. Use path-level data for Files panel first; fallback to summary markers when no concrete paths exist.

### 6) Harden Failure/Timeout Messaging
1. If launch fails before delegate starts (synthesis/auth/workspace), run tab must transition to `failed` with stage-specific error.
2. Add launch timeout guard (configurable) with explicit recovery hints.
3. Keep run tab open on failure to preserve diagnostics and resume actions.

## Implementation Workstreams

### Workstream 0: Observability and Guardrails
1. Add launch timing spans and counters:
2. `run_submit_to_tab_ms`, `run_tab_to_delegate_start_ms`, `run_stage_duration_ms`, `run_silent_window_ms`.
3. Add structured logs on stage transitions and refresh triggers.
4. Add one-shot warning for silent windows exceeding threshold.

Primary files:
1. `src/loom/tui/app.py`
2. `src/loom/telemetry.py` (if shared helper needed)

### Workstream 1: Immediate Run Shell
1. Refactor `/run` handling to allocate run ID + tab before expensive preflight.
2. Introduce `_start_process_run_shell(...)` and `_prepare_and_execute_process_run(run_id, request)`.
3. Ensure existing `/run close`, `/run resume`, `/run save` behavior remains unchanged.

Primary files:
1. `src/loom/tui/app.py`

### Workstream 2: Async Preflight Pipeline
1. Move ad hoc synthesis resolution into the run worker path with stage updates.
2. Move workspace provisioning + auth preflight into staged worker path.
3. Keep cancellations safe if run closed during preflight.
4. Preserve auth manager prompting flow while keeping tab responsive.

Primary files:
1. `src/loom/tui/app.py`

### Workstream 3: Non-Blank Progress Model
1. Extend `ProcessRunPane` to render launch-stage checklist while tasks are empty.
2. Map delegate event types to stage/substage labels.
3. Add heartbeat ticker updates to activity/progress surfaces.
4. Keep existing dedupe logic for repeated messages.

Primary files:
1. `src/loom/tui/app.py`
2. `tests/test_tui.py`

### Workstream 4: Workspace/File Coherency
1. Trigger immediate refresh when run workspace is created.
2. Add synthetic Files panel entry for run workspace folder creation.
3. Upgrade delegated file change payload handling to prefer concrete relative paths.
4. Maintain bounded dedupe and row caps.

Primary files:
1. `src/loom/tui/app.py`
2. `src/loom/tools/delegate_task.py`
3. `src/loom/engine/runner.py`
4. `tests/test_tui.py`
5. `tests/test_delegate_task.py` (or nearest delegate tool test file)
6. `tests/test_runner.py` (or nearest runner event payload tests)

### Workstream 5: Config, Docs, and Changelog
1. Add guarded tunables under `[tui]`:
2. `run_launch_heartbeat_interval_ms` (default 6000)
3. `run_launch_timeout_seconds` (default 300)
4. `run_preflight_async_enabled` (default true)
5. Document behavior and fallback semantics.

Primary files:
1. `src/loom/config.py`
2. `loom.toml.example`
3. `docs/CONFIG.md`
4. `CHANGELOG.md`

## Test Strategy

### Unit Tests
1. Run shell tab is created before preflight begins.
2. Stage transitions are monotonic and terminal-safe.
3. Progress panel renders stage rows when `tasks=[]`.
4. Heartbeat appears after silent interval and stops on terminal state.
5. Workspace refresh is requested immediately on run workspace creation.
6. Files panel receives synthetic create row for run workspace folder.

### Integration/TUI Tests
1. `/run` with ad hoc synthesis: tab appears immediately, then stage updates stream.
2. `/run` with auth ambiguity: tab remains live while prompt flow runs; no freeze.
3. Delegate planning delay simulation: progress surface is never blank.
4. External workspace tree reflects run folder without manual reload.

### Regression Coverage
1. Existing `/run resume`, `/run close`, `/run save` flows.
2. Existing cowork turn path and delegate progress handling.
3. Process command discovery, slash hints, and command palette behavior.

## Rollout Plan
1. Ship behind `tui.run_preflight_async_enabled` default `true`, with one-release rollback toggle.
2. Enable detailed launch telemetry in debug logs first.
3. Validate on large workspaces and slow model endpoints before removing any old path.
4. Remove dead pre-refactor code paths only after one stable release cycle.

## Risks and Mitigations
1. Risk: race conditions if run is closed during preflight.
2. Mitigation: run ID ownership checks before each stage mutation.
3. Risk: duplicate status noise from heartbeat + real events.
4. Mitigation: suppress heartbeat for N seconds after any real progress event.
5. Risk: auth prompt flow complexity in async launch.
6. Mitigation: isolate prompt flow into stage-specific helper and enforce single in-flight prompt per run.
7. Risk: increased UI churn from immediate refresh calls.
8. Mitigation: retain refresh coalescer and use immediate refresh only for run-workspace creation.

## Acceptance Criteria
1. Users always see a run tab almost immediately after `/run`.
2. No perceived freeze windows with empty/blank progress surfaces.
3. Run workspace folder creation is visible without manual reload.
4. Failure cases are explicit and actionable, not silent.
5. Existing process-run and cowork regressions remain zero in test suite.
