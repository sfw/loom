# TUI Concurrent Process Runs Plan (2026-02-17)

## Summary
The current TUI process execution path (`/process use` + `/run`) is functionally correct for single runs but structurally serialized. It uses a shared `_busy` gate and routes progress into shared chat/sidebar state, which makes the interface feel blocked and prevents multi-run workflows.

This plan introduces per-run tabs with isolated run state and background workers, while keeping cowork chat interactive.

## User-Visible Goals
1. Keep cowork chat usable while process runs are active.
2. Support multiple concurrent process runs.
3. Create one tab per process run.
4. Show clear status in each tab title:
   - Running
   - Completed
   - Failed
   - Cancelled (if implemented)
5. Show elapsed run time while active (and total duration when complete).
6. Preserve current `/process use` and `/run` ergonomics.

## Current-State Review

### 1) Global UI lock couples chat and process runs
- `src/loom/tui/app.py` uses a single `_busy` flag for both `@work _run_turn` and `_run_process_goal`.
- `on_user_submit` drops input when `_busy` is true, so `/run` blocks chat and chat blocks `/run`.

### 2) `/run` executes inline, not as independent run worker
- `/run` awaits `_run_process_goal` inside slash handling.
- `_run_process_goal` awaits `self._tools.execute("delegate_task", ...)` and holds app-wide busy state for the full run duration.

### 3) Progress routing is global rather than run-scoped
- `_on_process_progress_event` writes to shared chat/events/sidebar with a single de-dupe state.
- Multiple runs would interleave messages with no run identity.

### 4) Delegate tool orchestration instance is cached
- `src/loom/tools/delegate_task.py` stores `self._orchestrator` and reuses it.
- This creates avoidable coupling for parallel runs and complicates event isolation.

### 5) TUI tabs are static today
- `TabbedContent` currently has fixed panes: Chat / Files / Events.
- There is no dynamic run-pane lifecycle, title status mutation, or close/cancel affordance.

## Architecture Proposal

### A) Split busy state by domain
Replace single `_busy` with explicit lanes:
- `self._chat_busy: bool` for cowork turns.
- `self._process_runs: dict[str, ProcessRunState]` for run tasks.

Result:
- Chat remains interactive during process runs.
- `/run` submission is only blocked by configurable per-run limits, not chat activity.

### B) Introduce run state model
Add a dedicated state object (dataclass) in `src/loom/tui/app.py` or `src/loom/tui/process_runs.py`:
- `run_id`
- `process_name`
- `goal`
- `status` (`queued|running|completed|failed|cancelled`)
- `started_at`, `ended_at`
- `tab_id`
- `pane_id`
- `worker_name`
- `task_id` (from delegate payload when available)
- `progress_rows`
- `last_message` / token counters

### C) Run-per-tab UI model
Dynamic tab creation per run:
- Tab title format: `<status-indicator> <process-name> #<short-id> <elapsed>`
- Example indicators:
  - Running: `●` or `…`
  - Completed: `✓`
  - Failed: `✗`
  - Cancelled: `⏹`
- Elapsed display:
  - Running: live timer (`00:42`, `12:08`, `1:03:21`)
  - Terminal: frozen total duration

Status indicator requirement:
- Title updates on lifecycle transitions (start, terminal state, cancellation).
- Keep symbols plain-text for reliable Textual rendering.
- Timer updates at a bounded cadence (recommended: every 1s while tab is visible, 2-5s when not visible).

### D) Background worker execution for each run
Add `_start_process_run(goal)`:
1. Allocate run state + create tab.
2. Spawn non-exclusive worker for that run.
3. Worker executes delegate task and routes events to that run only.

Do not call `_run_process_goal` inline from slash handler.

### E) Run-scoped progress callback
Use callback closure bound to run ID:
- `"_progress_callback": lambda data: self._on_process_progress_event(run_id, data)`

Route updates to:
- that run tab log
- that run progress panel
- optional lightweight chat notices (`started`, `done`, `failed`)

### F) Orchestrator isolation per run
For true concurrency, do not share one cached orchestrator across runs.

Recommended change:
- `DelegateTaskTool` should create a fresh orchestrator per `execute()` call (or support explicit `reuse_orchestrator=False`).

Rationale:
- avoids cross-run state/event coupling
- simplifies cancellation and event filtering

### G) Sidebar behavior under concurrency
Current sidebar `TaskProgressPanel` holds one task list. For multi-run support:
- Option 1 (minimal): sidebar shows active tab run progress only.
- Option 2: sidebar shows a compact run summary list (one row per run).

Phase this after run-tab baseline to avoid over-scoping.

## Command and UX Surface

### Slash commands
- Keep `/run <goal>` as primary.
- Optional additions (phase 2):
  - `/runs` list active/recent runs
  - `/run cancel <run-id-prefix|current>`
  - `/run focus <run-id-prefix>`

### Command palette
Add actions:
- “List process runs”
- “Cancel current process run” (if supported)
- “Focus latest process run tab”

### Tab lifecycle
- Keep completed tabs open by default.
- Add close action per tab (manual) and optional auto-prune config.

## Execution Plan

### Phase 1: Non-blocking single-run architecture
1. Split `_busy` into chat/process lanes.
2. Run `/run` in background worker.
3. Keep one process-run pane (non-dynamic) as proving ground.
4. Validate cowork remains usable while run is active.

### Phase 2: Dynamic per-run tabs + title status indicators
1. Implement dynamic TabPane creation per run.
2. Implement status-indicator title updates.
3. Route logs/progress to run-specific pane.
4. Support multiple simultaneous runs.
5. Add elapsed-time timer loop and freeze duration on terminal states.

### Phase 3: Control and resilience
1. Add run cancellation path (best effort).
2. Add `/runs` listing and palette shortcuts.
3. Add stale-run cleanup and close-tab UX.

### Phase 4: Delegate/orchestrator concurrency hardening
1. Remove or gate orchestrator caching in `DelegateTaskTool`.
2. Add explicit per-run orchestrator lifecycle.
3. Ensure event subscriptions are always cleaned up.

## Testing Strategy

### Unit tests
- Run state transitions (`queued -> running -> terminal`).
- Tab title indicator transitions.
- Run-scoped callback routing (no cross-run bleed).
- Orchestrator-per-run behavior in delegate tool.

### TUI integration tests (Textual pilot)
- Start `/run`, then submit normal chat message while run active.
- Start two `/run` commands; assert two run tabs and independent progress.
- Verify tab titles show running then completed/failed.
- Verify elapsed timer increments while running and freezes after completion/failure.
- Verify no app freeze and input remains responsive.

### Regression tests
- Existing slash autocomplete and process selection behavior.
- Existing cowork tool-call streaming path.
- Existing sidebar file refresh behavior on subtask completion.

## Risks and Mitigations
1. **Textual dynamic tab API edge cases**
- Mitigation: build a small spike first for add/remove/rename tab reliability.

2. **Shared mutable state in tool registry**
- Mitigation: keep process runs delegated via isolated orchestrators; avoid chat tool interference.

3. **Event flood into UI**
- Mitigation: keep token heartbeat throttling and per-run de-dupe windows.

4. **Cancellation semantics unclear for in-flight tool calls**
- Mitigation: document best-effort cancellation first; harden by layer later.

## Acceptance Criteria
1. Launching `/run` does not block normal cowork chat usage.
2. Two concurrent `/run` invocations create two independent run tabs.
3. Each run tab title shows a clear status indicator and updates on completion/failure.
4. Each run tab shows elapsed runtime while active and final duration when done.
5. Progress/events in each run tab remain scoped to that run.
6. No regressions in existing slash command behavior, process selection, or command palette.

## Implementation Notes
- Start by preserving existing `/run` semantics and only changing execution model.
- Defer advanced run history persistence until concurrency UX is stable.
- Keep run IDs short and user-facing (prefix of UUID) for easy command targeting.
