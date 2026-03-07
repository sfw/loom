# 2026-03-07 TUI Core Subsystem Split Plan

## File
- `src/loom/tui/app/core.py` (~10,409 LOC)

## Goal
- Continue shrinking `core.py` into a thin orchestration shell by moving isolated subsystems into logical modules only.
- Keep behavior stable and preserve import contracts:
  - `from loom.tui.app import LoomApp`
  - `from loom.tui.app import ProcessRunPane, ProcessRunList, ProcessRunState, ProcessRunLaunchRequest, SteeringDirective`

## Current Status Snapshot
- Already extracted:
  - `slash/parsing.py`, `slash/registry.py`, `slash/handlers.py`
  - `process_runs/state.py`, `process_runs/launch.py`, `process_runs/events.py`, `process_runs/adhoc.py`, `process_runs/lifecycle.py`
  - `chat/history.py`, `chat/session.py`, `chat/turns.py`, `chat/steering.py`
  - `actions.py`, `rendering.py`, `constants.py`, `models.py`, `widgets.py`
- Remaining in `core.py` are still large cohesive clusters that can be split cleanly.

## Isolated Subsystem Inventory (Next Split Targets)

### 1) Chat Steering + Stop Control
- Current area:
  - steering state helpers near `~611-778`
  - queue management and steering flows `~7221-8898`
- Move to:
  - `src/loom/tui/app/chat/steering.py` (expand existing file)
  - optional `src/loom/tui/app/chat/stop.py` if stop-flow stays large
- Functions to move:
  - `_has_unfinalized_delegate_streams`, `_is_background_work_active`, `_is_cowork_stop_visible`
  - `_refresh_hint_panel`, `_sync_chat_stop_control`
  - `_new_steering_directive`, queue/index/remove/pop helpers
  - `_record_steering_event`, `_queue_chat_inject_instruction`, `_sync_pending_inject_apply_state`
  - `_request_chat_pause`, `_request_chat_resume`, `_clear_chat_steering`, `_request_chat_redirect`
  - `_chat_stop_*`, `_wait_for_chat_turn_settle`, `_finalize_unsettled_delegate_streams_for_stop`, `_request_chat_stop`
  - `action_stop_chat`, `action_inject_chat`, `action_redirect_chat`, `action_steer_queue_*`

### 2) Delegate Progress Streaming
- Current area: `~8918-9153`
- Move to:
  - `src/loom/tui/app/chat/delegate_progress.py` (new)
- Functions to move:
  - `_ensure_delegate_progress_widget`, `_append_delegate_progress_widget_line`
  - `_start_delegate_progress_stream`, `_finalize_delegate_progress_stream`
  - `_on_cowork_delegate_progress_event`

### 3) Process-Run UI State + Persistence
- Current area: `~3819-4714`
- Move to:
  - `src/loom/tui/app/process_runs/ui_state.py` (new)
  - `src/loom/tui/app/process_runs/persistence.py` (new)
- Functions to move:
  - id/elapsed/status/activity/stage helpers
  - `_refresh_process_run_progress`, `_set_process_run_launch_stage`, `_fail_process_run_launch`
  - `_serialize_process_run_state`, `_sync_process_runs_into_session_state`
  - `_persist_process_run_ui_state`, `_persisted_process_tabs_payload`
  - `_drop_process_run_tabs`, `_restore_process_run_tabs`
  - `_format_process_run_tab_title`, `_update_process_run_visuals`, `_tick_process_run_elapsed`

### 4) Process-Run Workspace Provisioning + Context
- Current area: `~4726-4920`
- Move to:
  - `src/loom/tui/app/process_runs/workspace.py` (new)
- Functions to move:
  - `_build_process_run_context`
  - `_llm_process_run_folder_name`, `_prepare_process_run_workspace`
  - `_next_available_process_run_folder_name`, `_materialize_process_run_workspace_selection`
  - `_prompt_process_run_workspace_choice`, `_choose_process_run_workspace`
- Keep utility pure functions already in `process_runs/launch.py`.

### 5) Process-Run Controls (close/pause/play/stop/inject/resume/restart)
- Current area: `~4961-6013`
- Move to:
  - `src/loom/tui/app/process_runs/controls.py` (new)
- Functions to move:
  - target/current resolution helpers
  - close/force-close/stop confirmation modals
  - cancel/pause/play/inject/question answer requests
  - close/pause/play/inject/stop from run and from target
  - restart-in-place flow and resume seed helpers

### 6) Process-Run Auth Preflight
- Current area: `~6154-6270`
- Move to:
  - `src/loom/tui/app/process_runs/auth.py` (new)
- Functions to move:
  - profile option rendering
  - auth selection prompt flow
  - open auth manager for run start
  - collect required resources + resolve overrides for run start

### 7) Process-Run Output Rows + Questions
- Current area: `~9364-9814`
- Move to:
  - `src/loom/tui/app/process_runs/rendering.py` (new)
  - `src/loom/tui/app/process_runs/questions.py` (new)
- Functions to move:
  - task normalization + phase inference + output rows
  - `_refresh_process_run_outputs`, `_mark_process_run_failed`
  - `_prompt_process_run_question`, `_handle_ask_user`
  - `_update_sidebar_tasks`, `_summarize_cowork_tasks`, `_refresh_sidebar_progress_summary`

### 8) Workspace Watch + Files Panel Ingestion
- Current area: `~9910-10281`
- Move to:
  - `src/loom/tui/app/workspace_watch.py` (new)
  - `src/loom/tui/app/files_panel.py` (new)
- Functions to move:
  - workspace signature scanning/poll/debounce/refresh
  - file path normalization/markers/operation hint
  - files panel ingestion from paths/tool events and turn summaries

### 9) Slash Hint/Completion + Command Catalog UI
- Current area:
  - command catalog/help/usage `~6585-6740`
  - hint/completion/input history glue `~6821-7139`
- Move to:
  - `src/loom/tui/app/slash/hints.py` (new)
  - `src/loom/tui/app/slash/completion.py` (new)
- Functions to move:
  - slash catalog/help line rendering
  - tool slash hint rendering
  - slash completion candidate matching/tab cycle helpers
  - slash hint rendering and UI sync helpers

### 10) Startup Surface + Manager Tabs + Session Bootstrap
- Current area:
  - startup/landing/mount lifecycle `~585-1057`
  - manager tabs `~1138-1335`
  - session bootstrap `~3163-3800` and session wrappers near `~7932-8023`
- Move to:
  - `src/loom/tui/app/startup.py` (new)
  - `src/loom/tui/app/manager_tabs.py` (new)
  - expand `src/loom/tui/app/chat/session.py`

## Execution Phases (Recommended Order)

### Phase A: Chat steering and delegate stream extraction
- Move subsystem 1 and 2 first.
- Rationale: high cohesion and low coupling to process-run internals.

### Phase B: Process-run controls + auth + workspace
- Move subsystem 4, 5, 6.
- Rationale: isolate all run command/control flows behind `process_runs/*`.

### Phase C: Process-run UI state, persistence, and output rendering
- Move subsystem 3 and 7.
- Rationale: consolidates run tab rendering/state transitions in one package.

### Phase D: Workspace/file refresh subsystem
- Move subsystem 8.
- Rationale: single-responsibility refresh pipeline and file panel ingestion.

### Phase E: Slash hint/completion/catalog UI
- Move subsystem 9.
- Rationale: keeps all slash UX behavior together and testable.

### Phase F: Startup + manager tabs + session bootstrap
- Move subsystem 10.
- Rationale: leaves `core.py` as wiring/orchestration only.

### Phase G: Final core compaction
- Remove temporary wrappers where tests pin new seams.
- Ensure `core.py` mainly contains:
  - class wiring
  - compose/on_mount/on_unmount shell hooks
  - minimal delegating methods.

## Guardrails
- Keep `src/loom/tui/app/__init__.py` as thin facade only.
- Internal modules must import concrete siblings (`.chat.*`, `.process_runs.*`, `.slash.*`) and never through `loom.tui.app`.
- Each new module must contain one concern domain only.
- No behavior changes during extraction unless bugfix is required to preserve existing tests.

## Test Plan Per Phase
- Always run:
  - `uv run pytest tests/tui/app`
  - `uv run pytest tests/test_tui.py tests/test_setup.py`
- Add focused tests as modules land:
  - `tests/tui/app/test_chat_steering.py`
  - `tests/tui/app/test_delegate_progress.py`
  - `tests/tui/app/test_process_runs_controls.py`
  - `tests/tui/app/test_process_runs_persistence.py`
  - `tests/tui/app/test_workspace_watch_refresh.py`
  - `tests/tui/app/test_slash_completion_hints.py`
  - `tests/tui/app/test_startup_manager_tabs.py`

## Exit Criteria
- `core.py` <= 2,500 LOC.
- All run controls/rendering/auth/workspace logic lives in `process_runs/*`.
- All steering/stop/delegate-stream logic lives in `chat/*`.
- All slash hint/completion/catalog logic lives in `slash/*`.
- Required tests above pass without regressions.
