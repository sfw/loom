# Cowork Delegate Dynamic Run Tabs Hardened Plan (2026-03-04)

## Objective
Implement dynamic process-run tabs for cowork `delegate_task` executions in TUI, reusing `/run` behavior so users get:
1. Clear working folder visibility.
2. Anticipated and observed outputs.
3. Live progress/activity with strong liveness guarantees.
4. No silent hangs caused by approval/wait-state mismatches.

This plan is production-oriented: bounded resource use, deterministic lifecycle, failure transparency, telemetry, and safe rollout.

## Scope
1. TUI cowork flow only (not API/web).
2. Delegate calls started via direct `delegate_task` and `run_tool -> delegate_task`.
3. Reuse existing process-run pane and state machinery where possible.
4. Explicitly align delegate defaults with `/run` decisions on execution policy.
5. Add test, config, telemetry, docs, and rollback paths.

## Non-Goals
1. Redesigning orchestrator planning semantics.
2. Replacing cowork chat delegate progress sections.
3. Creating a brand-new tab type when existing process-run pane can be reused.
4. Perfect output prediction for tasks that provide no deliverable metadata.

## Background and Incident Context
Observed incident: delegated cowork task appeared frozen for 20+ minutes. Root cause was approval mode mismatch leading to `waiting_approval` state in a path where no approval was surfaced to the user.

Alignment fix has already been applied:
1. TUI delegate default approval mode now resolves to `disabled`.
2. `/run` and delegate execution surfaces now share the same default decision.

This plan extends that hardening into UX/state parity by giving delegated work a first-class run tab, not just inline chat snippets.

## Baseline (Repo-Accurate)
1. Process-run infrastructure already exists:
2. `ProcessRunState` and `ProcessRunPane` support progress, outputs, activity, status, elapsed time, and working folder metadata.
3. `/run` creates tabs immediately and streams progress through `_on_process_progress_event(...)`.
4. Cowork delegate progress currently updates chat/sidebar but does not open a run tab.
5. Cowork already receives per-event delegate progress callbacks with `tool_call_id`.
6. Output rows currently rely on process deliverables; ad hoc fallback exists as "expected output" rows when tasks exist but deliverables do not.
7. Liveness/heartbeat, refresh coalescing, and tab status machinery are already in place for process runs.

## Product Requirements
### Must Have
1. Automatic tab creation when cowork starts a delegated task.
2. Tab header shows the effective working folder.
3. Progress list updates live from delegate events.
4. Activity log reflects milestone events and terminal result.
5. Outputs section shows anticipated outputs when available and observed output status as files appear.
6. Multiple concurrent delegate tasks do not mix state.
7. Tab lifecycle is deterministic: queued/running/paused/cancel_requested/completed/failed/cancelled.

### Hardened Behavior
1. No unbounded waiting without visible heartbeat.
2. Explicit state when delegate is blocked/waiting for input/approval.
3. Bounded per-tab memory and event rates.
4. Late callback events cannot resurrect finalized tabs.
5. Degrade gracefully when metadata is incomplete.

## UX Contract (Answers to Key User Questions)
1. Where files save:
2. `Working folder` is always shown in header metadata and logged in activity.
3. Relative path is preferred; absolute path fallback when outside workspace root.

4. Anticipated outputs:
5. If deliverables are known, show explicit planned file paths with status (`planned`, `pending`, `completed`, `missing`).
6. If deliverables are unknown, show inferred expected outputs from task labels/phase IDs and progressively replace with observed file paths from tool events.
7. Distinguish confidence/source (`declared`, `inferred`, `observed`) in row suffixes.

8. Progress and activity:
9. Progress rows reflect normalized task/subtask state.
10. Activity shows stage transitions, key tool events, heartbeat/liveness markers, and terminal summary.

## Design Principles
1. Reuse before rewrite: leverage `/run` state machine and pane components.
2. Single source of truth: one normalized run-state path for both `/run` and cowork delegate tabs.
3. Additive contracts only: extend payloads in backward-compatible form.
4. Deterministic ownership: each delegated call keyed by `tool_call_id`.
5. Bounded costs: caps for lines/events/rows and debounce for refresh.

## Architecture Decisions
### Decision 1: Reuse `ProcessRunState` for Delegate Tabs
Add source metadata rather than creating a separate tab state type.

Proposed additive fields:
1. `origin: "slash_run" | "cowork_delegate"`.
2. `origin_tool_call_id: str`.
3. `origin_tool_name: str`.
4. `output_candidates: list[dict]` (planned/inferred/observed descriptors).
5. `output_confidence_mode: "declared" | "inferred" | "mixed"`.

### Decision 2: Delegate Tab Lifecycle Keyed by `tool_call_id`
1. On tool start (`delegate_task` or `run_tool -> delegate_task`), create or reuse a tab bound to that call ID.
2. Route callback events to this run via a map:
3. `delegate_tool_call_id -> run_id`.
4. Finalize on tool completion event and freeze tab state.

### Decision 3: Shared Execution Defaults for TUI
Keep delegate behavior aligned with `/run`:
1. Approval mode default `disabled` in TUI surface.
2. Read roots/auth workspace wired consistently.
3. Progress callback always attached for delegate calls in TUI.

### Decision 4: Output Modeling Strategy
Priority order for anticipated outputs:
1. Declared deliverables from process definition (highest confidence).
2. Delegate event payload `expected_outputs` (new additive field).
3. Inferred outputs from tasks/phase labels (fallback).
4. Observed outputs from `files_changed_paths` events (ground truth).

### Decision 5: Keep Chat Delegate Sections
Do not remove chat collapsible delegate sections. Tabs complement chat and provide persistent structured run view.

## Detailed Implementation Plan
## Phase 0: Contract and State Plumbing
1. Add delegate-origin metadata to `ProcessRunState`.
2. Add maps:
3. `_delegate_run_by_tool_call_id: dict[str, str]`.
4. `_delegate_tab_finalized_tool_calls: set[str]`.
5. Add helper:
6. `_ensure_delegate_process_run_tab(tool_call_id, caller_tool_name, goal, workspace_hint, process_hint) -> ProcessRunState`.
7. Add helper:
8. `_resolve_delegate_workspace(payload, args) -> Path`.
9. Keep all changes additive and no behavior change when feature flag is off.

Primary files:
1. `src/loom/tui/app.py`

## Phase 1: Tab Creation and Event Routing
1. In cowork tool start path, detect delegated target and create delegate run tab immediately.
2. Initialize status to `running` and set launch stage `running`.
3. Route `_on_cowork_delegate_progress_event(...)` into `_on_process_progress_event(..., run_id=...)`.
4. Continue existing chat delegate stream updates in parallel.
5. On tool completion, finalize run status and terminal logs.
6. Ignore late callback events if tool call was finalized.

Primary files:
1. `src/loom/tui/app.py`
2. `tests/test_tui.py`

## Phase 2: Working Folder and Output Semantics
1. Ensure delegate run tab always has a resolved working folder:
2. Source priority: explicit run/delegate workspace -> event metadata -> app workspace.
3. Add output candidate synthesis helper:
4. `_delegate_output_rows(run, event_payload)`.
5. Extend delegate progress payload (additive) with:
6. `expected_outputs` list when known.
7. `workspace_root` and optional `phase_id -> deliverables` mapping when available.
8. Render output rows with source labels:
9. `(planned)` from declared deliverables.
10. `(expected output, inferred)` from heuristics.
11. `(observed)` when file exists or event lists path.

Primary files:
1. `src/loom/tools/delegate_task.py`
2. `src/loom/tui/app.py`
3. `tests/test_delegate_task.py`
4. `tests/test_tui.py`

## Phase 3: Liveness and Hang Hardening
1. Add explicit run state for blocked conditions:
2. `waiting_approval`, `waiting_input`, `stalled`.
3. If callback indicates waiting/blockage, emit high-visibility activity row with reason.
4. Use existing heartbeat path to avoid silent tabs.
5. Add timeout warnings for no-progress windows (non-terminal) with actionable hints.
6. Ensure cancel/pause/play controls for delegate tabs mirror `/run` policy where backend supports it.

Primary files:
1. `src/loom/tui/app.py`
2. `src/loom/tools/delegate_task.py`
3. `tests/test_tui.py`

## Phase 4: Capacity, Backpressure, and Cleanup
1. Cap concurrent delegate tabs (`max_active_delegate_tabs`); oldest terminal tabs auto-demoted/close-suggested.
2. Bound per-tab:
3. Activity log lines.
4. Progress rows.
5. Output rows.
6. Coalesce noisy event bursts and dedupe repeated lines.
7. Ensure run-state maps are cleaned on close/finalize to prevent leaks.

Primary files:
1. `src/loom/tui/app.py`
2. `src/loom/config.py`
3. `loom.toml.example`
4. `docs/CONFIG.md`

## Data Contract Changes (Additive)
### Delegate Progress Payload
Optional fields to add:
1. `expected_outputs: list[str]`
2. `workspace_root: str`
3. `deliverables_by_phase: dict[str, list[str]]`
4. `blocked_reason: str` (for wait states)
5. `status_detail: str` (human-readable)

Compatibility rules:
1. Existing consumers ignore unknown fields.
2. TUI falls back safely when fields absent.

## Telemetry and Operational Metrics
Add structured telemetry:
1. `delegate_tab_created` with `tool_call_id`, `origin_tool`, `run_id`.
2. `delegate_tab_create_latency_ms` from tool start event to tab creation.
3. `delegate_progress_event_lag_ms` callback receive to UI render.
4. `delegate_no_progress_window_ms` when heartbeats fire.
5. `delegate_tab_terminal_state` counts by outcome.
6. `delegate_output_source_ratio` (`declared` vs `inferred` vs `observed`).
7. `delegate_tab_drop_or_evict_count` for capacity controls.

SLO targets:
1. Tab visible within 250ms p95 from delegate tool start.
2. Progress updates rendered within 500ms p95.
3. Workspace/output status refresh within 1.5s p95 after mutating tool completion.
4. No unbounded silent window beyond configured heartbeat interval.

## Security and Safety Considerations
1. Normalize/sanitize file paths before display.
2. Avoid exposing paths outside workspace unless explicitly needed; mark external paths clearly.
3. Do not log secrets from tool args into activity lines.
4. Ensure replay persistence stores bounded, non-sensitive summaries only.

## Testing Strategy
### Unit Tests
1. Delegate tool start creates a run tab keyed to `tool_call_id`.
2. Multiple delegate calls create isolated tabs.
3. Late events after finalization are ignored.
4. Working folder resolution fallback chain works.
5. Output row synthesis precedence works (`declared > expected > inferred > observed`).
6. Bounded buffers enforce caps.

### Integration/TUI Tests
1. Cowork direct `delegate_task` path: tab appears, progress streams, terminal state finalizes.
2. `run_tool -> delegate_task` path: same behavior.
3. Delegate with known deliverables shows planned outputs and completion transitions.
4. Delegate without deliverables shows inferred outputs and upgrades to observed paths.
5. Simulated waiting/blocked state surfaces visible warning and heartbeat.

### Regression Tests
1. `/run` flows unchanged.
2. Existing chat delegate progress sections still render/finalize.
3. Workspace refresh and files panel updates remain stable.
4. Approval mode defaults stay aligned for TUI surface.

## Rollout Plan
### Stage A: Dark Launch (Flag Off by Default)
1. Ship code paths behind `tui.delegate_run_tabs_enabled`.
2. Emit telemetry in both legacy and new paths.
3. Validate no regressions in internal test sessions.

### Stage B: Internal Canary
1. Enable flag for internal dogfood users.
2. Monitor tab creation latency, stale/no-progress windows, error counts, and memory growth.
3. Fix high-volume event edge cases before broad enablement.

### Stage C: Default On
1. Enable by default with rollback toggle retained for at least one release.
2. Keep strict caps and heartbeat diagnostics enabled.
3. Publish docs and changelog updates.

Rollback:
1. Toggle `tui.delegate_run_tabs_enabled = false`.
2. Cowork reverts to chat/sidebar-only delegate progress model.

## Config Additions
Proposed `[tui]` settings:
1. `delegate_run_tabs_enabled = false` (initially)
2. `delegate_run_tabs_autofocus = true`
3. `delegate_run_tabs_max_active = 8`
4. `delegate_run_progress_refresh_interval_ms = 250`
5. `delegate_run_heartbeat_interval_ms = 6000`
6. `delegate_run_output_inference_enabled = true`
7. `delegate_run_activity_max_lines = 800`
8. `delegate_run_output_max_rows = 500`

## Documentation Updates
1. `docs/CONFIG.md`: new flags and behavior.
2. TUI usage docs: delegate tab semantics and controls.
3. `CHANGELOG.md`: feature summary, default state, and known limits.

## PR Slicing (Recommended)
1. PR1: state model + tab creation/routing + tests.
2. PR2: outputs/workspace semantics + payload additions + tests.
3. PR3: liveness/hardening/backpressure + telemetry.
4. PR4: docs/config/changelog and rollout flag wiring.

## Risks and Mitigations
1. Risk: too many tabs during heavy delegation.
2. Mitigation: concurrency cap, eviction policy, optional non-autofocus.

3. Risk: noisy events degrade UI responsiveness.
4. Mitigation: event filtering, dedupe windows, capped logs, refresh throttling.

5. Risk: weak anticipated outputs for metadata-poor tasks.
6. Mitigation: explicit confidence/source labeling and observed-path reconciliation.

7. Risk: lifecycle races between tool completion and callback tail events.
8. Mitigation: finalization guard keyed by `tool_call_id`.

9. Risk: behavior drift between `/run` and cowork delegate execution defaults.
10. Mitigation: shared resolver for execution-surface defaults and regression tests.

## Acceptance Criteria
1. Starting a cowork delegate call opens a dedicated run tab automatically.
2. Tab shows working folder reliably and updates progress/activity live.
3. Outputs pane shows anticipated outputs when known, plus observed output status as execution proceeds.
4. No silent indefinite waiting states without heartbeat and explanatory status.
5. Existing `/run` behavior and cowork chat replay remain regression-free.
6. Feature can be safely disabled by config without code rollback.

