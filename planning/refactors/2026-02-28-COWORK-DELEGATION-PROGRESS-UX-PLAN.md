# Cowork Delegation Progress UX Plan (2026-02-28)

## Objective
Improve cowork UX when the model uses `delegate_task`:
1. Show delegated subtask progress live in the sidebar Progress panel (not only after tool completion).
2. Stream delegated subagent context into chat inside a collapsed section so users can inspect progress without chat noise.
3. Eliminate routine manual TUI reloads by making file tree/files panel/chat/progress updates appear immediately.

## Why This Plan
Current behavior gives partial visibility:
1. `/run` already receives incremental delegate progress via `_progress_callback`.
2. Cowork updates sidebar tasks only once the `delegate_task` tool call finishes.
3. Chat shows a start row and a final tool result row, but no live delegated context in between.
4. File tree/files panel can lag behind real workspace changes unless explicit refresh paths fire.

Result: long-running delegation in cowork appears "stuck" until the tool returns.

## Baseline (Repo-Accurate)
1. `delegate_task` emits progress payloads incrementally through `_progress_callback` in `src/loom/tools/delegate_task.py`.
2. `/run` wires that callback in `src/loom/tui/app.py` and uses `_on_process_progress_event(...)` for live updates.
3. Cowork interaction loop (`_run_interaction`) handles tool start/completion events but does not wire live delegate callbacks.
4. Sidebar summary helpers already exist (`_update_sidebar_tasks`, `_refresh_sidebar_progress_summary`).
5. Chat supports collapsible tool output blocks (`ToolCallWidget`) but no live-updating collapsible stream for delegated progress.
6. Workspace tree refresh is trigger-based (`_refresh_workspace_tree`) with manual reload action; there is no continuous external-change observer.
7. `_WORKSPACE_REFRESH_TOOLS` is static and stale (currently includes `document_create`, which is not a current tool), so some writes are not reflected until later lifecycle events.
8. Files Changed panel is updated from `CoworkTurn` summary, so updates land after the turn loop, not at individual tool completion.
9. Sidebar `TaskProgressPanel` is mounted with default `auto_follow=False`.
10. Chat streaming flushes by chunk threshold; there is no periodic time-based flush for sparse streams.
11. Cowork chat now replays via `_append_chat_replay_event` + `_render_chat_event` + `_hydrate_chat_history_for_active_session`; new delegate stream UI events must integrate into this replay contract.

## Scope
1. Cowork mode UX changes in TUI only.
2. Reuse existing delegate progress payload shape and existing progress message formatter where possible.
3. Keep `/run` behavior unchanged except shared helper reuse if useful.
4. Add tests for new cowork progress plumbing and chat rendering behavior.
5. Support both direct `delegate_task` calls and `run_tool` calls that target `delegate_task`.
6. Include docs/changelog updates for cowork progress UX behavior.
7. Add realtime UI coherence for file tree/files panel/chat/progress, including external workspace edits and shell side effects.

## Non-Goals
1. Changes to orchestrator planning/execution semantics.
2. API/web frontend changes.
3. Persisting every delegate progress tick into conversation history (only bounded lifecycle events should be journaled for resume safety).
4. Replacing process-run tabs with cowork delegate tabs.
5. Hard real-time (<100ms) guarantees for very large workspaces.

## Production Requirements
1. UI staleness SLO:
2. Progress/chat updates visible within 500ms p95 after event emission.
3. Workspace tree updates visible within 1.5s p95 after local/external file mutation.
4. Backpressure:
5. Refresh requests are coalesced; no unbounded refresh queues.
6. No refresh storm under high-frequency tool/event streams.
7. Safety:
8. Background file-watch/poll loops never mutate widgets off the UI thread.
9. Watcher failures degrade gracefully to periodic polling/manual reload (no hard TUI failure).
10. Resource budget:
11. Idle watcher CPU target <2% on medium repos; bounded memory growth for file/progress/chat buffers.
12. Operability:
13. Add structured diagnostics for refresh requested/executed/skipped, queue depth, and elapsed latency.

## Implementation Strategy
Deliver in four phases to de-risk UX and preserve momentum.

### Phase 0: Decouple Progress Routing
Goal: avoid coupling sidebar updates with chat emission before cowork streaming work.

#### Design
1. Split current progress handling into separate responsibilities:
2. Normalize/update progress state for sidebar.
3. Emit chat/event-panel progress lines (optional per caller/context).
4. Keep `/run` behavior unchanged by default.

#### Proposed Changes
1. `src/loom/tui/app.py`
2. Extract helper(s) from `_on_process_progress_event(...)`:
3. `update_delegate_progress_state(...)` for task/sidebar/workspace refresh.
4. `emit_delegate_progress_chat(...)` for chat/event-panel emission.
5. Keep a compatibility wrapper for existing `/run` call sites.

#### Acceptance (Phase 0)
1. Existing `/run` tests remain green.
2. Cowork can consume progress state updates without implicit chat spam.

### Phase 1: Live Sidebar Progress in Cowork
Goal: parity with `/run`-style incremental progress visibility in cowork HUD.

#### Design
1. Add a cowork-session-level delegate progress callback hook.
2. Inject `_progress_callback` in one place (`_prepare_tool_execute_arguments`) so direct calls and `run_tool -> delegate_task` both pick it up.
3. Route callback payloads through the new state-only progress helper (no chat emission).
4. Keep callback optional so non-TUI usage of `CoworkSession` remains unaffected.
5. Add per-call metadata routing so callback payloads include `tool_call_id` and `tool_name` for downstream chat/section binding.

#### Proposed Changes
1. `src/loom/cowork/session.py`
2. Add optional constructor arg for delegate progress callback.
3. Extend `_prepare_tool_execute_arguments(...)` to inject `_progress_callback` for `delegate_task`.
4. Ensure `_dispatch_run_tool(...)` uses that same argument-prep path so delegated `delegate_task` receives `_progress_callback`.
5. Ensure approval prompt args remain clean (do not leak runtime-only callback noise).
6. Include call metadata (`tool_call_id`, `tool_name`) in callback payload for downstream chat mapping.
7. Update both streaming loops (`send(...)` and `send_streaming(...)`) so start/completion `ToolCallEvent` instances carry `tool_call_id`.
8. `src/loom/tui/app.py`
9. Pass callback when creating `CoworkSession` in all construction sites (initialize/new/switch/resume paths).
10. Callback target should call the new state-only progress helper with cowork scope.
11. Ensure sidebar summary remains concise (existing one-row cowork summary stays as HUD output).

#### Acceptance (Phase 1)
1. During cowork `delegate_task`, sidebar Progress row updates while tool is running.
2. No regression in `/run` progress behavior.
3. If no progress events are emitted (fast or failed delegate), behavior still degrades gracefully to final result update.
4. Cowork does not emit unstructured per-event info lines in chat during Phase 1.

### Phase 2: Collapsed Live Subagent Context in Chat
Goal: provide inspectable delegated context without flooding main chat.

#### Design
1. Introduce a dedicated live-updating collapsible widget for delegated progress stream.
2. One stream section per `delegate_task` call.
3. Section defaults collapsed, appends concise lines from a cowork-specific formatter mode, and shows completion/failure state.
4. Throttle/dedupe noisy events (`token_streamed`) similarly to existing process-run activity policy.
5. Guard against late callback events after tool completion/finalization.

#### UI Behavior
1. On `delegate_task` start:
2. Chat adds a collapsed "Delegated progress" section bound to that tool call.
3. While running:
4. High-signal progress events append lines inside the section.
5. On completion:
6. Section title or footer indicates success/failure and elapsed time.
7. Final `ToolCallWidget` remains as the canonical tool result row.

#### Proposed Changes
1. `src/loom/cowork/session.py`
2. Extend `ToolCallEvent` with `tool_call_id` so UI can bind stream updates to correct tool call row/section.
3. `src/loom/tui/widgets/chat_log.py`
4. Add APIs to create/update/finalize delegate progress sections by `tool_call_id`.
5. `src/loom/tui/widgets/tool_call.py` or new widget file
6. Implement `DelegateProgressWidget` (collapsed by default, mutable body).
7. `src/loom/tui/app.py`
8. On delegate tool start, initialize progress section.
9. On delegate progress callback, append formatted message to that section.
10. Track active delegate stream IDs and ignore late events after stream finalization.
11. On tool completion, finalize section state and stop routing updates.
12. Add cowork-specific message variants for terminal events (for example, avoid "Process run completed.").
13. Extend replay contract so delegate-section lifecycle is restorable.
14. Add normalized replay events (for example `delegate_progress_started` and `delegate_progress_finalized`) via `_append_chat_replay_event(...)`.
15. Update `_render_chat_event(...)` to render those events during hydrate/rerender paths.
16. Keep high-frequency line appends in-memory only (`persist=False`) or aggressively coalesced so chat journal does not balloon.

#### Event Filtering and Throttling Policy
1. Include: `task_planning`, `task_plan_ready`, `task_executing`, `subtask_*`, `tool_call_*`, `task_replanning`, `task_stalled*`, `task_completed`, `task_failed`.
2. Exclude or aggressively throttle: `token_streamed`.
3. Deduplicate repeated identical lines within short windows to avoid spam.
4. Cap retained lines per section (for example last 150) to bound memory/render cost.

#### Acceptance (Phase 2)
1. Every cowork `delegate_task` call gets its own collapsed progress section.
2. Section receives live incremental updates during execution.
3. Multiple delegated tool calls in one turn do not mix streams.
4. Chat remains readable; no runaway `token_streamed` spam.

### Phase 3: Realtime TUI Coherence (No Manual Reload Loop)
Goal: file/chat/progress surfaces update promptly from both in-app and external changes.

#### Design
1. Introduce a debounced workspace-refresh coordinator used by all refresh producers.
2. Replace static mutating-tool checks with dynamic detection (`tool.is_mutating`) where practical.
3. Add external workspace-change detection with explicit backend policy:
4. Default backend: polling snapshot loop (cross-platform, no extra dependency).
5. Optional backend: native watcher via `watchfiles` when dependency is installed/configured.
6. Automatic fallback from watcher backend to polling on runtime errors.
7. Update Files panel on tool completion events rather than only end-of-turn summaries.
8. Improve perceived chat/progress liveness (auto-follow and time-based stream flush).
9. Emit richer tool-completion event metadata (`files_changed`) so `/run` and cowork share one immediate-files path.
10. Bound all incremental UI buffers (files/progress/delegate stream lines) to prevent unbounded growth.
11. Define overflow behavior when workspace scan budget is exceeded (coarse refresh mode + one-time user notice).

#### Proposed Changes
1. `src/loom/tui/app.py`
2. Add `_request_workspace_refresh(reason, immediate=False)` with debounce/throttle.
3. Route existing refresh call sites through coordinator.
4. Replace static `_WORKSPACE_REFRESH_TOOLS` checks with helper-based mutating-tool detection (`self._tools.get(name).is_mutating` + narrow fallback allowlist during bootstrap).
5. Add low-cost external workspace polling interval on mount to detect non-tool edits and shell side effects.
6. Update process command index in background when workspace/process files change.
7. `src/loom/engine/runner.py`
8. Add `files_changed` to `tool_call_completed` event payloads (sanitized workspace-relative paths, capped length, truncation flag).
9. Keep payload contract additive/backward-compatible for existing consumers.
10. `src/loom/tools/delegate_task.py`
11. Preserve/forward enriched event payload data to progress callbacks.
12. `src/loom/tui/widgets/sidebar.py`
13. Enable auto-follow behavior for progress panel (or add active-row follow policy).
14. `src/loom/tui/widgets/chat_log.py`
15. Add periodic stream-buffer flush cadence for sparse token streams.
16. `src/loom/tui/widgets/file_panel.py` and `src/loom/tui/app.py`
17. Add per-tool completion ingestion path using `files_changed` (with dedupe window) for immediate Files panel updates.
18. Add bounded history caps (for example max file rows and max delegate stream lines).
19. `src/loom/config.py`, `loom.toml.example`, `docs/CONFIG.md`, `pyproject.toml`
20. Add explicit tunables and defaults:
21. `[tui].realtime_refresh_enabled = true`
22. `[tui].workspace_watch_backend = "poll"` (`"poll"` or `"native"`)
23. `[tui].workspace_poll_interval_ms = 1000`
24. `[tui].workspace_refresh_debounce_ms = 250`
25. `[tui].workspace_refresh_max_wait_ms = 1500`
26. `[tui].workspace_scan_max_entries = 20000`
27. `[tui].chat_stream_flush_interval_ms = 120`
28. `[tui].files_panel_max_rows = 2000`
29. `[tui].delegate_progress_max_lines = 150`

#### Acceptance (Phase 3)
1. Creating/modifying/deleting files in workspace (including external editor/shell) updates sidebar tree without manual reload.
2. Tool-driven file changes appear in Files panel at tool completion time, not only end-of-turn.
3. Progress panels auto-follow current activity so new rows are visible.
4. Chat streaming visibly advances under sparse chunk cadence without waiting for large chunk batches.
5. Reload action remains available but is no longer routinely required.
6. Refresh coordinator keeps p95 latency inside SLO while avoiding storms under bursty events.
7. Switching watch backend or backend failure does not require app restart and preserves refresh behavior.
8. When scan budget is exceeded, TUI enters documented coarse-refresh mode without freezing.

## Workstreams and File Touchpoints

### W1: Session Callback Plumbing (Phase 1 Foundation)
Files:
1. `/Users/sfw/Development/loom/src/loom/cowork/session.py`
2. `/Users/sfw/Development/loom/src/loom/tui/app.py`

Deliverables:
1. Optional delegate progress callback in `CoworkSession`.
2. Delegate-only `_progress_callback` injection through `_prepare_tool_execute_arguments(...)`.
3. `run_tool -> delegate_task` callback injection parity through shared argument preparation.
4. `ToolCallEvent.tool_call_id` populated in both `send(...)` and `send_streaming(...)`.
5. TUI hookup in all `CoworkSession(...)` creation paths.

### W2: Sidebar Live Update Integration
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`

Deliverables:
1. Extract progress-state update helper(s) from `_on_process_progress_event`.
2. Use state-only progress handling for cowork callback payloads.
3. Maintain concise sidebar summary via existing `_summarize_cowork_tasks`.

### W3: Collapsed Chat Stream Widget
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/widgets/chat_log.py`
2. `/Users/sfw/Development/loom/src/loom/tui/widgets/tool_call.py` (or new `delegate_progress.py`)
3. `/Users/sfw/Development/loom/src/loom/tui/app.py`
4. `/Users/sfw/Development/loom/src/loom/cowork/session.py`

Deliverables:
1. Mutable collapsible section keyed by `tool_call_id`.
2. Lifecycle hooks: start/update/finalize.
3. Stream routing with stale-event guards after finalization.
4. Cowork-specific progress message formatting.
5. Replay/hydration compatibility for delegate section lifecycle events.

### W4: Test Coverage
Files:
1. `/Users/sfw/Development/loom/tests/test_cowork.py`
2. `/Users/sfw/Development/loom/tests/test_tui.py`
3. `/Users/sfw/Development/loom/tests/test_orchestrator.py`
4. `/Users/sfw/Development/loom/tests/test_config.py`
5. Optional new widget-focused test module if needed.

Deliverables:
1. Session emits/forwards delegate progress callback payloads during tool execution.
2. Cowork sidebar updates before tool completion.
3. Collapsed section creation and incremental update behavior.
4. Multi-delegate isolation by `tool_call_id`.
5. Throttle/dedupe behavior for noisy events.
6. `run_tool -> delegate_task` callback injection coverage.
7. Late callback events are ignored after section/tool finalization.
8. Approval preview remains sane with runtime-only callback args present.
9. Runner `tool_call_completed` payload assertions updated for additive `files_changed` metadata.
10. Config parsing/default coverage for new `[tui]` realtime knobs.

### W5: Docs and Changelog
Files:
1. `/Users/sfw/Development/loom/CHANGELOG.md`
2. Relevant user-facing docs if cowork progress UX is documented elsewhere.

Deliverables:
1. Changelog entry for cowork delegated live sidebar progress.
2. Changelog/doc entry for collapsed delegated progress stream in chat.
3. Changelog/doc entry for realtime refresh behavior and new tunables/fallback behavior.

### W6: Realtime Refresh Infrastructure
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`
2. `/Users/sfw/Development/loom/src/loom/tui/widgets/sidebar.py`
3. `/Users/sfw/Development/loom/src/loom/tui/widgets/chat_log.py`
4. `/Users/sfw/Development/loom/src/loom/tui/widgets/file_panel.py`
5. `/Users/sfw/Development/loom/src/loom/engine/runner.py`
6. `/Users/sfw/Development/loom/src/loom/tools/delegate_task.py`
7. `/Users/sfw/Development/loom/src/loom/config.py`
8. `/Users/sfw/Development/loom/loom.toml.example`
9. `/Users/sfw/Development/loom/docs/CONFIG.md`
10. `/Users/sfw/Development/loom/pyproject.toml`

Deliverables:
1. Debounced workspace refresh coordinator + unified call sites.
2. External workspace-change detection loop.
3. Immediate Files panel updates on tool completion.
4. Progress auto-follow improvements and sparse-stream flush improvements.
5. Enriched `tool_call_completed` payloads with `files_changed`.
6. Configurable refresh/watch tunables with safe defaults.
7. Buffer caps and dedupe rules for long-running sessions.
8. Additive event-schema change documentation (payload size caps + truncation semantics).
9. Optional `watchfiles` dependency wiring for native backend.

## Test Plan

### Unit/Component
1. `CoworkSession`:
2. `delegate_task` args receive injected `_progress_callback` when callback hook exists.
3. Non-delegate tools are unaffected.
4. `run_tool` targeting `delegate_task` also receives callback injection.
5. `ToolCallEvent` contains `tool_call_id`.
6. Approval preview/input paths are not polluted by runtime callback payload fields.
7. Chat widget:
8. Section starts collapsed.
9. Appending lines updates visible content.
10. Finalization updates status label.
11. Late events after finalization are ignored.
12. Chat stream timer flushes buffered sparse chunks within configured latency budget.
13. Progress panel auto-follow behavior keeps newest relevant row visible.
14. Files panel ingestion deduplicates repeated tool-change rows.
15. `tool_call_completed` payload includes `files_changed` and preserves workspace confinement.
16. Refresh coordinator debounce/coalescing behavior is deterministic under synthetic burst.
17. Watcher/poller failure path downgrades gracefully and logs one actionable warning.
18. `files_changed` payload cap/truncation behavior is deterministic and documented.
19. Scan-budget overflow path triggers coarse-refresh mode and one-time operator notice.
20. Replay/hydrate path restores delegate-progress lifecycle rows without chat parse failures.
21. Delegate high-frequency line updates are not fully journal-persisted (bounded journal growth test).

### TUI Integration
1. During cowork interaction with mocked delegate progress stream, assert `sidebar.update_tasks(...)` called before final tool completion.
2. Assert chat progress section gets appended lines for selected event types.
3. Assert repeated `token_streamed` does not flood chat.
4. Assert progress sections are independent for sequential delegate calls.
5. Assert cowork terminal text uses cowork phrasing (not "/run" phrasing).
6. Simulate external workspace file change and assert tree refresh triggers without `action_reload_workspace`.
7. Assert mutating tool completion triggers workspace refresh via dynamic mutating-tool policy.
8. Assert Files panel receives update on tool completion before `CoworkTurn` finalization.
9. Assert external mutation bursts do not cause repeated redundant tree reloads.
10. Assert process command index refresh coalesces with workspace refresh bursts.
11. Assert backend fallback path (`native` -> `poll`) keeps UI updates functioning.

### Regression
1. Existing `/run` progress tests stay green.
2. Existing cowork tool-call rendering tests stay green.
3. No change in approval behavior for `delegate_task`.
4. No duplicate chat spam in cowork Phase 1 (state-only progress path).
5. No refresh storms under high event throughput (debounce respected).
6. No cross-thread widget mutation exceptions in watcher/poller paths.
7. Additive event payload changes do not break pre-existing progress formatting paths.
8. Existing orchestrator tool-event contract tests are updated from strict equality to additive assertions where applicable.

### Performance/Soak
1. Synthetic burst: 100+ progress/tool events in short interval keeps UI responsive.
2. Large-workspace polling scenario remains within CPU budget targets.
3. Long-running session soak does not exceed configured buffer caps.
4. Event payload size remains bounded under large file-change bursts.

## Rollout Plan
1. Merge Phase 0 first (progress routing split), no UX change expected.
2. Merge Phase 1 (live sidebar only), then validate in real cowork sessions with long-running delegation.
3. Merge Phase 2 (collapsed chat stream) after widget tests pass and manual UX validation.
4. Merge Phase 3 realtime-coherence improvements in small slices (refresh coordinator, then external watch, then chat/files polish).
5. Land W5 docs/changelog updates with each user-visible phase.
6. If chat noise/perf issues appear, keep sidebar improvements and gate collapsed stream behind a config toggle.
7. Feature-flag initial external watcher/poller path; enable by default after dogfood and perf checks pass.
8. Capture refresh metrics during rollout and validate SLO/CPU thresholds before broad enablement.
9. Canary rollout gate: 48h internal dogfood with zero critical refresh regressions and SLO compliance before default-on native watcher option.

## Risks and Mitigations
1. Risk: UI update churn from high-frequency events.
2. Mitigation: filter + throttle + dedupe + bounded retained lines.
3. Risk: event routing to wrong chat section.
4. Mitigation: propagate `tool_call_id` end-to-end and assert mapping in tests.
5. Risk: callback errors break tool execution.
6. Mitigation: callback invocation wrapped defensively (best-effort, non-fatal).
7. Risk: late async callback delivery updates closed sections.
8. Mitigation: active-stream registry + ignore-after-finalize checks.
9. Risk: constructor signature change ripples through tests.
10. Mitigation: callback arg optional with backward-compatible default.
11. Risk: incorrect cowork verbiage from reused `/run` formatter.
12. Mitigation: cowork-specific formatter mode with explicit tests.
13. Risk: external watcher/poller causes performance churn in large repos.
14. Mitigation: debounce, hidden-dir filtering, bounded scan scope, conservative interval.
15. Risk: frequent refreshes make UI noisy or jittery.
16. Mitigation: coalesced refresh coordinator + reason-tagged diagnostics.
17. Risk: missing `files_changed` in event payload blocks immediate Files panel parity between `/run` and cowork.
18. Mitigation: add payload contract in runner + delegate plumbing + test coverage.
19. Risk: unbounded panel buffers degrade long sessions.
20. Mitigation: add configurable caps + truncation indicators + soak tests.
21. Risk: oversized event payloads from mass file operations degrade UI responsiveness.
22. Mitigation: path-count caps + truncation metadata + payload-size regression tests.

## Acceptance Checklist
1. Cowork delegated tasks show live sidebar progress while running.
2. Cowork chat shows a collapsed delegated progress section with live updates.
3. Progress stream remains concise and readable under long runs.
4. Direct `delegate_task` and `run_tool -> delegate_task` both stream progress.
5. No `/run` regressions.
6. No session persistence/resume regressions.
7. Docs/changelog reflect shipped UX changes.
8. Normal file/chat/progress updates no longer require frequent manual reload.
9. Refresh and watcher behavior is configurable, observable, and safe under failure.
10. Watch backend defaults and fallback behavior are explicitly documented and tested.

## Suggested Execution Order
1. Phase 0 routing split in W2.
2. W1 callback plumbing (including `run_tool -> delegate_task`) + `ToolCallEvent.tool_call_id`.
3. W2 cowork sidebar live updates via state-only path.
4. W4 tests for Phase 0/1 baseline.
5. W3 collapsed chat stream widget, replay-event schema updates, routing, and stale-event guards.
6. W6 realtime refresh infrastructure (coordinator, external watch, files/chat/progress immediacy).
7. W4 tests for collapsed stream + replay/hydrate + realtime-coherence behavior.
8. W5 docs/changelog.
9. Manual cowork validation with long-running delegation and external file edits.
