# Process Run Cancel Hang Hardening Plan (2026-02-28)

## Objective
Eliminate TUI hangs when closing a running process tab (`Ctrl+W`) by making cancellation deterministic end-to-end (TUI, delegate tool, orchestrator), bounding UI wait paths, and adding backpressure/telemetry so cancel remains responsive under heavy progress traffic.

## Incident Summary
Observed behavior:
1. In a running process tab, `Ctrl+W` often appears to hang.
2. For completed/stopped process tabs, `Ctrl+W` works as expected.
3. During the hang, `Ctrl+C` can still open exit modal, so the app is degraded but not fully deadlocked.

Confirmed evidence:
1. Run log for the affected run continued emitting `model_invocation`/`token_streamed`/`progress_snapshot` events with no cancellation marker.
2. Current close path in TUI cancels local worker but does not call backend task cancellation API.
3. Orchestrator dispatch paths currently catch `BaseException`, which can absorb `asyncio.CancelledError` and convert cancellation into normal failed-subtask flow.

## Root Causes (Repo-Accurate)
1. Cancellation semantic mismatch:
2. `Ctrl+W` close flow is UI-local (`run.worker.cancel()`) and does not explicitly request orchestrator `cancel_task` for the underlying delegated task ID.
3. Cancellation propagation bug:
4. Orchestrator `execute_task` dispatch paths catch `BaseException` around subtask dispatch/gather outcomes, which includes `CancelledError`.
5. This can prevent task-level cancellation from short-circuiting execution.
6. Close flow liveness risk:
7. Close confirmation waits on an unbounded future; inflight guard blocks reentry while pending.
8. Under load, this can create perceived lock where repeated `Ctrl+W` does nothing.
9. Event pressure:
10. Running delegates can emit dense progress snapshots with large payloads, increasing UI update pressure during cancel windows.

## Production Requirements
1. `Ctrl+W` on a running tab must initiate true task cancellation, not only local UI worker cancellation.
2. Cancellation must propagate through delegate/orchestrator without swallowing `CancelledError`.
3. Close/cancel UX must remain responsive under high event rates.
4. User must always see one of: `Cancel requested`, `Cancelled`, `Cancel timed out`, or `Force close`.
5. No regression for non-running tab close behavior.
6. Existing `/run close`, `/run resume`, and `Ctrl+C` exit flows remain functional.
7. Add telemetry sufficient to prove cancel p95 latency and identify stalls.

## Non-Goals
1. Redesigning process package semantics.
2. Replacing delegate/orchestrator architecture.
3. Broad TUI visual redesign outside cancel/close ergonomics.

## Design Decisions

### 1) Separate Intent: Cancel Run vs Close Tab
1. Running tab close should become a two-step operation:
2. Step A: request cancellation for the underlying task.
3. Step B: close tab after terminal cancellation state or explicit force-close decision.
4. Completed/failed tabs still close immediately.

### 2) Task Cancellation Is Source of Truth
1. If `run.task_id` is present, TUI must call the cancel API path (`DELETE /tasks/{task_id}`) or equivalent engine cancel entrypoint.
2. Local worker cancellation remains a fallback, not the primary mechanism.

### 3) Cancellation Must Never Be Swallowed
1. In orchestrator dispatch loops, `CancelledError` must be re-raised, not converted to failed outcome.
2. Broad `BaseException` capture should be removed or narrowed with explicit cancellation passthrough.

### 4) Bound Close-Flow Waits
1. Confirmation/cancel waits must have bounded timeout and deterministic cleanup.
2. Inflight latch must always release on timeout/error/cancel paths.

### 5) Backpressure for Progress Handling
1. Coalesce or rate-limit run progress UI updates per run.
2. Keep high-frequency token activity lightweight and avoid expensive full-surface redraw on every burst.

## Implementation Workstreams

### Workstream 0: Cancellation Contract and State Machine
1. Define explicit run close/cancel states in TUI state model:
2. `running -> cancel_requested -> cancelled|cancel_failed|force_closed`.
3. Clarify transitions and user-visible messages for each state.

Primary files:
1. `src/loom/tui/app.py`
2. `src/loom/tui/screens/process_run_close.py`

### Workstream 1: TUI Close Flow Hardening
1. Refactor `_close_process_run` for running runs:
2. Prompt user with explicit cancel intent.
3. Issue task cancel request when `task_id` exists.
4. Start bounded wait for terminal acknowledgement.
5. Offer force-close path if timeout expires.
6. Keep inflight guard scoped per run (or ensure global guard cannot wedge future attempts).
7. Add explicit progress line in run activity: `Cancellation requested...`.
8. Ensure modal/future wait has timeout and guaranteed cleanup.

Primary files:
1. `src/loom/tui/app.py`
2. `src/loom/tui/screens/process_run_close.py`
3. `tests/test_tui.py`

### Workstream 2: Delegate Cancellation Bridging
1. Ensure delegate execution path records task ID early and uses it for cancel calls.
2. On cancellation of waiting delegate task, propagate cancellation to orchestrator task status if task exists.
3. Emit explicit cancel lifecycle events in delegate event log:
4. `task_cancel_requested`, `task_cancel_ack`, `task_cancel_timeout`.

Primary files:
1. `src/loom/tools/delegate_task.py`
2. `tests/test_delegate_task.py`

### Workstream 3: Orchestrator Cancellation Correctness
1. Replace `except BaseException` catch sites in dispatch loop with cancellation-safe handling:
2. `except asyncio.CancelledError: raise`
3. `except Exception as exc: ...`
4. In gather result processing, detect cancellation exceptions and re-raise cancellation.
5. Add regression tests proving cancellation propagation during:
6. single-subtask dispatch
7. multi-subtask gather
8. in-flight sleep/model/tool wait

Primary files:
1. `src/loom/engine/orchestrator.py`
2. `tests/test_orchestrator.py`

### Workstream 4: API and CLI Cancel Path Consistency
1. Reuse existing cancel endpoint behavior for TUI cancellation.
2. Normalize error handling for `404`, non-running status, and already-terminal states.
3. Return user-safe status messaging in TUI for each response class.

Primary files:
1. `src/loom/api/routes.py` (if needed for stronger status semantics)
2. `src/loom/tui/app.py`
3. `tests/test_api.py`

### Workstream 5: Progress Backpressure and UI Responsiveness
1. Coalesce process progress updates to a bounded cadence (for example 4-10 Hz per run).
2. Prioritize terminal/cancel events over token chatter.
3. Skip expensive full-output refresh when payload contains no structural task/output changes.
4. Keep event panel logging for non-token milestones; heavily throttle token-derived UI noise.

Primary files:
1. `src/loom/tui/app.py`
2. `src/loom/tools/delegate_task.py`
3. `tests/test_tui.py`

### Workstream 6: Telemetry, Diagnostics, and Ops Guardrails
1. Add cancel telemetry fields:
2. `run_cancel_requested_at`, `run_cancel_ack_ms`, `run_cancel_path` (`api`, `worker_fallback`), `run_cancel_result`.
3. Add close-flow timeout counters and inflight-duration metrics.
4. Add structured debug logs for every cancel state transition.

Primary files:
1. `src/loom/tui/app.py`
2. `src/loom/tools/delegate_task.py`
3. `src/loom/engine/orchestrator.py`
4. `src/loom/config.py` (feature flags and timeout tunables)
5. `loom.toml.example`
6. `docs/CONFIG.md`
7. `CHANGELOG.md`

## Test Strategy

### Unit Tests
1. Running-tab `Ctrl+W` requests task cancellation and does not only call worker cancel when task ID is known.
2. Inflight close guard always releases on confirm, cancel, timeout, and exception.
3. Orchestrator cancellation propagation tests for single and parallel dispatch paths.
4. Delegate emits cancel lifecycle events and returns terminal cancellation result deterministically.
5. Backpressure/coalescing logic preserves terminal event delivery.

### Integration and TUI Tests
1. Simulate high-frequency progress stream, then `Ctrl+W`; verify cancel request visible and tab does not wedge.
2. Verify `Ctrl+W` on completed run still closes immediately.
3. Verify cancel-timeout branch offers force-close and remains interactive.
4. Verify `Ctrl+C` exit flow still works during active run cancellation path.

### Fault Injection Tests
1. Cancel API unavailable: fallback path executes and user gets explicit failure/next-step message.
2. Orchestrator mid-tool wait cancellation: task transitions to cancelled and stops further retries/replans.
3. Delegate event storm: UI remains responsive and close modal processes input.

## Rollout Plan
1. Phase A (flagged, internal):
2. Add `tui.run_cancel_v2_enabled = false` default off in code path with telemetry on both legacy and v2 paths.
3. Enable in dogfood environments only.
4. Phase B (canary default-on):
5. Set `run_cancel_v2_enabled = true` in internal defaults after 48h clean metrics.
6. Keep legacy fallback path behind toggle for one release.
7. Phase C (stabilize):
8. Remove legacy close-only cancellation behavior after acceptance criteria and SLOs hold for one release cycle.

Rollback:
1. Toggle `run_cancel_v2_enabled = false` to restore legacy behavior.
2. Keep cancellation telemetry intact for postmortem comparison.

## SLO and Acceptance Criteria
1. `Ctrl+W` while run is active shows cancellation acknowledgement in UI within 300 ms p95.
2. Task reaches terminal `cancelled` (or explicit timeout/failure message) within 10s p95 for cancellable runs.
3. No repeatable TUI wedge where `_close_process_tab_inflight` remains stuck after user action.
4. Orchestrator does not continue normal execution after cancellation signal in covered regression tests.
5. High-load cancel scenario remains keyboard-responsive (`Ctrl+W`, modal keys, `Ctrl+C`).
6. All existing `/run` and process-tab tests remain green, plus new cancel hardening tests.

## Risks and Mitigations
1. Risk: Changing exception handling alters failure semantics for non-cancel exceptions.
2. Mitigation: explicit tests for non-cancel exception paths plus targeted review of telemetry deltas.
3. Risk: Forced close may hide still-running backend work.
4. Mitigation: force-close must include explicit warning and follow-up action path (task ID and `/cancel` equivalent).
5. Risk: Backpressure drops useful progress context.
6. Mitigation: never drop terminal/cancel events; only coalesce repetitive non-terminal updates.
7. Risk: API cancel path introduces coupling in offline/no-server TUI mode.
8. Mitigation: graceful fallback to local worker cancel with explicit status labeling.

## Delivery Sequence (PR Slices)
1. PR1: Orchestrator cancellation correctness (`CancelledError` propagation) + tests.
2. PR2: TUI cancel-v2 state machine + bounded close-flow waits + per-run inflight hardening.
3. PR3: Delegate cancel bridge + cancel lifecycle logging.
4. PR4: Progress backpressure/coalescing + responsiveness tests.
5. PR5: Config/docs/changelog + rollout flags + telemetry polish.

## Definition of Done
1. Root-cause paths are fixed in code and protected by regression tests.
2. Cancel behavior is deterministic, observable, and user-actionable.
3. Production rollout has a safe toggle and documented rollback.
4. Measured cancel responsiveness meets SLO targets in internal canary runs.
