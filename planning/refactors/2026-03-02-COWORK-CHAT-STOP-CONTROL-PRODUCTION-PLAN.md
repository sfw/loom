# Cowork Chat Stop Control Production Plan (2026-03-02)

## Objective
Ship a production-ready "Stop" control for in-flight cowork chat execution so users can reliably halt looping or long-running turns without quitting the app.

Required UX:
1. A `Stop` button appears on the same row as the chat input.
2. The button is visible only while cowork chat background work is active.
3. A `/stop` command provides equivalent behavior.
4. Stopping a turn never wedges the TUI and never requires app restart.

## Why This Needs Hardening
The basic "cancel the worker" concept is directionally correct but insufficiently specified for production. Without a hardened contract, we risk:
1. Stuck `_chat_busy` state after cancel races.
2. Inconsistent visibility/enabling of Stop when background activity changes rapidly.
3. Partial-turn persistence ambiguity (streamed text vs finalized turn rows).
4. Non-deterministic behavior across model streaming, tool execution, and delegate progress callbacks.
5. UX confusion between stopping cowork chat and cancelling `/run` process tabs.

## Baseline (Repo-Accurate)
1. Cowork turn execution:
   - `on_user_submit` blocks new non-slash submissions when `_chat_busy` is true.
   - `_run_turn` drives `_run_interaction`, which iterates `self._session.send_streaming(...)`.
   - File: `<repo-root>/src/loom/tui/app.py`
2. No stop surface today:
   - No `/stop` in `_SLASH_COMMANDS`.
   - No input-row Stop button.
3. Existing quit shortcut:
   - `Ctrl+C` triggers app quit flow (`request_quit`) and must remain unchanged.
4. Existing cancellation precedent:
   - `/run` has hardened cancel semantics (request/ack/wait/timeout/fallback).
   - File: `<repo-root>/src/loom/tui/app.py`
5. Existing steering plan:
   - Prior plan already called for active chat worker tracking and interrupt path.
   - File: `<repo-root>/planning/refactors/2026-02-25-CHAT-STEERING-IMPLEMENTATION-PLAN.md`

## Scope

### In Scope
1. Cowork chat turn stop control in TUI.
2. Input-row Stop button with conditional visibility.
3. `/stop` slash command.
4. Worker cancellation lifecycle + state reset + user feedback.
5. Replay/event handling for interrupted turns.
6. Telemetry and tests for reliability and responsiveness.

### Out of Scope
1. Native in-stream steering (`live_steer`) capabilities.
2. Queue/edit steering command family (`/steer ...`).
3. `/run` process tab cancellation redesign (already separately covered).

## Product Contract

### Behavior
1. If no cowork chat work is active, Stop is hidden.
2. If cowork chat work is active, Stop is shown and actionable.
3. Triggering Stop requests cancellation of the active cowork turn.
4. On successful interruption:
   - Input becomes usable immediately.
   - Status returns to `Ready`.
   - A clear info line is rendered (for example: "Stopped current chat execution.").
5. Triggering Stop while idle is a no-op with non-error informational feedback.

### Visibility Rule (Intentional)
Stop visibility is tied to cowork chat execution, not all app activity:
1. Visible when `_chat_busy` is true.
2. Optional extended visibility while unfinalized cowork delegate stream sections are active (if tied to current chat turn).
3. Not surfaced solely because `/run` tabs are active.

Rationale:
1. Prevents cross-domain confusion (cowork chat stop vs process-run cancel).
2. Keeps control semantics precise and predictable.

## Critique of Candidate Approaches

### Option A: Worker-Cancel Only
1. Pros:
   - Fastest to ship.
   - Minimal code movement.
2. Cons:
   - Best-effort only; some blocking work may not stop immediately.
   - No explicit cooperative check in `CoworkSession`.
   - Easier to leave ambiguous interrupted-turn artifacts.

### Option B: Cooperative Session Cancel Only
1. Pros:
   - Cleaner execution semantics.
   - Easier to reason about interruption boundaries.
2. Cons:
   - Larger surface area.
   - Does not guarantee interruption of all in-flight awaitables by itself.

### Option C: Hybrid (Recommended)
1. Request cooperative stop signal (session-level intent).
2. Apply worker cancellation as bounded fallback/fast-path.
3. Model after hardened `/run` cancel principles (ack, timeout, deterministic cleanup).

Recommendation:
1. Ship Hybrid (Option C) in the initial production change.
2. Hybrid means both paths are implemented at launch:
   - cooperative stop signal in cowork/session execution path
   - worker cancellation fallback when cooperative stop does not settle quickly
3. Do not ship a worker-only stop release.
4. Keep external UX stable while hardening internals.

## Architecture and State Machine

### New Chat Stop State
For cowork chat lane:
1. `idle`
2. `running`
3. `stop_requested`
4. `stopped` (transient terminal marker for one turn)

Transitions:
1. `idle -> running` on turn start.
2. `running -> stop_requested` when Stop invoked.
3. `stop_requested -> stopped` when cancellation observed.
4. `stopped -> idle` after final UI/state cleanup.

### Invariants
1. `_chat_busy` must be false on every terminal path (success, failure, stop).
2. Stop control visibility must match state on every transition.
3. At most one active cowork turn worker reference exists at a time.
4. Stop requests are idempotent while `stop_requested`.

## UX Design

### Input Row Layout
1. Add `Stop` button in the same horizontal row as `#user-input`.
2. Keep existing footer shortcut row unchanged.
3. Use a dedicated widget id (for example `#chat-stop-btn`) for styling and tests.

### Visibility/Enablement
1. Hidden when idle.
2. Visible + enabled while `running`.
3. Visible + disabled while `stop_requested` (optional label swap to `Stopping...`).

### Command Surface
1. Add `/stop` slash command.
2. Add usage/help text in slash help + command palette entry (optional but recommended).
3. `/stop` should work even during `_chat_busy` since slash commands are parsed before busy gate.

## Persistence and Replay Semantics
Interrupted turns must be represented cleanly in chat replay:
1. Persist a chat event indicating stop request and/or interruption completion.
2. Do not emit a normal final turn separator for a non-finalized interrupted turn.
3. Preserve already-streamed visible assistant text (do not retroactively erase).
4. Ensure hydrate/rerender path can replay interruption events without errors.

## Error Handling and Failure Modes
1. Stop pressed with no active worker:
   - Return non-fatal info message.
2. Worker cancellation raises:
   - Catch/log; still force UI state cleanup.
3. Tool callback arrives after stop:
   - Ignore or route safely without re-opening finalized sections.
4. Multiple rapid stop requests:
   - Coalesce/idempotent handling.
5. Cancellation timeout (hybrid bounded-settle path):
   - Surface explicit timeout status, invoke fallback behavior, and preserve input usability.

## Telemetry and Observability
Add structured events for cowork stop lifecycle:
1. `chat_stop_requested`
2. `chat_stop_ack`
3. `chat_stop_settled`
4. `chat_stop_timeout` (if timeout path introduced)
5. `chat_stop_failed`

Suggested fields:
1. `session_id`
2. `turn_number`
3. `ack_ms`
4. `settle_ms`
5. `stop_path` (`worker`, `cooperative`, `hybrid`)
6. `result` (`stopped`, `timeout`, `failed`)

SLO targets:
1. Stop acknowledgment visible in UI within 300 ms p95.
2. Turn settles to usable input state within 2s p95 for cancellable paths.

## Workstreams

### Workstream 1: UI and Command Surface
Files:
1. `<repo-root>/src/loom/tui/app.py`
2. `<repo-root>/src/loom/tui/commands.py`

Deliverables:
1. Input-row Stop button.
2. Conditional visibility state wiring.
3. `/stop` slash command + help text.

### Workstream 2: Turn Worker Lifecycle
Files:
1. `<repo-root>/src/loom/tui/app.py`
2. `<repo-root>/src/loom/cowork/session.py`

Deliverables:
1. Active chat worker reference tracking.
2. Stop action handler (idempotent).
3. Cooperative stop signal wiring into active cowork turn/session loop.
4. Deterministic finalization and `_chat_busy` reset across all paths.
5. Worker fallback cancellation path when cooperative cancellation does not settle in bounded time.

### Workstream 3: Replay/Event Semantics
Files:
1. `<repo-root>/src/loom/tui/app.py`

Deliverables:
1. Interrupted-turn replay event(s).
2. Safe hydrate rendering for interruption events.
3. No malformed turn separator behavior for interrupted turns.

### Workstream 4: Telemetry
Files:
1. `<repo-root>/src/loom/tui/app.py`
2. Optional config/docs files if new tunables are introduced.

Deliverables:
1. Structured stop lifecycle logs.
2. Basic latency instrumentation for ack/settle.

### Workstream 5: Tests
Files:
1. `<repo-root>/tests/test_tui.py`
2. Optional: `<repo-root>/tests/test_cowork.py` for session-level semantics.

Deliverables:
1. Stop button visibility toggles with cowork activity.
2. `/stop` while running requests cancellation.
3. `/stop` while idle is safe and non-fatal.
4. `_chat_busy` always resets after stop.
5. Next submission works immediately after stop.
6. `Ctrl+C` quit flow remains unchanged.
7. `/run` cancel behavior remains unaffected.

## Rollout Plan

### Phase A (Initial Ship)
1. Implement full Hybrid path (cooperative + worker fallback) with robust UI/state cleanup.
2. Add tests and telemetry proving both paths and fallback transitions.
3. Ship behind no flag if low risk, or gate with `tui.chat_stop_enabled` if preferred.

### Phase B (Hardening)
1. Tune bounded settle thresholds and improve fallback diagnostics.
2. Extend tests for timeout/fallback and late callback storms under load.
3. Refine user-facing messaging for rare timeout/failure branches.

### Rollback
1. Remove/hide Stop control and disable `/stop` path (or flip feature flag).
2. Keep telemetry events to support postmortem.

## Risks and Mitigations
1. Risk: Cancelled turn leaves inconsistent replay rows.
   - Mitigation: explicit interruption event type + hydrate tests.
2. Risk: Late async callbacks mutate stopped UI state.
   - Mitigation: finalized guards and worker identity checks.
3. Risk: User confuses Stop with `/run` cancellation.
   - Mitigation: scope-specific wording ("Stop chat turn"), keep `/run` semantics separate.
4. Risk: Non-cooperative tools continue briefly after cancel.
   - Mitigation: hybrid launch includes cooperative stop plus worker fallback, with clear user messaging and bounded settle behavior.

## Acceptance Checklist
1. Stop button is in the input row.
2. Stop button is hidden when cowork chat is idle.
3. Stop button becomes visible during in-flight cowork work.
4. `/stop` and button perform the same action.
5. App never wedges after stop; input is reusable immediately.
6. Existing `Ctrl+C` quit and `/run` cancel behavior are unchanged.
7. Replay/hydrate works with interrupted turns.
8. New tests and telemetry are in place.
9. Hybrid stop semantics are verified:
   - cooperative stop attempted first
   - worker fallback is invoked when cooperative stop does not settle in bound
   - final state is deterministic and input is reusable

## PR Slices
1. PR1: Input-row Stop UI + `/stop` command + visibility wiring.
2. PR2: Cooperative stop signal in cowork/session + stop state machine transitions.
3. PR3: Worker fallback cancellation + bounded settle + deterministic cleanup + replay interruption event.
4. PR4: Telemetry + test expansion + documentation/help polish.

## Definition of Done
1. Users can halt in-flight cowork chat loops without exiting Loom.
2. Stop behavior is deterministic, observable, and test-covered.
3. Interruption does not regress session integrity, replay, or next-turn usability.
4. Production telemetry confirms responsiveness and low failure rates.
