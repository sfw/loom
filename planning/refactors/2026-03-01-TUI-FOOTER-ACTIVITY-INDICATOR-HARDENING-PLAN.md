# TUI Footer Activity Indicator Hardening Plan (2026-03-01)

## Objective
Add a bottom-right animated activity indicator to the TUI footer that:
1. Preserves all existing progress/status text and interaction behavior.
2. Animates only while meaningful background work is active.
3. Falls back to dim static dots when idle.
4. Is production-safe (no layout jitter, no state flapping, deterministic tests).

## Problem Statement
Current feedback channels are strong (chat streaming, tool events, process progress), but there is no compact always-visible liveness affordance in the footer. Users can momentarily wonder whether work is still progressing when output is sparse.

## Scope
In scope:
1. Add a dedicated footer-right indicator widget (visual only).
2. Add a centralized active-state predicate in `LoomApp`.
3. Wire state updates at chat/process/delegate lifecycle points.
4. Add unit/integration tests for indicator behavior and active-signal correctness.

Out of scope:
1. Changes to existing chat/process progress messaging content.
2. Changes to tool/event semantics, orchestration, or model streaming behavior.
3. New telemetry backend requirements (optional local diagnostics only).

## Product Contract
1. Existing progress text remains unchanged.
2. Idle footer shows dim static dots (visible, no movement).
3. Active footer shows moving bright-dot animation.
4. Indicator motion must represent true background activity and stop promptly.
5. Feature must degrade safely if widget lookup/render fails (never block turn execution).

## Baseline (Repo-Accurate)
1. Status bar widget exists at `src/loom/tui/widgets/status_bar.py` but is hidden in CSS (`#status-bar { display: none; }`).
2. Footer keybind row is rendered by Textual `Footer` as `#app-footer` and currently remains unchanged.
3. Active chat state is tracked with `self._chat_busy`.
4. Active process runs are detectable through `self._has_active_process_runs()`.
5. Delegate progress state is tracked in `self._active_delegate_streams` with `finalized` flags.
6. There is prior heartbeat-dot logic for process launch activity text in `app.py`, but it is not footer UI.

## UX Specification (Locked)
1. Placement:
2. Indicator is right-aligned in the footer row.
3. It must not push/reflow existing footer keybind text unexpectedly.
4. Visual form:
5. 8-dot strip, fixed-width.
6. Idle state: all dots dim blue (`#4f6787`), static.
7. Active state: one bright cyan head dot (`#7dcfff`) moving across strip.
8. Optional smoothing neighbors in medium blue (`#5fa9d6`) only if readability is improved without blur.
9. Motion:
10. Ping-pong path (`0..7..0`) to avoid hard wrap jump.
11. Frame cadence target: 140 ms (acceptable range 120-180 ms).
12. Anti-flap hold: remain active for 300 ms after active signal drops.

## Architecture

### 1) New Widget: `ActivityIndicator`
File:
1. `src/loom/tui/widgets/activity_indicator.py` (new)

Responsibilities:
1. Own visual state (`active`, frame index, direction, hold-until timestamp).
2. Own animation timer lifecycle (start/stop safely).
3. Render deterministic Rich markup string for current frame.
4. Expose simple API: `set_active(bool)`.

Key constraints:
1. Fixed output width every frame.
2. No exceptions escaping render/timer callbacks.
3. Timer stopped on unmount to avoid leaks.

### 2) Footer Integration
Files:
1. `src/loom/tui/app.py`
2. `src/loom/tui/widgets/__init__.py`

Approach:
1. Add the indicator widget to the bottom-stack/footer area.
2. Keep existing `Footer(id="app-footer")` and shortcut labels unchanged.
3. Use container layout/CSS so indicator remains pinned at right edge and does not overlap keybind content.

### 3) Single Source of Truth for Activity
Add in `LoomApp`:
1. `_is_background_work_active() -> bool`
2. `_sync_activity_indicator() -> None`

Definition:
1. `True` when:
2. `_chat_busy` is true, or
3. `_has_active_process_runs()` is true, or
4. any `self._active_delegate_streams[...]` is not finalized.
5. Otherwise `False`.

Update triggers:
1. Start/end of `_run_turn`.
2. Process run status transitions that can activate/deactivate work.
3. Delegate stream start/finalize/update paths.
4. Mount/restore/session-load points where active runs may already exist.

## Design Critique and Hardening Decisions

### Critique 1: Risk of duplicate active logic across code paths
Issue:
1. Active checks can drift if done ad hoc in many handlers.

Decision:
1. Centralize in `_is_background_work_active()` and call `_sync_activity_indicator()` everywhere.
2. Ban direct indicator state mutation outside `_sync_activity_indicator()`.

### Critique 2: Footer layout fragility
Issue:
1. Footer row is currently a stock Textual footer; injecting widgets can break alignment.

Decision:
1. Preserve existing footer widget.
2. Place indicator in a sibling overlay/container anchored to bottom-right with fixed width.
3. Add visual regression assertions around footer key text presence.

### Critique 3: Flapping on micro-idle gaps
Issue:
1. During fast tool boundaries, active may rapidly toggle and look jittery.

Decision:
1. 300 ms hold-down debounce before switching to idle static dots.
2. Keep immediate transition to active (no delay) for responsiveness.

### Critique 4: Over-signaling background work
Issue:
1. If every transient callback marks active, indicator can stay active forever due to stale streams.

Decision:
1. Delegate stream state must honor `finalized` flag strictly.
2. Add stale-stream guard in sync logic for malformed entries (defensive handling).

### Critique 5: Render cost
Issue:
1. 140 ms timer updates should be lightweight.

Decision:
1. Keep render as short string ops only.
2. No markdown parsing or expensive object allocation per frame beyond tiny lists.

## Failure Modes and Mitigations
1. Widget lookup failure:
2. `_sync_activity_indicator()` catches lookup exceptions and no-ops.
3. Timer leak:
4. Stop timer on unmount and on transition to idle (post-hold).
5. Stuck active due to inconsistent state:
6. Recompute from source-of-truth predicate every sync call.
7. Layout collision with footer text:
8. Fixed-width indicator with right anchor; no dynamic width growth.
9. Test brittleness from animation timing:
10. Separate render-frame unit tests from timer-driven integration tests.

## Test Strategy

### Unit Tests (Widget)
Files:
1. `tests/test_tui.py` (extend existing `TestStatusBar`/widget section)

Cases:
1. Idle render shows 8 dim dots, fixed width.
2. Active render has bright head and frame advances deterministically.
3. Ping-pong index sequence is correct.
4. `set_active(False)` returns to static dim dots after hold policy.
5. Timer stop/start is idempotent.

### App Logic Tests
Cases:
1. `_is_background_work_active()` truth table:
2. chat busy only => true.
3. active process only => true.
4. unfinalized delegate only => true.
5. all false => false.
6. `_sync_activity_indicator()` updates widget state without exceptions when widget absent.

### Regression Tests
Cases:
1. Existing footer keybind labels still render.
2. Existing progress text behavior remains unchanged in key turn/process tests.
3. No regressions to `/run` process progress rows and chat streaming flow.

## Implementation Workstreams

### Workstream A: Widget Primitive
Files:
1. `src/loom/tui/widgets/activity_indicator.py`
2. `src/loom/tui/widgets/__init__.py`

Deliverables:
1. New widget class with idle/active render contract.
2. Timer-based ping-pong animation.
3. Safe `set_active()` API with hold-down debounce.

Acceptance:
1. Deterministic frame rendering.
2. No timer leaks on mount/unmount.

### Workstream B: App Wiring
File:
1. `src/loom/tui/app.py`

Deliverables:
1. Footer-right mounting and CSS placement.
2. `_is_background_work_active()` and `_sync_activity_indicator()`.
3. Sync hook calls at chat/process/delegate lifecycle boundaries.

Acceptance:
1. Existing footer/help text unchanged.
2. Indicator reflects actual background activity.

### Workstream C: Test Hardening
File:
1. `tests/test_tui.py`

Deliverables:
1. Widget render tests.
2. App truth-table/sync tests.
3. Regression tests for footer/progress stability.

Acceptance:
1. New tests pass consistently.
2. No expectation drift in unrelated TUI tests.

## Rollout and Verification Gates
1. Gate 1: Widget-only tests green.
2. Gate 2: App wiring + targeted TUI tests green.
3. Gate 3: Full `tests/test_tui.py` green locally.
4. Gate 4: Manual smoke in TUI:
5. idle screen shows dim static dots.
6. submit chat prompt => animation starts immediately.
7. wait until complete => returns to dim static dots.
8. `/run` long task => continuous animation while active.

## Rollback Plan
1. If footer integration causes layout regressions:
2. Disable mount of the indicator widget but retain isolated class/tests.
3. If state wiring causes instability:
4. Stub `_sync_activity_indicator()` to no-op and keep feature flag path.
5. Keep rollback as a single patch touching only TUI widget wiring surfaces.

## Acceptance Criteria (Release Gate)
1. Idle dots are visible and static.
2. Dots animate only during true background activity.
3. Existing progress/status text is unchanged.
4. Footer remains readable and stable at common terminal widths.
5. Tests cover render logic, state predicate, and regression stability.

## Implementation Checklist (Ready-to-Execute)
1. Add `ActivityIndicator` widget file and exports.
2. Add app compose/CSS placement in footer area.
3. Add active predicate + sync function.
4. Add sync hooks at chat/process/delegate transitions.
5. Add/adjust tests in `tests/test_tui.py`.
6. Run targeted tests and manual smoke checks.
7. Land with concise changelog note if user-facing TUI behavior docs are maintained.
