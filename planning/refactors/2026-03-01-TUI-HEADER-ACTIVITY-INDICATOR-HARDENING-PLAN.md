# TUI Header Activity Indicator Hardening Plan (2026-03-01)

## Objective
Add a top-bar activity indicator that appears immediately left of the clock and:
1. Keeps all existing progress/status text and behavior intact.
2. Animates only while meaningful background work is active.
3. Shows dim static dots when idle.
4. Is production-safe (stable layout, anti-flap behavior, deterministic tests).

## Problem Statement
Loom already provides rich progress surfaces (chat streaming, tool rows, process progress), but users still benefit from a compact always-visible liveness cue near the current time. The footer has become busier (shortcuts/hotkeys), so the indicator should move to the header clock cluster.

## Scope
In scope:
1. Add a dedicated activity indicator widget.
2. Place it in the top header, directly left of the clock.
3. Add one centralized background-activity predicate in `LoomApp`.
4. Wire lifecycle sync points for chat/process/delegate work.
5. Add unit/integration/regression tests.

Out of scope:
1. Changes to existing progress text content.
2. Changes to orchestration/tool semantics.
3. Footer layout redesign (other than preserving current behavior).

## Product Contract
1. Existing progress text and event surfaces remain unchanged.
2. Indicator location is fixed: header right cluster, immediately left of time.
3. Idle state is visible dim static dots (no motion).
4. Active state is animated moving dots.
5. Footer keybind/shortcut row remains unchanged.
6. Any indicator failure must degrade safely (no runtime interruption).

## Baseline (Repo-Accurate)
1. TUI currently renders `Header(show_clock=True)` at the top in `src/loom/tui/app.py`.
2. Footer row currently includes `Footer(id="app-footer")` plus auth/MCP shortcut buttons.
3. Active chat work is tracked with `self._chat_busy`.
4. Process activity is observable via `self._has_active_process_runs()`.
5. Delegate activity is tracked in `self._active_delegate_streams` with `finalized` state.
6. Existing status bar widget is hidden (`#status-bar { display: none; }`) and not the target surface.

## UX Specification (Locked)
1. Placement:
2. Indicator is rendered in header right cluster directly left of the clock text.
3. It must keep stable spacing and never overlap time/title.
4. Visual form:
5. 8-dot strip, fixed width.
6. Idle: all dots dim blue `#4f6787`.
7. Active: one bright cyan head dot `#7dcfff`.
8. Optional smoothing neighbors in medium blue `#5fa9d6` only if readability improves.
9. Motion:
10. Ping-pong sequence (`0..7..0`) to avoid jumpy wrap.
11. Frame interval target 140 ms (acceptable 120-180 ms).
12. Anti-flap hold: remain active for 300 ms after active signal drops.

## Architecture

### 1) Indicator Primitive
File:
1. `src/loom/tui/widgets/activity_indicator.py` (new)

Responsibilities:
1. Track `active` state and frame progression.
2. Own timer lifecycle (start/stop safely).
3. Render deterministic fixed-width Rich markup.
4. Expose `set_active(bool)` API.

Constraints:
1. No exceptions escape timer/render paths.
2. Timer always stops on unmount.
3. Render cost remains trivial (string/list operations only).

### 2) Header Integration (Clock-Adjacent)
Files:
1. `src/loom/tui/app.py`
2. `src/loom/tui/widgets/__init__.py`
3. Optional: `src/loom/tui/widgets/header_row.py` (if custom header composition is needed)

Integration contract:
1. Preserve visible header behavior and clock updates.
2. Place indicator immediately left of clock with deterministic order.
3. Avoid brittle dependence on private Textual internals.

Recommended strategy:
1. Keep top bar as a dedicated, explicit composition surface controlled by Loom.
2. If direct insertion into stock `Header` proves unstable, introduce a minimal Loom-owned header wrapper containing:
3. left: title
4. right: `[ActivityIndicator][Clock]`
5. Maintain existing visual styling close to current header.

### 3) Single Source of Truth for Activity
Add in `LoomApp`:
1. `_is_background_work_active() -> bool`
2. `_sync_activity_indicator() -> None`

Definition:
1. Active when any of:
2. `_chat_busy`
3. `_has_active_process_runs()`
4. any delegate stream not finalized in `_active_delegate_streams`
5. Otherwise inactive.

Update triggers:
1. start/end of `_run_turn`
2. process run status transitions
3. delegate stream start/finalize paths
4. mount/restore paths where active work can preexist

Rule:
1. Only `_sync_activity_indicator()` may mutate indicator state.

## Design Critique and Hardening Decisions

### Critique 1: Header internals can be fragile across Textual versions
Issue:
1. Injecting widgets into stock `Header` internals can break with upstream changes.

Decision:
1. Favor Loom-controlled composition for the right-side header cluster if needed.
2. Keep a minimal adaptation layer to isolate framework churn.

### Critique 2: Clock adjacency must be exact
Issue:
1. Right-docked overlays can drift and violate “immediately left of time.”

Decision:
1. Enforce explicit sibling order in a shared right-side container.
2. Add tests for adjacency order and stable spacing.

### Critique 3: Signal flapping at tool boundaries
Issue:
1. Rapid state transitions can cause visible jitter.

Decision:
1. Immediate activate.
2. Delayed idle (300 ms hold).

### Critique 4: Stale delegate streams can keep indicator active forever
Issue:
1. Malformed/stale entries may falsely signal in-flight work.

Decision:
1. Strictly honor `finalized` semantics.
2. Defensive filtering of malformed stream payloads in predicate logic.

### Critique 5: Regressing footer interactions
Issue:
1. Prior plan targeted footer; current app has active footer shortcuts.

Decision:
1. Do not touch footer interaction behavior.
2. Add explicit regression tests for footer shortcut presence/actions.

## Failure Modes and Mitigations
1. Header mount/query failure:
2. `_sync_activity_indicator()` no-ops safely on missing widget.
3. Timer leak:
4. Stop timer on unmount and post-hold idle transition.
5. Clock desync/regression:
6. Add tests ensuring clock still renders/updates with indicator present.
7. Layout overlap:
8. Fixed-width indicator, explicit spacing, and width constraints.
9. Persistent false-active:
10. Centralized predicate recomputation and stream-finalization checks.

## Test Strategy

### Unit Tests (Indicator)
File:
1. `tests/test_tui.py`

Cases:
1. Idle render is 8 dim dots and fixed width.
2. Active frame progression is deterministic.
3. Ping-pong index path is correct.
4. `set_active(False)` returns to static dim dots after hold window.
5. Timer start/stop is idempotent and cleanup-safe.

### App Logic Tests
Cases:
1. `_is_background_work_active()` truth table:
2. chat-only active => true
3. process-only active => true
4. delegate-only active => true
5. none active => false
6. malformed delegate entries do not break predicate.
7. `_sync_activity_indicator()` tolerates missing indicator widget.

### Header Integration Tests
Cases:
1. Indicator appears to the left of clock in header right cluster.
2. Clock remains visible and updates.
3. Header title remains readable at common terminal widths.

### Regression Tests
Cases:
1. Footer keybind/shortcut UI remains intact.
2. Existing chat/process progress text behavior remains unchanged.
3. `/run` progress and delegate-stream UX remain unchanged.

## Implementation Workstreams

### Workstream A: Widget Primitive
Files:
1. `src/loom/tui/widgets/activity_indicator.py`
2. `src/loom/tui/widgets/__init__.py`

Deliverables:
1. Activity indicator widget with idle/active contract.
2. Ping-pong timer animation + anti-flap hold.
3. Safe lifecycle cleanup.

Acceptance:
1. Deterministic render logic.
2. No timer leaks.

### Workstream B: Header Placement
Files:
1. `src/loom/tui/app.py`
2. Optional: `src/loom/tui/widgets/header_row.py`

Deliverables:
1. Header-right cluster with explicit `[indicator][clock]` order.
2. Stable spacing and style alignment with existing top bar.
3. No footer behavior regressions.

Acceptance:
1. Indicator is always immediately left of time.
2. Clock remains accurate and visible.

### Workstream C: App-State Wiring
File:
1. `src/loom/tui/app.py`

Deliverables:
1. `_is_background_work_active()` + `_sync_activity_indicator()`.
2. Sync hooks at chat/process/delegate lifecycle points.

Acceptance:
1. Indicator state reflects true background work.
2. No direct state mutation outside sync function.

### Workstream D: Test Hardening
File:
1. `tests/test_tui.py`

Deliverables:
1. Indicator unit tests.
2. Header placement and clock regression tests.
3. Footer/progress behavior regression tests.

Acceptance:
1. New tests are deterministic.
2. Existing unrelated tests do not drift.

## Rollout and Verification Gates
1. Gate 1: Indicator unit tests pass.
2. Gate 2: Header placement tests pass.
3. Gate 3: Targeted TUI suite passes.
4. Gate 4: Manual smoke:
5. idle => dim static dots beside time
6. active chat/run => animation starts immediately
7. completion => returns to dim dots after hold
8. footer shortcuts remain functional

## Rollback Plan
1. If header integration regresses clock/title:
2. disable header indicator mount while retaining widget/tests.
3. If state wiring is unstable:
4. stub `_sync_activity_indicator()` to no-op.
5. Keep rollback limited to header indicator surfaces.

## Acceptance Criteria (Release Gate)
1. Indicator renders left of time in header.
2. Idle state is dim static dots.
3. Animation appears only during true background activity.
4. Footer row behavior is unchanged.
5. Progress/status text remains unchanged.
6. Tests cover rendering, placement, predicate logic, and regressions.

## Implementation Checklist (Ready-to-Execute)
1. Add indicator widget file and export.
2. Implement header placement beside clock.
3. Add centralized active predicate + sync function.
4. Wire sync hooks in chat/process/delegate lifecycle.
5. Add/adjust tests for indicator + header + regressions.
6. Run targeted and full TUI test passes.
