# Cowork Chat Steering Hardened Production Plan (2026-03-02)

## Objective
Ship a production-ready steering system for cowork chat that allows users to safely change direction mid-execution without restarting the app or losing useful context.

This plan assumes the shipped hybrid Stop control is available and becomes a steering primitive.

## Finalized Product Decisions
1. Scope is cowork chat only (for now).
2. Steering primitives are all-in for this implementation:
   - `pause`
   - `resume`
   - `inject instruction`
   - `redirect goal`
   - `stop`
3. Input-row controls while cowork is busy:
   - `Stop` always visible when busy.
   - `Inject` and `Redirect` visible only when input has non-empty text.
4. Apply modes:
   - `Redirect` is immediate.
   - `Inject` applies at next safe boundary.
5. Input behavior:
   - both `Inject` and `Redirect` clear input after click.
6. Inject queue model:
   - one pending slot only (latest wins).
   - queued inject is editable/cancellable.
7. Steering queue popup:
   - replaces slash hints while active.
   - supports `Edit`, `Redirect Now`, `Dismiss`.
8. Redirect confirmation:
   - required only when a mutating tool is in-flight.
   - mutating detection reuses existing `_is_mutating_tool` policy.
9. Redirect context semantics:
   - retain prior evidence/tool outputs.
   - rebase tasks only (no full context reset).
10. Command surface:
   - `/pause`
   - `/inject <text>`
   - `/redirect <text>`
   - `/steer pause`
   - `/steer resume`
   - `/steer queue`
   - `/steer clear`
11. Queue persistence:
   - pending inject is ephemeral (not persisted across app restart/session switch).
12. No delayed-warning requirement for queued inject age.

## Why This Shape
1. Keeps user control strong without requiring a new execution engine.
2. Avoids command collisions (`/resume` remains session-resume, steering resume goes under `/steer resume`).
3. Makes redirect deterministic and audit-friendly while minimizing data loss.
4. Keeps inject low-risk by applying only at safe boundaries.

## Non-Goals
1. Steering for `/run` process tabs and `/<process-name>` flows (future extension).
2. Multi-item steering queues.
3. Mid-mutation force editing without confirmation.
4. Replay rewrites of already-emitted assistant content.

## UX Contract

### Input Row Behavior
When `cowork_chat_busy == false`:
1. Hide `Inject`, `Redirect`, `Stop` buttons.

When `cowork_chat_busy == true` and input is empty:
1. Show `Stop` only.
2. Hide `Inject` and `Redirect`.

When `cowork_chat_busy == true` and input has text:
1. Show `Inject`, `Redirect`, `Stop`.
2. `Inject` click queues text and clears input.
3. `Redirect` click starts immediate redirect flow and clears input.

### Queue Popup (Replaces Slash Hints)
Display queue popup while all are true:
1. cowork chat is busy,
2. pending inject exists.

Popup content:
1. Type: `Inject`
2. Prompt preview (truncated with full text retained in state)
3. Queue age / status

Popup actions:
1. `Edit`
   - remove pending inject,
   - restore full text to input,
   - focus input.
2. `Redirect Now`
   - uses queued text,
   - converts to immediate redirect,
   - runs redirect interrupt path.
3. `Dismiss`
   - clears pending inject.

### Transcript / User Feedback
All steering actions surface clear info lines and replay events:
1. queued
2. replaced
3. cleared/dismissed
4. applied
5. rejected/failed

## Command Contract

### New Commands
1. `/pause`
   - pauses further cowork turn progression at next cooperative checkpoint.
2. `/inject <text>`
   - queue instruction for next safe boundary.
3. `/redirect <text>`
   - immediate objective redirect (with confirmation gate if mutating tool is active).
4. `/steer pause`
   - alias for pause.
5. `/steer resume`
   - resume after pause.
6. `/steer queue`
   - show queued steering state.
7. `/steer clear`
   - clear pending inject and pause intent (if present).

### Existing Commands Unchanged
1. `/resume <session-id-prefix>` remains session-resume command.
2. `/stop` remains immediate stop control.

## Steering State Model
Add a cowork steering controller state in TUI lane (ephemeral):
1. `paused: bool`
2. `pending_inject: SteeringDirective | None`
3. `active_redirect: SteeringDirective | None`
4. `last_applied_directive_id: str`
5. `last_error: str`

Directive payload:
1. `id`
2. `kind` (`inject` | `redirect`)
3. `text`
4. `source` (`button` | `slash` | `queue_popup`)
5. `created_at`
6. `status` (`queued` | `applying` | `applied` | `dismissed` | `failed`)

Inject queue policy:
1. one slot only,
2. latest wins replacement,
3. replacement emits `replaced` event.

## Execution Semantics

### Safe Boundary Definition (Inject Apply)
Inject can apply only at cooperative checkpoints:
1. before next model request,
2. after model response iteration,
3. before tool dispatch,
4. after tool completion,
5. after stream chunk batch flush checkpoints already used for cooperative stop checks.

Inject never interrupts active mutating tool execution.

### Redirect Immediate Flow
1. Validate text.
2. Detect whether an in-flight tool is mutating via existing `_is_mutating_tool` policy.
3. If mutating in-flight:
   - open confirmation modal,
   - continue only on confirm.
4. Trigger hybrid interruption path (cooperative first, worker fallback).
5. On settle:
   - create new objective directive message,
   - retain prior evidence/tool outputs,
   - rebase task graph:
     - completed stays completed,
     - incompatible pending/in-progress becomes obsolete or reset,
     - generate next tasks for redirected goal.
6. Resume execution on new objective.

### Pause / Resume
1. `pause` sets paused flag and blocks further advancement at cooperative checkpoints.
2. active blocking tool call is allowed to complete naturally unless stopped/redirected.
3. `resume` clears paused flag and allows loop continuation.

## Persistence and Replay

### Persisted
Persist steering event rows to replay journal:
1. `steer_inject_queued`
2. `steer_inject_replaced`
3. `steer_inject_dismissed`
4. `steer_inject_applied`
5. `steer_redirect_requested`
6. `steer_redirect_confirmed`
7. `steer_redirect_applied`
8. `steer_pause_requested`
9. `steer_resumed`
10. `steer_failed`

### Not Persisted
1. In-memory pending inject slot state itself (ephemeral).
2. Transient popup open/closed state.

Hydration requirements:
1. steering events replay without errors,
2. no malformed separator emission,
3. historical steer actions remain audit-visible.

## Error Handling
1. `/inject` with empty text: return usage guidance.
2. `/redirect` with empty text: return usage guidance.
3. redirect confirm denied: emit `redirect_cancelled` info/event.
4. pause while already paused: idempotent ack.
5. resume while not paused: safe no-op ack.
6. clear with no queue: safe no-op ack.
7. failures in steering apply path:
   - keep app usable,
   - emit error info + telemetry,
   - clear broken pending state.

## Telemetry and Observability
Add structured latency/status events:
1. `steer_inject_queued`
2. `steer_inject_applied`
3. `steer_inject_replaced`
4. `steer_inject_dismissed`
5. `steer_redirect_requested`
6. `steer_redirect_confirm_required`
7. `steer_redirect_confirmed`
8. `steer_redirect_applied`
9. `steer_pause_requested`
10. `steer_resumed`
11. `steer_failed`

Suggested fields:
1. `session_id`
2. `turn_number`
3. `directive_id`
4. `directive_kind`
5. `source`
6. `apply_mode`
7. `queued_ms`
8. `interrupt_path` (`cooperative` | `worker_fallback` | `hybrid`)
9. `result` (`queued` | `applied` | `dismissed` | `failed`)

## Workstreams

### Workstream 1: Command Surface
Files:
1. `<repo-root>/src/loom/tui/app.py`
2. `<repo-root>/src/loom/tui/commands.py`

Deliverables:
1. Add `/pause`, `/inject`, `/redirect`, `/steer ...` handlers.
2. Help/completion updates.
3. Command palette additions for steering actions.

### Workstream 2: Input Row Controls
Files:
1. `<repo-root>/src/loom/tui/app.py`

Deliverables:
1. Add `Inject` and `Redirect` buttons in input row.
2. Visibility matrix wired to busy + input text state.
3. Input clear/focus behavior after action.

### Workstream 3: Steering State + Queue Popup
Files:
1. `<repo-root>/src/loom/tui/app.py`

Deliverables:
1. Ephemeral steering state model.
2. Queue popup renderer replacing slash hints while active.
3. Popup actions (`Edit`, `Redirect Now`, `Dismiss`).

### Workstream 4: Cowork Execution Integration
Files:
1. `<repo-root>/src/loom/tui/app.py`
2. `<repo-root>/src/loom/cowork/session.py`

Deliverables:
1. Safe-boundary inject application.
2. Pause/resume gating.
3. Redirect immediate interrupt/rebase flow.
4. Mutating confirmation using `_is_mutating_tool`.

### Workstream 5: Replay + Telemetry
Files:
1. `<repo-root>/src/loom/tui/app.py`

Deliverables:
1. Steering replay events and hydration support.
2. Structured telemetry for queue/apply/failure/confirm flows.

### Workstream 6: Tests
Files:
1. `<repo-root>/tests/test_tui.py`
2. `<repo-root>/tests/test_cowork.py`

Deliverables:
1. Button visibility matrix tests (busy/input combinations).
2. `/inject` queue + latest-wins replacement tests.
3. queue popup replace-slash-hints tests.
4. popup action tests (`Edit`, `Redirect Now`, `Dismiss`).
5. redirect confirm gate tests for mutating vs non-mutating in-flight tools.
6. inject apply-at-boundary tests.
7. pause/resume idempotency tests.
8. replay/hydration tests for steering event types.
9. regression tests: `/resume` session command unaffected, `/run` path unaffected.

## Risk Register
1. Risk: steering state races with stop state.
   - Mitigation: single authoritative steering state machine + idempotent transitions.
2. Risk: redirect while tool runs causes inconsistent outputs.
   - Mitigation: mutation confirmation gate + deterministic interrupt flow.
3. Risk: queue popup conflicts with slash-hint UX.
   - Mitigation: explicit replacement precedence + clear fallback to slash hints when queue clears.
4. Risk: command ambiguity around `/resume`.
   - Mitigation: reserve cowork flow resume under `/steer resume` only.

## Acceptance Criteria
1. Users can inject and redirect without app restart.
2. Redirect is immediate and deterministic.
3. Inject is queued and applied at safe boundary only.
4. Queue popup replaces slash hints while pending inject exists.
5. Popup actions behave exactly as specified.
6. Confirmation is shown only for in-flight mutating tools.
7. `/resume` session command remains intact.
8. `/run` and `/<process-name>` behavior unchanged.
9. Steering actions are replayable/auditable.
10. Lint and full test suite pass.

## Future Extension Note
When extending to `/run` and `/<process-name>` flows later:
1. keep cowork steering contracts unchanged,
2. introduce separate process-run steering adapter,
3. avoid mixing control planes until shared semantics are proven.
