# Chat Steering Implementation Plan (2026-02-25)

## Objective
Add first-class steering for interactive cowork chat so users can influence an in-flight response without waiting for full completion.

Required behavior:
1. Queue-only path exists and is editable while a turn is running.
2. Steering can be applied immediately when requested.
3. Prefer native in-stream steering when the active model/provider supports it.
4. Automatically fall back to interrupt-and-steer when native in-stream steering is unavailable.

## Why This Plan
Current TUI behavior is single-flight for chat turns and does not expose steer controls:
1. User submissions are blocked while `_chat_busy` is true.
2. There is no built-in `/steer` slash command.
3. `CoworkSession.send_streaming(...)` has no control channel for live steer injections.
4. Current providers stream one request at a time (no active request mutation path in existing provider implementations).

This plan introduces a control plane that preserves current reliability while enabling richer steering semantics.

## Baseline (Repo-Accurate)
1. Input handling:
- `/Users/sfw/Development/loom/src/loom/tui/app.py` (`on_user_submit`) clears input, handles slash commands, then returns early when `_chat_busy` is true.
2. Slash command catalog:
- `/Users/sfw/Development/loom/src/loom/tui/app.py` `_SLASH_COMMANDS` has no `/steer` command today.
3. Turn execution:
- `_run_turn(...)` and `_run_interaction(...)` execute one streaming loop at a time with no external mutation hook.
4. Session/model interface:
- `/Users/sfw/Development/loom/src/loom/cowork/session.py` has `send_streaming(...)` but no queued steer queue or control ingress.
- `/Users/sfw/Development/loom/src/loom/models/base.py` has no provider capability/contract for live stream steering.

## Product Behavior Contract

### A) Queue and Edit (always available)
1. While a chat turn is active, new user submissions are captured as queued messages instead of being dropped/ignored.
2. Queue can be inspected and edited before dispatch.
3. Queue supports: list, edit, remove, clear, send-now.
4. Queued messages dispatch in deterministic order.

### B) Steer Now (user-requested)
1. User can issue explicit steer command while a turn is running.
2. Runtime chooses one execution mode:
- `native_live_steer`: inject into active stream when provider supports it.
- `interrupt_and_steer`: cancel active stream and immediately start a new turn with steer instruction.
3. TUI surfaces which mode was used for each steer request.

### C) Fallback Rules
1. If provider capability is unknown or false, use interrupt-and-steer.
2. If native steer attempt fails at runtime, degrade to interrupt-and-steer once and report degradation.
3. If interrupt fails cleanly, queue steer request and notify user that steer is pending.

## Architecture

### 1) TUI Steering State
Add explicit chat-control state in `LoomApp`:
1. `chat_turn_worker` reference for active turn cancellation.
2. `pending_chat_queue` for queued user messages.
3. `pending_steer_requests` with mutable payload + status.
4. lightweight queue metadata in session UI state (for restore safety and debug visibility).

### 2) Session Control Plane
Extend `CoworkSession` with a control interface:
1. Accept steering control events during `send_streaming(...)`.
2. Expose `supports_live_steer()` based on provider capabilities + runtime wiring.
3. Keep message persistence rules explicit for steer events (what gets persisted as user turn vs control annotation).

### 3) Provider Capability Contract
Extend model/provider capability schema:
1. Add capability flags for steer support (for example `live_steer`).
2. Add optional provider API for controlled streams (for example a stream object or control callback).
3. Keep default provider behavior backward-compatible (`live_steer = false`).

### 4) Strategy Router
Centralize steer decision logic in one method:
1. Input: active turn state, provider capability, user steer request.
2. Output: selected strategy (`native_live_steer` or `interrupt_and_steer`) + rationale.
3. Instrument chosen strategy in chat info/event panel for operator transparency.

## API/UX Surface Proposal

### Slash Commands
Add `/steer` command family in TUI:
1. `/steer <instruction>` (apply now; native if possible, fallback otherwise)
2. `/steer queue <instruction>`
3. `/steer list`
4. `/steer edit <id> <instruction>`
5. `/steer remove <id>`
6. `/steer clear`
7. `/steer send <id>`

### Non-Slash Input While Busy
1. Plain Enter during busy state enqueues as pending chat item.
2. User gets immediate info line: queued item id + next action hints (`/steer list`, `/steer edit`).

### Command Palette
Add palette entries for:
1. Queue message.
2. List queued items.
3. Send next queued item.
4. Apply steer now.

## Persistence and Ordering Semantics
1. Persist queued/steer control events separately from regular user turns (session metadata + optional conversation annotation).
2. Only committed dispatched messages should become canonical `role=user` turns.
3. Maintain strict ordering when multiple controls are issued quickly:
- Active strategy execution first.
- Then pending queue drain according to FIFO (unless user explicitly sends a selected item first).

## Native Steer Feasibility Notes
1. Existing OpenAI-compatible, Anthropic, and Ollama provider implementations are request-stream based and do not currently expose mutation of an in-flight request.
2. In this repo, initial implementation of capability detection will likely mark currently configured providers as `live_steer=false` unless a new provider mode supports it.
3. The architecture must still include the native path so future provider integrations can opt-in without TUI/session redesign.

## Workstreams

### Workstream 1: UX Contract + Command Surface
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`
2. `/Users/sfw/Development/loom/src/loom/tui/commands.py`

Deliverables:
1. `/steer` command parsing and usage docs in help text/autocomplete.
2. Busy-input enqueue behavior replacing silent drop behavior.
3. Clear chat feedback for queue/steer state transitions.

Acceptance:
1. While busy, user-entered text is never silently lost.
2. `/steer list/edit/remove/send` works deterministically.

### Workstream 2: Chat Queue + Editable Pending State
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`
2. `/Users/sfw/Development/loom/src/loom/cowork/session_state.py`

Deliverables:
1. In-memory queue model with stable ids.
2. Edit/remove/clear operations.
3. Queue drain orchestration after turn completion or explicit send.

Acceptance:
1. Queue operations are idempotent and covered by tests.
2. Queue order is preserved across rapid submissions.

### Workstream 3: Interrupt-and-Steer Fallback Path (Option 2)
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`
2. `/Users/sfw/Development/loom/src/loom/cowork/session.py`

Deliverables:
1. Active turn cancellation path tied to steer-now command.
2. Immediate restart with steer instruction in fresh turn.
3. Deterministic handling of partial streamed output and status-bar updates.

Acceptance:
1. `/steer <text>` during busy state interrupts current turn and starts new steer turn.
2. No stuck `_chat_busy` states after cancel/restart.
3. Tool-call execution does not continue after cancellation boundary.

### Workstream 4: Provider Capability + Native Steer Interface (Option 3)
Files:
1. `/Users/sfw/Development/loom/src/loom/config.py`
2. `/Users/sfw/Development/loom/src/loom/models/base.py`
3. `/Users/sfw/Development/loom/src/loom/models/openai_provider.py`
4. `/Users/sfw/Development/loom/src/loom/models/anthropic_provider.py`
5. `/Users/sfw/Development/loom/src/loom/models/ollama_provider.py`

Deliverables:
1. Capability flag(s) for live steer support.
2. Optional provider control interface for live steer.
3. Provider defaults set to unsupported unless explicitly implemented.

Acceptance:
1. Strategy router can query provider steer capability at runtime.
2. No regressions for existing provider streaming paths.

### Workstream 5: Session Control Plane + Strategy Router
Files:
1. `/Users/sfw/Development/loom/src/loom/cowork/session.py`
2. `/Users/sfw/Development/loom/src/loom/tui/app.py`

Deliverables:
1. Strategy selection logic (`native_live_steer` vs `interrupt_and_steer`).
2. Runtime degrade-on-failure behavior.
3. Structured logging/events for steer decisions and failures.

Acceptance:
1. Native path attempted only when capability is true.
2. Runtime native failure degrades once to interrupt path and reports reason.

### Workstream 6: Tests
Files:
1. `/Users/sfw/Development/loom/tests/test_tui.py`
2. `/Users/sfw/Development/loom/tests/test_streaming.py`
3. `/Users/sfw/Development/loom/tests/test_model_router.py`
4. `/Users/sfw/Development/loom/tests/test_config.py`

Deliverables:
1. Busy-input queue tests (no dropped text).
2. Queue edit/remove/send behavior tests.
3. Interrupt-and-steer lifecycle tests (cancel + restart + state reset).
4. Capability-gated strategy selection tests.
5. Native steer interface contract tests (skipped or fake-provider-backed where needed).

Acceptance:
1. Added tests fail on current behavior and pass with implementation.
2. Existing streaming and slash command tests remain green.

### Workstream 7: Docs and Tutorial Updates
Files:
1. `/Users/sfw/Development/loom/README.md`
2. `/Users/sfw/Development/loom/docs/tutorial.html`
3. `/Users/sfw/Development/loom/docs/agent-integration.md` (if API parity notes are added)

Deliverables:
1. New `/steer` command docs.
2. Explanation of strategy fallback semantics.
3. Known limitations per provider.

Acceptance:
1. User can discover and operate steer flow from docs alone.

## Rollout Plan

### Phase 1 (Ship first)
1. Queue/edit semantics.
2. Interrupt-and-steer fallback.
3. `/steer` command surface and tests.

### Phase 2
1. Capability interface and native steer plumbing.
2. Enable native steer for providers that can actually support in-flight mutation.

### Phase 3
1. Optional UI enhancements (queue panel/modal, richer steer history).
2. Operational telemetry for steer usage/fallback rates.

## Risks and Mitigations
1. Race conditions between cancel + restart:
- Mitigation: single serialized control lock for turn lifecycle transitions.
2. Partial stream persistence ambiguity:
- Mitigation: define canonical rule that only finalized turns create standard assistant turn separators; interrupted turns get explicit interrupted marker.
3. Provider mismatch:
- Mitigation: capability flag defaults false; native steer path opt-in only.
4. UX confusion between queued chat and steer-now:
- Mitigation: explicit command feedback and consistent status labels.

## Acceptance Checklist
1. Busy chat input is queued and editable.
2. `/steer <instruction>` works during active turn.
3. Native steer path is attempted when capability is present.
4. Automatic fallback to interrupt-and-steer works when native steer is unavailable/fails.
5. No regression in normal non-busy chat behavior.
6. Tests and docs reflect final behavior.

## Estimated Effort
1. Phase 1 (queue/edit + fallback + tests): 1.5 to 3 days.
2. Phase 2 (native steer contract + first provider support if available): 2 to 5 days depending on provider transport requirements.
3. Phase 3 (UX polish + telemetry): 0.5 to 1.5 days.
