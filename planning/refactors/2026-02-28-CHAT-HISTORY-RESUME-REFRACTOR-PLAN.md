# Chat History Resume Refactor Plan (2026-02-28)

## Objective
Ship production-grade chat history replay in the TUI so resumed/switching cowork sessions render prior conversation in the chat panel with correct ordering, bounded latency, and no cross-session bleed.

## Why This Plan
Session persistence already exists, but chat-panel hydration is incomplete:
1. Session resume restores model context and metadata, not rendered chat transcript.
2. Session switch/new currently does not reset and rehydrate chat deterministically.
3. Long-session replay requires pagination and rendering limits to avoid UI degradation.
4. Current tests cover input-history/session metadata restore, not transcript replay guarantees.

## Current Baseline (Repo-Accurate)
1. Conversation turns are persisted append-only in `conversation_turns` with `role`, `content`, `tool_calls`, `tool_call_id`, `tool_name`, `created_at`.
2. `CoworkSession.resume(...)` restores only recent messages for model context (`resume_session(..., recent_limit=100)`), then normalizes tool payloads for token efficiency.
3. TUI startup resume and `/resume` switch paths call `session.resume(...)`, but do not re-render prior chat rows.
4. Chat widget APIs are present for all row types used during live turns (`add_user_message`, `add_model_text`, `add_tool_call`, `add_content_indicator`, `add_turn_separator`, `add_info`).
5. `/clear` currently clears only visible chat widgets and does not affect stored history.

## Problem Statement
Chat history is available in storage but not projected into chat UI state on resume. The system persists conversation semantics for model continuity, but the user-facing transcript view is not restored.

## Production Requirements
1. Correctness:
2. Resume startup and `/resume` must show transcript for the target session, in chronological order.
3. No mixed transcript rows across sessions after `/new` or `/resume`.
4. Tool calls and tool results remain paired and readable even with partial/malformed historical chains.
5. Performance:
6. Initial transcript hydrate p95 < 800ms for 300 rows on typical dev hardware.
7. Incremental "load older" page p95 < 350ms for 200 rows.
8. UI remains responsive during replay of very long sessions (>10k rows) via paging and caps.
9. Reliability:
10. Replay errors are non-fatal per-row; bad rows are skipped with diagnostics.
11. Legacy sessions (no new transcript journal) still replay via deterministic fallback.
12. Operability:
13. Emit diagnostics for hydrate source, row count, parse failures, and elapsed ms.

## Non-Goals
1. Replacing `conversation_turns` as model-context source of truth.
2. Replaying every historical visual detail exactly for pre-refactor sessions.
3. Web UI/API transcript hydration in this workstream.
4. Mutating/deleting historical chat records.

## Architecture Decision
Adopt a two-source transcript system:
1. Primary source (new): append-only cowork chat event journal for exact UI replay semantics.
2. Compatibility source (existing): deterministic synthesis from `conversation_turns` for legacy sessions and failure fallback.

Rationale:
1. Existing `conversation_turns` is sufficient for semantic replay but lossy for full UI fidelity.
2. Journalized chat events eliminate inference complexity for new sessions.
3. Legacy synthesis guarantees backward compatibility without mandatory offline migration.

## Proposed Design

### 1) Add Cowork Chat Event Journal
Create a new append-only table for UI transcript events:
1. `cowork_chat_events`
2. Columns:
3. `id INTEGER PRIMARY KEY AUTOINCREMENT`
4. `session_id TEXT NOT NULL`
5. `seq INTEGER NOT NULL`
6. `event_type TEXT NOT NULL`
7. `payload TEXT NOT NULL` (JSON, versioned)
8. `created_at TEXT NOT NULL DEFAULT (datetime('now'))`
9. Indexes:
10. Unique `(session_id, seq)`
11. `(session_id, created_at)`
12. `(session_id, id)`

Event types (v1):
1. `user_message`
2. `assistant_text`
3. `tool_call_started`
4. `tool_call_completed`
5. `content_indicator`
6. `turn_separator`
7. `info`

### 2) Store API Extensions
Extend `ConversationStore`:
1. `append_chat_event(session_id, event_type, payload, seq=None)`
2. `get_chat_events(session_id, before_seq=None, after_seq=None, limit=...)`
3. `get_last_chat_seq(session_id)`
4. `synthesize_chat_events_from_turns(session_id, before_turn=None, limit=...)` (legacy adapter helper)

Design constraints:
1. Enforce max page size (for example 500).
2. JSON parse failures return structured fallback rows, never raise to UI caller.
3. Sequence allocation must be serialized per session (single DB transaction).

### 3) Runtime Event Emission (TUI + Cowork Loop Integration)
Persist chat events when UI-visible rows are produced:
1. User submit -> `user_message`.
2. Final assistant segment per model response (no token-delta persistence) -> `assistant_text`.
3. Tool start/completion -> `tool_call_started` and `tool_call_completed`.
4. Multimodal indicators -> `content_indicator`.
5. Turn footer -> `turn_separator`.
6. Session/system notices relevant to transcript continuity -> `info`.

Hardening rules:
1. Do not persist every streaming token chunk.
2. Persist only completed rows to avoid replaying transient partials.
3. If event persistence fails, continue interaction and record warning diagnostics.

### 4) Transcript Hydration Engine in TUI
Add a dedicated rehydrate path for startup resume and `/resume` switch:
1. Clear visible chat rows on session transitions (`/new`, `/resume`, startup auto-resume target change).
2. Load latest event page from `cowork_chat_events`.
3. If journal rows are absent, synthesize rows from `conversation_turns`.
4. Render rows through existing `ChatLog` methods.
5. Track cursor (`oldest_seq_loaded` or `oldest_turn_loaded`) for optional older-page loading.

Paging policy:
1. Initial load: latest 200-300 rows (configurable).
2. Older history: explicit command (`/history older`) and/or top-scroll trigger.
3. Render cap: keep only newest N rendered rows (for example 1200), trimming oldest with one info sentinel row.

### 5) Legacy Synthesis Strategy (No Journal)
Reconstruct display rows from `conversation_turns` deterministically:
1. `user` -> `add_user_message`.
2. `assistant` with no `tool_calls` -> `add_model_text`.
3. `assistant` with `tool_calls` -> emit `tool_call_started` rows for each call.
4. `tool` -> parse `ToolResult.from_json(...)`, emit completed tool row, then optional content indicator.
5. `system` rows:
6. Skip internal system hints by default.
7. Keep user-facing system info only when explicitly tagged/known-safe.

Failure behavior:
1. Unknown/malformed tool payload -> render error-safe tool row with raw summary.
2. Orphan tool rows -> render as completed tool with `tool_name` fallback if available.

### 6) Session Transition Semantics
Standardize behavior:
1. Startup resume: hydrate transcript for resumed session before startup summary banner.
2. `/resume`: clear chat and hydrate target transcript before "Switched Session" info.
3. `/new`: clear chat (new transcript baseline) and show "New Session Created".
4. Block `/new` and `/resume` while `_chat_busy` to avoid mixed turn/session state.

## Workstreams and File Touchpoints

### W1: Data Layer and Schema
Files:
1. `/Users/sfw/Development/loom/src/loom/state/schema.sql`
2. `/Users/sfw/Development/loom/src/loom/state/conversation_store.py`
3. `/Users/sfw/Development/loom/tests/test_conversation_store.py`

Deliverables:
1. `cowork_chat_events` schema.
2. Store CRUD/paging for chat events.
3. Sequence and bounds validation.
4. Legacy synthesis helper contract tests.

### W2: Cowork/TUI Emission Pipeline
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`
2. `/Users/sfw/Development/loom/src/loom/cowork/session.py` (only if event metadata plumbing is needed)
3. `/Users/sfw/Development/loom/tests/test_tui.py`

Deliverables:
1. Event journal writes at each UI-visible row boundary.
2. Session transition guards (`_chat_busy` checks for session-changing commands).
3. Transition-safe chat clearing policy.

### W3: Hydration + Pagination + Rendering Caps
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`
2. `/Users/sfw/Development/loom/src/loom/tui/widgets/chat_log.py`
3. `/Users/sfw/Development/loom/tests/test_tui.py`

Deliverables:
1. Startup and `/resume` chat hydration path.
2. Legacy fallback replay adapter.
3. Older-page loading path and rendered-row cap.
4. Non-fatal per-row replay error handling.

### W4: Config and Docs
Files:
1. `/Users/sfw/Development/loom/src/loom/config.py`
2. `/Users/sfw/Development/loom/loom.toml.example`
3. `/Users/sfw/Development/loom/docs/CONFIG.md`
4. `/Users/sfw/Development/loom/CHANGELOG.md`

Proposed config knobs:
1. `[tui].chat_resume_page_size = 250`
2. `[tui].chat_resume_max_rendered_rows = 1200`
3. `[tui].chat_resume_use_event_journal = true`
4. `[tui].chat_resume_enable_legacy_fallback = true`

## Acceptance Criteria
1. Resuming a persisted session (startup auto-resume or `/resume`) shows prior chat messages in panel.
2. Switching sessions does not retain rows from previous session.
3. New sessions start with empty transcript except current-session info banners.
4. Tool rows replay as readable completed entries with args + status + output/error summary.
5. Legacy sessions (no journal rows) still replay from `conversation_turns`.
6. Large histories do not freeze UI due to paging and render cap.
7. All new behavior is covered by automated tests.

## Test Strategy

### Unit
1. Store event append/list/sequence monotonicity.
2. Event page boundaries (`before_seq`, `after_seq`, limit cap).
3. Legacy synthesis mapping for mixed role/tool-call chains.
4. Malformed tool JSON fallback behavior.

### TUI Unit/Integration
1. `_initialize_session` auto-resume hydrates chat rows (not only input history).
2. `_switch_to_session` clears and rehydrates transcript.
3. `_new_session` clears transcript baseline.
4. `/resume` while `_chat_busy` is rejected with clear info message.
5. Render cap trims oldest rows and preserves continuity sentinel.

### Regression
1. Existing session persistence tests remain green.
2. Existing `/clear` semantics remain local-view only.
3. No regressions for files/events/progress panel restore flows.

### Performance
1. Benchmark hydrate of 300/1000 row pages in CI-friendly thresholds.
2. Verify page load and render durations against SLOs.

## Rollout Plan
1. Phase A (dark launch): ship schema + store + hidden feature flag (`chat_resume_use_event_journal=false`).
2. Phase B (internal on): enable event journal writes and hydration in local/internal usage.
3. Phase C (default on): enable by default with legacy fallback still enabled.
4. Phase D (stabilize): monitor diagnostics and tune page/cap defaults.

Rollback:
1. Disable journal usage via config flag; replay continues via legacy synthesis.
2. No destructive migration required.

## Risk Register
1. Dual-write divergence between `conversation_turns` and journal.
2. Mitigation: journal is replay-only; context logic remains on `conversation_turns`.
3. Risk: unbounded chat event growth in long-lived sessions.
4. Mitigation: paging + render cap; no token-delta journaling.
5. Risk: session switch during active turn causing mixed UI state.
6. Mitigation: explicit `_chat_busy` guard for session-changing slash commands.
7. Risk: malformed historical payloads crash hydration.
8. Mitigation: per-row safe parsing with skip-and-log behavior.

## Observability
Emit structured diagnostics:
1. `chat_hydrate_started` (session_id, source, page_size)
2. `chat_hydrate_completed` (rows_rendered, rows_skipped, elapsed_ms)
3. `chat_hydrate_failed` (error_class, source)
4. `chat_event_append_failed` (event_type, error_class)
5. `chat_render_trimmed` (trimmed_rows, max_rows)

## Effort Estimate
1. W1 Data layer + tests: 0.5-1.0 day
2. W2 Emission + transition hardening: 1.0-1.5 days
3. W3 Hydration + paging + caps + tests: 1.0-1.5 days
4. W4 Config/docs/changelog: 0.5 day
5. Total: 3.0-4.5 engineer-days

## Definition of Done
1. All acceptance criteria are met.
2. Test suite additions pass in CI.
3. Chat history appears correctly on resume and session switch.
4. Performance and reliability diagnostics are in place.
5. Docs/config/changelog updated.
