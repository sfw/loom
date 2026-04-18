# Data Authority Unification Plan (2026-04-02)

## Objective
Eliminate split-authority drift across Loom by assigning one authoritative store per state domain, routing all writes through one persistence boundary per domain, and treating every other representation as an explicit projection, cache, or index.

This plan is the follow-on to the task-state authority fix that made `state.yaml` authoritative over the mirrored `tasks` row. The goal here is to finish that cleanup and apply the same discipline to cowork conversation and session state.

## Executive Summary
The review found four concrete classes of authority problems still in the repo:
1. Task state can still drift from the SQLite `tasks` row because several live paths write canonical task state directly and skip mirror sync.
2. A non-empty cowork replay journal can mask fresher `conversation_turns`, so partial transcript data can override the true semantic conversation history.
3. Some run-control endpoints still trust the `tasks` row directly even though task state is now authoritative.
4. Cowork session resume trusts the compact `cowork_sessions` snapshot even though turn history can be fresher.

The cleanup strategy is:
1. Finish the task-state authority refactor so every live task write goes through one snapshot persistence path.
2. Split cowork domains explicitly:
   - `conversation_turns` is authoritative for semantic conversation history
   - `cowork_chat_events` is a transcript projection, never the authority for conversation semantics
   - `cowork_sessions` is a checkpoint projection that must be advanced atomically with committed turns
3. Add freshness and coverage markers so projections cannot silently override fresher canonical data.
4. Add repair, backfill, and doctor checks so mismatches are detected and healed instead of silently tolerated.

## Why This Plan
The recent stale-seal failure exposed a broader architectural issue: Loom still has several places where two representations of the same logical state are both treated as writable and readable authorities.

That creates a recurring failure pattern:
1. Canonical state advances.
2. A projection or cache does not advance with it.
3. A later read trusts the stale projection.
4. The system fails or presents incorrect state after restart, resume, or UI refresh.

The task-state fix addressed one instance of that pattern. This plan addresses the remaining ones systematically instead of as one-off bugs.

## Repo-Accurate Audit

### 1) Task execution state
Current reality:
1. Task snapshots on disk are now the intended authority.
2. The `tasks` row is supposed to be a mirror used for fast listing and workspace queries.
3. Several live paths still call `self._state.save(task)` directly and bypass the mirrored save path.
4. Three run-control endpoints still gate on `engine.database.get_task(run_id)` instead of loading canonical task state first.

Implication:
1. Task row drift can still reappear for pause, resume, cancel, and clarification metadata.

### 2) Cowork semantic conversation history
Current reality:
1. `conversation_turns` is the append-only, write-through record of the actual conversation.
2. Status and recovery semantics should come from this history, not from UI replay rows.

Implication:
1. This should be the canonical semantic history for cowork.

### 3) Cowork UI transcript journal
Current reality:
1. `cowork_chat_events` stores UI-facing transcript rows.
2. Several API and TUI readers prefer the journal whenever it is non-empty.
3. Journal writes and turn writes are separate operations with no explicit completeness contract.

Implication:
1. A partial or stale journal can suppress synthesis from fresher turns.
2. Semantic readers currently risk consulting a projection instead of the canonical history.

### 4) Cowork session summary snapshot
Current reality:
1. `cowork_sessions` stores `turn_count`, `total_tokens`, `session_state`, and session metadata.
2. Turn persistence and session snapshot persistence happen through separate calls.
3. Resume restores the compact session snapshot first, even if the turn log is fresher.

Implication:
1. Session resume and session lists can observe stale counts or stale summary state after partial persistence failure.

## Authority Decisions
This refactor should make the authority model explicit by domain.

### Domain A: Task run state
Authority:
1. `state.yaml` plus the task's evidence ledger and companion task-state artifacts.

Projection:
1. SQLite `tasks` row.

Rules:
1. If task state exists on disk, it wins.
2. No live path may write task state except through the authoritative snapshot persistence API.
3. No stateful read path may trust row `plan` or row `metadata` over canonical task state.

### Domain B: Cowork semantic conversation state
Authority:
1. SQLite `conversation_turns`.

Projection:
1. `cowork_chat_events` transcript rows.
2. `cowork_sessions` summary and counters.

Rules:
1. Any semantic question about the conversation must come from turns or an explicit state derived from turns.
2. Transcript rows must never be used as the authority for pending prompt detection, completion semantics, or recovery decisions.

### Domain C: Cowork transcript rendering
Authority:
1. Committed transcript rows in `cowork_chat_events`, but only for the portion of the conversation that the journal explicitly declares it covers.

Fallback authority:
1. `conversation_turns` for any uncovered or stale range.

Rules:
1. A non-empty journal is not enough to claim authority.
2. Readers must know the journal coverage boundary.
3. If the requested transcript extends beyond journal coverage, the uncovered suffix must be synthesized from turns.

### Domain D: Cowork session summary
Authority:
1. `cowork_sessions` is allowed to be the authoritative summary only as an atomic checkpoint derived from committed turns.

Rules:
1. The session snapshot cannot be updated independently of the turn history it summarizes.
2. The snapshot must record the turn boundary through which it is valid.
3. Resume must validate that boundary before trusting the snapshot.

## Goals
1. No live code path has two writable authorities for the same domain.
2. Every projection has an explicit freshness or coverage contract.
3. State/control/recovery paths load canonical state first.
4. UI transcript code can preserve replay fidelity without being allowed to override semantic history.
5. Repair and observability paths exist so drift becomes diagnosable and fixable.

## Non-Goals
1. Replacing SQLite with another database.
2. Moving cowork session state to disk just to mirror the task-state design.
3. Rewriting the cowork transcript UX from scratch.
4. Removing `cowork_chat_events`; the goal is to demote it to a disciplined projection, not necessarily delete it.
5. General-purpose schema redesign outside the authority/freshness fields needed for this cleanup.

## Design Principles
1. One domain, one authority.
2. One authoritative writer per domain.
3. Projections are derived and repairable, not peer authorities.
4. Read paths must know freshness, not guess it.
5. Status logic uses semantic stores, not UI caches.
6. Migration-first for any SQLite schema changes.
7. Repair stale data instead of requiring manual database deletion.

## Proposed Architecture

### 1) Finish task-state authority closure
Introduce a single authoritative task persistence API and make every live task write route through it.

Recommended shape:
1. `persist_task_snapshot(task, *, reason)` writes canonical task state first.
2. The same path derives and updates the SQLite task-row mirror from that snapshot.
3. The same path stamps a freshness marker such as `snapshot_revision` or `state_snapshot_updated_at` in the mirror row.

Effects:
1. Direct `self._state.save(task)` in live code becomes an implementation bug.
2. Run-control and run-detail routes use canonical task state for existence and semantics.
3. SQLite remains fast query infrastructure, not a second task authority.

### 2) Separate cowork semantics from transcript projection
Make the boundary explicit:
1. `conversation_turns` answers semantic questions.
2. `cowork_chat_events` answers transcript rendering questions only within its declared coverage.

This means:
1. Pending prompt detection moves off the journal and onto turns or explicit semantic session state.
2. Transcript readers become coverage-aware instead of journal-first.
3. Streaming and resume paths may still prefer durable journal rows for exact UI fidelity, but only for the covered portion.

### 3) Introduce checkpoint and coverage metadata for cowork
Add explicit fields so projections can advertise what they actually cover.

Recommended fields:
1. On `cowork_sessions`:
   - `session_state_through_turn INTEGER NOT NULL DEFAULT 0`
   - `chat_journal_through_turn INTEGER NOT NULL DEFAULT 0`
   - optional `projection_revision INTEGER NOT NULL DEFAULT 0`
2. Optionally, if journal coverage needs finer fidelity than turn count alone:
   - `chat_journal_mode TEXT NOT NULL DEFAULT 'legacy'`
   - values such as `legacy`, `partial`, `complete`

Intent:
1. `session_state_through_turn` tells resume whether `session_state`, `turn_count`, and `total_tokens` are trustworthy through the latest committed turn.
2. `chat_journal_through_turn` tells transcript readers how much of the conversation is safely renderable from the journal alone.

### 4) Create one transactional cowork checkpoint writer
Create one store-level write boundary for committed cowork progress.

Recommended shape:
1. Append one or more `conversation_turns`.
2. Update `cowork_sessions.turn_count`, `total_tokens`, `last_active_at`, and `session_state`.
3. Advance `session_state_through_turn`.
4. Optionally append associated `cowork_chat_events` for the completed turn and advance `chat_journal_through_turn`.
5. Commit all of the above in one SQLite transaction.

Important nuance:
1. Some transcript rows are UI-only notices and may not correspond to turns.
2. Those may still append to `cowork_chat_events` independently, but they must never be used for semantic inference.
3. The authoritative coverage marker must only advance when the transcript is complete through a committed turn boundary.

### 5) Make transcript readers coverage-aware
Add a single transcript loading API with explicit merge behavior.

Recommended shape:
1. `get_transcript_page(session_id, before_cursor=None, limit=...)`
2. The API loads:
   - durable journal rows for the covered portion
   - synthesized rows from `conversation_turns` for any uncovered suffix or prefix
3. The API returns a unified transcript page plus cursor metadata.

Effects:
1. API and TUI stop implementing ad hoc "journal if any rows else synthesize" logic.
2. Partial journal data can no longer suppress fresher turns.
3. Live UI keeps replay fidelity where available without lying about completeness.

## Workstreams

## Workstream 0: Authority Contracts and Inventory

### Problem
The codebase still relies on implicit assumptions about which store is canonical in each subsystem.

### Plan
1. Write down the authority contract in code comments and developer docs for tasks, cowork turns, transcript journal, and session snapshot.
2. Add small internal helpers that encode those contracts instead of letting each route choose its own store.
3. Audit for direct writes and direct row-first reads that violate the contract.

### Primary Files
1. `src/loom/api/routes.py`
2. `src/loom/api/engine.py`
3. `src/loom/engine/orchestrator/core.py`
4. `src/loom/engine/runner/core.py`
5. `src/loom/state/conversation_store.py`
6. `src/loom/cowork/session.py`
7. `planning/refactors/2026-04-02-DATA-AUTHORITY-UNIFICATION-PLAN.md`

### Acceptance
1. The domain authority contract is explicit in code and docs.
2. We have a complete inventory of read/write surfaces to migrate.

## Workstream 1: Complete Task-State Authority

### Problem
Task state is intended to be authoritative, but several live paths still bypass the mirrored save path or read the mirrored task row first.

### Plan
1. Replace all live direct `self._state.save(task)` calls with the authoritative task snapshot persistence API.
2. Route pause, resume, cancel, clarification markers, replanning markers, run-id initialization, and finalization through the same path.
3. Make run-control routes state-first:
   - `/runs/{id}/pause`
   - `/runs/{id}/resume`
   - `/runs/{id}/message`
4. Add a freshness marker to the mirrored row so stale rows are diagnosable and repairable.
5. Remove or heavily restrict any direct `update_task_plan` / `update_task_metadata` usage that bypasses canonical snapshot writes.

### Likely Files
1. `src/loom/api/engine.py`
2. `src/loom/api/routes.py`
3. `src/loom/engine/orchestrator/core.py`
4. `src/loom/engine/orchestrator/output.py`
5. `src/loom/engine/orchestrator/runtime.py`
6. `src/loom/engine/runner/core.py`
7. `src/loom/state/memory.py`
8. `src/loom/state/task_state.py`

### Migration Considerations
If we add mirror freshness columns to `tasks`, follow the migration policy:
1. Update `src/loom/state/schema.sql`
2. Update `src/loom/state/schema/base.sql`
3. Add migration step files under `src/loom/state/migrations/steps/`
4. Register steps in `src/loom/state/migrations/registry.py`
5. Add migration and upgrade tests
6. Update changelog and operator docs

### Tests
1. Pause, resume, cancel, and clarification flows update both canonical state and mirror row.
2. Restart from stale mirror data prefers canonical state and repairs the mirror.
3. No run-control route 404s when canonical task state exists but the mirror row is missing or stale.
4. Freshness markers advance on every canonical snapshot save.

### Acceptance
1. No live task path writes state except through the authoritative snapshot writer.
2. No stateful route trusts the mirrored task row over canonical task state.

## Workstream 2: Stop Using Transcript Rows for Conversation Semantics

### Problem
Cowork transcript rows are currently consulted for semantics such as pending prompt detection even though they are a UI projection.

### Plan
1. Move semantic read helpers to `conversation_turns` or explicit semantic session state derived from turns.
2. Remove journal-based pending prompt detection and similar status logic.
3. Keep `cowork_chat_events` for transcript replay only.
4. Add narrow helper APIs so future features do not reintroduce transcript-as-authority reads.

### Likely Files
1. `src/loom/api/routes.py`
2. `src/loom/state/conversation_store.py`
3. `src/loom/cowork/session.py`

### Tests
1. `ask_user` pending status remains correct even when the chat journal is empty, partial, or stale.
2. Semantic status and transcript rendering can diverge safely without breaking conversation correctness.

### Acceptance
1. No conversation status API depends on `cowork_chat_events` for semantic truth.

## Workstream 3: Coverage-Aware Transcript Projection

### Problem
Current transcript readers treat any non-empty journal as authoritative, which lets partial journal data suppress fresher turn history.

### Plan
1. Add explicit journal coverage metadata, likely through `cowork_sessions.chat_journal_through_turn`.
2. Create one transcript read helper that merges:
   - journal-backed transcript rows for covered ranges
   - synthesized rows from turns for uncovered ranges
3. Update:
   - API conversation events
   - API conversation stream initial replay
   - TUI chat hydrate
   - TUI older-history paging
4. Treat legacy sessions as `chat_journal_through_turn = 0`.

### Likely Files
1. `src/loom/state/conversation_store.py`
2. `src/loom/api/routes.py`
3. `src/loom/tui/app/chat/history.py`
4. `tests/test_conversation_store.py`
5. `tests/test_api.py`
6. `tests/test_tui.py`

### Migration Considerations
Likely schema additions to `cowork_sessions`:
1. `chat_journal_through_turn`
2. optional `chat_journal_mode`
3. optional projection revision fields

Follow the database migration policy for any such changes.

### Tests
1. Partial journal plus fresher turns yields a complete merged transcript.
2. Resume and older-history paging never lose rows because the journal is merely non-empty.
3. Live stream reconnect can bridge journal-backed history with synthesized suffixes correctly.

### Acceptance
1. A stale or partial journal can no longer hide fresher turns.
2. Transcript replay remains high-fidelity for fully covered ranges.

## Workstream 4: Atomic Cowork Checkpoint Persistence

### Problem
Cowork turns, transcript rows, and session summary state are persisted separately, so partial failure can leave the summary or transcript lagging behind committed conversation history.

### Plan
1. Introduce one transactional persistence boundary in `ConversationStore` for committed turn progress.
2. Persist turn history and session summary together.
3. Advance journal coverage only when associated transcript rows are fully durable through a turn boundary.
4. Keep journal-only UI notices separate from semantic checkpoints.
5. Ensure `last_active_at`, `turn_count`, `total_tokens`, `session_state`, and through-turn markers all advance together.

### Recommended API Shape
1. `persist_checkpoint(session_id, checkpoint)` or equivalent
2. Inputs include:
   - appended turns
   - updated session summary
   - optional transcript events for the covered turn boundary
3. Outputs include:
   - latest committed turn number
   - latest transcript sequence
   - updated through-turn markers

### Likely Files
1. `src/loom/state/conversation_store.py`
2. `src/loom/cowork/session.py`
3. `src/loom/tui/app/chat/history.py`
4. `tests/test_conversation_store.py`
5. `tests/test_tui.py`

### Tests
1. If checkpoint persistence succeeds, turns and session snapshot are mutually consistent.
2. If transcript event persistence is skipped or fails, session semantics still remain correct and uncovered transcript ranges synthesize from turns.
3. Resume after interruption never shows a session snapshot that claims to be fresher than its through-turn marker.

### Acceptance
1. `cowork_sessions` is no longer independently mutated outside the checkpoint writer for semantic state.
2. Session summary drift from committed turn history is structurally prevented.

## Workstream 5: Resume, Repair, and Backfill

### Problem
Existing databases may already contain stale mirrors, partial journals, or stale session summaries.

### Plan
1. Add startup or on-demand repair helpers:
   - rebuild task-row mirrors from task state
   - recompute cowork session turn counts and checkpoint markers from turns
   - initialize legacy journal coverage as uncovered rather than pretending completeness
2. Teach `loom db doctor` or equivalent diagnostic paths to surface authority mismatches.
3. Add lightweight mismatch logging and counters so new drift is immediately visible.

### Likely Files
1. `src/loom/api/engine.py`
2. `src/loom/state/memory.py`
3. `src/loom/state/conversation_store.py`
4. `src/loom/cli` or DB doctor command modules
5. `tests/`

### Tests
1. Repair logic upgrades stale databases without destructive resets.
2. Doctor surfaces known mismatch cases with actionable output.

### Acceptance
1. Existing users can upgrade without deleting local state.
2. Authority drift becomes observable and repairable.

## Workstream 6: Guardrails and Regression Coverage

### Problem
Even after cleanup, split-authority regressions can creep back in through new direct writes or row-first reads.

### Plan
1. Add targeted tests that encode the authority contract.
2. Add code-search style assertions where appropriate:
   - no live direct `self._state.save(task)` outside the authoritative persistence helper
   - no run-control route gates solely on `database.get_task`
3. Add regression tests for:
   - stale task row plus fresh task state
   - partial chat journal plus fresher turns
   - stale cowork session snapshot plus fresh turns
4. Add comments or helper names that make authority violations visually obvious in future reviews.

### Acceptance
1. The next stale-authority bug should fail a targeted test before it reaches users.

## Suggested Implementation Order
1. Workstream 1: finish task-state authority closure first because it is already partially refactored and highest confidence.
2. Workstream 2: stop journal-based semantic reads next because it reduces correctness risk immediately.
3. Workstream 3: add transcript coverage metadata and unified transcript readers.
4. Workstream 4: introduce the atomic cowork checkpoint writer.
5. Workstream 5: backfill, repair, and doctor support.
6. Workstream 6: tighten guardrails and invariant tests throughout.

This order gives us correctness wins early and lets the cowork checkpoint work land on top of clearer domain boundaries.

## Risks and Mitigations

### Risk 1: Cowork session state may not be fully reconstructible from raw turns today
Mitigation:
1. Do not rely on post-hoc full reconstruction as the primary model.
2. Make the checkpoint writer atomic so future drift is prevented.
3. Use through-turn markers so stale snapshots are detectable even when perfect rebuild is not available.

### Risk 2: Transcript coverage rules add complexity to paging and streaming
Mitigation:
1. Centralize transcript loading behind one helper instead of duplicating merge logic across API and TUI.
2. Keep journal coverage turn-based unless finer granularity is proven necessary.

### Risk 3: Schema additions increase migration complexity
Mitigation:
1. Keep new fields minimal and purpose-specific.
2. Follow the repository migration-first policy strictly.
3. Add upgrade tests for partially stale real-world states.

### Risk 4: Performance regressions from more canonical reads
Mitigation:
1. Preserve mirrors and journals for fast queries and replay.
2. Use freshness markers so only stateful correctness paths pay the extra validation cost.
3. Measure hot routes after each phase.

## Definition of Done
1. Task state is the sole authority for task execution state, and every task write goes through one snapshot writer.
2. No run-control or run-detail semantics depend on stale task-row data.
3. `conversation_turns` is the sole authority for cowork conversation semantics.
4. `cowork_chat_events` is transcript-only and coverage-aware.
5. `cowork_sessions` advances only through an atomic checkpoint writer and advertises the turn boundary it covers.
6. Drift repair and diagnostics exist for old local databases.
7. Targeted regression tests enforce the authority contract across task and cowork systems.

## Deliverables
1. Code changes across task, conversation, and cowork persistence layers.
2. Any required schema and migration updates.
3. Repair and doctor support for stale local state.
4. Tests that codify the new authority model.
5. Changelog and operator notes for any migration-affecting changes.
