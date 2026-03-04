# Cowork Comprehensive Context Index: Hardened Production Implementation Plan (2026-03-04)

## Objective
Eliminate cowork's "recent window blind spot" by adding a production-grade, comprehensive session index that:
1. Covers both user and assistant turns (plus relevant tool evidence),
2. Extracts structured markers (`DECISION`, `PROPOSAL`, `RESEARCH`, `RATIONALE`, `OPEN_QUESTION`, etc.),
3. Keeps a compact always-in-context summary for the model,
4. Enables selective high-recall retrieval from full session history via tool calls.

The system must preserve existing guarantees:
1. Conversation turns remain append-only and lossless.
2. Runtime prompt windows stay bounded and token-safe.
3. Cowork remains responsive under long sessions.

## Problem Summary (Current State)
Current cowork behavior is strong on persistence but weak on structured recall:
1. Full turns are persisted (`conversation_turns`) and queryable, but retrieval is mostly keyword/range based (`LIKE`) and untyped.
2. Recall index is generated from:
   - omitted counts,
   - user-topic snippets,
   - tool names.
   It does not summarize assistant decisions/rationale in marker form.
3. Session state contains `key_decisions`, but today those are mostly populated from specific tool events (for example git commit), not from general conversational decisions.
4. The model can miss important prior decisions when user follow-ups are short and context-light.

## Research Basis (Why This Design)
This plan is based on established findings:
1. Hierarchical memory tiers and explicit memory movement are effective when context windows are limited (MemGPT).
2. Complete experience logs plus synthesized reflections and dynamic retrieval improve long-horizon behavior (Generative Agents).
3. Combining parametric model reasoning with non-parametric memory and provenance improves knowledge-intensive performance (RAG).
4. Tool use interleaved with reasoning improves reliability and interpretability for external information access (ReAct).
5. Long contexts are not sufficient by themselves; models are position-sensitive and can miss relevant middle context ("lost in the middle").
6. SQLite FTS5 with BM25 ranking and external-content indexing is suitable for high-performance lexical retrieval with deterministic operations.

## Product Decisions
1. Keep append-only `conversation_turns` as source of truth.
2. Add a separate typed cowork-memory index (no schema overloading of existing task `memory_entries`).
3. Use a hybrid retrieval path:
   - lexical candidate generation (FTS5/BM25),
   - optional semantic rerank/summarize for final packaging.
4. Build marker-oriented summaries (`DECISION`, `PROPOSAL`, `RESEARCH`, `RATIONALE`, `OPEN_QUESTION`) and expose them in prompt-visible recall index.
5. Route indexing/summarization model calls through role-based model selection:
   - primary target role: `compactor` (tier 1),
   - fallback: `extractor`, then active model only if role models unavailable.
6. Keep backward compatibility for current `conversation_recall` actions while adding v2 typed actions.

## Critique-Driven Refinements
1. Introduce strict role-adherence mode for indexer calls.
2. Config: `execution.cowork_indexer_model_role_strict = true|false` (default false in rollout).
3. When strict=true and `compactor`/`extractor` routing fails, skip LLM extraction for that batch and mark telemetry fallback reason instead of silently using primary model.

1. Add explicit token/size budgets to prevent prompt bloat.
2. Recall-index system message hard cap: 1,200 chars (configurable).
3. Per marker section cap: 4 lines.
4. Each line must include marker, terse summary, and turn refs.
5. Overflow policy: keep `DECISION` and `OPEN_QUESTION` first, then `PROPOSAL`, then `RESEARCH`.

1. Add queue backpressure and degradation behavior.
2. Per-session index queue cap: 32 pending batches.
3. If cap exceeded, coalesce into one "catch-up" batch and skip intermediate extraction passes.
4. Degrade mode uses deterministic marker parsing only until backlog recovers.

1. Add retention controls for index growth.
2. Keep all raw turns permanently.
3. For `cowork_memory_entries`, soft-retain active/resolved entries indefinitely and archive superseded/rejected entries older than N days via compaction job.
4. Retention defaults must be configurable and off by default for first rollout.

1. Add objective retrieval quality gates.
2. Build a golden replay set from historical long sessions.
3. Measure top-k recall for decisions/open questions and rationale coverage.
4. Promotion gate for default-on: >= 0.85 decision recall@10 and >= 0.75 rationale coverage on golden set.

## Scope
### In Scope
1. Cowork conversation memory indexing and retrieval.
2. Prompt-time recall index upgrade.
3. `conversation_recall` tool upgrade (typed retrieval).
4. Schema changes in SQLite and backfill strategy.
5. Telemetry, tests, rollout flags.

### Out of Scope (This Iteration)
1. Task-run memory subsystem redesign.
2. Cross-session global knowledge graph.
3. Embedding-only retrieval dependency (dense retrieval may be optional/future).

## Target Architecture

### Layer A: Durable Raw Conversation (Already Exists)
1. `conversation_turns`: verbatim source of truth.
2. `cowork_sessions`: session metadata/state.

### Layer B: Typed Cowork Memory Index (New)
Add two new storage structures:
1. `cowork_memory_entries` (typed structured facts with turn references).
2. `cowork_memory_fts` (FTS5 virtual table indexing entry text for fast retrieval).

Optional helper table:
1. `cowork_memory_index_state` (per-session indexing checkpoint and health).

### Layer C: Prompt-Visible Active Summary (New/Expanded)
1. Expand session-state payload with memory slices:
   - `active_decisions`
   - `active_proposals`
   - `recent_research`
   - `open_questions`
2. Upgrade recall-index system message to include marker sections and turn references.

### Layer D: On-Demand Retrieval Tool (Upgraded)
`conversation_recall` returns typed and evidence-backed context for selective pull.

## Data Model

### New Table: `cowork_memory_entries`
Proposed columns:
1. `id INTEGER PRIMARY KEY AUTOINCREMENT`
2. `session_id TEXT NOT NULL`
3. `entry_type TEXT NOT NULL`  
   Allowed: `decision`, `proposal`, `research`, `rationale`, `constraint`, `risk`, `open_question`, `action_item`.
4. `status TEXT NOT NULL DEFAULT 'active'`  
   Allowed: `active`, `superseded`, `resolved`, `rejected`.
5. `summary TEXT NOT NULL`
6. `rationale TEXT DEFAULT ''`
7. `topic TEXT DEFAULT ''`
8. `tags_json TEXT DEFAULT '[]'`
9. `entities_json TEXT DEFAULT '[]'`
10. `source_turn_start INTEGER NOT NULL`
11. `source_turn_end INTEGER NOT NULL`
12. `source_roles_json TEXT DEFAULT '[]'`
13. `evidence_excerpt TEXT DEFAULT ''`
14. `supersedes_entry_id INTEGER` (nullable self-reference)
15. `confidence REAL DEFAULT 0.0`
16. `fingerprint TEXT NOT NULL` (dedupe key)
17. `created_at TEXT NOT NULL DEFAULT (datetime('now'))`
18. `updated_at TEXT NOT NULL DEFAULT (datetime('now'))`

Indexes:
1. `(session_id, entry_type, status, updated_at DESC)`
2. `(session_id, source_turn_start, source_turn_end)`
3. `(session_id, fingerprint)` unique to enforce idempotence.

### New Virtual Table: `cowork_memory_fts`
Use FTS5 external-content mode:
1. Index columns: `summary`, `rationale`, `topic`, `tags_text`, `entities_text`, `evidence_excerpt`.
2. Content source: `cowork_memory_entries`.
3. Use triggers for insert/update/delete synchronization.
4. Backfill via `rebuild` and consistency checks.

Rationale:
1. Fast lexical retrieval with BM25 ranking.
2. Deterministic behavior and operationally simple deployment in SQLite.

### Optional Table: `cowork_memory_index_state`
Columns:
1. `session_id TEXT PRIMARY KEY`
2. `last_indexed_turn INTEGER NOT NULL DEFAULT 0`
3. `index_version INTEGER NOT NULL DEFAULT 1`
4. `last_indexed_at TEXT`
5. `last_error TEXT DEFAULT ''`
6. `failure_count INTEGER NOT NULL DEFAULT 0`

## Indexing Pipeline

### Ingestion Trigger
At end of each assistant turn (streaming and non-streaming):
1. enqueue indexing job for newly persisted turn range since checkpoint,
2. do not block user-visible response on index completion.

### New Component: `CoworkMemoryIndexer`
Create `src/loom/cowork/memory_indexer.py` with:
1. `index_session_delta(session_id, from_turn_exclusive, to_turn_inclusive)`
2. `extract_entries_from_turns(turns) -> list[MemoryEntryCandidate]`
3. `merge_entries(session_id, candidates)`
4. `build_active_snapshot(session_id) -> dict`

### Extraction Strategy (Hardened)
Two-stage extraction:
1. Deterministic pre-pass:
   - detect explicit markers in text (`DECISION:`, `PROPOSAL:`, `RESEARCH:`),
   - capture turn references and speaker roles.
2. LLM extraction pass (role-routed model):
   - prompt with strict JSON schema,
   - produce typed entries, status transitions, rationale, and evidence excerpt.

Hardening rules:
1. Strict JSON parse; retry once with repair prompt on invalid JSON.
2. Enforce output size and per-turn caps.
3. Deduplicate with `fingerprint`.
4. Guard against hallucinated evidence by requiring turn range references present in source batch.
5. If extraction fails, preserve checkpoint lag and retry later; never block cowork send path.

### Model Routing for Indexer
1. First choice: `role="compactor", tier=1`.
2. If unavailable: `role="extractor", tier=1`.
3. If unavailable: active cowork model.
4. Emit telemetry field `indexer_model_role` and `indexer_model_name`.

## Retrieval Pipeline (`conversation_recall` v2)

### Backward-Compatible Actions
Keep:
1. `search`
2. `range`
3. `tool_calls`
4. `summary`

### New Actions
1. `entries`
   - filters: `entry_type`, `status`, `topic`, `limit`.
2. `decision_context`
   - returns active decisions + rationale + supporting turn refs (optionally topic-filtered).
3. `timeline`
   - chronological marker stream for a topic.
4. `open_questions`
   - unresolved open questions and related proposals.
5. `source_turns`
   - fetches verbatim supporting turns for selected entry ids or turn ranges.

### Query Execution
1. Candidate generation: FTS5 `MATCH` with BM25 ordering.
2. Filter/boost:
   - boost active entries over superseded,
   - boost decision/proposal types for ambiguous "move on" follow-ups,
   - recency tie-break by `updated_at`.
3. Optional compactor pass for output packaging under tool-output size budget.
4. Render with explicit markers and evidence:
   - `[DECISION][active] ... (turns 12-14)`
   - `RATIONALE: ...`
   - `EVIDENCE: ...`

Response contract additions:
1. Always return `entry_id`, `entry_type`, `status`, and `turn_refs`.
2. Include `confidence` and `source="index" | "raw_turns_fallback"`.
3. When no results, include suggested follow-up queries instead of empty opaque output.

## Prompt Integration

### Recall Index Upgrade
Replace current `_build_recall_index_message` content logic with memory-aware summary:
1. counts:
   - omitted messages/tool messages,
   - indexed entries available.
2. marker sections (compact):
   - `Active DECISION`
   - `Open PROPOSAL`
   - `Recent RESEARCH`
   - `Open QUESTION`
3. each line includes turn refs.
4. include recommended recall actions with query templates.

Fallback behavior:
1. If no memory entries exist yet, use current legacy index format.
2. If index health is degraded, emit explicit "index_degraded" note and continue with legacy snippets.

### Session State Upgrade
Extend `SessionState` with compact active memory fields:
1. `active_decisions` (max N, with turn refs)
2. `active_proposals`
3. `open_questions`
4. `recent_research`

This ensures critical decisions remain in-context even when source turns roll out of live window.

## Operational Hardening

### Reliability
1. Indexing queue with bounded in-memory backlog per session.
2. Timeout budget for extraction calls.
3. Circuit breaker:
   - disable extraction temporarily after repeated failures,
   - keep lexical retrieval from raw turns available.
4. Idempotent writes via `fingerprint` unique index.
5. Single-flight indexing per session to prevent concurrent merge races.

### Performance
1. Batch turn ranges (coalesce bursts) to reduce model-call overhead.
2. Cache extracted entry hashes to skip unchanged deltas.
3. Use FTS5 rank for fast candidate selection.
4. Hard cap extraction batch size by characters and turns to avoid oversized model requests.

### Data Integrity
1. FTS external-content triggers + rebuild command for backfill.
2. Integrity-check operation on startup (sampled) and admin command for full rebuild.
3. Explicit migration guard if FTS5 unavailable:
   - disable v2 actions requiring FTS rank,
   - fallback to deterministic SQL `LIKE`.

### Security and Privacy
1. No external data export introduced; all storage remains local SQLite.
2. Continue escaping search input for SQL safety.
3. Redact sensitive tokens from telemetry previews.

## Telemetry and Observability
Add structured events:
1. `cowork_memory_index_started`
2. `cowork_memory_index_completed`
3. `cowork_memory_index_failed`
4. `cowork_memory_recall_query`
5. `cowork_memory_recall_result`
6. `cowork_memory_recall_fallback`

Key metrics:
1. indexing lag (`latest_turn - last_indexed_turn`)
2. extraction latency p50/p95
3. extraction parse-failure rate
4. dedupe hit rate
5. recall tool hit rate by action
6. `% turns with recall_index marker sections populated`
7. model-role adherence for indexer (`compactor` usage rate)
8. recall-index prompt size and marker fill-rate
9. queue backlog depth and time-in-queue p95

## Implementation Workstreams

### Workstream 1: Schema and Store Foundation
Files:
1. `src/loom/state/schema.sql`
2. `src/loom/state/conversation_store.py`
3. new tests in `tests/test_conversation_store.py`

Deliverables:
1. new memory index tables + FTS table + triggers,
2. store APIs for insert/query/update checkpoint and typed entry retrieval,
3. migration-safe initialization and rebuild hooks.

### Workstream 2: Memory Indexer
Files:
1. new `src/loom/cowork/memory_indexer.py`
2. `src/loom/cowork/session.py`
3. tests: `tests/test_cowork_memory_indexer.py`

Deliverables:
1. async delta indexing pipeline,
2. strict JSON extraction + repair path,
3. dedupe/status merge logic.

### Workstream 3: Session State + Recall Index Upgrade
Files:
1. `src/loom/cowork/session_state.py`
2. `src/loom/cowork/session.py`
3. tests: `tests/test_cowork.py`

Deliverables:
1. marker-rich active memory in prompt state,
2. upgraded recall-index message content,
3. compatibility fallback when index empty/unhealthy.

### Workstream 4: `conversation_recall` v2
Files:
1. `src/loom/tools/conversation_recall.py`
2. `src/loom/state/conversation_store.py`
3. tests: `tests/test_conversation_recall.py`

Deliverables:
1. typed actions (`entries`, `decision_context`, `timeline`, `open_questions`),
2. evidence-backed output formatting,
3. BM25-based ranking and fallback path.

### Workstream 5: TUI Wiring and Role-Adherence Telemetry
Files:
1. `src/loom/tui/app.py`
2. `src/loom/cowork/display.py` and `src/loom/tui/widgets/chat_log.py` (only if new badges needed)
3. tests: `tests/test_tui.py`

Deliverables:
1. ensure indexer receives role-selected model,
2. expose indexing/lag diagnostics in debug surfaces,
3. add replay-safe telemetry events.

### Workstream 6: Rollout, Backfill, and Guardrails
Files:
1. `src/loom/config.py`
2. `loom.toml.example`
3. test coverage updates

Deliverables:
1. feature flags:
   - `cowork_memory_index_enabled`
   - `cowork_memory_index_v2_actions_enabled`
   - `cowork_memory_index_force_fts`
2. one-time backfill command:
   - rebuild index for existing sessions.
3. operational runbook for rollback and integrity rebuild.
4. golden-set evaluation harness and score reporting command.

## Testing Strategy (Production Readiness)

### Unit Tests
1. extraction schema validation and repair fallback,
2. dedupe fingerprint behavior,
3. status transitions (proposal -> decision, decision -> superseded),
4. FTS query ranking and fallback behavior.

### Integration Tests
1. long session with interleaved decisions/research and short follow-ups,
2. verify recall-index includes marker sections with turn refs,
3. verify `conversation_recall decision_context` returns correct evidence from old turns,
4. verify assistant + user content both contribute to indexed entries.

### Failure/Resilience Tests
1. compactor/extractor unavailable -> fallback model path,
2. invalid extraction JSON -> repair and then fail-safe behavior,
3. DB restart/reopen with pending lag -> catch-up indexing,
4. FTS inconsistency -> rebuild path restores recall results.

### Performance Tests
1. 10k-turn session index build time and steady-state turn overhead,
2. recall query p95 latency under large sessions,
3. memory footprint of indexer queue.

## Rollout Plan

### Phase 1 (Shadow Indexing)
1. Build index in background, do not alter prompts/tool outputs.
2. Observe lag, failure rate, extraction quality.
3. Run golden-set offline evaluation and tune thresholds.

### Phase 2 (Tool v2 Dogfood)
1. Enable new recall actions behind flag.
2. Keep legacy recall-index prompt text.

### Phase 3 (Prompt Integration)
1. Enable marker-based recall index in prompt.
2. Keep automatic fallback to legacy index if memory index unhealthy.
3. Enforce recall-index char budgets and overflow eviction policy.

### Phase 4 (Default On)
1. Enable by default for cowork.
2. Keep rollback flags for one release cycle.

## Acceptance Criteria
1. In a 1+ hour cowork session, model can recover prior decisions/rationale using typed recall without user re-explaining.
2. Recall index in prompt shows marker-based sections (`DECISION`, `PROPOSAL`, `RESEARCH`, `OPEN_QUESTION`) with turn refs whenever history is omitted and indexed.
3. `conversation_recall` can return prior assistant decisions and their rationale, not just user-topic snippets.
4. Role adherence is visible: indexing/summarization calls prefer configured `compactor` model.
5. No regression in cowork latency beyond agreed budget (target: <10% median turn overhead after warm-up).
6. Golden-set retrieval quality meets threshold gates before default-on.

## Risks and Mitigations
1. Risk: extraction hallucination or mislabeling.
   Mitigation: evidence-turn validation, strict schema checks, confidence thresholds, and conservative rendering.
2. Risk: added latency/cost from extraction calls.
   Mitigation: async queue, batching/coalescing, cheap model role, bounded retries.
3. Risk: FTS drift/inconsistency.
   Mitigation: triggers, periodic integrity checks, rebuild tooling.
4. Risk: overgrowth of index entries.
   Mitigation: dedupe fingerprints, supersession links, optional archival compaction of stale inactive entries.

## References
1. MemGPT: [https://arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)
2. Generative Agents: [https://arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
3. Retrieval-Augmented Generation (RAG): [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
4. ReAct: [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
5. Lost in the Middle: [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)
6. SQLite FTS5: [https://www.sqlite.org/fts5.html](https://www.sqlite.org/fts5.html)
