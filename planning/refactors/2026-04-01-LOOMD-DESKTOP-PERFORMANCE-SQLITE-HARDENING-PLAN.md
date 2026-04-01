# Loomd Desktop Performance and SQLite Hardening Plan (2026-04-01)

## Objective
Reduce user-visible latency in the desktop app during active runs and conversations by fixing the event pipeline, reducing avoidable frontend load, and tuning SQLite for Loom's local-app access pattern without introducing a PostgreSQL dependency.

## Implementation Update (2026-04-01)
Completed in this refactor pass:
1. `src/loom/state/memory.py` now uses a governed SQLite runtime model:
   - one long-lived write connection
   - a small long-lived read pool
   - explicit pragmas for WAL desktop usage (`busy_timeout=5000`, `synchronous=NORMAL`, `wal_autocheckpoint=2000`, `temp_store=MEMORY`, `cache_size=-16384`)
   - lightweight connection/query counters for measurement
2. `src/loom/events/bus.py` now queues async handlers behind bounded worker queues instead of spawning one task per async handler invocation.
3. `EventPersister` now uses its own dedicated durable queue instead of the best-effort async-handler worker path, batches compliance-event inserts, and drains cleanly on shutdown/test synchronization.
4. `/runs/{run_id}/stream` no longer polls SQLite for each live event. It replays durable rows once, bridges through recent in-memory history for pre-flush reconnect correctness, and then follows live event-bus delivery with a sequence cursor.
5. `/notifications/stream` no longer polls SQLite for each live notification. It replays durable rows once, bridges through recent in-memory history for pre-flush reconnect correctness, and then follows live event-bus delivery with reconnect-by-event-id fallback.
6. Conversation SSE queues are now bounded with durable catch-up on overflow.
7. Desktop hooks now reduce overlap between SSE and polling:
   - notification-driven workspace refreshes fetch approvals + overview instead of the full five-endpoint workspace surface
   - run refresh polling backs off while stream activity is fresh
   - conversation live sync polling backs off while stream activity is fresh
   - desktop activity and runtime health polling intervals are less aggressive
8. Workspace-scoped task/session reads now query the DB/store directly instead of loading all rows and filtering in Python, and workspace overview batches latest-run/link lookups while counting approvals without materializing the full inbox.
9. Hot route paths moved more `TaskStateManager` work behind `asyncio.to_thread(...)` helpers instead of direct synchronous calls on the event loop.
10. Desktop workspace data now loads lazily by visible tab instead of always fetching overview, approvals, settings, inventory, and artifacts together.
11. Durable task-run lease acquisition and heartbeat writes now use the governed SQLite runtime path instead of opening ad hoc SQLite connections.
12. Key hot API surfaces now emit low-overhead latency diagnostics when `LOOM_LATENCY_DIAGNOSTICS=1`, including:
   - workspace overview
   - workspace artifacts
   - approvals
   - conversation detail/messages/events/status
   - run detail/timeline
   - run stream initial replay/open
13. Added `scripts/active_run_latency_smoke.py` as a synthetic active-run benchmark harness for the main desktop API hot paths.
14. No schema changes were required in this refactor pass.

Durability/performance tradeoff called out explicitly:
1. `synchronous=NORMAL` in WAL mode trades a small amount of worst-case crash durability for materially lower local write latency.
2. Batched event persistence means durable visibility is no longer one commit per emitted event; it is delayed by a short in-process queue/flush window.
3. When the durable event queue reaches its configured ceiling, the persister now synchronously flushes the oldest queued batch on the emit path to keep memory bounded and preserve ordering, trading producer latency under overload for durability and bounded queue growth.

Local measurement snapshot captured after the refactor:
1. Synthetic 1000-event benchmark:
   - naive fresh SQLite connection per event: `900.69 ms`
   - batched `EventPersister` path: `6.85 ms`
   - observed speedup: `131.42x`
2. Batched benchmark instrumentation snapshot:
   - connection opens: `3`
   - persisted rows: `1000`
   - batch count: `16`
   - max batch size observed: `64`

## Gap-Closure Update (2026-04-01)
Follow-up after implementation review closed the remaining correctness and load-shedding gaps from the first pass:
1. Durable compliance-event persistence is no longer routed through the event bus's best-effort bounded async-handler workers, so handler backpressure no longer silently drops persistent events.
2. Run and notification stream reconnects now bridge durable replay into recent in-memory event history, covering the short batching window before SQLite flush completes.
3. Overflow recovery for bounded run/notification stream queues now replays recent history before resuming live delivery so reconnect semantics remain lossless across slow-consumer recovery.
4. Workspace overview/run/conversation surfaces now batch relationship lookups and avoid full pending-approval materialization on hot overview requests.
5. Desktop workspace surface loading is now tab-aware and lazy on initial selection, so the shell no longer bursts all heavy workspace endpoints together by default.
6. Added targeted regression coverage for:
   - durable event persistence under queue pressure
   - run stream reconnect before SQLite flush
   - notification stream reconnect before SQLite flush
   - notification stream first connect before SQLite flush
   - bounded run-stream overflow recovery
   - batched workspace relationship queries
   - conversation stream-health backoff helper
   - lazy workspace tab loading
7. Added low-overhead route timing diagnostics for the main desktop hot paths and a synthetic active-run API benchmark harness (`scripts/active_run_latency_smoke.py`) so before/after latency and queue-depth snapshots are reproducible locally.

Remaining follow-up that is still useful but not required for correctness:
1. Broader fixture variants for the smoke harness (for example multiple workspaces, heavier artifact inventories, or longer-lived concurrent emitters) would improve benchmark breadth, but the core active-run measurement path now exists.

## Executive Summary
The current latency issue does not look like a fundamental "SQLite is too small" problem. It looks like a usage-model problem:
1. The backend persists many tiny event writes with fresh SQLite connections and commits.
2. Live stream routes often re-query SQLite to wait for those persisted rows before yielding.
3. The desktop app continues polling status, messages, events, activity, and workspace surfaces while SSE streams are already active.
4. Workspace overview and artifact surfaces are comparatively heavy read models and are refreshed in bursts.
5. Several async request paths still perform synchronous file or YAML work on the main event loop.

The recommended strategy is:
1. Keep SQLite as the primary store.
2. Refactor Loom to behave better with SQLite's strengths: one process, local disk, WAL mode, fast reads, and a controlled write path.
3. Add bounded queues, batching, and stream/load shedding before adding more concurrency.
4. Use threads for blocking filesystem work and dedicated writer/worker paths where that reduces event-loop stalls.
5. Re-evaluate PostgreSQL only if measured latency remains unacceptable after these refactors.

## Decision
SQLite remains the default and preferred local-app database for this program.

We should not migrate to PostgreSQL in this refactor wave because:
1. PostgreSQL would not by itself fix event write amplification, stream read amplification, or frontend over-polling.
2. Loomd is currently a local sidecar process, not a multi-host shared service.
3. Shipping and supervising PostgreSQL would add operational complexity, installation friction, backup complexity, and new failure modes for a desktop product.
4. The current hotspots are tractable within SQLite if we change the access pattern.

## Reported Symptoms
1. When a complex process is underway, opening or switching frontend surfaces becomes slow.
2. New frontend requests appear to wait behind ongoing backend activity.
3. Thread/conversation views can sit in a loading state while active runs are producing lots of events.
4. The issue is especially visible when a run or conversation is actively streaming.

## Repo-Accurate Findings
1. Database calls open fresh `aiosqlite` connections per query/write in `src/loom/state/memory.py`.
2. Event persistence is attached globally to the event bus and persists one event at a time.
3. Notification and run SSE routes often query SQLite again to wait for the just-emitted event to appear durably.
4. SSE route queues are currently unbounded.
5. Workspace overview, approvals, artifacts, and inventory are refreshed together in the desktop app.
6. Conversation and run surfaces continue polling while live streams are active.
7. Some async API routes still perform synchronous `TaskStateManager` IO directly.
8. There is no dominant hot mutex; the main serialization pressure is implicit in SQLite write behavior and event-loop work.

## Goals
1. Make thread and run surfaces responsive even during heavy event production.
2. Keep new requests fast while a run or conversation is active.
3. Preserve durability guarantees appropriate for a local desktop product.
4. Keep architecture compatible with the current sidecar `loomd` model.
5. Reduce queue growth, DB contention, and redundant read traffic.
6. Improve observability so future latency regressions are attributable.

## Non-Goals
1. Migrating Loom to PostgreSQL.
2. Replacing SSE with a totally new transport.
3. Rewriting the event system from scratch.
4. Changing Loom's product model from local sidecar to hosted service.
5. Broad UX redesign outside the performance-related frontend behaviors.

## Design Principles
1. Optimize for the local single-user sidecar case first.
2. Respect SQLite's concurrency model instead of fighting it.
3. Prefer one well-governed writer path over many concurrent tiny writes.
4. Keep the event loop free of blocking filesystem work.
5. Bound queue growth everywhere that can receive bursty event traffic.
6. Prefer coalescing and cache-backed reads over repeated full refreshes.
7. Add measurement before and during rollout so we can prove each phase helps.

## Performance Hypothesis
The likely failure chain today is:
1. A run or conversation emits many events.
2. Each event spawns async persistence work.
3. Persistence opens fresh SQLite connections and commits tiny writes repeatedly.
4. Live stream handlers then poll SQLite to wait for those same rows.
5. The frontend continues polling conversation status/events/messages, activity, and workspace surfaces.
6. Workspace refreshes trigger several expensive read-model requests in parallel.
7. New requests arrive while the server is already busy with event writes, event-follow-up reads, and blocking file loads.

Result:
1. Latency grows even for unrelated frontend actions.
2. "Pipe fills up" behavior appears because producers are effectively unbounded and consumers do redundant work.

## Workstream 0: Measurement and Reproduction Harness

### Problem
We need a repeatable way to measure p50/p95 latency before and after each refactor.

### Plan
1. Add request timing instrumentation for:
   - workspace overview
   - workspace artifacts
   - approvals
   - conversation detail/status/events/messages
   - run detail/timeline/stream
2. Add event-pipeline metrics:
   - event emit rate
   - pending async event-handler task count
   - event persistence queue depth
   - event persistence batch size
   - stream queue depth
3. Add SQLite metrics:
   - write transaction duration
   - busy/retry count
   - connection open count
   - query latency by endpoint family
4. Add a synthetic stress scenario that produces a high event rate from a run or conversation while exercising workspace switching and thread opening.

### Primary Files
1. `src/loom/api/routes.py`
2. `src/loom/events/bus.py`
3. `src/loom/state/memory.py`
4. `src/loom/api/engine.py`
5. `apps/desktop/src/hooks/useWorkspace.ts`
6. `apps/desktop/src/hooks/useConversation.ts`
7. `apps/desktop/src/hooks/useRuns.ts`

### Acceptance
1. We can compare before/after latency and queue depth for active-run scenarios.
2. We can identify whether time is spent in DB writes, DB reads, event fan-out, or frontend refresh storms.

## Workstream 1: SQLite Access Model Refactor

### Problem
Loom currently treats SQLite like a cheap per-operation connection factory, which is hostile to bursty event traffic.

### Plan
1. Replace fresh-per-query connection usage with a governed access model:
   - one dedicated write connection or writer task
   - one or a very small number of long-lived read connections
2. Add a write queue for event persistence instead of one fire-and-forget DB write per event.
3. Batch event inserts in explicit transactions.
4. Preserve ordering guarantees for per-task/per-run sequences.
5. Keep a clear shutdown/drain path so queued writes flush cleanly.
6. Audit other hot write paths for the same anti-pattern:
   - conversation replay events
   - conversation turns
   - task status updates
7. Avoid broad "more threads" fan-out against SQLite; use a controlled writer boundary instead.

### SQLite Behavior Tuning
As part of this refactor, explicitly evaluate and set:
1. `PRAGMA busy_timeout` to absorb brief lock contention instead of failing immediately.
2. `PRAGMA synchronous=NORMAL` for WAL-mode performance, if durability tradeoffs are acceptable for Loom's local-app model.
3. `PRAGMA wal_autocheckpoint` tuning based on measured write volume.
4. `PRAGMA temp_store=MEMORY` where it improves local query behavior safely.
5. `PRAGMA cache_size` and possibly `mmap_size` based on measured desktop workloads and platform support.

These should be introduced as explicit, documented runtime decisions rather than hidden defaults.

### Primary Files
1. `src/loom/state/memory.py`
2. `src/loom/events/bus.py`
3. `src/loom/state/conversation_store.py`
4. `src/loom/api/engine.py`

### Tests
1. Event ordering tests under bursty load.
2. Shutdown/drain tests for queued writes.
3. Concurrency tests ensuring reads remain available while write batches are active.
4. Regression tests for conversation replay sequence allocation.

### Acceptance
1. Event persistence no longer opens a fresh SQLite connection and commit for every event.
2. DB write contention and total connection churn drop materially under load.
3. Active-run frontend latency improves during stress scenarios.

## Workstream 2: Durable Stream Path Simplification

### Problem
Run and notification streams currently turn one event into multiple DB reads while waiting for persistence to catch up.

### Plan
1. Split "durable catch-up replay" from "live event delivery" more cleanly.
2. On initial stream connect, replay persisted rows from SQLite once.
3. After that, prefer direct event-bus payload delivery for live updates instead of repeatedly polling SQLite for the same event.
4. Retain durable catch-up only for reconnect/resume boundaries.
5. Make stream state transitions explicit so "resume from cursor" and "follow live" are separate phases.
6. Audit conversation, run, notification, task, and token streams for the same pattern.

### Queue Governance
1. Add bounded `asyncio.Queue` sizes for SSE streams.
2. Define per-stream drop/coalesce semantics where lossless delivery is not required for every intermediate event.
3. Add queue-depth telemetry and disconnect/slow-consumer diagnostics.

### Primary Files
1. `src/loom/api/routes.py`
2. `src/loom/events/bus.py`

### Tests
1. Stream resume correctness tests.
2. Slow-consumer tests proving queue growth is bounded.
3. Event ordering tests across replay-to-live transition.

### Acceptance
1. Live events do not require repeated DB polling to reach the frontend.
2. Slow consumers cannot grow unbounded in-memory queues.
3. Reconnect behavior remains correct.

## Workstream 3: Frontend Load Shedding and Refresh Discipline

### Problem
The desktop app currently combines SSE with periodic polling and whole-surface refresh bursts.

### Plan
1. Treat SSE as the primary live transport when healthy.
2. Reduce or suspend periodic polling while stream activity is recent and healthy.
3. Replace broad workspace refresh bursts with smaller targeted refreshes:
   - approval count only
   - notifications only
   - active run summary only
   - conversation status only
4. Coalesce notification-triggered refreshes more aggressively.
5. Avoid loading overview, approvals, settings, inventory, and artifacts together unless the view truly needs all of them.
6. Make inactive/background surfaces passive instead of polling continuously.
7. Revisit activity and runtime health polling intervals so they do not compete with active work.

### Primary Files
1. `apps/desktop/src/hooks/useWorkspace.ts`
2. `apps/desktop/src/hooks/useConversation.ts`
3. `apps/desktop/src/hooks/useRuns.ts`
4. `apps/desktop/src/hooks/useDesktopActivity.ts`
5. `apps/desktop/src/hooks/useConnection.ts`

### Tests
1. Hook tests proving polling backs off when stream health is good.
2. Notification refresh tests proving burst events do not trigger repeated full-surface reloads.
3. State-consistency tests for reconnect and stale stream recovery.

### Acceptance
1. Active conversation/run surfaces do not continue high-frequency polling while healthy SSE streams are flowing.
2. Workspace switching and thread opening remain responsive during active runs.
3. UI state remains correct after transient stream interruption.

## Workstream 4: Workspace Read-Model Cost Reduction

### Problem
Workspace overview and artifact surfaces are expensive and currently do more work than necessary.

### Plan
1. Replace "load all then filter in Python" with workspace-scoped queries where feasible.
2. Collapse obvious N+1 read patterns in workspace overview construction.
3. Cache or memoize short-lived workspace summary results where correctness allows.
4. Split heavy secondary data from primary shell data:
   - overview should be cheap
   - artifacts and inventory can load lazily
5. Revisit whether approval counts need full item materialization.
6. Audit artifact aggregation for per-run state-file reads that could be deferred or summarized.

### Primary Files
1. `src/loom/api/routes.py`
2. `src/loom/state/memory.py`
3. `src/loom/state/conversation_store.py`

### Tests
1. Endpoint response-shape parity tests.
2. Workspace overview latency tests on larger fixture datasets.
3. Artifact aggregation correctness tests after any lazy-loading changes.

### Acceptance
1. Workspace overview no longer performs broad full-table scans in hot paths.
2. Secondary surfaces are not loaded unless needed.
3. Workspace home remains fast under active event load.

## Workstream 5: Blocking IO Offload Audit

### Problem
Some async API handlers still do filesystem and YAML work synchronously, which can stall the main event loop.

### Plan
1. Audit all route handlers and hot helper paths for synchronous state-manager access.
2. Move blocking state-file reads/writes behind `asyncio.to_thread` or shared bounded IO helpers.
3. Ensure task creation and task reads do not perform direct blocking file work on the event loop.
4. Keep a bounded executor strategy rather than ad hoc thread fan-out.
5. Document which paths are intentionally synchronous because their cost is negligible.

### Primary Files
1. `src/loom/api/routes.py`
2. `src/loom/state/task_state.py`
3. `src/loom/api/engine.py`

### Tests
1. Route behavior parity tests.
2. Cancellation/time-budget tests around thread-offloaded IO.
3. Event-loop-lag smoke tests under concurrent task reads.

### Acceptance
1. Hot async API routes do not perform direct synchronous state-file loads/saves.
2. Event-loop stalls during task and workspace requests drop measurably.

## Workstream 6: Event Bus Backpressure and Handler Governance

### Problem
The event bus currently allows unbounded async handler task growth.

### Plan
1. Replace unrestricted fire-and-forget persistence fan-out with queued worker semantics.
2. Add pending-task counters and alarm thresholds.
3. Separate handler classes:
   - critical durable handlers
   - best-effort UI/live handlers
   - terminal webhooks
4. Define failure and retry policy per handler class.
5. Keep diagnostic emission from recursively amplifying overload conditions.

### Primary Files
1. `src/loom/events/bus.py`
2. `src/loom/api/engine.py`

### Tests
1. Burst tests ensuring pending task count remains bounded.
2. Failure-path tests for handler retries and diagnostics.
3. Shutdown/drain tests.

### Acceptance
1. Event fan-out no longer creates unbounded async task growth under heavy load.
2. Persistence remains reliable without overwhelming the server.

## Workstream 7: SQLite Runtime Policy and Documentation

### Problem
SQLite behavior is currently under-specified for Loom's local runtime.

### Plan
1. Introduce an explicit SQLite runtime policy section in docs and config comments covering:
   - WAL requirement
   - busy timeout policy
   - synchronous mode
   - checkpoint policy
   - write batching rationale
2. Document why SQLite remains the default for local Loom.
3. Document the threshold conditions that would justify a PostgreSQL track later.
4. Ensure any schema-affecting changes triggered by this work follow the migration-first workflow in `AGENTS.md`.

### Documentation Targets
1. `README.md`
2. `docs/CONFIG.md`
3. `docs/2026-02-28-SYSTEM-TECHNICAL-DESIGN.md`
4. `CHANGELOG.md`

### Acceptance
1. SQLite tuning choices are explicit, documented, and testable.
2. Contributors do not accidentally reintroduce per-event connection churn or broad polling patterns.

## Rollout Strategy

### Phase 1: Instrument and Prove
1. Add measurements and a reproducible stress fixture.
2. Capture baseline p50/p95 latency for thread opening, workspace switching, and run/conversation refresh.

### Phase 2: Backend Load Reduction
1. Implement governed SQLite access.
2. Add event persistence queueing/batching.
3. Remove repeated persisted-row polling from live stream paths.
4. Bound stream queues and event-handler growth.

### Phase 3: Frontend Load Reduction
1. Back off polling when streams are healthy.
2. Replace whole-surface refreshes with targeted refreshes.
3. Load heavy workspace surfaces lazily.

### Phase 4: IO and Read-Model Cleanup
1. Move remaining blocking file IO off the event loop.
2. Reduce overview/artifact read-model cost.
3. Tune SQLite pragmas based on measured results, not guesswork.

### Phase 5: Re-evaluate Architecture
Only after the above:
1. Review whether SQLite still bottlenecks the app.
2. If yes, isolate the remaining pain:
   - write concurrency
   - dataset size
   - cross-process access
   - operational durability needs
3. Open a separate PostgreSQL plan only if the evidence shows SQLite remains the limiting factor after the access-model fixes.

## Risks
1. Batching and delayed persistence can accidentally change ordering or replay semantics if not designed carefully.
2. Live-stream simplification can regress reconnect/resume correctness if replay/live boundaries are fuzzy.
3. Over-aggressive frontend load shedding can make state look stale after stream interruptions.
4. SQLite pragma tuning can trade durability for speed if chosen casually.
5. Added queues/workers can hide overload unless metrics and tests are included from the start.

## Open Questions
1. Which event classes must be strictly durable before the frontend may render them, and which can be live-first?
2. What maximum acceptable delay is allowed between event emit and durable persistence?
3. Should conversation replay events be batched independently from run timeline events?
4. Do we want one global writer queue or separate queues per persistence domain with shared transaction governance?
5. Which desktop surfaces truly require immediate artifact and approval detail refresh versus summary counters?

## Acceptance Criteria for the Overall Program
1. Opening a thread or switching workspace during a heavy active run feels responsive.
2. New frontend requests do not suffer large delays because an existing run is event-heavy.
3. Event persistence remains durable and replayable.
4. Stream and queue growth remain bounded under burst load.
5. SQLite remains the primary local-app database with clearly documented tuning and operational behavior.
