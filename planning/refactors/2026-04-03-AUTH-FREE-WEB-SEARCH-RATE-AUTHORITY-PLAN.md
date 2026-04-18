# Auth-Free Web Search Rate Authority Plan (2026-04-03)

## Objective
Promote `DuckDuckGo HTML` to the primary auth-free search provider while preventing Loom from hammering public search surfaces. The implementation must move provider pacing and cooldown state out of process-local memory and into one authoritative shared store, and it must account for Loom tool runtime limits by starting provider pacing only when a request is actually eligible to dispatch.

## Executive Summary
The current `web_search` rework improved parser quality and reduced burstiness, but it still has two structural problems:
1. Provider rate state is process-local (`Semaphore`, cooldown, cache metadata in memory), so multiple Loom runs can still collectively overload `DuckDuckGo`.
2. The current backend treats provider pacing as an in-process concern rather than a persisted shared authority, which conflicts with Loom's broader "one domain, one authority" direction.

This plan moves provider budget state into SQLite, introduces a shared provider-lease and pacing system, and changes the request lifecycle so provider timers are consumed only when a tool execution is ready to send a request. That lets us make `DuckDuckGo` primary and `Bing` secondary without relying on stealth or anti-detection evasion.

## Why This Plan
We have two goals that must be satisfied at the same time:
1. Better auth-free search quality on niche and proper-name queries.
2. Lower risk of temporary rate limits or soft-blocks from public search providers.

Recent live tests showed:
1. `DuckDuckGo HTML` performs better than `Bing HTML` for some entity-heavy and Canadian industry queries.
2. `Bing HTML` still performs well for straightforward general queries.
3. The current implementation can reduce bursts within one process, but it cannot enforce global pacing across all Loom activity because the state is not shared.

Promoting `DuckDuckGo` without fixing shared pacing would reintroduce the original failure mode. Promoting it after fixing shared pacing gives us a better auth-free default while keeping provider pressure bounded.

## Hard Constraints
1. One authoritative state store per provider-budget domain.
2. No second authority in process-local memory for provider cooldown or pacing decisions.
3. Tool runtime limits must be respected.
4. Provider pacing must start at actual dispatch time, not when a tool is queued or waiting to run.
5. No browser automation or identity-rotation work intended to bypass anti-bot systems.
6. Schema changes must follow the repo's migration-first workflow.

## Current Baseline In Repo
Current auth-free search behavior lives in:
1. [search_backend.py](/Users/sfw/Development/loom/src/loom/tools/search_backend.py)
2. [web_search.py](/Users/sfw/Development/loom/src/loom/tools/web_search.py)

Current protections:
1. Global in-process concurrency cap.
2. Per-provider in-process semaphores.
3. In-memory cache and in-flight dedupe.
4. In-memory cooldowns after `403`, `429`, `5xx`, or anti-bot markers in returned HTML.

Current gaps:
1. No shared rate authority across Loom processes.
2. No persisted `next_allowed_at` or provider lease.
3. No minimum dispatch spacing enforced across the whole app.
4. No runtime-budget-aware wait-or-fallback decision.
5. No migration-backed storage for provider pacing state.

## Problem Statement

### 1) Split-authority provider pacing
Right now the effective truth for provider pacing is split across:
1. Process-local semaphores.
2. Process-local cooldown values in `SearchRegistry`.
3. Per-process caches.

Implication:
1. A second Loom process can ignore the first process's provider cooldown and hit `DuckDuckGo` anyway.

### 2) Dispatch timing is not tied to real execution start
If we introduce wait intervals naively, we risk starting the interval while:
1. The tool is still queued in Loom.
2. The tool is waiting for a semaphore unrelated to provider dispatch.
3. The tool has not yet decided which provider it can afford to wait for.

Implication:
1. We would burn pacing budget before an actual request is sent, which both wastes runtime budget and distorts provider spacing.

### 3) Public-provider pressure should be shaped before hard blocks
The current implementation is good at backing off after an overt provider failure, but the safer design is to avoid concentrated bursts before:
1. HTTP `429`
2. HTTP `403`
3. CAPTCHA or challenge pages

Implication:
1. We need preemptive pacing, not only reactive cooldown.

## Goals
1. `DuckDuckGo` becomes the primary provider only after shared pacing is in place.
2. Provider pacing and cooldown state live in one authoritative SQLite-backed store.
3. A request only consumes provider budget when it is about to dispatch.
4. If waiting for `DuckDuckGo` would exceed the tool's remaining runtime budget, the backend immediately falls back to `Bing`.
5. The backend remains auth-free and does not use stealth automation.
6. The implementation remains testable offline with mocked network calls.

## Non-Goals
1. Building a stealth browser stack to disguise scraping.
2. Recreating search-engine ranking logic from scratch.
3. Solving all result-quality issues via custom ranking.
4. Creating a distributed coordination system outside Loom's existing state layer.
5. Requiring users to delete or reset their database after upgrade.

## Design Principles
1. One domain, one authority.
2. Dispatch-time accounting, not queue-time accounting.
3. Shared pacing before reactive cooldown.
4. Fast fallback when runtime budget is too small to wait safely.
5. Provider-specific policies are data-driven.
6. Migration-first for any persistent state changes.

## Authority Decision

### Domain: Search provider pacing and cooldown state
Authority:
1. SQLite-backed provider state owned by Loom's state subsystem.

Projection:
1. In-memory snapshots held by `web_search` during a single tool execution.

Rules:
1. In-memory values may cache a fresh read for the duration of one request attempt, but they are not authoritative.
2. Every provider dispatch decision must consult authoritative state first.
3. Every cooldown or dispatch-spacing update must write back through the authoritative store.
4. The provider store must be the only place that answers:
   - `next_allowed_at`
   - `cooldown_until`
   - current in-flight lease owner or lease expiry
   - consecutive failure counters
   - soft-block counters

## Proposed Architecture

### 1) Introduce a persisted provider-state table
Add a new SQLite table for provider pacing state, for example `search_provider_state`.

Recommended fields:
1. `provider TEXT PRIMARY KEY`
2. `enabled INTEGER NOT NULL DEFAULT 1`
3. `priority INTEGER NOT NULL`
4. `min_interval_seconds REAL NOT NULL DEFAULT 0`
5. `next_allowed_at REAL NOT NULL DEFAULT 0`
6. `cooldown_until REAL NOT NULL DEFAULT 0`
7. `lease_owner TEXT NOT NULL DEFAULT ''`
8. `lease_expires_at REAL NOT NULL DEFAULT 0`
9. `consecutive_failures INTEGER NOT NULL DEFAULT 0`
10. `soft_block_count INTEGER NOT NULL DEFAULT 0`
11. `last_status_code INTEGER`
12. `last_started_at REAL NOT NULL DEFAULT 0`
13. `last_finished_at REAL NOT NULL DEFAULT 0`
14. `last_success_at REAL NOT NULL DEFAULT 0`
15. `updated_at REAL NOT NULL DEFAULT 0`

Intent:
1. `next_allowed_at` enforces minimum spacing.
2. `cooldown_until` enforces hard backoff windows after failures or challenge pages.
3. `lease_owner` and `lease_expires_at` prevent two Loom executions from dispatching the same provider simultaneously.

### 2) Create a provider-budget store API
Add one store-level API that owns all provider-budget reads and writes.

Recommended operations:
1. `get_provider_state(provider)`
2. `acquire_dispatch_slot(provider, *, now, runtime_deadline, lease_owner)`
3. `mark_dispatch_started(provider, *, now, min_interval_seconds)`
4. `mark_dispatch_success(provider, *, now, latency_seconds)`
5. `mark_dispatch_failure(provider, *, now, status_code, soft_block)`
6. `release_expired_leases(now)`

Rules:
1. These operations must execute under one SQLite transaction per decision/update.
2. The search tool must not mutate provider pacing state directly.

### 3) Dispatch-time pacing lifecycle
Provider selection should follow this sequence:
1. Tool execution begins.
2. Compute the remaining runtime budget for this tool invocation.
3. Query provider state from the authoritative store.
4. For `DuckDuckGo` first:
   - if `cooldown_until > now`, skip to `Bing`
   - else compute required wait as `max(lease wait, next_allowed_at - now)`
   - if required wait exceeds the remaining runtime budget reserve, skip to `Bing`
   - else wait until allowed
5. Only when the tool is ready to send the request:
   - atomically acquire the provider lease
   - atomically advance `next_allowed_at = dispatch_time + min_interval_seconds`
   - mark `last_started_at`
6. Dispatch the actual HTTP request.
7. On completion:
   - write success or failure back to the authoritative store
   - release or expire the lease

This is the key runtime rule:
1. The spacing timer starts at dispatch, not while the tool is queued and not while it is merely waiting for permission to run.

### 4) Promote `DuckDuckGo` to primary only after shared pacing lands
Target provider order:
1. `DuckDuckGo`
2. `Bing`

But the promotion is conditional on:
1. Shared state store implemented.
2. Minimum DDG interval configured.
3. Runtime-budget-aware fallback implemented.
4. Migration and regression coverage in place.

### 5) Keep the registry, but make it authority-aware
`SearchRegistry` should remain the provider catalog and policy surface, but not the authority for live pacing state.

Registry responsibilities:
1. Ordered provider set and default priorities.
2. Static provider metadata:
   - search URL
   - default min interval
   - default cooldown duration
   - parser/transport adapter
3. Selection policy.

Store responsibilities:
1. Live cooldowns.
2. Dispatch leases.
3. Next-allowed times.
4. Failure counters.

### 6) Add soft-block handling before hard bans
Continue to treat obvious anti-bot signals as provider failures, but persist them centrally.

Soft-block signals:
1. CAPTCHA or challenge text in returned HTML.
2. Result pages with obvious provider-specific warning banners.
3. Sudden empty-result pages from a provider that normally returns results for the same class of query.

Recommended policy:
1. Soft-block increments `soft_block_count`.
2. Repeated soft-blocks within a window open a cooldown even if HTTP status is `200`.

### 7) Add budget-aware fallback
The provider chooser should not wait indefinitely for `DuckDuckGo`.

Required behavior:
1. Compute a minimum completion reserve for the tool.
2. If waiting for DDG would consume that reserve, skip DDG and use `Bing`.
3. This decision must be made from authoritative provider state plus the tool's actual remaining runtime.

## SQLite and Migration Plan
This change likely requires a new table and therefore must follow the repo's migration-first workflow.

Required implementation steps:
1. Update `src/loom/state/schema.sql`.
2. Update `src/loom/state/schema/base.sql`.
3. Add migration step(s) under `src/loom/state/migrations/steps/`.
4. Register them in `src/loom/state/migrations/registry.py`.
5. Add migration and upgrade tests.
6. Update operator docs and changelog.

No implementation should ship with runtime reliance on the new provider-state table until migration coverage exists.

## Workstreams

### W1: Authority Contract and Schema Design
1. Define the authoritative domain for search-provider state.
2. Finalize the `search_provider_state` schema.
3. Document provider fields and state transitions.

Exit criteria:
1. One clear authority model exists for provider pacing and cooldown.
2. Schema shape is stable enough for migration work.

### W2: Persistent Store and Lease Semantics
1. Build a store module for provider-state transactions.
2. Implement lease acquisition and expiry semantics.
3. Add `next_allowed_at` and cooldown update routines.

Exit criteria:
1. Two concurrent Loom executions cannot both dispatch the same provider simultaneously.
2. Shared cooldown and pacing survive process boundaries.

### W3: Runtime-Budget-Aware Dispatch Flow
1. Thread actual remaining tool runtime into provider dispatch decisions.
2. Ensure wait time is computed only after the tool is actively executing.
3. Add wait-or-fallback logic for `DuckDuckGo`.

Exit criteria:
1. Provider timers begin at dispatch time.
2. Requests do not spend most of their runtime budget waiting for DDG.

### W4: Provider Promotion and Selection Policy
1. Flip provider order to `DuckDuckGo` then `Bing`.
2. Preserve the registry abstraction and move live state queries to the authoritative store.
3. Keep cache and in-flight dedupe as performance helpers, but do not let them become a second authority for provider pacing.

Exit criteria:
1. DDG is primary in normal operation.
2. Bing remains immediate fallback under cooldown, lease contention, or budget constraints.

### W5: Soft-Block Detection and Cooldown Policy
1. Centralize HTML challenge detection.
2. Persist soft-block counters and last-failure context.
3. Define escalation thresholds for soft-block-driven cooldown.

Exit criteria:
1. The backend can back off before repeated hard bans.
2. Soft-blocks are visible and diagnosable.

### W6: Observability and Operator Tooling
1. Add logging and diagnostics for provider-state decisions.
2. Add a doctor/status view for search provider state if useful.
3. Expose enough information to debug why DDG was skipped or cooled down.

Exit criteria:
1. Operators can inspect provider state without guessing.
2. Failures and waits are attributable to explicit state transitions.

### W7: Tests and Migration Coverage
1. Add unit tests for lease, pacing, cooldown, and fallback behavior.
2. Add migration/upgrade coverage for the new table.
3. Add concurrency tests proving that shared store state prevents double-dispatch.
4. Add runtime-budget tests proving that wait starts at execution time and not before.

Exit criteria:
1. Schema and behavior are both covered by tests.
2. Upgrade paths do not require DB reset.

## Request Lifecycle Reference
This is the target execution flow for one `web_search` call:

1. Tool starts `execute(...)`.
2. Tool computes `runtime_deadline`.
3. Search backend normalizes query and checks result cache.
4. Backend asks authoritative provider-state store for `DuckDuckGo` eligibility.
5. Store returns one of:
   - `dispatch_now`
   - `wait_until`
   - `skip_due_to_cooldown`
   - `skip_due_to_lease`
6. Backend compares `wait_until` with actual remaining runtime budget.
7. If affordable, wait.
8. At actual dispatch time, atomically:
   - acquire provider lease
   - set `next_allowed_at`
   - stamp `last_started_at`
9. Send request.
10. Persist success or failure outcome.
11. If DDG is unavailable or too expensive to wait for, repeat with `Bing`.

## Test Plan

### Unit Tests
1. Provider-state transaction tests for `next_allowed_at`, cooldown, and lease expiry.
2. Selection tests proving DDG is primary only when eligible and affordable.
3. Soft-block detection tests for HTML challenge markers.
4. Runtime-budget tests covering:
   - enough time to wait
   - not enough time to wait
   - immediate Bing fallback

### Migration Tests
1. Fresh bootstrap includes `search_provider_state`.
2. Existing DB upgrades successfully to include the new table.
3. Seed/default rows are created deterministically if required.

### Concurrency Tests
1. Two concurrent search executions cannot both dispatch DDG at once.
2. Lease expiry recovers from abandoned dispatch state.
3. Process-local caches do not violate shared pacing authority.

### Integration Tests
1. Mocked end-to-end `web_search` tests exercising:
   - DDG success
   - DDG wait then success
   - DDG wait exceeds runtime budget then Bing fallback
   - DDG soft-block then Bing fallback
   - DDG hard cooldown respected across executions

## Risks
1. Overly conservative DDG pacing could make the tool feel slow.
2. Lease logic can deadlock or strand capacity if expiry semantics are wrong.
3. Adding SQLite coordination in the request path could increase complexity or lock contention.
4. Cross-process timing bugs could create subtle fairness issues if transaction boundaries are wrong.

## Recommended Initial Defaults
1. `DuckDuckGo min_interval_seconds`: start conservative, for example `2.0` to `3.0`.
2. `DuckDuckGo cooldown`: keep longer than Bing.
3. `Bing min_interval_seconds`: lower than DDG, but still non-zero if we want symmetric shaping.
4. Reserve enough runtime budget so the tool can still complete fetch/parse work after waiting.

These defaults should be configurable, but the first implementation should ship with safe built-in values.

## Recommended Rollout Sequence
1. Land schema and provider-state store.
2. Wire shared pacing into the current `Bing -> DuckDuckGo` backend.
3. Verify state, leases, and runtime-budget behavior under tests.
4. Flip order to `DuckDuckGo -> Bing`.
5. Re-run live testing against a small query set before broader rollout.

## Success Criteria
1. `DuckDuckGo` can be primary without bursty request patterns across Loom runs.
2. Provider state is consistent across processes because it has one authoritative store.
3. Wait windows are charged only when a tool execution is actually ready to dispatch.
4. `web_search` falls back to `Bing` quickly when waiting for DDG would violate runtime constraints.
5. Migration and regression coverage are in place before runtime dependence on the new schema ships.
