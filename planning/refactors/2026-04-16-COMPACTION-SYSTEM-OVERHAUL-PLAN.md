# Compaction System Overhaul Plan (2026-04-16)

## Objective
Overhaul Loom compaction so it is fast enough to leave on by default, reliable enough to keep requests within model limits, and conservative enough to avoid degrading output quality or triggering compactor overcalling.

This plan expands the older runner-focused compaction work into a single system plan covering:
1. runner request compaction,
2. semantic compactor orchestration,
3. cowork session memory/recall compaction,
4. operator defaults, telemetry, and rollout.

This plan should be treated as the new umbrella plan for compaction work. It builds on and partially supersedes:
1. `planning/refactors/2026-02-23-COMPACTION-POLICY-PLAN.md`
2. `planning/refactors/2026-02-23-DYNAMIC-COMPACTION-UTILITY-PLAN.md`
3. `planning/refactors/2026-03-07-COMPACTION-SUBSYSTEM-RELIABILITY-PLAN.md`

## Why A Full Overhaul Is Needed
Current user feedback is that compaction is often worse when enabled:
1. slower,
2. more prone to extra model calls,
3. more likely to overcall or thrash under pressure,
4. not predictably helpful enough to justify keeping it on.

The current system also has an architectural split:
1. runner compaction in `src/loom/engine/runner/compaction.py` performs multi-stage semantic reduction on live model requests,
2. cowork context management in `src/loom/cowork/session.py` builds a transient recent-tail plus recall-index view,
3. these two systems share goals but do not share a durable compaction boundary, policy model, or success criteria.

As a result, Loom pays compaction cost repeatedly without reusing enough of the value.

## Main Problems To Solve
### 1. Expensive compaction sits too close to the hot path
Runner compaction can invoke semantic compaction across tool args, tool outputs, historical messages, and merged prior-context summaries in `src/loom/engine/runner/compaction.py`.

That is too much LLM-driven work for the critical path. We need cheap deterministic reductions first, and we need the semantic compactor to become the last expensive step, not the first available hammer.

### 2. Compaction work is not reusable enough
Cowork session handling in `src/loom/cowork/session.py` rebuilds a compact archive index rather than persisting a reusable compact boundary and post-boundary summary/memory structure.

We need incremental compaction artifacts that survive across turns and session reloads.

### 3. The system is weak at saying “no”
Today the system can keep oversize or low-gain compaction outputs with warnings. That is better than failure, but it is not good enough for a default-on subsystem.

We need:
1. strict acceptance gates,
2. bounded retries,
3. circuit breakers,
4. deterministic fallback when compaction is unlikely to help.

### 4. Tail-preservation rules are not strong enough
Recent tool/result pairings, thinking blocks, and the active instruction tail need stronger invariants. If compaction degrades the last useful working set, output quality drops sharply.

### 5. Operator confidence is too low
It is hard to explain why compaction fired, what it changed, whether it actually helped, and why the system chose semantic compaction versus a cheaper strategy.

## Goals
1. Under budget: zero semantic compactor calls.
2. Mild pressure: mostly deterministic compaction, minimal extra model calls.
3. Heavy pressure: bounded semantic compaction with hard acceptance gates.
4. Reuse compaction work across turns instead of rebuilding from scratch.
5. Preserve the active tail and tool/result invariants by default.
6. Make “compaction on” faster and more reliable than “compaction off” for long-running sessions.
7. Give operators enough telemetry to understand every compaction decision.

## Non-Goals
1. Do not copy Claude Code prompts or implementation literally.
2. Do not introduce a large permanent config surface unless the defaults are clearly justified.
3. Do not rely on role-fallback compactor behavior as the normal path.
4. Do not require users to delete state or start fresh to benefit from the new system.

## Design Principles
1. Reliability before elegance: fitting requests is a hard runtime requirement.
2. Cheap before smart: use deterministic reductions before LLM summarization.
3. Reuse before recompute: persist compact boundaries and incremental memory where possible.
4. Protect the tail: active instructions, recent tool/result pairs, and latest useful reasoning must survive.
5. Refuse low-value work: if compaction is unlikely to help enough, skip it and take a different path.
6. Observability is part of correctness: every compaction path must be explainable after the fact.

## Target Architecture
Introduce a layered compaction system with four levels:

### Layer 0: Request budgeting and invariants
Before any compaction work:
1. estimate full request size using messages, tool schemas, and protocol envelope,
2. identify protected tail ranges,
3. compute required token reduction and max allowed compaction spend for this turn.

### Layer 1: Deterministic microcompaction
Apply cheap non-LLM transformations first:
1. replace stale read/search/web/shell outputs with concise deterministic summaries or pointers,
2. strip or stub payloads that can be regenerated or re-injected,
3. compress repeated tool chatter into grouped placeholders,
4. reduce duplicated tool-schema prompt content before touching live conversation text.

### Layer 2: Incremental session memory
Maintain reusable compacted memory/checkpoints:
1. summarize only newly stale ranges,
2. carry a “last summarized boundary” forward,
3. keep the recent tail raw,
4. use prior compact artifacts directly instead of re-summarizing whole history.

### Layer 3: Bounded semantic compaction
Only when Layers 0-2 are insufficient:
1. run semantic compaction on carefully selected lanes,
2. enforce post-compaction fit validation,
3. reject low-gain or still-oversize outputs unless a defined degraded mode allows them,
4. stop after bounded retries and enter a deterministic fallback path.

## Shared Vocabulary And State Model
Unify runner and cowork around the same concepts:
1. `protected_tail`: newest instructions, active write context, recent tool/result pairs, newest thinking block if preserved.
2. `stale_tail`: recent but compactable context outside the protected set.
3. `session_memory`: reusable compact summary/checkpoint artifacts for older history.
4. `compact_boundary`: the point in history below which raw replay is no longer required for normal turns.
5. `microcompactable_payload`: tool or machine-generated content eligible for deterministic reduction.

## Workstream 1: Build A Unified Compaction Control Plane
### Files
1. `src/loom/engine/runner/compaction.py`
2. `src/loom/engine/runner/core.py`
3. `src/loom/cowork/session.py`
4. `src/loom/config.py`
5. `tests/test_orchestrator.py`
6. new focused compaction tests if needed

### Changes
1. Define a single request-budget model used by both runner compaction and cowork context assembly.
2. Add shared helpers for:
   - protected-tail detection,
   - stale-range selection,
   - compact-boundary metadata,
   - full-request deficit estimation.
3. Stop treating runner compaction and cowork archive shaping as unrelated policies.

### Acceptance Criteria
1. Runner and cowork can describe context using the same boundary and tail terms.
2. Every compaction decision begins from the same budget and protection model.

## Workstream 2: Deterministic Microcompaction First
### Files
1. `src/loom/engine/runner/compaction.py`
2. `src/loom/cowork/session.py`
3. `src/loom/engine/runner/telemetry.py`
4. `tests/test_orchestrator.py`
5. focused tests for tool payload reduction

### Changes
1. Add a microcompaction pass before `SemanticCompactor` is considered.
2. Target old tool outputs first, especially:
   - file reads,
   - web/search results,
   - shell output,
   - machine-generated JSON blobs,
   - repeated tool chatter that can be collapsed losslessly enough for context usage.
3. Convert recoverable payloads into:
   - pointer + digest,
   - short deterministic summary,
   - grouped placeholder form.
4. Add explicit “do not microcompact” rules for active tool/result pairs and recent interactive state.

### Acceptance Criteria
1. Common long-running sessions reduce pressure without any extra model call in the normal case.
2. Old tool-heavy context shrinks materially before semantic compaction is attempted.

## Workstream 3: Incremental Session Memory And Reusable Checkpoints
### Files
1. `src/loom/cowork/session.py`
2. `src/loom/engine/runner/compaction.py`
3. `src/loom/engine/semantic_compactor/core.py`
4. `src/loom/engine/semantic_compactor/pipeline.py`
5. storage/state modules as needed
6. tests for session reload/resume behavior

### Changes
1. Replace transient archive-index rebuilding with explicit compact boundaries plus reusable session-memory payloads.
2. Track:
   - last summarized turn/message boundary,
   - summary/checkpoint digest,
   - preserved-tail metadata,
   - optional per-lane compact artifacts.
3. Summarize only newly stale slices instead of re-merging broad prior-conversation ranges every turn.
4. Feed session memory directly into context assembly for both runner and cowork.

### Acceptance Criteria
1. Repeated turns do not trigger repeated full-history re-compaction.
2. Reloaded sessions can reuse prior compaction work immediately.

## Workstream 4: Stronger Tail Invariants
### Files
1. `src/loom/engine/runner/compaction.py`
2. `src/loom/cowork/session.py`
3. `tests/test_orchestrator.py`
4. tests covering cowork session history assembly

### Changes
1. Expand the protected-tail model to preserve:
   - newest task instructions and steering,
   - newest assistant non-tool response,
   - recent `tool_use` / `tool_result` pairs,
   - current active work context,
   - most relevant/latest thinking block when thinking preservation is enabled.
2. Prevent compact boundaries from splitting required tool/result or reasoning groupings.
3. Ensure deterministic microcompaction also respects these pairing rules.

### Acceptance Criteria
1. Compaction never produces orphaned tool messages or broken recent reasoning context.
2. Output quality remains stable in tool-heavy and drafting-heavy sessions.

## Workstream 5: Semantic Compactor Guardrails And Acceptance Gates
### Files
1. `src/loom/engine/semantic_compactor/core.py`
2. `src/loom/engine/semantic_compactor/pipeline.py`
3. `src/loom/engine/runner/compaction.py`
4. `src/loom/config.py`
5. `tests/test_semantic_compactor.py`
6. `tests/test_orchestrator.py`

### Changes
1. Add a preflight feasibility gate before semantic compaction:
   - if required reduction is implausible, skip semantic compaction.
2. Add strict post-compaction acceptance logic:
   - reject outputs that are still too large unless a degraded mode explicitly allows them,
   - reject no-gain outputs,
   - track overshoot and low-gain streaks.
3. Add a circuit breaker after repeated failed compaction attempts.
4. Add bounded per-turn and per-label compactor call budgets.
5. Default to a dedicated compactor model/role where configured.
6. Disable or sharply limit role fallback for compaction by default.

### Acceptance Criteria
1. Compaction churn is bounded.
2. The system stops making expensive compactor calls that do not materially improve fit.
3. Role fallback is no longer the default escape hatch for compaction.

## Workstream 6: Deterministic Fallback Paths For Unfit Requests
### Files
1. `src/loom/engine/runner/compaction.py`
2. `src/loom/engine/runner/core.py`
3. `tests/test_orchestrator.py`

### Changes
1. Define an explicit fallback ladder when request pressure remains too high after bounded compaction:
   - strip duplicated tool-schema prompt material,
   - prune tool schemas to the minimum viable set,
   - use pointer forms for old machine payloads,
   - prefer preserved-tail + session-memory composition over large raw history,
   - only then use bounded degraded mode if required.
2. Make fallback decisions deterministic and observable.
3. Remove “keep trying semantic compaction” as the default last resort.

### Acceptance Criteria
1. Pathological long sessions converge to a bounded fallback behavior.
2. Small-context models avoid endless compaction retry loops.

## Workstream 7: Storage And Schema Decisions For Persistent Boundaries
### Files
1. state/storage modules to be determined
2. `src/loom/cowork/session.py`
3. schema/migration files only if persistence requires database changes
4. migration tests and docs if schema changes are introduced

### Changes
1. Decide where compact boundaries and session-memory checkpoints live:
   - existing persisted session state if already suitable,
   - or new DB-backed fields/tables if necessary.
2. Keep the first slice simple: prefer reusing current persisted session mechanisms if possible.
3. If schema changes are needed, follow the repo migration-first workflow:
   - update `src/loom/state/schema.sql`,
   - update `src/loom/state/schema/base.sql`,
   - add migration steps,
   - register migrations,
   - add upgrade tests,
   - update docs and changelog.

### Acceptance Criteria
1. Persistent compact state survives restart without fragile ad hoc rebuilding.
2. Any schema change lands with full migration coverage.

## Workstream 8: Telemetry, Operator UX, And Defaults
### Files
1. `src/loom/engine/runner/telemetry.py`
2. `src/loom/config.py`
3. `docs/CONFIG.md`
4. operator docs/changelog as needed
5. tests for config and telemetry payloads

### Changes
1. Add telemetry for:
   - request deficit before compaction,
   - microcompaction operations applied,
   - semantic compactor call count,
   - accepted vs rejected compaction outputs,
   - compaction circuit-breaker trips,
   - protected-tail size,
   - session-memory reuse hits,
   - terminal compaction state (`fit`, `fit_via_memory`, `fit_via_microcompact`, `degraded_fit`, `unfit`).
2. Keep config surface lean, but add enough control for rollout:
   - policy mode,
   - dedicated compactor model override,
   - semantic compaction call budget,
   - circuit-breaker thresholds.
3. Make the target default “safe and boring” enough that users do not need to turn compaction off.

### Acceptance Criteria
1. Operators can explain why compaction fired and whether it helped.
2. The default configuration materially reduces long-run pain without hand tuning.

## Testing Plan
1. Unit tests for full-request deficit estimation.
2. Unit tests for protected-tail selection and tool/result pairing invariants.
3. Unit tests for deterministic microcompaction of stale tool payloads.
4. Unit tests for semantic compaction feasibility and rejection logic.
5. Unit tests for circuit-breaker and no-gain behavior.
6. Session reload tests proving compact boundaries and session-memory reuse survive reload.
7. Integration tests for:
   - long tool-heavy threads,
   - drafting-heavy sessions,
   - small-context model limits,
   - compaction-off vs compaction-on comparative behavior.
8. Regression tests ensuring under-budget requests do not invoke the compactor.

## Rollout Plan
### Phase A: Instrument And Measure
1. Land the unified budget model and telemetry without changing default behavior.
2. Measure where compaction time and model calls are going today.

### Phase B: Ship Microcompaction And Guardrails Behind A New Mode
1. Add a new policy mode, for example `budgeted_v2`.
2. Enable deterministic microcompaction first.
3. Add semantic compactor acceptance gates and circuit breakers.

### Phase C: Add Session Memory And Persistent Boundaries
1. Introduce reusable compact checkpoints.
2. Switch cowork to boundary-aware reuse instead of transient archive rebuilding.

### Phase D: Promote The New Mode
1. Benchmark against `off`, `legacy`, and `tiered`.
2. Promote only if:
   - latency improves,
   - compactor calls drop,
   - success rate does not regress,
   - user-perceived quality remains stable.

## Success Metrics
1. Compactor model calls per long-running turn drop materially relative to current `tiered`.
2. Median compaction latency drops materially for tool-heavy sessions.
3. Under-budget turns perform zero semantic compactor calls.
4. Repeated long sessions show checkpoint/session-memory reuse instead of whole-history rework.
5. Users can leave compaction on without feeling compelled to disable it for speed or stability.

## Risks And Mitigations
1. Risk: persistent checkpoint state becomes hard to reason about.
   Mitigation: keep compact boundary metadata explicit and test reload behavior heavily.
2. Risk: deterministic microcompaction throws away nuance needed later.
   Mitigation: use pointer+digest forms and preserve a strong recent tail.
3. Risk: stricter acceptance gates increase `unfit` outcomes.
   Mitigation: add deterministic fallback paths before accepting failure.
4. Risk: config surface sprawls.
   Mitigation: ship with a small set of rollout knobs and hide experimental tuning.

## Recommended First Slice
1. Add unified full-request deficit estimation.
2. Add deterministic microcompaction for old tool outputs.
3. Add semantic compactor acceptance gates and circuit breakers.
4. Add telemetry proving whether compaction actually helped.

That first slice should already make compaction noticeably less annoying before we tackle persistent session-memory boundaries.
