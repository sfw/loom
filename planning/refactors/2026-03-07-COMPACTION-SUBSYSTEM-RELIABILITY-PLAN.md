# Compaction Subsystem Reliability Plan (2026-03-07)

## Objective
Make compaction reliably keep runner requests within model limits, especially for small-context models, while preserving high-value execution context and avoiding compaction thrash.

This plan supersedes the prior "pressure-stage only" framing by adding a request-budgeted controller with deterministic guardrails. It incorporates the strongest ideas from:
1. `2026-02-23-COMPACTION-POLICY-PLAN.md`
2. `2026-02-23-DYNAMIC-COMPACTION-UTILITY-PLAN.md`
3. `2026-02-23-FILETYPE-INGEST-AND-OVERFLOW-FALLBACK-PLAN.md`

## Why The Current Subsystem Is Still Unreliable
1. First-turn blind spot: compaction short-circuits on short history and the initial prompt turn is always protected, so initial oversize requests are not preventable by runner compaction.
2. Budget mismatch: pressure decisions currently use message-only estimates, but provider requests include message payload plus tool schemas and envelope overhead.
3. Prompt/schema duplication: tool schemas are injected into prompt text and also sent in provider `tools`, wasting context budget.
4. No strict fit guarantee: semantic compactor may keep over-target output by warning path, which can leave requests above safe budget.
5. Reactive overflow handling: overflow fallback is triggered only after provider rejection and mainly rewrites tool payloads, not oversized first-turn prompt composition.

## Design Principles
1. Reliability first: request-fitting is a hard runtime requirement.
2. Small-context-first: policy must work for 4k/8k windows, not only large windows.
3. Deterministic control plane: model advisory is optional and never required for correctness.
4. Utility-aware compression: compact lowest-value, most-recoverable context first.
5. Anti-thrash defaults: bounded compactor calls, cooldowns, feasibility gates, cache reuse.
6. No hard truncation for protected instruction lanes; deterministic clipping is allowed only for low-value machine payload lanes.
7. Full observability: every skip/apply decision emits deterministic reason codes.

## Target Policy Mode
Add a new mode:
1. `runner_compaction_policy_mode = "budgeted"`

Keep existing modes for compatibility:
1. `legacy`
2. `tiered`
3. `off`

Rollout intent:
1. Ship `budgeted` behind flag.
2. Dogfood and benchmark.
3. Promote to default once stability/quality criteria are met.

## Budget Model (Core Change)
Use full request estimation, not message-only estimation.

Per invocation, compute:
1. `request_est_tokens(messages, tools, payload envelope)` via `collect_request_diagnostics`.
2. `effective_context_budget_tokens`:
   - starts from `limits.runner.max_model_context_tokens`
   - bounded by model/provider known limits when available
   - reserves output headroom and protocol/tool envelope headroom
3. `target_budget_tokens = floor(effective_context_budget_tokens * safety_ratio)`
4. `required_delta_tokens = request_est_tokens - target_budget_tokens`

Policy behavior:
1. `required_delta_tokens <= 0`: no additional compaction.
2. `required_delta_tokens > 0`: run operation planner until deficit is cleared or bounded terminal state is reached.

## Lane Model (From Dynamic Utility Plan, Hardened)
Classify context into lanes with deterministic invariants:
1. `instruction`: current subtask instructions, acceptance criteria, active TODO guard.
2. `active_write`: most recent drafting/output-shaping narrative.
3. `source_evidence`: evidence summaries and citation-critical context.
4. `tool_trace`: assistant tool calls and tool results.
5. `historical_chat`: older conversational narrative.
6. `schema_overhead`: prompt tool-list text and provider tool schema set.

Protection rules:
1. Never compact away latest critical instructions.
2. Never remove active-write lane until tool-trace and schema-overhead reductions are exhausted.
3. Deterministic clipping allowed only in low-value machine payload lanes (`tool_trace`, selected `schema_overhead`) under emergency fit path.

## Operation Set (Ordered By Harm/Recoverability)
1. `reduce_prompt_tool_listing`:
   - move prompt tool section from full schema to names+1-line descriptions.
2. `prune_provider_tool_schemas`:
   - select minimal tool subset for current subtask intent; expand on demand.
3. `compact_tool_args`:
   - compact older assistant tool-call arguments first.
4. `compact_tool_outputs`:
   - compact older tool outputs with artifact/pointer preference.
5. `compact_historical_chat`:
   - compact stale narrative context.
6. `incremental_checkpoint_merge`:
   - merge only newly stale slices, not full-history remerge each turn.
7. `low_value_deterministic_cap`:
   - deterministic last-mile cap for low-value machine payload lanes only.
8. `overflow_rewrite_preflight`:
   - apply overflow rewrite proactively before provider call when deficit remains.

## Feasibility And Anti-Thrash Controls
1. Per-iteration compactor call budget.
2. Per-label cooldown turns after overshoot/no-gain streak.
3. Required-ratio feasibility gate:
   - if expected gain is implausible, skip expensive semantic compaction and choose alternate operations.
4. Cache reuse by `(digest, target, label)` for stable inputs.
5. Timeout-aware policy:
   - when near deadline, prefer deterministic low-cost operations over new semantic compaction calls.

## Workstreams

### Workstream 1: Request-Budget Preflight And First-Turn Fit
### Files
1. `<repo-root>/src/loom/engine/runner.py`
2. `<repo-root>/src/loom/models/request_diagnostics.py`
3. `<repo-root>/tests/test_orchestrator.py`

### Changes
1. Add request preflight budget computation before each model call, including first iteration.
2. Replace message-only pressure decision input with full request estimates (`messages + tools + envelope`).
3. Add first-turn fit path so initial oversized prompt does not bypass compaction controls.

### Acceptance Criteria
1. First model invocation fits budget or enters deterministic bounded fallback path.
2. Compaction decisions reference full-request deficit, not message-only ratio.

### Workstream 2: Prompt/Schema De-Duplication
### Files
1. `<repo-root>/src/loom/prompts/assembler.py`
2. `<repo-root>/src/loom/engine/runner.py`
3. `<repo-root>/tests/test_prompts.py`
4. `<repo-root>/tests/test_orchestrator.py`

### Changes
1. Change executor prompt tool section to concise listing mode by default in `budgeted` mode.
2. Keep full parameter schema in provider `tools` payload only.
3. Add configurable listing policy (`full|compact|names_only`) for controlled rollback.

### Acceptance Criteria
1. Prompt size decreases materially without loss of tool usability.
2. No duplicate full schema content in both prompt and `tools`.

### Workstream 3: Lane Classifier And Candidate Graph
### Files
1. `<repo-root>/src/loom/engine/runner.py`
2. `<repo-root>/tests/test_orchestrator.py`

### Changes
1. Implement lane assignment (`instruction`, `active_write`, `source_evidence`, `tool_trace`, `historical_chat`, `schema_overhead`).
2. Attach runtime metadata: age, recoverability, prior compaction gain, protection status.
3. Generate deterministic candidate operations with estimated token gain and harm score.

### Acceptance Criteria
1. Candidate graph is deterministic for equivalent runtime state.
2. Protected lanes remain untouched until lower-harm lanes are exhausted.

### Workstream 4: Budgeted Operation Planner
### Files
1. `<repo-root>/src/loom/engine/runner.py`
2. `<repo-root>/tests/test_orchestrator.py`

### Changes
1. Replace fixed stage pipeline with deficit-driven operation planner.
2. Preserve ordering invariants from tiered policy (tool trace before protected narrative), but make scope/targets adaptive to deficit.
3. Introduce incremental checkpoint merge (delta-based) instead of repeated full-history merge.

### Acceptance Criteria
1. Planner stops as soon as deficit is cleared.
2. No repeated full-history recompression under steady-state iterations.

### Workstream 5: Semantic Compactor Guardrails
### Files
1. `<repo-root>/src/loom/engine/semantic_compactor.py`
2. `<repo-root>/src/loom/engine/runner.py`
3. `<repo-root>/src/loom/config.py`
4. `<repo-root>/tests/test_semantic_compactor.py`
5. `<repo-root>/tests/test_orchestrator.py`

### Changes
1. Add compactor feasibility checks and per-label cooldown integration.
2. Carry forward warning/overshoot telemetry into planner gain model.
3. Add bounded emergency cap path for low-value machine payload lanes only.

### Acceptance Criteria
1. Compactor churn is bounded in long runs.
2. Protected instruction lanes never receive hard truncation.
3. Low-value emergency cap path is deterministic and explicitly logged.

### Workstream 6: Overflow Fallback Integration
### Files
1. `<repo-root>/src/loom/engine/runner.py`
2. `<repo-root>/tests/test_orchestrator.py`

### Changes
1. Move overflow rewrite from purely reactive mode to preflight-capable operation when deficit remains.
2. Expand fallback eligibility to include first-turn composition constraints (not only older tool messages).
3. Emit explicit skip reasons when fallback cannot improve fit.

### Acceptance Criteria
1. Overflow retry loops drop in frequency for small-context models.
2. Fallback paths are explainable in telemetry and activity logs.

### Workstream 7: Config, Telemetry, And Operator Visibility
### Files
1. `<repo-root>/src/loom/config.py`
2. `<repo-root>/docs/CONFIG.md`
3. `<repo-root>/src/loom/engine/runner.py`
4. `<repo-root>/src/loom/engine/orchestrator.py`
5. `<repo-root>/tests/test_config.py`
6. `<repo-root>/tests/test_orchestrator.py`
7. `<repo-root>/CHANGELOG.md`

### Changes
1. Add `budgeted` mode and minimal supporting knobs.
2. Add telemetry fields:
   - `compaction_required_delta_tokens`
   - `compaction_operations_applied`
   - `compaction_schema_tools_before/after`
   - `compaction_prompt_chars_before/after`
   - `compaction_terminal_state` (`fit|degraded_fit|unfit`)
3. Add operator docs for tuning small-model safety ratios and reserves.

### Acceptance Criteria
1. Operators can explain every fit/skip/degraded decision from telemetry.
2. Config surface remains minimal with safe defaults.

## Minimal Iteration Algorithm (Budgeted Mode)
1. Build prompt sections and initial message set.
2. Build provider tool schema set.
3. Compute full request diagnostics and token deficit.
4. If deficit <= 0, execute model call.
5. If deficit > 0, build lane-aware candidate graph.
6. Apply lowest-harm operations incrementally until deficit <= 0 or planner budgets are exhausted.
7. If still over deficit, apply bounded low-value deterministic cap path.
8. Recompute full request diagnostics.
9. If fit, call model; if not fit, emit terminal unfit/degraded state and fail fast with actionable diagnostics.
10. Persist per-label gain outcomes for future feasibility scoring.
11. Emit compaction decision + operation trace telemetry.

## Configuration Additions
1. `limits.runner.runner_compaction_policy_mode = "legacy" | "tiered" | "budgeted" | "off"`
2. `limits.runner.compaction_request_safety_ratio_small`
3. `limits.runner.compaction_request_safety_ratio_default`
4. `limits.runner.compaction_response_reserve_tokens`
5. `limits.runner.compaction_iteration_call_budget`
6. `limits.runner.compaction_label_cooldown_turns`
7. `limits.runner.compaction_feasibility_min_expected_ratio`
8. `limits.runner.prompt_tool_listing_mode = "full" | "compact" | "names_only"`

Defaults:
1. Must be safe for 4k/8k models without manual tuning.
2. Must preserve current behavior for non-`budgeted` modes.

## Test Plan
1. First-turn oversize prompt with many tools is reduced to fit in `budgeted` mode.
2. Full-request accounting test proves schema overhead contributes to deficit.
3. Prompt/schema de-dup test verifies no double full schema embedding.
4. Protected-lane integrity tests verify instructions and active-write context survive pressure.
5. Anti-thrash tests verify call budget, cooldown, and feasibility skip behavior.
6. Emergency cap test verifies deterministic clipping only touches low-value lanes.
7. Regression tests for existing `legacy`, `tiered`, and `off` behavior.

## Rollout Plan
1. Phase 1: ship `budgeted` mode behind explicit opt-in.
2. Phase 2: run replay benchmarks on small-context workloads and document quality/reliability deltas.
3. Phase 3: dogfood with telemetry review and tuning.
4. Phase 4: consider default switch from `tiered` to `budgeted`.

## Exit Criteria
1. Initial invocation overflow rate reduced to near-zero on representative 4k/8k model configs.
2. Fewer provider-level context errors and fewer overflow retry loops.
3. Compactor call churn remains bounded with stable completion latency.
4. No observed truncation of protected instruction lanes.
5. Operator telemetry is sufficient to explain outcomes without code inspection.
