# Context Compaction Policy Plan (2026-02-23)

## Objective
Define and implement deterministic rules for when to compact and when not to compact, with these hard requirements:
1. No hard-cap truncation in runner prompt assembly paths.
2. Critical task prompts/responses stay un-compacted whenever possible.
3. Tool-run payloads are compacted first under pressure.
4. Compaction is demand-driven (triggered by pressure), not eager by default.

## Failure Evidence (Run `20260222-215711-cowork-e773c9b6`)
1. Terminal failure: `subtask_failed` at seq `19411` with `Execution exceeded subtask time budget (1200s) before completion.`
2. Run failure: `task_failed` at seq `19413` (`completed=6`, `total=9`, `failed_subtasks=["draft-twelve-profiles"]`).
3. Near-failure pressure signal: extractor start at seq `19409` with `origin="runner.extract_memory.complete"` and `request_est_tokens=18650` (`messages_chars=74574`).
4. Compactor churn signal: seq `19408` shows `read_file tool output` compacting `21793 -> 6114` chars for `target=3600` with `compactor_retry_count=5`.
5. Repeated overshoot/repair loop: seq `19280+` includes `compactor_invalid_reason="output_exceeds_target"` and retry attempts.

## Scope
Runtime:
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`
2. `/Users/sfw/Development/loom/src/loom/engine/semantic_compactor.py`
3. `/Users/sfw/Development/loom/src/loom/config.py`
4. `/Users/sfw/Development/loom/src/loom/events/types.py` (only if new explicit events are added)

Tests:
1. `/Users/sfw/Development/loom/tests/test_orchestrator.py` (SubtaskRunner compaction behavior tests currently live here)
2. `/Users/sfw/Development/loom/tests/test_semantic_compactor.py`
3. `/Users/sfw/Development/loom/tests/test_verification.py` (for cross-component consistency around oversize outputs)

## Non-Negotiable Policy Rules
1. Do not add deterministic hard truncation for runner model-context compaction.
2. Preserve newest critical task context (active task prompt, latest user steering, latest assistant non-tool response, todo reminder).
3. Compact tool call arguments and tool result payloads before compacting critical narrative text.
4. Trigger compaction only when estimated request crosses pressure thresholds.
5. Memory extraction is best-effort and must not steal budget from critical-path execution.

## Workstream 1: Define Compaction Classes and Policy Matrix
### Design
Introduce explicit classes for compactability:
1. `critical`: active task instructions, recent steering turns, completion-relevant assistant text.
2. `tool_trace`: assistant tool-call args and tool outputs.
3. `historical_context`: older user/assistant/system text outside critical window.
4. `background_extraction`: memory-extractor prompt payloads.

### Implementation
1. Add a policy classifier in `SubtaskRunner` (`_classify_message_for_compaction(...)`).
2. Add a compaction plan builder (`_build_compaction_plan(...)`) that returns ordered candidates by class and age.
3. Add runner config knobs for explicit behavior:
   - `preserve_recent_critical_messages`
   - `compaction_pressure_ratio_soft`
   - `compaction_pressure_ratio_hard`
4. Preserve backward compatibility with defaults matching current practical behavior until flag switch.

### Acceptance Criteria
1. Policy decisions are deterministic for the same message set.
2. Critical turns are not compacted in soft-pressure mode.
3. Tool trace entries are first-class compaction targets.

## Workstream 2: Move to Lazy (Pressure-Triggered) Compaction
### Design
Replace eager per-iteration compaction with pressure-triggered staged compaction:
1. Stage 0: no compaction if request fits budget.
2. Stage 1: compact tool-call arguments.
3. Stage 2: compact tool outputs (older first).
4. Stage 3: compact historical_context text.
5. Stage 4: semantic merge of oldest context only if still above hard threshold.

### Implementation
1. In `runner.py`, gate `_compact_messages_for_model(...)` behind request-size estimation.
2. Re-run token estimate after each stage and stop immediately once under threshold.
3. Add message-level memoization key (content hash + target + label) to avoid repeated compactions in the same loop.
4. Ensure newest critical messages are excluded from stages 1-3 unless hard threshold is exceeded.

### Acceptance Criteria
1. When under context budget, compactor is not invoked.
2. Same iteration no longer re-compacts unchanged messages.
3. Token reduction proceeds in staged order and short-circuits when sufficient.

## Workstream 3: Pressure-Tier Strategy (When to Compact vs Not)
### Design
Define three pressure tiers:
1. `normal` (<= soft threshold): do not compact.
2. `pressure` (soft < usage <= hard): compact only `tool_trace` and old `historical_context`.
3. `critical` (> hard): allow old critical-context merge, still preserve the latest critical turns.

### Implementation
1. Add helper in runner to compute utilization ratio from estimated tokens and configured context budget.
2. Route stage limits by tier (e.g., preserve window shrinks only in `critical` tier).
3. Keep `tool_call` placeholders only for older turns; never replace the latest assistant/tool exchange context blindly.

### Acceptance Criteria
1. Tier transitions are visible in model invocation metadata.
2. `normal` tier performs no semantic compaction.
3. `critical` tier can recover from over-budget conversations without dropping latest instructions.

## Workstream 4: Eliminate Compactor Thrash (Retry/Churn Controls)
### Design
Prevent repetitive expensive compactions that do not improve fit.

### Implementation
1. Add per-message compaction result caching in runner loop (`compacted_once` marker with source hash + target).
2. If prior output exceeded target and was kept with warning, avoid immediate re-compaction of near-identical input in same loop.
3. Add retry backoff signal for semantic compactor usage from runner side (skip duplicate compactions when delta is negligible).
4. Emit counters in events/diagnostics:
   - `compaction_stage`
   - `compaction_candidate_count`
   - `compaction_skipped_reason` (`cache_hit`, `policy_preserve`, `no_pressure`, `no_gain`)

### Acceptance Criteria
1. Compactor invocation count per loop decreases for stable prompts.
2. Overshoot-warning loops do not repeatedly hit the same payload in adjacent iterations.
3. Observability shows why compaction was skipped/applied.

## Workstream 5: Timeout-Aware Compaction and Extraction
### Design
Use remaining subtask wall-clock budget to protect critical-path completion.

### Implementation
1. Add `remaining_seconds` checks before expensive compaction stages.
2. If remaining budget is low, skip non-essential compaction and attempt model call with minimal transformations.
3. Refactor `_extract_memory(...)` input formatting:
   - summarize tool arguments using existing compaction helpers (especially `document_write`-sized payloads),
   - bound extractor prompt via semantic summarization policy (not hard truncation),
   - optionally skip/defer extraction when subtask timeout is imminent.
4. Add explicit extractor diagnostics (`extractor_prompt_chars`, `extractor_prompt_est_tokens`, `extractor_compacted_fields`).

### Acceptance Criteria
1. Extractor prompts no longer include raw document-scale tool args.
2. Timeout-near runs do not spend final budget on non-critical extraction compaction.
3. Subtask critical path is not blocked by background memory extraction behavior.

## Workstream 6: Tests, Telemetry, Rollout, and Guardrails
### Tests
1. `test_no_compaction_when_under_budget`.
2. `test_compaction_order_tool_trace_before_critical_context`.
3. `test_preserve_latest_critical_turns_under_pressure`.
4. `test_critical_tier_old_context_merge_without_latest_instruction_loss`.
5. `test_memory_extractor_compacts_large_tool_args`.
6. `test_timeout_near_skips_nonessential_compaction`.
7. `test_no_hard_truncation_marker_inserted_by_runner_compaction_path`.

### Telemetry
1. Add run-level aggregates:
   - compactor calls per subtask iteration,
   - compaction skipped/applied by stage and reason,
   - extractor prompt token distribution.
2. Add warning thresholds for pathological churn (e.g., >N compactor invocations within one iteration).

### Rollout
1. Add feature flag in config: `runner_compaction_policy_mode = "legacy" | "tiered"` (default `legacy` for first merge).
2. Land with dual-path tests.
3. Enable `tiered` in dogfood config.
4. Remove legacy mode only after stability period and log review.

### Acceptance Criteria
1. Existing tests pass plus new compaction-policy tests.
2. No regressions in verification prompt assembly behavior.
3. Event logs show reduced compactor churn and lower extractor prompt size for large tool outputs.

## Execution Order
1. Implement Workstream 1 policy classification and config scaffolding.
2. Implement Workstream 2 staged lazy compaction flow.
3. Implement Workstream 3 tier thresholds and preserve windows.
4. Implement Workstream 4 anti-thrash caching/skip logic.
5. Implement Workstream 5 timeout-aware extraction + compaction gating.
6. Implement Workstream 6 tests, telemetry fields, and rollout flag.

## Exit Criteria
1. The six workstreams are merged behind tests and a rollout flag.
2. A replay run of document-heavy subtasks completes without compaction thrash dominating execution time.
3. Critical prompt/response context remains intact while tool traces are compacted first under pressure.
