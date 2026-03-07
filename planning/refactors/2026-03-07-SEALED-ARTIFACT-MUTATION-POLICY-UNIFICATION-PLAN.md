# Sealed Artifact Mutation Policy Unification Plan (2026-03-07)

## Executive Summary
The current sealed artifact system is correct for a narrow set of tools (`edit_file`, `write_file`, `document_write`) and inconsistent for the broader workspace-writing suite (for example `spreadsheet create` and many `output_path` tools). This creates stale seals, synthesis-gate false failures, and unverifiable mutation lineage.

This plan unifies mutation policy, resealing, and provenance under one production-grade contract:
1. Any workspace mutation must pass the same preflight sealed-artifact evidence gate.
2. Any successful workspace mutation must reseal touched tracked artifacts.
3. Any artifact provenance/seal-backfill path must be tool-agnostic, not hardcoded to specific tool names.
4. `edit_file` behavior becomes the default behavior for the full tool suite.

## Incident Anchor
Primary anchor:
1. Run: `cowork-2cd782fe`
2. Logs:
   - `/Users/sfw/.loom/logs/20260306-201007-cowork-2cd782fe.events.jsonl`
   - `/Users/sfw/.loom/logs/20260307-004143-cowork-2cd782fe.events.jsonl`
3. Sequence:
   - `332`: `write_file` wrote `competitor-pricing.csv` (sealed baseline).
   - `1377`: `spreadsheet create` overwrote `competitor-pricing.csv`.
   - `3919`: synthesis failed: `artifact_seal_mismatch`.
4. Root cause:
   - `spreadsheet` mutates files but is not treated as mutating by execution reseal flow.
   - Seal recording/backfill/provenance is hardcoded to `write_file`/`document_write`.

Related recurring pattern:
1. Other runs show the same final-stage `artifact_seal_mismatch` family.
2. Risk is systemic across tools writing through `output_path`.

## Goals
1. Enforce sealed-artifact policy uniformly for all workspace-mutating tools.
2. Make resealing deterministic and comprehensive for all successful mutations.
3. Remove tool-name special casing from core seal/provenance logic.
4. Preserve strict preflight evidence gating semantics currently seen with `edit_file`.
5. Provide high-confidence migration with canary controls and strong test gates.

## Non-Goals
1. Redesigning all tool UX or argument schemas.
2. Replacing current evidence model with a new storage backend.
3. Expanding seal policy to non-workspace external side effects in this phase.

## Required Invariants (Hard)
1. No tool may mutate a sealed, verified artifact without post-seal confirmation evidence.
2. Any successful mutation of a tracked sealed artifact updates its seal hash and metadata.
3. Synthesis gate must only fail on true mismatches, not stale internal metadata.
4. Seal/provenance behavior must not depend on hardcoded tool name allowlists.
5. All gated mutation failures and reseals must be observable via telemetry events.

## Current Gaps (Code-Level)
1. Reseal trigger depends on `tool.is_mutating`:
   - `src/loom/engine/runner/execution.py`
2. Many workspace-writing tools do not set `is_mutating=True`:
   - `src/loom/tools/spreadsheet.py`
   - multiple `output_path` tools
3. Seal recording/backfill only accepts `write_file`/`document_write`:
   - `src/loom/engine/orchestrator/evidence.py`
   - `src/loom/state/evidence.py`
   - `src/loom/engine/orchestrator/validity.py`
4. Path targeting for policy is partly allowlist-driven:
   - `src/loom/engine/runner/core.py`
   - `src/loom/engine/runner/policy.py`

## Target Architecture

### 1) Unified Mutation Contract
Use a single contract for all tools:
1. `is_mutating` truthfully indicates workspace mutation capability.
2. Mutating tools must emit `ToolResult.files_changed` for all touched workspace paths.
3. Policy targeting uses normalized paths from:
   - deterministic argument-derived paths, plus
   - `files_changed` for post-call reseal finalization.

No tool-name special cases in runner policy core.

### 2) Preflight Evidence Gate As Default Control
Apply current sealed edit gate to all mutating tools:
1. Resolve target paths.
2. For tracked verified seals, require post-seal confirmation evidence before execution.
3. Block execution with actionable reason if missing evidence.

This is the primary safety control (same behavior users already trust with `edit_file`).

### 3) Generic Reseal Pipeline
After successful mutating calls:
1. Compute affected paths from normalized target paths + `files_changed`.
2. For tracked paths, hash current on-disk bytes and update seals.
3. Persist `previous_sha256`, `sealed_at`, `subtask_id`, `tool`, `tool_call_id`, `run_id`.

### 4) Generic Artifact Provenance Extraction
Unify artifact provenance capture for any mutating tool that changes files:
1. Use file bytes from disk when available.
2. Fallback to structured content fields only when file bytes are unavailable.
3. Stop hardcoding provenance extraction to two tool names.

### 5) Defense-In-Depth Guard
Keep a post-call guard for non-deterministic tool behaviors:
1. Detect unexpected sealed path deltas not covered by policy target computation.
2. Emit high-severity telemetry.
3. Optional rollback via changelog in strict mode (feature-flagged).

Primary enforcement remains preflight gating, not rollback.

## Tool Suite Scope and Migration Strategy

### Tier P0 (Immediate Production Risk)
1. `spreadsheet` write operations (`create`, `add_rows`, `add_column`, `update_cell`)
2. Any currently enabled tool that writes via `output_path` and returns `files_changed`
3. `run_tool` passthrough path validation to ensure downstream mutating behavior is preserved

### Tier P1 (Full Workspace-Writing Suite)
Promote all workspace-writing tools to explicit mutating semantics where applicable, including:
1. report-generating analyzers with optional `output_path`
2. tools writing markdown/json/csv artifacts
3. OCR/export tools writing artifacts

### Tier P2 (External Side-Effect Tools)
Tools mutating external systems (`wp_cli`, provider agent tools, etc.):
1. Keep `is_mutating=True`
2. Out of scope for artifact reseal unless they touch workspace files
3. Ensure no regression in current policy behavior

## Implementation Plan

### Phase 0: Safety Rail and Inventory Freeze
1. Add temporary audit telemetry listing all successful tool calls with `files_changed` where tool `is_mutating=False`.
2. Capture 7-day baseline in local/dev runs.
3. Freeze expansion of new writing tools until Phase 2 lands.

Deliverables:
1. audit report artifact (tool name, paths, call counts)
2. risk-ranked migration list

### Phase 1: Mutating Semantics Normalization
1. Set `is_mutating=True` on all workspace-writing tools.
2. Ensure write paths are always workspace-relative and included in `files_changed`.
3. Add static lint/test preventing future workspace-writing tools from omitting mutating flag.

Files:
1. `src/loom/tools/*.py` (targeted tool classes)
2. new test/lint helper in `tests/` and optional script in `scripts/`

### Phase 2: Policy Target Path Unification
1. Replace hardcoded mutation-name branches with contract-driven logic.
2. Keep spreadsheet operation filtering but move it into tool-level behavior if needed.
3. Ensure policy target derivation supports all path-bearing argument shapes (`path`, `output_path`, etc.).

Files:
1. `src/loom/engine/runner/core.py`
2. `src/loom/engine/runner/policy.py`

### Phase 3: Provenance and Seal Backfill Generalization
1. Refactor artifact provenance extraction to use generic changed-file bytes.
2. Refactor `_record_artifact_seals` to include all successful mutating tool calls with changed files.
3. Refactor backfill to include all artifact records with path + hash, not just two tool names.

Files:
1. `src/loom/state/evidence.py`
2. `src/loom/engine/orchestrator/evidence.py`
3. `src/loom/engine/orchestrator/validity.py`

### Phase 4: Post-Call Guard (Strict Mode)
1. Compare pre/post seal hashes for tracked paths touched during call execution.
2. If mismatch is outside expected affected set, fail call and emit dedicated reason code.
3. Add feature flag:
   - `execution.sealed_artifact_post_call_guard = off|warn|enforce`

Files:
1. `src/loom/engine/runner/execution.py`
2. config plumbing in `src/loom/config.py` and setup docs

### Phase 5: Telemetry and Operator UX
1. Add events:
   - `sealed_policy_preflight_blocked`
   - `sealed_reseal_applied`
   - `sealed_unexpected_mutation_detected`
2. Update run scorecard to summarize:
   - gated mutation attempts
   - resealed path counts
   - stale-seal mismatch prevented vs observed

Files:
1. `src/loom/events/types.py`
2. orchestrator/runner telemetry emission paths
3. TUI messaging surfaces for actionable error text

## Test Strategy (Production Gate)

### Unit Tests
1. Parameterized test over all registered tools:
   - if tool can write workspace files, `is_mutating=True` is required.
2. Policy tests:
   - sealed verified path + no evidence => blocked before execution.
   - sealed verified path + valid post-seal evidence => allowed.
3. Reseal tests:
   - any successful mutating tool updates tracked seal hash when path changed.
4. Provenance tests:
   - changed-file artifacts produce hash/path records regardless of tool name.

### Integration Tests
1. Reproduce anchor sequence with `spreadsheet create`:
   - write -> seal -> spreadsheet mutate -> reseal -> synthesis gate passes.
2. `output_path` tools:
   - generate report over existing sealed file path with and without evidence.
3. mixed-tool sequence:
   - `document_write` then analyzer overwrite then synthesis.

### Regression Tests
1. Existing `edit_file` evidence gating behavior must remain unchanged.
2. No new false-positive blocks for read-only tool calls.

### Performance Tests
1. Measure added overhead for hash/reseal operations on typical artifact sizes.
2. Ensure no material latency regression in normal runs.

## Rollout and Risk Control

### Flags
1. `sealed_mutation_unification_enabled` (default off in first deploy)
2. `sealed_artifact_post_call_guard` (`warn` then `enforce`)

### Rollout Steps
1. Dev + CI with flags on.
2. Internal dogfood: enable unification, guard in `warn`.
3. Burn-in for mismatch telemetry.
4. Flip guard to `enforce` after zero unresolved alerts window.

### Backout
1. Disable unification flag to restore previous behavior quickly.
2. Keep telemetry active to preserve forensic visibility.

## Acceptance Criteria
1. `spreadsheet` and all workspace-writing tools follow preflight sealed evidence policy.
2. No synthesis failures caused by stale seals after internal tool mutations in canary runs.
3. Full test suite passes with new mutation contract checks.
4. Telemetry shows reseal events for non-`write_file`/`document_write` tools.
5. Operator-facing errors remain clear and actionable.

## Open Decisions
1. Should strict post-call rollback be enabled by default or only for high-rigor profiles?
2. Should we require explicit per-tool declaration of writable argument keys to avoid heuristics?
3. Should external-side-effect tools be split from workspace mutation semantics in the base `Tool` contract?

## Proposed Timeline
1. Week 1:
   - Phase 0, Phase 1
2. Week 2:
   - Phase 2, Phase 3
3. Week 3:
   - Phase 4, Phase 5, canary rollout

## Summary
Yes, sealed/reseal flow and policy should be brought to the full workspace-mutating tool suite. The hardened production path is to make preflight evidence gating (the current `edit_file` model) universal, make resealing/provenance tool-agnostic, and retain a post-call guard as defense-in-depth.
