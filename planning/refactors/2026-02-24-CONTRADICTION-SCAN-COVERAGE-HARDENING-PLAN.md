# Contradiction Scan Coverage Hardening Plan (2026-02-24)

## Objective
Reduce false-negative contradiction downgrades by making placeholder-claim contradiction checks coverage-aware, bounded, and deterministic.

The goal is to keep the recent agility win (avoid false hard-fails from bad verifier placeholder claims) while closing the medium-risk gap where the scan may miss a real placeholder outside the initial candidate set.

## Trigger
Current contradiction guard behavior can downgrade some placeholder failures to inconclusive based on a deterministic scan over a narrow candidate set. If the real placeholder exists outside that set, the downgrade can be too permissive.

## Hard Invariants
1. Hard invariant failures (`hard_invariant_failed`) always block.
2. Contradiction downgrade is allowed only when deterministic evidence is sufficiently broad and finds no placeholder markers.
3. Scan remains bounded (no unbounded workspace traversal).
4. Core logic remains domain-agnostic.

## Non-Goals
1. Do not convert contradiction guard into a full static-analysis system.
2. Do not add process/domain-specific placeholder semantics in core runtime.
3. Do not relax existing hard invariant or strict remediation terminal behavior.

## Scope
Primary:
1. `/Users/sfw/Development/loom/src/loom/engine/verification.py`
2. `/Users/sfw/Development/loom/src/loom/config.py`
3. `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py` (only if additional verifier context plumbing is needed)
4. `/Users/sfw/Development/loom/src/loom/events/types.py`
5. `/Users/sfw/Development/loom/tests/test_verification.py`
6. `/Users/sfw/Development/loom/tests/test_orchestrator.py` (if context propagation changes)
7. `/Users/sfw/Development/loom/tests/test_config.py`

Potential secondary touchpoints (only if needed):
1. `/Users/sfw/Development/loom/src/loom/engine/runner.py`
2. `/Users/sfw/Development/loom/src/loom/state/task_state.py` (avoid unless metadata persistence contract needs extension)

## Current Gap
The contradiction scan currently emphasizes canonical/current-phase artifacts and immediate changed files. That is directionally correct, but coverage can be insufficient when:
1. Placeholder text is in a nearby changed artifact not included in candidate list.
2. Placeholder remains in prior-attempt files while the current attempt touched a subset.
3. Relevant text artifacts exist in deliverable-adjacent files and are skipped.

## Target End State
Contradiction downgrade should only happen when all three conditions hold:
1. Placeholder/TODO-style verifier claim is present.
2. Deterministic scan coverage is marked sufficient.
3. No placeholder marker match is found across the bounded scan set.

If coverage is insufficient, do not downgrade on contradiction; keep the original semantic failure path.

## Proposed Design

### 1. Candidate Expansion (Deterministic, Prioritized)
Build scan candidates in ordered tiers:
1. Canonical deliverables for the active subtask (existing behavior).
2. Current attempt changed files from tool calls.
3. Prior successful attempt changed files for same subtask (available through existing retry evidence context; if not directly available, add minimal plumbing).
4. Evidence-artifact file paths already available to verifier context (text-like only).

Deduplicate and preserve first-seen order.

### 2. Bounded Fallback Workspace Scan
If primary candidate scan finds no placeholder markers, run a bounded fallback scan over text-like files in workspace with strict caps:
1. `contradiction_scan_max_files` (recommended default: `80`)
2. `contradiction_scan_max_total_bytes` (recommended default: `2_500_000`)
3. `contradiction_scan_max_file_bytes` (recommended default: `300_000`)
4. `contradiction_scan_allowed_suffixes` (text-like list only; no binaries)

Traversal rules:
1. Skip hidden/system dirs and known heavy dirs (`.git`, `.venv`, `node_modules`, `.mypy_cache`, etc.).
2. No symlink following.
3. Stop early when caps are reached.

### 3. Coverage Sufficiency Gate
Add a deterministic gate (`coverage_sufficient`) before downgrade:
1. Must scan at least one canonical deliverable or one changed-file candidate.
2. Must scan at least `min_files_for_contradiction` (recommended `>= 2`) unless exactly one canonical deliverable exists.
3. Must not terminate solely due to cap exhaustion before scanning any high-priority candidate.

If not sufficient:
1. Preserve original verifier fail (`reason_code` unchanged).
2. Annotate metadata: `contradiction_detected=false`, `coverage_sufficient=false`, `coverage_insufficient_reason=...`.

### 4. Decision Matrix
1. Placeholder claim + match found -> keep fail (no contradiction).
2. Placeholder claim + no match + sufficient coverage -> downgrade to inconclusive (`parse_inconclusive`) and mark `contradiction_detected=true`.
3. Placeholder claim + no match + insufficient coverage -> keep fail (no downgrade), include diagnostic metadata.
4. Non-placeholder claim -> unchanged behavior.

### 5. Observability Expansion
Emit detailed contradiction telemetry:
1. `scan_mode`: `targeted_only|targeted_plus_fallback`
2. `coverage_sufficient`: bool
3. `scanned_file_count`
4. `scanned_total_bytes`
5. `matched_file_count`
6. `candidate_source_counts` (canonical/current/prior/fallback)
7. `coverage_insufficient_reason` (when applicable)

Add counters:
1. `contradiction_downgrade_count`
2. `contradiction_detected_no_downgrade_count`
3. `cap_exhaustion_count`

### 6. Config Additions
In `VerificationConfig` and parser:
1. `contradiction_scan_max_files`
2. `contradiction_scan_max_total_bytes`
3. `contradiction_scan_max_file_bytes`
4. `contradiction_scan_allowed_suffixes`
5. `contradiction_scan_min_files_for_sufficiency`

All with safe defaults and numeric bounds.

### 7. Compatibility Strategy
1. Keep `contradiction_guard_enabled=true` default.
2. Add optional `contradiction_guard_strict_coverage=true` toggle:
   - `true`: require sufficiency for downgrade (recommended default).
   - `false`: preserve today’s permissive downgrade semantics for emergency rollback.

## Workstreams

### W1: Coverage Model and Candidate Plumbing
1. Refactor scan candidate builder with explicit source buckets.
2. Include prior-attempt candidate files if available from existing call context.
3. Normalize/resolve paths safely and cross-platform.

Exit criteria:
1. Candidate set includes canonical + current + prior paths.
2. No duplicate scan work.

### W2: Bounded Fallback Scanner
1. Add bounded workspace scanner with size/file/suffix limits.
2. Ensure deterministic ordering and early termination.

Exit criteria:
1. No unbounded traversal.
2. Scanner obeys all caps in tests.

### W3: Coverage Sufficiency Gate + Decision Logic
1. Implement sufficiency scoring.
2. Integrate gate into contradiction downgrade path.
3. Keep original fail when insufficient coverage.

Exit criteria:
1. Downgrade only with sufficient coverage.
2. Insufficient coverage never silently downgrades.

### W4: Telemetry and Event Schema
1. Extend contradiction events with coverage fields.
2. Add counters for downgrade/no-downgrade/cap-exhaustion.

Exit criteria:
1. Operators can distinguish safe contradiction downgrades from low-coverage cases.

### W5: Config and Validation
1. Add config fields + parser bounds.
2. Add tests for malformed values and clamping behavior.

Exit criteria:
1. Invalid config cannot crash verification.
2. Defaults remain safe and bounded.

### W6: Regression and Edge Tests
Add tests for:
1. Placeholder present in non-deliverable changed file -> no downgrade.
2. Placeholder present in prior-attempt artifact candidate -> no downgrade.
3. No placeholders + sufficient coverage -> downgrade to inconclusive.
4. No placeholders + insufficient coverage -> keep fail.
5. Cap exhaustion path marks insufficient coverage.
6. Binary/large files skipped safely.
7. Symlink and path traversal edge cases ignored.

Exit criteria:
1. Medium concern class is covered by deterministic tests.
2. Existing contradiction and verification-only retry tests still pass.

### W7: Rollout and Safety Controls
1. Ship behind strict-coverage default with metrics.
2. Run canary on processes known for large artifact sets.
3. Compare contradiction downgrade quality before/after.

Exit criteria:
1. No increase in bad-pass incidents from contradiction path.
2. False hard-fail reduction retained.

## Knock-On Problems and Mitigations
1. Performance regression in large workspaces:
   - Mitigation: strict byte/file caps, suffix whitelist, directory exclusions, early stop.
2. IO contention on networked filesystems:
   - Mitigation: short-circuit scanning after sufficient evidence; avoid scanning unchanged binaries.
3. False confidence from narrow fallback set:
   - Mitigation: explicit `coverage_sufficient` gate and no downgrade when insufficient.
4. Telemetry payload bloat:
   - Mitigation: only counts + small samples; avoid full path dumps in high-volume events.
5. Privacy/sensitivity in event data:
   - Mitigation: emit counts and relative paths only; avoid file content in contradiction events.
6. Path normalization drift (macOS/Linux/Windows):
   - Mitigation: centralize path normalization and test with mixed path styles.
7. Symlink traversal and security concerns:
   - Mitigation: `is_symlink` skip policy and no out-of-workspace traversal.
8. Interaction with remediation queue:
   - Mitigation: unchanged queue semantics; contradiction path remains verifier-only retry first.
9. Flaky tests due to filesystem timing:
   - Mitigation: deterministic temporary fixtures and sorted traversal order.
10. Backward compatibility of metadata consumers:
    - Mitigation: additive metadata only; no field removals/renames.

## Residual Risks After This Plan
1. Very deeply nested or excluded-path placeholders may still evade detection within caps.
2. Placeholder regex itself can still miss exotic placeholder formats not covered by marker set.
3. Workspace-wide fallback may still be expensive in pathological repos even when bounded.
4. Contradiction telemetry quality depends on consistent task/workspace path hygiene.

## Residual Risk Management
1. Monitor:
   - contradiction downgrade rate
   - downgrade reversal rate (later remediation finds actual placeholder)
   - cap-exhaustion frequency
2. Adjust caps/suffixes based on telemetry by process type.
3. If bad-pass risk rises, flip `contradiction_guard_strict_coverage` + tighten thresholds.

## Validation Plan
Targeted:
1. `uv run pytest tests/test_verification.py`
2. `uv run pytest tests/test_orchestrator.py` (if verifier context/plumbing touched)
3. `uv run pytest tests/test_config.py`

Broader:
1. `uv run pytest tests/test_retry.py`
2. `uv run pytest tests/test_processes.py`

## Acceptance Criteria
1. Contradiction downgrade only occurs with `coverage_sufficient=true`.
2. Known medium-risk reproduction (placeholder outside original candidate set) no longer downgrades incorrectly.
3. No regression in hard invariant handling.
4. Existing targeted suites pass.
5. Contradiction telemetry includes enough fields to audit downgrade quality.

## Rollout
Phase 1: Implement + tests + local validation.
Phase 2: Canary with telemetry inspection.
Phase 3: Tune caps/thresholds if needed.
Phase 4: Document defaults and operational playbook for contradiction events.
