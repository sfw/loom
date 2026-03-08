# Phase Fan-In and Deliverable Ownership Plan (2026-03-06)

## Executive Summary
Current orchestration allows parallel sibling subtasks, and file writes are last-writer-wins. If two siblings target the same canonical deliverable path, one can overwrite the other.

Proposed direction:
1. Add a global ownership invariant now: only one subtask can own/write a canonical deliverable at a time.
2. Add `fan_in` phase output strategy as opt-in now: worker subtasks write intermediate artifacts, one phase finalizer writes canonical deliverables.
3. Keep `is_synthesis` reserved for true cross-phase/final integration; do not reuse it for every phase finalizer.
4. Publish final canonical deliverables transactionally (all-or-none) to prevent partial final state.
5. After telemetry and process-test burn-in, consider flipping default strategy from `direct` to `fan_in`.

## Problem Statement
The current model mixes two concerns:
1. parallel research/build work
2. final canonical deliverable assembly

When multiple subtasks do both at once and target one filename, concurrency causes non-deterministic output replacement. Existing verification catches missing/invalid outputs, but it does not prevent concurrent same-file writes during execution.

## Goals
1. Eliminate concurrent multi-writer races on canonical deliverables.
2. Preserve parallelism for independent work.
3. Support all process types (research, coding, strategy, operations), not just one domain.
4. Keep backward compatibility for existing processes during rollout.
5. Provide deterministic and observable ownership/assembly behavior.

## Non-Goals
1. Redesign all verification policies.
2. Remove `direct` writing strategy immediately.
3. Force every phase finalizer to use synthesis-only gates.

## Key Design Decisions
1. Introduce phase output strategy:
   - `direct` (default initially): subtask writes canonical deliverables directly.
   - `fan_in` (opt-in): workers write intermediate artifacts, phase finalizer writes canonical deliverables.
2. Introduce separate role for phase finalizer (not `is_synthesis`):
   - `is_phase_finalizer=true` (or equivalent runtime role metadata).
3. Add global canonical deliverable ownership checks regardless of strategy.
4. Scheduler must avoid dispatching conflicting owners in the same batch.
5. Add deterministic fairness for deferred conflicting writers so they cannot starve.
6. Treat non-owner canonical writes as a first-class reason code (not generic failure) to avoid retry loops.

## Proposed Runtime Model

## 1) Ownership Map
At dispatch time, build a per-phase ownership map:
1. Canonical deliverables = `process.get_deliverables()[phase_id]`.
2. For `direct`:
   - exactly one owner subtask per canonical path in a runnable set.
   - if multiple candidate owners exist, serialize them and emit conflict telemetry.
3. For `fan_in`:
   - owner is the phase finalizer only.
   - non-finalizer workers are forbidden from mutating canonical deliverable paths.

## 2) Scheduler Conflict Guard (Global Invariant)
Before `batch = runnable[:max_parallel]`, partition runnable subtasks into:
1. non-conflicting writers
2. deferred writers that overlap canonical paths in current batch

Only non-conflicting group enters `asyncio.gather`. Deferred subtasks remain pending for later iterations.

Fairness requirement:
1. Deferred writer queue is FIFO by first-deferral timestamp.
2. Any subtask deferred for conflict N consecutive iterations gets priority bump.
3. Emit starvation telemetry when deferral count crosses threshold.

## 3) Phase Finalizer (Fan-In)
For each phase in `fan_in` mode:
1. Workers:
   - produce phase-scoped intermediate artifacts only (for example `.loom/phase-artifacts/<phase_id>/<subtask_id>.*`).
2. Finalizer:
   - depends on all worker subtasks in phase.
   - reads worker artifacts.
   - writes canonical deliverable filenames declared by phase.
   - publishes canonical outputs transactionally (stage + commit/rename) so partial writes do not leak.
3. Verification:
   - worker verification checks intermediate artifact quality/existence.
   - phase deliverable gates apply at finalizer completion.

## 4) Write Enforcement
Extend runner/tool-write guard input:
1. `expected_deliverables` (allowed set)
2. `forbidden_deliverables` (blocked set)

Behavior:
1. Any attempt to mutate a forbidden canonical path returns tool failure with explicit reason.
2. In `fan_in`, workers get canonical paths in `forbidden_deliverables`.
3. Finalizer gets canonical paths in `expected_deliverables`.

Failure classification requirement:
1. Forbidden write attempts produce `reason_code=forbidden_output_path` (or equivalent dedicated code).
2. Retry manager must route this class to targeted guidance, not blind generic retries.

## 5) Intermediate Artifact Contract
Fan-in requires explicit artifact contract to avoid ambiguous merges:
1. Each worker writes:
   - artifact payload file
   - artifact manifest entry (`phase_id`, `subtask_id`, `attempt`, `schema_version`, `generated_at`, `content_hash`)
2. Finalizer consumes only latest successful artifact per worker for current task/run.
3. Replan/retry cleanup:
   - stale artifact attempts excluded by manifest filtering
   - optional GC after phase finalizer success/failure

## 6) Canonical Path Normalization
Conflict and write guards must compare canonicalized paths:
1. normalize separators, dot segments, and case behavior by platform
2. resolve symlink traversal and enforce workspace-relative normalization
3. compare against normalized deliverable map to prevent bypass via path tricks

## Process Schema Changes

## 1) New Output Coordination Contract
Add process-level block:

```yaml
output_coordination:
  strategy: direct # direct | fan_in
  intermediate_root: .loom/phase-artifacts
  enforce_single_writer: true
  publish_mode: transactional # transactional | best_effort
  conflict_policy: defer_fifo # defer_fifo | fail_fast
```

Optional per-phase override:

```yaml
phases:
  - id: phase-a
    output_strategy: fan_in
```

Validation rules:
1. `strategy` must be `direct` or `fan_in`.
2. `intermediate_root` must be workspace-relative, normalized.
3. `publish_mode` must be `transactional` or `best_effort` (`transactional` default).
4. If `fan_in` enabled and phase has deliverables, phase must resolve to exactly one finalizer owner.
5. In `fan_in`, worker subtasks for the phase may not declare canonical deliverable ownership.

## 2) Plan/State Representation
Prefer adding explicit role metadata to subtasks:
1. `output_role`: `worker | phase_finalizer | synthesis`

Fallback if schema churn is undesirable in first pass:
1. compute role dynamically from phase strategy + dependency topology.

Finalizer injection rule:
1. Only inject a phase finalizer when a phase has more than one worker output producer.
2. Single-subtask phases may stay direct-owner to avoid unnecessary orchestration overhead.

## Design Review Findings and Knock-On Mitigations
1. Starvation risk from repeated deferrals.
   - Mitigation: FIFO deferral queue, priority bump after N deferrals, starvation counter telemetry.
2. Retry loop risk when worker keeps attempting forbidden canonical writes.
   - Mitigation: dedicated reason code + remediation prompt block with explicit intermediate artifact target.
3. Partial final publish risk when finalizer updates multiple deliverables and fails mid-way.
   - Mitigation: transactional publish using staged temp files + commit step.
4. Artifact contamination risk from prior attempts/replans.
   - Mitigation: manifest-scoped artifact selection keyed by task/run/subtask attempt.
5. Provenance loss risk (final output no longer obviously linked to worker evidence).
   - Mitigation: finalizer must carry worker artifact provenance map into evidence ledger/summary.
6. Verification false negatives during migration (worker subtasks expected to create canonical outputs).
   - Mitigation: role-aware expected deliverables; worker checks target artifacts, finalizer checks canonical outputs.
7. Plan churn risk during replanning (IDs dropped/rewired).
   - Mitigation: deterministic finalizer ID convention (`<phase_id>__finalize_output`) and replan retention checks.
8. Throughput regression risk.
   - Mitigation: serialize only conflicting canonical writers and keep full parallelism for disjoint paths.
9. UX regression risk (users think workers failed because canonical outputs absent until finalizer).
   - Mitigation: explicit phase progress events: `worker_artifacts_ready`, `phase_finalizer_pending`.
10. Security/path bypass risk.
   - Mitigation: centralized canonical path normalizer used by scheduler conflict detection and runner write guards.
11. Contract drift risk (`deliverables_touched` claims not matching actual writes).
   - Mitigation: compare completion contract payload to normalized `files_changed` and emit mismatch diagnostics.
12. Output noise risk (intermediate artifacts clutter user-facing output lists).
   - Mitigation: classify artifacts as `intermediate` vs `canonical` and hide intermediate by default in run summary views.
13. Finalizer bottleneck risk when one worker fails.
   - Mitigation: explicit phase policy for finalizer input strictness (`require_all_workers` default) with optional controlled partial mode for non-critical phases.

## Orchestrator Integration Plan

## W0: Guardrail (No Process Format Change Required)
Files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. `<repo-root>/tests/test_orchestrator.py`

Tasks:
1. Add batch-level canonical deliverable conflict detection.
2. Serialize conflicting writers even in `direct` mode.
3. Add FIFO deferral queue + starvation-prevention priority bump.
4. Emit telemetry events for deferred conflict subtasks and starvation warnings.

## W1: Schema + Loader + Validation
Files:
1. `<repo-root>/src/loom/processes/schema.py`
2. `<repo-root>/tests/` (schema loader validation tests)
3. docs/changelog updates

Tasks:
1. Add `output_coordination` contract parsing.
2. Add validation invariants.
3. Add deterministic finalizer ID convention and validation.
4. Document defaults and migration guidance.

## W2: Fan-In Runtime
Files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. `<repo-root>/src/loom/prompts/assembler.py`
3. `<repo-root>/src/loom/engine/runner.py`
4. tests for runner/path guards and fan-in lifecycle

Tasks:
1. Materialize phase finalizer subtasks when needed.
2. Route worker output expectations to intermediate paths.
3. Restrict worker writes to non-canonical paths.
4. Restrict canonical writes to finalizer owner.
5. Add transactional finalizer publish path (stage + commit).
6. Add forbidden-write reason code plumbing and targeted remediation guidance.
7. Add artifact manifest read/write and latest-successful selection.
8. Cross-check completion contract `deliverables_touched` against actual normalized file mutations.

## W3: Verification/Gates Alignment
Files:
1. `<repo-root>/src/loom/engine/verification.py`
2. `<repo-root>/src/loom/engine/iteration_gates.py`
3. tests for per-role verification behavior

Tasks:
1. Ensure deliverable existence/placeholder gates evaluate canonical outputs at phase finalizer boundary.
2. Keep worker gates focused on intermediate artifacts and quality signals.
3. Propagate worker artifact provenance into finalizer verification/evidence summaries.
4. Ensure iteration-loop gates do not require canonical outputs before finalizer execution.
5. Add phase-level `finalizer_input_policy` handling (`require_all_workers` vs controlled partial mode).

## W4: Process Rollout
Files:
1. Builtin process YAMLs under `<repo-root>/src/loom/processes/builtin/`
2. docs/changelog

Tasks:
1. Keep default `direct`.
2. Opt-in selected builtins to `fan_in`.
3. Collect telemetry and process-test outcomes.
4. Add UI/event updates for worker-artifact vs finalizer states.
5. Define objective criteria for default flip to `fan_in`.
6. Update output panels/log summaries to separate canonical deliverables from intermediate artifacts.

## W5: Operational Hardening
Files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. `<repo-root>/src/loom/tools/workspace.py`
3. `<repo-root>/tests/` (fault-injection and recovery suites)

Tasks:
1. Add restart/recovery handling so interrupted runs do not publish partial finalizer outputs.
2. Add stale artifact GC policy for `.loom/phase-artifacts`.
3. Add fault-injection tests for crash during finalizer publish and re-entry behavior.

## Test Strategy
1. Unit: conflict detector identifies overlapping canonical path owners.
2. Unit: scheduler batch excludes conflicting writers.
3. Unit: scheduler fairness prevents indefinite conflict deferral starvation.
4. Unit: runner rejects non-owner canonical writes with clear dedicated reason code.
5. Unit: canonical path normalizer blocks equivalent-path bypasses.
6. Unit: finalizer ID stability across replans.
7. Unit: artifact manifest selection excludes stale attempts.
8. Integration: transactional finalizer publish is all-or-none on injected failures.
9. Integration: `fan_in` phase with N workers + 1 finalizer produces canonical outputs.
10. Integration: iteration loops with worker-only artifacts do not fail deliverable gates early.
11. Regression: existing `direct` processes still pass.
12. Concurrency: `max_parallel_subtasks > 1` with overlapping paths does not race.

## Telemetry and Observability
Add events/counters:
1. `subtask_output_conflict_deferred`
2. `phase_finalizer_started/completed/failed`
3. `forbidden_canonical_write_blocked`
4. `subtask_output_conflict_starvation_warning`
5. `worker_artifacts_ready`
6. `phase_finalizer_pending`
7. per-run summary:
   - conflicts avoided count
   - blocked write attempts count
   - fan_in phases completed count
   - max deferral streak
   - transactional publish rollback count

## Rollout and Backward Compatibility
1. Phase 1 (safe): global conflict serialization in `direct`.
2. Phase 2 (opt-in): enable `fan_in` by process.
3. Phase 3 (candidate default): flip only after passing built-in process suite and telemetry thresholds.
4. Keep escape hatch:
   - process-level `strategy: direct` remains supported.

Default-flip go/no-go gates:
1. No starvation alerts above agreed threshold in staged rollouts.
2. No increase in terminal failure rate for built-in processes.
3. No unresolved `forbidden_output_path` loops after remediation tuning.
4. All process tests passing with both `direct` and opt-in `fan_in` cohorts.

## Risk Assessment
1. Risk: over-serialization reduces throughput.
   - Mitigation: serialize only overlapping canonical writers; preserve parallelism otherwise.
2. Risk: planner-generated subtasks missing phase IDs.
   - Mitigation: continue phase inference + strict mapping fallback.
3. Risk: prompt confusion on intermediate artifact targets.
   - Mitigation: explicit prompt contract blocks for worker vs finalizer roles.
4. Risk: finalizer transactional commit complexity introduces publish bugs.
   - Mitigation: stage/commit abstraction with dedicated fault-injection tests.
5. Risk: artifact store grows unbounded.
   - Mitigation: retention/GC policy with run-scoped cleanup points.

## Unstaged Changes Review (Current Workspace)
Observed unstaged files:
1. `<repo-root>/src/loom/engine/orchestrator.py`
2. `<repo-root>/tests/test_orchestrator.py`

Assessment:
1. These implement and test phase-hint deliverable mapping in retry/remediation path.
2. They are directly aligned with output-ownership correctness and reduce misrouting to wrong filenames.

Recommendation:
1. Do not revert these two changes.
2. Keep them as prerequisite hardening before fan-in/ownership work.

## Acceptance Criteria
1. No parallel batch may run two subtasks that can mutate the same canonical deliverable path.
2. In `fan_in`, only phase finalizer may write canonical deliverables.
3. Worker writes to canonical deliverables are blocked deterministically.
4. Transactional finalizer publish prevents partial canonical output state on failure.
5. Deferred conflicting writers cannot starve indefinitely.
6. Existing `direct` processes remain functional without YAML updates.
7. New and existing tests cover conflict prevention, fairness, and final-output correctness.
