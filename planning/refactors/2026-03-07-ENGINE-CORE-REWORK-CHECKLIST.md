# 2026-03-07 Engine Core Rework Agent Manual

## Purpose
This document is an execution manual for an agent implementing the full core-engine rework across these four plan files:

- `planning/refactors/2026-03-07-ENGINE-SEMANTIC-COMPACTOR-SPLIT-PLAN.md`
- `planning/refactors/2026-03-07-ENGINE-RUNNER-SPLIT-PLAN.md`
- `planning/refactors/2026-03-07-ENGINE-VERIFICATION-SPLIT-PLAN.md`
- `planning/refactors/2026-03-07-ENGINE-ORCHESTRATOR-SPLIT-PLAN.md`

The goal is to land the rework safely, in small revertable slices, with zero import-path breakage and no behavior regressions.

## Agent Mission
You are executing a high-risk refactor of the app’s core runtime. Your job is to:

1. Convert monolithic engine modules into package subfolders.
2. Extract internals incrementally as specified in each plan.
3. Preserve runtime behavior, event ordering, and public API imports.
4. Prove parity through tests and targeted snapshots at every phase.

## Non-Negotiable Constraints

1. Keep these import contracts valid after every PR:
   - `from loom.engine.orchestrator import Orchestrator, create_task`
   - `from loom.engine.runner import SubtaskRunner, SubtaskResult, ToolCallRecord`
   - `from loom.engine.verification import VerificationGates, VerificationResult`
   - `from loom.engine.semantic_compactor import SemanticCompactor`
2. Use atomic module-to-package conversion:
   - `git mv src/loom/engine/<module>.py src/loom/engine/<module>/__init__.py`
3. Do not import internal submodules through package facades (`__init__.py`).
4. Keep each phase small and revertable in one PR slice.
5. Do not combine multiple high-risk subsystem moves in the same PR.
6. No DB schema changes are expected. If schema changes become necessary, stop and follow `AGENTS.md` migration policy.

## Required Reading Before Changes

1. Read the four split plans listed above.
2. Read the current source modules:
   - `src/loom/engine/semantic_compactor.py`
   - `src/loom/engine/runner.py`
   - `src/loom/engine/verification.py`
   - `src/loom/engine/orchestrator.py`
3. Read existing tests that enforce behavior:
   - `tests/test_semantic_compactor.py`
   - `tests/test_verification.py`
   - `tests/test_orchestrator.py`
   - `tests/test_full_integration.py`
   - `tests/test_integrations.py`

## Global Execution Order

1. Semantic compactor package conversion and parity harness.
2. Runner package conversion and low-risk extractions.
3. Verification package conversion and deterministic/policy/parsing extraction.
4. Orchestrator extraction after runner and verification stabilize.

Do not change this order unless blocked and explicitly justified in the PR notes.

## PR Slicing Rules

Each PR slice must contain only one of these:

1. Package conversion only.
2. One subsystem extraction plus tests.
3. Compatibility cleanup after parity is already proven.

Each PR description must include:

1. What moved.
2. Why behavior is unchanged.
3. Which tests were run.
4. Any known residual risk.

## Standard Phase Workflow (Use For Every Slice)

1. Establish baseline:
   - Run targeted tests for impacted subsystem.
   - Capture baseline event payload keys/order where relevant.
2. Apply small change set:
   - Move files.
   - Add thin wrappers/re-exports.
   - Keep signatures stable.
3. Add or update tests immediately:
   - Import contract tests.
   - Parity tests for moved logic.
4. Run gates:
   - Unit tests first, then integration tests.
5. Verify runtime/observability parity:
   - Event payload keys unchanged.
   - Critical event ordering unchanged.
6. Document:
   - Update the related plan checklist status.
   - Record exact command/test evidence in PR notes.

## Test Gates (Minimum)

Run these across the full program, and at least the relevant subset for each phase:

- `tests/test_semantic_compactor.py`
- `tests/test_verification.py`
- `tests/test_orchestrator.py`
- `tests/test_full_integration.py`
- `tests/test_integrations.py`
- `tests/test_engine_import_contracts.py` (create and maintain)

Add and maintain:

1. Import/re-export parity tests for all four packages.
2. Import-cycle smoke tests for new submodule layouts.
3. Event payload snapshot tests for verification and orchestrator.
4. Event-order parity tests for orchestrator critical flows.

## Performance and Runtime Regression Gates

Track before/after metrics on representative flows:

1. End-to-end runtime.
2. Model invocation count.
3. Total token usage.
4. Verification latency.

If regression is material, stop and either:

1. Fix in same slice, or
2. Split out mitigation PR before continuing.

## Per-Subsystem Special Instructions

### Semantic Compactor

1. Preserve inflight dedupe semantics and cache lock boundaries.
2. Add cancellation/error-path tests for inflight cleanup.
3. Keep constructor defaults and behavior compatible.

### Runner

1. Introduce `RunnerSession` before deep loop extraction.
2. Keep `run()` behavior identical during incremental moves.
3. Avoid breaking private helper access used by current tests until migrated.

### Verification

1. Freeze policy-engine vs legacy-engine outcomes on a fixed corpus.
2. Preserve placeholder scan exclusions and file-size safety behavior.
3. Keep coercion/parsing behavior stable for malformed verifier output.

### Orchestrator

1. Preserve event sequencing and task-state mutation order.
2. Add debug/test invariants for plan version and status transitions.
3. Defer major orchestrator extraction until runner and verification are stable.

## Hard Stop Conditions

Stop and escalate before continuing if any occur:

1. Public import contract breaks.
2. Event-order parity breaks in orchestrator terminal flows.
3. Policy-vs-legacy verification decisions drift unexpectedly.
4. Core integration tests fail with unclear root cause.
5. Circular import appears in new package layout.

## Completion Criteria (Program-Level)

1. All four modules converted to package layouts with thin facades.
2. Planned extractions completed per subsystem plans.
3. All baseline and new regression tests passing.
4. Import contracts preserved.
5. Event payload and ordering parity validated.
6. Architecture docs and changelog updated.

## Suggested Command Checklist

1. `uv run pytest tests/test_engine_import_contracts.py`
2. `uv run pytest tests/test_semantic_compactor.py`
3. `uv run pytest tests/test_verification.py`
4. `uv run pytest tests/test_orchestrator.py`
5. `uv run pytest tests/test_full_integration.py tests/test_integrations.py`

Run narrower subsets during each slice, then full relevant gates before merge.
