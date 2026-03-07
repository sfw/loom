# 2026-03-07 Engine Verification Split Plan

## File

- `src/loom/engine/verification.py` (~4,797 LOC)

## Objective

- Split verification into maintainable modules while keeping tier semantics, outcome aggregation, and instrumentation behavior unchanged.
- Preserve public API imports:
  - `Check`
  - `VerificationResult`
  - `DeterministicVerifier`
  - `LLMVerifier`
  - `VotingVerifier`
  - `VerificationGates`

## Current Hotspots

- Tier implementations and orchestration are co-located in one file.
- Large methods:
  - `VerificationGates.verify` (~247 LOC)
  - `VerificationGates._scan_placeholder_markers` (~280 LOC)
  - `LLMVerifier.verify` (~245 LOC)
  - `LLMVerifier.__init__` (~196 LOC)
  - `LLMVerifier._invoke_and_parse` (~104 LOC)
  - `DeterministicVerifier.verify` (~221 LOC)
- Additional complexity:
  - Placeholder contradiction guard and filesystem scanning
  - YAML/JSON-like assessment parsing and coercion
  - Dual policy systems (legacy vs policy engine + shadow comparison)
  - Event instrumentation scattered across layers

## Initial Split Plan

### Target Package Layout

- `src/loom/engine/verification/`
  - Package replacing monolithic `verification.py` module.
- `src/loom/engine/verification/__init__.py`
  - Compatibility facade with class re-exports.
- `src/loom/engine/verification/types.py`
  - `Check`, `VerificationResult`, severity inference utilities.
- `src/loom/engine/verification/tier1.py`
  - `DeterministicVerifier` and placeholder regex extraction helpers.
- `src/loom/engine/verification/tier2.py`
  - `LLMVerifier` high-level flow.
- `src/loom/engine/verification/prompting.py`
  - Prompt assembly, tool/evidence/artifact section builders.
- `src/loom/engine/verification/parsing.py`
  - Response parsing/coercion/repair fallback pipeline.
- `src/loom/engine/verification/placeholder_guard.py`
  - Placeholder marker scanning + contradiction guard logic.
- `src/loom/engine/verification/policy.py`
  - Result aggregation, legacy fallback, shadow diff classification.
- `src/loom/engine/verification/events.py`
  - Event emission helpers and payload construction.
- `src/loom/engine/verification/gates.py`
  - `VerificationGates` orchestration shell.

### Refactor Pattern

- Keep tier classes explicit; avoid turning them into opaque generic “strategies.”
- Move pure logic first (types/parsing/policy), then side-effecting logic (filesystem scan, event emission).
- Maintain class names and constructor signatures for test compatibility.

## Incremental Execution Plan

### Phase 1: Extract Stable Types and Policy Functions

- Convert `verification.py` into `verification/` package with `__init__.py` facade.
- Move `Check`, `VerificationResult`, outcome/severity constants.
- Move shadow diff classification and non-failing aggregation to policy module.
- Add strict parity tests for outcome normalization and severity derivation.

### Phase 2: Extract Deterministic Tier

- Move `DeterministicVerifier` plus placeholder finding helpers.
- Preserve syntax-check and deliverable checks exactly.
- Add targeted tests for:
  - advisory vs hard tool failure behavior
  - synthesis input integrity checks
  - placeholder finding extraction and dedupe

### Phase 3: Split Tier-2 Internals

- Keep `LLMVerifier.verify` in one place but extract:
  - prompt assembly
  - assessment parsing/coercion
  - response repair/fallback parsing
- Add parser golden tests using malformed verifier outputs.

### Phase 4: Extract Placeholder Contradiction Guard

- Move candidate path discovery and marker scanning.
- Move contradiction guard application logic.
- Add tests covering:
  - symlink guard behavior
  - suffix and directory filtering
  - placeholder claim failure promotion

### Phase 5: Extract Event Instrumentation

- Centralize `VERIFICATION_*` event payload builders.
- Add contract tests for emitted payload shape and reason/outcome fields.

### Phase 6: Shrink `VerificationGates`

- Retain only orchestration of tier execution order and terminalization logic.
- Keep policy engine toggles and shadow compare behavior intact.

## Execution Status

- Phase 1a complete (2026-03-06 local): atomic module->package conversion (`verification.py` -> `verification/__init__.py`) with import/verification parity tests.
- Phase 1 complete (2026-03-06 local): extracted `verification/types.py` (`Check`, `VerificationResult`) and `verification/policy.py` (shadow diff classification + non-failing aggregation) with focused policy tests.
- Phase 2 complete (2026-03-06 local): moved `DeterministicVerifier` (including placeholder finding helpers) into `verification/tier1.py` with parity coverage on verification + consumer suites.
- Phase 3 complete (2026-03-06 local): extracted tier-2 parsing/coercion into `verification/parsing.py` and prompt/repair prompt assembly helpers into `verification/prompting.py` with compatibility wrappers on `LLMVerifier`.
- Phase 4 complete (2026-03-06 local): extracted placeholder guard flow into `verification/placeholder_guard.py` (candidate discovery, scan routine, contradiction application) with parity tests.
- Phase 5 complete (2026-03-06 local): centralized verification/rule/model event emissions into `verification/events.py` and added payload contract coverage in `tests/test_verification_events.py`.
- Phase 6 complete (2026-03-06 local): `VerificationGates` now delegates claim lifecycle, placeholder-scan path helpers, event instrumentation, and legacy/policy terminalization to sibling modules (`claims.py`, `placeholder_guard.py`, `events.py`, `policy.py`) while retaining orchestration and compatibility wrappers.
- Phase 6b complete (2026-03-07 local): extracted `LLMVerifier` into `verification/tier2.py` and `VotingVerifier`/`VerificationGates` into `verification/gates.py`; `verification/__init__.py` is now a thin compatibility facade exporting public contracts.

## Critique of Initial Plan

- Risk: Parsing logic extraction can change edge-case coercion behavior that currently tolerates malformed verifier responses.
- Risk: Policy vs legacy result paths are subtle; refactor can alter production pass/fail rates.
- Risk: Placeholder scan extraction can inadvertently broaden filesystem reads or reduce safety filtering.
- Risk: Many tests mock private members (`_tier2.verify`, internals); module moves can break those test seams.
- Risk: Event payload changes can silently break telemetry downstream even if verification result objects remain correct.

## Hardened Plan

- Freeze current behavior with characterization fixtures before moving code:
  - malformed tier-2 outputs
  - inconclusive tier-2 fallback to tier-1
  - shadow-compare diff classification
  - placeholder contradiction downgrade/upgrade
- Keep canonical outcome decision table in one module and reference it from both policy and gates logic.
- Introduce explicit `VerificationContext` object passed through gate flow to avoid argument drift.
- Add no-op compatibility wrappers for commonly accessed private methods during migration window.
- Treat event payload schema as contract: snapshot test at module boundary before and after extraction.

## Missing Items Added

- Package-conversion mechanics:
  - Perform `git mv src/loom/engine/verification.py src/loom/engine/verification/__init__.py` first to preserve `loom.engine.verification` import behavior.
  - Keep `Check`, `VerificationResult`, and verifier class exports stable in `verification/__init__.py`.
- Cross-plan sequencing:
  - Split semantic compactor package first (or in an isolated preparatory PR) so tier-2 compaction dependencies remain stable during verification extraction.
  - Land verification refactor before final orchestrator extraction to reduce simultaneous behavioral movement in gate callers and gate implementation.
- Scan safety and determinism:
  - Preserve contradiction-scan exclusion rules and file-size guards exactly; add tests for worst-case directory breadth.
  - Add timeout/latency budget assertions for placeholder scanning on large workspaces.
- Circular import guardrails:
  - Prevent `verification/__init__.py` from being imported by internal submodules; import concrete siblings only.
  - Add import-cycle smoke tests for tier/policy/events modules.
- Policy integrity:
  - Snapshot policy-engine vs legacy-engine decisions for a fixed corpus and require zero diff unless explicitly approved.

## Test Strategy

- Existing suites:
  - `tests/test_verification.py`
  - `tests/test_verification_golden.py`
  - `tests/test_orchestrator.py`
  - `tests/test_confidence.py`
- New suites:
  - `tests/test_verification_policy.py`
  - `tests/test_verification_parsing.py`
  - `tests/test_verification_placeholder_guard.py`
  - `tests/test_verification_events.py`
  - `tests/test_engine_import_contracts.py` (import/re-export parity for `loom.engine.verification`)

## Exit Criteria

- `src/loom/engine/verification/__init__.py` facade <= 200 LOC.
- `src/loom/engine/verification/gates.py` focused orchestration shell <= 500 LOC.
- Tier and policy modules each focused and independently testable.
- No change in golden verification outcomes or shadow diff classifications.
- No public import path breakage.
