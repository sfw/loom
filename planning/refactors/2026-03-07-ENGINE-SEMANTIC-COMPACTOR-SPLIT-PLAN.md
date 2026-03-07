# 2026-03-07 Engine Semantic Compactor Split Plan

## File
- `src/loom/engine/semantic_compactor.py` (~1,157 LOC)

## Objective
- Keep `SemanticCompactor` behavior stable while isolating cache concurrency, prompt/budget math, model invocation/retry, and response parsing/validation.
- Preserve external constructor and `compact()` API used by runner, verification, cowork, learning, and tools.

## Current Hotspots
- Compaction pipeline + retries:
  - `_compact_once` (~160 LOC)
  - `_invoke_compactor_model` (~141 LOC)
  - `compact` (~122 LOC)
- Validation and recovery logic:
  - `_next_validation_retry`
  - `_extract_compacted_text`
  - `_extract_partial_compressed_text`
  - `_validate_compacted_text`
- Cache/inflight synchronization in same class as model logic.

## Initial Split Plan

### Target Package Layout
- `src/loom/engine/semantic_compactor/`
  - Package replacing monolithic `semantic_compactor.py` module.
- `src/loom/engine/semantic_compactor/__init__.py`
  - Compatibility facade exposing `SemanticCompactor`.
- `src/loom/engine/semantic_compactor/config.py`
  - Config defaults and budget math helpers (`target chars`, `hard limit`, `token ceiling`).
- `src/loom/engine/semantic_compactor/cache.py`
  - Cache keying, inflight dedupe, lock-scoped ownership flow.
- `src/loom/engine/semantic_compactor/pipeline.py`
  - `compact`, chunked map/reduce, round-based reduction orchestration.
- `src/loom/engine/semantic_compactor/model.py`
  - model selection, invocation, retry policy wiring, model-event emission.
- `src/loom/engine/semantic_compactor/parse.py`
  - JSON/text extraction, fence stripping, partial recovery, output validation.

### Refactor Pattern
- Keep one public class (`SemanticCompactor`) composing internal helpers.
- Avoid introducing multiple public classes or strategy indirection.
- Keep all default constant values and constructor parameters unchanged.

## Incremental Execution Plan

### Phase 1: Extract Parse/Validation Helpers
- Convert `semantic_compactor.py` into `semantic_compactor/` package with `__init__.py` facade.
- Move text extraction and validation methods to parse module.
- Keep method wrappers in class temporarily.
- Add parser tests for:
  - fenced outputs
  - invalid JSON recovery
  - partial JSON fallback on `finish_reason=length`

### Phase 2: Extract Config/Budget Math
- Move token/char budget helpers and boundary clamps.
- Add boundary tests for small and near-budget targets.

### Phase 3: Extract Cache/Inflight Logic
- Move cache key + inflight join/owner flow into dedicated helper.
- Add async tests for concurrent `compact()` calls on same payload.

### Phase 4: Extract Model Invocation Layer
- Move selection/fallback role logic and retry-wrapped invocation.
- Keep event payload keys unchanged.
- Add tests for:
  - role fallback path
  - temperature-one-only retry behavior
  - retry/no-retry classification

### Phase 5: Extract Pipeline Orchestration
- Move chunked/map-reduce and round reduction flow.
- Keep `compact()` in class delegating to pipeline helper.
- Add end-to-end compaction parity tests against existing fixtures.

## Execution Status

- [x] Phase 1 complete (2026-03-06 local): atomic module->package conversion, parse extraction with class wrappers, parser parity tests, and import contract tests.
- [x] Phase 2 complete (2026-03-06 local): budget/config helper extraction with class wrappers and boundary tests.
- [x] Phase 3 complete (2026-03-06 local): cache/inflight helper extraction with lock-scope parity and cancellation/error-path dedupe cleanup tests.
- [x] Phase 4 complete (2026-03-06 local): model selection/invocation helper extraction with retry classification tests and event payload parity via existing suite.
- [x] Phase 5 complete (2026-03-06 local): pipeline orchestration extracted with compact wrapper delegation and pipeline parity tests.

## Critique of Initial Plan
- Risk: Over-splitting a 1.1k file could increase indirection without enough value.
- Risk: Cache/inflight extraction can create subtle race regressions if lock boundaries change.
- Risk: Budget helper extraction can alter off-by-one behavior, changing model max token requests.
- Risk: Event payload drift can reduce observability in runner/verification without obvious failures.

## Hardened Plan
- Limit split to four internal modules (config, cache, parse, model/pipeline combined) unless LOC pressure remains.
- Preserve exact lock scope and owner/inflight semantics with targeted concurrency tests before/after extraction.
- Freeze emitted event payload keys via snapshot tests.
- Add strict parity test corpus:
  - same input text + max_chars => same output or same fallback class
  - same retry path for known malformed model outputs
- Include lightweight micro-benchmark guard to detect major latency regressions from refactor overhead.

## Missing Items Added
- Package-conversion mechanics:
  - Perform `git mv src/loom/engine/semantic_compactor.py src/loom/engine/semantic_compactor/__init__.py` first to preserve `loom.engine.semantic_compactor` imports.
  - Keep `SemanticCompactor` constructor and defaults byte-for-byte compatible at the facade boundary.
- Cross-plan sequencing:
  - Land semantic compactor package conversion before runner/verification deep extraction phases, since both depend on compactor internals.
- Cancellation and backpressure safety:
  - Add tests confirming inflight dedupe futures resolve correctly on cancellation/error paths.
  - Preserve cache trimming and inflight cleanup behavior to avoid leaked futures and memory growth.
- Circular import guardrails:
  - Prevent internal modules from importing `semantic_compactor/__init__.py`; import concrete siblings only.
  - Add import-cycle smoke tests for config/cache/model/pipeline/parse modules.
- Consumer parity:
  - Add integration checks for direct consumers (`runner`, `verification`, `cowork.session`, `learning.reflection`, `tools.conversation_recall`) to confirm no behavior drift at call sites.

## Test Strategy
- Existing suites:
  - `tests/test_semantic_compactor.py`
  - indirect coverage from runner/verification tests
- New focused suites:
  - `tests/test_semantic_compactor_cache.py`
  - `tests/test_semantic_compactor_parse.py`
  - `tests/test_semantic_compactor_budget.py`
  - `tests/test_semantic_compactor_events.py`
  - `tests/test_engine_import_contracts.py` (import/re-export parity for `loom.engine.semantic_compactor`)

## Exit Criteria
- `src/loom/engine/semantic_compactor/__init__.py` facade <= 200 LOC.
- `src/loom/engine/semantic_compactor/pipeline.py` focused flow <= 350 LOC.
- No regression in compactor output validity guarantees.
- No regression in concurrency dedupe behavior.
- No external API or constructor signature breakage.
