# Dead Code Cleanup Plan (2026-02-22)

## Scope
This plan targets dead or orphaned Python code in `src/loom/` using reference scans plus full-suite coverage as evidence.

## Evidence Snapshot
- `uv run ruff check src tests --select F401,F841` -> no unused imports/locals (dead code is mostly orphaned symbols/modules, not lint-level issues).
- `uv run pytest --cov=loom --cov-report=term-missing:skip-covered -q` -> `1682 passed, 50 skipped`, overall `69%` coverage.
- Symbol reference scan (`rg` + AST pass) identified symbols with exactly one source reference (their own definition).
- Confirmation query:
  - `rg -n "...candidate symbols..." src -S` returned only definitions for the targets listed below.

## Findings

### High-confidence dead symbols (no runtime or test references)
1. `src/loom/api/schemas.py:91` `ErrorResponse`
2. `src/loom/api/schemas.py:124` `ContentBlockResponse`
3. `src/loom/auth/runtime.py:64` `_resolve_env_value`
4. `src/loom/cowork/approval.py:143` `async_terminal_approval_prompt`
5. `src/loom/mcp/config.py:598` `validate_legacy_toml`
6. `src/loom/models/retry.py:51` `is_retryable_model_error`
7. `src/loom/research/text.py:40` `coerce_int`
8. `src/loom/tools/inflation_calculator.py:217` `supported_year_range`
9. `src/loom/exceptions.py:14` `EngineError` (+ file-level hierarchy appears unused)

### High-confidence dead production modules (test-only usage)
1. `src/loom/cowork/display.py`
- Display entrypoints (`display_tool_start`, `display_tool_complete`, etc.) have no runtime references.
- Only `_extract_diff` is referenced in tests (`tests/test_tools.py:1110`).

2. `src/loom/prompts/constraints.py`
- Constraint constants/helpers are referenced only in tests (`tests/test_prompts.py:11` onward).
- Runtime prompt assembly currently uses template files in `src/loom/prompts/templates/`.

3. `src/loom/tui/api_client.py`
- `LoomAPIClient` is referenced only in tests (`tests/test_tui.py:14` onward).
- Current TUI path uses in-process orchestration (`src/loom/tui/app.py`), not this HTTP client.

## Cleanup Sequence

### Phase 1: Remove isolated orphan symbols (low-risk)
1. Delete dead symbols from:
- `src/loom/api/schemas.py`
- `src/loom/auth/runtime.py`
- `src/loom/cowork/approval.py`
- `src/loom/mcp/config.py`
- `src/loom/models/retry.py`
- `src/loom/research/text.py`
- `src/loom/tools/inflation_calculator.py`
- `src/loom/exceptions.py` (remove file if no imports emerge)
2. Run targeted tests for touched areas.
3. Run full test suite.

### Phase 2: Remove or re-home test-only modules
1. `src/loom/cowork/display.py`
- Preferred: remove module and drop `_extract_diff` tests from `tests/test_tools.py`.
- Alternative: if terminal cowork mode is planned, wire module into CLI flow explicitly and keep it.

2. `src/loom/prompts/constraints.py`
- Preferred: remove module and tests that only validate this orphaned layer.
- Alternative: integrate it into prompt assembly and de-duplicate template constraints.

3. `src/loom/tui/api_client.py`
- Preferred: remove module and legacy tests.
- Alternative: reintroduce server-mode TUI path and make this client active runtime code.

### Phase 3: Add dead-code guardrails
1. Add a CI check script for single-reference top-level symbols (allowlist dynamic plugin surfaces).
2. Keep `ruff` unused checks enabled and add `ERA001` policy decision:
- either enforce removal of commented-out code,
- or disable rule if comments are intentionally explanatory.
3. Add a quarterly `dead-code` audit checklist to `planning/refactors/PLAN.md`.

## Verification Gates
- `uv run ruff check src tests`
- `uv run pytest -q`
- `uv run pytest --cov=loom --cov-report=term-missing:skip-covered -q`
- `rg -n "<removed symbol names>" src tests` should return zero results (except changelog/planning docs).

## Risk Notes
- Tool discovery is dynamic (`src/loom/tools/__init__.py`), so module-level import graph alone is not sufficient evidence for tool deadness.
- For `cowork/display`, `prompts/constraints`, and `tui/api_client`, removal should be done only after confirming no near-term roadmap depends on these legacy paths.

## Exit Criteria
1. All phase-1 symbols removed and tests pass.
2. A decision recorded for each phase-2 module: `remove` or `reactivate`.
3. Dead-code guardrail added to prevent re-accumulation.
