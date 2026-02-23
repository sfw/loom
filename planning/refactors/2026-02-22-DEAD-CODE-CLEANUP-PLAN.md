# Dead Code Cleanup Plan (2026-02-22, revalidated 2026-02-23)

## Applicability Status (2026-02-23)
Status: still applicable.

Revalidation summary:
- All phase-1 symbol candidates are still present and still definition-only across `src/` and `tests/`.
- All phase-2 module candidates are still present and still test-only.
- No cleanup in this plan appears to have landed yet.

## Scope
This plan targets dead or orphaned Python code in `src/loom/` using reference scans plus test evidence.

## Evidence Snapshot

### 2026-02-22 baseline (original plan)
- `uv run ruff check src tests --select F401,F841` -> no unused imports/locals.
- `uv run pytest --cov=loom --cov-report=term-missing:skip-covered -q` -> `1682 passed, 50 skipped`, overall `69%` coverage.
- Symbol reference scan (`rg` + AST pass) identified symbols with exactly one source reference (their own definition).

### 2026-02-23 revalidation (current)
- `rg -n "\bErrorResponse\b|\bContentBlockResponse\b|\b_resolve_env_value\b|\basync_terminal_approval_prompt\b|\bvalidate_legacy_toml\b|\bis_retryable_model_error\b|\bcoerce_int\b|\bsupported_year_range\b|\bEngineError\b" src tests` -> only definition lines.
- `rg -n "from\\s+loom\\.cowork\\.display|import\\s+loom\\.cowork\\.display|\bdisplay_tool_start\b|\bdisplay_tool_complete\b|\b_extract_diff\b" src tests` -> only `src/loom/cowork/display.py` plus `_extract_diff` usage in `tests/test_tools.py`.
- `rg -n "from\\s+loom\\.prompts\\.constraints|import\\s+loom\\.prompts\\.constraints|\bget_constraints_for_role\b|\bCOMMON_CONSTRAINTS\b" src tests` -> only `src/loom/prompts/constraints.py` plus `tests/test_prompts.py`.
- `rg -n "from\\s+loom\\.tui\\.api_client|import\\s+loom\\.tui\\.api_client|\bLoomAPIClient\b" src tests` -> only `src/loom/tui/api_client.py` plus `tests/test_tui.py`.
- `rg -n "\bLoomError\b|\bEngineError\b|\bModelError\b|\bToolError\b|\bStateError\b" src tests` -> only `src/loom/exceptions.py` definitions.

## Current Findings

### Phase 1 candidates (still dead as of 2026-02-23)
1. `src/loom/api/schemas.py:91` `ErrorResponse`
2. `src/loom/api/schemas.py:124` `ContentBlockResponse`
3. `src/loom/auth/runtime.py:64` `_resolve_env_value`
4. `src/loom/cowork/approval.py:143` `async_terminal_approval_prompt`
5. `src/loom/mcp/config.py:598` `validate_legacy_toml`
6. `src/loom/models/retry.py:51` `is_retryable_model_error`
7. `src/loom/research/text.py:40` `coerce_int`
8. `src/loom/tools/inflation_calculator.py:217` `supported_year_range`
9. `src/loom/exceptions.py:14` `EngineError` (and sibling hierarchy in `src/loom/exceptions.py` appears unused)

### Phase 2 candidates (still test-only as of 2026-02-23)
1. `src/loom/cowork/display.py`
- Display entrypoints (`display_tool_start`, `display_tool_complete`, etc.) have no runtime references.
- Only `_extract_diff` is referenced in tests (`tests/test_tools.py:1110` onward).
2. `src/loom/prompts/constraints.py`
- Constraint constants/helpers are referenced in tests (`tests/test_prompts.py`) but not runtime code.
- Runtime prompt assembly currently uses template files in `src/loom/prompts/templates/`.
3. `src/loom/tui/api_client.py`
- `LoomAPIClient` is referenced in tests (`tests/test_tui.py`) but not runtime code.
- Current TUI path uses in-process orchestration (`src/loom/tui/app.py`), not this HTTP client.

## Execution Status
- Phase 1: not started.
- Phase 2: not started.
- Phase 3: not started.

## Cleanup Sequence

### Phase 1: Remove isolated orphan symbols (low risk)
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
