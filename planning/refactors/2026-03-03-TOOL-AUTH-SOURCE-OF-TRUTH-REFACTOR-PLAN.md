# Tool Auth Source-Of-Truth Refactor Plan (2026-03-03)

## Objective
Eliminate static/manual auth classification drift by making each built-in tool the single source of truth for auth posture and auth requirements.

Target state:
1. Tool-level metadata defines auth posture and requirements.
2. Runtime, API, and TUI derive behavior from discovered tool instances.
3. `auth_inventory.py` is removed as a manually maintained mapping.

## Why This Change
Current state has two representations:
1. Dynamic runtime auth requirements from tools (`Tool.auth_requirements`).
2. Static inventory map (`src/loom/tools/auth_inventory.py`) enforced by tests.

This duplicates source-of-truth and creates drift risk in a modular dynamic tool system.

## Current-System Audit

### Runtime auth behavior
Auth preflight already relies on tool declarations:
1. API process preflight collects tool `auth_requirements` dynamically:
   - `src/loom/api/routes.py` (`_required_auth_resources_for_process`)
2. TUI run-start preflight does the same:
   - `src/loom/tui/app.py` (`_collect_required_auth_resources_for_process`)
3. Tool catalog currently exposes only a boolean `auth_required` derived from `bool(tool.auth_requirements)`.

### Static inventory behavior
1. `src/loom/tools/auth_inventory.py` contains `FIRST_PARTY_TOOL_AUTH_CLASSIFICATION`.
2. `tests/test_tool_auth_inventory.py` enforces inventory coverage and consistency.
3. No runtime path depends on this inventory (audit shows test/audit use only).

### Built-in tool review (all discovered first-party non-MCP tools)
Audit result:
1. Total built-ins discovered: `60`.
2. Tools overriding `auth_requirements`: `0`.
3. Tools with non-empty declared `auth_requirements`: `0`.
4. Effective current auth posture: all `no_auth`.

Implication:
1. Removing static inventory now is low behavioral risk because current declarations are uniformly empty.
2. We still need stronger tool-level contracts for future optional/required auth tools.

## Proposed Design (Single Source Of Truth)

### 1) Add explicit tool-level auth posture
Add a new `Tool` property in `src/loom/tools/registry.py`:
1. `auth_mode: Literal["no_auth", "optional_auth", "required_auth"]`
2. Default: `"no_auth"`.

Keep `auth_requirements` as the requirement payload source.

### 2) Add registry-time validation
During tool registration (`ToolRegistry.register`), enforce:
1. `auth_mode` must be one of allowed values.
2. `required_auth` or `optional_auth` requires non-empty `auth_requirements`.
3. `no_auth` requires empty `auth_requirements`.

Fail fast on invalid tool metadata to prevent silent drift.

### 3) Replace inventory-based checks with dynamic checks
Retire static map and replace tests with dynamic discovery invariants:
1. Every discovered first-party tool has valid `auth_mode`.
2. Mode/requirements consistency holds for every tool.
3. No references remain to `auth_inventory.py`.

### 4) Expose richer auth metadata to consumers
Update tool catalog responses to include `auth_mode` (not just `auth_required`):
1. Keep `auth_required` for backward compatibility (derived: mode != `no_auth`).
2. Add `auth_mode` and serialized `auth_requirements` summary where appropriate.

This supports clearer UX and safer policy decisions.

## Built-In Tool Classification Review (Proposed)
All currently discovered built-ins remain `no_auth` in this migration because no tool currently declares auth requirements.

Tool set reviewed:
1. `academic_search`
2. `agent_capabilities`
3. `agent_run`
4. `analyze_code`
5. `archive_access`
6. `ask_user`
7. `calculator`
8. `citation_manager`
9. `conversation_recall`
10. `correspondence_analysis`
11. `delegate_task`
12. `delete_file`
13. `document_write`
14. `earnings_surprise_predictor`
15. `economic_data_api`
16. `edit_file`
17. `fact_checker`
18. `factor_exposure_engine`
19. `filing_event_parser`
20. `git_command`
21. `glob_find`
22. `historical_currency_normalizer`
23. `humanize_writing`
24. `inflation_calculator`
25. `insider_trading_tracker`
26. `list_directory`
27. `list_tools`
28. `macro_regime_engine`
29. `market_data_api`
30. `move_file`
31. `opportunity_ranker`
32. `options_flow_analyzer`
33. `peer_review_simulator`
34. `portfolio_evaluator`
35. `portfolio_optimizer`
36. `portfolio_recommender`
37. `primary_source_ocr`
38. `read_artifact`
39. `read_file`
40. `ripgrep_search`
41. `run_tool`
42. `search_files`
43. `sec_fundamentals_api`
44. `sentiment_feeds_api`
45. `shell_execute`
46. `short_interest_analyzer`
47. `social_network_mapper`
48. `spreadsheet`
49. `symbol_universe_api`
50. `task_tracker`
51. `timeline_visualizer`
52. `valuation_engine`
53. `web_fetch`
54. `web_fetch_html`
55. `web_search`
56. `wp_cli`
57. `wp_env`
58. `wp_quality_gate`
59. `wp_scaffold_block`
60. `write_file`

## Implementation Plan

### Phase R1: Introduce tool-level auth_mode contract
Files:
1. `src/loom/tools/registry.py`

Tasks:
1. Add `Tool.auth_mode` property defaulting to `no_auth`.
2. Add internal validator helpers for mode/requirements coherence.
3. Enforce validation in `ToolRegistry.register` with clear error messages.

### Phase R2: Migrate catalog consumers to dynamic auth metadata
Files:
1. `src/loom/tools/__init__.py`
2. `src/loom/cowork/session.py`
3. `src/loom/tools/list_tools.py` (if needed for schema/rows)
4. `src/loom/api/schemas.py` (if response model needs explicit field)

Tasks:
1. Preserve existing `auth_required` behavior.
2. Add `auth_mode` in catalog rows.
3. Ensure no consumer requires static inventory.

### Phase R3: Remove static inventory module
Files:
1. Delete `src/loom/tools/auth_inventory.py`
2. Replace `tests/test_tool_auth_inventory.py` with dynamic tests

Tasks:
1. Remove dead imports/usages.
2. Add new tests that enumerate discovered tools and assert metadata invariants directly from tool instances.

### Phase R4: Explicitly annotate tools that need auth (future-ready)
Files:
1. Tool modules that eventually require credentials.

Tasks:
1. For any tool transitioning to authenticated providers, set:
   - `auth_mode = "required_auth"` or `"optional_auth"`
   - concrete `auth_requirements` payload
2. Add focused tests per tool.

## Test Strategy

### Unit tests
1. `ToolRegistry.register` rejects invalid `auth_mode`.
2. Reject `required_auth`/`optional_auth` with empty requirements.
3. Reject `no_auth` with non-empty requirements.
4. Catalog rows include coherent `auth_mode` + `auth_required` derivation.

### Integration tests
1. API process preflight still resolves required resources from tool declarations.
2. TUI run-start auth resolution still derives from tool declarations.
3. Discovery/listing behavior unaffected for tools with no auth.

### Regression checks
1. Full suite: `uv run ruff check` and `uv run pytest -q`.
2. Ensure no static inventory references remain.

## Rollout and Risk

### Rollout
1. Ship behind no flag (low risk; behavior-preserving for current all-no-auth tools).
2. Monitor for registration-time failures revealing hidden metadata bugs.

### Risks
1. Risk: latent tool definitions violate new validator.
   - Mitigation: add explicit error messages and quick-fix guidance.
2. Risk: catalog/API schema consumers assume only `auth_required` bool.
   - Mitigation: keep bool and add `auth_mode` as additive field.

### Rollback
1. Revert validator enforcement and restore previous permissive behavior.
2. Re-introduce static inventory only if critical regression appears (not recommended long-term).

## Definition of Done
1. `auth_inventory.py` removed.
2. No runtime path relies on static tool auth maps.
3. All built-in tools validated dynamically from tool metadata.
4. API/TUI/tool catalog expose consistent auth posture from tool instances.
5. Lint and full tests pass.

## Post-Refactor Follow-up
1. Add a developer doc section: "How to declare auth for a new tool".
2. Add PR checklist item requiring auth_mode/auth_requirements review when adding tools.
3. Optionally add CI script that prints a generated auth report artifact for visibility.
