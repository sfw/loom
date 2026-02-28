# Slash Command Hint Catalog Hardening Plan (2026-02-28)

## Objective
Ship a production-ready slash hint UX in the TUI that is:
1. Complete: all eligible commands and processes are visible.
2. Scrollable: long catalogs remain usable in constrained terminal sizes.
3. Better ordered: results are predictable and grouped by task relevance.

## Why This Plan
Current behavior is functionally useful but operationally brittle for larger command catalogs:
1. Root slash hints can exceed viewport capacity and do not reliably expose full content at once.
2. `/process use` hint rows are intentionally truncated.
3. Ordering mixes static declaration order with dynamic alphabetical append, which is not task-priority oriented.
4. Tests mostly validate presence and prefix filtering, not complete rendering behavior under large catalogs.

## Current Baseline (Repo-Accurate)
1. Built-in slash commands are defined in `_SLASH_COMMANDS` in `src/loom/tui/app.py`.
2. Root hint text is generated via `_render_slash_hint("/")` and command catalogs from `_slash_command_catalog()`.
3. Process command aliases are generated dynamically from `ProcessLoader.list_available()` and filtered for built-in name collisions.
4. Hint panel is `#slash-hint` and currently uses line-count-driven sizing in `_set_slash_hint(...)`.
5. `/process use` hints currently cap rows via `max_rows = 12` and append an overflow line.
6. Completion and tab-cycle ordering are currently tied to `_SLASH_COMMANDS` declaration order and dynamic sorted process tokens.

## Production Requirements
1. Completeness:
2. Root `/` catalog must include all built-ins and all non-colliding dynamic process commands.
3. `/process use` contextual hints must list all matching processes, not an arbitrary top-N subset.
4. Scroll UX:
5. Hint panel must support vertical scrolling for long content.
6. Scroll behavior must work with mouse wheel and standard Textual scroll methods.
7. Input workflow must remain intact while hints are visible.
8. Ordering:
9. Built-ins should be grouped and ordered by practical workflow priority, not only declaration sequence.
10. Dynamic process commands should remain deterministic (alphabetical within group).
11. Prefix matching must preserve exact/prefix relevance before loose substring fallback.
12. Reliability and performance:
13. No crashes when process index refresh fails; fallback state remains readable.
14. Rendering should remain responsive with large process catalogs (for example 200+ processes).

## Non-Goals
1. Changing slash command semantics or adding/removing commands.
2. Redesigning the command palette (`Ctrl+P`) behavior.
3. Reworking process loader precedence or process schema naming rules.
4. Introducing network/data-store dependencies for hint rendering.

## Architecture Decisions
1. Treat the hint panel as a scroll container, not a static line-count-clamped block.
2. Separate ordering policy from command definition:
3. Keep `_SLASH_COMMANDS` as source of command metadata.
4. Add an explicit ordering/grouping policy layer used by hint rendering and optionally completion ranking.
5. Keep process command generation logic intact, but ensure rendering paths never hard-truncate matching rows.
6. Preserve collision-blocking behavior and surface blocked process commands clearly.

## Proposed Design

### 1) Scrollable Hint Container
1. Replace current static hint mounting with a scroll-capable structure:
2. `#slash-hint` as `VerticalScroll` container.
3. Child static/text body node for content updates (for example `#slash-hint-body`).
4. Update CSS to:
5. Keep hint panel docked at bottom.
6. Use bounded viewport height with `overflow-y` enabled.
7. Avoid line-count hard clamping as the primary visibility mechanism.
8. Update `_set_slash_hint(...)` to:
9. Update body text.
10. Toggle container visibility.
11. Reset scroll position to top on new hint payloads.

### 2) Complete Catalog Rendering
1. Root `/` renderer:
2. Always include full built-in and dynamic process entries.
3. Never trim rendered command rows for root catalog.
4. `/process use` renderer:
5. Remove `max_rows` truncation.
6. Render all matching rows with the same scroll container support.
7. Keep active-process marker behavior.
8. Preserve no-results and blocked-collision messaging.

### 3) Better Ordering Policy
1. Introduce deterministic grouped ordering for built-ins:
2. Session lifecycle: `/new`, `/sessions`, `/resume`, `/history`, `/session`.
3. Execution: `/run`, `/process`, `/processes`.
4. Integrations/auth: `/mcp`, `/auth`, `/tools`.
5. Environment/meta: `/model`, `/tokens`, `/setup`, `/help`, `/clear`, `/quit`.
6. Keep aliases tied to canonical command entries; do not duplicate rows.
7. Keep dynamic process commands in a dedicated section sorted alphabetically.
8. For prefix searches:
9. Rank exact token match first.
10. Then prefix matches.
11. Then substring fallback matches.
12. Preserve deterministic tie-break ordering.

### 4) Hardening for Dynamic Process Refresh
1. Keep current refresh cadence and collision filtering.
2. Ensure hint render paths never suppress existing catalog when refresh fails transiently.
3. Avoid list flicker during background refresh by rendering last known good catalog until refresh completion.
4. Continue showing blocked process collision notes in full help output.

## Workstreams and File Touchpoints

### W1: Hint Container Refactor
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`

Deliverables:
1. Scrollable hint container implementation.
2. Updated `_set_slash_hint(...)` logic.
3. CSS rules aligned with scrollable behavior.

### W2: Catalog Completeness
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`

Deliverables:
1. Remove `/process use` top-N truncation.
2. Ensure root `/` includes complete dynamic command list.

### W3: Ordering and Ranking
Files:
1. `/Users/sfw/Development/loom/src/loom/tui/app.py`
2. `/Users/sfw/Development/loom/tests/test_tui.py`

Deliverables:
1. Explicit ordering/grouping policy.
2. Prefix/fallback ranking consistency.
3. Updated completion order expectations where intended.

### W4: Tests and Regression Coverage
Files:
1. `/Users/sfw/Development/loom/tests/test_tui.py`

Deliverables:
1. Coverage for complete rendering with large command/process lists.
2. Coverage for scroll behavior (widget-level and run_test path where feasible).
3. Updated ordering/tab-cycle expectations.
4. Coverage for non-regression of no-match and alias matching behavior.

## Acceptance Criteria
1. Typing `/` shows a complete command catalog (built-ins + all eligible process commands).
2. Typing `/process use` with many processes shows all matches, with no `... and N more` truncation.
3. Hint panel is scrollable when content exceeds visible height.
4. Command ordering is grouped and deterministic, matching documented policy.
5. Existing slash behaviors remain correct:
6. Alias matching.
7. No-match fallback.
8. Collision-blocked process names are still excluded from dynamic slash commands.
9. Test suite passes with updated and additional coverage.

## Test Strategy

### Unit Tests
1. `_slash_command_catalog()` returns grouped/deterministic order.
2. `_matching_slash_commands(...)` respects exact > prefix > substring ranking.
3. `_render_process_use_hint(...)` includes all rows for large catalogs.
4. `_set_slash_hint(...)` updates scroll container and visibility correctly.

### TUI Integration Tests
1. With a large mock process catalog, root `/` hint renders expected first and last entries.
2. Hint container reports scrollable overflow when content exceeds viewport.
3. Tab completion cycle order matches updated policy.
4. Input focus and submission paths remain unaffected by hint visibility.

### Regression
1. Existing slash help and slash command handling tests remain green after intentional order expectation updates.
2. No regressions in `/process`, `/processes`, `/run`, `/resume`, `/help` command handling.

## Risks and Mitigations
1. Risk: ordering changes break user muscle memory and tests.
2. Mitigation: document order policy, update tests explicitly, keep consistent across hint/completion/help.
3. Risk: scroll-container refactor introduces focus/keyboard side effects.
4. Mitigation: add integration tests for input focus and tab-completion with visible hints.
5. Risk: very large catalogs degrade render speed.
6. Mitigation: precompute escaped rows once per render pass and avoid repeated expensive formatting.

## Rollout and Validation
1. Implement behind normal behavior without feature flags (local TUI-only surface).
2. Validate manually on small and large process catalogs.
3. Validate in narrow terminal and wide terminal layouts.
4. Run targeted tests:
5. `uv run pytest tests/test_tui.py -k slash`
6. Full TUI suite before merge.
7. Update changelog entry describing complete/scrollable/order improvements for slash hints.

## Estimated Effort
1. W1 + W2: 2 to 4 hours.
2. W3 + W4: 2 to 4 hours.
3. Total: 4 to 8 hours including test updates and manual validation.
