# Auth Resource Lifecycle + Auto-Discovery Implementation Checklist (2026-02-27)

## Scope
Execution checklist for:
- [2026-02-27-AUTH-RESOURCE-LIFECYCLE-AUTODISCOVERY-PLAN.md](/Users/sfw/Development/loom/planning/refactors/2026-02-27-AUTH-RESOURCE-LIFECYCLE-AUTODISCOVERY-PLAN.md)

Status:
1. Implementation has not started.
2. This checklist is the execution contract.

## Global Rules
1. Keep runtime behavior backward compatible until PR7.
2. Preserve deterministic headless behavior at every PR boundary.
3. Use cross-process file locking + atomic writes for every config mutation path.
4. Do not remove legacy provider paths before migration/audit tooling ships.
5. Prefer additive migrations first, destructive cleanup last.

## PR1: Resource Registry + Storage Foundation

## Goals
1. Introduce resource registry storage (`resource_id` based).
2. Add account/binding/default data structures and persistence primitives.
3. Add schema version markers and snapshot backup helper.

## File Touchpoints
1. `src/loom/auth/config.py`
2. `src/loom/auth/runtime.py`
3. `src/loom/mcp/config.py` (resource lifecycle hooks, alias mapping support)
4. `src/loom/tui/app.py` (loader wiring only; no UX changes yet)
5. `tests/test_auth_config.py`
6. `tests/test_tui.py`
7. New: `src/loom/auth/resources.py`

## Tasks
1. Define models:
   - `AuthResource`
   - `AuthAccount` (internal)
   - `AuthBinding`
   - `AuthDefaultsResourceMap` (workspace + user)
2. Add parsers/renderers:
   - `.loom/auth.resources.toml`
   - schema version + read/upgrade hooks
3. Add lock + atomic write helpers reusable by auth/resource files.
4. Add snapshot backup helper used by migration paths.
5. Add read APIs without changing existing selection behavior.

## Exit Criteria
1. New stores load/save with no runtime usage changes yet.
2. Legacy config still loads unchanged.
3. Locks prevent concurrent write corruption in tests.

## PR2: Discovery Scanner + Draft Engine

## Goals
1. Build discovery scanner for active-scope resources.
2. Build idempotent draft account + binding generator.
3. Add optional full-scan mode.

## File Touchpoints
1. `src/loom/auth/resources.py`
2. `src/loom/tools/registry.py`
3. `src/loom/auth/runtime.py`
4. `src/loom/tui/screens/auth_manager.py` (sync hook only)
5. `tests/test_tool_auth_inventory.py`
6. `tests/test_auth_config.py`
7. `tests/test_tui.py`

## Tasks
1. Implement discovery sources:
   - process-required auth
   - tool-declared `auth_requirements`
   - enabled MCP aliases + setup-time MCP auth templates
2. Implement `sync_missing_drafts(scope="active"|"full")`.
3. Enforce idempotency key (`resource_id`, `mode`, `generated_slot`).
4. Add no-op write guard (skip write when content unchanged).
5. Add partial-failure contract:
   - sync failure does not block Auth Manager opening.

## Exit Criteria
1. `/auth` open sync path callable and safe.
2. Repeated sync produces no duplicate drafts.
3. Full-scan creates superset of active-scope drafts.

## PR3: Runtime Resolver (Resource-First + Safe Fallback)

## Goals
1. Resolve auth by `resource_id` first.
2. Keep legacy provider fallback with binding guard.
3. Emit unresolved status taxonomy and migration states.

## File Touchpoints
1. `src/loom/auth/runtime.py`
2. `src/loom/processes/schema.py`
3. `src/loom/tui/app.py`
4. `src/loom/api/routes.py`
5. `tests/test_processes.py`
6. `tests/test_tui.py`
7. `tests/test_api.py`

## Tasks
1. Extend requirement contract to support `resource_ref/resource_id`.
2. Apply precedence matrix:
   - resource overrides/defaults
   - provider fallback
   - auto-select
   - prompt/fail
3. Add provider fallback guard:
   - provider-selected account must be bound to required resource.
4. Emit statuses:
   - `missing`
   - `draft_incomplete`
   - `draft_invalid`
   - `needs_rebind`
   - `blocked_ambiguous_binding`
   - `blocked_missing_resource`
5. Keep API unresolved payload machine-readable and backwards compatible.

## Exit Criteria
1. Resource-first resolution works end-to-end.
2. Legacy provider-only setups still run where valid.
3. Headless failures are deterministic and actionable.

## PR4: Auth Manager UX Refactor (Resource-First)

## Goals
1. Remove provider namespace burden from user-facing flows.
2. Add profile/account actions: duplicate, rebind, archive.
3. Auto-sync drafts on open with summary message.

## File Touchpoints
1. `src/loom/tui/screens/auth_manager.py`
2. `src/loom/tui/app.py`
3. `tests/test_tui.py`
4. `README.md`
5. `docs/tutorial.html`

## Tasks
1. Replace provider input with unified resource selector.
2. Keep mode dropdown constrained to supported runtime modes.
3. Keep workspace default checkbox (resource-scoped).
4. Add on-open sync call and sync result banner.
5. Add duplicate/rebind/archive controls and flows.
6. Ensure sync failure is non-blocking with warning surface.

## Exit Criteria
1. No manual provider entry required in TUI.
2. Opening `/auth` creates missing drafts automatically.
3. UX remains stable with large resource lists.

## PR5: Resource Lifecycle (Rename/Delete/Restore)

## Goals
1. Preserve bindings across rename/re-alias.
2. Guarantee no dangling bindings/defaults on delete.
3. Add deterministic restore semantics.

## File Touchpoints
1. `src/loom/mcp/config.py`
2. `src/loom/auth/resources.py`
3. `src/loom/tui/screens/mcp_manager.py`
4. `src/loom/tui/screens/auth_manager.py`
5. `tests/test_mcp_config_manager.py`
6. `tests/test_tui.py`
7. `tests/test_auth_config.py`

## Tasks
1. Rename path updates mutable resource fields only.
2. Delete path performs atomic binding/default cleanup.
3. Shared-account safety:
   - do not archive/delete accounts with remaining active bindings.
4. Interactive delete impact dialog:
   - linked accounts
   - linked bindings
   - defaults
5. Implement tombstone restore/recreate rules.

## Exit Criteria
1. Rename never breaks linked auth.
2. Delete leaves zero dangling bindings/defaults.
3. Shared accounts survive unrelated resource deletion.

## PR6: Migration + Audit + Rollback Tooling

## Goals
1. Ship safe migration from legacy provider/mcp alias paths.
2. Provide audit visibility before/after migration.
3. Ship rollback and snapshots.

## File Touchpoints
1. `src/loom/__main__.py` (CLI commands)
2. `src/loom/auth/config.py`
3. `src/loom/auth/resources.py`
4. `tests/test_cli.py`
5. `tests/test_auth_config.py`
6. `docs/tutorial.html`
7. `README.md`

## Tasks
1. Add `auth migrate` command:
   - infer bindings
   - write snapshots
   - mark ambiguous cases
2. Add `auth audit` command:
   - orphaned accounts
   - orphaned bindings
   - deleted-resource bindings
   - provider-only legacy mappings
3. Add `auth migrate --rollback <snapshot>`.
4. Add warning surfaces for unresolved migration states.

## Exit Criteria
1. Migration is reversible.
2. Audit output is sufficient for manual cleanup.
3. Legacy users can migrate without losing run capability.

## PR7: Cleanup + Deprecation Cutover

## Goals
1. Remove obsolete scaffold and legacy-only UX text.
2. Keep only bounded compatibility paths required by policy.

## File Touchpoints
1. `src/loom/auth/config.py`
2. `src/loom/auth/runtime.py`
3. `src/loom/tui/app.py`
4. `src/loom/tui/screens/auth_manager.py`
5. `docs/creating-packages.md`
6. `planning/2026-02-25-SYSTEM-TECHNICAL-DESIGN.md`
7. `tests/*` targeted cleanup updates

## Tasks
1. Remove deprecated selector language from user-facing help.
2. Remove dead migration shims past deprecation gate.
3. Update package-author docs for resource_ref declarations.
4. Rebaseline tests around final behavior.

## Exit Criteria
1. UX and docs fully resource-first.
2. Legacy scaffolding removed per migration policy.
3. No unresolved TODOs in auth/resource lifecycle paths.

## Cross-PR Validation Gates
Run at each PR:
1. `uv run ruff check`
2. `uv run pytest tests/test_auth_config.py tests/test_tui.py tests/test_processes.py tests/test_cli.py`
3. If runtime touched: `uv run pytest tests/test_api.py tests/test_mcp_config_manager.py`

Full-suite gates:
1. Run full `uv run pytest` at PR3, PR6, and PR7 minimum.

## Final Acceptance Checklist
1. `/auth` auto-sync creates missing drafts idempotently.
2. Users never type provider namespaces in TUI flow.
3. Resource rename keeps bindings/defaults intact.
4. Resource delete cleans bindings/defaults with shared-account safety.
5. Resource-first resolution works with deterministic fallback.
6. Migration/audit/rollback commands are complete and documented.
