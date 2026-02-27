# Auth Resource Lifecycle + Auto-Discovery Refactor Plan (2026-02-27)

## Objective
Deliver a high-quality auth UX where users do not need to know provider namespaces, while preserving system flexibility and deterministic behavior.

The refactor must:
1. Make auth resource-first in UX (tool/MCP/API integration selection).
2. Auto-create draft auth profiles from discovered requirements automatically on `/auth` open.
3. Preserve auth linkages on rename/re-alias.
4. Clean up auth profiles/bindings when resources are deleted.
5. Keep backward compatibility during migration.

## Product Principles
1. Users select resources, not internal namespaces.
2. Resource identity must be stable across rename/re-alias.
3. No dangling auth bindings after resource deletion.
4. Auto-discovery should not require live authenticated introspection.
5. Interactive UX and headless behavior must both be deterministic.

## Terminology Contract
1. UX term: `Profile` (what users create/select in TUI/CLI).
2. Internal schema term: `Account` (credential payload record).
3. Internal linkage term: `Binding` (`resource_id` -> `account_id` association).
4. In docs and code comments:
   - user-facing copy should say "profile"
   - schema/runtime internals should say "account" and "binding"

## Current Gaps
1. Runtime matching is provider-first, so users still configure `provider` directly.
2. MCP target binding exists, but only as alias string, not stable resource ID.
3. No first-class lifecycle rules for rename/delete across resources and auth profiles.
4. No draft-profile auto-generation for newly discovered auth requirements.
5. Some fields are still surfaced as low-level implementation details.

## Target End State
1. Auth Manager uses a unified resource selector:
   - MCP aliases
   - tools with declared auth requirements
   - registered API integrations
2. Profile creation starts from a discovered resource contract.
3. `provider` is hidden in TUI and derived internally from selected resource contract.
4. Resource links are stored by immutable `resource_id` and survive rename.
5. Resource delete triggers enforced cleanup flow with no dangling links.
6. Draft profiles are generated automatically and marked "needs completion".

## Data Model Refactor

## 1) Auth Resource Registry
Add an internal registry model:
1. `resource_id` (UUID, immutable)
2. `resource_kind` (`mcp`, `tool`, `api_integration`)
3. `resource_key` (human key, mutable; alias/name)
4. `display_name`
5. `provider` (internal namespace)
6. `auth_contract`:
   - supported modes
   - optional scopes
   - required env keys
   - source (`mcp` or `api`)
7. `status` (`active`, `deleted`)
8. lifecycle metadata (`created_at`, `updated_at`, `deleted_at`)

## 2) Auth Account Model
Separate reusable credential accounts from resource bindings.

`auth_account` model:
1. `account_id` (stable id, replaces profile-as-binding coupling)
2. `provider` (internal namespace only; hidden in TUI)
3. `mode`
4. `secret_ref`
5. `token_ref`
6. `env`
7. `scopes`
8. `metadata`
9. `status` (`draft`, `ready`, `archived`)
10. lifecycle metadata (`created_at`, `updated_at`, `archived_at`)

Draft semantics:
1. `draft` means auto-generated and not runnable yet.
2. `ready` means mode-specific validation passes.
3. `archived` means excluded from selection unless explicitly restored.

## 3) Auth Binding Model
Introduce explicit resource-to-account links:
1. `binding_id`
2. `resource_id`
3. `account_id`
4. `is_default_workspace` (boolean)
5. `priority` (tie-break among multiple candidates)
6. `generated_from` metadata (`tool:<name>`, `mcp:<alias>`, etc.)
7. lifecycle metadata (`created_at`, `updated_at`, `deleted_at`)

Compatibility during migration:
1. `mcp_server` field remains readable on legacy profiles.
2. provider-based defaults remain readable.
3. new writes prefer account + binding records.

## 4) Defaults Model
Move defaults to resource-first mapping:
1. workspace defaults: `resource_id -> account_id`
2. user defaults: `resource_id -> account_id`
3. keep legacy `provider -> account_id` defaults as compatibility fallback
4. write both resource-default scopes during migration window, read resource-first

Authoritative precedence:
1. explicit run override by `resource_id`
2. workspace resource default
3. user resource default
4. explicit run override by provider (legacy)
5. workspace provider default (legacy)
6. user provider default (legacy)
7. single-candidate auto-select
8. interactive prompt (interactive only)
9. unresolved error (headless/non-interactive)

Guard rule for legacy provider fallback:
1. For resource-scoped requirements, provider fallback is valid only if the
   selected account has an active binding to the required `resource_id`.
2. If provider fallback selects an unbound account, treat as unresolved and
   surface `needs_rebind`.

## 5) Storage Contract
1. User-scoped account store:
   - `~/.loom/auth.toml` (extends existing auth profile/account records)
   - unique key: `account_id`
2. Workspace-scoped resource state store:
   - `.loom/auth.resources.toml`
   - contains resource registry, bindings, workspace resource defaults,
     tombstones, and pending migration/sync markers
3. Legacy workspace defaults compatibility store:
   - `.loom/auth.defaults.toml` (provider defaults)
   - read/write during migration window, read-only after cutover

Uniqueness constraints:
1. `resource_id` unique in registry
2. `(resource_id, account_id)` unique in bindings
3. at most one active workspace default per `resource_id`
4. at most one active user default per `resource_id`

Atomicity boundaries:
1. Resource rename/delete/default changes are single-file atomic in
   `.loom/auth.resources.toml`.
2. Account edits are single-file atomic in `auth.toml`.
3. Cross-file operations (draft sync, migration) use idempotent two-step saga:
   - step A: write account draft/update
   - step B: write binding/default
   - compensation: if step B fails, keep account as `draft` and record pending
     operation marker; next sync retries and surfaces warning.

## Resource Lifecycle Rules (authoritative)

## 1) Rename / Re-alias
1. Renaming a resource updates only mutable display/key fields.
2. `resource_id` stays unchanged.
3. All account/default bindings remain intact automatically.
4. No user action required.

## 2) Delete Resource
1. Deletion must never leave dangling auth links.
2. On delete, system enumerates bound accounts/bindings/defaults.
3. Cleanup policy:
   - auto-delete drafts generated for that resource that were never completed
   - archive completed linked accounts by default only when they have zero
     remaining active bindings after delete
   - clear defaults bound to deleted resource
4. Interactive delete flow offers choices:
   - rebind bindings to another resource
   - archive accounts with zero remaining bindings
   - delete drafts with zero remaining bindings
5. Non-interactive delete requires explicit `--cascade` behavior selection.
6. Resource delete must remove or tombstone all related bindings atomically.

## 3) Restore / Recreate
1. Deleted resources remain tombstoned for audit window.
2. Recreating a resource with same alias gets new `resource_id` unless restored.
3. Restore operation can reattach archived accounts/bindings.
4. Restore is deterministic:
   - if original `resource_id` tombstone exists, restore it
   - else create new `resource_id` and require manual rebind/import

## 4) Archive Contract
1. Archived accounts are stored in normal auth config with `status=archived`.
2. Archived accounts are hidden from default selectors and run auto-resolution.
3. Archive retention defaults to indefinite until explicit purge.
4. Purge is explicit and irreversible.

## Auto-Discovery and Draft Generation

## 1) Tool-Declared Contracts
1. Keep tool-level `auth_requirements` as source of truth.
2. Build resource registry entries from these declarations.
3. Generate draft profiles/accounts for missing coverage.
4. Require declared requirement identity to include `resource_ref` when possible.

## 2) MCP Contract Strategy
Do not rely on authenticated live introspection for auth contract discovery.

Use setup-time contract sources:
1. MCP package manifest metadata when available.
2. MCP setup wizard prompts for provider/mode template.
3. Optional post-auth verify to refine scopes/checks.

## 3) Draft Profile Generation Rules
For each discovered resource lacking a usable profile:
1. create `draft` account with deterministic id pattern
2. prefill mode from contract if unambiguous
3. prefill placeholder refs:
   - `secret_ref` placeholder for `api_key`
   - `token_ref` placeholder for OAuth modes
4. create binding to `resource_id`
5. derive provider internally from resource contract
6. show status as "Needs setup"

Draft idempotency key:
1. (`resource_id`, `mode`, `generated_slot`) must be unique.
2. repeated sync updates existing draft metadata instead of creating duplicates.

## 4) Automatic Draft Sync Trigger
1. Running `/auth` (opening Auth Manager) executes `sync_missing_drafts()`.
2. Sync compares active discovered resources against existing non-archived accounts/bindings.
3. Missing coverage is auto-created as drafts before the screen is rendered.
4. Sync is idempotent (no duplicate drafts on repeated `/auth` opens).
5. Sync is additive-only:
   - create missing drafts
   - optionally refresh generated draft metadata
   - never delete accounts/bindings/defaults
6. Deletion/archival remains explicit in resource lifecycle flows, not in open-time sync.
7. Sync safety contract:
   - if write fails, auth manager still opens
   - show non-blocking warning with actionable remediation
   - keep in-memory pending draft recommendations for user visibility
8. Sync must be a no-op when there is no content diff.

## 5) Discovery Scope Rules
Default scope for open-time sync:
1. resources referenced by current workspace processes
2. installed/enabled MCP aliases
3. tools explicitly included/allowed by active process config

Optional full-scan command:
1. `auth sync --scope full` for pre-provisioning all discovered resources.

## UX Design Changes

## 1) Auth Manager
1. Replace raw provider input with resource selector.
2. Keep mode as dropdown constrained to supported modes.
3. Keep "Set as workspace default" checkbox.
4. Add actions:
   - `Duplicate Profile`
   - `Rebind Profile`
   - `Archive Profile`
5. On open, show sync summary message (for example, `Created 3 draft profiles (incomplete)`).

## 2) Run Start Flow
1. Preflight resolves required resources.
2. If exactly one `ready` account matches resource, auto-select.
3. If multiple, prompt user with profile labels.
4. If only `draft` exists, route user to complete draft before run.
5. Persist optional workspace default by resource.

Run-time status taxonomy:
1. `missing`: no account/binding candidate exists
2. `draft_incomplete`: draft exists but required fields missing
3. `draft_invalid`: draft exists but fails mode validation
4. `ready`: valid and selectable

## 3) Deletion and Rename UX
1. Resource delete dialog shows impact counts:
   - linked accounts
   - linked bindings
   - defaults
   - active processes referencing resource
2. Rename dialog confirms zero auth breakage because bindings are ID-based.

## Compatibility and Migration

## 1) Read Path
1. If `resource_id` binding exists, use it first.
2. Else fall back to legacy provider/mcp alias mapping.
3. Emit migration hint when legacy-only profile/account selection is used.

## 2) Write Path
1. New/edited TUI entries write `auth_account` + `auth_binding`.
2. During migration window also write legacy fields for compatibility.
3. Workspace defaults write resource-first mapping; mirror provider default when derivable.

## 3) Migration Tooling
1. Add one-shot migration command:
   - infer `resource_id` from `mcp_server` and provider contracts
   - flag ambiguous profiles/accounts for manual review
2. Add audit command:
   - list orphaned accounts
   - list orphaned bindings
   - list deleted-resource bindings
   - list provider-only legacy profiles/accounts

Ambiguous migration runtime states:
1. `needs_rebind`: account selected by provider fallback but not bound to required resource.
2. `blocked_ambiguous_binding`: multiple candidate bindings after migration inference.
3. `blocked_missing_resource`: legacy binding references non-existent resource.
4. Run preflight must fail fast for `blocked_*` states with remediation commands.

## 4) Schema Versioning and Rollback
1. Add explicit auth schema version marker.
2. Migration writes backup snapshots before mutation.
3. Failed migration auto-restores last known good snapshot.
4. Provide `auth migrate --rollback <snapshot>` command.

## Cleanup of Old Scaffold
1. Remove provider-first wording from TUI docs/help once resource-first UX is live.
2. Keep parser support for deprecated fields temporarily:
   - `auth.defaults` provider selectors
   - profile `mcp_server` alias-only links
3. Remove deprecated `mcp.*` selector path entirely after migration window.

## Implementation Workstreams

## W1: Resource Registry Foundation
1. Add registry schema, storage, and resolver APIs.
2. Ensure immutable IDs and tombstone lifecycle support.
3. Add cross-process file locking and atomic writes for registry/auth/default files.

## W2: Auto-Discovery + Draft Engine
1. Build scanner for tool/process declared auth requirements.
2. Build MCP setup-time contract capture.
3. Implement draft generation and idempotent re-run behavior.
4. Add Auth Manager open hook to run `sync_missing_drafts()` before render.
5. Add file-write guard so sync writes only when content changes.
6. Add cross-process lock to prevent concurrent sync races.
7. Add partial-failure handling (open screen even when sync write fails).

## W3: Runtime Resolver Migration
1. Resolve required auth by `resource_ref/resource_id` first.
2. Preserve provider fallback.
3. Keep unresolved-auth error payloads machine readable.
4. Include status taxonomy in unresolved payload (`missing`, `draft_incomplete`, etc.).

## W4: TUI/CLI UX Refactor
1. Auth manager resource selector + draft workflows.
2. Delete/rename lifecycle dialogs with cascade options.
3. Add duplicate/rebind/archive actions.

## W5: Migration + Audit + Cleanup
1. migration command
2. audit command
3. deprecation warnings and final removal gates
4. schema version and rollback support

## PR Sequence
1. PR1: Resource registry model + storage + tests
2. PR2: Discovery scanner + draft generator + tests
3. PR3: Runtime resource-first resolution + compatibility fallback
4. PR4: Auth manager resource-first UX + defaulting + duplication
5. PR5: Resource lifecycle delete/rename cleanup flows
6. PR6: Migration/audit commands + docs updates
7. PR7: Deprecated scaffold removal behind completed migration gate

## Test Strategy

## Unit Tests
1. registry create/rename/delete/tombstone invariants
2. account/binding model invariants by `resource_id`
3. draft generation idempotency
4. delete cascade rules with shared-account protection
5. rename preserves all bindings/defaults
6. precedence matrix (resource defaults vs provider fallback) determinism
7. sync safety behavior when write fails (screen still opens)

## Integration Tests
1. Add tool with auth requirement creates draft account + binding.
2. Add MCP with auth template creates draft account + binding.
3. Opening `/auth` auto-creates missing drafts and reports count.
4. Reopening `/auth` is idempotent (no duplicate drafts).
5. Complete draft and run succeeds without manual provider entry.
6. Delete resource cleans linked defaults/bindings and preserves accounts still bound elsewhere.
7. Rename resource keeps run auth resolution intact.
8. full-scan sync generates additional drafts beyond active-scope sync.

## Regression Tests
1. existing provider-based profiles/accounts continue to resolve
2. existing MCP alias profile bindings continue to work
3. unresolved-auth errors remain stable for API clients

## Acceptance Criteria
1. No TUI flow requires users to type provider namespaces.
2. Opening `/auth` auto-produces missing draft profiles for discovered resources.
3. Renaming resources never breaks profile/default linkages.
4. Deleting resources leaves zero dangling bindings.
5. Headless and interactive resolution behavior remains deterministic.
6. Legacy configs still function during migration window.
7. One account can be bound to multiple resources without credential duplication.
8. `/auth` open never hard-fails due to draft-sync write errors.
9. Provider fallback never selects an account unbound to required resource.

## Risks and Mitigations
1. Risk: Ambiguous migration from legacy profiles.
   Mitigation: explicit audit output + manual review queue.
2. Risk: Over-aggressive delete cleanup removes wanted profiles.
   Mitigation: archive completed profiles by default, delete only drafts automatically.
3. Risk: Contract drift between tool declaration and runtime expectations.
   Mitigation: declaration lint + contract tests for first-party and package tools.
4. Risk: Hidden conflict between resource and provider defaults during migration.
   Mitigation: explicit precedence matrix, conflict warning, and audit output.

## Out of Scope (this refactor)
1. Full OAuth token broker/refresh service redesign.
2. External secret manager service integration beyond current secret refs.
3. Full replacement of all legacy provider fields in one release.
