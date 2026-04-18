# Auth + MCP Usability Refactor Plan (2026-04-09)

## Goal
Make Loom auth and MCP setup feel simple, trustworthy, and predictable without throwing away the stronger underlying auth/resource model that Loom already has.

Target outcome:
1. A user can add a server, connect an account, test it, and understand what Loom will use at runtime without learning Loom internals first.
2. Secrets and OAuth state live in one coherent authority model.
3. Advanced flexibility remains available, but no longer dominates the default UX.

## Problem Summary
Current behavior is powerful but overly demanding.

### Primary UX problems
1. Loom exposes too many implementation concepts up front:
   - profile ids
   - target resources
   - provider derivation
   - defaults and bindings
   - secret refs and token refs
   - OAuth metadata fields
   - alias config and transport details
2. Setup and operations are mixed together:
   - create/edit config
   - login/logout
   - copy URL / enter callback
   - import token
   - test/probe
   - enable/disable/delete
3. Users must mentally translate between:
   - MCP alias
   - provider
   - auth profile
   - resource binding
   - current runtime state
4. Provenance and trust are not prominent enough:
   - where the server came from
   - whether the workspace/repo introduced it
   - whether the user has approved it
   - which stored credentials it will use

### Primary storage/model problems
1. `/auth` profile OAuth storage is relatively coherent and secure.
2. MCP OAuth storage is not:
   - alias-keyed
   - plaintext JSON file based
   - conceptually separate from Loom's stronger secret-ref model
3. Users see different auth concepts for what should feel like the same system.

## What To Preserve
This refactor should preserve Loom's strongest ideas.

1. Resource-first auth is the right long-term model.
2. MCP config should remain primarily about transport and connection.
3. Auth profiles should remain the durable home for credentials and account identity.
4. Headless/CLI paths must remain scriptable and deterministic.
5. Backward compatibility matters for existing `mcp.toml`, `/auth`, and CLI flows.

## Design Principles
1. One screen, one job.
2. Show source, trust, state, and next action before raw config.
3. Hide advanced fields until they are needed.
4. Prefer guided selection over free-form plumbing.
5. Store credentials once; refer to them everywhere else.
6. Separate "configure" from "operate".
7. Preserve a strong non-interactive contract.

## Target User-Facing Model
Users should mostly think in four concepts:

1. Server:
   - local or remote MCP endpoint
   - where it came from
   - whether it is enabled/trusted
2. Account:
   - credential identity for a provider or server
   - label like "Notion Marketing" or "Linear Personal"
3. Connection state:
   - configured
   - needs auth
   - auth in progress
   - connected
   - failed
   - disabled
   - pending approval
4. Effective runtime selection:
   - what Loom will use right now for this server/provider

Users should not need to think about `token_ref`, selector syntax, OAuth endpoint fields, or binding tables unless they open Advanced.

## Proposed Product Shape

### 1) MCP becomes the main setup entry point
Default user flow starts in `/mcp`, not `/auth`.

Primary actions:
1. Add local server
2. Add remote server
3. Connect account
4. Test connection
5. Enable/disable
6. View tools/resources/prompts
7. Inspect source and trust

Advanced actions:
1. Edit headers
2. Override auth metadata
3. Manual token import
4. Open advanced auth profile editor

### 2) `/auth` becomes account management, not setup plumbing
Default `/auth` job:
1. list accounts/profiles
2. rename/organize/archive them
3. inspect current status
4. refresh/logout
5. advanced edit only when necessary

The current "edit every field directly" experience becomes an advanced/profile-detail path rather than the starting point.

### 3) Separate config authoring from operations
For each MCP alias, Loom should present two views:

1. Configuration
   - type
   - URL or command
   - timeout
   - enabled/trust/provenance
   - minimal auth mode summary
2. Operations
   - connect/reconnect
   - authenticate
   - clear auth
   - test
   - inspect capabilities
   - view last error and remediation

### 4) Add explicit trust/provenance UX
Repo- or workspace-provided MCP servers should be visibly distinct from user-created ones.

Each server row should surface:
1. source layer
2. source path
3. trust state
4. approval requirement
5. whether the current workspace introduced it

Remote servers coming from a repo/workspace should require explicit approval before activation.

## Target Interaction Flows

### A) Add local stdio server
1. User chooses `Add MCP Server`.
2. User selects `Local`.
3. Loom asks for:
   - alias
   - command
   - args
   - cwd (optional)
   - env refs (optional)
4. Loom saves config.
5. Loom offers `Test now`.
6. Loom shows status and discovered tools.

No auth concepts should appear unless the server explicitly needs them.

### B) Add remote OAuth server
1. User chooses `Add MCP Server`.
2. User selects `Remote`.
3. Loom asks for:
   - alias
   - URL
   - scopes if needed
4. Loom validates transport and trust policy.
5. Loom saves config.
6. Loom offers `Connect account`.
7. Loom launches browser flow or gives manual fallback.
8. Loom stores credentials securely.
9. Loom reconnects and shows status/tools.

Default screen should not expose raw access token and refresh token fields.

### C) Use an existing account with a server
1. User opens a server.
2. Loom shows current auth state:
   - no account
   - one account linked
   - multiple candidate accounts
3. User chooses:
   - use existing account
   - create new account
   - disconnect
4. Loom shows the effective account label on the server detail screen.

### D) Advanced provider/profile editing
Advanced edit remains available for:
1. custom OAuth endpoints
2. CLI passthrough auth
3. env passthrough auth
4. unusual metadata or secret-ref cases

This should move behind an explicit Advanced affordance.

## Authority And Storage Model

### 1) Split transport authority from credential authority
MCP config owns:
1. alias
2. transport type
3. URL or command
4. timeout
5. safe connection flags
6. enabled state
7. source and trust metadata

Auth/profile state owns:
1. account label
2. provider/server credential identity
3. secret refs
4. OAuth token payloads
5. refresh/logout lifecycle

### 2) Unify secret storage
Adopt Loom's `/auth` secret-ref approach as the shared credential authority for MCP OAuth too.

Requirements:
1. MCP OAuth tokens move off alias-keyed plaintext JSON storage.
2. Stored token payloads use writable secret refs.
3. `keychain://...` remains the primary secure storage target.
4. CLI/TUI only display redacted summaries and state.

### 3) Fingerprint credentials by server identity
MCP OAuth credential bindings must key by server fingerprint, not alias alone.

Fingerprint should include at minimum:
1. server type
2. canonical URL
3. relevant auth-affecting headers
4. relevant OAuth identity config

This prevents silent credential reuse when an alias changes meaning.

### 4) Preserve compatibility with migration
Backward-compatible behavior:
1. existing `mcp_oauth_tokens.json` remains readable during migration
2. migration copies data into the new secure store
3. Loom records migration status and continues working
4. legacy store can be cleaned up after successful migration

## Command Surface Changes

### Keep
1. `loom mcp list/show/status/connect/test`
2. `loom mcp auth login/status/logout/refresh`
3. `loom auth list/show/check/sync/select`

### Refine
1. `loom mcp add`
   - guide toward local vs remote first
2. `loom mcp show`
   - emphasize source, trust, auth state, and effective account
3. `loom mcp status`
   - include actionable remediation and approval status
4. `loom auth show`
   - emphasize account/server/provider relationships instead of raw metadata first

### Add
1. `loom mcp approve <alias>`
2. `loom mcp reject <alias>`
3. `loom mcp explain <alias>`
   - why this server is enabled/disabled/pending/failed
   - what credentials it will use
4. `loom auth explain <profile-id>`
   - where the profile is used
   - what resources/servers it applies to

## TUI Refactor Plan

### MCP list view
Rework list rows to show:
1. alias
2. type
3. status
4. source
5. trust
6. effective account label

Group by source:
1. workspace/repo
2. user
3. explicit overlay
4. legacy

### MCP detail view
Replace the current single giant form with tabs or sections:
1. Summary
2. Connection
3. Auth
4. Advanced

Summary:
1. state
2. source path
3. trust
4. effective account
5. tools/resources/prompts counts
6. primary next action

Connection:
1. transport config only

Auth:
1. connect account
2. switch account
3. login/logout/refresh
4. manual fallback actions

Advanced:
1. headers
2. manual token import
3. debug details

### Auth manager
Reduce the default form to:
1. account label
2. target provider/server
3. mode
4. status
5. primary actions

Move into Advanced:
1. secret refs
2. token refs
3. auth check args
4. metadata KV pairs
5. raw OAuth endpoint fields

### Cross-links
Add obvious navigation:
1. from MCP detail -> open linked auth profile
2. from auth profile -> show linked MCP aliases/resources
3. from failures -> jump to remediation action

## Desktop Frontend Review
Loom's desktop frontend is already capable of hosting MCP/auth management, but the necessary product surface does not exist yet.

### Current desktop app state
1. The desktop app has a stable shell, central app state, and tab model in:
   - `apps/desktop/src/components/AppShell.tsx`
   - `apps/desktop/src/components/Sidebar.tsx`
   - `apps/desktop/src/hooks/useAppState.ts`
2. The current top-level tab model is limited to:
   - overview
   - threads
   - runs
   - files
   - settings
3. The desktop app already fetches workspace inventory through:
   - `apps/desktop/src/api.ts`
   - `apps/desktop/src/hooks/useWorkspace.ts`
4. That inventory already includes:
   - processes
   - MCP servers
   - tools
5. However, MCP/auth are only passive inventory data today:
   - no dedicated management screen
   - no mutation APIs for create/edit/connect/disconnect/test/approve
   - no detailed auth/account model exposed to the frontend
6. The current `WorkspaceInventory`/`MCPServerInfo` payload is too thin for management UX:
   - no trust/approval state
   - no source path
   - no connection/runtime state
   - no effective account label
   - no remediation details
   - no capabilities counts
7. The current `SettingsPanel` is runtime/app-settings oriented, not integration-management oriented.

### What this means
The desktop app should not be treated as a thin wrapper around the TUI managers.

Instead:
1. The desktop app should get its own first-class MCP/auth management UX.
2. The backend must expose management-oriented API resources rather than expecting the frontend to reconstruct state from CLI-shaped data.
3. The desktop app should become the best graphical entry point for setup, trust review, and account switching.

## Desktop Frontend Product Shape

### Recommended information architecture
Add a new top-level desktop tab:
1. `integrations`

Reasoning:
1. MCP/auth management is too large and operationally important to hide inside generic settings.
2. It is workspace-aware and should sit alongside runs/files rather than inside global appearance/runtime settings.
3. It cleanly separates:
   - app settings
   - workspace operations
   - integrations and auth

`settings` should remain for:
1. appearance
2. runtime details
3. global configuration summaries

`integrations` should own:
1. MCP servers
2. auth/accounts
3. trust/approval state
4. effective account routing summary

### Integrations tab structure
Recommended desktop layout:

1. Left rail or segmented header:
   - MCP Servers
   - Accounts
   - Activity / Issues
2. Main detail pane:
   - summary-first detail view
   - primary actions
   - advanced details hidden by default

### MCP Servers view
Primary list columns/cards:
1. alias
2. type
3. status
4. source
5. trust
6. effective account

Primary detail sections:
1. Summary
2. Connection
3. Auth
4. Advanced

Primary actions:
1. Add local server
2. Add remote server
3. Connect account
4. Reconnect
5. Test
6. Enable/disable
7. Approve/reject

### Accounts view
Primary list columns/cards:
1. account label
2. provider
3. linked server/resource count
4. status
5. source/scope

Primary detail sections:
1. Summary
2. Linked servers/resources
3. Status
4. Advanced

Primary actions:
1. Create account
2. Connect/login
3. Refresh
4. Logout/disconnect
5. Archive
6. Open linked server

### Activity / Issues view
This is optional in the first version, but valuable.

Show:
1. auth failures
2. pending approvals
3. disconnected servers
4. expired tokens
5. remediation actions

This can become the fastest operational path for fixing integration drift.

## Desktop Frontend API Contract Additions
The frontend will need dedicated management APIs rather than overloading the current workspace inventory response.

### New MCP API resources
1. `GET /workspaces/{workspace_id}/mcp`
   - list management-grade MCP rows
2. `GET /workspaces/{workspace_id}/mcp/{alias}`
   - one server detail
3. `POST /workspaces/{workspace_id}/mcp`
   - create server
4. `PATCH /workspaces/{workspace_id}/mcp/{alias}`
   - edit server
5. `DELETE /workspaces/{workspace_id}/mcp/{alias}`
   - delete server
6. `POST /workspaces/{workspace_id}/mcp/{alias}/test`
7. `POST /workspaces/{workspace_id}/mcp/{alias}/connect`
8. `POST /workspaces/{workspace_id}/mcp/{alias}/disconnect`
9. `POST /workspaces/{workspace_id}/mcp/{alias}/approve`
10. `POST /workspaces/{workspace_id}/mcp/{alias}/reject`

### New auth/account API resources
1. `GET /workspaces/{workspace_id}/auth/accounts`
2. `GET /workspaces/{workspace_id}/auth/accounts/{profile_id}`
3. `POST /workspaces/{workspace_id}/auth/accounts`
4. `PATCH /workspaces/{workspace_id}/auth/accounts/{profile_id}`
5. `POST /workspaces/{workspace_id}/auth/accounts/{profile_id}/login`
6. `POST /workspaces/{workspace_id}/auth/accounts/{profile_id}/refresh`
7. `POST /workspaces/{workspace_id}/auth/accounts/{profile_id}/logout`
8. `POST /workspaces/{workspace_id}/auth/accounts/{profile_id}/archive`

### New management payload requirements
MCP list/detail payloads should include:
1. alias
2. type
3. source
4. source_path
5. enabled
6. trust_state
7. approval_required
8. runtime_status
9. auth_state
10. oauth_enabled
11. effective_account_label
12. effective_account_profile_id
13. command/url summary
14. timeout
15. last_error
16. remediation
17. tools_count
18. prompts_count
19. resources_count

Account list/detail payloads should include:
1. profile_id
2. account_label
3. provider
4. mode
5. status
6. linked_mcp_aliases
7. linked_resources
8. token_state
9. expires_at
10. source/scope
11. writable_storage_kind

### Frontend OAuth flow contract
The desktop app should not parse CLI-style console output for auth flows.

Preferred API shape:
1. start login endpoint returns:
   - authorization URL
   - callback mode
   - browser-opened flag
   - expiry
2. polling or SSE endpoint returns:
   - pending
   - completed
   - failed
   - canceled
3. callback submit endpoint accepts manual callback URL/code

This lets the frontend render a proper task-oriented auth experience.

## Desktop Frontend Implementation Plan

### Phase A: Navigation and state scaffolding
1. Add `integrations` to `ViewTab`.
2. Add sidebar entry and command palette action for Integrations.
3. Introduce desktop state slices for:
   - MCP management rows
   - selected MCP alias
   - auth/account rows
   - selected account profile
   - auth flow pending state
   - integration notices/errors

Likely files:
1. `apps/desktop/src/utils.ts`
2. `apps/desktop/src/components/AppShell.tsx`
3. `apps/desktop/src/components/Sidebar.tsx`
4. `apps/desktop/src/hooks/useAppState.ts`
5. `apps/desktop/src/shell.ts`

Exit criteria:
1. The app has a dedicated integrations destination and state model.

### Phase B: Read-only management surfaces
1. Add `IntegrationsTab.tsx`.
2. Render MCP server list from management payloads.
3. Render account list from management payloads.
4. Show summary/detail views with source/trust/state first.
5. Keep actions disabled or minimal until mutation APIs land.

Likely files:
1. `apps/desktop/src/components/IntegrationsTab.tsx` (new)
2. `apps/desktop/src/components/AppShell.tsx`
3. `apps/desktop/src/api.ts`
4. `apps/desktop/src/hooks/useWorkspace.ts` or new `useIntegrations.ts`
5. desktop tests

Exit criteria:
1. Users can inspect integrations in the desktop app without opening the TUI.

### Phase C: MCP management actions
1. Add create/edit/test/enable/disable/delete actions.
2. Add approval/rejection actions.
3. Add source/trust messaging and remediation UI.
4. Add reconnect/status refresh behavior.

Likely files:
1. `apps/desktop/src/api.ts`
2. `apps/desktop/src/components/IntegrationsTab.tsx`
3. `apps/desktop/src/hooks/useIntegrations.ts` (new)
4. backend API routes and schemas
5. desktop tests

Exit criteria:
1. MCP lifecycle is manageable from desktop.

### Phase D: Auth/account management actions
1. Add account create/edit/connect/login/refresh/logout/archive actions.
2. Add existing-account vs create-new-account decision flow for servers.
3. Render login task state:
   - opening browser
   - waiting for callback
   - paste code fallback
   - completed
   - failed
4. Show effective account and linked-server relationships.

Likely files:
1. `apps/desktop/src/api.ts`
2. `apps/desktop/src/components/IntegrationsTab.tsx`
3. `apps/desktop/src/hooks/useIntegrations.ts`
4. backend API routes and schemas
5. desktop tests

Exit criteria:
1. Auth/account lifecycle is manageable from desktop.

### Phase E: Polish and operations
1. Add issues queue for broken integrations.
2. Add optimistic refresh and inline remediation.
3. Add command palette shortcuts:
   - add server
   - connect account
   - show broken integrations
4. Add richer search coverage for MCP/auth items.

Exit criteria:
1. Desktop becomes the preferred graphical path for integration setup and repair.

## Implementation Architecture

### Phase 0: Vocabulary and UX contract
1. Standardize terms:
   - server
   - account
   - profile
   - source
   - trust
   - effective account
2. Update copy in CLI/TUI to stop leading with internal jargon.

Likely files:
1. `src/loom/tui/screens/auth_manager.py`
2. `src/loom/tui/screens/mcp_manager.py`
3. `src/loom/cli/commands/mcp.py`
4. `src/loom/cli/commands/auth.py`

Exit criteria:
1. User-facing strings consistently describe the same concepts.

### Phase 1: Credential authority unification
1. Introduce a shared MCP credential store abstraction backed by secret refs.
2. Add server-fingerprint keying for MCP OAuth credentials.
3. Add compatibility reader for legacy alias-keyed JSON token storage.
4. Migrate write paths to the new storage model.

Likely files:
1. `src/loom/integrations/mcp/oauth.py`
2. `src/loom/auth/secrets.py`
3. `src/loom/auth/runtime.py`
4. `src/loom/cli/commands/mcp_auth.py`
5. `tests/test_mcp_oauth.py`
6. `tests/test_auth_runtime.py`

Exit criteria:
1. New writes never depend on alias-only plaintext token storage.
2. Legacy tokens still work until migrated.

### Phase 2: Trust and provenance model for MCP
1. Add explicit approval state for workspace/repo-provided MCP servers.
2. Add allow/deny policy hooks for:
   - remote URL patterns
   - local command patterns
3. Surface approval/trust state in CLI/TUI status outputs.

Likely files:
1. `src/loom/mcp/config.py`
2. `src/loom/config.py`
3. `src/loom/cli/commands/mcp.py`
4. `src/loom/tui/screens/mcp_manager.py`
5. `src/loom/api/routes.py`
6. `src/loom/api/schemas.py`
7. `tests/test_mcp_config.py`
8. `tests/test_cli.py`
9. desktop API tests

Exit criteria:
1. Repo/workspace remote servers no longer silently become active.
2. Users can tell where a server came from and whether it is trusted.

### Phase 3: MCP screen split into summary/config/auth/advanced
1. Rework `MCPManagerScreen` around list + detail rather than giant edit form.
2. Make Summary the default tab/section.
3. Move manual token entry and raw headers into Advanced.
4. Make auth actions task-oriented and stateful.

Likely files:
1. `src/loom/tui/screens/mcp_manager.py`
2. `src/loom/tui/app.py`
3. `src/loom/api/routes.py`
4. `src/loom/api/schemas.py`
5. `apps/desktop/src/components/IntegrationsTab.tsx`
6. `apps/desktop/src/api.ts`
7. `apps/desktop/src/hooks/useIntegrations.ts`
8. `tests/test_tui.py`
9. desktop tests

Exit criteria:
1. A new user can configure and connect a remote server without touching raw token fields.

### Phase 4: Auth screen simplification
1. Convert `/auth` into account/profile management first.
2. Collapse advanced fields by default.
3. Surface status, linked resources, and linked MCP aliases prominently.
4. Keep a path to full expert editing.

Likely files:
1. `src/loom/tui/screens/auth_manager.py`
2. `src/loom/cli/commands/auth.py`
3. `src/loom/auth/resources.py`
4. `src/loom/api/routes.py`
5. `src/loom/api/schemas.py`
6. `apps/desktop/src/components/IntegrationsTab.tsx`
7. `apps/desktop/src/api.ts`
8. `apps/desktop/src/hooks/useIntegrations.ts`
9. `tests/test_tui.py`
10. `tests/test_auth_config.py`
11. desktop tests

Exit criteria:
1. Default `/auth` flow feels like "manage accounts", not "author config internals".

### Phase 5: Guided add/connect flows
1. Add wizard-like entry paths for:
   - add local server
   - add remote server
   - connect existing account
   - create new account for server
2. Ensure each flow ends with test/status confirmation.
3. Add `explain`/remediation helpers to CLI.

Likely files:
1. `src/loom/tui/screens/mcp_manager.py`
2. `src/loom/tui/screens/auth_manager.py`
3. `src/loom/cli/commands/mcp.py`
4. `src/loom/cli/commands/mcp_auth.py`
5. `src/loom/api/routes.py`
6. `src/loom/api/schemas.py`
7. `apps/desktop/src/components/IntegrationsTab.tsx`
8. `apps/desktop/src/api.ts`
9. `apps/desktop/src/hooks/useIntegrations.ts`
10. `apps/desktop/src/components/AppShell.tsx`
11. `apps/desktop/src/components/Sidebar.tsx`
12. `tests/test_cli.py`
13. `tests/test_tui.py`
14. desktop tests

Exit criteria:
1. Core setup tasks complete through guided flows.

### Phase 6: Cleanup, docs, and de-emphasis of legacy paths
1. Update help text and planning/docs.
2. Document migration from:
   - raw MCP env credentials
   - manual token JSON store expectations
   - advanced auth field editing as default behavior
3. Keep compatibility paths but mark them as advanced.

Likely files:
1. `docs/`
2. `README` and command help text
3. `planning/`
4. `tests/`

Exit criteria:
1. Docs match the new mental model.

## Recommended First Slice
The first slice should be intentionally narrow:

1. Unify MCP OAuth storage with Loom's secret-ref/keychain model.
2. Fingerprint server credentials instead of alias-only storage.
3. Rework MCP list/detail to foreground:
   - source
   - trust
   - status
   - effective account
4. Hide manual token entry behind Advanced.
5. Add desktop read-only integrations tab backed by management-grade MCP/auth API payloads.

Why this first:
1. It fixes the sharpest security/coherence problem.
2. It improves UX without requiring a full auth-system redesign first.
3. It gives the desktop app a real landing zone for MCP/auth management early.
4. It creates a clean foundation for the guided setup flows.

## Testing Strategy
1. Compatibility tests for legacy alias-keyed MCP OAuth store migration.
2. CLI tests for status/explain/approval flows.
3. TUI tests for list grouping, default sections, and state-first rendering.
4. Runtime tests for:
   - fingerprinted credential lookup
   - migration fallback
   - approval gating
   - effective account resolution
5. Regression tests for manual/headless OAuth fallback.
6. Desktop tests for:
   - integrations navigation
   - read-only MCP/account rendering
   - auth flow state transitions
   - action gating and remediation rendering

## Risks
1. Blurring MCP-owned auth and `/auth`-owned auth can create duplicate authority if the refactor is partial.
2. A UI-only cleanup without storage unification will improve feel but not coherence.
3. Over-rotating toward wizards could make expert flows slower if advanced paths are not preserved.
4. Fingerprint migration must not strand existing tokens.

## Explicit Non-Goals
1. Replacing the resource-first auth model with a simpler but weaker global-token system.
2. Removing scriptable CLI auth paths.
3. Removing advanced auth modes such as env or CLI passthrough.
4. Rewriting all MCP runtime behavior in the same change as the UX cleanup.

## Success Criteria
1. A first-time remote MCP setup can be completed from TUI without editing raw OAuth/token fields.
2. Users can answer:
   - where did this server come from?
   - do I trust it?
   - what account will Loom use?
   - what should I do next?
3. MCP OAuth storage is secure by default and coherent with the rest of Loom auth.
4. Existing configs continue to work during migration.
