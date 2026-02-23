# Auth Routing UX Refinement Plan (2026-02-20)

## Goal
Make multi-account access for one MCP server intuitive and low-risk by separating concerns in UI and commands:
1. MCP server config: how to connect
2. Auth profile config: which credentials exist
3. Auth routing: which profile is used by default for provider/alias
4. Run overrides: one-run temporary selection

This plan addresses current user confusion where MCP and auth surfaces feel overlapping and selector semantics are too implicit.

## Problem Summary
Current behavior is functionally correct but cognitively expensive:
1. Auth and MCP managers share too much visual structure and mixed concepts.
2. Routing concepts (`defaults`, `mcp_alias_profiles`, `mcp.<alias>` selectors) are technically flexible but not obvious in UI.
3. Per-run multi-account selection exists as textual selector overrides, not guided interaction.
4. Users can mistake placeholder examples for active values.

Net result: users cannot quickly answer “which account will this MCP server use right now?”

## UX Principles
1. One screen, one job.
2. Show “effective account for this run” explicitly.
3. Prefer guided selection over free-form selector text.
4. Keep non-interactive CLI/CI path deterministic and scriptable.
5. Preserve backward compatibility while reducing conceptual surface area.

## Proposed Interaction Model

### 1) MCP Server Manager (`/mcp`)
Scope:
1. Alias
2. Command/args/cwd/timeout/enabled/env
3. Probe/test

Out of scope:
1. Profile defaults
2. Provider selector editing
3. Alias-to-profile routing edits

### 2) Auth Profile Manager (`/auth`)
Scope:
1. Profile id, provider, mode, label
2. Secret/token refs
3. Scopes/env/metadata/auth check
4. CRUD for profiles

Out of scope:
1. Workspace default selector fields
2. MCP alias binding fields

### 3) New Auth Routing Manager (`/auth routes`)
Scope:
1. Provider defaults (for example `notion -> notion_marketing`)
2. MCP alias defaults (for example `mcp.notion -> notion_sales`)
3. Workspace vs user scope visibility
4. CRUD for routes with dropdown-like selection semantics

Primary display columns:
1. Scope (`provider` or `mcp_alias`)
2. Key (`notion` or `notion`)
3. Profile id
4. Source (`workspace` / `user` / `explicit`)
5. Effective precedence indicator

### 4) Run Auth Picker (preflight in TUI)
When multiple valid profiles exist and no explicit route/override resolves uniquely:
1. Show modal picker for unresolved provider/alias requirements.
2. Let user select profile per unresolved requirement.
3. Optional checkbox/toggle: “Save as workspace default”.
4. Persist selections as run overrides and optionally routes.

No picker in CI/headless; ambiguity remains hard error.

## Runtime Selection Rules (User-Facing Contract)
Resolution order:
1. Run overrides (explicit for this run)
2. Workspace routes
3. User routes
4. Single unambiguous provider profile auto-select
5. TUI picker (interactive only)
6. Error with remediation (CLI/headless)

Specificity rule:
1. `mcp.<alias>` route wins over provider route for that alias.
2. Provider route applies when no alias route is set.

## Config/Data Model Refinement
Keep existing file compatibility, but present a clearer conceptual API.

### Current structures (kept for compatibility)
1. `auth.defaults` (provider-style selectors and optional `mcp.<alias>`)
2. `auth.mcp_alias_profiles`
3. workspace `auth.defaults.toml`

### Introduce internal normalized routing view
Add a service-level normalized route model:
1. `Route(scope_type="provider", key="<provider>", profile_id="<id>")`
2. `Route(scope_type="mcp_alias", key="<alias>", profile_id="<id>")`

Normalization reads both legacy maps and writes back to current persisted format during migration phase.

## CLI Refinement
Add dedicated route commands:
1. `loom auth route list`
2. `loom auth route set provider <provider> <profile-id> [--workspace|--user]`
3. `loom auth route set mcp <alias> <profile-id> [--workspace|--user]`
4. `loom auth route unset provider <provider> [--workspace|--user]`
5. `loom auth route unset mcp <alias> [--workspace|--user]`

Keep existing commands for compatibility:
1. `loom auth select/unset`
2. `loom auth bind-mcp/unbind-mcp`

Mark older forms as compatibility paths in help text once route commands ship.

## TUI Refactor Scope

### A) Simplify MCP Manager
1. Keep only MCP fields/actions.
2. Add clear status line: “Auth route managed in `/auth routes`”.
3. Optional quick action button: “Open Auth Routes for this alias”.

### B) Simplify Auth Profile Manager
1. Remove selector and MCP alias binding inputs.
2. Keep profile-centric fields only.
3. Add hint: “Routing handled in `/auth routes`”.

### C) New Auth Routes Manager
1. List-first view of effective routes.
2. Form controls:
   - route type
   - key
   - profile id
   - target scope (workspace/user)
3. Actions:
   - Add/Update route
   - Remove route (confirm)
   - Filter by provider/alias

### D) Run Picker Integration
1. Hook into process run kickoff and relevant run commands.
2. Present unresolved requirements and candidate profiles.
3. Save as run override and optionally persist route.

## Implementation Plan

## Phase 0: Contract and Terminology
1. Freeze user-facing vocabulary: server/profile/route/override.
2. Update help strings and docs to avoid “selector” jargon in primary UX.

Exit criteria:
1. Team glossary documented.
2. Command help and TUI hints align with glossary.

## Phase 1: Routing Service Layer
1. Add normalized route model + merge/precedence utilities.
2. Add compatibility readers for current persisted fields.
3. Add validators for provider route vs MCP alias route keys.

Likely files:
1. `src/loom/auth/config.py`
2. `src/loom/auth/runtime.py`
3. New module: `src/loom/auth/routes.py`

Exit criteria:
1. Existing behavior preserved via compatibility tests.
2. New route APIs available for CLI/TUI.

## Phase 2: CLI Route Commands
1. Implement `loom auth route ...` command group.
2. Keep existing commands but reference route commands in output.
3. Add `loom auth route explain <provider|mcp.alias>` (optional) for diagnostics.

Likely files:
1. `src/loom/__main__.py`
2. `tests/test_cli.py`
3. `tests/test_auth_config.py`

Exit criteria:
1. Route CRUD fully scriptable.
2. Headless users can configure all multi-account flows without TUI.

## Phase 3: TUI Screen Split and Clarity
1. Remove routing inputs from `AuthManagerScreen`.
2. Keep MCP manager server-only.
3. Add new `AuthRoutesScreen`.
4. Route `/auth routes` and command palette entry.

Likely files:
1. `src/loom/tui/screens/auth_manager.py`
2. `src/loom/tui/screens/mcp_manager.py`
3. `src/loom/tui/screens/auth_routes.py` (new)
4. `src/loom/tui/app.py`
5. `src/loom/tui/commands.py`
6. `tests/test_tui.py`

Exit criteria:
1. No mixed responsibility fields in MCP/Auth screens.
2. Routes are editable in dedicated UX.

## Phase 4: Run Picker for Ambiguity
1. Add preflight resolution summary.
2. If unresolved and interactive, show picker.
3. Persist optional workspace route when user opts in.
4. Ensure no picker path in non-interactive runs.

Likely files:
1. `src/loom/tui/app.py`
2. `src/loom/auth/runtime.py`
3. `src/loom/tools/delegate_task.py`
4. `tests/test_tui.py`
5. `tests/test_orchestrator.py`

Exit criteria:
1. Multi-account per MCP is selectable at run start without command syntax knowledge.
2. CI behavior remains deterministic and unchanged.

## Phase 5: Documentation and Migration Guidance
1. Document quick-start:
   - create profiles
   - route alias/provider defaults
   - run with override
2. Add migration section from old selector/bind commands.
3. Update examples in `docs/auth.toml.example`.

Likely files:
1. `INSTALL.md`
2. `docs/agent-integration.md`
3. `docs/auth.toml.example`
4. `README.md` (if needed)

Exit criteria:
1. New users can configure multi-account MCP flow in under 5 minutes.

## Test Strategy
1. Unit:
   - route normalization and precedence
   - route validation and compatibility parsing
2. CLI:
   - `auth route` CRUD
   - compatibility command parity
3. TUI:
   - `/mcp` opens server-only manager
   - `/auth` opens profile-only manager
   - `/auth routes` opens routing manager
   - run picker appears only on ambiguity
4. Integration:
   - one MCP alias + multiple profiles + route + run override behavior

## Risks and Mitigations
1. Risk: breaking existing selector-driven automation.
   Mitigation: keep legacy commands and parser; add deprecation messaging only.
2. Risk: route precedence confusion.
   Mitigation: add “effective route” view and optional `explain` command.
3. Risk: TUI complexity creep.
   Mitigation: strict scope boundaries per screen and phased rollout.

## Success Criteria
1. User can answer “which profile does alias X use?” in one screen.
2. User can switch alias X to another profile for one run in <= 3 interactions.
3. User can persist or clear defaults without remembering selector syntax.
4. Support ticket class “auth vs mcp manager confusion” is eliminated.
