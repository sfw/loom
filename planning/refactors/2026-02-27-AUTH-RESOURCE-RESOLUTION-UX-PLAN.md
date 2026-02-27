# Auth Resource Resolution UX Plan (2026-02-27)

## Goal
Deliver the best UX for credentials across MCPs, APIs, and tools by making auth selection happen at run start (not as a manual pre-step), while keeping execution deterministic in headless environments.

The target user experience:
1. User configures resources and auth profiles once.
2. User starts `/run` or `loom run`.
3. Loom resolves required auth automatically:
   - auto-select when there is one obvious choice
   - use stored defaults when present
   - prompt only when needed (interactive only)
   - fail with actionable remediation in headless mode

## Final Decisions

### A) Data model and relationship
1. Treat "things that require auth" as auth resources:
   - MCP servers
   - API providers used by tools
   - other auth-dependent integrations
2. Auth profiles are many-to-one to auth resources.
3. Auth profiles remain the single source of credentials.
4. MCP config should not duplicate account credentials when using auth profiles.

### B) Binding location
1. MCP/auth binding lives in auth profile metadata.
2. Add one explicit field on profile: `mcp_server`.
3. Cardinality: one profile -> one MCP server (recommended and accepted).

### C) Run-time selection UX
1. No required pre-run `/auth select ...` step.
2. Selection happens at run/process start based on required auth resources.
3. If exactly one matching profile exists for a required resource, it is implicitly selected as default for that run.
4. If multiple profiles match:
   - use persisted defaults if available
   - otherwise prompt in interactive contexts
   - otherwise fail in non-interactive contexts
5. If no profile matches, fail with remediation.

### D) CLI non-interactive behavior
1. For CLI/headless runs: if explicit profiles are not passed, use defaults.
2. If defaults do not resolve all required resources, error with explicit next steps.

## Why this is the right UX
1. Removes credential duplication (clean mental model).
2. Reduces command burden before each run.
3. Preserves deterministic behavior for CI/headless.
4. Keeps flexibility for multi-account workflows.
5. Keeps MCP connection config and credential ownership separate.

## Architecture Changes

## 1) Auth profile schema
Extend auth profile with optional MCP binding:

```toml
[auth.profiles.notion_prod]
provider = "notion"
mode = "env_passthrough"
account_label = "Notion Prod"
mcp_server = "notion"

[auth.profiles.notion_prod.env]
NOTION_TOKEN = "${NOTION_TOKEN_PROD}"
```

Notes:
1. `mcp_server` is optional (API-only profiles do not need it).
2. Validation: if `mcp_server` is set, it must reference an existing MCP alias at preflight time.

## 2) MCP config responsibility
Keep MCP config connection-oriented:
1. command/args/cwd/timeout/enabled
2. optional static env for backward compatibility
3. no required duplicated per-account credentials when auth profile is used

## 3) Run auth resolution service
Build a single resolution pass used by API runs and TUI `/run`:
1. Gather required auth resources from process + tools.
2. Resolve selected profile per resource using precedence:
   - explicit run overrides
   - workspace defaults
   - user defaults
   - single-match auto-select
   - interactive picker (interactive only)
3. For MCP-required resources, use profile `mcp_server` binding to select MCP alias.
4. Build run-scoped auth context.
5. Fail fast when unresolved.

## 4) MCP bridge integration
Update MCP auth env resolution path:
1. For alias `A`, ask run auth context for env overrides from selected profile bound to `A`.
2. If bound profile exists, inject resolved env from that profile.
3. If no bound profile, preserve current behavior (backward compatibility path).

## 5) Process schema updates
Add optional auth requirements in process contracts:

```yaml
auth:
  required:
    - provider: notion
      source: mcp
    - provider: acme_issues
      source: api
      modes: [api_key]
```

This allows preflight to know what must be selected before execution starts.

## 6) API unresolved-auth contract (critical for non-TUI clients)
API callers cannot use interactive picker flow, so unresolved auth must be machine-readable.

Add structured unresolved-auth response payload for task creation preflight failures:
1. unresolved resources (provider/source)
2. candidate profile IDs per resource
3. defaults currently in effect
4. remediation examples (CLI/API format)

Behavior:
1. API returns a distinct error code/class for unresolved auth.
2. TUI can still use local picker flow.
3. Web/SDK clients can render their own selection UX and retry with explicit metadata overrides.

## 7) Run/task isolation for MCP auth resolution
MCP tool discovery is dynamic and auth-sensitive; this must be isolated by run.

Requirements:
1. Do not let one run's selected profiles affect another run's MCP tool visibility.
2. Ensure task-scoped or run-scoped registry/discovery view for MCP tools.
3. Prevent global cache leakage of auth-constrained MCP tool lists between concurrent runs.

## Command Surface Plan

## Keep stable
1. `loom run --auth-profile ...` remains supported.
2. `/auth use ...` remains supported for run-local override in TUI.
3. `/auth select ...` remains supported for persisted workspace defaults.

## Extend
1. `loom auth profile add/edit`:
   - add `--mcp-server <alias>`
2. TUI `/auth add` and `/auth edit`:
   - add `--mcp-server <alias>`
3. `loom auth show` / `/auth show`:
   - display `mcp_server` binding when present
4. API task create validation response:
   - include structured unresolved-auth payload for client retry logic

No large new command family is required for this phase.

## Legacy Scaffold Cleanup Plan
This work must include explicit cleanup, not just additive changes.

## 1) Auth/MCP legacy routing remnants
1. Remove legacy concepts from active UX/docs:
   - `auth.mcp_alias_profiles`
   - `mcp.<...>` selector mental model
2. Keep parser compatibility only as a bounded migration layer, with deprecation notice.
3. Keep tests for compatibility behavior until deprecation window closes, then remove.

## 2) MCP credential duplication paths
1. Mark MCP `env` credentials as compatibility path, not recommended path.
2. Update CLI/TUI help text to indicate:
   - MCP config = connection config
   - Auth profiles = credential config
3. Add migration guidance from alias-env credentials to auth profiles with `mcp_server` binding.

## 3) UX and docs cleanup
1. Remove mixed terminology ("selectors/routes/bindings") from user-facing copy.
2. Use one vocabulary set consistently: `resource`, `profile`, `default`, `run override`.
3. Update docs/examples to show run-start selection flow as primary UX.

## 4) Package author docs (required)
Update package author documentation so custom tool authors can declare auth requirements correctly.

Required doc updates:
1. `docs/creating-packages.md`
   - add `auth.required` process contract docs
   - add custom-tool auth declaration contract with examples
   - add mode-specific examples (api_key, oauth2_pkce, env_passthrough, cli_passthrough)
2. package migration note:
   - how to move from in-tool ad-hoc env key usage to auth requirement declarations

## Defaulting Rules (authoritative)
For each required auth resource:
1. If explicit run override exists and is valid, use it.
2. Else if workspace default exists and is valid, use it.
3. Else if user default exists and is valid, use it.
4. Else if exactly one profile matches, auto-select it for this run.
5. Else if interactive, prompt user to choose.
6. Else fail.

## Interactive Picker UX
Run-start picker appears only for unresolved resources.

Each row shows:
1. Resource (for example `provider=notion`, source=`mcp`)
2. Candidate profile IDs
3. Account label
4. Bound MCP server (if any)

Actions:
1. Select profile for current run.
2. Optional checkbox to save as workspace default.
3. Confirm to continue run.

## End-to-End Example Workflow

## A) Setup once
1. Add MCP server:
```bash
uv run loom mcp add notion --command npx --arg -y --arg @modelcontextprotocol/server-notion
uv run loom mcp test notion
```
2. Add profiles:
```bash
uv run loom auth profile add notion_dev \
  --provider notion \
  --mode env_passthrough \
  --mcp-server notion \
  --env 'NOTION_TOKEN=${NOTION_TOKEN_DEV}'

uv run loom auth profile add notion_prod \
  --provider notion \
  --mode env_passthrough \
  --mcp-server notion \
  --env 'NOTION_TOKEN=${NOTION_TOKEN_PROD}'
```
3. Add API profile:
```bash
uv run loom auth profile add acme_api_prod \
  --provider acme_issues \
  --mode api_key \
  --secret-ref 'env://ACME_ISSUES_API_KEY_PROD'
```

## B) Run in TUI with no pre-select
1. User runs `/run Sync notes and issues`.
2. Loom preflight sees required resources: `notion (mcp)`, `acme_issues (api)`.
3. If `acme_issues` has one profile, auto-select.
4. If `notion` has two profiles and no default, show picker.
5. User picks `notion_prod`, optionally saves workspace default.
6. Run starts immediately with resolved auth context.

## C) Run in CLI headless
1. User runs:
```bash
uv run loom run "Sync notes and issues" --workspace /path --process /path/process.yaml
```
2. Loom attempts defaults and auto-single-profile rules.
3. If any resource remains ambiguous/unresolved, command fails with:
   - which resource is unresolved
   - candidate profiles found
   - exact commands to fix (`auth select` or `--auth-profile`)

## D) Single-profile implicit default behavior
If exactly one profile exists for a resource, Loom selects it automatically for the run even if no explicit default is stored.

## Tool API Key/OAuth Handling Model

## 1) Tool auth requirement declaration
Every tool that can require credentials must declare its auth requirements.

Example shape (conceptual):
1. `provider` (resource key)
2. `source` (`api` or `mcp`)
3. `modes` (allowed profile modes)
4. optional `scopes` (OAuth)
5. optional `required_env_keys` (for tools expecting multiple keys)

Without declaration:
1. tool is treated as no-auth by default
2. first-party tools must not remain undeclared if they call authenticated APIs

Custom package tooling contract:
1. Custom `Tool` classes must expose auth requirements via the same declaration interface as first-party tools.
2. If a package tool calls external authenticated APIs but omits declaration, process test/lint should fail.
3. Process-level `auth.required` remains a fallback/override, but tool-level declaration is preferred for accuracy.

## 2) Runtime behavior for an API-calling tool
When a tool contains an API call requiring keys:
1. preflight sees requirement from tool declaration
2. resolver selects profile for that provider
3. tool execution receives resolved credential material from auth context
4. tool injects credential into request headers/query/body based on adapter mapping
5. no plaintext key is read from process files

## 3) Tool catalog review and rollout
Yes, this plan includes reviewing built-in tools for auth requirements.

Required rollout steps:
1. inventory all tools and classify as:
   - no-auth
   - optional-auth
   - required-auth
2. add declarations for first-party authenticated tools
3. add regression tests ensuring declared requirements match runtime behavior

## 4) Exposure in auth TUI
Yes, required resources should be visible in auth UX.

Auth manager enhancements:
1. show known auth resources discovered from tools/processes
2. show profile coverage status per resource
3. show unresolved requirements during run-start picker
4. allow direct creation/edit of profiles from unresolved resource context

## Credential Modes and Type Handling
Current `mode` is metadata-only in code today; this plan makes it operational.

## Mode support in this plan
1. `api_key`
   - uses `secret_ref` or explicit env mappings
   - supports one or many keys via profile `env` map
2. `env_passthrough`
   - profile `env` values are resolved and injected to runtime/tool context
3. `oauth2_pkce` / `oauth2_device`
   - uses `token_ref` for token material
   - preflight checks presence/shape
   - refresh lifecycle can be staged (compat path first, brokered refresh next)
4. `cli_passthrough`
   - uses `command` + `auth_check` for lightweight preflight verification

## Validation requirements by mode
1. `api_key`: require `secret_ref` or required env keys
2. `oauth2_*`: require `token_ref`
3. `cli_passthrough`: require `command`
4. `env_passthrough`: require at least one env mapping when used for authenticated resources

## OAuth lifecycle handling (explicit)
To avoid hidden gaps, OAuth handling must define lifecycle behavior:
1. acquisition: how token_ref is initially created/populated (manual/bootstrap flow in this phase)
2. freshness check: preflight validates token presence/shape and optional expiry metadata
3. refresh behavior: if refresh is unsupported in phase 1, fail with explicit remediation
4. error taxonomy: distinguish `auth_missing`, `auth_invalid`, `auth_expired`

## Security posture
1. Resolve secrets at runtime only.
2. Redact all secret values in logs/UI/events.
3. Keep profile metadata in config, secret material in env/keychain refs.

## Migration and Compatibility
1. Existing MCP env-only setups continue to work.
2. Existing auth defaults and run overrides continue to work.
3. `mcp.*` selector route behavior remains unsupported.
4. New `mcp_server` profile field is additive and backward-compatible.

## Implementation Plan (phased)

## Phase 1: Schema and parsing
1. Add `mcp_server` field to auth profile dataclass/parsers/renderers.
2. Add CLI/TUI support for setting/editing/displaying `mcp_server`.
3. Add validation checks for field format.

## Phase 2: Resolver and preflight
1. Introduce required-resource collection.
2. Implement authoritative resolution rules.
3. Wire resolution into API task creation and runner preflight.
4. Add clear, actionable error payloads.
5. Add structured unresolved-auth API response schema.

## Phase 3: Interactive picker
1. Add unresolved-resource picker for TUI `/run`.
2. Add optional "save as workspace default" action from picker.
3. Ensure picker is never used in non-interactive mode.

## Phase 4: MCP runtime binding
1. Update MCP bridge env resolution to use profile->mcp binding.
2. Preserve fallback behavior for legacy env config.
3. Add telemetry/event fields for selected auth plan (redacted).

## Phase 5: Tooling inventory + docs + package authoring support
1. Complete first-party tool auth inventory and declarations.
2. Add guard tests for undeclared first-party authenticated tools.
3. Update `docs/creating-packages.md` with auth requirement authoring contract.
4. Add package test guidance for custom authenticated tools.

## Test Plan

## Unit tests
1. Auth profile parse/render for `mcp_server`.
2. Resolver precedence and branching:
   - explicit override
   - workspace/user defaults
   - single-profile auto-select
   - ambiguous interactive/non-interactive behavior
3. MCP alias binding validation.
4. API unresolved-auth response payload structure.

## Integration tests
1. Process requiring MCP + API resolves correctly with one run.
2. No manual pre-select required before `/run`.
3. CLI non-interactive unresolved case fails with remediation.
4. TUI unresolved case prompts picker and succeeds after selection.
5. Single profile case auto-selects without prompt/default.
6. Concurrent runs with different profile selections do not leak MCP tool visibility/auth context.
7. API client receives structured unresolved-auth response and can retry successfully with overrides.

## Regression tests
1. Existing auth commands keep behavior.
2. Existing MCP config commands keep behavior.
3. Legacy MCP env-only flows still function.
4. Custom package auth requirements are parsed/validated and reflected in preflight.

## Acceptance Criteria
1. User can start a run without pre-select commands and still complete auth selection.
2. Credential duplication between MCP config and auth profiles is no longer required in the primary path.
3. Single-profile resources auto-select by default.
4. Headless runs are deterministic and fail clearly when unresolved.
5. Processes that require MCP/API auth fail early with actionable remediation instead of failing mid-run.
6. Non-TUI/API clients can resolve auth ambiguity using structured error payloads and retry.
7. `docs/creating-packages.md` documents custom tool/process auth requirement authoring.

## Risks and mitigations
1. Risk: ambiguous mapping confusion.
   - Mitigation: picker labels include provider/profile/account label/mcp server.
2. Risk: profile binds to missing MCP alias.
   - Mitigation: preflight validation and targeted error.
3. Risk: migration confusion for existing users.
   - Mitigation: keep compatibility paths and document "auth-first" recommended setup.

## Out of scope
1. Full migration of model provider API keys out of `loom.toml`.
2. Cloud secret vault service.
3. New broad auth command families beyond `mcp_server` extension and run-start selection flow.
