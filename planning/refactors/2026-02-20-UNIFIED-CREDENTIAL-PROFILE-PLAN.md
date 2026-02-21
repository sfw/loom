# Unified Credential and Profile System Plan (2026-02-20)

## Objective
Refactor Loom credential handling from ad-hoc env/config usage into a unified, private, run-scoped auth system that works across:
1. MCP tools (stdio and hosted transports)
2. Bundled process package tools
3. Built-in tools that call external APIs
4. Any future tool adapter using OAuth, API keys, service accounts, or CLI passthrough auth

Hard requirements:
1. Secrets are never required to be stored in plaintext project files.
2. Multiple accounts per provider are first-class.
3. Account/profile selection can happen per run.
4. Headless/CI execution remains deterministic.
5. Existing MCP and tool behavior remains backward compatible during rollout.

## Relationship to Existing MCP Plan
This plan is Track C and complements:
`/Users/sfw/Development/loom/planning/refactors/2026-02-17-MCP-CONFIG-MANAGEMENT-PLAN.md`

Track B (MCP transport + OAuth) stays intact and becomes a dependency for hosted MCP support. Track C generalizes credential handling for all tools, not only MCP.

Authority boundary:
1. `loom auth` is the single credential/profile authority.
2. `loom mcp auth ...` remains as a compatibility-focused command surface that delegates to shared auth services.
3. Track B must consume Track C token/profile primitives rather than introducing a parallel token store.

## Current Gaps to Eliminate
1. No shared credential abstraction for non-MCP tools.
2. No universal account profile model.
3. No run-scoped profile selection primitive in orchestration.
4. No central token lifecycle for non-MCP OAuth providers.
5. Inconsistent secret storage practices across integrations.

## Non-Goals
1. Building a remote cloud secret service.
2. Replacing provider-native auth flows where a CLI already manages credentials well (we integrate, not reimplement).
3. Introducing team-shared secret sync in this phase.

## Design Principles
1. Store references, not secret values, in Loom config.
2. Resolve secrets at execution time only.
3. Scope credentials to run context and required tools.
4. Prefer OS keychain first, explicit plaintext fallback only by opt-in.
5. Keep provider/tool auth logic declarative where possible.

## Proposed Architecture

### 1) Core Types
1. `SecretRef`: typed pointer to a secret, not the secret itself.
2. `SecretBackend`: backend interface (`env`, `keychain`, optional external vault adapters).
3. `AuthProfile`: named non-secret profile (provider/account metadata + secret refs + auth mode).
4. `AuthProfileRegistry`: profile config loader (canonical user path + optional explicit overlay).
5. `CredentialResolver`: runtime resolver that turns `SecretRef` into ephemeral credential material.
6. `TokenBroker`: OAuth token lifecycle manager (issue/refresh/revoke metadata + shared store integration).
7. `RunAuthContext`: immutable per-run selection map exposed to tools.

### 2) Credential Modes
1. `api_key`
2. `oauth2_pkce`
3. `oauth2_device`
4. `service_account_json`
5. `env_passthrough`
6. `cli_passthrough` (tool shells out to provider CLI that already holds auth)

### 3) Profile Registry Files
Config files (metadata only):
1. `~/.loom/auth.toml`
2. optional `--auth-config <path>` overlay (for testing/team workflows)

Merge precedence:
1. explicit `--auth-config`
2. user `~/.loom/auth.toml`

Workspace-specific defaults (non-secret only):
1. Optional workspace mapping file: `./.loom/auth.defaults.toml`
2. Stores provider -> profile id defaults only (no secret refs)
3. Used for run disambiguation convenience

### 4) Example Profile Shapes

API key profile:
```toml
[auth.profiles.ga_acme_prod]
provider = "google_analytics"
mode = "api_key"
account_label = "ACME Prod"
property_id = "123456789"
secret_ref = "keychain://loom/google_analytics/ga_acme_prod/api_key"
```

OAuth profile:
```toml
[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
account_label = "Marketing Workspace"
scopes = ["read:content", "write:content"]
token_ref = "keychain://loom/notion/notion_marketing/tokens"
```

Service account profile:
```toml
[auth.profiles.gcp_reporting_bot]
provider = "google_cloud"
mode = "service_account_json"
account_label = "Reporting Bot"
secret_ref = "keychain://loom/google_cloud/gcp_reporting_bot/service_account"
```

## Tool Contract Changes

### 1) ToolContext Extension
Extend `ToolContext` with auth accessors (read-only):
1. selected profile id(s)
2. capability-scoped credential fetch API
3. non-secret profile metadata lookup

Tools should not read raw config files directly once migrated.

### 2) Tool Auth Declaration
Allow tools/processes to declare required auth capabilities:
1. provider id
2. required mode(s)
3. required scopes (for OAuth)

This powers run preflight and profile disambiguation.

## Runtime Behavior

### 1) Run Preflight
Before execution:
1. collect auth requirements from active process + tools (+ MCP servers)
2. resolve selected profiles using this order:
   - explicit run flags
   - process defaults
   - workspace defaults
   - user defaults
   - single unambiguous match
   - interactive selection if multiple (TTY only)
3. verify credentials are available and valid (including OAuth freshness)
4. fast-fail with actionable remediation if unresolved

Headless rule:
1. If no TTY or running in CI and selection is ambiguous, fail fast.
2. Required remediation: pass explicit `--auth-profile ...` mapping or define non-interactive defaults.

### 2) Run-Scoped Selection UX
CLI:
1. `loom run ... --auth-profile provider=profile_name`
2. `loom run ... --auth-profile google_analytics=ga_acme_prod --auth-profile notion=notion_marketing`

TUI:
1. if ambiguous, show account picker before run starts
2. persist optional workspace default mapping (non-secret)

CI/non-interactive:
1. picker is never used
2. ambiguity is an error

### 3) Exposure Model
Only selected profiles are available in `RunAuthContext`. Tools cannot list all profiles by default.

## How It Works in All Situations

### Situation A: Single account API key
1. user creates one profile with `mode=api_key`
2. secret stored in keychain (or env reference)
3. run preflight auto-selects the only profile
4. tool gets credential at runtime via resolver

### Situation B: Multiple accounts for same API
1. user creates `ga_acme_prod`, `ga_acme_stage`, `ga_client_x`
2. each has separate secret ref + metadata
3. run chooses profile by explicit flag or interactive picker
4. selection applies only to that run

### Situation C: Workspace default account
1. workspace `auth.defaults.toml` maps provider default to one profile id
2. all runs in workspace use it unless explicitly overridden
3. no secret duplication in repo files

### Situation D: Hosted MCP with OAuth
1. MCP server alias references OAuth profile
2. `loom mcp auth login <alias>` stores refresh/access tokens in keychain
3. preflight checks token status
4. token broker refreshes automatically near expiry

### Situation E: MCP stdio token env
1. profile mode `env_passthrough` points to environment source or keychain secret
2. runtime resolves and injects only selected env vars into spawned process
3. `${ENV_VAR}` references are expanded at runtime with explicit missing-var errors

### Situation F: Service account JSON
1. JSON is stored in keychain secret or secure local file reference outside repo
2. resolver materializes ephemeral bytes/file in scratch for run
3. cleanup occurs when run ends
4. crash-safe janitor reaps stale credential temp files on startup and periodic maintenance sweep

### Situation G: Tool delegates to provider CLI (own store)
1. profile mode `cli_passthrough` records required CLI (`gh`, `gcloud`, etc.)
2. preflight runs a lightweight auth check command
3. tool executes without Loom reading provider secrets directly

### Situation H: CI / headless
1. profiles reference env vars injected by CI secret manager
2. run includes explicit `--auth-profile` mapping
3. no interactive prompts; unresolved auth fails fast

### Situation I: No keychain available
1. explicit opt-in fallback to encrypted local token store or plaintext fallback flag
2. Loom emits strong warning and audit event
3. production guidance remains keychain/vault first

### Situation J: Token revocation / credential rotation
1. credential check fails at preflight or call-time
2. status becomes `auth_expired` or `auth_invalid`
3. actionable remediation: login/refresh/update secret ref

### Situation K: Mixed-process run (multiple providers)
1. process may require multiple providers simultaneously
2. selection map can include one profile per provider
3. resolver isolates each provider credential path

### Situation L: Shared repo, private local creds
1. repo stores only profile ids/default mappings and non-secret metadata
2. each contributor binds profile ids to local keychain/env references
3. no cross-user secret leakage through git

## Security and Privacy Controls
1. Redact secrets in logs, events, and CLI/TUI output.
2. Add denial-by-default for first-party tools and MCP adapters requesting undeclared provider credentials.
3. Add audit log events for:
   - profile create/edit/delete
   - secret ref updates
   - auth login/logout/refresh
4. Optional allowlist of binaries for `cli_passthrough`.
5. Time-bound in-memory credential cache for one run only.
6. Ephemeral credential files (service-account materialization) use restrictive permissions and short TTL.

Bundled tool boundary note:
1. Bundled tools are arbitrary Python and cannot be hard-isolated without additional runtime sandboxing.
2. In this phase we enforce declaration + preflight + audit for bundled tools.
3. Future hard sandboxing for bundled tools is a follow-up security track.

## Backend Compatibility Policy
Default backend order:
1. Explicit backend selection from profile config
2. OS keychain provider for current platform (if available/unlocked)
3. Environment-variable backend
4. Explicit opt-in fallback store

Platform behavior:
1. macOS desktop: Keychain preferred.
2. Windows desktop: Credential Manager preferred.
3. Linux desktop: Secret Service (`libsecret`) preferred when available.
4. Linux headless / CI: env backend expected by default.
5. If selected backend is unavailable, fail fast with actionable remediation.

## Repository Hygiene (Example + Git Ignore)
1. Ship canonical example file: `docs/auth.toml.example`.
2. Keep real credentials and secret refs out of repo-owned runtime files by default.
3. Recommend ignoring workspace-local auth files:
   - `.loom/auth.toml`
   - `.loom/auth.defaults.toml`
4. If teams intentionally version-control `auth.defaults.toml` (non-secret), require code review policy confirming no secret refs.

## Migration Strategy

### Phase M0: Compatibility + Safety
1. Keep existing MCP env behavior working.
2. Implement `${ENV}` expansion semantics with tests.
3. Keep current config parsing behavior for v1 fields.

### Phase M1: Core Auth Types
1. Add `SecretRef`, `AuthProfile`, registry loader/merge.
2. Add `CredentialResolver` with `env` + `keychain` backends.
3. Introduce `auth.toml` parsing and validation.

### Phase M2: UX and Commands
1. Add CLI commands:
   - `loom auth profile list/show/add/edit/remove`
   - `loom auth select` (workspace default mapping)
   - `loom auth check`
2. Add TUI `/auth` commands and selector flow.

### Phase M3: Orchestrator Integration
1. Add run preflight and run-scoped profile selection.
2. Add explicit provider requirement declarations in process/tool metadata.
3. Fail fast with actionable missing-auth messages.

### Phase M4: MCP and Tool Adoption
1. Bridge MCP aliases to auth profiles.
2. Migrate built-in API-calling tools to resolver API.
3. Add migration helper command from MCP env credentials to profiles.
4. Publish migration guide for package tool authors.

### Phase M5: OAuth Broker Generalization
1. Reuse Track B OAuth code for non-MCP providers where relevant.
2. Add common token status/refresh surfaces.
3. Standardize auth health states for CLI/TUI and process verifier feedback.

### Phase M6: Hardening and Cleanup
1. Enforce declaration-based credential access in tool runtime.
2. Add guard tests for redaction and no-plaintext persistence.
3. Deprecate direct ad-hoc secret access paths in first-party tools.

## PR Sequence (Suggested)
1. PR-C1: Env-ref expansion + tests (bridges Track B Phase 0)
2. PR-C2: `auth.toml` schema + loader + merge precedence
3. PR-C3: Secret backends (`env`, `keychain`) + resolver API
4. PR-C4: CLI auth profile commands
5. PR-C5: Run preflight + `--auth-profile` selection wiring
6. PR-C6: TUI account selection and `/auth` command family
7. PR-C7: MCP alias-to-profile binding and OAuth status unification
8. PR-C8: MCP env-to-profile migration tooling + compatibility wrappers
9. PR-C9: Built-in tool migration to resolver API
10. PR-C10: Package author docs + examples + policy checks
11. PR-C11: Hardening, audits, and deprecation toggles

## Test Plan

### Unit
1. profile parse/merge precedence
2. secret resolution per backend
3. redaction invariants for logs/CLI/TUI payloads
4. run selection resolution order and ambiguity handling
5. crash-safe stale-temp cleanup logic for materialized credential files

### Integration
1. run blocked on missing profile
2. run succeeds with explicit profile mapping
3. OAuth refresh path and auth-expired transitions
4. MCP stdio env injection with expanded refs
5. bundled tool accessing resolver API with multi-account selection
6. crash/restart scenario cleans stale service-account temp files before next run

### Security Regression
1. grep/guard tests ensuring no token/plaintext is written to `auth.toml` or `mcp.toml`
2. log snapshot tests for secret redaction
3. tests for undeclared credential access denial

## Exit Criteria
1. Any Loom tool integration can request credentials through one shared API.
2. Users can safely manage multiple accounts per provider.
3. Per-run account selection is available in CLI and TUI.
4. OAuth/API-key/service-account/CLI passthrough are supported under one model.
5. Secrets are not required in repo files and are redacted in output/logging.
6. MCP Track B and Track C behavior is coherent and documented together.
