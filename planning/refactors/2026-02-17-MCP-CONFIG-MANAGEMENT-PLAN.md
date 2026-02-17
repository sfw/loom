# MCP Config, Connection Management, and OAuth Plan (2026-02-17)

## Status
- Track A (done): MCP config separation + CLI/TUI management was implemented and tested.
- Track B (new): transport-aware MCP connections + OAuth support is required for major hosted MCP servers and is not yet implemented.

## Why This Refinement Exists
The original plan delivered config management well, but we now need first-class support for hosted MCP servers (Notion, Figma, Atlassian, Asana, etc.) that depend on OAuth and HTTP transports.

Without this, Loom can only reliably bridge local stdio MCP servers.

## What Has Been Completed (Audit of Existing Work)
Implemented from the earlier plan:
- Dedicated MCP config management in `src/loom/mcp/config.py`.
- Merge precedence:
  1. `--mcp-config <path>`
  2. `<workspace>/.loom/mcp.toml`
  3. `~/.loom/mcp.toml`
  4. legacy `[mcp]` in `loom.toml`
- CLI `loom mcp` command group:
  - `list/show/add/edit/remove/enable/disable/test/migrate`
- TUI `/mcp` command family:
  - `list/show/test/enable/disable/remove`
- Legacy migration flow and guidance.
- Redaction in CLI/TUI display.
- Unit/integration/TUI test coverage for the above.

## Gaps Found After Audit
1. External MCP bridging is stdio-only:
   - Current server schema is command/args/env/cwd/timeout/enabled only.
2. No OAuth lifecycle:
   - No login callback flow, no refresh-token handling, no token store.
3. Env ref semantics are incomplete:
   - `--env-ref KEY=VAR` stores `${VAR}`, but runtime does not expand those placeholders before spawn.
4. Docs mismatch:
   - Docs mention `loom mcp-serve --transport sse`, but CLI currently exposes stdio only.
5. No connection-health model:
   - We have probe/test, but no persistent runtime status model across transports/auth states.

## Requirement Confirmation
If Loom must support hosted MCP servers like Notion in a robust way, OAuth support is effectively a core requirement.

Token-only stdio alternatives exist for some ecosystems, but they are incomplete, vendor-specific, or not the official long-term path.

## Design Goals (Track B)
1. Support both local and hosted MCP servers with one config model.
2. Treat OAuth as first-class for hosted MCP.
3. Keep secrets out of `mcp.toml` where possible.
4. Preserve current stdio behavior and backward compatibility.
5. Make connection/auth state clear in CLI and TUI.
6. Align with MCP transport direction: Streamable HTTP as primary remote transport; legacy SSE compatibility mode only.
7. Prefer protocol-level correctness via MCP SDK integration over ad-hoc HTTP implementations where feasible.

## Proposed MCP Connection Model

### 1) Transport Types
Add explicit per-server transport:
- `transport = "stdio"` (default; current behavior)
- `transport = "streamable_http"` (new primary remote)
- `transport = "sse_legacy"` (compat only; optional/deprecated)

### 2) Auth Modes
Add explicit per-server auth policy:
- `auth = "none"`
- `auth = "env_bearer"` (token from env)
- `auth = "oauth"` (new)

### 3) Config Shape (v2)
Backward compatible with existing stdio blocks.

#### Stdio server example
```toml
[mcp.servers.notion_local]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-notion"]
timeout_seconds = 30
enabled = true

[mcp.servers.notion_local.env]
NOTION_TOKEN = "${NOTION_TOKEN}"
```

#### Hosted OAuth server example
```toml
[mcp.servers.notion]
transport = "streamable_http"
url = "https://mcp.notion.com/mcp"
auth = "oauth"
enabled = true
timeout_seconds = 30

[mcp.servers.notion.oauth]
scopes = ["read:content", "write:content"]
# optional overrides:
# client_id = "..."
# authorization_server = "https://..."
# audience = "..."
```

## Streaming vs SSE Decision

### Policy
- For hosted servers, Loom should target `streamable_http` first.
- `sse_legacy` is fallback-only compatibility for older servers.

### Behavior
1. If configured `streamable_http`, use that directly.
2. If configured `sse_legacy`, use explicit legacy flow.
3. Optional `compat_auto` mode (future):
   - Attempt Streamable HTTP first.
   - Fallback to legacy SSE only for transport-level incompatibility responses.

### Loom MCP Server (`loom mcp-serve`)
- Keep stdio as stable default.
- Either:
  - implement real remote transport mode(s), or
  - remove stale doc claims until implemented.
- Immediate action: docs must match current CLI behavior.
- Scope boundary for this plan: external MCP connectivity is priority; `loom mcp-serve` remote transports are optional follow-up unless explicitly scheduled.

## OAuth Architecture

### 1) Login Flow
- CLI command: `loom mcp auth login <alias>`
- Use Authorization Code + PKCE.
- Open browser and start local callback listener for redirect completion.
- Discover authorization server metadata from MCP/protected resource metadata when available; allow explicit override fields when discovery is absent.

### 2) Token Storage
- Do not write access/refresh tokens to `mcp.toml`.
- Store tokens in OS keychain via `keyring` (preferred).
- Keep only non-secret metadata in config/state (issuer, scopes, expiry metadata).
- Optional explicit fallback `--allow-plaintext-token-store` (off by default) for headless hosts where keychain is unavailable.

### 3) Refresh
- Automatic refresh before expiry.
- On refresh failure, mark server auth state as expired and surface re-login hint.

### 4) Commands
Add:
- `loom mcp auth login <alias>`
- `loom mcp auth status [alias]`
- `loom mcp auth logout <alias>`
- `loom mcp auth refresh <alias>` (manual repair path)
- Optional headless support:
  - `loom mcp auth login <alias> --device` for environments where browser callbacks are not possible.

## Runtime Connection Manager
Introduce a single manager for discovery, auth, and tool sync:

- `MCPConnectionManager`
  - resolves merged config
  - instantiates per-server connection adapters
  - tracks state:
    - configured
    - connecting
    - authenticated
    - auth_expired
    - healthy
    - degraded
    - error
  - exposes diagnostics for CLI/TUI

- Adapters:
  - `StdioConnection`
  - `StreamableHttpConnection`
  - optional `SSELegacyConnection`

- Registry integration:
  - keep current reconciliation behavior
  - enrich failures with transport/auth reason (not just generic discovery failure)

## Process/Orchestration Integration
To make MCP-backed tools reliable in processes:

1. Preflight at process activation/run:
   - Validate required MCP servers are configured and enabled.
   - Validate auth state for required OAuth servers.
2. Fast-fail with actionable message:
   - Example: "Process requires MCP server `notion` but OAuth login is missing. Run `loom mcp auth login notion`."
3. Optional process metadata extension:
   - `tools.required_mcp_servers: [notion, linear]`

## Security Model (Refined)
Defaults:
- Local-only mutation remains default.
- No remote config-write API on `loom serve`.
- Redaction always on for output/logging.

Additional hardening:
- Command allowlist option for stdio server binaries.
- Per-server "trusted" flag before enabling execution.
- Audit log for MCP config/auth mutations.

## Migration Strategy
1. Read legacy and v1 config seamlessly.
2. Keep writing compatible structure for stdio entries.
3. New fields (`transport`, `auth`, `url`, `oauth`) are additive.
4. Add migration helper:
   - `loom mcp migrate --to-v2` for explicit normalization.

## Implementation Plan (Track B)

## Phase 0: Correctness and Doc Parity
- Fix docs/CLI mismatch for `mcp-serve --transport sse`.
- Define final transport/auth schema and validation.
- Fix `${ENV}` interpolation semantics for stdio env refs.
- Lock implementation dependency strategy:
  - adopt MCP SDK client capabilities for Streamable HTTP/OAuth where practical
  - define minimum compatible `mcp` extra version for Loom.

Exit criteria:
- Docs match behavior.
- Env-ref semantics are explicit and tested.

## Phase 1: Transport-Aware Config + Adapter Layer
- Extend schema with `transport/auth/url/oauth`.
- Introduce connection adapter interface.
- Keep existing stdio path as adapter implementation.

Tests:
- Config parse/merge for old + new schema.
- Adapter selection and validation failures.

## Phase 2: Streamable HTTP Transport
- Implement remote MCP connection over Streamable HTTP.
- Add probe and tool discovery parity with stdio path.
- Optional compatibility adapter for legacy SSE servers.
- Prefer SDK-backed transport client path first; use custom client only for explicitly unsupported SDK paths.

Tests:
- Mock server integration for list/call flows.
- Retry/backoff and error classification.

## Phase 3: OAuth Support
- Add auth command group and PKCE login flow.
- Add token store abstraction (keyring + explicit fallback).
- Auto-refresh handling and auth status plumbing.

Tests:
- OAuth state machine unit tests.
- Token persistence/retrieval/refresh tests.
- End-to-end mock auth provider integration.

## Phase 4: TUI MCP Connection UX
- Add `/mcp` auth/status affordances:
  - status list with auth/health badges
  - trigger login/logout/test/reload
- Keep mutation controls reusing shared backend.

Tests:
- Slash command routing and status rendering.
- failure-state UX messages.

## Phase 5: Process Readiness and Preflight
- Add required-MCP preflight checks for process runs.
- Improve verification feedback for MCP connectivity/auth issues.

Tests:
- process run blocked on missing auth/config.
- success path when authenticated.

## Phase 6: Hardening and Release
- Add nightly canary with at least one hosted MCP mock.
- Performance and timeout tuning.
- Security review pass for token handling and logs.

## Updated Exit Criteria
- Loom can manage and use both local stdio and hosted MCP servers.
- OAuth-required MCP servers are supported with first-class login/logout/status.
- Process runs can depend on MCP servers with clear preflight behavior.
- Streamable HTTP is supported; legacy SSE is explicit compatibility mode only.
- Docs/tutorial/CLI help/TUI behavior are aligned.
- Backward compatibility is preserved for existing stdio-only `mcp.toml` entries without manual migration.

## Primary References
- MCP spec: transports and auth requirements
  - https://modelcontextprotocol.io/specification/2025-11-05/basic/transports
  - https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization
- Notion MCP OAuth requirements
  - https://developers.notion.com/guides/mcp/get-started-with-mcp
  - https://developers.notion.com/guides/mcp/build-mcp-client
