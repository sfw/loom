# MCP Tool Management and Config Separation Plan (2026-02-17)

## Execution Status (2026-02-17)
Status: Proposed

## Scope
- Add a first-class MCP management workflow for Loom users.
- Separate MCP server/tool config from `loom.toml` into dedicated MCP config.
- Provide safe, scriptable CLI commands and practical TUI controls.
- Define security boundaries for agent-driven configuration changes.

## Problem Summary
- MCP servers are currently configured inside `loom.toml`, mixing model/runtime settings with tool-bridge execution policy.
- `loom setup` rewrites `~/.loom/loom.toml`, which can unintentionally drop MCP configuration.
- No dedicated CLI/TUI UX exists for adding/editing/removing MCP servers.
- Users need a safe way to manage secrets and avoid accidental exposure.
- Exposing MCP config mutation to arbitrary remote agents is high risk.

## Recommendations
1. Introduce a dedicated MCP config file at `~/.loom/mcp.toml`.
2. Keep backward compatibility with `[mcp.servers.*]` in `loom.toml` during migration.
3. Ship a dedicated `loom mcp` CLI command group for lifecycle management.
4. Add a minimal TUI slash-command surface for MCP operations.
5. Keep MCP config mutation local-only by default; do not expose write APIs on `loom serve` unless explicitly enabled.

## Target Design

## 1) Config Layout and Merge Rules
- Primary MCP config: `~/.loom/mcp.toml`.
- Optional workspace override: `<workspace>/.loom/mcp.toml`.
- Legacy fallback: `[mcp.servers.*]` in active `loom.toml`.

Merge precedence (highest wins):
1. Explicit `--mcp-config <path>` (new CLI option for MCP-aware commands)
2. Workspace `./.loom/mcp.toml`
3. User `~/.loom/mcp.toml`
4. Legacy `[mcp]` section from active `loom.toml`

Server alias collision rule:
- Same alias in a higher-precedence source replaces lower-precedence alias.

`enabled = false` behavior:
- Disabled server remains in config but is excluded from discovery/registration.

## 2) File Schema
Keep existing schema shape for minimal migration cost:

```toml
[mcp.servers.notion]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-notion"]
cwd = ""
timeout_seconds = 30
enabled = true

[mcp.servers.notion.env]
NOTION_TOKEN = "${NOTION_TOKEN}"
```

Secret handling guidance:
- Prefer environment indirection (`"${VAR_NAME}"`) over plaintext literal tokens.
- CLI output must redact env values by default.

## 3) CLI Command Group
Add `loom mcp` subcommands:
- `loom mcp list [--json] [--verbose]`
- `loom mcp show <alias> [--json]`
- `loom mcp add <alias> --command <cmd> [--arg <arg> ...] [--env KEY=VALUE ...] [--env-ref KEY=ENV_VAR ...] [--cwd <path>] [--timeout <sec>] [--disabled]`
- `loom mcp edit <alias> [same mutating flags as add]`
- `loom mcp remove <alias>`
- `loom mcp enable <alias>`
- `loom mcp disable <alias>`
- `loom mcp test <alias>` (runs `tools/list` probe and prints discovered tool names + diagnostics)
- `loom mcp migrate` (moves legacy `[mcp]` from `loom.toml` to `mcp.toml`)

CLI behavior requirements:
- Atomic writes (`write temp + fsync + rename`).
- Preserve unknown sections/comments where feasible; if not, document normalized rewrite behavior.
- Redact sensitive values in stdout/stderr and logs.

## 4) TUI UX
Add slash commands:
- `/mcp` (help/overview)
- `/mcp list`
- `/mcp show <alias>`
- `/mcp test <alias>`
- `/mcp enable <alias>`
- `/mcp disable <alias>`
- `/mcp remove <alias>`

Phase 2 TUI enhancement:
- Modal/editor workflow for add/edit (`/mcp add`, `/mcp edit`) with explicit save confirmation.

TUI implementation note:
- Reuse same backend service as CLI (single MCP config manager API) to avoid duplicated parsing/writing logic.

## 5) Runtime Reload
- After MCP config mutation (CLI or TUI), trigger registry refresh in-process:
  - Reconcile tools via existing MCP synchronizer refresh hook.
  - Re-render `/tools` output and status hints.
- For external processes (`loom run`, `loom serve`), refreshed config applies on next process start unless an explicit runtime reload hook is added.

## 6) Security Model
Default posture:
- MCP config mutation is local and user-initiated only.
- `loom serve` exposes no config-write API by default.

Optional advanced posture (deferred):
- Admin API for MCP mutation behind explicit `--enable-admin-config` flag and auth token.
- Audit logging for all mutations.
- Optional allowlist for permitted command prefixes.

## 7) Agent Access Recommendation
- Yes, provide CLI-based management for automations and trusted local agents.
- No, do not allow untrusted/remote agents to mutate MCP config via public API by default.
- If API mutation is later enabled, require explicit operator opt-in and authentication.

## Implementation Plan

## Phase 1: MCP Config Manager Core
- Add `src/loom/mcp/config.py`:
  - load/merge logic for user/workspace/legacy sources
  - schema validation and normalization
  - atomic write helpers
  - redaction helpers
- Add tests:
  - precedence/merge behavior
  - alias override behavior
  - env redaction behavior

## Phase 2: CLI Commands
- Add `loom mcp` click group and subcommands in `src/loom/__main__.py`.
- Implement deterministic JSON output for script usage.
- Add command tests in `tests/test_cli.py`:
  - add/edit/remove/enable/disable
  - list/show redaction
  - migrate from `loom.toml`
  - test probe success/failure paths

## Phase 3: Runtime Integration
- Update startup/config plumbing to read merged MCP config and feed tool registry.
- Ensure legacy config still works with warning guidance to migrate.
- Add integration tests:
  - MCP tools loaded from `mcp.toml`
  - workspace override precedence
  - reload/refresh behavior after mutations

## Phase 4: TUI Slash Commands
- Extend slash command parser/help/autocomplete:
  - `/mcp` command family
- Show concise success/failure feedback in chat log.
- Add TUI tests:
  - parse/dispatch behavior
  - visible feedback
  - no regression for existing slash commands/autocomplete

## Phase 5: Docs and Migration Guidance
- Update:
  - `README.md`
  - `docs/tutorial.html`
  - `docs/agent-integration.md`
- Add examples for:
  - secure env reference usage
  - migration from legacy config
  - TUI + CLI workflows

## Test Plan
- Unit:
  - parser/merge/serializer/atomic write/redaction.
- CLI:
  - all `loom mcp` commands including error cases.
- Integration:
  - MCP tool discovery from new config path.
  - live probe with test MCP fixture.
- TUI:
  - slash command correctness + autocomplete.
- Regression:
  - existing setup flow still works and does not delete `mcp.toml`.

## Risks and Mitigations
- Risk: config sprawl/confusion.
  - Mitigation: clear precedence docs and `loom mcp doctor` (optional follow-up).
- Risk: accidental secret disclosure in logs.
  - Mitigation: default redaction and dedicated tests.
- Risk: alias conflicts across files.
  - Mitigation: explicit precedence and warning output showing source selected.
- Risk: user expectation mismatch on live reload.
  - Mitigation: explicit messaging after mutations: “applies immediately in current TUI, otherwise on next run.”

## Demo-Ready Exit Criteria
- MCP servers can be fully managed via CLI without manual file edits.
- TUI exposes practical MCP list/show/test/enable/disable/remove commands.
- MCP config survives `loom setup` reruns.
- Secrets are redacted in CLI/TUI/log output.
- Docs clearly explain user workflow and security boundaries.
