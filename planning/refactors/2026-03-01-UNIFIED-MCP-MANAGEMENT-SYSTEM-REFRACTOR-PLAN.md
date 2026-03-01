# Unified MCP Management System Refactor Plan (2026-03-01)

## Objective
Build a single MCP management architecture that combines:
1. OpenCode strengths: remote transport support, persistent connections, OAuth lifecycle, tools/prompts/resources coverage.
2. Loom strengths: layered config + migration, auth-scoped safety/isolation, strong CLI/TUI operations, broad test discipline.

Target outcome: Loom becomes a first-class MCP consumer for both local and hosted servers without regressing current safety, operator clarity, or reliability.

## Scope
In scope:
1. External MCP server management and runtime (Loom as MCP client).
2. Config model and migration for local + remote servers.
3. OAuth and remote auth lifecycle integration.
4. Tool registry discovery/invocation path upgrades for persistent MCP clients.
5. MCP prompts/resources support alongside tools.
6. CLI/TUI operational flows for auth/connection health.

Out of scope:
1. Re-architecting Loom's own `mcp-serve` server feature beyond parity fixes.
2. Broad non-MCP auth system rewrites unrelated to MCP credentials.
3. New remote write APIs on `loom serve`.

## Product Principles
1. Backward compatible by default for existing stdio MCP setups.
2. Secrets never stored in `mcp.toml` unless explicitly forced.
3. Transport/auth state always visible and actionable in CLI/TUI.
4. Shared policy path for approvals, auth scoping, and execution safety.
5. Prefer deterministic state machines over ad hoc connection logic.

## Production Readiness Criteria (Must-Have)
1. Reliability:
   - MCP call success rate >= 99.5% (excluding upstream 4xx auth/permission failures).
   - Automatic reconnect recovery for transient transport failures within 30 seconds at P95.
   - No unbounded queues, retries, or memory growth under failure.
2. Performance:
   - Repeated-call P95 latency improves by >= 25% vs current one-shot baseline for local MCP tools.
   - Manager overhead for idle configured servers remains bounded (no busy-loop polling).
3. Security:
   - Zero secret material in `mcp.toml`, CLI output, TUI output, or logs by default.
   - OAuth callback and token handling pass dedicated misuse/abuse tests.
   - Remote transport defaults to TLS (`https`) with strict certificate verification.
4. Operability:
   - Every degraded/error state includes a user-actionable remediation in CLI/TUI.
   - On-call runbook and rollback procedure are written and validated in dry run.
   - Canary error budget threshold: failed MCP operations <= 1.0% per rolling 24h before GA.
5. Compatibility:
   - Existing stdio server configs remain functional without edits.
   - Explicit compatibility matrix published for supported MCP features/transports.

## Best-of-Both Synthesis

### Keep from Loom
1. Config precedence and migration ergonomics (`--mcp-config` -> workspace -> user -> legacy).
2. Auth-scoped tool visibility and anti-leak behavior in discovery.
3. Operational command depth (`list/show/add/edit/remove/enable/disable/test/migrate`).
4. Strong integration-test posture across CLI/TUI/runtime.

### Adopt from OpenCode
1. First-class server type model (`local` and `remote`).
2. Persistent connection manager with explicit connect/disconnect/status.
3. Remote transport strategy (Streamable HTTP primary, SSE fallback).
4. OAuth browser callback flow with token refresh and client registration handling.
5. MCP protocol surface support for `tools`, `prompts`, and `resources`.

### Improve beyond both
1. Single canonical timeout policy (avoid config/runtime mismatch).
2. Strict callback correlation keyed only by OAuth state (avoid alias/state mismatch bugs).
3. Uniform header propagation across connect/auth flows.
4. Explicit state machine and health model used by CLI, TUI, and registry.

## Target Architecture

### 1) Config Model v2 (Backward Compatible)
Add typed server config while retaining stdio compatibility:

```toml
[mcp.servers.my_local]
type = "local"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
enabled = true
timeout_seconds = 30

[mcp.servers.notion]
type = "remote"
url = "https://mcp.notion.com/mcp"
enabled = true
timeout_seconds = 30
fallback_sse_url = "https://mcp.notion.com/sse"

[mcp.servers.notion.headers]
X-Team = "${NOTION_TEAM}"

[mcp.servers.notion.oauth]
enabled = true
scopes = ["read:content", "write:content"]
```

Rules:
1. `type=local` requires `command`; ignores remote-only fields.
2. `type=remote` requires `url`; disallows local-only process fields.
3. Existing server blocks without `type` map to `type=local`.

### 2) MCP Connection Manager
Introduce `MCPConnectionManager` with per-alias state:
1. `configured`
2. `connecting`
3. `needs_auth`
4. `auth_in_progress`
5. `healthy`
6. `degraded`
7. `error`
8. `disabled`

Responsibilities:
1. Resolve merged config and instantiate adapters.
2. Maintain persistent sessions for active MCP servers.
3. Trigger reconnect/backoff on transient failures.
4. Surface status and diagnostics for CLI/TUI and registry.
5. Publish capability cache (tools/prompts/resources) with invalidation.

### 3) Transport Adapters
Adapters:
1. `StdioAdapter` for local command servers.
2. `StreamableHttpAdapter` for remote primary path.
3. `SseFallbackAdapter` only when explicitly configured or negotiated fallback.

Policy:
1. Remote connect path starts with Streamable HTTP.
2. Fallback to SSE only for transport-level incompatibility.
3. Capture reason codes on fallback for observability.

### 4) OAuth and Credential Lifecycle
Flow:
1. `loom mcp auth login <alias>` starts PKCE auth.
2. Browser callback handled via local listener with strict `state` correlation.
3. Token storage delegated to Loom shared credential store.
4. Manager refreshes tokens before expiry; failures mark `needs_auth`.
5. `logout` clears tokens and pending callbacks for that alias.

Constraints:
1. Never persist access/refresh tokens in plain `mcp.toml` by default.
2. Always propagate configured remote headers during auth and normal connect.
3. Keep callback map keyed by OAuth state only.

### 5) Registry Integration
Replace one-shot request pattern with manager-backed client calls:
1. Discovery reads capabilities from active connections.
2. `tools/list_changed` triggers scoped capability refresh.
3. Invocation uses persistent client handle where possible.
4. Auth-scoped visibility rules remain unchanged.

### 6) Protocol Surface Expansion
Add first-class handling for:
1. `tools`: existing behavior, now persistent and transport-agnostic.
2. `prompts`: discoverable and invocable via Loom command pipeline.
3. `resources`: discoverable/selectable/readable for context ingestion.

### 7) Reliability and Resource Controls
1. Add per-server circuit breaker:
   - Open after 5 consecutive failures within 60 seconds.
   - Half-open single probe after 30-second cooldown.
   - Close on successful probe.
2. Add bounded concurrency controls:
   - Global max concurrent MCP requests: 64.
   - Per-server in-flight limit: 8; bounded wait queue: 32.
   - Backpressure error when queue limit exceeded.
3. Add strict timeout hierarchy:
   - connect timeout
   - list/discovery timeout
   - call/read timeout
   - auth timeout
4. Add cancellation propagation:
   - User cancel or process shutdown must terminate in-flight requests and subprocess descendants.
5. Reconnect backoff:
   - Exponential backoff with jitter, base 1 second, max 30 seconds.

### 8) Security Controls
1. Remote URL validation:
   - `https` required by default.
   - Explicit `allow_insecure_http=true` required for local/dev override.
   - Reject loopback/private-network remote URLs unless explicitly allowed by config.
2. Header policy:
   - Redact sensitive headers in logs.
   - Block unsafe hop-by-hop headers from user config.
3. OAuth hardening:
   - Callback listener bound to loopback only.
   - Strict state nonce + expiration.
   - PKCE verifier never logged/persisted beyond flow lifetime.
4. Local command hardening for `type=local`:
   - Optional allowlist/trust gate for executable commands.
   - Explicit warning and confirmation before first enable for untrusted commands.

### 9) Observability and Diagnostics
1. Structured logs with stable error codes (`MCP_CONN_*`, `MCP_AUTH_*`, `MCP_PROTO_*`).
2. Metrics:
   - request count, latency, success/failure by alias + operation
   - reconnect attempts
   - circuit breaker state transitions
   - auth refresh outcomes
3. Low-cardinality status snapshots for CLI/TUI and telemetry export.
4. Debug bundle command:
   - `loom mcp debug-bundle` exporting redacted diagnostics for support.
   - Default retention 24 hours; explicit cleanup command available.

### 10) Compatibility and Versioning
1. Define supported MCP SDK/protocol range and pin minimum versions.
2. Add protocol feature negotiation:
   - degrade gracefully when prompts/resources unsupported.
3. Add compatibility tests against:
   - reference local stdio server
   - reference Streamable HTTP server
   - SSE-only compatibility server
4. Define manager ownership model:
   - one `MCPConnectionManager` per Loom process.
   - shared config writes guarded by filesystem lock.
   - no cross-process shared connection state in v2.

## Implementation Workstreams

### Workstream A: Schema and Config Foundation
Files:
1. `src/loom/config.py`
2. `src/loom/mcp/config.py`
3. `tests/test_mcp.py`
4. `tests/test_mcp_config_manager.py`

Deliverables:
1. New typed MCP server schema with strict validation.
2. Legacy normalization (`type=local` default).
3. Round-trip serialization with redaction-safe display.
4. URL/header validation rules and insecure override semantics.
5. Centralized timeout constants consumed by config, runtime, and docs.

Acceptance:
1. Existing stdio configs load unchanged.
2. Invalid cross-type fields fail with precise errors.
3. Timeout defaults are centralized and consistent across docs/runtime/tests.
4. Remote `http` endpoints are rejected unless explicit insecure override is set.
5. Sensitive header keys are redacted in all display paths.

### Workstream B: Persistent Runtime Core
Files:
1. `src/loom/integrations/mcp_tools.py` (refactor entrypoints)
2. `src/loom/integrations/mcp/manager.py` (new)
3. `src/loom/integrations/mcp/adapters/*.py` (new)
4. `tests/test_mcp_tools_bridge.py`

Deliverables:
1. `MCPConnectionManager` with status model and lifecycle APIs.
2. Persistent session pool and reconnect policy.
3. Capability cache with refresh on notification or reconnect.
4. Circuit breaker + bounded queue/in-flight controls.
5. Deterministic cancellation and shutdown semantics.

Acceptance:
1. Tool discovery and invocation work without per-call process spawn for active servers.
2. Connection drops recover within bounded retries.
3. Manager state is inspectable from CLI/TUI.
4. No unbounded retry loops or queue growth under repeated failures.
5. Cancel/shutdown terminates in-flight subprocess descendants reliably.

### Workstream C: OAuth Integration
Files:
1. `src/loom/auth/runtime.py`
2. `src/loom/integrations/mcp/oauth.py` (new)
3. `src/loom/__main__.py`
4. `tests/test_auth_config.py`
5. `tests/test_cli.py`

Deliverables:
1. `loom mcp auth login/status/logout/refresh`.
2. PKCE callback flow with strict state handling.
3. Refresh-before-expiry behavior and degraded-state signaling.
4. Callback timeout/cleanup and concurrent-login collision handling.
5. Token lifecycle audit events (redacted) for supportability.

Acceptance:
1. OAuth login succeeds for configured remote server alias.
2. Expired tokens auto-refresh or downgrade to `needs_auth` with clear remediation.
3. No plaintext token leakage in config output/logging.
4. Stale/pending callbacks are always garbage-collected after timeout/logout.
5. Auth flow handles repeated login/logout cycles without orphaned state.

### Workstream D: Registry + Surface Expansion
Files:
1. `src/loom/tools/registry.py`
2. `src/loom/integrations/mcp_tools.py`
3. `src/loom/session/prompt*.py` (or equivalent prompt pipeline modules)
4. `tests/test_mcp_tools_bridge.py`
5. New prompt/resource integration tests.

Deliverables:
1. Manager-backed discovery and invocation path.
2. MCP prompt and resource ingestion flows.
3. Auth-scoped isolation retained for all MCP capability types.
4. Capability payload size limits and schema validation for external data.

Acceptance:
1. Tools/prompts/resources are discoverable and callable/readable per alias.
2. Capability updates do not leak across auth profiles or sessions.
3. Existing non-MCP tool behavior remains unchanged.
4. Oversized/malformed capability payloads fail safely without crashing session flow.

### Workstream E: CLI/TUI Operator Experience
Files:
1. `src/loom/__main__.py`
2. `src/loom/tui/screens/mcp_manager.py`
3. `tests/test_cli.py`
4. `tests/test_tui.py`

Deliverables:
1. `mcp status` with per-alias state, transport, auth, and last error.
2. `mcp connect/disconnect/reconnect` actions.
3. TUI status views and auth action shortcuts.
4. Doc parity updates for supported transports and commands.
5. `mcp debug-bundle` with redacted diagnostics.
6. On-call runbook for top failure classes.

Acceptance:
1. Operators can recover broken auth/transport issues using CLI/TUI only.
2. No stale docs for unsupported options.
3. Debug bundle omits tokens/secrets and is sufficient for remote triage.

### Workstream F: Hardening and Regression Gates
Files:
1. MCP-focused test modules under `tests/`.
2. `README.md`
3. `docs/agent-integration.md`

Deliverables:
1. Failure-mode tests: timeouts, dropped connections, invalid tokens, header propagation.
2. Regression tests for callback-state correlation and timeout consistency.
3. Updated docs and migration examples.
4. Load tests for concurrent MCP calls and reconnect storms.
5. Fault-injection tests (network flaps, partial responses, auth endpoint failures).
6. Security tests for callback misuse and redaction guarantees.

Acceptance:
1. Test suite catches known classes of bugs seen in comparable systems.
2. Migration guidance is executable without source inspection.
3. Hardening test suite blocks release when safety invariants regress.

## Rollout Plan

### Phase 0: Design Lock and Safety Gates
1. Finalize config schema and state machine.
2. Add regression tests for timeout consistency and OAuth callback state handling.
3. Confirm no docs/runtime mismatch before feature rollout.
4. Complete threat-model review and security sign-off checklist.

Exit criteria:
1. Schema and state machine approved.
2. Guardrail tests green.
3. Security sign-off completed.

### Phase 1: Local Runtime Migration
1. Introduce manager with local stdio adapter only.
2. Keep behavior parity for current Loom MCP setups.
3. Gate behind `mcp.runtime_v2` flag.
4. Run canary in internal/dev environments.

Exit criteria:
1. Existing stdio workflows stable.
2. Performance improves by avoiding one-shot subprocess churn on active sessions.
3. No regression in CLI/TUI operator flows.

### Phase 2: Remote Transport and OAuth
1. Add remote adapters and OAuth commands.
2. Integrate token lifecycle and state transitions.
3. Add CLI/TUI status and remediation commands.
4. Ship with kill switch for remote runtime path.

Exit criteria:
1. At least one remote OAuth server works end to end.
2. Error states are explicit and recoverable from CLI/TUI.
3. Canary error budget is met for 7 consecutive days.

### Phase 3: Prompts/Resources and Full UX
1. Enable prompts/resources discovery and usage.
2. Finalize manager-backed registry integration for all capability types.
3. Remove feature flag and make v2 default.
4. Publish runbook + migration guide + compatibility matrix.

Exit criteria:
1. Tools/prompts/resources all pass integration tests.
2. Operational UX is complete and documented.
3. Rollback drill from v2 to v1 path succeeds without config loss.

## Success Metrics
1. Remote MCP onboarding time (config + auth + first call) under 5 minutes for a new user.
2. P95 MCP tool call latency improved for repeated calls vs one-shot baseline.
3. Zero known token leaks in logs/config outputs.
4. Fewer manual recovery steps for auth/transport failures (measured by CLI action sequence length).
5. No regressions in auth-scoped tool visibility isolation.
6. Canary weekly error budget burn remains below threshold before GA.

## Risk Register
1. Risk: runtime complexity introduces flaky reconnect behavior.
   - Mitigation: explicit state machine + deterministic backoff tests.
2. Risk: mixed capability support (tools/prompts/resources) increases registry coupling.
   - Mitigation: capability-specific adapters behind a shared manager interface.
3. Risk: OAuth edge cases create stuck auth sessions.
   - Mitigation: strict state correlation, alias cleanup hooks, and timeout-based callback expiry.
4. Risk: transport fallback hides root cause.
   - Mitigation: structured reason codes and operator-visible diagnostics.
5. Risk: multi-process execution causes duplicated connection churn or stale status views.
   - Mitigation: define manager ownership model per process and add config/status file locking where shared writes occur.
6. Risk: high-cardinality metrics/logs increase cost and obscure signals.
   - Mitigation: enforce alias-level cardinality caps and standardized error codes.

## Production Go/No-Go Checklist
1. Functional:
   - Local stdio MCP behavior matches pre-refactor behavior across existing tests.
   - Remote OAuth server passes end-to-end auth, discovery, and invocation flow.
2. Security:
   - Redaction tests pass for headers, tokens, and OAuth artifacts.
   - Callback misuse tests pass (invalid state, replayed state, expired state, wrong alias).
   - `https` enforcement and insecure override behavior verified.
3. Reliability:
   - Circuit breaker and bounded queue tests pass.
   - Reconnect storm and network flap tests pass within error budget.
   - Cancellation/shutdown leaves no orphan subprocesses.
4. Operability:
   - `mcp status`, `connect/disconnect/reconnect`, and `debug-bundle` verified in CLI and TUI.
   - Runbook reviewed by maintainers and validated in a simulated incident.
5. Rollback:
   - Feature flag rollback tested in staging with no data/config corruption.
   - Migration compatibility verified both forward and backward.

## Decisions Locked for Implementation
1. SSE fallback is opt-in by config (or explicit negotiated fallback), not default-on.
2. Prompts/resources ship after remote transport + OAuth stabilization (Phase 3).
3. `mcp.runtime_v2` remains user-configurable during rollout and is removed after GA stabilization window.
4. `https` is required for remote servers unless explicit insecure override is set.

## Recommended Execution Order
1. Workstream A
2. Workstream B
3. Workstream C
4. Workstream E
5. Workstream D
6. Workstream F

Rationale:
1. Lock schema and runtime core first.
2. Add OAuth and operator controls before widening protocol surface.
3. Expand to prompts/resources only after connection/auth stability is proven.
