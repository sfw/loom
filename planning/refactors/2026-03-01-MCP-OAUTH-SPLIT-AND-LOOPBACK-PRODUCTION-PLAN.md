# MCP OAuth Split + Loopback Hardening Plan (2026-03-01)

## Decision Summary
This plan locks three architecture decisions:
1. MCP OAuth credential storage remains separate from `/auth`, with one active OAuth token set per MCP alias.
2. `/auth` remains the canonical system for API and non-MCP credentials (and can still support non-MCP OAuth profiles).
3. OAuth browser callback support is implemented as a shared loopback engine reusable by both MCP auth and `/auth` OAuth flows.

This aligns Loom with OpenCode's practical MCP model while preserving Loom's broader auth model.

## Why This Plan Exists
The current implementation has remote MCP OAuth token import/update flows, but production-grade browser auth is incomplete:
1. MCP OAuth is currently manual token import in CLI/TUI (`mcp auth login --access-token` or paste in TUI).
2. No first-class local loopback callback server for `authorization_code + PKCE` login.
3. Runtime has split responsibilities between MCP token storage and `/auth`, but the boundary is implicit and under-documented.
4. Operator guidance still assumes manual remediation for auth gaps.

## Critique of Prior Unified MCP Plan
Reference: `planning/refactors/2026-03-01-UNIFIED-MCP-MANAGEMENT-SYSTEM-REFRACTOR-PLAN.md`

What is strong:
1. Strong reliability controls (queue bounds, circuit breaker, timeout hierarchy).
2. Good runtime architecture and state model.
3. Strong rollout and production gate framing.

What needs correction before implementation:
1. It says "token storage delegated to Loom shared credential store"; this conflicts with the now-locked decision to keep MCP OAuth token storage alias-scoped and separate from `/auth`.
2. It treats OAuth as MCP-only; we need a shared OAuth engine to avoid duplicate flows when `/auth` OAuth providers are added.
3. It does not define migration from today's manual MCP token import UX to browser-first login with device/manual fallback.
4. It lacks an explicit threat model for callback binding, state replay, port conflicts, and concurrent login cancellation behavior.
5. It does not specify a deterministic contract for multi-process behavior of the loopback server and pending login state.

This hardened plan addresses those gaps directly.

## Scope
In scope:
1. Production-grade shared OAuth engine (PKCE + loopback callback + state lifecycle + timeout/cancel).
2. MCP OAuth browser login/logout/status/refresh using one token set per alias.
3. TUI MCP manager rebuild for browser-first OAuth flows and clear auth state.
4. Shared OAuth engine adapter for `/auth` OAuth modes (without migrating MCP tokens into `/auth`).
5. Docs, migration guidance, and comprehensive tests/lint gates.

Out of scope:
1. Multiple active OAuth token sets per single MCP alias.
2. Replacing existing `/auth` resource/binding selection model.
3. Major MCP prompts/resources redesign unrelated to OAuth lifecycle.

## Architecture Contract (Locked)

### 1) Credential Ownership
1. MCP alias credentials:
   - Stored in MCP OAuth store (`~/.loom/mcp_oauth_tokens.json`), keyed by alias.
   - One active credential set per alias.
   - Accessed only through MCP OAuth access APIs.
2. `/auth` credentials:
   - Stored/resolved by existing `/auth` profile + secret reference model.
   - Used by non-MCP APIs/tools and optional OAuth profiles outside MCP alias auth.

### 2) Shared OAuth Engine
Single engine module provides:
1. PKCE verifier/challenge generation.
2. State nonce generation, persistence hooks, and expiry validation.
3. Loopback callback listener bound to `127.0.0.1` only.
4. Browser open + manual URL fallback support.
5. Callback wait/cancel timeout handling.
6. OAuth error normalization and redact-safe diagnostics.

Engine consumers:
1. MCP OAuth login flow.
2. `/auth` OAuth profile login flow (future or incremental in this plan).

### 3) Transport and Token Use
1. Remote MCP requests continue to inject `Authorization` from alias token store if present.
2. If token expired/missing:
   - Manager transitions alias state to `needs_auth`.
   - CLI/TUI remediation points to browser login command/action.
3. Refresh behavior:
   - Attempt refresh when refresh token exists and token expiry is known.
   - On refresh failure, downgrade to `needs_auth` with actionable reason.

### 4) Multi-Account Semantics
1. Multiple accounts for same provider are represented as multiple MCP aliases.
2. One alias = one active OAuth token set.
3. No per-alias account switching selector in v1 of this plan.

## Implementation Workstreams

### Workstream A: Shared OAuth Engine Foundation
Files:
1. `src/loom/oauth/engine.py` (new)
2. `src/loom/oauth/loopback.py` (new)
3. `src/loom/oauth/state_store.py` (new)
4. `tests/test_oauth_engine.py` (new)
5. `tests/test_oauth_loopback.py` (new)

Deliverables:
1. Engine API for `start_auth`, `await_callback`, `finish_auth`, `cancel_auth`.
2. Loopback server lifecycle with single-process ownership and port conflict handling.
3. State map keyed by OAuth state only (never by alias).
4. Strict timeout and explicit cancellation semantics.
5. Redacted structured errors with stable reason codes.

Acceptance:
1. Callback accepts only known state, rejects missing/expired/replayed state.
2. Port busy path degrades gracefully with manual callback code option.
3. No verifier/state/token leakage in logs or exceptions.
4. Concurrent login attempts are isolated and deterministic.

### Workstream B: MCP OAuth Browser Flow (Single Alias Token)
Files:
1. `src/loom/integrations/mcp/oauth.py`
2. `src/loom/integrations/mcp_tools.py`
3. `src/loom/__main__.py`
4. `tests/test_cli.py`
5. `tests/test_mcp_tools_bridge.py`

Deliverables:
1. `loom mcp auth login <alias>` defaults to browser PKCE flow.
2. `--manual-token` compatibility path retained for controlled fallback.
3. `loom mcp auth status/logout/refresh` aligned to alias token store contract.
4. Runtime remediation messages updated to browser-first commands.
5. Robust token refresh and expiry handling.

Acceptance:
1. Browser auth completes end-to-end for remote OAuth-enabled alias.
2. Manual-token flow still works for headless environments.
3. Expired token + failed refresh leads to deterministic `needs_auth`.
4. Alias token store remains backwards-compatible with current JSON shape (additive fields only).

### Workstream C: TUI MCP Manager Rebuild for OAuth UX
Files:
1. `src/loom/tui/screens/mcp_manager.py`
2. `src/loom/tui/app.py` (if command routing/helpers needed)
3. `tests/test_tui.py`

Deliverables:
1. Replace "paste token first" UX with:
   - `OAuth Login (Browser)`
   - `Copy Login URL` fallback
   - `Enter Callback Code` fallback when auto-callback is unavailable
   - `OAuth Logout`
2. Real-time alias OAuth status panel with expiry and last failure reason.
3. Clear environment guidance for SSH/headless cases.
4. Keep legacy token import action as explicit advanced fallback, not default.

Acceptance:
1. New user can authenticate remote MCP via TUI without leaving management flow.
2. TUI never displays token values.
3. Status always converges after login/logout/refresh or timeout.

### Workstream D: `/auth` Integration via Shared Engine (No MCP Token Migration)
Files:
1. `src/loom/auth/runtime.py`
2. `src/loom/auth/config.py`
3. `src/loom/tui/screens/auth_manager.py`
4. `src/loom/__main__.py`
5. `tests/test_auth_config.py`

Deliverables:
1. Shared OAuth engine hooks available to `/auth` OAuth profile lifecycle commands.
2. `/auth` retains token_ref contract for profile-scoped OAuth tokens.
3. Explicit docs and validation ensure operators understand MCP tokens and `/auth` tokens are separate by design.
4. Guardrails preventing accidental cross-system token writes.

Acceptance:
1. `/auth` OAuth preflight remains stable and compatible.
2. Shared engine is used by both MCP and `/auth` flows without coupling storage.
3. No regression in existing auth resource resolution/binding behavior.

### Workstream E: Security, Reliability, and Operational Hardening
Files:
1. `tests/test_security_redaction.py` (new or extend)
2. `tests/test_cli.py`
3. `tests/test_tui.py`
4. `docs/CONFIG.md`
5. `docs/tutorial.html`
6. `docs/agent-integration.md`
7. `README.md`

Deliverables:
1. Threat-model-driven tests:
   - state replay
   - callback forgery
   - timeout races
   - concurrent login cancellation
   - port collision
2. Redaction tests across CLI/TUI/logs for auth artifacts.
3. Operator runbook for auth failures and rollback playbook.
4. Migration docs from manual token import to browser flow.

Acceptance:
1. Security test suite blocks release on callback/state/redaction regressions.
2. Docs are complete and consistent with runtime behavior.
3. On-call can resolve top auth incidents using documented steps only.

## Hardening Requirements (Non-Negotiable)
1. Callback listener must bind only to `127.0.0.1` (never wildcard).
2. State nonce must be high entropy, single-use, and expiration-bound.
3. Pending auth entries must always be cleared on success, timeout, cancel, logout, and shutdown.
4. All OAuth errors must be redact-safe.
5. Port conflict behavior must not deadlock login UX.
6. Runtime must fail closed (`needs_auth`) when token validity cannot be proven.
7. Token refresh must be rate-limited to avoid refresh storms.
8. Multi-process safety: lock-protected token store updates and deterministic last-writer semantics.

## Testing Plan

### Unit
1. PKCE/state generation and verification.
2. Loopback callback parsing and validation.
3. Token store read/write, lock, and corruption handling.
4. Error redaction and stable reason-code mapping.

### Integration
1. CLI `mcp auth login/status/logout/refresh` full cycle.
2. TUI browser login + fallback flows.
3. MCP runtime auth gating and remediation under expired/missing token states.
4. `/auth` OAuth profile checks using shared engine hooks.

### Fault Injection
1. Callback timeout.
2. Browser open failure.
3. Port already in use.
4. Invalid callback state.
5. Token refresh endpoint failure.
6. Concurrent login for same alias.

### Regression
1. Existing local stdio MCP behavior unchanged.
2. Existing manual token import commands still usable.
3. Existing `/auth` API key/env passthrough flows unaffected.

## Rollout Strategy
1. Phase 0: Land shared engine + tests behind feature flag `mcp.oauth_browser_login`.
2. Phase 1: Enable browser login in CLI with fallback path; canary on dev workspaces.
3. Phase 2: Enable TUI rebuilt OAuth manager.
4. Phase 3: Enable shared engine for `/auth` OAuth profile commands.
5. Phase 4: Make browser login default; keep manual token import as documented fallback.

Rollback:
1. Disable `mcp.oauth_browser_login` and revert to manual token import path.
2. Keep token store format backward-compatible so no credential loss on rollback.

## Risks and Mitigations
1. Risk: Loopback callback unavailable in remote/headless sessions.
   - Mitigation: first-class manual URL + callback code fallback path.
2. Risk: Engine reuse introduces coupling bugs between MCP and `/auth`.
   - Mitigation: strict interface boundaries with separate storage adapters.
3. Risk: Token refresh behavior causes flaky runtime states.
   - Mitigation: explicit refresh policy + cooldown + deterministic state transitions.
4. Risk: Operator confusion over MCP store vs `/auth`.
   - Mitigation: explicit UX copy, docs matrix, and command help examples.
5. Risk: Parallel login attempts produce stale pending state.
   - Mitigation: state-keyed pending map, alias-level mutex, guaranteed cleanup hooks.

## Definition of Done (Production Ready)
1. Browser-first OAuth login works in CLI and TUI for at least one real remote MCP provider.
2. Manual fallback works when browser callback cannot be used.
3. No plaintext token leakage in logs, status output, debug bundle, or docs examples.
4. MCP alias token contract remains one-token-per-alias and backward-compatible.
5. `/auth` remains fully functional for non-MCP credentials and uses shared OAuth engine safely.
6. Full test suite + lint pass in CI with new OAuth hardening tests included.
7. Runbook and migration docs are published and validated in a dry-run incident exercise.

## Execution Order
1. Workstream A (shared engine)
2. Workstream B (MCP CLI/runtime flow)
3. Workstream C (TUI rebuild)
4. Workstream D (`/auth` shared-engine adoption)
5. Workstream E (hardening/docs/final gates)

This ordering reduces delivery risk by proving engine and MCP paths first, then layering UX and broader auth integration.
