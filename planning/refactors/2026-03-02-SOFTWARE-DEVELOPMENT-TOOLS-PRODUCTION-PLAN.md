# Software Development Tools Production Plan (2026-03-02)

## Objective
Define and harden a production-ready implementation plan for a new software development tool suite focused on:
1. Coding-agent interoperability (`codex`, `claude code`, `opencode`).
2. WordPress development workflows (`wp-cli` and adjacent ecosystem tools).

This is a planning artifact only. No runtime/code changes are included here.

## Scope
In scope:
1. New Loom tools for external coding-agent orchestration.
2. New Loom tools for WordPress development flows.
3. Safety, approval, and telemetry model for these tools.
4. Test, rollout, and rollback strategy.

Out of scope:
1. Replacing Loom's existing generic coding tools (`shell_execute`, `git_command`, `ripgrep_search`, etc.).
2. Implementing remote cloud WordPress hosting/deployment providers.
3. Arbitrary plugin marketplace management automation beyond local development workflow support.

## External Research Snapshot (Primary Sources)
Key upstream capabilities validated against current docs:

1. Codex CLI supports non-interactive execution, JSON output, sandbox modes, approval policies, and MCP operations.
   Source: <https://developers.openai.com/codex/noninteractive>, <https://developers.openai.com/codex/cli/command-line-options>, <https://developers.openai.com/codex/mcp>
2. Claude Code documents permission modes and headless usage patterns.
   Source: <https://docs.claude.com/en/docs/claude-code/team>, <https://docs.claude.com/en/docs/claude-code/settings>, <https://docs.claude.com/en/docs/claude-code/sdk/sdk-headless>
3. OpenCode documents explicit permission controls, command allow/deny policy, and agent usage.
   Source: <https://opencode.ai/docs/permissions>, <https://opencode.ai/docs/cli>, <https://opencode.ai/docs/agents>
4. WordPress CLI command inventory and local development packages (`wp-env`, `create-block`, `@wordpress/scripts`) are maintained and suitable for standardized wrappers.
   Source: <https://developer.wordpress.org/cli/commands/>, <https://developer.wordpress.org/block-editor/reference-guides/packages/packages-env/>, <https://developer.wordpress.org/block-editor/reference-guides/packages/packages-create-block/>, <https://developer.wordpress.org/block-editor/reference-guides/packages/packages-scripts/>
5. WordPress code quality ecosystem has standardized checks via WPCS/PHPCS and Plugin Check.
   Source: <https://github.com/WordPress/WordPress-Coding-Standards>, <https://wordpress.org/plugins/plugin-check/>

## Current Loom Baseline
Already present in Loom:
1. Strong generic tooling substrate (`shell_execute`, `git_command`, file tools, ripgrep/glob).
2. Tool registry auto-discovery and schema-based dispatch.
3. Approval/safety controls and mutating tool pathways.
4. Hybrid tool discovery (`list_tools`, `run_tool`) and cowork execution integration.

Gap:
1. No first-class wrappers for external coding-agent CLIs.
2. No first-class WordPress workflow tools.
3. No domain-specific policy layer for high-risk WordPress DB and search-replace operations.

## Draft Proposal (V0)

### Proposed Tool Set

#### A) Agent Interop
1. `agent_capabilities`
   - Detect installed agent CLIs (`codex`, `claude`, `opencode`), versions, and supported flags.
   - Return normalized capability matrix and machine-readable incompatibility reasons.
2. `agent_run`
   - Run one agent command in structured mode with normalized inputs:
     - `provider`: `codex | claude_code | opencode`
     - `prompt`, `cwd`, `timeout_seconds`, `sandbox_mode`, `approval_mode`, `output_mode`
   - Normalize result into `stdout`, `stderr`, `exit_code`, `events`, `artifact_paths`.
3. `agent_session`
   - Optional follow-up operations where supported: `resume`, `status`, `stop`.

#### B) WordPress Workflow
1. `wp_cli`
   - Structured wrapper around common `wp` command groups:
     - `core`, `plugin`, `theme`, `option`, `user`, `db`, `search-replace`, `post`.
2. `wp_env`
   - Standardized local environment lifecycle wrapper for `@wordpress/env`:
     - `start`, `stop`, `destroy`, `run`, `logs`.
3. `wp_scaffold_block`
   - Wrapper around `@wordpress/create-block` for scaffolding and optional `--wp-env` startup.
4. `wp_quality_gate`
   - Unified quality checks for WordPress repos:
     - WPCS/PHPCS checks,
     - Plugin Check execution,
     - optional `@wordpress/scripts` lint/test commands.

### Why V0 Is Attractive
1. High leverage: converts fragile shell prompt composition into typed contracts.
2. Reuses Loom safety model and approval gating.
3. Improves interoperability with major coding-agent workflows plus WordPress development.

## Critique Of V0

### Critical weaknesses
1. Capability drift risk:
   - CLI flags and behaviors differ by version; V0 does not require robust version gates.
2. Safety ambiguity for WordPress:
   - `wp db` and `wp search-replace` can be destructive; V0 lacks explicit policy tiers.
3. Tool sprawl risk:
   - Too many wrappers can duplicate existing `shell_execute` capabilities without clear boundaries.
4. Session semantics mismatch:
   - `agent_session` assumes all providers expose stable resume/status semantics; they may not.
5. Output normalization mismatch:
   - Agent CLIs emit different output formats; V0 lacks strict parser contracts and fallback behavior.

### Operational weaknesses
1. No deterministic rollout gates.
2. No kill-switch flags per tool family.
3. No compatibility tests against missing binaries or stale versions.
4. No golden fixture strategy for parsing CLI outputs.

## Hardened Production Plan (V1)

## Product Decisions
1. Implement in two stages:
   - Stage A (MVP hardening): `agent_capabilities`, `agent_run`, `wp_cli`.
   - Stage B (expansion): `wp_env`, `wp_scaffold_block`, `wp_quality_gate`.
2. Defer `agent_session` to post-MVP until real provider semantics are stable in practice.
3. Keep wrappers narrow and typed; route unsupported/edge behavior through existing `shell_execute`.
4. Ship with opt-in feature flags per family for safe rollout.

## Hardening Deltas Applied (V0 -> V1)
1. Added explicit staged rollout with feature flags and rollback path.
2. Added deterministic safety tiers for WordPress operations.
3. Deferred unstable `agent_session` semantics until post-MVP.
4. Added compatibility/version-gate requirement before execution.
5. Added parser fixture strategy and structured error-code contract.

## Canonical Mode Mapping Contract
To avoid provider-specific behavior drift, Loom-facing modes are canonicalized and translated by provider adapters.

Canonical `sandbox_mode`:
1. `read_only`
2. `workspace_write`
3. `unrestricted`

Canonical `network_mode`:
1. `off`
2. `on`

Canonical `approval_mode`:
1. `untrusted`
2. `on_failure`
3. `on_request`
4. `never`

Mapping policy:
1. Provider adapters must map canonical enums to native CLI flags.
2. If no exact mapping exists for installed provider version, fail closed with `unsupported_mode_combination`.
3. `agent_capabilities` is the source of truth for supported combinations.
4. `agent_run` must reject combinations not reported by `agent_capabilities`.
5. Default for `agent_run` is `network_mode=on`, with global config override support.
6. If `network_mode=off` is requested (per call or config default) and provider cannot reliably enforce it, `agent_run` must fail closed for that mode.

## Tool Contracts (V1)

### 1) `agent_capabilities` (P0)
Purpose:
1. Discover local availability and compatibility before execution.

Input:
1. `providers` (optional list; default all supported providers).
2. `refresh` (optional bool).

Output:
1. Per-provider `installed`, `binary_path`, `version`, `supports` map.
2. `unsupported_reasons` list with deterministic codes.
3. `recommended_defaults` for `sandbox_mode` and `approval_mode` per provider.

Safety:
1. Read-only process execution only (`<cli> --version` / help queries).

### 2) `agent_run` (P0)
Purpose:
1. Unified non-interactive run for supported coding-agent CLIs.

Input:
1. `provider` (required).
2. `prompt` (required).
3. `cwd` (optional, workspace-bound).
4. `sandbox_mode` (enum mapped per provider).
5. `network_mode` (`off | on`, default `on` unless config overrides).
6. `approval_mode` (enum mapped per provider).
7. `timeout_seconds` (bounded; default 300, max 1800).
8. `output_mode` (`text | json | stream`).
9. `args` (optional bounded passthrough allowlist).

Output:
1. `success`, `exit_code`, `stdout`, `stderr`.
2. `parsed_payload` (when provider returns structured JSON).
3. `provider_command` (redacted), `duration_ms`.

Safety:
1. Allowlisted command templates per provider; no raw shell interpolation.
2. Workspace path enforcement for `cwd`.
3. Approval parity with mutating tool policy when provider is run in permissive modes.
4. Default egress policy is network-enabled, but `network_mode=off` must be enforceable when requested by config or call.
5. Provider runs must use a constrained environment allowlist (avoid inheriting broad host secrets/config by default).

Parser contract:
1. Provider-specific parser must produce stable normalized payload shape.
2. If parse fails, return raw output plus deterministic `output_parse_error` code.
3. Parsing must be version-gated by `agent_capabilities`.

### 3) `wp_cli` (P0)
Purpose:
1. Typed WordPress development operations using `wp` without freeform shell composition.

Input:
1. `group` (enum: `core | plugin | theme | option | user | post | db | search_replace`).
2. `action` (group-specific enum).
3. `args` (typed object per action).
4. `path` (optional WP install path, workspace-bound).

Output:
1. `success`, `exit_code`, `stdout`, `stderr`.
2. `parsed` structured fields for known commands.

Safety tiers:
1. Tier 0: read-only commands (no approval beyond standard).
2. Tier 1: state-changing commands (mutating approval path).
3. Tier 2: high-risk commands (`db reset/drop`, broad `search-replace`) require explicit additional confirmation policy.

Tier-2 defaults:
1. `search_replace` defaults to dry-run behavior.
2. Destructive DB operations require explicit `confirm_high_risk=true`.
3. Broad target scopes (site-wide table rewrites) require additional confirmation payload.

### 4) `wp_env` (P1)
Purpose:
1. Managed wrapper for local `@wordpress/env` lifecycle and command execution.

Input:
1. `operation` (`start | stop | destroy | run | logs`).
2. `cwd` (workspace-bound project root).
3. `run_args` (for `run`, allowlisted service targets).

Safety:
1. `destroy` requires high-risk confirmation.
2. Container operations scoped to project path.

### 5) `wp_scaffold_block` (P1)
Purpose:
1. Safe, repeatable block scaffolding via `@wordpress/create-block`.

Input:
1. `name`, `variant`, `namespace`, `target_dir`, `with_wp_env`.

Safety:
1. Path confinement and overwrite guards.

### 6) `wp_quality_gate` (P1)
Purpose:
1. One command surface for WordPress QA checks.

Input:
1. `checks` (`wpcs`, `plugin_check`, `wp_scripts_lint`, `wp_scripts_test`).
2. `cwd`, `fail_fast`, `report_format`.

Output:
1. Structured check summaries and failure inventory.

## Architecture and File Plan

### New files
1. `/Users/sfw/Development/loom/src/loom/tools/agent_capabilities.py`
2. `/Users/sfw/Development/loom/src/loom/tools/agent_run.py`
3. `/Users/sfw/Development/loom/src/loom/tools/wp_cli.py`
4. `/Users/sfw/Development/loom/src/loom/tools/wp_env.py` (Stage B)
5. `/Users/sfw/Development/loom/src/loom/tools/wp_scaffold_block.py` (Stage B)
6. `/Users/sfw/Development/loom/src/loom/tools/wp_quality_gate.py` (Stage B)
7. `/Users/sfw/Development/loom/src/loom/tools/tooling_common/command_runner.py` (shared allowlisted subprocess helper)
8. `/Users/sfw/Development/loom/src/loom/tools/tooling_common/version_matrix.py` (provider/version compatibility)

### Existing files to update
1. `/Users/sfw/Development/loom/src/loom/config.py`
   - Add feature flags and policy settings:
     - `execution.enable_agent_tools`
     - `execution.enable_wp_tools`
     - `execution.wp_high_risk_requires_confirmation`
     - `execution.agent_tools_allowed_providers`
     - `execution.agent_tools_max_timeout_seconds`
     - `execution.agent_tools_default_network_mode` (`on` by default, can be set to `off`)
2. `/Users/sfw/Development/loom/docs/CONFIG.md`
   - Document new config keys.
3. `/Users/sfw/Development/loom/README.md`
   - Document new tool capabilities and limitations.
4. `/Users/sfw/Development/loom/tests/test_new_tools.py`
   - Extend or split into dedicated test modules for new tools.

## Safety and Policy Hardening

### Command policy model
1. No raw command strings in public tool schemas.
2. Provider-specific command builders generate argv arrays only.
3. Strict allowlist for optional passthrough args.
4. Output size caps and truncation policy parity with `shell_execute`.
5. Timeout hard caps with deterministic timeout errors.
6. `shell_execute` must route WordPress destructive command patterns through the same high-risk approval path (no policy bypass).

### WordPress risk policy
1. Explicit high-risk operation registry (`wp_cli`):
   - `db drop`, `db reset`, wide-scope `search-replace`, destructive plugin/theme deletes.
2. Mandatory confirmation path for high-risk operations even when global approval is permissive.
3. Optional dry-run default for `search-replace` unless `confirm_apply=true`.

### High-Risk Confirmation UX Contract
When the existing approval modal is shown for destructive patterns, require clear risk framing:
1. Show risk level and action class (for example: `HIGH RISK: destructive database operation`).
2. Show concrete impact preview (target tables/site scope/object counts when available).
3. Show explicit consequences (data loss potential, rollback prerequisites).
4. Require an explicit confirm action tied to the exact operation payload.
5. For non-interactive contexts (API/autonomous), fail closed with `high_risk_confirmation_required` when confirmation cannot be obtained.

### Security considerations
1. Redact secrets from emitted command lines and stderr/stdout where predictable.
2. Do not capture environment dumps by default.
3. Deny execution outside workspace/read-root policy.

### Deterministic error codes
All new tools must emit stable machine-readable error codes in `ToolResult.data.error_code`:
1. `binary_not_found`
2. `unsupported_version`
3. `unsupported_mode_combination`
4. `path_outside_workspace`
5. `high_risk_confirmation_required`
6. `timeout_exceeded`
7. `output_parse_error`
8. `feature_disabled`
9. `network_disabled`

## Testing Strategy

### Unit tests
1. Capability parser tests per provider with fixture-based outputs.
2. Command builder tests validating argv generation and blocked arguments.
3. Safety tier tests for `wp_cli` operation classification.
4. Timeout and output truncation behavior tests.

### Integration tests (mocked subprocess)
1. `agent_run` end-to-end normalization for each provider.
2. `wp_cli` common command success/failure paths.
3. Feature flag gating behavior when disabled.
4. Deterministic error-code assertions for each blocked/failure path.

### Optional live tests (guarded)
1. Run when binaries are present in CI matrix or developer env.
2. Skip by default to avoid flaky dependency failures.

## Telemetry and Observability
1. Add events:
   - `agent_capabilities_checked`
   - `agent_run_started`
   - `agent_run_completed`
   - `wp_cli_invoked`
   - `wp_cli_high_risk_blocked`
2. Include fields:
   - `tool_name`, `provider`, `operation`, `duration_ms`, `exit_code`, `timed_out`, `truncated`.
3. Do not log sensitive prompt text by default in telemetry payloads.

## Rollout Plan

### Phase R1: Internal MVP (feature flags off by default)
1. Implement Stage A tools.
2. Land unit + mocked integration tests.
3. Verify no behavior changes when flags disabled.

Exit criteria:
1. Tool registry stable with flags off.
2. Test suite green.

### Phase R2: Opt-in dogfood
1. Enable `enable_agent_tools=true` for local dev and controlled sessions.
2. Gather telemetry on failure modes and version drift.
3. Harden parsers and compatibility matrix.

Exit criteria:
1. >=95% successful command normalization in dogfood runs.
2. No safety regressions.

### Phase R3: WordPress expansion
1. Add Stage B tools (`wp_env`, `wp_scaffold_block`, `wp_quality_gate`).
2. Keep `enable_wp_tools` default false until telemetry stabilizes.

Exit criteria:
1. High-risk WP operations always trigger expected policy path.
2. Structured outputs stable across supported environments.

### Phase R4: Default-on
1. Flip defaults after stability window and docs completion.
2. Preserve quick rollback via feature flags.

Preconditions for default-on:
1. Zero unresolved high-severity safety bugs from dogfood period.
2. Parsing success rate >=99% for supported provider/version combinations.
3. No increase in destructive-operation incidents versus baseline.

## Backward Compatibility
1. Existing workflows remain intact via current generic tools.
2. New wrappers are additive and optional.
3. Fallback guidance in tool errors points to `shell_execute` for unsupported edge cases.

## Risks and Mitigations
1. Risk: upstream CLI changes break parsers.
   - Mitigation: version gates + fixture updates + fail-closed unsupported-mode errors.
2. Risk: wrappers duplicate existing tools and confuse model behavior.
   - Mitigation: narrow contracts, clear descriptions, and intent-based selection guidance.
3. Risk: WordPress destructive operations executed accidentally.
   - Mitigation: high-risk tier policy + dry-run defaults + explicit confirmations.
4. Risk: cross-platform process differences.
   - Mitigation: normalize only portable subset first; broaden after telemetry.

## Production Readiness Checklist
1. Feature flags implemented and documented.
2. Capability/version gates implemented for all providers.
3. High-risk WP policy enforced with tests.
4. Deterministic timeout/truncation semantics covered by tests.
5. Telemetry events emitted and privacy-reviewed.
6. Rollback path validated (flags off restores baseline).
7. Operator docs include troubleshooting matrix for missing binaries and unsupported versions.

## Operational Runbook (Post-Launch)
1. If provider breakage occurs after CLI upgrade:
   - disable affected provider in `agent_tools_allowed_providers`,
   - mark unsupported version in compatibility matrix,
   - ship parser compatibility patch.
2. If WordPress high-risk false positive/negative is detected:
   - toggle `wp_high_risk_requires_confirmation=true`,
   - patch operation classifier,
   - add regression test fixture before re-enable.
3. If tool causes reliability regression:
   - disable feature flags,
   - retain generic fallback (`shell_execute`) path for continuity.

## Definition of Done
1. Stage A tools ship behind flags with full test coverage and safety gates.
2. Stage B ships with WordPress-specific risk policies and quality checks.
3. Plan exits with default-on readiness plus immediate rollback capability.
4. No regressions to existing Loom tool behavior with features disabled.

## Implementation Sequencing Summary
1. Build Stage A minimal, safe, typed wrappers first.
2. Run critique-driven hardening gates before Stage B.
3. Expand to WordPress lifecycle and quality tools after telemetry validates Stage A assumptions.
4. Promote to default-on only after reliability and safety thresholds are met.
