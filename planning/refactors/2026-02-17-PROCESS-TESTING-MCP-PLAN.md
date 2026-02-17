# Process Quality, Package Tests, and MCP Tooling Plan (2026-02-17)

## Execution Status (2026-02-17)
Status: Implemented (core phases completed)

Implemented highlights:
- Pytest taxonomy + strict markers + deterministic process contract suite.
- Package-level `tests:` manifest schema + `loom process test` runner.
- Built-in process manifests now include `smoke` and `live-canary` cases.
- Live canary test module (`process_live`) and nightly/manual artifact workflow.
- MCP tool bridge with namespaced registration (`mcp.<server>.<tool>`) from config.
- Docs/changelog updated and full lint/test suite passing.

Status update:
- Runtime MCP tool-set refresh is now wired through a registry refresh hook.
- Existing in-memory registries reconcile MCP add/remove tool changes on
  interval and on-demand (including unknown MCP tool execution fallback).

This plan covers three linked goals:
1. Build reliable process-level testing (deterministic + real-world).
2. Add first-class test metadata to process packages.
3. Integrate MCP-backed tools into process execution safely.

## Scope
Code areas this plan targets:
- `/Users/sfw/Development/loom/src/loom/processes/schema.py`
- `/Users/sfw/Development/loom/src/loom/processes/installer.py`
- `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- `/Users/sfw/Development/loom/src/loom/engine/verification.py`
- `/Users/sfw/Development/loom/src/loom/tools/__init__.py`
- `/Users/sfw/Development/loom/src/loom/tools/registry.py`
- `/Users/sfw/Development/loom/src/loom/tools/web.py`
- `/Users/sfw/Development/loom/src/loom/tools/web_search.py`
- `/Users/sfw/Development/loom/src/loom/processes/builtin/*.yaml`
- `/Users/sfw/Development/loom/tests/`
- `/Users/sfw/Development/loom/.github/workflows/`

## Research Findings (External)

### 1) Test suite structure and stability
- Pytest supports custom markers and marker-based selection (`-m`), and recommends registering markers to avoid unknown-marker drift.
- `--strict-markers` turns marker typos into hard errors.
- Pytest’s flaky guidance explicitly recommends splitting higher-level/integration tests from core unit gates.
- `monkeypatch` is the right primitive for deterministic isolation of network/env/global state in tests.

Implication for Loom: maintain a strict, fast deterministic gate for PRs and move network/model-dependent process cases into separately marked suites.

### 2) CI scheduling and observability
- GitHub scheduled workflows can be delayed or dropped at high-load times, run only from the default branch, and should avoid top-of-hour cron slots.
- Minimum schedule interval is 5 minutes.
- Artifacts are the native mechanism to persist logs/reports for debugging and trend tracking.

Implication for Loom: nightly process canaries should run off-hour, default-branch, and always upload machine-readable reports (junit/log/json).

### 3) MCP tool lifecycle and safety
- MCP tool discovery is explicit (`tools/list`, `tools/call`).
- MCP supports runtime tool-list changes (`notifications/tools/list_changed`) when `listChanged` capability is declared.
- MCP guidance emphasizes human-in-the-loop for tool invocation and confirmations for operations.
- MCP tool metadata includes `inputSchema` and optional `outputSchema` and tool-name guidance.

Implication for Loom: MCP-backed tools should be registered dynamically, namespaced, validated against schemas, and integrated with existing approval flows.

### 4) Financial/web source policy reality
- SEC explicitly documents fair-access controls and rate limits, and asks clients to declare a meaningful User-Agent.

Implication for Loom: for finance processes, we should optimize for compliant access and source fallback, not anti-bot bypass tactics.

## Target Design

### A. Process test pyramid
Three layers:
1. `process_contract` (deterministic, PR-gating)
- No real network, no real external model calls.
- Uses stubbed model/tool responses with strict assertions on phase behavior and deliverables.

2. `process_integration` (deterministic + local integration)
- Real orchestrator + real tool registry where safe.
- Controlled fixtures for files, planner output, and verifier outcomes.

3. `process_live` (nightly/manual)
- Real provider + real web where applicable.
- Non-blocking for PR merges; trend monitored.

### B. First-class package test manifest
Extend process package schema with optional `tests` metadata.

Proposed shape in `process.yaml`:
```yaml
tests:
  - id: smoke
    mode: deterministic   # deterministic | live
    goal: "Analyze Tesla for investment"
    timeout_seconds: 900
    requires_network: true
    requires_tools:
      - web_search
      - web_fetch
    acceptance:
      phases:
        must_include:
          - company-screening
          - financial-analysis
      deliverables:
        must_exist:
          - company-overview.md
          - investment-memo.md
      verification:
        forbidden_patterns:
          - "deliverable_.* not found"
```

Principles:
- Manifest describes expected behavior, not implementation internals.
- Deterministic tests remain executable in CI without paid APIs.
- Live tests are explicit opt-in and can be skipped by environment.

### C. MCP-backed tools for processes
Add MCP client bridge that maps remote MCP tools into Loom `ToolRegistry` with namespaced identifiers:
- `mcp.<server_alias>.<tool_name>`

Runtime behavior:
1. Connect to configured MCP servers at startup.
2. Call `tools/list`; register wrappers into registry.
3. On `notifications/tools/list_changed`, refresh tool set.
4. Respect existing Loom approval gate before `tools/call` execution.
5. Preserve schema validation and error wrapping as normal Loom tools.

Process integration:
- `tools.required` can reference MCP namespaced tools.
- Missing required MCP tools fail fast with actionable diagnostics.

## Implementation Plan

### Phase 1: Test taxonomy + harness (P0)
1. Register explicit pytest markers in config:
- `process_contract`
- `process_integration`
- `process_live`
- `network`
- `mcp`
2. Enable strict marker enforcement.
3. Add shared harness helpers:
- goal runner
- plan/deliverable assertions
- structured failure categorization
4. Add deterministic contract test per built-in process YAML.

Deliverable:
- New test module(s) with one deterministic contract case per built-in process.

### Phase 2: Built-in real-case canary suite (P1)
1. Add one live scenario per built-in process:
- `investment-analysis`
- `marketing-strategy`
- `research-report`
- `competitive-intel`
- `consulting-engagement`
2. Emit artifacts:
- junit xml
- raw event stream
- summarized failure report
3. Add workflow:
- nightly schedule + manual dispatch
- non-blocking status for PRs

Deliverable:
- Nightly canary workflow and report artifacts.

### Phase 3: Package test manifest support (P1)
1. Extend `ProcessDefinition` with optional `tests` field.
2. Validate manifest schema in `ProcessLoader`.
3. Add runner command:
- `loom process test <name-or-path> [--live] [--case <id>]`
4. Add docs for package authors.

Deliverable:
- Package-level test contracts runnable by CLI and CI.

### Phase 4: MCP tool bridge (P1/P2)
1. Add MCP client configuration for external servers.
2. Build MCP-to-ToolRegistry adapter with namespacing.
3. Add dynamic refresh on `tools/list_changed`.
4. Wire approvals and tool call telemetry.
5. Add tests:
- tool discovery
- name collision behavior
- required-tool enforcement with MCP tools

Deliverable:
- Processes can safely require and use MCP tools.

### Phase 5: Source policy hardening for web-heavy processes (P1)
1. Add source profile guidance for finance process definitions (SEC APIs first, fallback chains).
2. Keep compliant headers and throttling defaults.
3. Do not implement anti-bot cloaking/evasion.

Deliverable:
- Lower false-failure rates with policy-compliant access patterns.

### Phase 6: Demo-ready quality gates (P0)
1. Define SLOs:
- `process_contract`: 100% pass required for merge.
- `process_integration`: 100% pass required for merge.
- `process_live`: tracked, alert on degradation trend.
2. Add release checklist item: all built-ins pass deterministic contract suite.
3. Publish summary to changelog/release notes.

Deliverable:
- Explicit and repeatable demo-readiness criteria.

## Test Matrix (Initial)
1. Deterministic contract tests (PR gate)
- Planner conformance to `phase_mode`
- Deliverable expectations scoped per phase
- Tool-required/excluded enforcement
- Verifier signal quality and failure classification

2. Live canary tests (nightly/manual)
- External source availability and graceful degradation
- Real provider execution path
- MCP connectivity (when enabled)

3. Package tests (author-facing)
- Validate declared test manifest
- Run deterministic cases locally before publishing

## Risks and Mitigations
1. Live tests become noisy due to provider/network volatility.
- Mitigation: keep non-blocking, classify failure types, alert on trends not single failures.

2. MCP tool surface introduces unsafe operations.
- Mitigation: namespaced tools + explicit approval policies + deny-by-default for destructive categories.

3. Package test schema churn.
- Mitigation: start with minimal stable fields (`id`, `mode`, `goal`, `acceptance`) and version schema.

4. CI runtime growth.
- Mitigation: keep deterministic suite lean and isolate heavier tests via markers/workflows.

## Recommended Execution Order
1. Phase 1 (taxonomy + deterministic harness)
2. Phase 6 baseline gates for deterministic suites
3. Phase 2 (nightly live canaries)
4. Phase 3 (package test manifest)
5. Phase 4 (MCP bridge)
6. Phase 5 (web/source hardening pass)

## External References
- Pytest markers and strict markers: https://docs.pytest.org/en/stable/how-to/mark.html
- Pytest flaky test guidance: https://docs.pytest.org/en/stable/explanation/flaky.html
- Pytest monkeypatch: https://docs.pytest.org/en/stable/how-to/monkeypatch.html
- GitHub Actions schedule behavior: https://docs.github.com/en/actions/reference/events-that-trigger-workflows
- GitHub workflow syntax (`on.schedule`): https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions
- GitHub Actions artifacts: https://docs.github.com/actions/using-workflows/storing-workflow-data-as-artifacts
- MCP tools spec (latest/draft): https://modelcontextprotocol.io/specification/draft/server/tools
- MCP 2025-11-25 key changes: https://modelcontextprotocol.io/specification/2025-11-25/changelog
- SEC developer fair access: https://www.sec.gov/about/developer-resources
- SEC webmaster FAQ (User-Agent guidance): https://www.sec.gov/about/webmaster-frequently-asked-questions
