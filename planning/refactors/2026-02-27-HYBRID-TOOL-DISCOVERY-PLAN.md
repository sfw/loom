# Hybrid Tool Discovery Plan (2026-02-27)

## Objective
Adopt a hybrid tool exposure strategy for cowork mode that keeps common tool calls strongly typed while adding low-overhead discovery and generic execution for long-tail tools.

Required outcomes:
1. Reduce request size and token pressure for short and normal turns.
2. Preserve high reliability for common coding/research tool calls.
3. Avoid sending full tool schema catalogs on every turn.
4. Maintain existing safety, approval, and auth behavior.

## Why Hybrid
Pure generic dispatch (`list_tools` + `run_tool`) minimizes prompt size but lowers tool-call precision because models lose per-tool typed function signatures.
Pure typed-per-tool dispatch is reliable but expensive when many tools exist.
Hybrid keeps typed schemas for likely tools and provides generic fallback for the rest.

## Baseline (Measured)
For user input `hi` in cowork mode:
1. Full schema mode: 52 tools, 50,340 bytes, 12,584 estimated tokens.
2. Adaptive typed subset mode: 4 tools, 6,303 bytes, 1,575 estimated tokens.
3. Net savings from adaptive subset: 44,037 bytes and 11,009 estimated tokens.

## Proposed Design

### 1) Typed Core Lane (Primary)
1. Continue intent-aware typed tool subset selection per turn.
2. Keep a compact always-on core (`ask_user`, `task_tracker`, `conversation_recall`, `delegate_task`).
3. Include domain lanes (coding, web, writing, finance) via lightweight keyword routing.
4. Cap typed schemas per turn at `16` to keep prompt bounded.

### 2) Generic Fallback Lane (Secondary)
Add two new tools exposed on every cowork turn:
1. `list_tools`
   - Returns compact catalog of available tools.
   - Supports filtering (`query`, `category`, `mutating_only`, `auth_required_only`, `limit`, `offset`).
   - Returns minimal fields first (`name`, `summary`, `required_args`, `mutates`, `auth_required`).
   - Enforce response bounds: default `limit=20`, max `limit=50`, and bounded payload size.
2. `run_tool`
   - Executes a target tool via `{name, arguments}`.
   - Reuses existing registry execution path, approvals, auth resolution, and path safety.
   - Returns structured validation failures so the model can self-correct arguments.

### 3) Prompt/Protocol Contract
System guidance for cowork:
1. Prefer typed tools when available in current tool list.
2. If needed tool is missing or uncertain, call `list_tools`.
3. Then call `run_tool` with exact `name` and `arguments`.
4. On `run_tool` validation errors, repair arguments and retry.

### 4) Registry/Execution Semantics
1. `run_tool` must not bypass `ToolRegistry.execute`.
2. `run_tool` must pass through the same mutating-tool approval path as direct tool calls.
3. `run_tool` must execute with the same workspace/scratch/auth context as direct tool calls.
4. `list_tools` and `run_tool` must honor auth-scoped tool visibility (including MCP discovery scoping).
5. `run_tool` must reject `name == "run_tool"` to prevent recursive self-dispatch.
6. Existing telemetry and persistence must record underlying tool execution details.

### 5) Cap Accounting Contract
1. Always include fallback tools `list_tools` and `run_tool`.
2. Typed schema cap (`16`) applies to the primary typed lane only (excluding fallback tools).
3. Total tool-schema budget target is therefore `18` schemas maximum per request (`16` typed + `2` fallback).
4. Byte budget checks should consider total serialized tool schema bytes, not count alone.
### 6) Telemetry and Guardrails
Track:
1. `typed_tool_schema_count` per model call.
2. `total_tool_schema_count` per model call.
3. `generic_tool_list_calls` and `generic_run_tool_calls`.
4. `run_tool_validation_failures`.
5. Request size/tokens before and after hybrid rollout.
6. Success rate and retries for tool calls by lane (typed vs generic).

## Implementation Workstreams

### Workstream 1: Add Generic Fallback Tools
Files:
1. `src/loom/tools/list_tools.py` (new)
2. `src/loom/tools/run_tool.py` (new)
3. `src/loom/tools/__init__.py` (auto-discovery already present; verify inclusion tests only)

Deliverables:
1. `list_tools` tool with compact output and filters.
2. `run_tool` tool that dispatches through registry safely.
3. Structured error payloads for unknown tool, argument schema mismatch, and safety denial.
4. Enforced response limits for `list_tools` (limit bounds and payload bounds).

Acceptance:
1. `run_tool` can execute at least one read tool and one mutating tool through normal policy paths.
2. Validation failures are machine-readable and retry-friendly.
3. `list_tools` cannot exceed configured max item count or payload bounds.

### Workstream 2: Thread-Safe `run_tool` Binding
Files:
1. `src/loom/tools/registry.py`
2. `src/loom/cowork/session.py`

Deliverables:
1. Safe mechanism for `run_tool` to call registry without recursive deadlock.
2. Preserve workspace/scratch/auth context used by normal tool calls.
3. Guard against `run_tool` calling itself.
4. Preserve approval path parity with direct mutating tool calls.
5. Preserve auth-scoped tool visibility for both `list_tools` and `run_tool`.

Acceptance:
1. No recursion loop or deadlock under concurrent turns.
2. Underlying tool behavior matches direct invocation.
3. Approval prompt behavior for mutating tools is identical between direct call and `run_tool`.
4. Auth-scoped MCP tool visibility does not leak across profiles/sessions.

### Workstream 3: Cowork Hybrid Policy
Files:
1. `src/loom/cowork/session.py`
2. `src/loom/tui/app.py` (optional UI messaging)

Deliverables:
1. Always include `list_tools` and `run_tool` in selected schemas.
2. Keep typed subset cap and lane selection logic.
3. Add debug telemetry line for selected typed tools and generic lane usage.
4. Enforce cap accounting: `16` typed + `2` fallback max.

Acceptance:
1. Small-talk turns stay near compact payload targets.
2. Complex turns can still reach long-tail tools via generic lane.
3. Typed cap and total cap behavior are deterministic and testable.

### Workstream 4: Error Recovery UX
Files:
1. `src/loom/cowork/session.py`
2. `src/loom/models/openai_provider.py` (if additional normalization is needed)

Deliverables:
1. If model calls unknown typed tool name, inject concise correction to use `list_tools`.
2. If `run_tool` argument payload is malformed, return concise arg contract.
3. Prevent repeated identical malformed retries with bounded retry hints.

Acceptance:
1. Unknown-tool failures converge to successful execution without user intervention in common cases.
2. Retry loops terminate predictably.

### Workstream 5: Test Coverage
Files:
1. `tests/test_cowork.py`
2. `tests/test_tools.py`
3. `tests/test_new_tools.py` or new dedicated tests for `list_tools` and `run_tool`

Deliverables:
1. Unit tests for `list_tools` filters and pagination.
2. Unit tests for `run_tool` happy path, unknown tool, invalid args, and self-call guard.
3. Cowork integration tests proving hybrid payload selection and generic fallback use.
4. Regression tests for approval/auth/safety path preservation through `run_tool`.
5. Tests for `list_tools` response-size and per-call item bounds.
6. Tests proving auth-scoped tool visibility is preserved in `list_tools` and `run_tool`.

Acceptance:
1. Existing cowork and tool suites remain green.
2. Hybrid-specific tests fail on old behavior and pass on new.

### Workstream 6: Docs and Operator Guidance
Files:
1. `README.md`
2. `docs/tutorial.html`
3. `docs/agent-integration.md`

Deliverables:
1. Explain hybrid lane behavior.
2. Document when models should call `list_tools`.
3. Document `run_tool` input contract and error semantics.

Acceptance:
1. A user can understand and debug hybrid lane behavior from docs alone.

## Rollout Phases

### Phase R1: Generic Tool Scaffolding Behind Mode Gate
1. Add `list_tools` and `run_tool` implementation.
2. Gate usage with `execution.cowork_tool_exposure_mode`.
3. Supported values: `full`, `adaptive`, `hybrid`.
4. Keep default at `adaptive` during initial rollout.
5. Run tests and collect local telemetry.

Exit criteria:
1. No behavior change when mode remains `adaptive` (or `full`).
2. New tools function correctly in isolation.

### Phase R2: Enable Hybrid in Cowork (Opt-In)
1. Turn mode to `hybrid` for local/dev sessions.
2. Ensure typed subset includes generic fallback tools.
3. Verify no safety/approval regressions.

Exit criteria:
1. `hi` request size remains in compact range (target < 8 KB).
2. Common coding tasks still resolve with typed tools.

### Phase R3: Default-On Hybrid
1. Flip default mode from `adaptive` to `hybrid` after stability window.
2. Keep emergency rollback to `adaptive` quickly.
3. Monitor request size and tool-call success metrics.

Exit criteria:
1. Sustained request-size reduction on cowork turns.
2. No statistically meaningful drop in successful first-pass tool calls.

## Success Metrics
1. Median cowork request bytes reduced by at least 60%.
2. P95 cowork request bytes reduced by at least 40%.
3. First-pass successful tool-call rate within 2% of baseline.
4. `run_tool` validation error retry success rate above 80% within two retries.
5. `run_tool` fallback-lane first-pass success rate at or above 70% during opt-in rollout.
6. No increase greater than 2% in approval-related failures for mutating operations.

## Risks and Mitigations
1. Risk: generic lane lowers argument quality.
   - Mitigation: keep typed lane for common tools; return explicit required-arg feedback.
2. Risk: `run_tool` bypasses safeguards.
   - Mitigation: mandatory dispatch through existing registry/approval/auth path.
3. Risk: additional model round-trips when tool is not in typed subset.
   - Mitigation: improve lane keyword routing and maintain compact but useful general lane.
4. Risk: recursion or deadlock.
   - Mitigation: explicit self-call guard and bounded execution path tests.

## Decisions (Resolved 2026-02-27)
1. Config mode and timeline:
   - Use enum `execution.cowork_tool_exposure_mode = "full" | "adaptive" | "hybrid"`.
   - Immediate default remains `adaptive`.
   - Promote default to `hybrid` after rollout metrics are stable.
2. Typed schema cap:
   - Hard cap typed schemas per turn at `16`.
   - Add typed-schema byte budget target around `12 KB` (enforced or emitted as guard telemetry).
3. `list_tools` default response shape:
   - Default to compact payload: `name`, `summary`, `required_args`, `mutates`, `auth_required`.
   - Support opt-in detailed schema mode (for example `detail="schema"`).
   - Enforce default and max result size constraints:
     - default `limit=20`
     - max `limit=50`
     - response payload bounded by configured max bytes.
4. `run_tool` batching:
   - Do not batch in v1; keep `run_tool` single-call.
   - If needed later, add a separate `run_tool_batch` with explicit approval/error semantics.
