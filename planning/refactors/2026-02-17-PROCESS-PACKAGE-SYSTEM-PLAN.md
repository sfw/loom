# Process/Package System Review + Refactor Plan (2026-02-17)

## Execution Status (2026-02-17)
Status: Completed

All implementation phases in this plan have been executed and validated.
Final pass included:
- Runtime semantics for `is_critical_path`, `is_synthesis`, and `replanning.triggers`.
- Isolated dependency install path (`loom install --isolated-deps`) plus loader/runtime activation.
- Bundled tool collision diagnostics and skip-on-conflict behavior.
- Docs realignment (`README.md`, `docs/creating-packages.md`, `docs/agent-integration.md`) including compatibility checklist and troubleshooting.
- Full lint + full test suite validation.

## Scope
Reviewed the process and package stack end to end:
- Process schema/loader: `/Users/sfw/Development/loom/src/loom/processes/schema.py`
- Package install/uninstall: `/Users/sfw/Development/loom/src/loom/processes/installer.py`
- Built-in process definitions: `/Users/sfw/Development/loom/src/loom/processes/builtin/*.yaml`
- Orchestrator integration: `/Users/sfw/Development/loom/src/loom/engine/orchestrator.py`
- Prompt integration: `/Users/sfw/Development/loom/src/loom/prompts/assembler.py`
- API integration: `/Users/sfw/Development/loom/src/loom/api/routes.py`, `/Users/sfw/Development/loom/src/loom/api/engine.py`
- TUI integration: `/Users/sfw/Development/loom/src/loom/tui/app.py`, `/Users/sfw/Development/loom/src/loom/tui/commands.py`
- Delegate bridge: `/Users/sfw/Development/loom/src/loom/tools/delegate_task.py`
- Config and docs/tests alignment.

## Executive Summary
The process system is strong in `loom run` / API task execution, but only partially active in TUI cowork mode. Today, process behavior in TUI is mostly persona/tool-guidance text and tool exclusion; the phase/verification/memory workflow does not reliably carry through delegated task execution. Package installation works, but package runtime and docs have drifted from actual behavior.

## Best Utilization Right Now (Before Refactor)
1. Use `loom run --process <name>` (or API `POST /tasks` with `process`) when you want full process behavior.
2. Treat TUI `--process` as soft guidance, not strict phase orchestration.
3. In TUI, explicitly ask for `delegate_task` only as a workaround; it is not currently process-aware by default.

## Findings

### P0: Process behavior is inconsistent between modes
1. TUI process loading is partial.
- In `/Users/sfw/Development/loom/src/loom/tui/app.py`, process load adds persona/tool guidance and excludes tools, but does not inject full process behavior via process-aware orchestrator/prompt stack.

2. `delegate_task` in TUI is not process-aware.
- In `/Users/sfw/Development/loom/src/loom/tui/app.py`, `_bind_session_tools` builds `Orchestrator(...)` without `process=...`.
- Result: delegated complex work in cowork mode ignores active process phases/rules.

3. Session operations in TUI drop process context.
- `_new_session` and `_switch_to_session` rebuild generic cowork prompt and do not re-apply loaded process definition.

4. Bundled package tools are not reliably available in TUI.
- TUI tool registry is created before process loading; bundled tools are imported later by loader side-effects.
- Existing registry is not rebuilt/rebound after bundled tool import.

5. Process "strictness" is mostly advisory today.
- `phase_mode`, `is_critical_path`, `is_synthesis`, and `replanning.triggers` are parsed, but enforcement is weak or absent.
- Planner prompt contains instructions, but engine does not guarantee conformance to declared process DAG semantics.

6. `tools.required` is not enforced.
- Schema validates overlap with excluded tools, but runtime does not assert required tools exist.
- Docs currently imply warnings exist.

### P1: Package lifecycle and runtime gaps
1. Installer copies only a subset of package files.
- `/Users/sfw/Development/loom/src/loom/processes/installer.py` copies `process.yaml`, `tools/`, and limited optional files.
- Any additional package assets are dropped.

2. Tool-name collision risk for bundled tools.
- Bundled tools register globally and can collide with built-ins/other bundled tools.
- Collision handling is currently fail-fast in registry, with no namespace strategy.

3. Package dependency install is environment-global.
- `loom install` installs dependencies into current Python environment.
- No per-package isolation option exists.

4. Source format/documentation drift.
- Package README example in `/Users/sfw/Development/loom/packages/google-analytics/README.md` uses `loom install github.com/...`, which is not accepted by current shorthand parser.

### P2: Configuration and UX drift
1. `config.process.default` appears unused.
- Config supports default process but CLI/TUI/API startup paths do not apply it automatically.

2. TUI has no process controls.
- No `/process` slash command family and no command-palette actions for process lifecycle.
- Users cannot inspect/switch/deactivate process in-session.

3. Docs overstate runtime guarantees.
- Package docs imply strict phase execution semantics and required-tool warnings not fully implemented.

## Recommended Target Design

### 1) Introduce a single `ProcessRuntime` abstraction
- Resolve process once per run/session and store:
  - loaded definition
  - source path
  - active tool policy (required/excluded)
  - runtime flags (strict enforcement enabled, warnings)
- Pass the same object through TUI session, delegate factory, and orchestrator creation.

### 2) Make process behavior explicit in TUI
- Add slash commands:
  - `/process` (show active)
  - `/process list`
  - `/process use <name-or-path>`
  - `/process off`
  - `/process run <goal>` (force process-aware delegated execution)
- Show active process badge in status/sidebar.
- Rebind session and delegate orchestrator after process changes.

### 3) Align execution semantics with process schema
- Enforce `tools.required` at process activation.
- Add planner output validation against process DAG when `phase_mode=strict`:
  - missing required phases => reject/replan
  - invalid dependencies => reject/replan
- Define clear behavior for `is_critical_path` and `is_synthesis` at execution level.
- Use `replanning.triggers` in replanner decision prompt/context.

### 4) Fix package runtime and distribution
- Installer should copy package directory contents (with denylist for VCS/cache artifacts) rather than selective allowlist.
- Add optional `--isolated-deps` mode (per-package venv or managed env).
- Add bundled tool namespace convention and collision policy:
  - recommended: require explicit unique tool names and emit actionable conflict diagnostics.

### 5) Documentation realignment
- Update docs to reflect exact current/target behavior by mode (`run` vs `cowork`).
- Fix install source examples and required-tool semantics documentation.

## Implementation Plan

### Phase 1: Correctness First (P0)
1. Persist loaded process definition in `LoomApp` and reapply across new/switch session.
2. Pass active process into delegate-created orchestrators.
3. Rebuild/rebind tool registry after bundled tool loading in TUI process activation path.
4. Add tests for TUI process propagation and delegate_task process awareness.

### Phase 2: Policy Enforcement (P0/P1)
1. Implement `tools.required` runtime checks and user-facing diagnostics.
2. Add strict-phase planner conformance checks when `phase_mode=strict`.
3. Define/implement runtime semantics for `is_critical_path` and `is_synthesis`.

### Phase 3: Package Hardening (P1)
1. Expand installer copy behavior to include package assets safely.
2. Add bundled tool collision diagnostics and test coverage.
3. Add optional dependency isolation path.

### Phase 4: UX + Operability (P2)
1. Add `/process` command family and palette actions.
2. Apply `config.process.default` when `--process` is omitted.
3. Surface active process clearly in TUI status/help.

### Phase 5: Docs and Demo Readiness
1. Update README + `docs/creating-packages.md` + `docs/agent-integration.md` with mode-accurate behavior.
2. Add a process package compatibility checklist and troubleshooting section.
3. Validate end-to-end demo scenarios.

## Testing Plan
1. Unit tests
- Process enforcement logic: strict mode validation, required tool checks, critical path handling.
- Package installer copy behavior and collision diagnostics.

2. Integration tests
- TUI with `--process` plus delegated task execution using process-aware orchestrator.
- Session lifecycle (`/new`, `/resume`) preserving process context.
- Bundled tool availability in TUI and API process tasks.

3. End-to-end tests
- `loom run --process marketing-strategy` produces phase-aligned outputs.
- `loom --process marketing-strategy` + `/process run ...` produces equivalent behavior.
- Install local package with bundled tool and verify tool callable in session.

## Demo-Ready Exit Criteria
1. A process selected in TUI behaves consistently with `loom run` for delegated/process-run flows.
2. Bundled tools from installed packages are usable in both TUI and API task execution.
3. Strict process mode has deterministic enforcement, not just prompt hints.
4. Required tools and package/runtime conflicts produce clear, actionable errors.
5. Docs match observed behavior and commands.

## Recommended Execution Order
1. Phase 1 (correctness)
2. Phase 2 (enforcement)
3. Phase 4 (TUI process UX)
4. Phase 3 (package hardening)
5. Phase 5 (docs/demo gate)
