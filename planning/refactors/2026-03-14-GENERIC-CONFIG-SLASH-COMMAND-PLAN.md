# Generic `/config` Slash Command Plan (2026-03-14)

## Executive Summary
Loom already has three separate config interaction patterns:
1. Static config loaded from `loom.toml` at startup.
2. Ad hoc TUI slash commands that expose or mutate one narrow setting family (`/telemetry`, `/mcp`).
3. A growing set of runtime accessors in `src/loom/tui/app/runtime_config.py` that read from `self._config` but do not offer a unified mutation story.

This plan introduces a generic `/config` command family in the TUI that can:
1. Discover config keys quickly.
2. Show configured, runtime, and effective values from the chat window.
3. Change process-local runtime config immediately.
4. Persist supported config keys back to the active `loom.toml`.
5. Provide high-quality autocomplete and contextual hints so users do not need to memorize paths or allowed values.
6. For selected keys, including delegated run timeout, apply runtime changes to already-active work rather than only future runs.

The core refactor is not the slash command itself. The core refactor is a shared config schema and mutation layer that both the TUI and future API/CLI settings surfaces can reuse.

## Current State Snapshot
1. Slash commands are statically declared in `src/loom/tui/app/constants.py`.
2. Slash routing lives in `src/loom/tui/app/slash/handlers.py`.
3. Slash completion is currently root-token oriented in `src/loom/tui/app/slash/completion.py`.
4. Slash hints live in `src/loom/tui/app/slash/hints.py`.
5. Config is loaded once from `loom.toml` by `src/loom/config.py::load_config()`.
6. Config dataclasses are frozen, so direct in-place mutation is not a good fit.
7. TUI runtime behavior largely reads from `self._config` through helper accessors in `src/loom/tui/app/runtime_config.py`.
8. Telemetry already has a bespoke runtime override path and persisted write flow, which is useful prior art but too narrow to scale.

## Problem Statement
Today, users can inspect some state (`/model`, `/models`, `/telemetry`) and edit some subsystems (`/mcp`), but there is no generic operator-facing way to:
1. Find a config key.
2. Learn what values it accepts.
3. See whether the live app is using the configured or overridden value.
4. Change the value from the chat window without editing TOML by hand.

That gap makes tuning harder precisely where Loom is most interactive: while a session is running and the user has immediate feedback.

## Goals
1. Add a generic `/config` slash command family for discovery, inspection, runtime override, and persistence.
2. Make autocomplete first-class for command shape, config path, flags, and enum values.
3. Define one shared registry of mutable/viewable config paths with type metadata, normalization, help text, and mutability policy.
4. Support safe process-local runtime overrides without requiring app restart.
5. Support persisted writes to `loom.toml` for explicitly allowed keys with conflict-aware atomic writes.
6. Keep output readable in chat with explicit configured/runtime/effective distinctions.
7. Preserve backward compatibility for existing `/telemetry` and `/mcp` flows during rollout.
8. Support mid-flight updates for `execution.delegate_task_timeout_seconds` on already-running delegated work.

## Non-Goals
1. Making every config value dynamically mutable on day one.
2. Replacing `loom setup` or hand-editing TOML for full-model/provider setup.
3. Exposing secret values in chat output.
4. Auto-reloading every subsystem across all processes.
5. Solving cross-process settings propagation in this phase.
6. Guaranteeing that every runtime-editable key affects already-running work.

## Proposed UX

## Command Family
Add a new slash command:

```text
/config [list|search <query>|show <path>|set <path> <value> [--scope runtime|persist|both]|reset <path> [--scope runtime|persist|both]]
```

Canonical first-phase subcommands:
1. `/config`
   Shows summary help, active config source path, and examples.
2. `/config list`
   Lists supported config paths grouped by section.
3. `/config search <query>`
   Fuzzy-ish search over supported config paths, titles, and aliases.
4. `/config show <path>`
   Shows metadata plus configured/runtime/effective value.
5. `/config set <path> <value> [--scope ...]`
   Validates and applies a value.
6. `/config reset <path> [--scope ...]`
   Removes runtime override, persisted override, or both.

Optional later subcommands:
1. `/config diff`
2. `/config changed`
3. `/config export`

## Output Contract
`/config show execution.delegate_task_timeout_seconds` should render:
1. Path
2. Description
3. Type
4. Allowed values or range
5. Default
6. Configured value
7. Runtime override value or `(none)`
8. Effective value
9. Scope support
10. Whether restart is required

Example status shape:
```text
Config: execution.delegate_task_timeout_seconds
type: int
description: Timeout for delegated orchestration calls (/run, delegate_task)
configured: 3600
runtime override: 7200
effective: 7200
supports: runtime, persist
applies to active runs: yes
restart required: no
source: /Users/sfw/.loom/loom.toml
```

## Scope Semantics
Define three write scopes:
1. `runtime`
   Process-local only. Takes effect immediately in the current TUI process.
2. `persist`
   Writes to `loom.toml` only. Reload semantics are explicit:
   1. either immediately refresh `self._config`, or
   2. report "persisted for next reload" for non-hot-reloadable fields.
3. `both`
   Applies runtime override and persists configured value in one action.

Default scope policy:
1. If a key supports runtime override, default `set` scope is `runtime`.
2. If a key is persist-only, default scope is `persist`.
3. The response must always say exactly what changed.

## Runtime Application Semantics
Each config entry should declare one application class:
1. `live`
   Runtime changes affect already-running work.
2. `next_call`
   Runtime changes affect future tool/process invocations in the current TUI process, but not work already in progress.
3. `next_run`
   Runtime changes affect future `/run` launches only.
4. `restart_required`
   Runtime mutation is unsupported or should only be persisted for a later restart.

Required first-phase guarantee:
1. `execution.delegate_task_timeout_seconds` must be `live`, not merely `next_call`.
2. Most other first-phase runtime keys may remain `next_call` or `next_run` if the code path currently snapshots settings.

## Supported Key Policy
First phase should not expose the entire `Config` tree blindly. Introduce an allowlisted registry with metadata per path:
1. `path`
2. `section`
3. `type`
4. `description`
5. `default`
6. `enum values` or numeric range
7. `supports_runtime`
8. `supports_persist`
9. `application_class` (`live|next_call|next_run|restart_required`)
10. `requires_restart`
11. `redact_in_output`
12. `aliases`
13. `normalizer/parser`

Phase 1 target families:
1. `execution.delegate_task_timeout_seconds`
2. `execution.ask_user_timeout_seconds`
3. `execution.ask_user_policy`
4. `execution.agent_tools_max_timeout_seconds`
5. `execution.cowork_tool_exposure_mode`
6. `telemetry.mode`
7. `tui.run_launch_timeout_seconds`
8. `tui.run_close_modal_timeout_seconds`
9. `tui.run_cancel_wait_timeout_seconds`
10. `tui.chat_stream_flush_interval_ms`
11. `tui.delegate_progress_max_lines`
12. `tui.run_progress_refresh_interval_ms`

Special first-phase handling:
1. `execution.delegate_task_timeout_seconds` is the only required `live` key in phase 1.
2. The initial `/config show` UX should state `applies to active runs: yes` for this key and `no` or `future calls only` for others as appropriate.

Deferred from phase 1:
1. `models.*`
2. secret-bearing fields like API keys
3. nested list-heavy structures without dedicated UX
4. MCP config, which already has its own config system and command surface

## Autocomplete and Hinting Requirements
Autocomplete is critical and should be treated as a feature workstream, not polish.

Required behavior:
1. Typing `/con<Tab>` completes to `/config`.
2. Typing `/config <Tab>` cycles subcommands.
3. Typing `/config sh<Tab>` completes `show`.
4. Typing `/config show t<Tab>` cycles supported paths starting with `t`.
5. Typing `/config set telemetry.mode <Tab>` cycles allowed enum values.
6. Typing `/config set tui.run_<Tab>` cycles matching paths.
7. Typing `/config set ... --scope <Tab>` cycles `runtime`, `persist`, `both`.
8. Hint panel should show description and current value preview for the currently selected path.

Implementation direction:
1. Extend slash completion beyond the first token.
2. Introduce command-aware completion providers rather than embedding `/config` logic directly into the generic root completer.
3. Reuse the `/tool` special-case pattern, but generalize it into pluggable completers so `/config` does not become another hard-coded one-off.

## Proposed Architecture

## 1) Shared Config Registry
Add a new module family, for example:
1. `src/loom/config_runtime/registry.py`
2. `src/loom/config_runtime/schema.py`
3. `src/loom/config_runtime/store.py`
4. `src/loom/config_runtime/toml_edit.py`

Responsibilities:
1. Enumerate supported config entries.
2. Provide lookup by path and alias.
3. Parse string input into typed values.
4. Validate ranges and enums.
5. Generate help/autocomplete metadata.
6. Indicate whether a field supports runtime, persist, or both.
7. Indicate whether a field applies `live`, `next_call`, `next_run`, or `restart_required`.

## 2) Runtime Override Store
Introduce a process-local override store for the TUI.

Why:
1. Config dataclasses are frozen.
2. We need explicit configured vs runtime vs effective values.
3. We do not want dozens of ad hoc fields like `_telemetry_runtime_override_mode`.

Suggested shape:
1. Base config snapshot: `self._config`
2. Runtime override map: `dict[str, object]`
3. Effective lookup helpers:
   1. `get_config_value(path, effective=True)`
   2. `configured_config_value(path)`
   3. `runtime_override_value(path)`
4. Accessor helpers in `runtime_config.py` should move toward registry-backed resolution for mutable fields.

Recommended implementation detail:
1. For hot fields currently accessed through `runtime_config.py`, teach those accessors to consult the override store first.
2. For slash output and future settings surfaces, use the same store directly.
3. Do not mutate frozen config dataclasses in place as the primary mechanism.
4. The store should expose application-class metadata so the TUI can explain whether a change affects active work.

## 2a) Delegate Timeout Live-Update Path
`execution.delegate_task_timeout_seconds` needs additional implementation beyond the generic override store.

Current constraint:
1. The tool registry wraps tool execution in a fixed `asyncio.wait_for(...)` timeout budget.
2. `DelegateTaskTool.timeout_seconds` is resolved at call start.
3. That means mid-flight config changes do not extend or shorten an already-running delegated call today.

Required refactor:
1. Introduce a dynamic timeout budget path for `DelegateTaskTool`.
2. Replace the fixed one-shot timeout enforcement for this tool with a deadline monitor that can observe config updates while the task is in flight.
3. Resolve the effective delegate timeout from the shared runtime config store rather than only constructor-time config.
4. Support safe extension and contraction rules:
   1. extending the timeout should lengthen the remaining budget for the active delegated call
   2. reducing the timeout should only cancel if the run has already exceeded the newly allowed total budget

Suggested implementation directions:
1. Add tool capability metadata such as `supports_live_timeout_updates`.
2. For tools without that capability, keep the existing `asyncio.wait_for(...)` path.
3. For `DelegateTaskTool`, run `tool.execute(...)` in a task and supervise it with a lightweight polling loop that recomputes the allowed budget from the runtime config store.
4. Base the live decision on total elapsed time since tool execution started, not on repeated relative timeouts.

Out of scope for this phase:
1. Making all runner-internal settings live-updatable for already-running delegated orchestration.
2. Retrofitting every tool in the registry with live timeout semantics.

## 3) Persisted Write Engine
Generalize the telemetry persisted-write pattern into a reusable TOML write helper.

Requirements:
1. Atomic write with temp file + replace.
2. Conflict detection via mtime snapshot.
3. Advisory lock file.
4. Preserve unrelated config.
5. Create missing section/table headers when needed.
6. Support removing keys for `reset`.

Important scope decision:
1. Phase 1 may use a narrow, surgical TOML editor for scalar keys only.
2. We do not need a full formatting-preserving TOML AST editor immediately.
3. If preserving comments/order becomes too costly, document that persisted writes normalize only the touched section formatting.

## 4) TUI Slash Layer Refactor
Refactor slash composition so command-specific completion and hints are pluggable.

Suggested new pieces:
1. `src/loom/tui/app/slash/config_command.py`
2. `src/loom/tui/app/slash/providers.py`

Responsibilities:
1. Parse `/config` arguments.
2. Render `/config` hints.
3. Provide `/config` completion candidates at each argument position.
4. Dispatch `/config` subcommands.

This reduces pressure on:
1. `src/loom/tui/app/slash/handlers.py`
2. `src/loom/tui/app/slash/completion.py`
3. `src/loom/tui/app/slash/hints.py`

## 5) Compatibility Strategy
Keep `/telemetry` for one release cycle as a specialized alias over the shared config backend.

Behavior:
1. `/telemetry` continues to work.
2. Its implementation delegates to the new config registry/store for `telemetry.mode`.
3. Help text points users toward `/config show telemetry.mode` and `/config set telemetry.mode ...`.

## Config Source Path Handling
The TUI already receives `legacy_config_path`, but the naming reflects MCP legacy layering rather than generic config ownership.

Refactor recommendation:
1. Introduce a clearly named `_config_source_path`.
2. Preserve `legacy_config_path` only where MCP migration logic needs it.
3. Have `/config` responses show the actual source path when present.
4. If the session started with defaults and no config file exists, say so explicitly and only allow `persist` if a target path is deterministically chosen.

## Safety and Policy
1. Redact secret-bearing fields from list/show output.
2. Refuse runtime mutation for unsupported or restart-required keys unless explicitly marked safe.
3. For persist writes, reject unknown paths rather than attempting generic TOML insertion.
4. Every mutation response should include whether the change was:
   1. applied now
   2. persisted for future runs
   3. partially applied
5. Emit telemetry or audit log events for config mutations.

## Workstreams

## Workstream 1: Config Registry and Types
Files:
1. `src/loom/config_runtime/schema.py`
2. `src/loom/config_runtime/registry.py`

Tasks:
1. Define config entry metadata model.
2. Register first-phase mutable/viewable paths.
3. Add parsing and validation helpers.
4. Add path search and alias matching.

Acceptance:
1. Registry can enumerate supported keys deterministically.
2. Registry can validate scalar string input into typed values.

## Workstream 2: Runtime Override Store
Files:
1. `src/loom/config_runtime/store.py`
2. `src/loom/tui/app/state_init.py`
3. `src/loom/tui/app/runtime_config.py`

Tasks:
1. Replace `_telemetry_runtime_override_mode` special case with generic override support.
2. Add lookup helpers for configured/runtime/effective values.
3. Route hot runtime accessors through the override store for supported fields.
4. Mark each supported key with an application class and expose that in lookups.

Acceptance:
1. Runtime changes take effect without restarting the TUI for supported keys.
2. Telemetry mode becomes one instance of the generic path rather than a one-off.
3. The UI can distinguish `live` changes from `future calls only` changes.

## Workstream 3: Persisted TOML Mutation Layer
Files:
1. `src/loom/config_runtime/toml_edit.py`
2. `src/loom/tui/app/core.py` or dedicated config service module

Tasks:
1. Generalize atomic persisted update logic from telemetry.
2. Add scalar key upsert/remove helpers.
3. Handle config-path conflict detection and target-path resolution.

Acceptance:
1. Persisted writes are atomic and conflict-aware.
2. Reset can remove a persisted scalar key cleanly.

## Workstream 3a: Delegate Timeout Live Enforcement
Files:
1. `src/loom/tools/registry.py`
2. `src/loom/tools/delegate_task.py`
3. `src/loom/tools/__init__.py`
4. `tests/test_tools.py`
5. `tests/test_runner_execution.py` or focused delegate timeout tests

Tasks:
1. Refactor `DelegateTaskTool` to resolve timeout from the shared runtime config source.
2. Add a registry execution path that supports live timeout updates for delegate-task calls already in progress.
3. Define cancellation semantics when the timeout is reduced mid-flight.
4. Surface active-run applicability in `/config show execution.delegate_task_timeout_seconds`.

Acceptance:
1. Increasing `execution.delegate_task_timeout_seconds` during an active delegated `/run` extends the allowed total runtime for that in-flight call.
2. Decreasing the timeout only cancels once elapsed runtime exceeds the new total budget.
3. Non-delegate tools retain the existing simpler timeout path unless explicitly upgraded.

## Workstream 4: `/config` Slash Command and UX
Files:
1. `src/loom/tui/app/constants.py`
2. `src/loom/tui/app/slash/handlers.py`
3. `src/loom/tui/app/slash/completion.py`
4. `src/loom/tui/app/slash/hints.py`
5. new `/config`-specific module(s)

Tasks:
1. Add `/config` to command registry and help.
2. Implement list/search/show/set/reset flows.
3. Add contextual hint rendering.
4. Add multi-position autocomplete.

Acceptance:
1. Users can discover and mutate supported settings from the chat window without reading docs.
2. Completion quality is high enough that config paths do not need to be memorized.

## Workstream 5: Compatibility Cleanup
Files:
1. `src/loom/tui/app/slash/handlers.py`
2. `tests/tui/test_slash_commands_telemetry.py`
3. `docs/CONFIG.md`

Tasks:
1. Re-implement `/telemetry` on top of shared config runtime services.
2. Update docs to position `/config` as the generic path.
3. Document supported scopes and limits.

Acceptance:
1. No regression in `/telemetry`.
2. `/config` becomes the documented default operator surface.

## Testing Plan
1. Registry unit tests:
   1. path lookup
   2. alias lookup
   3. type parsing
   4. range validation
   5. enum validation
2. Runtime-store unit tests:
   1. configured/runtime/effective resolution
   2. reset behavior
   3. unsupported runtime mutation rejection
3. TOML mutation tests:
   1. upsert into existing section
   2. create missing section
   3. remove key on reset
   4. mtime conflict handling
4. TUI slash command tests:
   1. `/config`
   2. `/config list`
   3. `/config search timeout`
   4. `/config show execution.delegate_task_timeout_seconds`
   5. `/config set telemetry.mode debug`
   6. `/config set tui.run_launch_timeout_seconds 90 --scope runtime`
   7. `/config reset telemetry.mode --scope runtime`
5. Completion tests:
   1. root completion includes `/config`
   2. subcommand completion
   3. path completion
   4. enum-value completion
   5. scope-flag completion
6. Hint rendering tests:
   1. show current value preview
   2. show allowed values
   3. show no-match feedback
7. Delegate live-timeout tests:
   1. active delegate run survives a mid-flight timeout extension
   2. active delegate run is cancelled after a mid-flight timeout reduction when elapsed exceeds the new budget
   3. `/config show execution.delegate_task_timeout_seconds` reports live applicability correctly

## Rollout Plan
## Phase A
1. Build registry, runtime store, and `/config show|set` for a small allowlist.
2. Cover `telemetry.mode` and the key timeout knobs first.
3. Include `execution.delegate_task_timeout_seconds` as the first `live` key, not just a generic runtime key.

## Phase B
1. Add `list`, `search`, `reset`, and richer hint text.
2. Generalize completion providers.
3. Expand the mutability matrix to more keys as code paths are refactored.

## Phase C
1. Migrate `/telemetry` to shared backend.
2. Expand supported config families.

## Risks and Mitigations
1. Risk: runtime overrides do not actually affect live behavior.
   Mitigation: only mark keys `supports_runtime=true` after their accessors consult the shared override store.
2. Risk: users assume `execution.delegate_task_timeout_seconds` is live-updatable when only future delegate calls changed.
   Mitigation: do not ship that key as `live` until the tool registry deadline enforcement has been refactored and tested.
3. Risk: persisted writes corrupt or over-normalize `loom.toml`.
   Mitigation: scalar-only writes first, atomic replace, conflict detection, focused tests.
4. Risk: autocomplete complexity bloats generic slash code.
   Mitigation: introduce command-specific completion providers.
5. Risk: users assume every config field is hot-reloadable.
   Mitigation: show `supports`, `application_class`, and `restart required` in every `/config show` response.

## Acceptance Criteria
1. A user can discover `execution.delegate_task_timeout_seconds` from the TUI without external docs.
2. A user can change `telemetry.mode` and supported timeout values from chat and see the effective value immediately.
3. A user can increase `execution.delegate_task_timeout_seconds` during an active delegated `/run` and the in-flight run receives the larger budget.
4. A user can persist supported scalar settings back to `loom.toml` safely.
5. Slash completion materially assists path and value selection.
6. Existing `/telemetry` behavior remains intact during migration.

## Recommended First Slice
Implement this smallest valuable slice first:
1. registry for 6-10 keys
2. generic runtime override store
3. `/config show`
4. `/config set`
5. path + enum + scope autocomplete
6. persisted scalar writes for the same allowlist
7. live timeout support for `execution.delegate_task_timeout_seconds`

That slice is enough to prove the architecture while also solving the one active-run timeout knob we know operators need in practice.
