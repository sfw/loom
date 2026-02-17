# Changelog

All notable changes to Loom are documented in this file.

## [Unreleased]

### Added
- **Tree-sitter code analysis** (`tools/treesitter.py`) -- optional syntax-tree-based backend for `analyze_code` and `edit_file` structural matching. Uses `tree-sitter-language-pack` when installed, falls back silently to regex extractors. Supports Python, JavaScript/TypeScript, Go, and Rust. Provides accurate class/function extraction including nested definitions, decorators, and doc-strings. Also used by `edit_file` to find structural candidates for fuzzy matching. Install with `uv sync --extra treesitter`.
- **Adaptive Learning Memory (ALM)** (`learning/reflection.py`) -- behavioral pattern extraction via task completion gap analysis. Detects the gap between what the model delivered and what the user actually wanted. Extracts general behavioral rules ("run tests after writing code") from implicit follow-ups ("test and lint it") and explicit corrections ("no, use JSON"). Patterns are frequency-weighted and injected into future system prompts. No regex -- gaps are structural, not lexical.
- **`loom learned` CLI command** (`__main__.py`) -- review, filter, and delete learned patterns from the terminal. Supports `--type` filtering, `--delete ID` for removal, and `--limit` for output control.
- **`/learned` TUI slash command** (`tui/app.py`, `tui/screens/learned.py`) -- interactive modal for reviewing and deleting learned patterns. Shows pattern type, description, frequency, last seen date, and per-row delete buttons. Added to `/help` output.
- **`LearningManager.delete_pattern()`** (`learning/manager.py`) -- delete individual learned patterns by ID. Enables selective curation of the learning database.
- **`LearningManager.query_all()`** (`learning/manager.py`) -- query all patterns regardless of type, for use by review interfaces.
- **Word and PowerPoint document support** (`tools/file_ops.py`, `content_utils.py`) -- `read_file` now extracts text from `.docx`/`.doc` and `.pptx`/`.ppt` files using `python-docx` and `python-pptx`. Returns a `DocumentBlock` with extracted text for all model providers. Both libraries are required dependencies (not optional).
- **Process test harness and CLI runner** (`processes/testing.py`, `__main__.py`, `tests/test_process_testing.py`) -- added deterministic/live process case execution, default smoke-case generation, case filtering, acceptance checks, and `loom process test <name-or-path> [--live] [--case <id>]`.
- **Built-in deterministic process contract suite** (`tests/test_process_contracts.py`) -- each shipped process now has deterministic contract coverage via `pytest` marker `process_contract`.
- **Live process canary suite and workflow** (`tests/test_process_live.py`, `.github/workflows/process-canary.yml`) -- added opt-in live tests (`process_live`) plus nightly/manual artifacted workflow (JUnit, logs, per-process JSON summaries).
- **MCP tool bridge for processes and runtime tools** (`integrations/mcp_tools.py`, `config.py`, `tools/__init__.py`, `tests/test_mcp_tools_bridge.py`) -- Loom now discovers external MCP tools from config and registers them as namespaced tools (`mcp.<server>.<tool>`), enabling process `tools.required` checks against MCP-backed capabilities.
- **Dedicated MCP config manager** (`mcp/config.py`, `mcp/__init__.py`, `tests/test_mcp_config_manager.py`) -- added shared MCP load/merge/mutate backend with layered precedence (`--mcp-config` > workspace `./.loom/mcp.toml` > user `~/.loom/mcp.toml` > legacy `[mcp]`), atomic writes, redaction helpers, and legacy migration support.
- **`loom mcp` command group** (`__main__.py`, `tests/test_cli.py`) -- new MCP lifecycle commands: `list`, `show`, `add`, `edit`, `remove`, `enable`, `disable`, `test`, and `migrate`.
- **TUI `/mcp` command family** (`tui/app.py`, `tests/test_tui.py`) -- added MCP list/show/test/enable/disable/remove controls in-session, with immediate MCP tool refresh after successful mutations.

### Changed
- **Unified TUI as default interface** (`tui/app.py`, `__main__.py`) -- `loom` with no subcommand now launches the Textual TUI with full cowork backend: session persistence (SQLite), conversation recall, task delegation, process definitions, and session management. The separate plain-text REPL is removed. `loom cowork` is an alias for the default TUI. New slash commands: `/sessions`, `/new`, `/session`, `/resume <id>`.
- **Setup wizard moved into TUI** (`tui/screens/setup.py`, `__main__.py`) -- first-run configuration now launches as a multi-step modal inside the TUI instead of CLI prompts. Five-step keyboard-driven flow: provider selection, model details, role assignment, optional utility model, and confirmation. Reconfigure anytime with the `/setup` slash command. The `loom setup` CLI command is retained as a headless fallback.
- **Provider model selection now auto-discovers from endpoints** (`setup.py`, `tui/screens/setup.py`) -- setup no longer relies on hardcoded Anthropic model names. It probes provider APIs (`Anthropic /v1/models`, OpenAI-compatible `/models` with `/v1/models` fallback, Ollama `/api/tags`) and presents discovered options; if discovery fails, setup falls back to manual model entry.
- **Setup wizard interaction/feedback polish** (`tui/screens/setup.py`) -- details step now enforces endpoint/auth-first discovery flow (URL + required key before discovery), caps rendered discovered models to avoid modal overflow, and adds explicit visual selection feedback plus clearer per-step keybinding confirmation (including save-step `Enter/Y/S` and `B/Esc`). Confirm summary is now scroll-bounded so hotkey hints stay visible.
- **Quit flow now requires confirmation in TUI** (`tui/app.py`, `tui/screens/confirm_exit.py`) -- `Ctrl+C`, slash-command quit (`/quit`, `/exit`, `/q`), and command palette quit now route through a confirmation modal before exiting, preventing accidental session termination.
- **Live slash-command autocomplete hints in TUI input** (`tui/app.py`) -- typing `/` now shows available slash commands, and each subsequent keypress incrementally filters matches in a dedicated hint panel below the input (`/res` -> `/resume`, etc.). Unmatched prefixes show an inline "Try /help" hint.
- **Slash command registry refactor for consistency** (`tui/app.py`) -- slash autocomplete and `/help` output now derive from one shared command specification, preventing drift and missing entries. `/resume` now has explicit usage guidance (`/resume <session-id-prefix>`) when invoked without an argument.
- **Process propagation in TUI session lifecycle** (`tui/app.py`) -- active process definitions now persist as app state and are reapplied when creating or switching sessions, so cowork system prompts consistently include domain persona/tool guidance across `/new` and `/resume`.
- **Process-aware delegate orchestration in TUI** (`tui/app.py`) -- `delegate_task` factories created by the TUI now pass the active process definition into per-task orchestrators, aligning delegated execution with process rules in cowork mode.
- **Process required-tool enforcement** (`engine/orchestrator.py`, `api/routes.py`, `tui/app.py`) -- process runs now validate `tools.required` against the active registry. Missing requirements fail fast in orchestrator creation, are rejected early via API task creation (HTTP 400), and are surfaced as explicit TUI warnings.
- **`process.default` now applied across CLI and API entrypoints** (`__main__.py`, `api/routes.py`) -- when process is omitted, `loom`, `loom cowork`, `loom run`, and API task creation now use `config.process.default` if configured (explicit request process still wins).
- **In-session process controls in TUI** (`tui/app.py`) -- added `/process` command family for live process management: `/process` (status/usage), `/process list`, `/process use <name-or-path>`, and `/process off`.
- **Forced process execution slash command in TUI** (`tui/app.py`) -- added `/run <goal>` to execute the active process through `delegate_task` orchestration directly (instead of relying on cowork-mode inference). Includes usage guards when no process is active and appears in `/help` plus slash autocomplete/tab completion.
- **Dynamic process slash commands in TUI** (`tui/app.py`, `tui/commands.py`) -- selectable process definitions now expose direct slash commands (`/<process-name> <goal>`) that run with per-run process scoping (no global active-process mutation). Slash autocomplete/help and the `Ctrl+P` palette now surface these commands dynamically.
- **Concurrent process run tabs in TUI** (`tui/app.py`) -- `/run` now launches non-exclusive background workers with one dynamic tab per run, allowing cowork chat to remain interactive while process orchestrations execute. Run tabs show status indicators (`queued/running/completed/failed`) plus live elapsed timers and keep per-run progress/activity isolated instead of interleaving in the main chat panel.
- **Process-run tab close controls in TUI** (`tui/app.py`, `tui/screens/process_run_close.py`, `tui/commands.py`) -- added confirmed close flow for dynamic process tabs via `Ctrl+W`, command palette action, and `/run close [run-id-prefix]`. Closing an active run now cancels it and marks it failed before removing the tab.
- **Process progress visibility in TUI sidebar** (`tools/delegate_task.py`, `tui/app.py`, `tui/widgets/sidebar.py`) -- delegated process runs now publish structured subtask progress, and the sidebar Progress panel renders all process steps (including failed/skipped states) with a bounded, scrollable region.
- **Process-run context handoff from cowork session** (`tui/app.py`) -- delegated `/run` executions now include compact cowork context (workspace, recent user/assistant messages, focus, and recent decisions) so process runs can incorporate immediate chat context like “using this information…”.
- **Executor prompt now enforces exact process deliverable filenames** (`prompts/assembler.py`) -- when a process phase declares deliverables, the active subtask prompt now includes an explicit `REQUIRED OUTPUT FILES (EXACT FILENAMES)` section to reduce verification misses from renamed output files.
- **Process actions in command palette** (`tui/commands.py`, `tui/app.py`) -- added `Ctrl+P` actions for process inspection and control (`Show process info`, `List processes`, `Disable process`).
- **Process visibility in TUI status/session output** (`tui/widgets/status_bar.py`, `tui/app.py`) -- active process now appears in the bottom status bar (`process:<name>`) and `/session` output.
- **Installer now preserves full package assets with artifact filtering** (`processes/installer.py`) -- package installation now copies full process directories (templates/examples/docs and other assets), while excluding VCS/cache/bytecode artifacts such as `.git/`, `__pycache__/`, and `*.pyc`.
- **Installer isolated dependency mode** (`__main__.py`, `processes/installer.py`, `processes/schema.py`) -- added `loom install --isolated-deps`, which installs process dependencies into `<target>/.deps/<process-name>/` and activates those site-packages automatically when loading that process package.
- **Strict process phase-mode enforcement in planning** (`engine/orchestrator.py`) -- when a process declares `phase_mode: strict`, planner output is now normalized to the declared phase blueprint (IDs/order/dependencies/tier/acceptance), preventing drift from required process DAG structure.
- **Process runtime semantics tightened** (`engine/orchestrator.py`, `engine/scheduler.py`, `prompts/assembler.py`) -- `is_critical_path` phases now abort remaining pending work after retries are exhausted (instead of replanning), `is_synthesis` phases are held until all non-synthesis subtasks complete, and replanner prompts now include an explicit replan reason plus process `replanning.triggers` context.
- **Bundled tool collision diagnostics** (`processes/schema.py`) -- process loader now detects bundled tool name collisions with already-registered tools, logs actionable warnings, and skips only the colliding bundled tools to keep registry creation stable.
- **Google Analytics package install docs corrected** (`packages/google-analytics/README.md`) -- updated install example to a valid source format (`https://github.com/...`) accepted by `loom install`.
- **README TUI docs updated for process controls** (`README.md`) -- interface section now documents in-session process commands (`/process list`, `/process use`, `/process off`).
- **Process/package docs parity updates** (`README.md`, `docs/creating-packages.md`, `docs/agent-integration.md`) -- docs now reflect fail-fast `tools.required`, built-in phase modes, isolated dependency install mode, collision policy, and package troubleshooting/checklist guidance.
- **Process/package system refactor plan documented** (`planning/refactors/2026-02-17-PROCESS-PACKAGE-SYSTEM-PLAN.md`) -- added a prioritized execution plan covering TUI/API parity, strictness enforcement, package hardening, and demo-readiness criteria.
- **Web tool request resilience** (`tools/web.py`, `tools/web_search.py`) -- `web_fetch` and `web_search` now send explicit browser-compatible headers (with `LOOM_WEB_USER_AGENT` override), apply bounded retry/backoff for transient network/HTTP failures, and search now falls back across DuckDuckGo HTML endpoints before surfacing failure.
- **Tool registry creation now config-aware** (`tools/__init__.py`, `__main__.py`, `api/engine.py`, `api/routes.py`, `tui/app.py`) -- default registry builders now accept config so MCP tools are consistently available in CLI, TUI, API task validation, and process test runs.
- **CLI MCP layering and workspace-aware config resolution** (`__main__.py`) -- CLI startup now resolves active `loom.toml`, applies layered MCP config overlays (including `--mcp-config` and workspace override), and uses that merged config for `loom`, `cowork`, `run`, `serve`, and `mcp-serve`.
- **Process package author docs updated for test manifests** (`docs/creating-packages.md`, `README.md`) -- docs now cover `tests:` in `process.yaml`, `loom process test`, and MCP server config snippets.

### Fixed
- **Process deliverable verification scoping** (`engine/verification.py`) -- deterministic deliverable checks are now phase-scoped to the active subtask ID instead of flattening all process deliverables into every subtask verification, preventing early-phase false negatives in strict workflows.
- **Web tool verification brittleness** (`engine/verification.py`) -- deterministic verification now treats transient `web_fetch`/`web_search` failures (403/429/5xx/timeouts/connectivity) as advisory while still failing on safety/policy violations and malformed tool usage.
- **API process cross-task leakage** (`api/routes.py`, `api/engine.py`) -- process-backed task execution now uses an isolated per-task orchestrator instead of mutating shared orchestrator internals (`_process`, `_prompts`). Prevents process collisions and state bleed between concurrent tasks.
- **API process configuration parity** (`api/routes.py`) -- process loading now honors `config.process.search_paths`, matching TUI/CLI resolution behavior.
- **TUI files panel session bleed** (`tui/app.py`) -- file-change history/diff view is now reset on `/new` session creation and `/resume` session switching.
- **TUI streaming text crash on Textual API change** (`tui/widgets/chat_log.py`) -- streaming output no longer reads `Static.renderable` (missing in newer Textual). Chat log now tracks streamed text internally, preventing `AttributeError` during normal chat responses.
- **Streaming HTTP error crash in model providers** (`models/openai_provider.py`, `models/ollama_provider.py`, `models/anthropic_provider.py`) -- providers no longer access `response.text` on streaming `HTTPStatusError` responses (which can raise `Attempted to access streaming response content...`). Stream error bodies are now read before stream teardown, then surfaced via `ModelConnectionError` (including Anthropic API message extraction when present).
- **Slash autocomplete completeness in TUI** (`tui/app.py`) -- autocomplete now matches canonical commands and aliases consistently by prefix (and fallback substring), so prefixes like `/h`, `/n`, and `/l` reliably surface `/help`, `/new`, and `/learned`. The hint panel also allows more visible rows.
- **Exit confirmation modal key-capture deadlock** (`tui/screens/confirm_exit.py`, `tui/app.py`) -- exit confirm now explicitly handles `Enter`, `Esc`, `Y/N`, and `Ctrl+C` at the modal level, disables inherited app bindings while open, and de-duplicates concurrent confirmation prompts to prevent stacked modals and apparent freezes.
- **Quit binding modal deadlock** (`tui/app.py`) -- `Ctrl+C`, `/quit`, and command-palette quit now start the exit-confirm flow in a worker instead of awaiting it inline in the originating key/event handler, preventing message-loop stalls where the confirm modal appears but won't accept input.
- **Slash hint off-by-one input lag** (`tui/app.py`) -- slash autocomplete now reads the live input widget value on each change instead of relying on event payloads, preventing one-keystroke-late matches (e.g., `/h` showing `/` suggestions). Hint panel now resets scroll each update, and its height is explicitly set from rendered line count (capped) to avoid clipping the final match row.
- **Slash hint bottom-line clipping** (`tui/app.py`) -- while slash autocomplete is visible, both bottom bars (footer and status bar) are temporarily hidden so the hint panel is not occluded by docked widgets. This prevents the last suggestion row from disappearing (e.g., `/sessions`, `/tokens`) and fixes apparent blank single-match prefixes.
- **Slash hint overlap while typing command arguments** (`tui/app.py`) -- slash autocomplete hints now hide once argument entry begins (after first whitespace), so command completion UI no longer obscures active input context.
- **Tab-cycling slash autocomplete** (`tui/app.py`) -- when the chat input contains a slash command token, `Tab`/`Shift+Tab` now cycle matching commands (`/s` -> `/setup` -> `/session` -> `/sessions`). Outside slash autocomplete contexts, existing tab behavior is unchanged.
- **Workspace reload hotkey + palette action in TUI** (`tui/app.py`, `tui/commands.py`) -- added `Ctrl+R` to manually reload the sidebar workspace tree, plus a matching `Reload workspace tree` command in the `Ctrl+P` palette for discoverability.
- **Workspace tree refresh after writes** (`tui/app.py`, `tui/widgets/sidebar.py`) -- sidebar file tree now reloads after file-modifying tool calls and successful `/run` orchestrations so newly written files appear without restarting the TUI.
- **Process slash-command collision guard** (`tui/app.py`) -- process names that collide with built-in slash commands (e.g. `run`, `help`) are now blocked in TUI command indexing and `/process use`, with explicit user-facing warnings at startup.
- **Sidebar progress signal-to-noise for process runs** (`tui/app.py`) -- main HUD progress now renders concise summaries (one row per open process run plus a compact cowork-delegation summary) while detailed subtask/activity output remains inside each process tab.
- **Workspace tree refresh on subtask completion during `/run`** (`tools/delegate_task.py`, `tui/app.py`) -- `/run` now subscribes to orchestrator subtask events and triggers workspace tree reload as subtasks complete/fail, so process-generated files show up before full orchestration completes.
- **`/run` delegate binding regression in persisted TUI sessions** (`tui/app.py`) -- fixed tool-instance mismatch where auto-discovered `delegate_task` could remain unbound while TUI bound a different instance, causing `Task delegation is not available (no orchestrator configured)`.
- **`/run` timeout failure surfacing in progress panel** (`tui/app.py`) -- failed process runs now write an explicit failed status row in sidebar progress; timeout failures include guidance to increase `LOOM_DELEGATE_TIMEOUT_SECONDS`.
- **Long-running `/run` orchestration timeout increased and configurable** (`tools/delegate_task.py`) -- `delegate_task` timeout now defaults to 3600s and supports `LOOM_DELEGATE_TIMEOUT_SECONDS` override for multi-hour workflows.
- **Concurrent `/run` orchestrator isolation** (`tools/delegate_task.py`) -- `delegate_task` now creates a fresh orchestrator instance per execute call instead of reusing a cached singleton, preventing cross-run event/state coupling during parallel process runs.
- **TUI import regression for process-run close modal** (`tui/screens/__init__.py`, `tui/screens/process_run_close.py`) -- added and exported `ProcessRunCloseScreen` so TUI startup/tests no longer fail with `ImportError`.
- **Missing required tool args blocked before execution** (`models/router.py`, `prompts/templates/executor.yaml`) -- tool-call validation now enforces required schema fields (e.g. `document_write.path`, `spreadsheet.path`) and returns corrective guidance before dispatch; executor constraints now explicitly require path arguments for write/create tools.
- **Slash hint now surfaces dynamic process commands** (`tui/app.py`, `tests/test_tui.py`) -- slash popup now refreshes process command discovery while typing and uses a taller hint cap so `/`, `/process`, `/run`, and dynamic `/<process-name> <goal>` entries are visible instead of being clipped.
- **Workspace tree refresh on document tools** (`tui/app.py`, `tests/test_tui.py`) -- successful `document_write`/`document_create` tool completions now force a sidebar workspace tree refresh in chat and process-progress flows, so newly written docs appear immediately.
- **OpenAI-compatible role/message normalization for tool loops** (`models/openai_provider.py`, `engine/runner.py`, `cowork/session.py`, `tests/test_error_handling.py`) -- normalized assistant/tool messages now enforce non-null content, fallback tool-call IDs, and repaired tool-call references for stricter providers. Subtask runner reminder injections now use a user-role follow-up instead of mid-loop system-role messages to avoid provider role-order HTTP 400s.
- **Web fetch large-response handling** (`tools/web.py`, `engine/verification.py`) -- `web_fetch` now performs bounded streaming reads (no unbounded body load), returns truncated content instead of hard-failing on large pages, and deterministic verification treats `Response too large` web failures as advisory to avoid derailing entire subtasks.
- **Learned-pattern UX noise in TUI/CLI** (`tui/app.py`, `tui/screens/learned.py`, `__main__.py`) -- `/learned` and `loom learned` now default to behavioral patterns used for prompt personalization; internal operational telemetry patterns remain available via `loom learned --all`.
- **Setup and MCP config separation safety** (`tests/test_setup.py`) -- added regression coverage ensuring `loom setup` / `/setup` updates `loom.toml` without mutating a separate `mcp.toml`.
- **Workspace file viewer modal in TUI** (`tui/app.py`, `tui/screens/file_viewer.py`) -- selecting a file in the workspace tree now opens a read-only preview modal with extension-based renderer dispatch. Added renderers for Markdown, code/text with syntax highlighting (including TypeScript/CSS), JSON formatting, CSV/TSV table previews, HTML text extraction, diff/patch, Word/PowerPoint text extraction, PDF text preview, and image metadata.
- **Workspace file viewer backdrop dismissal** (`tui/screens/file_viewer.py`) -- clicking outside the centered file viewer dialog now closes the modal, matching expected modal behavior while preserving in-dialog interaction.
- **Tree-sitter CI gate** (`.github/workflows/ci.yml`) -- added a dedicated workflow job that installs `--extra treesitter`, asserts backend availability, and runs `tests/test_treesitter.py` so tree-sitter regressions are not silently skipped.
- **Test suite false-positive hardening** (`tests/test_tui.py`, `tests/test_setup.py`, `tests/test_installer.py`, `tests/test_integrations.py`, `tests/test_memory.py`, `tests/test_multimodal_integration.py`, `tests/test_processes.py`) -- replaced vacuous tests (attribute/assignment checks and no-op smoke cases) with behavior assertions that validate selectors, async setup invalidation, files-panel reset flows, dependency installer no-op behavior, event persistence error paths, idempotent DB initialization, runner no-bus emission, and bundled-tools no-op imports.
- **Tool auto-discovery collision with dynamic MCP proxy classes** (`tools/registry.py`, `integrations/mcp_tools.py`) -- dynamic MCP proxy tools are now explicitly excluded from global built-in auto-discovery to prevent constructor crashes during registry bootstrap.
- **Verifier JSON parse resilience in critical-path runs** (`models/router.py`, `engine/verification.py`, `tests/test_model_router.py`, `tests/test_verification.py`) -- JSON validation now extracts embedded JSON payloads from wrapped model output and prefers objects containing required keys (like `passed`) instead of failing on leading/trailing prose. Verifier feedback mapping now accepts `feedback` (prompt-native) with `suggestion` fallback, reducing inconclusive parse failures that previously aborted critical-path subtasks.

### Added
- **Process definition plugin architecture** (`processes/schema.py`) -- YAML-based domain specialization. Process definitions inject personas, phase blueprints, verification rules, tool guidance, and memory extraction types into the engine without code changes. Multi-path discovery (builtin → user-global → workspace-local), comprehensive validation with dependency cycle detection, and support for process packages that bundle tools.
- **5 built-in process definitions** (`processes/builtin/`) -- `investment-analysis` (5-phase strict financial workflow), `marketing-strategy` (6-phase guided GTM), `research-report` (4-phase research pipeline), `competitive-intel` (3-phase competitive analysis), `consulting-engagement` (5-phase McKinsey-style issue tree).
- **Process-aware verification** (`engine/verification.py`) -- DeterministicVerifier now runs process-specific regex rules and deliverables existence checks.
- **`calculator` tool** (`tools/calculator.py`) -- safe AST-based math evaluation with financial functions (NPV, CAGR, WACC, PMT). Exponent limits prevent OOM.
- **`spreadsheet` tool** (`tools/spreadsheet.py`) -- CSV operations: create, read, add rows/columns, update cells, summary. 5MB file size limit.
- **`document_write` tool** (`tools/document_write.py`) -- structured Markdown generation with sections, frontmatter metadata, and append mode.
- **`--process` CLI flag** (`__main__.py`) -- on `run` and `cowork` commands. Loads and applies a named process definition.
- **`loom processes` command** (`__main__.py`) -- lists all available process definitions with metadata.
- **`loom install` command** (`__main__.py`, `processes/installer.py`) -- install process packages from GitHub repos (`loom install user/repo`), URLs, or local paths. Validates structure, auto-installs Python dependencies (tries `uv` then `pip`), copies to `~/.loom/processes/` or workspace-local with `--workspace`.
- **`loom uninstall` command** (`__main__.py`, `processes/installer.py`) -- remove installed process packages by name. Built-in processes cannot be removed.
- **`dependencies` field in process.yaml** (`processes/schema.py`) -- declare pip packages that are auto-installed during `loom install`.
- **`[process]` config section** (`config.py`) -- `default` process and `search_paths` for additional process directories.
- **Session switching** (`__main__.py`) -- `/sessions`, `/new`, `/session` commands for mid-session switching between cowork sessions across workspaces.
- **EventBus unsubscribe** (`events/bus.py`) -- `unsubscribe()` and `unsubscribe_all()` methods to prevent handler leaks on SSE disconnect.

### Fixed

#### Security
- **SSRF via redirect** (`tools/web.py`) -- `follow_redirects=True` allowed redirects to private IPs, bypassing SSRF checks. Now validates each redirect target.
- **Webhook SSRF** (`events/webhook.py`) -- callback URLs are now validated against private network blocklist before registration.
- **Shell sandbox bypasses** (`tools/shell.py`) -- expanded blocked patterns to cover flag reordering (`rm -r -f /`), command substitution (`$(rm ...)`), and interpreter flags (`python -c`, `perl -e`).
- **Git config removed** (`tools/git.py`) -- `config` subcommand removed to prevent arbitrary code execution via hooks/aliases.
- **Approval gate defaults** (`cowork/approval.py`, `recovery/approval.py`) -- no-callback now denies by default; timeout denies instead of auto-approving.
- **Delegate task approval** (`tools/delegate_task.py`) -- default changed from `"auto"` to `"confidence_threshold"`.
- **Path traversal** (`tools/glob_find.py`, `tools/ripgrep.py`, `tools/workspace.py`) -- workspace containment checks, snapshot path validation.
- **Prompt injection** (`prompts/assembler.py`) -- replaced `.format()` then `{→${` corruption with safe per-key replacement.
- **Task ID validation** (`__main__.py`) -- regex validation prevents path traversal in URL interpolation.

#### Critical Bugs
- **SubtaskResult field mismatch** (`engine/orchestrator.py`) -- exception handler used non-existent `output`/`error` fields instead of `summary`. Would crash on any parallel subtask exception.
- **VerificationResult empty instantiation** (`engine/orchestrator.py`) -- missing required `tier` and `passed` fields. Would crash on parallel exception handling.
- **Attempt tracking off-by-one** (`engine/orchestrator.py`) -- stale variable caused wrong retry counts and escalation tier calculations.
- **Iteration counter * batch size** (`engine/orchestrator.py`) -- `max_iterations` was effectively divided by parallelism.
- **Verification gate bypass** (`engine/verification.py`) -- gates returned `passed=True` when disabled or verifier unavailable.
- **Data loss in schema** (`state/schema.sql`) -- removed `UNIQUE(session_id, turn_number)` constraint that silently dropped messages.
- **Text loss in send()** (`cowork/session.py`) -- only last iteration's text was kept in tool loop; now accumulates across iterations.
- **Memory extractor format mismatch** (`engine/runner.py`) -- parser expected `{"entries": [...]}` but template asked for JSON arrays.

#### High Severity
- **MCP JSON parse crashes** (`integrations/mcp_server.py`) -- all `.json()` calls now wrapped in try-except; error handler fixed from `"request" in dir()` to proper local variable.
- **Background task error loss** (`api/routes.py`) -- uncaught exceptions now logged and task marked failed instead of silently swallowed.
- **SSE handler leak** (`api/routes.py`) -- event handlers now unsubscribed in `finally` blocks on client disconnect.
- **Ripgrep process leak** (`tools/ripgrep.py`) -- subprocess now killed on timeout instead of left running.
- **Shell output OOM** (`tools/shell.py`) -- bounded output buffer to 1MB to prevent memory exhaustion from `yes` etc.
- **OpenAI/Ollama parsing** (`models/openai_provider.py`, `models/ollama_provider.py`) -- empty choices validated, `json.loads()` wrapped in try-except, defensive `.get()` for all dict access.
- **Session lifecycle** (`__main__.py`) -- session marked inactive on `/quit` and Ctrl-C; `config.data_dir` replaced with `config.workspace.scratch_dir`.
- **PromptAssembler init** (`__main__.py`) -- was passing `config` as `templates_dir`.

#### Medium Severity
- **Fire-and-forget persist** (`cowork/session.py`) -- `asyncio.ensure_future` replaced with awaited async call to prevent data loss.
- **Transactional batch insert** (`state/memory.py`) -- `store_many()` now runs in a single transaction.
- **LIKE wildcard injection** (`state/memory.py`, `state/conversation_store.py`) -- `%` and `_` in search queries escaped.
- **Web fetch memory** (`tools/web.py`) -- Content-Length checked before downloading to prevent OOM.
- **Memory extraction logging** (`engine/runner.py`) -- silent failures now logged at DEBUG level.
- **SQLite WAL mode** (`state/memory.py`) -- enabled for concurrent read/write safety.
- **SessionState resilience** (`cowork/session_state.py`) -- handles malformed JSON and `files_touched` entries.
- **TaskTracker locking** (`tools/task_tracker.py`) -- async lock for concurrent safety.
- **Fallback plan** (`engine/orchestrator.py`) -- includes actual goal text instead of generic description.

#### Low Severity
- **EditFileTool empty old_str** (`tools/file_ops.py`) -- rejected to prevent nonsensical replacements.
- **WriteFileTool size limit** (`tools/file_ops.py`) -- 1MB content limit.
- **_glob_match** (`tools/search.py`) -- replaced simplistic pattern with `fnmatch.fnmatch`.
- **Anthropic unknown roles** (`models/anthropic_provider.py`) -- logged and preserved instead of silently dropped.
- **Summary truncation** (`state/task_state.py`) -- increased from 100 to 200 chars with `...` indicator.
- **Dead code removed** (`cowork/session.py`) -- unused `_is_safe_cut_point` method.
- **Display bug** (`cowork/display.py`) -- operator precedence fix.
- **Port 0 falsy** (`__main__.py`) -- `port if port is not None` instead of `port if port`.
- **API key redaction** (`config.py`) -- `ModelConfig.__repr__` shows only last 4 chars.
- **delegate_task files** (`tools/delegate_task.py`) -- reports change counts instead of returning timestamp as path.

### Previously added
- **Cowork mode** (`cowork/session.py`) -- conversation-first interactive execution. No planning phase, no subtask decomposition -- just a continuous tool-calling loop driven by natural conversation with the developer. Full conversation history maintained as context. Now the default `loom` interface via the unified TUI.
- **`loom cowork` CLI command** -- interactive session with real-time tool call display and special commands (`/quit`, `/help`, `/sessions`, `/new`, `/resume`). Usage: `loom -w /path/to/project` (or `loom cowork -w /path/to/project`).
- **`ask_user` tool** -- lets the model ask the developer questions mid-execution instead of guessing. Supports free-text and multiple-choice options. The cowork CLI intercepts these and prompts the user.
- **`ripgrep_search` tool** -- fast content search that shells out to `rg` (ripgrep). Falls back to `grep`, then pure Python. Supports regex, file type filtering, context lines, case insensitivity, and files-only mode.
- **`glob_find` tool** -- fast file discovery by glob pattern (`**/*.py`, `src/**/*.ts`). Results sorted by modification time, automatically skips `.git`, `node_modules`, `__pycache__`, etc.
- **`web_search` tool** -- internet search via DuckDuckGo (no API key required). Returns titles, URLs, and snippets. Use to find docs, solutions, package info, etc.
- **Anthropic/Claude model provider** (`models/anthropic_provider.py`) -- full Claude API support via the Messages API. Native tool use, message format conversion (OpenAI <-> Anthropic), and streaming via SSE. Configure with `provider = "anthropic"` in `loom.toml`.
- **`api_key` and `tier` fields** in `ModelConfig` -- supports API-key-authenticated providers and explicit tier assignment.
- **Gap analysis document** (`planning/gap-analysis-vs-claude-code.md`) -- 10-dimension comparison of Loom vs Claude Code's coworking model with prioritized implementation roadmap.

- **Per-tool-call approval** (`cowork/approval.py`) -- interactive approval system for cowork mode. Read-only tools (read_file, search, glob, web_search, etc.) auto-approved. Write/execute tools (shell, git, edit, delete) prompt with `[y]es / [n]o / [a]lways allow <tool>`. "Always" remembers per-tool for the session.
- **`task_tracker` tool** -- in-memory progress tracking for multi-step tasks. Actions: add, update (pending/in_progress/completed), list, clear. Helps the model organize complex work and show progress.
- **PDF/image/office file support** in `read_file` -- PDFs: extracts text page-by-page via `pypdf` (optional dep). Images: returns multimodal content blocks. Word (.docx) and PowerPoint (.pptx): extracts text via `python-docx` and `python-pptx` (required deps). All fall back gracefully to text.

### Changed
- **TUI is the unified interactive interface** (`tui/app.py`) -- runs `CoworkSession` directly with full persistence, streaming text, tool approval modals, `ask_user` modals, conversation recall, and task delegation. Launched as the default `loom` command.
- **Streaming by default** -- text tokens display incrementally as they arrive instead of waiting for the full response.
- **Git tool** -- `push` and `remote` added to allowed subcommands (force push still blocked).
- **Shell tool** -- timeout increased from 60s to 120s for longer-running commands.
- **Tool output limit** -- increased from 10KB to 30KB for richer tool results.
- **Model router** -- wired to support `"anthropic"` provider type alongside `"ollama"` and `"openai_compatible"`.

---

## [0.4.0] -- Plugin Auto-Discovery & Parallel Dispatch

### Added
- **Plugin auto-discovery for tools** -- `Tool` base class now uses `__init_subclass__` to automatically collect concrete subclasses. New `discover_tools()` function scans all modules in `loom.tools` via `pkgutil.walk_packages`. Adding a new tool is now just: create a file, subclass `Tool`, done.
- **SubtaskRunner** (`engine/runner.py`) -- isolated subtask execution class that owns the tool-calling loop, response validation, verification gates, and memory extraction. The orchestrator no longer touches raw prompts or messages.
- **Parallel subtask dispatch** -- independent subtasks (no unmet `depends_on`) now execute concurrently via `asyncio.gather()`, up to `max_parallel_subtasks` (default 3). Sequential dependency chains are unaffected.
- **Fire-and-forget memory extraction** -- memory entries are extracted as background tasks, no longer blocking the next subtask from starting.
- **`max_parallel_subtasks` config** -- new `[execution]` setting to control concurrency (default 3, set to 1 for fully sequential behavior).
- **`asyncio.Lock` for state saves** -- prevents race conditions when concurrent subtasks complete simultaneously.
- **Full end-to-end integration test suite** (`tests/test_full_integration.py`) -- 11 tests exercising the entire orchestration pipeline with real components and a deterministic `FakeModelProvider`:
  - Happy-path lifecycle (plan, execute, verify, memory, learn, complete)
  - Multi-subtask dependency chains
  - Verification failure with retry and escalation
  - All retries exhausted leading to task failure
  - Text-only subtasks (no tool calls)
  - Hallucinated tool name handling
  - Edit file with changelog tracking
  - Event persistence roundtrip (SQLite)
  - Memory extraction and retrieval roundtrip
  - Parallel independent subtask execution
  - `max_parallel_subtasks=1` sequential fallback
- **Agent connection documentation** (`docs/agent-integration.md`) -- comprehensive guide covering REST API (poll/SSE/webhook), MCP server (stdio/SSE), direct Python engine usage, custom model providers, custom tools, event subscriptions, approval modes, and configuration.

### Changed
- **`create_default_registry()` refactored** -- no longer manually lists every tool. Now calls `discover_tools()` which auto-imports tool modules and instantiates discovered classes. Public API unchanged.
- **Orchestrator refactored** -- reduced from ~714 lines handling 15+ responsibilities to ~586 lines focused on lifecycle management, scheduling, retry/replan decisions, approval gating, and event emission. Subtask execution delegated to `SubtaskRunner`.
- **Architecture diagram** updated in README to reflect parallel dispatch and runner separation.
- **Documentation updated** -- README, tutorial.html, agent-integration.md, and 05-TOOL-SYSTEM.md now list all 11 built-in tools and describe the plugin discovery mechanism.

---

## [0.3.0] -- Phase 2 Complete

### Added
- **TUI client** (`tui/app.py`) -- Textual-based terminal dashboard with live SSE streaming, task detail views, approval/rejection modals, and steering input.
- **MCP server** (`integrations/mcp_server.py`) -- Model Context Protocol server exposing `loom_execute_task`, `loom_task_status`, and `loom_list_tasks` tools via stdio or SSE transport.
- **Learning system** (`learning/manager.py`) -- pattern extraction from execution history: subtask success patterns, retry patterns, model failure patterns, and task templates. Queryable by type.
- **Confidence scoring** (`recovery/confidence.py`) -- weighted scoring (verification 40%, tool success 30%, model quality 15%, complexity 15%) with band classification (high/medium/low/zero).
- **Approval gates** (`recovery/approval.py`) -- auto, manual, and confidence-threshold modes. Always-gate for destructive operations (`rm -rf`, `sudo`, `.env` writes).
- **Retry escalation** (`recovery/retry.py`) -- automatic tier escalation ladder (same tier -> next tier -> highest tier -> human flag).
- **Webhook delivery** (`events/webhook.py`) -- callback URL notification on task completion/failure with exponential backoff retry.
- **Event persistence** (`events/bus.py`) -- `EventPersister` subscribes to all events and writes to SQLite for audit and replay.

### Changed
- Orchestrator wired with verification gates, changelog tracking, memory extraction, and re-planning.

---

## [0.2.0] -- Phase 1 Complete

### Added
- **Orchestrator loop** (`engine/orchestrator.py`) -- core agentic loop: plan -> execute subtasks -> verify -> finalize. The harness drives work, not the model.
- **API server** (`api/server.py`, `api/routes.py`) -- FastAPI REST endpoints for task CRUD, SSE streaming, steer/approve/feedback. Interactive docs at `/docs`.
- **Task state management** (`state/task_state.py`) -- YAML-based task state with atomic file writes. Compact prompt injection format.
- **Memory layer** (`state/memory.py`) -- SQLite archive with task/subtask/type/tag queries and full-text search.
- **Prompt templates** (`prompts/assembler.py`, `prompts/templates/`) -- 7-section prompt assembly from YAML templates: role, task state, subtask, memory, tools, output format, constraints.
- **Model router** (`models/router.py`) -- role + tier model selection with Ollama and OpenAI-compatible providers.
- **Tool system** (`tools/`) -- registry with timeout, file read/write/edit, shell execution (with safety blocklist), file search, directory listing.
- **Workspace management** (`tools/workspace.py`) -- changelog with before-snapshots, diff generation, revert at file/subtask/task level. Path traversal prevention.
- **Scheduler** (`engine/scheduler.py`) -- dependency graph resolution for subtask ordering.
- **Verification gates** (`engine/verification.py`) -- three-tier verification: deterministic checks (syntax, file existence), independent LLM review, multi-vote verification.
- **Event bus** (`events/bus.py`) -- in-process async pub/sub for real-time updates.
- **CLI** (`__main__.py`) -- Click-based CLI with default TUI, serve, run, status, cancel, models, cowork, mcp-serve commands.

---

## [0.1.0] -- Project Scaffold

### Added
- Project structure: `pyproject.toml`, CI configuration, test infrastructure.
- 15 specification documents covering all system components.
- `loom.toml` configuration loader with sensible defaults.
