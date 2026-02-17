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
- **Process required-tool enforcement** (`engine/orchestrator.py`, `tui/app.py`) -- process runs now validate `tools.required` against the active registry. Missing requirements fail fast in orchestrator creation and are surfaced as explicit TUI warnings.
- **`process.default` now applied by CLI launch paths** (`__main__.py`) -- when `--process` is omitted, `loom`, `loom cowork`, and `loom run` now automatically use `config.process.default` if configured.
- **In-session process controls in TUI** (`tui/app.py`) -- added `/process` command family for live process management: `/process` (status/usage), `/process list`, `/process use <name-or-path>`, and `/process off`.
- **Process visibility in TUI status/session output** (`tui/widgets/status_bar.py`, `tui/app.py`) -- active process now appears in the bottom status bar (`process:<name>`) and `/session` output.
- **Process/package system refactor plan documented** (`planning/refactors/2026-02-17-PROCESS-PACKAGE-SYSTEM-PLAN.md`) -- added a prioritized execution plan covering TUI/API parity, strictness enforcement, package hardening, and demo-readiness criteria.

### Fixed
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
- **Tab-cycling slash autocomplete** (`tui/app.py`) -- when the chat input contains a slash command token, `Tab`/`Shift+Tab` now cycle matching commands (`/s` -> `/setup` -> `/session` -> `/sessions`). Outside slash autocomplete contexts, existing tab behavior is unchanged.
- **Tree-sitter CI gate** (`.github/workflows/ci.yml`) -- added a dedicated workflow job that installs `--extra treesitter`, asserts backend availability, and runs `tests/test_treesitter.py` so tree-sitter regressions are not silently skipped.
- **Test suite false-positive hardening** (`tests/test_tui.py`, `tests/test_setup.py`, `tests/test_installer.py`, `tests/test_integrations.py`, `tests/test_memory.py`, `tests/test_multimodal_integration.py`, `tests/test_processes.py`) -- replaced vacuous tests (attribute/assignment checks and no-op smoke cases) with behavior assertions that validate selectors, async setup invalidation, files-panel reset flows, dependency installer no-op behavior, event persistence error paths, idempotent DB initialization, runner no-bus emission, and bundled-tools no-op imports.

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
