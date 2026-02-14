# Changelog

All notable changes to Loom are documented in this file.

## [Unreleased]

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
- **CLI** (`__main__.py`) -- Click-based CLI with serve, run, status, cancel, models, tui, mcp-serve commands.

---

## [0.1.0] -- Project Scaffold

### Added
- Project structure: `pyproject.toml`, CI configuration, test infrastructure.
- 15 specification documents covering all system components.
- `loom.toml` configuration loader with sensible defaults.
