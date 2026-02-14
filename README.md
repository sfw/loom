# Loom

**Local model orchestration engine** -- task decomposition, execution, and verification using local LLMs.

Loom takes a high-level goal, breaks it into subtasks, executes them with local models (Ollama, OpenAI-compatible APIs, or Claude), verifies results, and learns from the process. The model never decides to "continue" -- the harness does.

**Two modes of operation:**
- **Cowork mode** (`loom cowork`) -- interactive pair programming. You and the AI have a conversation, it uses tools in real time, you can interrupt and redirect. No planning overhead.
- **Task mode** (`loom run`) -- autonomous execution. Submit a goal, Loom decomposes it into subtasks, executes, verifies, and reports back.

## How It Works

```
                    ┌──────────────────────────┐
Goal -> Planner ->  │ [Subtask A]  [Subtask B] │  parallel batch
                    │     |             |       │  (if independent)
                    │  Execute       Execute    │
                    │  Verify        Verify     │
                    │  Extract*      Extract*   │  * fire-and-forget
                    └──────────────────────────┘
                               |
                         [Subtask C]  (depends on A+B)
                              |
                          Completed
```

1. **Plan** -- A planner model decomposes the goal into ordered subtasks with a dependency graph
2. **Schedule** -- Independent subtasks are dispatched in parallel (up to `max_parallel_subtasks`)
3. **Execute** -- Each subtask runs in an isolated `SubtaskRunner` with its own tool-calling loop
4. **Verify** -- An independent verifier checks each result against acceptance criteria
5. **Extract** -- Key decisions, errors, and discoveries are extracted into structured memory (fire-and-forget)
6. **Replan** -- If subtasks fail or new information emerges, the plan is revised

## Features

**Core orchestration:**

- **Task decomposition** with dependency graphs and automatic scheduling
- **Parallel subtask execution** -- independent subtasks run concurrently (configurable `max_parallel_subtasks`)
- **Isolated execution** -- each subtask runs in a `SubtaskRunner` with its own context (no cross-contamination)
- **Three model backends** -- Ollama, OpenAI-compatible APIs (LM Studio, vLLM, etc.), and Anthropic/Claude
- **Role-based routing** -- planner, executor, extractor, verifier roles with tier selection
- **Tool system** -- 16 built-in tools (file ops, shell, git, ripgrep search, glob find, web search, web fetch, code analysis, task tracker, ask user) with plugin auto-discovery
- **Workspace safety** -- path traversal prevention, destructive command blocking
- **Full undo** -- changelog with before-snapshots, revert at file/subtask/task level
- **Token budgeting** -- prompt assembly with 7-section ordering and trim-to-budget

**Verification and recovery:**

- **Three-tier verification** -- deterministic checks, independent LLM review, voting verification
- **Confidence scoring** -- weighted scoring with band classification (high/medium/low/zero)
- **Approval gates** -- auto, manual, or confidence-threshold modes with always-gate for destructive ops
- **Retry escalation** -- automatic tier escalation ladder with human flagging after max attempts
- **Re-planning** -- automatic re-planning when subtasks fail or new information emerges

**State and events:**

- **Structured memory** -- SQLite archive with task/subtask/type/tag queries
- **Anti-amnesia** -- TODO reminders injected after every tool call
- **Event bus** -- pub/sub for real-time updates to any number of clients
- **Event persistence** -- all events persisted to SQLite for audit and replay
- **Webhook delivery** -- callback URLs notified on task completion/failure with retry

**Interfaces:**

- **Cowork mode** -- interactive conversation loop with streaming, real-time tool display, per-tool-call approval, and full context
- **REST API** -- full task CRUD, SSE streaming, steer/approve/feedback
- **Terminal UI** -- Textual-based cowork interface with streaming chat, tool approval modals ([y]es/[a]lways/[n]o), ask_user modals, and scrollable chat log. Runs standalone (no server required).
- **MCP server** -- Model Context Protocol integration for use as an agent tool
- **Learning system** -- pattern extraction from execution history (success patterns, retry hints, templates)

## Quick Start

```bash
# Install (using uv, recommended)
uv sync

# Or with pip
pip install -e .

# Configure models
mkdir -p ~/.loom
cp loom.toml ~/.loom/loom.toml
# Edit ~/.loom/loom.toml with your model endpoints

# Start the server
loom serve

# Submit a task (in another terminal)
curl -X POST http://localhost:9000/tasks \
  -H "Content-Type: application/json" \
  -d '{"goal": "Create a Python CLI that converts CSV to JSON", "workspace": "/tmp/myproject"}'

# Or use the CLI
loom run "Create a Python CLI that converts CSV to JSON" --workspace /tmp/myproject

# Or start an interactive cowork session
loom cowork -w /tmp/myproject

# Or launch the Textual TUI (same features, richer interface)
loom tui -w /tmp/myproject
```

## CLI Commands

```
loom serve              Start the API server
loom cowork             Start an interactive cowork session (pair programming, CLI)
loom tui                Launch the Textual TUI (cowork with modals and scrollback)
loom run GOAL           Submit a task and stream progress inline
loom status ID          Check status of a task
loom cancel ID          Cancel a running task
loom models             List configured models
loom mcp-serve          Start the MCP server (stdio transport)
loom reset-learning     Clear all learned patterns
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks` | Create and start a task |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/tasks/{id}` | Get full task state |
| `GET` | `/tasks/{id}/stream` | SSE event stream |
| `PATCH` | `/tasks/{id}` | Inject instructions (steer) |
| `DELETE` | `/tasks/{id}` | Cancel a task |
| `POST` | `/tasks/{id}/approve` | Approve a gated step |
| `POST` | `/tasks/{id}/feedback` | Provide mid-task feedback |
| `GET` | `/tasks/{id}/subtasks` | List subtasks |
| `GET` | `/tasks/{id}/memory` | Query task memory |
| `GET` | `/memory/search` | Search across all memory |
| `GET` | `/models` | Available models |
| `GET` | `/tools` | Available tools |
| `GET` | `/health` | Health check |
| `GET` | `/config` | Current configuration |

Interactive API docs are available at `http://localhost:9000/docs` when the server is running.

## Configuration

Loom reads `loom.toml` from the current directory or `~/.loom/loom.toml`:

```toml
[server]
host = "127.0.0.1"
port = 9000

[models.primary]
provider = "ollama"                    # or "openai_compatible" or "anthropic"
base_url = "http://localhost:11434"
model = "qwen3:14b"
max_tokens = 4096
temperature = 0.1
roles = ["planner", "executor"]

[models.utility]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:8b"
max_tokens = 2048
temperature = 0.0
roles = ["extractor", "verifier"]

# Optional: Anthropic/Claude as an additional or alternative provider
# [models.claude]
# provider = "anthropic"
# model = "claude-sonnet-4-5-20250929"
# api_key = "sk-ant-..."               # or set ANTHROPIC_API_KEY env var
# max_tokens = 8192
# tier = 3
# roles = ["executor", "planner"]

[workspace]
default_path = "~/projects"
scratch_dir = "~/.loom/scratch"

[execution]
max_subtask_retries = 3
max_loop_iterations = 50
max_parallel_subtasks = 3    # Independent subtasks run concurrently

[verification]
tier1_enabled = true    # Deterministic checks (syntax, file existence)
tier2_enabled = true    # Independent LLM verification
tier3_enabled = false   # Multi-vote verification (expensive)

[memory]
database_path = "~/.loom/loom.db"
```

See [INSTALL.md](INSTALL.md) for detailed setup instructions.

## Architecture

```
src/loom/
  __main__.py            CLI entry point (Click)
  config.py              TOML config loader
  api/
    server.py            FastAPI app factory
    routes.py            All REST endpoints
    schemas.py           Pydantic request/response models
    engine.py            Component wiring and lifecycle
  cowork/
    session.py           Conversation-first interactive execution engine
    approval.py          Per-tool-call approval (auto/approve/always/deny)
    display.py           Terminal display with ANSI colors for tool calls
  engine/
    orchestrator.py      Core loop: plan -> schedule -> dispatch -> finalize
    runner.py            Isolated subtask execution (tool loop, verify, extract)
    scheduler.py         Dependency-based subtask ordering + parallel batch selection
    verification.py      Three-tier verification gates
  events/
    bus.py               In-process pub/sub + event persistence
    types.py             Event type constants
    webhook.py           Callback URL delivery with retry
  integrations/
    mcp_server.py        Model Context Protocol server (3 tools)
  learning/
    manager.py           Pattern extraction and query from execution history
  models/
    base.py              Provider ABC, response types
    anthropic_provider.py  Anthropic/Claude API client
    ollama_provider.py   Ollama API client
    openai_provider.py   OpenAI-compatible API client
    router.py            Role+tier model selection
  prompts/
    assembler.py         7-section prompt builder
    constraints.py       Safety and behavior constraints
    templates/           YAML prompt templates
  recovery/
    approval.py          Approval gates (auto/manual/threshold)
    confidence.py        Weighted confidence scoring with band classification
    retry.py             Retry escalation ladder with tier promotion
  state/
    task_state.py        Task/Subtask dataclasses, YAML state manager
    memory.py            SQLite memory archive
    schema.sql           Database schema
  tools/
    registry.py          Tool ABC with auto-discovery via __init_subclass__
    file_ops.py          Read, write, edit, delete, move files
    shell.py             Shell execution with safety
    git.py               Git operations with allowlist (incl. push)
    search.py            File search and directory listing
    ripgrep.py           Ripgrep-powered content search with fallbacks
    glob_find.py         Fast file discovery by glob pattern
    ask_user.py          Ask the developer questions mid-execution
    code_analysis.py     Code structure analysis (tree-sitter)
    web.py               Web fetch with URL safety
    web_search.py        Internet search via DuckDuckGo (no API key)
    task_tracker.py      Progress tracking for multi-step tasks
    workspace.py         Changelog, diff, revert
  tui/
    app.py               Textual TUI (cowork chat, approval/ask_user modals)
    api_client.py        Async HTTP + SSE client (legacy server-mode)
```

## Development

```bash
# Install with dev dependencies (using uv)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/ tests/

# Run tests with coverage
pytest --cov=loom --cov-report=term-missing
```

## Requirements

- Python 3.11+
- A model backend: [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), any OpenAI-compatible API, or [Anthropic/Claude](https://console.anthropic.com)

## License

MIT
