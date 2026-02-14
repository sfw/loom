# Loom

**Local model orchestration engine** -- task decomposition, execution, and verification using local LLMs.

Loom takes a high-level goal, breaks it into subtasks, executes them with local models (Ollama, OpenAI-compatible APIs), verifies results, and learns from the process. The model never decides to "continue" -- the harness does.

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
- **Dual model backends** -- Ollama and OpenAI-compatible APIs (LM Studio, vLLM, etc.)
- **Role-based routing** -- planner, executor, extractor, verifier roles with tier selection
- **Tool system** -- file read/write/edit, shell execution (with safety blocklist), search
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

- **REST API** -- full task CRUD, SSE streaming, steer/approve/feedback
- **Terminal UI** -- Textual-based dashboard with live streaming, steering, and approval modals
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

# Or launch the terminal UI
loom tui
```

## CLI Commands

```
loom serve              Start the API server
loom run GOAL           Submit a task and stream progress inline
loom status ID          Check status of a task
loom cancel ID          Cancel a running task
loom models             List configured models
loom tui                Launch the terminal UI dashboard
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
provider = "ollama"                    # or "openai_compatible"
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
    registry.py          Tool dispatch with timeout
    file_ops.py          Read, write, edit files
    shell.py             Shell execution with safety
    search.py            File search and directory listing
    workspace.py         Changelog, diff, revert
  tui/
    api_client.py        Async HTTP + SSE client for Loom API
    app.py               Textual TUI (dashboard, detail views, modals)
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
- A local model backend: [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or any OpenAI-compatible API

## License

MIT
