# Loom

**Local model orchestration engine** -- task decomposition, execution, and verification using local LLMs.

Loom takes a high-level goal, breaks it into subtasks, executes them with local models (Ollama, OpenAI-compatible APIs), verifies results, and learns from the process. The model never decides to "continue" -- the harness does.

## How It Works

```
Goal -> Planner -> [Subtask 1] -> [Subtask 2] -> ... -> Completed
                       |              |
                    Execute         Execute
                    Verify          Verify
                    Extract         Extract
```

1. **Plan** -- A planner model decomposes the goal into ordered subtasks with dependencies
2. **Execute** -- Each subtask runs in a tool-calling loop (read files, write files, run shell commands)
3. **Verify** -- An independent verifier checks each result against acceptance criteria
4. **Extract** -- Key decisions, errors, and discoveries are extracted into structured memory
5. **Replan** -- If subtasks fail or new information emerges, the plan is revised

## Features

- **Task decomposition** with dependency graphs and automatic scheduling
- **Dual model backends** -- Ollama and OpenAI-compatible APIs (LM Studio, vLLM, etc.)
- **Role-based routing** -- planner, executor, extractor, verifier roles with tier selection
- **Tool system** -- file read/write/edit, shell execution (with safety blocklist), search
- **Workspace safety** -- path traversal prevention, destructive command blocking
- **Full undo** -- changelog with before-snapshots, revert at file/subtask/task level
- **Structured memory** -- SQLite archive with task/subtask/type/tag queries
- **Anti-amnesia** -- TODO reminders injected after every tool call
- **REST API** -- full task CRUD, SSE streaming, steer/approve/feedback
- **Event bus** -- pub/sub for real-time updates to any number of clients
- **Token budgeting** -- prompt assembly with 7-section ordering and trim-to-budget

## Quick Start

```bash
# Install
pip install -e .

# Configure models
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
```

## CLI Commands

```
loom serve          Start the API server
loom run GOAL       Submit a task and stream progress inline
loom status ID      Check status of a task
loom cancel ID      Cancel a running task
loom models         List configured models
loom tui            Launch terminal UI (coming soon)
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

[memory]
database_path = "~/.loom/loom.db"
```

See [INSTALL.md](INSTALL.md) for detailed setup instructions.

## Architecture

```
src/loom/
  __main__.py          CLI entry point (Click)
  config.py            TOML config loader
  api/
    server.py          FastAPI app factory
    routes.py          All REST endpoints
    schemas.py         Pydantic request/response models
    engine.py          Component wiring and lifecycle
  engine/
    orchestrator.py    Core loop: plan -> execute -> finalize
    scheduler.py       Dependency-based subtask ordering
  events/
    bus.py             In-process pub/sub
    types.py           Event type constants
  models/
    base.py            Provider ABC, response types
    ollama_provider.py Ollama API client
    openai_provider.py OpenAI-compatible API client
    router.py          Role+tier model selection
  prompts/
    assembler.py       7-section prompt builder
    constraints.py     Safety and behavior constraints
    templates/         YAML prompt templates
  state/
    task_state.py      Task/Subtask dataclasses, YAML state manager
    memory.py          SQLite memory archive
    schema.sql         Database schema
  tools/
    registry.py        Tool dispatch with timeout
    file_ops.py        Read, write, edit files
    shell.py           Shell execution with safety
    search.py          File search and directory listing
    workspace.py       Changelog, diff, revert
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/ tests/

# Run tests with coverage
coverage run -m pytest && coverage report
```

## Requirements

- Python 3.11+
- A local model backend: [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), or any OpenAI-compatible API

## License

MIT
