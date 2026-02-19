# Spec 01: Project Structure

## Overview

Repository layout, dependencies, configuration files, and development environment for Loom.

## Directory Structure

```
loom/
├── pyproject.toml                  # Project metadata, dependencies
├── loom.toml                 # Runtime configuration (user-editable)
├── README.md
│
├── src/
│   └── loom/
│       ├── __init__.py
│       ├── __main__.py             # CLI entry point
│       ├── config.py               # Configuration loader (TOML)
│       │
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── orchestrator.py     # Core agentic loop (Spec 02)
│       │   ├── planner.py          # Task decomposition and re-planning
│       │   ├── executor.py         # Single subtask execution
│       │   └── scheduler.py        # Subtask ordering, dependency resolution
│       │
│       ├── state/
│       │   ├── __init__.py
│       │   ├── task_state.py       # Always-in-context YAML state object
│       │   ├── memory.py           # SQLite memory manager (Spec 03)
│       │   └── schema.sql          # SQLite schema definitions
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── router.py           # Model selection and routing (Spec 04)
│       │   ├── base.py             # Abstract model interface
│       │   ├── mlx_provider.py     # MLX/LM Studio backend
│       │   ├── ollama_provider.py  # Ollama backend
│       │   └── openai_provider.py  # OpenAI-compatible API backend
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── registry.py         # Tool registration and dispatch (Spec 05)
│       │   ├── file_ops.py         # Read, write, edit files
│       │   ├── shell.py            # Execute shell commands
│       │   ├── search.py           # Grep, find, search files
│       │   └── workspace.py        # Workspace management (Spec 11)
│       │
│       ├── verification/
│       │   ├── __init__.py
│       │   ├── gates.py            # Verification gate orchestrator (Spec 06)
│       │   ├── deterministic.py    # Tier 1: schema, type, existence checks
│       │   ├── llm_verify.py       # Tier 2: independent LLM verification
│       │   └── voting.py           # Tier 3: multi-run voting verification
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── server.py           # FastAPI application (Spec 07)
│       │   ├── routes_tasks.py     # Task CRUD + streaming endpoints
│       │   ├── routes_system.py    # Health, models, tools endpoints
│       │   ├── sse.py              # SSE stream manager
│       │   └── schemas.py          # Pydantic request/response models
│       │
│       ├── events/
│       │   ├── __init__.py
│       │   ├── bus.py              # Event bus implementation (Spec 08)
│       │   ├── types.py            # Event type definitions
│       │   └── logger.py           # Event persistence to SQLite
│       │
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── assembler.py        # Context assembly engine (Spec 12)
│       │   ├── templates/
│       │   │   ├── planner.yaml    # Task decomposition prompt
│       │   │   ├── executor.yaml   # Subtask execution prompt
│       │   │   ├── verifier.yaml   # Verification prompt
│       │   │   ├── extractor.yaml  # Memory extraction prompt
│       │   │   └── replanner.yaml  # Re-planning prompt
│       │   └── constraints.py      # Constraint library for local models
│       │
│       ├── processes/
│       │   ├── __init__.py
│       │   ├── schema.py             # ProcessDefinition, ProcessLoader, validation
│       │   └── builtin/              # 5 built-in YAML process definitions
│       │       ├── investment-analysis.yaml
│       │       ├── marketing-strategy.yaml
│       │       ├── research-report.yaml
│       │       ├── competitive-intel.yaml
│       │       └── consulting-engagement.yaml
│       │
│       ├── tui/
│       │   ├── __init__.py
│       │   ├── app.py              # Textual application (Spec 09)
│       │   ├── dashboard.py        # Task list view
│       │   ├── task_detail.py      # Single task monitoring view
│       │   └── widgets.py          # Custom widgets (progress, diffs, approval)
│       │
│       ├── integrations/
│       │   ├── __init__.py
│       │   └── mcp_server.py       # MCP server exposure (Spec 10)
│       │
│       └── recovery/
│           ├── __init__.py
│           ├── retry.py            # Retry and escalation logic (Spec 13)
│           └── confidence.py       # Confidence scoring (Spec 14)
│
├── tests/
│   ├── conftest.py
│   ├── test_orchestrator.py
│   ├── test_memory.py
│   ├── test_model_router.py
│   ├── test_tools.py
│   ├── test_verification.py
│   ├── test_api.py
│   └── test_events.py
│
├── templates/                       # Built-in task templates (Spec 15 cold start)
│   ├── code_refactor.yaml
│   ├── file_organize.yaml
│   └── document_process.yaml
│
└── data/                            # Runtime data directory (gitignored)
    ├── loom.db                # SQLite database
    ├── tasks/                       # Per-task working data
    │   └── {task_id}/
    │       ├── state.yaml
    │       ├── changelog.json
    │       └── scratch/
    └── logs/                        # Event logs
```

## Dependencies

### pyproject.toml

```toml
[project]
name = "loom"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    # API server
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "sse-starlette>=2.0.0",

    # HTTP client (for model APIs)
    "httpx>=0.27.0",

    # Data handling
    "pyyaml>=6.0.2",
    "pydantic>=2.10.0",

    # Terminal UI
    "textual>=1.0.0",

    # Configuration
    "tomli>=2.0.0;python_version<'3.11'",

    # Async
    "anyio>=4.0.0",

    # CLI
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-httpx>=0.30.0",
    "ruff>=0.8.0",
]
mcp = [
    "mcp>=1.0.0",
]
docker = [
    "docker>=7.0.0",
]

[project.scripts]
loom = "loom.__main__:cli"

[tool.ruff]
line-length = 100
target-version = "py312"
```

## Configuration File

### loom.toml

```toml
[server]
host = "127.0.0.1"
port = 9000

[models.primary]
# MiniMax M2.1 via MLX / LM Studio
provider = "openai_compatible"
base_url = "http://localhost:1234/v1"
model = "minimax-m2.1"
max_tokens = 4096
temperature = 0.1
roles = ["planner", "executor"]

[models.utility]
# Qwen3 8B via Ollama
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:8b"
max_tokens = 2048
temperature = 0.0
roles = ["extractor", "verifier"]

[workspace]
# Default working directory (can be overridden per-task)
default_path = "~/projects"
scratch_dir = "~/.loom/scratch"

[execution]
max_subtask_retries = 3
max_loop_iterations = 50
delegate_task_timeout_seconds = 3600
auto_approve_confidence_threshold = 0.8
compaction_threshold = 0.85  # Not used (structured state), reserved for future

[verification]
tier1_enabled = true
tier2_enabled = true
tier3_enabled = false  # Voting verification, expensive — opt-in
tier3_vote_count = 3

[memory]
database_path = "~/.loom/loom.db"

[logging]
level = "INFO"
event_log_path = "~/.loom/logs"
```

## CLI Entry Points

```
loom                    # Launch the interactive TUI (default)
loom cowork             # Alias for the interactive TUI
loom run <goal>         # Quick run: submit task, show progress, return result
loom run <goal> --workspace /path/to/project
loom serve              # Start the API server
loom status <task_id>   # Check task status
loom cancel <task_id>   # Cancel a running task
loom models             # List available models and status
loom mcp-serve          # Start as MCP server
```

## Implementation Notes

- Use `pathlib.Path` throughout, never raw strings for paths.
- All async functions use `anyio` for backend-agnostic async (works with both asyncio and trio).
- SQLite access through `aiosqlite` for non-blocking database operations.
- Add `aiosqlite>=0.20.0` to dependencies.
- Every module should be independently testable with no global state.
- Configuration is loaded once at startup and passed via dependency injection, never via module-level globals.
- Use Python `dataclasses` for internal data structures, `pydantic` models only at API boundaries.

## Acceptance Criteria

- [ ] `pip install -e .` succeeds with all dependencies
- [ ] `loom --help` shows all CLI commands
- [ ] `loom serve` starts FastAPI on configured port
- [ ] `loom models` connects to configured model providers and reports status
- [ ] SQLite database is created on first run at configured path
- [ ] Configuration is loaded from `loom.toml` with sensible defaults when file is absent
- [ ] All directories in the structure exist and have `__init__.py` files
- [ ] `pytest` discovers and runs test stubs successfully
