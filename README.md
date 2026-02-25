# Loom
[![CI](https://github.com/sfw/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/sfw/loom/actions/workflows/ci.yml)
[![Process Canary](https://github.com/sfw/loom/actions/workflows/process-canary.yml/badge.svg)](https://github.com/sfw/loom/actions/workflows/process-canary.yml)

**Local model orchestration engine** -- task decomposition, execution, and verification using local LLMs.

Loom turns local models into working agents. Give Kimi, Minimax, GLM, or any other model the scaffolding it needs to actually get things done: tool calling, structured planning, parallel execution, verification, and persistent memory -- without sending a single token to someone else's server.

It handles anything a capable assistant should: writing and editing code, researching topics, analyzing documents (PDFs, Word docs, PowerPoint decks), generating reports, running multi-step business workflows, or any task you can define as a process. Ship a consulting engagement, an investment analysis, or a codebase refactor with the same engine.

It also works with Claude and any OpenAI-compatible API, so you can mix local and cloud models in the same task.

**Claude-class cowork UX, local-first.** Tools like Claude Code and Claude cowork deliver strong agentic experiences, and Claude Code can be paired with local model stacks depending on your setup. Loom's focus is different: a model-agnostic harness designed to keep local and mixed local/cloud execution reliable with structured planning, tool safety, independent verification, and persistent memory. Loom is also cross-platform, while Claude cowork is currently macOS + Claude-model oriented. The result is an agentic workflow that stays robust on your own hardware without locking you to one provider.

Loom also exposes a REST API and an MCP server built for agentic systems. Orchestrators like OpenClaw can call Loom's 19 REST endpoints -- or connect via the Model Context Protocol -- to offload complex multi-step tasks: decomposition, tool calling, verification, and memory. Most orchestrators are largely inept at executing these on their own. Instead of hoping a single LLM call handles a 15-step workflow, hand it to Loom and let the harness drive. The MCP integration also means any MCP-compatible agent or IDE can use Loom as a tool provider out of the box.

## Why Loom Exists

Cloud-hosted agents are good but come with constraints: token costs, rate limits, privacy concerns, and vendor lock-in. Local models are getting capable enough for real work, but they lack the scaffolding that makes cloud agents useful. A 14B model can write correct code, draft a solid analysis, or answer complex questions -- but it can't plan multi-step work, verify its own output, recover from mistakes, or remember what it did three turns ago without help.

Loom is that help. It's the harness that turns a local model into a working agent -- for code, research, analysis, or whatever process you define.

## Two Ways to Work

**Interactive** (`loom`) -- Work with a model in a rich terminal UI. You talk, the model responds and uses tools, you see what it's doing in real time. Streaming text, inline diffs, per-tool-call approval, session persistence, conversation recall, and slash commands for control.

**Autonomous** (`loom run`) -- Give Loom a goal, walk away. It decomposes the work into subtasks with a dependency graph, runs independent subtasks in parallel, verifies each result with an independent model, and replans when things go wrong.

```
                    +----------------------------+
Goal -> Planner ->  | [Subtask A]  [Subtask B]   |  parallel batch
                    |     |             |         |  (if independent)
                    |  Execute       Execute      |
                    |  Verify        Verify       |
                    |  Extract*      Extract*     |  * fire-and-forget
                    +----------------------------+
                               |
                         [Subtask C]  (depends on A+B)
                              |
                          Completed
```

## What Makes It Different

**Built for local model weaknesses.** Cloud models reproduce strings precisely. Local models don't -- they drift on whitespace, swap tabs for spaces, drop trailing newlines. Loom's edit tool handles this with fuzzy matching: when an exact string match fails, it normalizes whitespace and finds the closest candidate above a similarity threshold. It also rejects ambiguous matches (two similar regions) so it won't silently edit the wrong place. This is the difference between a tool that works with MiniMax and one that fails 30% of the time.

**Lossless memory, not lossy summarization.** Most agents compress old conversation turns into summaries when context fills up. This destroys information. Loom takes a different approach: every cowork turn is persisted verbatim to SQLite. When context fills up, old turns drop out of the model's window but remain fully searchable. The model has a `conversation_recall` tool to retrieve anything it needs -- specific turns, tool call history, full-text search. Resume any previous session exactly where you left off with `--resume`. This archival guarantee is for cowork history; `/run` and `loom run` may semantically compact model-facing payloads to stay within context budgets, while preserving source artifacts/logs.

**The harness drives, not the model.** The model is a reasoning engine called repeatedly with scoped prompts. The orchestrator decides what happens next: which subtasks to run, when to verify, when to replan, when to escalate. This means a weaker model in a strong harness outperforms a stronger model in a weak one.

**Verification as a separate concern.** The model never checks its own work. An independent verifier (which can be a different, cheaper model) validates results at three tiers: deterministic checks (does the output exist? does it meet structural requirements?), independent LLM review, and multi-vote consensus for high-stakes changes.

**Full undo.** Every file write is preceded by a snapshot. You can revert any individual change, all changes from a subtask, or the entire task. The changelog tracks creates, modifies, deletes, and renames with before-state snapshots.

**Dozens of built-in tools.** File operations (read, write, edit with fuzzy match and batch edits, delete, move) with native support for PDFs, Word documents (.docx), PowerPoint presentations (.pptx), and images. Shell execution with safety checks, git with destructive command blocking, ripgrep search, glob find, web fetch (bounded streaming + truncation for large pages), web search (DuckDuckGo, no API key), code analysis (tree-sitter when installed, regex fallback), calculator (AST-based, safe), spreadsheet operations, document generation, task tracking, conversation recall, delegate_task for spawning sub-agents, ask_user for mid-execution questions, and dedicated research helpers (academic/archives/citations/fact-checking/timeline/inflation plus keyless economic data, historical currency normalization, primary-source OCR, correspondence analysis, social network mapping, and a full keyless investment suite for market data, SEC fundamentals, macro regime scoring, factor exposure, valuation, opportunity ranking, portfolio optimization/evaluation, and portfolio recommendation). All tools auto-discovered via `__init_subclass__`.

**Inline diffs.** Every file edit produces a unified diff in the tool result. Diffs render with Rich markup syntax highlighting in the TUI -- green additions, red removals. You always see exactly what changed.

**Process definitions.** YAML-based domain specialization lets you define personas, phase blueprints, verification/remediation policy, evidence contracts, and prompt constraints for any workflow (`schema_version: 2`). A process can represent a consulting methodology, an investment analysis framework, a research protocol, or a coding standard -- the engine doesn't care. Loom ships with 6 built-in processes and supports installing more from GitHub.

## Quick Start

```bash
# Install
uv sync          # or: pip install -e .

# Launch — the setup wizard runs automatically on first start
loom -w /path/to/workspace

# With a process definition
loom -w /path/to/workspace --process consulting-engagement

# Force process orchestration from inside the TUI (no loom serve required)
# /process use investment-analysis
# /run Analyze Tesla for investment
# /run problem.md            # load goal from workspace file
# /run @problem.md prioritize parser issues
# /run close                 # close current run tab (with confirmation)
# /investment-analysis Analyze Tesla for investment

# Resume a previous session
loom --resume <session-id>

# Autonomous task execution
loom run "Refactor the auth module to use JWT" --workspace /path/to/project
loom run "Research competitive landscape for X and produce a briefing" -w /tmp/research
loom run "Analyze Q3 financials and flag anomalies" -w /tmp/analysis

# Start the API server (for programmatic access)
loom serve
```

## Configuration

On first launch, Loom's built-in setup wizard walks you through provider selection, model configuration, and role assignment — all inside the TUI. The wizard writes `~/.loom/loom.toml` for you. Run `/setup` from inside the TUI at any time to reconfigure, or `loom setup` from the CLI.

You can also create the config manually. Loom reads `loom.toml` from the current directory or `~/.loom/loom.toml`:

```toml
[models.primary]
provider = "ollama"                    # or "openai_compatible" or "anthropic"
base_url = "http://localhost:11434"
model = "kimi-k2.5"
max_tokens = 8192
temperature = 0.1
roles = ["planner", "verifier"]

[models.utility]
provider = "ollama"
base_url = "http://localhost:11434"
model = "minimax-m2.1"
max_tokens = 2048
temperature = 0.0
roles = ["extractor", "executor", "compactor"]

[execution]
max_subtask_retries = 3
max_loop_iterations = 50
max_parallel_subtasks = 3
delegate_task_timeout_seconds = 3600

[limits.runner]
enable_filetype_ingest_router = true
enable_artifact_telemetry_events = true
artifact_telemetry_max_metadata_chars = 1200
enable_model_overflow_fallback = true
ingest_artifact_retention_max_age_days = 14
ingest_artifact_retention_max_files_per_scope = 96
ingest_artifact_retention_max_bytes_per_scope = 268435456
```

Three model backends: Ollama, OpenAI-compatible APIs (LM Studio, vLLM, text-generation-webui), and Anthropic/Claude. Models are assigned roles (`planner`, `executor`, `extractor`, `verifier`, `compactor`). A common split is stronger model for planning + verification and cheaper model for extraction + execution + compaction.
Manage external MCP servers in `~/.loom/mcp.toml` (or workspace `./.loom/mcp.toml`):

```toml
[mcp.servers.notion]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-notion"]
timeout_seconds = 30
enabled = true

[mcp.servers.notion.env]
NOTION_TOKEN = "${NOTION_TOKEN}"
```

MCP merge precedence is: `--mcp-config` > `./.loom/mcp.toml` > `~/.loom/mcp.toml` > legacy `[mcp]` in `loom.toml`.
Configured MCP servers are auto-discovered at startup and registered as namespaced tools (`mcp.<server>.<tool>`).
`delegate_task` (used by `/run`) defaults to a 3600s timeout. Configure this in
`loom.toml` under `[execution].delegate_task_timeout_seconds`; env override
`LOOM_DELEGATE_TIMEOUT_SECONDS` still applies when set.

For artifact and overflow transparency telemetry in `.events.jsonl`, enable
`[limits.runner].enable_artifact_telemetry_events` (default `true`; set to `false` to disable).
Use `artifact_telemetry_max_metadata_chars` to bound handler metadata payload size.
For large fetched binaries/documents (PDFs, Office files, archives), tune
`[limits.runner]` retention keys to control cleanup pressure:
`ingest_artifact_retention_max_age_days`,
`ingest_artifact_retention_max_files_per_scope`, and
`ingest_artifact_retention_max_bytes_per_scope`.

## Process Definitions

A process definition injects a persona, phase blueprint, verification/remediation policy, evidence schema, and prompt constraints without changing engine code. Loom ships with 6 built-in processes: investment analysis, marketing strategy, research report, competitive intelligence, consulting engagement, and market research. You can [create your own](docs/creating-packages.md) or install them from GitHub:

```bash
loom processes                              # list available
loom -w /tmp/acme --process consulting-engagement
loom install user/repo                      # install from GitHub
loom install user/repo --isolated-deps      # per-process dependency env
loom process test consulting-engagement     # run process test cases
```

Process-required tools are enforced at runtime: if `tools.required` contains
missing tools, process activation/task creation fails fast with a clear error.

Process contract v2 is the recommended authoring format (`schema_version: 2`),
with behavior declared under `verification.policy`, `verification.remediation`,
`evidence`, and `prompt_contracts`. v1 definitions still load in compatibility
mode, with compatibility removal targeted for June 30, 2026.

## Adaptive Learning

Loom learns from your interactions so you never repeat yourself. Two learning modes work together:

**Operational learning** (autonomous tasks) -- after every task, Loom extracts model success rates, retry patterns, and successful plan templates. These inform future model selection and planning.

**Behavioral learning** (all interactions) -- Loom detects the gap between what the model delivered and what you actually wanted. When you say "test and lint it" after the model considers its code done, that's a gap signal. Loom extracts a general behavioral rule ("run tests and linter after writing code") and injects it into future prompts. Explicit corrections ("no, use JSON not YAML") are captured the same way.

Patterns are frequency-weighted -- the more a pattern is observed, the higher it ranks. High-frequency patterns persist indefinitely; low-frequency ones are pruned after 90 days. All data stays local in your SQLite database.

```bash
loom learned                              # review learned behavioral patterns
loom learned --all                        # include internal operational patterns
loom learned --type behavioral_gap        # filter by type
loom learned --delete 5                   # remove a specific pattern
loom reset-learning                       # clear everything
```

In the TUI, use `/learned` to open an interactive review screen for learned behavioral patterns, where you can inspect and delete individual items.

## Interfaces

- **Interactive TUI** (`loom`) -- rich terminal interface with chat panel, sidebar, file changes panel with diff viewer, tool approval modals, event log with token sparkline. Built-in setup wizard on first launch. Full session persistence, conversation recall, task delegation, session management (`/sessions`, `/new`, `/resume`, `/setup`), in-session process controls (`/process list`, `/process use <name-or-path>`, `/process off`), forced process orchestration (`/run <goal|@goal-file [goal]|close [run-id-prefix]>`), dynamic direct process commands (`/<process-name> <goal>`), learned pattern review (`/learned`), MCP config controls (`/mcp list`, `/mcp show`, `/mcp test`, `/mcp enable`, `/mcp disable`, `/mcp remove`), auth profile controls (`/auth list`, `/auth show`, `/auth use`, `/auth add`, `/auth edit`, `/auth remove`, `/auth manage`), and click-to-open workspace file previews (Markdown, code/text with syntax highlighting including TypeScript/CSS, JSON, CSV/TSV, HTML, diff/patch, Office docs, PDF text, and image metadata). `Ctrl+W` closes the active process-run tab with confirmation. `/run` executes in-process and does not require `loom serve`; single-token file goals (`/run problem.md`) and explicit file input (`/run @problem.md optional-goal`) load file content into planning context immediately.
- **REST API** -- 19 endpoints for task CRUD, SSE streaming, steering, approval, feedback, memory search
- **MCP server** -- Model Context Protocol integration so other agents can use Loom as a tool

## CLI Commands

```
loom                    Launch the interactive TUI (default; setup wizard on first run)
loom cowork             Alias for the interactive TUI
loom setup              Run the configuration wizard (CLI fallback)
loom run GOAL           Autonomous task execution (server-backed) with `/run`-equivalent process resolution
loom serve              Start the API server
loom status ID          Check task status
loom cancel ID          Cancel a running task
loom models             List configured models
loom auth ...           Manage auth profiles/default selectors
loom processes          List available process definitions
loom install SOURCE     Install a process package
loom uninstall NAME     Remove a process package
loom process test NAME  Run process package test cases
loom mcp ...            Manage MCP server config (list/show/add/edit/remove/test/migrate)
loom mcp-serve          Start the MCP server (stdio transport)
loom learned            Review learned patterns (behavioral by default)
loom reset-learning     Clear all learned patterns
```

Common flags for `loom` / `loom cowork`:
- `-w /path` -- workspace directory
- `--mcp-config /path/to/mcp.toml` -- explicit MCP config layer
- `-m model` -- explicit cowork model override from config (can bypass role routing)
- `--resume <id>` -- resume a previous session
- `--process <name>` -- load a process definition

Role routing note:
- Orchestrator and verifier paths route by role (`planner`, `executor`, `extractor`, `verifier`, `compactor`).
- TUI helper calls (ad hoc process synthesis, run-folder naming) use role-selected helper models when configured.
- Run-folder naming is guardrailed: Loom accepts only clean kebab-case slugs and falls back to deterministic naming when model output is low quality.

## Architecture

53,964 lines of Python in `src/`. 1,824 tests collected. No frameworks (no LangChain, no CrewAI).

```
src/loom/
  __main__.py            CLI (Click), TUI launcher (default command)
  config.py              TOML config loader
  mcp/                   MCP config manager + merge/migration logic
  api/                   FastAPI server, REST routes, SSE streaming
  cowork/                Conversation session, approval, session state
  engine/                Orchestrator, subtask runner, scheduler, verification
  events/                Pub/sub event bus, persistence, webhooks
  integrations/          MCP server
  learning/              Pattern extraction from execution history
  models/                Provider ABC + Ollama, OpenAI, Anthropic backends
  processes/             Process definition loader + 6 built-in YAML processes
  prompts/               7-section prompt assembler with budget trimming
  recovery/              Approval gates, confidence scoring, retry escalation
  state/                 Task state, SQLite memory archive, conversation store
  tools/                 30 built-in tools with auto-discovery, safety, changelog + tree-sitter backend
  tui/                   Textual TUI: chat, sidebar, diff viewer, modals, events
```

## Development

```bash
uv sync --extra dev     # or: pip install -e ".[dev]"
pytest                  # 1,824 tests collected
ruff check src/ tests/  # lint
```

## Requirements

- Python 3.11+
- A model backend: [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), any OpenAI-compatible API, or [Anthropic/Claude](https://console.anthropic.com)

## License

MIT
