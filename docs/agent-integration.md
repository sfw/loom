# Connecting Agents to Loom

This guide shows how to connect external agents (Claude Code, custom scripts, cron jobs, other AI agents) to Loom as clients.

**Two execution models:**
- **Interactive** (`loom`) — conversation-first TUI with full session persistence. The agent and developer collaborate in real time with 21 tools, streaming, conversation recall, and per-tool-call approval. No server needed. Optional `--process` flag loads a domain-specific process definition, and `/run <goal>` forces process orchestration in-session.
- **Task mode** (REST API / MCP) — autonomous, fire-and-forget. Loom decomposes the goal into subtasks, executes, verifies, and reports back.

This document covers both modes.

## Architecture

```
 Agent (client)          Loom Engine (server)
 ┌──────────┐           ┌─────────────────────────────────────┐
 │ Claude   │  REST/    │  API Server (FastAPI)               │
 │ Code,    │──SSE──────│    ↓                                │
 │ scripts, │  or MCP   │  Orchestrator (lifecycle, schedule) │
 │ cron     │           │    ├─ Planner (model tier 2)        │
 │          │◄──────────│    ├─ SubtaskRunner(s) ─ parallel   │
 │          │  results  │    │   ├─ Executor (model tier 1)   │
 │          │           │    │   ├─ Verifier (independent)    │
 │          │           │    │   └─ Extractor (fire & forget) │
 │          │           │    ├─ Tools (file, shell)           │
 │          │           │    ├─ Memory (SQLite)               │
 │          │           │    └─ Learning (patterns)           │
 └──────────┘           └─────────────────────────────────────┘
```

Independent subtasks (no unmet dependencies) are dispatched in parallel up to `max_parallel_subtasks`. Sequential tasks with dependencies execute in order automatically.

Agents connect via three mechanisms:

| Method | Best for | Latency |
|--------|----------|---------|
| REST API (poll) | Simple scripts, cron jobs | Seconds (polling interval) |
| REST API (SSE) | Real-time monitoring, dashboards | Instant (streaming) |
| MCP Server | AI agent frameworks (Claude Code, etc.) | Instant (stdio/SSE) |

---

## 1. REST API Integration

### Submit a Task

```python
import httpx

client = httpx.Client(base_url="http://localhost:9000")

response = client.post("/tasks", json={
    "goal": "Refactor the authentication module to use JWT tokens",
    "workspace": "/projects/myapp",
    "context": {
        "constraints": ["Don't break existing tests", "Keep backward compat"],
        "focus_dirs": ["src/auth", "src/middleware"],
    },
    "approval_mode": "auto",           # "auto" | "manual" | "confidence_threshold"
    "callback_url": "http://agent:8080/done",  # optional webhook
})

task = response.json()
task_id = task["task_id"]
print(f"Task submitted: {task_id}")
```

**Request fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `goal` | Yes | Natural language description of what to accomplish |
| `workspace` | No | Absolute path to the working directory for file operations |
| `process` | No | Name of a process definition to apply (e.g., `"investment-analysis"`) |
| `context` | No | Dict with constraints, focus areas, preferences |
| `approval_mode` | No | `"auto"` (default), `"manual"`, or `"confidence_threshold"` |
| `callback_url` | No | URL to POST results to on completion/failure |

### Wait for Results

**Option A: Poll**

```python
import time

while True:
    status = client.get(f"/tasks/{task_id}").json()
    if status["status"] in ("completed", "failed", "cancelled"):
        break
    print(f"Status: {status['status']} — {status.get('progress', '')}")
    time.sleep(5)

if status["status"] == "completed":
    print("Task completed successfully")
else:
    print(f"Task failed: {status.get('errors', [])}")
```

**Option B: SSE Stream (preferred)**

```python
with httpx.stream("GET", f"http://localhost:9000/tasks/{task_id}/stream") as stream:
    for line in stream.iter_lines():
        if not line.strip() or not line.startswith("data: "):
            continue
        event = json.loads(line.removeprefix("data: "))
        print(f"[{event['event_type']}] {event.get('data', {})}")

        if event["event_type"] in ("task_completed", "task_failed"):
            break
```

**Option C: Webhook Callback (fire-and-forget)**

Set `callback_url` when creating the task. Loom will POST the result:

```json
{
  "task_id": "a1b2c3d4",
  "status": "completed",
  "goal": "Refactor auth module",
  "subtasks_completed": 5,
  "subtasks_failed": 0,
  "workspace_changes": {
    "files_created": 2,
    "files_modified": 7,
    "files_deleted": 0
  }
}
```

### Other API Endpoints

```python
# List all tasks
tasks = client.get("/tasks").json()

# Get task details (includes plan, subtask statuses, errors)
task = client.get(f"/tasks/{task_id}").json()

# Cancel a running task
client.post(f"/tasks/{task_id}/cancel")

# Steer a running task (inject new instructions mid-flight)
client.post(f"/tasks/{task_id}/steer", json={
    "instruction": "Focus on the login flow first, skip registration for now"
})

# Approve/reject a subtask waiting for human review
client.post(f"/tasks/{task_id}/approve", json={
    "subtask_id": "s3",
    "approved": True
})

# Provide feedback on a completed/failed task
client.post(f"/tasks/{task_id}/feedback", json={
    "rating": 4,
    "comment": "Good but missed the edge case in password reset"
})

# System health
health = client.get("/health").json()

# List available models
models = client.get("/models").json()

# List available tools
tools = client.get("/tools").json()
```

---

## 2. MCP Server Integration

Loom exposes itself as an [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server, making it discoverable as a tool by any MCP-compatible agent.

### Available MCP Tools

**`loom_execute_task`** — Submit a task and optionally wait for completion.

```json
{
  "name": "loom_execute_task",
  "description": "Submit a complex multi-step task to the Loom orchestration engine.",
  "input_schema": {
    "type": "object",
    "properties": {
      "goal":          { "type": "string", "description": "What to accomplish" },
      "workspace":     { "type": "string", "description": "Absolute path to working directory" },
      "context":       { "type": "object", "description": "Constraints, focus areas" },
      "approval_mode": { "type": "string", "enum": ["auto", "manual", "confidence_threshold"] },
      "wait":          { "type": "boolean", "default": true, "description": "Block until done?" }
    },
    "required": ["goal"]
  }
}
```

**`loom_task_status`** — Check the status of a running task.

**`loom_list_tasks`** — List all tasks with current statuses.

### Launching the MCP Server

```bash
# stdio transport (for local agents like Claude Code)
loom mcp-serve

# SSE transport (for remote agents)
loom mcp-serve --transport sse
```

### Configuring Claude Code to Use Loom

Add to your Claude Code MCP config (`.claude/mcp_servers.json` or equivalent):

```json
{
  "loom": {
    "command": "loom",
    "args": ["mcp-serve"],
    "env": {}
  }
}
```

Once configured, Claude Code can call Loom as a tool:

```
User: "Migrate this Express app to TypeScript"

Claude Code (thinking): Complex multi-step task — I'll delegate to Loom.

→ loom_execute_task(
    goal="Migrate Express app from JavaScript to TypeScript",
    workspace="/Users/dev/express-app",
    approval_mode="auto"
  )

← {
    "status": "completed",
    "summary": "Migrated 47 files to TypeScript. All tests passing.",
    "subtasks_completed": 7
  }

Claude Code: "Done! I migrated all 47 files to TypeScript..."
```

---

## 3. Programmatic Python Integration

For agents written in Python, you can use Loom's engines directly without the HTTP layer.

### Cowork Session (Interactive)

Use CoworkSession for conversation-first interactive work with all 21 tools:

```python
import asyncio
from pathlib import Path

from loom.config import load_config
from loom.cowork.session import CoworkSession, CoworkTurn, ToolCallEvent, build_cowork_system_prompt
from loom.models.router import ModelRouter
from loom.tools import create_default_registry

async def cowork():
    config = load_config()
    router = ModelRouter.from_config(config)
    model = router.select(role="executor")
    tools = create_default_registry()
    workspace = Path("/projects/myapp")

    session = CoworkSession(
        model=model,
        tools=tools,
        workspace=workspace,
        system_prompt=build_cowork_system_prompt(workspace),
    )

    # Send a message and process events
    async for event in session.send_streaming("Fix the auth bug"):
        if isinstance(event, str):
            print(event, end="")  # streamed text token
        elif isinstance(event, ToolCallEvent) and event.result:
            print(f"  {event.name}: {'ok' if event.result.success else 'err'}")
        elif isinstance(event, CoworkTurn):
            print(f"\n[{len(event.tool_calls)} tools, {event.tokens_used} tokens]")

asyncio.run(cowork())
```

### Direct Engine Usage (Autonomous Tasks)

```python
import asyncio
from pathlib import Path

from loom.config import Config, load_config
from loom.api.engine import create_engine
from loom.engine.orchestrator import create_task

async def run_task():
    config = load_config()  # reads loom.toml
    engine = await create_engine(config)

    task = create_task(
        goal="Add input validation to the user registration endpoint",
        workspace="/projects/myapp",
        approval_mode="auto",
    )

    result = await engine.orchestrator.execute_task(task)

    print(f"Status: {result.status}")
    print(f"Progress: {result.progress}")
    for subtask in result.plan.subtasks:
        print(f"  [{subtask.status}] {subtask.id}: {subtask.summary}")

    await engine.shutdown()

asyncio.run(run_task())
```

### Custom Model Providers

To use your own model (local or remote), implement the `ModelProvider` interface:

```python
from loom.models.base import ModelProvider, ModelResponse, TokenUsage, ToolCall

class MyCustomProvider(ModelProvider):
    """Example: wrap any LLM API as a Loom model provider."""

    def __init__(self, api_url: str, model_name: str, tier: int = 1):
        self._api_url = api_url
        self._model_name = model_name
        self._tier = tier

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        # Call your LLM API here
        # Parse the response into ModelResponse
        # Handle tool_calls if the model supports function calling
        ...
        return ModelResponse(
            text="model output",
            tool_calls=[...],  # or None if text-only
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            model=self._model_name,
        )

    async def health_check(self) -> bool:
        # Return True if the model is reachable
        ...

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def tier(self) -> int:
        return self._tier

    @property
    def roles(self) -> list[str]:
        # Which roles can this model fulfill?
        # Options: "executor", "planner", "verifier", "extractor"
        return ["executor", "planner"]
```

Register it with the router:

```python
from loom.models.router import ModelRouter

router = ModelRouter()
router.add_provider("my-model", MyCustomProvider(
    api_url="http://localhost:11434",
    model_name="my-model-v1",
    tier=2,
))
```

### Custom Tools

Extend Loom's capabilities by registering custom tools:

```python
from loom.tools.registry import Tool, ToolContext, ToolResult

class RunTestsTool(Tool):
    """Custom tool that runs the project test suite."""

    @property
    def name(self) -> str:
        return "run_tests"

    @property
    def description(self) -> str:
        return "Run the project's test suite and return results."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "test_path": {
                    "type": "string",
                    "description": "Path to test file or directory (relative to workspace)",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Show verbose output",
                    "default": False,
                },
            },
        }

    @property
    def timeout_seconds(self) -> int:
        return 120  # tests can take a while

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        import subprocess

        test_path = args.get("test_path", ".")
        verbose = args.get("verbose", False)

        cmd = ["python", "-m", "pytest", test_path]
        if verbose:
            cmd.append("-v")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(ctx.workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if result.returncode == 0:
                return ToolResult.ok(result.stdout)
            else:
                return ToolResult.fail(
                    f"Tests failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
                )
        except subprocess.TimeoutExpired:
            return ToolResult.fail("Test suite timed out")
```

Register it:

```python
# Option 1: Auto-discovery (recommended for built-in tools)
# Place your tool in src/loom/tools/my_tool.py and subclass Tool.
# It will be found automatically by discover_tools() on import.

# Option 2: Manual registration (for tools outside the tools/ package)
from loom.tools import create_default_registry

registry = create_default_registry()
registry.register(RunTestsTool())
```

### Listening to Events

Subscribe to events for custom monitoring, logging, or triggering side effects:

```python
from loom.events.bus import Event, EventBus
from loom.events.types import TASK_COMPLETED, SUBTASK_FAILED

event_bus = EventBus()

# Listen to specific event types
def on_task_complete(event: Event):
    print(f"Task {event.task_id} completed!")
    # Send Slack notification, update dashboard, etc.

event_bus.subscribe(TASK_COMPLETED, on_task_complete)

# Listen to all events
async def log_all(event: Event):
    print(f"[{event.event_type}] task={event.task_id} data={event.data}")

event_bus.subscribe_all(log_all)
```

---

## 4. Process Definitions

Process definitions inject domain expertise into Loom without changing engine code. They're YAML files that specify personas, phase blueprints, verification rules, tool guidance, and memory extraction types.

### Built-in Processes

```bash
# List available processes
loom processes

# Use with CLI
loom run "Analyze ACME Corp" --workspace /tmp/acme --process investment-analysis
loom -w /tmp/project --process consulting-engagement
```

| Process | Phases | Mode | Domain |
|---------|--------|------|--------|
| `investment-analysis` | 5 | strict | Financial due diligence and valuation |
| `marketing-strategy` | 6 | guided | Go-to-market strategy and campaigns |
| `research-report` | 4 | guided | Research synthesis and reporting |
| `competitive-intel` | 3 | suggestive | Competitive landscape analysis |
| `consulting-engagement` | 5 | strict | McKinsey-style issue tree consulting |

### Using with REST API

Include `"process"` in the task creation payload:

```python
response = client.post("/tasks", json={
    "goal": "Evaluate Tesla as an investment opportunity",
    "workspace": "/projects/tesla-analysis",
    "process": "investment-analysis",
})
```

### Using with Python

```python
from loom.processes.schema import ProcessLoader

# Discover and load a process
loader = ProcessLoader(workspace=Path("/projects/myapp"))
process = loader.load("investment-analysis")

# List available processes
available = loader.list_available()
for p in available:
    print(f"{p['name']} v{p['version']}: {p['description']}")
```

### Creating Custom Processes

Create a YAML file in `~/.loom/processes/` or `<workspace>/.loom/processes/`:

```yaml
name: my-custom-process
version: "1.0"
description: "Custom domain process"
persona: |
  You are a domain expert specializing in...

phase_mode: guided  # strict | guided | suggestive

phases:
  - id: research
    description: "Gather and analyze data"
    depends_on: []
    model_tier: 2
    verification_tier: 1
    deliverables:
      - "research-notes.md"
    acceptance_criteria: "Notes cover all key areas"

  - id: synthesize
    description: "Synthesize findings into report"
    depends_on: [research]
    model_tier: 3
    verification_tier: 2
    deliverables:
      - "final-report.md"
    acceptance_criteria: "Report is comprehensive and actionable"

verification:
  rules:
    - name: no-placeholders
      description: "No placeholder text in output"
      check: "(?i)(TODO|PLACEHOLDER|TBD|INSERT HERE)"
      severity: error
      type: regex

memory:
  extract_types:
    - key_finding
    - recommendation

tool_guidance: |
  Use the calculator tool for any quantitative analysis.
  Use the spreadsheet tool for data organization.
```

### Process Packages

For more complex processes, create a directory with bundled tools:

```
~/.loom/processes/my-process/
  process.yaml          # Process definition (with optional dependencies: list)
  tools/
    custom_tool.py      # Automatically loaded and registered
  templates/
    report_template.md  # Available as reference
```

### Installing Process Packages

Process packages can be installed from GitHub repos, shorthands, or local paths:

```bash
# From a GitHub URL
loom install https://github.com/acme/loom-google-analytics

# From GitHub shorthand (user/repo)
loom install acme/loom-google-analytics

# From a local directory
loom install ./my-local-process

# Into a workspace (instead of global ~/.loom/processes/)
loom install ./my-process -w /path/to/project

# Skip auto-installing Python dependencies
loom install acme/loom-analytics --skip-deps

# Install dependencies in isolated per-process envs
loom install acme/loom-analytics --isolated-deps

# Uninstall
loom uninstall google-analytics
```

Run packaged test cases (deterministic by default, or live with real providers):

```bash
loom process test investment-analysis
loom process test ./my-local-process --case smoke
loom process test ./my-local-process --live
```

The installer expects a `process.yaml` at the repo root. Python dependencies
declared in the `dependencies` field are auto-installed (tries `uv` first, falls back to `pip`):

```yaml
name: google-analytics
version: "1.0"
description: "Google Analytics analysis workflow"
dependencies:
  - google-analytics-data>=0.18.0
  - pandas>=2.0
# ... rest of process definition
```

Process packages can also declare a `tests:` manifest in `process.yaml` with
test IDs, deterministic/live mode, optional tool/network requirements, and
acceptance checks for phases, deliverables, and verification output.

Notes:
- `tools.required` is enforced at runtime. Missing required tools fail process activation/task creation.
- Bundled tool name collisions are rejected at load time for the colliding tool (warning + skip).
- `--isolated-deps` installs dependencies into `<install-target>/.deps/<process-name>/`.

### Process package troubleshooting

- `Process '<name>' requires missing tool(s): ...`:
  ensure those tools are available in the active registry, or remove them from `tools.required`.
- `Bundled tool '<name>' ... conflicts with existing tool class ...; skipping bundled tool`:
  rename the bundled tool to a globally unique name and reinstall.
- `Failed to install isolated dependencies`:
  verify dependency names/versions and Python packaging access in the isolated environment.

---

## 5. Learned Patterns (Adaptive Learning)

Loom learns behavioral patterns from your interactions and uses them to personalize future prompts. You can query and manage these patterns programmatically.

### CLI

```bash
# List learned behavioral patterns (default)
loom learned

# Include internal operational patterns
loom learned --all

# Filter by type
loom learned --type behavioral_gap
loom learned --type behavioral_correction

# Delete a specific pattern
loom learned --delete 5

# Clear everything
loom reset-learning
```

### TUI

Use the `/learned` slash command to open an interactive review screen for behavioral patterns. Each pattern shows its type, description, frequency count, and last-seen date, with a Delete button for removal.

### Python API

```python
from loom.learning.manager import LearningManager
from loom.state.memory import Database

db = Database("~/.loom/loom.db")
await db.initialize()
mgr = LearningManager(db)

# Query all patterns
patterns = await mgr.query_all(limit=50)

# Query behavioral patterns only (for prompt injection)
behavioral = await mgr.query_behavioral(limit=15)

# Delete a specific pattern
await mgr.delete_pattern(pattern_id=5)

# Clear all patterns
await mgr.clear_all()
```

### Pattern Types

| Type | Source | Example |
|------|--------|---------|
| `behavioral_gap` | Implicit -- user had to ask for something the model should have done | "Run tests after writing code" |
| `behavioral_correction` | Explicit -- user contradicted the model's output | "Use JSON not YAML" |
| `subtask_success` | Post-task -- model success rates per subtask type | Success rate for "install-deps" |
| `retry_pattern` | Post-task -- which subtasks needed escalation | "Schema conversion needs tier 2" |
| `task_template` | Post-task -- successful plans as reusable templates | 7-step migration plan |

Behavioral patterns are injected into system prompts to personalize future interactions. Operational patterns (subtask_success, retry_pattern, task_template) inform model routing and planning.

---

## 6. Approval Modes

Control how much autonomy Loom has:

| Mode | Behavior |
|------|----------|
| `auto` | Proceed at high confidence (>=0.8), gate destructive ops, pause at low confidence |
| `manual` | Pause every subtask for human review |
| `confidence_threshold` | Auto-proceed above the configured threshold, pause below |

In `auto` mode, Loom always gates operations matching dangerous patterns (`rm -rf`, `drop table`, writes to `.env` files, etc.) regardless of confidence score.

### Responding to Approval Requests

When Loom pauses for approval, it emits an `approval_requested` event. Your agent responds via the API:

```python
# Approve
client.post(f"/tasks/{task_id}/approve", json={
    "subtask_id": "s3",
    "approved": True,
})

# Reject
client.post(f"/tasks/{task_id}/approve", json={
    "subtask_id": "s3",
    "approved": False,
})
```

---

## 7. Configuration

Loom reads `loom.toml` from the current directory or `~/.loom/loom.toml`:

```toml
[server]
host = "127.0.0.1"
port = 9000

[models.executor]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:8b"
roles = ["executor", "extractor"]

[models.planner]
provider = "ollama"
base_url = "http://localhost:11434"
model = "minimax-m2.1"
roles = ["planner", "verifier"]
max_tokens = 8192

[execution]
max_subtask_retries = 3
max_loop_iterations = 50
max_parallel_subtasks = 3     # Independent subtasks run concurrently
auto_approve_confidence_threshold = 0.8

[verification]
tier1_enabled = true    # Deterministic checks (free, instant)
tier2_enabled = true    # Independent LLM verification
tier3_enabled = false   # Voting verification (N independent checks)
tier3_vote_count = 3

[memory]
database_path = "~/.loom/loom.db"

[workspace]
default_path = "~/projects"
scratch_dir = "~/.loom/scratch"

[mcp.servers.notion]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-notion"]
timeout_seconds = 30

[mcp.servers.notion.env]
NOTION_TOKEN = "your-token"
```

Configured MCP servers are auto-discovered and exposed as namespaced tools:
`mcp.<server>.<tool>`. Tool registrations refresh at runtime as connected MCP
servers publish new tool lists.

---

## 8. End-to-End Example: Bash Script Agent

A minimal agent that submits a task and waits for results:

```bash
#!/usr/bin/env bash
set -euo pipefail

LOOM_URL="${LOOM_URL:-http://localhost:9000}"
GOAL="$1"
WORKSPACE="${2:-$(pwd)}"

# Submit
TASK_ID=$(curl -s -X POST "$LOOM_URL/tasks" \
  -H "Content-Type: application/json" \
  -d "{\"goal\": \"$GOAL\", \"workspace\": \"$WORKSPACE\"}" \
  | jq -r '.task_id')

echo "Task $TASK_ID submitted"

# Poll
while true; do
  STATUS=$(curl -s "$LOOM_URL/tasks/$TASK_ID" | jq -r '.status')
  case "$STATUS" in
    completed) echo "Done!"; exit 0 ;;
    failed)    echo "Failed"; exit 1 ;;
    cancelled) echo "Cancelled"; exit 2 ;;
    *) echo "Status: $STATUS"; sleep 5 ;;
  esac
done
```

Usage:

```bash
./loom-agent.sh "Add comprehensive error handling to the API routes" /projects/myapp
```

---

## Event Types Reference

Events your agent can listen for via SSE or event bus subscription:

| Event | Description |
|-------|-------------|
| `task_planning` | Task decomposition started |
| `task_plan_ready` | Plan created, subtasks defined |
| `task_executing` | Execution loop started |
| `task_replanning` | Plan being revised after failures |
| `task_completed` | All subtasks completed successfully |
| `task_failed` | Task failed (retries exhausted or fatal error) |
| `task_cancelled` | Task cancelled by user/agent |
| `subtask_started` | Individual subtask execution began |
| `subtask_completed` | Subtask finished and verified |
| `subtask_failed` | Subtask failed verification |
| `subtask_retrying` | Subtask being retried with escalation |
| `approval_requested` | Paused for human review |
| `approval_received` | Human responded to approval request |
| `token_streamed` | Model token generated (streaming mode) |
