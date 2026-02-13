# Spec 10: Agent Integration

## Overview

Loom is designed so that external agents are first-class clients. Any agent — Claude Code, Kimi Code, a custom Python script, a cron job — can submit tasks to Loom via the REST API or by calling Loom as an MCP tool. The agent says "do this" and Loom handles decomposition, execution, verification, and returns verified results.

## Agent-as-Client via REST

An external agent submitting a task:

```python
import httpx

# Submit task
response = httpx.post("http://localhost:9000/tasks", json={
    "goal": "Refactor the authentication module to use JWT tokens",
    "workspace": "/projects/myapp",
    "context": {
        "constraints": ["Don't break existing tests", "Keep backward compat"],
        "focus_dirs": ["src/auth", "src/middleware"],
    },
    "approval_mode": "auto",
    "callback_url": "http://agent:8080/task-complete",  # optional
})
task_id = response.json()["task_id"]

# Option A: Poll for completion
while True:
    status = httpx.get(f"http://localhost:9000/tasks/{task_id}").json()
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(5)

# Option B: Monitor via SSE (preferred)
with httpx.stream("GET", f"http://localhost:9000/tasks/{task_id}/stream") as stream:
    for line in stream.iter_lines():
        if not line.strip():
            continue
        event = json.loads(line.removeprefix("data: "))
        if event["event_type"] in ("task_completed", "task_failed"):
            result = event["data"]
            break

# Option C: Receive webhook callback (fire and forget)
# Agent registers callback_url, Loom POSTs result when done
```

## MCP Server Exposure

Loom exposes itself as an MCP (Model Context Protocol) server. Any MCP-compatible agent framework can discover and call Loom as a tool.

### Tool Definition

```json
{
  "name": "loom_execute_task",
  "description": "Submit a complex multi-step task to the Loom orchestration engine for decomposed execution with verification. Loom will break down the task, execute each step using local models, verify results, and return the outcome. Use this for tasks that require multiple steps, file modifications, or careful coordination.",
  "input_schema": {
    "type": "object",
    "properties": {
      "goal": {
        "type": "string",
        "description": "Natural language description of what needs to be accomplished"
      },
      "workspace": {
        "type": "string",
        "description": "Absolute path to the working directory"
      },
      "context": {
        "type": "object",
        "description": "Additional context: constraints, focus areas, preferences"
      },
      "approval_mode": {
        "type": "string",
        "enum": ["auto", "manual", "confidence_threshold"],
        "default": "auto"
      },
      "wait": {
        "type": "boolean",
        "default": true,
        "description": "If true, block until task completes. If false, return task_id immediately."
      }
    },
    "required": ["goal"]
  }
}
```

### Additional MCP Tools

```json
[
  {
    "name": "loom_task_status",
    "description": "Check the status of a Loom task.",
    "input_schema": {
      "type": "object",
      "properties": {
        "task_id": {"type": "string"}
      },
      "required": ["task_id"]
    }
  },
  {
    "name": "loom_list_tasks",
    "description": "List all Loom tasks with their current status.",
    "input_schema": {
      "type": "object",
      "properties": {
        "status_filter": {
          "type": "string",
          "enum": ["all", "running", "completed", "failed"]
        }
      }
    }
  }
]
```

### MCP Server Implementation

```python
# integrations/mcp_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent

class LoomMCPServer:
    def __init__(self, engine_url: str = "http://localhost:9000"):
        self.server = Server("loom")
        self.engine_url = engine_url
        self._register_tools()

    def _register_tools(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="loom_execute_task",
                    description="Submit a complex multi-step task...",
                    inputSchema={...},
                ),
                Tool(name="loom_task_status", ...),
                Tool(name="loom_list_tasks", ...),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "loom_execute_task":
                return await self._execute_task(arguments)
            elif name == "loom_task_status":
                return await self._task_status(arguments)
            elif name == "loom_list_tasks":
                return await self._list_tasks(arguments)

    async def _execute_task(self, args: dict) -> list[TextContent]:
        """Submit task and optionally wait for completion."""
        async with httpx.AsyncClient() as client:
            # Create task
            response = await client.post(f"{self.engine_url}/tasks", json={
                "goal": args["goal"],
                "workspace": args.get("workspace"),
                "context": args.get("context", {}),
                "approval_mode": args.get("approval_mode", "auto"),
            })
            task = response.json()

            if not args.get("wait", True):
                return [TextContent(type="text", text=json.dumps(task))]

            # Wait for completion via polling
            while True:
                status = await client.get(f"{self.engine_url}/tasks/{task['task_id']}")
                data = status.json()
                if data["status"] in ("completed", "failed"):
                    return [TextContent(type="text", text=json.dumps(data))]
                await asyncio.sleep(2)
```

### Launch

```bash
loom mcp-serve                    # Start as MCP server (stdio transport)
loom mcp-serve --transport sse    # SSE transport for remote connections
```

## Usage Scenario: Claude Code Delegates to Loom

```
User (in Claude Code): "Migrate this Express app to TypeScript"

Claude Code (thinking): This is a complex multi-step task. I'll delegate to Loom.

Claude Code → MCP tool call → loom_execute_task:
  goal: "Migrate Express app from JavaScript to TypeScript"
  workspace: "/Users/scott/projects/express-app"
  approval_mode: "auto"

Loom engine:
  1. Plans: install deps → rename files → add types → fix errors → run tests
  2. Executes each subtask with verification
  3. Streams progress (visible in Loom TUI if running)
  4. Returns verified result

Claude Code receives:
  {
    "status": "completed",
    "summary": "Migrated 47 files to TypeScript. All tests passing.",
    "artifacts": ["tsconfig.json", "src/**/*.ts"],
    "subtasks_completed": 7,
    "subtasks_failed": 0
  }

Claude Code: "Done! I've migrated all 47 files to TypeScript..."
```

## Acceptance Criteria

- [ ] External agents can submit tasks via POST /tasks and receive results
- [ ] SSE streaming works for agent monitoring
- [ ] Webhook callbacks fire on task completion/failure
- [ ] MCP server exposes loom_execute_task, loom_task_status, loom_list_tasks tools
- [ ] MCP server works with stdio transport (for local agent integration)
- [ ] Blocking mode (wait=true) correctly polls until completion
- [ ] Non-blocking mode returns task_id immediately
- [ ] MCP tool responses include enough detail for the calling agent to proceed
- [ ] `loom mcp-serve` starts and is discoverable by MCP clients
