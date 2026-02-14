"""MCP server exposing Loom as a tool for external agents.

Provides three tools:
- loom_execute_task: Submit and optionally wait for a task
- loom_task_status: Check task status
- loom_list_tasks: List all tasks
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# --- Tool Schemas ---

EXECUTE_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": "Natural language description of what needs to be accomplished",
        },
        "workspace": {
            "type": "string",
            "description": "Absolute path to the working directory",
        },
        "context": {
            "type": "object",
            "description": "Additional context: constraints, focus areas, preferences",
        },
        "approval_mode": {
            "type": "string",
            "enum": ["auto", "manual", "confidence_threshold"],
            "default": "auto",
        },
        "wait": {
            "type": "boolean",
            "default": True,
            "description": (
                "If true, block until task completes. "
                "If false, return task_id immediately."
            ),
        },
    },
    "required": ["goal"],
}

TASK_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "string", "description": "The task ID to check"},
    },
    "required": ["task_id"],
}

LIST_TASKS_SCHEMA = {
    "type": "object",
    "properties": {
        "status_filter": {
            "type": "string",
            "enum": ["all", "running", "completed", "failed"],
            "description": "Filter tasks by status",
        },
    },
}


class LoomMCPServer:
    """MCP server that exposes Loom as a tool.

    Communicates with the Loom API server over HTTP.
    Can be used with stdio or SSE transport.
    """

    def __init__(self, engine_url: str = "http://localhost:9000"):
        self.engine_url = engine_url.rstrip("/")
        self._tools = self._build_tool_list()

    def _build_tool_list(self) -> list[dict]:
        return [
            {
                "name": "loom_execute_task",
                "description": (
                    "Submit a complex multi-step task to the Loom orchestration engine "
                    "for decomposed execution with verification. Loom will break down "
                    "the task, execute each step using local models, verify results, "
                    "and return the outcome."
                ),
                "inputSchema": EXECUTE_TASK_SCHEMA,
            },
            {
                "name": "loom_task_status",
                "description": "Check the status of a Loom task.",
                "inputSchema": TASK_STATUS_SCHEMA,
            },
            {
                "name": "loom_list_tasks",
                "description": "List all Loom tasks with their current status.",
                "inputSchema": LIST_TASKS_SCHEMA,
            },
        ]

    def list_tools(self) -> list[dict]:
        """Return available MCP tool definitions."""
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> list[dict]:
        """Dispatch an MCP tool call.

        Returns a list of content blocks (text).
        """
        handlers = {
            "loom_execute_task": self._execute_task,
            "loom_task_status": self._task_status,
            "loom_list_tasks": self._list_tasks,
        }

        handler = handlers.get(name)
        if handler is None:
            return [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {name}"})}]

        result = await handler(arguments)
        return [{"type": "text", "text": json.dumps(result, indent=2)}]

    async def _execute_task(self, args: dict) -> dict:
        """Submit task and optionally wait for completion."""
        payload: dict[str, Any] = {
            "goal": args["goal"],
            "approval_mode": args.get("approval_mode", "auto"),
        }
        if args.get("workspace"):
            payload["workspace"] = args["workspace"]
        if args.get("context"):
            payload["context"] = args["context"]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.engine_url}/tasks", json=payload,
                )
                if response.status_code != 201:
                    return {"error": response.text[:500], "status_code": response.status_code}

                try:
                    task = response.json()
                except (json.JSONDecodeError, ValueError):
                    return {"error": "Invalid JSON in task creation response"}

                task_id = task.get("task_id", "")
                if not task_id:
                    return {"error": "Missing task_id in response"}

                if not args.get("wait", True):
                    return task

                # Poll until completion (~20 minutes max)
                for _ in range(600):
                    status_resp = await client.get(f"{self.engine_url}/tasks/{task_id}")
                    if status_resp.status_code != 200:
                        return {"error": "Failed to fetch task status", "task_id": task_id}

                    try:
                        data = status_resp.json()
                    except (json.JSONDecodeError, ValueError):
                        return {"error": "Invalid JSON in status response", "task_id": task_id}

                    if data.get("status") in ("completed", "failed", "cancelled"):
                        return data

                    await asyncio.sleep(2)

                return {"error": "Task timed out after 20 minutes", "task_id": task_id}
        except httpx.ConnectError:
            return {"error": f"Cannot connect to Loom API at {self.engine_url}"}
        except httpx.TimeoutException:
            return {"error": "Request to Loom API timed out"}

    async def _task_status(self, args: dict) -> dict:
        """Check task status."""
        task_id = args.get("task_id", "")
        if not task_id:
            return {"error": "task_id is required"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.engine_url}/tasks/{task_id}")
                if response.status_code == 404:
                    return {"error": f"Task not found: {task_id}"}
                if response.status_code >= 400:
                    return {"error": f"API error ({response.status_code})", "task_id": task_id}
                try:
                    return response.json()
                except (json.JSONDecodeError, ValueError):
                    return {"error": "Invalid JSON in response", "task_id": task_id}
        except httpx.ConnectError:
            return {"error": f"Cannot connect to Loom API at {self.engine_url}"}
        except httpx.TimeoutException:
            return {"error": "Request timed out"}

    async def _list_tasks(self, args: dict) -> dict:
        """List tasks with optional filtering."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.engine_url}/tasks")
                if response.status_code >= 400:
                    return {"error": f"API error ({response.status_code})", "tasks": []}
                try:
                    tasks = response.json()
                except (json.JSONDecodeError, ValueError):
                    return {"error": "Invalid JSON in response", "tasks": []}
        except httpx.ConnectError:
            return {"error": f"Cannot connect to Loom API at {self.engine_url}", "tasks": []}
        except httpx.TimeoutException:
            return {"error": "Request timed out", "tasks": []}

        status_filter = args.get("status_filter", "all")
        if status_filter != "all":
            running_statuses = {"running", "executing", "planning", "waiting_approval"}
            if status_filter == "running":
                tasks = [t for t in tasks if t.get("status") in running_statuses]
            else:
                tasks = [t for t in tasks if t.get("status") == status_filter]

        return {"tasks": tasks, "count": len(tasks)}

    async def run_stdio(self) -> None:
        """Run the MCP server on stdio transport.

        Uses the MCP SDK if available, falls back to simple JSON-RPC.
        """
        try:
            from mcp.server import Server
            from mcp.server.stdio import stdio_server
            from mcp.types import TextContent, Tool

            server = Server("loom")

            @server.list_tools()
            async def handle_list_tools() -> list[Tool]:
                return [
                    Tool(
                        name=t["name"],
                        description=t["description"],
                        inputSchema=t["inputSchema"],
                    )
                    for t in self._tools
                ]

            @server.call_tool()
            async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
                results = await self.call_tool(name, arguments)
                return [TextContent(type="text", text=r["text"]) for r in results]

            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream)

        except ImportError:
            logger.warning("MCP SDK not installed. Install with: pip install mcp")
            # Fallback: simple JSON-RPC on stdin/stdout
            await self._run_simple_stdio()

    async def _run_simple_stdio(self) -> None:
        """Simple JSON-RPC fallback when MCP SDK is not installed."""
        import sys

        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break

            request_id = None
            try:
                request = json.loads(line.strip())
                request_id = request.get("id")
                method = request.get("method", "")

                if method == "tools/list":
                    result = {"tools": self.list_tools()}
                elif method == "tools/call":
                    params = request.get("params", {})
                    result = {
                        "content": await self.call_tool(
                            params.get("name", ""),
                            params.get("arguments", {}),
                        )
                    }
                else:
                    result = {"error": f"Unknown method: {method}"}

                response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result,
                })
                sys.stdout.write(response + "\n")
                sys.stdout.flush()

            except Exception as e:
                logger.exception("MCP stdio handler error")
                error_response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -1, "message": str(e)},
                })
                sys.stdout.write(error_response + "\n")
                sys.stdout.flush()
