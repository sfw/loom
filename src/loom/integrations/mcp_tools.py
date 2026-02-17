"""MCP tool bridge for exposing external MCP tools in Loom's ToolRegistry.

This adapter discovers tools from configured MCP servers and registers them
as namespaced Loom tools:

  mcp.<server_alias>.<tool_name>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import selectors
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loom.config import MCPConfig, MCPServerConfig
from loom.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9_-]+")
_DEFAULT_TIMEOUT_SECONDS = 30
_MIN_TIMEOUT_SECONDS = 5
_MAX_TIMEOUT_SECONDS = 120


def _sanitize_segment(raw: str, fallback: str) -> str:
    value = _SAFE_SEGMENT.sub("_", raw.strip())
    value = value.strip("._-")
    return value or fallback


def _normalize_timeout_seconds(value: int) -> int:
    return max(_MIN_TIMEOUT_SECONDS, min(_MAX_TIMEOUT_SECONDS, int(value)))


def _error_to_text(error_payload: Any) -> str:
    if isinstance(error_payload, dict):
        code = error_payload.get("code", "")
        message = str(error_payload.get("message", "Unknown MCP error"))
        data = error_payload.get("data")
        if data:
            return f"{message} (code={code}, data={data})"
        return f"{message} (code={code})" if code != "" else message
    return str(error_payload)


def _content_blocks_to_text(content: list[Any]) -> str:
    lines: list[str] = []
    for block in content:
        if isinstance(block, dict):
            block_type = str(block.get("type", ""))
            if block_type == "text":
                lines.append(str(block.get("text", "")))
                continue
            if block_type == "image":
                url = block.get("url") or block.get("data") or "<image>"
                lines.append(f"[image] {url}")
                continue
            if block_type == "resource":
                uri = block.get("uri", "")
                lines.append(f"[resource] {uri}")
                continue
            lines.append(json.dumps(block, ensure_ascii=True))
            continue
        lines.append(str(block))
    return "\n".join(line for line in lines if line).strip()


def _coerce_tool_schema(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    name = str(raw.get("name", "")).strip()
    if not name:
        return None

    description = str(raw.get("description", "")).strip()
    params = raw.get("inputSchema")
    if not isinstance(params, dict):
        params = raw.get("parameters")
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}

    return {
        "name": name,
        "description": description or f"MCP tool {name}",
        "parameters": params,
    }


@dataclass
class _MCPStdioClient:
    """One-shot stdio JSON-RPC client for an MCP server command."""

    alias: str
    server: MCPServerConfig

    def _command(self) -> list[str]:
        command = self.server.command.strip()
        if not command:
            raise ValueError(
                f"MCP server {self.alias!r} is missing required 'command'"
            )
        return [command, *self.server.args]

    def _timeout_seconds(self) -> int:
        try:
            timeout = int(self.server.timeout_seconds)
        except (TypeError, ValueError):
            timeout = _DEFAULT_TIMEOUT_SECONDS
        return _normalize_timeout_seconds(timeout)

    def _spawn(self) -> subprocess.Popen:
        env = os.environ.copy()
        env.update(self.server.env)
        cwd = self.server.cwd.strip() or None

        return subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
            env=env,
        )

    def _send_request(
        self,
        process: subprocess.Popen,
        *,
        request_id: int,
        method: str,
        params: dict[str, Any] | None,
    ) -> None:
        if process.stdin is None:
            raise RuntimeError("MCP process stdin unavailable")
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        process.stdin.flush()

    def _send_notification(
        self,
        process: subprocess.Popen,
        *,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        if process.stdin is None:
            return
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        process.stdin.flush()

    def _read_message(
        self,
        process: subprocess.Popen,
        *,
        deadline: float,
        selector: selectors.BaseSelector,
    ) -> dict[str, Any]:
        if process.stdout is None:
            raise RuntimeError("MCP process stdout unavailable")

        while True:
            line = ""
            raw_stdout = getattr(process.stdout, "buffer", None)
            if raw_stdout is not None and hasattr(raw_stdout, "peek"):
                try:
                    if raw_stdout.peek(1):
                        line = process.stdout.readline()
                except Exception:
                    line = ""

            if not line:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"MCP request timed out for server {self.alias!r}"
                    )

                ready = selector.select(timeout=remaining)
                if not ready:
                    raise TimeoutError(
                        f"MCP request timed out for server {self.alias!r}"
                    )

                line = process.stdout.readline()

            if line == "":
                stderr = ""
                if process.stderr is not None:
                    stderr = process.stderr.read().strip()
                message = (
                    f"MCP server {self.alias!r} exited before replying"
                )
                if stderr:
                    message += f": {stderr}"
                raise RuntimeError(message)

            payload = line.strip()
            if not payload:
                continue

            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                logger.debug(
                    "Ignoring non-JSON MCP line from %s: %s",
                    self.alias,
                    payload[:200],
                )
                continue

            if isinstance(parsed, dict):
                return parsed

    def _await_response(
        self,
        process: subprocess.Popen,
        *,
        request_id: int,
        deadline: float,
        selector: selectors.BaseSelector,
    ) -> tuple[dict[str, Any], bool]:
        saw_list_changed = False
        while True:
            message = self._read_message(
                process,
                deadline=deadline,
                selector=selector,
            )
            method = str(message.get("method", ""))
            if method == "notifications/tools/list_changed":
                saw_list_changed = True
                logger.debug(
                    "MCP server %s notified tool list change", self.alias
                )
                continue

            if message.get("id") != request_id:
                continue

            if "error" in message and message["error"] is not None:
                raise RuntimeError(
                    _error_to_text(message["error"])
                )

            result = message.get("result", {})
            parsed = result if isinstance(result, dict) else {"result": result}
            return parsed, saw_list_changed

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        process = self._spawn()
        timeout_seconds = self._timeout_seconds()
        deadline = time.monotonic() + timeout_seconds
        try:
            if process.stdout is None:
                raise RuntimeError("MCP process stdout unavailable")
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)

                # Best-effort MCP initialize handshake for spec-compliant servers.
                self._send_request(
                    process,
                    request_id=1,
                    method="initialize",
                    params={
                        "protocolVersion": "2025-11-05",
                        "capabilities": {"tools": {"listChanged": True}},
                        "clientInfo": {"name": "loom", "version": "0.1.0"},
                    },
                )
                try:
                    self._await_response(
                        process,
                        request_id=1,
                        deadline=min(deadline, time.monotonic() + 5),
                        selector=selector,
                    )
                except Exception:
                    # Some lightweight JSON-RPC servers (including Loom's fallback)
                    # don't implement initialize. Continue with method call.
                    pass

                self._send_notification(
                    process,
                    method="notifications/initialized",
                )

                request_id = 2
                self._send_request(
                    process,
                    request_id=request_id,
                    method=method,
                    params=params or {},
                )
                return self._await_response(
                    process,
                    request_id=request_id,
                    deadline=deadline,
                    selector=selector,
                )
        finally:
            self._terminate(process)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

    def list_tools(self) -> list[dict[str, Any]]:
        response, _changed = self.request("tools/list", {})
        tools_raw: Any
        if isinstance(response, dict) and "tools" in response:
            tools_raw = response.get("tools", [])
        else:
            tools_raw = response.get("result", response)

        if not isinstance(tools_raw, list):
            return []

        tools: list[dict[str, Any]] = []
        for raw_tool in tools_raw:
            parsed = _coerce_tool_schema(raw_tool)
            if parsed:
                tools.append(parsed)
        return tools

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )


class MCPToolProxy(Tool):
    """Expose an external MCP tool through Loom's Tool interface."""

    __loom_register__ = False

    def __init__(
        self,
        *,
        local_name: str,
        remote_name: str,
        description: str,
        parameters: dict[str, Any],
        client: _MCPStdioClient,
        timeout_seconds: int,
        refresh_callback: Callable[..., None] | None = None,
    ) -> None:
        self._local_name = local_name
        self._remote_name = remote_name
        self._description = description
        self._parameters = parameters
        self._client = client
        self._timeout = timeout_seconds
        self._refresh_callback = refresh_callback

    @property
    def name(self) -> str:
        return self._local_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    @property
    def timeout_seconds(self) -> int:
        return self._timeout

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        try:
            result, list_changed = await asyncio.to_thread(
                self._client.call_tool,
                self._remote_name,
                args,
            )
            if list_changed and self._refresh_callback is not None:
                await asyncio.to_thread(self._refresh_callback, force=True)
        except TimeoutError:
            return ToolResult.fail(
                f"MCP tool {self._local_name!r} timed out"
            )
        except Exception as e:
            return ToolResult.fail(
                f"MCP tool {self._local_name!r} failed: {e}"
            )

        if not isinstance(result, dict):
            return ToolResult.ok(str(result), data={"mcp_result": result})

        if result.get("isError") is True:
            content = result.get("content")
            if isinstance(content, list):
                detail = _content_blocks_to_text(content)
            else:
                detail = json.dumps(result, ensure_ascii=True)
            if (
                self._refresh_callback is not None
                and "unknown tool" in detail.lower()
            ):
                await asyncio.to_thread(self._refresh_callback, force=True)
            return ToolResult.fail(detail or "MCP tool returned isError=true")

        content = result.get("content")
        if isinstance(content, list):
            text = _content_blocks_to_text(content)
            if text:
                return ToolResult.ok(text, data={"mcp_result": result})

        return ToolResult.ok(
            json.dumps(result, ensure_ascii=True),
            data={"mcp_result": result},
        )


def register_mcp_tools(
    registry: ToolRegistry,
    *,
    mcp_config: MCPConfig,
) -> list[str]:
    """Discover/register MCP tools and install runtime refresh hook."""
    synchronizer = _MCPRegistrySynchronizer(
        registry=registry,
        mcp_config=mcp_config,
    )
    registered = synchronizer.refresh(force=True)
    registry.set_mcp_refresh_hook(synchronizer.refresh, interval_seconds=15.0)
    # Keep a strong reference so the hook target isn't GC'ed.
    setattr(registry, "_mcp_synchronizer", synchronizer)
    return registered


class _MCPRegistrySynchronizer:
    """Reconciles runtime MCP tool inventory into a ToolRegistry."""

    def __init__(self, *, registry: ToolRegistry, mcp_config: MCPConfig):
        self._registry = registry
        self._mcp_config = mcp_config
        self._lock = threading.Lock()
        self._clients: dict[str, _MCPStdioClient] = {}

    def _client_for(self, alias: str, server: MCPServerConfig) -> _MCPStdioClient:
        existing = self._clients.get(alias)
        if existing is not None and existing.server == server:
            return existing
        client = _MCPStdioClient(alias=alias, server=server)
        self._clients[alias] = client
        return client

    def _discover(self) -> dict[str, MCPToolProxy]:
        discovered: dict[str, MCPToolProxy] = {}

        for alias_raw, server in self._mcp_config.servers.items():
            if not server.enabled:
                continue

            alias = _sanitize_segment(str(alias_raw), "server")
            if not server.command.strip():
                logger.warning(
                    "Skipping MCP server %s: missing command",
                    alias_raw,
                )
                continue

            client = self._client_for(alias, server)
            try:
                tools = client.list_tools()
            except Exception as e:
                logger.warning(
                    "Failed to discover MCP tools for %s: %s",
                    alias,
                    e,
                )
                continue

            for tool_schema in tools:
                remote_name = tool_schema["name"]
                local_tool_segment = _sanitize_segment(remote_name, "tool")
                local_name = f"mcp.{alias}.{local_tool_segment}"

                if local_name in discovered:
                    logger.warning(
                        "Skipping MCP tool collision: %s (remote=%s/%s)",
                        local_name,
                        alias,
                        remote_name,
                    )
                    continue

                discovered[local_name] = MCPToolProxy(
                    local_name=local_name,
                    remote_name=remote_name,
                    description=tool_schema["description"],
                    parameters=tool_schema["parameters"],
                    client=client,
                    timeout_seconds=_normalize_timeout_seconds(
                        server.timeout_seconds
                    ),
                    refresh_callback=self.refresh,
                )

        return discovered

    def refresh(self, *, force: bool = False) -> list[str]:
        """Reconcile MCP tools in-place and return active MCP tool names."""
        del force  # currently always reconciles; reserved for future policies
        with self._lock:
            discovered = self._discover()
            desired = set(discovered.keys())
            current = {
                name for name in self._registry._tools.keys()  # noqa: SLF001
                if name.startswith("mcp.")
            }

            # Remove stale MCP tools.
            for name in sorted(current - desired):
                self._registry.exclude(name)

            # Add new tools and replace changed schemas.
            for name in sorted(desired):
                incoming = discovered[name]
                existing = self._registry._tools.get(name)  # noqa: SLF001
                if existing is None:
                    self._registry.register(incoming)
                    continue
                if existing.schema() != incoming.schema():
                    self._registry.exclude(name)
                    self._registry.register(incoming)

            return sorted(desired)
