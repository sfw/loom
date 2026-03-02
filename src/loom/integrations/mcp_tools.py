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
from dataclasses import dataclass, field
from typing import Any

from loom.config import (
    MCP_DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    MCP_DEFAULT_TIMEOUT_SECONDS,
    MCP_SERVER_TYPE_LOCAL,
    MCP_SERVER_TYPE_REMOTE,
    MCPConfig,
    MCPServerConfig,
    normalize_mcp_discovery_timeout_seconds,
    normalize_mcp_timeout_seconds,
)
from loom.integrations.mcp.oauth import (
    bearer_auth_header_for_alias,
    ensure_mcp_oauth_ready,
)
from loom.tools.registry import Tool, ToolContext, ToolRegistry, ToolResult
from loom.utils.latency import log_latency_event

logger = logging.getLogger(__name__)

_SAFE_SEGMENT = re.compile(r"[^a-zA-Z0-9_-]+")
_ENV_REF_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(authorization|proxy-authorization)\b\s*[:=]\s*"
            r"([^\s,;]+(?:\s+[^\s,;]+)?)"
        ),
        r"\1=<redacted>",
    ),
    (
        re.compile(r"(?i)\b(access_token|refresh_token|id_token)\b\s*[:=]\s*([^\s,;]+)"),
        r"\1=<redacted>",
    ),
)

_MCP_CIRCUIT_FAILURE_THRESHOLD = 5
_MCP_CIRCUIT_FAILURE_WINDOW_SECONDS = 60.0
_MCP_CIRCUIT_COOLDOWN_SECONDS = 30.0
_MCP_GLOBAL_MAX_IN_FLIGHT = 64
_MCP_PER_SERVER_MAX_IN_FLIGHT = 8
_MCP_PER_SERVER_MAX_QUEUE = 32


def _sanitize_segment(raw: str, fallback: str) -> str:
    value = _SAFE_SEGMENT.sub("_", raw.strip())
    value = value.strip("._-")
    return value or fallback


def _normalize_timeout_seconds(value: int) -> int:
    return normalize_mcp_timeout_seconds(
        value,
        default=MCP_DEFAULT_TIMEOUT_SECONDS,
    )


def _normalize_discovery_timeout_seconds(value: int) -> int:
    return normalize_mcp_discovery_timeout_seconds(
        value,
        default=MCP_DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    )


def _discovery_timeout_seconds() -> int:
    raw = os.environ.get("LOOM_MCP_DISCOVERY_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return MCP_DEFAULT_DISCOVERY_TIMEOUT_SECONDS
    try:
        return _normalize_discovery_timeout_seconds(int(raw))
    except (TypeError, ValueError):
        return MCP_DEFAULT_DISCOVERY_TIMEOUT_SECONDS


def _resolve_env_map(
    raw_env: dict[str, str],
    *,
    base_env: dict[str, str],
) -> dict[str, str]:
    """Resolve env refs like `${TOKEN}` against a base environment."""
    resolved: dict[str, str] = {}
    missing_refs: list[str] = []
    for key, value in raw_env.items():
        text = str(value)
        match = _ENV_REF_PATTERN.match(text.strip())
        if match is None:
            resolved[key] = text
            continue
        ref_name = match.group(1)
        ref_value = base_env.get(ref_name)
        if ref_value is None:
            missing_refs.append(ref_name)
            continue
        resolved[key] = ref_value
    if missing_refs:
        missing = ", ".join(sorted(set(missing_refs)))
        raise RuntimeError(
            "Missing environment variable(s) for MCP env refs: "
            f"{missing}"
        )
    return resolved


def _error_to_text(error_payload: Any) -> str:
    if isinstance(error_payload, dict):
        code = error_payload.get("code", "")
        message = str(error_payload.get("message", "Unknown MCP error"))
        data = error_payload.get("data")
        if data:
            return f"{message} (code={code}, data={data})"
        return f"{message} (code={code})" if code != "" else message
    return str(error_payload)


def _redact_error_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


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
    """Persistent stdio JSON-RPC client for one local MCP server."""

    alias: str
    server: MCPServerConfig
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _selector: selectors.BaseSelector | None = field(default=None, init=False, repr=False)
    _line_buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    _next_request_id: int = field(default=1, init=False, repr=False)
    _session_env_signature: tuple[tuple[str, str], ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
    )
    _last_error: str = field(default="", init=False, repr=False)
    _last_connected_at: float | None = field(default=None, init=False, repr=False)
    _last_activity_at: float | None = field(default=None, init=False, repr=False)

    def _command(self) -> list[str]:
        if self.server.type != MCP_SERVER_TYPE_LOCAL:
            raise ValueError(
                f"MCP server {self.alias!r} uses unsupported bridge type "
                f"{self.server.type!r}; stdio bridge supports only local."
            )
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
            timeout = MCP_DEFAULT_TIMEOUT_SECONDS
        return _normalize_timeout_seconds(timeout)

    def _spawn(
        self,
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> subprocess.Popen:
        env = os.environ.copy()
        env.update(_resolve_env_map(self.server.env, base_env=env))
        if env_overrides:
            env.update(env_overrides)
        cwd = self.server.cwd.strip() or None

        return subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=cwd,
            env=env,
        )

    @staticmethod
    def _env_signature(env_overrides: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        if not env_overrides:
            return ()
        items: list[tuple[str, str]] = []
        for key in sorted(env_overrides):
            items.append((str(key), str(env_overrides[key])))
        return tuple(items)

    @property
    def connected_pid(self) -> int | None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return None
            return int(process.pid)

    def status_snapshot(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            connected = process is not None and process.poll() is None
            return {
                "alias": self.alias,
                "type": self.server.type,
                "status": "healthy" if connected else "disconnected",
                "connected": connected,
                "pid": int(process.pid) if connected and process is not None else None,
                "last_connected_at": self._last_connected_at,
                "last_activity_at": self._last_activity_at,
                "last_error": self._last_error,
            }

    def _reset_session(self) -> None:
        selector = self._selector
        process = self._process
        self._selector = None
        self._process = None
        self._line_buffer = bytearray()
        self._initialized = False
        self._next_request_id = 1
        self._session_env_signature = ()
        if selector is not None:
            try:
                selector.close()
            except Exception:
                pass
        if process is not None:
            self._terminate(process)

    def close(self) -> None:
        with self._lock:
            self._reset_session()

    def _ensure_session(
        self,
        *,
        env_overrides: dict[str, str] | None,
    ) -> tuple[subprocess.Popen, selectors.BaseSelector]:
        desired_signature = self._env_signature(env_overrides)
        process = self._process
        selector = self._selector
        if (
            process is not None
            and selector is not None
            and process.poll() is None
            and self._session_env_signature == desired_signature
        ):
            return process, selector

        self._reset_session()
        process = self._spawn(env_overrides=env_overrides)
        if process.stdout is None:
            self._terminate(process)
            raise RuntimeError("MCP process stdout unavailable")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        self._process = process
        self._selector = selector
        self._session_env_signature = desired_signature
        self._last_connected_at = time.time()
        self._last_activity_at = time.time()
        return process, selector

    def _consume_request_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

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
        if process.poll() is not None:
            raise RuntimeError(
                f"MCP server {self.alias!r} exited before accepting {method!r}: "
                f"{self._exit_context(process)}"
            )
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        try:
            process.stdin.write(
                (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            # Child exited between spawn and write/flush.
            raise RuntimeError(
                f"MCP server {self.alias!r} closed stdin during {method!r}: "
                f"{self._exit_context(process)}"
            ) from e

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
        try:
            process.stdin.write(
                (json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8")
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            # Best-effort notification; caller will fail on next request if needed.
            logger.debug(
                "MCP server %s did not accept notification %s (%s)",
                self.alias,
                method,
                self._exit_context(process),
            )

    @staticmethod
    def _exit_context(process: subprocess.Popen) -> str:
        code = process.poll()
        base = (
            f"exit code {code}"
            if code is not None
            else "process still running"
        )
        stderr = ""
        if code is not None and process.stderr is not None:
            try:
                raw = process.stderr.read()
                if isinstance(raw, bytes):
                    stderr = raw.decode("utf-8", errors="replace").strip()
                else:
                    stderr = str(raw).strip()
            except Exception:
                stderr = ""
        if stderr:
            return f"{base}; stderr: {stderr}"
        return base

    def _read_message(
        self,
        process: subprocess.Popen,
        *,
        deadline: float,
        selector: selectors.BaseSelector,
        line_buffer: bytearray,
    ) -> dict[str, Any]:
        if process.stdout is None:
            raise RuntimeError("MCP process stdout unavailable")

        while True:
            newline_pos = line_buffer.find(b"\n")
            if newline_pos < 0:
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

                try:
                    chunk = os.read(process.stdout.fileno(), 4096)
                except OSError as e:
                    raise RuntimeError(
                        f"MCP server {self.alias!r} read failed: {e}"
                    ) from e
                if not chunk:
                    stderr = ""
                    if process.stderr is not None:
                        try:
                            raw = process.stderr.read()
                            if isinstance(raw, bytes):
                                stderr = raw.decode(
                                    "utf-8",
                                    errors="replace",
                                ).strip()
                            else:
                                stderr = str(raw).strip()
                        except Exception:
                            stderr = ""
                    message = (
                        f"MCP server {self.alias!r} exited before replying"
                    )
                    if stderr:
                        message += f": {stderr}"
                    raise RuntimeError(message)
                line_buffer.extend(chunk)
                continue

            raw_line = bytes(line_buffer[:newline_pos])
            del line_buffer[:newline_pos + 1]
            payload = raw_line.decode("utf-8", errors="replace").strip()
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
        line_buffer: bytearray,
    ) -> tuple[dict[str, Any], bool]:
        saw_list_changed = False
        while True:
            message = self._read_message(
                process,
                deadline=deadline,
                selector=selector,
                line_buffer=line_buffer,
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
        *,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        timeout = (
            _normalize_discovery_timeout_seconds(timeout_seconds)
            if timeout_seconds is not None
            else self._timeout_seconds()
        )
        with self._lock:
            for attempt in range(2):
                process: subprocess.Popen | None = None
                selector: selectors.BaseSelector | None = None
                deadline = time.monotonic() + timeout
                try:
                    process, selector = self._ensure_session(env_overrides=env_overrides)
                    if not self._initialized:
                        init_id = self._consume_request_id()
                        self._send_request(
                            process,
                            request_id=init_id,
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
                                request_id=init_id,
                                deadline=min(deadline, time.monotonic() + 5),
                                selector=selector,
                                line_buffer=self._line_buffer,
                            )
                        except Exception:
                            # Some lightweight JSON-RPC servers don't implement initialize.
                            pass
                        self._send_notification(
                            process,
                            method="notifications/initialized",
                        )
                        self._initialized = True

                    request_id = self._consume_request_id()
                    self._send_request(
                        process,
                        request_id=request_id,
                        method=method,
                        params=params or {},
                    )
                    result = self._await_response(
                        process,
                        request_id=request_id,
                        deadline=deadline,
                        selector=selector,
                        line_buffer=self._line_buffer,
                    )
                    self._last_error = ""
                    self._last_activity_at = time.time()
                    return result
                except Exception as e:
                    self._last_error = str(e)
                    self._reset_session()
                    if attempt >= 1:
                        raise
                    logger.debug(
                        "Retrying MCP request after session reset alias=%s method=%s error=%s",
                        self.alias,
                        method,
                        e,
                    )
                    continue

    @staticmethod
    def _close_pipe(stream: Any) -> None:
        """Close one process pipe, swallowing teardown-time broken pipes."""
        if stream is None:
            return
        try:
            stream.close()
        except (BrokenPipeError, OSError, ValueError):
            # Broken pipes are expected when child exited before stdin flush.
            pass

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        try:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    process.wait(timeout=1)
        finally:
            _MCPStdioClient._close_pipe(process.stdin)
            _MCPStdioClient._close_pipe(process.stdout)
            _MCPStdioClient._close_pipe(process.stderr)

    def list_tools(
        self,
        *,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        response, _changed = self.request(
            "tools/list",
            {},
            env_overrides=env_overrides,
            timeout_seconds=timeout_seconds,
        )
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
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            env_overrides=env_overrides,
        )


@dataclass
class _MCPRemoteHTTPClient:
    """Best-effort JSON-RPC client for remote MCP endpoints."""

    alias: str
    server: MCPServerConfig

    def _timeout_seconds(self) -> int:
        try:
            timeout = int(self.server.timeout_seconds)
        except (TypeError, ValueError):
            timeout = MCP_DEFAULT_TIMEOUT_SECONDS
        return _normalize_timeout_seconds(timeout)

    def _headers(
        self,
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> dict[str, str]:
        base_env = os.environ.copy()
        if env_overrides:
            base_env.update(env_overrides)
        headers = _resolve_env_map(self.server.headers, base_env=base_env)
        auth_header = bearer_auth_header_for_alias(self.alias)
        if auth_header and "Authorization" not in headers:
            headers["Authorization"] = auth_header
        headers.setdefault("Content-Type", "application/json")
        return headers

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if self.server.type != MCP_SERVER_TYPE_REMOTE:
            raise ValueError(
                f"MCP server {self.alias!r} is not remote (type={self.server.type!r})"
            )
        timeout = (
            _normalize_discovery_timeout_seconds(timeout_seconds)
            if timeout_seconds is not None
            else self._timeout_seconds()
        )
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        try:
            import httpx

            response = httpx.post(
                self.server.url,
                headers=self._headers(env_overrides=env_overrides),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as e:
            raise RuntimeError(
                f"Remote MCP request failed for {self.alias!r}: {e}"
            ) from e
        if not isinstance(body, dict):
            raise RuntimeError(
                f"Remote MCP response for {self.alias!r} is not an object"
            )
        if "error" in body and body["error"] is not None:
            raise RuntimeError(_error_to_text(body["error"]))
        result = body.get("result", {})
        parsed = result if isinstance(result, dict) else {"result": result}
        return parsed, False

    def list_tools(
        self,
        *,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        response, _changed = self.request(
            "tools/list",
            {},
            env_overrides=env_overrides,
            timeout_seconds=timeout_seconds,
        )
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
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            env_overrides=env_overrides,
        )


@dataclass(frozen=True)
class MCPConnectionState:
    """Runtime status for one configured MCP alias."""

    alias: str
    type: str
    enabled: bool
    status: str
    last_error: str = ""
    reconnect_attempts: int = 0
    last_connected_at: float | None = None
    last_activity_at: float | None = None
    pid: int | None = None
    circuit_state: str = "closed"
    circuit_open_until: float | None = None
    queue_depth: int = 0
    in_flight: int = 0
    remediation: str = ""


@dataclass
class _MCPBreakerState:
    state: str = "closed"
    failures: list[float] = field(default_factory=list)
    opened_at: float | None = None
    probe_in_flight: bool = False


class MCPConnectionManager:
    """Owns MCP transport clients and runtime connection state."""

    def __init__(self, *, mcp_config: MCPConfig) -> None:
        self._mcp_config = mcp_config
        self._lock = threading.RLock()
        self._gate = threading.Condition(self._lock)
        self._clients: dict[str, _MCPStdioClient | _MCPRemoteHTTPClient] = {}
        self._states: dict[str, MCPConnectionState] = {}
        self._breakers: dict[str, _MCPBreakerState] = {}
        self._alias_waiters: dict[str, int] = {}
        self._alias_inflight: dict[str, int] = {}
        self._global_inflight = 0

    def update_config(self, mcp_config: MCPConfig) -> None:
        with self._lock:
            self._mcp_config = mcp_config
            configured = set(mcp_config.servers.keys())
            stale = [alias for alias in self._clients if alias not in configured]
            for alias in stale:
                client = self._clients.pop(alias, None)
                if isinstance(client, _MCPStdioClient):
                    client.close()
                self._states.pop(alias, None)
                self._breakers.pop(alias, None)
                self._alias_waiters.pop(alias, None)
                inflight = int(self._alias_inflight.pop(alias, 0))
                if inflight > 0:
                    self._global_inflight = max(0, self._global_inflight - inflight)

    def _breaker_for(self, alias: str) -> _MCPBreakerState:
        breaker = self._breakers.get(alias)
        if breaker is None:
            breaker = _MCPBreakerState()
            self._breakers[alias] = breaker
        return breaker

    def _operation_timeout_seconds(
        self,
        *,
        server: MCPServerConfig,
        timeout_seconds: int | None,
    ) -> int:
        if timeout_seconds is None:
            return _normalize_timeout_seconds(server.timeout_seconds)
        return _normalize_discovery_timeout_seconds(timeout_seconds)

    def _state_remediation(
        self,
        *,
        alias: str,
        status: str,
        last_error: str,
    ) -> str:
        if status == "needs_auth":
            return f"Run `loom mcp auth login {alias}`."
        if status == "disabled":
            return f"Run `loom mcp enable {alias}`."
        if status in {"configured", "disconnected"}:
            return f"Run `loom mcp connect {alias}`."
        if status == "degraded":
            if "circuit open" in last_error.lower():
                return f"Wait for cooldown, then run `loom mcp reconnect {alias}`."
            return f"Run `loom mcp reconnect {alias}`."
        if status == "error":
            return f"Inspect `loom mcp status` and run `loom mcp reconnect {alias}`."
        return ""

    def _circuit_open_until(self, alias: str) -> float | None:
        breaker = self._breakers.get(alias)
        if breaker is None or breaker.state != "open" or breaker.opened_at is None:
            return None
        remaining = _MCP_CIRCUIT_COOLDOWN_SECONDS - (time.monotonic() - breaker.opened_at)
        if remaining <= 0:
            return None
        return time.time() + remaining

    def _set_state(
        self,
        *,
        alias: str,
        server: MCPServerConfig,
        status: str,
        last_error: str = "",
        reconnect_attempts_delta: int = 0,
        pid: int | None = None,
        remediation: str | None = None,
    ) -> None:
        current = self._states.get(alias)
        reconnect_attempts = (
            int(current.reconnect_attempts) if current is not None else 0
        ) + max(0, int(reconnect_attempts_delta))
        previous_pid = current.pid if current is not None else None
        next_pid = pid if pid is not None else previous_pid
        next_connected_at = current.last_connected_at if current is not None else None
        if pid is not None and pid != previous_pid:
            next_connected_at = time.time()
        elif status == "healthy" and next_connected_at is None:
            next_connected_at = time.time()
        redacted_error = _redact_error_text(last_error)
        circuit_state = self._breaker_for(alias).state
        queue_depth = int(self._alias_waiters.get(alias, 0))
        in_flight = int(self._alias_inflight.get(alias, 0))
        next_remediation = (
            remediation
            if remediation is not None
            else self._state_remediation(
                alias=alias,
                status=status,
                last_error=redacted_error,
            )
        )
        self._states[alias] = MCPConnectionState(
            alias=alias,
            type=server.type,
            enabled=bool(server.enabled),
            status=status,
            last_error=redacted_error,
            reconnect_attempts=reconnect_attempts,
            last_connected_at=next_connected_at,
            last_activity_at=time.time(),
            pid=next_pid,
            circuit_state=circuit_state,
            circuit_open_until=self._circuit_open_until(alias),
            queue_depth=queue_depth,
            in_flight=in_flight,
            remediation=next_remediation,
        )

    def _acquire_slot(
        self,
        *,
        alias: str,
        timeout_seconds: int,
    ) -> None:
        waiters = int(self._alias_waiters.get(alias, 0))
        if waiters >= _MCP_PER_SERVER_MAX_QUEUE:
            raise RuntimeError(
                f"MCP server {alias!r} backpressure: queue is full "
                f"(max={_MCP_PER_SERVER_MAX_QUEUE})"
            )
        self._alias_waiters[alias] = waiters + 1
        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        try:
            while True:
                alias_inflight = int(self._alias_inflight.get(alias, 0))
                if (
                    self._global_inflight < _MCP_GLOBAL_MAX_IN_FLIGHT
                    and alias_inflight < _MCP_PER_SERVER_MAX_IN_FLIGHT
                ):
                    self._global_inflight += 1
                    self._alias_inflight[alias] = alias_inflight + 1
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"MCP server {alias!r} request queue timeout"
                    )
                self._gate.wait(timeout=min(remaining, 0.25))
        finally:
            next_waiters = max(0, int(self._alias_waiters.get(alias, 1)) - 1)
            if next_waiters == 0:
                self._alias_waiters.pop(alias, None)
            else:
                self._alias_waiters[alias] = next_waiters

    def _release_slot(self, *, alias: str) -> None:
        self._global_inflight = max(0, self._global_inflight - 1)
        alias_inflight = int(self._alias_inflight.get(alias, 0))
        if alias_inflight <= 1:
            self._alias_inflight.pop(alias, None)
        else:
            self._alias_inflight[alias] = alias_inflight - 1
        self._gate.notify_all()

    def _check_circuit_before_request(
        self,
        *,
        alias: str,
    ) -> None:
        breaker = self._breaker_for(alias)
        now = time.monotonic()
        if breaker.state == "open":
            opened_at = breaker.opened_at if breaker.opened_at is not None else now
            elapsed = now - opened_at
            if elapsed < _MCP_CIRCUIT_COOLDOWN_SECONDS:
                remaining = _MCP_CIRCUIT_COOLDOWN_SECONDS - elapsed
                raise RuntimeError(
                    f"Circuit open due to repeated failures; retry in {remaining:.1f}s"
                )
            breaker.state = "half_open"
            breaker.opened_at = None
            breaker.probe_in_flight = False

        if breaker.state == "half_open":
            if breaker.probe_in_flight:
                raise RuntimeError(
                    "Circuit half-open probe already in progress"
                )
            breaker.probe_in_flight = True

    def _record_success(self, *, alias: str) -> None:
        breaker = self._breaker_for(alias)
        breaker.failures.clear()
        breaker.state = "closed"
        breaker.opened_at = None
        breaker.probe_in_flight = False

    def _record_failure(self, *, alias: str) -> str:
        breaker = self._breaker_for(alias)
        now = time.monotonic()
        window_start = now - _MCP_CIRCUIT_FAILURE_WINDOW_SECONDS
        breaker.failures = [ts for ts in breaker.failures if ts >= window_start]
        breaker.failures.append(now)
        if breaker.state == "half_open":
            breaker.state = "open"
            breaker.opened_at = now
            breaker.probe_in_flight = False
            return breaker.state
        if len(breaker.failures) >= _MCP_CIRCUIT_FAILURE_THRESHOLD:
            breaker.state = "open"
            breaker.opened_at = now
            breaker.probe_in_flight = False
            return breaker.state
        if breaker.state != "open":
            breaker.state = "closed"
            breaker.opened_at = None
        breaker.probe_in_flight = False
        return breaker.state

    def _clear_half_open_probe(self, *, alias: str) -> None:
        breaker = self._breakers.get(alias)
        if breaker is None:
            return
        breaker.probe_in_flight = False

    def _invoke_with_policy(
        self,
        *,
        alias: str,
        server: MCPServerConfig,
        timeout_seconds: int | None,
        invoke: Callable[
            [_MCPStdioClient | _MCPRemoteHTTPClient, int],
            Any,
        ],
    ) -> Any:
        effective_timeout = self._operation_timeout_seconds(
            server=server,
            timeout_seconds=timeout_seconds,
        )
        acquired_slot = False
        request_started = False
        client: _MCPStdioClient | _MCPRemoteHTTPClient | None = None
        try:
            with self._lock:
                if not server.enabled:
                    self._set_state(alias=alias, server=server, status="disabled")
                    raise RuntimeError(f"MCP server {alias!r} is disabled")
                if (
                    server.type == MCP_SERVER_TYPE_REMOTE
                    and not self._remote_oauth_ready(alias, server)
                ):
                    raise RuntimeError(f"MCP server {alias!r} requires OAuth login")
                self._acquire_slot(alias=alias, timeout_seconds=effective_timeout)
                acquired_slot = True
                self._check_circuit_before_request(alias=alias)
                client = self._client_for(alias, server)
                self._set_state(alias=alias, server=server, status="connecting")
                request_started = True

            result = invoke(client, effective_timeout)
            pid = client.connected_pid if isinstance(client, _MCPStdioClient) else None
            with self._lock:
                self._record_success(alias=alias)
                self._set_state(
                    alias=alias,
                    server=server,
                    status="healthy",
                    last_error="",
                    pid=pid,
                )
            return result
        except Exception as e:
            with self._lock:
                if request_started:
                    circuit_state = self._record_failure(alias=alias)
                    status = "degraded" if circuit_state == "open" else "error"
                    self._set_state(
                        alias=alias,
                        server=server,
                        status=status,
                        last_error=str(e),
                        reconnect_attempts_delta=1,
                    )
                else:
                    text = str(e)
                    lowered = text.lower()
                    status = "error"
                    if "disabled" in lowered:
                        status = "disabled"
                    elif "requires oauth login" in lowered:
                        status = "needs_auth"
                    elif "circuit " in lowered:
                        status = "degraded"
                    self._set_state(
                        alias=alias,
                        server=server,
                        status=status,
                        last_error=text,
                    )
                    self._clear_half_open_probe(alias=alias)
            raise
        finally:
            if acquired_slot:
                with self._lock:
                    self._release_slot(alias=alias)

    def _client_for(
        self,
        alias: str,
        server: MCPServerConfig,
    ) -> _MCPStdioClient | _MCPRemoteHTTPClient:
        existing = self._clients.get(alias)
        if existing is not None and getattr(existing, "server", None) == server:
            return existing
        if isinstance(existing, _MCPStdioClient):
            existing.close()
        if server.type == MCP_SERVER_TYPE_LOCAL:
            client: _MCPStdioClient | _MCPRemoteHTTPClient = _MCPStdioClient(
                alias=alias,
                server=server,
            )
        else:
            client = _MCPRemoteHTTPClient(alias=alias, server=server)
        self._clients[alias] = client
        return client

    def _remote_oauth_ready(self, alias: str, server: MCPServerConfig) -> bool:
        if not server.oauth.enabled:
            return True
        readiness = ensure_mcp_oauth_ready(alias)
        if not readiness.ready:
            self._set_state(
                alias=alias,
                server=server,
                status="needs_auth",
                last_error=readiness.reason or "OAuth token missing or expired",
            )
            return False
        return True

    def list_tools(
        self,
        *,
        alias: str,
        server: MCPServerConfig,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        tools = self._invoke_with_policy(
            alias=alias,
            server=server,
            timeout_seconds=timeout_seconds,
            invoke=lambda client, effective_timeout: client.list_tools(
                env_overrides=env_overrides,
                timeout_seconds=effective_timeout,
            ),
        )
        if not isinstance(tools, list):
            return []
        return tools

    def call_tool(
        self,
        *,
        alias: str,
        server: MCPServerConfig,
        name: str,
        arguments: dict[str, Any],
        env_overrides: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        response = self._invoke_with_policy(
            alias=alias,
            server=server,
            timeout_seconds=None,
            invoke=lambda client, _effective_timeout: client.call_tool(
                name,
                arguments,
                env_overrides=env_overrides,
            ),
        )
        if not isinstance(response, tuple):
            raise RuntimeError(
                f"Invalid MCP call response type for {alias!r}: {type(response)}"
            )
        return response

    def close_alias(self, alias: str) -> None:
        with self._lock:
            client = self._clients.pop(alias, None)
            server = self._mcp_config.servers.get(alias)
            if isinstance(client, _MCPStdioClient):
                client.close()
            if server is not None:
                self._breakers.pop(alias, None)
                self._alias_waiters.pop(alias, None)
                inflight = int(self._alias_inflight.pop(alias, 0))
                if inflight > 0:
                    self._global_inflight = max(0, self._global_inflight - inflight)
                self._set_state(alias=alias, server=server, status="disconnected")
            self._gate.notify_all()

    def close_all(self) -> None:
        with self._lock:
            aliases = list(self._clients.keys())
        for alias in aliases:
            self.close_alias(alias)

    def state_for(
        self,
        *,
        alias: str,
        server: MCPServerConfig | None = None,
    ) -> MCPConnectionState:
        with self._lock:
            state = self._states.get(alias)
            if state is not None:
                return state
            effective_server = server or self._mcp_config.servers.get(alias)
            if effective_server is None:
                raise RuntimeError(f"MCP server {alias!r} is not configured")
            self._set_state(
                alias=alias,
                server=effective_server,
                status="disabled" if not effective_server.enabled else "configured",
            )
            created = self._states.get(alias)
            if created is None:
                raise RuntimeError(f"MCP server {alias!r} state unavailable")
            return created

    def connect(
        self,
        *,
        alias: str,
        server: MCPServerConfig,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> MCPConnectionState:
        self.list_tools(
            alias=alias,
            server=server,
            env_overrides=env_overrides,
            timeout_seconds=timeout_seconds,
        )
        return self.state_for(alias=alias, server=server)

    def reconnect(
        self,
        *,
        alias: str,
        server: MCPServerConfig,
        env_overrides: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> MCPConnectionState:
        self.close_alias(alias)
        return self.connect(
            alias=alias,
            server=server,
            env_overrides=env_overrides,
            timeout_seconds=timeout_seconds,
        )

    def disconnect(
        self,
        *,
        alias: str,
        server: MCPServerConfig,
    ) -> MCPConnectionState:
        self.close_alias(alias)
        return self.state_for(alias=alias, server=server)

    def debug_snapshot(self) -> dict[str, Any]:
        with self._lock:
            states = self.states()
            by_alias = {item.alias: item for item in states}
            aliases = sorted(set(self._mcp_config.servers.keys()) | set(by_alias.keys()))
            per_alias: dict[str, dict[str, Any]] = {}
            for alias in aliases:
                state = by_alias.get(alias)
                breaker = self._breakers.get(alias)
                per_alias[alias] = {
                    "status": state.status if state is not None else "unknown",
                    "last_error": state.last_error if state is not None else "",
                    "queue_depth": state.queue_depth if state is not None else 0,
                    "in_flight": state.in_flight if state is not None else 0,
                    "reconnect_attempts": (
                        state.reconnect_attempts if state is not None else 0
                    ),
                    "circuit_state": (
                        state.circuit_state
                        if state is not None
                        else (breaker.state if breaker is not None else "closed")
                    ),
                    "circuit_failures_recent": (
                        len(breaker.failures) if breaker is not None else 0
                    ),
                    "remediation": state.remediation if state is not None else "",
                }
            return {
                "limits": {
                    "global_max_in_flight": _MCP_GLOBAL_MAX_IN_FLIGHT,
                    "per_server_max_in_flight": _MCP_PER_SERVER_MAX_IN_FLIGHT,
                    "per_server_max_queue": _MCP_PER_SERVER_MAX_QUEUE,
                    "circuit_failure_threshold": _MCP_CIRCUIT_FAILURE_THRESHOLD,
                    "circuit_failure_window_seconds": _MCP_CIRCUIT_FAILURE_WINDOW_SECONDS,
                    "circuit_cooldown_seconds": _MCP_CIRCUIT_COOLDOWN_SECONDS,
                },
                "global_in_flight": self._global_inflight,
                "aliases": per_alias,
            }

    def states(self) -> list[MCPConnectionState]:
        with self._lock:
            # Ensure every configured alias has at least configured/disabled state.
            for alias, server in self._mcp_config.servers.items():
                if alias in self._states:
                    continue
                self._set_state(
                    alias=alias,
                    server=server,
                    status="disabled" if not server.enabled else "configured",
                )
            return sorted(self._states.values(), key=lambda item: item.alias)


class MCPToolProxy(Tool):
    """Expose an external MCP tool through Loom's Tool interface."""

    __loom_register__ = False

    def __init__(
        self,
        *,
        alias: str,
        local_name: str,
        remote_name: str,
        description: str,
        parameters: dict[str, Any],
        server: MCPServerConfig,
        connection_manager: MCPConnectionManager,
        timeout_seconds: int,
        refresh_callback: Callable[..., None] | None = None,
    ) -> None:
        self._alias = alias
        self._local_name = local_name
        self._remote_name = remote_name
        self._description = description
        self._parameters = parameters
        self._server = server
        self._connection_manager = connection_manager
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
        env_overrides: dict[str, str] | None = None
        auth_ctx = getattr(ctx, "auth_context", None)
        if auth_ctx is not None:
            try:
                env_overrides = auth_ctx.env_for_mcp_alias(self._alias)
            except Exception as e:
                return ToolResult.fail(
                    f"MCP auth context error for {self._alias!r}: {e}"
                )

        try:
            result, list_changed = await asyncio.to_thread(
                self._connection_manager.call_tool,
                alias=self._alias,
                server=self._server,
                name=self._remote_name,
                arguments=args,
                env_overrides=env_overrides,
            )
            if list_changed and self._refresh_callback is not None:
                await asyncio.to_thread(
                    self._refresh_callback,
                    force=True,
                    auth_context=auth_ctx,
                )
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
                await asyncio.to_thread(
                    self._refresh_callback,
                    force=True,
                    auth_context=auth_ctx,
                )
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
    startup_mode: str = "sync",
) -> list[str]:
    """Discover/register MCP tools and install runtime refresh hook."""
    synchronizer = _MCPRegistrySynchronizer(
        registry=registry,
        mcp_config=mcp_config,
    )
    registry.set_mcp_refresh_hook(synchronizer.refresh, interval_seconds=15.0)
    registry.set_mcp_discovery_hook(synchronizer.discover, interval_seconds=15.0)
    # Keep a strong reference so the hook target isn't GC'ed.
    setattr(registry, "_mcp_synchronizer", synchronizer)
    setattr(registry, "_mcp_connection_manager", synchronizer._connection_manager)  # noqa: SLF001

    mode = str(startup_mode or "sync").strip().lower()
    if mode not in {"sync", "background"}:
        mode = "sync"
    if mode == "background":
        def _warmup() -> None:
            try:
                synchronizer.refresh(force=True)
            except Exception as e:
                logger.warning("Background MCP warmup failed: %s", e)

        thread = threading.Thread(
            target=_warmup,
            name="loom-mcp-startup-warmup",
            daemon=True,
        )
        thread.start()
        return []

    return synchronizer.refresh(force=True)


def runtime_connection_states(registry: ToolRegistry) -> list[MCPConnectionState]:
    """Return runtime MCP connection states for a registry instance."""
    synchronizer = getattr(registry, "_mcp_synchronizer", None)
    if synchronizer is None:
        return []
    states_fn = getattr(synchronizer, "connection_states", None)
    if not callable(states_fn):
        return []
    try:
        states = states_fn()
    except Exception:
        return []
    if not isinstance(states, list):
        return []
    return [item for item in states if isinstance(item, MCPConnectionState)]


def _runtime_synchronizer(registry: ToolRegistry) -> Any:
    synchronizer = getattr(registry, "_mcp_synchronizer", None)
    if synchronizer is None:
        raise RuntimeError("MCP runtime is not initialized for this registry.")
    return synchronizer


def runtime_connect_alias(
    registry: ToolRegistry,
    *,
    alias: str,
    timeout_seconds: int | None = None,
) -> MCPConnectionState:
    """Connect one configured MCP alias and return runtime state."""
    connect_fn = getattr(_runtime_synchronizer(registry), "connect_alias", None)
    if not callable(connect_fn):
        raise RuntimeError("MCP runtime does not support connect action.")
    return connect_fn(alias=alias, timeout_seconds=timeout_seconds)


def runtime_disconnect_alias(
    registry: ToolRegistry,
    *,
    alias: str,
) -> MCPConnectionState:
    """Disconnect one configured MCP alias and return runtime state."""
    disconnect_fn = getattr(_runtime_synchronizer(registry), "disconnect_alias", None)
    if not callable(disconnect_fn):
        raise RuntimeError("MCP runtime does not support disconnect action.")
    return disconnect_fn(alias=alias)


def runtime_reconnect_alias(
    registry: ToolRegistry,
    *,
    alias: str,
    timeout_seconds: int | None = None,
) -> MCPConnectionState:
    """Reconnect one configured MCP alias and return runtime state."""
    reconnect_fn = getattr(_runtime_synchronizer(registry), "reconnect_alias", None)
    if not callable(reconnect_fn):
        raise RuntimeError("MCP runtime does not support reconnect action.")
    return reconnect_fn(alias=alias, timeout_seconds=timeout_seconds)


def runtime_debug_snapshot(registry: ToolRegistry) -> dict[str, Any]:
    """Return manager diagnostics for support/debug tooling."""
    debug_fn = getattr(_runtime_synchronizer(registry), "debug_snapshot", None)
    if not callable(debug_fn):
        raise RuntimeError("MCP runtime does not support debug snapshot.")
    snapshot = debug_fn()
    if not isinstance(snapshot, dict):
        return {}
    return snapshot


class _MCPRegistrySynchronizer:
    """Reconciles runtime MCP tool inventory into a ToolRegistry."""

    def __init__(self, *, registry: ToolRegistry, mcp_config: MCPConfig):
        self._registry = registry
        self._mcp_config = mcp_config
        self._lock = threading.Lock()
        self._connection_manager = MCPConnectionManager(mcp_config=mcp_config)
        self._discovery_timeout_seconds = _discovery_timeout_seconds()

    def _discover(
        self,
        *,
        auth_context: Any = None,
        enable_registry_refresh: bool = True,
    ) -> dict[str, MCPToolProxy]:
        discovered: dict[str, MCPToolProxy] = {}
        active_aliases: set[str] = set()

        for alias_raw, server in self._mcp_config.servers.items():
            if not server.enabled:
                continue

            alias = _sanitize_segment(str(alias_raw), "server")
            if server.type == MCP_SERVER_TYPE_LOCAL and not server.command.strip():
                logger.warning(
                    "Skipping MCP server %s: missing command",
                    alias_raw,
                )
                continue

            active_aliases.add(alias_raw)
            env_overrides: dict[str, str] | None = None
            if auth_context is not None:
                try:
                    env_overrides = auth_context.env_for_mcp_alias(alias_raw)
                except Exception as e:
                    logger.warning(
                        "Failed to resolve MCP auth env for %s: %s",
                        alias_raw,
                        e,
                    )
                    continue
            try:
                tools = self._connection_manager.list_tools(
                    alias=alias_raw,
                    server=server,
                    env_overrides=env_overrides,
                    timeout_seconds=self._discovery_timeout_seconds,
                )
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
                    alias=alias_raw,
                    local_name=local_name,
                    remote_name=remote_name,
                    description=tool_schema["description"],
                    parameters=tool_schema["parameters"],
                    server=server,
                    connection_manager=self._connection_manager,
                    timeout_seconds=_normalize_timeout_seconds(
                        server.timeout_seconds
                    ),
                    refresh_callback=self.refresh if enable_registry_refresh else None,
                )

        stale_aliases = set(self._mcp_config.servers.keys()) - active_aliases
        for stale in stale_aliases:
            self._connection_manager.close_alias(stale)

        return discovered

    def refresh(self, *, force: bool = False, auth_context: Any = None) -> list[str]:
        """Reconcile MCP tools in-place and return active MCP tool names."""
        del force  # currently always reconciles; reserved for future policies
        started = time.monotonic()
        with self._lock:
            self._connection_manager.update_config(self._mcp_config)
            discovered = self._discover(auth_context=auth_context)
            desired = set(discovered.keys())
            with self._registry._tools_lock:  # noqa: SLF001
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
                with self._registry._tools_lock:  # noqa: SLF001
                    existing = self._registry._tools.get(name)  # noqa: SLF001
                if existing is None:
                    self._registry.register(incoming)
                    continue
                if existing.schema() != incoming.schema():
                    self._registry.exclude(name)
                    self._registry.register(incoming)

            names = sorted(desired)
            log_latency_event(
                logger,
                event="mcp_registry_refresh",
                duration_seconds=time.monotonic() - started,
                fields={"tools": len(names)},
            )
            return names

    def discover(self, *, auth_context: Any = None) -> dict[str, MCPToolProxy]:
        """Return auth-scoped MCP discovery map without mutating global registry."""
        started = time.monotonic()
        with self._lock:
            self._connection_manager.update_config(self._mcp_config)
            discovered = self._discover(
                auth_context=auth_context,
                enable_registry_refresh=False,
            )
        log_latency_event(
            logger,
            event="mcp_registry_discover",
            duration_seconds=time.monotonic() - started,
            fields={"tools": len(discovered)},
        )
        return discovered

    def connection_states(self) -> list[MCPConnectionState]:
        """Return runtime MCP connection state snapshots."""
        with self._lock:
            self._connection_manager.update_config(self._mcp_config)
            return self._connection_manager.states()

    def connect_alias(
        self,
        *,
        alias: str,
        timeout_seconds: int | None = None,
    ) -> MCPConnectionState:
        with self._lock:
            self._connection_manager.update_config(self._mcp_config)
            server = self._mcp_config.servers.get(alias)
            if server is None:
                raise RuntimeError(f"MCP server {alias!r} is not configured")
        return self._connection_manager.connect(
            alias=alias,
            server=server,
            timeout_seconds=timeout_seconds,
        )

    def disconnect_alias(self, *, alias: str) -> MCPConnectionState:
        with self._lock:
            self._connection_manager.update_config(self._mcp_config)
            server = self._mcp_config.servers.get(alias)
            if server is None:
                raise RuntimeError(f"MCP server {alias!r} is not configured")
        return self._connection_manager.disconnect(alias=alias, server=server)

    def reconnect_alias(
        self,
        *,
        alias: str,
        timeout_seconds: int | None = None,
    ) -> MCPConnectionState:
        with self._lock:
            self._connection_manager.update_config(self._mcp_config)
            server = self._mcp_config.servers.get(alias)
            if server is None:
                raise RuntimeError(f"MCP server {alias!r} is not configured")
        return self._connection_manager.reconnect(
            alias=alias,
            server=server,
            timeout_seconds=timeout_seconds,
        )

    def debug_snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._connection_manager.update_config(self._mcp_config)
            return self._connection_manager.debug_snapshot()

    def close(self) -> None:
        with self._lock:
            self._connection_manager.close_all()
