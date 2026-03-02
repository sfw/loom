"""Tests for MCP-backed tool discovery and execution bridge."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

import loom.integrations.mcp_tools as mcp_tools_module
from loom.config import Config, MCPConfig, MCPOAuthConfig, MCPServerConfig
from loom.integrations.mcp.oauth import MCPOAuthReadiness
from loom.integrations.mcp_tools import (
    MCPConnectionManager,
    _MCPRemoteHTTPClient,
    _MCPStdioClient,
    register_mcp_tools,
    runtime_connection_states,
)
from loom.mcp.config import apply_mcp_overrides
from loom.tools import create_default_registry
from loom.tools.registry import ToolRegistry


def _write_fake_mcp_server(
    tmp_path: Path,
    *,
    tools: list[dict],
) -> Path:
    script_path = tmp_path / "fake_mcp_server.py"
    payload = json.dumps(tools, ensure_ascii=True)
    script_path.write_text(
        f"""\
import json
import sys

TOOLS = json.loads({payload!r})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({{
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {{"capabilities": {{"tools": {{"listChanged": True}}}}}},
        }}), flush=True)
        continue
    if method == "notifications/initialized":
        continue
    if method == "tools/list":
        print(json.dumps({{
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {{"tools": TOOLS}},
        }}), flush=True)
        continue
    if method == "tools/call":
        params = msg.get("params", {{}})
        name = params.get("name", "")
        arguments = params.get("arguments", {{}})
        if any(t.get("name") == name for t in TOOLS):
            text = f"{{name}}:{{arguments.get('text', '')}}"
            result = {{
                "content": [{{"type": "text", "text": text}}],
                "meta": {{"ok": True}},
            }}
        else:
            result = {{
                "isError": True,
                "content": [{{"type": "text", "text": "unknown tool"}}],
            }}
        print(json.dumps({{
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }}), flush=True)
        continue
    if req_id is not None:
        print(json.dumps({{
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {{"code": -32601, "message": f"Unknown method: {{method}}"}},
        }}), flush=True)
"""
    )
    return script_path


def _write_env_echo_mcp_server(tmp_path: Path) -> Path:
    script_path = tmp_path / "env_echo_mcp_server.py"
    script_path.write_text(
        """\
import json
import os
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"capabilities": {"tools": {"listChanged": True}}},
        }), flush=True)
        continue
    if method == "notifications/initialized":
        continue
    if method == "tools/list":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": [{
                "name": "echo_env",
                "description": "Echo env token",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }]},
        }), flush=True)
        continue
    if method == "tools/call":
        params = msg.get("params", {})
        arguments = params.get("arguments", {})
        token = os.environ.get("MCP_TOKEN", "")
        text = arguments.get("text", "")
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": f"{token}:{text}"}]},
        }), flush=True)
        continue
"""
    )
    return script_path


def _write_auth_gated_mcp_server(tmp_path: Path) -> Path:
    script_path = tmp_path / "auth_gated_mcp_server.py"
    script_path.write_text(
        """\
import json
import os
import sys

EXPECTED = "run-token"

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"capabilities": {"tools": {"listChanged": True}}},
        }), flush=True)
        continue
    if method == "notifications/initialized":
        continue
    if method == "tools/list":
        token = os.environ.get("MCP_TOKEN", "")
        tools = []
        if token == EXPECTED:
            tools = [{
                "name": "echo_env",
                "description": "Echo env token",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            }]
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": tools},
        }), flush=True)
        continue
    if method == "tools/call":
        params = msg.get("params", {})
        arguments = params.get("arguments", {})
        token = os.environ.get("MCP_TOKEN", "")
        text = arguments.get("text", "")
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": f"{token}:{text}"}]},
        }), flush=True)
        continue
"""
    )
    return script_path


def _build_config(script_path: Path) -> Config:
    return Config(
        mcp=MCPConfig(
            servers={
                "demo": MCPServerConfig(
                    command=sys.executable,
                    args=[str(script_path)],
                    env={},
                    timeout_seconds=20,
                )
            }
        )
    )


def _build_config_with_env(
    script_path: Path,
    env: dict[str, str],
    *,
    timeout_seconds: int = 20,
) -> Config:
    return Config(
        mcp=MCPConfig(
            servers={
                "demo": MCPServerConfig(
                    command=sys.executable,
                    args=[str(script_path)],
                    env=env,
                    timeout_seconds=timeout_seconds,
                )
            }
        )
    )


def _write_stateful_fake_mcp_server(tmp_path: Path) -> tuple[Path, Path]:
    tools_file = tmp_path / "tools.json"
    tools_file.write_text(json.dumps([
        {
            "name": "echo",
            "description": "Echo tool",
            "inputSchema": {"type": "object"},
        }
    ]))

    script_path = tmp_path / "stateful_fake_mcp_server.py"
    script_path.write_text(
        """\
import json
import os
import sys

TOOLS_FILE = os.environ["TOOLS_FILE"]

def read_tools():
    with open(TOOLS_FILE) as f:
        return json.load(f)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"capabilities": {"tools": {"listChanged": True}}},
        }), flush=True)
        continue
    if method == "notifications/initialized":
        continue
    if method == "tools/list":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": read_tools()},
        }), flush=True)
        continue
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        tools = read_tools()
        if any(t.get("name") == name for t in tools):
            result = {
                "content": [{"type": "text", "text": f"{name}:{arguments.get('text', '')}"}],
            }
        else:
            result = {
                "isError": True,
                "content": [{"type": "text", "text": "unknown tool"}],
            }
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }), flush=True)
        continue
"""
    )
    return script_path, tools_file


def _write_notifying_fake_mcp_server(tmp_path: Path) -> tuple[Path, Path]:
    tools_file = tmp_path / "tools_notify.json"
    tools_file.write_text(json.dumps([
        {
            "name": "echo",
            "description": "Echo tool",
            "inputSchema": {"type": "object"},
        }
    ]))

    script_path = tmp_path / "notify_fake_mcp_server.py"
    script_path.write_text(
        """\
import json
import os
import sys

TOOLS_FILE = os.environ["TOOLS_FILE"]

def read_tools():
    with open(TOOLS_FILE) as f:
        return json.load(f)

def write_tools(tools):
    with open(TOOLS_FILE, "w") as f:
        json.dump(tools, f)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    if method == "initialize":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"capabilities": {"tools": {"listChanged": True}}},
        }), flush=True)
        continue
    if method == "notifications/initialized":
        continue
    if method == "tools/list":
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": read_tools()},
        }), flush=True)
        continue
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        tools = read_tools()
        if arguments.get("mutate"):
            tools.append({
                "name": "ping",
                "description": "Ping tool",
                "inputSchema": {"type": "object"},
            })
            write_tools(tools)
            print(json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/tools/list_changed",
                "params": {},
            }), flush=True)
        if any(t.get("name") == name for t in tools):
            result = {
                "content": [{"type": "text", "text": f"{name}:{arguments.get('text', '')}"}],
            }
        else:
            result = {
                "isError": True,
                "content": [{"type": "text", "text": "unknown tool"}],
            }
        print(json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }), flush=True)
        continue
"""
    )
    return script_path, tools_file


def test_registers_namespaced_mcp_tools(tmp_path):
    script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {
                "name": "echo",
                "description": "Echo tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            }
        ],
    )
    cfg = _build_config(script)
    registry = ToolRegistry()

    registered = register_mcp_tools(registry, mcp_config=cfg.mcp)

    assert registered == ["mcp.demo.echo"]
    assert registry.has("mcp.demo.echo")


def test_register_mcp_tools_skips_remote_servers_without_bridge_support(tmp_path):
    cfg = Config(
        mcp=MCPConfig(
            servers={
                "remote_demo": MCPServerConfig(
                    type="remote",
                    url="https://api.example.com/mcp",
                    headers={"Authorization": "${TOKEN}"},
                    enabled=True,
                )
            }
        )
    )
    registry = ToolRegistry()

    registered = register_mcp_tools(registry, mcp_config=cfg.mcp)

    assert registered == []
    assert registry.list_tools() == []


def test_mcp_client_terminate_closes_pipes_for_exited_process() -> None:
    class _BrokenPipeStream:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True
            raise BrokenPipeError("[Errno 32] Broken pipe")

    class _TrackingStream:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _ExitedProcess:
        def __init__(self) -> None:
            self.stdin = _BrokenPipeStream()
            self.stdout = _TrackingStream()
            self.stderr = _TrackingStream()

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            raise AssertionError("terminate() should not be called for exited process")

        def wait(self, timeout: float | None = None) -> int:
            raise AssertionError("wait() should not be called for exited process")

    proc = _ExitedProcess()
    _MCPStdioClient._terminate(proc)  # should not raise

    assert proc.stdin.closed is True
    assert proc.stdout.closed is True
    assert proc.stderr.closed is True


def test_mcp_client_send_request_reports_process_exit_context() -> None:
    class _BrokenStdin:
        def write(self, _payload: str) -> int:
            raise BrokenPipeError("[Errno 32] Broken pipe")

        def flush(self) -> None:
            raise BrokenPipeError("[Errno 32] Broken pipe")

    class _ErrStream:
        def read(self) -> str:
            return "oauth required"

    class _ExitedProcess:
        def __init__(self) -> None:
            self.stdin = _BrokenStdin()
            self.stderr = _ErrStream()

        def poll(self) -> int:
            return 1

    client = _MCPStdioClient(
        alias="demo",
        server=MCPServerConfig(command=sys.executable, args=[]),
    )
    with pytest.raises(RuntimeError) as exc:
        client._send_request(  # noqa: SLF001 - exercising write-path failure
            _ExitedProcess(),
            request_id=1,
            method="tools/list",
            params={},
        )
    text = str(exc.value)
    assert "tools/list" in text
    assert "Broken pipe" not in text
    assert "exit code 1" in text
    assert "oauth required" in text


def test_persistent_stdio_session_reuses_spawn_for_same_env(tmp_path, monkeypatch):
    script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {"name": "echo", "description": "Echo tool", "inputSchema": {"type": "object"}}
        ],
    )
    client = _MCPStdioClient(
        alias="demo",
        server=MCPServerConfig(
            command=sys.executable,
            args=[str(script)],
            timeout_seconds=20,
        ),
    )
    spawn_calls = 0
    original_spawn = client._spawn

    def _wrapped_spawn(*, env_overrides=None):
        nonlocal spawn_calls
        spawn_calls += 1
        return original_spawn(env_overrides=env_overrides)

    monkeypatch.setattr(client, "_spawn", _wrapped_spawn)
    client.list_tools()
    client.call_tool("echo", {"text": "one"})
    client.call_tool("echo", {"text": "two"})
    assert spawn_calls == 1

    client.call_tool("echo", {"text": "three"}, env_overrides={"MCP_TOKEN": "alt"})
    assert spawn_calls == 2
    client.close()


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_mcp_tool_executes_through_registry(tmp_path):
    script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {
                "name": "echo",
                "description": "Echo tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            }
        ],
    )
    cfg = _build_config(script)
    registry = ToolRegistry()
    register_mcp_tools(registry, mcp_config=cfg.mcp)

    result = await registry.execute(
        "mcp.demo.echo",
        {"text": "hello"},
        workspace=tmp_path,
    )
    assert result.success
    assert "echo:hello" in result.output


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_env_ref_values_expand_before_mcp_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv("REAL_MCP_TOKEN", "secret-123")
    script = _write_env_echo_mcp_server(tmp_path)
    cfg = _build_config_with_env(
        script,
        {"MCP_TOKEN": "${REAL_MCP_TOKEN}"},
    )
    registry = ToolRegistry()
    register_mcp_tools(registry, mcp_config=cfg.mcp)

    result = await registry.execute(
        "mcp.demo.echo_env",
        {"text": "hello"},
        workspace=tmp_path,
    )
    assert result.success
    assert "secret-123:hello" in result.output


def test_mcp_name_collisions_are_skipped(tmp_path):
    script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {"name": "alpha beta", "description": "A", "inputSchema": {"type": "object"}},
            {"name": "alpha+beta", "description": "B", "inputSchema": {"type": "object"}},
        ],
    )
    cfg = _build_config(script)
    registry = ToolRegistry()

    registered = register_mcp_tools(registry, mcp_config=cfg.mcp)

    assert registered == ["mcp.demo.alpha_beta"]
    assert registry.has("mcp.demo.alpha_beta")


def test_create_default_registry_includes_mcp_tools(tmp_path):
    script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {"name": "echo", "description": "Echo tool", "inputSchema": {"type": "object"}}
        ],
    )
    cfg = _build_config(script)

    registry = create_default_registry(cfg)

    assert registry.has("mcp.demo.echo")


def test_mcp_tools_satisfy_required_tool_checks(tmp_path):
    script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {"name": "echo", "description": "Echo tool", "inputSchema": {"type": "object"}}
        ],
    )
    cfg = _build_config(script)
    available = set(create_default_registry(cfg).list_tools())
    required = {"mcp.demo.echo"}

    missing = sorted(name for name in required if name not in available)
    assert missing == []


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_unknown_mcp_tool_triggers_runtime_refresh(tmp_path):
    script, tools_file = _write_stateful_fake_mcp_server(tmp_path)
    cfg = _build_config_with_env(
        script,
        {"TOOLS_FILE": str(tools_file)},
    )
    registry = create_default_registry(cfg)
    assert registry.has("mcp.demo.echo")

    # Simulate MCP server tool list change after registry bootstrap.
    tools_file.write_text(json.dumps([
        {
            "name": "echo",
            "description": "Echo tool",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "ping",
            "description": "Ping tool",
            "inputSchema": {"type": "object"},
        },
    ]))

    result = await registry.execute(
        "mcp.demo.ping",
        {"text": "now"},
        workspace=tmp_path,
    )
    assert result.success
    assert "ping:now" in result.output


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_runtime_refresh_routes_mcp_auth_via_profile_binding(tmp_path):
    script = _write_auth_gated_mcp_server(tmp_path)
    cfg = _build_config_with_env(
        script,
        {},
    )
    registry = create_default_registry(cfg)
    assert not registry.has("mcp.demo.echo_env")

    class _AuthContext:
        def env_for_mcp_alias(self, alias: str) -> dict[str, str]:
            if alias == "demo":
                return {"MCP_TOKEN": "run-token"}
            return {}

        def mcp_discovery_fingerprint(self) -> str:
            return "demo:run-token"

    auth_context = _AuthContext()
    assert registry.has("mcp.demo.echo_env", auth_context=auth_context)
    # Auth-scoped discovery must not leak into the global MCP registry view.
    assert not registry.has("mcp.demo.echo_env")

    result = await registry.execute(
        "mcp.demo.echo_env",
        {"text": "hello"},
        workspace=tmp_path,
        auth_context=auth_context,
    )
    assert result.success
    assert "run-token:hello" in result.output
    assert not registry.has("mcp.demo.echo_env")


def test_runtime_connection_states_exposed(tmp_path):
    script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {"name": "echo", "description": "Echo tool", "inputSchema": {"type": "object"}}
        ],
    )
    cfg = _build_config(script)
    registry = ToolRegistry()

    register_mcp_tools(registry, mcp_config=cfg.mcp)
    states = runtime_connection_states(registry)

    assert states
    demo = next(state for state in states if state.alias == "demo")
    assert demo.status in {"configured", "healthy", "connecting", "error"}
    synchronizer = getattr(registry, "_mcp_synchronizer", None)
    close_fn = getattr(synchronizer, "close", None)
    if callable(close_fn):
        close_fn()


def test_connection_manager_opens_circuit_after_repeated_failures(monkeypatch):
    server = MCPServerConfig(command=sys.executable, args=["-m", "demo"])
    manager = MCPConnectionManager(
        mcp_config=MCPConfig(servers={"demo": server})
    )

    class _FailingClient:
        def __init__(self, configured_server: MCPServerConfig) -> None:
            self.server = configured_server
            self.connected_pid = None

        def list_tools(self, **_kwargs):
            raise RuntimeError("dial failed")

        def call_tool(self, *_args, **_kwargs):
            raise RuntimeError("dial failed")

    failing_client = _FailingClient(server)
    monkeypatch.setattr(
        manager,
        "_client_for",
        lambda alias, srv: failing_client,
    )

    for _ in range(5):
        with pytest.raises(RuntimeError, match="dial failed"):
            manager.list_tools(alias="demo", server=server)

    with pytest.raises(RuntimeError, match="Circuit open"):
        manager.list_tools(alias="demo", server=server)

    state = manager.state_for(alias="demo", server=server)
    assert state.circuit_state == "open"
    assert state.status == "degraded"
    assert "reconnect" in state.remediation.lower()


def test_connection_manager_backpressure_guard(monkeypatch):
    server = MCPServerConfig(command=sys.executable, args=["-m", "demo"])
    manager = MCPConnectionManager(
        mcp_config=MCPConfig(servers={"demo": server})
    )

    monkeypatch.setattr(mcp_tools_module, "_MCP_GLOBAL_MAX_IN_FLIGHT", 1)
    monkeypatch.setattr(mcp_tools_module, "_MCP_PER_SERVER_MAX_IN_FLIGHT", 1)
    monkeypatch.setattr(mcp_tools_module, "_MCP_PER_SERVER_MAX_QUEUE", 1)

    started = threading.Event()
    release = threading.Event()
    worker_errors: list[Exception] = []

    class _SlowClient:
        def __init__(self, configured_server: MCPServerConfig) -> None:
            self.server = configured_server
            self.connected_pid = None

        def list_tools(self, **_kwargs):
            return []

        def call_tool(self, *_args, **_kwargs):
            started.set()
            release.wait(timeout=2.0)
            return {"content": [{"type": "text", "text": "ok"}]}, False

    slow_client = _SlowClient(server)
    monkeypatch.setattr(
        manager,
        "_client_for",
        lambda alias, srv: slow_client,
    )

    def _invoke_worker() -> None:
        try:
            manager.call_tool(
                alias="demo",
                server=server,
                name="echo",
                arguments={"text": "hello"},
            )
        except Exception as e:  # pragma: no cover - assertion captures failures
            worker_errors.append(e)

    first = threading.Thread(target=_invoke_worker)
    second = threading.Thread(target=_invoke_worker)
    first.start()
    assert started.wait(timeout=1.0), "first worker never entered MCP call"
    second.start()

    for _ in range(100):
        state = manager.state_for(alias="demo", server=server)
        if state.queue_depth >= 1:
            break
        time.sleep(0.01)

    with pytest.raises(RuntimeError, match="backpressure"):
        manager.call_tool(
            alias="demo",
            server=server,
            name="echo",
            arguments={"text": "overflow"},
        )

    release.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    assert first.is_alive() is False
    assert second.is_alive() is False
    assert worker_errors == []


def test_connection_manager_redacts_oauth_failure_reason(monkeypatch):
    server = MCPServerConfig(
        type="remote",
        url="https://api.example.com/mcp",
        oauth=MCPOAuthConfig(enabled=True, scopes=["read"]),
    )
    manager = MCPConnectionManager(
        mcp_config=MCPConfig(servers={"demo": server})
    )

    monkeypatch.setattr(
        mcp_tools_module,
        "ensure_mcp_oauth_ready",
        lambda _alias: MCPOAuthReadiness(
            ready=False,
            state="needs_auth",
            reason="refresh_token=secret Authorization=Bearer abc123",
        ),
    )

    ready = manager._remote_oauth_ready("demo", server)  # noqa: SLF001
    assert ready is False
    state = manager.state_for(alias="demo", server=server)
    assert state.status == "needs_auth"
    assert "secret" not in state.last_error
    assert "abc123" not in state.last_error
    assert "<redacted>" in state.last_error


def test_remote_http_client_oauth_header_overrides_static_authorization(monkeypatch):
    server = MCPServerConfig(
        type="remote",
        url="https://api.example.com/mcp",
        headers={"authorization": "Bearer stale-token"},
        oauth=MCPOAuthConfig(enabled=True),
    )
    client = _MCPRemoteHTTPClient(alias="demo", server=server)

    monkeypatch.setattr(
        mcp_tools_module,
        "bearer_auth_header_for_alias",
        lambda _alias: "Bearer live-token",
    )

    headers = client._headers()
    assert headers["Authorization"] == "Bearer live-token"
    assert "authorization" not in headers
    assert headers["Accept"] == "application/json, text/event-stream"


def test_remote_http_client_streamable_initialize_and_tools_list(monkeypatch):
    json_module = json

    class _FakeResponse:
        def __init__(
            self,
            *,
            status_code: int,
            headers: dict[str, str] | None = None,
            text: str = "",
            raise_after_lines: bool = False,
        ) -> None:
            self.status_code = status_code
            self.headers = headers or {}
            self.text = text
            self.content = text.encode("utf-8")
            self._raise_after_lines = bool(raise_after_lines)

        def json(self):
            return json.loads(self.text or "{}")

        def read(self) -> bytes:
            return self.content

        def iter_lines(self):  # noqa: ANN201
            yield from self.text.splitlines()
            if self._raise_after_lines:
                raise RuntimeError("stream remained open")

    class _FakeStream:
        def __init__(self, response: _FakeResponse) -> None:
            self._response = response

        def __enter__(self) -> _FakeResponse:
            return self._response

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

    calls: list[dict[str, object]] = []

    def _fake_stream(method, url, *, headers, json, timeout):  # noqa: ANN001
        assert method == "POST"
        calls.append({
            "url": url,
            "headers": dict(headers),
            "json": dict(json),
            "timeout": timeout,
        })
        method = str(json.get("method", "")).strip()
        if method == "initialize":
            init_payload = json_module.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"capabilities": {"tools": {"listChanged": True}}},
            }, separators=(",", ":"))
            return _FakeResponse(
                status_code=200,
                headers={
                    "content-type": "text/event-stream",
                    "Mcp-Session-Id": "session-1",
                },
                text=(
                    'event: message\n'
                    f"data: {init_payload}\n\n"
                ),
                raise_after_lines=True,
            )
        if method == "notifications/initialized":
            return _FakeStream(_FakeResponse(status_code=202, headers={}))
        if method == "tools/list":
            assert headers.get("Mcp-Session-Id") == "session-1"
            tools_payload = json_module.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "tools": [
                        {
                            "name": "demo_tool",
                            "description": "Demo",
                            "inputSchema": {"type": "object"},
                        }
                    ]
                },
            }, separators=(",", ":"))
            return _FakeResponse(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                text=(
                    'event: message\n'
                    f"data: {tools_payload}\n\n"
                ),
                raise_after_lines=True,
            )
        raise AssertionError(f"Unexpected method: {method}")

    def _dispatch_stream(method, url, *, headers, json, timeout):  # noqa: ANN001
        response = _fake_stream(
            method,
            url,
            headers=headers,
            json=json,
            timeout=timeout,
        )
        if isinstance(response, _FakeStream):
            return response
        return _FakeStream(response)

    monkeypatch.setattr(httpx, "stream", _dispatch_stream)
    monkeypatch.setattr(
        mcp_tools_module,
        "bearer_auth_header_for_alias",
        lambda _alias: "Bearer oauth-token",
    )

    client = _MCPRemoteHTTPClient(
        alias="demo",
        server=MCPServerConfig(
            type="remote",
            url="https://api.example.com/mcp",
            oauth=MCPOAuthConfig(enabled=True),
        ),
    )
    tools = client.list_tools()

    assert len(tools) == 1
    assert tools[0]["name"] == "demo_tool"
    assert [str(item["json"]["method"]) for item in calls] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
    ]
    assert calls[0]["headers"].get("Accept") == "application/json, text/event-stream"
    assert calls[2]["headers"].get("Authorization") == "Bearer oauth-token"


@pytest.mark.asyncio
@pytest.mark.mcp
async def test_list_changed_notification_triggers_runtime_refresh(tmp_path):
    script, tools_file = _write_notifying_fake_mcp_server(tmp_path)
    cfg = _build_config_with_env(
        script,
        {"TOOLS_FILE": str(tools_file)},
        timeout_seconds=60,
    )
    registry = create_default_registry(cfg)
    assert registry.has("mcp.demo.echo")
    assert not registry.has("mcp.demo.ping")

    # This interaction can be timing-sensitive on loaded CI workers, so allow
    # a few short retries before failing the behavior assertion.
    result = None
    for _attempt in range(3):
        result = await registry.execute(
            "mcp.demo.echo",
            {"text": "refresh", "mutate": True},
            workspace=tmp_path,
        )
        if result.success:
            break
        await asyncio.sleep(0.05)
    assert result is not None
    assert result.success
    assert "echo:refresh" in result.output
    assert registry.has("mcp.demo.ping")


def test_create_registry_loads_mcp_from_external_mcp_toml(tmp_path):
    script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {"name": "echo", "description": "Echo tool", "inputSchema": {"type": "object"}}
        ],
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    user_mcp = tmp_path / "user-mcp.toml"
    user_mcp.write_text(
        f"""
[mcp.servers.demo]
command = "{sys.executable}"
args = ["{script}"]
enabled = true
"""
    )

    cfg = apply_mcp_overrides(
        Config(),
        workspace=workspace,
        user_path=user_mcp,
    )
    registry = create_default_registry(cfg)
    assert registry.has("mcp.demo.echo")


def test_workspace_mcp_toml_overrides_user_layer(tmp_path):
    user_script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {
                "name": "user_tool",
                "description": "User-level",
                "inputSchema": {"type": "object"},
            }
        ],
    )
    workspace_script = _write_fake_mcp_server(
        tmp_path,
        tools=[
            {
                "name": "workspace_tool",
                "description": "Workspace-level",
                "inputSchema": {"type": "object"},
            }
        ],
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    workspace_mcp = workspace / ".loom" / "mcp.toml"
    workspace_mcp.parent.mkdir(parents=True)
    workspace_mcp.write_text(
        f"""
[mcp.servers.demo]
command = "{sys.executable}"
args = ["{workspace_script}"]
enabled = true
"""
    )

    user_mcp = tmp_path / "user-mcp.toml"
    user_mcp.write_text(
        f"""
[mcp.servers.demo]
command = "{sys.executable}"
args = ["{user_script}"]
enabled = true
"""
    )

    cfg = apply_mcp_overrides(
        Config(),
        workspace=workspace,
        user_path=user_mcp,
    )
    registry = create_default_registry(cfg)
    assert registry.has("mcp.demo.workspace_tool")
    assert not registry.has("mcp.demo.user_tool")
