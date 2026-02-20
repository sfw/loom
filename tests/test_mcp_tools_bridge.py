"""Tests for MCP-backed tool discovery and execution bridge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from loom.config import Config, MCPConfig, MCPServerConfig
from loom.integrations.mcp_tools import register_mcp_tools
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


def _build_config_with_env(script_path: Path, env: dict[str, str]) -> Config:
    return Config(
        mcp=MCPConfig(
            servers={
                "demo": MCPServerConfig(
                    command=sys.executable,
                    args=[str(script_path)],
                    env=env,
                    timeout_seconds=20,
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
async def test_list_changed_notification_triggers_runtime_refresh(tmp_path):
    script, tools_file = _write_notifying_fake_mcp_server(tmp_path)
    cfg = _build_config_with_env(
        script,
        {"TOOLS_FILE": str(tools_file)},
    )
    registry = create_default_registry(cfg)
    assert registry.has("mcp.demo.echo")
    assert not registry.has("mcp.demo.ping")

    result = await registry.execute(
        "mcp.demo.echo",
        {"text": "refresh", "mutate": True},
        workspace=tmp_path,
    )
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
