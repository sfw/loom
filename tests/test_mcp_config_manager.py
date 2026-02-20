"""Tests for MCP config manager and merge semantics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from loom.config import Config, MCPConfig, MCPServerConfig, load_config
from loom.mcp.config import (
    MCPConfigManager,
    load_mcp_file,
    load_merged_mcp_config,
    redact_server_env,
)


def _server(command: str, *, enabled: bool = True) -> MCPServerConfig:
    return MCPServerConfig(
        command=command,
        args=["--stdio"],
        env={"TOKEN": "secret", "REF": "${TOKEN}"},
        cwd="",
        timeout_seconds=30,
        enabled=enabled,
    )


def test_merged_mcp_precedence(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    workspace_mcp = workspace / ".loom" / "mcp.toml"
    workspace_mcp.parent.mkdir(parents=True)
    workspace_mcp.write_text(
        """
[mcp.servers.shared]
command = "workspace"

[mcp.servers.workspace_only]
command = "workspace-only"
"""
    )

    user_mcp = tmp_path / "user-mcp.toml"
    user_mcp.write_text(
        """
[mcp.servers.shared]
command = "user"

[mcp.servers.user_only]
command = "user-only"
"""
    )

    explicit_mcp = tmp_path / "explicit.toml"
    explicit_mcp.write_text(
        """
[mcp.servers.shared]
command = "explicit"

[mcp.servers.explicit_only]
command = "explicit-only"
"""
    )

    legacy = Config(
        mcp=MCPConfig(
            servers={
                "shared": _server("legacy"),
                "legacy_only": _server("legacy-only"),
            }
        )
    )

    merged = load_merged_mcp_config(
        config=legacy,
        workspace=workspace,
        explicit_path=explicit_mcp,
        user_path=user_mcp,
        legacy_config_path=tmp_path / "loom.toml",
    )

    servers = merged.config.servers
    assert servers["shared"].command == "explicit"
    assert servers["legacy_only"].command == "legacy-only"
    assert servers["user_only"].command == "user-only"
    assert servers["workspace_only"].command == "workspace-only"
    assert servers["explicit_only"].command == "explicit-only"
    assert merged.get("shared").source == "explicit"
    assert merged.get("legacy_only").source == "legacy"


def test_redaction_preserves_env_references():
    server = _server("demo")
    redacted = redact_server_env(server)
    assert redacted["TOKEN"] == "<redacted>"
    assert redacted["REF"] == "${TOKEN}"


def test_manager_add_edit_remove_with_explicit_path(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    explicit_mcp = tmp_path / "explicit.toml"
    user_mcp = tmp_path / "user.toml"

    manager = MCPConfigManager(
        config=Config(),
        workspace=workspace,
        explicit_path=explicit_mcp,
        user_path=user_mcp,
    )

    target = manager.add_server("demo", _server("python"))
    assert target == explicit_mcp
    assert load_mcp_file(explicit_mcp).servers["demo"].command == "python"

    updated_path, updated = manager.edit_server(
        "demo",
        lambda current: replace(current, enabled=False, timeout_seconds=45),
    )
    assert updated_path == explicit_mcp
    assert updated.enabled is False
    assert load_mcp_file(explicit_mcp).servers["demo"].timeout_seconds == 45

    removed_path = manager.remove_server("demo")
    assert removed_path == explicit_mcp
    assert manager.list_views() == []


def test_manager_migrate_legacy_moves_entries_and_removes_section(tmp_path: Path):
    loom_toml = tmp_path / "loom.toml"
    loom_toml.write_text(
        """
[server]
host = "127.0.0.1"
port = 9000

[mcp.servers.demo]
command = "python"
args = ["-m", "demo"]

[models.primary]
provider = "ollama"
base_url = "http://localhost:11434"
model = "qwen3:14b"
roles = ["executor"]
"""
    )
    base = load_config(loom_toml)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    explicit_mcp = tmp_path / "mcp.toml"

    manager = MCPConfigManager(
        config=base,
        workspace=workspace,
        explicit_path=explicit_mcp,
        legacy_config_path=loom_toml,
        user_path=tmp_path / "user.toml",
    )

    target, copied, removed = manager.migrate_legacy()

    assert target == explicit_mcp
    assert copied == 1
    assert removed is True
    assert "demo" in load_mcp_file(explicit_mcp).servers
    rewritten = loom_toml.read_text()
    assert "[mcp" not in rewritten
    assert "[server]" in rewritten
    assert "[models.primary]" in rewritten


def test_edit_legacy_alias_writes_override_to_target(tmp_path: Path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = tmp_path / "mcp.toml"
    manager = MCPConfigManager(
        config=Config(mcp=MCPConfig(servers={"legacy_demo": _server("legacy")})),
        workspace=workspace,
        explicit_path=target,
        user_path=tmp_path / "user.toml",
    )

    manager.edit_server(
        "legacy_demo",
        lambda current: replace(current, timeout_seconds=55),
    )

    saved = load_mcp_file(target).servers["legacy_demo"]
    assert saved.timeout_seconds == 55
    view = manager.get_view("legacy_demo")
    assert view is not None
    assert view.source == "explicit"

