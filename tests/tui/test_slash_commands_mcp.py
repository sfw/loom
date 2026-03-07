"""TUI MCP slash command tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestMCPSlashCommands:
    @pytest.mark.asyncio
    async def test_mcp_without_args_opens_manager(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._open_mcp_manager_screen = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp")

        assert handled is True
        app._open_mcp_manager_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_mcp_list_uses_manager(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        merged = MagicMock()
        merged.as_views.return_value = [
            SimpleNamespace(
                alias="demo",
                server=SimpleNamespace(enabled=True),
                source="user",
            )
        ]
        manager.load.return_value = merged
        app._mcp_manager = MagicMock(return_value=manager)
        app._render_mcp_list = MagicMock(return_value="mcp-catalog")
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp list")

        assert handled is True
        manager.load.assert_called_once()
        merged.as_views.assert_called_once()
        app._render_mcp_list.assert_called_once()
        chat.add_info.assert_called_once_with("mcp-catalog")

    @pytest.mark.asyncio
    async def test_mcp_list_shows_legacy_migration_hint(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        merged = MagicMock()
        merged.as_views.return_value = [
            SimpleNamespace(
                alias="legacy_demo",
                server=SimpleNamespace(enabled=True),
                source="legacy",
            )
        ]
        manager.load.return_value = merged
        app._mcp_manager = MagicMock(return_value=manager)
        app._render_mcp_list = MagicMock(return_value="mcp-catalog")
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp list")

        assert handled is True
        chat.add_info.assert_called_once()
        assert "loom mcp migrate" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_show_requires_alias(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._mcp_manager = MagicMock(return_value=MagicMock())
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp show")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Usage" in message
        assert "/mcp show" in message
        assert "<alias>" in message

    @pytest.mark.asyncio
    async def test_mcp_show_missing_alias(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        manager.get_view.return_value = None
        app._mcp_manager = MagicMock(return_value=manager)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp show missing")

        assert handled is True
        manager.get_view.assert_called_once_with("missing")
        chat.add_info.assert_called_once()
        assert "MCP server not found" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_enable_edits_and_reloads(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        manager.edit_server.return_value = (Path("/tmp/mcp.toml"), MagicMock())
        app._mcp_manager = MagicMock(return_value=manager)
        app._reload_mcp_runtime = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp enable demo")

        assert handled is True
        manager.edit_server.assert_called_once()
        app._reload_mcp_runtime.assert_awaited_once()
        chat.add_info.assert_called_once()
        assert "enabled" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_remove_edits_and_reloads(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        manager.remove_server.return_value = Path("/tmp/mcp.toml")
        app._mcp_manager = MagicMock(return_value=manager)
        app._reload_mcp_runtime = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp remove demo")

        assert handled is True
        manager.remove_server.assert_called_once_with("demo")
        app._reload_mcp_runtime.assert_awaited_once()
        chat.add_info.assert_called_once()
        assert "removed" in chat.add_info.call_args.args[0]

    @pytest.mark.asyncio
    async def test_mcp_test_reports_discovered_tools(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        view = SimpleNamespace(alias="demo")
        manager.probe_server.return_value = (
            view,
            [{"name": "echo"}, {"name": "ping"}],
        )
        app._mcp_manager = MagicMock(return_value=manager)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp test demo")

        assert handled is True
        manager.probe_server.assert_called_once_with("demo")
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Tools discovered: 2" in message
        assert "echo" in message
        assert "ping" in message

    @pytest.mark.asyncio
    async def test_mcp_manage_routes_to_manager_screen(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._open_mcp_manager_screen = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/mcp manage")

        assert handled is True
        app._open_mcp_manager_screen.assert_called_once()

    def test_mcp_manager_inherits_explicit_config_paths(self, monkeypatch):
        from loom.tui.app import LoomApp

        captured: dict[str, object] = {}

        class DummyManager:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("loom.mcp.config.MCPConfigManager", DummyManager)

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=MagicMock(),
            explicit_mcp_path=Path("/tmp/override-mcp.toml"),
            legacy_config_path=Path("/tmp/loom.toml"),
        )

        manager = app._mcp_manager()

        assert isinstance(manager, DummyManager)
        assert captured["config"] is None
        assert captured["explicit_path"] == Path("/tmp/override-mcp.toml")
        assert captured["legacy_config_path"] == Path("/tmp/loom.toml")

    @pytest.mark.asyncio
    async def test_mcp_add_parses_flags_and_adds_server(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        manager = MagicMock()
        manager.add_server.return_value = Path("/tmp/mcp.toml")
        app._mcp_manager = MagicMock(return_value=manager)
        app._reload_mcp_runtime = AsyncMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command(
            "/mcp add demo --command python --arg -m --arg demo.server "
            "--env-ref TOKEN=REAL_TOKEN --timeout 45",
        )

        assert handled is True
        manager.add_server.assert_called_once()
        app._reload_mcp_runtime.assert_awaited_once()
        chat.add_info.assert_called_once()
        assert "added" in chat.add_info.call_args.args[0]
