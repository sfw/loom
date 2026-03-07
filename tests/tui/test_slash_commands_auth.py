"""TUI auth slash command tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestAuthSlashCommands:
    @pytest.mark.asyncio
    async def test_auth_without_args_opens_manager(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()
        app._open_auth_manager_screen = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/auth")

        assert handled is True
        app._open_auth_manager_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_auth_manage_routes_to_manager_screen(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._open_auth_manager_screen = MagicMock()
        app._refresh_process_command_index = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/auth manage")

        assert handled is True
        app._open_auth_manager_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_auth_non_manager_subcommand_shows_cli_handoff(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._refresh_process_command_index = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/auth list")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "manager-first" in message
        assert "/auth manage" in message
        assert "loom auth" in message
