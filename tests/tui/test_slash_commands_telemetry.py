"""TUI telemetry slash command tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestTelemetrySlashCommands:
    @pytest.mark.asyncio
    async def test_telemetry_status_shows_process_local_snapshot(self):
        from loom.config import Config, TelemetryConfig
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(
                telemetry=TelemetryConfig(
                    mode="active",
                    configured_mode_input="active",
                ),
            ),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/telemetry")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Telemetry Mode" in rendered
        assert "configured: [bold]active[/bold]" in rendered
        assert "runtime override: [bold](none)[/bold]" in rendered
        assert "effective: [bold]active[/bold]" in rendered
        assert "scope: [bold]process_local[/bold]" in rendered

    @pytest.mark.asyncio
    async def test_telemetry_set_mode_normalizes_alias(self):
        from loom.config import Config, TelemetryConfig
        from loom.tools.registry import ToolRegistry
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=ToolRegistry(),
            workspace=Path("/tmp"),
            config=Config(
                telemetry=TelemetryConfig(
                    mode="active",
                    configured_mode_input="active",
                ),
            ),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/telemetry internal_only")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "effective mode: [bold]all_typed[/bold]" in rendered
        assert "telemetry_mode_alias_normalized" in rendered
        assert app._effective_telemetry_mode() == "all_typed"
