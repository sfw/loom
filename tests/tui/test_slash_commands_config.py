"""TUI `/config` slash command tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestConfigSlashCommands:
    @pytest.mark.asyncio
    async def test_config_without_args_renders_help(self, tmp_path: Path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            legacy_config_path=tmp_path / "loom.toml",
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/config")

        assert handled is True
        rendered = chat.add_info.call_args.args[0]
        assert "/config list" in rendered
        assert str(tmp_path / "loom.toml") in rendered

    @pytest.mark.asyncio
    async def test_config_show_displays_snapshot(self):
        from loom.config import Config, TelemetryConfig
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(
                telemetry=TelemetryConfig(mode="active", configured_mode_input="active"),
            ),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/config show telemetry.mode")

        assert handled is True
        rendered = chat.add_info.call_args.args[0]
        assert "Config: telemetry.mode" in rendered
        assert "default: [bold]active[/bold]" in rendered
        assert "configured: [bold]active[/bold]" in rendered
        assert "applies to active runs: [bold]yes[/bold]" in rendered

    @pytest.mark.asyncio
    async def test_config_set_runtime_updates_effective_value_and_telemetry(self):
        from loom.config import Config, TelemetryConfig
        from loom.tools.registry import ToolRegistry
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=ToolRegistry(),
            workspace=Path("/tmp"),
            config=Config(
                telemetry=TelemetryConfig(mode="active", configured_mode_input="active"),
            ),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/config set telemetry.mode internal_only")

        assert handled is True
        rendered = chat.add_info.call_args.args[0]
        assert "Config updated." in rendered
        assert "telemetry_mode_alias_normalized" in rendered
        assert "effective: [bold]all_typed[/bold]" in rendered
        assert app._effective_telemetry_mode() == "all_typed"

    @pytest.mark.asyncio
    async def test_config_set_persist_writes_loom_toml(self, tmp_path: Path):
        from loom.tools.registry import ToolRegistry
        from loom.tui.app import LoomApp

        config_path = tmp_path / "loom.toml"
        app = LoomApp(
            model=MagicMock(name="model"),
            tools=ToolRegistry(),
            workspace=Path("/tmp"),
            legacy_config_path=config_path,
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command(
            "/config set tui.run_launch_timeout_seconds 90 --scope persist",
        )

        assert handled is True
        assert "run_launch_timeout_seconds = 90" in config_path.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_config_reset_runtime_clears_override(self):
        from loom.config import Config, TelemetryConfig
        from loom.tools.registry import ToolRegistry
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=ToolRegistry(),
            workspace=Path("/tmp"),
            config=Config(
                telemetry=TelemetryConfig(mode="active", configured_mode_input="active"),
            ),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        await app._handle_slash_command("/config set telemetry.mode debug")
        handled = await app._handle_slash_command("/config reset telemetry.mode --scope runtime")

        assert handled is True
        rendered = chat.add_info.call_args.args[0]
        assert "Config reset." in rendered
        assert "runtime override: [bold](none)[/bold]" in rendered
        assert app._effective_telemetry_mode() == "active"


class TestConfigSlashCompletionAndHints:
    def test_slash_completion_candidates_include_config(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._create_process_loader = MagicMock(return_value=SimpleNamespace(
            list_available=MagicMock(return_value=[]),
        ))

        assert app._slash_completion_candidates("/c") == ["/config", "/clear"]

    def test_slash_tab_completion_cycles_config_subcommands(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        input_widget = SimpleNamespace(value="/config s", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/config search"

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/config show"

    def test_slash_tab_completion_config_paths_and_enum_values(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        input_widget = SimpleNamespace(value="/config set tele", cursor_position=0)
        app.query_one = MagicMock(return_value=input_widget)

        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/config set telemetry.mode"

        input_widget.value = "/config set telemetry.mode d"
        assert app._apply_slash_tab_completion(reverse=False) is True
        assert input_widget.value == "/config set telemetry.mode debug"

    def test_render_slash_hint_for_config_path_prefix(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        hint = app._render_slash_hint("/config show tele")

        assert "Matching config paths for tele" in hint
        assert "telemetry.mode" in hint
