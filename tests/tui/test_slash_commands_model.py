"""TUI model slash command tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


class TestModelSlashCommands:
    @pytest.mark.asyncio
    async def test_model_without_args_renders_detailed_active_model(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        active_model = SimpleNamespace(
            name="primary",
            model="gpt-4o-mini",
            tier=2,
            roles=["executor", "planner"],
            configured_temperature=0.15,
            configured_max_tokens=16000,
            _config=SimpleNamespace(reasoning_effort="medium"),
            _capabilities=SimpleNamespace(
                vision=True,
                native_pdf=False,
                thinking=False,
                citations=False,
                audio_input=False,
                audio_output=False,
            ),
        )

        app = LoomApp(
            model=active_model,
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "primary": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    max_tokens=16000,
                    temperature=0.15,
                    roles=["executor", "planner"],
                    api_key="sk-test-secret",
                    reasoning_effort="medium",
                )
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/model")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Active Model" in rendered
        assert "primary" in rendered
        assert "openai-chat-completions" in rendered
        assert "https://api.example.com/v1" in rendered
        assert "gpt-4o-mini" in rendered
        assert "sk-test-secret" not in rendered

    @pytest.mark.asyncio
    async def test_model_with_args_shows_usage(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/model secondary")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Usage" in rendered
        assert "/model" in rendered
        assert "Runtime model switching is not supported yet." in rendered

    @pytest.mark.asyncio
    async def test_models_without_args_renders_catalog_and_redacts(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        active_model = SimpleNamespace(
            name="primary",
            model="gpt-4o-mini",
            tier=2,
            roles=["executor"],
            configured_temperature=0.1,
            configured_max_tokens=8192,
        )
        app = LoomApp(
            model=active_model,
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "primary": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1?token=leak",
                    model="gpt-4o-mini",
                    api_key="sk-primary-secret",
                    roles=["executor"],
                ),
                "secondary": ModelConfig(
                    provider="anthropic",
                    base_url="",
                    model="claude-3-5-sonnet-20241022",
                    api_key="sk-secondary-secret",
                    roles=["planner"],
                ),
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/models")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Configured Models" in rendered
        assert "primary" in rendered
        assert "secondary" in rendered
        assert rendered.count("active:[/] yes") == 1
        assert "https://api.example.com/v1" in rendered
        assert "https://api.anthropic.com" in rendered
        assert "sk-primary-secret" not in rendered
        assert "sk-secondary-secret" not in rendered
        assert "token=leak" not in rendered

    @pytest.mark.asyncio
    async def test_models_with_args_shows_usage(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/models anything")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "Usage" in rendered
        assert "/models" in rendered

    @pytest.mark.asyncio
    async def test_model_invalid_endpoint_never_echoes_raw_value(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="primary", model="gpt-4o-mini", tier=1, roles=["executor"]),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "primary": ModelConfig(
                    provider="openai_compatible",
                    base_url="not-a-url?token=secret",
                    model="gpt-4o-mini",
                    roles=["executor"],
                )
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/model")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "(invalid-configured-url)" in rendered
        assert "token=secret" not in rendered

    @pytest.mark.asyncio
    async def test_models_catalog_orders_active_alias_first(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(name="zzz", model="gpt-4o-mini", tier=1, roles=["executor"]),
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "aaa": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    roles=["executor"],
                ),
                "zzz": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    roles=["executor"],
                ),
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/models")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert rendered.index("[bold]zzz[/bold]") < rendered.index("[bold]aaa[/bold]")

    @pytest.mark.asyncio
    async def test_models_with_no_config_shows_runtime_only(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=SimpleNamespace(
                name="ephemeral",
                model="qwen3:8b",
                tier=1,
                roles=["executor"],
                configured_temperature=0.2,
                configured_max_tokens=4096,
                _config=SimpleNamespace(provider="ollama", base_url="http://localhost:11434"),
            ),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/models")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "No configured models." in rendered
        assert "(runtime-only)" in rendered

    @pytest.mark.asyncio
    async def test_model_marks_alias_as_ambiguous(self):
        from loom.config import Config, ModelConfig
        from loom.tui.app import LoomApp

        runtime_model = SimpleNamespace(
            name="runtime",
            model="gpt-4o-mini",
            tier=2,
            roles=["executor"],
            configured_temperature=0.1,
            configured_max_tokens=8192,
            _config=SimpleNamespace(
                provider="openai_compatible",
                base_url="https://api.example.com/v1",
            ),
        )
        app = LoomApp(
            model=runtime_model,
            tools=MagicMock(),
            workspace=Path("/tmp"),
            config=Config(models={
                "alpha": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    roles=["executor"],
                ),
                "beta": ModelConfig(
                    provider="openai_compatible",
                    base_url="https://api.example.com/v1",
                    model="gpt-4o-mini",
                    roles=["executor"],
                ),
            }),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/model")

        assert handled is True
        chat.add_info.assert_called_once()
        rendered = chat.add_info.call_args.args[0]
        assert "active_alias:[/] ambiguous" in rendered
        assert "candidates:[/] alpha, beta" in rendered
