"""Tests for error handling across the Loom system.

Covers model connection errors, config parsing errors, and
graceful degradation paths.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from loom.config import ConfigError, load_config
from loom.models.base import ModelConnectionError

# ---------------------------------------------------------------------------
# ModelConnectionError
# ---------------------------------------------------------------------------


class TestModelConnectionError:
    def test_basic(self):
        err = ModelConnectionError("server down")
        assert str(err) == "server down"
        assert err.original is None

    def test_with_original(self):
        orig = ConnectionError("refused")
        err = ModelConnectionError("can't connect", original=orig)
        assert err.original is orig
        assert "can't connect" in str(err)


# ---------------------------------------------------------------------------
# OpenAI provider error handling
# ---------------------------------------------------------------------------


class TestOpenAIProviderErrors:
    def _make_provider(self):
        from loom.config import ModelConfig
        from loom.models.openai_provider import OpenAICompatibleProvider

        config = ModelConfig(
            provider="openai_compatible",
            base_url="http://localhost:9999",
            model="test-model",
        )
        return OpenAICompatibleProvider(config)

    @pytest.mark.asyncio
    async def test_complete_connect_error(self):
        provider = self._make_provider()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(
            side_effect=httpx.ConnectError("refused"),
        )
        provider._client.base_url = "http://localhost:9999"

        with pytest.raises(ModelConnectionError, match="Cannot connect"):
            await provider.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_complete_timeout(self):
        provider = self._make_provider()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("timed out"),
        )

        with pytest.raises(ModelConnectionError, match="timed out"):
            await provider.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_complete_http_error(self):
        provider = self._make_provider()
        provider._client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"
        mock_response.raise_for_status.side_effect = (
            httpx.HTTPStatusError(
                "429", request=MagicMock(), response=mock_response,
            )
        )
        provider._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(ModelConnectionError, match="HTTP 429"):
            await provider.complete([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Ollama provider error handling
# ---------------------------------------------------------------------------


class TestOllamaProviderErrors:
    def _make_provider(self):
        from loom.config import ModelConfig
        from loom.models.ollama_provider import OllamaProvider

        config = ModelConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3",
        )
        return OllamaProvider(config)

    @pytest.mark.asyncio
    async def test_complete_connect_error(self):
        provider = self._make_provider()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(
            side_effect=httpx.ConnectError("refused"),
        )
        provider._client.base_url = "http://localhost:11434"

        with pytest.raises(ModelConnectionError, match="Cannot connect"):
            await provider.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_complete_timeout(self):
        provider = self._make_provider()
        provider._client = AsyncMock()
        provider._client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("timed out"),
        )

        with pytest.raises(ModelConnectionError, match="timed out"):
            await provider.complete([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Anthropic provider error handling
# ---------------------------------------------------------------------------


class TestAnthropicProviderErrors:
    def _make_provider(self):
        from loom.models.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            name="test",
            model="claude-sonnet-4-5-20250929",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_complete_connect_error(self):
        provider = self._make_provider()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("refused"),
        )
        provider._client = mock_client
        provider._base_url = "https://api.anthropic.com"

        with pytest.raises(ModelConnectionError, match="Cannot connect"):
            await provider.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_complete_auth_error(self):
        provider = self._make_provider()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "invalid api key"
        mock_response.raise_for_status.side_effect = (
            httpx.HTTPStatusError(
                "401", request=MagicMock(), response=mock_response,
            )
        )
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        with pytest.raises(
            ModelConnectionError, match="Invalid Anthropic API key",
        ):
            await provider.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_complete_rate_limit(self):
        provider = self._make_provider()
        mock_client = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"
        mock_response.raise_for_status.side_effect = (
            httpx.HTTPStatusError(
                "429", request=MagicMock(), response=mock_response,
            )
        )
        mock_client.post = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        with pytest.raises(
            ModelConnectionError, match="rate limit exceeded",
        ):
            await provider.complete([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Config error handling
# ---------------------------------------------------------------------------


class TestConfigErrors:
    def test_malformed_toml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False,
        ) as f:
            f.write("[server\nhost = 'bad'")  # invalid TOML
            f.flush()
            path = Path(f.name)

        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_config(path)

        path.unlink()

    def test_missing_file_returns_defaults(self):
        config = load_config(Path("/nonexistent/loom.toml"))
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000

    def test_config_error_class(self):
        err = ConfigError("bad config")
        assert str(err) == "bad config"


# ---------------------------------------------------------------------------
# CLI network error handling
# ---------------------------------------------------------------------------


class TestCLINetworkErrors:
    @pytest.mark.asyncio
    async def test_check_status_connect_error(self):
        """_check_status should handle connection errors."""
        from loom.__main__ import _check_status

        with pytest.raises(SystemExit):
            await _check_status("http://localhost:1", "task-123")

    @pytest.mark.asyncio
    async def test_cancel_task_connect_error(self):
        """_cancel_task should handle connection errors."""
        from loom.__main__ import _cancel_task

        with pytest.raises(SystemExit):
            await _cancel_task("http://localhost:1", "task-123")
