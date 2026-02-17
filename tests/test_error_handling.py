"""Tests for error handling across the Loom system.

Covers model connection errors, config parsing errors, and
graceful degradation paths.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from loom.config import ConfigError, load_config
from loom.models.base import ModelConnectionError


class _StreamingHTTPErrorResponse:
    """Simulate httpx streaming responses where .text is unavailable."""

    def __init__(
        self,
        status_code: int,
        *,
        body: str = "",
        fail_aread: bool = False,
    ) -> None:
        self.status_code = status_code
        self.headers = {}
        self._body = body
        self._fail_aread = fail_aread

    async def aread(self) -> bytes:
        if self._fail_aread:
            raise RuntimeError("read failed")
        return self._body.encode("utf-8")

    @property
    def text(self) -> str:
        raise RuntimeError(
            "Attempted to access streaming response content, "
            "without having called `read()`.",
        )

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            str(self.status_code),
            request=MagicMock(),
            response=self,
        )

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400


class _StreamingSuccessResponse:
    """Simulate a successful streaming response with SSE lines."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self.headers = {}
        self._lines = lines

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _HTTPErrorResponse:
    """Simple non-streaming response that raises HTTPStatusError."""

    def __init__(self, status_code: int, body: str = "") -> None:
        self.status_code = status_code
        self.headers = {}
        self._body = body

    async def aread(self) -> bytes:
        return self._body.encode("utf-8")

    @property
    def text(self) -> str:
        return self._body

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            str(self.status_code),
            request=MagicMock(),
            response=self,
        )


class _ErrorStreamContext:
    """Async context manager that yields a response raising HTTPStatusError."""

    def __init__(self, response) -> None:
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


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
    def _make_provider(self, *, api_key: str = ""):
        from loom.config import ModelConfig
        from loom.models.openai_provider import OpenAICompatibleProvider

        config = ModelConfig(
            provider="openai_compatible",
            base_url="http://localhost:9999",
            model="test-model",
            api_key=api_key,
        )
        return OpenAICompatibleProvider(config)

    def test_constructor_applies_bearer_auth_header_when_api_key_is_set(self):
        provider = self._make_provider(api_key="test-key")
        try:
            assert provider._client.headers.get("authorization") == "Bearer test-key"
        finally:
            asyncio.run(provider._client.aclose())

    def test_constructor_omits_auth_header_when_api_key_is_empty(self):
        provider = self._make_provider(api_key="")
        try:
            assert provider._client.headers.get("authorization") is None
        finally:
            asyncio.run(provider._client.aclose())

    def test_build_messages_normalizes_assistant_tool_calls(self):
        provider = self._make_provider()
        try:
            messages = [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "a.md"},
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "",
                    "content": None,
                },
            ]

            normalized = provider._build_openai_messages(messages)

            assert normalized[0]["content"] == ""
            assert normalized[0]["tool_calls"][0]["id"].startswith("call_")
            assert isinstance(
                normalized[0]["tool_calls"][0]["function"]["arguments"], str,
            )
            assert normalized[1]["content"] == ""
            assert normalized[1]["tool_call_id"] == normalized[0]["tool_calls"][0]["id"]
        finally:
            asyncio.run(provider._client.aclose())

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

    @pytest.mark.asyncio
    async def test_complete_retries_with_reasoning_content_for_tool_calls(self):
        provider = self._make_provider(api_key="test-key")
        provider._client = AsyncMock()
        provider._client.base_url = "https://api.example.test/v1"

        first = _HTTPErrorResponse(
            400,
            body=(
                '{"error":{"message":"thinking is enabled but reasoning_content '
                'is missing in assistant tool call message at index 1"}}'
            ),
        )
        second = MagicMock()
        second.raise_for_status.return_value = None
        second.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        provider._client.post = AsyncMock(side_effect=[first, second])

        messages = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": "Calling tool",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_fetch",
                        "arguments": "{\"url\":\"https://example.com\"}",
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"ok\":true}",
            },
        ]

        response = await provider.complete(messages)
        assert response.text == "ok"
        assert provider._client.post.await_count == 2

        first_payload = provider._client.post.await_args_list[0].kwargs["json"]
        second_payload = provider._client.post.await_args_list[1].kwargs["json"]
        assert "reasoning_content" not in first_payload["messages"][1]
        assert second_payload["messages"][1]["reasoning_content"] == "Calling tool"

    @pytest.mark.asyncio
    async def test_complete_assigns_fallback_tool_call_id_when_missing(self):
        provider = self._make_provider()
        provider._client = AsyncMock()

        ok = MagicMock()
        ok.raise_for_status.return_value = None
        ok.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "write_file",
                            "arguments": "{\"path\":\"memo.md\",\"content\":\"x\"}",
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        provider._client.post = AsyncMock(return_value=ok)

        response = await provider.complete([{"role": "user", "content": "hi"}])

        assert response.tool_calls is not None
        assert response.tool_calls[0].id == "call_0"

    @pytest.mark.asyncio
    async def test_complete_retry_injects_non_empty_reasoning_content_when_content_empty(self):
        provider = self._make_provider(api_key="test-key")
        provider._client = AsyncMock()
        provider._client.base_url = "https://api.example.test/v1"

        first = _HTTPErrorResponse(
            400,
            body=(
                '{"error":{"message":"thinking is enabled but reasoning_content '
                'is missing in assistant tool call message at index 1"}}'
            ),
        )
        second = MagicMock()
        second.raise_for_status.return_value = None
        second.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        provider._client.post = AsyncMock(side_effect=[first, second])

        messages = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_fetch",
                        "arguments": "{\"url\":\"https://example.com\"}",
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"ok\":true}",
            },
        ]

        response = await provider.complete(messages)
        assert response.text == "ok"

        retry_payload = provider._client.post.await_args_list[1].kwargs["json"]
        assert (
            retry_payload["messages"][1]["reasoning_content"]
            == "Tool call required to continue."
        )

    @pytest.mark.asyncio
    async def test_stream_http_error_does_not_access_response_text(self):
        """Streaming HTTP errors should be reported without .text access crashes."""
        provider = self._make_provider()
        provider._client = MagicMock()
        response = _StreamingHTTPErrorResponse(
            401, body="unauthorized",
        )
        provider._client.stream = MagicMock(
            return_value=_ErrorStreamContext(response),
        )

        with pytest.raises(ModelConnectionError, match="HTTP 401") as exc:
            async for _chunk in provider.stream(
                [{"role": "user", "content": "hi"}],
            ):
                pass

        assert "unauthorized" in str(exc.value)

    @pytest.mark.asyncio
    async def test_stream_retries_with_reasoning_content_for_tool_calls(self):
        provider = self._make_provider(api_key="test-key")
        provider._client = MagicMock()

        first = _StreamingHTTPErrorResponse(
            400,
            body=(
                '{"error":{"message":"thinking is enabled but reasoning_content '
                'is missing in assistant tool call message at index 1"}}'
            ),
        )
        second = _StreamingSuccessResponse([
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ])
        provider._client.stream = MagicMock(side_effect=[
            _ErrorStreamContext(first),
            _ErrorStreamContext(second),
        ])

        messages = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_fetch",
                        "arguments": "{\"url\":\"https://example.com\"}",
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "{\"ok\":true}",
            },
        ]

        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert "".join(c.text for c in chunks) == "ok"
        assert chunks[-1].done is True
        assert provider._client.stream.call_count == 2

        first_payload = provider._client.stream.call_args_list[0].kwargs["json"]
        second_payload = provider._client.stream.call_args_list[1].kwargs["json"]
        assert "reasoning_content" not in first_payload["messages"][1]
        assert (
            second_payload["messages"][1]["reasoning_content"]
            == "Tool call required to continue."
        )

    @pytest.mark.asyncio
    async def test_stream_assigns_fallback_tool_call_id_when_missing(self):
        provider = self._make_provider()
        provider._client = MagicMock()
        response = _StreamingSuccessResponse([
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"name":"write_file"}}]}}]}'
            ),
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{\\"path\\":\\"memo.md\\",\\"content\\":\\"x\\"}"}}]}}]}'
            ),
            "data: [DONE]",
        ])
        provider._client.stream = MagicMock(
            return_value=_ErrorStreamContext(response),
        )

        chunks = []
        async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert chunks[-1].done is True
        assert chunks[-1].tool_calls is not None
        assert chunks[-1].tool_calls[0].id == "call_0"


# ---------------------------------------------------------------------------
# Ollama provider error handling
# ---------------------------------------------------------------------------


class TestOllamaProviderErrors:
    def _make_provider(self, *, api_key: str = ""):
        from loom.config import ModelConfig
        from loom.models.ollama_provider import OllamaProvider

        config = ModelConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3",
            api_key=api_key,
        )
        return OllamaProvider(config)

    def test_constructor_applies_bearer_auth_header_when_api_key_is_set(self):
        provider = self._make_provider(api_key="test-key")
        try:
            assert provider._client.headers.get("authorization") == "Bearer test-key"
        finally:
            asyncio.run(provider._client.aclose())

    def test_constructor_omits_auth_header_when_api_key_is_empty(self):
        provider = self._make_provider(api_key="")
        try:
            assert provider._client.headers.get("authorization") is None
        finally:
            asyncio.run(provider._client.aclose())

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

    @pytest.mark.asyncio
    async def test_stream_http_error_with_unreadable_body(self):
        """Streaming HTTP errors should degrade gracefully when body is unreadable."""
        provider = self._make_provider()
        provider._client = MagicMock()
        response = _StreamingHTTPErrorResponse(
            503, fail_aread=True,
        )
        provider._client.stream = MagicMock(
            return_value=_ErrorStreamContext(response),
        )

        with pytest.raises(ModelConnectionError, match="HTTP 503"):
            async for _chunk in provider.stream(
                [{"role": "user", "content": "hi"}],
            ):
                pass


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

    @pytest.mark.asyncio
    async def test_stream_http_error_does_not_crash_on_response_text(self):
        provider = self._make_provider()
        provider._client = MagicMock()
        response = _StreamingHTTPErrorResponse(
            429, body="rate limited",
        )
        provider._client.stream = MagicMock(
            return_value=_ErrorStreamContext(response),
        )

        with pytest.raises(
            ModelConnectionError, match="rate limit exceeded",
        ) as exc:
            async for _chunk in provider.stream(
                [{"role": "user", "content": "hi"}],
            ):
                pass

        assert "rate limited" in str(exc.value)


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
