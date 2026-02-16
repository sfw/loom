"""Tests for the streaming model interface."""

from __future__ import annotations

from loom.models.base import (
    ModelProvider,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCall,
)


class FakeProvider(ModelProvider):
    """Minimal provider for testing the default stream() fallback."""

    async def complete(self, messages, tools=None, temperature=None,
                       max_tokens=None, response_format=None):
        return ModelResponse(
            text="Hello world",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "x"})],
            usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            model="fake",
        )

    async def health_check(self):
        return True

    @property
    def name(self):
        return "fake"

    @property
    def tier(self):
        return 1

    @property
    def roles(self):
        return ["executor"]


class TestStreamChunk:
    def test_defaults(self):
        chunk = StreamChunk()
        assert chunk.text == ""
        assert chunk.done is False
        assert chunk.tool_calls is None
        assert chunk.usage is None

    def test_with_values(self):
        chunk = StreamChunk(
            text="hello",
            done=True,
            tool_calls=[ToolCall(id="1", name="t", arguments={})],
            usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        )
        assert chunk.text == "hello"
        assert chunk.done is True
        assert len(chunk.tool_calls) == 1


class TestDefaultStreamFallback:
    async def test_fallback_yields_single_chunk(self):
        provider = FakeProvider()
        chunks = []
        async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].done is True
        assert chunks[0].tool_calls is not None
        assert len(chunks[0].tool_calls) == 1
        assert chunks[0].usage.total_tokens == 15
