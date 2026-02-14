"""Tests for the Anthropic model provider."""

from __future__ import annotations

import json

import pytest

from loom.models.anthropic_provider import AnthropicProvider


@pytest.fixture
def provider():
    return AnthropicProvider(
        name="test-claude",
        model="claude-sonnet-4-5-20250929",
        api_key="test-key",
        max_tokens=1024,
        tier=2,
        roles=["executor", "planner"],
    )


class TestAnthropicProviderProperties:
    def test_name(self, provider):
        assert provider.name == "test-claude"

    def test_model(self, provider):
        assert provider.model == "claude-sonnet-4-5-20250929"

    def test_tier(self, provider):
        assert provider.tier == 2

    def test_roles(self, provider):
        assert provider.roles == ["executor", "planner"]


class TestMessageConversion:
    def test_system_message_extraction(self, provider):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, msgs = provider._convert_messages(messages)
        assert system == "You are helpful."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_user_message(self, provider):
        messages = [{"role": "user", "content": "Hello"}]
        system, msgs = provider._convert_messages(messages)
        assert system is None
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello"

    def test_assistant_with_tool_calls(self, provider):
        messages = [
            {"role": "user", "content": "Read file"},
            {
                "role": "assistant",
                "content": "Let me read that.",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "test.py"}),
                        },
                    }
                ],
            },
        ]
        system, msgs = provider._convert_messages(messages)
        assert len(msgs) == 2
        # Assistant message should have text + tool_use blocks
        content = msgs[1]["content"]
        assert any(b.get("type") == "text" for b in content)
        assert any(b.get("type") == "tool_use" for b in content)

    def test_tool_result_message(self, provider):
        messages = [
            {"role": "user", "content": "Read file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "test.py"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "content": '{"success": true, "output": "file content"}',
            },
        ]
        system, msgs = provider._convert_messages(messages)
        # tool result should become a user message with tool_result block
        assert any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
            for m in msgs
            if m["role"] == "user"
        )

    def test_multiple_tool_results_batched(self, provider):
        """Multiple tool results in sequence should be batched into one user message."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "tc1", "type": "function",
                     "function": {"name": "t1", "arguments": "{}"}},
                    {"id": "tc2", "type": "function",
                     "function": {"name": "t2", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
            {"role": "tool", "tool_call_id": "tc2", "content": "result2"},
        ]
        system, msgs = provider._convert_messages(messages)
        # The two tool results should be in one user message
        tool_result_msgs = [
            m for m in msgs
            if m["role"] == "user"
            and isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_msgs) == 1
        assert len(tool_result_msgs[0]["content"]) == 2


class TestToolConversion:
    def test_loom_format_tools(self, provider):
        """Convert Loom tool schemas to Anthropic format."""
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]
        result = provider._convert_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "read_file"
        assert "input_schema" in result[0]

    def test_none_tools(self, provider):
        assert provider._convert_tools(None) is None

    def test_empty_tools(self, provider):
        assert provider._convert_tools([]) is None


class TestResponseParsing:
    def test_text_response(self, provider):
        data = {
            "content": [{"type": "text", "text": "Hello there!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = provider._parse_response(data)
        assert response.text == "Hello there!"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_tool_use_response(self, provider):
        data = {
            "content": [
                {"type": "text", "text": "Let me read that file."},
                {
                    "type": "tool_use",
                    "id": "tu_123",
                    "name": "read_file",
                    "input": {"path": "main.py"},
                },
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        response = provider._parse_response(data)
        assert "read that file" in response.text
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "read_file"
        assert response.tool_calls[0].arguments == {"path": "main.py"}

    def test_empty_response(self, provider):
        data = {"content": [], "usage": {"input_tokens": 5, "output_tokens": 0}}
        response = provider._parse_response(data)
        assert response.text == ""
        assert response.tool_calls == []
