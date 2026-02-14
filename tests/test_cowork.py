"""Tests for the cowork session engine."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.cowork.session import (
    CoworkSession,
    CoworkTurn,
    ToolCallEvent,
    build_cowork_system_prompt,
)
from loom.models.base import ModelProvider, ModelResponse, TokenUsage, ToolCall
from loom.tools import create_default_registry

# --- Fixtures ---


class MockProvider(ModelProvider):
    """A mock model provider for testing."""

    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, messages, tools=None, **kwargs):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return ModelResponse(text="No more responses.", usage=TokenUsage())

    async def health_check(self):
        return True

    @property
    def name(self):
        return "mock-model"

    @property
    def tier(self):
        return 1

    @property
    def roles(self):
        return ["executor"]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    print('hello')\n")
    return tmp_path


@pytest.fixture
def tools():
    return create_default_registry()


# --- Tests ---


class TestCoworkSession:
    async def test_simple_text_response(self, workspace, tools):
        """Model returns text only — one turn, no tool calls."""
        provider = MockProvider([
            ModelResponse(text="Hello! How can I help?", usage=TokenUsage(total_tokens=10)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send("Hi"):
            events.append(event)

        # Should get a CoworkTurn with the text
        turns = [e for e in events if isinstance(e, CoworkTurn)]
        assert len(turns) == 1
        assert turns[0].text == "Hello! How can I help?"
        assert turns[0].tokens_used == 10

    async def test_tool_call_then_response(self, workspace, tools):
        """Model calls a tool, then responds with text."""
        provider = MockProvider([
            # First response: tool call
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tc1", name="read_file",
                    arguments={"path": "src/main.py"},
                )],
                usage=TokenUsage(total_tokens=20),
            ),
            # Second response: text after seeing tool result
            ModelResponse(
                text="The file contains a hello function.",
                usage=TokenUsage(total_tokens=15),
            ),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        tool_result_snapshots = []
        async for event in session.send("What's in main.py?"):
            events.append(event)
            if isinstance(event, ToolCallEvent):
                # Snapshot the result state at yield time
                tool_result_snapshots.append(event.result)

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        # Each tool call emits twice: start (no result) and complete (with result)
        assert len(tool_events) == 2
        assert tool_events[0].name == "read_file"
        # First yield: no result yet; second yield: has result
        assert tool_result_snapshots[0] is None
        assert tool_result_snapshots[1] is not None
        assert tool_result_snapshots[1].success is True

        turns = [e for e in events if isinstance(e, CoworkTurn)]
        assert len(turns) == 1
        assert "hello function" in turns[0].text
        assert turns[0].tokens_used == 35

    async def test_conversation_history_maintained(self, workspace, tools):
        """Conversation history grows with each exchange."""
        provider = MockProvider([
            ModelResponse(text="First response.", usage=TokenUsage()),
            ModelResponse(text="Second response.", usage=TokenUsage()),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        async for _ in session.send("First message"):
            pass

        # Should have: user + assistant = 2 messages (no system prompt)
        assert len(session.messages) == 2
        assert session.messages[-1]["role"] == "assistant"
        assert session.messages[-1]["content"] == "First response."

        async for _ in session.send("Second message"):
            pass

        # Should now have: user + assistant + user + assistant = 4
        assert len(session.messages) == 4
        assert session.messages[-1]["content"] == "Second response."

    async def test_ask_user_tool_pauses(self, workspace, tools):
        """When ask_user is called, the loop should pause."""
        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tc1", name="ask_user",
                    arguments={"question": "Which language?", "options": ["Python", "Rust"]},
                )],
                usage=TokenUsage(total_tokens=10),
            ),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send("Create a project"):
            events.append(event)

        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert any(e.name == "ask_user" for e in tool_events)

    async def test_context_window_trimming(self, workspace, tools):
        """Session trims context when token budget is exceeded."""
        provider = MockProvider([
            ModelResponse(text=f"Response {i}", usage=TokenUsage())
            for i in range(10)
        ])
        session = CoworkSession(
            model=provider, tools=tools, workspace=workspace,
            system_prompt="Test system prompt.",
            max_context_messages=5,
            # Very small token budget so short messages get trimmed
            max_context_tokens=200,
        )

        for i in range(10):
            async for _ in session.send(f"Message {i}"):
                pass

        # Context window should be trimmed by token budget
        context = session._context_window()
        assert len(context) < 20  # less than all 20 messages
        # System prompt should always be preserved
        assert context[0]["role"] == "system"


class TestBuildSystemPrompt:
    def test_with_workspace(self, workspace):
        prompt = build_cowork_system_prompt(workspace)
        assert str(workspace) in prompt
        assert "collaborative coding assistant" in prompt.lower()

    def test_without_workspace(self):
        prompt = build_cowork_system_prompt(None)
        assert "No workspace set" in prompt

    def test_includes_tool_guidance(self, workspace):
        prompt = build_cowork_system_prompt(workspace)
        assert "glob_find" in prompt
        assert "ripgrep_search" in prompt
        assert "ask_user" in prompt
