"""Tests for the cowork session engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from loom.cowork.session import (
    CoworkSession,
    CoworkTurn,
    ToolCallEvent,
    build_cowork_system_prompt,
)
from loom.models.base import ModelProvider, ModelResponse, TokenUsage, ToolCall
from loom.state.conversation_store import ConversationStore
from loom.state.memory import Database
from loom.tools import create_default_registry
from loom.tools.registry import Tool, ToolResult

# --- Fixtures ---


class MockProvider(ModelProvider):
    """A mock model provider for testing."""

    def __init__(self, responses: list[ModelResponse | Exception]):
        self._responses = list(responses)
        self._call_count = 0
        self.tool_payloads: list[list[dict] | None] = []

    async def complete(self, messages, tools=None, **kwargs):
        self.tool_payloads.append(tools)
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, Exception):
                raise resp
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

    async def test_small_talk_uses_compact_tool_schemas(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(text="Hello!", usage=TokenUsage(total_tokens=3)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        async for _ in session.send("hi"):
            pass

        assert provider.tool_payloads
        first_call = provider.tool_payloads[0] or []
        names = {schema.get("name", "") for schema in first_call}
        assert names == {"ask_user", "task_tracker", "conversation_recall", "delegate_task"}
        assert len(first_call) < len(tools.all_schemas())

    async def test_file_request_includes_coding_tools(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(text="Sure.", usage=TokenUsage(total_tokens=5)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        async for _ in session.send("Read src/main.py and explain it."):
            pass

        assert provider.tool_payloads
        first_call = provider.tool_payloads[0] or []
        names = {schema.get("name", "") for schema in first_call}
        assert "read_file" in names
        assert "ripgrep_search" in names
        assert "glob_find" in names

    async def test_hybrid_mode_includes_fallback_tools(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(text="Hello!", usage=TokenUsage(total_tokens=3)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            tool_exposure_mode="hybrid",
        )

        async for _ in session.send("hi"):
            pass

        assert provider.tool_payloads
        first_call = provider.tool_payloads[0] or []
        names = {schema.get("name", "") for schema in first_call}
        assert "list_tools" in names
        assert "run_tool" in names
        assert len(first_call) <= 18

    async def test_full_mode_sends_all_tool_schemas(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(text="Hello!", usage=TokenUsage(total_tokens=3)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            tool_exposure_mode="full",
        )

        async for _ in session.send("hi"):
            pass

        assert provider.tool_payloads
        first_call = provider.tool_payloads[0] or []
        assert len(first_call) == len(tools.all_schemas())

    async def test_hybrid_unknown_tool_error_suggests_list_tools(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tc1",
                    name="definitely_unknown_tool_name",
                    arguments={},
                )],
                usage=TokenUsage(total_tokens=8),
            ),
            ModelResponse(text="done", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            tool_exposure_mode="hybrid",
        )

        events = []
        async for event in session.send("try tool"):
            events.append(event)

        completed = [e for e in events if isinstance(e, ToolCallEvent) and e.result is not None]
        assert completed
        assert completed[0].result is not None
        assert completed[0].result.success is False
        assert "list_tools" in (completed[0].result.error or "")

    async def test_hybrid_mode_retries_with_fallback_hint_on_tool_stall(
        self,
        workspace,
        tools,
    ):
        provider = MockProvider([
            ModelResponse(
                text=(
                    "I can see the MCP Notion tool is available, but I don't have "
                    "a direct tool to call it from my immediate tool set."
                ),
                usage=TokenUsage(total_tokens=12),
            ),
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tc1",
                    name="list_tools",
                    arguments={"query": "notion", "limit": 3},
                )],
                usage=TokenUsage(total_tokens=8),
            ),
            ModelResponse(
                text="I'll proceed with the discovered tool now.",
                usage=TokenUsage(total_tokens=6),
            ),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            tool_exposure_mode="hybrid",
        )

        events = []
        async for event in session.send("Search Notion for chicken recipes"):
            events.append(event)

        completed = [e for e in events if isinstance(e, ToolCallEvent) and e.result is not None]
        assert any(e.name == "list_tools" and e.result and e.result.success for e in completed)
        assert provider._call_count == 3

    async def test_adaptive_mode_includes_mcp_tools_when_alias_is_mentioned(
        self,
        workspace,
        tools,
    ):
        class _DummyMCPTool(Tool):
            __loom_register__ = False

            def __init__(self, tool_name: str) -> None:
                self._tool_name = tool_name

            @property
            def name(self) -> str:
                return self._tool_name

            @property
            def description(self) -> str:
                return "Dummy MCP tool"

            @property
            def parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, args: dict, ctx) -> ToolResult:
                return ToolResult.ok("ok")

        tools.register(_DummyMCPTool("mcp.notion.search"))
        tools.register(_DummyMCPTool("mcp.notion.query_database"))

        provider = MockProvider([
            ModelResponse(text="Sure.", usage=TokenUsage(total_tokens=5)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            tool_exposure_mode="adaptive",
        )

        async for _ in session.send("use notion mcp to find meeting notes"):
            pass

        assert provider.tool_payloads
        first_call = provider.tool_payloads[0] or []
        names = {schema.get("name", "") for schema in first_call}
        assert any(name.startswith("mcp.notion.") for name in names)

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

    async def test_web_fetch_call_injects_ingest_flag_and_scratch_dir(
        self,
        workspace,
        tools,
        tmp_path: Path,
    ):
        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="tc1",
                    name="web_fetch",
                    arguments={"url": "https://example.com/report.pdf"},
                )],
                usage=TokenUsage(total_tokens=10),
            ),
            ModelResponse(
                text="Done.",
                usage=TokenUsage(total_tokens=5),
            ),
        ])
        tools.execute = AsyncMock(return_value=ToolResult.ok("fetched"))
        scratch_dir = tmp_path / "scratch"

        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            scratch_dir=scratch_dir,
            enable_filetype_ingest_router=False,
        )
        async for _event in session.send("Fetch the report"):
            pass

        tools.execute.assert_called_once()
        args = tools.execute.call_args.args
        kwargs = tools.execute.call_args.kwargs
        assert args[0] == "web_fetch"
        assert args[1]["_enable_filetype_ingest_router"] is False
        assert args[1]["_artifact_retention_max_age_days"] == 14
        assert args[1]["_artifact_retention_max_files_per_scope"] == 96
        assert args[1]["_artifact_retention_max_bytes_per_scope"] == 268_435_456
        assert kwargs["workspace"] == workspace
        assert kwargs["scratch_dir"] == scratch_dir

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

    async def test_context_window_includes_recall_index_when_history_omitted(
        self,
        workspace,
        tools,
    ):
        provider = MockProvider([
            ModelResponse(text=f"Response {i}", usage=TokenUsage())
            for i in range(12)
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            system_prompt="Test system prompt.",
            max_context_messages=8,
            max_context_tokens=6000,
        )

        for i in range(12):
            async for _ in session.send(
                f"Message {i} with extra context to force omission in compact mode",
            ):
                pass

        context = session._context_window()
        assert context[0]["role"] == "system"
        assert any(
            msg.get("role") == "system"
            and "conversation_recall" in str(msg.get("content", ""))
            and "Compact archive index" in str(msg.get("content", ""))
            for msg in context[1:]
        )

    async def test_resume_compacts_persisted_tool_payload_for_model_context(
        self,
        tmp_path: Path,
    ):
        database = Database(tmp_path / "cowork.db")
        await database.initialize()
        store = ConversationStore(database)
        sid = await store.create_session(workspace=str(tmp_path), model_name="mock")

        huge_tool_payload = json.dumps({
            "success": True,
            "output": "X" * 12_000,
            "error": "",
            "data": {"raw": "Y" * 10_000, "kind": "demo"},
            "files_changed": [f"src/file_{i}.py" for i in range(20)],
            "content_blocks": [{"type": "text"} for _ in range(4)],
        })
        await store.append_turn(sid, 1, "user", "please resume this")
        await store.append_turn(
            sid,
            2,
            "tool",
            huge_tool_payload,
            tool_call_id="call_1",
            tool_name="web_fetch",
        )

        provider = MockProvider([ModelResponse(text="ok", usage=TokenUsage(total_tokens=4))])
        session = CoworkSession(
            model=provider,
            tools=create_default_registry(),
            workspace=tmp_path,
            system_prompt="test",
            store=store,
        )

        await session.resume(sid)
        tool_messages = [m for m in session.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        compact_content = str(tool_messages[0].get("content", ""))
        assert len(compact_content) < 3500
        assert "output_truncated" in compact_content
        assert "files_changed_count" in compact_content

    async def test_send_retries_transient_model_failure(self, workspace, tools):
        provider = MockProvider([
            RuntimeError("transient model failure"),
            ModelResponse(text="Recovered response.", usage=TokenUsage(total_tokens=7)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send("Retry please"):
            events.append(event)

        turns = [e for e in events if isinstance(e, CoworkTurn)]
        assert len(turns) == 1
        assert turns[0].text == "Recovered response."
        assert provider._call_count == 2

    async def test_send_streaming_retries_transient_model_failure(self, workspace, tools):
        provider = MockProvider([
            RuntimeError("stream setup failed"),
            ModelResponse(text="Recovered stream.", usage=TokenUsage(total_tokens=4)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send_streaming("Retry stream please"):
            events.append(event)

        chunks = [e for e in events if isinstance(e, str)]
        turns = [e for e in events if isinstance(e, CoworkTurn)]
        assert "".join(chunks) == "Recovered stream."
        assert len(turns) == 1
        assert turns[0].text == "Recovered stream."
        assert provider._call_count == 2

    async def test_send_streaming_estimates_tokens_when_usage_missing(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(
                text="Token estimation fallback should prevent zero token turns.",
                usage=TokenUsage(),
            ),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send_streaming("estimate tokens"):
            events.append(event)

        turns = [e for e in events if isinstance(e, CoworkTurn)]
        assert len(turns) == 1
        assert turns[0].tokens_used > 0
        assert session.total_tokens == turns[0].tokens_used
        assert turns[0].latency_ms > 0
        assert turns[0].total_time_ms >= turns[0].latency_ms
        assert turns[0].tokens_per_second > 0
        assert turns[0].context_tokens > 0
        assert turns[0].context_messages > 0

    async def test_send_estimates_tokens_when_usage_missing(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(
                text="Token estimation fallback should also work for non-streaming sends.",
                usage=TokenUsage(),
            ),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send("estimate tokens"):
            events.append(event)

        turns = [e for e in events if isinstance(e, CoworkTurn)]
        assert len(turns) == 1
        assert turns[0].tokens_used > 0
        assert session.total_tokens == turns[0].tokens_used
        assert turns[0].latency_ms > 0
        assert turns[0].total_time_ms >= turns[0].latency_ms
        assert turns[0].tokens_per_second > 0
        assert turns[0].context_tokens > 0
        assert turns[0].context_messages > 0


class TestBuildSystemPrompt:
    def test_with_workspace(self, workspace):
        prompt = build_cowork_system_prompt(workspace)
        assert str(workspace) in prompt
        assert "collaborative assistant for complex tasks" in prompt.lower()

    def test_without_workspace(self):
        prompt = build_cowork_system_prompt(None)
        assert "No workspace set" in prompt

    def test_includes_tool_guidance(self, workspace):
        prompt = build_cowork_system_prompt(workspace)
        assert "glob_find" in prompt
        assert "ripgrep_search" in prompt
        assert "ask_user" in prompt
