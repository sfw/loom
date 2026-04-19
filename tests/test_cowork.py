"""Tests for the cowork session engine."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from loom.cowork.session import (
    CoworkSession,
    CoworkStopRequestedError,
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
    def test_stop_request_lifecycle(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(text="ok", usage=TokenUsage(total_tokens=1)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        assert session.stop_requested is False
        session.request_stop("user_requested")
        assert session.stop_requested is True
        session.clear_stop_request()
        assert session.stop_requested is False

    def test_pause_request_lifecycle(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(text="ok", usage=TokenUsage(total_tokens=1)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        assert session.pause_requested is False
        session.request_pause()
        assert session.pause_requested is True
        session.request_resume()
        assert session.pause_requested is False

    async def test_send_raises_when_stop_requested_mid_turn(self, workspace, tools):
        gate = asyncio.Event()

        class _BlockingProvider(MockProvider):
            async def complete(self, messages, tools=None, **kwargs):
                await gate.wait()
                return ModelResponse(text="Should not complete", usage=TokenUsage(total_tokens=2))

        session = CoworkSession(
            model=_BlockingProvider([]),
            tools=tools,
            workspace=workspace,
        )

        async def _consume() -> None:
            async for _event in session.send("Please stop this turn"):
                pass

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0)
        session.request_stop("user_requested")
        gate.set()

        with pytest.raises(CoworkStopRequestedError):
            await task

    async def test_send_waits_while_paused_then_resumes(self, workspace, tools):
        provider = MockProvider([
            ModelResponse(text="resumed", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)
        session.request_pause()
        events: list[object] = []

        async def _consume() -> None:
            async for event in session.send("wait for resume"):
                events.append(event)

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0.05)
        assert task.done() is False

        session.request_resume()
        await asyncio.wait_for(task, timeout=1.0)

        turns = [event for event in events if isinstance(event, CoworkTurn)]
        assert len(turns) == 1
        assert turns[0].text == "resumed"

    async def test_send_applies_pending_inject_before_model_request(self, workspace, tools):
        captured_messages: list[list[dict]] = []

        class _CaptureProvider(MockProvider):
            async def complete(self, messages, tools=None, **kwargs):
                captured_messages.append([dict(item) for item in messages])
                return ModelResponse(
                    text="done",
                    usage=TokenUsage(total_tokens=2),
                )

        session = CoworkSession(
            model=_CaptureProvider([]),
            tools=tools,
            workspace=workspace,
        )
        session.queue_inject_instruction("Prioritize test coverage.")

        async for _event in session.send("continue"):
            pass

        assert session.has_pending_inject_instruction is False
        assert captured_messages
        first_context = captured_messages[0]
        assert any(
            str(item.get("role")) == "system"
            and "Steering instruction from user: Prioritize test coverage."
            in str(item.get("content", ""))
            for item in first_context
        )

    async def test_send_applies_stacked_inject_instructions_fifo(self, workspace, tools):
        captured_messages: list[list[dict]] = []

        class _CaptureProvider(MockProvider):
            async def complete(self, messages, tools=None, **kwargs):
                captured_messages.append([dict(item) for item in messages])
                return ModelResponse(
                    text="done",
                    usage=TokenUsage(total_tokens=2),
                )

        session = CoworkSession(
            model=_CaptureProvider([]),
            tools=tools,
            workspace=workspace,
        )
        session.queue_inject_instruction("First directive.")
        session.queue_inject_instruction("Second directive.")

        assert session.pending_inject_instruction_count == 2
        async for _event in session.send("continue"):
            pass

        assert session.pending_inject_instruction_count == 1
        assert captured_messages
        first_context = captured_messages[0]
        assert any(
            str(item.get("role")) == "system"
            and "Steering instruction from user: First directive."
            in str(item.get("content", ""))
            for item in first_context
        )

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
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            tool_exposure_mode="adaptive",
        )

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
        assert len(first_call) == len(
            tools.all_schemas(
                execution_surface="tui",
                runnable_only=True,
            ),
        )

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

    async def test_delegate_progress_callback_adds_tool_call_metadata(self, workspace, tools):
        progress: list[dict] = []

        def _capture(payload: dict) -> None:
            progress.append(dict(payload))

        session = CoworkSession(
            model=MockProvider([ModelResponse(text="ok", usage=TokenUsage(total_tokens=1))]),
            tools=tools,
            workspace=workspace,
            delegate_progress_callback=_capture,
        )
        existing_progress: list[dict] = []
        execute_args = session._prepare_tool_execute_arguments(
            "delegate_task",
            {"goal": "Analyze", "_progress_callback": existing_progress.append},
            tool_call_id="call_123",
            caller_tool_name="run_tool",
            include_delegate_callback=True,
        )
        wrapped = execute_args.get("_progress_callback")
        assert callable(wrapped)
        wrapped({"event_type": "subtask_started", "event_data": {"subtask_id": "scope"}})

        assert progress
        assert progress[0]["tool_call_id"] == "call_123"
        assert progress[0]["caller_tool_name"] == "run_tool"
        assert progress[0]["tool_name"] == "delegate_task"
        assert existing_progress
        assert existing_progress[0]["tool_call_id"] == "call_123"

    async def test_run_tool_dispatch_strips_internal_parent_metadata(self, workspace, tools):
        session = CoworkSession(
            model=MockProvider([ModelResponse(text="ok", usage=TokenUsage(total_tokens=1))]),
            tools=tools,
            workspace=workspace,
            delegate_progress_callback=lambda _payload: None,
        )
        session._tools.has = lambda *_args, **_kwargs: True
        execute_mock = AsyncMock(return_value=ToolResult.ok("ok"))
        session._tools.execute = execute_mock

        result = await session._dispatch_run_tool(
            "delegate_task",
            {
                "goal": "Analyze",
                "_loom_parent_tool_call_id": "call_parent",
                "_loom_parent_tool_name": "run_tool",
            },
            ctx=SimpleNamespace(auth_context=None),
        )

        assert result.success is True
        assert execute_mock.await_count == 1
        executed_args = execute_mock.await_args.args[1]
        assert "_loom_parent_tool_call_id" not in executed_args
        assert "_loom_parent_tool_name" not in executed_args
        assert callable(executed_args.get("_progress_callback"))

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

    async def test_context_window_prefers_marker_sections_when_memory_snapshot_available(
        self,
        workspace,
        tools,
    ):
        provider = MockProvider([
            ModelResponse(text="ok", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            system_prompt="test",
            max_context_messages=6,
            max_context_tokens=5000,
        )
        session.session_state.update_memory_snapshot({
            "active_decisions": [
                {
                    "id": 108,
                    "entry_type": "decision",
                    "status": "active",
                    "summary": "Use compactor model for cowork memory extraction",
                    "source_turn_start": 11,
                    "source_turn_end": 12,
                }
            ],
            "open_questions": [
                {
                    "entry_type": "open_question",
                    "status": "active",
                    "summary": "Should we force fts mode for recall queries?",
                    "source_turn_start": 13,
                    "source_turn_end": 13,
                }
            ],
        })
        session.session_state.update_memory_index_meta(
            last_indexed_turn=20,
            degraded=False,
            failure_count=0,
        )

        session._messages = [{"role": "system", "content": "test"}]
        for idx in range(14):
            session._messages.append(
                {"role": "user", "content": f"Message {idx} with archive detail"},
            )

        context = session._context_window()
        recall_messages = [
            msg
            for msg in context
            if msg.get("role") == "system"
            and "Compact archive index" in str(msg.get("content", ""))
        ]
        assert recall_messages
        recall_content = str(recall_messages[0].get("content", ""))
        assert "Active DECISION" in recall_content
        assert "Open QUESTION" in recall_content
        assert "id=108" in recall_content
        assert "decision_context" in recall_content

    async def test_context_window_preserves_latest_tool_exchange_when_history_is_trimmed(
        self,
        workspace,
        tools,
    ):
        provider = MockProvider([
            ModelResponse(text="ok", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            system_prompt="test",
            max_context_messages=8,
            max_context_tokens=5200,
        )

        session._messages = [{"role": "system", "content": "test"}]
        for idx in range(10):
            session._messages.append({
                "role": "user",
                "content": f"Older context {idx}: " + ("archive " * 60),
            })
            session._messages.append({
                "role": "assistant",
                "content": f"Older reply {idx}: " + ("detail " * 40),
            })

        session._messages.extend([
            {
                "role": "assistant",
                "content": "Checking the latest workspace state.",
                "tool_calls": [
                    {
                        "id": "tc-latest",
                        "type": "function",
                        "function": {"name": "ripgrep_search", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc-latest",
                "content": json.dumps({
                    "success": True,
                    "output": "latest tool output",
                    "error": "",
                    "files_changed": [],
                }),
            },
            {"role": "assistant", "content": "LATEST CRITICAL: preserve exact matching paths."},
            {"role": "user", "content": "LATEST USER: keep the newest search result in context."},
        ])

        context = session._context_window()

        assert any(
            msg.get("role") == "assistant"
            and any(
                call.get("id") == "tc-latest"
                for call in list(msg.get("tool_calls", []))
                if isinstance(call, dict)
            )
            for msg in context
            if isinstance(msg, dict)
        )
        assert any(
            msg.get("role") == "tool" and msg.get("tool_call_id") == "tc-latest"
            for msg in context
            if isinstance(msg, dict)
        )
        assert session.session_state.has_compact_memory is True
        assert "Older user topics" in session.session_state.compact_summary

    async def test_context_window_reuses_persisted_compact_memory_on_recent_tail_only(
        self,
        workspace,
        tools,
    ):
        provider = MockProvider([
            ModelResponse(text="ok", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            system_prompt="test",
            max_context_messages=8,
            max_context_tokens=9000,
        )
        session.session_state.update_compact_memory(
            summary=(
                "- Older user topics: auth edge cases\n"
                "- Older tool activity: ripgrep_search, read_file"
            ),
            boundary_message_count=40,
            message_count=40,
            tool_message_count=11,
        )
        session._messages = [
            {"role": "system", "content": "test"},
            {"role": "assistant", "content": "Recent assistant note."},
            {"role": "user", "content": "Recent user follow-up."},
        ]

        context = session._context_window()

        assert any(
            msg.get("role") == "system"
            and "Compact archive index" in str(msg.get("content", ""))
            and "Older tool activity" in str(msg.get("content", ""))
            for msg in context[1:]
        )

    async def test_context_window_repairs_dangling_assistant_tool_calls(
        self,
        workspace,
        tools,
    ):
        provider = MockProvider([
            ModelResponse(text="ok", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            system_prompt="test",
        )

        session._messages = [
            {"role": "system", "content": "test"},
            {
                "role": "assistant",
                "content": "Searching once",
                "tool_calls": [
                    {
                        "id": "tc-ok",
                        "type": "function",
                        "function": {"name": "ripgrep_search", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc-ok",
                "content": "{\"success\":true,\"output\":\"ok\"}",
            },
            {
                "role": "assistant",
                "content": "Dangling tool call",
                "tool_calls": [
                    {
                        "id": "tc-missing",
                        "type": "function",
                        "function": {"name": "ripgrep_search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "user", "content": "continue"},
        ]

        window = session._context_window()
        dangling = [
            msg
            for msg in window
            if msg.get("role") == "assistant"
            and str(msg.get("content", "")) == "Dangling tool call"
        ]
        assert dangling
        assert "tool_calls" not in dangling[0]

    async def test_context_window_drops_orphan_tool_messages(
        self,
        workspace,
        tools,
    ):
        provider = MockProvider([
            ModelResponse(text="ok", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            system_prompt="test",
        )

        session._messages = [
            {"role": "system", "content": "test"},
            {"role": "tool", "tool_call_id": "orphan", "content": "{\"success\":true}"},
            {"role": "user", "content": "hello"},
        ]

        window = session._context_window()
        assert all(str(msg.get("role", "")) != "tool" for msg in window)

    async def test_context_window_skips_oversized_latest_message_instead_of_collapsing(
        self,
        workspace,
        tools,
    ):
        provider = MockProvider([
            ModelResponse(text="ok", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(
            model=provider,
            tools=tools,
            workspace=workspace,
            system_prompt="test",
            max_context_messages=12,
            max_context_tokens=4600,
        )

        # With max_context_tokens=4600 and a fixed output reserve of 4000,
        # context budget is intentionally tight (~hundreds of tokens).
        huge_tool_payload = json.dumps({
            "success": True,
            "output": "X" * 12000,
            "error": "",
            "files_changed": [],
        })
        session._messages = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "is that in all subdirectories too?"},
            {
                "role": "assistant",
                "content": "Let me verify with ripgrep.",
                "tool_calls": [
                    {
                        "id": "tc-big",
                        "type": "function",
                        "function": {"name": "ripgrep_search", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tc-big",
                "content": huge_tool_payload,
            },
        ]

        window = session._context_window()
        assert any(
            msg.get("role") == "user"
            and "all subdirectories" in str(msg.get("content", ""))
            for msg in window
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

    async def test_memory_indexer_captures_marker_entries_from_turns(self, tmp_path: Path):
        database = Database(tmp_path / "cowork-memory.db")
        await database.initialize()
        store = ConversationStore(database)
        sid = await store.create_session(workspace=str(tmp_path), model_name="mock")

        provider = MockProvider([
            ModelResponse(
                text="PROPOSAL: Add typed conversation_recall actions for decision context.",
                usage=TokenUsage(total_tokens=6),
            ),
        ])
        session = CoworkSession(
            model=provider,
            tools=create_default_registry(),
            workspace=tmp_path,
            system_prompt="test",
            store=store,
            session_id=sid,
            memory_index_enabled=True,
            memory_index_llm_extraction_enabled=False,
        )

        async for _event in session.send(
            "DECISION: Use compactor role for cowork memory extraction.",
        ):
            pass

        assert await session.wait_for_memory_index_idle(timeout_seconds=2.0)
        assert session.session_state.active_decisions
        assert session.session_state.active_proposals
        indexed = await store.search_cowork_memory_entries(
            sid,
            entry_type="decision",
            status="active",
            limit=10,
        )
        assert indexed

    async def test_resume_replays_canonical_turns_past_stale_session_checkpoint(
        self,
        tmp_path: Path,
    ):
        database = Database(tmp_path / "stale-checkpoint.db")
        await database.initialize()
        store = ConversationStore(database)
        sid = await store.create_session(workspace=str(tmp_path), model_name="mock")

        await store.append_turn(sid, 1, "user", "Old focus")
        await store.update_session(
            sid,
            total_tokens=5,
            turn_count=1,
            session_state={
                "session_id": sid,
                "workspace": str(tmp_path),
                "model_name": "mock",
                "turn_count": 1,
                "total_tokens": 5,
                "current_focus": "Old focus",
            },
            session_state_through_turn=1,
        )
        await store.append_turn(sid, 2, "assistant", "Old answer")
        await store.append_turn(sid, 3, "user", "Fresh followup")

        session = CoworkSession(
            model=MockProvider([ModelResponse(text="ok", usage=TokenUsage(total_tokens=1))]),
            tools=create_default_registry(),
            workspace=tmp_path,
            system_prompt="test",
            store=store,
        )

        await session.resume(sid)

        assert session.persisted_turn_count == 3
        assert session.session_state.turn_count == 2
        assert session.session_state.current_focus == "Fresh followup"
        assert session.messages[-1]["content"] == "Fresh followup"

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

    async def test_send_streaming_stops_repeated_identical_tool_batches(
        self,
        workspace,
        tools,
    ):
        repeated_tool_response = ModelResponse(
            text="",
            tool_calls=[ToolCall(
                id="tc-repeat",
                name="ripgrep_search",
                arguments={"pattern": "spark"},
            )],
            usage=TokenUsage(total_tokens=8),
        )
        provider = MockProvider([
            repeated_tool_response,
            repeated_tool_response,
            repeated_tool_response,
            repeated_tool_response,
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send_streaming("can you check again?"):
            events.append(event)

        turns = [e for e in events if isinstance(e, CoworkTurn)]
        assert len(turns) == 1
        assert "repeated identical tool calls" in turns[0].text.lower()

        # Two executions, then recovery hint, then deterministic stop.
        assert len(turns[0].tool_calls) == 2

    async def test_send_streaming_recovers_when_tool_turn_ends_without_final_text(
        self,
        workspace,
        tools,
    ):
        class _EchoTool(Tool):
            @property
            def name(self) -> str:
                return "echo_tool"

            @property
            def description(self) -> str:
                return "Return deterministic test output."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                }

            async def execute(self, args: dict, ctx) -> ToolResult:
                return ToolResult.ok(f"tool result: {args.get('value', '')}")

        tools.register(_EchoTool())
        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="echo-1",
                    name="echo_tool",
                    arguments={"value": "hello"},
                )],
                usage=TokenUsage(total_tokens=3),
            ),
            ModelResponse(text="", usage=TokenUsage(total_tokens=1)),
            ModelResponse(text="Final recommendation.", usage=TokenUsage(total_tokens=2)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send_streaming("use the tool then answer"):
            events.append(event)

        turns = [event for event in events if isinstance(event, CoworkTurn)]
        assert len(turns) == 1
        assert turns[0].text == "Final recommendation."
        assert provider._call_count == 3

    async def test_send_recovers_with_fallback_when_tool_turn_still_has_no_final_text(
        self,
        workspace,
        tools,
    ):
        class _EchoTool(Tool):
            @property
            def name(self) -> str:
                return "echo_tool"

            @property
            def description(self) -> str:
                return "Return deterministic test output."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                }

            async def execute(self, args: dict, ctx) -> ToolResult:
                return ToolResult.ok(f"tool result: {args.get('value', '')}")

        tools.register(_EchoTool())
        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="echo-1",
                    name="echo_tool",
                    arguments={"value": "world"},
                )],
                usage=TokenUsage(total_tokens=3),
            ),
            ModelResponse(text="", usage=TokenUsage(total_tokens=1)),
            ModelResponse(text="", usage=TokenUsage(total_tokens=1)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        async for event in session.send("use the tool then answer"):
            events.append(event)

        turns = [event for event in events if isinstance(event, CoworkTurn)]
        assert len(turns) == 1
        assert "failed to produce a final answer" in turns[0].text.lower()
        assert "echo_tool" in turns[0].text
        assert "tool result: world" in turns[0].text

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

    async def test_append_tool_result_uses_fast_preview_without_semantic_compactor(
        self,
        workspace,
        tools,
    ):
        session = CoworkSession(
            model=MockProvider([ModelResponse(text="ok", usage=TokenUsage(total_tokens=1))]),
            tools=tools,
            workspace=workspace,
        )

        class _ExplodingCompactor:
            async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
                raise AssertionError("semantic compactor should not run for tool results")

        session._compactor = _ExplodingCompactor()
        long_text = "important evidence " * 400

        await session._append_tool_result(
            "call-fast-preview",
            "web_fetch",
            ToolResult.ok(long_text, data={"body": long_text}),
        )

        assert session._messages
        payload = json.loads(str(session._messages[-1]["content"]))
        assert str(payload["output"]).endswith("...[truncated]")
        assert str(payload["data"]["body"]).endswith("...[truncated]")

    async def test_send_parallelizes_safe_web_tool_batches(self, workspace, tools):
        tools.exclude("web_fetch")
        state = {"active": 0, "max_active": 0}

        class _SlowWebFetchTool(Tool):
            @property
            def name(self) -> str:
                return "web_fetch"

            @property
            def description(self) -> str:
                return "Test web fetch tool."

            @property
            def parameters(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                    },
                    "required": ["url"],
                }

            async def execute(self, args: dict, ctx) -> ToolResult:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                try:
                    await asyncio.sleep(0.05)
                finally:
                    state["active"] -= 1
                return ToolResult.ok(
                    f"Fetched {args.get('url', '')}",
                    data={"url": args.get("url", "")},
                )

        tools.register(_SlowWebFetchTool())
        provider = MockProvider([
            ModelResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="wf-1",
                        name="web_fetch",
                        arguments={"url": "https://example.com/1"},
                    ),
                    ToolCall(
                        id="wf-2",
                        name="web_fetch",
                        arguments={"url": "https://example.com/2"},
                    ),
                ],
                usage=TokenUsage(total_tokens=6),
            ),
            ModelResponse(text="done", usage=TokenUsage(total_tokens=1)),
        ])
        session = CoworkSession(model=provider, tools=tools, workspace=workspace)

        events = []
        started_ids: list[str] = []
        async for event in session.send("fetch both"):
            events.append(event)
            if isinstance(event, ToolCallEvent) and event.result is None:
                started_ids.append(event.tool_call_id)

        completed = [
            event for event in events
            if isinstance(event, ToolCallEvent) and event.result is not None
        ]
        assert started_ids == ["wf-1", "wf-2"]
        assert sorted({event.tool_call_id for event in completed}) == ["wf-1", "wf-2"]
        assert state["max_active"] >= 2


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
        assert "verification_helper" in prompt
        assert "browser_session" in prompt
        assert "shell_execute" in prompt
