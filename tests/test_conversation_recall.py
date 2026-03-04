"""Tests for the conversation_recall tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.state.conversation_store import ConversationStore
from loom.state.memory import Database
from loom.tools.conversation_recall import ConversationRecallTool
from loom.tools.registry import ToolContext


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    await database.initialize()
    return database


@pytest.fixture
async def store(db: Database) -> ConversationStore:
    return ConversationStore(db)


@pytest.fixture
async def populated_store(store: ConversationStore) -> tuple[ConversationStore, str]:
    """A store with a session and some turns."""
    sid = await store.create_session(workspace="/tmp", model_name="m")

    await store.append_turn(sid, 1, "user", "Let's add JWT authentication")
    await store.append_turn(sid, 2, "assistant", "I'll implement JWT support using PyJWT")
    await store.append_turn(
        sid, 3, "tool", "file content here",
        tool_name="read_file", tool_call_id="c1",
    )
    await store.append_turn(sid, 4, "assistant", "I see the current auth module uses basic tokens")
    await store.append_turn(sid, 5, "user", "Use RS256 algorithm")
    await store.append_turn(sid, 6, "assistant", "Updating to RS256 algorithm")
    await store.append_turn(sid, 7, "tool", "success", tool_name="edit_file", tool_call_id="c2")
    await store.append_turn(sid, 8, "assistant", "Done! The auth module now uses RS256")
    await store.append_turn(sid, 9, "user", "Now fix the database migration")
    await store.append_turn(
        sid, 10, "tool", "migration output",
        tool_name="shell_execute", tool_call_id="c3",
    )

    await store.upsert_cowork_memory_entries(
        sid,
        [
            {
                "entry_type": "decision",
                "status": "active",
                "summary": "Use RS256 for JWT signing",
                "rationale": "Meets security requirement from user",
                "topic": "jwt-auth",
                "source_turn_start": 5,
                "source_turn_end": 8,
                "source_roles": ["user", "assistant"],
                "evidence_excerpt": "Use RS256 algorithm",
                "confidence": 0.94,
                "fingerprint": "recall-fp-1",
            },
            {
                "entry_type": "open_question",
                "status": "active",
                "summary": "How should migration be validated post-change?",
                "topic": "database-migration",
                "source_turn_start": 9,
                "source_turn_end": 10,
                "source_roles": ["user", "tool"],
                "confidence": 0.61,
                "fingerprint": "recall-fp-2",
            },
        ],
    )

    return store, sid


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


class TestConversationRecallTool:
    async def test_search(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "search", "query": "JWT"}, ctx)
        assert result.success
        assert "JWT" in result.output

    async def test_search_no_results(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "search", "query": "nonexistent_term"}, ctx)
        assert result.success
        assert "No messages found" in result.output

    async def test_search_requires_query(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "search"}, ctx)
        assert not result.success

    async def test_range(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "range", "start_turn": 3, "end_turn": 6}, ctx)
        assert result.success
        assert "Turn 3" in result.output
        assert "Turn 6" in result.output

    async def test_range_no_results(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "range", "start_turn": 99, "end_turn": 100}, ctx)
        assert result.success
        assert "No turns found" in result.output

    async def test_range_invalid(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "range", "start_turn": 10, "end_turn": 5}, ctx)
        assert not result.success

    async def test_tool_calls(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "tool_calls", "tool_name": "read_file"}, ctx)
        assert result.success
        assert "read_file" in result.output

    async def test_tool_calls_no_results(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "tool_calls", "tool_name": "web_search"}, ctx)
        assert result.success
        assert "No calls to" in result.output

    async def test_tool_calls_requires_name(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "tool_calls"}, ctx)
        assert not result.success

    async def test_summary_with_state(self, populated_store, ctx):
        from loom.cowork.session_state import SessionState
        store, sid = populated_store

        state = SessionState(session_id=sid, workspace="/tmp", model_name="m", turn_count=10)
        state.record_decision("Use JWT", 1)

        tool = ConversationRecallTool(store=store, session_id=sid, session_state=state)
        result = await tool.execute({"action": "summary"}, ctx)
        assert result.success
        assert "JWT" in result.output

    async def test_summary_without_state(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "summary"}, ctx)
        assert result.success

    async def test_unbound_tool(self, ctx):
        tool = ConversationRecallTool()
        result = await tool.execute({"action": "search", "query": "test"}, ctx)
        assert not result.success
        assert "not available" in result.error

    async def test_bind(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool()
        tool.bind(store=store, session_id=sid)

        result = await tool.execute({"action": "search", "query": "JWT"}, ctx)
        assert result.success

    async def test_unknown_action(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "invalid"}, ctx)
        assert not result.success

    async def test_limit_parameter(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "search", "query": "the", "limit": 2}, ctx)
        assert result.success

    async def test_entries_action(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute(
            {"action": "entries", "entry_type": "decision", "status": "active"},
            ctx,
        )
        assert result.success
        assert "DECISION" in result.output
        assert "source=index" in result.output

    async def test_decision_context_action(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute(
            {"action": "decision_context", "topic": "jwt"},
            ctx,
        )
        assert result.success
        assert "RS256" in result.output or "DECISION" in result.output

    async def test_timeline_action(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute(
            {"action": "timeline", "topic": "jwt-auth"},
            ctx,
        )
        assert result.success
        assert "turn" in result.output

    async def test_open_questions_action(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(store=store, session_id=sid)

        result = await tool.execute({"action": "open_questions"}, ctx)
        assert result.success
        assert "OPEN_QUESTION" in result.output or "open questions" in result.output.lower()

    async def test_source_turns_by_entry_ids_action(self, populated_store, ctx):
        store, sid = populated_store
        entries = await store.search_cowork_memory_entries(
            sid,
            entry_type="decision",
            status="active",
            limit=5,
        )
        assert entries
        entry_id = entries[0]["id"]
        tool = ConversationRecallTool(store=store, session_id=sid)
        result = await tool.execute(
            {"action": "source_turns", "entry_ids": [entry_id]},
            ctx,
        )
        assert result.success
        assert "Turn" in result.output

    async def test_v2_actions_can_be_disabled(self, populated_store, ctx):
        store, sid = populated_store
        tool = ConversationRecallTool(
            store=store,
            session_id=sid,
            v2_actions_enabled=False,
        )
        result = await tool.execute({"action": "entries"}, ctx)
        assert result.success is False
        assert "disabled" in (result.error or "")


class TestRecallHint:
    """Test dangling reference detection in CoworkSession."""

    def test_hint_triggered(self):
        from loom.cowork.session import CoworkSession
        from loom.models.base import ModelResponse, TokenUsage
        from loom.tools import create_default_registry
        from tests.test_cowork import MockProvider

        provider = MockProvider([ModelResponse(text="ok", usage=TokenUsage())])
        session = CoworkSession(
            model=provider,
            tools=create_default_registry(),
            system_prompt="test",
        )

        assert session._maybe_recall_hint("like we discussed earlier") is not None
        assert session._maybe_recall_hint("remember when we fixed that?") is not None
        assert session._maybe_recall_hint("go back to that file") is not None

    def test_hint_not_triggered(self):
        from loom.cowork.session import CoworkSession
        from loom.models.base import ModelResponse, TokenUsage
        from loom.tools import create_default_registry
        from tests.test_cowork import MockProvider

        provider = MockProvider([ModelResponse(text="ok", usage=TokenUsage())])
        session = CoworkSession(
            model=provider,
            tools=create_default_registry(),
            system_prompt="test",
        )

        assert session._maybe_recall_hint("Add a new function") is None
        assert session._maybe_recall_hint("Run the tests") is None
        assert session._maybe_recall_hint("Read src/main.py") is None
