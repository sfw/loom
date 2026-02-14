"""Tests for the conversation store (persistence layer)."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.state.conversation_store import ConversationStore
from loom.state.memory import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    await database.initialize()
    return database


@pytest.fixture
async def store(db: Database) -> ConversationStore:
    return ConversationStore(db)


class TestConversationStore:
    async def test_tables_created(self, db: Database):
        """Schema creates conversation_turns and cowork_sessions tables."""
        tables = await db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {t["name"] for t in tables}
        assert "conversation_turns" in table_names
        assert "cowork_sessions" in table_names

    async def test_create_session(self, store: ConversationStore):
        session_id = await store.create_session(
            workspace="/tmp/ws", model_name="test-model", system_prompt="Hello",
        )
        assert session_id
        session = await store.get_session(session_id)
        assert session is not None
        assert session["workspace_path"] == "/tmp/ws"
        assert session["model_name"] == "test-model"
        assert session["is_active"] == 1

    async def test_list_sessions(self, store: ConversationStore):
        await store.create_session(workspace="/tmp/a", model_name="m1")
        await store.create_session(workspace="/tmp/b", model_name="m2")
        await store.create_session(workspace="/tmp/a", model_name="m3")

        all_sessions = await store.list_sessions()
        assert len(all_sessions) == 3

        ws_a = await store.list_sessions(workspace="/tmp/a")
        assert len(ws_a) == 2

    async def test_append_and_get_turns(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        await store.append_turn(sid, 1, "user", "Hello")
        await store.append_turn(sid, 2, "assistant", "Hi there!")
        await store.append_turn(sid, 3, "user", "Help me")

        turns = await store.get_turns(sid)
        assert len(turns) == 3
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Hello"
        assert turns[1]["role"] == "assistant"
        assert turns[2]["turn_number"] == 3

    async def test_get_recent_turns(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        for i in range(20):
            await store.append_turn(sid, i + 1, "user", f"Message {i}")

        recent = await store.get_recent_turns(sid, limit=5)
        assert len(recent) == 5
        assert recent[0]["turn_number"] == 16  # most recent 5, in order
        assert recent[-1]["turn_number"] == 20

    async def test_search_turns(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        await store.append_turn(sid, 1, "user", "Let's add JWT authentication")
        await store.append_turn(sid, 2, "assistant", "I'll implement JWT support")
        await store.append_turn(sid, 3, "user", "Now fix the database migration")
        await store.append_turn(sid, 4, "assistant", "Looking at the migration...")

        results = await store.search_turns(sid, "JWT")
        assert len(results) == 2
        assert any("JWT" in r["content"] for r in results)

        results = await store.search_turns(sid, "migration")
        assert len(results) == 2

    async def test_search_tool_calls(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        await store.append_turn(sid, 1, "tool", "file content", tool_name="read_file", tool_call_id="c1")
        await store.append_turn(sid, 2, "tool", "success", tool_name="write_file", tool_call_id="c2")
        await store.append_turn(sid, 3, "tool", "more content", tool_name="read_file", tool_call_id="c3")

        results = await store.search_tool_calls(sid, "read_file")
        assert len(results) == 2

        results = await store.search_tool_calls(sid, "write_file")
        assert len(results) == 1

    async def test_get_turn_range(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        for i in range(10):
            await store.append_turn(sid, i + 1, "user", f"Message {i}")

        turns = await store.get_turn_range(sid, 3, 7)
        assert len(turns) == 5
        assert turns[0]["turn_number"] == 3
        assert turns[-1]["turn_number"] == 7

    async def test_get_turn_count(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        assert await store.get_turn_count(sid) == 0

        await store.append_turn(sid, 1, "user", "Hello")
        await store.append_turn(sid, 2, "assistant", "Hi")
        assert await store.get_turn_count(sid) == 2

    async def test_update_session(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        await store.update_session(
            sid, total_tokens=5000, turn_count=10,
            session_state={"focus": "auth"}, is_active=False,
        )

        session = await store.get_session(sid)
        assert session["total_tokens"] == 5000
        assert session["turn_count"] == 10
        assert session["is_active"] == 0
        assert '"focus"' in session["session_state"]

    async def test_resume_session(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        await store.append_turn(sid, 1, "user", "Hello")
        await store.append_turn(sid, 2, "assistant", "Hi!")
        await store.append_turn(sid, 3, "user", "Help me code")

        messages = await store.resume_session(sid)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[-1]["content"] == "Help me code"

    async def test_tool_calls_persisted(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        tc = [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]
        await store.append_turn(sid, 1, "assistant", "Reading...", tool_calls=tc)

        turns = await store.get_turns(sid)
        assert turns[0]["tool_calls"] is not None

        messages = await store.resume_session(sid)
        assert "tool_calls" in messages[0]

    async def test_token_estimation(self, store: ConversationStore):
        """Turns get token count estimates stored."""
        sid = await store.create_session(workspace="/tmp", model_name="m")

        await store.append_turn(sid, 1, "user", "x" * 400)  # ~100 tokens

        turns = await store.get_turns(sid)
        assert turns[0]["token_count"] == 100

    async def test_sessions_isolated(self, store: ConversationStore):
        """Turns from different sessions don't mix."""
        s1 = await store.create_session(workspace="/tmp", model_name="m")
        s2 = await store.create_session(workspace="/tmp", model_name="m")

        await store.append_turn(s1, 1, "user", "Session 1 message")
        await store.append_turn(s2, 1, "user", "Session 2 message")

        t1 = await store.get_turns(s1)
        t2 = await store.get_turns(s2)
        assert len(t1) == 1
        assert len(t2) == 1
        assert t1[0]["content"] == "Session 1 message"
        assert t2[0]["content"] == "Session 2 message"
