"""Tests for the conversation store (persistence layer)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from loom.state.conversation_store import ConversationStore
from loom.state.memory import Database
from loom.tools.registry import ToolResult


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
        assert "cowork_chat_events" in table_names
        assert "cowork_memory_entries" in table_names
        assert "cowork_memory_index_state" in table_names

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

    async def test_upsert_and_search_cowork_memory_entries(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        await store.upsert_cowork_memory_entry(
            sid,
            {
                "entry_type": "decision",
                "status": "active",
                "summary": "Use compactor for cowork memory extraction",
                "rationale": "Lower latency and cost",
                "topic": "cowork-memory",
                "source_turn_start": 4,
                "source_turn_end": 5,
                "source_roles": ["assistant"],
                "evidence_excerpt": "DECISION: use compactor role",
                "confidence": 0.93,
                "fingerprint": "fp-1",
            },
        )
        await store.upsert_cowork_memory_entry(
            sid,
            {
                "entry_type": "open_question",
                "status": "active",
                "summary": "Should force_fts be enabled?",
                "topic": "cowork-memory",
                "source_turn_start": 6,
                "source_turn_end": 6,
                "source_roles": ["user"],
                "confidence": 0.51,
                "fingerprint": "fp-2",
            },
        )

        rows = await store.search_cowork_memory_entries(
            sid,
            query="compactor",
            limit=10,
        )
        assert rows
        assert rows[0]["entry_type"] in {"decision", "open_question"}
        assert rows[0]["status"] == "active"
        assert isinstance(rows[0]["source_roles"], list)

        snapshot = await store.get_cowork_memory_active_snapshot(sid)
        assert snapshot["active_decisions"]
        assert snapshot["open_questions"]

    async def test_cowork_memory_index_state_round_trip(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        state = await store.get_cowork_memory_index_state(sid)
        assert state["last_indexed_turn"] == 0
        assert state["index_degraded"] is False

        await store.upsert_cowork_memory_index_state(
            sid,
            last_indexed_turn=17,
            index_degraded=True,
            last_error="parse-failed",
            failure_count=2,
            index_version=2,
        )
        updated = await store.get_cowork_memory_index_state(sid)
        assert updated["last_indexed_turn"] == 17
        assert updated["index_degraded"] is True
        assert updated["failure_count"] == 2

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

        await store.append_turn(
            sid, 1, "tool", "file content",
            tool_name="read_file", tool_call_id="c1",
        )
        await store.append_turn(
            sid, 2, "tool", "success",
            tool_name="write_file", tool_call_id="c2",
        )
        await store.append_turn(
            sid, 3, "tool", "more content",
            tool_name="read_file", tool_call_id="c3",
        )

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

    async def test_link_run_round_trip(self, store: ConversationStore, db: Database):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        await db.insert_task(
            task_id="task-1",
            goal="Run task",
            workspace_path="/tmp",
            status="pending",
        )

        await store.link_run(sid, "task-1")
        await store.link_run(sid, "task-1")

        linked_runs = await store.list_linked_runs(sid)
        assert len(linked_runs) == 1
        assert linked_runs[0]["run_id"] == "task-1"

        linked_conversations = await store.list_linked_conversations("task-1")
        assert len(linked_conversations) == 1
        assert linked_conversations[0]["session_id"] == sid

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

        tc = [{"id": "c1", "type": "function",
              "function": {"name": "read_file", "arguments": "{}"}}]
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

    async def test_append_and_get_chat_events(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        seq1 = await store.append_chat_event(
            sid,
            "user_message",
            {"text": "hello"},
        )
        seq2 = await store.append_chat_event(
            sid,
            "assistant_text",
            {"text": "hi"},
        )

        events = await store.get_chat_events(sid, limit=10)
        assert seq1 == 1
        assert seq2 == 2
        assert len(events) == 2
        assert events[0]["event_type"] == "user_message"
        assert events[0]["payload"]["text"] == "hello"
        assert events[1]["event_type"] == "assistant_text"
        assert events[1]["seq"] == 2

    async def test_append_chat_event_with_concurrent_readers(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        stop = asyncio.Event()

        async def reader() -> None:
            while not stop.is_set():
                await store.get_session(sid)
                await store.get_chat_events(sid, limit=64)
                await store.get_last_chat_seq(sid)
                await asyncio.sleep(0)

        readers = [asyncio.create_task(reader()) for _ in range(8)]
        await asyncio.sleep(0)
        try:
            await store.append_chat_event(
                sid,
                "user_message",
                {"text": "hello"},
            )
            for index in range(1, 33):
                seq = await store.append_chat_event(
                    sid,
                    "assistant_thinking",
                    {"text": f"chunk {index}", "streaming": True},
                )
                assert seq == index + 1
            final_seq = await store.append_chat_event(
                sid,
                "assistant_text",
                {"text": "done"},
            )
        finally:
            stop.set()
            await asyncio.gather(*readers)

        assert final_seq == 34
        events = await store.get_chat_events(sid, limit=40)
        assert [row["seq"] for row in events] == list(range(1, 35))
        assert events[-1]["event_type"] == "assistant_text"

    async def test_get_chat_events_before_seq(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        for index in range(1, 6):
            await store.append_chat_event(
                sid,
                "info",
                {"text": f"line {index}"},
            )

        page = await store.get_chat_events(sid, before_seq=5, limit=2)
        assert len(page) == 2
        assert [row["seq"] for row in page] == [3, 4]

    async def test_get_chat_events_between_after_and_before_seq(
        self, store: ConversationStore,
    ):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        for index in range(1, 9):
            await store.append_chat_event(
                sid,
                "info",
                {"text": f"line {index}"},
            )

        page = await store.get_chat_events(
            sid,
            after_seq=3,
            before_seq=7,
            limit=10,
        )
        assert [row["seq"] for row in page] == [4, 5, 6]

    async def test_get_chat_events_payload_parse_failure_non_fatal(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        await store.append_chat_event(
            sid,
            "info",
            "{not-json",
        )

        events = await store.get_chat_events(sid, limit=10)
        assert len(events) == 1
        assert events[0]["payload_parse_error"] is True
        assert events[0]["payload"]["raw"] == "{not-json"

    async def test_append_chat_event_rejects_non_positive_explicit_seq(
        self, store: ConversationStore,
    ):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        with pytest.raises(ValueError):
            await store.append_chat_event(
                sid,
                "info",
                {"text": "x"},
                seq=0,
            )

    async def test_synthesize_chat_events_from_turns(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="m")

        await store.append_turn(sid, 1, "user", "Find TODOs")
        tool_calls = [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "ripgrep_search",
                "arguments": "{\"pattern\":\"TODO\"}",
            },
        }]
        await store.append_turn(
            sid,
            2,
            "assistant",
            "Searching now.",
            tool_calls=tool_calls,
        )
        result = ToolResult.ok("2 matches")
        await store.append_turn(
            sid,
            3,
            "tool",
            result.to_json(),
            tool_call_id="call_1",
            tool_name="ripgrep_search",
        )

        events = await store.synthesize_chat_events_from_turns(sid, limit=10)
        assert [row["event_type"] for row in events] == [
            "user_message",
            "assistant_text",
            "tool_call_started",
            "tool_call_completed",
        ]
        assert events[0]["payload"]["text"] == "Find TODOs"
        assert events[2]["payload"]["tool_name"] == "ripgrep_search"
        assert events[3]["payload"]["success"] is True

    async def test_synthesize_chat_events_handles_malformed_tool_payload(
        self, store: ConversationStore,
    ):
        sid = await store.create_session(workspace="/tmp", model_name="m")
        await store.append_turn(
            sid,
            1,
            "tool",
            "<<<not-json>>>",
            tool_call_id="call_1",
            tool_name="ripgrep_search",
        )

        events = await store.synthesize_chat_events_from_turns(sid, limit=10)
        assert [row["event_type"] for row in events] == ["tool_call_completed"]
        payload = events[0]["payload"]
        assert payload["success"] is False
        assert payload["error"] == "Malformed tool result payload"
        assert "not-json" in payload["output"]
