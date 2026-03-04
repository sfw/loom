"""Tests for cowork memory indexer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from loom.cowork.memory_indexer import CoworkMemoryIndexer
from loom.cowork.session_state import SessionState
from loom.state.conversation_store import ConversationStore
from loom.state.memory import Database


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "indexer.db")
    await database.initialize()
    return database


@pytest.fixture
async def store(db: Database) -> ConversationStore:
    return ConversationStore(db)


class TestCoworkMemoryIndexer:
    def test_extract_entries_from_explicit_markers(self, store: ConversationStore):
        state = SessionState(session_id="s", workspace="/tmp", model_name="m")
        indexer = CoworkMemoryIndexer(
            store=store,
            session_id="s",
            session_state=state,
            llm_extraction_enabled=False,
        )
        rows = [
            {
                "turn_number": 1,
                "role": "user",
                "content": "DECISION: Route compaction to compactor role.",
            },
            {
                "turn_number": 2,
                "role": "assistant",
                "content": "PROPOSAL: Add typed recall-index sections.",
            },
            {
                "turn_number": 3,
                "role": "assistant",
                "content": "OPEN_QUESTION: Should force_fts be default?",
            },
        ]
        entries = indexer._extract_entries_from_turns(rows)
        entry_types = [entry.entry_type for entry in entries]
        assert "decision" in entry_types
        assert "proposal" in entry_types
        assert "open_question" in entry_types

    async def test_strict_role_skips_llm_extraction(self, store: ConversationStore):
        state = SessionState(session_id="s", workspace="/tmp", model_name="m")
        model = SimpleNamespace(complete=AsyncMock())
        indexer = CoworkMemoryIndexer(
            store=store,
            session_id="s",
            session_state=state,
            model=model,
            model_role="active",
            llm_extraction_enabled=True,
            role_strict=True,
        )
        rows = [
            {"turn_number": 1, "role": "user", "content": "DECISION: Keep strict role routing."},
        ]
        extracted = await indexer._extract_entries_with_model(rows)
        assert extracted == []
        model.complete.assert_not_awaited()

    async def test_index_to_turn_updates_state_snapshot(self, store: ConversationStore):
        sid = await store.create_session(workspace="/tmp", model_name="mock")
        await store.append_turn(sid, 1, "user", "DECISION: Keep long-session recall index.")
        await store.append_turn(sid, 2, "assistant", "RESEARCH: FTS fallback handles odd queries.")

        state = SessionState(session_id=sid, workspace="/tmp", model_name="mock")
        indexer = CoworkMemoryIndexer(
            store=store,
            session_id=sid,
            session_state=state,
            llm_extraction_enabled=False,
        )

        await indexer.reindex_up_to_turn(2)
        assert state.active_decisions
        assert state.recent_research
        assert state.memory_index_last_indexed_turn >= 2
