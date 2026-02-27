"""Tests for Layer 2: Database and memory management."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.state.memory import Database, MemoryEntry, MemoryManager


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    """Provide an initialized database for testing."""
    database = Database(tmp_path / "test.db")
    await database.initialize()
    return database


@pytest.fixture
async def memory(db: Database) -> MemoryManager:
    """Provide a memory manager backed by a test database."""
    return MemoryManager(db)


class TestDatabase:
    """Test the SQLite database wrapper."""

    async def test_initialize_creates_tables(self, db: Database):
        tables = await db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {t["name"] for t in tables}
        assert "tasks" in table_names
        assert "memory_entries" in table_names
        assert "events" in table_names
        assert "learned_patterns" in table_names

    async def test_initialize_idempotent(self, db: Database):
        # Calling initialize again should not error
        await db.initialize()
        tables = await db.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = {t["name"] for t in tables}
        assert "tasks" in table_names
        assert "memory_entries" in table_names

    async def test_insert_and_get_task(self, db: Database):
        await db.insert_task(
            task_id="t1",
            goal="Test goal",
            workspace_path="/tmp/ws",
            status="pending",
        )
        task = await db.get_task("t1")
        assert task is not None
        assert task["goal"] == "Test goal"
        assert task["status"] == "pending"
        assert task["workspace_path"] == "/tmp/ws"

    async def test_get_task_missing(self, db: Database):
        result = await db.get_task("nonexistent")
        assert result is None

    async def test_update_task_status(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.update_task_status("t1", "executing")
        task = await db.get_task("t1")
        assert task["status"] == "executing"
        assert task["completed_at"] is None

    async def test_update_task_status_completed(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.update_task_status("t1", "completed")
        task = await db.get_task("t1")
        assert task["status"] == "completed"
        assert task["completed_at"] is not None

    async def test_list_tasks(self, db: Database):
        await db.insert_task(task_id="t1", goal="First")
        await db.insert_task(task_id="t2", goal="Second")
        tasks = await db.list_tasks()
        assert len(tasks) == 2

    async def test_list_tasks_filtered(self, db: Database):
        await db.insert_task(task_id="t1", goal="First", status="pending")
        await db.insert_task(task_id="t2", goal="Second", status="completed")
        pending = await db.list_tasks(status="pending")
        assert len(pending) == 1
        assert pending[0]["id"] == "t1"

    async def test_insert_memory_entry(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        entry = MemoryEntry(
            task_id="t1",
            subtask_id="s1",
            entry_type="decision",
            summary="Chose PostgreSQL",
            detail="We chose PostgreSQL for better JSON support.",
            tags="database,postgresql",
            relevance_to="s2,s3",
        )
        entry_id = await db.insert_memory_entry(entry)
        assert entry_id is not None
        assert entry_id > 0

    async def test_query_memory_by_task(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        for i in range(3):
            await db.insert_memory_entry(MemoryEntry(
                task_id="t1", subtask_id=f"s{i}",
                entry_type="decision", summary=f"Decision {i}",
            ))
        entries = await db.query_memory("t1")
        assert len(entries) == 3

    async def test_query_memory_by_subtask(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", subtask_id="s1", entry_type="decision", summary="A",
        ))
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", subtask_id="s2", entry_type="decision", summary="B",
        ))
        entries = await db.query_memory("t1", subtask_id="s1")
        assert len(entries) == 1
        assert entries[0].summary == "A"

    async def test_query_memory_by_type(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", entry_type="decision", summary="A",
        ))
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", entry_type="error", summary="B",
        ))
        entries = await db.query_memory("t1", entry_type="error")
        assert len(entries) == 1
        assert entries[0].summary == "B"

    async def test_query_memory_by_tags(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", entry_type="decision", summary="A",
            tags="database,postgresql",
        ))
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", entry_type="decision", summary="B",
            tags="api,rest",
        ))
        entries = await db.query_memory("t1", tags=["database"])
        assert len(entries) == 1
        assert entries[0].summary == "A"

    async def test_query_relevant_memory(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        # Direct match
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", subtask_id="s2", entry_type="tool_result",
            summary="Direct",
        ))
        # Relevant via relevance_to
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", subtask_id="s1", entry_type="discovery",
            summary="Relevant", relevance_to="s2,s3",
        ))
        # Global decision (always included)
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", subtask_id="s0", entry_type="decision",
            summary="Global",
        ))
        # Unrelated
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", subtask_id="s99", entry_type="tool_result",
            summary="Unrelated",
        ))
        entries = await db.query_relevant_memory("t1", "s2")
        summaries = {e.summary for e in entries}
        assert "Direct" in summaries
        assert "Relevant" in summaries
        assert "Global" in summaries
        assert "Unrelated" not in summaries

    async def test_search_memory(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", entry_type="discovery",
            summary="Found PostgreSQL ENUM issue",
            detail="ENUM types need conversion",
        ))
        await db.insert_memory_entry(MemoryEntry(
            task_id="t1", entry_type="decision",
            summary="Using REST API",
        ))
        results = await db.search_memory("t1", "PostgreSQL")
        assert len(results) == 1
        assert "ENUM" in results[0].summary

    async def test_insert_and_query_events(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.insert_event("t1", "corr-1", "task_started", {"msg": "hello"})
        await db.insert_event("t1", "corr-1", "subtask_completed", {"id": "s1"})

        events = await db.query_events("t1")
        assert len(events) == 2

        typed = await db.query_events("t1", event_type="task_started")
        assert len(typed) == 1

    async def test_task_run_lifecycle(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.insert_task_run(
            run_id="run-1",
            task_id="t1",
            status="queued",
            process_name="demo",
        )
        acquired = await db.acquire_task_run_lease(
            run_id="run-1",
            lease_owner="worker-A",
            lease_seconds=30,
        )
        assert acquired is True
        heartbeat = await db.heartbeat_task_run(
            run_id="run-1",
            lease_owner="worker-A",
            lease_seconds=30,
        )
        assert heartbeat is True
        await db.complete_task_run(run_id="run-1", status="completed")
        row = await db.get_task_run("run-1")
        assert row is not None
        assert row["status"] == "completed"

    async def test_mutation_ledger_roundtrip(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.upsert_mutation_ledger_entry(
            idempotency_key="idem-1",
            task_id="t1",
            run_id="run-1",
            subtask_id="s1",
            tool_name="write_file",
            args_hash="abc123",
            status="success",
            result_json="{\"success\":true}",
        )
        row = await db.get_mutation_ledger_entry("idem-1")
        assert row is not None
        assert row["tool_name"] == "write_file"
        assert row["status"] == "success"

    async def test_remediation_item_roundtrip(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.upsert_remediation_item({
            "id": "rem-1",
            "task_id": "t1",
            "run_id": "run-1",
            "subtask_id": "s1",
            "strategy": "unconfirmed_data",
            "state": "queued",
            "blocking": True,
            "missing_targets": ["a", "b"],
        })
        rows = await db.list_remediation_items(task_id="t1")
        assert len(rows) == 1
        assert rows[0]["id"] == "rem-1"
        assert rows[0]["blocking"] is True
        assert rows[0]["missing_targets"] == ["a", "b"]


class TestMemoryManager:
    """Test the high-level memory manager."""

    async def test_store_entry(self, memory: MemoryManager, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        entry_id = await memory.store(MemoryEntry(
            task_id="t1", entry_type="decision", summary="Test decision",
        ))
        assert entry_id > 0

    async def test_store_invalid_type_raises(self, memory: MemoryManager):
        with pytest.raises(ValueError, match="Invalid entry_type"):
            await memory.store(MemoryEntry(
                task_id="t1", entry_type="invalid_type", summary="Bad",
            ))

    async def test_store_many(self, memory: MemoryManager, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        entries = [
            MemoryEntry(task_id="t1", entry_type="decision", summary=f"D{i}")
            for i in range(5)
        ]
        ids = await memory.store_many(entries)
        assert len(ids) == 5

    async def test_query_relevant(self, memory: MemoryManager, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await memory.store(MemoryEntry(
            task_id="t1", subtask_id="s1", entry_type="decision", summary="A",
        ))
        results = await memory.query_relevant("t1", "s1")
        assert len(results) >= 1

    async def test_query_with_filters(self, memory: MemoryManager, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await memory.store(MemoryEntry(
            task_id="t1", entry_type="error", summary="Err",
        ))
        await memory.store(MemoryEntry(
            task_id="t1", entry_type="decision", summary="Dec",
        ))
        results = await memory.query("t1", entry_type="error")
        assert len(results) == 1
        assert results[0].summary == "Err"

    async def test_search(self, memory: MemoryManager, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await memory.store(MemoryEntry(
            task_id="t1", entry_type="discovery",
            summary="Found config file", tags="config",
        ))
        results = await memory.search("t1", "config")
        assert len(results) >= 1

    async def test_format_for_prompt_empty(self, memory: MemoryManager):
        result = memory.format_for_prompt([])
        assert "No relevant prior context" in result

    async def test_format_for_prompt(self, memory: MemoryManager):
        entries = [
            MemoryEntry(
                task_id="t1", subtask_id="s1",
                entry_type="decision", summary="Chose PostgreSQL",
            ),
            MemoryEntry(
                task_id="t1",
                entry_type="error", summary="Connection timeout",
                detail="Short detail",
            ),
        ]
        result = memory.format_for_prompt(entries)
        assert "[decision]" in result
        assert "Chose PostgreSQL" in result
        assert "[error]" in result
        assert "Short detail" in result
