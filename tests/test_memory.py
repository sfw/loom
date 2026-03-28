"""Tests for Layer 2: Database and memory management."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from loom.state.memory import Database, MemoryEntry, MemoryManager
from loom.state.migrations import MIGRATIONS
from loom.state.migrations.runner import MigrationStep


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

    async def test_initialize_migrates_legacy_events_before_sequence_indexes(
        self,
        tmp_path: Path,
    ):
        legacy_path = tmp_path / "legacy.db"
        async with aiosqlite.connect(legacy_path) as conn:
            await conn.executescript(
                """CREATE TABLE tasks (
                       id TEXT PRIMARY KEY,
                       goal TEXT NOT NULL,
                       status TEXT NOT NULL DEFAULT 'pending',
                       created_at TEXT NOT NULL DEFAULT (datetime('now')),
                       updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                   );
                   CREATE TABLE events (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       task_id TEXT NOT NULL,
                       correlation_id TEXT NOT NULL,
                       timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                       event_type TEXT NOT NULL,
                       data TEXT NOT NULL,
                       FOREIGN KEY (task_id) REFERENCES tasks(id)
                   );
                   CREATE INDEX idx_events_task ON events(task_id);
                   CREATE INDEX idx_events_correlation ON events(correlation_id);
                   CREATE INDEX idx_events_type ON events(event_type);
                """,
            )
            await conn.execute(
                """INSERT INTO tasks (id, goal, status, created_at, updated_at)
                   VALUES ('t1', 'legacy', 'pending', datetime('now'), datetime('now'))""",
            )
            await conn.execute(
                """INSERT INTO events (task_id, correlation_id, event_type, data)
                   VALUES ('t1', 'corr-1', 'task_started', '{}')""",
            )
            await conn.commit()

        db = Database(legacy_path)
        await db.initialize()

        async with aiosqlite.connect(legacy_path) as conn:
            cursor = await conn.execute("PRAGMA table_info(events)")
            columns = {str(row[1]) for row in await cursor.fetchall()}
            required_columns = {
                "run_id",
                "event_id",
                "sequence",
                "source_component",
                "schema_version",
            }
            assert required_columns.issubset(columns)

            cursor = await conn.execute("SELECT sequence FROM events LIMIT 1")
            row = await cursor.fetchone()
            assert row is not None
            assert int(row[0]) == 0

            for index_name in (
                "idx_events_task_sequence",
                "idx_events_run_sequence",
                "idx_events_event_id",
            ):
                cursor = await conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='index' AND name=? LIMIT 1",
                    (index_name,),
                )
                assert await cursor.fetchone() is not None

    async def test_initialize_stamps_schema_migrations_to_latest(self, tmp_path: Path):
        db_path = tmp_path / "fresh.db"
        db = Database(db_path)
        await db.initialize()
        await db.initialize()

        rows = await db.query("SELECT id FROM schema_migrations ORDER BY id")
        assert [str(row["id"]) for row in rows] == [step.id for step in MIGRATIONS]

    async def test_initialize_upgrades_pre_task_questions_fixture(self, tmp_path: Path):
        db_path = tmp_path / "pre-task-questions.db"
        first = MIGRATIONS[0]
        async with aiosqlite.connect(db_path) as conn:
            await conn.executescript(
                """CREATE TABLE tasks (
                       id TEXT PRIMARY KEY,
                       goal TEXT NOT NULL,
                       status TEXT NOT NULL DEFAULT 'pending',
                       created_at TEXT NOT NULL DEFAULT (datetime('now')),
                       updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                   );
                   CREATE TABLE events (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       task_id TEXT NOT NULL,
                       run_id TEXT NOT NULL DEFAULT '',
                       correlation_id TEXT NOT NULL,
                       event_id TEXT NOT NULL DEFAULT '',
                       sequence INTEGER NOT NULL DEFAULT 0,
                       timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                       event_type TEXT NOT NULL,
                       source_component TEXT NOT NULL DEFAULT '',
                       schema_version INTEGER NOT NULL DEFAULT 1,
                       data TEXT NOT NULL,
                       FOREIGN KEY (task_id) REFERENCES tasks(id)
                   );
                   CREATE TABLE schema_migrations (
                       id TEXT PRIMARY KEY,
                       applied_at TEXT NOT NULL,
                       duration_ms INTEGER NOT NULL DEFAULT 0,
                       checksum TEXT NOT NULL,
                       notes TEXT NOT NULL DEFAULT ''
                   );
                """,
            )
            await conn.execute(
                """INSERT INTO schema_migrations (id, applied_at, duration_ms, checksum, notes)
                   VALUES (?, datetime('now'), 1, ?, 'legacy bootstrap')""",
                (first.id, first.checksum),
            )
            await conn.commit()

        db = Database(db_path)
        await db.initialize()

        ids = await db.query("SELECT id FROM schema_migrations ORDER BY id")
        assert [str(row["id"]) for row in ids] == [step.id for step in MIGRATIONS]

    async def test_initialize_upgrades_pre_validity_fixture(self, tmp_path: Path):
        db_path = tmp_path / "pre-validity.db"
        first = MIGRATIONS[0]
        second = MIGRATIONS[1]
        async with aiosqlite.connect(db_path) as conn:
            await conn.executescript(
                """CREATE TABLE tasks (
                       id TEXT PRIMARY KEY,
                       goal TEXT NOT NULL,
                       status TEXT NOT NULL DEFAULT 'pending',
                       created_at TEXT NOT NULL DEFAULT (datetime('now')),
                       updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                   );
                   CREATE TABLE events (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       task_id TEXT NOT NULL,
                       run_id TEXT NOT NULL DEFAULT '',
                       correlation_id TEXT NOT NULL,
                       event_id TEXT NOT NULL DEFAULT '',
                       sequence INTEGER NOT NULL DEFAULT 0,
                       timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                       event_type TEXT NOT NULL,
                       source_component TEXT NOT NULL DEFAULT '',
                       schema_version INTEGER NOT NULL DEFAULT 1,
                       data TEXT NOT NULL,
                       FOREIGN KEY (task_id) REFERENCES tasks(id)
                   );
                   CREATE TABLE task_questions (
                       question_id TEXT PRIMARY KEY,
                       task_id TEXT NOT NULL,
                       subtask_id TEXT NOT NULL DEFAULT '',
                       status TEXT NOT NULL DEFAULT 'pending',
                       request_payload TEXT NOT NULL,
                       answer_payload TEXT,
                       created_at TEXT NOT NULL DEFAULT (datetime('now')),
                       updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                       resolved_at TEXT,
                       timeout_at TEXT,
                       FOREIGN KEY (task_id) REFERENCES tasks(id)
                   );
                   CREATE TABLE schema_migrations (
                       id TEXT PRIMARY KEY,
                       applied_at TEXT NOT NULL,
                       duration_ms INTEGER NOT NULL DEFAULT 0,
                       checksum TEXT NOT NULL,
                       notes TEXT NOT NULL DEFAULT ''
                   );
                """,
            )
            await conn.execute(
                """INSERT INTO schema_migrations (id, applied_at, duration_ms, checksum, notes)
                   VALUES (?, datetime('now'), 1, ?, 'legacy bootstrap')""",
                (first.id, first.checksum),
            )
            await conn.execute(
                """INSERT INTO schema_migrations (id, applied_at, duration_ms, checksum, notes)
                   VALUES (?, datetime('now'), 1, ?, 'legacy bootstrap')""",
                (second.id, second.checksum),
            )
            await conn.commit()

        db = Database(db_path)
        await db.initialize()

        tables = await db.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_claims'",
        )
        assert tables

    async def test_initialize_rolls_back_on_failed_migration(self, tmp_path: Path, monkeypatch):
        db_path = tmp_path / "rollback.db"

        async def _apply(conn):
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS rollback_probe(id INTEGER PRIMARY KEY AUTOINCREMENT)",
            )
            raise RuntimeError("forced migration failure")

        async def _verify(_conn):
            return None

        monkeypatch.setattr(
            "loom.state.memory.MIGRATIONS",
            (
                MigrationStep(
                    id="99990101_001_forced_failure",
                    description="forced failure",
                    checksum="forced-failure",
                    apply=_apply,
                    verify=_verify,
                ),
            ),
        )

        db = Database(db_path)
        with pytest.raises(RuntimeError, match="forced migration failure"):
            await db.initialize()

        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            )
            row = await cursor.fetchone()
            assert row is not None
            # Transaction rollback must remove partial schema/migration artifacts.
            assert int(row[0]) == 0

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
        await db.insert_event(
            "t1",
            "corr-1",
            "task_started",
            {"msg": "hello"},
            timestamp="2026-03-06T00:00:00+00:00",
            event_id="evt-1",
            sequence=1,
            run_id="run-1",
            source_component="tests",
            schema_version=1,
        )
        await db.insert_event(
            "t1",
            "corr-1",
            "subtask_completed",
            {"id": "s1"},
            timestamp="2026-03-06T00:00:01+00:00",
            event_id="evt-2",
            sequence=2,
            run_id="run-1",
            source_component="tests",
            schema_version=1,
        )

        events = await db.query_events("t1")
        assert len(events) == 2
        assert events[0]["event_id"] == "evt-2"
        assert events[0]["sequence"] == 2
        assert events[1]["event_id"] == "evt-1"
        assert events[1]["timestamp"] == "2026-03-06T00:00:00+00:00"

        typed = await db.query_events("t1", event_type="task_started")
        assert len(typed) == 1

        ascending = await db.query_events("t1", after_id=1, ascending=True)
        assert len(ascending) == 1
        assert ascending[0]["event_id"] == "evt-2"

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

    async def test_task_question_lifecycle(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        pending = await db.upsert_pending_task_question(
            question_id="q-1",
            task_id="t1",
            subtask_id="s1",
            request_payload={
                "question_id": "q-1",
                "question": "Choose stack",
                "question_type": "single_choice",
            },
        )
        assert pending["status"] == "pending"
        assert pending["request_payload"]["question"] == "Choose stack"

        listed_pending = await db.list_pending_task_questions("t1")
        assert len(listed_pending) == 1
        assert listed_pending[0]["question_id"] == "q-1"

        resolved = await db.resolve_task_question(
            task_id="t1",
            question_id="q-1",
            status="answered",
            answer_payload={
                "question_id": "q-1",
                "response_type": "single_choice",
                "selected_option_ids": ["py"],
                "selected_labels": ["Python"],
                "custom_response": "",
                "source": "api",
            },
        )
        assert resolved is not None
        assert resolved["status"] == "answered"
        assert resolved["answer_payload"]["selected_option_ids"] == ["py"]

        listed_pending = await db.list_pending_task_questions("t1")
        assert listed_pending == []
        all_rows = await db.list_task_questions("t1")
        assert len(all_rows) == 1
        assert all_rows[0]["status"] == "answered"

    async def test_task_question_scope_reuses_existing_pending_row(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        first = await db.upsert_pending_task_question(
            question_id="q-1",
            task_id="t1",
            subtask_id="s1",
            request_payload={"question_id": "q-1", "question": "First"},
        )
        second = await db.upsert_pending_task_question(
            question_id="q-2",
            task_id="t1",
            subtask_id="s1",
            request_payload={"question_id": "q-2", "question": "Second"},
        )
        assert first["question_id"] == "q-1"
        assert second["question_id"] == "q-1"

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

    async def test_iteration_tables_roundtrip(self, db: Database):
        await db.insert_task(task_id="t1", goal="Test")
        await db.upsert_iteration_run(
            loop_run_id="iter-1",
            task_id="t1",
            run_id="run-1",
            subtask_id="s1",
            phase_id="rewrite",
            policy_snapshot={"max_attempts": 4},
            terminal_reason="",
            attempt_count=1,
            replan_count=0,
            exhaustion_fingerprint="",
            metadata={"foo": "bar"},
        )
        attempt_id = await db.insert_iteration_attempt(
            loop_run_id="iter-1",
            task_id="t1",
            run_id="run-1",
            subtask_id="s1",
            phase_id="rewrite",
            attempt_index=1,
            status="retrying",
            summary="score below threshold",
            gate_summary={"blocking_failures": ["score"]},
            budget_snapshot={"used": {"tokens": 1200}},
        )
        assert attempt_id > 0
        gate_id = await db.insert_iteration_gate_result(
            loop_run_id="iter-1",
            attempt_id=attempt_id,
            task_id="t1",
            run_id="run-1",
            subtask_id="s1",
            phase_id="rewrite",
            attempt_index=1,
            gate_id="score",
            gate_type="tool_metric",
            status="fail",
            blocking=True,
            reason_code="gate_threshold_not_met",
            measured_value=72,
            threshold_value=80,
            detail="score too low",
        )
        assert gate_id > 0

        runs = await db.list_iteration_runs(task_id="t1")
        assert len(runs) == 1
        assert runs[0]["loop_run_id"] == "iter-1"
        assert runs[0]["policy_snapshot"]["max_attempts"] == 4

        attempts = await db.list_iteration_attempts(loop_run_id="iter-1")
        assert len(attempts) == 1
        assert attempts[0]["gate_summary"]["blocking_failures"] == ["score"]

        gates = await db.list_iteration_gate_results(loop_run_id="iter-1")
        assert len(gates) == 1
        assert gates[0]["gate_id"] == "score"
        assert gates[0]["blocking"] is True
        assert gates[0]["measured_value"] == 72

    async def test_claim_validity_lineage_tables_roundtrip(self, db: Database):
        await db.insert_task(task_id="t1", goal="Claim lineage")
        claim_ids = await db.insert_artifact_claims(
            task_id="t1",
            run_id="run-1",
            subtask_id="s1",
            phase_id="analysis",
            claims=[
                {
                    "claim_id": "CLM-001",
                    "text": "Revenue increased 12%.",
                    "claim_type": "numeric",
                    "criticality": "critical",
                    "status": "supported",
                    "reason_code": "claim_supported",
                    "metadata": {"as_of": "2025-12-31"},
                },
            ],
        )
        assert len(claim_ids) == 1

        verification_ids = await db.insert_claim_verification_results(
            task_id="t1",
            run_id="run-1",
            subtask_id="s1",
            phase_id="analysis",
            results=[
                {
                    "claim_id": "CLM-001",
                    "status": "supported",
                    "reason_code": "claim_supported",
                    "verifier": "verification_gates",
                    "confidence": 0.92,
                    "metadata": {"notes": "fact_checker verdict supported"},
                },
            ],
        )
        assert len(verification_ids) == 1

        link_ids = await db.insert_claim_evidence_links(
            task_id="t1",
            run_id="run-1",
            subtask_id="s1",
            links=[
                {
                    "claim_id": "CLM-001",
                    "evidence_id": "EV-WRITE-ABC123",
                    "link_type": "supporting",
                    "score": 1.0,
                    "metadata": {"source": "artifact"},
                },
            ],
        )
        assert len(link_ids) == 1

        summary_id = await db.insert_artifact_validity_summary(
            task_id="t1",
            run_id="run-1",
            subtask_id="s1",
            phase_id="analysis",
            extracted_count=1,
            supported_count=1,
            contradicted_count=0,
            insufficient_evidence_count=0,
            pruned_count=0,
            supported_ratio=1.0,
            gate_decision="pass",
            reason_code="claim_supported",
            metadata={"validity_contract_hash": "hash-123"},
        )
        assert summary_id > 0

        claims = await db.query(
            "SELECT claim_id, lifecycle_state FROM artifact_claims WHERE task_id = ?",
            ("t1",),
        )
        assert len(claims) == 1
        assert claims[0]["claim_id"] == "CLM-001"
        assert claims[0]["lifecycle_state"] == "supported"

        results = await db.query(
            (
                "SELECT claim_id, status, reason_code "
                "FROM claim_verification_results WHERE task_id = ?"
            ),
            ("t1",),
        )
        assert len(results) == 1
        assert results[0]["claim_id"] == "CLM-001"
        assert results[0]["status"] == "supported"

        links = await db.query(
            "SELECT claim_id, evidence_id FROM claim_evidence_links WHERE task_id = ?",
            ("t1",),
        )
        assert len(links) == 1
        assert links[0]["evidence_id"] == "EV-WRITE-ABC123"

        summaries = await db.query(
            (
                "SELECT gate_decision, supported_count "
                "FROM artifact_validity_summaries WHERE task_id = ?"
            ),
            ("t1",),
        )
        assert len(summaries) == 1
        assert summaries[0]["gate_decision"] == "pass"
        assert summaries[0]["supported_count"] == 1


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
