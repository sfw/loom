"""Migration registry and schema guard tests."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from loom.state.memory import Database
from loom.state.migrations import MIGRATIONS, migration_status
from loom.state.migrations.runner import MigrationStep, apply_pending_migrations


class TestMigrationGuards:
    async def test_registry_ids_are_unique_and_monotonic(self):
        ids = [step.id for step in MIGRATIONS]
        assert len(ids) == len(set(ids))
        assert ids == sorted(ids)

    async def test_latest_registry_version_matches_fresh_schema(self, tmp_path):
        db_path = tmp_path / "fresh.db"
        db = Database(db_path)
        await db.initialize()

        async with aiosqlite.connect(db_path) as conn:
            payload = await migration_status(conn, steps=MIGRATIONS)
        pending = list(payload.get("pending_ids", []))
        assert pending == []
        assert payload.get("latest_id") == MIGRATIONS[-1].id

    async def test_migration_runner_emits_diagnostics_callbacks(self, tmp_path):
        db_path = tmp_path / "diag.db"
        emitted: list[str] = []

        async def _apply(conn):
            await conn.execute("CREATE TABLE IF NOT EXISTS diag_probe(id INTEGER)")

        async def _verify(conn):
            cursor = await conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='diag_probe' LIMIT 1",
            )
            assert await cursor.fetchone() is not None

        step = MigrationStep(
            id="99990101_002_diag_probe",
            description="diagnostic probe",
            checksum="diag-probe",
            apply=_apply,
            verify=_verify,
        )

        def _report(event_type: str, _payload: dict[str, object]) -> None:
            emitted.append(event_type)

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("BEGIN IMMEDIATE")
            await apply_pending_migrations(
                conn,
                steps=(step,),
                reporter=_report,
                db_path=str(db_path),
            )
            await conn.commit()

        assert "db_migration_start" in emitted
        assert "db_migration_applied" in emitted

    def test_base_schema_file_exists(self):
        base_path = Path("src/loom/state/schema/base.sql")
        assert base_path.exists()

    async def test_fresh_schema_contains_workspace_registry_tables(self, tmp_path):
        db_path = tmp_path / "workspace-fresh.db"
        db = Database(db_path)
        await db.initialize()

        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                  AND name IN ('workspaces', 'workspace_settings', 'conversation_run_links')
                ORDER BY name
                """
            )
            rows = await cursor.fetchall()
        assert [row[0] for row in rows] == [
            "conversation_run_links",
            "workspace_settings",
            "workspaces",
        ]

    async def test_fresh_schema_contains_conversation_turn_metadata(self, tmp_path):
        db_path = tmp_path / "conversation-turn-metadata.db"
        db = Database(db_path)
        await db.initialize()

        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute("PRAGMA table_info(conversation_turns)")
            rows = await cursor.fetchall()

        column_names = {str(row[1]) for row in rows}
        assert "metadata" in column_names
