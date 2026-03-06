"""SQLite schema migration runner for Loom."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import aiosqlite

from loom.events.types import (
    DB_MIGRATION_APPLIED,
    DB_MIGRATION_FAILED,
    DB_MIGRATION_START,
    DB_MIGRATION_VERIFY_FAILED,
)

MigrationFn = Callable[[aiosqlite.Connection], Awaitable[None]]
MigrationReporter = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class MigrationStep:
    """One schema migration step with explicit verification."""

    id: str
    description: str
    checksum: str
    apply: MigrationFn
    verify: MigrationFn


class MigrationExecutionError(RuntimeError):
    """Raised when a specific migration step fails during apply/verify."""

    def __init__(
        self,
        *,
        migration_id: str,
        phase: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.migration_id = str(migration_id)
        self.phase = str(phase)


def _emit_diagnostic(
    reporter: MigrationReporter | None,
    event_type: str,
    payload: dict[str, object],
) -> None:
    if reporter is None:
        return
    try:
        reporter(event_type, payload)
    except Exception:
        # Diagnostics must never break migration execution.
        return


async def has_user_tables(db: aiosqlite.Connection) -> bool:
    """Return True when DB has any non-internal tables."""
    cursor = await db.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "AND name <> 'schema_migrations' LIMIT 1",
    )
    row = await cursor.fetchone()
    return row is not None


async def table_exists(db: aiosqlite.Connection, table_name: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    )
    return await cursor.fetchone() is not None


async def index_exists(db: aiosqlite.Connection, index_name: str) -> bool:
    cursor = await db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=? LIMIT 1",
        (index_name,),
    )
    return await cursor.fetchone() is not None


async def table_columns(db: aiosqlite.Connection, table_name: str) -> set[str]:
    cursor = await db.execute(f"PRAGMA table_info({table_name})")
    rows = await cursor.fetchall()
    return {str(row[1]) for row in rows}


async def ensure_migration_table(db: aiosqlite.Connection) -> None:
    await db.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
               id TEXT PRIMARY KEY,
               applied_at TEXT NOT NULL,
               duration_ms INTEGER NOT NULL DEFAULT 0,
               checksum TEXT NOT NULL,
               notes TEXT NOT NULL DEFAULT ''
           )""",
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at "
        "ON schema_migrations(applied_at)",
    )


async def applied_migration_checksums(db: aiosqlite.Connection) -> dict[str, str]:
    if not await table_exists(db, "schema_migrations"):
        return {}
    cursor = await db.execute("SELECT id, checksum FROM schema_migrations")
    rows = await cursor.fetchall()
    return {
        str(row[0]): str(row[1] or "")
        for row in rows
    }


async def apply_pending_migrations(
    db: aiosqlite.Connection,
    *,
    steps: tuple[MigrationStep, ...],
    reporter: MigrationReporter | None = None,
    db_path: str = "",
) -> list[str]:
    """Apply all pending migration steps and return applied IDs."""
    await ensure_migration_table(db)
    applied = await applied_migration_checksums(db)
    applied_now: list[str] = []
    seen_ids: set[str] = set()

    for step in steps:
        if step.id in seen_ids:
            raise RuntimeError(f"Duplicate migration id in registry: {step.id}")
        seen_ids.add(step.id)
        existing_checksum = applied.get(step.id)
        if existing_checksum is not None:
            if existing_checksum and existing_checksum != step.checksum:
                _emit_diagnostic(
                    reporter,
                    DB_MIGRATION_FAILED,
                    {
                        "migration_id": step.id,
                        "phase": "checksum",
                        "db_path": db_path,
                        "error_class": "MigrationChecksumMismatch",
                        "error_message": (
                            "Database was initialized with a different migration definition."
                        ),
                        "actionable_suggestion_key": "loom_db_doctor_then_migrate",
                    },
                )
                raise RuntimeError(
                    f"Migration checksum mismatch for {step.id}. "
                    "Database was initialized with a different migration definition."
                )
            continue

        _emit_diagnostic(
            reporter,
            DB_MIGRATION_START,
            {
                "migration_id": step.id,
                "db_path": db_path,
                "description": step.description,
            },
        )
        started = time.monotonic()
        try:
            await step.apply(db)
        except Exception as e:
            _emit_diagnostic(
                reporter,
                DB_MIGRATION_FAILED,
                {
                    "migration_id": step.id,
                    "phase": "apply",
                    "db_path": db_path,
                    "error_class": e.__class__.__name__,
                    "error_message": str(e),
                    "actionable_suggestion_key": "loom_db_doctor_then_migrate",
                },
            )
            raise MigrationExecutionError(
                migration_id=step.id,
                phase="apply",
                message=str(e),
            ) from e

        try:
            await step.verify(db)
        except Exception as e:
            _emit_diagnostic(
                reporter,
                DB_MIGRATION_VERIFY_FAILED,
                {
                    "migration_id": step.id,
                    "phase": "verify",
                    "db_path": db_path,
                    "error_class": e.__class__.__name__,
                    "error_message": str(e),
                    "actionable_suggestion_key": "loom_db_doctor_then_migrate",
                },
            )
            raise MigrationExecutionError(
                migration_id=step.id,
                phase="verify",
                message=str(e),
            ) from e

        elapsed_ms = max(1, int((time.monotonic() - started) * 1000))
        await db.execute(
            """INSERT INTO schema_migrations (id, applied_at, duration_ms, checksum, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (
                step.id,
                datetime.now(UTC).isoformat(),
                elapsed_ms,
                step.checksum,
                step.description,
            ),
        )
        _emit_diagnostic(
            reporter,
            DB_MIGRATION_APPLIED,
            {
                "migration_id": step.id,
                "db_path": db_path,
                "duration_ms": elapsed_ms,
                "checksum": step.checksum,
            },
        )
        applied_now.append(step.id)

    return applied_now


async def verify_schema(
    db: aiosqlite.Connection,
    *,
    steps: tuple[MigrationStep, ...],
) -> None:
    """Run verification hooks for all registered migration steps."""
    for step in steps:
        await step.verify(db)


async def migration_status(
    db: aiosqlite.Connection,
    *,
    steps: tuple[MigrationStep, ...],
) -> dict[str, object]:
    """Return migration status payload for CLI/reporting."""
    await ensure_migration_table(db)
    applied = await applied_migration_checksums(db)
    all_ids = [step.id for step in steps]
    pending = [mid for mid in all_ids if mid not in applied]
    return {
        "applied_ids": sorted(applied),
        "pending_ids": pending,
        "latest_id": all_ids[-1] if all_ids else "",
    }
