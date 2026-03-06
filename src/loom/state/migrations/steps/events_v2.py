"""Migration step for enhanced `events` schema."""

from __future__ import annotations

from loom.state.migrations.runner import index_exists, table_columns, table_exists

_EVENTS_REQUIRED_COLUMNS = (
    "task_id",
    "run_id",
    "correlation_id",
    "event_id",
    "sequence",
    "timestamp",
    "event_type",
    "source_component",
    "schema_version",
    "data",
)
_EVENTS_REQUIRED_INDEXES = (
    "idx_events_task_sequence",
    "idx_events_run_sequence",
    "idx_events_event_id",
)


async def apply(conn) -> None:
    """Ensure `events` has required v2 fields + indexes."""
    if not await table_exists(conn, "events"):
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS events (
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
               )""",
        )

    columns = await table_columns(conn, "events")
    migrations = (
        ("run_id", "TEXT NOT NULL DEFAULT ''"),
        ("event_id", "TEXT NOT NULL DEFAULT ''"),
        ("sequence", "INTEGER NOT NULL DEFAULT 0"),
        ("source_component", "TEXT NOT NULL DEFAULT ''"),
        ("schema_version", "INTEGER NOT NULL DEFAULT 1"),
    )
    for column_name, column_def in migrations:
        if column_name in columns:
            continue
        await conn.execute(f"ALTER TABLE events ADD COLUMN {column_name} {column_def}")

    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_task_sequence ON events(task_id, sequence)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_run_sequence ON events(run_id, sequence)",
    )
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_event_id "
        "ON events(event_id) WHERE event_id <> ''",
    )

    # Refresh columns after ALTER TABLE operations.
    columns = await table_columns(conn, "events")
    if "sequence" in columns:
        await conn.execute(
            "UPDATE events SET sequence = 0 WHERE sequence IS NULL",
        )


async def verify(conn) -> None:
    if not await table_exists(conn, "events"):
        raise RuntimeError("events table missing after migration.")

    columns = await table_columns(conn, "events")
    missing = [name for name in _EVENTS_REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise RuntimeError(
            "events migration incomplete; missing columns: " + ", ".join(missing)
        )

    missing_indexes = [
        name for name in _EVENTS_REQUIRED_INDEXES if not await index_exists(conn, name)
    ]
    if missing_indexes:
        raise RuntimeError(
            "events migration incomplete; missing indexes: "
            + ", ".join(missing_indexes)
        )
