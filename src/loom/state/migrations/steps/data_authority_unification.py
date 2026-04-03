"""Migration step for task/cowork authority checkpoint metadata."""

from __future__ import annotations

from loom.state.migrations.runner import table_columns, table_exists


async def _add_column_if_missing(conn, table_name: str, column_name: str, column_def: str) -> None:
    columns = await table_columns(conn, table_name)
    if column_name in columns:
        return
    await conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


async def apply(conn) -> None:
    """Add freshness/coverage columns used by authority-aware projections."""
    if not await table_exists(conn, "tasks"):
        return

    await _add_column_if_missing(
        conn,
        "tasks",
        "state_snapshot_updated_at",
        "TEXT",
    )
    await conn.execute(
        """
        UPDATE tasks
        SET state_snapshot_updated_at = COALESCE(
            NULLIF(state_snapshot_updated_at, ''),
            NULLIF(updated_at, ''),
            NULLIF(created_at, ''),
            datetime('now')
        )
        WHERE COALESCE(state_snapshot_updated_at, '') = ''
        """,
    )

    if not await table_exists(conn, "cowork_sessions"):
        return

    await _add_column_if_missing(
        conn,
        "cowork_sessions",
        "session_state_through_turn",
        "INTEGER NOT NULL DEFAULT 0",
    )
    await _add_column_if_missing(
        conn,
        "cowork_sessions",
        "chat_journal_through_turn",
        "INTEGER NOT NULL DEFAULT 0",
    )
    await _add_column_if_missing(
        conn,
        "cowork_sessions",
        "chat_journal_through_seq",
        "INTEGER NOT NULL DEFAULT 0",
    )


async def verify(conn) -> None:
    if not await table_exists(conn, "tasks"):
        raise RuntimeError("data authority migration incomplete; tasks table missing.")
    task_columns = await table_columns(conn, "tasks")
    if "state_snapshot_updated_at" not in task_columns:
        raise RuntimeError(
            "data authority migration incomplete; tasks.state_snapshot_updated_at missing.",
        )

    if await table_exists(conn, "cowork_sessions"):
        session_columns = await table_columns(conn, "cowork_sessions")
        for column_name in (
            "session_state_through_turn",
            "chat_journal_through_turn",
            "chat_journal_through_seq",
        ):
            if column_name not in session_columns:
                raise RuntimeError(
                    "data authority migration incomplete; "
                    f"cowork_sessions.{column_name} missing.",
                )
