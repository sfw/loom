"""Migration step for durable conversation turn attachment metadata."""

from __future__ import annotations

from loom.state.migrations.runner import table_columns, table_exists


async def apply(conn) -> None:
    if not await table_exists(conn, "conversation_turns"):
        return

    columns = await table_columns(conn, "conversation_turns")
    if "metadata" not in columns:
        await conn.execute(
            "ALTER TABLE conversation_turns ADD COLUMN metadata TEXT",
        )


async def verify(conn) -> None:
    if not await table_exists(conn, "conversation_turns"):
        return

    columns = await table_columns(conn, "conversation_turns")
    if "metadata" not in columns:
        raise RuntimeError(
            "conversation turn metadata migration incomplete; "
            "conversation_turns.metadata missing.",
        )
