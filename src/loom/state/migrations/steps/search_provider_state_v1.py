"""Migration step for authoritative auth-free search provider pacing state."""

from __future__ import annotations

from loom.state.migrations.runner import index_exists, table_columns, table_exists

_REQUIRED_COLUMNS = {
    "provider",
    "enabled",
    "priority",
    "min_interval_seconds",
    "next_allowed_at",
    "cooldown_until",
    "lease_owner",
    "lease_expires_at",
    "consecutive_failures",
    "soft_block_count",
    "last_status_code",
    "last_started_at",
    "last_finished_at",
    "last_success_at",
    "updated_at",
}
_REQUIRED_INDEXES = (
    "idx_search_provider_enabled_priority",
    "idx_search_provider_retry_windows",
)


async def apply(conn) -> None:
    await conn.execute(
        """CREATE TABLE IF NOT EXISTS search_provider_state (
               provider TEXT PRIMARY KEY,
               enabled INTEGER NOT NULL DEFAULT 1,
               priority INTEGER NOT NULL DEFAULT 0,
               min_interval_seconds REAL NOT NULL DEFAULT 0,
               next_allowed_at REAL NOT NULL DEFAULT 0,
               cooldown_until REAL NOT NULL DEFAULT 0,
               lease_owner TEXT NOT NULL DEFAULT '',
               lease_expires_at REAL NOT NULL DEFAULT 0,
               consecutive_failures INTEGER NOT NULL DEFAULT 0,
               soft_block_count INTEGER NOT NULL DEFAULT 0,
               last_status_code INTEGER,
               last_started_at REAL NOT NULL DEFAULT 0,
               last_finished_at REAL NOT NULL DEFAULT 0,
               last_success_at REAL NOT NULL DEFAULT 0,
               updated_at REAL NOT NULL DEFAULT 0
           )""",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_search_provider_enabled_priority "
        "ON search_provider_state(enabled, priority DESC, provider)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_search_provider_retry_windows "
        "ON search_provider_state(cooldown_until, next_allowed_at, lease_expires_at)",
    )


async def verify(conn) -> None:
    if not await table_exists(conn, "search_provider_state"):
        raise RuntimeError("search provider state migration incomplete; table missing.")
    columns = await table_columns(conn, "search_provider_state")
    missing_columns = sorted(_REQUIRED_COLUMNS - columns)
    if missing_columns:
        raise RuntimeError(
            "search provider state migration incomplete; missing columns: "
            + ", ".join(missing_columns),
        )
    missing_indexes = [
        name
        for name in _REQUIRED_INDEXES
        if not await index_exists(conn, name)
    ]
    if missing_indexes:
        raise RuntimeError(
            "search provider state migration incomplete; missing indexes: "
            + ", ".join(missing_indexes),
        )
