"""Migration step for workspace registry tables."""

from __future__ import annotations

from loom.state.migrations.runner import index_exists, table_columns, table_exists

_REQUIRED_TABLES = (
    "workspaces",
    "workspace_settings",
    "conversation_run_links",
)
_REQUIRED_INDEXES = (
    "idx_workspaces_path",
    "idx_workspaces_archived_order",
    "idx_crl_session_run_type",
    "idx_crl_run",
)


async def apply(conn) -> None:
    """Add workspace registry/settings/link tables."""
    if not await table_exists(conn, "workspaces"):
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                canonical_path TEXT NOT NULL,
                display_name TEXT NOT NULL,
                workspace_type TEXT NOT NULL DEFAULT 'local',
                sort_order INTEGER NOT NULL DEFAULT 0,
                last_opened_at TEXT,
                is_archived INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """,
        )
    if not await table_exists(conn, "workspace_settings"):
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_settings (
                workspace_id TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
            )
            """,
        )
    if not await table_exists(conn, "conversation_run_links"):
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_run_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                link_type TEXT NOT NULL DEFAULT 'origin',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES cowork_sessions(id),
                FOREIGN KEY (run_id) REFERENCES tasks(id)
            )
            """,
        )

    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_workspaces_path ON workspaces(canonical_path)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_workspaces_archived_order "
        "ON workspaces(is_archived, sort_order, display_name)",
    )
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_crl_session_run_type "
        "ON conversation_run_links(session_id, run_id, link_type)",
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_crl_run ON conversation_run_links(run_id)",
    )


async def verify(conn) -> None:
    missing_tables = [name for name in _REQUIRED_TABLES if not await table_exists(conn, name)]
    if missing_tables:
        raise RuntimeError(
            "workspace migration incomplete; missing tables: " + ", ".join(missing_tables)
        )

    workspace_columns = await table_columns(conn, "workspaces")
    for required in (
        "id",
        "canonical_path",
        "display_name",
        "workspace_type",
        "sort_order",
        "is_archived",
        "metadata",
        "created_at",
        "updated_at",
    ):
        if required not in workspace_columns:
            raise RuntimeError(f"workspaces migration incomplete; missing column {required}.")

    missing_indexes = [
        name for name in _REQUIRED_INDEXES if not await index_exists(conn, name)
    ]
    if missing_indexes:
        raise RuntimeError(
            "workspace migration incomplete; missing indexes: "
            + ", ".join(missing_indexes)
        )
