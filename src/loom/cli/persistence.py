"""CLI persistence bootstrap helpers."""

from __future__ import annotations

import asyncio

import click

from loom.config import Config


class PersistenceInitError(RuntimeError):
    """Raised when an existing Loom database cannot be initialized safely."""


def _init_persistence(config: Config, *, allow_ephemeral: bool = False):
    """Initialize database and conversation store.

    Returns (db, store) on success, or (None, None) when creating a new DB
    fails and ephemeral fallback is explicitly enabled.

    Raises:
        PersistenceInitError: Existing DB file could not be initialized. This
        is treated as a blocking upgrade failure.
    """
    from loom.state.conversation_store import ConversationStore
    from loom.state.memory import Database
    from loom.state.migrations import MigrationExecutionError

    db_path = config.database_path
    existing_db = False
    try:
        existing_db = (
            db_path.exists()
            and db_path.is_file()
            and db_path.stat().st_size > 0
        )
    except Exception:
        existing_db = False

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)

        # Run async init synchronously — each aiosqlite call opens its
        # own connection so there's no state leaking into Textual's loop.
        asyncio.run(db.initialize())

        store = ConversationStore(db)
        return db, store
    except Exception as e:
        click.echo(f"Warning: database init failed: {e}", err=True)
        if existing_db:
            if isinstance(e, MigrationExecutionError):
                raise PersistenceInitError(
                    "Existing Loom database migration failed "
                    f"({e.migration_id}/{e.phase}): {e}. "
                    "Run `loom db doctor` and `loom db migrate`, then retry."
                ) from e
            raise PersistenceInitError(
                "Existing Loom database could not be upgraded. "
                "Run `loom db doctor` and `loom db migrate`, then retry."
            ) from e
        if not allow_ephemeral:
            raise PersistenceInitError(
                "Database initialization failed for a new database path. "
                "Fix filesystem permissions/path, or rerun with `--ephemeral` "
                "to continue without persistence."
            ) from e
        return None, None
