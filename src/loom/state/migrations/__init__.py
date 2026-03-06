"""Schema migration entrypoints."""

from loom.state.migrations.registry import MIGRATIONS
from loom.state.migrations.runner import (
    MigrationExecutionError,
    apply_pending_migrations,
    ensure_migration_table,
    has_user_tables,
    migration_status,
    verify_schema,
)

__all__ = [
    "MIGRATIONS",
    "MigrationExecutionError",
    "apply_pending_migrations",
    "ensure_migration_table",
    "has_user_tables",
    "migration_status",
    "verify_schema",
]
