"""Database management CLI commands."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import click

from loom.cli.context import _effective_config


@click.group(name="db")
def db_group() -> None:
    """Inspect and manage Loom SQLite migrations."""


@db_group.command(name="status")
@click.pass_context
def db_status(ctx: click.Context) -> None:
    """Show migration status for the configured Loom database."""
    from loom.state.migrations import MIGRATIONS, migration_status, verify_schema

    config = _effective_config(ctx, None)
    db_path = config.database_path
    if not db_path.exists():
        click.echo(f"Database: {db_path}")
        click.echo("State: missing (will initialize on first run)")
        return

    async def _run() -> tuple[dict[str, object], str]:
        import aiosqlite

        async with aiosqlite.connect(db_path) as conn:
            payload = await migration_status(conn, steps=MIGRATIONS)
            health = "ok"
            try:
                await verify_schema(conn, steps=MIGRATIONS)
            except Exception as e:
                health = f"failed ({e})"
            return payload, health

    payload, health = asyncio.run(_run())
    applied_ids = list(payload.get("applied_ids", []))
    pending_ids = list(payload.get("pending_ids", []))
    latest_id = str(payload.get("latest_id", "") or "")

    click.echo(f"Database: {db_path}")
    click.echo(f"Latest migration: {latest_id or '(none)'}")
    click.echo(f"Applied: {len(applied_ids)}")
    click.echo(f"Pending: {len(pending_ids)}")
    click.echo(f"Schema health: {health}")
    for mid in pending_ids:
        click.echo(f"  - {mid}")


@db_group.command(name="migrate")
@click.pass_context
def db_migrate(ctx: click.Context) -> None:
    """Apply pending migrations to the configured Loom database."""
    import aiosqlite

    from loom.state.memory import Database
    from loom.state.migrations import MIGRATIONS, migration_status

    config = _effective_config(ctx, None)
    db_path = config.database_path

    async def _run() -> list[tuple[str, int]]:
        pending_before: list[str] = []
        async with aiosqlite.connect(db_path) as conn:
            status_before = await migration_status(conn, steps=MIGRATIONS)
            pending_before = list(status_before.get("pending_ids", []))

        db = Database(str(db_path))
        await db.initialize()
        if not pending_before:
            return []

        placeholders = ",".join("?" for _ in pending_before)
        query = (
            "SELECT id, duration_ms FROM schema_migrations "
            f"WHERE id IN ({placeholders}) ORDER BY id"
        )
        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute(query, tuple(pending_before))
            rows = await cursor.fetchall()
        return [(str(row[0]), int(row[1] or 0)) for row in rows]

    try:
        applied = asyncio.run(_run())
    except Exception as e:
        click.echo(f"Migration failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"Migrations applied successfully for {db_path}")
    if not applied:
        click.echo("No pending migrations.")
        return
    click.echo("Applied migrations:")
    for migration_id, duration_ms in applied:
        click.echo(f"  - {migration_id} ({duration_ms} ms)")


@db_group.command(name="doctor")
@click.pass_context
def db_doctor(ctx: click.Context) -> None:
    """Run migration and schema verification checks."""
    from loom.state.migrations import MIGRATIONS, verify_schema

    config = _effective_config(ctx, None)
    db_path = config.database_path
    if not db_path.exists():
        click.echo(f"Database: {db_path}")
        click.echo("Doctor: database file does not exist yet.")
        return

    async def _run() -> None:
        import aiosqlite

        async with aiosqlite.connect(db_path) as conn:
            await verify_schema(conn, steps=MIGRATIONS)

    try:
        asyncio.run(_run())
    except Exception as e:
        click.echo(f"Doctor failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"Doctor passed: {db_path}")


@db_group.command(name="backup")
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional destination path. Defaults to <db>.backup-<timestamp>.",
)
@click.pass_context
def db_backup(ctx: click.Context, output_path: Path | None) -> None:
    """Create a timestamped backup copy of the Loom database."""
    config = _effective_config(ctx, None)
    db_path = config.database_path
    if not db_path.exists():
        click.echo(f"Database does not exist: {db_path}", err=True)
        sys.exit(1)
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = db_path.with_name(f"{db_path.name}.backup-{stamp}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as source:
            with sqlite3.connect(output_path) as target:
                source.backup(target)
    except Exception as e:
        click.echo(f"Backup failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"Backup created: {output_path}")


def register_db_commands(cli_group: click.Group) -> None:
    cli_group.add_command(db_group)
