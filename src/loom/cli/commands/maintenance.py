"""Maintenance and setup CLI commands."""

from __future__ import annotations

import asyncio

import click

from loom.cli.context import _effective_config


@click.command()
@click.option(
    "--type",
    "pattern_type",
    default=None,
    help="Filter by pattern type (e.g., behavioral_gap, behavioral_correction).",
)
@click.option(
    "--delete",
    "delete_id",
    type=int,
    default=None,
    help="Delete a pattern by ID.",
)
@click.option(
    "--limit",
    default=30,
    show_default=True,
    help="Max number of patterns to show.",
)
@click.option(
    "--all",
    "include_all",
    is_flag=True,
    default=False,
    help="Include internal operational patterns (task templates, retries, failures).",
)
@click.pass_context
def learned(
    ctx: click.Context,
    pattern_type: str | None,
    delete_id: int | None,
    limit: int,
    include_all: bool,
) -> None:
    """Review learned patterns.

    By default, shows learned behavioral patterns used to personalize
    cowork interactions. Use --all to include internal operational
    patterns from autonomous task execution.

    Use --delete ID to remove a specific pattern.

    \b
    Examples:
      loom learned                              # list behavioral patterns
      loom learned --all                        # include internal patterns
      loom learned --type behavioral_gap        # filter by type
      loom learned --delete 5                   # delete pattern #5
    """
    from loom.learning.manager import LearningManager
    from loom.state.memory import Database

    config = _effective_config(ctx, None)

    async def _run():
        db = Database(str(config.database_path))
        await db.initialize()
        mgr = LearningManager(db)

        if delete_id is not None:
            deleted = await mgr.delete_pattern(delete_id)
            if deleted:
                click.echo(f"Deleted pattern {delete_id}.")
            else:
                click.echo(f"Pattern {delete_id} not found.", err=True)
            return

        if pattern_type:
            patterns = await mgr.query_patterns(
                pattern_type=pattern_type,
                limit=limit,
            )
        elif include_all:
            patterns = await mgr.query_all(limit=limit)
        else:
            patterns = await mgr.query_behavioral(limit=limit)

        if not patterns:
            click.echo("No learned patterns.")
            return

        click.echo(f"{'ID':>4}  {'Type':<24} {'Freq':>4}  {'Last Seen':<12} Description")
        click.echo("-" * 80)
        for p in patterns:
            desc = p.data.get("description", p.pattern_key)[:40]
            ptype = p.pattern_type
            last = p.last_seen[:10] if p.last_seen else ""
            click.echo(f"{p.id:>4}  {ptype:<24} {p.frequency:>4}  {last:<12} {desc}")

        click.echo(f"\n{len(patterns)} pattern(s). Use --delete ID to remove one.")

    asyncio.run(_run())


@click.command(name="reset-learning")
@click.confirmation_option(prompt="Are you sure you want to clear all learned patterns?")
@click.pass_context
def reset_learning(ctx: click.Context) -> None:
    """Clear all learned patterns from the database."""
    from loom.learning.manager import LearningManager
    from loom.state.memory import Database

    config = _effective_config(ctx, None)

    async def _reset():
        db = Database(str(config.database_path))
        await db.initialize()
        manager = LearningManager(db)
        await manager.clear_all()
        click.echo("Learning database cleared.")

    asyncio.run(_reset())


@click.command()
def setup() -> None:
    """Run the interactive configuration wizard.

    Creates or overwrites ~/.loom/loom.toml with provider settings
    collected via guided prompts. Automatically triggered on first
    run if no configuration file exists.
    """
    from loom.setup import run_setup

    run_setup(reconfigure=True)


def register_maintenance_commands(cli_group: click.Group) -> None:
    cli_group.add_command(learned)
    cli_group.add_command(reset_learning)
    cli_group.add_command(setup)
