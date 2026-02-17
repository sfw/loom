"""CLI entry point for Loom."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from loom import __version__
from loom.config import Config, ConfigError, load_config


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="loom")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to loom.toml configuration file.",
)
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace directory. Defaults to current directory.",
)
@click.option("--model", "-m", default=None, help="Model name from config to use.")
@click.option("--resume", "resume_session", default=None, help="Resume a previous session by ID.")
@click.option(
    "--process", "process_name", default=None,
    help="Process definition name or path.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Path | None,
    workspace: Path | None,
    model: str | None,
    resume_session: str | None,
    process_name: str | None,
) -> None:
    """Loom — Local model orchestration engine.

    When invoked without a subcommand, launches the interactive TUI
    with full session persistence, conversation recall, and task
    delegation.
    """
    ctx.ensure_object(dict)
    try:
        ctx.obj["config"] = load_config(config_path)
    except ConfigError as e:
        if ctx.invoked_subcommand == "setup":
            # Let setup proceed even with broken/missing config
            ctx.obj["config"] = Config()
        else:
            click.echo(f"Configuration error: {e}", err=True)
            sys.exit(1)

    if ctx.invoked_subcommand is None:
        # Default: launch the TUI
        _launch_tui(
            ctx.obj["config"], workspace, model,
            resume_session, process_name,
        )


def _resolve_model(config: Config, model_name: str | None):
    """Resolve a model provider from config."""
    from loom.models.router import ModelRouter

    router = ModelRouter.from_config(config)
    if model_name:
        for name, p in router._providers.items():
            if name == model_name:
                return p
        click.echo(f"Model '{model_name}' not found in config.", err=True)
        sys.exit(1)
    try:
        return router.select(role="executor")
    except Exception as e:
        click.echo(f"No model available: {e}", err=True)
        sys.exit(1)


def _launch_tui(
    config: Config,
    workspace: Path | None,
    model_name: str | None,
    resume_session: str | None,
    process_name: str | None,
) -> None:
    """Launch the Loom TUI with full cowork backend."""
    from loom.tools import create_default_registry
    from loom.tui.app import LoomApp

    ws = (workspace or Path.cwd()).resolve()
    tools = create_default_registry()

    # Resolve model — None triggers the TUI setup wizard
    provider = None
    if config.models:
        provider = _resolve_model(config, model_name)

    # Initialize database and conversation store (fall back to ephemeral)
    db, store = _init_persistence(config)
    if db is None:
        click.echo(
            "Warning: database unavailable, running in ephemeral mode.",
            err=True,
        )
        if resume_session:
            click.echo(
                "Error: --resume requires database. Fix the database path "
                "and retry.",
                err=True,
            )
            sys.exit(1)

    effective_process = process_name or config.process.default or None

    app = LoomApp(
        model=provider,
        tools=tools,
        workspace=ws,
        config=config,
        db=db,
        store=store,
        resume_session=resume_session,
        process_name=effective_process,
    )
    app.run()


def _init_persistence(config: Config):
    """Initialize database and conversation store.

    Returns (db, store) on success, or (None, None) if initialization fails.
    The TUI will fall back to ephemeral mode when store is None.
    """
    from loom.state.conversation_store import ConversationStore
    from loom.state.memory import Database

    try:
        db_path = Path(config.memory.database_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = Database(db_path)

        # Run async init synchronously — each aiosqlite call opens its
        # own connection so there's no state leaking into Textual's loop.
        asyncio.run(db.initialize())

        store = ConversationStore(db)
        return db, store
    except Exception as e:
        click.echo(f"Warning: database init failed: {e}", err=True)
        return None, None


# -- Subcommands that launch the TUI (aliases) ----------------------------

@cli.command()
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace directory. Defaults to current directory.",
)
@click.option("--model", "-m", default=None, help="Model name from config to use.")
@click.option("--resume", "resume_session", default=None, help="Resume a previous session by ID.")
@click.option(
    "--process", "process_name", default=None,
    help="Process definition name or path.",
)
@click.pass_context
def cowork(
    ctx: click.Context,
    workspace: Path | None,
    model: str | None,
    resume_session: str | None,
    process_name: str | None,
) -> None:
    """Start an interactive cowork session (alias for default TUI).

    Opens a conversation loop where you and the AI collaborate directly.
    No planning phase, no subtask decomposition — just a continuous
    tool-calling loop driven by natural conversation.

    All conversation history is persisted to SQLite. Use --resume to
    continue a previous session.
    """
    config = ctx.obj["config"]
    _launch_tui(config, workspace, model, resume_session, process_name)


# -- Server and task commands ----------------------------------------------

@cli.command()
@click.option("--host", default=None, help="Override server host.")
@click.option("--port", default=None, type=int, help="Override server port.")
@click.pass_context
def serve(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Start the Loom API server."""
    config = ctx.obj["config"]
    actual_host = host if host is not None else config.server.host
    actual_port = port if port is not None else config.server.port

    click.echo(f"Starting Loom server on {actual_host}:{actual_port}")

    import uvicorn

    from loom.api.server import create_app

    try:
        app = create_app(config)
        uvicorn.run(
            app, host=actual_host, port=actual_port, log_level="info",
        )
    except Exception as e:
        click.echo(f"Server failed to start: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("goal")
@click.option("--workspace", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.option(
    "--process", "process_name", default=None,
    help="Process definition name or path.",
)
@click.pass_context
def run(
    ctx: click.Context, goal: str, workspace: Path | None,
    server_url: str | None, process_name: str | None,
) -> None:
    """Submit a task and stream progress inline."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"
    ws = str(workspace.resolve()) if workspace else None
    effective_process = process_name or config.process.default or None

    click.echo(f"Submitting task to {url}: {goal}")
    if ws:
        click.echo(f"Workspace: {ws}")
    if effective_process:
        click.echo(f"Process: {effective_process}")

    asyncio.run(_run_task(url, goal, ws, process_name=effective_process))


def _validate_task_id(task_id: str) -> str:
    """Validate task_id contains only safe characters for URL interpolation."""
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
        click.echo(f"Invalid task ID: {task_id}", err=True)
        sys.exit(1)
    return task_id


async def _run_task(
    server_url: str, goal: str, workspace: str | None,
    process_name: str | None = None,
) -> None:
    """Submit task and stream progress."""
    import httpx

    try:
        async with httpx.AsyncClient(base_url=server_url, timeout=300) as client:
            payload: dict = {"goal": goal}
            if workspace:
                payload["workspace"] = workspace
            if process_name:
                payload["process"] = process_name

            response = await client.post("/tasks", json=payload)
            if response.status_code != 201:
                click.echo(f"Error: {response.text}", err=True)
                sys.exit(1)

            task = response.json()
            task_id = task["task_id"]
            click.echo(f"Task created: {task_id}")

            # Stream events
            async with client.stream("GET", f"/tasks/{task_id}/stream") as stream:
                async for line in stream.aiter_lines():
                    if not line.strip() or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        click.echo(line[6:])
    except httpx.ConnectError:
        click.echo(f"Error: Cannot connect to server at {server_url}", err=True)
        sys.exit(1)
    except httpx.TimeoutException:
        click.echo("Error: Request timed out", err=True)
        sys.exit(1)


@cli.command()
@click.argument("task_id")
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.pass_context
def status(ctx: click.Context, task_id: str, server_url: str | None) -> None:
    """Check status of a task."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    asyncio.run(_check_status(url, task_id))


async def _check_status(server_url: str, task_id: str) -> None:
    """Fetch and display task status."""
    task_id = _validate_task_id(task_id)
    import httpx

    try:
        async with httpx.AsyncClient(base_url=server_url) as client:
            response = await client.get(f"/tasks/{task_id}")
            if response.status_code == 404:
                click.echo(f"Task not found: {task_id}", err=True)
                sys.exit(1)
            data = response.json()
            click.echo(f"Task:   {data['task_id']}")
            click.echo(f"Status: {data['status']}")
            click.echo(f"Goal:   {data.get('goal', 'N/A')}")
    except httpx.ConnectError:
        click.echo(
            f"Error: Cannot connect to server at {server_url}",
            err=True,
        )
        sys.exit(1)


@cli.command()
@click.argument("task_id")
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.pass_context
def cancel(ctx: click.Context, task_id: str, server_url: str | None) -> None:
    """Cancel a running task."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    asyncio.run(_cancel_task(url, task_id))


async def _cancel_task(server_url: str, task_id: str) -> None:
    """Cancel a task."""
    task_id = _validate_task_id(task_id)
    import httpx

    try:
        async with httpx.AsyncClient(base_url=server_url) as client:
            response = await client.delete(f"/tasks/{task_id}")
            if response.status_code == 404:
                click.echo(f"Task not found: {task_id}", err=True)
                sys.exit(1)
            if response.status_code >= 400:
                msg = f"Cancel failed ({response.status_code}): {response.text[:200]}"
                click.echo(msg, err=True)
                sys.exit(1)
            click.echo(f"Task {task_id} cancelled.")
    except httpx.ConnectError:
        click.echo(
            f"Error: Cannot connect to server at {server_url}",
            err=True,
        )
        sys.exit(1)


@cli.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """List available models and their status."""
    config = ctx.obj["config"]

    if not config.models:
        click.echo("No models configured. Add model sections to loom.toml.")
        return

    for name, model in config.models.items():
        roles = ", ".join(model.roles)
        click.echo(f"  {name}: {model.model} ({model.provider}) [{roles}]")
        click.echo(f"    URL: {model.base_url}")


@cli.command(name="mcp-serve")
@click.option("--server", "server_url", default=None, help="Loom API server URL.")
@click.pass_context
def mcp_serve(ctx: click.Context, server_url: str | None) -> None:
    """Start Loom as an MCP server (stdio transport)."""
    config = ctx.obj["config"]
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    from loom.integrations.mcp_server import LoomMCPServer

    server = LoomMCPServer(engine_url=url)
    click.echo(f"Starting Loom MCP server (engine: {url})", err=True)
    asyncio.run(server.run_stdio())


# -- Process management commands -------------------------------------------

@cli.command()
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace to search for local process definitions.",
)
@click.pass_context
def processes(ctx: click.Context, workspace: Path | None) -> None:
    """List available process definitions."""
    from loom.processes.schema import ProcessLoader

    config = ctx.obj["config"]
    ws = (workspace or Path.cwd()).resolve()
    extra = [Path(p) for p in config.process.search_paths]
    loader = ProcessLoader(workspace=ws, extra_search_paths=extra)
    available = loader.list_available()

    if not available:
        click.echo("No process definitions found.")
        click.echo("  Built-in: src/loom/processes/builtin/")
        click.echo("  User:     ~/.loom/processes/")
        click.echo("  Local:    ./loom-processes/")
        return

    click.echo("Available processes:\n")
    for proc in available:
        name = proc["name"]
        ver = proc["version"]
        desc = proc.get("description", "")
        # Truncate description to one line
        if desc:
            desc = desc.strip().split("\n")[0][:60]
        click.echo(f"  {name:30s} v{ver:6s} {desc}")
    click.echo(
        f"\n{len(available)} process(es) found. "
        f"Use --process <name> with 'run' or 'cowork'.",
    )


@cli.command(name="install")
@click.argument("source")
@click.option(
    "--workspace", "-w", "install_workspace",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Install to <workspace>/loom-processes/ instead of global ~/.loom/processes/.",
)
@click.option(
    "--skip-deps", is_flag=True, default=False,
    help="Skip installing Python dependencies.",
)
@click.option(
    "--yes", "-y", is_flag=True, default=False,
    help="Skip interactive review and approve automatically.",
)
@click.pass_context
def install(
    ctx: click.Context,
    source: str,
    install_workspace: Path | None,
    skip_deps: bool,
    yes: bool,
) -> None:
    """Install a process package from a GitHub repo or local path.

    SOURCE can be:

    \b
      - A GitHub URL: https://github.com/user/loom-my-process
      - A shorthand:  user/loom-my-process
      - A local path:  /path/to/my-process/

    The package must contain a process.yaml at its root. Python dependencies
    listed in the 'dependencies' field of process.yaml are automatically
    installed (use --skip-deps to disable).

    Before installation, you'll see a full security review of the package
    contents (dependencies, bundled code) and must confirm. Use -y to skip
    this review (not recommended for untrusted sources).

    Examples:

    \b
      loom install https://github.com/acme/loom-google-analytics
      loom install acme/loom-google-analytics
      loom install ./my-local-process
      loom install ./my-local-process -w /path/to/project
    """
    from loom.processes.installer import (
        InstallError,
        format_review_for_terminal,
        install_process,
    )

    if install_workspace:
        target_dir = install_workspace.resolve() / "loom-processes"
    else:
        target_dir = Path.home() / ".loom" / "processes"

    click.echo(f"Resolving source: {source}")

    def _review_and_prompt(review) -> bool:
        """Display review and ask user for confirmation."""
        click.echo(format_review_for_terminal(review))
        if yes:
            click.echo("  --yes flag set: auto-approving.")
            return True
        return click.confirm("  Proceed with installation?", default=False)

    try:
        dest = install_process(
            source,
            target_dir=target_dir,
            skip_deps=skip_deps,
            review_callback=_review_and_prompt,
        )
        click.echo(f"Installed to: {dest}")
        click.echo("Done. Use --process <name> with 'run' or 'cowork'.")
    except InstallError as e:
        click.echo(f"Install failed: {e}", err=True)
        sys.exit(1)


@cli.command(name="uninstall")
@click.argument("name")
@click.option(
    "--workspace", "-w", "uninstall_workspace",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Also search <workspace>/loom-processes/.",
)
@click.confirmation_option(prompt="Are you sure you want to remove this process?")
@click.pass_context
def uninstall(
    ctx: click.Context, name: str, uninstall_workspace: Path | None,
) -> None:
    """Remove an installed process package by name.

    Only removes user-installed processes. Built-in processes cannot be
    removed.
    """
    from loom.processes.installer import UninstallError, uninstall_process

    search_dirs = [Path.home() / ".loom" / "processes"]
    if uninstall_workspace:
        search_dirs.append(uninstall_workspace.resolve() / "loom-processes")

    try:
        removed = uninstall_process(name, search_dirs=search_dirs)
        click.echo(f"Removed: {removed}")
    except UninstallError as e:
        click.echo(f"Uninstall failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--type", "pattern_type", default=None,
    help="Filter by pattern type (e.g., behavioral_gap, behavioral_correction).",
)
@click.option(
    "--delete", "delete_id", type=int, default=None,
    help="Delete a pattern by ID.",
)
@click.option(
    "--limit", default=30, show_default=True,
    help="Max number of patterns to show.",
)
@click.pass_context
def learned(
    ctx: click.Context,
    pattern_type: str | None,
    delete_id: int | None,
    limit: int,
) -> None:
    """Review learned patterns.

    Lists all patterns the learning system has extracted from your
    interactions.  Use --delete ID to remove a specific pattern.

    \b
    Examples:
      loom learned                              # list all
      loom learned --type behavioral_gap        # filter by type
      loom learned --delete 5                   # delete pattern #5
    """
    from loom.learning.manager import LearningManager
    from loom.state.memory import Database

    config = ctx.obj["config"]

    async def _run():
        db = Database(str(Path(config.memory.database_path).expanduser()))
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
                pattern_type=pattern_type, limit=limit,
            )
        else:
            patterns = await mgr.query_all(limit=limit)

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


@cli.command(name="reset-learning")
@click.confirmation_option(prompt="Are you sure you want to clear all learned patterns?")
@click.pass_context
def reset_learning(ctx: click.Context) -> None:
    """Clear all learned patterns from the database."""
    from loom.learning.manager import LearningManager
    from loom.state.memory import Database

    config = ctx.obj["config"]

    async def _reset():
        db = Database(str(Path(config.memory.database_path).expanduser()))
        await db.initialize()
        manager = LearningManager(db)
        await manager.clear_all()
        click.echo("Learning database cleared.")

    asyncio.run(_reset())


@cli.command()
def setup() -> None:
    """Run the interactive configuration wizard.

    Creates or overwrites ~/.loom/loom.toml with provider settings
    collected via guided prompts.  Automatically triggered on first
    run if no configuration file exists.
    """
    from loom.setup import run_setup

    run_setup(reconfigure=True)


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
