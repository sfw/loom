"""CLI entry point for Loom."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from loom import __version__
from loom.config import Config, ConfigError, load_config
from loom.mcp.config import (
    MCPConfigManager,
    MCPConfigManagerError,
    MCPServerView,
    apply_mcp_overrides,
    ensure_valid_alias,
    merge_server_edits,
    parse_mcp_server_from_flags,
    redact_server_env,
)


def _resolve_config_path(config_path: Path | None) -> Path | None:
    """Resolve the effective loom.toml path used for config loading."""
    if config_path is not None:
        return config_path
    candidates = [
        Path.cwd() / "loom.toml",
        Path.home() / ".loom" / "loom.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_workspace(workspace: Path | None) -> Path:
    return (workspace or Path.cwd()).resolve()


def _apply_mcp_layers(
    *,
    base_config: Config,
    workspace: Path | None,
    explicit_mcp_path: Path | None,
    legacy_config_path: Path | None,
) -> Config:
    """Apply merged MCP config layers to an already-loaded base config."""
    return apply_mcp_overrides(
        base_config,
        workspace=workspace,
        explicit_path=explicit_mcp_path,
        legacy_config_path=legacy_config_path,
    )


def _effective_config(ctx: click.Context, workspace: Path | None = None) -> Config:
    """Return config with MCP layering using the provided workspace."""
    base = ctx.obj["base_config"]
    explicit_mcp = ctx.obj.get("explicit_mcp_path")
    legacy_cfg = ctx.obj.get("config_path")
    active_workspace = workspace or ctx.obj.get("workspace")
    return _apply_mcp_layers(
        base_config=base,
        workspace=active_workspace,
        explicit_mcp_path=explicit_mcp,
        legacy_config_path=legacy_cfg,
    )


def _mcp_manager(ctx: click.Context, workspace: Path | None = None) -> MCPConfigManager:
    """Build shared MCP config manager from CLI context."""
    active_workspace = workspace or ctx.obj.get("workspace")
    return MCPConfigManager(
        config=ctx.obj["base_config"],
        workspace=active_workspace,
        explicit_path=ctx.obj.get("explicit_mcp_path"),
        legacy_config_path=ctx.obj.get("config_path"),
    )


def _serialize_mcp_view(
    view: MCPServerView,
    *,
    redacted: bool = True,
) -> dict:
    env = redact_server_env(view.server) if redacted else dict(view.server.env)
    payload = {
        "alias": view.alias,
        "source": view.source,
        "source_path": str(view.source_path) if view.source_path else None,
        "command": view.server.command,
        "args": list(view.server.args),
        "cwd": view.server.cwd,
        "timeout_seconds": view.server.timeout_seconds,
        "enabled": view.server.enabled,
        "env": env,
    }
    return payload


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
@click.option(
    "--mcp-config",
    "mcp_config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to mcp.toml (highest precedence MCP config layer).",
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
    mcp_config_path: Path | None,
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
    resolved_config_path = _resolve_config_path(config_path)
    try:
        base = load_config(resolved_config_path)
        effective = _apply_mcp_layers(
            base_config=base,
            workspace=workspace,
            explicit_mcp_path=mcp_config_path,
            legacy_config_path=resolved_config_path,
        )
        ctx.obj["base_config"] = base
        ctx.obj["config"] = effective
        ctx.obj["config_path"] = resolved_config_path
        ctx.obj["workspace"] = _resolve_workspace(workspace)
        ctx.obj["explicit_mcp_path"] = (
            mcp_config_path.expanduser().resolve() if mcp_config_path else None
        )
    except ConfigError as e:
        if ctx.invoked_subcommand == "setup":
            # Let setup proceed even with broken/missing config
            ctx.obj["config"] = Config()
            ctx.obj["base_config"] = ctx.obj["config"]
            ctx.obj["config_path"] = resolved_config_path
            ctx.obj["workspace"] = _resolve_workspace(workspace)
            ctx.obj["explicit_mcp_path"] = (
                mcp_config_path.expanduser().resolve() if mcp_config_path else None
            )
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
    tools = create_default_registry(config)

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
    config = _effective_config(ctx, workspace)
    _launch_tui(config, workspace, model, resume_session, process_name)


# -- Server and task commands ----------------------------------------------

@cli.command()
@click.option("--host", default=None, help="Override server host.")
@click.option("--port", default=None, type=int, help="Override server port.")
@click.pass_context
def serve(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Start the Loom API server."""
    config = _effective_config(ctx, None)
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
    config = _effective_config(ctx, workspace)
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
    config = _effective_config(ctx, None)
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
    config = _effective_config(ctx, None)
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
    config = _effective_config(ctx, None)

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
    config = _effective_config(ctx, None)
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    from loom.integrations.mcp_server import LoomMCPServer

    server = LoomMCPServer(engine_url=url)
    click.echo(f"Starting Loom MCP server (engine: {url})", err=True)
    asyncio.run(server.run_stdio())


# -- MCP configuration management commands --------------------------------

@cli.group()
def mcp() -> None:
    """Manage external MCP server configuration."""


@mcp.command(name="list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Include expanded command/env/source details.",
)
@click.pass_context
def mcp_list(ctx: click.Context, as_json: bool, verbose: bool) -> None:
    """List merged MCP server configuration."""
    manager = _mcp_manager(ctx)
    try:
        merged = manager.load()
        views = merged.as_views()
    except MCPConfigManagerError as e:
        click.echo(f"List failed: {e}", err=True)
        sys.exit(1)
    legacy_present = any(view.source == "legacy" for view in views)

    if as_json:
        payload = {
            "servers": [
                _serialize_mcp_view(view, redacted=True)
                for view in views
            ],
            "legacy_sources_detected": legacy_present,
        }
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not views:
        click.echo("No MCP servers configured.")
        return

    click.echo("MCP servers:")
    for view in views:
        status = "enabled" if view.server.enabled else "disabled"
        source_path = str(view.source_path) if view.source_path else "-"
        click.echo(f"  {view.alias:16} {status:8} source={view.source}")
        if verbose:
            args = " ".join(view.server.args)
            cmd = f"{view.server.command} {args}".strip()
            click.echo(f"    path: {source_path}")
            click.echo(f"    command: {cmd}")
            click.echo(f"    cwd: {view.server.cwd or '-'}")
            click.echo(f"    timeout_seconds: {view.server.timeout_seconds}")
            env = redact_server_env(view.server)
            if env:
                click.echo("    env:")
                for key, value in env.items():
                    click.echo(f"      {key}={value}")
            else:
                click.echo("    env: (none)")

    if legacy_present:
        click.echo(
            "\nLegacy MCP config detected in loom.toml. "
            "Run `loom mcp migrate` to move it into mcp.toml."
        )


@mcp.command(name="show")
@click.argument("alias")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def mcp_show(ctx: click.Context, alias: str, as_json: bool) -> None:
    """Show one merged MCP server configuration."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
        view = manager.get_view(clean_alias)
    except MCPConfigManagerError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    if view is None:
        click.echo(f"MCP server not found: {clean_alias}", err=True)
        sys.exit(1)

    payload = _serialize_mcp_view(view, redacted=True)
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Alias: {view.alias}")
    click.echo(f"Source: {view.source}")
    click.echo(f"Source path: {view.source_path or '-'}")
    click.echo(f"Enabled: {view.server.enabled}")
    click.echo(f"Command: {view.server.command}")
    click.echo(f"Args: {' '.join(view.server.args) if view.server.args else '-'}")
    click.echo(f"Cwd: {view.server.cwd or '-'}")
    click.echo(f"Timeout: {view.server.timeout_seconds}s")
    env = redact_server_env(view.server)
    if env:
        click.echo("Env:")
        for key, value in env.items():
            click.echo(f"  {key}={value}")
    else:
        click.echo("Env: (none)")
    if view.source == "legacy":
        click.echo(
            "Note: this alias is from legacy loom.toml. "
            "Run `loom mcp migrate` to move it into mcp.toml."
        )


@mcp.command(name="add")
@click.argument("alias")
@click.option("--command", "command", required=True, help="Server command binary.")
@click.option("--arg", "args", multiple=True, help="Server command argument.")
@click.option("--env", "env_pairs", multiple=True, help="Literal env KEY=VALUE.")
@click.option(
    "--env-ref",
    "env_refs",
    multiple=True,
    help="Env indirection KEY=ENV_VAR (stored as ${ENV_VAR}).",
)
@click.option("--cwd", default="", help="Working directory for server process.")
@click.option("--timeout", type=int, default=30, show_default=True)
@click.option("--disabled", is_flag=True, default=False, help="Add server as disabled.")
@click.pass_context
def mcp_add(
    ctx: click.Context,
    alias: str,
    command: str,
    args: tuple[str, ...],
    env_pairs: tuple[str, ...],
    env_refs: tuple[str, ...],
    cwd: str,
    timeout: int,
    disabled: bool,
) -> None:
    """Add a new MCP server config entry."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
        server = parse_mcp_server_from_flags(
            command=command,
            args=args,
            env_pairs=env_pairs,
            env_refs=env_refs,
            cwd=cwd,
            timeout=timeout,
            disabled=disabled,
        )
        target = manager.add_server(clean_alias, server)
    except MCPConfigManagerError as e:
        click.echo(f"Add failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Added MCP server '{clean_alias}' to {target}")


@mcp.command(name="edit")
@click.argument("alias")
@click.option("--command", "command", default=None, help="Server command binary.")
@click.option("--arg", "args", multiple=True, help="Server command argument.")
@click.option("--env", "env_pairs", multiple=True, help="Literal env KEY=VALUE.")
@click.option(
    "--env-ref",
    "env_refs",
    multiple=True,
    help="Env indirection KEY=ENV_VAR (stored as ${ENV_VAR}).",
)
@click.option("--cwd", default=None, help="Working directory for server process.")
@click.option("--timeout", type=int, default=None, help="Tool timeout seconds.")
@click.option("--disabled", is_flag=True, default=False, help="Disable server.")
@click.pass_context
def mcp_edit(
    ctx: click.Context,
    alias: str,
    command: str | None,
    args: tuple[str, ...],
    env_pairs: tuple[str, ...],
    env_refs: tuple[str, ...],
    cwd: str | None,
    timeout: int | None,
    disabled: bool,
) -> None:
    """Edit an existing MCP server config entry."""
    manager = _mcp_manager(ctx)
    if (
        command is None
        and not args
        and not env_pairs
        and not env_refs
        and cwd is None
        and timeout is None
        and not disabled
    ):
        click.echo("No edit flags provided.", err=True)
        sys.exit(1)

    try:
        clean_alias = ensure_valid_alias(alias)
        updated_path, _updated = manager.edit_server(
            clean_alias,
            lambda current: merge_server_edits(
                current=current,
                command=command,
                args=args,
                env_pairs=env_pairs,
                env_refs=env_refs,
                cwd=cwd,
                timeout=timeout,
                disabled=disabled,
            ),
        )
    except MCPConfigManagerError as e:
        click.echo(f"Edit failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Updated MCP server '{clean_alias}' in {updated_path}")


@mcp.command(name="remove")
@click.argument("alias")
@click.pass_context
def mcp_remove(ctx: click.Context, alias: str) -> None:
    """Remove an MCP server config entry."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
        path = manager.remove_server(clean_alias)
    except MCPConfigManagerError as e:
        click.echo(f"Remove failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"Removed MCP server '{clean_alias}' from {path}")


@mcp.command(name="enable")
@click.argument("alias")
@click.pass_context
def mcp_enable(ctx: click.Context, alias: str) -> None:
    """Enable an MCP server."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
        path, _updated = manager.edit_server(
            clean_alias,
            lambda current: current.__class__(
                command=current.command,
                args=list(current.args),
                env=dict(current.env),
                cwd=current.cwd,
                timeout_seconds=current.timeout_seconds,
                enabled=True,
            ),
        )
    except MCPConfigManagerError as e:
        click.echo(f"Enable failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"Enabled MCP server '{clean_alias}' in {path}")


@mcp.command(name="disable")
@click.argument("alias")
@click.pass_context
def mcp_disable(ctx: click.Context, alias: str) -> None:
    """Disable an MCP server."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
        path, _updated = manager.edit_server(
            clean_alias,
            lambda current: current.__class__(
                command=current.command,
                args=list(current.args),
                env=dict(current.env),
                cwd=current.cwd,
                timeout_seconds=current.timeout_seconds,
                enabled=False,
            ),
        )
    except MCPConfigManagerError as e:
        click.echo(f"Disable failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"Disabled MCP server '{clean_alias}' in {path}")


@mcp.command(name="test")
@click.argument("alias")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def mcp_test(ctx: click.Context, alias: str, as_json: bool) -> None:
    """Probe MCP server connectivity and tools/list behavior."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
    except MCPConfigManagerError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    try:
        view, tools = manager.probe_server(clean_alias)
        if view is None:
            click.echo(f"MCP server not found: {clean_alias}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"MCP probe failed for '{clean_alias}': {e}", err=True)
        sys.exit(1)

    tool_names = [str(tool.get("name", "")) for tool in tools]
    payload = {
        "alias": view.alias,
        "source": view.source,
        "source_path": str(view.source_path) if view.source_path else None,
        "tool_count": len(tool_names),
        "tools": tool_names,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(
        f"MCP probe succeeded for '{view.alias}' "
        f"({len(tool_names)} tool(s) discovered)."
    )
    if tool_names:
        for tool_name in tool_names:
            click.echo(f"  - {tool_name}")
    else:
        click.echo("  (no tools returned)")


@mcp.command(name="migrate")
@click.pass_context
def mcp_migrate(ctx: click.Context) -> None:
    """Move legacy `[mcp]` config from loom.toml into mcp.toml."""
    manager = _mcp_manager(ctx)
    try:
        target, copied, removed = manager.migrate_legacy()
    except MCPConfigManagerError as e:
        click.echo(f"Migrate failed: {e}", err=True)
        sys.exit(1)

    if copied == 0:
        click.echo("No legacy MCP config entries found to migrate.")
        return

    click.echo(f"Migrated {copied} MCP server(s) into {target}")
    if removed:
        click.echo("Removed legacy [mcp] sections from loom.toml.")
    else:
        click.echo(
            "Legacy mcp sections were not removed automatically. "
            "Review loom.toml if needed."
        )


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

    config = _effective_config(ctx, workspace)
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


@cli.group()
def process() -> None:
    """Process subcommands."""


@process.command(name="test")
@click.argument("name_or_path")
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace for process execution and local process discovery.",
)
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Include live test cases from process.yaml (requires configured models).",
)
@click.option(
    "--case",
    "case_id",
    default=None,
    help="Run a single process test case by ID.",
)
@click.pass_context
def process_test(
    ctx: click.Context,
    name_or_path: str,
    workspace: Path | None,
    live: bool,
    case_id: str | None,
) -> None:
    """Run declared (or default) process test cases.

    NAME_OR_PATH can be either a process name from discovery or a direct
    path to a process YAML/package directory.
    """
    from loom.processes.schema import ProcessLoader
    from loom.processes.testing import run_process_tests

    ws = (workspace or Path.cwd()).resolve()
    config = _effective_config(ctx, ws)
    extra = [Path(p) for p in config.process.search_paths]
    loader = ProcessLoader(workspace=ws, extra_search_paths=extra)

    try:
        process_def = loader.load(name_or_path)
    except Exception as e:
        click.echo(f"Failed to load process {name_or_path!r}: {e}", err=True)
        sys.exit(1)

    click.echo(
        f"Running process tests for {process_def.name} v{process_def.version}"
    )

    try:
        results = asyncio.run(run_process_tests(
            process_def,
            config=config,
            workspace=ws,
            include_live=live,
            case_id=case_id,
        ))
    except ValueError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    if not results:
        click.echo("No matching process test cases selected.")
        sys.exit(1)

    failed = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        click.echo(
            f"[{status}] case={result.case_id} mode={result.mode} "
            f"task_status={result.task_status or 'n/a'} "
            f"duration={result.duration_seconds:.2f}s"
        )
        if result.message:
            click.echo(f"  {result.message}")
        for detail in result.details:
            click.echo(f"  - {detail}")
        if not result.passed:
            failed += 1

    click.echo(f"\n{len(results) - failed}/{len(results)} case(s) passed.")
    if failed:
        sys.exit(1)


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
    "--isolated-deps", is_flag=True, default=False,
    help=(
        "Install package dependencies into <target>/.deps/<process-name>/ "
        "instead of the current Python environment."
    ),
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
    isolated_deps: bool,
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
      loom install ./my-local-process --isolated-deps
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
    if isolated_deps and not skip_deps:
        click.echo(
            "Dependency mode: isolated "
            "(per-process env under <target>/.deps/...)"
        )
    elif isolated_deps and skip_deps:
        click.echo(
            "Note: --isolated-deps has no effect when --skip-deps is set."
        )

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
            isolated_deps=isolated_deps,
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
@click.option(
    "--all", "include_all", is_flag=True, default=False,
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


@cli.command(name="reset-learning")
@click.confirmation_option(prompt="Are you sure you want to clear all learned patterns?")
@click.pass_context
def reset_learning(ctx: click.Context) -> None:
    """Clear all learned patterns from the database."""
    from loom.learning.manager import LearningManager
    from loom.state.memory import Database

    config = _effective_config(ctx, None)

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
