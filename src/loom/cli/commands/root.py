"""Root CLI group and top-level commands."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from loom import __version__
from loom.auth.runtime import AuthResolutionError, parse_auth_profile_overrides
from loom.cli.context import (
    _apply_mcp_layers,
    _effective_config,
    _resolve_config_path,
    _resolve_workspace,
)
from loom.cli.http_tasks import _cancel_task, _check_status, _run_task
from loom.cli.persistence import PersistenceInitError, _init_persistence
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
    "--workspace",
    "-w",
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
@click.option(
    "--auth-config",
    "auth_config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to auth.toml (overlays ~/.loom/auth.toml).",
)
@click.option("--model", "-m", default=None, help="Model name from config to use.")
@click.option("--resume", "resume_session", default=None, help="Resume a previous session by ID.")
@click.option(
    "--ephemeral",
    "allow_ephemeral",
    is_flag=True,
    default=False,
    help="Allow startup without SQLite persistence when DB initialization fails.",
)
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Path | None,
    workspace: Path | None,
    mcp_config_path: Path | None,
    auth_config_path: Path | None,
    model: str | None,
    resume_session: str | None,
    allow_ephemeral: bool,
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
        ctx.obj["explicit_auth_path"] = (
            auth_config_path.expanduser().resolve() if auth_config_path else None
        )
        ctx.obj["allow_ephemeral"] = bool(allow_ephemeral)
    except ConfigError as e:
        if ctx.invoked_subcommand == "setup":
            # Let setup proceed even with broken/missing config.
            ctx.obj["config"] = Config()
            ctx.obj["base_config"] = ctx.obj["config"]
            ctx.obj["config_path"] = resolved_config_path
            ctx.obj["workspace"] = _resolve_workspace(workspace)
            ctx.obj["explicit_mcp_path"] = (
                mcp_config_path.expanduser().resolve() if mcp_config_path else None
            )
            ctx.obj["explicit_auth_path"] = (
                auth_config_path.expanduser().resolve() if auth_config_path else None
            )
            ctx.obj["allow_ephemeral"] = bool(allow_ephemeral)
        else:
            click.echo(f"Configuration error: {e}", err=True)
            sys.exit(1)

    if ctx.invoked_subcommand is None:
        _launch_tui(
            ctx.obj["config"],
            workspace,
            model,
            resume_session,
            ctx.obj.get("explicit_mcp_path"),
            ctx.obj.get("config_path"),
            ctx.obj.get("explicit_auth_path"),
            ctx.obj.get("allow_ephemeral", False),
        )


def _resolve_model(config: Config, model_name: str | None):
    """Resolve a model provider from config."""
    from loom.models.router import ModelRouter

    router = ModelRouter.from_config(config)
    if model_name:
        for name, provider in router._providers.items():
            if name == model_name:
                return provider
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
    explicit_mcp_path: Path | None = None,
    legacy_config_path: Path | None = None,
    explicit_auth_path: Path | None = None,
    allow_ephemeral: bool = False,
) -> None:
    """Launch the Loom TUI with full cowork backend."""
    from loom.tools import create_default_registry
    from loom.tui.app import LoomApp

    ws = (workspace or Path.cwd()).resolve()
    tools = create_default_registry(config, mcp_startup_mode="background")

    provider = None
    if config.models:
        provider = _resolve_model(config, model_name)

    try:
        db, store = _init_persistence(config, allow_ephemeral=allow_ephemeral)
    except PersistenceInitError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    if db is None:
        click.echo(
            "Warning: database unavailable, running in ephemeral mode.",
            err=True,
        )
        if resume_session:
            click.echo(
                "Error: --resume requires database. Fix the database path and retry.",
                err=True,
            )
            sys.exit(1)

    app = LoomApp(
        model=provider,
        tools=tools,
        workspace=ws,
        config=config,
        db=db,
        store=store,
        resume_session=resume_session,
        explicit_mcp_path=explicit_mcp_path,
        legacy_config_path=legacy_config_path,
        explicit_auth_path=explicit_auth_path,
    )
    # Explicitly enable mouse support so click/scroll interactions stay
    # available even if Textual changes defaults across versions.
    app.run(mouse=True)


@cli.command()
@click.option(
    "--workspace",
    "-w",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Workspace directory. Defaults to current directory.",
)
@click.option("--model", "-m", default=None, help="Model name from config to use.")
@click.option("--resume", "resume_session", default=None, help="Resume a previous session by ID.")
@click.option(
    "--ephemeral",
    "allow_ephemeral",
    is_flag=True,
    default=False,
    help="Allow startup without SQLite persistence when DB initialization fails.",
)
@click.pass_context
def cowork(
    ctx: click.Context,
    workspace: Path | None,
    model: str | None,
    resume_session: str | None,
    allow_ephemeral: bool,
) -> None:
    """Start an interactive cowork session (alias for default TUI).

    Opens a conversation loop where you and the AI collaborate directly.
    No planning phase, no subtask decomposition — just a continuous
    tool-calling loop driven by natural conversation.

    All conversation history is persisted to SQLite. Use --resume to
    continue a previous session.
    """
    config = _effective_config(ctx, workspace)
    _launch_tui(
        config,
        workspace,
        model,
        resume_session,
        ctx.obj.get("explicit_mcp_path"),
        ctx.obj.get("config_path"),
        ctx.obj.get("explicit_auth_path"),
        bool(allow_ephemeral or ctx.obj.get("allow_ephemeral", False)),
    )


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
            app,
            host=actual_host,
            port=actual_port,
            log_level="info",
        )
    except Exception as e:
        click.echo(f"Server failed to start: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("goal")
@click.option("--workspace", type=click.Path(exists=True, path_type=Path), default=None)
@click.option(
    "--server",
    "server_url",
    default=None,
    help="Server URL. Defaults to configured Loom API server.",
)
@click.option(
    "--process",
    "process_name",
    default=None,
    help="Process definition name or path.",
)
@click.option(
    "--fresh",
    "-f",
    "fresh_adhoc",
    is_flag=True,
    default=False,
    help="Bypass ad hoc process cache when no explicit/default process is set.",
)
@click.option(
    "--auth-profile",
    "auth_profile_pairs",
    multiple=True,
    help="Auth selector mapping (selector=profile_id). Repeatable.",
)
@click.pass_context
def run(
    ctx: click.Context,
    goal: str,
    workspace: Path | None,
    server_url: str | None,
    process_name: str | None,
    fresh_adhoc: bool,
    auth_profile_pairs: tuple[str, ...],
) -> None:
    """Submit a run goal and stream server-side process execution progress."""
    config = _effective_config(ctx, workspace)
    ws_path = _resolve_workspace(workspace or ctx.obj.get("workspace"))
    effective_process = process_name or config.process.default or None
    url = server_url or f"http://{config.server.host}:{config.server.port}"
    try:
        overrides = parse_auth_profile_overrides(auth_profile_pairs)
    except AuthResolutionError as e:
        click.echo(f"Invalid --auth-profile value: {e}", err=True)
        sys.exit(1)
    explicit_auth = ctx.obj.get("explicit_auth_path")

    metadata: dict[str, object] = {"execution_surface": "cli"}
    if overrides:
        metadata["auth_profile_overrides"] = overrides
    if explicit_auth is not None:
        metadata["auth_config_path"] = str(explicit_auth)

    resolved_process_name: str | None = effective_process
    run_workspace = ws_path
    try:
        resolved_process_name, run_workspace = asyncio.run(
            _prepare_server_run_payload(
                config=config,
                workspace=ws_path,
                goal=goal,
                process_name=effective_process,
                fresh_adhoc=fresh_adhoc,
            )
        )
    except Exception as e:
        click.echo(f"Failed to prepare run process: {e}", err=True)
        sys.exit(1)

    ws = str(run_workspace)

    click.echo(f"Submitting task to {url}: {goal}")
    click.echo(f"Workspace: {ws}")
    if resolved_process_name:
        click.echo(f"Process: {resolved_process_name}")
    if overrides:
        rendered = ", ".join(
            f"{selector}={profile_id}" for selector, profile_id in sorted(overrides.items())
        )
        click.echo(f"Auth profiles: {rendered}")

    asyncio.run(
        _run_task(
            url,
            goal,
            ws,
            process_name=resolved_process_name,
            metadata=metadata if metadata else None,
        )
    )


async def _prepare_server_run_payload(
    *,
    config: Config,
    workspace: Path,
    goal: str,
    process_name: str | None,
    fresh_adhoc: bool,
) -> tuple[str, Path]:
    """Resolve server payload inputs to mirror TUI `/run` process behavior."""
    from loom.processes.schema import ProcessLoader
    from loom.tools import create_default_registry
    from loom.tui.app import LoomApp

    root_workspace = workspace.resolve()
    tools = create_default_registry(config, mcp_startup_mode="background")
    helper = LoomApp(
        model=None,
        tools=tools,
        workspace=root_workspace,
        config=config,
        db=None,
        store=None,
    )

    if process_name:
        extra = [Path(p) for p in config.process.search_paths]
        loader = ProcessLoader(
            workspace=root_workspace,
            extra_search_paths=extra,
            require_rule_scope_metadata=bool(
                getattr(config.process, "require_rule_scope_metadata", False),
            ),
            require_v2_contract=bool(
                getattr(config.process, "require_v2_contract", False),
            ),
        )
        loaded = loader.load(process_name)
        process_label = str(getattr(loaded, "name", "") or process_name)
        run_workspace = await helper._prepare_process_run_workspace(process_label, goal)
        return process_name, run_workspace

    click.echo("Synthesizing ad hoc process for run goal...")
    entry, from_cache = await helper._get_or_create_adhoc_process(
        goal,
        fresh=fresh_adhoc,
    )
    process_defn = entry.process_defn
    process_label = str(getattr(process_defn, "name", "") or "process-run")
    run_workspace = await helper._prepare_process_run_workspace(process_label, goal)
    cache_status = "cache-hit" if from_cache else "cache-miss"
    click.echo(
        f"Ad hoc process: {process_label} ({cache_status}) with {len(process_defn.phases)} phases."
    )
    for line in helper._adhoc_synthesis_activity_lines(
        entry,
        from_cache=from_cache,
        fresh=fresh_adhoc,
    ):
        click.echo(f"  - {line}")
    if entry.recommended_tools:
        click.echo(
            "Recommended additional tools: "
            + ", ".join(sorted(entry.recommended_tools)),
        )

    runtime_process_path = _persist_runtime_adhoc_process(
        helper=helper,
        entry=entry,
    )
    click.echo(f"Ad hoc runtime process file: {runtime_process_path}")
    return str(runtime_process_path), run_workspace


def _persist_runtime_adhoc_process(*, helper, entry) -> Path:
    """Persist ad hoc definition as a process.yaml-like file for server loading."""
    import yaml

    process_defn = entry.process_defn
    key = str(getattr(entry, "key", "") or helper._adhoc_cache_key(entry.goal))
    safe_key = helper._sanitize_kebab_token(
        key,
        fallback="adhoc-runtime",
        max_len=32,
    )
    runtime_dir = helper._adhoc_cache_dir() / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    process_path = runtime_dir / f"{safe_key}.process.yaml"

    payload = helper._serialize_process_for_package(process_defn)
    if not str(payload.get("name", "")).strip():
        payload["name"] = f"adhoc-{safe_key}"
    process_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return process_path


@cli.command()
@click.argument("task_id")
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.pass_context
def status(ctx: click.Context, task_id: str, server_url: str | None) -> None:
    """Check status of a task."""
    config = _effective_config(ctx, None)
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    asyncio.run(_check_status(url, task_id))


@cli.command()
@click.argument("task_id")
@click.option("--server", "server_url", default=None, help="Server URL.")
@click.pass_context
def cancel(ctx: click.Context, task_id: str, server_url: str | None) -> None:
    """Cancel a running task."""
    config = _effective_config(ctx, None)
    url = server_url or f"http://{config.server.host}:{config.server.port}"

    asyncio.run(_cancel_task(url, task_id))


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


_CLI_BUILT = False


def build_cli() -> click.Group:
    """Attach modular command families to the root CLI group."""
    global _CLI_BUILT
    if _CLI_BUILT:
        return cli

    from loom.cli.commands.auth import register_auth_commands
    from loom.cli.commands.db import register_db_commands
    from loom.cli.commands.maintenance import register_maintenance_commands
    from loom.cli.commands.mcp import register_mcp_commands
    from loom.cli.commands.process import register_process_commands

    # Preserve command ordering from the previous monolithic entrypoint.
    register_auth_commands(cli)
    register_mcp_commands(cli)
    register_process_commands(cli)
    register_db_commands(cli)
    register_maintenance_commands(cli)

    _CLI_BUILT = True
    return cli


def main() -> None:
    """CLI runtime entry point."""
    build_cli()()


build_cli()
