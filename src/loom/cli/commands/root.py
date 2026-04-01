"""Root CLI group and top-level commands."""

from __future__ import annotations

import asyncio
import os
import platform
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
from loom.runtime.capabilities import (
    optional_addon_status_by_key,
    optional_addon_statuses,
)


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
        if ctx.invoked_subcommand in {"setup", "doctor"}:
            # Let setup/doctor proceed even with broken/missing config.
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
            ctx.obj["config_error"] = str(e)
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
@click.option(
    "--require-addon",
    "required_addons",
    multiple=True,
    help="Require one optional runtime addon by key; exits non-zero when missing.",
)
@click.pass_context
def doctor(ctx: click.Context, required_addons: tuple[str, ...]) -> None:
    """Report Loom runtime health, configuration, and addon availability."""
    from loom.auth.config import AuthConfigError, load_merged_auth_config
    from loom.mcp.config import MCPConfigManagerError, load_merged_mcp_config
    from loom.models.base import ModelNotAvailableError
    from loom.models.router import ModelRouter

    statuses = optional_addon_statuses()
    status_by_key = {status.key: status for status in statuses}
    warnings: list[str] = []
    errors: list[str] = []
    config_error = str(ctx.obj.get("config_error", "") or "").strip()
    config = ctx.obj.get("config") or Config()
    workspace = (ctx.obj.get("workspace") or Path.cwd()).resolve()
    config_path = ctx.obj.get("config_path")
    explicit_mcp_path = ctx.obj.get("explicit_mcp_path")
    explicit_auth_path = ctx.obj.get("explicit_auth_path")

    click.echo("Runtime doctor")

    click.echo()
    click.echo("Environment")
    click.echo(f"- Loom version: {__version__}")
    click.echo(f"- Python: {platform.python_version()}")
    click.echo(f"- Platform: {platform.system()} {platform.release()}")
    click.echo(f"- Workspace: {workspace}")
    click.echo(f"- Config path: {config_path or '(built-in defaults only)'}")
    click.echo(f"- Explicit MCP config: {explicit_mcp_path or '(none)'}")
    click.echo(f"- Explicit auth config: {explicit_auth_path or '(none)'}")

    click.echo()
    click.echo("Configuration")
    if config_error:
        click.echo("- Status: invalid")
        click.echo(f"  Detail: {config_error}")
        errors.append(f"Invalid Loom config: {config_error}")
        click.echo("- Models: skipped because configuration is invalid")
    else:
        source_label = config.source_path or "(built-in defaults)"
        click.echo("- Status: ok")
        click.echo(f"  Source: {source_label}")
        click.echo(f"- Server: {config.server.host}:{config.server.port}")
        click.echo(f"- Models configured: {len(config.models)}")
        if not config.models:
            warnings.append(
                "No models are configured; `loom setup` or a populated loom.toml is still needed."
            )
            click.echo("  Detail: no configured model providers")
        else:
            router = ModelRouter.from_config(config)
            for role in ("executor", "planner", "verifier"):
                try:
                    provider = router.select(role=role)
                    provider_name = str(getattr(provider, "name", "") or "<unnamed>")
                    click.echo(f"- Model role {role}: {provider_name}")
                except ModelNotAvailableError as e:
                    click.echo(f"- Model role {role}: missing")
                    warnings.append(str(e))
        click.echo(f"- Scratch path: {config.scratch_path}")
        click.echo(f"- Log path: {config.log_path}")

    click.echo()
    click.echo("Persistence")
    if config_error:
        click.echo("- Status: skipped because configuration is invalid")
    else:
        db_path = config.database_path
        click.echo(f"- Database path: {db_path}")
        parent = db_path.parent
        if parent.exists():
            parent_state = "writable" if os.access(parent, os.W_OK) else "not writable"
        else:
            parent_state = "missing (will be created on first run)"
        click.echo(f"- Database parent: {parent} ({parent_state})")
        if parent.exists() and not parent.is_dir():
            errors.append(f"Database parent is not a directory: {parent}")
            click.echo("- Status: failed")
            click.echo("  Detail: database parent path exists but is not a directory")
        elif parent.exists() and not os.access(parent, os.W_OK):
            errors.append(f"Database parent directory is not writable: {parent}")
            click.echo("- Status: failed")
            click.echo("  Detail: database parent directory is not writable")
        elif not db_path.exists():
            click.echo("- Status: missing (will initialize on first run)")
        else:
            async def _inspect_db() -> tuple[dict[str, object], str]:
                import aiosqlite

                from loom.state.migrations import MIGRATIONS, migration_status, verify_schema

                async with aiosqlite.connect(db_path) as conn:
                    payload = await migration_status(conn, steps=MIGRATIONS)
                    await verify_schema(conn, steps=MIGRATIONS)
                    return payload, "ok"

            try:
                payload, _health = asyncio.run(_inspect_db())
            except Exception as e:
                click.echo("- Status: failed")
                click.echo(f"  Detail: {e}")
                errors.append(f"Database schema check failed: {e}")
            else:
                applied_ids = list(payload.get("applied_ids", []))
                pending_ids = list(payload.get("pending_ids", []))
                click.echo("- Status: ok")
                click.echo(f"  Applied migrations: {len(applied_ids)}")
                click.echo(f"  Pending migrations: {len(pending_ids)}")

    click.echo()
    click.echo("Auth")
    try:
        auth = load_merged_auth_config(
            workspace=workspace,
            explicit_path=explicit_auth_path,
        )
    except AuthConfigError as e:
        click.echo("- Status: failed")
        click.echo(f"  Detail: {e}")
        errors.append(f"Auth config invalid: {e}")
    else:
        click.echo("- Status: ok")
        click.echo(
            f"  User config: {auth.user_path} "
            f"({'present' if auth.user_path.exists() else 'missing'})"
        )
        if auth.explicit_path is not None:
            click.echo(
                f"  Explicit overlay: {auth.explicit_path} "
                f"({'present' if auth.explicit_path.exists() else 'missing'})"
            )
        if auth.workspace_defaults_path is not None:
            click.echo(
                f"  Workspace defaults: {auth.workspace_defaults_path} "
                f"({'present' if auth.workspace_defaults_path.exists() else 'missing'})"
            )
        click.echo(f"- Profiles: {len(auth.config.profiles)}")
        click.echo(f"- User defaults: {len(auth.config.defaults)}")
        click.echo(f"- Resource defaults: {len(auth.config.resource_defaults)}")
        click.echo(f"- Workspace defaults: {len(auth.workspace_defaults)}")

    click.echo()
    click.echo("MCP")
    if config_error:
        click.echo("- Status: skipped legacy loom.toml MCP layer because configuration is invalid")
        click.echo("- Detail: user/workspace/explicit MCP files were not merged")
    else:
        try:
            merged_mcp = load_merged_mcp_config(
                config=config,
                workspace=workspace,
                explicit_path=explicit_mcp_path,
                legacy_config_path=config_path,
            )
        except MCPConfigManagerError as e:
            click.echo("- Status: failed")
            click.echo(f"  Detail: {e}")
            errors.append(f"MCP config invalid: {e}")
        else:
            servers = merged_mcp.as_views()
            enabled_count = sum(1 for view in servers if view.server.enabled)
            oauth_count = sum(1 for view in servers if view.server.oauth.enabled)
            remote_count = sum(1 for view in servers if view.server.type == "remote")
            click.echo("- Status: ok")
            click.echo(
                f"  User config: {merged_mcp.user_path} "
                f"({'present' if merged_mcp.user_path.exists() else 'missing'})"
            )
            click.echo(
                f"  Workspace config: {merged_mcp.workspace_path} "
                f"({'present' if merged_mcp.workspace_path.exists() else 'missing'})"
            )
            if merged_mcp.explicit_path is not None:
                click.echo(
                    f"  Explicit overlay: {merged_mcp.explicit_path} "
                    f"({'present' if merged_mcp.explicit_path.exists() else 'missing'})"
                )
            if merged_mcp.legacy_config_path is not None:
                click.echo(f"  Legacy layer: {merged_mcp.legacy_config_path}")
            click.echo(f"- Servers: {len(servers)} total, {enabled_count} enabled")
            click.echo(f"- Remote servers: {remote_count}")
            click.echo(f"- OAuth-enabled servers: {oauth_count}")
            if enabled_count and not status_by_key.get("mcp", None):
                warnings.append("MCP addon status could not be resolved.")
            elif enabled_count and not status_by_key["mcp"].installed:
                warnings.append(
                    "MCP servers are configured but the MCP addon is missing."
                )

    click.echo()
    click.echo("Optional Addons")
    if statuses:
        for status in statuses:
            state = "installed" if status.installed else "missing"
            click.echo(f"- {status.label} ({status.key}): {state}")
            click.echo(f"  Required for: {status.required_for}")
            click.echo(f"  Install: {status.install_hint}")
            if status.detail:
                click.echo(f"  Detail: {status.detail}")
    else:
        click.echo("No optional addons registered.")

    missing_required: list[str] = []
    for raw_key in required_addons:
        key = str(raw_key or "").strip().lower()
        status = optional_addon_status_by_key(key)
        if status is None:
            errors.append(f"Unknown addon key: {raw_key}")
            continue
        if not status.installed:
            missing_required.append(key)

    if warnings:
        click.echo()
        click.echo("Warnings")
        for item in warnings:
            click.echo(f"- {item}")

    if missing_required:
        rendered = ", ".join(sorted(missing_required))
        errors.append(f"Missing required addon(s): {rendered}")

    if errors:
        click.echo()
        click.echo("Errors")
        for item in errors:
            click.echo(f"- {item}")
        click.echo("Doctor failed", err=True)
        sys.exit(1)

    if warnings:
        click.echo()
        click.echo("Doctor passed with warnings")
        return

    click.echo("Doctor passed")


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
