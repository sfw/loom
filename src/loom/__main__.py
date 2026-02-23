"""CLI entry point for Loom."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from pathlib import Path

import click

from loom import __version__
from loom.auth.config import (
    AuthConfigError,
    AuthProfile,
    default_workspace_auth_defaults_path,
    load_merged_auth_config,
    remove_auth_profile,
    resolve_auth_write_path,
    set_workspace_auth_default,
    upsert_auth_profile,
)
from loom.auth.runtime import (
    AuthResolutionError,
    parse_auth_profile_overrides,
)
from loom.config import Config, ConfigError, load_config
from loom.mcp.config import (
    MCPConfigManager,
    MCPConfigManagerError,
    MCPServerView,
    apply_mcp_overrides,
    ensure_valid_alias,
    ensure_valid_env_key,
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


def _merged_auth_config(
    ctx: click.Context,
    workspace: Path | None = None,
):
    """Load merged auth profile config for CLI commands."""
    active_workspace = workspace or ctx.obj.get("workspace")
    try:
        return load_merged_auth_config(
            workspace=active_workspace,
            explicit_path=ctx.obj.get("explicit_auth_path"),
        )
    except AuthConfigError as e:
        click.echo(f"Auth config error: {e}", err=True)
        sys.exit(1)


def _auth_write_path(ctx: click.Context) -> Path:
    """Resolve writable auth.toml path for auth mutations."""
    return resolve_auth_write_path(
        explicit_path=ctx.obj.get("explicit_auth_path"),
    )


def _parse_kv_pairs(
    pairs: tuple[str, ...],
    *,
    option_name: str,
    validate_key: Callable[[str], str] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in pairs:
        raw = str(pair or "").strip()
        if not raw:
            continue
        if "=" not in raw:
            raise AuthConfigError(
                f"Invalid {option_name} value {pair!r}; expected KEY=VALUE."
            )
        key, value = raw.split("=", 1)
        clean_key = key.strip()
        if validate_key is not None:
            try:
                clean_key = validate_key(clean_key)
            except Exception as e:
                raise AuthConfigError(str(e)) from e
        if not clean_key:
            raise AuthConfigError(
                f"Invalid {option_name} value {pair!r}; key cannot be empty."
            )
        result[clean_key] = value
    return result


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
    "--process", "process_name", default=None,
    help="Process definition name or path.",
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
        ctx.obj["explicit_auth_path"] = (
            auth_config_path.expanduser().resolve() if auth_config_path else None
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
            ctx.obj["explicit_auth_path"] = (
                auth_config_path.expanduser().resolve() if auth_config_path else None
            )
        else:
            click.echo(f"Configuration error: {e}", err=True)
            sys.exit(1)

    if ctx.invoked_subcommand is None:
        # Default: launch the TUI
        _launch_tui(
            ctx.obj["config"], workspace, model,
            resume_session, process_name,
            ctx.obj.get("explicit_mcp_path"),
            ctx.obj.get("config_path"),
            ctx.obj.get("explicit_auth_path"),
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
    explicit_mcp_path: Path | None = None,
    legacy_config_path: Path | None = None,
    explicit_auth_path: Path | None = None,
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
        explicit_mcp_path=explicit_mcp_path,
        legacy_config_path=legacy_config_path,
        explicit_auth_path=explicit_auth_path,
    )
    # Explicitly enable mouse support so click/scroll interactions stay
    # available even if Textual changes defaults across versions.
    app.run(mouse=True)


def _init_persistence(config: Config):
    """Initialize database and conversation store.

    Returns (db, store) on success, or (None, None) if initialization fails.
    The TUI will fall back to ephemeral mode when store is None.
    """
    from loom.state.conversation_store import ConversationStore
    from loom.state.memory import Database

    try:
        db_path = config.database_path
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
    _launch_tui(
        config,
        workspace,
        model,
        resume_session,
        process_name,
        ctx.obj.get("explicit_mcp_path"),
        ctx.obj.get("config_path"),
        ctx.obj.get("explicit_auth_path"),
    )


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
@click.option(
    "--server",
    "server_url",
    default=None,
    help="Server URL. Defaults to configured Loom API server.",
)
@click.option(
    "--process", "process_name", default=None,
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
    ctx: click.Context, goal: str, workspace: Path | None,
    server_url: str | None, process_name: str | None,
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

    metadata: dict[str, object] = {}
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
    tools = create_default_registry(config)
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
        f"Ad hoc process: {process_label} ({cache_status}) "
        f"with {len(process_defn.phases)} phases."
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
    metadata: dict | None = None,
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
            if metadata:
                payload["metadata"] = metadata

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


# -- Auth profile management commands -------------------------------------

@cli.group()
def auth() -> None:
    """Manage auth profile configuration."""


@auth.command(name="list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Include all profile metadata fields.",
)
@click.pass_context
def auth_list(ctx: click.Context, as_json: bool, verbose: bool) -> None:
    """List merged auth profiles and defaults."""
    merged = _merged_auth_config(ctx)
    profiles = merged.config.profiles
    effective_defaults = {
        selector: profile_id
        for selector, profile_id in {
            **merged.config.defaults,
            **merged.workspace_defaults,
        }.items()
        if not selector.startswith("mcp.")
    }

    payload = {
        "sources": {
            "user_path": str(merged.user_path),
            "explicit_path": (
                str(merged.explicit_path)
                if merged.explicit_path is not None else None
            ),
            "workspace_defaults_path": (
                str(merged.workspace_defaults_path)
                if merged.workspace_defaults_path is not None else None
            ),
        },
        "defaults": effective_defaults,
        "profiles": [],
    }
    for profile_id in sorted(profiles):
        profile = profiles[profile_id]
        item = {
            "id": profile.profile_id,
            "provider": profile.provider,
            "mode": profile.mode,
            "account_label": profile.account_label,
        }
        if verbose:
            item.update({
                "secret_ref": profile.secret_ref,
                "token_ref": profile.token_ref,
                "scopes": list(profile.scopes),
                "env_keys": sorted(profile.env.keys()),
                "command": profile.command,
                "auth_check": list(profile.auth_check),
                "metadata": dict(profile.metadata),
            })
        payload["profiles"].append(item)

    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("Auth profiles:")
    click.echo(f"  user:     {merged.user_path}")
    click.echo(f"  explicit: {merged.explicit_path or '-'}")
    click.echo(f"  workspace defaults: {merged.workspace_defaults_path or '-'}")
    if not profiles:
        click.echo("  (none)")
    else:
        for item in payload["profiles"]:
            click.echo(
                f"  {item['id']:24} provider={item['provider']} mode={item['mode']}"
            )
            if item.get("account_label"):
                click.echo(f"    label: {item['account_label']}")
            if verbose:
                env_keys = ", ".join(item.get("env_keys", [])) or "-"
                click.echo(f"    env_keys: {env_keys}")
                if item.get("secret_ref"):
                    click.echo(f"    secret_ref: {item['secret_ref']}")
                if item.get("token_ref"):
                    click.echo(f"    token_ref: {item['token_ref']}")
                if item.get("command"):
                    click.echo(f"    command: {item['command']}")

    if effective_defaults:
        click.echo("Defaults:")
        for selector, profile_id in sorted(effective_defaults.items()):
            click.echo(f"  {selector} -> {profile_id}")


@auth.command(name="show")
@click.argument("profile_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def auth_show(ctx: click.Context, profile_id: str, as_json: bool) -> None:
    """Show one auth profile."""
    merged = _merged_auth_config(ctx)
    profile = merged.config.profiles.get(profile_id)
    if profile is None:
        click.echo(f"Auth profile not found: {profile_id}", err=True)
        sys.exit(1)

    payload = {
        "id": profile.profile_id,
        "provider": profile.provider,
        "mode": profile.mode,
        "account_label": profile.account_label,
        "secret_ref": profile.secret_ref,
        "token_ref": profile.token_ref,
        "scopes": list(profile.scopes),
        "env": dict(profile.env),
        "command": profile.command,
        "auth_check": list(profile.auth_check),
        "metadata": dict(profile.metadata),
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Profile: {payload['id']}")
    click.echo(f"Provider: {payload['provider']}")
    click.echo(f"Mode: {payload['mode']}")
    click.echo(f"Label: {payload['account_label'] or '-'}")
    click.echo(f"Secret ref: {payload['secret_ref'] or '-'}")
    click.echo(f"Token ref: {payload['token_ref'] or '-'}")
    click.echo(f"Scopes: {', '.join(payload['scopes']) or '-'}")
    click.echo(f"Command: {payload['command'] or '-'}")
    if payload["env"]:
        click.echo("Env keys:")
        for key in sorted(payload["env"]):
            click.echo(f"  - {key}")


@auth.command(name="check")
@click.pass_context
def auth_check(ctx: click.Context) -> None:
    """Validate auth profile references and defaults."""
    merged = _merged_auth_config(ctx)
    profiles = merged.config.profiles
    effective_defaults = {
        selector: profile_id
        for selector, profile_id in {
            **merged.config.defaults,
            **merged.workspace_defaults,
        }.items()
        if not selector.startswith("mcp.")
    }

    errors: list[str] = []
    for selector, profile_id in sorted(effective_defaults.items()):
        if profile_id not in profiles:
            errors.append(
                f"default selector {selector!r} references unknown profile {profile_id!r}"
            )
            continue
        profile = profiles[profile_id]
        if selector != profile.provider:
            errors.append(
                f"default selector {selector!r} must match profile provider {profile.provider!r}"
            )

    if errors:
        click.echo("Auth config validation failed:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    click.echo("Auth config is valid.")
    click.echo(f"Profiles: {len(profiles)}")
    click.echo(f"Defaults: {len(effective_defaults)}")
    defaults_path = (
        merged.workspace_defaults_path
        or default_workspace_auth_defaults_path(ctx.obj.get("workspace"))
    )
    click.echo(
        f"Workspace defaults file: {defaults_path}"
    )


@auth.command(name="select")
@click.argument("selector")
@click.argument("profile_id", required=False)
@click.option(
    "--unset",
    is_flag=True,
    default=False,
    help="Remove selector mapping from workspace defaults.",
)
@click.pass_context
def auth_select(
    ctx: click.Context,
    selector: str,
    profile_id: str | None,
    unset: bool,
) -> None:
    """Set/clear workspace auth default selector mapping."""
    merged = _merged_auth_config(ctx)
    clean_selector = str(selector or "").strip()
    if not clean_selector:
        click.echo("Selector cannot be empty.", err=True)
        sys.exit(1)
    if clean_selector.startswith("mcp."):
        click.echo(
            "MCP selectors are no longer supported in `loom auth select`. "
            "Use separate MCP aliases in `loom mcp` for multi-account MCP auth.",
            err=True,
        )
        sys.exit(1)

    workspace = ctx.obj.get("workspace")
    defaults_path = default_workspace_auth_defaults_path(workspace)

    if unset:
        if profile_id:
            click.echo(
                "Do not pass profile_id when using --unset.",
                err=True,
            )
            sys.exit(1)
        try:
            updated = set_workspace_auth_default(
                defaults_path,
                selector=clean_selector,
                profile_id=None,
            )
        except AuthConfigError as e:
            click.echo(f"Auth select failed: {e}", err=True)
            sys.exit(1)
        click.echo(f"Removed default mapping: {clean_selector}")
        click.echo(f"Workspace defaults file: {defaults_path}")
        click.echo(f"Remaining defaults: {len(updated)}")
        return

    clean_profile_id = str(profile_id or "").strip()
    if not clean_profile_id:
        click.echo("Missing profile_id. Usage: loom auth select <selector> <profile_id>", err=True)
        sys.exit(1)
    profile = merged.config.profiles.get(clean_profile_id)
    if profile is None:
        click.echo(f"Unknown auth profile: {clean_profile_id}", err=True)
        sys.exit(1)
    if clean_selector != profile.provider:
        click.echo(
            (
                f"Selector {clean_selector!r} must match profile provider "
                f"{profile.provider!r}."
            ),
            err=True,
        )
        sys.exit(1)

    try:
        set_workspace_auth_default(
            defaults_path,
            selector=clean_selector,
            profile_id=clean_profile_id,
        )
    except AuthConfigError as e:
        click.echo(f"Auth select failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Set workspace default: {clean_selector} -> {clean_profile_id}")
    click.echo(f"Workspace defaults file: {defaults_path}")


@auth.group(name="profile")
def auth_profile() -> None:
    """Manage auth profile definitions."""


@auth_profile.command(name="list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Include all profile metadata fields.",
)
@click.pass_context
def auth_profile_list(ctx: click.Context, as_json: bool, verbose: bool) -> None:
    """Alias for `loom auth list`."""
    ctx.invoke(auth_list, as_json=as_json, verbose=verbose)


@auth_profile.command(name="show")
@click.argument("profile_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def auth_profile_show(ctx: click.Context, profile_id: str, as_json: bool) -> None:
    """Alias for `loom auth show`."""
    ctx.invoke(auth_show, profile_id=profile_id, as_json=as_json)


@auth_profile.command(name="add")
@click.argument("profile_id")
@click.option("--provider", required=True, help="Provider id (e.g. notion).")
@click.option(
    "--mode",
    required=True,
    help="Credential mode (api_key, oauth2_pkce, env_passthrough, ...).",
)
@click.option("--label", "account_label", default="", help="Human-friendly account label.")
@click.option("--secret-ref", default="", help="Secret ref (env://... or keychain://...).")
@click.option("--token-ref", default="", help="OAuth token ref (env://... or keychain://...).")
@click.option("--scope", "scopes", multiple=True, help="OAuth scope. Repeatable.")
@click.option("--env", "env_pairs", multiple=True, help="Env mapping KEY=VALUE.")
@click.option("--command", default="", help="CLI binary for cli_passthrough mode.")
@click.option(
    "--auth-check",
    "auth_check",
    multiple=True,
    help="CLI auth check arg token. Repeatable.",
)
@click.option("--meta", "meta_pairs", multiple=True, help="Metadata KEY=VALUE.")
@click.pass_context
def auth_profile_add(
    ctx: click.Context,
    profile_id: str,
    provider: str,
    mode: str,
    account_label: str,
    secret_ref: str,
    token_ref: str,
    scopes: tuple[str, ...],
    env_pairs: tuple[str, ...],
    command: str,
    auth_check: tuple[str, ...],
    meta_pairs: tuple[str, ...],
) -> None:
    """Add a new auth profile entry."""
    clean_profile_id = str(profile_id or "").strip()
    if not clean_profile_id:
        click.echo("Profile id cannot be empty.", err=True)
        sys.exit(1)

    try:
        env = _parse_kv_pairs(
            env_pairs,
            option_name="--env",
            validate_key=ensure_valid_env_key,
        )
        metadata = _parse_kv_pairs(meta_pairs, option_name="--meta")
    except AuthConfigError as e:
        click.echo(f"Add failed: {e}", err=True)
        sys.exit(1)

    profile = AuthProfile(
        profile_id=clean_profile_id,
        provider=str(provider or "").strip(),
        mode=str(mode or "").strip(),
        account_label=str(account_label or "").strip(),
        secret_ref=str(secret_ref or "").strip(),
        token_ref=str(token_ref or "").strip(),
        scopes=[scope for scope in (str(s).strip() for s in scopes) if scope],
        env=env,
        command=str(command or "").strip(),
        auth_check=[token for token in (str(a).strip() for a in auth_check) if token],
        metadata=metadata,
    )

    if not profile.provider:
        click.echo("--provider cannot be empty.", err=True)
        sys.exit(1)
    if not profile.mode:
        click.echo("--mode cannot be empty.", err=True)
        sys.exit(1)

    target = _auth_write_path(ctx)
    try:
        updated = upsert_auth_profile(
            target,
            profile,
            must_exist=False,
        )
    except AuthConfigError as e:
        click.echo(f"Add failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Added auth profile '{clean_profile_id}' to {target}")
    click.echo(f"Profiles in file: {len(updated.profiles)}")


@auth_profile.command(name="edit")
@click.argument("profile_id")
@click.option("--provider", default=None, help="Provider id.")
@click.option("--mode", default=None, help="Credential mode.")
@click.option("--label", "account_label", default=None, help="Account label.")
@click.option("--secret-ref", default=None, help="Secret ref.")
@click.option("--token-ref", default=None, help="Token ref.")
@click.option("--scope", "scopes", multiple=True, help="Replace scopes.")
@click.option("--clear-scopes", is_flag=True, default=False, help="Clear scopes.")
@click.option("--env", "env_pairs", multiple=True, help="Merge env KEY=VALUE.")
@click.option("--clear-env", is_flag=True, default=False, help="Clear env mapping.")
@click.option("--command", default=None, help="Command value.")
@click.option(
    "--auth-check",
    "auth_check",
    multiple=True,
    help="Replace auth_check list values.",
)
@click.option(
    "--clear-auth-check",
    is_flag=True,
    default=False,
    help="Clear auth_check entries.",
)
@click.option("--meta", "meta_pairs", multiple=True, help="Merge metadata KEY=VALUE.")
@click.option("--clear-meta", is_flag=True, default=False, help="Clear metadata.")
@click.pass_context
def auth_profile_edit(
    ctx: click.Context,
    profile_id: str,
    provider: str | None,
    mode: str | None,
    account_label: str | None,
    secret_ref: str | None,
    token_ref: str | None,
    scopes: tuple[str, ...],
    clear_scopes: bool,
    env_pairs: tuple[str, ...],
    clear_env: bool,
    command: str | None,
    auth_check: tuple[str, ...],
    clear_auth_check: bool,
    meta_pairs: tuple[str, ...],
    clear_meta: bool,
) -> None:
    """Edit an existing auth profile entry."""
    clean_profile_id = str(profile_id or "").strip()
    if not clean_profile_id:
        click.echo("Profile id cannot be empty.", err=True)
        sys.exit(1)

    target = _auth_write_path(ctx)
    try:
        current_cfg = load_merged_auth_config(
            workspace=ctx.obj.get("workspace"),
            explicit_path=ctx.obj.get("explicit_auth_path"),
        ).config
    except AuthConfigError as e:
        click.echo(f"Edit failed: {e}", err=True)
        sys.exit(1)
    current = current_cfg.profiles.get(clean_profile_id)
    if current is None:
        click.echo(f"Auth profile not found: {clean_profile_id}", err=True)
        sys.exit(1)

    try:
        env_updates = _parse_kv_pairs(
            env_pairs,
            option_name="--env",
            validate_key=ensure_valid_env_key,
        )
        meta_updates = _parse_kv_pairs(meta_pairs, option_name="--meta")
    except AuthConfigError as e:
        click.echo(f"Edit failed: {e}", err=True)
        sys.exit(1)

    next_scopes = list(current.scopes)
    if clear_scopes:
        next_scopes = []
    elif scopes:
        next_scopes = [scope for scope in (str(s).strip() for s in scopes) if scope]

    next_env = {} if clear_env else dict(current.env)
    next_env.update(env_updates)

    next_auth_check = list(current.auth_check)
    if clear_auth_check:
        next_auth_check = []
    elif auth_check:
        next_auth_check = [token for token in (str(a).strip() for a in auth_check) if token]

    next_metadata = {} if clear_meta else dict(current.metadata)
    next_metadata.update(meta_updates)

    updated_profile = AuthProfile(
        profile_id=current.profile_id,
        provider=current.provider if provider is None else str(provider).strip(),
        mode=current.mode if mode is None else str(mode).strip(),
        account_label=(
            current.account_label
            if account_label is None else str(account_label).strip()
        ),
        secret_ref=current.secret_ref if secret_ref is None else str(secret_ref).strip(),
        token_ref=current.token_ref if token_ref is None else str(token_ref).strip(),
        scopes=next_scopes,
        env=next_env,
        command=current.command if command is None else str(command).strip(),
        auth_check=next_auth_check,
        metadata=next_metadata,
    )
    if not updated_profile.provider:
        click.echo("Provider cannot be empty after edit.", err=True)
        sys.exit(1)
    if not updated_profile.mode:
        click.echo("Mode cannot be empty after edit.", err=True)
        sys.exit(1)

    try:
        upsert_auth_profile(
            target,
            updated_profile,
            must_exist=True,
        )
    except AuthConfigError as e:
        click.echo(f"Edit failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Updated auth profile '{clean_profile_id}' in {target}")


@auth_profile.command(name="remove")
@click.argument("profile_id")
@click.pass_context
def auth_profile_remove(ctx: click.Context, profile_id: str) -> None:
    """Remove an auth profile entry."""
    clean_profile_id = str(profile_id or "").strip()
    if not clean_profile_id:
        click.echo("Profile id cannot be empty.", err=True)
        sys.exit(1)
    target = _auth_write_path(ctx)
    try:
        updated = remove_auth_profile(target, clean_profile_id)
    except AuthConfigError as e:
        click.echo(f"Remove failed: {e}", err=True)
        sys.exit(1)
    click.echo(f"Removed auth profile '{clean_profile_id}' from {target}")
    click.echo(f"Profiles remaining: {len(updated.profiles)}")


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
    loader = ProcessLoader(
        workspace=ws,
        extra_search_paths=extra,
        require_rule_scope_metadata=bool(
            getattr(config.process, "require_rule_scope_metadata", False),
        ),
        require_v2_contract=bool(
            getattr(config.process, "require_v2_contract", False),
        ),
    )
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
    loader = ProcessLoader(
        workspace=ws,
        extra_search_paths=extra,
        require_rule_scope_metadata=bool(
            getattr(config.process, "require_rule_scope_metadata", False),
        ),
        require_v2_contract=bool(
            getattr(config.process, "require_v2_contract", False),
        ),
    )

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
        db = Database(str(config.database_path))
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
