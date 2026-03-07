"""`loom mcp ...` CLI commands."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import click

from loom.auth.resources import cleanup_deleted_resource, resource_delete_impact
from loom.cli.commands.mcp_auth import attach_mcp_auth_commands
from loom.cli.context import (
    _close_runtime_mcp_registry,
    _mcp_manager,
    _open_runtime_mcp_registry,
    _serialize_mcp_view,
)
from loom.integrations.mcp.oauth import default_mcp_oauth_store_path, oauth_state_for_alias
from loom.integrations.mcp_tools import (
    runtime_connect_alias,
    runtime_connection_states,
    runtime_debug_snapshot,
    runtime_disconnect_alias,
    runtime_reconnect_alias,
)
from loom.mcp.config import (
    MCPConfigManagerError,
    ensure_valid_alias,
    merge_server_edits,
    parse_mcp_server_from_flags,
    redact_server_env,
    redact_server_headers,
)


@click.group()
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
                _serialize_mcp_view(
                    view,
                    redacted=True,
                    oauth_store_path=default_mcp_oauth_store_path(),
                )
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
        click.echo(
            f"  {view.alias:16} {status:8} type={view.server.type:6} source={view.source}"
        )
        if verbose:
            click.echo(f"    path: {source_path}")
            if view.server.type == "remote":
                click.echo(f"    url: {view.server.url or '-'}")
                click.echo(f"    fallback_sse_url: {view.server.fallback_sse_url or '-'}")
                click.echo(
                    "    oauth: "
                    f"{'enabled' if view.server.oauth.enabled else 'disabled'}"
                )
                if view.server.oauth.scopes:
                    click.echo(f"    oauth_scopes: {', '.join(view.server.oauth.scopes)}")
                headers = redact_server_headers(view.server)
                if headers:
                    click.echo("    headers:")
                    for key, value in headers.items():
                        click.echo(f"      {key}: {value}")
                else:
                    click.echo("    headers: (none)")
                click.echo(
                    "    allow_insecure_http: "
                    f"{'true' if view.server.allow_insecure_http else 'false'}"
                )
                click.echo(
                    "    allow_private_network: "
                    f"{'true' if view.server.allow_private_network else 'false'}"
                )
            else:
                args = " ".join(view.server.args)
                cmd = f"{view.server.command} {args}".strip()
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
        payload = _serialize_mcp_view(
            view,
            redacted=True,
            oauth_store_path=default_mcp_oauth_store_path(),
        )
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Alias: {view.alias}")
    click.echo(f"Source: {view.source}")
    click.echo(f"Source path: {view.source_path or '-'}")
    click.echo(f"Type: {view.server.type}")
    click.echo(f"Enabled: {view.server.enabled}")
    if view.server.type == "remote":
        click.echo(f"URL: {view.server.url or '-'}")
        click.echo(f"Fallback SSE URL: {view.server.fallback_sse_url or '-'}")
        click.echo(
            "OAuth: "
            f"{'enabled' if view.server.oauth.enabled else 'disabled'}"
        )
        if view.server.oauth.enabled:
            oauth_state = oauth_state_for_alias(view.alias)
            click.echo(f"OAuth state: {oauth_state.get('state', 'unknown')}")
        click.echo(
            "OAuth scopes: "
            f"{', '.join(view.server.oauth.scopes) if view.server.oauth.scopes else '-'}"
        )
        click.echo(
            "Allow insecure HTTP: "
            f"{'true' if view.server.allow_insecure_http else 'false'}"
        )
        click.echo(
            "Allow private network: "
            f"{'true' if view.server.allow_private_network else 'false'}"
        )
        headers = redact_server_headers(view.server)
        if headers:
            click.echo("Headers:")
            for key, value in headers.items():
                click.echo(f"  {key}: {value}")
        else:
            click.echo("Headers: (none)")
    else:
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


@mcp.command(name="status")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def mcp_status(ctx: click.Context, as_json: bool) -> None:
    """Show runtime MCP connection state and auth readiness."""
    manager = _mcp_manager(ctx)
    try:
        views = manager.list_views()
    except MCPConfigManagerError as e:
        click.echo(f"Status failed: {e}", err=True)
        sys.exit(1)

    registry = _open_runtime_mcp_registry(ctx)
    try:
        states = runtime_connection_states(registry)
    finally:
        _close_runtime_mcp_registry(registry)

    state_by_alias = {state.alias: state for state in states}
    payload_servers: list[dict[str, object]] = []
    for view in views:
        runtime = state_by_alias.get(view.alias)
        oauth_state = (
            oauth_state_for_alias(view.alias)
            if view.server.oauth.enabled
            else {
                "state": "disabled",
                "has_token": False,
                "expired": False,
                "expires_at": None,
                "token_type": None,
                "scopes": [],
            }
        )
        payload_servers.append(
            {
                "alias": view.alias,
                "type": view.server.type,
                "enabled": view.server.enabled,
                "source": view.source,
                "runtime": {
                    "status": runtime.status if runtime is not None else (
                        "disabled" if not view.server.enabled else "configured"
                    ),
                    "last_error": runtime.last_error if runtime is not None else "",
                    "pid": runtime.pid if runtime is not None else None,
                    "queue_depth": runtime.queue_depth if runtime is not None else 0,
                    "in_flight": runtime.in_flight if runtime is not None else 0,
                    "reconnect_attempts": (
                        runtime.reconnect_attempts if runtime is not None else 0
                    ),
                    "circuit_state": (
                        runtime.circuit_state if runtime is not None else "closed"
                    ),
                    "circuit_open_until": (
                        runtime.circuit_open_until if runtime is not None else None
                    ),
                    "last_connected_at": (
                        runtime.last_connected_at if runtime is not None else None
                    ),
                    "last_activity_at": (
                        runtime.last_activity_at if runtime is not None else None
                    ),
                    "remediation": runtime.remediation if runtime is not None else "",
                },
                "oauth": oauth_state,
            }
        )

    payload = {"servers": payload_servers}
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not payload_servers:
        click.echo("No MCP servers configured.")
        return
    click.echo("MCP runtime status:")
    for item in payload_servers:
        alias = str(item["alias"])
        server_type = str(item["type"])
        enabled = bool(item["enabled"])
        runtime = item["runtime"] if isinstance(item["runtime"], dict) else {}
        oauth = item["oauth"] if isinstance(item["oauth"], dict) else {}
        status = str(runtime.get("status", "unknown"))
        click.echo(
            f"  {alias:16} type={server_type:6} "
            f"{'enabled' if enabled else 'disabled':8} runtime={status}"
        )
        if runtime.get("last_error"):
            click.echo(f"    error: {runtime['last_error']}")
        if runtime.get("circuit_state") and runtime.get("circuit_state") != "closed":
            click.echo(f"    circuit: {runtime['circuit_state']}")
        if runtime.get("remediation"):
            click.echo(f"    next: {runtime['remediation']}")
        if runtime.get("pid"):
            click.echo(f"    pid: {runtime['pid']}")
        oauth_state = str(oauth.get("state", "disabled"))
        if oauth_state != "disabled":
            click.echo(f"    oauth: {oauth_state}")


def _serialize_runtime_state(state) -> dict[str, object]:
    return {
        "alias": state.alias,
        "type": state.type,
        "enabled": state.enabled,
        "status": state.status,
        "last_error": state.last_error,
        "pid": state.pid,
        "queue_depth": state.queue_depth,
        "in_flight": state.in_flight,
        "reconnect_attempts": state.reconnect_attempts,
        "circuit_state": state.circuit_state,
        "circuit_open_until": state.circuit_open_until,
        "last_connected_at": state.last_connected_at,
        "last_activity_at": state.last_activity_at,
        "remediation": state.remediation,
    }


@mcp.command(name="connect")
@click.argument("alias")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def mcp_connect(ctx: click.Context, alias: str, as_json: bool) -> None:
    """Connect one MCP alias and refresh runtime health state."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
        view = manager.get_view(clean_alias)
    except MCPConfigManagerError as e:
        click.echo(f"Connect failed: {e}", err=True)
        sys.exit(1)
    if view is None:
        click.echo(f"MCP server not found: {clean_alias}", err=True)
        sys.exit(1)

    registry = _open_runtime_mcp_registry(ctx)
    try:
        state = runtime_connect_alias(registry, alias=clean_alias)
    except Exception as e:
        click.echo(f"Connect failed for '{clean_alias}': {e}", err=True)
        sys.exit(1)
    finally:
        _close_runtime_mcp_registry(registry)

    payload = _serialize_runtime_state(state)
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Connected MCP server '{clean_alias}' (runtime={state.status}).")
    if state.last_error:
        click.echo(f"  error: {state.last_error}")
    if state.remediation:
        click.echo(f"  next: {state.remediation}")


@mcp.command(name="disconnect")
@click.argument("alias")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def mcp_disconnect(ctx: click.Context, alias: str, as_json: bool) -> None:
    """Disconnect one MCP alias and close any persistent runtime session."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
        view = manager.get_view(clean_alias)
    except MCPConfigManagerError as e:
        click.echo(f"Disconnect failed: {e}", err=True)
        sys.exit(1)
    if view is None:
        click.echo(f"MCP server not found: {clean_alias}", err=True)
        sys.exit(1)

    registry = _open_runtime_mcp_registry(ctx)
    try:
        state = runtime_disconnect_alias(registry, alias=clean_alias)
    except Exception as e:
        click.echo(f"Disconnect failed for '{clean_alias}': {e}", err=True)
        sys.exit(1)
    finally:
        _close_runtime_mcp_registry(registry)

    payload = _serialize_runtime_state(state)
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Disconnected MCP server '{clean_alias}' (runtime={state.status}).")


@mcp.command(name="reconnect")
@click.argument("alias")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def mcp_reconnect(ctx: click.Context, alias: str, as_json: bool) -> None:
    """Reconnect one MCP alias by forcing a disconnect then connect."""
    manager = _mcp_manager(ctx)
    try:
        clean_alias = ensure_valid_alias(alias)
        view = manager.get_view(clean_alias)
    except MCPConfigManagerError as e:
        click.echo(f"Reconnect failed: {e}", err=True)
        sys.exit(1)
    if view is None:
        click.echo(f"MCP server not found: {clean_alias}", err=True)
        sys.exit(1)

    registry = _open_runtime_mcp_registry(ctx)
    try:
        state = runtime_reconnect_alias(registry, alias=clean_alias)
    except Exception as e:
        click.echo(f"Reconnect failed for '{clean_alias}': {e}", err=True)
        sys.exit(1)
    finally:
        _close_runtime_mcp_registry(registry)

    payload = _serialize_runtime_state(state)
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Reconnected MCP server '{clean_alias}' (runtime={state.status}).")
    if state.last_error:
        click.echo(f"  error: {state.last_error}")
    if state.remediation:
        click.echo(f"  next: {state.remediation}")


@mcp.command(name="debug-bundle")
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write bundle JSON to this path (defaults to ~/.loom/debug/...).",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def mcp_debug_bundle(
    ctx: click.Context,
    output_path: Path | None,
    as_json: bool,
) -> None:
    """Export redacted MCP diagnostics for support triage."""
    manager = _mcp_manager(ctx)
    try:
        views = manager.list_views()
    except MCPConfigManagerError as e:
        click.echo(f"Debug bundle failed: {e}", err=True)
        sys.exit(1)

    registry = _open_runtime_mcp_registry(ctx)
    try:
        states = runtime_connection_states(registry)
        runtime_snapshot = runtime_debug_snapshot(registry)
    except Exception as e:
        click.echo(f"Debug bundle failed: {e}", err=True)
        sys.exit(1)
    finally:
        _close_runtime_mcp_registry(registry)

    from loom import __version__

    payload = {
        "created_at_unix": int(time.time()),
        "loom_version": __version__,
        "workspace": str(ctx.obj.get("workspace") or Path.cwd()),
        "config_path": (
            str(ctx.obj.get("config_path"))
            if ctx.obj.get("config_path") is not None
            else None
        ),
        "mcp_config_path": (
            str(ctx.obj.get("explicit_mcp_path"))
            if ctx.obj.get("explicit_mcp_path") is not None
            else None
        ),
        "servers": [
            _serialize_mcp_view(
                view,
                redacted=True,
                oauth_store_path=default_mcp_oauth_store_path(),
            )
            for view in views
        ],
        "runtime": {
            "states": [_serialize_runtime_state(state) for state in states],
            "diagnostics": runtime_snapshot,
        },
    }

    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    target = output_path
    if target is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        target = Path.home() / ".loom" / "debug" / f"mcp-debug-{stamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    click.echo(f"Wrote MCP debug bundle to {target}")


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
    workspace = ctx.obj.get("workspace")
    impact = None
    try:
        impact = resource_delete_impact(
            workspace=workspace,
            resource_kind="mcp",
            resource_key=str(alias or "").strip(),
        )
    except Exception:
        impact = None
    try:
        clean_alias = ensure_valid_alias(alias)
        path = manager.remove_server(clean_alias)
    except MCPConfigManagerError as e:
        click.echo(f"Remove failed: {e}", err=True)
        sys.exit(1)
    try:
        cleanup_deleted_resource(
            workspace=workspace,
            explicit_auth_path=ctx.obj.get("explicit_auth_path"),
            resource_kind="mcp",
            resource_key=clean_alias,
        )
    except Exception as e:
        click.echo(f"Auth cleanup warning: {e}", err=True)
    click.echo(f"Removed MCP server '{clean_alias}' from {path}")
    if impact is not None and impact.resource_id:
        click.echo(
            "Auth cleanup impact: "
            f"bindings={len(impact.active_binding_ids)}, "
            f"profiles={len(impact.active_profile_ids)}, "
            f"default={'yes' if impact.workspace_default_profile_id else 'no'}, "
            f"process_refs={len(impact.referencing_processes)}"
        )


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
            lambda current: replace(current, enabled=True),
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
            lambda current: replace(current, enabled=False),
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


attach_mcp_auth_commands(mcp)


def register_mcp_commands(cli_group: click.Group) -> None:
    cli_group.add_command(mcp)
