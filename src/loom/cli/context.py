"""Shared CLI context and config helpers."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from loom.auth.config import (
    AuthConfigError,
    load_merged_auth_config,
    resolve_auth_write_path,
)
from loom.config import Config
from loom.integrations.mcp.oauth import default_mcp_oauth_store_path, oauth_state_for_alias
from loom.mcp.config import (
    MCPConfigManager,
    MCPServerView,
    apply_mcp_overrides,
    redact_server_env,
    redact_server_headers,
)
from loom.tools import create_default_registry


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
) -> Any:
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
    oauth_store_path: Path | None = None,
) -> dict[str, object]:
    env = redact_server_env(view.server) if redacted else dict(view.server.env)
    headers = redact_server_headers(view.server) if redacted else dict(view.server.headers)
    payload: dict[str, object] = {
        "alias": view.alias,
        "source": view.source,
        "source_path": str(view.source_path) if view.source_path else None,
        "type": view.server.type,
        "command": view.server.command,
        "url": view.server.url,
        "fallback_sse_url": view.server.fallback_sse_url,
        "args": list(view.server.args),
        "cwd": view.server.cwd,
        "headers": headers,
        "oauth": {
            "enabled": view.server.oauth.enabled,
            "scopes": list(view.server.oauth.scopes),
        },
        "approval_required": bool(view.server.approval_required),
        "approval_state": str(view.server.approval_state or "not_required"),
        "allow_insecure_http": view.server.allow_insecure_http,
        "allow_private_network": view.server.allow_private_network,
        "timeout_seconds": view.server.timeout_seconds,
        "enabled": view.server.enabled,
        "env": env,
    }
    if view.server.oauth.enabled:
        try:
            payload["oauth_state"] = oauth_state_for_alias(
                view.alias,
                server=view.server,
                store_path=oauth_store_path,
            )
        except Exception:
            payload["oauth_state"] = {
                "state": "error",
                "has_token": False,
                "expired": False,
            }
    return payload


def _open_runtime_mcp_registry(ctx: click.Context):
    """Create a registry instance with MCP runtime initialized."""
    config = _effective_config(ctx)
    return create_default_registry(config, mcp_startup_mode="sync")


def _close_runtime_mcp_registry(registry) -> None:
    synchronizer = getattr(registry, "_mcp_synchronizer", None)
    close_fn = getattr(synchronizer, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


def default_oauth_store_path() -> Path:
    """Return default MCP OAuth store path."""
    return default_mcp_oauth_store_path()
