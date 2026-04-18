"""`loom mcp auth ...` CLI commands."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import click

from loom.cli.context import _effective_config, _mcp_manager
from loom.integrations.mcp.oauth import (
    MCPOAuthFlowError,
    MCPOAuthStoreError,
    oauth_state_for_alias,
    refresh_mcp_oauth_token,
    remove_mcp_oauth_token,
    resolve_mcp_oauth_provider,
    upsert_mcp_oauth_token,
)
from loom.mcp.config import MCPConfigManagerError, MCPServerView, ensure_valid_alias
from loom.oauth.engine import OAuthEngine, OAuthEngineError, OAuthProviderConfig


def _resolve_access_token(
    *,
    access_token: str | None,
    access_token_env: str | None,
) -> str:
    direct = str(access_token or "").strip()
    if direct:
        return direct
    env_name = str(access_token_env or "").strip()
    if env_name:
        return str(os.environ.get(env_name, "")).strip()
    return ""


def _merge_oauth_scopes(*scope_groups: object) -> list[str]:
    merged: list[str] = []
    for group in scope_groups:
        if isinstance(group, str):
            candidates = [item for item in group.split(" ") if item.strip()]
        elif isinstance(group, (list, tuple, set)):
            candidates = [str(item).strip() for item in group]
        else:
            continue
        for candidate in candidates:
            scope = str(candidate or "").strip()
            if not scope:
                continue
            merged.append(scope)
    return list(dict.fromkeys(merged))


def _parse_expiry_epoch_from_token_payload(token_payload: dict[str, object]) -> int | None:
    expires_in = token_payload.get("expires_in")
    if expires_in not in (None, ""):
        try:
            return int(time.time()) + max(1, int(expires_in))
        except (TypeError, ValueError):
            pass
    for key in ("expires_at", "expires_on"):
        raw = token_payload.get(key)
        if raw in (None, ""):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _resolve_mcp_oauth_provider_config(
    *,
    view: MCPServerView,
    scopes: tuple[str, ...],
    authorize_url: str | None,
    token_url: str | None,
    client_id: str | None,
    redirect_port: int,
) -> OAuthProviderConfig:
    if view.server.type != "remote":
        raise MCPOAuthFlowError(
            "Browser OAuth login currently supports only remote MCP aliases."
        )
    provider = resolve_mcp_oauth_provider(
        server_url=view.server.url,
        scopes=_merge_oauth_scopes(view.server.oauth.scopes, scopes),
        authorization_endpoint=authorize_url,
        token_endpoint=token_url,
        client_id=client_id,
        redirect_uris=(
            f"http://127.0.0.1:{max(1, int(redirect_port))}/oauth/callback",
            f"http://localhost:{max(1, int(redirect_port))}/oauth/callback",
            "urn:ietf:wg:oauth:2.0:oob",
        ),
        client_name=f"Loom MCP ({view.alias})",
    )
    return OAuthProviderConfig(
        authorization_endpoint=provider.authorization_endpoint,
        token_endpoint=provider.token_endpoint,
        client_id=provider.client_id,
        scopes=provider.scopes,
        authorize_params=dict(provider.authorize_params),
        token_params=dict(provider.token_params),
    )


def _require_mcp_server_approval(view: MCPServerView) -> None:
    if not bool(view.server.approval_required):
        return
    approval_state = str(view.server.approval_state or "").strip().lower()
    if approval_state == "approved":
        return
    if approval_state == "rejected":
        raise MCPOAuthFlowError(
            "This workspace-defined remote MCP server is rejected. "
            "Run `loom mcp approve <alias>` after re-checking provenance."
        )
    raise MCPOAuthFlowError(
        "This workspace-defined remote MCP server needs approval first. "
        "Run `loom mcp approve <alias>` before connecting an account."
    )


def attach_mcp_auth_commands(mcp_group: click.Group) -> click.Group:
    @mcp_group.group(name="auth")
    def mcp_auth() -> None:
        """Manage OAuth tokens for remote MCP server aliases."""

    @mcp_auth.command(name="login")
    @click.argument("alias")
    @click.option(
        "--manual-token",
        is_flag=True,
        default=False,
        help="Use manual token import path instead of browser PKCE flow.",
    )
    @click.option("--access-token", default=None, help="OAuth access token.")
    @click.option(
        "--access-token-env",
        default=None,
        help="Environment variable containing access token.",
    )
    @click.option("--refresh-token", default="", help="Optional refresh token.")
    @click.option("--token-type", default="Bearer", show_default=True)
    @click.option("--scope", "scopes", multiple=True, help="OAuth scope. Repeatable.")
    @click.option("--expires-in", type=int, default=None, help="Token lifetime in seconds.")
    @click.option(
        "--authorize-url",
        default=None,
        help="OAuth authorization endpoint override.",
    )
    @click.option(
        "--token-url",
        default=None,
        help="OAuth token endpoint override.",
    )
    @click.option(
        "--client-id",
        default=None,
        help=(
            "OAuth client id override (defaults to LOOM_MCP_OAUTH_CLIENT_ID, "
            "dynamic registration when available, or loom-cli)."
        ),
    )
    @click.option(
        "--redirect-port",
        type=int,
        default=8765,
        show_default=True,
        help="Loopback callback port for browser auth.",
    )
    @click.option(
        "--timeout-seconds",
        type=int,
        default=180,
        show_default=True,
        help="Max seconds to wait for OAuth callback completion.",
    )
    @click.option(
        "--no-browser",
        is_flag=True,
        default=False,
        help="Do not auto-open browser; print URL and accept manual callback input.",
    )
    @click.option(
        "--callback-code",
        default=None,
        help="Callback URL or authorization code for manual completion.",
    )
    @click.option(
        "--oauth-store",
        "oauth_store_path",
        type=click.Path(path_type=Path),
        default=None,
        help="Override OAuth token store path.",
    )
    @click.pass_context
    def mcp_auth_login(
        ctx: click.Context,
        alias: str,
        manual_token: bool,
        access_token: str | None,
        access_token_env: str | None,
        refresh_token: str,
        token_type: str,
        scopes: tuple[str, ...],
        expires_in: int | None,
        authorize_url: str | None,
        token_url: str | None,
        client_id: str | None,
        redirect_port: int,
        timeout_seconds: int,
        no_browser: bool,
        callback_code: str | None,
        oauth_store_path: Path | None,
    ) -> None:
        """Authenticate one MCP alias (browser PKCE by default)."""
        manager = _mcp_manager(ctx)
        try:
            clean_alias = ensure_valid_alias(alias)
            view = manager.get_view(clean_alias)
        except MCPConfigManagerError as e:
            click.echo(f"MCP auth login failed: {e}", err=True)
            sys.exit(1)
        if view is None:
            click.echo(f"MCP server not found: {clean_alias}", err=True)
            sys.exit(1)
        if not view.server.oauth.enabled:
            click.echo(
                f"MCP server '{clean_alias}' does not have oauth.enabled=true in config.",
                err=True,
            )
            sys.exit(1)
        try:
            _require_mcp_server_approval(view)
        except MCPOAuthFlowError as e:
            click.echo(f"MCP auth login failed: {e}", err=True)
            sys.exit(1)

        manual_access_token = _resolve_access_token(
            access_token=access_token,
            access_token_env=access_token_env,
        )
        use_manual_token_path = bool(manual_token or manual_access_token)
        if use_manual_token_path:
            token = manual_access_token
            if not token:
                token = click.prompt("Access token", hide_input=True).strip()
            if not token:
                click.echo("Access token cannot be empty.", err=True)
                sys.exit(1)
            expires_at_unix: int | None = None
            if expires_in is not None:
                expires_at_unix = int(time.time()) + max(1, int(expires_in))
            merged_scopes = _merge_oauth_scopes(view.server.oauth.scopes, scopes)
            try:
                path = upsert_mcp_oauth_token(
                    alias=clean_alias,
                    server=view.server,
                    access_token=token,
                    refresh_token=refresh_token,
                    token_type=token_type,
                    scopes=merged_scopes,
                    expires_at_unix=expires_at_unix,
                    store_path=oauth_store_path,
                    obtained_via="manual_token",
                )
            except MCPOAuthStoreError as e:
                click.echo(f"MCP auth login failed: {e}", err=True)
                sys.exit(1)
            click.echo(f"Stored MCP OAuth token for '{clean_alias}' in {path}")
            return

        if not bool(_effective_config(ctx).mcp.oauth_browser_login):
            click.echo(
                "MCP browser OAuth login is disabled by config "
                "(mcp.oauth_browser_login=false).",
                err=True,
            )
            click.echo(
                "Fallback: use `loom mcp auth login <alias> --manual-token --access-token ...`.",
                err=True,
            )
            sys.exit(1)

        try:
            provider = _resolve_mcp_oauth_provider_config(
                view=view,
                scopes=scopes,
                authorize_url=authorize_url,
                token_url=token_url,
                client_id=client_id,
                redirect_port=redirect_port,
            )
        except MCPOAuthFlowError as e:
            click.echo(f"MCP auth login failed: {e}", err=True)
            click.echo(
                "Fallback: use `loom mcp auth login <alias> --manual-token --access-token ...`.",
                err=True,
            )
            sys.exit(1)

        engine = OAuthEngine()
        try:
            started = engine.start_auth(
                provider=provider,
                preferred_port=max(1, int(redirect_port)),
                open_browser=not no_browser,
                allow_manual_fallback=True,
            )
            click.echo("MCP OAuth login URL:")
            click.echo(started.authorization_url)
            if started.callback_mode == "manual":
                click.echo(
                    "Loopback callback unavailable; using manual callback mode.",
                    err=True,
                )
            if started.browser_error:
                click.echo(
                    f"Browser open warning: {started.browser_error}",
                    err=True,
                )

            manual_input = str(callback_code or "").strip()
            if not manual_input and (no_browser or started.callback_mode == "manual"):
                manual_input = click.prompt(
                    "Paste callback URL or authorization code",
                    hide_input=False,
                ).strip()
            if manual_input:
                engine.submit_callback_input(
                    state=started.state,
                    raw_input=manual_input,
                )

            callback = engine.await_callback(
                state=started.state,
                timeout_seconds=max(1, int(timeout_seconds)),
            )
            token_payload = engine.finish_auth(
                provider=provider,
                state=started.state,
                callback=callback,
                timeout_seconds=max(1, int(timeout_seconds)),
            )
        except OAuthEngineError as e:
            click.echo(
                f"MCP auth login failed ({e.reason_code}): {e}",
                err=True,
            )
            click.echo(
                "Fallback: use `loom mcp auth login <alias> --manual-token --access-token ...`.",
                err=True,
            )
            sys.exit(1)
        finally:
            engine.shutdown()

        access_token_value = str(token_payload.get("access_token", "")).strip()
        if not access_token_value:
            click.echo("MCP auth login failed: token response missing access_token.", err=True)
            sys.exit(1)
        store_scopes = _merge_oauth_scopes(
            provider.scopes,
            scopes,
            token_payload.get("scope"),
        )
        expires_at_unix = _parse_expiry_epoch_from_token_payload(token_payload)
        refresh_token_value = str(token_payload.get("refresh_token", "")).strip() or refresh_token
        token_type_value = str(token_payload.get("token_type", "")).strip() or token_type
        client_secret = str(dict(provider.token_params).get("client_secret", "")).strip()

        try:
            path = upsert_mcp_oauth_token(
                alias=clean_alias,
                server=view.server,
                access_token=access_token_value,
                refresh_token=refresh_token_value,
                token_type=token_type_value,
                scopes=store_scopes,
                expires_at_unix=expires_at_unix,
                token_endpoint=provider.token_endpoint,
                authorization_endpoint=provider.authorization_endpoint,
                client_id=provider.client_id,
                obtained_via="browser_pkce",
                extra_fields={"client_secret": client_secret} if client_secret else None,
                store_path=oauth_store_path,
            )
        except MCPOAuthStoreError as e:
            click.echo(f"MCP auth login failed: {e}", err=True)
            sys.exit(1)
        click.echo(f"Stored MCP OAuth token for '{clean_alias}' in {path}")

    @mcp_auth.command(name="status")
    @click.argument("alias", required=False)
    @click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
    @click.option(
        "--oauth-store",
        "oauth_store_path",
        type=click.Path(path_type=Path),
        default=None,
        help="Override OAuth token store path.",
    )
    @click.pass_context
    def mcp_auth_status(
        ctx: click.Context,
        alias: str | None,
        as_json: bool,
        oauth_store_path: Path | None,
    ) -> None:
        """Show OAuth token readiness for MCP aliases."""
        manager = _mcp_manager(ctx)
        try:
            views = manager.list_views()
        except MCPConfigManagerError as e:
            click.echo(f"MCP auth status failed: {e}", err=True)
            sys.exit(1)

        selected: list[MCPServerView] = []
        if alias is not None:
            clean_alias = ensure_valid_alias(alias)
            view = manager.get_view(clean_alias)
            if view is None:
                click.echo(f"MCP server not found: {clean_alias}", err=True)
                sys.exit(1)
            selected = [view]
        else:
            selected = [view for view in views if view.server.oauth.enabled]

        payload = []
        for view in selected:
            state = oauth_state_for_alias(
                view.alias,
                server=view.server,
                store_path=oauth_store_path,
            )
            payload.append(
                {
                    "alias": view.alias,
                    "type": view.server.type,
                    "oauth_enabled": view.server.oauth.enabled,
                    "state": state["state"],
                    "expired": bool(state.get("expired", False)),
                    "has_token": bool(state.get("has_token", False)),
                    "expires_at": state.get("expires_at"),
                    "scopes": list(state.get("scopes", []) or []),
                    "last_failure_reason": str(state.get("last_failure_reason", "") or ""),
                    "last_failure_at": state.get("last_failure_at"),
                }
            )

        if as_json:
            click.echo(json.dumps({"aliases": payload}, indent=2, sort_keys=True))
            return
        if not payload:
            click.echo("No oauth-enabled MCP aliases configured.")
            return
        click.echo("MCP OAuth status:")
        for item in payload:
            click.echo(
                f"  {item['alias']:16} state={item['state']:8} "
                f"expired={'yes' if item['expired'] else 'no'}"
            )
            if item["expires_at"]:
                click.echo(f"    expires_at: {item['expires_at']}")
            if item.get("last_failure_reason"):
                click.echo(f"    last_failure: {item['last_failure_reason']}")

    @mcp_auth.command(name="logout")
    @click.argument("alias")
    @click.option(
        "--oauth-store",
        "oauth_store_path",
        type=click.Path(path_type=Path),
        default=None,
        help="Override OAuth token store path.",
    )
    @click.pass_context
    def mcp_auth_logout(
        ctx: click.Context,
        alias: str,
        oauth_store_path: Path | None,
    ) -> None:
        """Delete stored OAuth token for one MCP alias."""
        try:
            clean_alias = ensure_valid_alias(alias)
            manager = _mcp_manager(ctx)
            view = manager.get_view(clean_alias)
            if view is None:
                raise MCPConfigManagerError(f"MCP server not found: {clean_alias}")
            path = remove_mcp_oauth_token(
                clean_alias,
                server=view.server,
                store_path=oauth_store_path,
            )
        except (MCPConfigManagerError, MCPOAuthStoreError) as e:
            click.echo(f"MCP auth logout failed: {e}", err=True)
            sys.exit(1)
        click.echo(f"Removed MCP OAuth token for '{clean_alias}' from {path}")

    @mcp_auth.command(name="refresh")
    @click.argument("alias")
    @click.option("--access-token", default=None, help="New access token.")
    @click.option(
        "--access-token-env",
        default=None,
        help="Environment variable containing new access token.",
    )
    @click.option("--refresh-token", default="", help="Optional replacement refresh token.")
    @click.option("--token-type", default="Bearer", show_default=True)
    @click.option("--scope", "scopes", multiple=True, help="OAuth scope. Repeatable.")
    @click.option("--expires-in", type=int, default=None, help="Token lifetime in seconds.")
    @click.option("--token-url", default=None, help="OAuth token endpoint override for refresh.")
    @click.option("--client-id", default=None, help="OAuth client id override for refresh.")
    @click.option(
        "--timeout-seconds",
        type=int,
        default=15,
        show_default=True,
        help="HTTP timeout for refresh grant requests.",
    )
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Attempt refresh even when token is not yet expired.",
    )
    @click.option(
        "--oauth-store",
        "oauth_store_path",
        type=click.Path(path_type=Path),
        default=None,
        help="Override OAuth token store path.",
    )
    @click.pass_context
    def mcp_auth_refresh(
        ctx: click.Context,
        alias: str,
        access_token: str | None,
        access_token_env: str | None,
        refresh_token: str,
        token_type: str,
        scopes: tuple[str, ...],
        expires_in: int | None,
        token_url: str | None,
        client_id: str | None,
        timeout_seconds: int,
        force: bool,
        oauth_store_path: Path | None,
    ) -> None:
        """Refresh/update OAuth token payload for one MCP alias."""
        manager = _mcp_manager(ctx)
        try:
            clean_alias = ensure_valid_alias(alias)
            view = manager.get_view(clean_alias)
        except MCPConfigManagerError as e:
            click.echo(f"MCP auth refresh failed: {e}", err=True)
            sys.exit(1)
        if view is None:
            click.echo(f"MCP server not found: {clean_alias}", err=True)
            sys.exit(1)
        try:
            _require_mcp_server_approval(view)
        except MCPOAuthFlowError as e:
            click.echo(f"MCP auth refresh failed: {e}", err=True)
            sys.exit(1)

        token = _resolve_access_token(
            access_token=access_token,
            access_token_env=access_token_env,
        )
        if token:
            expires_at_unix: int | None = None
            if expires_in is not None:
                expires_at_unix = int(time.time()) + max(1, int(expires_in))
            merged_scopes = _merge_oauth_scopes(view.server.oauth.scopes, scopes)
            try:
                path = upsert_mcp_oauth_token(
                    alias=clean_alias,
                    server=view.server,
                    access_token=token,
                    refresh_token=refresh_token,
                    token_type=token_type,
                    scopes=merged_scopes,
                    expires_at_unix=expires_at_unix,
                    store_path=oauth_store_path,
                    obtained_via="manual_refresh",
                )
            except MCPOAuthStoreError as e:
                click.echo(f"MCP auth refresh failed: {e}", err=True)
                sys.exit(1)
            click.echo(f"Refreshed MCP OAuth token for '{clean_alias}' in {path}")
            return

        refreshed = refresh_mcp_oauth_token(
            clean_alias,
            server=view.server,
            store_path=oauth_store_path,
            token_endpoint=token_url,
            client_id=client_id,
            timeout_seconds=max(1, int(timeout_seconds)),
            force=force,
        )
        if not refreshed.refreshed:
            click.echo(
                f"MCP auth refresh failed: {refreshed.reason or 'refresh not completed.'}",
                err=True,
            )
            click.echo(
                "Manual fallback: pass --access-token or --access-token-env.",
                err=True,
            )
            sys.exit(1)
        click.echo(f"Refreshed MCP OAuth token for '{clean_alias}'.")

    return mcp_auth
