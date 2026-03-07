"""`loom auth profile ...` CLI commands."""

from __future__ import annotations

import json
import sys

import click

from loom.auth.config import (
    AuthConfigError,
    AuthProfile,
    load_merged_auth_config,
    remove_auth_profile,
    upsert_auth_profile,
)
from loom.auth.oauth_profiles import (
    OAuthProfileError,
    login_oauth_profile,
    logout_oauth_profile,
    oauth_state_for_profile,
    refresh_oauth_profile,
)
from loom.cli.context import _auth_write_path, _merged_auth_config, _parse_kv_pairs
from loom.mcp.config import ensure_valid_alias, ensure_valid_env_key


def attach_auth_profile_commands(
    auth_group: click.Group,
    *,
    auth_list_callback,
    auth_show_callback,
) -> click.Group:
    @auth_group.group(name="profile")
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
        ctx.invoke(auth_list_callback, as_json=as_json, verbose=verbose)

    @auth_profile.command(name="show")
    @click.argument("profile_id")
    @click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
    @click.pass_context
    def auth_profile_show(ctx: click.Context, profile_id: str, as_json: bool) -> None:
        """Alias for `loom auth show`."""
        ctx.invoke(auth_show_callback, profile_id=profile_id, as_json=as_json)

    @auth_profile.command(name="add")
    @click.argument("profile_id")
    @click.option("--provider", required=True, help="Provider id (e.g. notion).")
    @click.option(
        "--mode",
        required=True,
        help="Credential mode (api_key, oauth2_pkce, env_passthrough, ...).",
    )
    @click.option("--label", "account_label", default="", help="Human-friendly account label.")
    @click.option("--mcp-server", default="", help="Optional MCP alias binding for this profile.")
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
        mcp_server: str,
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
            mcp_server=str(mcp_server or "").strip(),
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
        if profile.mcp_server:
            try:
                ensure_valid_alias(profile.mcp_server)
            except Exception as e:
                click.echo(f"Invalid --mcp-server value: {e}", err=True)
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
    @click.option("--mcp-server", default=None, help="MCP alias binding.")
    @click.option(
        "--clear-mcp-server",
        is_flag=True,
        default=False,
        help="Clear MCP alias binding.",
    )
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
        mcp_server: str | None,
        clear_mcp_server: bool,
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

        next_mcp_server = current.mcp_server
        if clear_mcp_server:
            next_mcp_server = ""
        elif mcp_server is not None:
            next_mcp_server = str(mcp_server).strip()
            if next_mcp_server:
                try:
                    ensure_valid_alias(next_mcp_server)
                except Exception as e:
                    click.echo(f"Invalid --mcp-server value: {e}", err=True)
                    sys.exit(1)

        updated_profile = AuthProfile(
            profile_id=current.profile_id,
            provider=current.provider if provider is None else str(provider).strip(),
            mode=current.mode if mode is None else str(mode).strip(),
            account_label=(
                current.account_label
                if account_label is None
                else str(account_label).strip()
            ),
            mcp_server=next_mcp_server,
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

    @auth_profile.command(name="login")
    @click.argument("profile_id")
    @click.option("--scope", "scopes", multiple=True, help="OAuth scope. Repeatable.")
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
        help="OAuth client id override.",
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
    @click.pass_context
    def auth_profile_login(
        ctx: click.Context,
        profile_id: str,
        scopes: tuple[str, ...],
        authorize_url: str | None,
        token_url: str | None,
        client_id: str | None,
        redirect_port: int,
        timeout_seconds: int,
        no_browser: bool,
        callback_code: str | None,
    ) -> None:
        """Run browser OAuth login for one `/auth` OAuth profile."""
        clean_profile_id = str(profile_id or "").strip()
        if not clean_profile_id:
            click.echo("Profile id cannot be empty.", err=True)
            sys.exit(1)

        merged = _merged_auth_config(ctx)
        profile = merged.config.profiles.get(clean_profile_id)
        if profile is None:
            click.echo(f"Auth profile not found: {clean_profile_id}", err=True)
            sys.exit(1)

        def _emit_login_start(started) -> None:
            click.echo("Auth profile OAuth login URL:")
            click.echo(started.authorization_url)
            if str(getattr(started, "callback_mode", "")).strip() == "manual":
                click.echo(
                    "Loopback callback unavailable; using manual callback mode.",
                    err=True,
                )
            browser_warning = str(getattr(started, "browser_error", "") or "").strip()
            if browser_warning:
                click.echo(f"Browser open warning: {browser_warning}", err=True)

        try:
            result = login_oauth_profile(
                profile,
                scopes=scopes,
                authorize_url=authorize_url,
                token_url=token_url,
                client_id=client_id,
                redirect_port=redirect_port,
                timeout_seconds=timeout_seconds,
                no_browser=no_browser,
                callback_code=callback_code,
                callback_prompt=(
                    lambda prompt: click.prompt(prompt, hide_input=False).strip()
                ),
                on_start=_emit_login_start,
            )
        except OAuthProfileError as e:
            click.echo(
                f"Auth profile login failed ({e.reason_code}): {e}",
                err=True,
            )
            if e.reason_code in {"token_ref_not_writable", "oauth_metadata_missing"}:
                click.echo(
                    "Set token_ref to keychain://... and ensure OAuth metadata is present.",
                    err=True,
                )
            sys.exit(1)
        click.echo(
            f"Stored OAuth token for auth profile '{clean_profile_id}' in {result.token_ref}"
        )
        if result.expires_at is not None:
            click.echo(f"Expires at (unix): {result.expires_at}")
        if result.scopes:
            click.echo(f"Scopes: {', '.join(result.scopes)}")

    @auth_profile.command(name="status")
    @click.argument("profile_id")
    @click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
    @click.pass_context
    def auth_profile_status(ctx: click.Context, profile_id: str, as_json: bool) -> None:
        """Show OAuth token state for one `/auth` OAuth profile."""
        clean_profile_id = str(profile_id or "").strip()
        if not clean_profile_id:
            click.echo("Profile id cannot be empty.", err=True)
            sys.exit(1)

        merged = _merged_auth_config(ctx)
        profile = merged.config.profiles.get(clean_profile_id)
        if profile is None:
            click.echo(f"Auth profile not found: {clean_profile_id}", err=True)
            sys.exit(1)

        state = oauth_state_for_profile(profile)
        payload = {
            "profile_id": clean_profile_id,
            "provider": profile.provider,
            "mode": profile.mode,
            "token_ref": profile.token_ref,
            "state": state.state,
            "has_token": state.has_token,
            "expired": state.expired,
            "expires_at": state.expires_at,
            "token_type": state.token_type,
            "scopes": list(state.scopes),
            "reason": state.reason,
        }
        if as_json:
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return

        click.echo(f"Profile: {clean_profile_id}")
        click.echo(f"Provider: {profile.provider}")
        click.echo(f"Mode: {profile.mode}")
        click.echo(f"Token ref: {profile.token_ref or '-'}")
        click.echo(f"OAuth state: {state.state}")
        click.echo(f"Has token: {'yes' if state.has_token else 'no'}")
        click.echo(f"Expired: {'yes' if state.expired else 'no'}")
        click.echo(f"Expires at: {state.expires_at if state.expires_at is not None else '-'}")
        click.echo(f"Token type: {state.token_type or '-'}")
        click.echo(f"Scopes: {', '.join(state.scopes) if state.scopes else '-'}")
        if state.reason:
            click.echo(f"Reason: {state.reason}")

    @auth_profile.command(name="logout")
    @click.argument("profile_id")
    @click.pass_context
    def auth_profile_logout(ctx: click.Context, profile_id: str) -> None:
        """Clear stored OAuth token for one `/auth` OAuth profile."""
        clean_profile_id = str(profile_id or "").strip()
        if not clean_profile_id:
            click.echo("Profile id cannot be empty.", err=True)
            sys.exit(1)

        merged = _merged_auth_config(ctx)
        profile = merged.config.profiles.get(clean_profile_id)
        if profile is None:
            click.echo(f"Auth profile not found: {clean_profile_id}", err=True)
            sys.exit(1)

        try:
            token_ref = logout_oauth_profile(profile)
        except OAuthProfileError as e:
            click.echo(f"Auth profile logout failed ({e.reason_code}): {e}", err=True)
            sys.exit(1)

        click.echo(
            f"Cleared OAuth token for auth profile '{clean_profile_id}' in {token_ref}"
        )

    @auth_profile.command(name="refresh")
    @click.argument("profile_id")
    @click.option("--scope", "scopes", multiple=True, help="OAuth scope hint. Repeatable.")
    @click.option(
        "--token-url",
        default=None,
        help="OAuth token endpoint override.",
    )
    @click.option(
        "--client-id",
        default=None,
        help="OAuth client id override.",
    )
    @click.option(
        "--client-secret",
        default=None,
        help="OAuth client secret override.",
    )
    @click.option(
        "--timeout-seconds",
        type=int,
        default=15,
        show_default=True,
        help="HTTP timeout for refresh request.",
    )
    @click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
    @click.pass_context
    def auth_profile_refresh(
        ctx: click.Context,
        profile_id: str,
        scopes: tuple[str, ...],
        token_url: str | None,
        client_id: str | None,
        client_secret: str | None,
        timeout_seconds: int,
        as_json: bool,
    ) -> None:
        """Refresh OAuth token for one `/auth` OAuth profile."""
        clean_profile_id = str(profile_id or "").strip()
        if not clean_profile_id:
            click.echo("Profile id cannot be empty.", err=True)
            sys.exit(1)

        merged = _merged_auth_config(ctx)
        profile = merged.config.profiles.get(clean_profile_id)
        if profile is None:
            click.echo(f"Auth profile not found: {clean_profile_id}", err=True)
            sys.exit(1)

        try:
            result = refresh_oauth_profile(
                profile,
                token_endpoint=token_url,
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
                timeout_seconds=timeout_seconds,
            )
        except OAuthProfileError as e:
            click.echo(f"Auth profile refresh failed ({e.reason_code}): {e}", err=True)
            sys.exit(1)

        payload = {
            "profile_id": clean_profile_id,
            "token_ref": result.token_ref,
            "expires_at": result.expires_at,
            "scopes": list(result.scopes),
            "refreshed": True,
        }
        if as_json:
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
            return

        click.echo(
            f"Refreshed OAuth token for auth profile '{clean_profile_id}' in {result.token_ref}"
        )
        if result.expires_at is not None:
            click.echo(f"Expires at (unix): {result.expires_at}")
        if result.scopes:
            click.echo(f"Scopes: {', '.join(result.scopes)}")

    return auth_profile
