"""Shared OAuth lifecycle helpers for `/auth` profiles."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from loom.auth.config import AuthProfile
from loom.auth.runtime import oauth_provider_config_for_profile
from loom.auth.secrets import SecretResolutionError, SecretResolver
from loom.oauth.engine import (
    OAuthEngine,
    OAuthEngineError,
    OAuthProviderConfig,
    OAuthStartResult,
)


class OAuthProfileError(Exception):
    """Raised when `/auth` OAuth profile lifecycle operations fail."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class OAuthProfileTokenState:
    """Resolved OAuth token state for one auth profile."""

    state: str
    has_token: bool
    expired: bool
    expires_at: int | None
    token_type: str | None = None
    scopes: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class OAuthProfileRefreshResult:
    """Result payload for OAuth refresh lifecycle action."""

    token_ref: str
    expires_at: int | None
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class OAuthProfileLoginResult:
    """Result payload for OAuth browser login lifecycle action."""

    token_ref: str
    callback_mode: str
    authorization_url: str
    expires_at: int | None
    scopes: tuple[str, ...]
    browser_warning: str = ""


@dataclass(frozen=True)
class OAuthProfileStoreResult:
    """Stored token payload summary for split OAuth flows."""

    token_ref: str
    expires_at: int | None
    scopes: tuple[str, ...]


def _ensure_oauth_profile(profile: AuthProfile) -> None:
    mode = str(getattr(profile, "mode", "") or "").strip().lower()
    if mode not in {"oauth2_pkce", "oauth2_device"}:
        raise OAuthProfileError(
            "profile_not_oauth",
            (
                f"Profile {profile.profile_id!r} mode {profile.mode!r} "
                "does not support OAuth lifecycle actions."
            ),
        )


def _merge_oauth_scopes(*scope_groups: object) -> tuple[str, ...]:
    merged: list[str] = []
    for group in scope_groups:
        if isinstance(group, str):
            candidates = [item for item in group.replace(",", " ").split(" ") if item.strip()]
        elif isinstance(group, (list, tuple, set)):
            candidates = [str(item).strip() for item in group]
        else:
            continue
        for candidate in candidates:
            scope = str(candidate or "").strip()
            if scope:
                merged.append(scope)
    return tuple(dict.fromkeys(merged))


def _parse_expiry_epoch(raw: object) -> int | None:
    if raw in (None, ""):
        return None
    value: float
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            normalized = text
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            value = parsed.timestamp()
    if value > 10_000_000_000:
        value = value / 1000.0
    if value <= 0:
        return None
    return int(value)


def _parse_expiry_from_payload(
    payload: dict[str, Any],
    *,
    now_unix: int,
) -> int | None:
    for key in ("expires_at", "expires_at_epoch", "expires_on"):
        parsed = _parse_expiry_epoch(payload.get(key))
        if parsed is not None:
            return parsed

    expires_in = payload.get("expires_in")
    if expires_in not in (None, ""):
        try:
            seconds = max(1, int(expires_in))
        except (TypeError, ValueError):
            seconds = 0
        if seconds > 0:
            obtained_at = _parse_expiry_epoch(payload.get("obtained_at"))
            baseline = obtained_at if obtained_at is not None else now_unix
            return int(baseline + seconds)
    return None


def _resolve_token_ref(profile: AuthProfile) -> str:
    token_ref = str(getattr(profile, "token_ref", "") or "").strip()
    if not token_ref:
        raise OAuthProfileError(
            "token_ref_missing",
            f"Profile {profile.profile_id!r} has no token_ref.",
        )
    return token_ref


def _resolve_token_value(
    *,
    profile: AuthProfile,
    resolver: SecretResolver,
    token_ref: str,
) -> str:
    try:
        return resolver.resolve(token_ref).strip()
    except SecretResolutionError as e:
        raise OAuthProfileError(
            "token_resolve_failed",
            (
                f"Failed to resolve token_ref for profile {profile.profile_id!r}: {e}"
            ),
        ) from e


def _parse_json_payload(
    *,
    profile: AuthProfile,
    token_value: str,
) -> dict[str, Any] | None:
    if not token_value:
        return None
    try:
        parsed = json.loads(token_value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        raise OAuthProfileError(
            "token_payload_invalid",
            (
                f"Profile {profile.profile_id!r} token payload must be a JSON object."
            ),
        )
    return dict(parsed)


def oauth_state_for_profile(
    profile: AuthProfile,
    *,
    resolver: SecretResolver | None = None,
    now_unix: int | None = None,
) -> OAuthProfileTokenState:
    """Resolve current OAuth token state for one profile."""
    mode = str(getattr(profile, "mode", "") or "").strip().lower()
    if mode not in {"oauth2_pkce", "oauth2_device"}:
        return OAuthProfileTokenState(
            state="unsupported",
            has_token=False,
            expired=False,
            expires_at=None,
            reason=(
                f"Profile {profile.profile_id!r} mode {profile.mode!r} is not OAuth."
            ),
        )

    token_ref = str(getattr(profile, "token_ref", "") or "").strip()
    if not token_ref:
        return OAuthProfileTokenState(
            state="missing",
            has_token=False,
            expired=False,
            expires_at=None,
            reason="token_ref is missing.",
        )

    secret_resolver = resolver or SecretResolver()
    try:
        token_value = secret_resolver.resolve(token_ref).strip()
    except SecretResolutionError as e:
        return OAuthProfileTokenState(
            state="missing",
            has_token=False,
            expired=False,
            expires_at=None,
            reason=str(e),
        )
    if not token_value:
        return OAuthProfileTokenState(
            state="missing",
            has_token=False,
            expired=False,
            expires_at=None,
            reason="Stored token value is empty.",
        )

    parsed = _parse_json_payload(profile=profile, token_value=token_value)
    if parsed is None:
        return OAuthProfileTokenState(
            state="ready",
            has_token=True,
            expired=False,
            expires_at=None,
            reason="token payload is not JSON; expiry cannot be evaluated",
        )

    access_token = str(parsed.get("access_token", "")).strip()
    if not access_token:
        return OAuthProfileTokenState(
            state="invalid",
            has_token=False,
            expired=False,
            expires_at=None,
            reason="token payload is missing access_token",
        )

    now = int(now_unix if now_unix is not None else time.time())
    expires_at = _parse_expiry_from_payload(parsed, now_unix=now)
    expired = bool(expires_at is not None and expires_at <= (now + 30))
    scopes = _merge_oauth_scopes(
        parsed.get("scopes", ()),
        parsed.get("scope", ""),
    )
    return OAuthProfileTokenState(
        state="expired" if expired else "ready",
        has_token=True,
        expired=expired,
        expires_at=expires_at,
        token_type=str(parsed.get("token_type", "") or "").strip() or None,
        scopes=scopes,
    )


def logout_oauth_profile(
    profile: AuthProfile,
    *,
    resolver: SecretResolver | None = None,
) -> str:
    """Clear stored OAuth token payload for one profile."""
    _ensure_oauth_profile(profile)
    token_ref = _resolve_token_ref(profile)
    secret_resolver = resolver or SecretResolver()
    try:
        secret_resolver.validate_writable(token_ref)
        secret_resolver.store(token_ref, "")
    except SecretResolutionError as e:
        raise OAuthProfileError(
            "token_store_failed",
            str(e),
        ) from e
    return token_ref


def refresh_oauth_profile(
    profile: AuthProfile,
    *,
    resolver: SecretResolver | None = None,
    token_endpoint: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    scopes: tuple[str, ...] = (),
    timeout_seconds: int = 15,
) -> OAuthProfileRefreshResult:
    """Run refresh_token grant for one `/auth` OAuth profile."""
    _ensure_oauth_profile(profile)
    token_ref = _resolve_token_ref(profile)
    secret_resolver = resolver or SecretResolver()
    try:
        secret_resolver.validate_writable(token_ref)
    except SecretResolutionError as e:
        raise OAuthProfileError("token_ref_not_writable", str(e)) from e

    token_value = _resolve_token_value(
        profile=profile,
        resolver=secret_resolver,
        token_ref=token_ref,
    )
    if not token_value:
        raise OAuthProfileError(
            "token_missing",
            f"Profile {profile.profile_id!r} has no stored OAuth token payload.",
        )

    payload = _parse_json_payload(profile=profile, token_value=token_value)
    if payload is None:
        raise OAuthProfileError(
            "token_payload_invalid",
            "Stored token payload must be JSON for refresh.",
        )

    refresh_token = str(payload.get("refresh_token", "") or "").strip()
    if not refresh_token:
        raise OAuthProfileError(
            "refresh_token_missing",
            "Stored token payload is missing refresh_token.",
        )

    provider_cfg = oauth_provider_config_for_profile(profile)
    resolved_token_endpoint = str(
        token_endpoint
        or (provider_cfg.token_endpoint if provider_cfg is not None else "")
        or payload.get("token_endpoint", "")
        or ""
    ).strip()
    resolved_client_id = str(
        client_id
        or (provider_cfg.client_id if provider_cfg is not None else "")
        or payload.get("client_id", "")
        or ""
    ).strip()
    resolved_client_secret = str(
        client_secret
        or profile.metadata.get("oauth_client_secret", "")
        or payload.get("client_secret", "")
        or ""
    ).strip()

    if not resolved_token_endpoint or not resolved_client_id:
        raise OAuthProfileError(
            "refresh_config_missing",
            "Refresh metadata missing token_endpoint/client_id.",
        )

    request_data: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": resolved_client_id,
    }
    if resolved_client_secret:
        request_data["client_secret"] = resolved_client_secret

    try:
        response = httpx.post(
            resolved_token_endpoint,
            data=request_data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=max(1, int(timeout_seconds)),
        )
    except Exception as e:
        raise OAuthProfileError(
            "refresh_request_failed",
            f"Refresh request failed: {e}",
        ) from e

    body: dict[str, Any] = {}
    if response.content:
        try:
            parsed = response.json()
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            body = dict(parsed)

    if response.status_code >= 400:
        detail = str(
            body.get("error_description")
            or body.get("error")
            or response.text
            or "refresh failed"
        ).strip()
        raise OAuthProfileError("refresh_failed", detail)

    access_token = str(body.get("access_token", "") or "").strip()
    if not access_token:
        raise OAuthProfileError(
            "refresh_payload_invalid",
            "Refresh response missing access_token.",
        )

    now = int(time.time())
    merged_scopes = _merge_oauth_scopes(
        payload.get("scopes", ()),
        payload.get("scope", ""),
        profile.scopes,
        (provider_cfg.scopes if provider_cfg is not None else ()),
        scopes,
        body.get("scopes", ()),
        body.get("scope", ""),
    )
    next_refresh_token = str(body.get("refresh_token", "") or "").strip() or refresh_token
    token_type = str(
        body.get("token_type", payload.get("token_type", "Bearer")) or "Bearer"
    ).strip() or "Bearer"

    next_payload = dict(payload)
    next_payload.update(body)
    next_payload["access_token"] = access_token
    next_payload["refresh_token"] = next_refresh_token
    next_payload["token_type"] = token_type
    next_payload["obtained_via"] = "refresh_token"
    next_payload["obtained_at"] = now
    next_payload["token_endpoint"] = resolved_token_endpoint
    next_payload["client_id"] = resolved_client_id
    if merged_scopes:
        next_payload["scope"] = " ".join(merged_scopes)
        next_payload["scopes"] = list(merged_scopes)

    expires_at = _parse_expiry_from_payload(next_payload, now_unix=now)
    if expires_at is not None:
        next_payload["expires_at"] = expires_at

    try:
        secret_resolver.store(
            token_ref,
            json.dumps(next_payload, sort_keys=True, separators=(",", ":")),
        )
    except SecretResolutionError as e:
        raise OAuthProfileError(
            "token_store_failed",
            str(e),
        ) from e

    return OAuthProfileRefreshResult(
        token_ref=token_ref,
        expires_at=expires_at,
        scopes=merged_scopes,
    )


def store_oauth_profile_token_payload(
    profile: AuthProfile,
    *,
    token_payload: dict[str, Any],
    provider_scopes: tuple[str, ...] = (),
    scopes: tuple[str, ...] = (),
    resolver: SecretResolver | None = None,
    obtained_via: str = "browser_pkce",
) -> OAuthProfileStoreResult:
    """Persist a finished OAuth token payload for one `/auth` profile."""
    _ensure_oauth_profile(profile)
    token_ref = _resolve_token_ref(profile)
    secret_resolver = resolver or SecretResolver()
    try:
        secret_resolver.validate_writable(token_ref)
    except SecretResolutionError as e:
        raise OAuthProfileError("token_ref_not_writable", str(e)) from e

    access_token_value = str(token_payload.get("access_token", "")).strip()
    if not access_token_value:
        raise OAuthProfileError(
            "token_payload_invalid",
            "Token response missing access_token.",
        )

    stored_payload = dict(token_payload)
    stored_payload["access_token"] = access_token_value
    stored_payload.setdefault("token_type", "Bearer")
    merged_scopes = _merge_oauth_scopes(
        provider_scopes,
        profile.scopes,
        scopes,
        token_payload.get("scope"),
        token_payload.get("scopes"),
    )
    if merged_scopes:
        stored_payload["scope"] = " ".join(merged_scopes)
        stored_payload["scopes"] = list(merged_scopes)
    expires_at_unix = _parse_expiry_from_payload(
        stored_payload,
        now_unix=int(time.time()),
    )
    if expires_at_unix is not None:
        stored_payload["expires_at"] = expires_at_unix
    stored_payload["obtained_via"] = str(obtained_via or "browser_pkce").strip()
    stored_payload["obtained_at"] = int(time.time())

    try:
        secret_resolver.store(
            token_ref,
            json.dumps(stored_payload, sort_keys=True, separators=(",", ":")),
        )
    except SecretResolutionError as e:
        raise OAuthProfileError("token_store_failed", str(e)) from e

    return OAuthProfileStoreResult(
        token_ref=token_ref,
        expires_at=expires_at_unix,
        scopes=merged_scopes,
    )


def login_oauth_profile(
    profile: AuthProfile,
    *,
    scopes: tuple[str, ...] = (),
    authorize_url: str | None = None,
    token_url: str | None = None,
    client_id: str | None = None,
    redirect_port: int = 8765,
    timeout_seconds: int = 180,
    no_browser: bool = False,
    callback_code: str | None = None,
    callback_prompt: Any = None,
    on_start: Callable[[OAuthStartResult], None] | None = None,
    resolver: SecretResolver | None = None,
) -> OAuthProfileLoginResult:
    """Run browser OAuth login for one `/auth` OAuth profile."""
    _ensure_oauth_profile(profile)
    token_ref = _resolve_token_ref(profile)
    secret_resolver = resolver or SecretResolver()
    try:
        secret_resolver.validate_writable(token_ref)
    except SecretResolutionError as e:
        raise OAuthProfileError("token_ref_not_writable", str(e)) from e

    base_provider = oauth_provider_config_for_profile(profile)
    if base_provider is None:
        raise OAuthProfileError(
            "oauth_metadata_missing",
            (
                f"Profile {profile.profile_id!r} is missing OAuth metadata. "
                "Required: oauth_authorization_endpoint, oauth_token_endpoint, oauth_client_id."
            ),
        )

    provider = OAuthProviderConfig(
        authorization_endpoint=str(
            authorize_url or base_provider.authorization_endpoint
        ).strip(),
        token_endpoint=str(token_url or base_provider.token_endpoint).strip(),
        client_id=str(client_id or base_provider.client_id).strip(),
        scopes=_merge_oauth_scopes(
            base_provider.scopes,
            profile.scopes,
            scopes,
        ),
        authorize_params=dict(base_provider.authorize_params),
        token_params=dict(base_provider.token_params),
    )

    engine = OAuthEngine()
    try:
        started = engine.start_auth(
            provider=provider,
            preferred_port=max(1, int(redirect_port)),
            open_browser=not no_browser,
            allow_manual_fallback=True,
        )
        if callable(on_start):
            on_start(started)

        manual_input = str(callback_code or "").strip()
        if not manual_input and (no_browser or started.callback_mode == "manual"):
            if callable(callback_prompt):
                manual_input = str(
                    callback_prompt("Paste callback URL or authorization code")
                ).strip()
            else:
                raise OAuthProfileError(
                    "callback_missing",
                    "Callback URL/code required for manual OAuth completion.",
                )
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
    except OAuthProfileError:
        raise
    except OAuthEngineError as e:
        raise OAuthProfileError(
            e.reason_code,
            str(e),
        ) from e
    finally:
        engine.shutdown()

    stored = store_oauth_profile_token_payload(
        profile,
        token_payload=token_payload,
        provider_scopes=provider.scopes,
        scopes=scopes,
        resolver=secret_resolver,
        obtained_via="browser_pkce",
    )

    return OAuthProfileLoginResult(
        token_ref=stored.token_ref,
        callback_mode=started.callback_mode,
        authorization_url=started.authorization_url,
        expires_at=stored.expires_at,
        scopes=stored.scopes,
        browser_warning=str(getattr(started, "browser_error", "") or "").strip(),
    )
