"""MCP OAuth token storage, readiness, and refresh helpers."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx


class MCPOAuthStoreError(Exception):
    """Raised when MCP OAuth token state cannot be read or written."""


class MCPOAuthFlowError(Exception):
    """Raised when MCP OAuth browser flow metadata cannot be resolved safely."""


@dataclass(frozen=True)
class MCPOAuthProviderConfig:
    """Resolved OAuth provider details for one MCP alias login flow."""

    authorization_endpoint: str
    token_endpoint: str
    client_id: str
    scopes: tuple[str, ...]


@dataclass(frozen=True)
class MCPOAuthRefreshResult:
    """Result of one MCP alias refresh-token attempt."""

    status: str
    reason: str = ""
    refreshed: bool = False


@dataclass(frozen=True)
class MCPOAuthReadiness:
    """Resolved readiness for runtime transport gating."""

    ready: bool
    state: str
    reason: str = ""
    refreshed: bool = False


_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(access_token|refresh_token|id_token|client_secret)\b\s*[:=]\s*([^\s,;]+)"
        ),
        r"\1=<redacted>",
    ),
    (
        re.compile(
            r"(?i)\b(authorization|proxy-authorization)\b\s*[:=]\s*"
            r"([^\s,;]+(?:\s+[^\s,;]+)?)"
        ),
        r"\1=<redacted>",
    ),
    (
        re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]+"),
        r"\1 <redacted>",
    ),
)

_REFRESH_ATTEMPT_LOCK = threading.RLock()
_REFRESH_LAST_ATTEMPT: dict[tuple[str, str], float] = {}
_DEFAULT_REFRESH_COOLDOWN_SECONDS = 30


def default_mcp_oauth_store_path() -> Path:
    """Default MCP OAuth token store path."""
    return Path.home() / ".loom" / "mcp_oauth_tokens.json"


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        try:
            import fcntl  # POSIX only

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            # Best effort on non-POSIX platforms.
            pass
        try:
            yield
        finally:
            try:
                import fcntl  # POSIX only

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _coerce_store_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"aliases": {}}
    aliases_raw = raw.get("aliases", {})
    aliases: dict[str, dict[str, Any]] = {}
    if isinstance(aliases_raw, dict):
        for key, value in aliases_raw.items():
            alias = str(key or "").strip()
            if not alias or not isinstance(value, dict):
                continue
            aliases[alias] = dict(value)
    return {"aliases": aliases}


def load_mcp_oauth_store(path: Path | None = None) -> dict[str, Any]:
    """Load token store payload."""
    target = (path or default_mcp_oauth_store_path()).expanduser().resolve()
    if not target.exists():
        return {"aliases": {}}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise MCPOAuthStoreError(f"Invalid JSON in {target}: {e}") from e
    except OSError as e:
        raise MCPOAuthStoreError(f"Cannot read {target}: {e}") from e
    return _coerce_store_payload(raw)


def _write_store(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def redact_oauth_error_text(value: object) -> str:
    """Best-effort redaction for OAuth-related error surfaces."""
    text = str(value or "").strip()
    if not text:
        return ""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def upsert_mcp_oauth_token(
    *,
    alias: str,
    access_token: str,
    refresh_token: str = "",
    token_type: str = "Bearer",
    scopes: list[str] | None = None,
    expires_at_unix: int | None = None,
    token_endpoint: str = "",
    authorization_endpoint: str = "",
    client_id: str = "",
    obtained_via: str = "",
    extra_fields: dict[str, Any] | None = None,
    store_path: Path | None = None,
) -> Path:
    """Insert or update one alias token payload."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        raise MCPOAuthStoreError("Alias cannot be empty.")
    token = str(access_token or "").strip()
    if not token:
        raise MCPOAuthStoreError("Access token cannot be empty.")

    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    lock_path = target.with_suffix(target.suffix + ".lock")
    with _file_lock(lock_path):
        store = load_mcp_oauth_store(target)
        aliases = store.setdefault("aliases", {})
        existing = aliases.get(clean_alias)
        payload = dict(existing) if isinstance(existing, dict) else {}

        payload.update(
            {
                "access_token": token,
                "refresh_token": str(refresh_token or "").strip(),
                "token_type": str(token_type or "Bearer").strip() or "Bearer",
                "scopes": [
                    str(scope).strip()
                    for scope in (scopes or [])
                    if str(scope).strip()
                ],
                "obtained_at": int(time.time()),
                "expires_at": (
                    int(expires_at_unix)
                    if expires_at_unix is not None and int(expires_at_unix) > 0
                    else None
                ),
            }
        )

        token_ep = str(token_endpoint or "").strip()
        if token_ep:
            payload["token_endpoint"] = token_ep
        authorization_ep = str(authorization_endpoint or "").strip()
        if authorization_ep:
            payload["authorization_endpoint"] = authorization_ep
        clean_client_id = str(client_id or "").strip()
        if clean_client_id:
            payload["client_id"] = clean_client_id
        via = str(obtained_via or "").strip()
        if via:
            payload["obtained_via"] = via
        if extra_fields:
            payload.update({
                str(key): value
                for key, value in extra_fields.items()
                if str(key).strip()
            })
        # Successful write clears stale failure markers.
        payload.pop("last_failure_reason", None)
        payload.pop("last_failure_at", None)

        aliases[clean_alias] = payload
        _write_store(target, store)
    return target


def _annotate_alias_payload(
    *,
    alias: str,
    store_path: Path | None,
    fields: dict[str, Any],
) -> None:
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return
    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    lock_path = target.with_suffix(target.suffix + ".lock")
    with _file_lock(lock_path):
        store = load_mcp_oauth_store(target)
        aliases = store.setdefault("aliases", {})
        payload = aliases.get(clean_alias)
        if not isinstance(payload, dict):
            return
        payload = dict(payload)
        sanitized_fields = dict(fields)
        if "last_failure_reason" in sanitized_fields:
            sanitized_fields["last_failure_reason"] = redact_oauth_error_text(
                sanitized_fields.get("last_failure_reason", ""),
            )
        payload.update(sanitized_fields)
        aliases[clean_alias] = payload
        _write_store(target, store)


def remove_mcp_oauth_token(alias: str, *, store_path: Path | None = None) -> Path:
    """Delete one alias token entry."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        raise MCPOAuthStoreError("Alias cannot be empty.")
    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    lock_path = target.with_suffix(target.suffix + ".lock")
    with _file_lock(lock_path):
        store = load_mcp_oauth_store(target)
        aliases = store.setdefault("aliases", {})
        aliases.pop(clean_alias, None)
        _write_store(target, store)
    return target


def get_mcp_oauth_token(alias: str, *, store_path: Path | None = None) -> dict[str, Any] | None:
    """Return token payload for one alias."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return None
    store = load_mcp_oauth_store(store_path)
    aliases = store.get("aliases", {})
    if not isinstance(aliases, dict):
        return None
    raw = aliases.get(clean_alias)
    if not isinstance(raw, dict):
        return None
    return dict(raw)


def token_expired(token_payload: dict[str, Any], *, skew_seconds: int = 30) -> bool:
    """Return True when token payload is expired or unusable."""
    access_token = str(token_payload.get("access_token", "")).strip()
    if not access_token:
        return True
    expires_at = token_payload.get("expires_at")
    if expires_at in (None, ""):
        return False
    try:
        expires_unix = int(expires_at)
    except (TypeError, ValueError):
        return True
    return (int(time.time()) + max(0, int(skew_seconds))) >= expires_unix


def oauth_state_for_alias(
    alias: str,
    *,
    store_path: Path | None = None,
) -> dict[str, Any]:
    """Summarize OAuth readiness for one alias."""
    token_payload = get_mcp_oauth_token(alias, store_path=store_path)
    if token_payload is None:
        return {
            "state": "missing",
            "has_token": False,
            "expired": False,
            "expires_at": None,
            "token_type": None,
            "scopes": [],
            "last_failure_reason": "",
            "last_failure_at": None,
        }
    expired = token_expired(token_payload)
    return {
        "state": "expired" if expired else "ready",
        "has_token": True,
        "expired": expired,
        "expires_at": token_payload.get("expires_at"),
        "token_type": token_payload.get("token_type", "Bearer"),
        "scopes": list(token_payload.get("scopes", []) or []),
        "last_failure_reason": redact_oauth_error_text(
            token_payload.get("last_failure_reason", ""),
        ),
        "last_failure_at": token_payload.get("last_failure_at"),
        "last_refresh_at": token_payload.get("last_refresh_at"),
    }


def bearer_auth_header_for_alias(
    alias: str,
    *,
    store_path: Path | None = None,
) -> str | None:
    """Return Authorization header value when a usable token is present."""
    token_payload = get_mcp_oauth_token(alias, store_path=store_path)
    if token_payload is None or token_expired(token_payload):
        return None
    access_token = str(token_payload.get("access_token", "")).strip()
    token_type = str(token_payload.get("token_type", "Bearer")).strip() or "Bearer"
    if not access_token:
        return None
    return f"{token_type} {access_token}"


def _refresh_attempt_key(alias: str, store_path: Path | None) -> tuple[str, str]:
    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    return str(target), str(alias or "").strip()


def _parse_expiry_unix(raw: object) -> int | None:
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value > 0:
        return value
    return None


def refresh_mcp_oauth_token(
    alias: str,
    *,
    store_path: Path | None = None,
    token_endpoint: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    timeout_seconds: int = 15,
    min_interval_seconds: int = _DEFAULT_REFRESH_COOLDOWN_SECONDS,
    force: bool = False,
) -> MCPOAuthRefreshResult:
    """Attempt refresh_token grant for one alias with cooldown protection."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return MCPOAuthRefreshResult(
            status="failed",
            reason="Alias cannot be empty.",
        )

    payload = get_mcp_oauth_token(clean_alias, store_path=store_path)
    if payload is None:
        return MCPOAuthRefreshResult(status="failed", reason="Missing OAuth token.")

    if not force and not token_expired(payload):
        return MCPOAuthRefreshResult(status="skipped", reason="Token is not expired.")

    refresh_token = str(payload.get("refresh_token", "") or "").strip()
    if not refresh_token:
        reason = "Refresh token is missing."
        _annotate_alias_payload(
            alias=clean_alias,
            store_path=store_path,
            fields={
                "last_failure_reason": reason,
                "last_failure_at": int(time.time()),
            },
        )
        return MCPOAuthRefreshResult(status="failed", reason=reason)

    resolved_endpoint = str(token_endpoint or payload.get("token_endpoint", "") or "").strip()
    resolved_client_id = str(client_id or payload.get("client_id", "") or "").strip()
    if not resolved_endpoint or not resolved_client_id:
        reason = "Refresh metadata missing token_endpoint/client_id."
        _annotate_alias_payload(
            alias=clean_alias,
            store_path=store_path,
            fields={
                "last_failure_reason": reason,
                "last_failure_at": int(time.time()),
            },
        )
        return MCPOAuthRefreshResult(status="failed", reason=reason)

    cooldown = max(1, int(min_interval_seconds))
    key = _refresh_attempt_key(clean_alias, store_path)
    now = time.monotonic()
    with _REFRESH_ATTEMPT_LOCK:
        last_attempt = _REFRESH_LAST_ATTEMPT.get(key)
        if not force and last_attempt is not None and (now - last_attempt) < cooldown:
            remaining = cooldown - (now - last_attempt)
            reason = f"Refresh cooldown active ({remaining:.1f}s remaining)."
            return MCPOAuthRefreshResult(status="skipped", reason=reason)
        _REFRESH_LAST_ATTEMPT[key] = now

    request_data: dict[str, str] = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": resolved_client_id,
    }
    resolved_client_secret = str(
        client_secret
        or payload.get("client_secret", "")
        or ""
    ).strip()
    if resolved_client_secret:
        request_data["client_secret"] = resolved_client_secret

    try:
        response = httpx.post(
            resolved_endpoint,
            data=request_data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=max(1, int(timeout_seconds)),
        )
    except Exception as e:
        reason = f"Refresh request failed: {redact_oauth_error_text(e)}"
        _annotate_alias_payload(
            alias=clean_alias,
            store_path=store_path,
            fields={
                "last_failure_reason": reason,
                "last_failure_at": int(time.time()),
            },
        )
        return MCPOAuthRefreshResult(status="failed", reason=reason)

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
        reason = redact_oauth_error_text(detail)
        _annotate_alias_payload(
            alias=clean_alias,
            store_path=store_path,
            fields={
                "last_failure_reason": reason,
                "last_failure_at": int(time.time()),
            },
        )
        return MCPOAuthRefreshResult(status="failed", reason=reason)

    access_token = str(body.get("access_token", "") or "").strip()
    if not access_token:
        reason = "Refresh response missing access_token."
        _annotate_alias_payload(
            alias=clean_alias,
            store_path=store_path,
            fields={
                "last_failure_reason": reason,
                "last_failure_at": int(time.time()),
            },
        )
        return MCPOAuthRefreshResult(status="failed", reason=reason)

    expires_at_unix: int | None = None
    expires_in = body.get("expires_in")
    if expires_in not in (None, ""):
        try:
            expires_at_unix = int(time.time()) + max(1, int(expires_in))
        except (TypeError, ValueError):
            expires_at_unix = _parse_expiry_unix(body.get("expires_at"))
    else:
        expires_at_unix = _parse_expiry_unix(body.get("expires_at"))

    scope_text = str(body.get("scope", "") or "").strip()
    merged_scopes = [
        str(scope).strip()
        for scope in list(payload.get("scopes", []) or [])
        if str(scope).strip()
    ]
    if scope_text:
        merged_scopes.extend(
            scope
            for scope in scope_text.split(" ")
            if scope.strip()
        )
    merged_scopes = list(dict.fromkeys(merged_scopes))

    next_refresh_token = str(body.get("refresh_token", "") or "").strip() or refresh_token
    token_type = (
        str(body.get("token_type", payload.get("token_type", "Bearer")) or "Bearer")
        .strip()
        or "Bearer"
    )

    try:
        upsert_mcp_oauth_token(
            alias=clean_alias,
            access_token=access_token,
            refresh_token=next_refresh_token,
            token_type=token_type,
            scopes=merged_scopes,
            expires_at_unix=expires_at_unix,
            token_endpoint=resolved_endpoint,
            authorization_endpoint=str(payload.get("authorization_endpoint", "") or "").strip(),
            client_id=resolved_client_id,
            obtained_via="refresh_token",
            extra_fields={
                "last_refresh_at": int(time.time()),
            },
            store_path=store_path,
        )
    except MCPOAuthStoreError as e:
        return MCPOAuthRefreshResult(
            status="failed",
            reason=f"Refresh write failed: {redact_oauth_error_text(e)}",
        )

    return MCPOAuthRefreshResult(status="refreshed", refreshed=True)


def ensure_mcp_oauth_ready(
    alias: str,
    *,
    store_path: Path | None = None,
    refresh_timeout_seconds: int = 15,
    refresh_cooldown_seconds: int = _DEFAULT_REFRESH_COOLDOWN_SECONDS,
) -> MCPOAuthReadiness:
    """Return deterministic runtime readiness for one OAuth-enabled alias."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return MCPOAuthReadiness(
            ready=False,
            state="missing",
            reason="Alias cannot be empty.",
        )

    payload = get_mcp_oauth_token(clean_alias, store_path=store_path)
    if payload is None:
        return MCPOAuthReadiness(
            ready=False,
            state="missing",
            reason="OAuth token missing.",
        )

    if not token_expired(payload):
        return MCPOAuthReadiness(ready=True, state="ready")

    if not str(payload.get("refresh_token", "") or "").strip():
        return MCPOAuthReadiness(
            ready=False,
            state="needs_auth",
            reason="OAuth token expired and refresh token is unavailable.",
        )

    refreshed = refresh_mcp_oauth_token(
        clean_alias,
        store_path=store_path,
        timeout_seconds=refresh_timeout_seconds,
        min_interval_seconds=refresh_cooldown_seconds,
    )
    if refreshed.refreshed:
        return MCPOAuthReadiness(
            ready=True,
            state="ready",
            refreshed=True,
        )

    reason = refreshed.reason or "OAuth refresh failed."
    return MCPOAuthReadiness(
        ready=False,
        state="needs_auth",
        reason=reason,
    )


def discover_remote_oauth_provider(
    server_url: str,
    *,
    timeout_seconds: int = 5,
) -> dict[str, str]:
    """Probe common OAuth metadata endpoints for remote MCP providers."""
    clean_url = str(server_url or "").strip()
    parsed = urlparse(clean_url)
    if not parsed.scheme or not parsed.netloc:
        return {}
    origin = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [
        f"{origin}/.well-known/oauth-authorization-server",
        f"{origin}/.well-known/openid-configuration",
    ]
    for url in candidates:
        try:
            response = httpx.get(
                url,
                headers={"Accept": "application/json"},
                timeout=max(1, int(timeout_seconds)),
            )
            if response.status_code >= 400:
                continue
            payload = response.json()
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        authorization_endpoint = str(payload.get("authorization_endpoint", "") or "").strip()
        token_endpoint = str(payload.get("token_endpoint", "") or "").strip()
        if authorization_endpoint and token_endpoint:
            return {
                "authorization_endpoint": authorization_endpoint,
                "token_endpoint": token_endpoint,
                "issuer": str(payload.get("issuer", "") or "").strip(),
            }
    return {}


def resolve_mcp_oauth_provider(
    *,
    server_url: str,
    scopes: list[str] | tuple[str, ...] | None = None,
    authorization_endpoint: str | None = None,
    token_endpoint: str | None = None,
    client_id: str | None = None,
    timeout_seconds: int = 5,
) -> MCPOAuthProviderConfig:
    """Resolve provider endpoints with discovery + explicit override precedence."""
    discovered = discover_remote_oauth_provider(
        server_url,
        timeout_seconds=timeout_seconds,
    )
    resolved_authorization_endpoint = (
        str(authorization_endpoint or "").strip()
        or str(discovered.get("authorization_endpoint", "") or "").strip()
    )
    resolved_token_endpoint = (
        str(token_endpoint or "").strip()
        or str(discovered.get("token_endpoint", "") or "").strip()
    )
    resolved_client_id = (
        str(client_id or "").strip()
        or str(os.environ.get("LOOM_MCP_OAUTH_CLIENT_ID", "") or "").strip()
        or "loom-cli"
    )

    if not resolved_authorization_endpoint or not resolved_token_endpoint:
        raise MCPOAuthFlowError(
            "OAuth provider metadata unavailable. "
            "Set --authorize-url and --token-url or ensure the provider exposes "
            "standard metadata endpoints."
        )

    normalized_scopes = tuple(
        str(scope).strip()
        for scope in (scopes or ())
        if str(scope).strip()
    )
    return MCPOAuthProviderConfig(
        authorization_endpoint=resolved_authorization_endpoint,
        token_endpoint=resolved_token_endpoint,
        client_id=resolved_client_id,
        scopes=normalized_scopes,
    )
