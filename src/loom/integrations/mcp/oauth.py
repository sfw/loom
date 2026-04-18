"""MCP OAuth token storage, readiness, and refresh helpers."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from loom.auth.secrets import SecretResolutionError, SecretResolver
from loom.config import MCPServerConfig


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
    authorize_params: dict[str, str] = field(default_factory=dict)
    token_params: dict[str, str] = field(default_factory=dict)


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
_MCP_OAUTH_STORE_VERSION = 2


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
    payload = {
        "version": _MCP_OAUTH_STORE_VERSION,
        "aliases": {},
        "servers": {},
        "alias_bindings": {},
    }
    if not isinstance(raw, dict):
        return payload

    aliases_raw = raw.get("aliases", {})
    aliases: dict[str, dict[str, Any]] = {}
    if isinstance(aliases_raw, dict):
        for key, value in aliases_raw.items():
            alias = str(key or "").strip()
            if not alias or not isinstance(value, dict):
                continue
            aliases[alias] = dict(value)
    payload["aliases"] = aliases

    servers_raw = raw.get("servers", {})
    servers: dict[str, dict[str, Any]] = {}
    if isinstance(servers_raw, dict):
        for key, value in servers_raw.items():
            server_fingerprint = str(key or "").strip()
            if not server_fingerprint or not isinstance(value, dict):
                continue
            entry = dict(value)
            token_ref = str(entry.get("token_ref", "") or "").strip()
            aliases_for_server = [
                str(item).strip()
                for item in list(entry.get("aliases", []) or [])
                if str(item).strip()
            ]
            sanitized: dict[str, Any] = {}
            if token_ref:
                sanitized["token_ref"] = token_ref
            if aliases_for_server:
                sanitized["aliases"] = list(dict.fromkeys(aliases_for_server))
            for key_name in (
                "credential_fingerprint",
                "authorization_endpoint",
                "token_endpoint",
                "client_id",
            ):
                value_text = str(entry.get(key_name, "") or "").strip()
                if value_text:
                    sanitized[key_name] = value_text
            for key_name in ("updated_at", "migrated_from_legacy_at"):
                try:
                    value_int = int(entry.get(key_name))
                except (TypeError, ValueError):
                    continue
                if value_int > 0:
                    sanitized[key_name] = value_int
            if sanitized:
                servers[server_fingerprint] = sanitized
    payload["servers"] = servers

    bindings_raw = raw.get("alias_bindings", {})
    bindings: dict[str, dict[str, Any]] = {}
    if isinstance(bindings_raw, dict):
        for key, value in bindings_raw.items():
            alias = str(key or "").strip()
            if not alias or not isinstance(value, dict):
                continue
            server_fingerprint = str(value.get("server_fingerprint", "") or "").strip()
            if not server_fingerprint:
                continue
            binding: dict[str, Any] = {"server_fingerprint": server_fingerprint}
            try:
                updated_at = int(value.get("updated_at"))
            except (TypeError, ValueError):
                updated_at = 0
            if updated_at > 0:
                binding["updated_at"] = updated_at
            bindings[alias] = binding
    payload["alias_bindings"] = bindings
    return payload


def load_mcp_oauth_store(path: Path | None = None) -> dict[str, Any]:
    """Load token store payload."""
    target = (path or default_mcp_oauth_store_path()).expanduser().resolve()
    if not target.exists():
        return _coerce_store_payload({})
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


def _canonical_remote_server_url(raw_url: str) -> str:
    parsed = urlparse(str(raw_url or "").strip())
    scheme = str(parsed.scheme or "").strip().lower()
    hostname = str(parsed.hostname or "").strip().lower()
    if not scheme or not hostname:
        return str(raw_url or "").strip()
    port = parsed.port
    default_port = 443 if scheme == "https" else 80 if scheme == "http" else None
    netloc = hostname
    if port is not None and port != default_port:
        netloc = f"{hostname}:{port}"
    path = parsed.path or ""
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def fingerprint_mcp_server(server: MCPServerConfig | None) -> str:
    """Return a stable credential-binding fingerprint for one MCP server config."""
    if server is None:
        return ""
    payload = {
        "type": str(server.type or "").strip().lower(),
        "url": _canonical_remote_server_url(server.url),
        "headers": [
            [str(key).strip().lower(), str(value)]
            for key, value in sorted(
                (server.headers or {}).items(),
                key=lambda item: str(item[0]).strip().lower(),
            )
            if str(key).strip()
        ],
        "oauth": {
            "enabled": bool(getattr(server.oauth, "enabled", False)),
            "scopes": sorted(
                {
                    str(scope).strip()
                    for scope in list(getattr(server.oauth, "scopes", []) or [])
                    if str(scope).strip()
                }
            ),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _alias_fingerprint(alias: str) -> str:
    encoded = json.dumps(
        {"kind": "legacy_alias", "alias": str(alias or "").strip()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _credential_fingerprint(
    *,
    server_fingerprint: str,
    authorization_endpoint: str,
    token_endpoint: str,
    client_id: str,
) -> str:
    encoded = json.dumps(
        {
            "server_fingerprint": server_fingerprint,
            "authorization_endpoint": str(authorization_endpoint or "").strip(),
            "token_endpoint": str(token_endpoint or "").strip(),
            "client_id": str(client_id or "").strip(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _token_ref_for_server_fingerprint(server_fingerprint: str) -> str:
    return f"keychain://loom/mcp/oauth/{server_fingerprint}/tokens"


def _wrap_secret_storage_error(error: SecretResolutionError) -> MCPOAuthStoreError:
    return MCPOAuthStoreError(
        "MCP OAuth token persistence requires writable keychain storage. "
        f"{error}"
    )


def _server_entry_for_alias(
    *,
    alias: str,
    server: MCPServerConfig | None,
    store: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    server_fingerprint = fingerprint_mcp_server(server) if server is not None else ""
    servers = store.get("servers", {})
    if server_fingerprint:
        raw = servers.get(server_fingerprint)
        return server_fingerprint, dict(raw) if isinstance(raw, dict) else None

    binding = store.get("alias_bindings", {}).get(alias)
    if not isinstance(binding, dict):
        return "", None
    server_fingerprint = str(binding.get("server_fingerprint", "") or "").strip()
    if not server_fingerprint:
        return "", None
    raw = servers.get(server_fingerprint)
    return server_fingerprint, dict(raw) if isinstance(raw, dict) else None


def _legacy_alias_payload(alias: str, *, store: dict[str, Any]) -> dict[str, Any] | None:
    raw = store.get("aliases", {}).get(alias)
    if not isinstance(raw, dict):
        return None
    return dict(raw)


def _resolve_bound_token_payload(
    *,
    entry: dict[str, Any],
    resolver: SecretResolver | None = None,
) -> tuple[dict[str, Any] | None, str]:
    token_ref = str(entry.get("token_ref", "") or "").strip()
    if not token_ref:
        return None, "token_ref is missing."
    secret_resolver = resolver or SecretResolver()
    try:
        token_value = secret_resolver.resolve(token_ref).strip()
    except SecretResolutionError as e:
        return None, str(e)
    if not token_value:
        return None, "Stored token value is empty."
    try:
        payload = json.loads(token_value)
    except json.JSONDecodeError as e:
        return None, f"Stored token payload is not valid JSON: {e}"
    if not isinstance(payload, dict):
        return None, "Stored token payload must be a JSON object."
    return dict(payload), ""


def _unbind_alias(store: dict[str, Any], alias: str) -> None:
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return
    bindings = store.setdefault("alias_bindings", {})
    binding = bindings.pop(clean_alias, None)
    if not isinstance(binding, dict):
        return
    server_fingerprint = str(binding.get("server_fingerprint", "") or "").strip()
    if not server_fingerprint:
        return
    servers = store.setdefault("servers", {})
    raw_entry = servers.get(server_fingerprint)
    if not isinstance(raw_entry, dict):
        return
    entry = dict(raw_entry)
    aliases = [
        item
        for item in list(entry.get("aliases", []) or [])
        if str(item).strip() and str(item).strip() != clean_alias
    ]
    if aliases:
        entry["aliases"] = aliases
    else:
        entry.pop("aliases", None)
    servers[server_fingerprint] = entry


def _remove_server_binding(
    *,
    store: dict[str, Any],
    server_fingerprint: str,
) -> None:
    clean_fingerprint = str(server_fingerprint or "").strip()
    if not clean_fingerprint:
        return
    store.setdefault("servers", {}).pop(clean_fingerprint, None)
    bindings = store.setdefault("alias_bindings", {})
    aliases_to_remove = [
        alias
        for alias, raw in bindings.items()
        if isinstance(raw, dict)
        and str(raw.get("server_fingerprint", "") or "").strip() == clean_fingerprint
    ]
    for alias in aliases_to_remove:
        bindings.pop(alias, None)


def _store_bound_token_payload(
    *,
    target: Path,
    alias: str,
    server: MCPServerConfig | None,
    token_payload: dict[str, Any],
    authorization_endpoint: str,
    token_endpoint: str,
    client_id: str,
    secret_resolver: SecretResolver | None = None,
    migrated_from_legacy: bool = False,
) -> Path:
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        raise MCPOAuthStoreError("Alias cannot be empty.")
    server_fingerprint = fingerprint_mcp_server(server) if server is not None else ""
    if not server_fingerprint:
        server_fingerprint = _alias_fingerprint(clean_alias)
    token_ref = _token_ref_for_server_fingerprint(server_fingerprint)
    resolver = secret_resolver or SecretResolver()
    try:
        resolver.validate_writable(token_ref)
        resolver.store(
            token_ref,
            json.dumps(token_payload, sort_keys=True) + "\n",
        )
    except SecretResolutionError as e:
        raise _wrap_secret_storage_error(e) from e

    now = int(time.time())
    lock_path = target.with_suffix(target.suffix + ".lock")
    with _file_lock(lock_path):
        store = load_mcp_oauth_store(target)
        _unbind_alias(store, clean_alias)
        servers = store.setdefault("servers", {})
        aliases = [
            str(item).strip()
            for item in list(dict(servers.get(server_fingerprint) or {}).get("aliases", []) or [])
            if str(item).strip()
        ]
        if clean_alias not in aliases:
            aliases.append(clean_alias)
        servers[server_fingerprint] = {
            "token_ref": token_ref,
            "aliases": aliases,
            "updated_at": now,
            "credential_fingerprint": _credential_fingerprint(
                server_fingerprint=server_fingerprint,
                authorization_endpoint=authorization_endpoint,
                token_endpoint=token_endpoint,
                client_id=client_id,
            ),
            "authorization_endpoint": authorization_endpoint,
            "token_endpoint": token_endpoint,
            "client_id": client_id,
            **(
                {"migrated_from_legacy_at": now}
                if migrated_from_legacy
                else {}
            ),
        }
        store.setdefault("alias_bindings", {})[clean_alias] = {
            "server_fingerprint": server_fingerprint,
            "updated_at": now,
        }
        _write_store(target, store)
    return target


def _maybe_migrate_legacy_alias_token(
    *,
    alias: str,
    server: MCPServerConfig | None,
    store_path: Path | None,
    resolver: SecretResolver | None = None,
) -> dict[str, Any] | None:
    if server is None:
        return None
    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    store = load_mcp_oauth_store(target)
    legacy_payload = _legacy_alias_payload(alias, store=store)
    if legacy_payload is None:
        return None
    server_fingerprint, entry = _server_entry_for_alias(alias=alias, server=server, store=store)
    if server_fingerprint and entry is not None:
        payload, _ = _resolve_bound_token_payload(entry=entry, resolver=resolver)
        if payload is not None:
            return payload
    try:
        _store_bound_token_payload(
            target=target,
            alias=alias,
            server=server,
            token_payload=legacy_payload,
            authorization_endpoint=str(
                legacy_payload.get("authorization_endpoint", "") or ""
            ).strip(),
            token_endpoint=str(
                legacy_payload.get("token_endpoint", "") or ""
            ).strip(),
            client_id=str(legacy_payload.get("client_id", "") or "").strip(),
            secret_resolver=resolver,
            migrated_from_legacy=True,
        )
    except MCPOAuthStoreError:
        return legacy_payload
    migrated_store = load_mcp_oauth_store(target)
    _, migrated_entry = _server_entry_for_alias(alias=alias, server=server, store=migrated_store)
    if migrated_entry is None:
        return legacy_payload
    payload, _ = _resolve_bound_token_payload(entry=migrated_entry, resolver=resolver)
    return payload or legacy_payload


def upsert_mcp_oauth_token(
    *,
    alias: str,
    access_token: str,
    server: MCPServerConfig | None = None,
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
    secret_resolver: SecretResolver | None = None,
) -> Path:
    """Insert or update one MCP OAuth token payload."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        raise MCPOAuthStoreError("Alias cannot be empty.")
    token = str(access_token or "").strip()
    if not token:
        raise MCPOAuthStoreError("Access token cannot be empty.")

    payload: dict[str, Any] = {
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
    payload.pop("last_failure_reason", None)
    payload.pop("last_failure_at", None)

    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    return _store_bound_token_payload(
        target=target,
        alias=clean_alias,
        server=server,
        token_payload=payload,
        authorization_endpoint=authorization_ep,
        token_endpoint=token_ep,
        client_id=clean_client_id,
        secret_resolver=secret_resolver,
    )


def _annotate_alias_payload(
    *,
    alias: str,
    store_path: Path | None,
    server: MCPServerConfig | None,
    fields: dict[str, Any],
) -> None:
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return
    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    store = load_mcp_oauth_store(target)
    payload: dict[str, Any] | None = None
    _, entry = _server_entry_for_alias(
        alias=clean_alias,
        server=server,
        store=store,
    )
    if entry is not None:
        payload, _ = _resolve_bound_token_payload(entry=entry)
    if payload is None:
        payload = _legacy_alias_payload(clean_alias, store=store)
    if payload is None:
        return
    sanitized_fields = dict(fields)
    if "last_failure_reason" in sanitized_fields:
        sanitized_fields["last_failure_reason"] = redact_oauth_error_text(
            sanitized_fields.get("last_failure_reason", ""),
        )
    payload.update(sanitized_fields)
    if entry is not None:
        try:
            _store_bound_token_payload(
                target=target,
                alias=clean_alias,
                server=server,
                token_payload=payload,
                authorization_endpoint=str(
                    entry.get("authorization_endpoint", "")
                    or payload.get("authorization_endpoint", "")
                    or ""
                ).strip(),
                token_endpoint=str(
                    entry.get("token_endpoint", "")
                    or payload.get("token_endpoint", "")
                    or ""
                ).strip(),
                client_id=str(
                    entry.get("client_id", "")
                    or payload.get("client_id", "")
                    or ""
                ).strip(),
            )
        except MCPOAuthStoreError:
            return
        return
    lock_path = target.with_suffix(target.suffix + ".lock")
    with _file_lock(lock_path):
        store = load_mcp_oauth_store(target)
        store.setdefault("aliases", {})[clean_alias] = payload
        _write_store(target, store)


def remove_mcp_oauth_token(
    alias: str,
    *,
    server: MCPServerConfig | None = None,
    store_path: Path | None = None,
    secret_resolver: SecretResolver | None = None,
) -> Path:
    """Delete one MCP OAuth token entry."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        raise MCPOAuthStoreError("Alias cannot be empty.")
    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    lock_path = target.with_suffix(target.suffix + ".lock")
    with _file_lock(lock_path):
        store = load_mcp_oauth_store(target)
        server_fingerprint, entry = _server_entry_for_alias(
            alias=clean_alias,
            server=server,
            store=store,
        )
        if entry is not None:
            token_ref = str(entry.get("token_ref", "") or "").strip()
            if token_ref:
                resolver = secret_resolver or SecretResolver()
                try:
                    resolver.validate_writable(token_ref)
                    resolver.store(token_ref, "")
                except SecretResolutionError as e:
                    raise _wrap_secret_storage_error(e) from e
            _remove_server_binding(store=store, server_fingerprint=server_fingerprint)
        _unbind_alias(store, clean_alias)
        aliases = store.setdefault("aliases", {})
        aliases.pop(clean_alias, None)
        _write_store(target, store)
    return target


def get_mcp_oauth_token(
    alias: str,
    *,
    server: MCPServerConfig | None = None,
    store_path: Path | None = None,
    secret_resolver: SecretResolver | None = None,
) -> dict[str, Any] | None:
    """Return token payload for one alias."""
    clean_alias = str(alias or "").strip()
    if not clean_alias:
        return None
    store = load_mcp_oauth_store(store_path)
    _, entry = _server_entry_for_alias(alias=clean_alias, server=server, store=store)
    if entry is not None:
        payload, _ = _resolve_bound_token_payload(entry=entry, resolver=secret_resolver)
        if payload is not None:
            return payload
    migrated = _maybe_migrate_legacy_alias_token(
        alias=clean_alias,
        server=server,
        store_path=store_path,
        resolver=secret_resolver,
    )
    if migrated is not None:
        return migrated
    return _legacy_alias_payload(clean_alias, store=store)


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
    server: MCPServerConfig | None = None,
    store_path: Path | None = None,
) -> dict[str, Any]:
    """Summarize OAuth readiness for one alias."""
    clean_alias = str(alias or "").strip()
    store = load_mcp_oauth_store(store_path)
    _, entry = _server_entry_for_alias(alias=clean_alias, server=server, store=store)
    token_payload: dict[str, Any] | None = None
    resolution_error = ""
    storage = "missing"
    if entry is not None:
        token_payload, resolution_error = _resolve_bound_token_payload(entry=entry)
        storage = "secret_ref"
    if token_payload is None:
        token_payload = _maybe_migrate_legacy_alias_token(
            alias=clean_alias,
            server=server,
            store_path=store_path,
        )
        if token_payload is not None:
            storage = "secret_ref" if server is not None else "legacy_alias_store"
    if token_payload is None:
        token_payload = _legacy_alias_payload(clean_alias, store=store)
        if token_payload is not None:
            storage = "legacy_alias_store"
    if token_payload is None:
        return {
            "state": "missing",
            "has_token": False,
            "expired": False,
            "expires_at": None,
            "token_type": None,
            "scopes": [],
            "last_failure_reason": resolution_error,
            "last_failure_at": None,
            "storage": storage,
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
        "storage": storage,
    }


def bearer_auth_header_for_alias(
    alias: str,
    *,
    server: MCPServerConfig | None = None,
    store_path: Path | None = None,
) -> str | None:
    """Return Authorization header value when a usable token is present."""
    token_payload = get_mcp_oauth_token(alias, server=server, store_path=store_path)
    if token_payload is None or token_expired(token_payload):
        return None
    access_token = str(token_payload.get("access_token", "")).strip()
    token_type = str(token_payload.get("token_type", "Bearer")).strip() or "Bearer"
    if token_type.lower() == "bearer":
        token_type = "Bearer"
    if not access_token:
        return None
    return f"{token_type} {access_token}"


def _refresh_attempt_key(
    alias: str,
    store_path: Path | None,
    server: MCPServerConfig | None,
) -> tuple[str, str]:
    target = (store_path or default_mcp_oauth_store_path()).expanduser().resolve()
    server_fingerprint = fingerprint_mcp_server(server) if server is not None else ""
    return str(target), server_fingerprint or str(alias or "").strip()


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
    server: MCPServerConfig | None = None,
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

    payload = get_mcp_oauth_token(clean_alias, server=server, store_path=store_path)
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
            server=server,
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
            server=server,
            fields={
                "last_failure_reason": reason,
                "last_failure_at": int(time.time()),
            },
        )
        return MCPOAuthRefreshResult(status="failed", reason=reason)

    cooldown = max(1, int(min_interval_seconds))
    key = _refresh_attempt_key(clean_alias, store_path, server)
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
            server=server,
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
            server=server,
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
            server=server,
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
            server=server,
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
    server: MCPServerConfig | None = None,
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

    payload = get_mcp_oauth_token(clean_alias, server=server, store_path=store_path)
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
        server=server,
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


def _append_unique(values: list[str], raw: object) -> None:
    text = str(raw or "").strip()
    if text and text not in values:
        values.append(text)


def _origin_for_url(raw_url: str) -> str:
    parsed = urlparse(str(raw_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def _base_url_for_metadata(raw_url: str) -> str:
    parsed = urlparse(str(raw_url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _oauth_metadata_candidates(base_url: str) -> list[str]:
    base = _base_url_for_metadata(base_url)
    if not base:
        return []
    origin = _origin_for_url(base)
    candidates: list[str] = []
    _append_unique(candidates, f"{base}/.well-known/oauth-authorization-server")
    _append_unique(candidates, f"{base}/.well-known/openid-configuration")
    if origin and origin != base:
        _append_unique(candidates, f"{origin}/.well-known/oauth-authorization-server")
        _append_unique(candidates, f"{origin}/.well-known/openid-configuration")
    return candidates


def _oauth_protected_resource_candidates(server_url: str) -> list[str]:
    clean_url = str(server_url or "").strip().rstrip("/")
    origin = _origin_for_url(clean_url)
    candidates: list[str] = []
    if origin:
        _append_unique(candidates, f"{origin}/.well-known/oauth-protected-resource")
    if clean_url:
        _append_unique(candidates, f"{clean_url}/.well-known/oauth-protected-resource")
    return candidates


def _fetch_oauth_metadata_json(
    url: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    try:
        response = httpx.get(
            url,
            headers={"Accept": "application/json"},
            timeout=max(1, int(timeout_seconds)),
        )
        if response.status_code >= 400:
            return {}
        payload = response.json()
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def register_remote_oauth_client(
    *,
    registration_endpoint: str,
    redirect_uris: list[str] | tuple[str, ...],
    scopes: list[str] | tuple[str, ...] | None = None,
    timeout_seconds: int = 10,
    client_name: str = "Loom MCP Client",
) -> dict[str, str]:
    """Register a public OAuth client and return client credentials."""
    endpoint = str(registration_endpoint or "").strip()
    if not endpoint:
        raise MCPOAuthFlowError("OAuth client registration endpoint is required.")

    clean_redirect_uris: list[str] = []
    for raw in redirect_uris:
        _append_unique(clean_redirect_uris, raw)
    if not clean_redirect_uris:
        raise MCPOAuthFlowError("OAuth client registration requires redirect URI values.")

    payload: dict[str, Any] = {
        "client_name": str(client_name or "Loom MCP Client").strip() or "Loom MCP Client",
        "redirect_uris": clean_redirect_uris,
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    scope_values = [
        str(scope).strip()
        for scope in (scopes or ())
        if str(scope).strip()
    ]
    if scope_values:
        payload["scope"] = " ".join(scope_values)

    try:
        response = httpx.post(
            endpoint,
            json=payload,
            headers={"Accept": "application/json"},
            timeout=max(1, int(timeout_seconds)),
        )
    except Exception as e:
        raise MCPOAuthFlowError(
            "OAuth client registration request failed: "
            f"{redact_oauth_error_text(e)}"
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
            or "OAuth client registration failed."
        ).strip()
        raise MCPOAuthFlowError(
            "OAuth client registration failed: "
            f"{redact_oauth_error_text(detail)}"
        )

    resolved_client_id = str(body.get("client_id", "") or "").strip()
    if not resolved_client_id:
        raise MCPOAuthFlowError(
            "OAuth client registration response missing client_id."
        )
    return {
        "client_id": resolved_client_id,
        "client_secret": str(body.get("client_secret", "") or "").strip(),
    }


def discover_remote_oauth_provider(
    server_url: str,
    *,
    timeout_seconds: int = 5,
) -> dict[str, str]:
    """Probe MCP OAuth metadata endpoints and resolve auth-server endpoints."""
    clean_url = str(server_url or "").strip()
    parsed = urlparse(clean_url)
    if not parsed.scheme or not parsed.netloc:
        return {}

    auth_server_candidates: list[str] = []
    resource_candidates: list[str] = []
    _append_unique(auth_server_candidates, clean_url)
    _append_unique(auth_server_candidates, f"{parsed.scheme}://{parsed.netloc}")

    for metadata_url in _oauth_protected_resource_candidates(clean_url):
        metadata = _fetch_oauth_metadata_json(
            metadata_url,
            timeout_seconds=timeout_seconds,
        )
        if not metadata:
            continue
        _append_unique(resource_candidates, metadata.get("resource"))
        _append_unique(resource_candidates, clean_url)
        _append_unique(resource_candidates, f"{parsed.scheme}://{parsed.netloc}")
        _append_unique(auth_server_candidates, metadata.get("authorization_server"))
        raw_servers = metadata.get("authorization_servers", [])
        if isinstance(raw_servers, list):
            for item in raw_servers:
                _append_unique(auth_server_candidates, item)

    for auth_base in auth_server_candidates:
        for metadata_url in _oauth_metadata_candidates(auth_base):
            metadata = _fetch_oauth_metadata_json(
                metadata_url,
                timeout_seconds=timeout_seconds,
            )
            if not metadata:
                continue
            authorization_endpoint = str(
                metadata.get("authorization_endpoint", "") or ""
            ).strip()
            token_endpoint = str(metadata.get("token_endpoint", "") or "").strip()
            if authorization_endpoint and token_endpoint:
                return {
                    "authorization_endpoint": authorization_endpoint,
                    "token_endpoint": token_endpoint,
                    "registration_endpoint": str(
                        metadata.get("registration_endpoint", "") or ""
                    ).strip(),
                    "issuer": str(metadata.get("issuer", "") or "").strip(),
                    "resource": (
                        str(metadata.get("resource", "") or "").strip()
                        or (resource_candidates[0] if resource_candidates else "")
                    ),
                }
    return {}


def resolve_mcp_oauth_provider(
    *,
    server_url: str,
    scopes: list[str] | tuple[str, ...] | None = None,
    authorization_endpoint: str | None = None,
    token_endpoint: str | None = None,
    client_id: str | None = None,
    redirect_uris: list[str] | tuple[str, ...] | None = None,
    client_name: str = "Loom MCP Client",
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
    resolved_registration_endpoint = str(
        discovered.get("registration_endpoint", "") or ""
    ).strip()
    resolved_resource = str(discovered.get("resource", "") or "").strip()
    resolved_client_id = str(client_id or "").strip() or str(
        os.environ.get("LOOM_MCP_OAUTH_CLIENT_ID", "") or ""
    ).strip()

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
    authorize_params: dict[str, str] = {}
    token_params: dict[str, str] = {}
    if resolved_resource:
        authorize_params["resource"] = resolved_resource
        token_params["resource"] = resolved_resource
    if not resolved_client_id and resolved_registration_endpoint:
        candidate_redirect_uris = tuple(
            str(uri).strip()
            for uri in (
                redirect_uris
                or (
                    "http://127.0.0.1:8765/oauth/callback",
                    "http://localhost:8765/oauth/callback",
                    "urn:ietf:wg:oauth:2.0:oob",
                )
            )
            if str(uri).strip()
        )
        registration = register_remote_oauth_client(
            registration_endpoint=resolved_registration_endpoint,
            redirect_uris=candidate_redirect_uris,
            scopes=normalized_scopes,
            timeout_seconds=timeout_seconds,
            client_name=client_name,
        )
        resolved_client_id = registration["client_id"]
        client_secret = registration.get("client_secret", "")
        if client_secret:
            token_params["client_secret"] = client_secret
    if not resolved_client_id:
        resolved_client_id = "loom-cli"

    return MCPOAuthProviderConfig(
        authorization_endpoint=resolved_authorization_endpoint,
        token_endpoint=resolved_token_endpoint,
        client_id=resolved_client_id,
        scopes=normalized_scopes,
        authorize_params=authorize_params,
        token_params=token_params,
    )
