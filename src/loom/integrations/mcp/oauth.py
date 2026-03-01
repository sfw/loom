"""MCP OAuth token storage and status helpers."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class MCPOAuthStoreError(Exception):
    """Raised when MCP OAuth token state cannot be read or written."""


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


def upsert_mcp_oauth_token(
    *,
    alias: str,
    access_token: str,
    refresh_token: str = "",
    token_type: str = "Bearer",
    scopes: list[str] | None = None,
    expires_at_unix: int | None = None,
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
        aliases[clean_alias] = {
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
        _write_store(target, store)
    return target


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
        return False
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
        }
    expired = token_expired(token_payload)
    return {
        "state": "expired" if expired else "ready",
        "has_token": True,
        "expired": expired,
        "expires_at": token_payload.get("expires_at"),
        "token_type": token_payload.get("token_type", "Bearer"),
        "scopes": list(token_payload.get("scopes", []) or []),
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
