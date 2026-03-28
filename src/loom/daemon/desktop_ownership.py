"""Desktop ownership helpers for app-managed `loomd` runtimes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now_unix_ms() -> int:
    return int(time.time() * 1000)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


@dataclass(slots=True)
class DesktopLeaseSnapshot:
    instance_id: str
    desktop_pid: int
    created_at_unix_ms: int
    updated_at_unix_ms: int
    lease_expires_unix_ms: int


@dataclass(slots=True)
class SidecarStateSnapshot:
    instance_id: str
    pid: int
    host: str
    port: int
    base_url: str
    database_path: str
    lease_path: str
    started_at_unix_ms: int


def read_desktop_lease(path: Path) -> DesktopLeaseSnapshot | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return DesktopLeaseSnapshot(
            instance_id=str(payload["instance_id"]),
            desktop_pid=int(payload["desktop_pid"]),
            created_at_unix_ms=int(payload["created_at_unix_ms"]),
            updated_at_unix_ms=int(payload["updated_at_unix_ms"]),
            lease_expires_unix_ms=int(payload["lease_expires_unix_ms"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def lease_matches_owner(
    snapshot: DesktopLeaseSnapshot | None,
    *,
    expected_instance_id: str,
    now_unix_ms: int | None = None,
) -> bool:
    if snapshot is None:
        return False
    now_ms = _now_unix_ms() if now_unix_ms is None else int(now_unix_ms)
    return snapshot.instance_id == expected_instance_id and snapshot.lease_expires_unix_ms > now_ms


def write_sidecar_state(
    *,
    path: Path,
    instance_id: str,
    host: str,
    port: int,
    database_path: Path,
    lease_path: Path,
) -> None:
    snapshot = SidecarStateSnapshot(
        instance_id=instance_id,
        pid=os.getpid(),
        host=str(host),
        port=int(port),
        base_url=f"http://{host}:{int(port)}",
        database_path=str(database_path),
        lease_path=str(lease_path),
        started_at_unix_ms=_now_unix_ms(),
    )
    _write_json_atomic(path, asdict(snapshot))


def clear_sidecar_state(path: Path, *, expected_instance_id: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    if str(payload.get("instance_id", "")) != expected_instance_id:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


async def monitor_desktop_lease(
    *,
    app: Any,
    server: Any,
    expected_instance_id: str,
    lease_path: Path,
    poll_interval_seconds: float = 2.0,
) -> None:
    """Stop the sidecar when the owning desktop lease disappears or changes."""
    interval = max(0.1, float(poll_interval_seconds))
    while not bool(getattr(server, "should_exit", False)):
        snapshot = read_desktop_lease(lease_path)
        if lease_matches_owner(snapshot, expected_instance_id=expected_instance_id):
            await asyncio.sleep(interval)
            continue

        logger.warning(
            "Desktop ownership lease expired or changed; shutting down loomd for owner %s",
            expected_instance_id,
        )
        engine = getattr(getattr(app, "state", None), "engine", None)
        if engine is not None:
            try:
                await engine.pause_active_task_runs_for_shutdown()
            except Exception:
                logger.warning(
                    "Failed pausing active task runs before loomd lease shutdown",
                    exc_info=True,
                )
        server.should_exit = True
        return
