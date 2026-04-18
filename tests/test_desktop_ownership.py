from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from loom.daemon.desktop_ownership import (
    DesktopLeaseSnapshot,
    clear_sidecar_state,
    lease_matches_owner,
    monitor_desktop_lease,
    read_desktop_lease,
    write_sidecar_state,
)


def test_lease_matches_expected_owner_and_expiry(tmp_path: Path) -> None:
    now_ms = int(time.time() * 1000)
    lease_path = tmp_path / "desktop.instance.json"
    lease_path.write_text(
        json.dumps(
            {
                "instance_id": "desktop-123",
                "desktop_pid": 42,
                "created_at_unix_ms": now_ms - 1_000,
                "updated_at_unix_ms": now_ms,
                "lease_expires_unix_ms": now_ms + 10_000,
            }
        ),
        encoding="utf-8",
    )

    snapshot = read_desktop_lease(lease_path)
    assert snapshot == DesktopLeaseSnapshot(
        instance_id="desktop-123",
        desktop_pid=42,
        created_at_unix_ms=now_ms - 1_000,
        updated_at_unix_ms=now_ms,
        lease_expires_unix_ms=now_ms + 10_000,
    )
    assert lease_matches_owner(snapshot, expected_instance_id="desktop-123", now_unix_ms=now_ms)
    assert not lease_matches_owner(snapshot, expected_instance_id="desktop-999", now_unix_ms=now_ms)
    assert not lease_matches_owner(
        snapshot,
        expected_instance_id="desktop-123",
        now_unix_ms=now_ms + 20_000,
    )


def test_clear_sidecar_state_only_removes_matching_instance(tmp_path: Path) -> None:
    state_path = tmp_path / "loomd.sidecar.json"
    write_sidecar_state(
        path=state_path,
        instance_id="desktop-123",
        host="127.0.0.1",
        port=9000,
        database_path=tmp_path / "loomd.db",
        lease_path=tmp_path / "desktop.instance.json",
    )

    clear_sidecar_state(state_path, expected_instance_id="desktop-999")
    assert state_path.exists()

    clear_sidecar_state(state_path, expected_instance_id="desktop-123")
    assert not state_path.exists()


def test_monitor_desktop_lease_requests_shutdown_when_owner_disappears(tmp_path: Path) -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.pause_calls = 0

        async def pause_active_task_runs_for_shutdown(self) -> int:
            self.pause_calls += 1
            return 1

    class FakeState:
        def __init__(self) -> None:
            self.engine = FakeEngine()

    class FakeApp:
        def __init__(self) -> None:
            self.state = FakeState()

    class FakeServer:
        def __init__(self) -> None:
            self.should_exit = False

    lease_path = tmp_path / "desktop.instance.json"
    now_ms = int(time.time() * 1000)
    lease_path.write_text(
        json.dumps(
            {
                "instance_id": "desktop-old",
                "desktop_pid": 42,
                "created_at_unix_ms": now_ms - 1_000,
                "updated_at_unix_ms": now_ms,
                "lease_expires_unix_ms": now_ms + 10_000,
            }
        ),
        encoding="utf-8",
    )

    app = FakeApp()
    server = FakeServer()
    asyncio.run(
        monitor_desktop_lease(
            app=app,
            server=server,
            expected_instance_id="desktop-new",
            lease_path=lease_path,
            poll_interval_seconds=0.01,
        )
    )

    assert server.should_exit is True
    assert app.state.engine.pause_calls == 1
