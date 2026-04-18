#!/usr/bin/env python3
"""Smoke-test a packaged macOS Loom Desktop app bundle."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DESKTOP_LEASE_TTL_SECONDS = 20
DESKTOP_LEASE_HEARTBEAT_SECONDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the bundled loomd runtime from a built macOS app bundle."
    )
    parser.add_argument(
        "--app-bundle",
        type=Path,
        required=True,
        help="Path to Loom Desktop.app.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="How long to wait for /runtime readiness.",
    )
    return parser.parse_args()


def load_manifest(resources_root: Path) -> tuple[dict[str, object], Path, Path, Path, Path]:
    manifest_path = resources_root / "loom-desktop-bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    python_home = resources_root / str(manifest["python_home_relative_path"])
    python_executable = resources_root / str(manifest["python_executable_relative_path"])
    environment_root = resources_root / str(manifest["environment_root_relative_path"])
    site_packages = resources_root / str(manifest["site_packages_relative_path"])
    required_paths = [python_home, python_executable, environment_root, site_packages]
    for path in required_paths:
        if not path.exists():
            raise SystemExit(f"bundle resource missing: {path}")
    return manifest, python_home, python_executable, environment_root, site_packages


def choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _now_unix_ms() -> int:
    return int(time.time() * 1000)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        tmp_path = Path(handle.name)
    try:
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_desktop_lease_record(
    lease_path: Path,
    instance_id: str,
    *,
    ttl_seconds: int = DESKTOP_LEASE_TTL_SECONDS,
) -> None:
    now_ms = _now_unix_ms()
    _write_json_atomic(
        lease_path,
        {
            "instance_id": instance_id,
            "desktop_pid": os.getpid(),
            "created_at_unix_ms": now_ms,
            "updated_at_unix_ms": now_ms,
            "lease_expires_unix_ms": now_ms + max(1, int(ttl_seconds)) * 1000,
        },
    )


def remove_desktop_lease_if_owned(lease_path: Path, instance_id: str) -> None:
    try:
        payload = json.loads(lease_path.read_text(encoding="utf-8"))
    except Exception:
        payload = None
    if isinstance(payload, dict) and str(payload.get("instance_id", "")) != instance_id:
        return
    lease_path.unlink(missing_ok=True)


class DesktopLeaseHeartbeater:
    def __init__(
        self,
        *,
        lease_path: Path,
        instance_id: str,
        ttl_seconds: int = DESKTOP_LEASE_TTL_SECONDS,
        heartbeat_seconds: int = DESKTOP_LEASE_HEARTBEAT_SECONDS,
    ) -> None:
        self._lease_path = lease_path
        self._instance_id = instance_id
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._heartbeat_seconds = max(1, int(heartbeat_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        write_desktop_lease_record(
            self._lease_path,
            self._instance_id,
            ttl_seconds=self._ttl_seconds,
        )

        def _heartbeat() -> None:
            while not self._stop.wait(self._heartbeat_seconds):
                write_desktop_lease_record(
                    self._lease_path,
                    self._instance_id,
                    ttl_seconds=self._ttl_seconds,
                )

        self._thread = threading.Thread(
            target=_heartbeat,
            name="loom-desktop-lease-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._heartbeat_seconds + 1)
        remove_desktop_lease_if_owned(self._lease_path, self._instance_id)


def _read_log_tail(log_path: Path | None, *, max_chars: int = 4000) -> str:
    if log_path is None or not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    trimmed = content[-max_chars:].strip()
    if not trimmed:
        return ""
    return f"\nRecent loomd log output:\n{trimmed}"


def wait_for_runtime(
    base_url: str,
    timeout_seconds: float,
    *,
    process: subprocess.Popen[str] | None = None,
    log_path: Path | None = None,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    last_error = "runtime never became ready"
    while time.monotonic() < deadline:
        if process is not None:
            returncode = process.poll()
            if returncode is not None:
                raise SystemExit(
                    "packaged loomd smoke test failed: "
                    f"process exited before /runtime became ready with code {returncode}"
                    f"{_read_log_tail(log_path)}"
                )
        try:
            with urllib.request.urlopen(f"{base_url}/runtime", timeout=1.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200:
                    return payload
        except urllib.error.URLError as error:
            last_error = str(error)
        except TimeoutError as error:
            last_error = str(error)
        time.sleep(0.25)
    raise SystemExit(
        f"packaged loomd smoke test timed out after {timeout_seconds:.1f}s: {last_error}"
        f"{_read_log_tail(log_path)}"
    )


def main() -> None:
    args = parse_args()
    app_bundle = args.app_bundle.expanduser().resolve()
    resources_root = app_bundle / "Contents" / "Resources"
    if not resources_root.exists():
        raise SystemExit(f"bundle resources directory not found: {resources_root}")

    manifest, python_home, python_executable, environment_root, site_packages = load_manifest(
        resources_root
    )
    port = choose_port()

    with tempfile.TemporaryDirectory(prefix="loom-desktop-bundle-smoke-") as tmpdir:
        tmp_root = Path(tmpdir)
        runtime_root = tmp_root / "runtime"
        logs_root = tmp_root / "logs"
        scratch_dir = runtime_root / "scratch"
        python_cache = runtime_root / "python-cache"
        workspace_root = tmp_root / "workspace"
        for path in (runtime_root, logs_root, scratch_dir, python_cache, workspace_root):
            path.mkdir(parents=True, exist_ok=True)

        lease_path = runtime_root / "desktop.instance.json"
        state_path = runtime_root / "loomd.sidecar.json"
        log_path = logs_root / "loomd.log"
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            [
                str(environment_root / "bin"),
                str(python_home / "bin"),
                env.get("PATH", ""),
            ]
        )
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONHOME"] = str(python_home)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(site_packages)
        env["PYTHONPYCACHEPREFIX"] = str(python_cache)
        env["PYTHONUNBUFFERED"] = "1"

        instance_token = "desktop-bundle-smoke"
        base_url = f"http://127.0.0.1:{port}"
        args_list = [
            str(python_executable),
            "-m",
            str(manifest["entry_module"]),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database-path",
            str(runtime_root / "loomd.db"),
            "--scratch-dir",
            str(scratch_dir),
            "--workspace-default-path",
            str(workspace_root),
            "--desktop-instance-token",
            instance_token,
            "--desktop-lease-path",
            str(lease_path),
            "--desktop-sidecar-state-path",
            str(state_path),
        ]

        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                args_list,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )
        lease_heartbeater = DesktopLeaseHeartbeater(
            lease_path=lease_path,
            instance_id=instance_token,
        )
        lease_heartbeater.start()
        try:
            runtime_payload = wait_for_runtime(
                base_url,
                args.timeout_seconds,
                process=process,
                log_path=log_path,
            )
            print(f"Smoke OK: {base_url}")
            print(
                json.dumps(
                    {
                        "loom_version": manifest.get("loom_version"),
                        "python_version": manifest.get("python_version"),
                        "python_request": manifest.get("python_request"),
                        "uv_version": manifest.get("uv_version"),
                        "runtime_payload": runtime_payload,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        finally:
            lease_heartbeater.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
