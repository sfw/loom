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
import time
import urllib.error
import urllib.request
from pathlib import Path


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
        default=30.0,
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


def wait_for_runtime(base_url: str, timeout_seconds: float) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    last_error = "runtime never became ready"
    while time.time() < deadline:
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
    raise SystemExit(f"packaged loomd smoke test timed out: {last_error}")


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
        try:
            runtime_payload = wait_for_runtime(base_url, args.timeout_seconds)
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
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


if __name__ == "__main__":
    main()
