from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _snapshot(name: str) -> str:
    path = Path(__file__).parent / "snapshots" / name
    return path.read_text(encoding="utf-8")


def _assert_help(args: list[str], snapshot_name: str) -> None:
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    result = subprocess.run(
        [sys.executable, "-m", "loom", *args, "--help"],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == _snapshot(snapshot_name)


def test_root_help_snapshot() -> None:
    _assert_help([], "help.txt")


def test_auth_help_snapshot() -> None:
    _assert_help(["auth"], "auth_help.txt")


def test_mcp_help_snapshot() -> None:
    _assert_help(["mcp"], "mcp_help.txt")


def test_db_help_snapshot() -> None:
    _assert_help(["db"], "db_help.txt")


def test_process_help_snapshot() -> None:
    _assert_help(["process"], "process_help.txt")
