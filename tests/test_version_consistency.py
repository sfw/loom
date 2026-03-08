"""Version consistency tests."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from loom import __version__


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    root = _project_root()
    with (root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    version = data.get("project", {}).get("version")
    assert isinstance(version, str) and version.strip()
    return version.strip()


def test_runtime_version_matches_pyproject() -> None:
    assert __version__ == _pyproject_version()


def test_version_consistency_script_passes() -> None:
    root = _project_root()
    script = root / "scripts" / "check_version_consistency.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}".strip()

