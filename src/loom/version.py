"""Central version resolver for Loom.

Authoritative version source is ``pyproject.toml`` ([project].version).
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path


def _read_version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.exists():
        return None
    try:
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Unable to locate {pyproject_path}") from exc

    project = data.get("project", {})
    version = project.get("version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Missing [project].version in {pyproject_path}")
    return version.strip()


@lru_cache(maxsize=1)
def get_version() -> str:
    """Resolve Loom package version."""
    source_version = _read_version_from_pyproject()
    if source_version is not None:
        return source_version
    try:
        return package_version("loom")
    except PackageNotFoundError:
        raise RuntimeError("Unable to resolve Loom version from pyproject.toml or package metadata")


__version__ = get_version()
