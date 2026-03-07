"""Guardrails preventing regressions to the legacy orchestrator monolith path."""

from __future__ import annotations

import re
from pathlib import Path

LEGACY_ORCHESTRATOR_FILE = Path("tests/test_orchestrator" + ".py")
ALLOWLIST_FILES = {
    Path("CHANGELOG.md"),
}
ALLOWLIST_PREFIXES = (
    Path("planning"),
)


def _iter_text_files(repo_root: Path):
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)
        if any(
            part in {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
            for part in rel.parts
        ):
            continue
        if rel.suffix in {".pyc", ".pyo"}:
            continue
        yield path, rel


def test_legacy_orchestrator_monolith_file_removed() -> None:
    assert not LEGACY_ORCHESTRATOR_FILE.exists()


def test_legacy_orchestrator_path_references_are_allowlisted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    legacy_paths = ["tests/test_orchestrator" + ".py", "test_orchestrator" + ".py"]
    pattern = re.compile("|".join(re.escape(path) for path in legacy_paths))
    violations: list[str] = []

    for path, rel in _iter_text_files(repo_root):
        if rel in ALLOWLIST_FILES:
            continue
        if any(rel.parts[: len(prefix.parts)] == prefix.parts for prefix in ALLOWLIST_PREFIXES):
            continue
        if rel == Path("tests/test_orchestrator_legacy_path_guard.py"):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")

        if pattern.search(text):
            violations.append(str(rel))

    assert not violations, (
        "Found non-allowlisted references to the legacy orchestrator test path: "
        + ", ".join(sorted(violations))
    )
