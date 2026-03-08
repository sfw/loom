#!/usr/bin/env python3
"""Validate Loom version consistency.

Authoritative source: pyproject.toml [project].version.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

RUNTIME_FILES_NO_SEMVER = (
    ROOT / "src/loom/__init__.py",
    ROOT / "src/loom/api/server.py",
    ROOT / "src/loom/integrations/mcp_tools.py",
    ROOT / "src/loom/tui/app/process_runs/adhoc.py",
)

PRIMARY_DOC_EXPECTATIONS: dict[Path, tuple[str, ...]] = {
    ROOT / "INSTALL.md": (
        "# loom, version {version}",
        '# {{"status":"ok","version":"{version}"}}',
    ),
    ROOT / "docs/tutorial.html": (
        "# loom, version {version}",
        '# {{"status":"ok","version":"{version}"}}',
    ),
}

SEMVER_LITERAL_RE = re.compile(r"""['"]\d+\.\d+\.\d+['"]""")


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError("Missing [project].version in pyproject.toml")
    return version.strip()


def _runtime_version() -> str:
    src_root = ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from loom import __version__

    return str(__version__).strip()


def main() -> int:
    expected = _pyproject_version()
    actual = _runtime_version()
    errors: list[str] = []

    if actual != expected:
        errors.append(
            "Runtime version mismatch: "
            f"pyproject.toml={expected!r}, loom.__version__={actual!r}."
        )

    for path in RUNTIME_FILES_NO_SEMVER:
        text = path.read_text(encoding="utf-8")
        for match in SEMVER_LITERAL_RE.finditer(text):
            errors.append(
                f"{path.relative_to(ROOT)} contains hardcoded semver literal {match.group(0)}."
            )

    for path, snippets in PRIMARY_DOC_EXPECTATIONS.items():
        text = path.read_text(encoding="utf-8")
        for snippet in snippets:
            rendered = snippet.format(version=expected)
            if rendered not in text:
                errors.append(
                    f"{path.relative_to(ROOT)} missing expected version snippet: {rendered}"
                )

    if errors:
        print("version-consistency: FAILED")
        for item in errors:
            print(f"- {item}")
        return 1

    print(f"version-consistency: OK ({expected})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
