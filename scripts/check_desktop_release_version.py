#!/usr/bin/env python3
"""Validate Loom Desktop release version metadata."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP_ROOT = ROOT / "apps" / "desktop"
TAURI_CONFIG = DESKTOP_ROOT / "src-tauri" / "tauri.conf.json"
CARGO_TOML = DESKTOP_ROOT / "src-tauri" / "Cargo.toml"
PACKAGE_JSON = DESKTOP_ROOT / "package.json"


def _load_tauri_version() -> str:
    data = json.loads(TAURI_CONFIG.read_text(encoding="utf-8"))
    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Missing version in {TAURI_CONFIG.relative_to(ROOT)}")
    return version.strip()


def _load_cargo_version() -> str:
    data = tomllib.loads(CARGO_TOML.read_text(encoding="utf-8"))
    version = data.get("package", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Missing package.version in {CARGO_TOML.relative_to(ROOT)}")
    return version.strip()


def _load_package_json_version() -> str:
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Missing version in {PACKAGE_JSON.relative_to(ROOT)}")
    return version.strip()


def _normalize_tag(tag: str) -> str:
    normalized = tag.strip()
    prefix = "refs/tags/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate desktop version metadata and an optional release tag."
    )
    parser.add_argument(
        "--tag",
        help="Release tag to validate, e.g. desktop-v0.3.0 or refs/tags/desktop-v0.3.0.",
    )
    parser.add_argument(
        "--tag-prefix",
        default="desktop-v",
        help="Expected release tag prefix.",
    )
    args = parser.parse_args()

    tauri_version = _load_tauri_version()
    cargo_version = _load_cargo_version()
    package_version = _load_package_json_version()

    errors: list[str] = []

    versions = {
        "tauri.conf.json": tauri_version,
        "Cargo.toml": cargo_version,
        "package.json": package_version,
    }
    unique_versions = sorted(set(versions.values()))
    if len(unique_versions) != 1:
        errors.append(
            "Desktop version mismatch: "
            + ", ".join(f"{name}={version}" for name, version in versions.items())
            + "."
        )

    expected_version = tauri_version

    if args.tag:
        normalized_tag = _normalize_tag(args.tag)
        if not normalized_tag.startswith(args.tag_prefix):
            errors.append(
                f"Release tag {normalized_tag!r} must start with {args.tag_prefix!r}."
            )
        else:
            tag_version = normalized_tag[len(args.tag_prefix) :]
            if tag_version != expected_version:
                errors.append(
                    "Release tag version mismatch: "
                    f"tag={tag_version!r}, desktop={expected_version!r}."
                )

    if errors:
        print("desktop-release-version: FAILED")
        for item in errors:
            print(f"- {item}")
        return 1

    print(f"desktop-release-version: OK ({expected_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
