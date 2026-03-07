"""Shared path normalization helpers for engine policy checks."""

from __future__ import annotations

import re
from pathlib import Path


def normalize_path_for_policy(path_text: str, workspace: Path | None) -> str:
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_absolute():
        if workspace is None:
            return path.as_posix().lstrip("./")
        try:
            return path.resolve().relative_to(workspace.resolve()).as_posix()
        except Exception:
            return path.as_posix().lstrip("./")
    parts = [part for part in path.parts if part not in {"", "."}]
    if workspace is not None and parts and parts[0] == workspace.name:
        parts = parts[1:]
    if not parts:
        return ""
    return Path(*parts).as_posix().lstrip("./")


def normalize_deliverable_paths(
    expected_deliverables: list[str],
    *,
    workspace: Path | None,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in expected_deliverables:
        value = normalize_path_for_policy(str(item), workspace)
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def looks_like_deliverable_variant(
    *,
    candidate: str,
    canonical: str,
    variant_suffix_markers: tuple[str, ...],
) -> bool:
    cand = Path(candidate)
    base = Path(canonical)
    if cand == base:
        return False
    if cand.parent != base.parent or cand.suffix.lower() != base.suffix.lower():
        return False
    cand_stem = cand.stem.lower()
    base_stem = base.stem.lower()
    if not cand_stem.startswith(base_stem):
        return False
    remainder = cand_stem[len(base_stem):]
    if not remainder or remainder[0] not in {"-", "_"}:
        return False
    tail = remainder[1:]
    for marker in variant_suffix_markers:
        if tail == marker:
            return True
        if tail.startswith(marker):
            suffix = tail[len(marker):]
            if not suffix:
                return True
            if suffix.isdigit():
                return True
            if suffix.startswith("-") and suffix[1:].isdigit():
                return True
            if suffix.startswith("_") and suffix[1:].isdigit():
                return True
    return bool(re.fullmatch(r"[a-z]{1,4}\d+", tail))
