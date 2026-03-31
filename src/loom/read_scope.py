"""Helpers for explicit task read scopes and attached workspace context."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def normalize_workspace_relative_read_path(value: object) -> str:
    """Normalize a workspace-relative read path or return ``""`` when invalid."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return ""
    parts: list[str] = []
    for part in candidate.parts:
        clean = str(part or "").strip()
        if not clean or clean == ".":
            continue
        if clean == "..":
            return ""
        parts.append(clean)
    return "/".join(parts)


def iter_context_workspace_paths(context: dict[str, Any] | None) -> list[str]:
    """Return normalized attached workspace paths in stable user-facing order."""
    if not isinstance(context, dict):
        return []

    ordered: list[str] = []
    seen: set[str] = set()
    for key in ("workspace_paths", "workspace_directories", "workspace_files"):
        raw_values = context.get(key, [])
        if isinstance(raw_values, str):
            raw_values = [raw_values]
        if not isinstance(raw_values, list):
            continue
        for item in raw_values:
            normalized = normalize_workspace_relative_read_path(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def build_attached_read_scope(
    source_workspace_root: str | Path | None,
    context: dict[str, Any] | None,
) -> tuple[list[str], dict[str, str]]:
    """Resolve attached context paths into directory roots + exact path map."""
    if source_workspace_root is None or not isinstance(context, dict):
        return [], {}
    try:
        root = Path(source_workspace_root).expanduser().resolve()
    except Exception:
        return [], {}

    read_roots: list[str] = []
    read_path_map: dict[str, str] = {}
    seen_roots: set[str] = set()
    for relpath in iter_context_workspace_paths(context):
        try:
            candidate = (root / relpath).resolve()
            candidate.relative_to(root)
        except Exception:
            continue
        if not candidate.exists():
            continue
        read_path_map[relpath] = str(candidate)
        if candidate.is_dir():
            text = str(candidate)
            if text not in seen_roots:
                seen_roots.add(text)
                read_roots.append(text)
    return read_roots, read_path_map


def resolve_source_workspace_root(
    workspace: Path | None,
    metadata: dict[str, Any] | None,
) -> Path | None:
    """Return validated source workspace root when it scopes the run workspace."""
    if workspace is None or not isinstance(metadata, dict):
        return None
    raw = str(metadata.get("source_workspace_root", "") or "").strip()
    if not raw:
        return None
    try:
        source_root = Path(raw).expanduser().resolve()
        workspace_resolved = workspace.resolve()
    except Exception:
        return None
    if source_root == Path(source_root.anchor):
        return None
    if not _is_relative_to(workspace_resolved, source_root):
        return None
    return source_root


def resolve_read_roots_from_metadata(
    workspace: Path | None,
    metadata: dict[str, Any] | None,
) -> list[Path]:
    """Resolve validated read-only directory roots from task metadata."""
    if workspace is None or not isinstance(metadata, dict):
        return []
    try:
        workspace_resolved = workspace.resolve()
    except Exception:
        return []

    source_root = resolve_source_workspace_root(workspace_resolved, metadata)
    raw_roots = metadata.get("read_roots", [])
    if isinstance(raw_roots, str):
        raw_roots = [raw_roots]
    if not isinstance(raw_roots, list):
        return []

    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_roots:
        try:
            candidate = Path(str(raw or "")).expanduser().resolve()
        except Exception:
            continue
        if candidate == Path(candidate.anchor):
            continue
        allowed = _is_relative_to(workspace_resolved, candidate)
        if not allowed and source_root is not None:
            allowed = _is_relative_to(candidate, source_root)
        if not allowed:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        roots.append(candidate)
    return roots


def resolve_read_path_map_from_metadata(
    workspace: Path | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Path]:
    """Resolve validated attached path map from task metadata."""
    if workspace is None or not isinstance(metadata, dict):
        return {}
    try:
        workspace_resolved = workspace.resolve()
    except Exception:
        return {}

    source_root = resolve_source_workspace_root(workspace_resolved, metadata)
    read_roots = resolve_read_roots_from_metadata(workspace_resolved, metadata)
    raw_map = metadata.get("attached_read_path_map", {})
    if not isinstance(raw_map, dict):
        return {}

    resolved: dict[str, Path] = {}
    for raw_key, raw_value in raw_map.items():
        normalized_key = normalize_workspace_relative_read_path(raw_key)
        if not normalized_key:
            continue
        try:
            candidate = Path(str(raw_value or "")).expanduser().resolve()
        except Exception:
            continue
        if candidate == Path(candidate.anchor):
            continue
        allowed = _is_relative_to(candidate, workspace_resolved)
        if not allowed and source_root is not None:
            allowed = _is_relative_to(candidate, source_root)
        if not allowed:
            allowed = any(
                candidate == root or _is_relative_to(candidate, root)
                for root in read_roots
            )
        if not allowed:
            continue
        resolved[normalized_key] = candidate
    return resolved
