"""Artifact persistence for fetched binary/document payloads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

DEFAULT_RETENTION_MAX_AGE_DAYS = 14
DEFAULT_RETENTION_MAX_FILES_PER_SCOPE = 96
DEFAULT_RETENTION_MAX_BYTES_PER_SCOPE = 256 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactRecord:
    """Stored artifact metadata."""

    artifact_ref: str
    path: Path
    source_url: str
    media_type: str
    content_kind: str
    size_bytes: int
    created_at: str
    workspace_relpath: str = ""
    cleanup_stats: dict[str, int] = field(default_factory=dict)


_MEDIA_TYPE_EXTENSION = {
    "application/pdf": ".pdf",
    "application/json": ".json",
    "application/xml": ".xml",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/gzip": ".gz",
    "application/x-gzip": ".gz",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "text/html": ".html",
    "text/plain": ".txt",
}


def _default_artifact_root(
    *,
    workspace: Path | None,
    scratch_dir: Path | None,
) -> Path:
    if workspace is not None:
        try:
            return workspace.resolve() / ".loom_artifacts" / "fetched"
        except Exception:
            pass
    if scratch_dir is not None:
        try:
            return scratch_dir.expanduser().resolve() / "fetched_artifacts"
        except Exception:
            pass
    return Path("~/.loom/scratch/fetched_artifacts").expanduser()


def _safe_scope(subtask_id: str) -> str:
    value = str(subtask_id or "").strip()
    if not value:
        return "default"
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return cleaned or "default"


def _iter_scope_dirs(root: Path) -> list[Path]:
    try:
        if not root.exists() or not root.is_dir():
            return []
        dirs = [entry for entry in root.iterdir() if entry.is_dir()]
    except OSError:
        return []
    return sorted(dirs, key=lambda p: p.name)


def _guess_extension(*, media_type: str, source_url: str) -> str:
    parsed = urlparse(source_url or "")
    suffix = Path(parsed.path).suffix.lower()
    if suffix:
        return suffix
    ext = _MEDIA_TYPE_EXTENSION.get(str(media_type or "").strip().lower())
    return ext or ".bin"


def _read_manifest_entries(scope_dir: Path) -> list[dict]:
    manifest_path = scope_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    entries: list[dict] = []
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    entries.append(parsed)
    except OSError:
        return []
    return entries


def _write_manifest_entries(scope_dir: Path, entries: list[dict]) -> None:
    manifest_path = scope_dir / "manifest.jsonl"
    if not entries:
        try:
            manifest_path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    tmp_path = scope_dir / "manifest.jsonl.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")
    tmp_path.replace(manifest_path)


def _find_manifest_entry(scope_dir: Path, artifact_ref: str) -> dict | None:
    for entry in reversed(_read_manifest_entries(scope_dir)):
        if str(entry.get("artifact_ref", "")).strip() == artifact_ref:
            return entry
    return None


def _collect_scope_files(scope_dir: Path) -> list[tuple[Path, int, float]]:
    items: list[tuple[Path, int, float]] = []
    try:
        entries = list(scope_dir.iterdir())
    except OSError:
        return items
    for entry in entries:
        if not entry.is_file() or entry.name == "manifest.jsonl":
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        items.append((entry, int(stat.st_size), float(stat.st_mtime)))
    return items


def _delete_file(path: Path) -> bool:
    try:
        path.unlink()
    except OSError:
        return False
    return True


def _cleanup_scope_artifacts(
    scope_dir: Path,
    *,
    max_age_days: int,
    max_files_per_scope: int,
    max_bytes_per_scope: int,
) -> dict[str, int]:
    stats = {"files_deleted": 0, "bytes_deleted": 0}
    items = _collect_scope_files(scope_dir)
    if not items:
        _prune_scope_manifest(scope_dir)
        return stats

    if max_age_days > 0:
        cutoff_ts = (datetime.now(UTC) - timedelta(days=max_age_days)).timestamp()
        stale = [item for item in items if item[2] < cutoff_ts]
        for path, size, _mtime in stale:
            if _delete_file(path):
                stats["files_deleted"] += 1
                stats["bytes_deleted"] += size
        items = _collect_scope_files(scope_dir)

    if max_files_per_scope > 0 or max_bytes_per_scope > 0:
        items = sorted(items, key=lambda t: t[2])  # oldest first
        total_bytes = sum(size for _path, size, _mtime in items)
        while items and (
            (max_files_per_scope > 0 and len(items) > max_files_per_scope)
            or (max_bytes_per_scope > 0 and total_bytes > max_bytes_per_scope)
        ):
            path, size, _mtime = items.pop(0)
            if _delete_file(path):
                stats["files_deleted"] += 1
                stats["bytes_deleted"] += size
                total_bytes = max(0, total_bytes - size)

    _prune_scope_manifest(scope_dir)
    return stats


def _prune_scope_manifest(scope_dir: Path) -> None:
    entries = _read_manifest_entries(scope_dir)
    if not entries:
        return

    kept: list[dict] = []
    for entry in entries:
        artifact_ref = str(entry.get("artifact_ref", "")).strip()
        if not artifact_ref:
            continue
        raw_path = str(entry.get("path", "")).strip()
        if not raw_path:
            continue
        try:
            exists = Path(raw_path).exists()
        except OSError:
            exists = False
        if not exists:
            continue
        kept.append(entry)

    _write_manifest_entries(scope_dir, kept)


def cleanup_fetch_artifacts(
    *,
    workspace: Path | None = None,
    scratch_dir: Path | None = None,
    subtask_id: str = "",
    max_age_days: int = DEFAULT_RETENTION_MAX_AGE_DAYS,
    max_files_per_scope: int = DEFAULT_RETENTION_MAX_FILES_PER_SCOPE,
    max_bytes_per_scope: int = DEFAULT_RETENTION_MAX_BYTES_PER_SCOPE,
    scan_all_scopes: bool = False,
) -> dict[str, int]:
    """Best-effort artifact retention cleanup under the selected artifact root."""
    max_age_days = max(0, int(max_age_days))
    max_files_per_scope = max(0, int(max_files_per_scope))
    max_bytes_per_scope = max(0, int(max_bytes_per_scope))

    root = _default_artifact_root(workspace=workspace, scratch_dir=scratch_dir)
    stats = {"scopes_scanned": 0, "files_deleted": 0, "bytes_deleted": 0}
    if not root.exists():
        return stats

    scope_dirs: list[Path] = []
    if subtask_id:
        scoped = root / _safe_scope(subtask_id)
        if scoped.exists() and scoped.is_dir():
            scope_dirs.append(scoped)
    if scan_all_scopes or not scope_dirs:
        for scope_dir in _iter_scope_dirs(root):
            if scope_dir not in scope_dirs:
                scope_dirs.append(scope_dir)

    for scope_dir in scope_dirs:
        scoped_stats = _cleanup_scope_artifacts(
            scope_dir,
            max_age_days=max_age_days,
            max_files_per_scope=max_files_per_scope,
            max_bytes_per_scope=max_bytes_per_scope,
        )
        stats["scopes_scanned"] += 1
        stats["files_deleted"] += int(scoped_stats.get("files_deleted", 0))
        stats["bytes_deleted"] += int(scoped_stats.get("bytes_deleted", 0))
    return stats


def resolve_fetch_artifact(
    *,
    artifact_ref: str,
    workspace: Path | None = None,
    scratch_dir: Path | None = None,
    subtask_id: str = "",
) -> ArtifactRecord | None:
    """Resolve artifact metadata/path from stored manifests/files."""
    normalized_ref = str(artifact_ref or "").strip()
    if not normalized_ref:
        return None

    root = _default_artifact_root(workspace=workspace, scratch_dir=scratch_dir)
    if not root.exists():
        return None

    candidate_scopes: list[Path] = []
    if subtask_id:
        scope_dir = root / _safe_scope(subtask_id)
        if scope_dir.exists() and scope_dir.is_dir():
            candidate_scopes.append(scope_dir)
    for scope_dir in _iter_scope_dirs(root):
        if scope_dir not in candidate_scopes:
            candidate_scopes.append(scope_dir)

    for scope_dir in candidate_scopes:
        resolved = _resolve_fetch_artifact_in_scope(
            artifact_ref=normalized_ref,
            scope_dir=scope_dir,
            workspace=workspace,
        )
        if resolved is not None:
            return resolved
    return None


def _resolve_fetch_artifact_in_scope(
    *,
    artifact_ref: str,
    scope_dir: Path,
    workspace: Path | None,
) -> ArtifactRecord | None:
    try:
        candidates = [
            path
            for path in scope_dir.glob(f"{artifact_ref}.*")
            if path.is_file() and path.name != "manifest.jsonl"
        ]
    except OSError:
        candidates = []
    if not candidates:
        return None

    def _mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    artifact_path = max(candidates, key=_mtime)
    manifest_entry = _find_manifest_entry(scope_dir, artifact_ref) or {}

    try:
        size_bytes = int(artifact_path.stat().st_size)
    except OSError:
        size_bytes = 0

    created_at = str(manifest_entry.get("created_at", "")).strip()
    if not created_at:
        created_at = datetime.fromtimestamp(
            _mtime(artifact_path),
            tz=UTC,
        ).isoformat()

    workspace_relpath = str(manifest_entry.get("workspace_relpath", "")).strip()
    if not workspace_relpath and workspace is not None:
        try:
            workspace_relpath = str(
                artifact_path.resolve().relative_to(workspace.resolve()),
            )
        except Exception:
            workspace_relpath = ""

    return ArtifactRecord(
        artifact_ref=artifact_ref,
        path=artifact_path.resolve(),
        source_url=str(manifest_entry.get("source_url", "")).strip(),
        media_type=str(manifest_entry.get("media_type", "")).strip(),
        content_kind=str(manifest_entry.get("content_kind", "")).strip(),
        size_bytes=size_bytes,
        created_at=created_at,
        workspace_relpath=workspace_relpath,
    )


def persist_fetch_artifact(
    *,
    content_bytes: bytes,
    source_url: str,
    media_type: str,
    content_kind: str,
    workspace: Path | None = None,
    scratch_dir: Path | None = None,
    subtask_id: str = "",
    retention_max_age_days: int = DEFAULT_RETENTION_MAX_AGE_DAYS,
    retention_max_files_per_scope: int = DEFAULT_RETENTION_MAX_FILES_PER_SCOPE,
    retention_max_bytes_per_scope: int = DEFAULT_RETENTION_MAX_BYTES_PER_SCOPE,
) -> ArtifactRecord:
    """Persist fetched bytes and append a run-scope manifest entry."""
    now = datetime.now(UTC).isoformat()
    artifact_ref = f"af_{uuid4().hex[:16]}"
    scope = _safe_scope(subtask_id)
    root = _default_artifact_root(workspace=workspace, scratch_dir=scratch_dir)
    scope_dir = root / scope
    scope_dir.mkdir(parents=True, exist_ok=True)

    ext = _guess_extension(media_type=media_type, source_url=source_url)
    artifact_path = scope_dir / f"{artifact_ref}{ext}"
    artifact_path.write_bytes(bytes(content_bytes))

    relpath = ""
    if workspace is not None:
        try:
            relpath = str(artifact_path.resolve().relative_to(workspace.resolve()))
        except Exception:
            relpath = ""

    resolved_path = artifact_path.resolve()
    source_url_text = str(source_url or "")
    media_type_text = str(media_type or "")
    content_kind_text = str(content_kind or "")
    size_bytes = len(content_bytes)

    manifest_path = scope_dir / "manifest.jsonl"
    manifest_entry = {
        "artifact_ref": artifact_ref,
        "path": str(resolved_path),
        "workspace_relpath": relpath,
        "source_url": source_url_text,
        "media_type": media_type_text,
        "content_kind": content_kind_text,
        "size_bytes": size_bytes,
        "created_at": now,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest_entry, ensure_ascii=False))
        handle.write("\n")

    # Run scope cleanup every write and periodically sweep all scopes.
    scan_all_scopes = artifact_ref[-1] in {"0", "1"}
    cleanup_stats = cleanup_fetch_artifacts(
        workspace=workspace,
        scratch_dir=scratch_dir,
        subtask_id=subtask_id,
        max_age_days=retention_max_age_days,
        max_files_per_scope=retention_max_files_per_scope,
        max_bytes_per_scope=retention_max_bytes_per_scope,
        scan_all_scopes=scan_all_scopes,
    )

    return ArtifactRecord(
        artifact_ref=artifact_ref,
        path=resolved_path,
        source_url=source_url_text,
        media_type=media_type_text,
        content_kind=content_kind_text,
        size_bytes=size_bytes,
        created_at=now,
        workspace_relpath=relpath,
        cleanup_stats={
            "scopes_scanned": int(cleanup_stats.get("scopes_scanned", 0)),
            "files_deleted": int(cleanup_stats.get("files_deleted", 0)),
            "bytes_deleted": int(cleanup_stats.get("bytes_deleted", 0)),
        },
    )
