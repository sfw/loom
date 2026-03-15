"""Small TOML editing helpers for scalar loom.toml updates."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class ConfigPersistConflictError(RuntimeError):
    """Raised when loom.toml changed since the caller's last read."""


class ConfigPersistDisabledError(RuntimeError):
    """Raised when a persisted config write cannot be completed."""


def read_mtime_ns(path: Path | None) -> int | None:
    """Return nanosecond mtime for one path when available."""
    if path is None or not path.exists():
        return None
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def render_toml_scalar(value: object) -> str:
    """Render a basic TOML scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value or "")
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def upsert_scalar_value(
    text: str,
    *,
    section: str,
    key: str,
    rendered_value: str,
) -> str:
    """Insert or replace one scalar key in a single-level TOML section."""
    lines = text.splitlines()
    section_header = f"[{section}]"
    key_prefix = f"{key} ="
    start = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if line.strip() == section_header:
            start = idx
            break
    if start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([section_header, f"{key} = {rendered_value}"])
        return "\n".join(lines) + "\n"

    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = idx
            break
    for idx in range(start + 1, end):
        if lines[idx].strip().startswith(key_prefix):
            lines[idx] = f"{key} = {rendered_value}"
            return "\n".join(lines) + "\n"

    insert_at = end
    if insert_at > start + 1 and lines[insert_at - 1].strip():
        lines.insert(insert_at, f"{key} = {rendered_value}")
    else:
        lines.insert(insert_at, f"{key} = {rendered_value}")
    return "\n".join(lines) + "\n"


def remove_scalar_value(text: str, *, section: str, key: str) -> str:
    """Remove one scalar key from a single-level TOML section."""
    lines = text.splitlines()
    section_header = f"[{section}]"
    key_prefix = f"{key} ="
    start = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if line.strip() == section_header:
            start = idx
            break
    if start is None:
        return text if text.endswith("\n") or not text else text + "\n"
    for idx in range(start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = idx
            break
    kept = lines[:]
    for idx in range(start + 1, end):
        if kept[idx].strip().startswith(key_prefix):
            kept.pop(idx)
            break
    return "\n".join(kept).rstrip() + "\n"


def atomic_write_text(
    path: Path,
    *,
    text: str,
    expected_mtime_ns: int | None,
) -> int | None:
    """Atomically replace one file after optional mtime conflict detection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    temp_path: Path | None = None
    try:
        import fcntl
    except ImportError as e:  # pragma: no cover
        raise ConfigPersistDisabledError(
            "Persisted config writes are unavailable on this platform.",
        ) from e

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        current_mtime_ns = read_mtime_ns(path)
        if (
            expected_mtime_ns is not None
            and current_mtime_ns is not None
            and current_mtime_ns != expected_mtime_ns
        ):
            raise ConfigPersistConflictError(
                "loom.toml changed since the last config read; reload and retry.",
            )
        fd, temp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            temp_path = Path(temp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
    return read_mtime_ns(path)
