"""Helpers for resolving external binary paths and reporting failures."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BinaryResolution:
    """Outcome of resolving one external binary."""

    binary_name: str
    path: str = ""
    source: str = ""
    error_code: str = ""
    message: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.path)


def normalize_binary_overrides(raw: object) -> dict[str, str]:
    """Normalize config-provided binary overrides into a clean string map."""
    overrides: dict[str, str] = {}
    if not isinstance(raw, dict):
        return overrides
    for key, value in raw.items():
        clean_key = str(key or "").strip().lower()
        clean_value = str(value or "").strip()
        if clean_key and clean_value:
            overrides[clean_key] = clean_value
    return overrides


def configured_binary_override(
    overrides: dict[str, str] | None,
    *keys: str,
) -> str:
    """Return the first configured override matching one of the provided keys."""
    normalized = normalize_binary_overrides(overrides or {})
    for key in keys:
        clean = str(key or "").strip().lower()
        if clean and clean in normalized:
            return normalized[clean]
    return ""


def resolve_binary(
    binary_name: str,
    *,
    override: str = "",
) -> BinaryResolution:
    """Resolve one binary via explicit override first, then PATH lookup."""
    clean_name = str(binary_name or "").strip()
    requested = str(override or "").strip()
    if requested:
        if _looks_like_explicit_path(requested):
            return _resolve_explicit_path(clean_name, requested)
        resolved = shutil.which(requested)
        if resolved:
            return BinaryResolution(
                binary_name=clean_name,
                path=resolved,
                source="configured_command",
                metadata={"requested": requested},
            )
        return BinaryResolution(
            binary_name=clean_name,
            error_code="binary_not_found",
            message=f"Configured binary override could not be resolved: {requested}",
            metadata={"requested": requested},
        )

    resolved = shutil.which(clean_name)
    if resolved:
        return BinaryResolution(
            binary_name=clean_name,
            path=resolved,
            source="path_lookup",
        )
    return BinaryResolution(
        binary_name=clean_name,
        error_code="binary_not_found",
        message=f"Binary not found: {clean_name}",
    )


def _looks_like_explicit_path(value: str) -> bool:
    if value.startswith(("~", ".", "/")):
        return True
    separators = {os.sep}
    if os.altsep:
        separators.add(os.altsep)
    return any(sep in value for sep in separators if sep)


def _resolve_explicit_path(binary_name: str, raw_path: str) -> BinaryResolution:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    if not candidate.exists():
        return BinaryResolution(
            binary_name=binary_name,
            error_code="binary_not_found",
            message=f"Configured binary path does not exist: {candidate}",
            metadata={"requested": raw_path},
        )
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return BinaryResolution(
            binary_name=binary_name,
            error_code="binary_not_executable",
            message=f"Configured binary path is not executable: {candidate}",
            metadata={"requested": raw_path},
        )
    return BinaryResolution(
        binary_name=binary_name,
        path=str(candidate),
        source="configured_path",
        metadata={"requested": raw_path},
    )
