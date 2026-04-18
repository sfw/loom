"""Runtime config state and persistence helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from loom.config import Config

from .registry import display_value, find_entry
from .schema import ConfigRuntimeEntry, ParsedConfigValue
from .toml_edit import (
    ConfigPersistConflictError,
    ConfigPersistDisabledError,
    atomic_write_text,
    read_mtime_ns,
    remove_scalar_value,
    render_toml_scalar,
    upsert_scalar_value,
)

_DEFAULT_CONFIG_WRITE_PATH = Path.home() / ".loom" / "loom.toml"


def _apply_entry_value(
    config: Config | None,
    *,
    entry: ConfigRuntimeEntry,
    value: Any,
) -> Config:
    base = config or Config()
    section_obj = getattr(base, entry.section)
    updated_section = replace(section_obj, **{entry.field: value})
    return replace(base, **{entry.section: updated_section})


class ConfigRuntimeStore:
    """Process-local runtime config base + override state."""

    def __init__(
        self,
        config: Config | None,
        *,
        source_path: Path | None = None,
    ) -> None:
        self._lock = RLock()
        self._base_config = config or Config()
        self._source_path = source_path
        self._source_mtime_ns = read_mtime_ns(source_path)
        self._runtime_overrides: dict[str, Any] = {}
        self._updated_at: dict[str, str] = {}

    def set_config(
        self,
        config: Config | None,
        *,
        source_path: Path | None = None,
    ) -> None:
        """Replace base config while preserving runtime overrides."""
        with self._lock:
            self._base_config = config or Config()
            if source_path is not None or self._source_path is None:
                self._source_path = source_path
            self._source_mtime_ns = read_mtime_ns(self._source_path)

    def base_config(self) -> Config:
        with self._lock:
            return self._base_config

    def source_path(self) -> Path | None:
        with self._lock:
            return self._source_path

    def persist_target_path(self) -> Path:
        with self._lock:
            return self._source_path or _DEFAULT_CONFIG_WRITE_PATH

    def effective_config(self) -> Config:
        with self._lock:
            effective = self._base_config
            for path, value in self._runtime_overrides.items():
                entry = find_entry(path)
                if entry is None:
                    continue
                effective = _apply_entry_value(effective, entry=entry, value=value)
            return effective

    def configured_value(self, path: str) -> Any:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        with self._lock:
            return getattr(getattr(self._base_config, entry.section), entry.field)

    def runtime_override_value(self, path: str) -> Any:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        with self._lock:
            return self._runtime_overrides.get(entry.path)

    def effective_value(self, path: str) -> Any:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        with self._lock:
            if entry.path in self._runtime_overrides:
                return self._runtime_overrides[entry.path]
            return getattr(getattr(self._base_config, entry.section), entry.field)

    def updated_at(self, path: str) -> str:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        with self._lock:
            return self._updated_at.get(entry.path, "")

    def set_runtime_value(
        self,
        path: str,
        raw_value: object,
    ) -> tuple[ConfigRuntimeEntry, ParsedConfigValue]:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        if not entry.supports_runtime:
            raise ConfigPersistDisabledError(
                f"Runtime overrides are not supported for {entry.path}.",
            )
        parsed = entry.parse(raw_value)
        now = datetime.now(UTC).isoformat()
        with self._lock:
            configured = getattr(getattr(self._base_config, entry.section), entry.field)
            if parsed.value == configured:
                self._runtime_overrides.pop(entry.path, None)
            else:
                self._runtime_overrides[entry.path] = parsed.value
            self._updated_at[entry.path] = now
        return entry, parsed

    def clear_runtime_value(self, path: str) -> ConfigRuntimeEntry:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        with self._lock:
            self._runtime_overrides.pop(entry.path, None)
            self._updated_at[entry.path] = datetime.now(UTC).isoformat()
        return entry

    def persist_value(
        self,
        path: str,
        raw_value: object,
    ) -> tuple[ConfigRuntimeEntry, ParsedConfigValue]:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        if not entry.supports_persist:
            raise ConfigPersistDisabledError(
                f"Persisted writes are not supported for {entry.path}.",
            )
        parsed = entry.parse(raw_value)
        target = self.persist_target_path()
        original_text = target.read_text(encoding="utf-8") if target.exists() else ""
        updated_text = upsert_scalar_value(
            original_text,
            section=entry.section,
            key=entry.field,
            rendered_value=render_toml_scalar(parsed.value),
        )
        with self._lock:
            new_mtime = atomic_write_text(
                target,
                text=updated_text,
                expected_mtime_ns=self._source_mtime_ns if target == self._source_path else None,
            )
            self._base_config = _apply_entry_value(
                self._base_config,
                entry=entry,
                value=parsed.value,
            )
            self._source_path = target
            self._source_mtime_ns = new_mtime
            if self._runtime_overrides.get(entry.path) == parsed.value:
                self._runtime_overrides.pop(entry.path, None)
            self._updated_at[entry.path] = datetime.now(UTC).isoformat()
        return entry, parsed

    def reset_persisted_value(self, path: str) -> ConfigRuntimeEntry:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        if not entry.supports_persist:
            raise ConfigPersistDisabledError(
                f"Persisted writes are not supported for {entry.path}.",
            )
        target = self.persist_target_path()
        original_text = target.read_text(encoding="utf-8") if target.exists() else ""
        updated_text = remove_scalar_value(
            original_text,
            section=entry.section,
            key=entry.field,
        )
        with self._lock:
            new_mtime = atomic_write_text(
                target,
                text=updated_text,
                expected_mtime_ns=self._source_mtime_ns if target == self._source_path else None,
            )
            self._base_config = _apply_entry_value(
                self._base_config,
                entry=entry,
                value=entry.default,
            )
            self._source_path = target
            self._source_mtime_ns = new_mtime
            self._updated_at[entry.path] = datetime.now(UTC).isoformat()
        return entry

    def snapshot(self, path: str) -> dict[str, Any]:
        entry = find_entry(path)
        if entry is None:
            raise KeyError(path)
        with self._lock:
            configured = getattr(getattr(self._base_config, entry.section), entry.field)
            runtime = self._runtime_overrides.get(entry.path)
            effective = runtime if runtime is not None else configured
            return {
                "path": entry.path,
                "section": entry.section,
                "field": entry.field,
                "description": entry.description,
                "supports_runtime": entry.supports_runtime,
                "supports_persist": entry.supports_persist,
                "application_class": entry.application_class,
                "requires_restart": entry.requires_restart,
                "exposure_level": entry.exposure_level,
                "configured": configured,
                "configured_display": display_value(
                    configured,
                    redact=entry.redact_in_output,
                ),
                "runtime_override": runtime,
                "runtime_display": display_value(
                    runtime,
                    redact=entry.redact_in_output,
                ),
                "effective": effective,
                "effective_display": display_value(
                    effective,
                    redact=entry.redact_in_output,
                ),
                "updated_at": self._updated_at.get(entry.path, ""),
                "source_path": str(self._source_path) if self._source_path else "",
            }


__all__ = [
    "ConfigPersistConflictError",
    "ConfigPersistDisabledError",
    "ConfigRuntimeStore",
]
