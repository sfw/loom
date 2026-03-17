"""Focused tests for runtime config registry/store helpers."""

from __future__ import annotations

from pathlib import Path

from loom.config import Config, TelemetryConfig
from loom.config_runtime import ConfigRuntimeStore, find_entry, search_entries


def test_find_entry_and_search_entries() -> None:
    entry = find_entry("execution.delegate_task_timeout_seconds")
    assert entry is not None
    assert entry.application_class == "live"

    matches = search_entries("timeout")
    paths = [item.path for item in matches]
    assert "execution.delegate_task_timeout_seconds" in paths
    assert "tui.run_launch_timeout_seconds" in paths


def test_runtime_store_set_runtime_value_updates_effective_config() -> None:
    store = ConfigRuntimeStore(
        Config(
            telemetry=TelemetryConfig(
                mode="active",
                configured_mode_input="active",
            ),
        ),
    )

    entry, parsed = store.set_runtime_value("telemetry.mode", "internal_only")

    assert entry.path == "telemetry.mode"
    assert parsed.value == "all_typed"
    assert parsed.warning_code == "telemetry_mode_alias_normalized"
    assert store.snapshot("telemetry.mode")["effective"] == "all_typed"
    assert store.effective_config().telemetry.mode == "all_typed"


def test_runtime_store_persist_and_reset_scalar_value(tmp_path: Path) -> None:
    config_path = tmp_path / "loom.toml"
    store = ConfigRuntimeStore(Config(), source_path=config_path)

    entry, parsed = store.persist_value("execution.delegate_task_timeout_seconds", "7200")

    assert entry.path == "execution.delegate_task_timeout_seconds"
    assert parsed.value == 7200
    assert "delegate_task_timeout_seconds = 7200" in config_path.read_text(encoding="utf-8")
    assert store.snapshot(entry.path)["configured"] == 7200

    reset_entry = store.reset_persisted_value(entry.path)

    assert reset_entry.path == entry.path
    assert "delegate_task_timeout_seconds" not in config_path.read_text(encoding="utf-8")
    assert store.snapshot(entry.path)["configured"] == 14400
