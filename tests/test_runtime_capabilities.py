"""Tests for runtime capability and optional-addon helpers."""

from __future__ import annotations

from loom.runtime.capabilities import (
    browser_addon_status,
    optional_addon_status_by_key,
    optional_addon_statuses,
)


def test_browser_addon_status_reports_missing_when_playwright_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "loom.runtime.capabilities.python_module_available",
        lambda _module_name: False,
    )

    status = browser_addon_status()

    assert status.key == "browser"
    assert status.installed is False
    assert status.install_hint == "uv sync --extra browser"
    assert "fallback engine" in status.detail


def test_optional_addon_lookup_returns_browser_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "loom.runtime.capabilities.python_module_available",
        lambda _module_name: True,
    )

    statuses = optional_addon_statuses()
    browser = optional_addon_status_by_key("browser")

    assert len(statuses) == 1
    assert browser is not None
    assert browser.installed is True
    assert "browser binaries are validated at runtime" in browser.detail
