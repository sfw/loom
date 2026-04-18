"""Tests for runtime capability and optional-addon helpers."""

from __future__ import annotations

from loom.runtime.capabilities import (
    browser_addon_status,
    mcp_addon_status,
    optional_addon_status_by_key,
    optional_addon_statuses,
    treesitter_addon_status,
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
    treesitter = optional_addon_status_by_key("treesitter")
    mcp = optional_addon_status_by_key("mcp")

    assert len(statuses) == 3
    assert browser is not None
    assert treesitter is not None
    assert mcp is not None
    assert browser.installed is True
    assert treesitter.installed is True
    assert mcp.installed is True
    assert "browser binaries are validated at runtime" in browser.detail


def test_treesitter_addon_status_reports_missing_when_module_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "loom.runtime.capabilities.python_module_available",
        lambda _module_name: False,
    )

    status = treesitter_addon_status()

    assert status.key == "treesitter"
    assert status.installed is False
    assert status.install_hint == "uv sync --extra treesitter"
    assert "edit_file" in status.detail


def test_mcp_addon_status_reports_missing_when_module_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "loom.runtime.capabilities.python_module_available",
        lambda _module_name: False,
    )

    status = mcp_addon_status()

    assert status.key == "mcp"
    assert status.installed is False
    assert status.install_hint == "uv sync --extra mcp"
    assert "MCP-backed tool discovery" in status.detail
