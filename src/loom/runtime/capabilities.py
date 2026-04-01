"""Runtime capability and optional-addon checks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class OptionalAddonStatus:
    """Status payload for one optional runtime addon."""

    key: str
    label: str
    installed: bool
    required_for: str
    install_hint: str
    detail: str = ""


def python_module_available(module_name: str) -> bool:
    """Return True when one Python module can be imported."""
    normalized = str(module_name or "").strip()
    if not normalized:
        return False
    return importlib.util.find_spec(normalized) is not None


def browser_addon_status() -> OptionalAddonStatus:
    """Return the Playwright/browser addon status."""
    installed = python_module_available("playwright")
    detail = (
        "Playwright package importable; browser binaries are validated at runtime."
        if installed
        else "Playwright package is not installed; browser_session will use the fallback engine."
    )
    return OptionalAddonStatus(
        key="browser",
        label="Browser Addon",
        installed=installed,
        required_for="Full JS-capable browser_session execution",
        install_hint="uv sync --extra browser",
        detail=detail,
    )


def treesitter_addon_status() -> OptionalAddonStatus:
    """Return the tree-sitter addon status."""
    installed = python_module_available("tree_sitter_language_pack")
    detail = (
        "tree-sitter-language-pack importable; analyze_code and edit_file can use "
        "structural parsing for supported languages."
        if installed
        else "tree-sitter-language-pack is not installed; analyze_code and edit_file "
        "will fall back to non-structural parsing or matching."
    )
    return OptionalAddonStatus(
        key="treesitter",
        label="Tree-sitter Addon",
        installed=installed,
        required_for="Structural code analysis and structured edit_file matching",
        install_hint="uv sync --extra treesitter",
        detail=detail,
    )


def mcp_addon_status() -> OptionalAddonStatus:
    """Return the MCP addon status."""
    installed = python_module_available("mcp")
    detail = (
        "MCP package importable; configured MCP servers can be loaded."
        if installed
        else "MCP package is not installed; MCP-backed tool discovery and external "
        "server integrations are unavailable."
    )
    return OptionalAddonStatus(
        key="mcp",
        label="MCP Addon",
        installed=installed,
        required_for="MCP-backed tool discovery and external MCP server integrations",
        install_hint="uv sync --extra mcp",
        detail=detail,
    )


def optional_addon_statuses() -> list[OptionalAddonStatus]:
    """Return known optional addon statuses in display order."""
    return [
        browser_addon_status(),
        treesitter_addon_status(),
        mcp_addon_status(),
    ]


def optional_addon_status_by_key(key: str) -> OptionalAddonStatus | None:
    """Return one optional addon status by stable key."""
    normalized = str(key or "").strip().lower()
    if not normalized:
        return None
    for status in optional_addon_statuses():
        if status.key == normalized:
            return status
    return None
