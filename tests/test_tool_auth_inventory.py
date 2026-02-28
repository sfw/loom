"""Guard tests for first-party tool auth declarations."""

from __future__ import annotations

from loom.tools import create_default_registry, discover_tools
from loom.tools.auth_inventory import (
    FIRST_PARTY_TOOL_AUTH_CLASSIFICATION,
)
from loom.tools.registry import ToolRegistry


def _first_party_tool_names() -> set[str]:
    registry = ToolRegistry()
    for tool_cls in discover_tools():
        module_name = str(getattr(tool_cls, "__module__", "") or "")
        if not module_name.startswith("loom.tools."):
            continue
        registry.register(tool_cls())
    return {
        str(name)
        for name in registry.list_tools()
        if str(name).strip() and not str(name).startswith("mcp.")
    }


def test_inventory_covers_all_first_party_non_mcp_tools():
    discovered = _first_party_tool_names()
    declared = set(FIRST_PARTY_TOOL_AUTH_CLASSIFICATION.keys())
    assert discovered == declared


def test_inventory_uses_valid_classifications():
    allowed = {"no_auth", "optional_auth", "required_auth"}
    assert set(FIRST_PARTY_TOOL_AUTH_CLASSIFICATION.values()).issubset(allowed)


def test_optional_or_required_auth_tools_must_declare_auth_requirements():
    registry = create_default_registry()
    for tool_name, classification in FIRST_PARTY_TOOL_AUTH_CLASSIFICATION.items():
        tool = registry.get(tool_name)
        assert tool is not None, f"Tool not found in registry: {tool_name}"
        declared = list(getattr(tool, "auth_requirements", []) or [])
        if classification in {"optional_auth", "required_auth"}:
            assert declared, (
                f"{tool_name} is classified as {classification} but does not "
                "declare auth_requirements."
            )
        if declared:
            assert classification in {"optional_auth", "required_auth"}, (
                f"{tool_name} declares auth_requirements but is classified as "
                f"{classification}."
            )
