"""Guard tests for dynamic first-party tool auth declarations."""

from __future__ import annotations

import pytest

from loom.tools import discover_tools
from loom.tools.registry import Tool, ToolAuthMode, ToolContext, ToolRegistry, ToolResult


def _first_party_tools() -> dict[str, Tool]:
    registry = ToolRegistry()
    for tool_cls in discover_tools():
        module_name = str(getattr(tool_cls, "__module__", "") or "")
        if not module_name.startswith("loom.tools."):
            continue
        registry.register(tool_cls())
    result: dict[str, Tool] = {}
    for name in registry.list_tools():
        clean = str(name).strip()
        if not clean or clean.startswith("mcp."):
            continue
        tool = registry.get(clean)
        if tool is None:
            continue
        result[clean] = tool
    return result


def test_first_party_discovery_not_empty():
    tools = _first_party_tools()
    assert tools
    assert "read_file" in tools


def test_all_first_party_tools_use_valid_auth_modes():
    tools = _first_party_tools()
    allowed = {"no_auth", "optional_auth", "required_auth"}
    for tool_name, tool in tools.items():
        mode = str(getattr(tool, "auth_mode", "") or "").strip().lower()
        assert mode in allowed, f"{tool_name} has invalid auth_mode {mode!r}"


def test_auth_mode_and_requirements_contract_holds_for_all_first_party_tools():
    tools = _first_party_tools()
    for tool_name, tool in tools.items():
        mode = str(getattr(tool, "auth_mode", "") or "").strip().lower()
        declared = list(getattr(tool, "auth_requirements", []) or [])
        if mode in {"optional_auth", "required_auth"}:
            assert declared, (
                f"{tool_name} is classified as {mode} but does not "
                "declare auth_requirements."
            )
        if mode == "no_auth":
            assert not declared, (
                f"{tool_name} declares auth_requirements but auth_mode is no_auth."
            )


class _DummyToolBase(Tool):
    __loom_register__ = False

    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "dummy"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, _args: dict, _ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("ok")


def test_registry_rejects_invalid_auth_mode():
    class _InvalidModeTool(_DummyToolBase):
        @property
        def auth_mode(self):  # intentional invalid return type/value
            return "maybe"

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="invalid auth_mode"):
        registry.register(_InvalidModeTool())


def test_registry_rejects_required_auth_without_requirements():
    class _RequiredWithoutRequirementsTool(_DummyToolBase):
        @property
        def auth_mode(self) -> ToolAuthMode:
            return "required_auth"

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="does not declare auth_requirements"):
        registry.register(_RequiredWithoutRequirementsTool())


def test_registry_rejects_no_auth_with_requirements():
    class _NoAuthWithRequirementsTool(_DummyToolBase):
        @property
        def auth_mode(self) -> ToolAuthMode:
            return "no_auth"

        @property
        def auth_requirements(self) -> list[dict]:
            return [{"provider": "demo", "source": "api"}]

    registry = ToolRegistry()
    with pytest.raises(ValueError, match="auth_mode='no_auth'"):
        registry.register(_NoAuthWithRequirementsTool())


def test_registry_accepts_optional_auth_with_requirements():
    class _OptionalAuthTool(_DummyToolBase):
        @property
        def auth_mode(self) -> ToolAuthMode:
            return "optional_auth"

        @property
        def auth_requirements(self) -> list[dict]:
            return [{"provider": "demo", "source": "api"}]

    registry = ToolRegistry()
    registry.register(_OptionalAuthTool())
    assert registry.get("dummy_tool") is not None
