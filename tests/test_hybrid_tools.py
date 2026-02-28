"""Tests for hybrid cowork fallback tools (list_tools + run_tool)."""

from __future__ import annotations

import json
from pathlib import Path

from loom.tools.list_tools import ListToolsTool
from loom.tools.registry import ToolContext, ToolResult
from loom.tools.run_tool import RunToolTool


def _fake_rows(count: int = 10) -> list[dict]:
    rows: list[dict] = []
    for idx in range(count):
        rows.append({
            "name": f"tool_{idx}",
            "summary": f"Summary {idx}",
            "description": f"Description {idx}",
            "category": "other",
            "mutates": bool(idx % 2),
            "auth_required": bool(idx % 3 == 0),
            "parameters": {
                "type": "object",
                "properties": {
                    "required_field": {"type": "string"},
                    "optional_field": {"type": "integer"},
                },
                "required": ["required_field"],
            },
        })
    return rows


class TestListToolsTool:
    async def test_unbound_fails(self):
        tool = ListToolsTool()
        result = await tool.execute({}, ToolContext(workspace=Path.cwd()))
        assert result.success is False
        assert "not bound" in (result.error or "")

    async def test_compact_mode_returns_full_filtered_set_without_paging(self):
        tool = ListToolsTool(catalog_provider=lambda _auth: _fake_rows(100))
        result = await tool.execute(
            {"limit": 1, "offset": 99, "max_payload_bytes": 65_536},
            ToolContext(workspace=Path.cwd()),
        )
        assert result.success is True
        assert isinstance(result.data, dict)
        payload = result.data
        assert payload["detail"] == "compact"
        assert payload["offset"] == 0
        assert payload["limit"] == 100
        assert payload["has_more"] is False
        assert payload["next_offset"] is None
        assert len(payload["tools"]) == 100
        encoded = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        assert encoded <= 65_536
        assert "required_args" in payload["tools"][0]

    async def test_compact_mode_fails_when_payload_budget_too_small(self):
        tool = ListToolsTool(catalog_provider=lambda _auth: _fake_rows(100))
        result = await tool.execute(
            {"max_payload_bytes": 1_200},
            ToolContext(workspace=Path.cwd()),
        )
        assert result.success is False
        assert "exceeds max_payload_bytes" in (result.error or "")

    async def test_schema_mode_includes_parameters(self):
        tool = ListToolsTool(catalog_provider=lambda _auth: _fake_rows(3))
        result = await tool.execute(
            {"detail": "schema", "query": "tool_1", "limit": 2},
            ToolContext(workspace=Path.cwd()),
        )
        assert result.success is True
        payload = result.data or {}
        assert payload["detail"] == "schema"
        assert len(payload["tools"]) == 1
        assert "parameters" in payload["tools"][0]

    async def test_schema_mode_requires_narrow_filter(self):
        tool = ListToolsTool(catalog_provider=lambda _auth: _fake_rows(3))
        result = await tool.execute(
            {"detail": "schema", "limit": 2},
            ToolContext(workspace=Path.cwd()),
        )
        assert result.success is False
        assert "requires a narrow filter" in (result.error or "")


class TestRunToolTool:
    async def test_unbound_fails(self):
        tool = RunToolTool()
        result = await tool.execute({"name": "read_file"}, ToolContext(workspace=Path.cwd()))
        assert result.success is False
        assert "not bound" in (result.error or "")

    async def test_self_call_guard(self):
        tool = RunToolTool(dispatcher=lambda _name, _args, _ctx: ToolResult.ok("ok"))
        result = await tool.execute(
            {"name": "run_tool", "arguments": {}},
            ToolContext(workspace=Path.cwd()),
        )
        assert result.success is False
        assert "cannot invoke itself" in (result.error or "")

    async def test_dispatch_success(self):
        async def _dispatch(name: str, arguments: dict, _ctx: ToolContext) -> ToolResult:
            return ToolResult.ok(
                f"ran:{name}",
                data={"echo": arguments},
            )

        tool = RunToolTool(dispatcher=_dispatch)
        result = await tool.execute(
            {"name": "read_file", "arguments": {"path": "README.md"}},
            ToolContext(workspace=Path.cwd()),
        )
        assert result.success is True
        assert "ran:read_file" in result.output
        assert isinstance(result.data, dict)
        assert result.data.get("delegated_tool") == "read_file"
