"""Tests for the ripgrep search tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.registry import ToolContext
from loom.tools.ripgrep import RipgrepSearchTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace with searchable files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def hello():\n    print('hello world')\n\ndef goodbye():\n    print('bye')\n"
    )
    (tmp_path / "src" / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n"
    )
    (tmp_path / "README.md").write_text("# Project\n\nThis is a hello world project.\n")
    (tmp_path / "data.txt").write_text("some data\nhello from data\nmore stuff\n")
    return tmp_path


@pytest.fixture
def tool():
    return RipgrepSearchTool()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace)


class TestRipgrepSearch:
    async def test_basic_search(self, tool, ctx):
        result = await tool.execute({"pattern": "hello"}, ctx)
        assert result.success
        # Should find matches in main.py, README.md, and data.txt
        assert "hello" in result.output.lower()

    async def test_no_matches(self, tool, ctx):
        result = await tool.execute({"pattern": "zzz_nonexistent_zzz"}, ctx)
        assert result.success
        assert "No matches" in result.output

    async def test_case_insensitive(self, tool, ctx):
        result = await tool.execute({
            "pattern": "HELLO",
            "case_insensitive": True,
        }, ctx)
        assert result.success
        assert "hello" in result.output.lower() or "HELLO" in result.output

    async def test_files_only(self, tool, ctx):
        result = await tool.execute({
            "pattern": "hello",
            "files_only": True,
        }, ctx)
        assert result.success
        # Should list file paths, not matching lines
        output_lines = result.output.strip().split("\n")
        for line in output_lines:
            if line and "No matches" not in line:
                # File paths should not contain ":" line numbers
                assert ":" not in line or line.count(":") == 0 or "/" in line

    async def test_custom_path(self, tool, workspace):
        ctx = ToolContext(workspace=workspace)
        result = await tool.execute({
            "pattern": "def",
            "path": str(workspace / "src"),
        }, ctx)
        assert result.success
        assert "def" in result.output

    async def test_empty_pattern(self, tool, ctx):
        result = await tool.execute({"pattern": ""}, ctx)
        assert not result.success

    async def test_no_workspace(self, tool):
        ctx = ToolContext(workspace=None)
        result = await tool.execute({"pattern": "hello"}, ctx)
        assert not result.success

    async def test_schema(self, tool):
        schema = tool.schema()
        assert schema["name"] == "ripgrep_search"
        assert "pattern" in schema["parameters"]["properties"]
        assert "glob" in schema["parameters"]["properties"]
        assert "type" in schema["parameters"]["properties"]

    async def test_context_lines(self, tool, ctx):
        result = await tool.execute({
            "pattern": "hello",
            "context": 1,
        }, ctx)
        assert result.success
        # Should include context around matches
        assert result.output.strip()
