"""Tests for the glob file finder tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.glob_find import GlobFindTool
from loom.tools.registry import ToolContext


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace with a variety of files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "src" / "utils.py").write_text("# utils")
    (tmp_path / "src" / "app.ts").write_text("// app")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("# test")
    (tmp_path / "README.md").write_text("# readme")
    (tmp_path / "config.yaml").write_text("key: value")
    # Create a dir that should be skipped
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "cached.pyc").write_bytes(b"cache")
    return tmp_path


@pytest.fixture
def tool():
    return GlobFindTool()


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace)


class TestGlobFind:
    async def test_find_python_files(self, tool, ctx):
        result = await tool.execute({"pattern": "**/*.py"}, ctx)
        assert result.success
        assert "main.py" in result.output
        assert "utils.py" in result.output
        assert "test_main.py" in result.output

    async def test_find_specific_pattern(self, tool, ctx):
        result = await tool.execute({"pattern": "**/test_*.py"}, ctx)
        assert result.success
        assert "test_main.py" in result.output
        assert "utils.py" not in result.output

    async def test_find_yaml(self, tool, ctx):
        result = await tool.execute({"pattern": "*.yaml"}, ctx)
        assert result.success
        assert "config.yaml" in result.output

    async def test_skips_pycache(self, tool, ctx):
        result = await tool.execute({"pattern": "**/*"}, ctx)
        assert result.success
        assert "cached.pyc" not in result.output
        assert "__pycache__" not in result.output

    async def test_no_matches(self, tool, ctx):
        result = await tool.execute({"pattern": "**/*.rs"}, ctx)
        assert result.success
        assert "No files matched" in result.output

    async def test_empty_pattern(self, tool, ctx):
        result = await tool.execute({"pattern": ""}, ctx)
        assert not result.success

    async def test_custom_path(self, tool, workspace):
        ctx = ToolContext(workspace=workspace)
        result = await tool.execute({
            "pattern": "*.py",
            "path": str(workspace / "src"),
        }, ctx)
        assert result.success
        assert "main.py" in result.output

    async def test_schema(self, tool):
        schema = tool.schema()
        assert schema["name"] == "glob_find"
        assert "pattern" in schema["parameters"]["properties"]

    async def test_data_has_count(self, tool, ctx):
        result = await tool.execute({"pattern": "**/*.py"}, ctx)
        assert result.success
        assert result.data["count"] == 3

    async def test_relative_path_with_workspace(self, tool, workspace):
        """Relative path='src' should resolve against workspace."""
        ctx = ToolContext(workspace=workspace)
        result = await tool.execute({
            "pattern": "*.py",
            "path": "src",
        }, ctx)
        assert result.success
        assert "main.py" in result.output
        assert "test_main.py" not in result.output
