"""Tests for the ask_user tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.ask_user import AskUserTool
from loom.tools.registry import ToolContext


@pytest.fixture
def tool():
    return AskUserTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


class TestAskUserTool:
    async def test_simple_question(self, tool, ctx):
        result = await tool.execute(
            {"question": "Which framework should I use?"},
            ctx,
        )
        assert result.success
        assert "Which framework should I use?" in result.output
        assert result.data["question"] == "Which framework should I use?"
        assert result.data["awaiting_input"] is True

    async def test_question_with_options(self, tool, ctx):
        result = await tool.execute(
            {
                "question": "Pick a language:",
                "options": ["Python", "Rust", "Go"],
            },
            ctx,
        )
        assert result.success
        assert "Pick a language:" in result.output
        assert "Python" in result.output
        assert result.data["options"] == ["Python", "Rust", "Go"]

    async def test_question_without_options(self, tool, ctx):
        result = await tool.execute(
            {"question": "What should the API look like?"},
            ctx,
        )
        assert result.success
        assert result.data["options"] == []

    async def test_schema(self, tool):
        schema = tool.schema()
        assert schema["name"] == "ask_user"
        assert "question" in schema["parameters"]["properties"]
        assert "options" in schema["parameters"]["properties"]
        assert "question" in schema["parameters"]["required"]
