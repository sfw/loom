"""Tests for the ask_user tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.tools.ask_user import AskUserTool, normalize_ask_user_args
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
        assert result.data["question_type"] == "single_choice"
        assert result.data["options_v2"][0]["label"] == "Python"

    async def test_question_without_options(self, tool, ctx):
        result = await tool.execute(
            {"question": "What should the API look like?"},
            ctx,
        )
        assert result.success
        assert result.data["options"] == []
        assert result.data["question_type"] == "free_text"

    async def test_structured_options_are_supported(self, tool, ctx):
        result = await tool.execute(
            {
                "question": "Choose deployment target",
                "question_type": "single_choice",
                "options": [
                    {"id": "aws", "label": "AWS", "description": "Managed cloud"},
                    {"id": "gcp", "label": "GCP"},
                ],
                "allow_custom_response": False,
                "context_note": "Production launch depends on this",
                "urgency": "high",
                "default_option_id": "aws",
            },
            ctx,
        )
        assert result.success
        assert "AWS" in result.output
        assert result.data["options"] == ["AWS", "GCP"]
        assert result.data["options_v2"][0]["id"] == "aws"
        assert result.data["allow_custom_response"] is False
        assert result.data["context_note"] == "Production launch depends on this"
        assert result.data["urgency"] == "high"
        assert result.data["default_option_id"] == "aws"

    async def test_schema(self, tool):
        schema = tool.schema()
        assert schema["name"] == "ask_user"
        assert "question" in schema["parameters"]["properties"]
        assert "options" in schema["parameters"]["properties"]
        assert "question_type" in schema["parameters"]["properties"]
        assert "allow_custom_response" in schema["parameters"]["properties"]
        assert "question" in schema["parameters"]["required"]
        assert schema["x_supported_execution_surfaces"] == ["tui"]


class TestAskUserNormalization:
    def test_legacy_options_normalize_to_structured(self):
        normalized = normalize_ask_user_args({
            "question": "Pick one",
            "options": ["Python", "Rust"],
        })
        assert normalized["question_type"] == "single_choice"
        assert normalized["legacy_options"] == ["Python", "Rust"]
        assert normalized["options"][0]["label"] == "Python"
        assert normalized["options"][0]["id"] == "python"

    def test_multichoice_bounds_clamp_to_option_count(self):
        normalized = normalize_ask_user_args({
            "question": "Select stacks",
            "question_type": "multi_choice",
            "options": ["A", "B", "C"],
            "min_selections": 5,
            "max_selections": 9,
        })
        assert normalized["min_selections"] == 3
        assert normalized["max_selections"] == 3

    def test_invalid_type_and_urgency_fallback(self):
        normalized = normalize_ask_user_args({
            "question": "Clarify?",
            "question_type": "something_else",
            "urgency": "urgent-now",
        })
        assert normalized["question_type"] == "free_text"
        assert normalized["urgency"] == "normal"

    def test_unknown_default_option_id_is_dropped(self):
        normalized = normalize_ask_user_args({
            "question": "Pick",
            "options": ["One", "Two"],
            "default_option_id": "missing",
        })
        assert normalized["default_option_id"] == ""

    def test_options_string_normalizes_to_structured_options(self):
        normalized = normalize_ask_user_args({
            "question": "Choose language",
            "question_type": "single_choice",
            "options": "Python\nRust\nGo",
        })
        assert normalized["options"]
        assert [item["label"] for item in normalized["options"]] == [
            "Python",
            "Rust",
            "Go",
        ]
        assert normalized["question_type"] == "single_choice"

    def test_choices_alias_is_supported(self):
        normalized = normalize_ask_user_args({
            "question": "Choose runtime",
            "choices": ["Node.js", "Deno"],
        })
        assert [item["label"] for item in normalized["options"]] == [
            "Node.js",
            "Deno",
        ]
        assert normalized["question_type"] == "single_choice"

    def test_dict_options_map_is_supported(self):
        normalized = normalize_ask_user_args({
            "question": "Pick one",
            "question_type": "single_choice",
            "options": {
                "web_app": "Web application",
                "mobile_app": "Mobile application",
            },
        })
        assert [item["id"] for item in normalized["options"]] == [
            "web_app",
            "mobile_app",
        ]
        assert [item["label"] for item in normalized["options"]] == [
            "Web application",
            "Mobile application",
        ]
