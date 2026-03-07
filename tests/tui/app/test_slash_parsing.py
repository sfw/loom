from __future__ import annotations

from loom.tui.app.models import SlashCommandSpec
from loom.tui.app.slash import parsing, registry


def test_split_slash_args_handles_quotes() -> None:
    assert parsing.split_slash_args('one "two words" three') == ["one", "two words", "three"]


def test_split_slash_args_forgiving_unclosed_quote() -> None:
    assert parsing.split_slash_args_forgiving("'hello world") == ["'hello world"]


def test_parse_tool_slash_arguments_json_and_kv() -> None:
    parsed_json, err_json = parsing.parse_tool_slash_arguments('{"path":"README.md","limit":2}')
    assert err_json == ""
    assert parsed_json == {"path": "README.md", "limit": 2}

    parsed_kv, err_kv = parsing.parse_tool_slash_arguments("path=README.md limit=2 force=true")
    assert err_kv == ""
    assert parsed_kv == {"path": "README.md", "limit": 2, "force": True}


def test_registry_matching_and_completion() -> None:
    specs = [
        SlashCommandSpec(canonical="/help", description="help"),
        SlashCommandSpec(canonical="/history", description="history", aliases=("/h",)),
    ]
    ordered = registry.ordered_slash_specs(specs, priority={"/history": 1, "/help": 2})
    token, matches = registry.matching_slash_commands(
        "/hi",
        ordered_specs=ordered,
        process_command_map={"/plan": "plan"},
    )
    assert token == "/hi"
    assert matches and matches[0][0].startswith("/history")

    candidates = registry.slash_completion_candidates(
        "/h",
        ordered_specs=ordered,
        process_command_map={"/plan": "plan"},
    )
    assert "/help" in candidates
    assert "/history" in candidates
