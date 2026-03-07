"""Focused tests for extracted runner compaction helpers."""

from __future__ import annotations

import json

from loom.engine.runner import SubtaskRunner
from loom.engine.runner import compaction as runner_compaction


class _CompactionRunnerStub:
    OVERFLOW_FALLBACK_TOOL_MESSAGE_MIN_CHARS = 4000
    OVERFLOW_FALLBACK_TOOL_OUTPUT_EXCERPT_CHARS = 1200
    _OVERFLOW_BINARY_CONTENT_KINDS = SubtaskRunner._OVERFLOW_BINARY_CONTENT_KINDS
    _HEAVY_OUTPUT_TOOLS = SubtaskRunner._HEAVY_OUTPUT_TOOLS

    def __init__(self) -> None:
        self._overflow_fallback_tool_message_min_chars = 120
        self._overflow_fallback_tool_output_excerpt_chars = 60


def test_is_model_request_overflow_error_matches_known_markers() -> None:
    assert runner_compaction.is_model_request_overflow_error(
        "Total message size exceeds limit",
    )
    assert runner_compaction.is_model_request_overflow_error(
        "context_length_exceeded",
    )
    assert not runner_compaction.is_model_request_overflow_error("random failure")


def test_rewrite_tool_payload_for_overflow_rewrites_large_json_payload() -> None:
    runner = _CompactionRunnerStub()
    content = json.dumps({
        "success": True,
        "output": "x" * 500,
        "error": None,
        "files_changed": [f"file_{idx}.txt" for idx in range(12)],
        "data": {"content_kind": "pdf", "artifact_ref": "artifact://abc"},
    })

    rewritten, delta = runner_compaction.rewrite_tool_payload_for_overflow(
        runner,
        content=content,
        tool_name="read_file",
    )

    assert rewritten is not None
    assert delta > 0
    payload = json.loads(rewritten)
    assert payload["data"]["overflow_fallback"] is True
    assert payload["data"]["tool_name"] == "read_file"
    assert "overflow fallback applied" in payload["output"]


def test_rewrite_tool_payload_for_overflow_skips_small_payload() -> None:
    runner = _CompactionRunnerStub()
    content = json.dumps({"success": True, "output": "short"})

    rewritten, delta = runner_compaction.rewrite_tool_payload_for_overflow(
        runner,
        content=content,
        tool_name="read_file",
    )

    assert rewritten is None
    assert delta == 0
