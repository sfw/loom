"""Focused parser/validation tests for semantic compactor internals."""

from __future__ import annotations

from loom.engine.semantic_compactor import SemanticCompactor
from loom.engine.semantic_compactor import parse as compactor_parse


def test_extract_compacted_text_handles_fenced_json() -> None:
    raw = """```json
{"compressed_text":"compacted from fenced payload"}
```"""

    compacted, error = compactor_parse.extract_compacted_text(raw)

    assert error == ""
    assert compacted == "compacted from fenced payload"


def test_extract_compacted_text_recovers_embedded_json_object() -> None:
    raw = (
        "assistant preface that should be ignored\n"
        '{"compressed_text":"embedded compacted output"}\n'
        "trailing explanation"
    )

    compacted, error = compactor_parse.extract_compacted_text(raw)

    assert error == ""
    assert compacted == "embedded compacted output"


def test_extract_partial_compressed_text_recovers_truncated_json_for_length_finish() -> None:
    raw = '{"compressed_text":"Recovered fragment with newline\\nand quote: \\"ok'

    compacted, error = compactor_parse.extract_compacted_text(raw)
    assert compacted is None
    assert error == "invalid_json"

    partial = compactor_parse.extract_partial_compressed_text(raw)
    assert partial == 'Recovered fragment with newline\nand quote: "ok'


def test_semantic_compactor_wrapper_validation_is_compatible() -> None:
    reason = SemanticCompactor._validate_compacted_text(
        "The user wants me to summarize this task.",
        limit=240,
    )

    assert reason == "meta_commentary_prefix"
