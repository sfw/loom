"""Focused tests for extracted orchestrator evidence helpers."""

from __future__ import annotations

from loom.engine.orchestrator import evidence as orchestrator_evidence


def test_stringify_evidence_csv_value_handles_scalars_and_nested() -> None:
    assert orchestrator_evidence.stringify_evidence_csv_value(None) == ""
    assert orchestrator_evidence.stringify_evidence_csv_value(42) == "42"
    nested = orchestrator_evidence.stringify_evidence_csv_value({"b": 2, "a": 1})
    assert nested == '{"a": 1, "b": 2}'


def test_evidence_csv_fieldnames_merges_base_and_sorted_extras() -> None:
    fieldnames = orchestrator_evidence.evidence_csv_fieldnames(
        base_fields=("task_id", "subtask_id"),
        rows=[
            {"task_id": "t1", "extra_b": "x"},
            {"subtask_id": "s1", "extra_a": "y"},
        ],
    )

    assert fieldnames == ["task_id", "subtask_id", "extra_a", "extra_b"]


def test_evidence_csv_rows_normalizes_records_to_string_maps() -> None:
    rows = orchestrator_evidence.evidence_csv_rows([
        {"task_id": "t1", "count": 2, "meta": {"k": "v"}},
        "skip-me",
        {"subtask_id": "s1"},
    ])

    assert len(rows) == 2
    assert rows[0]["count"] == "2"
    assert rows[0]["meta"] == '{"k": "v"}'
    assert rows[1]["subtask_id"] == "s1"
