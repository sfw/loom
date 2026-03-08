"""Focused tests for extracted orchestrator evidence helpers."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

from loom.engine.orchestrator import evidence as orchestrator_evidence
from loom.engine.runner import ToolCallRecord
from loom.state.task_state import Task
from loom.tools.registry import ToolResult


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


def test_record_artifact_seals_tracks_spreadsheet_mutations(tmp_path) -> None:
    relpath = "competitor-pricing.csv"
    artifact = tmp_path / relpath
    artifact.write_text("name,price\nA,10\n", encoding="utf-8")
    task = Task(id="task-1", goal="seal", workspace=str(tmp_path), metadata={})

    class _Stub:
        def _artifact_seal_registry(self, task):
            return orchestrator_evidence._artifact_seal_registry(self, task)

        def _is_intermediate_artifact_path(self, *, task, relpath):
            del task, relpath
            return False

        def _task_run_id(self, task):
            del task
            return "run-1"

        def _artifact_content_for_call(self, tool_name, args, result_data):
            return orchestrator_evidence._artifact_content_for_call(tool_name, args, result_data)

    stub = _Stub()
    calls = [
        ToolCallRecord(
            tool="spreadsheet",
            args={"operation": "create", "path": relpath},
            result=ToolResult.ok("ok", files_changed=[relpath]),
            call_id="call-1",
        ),
    ]
    updated = orchestrator_evidence._record_artifact_seals(
        stub,
        task=task,
        subtask_id="subtask-1",
        tool_calls=calls,
    )
    assert updated == 1
    seal = task.metadata["artifact_seals"][relpath]
    assert seal["tool"] == "spreadsheet"
    assert seal["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_backfill_artifact_seals_uses_generic_artifact_evidence(tmp_path) -> None:
    task = Task(id="task-1", goal="seal", workspace=str(tmp_path), metadata={})
    records = [{
        "tool": "spreadsheet",
        "artifact_workspace_relpath": "reports/pricing.csv",
        "artifact_sha256": "abc123",
        "artifact_size_bytes": 12,
        "tool_call_id": "call-2",
        "subtask_id": "subtask-2",
        "created_at": "2026-03-07T00:00:00",
    }]

    class _Stub:
        _state = SimpleNamespace(load_evidence_records=lambda task_id: records)

        def _artifact_seal_registry(self, task):
            return orchestrator_evidence._artifact_seal_registry(self, task)

        def _is_intermediate_artifact_path(self, *, task, relpath):
            del task, relpath
            return False

        def _task_run_id(self, task):
            del task
            return "run-1"

    stub = _Stub()
    updated = orchestrator_evidence._backfill_artifact_seals_from_evidence(stub, task)
    assert updated == 1
    assert task.metadata["artifact_seals"]["reports/pricing.csv"]["sha256"] == "abc123"


def test_validate_artifact_seals_recovers_after_spreadsheet_reseal(tmp_path) -> None:
    relpath = "competitor-pricing.csv"
    artifact = tmp_path / relpath
    artifact.write_text("name,price\nA,10\n", encoding="utf-8")
    task = Task(id="task-1", goal="seal", workspace=str(tmp_path), metadata={})

    class _Stub:
        _state = SimpleNamespace(load_evidence_records=lambda task_id: [])

        def _artifact_seal_registry(self, task):
            return orchestrator_evidence._artifact_seal_registry(self, task)

        def _is_intermediate_artifact_path(self, *, task, relpath):
            del task, relpath
            return False

        def _task_run_id(self, task):
            del task
            return "run-1"

        def _artifact_content_for_call(self, tool_name, args, result_data):
            return orchestrator_evidence._artifact_content_for_call(tool_name, args, result_data)

        def _backfill_artifact_seals_from_evidence(self, task):
            return orchestrator_evidence._backfill_artifact_seals_from_evidence(self, task)

    stub = _Stub()
    seed_calls = [
        ToolCallRecord(
            tool="write_file",
            args={"path": relpath, "content": artifact.read_text(encoding="utf-8")},
            result=ToolResult.ok("ok", files_changed=[relpath]),
            call_id="call-seed",
        ),
    ]
    updated = orchestrator_evidence._record_artifact_seals(
        stub,
        task=task,
        subtask_id="seed-subtask",
        tool_calls=seed_calls,
    )
    assert updated == 1

    artifact.write_text("name,price\nA,20\n", encoding="utf-8")
    passed, mismatches, validated = orchestrator_evidence._validate_artifact_seals(
        stub,
        task=task,
    )
    assert passed is False
    assert validated == 1
    assert mismatches[0]["reason"] == "artifact_seal_mismatch"

    spreadsheet_calls = [
        ToolCallRecord(
            tool="spreadsheet",
            args={"operation": "create", "path": relpath},
            result=ToolResult.ok("ok", files_changed=[relpath]),
            call_id="call-spreadsheet",
        ),
    ]
    resealed = orchestrator_evidence._record_artifact_seals(
        stub,
        task=task,
        subtask_id="spreadsheet-subtask",
        tool_calls=spreadsheet_calls,
    )
    assert resealed == 1

    passed, mismatches, validated = orchestrator_evidence._validate_artifact_seals(
        stub,
        task=task,
    )
    assert passed is True
    assert mismatches == []
    assert validated == 1
