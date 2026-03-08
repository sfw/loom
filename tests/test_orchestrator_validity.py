"""Focused tests for extracted orchestrator validity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from loom.engine.orchestrator import validity as orchestrator_validity
from loom.engine.runner import ToolCallRecord
from loom.tools.registry import ToolResult


def test_normalize_validity_contract_clamps_defaults_and_modes() -> None:
    normalized = orchestrator_validity.normalize_validity_contract({
        "enabled": "yes",
        "claim_extraction": {"enabled": "true"},
        "critical_claim_types": ["numeric", "numeric", " date "],
        "min_supported_ratio": 1.4,
        "max_unverified_ratio": -0.5,
        "max_contradicted_count": -10,
        "prune_mode": "invalid",
        "final_gate": {
            "synthesis_min_verification_tier": 0,
            "critical_claim_support_ratio": "0.9",
            "temporal_consistency": {"enabled": "on"},
        },
    })

    assert normalized["enabled"] is True
    assert normalized["claim_extraction"]["enabled"] is True
    assert normalized["critical_claim_types"] == ["numeric", "date"]
    assert normalized["min_supported_ratio"] == 1.0
    assert normalized["max_unverified_ratio"] == 0.0
    assert normalized["max_contradicted_count"] == 0
    assert normalized["prune_mode"] == "drop"
    assert normalized["final_gate"]["synthesis_min_verification_tier"] == 1


def test_numeric_parsers_and_ratio_helpers() -> None:
    assert orchestrator_validity.to_int_or_none("1,200") == 1200
    assert orchestrator_validity.to_int_or_none("abc") is None
    assert orchestrator_validity.to_ratio_or_none("75%") == 0.75
    assert orchestrator_validity.to_ratio_or_none("200%") == 0.02
    assert orchestrator_validity.to_float_or_none("2.5") == 2.5
    assert orchestrator_validity.to_float_or_none("x") is None


def test_normalize_placeholder_findings_dedupes_and_caps() -> None:
    findings = [
        {"file_path": "report.md", "line": "1", "column": "2", "token": "N/A", "pattern": "N/A"},
        {"file_path": "report.md", "line": 1, "column": 2, "token": "N/A", "pattern": "N/A"},
        {"file_path": "report.md", "line": 3, "column": 1, "token": "TBD", "pattern": "TBD"},
    ]

    normalized = orchestrator_validity.normalize_placeholder_findings(findings, max_items=2)

    assert len(normalized) == 2
    assert normalized[0]["line"] == 1
    assert normalized[0]["column"] == 2


def test_normalize_workspace_relpath_rejects_outside_workspace(tmp_path: Path) -> None:
    inside = tmp_path / "inside.md"
    inside.write_text("ok", encoding="utf-8")
    outside = tmp_path.parent / "outside.md"
    outside.write_text("nope", encoding="utf-8")

    assert orchestrator_validity.normalize_workspace_relpath(tmp_path, "inside.md") == "inside.md"
    assert orchestrator_validity.normalize_workspace_relpath(tmp_path, str(inside)) == "inside.md"
    assert orchestrator_validity.normalize_workspace_relpath(tmp_path, str(outside)) is None


def test_hash_validity_contract_is_stable_for_same_payload_order() -> None:
    first = {"enabled": True, "critical_claim_types": ["numeric", "date"]}
    second = {"critical_claim_types": ["numeric", "date"], "enabled": True}

    assert orchestrator_validity.hash_validity_contract(first) == (
        orchestrator_validity.hash_validity_contract(second)
    )


def test_compact_failure_resolution_metadata_value_limits_depth_and_length() -> None:
    value = {
        "nested": {"deeper": {"value": "x" * 400}},
        "items": list(range(20)),
    }

    compacted = orchestrator_validity.compact_failure_resolution_metadata_value(
        value,
        max_depth=2,
        max_list_items=3,
        max_text_chars=40,
    )

    assert isinstance(compacted, dict)
    assert isinstance(compacted["items"], list)
    assert compacted["items"][-1] == "...[17 more items]"


def test_summarize_failure_resolution_metadata_prioritizes_known_keys() -> None:
    metadata = {
        "remediation_required": True,
        "missing_targets": ["report.md"],
        "deterministic_placeholder_scan": {
            "scan_mode": "targeted_only",
            "scanned_file_count": 1,
            "matched_file_count": 1,
        },
        "extra": "value",
    }

    summary = orchestrator_validity.summarize_failure_resolution_metadata(
        metadata,
        keys=("remediation_required", "missing_targets"),
    )

    assert summary["remediation_required"] is True
    assert summary["missing_targets"] == ["report.md"]
    assert "deterministic_placeholder_scan" in summary


def test_artifact_provenance_evidence_includes_generic_changed_files(tmp_path: Path) -> None:
    relpath = "reports/competitor-pricing.csv"
    artifact = tmp_path / relpath
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("name,price\nA,25\n", encoding="utf-8")
    calls = [
        ToolCallRecord(
            tool="spreadsheet",
            args={"operation": "create", "path": relpath},
            result=ToolResult.ok("ok", files_changed=[relpath]),
            call_id="call-1",
        ),
    ]

    records = orchestrator_validity._artifact_provenance_evidence(
        task_id="task-1",
        subtask_id="subtask-1",
        tool_calls=calls,
        existing_ids=set(),
        workspace=tmp_path,
    )

    assert len(records) == 1
    record = records[0]
    assert record["artifact_workspace_relpath"] == relpath
    assert record["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert record["tool"] == "spreadsheet"
