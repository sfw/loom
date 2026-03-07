"""Focused tests for extracted placeholder-guard helpers."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from loom.engine.verification.placeholder_guard import (
    apply_placeholder_contradiction_guard,
    expected_deliverables_for_subtask,
    is_placeholder_claim_failure,
    normalize_candidate_path,
    normalized_scan_suffixes,
    scan_placeholder_markers,
)
from loom.engine.verification.types import VerificationResult


def test_is_placeholder_claim_failure_uses_reason_code_and_feedback_markers() -> None:
    gates = SimpleNamespace(
        _PLACEHOLDER_CLAIM_REASON_CODES={"incomplete_deliverable_placeholder"},
        _PLACEHOLDER_MARKER_PATTERN=re.compile(r"TODO|\\[TBD\\]", re.IGNORECASE),
    )
    by_reason = VerificationResult(
        tier=2,
        passed=False,
        outcome="fail",
        reason_code="incomplete_deliverable_placeholder",
    )
    by_feedback = VerificationResult(
        tier=2,
        passed=False,
        outcome="fail",
        reason_code="llm_semantic_failed",
        feedback="Contains TODO marker in final output.",
    )

    assert is_placeholder_claim_failure(gates, by_reason) is True
    assert is_placeholder_claim_failure(gates, by_feedback) is True


def test_scan_placeholder_markers_workspace_none_returns_unavailable_payload() -> None:
    gates = SimpleNamespace()
    payload = scan_placeholder_markers(
        gates,
        workspace=None,
        candidate_data={},
    )

    assert payload["scan_mode"] == "targeted_only"
    assert payload["scanned_file_count"] == 0
    assert payload["coverage_sufficient"] is False
    assert payload["coverage_insufficient_reason"] == "workspace_unavailable"


def test_normalize_candidate_helpers_preserve_suffix_defaults_and_workspace_paths(tmp_path) -> None:
    gates = SimpleNamespace(
        _DEFAULT_CONTRADICTION_SCAN_ALLOWED_SUFFIXES=(".md", ".txt"),
    )

    suffixes = normalized_scan_suffixes(gates, "md, txt")
    assert suffixes == (".md", ".txt")
    assert normalized_scan_suffixes(gates, None) == (".md", ".txt")

    report = tmp_path / "report.md"
    report.write_text("ok", encoding="utf-8")
    assert normalize_candidate_path(workspace=tmp_path, raw_path="report.md") == "report.md"
    assert normalize_candidate_path(workspace=tmp_path, raw_path=report) == "report.md"
    assert normalize_candidate_path(workspace=tmp_path, raw_path="../outside.md") is None


def test_expected_deliverables_for_subtask_prefers_phase_id_match() -> None:
    process = SimpleNamespace(
        phases=[],
        get_deliverables=lambda: {
            "phase-a": ["report.md", "summary.txt"],
            "phase-b": ["appendix.md"],
        },
    )
    gates = SimpleNamespace(_process=process)
    subtask = SimpleNamespace(
        id="phase-a-step-1",
        phase_id="phase-a",
        description="Generate report",
        acceptance_criteria="Include summary",
    )

    deliverables = expected_deliverables_for_subtask(gates, subtask)

    assert deliverables == ["report.md", "summary.txt"]


@pytest.mark.asyncio
async def test_apply_placeholder_contradiction_guard_short_circuits_when_disabled() -> None:
    gates = SimpleNamespace(_config=SimpleNamespace(contradiction_guard_enabled=False))
    subtask = SimpleNamespace()
    original = VerificationResult(
        tier=2,
        passed=False,
        outcome="fail",
        reason_code="incomplete_deliverable_placeholder",
    )

    result = await apply_placeholder_contradiction_guard(
        gates,
        subtask=subtask,  # type: ignore[arg-type]
        result=original,
        workspace=None,
        tool_calls=[],
    )

    assert result is original
