"""Focused tests for extracted runner policy helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from loom.engine.runner import policy as runner_policy


def test_validate_deliverable_write_policy_blocks_forbidden_path() -> None:
    error = runner_policy.validate_deliverable_write_policy(
        tool_name="write_file",
        tool_args={"path": "analysis.md"},
        workspace=None,
        expected_deliverables=["report.md"],
        forbidden_deliverables=["analysis.md"],
        allowed_output_prefixes=[],
        enforce_deliverable_paths=False,
        edit_existing_only=False,
        normalize_deliverable_paths=lambda paths, workspace=None: list(paths),
        target_paths_for_policy=lambda **kwargs: ["analysis.md"],
        looks_like_deliverable_variant=lambda **kwargs: False,
    )

    assert error is not None
    assert "reason_code=forbidden_output_path" in error
    assert "reserved for a phase finalizer" in error


def test_validate_deliverable_write_policy_blocks_prefix_violation() -> None:
    error = runner_policy.validate_deliverable_write_policy(
        tool_name="write_file",
        tool_args={"path": "reports/final.md"},
        workspace=None,
        expected_deliverables=[],
        forbidden_deliverables=[],
        allowed_output_prefixes=["intermediate"],
        enforce_deliverable_paths=False,
        edit_existing_only=False,
        normalize_deliverable_paths=lambda paths, workspace=None: list(paths),
        target_paths_for_policy=lambda **kwargs: ["reports/final.md"],
        looks_like_deliverable_variant=lambda **kwargs: False,
    )

    assert error is not None
    assert "reason_code=forbidden_output_path" in error
    assert "intermediate artifact prefix" in error


def test_validate_sealed_artifact_mutation_policy_allow_and_deny_paths() -> None:
    task = SimpleNamespace(metadata={})

    denied = runner_policy.validate_sealed_artifact_mutation_policy(
        task=task,  # type: ignore[arg-type]
        tool_name="edit_file",
        tool_args={"path": "report.md"},
        workspace=None,
        prior_successful_tool_calls=[],
        current_tool_calls=[],
        target_paths_for_policy=lambda **kwargs: ["report.md"],
        artifact_seal_registry=lambda task: {"report.md": {"sha256": "abc"}},
        seal_origin_is_verified=lambda **kwargs: True,
        latest_seal_timestamp=lambda protected_paths: "2026-03-01T12:00:00",
        has_post_seal_confirmation_evidence=lambda **kwargs: False,
    )

    assert denied is not None
    assert "Sealed artifact edit blocked" in denied

    allowed = runner_policy.validate_sealed_artifact_mutation_policy(
        task=task,  # type: ignore[arg-type]
        tool_name="edit_file",
        tool_args={"path": "report.md"},
        workspace=None,
        prior_successful_tool_calls=[],
        current_tool_calls=[],
        target_paths_for_policy=lambda **kwargs: ["report.md"],
        artifact_seal_registry=lambda task: {"report.md": {"sha256": "abc"}},
        seal_origin_is_verified=lambda **kwargs: True,
        latest_seal_timestamp=lambda protected_paths: "2026-03-01T12:00:00",
        has_post_seal_confirmation_evidence=lambda **kwargs: True,
    )

    assert allowed is None


def test_path_normalization_and_target_paths_for_policy(tmp_path: Path) -> None:
    rel = runner_policy.normalize_path_for_policy("reports/final.md", tmp_path)
    abs_path = runner_policy.normalize_path_for_policy(
        str(tmp_path / "reports" / "final.md"),
        tmp_path,
    )

    assert rel == "reports/final.md"
    assert abs_path == "reports/final.md"

    paths = runner_policy.target_paths_for_policy(
        tool_name="write_file",
        tool_args={"path": "reports/final.md", "destination": "reports/final.md"},
        workspace=tmp_path,
        is_mutating_file_tool_fn=lambda tool_name, tool_args: runner_policy.is_mutating_file_tool(
            tool_name=tool_name,
            tool_args=tool_args,
            write_mutating_tools={"write_file", "spreadsheet"},
            spreadsheet_write_operations={"update_cell"},
        ),
    )

    assert paths == ["reports/final.md"]


def test_looks_like_deliverable_variant_detection() -> None:
    assert runner_policy.looks_like_deliverable_variant(
        candidate="report-v2.md",
        canonical="report.md",
        variant_suffix_markers=("v", "copy"),
    )
    assert runner_policy.looks_like_deliverable_variant(
        candidate="report_copy.md",
        canonical="report.md",
        variant_suffix_markers=("v", "copy"),
    )
    assert not runner_policy.looks_like_deliverable_variant(
        candidate="summary.md",
        canonical="report.md",
        variant_suffix_markers=("v", "copy"),
    )
