"""Focused tests for extracted runner policy helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from loom.engine.runner import policy as runner_policy
from loom.tools.registry import ToolResult


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
    assert "Sealed artifact mutation blocked" in denied

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
        is_mutating_tool=True,
        is_mutating_file_tool_fn=lambda tool_name, tool_args: runner_policy.is_mutating_file_tool(
            tool_name=tool_name,
            tool_args=tool_args,
            is_mutating_tool=True,
            write_mutating_tools={"write_file", "spreadsheet"},
            spreadsheet_write_operations={"update_cell"},
        ),
    )

    assert paths == ["reports/final.md"]


def test_target_paths_for_policy_supports_output_path_and_nested_args(tmp_path: Path) -> None:
    paths = runner_policy.target_paths_for_policy(
        tool_name="run_tool",
        tool_args={
            "name": "fact_checker",
            "arguments": {
                "output_path": "reports/check.md",
            },
        },
        workspace=tmp_path,
        is_mutating_tool=True,
        is_mutating_file_tool_fn=lambda tool_name, tool_args, **kwargs: True,
    )
    assert paths == ["reports/check.md"]


@pytest.mark.parametrize(
    ("arg_key", "expected_path"),
    [
        ("output_json_path", "reports/check.json"),
        ("searchable_output_path", "reports/search-index.md"),
        ("report_path", "reports/final-report.md"),
    ],
)
def test_target_paths_for_policy_supports_output_path_variants(
    tmp_path: Path,
    arg_key: str,
    expected_path: str,
) -> None:
    paths = runner_policy.target_paths_for_policy(
        tool_name="fact_checker",
        tool_args={arg_key: expected_path},
        workspace=tmp_path,
        is_mutating_tool=True,
        is_mutating_file_tool_fn=lambda tool_name, tool_args, **kwargs: True,
    )
    assert paths == [expected_path]


def test_validate_sealed_artifact_mutation_policy_blocks_output_path_without_evidence() -> None:
    task = SimpleNamespace(metadata={})
    blocked = runner_policy.validate_sealed_artifact_mutation_policy(
        task=task,  # type: ignore[arg-type]
        tool_name="fact_checker",
        tool_args={"output_path": "reports/competitor-pricing.csv"},
        workspace=None,
        is_mutating_tool=True,
        prior_successful_tool_calls=[],
        current_tool_calls=[],
        target_paths_for_policy=lambda **kwargs: ["reports/competitor-pricing.csv"],
        artifact_seal_registry=lambda task: {"reports/competitor-pricing.csv": {"sha256": "abc"}},
        seal_origin_is_verified=lambda **kwargs: True,
        latest_seal_timestamp=lambda protected_paths: "2026-03-01T12:00:00",
        has_post_seal_confirmation_evidence=lambda **kwargs: False,
    )
    assert blocked is not None
    assert "Sealed artifact mutation blocked" in blocked


def test_validate_sealed_artifact_mutation_policy_allows_output_path_with_evidence() -> None:
    task = SimpleNamespace(metadata={})
    allowed = runner_policy.validate_sealed_artifact_mutation_policy(
        task=task,  # type: ignore[arg-type]
        tool_name="fact_checker",
        tool_args={"output_path": "reports/competitor-pricing.csv"},
        workspace=None,
        is_mutating_tool=True,
        prior_successful_tool_calls=[],
        current_tool_calls=[],
        target_paths_for_policy=lambda **kwargs: ["reports/competitor-pricing.csv"],
        artifact_seal_registry=lambda task: {"reports/competitor-pricing.csv": {"sha256": "abc"}},
        seal_origin_is_verified=lambda **kwargs: True,
        latest_seal_timestamp=lambda protected_paths: "2026-03-01T12:00:00",
        has_post_seal_confirmation_evidence=lambda **kwargs: True,
    )
    assert allowed is None


def test_reseal_tracked_artifacts_after_spreadsheet_mutation_updates_seal(tmp_path: Path) -> None:
    relpath = "competitor-pricing.csv"
    artifact = tmp_path / relpath
    artifact.write_text("name,price\nA,10\n", encoding="utf-8")
    old_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    task = SimpleNamespace(
        metadata={
            "run_id": "run-1",
            "artifact_seals": {
                relpath: {
                    "path": relpath,
                    "sha256": old_sha,
                    "verified_origin": True,
                },
            },
        },
    )

    artifact.write_text("name,price\nA,20\n", encoding="utf-8")
    updated = runner_policy.reseal_tracked_artifacts_after_mutation(
        task=task,  # type: ignore[arg-type]
        workspace=tmp_path,
        tool_name="spreadsheet",
        tool_args={"operation": "create", "path": relpath},
        tool_result=ToolResult.ok("ok", files_changed=[relpath]),
        is_mutating_tool=True,
        mutation_target_arg_keys=None,
        subtask_id="subtask-1",
        tool_call_id="call-1",
        artifact_seal_registry=lambda task: task.metadata["artifact_seals"],
        mutation_paths_for_reseal=lambda **kwargs: [relpath],
        normalize_path_for_policy=lambda path_text, workspace: path_text,
        seal_origin_is_verified=lambda **kwargs: True,
    )

    assert updated == 1
    seal = task.metadata["artifact_seals"][relpath]
    assert seal["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert seal["previous_sha256"] == old_sha


def test_unexpected_sealed_mutation_paths_detects_out_of_scope_sealed_changes(
    tmp_path: Path,
) -> None:
    sealed_relpath = "reports/competitor-pricing.csv"
    expected_relpath = "reports/summary.md"
    sealed_path = tmp_path / sealed_relpath
    expected_path = tmp_path / expected_relpath
    sealed_path.parent.mkdir(parents=True, exist_ok=True)
    sealed_path.write_text("name,price\nA,10\n", encoding="utf-8")
    expected_path.write_text("summary\n", encoding="utf-8")

    task = SimpleNamespace(
        metadata={
            "artifact_seals": {
                sealed_relpath: {
                    "path": sealed_relpath,
                    "sha256": hashlib.sha256(sealed_path.read_bytes()).hexdigest(),
                    "verified_origin": True,
                },
                expected_relpath: {
                    "path": expected_relpath,
                    "sha256": hashlib.sha256(expected_path.read_bytes()).hexdigest(),
                    "verified_origin": True,
                },
            },
        },
    )
    pre_call_hashes = runner_policy.snapshot_tracked_artifact_hashes(
        task=task,  # type: ignore[arg-type]
        workspace=tmp_path,
        artifact_seal_registry=lambda task: task.metadata["artifact_seals"],
    )
    sealed_path.write_text("name,price\nA,22\n", encoding="utf-8")

    unexpected = runner_policy.unexpected_sealed_mutation_paths(
        task=task,  # type: ignore[arg-type]
        workspace=tmp_path,
        tool_name="fact_checker",
        tool_args={"output_path": expected_relpath},
        tool_result=ToolResult.ok("ok", files_changed=[expected_relpath]),
        is_mutating_tool=True,
        mutation_target_arg_keys=None,
        pre_call_hashes=pre_call_hashes,
        artifact_seal_registry=lambda task: task.metadata["artifact_seals"],
        mutation_paths_for_reseal=lambda **kwargs: [expected_relpath],
        normalize_path_for_policy=runner_policy.normalize_path_for_policy,
    )

    assert unexpected == [sealed_relpath]


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
