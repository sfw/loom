"""Tests for iteration gate evaluator."""

from __future__ import annotations

import sys

import pytest

from loom.engine.iteration_gates import IterationGateEvaluator
from loom.engine.runner import SubtaskResult, SubtaskResultStatus, ToolCallRecord
from loom.engine.verification import VerificationResult
from loom.processes.schema import IterationGate, IterationPolicy
from loom.tools.registry import ToolResult


def _verification_ok() -> VerificationResult:
    return VerificationResult(
        tier=2,
        passed=True,
        outcome="pass",
        reason_code="",
        feedback="ok",
    )


def _tool_metric_result(score: float) -> SubtaskResult:
    return SubtaskResult(
        status=SubtaskResultStatus.SUCCESS,
        summary="done",
        tool_calls=[
            ToolCallRecord(
                tool="humanize_writing",
                args={"operation": "evaluate"},
                result=ToolResult.ok(
                    "ok",
                    data={"report": {"humanization_score": score}},
                ),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_tool_metric_gate_passes_when_threshold_met():
    policy = IterationPolicy(
        enabled=True,
        gates=[
            IterationGate(
                id="score",
                type="tool_metric",
                blocking=True,
                tool="humanize_writing",
                metric_path="report.humanization_score",
                operator="gte",
                value=80,
            ),
        ],
    )
    evaluator = IterationGateEvaluator(command_allowlisted_prefixes=["pytest"])
    result = await evaluator.evaluate(
        policy=policy,
        result=_tool_metric_result(82),
        verification=_verification_ok(),
        workspace=None,
    )

    assert result.all_blocking_passed is True
    assert result.results[0].status == "pass"


@pytest.mark.asyncio
async def test_tool_metric_missing_path_becomes_unevaluable():
    policy = IterationPolicy(
        enabled=True,
        gates=[
            IterationGate(
                id="score",
                type="tool_metric",
                blocking=True,
                tool="humanize_writing",
                metric_path="report.missing",
                operator="gte",
                value=80,
            ),
        ],
    )
    evaluator = IterationGateEvaluator(command_allowlisted_prefixes=["pytest"])
    result = await evaluator.evaluate(
        policy=policy,
        result=_tool_metric_result(82),
        verification=_verification_ok(),
        workspace=None,
    )

    assert result.all_blocking_passed is False
    assert result.results[0].status == "unevaluable"
    assert result.results[0].reason_code == "gate_unevaluable"


@pytest.mark.asyncio
async def test_artifact_regex_gate_fails_when_pattern_found(tmp_path):
    artifact = tmp_path / "draft.md"
    artifact.write_text("This is draft text. [TODO] replace this section.")
    policy = IterationPolicy(
        enabled=True,
        gates=[
            IterationGate(
                id="no-placeholders",
                type="artifact_regex",
                blocking=True,
                pattern=r"\[TODO\]",
                expect_match=False,
            ),
        ],
    )
    evaluator = IterationGateEvaluator(command_allowlisted_prefixes=["pytest"])
    result = await evaluator.evaluate(
        policy=policy,
        result=SubtaskResult(status=SubtaskResultStatus.SUCCESS, summary="done"),
        verification=_verification_ok(),
        workspace=tmp_path,
        expected_deliverables=["draft.md"],
    )

    assert result.all_blocking_passed is False
    assert result.results[0].status == "fail"
    assert result.results[0].reason_code == "gate_regex_expectation_failed"


@pytest.mark.asyncio
async def test_artifact_regex_summary_target_uses_result_summary():
    policy = IterationPolicy(
        enabled=True,
        gates=[
            IterationGate(
                id="no-placeholders",
                type="artifact_regex",
                blocking=True,
                target="summary",
                pattern=r"\[TODO\]",
                expect_match=False,
            ),
        ],
    )
    evaluator = IterationGateEvaluator(command_allowlisted_prefixes=["pytest"])
    result = await evaluator.evaluate(
        policy=policy,
        result=SubtaskResult(
            status=SubtaskResultStatus.SUCCESS,
            summary="Final copy. [TODO] still left here.",
        ),
        verification=_verification_ok(),
        workspace=None,
    )

    assert result.results[0].status == "fail"
    assert result.results[0].reason_code == "gate_regex_expectation_failed"


@pytest.mark.asyncio
async def test_artifact_regex_changed_files_target_scans_touched_files(tmp_path):
    artifact = tmp_path / "draft.md"
    artifact.write_text("All clean now.")
    policy = IterationPolicy(
        enabled=True,
        gates=[
            IterationGate(
                id="no-placeholders",
                type="artifact_regex",
                blocking=True,
                target="changed_files",
                pattern=r"\[TODO\]",
                expect_match=False,
            ),
        ],
    )
    evaluator = IterationGateEvaluator(command_allowlisted_prefixes=["pytest"])
    result = await evaluator.evaluate(
        policy=policy,
        result=SubtaskResult(
            status=SubtaskResultStatus.SUCCESS,
            summary="done",
            tool_calls=[
                ToolCallRecord(
                    tool="write_file",
                    args={"path": "draft.md"},
                    result=ToolResult.ok("ok", files_changed=["draft.md"]),
                ),
            ],
        ),
        verification=_verification_ok(),
        workspace=tmp_path,
    )

    assert result.results[0].status == "pass"


@pytest.mark.asyncio
async def test_command_exit_gate_disabled_by_default():
    command = [sys.executable, "-c", "import sys; sys.exit(0)"]
    policy = IterationPolicy(
        enabled=True,
        gates=[
            IterationGate(
                id="tests-pass",
                type="command_exit",
                blocking=True,
                command=command,
                operator="eq",
                value=0,
                timeout_seconds=10,
            ),
        ],
    )

    evaluator = IterationGateEvaluator(
        command_allowlisted_prefixes=[f"{sys.executable} -c"],
    )
    allowed = await evaluator.evaluate(
        policy=policy,
        result=SubtaskResult(status=SubtaskResultStatus.SUCCESS, summary="done"),
        verification=_verification_ok(),
        workspace=None,
    )
    assert allowed.results[0].status == "unevaluable"
    assert "disabled" in allowed.results[0].detail


@pytest.mark.asyncio
async def test_command_exit_gate_respects_allowlist_when_enabled():
    command = [sys.executable, "-c", "import sys; sys.exit(0)"]
    policy = IterationPolicy(
        enabled=True,
        gates=[
            IterationGate(
                id="tests-pass",
                type="command_exit",
                blocking=True,
                command=command,
                operator="eq",
                value=0,
                timeout_seconds=10,
            ),
        ],
    )

    evaluator = IterationGateEvaluator(
        command_allowlisted_prefixes=[f"{sys.executable} -c"],
        enable_command_exit=True,
    )
    allowed = await evaluator.evaluate(
        policy=policy,
        result=SubtaskResult(status=SubtaskResultStatus.SUCCESS, summary="done"),
        verification=_verification_ok(),
        workspace=None,
    )
    assert allowed.results[0].status == "pass"

    evaluator_denied = IterationGateEvaluator(
        command_allowlisted_prefixes=["pytest"],
        enable_command_exit=True,
    )
    denied = await evaluator_denied.evaluate(
        policy=policy,
        result=SubtaskResult(status=SubtaskResultStatus.SUCCESS, summary="done"),
        verification=_verification_ok(),
        workspace=None,
    )
    assert denied.results[0].status == "unevaluable"
    assert denied.results[0].reason_code == "gate_unevaluable"
