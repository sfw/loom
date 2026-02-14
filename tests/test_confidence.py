"""Tests for the confidence scoring system."""

from __future__ import annotations

from dataclasses import dataclass, field

from loom.engine.verification import Check, VerificationResult
from loom.recovery.confidence import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    MEDIUM_THRESHOLD,
    ConfidenceScorer,
)
from loom.state.task_state import Subtask
from loom.tools.registry import ToolResult

# --- Helpers ---


@dataclass
class MockToolCallRecord:
    tool: str
    args: dict
    result: ToolResult


@dataclass
class MockSubtaskResult:
    status: str = "success"
    summary: str = "Done"
    tool_calls: list = field(default_factory=list)


def _make_subtask(retry_count: int = 0, max_retries: int = 3) -> Subtask:
    return Subtask(
        id="s1",
        description="Test",
        retry_count=retry_count,
        max_retries=max_retries,
    )


# --- Tests ---


class TestConfidenceScorer:
    def test_perfect_score_returns_high_band(self):
        scorer = ConfidenceScorer()
        verification = VerificationResult(
            tier=1,
            passed=True,
            checks=[Check(name="c1", passed=True)],
        )
        result = MockSubtaskResult()
        subtask = _make_subtask()

        score = scorer.score(subtask, result, verification)

        assert score.score >= HIGH_THRESHOLD
        assert score.band == "high"

    def test_tier2_verification_contributes(self):
        scorer = ConfidenceScorer()
        verification = VerificationResult(
            tier=2,
            passed=True,
            confidence=0.9,
            checks=[Check(name="c1", passed=True)],
        )
        result = MockSubtaskResult()
        subtask = _make_subtask()

        score = scorer.score(subtask, result, verification)

        assert score.score >= HIGH_THRESHOLD
        assert "tier2_confidence" in score.components
        assert score.components["tier2_confidence"] == 0.9

    def test_retries_reduce_score(self):
        scorer = ConfidenceScorer()
        verification = VerificationResult(tier=1, passed=True, checks=[])
        result = MockSubtaskResult()

        # No retries
        score_no_retry = scorer.score(
            _make_subtask(retry_count=0), result, verification,
        )
        # Max retries
        score_max_retry = scorer.score(
            _make_subtask(retry_count=3, max_retries=3), result, verification,
        )

        assert score_no_retry.score > score_max_retry.score

    def test_destructive_operation_reduces_score(self):
        scorer = ConfidenceScorer()
        verification = VerificationResult(tier=1, passed=True, checks=[])

        safe_result = MockSubtaskResult()
        destructive_result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="shell_execute",
                args={"command": "rm -rf /tmp/test"},
                result=ToolResult.ok("ok"),
            )],
        )
        subtask = _make_subtask()

        score_safe = scorer.score(subtask, safe_result, verification)
        score_destructive = scorer.score(subtask, destructive_result, verification)

        assert score_safe.score > score_destructive.score

    def test_failed_tool_calls_reduce_score(self):
        scorer = ConfidenceScorer()
        verification = VerificationResult(tier=1, passed=True, checks=[])
        subtask = _make_subtask()

        success_result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="write_file",
                args={"path": "f.py"},
                result=ToolResult.ok("ok"),
            )],
        )
        failed_result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="write_file",
                args={"path": "f.py"},
                result=ToolResult.fail("Error"),
            )],
        )

        score_success = scorer.score(subtask, success_result, verification)
        score_failed = scorer.score(subtask, failed_result, verification)

        assert score_success.score > score_failed.score

    def test_score_always_between_0_and_1(self):
        scorer = ConfidenceScorer()
        verification = VerificationResult(
            tier=2, passed=True, confidence=0.0,
            checks=[Check(name="c1", passed=False)],
        )
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="shell_execute",
                args={"command": "rm -rf /"},
                result=ToolResult.fail("blocked"),
            )],
        )
        subtask = _make_subtask(retry_count=3, max_retries=3)

        score = scorer.score(subtask, result, verification)

        assert 0.0 <= score.score <= 1.0

    def test_zero_band_classification(self):
        scorer = ConfidenceScorer()
        assert scorer._classify_band(0.1) == "zero"
        assert scorer._classify_band(0.0) == "zero"

    def test_low_band_classification(self):
        scorer = ConfidenceScorer()
        assert scorer._classify_band(0.3) == "low"
        assert scorer._classify_band(LOW_THRESHOLD) == "low"

    def test_medium_band_classification(self):
        scorer = ConfidenceScorer()
        assert scorer._classify_band(0.6) == "medium"
        assert scorer._classify_band(MEDIUM_THRESHOLD) == "medium"

    def test_high_band_classification(self):
        scorer = ConfidenceScorer()
        assert scorer._classify_band(0.9) == "high"
        assert scorer._classify_band(HIGH_THRESHOLD) == "high"

    def test_is_destructive_detects_rm(self):
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="shell_execute",
                args={"command": "rm -rf /tmp/data"},
                result=ToolResult.ok("ok"),
            )],
        )
        assert ConfidenceScorer._is_destructive(result)

    def test_is_destructive_ignores_safe_commands(self):
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="shell_execute",
                args={"command": "ls -la /tmp"},
                result=ToolResult.ok("ok"),
            )],
        )
        assert not ConfidenceScorer._is_destructive(result)

    def test_is_destructive_ignores_non_shell(self):
        result = MockSubtaskResult(
            tool_calls=[MockToolCallRecord(
                tool="write_file",
                args={"path": "test.py", "content": "x = 1"},
                result=ToolResult.ok("ok"),
            )],
        )
        assert not ConfidenceScorer._is_destructive(result)

    def test_no_tool_calls_gives_full_tool_score(self):
        scorer = ConfidenceScorer()
        verification = VerificationResult(tier=1, passed=True, checks=[])
        result = MockSubtaskResult(tool_calls=[])
        subtask = _make_subtask()

        score = scorer.score(subtask, result, verification)
        assert score.components["tool_success"] == 1.0

    def test_components_populated(self):
        scorer = ConfidenceScorer()
        verification = VerificationResult(
            tier=2, passed=True, confidence=0.85,
            checks=[Check(name="c1", passed=True)],
        )
        result = MockSubtaskResult()
        subtask = _make_subtask()

        score = scorer.score(subtask, result, verification)

        assert "tier1_checks" in score.components
        assert "tier2_confidence" in score.components
        assert "no_retries" in score.components
        assert "non_destructive" in score.components
        assert "tool_success" in score.components
