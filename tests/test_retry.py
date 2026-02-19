"""Tests for the retry manager with escalation ladder."""

from __future__ import annotations

from loom.recovery.retry import AttemptRecord, RetryManager, RetryStrategy


class TestRetryManager:
    def test_first_attempt_uses_original_tier(self):
        mgr = RetryManager(max_retries=3)
        tier = mgr.get_escalation_tier(attempt=0, original_tier=1)
        assert tier == 1

    def test_second_attempt_uses_same_tier(self):
        mgr = RetryManager(max_retries=3)
        tier = mgr.get_escalation_tier(attempt=1, original_tier=1)
        assert tier == 1

    def test_third_attempt_escalates_tier(self):
        mgr = RetryManager(max_retries=3)
        tier = mgr.get_escalation_tier(attempt=2, original_tier=1)
        assert tier == 2

    def test_fourth_attempt_uses_max_tier(self):
        mgr = RetryManager(max_retries=3)
        tier = mgr.get_escalation_tier(attempt=3, original_tier=1)
        assert tier == 3

    def test_escalation_respects_max_tier(self):
        mgr = RetryManager(max_retries=3, max_tier=2)
        tier = mgr.get_escalation_tier(attempt=2, original_tier=2)
        assert tier == 2

    def test_escalation_from_high_tier(self):
        mgr = RetryManager(max_retries=3)
        tier = mgr.get_escalation_tier(attempt=2, original_tier=3)
        assert tier == 3  # Already at max

    def test_build_retry_context_empty(self):
        mgr = RetryManager()
        context = mgr.build_retry_context([])
        assert context == ""

    def test_build_retry_context_with_attempts(self):
        mgr = RetryManager()
        attempts = [
            AttemptRecord(attempt=1, tier=1, feedback="File empty", error=None),
            AttemptRecord(attempt=2, tier=1, feedback="Syntax error", error="SyntaxError"),
        ]
        context = mgr.build_retry_context(attempts)

        assert "PREVIOUS ATTEMPTS" in context
        assert "Attempt 1" in context
        assert "File empty" in context
        assert "Attempt 2" in context
        assert "SyntaxError" in context
        assert "different approach" in context

    def test_build_retry_context_with_feedback_only(self):
        mgr = RetryManager()
        attempts = [
            AttemptRecord(attempt=1, tier=1, feedback="Missing import"),
        ]
        context = mgr.build_retry_context(attempts)

        assert "Missing import" in context
        assert "Error" not in context  # No error field

    def test_should_flag_for_human(self):
        mgr = RetryManager(max_retries=3)
        assert not mgr.should_flag_for_human(0)
        assert not mgr.should_flag_for_human(1)
        assert not mgr.should_flag_for_human(2)
        assert mgr.should_flag_for_human(3)
        assert mgr.should_flag_for_human(4)

    def test_max_retries_property(self):
        mgr = RetryManager(max_retries=5)
        assert mgr.max_retries == 5

    def test_classify_failure_verifier_parse(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="Verification inconclusive: could not parse verifier output.",
            execution_error="",
        )
        assert strategy == RetryStrategy.VERIFIER_PARSE
        assert markets == []

    def test_classify_failure_verifier_exception_is_verifier_parse(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback=(
                "Verification inconclusive: verifier raised an exception: timeout"
            ),
            execution_error="",
        )
        assert strategy == RetryStrategy.VERIFIER_PARSE
        assert markets == []

    def test_classify_failure_rate_limit(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="Execution encountered a critical tool failure (HTTP 429).",
            execution_error="",
        )
        assert strategy == RetryStrategy.RATE_LIMIT
        assert markets == []

    def test_classify_failure_evidence_gap_extracts_markets(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback=(
                "Verification failed: No successful tool-call evidence found for market "
                "'Arizona Water/Wastewater'. No successful tool-call evidence found for "
                "market 'New Mexico Water/Wastewater'."
            ),
            execution_error="",
        )
        assert strategy == RetryStrategy.EVIDENCE_GAP
        assert markets == ["Arizona Water/Wastewater", "New Mexico Water/Wastewater"]

    def test_build_retry_context_includes_evidence_gap_plan(self):
        mgr = RetryManager()
        attempts = [
            AttemptRecord(
                attempt=1,
                tier=2,
                feedback=(
                    "No successful tool-call evidence found for market "
                    "'Arizona Water/Wastewater'."
                ),
                retry_strategy=RetryStrategy.EVIDENCE_GAP,
                missing_markets=["Arizona Water/Wastewater"],
            )
        ]
        context = mgr.build_retry_context(attempts)

        assert "TARGETED RETRY PLAN" in context
        assert "missing evidence coverage" in context
        assert "Arizona Water/Wastewater" in context

    def test_classify_failure_unconfirmed_data(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback=(
                "Recommendations include unconfirmed claims; confirm-or-prune "
                "remediation is required."
            ),
            execution_error="",
        )
        assert strategy == RetryStrategy.UNCONFIRMED_DATA
        assert markets == []

    def test_build_retry_context_includes_unconfirmed_plan(self):
        mgr = RetryManager()
        attempts = [
            AttemptRecord(
                attempt=1,
                tier=2,
                feedback="recommendation_unconfirmed",
                retry_strategy=RetryStrategy.UNCONFIRMED_DATA,
            ),
        ]
        context = mgr.build_retry_context(attempts)

        assert "TARGETED RETRY PLAN" in context
        assert "confirm-or-prune" in context
