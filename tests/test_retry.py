"""Tests for the retry manager with escalation ladder."""

from __future__ import annotations

from loom.recovery.retry import AttemptRecord, RetryManager


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
