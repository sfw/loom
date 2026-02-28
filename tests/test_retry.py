"""Tests for the retry manager with escalation ladder."""

from __future__ import annotations

import pytest

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

    def test_classify_failure_evidence_gap_extracts_targets(self):
        strategy, targets = RetryManager.classify_failure(
            verification_feedback=(
                "Verification failed: No successful tool-call evidence found for target "
                "'Arizona Water/Wastewater'. No successful tool-call evidence found for "
                "target 'New Mexico Water/Wastewater'."
            ),
            execution_error="",
        )
        assert strategy == RetryStrategy.EVIDENCE_GAP
        assert targets == ["Arizona Water/Wastewater", "New Mexico Water/Wastewater"]

    def test_build_retry_context_includes_evidence_gap_plan(self):
        mgr = RetryManager()
        attempts = [
            AttemptRecord(
                attempt=1,
                tier=2,
                feedback=(
                    "No successful tool-call evidence found for target "
                    "'Arizona Water/Wastewater'."
                ),
                retry_strategy=RetryStrategy.EVIDENCE_GAP,
                missing_targets=["Arizona Water/Wastewater"],
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

    def test_classify_failure_prefers_structured_reason_code(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="Totally unrelated text",
            execution_error="",
            verification={"reason_code": "recommendation_unconfirmed"},
        )
        assert strategy == RetryStrategy.UNCONFIRMED_DATA
        assert markets == []

    @pytest.mark.parametrize("reason_code", [
        "incomplete_deliverable_placeholder",
        "incomplete_deliverable_content",
        "unsupported_claims_and_incomplete_evidence",
        "insufficient_evidence",
        "recommendation_unconfirmed",
    ])
    def test_classify_failure_routes_semantic_reason_codes_to_unconfirmed_data(
        self,
        reason_code: str,
    ):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="verification failed",
            execution_error="",
            verification={"reason_code": reason_code},
        )
        assert strategy == RetryStrategy.UNCONFIRMED_DATA
        assert markets == []

    def test_classify_failure_keeps_hard_invariant_on_strict_path(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="hard invariant failed",
            execution_error="",
            verification={"reason_code": "hard_invariant_failed"},
        )
        assert strategy == RetryStrategy.GENERIC
        assert markets == []

    def test_classify_failure_prefers_structured_remediation_mode(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="No obvious markers",
            execution_error="",
            verification={"metadata": {"remediation_mode": "queue_follow_up"}},
        )
        assert strategy == RetryStrategy.UNCONFIRMED_DATA
        assert markets == []

    def test_classify_failure_structured_fields_override_text_markers(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="HTTP 429 rate limit while running verifier",
            execution_error="",
            verification={"reason_code": "recommendation_unconfirmed"},
        )
        assert strategy == RetryStrategy.UNCONFIRMED_DATA
        assert markets == []

    def test_classify_failure_uses_structured_severity_inconclusive(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="No obvious markers",
            execution_error="",
            verification={"severity_class": "inconclusive"},
        )
        assert strategy == RetryStrategy.VERIFIER_PARSE
        assert markets == []

    def test_classify_failure_uses_structured_severity_infra_rate_limit(self):
        strategy, markets = RetryManager.classify_failure(
            verification_feedback="HTTP 429 from upstream",
            execution_error="",
            verification={"severity_class": "infra"},
        )
        assert strategy == RetryStrategy.RATE_LIMIT
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
        assert "Resolve verification findings" in context

    def test_build_retry_context_includes_edit_in_place_file_guidance(self):
        class _Call:
            def __init__(self, files_changed):
                self.result = type("Result", (), {"files_changed": files_changed})()

        mgr = RetryManager()
        attempts = [
            AttemptRecord(
                attempt=1,
                tier=2,
                feedback="retry",
                successful_tool_calls=[_Call(["analysis.md", "evidence.csv"])],
            ),
        ]
        context = mgr.build_retry_context(attempts)

        assert "EDIT IN PLACE" in context
        assert "analysis.md" in context
        assert "evidence.csv" in context
        assert "-v2" in context

    def test_build_retry_context_includes_model_planned_remediation(self):
        mgr = RetryManager()
        attempts = [
            AttemptRecord(
                attempt=1,
                tier=2,
                feedback="verification failed",
                resolution_plan=(
                    "Diagnosis: canonical filename mismatch\n"
                    "Actions:\n"
                    "1. Update canonical file in place\n"
                    "2. Re-run verification"
                ),
            ),
        ]

        context = mgr.build_retry_context(attempts)

        assert "Model-planned remediation" in context
        assert "canonical filename mismatch" in context
        assert "Update canonical file in place" in context
