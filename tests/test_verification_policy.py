"""Focused tests for extracted verification policy helpers."""

from __future__ import annotations

from loom.engine.verification.policy import (
    aggregate_non_failing,
    classify_shadow_diff,
    fallback_from_tier1_for_inconclusive_tier2,
    legacy_failure_action,
    legacy_result_from_tiers,
    normalize_profile,
    resolve_policy_decision,
)
from loom.engine.verification.types import Check, VerificationResult


def test_classify_shadow_diff_branches() -> None:
    old_fail = VerificationResult(tier=1, passed=False, outcome="fail", reason_code="r1")
    new_pass = VerificationResult(tier=2, passed=True, outcome="pass")
    assert classify_shadow_diff(old_fail, new_pass) == "old_fail_new_pass"

    old_pass = VerificationResult(tier=1, passed=True, outcome="pass")
    new_fail = VerificationResult(tier=2, passed=False, outcome="fail", reason_code="r2")
    assert classify_shadow_diff(old_pass, new_fail) == "old_pass_new_fail"

    fail_a = VerificationResult(tier=1, passed=False, outcome="fail", reason_code="a")
    fail_b = VerificationResult(tier=2, passed=False, outcome="fail", reason_code="b")
    assert classify_shadow_diff(fail_a, fail_b) == "both_fail_reason_diff"

    same = VerificationResult(tier=2, passed=True, outcome="pass")
    assert classify_shadow_diff(same, same) == "no_diff"


def test_aggregate_non_failing_partial_verified_merges_metadata_feedback_and_checks() -> None:
    t1 = VerificationResult(
        tier=1,
        passed=True,
        checks=[Check(name="c1", passed=True)],
        feedback="Warn 1",
        outcome="pass_with_warnings",
        metadata={"k1": "v1"},
    )
    t2 = VerificationResult(
        tier=2,
        passed=True,
        checks=[Check(name="c2", passed=True)],
        feedback="Warn 2",
        outcome="partial_verified",
        reason_code="missing_evidence",
        metadata={"k2": "v2"},
    )

    merged = aggregate_non_failing([t1, t2])

    assert merged.tier == 2
    assert merged.passed is True
    assert merged.outcome == "partial_verified"
    assert merged.reason_code == "missing_evidence"
    assert len(merged.checks) == 2
    assert "Warn 1" in str(merged.feedback)
    assert "Warn 2" in str(merged.feedback)
    assert merged.metadata["tier1"] == {"k1": "v1"}
    assert merged.metadata["tier2"] == {"k2": "v2"}


def test_legacy_result_from_tiers_enforces_advisory_regex_failure() -> None:
    t1 = VerificationResult(
        tier=1,
        passed=True,
        checks=[
            Check(
                name="process_rule_no_placeholder_tokens",
                passed=True,
                detail="Regex matched (advisory)",
            ),
        ],
    )
    t2 = VerificationResult(tier=2, passed=True, outcome="pass")

    legacy = legacy_result_from_tiers([t1, t2])

    assert legacy.passed is False
    assert legacy.reason_code == "legacy_regex_failure"
    assert legacy.metadata["legacy_mode"] is True


def test_fallback_from_tier1_for_inconclusive_tier2_preserves_warning_metadata() -> None:
    t1 = VerificationResult(
        tier=1,
        passed=True,
        confidence=0.6,
        checks=[Check(name="tool_success", passed=True)],
        outcome="pass",
    )
    t2 = VerificationResult(
        tier=2,
        passed=False,
        confidence=0.2,
        feedback="Could not parse verifier output",
        outcome="fail",
        reason_code="parse_inconclusive",
        severity_class="inconclusive",
        metadata={"raw": "malformed-json"},
    )

    fallback = fallback_from_tier1_for_inconclusive_tier2(
        tier1_result=t1,
        tier2_result=t2,
    )

    assert fallback is not None
    assert fallback.passed is True
    assert fallback.outcome == "pass_with_warnings"
    assert fallback.reason_code == "infra_verifier_error"
    assert fallback.metadata["fallback"] == "tier1_due_to_tier2_parse_inconclusive"


def test_decision_matrix_blocks_hard_invariant_for_research() -> None:
    decision = resolve_policy_decision(
        severity_class="hard_invariant",
        reason_code="hard_invariant_failed",
        profile="research",
        mode="enforce",
    )
    assert decision.action == "block"
    assert decision.reason == "hard_invariant_failed"


def test_decision_matrix_warns_non_research_inconclusive_coverage() -> None:
    decision = resolve_policy_decision(
        severity_class="semantic",
        reason_code="coverage_below_threshold",
        profile="coding",
        mode="enforce",
    )
    assert decision.action == "pass_with_warnings"


def test_decision_matrix_warns_dev_verifier_timeout_for_coding() -> None:
    decision = resolve_policy_decision(
        severity_class="infra",
        reason_code="dev_verifier_timeout",
        profile="coding",
        mode="enforce",
    )
    assert decision.action == "pass_with_warnings"


def test_decision_matrix_retries_targeted_for_dev_test_failures() -> None:
    decision = resolve_policy_decision(
        severity_class="semantic",
        reason_code="dev_test_failed",
        profile="coding",
        mode="enforce",
    )
    assert decision.action == "retry_targeted"


def test_decision_matrix_shadow_mode_keeps_legacy_action() -> None:
    decision = resolve_policy_decision(
        severity_class="semantic",
        reason_code="coverage_below_threshold",
        profile="coding",
        mode="shadow",
    )
    assert decision.action == legacy_failure_action(
        severity_class="semantic",
        reason_code="coverage_below_threshold",
    )
    assert decision.shadow_diff == "legacy_retry_semantic_new_pass_with_warnings"


def test_decision_matrix_off_mode_uses_legacy_and_normalizes_profile() -> None:
    decision = resolve_policy_decision(
        severity_class="semantic",
        reason_code="claim_inconclusive",
        profile=normalize_profile("unknown"),
        mode="off",
    )
    assert decision.profile == "hybrid"
    assert decision.action == "retry_semantic"
