"""Verification policy helpers for result aggregation and shadow diffing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import Check, VerificationResult

VerificationProfile = Literal["research", "coding", "data_ops", "hybrid"]
VerificationAction = Literal[
    "block",
    "retry_targeted",
    "retry_semantic",
    "pass_with_warnings",
]

_VALID_RESILIENCE_POLICY_MODES = {"enforce", "shadow", "off"}
_HARD_BLOCK_REASON_CODES = frozenset({
    "artifact_seal_invalid",
    "hard_invariant_failed",
    "manifest_input_policy_violation",
    "output_publish_commit_failed",
    "required_verifier_missing",
    "required_verifier_empty",
    "temporal_conflict",
})
_CONTRADICTION_REASON_CODES = frozenset({
    "claim_contradicted",
    "contradiction_detected",
})
_TARGETED_RETRY_REASON_CODES = frozenset({
    "csv_schema_mismatch",
    "forbidden_output_path",
    "output_path_policy_violation",
    "missing_evidence",
    "insufficient_evidence_targets",
    "missing_precedent_transactions",
})
_INCONCLUSIVE_REASON_CODES = frozenset({
    "claim_inconclusive",
    "parse_inconclusive",
    "infra_verifier_error",
    "semantic_inconclusive",
    "verifier_parse_inconclusive",
    "verifier_unavailable",
})


@dataclass(frozen=True)
class VerificationPolicyDecision:
    """Resolved decision for verification handling with audit metadata."""

    action: VerificationAction
    reason: str
    profile: VerificationProfile
    mode: str
    legacy_action: VerificationAction
    shadow_diff: str


def normalize_profile(value: object) -> VerificationProfile:
    """Normalize profile values into one of the supported profile lanes."""
    text = str(value or "").strip().lower()
    if text in {"research", "coding", "data_ops", "hybrid"}:
        return text  # type: ignore[return-value]
    return "hybrid"


def normalize_resilience_policy_mode(value: object) -> str:
    """Normalize rollout mode for resilience decision routing."""
    mode = str(value or "").strip().lower()
    if mode in _VALID_RESILIENCE_POLICY_MODES:
        return mode
    return "enforce"


def legacy_failure_action(
    *,
    severity_class: str,
    reason_code: str,
) -> VerificationAction:
    """Approximate pre-refactor action routing for shadow comparisons."""
    severity = str(severity_class or "").strip().lower()
    reason = str(reason_code or "").strip().lower()
    if severity == "hard_invariant" or reason in _HARD_BLOCK_REASON_CODES:
        return "block"
    if reason in _TARGETED_RETRY_REASON_CODES:
        return "retry_targeted"
    return "retry_semantic"


def resolve_failure_action(
    *,
    severity_class: str,
    reason_code: str,
    profile: VerificationProfile,
    contradiction_detected: bool = False,
    profile_confidence: float = 1.0,
) -> VerificationAction:
    """Resolve policy action for a failed verification outcome."""
    severity = str(severity_class or "").strip().lower()
    reason = str(reason_code or "").strip().lower()
    lane = normalize_profile(profile)
    confidence = max(0.0, min(1.0, float(profile_confidence or 0.0)))
    effective_lane: VerificationProfile = lane if confidence >= 0.5 else "hybrid"

    if contradiction_detected or reason in _CONTRADICTION_REASON_CODES:
        return "block"
    if reason in _HARD_BLOCK_REASON_CODES:
        return "block"
    if severity == "hard_invariant":
        if reason in _TARGETED_RETRY_REASON_CODES:
            return "retry_targeted"
        return "block"

    if reason in _TARGETED_RETRY_REASON_CODES:
        return "retry_targeted"
    if reason in _INCONCLUSIVE_REASON_CODES:
        if effective_lane == "research":
            return "retry_semantic"
        return "pass_with_warnings"
    if reason == "coverage_below_threshold":
        if effective_lane == "research":
            return "retry_semantic"
        return "pass_with_warnings"
    if reason in {"claim_insufficient_evidence", "evidence_gap", "insufficient_evidence"}:
        if effective_lane == "research":
            return "retry_semantic"
        return "retry_targeted"

    if severity in {"infra", "inconclusive"}:
        return "retry_semantic"
    if severity == "semantic":
        return "retry_semantic"
    return "retry_semantic"


def resolve_policy_decision(
    *,
    severity_class: str,
    reason_code: str,
    profile: VerificationProfile,
    mode: str,
    contradiction_detected: bool = False,
    profile_confidence: float = 1.0,
) -> VerificationPolicyDecision:
    """Resolve enforced action with rollout mode and shadow diff metadata."""
    normalized_mode = normalize_resilience_policy_mode(mode)
    normalized_profile = normalize_profile(profile)
    new_action = resolve_failure_action(
        severity_class=severity_class,
        reason_code=reason_code,
        profile=normalized_profile,
        contradiction_detected=contradiction_detected,
        profile_confidence=profile_confidence,
    )
    legacy_action = legacy_failure_action(
        severity_class=severity_class,
        reason_code=reason_code,
    )
    if normalized_mode == "off":
        selected = legacy_action
    elif normalized_mode == "shadow":
        selected = legacy_action
    else:
        selected = new_action

    if legacy_action == new_action:
        shadow_diff = "no_diff"
    else:
        shadow_diff = f"legacy_{legacy_action}_new_{new_action}"

    reason = "policy_matrix"
    if contradiction_detected:
        reason = "contradiction_hard_block"
    elif str(reason_code or "").strip():
        reason = str(reason_code or "").strip().lower()
    elif str(severity_class or "").strip():
        reason = f"severity_{str(severity_class).strip().lower()}"

    return VerificationPolicyDecision(
        action=selected,
        reason=reason,
        profile=normalized_profile,
        mode=normalized_mode,
        legacy_action=legacy_action,
        shadow_diff=shadow_diff,
    )


def classify_shadow_diff(
    legacy_result: VerificationResult,
    result: VerificationResult,
) -> str:
    """Classify delta between legacy and policy verification outcomes."""
    if not legacy_result.passed and result.passed:
        return "old_fail_new_pass"
    if legacy_result.passed and not result.passed:
        return "old_pass_new_fail"
    if (
        not legacy_result.passed
        and not result.passed
        and legacy_result.reason_code != result.reason_code
    ):
        return "both_fail_reason_diff"
    return "no_diff"


def aggregate_non_failing(results: list[VerificationResult]) -> VerificationResult:
    """Merge tier outcomes when policy engine allows non-failing aggregation."""
    if not results:
        return VerificationResult(
            tier=0,
            passed=True,
            confidence=0.5,
            outcome="pass",
        )

    merged_checks: list[Check] = []
    feedbacks: list[str] = []
    for item in results:
        merged_checks.extend(item.checks or [])
        if item.feedback:
            feedbacks.append(item.feedback)

    outcome = "pass"
    reason_code = ""
    if any(item.outcome == "partial_verified" for item in results):
        outcome = "partial_verified"
        reason_code = next(
            (
                item.reason_code
                for item in results
                if item.outcome == "partial_verified" and item.reason_code
            ),
            "",
        )
    elif any(item.outcome == "pass_with_warnings" for item in results):
        outcome = "pass_with_warnings"

    highest = max(results, key=lambda item: item.tier)
    merged_feedback = None
    if outcome in {"pass_with_warnings", "partial_verified"} and feedbacks:
        merged_feedback = "\n".join(dict.fromkeys(feedbacks))

    merged_metadata: dict[str, object] = {}
    for item in results:
        if item.metadata:
            merged_metadata[f"tier{item.tier}"] = item.metadata

    return VerificationResult(
        tier=highest.tier,
        passed=True,
        confidence=highest.confidence,
        checks=merged_checks,
        feedback=merged_feedback,
        outcome=outcome,
        reason_code=reason_code,
        metadata=merged_metadata,
    )


def legacy_result_from_tiers(results: list[VerificationResult]) -> VerificationResult:
    """Compute legacy pass/fail behavior from tiered verification results."""
    if not results:
        return VerificationResult(
            tier=0,
            passed=True,
            confidence=0.5,
            outcome="pass",
        )

    t1 = next((item for item in results if item.tier == 1), None)
    if t1 is not None:
        legacy_regex_hits = [
            check
            for check in (t1.checks or [])
            if check.passed
            and check.name.startswith("process_rule_")
            and "(advisory)" in str(check.detail or "").lower()
        ]
        if legacy_regex_hits:
            return VerificationResult(
                tier=1,
                passed=False,
                confidence=t1.confidence,
                checks=list(t1.checks or []),
                feedback=(
                    "Legacy verification would fail on advisory regex rule match."
                ),
                outcome="fail",
                reason_code="legacy_regex_failure",
                metadata={"legacy_mode": True},
            )

    for item in results:
        if not item.passed:
            return VerificationResult(
                tier=item.tier,
                passed=False,
                confidence=item.confidence,
                checks=list(item.checks or []),
                feedback=item.feedback,
                outcome="fail",
                reason_code=item.reason_code,
                metadata={"legacy_mode": True},
            )

    highest = max(results, key=lambda item: item.tier)
    return VerificationResult(
        tier=highest.tier,
        passed=True,
        confidence=highest.confidence,
        checks=list(highest.checks or []),
        outcome="pass",
        metadata={"legacy_mode": True},
    )


def fallback_from_tier1_for_inconclusive_tier2(
    *,
    tier1_result: VerificationResult | None,
    tier2_result: VerificationResult,
) -> VerificationResult | None:
    """Fallback to tier-1 pass when tier-2 is inconclusive and non-contradictory."""
    if tier1_result is None or not tier1_result.passed:
        return None
    if (
        isinstance(tier2_result.metadata, dict)
        and bool(tier2_result.metadata.get("contradiction_detected", False))
    ):
        return None
    reason = str(tier2_result.reason_code or "").strip().lower()
    severity = str(tier2_result.severity_class or "").strip().lower()
    if reason != "parse_inconclusive" and severity != "inconclusive":
        return None
    note = (
        "Tier-2 verifier output was inconclusive; accepting Tier-1 checks "
        "with warning."
    )
    merged_feedback = "\n".join(
        part for part in [tier2_result.feedback or "", note] if part
    )
    metadata: dict[str, object] = {
        "fallback": "tier1_due_to_tier2_parse_inconclusive",
        "tier2_reason_code": reason or str(tier2_result.reason_code or ""),
        "tier2_outcome": tier2_result.outcome,
    }
    if isinstance(tier2_result.metadata, dict) and tier2_result.metadata:
        metadata["tier2"] = dict(tier2_result.metadata)
    return VerificationResult(
        tier=2,
        passed=True,
        confidence=max(0.5, float(tier1_result.confidence)),
        checks=list(tier1_result.checks or []),
        feedback=merged_feedback or None,
        outcome="pass_with_warnings",
        reason_code="infra_verifier_error",
        severity_class="infra",
        metadata=metadata,
    )
