"""Retry manager with escalation ladder and error-aware context.

Implements structured retry and escalation for failed subtasks:
1. Same model + verification feedback + error-specific hints
2. Escalate to next tier model
3. Escalate to highest tier
4. Flag for human review
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from loom.recovery.errors import ErrorCategory, categorize_error


class RetryStrategy(StrEnum):
    GENERIC = "generic"
    RATE_LIMIT = "rate_limit"
    VERIFIER_PARSE = "verifier_parse"
    EVIDENCE_GAP = "evidence_gap"
    UNCONFIRMED_DATA = "unconfirmed_data"


@dataclass
class AttemptRecord:
    """Record of a single execution attempt."""

    attempt: int
    tier: int
    feedback: str | None = None
    error: str | None = None
    successful_tool_calls: list = field(default_factory=list)
    evidence_records: list[dict] = field(default_factory=list)
    retry_strategy: RetryStrategy = RetryStrategy.GENERIC
    missing_targets: list[str] = field(default_factory=list)
    error_category: ErrorCategory | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        # Auto-categorize if error is present but category is not
        if self.error and self.error_category is None:
            self.error_category = categorize_error(self.error).category


class RetryManager:
    """Manages retry logic with tier escalation and error-aware context.

    Escalation ladder:
    - Attempt 1: original tier, no retry context
    - Attempt 2: same tier + verification feedback + error-specific hints
    - Attempt 3: next tier + fresh context + all feedback
    - Attempt 4: highest tier + all feedback
    - Beyond: flag for human review
    """

    def __init__(self, max_retries: int = 3, max_tier: int = 3):
        self._max_retries = max_retries
        self._max_tier = max_tier

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def get_escalation_tier(self, attempt: int, original_tier: int) -> int:
        """Determine model tier for this attempt.

        Args:
            attempt: 0-indexed attempt number
            original_tier: the tier from the plan
        """
        if attempt <= 1:
            # Attempts 0 and 1: same tier
            return original_tier
        elif attempt == 2:
            # Attempt 2: escalate to next tier
            return min(original_tier + 1, self._max_tier)
        else:
            # Attempt 3+: highest tier
            return self._max_tier

    def build_retry_context(self, attempts: list[AttemptRecord]) -> str:
        """Build context from prior failed attempts.

        Includes what went wrong, verification feedback, and
        error-specific recovery hints when available.
        """
        if not attempts:
            return ""

        lines = ["PREVIOUS ATTEMPTS (all failed):"]
        for a in attempts:
            lines.append(f"\nAttempt {a.attempt} (model tier {a.tier}):")
            if a.feedback:
                lines.append(f"  Verification feedback: {a.feedback}")
            if a.error:
                lines.append(f"  Error: {a.error}")
            if a.retry_strategy and a.retry_strategy != RetryStrategy.GENERIC:
                lines.append(f"  Retry strategy: {a.retry_strategy.value}")
            if a.missing_targets:
                lines.append(
                    "  Missing targets: " + ", ".join(a.missing_targets)
                )
            if a.error_category and a.error_category != ErrorCategory.UNKNOWN:
                categorized = categorize_error(a.error or "")
                lines.append(f"  Error type: {a.error_category.value}")
                lines.append(f"  Recovery hint: {categorized.recovery_hint}")

        strategy = attempts[-1].retry_strategy if attempts else RetryStrategy.GENERIC
        if strategy == RetryStrategy.RATE_LIMIT:
            lines.append(
                "\nTARGETED RETRY PLAN:\n"
                "- Prior attempts hit rate limits/transient upstream errors.\n"
                "- Reuse existing evidence and outputs; do not redo solved work.\n"
                "- Fill only missing coverage gaps with minimal additional fetches.\n"
                "- Prefer alternate sources and lighter requests."
            )
        elif strategy == RetryStrategy.EVIDENCE_GAP:
            missing_targets = attempts[-1].missing_targets
            if missing_targets:
                target_list = ", ".join(missing_targets)
            else:
                target_list = "missing targets from verifier feedback"
            lines.append(
                "\nTARGETED RETRY PLAN:\n"
                "- Focus only on missing evidence coverage.\n"
                f"- Gather evidence for: {target_list}.\n"
                "- Keep existing validated evidence; avoid broad reruns.\n"
                "- Update only rows/claims that currently lack evidence."
            )
        elif strategy == RetryStrategy.UNCONFIRMED_DATA:
            lines.append(
                "\nTARGETED RETRY PLAN:\n"
                "- Resolve verification findings using process remediation guidance.\n"
                "- Preserve already validated findings.\n"
                "- Confirm unsupported claims with evidence or relabel them as "
                "unverified according to process policy.\n"
                "- Avoid broad reruns when only targeted remediation is needed."
            )

        lines.append(
            "\nFix the issues identified above. "
            "Take a different approach if needed."
        )
        return "\n".join(lines)

    @staticmethod
    def classify_failure(
        *,
        verification_feedback: str | None,
        execution_error: str | None,
        verification: object | None = None,
    ) -> tuple[RetryStrategy, list[str]]:
        """Classify failure type and extract optional targeting details."""
        feedback = str(verification_feedback or "")
        error = str(execution_error or "")
        haystack = " ".join([feedback, error]).lower()
        reason_code = RetryManager._extract_reason_code(verification)
        remediation_mode = RetryManager._extract_remediation_mode(verification)
        severity_class = RetryManager._extract_severity_class(verification)
        missing_targets = RetryManager._extract_missing_targets_from_verification(
            verification,
        )

        # Prefer structured verification contract when available.
        if reason_code in {"parse_inconclusive", "infra_verifier_error"}:
            return RetryStrategy.VERIFIER_PARSE, []
        if reason_code in {
            "evidence_gap",
            "missing_evidence",
            "insufficient_evidence_targets",
        }:
            return RetryStrategy.EVIDENCE_GAP, missing_targets
        if reason_code in {
            "policy_remediation_required",
            "unconfirmed_claims",
            "insufficient_evidence",
            "pending_remediation",
        }:
            return RetryStrategy.UNCONFIRMED_DATA, []
        if "unconfirmed" in reason_code or "remediation" in reason_code:
            return RetryStrategy.UNCONFIRMED_DATA, []
        if remediation_mode in {
            "confirm_or_prune",
            "queue_follow_up",
            "targeted_remediation",
            "remediate_and_retry",
        }:
            return RetryStrategy.UNCONFIRMED_DATA, []
        if severity_class == "inconclusive":
            return RetryStrategy.VERIFIER_PARSE, []
        if severity_class == "infra":
            if RetryManager._is_rate_limit_haystack(haystack):
                return RetryStrategy.RATE_LIMIT, []
            return RetryStrategy.VERIFIER_PARSE, []
        if severity_class == "hard_invariant":
            return RetryStrategy.GENERIC, []

        # Compatibility fallback path for legacy/non-structured failures.
        if RetryManager._is_rate_limit_haystack(haystack):
            return RetryStrategy.RATE_LIMIT, []

        if "could not parse verifier output" in haystack:
            return RetryStrategy.VERIFIER_PARSE, []
        if any(marker in haystack for marker in (
            "parse_inconclusive",
            "infra_verifier_error",
            "verifier raised an exception",
            "verification inconclusive:",
        )):
            return RetryStrategy.VERIFIER_PARSE, []

        if any(marker in haystack for marker in (
            "unconfirmed",
            "insufficiently confirmed",
            "remediation required",
            "queue follow-up",
            "queue_follow_up",
            "confirm_or_prune",
            "partial_verified",
        )):
            return RetryStrategy.UNCONFIRMED_DATA, []

        if any(marker in haystack for marker in (
            "no successful tool-call evidence found for target",
            "missing evidence for",
            "missing evidence coverage",
            "insufficient evidence for",
        )):
            return RetryStrategy.EVIDENCE_GAP, RetryManager._extract_missing_targets(
                feedback,
            )

        return RetryStrategy.GENERIC, []

    @staticmethod
    def _extract_reason_code(verification: object | None) -> str:
        if verification is None:
            return ""
        if isinstance(verification, dict):
            raw = verification.get("reason_code", "")
        else:
            raw = getattr(verification, "reason_code", "")
        return str(raw or "").strip().lower()

    @staticmethod
    def _extract_remediation_mode(verification: object | None) -> str:
        if verification is None:
            return ""
        metadata = {}
        if isinstance(verification, dict):
            candidate = verification.get("metadata", {})
            if isinstance(candidate, dict):
                metadata = candidate
        else:
            candidate = getattr(verification, "metadata", {})
            if isinstance(candidate, dict):
                metadata = candidate
        raw = metadata.get("remediation_mode", "")
        return str(raw or "").strip().lower()

    @staticmethod
    def _extract_missing_targets_from_verification(
        verification: object | None,
    ) -> list[str]:
        if verification is None:
            return []
        metadata = {}
        if isinstance(verification, dict):
            candidate = verification.get("metadata", {})
            if isinstance(candidate, dict):
                metadata = candidate
        else:
            candidate = getattr(verification, "metadata", {})
            if isinstance(candidate, dict):
                metadata = candidate

        targets = metadata.get("missing_targets", [])
        if isinstance(targets, str):
            text = targets.strip()
            return [text] if text else []
        if isinstance(targets, list):
            normalized = []
            for item in targets:
                text = str(item or "").strip()
                if text and text not in normalized:
                    normalized.append(text)
            return normalized
        return []

    @staticmethod
    def _extract_severity_class(verification: object | None) -> str:
        if verification is None:
            return ""
        if isinstance(verification, dict):
            raw = verification.get("severity_class", "")
        else:
            raw = getattr(verification, "severity_class", "")
        return str(raw or "").strip().lower()

    @staticmethod
    def _is_rate_limit_haystack(haystack: str) -> bool:
        return any(marker in haystack for marker in (
            "http 429",
            "rate limit",
            "rate-limited",
            "too many requests",
        ))

    @staticmethod
    def _extract_missing_targets(feedback: str) -> list[str]:
        targets: list[str] = []
        for match in re.finditer(
            r"(?:target|entity|item)\s+'([^']+)'",
            str(feedback or ""),
            flags=re.IGNORECASE,
        ):
            target = str(match.group(1)).strip()
            if target and target not in targets:
                targets.append(target)
        if targets:
            return targets

        for match in re.finditer(
            r"for\s+'([^']+)'",
            str(feedback or ""),
            flags=re.IGNORECASE,
        ):
            target = str(match.group(1)).strip()
            if target and target not in targets:
                targets.append(target)
        return targets

    def should_flag_for_human(self, attempt: int) -> bool:
        """Check if all retries are exhausted and human review is needed."""
        return attempt >= self._max_retries

    def is_transient_error(self, attempts: list[AttemptRecord]) -> bool:
        """Check if the most recent error is likely transient (worth retrying as-is)."""
        if not attempts:
            return False
        last = attempts[-1]
        return last.error_category in (
            ErrorCategory.MODEL_ERROR,
            ErrorCategory.TIMEOUT,
        )
