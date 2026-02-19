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
    missing_markets: list[str] = field(default_factory=list)
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
            if a.missing_markets:
                lines.append(
                    "  Missing markets: " + ", ".join(a.missing_markets)
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
            markets = attempts[-1].missing_markets
            if markets:
                market_list = ", ".join(markets)
            else:
                market_list = "missing markets from verifier feedback"
            lines.append(
                "\nTARGETED RETRY PLAN:\n"
                "- Focus only on missing evidence coverage.\n"
                f"- Gather evidence for: {market_list}.\n"
                "- Keep existing validated evidence; avoid broad reruns.\n"
                "- Update only rows/claims that currently lack evidence."
            )
        elif strategy == RetryStrategy.UNCONFIRMED_DATA:
            lines.append(
                "\nTARGETED RETRY PLAN:\n"
                "- Run confirm-or-prune on unconfirmed claims.\n"
                "- Preserve already confirmed findings.\n"
                "- Remove or relabel any claim that cannot be confirmed.\n"
                "- Keep recommendations fully confirmed."
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
    ) -> tuple[RetryStrategy, list[str]]:
        """Classify failure type and extract optional targeting details."""
        feedback = str(verification_feedback or "")
        error = str(execution_error or "")
        haystack = " ".join([feedback, error]).lower()

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
            "recommendation_unconfirmed",
            "unconfirmed supporting",
            "unconfirmed claims",
            "confirm-or-prune",
            "partial_verified",
        )):
            return RetryStrategy.UNCONFIRMED_DATA, []

        if any(marker in haystack for marker in (
            "http 429",
            "rate limit",
            "rate-limited",
            "too many requests",
        )):
            return RetryStrategy.RATE_LIMIT, []

        if "no successful tool-call evidence found for market" in haystack:
            markets = RetryManager._extract_missing_markets(feedback)
            return RetryStrategy.EVIDENCE_GAP, markets

        return RetryStrategy.GENERIC, []

    @staticmethod
    def _extract_missing_markets(feedback: str) -> list[str]:
        markets: list[str] = []
        for match in re.finditer(
            r"market\s+'([^']+)'",
            str(feedback or ""),
            flags=re.IGNORECASE,
        ):
            market = str(match.group(1)).strip()
            if market and market not in markets:
                markets.append(market)
        return markets

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
