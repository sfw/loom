"""Retry manager with escalation ladder and error-aware context.

Implements structured retry and escalation for failed subtasks:
1. Same model + verification feedback + error-specific hints
2. Escalate to next tier model
3. Escalate to highest tier
4. Flag for human review
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loom.recovery.errors import ErrorCategory, categorize_error


@dataclass
class AttemptRecord:
    """Record of a single execution attempt."""

    attempt: int
    tier: int
    feedback: str | None = None
    error: str | None = None
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
            if a.error_category and a.error_category != ErrorCategory.UNKNOWN:
                categorized = categorize_error(a.error or "")
                lines.append(f"  Error type: {a.error_category.value}")
                lines.append(f"  Recovery hint: {categorized.recovery_hint}")

        lines.append(
            "\nFix the issues identified above. "
            "Take a different approach if needed."
        )
        return "\n".join(lines)

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
