"""Confidence scoring for subtask results.

Computes a 0.0-1.0 confidence score based on verification results,
retry history, tool success, and destructiveness analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

from loom.engine.verification import VerificationResult
from loom.state.task_state import Subtask


@dataclass
class ConfidenceScore:
    """Computed confidence score with component breakdown."""

    score: float
    band: str  # "high", "medium", "low", "zero"
    components: dict[str, float]


# Confidence band thresholds
HIGH_THRESHOLD = 0.8
MEDIUM_THRESHOLD = 0.5
LOW_THRESHOLD = 0.2


class ConfidenceScorer:
    """Compute confidence score (0.0-1.0) for a completed subtask."""

    def score(
        self,
        subtask: Subtask,
        result: object,
        verification: VerificationResult,
    ) -> ConfidenceScore:
        """Compute weighted confidence from multiple signals."""
        components: dict[str, float] = {}
        score = 0.0
        weights_total = 0.0

        # Verification tier 1 checks (weight: 0.3)
        if verification.tier >= 1 and verification.checks:
            passed_checks = [c for c in verification.checks if c.passed]
            t1_score = len(passed_checks) / max(len(verification.checks), 1)
            components["tier1_checks"] = t1_score
            score += t1_score * 0.3
        else:
            components["tier1_checks"] = 1.0
            score += 1.0 * 0.3
        weights_total += 0.3

        # Verification tier 2 confidence (weight: 0.3)
        if verification.tier >= 2:
            components["tier2_confidence"] = verification.confidence
            score += verification.confidence * 0.3
            weights_total += 0.3
        else:
            # Tier 2 wasn't run — use full weight from tier 1
            weights_total += 0.0  # Don't count tier 2 if not run

        # No retries needed (weight: 0.2)
        max_retries = max(subtask.max_retries, 1)
        retry_penalty = min(subtask.retry_count / max_retries, 1.0)
        retry_score = 1.0 - retry_penalty
        components["no_retries"] = retry_score
        score += retry_score * 0.2
        weights_total += 0.2

        # Not a destructive operation (weight: 0.1)
        is_destructive = self._is_destructive(result)
        destructive_score = 0.0 if is_destructive else 1.0
        components["non_destructive"] = destructive_score
        score += destructive_score * 0.1
        weights_total += 0.1

        # All tool calls succeeded (weight: 0.1)
        tool_calls = getattr(result, "tool_calls", [])
        if tool_calls:
            tool_success = sum(1 for tc in tool_calls if tc.result.success)
            tool_score = tool_success / len(tool_calls)
        else:
            tool_score = 1.0
        components["tool_success"] = tool_score
        score += tool_score * 0.1
        weights_total += 0.1

        final = score / weights_total if weights_total > 0 else 0.0
        final = max(0.0, min(1.0, final))

        return ConfidenceScore(
            score=final,
            band=self._classify_band(final),
            components=components,
        )

    @staticmethod
    def _classify_band(score: float) -> str:
        """Classify score into confidence band."""
        if score >= HIGH_THRESHOLD:
            return "high"
        elif score >= MEDIUM_THRESHOLD:
            return "medium"
        elif score >= LOW_THRESHOLD:
            return "low"
        else:
            return "zero"

    @staticmethod
    def _is_destructive(result: object) -> bool:
        """Check if the subtask performed destructive operations."""
        destructive_tools = {"shell_execute"}
        destructive_patterns = ["rm ", "drop ", "delete ", "truncate ", "rmdir "]

        tool_calls = getattr(result, "tool_calls", [])
        for tc in tool_calls:
            if tc.tool in destructive_tools:
                cmd = tc.args.get("command", "")
                if any(p in cmd.lower() for p in destructive_patterns):
                    return True
        return False
