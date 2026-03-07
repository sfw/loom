"""Verification result and check data structures."""

from __future__ import annotations

from dataclasses import dataclass, field

_VALID_OUTCOMES = {
    "pass",
    "pass_with_warnings",
    "partial_verified",
    "fail",
}

_VALID_SEVERITY_CLASSES = {
    "hard_invariant",
    "semantic",
    "inconclusive",
    "infra",
}

_REASON_CODE_SEVERITY: dict[str, str] = {
    "hard_invariant_failed": "hard_invariant",
    "parse_inconclusive": "inconclusive",
    "infra_verifier_error": "infra",
}


@dataclass
class Check:
    """A single verification check result."""

    name: str
    passed: bool
    detail: str | None = None


@dataclass
class VerificationResult:
    """Result of running verification gates."""

    tier: int
    passed: bool
    confidence: float = 1.0
    checks: list[Check] = field(default_factory=list)
    feedback: str | None = None
    outcome: str = "pass"
    reason_code: str = ""
    severity_class: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = str(self.outcome or "pass").strip().lower()
        self.outcome = normalized if normalized in _VALID_OUTCOMES else "pass"
        if self.outcome == "fail":
            self.passed = False
        severity = str(self.severity_class or "").strip().lower()
        if severity not in _VALID_SEVERITY_CLASSES:
            severity = self._infer_severity_class(
                reason_code=self.reason_code,
                outcome=self.outcome,
                tier=self.tier,
                passed=self.passed,
            )
        self.severity_class = severity

    @staticmethod
    def _infer_severity_class(
        *,
        reason_code: str,
        outcome: str,
        tier: int,
        passed: bool,
    ) -> str:
        normalized_reason = str(reason_code or "").strip().lower()
        if normalized_reason in _REASON_CODE_SEVERITY:
            return _REASON_CODE_SEVERITY[normalized_reason]
        if outcome == "fail" and tier <= 1:
            return "hard_invariant"
        if not passed and outcome == "fail":
            return "semantic"
        return "semantic"
