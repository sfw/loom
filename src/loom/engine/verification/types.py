"""Verification result and check data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
_VALID_ASSERTION_TYPES = {
    "fact",
    "behavior",
    "format",
    "safety",
    "policy",
}
_VALID_ASSERTION_VERDICTS = {
    "supported",
    "partially_supported",
    "contradicted",
    "inconclusive",
    "failed_contract",
}

_REASON_CODE_SEVERITY: dict[str, str] = {
    "dev_browser_check_failed": "semantic",
    "dev_build_failed": "semantic",
    "dev_contract_failed": "semantic",
    "dev_report_contract_violation": "infra",
    "dev_test_failed": "semantic",
    "dev_verifier_capability_unavailable": "infra",
    "dev_verifier_timeout": "infra",
    "hard_invariant_failed": "hard_invariant",
    "provider_binary_not_found": "infra",
    "provider_binary_unsupported": "infra",
    "parse_inconclusive": "inconclusive",
    "infra_verifier_error": "infra",
    "tool_capability_unavailable": "infra",
    "tool_runtime_capability_unavailable": "infra",
}

def severity_class_for_reason_code(reason_code: str | None) -> str:
    """Return the normalized severity class for a structured reason code."""
    return _REASON_CODE_SEVERITY.get(
        str(reason_code or "").strip().lower(),
        "",
    )


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
        severity = severity_class_for_reason_code(reason_code)
        if severity:
            return severity
        if outcome == "fail" and tier <= 1:
            return "hard_invariant"
        if not passed and outcome == "fail":
            return "semantic"
        return "semantic"


@dataclass
class AssertionEnvelope:
    """Typed verification assertion for cross-task policy consumption."""

    assertion_id: str
    assertion_type: str
    verdict: str
    confidence: float = 0.5
    reason_code: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    remediation_hint: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        assertion_type = str(self.assertion_type or "").strip().lower()
        if assertion_type not in _VALID_ASSERTION_TYPES:
            assertion_type = "fact"
        self.assertion_type = assertion_type

        verdict = str(self.verdict or "").strip().lower()
        if verdict not in _VALID_ASSERTION_VERDICTS:
            verdict = "inconclusive"
        self.verdict = verdict

        confidence = float(self.confidence or 0.0)
        self.confidence = max(0.0, min(1.0, confidence))
        self.reason_code = str(self.reason_code or "").strip().lower()

        refs = self.evidence_refs
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            refs = []
        normalized_refs: list[str] = []
        for item in refs:
            text = str(item or "").strip()
            if text and text not in normalized_refs:
                normalized_refs.append(text)
        self.evidence_refs = normalized_refs
        self.remediation_hint = str(self.remediation_hint or "").strip()
        self.source = str(self.source or "").strip().lower()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
