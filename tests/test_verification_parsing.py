"""Focused tests for extracted tier-2 verification parsing helpers."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.verification import parsing as verification_parsing


def test_parse_verifier_response_uses_validator_payload_when_valid() -> None:
    validator = SimpleNamespace(
        validate_json_response=lambda *args, **kwargs: SimpleNamespace(
            valid=True,
            parsed={
                "passed": True,
                "confidence": 0.9,
                "issues": [],
                "feedback": "Looks good",
                "outcome": "pass",
                "metadata": {"remediation_required": False},
            },
        ),
    )
    response = SimpleNamespace(text="ignored")

    result = verification_parsing.parse_verifier_response(
        response=response,
        validator=validator,
        expected_keys=["passed", "confidence", "issues", "feedback"],
        valid_outcomes={"pass", "pass_with_warnings", "partial_verified", "fail"},
        valid_severity_classes={"hard_invariant", "semantic", "inconclusive", "infra"},
    )

    assert result is not None
    assert result.passed is True
    assert result.outcome == "pass"
    assert result.feedback == "Looks good"


def test_parse_verifier_response_falls_back_to_text_coercion() -> None:
    validator = SimpleNamespace(
        validate_json_response=lambda *args, **kwargs: SimpleNamespace(
            valid=False,
            parsed=None,
        ),
    )
    response = SimpleNamespace(
        text=(
            "passed: false\n"
            "confidence: 45%\n"
            "reason_code: parse_inconclusive\n"
            "severity_class: inconclusive\n"
            "issues:\n"
            "- missing citation\n"
        ),
    )

    result = verification_parsing.parse_verifier_response(
        response=response,
        validator=validator,
        expected_keys=["passed", "confidence", "issues", "feedback"],
        valid_outcomes={"pass", "pass_with_warnings", "partial_verified", "fail"},
        valid_severity_classes={"hard_invariant", "semantic", "inconclusive", "infra"},
    )

    assert result is not None
    assert result.passed is False
    assert result.reason_code == "parse_inconclusive"
    assert result.severity_class == "inconclusive"
    assert result.confidence == 0.45


def test_assessment_to_result_promotes_pass_with_issues_to_warnings() -> None:
    result = verification_parsing.assessment_to_result(
        {
            "passed": True,
            "confidence": 0.8,
            "issues": ["minor caveat"],
            "feedback": "Mostly correct",
            "outcome": "pass",
            "metadata": {},
        },
        valid_outcomes={"pass", "pass_with_warnings", "partial_verified", "fail"},
        valid_severity_classes={"hard_invariant", "semantic", "inconclusive", "infra"},
    )

    assert result.passed is True
    assert result.outcome == "pass_with_warnings"
    assert result.checks[0].name == "llm_assessment"
