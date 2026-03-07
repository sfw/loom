"""Tier-2 verifier response parsing and coercion helpers."""

from __future__ import annotations

import re
from typing import Any

from .types import Check, VerificationResult


def to_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "pass", "passed", "success", "successful"}:
            return True
        if lowered in {"false", "no", "fail", "failed", "failure", "unsuccessful"}:
            return False
    return None


def to_confidence(value: object) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1.0 and numeric <= 100.0:
            numeric = numeric / 100.0
        return max(0.0, min(1.0, numeric))

    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text.endswith("%"):
            try:
                numeric = float(text[:-1]) / 100.0
                return max(0.0, min(1.0, numeric))
            except ValueError:
                return None
        try:
            numeric = float(text)
        except ValueError:
            return None
        if numeric > 1.0 and numeric <= 100.0:
            numeric = numeric / 100.0
        return max(0.0, min(1.0, numeric))
    return None


def to_issues(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "no issues", "n/a"}:
            return []
        pieces = re.split(r"[;\n]+", text)
        return [piece.strip(" -\t") for piece in pieces if piece.strip(" -\t")]
    if isinstance(value, list):
        issues: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                issues.append(text)
        return issues
    text = str(value).strip()
    return [text] if text else []


def to_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def to_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        items = [str(item or "").strip() for item in value]
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        items = re.split(r"[,\n;]+", text)
    else:
        return []
    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def normalize_verifier_metadata(metadata: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {
        str(key).strip(): value
        for key, value in metadata.items()
        if str(key).strip()
    }
    lookup: dict[str, str] = {key.lower(): key for key in normalized}

    def value_for(field_name: str) -> object | None:
        key = lookup.get(field_name)
        if key is None:
            return None
        return normalized.get(key)

    remediation_required_raw = value_for("remediation_required")
    if remediation_required_raw is not None:
        remediation_required = to_bool(remediation_required_raw)
        if remediation_required is None:
            remediation_required = bool(remediation_required_raw)
        normalized["remediation_required"] = remediation_required

    remediation_mode_raw = value_for("remediation_mode")
    if remediation_mode_raw is not None:
        normalized["remediation_mode"] = str(remediation_mode_raw or "").strip().lower()

    missing_targets_raw = value_for("missing_targets")
    if missing_targets_raw is not None:
        normalized["missing_targets"] = to_string_list(missing_targets_raw)

    unverified_claim_count_raw = value_for("unverified_claim_count")
    if unverified_claim_count_raw is not None:
        unverified_claim_count = to_int(unverified_claim_count_raw)
        if unverified_claim_count is not None:
            normalized["unverified_claim_count"] = max(0, unverified_claim_count)

    verified_claim_count_raw = value_for("verified_claim_count")
    if verified_claim_count_raw is not None:
        verified_claim_count = to_int(verified_claim_count_raw)
        if verified_claim_count is not None:
            normalized["verified_claim_count"] = max(0, verified_claim_count)

    supporting_ratio_raw = value_for("supporting_ratio")
    if supporting_ratio_raw is not None:
        supporting_ratio = to_confidence(supporting_ratio_raw)
        if supporting_ratio is not None:
            normalized["supporting_ratio"] = supporting_ratio

    return normalized


def normalize_outcome(
    value: object,
    *,
    passed: bool,
    valid_outcomes: set[str],
) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in valid_outcomes:
        return normalized
    return "pass" if passed else "fail"


def extract_reason_code_from_text(text: str) -> str:
    match = re.search(
        r"\breason_code\b\s*[:=]\s*([a-z0-9_]+)",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    return str(match.group(1)).strip().lower() if match else ""


def extract_severity_class_from_text(
    text: str,
    *,
    valid_severity_classes: set[str],
) -> str:
    match = re.search(
        r"\b(?:severity_class|severity)\b\s*[:=]\s*([a-z_]+)",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    severity = str(match.group(1)).strip().lower()
    if severity in valid_severity_classes:
        return severity
    return ""


def extract_outcome_from_text(
    text: str,
    *,
    passed: bool,
    valid_outcomes: set[str],
) -> str:
    match = re.search(
        r"\boutcome\b\s*[:=]\s*([a-z_]+)",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return "pass" if passed else "fail"
    return normalize_outcome(
        match.group(1),
        passed=passed,
        valid_outcomes=valid_outcomes,
    )


def coerce_assessment_mapping(
    payload: dict,
    *,
    valid_outcomes: set[str],
    valid_severity_classes: set[str],
) -> dict | None:
    normalized: dict[str, object] = {
        str(key).strip().lower(): value
        for key, value in payload.items()
    }
    passed_raw = (
        normalized.get("passed")
        if "passed" in normalized
        else normalized.get("pass", normalized.get("result"))
    )
    passed = to_bool(passed_raw)
    if passed is None and isinstance(passed_raw, str):
        passed = extract_passed_from_text(passed_raw)
    if passed is None:
        return None

    confidence = to_confidence(normalized.get("confidence"))
    if confidence is None:
        confidence = 0.8 if passed else 0.4

    feedback_raw = (
        normalized.get("feedback")
        or normalized.get("suggestion")
        or normalized.get("reason")
        or normalized.get("rationale")
    )
    feedback = str(feedback_raw).strip() if feedback_raw is not None else None

    issues = to_issues(normalized.get("issues"))
    outcome = normalize_outcome(
        normalized.get("outcome"),
        passed=passed,
        valid_outcomes=valid_outcomes,
    )
    reason_code = str(
        normalized.get("reason_code")
        or normalized.get("failure_reason")
        or "",
    ).strip().lower()
    severity_class = str(
        normalized.get("severity_class")
        or normalized.get("severity")
        or "",
    ).strip().lower()
    if severity_class not in valid_severity_classes:
        severity_class = ""
    metadata: dict[str, object] = {}
    metadata_raw = normalized.get("metadata", {})
    if isinstance(metadata_raw, dict):
        metadata = {
            str(key).strip(): value
            for key, value in metadata_raw.items()
            if str(key).strip()
        }

    base_keys = {
        "passed",
        "pass",
        "result",
        "confidence",
        "feedback",
        "suggestion",
        "reason",
        "rationale",
        "issues",
        "outcome",
        "reason_code",
        "failure_reason",
        "severity_class",
        "severity",
        "metadata",
    }
    for key, value in normalized.items():
        if key in base_keys:
            continue
        if key in metadata:
            continue
        metadata[key] = value

    return {
        "passed": passed,
        "confidence": confidence,
        "feedback": feedback,
        "issues": issues,
        "outcome": outcome,
        "reason_code": reason_code,
        "severity_class": severity_class,
        "metadata": metadata,
    }


def strip_outer_fences(text: str) -> str:
    value = text.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if not lines:
        return value
    body = "\n".join(lines[1:])
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def parse_yaml_like_assessment(
    text: str,
    *,
    valid_outcomes: set[str],
    valid_severity_classes: set[str],
) -> dict | None:
    try:
        import yaml
    except Exception:
        return None

    cleaned = strip_outer_fences(str(text or ""))
    if not cleaned:
        return None

    candidates = [cleaned]
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = yaml.safe_load(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            coerced = coerce_assessment_mapping(
                parsed,
                valid_outcomes=valid_outcomes,
                valid_severity_classes=valid_severity_classes,
            )
            if coerced is not None:
                return coerced
    return None


def extract_passed_from_text(text: str) -> bool | None:
    lowered = str(text or "").lower()
    checks: list[tuple[str, bool]] = [
        (r"\bpassed?\s*[:=]\s*(true|yes|pass(?:ed)?|success(?:ful)?)\b", True),
        (
            r"\bpassed?\s*[:=]\s*(false|no|fail(?:ed)?|failure|unsuccessful)\b",
            False,
        ),
        (
            r"\b(verdict|result|outcome|assessment)\s*[:=]\s*"
            r"(pass(?:ed)?|success(?:ful)?)\b",
            True,
        ),
        (
            r"\b(verdict|result|outcome|assessment)\s*[:=]\s*"
            r"(fail(?:ed)?|failure|unsuccessful)\b",
            False,
        ),
        (r"\bsubtask\s+(passed|succeeded)\b", True),
        (r"\bsubtask\s+(failed|did not pass)\b", False),
        (r"\bacceptance criteria\s+(were|was)\s+met\b", True),
        (r"\bacceptance criteria\s+(were|was)\s+not met\b", False),
    ]
    for pattern, passed in checks:
        if re.search(pattern, lowered):
            return passed
    return None


def extract_confidence_from_text(text: str) -> float | None:
    lowered = str(text or "").lower()
    match = re.search(r"\bconfidence\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?%?)", lowered)
    if not match:
        return None
    return to_confidence(match.group(1))


def extract_feedback_from_text(text: str) -> str | None:
    for label in ("feedback", "suggestion", "reason", "rationale"):
        match = re.search(
            rf"\b{label}\b\s*[:=]\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            value = match.group(1).strip()
            if value:
                return value
    return None


def extract_issues_from_text(text: str) -> list[str]:
    lines = str(text or "").splitlines()
    issues: list[str] = []
    capture = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if capture:
                break
            continue
        lowered = line.lower()
        if lowered.startswith("issues:"):
            capture = True
            trailing = line.split(":", 1)[1].strip()
            if trailing and trailing.lower() not in {"none", "no issues", "n/a"}:
                issues.append(trailing)
            continue
        if capture:
            if line.startswith(("-", "*", "•")):
                issues.append(line.lstrip("-*• ").strip())
                continue
            if ":" in line and not line.startswith(("-", "*", "•")):
                break
            issues.append(line.strip())
    return [item for item in issues if item]


def coerce_assessment_from_text(
    text: str,
    *,
    valid_outcomes: set[str],
    valid_severity_classes: set[str],
) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    structured = parse_yaml_like_assessment(
        raw,
        valid_outcomes=valid_outcomes,
        valid_severity_classes=valid_severity_classes,
    )
    if structured is not None:
        return structured

    passed = extract_passed_from_text(raw)
    if passed is None:
        return None

    confidence = extract_confidence_from_text(raw)
    if confidence is None:
        confidence = 0.8 if passed else 0.4

    feedback = extract_feedback_from_text(raw)
    issues = extract_issues_from_text(raw)

    return {
        "passed": passed,
        "confidence": confidence,
        "feedback": feedback,
        "issues": issues,
        "outcome": extract_outcome_from_text(
            raw,
            passed=passed,
            valid_outcomes=valid_outcomes,
        ),
        "reason_code": extract_reason_code_from_text(raw),
        "severity_class": extract_severity_class_from_text(
            raw,
            valid_severity_classes=valid_severity_classes,
        ),
        "metadata": {},
    }


def assessment_to_result(
    assessment: dict,
    *,
    valid_outcomes: set[str],
    valid_severity_classes: set[str],
) -> VerificationResult:
    passed = bool(assessment.get("passed", True))
    confidence = float(assessment.get("confidence", 0.5))
    issues = to_issues(assessment.get("issues"))
    feedback = assessment.get("feedback") or assessment.get("suggestion")
    feedback_text = str(feedback).strip() if feedback is not None else None
    detail = "; ".join(issues) if issues else None
    outcome = normalize_outcome(
        assessment.get("outcome"),
        passed=passed,
        valid_outcomes=valid_outcomes,
    )
    reason_code = str(assessment.get("reason_code") or "").strip().lower()
    severity_class = str(assessment.get("severity_class") or "").strip().lower()
    if severity_class not in valid_severity_classes:
        severity_class = ""
    if passed and outcome == "pass" and issues:
        outcome = "pass_with_warnings"
    metadata = assessment.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = normalize_verifier_metadata(metadata)
    return VerificationResult(
        tier=2,
        passed=passed,
        confidence=confidence,
        checks=[Check(
            name="llm_assessment",
            passed=passed,
            detail=detail,
        )],
        feedback=feedback_text,
        outcome=outcome,
        reason_code=reason_code,
        severity_class=severity_class,
        metadata={**metadata, "issues": issues},
    )


def parse_verifier_response(
    *,
    response: Any,
    validator: Any,
    expected_keys: list[str],
    valid_outcomes: set[str],
    valid_severity_classes: set[str],
) -> VerificationResult | None:
    validation = validator.validate_json_response(
        response,
        expected_keys=expected_keys,
    )
    if validation.valid and validation.parsed is not None:
        assessment = coerce_assessment_mapping(
            validation.parsed,
            valid_outcomes=valid_outcomes,
            valid_severity_classes=valid_severity_classes,
        )
        if assessment is not None:
            return assessment_to_result(
                assessment,
                valid_outcomes=valid_outcomes,
                valid_severity_classes=valid_severity_classes,
            )

    fallback = coerce_assessment_from_text(
        response.text or "",
        valid_outcomes=valid_outcomes,
        valid_severity_classes=valid_severity_classes,
    )
    if fallback is not None:
        return assessment_to_result(
            fallback,
            valid_outcomes=valid_outcomes,
            valid_severity_classes=valid_severity_classes,
        )
    return None
