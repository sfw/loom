"""Claim-lifecycle extraction and metadata attachment helpers."""

from __future__ import annotations

import hashlib
import re

from . import events as verification_events
from .types import AssertionEnvelope, VerificationResult


def normalize_claim_type(text: str) -> str:
    value = str(text or "")
    lowered = value.lower()
    if re.search(r"\b\d[\d,]*(?:\.\d+)?\b", lowered):
        return "numeric"
    if re.search(r"\b(?:19|20)\d{2}\b", lowered):
        return "date"
    if re.search(r"\b(?:forecast|project|expect|guidance|assum)\b", lowered):
        return "forecast_assumption"
    if re.search(r"\b(?:company|entity|organization|person|team)\b", lowered):
        return "entity_fact"
    return "qualitative"


def claim_id(text: str) -> str:
    digest = hashlib.sha1(
        str(text or "").encode("utf-8", errors="replace"),
    ).hexdigest().upper()
    return f"CLM-{digest[:10]}"


def claim_status_from_fact_verdict(verdict: str, reason_hint: str = "") -> tuple[str, str]:
    normalized = str(verdict or "").strip().lower()
    if normalized == "supported":
        return "supported", "claim_supported"
    if normalized == "partially_supported":
        return "partially_supported", "claim_partially_supported"
    if normalized == "contradicted":
        return "contradicted", "claim_contradicted"
    if normalized == "stale":
        return "stale", "claim_stale_source"
    normalized_reason = str(reason_hint or "").strip().lower()
    if normalized_reason in {
        "semantic_inconclusive",
        "verifier_unavailable",
        "verifier_parse_inconclusive",
    }:
        return "insufficient_evidence", "claim_inconclusive"
    return "insufficient_evidence", "claim_insufficient_evidence"


def critical_claim_types(validity_contract: dict[str, object]) -> set[str]:
    raw = validity_contract.get("critical_claim_types", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return set()
    return {
        str(item or "").strip().lower()
        for item in raw
        if str(item or "").strip()
    }


def extract_claim_lifecycle(
    gates,
    *,
    tool_calls: list,
    result: VerificationResult,
    validity_contract: dict[str, object],
) -> list[dict[str, object]]:
    del result  # retained for compatibility with prior method signature
    claims: list[dict[str, object]] = []
    critical_types = critical_claim_types(validity_contract)
    for tc in tool_calls:
        if str(getattr(tc, "tool", "") or "").strip().lower() != "fact_checker":
            continue
        tool_result = getattr(tc, "result", None)
        if tool_result is None or not bool(getattr(tool_result, "success", False)):
            continue
        data = getattr(tool_result, "data", {})
        if not isinstance(data, dict):
            data = {}
        verdicts = data.get("verdicts", [])
        if not isinstance(verdicts, list):
            continue
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                continue
            text = str(verdict.get("claim", "") or "").strip()
            if not text:
                continue
            status, reason = claim_status_from_fact_verdict(
                str(verdict.get("verdict", "") or ""),
                str(verdict.get("reason_code", "") or ""),
            )
            claim_type = normalize_claim_type(text)
            source_hint = str(verdict.get("source", "") or "").strip()
            as_of = str(verdict.get("as_of", "") or "").strip()
            source_as_of = str(verdict.get("source_as_of", "") or "").strip()
            period_start = str(verdict.get("period_start", "") or "").strip()
            period_end = str(verdict.get("period_end", "") or "").strip()
            claims.append({
                "claim_id": claim_id(text),
                "text": text,
                "claim_type": claim_type,
                "criticality": (
                    "critical" if claim_type in critical_types else "important"
                ),
                "status": status,
                "reason_code": reason,
                "evidence_refs": [source_hint] if source_hint else [],
                "as_of": as_of,
                "source_as_of": source_as_of,
                "period_start": period_start,
                "period_end": period_end,
                "lifecycle": ["extracted", status],
            })

    if claims:
        return claims

    # Do not fabricate synthetic supported claims from aggregate counters.
    # Missing concrete claim verdicts should remain "no extracted claims" so
    # synthesis gates can enforce stricter behavior.
    return claims


def claim_counts(claims: list[dict[str, object]]) -> dict[str, int]:
    counts = {
        "extracted": len(claims),
        "supported": 0,
        "partially_supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
        "stale": 0,
        "pruned": 0,
    }
    for claim in claims:
        status = str(claim.get("status", "") or "").strip().lower()
        if status in counts:
            counts[status] += 1
    return counts


def assertion_verdict_from_claim_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "supported":
        return "supported"
    if normalized == "partially_supported":
        return "partially_supported"
    if normalized == "contradicted":
        return "contradicted"
    if normalized == "stale":
        return "failed_contract"
    if normalized in {"insufficient_evidence", "extracted", "pruned"}:
        return "inconclusive"
    return "inconclusive"


def assertions_from_claim_lifecycle(
    claims: list[dict[str, object]],
) -> list[AssertionEnvelope]:
    assertions: list[AssertionEnvelope] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = str(claim.get("text", "") or "").strip()
        if not text:
            continue
        status = str(claim.get("status", "") or "").strip().lower()
        evidence_refs = claim.get("evidence_refs", [])
        if isinstance(evidence_refs, str):
            evidence_refs = [evidence_refs]
        if not isinstance(evidence_refs, list):
            evidence_refs = []
        assertion = AssertionEnvelope(
            assertion_id=str(claim.get("claim_id", "") or claim_id(text)).strip(),
            assertion_type="fact",
            verdict=assertion_verdict_from_claim_status(status),
            confidence=(
                0.9 if status == "supported"
                else 0.7 if status == "partially_supported"
                else 0.2 if status in {"contradicted", "stale"}
                else 0.35
            ),
            reason_code=str(claim.get("reason_code", "") or "").strip().lower(),
            evidence_refs=[
                str(ref or "").strip()
                for ref in evidence_refs
                if str(ref or "").strip()
            ],
            remediation_hint=(
                "Collect additional source evidence."
                if status in {"insufficient_evidence", "extracted"}
                else "Correct contradictory or stale claim text."
                if status in {"contradicted", "stale"}
                else ""
            ),
            source="fact_checker",
        )
        assertions.append(assertion)
    return assertions


def assertion_counts(assertions: list[AssertionEnvelope]) -> dict[str, int]:
    counts = {
        "total": 0,
        "supported": 0,
        "partially_supported": 0,
        "contradicted": 0,
        "inconclusive": 0,
        "failed_contract": 0,
    }
    for assertion in assertions:
        counts["total"] += 1
        verdict = str(assertion.verdict or "").strip().lower()
        if verdict in counts:
            counts[verdict] += 1
    return counts


def attach_claim_lifecycle(
    gates,
    *,
    task_id: str,
    subtask_id: str,
    result: VerificationResult,
    tool_calls: list,
    validity_contract: dict[str, object],
) -> VerificationResult:
    metadata = dict(result.metadata) if isinstance(result.metadata, dict) else {}
    claims = extract_claim_lifecycle(
        gates,
        tool_calls=tool_calls,
        result=result,
        validity_contract=validity_contract,
    )
    counts = claim_counts(claims)
    assertions = assertions_from_claim_lifecycle(claims)
    assertion_count_summary = assertion_counts(assertions)
    metadata["claim_lifecycle"] = claims
    metadata["claim_status_counts"] = counts
    metadata["assertion_envelope"] = [item.to_dict() for item in assertions]
    metadata["assertion_counts"] = assertion_count_summary
    metadata["claim_reason_codes"] = sorted({
        str(item.get("reason_code", "") or "").strip().lower()
        for item in claims
        if str(item.get("reason_code", "") or "").strip()
    })
    verification_events.emit_claim_verification_summary(
        getattr(gates, "_event_bus", None),
        task_id=task_id,
        subtask_id=subtask_id,
        counts=counts,
    )
    return VerificationResult(
        tier=result.tier,
        passed=result.passed,
        confidence=result.confidence,
        checks=list(result.checks or []),
        feedback=result.feedback,
        outcome=result.outcome,
        reason_code=result.reason_code,
        severity_class=result.severity_class,
        metadata=metadata,
    )
