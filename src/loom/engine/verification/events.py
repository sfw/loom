"""Verification event emission helpers and payload normalization."""

from __future__ import annotations

from collections.abc import Callable

from loom.events.bus import Event, EventBus
from loom.events.types import (
    CLAIM_VERIFICATION_SUMMARY,
    MODEL_INVOCATION,
    PLACEHOLDER_FINDINGS_EXTRACTED,
    VERIFICATION_CONTRADICTION_DETECTED,
    VERIFICATION_DETERMINISTIC_BLOCK_RATE,
    VERIFICATION_FAILED,
    VERIFICATION_FALSE_NEGATIVE_CANDIDATE,
    VERIFICATION_INCONCLUSIVE_RATE,
    VERIFICATION_OUTCOME,
    VERIFICATION_PASSED,
    VERIFICATION_RULE_APPLIED,
    VERIFICATION_RULE_FAILURE_BY_TYPE,
    VERIFICATION_RULE_SKIPPED,
    VERIFICATION_SHADOW_DIFF,
    VERIFICATION_STARTED,
)

from .development import event_safe_development_summary
from .policy import classify_shadow_diff
from .types import VerificationResult


def emit_rule_scope_event(
    event_bus: EventBus | None,
    *,
    task_id: str,
    subtask_id: str,
    applied: bool,
    rule: object,
    reason: str,
) -> None:
    if not event_bus or not task_id:
        return
    data: dict[str, object] = {
        "subtask_id": subtask_id,
        "rule_id": str(getattr(rule, "name", "") or ""),
        "reason": reason,
        "rule_type": str(getattr(rule, "type", "") or ""),
        "severity": str(getattr(rule, "severity", "") or ""),
        "enforcement": str(getattr(rule, "enforcement", "") or ""),
        "scope": str(getattr(rule, "scope", "") or ""),
        "applies_to_phases": list(getattr(rule, "applies_to_phases", []) or []),
    }
    event_bus.emit(Event(
        event_type=(VERIFICATION_RULE_APPLIED if applied else VERIFICATION_RULE_SKIPPED),
        task_id=task_id,
        data=data,
    ))


def emit_model_invocation_event(
    event_bus: EventBus | None,
    *,
    task_id: str,
    subtask_id: str,
    model_name: str,
    phase: str,
    details: dict | None = None,
) -> None:
    if not event_bus or not task_id:
        return
    data: dict = {
        "subtask_id": subtask_id,
        "model": model_name,
        "phase": phase,
    }
    if isinstance(details, dict) and details:
        data.update(details)
    event_bus.emit(Event(
        event_type=MODEL_INVOCATION,
        task_id=task_id,
        data=data,
    ))


def emit_claim_verification_summary(
    event_bus: EventBus | None,
    *,
    task_id: str,
    subtask_id: str,
    counts: dict[str, int],
) -> None:
    if not event_bus or not task_id:
        return
    event_bus.emit(Event(
        event_type=CLAIM_VERIFICATION_SUMMARY,
        task_id=task_id,
        data={
            "subtask_id": subtask_id,
            "extracted": int(counts.get("extracted", 0)),
            "supported": int(counts.get("supported", 0)),
            "partially_supported": int(counts.get("partially_supported", 0)),
            "contradicted": int(counts.get("contradicted", 0)),
            "insufficient_evidence": int(counts.get("insufficient_evidence", 0)),
            "pruned": int(counts.get("pruned", 0)),
        },
    ))


def emit_verification_outcome(
    event_bus: EventBus | None,
    *,
    task_id: str,
    subtask_id: str,
    result: VerificationResult,
) -> None:
    if not event_bus or not task_id:
        return
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    dev_summary = event_safe_development_summary(
        metadata.get("dev_verification_summary", {}),
    )
    data: dict[str, object] = {
        "subtask_id": subtask_id,
        "tier": result.tier,
        "passed": result.passed,
        "outcome": result.outcome,
        "reason_code": result.reason_code,
        "severity_class": result.severity_class,
        "confidence": result.confidence,
        "source_component": "verification",
    }
    if dev_summary:
        data["dev_verification_summary"] = dev_summary
    event_bus.emit(Event(
        event_type=VERIFICATION_OUTCOME,
        task_id=task_id,
        data=data,
    ))


def emit_verification_started(
    event_bus: EventBus | None,
    *,
    task_id: str,
    subtask_id: str,
    target_tier: int,
) -> None:
    if not event_bus or not task_id:
        return
    event_bus.emit(Event(
        event_type=VERIFICATION_STARTED,
        task_id=task_id,
        data={
            "subtask_id": subtask_id,
            "target_tier": max(1, int(target_tier or 1)),
            "source_component": "verification",
        },
    ))


def emit_verification_terminal(
    event_bus: EventBus | None,
    *,
    task_id: str,
    subtask_id: str,
    result: VerificationResult,
) -> None:
    if not event_bus or not task_id:
        return
    terminal_type = VERIFICATION_PASSED if bool(result.passed) else VERIFICATION_FAILED
    outcome = str(result.outcome or "").strip() or ("pass" if result.passed else "fail")
    reason_code = str(result.reason_code or "").strip()
    if not reason_code:
        reason_code = "verification_passed" if result.passed else "verification_failed"
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    dev_summary = event_safe_development_summary(
        metadata.get("dev_verification_summary", {}),
    )
    data: dict[str, object] = {
        "subtask_id": subtask_id,
        "tier": int(result.tier),
        "outcome": outcome,
        "reason_code": reason_code,
        "severity_class": str(result.severity_class or ""),
        "confidence": float(result.confidence),
        "source_component": "verification",
    }
    if dev_summary:
        data["dev_verification_summary"] = dev_summary
    event_bus.emit(Event(
        event_type=terminal_type,
        task_id=task_id,
        data=data,
    ))


def emit_placeholder_findings_extracted(
    event_bus: EventBus | None,
    *,
    task_id: str,
    subtask_id: str,
    result: VerificationResult,
) -> None:
    if not event_bus or not task_id:
        return
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    raw_findings = metadata.get("placeholder_findings")
    if not isinstance(raw_findings, list) or not raw_findings:
        return
    findings: list[dict[str, object]] = []
    for raw in raw_findings:
        if not isinstance(raw, dict):
            continue
        findings.append({
            "rule_name": str(raw.get("rule_name", "") or "").strip(),
            "file_path": str(raw.get("file_path", "") or "").strip(),
            "line": _safe_int(raw.get("line", 0)),
            "column": _safe_int(raw.get("column", 0)),
            "token": str(raw.get("token", "") or ""),
            "context": str(raw.get("context", "") or ""),
        })
        if len(findings) >= 120:
            break
    if not findings:
        return
    finding_count = int(
        metadata.get("placeholder_finding_count", len(findings)) or len(findings),
    )
    missing_targets_raw = metadata.get("missing_targets", [])
    missing_targets = (
        list(missing_targets_raw)
        if isinstance(missing_targets_raw, list)
        else []
    )
    event_bus.emit(Event(
        event_type=PLACEHOLDER_FINDINGS_EXTRACTED,
        task_id=task_id,
        data={
            "subtask_id": subtask_id,
            "tier": result.tier,
            "reason_code": result.reason_code,
            "remediation_mode": str(metadata.get("remediation_mode", "") or ""),
            "failure_class": str(metadata.get("failure_class", "") or ""),
            "finding_count": finding_count,
            "missing_targets": missing_targets,
            "findings": findings,
        },
    ))


def emit_instrumentation_events(
    event_bus: EventBus | None,
    *,
    task_id: str,
    subtask_id: str,
    result: VerificationResult,
    legacy_result: VerificationResult | None = None,
    classify_shadow_diff_fn: Callable[[VerificationResult, VerificationResult], str] = (
        classify_shadow_diff
    ),
) -> None:
    if not event_bus or not task_id:
        return

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    scan = metadata.get("deterministic_placeholder_scan", {})
    if isinstance(scan, dict):
        scan_dict = scan if isinstance(scan, dict) else {}
        candidate_source_counts = scan_dict.get("candidate_source_counts", {})
        if not isinstance(candidate_source_counts, dict):
            candidate_source_counts = {}
        contradiction_downgraded = bool(metadata.get("contradiction_downgraded", False))
        contradiction_detected_no_downgrade = bool(
            metadata.get("contradiction_detected_no_downgrade", False),
        )
        event_bus.emit(Event(
            event_type=VERIFICATION_CONTRADICTION_DETECTED,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "tier": result.tier,
                "reason_code": result.reason_code,
                "contradicted_reason_code": str(
                    metadata.get("contradicted_reason_code", ""),
                ),
                "scanned_file_count": int(
                    scan_dict.get("scanned_file_count", 0) or 0,
                ),
                "scanned_total_bytes": int(
                    scan_dict.get("scanned_total_bytes", 0) or 0,
                ),
                "matched_file_count": int(
                    scan_dict.get("matched_file_count", 0) or 0,
                ),
                "scan_mode": str(
                    scan_dict.get("scan_mode", "targeted_only") or "targeted_only",
                ),
                "coverage_sufficient": bool(
                    scan_dict.get("coverage_sufficient", False),
                ),
                "candidate_source_counts": candidate_source_counts,
                "coverage_insufficient_reason": str(
                    scan_dict.get("coverage_insufficient_reason", "") or "",
                ),
                "contradiction_downgrade_count": 1 if contradiction_downgraded else 0,
                "contradiction_detected_no_downgrade_count": (
                    1 if contradiction_detected_no_downgrade else 0
                ),
                "cap_exhaustion_count": (
                    1 if bool(scan_dict.get("cap_exhausted", False)) else 0
                ),
            },
        ))

    if result.reason_code == "parse_inconclusive":
        event_bus.emit(Event(
            event_type=VERIFICATION_INCONCLUSIVE_RATE,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "tier": result.tier,
                "reason_code": result.reason_code,
            },
        ))

    if result.tier == 1 and not result.passed:
        event_bus.emit(Event(
            event_type=VERIFICATION_DETERMINISTIC_BLOCK_RATE,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "reason_code": result.reason_code,
                "failed_checks": len([c for c in result.checks if not c.passed]),
            },
        ))

    for check in result.checks:
        if check.passed:
            continue
        if not check.name.startswith("process_rule_"):
            continue
        event_bus.emit(Event(
            event_type=VERIFICATION_RULE_FAILURE_BY_TYPE,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "check_name": check.name,
                "tier": result.tier,
                "reason_code": result.reason_code,
            },
        ))
    if result.tier >= 2 and not result.passed:
        event_bus.emit(Event(
            event_type=VERIFICATION_RULE_FAILURE_BY_TYPE,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "check_name": "llm_assessment",
                "tier": result.tier,
                "reason_code": result.reason_code,
            },
        ))

    if legacy_result is None:
        return

    classification = classify_shadow_diff_fn(legacy_result, result)
    if classification == "old_fail_new_pass":
        event_bus.emit(Event(
            event_type=VERIFICATION_FALSE_NEGATIVE_CANDIDATE,
            task_id=task_id,
            data={
                "subtask_id": subtask_id,
                "old_outcome": legacy_result.outcome,
                "new_outcome": result.outcome,
            },
        ))

    event_bus.emit(Event(
        event_type=VERIFICATION_SHADOW_DIFF,
        task_id=task_id,
        data={
            "subtask_id": subtask_id,
            "classification": classification,
            "old_passed": legacy_result.passed,
            "new_passed": result.passed,
            "old_reason_code": legacy_result.reason_code,
            "new_reason_code": result.reason_code,
            "old_outcome": legacy_result.outcome,
            "new_outcome": result.outcome,
        },
    ))


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
