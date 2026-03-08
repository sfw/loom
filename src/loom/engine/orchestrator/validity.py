"""Pure validity-contract and metadata normalization helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from loom.engine.runner import SubtaskResult, SubtaskResultStatus, ToolCallRecord
from loom.engine.verification import VerificationResult
from loom.events.types import CLAIMS_PRUNED
from loom.state.evidence import merge_evidence_records
from loom.state.task_state import Subtask, Task

logger = logging.getLogger(__name__)

_CLAIM_TERMINAL_UNRESOLVED = frozenset({
    "contradicted",
    "insufficient_evidence",
    "extracted",
    "stale",
})
_CLAIM_REASON_CODES = {
    "supported": "claim_supported",
    "contradicted": "claim_contradicted",
    "insufficient_evidence": "claim_insufficient_evidence",
    "stale": "claim_stale_source",
    "pruned": "claim_pruned",
}
_CLAIM_RECOVERABLE_FAILURE_CODES = frozenset({
    "recommendation_unconfirmed",
    "unconfirmed_noncritical",
    "unconfirmed_critical_path",
    "claim_insufficient_evidence",
    "claim_contradicted",
    "claim_stale_source",
    "coverage_below_threshold",
})


def hash_validity_contract(contract: dict[str, object]) -> str:
    """Hash normalized validity contracts for snapshot drift detection."""
    serialized = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        serialized.encode("utf-8", errors="replace"),
    ).hexdigest()


def to_bool(value: object, default: bool = False) -> bool:
    """Parse permissive boolean values from process payloads."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value or "").strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def to_ratio(value: object, default: float) -> float:
    """Clamp ratio-like values into [0, 1]."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(0.0, min(1.0, parsed))


def to_non_negative_int(value: object, default: int) -> int:
    """Parse int values and clamp negatives to zero."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    return max(0, parsed)


def normalize_validity_contract(contract: dict[str, object] | None) -> dict[str, object]:
    """Normalize validity contract payloads into canonical shape."""
    payload = dict(contract or {})
    claim_extraction_raw = payload.get("claim_extraction", {})
    if isinstance(claim_extraction_raw, bool):
        claim_extraction_raw = {"enabled": claim_extraction_raw}
    if not isinstance(claim_extraction_raw, dict):
        claim_extraction_raw = {}

    final_gate_raw = payload.get("final_gate", {})
    if not isinstance(final_gate_raw, dict):
        final_gate_raw = {}
    temporal_raw = final_gate_raw.get("temporal_consistency", {})
    if not isinstance(temporal_raw, dict):
        temporal_raw = {}

    critical_claim_types_raw = payload.get("critical_claim_types", [])
    if isinstance(critical_claim_types_raw, str):
        critical_claim_types_raw = [critical_claim_types_raw]
    if not isinstance(critical_claim_types_raw, list):
        critical_claim_types_raw = []

    prune_mode = str(payload.get("prune_mode", "drop") or "").strip().lower()
    if prune_mode not in {"drop", "rewrite_uncertainty"}:
        prune_mode = "drop"

    return {
        "enabled": to_bool(payload.get("enabled", False), False),
        "claim_extraction": {
            "enabled": to_bool(claim_extraction_raw.get("enabled", False), False),
        },
        "critical_claim_types": list(dict.fromkeys(
            str(item or "").strip().lower()
            for item in critical_claim_types_raw
            if str(item or "").strip()
        )),
        "min_supported_ratio": to_ratio(payload.get("min_supported_ratio", 0.75), 0.75),
        "max_unverified_ratio": to_ratio(payload.get("max_unverified_ratio", 0.25), 0.25),
        "max_contradicted_count": to_non_negative_int(
            payload.get("max_contradicted_count", 0),
            0,
        ),
        "prune_mode": prune_mode,
        "require_fact_checker_for_synthesis": to_bool(
            payload.get("require_fact_checker_for_synthesis", False),
            False,
        ),
        "final_gate": {
            "enforce_verified_context_only": to_bool(
                final_gate_raw.get("enforce_verified_context_only", True),
                True,
            ),
            "synthesis_min_verification_tier": max(
                1,
                to_non_negative_int(
                    final_gate_raw.get("synthesis_min_verification_tier", 2),
                    2,
                ),
            ),
            "critical_claim_support_ratio": to_ratio(
                final_gate_raw.get("critical_claim_support_ratio", 1.0),
                1.0,
            ),
            "temporal_consistency": {
                "enabled": to_bool(temporal_raw.get("enabled", False), False),
                "require_as_of_alignment": to_bool(
                    temporal_raw.get("require_as_of_alignment", False),
                    False,
                ),
                "enforce_cross_claim_date_conflict_check": to_bool(
                    temporal_raw.get("enforce_cross_claim_date_conflict_check", False),
                    False,
                ),
                "max_source_age_days": to_non_negative_int(
                    temporal_raw.get("max_source_age_days", 0),
                    0,
                ),
                "as_of": str(temporal_raw.get("as_of", "") or "").strip(),
            },
        },
    }


def normalize_missing_targets(raw: object) -> list[str]:
    """Normalize missing-target fields from metadata payloads."""
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if "," in text:
            candidates = [item.strip() for item in text.split(",")]
        else:
            candidates = [text]
        return [item for item in candidates if item]
    if isinstance(raw, list):
        deduped: list[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text and text not in deduped:
                deduped.append(text)
        return deduped
    return []


def to_int_or_none(value: object) -> int | None:
    """Best-effort integer parsing for loose metadata values."""
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


def to_ratio_or_none(value: object) -> float | None:
    """Best-effort ratio parsing supporting whole percentages."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text.rstrip("%"))
        except ValueError:
            return None
        if text.endswith("%"):
            numeric /= 100.0
    else:
        return None
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return max(0.0, min(1.0, numeric))


def to_float_or_none(value: object) -> float | None:
    """Best-effort float parsing for retry/backoff metadata."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def normalize_placeholder_findings(
    findings: object,
    *,
    max_items: int = 120,
) -> list[dict[str, object]]:
    """Normalize and dedupe placeholder findings for downstream consumers."""
    if not isinstance(findings, list):
        return []
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, int, int, str, str]] = set()
    for raw in findings:
        if not isinstance(raw, dict):
            continue
        file_path = str(raw.get("file_path", "") or "").strip()
        token = str(raw.get("token", "") or "")
        rule_name = str(raw.get("rule_name", "") or "").strip()
        pattern = str(raw.get("pattern", "") or "")
        line = to_int_or_none(raw.get("line")) or 0
        column = to_int_or_none(raw.get("column")) or 0
        key = (file_path, max(0, line), max(0, column), token, pattern)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "rule_name": rule_name,
            "pattern": pattern,
            "source": str(raw.get("source", "") or ""),
            "file_path": file_path,
            "line": max(0, line),
            "column": max(0, column),
            "token": token,
            "context": str(raw.get("context", "") or ""),
        })
        if len(normalized) >= max_items:
            break
    return normalized


def normalize_workspace_relpath(workspace: Path, raw_path: str) -> str | None:
    """Normalize file paths into workspace-relative POSIX paths."""
    text = str(raw_path or "").strip()
    if not text:
        return None
    root = workspace.resolve(strict=False)
    candidate = Path(text)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (root / candidate).resolve(strict=False)
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        return None
    rel_text = rel.as_posix().strip()
    if not rel_text or rel_text == ".":
        return None
    return rel_text


def compact_failure_resolution_metadata_value(
    value: object,
    *,
    depth: int = 0,
    max_depth: int = 3,
    max_list_items: int = 8,
    max_dict_items: int = 12,
    max_text_chars: int = 220,
) -> object:
    """Compact arbitrary metadata for bounded prompt inclusion."""
    if depth >= max_depth:
        text = str(value or "").strip()
        if len(text) > max_text_chars:
            return text[: max_text_chars - 14] + "...[truncated]"
        return text

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        text = value.strip()
        if len(text) > max_text_chars:
            return text[: max_text_chars - 14] + "...[truncated]"
        return text

    if isinstance(value, list):
        items: list[object] = []
        for item in value[:max_list_items]:
            items.append(
                compact_failure_resolution_metadata_value(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_list_items=max_list_items,
                    max_dict_items=max_dict_items,
                    max_text_chars=max_text_chars,
                ),
            )
        remainder = len(value) - len(items)
        if remainder > 0:
            items.append(f"...[{remainder} more items]")
        return items

    if isinstance(value, dict):
        compact: dict[str, object] = {}
        items = list(value.items())[:max_dict_items]
        for key, raw in items:
            key_text = str(key or "").strip()[:80]
            if not key_text:
                continue
            compact[key_text] = compact_failure_resolution_metadata_value(
                raw,
                depth=depth + 1,
                max_depth=max_depth,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                max_text_chars=max_text_chars,
            )
        remainder = len(value) - len(items)
        if remainder > 0:
            compact["_truncated_keys"] = remainder
        return compact

    text = str(value or "").strip()
    if len(text) > max_text_chars:
        return text[: max_text_chars - 14] + "...[truncated]"
    return text


def summarize_failure_resolution_metadata(
    metadata: dict[str, object],
    *,
    keys: tuple[str, ...],
) -> dict[str, object]:
    """Summarize failure-resolution metadata to prioritized compact fields."""
    if not isinstance(metadata, dict):
        return {}

    summary: dict[str, object] = {}
    for key in keys:
        if key not in metadata:
            continue
        summary[key] = compact_failure_resolution_metadata_value(
            metadata.get(key),
        )

    scan = metadata.get("deterministic_placeholder_scan")
    if isinstance(scan, dict):
        prioritized_scan = {
            key: scan.get(key)
            for key in (
                "scan_mode",
                "scanned_file_count",
                "matched_file_count",
                "coverage_sufficient",
                "coverage_insufficient_reason",
                "cap_exhausted",
                "cap_exhaustion_reason",
                "candidate_source_counts",
            )
            if key in scan
        }
        if prioritized_scan:
            summary["deterministic_placeholder_scan"] = (
                compact_failure_resolution_metadata_value(
                    prioritized_scan,
                )
            )

    if summary:
        return summary

    # Fallback for unknown schemas: include a small, compact preview.
    for key, raw in metadata.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue
        summary[key_text[:64]] = compact_failure_resolution_metadata_value(raw)
        if len(summary) >= 6:
            break
    return summary


# Extracted validity + claim lifecycle orchestration helpers

def _validity_contract_for_subtask(self, subtask: Subtask) -> dict[str, object]:
    contract = (
        dict(subtask.validity_contract_snapshot)
        if isinstance(subtask.validity_contract_snapshot, dict)
        and subtask.validity_contract_snapshot
        else self._resolve_subtask_validity_contract(subtask=subtask)
    )
    normalized = self._normalize_validity_contract(contract)
    subtask.validity_contract_snapshot = normalized
    subtask.validity_contract_hash = self._hash_validity_contract(normalized)
    return normalized

def _synthesis_verification_floor(self, subtask: Subtask) -> int:
    if not subtask.is_synthesis:
        return max(1, int(subtask.verification_tier or 1))
    contract = self._validity_contract_for_subtask(subtask)
    final_gate = contract.get("final_gate", {})
    if isinstance(final_gate, dict):
        floor = max(
            1,
            self._to_non_negative_int(
                final_gate.get("synthesis_min_verification_tier", 2),
                2,
            ),
        )
    else:
        floor = 2
    return max(floor, int(subtask.verification_tier or 1))

def _tool_call_succeeded(call: ToolCallRecord) -> bool:
    result = getattr(call, "result", None)
    return bool(result is not None and getattr(result, "success", False))

def _fact_checker_used(self, tool_calls: list[ToolCallRecord]) -> bool:
    for call in tool_calls:
        if str(getattr(call, "tool", "") or "").strip().lower() != "fact_checker":
            continue
        if self._tool_call_succeeded(call):
            return True
    return False

def _fact_checker_verdict_count(self, tool_calls: list[ToolCallRecord]) -> int:
    total = 0
    for call in tool_calls:
        if str(getattr(call, "tool", "") or "").strip().lower() != "fact_checker":
            continue
        result = getattr(call, "result", None)
        if result is None or not bool(getattr(result, "success", False)):
            continue
        data = getattr(result, "data", {})
        if not isinstance(data, dict):
            continue
        verdicts = data.get("verdicts", [])
        if not isinstance(verdicts, list):
            continue
        total += sum(
            1
            for verdict in verdicts
            if isinstance(verdict, dict)
            and str(verdict.get("claim", "") or "").strip()
        )
    return total

def _requires_fact_checker_for_subtask(self, subtask: Subtask) -> bool:
    if not subtask.is_synthesis:
        return False
    contract = self._validity_contract_for_subtask(subtask)
    if not self._to_bool(contract.get("enabled", False), False):
        return False
    return self._to_bool(
        contract.get("require_fact_checker_for_synthesis", False),
        False,
    )

def _claim_graph_state(self, task: Task) -> dict[str, object]:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    graph = metadata.get("claim_graph")
    if not isinstance(graph, dict):
        graph = {}
    supported = graph.get("supported_by_subtask")
    if not isinstance(supported, dict):
        supported = {}
    graph["supported_by_subtask"] = supported
    unresolved = graph.get("unresolved_by_subtask")
    if not isinstance(unresolved, dict):
        unresolved = {}
    graph["unresolved_by_subtask"] = unresolved
    metadata["claim_graph"] = graph
    task.metadata = metadata
    return graph

def _update_claim_graph_from_verification(
    self,
    *,
    task: Task,
    subtask: Subtask,
    verification: VerificationResult,
) -> None:
    metadata = verification.metadata if isinstance(verification.metadata, dict) else {}
    claims = metadata.get("claim_lifecycle", [])
    if not isinstance(claims, list):
        return
    graph = self._claim_graph_state(task)
    supported_by_subtask = graph.get("supported_by_subtask")
    unresolved_by_subtask = graph.get("unresolved_by_subtask")
    if (
        not isinstance(supported_by_subtask, dict)
        or not isinstance(unresolved_by_subtask, dict)
    ):
        return
    supported_claims: list[dict[str, object]] = []
    unresolved_claims: list[dict[str, object]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        status = str(claim.get("status", "") or "").strip().lower()
        if status == "supported":
            supported_claims.append(dict(claim))
        elif status in _CLAIM_TERMINAL_UNRESOLVED:
            unresolved_claims.append(dict(claim))
    supported_by_subtask[subtask.id] = supported_claims
    unresolved_by_subtask[subtask.id] = unresolved_claims

def _claims_from_verification(verification: VerificationResult) -> list[dict[str, object]]:
    metadata = verification.metadata if isinstance(verification.metadata, dict) else {}
    raw_claims = metadata.get("claim_lifecycle", [])
    if not isinstance(raw_claims, list):
        return []
    claims: list[dict[str, object]] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        claim = dict(item)
        status = str(claim.get("status", "extracted") or "extracted").strip().lower()
        if not status:
            status = "extracted"
        claim["status"] = status
        claim["claim_id"] = str(claim.get("claim_id", "") or "").strip()
        claim["text"] = str(claim.get("text", "") or "").strip()
        claim["claim_type"] = str(
            claim.get("claim_type", "qualitative") or "qualitative",
        ).strip().lower()
        claim["criticality"] = str(
            claim.get("criticality", "important") or "important",
        ).strip().lower()
        reason_code = str(claim.get("reason_code", "") or "").strip().lower()
        claim["reason_code"] = reason_code
        refs = claim.get("evidence_refs", [])
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            refs = []
        normalized_refs = [
            str(ref or "").strip()
            for ref in refs
            if str(ref or "").strip()
        ]
        claim["evidence_refs"] = normalized_refs
        lifecycle = claim.get("lifecycle", [])
        if isinstance(lifecycle, str):
            lifecycle = [lifecycle]
        if not isinstance(lifecycle, list):
            lifecycle = []
        normalized_lifecycle = [
            str(step or "").strip().lower()
            for step in lifecycle
            if str(step or "").strip()
        ]
        if "extracted" not in normalized_lifecycle:
            normalized_lifecycle.insert(0, "extracted")
        if status not in normalized_lifecycle:
            normalized_lifecycle.append(status)
        claim["lifecycle"] = normalized_lifecycle
        claims.append(claim)
    return claims

def _normalize_claim_reason_code(status: str, reason_code: str) -> str:
    normalized_reason = str(reason_code or "").strip().lower()
    if normalized_reason:
        return normalized_reason
    normalized_status = str(status or "").strip().lower()
    if normalized_status in _CLAIM_REASON_CODES:
        return _CLAIM_REASON_CODES[normalized_status]
    return "claim_insufficient_evidence"

def _claim_counts(claims: list[dict[str, object]]) -> dict[str, int]:
    counts = {
        "extracted": len(claims),
        "supported": 0,
        "contradicted": 0,
        "insufficient_evidence": 0,
        "stale": 0,
        "pruned": 0,
        "unresolved": 0,
        "critical_total": 0,
        "critical_supported": 0,
        "critical_contradicted": 0,
    }
    for claim in claims:
        status = str(claim.get("status", "") or "").strip().lower()
        if status in counts:
            counts[status] += 1
        if status in _CLAIM_TERMINAL_UNRESOLVED:
            counts["unresolved"] += 1
        criticality = str(claim.get("criticality", "") or "").strip().lower()
        if criticality != "critical":
            continue
        counts["critical_total"] += 1
        if status == "supported":
            counts["critical_supported"] += 1
        elif status == "contradicted":
            counts["critical_contradicted"] += 1
    return counts

def _claim_ratios(counts: dict[str, int]) -> dict[str, float]:
    extracted = max(0, int(counts.get("extracted", 0) or 0))
    supported = max(0, int(counts.get("supported", 0) or 0))
    unresolved = max(0, int(counts.get("unresolved", 0) or 0))
    critical_total = max(0, int(counts.get("critical_total", 0) or 0))
    critical_supported = max(0, int(counts.get("critical_supported", 0) or 0))
    return {
        "supported_ratio": (float(supported) / float(extracted)) if extracted > 0 else 1.0,
        "unverified_ratio": (float(unresolved) / float(extracted)) if extracted > 0 else 0.0,
        "critical_support_ratio": (
            float(critical_supported) / float(critical_total)
        ) if critical_total > 0 else 1.0,
    }

def _verification_with_metadata(
    verification: VerificationResult,
    *,
    metadata: dict[str, object],
    passed: bool | None = None,
    outcome: str | None = None,
    reason_code: str | None = None,
    feedback: str | None = None,
    severity_class: str | None = None,
    confidence: float | None = None,
) -> VerificationResult:
    return VerificationResult(
        tier=int(verification.tier),
        passed=verification.passed if passed is None else bool(passed),
        confidence=float(verification.confidence if confidence is None else confidence),
        checks=list(verification.checks or []),
        feedback=verification.feedback if feedback is None else feedback,
        outcome=str(verification.outcome if outcome is None else outcome),
        reason_code=str(verification.reason_code if reason_code is None else reason_code),
        severity_class=(
            str(verification.severity_class or "")
            if severity_class is None
            else str(severity_class)
        ),
        metadata=metadata,
    )

def _apply_intermediate_claim_pruning(
    self,
    *,
    task: Task,
    subtask: Subtask,
    result: SubtaskResult,
    verification: VerificationResult,
    contract: dict[str, object],
) -> VerificationResult:
    if subtask.is_synthesis:
        return verification
    claims = self._claims_from_verification(verification)
    if not claims:
        return verification
    unresolved = [
        claim for claim in claims
        if str(claim.get("status", "") or "").strip().lower() in _CLAIM_TERMINAL_UNRESOLVED
    ]
    if not unresolved:
        return verification

    prune_mode = str(contract.get("prune_mode", "drop") or "").strip().lower()
    if prune_mode not in {"drop", "rewrite_uncertainty"}:
        prune_mode = "drop"

    supported_claims: list[dict[str, object]] = []
    pruned_claims: list[dict[str, object]] = []
    uncertainty_notes: list[str] = []
    for claim in claims:
        normalized = dict(claim)
        status = str(normalized.get("status", "") or "").strip().lower()
        reason = self._normalize_claim_reason_code(
            status=status,
            reason_code=str(normalized.get("reason_code", "") or ""),
        )
        normalized["reason_code"] = reason
        lifecycle = normalized.get("lifecycle", [])
        if isinstance(lifecycle, str):
            lifecycle = [lifecycle]
        if not isinstance(lifecycle, list):
            lifecycle = []
        lifecycle_norm = [
            str(item or "").strip().lower()
            for item in lifecycle
            if str(item or "").strip()
        ]
        if "extracted" not in lifecycle_norm:
            lifecycle_norm.insert(0, "extracted")

        if status in _CLAIM_TERMINAL_UNRESOLVED:
            pruned = dict(normalized)
            pruned["status"] = "pruned"
            pruned["reason_code"] = _CLAIM_REASON_CODES["pruned"]
            if "pruned" not in lifecycle_norm:
                lifecycle_norm.append("pruned")
            pruned["lifecycle"] = lifecycle_norm
            pruned_claims.append(pruned)
            if prune_mode == "rewrite_uncertainty":
                text = str(pruned.get("text", "") or "").strip()
                if text:
                    uncertainty_notes.append(
                        f"Uncertain claim excluded from synthesis: {text}",
                    )
            continue

        if status not in lifecycle_norm:
            lifecycle_norm.append(status)
        normalized["lifecycle"] = lifecycle_norm
        supported_claims.append(normalized)

    pruned_count = len(pruned_claims)
    updated_claims = supported_claims + pruned_claims
    counts = self._claim_counts(updated_claims)
    ratios = self._claim_ratios(counts)

    metadata = dict(verification.metadata) if isinstance(verification.metadata, dict) else {}
    metadata["claim_lifecycle_original"] = claims
    metadata["claim_lifecycle"] = updated_claims
    metadata["claim_pruned"] = bool(pruned_count > 0)
    metadata["claim_prune_mode"] = prune_mode
    metadata["claim_pruned_count"] = int(pruned_count)
    metadata["claim_status_counts"] = counts
    metadata["claim_reason_codes"] = sorted({
        self._normalize_claim_reason_code(
            status=str(item.get("status", "") or ""),
            reason_code=str(item.get("reason_code", "") or ""),
        )
        for item in updated_claims
    })
    metadata["supported_ratio"] = ratios["supported_ratio"]
    metadata["unverified_ratio"] = ratios["unverified_ratio"]
    min_supported_ratio = self._to_ratio(contract.get("min_supported_ratio", 0.75), 0.75)
    max_unverified_ratio = self._to_ratio(contract.get("max_unverified_ratio", 0.25), 0.25)
    max_contradicted = self._to_non_negative_int(
        contract.get("max_contradicted_count", 0),
        0,
    )
    metadata["claim_gate_thresholds"] = {
        "min_supported_ratio": min_supported_ratio,
        "max_unverified_ratio": max_unverified_ratio,
        "max_contradicted_count": max_contradicted,
    }
    post_prune_gate_passed = (
        int(counts.get("contradicted", 0) or 0) <= max_contradicted
        and float(ratios.get("supported_ratio", 0.0) or 0.0) >= min_supported_ratio
        and float(ratios.get("unverified_ratio", 0.0) or 0.0) <= max_unverified_ratio
    )
    metadata["post_prune_gate_passed"] = bool(post_prune_gate_passed)
    if uncertainty_notes:
        metadata["uncertainty_annotations"] = uncertainty_notes[:20]
        note = (
            "Unsupported or uncertain claims were rewritten as uncertainty "
            "annotations and excluded from downstream verified context."
        )
        result.summary = "\n".join(part for part in [result.summary, note] if part).strip()

    self._emit(CLAIMS_PRUNED, task.id, {
        "subtask_id": subtask.id,
        "phase_id": subtask.phase_id,
        "pruned_count": int(pruned_count),
        "supported_count": int(counts.get("supported", 0)),
        "contradicted_count": int(counts.get("contradicted", 0)),
        "insufficient_evidence_count": int(counts.get("insufficient_evidence", 0)),
        "prune_mode": prune_mode,
    })

    reason_code = str(verification.reason_code or "").strip().lower()
    if (
        not verification.passed
        and reason_code in _CLAIM_RECOVERABLE_FAILURE_CODES
        and post_prune_gate_passed
    ):
        note = (
            "Intermediate validity policy pruned unsupported claims and "
            "allowed execution to continue."
        )
        result.status = SubtaskResultStatus.SUCCESS
        return self._verification_with_metadata(
            verification,
            metadata=metadata,
            passed=True,
            outcome=(
                "partial_verified"
                if self._config.verification.allow_partial_verified
                else "pass_with_warnings"
            ),
            reason_code="claim_pruned",
            feedback="\n".join(part for part in [verification.feedback or "", note] if part),
            severity_class="semantic",
            confidence=min(0.8, max(0.3, float(verification.confidence or 0.5))),
        )
    if not post_prune_gate_passed:
        gate_reason = "coverage_below_threshold"
        if int(counts.get("contradicted", 0) or 0) > max_contradicted:
            gate_reason = "claim_contradicted"
        elif float(ratios.get("unverified_ratio", 0.0) or 0.0) > max_unverified_ratio:
            gate_reason = "claim_insufficient_evidence"
        threshold_note = (
            "Intermediate validity policy pruned unsupported claims, but "
            "post-prune coverage did not satisfy contract thresholds."
        )
        return self._verification_with_metadata(
            verification,
            metadata=metadata,
            passed=False,
            outcome="fail",
            reason_code=gate_reason,
            feedback="\n".join(
                part
                for part in [verification.feedback or "", threshold_note]
                if part
            ),
            severity_class="semantic",
            confidence=min(float(verification.confidence or 0.5), 0.45),
        )
    return self._verification_with_metadata(
        verification,
        metadata=metadata,
    )

def _parse_temporal_date_token(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if isinstance(value, datetime):
        return value
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
    if parsed is not None:
        return parsed

    ymd_match = re.search(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text)
    if ymd_match:
        try:
            return datetime.strptime(ymd_match.group(0).replace("/", "-"), "%Y-%m-%d")
        except ValueError:
            pass
    ym_match = re.search(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])\b", text)
    if ym_match:
        try:
            return datetime.strptime(ym_match.group(0).replace("/", "-"), "%Y-%m")
        except ValueError:
            pass
    year_match = re.search(r"\b(19|20)\d{2}\b", text)
    if year_match:
        try:
            return datetime.strptime(year_match.group(0), "%Y")
        except ValueError:
            return None
    return None

def _extract_temporal_dates_from_text(cls, text: str) -> list[datetime]:
    parsed: list[datetime] = []
    seen: set[str] = set()
    pattern = (
        r"\b(?:19|20)\d{2}"
        r"(?:[-/](?:0[1-9]|1[0-2])"
        r"(?:[-/](?:0[1-9]|[12]\d|3[01]))?)?\b"
    )
    for match in re.finditer(pattern, text):
        token = str(match.group(0) or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        parsed_date = cls._parse_temporal_date_token(token)
        if parsed_date is not None:
            parsed.append(parsed_date)
    return parsed

def _claim_temporal_scope(cls, claim: dict[str, object]) -> dict[str, object]:
    metadata = claim.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    source_scope = claim.get("source_time_scope", {})
    if not isinstance(source_scope, dict):
        source_scope = {}

    def _lookup(*keys: str) -> str:
        for key in keys:
            for container in (claim, metadata, source_scope):
                value = container.get(key)
                text = str(value or "").strip()
                if text:
                    return text
        return ""

    as_of_text = _lookup("as_of")
    source_as_of_text = _lookup("source_as_of", "evidence_as_of")
    period_start_text = _lookup("period_start")
    period_end_text = _lookup("period_end")
    text_dates = cls._extract_temporal_dates_from_text(str(claim.get("text", "") or ""))
    return {
        "as_of": cls._parse_temporal_date_token(as_of_text),
        "source_as_of": cls._parse_temporal_date_token(source_as_of_text),
        "period_start": cls._parse_temporal_date_token(period_start_text),
        "period_end": cls._parse_temporal_date_token(period_end_text),
        "text_dates": text_dates,
    }

def _temporal_claim_key(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return ""
    cleaned = re.sub(
        r"\b(?:19|20)\d{2}(?:[-/](?:0[1-9]|1[0-2])(?:[-/](?:0[1-9]|[12]\d|3[01]))?)?\b",
        " ",
        lowered,
    )
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\b", " ", cleaned)
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", cleaned)
        if len(token) >= 3
    ]
    if not tokens:
        return ""
    return " ".join(tokens[:8])

def _enforce_temporal_consistency_gate(
    self,
    *,
    subtask: Subtask,
    verification: VerificationResult,
    contract: dict[str, object],
) -> VerificationResult:
    if not subtask.is_synthesis:
        return verification
    claims = self._claims_from_verification(verification)
    if not claims:
        return verification

    final_gate = contract.get("final_gate", {})
    temporal: dict[str, object] = {}
    if isinstance(final_gate, dict):
        raw_temporal = final_gate.get("temporal_consistency", {})
        if isinstance(raw_temporal, dict):
            temporal = raw_temporal
    enabled = self._to_bool(temporal.get("enabled", False), False)
    if not enabled:
        return verification

    require_as_of_alignment = self._to_bool(
        temporal.get("require_as_of_alignment", False),
        False,
    )
    enforce_date_conflicts = self._to_bool(
        temporal.get("enforce_cross_claim_date_conflict_check", False),
        False,
    )
    max_source_age_days = self._to_non_negative_int(
        temporal.get("max_source_age_days", 0),
        0,
    )
    as_of_text = str(temporal.get("as_of", "") or "").strip()
    reference_dt = self._parse_temporal_date_token(as_of_text) or datetime.now()
    reference_date = reference_dt.date()

    as_of_values: set[str] = set()
    stale_claims: list[dict[str, object]] = []
    future_source_claims: list[str] = []
    conflict_index: dict[str, set[str]] = {}

    stale_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        status = str(claim.get("status", "") or "").strip().lower()
        if status != "supported":
            continue
        scope = self._claim_temporal_scope(claim)
        as_of_dt = scope.get("as_of")
        source_as_of_dt = scope.get("source_as_of")
        period_end_dt = scope.get("period_end")
        text_dates = scope.get("text_dates", [])
        if not isinstance(text_dates, list):
            text_dates = []

        if require_as_of_alignment and isinstance(as_of_dt, datetime):
            as_of_values.add(as_of_dt.date().isoformat())

        source_dt = (
            source_as_of_dt
            if isinstance(source_as_of_dt, datetime)
            else (
                as_of_dt
                if isinstance(as_of_dt, datetime)
                else (
                    period_end_dt
                    if isinstance(period_end_dt, datetime)
                    else (text_dates[0] if text_dates else None)
                )
            )
        )
        if isinstance(source_dt, datetime) and max_source_age_days > 0:
            age_days = (reference_date - source_dt.date()).days
            if age_days > max_source_age_days:
                claim_id = str(claim.get("claim_id", "") or "").strip()
                stale_ids.add(claim_id)
                stale_claims.append({
                    "claim_id": claim_id,
                    "text": str(claim.get("text", "") or "")[:200],
                    "age_days": int(age_days),
                    "source_date": source_dt.date().isoformat(),
                })
            elif age_days < -1:
                text = str(claim.get("text", "") or "").strip()
                if text:
                    future_source_claims.append(text[:160])

        if enforce_date_conflicts:
            key = self._temporal_claim_key(str(claim.get("text", "") or ""))
            if not key:
                continue
            anchor_dt = (
                as_of_dt
                if isinstance(as_of_dt, datetime)
                else (
                    period_end_dt
                    if isinstance(period_end_dt, datetime)
                    else (text_dates[0] if text_dates else None)
                )
            )
            if not isinstance(anchor_dt, datetime):
                continue
            conflict_index.setdefault(key, set()).add(anchor_dt.date().isoformat())

    temporal_conflicts: list[dict[str, object]] = []
    if require_as_of_alignment and len(as_of_values) > 1:
        temporal_conflicts.append({
            "kind": "as_of_misalignment",
            "observed_as_of_values": sorted(as_of_values),
        })
    if future_source_claims:
        temporal_conflicts.append({
            "kind": "future_source_date",
            "claims": future_source_claims[:10],
        })
    if enforce_date_conflicts:
        for key, values in conflict_index.items():
            if len(values) <= 1:
                continue
            temporal_conflicts.append({
                "kind": "cross_claim_date_conflict",
                "claim_key": key,
                "observed_dates": sorted(values),
            })

    updated_claims = [dict(item) for item in claims]
    if stale_ids:
        for claim in updated_claims:
            claim_id = str(claim.get("claim_id", "") or "").strip()
            if claim_id not in stale_ids:
                continue
            claim["status"] = "stale"
            claim["reason_code"] = "claim_stale_source"
            lifecycle = claim.get("lifecycle", [])
            if isinstance(lifecycle, str):
                lifecycle = [lifecycle]
            if not isinstance(lifecycle, list):
                lifecycle = []
            lifecycle_norm = [
                str(step or "").strip().lower()
                for step in lifecycle
                if str(step or "").strip()
            ]
            if "stale" not in lifecycle_norm:
                lifecycle_norm.append("stale")
            claim["lifecycle"] = lifecycle_norm

    metadata = dict(verification.metadata) if isinstance(verification.metadata, dict) else {}
    metadata["claim_lifecycle"] = updated_claims
    metadata["claim_status_counts"] = self._claim_counts(updated_claims)
    metadata["claim_reason_codes"] = sorted({
        self._normalize_claim_reason_code(
            status=str(item.get("status", "") or ""),
            reason_code=str(item.get("reason_code", "") or ""),
        )
        for item in updated_claims
    })
    metadata["temporal_consistency"] = {
        "enabled": True,
        "reference_as_of": reference_date.isoformat(),
        "require_as_of_alignment": require_as_of_alignment,
        "enforce_cross_claim_date_conflict_check": enforce_date_conflicts,
        "max_source_age_days": max_source_age_days,
        "stale_claim_count": len(stale_claims),
        "conflict_count": len(temporal_conflicts),
    }
    if stale_claims:
        metadata["stale_claims"] = stale_claims[:20]
    if temporal_conflicts:
        metadata["temporal_conflicts"] = temporal_conflicts[:20]

    if not stale_claims and not temporal_conflicts:
        return self._verification_with_metadata(
            verification,
            metadata=metadata,
        )

    fail_reason_code = "claim_stale_source" if stale_claims else "temporal_conflict"
    feedback_lines = [str(verification.feedback or "").strip()]
    if stale_claims:
        feedback_lines.append(
            "Temporal gate failed: stale source dates exceeded max_source_age_days.",
        )
    if temporal_conflicts:
        feedback_lines.append(
            "Temporal gate failed: as_of alignment or cross-claim date consistency violation.",
        )
    return self._verification_with_metadata(
        verification,
        metadata=metadata,
        passed=False,
        outcome="fail",
        reason_code=fail_reason_code,
        feedback="\n".join(line for line in feedback_lines if line),
        severity_class="semantic",
        confidence=min(float(verification.confidence or 0.5), 0.45),
    )

def _enforce_synthesis_claim_gate(
    self,
    *,
    subtask: Subtask,
    verification: VerificationResult,
    contract: dict[str, object],
) -> VerificationResult:
    if not subtask.is_synthesis:
        return verification
    claims = self._claims_from_verification(verification)
    if not claims:
        return verification

    counts = self._claim_counts(claims)
    ratios = self._claim_ratios(counts)
    min_supported_ratio = self._to_ratio(contract.get("min_supported_ratio", 0.75), 0.75)
    max_unverified_ratio = self._to_ratio(contract.get("max_unverified_ratio", 0.25), 0.25)
    max_contradicted = self._to_non_negative_int(
        contract.get("max_contradicted_count", 0),
        0,
    )
    final_gate = contract.get("final_gate", {})
    critical_support_floor = 1.0
    if isinstance(final_gate, dict):
        critical_support_floor = self._to_ratio(
            final_gate.get("critical_claim_support_ratio", 1.0),
            1.0,
        )

    fail_reasons: list[str] = []
    reason_code = ""
    if counts["contradicted"] > max_contradicted:
        fail_reasons.append(
            "Synthesis claim gate failed: contradicted claims exceed contract threshold.",
        )
        reason_code = "claim_contradicted"
    if counts["critical_supported"] < counts["critical_total"]:
        if counts["critical_contradicted"] > 0:
            fail_reasons.append(
                "Synthesis claim gate failed: contradicted critical claims detected.",
            )
            reason_code = "claim_contradicted"
        else:
            fail_reasons.append(
                "Synthesis claim gate failed: unsupported critical claims remain.",
            )
            if not reason_code:
                reason_code = "claim_insufficient_evidence"
    if ratios["critical_support_ratio"] < critical_support_floor:
        fail_reasons.append(
            "Synthesis claim gate failed: critical claim support ratio below threshold.",
        )
        if not reason_code:
            reason_code = "coverage_below_threshold"
    if ratios["supported_ratio"] < min_supported_ratio:
        fail_reasons.append(
            "Synthesis claim gate failed: supported-claim ratio below threshold.",
        )
        if not reason_code:
            reason_code = "coverage_below_threshold"
    if ratios["unverified_ratio"] > max_unverified_ratio:
        fail_reasons.append(
            "Synthesis claim gate failed: unresolved claim ratio above threshold.",
        )
        if not reason_code:
            reason_code = "claim_insufficient_evidence"

    orphan_critical_numeric_claims: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_type = str(claim.get("claim_type", "") or "").strip().lower()
        criticality = str(claim.get("criticality", "") or "").strip().lower()
        if claim_type != "numeric" or criticality != "critical":
            continue
        refs = claim.get("evidence_refs", [])
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            refs = []
        normalized_refs = [
            str(ref or "").strip()
            for ref in refs
            if str(ref or "").strip()
        ]
        if normalized_refs:
            continue
        claim_text = str(claim.get("text", "") or "").strip()
        orphan_critical_numeric_claims.append(claim_text[:160] if claim_text else "")
    if orphan_critical_numeric_claims:
        fail_reasons.append(
            "Synthesis claim gate failed: critical numeric claim missing evidence lineage.",
        )
        if not reason_code:
            reason_code = "claim_insufficient_evidence"

    metadata = dict(verification.metadata) if isinstance(verification.metadata, dict) else {}
    metadata["claim_status_counts"] = counts
    metadata["supported_ratio"] = ratios["supported_ratio"]
    metadata["unverified_ratio"] = ratios["unverified_ratio"]
    metadata["critical_support_ratio"] = ratios["critical_support_ratio"]
    if orphan_critical_numeric_claims:
        metadata["orphan_critical_numeric_claims"] = orphan_critical_numeric_claims[:10]
    metadata["claim_gate_thresholds"] = {
        "min_supported_ratio": min_supported_ratio,
        "max_unverified_ratio": max_unverified_ratio,
        "max_contradicted_count": max_contradicted,
        "critical_claim_support_ratio": critical_support_floor,
    }

    if not fail_reasons:
        return self._verification_with_metadata(verification, metadata=metadata)
    return self._verification_with_metadata(
        verification,
        metadata=metadata,
        passed=False,
        outcome="fail",
        reason_code=reason_code or "coverage_below_threshold",
        feedback="\n".join([
            *(part for part in [verification.feedback or ""] if part),
            *fail_reasons,
        ]),
        severity_class="semantic",
        confidence=min(float(verification.confidence or 0.5), 0.45),
    )

def _artifact_provenance_evidence(
    *,
    task_id: str,
    subtask_id: str,
    tool_calls: list[ToolCallRecord] | None,
    existing_ids: set[str],
    workspace: Path | None = None,
) -> list[dict[str, object]]:
    def _fallback_artifact_content(
        tool_name: str,
        args: dict[str, object],
        data: dict[str, object],
    ) -> str:
        if tool_name == "write_file":
            return str(args.get("content", "") or "")
        if tool_name == "document_write":
            parts: list[str] = []
            title = str(args.get("title", "") or "").strip()
            if title:
                parts.append(title)
            body = str(args.get("content", "") or "")
            if body:
                parts.append(body)
            sections = args.get("sections", [])
            if isinstance(sections, list):
                for section in sections[:8]:
                    if not isinstance(section, dict):
                        continue
                    heading = str(section.get("heading", "") or "").strip()
                    if heading:
                        parts.append(heading)
                    section_body = str(section.get("body", "") or "")
                    if section_body:
                        parts.append(section_body)
            if parts:
                return "\n\n".join(parts)
            return str(data.get("content", "") or "")
        return ""

    def _read_artifact_bytes(workspace_path: Path | None, relpath: str) -> bytes:
        if workspace_path is None:
            return b""
        normalized = normalize_workspace_relpath(workspace_path, relpath)
        if not normalized:
            return b""
        try:
            artifact_path = (workspace_path / normalized).resolve()
            artifact_path.relative_to(workspace_path)
        except Exception:
            return b""
        if not artifact_path.exists() or not artifact_path.is_file():
            return b""
        try:
            return artifact_path.read_bytes()
        except Exception:
            return b""

    records: list[dict[str, object]] = []
    workspace_path = workspace if isinstance(workspace, Path) else None
    for call in tool_calls or []:
        tool = str(getattr(call, "tool", "") or "").strip().lower()
        result = getattr(call, "result", None)
        if result is None or not bool(getattr(result, "success", False)):
            continue
        args = getattr(call, "args", {})
        if not isinstance(args, dict):
            args = {}
        data = getattr(result, "data", {})
        if not isinstance(data, dict):
            data = {}
        relpaths: list[str] = []
        seen_paths: set[str] = set()
        for raw in list(getattr(result, "files_changed", []) or []):
            relpath = str(raw or "").strip()
            if not relpath:
                continue
            if workspace_path is not None:
                normalized = normalize_workspace_relpath(workspace_path, relpath)
                if normalized:
                    relpath = normalized
            if relpath in seen_paths:
                continue
            seen_paths.add(relpath)
            relpaths.append(relpath)
        if not relpaths:
            fallback = str(
                args.get("path")
                or args.get("file_path")
                or data.get("path")
                or "",
            ).strip()
            if fallback:
                relpaths = [fallback]
        if not relpaths:
            continue
        fallback_content = _fallback_artifact_content(tool, args, data)
        for relpath in relpaths:
            payload_bytes = _read_artifact_bytes(workspace_path, relpath)
            if not payload_bytes and fallback_content and len(relpaths) == 1:
                payload_bytes = fallback_content.encode("utf-8", errors="replace")
            sha256 = hashlib.sha256(payload_bytes).hexdigest() if payload_bytes else ""
            size_bytes = len(payload_bytes)
            payload = f"{tool}|{subtask_id}|{relpath}|{sha256 or fallback_content[:120]}"
            evidence_id = "EV-ART-" + hashlib.sha1(
                payload.encode("utf-8", errors="replace"),
            ).hexdigest().upper()[:10]
            if evidence_id in existing_ids:
                continue
            existing_ids.add(evidence_id)
            records.append({
                "evidence_id": evidence_id,
                "task_id": task_id,
                "subtask_id": subtask_id,
                "tool": tool,
                "evidence_kind": "artifact",
                "tool_call_id": str(getattr(call, "call_id", "") or ""),
                "query": relpath,
                "source_url": "",
                "facets": {"artifact_path": relpath[:120]},
                "artifact_workspace_relpath": relpath,
                "artifact_sha256": sha256,
                "artifact_size_bytes": int(size_bytes),
                "snippet": f"{tool}:{relpath}",
                "context_text": f"{tool} mutated {relpath}",
                "quality": 1.0,
                "created_at": str(getattr(call, "timestamp", "") or datetime.now().isoformat()),
            })
    return records

def _claim_evidence_links(
    *,
    claims: list[dict[str, object]],
    evidence_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    evidence_ids: list[str] = []
    source_index: dict[str, str] = {}
    artifact_index: dict[str, str] = {}
    for record in evidence_records:
        if not isinstance(record, dict):
            continue
        evidence_id = str(record.get("evidence_id", "") or "").strip()
        if not evidence_id:
            continue
        evidence_ids.append(evidence_id)
        source_url = str(record.get("source_url", "") or "").strip()
        if source_url and source_url not in source_index:
            source_index[source_url] = evidence_id
        artifact_path = str(record.get("artifact_workspace_relpath", "") or "").strip()
        if artifact_path and artifact_path not in artifact_index:
            artifact_index[artifact_path] = evidence_id
        facets = record.get("facets", {})
        if isinstance(facets, dict):
            facet_path = str(facets.get("artifact_path", "") or "").strip()
            if facet_path and facet_path not in artifact_index:
                artifact_index[facet_path] = evidence_id

    links: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id", "") or "").strip()
        if not claim_id:
            continue
        refs = claim.get("evidence_refs", [])
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            refs = []
        matched_ids: list[str] = []
        for ref in refs:
            ref_text = str(ref or "").strip()
            if not ref_text:
                continue
            if ref_text in evidence_ids:
                matched_ids.append(ref_text)
                continue
            matched_source = False
            for source_url, evidence_id in source_index.items():
                if ref_text == source_url or ref_text in source_url or source_url in ref_text:
                    matched_ids.append(evidence_id)
                    matched_source = True
                    break
            if matched_source:
                continue
            for artifact_path, evidence_id in artifact_index.items():
                if (
                    ref_text == artifact_path
                    or ref_text.endswith(artifact_path)
                    or artifact_path.endswith(ref_text)
                ):
                    matched_ids.append(evidence_id)
                    break
        for evidence_id in matched_ids:
            key = (claim_id, evidence_id)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            links.append({
                "claim_id": claim_id,
                "evidence_id": evidence_id,
                "link_type": "supporting",
                "score": 1.0,
                "metadata": {},
            })
    return links

async def _persist_claim_validity_artifacts(
    self,
    *,
    task: Task,
    subtask: Subtask,
    verification: VerificationResult,
    evidence_records: list[dict],
    tool_calls: list[ToolCallRecord] | None = None,
) -> None:
    claims = self._claims_from_verification(verification)
    if not claims:
        return
    normalized_evidence = [
        item for item in evidence_records if isinstance(item, dict)
    ]
    existing_ids = {
        str(item.get("evidence_id", "") or "").strip()
        for item in normalized_evidence
        if str(item.get("evidence_id", "") or "").strip()
    }
    workspace_path: Path | None = None
    if task.workspace:
        try:
            workspace_path = Path(task.workspace).expanduser().resolve()
        except Exception:
            workspace_path = None
    provenance_records = self._artifact_provenance_evidence(
        task_id=task.id,
        subtask_id=subtask.id,
        tool_calls=tool_calls,
        existing_ids=existing_ids,
        workspace=workspace_path,
    )
    if provenance_records:
        normalized_evidence = merge_evidence_records(
            normalized_evidence,
            provenance_records,
        )
    counts = self._claim_counts(claims)
    ratios = self._claim_ratios(counts)
    run_id = self._task_run_id(task)
    phase_id = str(getattr(subtask, "phase_id", "") or "")
    links = self._claim_evidence_links(
        claims=claims,
        evidence_records=normalized_evidence,
    )
    claim_results = []
    for claim in claims:
        status = str(claim.get("status", "extracted") or "extracted").strip().lower()
        claim_results.append({
            "claim_id": str(claim.get("claim_id", "") or "").strip(),
            "status": status,
            "reason_code": self._normalize_claim_reason_code(
                status=status,
                reason_code=str(claim.get("reason_code", "") or ""),
            ),
            "verifier": "verification_gates",
            "confidence": float(verification.confidence or 0.0),
            "metadata": {
                "claim_type": str(claim.get("claim_type", "qualitative") or "qualitative"),
                "criticality": str(claim.get("criticality", "important") or "important"),
            },
        })
    try:
        await self._memory.insert_artifact_claims(
            task_id=task.id,
            run_id=run_id,
            subtask_id=subtask.id,
            phase_id=phase_id,
            claims=claims,
        )
        await self._memory.insert_claim_verification_results(
            task_id=task.id,
            run_id=run_id,
            subtask_id=subtask.id,
            phase_id=phase_id,
            results=claim_results,
        )
        if links:
            await self._memory.insert_claim_evidence_links(
                task_id=task.id,
                run_id=run_id,
                subtask_id=subtask.id,
                links=links,
            )
        await self._memory.insert_artifact_validity_summary(
            task_id=task.id,
            run_id=run_id,
            subtask_id=subtask.id,
            phase_id=phase_id,
            extracted_count=int(counts["extracted"]),
            supported_count=int(counts["supported"]),
            contradicted_count=int(counts["contradicted"]),
            insufficient_evidence_count=int(counts["insufficient_evidence"]),
            pruned_count=int(counts["pruned"]),
            supported_ratio=float(ratios["supported_ratio"]),
            gate_decision="pass" if verification.passed else "fail",
            reason_code=str(verification.reason_code or ""),
            metadata={
                "critical_total": int(counts["critical_total"]),
                "critical_supported": int(counts["critical_supported"]),
                "critical_support_ratio": float(ratios["critical_support_ratio"]),
                "validity_contract_hash": str(
                    getattr(subtask, "validity_contract_hash", "") or "",
                ),
            },
        )
    except Exception:
        logger.debug(
            "Failed persisting claim validity artifacts for %s/%s",
            task.id,
            subtask.id,
            exc_info=True,
        )

def _verified_context_for_synthesis(
    self,
    *,
    task: Task,
    subtask: Subtask,
) -> tuple[bool, str, str]:
    contract = self._validity_contract_for_subtask(subtask)
    claim_extraction = contract.get("claim_extraction", {})
    claim_extraction_enabled = isinstance(claim_extraction, dict) and self._to_bool(
        claim_extraction.get("enabled", False),
        False,
    )
    final_gate = contract.get("final_gate", {})
    enforce_verified_context = isinstance(final_gate, dict) and self._to_bool(
        final_gate.get("enforce_verified_context_only", False),
        False,
    )
    if not (subtask.is_synthesis and claim_extraction_enabled and enforce_verified_context):
        return True, "", ""

    graph = self._claim_graph_state(task)
    supported_by_subtask = graph.get("supported_by_subtask", {})
    if not isinstance(supported_by_subtask, dict):
        supported_by_subtask = {}
    unresolved_by_subtask = graph.get("unresolved_by_subtask", {})
    if not isinstance(unresolved_by_subtask, dict):
        unresolved_by_subtask = {}

    lines: list[str] = []
    total_supported = 0
    for subtask_id, claims in supported_by_subtask.items():
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = str(claim.get("text", "") or "").strip()
            if not text:
                continue
            total_supported += 1
            lines.append(f"- [{subtask_id}] {text}")
    total_unresolved = sum(
        len(claims)
        for claims in unresolved_by_subtask.values()
        if isinstance(claims, list)
    )
    if total_supported <= 0 and total_unresolved <= 0:
        return True, "", ""
    if total_supported <= 0:
        return (
            False,
            "",
            "Synthesis gate blocked: no supported claims available in verified context bundle.",
        )
    bundle = "\n".join(lines[:80]).strip()
    return True, bundle, ""

def _enforce_required_fact_checker(
    self,
    *,
    subtask: Subtask,
    result: SubtaskResult,
    verification: VerificationResult,
) -> VerificationResult:
    if not self._requires_fact_checker_for_subtask(subtask):
        return verification
    tool_calls = list(result.tool_calls or [])
    if not self._fact_checker_used(tool_calls):
        result.status = SubtaskResultStatus.FAILED
        metadata = (
            dict(verification.metadata)
            if isinstance(verification.metadata, dict)
            else {}
        )
        metadata["required_tool"] = "fact_checker"
        metadata["required_verifier_missing"] = True
        details = (
            "Synthesis requires fact grounding, but no successful "
            "`fact_checker` invocation was observed."
        )
        return VerificationResult(
            tier=max(verification.tier, int(subtask.verification_tier or 1)),
            passed=False,
            confidence=min(verification.confidence, 0.3),
            checks=list(verification.checks or []),
            feedback=details,
            outcome="fail",
            reason_code="required_verifier_missing",
            severity_class="semantic",
            metadata=metadata,
        )

    contract = self._validity_contract_for_subtask(subtask)
    claim_extraction = contract.get("claim_extraction", {})
    claim_extraction_enabled = isinstance(claim_extraction, dict) and self._to_bool(
        claim_extraction.get("enabled", False),
        False,
    )
    verdict_count = self._fact_checker_verdict_count(tool_calls)
    if claim_extraction_enabled and verdict_count <= 0:
        result.status = SubtaskResultStatus.FAILED
        metadata = (
            dict(verification.metadata)
            if isinstance(verification.metadata, dict)
            else {}
        )
        metadata["required_tool"] = "fact_checker"
        metadata["required_verifier_empty"] = True
        metadata["fact_checker_verdict_count"] = 0
        details = (
            "Synthesis requires claim-level fact grounding, but `fact_checker` "
            "returned no claim verdicts."
        )
        return VerificationResult(
            tier=max(verification.tier, int(subtask.verification_tier or 1)),
            passed=False,
            confidence=min(verification.confidence, 0.3),
            checks=list(verification.checks or []),
            feedback=details,
            outcome="fail",
            reason_code="required_verifier_empty",
            severity_class="semantic",
            metadata=metadata,
        )

    return verification
