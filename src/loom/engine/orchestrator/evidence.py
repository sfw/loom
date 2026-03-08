"""Evidence-ledger formatting helpers for orchestrator exports."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from loom.engine.path_policy import normalize_path_for_policy
from loom.engine.runner import ToolCallRecord
from loom.engine.verification import VerificationResult
from loom.events.types import RUN_VALIDITY_SCORECARD
from loom.state.evidence import merge_evidence_records
from loom.state.task_state import Subtask, Task, TaskStatus

logger = logging.getLogger(__name__)


def stringify_evidence_csv_value(value: object) -> str:
    """Convert evidence values to stable CSV-safe text."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def evidence_csv_fieldnames(
    *,
    base_fields: tuple[str, ...] | list[str],
    rows: list[dict[str, str]],
) -> list[str]:
    """Build deterministic CSV fieldnames from base + observed keys."""
    base = list(base_fields)
    extras: set[str] = set()
    for row in rows:
        for key in row:
            if key and key not in base:
                extras.add(key)
    return base + sorted(extras)


def evidence_csv_rows(records: list[dict]) -> list[dict[str, str]]:
    """Normalize evidence records into flat string maps for CSV export."""
    rows: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        row: dict[str, str] = {}
        for raw_key, value in record.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            row[key] = stringify_evidence_csv_value(value)
        if row:
            rows.append(row)
    return rows


# Extracted evidence/artifact/scorecard orchestration helpers

def _evidence_for_subtask(self, task_id: str, subtask_id: str) -> list[dict]:
    """Load persisted evidence records scoped to one subtask."""
    try:
        records = self._state.load_evidence_records(task_id)
    except Exception as e:
        logger.warning("Failed loading evidence ledger for %s: %s", task_id, e)
        return []
    scoped: list[dict] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        if str(item.get("subtask_id", "")).strip() != subtask_id:
            continue
        scoped.append(item)
    return scoped

def _persist_subtask_evidence(
    self,
    task_id: str,
    subtask_id: str,
    evidence_records: list[dict] | None,
    *,
    tool_calls: list[ToolCallRecord] | None = None,
    workspace: str | Path | None = None,
) -> None:
    """Persist newly captured evidence records."""
    scoped: list[dict] = []
    if evidence_records:
        for item in evidence_records:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized["subtask_id"] = subtask_id
            normalized.setdefault("task_id", task_id)
            scoped.append(normalized)
    existing_ids: set[str] = {
        str(item.get("evidence_id", "") or "").strip()
        for item in scoped
        if isinstance(item, dict) and str(item.get("evidence_id", "") or "").strip()
    }
    workspace_path: Path | None = None
    if workspace:
        try:
            workspace_path = Path(str(workspace)).expanduser().resolve()
        except Exception:
            workspace_path = None
    provenance_records = self._artifact_provenance_evidence(
        task_id=task_id,
        subtask_id=subtask_id,
        tool_calls=tool_calls,
        existing_ids=existing_ids,
        workspace=workspace_path,
    )
    if provenance_records:
        scoped = merge_evidence_records(scoped, provenance_records)
    if not scoped:
        return
    try:
        self._state.append_evidence_records(task_id, scoped)
    except Exception as e:
        logger.warning("Failed persisting evidence ledger for %s: %s", task_id, e)

def _artifact_content_for_call(
    tool_name: str,
    args: dict[str, object],
    result_data: dict[str, object],
) -> str:
    if tool_name == "write_file":
        return str(args.get("content", "") or "")
    if tool_name == "document_write":
        parts: list[str] = []
        title = str(args.get("title", "") or "").strip()
        if title:
            parts.append(title)
        content = str(args.get("content", "") or "")
        if content:
            parts.append(content)
        sections = args.get("sections", [])
        if isinstance(sections, list):
            for section in sections[:8]:
                if not isinstance(section, dict):
                    continue
                heading = str(section.get("heading", "") or "").strip()
                body = str(section.get("body", "") or "")
                if heading:
                    parts.append(heading)
                if body:
                    parts.append(body)
        if parts:
            return "\n\n".join(parts)
        return str(result_data.get("content", "") or "")
    return ""

def _artifact_seal_registry(self, task: Task) -> dict[str, dict[str, object]]:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    registry = metadata.get("artifact_seals")
    if not isinstance(registry, dict):
        registry = {}
        metadata["artifact_seals"] = registry
    task.metadata = metadata
    return registry


def _changed_workspace_paths_for_call(call: ToolCallRecord) -> list[str]:
    result = getattr(call, "result", None)
    if result is None or not bool(getattr(result, "success", False)):
        return []
    changed = list(getattr(result, "files_changed", []) or [])
    relpaths: list[str] = []
    seen: set[str] = set()
    for raw in changed:
        relpath = str(raw or "").strip()
        if not relpath or relpath in seen:
            continue
        seen.add(relpath)
        relpaths.append(relpath)
    return relpaths


def _record_artifact_seals(
    self,
    *,
    task: Task,
    subtask_id: str,
    tool_calls: list[ToolCallRecord] | None,
) -> int:
    if not tool_calls:
        return 0
    workspace_text = str(task.workspace or "").strip()
    if not workspace_text:
        return 0
    try:
        workspace = Path(workspace_text).expanduser().resolve()
    except Exception:
        return 0
    if not workspace.exists():
        return 0

    seals = self._artifact_seal_registry(task)
    updated = 0
    for call in tool_calls:
        tool_name = str(getattr(call, "tool", "") or "").strip().lower()
        result = getattr(call, "result", None)
        if result is None or not bool(getattr(result, "success", False)):
            continue
        relpaths = _changed_workspace_paths_for_call(call)
        if not relpaths:
            continue
        args = getattr(call, "args", {})
        if not isinstance(args, dict):
            args = {}
        result_data = getattr(result, "data", {})
        if not isinstance(result_data, dict):
            result_data = {}
        fallback_content = self._artifact_content_for_call(tool_name, args, result_data)
        for relpath in relpaths:
            try:
                normalized = normalize_path_for_policy(relpath, workspace)
            except Exception:
                normalized = relpath
            if not normalized:
                continue
            if self._is_intermediate_artifact_path(task=task, relpath=normalized):
                # Intermediate fan-in artifacts are not part of the canonical seal set.
                continue
            try:
                resolved = (workspace / normalized).resolve()
                resolved.relative_to(workspace)
            except Exception:
                continue

            sha256 = ""
            size_bytes = 0
            if resolved.exists() and resolved.is_file():
                try:
                    payload = resolved.read_bytes()
                except Exception:
                    payload = b""
                if payload:
                    size_bytes = len(payload)
                    sha256 = hashlib.sha256(payload).hexdigest()
            if not sha256 and fallback_content and len(relpaths) == 1:
                payload = fallback_content.encode("utf-8", errors="replace")
                size_bytes = len(payload)
                sha256 = hashlib.sha256(payload).hexdigest()
            if not sha256:
                continue

            seals[normalized] = {
                "path": normalized,
                "sha256": sha256,
                "size_bytes": int(size_bytes),
                "tool": tool_name,
                "tool_call_id": str(getattr(call, "call_id", "") or ""),
                "subtask_id": subtask_id,
                "run_id": self._task_run_id(task),
                "sealed_at": datetime.now().isoformat(),
            }
            updated += 1

    if updated > 0:
        task.metadata["artifact_seals"] = seals
    return updated

def _is_intermediate_artifact_path(self, *, task: Task, relpath: str) -> bool:
    workspace = Path(task.workspace) if task.workspace else None
    if workspace is None:
        return False
    normalized = normalize_path_for_policy(str(relpath), workspace)
    if not normalized:
        return False
    intermediate_root = normalize_path_for_policy(
        self._output_intermediate_root(),
        workspace,
    )
    if not intermediate_root:
        return False
    return normalized == intermediate_root or normalized.startswith(intermediate_root + "/")

def _validate_artifact_seals(
    self,
    *,
    task: Task,
) -> tuple[bool, list[dict[str, object]], int]:
    seals = self._artifact_seal_registry(task)
    if not seals:
        self._backfill_artifact_seals_from_evidence(task)
        seals = self._artifact_seal_registry(task)
    if not seals:
        return True, [], 0

    workspace_text = str(task.workspace or "").strip()
    if not workspace_text:
        return True, [], 0
    try:
        workspace = Path(workspace_text).expanduser().resolve()
    except Exception:
        return True, [], 0
    if not workspace.exists():
        return True, [], 0

    mismatches: list[dict[str, object]] = []
    validated = 0
    for relpath, seal in seals.items():
        if not isinstance(seal, dict):
            continue
        if self._is_intermediate_artifact_path(task=task, relpath=str(relpath)):
            continue
        expected = str(seal.get("sha256", "") or "").strip()
        if not expected:
            continue
        try:
            artifact_path = (workspace / str(relpath)).resolve()
            artifact_path.relative_to(workspace)
        except Exception:
            mismatches.append({
                "path": str(relpath),
                "reason": "path_outside_workspace",
            })
            continue
        validated += 1
        if not artifact_path.exists() or not artifact_path.is_file():
            mismatches.append({
                "path": str(relpath),
                "reason": "artifact_missing",
            })
            continue
        try:
            observed = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        except Exception:
            mismatches.append({
                "path": str(relpath),
                "reason": "artifact_unreadable",
            })
            continue
        if observed != expected:
            mismatches.append({
                "path": str(relpath),
                "reason": "artifact_seal_mismatch",
                "expected_sha256": expected,
                "observed_sha256": observed,
            })
    return len(mismatches) == 0, mismatches, validated

def _backfill_artifact_seals_from_evidence(self, task: Task) -> int:
    seals = self._artifact_seal_registry(task)
    if seals:
        return 0
    try:
        records = self._state.load_evidence_records(task.id)
    except Exception:
        return 0
    latest_by_path: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        relpath = str(record.get("artifact_workspace_relpath", "") or "").strip()
        sha256 = str(record.get("artifact_sha256", "") or "").strip()
        if not relpath or not sha256:
            continue
        if self._is_intermediate_artifact_path(task=task, relpath=relpath):
            continue
        current = latest_by_path.get(relpath)
        if current is None:
            latest_by_path[relpath] = record
            continue
        current_ts = str(current.get("created_at", "") or "")
        record_ts = str(record.get("created_at", "") or "")
        if record_ts >= current_ts:
            latest_by_path[relpath] = record

    if not latest_by_path:
        return 0

    for relpath, record in latest_by_path.items():
        seals[relpath] = {
            "path": relpath,
            "sha256": str(record.get("artifact_sha256", "") or ""),
            "size_bytes": int(record.get("artifact_size_bytes", 0) or 0),
            "tool": str(record.get("tool", "") or ""),
            "tool_call_id": str(record.get("tool_call_id", "") or ""),
            "subtask_id": str(record.get("subtask_id", "") or ""),
            "run_id": self._task_run_id(task),
            "sealed_at": str(record.get("created_at", "") or ""),
            "backfilled_from_evidence": True,
        }
    task.metadata["artifact_seals"] = seals
    return len(latest_by_path)

def _validity_scorecard_state(self, task: Task) -> dict[str, object]:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    scorecard = metadata.get("validity_scorecard")
    if not isinstance(scorecard, dict):
        scorecard = {}
    per_subtask = scorecard.get("subtask_metrics")
    if not isinstance(per_subtask, dict):
        per_subtask = {}
    scorecard["subtask_metrics"] = per_subtask
    metadata["validity_scorecard"] = scorecard
    task.metadata = metadata
    return scorecard

def _record_subtask_validity_metrics(
    self,
    *,
    task: Task,
    subtask: Subtask,
    verification: VerificationResult,
) -> None:
    if verification is None:
        return
    scorecard = self._validity_scorecard_state(task)
    per_subtask = scorecard.get("subtask_metrics", {})
    if not isinstance(per_subtask, dict):
        per_subtask = {}
        scorecard["subtask_metrics"] = per_subtask

    metadata = verification.metadata if isinstance(verification.metadata, dict) else {}
    counts = metadata.get("claim_status_counts")
    if not isinstance(counts, dict):
        counts = self._claim_counts(self._claims_from_verification(verification))
    counts = {
        "extracted": int(counts.get("extracted", 0) or 0),
        "supported": int(counts.get("supported", 0) or 0),
        "contradicted": int(counts.get("contradicted", 0) or 0),
        "insufficient_evidence": int(counts.get("insufficient_evidence", 0) or 0),
        "stale": int(counts.get("stale", 0) or 0),
        "pruned": int(counts.get("pruned", 0) or 0),
        "unresolved": int(
            counts.get(
                "unresolved",
                int(counts.get("contradicted", 0) or 0)
                + int(counts.get("insufficient_evidence", 0) or 0)
                + int(counts.get("stale", 0) or 0),
            ) or 0,
        ),
        "critical_total": int(counts.get("critical_total", 0) or 0),
        "critical_supported": int(counts.get("critical_supported", 0) or 0),
        "critical_contradicted": int(counts.get("critical_contradicted", 0) or 0),
    }
    ratios = self._claim_ratios(counts)
    reason_codes = metadata.get("claim_reason_codes")
    if not isinstance(reason_codes, list):
        reason_codes = []
    normalized_reason_codes = sorted({
        str(item or "").strip().lower()
        for item in reason_codes
        if str(item or "").strip()
    })
    per_subtask[subtask.id] = {
        "subtask_id": subtask.id,
        "phase_id": str(subtask.phase_id or ""),
        "is_synthesis": bool(subtask.is_synthesis),
        "verification_outcome": str(verification.outcome or ""),
        "reason_code": str(verification.reason_code or "").strip().lower(),
        "counts": counts,
        "ratios": {
            "supported_ratio": float(ratios.get("supported_ratio", 0.0)),
            "unverified_ratio": float(ratios.get("unverified_ratio", 0.0)),
            "critical_support_ratio": float(ratios.get("critical_support_ratio", 0.0)),
        },
        "reason_codes": normalized_reason_codes,
        "updated_at": datetime.now().isoformat(),
    }
    scorecard["run"] = self._build_run_validity_scorecard(task)

def _scorecard_source_window(self, task: Task) -> dict[str, str]:
    try:
        records = self._state.load_evidence_records(task.id)
    except Exception:
        return {"min": "", "max": ""}
    timestamps = sorted({
        str(record.get("created_at", "") or "").strip()
        for record in records
        if isinstance(record, dict) and str(record.get("created_at", "") or "").strip()
    })
    if not timestamps:
        return {"min": "", "max": ""}
    return {"min": timestamps[0], "max": timestamps[-1]}

def _build_run_validity_scorecard(self, task: Task) -> dict[str, object]:
    scorecard = self._validity_scorecard_state(task)
    per_subtask = scorecard.get("subtask_metrics", {})
    if not isinstance(per_subtask, dict):
        per_subtask = {}
    aggregate = {
        "extracted": 0,
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
    reason_codes: set[str] = set()
    for entry in per_subtask.values():
        if not isinstance(entry, dict):
            continue
        counts = entry.get("counts", {})
        if not isinstance(counts, dict):
            continue
        for key in aggregate:
            aggregate[key] += int(counts.get(key, 0) or 0)
        raw_codes = entry.get("reason_codes", [])
        if isinstance(raw_codes, list):
            reason_codes.update(
                str(item or "").strip().lower()
                for item in raw_codes
                if str(item or "").strip()
            )
        entry_reason = str(entry.get("reason_code", "") or "").strip().lower()
        if entry_reason:
            reason_codes.add(entry_reason)

    ratios = self._claim_ratios(aggregate)
    extracted = max(0, int(aggregate.get("extracted", 0) or 0))
    contradicted = max(0, int(aggregate.get("contradicted", 0) or 0))
    contradicted_ratio = (float(contradicted) / float(extracted)) if extracted > 0 else 0.0
    trust_score = max(
        0.0,
        min(
            1.0,
            float(ratios.get("supported_ratio", 0.0))
            - (0.6 * float(ratios.get("unverified_ratio", 0.0)))
            - (0.9 * contradicted_ratio),
        ),
    )
    source_window = self._scorecard_source_window(task)
    return {
        "analysis_timestamp": datetime.now().isoformat(),
        "source_time_window": source_window,
        "counts": aggregate,
        "supported_ratio": round(float(ratios.get("supported_ratio", 0.0)), 4),
        "unverified_ratio": round(float(ratios.get("unverified_ratio", 0.0)), 4),
        "critical_support_ratio": round(
            float(ratios.get("critical_support_ratio", 0.0)),
            4,
        ),
        "trust_score": round(trust_score, 4),
        "reason_codes": sorted(reason_codes),
        "verification_report_path": self._VALIDITY_SCORECARD_JSON_NAME,
    }

def _refresh_run_validity_scorecard(self, task: Task) -> dict[str, object]:
    scorecard = self._validity_scorecard_state(task)
    run_summary = self._build_run_validity_scorecard(task)
    scorecard["run"] = run_summary
    task.metadata["validity_scorecard"] = scorecard
    return run_summary

def _export_validity_scorecard_json(self, task: Task) -> None:
    workspace_text = str(task.workspace or "").strip()
    if not workspace_text:
        return
    workspace = Path(workspace_text).expanduser()
    if not workspace.exists() or not workspace.is_dir():
        return
    run_summary = self._refresh_run_validity_scorecard(task)
    output = workspace / self._VALIDITY_SCORECARD_JSON_NAME
    payload = {
        "task_id": task.id,
        "run_id": self._task_run_id(task),
        "status": str(
            task.status.value if isinstance(task.status, TaskStatus) else task.status,
        ),
        "summary": run_summary,
    }
    try:
        output.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Failed exporting validity scorecard for %s: %s", task.id, e)

def _emit_run_validity_scorecard(self, task: Task) -> None:
    run_summary = self._refresh_run_validity_scorecard(task)
    self._emit(RUN_VALIDITY_SCORECARD, task.id, {
        "run_id": self._task_run_id(task),
        **run_summary,
    })

def _append_synthesis_provenance_footer(
    self,
    *,
    task: Task,
    summary: str,
) -> str:
    base = str(summary or "").strip()
    marker = "VALIDITY_PROVENANCE_FOOTER:"
    if marker in base:
        return base
    run_summary = self._refresh_run_validity_scorecard(task)
    source_window = run_summary.get("source_time_window", {})
    if not isinstance(source_window, dict):
        source_window = {}
    footer_lines = [
        marker,
        f"analysis_timestamp={run_summary.get('analysis_timestamp', '')}",
        f"source_time_window={source_window.get('min', '')}..{source_window.get('max', '')}",
        f"supported_ratio={run_summary.get('supported_ratio', 0.0)}",
        f"critical_support_ratio={run_summary.get('critical_support_ratio', 0.0)}",
        f"trust_score={run_summary.get('trust_score', 0.0)}",
        "verification_report="
        + str(
            run_summary.get(
                "verification_report_path",
                self._VALIDITY_SCORECARD_JSON_NAME,
            ),
        ),
    ]
    footer = "\n".join(footer_lines).strip()
    if not base:
        return footer
    return f"{base}\n\n{footer}"

def _export_evidence_ledger_csv(self, task: Task) -> None:
    """Best-effort evidence ledger export to the task workspace."""
    workspace_text = str(task.workspace or "").strip()
    if not workspace_text:
        return
    workspace = Path(workspace_text).expanduser()
    if not workspace.exists() or not workspace.is_dir():
        return
    try:
        records = self._state.load_evidence_records(task.id)
    except Exception as e:
        logger.warning("Failed loading evidence ledger for CSV export %s: %s", task.id, e)
        return
    rows = self._evidence_csv_rows(records)
    if not rows:
        return
    output_path = workspace / self._EVIDENCE_LEDGER_CSV_NAME
    fieldnames = self._evidence_csv_fieldnames(rows)
    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        logger.warning(
            "Failed exporting evidence ledger CSV for %s to %s: %s",
            task.id,
            output_path,
            e,
        )
