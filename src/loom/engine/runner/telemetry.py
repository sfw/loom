"""Telemetry and event payload helpers for runner."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from loom.events.types import (
    ARTIFACT_INGEST_CLASSIFIED,
    ARTIFACT_INGEST_COMPLETED,
    ARTIFACT_READ_COMPLETED,
    ARTIFACT_RETENTION_PRUNED,
    COMPACTION_POLICY_DECISION,
    OVERFLOW_FALLBACK_APPLIED,
)
from loom.tools.registry import ToolResult


def new_subtask_telemetry_counters() -> dict[str, int]:
    return {
        "model_invocations": 0,
        "tool_calls": 0,
        "mutating_tool_calls": 0,
        "artifact_ingests": 0,
        "artifact_reads": 0,
        "artifact_retention_deletes": 0,
        "compaction_policy_decisions": 0,
        "overflow_fallback_count": 0,
        "compactor_warning_count": 0,
    }


def safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_reason_code(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "unspecified"
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized or "unspecified"


def telemetry_events_enabled(runner: Any) -> bool:
    return bool(getattr(runner, "_enable_artifact_telemetry_events", False))


def increment_subtask_counter(runner: Any, key: str, amount: int = 1) -> None:
    counters = getattr(runner, "_active_subtask_telemetry_counters", None)
    if not isinstance(counters, dict):
        return
    step = max(0, safe_int(amount))
    if step <= 0:
        return
    counters[key] = safe_int(counters.get(key, 0)) + step


def sanitize_url_for_telemetry(raw_url: Any) -> str:
    text = str(raw_url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        netloc = host or parsed.netloc
        return urlunsplit((parsed.scheme, netloc, parsed.path or "", "", ""))
    except Exception:
        return text.split("?", 1)[0].split("#", 1)[0]


def normalize_handler_metadata_value(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (str, int, float, bool)):
        return raw
    if isinstance(raw, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(raw.keys(), key=lambda item: str(item)):
            normalized[str(key)] = normalize_handler_metadata_value(raw.get(key))
        return normalized
    if isinstance(raw, (list, tuple)):
        return [normalize_handler_metadata_value(item) for item in raw]
    return str(raw)


def summarize_oversize_handler_metadata(
    runner: Any,
    *,
    normalized: Any,
    original_chars: int,
    max_chars: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "_loom_meta": "metadata_omitted",
        "original_type": type(normalized).__name__,
        "original_chars": max(0, int(original_chars)),
        "sha1": runner._stable_json_digest(normalized),
    }
    if isinstance(normalized, dict):
        keys = [str(key) for key in sorted(normalized.keys())]
        summary["key_count"] = len(keys)
        if keys:
            summary["keys"] = keys[:6]
            if len(keys) > 6:
                summary["keys_omitted"] = len(keys) - 6
    elif isinstance(normalized, list):
        summary["item_count"] = len(normalized)
    elif isinstance(normalized, str):
        summary["string_chars"] = len(normalized)

    if runner._stable_json_length(summary) <= max_chars:
        return summary

    for optional_key in (
        "keys",
        "keys_omitted",
        "item_count",
        "string_chars",
        "key_count",
    ):
        summary.pop(optional_key, None)
        if runner._stable_json_length(summary) <= max_chars:
            return summary

    minimal = {
        "_loom_meta": "metadata_omitted",
        "original_type": type(normalized).__name__,
        "sha1": summary["sha1"],
    }
    return minimal


def sanitize_handler_metadata(runner: Any, raw: Any) -> Any:
    normalized = normalize_handler_metadata_value(raw)
    if normalized is None:
        return None
    max_chars = int(
        getattr(
            runner,
            "_artifact_telemetry_max_metadata_chars",
            runner.ARTIFACT_TELEMETRY_MAX_METADATA_CHARS,
        ),
    )
    max_chars = max(120, max_chars)
    original_chars = runner._stable_json_length(normalized)
    if original_chars <= max_chars:
        return normalized
    return summarize_oversize_handler_metadata(
        runner,
        normalized=normalized,
        original_chars=original_chars,
        max_chars=max_chars,
    )


def emit_telemetry_event(
    runner: Any,
    *,
    event_type: str,
    task_id: str,
    data: dict[str, Any],
    counter_key: str = "",
    counter_amount: int = 1,
) -> None:
    if not runner._event_bus:
        return
    from loom.events.bus import Event

    runner._event_bus.emit(Event(event_type=event_type, task_id=task_id, data=data))
    if counter_key:
        increment_subtask_counter(runner, counter_key, counter_amount)


def artifact_event_common_payload(
    runner: Any,
    *,
    subtask_id: str,
    tool_name: str,
    tool_args: dict,
    result: ToolResult,
) -> dict[str, Any]:
    data = result.data if isinstance(result.data, dict) else {}
    url = data.get("url", "") or data.get("source_url", "") or tool_args.get("url", "")
    content_kind = str(data.get("content_kind", "")).strip() or "unknown"
    content_type = str(
        data.get("content_type", "") or data.get("media_type", ""),
    ).strip()
    return {
        "subtask_id": subtask_id,
        "tool": tool_name,
        "url": sanitize_url_for_telemetry(url),
        "content_kind": content_kind,
        "content_type": content_type,
        "status": "ok" if result.success else "error",
    }


def emit_artifact_ingest_telemetry(
    runner: Any,
    *,
    task_id: str,
    subtask_id: str,
    tool_name: str,
    tool_args: dict,
    result: ToolResult,
) -> None:
    if not telemetry_events_enabled(runner) or not runner._event_bus:
        return
    if tool_name not in {"web_fetch", "web_fetch_html"}:
        return
    data = result.data if isinstance(result.data, dict) else {}
    artifact_ref = str(data.get("artifact_ref", "")).strip()
    artifact_relpath = str(data.get("artifact_workspace_relpath", "")).strip()
    artifact_path = str(data.get("artifact_path", "")).strip()
    if not (artifact_ref or artifact_relpath or artifact_path):
        return

    payload = artifact_event_common_payload(
        runner,
        subtask_id=subtask_id,
        tool_name=tool_name,
        tool_args=tool_args,
        result=result,
    )
    if artifact_ref:
        payload["artifact_ref"] = artifact_ref
    if artifact_relpath:
        payload["artifact_workspace_relpath"] = artifact_relpath
    elif artifact_path:
        payload["artifact_path"] = artifact_path
    if "size_bytes" in data:
        payload["size_bytes"] = safe_int(data.get("size_bytes"))
    if "declared_size_bytes" in data:
        payload["declared_size_bytes"] = safe_int(data.get("declared_size_bytes"))
    handler = str(data.get("handler", "")).strip()
    if handler:
        payload["handler"] = handler
    if "extracted_chars" in data:
        payload["extracted_chars"] = safe_int(data.get("extracted_chars"))
    if "extraction_truncated" in data:
        payload["extraction_truncated"] = bool(data.get("extraction_truncated"))
    metadata = sanitize_handler_metadata(runner, data.get("handler_metadata"))
    if metadata is not None:
        payload["handler_metadata"] = metadata

    emit_telemetry_event(
        runner,
        event_type=ARTIFACT_INGEST_CLASSIFIED,
        task_id=task_id,
        data=dict(payload),
    )
    emit_telemetry_event(
        runner,
        event_type=ARTIFACT_INGEST_COMPLETED,
        task_id=task_id,
        data=dict(payload),
        counter_key="artifact_ingests",
    )

    retention = data.get("artifact_retention")
    if isinstance(retention, dict):
        files_deleted = max(0, safe_int(retention.get("files_deleted")))
        if files_deleted > 0:
            retention_payload = dict(payload)
            retention_payload["scopes_scanned"] = max(
                0,
                safe_int(retention.get("scopes_scanned")),
            )
            retention_payload["files_deleted"] = files_deleted
            retention_payload["bytes_deleted"] = max(
                0,
                safe_int(retention.get("bytes_deleted")),
            )
            emit_telemetry_event(
                runner,
                event_type=ARTIFACT_RETENTION_PRUNED,
                task_id=task_id,
                data=retention_payload,
                counter_key="artifact_retention_deletes",
                counter_amount=files_deleted,
            )


def emit_artifact_read_telemetry(
    runner: Any,
    *,
    task_id: str,
    subtask_id: str,
    tool_name: str,
    tool_args: dict,
    result: ToolResult,
) -> None:
    if not telemetry_events_enabled(runner) or not runner._event_bus:
        return
    if tool_name != "read_artifact":
        return

    data = result.data if isinstance(result.data, dict) else {}
    payload = artifact_event_common_payload(
        runner,
        subtask_id=subtask_id,
        tool_name=tool_name,
        tool_args=tool_args,
        result=result,
    )
    artifact_ref = str(
        data.get("artifact_ref", "") or tool_args.get("artifact_ref", ""),
    ).strip()
    if artifact_ref:
        payload["artifact_ref"] = artifact_ref
    artifact_relpath = str(data.get("artifact_workspace_relpath", "")).strip()
    artifact_path = str(data.get("artifact_path", "")).strip()
    if artifact_relpath:
        payload["artifact_workspace_relpath"] = artifact_relpath
    elif artifact_path:
        payload["artifact_path"] = artifact_path
    if "size_bytes" in data:
        payload["size_bytes"] = safe_int(data.get("size_bytes"))
    if "declared_size_bytes" in data:
        payload["declared_size_bytes"] = safe_int(data.get("declared_size_bytes"))
    handler = str(data.get("handler", "")).strip()
    if handler:
        payload["handler"] = handler
    if "extracted_chars" in data:
        payload["extracted_chars"] = safe_int(data.get("extracted_chars"))
    if "extraction_truncated" in data:
        payload["extraction_truncated"] = bool(data.get("extraction_truncated"))
    metadata = sanitize_handler_metadata(runner, data.get("handler_metadata"))
    if metadata is not None:
        payload["handler_metadata"] = metadata
    if not result.success and result.error:
        payload["error"] = str(result.error)

    emit_telemetry_event(
        runner,
        event_type=ARTIFACT_READ_COMPLETED,
        task_id=task_id,
        data=payload,
        counter_key="artifact_reads",
    )


def compaction_decision_from_diagnostics(
    runner: Any,
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    skip_reason = normalize_reason_code(
        str(diagnostics.get("compaction_skipped_reason", "")).strip(),
    )
    stage = str(diagnostics.get("compaction_stage", "")).strip().lower()
    if skip_reason and skip_reason not in {"none", "unspecified"}:
        return "skip", skip_reason
    if stage in {"stage_1_tool_args", "stage_2_tool_outputs"}:
        reason = (
            "tool_args_compacted"
            if stage == "stage_1_tool_args"
            else "tool_output_compacted"
        )
        return "compact_tool", reason
    if stage in {"stage_2_general", "stage_3_historical", "stage_3_minimal", "stage_4_merge"}:
        reason = "history_merged" if stage == "stage_4_merge" else "history_compacted"
        return "compact_history", reason
    return "skip", "no_change"


def emit_compaction_policy_decision_from_diagnostics(
    runner: Any,
    *,
    task_id: str,
    subtask_id: str,
) -> None:
    if not telemetry_events_enabled(runner) or not runner._event_bus:
        return
    diagnostics = dict(getattr(runner, "_last_compaction_diagnostics", {}))
    mode = str(
        diagnostics.get(
            "compaction_policy_mode",
            runner._runner_compaction_mode(),
        ),
    ).strip().lower()
    pressure_ratio = safe_float(diagnostics.get("compaction_pressure_ratio", 0.0))
    if pressure_ratio <= 0.0:
        before = safe_float(diagnostics.get("compaction_est_tokens_before", 0.0))
        budget = float(
            max(
                1,
                int(
                    getattr(
                        runner,
                        "_max_model_context_tokens",
                        runner.MAX_MODEL_CONTEXT_TOKENS,
                    ),
                ),
            ),
        )
        pressure_ratio = before / budget if before > 0 else 0.0
    decision, reason = compaction_decision_from_diagnostics(runner, diagnostics)
    emit_telemetry_event(
        runner,
        event_type=COMPACTION_POLICY_DECISION,
        task_id=task_id,
        data={
            "subtask_id": subtask_id,
            "pressure_ratio": round(pressure_ratio, 4),
            "policy_mode": mode or runner.RUNNER_COMPACTION_POLICY_MODE,
            "decision": decision,
            "reason": reason,
        },
        counter_key="compaction_policy_decisions",
    )


def emit_overflow_fallback_telemetry(
    runner: Any,
    *,
    task_id: str,
    subtask_id: str,
    report: dict[str, Any],
) -> None:
    if not telemetry_events_enabled(runner) or not runner._event_bus:
        return
    if not bool(report.get("overflow_fallback_applied", False)):
        return
    diagnostics = dict(getattr(runner, "_last_compaction_diagnostics", {}))
    mode = str(
        diagnostics.get(
            "compaction_policy_mode",
            runner._runner_compaction_mode(),
        ),
    ).strip().lower() or runner.RUNNER_COMPACTION_POLICY_MODE
    pressure_ratio = safe_float(diagnostics.get("compaction_pressure_ratio", 0.0))
    payload = {
        "subtask_id": subtask_id,
        "pressure_ratio": round(pressure_ratio, 4),
        "policy_mode": mode,
        "decision": "fallback_rewrite",
        "reason": "overflow_context_limit",
        "rewritten_messages": max(
            0,
            safe_int(report.get("overflow_fallback_rewritten_messages", 0)),
        ),
        "chars_reduced": max(
            0,
            safe_int(report.get("overflow_fallback_chars_reduced", 0)),
        ),
        "preserved_recent_messages": max(
            0,
            safe_int(report.get("overflow_fallback_preserved_recent_messages", 0)),
        ),
    }
    emit_telemetry_event(
        runner,
        event_type=COMPACTION_POLICY_DECISION,
        task_id=task_id,
        data={
            "subtask_id": payload["subtask_id"],
            "pressure_ratio": payload["pressure_ratio"],
            "policy_mode": payload["policy_mode"],
            "decision": payload["decision"],
            "reason": payload["reason"],
        },
        counter_key="compaction_policy_decisions",
    )
    emit_telemetry_event(
        runner,
        event_type=OVERFLOW_FALLBACK_APPLIED,
        task_id=task_id,
        data=payload,
        counter_key="overflow_fallback_count",
    )
