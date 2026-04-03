"""API route handlers for Loom."""

from __future__ import annotations

import asyncio
import csv
import ipaddress
import json
import logging
import mimetypes
import os
import time
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from loom import __version__
from loom.api.engine import (
    Engine,
    TelemetryPersistConflictError,
    TelemetryPersistDisabledError,
)
from loom.api.schemas import (
    ActivitySummaryResponse,
    ApprovalFeedItemResponse,
    ApprovalReplyRequest,
    ApprovalRequest,
    ConversationApprovalDecisionRequest,
    ConversationCreateRequest,
    ConversationInjectRequest,
    ConversationMessageRequest,
    ConversationPatchRequest,
    ConversationSummaryResponse,
    FeedbackRequest,
    HealthResponse,
    MCPServerInfoResponse,
    ModelCapabilitiesResponse,
    ModelInfo,
    PlanResponse,
    ProcessInfoResponse,
    ProgressResponse,
    RunArtifactResponse,
    RunFailureAnalysisResponse,
    RunFailureRemediationResponse,
    RunSummaryResponse,
    RuntimeStatusResponse,
    SettingsPatchRequest,
    SubtaskSummaryResponse,
    TaskCreateRequest,
    TaskCreateResponse,
    TaskListItem,
    TaskQuestionAnswerRequest,
    TaskQuestionResponse,
    TaskResponse,
    TaskSteerRequest,
    TelemetrySettingsPatchRequest,
    TelemetrySettingsResponse,
    ToolInfo,
    WorkspaceArtifactResponse,
    WorkspaceCreateRequest,
    WorkspaceFileEntryResponse,
    WorkspaceFilePreviewResponse,
    WorkspaceFilePreviewTableResponse,
    WorkspaceInventoryResponse,
    WorkspaceOverviewResponse,
    WorkspacePatchRequest,
    WorkspaceSearchItemResponse,
    WorkspaceSearchResponse,
    WorkspaceSettingsPatchRequest,
    WorkspaceSummaryResponse,
)
from loom.auth.runtime import (
    AuthResolutionError,
    UnresolvedAuthResourcesError,
    build_run_auth_context,
    coerce_auth_requirements,
    serialize_auth_requirements,
)
from loom.config_runtime import (
    ConfigPersistConflictError,
    ConfigPersistDisabledError,
    list_entries,
)
from loom.content import serialize_block
from loom.content_utils import extract_docx_text, extract_pptx_text, get_image_dimensions
from loom.cowork.approval import ApprovalDecision as CoworkApprovalDecision
from loom.cowork.approval import ToolApprover
from loom.cowork.session import (
    CoworkSession,
    CoworkStopRequestedError,
    CoworkTurn,
    ToolCallEvent,
    build_cowork_system_prompt,
)
from loom.engine.orchestrator import create_task
from loom.events.bus import Event
from loom.events.types import (
    APPROVAL_RECEIVED,
    APPROVAL_REQUESTED,
    ASK_USER_ANSWERED,
    ASK_USER_CANCELLED,
    ASK_USER_REQUESTED,
    ASK_USER_TIMEOUT,
    CONVERSATION_MESSAGE,
    STEER_INSTRUCTION,
    TASK_CANCEL_REQUESTED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_CREATED,
    TASK_FAILED,
    TASK_RESTARTED,
    TOKEN_STREAMED,
)
from loom.events.verbosity import should_deliver_operator
from loom.processes.run_workspace import provision_scoped_run_workspace
from loom.processes.schema import ProcessLoader
from loom.read_scope import build_attached_read_scope, iter_context_workspace_paths
from loom.state.memory import MemoryEntry
from loom.state.task_state import Plan, Subtask, SubtaskStatus, Task, TaskStatus
from loom.state.workspaces import canonicalize_workspace_path
from loom.tools.ask_user import normalize_ask_user_args
from loom.tools.registry import (
    ToolResult,
    normalize_tool_auth_mode,
    normalize_tool_execution_surface,
    normalize_tool_execution_surfaces,
    tool_auth_required,
)
from loom.tools.workspace import validate_workspace
from loom.utils.latency import log_latency_event, timed_block

router = APIRouter()

_STREAM_QUEUE_MAXSIZE = 256
logger = logging.getLogger(__name__)


def _latency_fields(**fields: object) -> dict[str, object]:
    return {
        str(key): value
        for key, value in fields.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }


def _cowork_session_cls() -> type[CoworkSession]:
    return CoworkSession


def _required_auth_resources_for_process(
    process_def: object | None,
    *,
    tool_registry: object | None = None,
) -> list[dict[str, object]]:
    """Collect auth requirement declarations from process + allowed tools."""
    if process_def is None:
        return []

    raw_items: list[object] = []
    auth_block = getattr(process_def, "auth", None)
    process_required = getattr(auth_block, "required", [])
    if isinstance(process_required, list):
        raw_items.extend(process_required)

    if tool_registry is not None:
        tools_cfg = getattr(process_def, "tools", None)
        excluded = {
            str(item).strip()
            for item in (getattr(tools_cfg, "excluded", []) or [])
            if str(item).strip()
        }
        required = {
            str(item).strip()
            for item in (getattr(tools_cfg, "required", []) or [])
            if str(item).strip()
        }
        list_tools = getattr(tool_registry, "list_tools", None)
        get_tool = getattr(tool_registry, "get", None)
        if callable(list_tools) and callable(get_tool):
            if required:
                candidate_tool_names = sorted(required - excluded)
            else:
                candidate_tool_names = sorted(
                    {
                        str(name).strip()
                        for name in list_tools()
                        if str(name).strip() and str(name).strip() not in excluded
                    }
                )

            for tool_name in candidate_tool_names:
                tool = get_tool(tool_name)
                if tool is None:
                    continue
                declared = getattr(tool, "auth_requirements", [])
                if isinstance(declared, list):
                    raw_items.extend(declared)

    normalized = coerce_auth_requirements(raw_items)
    return serialize_auth_requirements(normalized)


def _prepare_task_for_restart_from_failure(task: Task) -> tuple[Task | None, str | None]:
    """Reset a terminal task for another execution pass, preserving completed work."""
    if not task.plan or not task.plan.subtasks:
        return None, (
            f"Cannot restart task '{task.id}': no saved subtask plan available."
        )

    needs_work = False
    for subtask in task.plan.subtasks:
        if subtask.status == SubtaskStatus.COMPLETED:
            continue
        needs_work = True
        if subtask.status in {
            SubtaskStatus.PENDING,
            SubtaskStatus.RUNNING,
            SubtaskStatus.FAILED,
            SubtaskStatus.BLOCKED,
            SubtaskStatus.SKIPPED,
        }:
            subtask.status = SubtaskStatus.PENDING
        subtask.retry_count = 0
        subtask.active_issue = ""
        if subtask.status == SubtaskStatus.PENDING:
            subtask.summary = ""

    if not needs_work:
        return None, f"Cannot restart task '{task.id}': no remaining work."

    task.completed_at = ""
    task.status = TaskStatus.PENDING
    task.add_decision("Resumed execution from prior task state.")
    return task, None


def _plan_from_json_dict(plan_data: dict[str, Any]) -> Plan:
    """Reconstruct a persisted plan JSON payload into the dataclass form."""
    subtasks: list[Subtask] = []
    for raw in plan_data.get("subtasks", []) or []:
        if not isinstance(raw, dict):
            continue
        validity_snapshot_raw = raw.get("validity_contract_snapshot", {})
        if not isinstance(validity_snapshot_raw, dict):
            validity_snapshot_raw = {}
        best_score_raw = raw.get("iteration_best_score")
        try:
            best_score = float(best_score_raw) if best_score_raw is not None else None
        except (TypeError, ValueError):
            best_score = None
        subtasks.append(Subtask(
            id=str(raw.get("id", "") or ""),
            description=str(raw.get("description", "") or ""),
            status=SubtaskStatus(str(raw.get("status", "pending") or "pending")),
            summary=str(raw.get("summary", "") or ""),
            active_issue=str(raw.get("active_issue", "") or ""),
            depends_on=[
                str(item).strip()
                for item in (raw.get("depends_on", []) or [])
                if str(item).strip()
            ],
            phase_id=str(raw.get("phase_id", "") or ""),
            output_role=str(raw.get("output_role", "") or ""),
            output_strategy=str(raw.get("output_strategy", "") or ""),
            model_tier=int(raw.get("model_tier", 1) or 1),
            verification_tier=int(raw.get("verification_tier", 1) or 1),
            is_critical_path=bool(raw.get("is_critical_path", False)),
            is_synthesis=bool(raw.get("is_synthesis", False)),
            acceptance_criteria=str(raw.get("acceptance_criteria", "") or ""),
            validity_contract_snapshot=dict(validity_snapshot_raw),
            validity_contract_hash=str(raw.get("validity_contract_hash", "") or ""),
            retry_count=int(raw.get("retry_count", 0) or 0),
            max_retries=int(raw.get("max_retries", 3) or 3),
            iteration_attempt=int(raw.get("iteration_attempt", 0) or 0),
            iteration_runner_invocations=int(raw.get("iteration_runner_invocations", 0) or 0),
            iteration_max_attempts=int(raw.get("iteration_max_attempts", 0) or 0),
            iteration_no_improvement_count=int(
                raw.get("iteration_no_improvement_count", 0) or 0,
            ),
            iteration_best_score=best_score,
            iteration_terminal_reason=str(raw.get("iteration_terminal_reason", "") or ""),
            iteration_loop_run_id=str(raw.get("iteration_loop_run_id", "") or ""),
            iteration_replan_count=int(raw.get("iteration_replan_count", 0) or 0),
            iteration_last_gate_summary=str(raw.get("iteration_last_gate_summary", "") or ""),
        ))
    return Plan(
        subtasks=subtasks,
        version=int(plan_data.get("version", 1) or 1),
        last_replanned=str(plan_data.get("last_replanned", "") or ""),
    )


def _get_engine(request: Request) -> Engine:
    """Get the engine from the app state."""
    return request.app.state.engine


def _normalize_host(raw_host: str) -> str:
    host = str(raw_host or "").strip().lower()
    if not host:
        return ""
    if host.startswith("[") and "]" in host:
        host = host[1:host.index("]")]
    elif host.count(":") == 1:
        candidate, maybe_port = host.rsplit(":", 1)
        if maybe_port.isdigit():
            host = candidate
    return host


def _is_loopback_host(raw_host: str) -> bool:
    host = _normalize_host(raw_host)
    if host in {"localhost", "127.0.0.1", "::1", "testclient", "testserver", "test"}:
        return True
    try:
        return bool(ipaddress.ip_address(host).is_loopback)
    except ValueError:
        return False


def _request_is_local(request: Request) -> bool:
    client_host = _normalize_host(getattr(request.client, "host", "") if request.client else "")
    if not _is_loopback_host(client_host):
        return False

    forwarded_for = str(request.headers.get("x-forwarded-for", "") or "").strip()
    if forwarded_for:
        first_hop = forwarded_for.split(",", 1)[0].strip()
        if not _is_loopback_host(first_hop):
            return False

    origin = str(request.headers.get("origin", "") or "").strip()
    if origin:
        parsed = urlparse(origin)
        if not _is_loopback_host(parsed.hostname or ""):
            return False
    return True


def _extract_admin_token(request: Request) -> str:
    explicit = str(request.headers.get("x-loom-admin-token", "") or "").strip()
    if explicit:
        return explicit
    authorization = str(request.headers.get("authorization", "") or "").strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _build_telemetry_settings_response(engine: Engine) -> TelemetrySettingsResponse:
    snapshot = engine.telemetry_settings_snapshot()
    return TelemetrySettingsResponse(
        configured_mode=str(snapshot.get("configured_mode", "active") or "active"),
        runtime_override_mode=str(snapshot.get("runtime_override_mode", "") or ""),
        effective_mode=str(snapshot.get("effective_mode", "active") or "active"),
        scope=str(snapshot.get("scope", "process_local") or "process_local"),
        updated_at=str(snapshot.get("updated_at", "") or ""),
    )


def _require_telemetry_mutation_access(request: Request, engine: Engine) -> None:
    telemetry_cfg = getattr(engine.config, "telemetry", None)
    if not bool(getattr(telemetry_cfg, "runtime_override_api_enabled", False)):
        raise HTTPException(status_code=404, detail="Telemetry settings API is disabled.")
    if not _request_is_local(request):
        raise HTTPException(
            status_code=403,
            detail="Telemetry settings mutation requires a local loopback caller.",
        )
    expected_token = str(getattr(telemetry_cfg, "runtime_override_api_token", "") or "").strip()
    provided_token = _extract_admin_token(request)
    if not expected_token or provided_token != expected_token:
        raise HTTPException(
            status_code=403,
            detail="Telemetry settings mutation requires a valid admin token.",
        )


def _serialize_task_question(row: dict) -> TaskQuestionResponse:
    payload = dict(row or {})
    return TaskQuestionResponse(
        question_id=str(payload.get("question_id", "") or ""),
        task_id=str(payload.get("task_id", "") or ""),
        subtask_id=str(payload.get("subtask_id", "") or ""),
        status=str(payload.get("status", "") or ""),
        request_payload=(
            payload.get("request_payload")
            if isinstance(payload.get("request_payload"), dict)
            else {}
        ),
        answer_payload=(
            payload.get("answer_payload")
            if isinstance(payload.get("answer_payload"), dict)
            else {}
        ),
        created_at=str(payload.get("created_at", "") or ""),
        updated_at=str(payload.get("updated_at", "") or ""),
        resolved_at=str(payload.get("resolved_at", "") or ""),
        timeout_at=str(payload.get("timeout_at", "") or ""),
    )


def _task_execution_surface(task: object) -> str:
    metadata = getattr(task, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    raw = metadata.get("execution_surface", "")
    if not str(raw or "").strip():
        task_id = str(getattr(task, "id", "") or "").strip().lower()
        if task_id.startswith("cowork-"):
            return "tui"
    return normalize_tool_execution_surface(
        raw,
        default="api",
    )


def _json_object(raw_value: object, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = dict(default or {})
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if not isinstance(raw_value, str):
        return fallback
    text = raw_value.strip()
    if not text:
        return fallback
    try:
        parsed = json.loads(text)
    except Exception:
        return fallback
    return dict(parsed) if isinstance(parsed, dict) else fallback


def _json_list(raw_value: object) -> list[Any]:
    if isinstance(raw_value, list):
        return list(raw_value)
    if not isinstance(raw_value, str):
        return []
    text = raw_value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _workspace_matches_path(workspace: dict[str, Any], workspace_path: object) -> bool:
    return canonicalize_workspace_path(workspace_path) == str(
        workspace.get("canonical_path", "") or "",
    )


def _task_workspace_group_root(task_row: dict[str, Any]) -> str:
    metadata = _json_object(task_row.get("metadata"))
    source_root = canonicalize_workspace_path(metadata.get("source_workspace_root"))
    if source_root:
        return source_root
    return canonicalize_workspace_path(task_row.get("workspace_path"))


def _workspace_matches_task(workspace: dict[str, Any], task_row: dict[str, Any]) -> bool:
    return _task_workspace_group_root(task_row) == str(
        workspace.get("canonical_path", "") or "",
    )


def _task_run_workspace_prefix(task_row: dict[str, Any]) -> str:
    metadata = _json_object(task_row.get("metadata"))
    raw_prefix = str(metadata.get("run_workspace_relative", "") or "").strip()
    if not raw_prefix:
        return ""
    return raw_prefix.strip("/\\")


def _workspace_visible_artifact_path(task_row: dict[str, Any], relpath: object) -> str:
    artifact_relpath = str(relpath or "").strip().strip("/\\")
    if not artifact_relpath:
        return ""
    prefix = _task_run_workspace_prefix(task_row)
    if not prefix:
        return artifact_relpath
    return f"{prefix}/{artifact_relpath}"


def _artifact_relpath_within_root(workspace_path: object, relpath: object) -> str:
    workspace_text = canonicalize_workspace_path(workspace_path)
    artifact_relpath = str(relpath or "").strip()
    if not workspace_text or not artifact_relpath:
        return ""
    try:
        workspace_root = Path(workspace_text).expanduser().resolve()
        candidate = Path(artifact_relpath).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
        else:
            resolved = (workspace_root / candidate).resolve(strict=False)
        relative = resolved.relative_to(workspace_root)
    except Exception:
        return ""
    relative_text = str(relative).replace("\\", "/").strip().strip("/\\")
    if not relative_text or relative_text == ".":
        return ""
    return relative_text


def _artifact_matches_attached_context(task_row: dict[str, Any], relpath: str) -> bool:
    artifact_relpath = str(relpath or "").strip().strip("/\\")
    if not artifact_relpath:
        return False
    context = _json_object(task_row.get("context"))
    for attached in iter_context_workspace_paths(context):
        if artifact_relpath == attached or artifact_relpath.startswith(f"{attached}/"):
            return True
    return False


def _artifact_fallback_visible_path(task_row: dict[str, Any], relpath: object) -> str:
    artifact_relpath = str(relpath or "").strip().replace("\\", "/")
    if not artifact_relpath:
        return ""
    try:
        candidate = Path(artifact_relpath).expanduser()
    except Exception:
        candidate = Path(artifact_relpath)
    if candidate.is_absolute():
        return artifact_relpath
    return _workspace_visible_artifact_path(task_row, artifact_relpath)


def _resolve_artifact_locator(
    task_row: dict[str, Any],
    relpath: object,
    *,
    prefer_run_workspace: bool = False,
) -> tuple[str, str, bool]:
    artifact_relpath = str(relpath or "").strip()
    if not artifact_relpath:
        return "", "", False

    metadata = _json_object(task_row.get("metadata"))
    run_workspace = canonicalize_workspace_path(task_row.get("workspace_path"))
    source_workspace = canonicalize_workspace_path(metadata.get("source_workspace_root"))
    if source_workspace == run_workspace:
        source_workspace = ""

    run_relpath = _artifact_relpath_within_root(run_workspace, artifact_relpath)
    source_relpath = (
        _artifact_relpath_within_root(source_workspace, artifact_relpath)
        if source_workspace
        else ""
    )
    run_exists = (
        _artifact_exists_on_disk(run_workspace, run_relpath)
        if run_relpath
        else False
    )
    source_exists = (
        _artifact_exists_on_disk(source_workspace, source_relpath)
        if source_relpath
        else False
    )

    if prefer_run_workspace and run_relpath:
        return (
            f"run:{run_relpath}",
            _workspace_visible_artifact_path(task_row, run_relpath),
            run_exists,
        )
    if run_exists and run_relpath:
        return (
            f"run:{run_relpath}",
            _workspace_visible_artifact_path(task_row, run_relpath),
            True,
        )
    if source_exists and source_relpath:
        return f"source:{source_relpath}", source_relpath, True
    if source_relpath and _artifact_matches_attached_context(task_row, source_relpath):
        return f"source:{source_relpath}", source_relpath, False
    if run_relpath:
        return (
            f"run:{run_relpath}",
            _workspace_visible_artifact_path(task_row, run_relpath),
            False,
        )
    if source_relpath:
        return f"source:{source_relpath}", source_relpath, False

    fallback = _artifact_fallback_visible_path(task_row, artifact_relpath)
    return f"raw:{fallback or artifact_relpath}", fallback, False


def _latest_timestamp(*values: object) -> str:
    normalized = [str(value or "").strip() for value in values if str(value or "").strip()]
    return max(normalized) if normalized else ""


def _append_unique(items: list[str], value: object) -> None:
    text = str(value or "").strip()
    if text and text not in items:
        items.append(text)


def _text_matches_query(query: str, *values: object) -> bool:
    normalized_query = str(query or "").strip().lower()
    if not normalized_query:
        return True
    haystack = " ".join(str(value or "") for value in values if str(value or "").strip()).lower()
    return normalized_query in haystack


def _trim_snippet(value: object, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


_FAILURE_REASON_FAMILIES: dict[str, str] = {
    "auth_preflight_failed": "auth",
    "blocked_pending_subtasks": "scheduler",
    "coverage_below_threshold": "contract_failure",
    "dev_browser_check_failed": "product_verification",
    "dev_build_failed": "product_verification",
    "dev_contract_failed": "product_verification",
    "dev_report_contract_violation": "verification_infra",
    "dev_test_failed": "product_verification",
    "dev_verifier_capability_unavailable": "verification_infra",
    "dev_verifier_timeout": "verification_infra",
    "forbidden_output_path": "policy",
    "hard_invariant_failed": "contract_failure",
    "infra_verifier_error": "verification_infra",
    "manifest_input_policy_violation": "policy",
    "parse_inconclusive": "unconfirmed_data",
    "policy_remediation_required": "policy",
    "recommendation_unconfirmed": "unconfirmed_data",
    "unconfirmed_critical_path": "unconfirmed_data",
    "uncaught_exception": "runtime",
}


def _event_payload_text(value: object, *, limit: int = 320) -> str:
    return _trim_snippet(value, limit=limit)


def _first_nonempty_line(text: object, *, limit: int = 280) -> str:
    for line in str(text or "").splitlines():
        compact = _event_payload_text(line, limit=limit)
        if compact:
            return compact
    return ""


def _latest_event(
    events: list[dict[str, Any]],
    *,
    event_types: set[str],
    subtask_id: str = "",
) -> dict[str, Any] | None:
    for event in reversed(events):
        if str(event.get("event_type", "") or "") not in event_types:
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}
        if subtask_id and str(data.get("subtask_id", "") or "").strip() != subtask_id:
            continue
        return event
    return None


def _reason_family(reason_code: str, task_reason: str) -> str:
    normalized = str(reason_code or "").strip().lower()
    if normalized:
        family = _FAILURE_REASON_FAMILIES.get(normalized, "")
        if family:
            return family
        if normalized.startswith("dev_"):
            return "product_verification"
    task_reason_normalized = str(task_reason or "").strip().lower()
    if task_reason_normalized == "blocked_pending_subtasks":
        return "scheduler"
    if task_reason_normalized == "uncaught_exception":
        return "runtime"
    if task_reason_normalized == "blocking_remediation_unresolved":
        return "remediation"
    return "verification"


def _plan_label_lookup(task_obj: Task | None) -> dict[str, str]:
    if task_obj is None or task_obj.plan is None:
        return {}
    lookup: dict[str, str] = {}
    for subtask in task_obj.plan.subtasks:
        subtask_id = str(getattr(subtask, "id", "") or "").strip()
        label = str(getattr(subtask, "description", "") or "").strip()
        if subtask_id:
            lookup[subtask_id] = label or subtask_id
    return lookup


def _build_tool_failure_detail(
    failed_tool_events: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    if not failed_tool_events:
        return "", []
    errors: list[tuple[str, str]] = []
    for event in failed_tool_events:
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}
        tool_name = str(data.get("tool", "") or "").strip() or "tool"
        error = _event_payload_text(data.get("error", ""), limit=220)
        if error:
            errors.append((tool_name, error))
    if not errors:
        return "", []

    counts = Counter(errors)
    (tool_name, top_error), top_count = counts.most_common(1)[0]
    evidence = [
        f"{tool}: {error}" if count == 1 else f"{tool}: {error} ({count}x)"
        for (tool, error), count in counts.most_common(2)
    ]
    lowered = top_error.lower()
    if "http 999" in lowered or "anti-bot denied" in lowered:
        repeat_prefix = (
            "Repeated attempts"
            if len(failed_tool_events) > 1 or top_count > 1
            else "The fetch attempt"
        )
        return (
            f"{repeat_prefix} to read the target site were denied with HTTP 999, "
            "which usually means the site blocked the requests as automated traffic.",
            evidence,
        )
    if "http 403" in lowered or "forbidden" in lowered:
        repeat_prefix = (
            "Repeated attempts"
            if len(failed_tool_events) > 1 or top_count > 1
            else "The fetch attempt"
        )
        return (
            f"{repeat_prefix} to read the target site were denied with an access-control response "
            f"({tool_name}: {top_error}).",
            evidence,
        )
    if len(failed_tool_events) > 1 or top_count > 1:
        return (
            f"Repeated tool failures prevented recovery; the most common error was "
            f"{tool_name}: {top_error}.",
            evidence,
        )
    return (f"The final tool failure was {tool_name}: {top_error}.", evidence)


def _build_run_failure_analysis(
    *,
    resolved_status: str,
    task_obj: Task | None,
    event_rows: list[dict[str, Any]],
) -> RunFailureAnalysisResponse | None:
    if str(resolved_status or "").strip().lower() != "failed":
        return None

    events = [_serialize_run_timeline_row(row) for row in event_rows]
    if not events:
        return RunFailureAnalysisResponse(
            headline="Run failed.",
            summary=(
                "The run ended in a failed state, but no persisted failure events "
                "were available."
            ),
            reason_family="runtime",
            remediation=RunFailureRemediationResponse(),
        )

    task_failed = _latest_event(events, event_types={"task_failed"}) or {}
    task_failed_data = (
        dict(task_failed.get("data", {}))
        if isinstance(task_failed.get("data"), dict)
        else {}
    )
    telemetry_summary = _latest_event(events, event_types={"telemetry_run_summary"}) or {}
    telemetry_data = (
        dict(telemetry_summary.get("data", {}))
        if isinstance(telemetry_summary.get("data"), dict)
        else {}
    )
    label_lookup = _plan_label_lookup(task_obj)

    failed_subtasks = [
        str(item or "").strip()
        for item in (task_failed_data.get("failed_subtasks", []) or [])
        if str(item or "").strip()
    ]
    blocked_subtasks = task_failed_data.get("blocked_subtasks", []) or []
    blocked_subtask_ids = [
        str(item.get("subtask_id", "") or "").strip()
        for item in blocked_subtasks
        if isinstance(item, dict) and str(item.get("subtask_id", "") or "").strip()
    ]
    latest_subtask_failed = _latest_event(events, event_types={"subtask_failed"})
    latest_subtask_failed_data = (
        dict(latest_subtask_failed.get("data", {}))
        if isinstance(latest_subtask_failed and latest_subtask_failed.get("data"), dict)
        else {}
    )
    failing_subtask_id = (
        (failed_subtasks[-1] if failed_subtasks else "")
        or (blocked_subtask_ids[0] if blocked_subtask_ids else "")
        or str(latest_subtask_failed_data.get("subtask_id", "") or "").strip()
    )
    failing_subtask_label = label_lookup.get(failing_subtask_id, failing_subtask_id)

    verification_terminal = _latest_event(
        events,
        event_types={"verification_failed", "verification_outcome"},
        subtask_id=failing_subtask_id,
    ) or {}
    verification_data = (
        dict(verification_terminal.get("data", {}))
        if isinstance(verification_terminal.get("data"), dict)
        else {}
    )
    subtask_failed_event = _latest_event(
        events,
        event_types={"subtask_failed"},
        subtask_id=failing_subtask_id,
    ) or latest_subtask_failed or {}
    subtask_failed_data = (
        dict(subtask_failed_event.get("data", {}))
        if isinstance(subtask_failed_event.get("data"), dict)
        else {}
    )

    primary_reason_code = (
        str(verification_data.get("reason_code", "") or "").strip()
        or str(subtask_failed_data.get("reason_code", "") or "").strip()
        or str(task_failed_data.get("reason", "") or "").strip()
    )
    task_reason = str(task_failed_data.get("reason", "") or "").strip()
    feedback = (
        _first_nonempty_line(subtask_failed_data.get("feedback", ""), limit=320)
        or _first_nonempty_line(task_failed_data.get("message", ""), limit=320)
        or _first_nonempty_line(task_failed_data.get("error", ""), limit=320)
    )

    failed_tool_events = [
        event
        for event in events
        if str(event.get("event_type", "") or "") == "tool_call_completed"
        and isinstance(event.get("data"), dict)
        and not bool(dict(event.get("data", {})).get("success", False))
        and (
            not failing_subtask_id
            or str(dict(event.get("data", {})).get("subtask_id", "") or "").strip()
            == failing_subtask_id
        )
    ]
    technical_detail, tool_evidence = _build_tool_failure_detail(failed_tool_events)

    verification_reason_counts = telemetry_data.get("verification_reason_counts", {})
    if not isinstance(verification_reason_counts, dict):
        verification_reason_counts = {}
    remediation_counts = telemetry_data.get("remediation_lifecycle_counts", {})
    if not isinstance(remediation_counts, dict):
        remediation_counts = {}

    queued_count = int(remediation_counts.get("queued", 0) or 0)
    attempt_count = int(remediation_counts.get("attempt", 0) or 0)
    resolved_count = int(remediation_counts.get("resolved", 0) or 0)
    failed_count = int(remediation_counts.get("failed", 0) or 0)
    expired_count = int(remediation_counts.get("expired", 0) or 0)

    remediation_attempted = any(
        count > 0
        for count in (queued_count, attempt_count, resolved_count, failed_count, expired_count)
    )
    why_not_remedied = ""
    if primary_reason_code == "hard_invariant_failed":
        why_not_remedied = (
            "This was treated as a hard invariant verification failure, so Loom did not "
            "queue follow-up remediation and the run stopped after retries were exhausted."
        )
    elif resolved_count > 0:
        why_not_remedied = (
            "Remediation recovered some failures, but the final blocking issue "
            "remained unresolved."
        )
    elif queued_count > 0 and not resolved_count:
        why_not_remedied = (
            "Follow-up remediation was queued, but it did not resolve the blocking issue "
            "before the run ended."
        )
    elif attempt_count > 0 and not resolved_count:
        why_not_remedied = "Remediation was attempted, but it never satisfied the verifier."
    elif task_reason == "blocked_pending_subtasks":
        why_not_remedied = (
            "The run stalled with blocked pending subtasks, so Loom had no safe automatic path "
            "to continue."
        )
    elif task_reason == "blocking_remediation_unresolved":
        why_not_remedied = (
            "A blocking remediation item remained unresolved, so the run was "
            "terminated."
        )
    elif task_reason == "uncaught_exception":
        why_not_remedied = "The orchestrator hit an uncaught exception and stopped the run."

    reason_family = _reason_family(primary_reason_code, task_reason)
    headline = feedback or "The run ended in a failed state."
    if failing_subtask_label:
        headline = (
            f"{failing_subtask_label} failed. {headline}"
            if feedback and not headline.lower().startswith(failing_subtask_label.lower())
            else headline
        )

    summary_parts = [part for part in (feedback, technical_detail, why_not_remedied) if part]
    summary = " ".join(summary_parts) or "The run failed without a more specific explanation."

    evidence: list[str] = []
    if failing_subtask_label:
        evidence.append(f"Failing subtask: {failing_subtask_label}")
    if primary_reason_code:
        evidence.append(f"Verifier reason: {primary_reason_code}")
    if tool_evidence:
        evidence.extend(tool_evidence[:2])
    if verification_reason_counts:
        top_reason, top_count = max(
            (
                (str(reason or "").strip(), int(count or 0))
                for reason, count in verification_reason_counts.items()
                if str(reason or "").strip()
            ),
            key=lambda item: item[1],
            default=("", 0),
        )
        if top_reason:
            evidence.append(f"Most frequent verifier result: {top_reason} ({top_count})")

    next_actions: list[str] = []
    summary_lower = summary.lower()
    if "http 999" in summary_lower or "linkedin" in summary_lower:
        next_actions = [
            "Use authenticated or browser-backed fetches for the blocked site.",
            "Relax the LinkedIn coverage requirement or allow alternate contact channels.",
        ]
    elif reason_family == "product_verification":
        next_actions = [
            "Inspect the failing verification feedback and rerun after fixing the "
            "underlying product issue.",
        ]
    elif reason_family == "policy":
        next_actions = [
            "Adjust the process contract or write scope so the task can satisfy "
            "policy constraints.",
        ]

    return RunFailureAnalysisResponse(
        headline=_event_payload_text(headline, limit=260),
        summary=_event_payload_text(summary, limit=700),
        failing_subtask_id=failing_subtask_id,
        failing_subtask_label=failing_subtask_label,
        primary_reason_code=primary_reason_code,
        reason_family=reason_family,
        technical_detail=_event_payload_text(technical_detail, limit=320),
        evidence=evidence[:4],
        next_actions=next_actions[:3],
        remediation=RunFailureRemediationResponse(
            attempted=remediation_attempted,
            queued=queued_count > 0,
            resolved=resolved_count > 0,
            failed=failed_count > 0,
            expired=expired_count > 0,
            why_not_remedied=_event_payload_text(why_not_remedied, limit=320),
        ),
    )


def _recent_conversation_search_text(
    turns: list[dict[str, Any]],
    *,
    limit: int = 1200,
) -> str:
    blocks: list[str] = []
    for turn in turns:
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        role = str(turn.get("role", "") or "").strip().lower()
        if role:
            blocks.append(f"{role}: {content}")
        else:
            blocks.append(content)
    text = " ".join(blocks)
    return text[:limit]


def _artifact_is_intermediate(relpath: object) -> bool:
    text = str(relpath or "").strip()
    return bool(
        text.startswith(".loom/phase-artifacts/")
        or text.startswith("loom/phase-artifacts/")
    )


def _artifact_category(
    relpath: object,
    *,
    evidence_kind: object = "",
    facets: dict[str, Any] | None = None,
) -> str:
    if _artifact_is_intermediate(relpath):
        return "intermediate"
    if isinstance(facets, dict):
        facet_category = str(facets.get("category", "") or "").strip()
        if facet_category:
            return facet_category
    kind = str(evidence_kind or "").strip()
    if kind and kind != "artifact":
        return kind
    text = str(relpath or "").strip().lower()
    if text.startswith(".loom_artifacts/"):
        return "fetched_artifact"
    suffix = Path(text).suffix.lower()
    if suffix in {".md", ".txt", ".rst"}:
        return "document"
    if suffix in {".json", ".jsonl", ".yaml", ".yml", ".csv"}:
        return "structured_data"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
        return "image"
    if suffix in {".pdf"}:
        return "reference"
    return "workspace_file"


def _artifact_exists_on_disk(workspace_path: object, relpath: object) -> bool:
    workspace_text = str(workspace_path or "").strip()
    artifact_relpath = str(relpath or "").strip()
    if not workspace_text or not artifact_relpath:
        return False
    try:
        workspace_root = Path(workspace_text).expanduser().resolve()
        artifact_path = (workspace_root / artifact_relpath).resolve()
        artifact_path.relative_to(workspace_root)
    except Exception:
        return False
    return artifact_path.exists()


_WORKSPACE_FILE_PREVIEW_MAX_BYTES = 512 * 1024
_WORKSPACE_FILE_PREVIEW_MAX_CHARS = 400_000
_WORKSPACE_FILE_MAX_TABLE_ROWS = 200
_WORKSPACE_FILE_MAX_TABLE_COLS = 24
_WORKSPACE_FILE_MAX_CELL_CHARS = 120
_WORKSPACE_FILE_MAX_PDF_PAGES = 20
_WORKSPACE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
_WORKSPACE_DOC_EXTS = {".docx", ".pptx"}
_WORKSPACE_TEXT_EXTS = {
    ".txt", ".log", ".rst", ".md", ".mdx", ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx",
    ".css", ".scss", ".less", ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".toml", ".ini", ".cfg", ".conf", ".yaml", ".yml", ".xml",
    ".sql", ".c", ".h", ".cc", ".cpp", ".hpp", ".hh", ".json", ".jsonl", ".html", ".htm",
    ".diff", ".patch",
}
_WORKSPACE_LANGUAGE_BY_EXT = {
    ".txt": "text",
    ".log": "text",
    ".rst": "rst",
    ".md": "markdown",
    ".mdx": "markdown",
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".css": "css",
    ".scss": "scss",
    ".less": "lesscss",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".sql": "sql",
    ".json": "json",
    ".jsonl": "json",
    ".html": "html",
    ".htm": "html",
    ".diff": "diff",
    ".patch": "diff",
}


def _workspace_root_path(workspace: dict[str, Any]) -> Path:
    root = Path(str(workspace.get("canonical_path", "") or "")).expanduser().resolve()
    return root


def _resolve_workspace_relative_path(
    workspace: dict[str, Any],
    relpath: str = "",
) -> Path:
    workspace_root = _workspace_root_path(workspace)
    target = (workspace_root / str(relpath or "").strip()).resolve()
    target.relative_to(workspace_root)
    return target


def _safe_workspace_relpath(workspace_root: Path, target: Path) -> str:
    return str(target.relative_to(workspace_root)).replace("\\", "/")


def _file_modified_at(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
    except Exception:
        return ""


def _parent_relpath(path: str) -> str:
    normalized = str(path or "").replace("\\", "/").rstrip("/")
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def _truncate_preview_text(
    text: str,
    *,
    max_chars: int = _WORKSPACE_FILE_PREVIEW_MAX_CHARS,
) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _truncate_table_cell(text: str) -> str:
    if len(text) <= _WORKSPACE_FILE_MAX_CELL_CHARS:
        return text
    return text[: _WORKSPACE_FILE_MAX_CELL_CHARS - 3] + "..."


def _strip_html_text(content: str) -> str:
    import html as html_lib
    import re

    without_script = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    without_tags = re.sub(r"<[^>]+>", " ", without_script)
    return re.sub(r"\s+", " ", html_lib.unescape(without_tags)).strip()


def _is_probably_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    sample = data[:2048]
    non_text = sum(
        1 for byte in sample
        if byte < 9 or (13 < byte < 32) or byte == 127
    )
    return non_text > max(8, len(sample) // 10)


def _list_workspace_directory(
    workspace: dict[str, Any],
    relpath: str = "",
) -> list[WorkspaceFileEntryResponse]:
    directory = _resolve_workspace_relative_path(workspace, relpath)
    if not directory.exists():
        raise HTTPException(status_code=404, detail="Workspace directory does not exist.")
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail="Requested path is not a directory.")
    workspace_root = _workspace_root_path(workspace)
    rows: list[WorkspaceFileEntryResponse] = []
    try:
        children = sorted(
            directory.iterdir(),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list workspace files: {exc}",
        ) from exc
    for child in children:
        try:
            relative = _safe_workspace_relpath(workspace_root, child)
            stat = child.stat()
        except Exception:
            continue
        rows.append(
            WorkspaceFileEntryResponse(
                path=relative,
                name=child.name,
                is_dir=child.is_dir(),
                size_bytes=0 if child.is_dir() else int(stat.st_size or 0),
                modified_at=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                extension="" if child.is_dir() else child.suffix.lower(),
            ),
        )
    return rows


def _search_workspace_files(
    workspace: dict[str, Any],
    *,
    query: str,
    limit: int,
    scan_limit: int = 4000,
) -> list[WorkspaceSearchItemResponse]:
    workspace_root = _workspace_root_path(workspace)
    if not workspace_root.exists() or not workspace_root.is_dir():
        return []
    items: list[WorkspaceSearchItemResponse] = []
    scanned = 0
    for root, dirnames, filenames in os.walk(workspace_root):
        dirnames.sort(key=str.lower)
        filenames.sort(key=str.lower)
        for filename in filenames:
            scanned += 1
            if scanned > scan_limit:
                return items
            path = Path(root) / filename
            try:
                relpath = _safe_workspace_relpath(workspace_root, path)
                stat = path.stat()
            except Exception:
                continue
            parent_path = _parent_relpath(relpath)
            extension = path.suffix.lower()
            if not _text_matches_query(
                query,
                relpath,
                filename,
                extension,
                parent_path,
            ):
                continue
            items.append(
                WorkspaceSearchItemResponse(
                    kind="file",
                    item_id=relpath,
                    title=relpath,
                    subtitle=extension or "file",
                    snippet=_trim_snippet(parent_path),
                    badges=[
                        badge
                        for badge in [
                            extension or "",
                            _artifact_category(relpath) if extension else "",
                            f"{int(stat.st_size or 0)} bytes" if stat.st_size else "",
                        ]
                        if badge
                    ],
                    path=relpath,
                    metadata={
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime,
                            UTC,
                        ).isoformat(),
                    },
                ),
            )
            if len(items) >= limit:
                return items
    return items


def _preview_table_file(path: Path) -> WorkspaceFilePreviewTableResponse:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        rows: list[list[str]] = []
        truncated = False
        for index, row in enumerate(reader):
            if index > _WORKSPACE_FILE_MAX_TABLE_ROWS:
                truncated = True
                break
            rows.append([str(cell or "").strip() for cell in row])
    if not rows:
        return WorkspaceFilePreviewTableResponse(columns=[], rows=[], truncated=False)
    max_cols = min(max((len(row) for row in rows), default=0), _WORKSPACE_FILE_MAX_TABLE_COLS)
    header = rows[0]
    body = rows[1:]
    columns = [
        _truncate_table_cell(
            header[index]
            if index < len(header) and header[index]
            else f"col{index + 1}",
        )
        for index in range(max_cols)
    ]
    body_rows = [
        [
            _truncate_table_cell(row[index] if index < len(row) else "")
            for index in range(max_cols)
        ]
        for row in body
    ]
    if any(len(row) > max_cols for row in rows):
        truncated = True
    return WorkspaceFilePreviewTableResponse(columns=columns, rows=body_rows, truncated=truncated)


def _preview_pdf_file(path: Path) -> tuple[str, bool, dict[str, Any], str]:
    try:
        import pypdf
    except ImportError:
        return "", False, {}, "PDF preview unavailable because pypdf is not installed."
    try:
        reader = pypdf.PdfReader(path)
    except Exception as exc:
        return "", False, {}, f"Failed to read PDF: {exc}"
    total_pages = len(reader.pages)
    pages_to_read = min(total_pages, _WORKSPACE_FILE_MAX_PDF_PAGES)
    blocks: list[str] = []
    for index in range(pages_to_read):
        try:
            page_text = reader.pages[index].extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            blocks.append(f"--- Page {index + 1} ---\n{page_text.strip()}")
    text = (
        f"[PDF: {path.name}, {total_pages} pages, no extractable text]"
        if not blocks
        else "\n\n".join(blocks)
    )
    if total_pages > pages_to_read:
        text += f"\n\n[Showing first {pages_to_read} of {total_pages} pages.]"
    text, truncated = _truncate_preview_text(text)
    return text, truncated, {"page_count": total_pages}, ""


def _build_workspace_file_preview(
    workspace: dict[str, Any],
    relpath: str,
) -> WorkspaceFilePreviewResponse:
    target = _resolve_workspace_relative_path(workspace, relpath)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Workspace file does not exist.")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Requested path is not a file.")
    extension = target.suffix.lower()
    size_bytes = int(target.stat().st_size or 0)
    modified_at = _file_modified_at(target)

    if extension in {".csv", ".tsv"}:
        table = _preview_table_file(target)
        return WorkspaceFilePreviewResponse(
            path=relpath,
            name=target.name,
            extension=extension,
            size_bytes=size_bytes,
            modified_at=modified_at,
            preview_kind="table",
            table=table,
            truncated=table.truncated,
        )

    if extension == ".pdf":
        text, truncated, metadata, error = _preview_pdf_file(target)
        return WorkspaceFilePreviewResponse(
            path=relpath,
            name=target.name,
            extension=extension,
            size_bytes=size_bytes,
            modified_at=modified_at,
            preview_kind="document",
            text_content=text,
            metadata=metadata,
            truncated=truncated,
            error=error,
        )

    if extension == ".docx":
        try:
            text = extract_docx_text(target)
            error = ""
        except Exception as exc:
            text = ""
            error = f"Failed to read Word document: {exc}"
        text = text or f"[Word document: {target.name}, no extractable text]"
        text, truncated = _truncate_preview_text(text)
        return WorkspaceFilePreviewResponse(
            path=relpath,
            name=target.name,
            extension=extension,
            size_bytes=size_bytes,
            modified_at=modified_at,
            preview_kind="document",
            text_content=text,
            truncated=truncated,
            error=error,
        )

    if extension == ".pptx":
        try:
            text = extract_pptx_text(target)
            error = ""
        except Exception as exc:
            text = ""
            error = f"Failed to read PowerPoint file: {exc}"
        text = text or f"[PowerPoint: {target.name}, no extractable text]"
        text, truncated = _truncate_preview_text(text)
        return WorkspaceFilePreviewResponse(
            path=relpath,
            name=target.name,
            extension=extension,
            size_bytes=size_bytes,
            modified_at=modified_at,
            preview_kind="document",
            text_content=text,
            truncated=truncated,
            error=error,
        )

    if extension in _WORKSPACE_IMAGE_EXTS:
        width, height = get_image_dimensions(target)
        mime_type = mimetypes.guess_type(str(target))[0] or ""
        return WorkspaceFilePreviewResponse(
            path=relpath,
            name=target.name,
            extension=extension,
            size_bytes=size_bytes,
            modified_at=modified_at,
            preview_kind="image",
            metadata={
                "mime_type": mime_type,
                "width": width,
                "height": height,
            },
        )

    raw = target.read_bytes()[:_WORKSPACE_FILE_PREVIEW_MAX_BYTES]
    if _is_probably_binary(raw) and extension not in _WORKSPACE_TEXT_EXTS:
        return WorkspaceFilePreviewResponse(
            path=relpath,
            name=target.name,
            extension=extension,
            size_bytes=size_bytes,
            modified_at=modified_at,
            preview_kind="unsupported",
            error="Binary preview is not available for this file type yet.",
            metadata={"mime_type": mimetypes.guess_type(str(target))[0] or ""},
        )
    text = raw.decode("utf-8", errors="replace")
    if extension in {".html", ".htm"}:
        stripped = _strip_html_text(text)
        if stripped:
            text = stripped
    text, truncated = _truncate_preview_text(text)
    language = _WORKSPACE_LANGUAGE_BY_EXT.get(extension, "text")
    return WorkspaceFilePreviewResponse(
        path=relpath,
        name=target.name,
        extension=extension,
        size_bytes=size_bytes,
        modified_at=modified_at,
        preview_kind="text",
        language=language,
        text_content=text,
        truncated=truncated or size_bytes > _WORKSPACE_FILE_PREVIEW_MAX_BYTES,
    )


def _task_phase_ids(task: object) -> dict[str, str]:
    phase_ids: dict[str, str] = {}
    plan = getattr(task, "plan", None)
    subtasks = getattr(plan, "subtasks", []) if plan is not None else []
    for subtask in subtasks:
        subtask_id = str(getattr(subtask, "id", "") or "").strip()
        phase_id = str(getattr(subtask, "phase_id", "") or "").strip()
        if subtask_id and phase_id:
            phase_ids[subtask_id] = phase_id
    return phase_ids


def _normalized_timestamp_for_compare(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed.isoformat()


def _merge_artifact_seal_registries(*registries: object) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for registry in registries:
        if not isinstance(registry, dict):
            continue
        for raw_path, raw_seal in registry.items():
            relpath = str(raw_path or "").strip()
            if not relpath or not isinstance(raw_seal, dict):
                continue
            incoming = deepcopy(raw_seal)
            current = merged.get(relpath)
            if not isinstance(current, dict):
                merged[relpath] = incoming
                continue
            current_key = (
                _normalized_timestamp_for_compare(current.get("sealed_at")),
                str(current.get("sha256", "") or "").strip(),
            )
            incoming_key = (
                _normalized_timestamp_for_compare(incoming.get("sealed_at")),
                str(incoming.get("sha256", "") or "").strip(),
            )
            if incoming_key >= current_key:
                merged[relpath] = incoming
    return merged


def _merge_task_metadata(*metadata_sources: object) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    seal_registries: list[dict[str, Any]] = []
    saw_artifact_seals = False
    for source in metadata_sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if key == "artifact_seals":
                saw_artifact_seals = True
                if isinstance(value, dict):
                    seal_registries.append(value)
                continue
            merged[key] = deepcopy(value)
    if saw_artifact_seals:
        merged["artifact_seals"] = _merge_artifact_seal_registries(*seal_registries)
    return merged


async def _load_task_state(engine: Engine, task_id: str) -> Task | None:
    try:
        task = await asyncio.to_thread(engine.state_manager.load, task_id)
    except Exception:
        return None
    return task if isinstance(task, Task) else None


async def _task_state_exists(engine: Engine, task_id: str) -> bool:
    try:
        return bool(await asyncio.to_thread(engine.state_manager.exists, task_id))
    except Exception:
        return False


async def _save_task_state(engine: Engine, task: Task) -> None:
    await engine._save_task_state(task)


async def _persist_task_snapshot(engine: Engine, task: Task) -> None:
    await _save_task_state(engine, task)


def _task_row_projection_from_task(
    task: Task,
    *,
    base_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(base_row or {})
    status_value = (
        task.status.value
        if hasattr(task.status, "value")
        else str(task.status or TaskStatus.PENDING.value)
    )
    plan_json = (
        json.dumps(asdict(task.plan), ensure_ascii=False)
        if task.plan and (
            task.plan.subtasks
            or int(getattr(task.plan, "version", 1) or 1) != 1
            or str(getattr(task.plan, "last_replanned", "") or "").strip()
        )
        else None
    )
    row.update({
        "id": task.id,
        "goal": task.goal,
        "context": json.dumps(task.context, ensure_ascii=False) if task.context else None,
        "workspace_path": task.workspace,
        "status": status_value,
        "plan": plan_json,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "state_snapshot_updated_at": task.updated_at,
        "completed_at": str(task.completed_at or "").strip() or None,
        "approval_mode": task.approval_mode,
        "callback_url": task.callback_url or None,
        "metadata": json.dumps(task.metadata, ensure_ascii=False) if task.metadata else None,
    })
    return row


async def _load_task_snapshot_projection(
    engine: Engine,
    task_id: str,
) -> tuple[Task | None, dict[str, Any] | None]:
    task, task_row = await asyncio.gather(
        _load_task_state(engine, task_id),
        engine.database.get_task(task_id),
    )
    row_dict = dict(task_row) if task_row is not None else None
    if task is not None:
        return task, _task_row_projection_from_task(task, base_row=row_dict)
    return None, row_dict


async def _load_task_evidence_records(
    engine: Engine,
    task_id: str,
) -> list[dict[str, Any]]:
    try:
        records = await asyncio.to_thread(
            engine.state_manager.load_evidence_records,
            task_id,
        )
    except Exception:
        return []
    return [record for record in records if isinstance(record, dict)]


async def _build_run_artifacts(
    engine: Engine,
    task_row: dict[str, Any],
) -> list[RunArtifactResponse]:
    task_id = str(task_row.get("id", "") or "").strip()
    loaded_task = await _load_task_state(engine, task_id)
    metadata = _merge_task_metadata(
        _json_object(task_row.get("metadata")),
        loaded_task.metadata if loaded_task is not None else None,
    )
    seals = metadata.get("artifact_seals")
    if not isinstance(seals, dict):
        seals = {}
    evidence_records = await _load_task_evidence_records(engine, task_id)

    phase_by_subtask: dict[str, str] = {}
    if loaded_task is not None:
        phase_by_subtask = _task_phase_ids(loaded_task)

    artifact_rows: dict[str, dict[str, Any]] = {}

    for relpath, seal in seals.items():
        artifact_relpath = str(relpath or "").strip()
        if not artifact_relpath or not isinstance(seal, dict):
            continue
        artifact_key, workspace_visible_path, exists_on_disk = _resolve_artifact_locator(
            task_row,
            artifact_relpath,
            prefer_run_workspace=True,
        )
        subtask_ids: list[str] = []
        phase_ids: list[str] = []
        subtask_id = str(seal.get("subtask_id", "") or "").strip()
        _append_unique(subtask_ids, subtask_id)
        _append_unique(phase_ids, phase_by_subtask.get(subtask_id, ""))
        artifact_rows[artifact_key or artifact_relpath] = {
            "path": workspace_visible_path,
            "category": _artifact_category(artifact_relpath),
            "source": "seal",
            "sha256": str(seal.get("sha256", "") or ""),
            "size_bytes": int(seal.get("size_bytes", 0) or 0),
            "exists_on_disk": exists_on_disk,
            "is_intermediate": _artifact_is_intermediate(artifact_relpath),
            "created_at": str(seal.get("sealed_at", "") or ""),
            "tool_name": str(seal.get("tool", "") or ""),
            "subtask_ids": subtask_ids,
            "phase_ids": phase_ids,
            "facets": {},
        }

    for record in evidence_records:
        if not isinstance(record, dict):
            continue
        artifact_relpath = str(record.get("artifact_workspace_relpath", "") or "").strip()
        if not artifact_relpath:
            continue
        artifact_key, workspace_visible_path, exists_on_disk = _resolve_artifact_locator(
            task_row,
            artifact_relpath,
            prefer_run_workspace=artifact_relpath in seals,
        )
        facets = record.get("facets")
        if not isinstance(facets, dict):
            facets = {}
        row = artifact_rows.get(artifact_key or artifact_relpath)
        if row is None:
            row = {
                "path": workspace_visible_path,
                "category": _artifact_category(
                    artifact_relpath,
                    evidence_kind=record.get("evidence_kind"),
                    facets=facets,
                ),
                "source": "evidence",
                "sha256": "",
                "size_bytes": 0,
                "exists_on_disk": exists_on_disk,
                "is_intermediate": _artifact_is_intermediate(artifact_relpath),
                "created_at": "",
                "tool_name": "",
                "subtask_ids": [],
                "phase_ids": [],
                "facets": {},
            }
            artifact_rows[artifact_key or artifact_relpath] = row
        elif row.get("source") == "seal":
            row["source"] = "seal+evidence"

        evidence_sha = str(record.get("artifact_sha256", "") or "").strip()
        if evidence_sha:
            row["sha256"] = evidence_sha
        evidence_size = int(record.get("artifact_size_bytes", 0) or 0)
        if evidence_size > 0:
            row["size_bytes"] = evidence_size
        evidence_created_at = str(record.get("created_at", "") or "").strip()
        if evidence_created_at and evidence_created_at >= str(row.get("created_at", "") or ""):
            row["created_at"] = evidence_created_at
        tool_name = str(record.get("tool", "") or "").strip()
        if tool_name:
            row["tool_name"] = tool_name
        row["path"] = workspace_visible_path
        row["exists_on_disk"] = exists_on_disk
        evidence_category = _artifact_category(
            artifact_relpath,
            evidence_kind=record.get("evidence_kind"),
            facets=facets,
        )
        if evidence_category and (
            not str(row.get("category", "") or "").strip()
            or str(row.get("category", "") or "").strip() == "document"
        ):
            row["category"] = evidence_category
        merged_facets = row.get("facets")
        if not isinstance(merged_facets, dict):
            merged_facets = {}
        merged_facets.update(facets)
        row["facets"] = merged_facets
        _append_unique(row["subtask_ids"], record.get("subtask_id"))
        phase_id = str(record.get("phase_id", "") or "").strip()
        if not phase_id:
            phase_id = phase_by_subtask.get(str(record.get("subtask_id", "") or "").strip(), "")
        _append_unique(row["phase_ids"], phase_id)

    items = [
        RunArtifactResponse(
            path=str(row.get("path", "") or ""),
            category=str(row.get("category", "") or ""),
            source=str(row.get("source", "") or ""),
            sha256=str(row.get("sha256", "") or ""),
            size_bytes=int(row.get("size_bytes", 0) or 0),
            exists_on_disk=bool(row.get("exists_on_disk", False)),
            is_intermediate=bool(row.get("is_intermediate", False)),
            created_at=str(row.get("created_at", "") or ""),
            tool_name=str(row.get("tool_name", "") or ""),
            subtask_ids=[
                str(item or "").strip()
                for item in list(row.get("subtask_ids", []) or [])
                if str(item or "").strip()
            ],
            phase_ids=[
                str(item or "").strip()
                for item in list(row.get("phase_ids", []) or [])
                if str(item or "").strip()
            ],
            facets=(
                row.get("facets")
                if isinstance(row.get("facets"), dict)
                else {}
            ),
        )
        for row in artifact_rows.values()
        if str(row.get("path", "") or "").strip()
    ]
    items.sort(
        key=lambda item: (
            0 if not item.is_intermediate else 1,
            -int(bool(item.exists_on_disk)),
            -(1 if item.created_at else 0),
            item.created_at,
            item.path,
        ),
    )
    return items


async def _build_workspace_artifacts(
    engine: Engine,
    workspace: dict[str, Any],
) -> list[WorkspaceArtifactResponse]:
    tasks = await _workspace_tasks(engine, workspace)
    aggregated: dict[str, dict[str, Any]] = {}
    for task_row in tasks:
        run_id = str(task_row.get("id", "") or "").strip()
        if not run_id:
            continue
        for artifact in await _build_run_artifacts(engine, task_row):
            current = aggregated.get(artifact.path)
            if current is None:
                aggregated[artifact.path] = {
                    **artifact.model_dump(),
                    "latest_run_id": run_id,
                    "run_ids": [run_id],
                    "run_count": 1,
                }
                continue
            if run_id not in current["run_ids"]:
                current["run_ids"].append(run_id)
                current["run_count"] = len(current["run_ids"])
            current_created_at = str(current.get("created_at", "") or "")
            if artifact.created_at and artifact.created_at >= current_created_at:
                current["latest_run_id"] = run_id
                current["created_at"] = artifact.created_at
                current["source"] = artifact.source
                current["tool_name"] = artifact.tool_name
                current["sha256"] = artifact.sha256
                current["size_bytes"] = artifact.size_bytes
            current["exists_on_disk"] = bool(
                current.get("exists_on_disk", False) or artifact.exists_on_disk,
            )
            current["is_intermediate"] = bool(
                current.get("is_intermediate", False) or artifact.is_intermediate,
            )
            for subtask_id in artifact.subtask_ids:
                _append_unique(current["subtask_ids"], subtask_id)
            for phase_id in artifact.phase_ids:
                _append_unique(current["phase_ids"], phase_id)
            facets = current.get("facets")
            if not isinstance(facets, dict):
                facets = {}
            facets.update(artifact.facets)
            current["facets"] = facets

    items = [
        WorkspaceArtifactResponse(
            path=str(row.get("path", "") or ""),
            category=str(row.get("category", "") or ""),
            source=str(row.get("source", "") or ""),
            sha256=str(row.get("sha256", "") or ""),
            size_bytes=int(row.get("size_bytes", 0) or 0),
            exists_on_disk=bool(row.get("exists_on_disk", False)),
            is_intermediate=bool(row.get("is_intermediate", False)),
            created_at=str(row.get("created_at", "") or ""),
            tool_name=str(row.get("tool_name", "") or ""),
            subtask_ids=list(row.get("subtask_ids", []) or []),
            phase_ids=list(row.get("phase_ids", []) or []),
            facets=row.get("facets") if isinstance(row.get("facets"), dict) else {},
            latest_run_id=str(row.get("latest_run_id", "") or ""),
            run_ids=list(row.get("run_ids", []) or []),
            run_count=int(row.get("run_count", 0) or 0),
        )
        for row in aggregated.values()
        if str(row.get("path", "") or "").strip()
    ]
    items.sort(
        key=lambda item: (
            0 if not item.is_intermediate else 1,
            -int(bool(item.exists_on_disk)),
            -int(item.run_count or 0),
            -(1 if item.created_at else 0),
            item.created_at,
            item.path,
        ),
    )
    return items


async def _build_workspace_search_response(
    engine: Engine,
    workspace: dict[str, Any],
    *,
    query: str,
    limit_per_group: int,
) -> WorkspaceSearchResponse:
    summary = await _build_workspace_summary(engine, workspace)
    workspace_id = str(workspace.get("id", "") or "")
    workspace_display_name = str(summary.display_name or "")
    workspace_path = str(summary.canonical_path or "")
    clean_query = str(query or "").strip()
    limit = max(1, min(int(limit_per_group or 5), 10))

    def make_search_item(**kwargs: Any) -> WorkspaceSearchItemResponse:
        return WorkspaceSearchItemResponse(
            workspace_id=workspace_id,
            workspace_display_name=workspace_display_name,
            workspace_path=workspace_path,
            **kwargs,
        )

    tasks = await _workspace_tasks(engine, workspace)
    sessions = await _workspace_sessions(engine, workspace)
    approvals = await _list_pending_approval_items(engine, workspace_id=workspace_id)
    artifacts = await _build_workspace_artifacts(engine, workspace)
    file_rows = _search_workspace_files(
        workspace,
        query=clean_query,
        limit=limit,
    )
    file_items = [
        row.model_copy(update={
            "workspace_id": workspace_id,
            "workspace_display_name": workspace_display_name,
            "workspace_path": workspace_path,
        })
        for row in file_rows
    ]

    conversation_items: list[WorkspaceSearchItemResponse] = []
    for session in sessions[:25]:
        conversation = await _build_conversation_summary(engine, workspace_id, session)
        recent_turns = await engine.conversation_store.get_recent_turns(conversation.id, limit=40)
        conversation_search_text = _recent_conversation_search_text(list(reversed(recent_turns)))
        snippet = ""
        for turn in reversed(recent_turns):
            content = str(turn.get("content", "") or "").strip()
            if content:
                snippet = content
                break
        if not _text_matches_query(
            clean_query,
            conversation.title,
            conversation.model_name,
            snippet,
            conversation_search_text,
            " ".join(conversation.linked_run_ids),
        ):
            continue
        conversation_items.append(
            make_search_item(
                kind="conversation",
                item_id=conversation.id,
                title=conversation.title,
                subtitle=conversation.model_name or "default model",
                snippet=_trim_snippet(snippet or conversation.last_active_at),
                badges=[
                    badge
                    for badge in [
                        f"{conversation.turn_count} turns" if conversation.turn_count else "",
                        f"{conversation.total_tokens} tokens" if conversation.total_tokens else "",
                    ]
                    if badge
                ],
                conversation_id=conversation.id,
                metadata={"linked_run_ids": conversation.linked_run_ids},
            ),
        )
        if len(conversation_items) >= limit:
            break

    run_items: list[WorkspaceSearchItemResponse] = []
    for task_row in tasks[:25]:
        run = await _build_run_summary(engine, workspace_id, task_row)
        events = await engine.database.query_events(run.id, limit=12)
        snippet = ""
        for row in reversed(events):
            data = _json_object(row.get("data"))
            snippet = _trim_snippet(
                data.get("message")
                or data.get("summary")
                or data.get("error")
                or data.get("instruction")
                or "",
            )
            if snippet:
                break
        if not _text_matches_query(
            clean_query,
            run.goal,
            run.status,
            run.process_name,
            snippet,
            " ".join(run.linked_conversation_ids),
        ):
            continue
        run_items.append(
            make_search_item(
                kind="run",
                item_id=run.id,
                title=run.goal,
                subtitle=run.process_name or run.status,
                snippet=snippet,
                badges=[
                    badge
                    for badge in [
                        run.status,
                        f"{run.changed_files_count} files" if run.changed_files_count else "",
                    ]
                    if badge
                ],
                run_id=run.id,
                metadata={"linked_conversation_ids": run.linked_conversation_ids},
            ),
        )
        if len(run_items) >= limit:
            break

    approval_items: list[WorkspaceSearchItemResponse] = []
    for item in approvals:
        if not _text_matches_query(
            clean_query,
            item.title,
            item.summary,
            item.kind,
            item.tool_name,
            item.request_payload,
        ):
            continue
        approval_items.append(
            make_search_item(
                kind="approval",
                item_id=item.id,
                title=item.title,
                subtitle=item.kind.replace("_", " "),
                snippet=_trim_snippet(item.summary or item.request_payload),
                badges=[
                    badge
                    for badge in [item.tool_name, item.risk_level]
                    if badge
                ],
                conversation_id=item.conversation_id,
                run_id=item.task_id,
                approval_item_id=item.id,
                metadata={"request_payload": item.request_payload},
            ),
        )
        if len(approval_items) >= limit:
            break

    artifact_items: list[WorkspaceSearchItemResponse] = []
    for artifact in artifacts:
        if not _text_matches_query(
            clean_query,
            artifact.path,
            artifact.category,
            artifact.tool_name,
            artifact.latest_run_id,
            artifact.facets,
            " ".join(artifact.phase_ids),
            " ".join(artifact.subtask_ids),
        ):
            continue
        artifact_items.append(
            make_search_item(
                kind="artifact",
                item_id=artifact.path,
                title=artifact.path,
                subtitle=artifact.category or "artifact",
                snippet=_trim_snippet(artifact.facets),
                badges=[
                    badge
                    for badge in [
                        artifact.tool_name,
                        f"{artifact.run_count} runs" if artifact.run_count else "",
                        "intermediate" if artifact.is_intermediate else "",
                    ]
                    if badge
                ],
                run_id=artifact.latest_run_id,
                path=artifact.path,
                metadata={"run_ids": artifact.run_ids, "facets": artifact.facets},
            ),
        )
        if len(artifact_items) >= limit:
            break

    workspace_path = str(workspace.get("canonical_path", "") or "").strip()
    loader = ProcessLoader(
        workspace=Path(workspace_path).expanduser() if workspace_path else None,
        extra_search_paths=[
            Path(path).expanduser()
            for path in list(getattr(engine.config.process, "search_paths", []) or [])
        ],
        require_rule_scope_metadata=bool(
            getattr(engine.config.process, "require_rule_scope_metadata", False),
        ),
        require_v2_contract=bool(
            getattr(engine.config.process, "require_v2_contract", False),
        ),
    )
    process_items: list[WorkspaceSearchItemResponse] = []
    for row in loader.list_available():
        name = str(row.get("name", "") or "")
        description = str(row.get("description", "") or "")
        author = str(row.get("author", "") or "")
        path = str(row.get("path", "") or "")
        if not _text_matches_query(clean_query, name, description, author, path):
            continue
        process_items.append(
            make_search_item(
                kind="process",
                item_id=name,
                title=name,
                subtitle=description or path,
                snippet=_trim_snippet(author),
                badges=[badge for badge in [str(row.get("version", "") or "")] if badge],
                path=path,
            ),
        )
        if len(process_items) >= limit:
            break

    mcp_items: list[WorkspaceSearchItemResponse] = []
    for alias, server in sorted(engine.config.mcp.servers.items()):
        title = str(alias or "")
        subtitle = str(server.url or server.fallback_sse_url or server.command or server.type or "")
        if not _text_matches_query(
            clean_query,
            title,
            subtitle,
            server.type,
            server.cwd,
        ):
            continue
        mcp_items.append(
            make_search_item(
                kind="mcp_server",
                item_id=title,
                title=title,
                subtitle=subtitle,
                snippet=_trim_snippet(server.cwd),
                badges=[
                    badge
                    for badge in [
                        str(server.type or ""),
                        "oauth" if bool(getattr(server.oauth, "enabled", False)) else "",
                    ]
                    if badge
                ],
            ),
        )
        if len(mcp_items) >= limit:
            break

    tool_items: list[WorkspaceSearchItemResponse] = []
    for tool in _tool_info_rows(engine):
        if not _text_matches_query(
            clean_query,
            tool.name,
            tool.description,
            tool.auth_mode,
            " ".join(tool.execution_surfaces),
        ):
            continue
        tool_items.append(
            make_search_item(
                kind="tool",
                item_id=tool.name,
                title=tool.name,
                subtitle=tool.description,
                snippet=_trim_snippet(", ".join(tool.execution_surfaces)),
                badges=[
                    badge
                    for badge in [
                        tool.auth_mode,
                        "auth" if tool.auth_required else "",
                    ]
                    if badge
                ],
            ),
        )
        if len(tool_items) >= limit:
            break

    total_results = sum(
        len(group)
        for group in [
            [],
            conversation_items,
            run_items,
            approval_items,
            artifact_items,
            file_items,
            process_items,
            mcp_items,
            tool_items,
        ]
    )
    return WorkspaceSearchResponse(
        workspace=summary,
        query=clean_query,
        total_results=total_results,
        workspaces=[],
        conversations=conversation_items,
        runs=run_items,
        approvals=approval_items,
        artifacts=artifact_items,
        files=file_items,
        processes=process_items,
        mcp_servers=mcp_items,
        tools=tool_items,
    )


def _search_item_sort_key(
    item: WorkspaceSearchItemResponse,
    query: str,
) -> tuple[int, int, int, int, int, str]:
    clean_query = str(query or "").strip().lower()
    title = str(item.title or "").lower()
    subtitle = str(item.subtitle or "").lower()
    snippet = str(item.snippet or "").lower()
    workspace_label = str(item.workspace_display_name or "").lower()
    exact = 0 if title == clean_query and clean_query else 1
    title_prefix = 0 if clean_query and title.startswith(clean_query) else 1
    title_contains = 0 if clean_query and clean_query in title else 1
    subtitle_contains = 0 if clean_query and clean_query in subtitle else 1
    secondary_contains = 0 if clean_query and (
        clean_query in snippet
        or clean_query in workspace_label
    ) else 1
    return (
        exact,
        title_prefix,
        title_contains,
        subtitle_contains,
        secondary_contains,
        title,
    )


async def _build_global_search_response(
    engine: Engine,
    *,
    query: str,
    limit_per_group: int,
) -> WorkspaceSearchResponse:
    clean_query = str(query or "").strip()
    limit = max(1, min(int(limit_per_group or 5), 10))
    workspaces = await engine.workspace_registry.list(include_archived=False)

    workspace_items: list[WorkspaceSearchItemResponse] = []
    per_workspace_payloads: list[WorkspaceSearchResponse] = []

    for workspace in workspaces:
        payload = await _build_workspace_search_response(
            engine,
            workspace,
            query=clean_query,
            limit_per_group=limit,
        )
        per_workspace_payloads.append(payload)

        summary = payload.workspace
        if summary and _text_matches_query(
            clean_query,
            summary.display_name,
            summary.canonical_path,
            summary.workspace_type,
            summary.metadata,
        ):
            workspace_items.append(
                WorkspaceSearchItemResponse(
                    kind="workspace",
                    item_id=summary.id,
                    title=summary.display_name,
                    subtitle=summary.canonical_path,
                    snippet=_trim_snippet(summary.metadata),
                    badges=[
                        badge
                        for badge in [
                            (
                                f"{summary.conversation_count} threads"
                                if summary.conversation_count
                                else ""
                            ),
                            (
                                f"{summary.run_count} runs"
                                if summary.run_count
                                else ""
                            ),
                            "archived" if summary.is_archived else "",
                        ]
                        if badge
                    ],
                    workspace_id=summary.id,
                    workspace_display_name=summary.display_name,
                    workspace_path=summary.canonical_path,
                ),
            )

    def combined(group_name: str) -> list[WorkspaceSearchItemResponse]:
        rows: list[WorkspaceSearchItemResponse] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for payload in per_workspace_payloads:
            for item in list(getattr(payload, group_name, []) or []):
                if item.kind in {"tool", "mcp_server"}:
                    dedupe_key = (item.kind, item.item_id, "")
                else:
                    dedupe_key = (item.kind, item.item_id, item.workspace_id)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                rows.append(item)
        rows.sort(key=lambda item: _search_item_sort_key(item, clean_query))
        return rows[:limit]

    conversations = combined("conversations")
    runs = combined("runs")
    approvals = combined("approvals")
    artifacts = combined("artifacts")
    files = combined("files")
    processes = combined("processes")
    mcp_servers = combined("mcp_servers")
    tools = combined("tools")
    workspace_items.sort(key=lambda item: _search_item_sort_key(item, clean_query))
    workspace_items = workspace_items[:limit]

    total_results = sum(
        len(group)
        for group in [
            workspace_items,
            conversations,
            runs,
            approvals,
            artifacts,
            files,
            processes,
            mcp_servers,
            tools,
        ]
    )
    return WorkspaceSearchResponse(
        workspace=None,
        query=clean_query,
        total_results=total_results,
        workspaces=workspace_items,
        conversations=conversations,
        runs=runs,
        approvals=approvals,
        artifacts=artifacts,
        files=files,
        processes=processes,
        mcp_servers=mcp_servers,
        tools=tools,
    )


def _workspace_title_from_session(session: dict[str, Any]) -> str:
    session_state = _json_object(session.get("session_state"))
    raw_title = str(session_state.get("title", "") or "").strip()
    if raw_title:
        return raw_title
    session_id = str(session.get("id", "") or "").strip()
    if session_id:
        return f"Conversation {session_id[:8]}"
    return "Conversation"


async def _workspace_tasks(engine: Engine, workspace: dict[str, Any]) -> list[dict[str, Any]]:
    workspace_path = str(workspace.get("canonical_path", "") or "").strip()
    if not workspace_path:
        return []
    return await engine.database.list_tasks_for_workspace(workspace_path)


async def _workspace_sessions(engine: Engine, workspace: dict[str, Any]) -> list[dict[str, Any]]:
    workspace_path = str(workspace.get("canonical_path", "") or "").strip()
    if not workspace_path:
        return []
    return await engine.conversation_store.list_sessions(workspace=workspace_path)


_LIVE_TASK_STATUS_CANDIDATES = {"pending", "planning", "executing", "paused"}


def _task_workspace_scope(task_row: dict[str, Any]) -> str:
    metadata = _json_object(task_row.get("metadata"))
    return str(
        metadata.get("source_workspace_root")
        or task_row.get("workspace_path")
        or "",
    ).strip()


async def _workspace_relationship_maps(
    engine: Engine,
    *,
    task_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    task_ids = [
        str(row.get("id", "") or "").strip()
        for row in task_rows
        if str(row.get("id", "") or "").strip()
    ]
    session_ids = [
        str(row.get("id", "") or "").strip()
        for row in session_rows
        if str(row.get("id", "") or "").strip()
    ]
    latest_runs_by_task, linked_conversations_by_run, linked_runs_by_session = await asyncio.gather(
        engine.database.get_latest_task_runs_for_tasks(task_ids),
        engine.conversation_store.list_linked_conversations_for_runs(task_ids),
        engine.conversation_store.list_linked_runs_for_sessions(session_ids),
    )
    return latest_runs_by_task, linked_conversations_by_run, linked_runs_by_session


async def _count_pending_approval_items(
    engine: Engine,
    *,
    workspace: dict[str, Any] | None = None,
) -> int:
    workspace_path = str((workspace or {}).get("canonical_path", "") or "").strip()
    pending_task_approvals = engine.approval_manager.list_pending_approvals()
    total = 0
    if pending_task_approvals:
        task_rows_by_id = await engine.database.get_tasks_by_ids([
            str(getattr(item, "task_id", "") or "").strip()
            for item in pending_task_approvals
        ])
        for pending in pending_task_approvals:
            if workspace_path:
                task_row = task_rows_by_id.get(
                    str(getattr(pending, "task_id", "") or "").strip(),
                    {},
                )
                if _task_workspace_scope(task_row) != workspace_path:
                    continue
            total += 1

    if workspace_path:
        row = await engine.database.query_one(
            """
            SELECT COUNT(*) AS cnt
            FROM task_questions AS q
            JOIN tasks AS t ON t.id = q.task_id
            WHERE q.status = 'pending'
              AND COALESCE(
                    NULLIF(json_extract(COALESCE(t.metadata, '{}'), '$.source_workspace_root'), ''),
                    t.workspace_path
                  ) = ?
            """,
            (workspace_path,),
        )
    else:
        row = await engine.database.query_one(
            "SELECT COUNT(*) AS cnt FROM task_questions WHERE status = 'pending'",
        )
    total += int((row or {}).get("cnt", 0) or 0)

    pending_conversation_approvals = engine.list_pending_conversation_approvals()
    if pending_conversation_approvals:
        sessions_by_id = await engine.conversation_store.get_sessions_by_ids([
            str(item.get("conversation_id", "") or "").strip()
            for item in pending_conversation_approvals
            if str(item.get("conversation_id", "") or "").strip()
        ])
        for pending in pending_conversation_approvals:
            if workspace_path:
                session = sessions_by_id.get(
                    str(pending.get("conversation_id", "") or "").strip(),
                    {},
                )
                if str(session.get("workspace_path", "") or "").strip() != workspace_path:
                    continue
            total += 1

    return total


async def _build_conversation_summary(
    engine: Engine,
    workspace_id: str,
    session: dict[str, Any],
    *,
    linked_runs: list[dict[str, Any]] | None = None,
) -> ConversationSummaryResponse:
    linked_run_rows = (
        linked_runs
        if linked_runs is not None
        else await engine.conversation_store.list_linked_runs(str(session.get("id", "") or ""))
    )
    return ConversationSummaryResponse(
        id=str(session.get("id", "") or ""),
        workspace_id=workspace_id,
        workspace_path=str(session.get("workspace_path", "") or ""),
        model_name=str(session.get("model_name", "") or ""),
        title=_workspace_title_from_session(session),
        turn_count=int(session.get("turn_count", 0) or 0),
        total_tokens=int(session.get("total_tokens", 0) or 0),
        last_active_at=str(session.get("last_active_at", "") or ""),
        started_at=str(session.get("started_at", "") or ""),
        is_active=bool(session.get("is_active", 0)),
        linked_run_ids=[
            str(link.get("run_id", "") or "")
            for link in linked_run_rows
            if str(link.get("run_id", "") or "").strip()
        ],
    )


async def _build_run_summary(
    engine: Engine,
    workspace_id: str,
    task_row: dict[str, Any],
    *,
    latest_run: dict[str, Any] | None = None,
    linked_conversations: list[dict[str, Any]] | None = None,
    live_status: str | None = None,
    failure_analysis: RunFailureAnalysisResponse | None = None,
) -> RunSummaryResponse:
    task_id = str(task_row.get("id", "") or "").strip()
    metadata = _json_object(task_row.get("metadata"))
    latest_run_row = (
        latest_run
        if latest_run is not None
        else await engine.database.get_latest_task_run_for_task(task_id)
    )
    linked_conversation_rows = (
        linked_conversations
        if linked_conversations is not None
        else await engine.conversation_store.list_linked_conversations(task_id)
    )

    resolved_status = (
        live_status
        if live_status is not None
        else await _resolve_task_status(engine, task_row)
    )

    return RunSummaryResponse(
        id=task_id,
        workspace_id=workspace_id,
        workspace_path=str(task_row.get("workspace_path", "") or ""),
        goal=str(task_row.get("goal", "") or ""),
        status=resolved_status,
        created_at=str(task_row.get("created_at", "") or ""),
        updated_at=str(task_row.get("updated_at", "") or ""),
        execution_run_id=str((latest_run_row or {}).get("run_id", "") or ""),
        process_name=str(
            (latest_run_row or {}).get("process_name", "")
            or metadata.get("process", "")
            or "",
        ),
        linked_conversation_ids=[
            str(link.get("session_id", "") or "")
            for link in linked_conversation_rows
            if str(link.get("session_id", "") or "").strip()
        ],
        changed_files_count=0,
        failure_analysis=failure_analysis,
    )


async def _resolve_task_status(engine: Engine, task_row: dict[str, Any]) -> str:
    task_id = str(task_row.get("id", "") or "").strip()
    db_status = str(task_row.get("status", "") or "").strip().lower()
    if not task_id:
        return db_status
    if db_status and db_status not in _LIVE_TASK_STATUS_CANDIDATES:
        return db_status
    live_task = await _load_task_state(engine, task_id)
    if live_task is not None:
        return str(live_task.status.value or "").strip().lower()
    return db_status


async def _build_workspace_summary(
    engine: Engine,
    workspace: dict[str, Any],
    *,
    tasks: list[dict[str, Any]] | None = None,
    sessions: list[dict[str, Any]] | None = None,
) -> WorkspaceSummaryResponse:
    workspace_id = str(workspace.get("id", "") or "")
    task_rows = tasks if tasks is not None else await _workspace_tasks(engine, workspace)
    session_rows = (
        sessions
        if sessions is not None
        else await _workspace_sessions(engine, workspace)
    )
    last_activity_at = ""
    for row in task_rows:
        last_activity_at = _latest_timestamp(
            last_activity_at,
            row.get("updated_at"),
            row.get("completed_at"),
        )
    for row in session_rows:
        last_activity_at = _latest_timestamp(last_activity_at, row.get("last_active_at"))
    active_candidate_rows = [
        row
        for row in task_rows
        if str(row.get("status", "") or "").strip().lower() in _LIVE_TASK_STATUS_CANDIDATES
    ]
    resolved_candidate_statuses = await asyncio.gather(
        *(_resolve_task_status(engine, row) for row in active_candidate_rows),
    ) if active_candidate_rows else []
    resolved_status_by_task = {
        str(row.get("id", "") or "").strip(): status
        for row, status in zip(active_candidate_rows, resolved_candidate_statuses, strict=False)
        if str(row.get("id", "") or "").strip()
    }
    workspace_path = str(workspace.get("canonical_path", "") or "")
    return WorkspaceSummaryResponse(
        id=workspace_id,
        canonical_path=workspace_path,
        display_name=str(workspace.get("display_name", "") or ""),
        workspace_type=str(workspace.get("workspace_type", "local") or "local"),
        is_archived=bool(workspace.get("is_archived", False)),
        sort_order=int(workspace.get("sort_order", 0) or 0),
        last_opened_at=str(workspace.get("last_opened_at", "") or ""),
        created_at=str(workspace.get("created_at", "") or ""),
        updated_at=str(workspace.get("updated_at", "") or ""),
        metadata=(
            workspace.get("metadata")
            if isinstance(workspace.get("metadata"), dict)
            else {}
        ),
        exists_on_disk=Path(workspace_path).exists() if workspace_path else False,
        conversation_count=len(session_rows),
        run_count=len(task_rows),
        active_run_count=sum(
            1
            for row in task_rows
            if (
                resolved_status_by_task.get(str(row.get("id", "") or "").strip())
                or str(row.get("status", "") or "").strip().lower()
            ) in _LIVE_TASK_STATUS_CANDIDATES
        ),
        last_activity_at=last_activity_at,
    )


async def _require_workspace(engine: Engine, workspace_id: str) -> dict[str, Any]:
    workspace = await engine.workspace_registry.get(workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}")
    return workspace


def _tool_info_rows(engine: Engine) -> list[ToolInfo]:
    rows: list[ToolInfo] = []
    for schema in engine.tool_registry.all_schemas():
        name = str(schema.get("name", "") or "").strip()
        tool = engine.tool_registry.get(name) if name else None
        auth_requirements = getattr(tool, "auth_requirements", [])
        if not isinstance(auth_requirements, list):
            auth_requirements = []
        rows.append(
            ToolInfo(
                name=schema["name"],
                description=schema.get("description", ""),
                auth_mode=normalize_tool_auth_mode(
                    getattr(tool, "auth_mode", "no_auth"),
                ),
                auth_required=tool_auth_required(tool),
                auth_requirements=list(auth_requirements),
                execution_surfaces=list(normalize_tool_execution_surfaces(
                    schema.get("x_supported_execution_surfaces", []),
                )),
            ),
        )
    return rows


def _approval_feed_id(kind: str, *parts: object) -> str:
    tokens = [str(kind or "").strip()]
    tokens.extend(str(part or "").strip() for part in parts if str(part or "").strip())
    return ":".join(token for token in tokens if token)


async def _workspace_context_for_path(
    engine: Engine,
    workspace_path: object,
) -> tuple[str, str, str]:
    clean_path = canonicalize_workspace_path(workspace_path)
    if not clean_path:
        return "", "", ""
    workspace = await engine.workspace_registry.get_by_path(clean_path)
    if workspace is None:
        return "", clean_path, Path(clean_path).name
    return (
        str(workspace.get("id", "") or ""),
        str(workspace.get("canonical_path", clean_path) or clean_path),
        str(workspace.get("display_name", "") or Path(clean_path).name),
    )


async def _task_context(
    engine: Engine,
    task_id: str,
) -> tuple[dict[str, Any], str, str, str]:
    _, task_row = await _load_task_snapshot_projection(engine, task_id)
    task_row = dict(task_row or {})
    workspace_id, workspace_path, workspace_display_name = await _workspace_context_for_path(
        engine,
        _task_workspace_group_root(task_row),
    )
    return task_row, workspace_id, workspace_path, workspace_display_name


async def _conversation_context(
    engine: Engine,
    conversation_id: str,
) -> tuple[dict[str, Any], str, str, str]:
    session = await engine.conversation_store.get_session(conversation_id)
    session = dict(session or {})
    workspace_id, workspace_path, workspace_display_name = await _workspace_context_for_path(
        engine,
        session.get("workspace_path"),
    )
    return session, workspace_id, workspace_path, workspace_display_name


async def _build_task_approval_item(
    engine: Engine,
    request: object,
) -> ApprovalFeedItemResponse:
    task_id = str(getattr(request, "task_id", "") or "").strip()
    subtask_id = str(getattr(request, "subtask_id", "") or "").strip()
    task_row, workspace_id, workspace_path, workspace_display_name = await _task_context(
        engine,
        task_id,
    )
    proposed_action = str(getattr(request, "proposed_action", "") or "").strip()
    reason = str(getattr(request, "reason", "") or "").strip()
    title = proposed_action or reason or f"Approval for {subtask_id or task_id}"
    summary = reason or proposed_action or str(task_row.get("goal", "") or "").strip()
    details = getattr(request, "details", {})
    return ApprovalFeedItemResponse(
        id=_approval_feed_id("task", task_id, subtask_id),
        kind="task_approval",
        status="pending",
        created_at=str(getattr(request, "created_at", "") or ""),
        title=title,
        summary=summary,
        workspace_id=workspace_id,
        workspace_path=workspace_path,
        workspace_display_name=workspace_display_name,
        task_id=task_id,
        run_id=task_id,
        subtask_id=subtask_id,
        risk_level=str(getattr(request, "risk_level", "") or "").strip(),
        request_payload={
            "reason": reason,
            "proposed_action": proposed_action,
            "details": details if isinstance(details, dict) else {},
            "auto_approve_timeout": getattr(request, "auto_approve_timeout", None),
        },
        metadata={
            "task_goal": str(task_row.get("goal", "") or "").strip(),
        },
    )


async def _build_task_question_item(
    engine: Engine,
    row: dict[str, Any],
) -> ApprovalFeedItemResponse:
    task_id = str(row.get("task_id", "") or "").strip()
    question_id = str(row.get("question_id", "") or "").strip()
    task_row, workspace_id, workspace_path, workspace_display_name = await _task_context(
        engine,
        task_id,
    )
    request_payload = row.get("request_payload")
    if not isinstance(request_payload, dict):
        request_payload = {}
    question = str(request_payload.get("question", "") or "").strip()
    context_note = str(request_payload.get("context_note", "") or "").strip()
    return ApprovalFeedItemResponse(
        id=_approval_feed_id("question", task_id, question_id),
        kind="task_question",
        status=str(row.get("status", "pending") or "pending"),
        created_at=str(row.get("created_at", "") or ""),
        title=question or f"Question for {task_id}",
        summary=context_note or str(task_row.get("goal", "") or "").strip(),
        workspace_id=workspace_id,
        workspace_path=workspace_path,
        workspace_display_name=workspace_display_name,
        task_id=task_id,
        run_id=task_id,
        subtask_id=str(row.get("subtask_id", "") or "").strip(),
        question_id=question_id,
        request_payload=request_payload,
        metadata={
            "task_goal": str(task_row.get("goal", "") or "").strip(),
        },
    )


async def _build_conversation_approval_item(
    engine: Engine,
    row: dict[str, Any],
) -> ApprovalFeedItemResponse:
    conversation_id = str(row.get("conversation_id", "") or "").strip()
    approval_id = str(row.get("approval_id", "") or "").strip()
    session, workspace_id, workspace_path, workspace_display_name = await _conversation_context(
        engine,
        conversation_id,
    )
    args = row.get("args")
    if not isinstance(args, dict):
        args = {}
    risk_info = row.get("risk_info")
    if not isinstance(risk_info, dict):
        risk_info = {}
    preview = (
        str(args.get("command", "") or "").strip()
        or str(args.get("path", "") or "").strip()
        or str(args.get("question", "") or "").strip()
        or str(args.get("text", "") or "").strip()
    )
    return ApprovalFeedItemResponse(
        id=_approval_feed_id("conversation", conversation_id, approval_id),
        kind="conversation_approval",
        status="pending",
        created_at=str(row.get("created_at", "") or ""),
        title=f"{str(row.get('tool_name', '') or 'Tool').strip()} approval",
        summary=preview or _workspace_title_from_session(session),
        workspace_id=workspace_id,
        workspace_path=workspace_path,
        workspace_display_name=workspace_display_name,
        conversation_id=conversation_id,
        approval_id=approval_id,
        tool_name=str(row.get("tool_name", "") or "").strip(),
        risk_level=str(risk_info.get("risk_level", "") or "").strip(),
        request_payload=args,
        metadata={
            "risk_info": risk_info,
            "conversation_title": _workspace_title_from_session(session),
        },
    )


async def _list_pending_approval_items(
    engine: Engine,
    *,
    workspace_id: str = "",
) -> list[ApprovalFeedItemResponse]:
    items: list[ApprovalFeedItemResponse] = []

    for pending in engine.approval_manager.list_pending_approvals():
        item = await _build_task_approval_item(engine, pending)
        if workspace_id and item.workspace_id != workspace_id:
            continue
        items.append(item)

    question_rows = await engine.database.query(
        """
        SELECT *
        FROM task_questions
        WHERE status = 'pending'
        ORDER BY created_at DESC
        """,
    )
    for row in question_rows:
        item = await _build_task_question_item(engine, row)
        if workspace_id and item.workspace_id != workspace_id:
            continue
        items.append(item)

    for row in engine.list_pending_conversation_approvals():
        item = await _build_conversation_approval_item(engine, row)
        if workspace_id and item.workspace_id != workspace_id:
            continue
        items.append(item)

    items.sort(key=lambda item: item.created_at, reverse=True)
    return items


async def _notification_payload_from_event(
    engine: Engine,
    event: Event,
) -> dict[str, Any] | None:
    data = event.data if isinstance(event.data, dict) else {}
    conversation_id = str(data.get("conversation_id", "") or "").strip()
    if conversation_id:
        session, workspace_id, workspace_path, workspace_display_name = await _conversation_context(
            engine,
            conversation_id,
        )
        return {
            "id": str(data.get("event_id", "") or ""),
            "stream_id": None,
            "event_type": event.event_type,
            "created_at": event.timestamp,
            "workspace_id": workspace_id,
            "workspace_path": workspace_path,
            "workspace_display_name": workspace_display_name,
            "conversation_id": conversation_id,
            "approval_id": str(data.get("approval_id", "") or ""),
            "task_id": "",
            "kind": "conversation_approval",
            "title": str(event.event_type or "").replace("_", " "),
            "summary": str(data.get("tool_name", "") or "").strip()
            or _workspace_title_from_session(session),
            "payload": data,
        }

    task_row, workspace_id, workspace_path, workspace_display_name = await _task_context(
        engine,
        event.task_id,
    )
    kind = "task_notification"
    title = str(event.event_type or "").replace("_", " ")
    if event.event_type in {APPROVAL_REQUESTED, APPROVAL_RECEIVED}:
        kind = "task_approval"
        title = "Task approval"
    elif event.event_type in {
        ASK_USER_REQUESTED,
        ASK_USER_ANSWERED,
        ASK_USER_TIMEOUT,
        ASK_USER_CANCELLED,
    }:
        kind = "task_question"
        title = "Task question"
    return {
        "id": str(data.get("event_id", "") or ""),
        "stream_id": None,
        "event_type": event.event_type,
        "created_at": event.timestamp,
        "workspace_id": workspace_id,
        "workspace_path": workspace_path,
        "workspace_display_name": workspace_display_name,
        "task_id": event.task_id,
        "conversation_id": "",
        "approval_id": "",
        "kind": kind,
        "title": title,
        "summary": str(data.get("question", "") or "").strip()
        or str(data.get("proposed_action", "") or "").strip()
        or str(task_row.get("goal", "") or "").strip(),
        "payload": data,
    }


async def _notification_payload_from_row(
    engine: Engine,
    row: dict[str, Any],
) -> dict[str, Any] | None:
    payload = await _notification_payload_from_event(
        engine,
        Event(
            event_type=str(row.get("event_type", "") or ""),
            task_id=str(row.get("task_id", "") or ""),
            data=_json_object(row.get("data")),
            timestamp=str(row.get("timestamp", "") or ""),
        ),
    )
    if payload is None:
        return None
    payload["stream_id"] = int(row.get("id", 0) or 0)
    return payload


async def _resolve_notification_stream_cursor(
    engine: Engine,
    *,
    after_id: int,
    last_event_id: str,
) -> int:
    cursor = max(0, int(after_id))
    header_cursor = str(last_event_id or "").strip()
    if not header_cursor:
        return cursor
    if header_cursor.isdigit():
        return max(cursor, int(header_cursor))
    if ":" not in header_cursor:
        return cursor
    _, event_id = header_cursor.split(":", 1)
    clean_event_id = str(event_id or "").strip()
    if not clean_event_id:
        return cursor
    row = await engine.database.query_one(
        "SELECT id FROM events WHERE event_id = ?",
        (clean_event_id,),
    )
    if row is None:
        return cursor
    return max(cursor, int(row.get("id", 0) or 0))


def _notification_resume_event_id(last_event_id: str) -> str:
    header_cursor = str(last_event_id or "").strip()
    if not header_cursor.startswith("event:"):
        return ""
    return header_cursor.removeprefix("event:").strip()


async def _notification_history_payloads(
    engine: Engine,
    *,
    relevant_events: set[str],
    workspace_filter: str,
    after_event_id: str,
    seen_event_ids: set[str],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    armed = not after_event_id
    for event in engine.event_bus.recent_events(limit=1000):
        if event.event_type not in relevant_events:
            continue
        if not should_deliver_operator(
            event.event_type,
            engine.effective_telemetry_mode(),
        ):
            continue
        payload_data = event.data if isinstance(event.data, dict) else {}
        event_id = str(payload_data.get("event_id", "") or "").strip()
        if after_event_id and not armed:
            if event_id == after_event_id:
                armed = True
            continue
        payload = await _notification_payload_from_event(engine, event)
        if payload is None:
            continue
        if workspace_filter and str(payload.get("workspace_id", "") or "") != workspace_filter:
            continue
        payload_id = str(payload.get("id", "") or "").strip()
        if payload_id and payload_id in seen_event_ids:
            continue
        if payload_id:
            seen_event_ids.add(payload_id)
        payloads.append(payload)
    return payloads


def _settings_payload(engine: Engine) -> dict[str, Any]:
    entries_payload: list[dict[str, Any]] = []
    for entry in list_entries():
        snapshot = dict(engine.config_runtime_store.snapshot(entry.path))
        entries_payload.append(snapshot)
    return {
        "basic": [item for item in entries_payload if item.get("exposure_level") == "basic"],
        "advanced": [
            item for item in entries_payload if item.get("exposure_level") != "basic"
        ],
        "updated_at": _latest_timestamp(*(item.get("updated_at") for item in entries_payload)),
    }


def _conversation_model_name(session: dict[str, Any]) -> str:
    configured = str(session.get("model_name", "") or "").strip()
    if configured:
        return configured
    return ""


def _resolve_cowork_model(engine: Engine, model_name: str):
    clean_name = str(model_name or "").strip()
    if clean_name:
        provider = engine.model_router.get(clean_name)
        if provider is not None:
            return provider
    return engine.model_router.select(role="executor")


def _build_api_cowork_session(engine: Engine, session: dict[str, Any]) -> CoworkSession:
    model = _resolve_cowork_model(engine, _conversation_model_name(session))
    config = engine.config
    execution = config.execution
    scratch_dir = Path(str(config.workspace.scratch_dir or "")).expanduser()
    session_cls = _cowork_session_cls()
    conversation_id = str(session.get("id", "") or "")

    async def approval_callback(tool_name: str, args: dict) -> CoworkApprovalDecision:
        request = engine.begin_conversation_approval(
            conversation_id,
            tool_name=tool_name,
            args=args,
        )
        engine.event_bus.emit(Event(
            event_type=APPROVAL_REQUESTED,
            task_id=conversation_id,
            data={
                "conversation_id": conversation_id,
                "workspace_path": str(session.get("workspace_path", "") or ""),
                "approval_id": request.approval_id,
                "tool_name": request.tool_name,
                "risk_info": request.to_dict().get("risk_info"),
                "source_component": "cowork_api",
            },
        ))
        await _append_conversation_replay_event(
            engine,
            conversation_id,
            "approval_requested",
            request.to_dict(),
        )
        decision = await engine.wait_for_conversation_approval(
            conversation_id,
            request.approval_id,
        )
        engine.event_bus.emit(Event(
            event_type=APPROVAL_RECEIVED,
            task_id=conversation_id,
            data={
                "conversation_id": conversation_id,
                "workspace_path": str(session.get("workspace_path", "") or ""),
                "approval_id": request.approval_id,
                "tool_name": request.tool_name,
                "decision": decision.value,
                "source_component": "cowork_api",
            },
        ))
        await _append_conversation_replay_event(
            engine,
            conversation_id,
            "approval_resolved",
            {
                **request.to_dict(),
                "decision": decision.value,
            },
        )
        return decision

    return session_cls(
        model=model,
        tools=engine.tool_registry,
        workspace=Path(str(session.get("workspace_path", "") or "")).expanduser(),
        scratch_dir=scratch_dir,
        system_prompt=(
            str(session.get("system_prompt", "") or "").strip()
            or build_cowork_system_prompt(
                workspace=Path(str(session.get("workspace_path", "") or "")).expanduser()
                if str(session.get("workspace_path", "") or "").strip()
                else None,
            )
        ),
        approver=ToolApprover(prompt_callback=approval_callback),
        store=engine.conversation_store,
        session_id=str(session.get("id", "") or ""),
        max_context_tokens=int(getattr(execution, "cowork_max_context_tokens", 32_000)),
        tool_exposure_mode=str(getattr(execution, "cowork_tool_exposure_mode", "hybrid")),
        enable_filetype_ingest_router=bool(
            getattr(execution, "enable_filetype_ingest_router", True),
        ),
        ingest_artifact_retention_max_age_days=int(
            getattr(execution, "ingest_artifact_retention_max_age_days", 14),
        ),
        ingest_artifact_retention_max_files_per_scope=int(
            getattr(execution, "ingest_artifact_retention_max_files_per_scope", 96),
        ),
        ingest_artifact_retention_max_bytes_per_scope=int(
            getattr(execution, "ingest_artifact_retention_max_bytes_per_scope", 268_435_456),
        ),
        memory_index_enabled=bool(getattr(execution, "cowork_memory_index_enabled", True)),
        memory_index_llm_extraction_enabled=bool(
            getattr(execution, "cowork_memory_index_llm_extraction_enabled", True),
        ),
        memory_index_role_strict=bool(
            getattr(execution, "cowork_indexer_model_role_strict", False),
        ),
        memory_index_queue_max_batches=int(
            getattr(execution, "cowork_memory_index_queue_max_batches", 32),
        ),
        memory_index_section_limit=int(
            getattr(execution, "cowork_memory_index_section_limit", 4),
        ),
        recall_index_max_chars=int(
            getattr(execution, "cowork_recall_index_max_chars", 1200),
        ),
    )


async def _append_conversation_replay_event(
    engine: Engine,
    conversation_id: str,
    event_type: str,
    payload: dict[str, Any],
    *,
    journal_through_turn: int | None = None,
) -> None:
    seq = await engine.conversation_store.append_chat_event(
        conversation_id,
        event_type,
        payload,
        journal_through_turn=journal_through_turn,
    )
    # Emit through event bus for instant SSE delivery
    engine.event_bus.emit(Event(
        event_type=CONVERSATION_MESSAGE,
        task_id=conversation_id,
        data={
            "conversation_id": conversation_id,
            "chat_event_type": event_type,
            "seq": seq,
            "payload": payload,
        },
    ))


def _normalize_ask_user_prompt_payload(payload: object) -> dict[str, Any] | None:
    """Coerce ask_user args/result payloads into one stable API shape."""
    if not isinstance(payload, dict):
        return None
    candidate = dict(payload)
    options_v2 = candidate.get("options_v2")
    if isinstance(options_v2, list) and options_v2:
        candidate["options"] = options_v2
    normalized = normalize_ask_user_args(candidate)
    question = str(normalized.get("question", "") or "").strip()
    if not question:
        return None
    return normalized


def _conversation_pending_prompt_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a pending ask_user prompt from one replay journal event."""
    if str(event.get("event_type", "") or "") != "tool_call_completed":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    if str(payload.get("tool_name", "") or "").strip() != "ask_user":
        return None
    if not bool(payload.get("success", False)):
        return None

    result_data = payload.get("data")
    if isinstance(result_data, dict) and "awaiting_input" in result_data:
        if not bool(result_data.get("awaiting_input", False)):
            return None

    prompt = (
        _normalize_ask_user_prompt_payload(payload.get("question_payload"))
        or _normalize_ask_user_prompt_payload(result_data)
        or _normalize_ask_user_prompt_payload(payload.get("args"))
    )
    if prompt is None:
        return None

    tool_call_id = str(payload.get("tool_call_id", "") or "").strip()
    if tool_call_id:
        prompt = {**prompt, "tool_call_id": tool_call_id}
    return prompt


def _conversation_pending_prompt_from_turn(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a pending ask_user prompt from the canonical turn log."""
    if str(row.get("role", "") or "").strip().lower() != "tool":
        return None
    if str(row.get("tool_name", "") or "").strip() != "ask_user":
        return None
    result = ToolResult.from_json(str(row.get("content", "") or ""))
    if not result.success or not isinstance(result.data, dict):
        return None
    if "awaiting_input" in result.data and not bool(result.data.get("awaiting_input", False)):
        return None
    prompt = (
        _normalize_ask_user_prompt_payload(result.data)
        or _normalize_ask_user_prompt_payload(getattr(result, "args", None))
    )
    if prompt is None:
        return None
    tool_call_id = str(row.get("tool_call_id", "") or "").strip()
    if tool_call_id:
        prompt = {**prompt, "tool_call_id": tool_call_id}
    return prompt


def _normalize_conversation_approval_decision(raw: object) -> CoworkApprovalDecision:
    value = str(raw or "").strip().lower()
    if value == CoworkApprovalDecision.APPROVE_ALL.value:
        return CoworkApprovalDecision.APPROVE_ALL
    if value == CoworkApprovalDecision.DENY.value:
        return CoworkApprovalDecision.DENY
    return CoworkApprovalDecision.APPROVE


_CONVERSATION_PREFIX_EVENT_TYPES = {"assistant_text", "assistant_thinking"}
_CONVERSATION_PREFIX_EXTRA_ROW_BUDGET = 64
_CONVERSATION_PREFIX_FETCH_BATCH = 32


async def _expand_conversation_event_page_prefix(
    engine: Engine,
    conversation_id: str,
    rows: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Ensure a history page does not begin mid-assistant transcript run.

    Streamed assistant replies are persisted as many small replay rows. If a
    paged history fetch starts on one of those rows, the desktop transcript can
    render incomplete Markdown until older rows hydrate. When the first row is
    assistant text/thinking, we pull earlier rows until the page starts at a
    stable boundary or we exhaust a bounded prefix budget.
    """
    expanded = list(rows)
    extra_rows = 0
    while expanded:
        first = expanded[0]
        event_type = str(first.get("event_type", "") or "").strip()
        if event_type not in _CONVERSATION_PREFIX_EVENT_TYPES:
            break
        first_seq = int(first.get("seq", 0) or 0)
        if first_seq <= 1:
            break
        remaining_budget = _CONVERSATION_PREFIX_EXTRA_ROW_BUDGET - extra_rows
        if remaining_budget <= 0:
            break
        older = await engine.conversation_store.get_chat_events(
            conversation_id,
            before_seq=first_seq,
            limit=min(limit, _CONVERSATION_PREFIX_FETCH_BATCH, remaining_budget),
        )
        if not older:
            break
        expanded = [*older, *expanded]
        extra_rows += len(older)
    return expanded


async def _conversation_pending_prompt(
    engine: Engine,
    conversation_id: str,
) -> dict[str, Any] | None:
    """Return the last unresolved ask_user prompt for a conversation."""
    rows = await engine.conversation_store.get_recent_turns(conversation_id, limit=64)
    for row in reversed(rows):
        if str(row.get("role", "") or "").strip().lower() == "user":
            return None
        prompt = _conversation_pending_prompt_from_turn(row)
        if prompt is not None:
            return prompt
    return None


async def _run_cowork_turn_for_api(
    engine: Engine,
    *,
    conversation_id: str,
    session: CoworkSession,
    user_message: str,
) -> None:
    try:
        await session.resume(conversation_id)
        await _append_conversation_replay_event(
            engine,
            conversation_id,
            "user_message",
            {"text": user_message},
        )
        streamed_text = False
        async for event in session.send_streaming(user_message):
            if isinstance(event, tuple) and len(event) == 2 and event[0] == "thinking":
                thinking_text = str(event[1] or "")
                if thinking_text:
                    await _append_conversation_replay_event(
                        engine,
                        conversation_id,
                        "assistant_thinking",
                        {"text": thinking_text, "streaming": True},
                    )
                continue
            if isinstance(event, str):
                if event:
                    streamed_text = True
                    await _append_conversation_replay_event(
                        engine,
                        conversation_id,
                        "assistant_text",
                        {"text": event, "streaming": True},
                    )
                continue

            if isinstance(event, ToolCallEvent):
                if event.result is None:
                    await _append_conversation_replay_event(
                        engine,
                        conversation_id,
                        "tool_call_started",
                        {
                            "tool_name": event.name,
                            "tool_call_id": event.tool_call_id,
                            "args": dict(event.args or {}),
                        },
                    )
                    continue

                output = event.result.output if event.result.success else ""
                error = event.result.error or ""
                payload: dict[str, Any] = {
                    "tool_name": event.name,
                    "tool_call_id": event.tool_call_id,
                    "args": dict(event.args or {}),
                    "success": bool(event.result.success),
                    "elapsed_ms": int(event.elapsed_ms or 0),
                    "output": output,
                    "error": error,
                }
                if isinstance(event.result.data, dict):
                    payload["data"] = dict(event.result.data)
                if event.name == "ask_user":
                    prompt = (
                        _normalize_ask_user_prompt_payload(event.result.data)
                        or _normalize_ask_user_prompt_payload(event.args)
                    )
                    if prompt is not None:
                        payload["question_payload"] = prompt
                await _append_conversation_replay_event(
                    engine,
                    conversation_id,
                    "tool_call_completed",
                    payload,
                )
                if event.result.success and event.result.content_blocks:
                    await _append_conversation_replay_event(
                        engine,
                        conversation_id,
                        "content_indicator",
                        {
                            "content_blocks": [
                                serialize_block(block)
                                for block in event.result.content_blocks
                            ],
                        },
                    )
                continue

            if isinstance(event, CoworkTurn):
                if event.text and not streamed_text:
                    # Only emit full text if we didn't already stream it incrementally
                    await _append_conversation_replay_event(
                        engine,
                        conversation_id,
                        "assistant_text",
                        {
                            "text": event.text,
                            "markup": False,
                        },
                    )
                await _append_conversation_replay_event(
                    engine,
                    conversation_id,
                    "turn_separator",
                    {
                        "tool_count": len(event.tool_calls),
                        "tokens": int(event.tokens_used),
                        "model": event.model,
                        "tokens_per_second": float(event.tokens_per_second),
                        "latency_ms": int(event.latency_ms),
                        "total_time_ms": int(event.total_time_ms),
                        "context_tokens": int(event.context_tokens),
                        "context_messages": int(event.context_messages),
                        "omitted_messages": int(event.omitted_messages),
                        "recall_index_used": bool(event.recall_index_used),
                    },
                    journal_through_turn=int(
                        getattr(session, "persisted_turn_count", 0) or 0
                    ),
                )
    except CoworkStopRequestedError as exc:
        await _append_conversation_replay_event(
            engine,
            conversation_id,
            "turn_interrupted",
            {
                "message": f"Conversation turn stopped ({exc.reason}).",
                "reason": exc.reason,
                "stage": exc.stage,
                "markup": False,
            },
        )
    except Exception as exc:
        logger.exception("Conversation turn failed for %s", conversation_id)
        await _append_conversation_replay_event(
            engine,
            conversation_id,
            "assistant_text",
            {
                "text": f"Conversation turn failed: {type(exc).__name__}: {exc}",
                "markup": False,
            },
        )


# --- Task Lifecycle ---


@router.post("/tasks", response_model=TaskCreateResponse, status_code=201)
async def create_new_task(request: Request, body: TaskCreateRequest):
    """Create and start a new task."""
    started = time.monotonic()
    engine = _get_engine(request)
    effective_process = body.process or engine.config.process.default or None
    requested_workspace = str(body.workspace or "").strip()
    if requested_workspace:
        requested_workspace = (
            canonicalize_workspace_path(requested_workspace) or requested_workspace
        )

    # Validate workspace if provided
    if requested_workspace:
        valid, msg = validate_workspace(requested_workspace)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)

    # Resolve process definition if specified
    process_def = None
    if effective_process:
        load_started = time.monotonic()
        from loom.processes.schema import ProcessLoader, ProcessNotFoundError

        extra = [Path(p) for p in engine.config.process.search_paths]
        loader = ProcessLoader(
            workspace=Path(requested_workspace) if requested_workspace else None,
            extra_search_paths=extra,
            require_rule_scope_metadata=bool(
                getattr(engine.config.process, "require_rule_scope_metadata", False),
            ),
            require_v2_contract=bool(
                getattr(engine.config.process, "require_v2_contract", False),
            ),
        )
        try:
            process_def = await asyncio.to_thread(loader.load, effective_process)
            log_latency_event(
                logger,
                event="api_task_process_load",
                duration_seconds=time.monotonic() - load_started,
                fields={"process": effective_process},
            )
        except ProcessNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    registry_for_process: object | None = None
    # Validate required tools early so clients get immediate feedback.
    if process_def is not None:
        preflight_started = time.monotonic()
        tools_cfg = getattr(process_def, "tools", None)
        required = list(getattr(tools_cfg, "required", []) or [])
        excluded = set(getattr(tools_cfg, "excluded", []) or [])
        from loom.tools import create_default_registry

        registry_for_process = await asyncio.to_thread(
            create_default_registry,
            engine.config,
        )
        if required:
            available = (
                set(await asyncio.to_thread(registry_for_process.list_tools))
                - excluded
            )
            missing = sorted(name for name in required if name not in available)
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Process '{getattr(process_def, 'name', body.process)}' "
                        f"requires missing tool(s): {', '.join(missing)}"
                    ),
                )
        log_latency_event(
            logger,
            event="api_task_tool_preflight",
            duration_seconds=time.monotonic() - preflight_started,
            fields={"required": len(required)},
        )

    metadata = body.metadata if isinstance(body.metadata, dict) else {}
    metadata = dict(metadata)
    task_workspace = requested_workspace

    if body.auto_subfolder and requested_workspace:
        try:
            scoped_workspace, relative_folder = provision_scoped_run_workspace(
                Path(requested_workspace),
                process_name=str(effective_process or ""),
                goal=body.goal,
            )
        except OSError as error:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to prepare run workspace: {error}",
            ) from error
        task_workspace = str(scoped_workspace)
        metadata["source_workspace_root"] = requested_workspace
        metadata["run_workspace_relative"] = relative_folder
        metadata["run_workspace_mode"] = "scoped_subfolder"
        attached_read_roots, attached_read_path_map = build_attached_read_scope(
            requested_workspace,
            body.context if isinstance(body.context, dict) else {},
        )
        if attached_read_roots:
            metadata["read_roots"] = attached_read_roots
        if attached_read_path_map:
            metadata["attached_read_path_map"] = attached_read_path_map

    # API-created tasks always run on the API surface. Do not permit callers
    # to elevate to interactive-only surfaces (for example, tui).
    metadata["execution_surface"] = "api"
    metadata["process"] = effective_process or ""
    required_auth_resources = await asyncio.to_thread(
        _required_auth_resources_for_process,
        process_def,
        tool_registry=registry_for_process,
    )
    if required_auth_resources:
        metadata["auth_required_resources"] = required_auth_resources

    # Validate auth profile selection early so auth errors fail fast.
    workspace_path = Path(task_workspace) if task_workspace else None
    try:
        build_run_auth_context(
            workspace=workspace_path,
            metadata=metadata,
            required_resources=coerce_auth_requirements(required_auth_resources),
            available_mcp_aliases=set(engine.config.mcp.servers.keys()),
        )
    except UnresolvedAuthResourcesError as e:
        raise HTTPException(status_code=400, detail=e.to_payload())
    except AuthResolutionError as e:
        raise HTTPException(status_code=400, detail=f"Auth preflight failed: {e}")

    task = create_task(
        goal=body.goal,
        workspace=task_workspace,
        approval_mode=body.approval_mode,
        callback_url=body.callback_url or "",
        context=body.context,
        metadata=metadata,
    )

    await _persist_task_snapshot(engine, task)

    # Register webhook if callback_url provided
    if task.callback_url:
        engine.webhook_delivery.register(task.id, task.callback_url)

    # Launch execution in background via engine run manager
    run_id = await engine.submit_task(
        task=task,
        process=process_def,
        process_name=effective_process or "",
    )
    engine.event_bus.emit(Event(
        event_type=TASK_CREATED,
        task_id=task.id,
        data={
            "run_id": run_id,
            "goal": task.goal,
            "workspace": task.workspace,
            "approval_mode": task.approval_mode,
            "execution_surface": metadata.get("execution_surface", "api"),
            "process": effective_process or "",
        },
    ))
    log_latency_event(
        logger,
        event="api_task_create",
        duration_seconds=time.monotonic() - started,
        fields={"has_process": bool(process_def)},
    )

    return TaskCreateResponse(
        task_id=task.id,
        status=task.status.value,
        message="Task created and execution started.",
        run_id=run_id,
    )


async def _execute_in_background(engine: Engine, task, process_def=None) -> None:
    """Legacy helper retained for test compatibility."""
    import logging
    _bg_logger = logging.getLogger(__name__)
    try:
        orchestrator = engine.orchestrator
        if process_def is not None:
            orchestrator = engine.create_task_orchestrator(process_def)

        result = await orchestrator.execute_task(task)
        try:
            await engine._sync_task_row_snapshot(result)
        except Exception:
            _bg_logger.warning("Failed to sync task %s status to DB", result.id)
    except Exception as e:
        _bg_logger.exception("Task %s failed with uncaught exception: %s", task.id, e)
        try:
            task.status = TaskStatus.FAILED
            await _persist_task_snapshot(engine, task)
        except Exception:
            _bg_logger.exception("Failed to save error state for task %s", task.id)


@router.get("/tasks", response_model=list[TaskListItem])
async def list_tasks(request: Request, status: str | None = None):
    """List all tasks, optionally filtered by status."""
    engine = _get_engine(request)
    tasks = await engine.database.list_tasks(status=status)
    return [
        TaskListItem(
            task_id=t["id"],
            goal=t["goal"],
            status=t["status"],
            created_at=t["created_at"],
        )
        for t in tasks
    ]


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(request: Request, task_id: str):
    """Get full task state."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    # Build plan response
    plan_response = None
    if task.plan and task.plan.subtasks:
        plan_response = PlanResponse(
            version=task.plan.version,
            subtasks=[
                SubtaskSummaryResponse(
                    id=s.id,
                    description=s.description,
                    status=s.status.value,
                    depends_on=s.depends_on,
                    retry_count=s.retry_count,
                    summary=s.summary or "",
                )
                for s in task.plan.subtasks
            ],
        )

    # Build progress
    completed, total = task.progress
    failed = sum(
        1 for s in task.plan.subtasks if s.status == SubtaskStatus.FAILED
    ) if task.plan else 0
    pending = sum(
        1 for s in task.plan.subtasks if s.status == SubtaskStatus.PENDING
    ) if task.plan else 0
    running = sum(
        1 for s in task.plan.subtasks if s.status == SubtaskStatus.RUNNING
    ) if task.plan else 0

    progress = ProgressResponse(
        total_subtasks=total,
        completed=completed,
        failed=failed,
        pending=pending,
        running=running,
        percent_complete=(completed / total * 100) if total > 0 else 0,
    )
    run_id = ""
    if isinstance(task.metadata, dict):
        run_id = str(task.metadata.get("run_id", "") or "")

    return TaskResponse(
        task_id=task.id,
        run_id=run_id,
        goal=task.goal,
        status=task.status.value,
        workspace=task.workspace or None,
        plan=plan_response,
        created_at=task.created_at,
        updated_at=task.updated_at,
        approval_mode=task.approval_mode,
        progress=progress,
    )


@router.get("/tasks/{task_id}/stream")
async def stream_task_events(request: Request, task_id: str):
    """SSE stream of real-time task events."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    async def event_generator():
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)

        def handler(event: Event):
            if (
                event.task_id == task_id
                and should_deliver_operator(
                    event.event_type,
                    engine.effective_telemetry_mode(),
                )
            ):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    return

        engine.event_bus.subscribe_all(handler)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        "event": event.event_type,
                        "data": json.dumps({
                            "task_id": event.task_id,
                            **event.data,
                            "timestamp": event.timestamp,
                        }),
                    }
                    # Stop streaming when task is terminal
                    if event.event_type in (
                        TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED,
                    ):
                        return
                except TimeoutError:
                    # Send keepalive comment
                    yield {"comment": "keepalive"}
        except asyncio.CancelledError:
            return
        finally:
            engine.event_bus.unsubscribe_all(handler)

    return EventSourceResponse(event_generator())


@router.get("/tasks/{task_id}/tokens")
async def stream_task_tokens(request: Request, task_id: str):
    """SSE stream of raw model tokens for the active subtask."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    async def token_generator():
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)

        def handler(event: Event):
            if (
                event.task_id == task_id
                and event.event_type == TOKEN_STREAMED
                and should_deliver_operator(
                    event.event_type,
                    engine.effective_telemetry_mode(),
                )
            ):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    return

        # Also listen for terminal events to know when to stop
        terminal_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1)

        def terminal_handler(event: Event):
            if event.task_id == task_id and event.event_type in (
                TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED,
            ):
                if terminal_queue.empty():
                    terminal_queue.put_nowait(event)

        engine.event_bus.subscribe(TOKEN_STREAMED, handler)
        engine.event_bus.subscribe_all(terminal_handler)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield {
                        "event": "token",
                        "data": json.dumps({
                            "token": event.data.get("token", ""),
                            "subtask_id": event.data.get("subtask_id", ""),
                            "model": event.data.get("model", ""),
                        }),
                    }
                except TimeoutError:
                    # Check if task is done
                    if not terminal_queue.empty():
                        return
                    yield {"comment": "keepalive"}
        except asyncio.CancelledError:
            return
        finally:
            engine.event_bus.unsubscribe(TOKEN_STREAMED, handler)
            engine.event_bus.unsubscribe_all(terminal_handler)

    return EventSourceResponse(token_generator())


@router.get("/tasks/{task_id}/questions", response_model=list[TaskQuestionResponse])
async def list_task_questions(request: Request, task_id: str):
    """List pending clarification questions for a task."""
    engine = _get_engine(request)
    if not bool(getattr(engine.config.execution, "ask_user_api_enabled", False)):
        raise HTTPException(status_code=404, detail="Question API is disabled.")

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if _task_execution_surface(task) != "tui":
        raise HTTPException(
            status_code=409,
            detail="Question API is only available for TUI execution surfaces.",
        )

    manager = getattr(engine, "question_manager", None)
    if manager is None:
        raise HTTPException(status_code=404, detail="Question manager is not available.")

    rows = await manager.list_pending_questions(task_id)
    return [_serialize_task_question(row) for row in rows if isinstance(row, dict)]


@router.post(
    "/tasks/{task_id}/questions/{question_id}/answer",
    response_model=TaskQuestionResponse,
)
async def answer_task_question(
    request: Request,
    task_id: str,
    question_id: str,
    body: TaskQuestionAnswerRequest,
):
    """Submit an answer to a pending clarification question."""
    engine = _get_engine(request)
    if not bool(getattr(engine.config.execution, "ask_user_api_enabled", False)):
        raise HTTPException(status_code=404, detail="Question API is disabled.")

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if _task_execution_surface(task) != "tui":
        raise HTTPException(
            status_code=409,
            detail="Question API is only available for TUI execution surfaces.",
        )

    manager = getattr(engine, "question_manager", None)
    if manager is None:
        raise HTTPException(status_code=404, detail="Question manager is not available.")

    existing = await manager.get_question(task_id, question_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question not found: {question_id}",
        )

    status = str(existing.get("status", "") or "").strip().lower()
    if status != "pending":
        if status == "answered":
            return _serialize_task_question(existing)
        raise HTTPException(
            status_code=409,
            detail=f"Question is not pending (status={status or 'unknown'}).",
        )

    answer_payload = body.model_dump(exclude_none=True)
    try:
        resolved = await manager.answer_question(
            task_id=task_id,
            question_id=question_id,
            answer_payload=answer_payload,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"Question not found: {question_id}",
        )
    return _serialize_task_question(resolved)


@router.patch("/tasks/{task_id}")
async def steer_task(request: Request, task_id: str, body: TaskSteerRequest):
    """Inject instructions into a running task."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if task.status not in (TaskStatus.EXECUTING, TaskStatus.PLANNING, TaskStatus.PAUSED):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot steer task in status: {task.status.value}",
        )

    # Store as user instruction in memory
    await engine.memory_manager.store(MemoryEntry(
        task_id=task_id,
        entry_type="user_instruction",
        summary=body.instruction[:150],
        detail=body.instruction,
        tags="steer",
    ))
    engine.event_bus.emit(Event(
        event_type=STEER_INSTRUCTION,
        task_id=task_id,
        data={
            "instruction_chars": len(str(body.instruction or "")),
            "source": "api_patch",
        },
    ))

    return {"status": "ok", "message": "Instruction injected."}


@router.delete("/tasks/{task_id}")
async def cancel_task(request: Request, task_id: str):
    """Cancel a running task."""
    engine = _get_engine(request)
    task, task_row = await _load_task_snapshot_projection(engine, task_id)
    if task_row is None and task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    if task is not None:
        if task.status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            engine.orchestrator.cancel_task(task)
            if task.status == TaskStatus.CANCELLED:
                await _persist_task_snapshot(engine, task)
                latest_run_id = ""
                if isinstance(task.metadata, dict):
                    latest_run_id = str(task.metadata.get("run_id", "") or "").strip()
                if latest_run_id:
                    await engine.database.complete_task_run(
                        run_id=latest_run_id,
                        status="cancelled",
                        last_error="cancel_requested",
                    )
        stop_requested = engine.stop_task_worker(task_id)
        if not stop_requested:
            if task.status != TaskStatus.CANCELLED:
                task.status = TaskStatus.CANCELLED
                await _persist_task_snapshot(engine, task)
            latest_run_id = ""
            if isinstance(task.metadata, dict):
                latest_run_id = str(task.metadata.get("run_id", "") or "").strip()
            engine.event_bus.emit(Event(
                event_type=TASK_CANCELLED,
                task_id=task_id,
                data={
                    "run_id": latest_run_id,
                    "reason": "cancel_requested",
                    "outcome": "cancelled",
                },
            ))
        return {
            "status": "ok",
            "message": f"Task {task_id} cancelled.",
            "task_status": TaskStatus.CANCELLED.value,
            "stop_requested": bool(stop_requested),
        }

    current_status = str((task_row or {}).get("status", "") or "").strip().lower()
    if current_status in ("completed", "failed", "cancelled"):
        return {
            "status": "ok",
            "message": f"Task already {current_status}.",
            "task_status": current_status,
            "stop_requested": False,
        }

    await engine.database.update_task_status(task_id, TaskStatus.CANCELLED.value)
    latest_run = await engine.database.get_latest_task_run_for_task(task_id)
    latest_run_id = str((latest_run or {}).get("run_id", "") or "").strip()
    if latest_run_id:
        await engine.database.complete_task_run(
            run_id=latest_run_id,
            status="cancelled",
            last_error="cancel_requested",
        )
    engine.event_bus.emit(Event(
        event_type=TASK_CANCEL_REQUESTED,
        task_id=task_id,
        data={
            "requested": True,
            "path": "api_fallback",
            "run_id": latest_run_id,
        },
    ))
    engine.event_bus.emit(Event(
        event_type=TASK_CANCELLED,
        task_id=task_id,
        data={
            "run_id": latest_run_id,
            "reason": "cancel_requested",
            "outcome": "cancelled",
        },
    ))
    return {
        "status": "ok",
        "message": f"Task {task_id} cancelled.",
        "task_status": TaskStatus.CANCELLED.value,
        "stop_requested": False,
    }


@router.post("/tasks/{task_id}/pause")
async def pause_task(request: Request, task_id: str):
    """Pause an active task run."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if task.status not in (TaskStatus.EXECUTING, TaskStatus.PLANNING, TaskStatus.PAUSED):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot pause task in status: {task.status.value}",
        )

    engine.orchestrator.pause_task(task)
    await _persist_task_snapshot(engine, task)
    return {
        "status": "ok",
        "message": f"Task {task_id} paused.",
        "task_status": task.status.value,
    }


@router.post("/tasks/{task_id}/resume")
async def resume_task(request: Request, task_id: str):
    """Resume a paused task run."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if task.status not in (TaskStatus.PAUSED, TaskStatus.EXECUTING, TaskStatus.PLANNING):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot resume task in status: {task.status.value}",
        )

    engine.orchestrator.resume_task(task)
    await _persist_task_snapshot(engine, task)
    if bool(getattr(engine.config.execution, "enable_durable_task_runner", False)):
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        run_id = str(metadata.get("run_id", "") or "").strip()
        task_run = await engine.database.get_task_run(run_id) if run_id else None
        task_run_status = str((task_run or {}).get("status", "") or "").strip().lower()
        should_spawn_worker = task.status in (TaskStatus.EXECUTING, TaskStatus.PLANNING) and (
            task_run is None or task_run_status == "queued"
        )
        if should_spawn_worker:
            process_name = str(
                (task_run or {}).get("process_name", "")
                or metadata.get("process", "")
                or "",
            ).strip()
            process = await engine._resolve_process_definition(
                process_name=process_name,
                workspace=Path(task.workspace) if task.workspace else None,
            )
            await engine.submit_task(
                task=task,
                process=process,
                process_name=process_name,
                run_id=run_id or None,
                recovered=bool(task.plan and task.plan.subtasks),
            )
    return {
        "status": "ok",
        "message": f"Task {task_id} resumed.",
        "task_status": task.status.value,
    }


@router.post("/tasks/{task_id}/approve")
async def approve_task(request: Request, task_id: str, body: ApprovalRequest):
    """Approve or reject a gated step."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    # Resolve pending approval via the approval manager
    resolved = engine.approval_manager.resolve_approval(
        task_id=task_id,
        subtask_id=body.subtask_id,
        approved=body.approved,
    )

    # Store approval decision in memory
    content = f"{'Approved' if body.approved else 'Rejected'}: {body.reason or 'No reason given'}"
    await engine.memory_manager.store(MemoryEntry(
        task_id=task_id,
        subtask_id=body.subtask_id,
        entry_type="decision",
        summary=content[:150],
        detail=content,
        tags="approval",
    ))

    return {
        "status": "ok",
        "approved": body.approved,
        "subtask_id": body.subtask_id,
        "resolved_pending": resolved,
    }


@router.post("/tasks/{task_id}/feedback")
async def submit_feedback(request: Request, task_id: str, body: FeedbackRequest):
    """Provide mid-task feedback."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    await engine.memory_manager.store(MemoryEntry(
        task_id=task_id,
        subtask_id=body.subtask_id or "",
        entry_type="user_instruction",
        summary=body.feedback[:150],
        detail=body.feedback,
        tags="feedback",
    ))

    return {"status": "ok", "message": "Feedback recorded."}


@router.post("/tasks/{task_id}/message")
async def send_conversation_message(
    request: Request, task_id: str, body: ConversationMessageRequest,
):
    """Send a conversational message to a running task.

    Messages are injected into the executor's context as memory entries,
    enabling back-and-forth clarification during execution.
    """
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if task.status not in (TaskStatus.EXECUTING, TaskStatus.PLANNING):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot send message to task in status: {task.status.value}",
        )

    # Store as conversation turn in memory
    await engine.memory_manager.store(MemoryEntry(
        task_id=task_id,
        entry_type="user_instruction",
        summary=body.message[:150],
        detail=body.message,
        tags="conversation",
    ))

    # Emit conversation event
    engine.event_bus.emit(Event(
        event_type=CONVERSATION_MESSAGE,
        task_id=task_id,
        data={
            "role": body.role,
            "message": body.message,
        },
    ))

    return {
        "status": "ok",
        "message": "Message delivered.",
        "task_id": task_id,
    }


@router.get("/tasks/{task_id}/conversation")
async def get_conversation_history(request: Request, task_id: str):
    """Retrieve conversation history for a task."""
    engine = _get_engine(request)

    _task, task_row = await _load_task_snapshot_projection(engine, task_id)
    if task_row is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    entries = await engine.memory_manager.query(
        task_id=task_id,
        entry_type="user_instruction",
    )
    return [
        {
            "id": e.id,
            "message": e.detail,
            "summary": e.summary,
            "tags": e.tags,
            "timestamp": e.timestamp,
        }
        for e in entries
    ]


# --- Subtask Level ---


@router.get("/tasks/{task_id}/subtasks")
async def list_subtasks(request: Request, task_id: str):
    """List all subtasks with status."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if not task.plan:
        return []

    return [
        SubtaskSummaryResponse(
            id=s.id,
            description=s.description,
            status=s.status.value,
            depends_on=s.depends_on,
            retry_count=s.retry_count,
            summary=s.summary or "",
        )
        for s in task.plan.subtasks
    ]


@router.get("/tasks/{task_id}/subtasks/{subtask_id}")
async def get_subtask(request: Request, task_id: str, subtask_id: str):
    """Get subtask detail."""
    engine = _get_engine(request)

    if not await _task_state_exists(engine, task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = await _load_task_state(engine, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    subtask = task.get_subtask(subtask_id)
    if subtask is None:
        raise HTTPException(status_code=404, detail=f"Subtask not found: {subtask_id}")

    return SubtaskSummaryResponse(
        id=subtask.id,
        description=subtask.description,
        status=subtask.status.value,
        depends_on=subtask.depends_on,
        retry_count=subtask.retry_count,
        summary=subtask.summary or "",
    )


# --- Memory ---


@router.get("/tasks/{task_id}/memory")
async def query_task_memory(request: Request, task_id: str, entry_type: str | None = None):
    """Query memory entries for a task."""
    engine = _get_engine(request)

    entries = await engine.memory_manager.query(
        task_id=task_id,
        entry_type=entry_type,
    )
    return [
        {
            "id": e.id,
            "task_id": e.task_id,
            "subtask_id": e.subtask_id,
            "entry_type": e.entry_type,
            "summary": e.summary,
            "detail": e.detail,
            "tags": e.tags,
            "timestamp": e.timestamp,
        }
        for e in entries
    ]


@router.get("/memory/search")
async def search_memory(request: Request, q: str, task_id: str | None = None):
    """Search across task memory."""
    engine = _get_engine(request)
    if not task_id:
        return []
    entries = await engine.memory_manager.search(task_id=task_id, query=q)
    return [
        {
            "id": e.id,
            "task_id": e.task_id,
            "entry_type": e.entry_type,
            "summary": e.summary,
            "detail": e.detail,
            "tags": e.tags,
        }
        for e in entries
    ]


# --- Workspace-first Desktop API ---


@router.get("/runtime", response_model=RuntimeStatusResponse)
async def get_runtime_status(request: Request):
    """Return runtime contract details for desktop bootstrap."""
    engine = _get_engine(request)
    return RuntimeStatusResponse(**engine.runtime_status_snapshot())


@router.get("/activity", response_model=ActivitySummaryResponse)
async def get_activity_summary(request: Request):
    """Return a global snapshot of active desktop-visible backend work."""
    engine = _get_engine(request)
    return ActivitySummaryResponse(**(await engine.activity_summary_snapshot()))


@router.get("/workspaces", response_model=list[WorkspaceSummaryResponse])
async def list_workspaces(request: Request, include_archived: bool = False):
    """List registered workspaces with lightweight overview counts."""
    engine = _get_engine(request)
    workspaces = await engine.workspace_registry.list(include_archived=include_archived)
    tasks = await engine.database.list_tasks()
    sessions = await engine.conversation_store.list_sessions()
    payload: list[WorkspaceSummaryResponse] = []
    for workspace in workspaces:
        workspace_tasks = [
            row for row in tasks if _workspace_matches_task(workspace, row)
        ]
        workspace_sessions = [
            row for row in sessions if _workspace_matches_path(workspace, row.get("workspace_path"))
        ]
        payload.append(
            await _build_workspace_summary(
                engine,
                workspace,
                tasks=workspace_tasks,
                sessions=workspace_sessions,
            ),
        )
    return payload


@router.post("/workspaces", response_model=WorkspaceSummaryResponse, status_code=201)
async def create_workspace(request: Request, body: WorkspaceCreateRequest):
    """Register a workspace in the workspace-first desktop registry."""
    engine = _get_engine(request)
    valid, message = validate_workspace(body.path)
    if not valid:
        raise HTTPException(status_code=400, detail=message)
    existing = await engine.workspace_registry.get_by_path(body.path)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Workspace already registered: {existing['id']}",
        )
    created = await engine.workspace_registry.ensure_workspace(
        body.path,
        display_name=body.display_name,
        metadata=body.metadata,
    )
    if created is None:
        raise HTTPException(status_code=500, detail="Workspace registry insert failed.")
    return await _build_workspace_summary(engine, created)


@router.patch("/workspaces/{workspace_id}", response_model=WorkspaceSummaryResponse)
async def patch_workspace(request: Request, workspace_id: str, body: WorkspacePatchRequest):
    """Update workspace metadata without rewriting child records."""
    engine = _get_engine(request)
    await _require_workspace(engine, workspace_id)
    updated = await engine.workspace_registry.update(
        workspace_id,
        display_name=body.display_name,
        sort_order=body.sort_order,
        metadata=body.metadata,
        last_opened_at=body.last_opened_at,
        is_archived=body.archived,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}")
    return await _build_workspace_summary(engine, updated)


@router.delete("/workspaces/{workspace_id}")
async def archive_workspace(request: Request, workspace_id: str):
    """Soft-archive a workspace registry entry."""
    engine = _get_engine(request)
    if not await engine.workspace_registry.archive(workspace_id):
        raise HTTPException(status_code=404, detail=f"Workspace not found: {workspace_id}")
    return {"status": "ok", "workspace_id": workspace_id, "archived": True}


@router.get("/workspaces/{workspace_id}/overview", response_model=WorkspaceOverviewResponse)
async def get_workspace_overview(request: Request, workspace_id: str):
    """Return the workspace overview read model used by the desktop shell."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_workspace_overview",
        fields=_latency_fields(workspace_id=workspace_id),
    ):
        workspace = await _require_workspace(engine, workspace_id)
        tasks = await _workspace_tasks(engine, workspace)
        sessions = await _workspace_sessions(engine, workspace)
        (
            latest_runs_by_task,
            linked_conversations_by_run,
            linked_runs_by_session,
        ) = await _workspace_relationship_maps(
            engine,
            task_rows=tasks,
            session_rows=sessions,
        )
        summary = await _build_workspace_summary(engine, workspace, tasks=tasks, sessions=sessions)
        recent_conversations = [
            await _build_conversation_summary(
                engine,
                workspace_id,
                row,
                linked_runs=linked_runs_by_session.get(str(row.get("id", "") or "").strip(), []),
            )
            for row in sessions[:10]
        ]
        recent_runs = [
            await _build_run_summary(
                engine,
                workspace_id,
                row,
                latest_run=latest_runs_by_task.get(str(row.get("id", "") or "").strip()),
                linked_conversations=linked_conversations_by_run.get(
                    str(row.get("id", "") or "").strip(),
                    [],
                ),
            )
            for row in tasks[:10]
        ]
        pending_approvals_count = await _count_pending_approval_items(engine, workspace=workspace)
        return WorkspaceOverviewResponse(
            workspace=summary,
            recent_conversations=recent_conversations,
            recent_runs=recent_runs,
            pending_approvals_count=pending_approvals_count,
            counts={
                "workspaces": 1,
                "conversations": len(sessions),
                "runs": len(tasks),
                "active_runs": summary.active_run_count,
            },
        )


@router.get(
    "/workspaces/{workspace_id}/inventory",
    response_model=WorkspaceInventoryResponse,
)
async def get_workspace_inventory(request: Request, workspace_id: str):
    """Return workspace-scoped process, MCP, and tool inventory for desktop surfaces."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    summary = await _build_workspace_summary(engine, workspace)
    workspace_path = str(workspace.get("canonical_path", "") or "").strip()
    loader = ProcessLoader(
        workspace=Path(workspace_path).expanduser() if workspace_path else None,
        extra_search_paths=[
            Path(path).expanduser()
            for path in list(getattr(engine.config.process, "search_paths", []) or [])
        ],
        require_rule_scope_metadata=bool(
            getattr(engine.config.process, "require_rule_scope_metadata", False),
        ),
        require_v2_contract=bool(
            getattr(engine.config.process, "require_v2_contract", False),
        ),
    )
    process_rows = [
        ProcessInfoResponse(
            name=str(row.get("name", "") or ""),
            version=str(row.get("version", "") or ""),
            description=str(row.get("description", "") or ""),
            author=str(row.get("author", "") or ""),
            path=str(row.get("path", "") or ""),
        )
        for row in await asyncio.to_thread(loader.list_available)
    ]
    mcp_rows = [
        MCPServerInfoResponse(
            alias=str(alias or ""),
            type=str(server.type or ""),
            enabled=bool(server.enabled),
            source="config",
            command=str(server.command or ""),
            url=str(server.url or server.fallback_sse_url or ""),
            cwd=str(server.cwd or ""),
            timeout_seconds=int(server.timeout_seconds or 0),
            oauth_enabled=bool(getattr(server.oauth, "enabled", False)),
        )
        for alias, server in sorted(engine.config.mcp.servers.items())
    ]
    tool_rows = _tool_info_rows(engine)
    return WorkspaceInventoryResponse(
        workspace=summary,
        processes=process_rows,
        mcp_servers=mcp_rows,
        tools=tool_rows,
        counts={
            "processes": len(process_rows),
            "mcp_servers": len(mcp_rows),
            "tools": len(tool_rows),
        },
    )


@router.get(
    "/workspaces/{workspace_id}/files",
    response_model=list[WorkspaceFileEntryResponse],
)
async def list_workspace_files(
    request: Request,
    workspace_id: str,
    directory: str = "",
):
    """List one directory of workspace files for the desktop file browser."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    try:
        return _list_workspace_directory(workspace, directory)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace path.") from None


@router.get(
    "/workspaces/{workspace_id}/files/preview",
    response_model=WorkspaceFilePreviewResponse,
)
async def preview_workspace_file(
    request: Request,
    workspace_id: str,
    path: str,
):
    """Return a safe workspace file preview payload for the desktop shell."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    try:
        return _build_workspace_file_preview(workspace, path)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid workspace path.") from None


@router.get(
    "/workspaces/{workspace_id}/artifacts",
    response_model=list[WorkspaceArtifactResponse],
)
async def list_workspace_artifacts(request: Request, workspace_id: str):
    """Return a workspace-level artifact index aggregated from existing runs."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_workspace_artifacts",
        fields=_latency_fields(workspace_id=workspace_id),
    ):
        workspace = await _require_workspace(engine, workspace_id)
        return await _build_workspace_artifacts(engine, workspace)


@router.get(
    "/workspaces/{workspace_id}/search",
    response_model=WorkspaceSearchResponse,
)
async def search_workspace(
    request: Request,
    workspace_id: str,
    q: str,
    limit_per_group: int = 5,
):
    """Search a workspace across existing threads, runs, approvals, artifacts, and inventory."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    return await _build_workspace_search_response(
        engine,
        workspace,
        query=q,
        limit_per_group=limit_per_group,
    )


@router.get("/search", response_model=WorkspaceSearchResponse)
async def search_all_workspaces(
    request: Request,
    q: str,
    limit_per_group: int = 5,
):
    """Search across all registered workspaces for desktop global search."""
    engine = _get_engine(request)
    return await _build_global_search_response(
        engine,
        query=q,
        limit_per_group=limit_per_group,
    )


@router.get("/approvals", response_model=list[ApprovalFeedItemResponse])
async def list_approvals(request: Request, workspace_id: str | None = None):
    """Return a unified pending approvals/questions inbox."""
    engine = _get_engine(request)
    workspace_filter = str(workspace_id or "").strip()
    with timed_block(
        logger,
        event="api_approvals_list",
        fields=_latency_fields(workspace_id=workspace_filter or None),
    ):
        if workspace_filter:
            await _require_workspace(engine, workspace_filter)
        return await _list_pending_approval_items(engine, workspace_id=workspace_filter)


@router.post("/approvals/{approval_item_id}/reply")
async def reply_approval(
    request: Request,
    approval_item_id: str,
    body: ApprovalReplyRequest,
):
    """Resolve a unified approval/question item."""
    engine = _get_engine(request)
    raw_parts = [str(part or "").strip() for part in str(approval_item_id or "").split(":")]
    if not raw_parts:
        raise HTTPException(status_code=404, detail="Approval item not found.")

    kind = raw_parts[0]
    if kind == "conversation":
        if len(raw_parts) < 3 or not raw_parts[1] or not raw_parts[2]:
            raise HTTPException(status_code=404, detail="Approval item not found.")
        conversation_id, approval_id = raw_parts[1], raw_parts[2]
        decision = _normalize_conversation_approval_decision(body.decision)
        resolved = engine.resolve_conversation_approval(
            conversation_id,
            approval_id,
            decision,
        )
        if not resolved:
            raise HTTPException(status_code=404, detail="Pending conversation approval not found.")
        return {
            "status": "ok",
            "kind": "conversation_approval",
            "conversation_id": conversation_id,
            "approval_id": approval_id,
            "decision": decision.value,
        }

    if kind == "task":
        if len(raw_parts) < 2 or not raw_parts[1]:
            raise HTTPException(status_code=404, detail="Approval item not found.")
        task_id = raw_parts[1]
        subtask_id = raw_parts[2] if len(raw_parts) >= 3 else ""
        approved = str(body.decision or "").strip().lower() not in {"deny", "reject", "false"}
        resolved = engine.approval_manager.resolve_approval(task_id, subtask_id, approved)
        if not resolved:
            raise HTTPException(status_code=404, detail="Pending task approval not found.")
        content = f"{'Approved' if approved else 'Rejected'}: {body.reason or 'No reason given'}"
        await engine.memory_manager.store(MemoryEntry(
            task_id=task_id,
            subtask_id=subtask_id,
            entry_type="decision",
            summary=content[:150],
            detail=content,
            tags="approval",
        ))
        return {
            "status": "ok",
            "kind": "task_approval",
            "task_id": task_id,
            "subtask_id": subtask_id,
            "approved": approved,
        }

    if kind == "question":
        if len(raw_parts) < 3 or not raw_parts[1] or not raw_parts[2]:
            raise HTTPException(status_code=404, detail="Approval item not found.")
        task_id, question_id = raw_parts[1], raw_parts[2]
        manager = getattr(engine, "question_manager", None)
        if manager is None:
            raise HTTPException(status_code=404, detail="Question manager is not available.")
        resolved = await manager.answer_question(
            task_id=task_id,
            question_id=question_id,
            answer_payload=body.model_dump(exclude_none=True),
        )
        if resolved is None:
            raise HTTPException(status_code=404, detail="Pending task question not found.")
        return _serialize_task_question(resolved)

    raise HTTPException(status_code=404, detail="Approval item not found.")


@router.get("/notifications/stream")
async def stream_notifications(
    request: Request,
    workspace_id: str | None = None,
    after_id: int = 0,
    follow: bool = True,
):
    """SSE stream of approval/question notifications for desktop inbox surfaces."""
    engine = _get_engine(request)
    workspace_filter = str(workspace_id or "").strip()
    if workspace_filter:
        await _require_workspace(engine, workspace_filter)

    relevant_events = {
        APPROVAL_REQUESTED,
        APPROVAL_RECEIVED,
        ASK_USER_REQUESTED,
        ASK_USER_ANSWERED,
        ASK_USER_TIMEOUT,
        ASK_USER_CANCELLED,
    }

    notification_event_types = tuple(sorted(relevant_events))

    async def query_notification_rows(*, cursor: int) -> list[dict[str, Any]]:
        placeholders = ", ".join("?" for _ in notification_event_types)
        return await engine.database.query(
            f"""
            SELECT *
            FROM events
            WHERE id > ?
              AND event_type IN ({placeholders})
            ORDER BY id ASC
            LIMIT 500
            """,
            (cursor, *notification_event_types),
        )

    async def event_generator():
        last_event_id = str(request.headers.get("last-event-id", "") or "").strip()
        cursor = await _resolve_notification_stream_cursor(
            engine,
            after_id=after_id,
            last_event_id=last_event_id,
        )
        history_resume_event_id = _notification_resume_event_id(last_event_id)
        seen_event_ids: set[str] = set()
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
        overflowed = False

        def handler(event: Event):
            nonlocal overflowed
            if (
                event.event_type in relevant_events
                and should_deliver_operator(
                    event.event_type,
                    engine.effective_telemetry_mode(),
                )
            ):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    overflowed = True

        engine.event_bus.subscribe_all(handler)
        try:
            yield {"comment": "open"}
            replay_rows = await query_notification_rows(cursor=cursor)
            for row in replay_rows:
                row_id = int(row.get("id", 0) or 0)
                if row_id <= cursor:
                    continue
                cursor = row_id
                payload = await _notification_payload_from_row(engine, row)
                if payload is None:
                    continue
                if (
                    workspace_filter
                    and str(payload.get("workspace_id", "") or "") != workspace_filter
                ):
                    continue
                payload_id = str(payload.get("id", "") or "").strip()
                if payload_id:
                    seen_event_ids.add(payload_id)
                    history_resume_event_id = payload_id
                yield {
                    "event": "notification",
                    "id": str(row_id),
                    "data": json.dumps(payload),
                }
            history_payloads = await _notification_history_payloads(
                engine,
                relevant_events=relevant_events,
                workspace_filter=workspace_filter,
                after_event_id=history_resume_event_id,
                seen_event_ids=seen_event_ids,
            )
            for payload in history_payloads:
                payload_id = str(payload.get("id", "") or "").strip()
                if payload_id:
                    history_resume_event_id = payload_id
                response: dict[str, str] = {
                    "event": "notification",
                    "data": json.dumps(payload),
                }
                if payload_id:
                    response["id"] = f"event:{payload_id}"
                yield response
            if not follow:
                return
            while True:
                if overflowed:
                    overflowed = False
                    replay_rows = await query_notification_rows(cursor=cursor)
                    for row in replay_rows:
                        row_id = int(row.get("id", 0) or 0)
                        if row_id <= cursor:
                            continue
                        cursor = row_id
                        payload = await _notification_payload_from_row(engine, row)
                        if payload is None:
                            continue
                        if (
                            workspace_filter
                            and str(payload.get("workspace_id", "") or "") != workspace_filter
                        ):
                            continue
                        payload_id = str(payload.get("id", "") or "").strip()
                        if payload_id:
                            seen_event_ids.add(payload_id)
                            history_resume_event_id = payload_id
                        yield {
                            "event": "notification",
                            "id": str(row_id),
                            "data": json.dumps(payload),
                        }
                    history_payloads = await _notification_history_payloads(
                        engine,
                        relevant_events=relevant_events,
                        workspace_filter=workspace_filter,
                        after_event_id=history_resume_event_id,
                        seen_event_ids=seen_event_ids,
                    )
                    for payload in history_payloads:
                        payload_id = str(payload.get("id", "") or "").strip()
                        if payload_id:
                            history_resume_event_id = payload_id
                        response: dict[str, str] = {
                            "event": "notification",
                            "data": json.dumps(payload),
                        }
                        if payload_id:
                            response["id"] = f"event:{payload_id}"
                        yield response
                    continue
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    yield {"comment": "keepalive"}
                    continue
                if await request.is_disconnected():
                    return
                payload = await _notification_payload_from_event(engine, event)
                if payload is None:
                    continue
                if (
                    workspace_filter
                    and str(payload.get("workspace_id", "") or "") != workspace_filter
                ):
                    continue
                event_id = str(payload.get("id", "") or "").strip()
                if event_id and event_id in seen_event_ids:
                    continue
                if event_id:
                    seen_event_ids.add(event_id)
                    history_resume_event_id = event_id
                response: dict[str, str] = {
                    "event": "notification",
                    "data": json.dumps(payload),
                }
                if event_id:
                    response["id"] = f"event:{event_id}"
                yield response
        except asyncio.CancelledError:
            return
        finally:
            engine.event_bus.unsubscribe_all(handler)

    return EventSourceResponse(event_generator())


@router.get(
    "/workspaces/{workspace_id}/conversations",
    response_model=list[ConversationSummaryResponse],
)
async def list_workspace_conversations(request: Request, workspace_id: str):
    """List conversations/session threads that belong to a workspace."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    sessions = await _workspace_sessions(engine, workspace)
    linked_runs_by_session = await engine.conversation_store.list_linked_runs_for_sessions([
        str(row.get("id", "") or "").strip()
        for row in sessions
        if str(row.get("id", "") or "").strip()
    ])
    return [
        await _build_conversation_summary(
            engine,
            workspace_id,
            row,
            linked_runs=linked_runs_by_session.get(str(row.get("id", "") or "").strip(), []),
        )
        for row in sessions
    ]


@router.post(
    "/workspaces/{workspace_id}/conversations",
    response_model=ConversationSummaryResponse,
    status_code=201,
)
async def create_workspace_conversation(
    request: Request,
    workspace_id: str,
    body: ConversationCreateRequest,
):
    """Create a new durable conversation thread under a workspace."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    model_name = str(body.model_name or "").strip()
    if not model_name:
        providers = engine.model_router.list_providers()
        model_name = str(providers[0]["name"]) if providers else "default"
    session_id = await engine.conversation_store.create_session(
        workspace=str(workspace.get("canonical_path", "") or ""),
        model_name=model_name,
        system_prompt=body.system_prompt,
    )
    session = await engine.conversation_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=500, detail="Conversation creation failed.")
    await engine.workspace_registry.update(workspace_id, last_opened_at=_now_iso())
    return await _build_conversation_summary(engine, workspace_id, session)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str):
    """Delete a conversation/thread and all its messages."""
    engine = _get_engine(request)
    session = await engine.conversation_store.get_session(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")
    await engine.conversation_store.delete_session(conversation_id)
    return {"status": "ok", "message": f"Thread {conversation_id} deleted."}


@router.get("/conversations/{conversation_id}")
async def get_conversation(request: Request, conversation_id: str):
    """Return one conversation/thread with session-state metadata."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_conversation_detail",
        fields=_latency_fields(conversation_id=conversation_id),
    ):
        session = await engine.conversation_store.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}",
            )
        workspace = await engine.workspace_registry.get_by_path(session.get("workspace_path"))
        workspace_id = str((workspace or {}).get("id", "") or "")
        summary = await _build_conversation_summary(engine, workspace_id, session)
        session_state = _json_object(session.get("session_state"))
        return {
            **summary.model_dump(),
            "system_prompt": str(session.get("system_prompt", "") or ""),
            "session_state": session_state,
            "workspace": workspace or {},
        }


@router.patch("/conversations/{conversation_id}")
async def patch_conversation(
    conversation_id: str,
    body: ConversationPatchRequest,
    request: Request,
):
    """Update conversation metadata (e.g. title)."""
    engine = _get_engine(request)
    session = await engine.conversation_store.get_session(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    if body.title is not None:
        await engine.conversation_store.patch_session_state_metadata(
            conversation_id,
            title=body.title,
        )

    # Re-fetch and return the updated detail using the same shape as GET.
    session = await engine.conversation_store.get_session(conversation_id)
    workspace = await engine.workspace_registry.get_by_path(session.get("workspace_path"))
    workspace_id = str((workspace or {}).get("id", "") or "")
    summary = await _build_conversation_summary(engine, workspace_id, session)
    session_state = _json_object(session.get("session_state"))
    return {
        **summary.model_dump(),
        "system_prompt": str(session.get("system_prompt", "") or ""),
        "session_state": session_state,
        "workspace": workspace or {},
    }


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    request: Request,
    conversation_id: str,
    offset: int = 0,
    limit: int = 100,
    before_turn: int | None = None,
    latest: bool = False,
):
    """Return persisted conversation turns for a thread."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_conversation_messages",
        fields=_latency_fields(
            conversation_id=conversation_id,
            latest=latest,
            before_turn=before_turn,
            limit=limit,
        ),
    ):
        session = await engine.conversation_store.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}",
            )
        if before_turn is not None:
            turns = await engine.conversation_store.get_turns_before(
                conversation_id,
                before_turn=max(1, int(before_turn)),
                limit=limit,
            )
        elif latest:
            turns = await engine.conversation_store.get_recent_turns(
                conversation_id,
                limit=limit,
            )
        else:
            turns = await engine.conversation_store.get_turns(
                conversation_id,
                offset=offset,
                limit=limit,
            )
        rows: list[dict[str, Any]] = []
        for row in turns:
            item = dict(row)
            item["tool_calls"] = _json_list(item.get("tool_calls"))
            rows.append(item)
        return rows


@router.get("/conversations/{conversation_id}/events")
async def get_conversation_events(
    request: Request,
    conversation_id: str,
    before_seq: int | None = None,
    before_turn: int | None = None,
    after_seq: int = 0,
    limit: int = 200,
):
    """Return replay-journal events for a conversation thread."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_conversation_events",
        fields=_latency_fields(
            conversation_id=conversation_id,
            before_seq=before_seq,
            before_turn=before_turn,
            after_seq=after_seq,
            limit=limit,
        ),
    ):
        session = await engine.conversation_store.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}",
            )
        rows = await engine.conversation_store.get_transcript_page(
            conversation_id,
            before_seq=(
                None if before_seq is None else max(0, int(before_seq))
            ),
            before_turn=before_turn,
            after_seq=max(0, int(after_seq)),
            limit=limit,
        )
        if rows:
            if (
                (before_seq is not None or after_seq <= 0)
                and all("turn_number" not in row for row in rows)
            ):
                rows = await _expand_conversation_event_page_prefix(
                    engine,
                    conversation_id,
                    rows,
                    limit=limit,
                )
            return rows
        if after_seq > 0 and before_seq is None:
            # Incremental polling/refresh should return only truly new rows.
            # Falling back to synthesized turn events here duplicates content
            # that the client may already have from durable chat-event rows.
            return []
        return []


@router.get("/conversations/{conversation_id}/status")
async def get_conversation_status(
    request: Request,
    conversation_id: str,
):
    """Return processing state for a persisted conversation thread."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_conversation_status",
        fields=_latency_fields(conversation_id=conversation_id),
    ):
        session = await engine.conversation_store.get_session(conversation_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}",
            )
        processing = engine.conversation_turn_inflight(conversation_id)
        pending_approval = engine.get_pending_conversation_approval(conversation_id)
        pending_prompt = (
            None
            if processing or pending_approval is not None
            else await _conversation_pending_prompt(engine, conversation_id)
        )
        return {
            "conversation_id": conversation_id,
            "processing": processing,
            "stop_requested": engine.conversation_stop_requested(conversation_id),
            "pending_inject_count": engine.conversation_pending_inject_count(conversation_id),
            "awaiting_approval": pending_approval is not None,
            "pending_approval": pending_approval,
            "awaiting_user_input": pending_prompt is not None,
            "pending_prompt": pending_prompt,
        }


@router.post("/conversations/{conversation_id}/messages", status_code=202)
async def post_conversation_message(
    request: Request,
    conversation_id: str,
    body: ConversationMessageRequest,
):
    """Start one cowork turn for a persisted conversation thread."""
    engine = _get_engine(request)
    session = await engine.conversation_store.get_session(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    message = str(body.message or "").strip()
    if not message:
        raise HTTPException(status_code=422, detail="message is required")

    if engine.conversation_turn_inflight(conversation_id):
        raise HTTPException(
            status_code=409,
            detail="Conversation is already processing a turn.",
        )

    try:
        cowork_session = _build_api_cowork_session(engine, session)
        engine.start_conversation_worker(
            conversation_id,
            cowork_session,
            _run_cowork_turn_for_api(
                engine,
                conversation_id=conversation_id,
                session=cowork_session,
                user_message=message,
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "accepted",
        "conversation_id": conversation_id,
        "message": "Conversation turn started.",
    }


@router.post("/conversations/{conversation_id}/stop")
async def stop_conversation_turn(
    request: Request,
    conversation_id: str,
):
    """Request cooperative interruption for the active conversation turn."""
    engine = _get_engine(request)
    session = await engine.conversation_store.get_session(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")
    if not engine.request_conversation_stop(conversation_id):
        raise HTTPException(
            status_code=409,
            detail="Conversation is not currently processing a turn.",
        )
    return {
        "status": "accepted",
        "conversation_id": conversation_id,
        "message": "Conversation stop requested.",
    }


@router.post("/conversations/{conversation_id}/inject")
async def inject_conversation_instruction(
    request: Request,
    conversation_id: str,
    body: ConversationInjectRequest,
):
    """Queue a steering instruction for the active cowork turn."""
    engine = _get_engine(request)
    session = await engine.conversation_store.get_session(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    instruction = str(body.instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=422, detail="instruction is required")

    queued = engine.queue_conversation_inject_instruction(conversation_id, instruction)
    if queued < 0:
        raise HTTPException(
            status_code=409,
            detail="Conversation is not currently processing a turn.",
        )

    await _append_conversation_replay_event(
        engine,
        conversation_id,
        "steering_instruction",
        {
            "instruction": instruction,
            "source": "api_inject",
            "pending_inject_count": queued,
        },
    )
    return {
        "status": "accepted",
        "conversation_id": conversation_id,
        "message": "Instruction queued for the active conversation turn.",
        "pending_inject_count": queued,
    }


@router.post("/conversations/{conversation_id}/approvals/{approval_id}")
async def resolve_conversation_approval(
    request: Request,
    conversation_id: str,
    approval_id: str,
    body: ConversationApprovalDecisionRequest,
):
    """Resolve one pending cowork tool approval for a conversation."""
    engine = _get_engine(request)
    session = await engine.conversation_store.get_session(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    decision = _normalize_conversation_approval_decision(body.decision)
    resolved = engine.resolve_conversation_approval(
        conversation_id,
        approval_id,
        decision,
    )
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail="Pending conversation approval not found.",
        )
    return {
        "status": "ok",
        "conversation_id": conversation_id,
        "approval_id": approval_id,
        "decision": decision.value,
        "message": f"Conversation approval resolved: {decision.value}.",
    }


@router.get("/conversations/{conversation_id}/stream")
async def stream_conversation_events(
    request: Request,
    conversation_id: str,
    after_seq: int = 0,
    follow: bool = True,
):
    """Event-bus SSE stream of conversation replay events with instant delivery."""
    engine = _get_engine(request)
    session = await engine.conversation_store.get_session(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")

    async def event_generator():
        last_event_id = str(request.headers.get("last-event-id", "") or "").strip()
        header_cursor = int(last_event_id) if last_event_id.isdigit() else 0
        cursor = max(0, int(after_seq), header_cursor)
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
        overflowed = False

        def handler(event: Event):
            nonlocal overflowed
            if (
                event.event_type == CONVERSATION_MESSAGE
                and event.task_id == conversation_id
            ):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    overflowed = True

        engine.event_bus.subscribe_all(handler)

        try:
            yield {"comment": "open"}
            # Subscribe first, then replay durable rows so we don't miss
            # events emitted between the initial query and subscription setup.
            rows = await engine.conversation_store.get_transcript_page(
                conversation_id,
                after_seq=cursor,
                limit=500,
            )
            if cursor <= 0 and rows and all("turn_number" not in row for row in rows):
                rows = await _expand_conversation_event_page_prefix(
                    engine,
                    conversation_id,
                    rows,
                    limit=500,
                )
            for row in rows:
                cursor = max(cursor, int(row.get("seq", 0) or 0))
                yield {
                    "event": "chat_event",
                    "id": str(int(row.get("seq", 0) or 0)),
                    "data": json.dumps(row),
                }
            if not follow:
                return

            while True:
                if overflowed:
                    overflowed = False
                    rows = await engine.conversation_store.get_transcript_page(
                        conversation_id,
                        after_seq=cursor,
                        limit=500,
                    )
                    for row in rows:
                        seq = int(row.get("seq", 0) or 0)
                        if seq <= cursor:
                            continue
                        cursor = seq
                        yield {
                            "event": "chat_event",
                            "id": str(seq),
                            "data": json.dumps(row),
                        }
                    continue
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    chat_event_type = str(event.data.get("chat_event_type", "") or "")
                    seq = int(event.data.get("seq", 0) or 0)
                    payload = event.data.get("payload", {})
                    if seq > cursor:
                        cursor = seq
                        yield {
                            "event": "chat_event",
                            "id": str(seq),
                            "data": json.dumps({
                                "session_id": conversation_id,
                                "seq": seq,
                                "event_type": chat_event_type,
                                "payload": payload if isinstance(payload, dict) else {},
                                "payload_parse_error": False,
                                "created_at": event.timestamp,
                            }),
                        }
                except TimeoutError:
                    yield {"comment": "keepalive"}
        except asyncio.CancelledError:
            return
        finally:
            engine.event_bus.unsubscribe_all(handler)

    return EventSourceResponse(event_generator())


@router.get("/workspaces/{workspace_id}/runs", response_model=list[RunSummaryResponse])
async def list_workspace_runs(request: Request, workspace_id: str):
    """List task-backed runs that belong to a workspace."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    tasks = await _workspace_tasks(engine, workspace)
    task_ids = [
        str(row.get("id", "") or "").strip()
        for row in tasks
        if str(row.get("id", "") or "").strip()
    ]
    latest_runs_by_task, linked_conversations_by_run = await asyncio.gather(
        engine.database.get_latest_task_runs_for_tasks(task_ids),
        engine.conversation_store.list_linked_conversations_for_runs(task_ids),
    )
    return [
        await _build_run_summary(
            engine,
            workspace_id,
            row,
            latest_run=latest_runs_by_task.get(str(row.get("id", "") or "").strip()),
            linked_conversations=linked_conversations_by_run.get(
                str(row.get("id", "") or "").strip(),
                [],
            ),
        )
        for row in tasks
    ]


@router.get("/runs/{run_id}")
async def get_run(request: Request, run_id: str):
    """Return one run backed by the existing task/task_run stores."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_run_detail",
        fields=_latency_fields(run_id=run_id),
    ):
        task_obj, task_row = await _load_task_snapshot_projection(engine, run_id)
        if task_row is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        workspace = await engine.workspace_registry.get_by_path(
            _task_workspace_group_root(task_row),
        )
        workspace_id = str((workspace or {}).get("id", "") or "")
        latest_run = await engine.database.get_latest_task_run_for_task(run_id)
        event_rows = await engine.database.query_events(run_id, limit=1000, ascending=True)
        resolved_status = await _resolve_task_status(engine, task_row)
        failure_analysis = _build_run_failure_analysis(
            resolved_status=resolved_status,
            task_obj=task_obj,
            event_rows=event_rows,
        )
        summary = await _build_run_summary(
            engine,
            workspace_id,
            task_row,
            latest_run=latest_run,
            live_status=resolved_status,
            failure_analysis=failure_analysis,
        )

        # Include plan/subtask data from state manager when available
        plan_data: list[dict[str, Any]] = []
        if task_obj is not None and task_obj.plan and task_obj.plan.subtasks:
            for s in task_obj.plan.subtasks:
                plan_data.append({
                    "id": s.id,
                    "description": s.description,
                    "status": s.status.value,
                    "depends_on": s.depends_on,
                    "phase_id": s.phase_id,
                    "summary": s.summary or "",
                    "is_critical_path": s.is_critical_path,
                    "is_synthesis": s.is_synthesis,
                })

        return {
            **summary.model_dump(),
            "task": task_row,
            "task_run": latest_run or {},
            "events_count": len(event_rows),
            "workspace": workspace or {},
            "plan_subtasks": plan_data,
        }


@router.get("/runs/{run_id}/artifacts", response_model=list[RunArtifactResponse])
async def get_run_artifacts(request: Request, run_id: str):
    """Return durable run artifacts merged from task seals and evidence ledgers."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_run_artifacts",
        fields=_latency_fields(run_id=run_id),
    ):
        _task, task_row = await _load_task_snapshot_projection(engine, run_id)
        if task_row is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return await _build_run_artifacts(engine, task_row)


_RUN_TERMINAL_EVENT_TYPES = {
    TASK_COMPLETED,
    TASK_FAILED,
    TASK_CANCELLED,
    TASK_CANCEL_REQUESTED,
}

_RUN_TIMELINE_NOISE_EVENT_TYPES = {
    "telemetry_diagnostic",
    "telemetry_run_summary",
    "telemetry_mode_changed",
    "telemetry_settings_warning",
    "webhook_delivery_attempted",
    "webhook_delivery_succeeded",
    "webhook_delivery_failed",
    "webhook_delivery_dropped",
    "db_migration_start",
    "db_migration_applied",
    "db_migration_verify_failed",
    "db_migration_failed",
    "db_schema_ready",
    "task_run_heartbeat",
    "compaction_policy_decision",
    "overflow_fallback_applied",
    "token_streamed",
    "tool_call_completed",
    "model_invocation",
}


def _should_include_run_timeline_event(
    event_type: str,
    data: dict[str, Any] | None = None,
    *,
    include_noise: bool,
) -> bool:
    if include_noise:
        return True
    normalized = str(event_type or "").strip().lower()
    payload = data if isinstance(data, dict) else {}
    if normalized == "model_invocation" and bool(payload.get("retry_scheduled")):
        return True
    if normalized in _RUN_TIMELINE_NOISE_EVENT_TYPES:
        return False
    if normalized == "claim_verification_summary":
        extracted = int(payload.get("extracted", 0) or 0)
        supported = int(payload.get("supported", 0) or 0)
        partially_supported = int(payload.get("partially_supported", 0) or 0)
        contradicted = int(payload.get("contradicted", 0) or 0)
        insufficient = int(payload.get("insufficient_evidence", 0) or 0)
        pruned = int(payload.get("pruned", 0) or 0)
        if (
            extracted <= 0
            and supported <= 0
            and partially_supported <= 0
            and contradicted <= 0
            and insufficient <= 0
            and pruned <= 0
        ):
            return False
    return True


async def _resolve_run_stream_cursor(
    engine: Engine,
    run_id: str,
    *,
    after_sequence: int = 0,
    after_id: int = 0,
    last_event_id: str = "",
) -> int:
    cursor = max(0, int(after_sequence))
    candidates: list[int] = []
    if int(after_id) > 0:
        candidates.append(int(after_id))
    header_cursor = str(last_event_id or "").strip()
    if header_cursor.startswith("seq:"):
        suffix = header_cursor.removeprefix("seq:").strip()
        if suffix.isdigit():
            candidates.append(int(suffix))
    elif header_cursor.isdigit():
        candidates.append(int(header_cursor))
    for candidate in candidates:
        row = await engine.database.query_one(
            "SELECT sequence FROM events WHERE task_id = ? AND id = ?",
            (run_id, candidate),
        )
        if row is not None:
            cursor = max(cursor, int(row.get("sequence", 0) or 0))
        else:
            cursor = max(cursor, candidate)
    return cursor


def _recent_run_history_payloads(
    engine: Engine,
    run_id: str,
    *,
    after_sequence: int,
    include_noise: bool = True,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen_sequences: set[int] = set()
    for event in engine.event_bus.recent_events(limit=1000):
        if event.task_id != run_id:
            continue
        if not should_deliver_operator(
            event.event_type,
            engine.effective_telemetry_mode(),
        ):
            continue
        payload = _run_stream_payload_from_event(event)
        if not _should_include_run_timeline_event(
            str(payload.get("event_type", "") or ""),
            payload.get("data") if isinstance(payload.get("data"), dict) else None,
            include_noise=include_noise,
        ):
            continue
        sequence = int(payload.get("sequence", 0) or 0)
        if sequence <= after_sequence or sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        payloads.append(payload)
    payloads.sort(key=lambda item: int(item.get("sequence", 0) or 0))
    return payloads


def _serialize_run_timeline_row(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["data"] = _json_object(item.get("data"))
    return item


def _run_stream_payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    item = _serialize_run_timeline_row(row)
    payload = dict(item["data"]) if isinstance(item.get("data"), dict) else {}
    event_type = str(item.get("event_type", "") or "")
    status = str(payload.get("status", "") or "").strip().lower()
    terminal = event_type in _RUN_TERMINAL_EVENT_TYPES
    return {
        **item,
        "status": status or None,
        "terminal": terminal,
        "streaming": not terminal and status != "paused",
    }


def _run_stream_payload_from_event(event: Event) -> dict[str, Any]:
    payload = dict(event.data) if isinstance(event.data, dict) else {}
    event_type = str(event.event_type or "")
    status = str(payload.get("status", "") or "").strip().lower()
    terminal = event_type in _RUN_TERMINAL_EVENT_TYPES
    return {
        "task_id": event.task_id,
        "run_id": str(payload.get("run_id", "") or ""),
        "correlation_id": str(payload.get("correlation_id", "") or ""),
        "event_id": str(payload.get("event_id", "") or ""),
        "sequence": int(payload.get("sequence", 0) or 0),
        "timestamp": event.timestamp,
        "event_type": event_type,
        "source_component": str(payload.get("source_component", "") or ""),
        "schema_version": int(payload.get("schema_version", 1) or 1),
        "data": payload,
        "status": status or None,
        "terminal": terminal,
        "streaming": not terminal and status != "paused",
    }

@router.get("/runs/{run_id}/timeline")
async def get_run_timeline(
    request: Request,
    run_id: str,
    limit: int = 5000,
    include_noise: bool = True,
):
    """Return persisted event-timeline rows for a run/task."""
    engine = _get_engine(request)
    with timed_block(
        logger,
        event="api_run_timeline",
        fields=_latency_fields(run_id=run_id, limit=limit),
    ):
        _task, task_row = await _load_task_snapshot_projection(engine, run_id)
        if task_row is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        rows = await engine.database.query_events(run_id, limit=limit, ascending=True)
        payload: list[dict[str, Any]] = []
        for row in rows:
            serialized = _serialize_run_timeline_row(row)
            if not _should_include_run_timeline_event(
                str(serialized.get("event_type", "") or ""),
                serialized.get("data") if isinstance(serialized.get("data"), dict) else None,
                include_noise=include_noise,
            ):
                continue
            payload.append(serialized)
        return payload


@router.get("/runs/{run_id}/stream")
async def stream_run_events(
    request: Request,
    run_id: str,
    after_id: int = 0,
    after_sequence: int = 0,
    include_noise: bool = True,
    follow: bool = True,
):
    """Durable-first SSE stream of real-time run events."""
    engine = _get_engine(request)
    task, task_row = await _load_task_snapshot_projection(engine, run_id)
    if task_row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    resolved_status = str(task_row.get("status", "") or "").strip().lower()
    is_terminal = resolved_status in ("completed", "failed", "cancelled")

    if task is None and is_terminal:
        async def inactive_generator():
            last_event_id = str(request.headers.get("last-event-id", "") or "").strip()
            cursor = await _resolve_run_stream_cursor(
                engine,
                run_id,
                after_sequence=after_sequence,
                after_id=after_id,
                last_event_id=last_event_id,
            )
            replay_started = time.monotonic()
            rows = await engine.database.query_events(
                run_id,
                limit=500,
                after_sequence=cursor,
                ascending=True,
            )
            for row in rows:
                sequence = int(row.get("sequence", 0) or 0)
                if sequence <= cursor:
                    continue
                cursor = sequence
                payload = _run_stream_payload_from_row(row)
                if not _should_include_run_timeline_event(
                    str(payload.get("event_type", "") or ""),
                    payload.get("data") if isinstance(payload.get("data"), dict) else None,
                    include_noise=include_noise,
                ):
                    continue
                yield {
                    "event": "run_event",
                    "id": str(sequence),
                    "data": json.dumps(payload),
                }
            for payload in _recent_run_history_payloads(
                engine,
                run_id,
                after_sequence=cursor,
                include_noise=include_noise,
            ):
                sequence = int(payload.get("sequence", 0) or 0)
                if sequence <= cursor:
                    continue
                cursor = sequence
                yield {
                    "event": "run_event",
                    "id": str(sequence),
                    "data": json.dumps(payload),
                }
                if payload["terminal"] is True:
                    log_latency_event(
                        logger,
                        event="api_run_stream_open",
                        duration_seconds=time.monotonic() - replay_started,
                        fields=_latency_fields(run_id=run_id, phase="inactive_replay"),
                    )
                    return
            log_latency_event(
                logger,
                event="api_run_stream_open",
                duration_seconds=time.monotonic() - replay_started,
                fields=_latency_fields(run_id=run_id, phase="inactive_replay"),
            )
            if not follow:
                return
            yield {
                "event": "run_event",
                "data": json.dumps({
                    "event_type": "run_snapshot",
                    "task_id": run_id,
                    "timestamp": _now_iso(),
                    "status": resolved_status,
                    "terminal": True,
                    "streaming": False,
                }),
            }

        return EventSourceResponse(inactive_generator())

    async def event_generator():
        last_event_id = str(request.headers.get("last-event-id", "") or "").strip()
        cursor = await _resolve_run_stream_cursor(
            engine,
            run_id,
            after_sequence=after_sequence,
            after_id=after_id,
            last_event_id=last_event_id,
        )
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_STREAM_QUEUE_MAXSIZE)
        overflowed = False

        def handler(event: Event):
            nonlocal overflowed
            if (
                event.task_id == run_id
                and should_deliver_operator(
                    event.event_type,
                    engine.effective_telemetry_mode(),
                )
                and _should_include_run_timeline_event(
                    event.event_type,
                    event.data if isinstance(event.data, dict) else None,
                    include_noise=include_noise,
                )
            ):
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    overflowed = True

        engine.event_bus.subscribe_all(handler)

        try:
            yield {"comment": "open"}
            replay_started = time.monotonic()
            replay_rows = await engine.database.query_events(
                run_id,
                limit=500,
                after_sequence=cursor,
                ascending=True,
            )
            for row in replay_rows:
                sequence = int(row.get("sequence", 0) or 0)
                if sequence <= cursor:
                    continue
                cursor = sequence
                payload = _run_stream_payload_from_row(row)
                if not _should_include_run_timeline_event(
                    str(payload.get("event_type", "") or ""),
                    payload.get("data") if isinstance(payload.get("data"), dict) else None,
                    include_noise=include_noise,
                ):
                    continue
                yield {
                    "event": "run_event",
                    "id": str(sequence),
                    "data": json.dumps(payload),
                }
                if payload["terminal"] is True:
                    log_latency_event(
                        logger,
                        event="api_run_stream_open",
                        duration_seconds=time.monotonic() - replay_started,
                        fields=_latency_fields(run_id=run_id, phase="active_replay"),
                    )
                    return
            for payload in _recent_run_history_payloads(
                engine,
                run_id,
                after_sequence=cursor,
                include_noise=include_noise,
            ):
                sequence = int(payload.get("sequence", 0) or 0)
                if sequence <= cursor:
                    continue
                cursor = sequence
                yield {
                    "event": "run_event",
                    "id": str(sequence),
                    "data": json.dumps(payload),
                }
                if payload["terminal"] is True:
                    log_latency_event(
                        logger,
                        event="api_run_stream_open",
                        duration_seconds=time.monotonic() - replay_started,
                        fields=_latency_fields(run_id=run_id, phase="active_replay"),
                    )
                    return
            log_latency_event(
                logger,
                event="api_run_stream_open",
                duration_seconds=time.monotonic() - replay_started,
                fields=_latency_fields(run_id=run_id, phase="active_replay"),
            )
            if not follow:
                return

            while True:
                if overflowed:
                    overflowed = False
                    replay_rows = await engine.database.query_events(
                        run_id,
                        limit=500,
                        after_sequence=cursor,
                        ascending=True,
                    )
                    for row in replay_rows:
                        sequence = int(row.get("sequence", 0) or 0)
                        if sequence <= cursor:
                            continue
                        cursor = sequence
                        payload = _run_stream_payload_from_row(row)
                        if not _should_include_run_timeline_event(
                            str(payload.get("event_type", "") or ""),
                            payload.get("data") if isinstance(payload.get("data"), dict) else None,
                            include_noise=include_noise,
                        ):
                            continue
                        yield {
                            "event": "run_event",
                            "id": str(sequence),
                            "data": json.dumps(payload),
                        }
                        if payload["terminal"] is True:
                            return
                    for payload in _recent_run_history_payloads(
                        engine,
                        run_id,
                        after_sequence=cursor,
                        include_noise=include_noise,
                    ):
                        sequence = int(payload.get("sequence", 0) or 0)
                        if sequence <= cursor:
                            continue
                        cursor = sequence
                        yield {
                            "event": "run_event",
                            "id": str(sequence),
                            "data": json.dumps(payload),
                        }
                        if payload["terminal"] is True:
                            return
                    continue
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except TimeoutError:
                    yield {"comment": "keepalive"}
                    continue
                payload = _run_stream_payload_from_event(event)
                sequence = int(payload.get("sequence", 0) or 0)
                if sequence <= cursor:
                    continue
                cursor = sequence
                if not _should_include_run_timeline_event(
                    str(payload.get("event_type", "") or ""),
                    payload.get("data") if isinstance(payload.get("data"), dict) else None,
                    include_noise=include_noise,
                ):
                    continue
                yield {
                    "event": "run_event",
                    "id": str(sequence),
                    "data": json.dumps(payload),
                }
                if payload["terminal"] is True:
                    return
        except asyncio.CancelledError:
            return
        finally:
            engine.event_bus.unsubscribe_all(handler)

    return EventSourceResponse(event_generator())


@router.post("/runs/{run_id}/cancel")
async def cancel_run(request: Request, run_id: str):
    """Cancel a run through the existing task cancellation surface."""
    engine = _get_engine(request)
    task, task_row = await _load_task_snapshot_projection(engine, run_id)
    if task_row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    # Try the orchestrator cancel path first (requires state_manager).
    if task is not None:
        try:
            return await cancel_task(request, run_id)
        except Exception:
            pass  # Fall through to direct DB update

    # Fallback: directly mark cancelled in database (handles tasks not
    # loaded in the in-memory state manager).
    current_status = str(task_row.get("status", "") or "").strip().lower()
    if current_status in ("completed", "failed", "cancelled"):
        return {"status": "ok", "message": f"Run already {current_status}."}
    await engine.database.update_task_status(run_id, "cancelled")
    engine.event_bus.emit(Event(
        event_type=TASK_CANCEL_REQUESTED,
        task_id=run_id,
        data={"requested": True, "path": "api_fallback"},
    ))
    return {"status": "ok", "message": f"Run {run_id} cancelled."}


@router.delete("/runs/{run_id}")
async def delete_run(request: Request, run_id: str):
    """Remove a run from the database.

    For non-terminal runs, force-cancel them first so the orchestrator
    can clean up, then delete the database rows.
    """
    engine = _get_engine(request)
    task, task_row = await _load_task_snapshot_projection(engine, run_id)
    if task_row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    status = str(task_row.get("status", "") or "").strip().lower()
    # Force-cancel if still active
    if status not in ("completed", "failed", "cancelled"):
        try:
            cancel_result = await cancel_run(request, run_id)
        except Exception:
            cancel_result = None
        if engine.task_run_inflight(run_id):
            for _ in range(100):
                if not engine.task_run_inflight(run_id):
                    break
                await asyncio.sleep(0.01)
        if engine.task_run_inflight(run_id):
            raise HTTPException(
                status_code=409,
                detail=f"Run {run_id} is still shutting down; retry deletion shortly.",
            )
        if isinstance(cancel_result, dict):
            status = str(cancel_result.get("task_status", "") or status).strip().lower()

    await engine.database.execute(
        "DELETE FROM events WHERE task_id=?", (run_id,),
    )
    await engine.database.execute(
        "DELETE FROM task_runs WHERE task_id=?", (run_id,),
    )
    await engine.database.execute(
        "DELETE FROM tasks WHERE id=?", (run_id,),
    )
    try:
        await asyncio.to_thread(engine.state_manager.delete, run_id)
    except Exception:
        logger.warning("Failed deleting task snapshot for %s", run_id, exc_info=True)
    return {"status": "ok", "message": f"Run {run_id} deleted."}


@router.post("/runs/{run_id}/restart")
async def restart_run(request: Request, run_id: str):
    """Restart a terminal run in place, resuming from saved task state."""
    engine = _get_engine(request)
    task, task_row = await _load_task_snapshot_projection(engine, run_id)
    if task_row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    status = str(task_row.get("status", "") or "").strip().lower()
    if status not in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot restart a run with status '{status}'.",
        )
    row_metadata = _json_object(task_row.get("metadata"))
    context = _json_object(task_row.get("context"))
    goal = str(task_row.get("goal", "") or "").strip() or "Untitled run"
    run_workspace = str(task_row.get("workspace_path", "") or "").strip()
    source_workspace_root = str(row_metadata.get("source_workspace_root", "") or "").strip()
    process_name = (
        str(task_row.get("process_name", "") or "").strip()
        or str(row_metadata.get("process", "") or "").strip()
        or None
    )
    approval_mode = str(task_row.get("approval_mode", "") or "auto").strip()
    callback_url = str(task_row.get("callback_url", "") or "").strip() or None
    next_run_id = f"run-{uuid.uuid4().hex[:12]}"
    persisted_plan = Plan()
    raw_plan = task_row.get("plan")
    if raw_plan:
        try:
            plan_data = json.loads(str(raw_plan))
            if isinstance(plan_data, dict):
                persisted_plan = _plan_from_json_dict(plan_data)
        except Exception:
            logger.warning("Failed to parse persisted plan for restart %s", run_id, exc_info=True)

    if task is None:
        task = Task(
            id=run_id,
            goal=goal,
            status=TaskStatus.PENDING,
            workspace=run_workspace,
            plan=persisted_plan,
            approval_mode=approval_mode,
            callback_url=callback_url or "",
            context=context,
            metadata=row_metadata,
            created_at=str(task_row.get("created_at", "") or ""),
            updated_at=str(task_row.get("updated_at", "") or ""),
            completed_at=str(task_row.get("completed_at", "") or ""),
        )

    merged_metadata = _merge_task_metadata(
        row_metadata,
        task.metadata if isinstance(task.metadata, dict) else None,
    )
    task.goal = goal
    task.workspace = run_workspace
    task.approval_mode = approval_mode
    task.callback_url = callback_url or ""
    task.context = context
    task.metadata = merged_metadata
    source_workspace_root = str(task.metadata.get("source_workspace_root", "") or "").strip()
    process_name = (
        str(process_name or "").strip()
        or str(task.metadata.get("process", "") or "").strip()
        or None
    )
    previous_run_id = str(task.metadata.get("run_id", "") or "").strip()
    for key in (
        "cancel_reason",
        "paused_from_status",
        "shutdown_paused",
        "shutdown_pause_reason",
        "blocked_subtasks",
    ):
        task.metadata.pop(key, None)
    task.metadata["run_id"] = next_run_id
    task.metadata["restart_count"] = int(task.metadata.get("restart_count", 0) or 0) + 1
    task.metadata["restarted_from_run_id"] = previous_run_id
    task.metadata["last_restarted_at"] = _now_iso()

    task, restart_error = _prepare_task_for_restart_from_failure(task)
    if task is None:
        raise HTTPException(status_code=409, detail=restart_error or "Failed to restart run.")

    await _persist_task_snapshot(engine, task)

    process = await engine._resolve_process_definition(
        process_name=process_name or "",
        workspace=(
            Path(source_workspace_root or run_workspace)
            if (source_workspace_root or run_workspace)
            else None
        ),
    )
    engine.event_bus.emit(Event(
        event_type=TASK_RESTARTED,
        task_id=task.id,
        data={
            "run_id": next_run_id,
            "previous_run_id": previous_run_id,
            "message": "Run resumed from saved task state.",
            "recovered": True,
        },
    ))
    await engine.submit_task(
        task=task,
        process=process,
        process_name=process_name or "",
        run_id=next_run_id,
        recovered=True,
    )
    return TaskCreateResponse(
        task_id=task.id,
        status=TaskStatus.PENDING.value,
        message=f"Restarted run {task.id}.",
        run_id=next_run_id,
    )


@router.post("/runs/{run_id}/pause")
async def pause_run(request: Request, run_id: str):
    """Pause a run through the existing task pause surface."""
    engine = _get_engine(request)
    if not await _task_state_exists(engine, run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return await pause_task(request, run_id)


@router.post("/runs/{run_id}/resume")
async def resume_run(request: Request, run_id: str):
    """Resume a run through the existing task resume surface."""
    engine = _get_engine(request)
    if not await _task_state_exists(engine, run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return await resume_task(request, run_id)


@router.post("/runs/{run_id}/message")
async def send_run_message(request: Request, run_id: str, body: ConversationMessageRequest):
    """Send an operator message to a run via the task conversation surface."""
    engine = _get_engine(request)
    if not await _task_state_exists(engine, run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return await send_conversation_message(request, run_id, body)


@router.get("/settings")
async def get_settings(request: Request):
    """Return runtime-config entries grouped for progressive disclosure."""
    engine = _get_engine(request)
    return _settings_payload(engine)


@router.patch("/settings")
async def patch_settings(request: Request, body: SettingsPatchRequest):
    """Update global settings through the runtime config registry/store."""
    engine = _get_engine(request)
    if not _request_is_local(request):
        raise HTTPException(status_code=403, detail="Settings mutation requires a local caller.")
    try:
        for path in body.clear_paths:
            clean_path = str(path or "").strip()
            if not clean_path:
                continue
            if body.persist:
                engine.config_runtime_store.reset_persisted_value(clean_path)
            else:
                engine.config_runtime_store.clear_runtime_value(clean_path)
        for path, raw_value in body.values.items():
            clean_path = str(path or "").strip()
            if not clean_path:
                continue
            if body.persist:
                engine.config_runtime_store.persist_value(clean_path, raw_value)
            else:
                engine.config_runtime_store.set_runtime_value(clean_path, raw_value)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=f"Unknown setting path: {e.args[0]}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ConfigPersistConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ConfigPersistDisabledError as e:
        raise HTTPException(status_code=400, detail=str(e))
    engine.refresh_config_from_runtime_store()
    return _settings_payload(engine)


@router.get("/workspaces/{workspace_id}/settings")
async def get_workspace_settings(request: Request, workspace_id: str):
    """Return workspace-scoped overrides/preferences."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    settings = await engine.workspace_registry.get_workspace_settings(workspace_id)
    return {
        "workspace": workspace,
        **settings,
    }


@router.patch("/workspaces/{workspace_id}/settings")
async def patch_workspace_settings(
    request: Request,
    workspace_id: str,
    body: WorkspaceSettingsPatchRequest,
):
    """Persist workspace-local settings without introducing a parallel config system."""
    engine = _get_engine(request)
    workspace = await _require_workspace(engine, workspace_id)
    updated = await engine.workspace_registry.patch_workspace_settings(
        workspace_id,
        overrides=body.overrides,
    )
    return {
        "workspace": workspace,
        **updated,
    }


# --- System ---


@router.get("/models", response_model=list[ModelInfo])
async def list_models(request: Request):
    """List available models and their health status."""
    engine = _get_engine(request)
    providers = engine.model_router.list_providers()
    result = []
    for p in providers:
        caps = p.get("capabilities")
        caps_response = None
        if caps:
            caps_response = ModelCapabilitiesResponse(
                vision=caps.get("vision", False),
                native_pdf=caps.get("native_pdf", False),
                thinking=caps.get("thinking", False),
                citations=caps.get("citations", False),
                audio_input=caps.get("audio_input", False),
                audio_output=caps.get("audio_output", False),
            )
        result.append(ModelInfo(
            name=p["name"],
            model=p["model"],
            model_id=p.get("model_id", ""),
            tier=p["tier"],
            roles=p["roles"],
            capabilities=caps_response,
        ))
    return result


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(request: Request):
    """List available tools and schemas."""
    engine = _get_engine(request)
    return _tool_info_rows(engine)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """System health check."""
    engine = _get_engine(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        ready=True,
        runtime_role=engine.runtime_role,
    )


@router.get("/config")
async def get_config(request: Request):
    """Current configuration (redacted)."""
    engine = _get_engine(request)
    config = engine.config

    def _caps_dict(m):
        caps = m.resolved_capabilities
        return {
            "vision": caps.vision,
            "native_pdf": caps.native_pdf,
            "thinking": caps.thinking,
        }

    return {
        "server": {"host": config.server.host, "port": config.server.port},
        "models": {
            name: {
                "provider": m.provider,
                "model": m.model,
                "roles": m.roles,
                "max_tokens": m.max_tokens,
                "capabilities": _caps_dict(m),
            }
            for name, m in config.models.items()
        },
        "execution": {
            "max_loop_iterations": config.execution.max_loop_iterations,
            "max_subtask_retries": config.execution.max_subtask_retries,
            "enable_global_run_budget": bool(config.execution.enable_global_run_budget),
            "executor_completion_contract_mode": str(
                config.execution.executor_completion_contract_mode,
            ),
            "planner_degraded_mode": str(config.execution.planner_degraded_mode),
            "enable_sqlite_remediation_queue": bool(
                config.execution.enable_sqlite_remediation_queue,
            ),
            "enable_durable_task_runner": bool(
                config.execution.enable_durable_task_runner,
            ),
            "enable_mutation_idempotency": bool(
                config.execution.enable_mutation_idempotency,
            ),
            "enable_slo_metrics": bool(config.execution.enable_slo_metrics),
        },
        "telemetry": {
            "mode": str(getattr(config.telemetry, "mode", "active")),
            "runtime_override_enabled": bool(
                getattr(config.telemetry, "runtime_override_enabled", True),
            ),
            "runtime_override_api_enabled": bool(
                getattr(config.telemetry, "runtime_override_api_enabled", False),
            ),
            "persist_runtime_override": bool(
                getattr(config.telemetry, "persist_runtime_override", False),
            ),
            "debug_diagnostics_rate_per_minute": int(
                getattr(config.telemetry, "debug_diagnostics_rate_per_minute", 120),
            ),
            "debug_diagnostics_burst": int(
                getattr(config.telemetry, "debug_diagnostics_burst", 30),
            ),
        },
    }


@router.get("/settings/telemetry", response_model=TelemetrySettingsResponse)
async def get_telemetry_settings(request: Request):
    """Return configured, runtime-override, and effective telemetry mode state."""
    engine = _get_engine(request)
    return _build_telemetry_settings_response(engine)


@router.patch("/settings/telemetry", response_model=TelemetrySettingsResponse)
async def patch_telemetry_settings(request: Request, body: TelemetrySettingsPatchRequest):
    """Update runtime telemetry mode with loopback + admin-token guard."""
    engine = _get_engine(request)
    _require_telemetry_mutation_access(request, engine)
    actor = str(request.headers.get("x-loom-actor", "") or "").strip() or "api"
    source = f"api:{request.url.path}"
    try:
        engine.set_runtime_telemetry_mode(
            mode_input=body.mode,
            actor=actor,
            source=source,
            persist=bool(body.persist),
        )
    except TelemetryPersistConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except TelemetryPersistDisabledError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _build_telemetry_settings_response(engine)


@router.get("/slo")
async def get_slo_snapshot(request: Request):
    """Return aggregate SLO-oriented metrics derived from persisted runs/events."""
    engine = _get_engine(request)
    if not bool(getattr(engine.config.execution, "enable_slo_metrics", False)):
        return {
            "enabled": False,
            "message": "SLO metrics disabled. Set execution.enable_slo_metrics=true.",
        }
    snapshot = await engine.database.compute_slo_snapshot()
    return {"enabled": True, **snapshot}
