"""Runner execution-loop extraction helpers."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from pathlib import Path
from typing import Any

from loom.auth.runtime import AuthResolutionError, build_run_auth_context
from loom.engine.verification import Check, VerificationResult
from loom.events.types import (
    FORBIDDEN_CANONICAL_WRITE_BLOCKED,
    SEALED_UNEXPECTED_MUTATION_DETECTED,
    TOOL_CALL_COMPLETED,
    TOOL_CALL_DEDUPLICATED,
    TOOL_CALL_STARTED,
)
from loom.models.base import ModelEmptyResponseError, ModelResponse
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import (
    ModelRetryPolicy,
    build_model_retry_event_payload,
    call_with_model_retry,
)
from loom.recovery.questions import QuestionRequest
from loom.state.evidence import (
    extract_evidence_records,
    merge_evidence_records,
    summarize_evidence_records,
)
from loom.state.task_state import Subtask, Task
from loom.tools.registry import ToolResult
from loom.tools.workspace import ChangeLog

from . import session as runner_session
from . import tool_routing as runner_tool_routing
from .types import SubtaskResult, SubtaskResultStatus, ToolCallRecord

logger = logging.getLogger(__name__)
compactor_event_context: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("runner_compactor_event_context", default=None)
)
INFRA_MESSAGE_CONTRACT_VIOLATION = "infra_message_contract_violation"
_INFRA_MODEL_REASON_CODES = {
    ModelEmptyResponseError.reason_code,
    "model_stream_empty",
    "infra_runner_context_unfit",
    INFRA_MESSAGE_CONTRACT_VIOLATION,
}
_TERMINAL_WEB_SOURCE_MARKERS = (
    "http 401",
    "http 403",
    "http 404",
    "http 410",
    "anti-bot denied",
    "access denied",
    "login required",
    "authentication required",
)

# Backwards-compatible name used by runner internals.
_COMPACTOR_EVENT_CONTEXT = compactor_event_context


def _response_has_no_assistant_output(response: ModelResponse) -> bool:
    text = str(getattr(response, "text", "") or "").strip()
    tool_calls = getattr(response, "tool_calls", None)
    finish_reason = str(getattr(response, "finish_reason", "") or "").strip()
    return not text and not tool_calls and not finish_reason


def _web_target_key(arguments: dict[str, Any]) -> str:
    """Return a stable fetch target independent of extraction/query hints."""
    target = str(arguments.get("url", "") or "").strip()
    if not target:
        return ""
    return target.split("#", 1)[0].rstrip("/").lower()


def _is_terminal_web_source_failure(error: object) -> bool:
    """Return whether retrying the same URL with the same fetch method is futile."""
    normalized = " ".join(str(error or "").strip().lower().split())
    return bool(
        normalized
        and any(marker in normalized for marker in _TERMINAL_WEB_SOURCE_MARKERS)
    )


def _exhausted_web_target_result(*, url: str, prior_error: str) -> ToolResult:
    return ToolResult.fail(
        "SOURCE METHOD EXHAUSTED: this URL already failed with a non-retryable "
        f"access/not-found response ({prior_error}). Do not fetch {url} again. "
        "Use web_search to discover alternate public sources, fetch a different URL, "
        "or proceed with other evidence."
    )


def _response_raw_dict(response: ModelResponse) -> dict[str, Any]:
    raw = getattr(response, "raw", None)
    return dict(raw) if isinstance(raw, dict) else {}


def _model_enforces_context_window(model: Any) -> bool:
    """Return whether a provider should be hard-stopped by context preflight."""
    explicit = getattr(model, "enforces_context_window", None)
    if isinstance(explicit, bool):
        return explicit
    module_name = str(getattr(type(model), "__module__", "") or "")
    return module_name.startswith("loom.models.")


def _empty_response_metadata(
    *,
    response: ModelResponse,
    request_diag,
    invocation_attempt: int,
    max_attempts: int,
    operation: str,
    iteration: int,
    model_name: str,
    compaction_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    request_payload = request_diag.to_event_payload() if request_diag else {}
    response_payload = collect_response_diagnostics(response).to_event_payload()
    raw = _response_raw_dict(response)
    stream_close_reason = str(raw.get("stream_close_reason", "") or "").strip()
    if not stream_close_reason:
        stream_close_reason = "not_streaming" if operation != "stream" else "unknown"
    reason_code = (
        "model_stream_empty"
        if operation == "stream"
        else ModelEmptyResponseError.reason_code
    )
    provider_status = str(
        raw.get("provider_status", raw.get("http_status", "")) or "",
    ).strip() or "unknown"
    return {
        "reason_code": reason_code,
        "failure_class": reason_code,
        "provider_status": provider_status,
        "stream_close_reason": stream_close_reason,
        "stream_final_chunk_seen": bool(raw.get("stream_final_chunk_seen", False)),
        "stream_chunk_count": int(raw.get("stream_chunk_count", 0) or 0),
        "stream_content_chunk_count": int(
            raw.get("stream_content_chunk_count", 0) or 0,
        ),
        "retry_count": max(0, invocation_attempt - 1),
        "invocation_attempt": invocation_attempt,
        "invocation_max_attempts": max_attempts,
        "iteration": iteration,
        "operation": operation,
        "model": model_name,
        "request_bytes": int(request_payload.get("request_bytes", 0) or 0),
        "request_est_tokens": int(request_payload.get("request_est_tokens", 0) or 0),
        "request_size_tier": str(request_payload.get("request_size_tier", "") or ""),
        "message_count": int(request_payload.get("message_count", 0) or 0),
        "tool_count": int(request_payload.get("tool_count", 0) or 0),
        "response_chars": int(response_payload.get("response_chars", 0) or 0),
        "response_tool_calls": int(
            response_payload.get("response_tool_calls", 0) or 0,
        ),
        "response_finish_reason": str(
            response_payload.get("response_finish_reason", "") or "",
        ),
        "compaction_policy_mode": str(
            compaction_diagnostics.get("compaction_policy_mode", "") or "",
        ),
        "compaction_terminal_state": str(
            compaction_diagnostics.get("compaction_terminal_state", "") or "",
        ),
        "compaction_pressure_ratio": float(
            compaction_diagnostics.get("compaction_pressure_ratio", 0.0) or 0.0,
        ),
        "compaction_skipped_reason": str(
            compaction_diagnostics.get("compaction_skipped_reason", "") or "",
        ),
    }


class ModelMessageContractError(Exception):
    """Raised when runner history would violate provider tool-message rules."""

    reason_code = INFRA_MESSAGE_CONTRACT_VIOLATION

    def __init__(self, message: str, *, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.metadata = dict(metadata or {})


def _model_failure_metadata(error: BaseException) -> dict[str, Any]:
    if isinstance(error, (ModelEmptyResponseError, ModelMessageContractError)):
        return dict(getattr(error, "metadata", {}) or {})
    return {}


def _tool_call_ids(tool_calls: object) -> list[str]:
    if not isinstance(tool_calls, list):
        return []
    ids: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            ids.append("")
            continue
        call_id = str(call.get("id", "") or "").strip()
        ids.append(call_id)
    return ids


def _message_contract_metadata(
    *,
    request_payload: dict[str, Any],
    invocation_attempt: int,
    max_attempts: int,
    operation: str,
    iteration: int,
    model_name: str,
    compaction_diagnostics: dict[str, Any],
    violation: str,
    message_index: int,
    message_role: str,
    assistant_message_index: int = -1,
    tool_call_id: str = "",
    expected_tool_call_ids: list[str] | None = None,
    missing_tool_call_ids: list[str] | None = None,
) -> dict[str, Any]:
    metadata = {
        "reason_code": INFRA_MESSAGE_CONTRACT_VIOLATION,
        "failure_class": "message_contract_violation",
        "provider_status": "not_sent",
        "stream_close_reason": "not_started",
        "retry_count": max(0, invocation_attempt - 1),
        "invocation_attempt": invocation_attempt,
        "invocation_max_attempts": max_attempts,
        "iteration": iteration,
        "operation": operation,
        "model": model_name,
        "request_bytes": int(request_payload.get("request_bytes", 0) or 0),
        "request_est_tokens": int(request_payload.get("request_est_tokens", 0) or 0),
        "request_size_tier": str(request_payload.get("request_size_tier", "") or ""),
        "message_count": int(request_payload.get("message_count", 0) or 0),
        "tool_count": int(request_payload.get("tool_count", 0) or 0),
        "compaction_policy_mode": str(
            compaction_diagnostics.get("compaction_policy_mode", "") or "",
        ),
        "compaction_terminal_state": str(
            compaction_diagnostics.get("compaction_terminal_state", "") or "",
        ),
        "compaction_pressure_ratio": float(
            compaction_diagnostics.get("compaction_pressure_ratio", 0.0) or 0.0,
        ),
        "compaction_skipped_reason": str(
            compaction_diagnostics.get("compaction_skipped_reason", "") or "",
        ),
        "message_contract_violation": violation,
        "message_index": message_index,
        "message_role": message_role,
    }
    if assistant_message_index >= 0:
        metadata["assistant_message_index"] = assistant_message_index
    if tool_call_id:
        metadata["tool_call_id"] = tool_call_id
    if expected_tool_call_ids is not None:
        metadata["expected_tool_call_ids"] = list(expected_tool_call_ids)
    if missing_tool_call_ids is not None:
        metadata["missing_tool_call_ids"] = list(missing_tool_call_ids)
    return metadata


def _validate_model_message_contract(
    messages: list[dict],
    *,
    request_payload: dict[str, Any],
    invocation_attempt: int,
    max_attempts: int,
    operation: str,
    iteration: int,
    model_name: str,
    compaction_diagnostics: dict[str, Any],
) -> None:
    pending_ids: list[str] = []
    pending_set: set[str] = set()
    assistant_idx = -1

    def _raise(
        message: str,
        *,
        violation: str,
        message_index: int,
        message_role: str,
        tool_call_id: str = "",
        expected_tool_call_ids: list[str] | None = None,
        missing_tool_call_ids: list[str] | None = None,
    ) -> None:
        raise ModelMessageContractError(
            message,
            metadata=_message_contract_metadata(
                request_payload=request_payload,
                invocation_attempt=invocation_attempt,
                max_attempts=max_attempts,
                operation=operation,
                iteration=iteration,
                model_name=model_name,
                compaction_diagnostics=compaction_diagnostics,
                violation=violation,
                message_index=message_index,
                message_role=message_role,
                assistant_message_index=assistant_idx,
                tool_call_id=tool_call_id,
                expected_tool_call_ids=expected_tool_call_ids,
                missing_tool_call_ids=missing_tool_call_ids,
            ),
        )

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "") or "").strip().lower()
        if role == "assistant":
            if pending_ids:
                _raise(
                    "Assistant tool call history is missing required tool responses.",
                    violation="assistant_tool_calls_missing_responses",
                    message_index=idx,
                    message_role=role,
                    expected_tool_call_ids=pending_ids,
                    missing_tool_call_ids=pending_ids,
                )
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                assistant_idx = idx
                call_ids = _tool_call_ids(tool_calls)
                missing_id_positions = [
                    pos for pos, call_id in enumerate(call_ids) if not call_id
                ]
                if missing_id_positions:
                    _raise(
                        "Assistant tool call history contains a blank tool_call id.",
                        violation="assistant_tool_call_missing_id",
                        message_index=idx,
                        message_role=role,
                        expected_tool_call_ids=call_ids,
                        missing_tool_call_ids=[str(pos) for pos in missing_id_positions],
                    )
                if len(set(call_ids)) != len(call_ids):
                    _raise(
                        "Assistant tool call history contains duplicate tool_call ids.",
                        violation="assistant_tool_call_duplicate_id",
                        message_index=idx,
                        message_role=role,
                        expected_tool_call_ids=call_ids,
                    )
                assistant_idx = idx
                pending_ids = call_ids
                pending_set = set(call_ids)
            else:
                assistant_idx = -1
                pending_ids = []
                pending_set = set()
            continue

        if role == "tool":
            tool_call_id = str(msg.get("tool_call_id", "") or "").strip()
            if not pending_ids:
                _raise(
                    "Tool response has no matching assistant tool_call id.",
                    violation=(
                        "tool_message_missing_tool_call_id"
                        if not tool_call_id
                        else "tool_call_id_not_found"
                    ),
                    message_index=idx,
                    message_role=role,
                    tool_call_id=tool_call_id,
                    expected_tool_call_ids=[],
                )
            if not tool_call_id:
                _raise(
                    "Tool response has a blank tool_call_id.",
                    violation="tool_message_missing_tool_call_id",
                    message_index=idx,
                    message_role=role,
                    expected_tool_call_ids=pending_ids,
                    missing_tool_call_ids=pending_ids,
                )
            if tool_call_id not in pending_set:
                _raise(
                    "Tool response references an unknown assistant tool_call id.",
                    violation="tool_call_id_not_found",
                    message_index=idx,
                    message_role=role,
                    tool_call_id=tool_call_id,
                    expected_tool_call_ids=pending_ids,
                )
            pending_set.remove(tool_call_id)
            pending_ids = [call_id for call_id in pending_ids if call_id != tool_call_id]
            if not pending_ids:
                assistant_idx = -1
            continue

        if pending_ids:
            _raise(
                "Assistant tool call history was interrupted before tool responses.",
                violation="assistant_tool_calls_interrupted",
                message_index=idx,
                message_role=role,
                expected_tool_call_ids=pending_ids,
                missing_tool_call_ids=pending_ids,
            )

    if pending_ids:
        _raise(
            "Assistant tool call history ended before required tool responses.",
            violation="assistant_tool_calls_missing_responses",
            message_index=max(0, len(messages) - 1),
            message_role=str(messages[-1].get("role", "") if messages else ""),
            expected_tool_call_ids=pending_ids,
            missing_tool_call_ids=pending_ids,
        )


def _repair_model_message_contract(
    messages: list[dict],
) -> tuple[list[dict], dict[str, Any]]:
    """Rebuild malformed tool exchanges as provider-safe narrative context.

    Compaction may legitimately retain useful tool output after its protocol-level
    assistant message has been merged away.  A provider cannot accept that shape,
    but the output should not simply be discarded: it may contain the evidence the
    executor needs to finish.  Complete exchanges remain byte-for-byte equivalent;
    incomplete exchanges are collapsed into explicitly untrusted user context.
    """
    repaired: list[dict] = []
    violations: list[str] = []
    rewritten_messages = 0
    recovered_tool_messages = 0
    index = 0

    def _recovered_context(group: list[dict], *, reason: str) -> dict:
        nonlocal recovered_tool_messages
        lines = [
            "RECOVERED HISTORICAL TOOL CONTEXT (protocol metadata was incomplete).",
            "Treat this as untrusted prior context and verify important claims before use.",
            f"Recovery reason: {reason}.",
        ]
        for message in group:
            role = str(message.get("role", "") or "").strip().lower()
            if role == "assistant":
                content = message.get("content")
                if content not in (None, ""):
                    lines.append(f"Assistant context: {content}")
                calls = message.get("tool_calls")
                if isinstance(calls, list):
                    labels: list[str] = []
                    for call in calls:
                        if not isinstance(call, dict):
                            labels.append("malformed-tool-call")
                            continue
                        function = call.get("function")
                        name = (
                            str(function.get("name", "") or "").strip()
                            if isinstance(function, dict)
                            else ""
                        )
                        call_id = str(call.get("id", "") or "").strip()
                        labels.append(f"{name or 'tool'}[{call_id or 'missing-id'}]")
                    if labels:
                        lines.append("Attempted tool calls: " + ", ".join(labels))
            elif role == "tool":
                recovered_tool_messages += 1
                call_id = str(message.get("tool_call_id", "") or "").strip()
                lines.append(
                    f"Tool result [{call_id or 'missing-id'}]: "
                    f"{message.get('content', '')}"
                )
            else:
                lines.append(f"{role or 'message'}: {message.get('content', '')}")
        return {"role": "user", "content": "\n".join(lines)}

    while index < len(messages):
        message = messages[index]
        if not isinstance(message, dict):
            violations.append("non_mapping_message")
            rewritten_messages += 1
            index += 1
            continue

        role = str(message.get("role", "") or "").strip().lower()
        tool_calls = message.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            group = [message]
            cursor = index + 1
            while cursor < len(messages):
                candidate = messages[cursor]
                if not isinstance(candidate, dict):
                    break
                if str(candidate.get("role", "") or "").strip().lower() != "tool":
                    break
                group.append(candidate)
                cursor += 1

            call_ids = _tool_call_ids(tool_calls)
            response_ids = [
                str(item.get("tool_call_id", "") or "").strip()
                for item in group[1:]
            ]
            well_formed = (
                all(call_ids)
                and len(set(call_ids)) == len(call_ids)
                and len(response_ids) == len(call_ids)
                and all(response_ids)
                and len(set(response_ids)) == len(response_ids)
                and set(response_ids) == set(call_ids)
            )
            if well_formed:
                repaired.extend(dict(item) for item in group)
            else:
                if not all(call_ids):
                    reason = "assistant_tool_call_missing_id"
                elif len(set(call_ids)) != len(call_ids):
                    reason = "assistant_tool_call_duplicate_id"
                elif any(response_id not in set(call_ids) for response_id in response_ids):
                    reason = "tool_call_id_not_found"
                else:
                    reason = "assistant_tool_calls_missing_responses"
                violations.append(reason)
                rewritten_messages += len(group)
                repaired.append(_recovered_context(group, reason=reason))
            index = cursor
            continue

        if role == "tool":
            violations.append("tool_call_id_not_found")
            rewritten_messages += 1
            repaired.append(
                _recovered_context([message], reason="tool_call_id_not_found"),
            )
            index += 1
            continue

        repaired.append(dict(message))
        index += 1

    unique_violations = list(dict.fromkeys(violations))
    return repaired, {
        "message_contract_repair_applied": bool(unique_violations),
        "message_contract_repair_status": (
            "recovered" if unique_violations else "not_needed"
        ),
        "message_contract_repair_violations": unique_violations,
        "message_contract_repair_messages_rewritten": rewritten_messages,
        "message_contract_repair_tool_messages_recovered": recovered_tool_messages,
        "message_contract_repair_message_count_before": len(messages),
        "message_contract_repair_message_count_after": len(repaired),
    }


def _context_terminal_state(diagnostics: dict[str, Any]) -> str:
    return str(diagnostics.get("compaction_terminal_state", "") or "").strip().lower()


def _context_policy_mode(diagnostics: dict[str, Any]) -> str:
    return str(diagnostics.get("compaction_policy_mode", "") or "").strip().lower()


def _is_disabled_unfit_context(diagnostics: dict[str, Any]) -> bool:
    return (
        _context_policy_mode(diagnostics) == "off"
        and _context_terminal_state(diagnostics) == "unfit"
    )


async def _try_emergency_context_rescue(
    runner: Any,
    *,
    messages: list[dict],
    tool_schemas: list[dict],
    remaining_seconds: float,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    configured_mode = str(
        getattr(
            runner,
            "_runner_compaction_policy_mode",
            runner.RUNNER_COMPACTION_POLICY_MODE,
        )
        or "",
    ).strip().lower()
    original_mode = getattr(runner, "_runner_compaction_policy_mode", configured_mode)
    runner._reset_compaction_runtime_stats()
    try:
        runner._runner_compaction_policy_mode = "deterministic"
        rescued_messages = await runner._compact_messages_for_model_tiered(
            messages,
            tools=tool_schemas,
            remaining_seconds=remaining_seconds,
        )
        rescued_tools = tool_schemas
        if rescued_tools:
            rescued_tools, tool_schema_prune_report = (
                runner._prune_tool_schemas_for_request_fit(
                    rescued_messages,
                    rescued_tools,
                )
            )
            if isinstance(tool_schema_prune_report, dict):
                diagnostics = dict(getattr(runner, "_last_compaction_diagnostics", {}))
                diagnostics.update({
                    "compaction_tool_schema_pruned": bool(
                        tool_schema_prune_report.get("applied", False),
                    ),
                    "compaction_tool_schema_prune_report": tool_schema_prune_report,
                })
                if bool(tool_schema_prune_report.get("applied", False)):
                    applied_stages = list(
                        diagnostics.get("compaction_applied_stages", []),
                    )
                    if "tool_schema_prune" not in applied_stages:
                        applied_stages.append("tool_schema_prune")
                    diagnostics["compaction_applied_stages"] = applied_stages
                    diagnostics["compaction_stage"] = "tool_schema_prune"
                    diagnostics["compaction_est_tokens_after"] = int(
                        tool_schema_prune_report.get(
                            "request_est_tokens_after",
                            diagnostics.get("compaction_est_tokens_after", 0),
                        ),
                    )
                    context_budget = int(
                        getattr(
                            runner,
                            "_max_model_context_tokens",
                            runner.MAX_MODEL_CONTEXT_TOKENS,
                        ),
                    )
                    diagnostics["compaction_pressure_ratio_after"] = round(
                        int(diagnostics["compaction_est_tokens_after"])
                        / max(1, context_budget),
                        4,
                    )
                    diagnostics["compaction_terminal_state"] = (
                        "degraded_fit"
                        if int(diagnostics["compaction_est_tokens_after"]) <= context_budget
                        else "unfit"
                    )
                runner._last_compaction_diagnostics = diagnostics
    finally:
        runner._runner_compaction_policy_mode = original_mode

    diagnostics = dict(getattr(runner, "_last_compaction_diagnostics", {}))
    diagnostics.update({
        "compaction_emergency_rescue_attempted": True,
        "compaction_emergency_rescue_mode": "deterministic",
        "compaction_policy_mode_configured": configured_mode or "off",
    })
    runner._last_compaction_diagnostics = diagnostics
    return rescued_messages, rescued_tools, diagnostics


async def run_subtask(
    runner,
    task: Task,
    subtask: Subtask,
    *,
    model_tier: int | None = None,
    retry_context: str = "",
    changelog: ChangeLog | None = None,
    prior_successful_tool_calls: list[ToolCallRecord] | None = None,
    prior_evidence_records: list[dict] | None = None,
    expected_deliverables: list[str] | None = None,
    forbidden_deliverables: list[str] | None = None,
    allowed_output_prefixes: list[str] | None = None,
    enforce_deliverable_paths: bool = False,
    edit_existing_only: bool = False,
    retry_strategy: str = "",
    build_run_auth_context_fn=build_run_auth_context,
) -> tuple[SubtaskResult, VerificationResult]:
    """Execute a subtask: prompt → tool loop → verify → extract memory.

    Returns (SubtaskResult, VerificationResult).
    Memory extraction is fire-and-forget — it does not block the return.
    """
    start_time = time.monotonic()
    runner._subtask_deadline_monotonic = (
        start_time + runner._max_subtask_wall_clock_seconds
    )
    runner._reset_compaction_runtime_stats()
    runner._last_compaction_diagnostics = {
        "compaction_policy_mode": str(
            getattr(
                runner,
                "_runner_compaction_policy_mode",
                runner.RUNNER_COMPACTION_POLICY_MODE,
            ),
        ),
        "compaction_stage": "none",
        "compaction_candidate_count": 0,
        "compaction_skipped_reason": "not_started",
    }
    telemetry_counters = runner._new_subtask_telemetry_counters()
    runner._active_subtask_telemetry_counters = telemetry_counters
    compactor_context_token = compactor_event_context.set((task.id, subtask.id))
    try:
        workspace = Path(task.workspace) if task.workspace else None
        read_roots = runner._read_roots_for_task(task, workspace)
        read_path_map = runner._read_path_map_for_task(task, workspace)
        auth_context = None
        try:
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            auth_context = build_run_auth_context_fn(
                workspace=workspace,
                metadata=metadata,
                available_mcp_aliases=set(runner._config.mcp.servers.keys()),
            )
        except AuthResolutionError as e:
            failure_summary = f"Auth preflight failed: {e}"
            result = SubtaskResult(
                status=SubtaskResultStatus.FAILED,
                summary=failure_summary,
                duration_seconds=time.monotonic() - start_time,
                model_used="",
                telemetry_counters=dict(telemetry_counters),
            )
            verification = VerificationResult(
                tier=1,
                passed=False,
                checks=[Check(name="auth_preflight", passed=False, detail=str(e))],
                feedback=failure_summary,
                outcome="fail",
                reason_code="auth_preflight_failed",
                metadata={"auth_error": str(e)},
            )
            return result, verification

        # 1. Assemble prompt
        memory_entries = await runner._memory.query_relevant(task.id, subtask.id)
        evidence_summary = summarize_evidence_records(
            prior_evidence_records or [],
            max_entries=10,
        )
        execution_surface = runner._execution_surface_for_task(task)
        prompt = runner._prompts.build_executor_prompt(
            task=task,
            subtask=subtask,
            state_manager=runner._state,
            memory_entries=memory_entries,
            available_tools=runner._tools.all_schemas(
                auth_context=auth_context,
                execution_surface=execution_surface,
                runnable_only=True,
            ),
            evidence_ledger_summary=evidence_summary,
        )
        if retry_context:
            prompt = prompt + "\n\n" + retry_context

        # 2. Select model
        effective_tier = model_tier if model_tier is not None else subtask.model_tier
        model = runner._router.select(tier=effective_tier, role="executor")

        # 3. Tool-calling loop
        session = runner_session.new_runner_session(
            prompt=prompt,
            prior_successful_tool_calls=prior_successful_tool_calls,
            prior_evidence_records=prior_evidence_records,
        )
        messages = session.messages
        tool_calls_record = session.tool_calls_record
        evidence_records_current = session.evidence_records_current
        known_evidence_ids = session.known_evidence_ids
        historical_successful_tool_calls = session.historical_successful_tool_calls
        streaming = runner._config.execution.enable_streaming
        canonical_deliverables = runner._normalize_deliverable_paths(
            expected_deliverables or [],
            workspace=workspace,
        )
        canonical_forbidden_deliverables = runner._normalize_deliverable_paths(
            forbidden_deliverables or [],
            workspace=workspace,
        )
        normalized_allowed_output_prefixes = runner._normalize_deliverable_paths(
            allowed_output_prefixes or [],
            workspace=workspace,
        )
        canonical_deliverable_set = set(canonical_deliverables)
        one_shot_direct_deliverable_mode = (
            bool(canonical_deliverables)
            and not normalized_allowed_output_prefixes
        )
        touched_canonical_deliverables: set[str] = set()
        completion_only_after_deliverables = False
        completion_only_instruction_sent = False
        iteration_budget = runner._tool_iteration_budget(
            subtask=subtask,
            retry_strategy=retry_strategy,
            has_expected_deliverables=bool(canonical_deliverables),
            base_budget=runner._max_tool_iterations,
        )
        last_model_failure_metadata: dict[str, Any] = {}
        last_model_failure_reason_code = ""
        checkpoint_instruction_sent = False

        for iteration in range(iteration_budget):
            if not await runner._wait_for_task_control_window(task):
                session.interruption_reason = "Execution cancelled before completion."
                break
            # Wall-clock timeout check
            remaining_seconds = runner._remaining_subtask_seconds()
            if remaining_seconds <= 0:
                session.interruption_reason = (
                    "Execution exceeded subtask time budget "
                    f"({runner._max_subtask_wall_clock_seconds}s) before completion."
                )
                break
            remaining_iterations = iteration_budget - iteration
            checkpoint_reserve = min(
                max(
                    1,
                    int(
                        getattr(
                            runner,
                            "_runner_checkpoint_reserve_iterations",
                            2,
                        )
                        or 2
                    ),
                ),
                iteration_budget,
            )
            if (
                remaining_iterations <= checkpoint_reserve
                and not checkpoint_instruction_sent
                and not completion_only_after_deliverables
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "EXECUTION BUDGET CHECKPOINT: only "
                        f"{remaining_iterations} model/tool turn(s) remain in this pass. "
                        "Stop broad exploration. Reuse current evidence and artifacts. "
                        "Either complete the smallest remaining deliverable work now, or "
                        "return a concise partial completion contract that names exact "
                        "remaining targets so Loom can continue from this checkpoint."
                    ),
                })
                checkpoint_instruction_sent = True
            if completion_only_after_deliverables and not completion_only_instruction_sent:
                session.messages.append({
                    "role": "user",
                    "content": (
                        "CANONICAL DELIVERABLE WRITE COMPLETE: all required deliverables "
                        "for this subtask have already been written once. Do not call "
                        "more tools or modify files. Respond with your final completion "
                        "message only."
                    ),
                })
                completion_only_instruction_sent = True
            if completion_only_after_deliverables:
                tool_schemas = []
            else:
                tool_schemas = runner._tools.all_schemas(
                    auth_context=auth_context,
                    execution_surface=execution_surface,
                    runnable_only=True,
                )
            session.messages = await runner._compact_messages_for_model(
                session.messages,
                tools=tool_schemas,
                remaining_seconds=remaining_seconds,
            )
            if tool_schemas and runner._runner_compaction_mode() != "off":
                tool_schemas, tool_schema_prune_report = (
                    runner._prune_tool_schemas_for_request_fit(
                        session.messages,
                        tool_schemas,
                    )
                )
                if isinstance(tool_schema_prune_report, dict):
                    diagnostics = dict(getattr(runner, "_last_compaction_diagnostics", {}))
                    diagnostics.update({
                        "compaction_tool_schema_pruned": bool(
                            tool_schema_prune_report.get("applied", False),
                        ),
                        "compaction_tool_schema_prune_report": tool_schema_prune_report,
                    })
                    if bool(tool_schema_prune_report.get("applied", False)):
                        applied_stages = list(
                            diagnostics.get("compaction_applied_stages", []),
                        )
                        if "tool_schema_prune" not in applied_stages:
                            applied_stages.append("tool_schema_prune")
                        diagnostics["compaction_applied_stages"] = applied_stages
                        diagnostics["compaction_stage"] = "tool_schema_prune"
                        diagnostics["compaction_est_tokens_after"] = int(
                            tool_schema_prune_report.get(
                                "request_est_tokens_after",
                                diagnostics.get("compaction_est_tokens_after", 0),
                            ),
                        )
                        context_budget = int(
                            getattr(
                                runner,
                                "_max_model_context_tokens",
                                runner.MAX_MODEL_CONTEXT_TOKENS,
                            ),
                        )
                        diagnostics["compaction_pressure_ratio_after"] = round(
                            int(diagnostics["compaction_est_tokens_after"])
                            / max(1, context_budget),
                            4,
                        )
                        diagnostics["compaction_terminal_state"] = (
                            "degraded_fit"
                            if int(diagnostics["compaction_est_tokens_after"]) <= context_budget
                            else "unfit"
                        )
                    runner._last_compaction_diagnostics = diagnostics
            messages = session.messages
            runner._emit_compaction_policy_decision_from_diagnostics(
                task_id=task.id,
                subtask_id=subtask.id,
            )
            operation = "stream" if streaming else "complete"
            session.response = None
            policy = ModelRetryPolicy.from_execution_config(runner._config.execution)
            invocation_attempt = 0
            request_diag = None
            last_model_failure_metadata = {}
            last_model_failure_reason_code = ""
            overflow_fallback_pending = False
            overflow_fallback_attempted = False
            overflow_fallback_report: dict[str, Any] | None = None
            compaction_diagnostics = dict(
                getattr(runner, "_last_compaction_diagnostics", {}),
            )
            if (
                _model_enforces_context_window(model)
                and _is_disabled_unfit_context(compaction_diagnostics)
            ):
                session.messages, tool_schemas, compaction_diagnostics = (
                    await _try_emergency_context_rescue(
                        runner,
                        messages=session.messages,
                        tool_schemas=tool_schemas,
                        remaining_seconds=remaining_seconds,
                    )
                )
                messages = session.messages
                runner._emit_compaction_policy_decision_from_diagnostics(
                    task_id=task.id,
                    subtask_id=subtask.id,
                )
                if _context_terminal_state(compaction_diagnostics) == "unfit":
                    request_diag = collect_request_diagnostics(
                        messages=session.messages,
                        tools=tool_schemas,
                        origin=f"runner.execute_subtask.{operation}.preflight",
                    )
                    request_payload = request_diag.to_event_payload()
                    last_model_failure_reason_code = "infra_runner_context_unfit"
                    last_model_failure_metadata = {
                        "reason_code": last_model_failure_reason_code,
                        "failure_class": "context_unfit",
                        "provider_status": "not_sent",
                        "stream_close_reason": "not_started",
                        "retry_count": 0,
                        "invocation_attempt": 0,
                        "invocation_max_attempts": policy.max_attempts,
                        "iteration": iteration + 1,
                        "operation": operation,
                        "model": model.name,
                        "request_bytes": int(request_payload.get("request_bytes", 0) or 0),
                        "request_est_tokens": int(
                            request_payload.get("request_est_tokens", 0) or 0,
                        ),
                        "request_size_tier": str(
                            request_payload.get("request_size_tier", "") or "",
                        ),
                        "message_count": int(request_payload.get("message_count", 0) or 0),
                        "tool_count": int(request_payload.get("tool_count", 0) or 0),
                        "compaction_policy_mode": str(
                            compaction_diagnostics.get("compaction_policy_mode", "")
                            or "",
                        ),
                        "compaction_policy_mode_configured": str(
                            compaction_diagnostics.get(
                                "compaction_policy_mode_configured",
                                "off",
                            )
                            or "",
                        ),
                        "compaction_terminal_state": "unfit",
                        "compaction_pressure_ratio": float(
                            compaction_diagnostics.get("compaction_pressure_ratio", 0.0)
                            or 0.0,
                        ),
                        "compaction_skipped_reason": str(
                            compaction_diagnostics.get(
                                "compaction_skipped_reason",
                                "",
                            )
                            or "",
                        ),
                        "compaction_emergency_rescue_attempted": bool(
                            compaction_diagnostics.get(
                                "compaction_emergency_rescue_attempted",
                                False,
                            ),
                        ),
                        "compaction_emergency_rescue_mode": str(
                            compaction_diagnostics.get(
                                "compaction_emergency_rescue_mode",
                                "",
                            )
                            or "",
                        ),
                    }
                    runner._emit_model_event(
                        task_id=task.id,
                        subtask_id=subtask.id,
                        model_name=model.name,
                        phase="done",
                        details={
                            "origin": request_diag.origin,
                            **last_model_failure_metadata,
                        },
                    )
                    session.interruption_reason = (
                        "Model request preflight failed: context fit remained unfit "
                        "after emergency compaction rescue."
                    )
                    break

            session.messages, contract_repair = _repair_model_message_contract(
                session.messages,
            )
            messages = session.messages
            if contract_repair["message_contract_repair_applied"]:
                compaction_diagnostics = dict(
                    getattr(runner, "_last_compaction_diagnostics", {}),
                )
                compaction_diagnostics.update(contract_repair)
                runner._last_compaction_diagnostics = compaction_diagnostics
                repair_diag = collect_request_diagnostics(
                    messages=session.messages,
                    tools=tool_schemas,
                    origin=f"runner.execute_subtask.{operation}.contract_repair",
                )
                runner._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="recovered",
                    details={
                        "origin": repair_diag.origin,
                        "iteration": iteration + 1,
                        "operation": operation,
                        "provider_status": "not_sent",
                        "recovery_action": "rebuild_provider_safe_transcript",
                        **contract_repair,
                        **repair_diag.to_event_payload(),
                    },
                )

            async def _invoke_model():
                nonlocal invocation_attempt, request_diag
                invocation_attempt += 1
                runner._increment_subtask_counter("model_invocations")
                request_diag = collect_request_diagnostics(
                    messages=session.messages,
                    tools=tool_schemas,
                    origin=f"runner.execute_subtask.{operation}",
                )
                request_payload = request_diag.to_event_payload()
                _validate_model_message_contract(
                    session.messages,
                    request_payload=request_payload,
                    invocation_attempt=invocation_attempt,
                    max_attempts=policy.max_attempts,
                    operation=operation,
                    iteration=iteration + 1,
                    model_name=model.name,
                    compaction_diagnostics=dict(
                        getattr(runner, "_last_compaction_diagnostics", {}),
                    ),
                )
                runner._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="start",
                    details={
                        **request_payload,
                        "iteration": iteration + 1,
                        "operation": operation,
                        "invocation_attempt": invocation_attempt,
                        "invocation_max_attempts": policy.max_attempts,
                        "remaining_subtask_seconds": round(
                            runner._remaining_subtask_seconds(),
                            3,
                        ),
                        **dict(getattr(runner, "_last_compaction_diagnostics", {})),
                    },
                )
                if streaming:
                    response = await runner._stream_completion(
                        model,
                        session.messages,
                        tool_schemas,
                        task_id=task.id,
                        subtask_id=subtask.id,
                    )
                else:
                    response = await model.complete(
                        session.messages,
                        tools=tool_schemas,
                    )
                if _response_has_no_assistant_output(response):
                    metadata = _empty_response_metadata(
                        response=response,
                        request_diag=request_diag,
                        invocation_attempt=invocation_attempt,
                        max_attempts=policy.max_attempts,
                        operation=operation,
                        iteration=iteration + 1,
                        model_name=model.name,
                        compaction_diagnostics=dict(
                            getattr(runner, "_last_compaction_diagnostics", {}),
                        ),
                    )
                    raise ModelEmptyResponseError(
                        (
                            "Model invocation returned no assistant text, no tool "
                            "calls, and no finish reason."
                        ),
                        metadata=metadata,
                    )
                return response

            def _should_retry_invocation(error: BaseException) -> bool:
                nonlocal overflow_fallback_pending
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    return False
                if isinstance(error, ModelMessageContractError):
                    return False
                if runner._is_model_request_overflow_error(error):
                    if runner._enable_model_overflow_fallback and not overflow_fallback_attempted:
                        overflow_fallback_pending = True
                        return True
                    return False
                return True

            def _on_invocation_failure(
                attempt: int,
                max_attempts: int,
                error: BaseException,
                remaining: int,
            ) -> None:
                nonlocal last_model_failure_metadata
                nonlocal last_model_failure_reason_code
                nonlocal overflow_fallback_pending
                nonlocal overflow_fallback_attempted, overflow_fallback_report
                error_metadata = _model_failure_metadata(error)
                if error_metadata:
                    last_model_failure_metadata = error_metadata
                    last_model_failure_reason_code = str(
                        error_metadata.get("reason_code", "") or "",
                    ).strip()
                if overflow_fallback_pending:
                    overflow_fallback_pending = False
                    overflow_fallback_attempted = True
                    (
                        session.messages,
                        overflow_fallback_report,
                    ) = runner._apply_model_overflow_fallback(session.messages)
                    if overflow_fallback_report:
                        runner._emit_overflow_fallback_telemetry(
                            task_id=task.id,
                            subtask_id=subtask.id,
                            report=overflow_fallback_report,
                        )
                runner._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "origin": request_diag.origin if request_diag else "",
                        "iteration": iteration + 1,
                        "operation": operation,
                        "invocation_attempt": attempt,
                        "invocation_max_attempts": max_attempts,
                        "retry_queue_remaining": remaining,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "overflow_error_detected": runner._is_model_request_overflow_error(error),
                        "overflow_fallback_attempted": overflow_fallback_attempted,
                        **error_metadata,
                        **(overflow_fallback_report or {}),
                    },
                )

            def _on_invocation_retry_scheduled(
                attempt: int,
                max_attempts: int,
                error: BaseException,
                remaining: int,
                delay_seconds: float,
            ) -> None:
                error_metadata = _model_failure_metadata(error)
                runner._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "origin": request_diag.origin if request_diag else "",
                        "iteration": iteration + 1,
                        "operation": operation,
                        "invocation_attempt": attempt,
                        "invocation_max_attempts": max_attempts,
                        "retry_queue_remaining": remaining,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "overflow_error_detected": (
                            runner._is_model_request_overflow_error(error)
                        ),
                        "overflow_fallback_attempted": overflow_fallback_attempted,
                        **error_metadata,
                        **build_model_retry_event_payload(
                            error,
                            delay_seconds=delay_seconds,
                        ),
                        **(overflow_fallback_report or {}),
                    },
                )

            try:
                session.response = await call_with_model_retry(
                    _invoke_model,
                    policy=policy,
                    should_retry=_should_retry_invocation,
                    on_failure=_on_invocation_failure,
                    on_retry_scheduled=_on_invocation_retry_scheduled,
                )
            except Exception as e:
                error_metadata = _model_failure_metadata(e)
                if error_metadata:
                    last_model_failure_metadata = error_metadata
                    last_model_failure_reason_code = str(
                        error_metadata.get("reason_code", "") or "",
                    ).strip()
                session.interruption_reason = (
                    "Model invocation failed after "
                    f"{invocation_attempt} attempt(s): {type(e).__name__}: {e}"
                )
                session.response = None
            else:
                response_diag = collect_response_diagnostics(session.response)
                runner._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "origin": request_diag.origin if request_diag else "",
                        "iteration": iteration + 1,
                        "operation": operation,
                        "invocation_attempt": invocation_attempt,
                        "invocation_max_attempts": policy.max_attempts,
                        **response_diag.to_event_payload(),
                    },
                )
                session.total_tokens += session.response.usage.total_tokens

            messages = session.messages
            if session.interruption_reason:
                break
            if session.response is None:
                session.interruption_reason = (
                    "Execution ended before receiving a model response."
                )
                break

            if session.response.has_tool_calls():
                if not await runner._wait_for_task_control_window(task):
                    session.interruption_reason = "Execution cancelled before completion."
                    break
                # Validate tool calls before execution
                validation = runner._validator.validate_tool_calls(
                    session.response,
                    runner._tools.all_schemas(
                        auth_context=auth_context,
                        execution_surface=execution_surface,
                        runnable_only=True,
                    ),
                )
                if not validation.valid:
                    messages.append({
                        "role": "assistant",
                        "content": session.response.text or "",
                    })
                    messages.append({
                        "role": "system",
                        "content": (
                            f"TOOL CALL ERROR: {validation.error}\n"
                            f"{validation.suggestion}\n"
                            "Please retry with valid tool calls."
                        ),
                    })
                    continue

                # Process validated tool calls
                compact_tool_calls = await runner._serialize_tool_calls_for_message(
                    session.response.tool_calls or []
                )
                messages.append({
                    "role": "assistant",
                    "content": session.response.text or "",
                    "tool_calls": compact_tool_calls,
                })

                for tc in session.response.tool_calls:
                    if not await runner._wait_for_task_control_window(task):
                        session.interruption_reason = "Execution cancelled before completion."
                        break
                    resolved_tool_name, resolved_tool_args, route_metadata = (
                        runner_tool_routing.route_tool_call_for_process(
                            tool_name=tc.name,
                            tool_args=tc.arguments,
                            process=getattr(runner._prompts, "process", None),
                            workspace=workspace,
                            subtask_id=subtask.id,
                            execution_surface=execution_surface,
                        )
                    )
                    runner._emit_tool_event(
                        TOOL_CALL_STARTED, task.id, subtask.id,
                        resolved_tool_name, resolved_tool_args,
                    )
                    tool_call_id = str(getattr(tc, "id", "") or "")
                    tool_obj = runner._tools.get(resolved_tool_name)
                    is_mutating_tool = bool(getattr(tool_obj, "is_mutating", False))
                    mutation_target_arg_keys = tuple(
                        getattr(tool_obj, "mutation_target_arg_keys", ()) or (),
                    )
                    attempted_paths = runner._target_paths_for_policy(
                        tool_name=resolved_tool_name,
                        tool_args=resolved_tool_args,
                        workspace=workspace,
                        is_mutating_tool=is_mutating_tool,
                        mutation_target_arg_keys=mutation_target_arg_keys,
                    )
                    runner._increment_subtask_counter("tool_calls")
                    if is_mutating_tool:
                        runner._increment_subtask_counter("mutating_tool_calls")
                    if (
                        completion_only_after_deliverables
                        and is_mutating_tool
                        and attempted_paths
                    ):
                        policy_error = (
                            "reason_code=forbidden_output_path; "
                            "Canonical deliverable completion violation: "
                            "all required deliverables for this subtask attempt "
                            "have already been written. Do not call mutating tools "
                            "after the canonical write; return the completion "
                            "response instead."
                        )
                    else:
                        policy_error = runner._validate_deliverable_write_policy(
                            tool_name=resolved_tool_name,
                            tool_args=resolved_tool_args,
                            workspace=workspace,
                            is_mutating_tool=is_mutating_tool,
                            mutation_target_arg_keys=mutation_target_arg_keys,
                            expected_deliverables=canonical_deliverables,
                            forbidden_deliverables=canonical_forbidden_deliverables,
                            allowed_output_prefixes=normalized_allowed_output_prefixes,
                            enforce_deliverable_paths=enforce_deliverable_paths,
                            edit_existing_only=edit_existing_only,
                            already_touched_deliverables=(
                                touched_canonical_deliverables
                                if one_shot_direct_deliverable_mode
                                else None
                            ),
                        )
                    if (
                        policy_error
                        and runner._is_forbidden_output_path_error(policy_error)
                    ):
                        sanitized_args, suppressed_keys = (
                            runner_tool_routing.suppress_optional_output_side_effects(
                                tool_name=resolved_tool_name,
                                tool_args=resolved_tool_args,
                            )
                        )
                        if suppressed_keys:
                            sanitized_error = runner._validate_deliverable_write_policy(
                                tool_name=resolved_tool_name,
                                tool_args=sanitized_args,
                                workspace=workspace,
                                is_mutating_tool=is_mutating_tool,
                                mutation_target_arg_keys=mutation_target_arg_keys,
                                expected_deliverables=canonical_deliverables,
                                forbidden_deliverables=canonical_forbidden_deliverables,
                                allowed_output_prefixes=normalized_allowed_output_prefixes,
                                enforce_deliverable_paths=enforce_deliverable_paths,
                                edit_existing_only=edit_existing_only,
                                already_touched_deliverables=(
                                    touched_canonical_deliverables
                                    if one_shot_direct_deliverable_mode
                                    else None
                                ),
                            )
                            if not sanitized_error:
                                resolved_tool_args = sanitized_args
                                attempted_paths = runner._target_paths_for_policy(
                                    tool_name=resolved_tool_name,
                                    tool_args=resolved_tool_args,
                                    workspace=workspace,
                                    is_mutating_tool=is_mutating_tool,
                                    mutation_target_arg_keys=mutation_target_arg_keys,
                                )
                                route_metadata.update({
                                    "optional_output_side_effects_suppressed": True,
                                    "suppressed_argument_keys": suppressed_keys,
                                    "suppression_reason": "canonical_output_policy",
                                })
                                policy_error = None
                    if not policy_error:
                        policy_error = runner._validate_sealed_artifact_mutation_policy(
                            task=task,
                            tool_name=resolved_tool_name,
                            tool_args=resolved_tool_args,
                            workspace=workspace,
                            is_mutating_tool=is_mutating_tool,
                            mutation_target_arg_keys=mutation_target_arg_keys,
                            prior_successful_tool_calls=historical_successful_tool_calls,
                            current_tool_calls=tool_calls_record,
                        )
                    if policy_error:
                        if runner._is_forbidden_output_path_error(policy_error):
                            runner._emit_telemetry_event(
                                event_type=FORBIDDEN_CANONICAL_WRITE_BLOCKED,
                                task_id=task.id,
                                data={
                                    "subtask_id": subtask.id,
                                    "tool": resolved_tool_name,
                                    "attempted_paths": attempted_paths,
                                    "expected_deliverables": list(canonical_deliverables),
                                    "forbidden_deliverables": list(
                                        canonical_forbidden_deliverables,
                                    ),
                                    "allowed_output_prefixes": list(
                                        normalized_allowed_output_prefixes,
                                    ),
                                    "policy_error": policy_error,
                                },
                            )
                        elif runner._is_sealed_artifact_mutation_policy_error(policy_error):
                            runner._emit_sealed_policy_preflight_blocked(
                                task_id=task.id,
                                subtask_id=subtask.id,
                                tool_name=resolved_tool_name,
                                attempted_paths=attempted_paths,
                                policy_error=policy_error,
                            )
                        tool_result = ToolResult.fail(policy_error)
                    else:
                        deduped = False
                        idempotency_key = ""
                        args_hash = ""
                        guard_mode = runner._sealed_artifact_post_call_guard_mode()
                        pre_call_seal_hashes: dict[str, str] = {}
                        if runner._enable_mutation_idempotency and is_mutating_tool:
                            idempotency_key, args_hash = runner._mutation_idempotency_key(
                                task=task,
                                subtask=subtask,
                                tool_name=resolved_tool_name,
                                arguments=resolved_tool_args,
                            )
                            try:
                                ledger_entry = await runner._memory.get_mutation_ledger_entry(
                                    idempotency_key,
                                )
                            except Exception:
                                ledger_entry = None
                            if (
                                isinstance(ledger_entry, dict)
                                and str(ledger_entry.get("status", "")).strip().lower()
                                == "success"
                            ):
                                tool_result = ToolResult.from_json(
                                    str(ledger_entry.get("result_json", "") or ""),
                                )
                                deduped = True
                                runner._emit_telemetry_event(
                                        event_type=TOOL_CALL_DEDUPLICATED,
                                        task_id=task.id,
                                        data={
                                            "subtask_id": subtask.id,
                                            "tool": resolved_tool_name,
                                            "tool_call_id": tool_call_id,
                                            "idempotency_key": idempotency_key,
                                            "run_id": runner._normalize_run_id(task),
                                    },
                                )
                        if (
                            is_mutating_tool
                            and not deduped
                            and guard_mode != "off"
                        ):
                            pre_call_seal_hashes = runner._snapshot_tracked_artifact_hashes(
                                task=task,
                                workspace=workspace,
                            )
                        execute_args = dict(resolved_tool_args)
                        if (
                            resolved_tool_name == "ask_user"
                            and not runner._tools.has(
                                "ask_user",
                                execution_surface=execution_surface,
                            )
                        ):
                            tool_result = runner._ask_user_limit_error(
                                "ask_user is unavailable for this execution surface.",
                            )
                        elif (
                            resolved_tool_name == "ask_user"
                            and runner._ask_user_runtime_enabled()
                        ):
                            now = time.monotonic()
                            if (
                                session.ask_user_questions_asked
                                >= runner._ask_user_max_questions_per_subtask
                            ):
                                tool_result = runner._ask_user_limit_error(
                                    "ask_user question cap reached for this subtask.",
                                )
                            elif (
                                session.last_ask_user_requested_at > 0
                                and runner._ask_user_min_seconds_between_questions > 0
                                and (
                                    now - session.last_ask_user_requested_at
                                ) < runner._ask_user_min_seconds_between_questions
                            ):
                                wait_seconds = (
                                    runner._ask_user_min_seconds_between_questions
                                    - (now - session.last_ask_user_requested_at)
                                )
                                tool_result = runner._ask_user_limit_error(
                                    "ask_user called too quickly "
                                    f"({max(0.0, wait_seconds):.1f}s minimum wait remaining).",
                                )
                            else:
                                question_manager = runner._question_manager
                                if question_manager is None:
                                    tool_result = ToolResult.fail(
                                        "ask_user runtime manager is unavailable.",
                                    )
                                else:
                                    request = QuestionRequest.from_ask_user_args(
                                        resolved_tool_args,
                                        timeout_policy=runner._ask_user_policy,
                                        timeout_seconds=runner._ask_user_timeout_seconds,
                                        timeout_default_response=(
                                            runner._ask_user_timeout_default_response or None
                                        ),
                                        tool_call_id=str(getattr(tc, "id", "") or ""),
                                        retry_attempt=max(
                                            0,
                                            int(getattr(subtask, "retry_count", 0) or 0),
                                        ),
                                    )
                                    if not request.question_id and request.tool_call_id:
                                        request.question_id = (
                                            question_manager.deterministic_question_id(
                                                task_id=task.id,
                                                subtask_id=subtask.id,
                                                tool_call_id=request.tool_call_id,
                                                retry_attempt=request.retry_attempt,
                                            )
                                        )
                                    if not request.question_id:
                                        request.question_id = (
                                            question_manager.deterministic_question_id_for_request(
                                                task_id=task.id,
                                                subtask_id=subtask.id,
                                                request=request,
                                            )
                                        )
                                    pending = await question_manager.list_pending_questions(
                                        task.id,
                                    )
                                    pending_ids = {
                                        str(row.get("question_id", "")).strip()
                                        for row in pending
                                        if isinstance(row, dict)
                                    }
                                    has_same_pending_question = bool(
                                        request.question_id
                                        and request.question_id in pending_ids
                                    )
                                    if (
                                        runner._ask_user_max_pending_per_task > 0
                                        and len(pending) >= runner._ask_user_max_pending_per_task
                                        and not has_same_pending_question
                                    ):
                                        tool_result = runner._ask_user_limit_error(
                                            "ask_user pending question limit reached for task.",
                                        )
                                    else:
                                        session.ask_user_questions_asked += 1
                                        session.last_ask_user_requested_at = now
                                        await runner._set_waiting_for_user_input(
                                            task=task,
                                            subtask=subtask,
                                            request=request,
                                        )
                                        def _check_task_control() -> str:
                                            return runner._task_status_text(task)

                                        try:
                                            answer = await question_manager.request_question(
                                                task_id=task.id,
                                                subtask_id=subtask.id,
                                                request=request,
                                                check_task_control=_check_task_control,
                                            )
                                        finally:
                                            await runner._clear_waiting_for_user_input(
                                                task=task,
                                                question_id=request.question_id,
                                            )
                                        answer_payload = answer.to_payload()
                                        answer_status = str(
                                            getattr(answer.status, "value", answer.status),
                                        ).strip().lower()
                                        if (
                                            answer_status == "timeout"
                                            and runner._ask_user_policy == "fail_closed"
                                        ):
                                            tool_result = ToolResult(
                                                success=False,
                                                output="",
                                                error=(
                                                    "ask_user timed out without a valid "
                                                    "default response."
                                                ),
                                                data=answer_payload,
                                            )
                                        elif answer_status == "cancelled":
                                            tool_result = ToolResult(
                                                success=False,
                                                output="",
                                                error="ask_user request cancelled.",
                                                data=answer_payload,
                                            )
                                        else:
                                            answer_text = answer.text_response.strip()
                                            if not answer_text:
                                                response_type = str(
                                                    answer_payload.get("response_type", ""),
                                                ).strip()
                                                answer_text = (
                                                    response_type or "Clarification received."
                                                )
                                            tool_result = ToolResult.ok(
                                                answer_text,
                                                data=answer_payload,
                                            )
                                        await runner._persist_ask_user_answer_memory(
                                            task=task,
                                            subtask=subtask,
                                            request=request,
                                            answer=answer,
                                        )
                        elif not deduped:
                            web_target_key = ""
                            if resolved_tool_name in {"web_fetch", "web_fetch_html"}:
                                web_target_key = _web_target_key(resolved_tool_args)
                                prior_web_error = session.exhausted_web_targets.get(
                                    web_target_key,
                                    "",
                                )
                                if web_target_key and prior_web_error:
                                    tool_result = _exhausted_web_target_result(
                                        url=str(resolved_tool_args.get("url", "") or ""),
                                        prior_error=prior_web_error,
                                    )
                                    deduped = True
                                execute_args["_enable_filetype_ingest_router"] = bool(
                                    runner._enable_filetype_ingest_router,
                                )
                                execute_args["_artifact_retention_max_age_days"] = int(
                                    runner._ingest_artifact_retention_max_age_days,
                                )
                                execute_args["_artifact_retention_max_files_per_scope"] = int(
                                    runner._ingest_artifact_retention_max_files_per_scope,
                                )
                                execute_args["_artifact_retention_max_bytes_per_scope"] = int(
                                    runner._ingest_artifact_retention_max_bytes_per_scope,
                                )
                            if not deduped:
                                tool_result = await runner._tools.execute(
                                    resolved_tool_name, execute_args,
                                    workspace=workspace,
                                    read_roots=read_roots,
                                    read_path_map=read_path_map,
                                    scratch_dir=runner._config.scratch_path,
                                    changelog=changelog,
                                    subtask_id=subtask.id,
                                    auth_context=auth_context,
                                    execution_surface=execution_surface,
                                )
                                if (
                                    web_target_key
                                    and not tool_result.success
                                    and _is_terminal_web_source_failure(tool_result.error)
                                ):
                                    session.exhausted_web_targets[web_target_key] = str(
                                        tool_result.error or "access denied"
                                    )
                        if route_metadata:
                            route_data = (
                                dict(tool_result.data)
                                if isinstance(tool_result.data, dict)
                                else {}
                            )
                            route_data.update(route_metadata)
                            tool_result.data = route_data
                        if (
                            is_mutating_tool
                            and not deduped
                            and tool_result.success
                        ):
                            unexpected_paths: list[str] = []
                            if guard_mode != "off":
                                unexpected_paths = runner._unexpected_sealed_mutation_paths(
                                    task=task,
                                    workspace=workspace,
                                    tool_name=resolved_tool_name,
                                    tool_args=resolved_tool_args,
                                    tool_result=tool_result,
                                    is_mutating_tool=is_mutating_tool,
                                    mutation_target_arg_keys=mutation_target_arg_keys,
                                    pre_call_hashes=pre_call_seal_hashes,
                                )
                            if unexpected_paths:
                                runner._emit_sealed_unexpected_mutation_detected(
                                    task_id=task.id,
                                    subtask_id=subtask.id,
                                    tool_name=resolved_tool_name,
                                    tool_call_id=tool_call_id,
                                    mode=guard_mode,
                                    unexpected_paths=unexpected_paths,
                                )
                                merged_files = list(tool_result.files_changed)
                                seen_files = set(merged_files)
                                for relpath in unexpected_paths:
                                    if relpath in seen_files:
                                        continue
                                    seen_files.add(relpath)
                                    merged_files.append(relpath)
                                if merged_files != list(tool_result.files_changed):
                                    tool_result = ToolResult(
                                        success=tool_result.success,
                                        output=tool_result.output,
                                        content_blocks=tool_result.content_blocks,
                                        data=tool_result.data,
                                        files_changed=merged_files,
                                        error=tool_result.error,
                                    )
                            resealed_count = runner._reseal_tracked_artifacts_after_mutation(
                                task=task,
                                workspace=workspace,
                                tool_name=tc.name,
                                tool_args=tc.arguments,
                                tool_result=tool_result,
                                is_mutating_tool=is_mutating_tool,
                                mutation_target_arg_keys=mutation_target_arg_keys,
                                subtask_id=subtask.id,
                                tool_call_id=tool_call_id,
                            )
                            if resealed_count > 0:
                                runner._emit_sealed_reseal_applied(
                                    task_id=task.id,
                                    subtask_id=subtask.id,
                                    tool_name=tc.name,
                                    tool_call_id=tool_call_id,
                                    path_count=resealed_count,
                                )
                            if unexpected_paths and guard_mode == "enforce":
                                guard_data = (
                                    dict(tool_result.data)
                                    if isinstance(tool_result.data, dict)
                                    else {}
                                )
                                guard_data.update({
                                    "sealed_unexpected_mutation_detected": True,
                                    "unexpected_paths": list(unexpected_paths),
                                    "guard_mode": guard_mode,
                                    "event_type": SEALED_UNEXPECTED_MUTATION_DETECTED,
                                })
                                tool_result = ToolResult(
                                    success=False,
                                    output="",
                                    error=(
                                        "Post-call sealed artifact guard blocked this mutation: "
                                        "tool changed sealed path(s) outside declared/returned "
                                        f"targets: {', '.join(unexpected_paths)}."
                                    ),
                                    data=guard_data,
                                    files_changed=list(tool_result.files_changed),
                                )
                        if (
                            runner._enable_mutation_idempotency
                            and is_mutating_tool
                            and idempotency_key
                            and not deduped
                        ):
                            try:
                                await runner._memory.upsert_mutation_ledger_entry(
                                    idempotency_key=idempotency_key,
                                    task_id=task.id,
                                    run_id=runner._normalize_run_id(task),
                                    subtask_id=subtask.id,
                                    tool_name=resolved_tool_name,
                                    args_hash=args_hash,
                                    status="success" if tool_result.success else "failure",
                                    result_json=tool_result.to_json(),
                                )
                            except Exception:
                                logger.debug(
                                    "Failed persisting idempotency ledger entry %s",
                                    idempotency_key,
                                    exc_info=True,
                                )
                    record = ToolCallRecord(
                        tool=resolved_tool_name,
                        args=resolved_tool_args,
                        result=tool_result,
                        call_id=str(getattr(tc, "id", "") or ""),
                    )
                    tool_calls_record.append(record)
                    new_evidence = extract_evidence_records(
                        task_id=task.id,
                        subtask_id=subtask.id,
                        tool_calls=[record],
                        existing_ids=known_evidence_ids,
                        context_text_max_chars=runner._evidence_context_text_max_chars,
                    )
                    if new_evidence:
                        evidence_records_current.extend(new_evidence)
                        for item in new_evidence:
                            evidence_id = str(item.get("evidence_id", "")).strip()
                            if evidence_id:
                                known_evidence_ids.add(evidence_id)
                        data = tool_result.data if isinstance(tool_result.data, dict) else {}
                        data = dict(data)
                        data["evidence_ids"] = [
                            str(item.get("evidence_id", "")).strip()
                            for item in new_evidence
                            if str(item.get("evidence_id", "")).strip()
                        ]
                        if not tool_result.data:
                            tool_result.data = data
                        else:
                            tool_result.data = data
                    if (
                        one_shot_direct_deliverable_mode
                        and is_mutating_tool
                        and tool_result.success
                    ):
                        touched_paths = [
                            path
                            for path in attempted_paths
                            if path in canonical_deliverable_set
                        ]
                        if touched_paths:
                            touched_canonical_deliverables.update(touched_paths)
                            if canonical_deliverable_set.issubset(
                                touched_canonical_deliverables,
                            ):
                                completion_only_after_deliverables = True
                    runner._emit_tool_event(
                        TOOL_CALL_COMPLETED, task.id, subtask.id,
                        resolved_tool_name, resolved_tool_args,
                        result=tool_result,
                        workspace=workspace,
                    )
                    runner._emit_artifact_ingest_telemetry(
                        task_id=task.id,
                        subtask_id=subtask.id,
                        tool_name=resolved_tool_name,
                        tool_args=resolved_tool_args,
                        result=tool_result,
                    )
                    runner._emit_artifact_read_telemetry(
                        task_id=task.id,
                        subtask_id=subtask.id,
                        tool_name=resolved_tool_name,
                        tool_args=resolved_tool_args,
                        result=tool_result,
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": await runner._serialize_tool_result_for_model(
                            resolved_tool_name, tool_result,
                        ),
                    })
                if session.interruption_reason:
                    break

                # Anti-amnesia reminder
                if not completion_only_after_deliverables:
                    messages.append({
                        # Some OpenAI-compatible providers reject repeated in-thread
                        # system messages during tool-call loops.
                        "role": "user",
                        "content": runner._build_todo_reminder(task, subtask),
                    })
            else:
                # Text-only response. Depending on configured mode, require
                # explicit completion contract payload before termination.
                mode = str(
                    getattr(
                        runner,
                        "_executor_completion_contract_mode",
                        runner.EXECUTOR_COMPLETION_CONTRACT_MODE,
                    ),
                ).strip().lower()
                if mode in {"warn", "enforce"}:
                    valid_contract, contract_error = runner._validate_completion_contract(
                        session.response.text or "",
                    )
                    if not valid_contract and mode == "enforce":
                        messages.append({
                            "role": "assistant",
                            "content": session.response.text or "",
                        })
                        messages.append({
                            "role": "system",
                            "content": (
                                "COMPLETION CONTRACT ERROR: "
                                f"{contract_error}\n"
                                "Respond with a JSON object containing keys: "
                                "status, deliverables_touched, verification_notes."
                            ),
                        })
                        continue
                    if not valid_contract and mode == "warn":
                        runner._emit_model_event(
                            task_id=task.id,
                            subtask_id=subtask.id,
                            model_name=model.name,
                            phase="done",
                            details={
                                "operation": "completion_contract_warn",
                                "warning": contract_error,
                            },
                        )
                session.completed_normally = True
                break
        else:
            if session.response is not None and session.response.has_tool_calls():
                session.budget_exhaustion_note = (
                    "Execution reached tool-iteration budget "
                    f"({iteration_budget} turns) while additional "
                    "tool calls were still required."
                )

        if session.interruption_reason is None and not session.completed_normally:
            if session.response is None:
                session.interruption_reason = (
                    "Execution ended before receiving a model response."
                )

        elapsed = time.monotonic() - start_time
        model_output = (
            session.response.text
            if session.response and session.response.text
            else ""
        )
        model_output_clean = runner._strip_tool_call_placeholders(model_output)
        if session.interruption_reason:
            if model_output_clean:
                model_output = (
                    f"{session.interruption_reason} Last model response: {model_output_clean}"
                )
            else:
                model_output = session.interruption_reason
        else:
            model_output = model_output_clean
            if session.budget_exhaustion_note:
                if model_output_clean:
                    model_output = (
                        f"{session.budget_exhaustion_note} Last model response: "
                        f"{model_output_clean}"
                    )
                else:
                    model_output = session.budget_exhaustion_note
            contract_mismatch = runner._completion_contract_mutation_mismatch(
                response_text=(
                    session.response.text
                    if session.response and session.response.text
                    else ""
                ),
                tool_calls=tool_calls_record,
                workspace=workspace,
            )
            if contract_mismatch:
                model_output = (
                    f"{model_output}\n\n{contract_mismatch}".strip()
                    if model_output
                    else contract_mismatch
                )
        summary = await runner._summarize_model_output(
            model_output,
            max_chars=runner._max_state_summary_chars,
            label="subtask state summary",
        )
        verification_summary = await runner._summarize_model_output(
            model_output,
            max_chars=runner._max_verification_summary_chars,
            label="subtask verification summary",
        )

        result = SubtaskResult(
            status=(
                SubtaskResultStatus.FAILED
                if session.interruption_reason
                else SubtaskResultStatus.SUCCESS
            ),
            summary=summary,
            tool_calls=tool_calls_record,
            duration_seconds=elapsed,
            tokens_used=session.total_tokens,
            model_used=model.name,
            evidence_records=evidence_records_current,
            telemetry_counters=dict(telemetry_counters),
        )

        if session.interruption_reason:
            reason_code = (
                last_model_failure_reason_code
                or "model_invocation_failed"
            )
            metadata = dict(last_model_failure_metadata)
            if reason_code in {
                ModelEmptyResponseError.reason_code,
                "model_stream_empty",
            }:
                metadata.setdefault("root_cause", "model_empty_response")
                metadata.setdefault("downstream_failure", "deliverable_missing")
            elif reason_code == INFRA_MESSAGE_CONTRACT_VIOLATION:
                metadata.setdefault("root_cause", "message_contract_violation")
                metadata.setdefault("downstream_failure", "model_request_not_sent")
            verification = VerificationResult(
                tier=1,
                passed=False,
                confidence=0.0,
                checks=[Check(
                    name="execution_completed",
                    passed=False,
                    detail=session.interruption_reason,
                )],
                feedback=session.interruption_reason,
                outcome="fail",
                reason_code=reason_code,
                severity_class=(
                    "infra"
                    if reason_code in _INFRA_MODEL_REASON_CODES
                    else ""
                ),
                metadata=metadata,
            )
            runner._spawn_memory_extraction(task.id, subtask.id, result)
            return result, verification

        if session.budget_exhaustion_note:
            result.status = SubtaskResultStatus.FAILED
            verification = VerificationResult(
                tier=1,
                passed=False,
                confidence=0.0,
                checks=[Check(
                    name="runner_budget_checkpoint",
                    passed=False,
                    detail=session.budget_exhaustion_note,
                )],
                feedback=(
                    f"{session.budget_exhaustion_note} Preserve existing artifacts and "
                    "continue only the exact unfinished work from this checkpoint."
                ),
                outcome="fail",
                reason_code="runner_tool_budget_exhausted",
                severity_class="infra",
                metadata={
                    "checkpoint_required": True,
                    "iteration_budget": iteration_budget,
                    "tool_call_count": len(tool_calls_record),
                },
            )
            runner._spawn_memory_extraction(task.id, subtask.id, result)
            return result, verification

        # 4. Verification
        evidence_tool_calls = list(prior_successful_tool_calls or [])
        combined_evidence_records = merge_evidence_records(
            prior_evidence_records or [],
            evidence_records_current,
        )
        verification = await runner._verification.verify(
            subtask=subtask,
            result_summary=verification_summary,
            tool_calls=tool_calls_record,
            evidence_tool_calls=evidence_tool_calls,
            evidence_records=combined_evidence_records,
            retry_writable_deliverables=list(canonical_deliverables),
            validity_contract=(
                dict(subtask.validity_contract_snapshot)
                if isinstance(subtask.validity_contract_snapshot, dict)
                else {}
            ),
            workspace=workspace,
            tier=subtask.verification_tier,
            task_id=task.id,
        )

        if not verification.passed:
            result.status = SubtaskResultStatus.FAILED

        # 5. Memory extraction — fire-and-forget
        runner._spawn_memory_extraction(task.id, subtask.id, result)

        return result, verification
    finally:
        runner._subtask_deadline_monotonic = None
        runner._active_subtask_telemetry_counters = None
        compactor_event_context.reset(compactor_context_token)
