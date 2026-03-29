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
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
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

# Backwards-compatible name used by runner internals.
_COMPACTOR_EVENT_CONTEXT = compactor_event_context

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
        iteration_budget = runner._tool_iteration_budget(
            subtask=subtask,
            retry_strategy=retry_strategy,
            has_expected_deliverables=bool(canonical_deliverables),
            base_budget=runner._max_tool_iterations,
        )

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
            session.messages = await runner._compact_messages_for_model(
                session.messages,
                remaining_seconds=remaining_seconds,
            )
            messages = session.messages
            runner._emit_compaction_policy_decision_from_diagnostics(
                task_id=task.id,
                subtask_id=subtask.id,
            )
            tool_schemas = runner._tools.all_schemas(
                auth_context=auth_context,
                execution_surface=execution_surface,
            )
            operation = "stream" if streaming else "complete"
            session.response = None
            policy = ModelRetryPolicy.from_execution_config(runner._config.execution)
            invocation_attempt = 0
            request_diag = None
            overflow_fallback_pending = False
            overflow_fallback_attempted = False
            overflow_fallback_report: dict[str, Any] | None = None

            async def _invoke_model():
                nonlocal invocation_attempt, request_diag
                invocation_attempt += 1
                runner._increment_subtask_counter("model_invocations")
                request_diag = collect_request_diagnostics(
                    messages=session.messages,
                    tools=tool_schemas,
                    origin=f"runner.execute_subtask.{operation}",
                )
                runner._emit_model_event(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="start",
                    details={
                        **request_diag.to_event_payload(),
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
                    return await runner._stream_completion(
                        model,
                        session.messages,
                        tool_schemas,
                        task_id=task.id,
                        subtask_id=subtask.id,
                    )
                return await model.complete(session.messages, tools=tool_schemas)

            def _should_retry_invocation(error: BaseException) -> bool:
                nonlocal overflow_fallback_pending
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
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
                nonlocal overflow_fallback_pending
                nonlocal overflow_fallback_attempted, overflow_fallback_report
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
                        **(overflow_fallback_report or {}),
                    },
                )

            try:
                session.response = await call_with_model_retry(
                    _invoke_model,
                    policy=policy,
                    should_retry=_should_retry_invocation,
                    on_failure=_on_invocation_failure,
                )
            except Exception as e:
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
                    runner._increment_subtask_counter("tool_calls")
                    if is_mutating_tool:
                        runner._increment_subtask_counter("mutating_tool_calls")
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
                    )
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
                        blocked_paths = runner._target_paths_for_policy(
                            tool_name=tc.name,
                            tool_args=tc.arguments,
                            workspace=workspace,
                            is_mutating_tool=is_mutating_tool,
                            mutation_target_arg_keys=mutation_target_arg_keys,
                        )
                        if runner._is_forbidden_output_path_error(policy_error):
                            runner._emit_telemetry_event(
                                event_type=FORBIDDEN_CANONICAL_WRITE_BLOCKED,
                                task_id=task.id,
                                data={
                                    "subtask_id": subtask.id,
                                    "tool": resolved_tool_name,
                                    "attempted_paths": blocked_paths,
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
                                attempted_paths=blocked_paths,
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
                                        runner._set_waiting_for_user_input(
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
                                            runner._clear_waiting_for_user_input(
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
                            if resolved_tool_name in {"web_fetch", "web_fetch_html"}:
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
                            tool_result = await runner._tools.execute(
                                resolved_tool_name, execute_args,
                                workspace=workspace,
                                read_roots=read_roots,
                                scratch_dir=runner._config.scratch_path,
                                changelog=changelog,
                                subtask_id=subtask.id,
                                auth_context=auth_context,
                                execution_surface=execution_surface,
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
