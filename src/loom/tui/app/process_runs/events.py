"""Event formatting helpers for process runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import ProcessRunState


_WORKSPACE_REFRESH_EVENT_TYPES = {
    "subtask_completed",
    "subtask_failed",
    "task_completed",
    "task_failed",
    "task_cancelled",
}

_FORCE_REFRESH_EVENT_TYPES = {
    "task_planning",
    "task_plan_ready",
    "task_executing",
    "task_replanning",
    "task_completed",
    "task_failed",
    "task_cancelled",
    "task_cancel_requested",
    "task_cancel_ack",
    "task_cancel_timeout",
    "task_paused",
    "task_resumed",
    "task_injected",
    "verification_started",
    "verification_passed",
    "verification_failed",
    "verification_outcome",
    "verification_contradiction_detected",
    "rule_failure_by_type",
    "deterministic_block_rate",
    "verification_inconclusive_rate",
    "claim_verification_summary",
    "remediation_queued",
    "remediation_started",
    "remediation_attempt",
    "remediation_resolved",
    "remediation_failed",
    "remediation_expired",
    "remediation_terminal",
    "subtask_blocked",
    "run_validity_scorecard",
    "telemetry_run_summary",
    "subtask_started",
    "subtask_retrying",
    "subtask_completed",
    "subtask_failed",
    "tool_call_completed",
    "ask_user_requested",
    "ask_user_answered",
    "ask_user_timeout",
    "ask_user_cancelled",
}


def subtask_content(
    data: dict[str, Any],
    subtask_id: str,
    *,
    run: Any | None,
    plain_text: Callable[[object | None], str],
) -> str:
    """Lookup subtask label, preferring stable run-normalized labels."""
    if not subtask_id:
        return ""

    if run is not None:
        labels = getattr(run, "task_labels", {})
        if isinstance(labels, dict):
            label = str(labels.get(subtask_id, "")).strip()
            if label and label != subtask_id:
                return label

        run_tasks = getattr(run, "tasks", [])
        if isinstance(run_tasks, list):
            for row in run_tasks:
                if not isinstance(row, dict):
                    continue
                if str(row.get("id", "")) != subtask_id:
                    continue
                content = plain_text(row.get("content", "")).strip()
                if content and content != subtask_id:
                    return content
                break

    tasks = data.get("tasks")
    if isinstance(tasks, list):
        for row in tasks:
            if not isinstance(row, dict):
                continue
            if str(row.get("id", "")) != subtask_id:
                continue
            content = plain_text(row.get("content", "")).strip()
            if content and content != subtask_id:
                return content
            break
    return ""


def format_process_progress_event(
    self,
    data: dict,
    *,
    run: ProcessRunState | None = None,
    context: str = "process_run",
) -> str | None:
    """Format orchestrator progress events into concise chat messages."""
    event_type = str(data.get("event_type") or "")
    event_data = data.get("event_data")
    if not event_type:
        return None
    if not isinstance(event_data, dict):
        event_data = {}
    mode = str(context or "process_run").strip().lower()

    subtask_id = str(event_data.get("subtask_id", "")).strip()
    subtask_content = self._subtask_content(data, subtask_id, run)
    subtask_label = subtask_id or "subtask"
    if subtask_content:
        subtask_label = f"{subtask_label} - {self._one_line(subtask_content, 90)}"

    if event_type == "task_planning":
        if mode == "cowork_delegate":
            return "Planning delegated task..."
        return "Planning process run..."
    if event_type == "task_plan_ready":
        count = len(data.get("tasks", [])) if isinstance(data.get("tasks"), list) else 0
        if mode == "cowork_delegate":
            return f"Delegation plan ready: {count} subtasks."
        return f"Plan ready: {count} subtasks."
    if event_type == "task_executing":
        if mode == "cowork_delegate":
            return "Executing delegated subtasks..."
        return "Executing subtasks..."
    if event_type == "model_invocation":
        phase = str(event_data.get("phase", "")).strip()
        model_name = str(event_data.get("model", "")).strip()
        label = subtask_label

        def _int_value(value: object) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        request_bytes = _int_value(event_data.get("request_bytes"))
        request_tokens = _int_value(event_data.get("request_est_tokens"))
        message_count = _int_value(event_data.get("message_count"))
        assistant_tool_calls = _int_value(event_data.get("assistant_tool_calls"))
        origin = str(event_data.get("origin", "")).strip()

        details: list[str] = []
        if request_tokens > 0:
            details.append(f"{request_tokens:,} est tokens")
        if request_bytes > 0:
            if request_bytes >= 1024 * 1024:
                details.append(f"{request_bytes / (1024 * 1024):.2f} MB")
            else:
                details.append(f"{request_bytes / 1024:.0f} KB")
        if message_count > 0:
            details.append(f"{message_count} msgs")
        if assistant_tool_calls > 0:
            details.append(f"{assistant_tool_calls} tool calls in ctx")
        if origin and not origin.startswith("runner.execute_subtask."):
            details.append(origin)

        if phase == "start":
            if request_bytes >= 3_500_000:
                size_text = (
                    f"{request_bytes / (1024 * 1024):.2f} MB"
                    if request_bytes > 0
                    else "oversize"
                )
                token_text = (
                    f"{request_tokens:,} est tokens"
                    if request_tokens > 0
                    else "estimated tokens unavailable"
                )
                return (
                    f"Request-size risk for {label}: {size_text}, {token_text}. "
                    "Compaction/plumbing needed."
                )
            if model_name:
                if details:
                    return (
                        f"Thinking on {label} with {model_name} "
                        f"({', '.join(details)})..."
                    )
                return f"Thinking on {label} with {model_name}..."
            if details:
                return f"Thinking on {label} ({', '.join(details)})..."
            return f"Thinking on {label}..."
        if phase == "done":
            return None
        return None
    if event_type == "token_streamed":
        count = event_data.get("token_count")
        try:
            token_count = int(count)
        except (TypeError, ValueError):
            token_count = 0
        if token_count > 0:
            return f"Working on {subtask_label}... ({token_count} streamed chunks)"
        return None
    if event_type == "tool_call_started":
        tool = str(event_data.get("tool", "")).strip() or "tool"
        return f"Using {tool} for {subtask_label}."
    if event_type == "tool_call_completed":
        tool = str(event_data.get("tool", "")).strip() or "tool"
        success = event_data.get("success")
        if success is True:
            return f"Finished {tool} for {subtask_label}."
        error = self._one_line(event_data.get("error", ""), 120)
        if error:
            return f"{tool} failed for {subtask_label}: {error}"
        return f"{tool} failed for {subtask_label}."
    if event_type == "ask_user_requested":
        question = self._one_line(event_data.get("question", ""), 140)
        if question:
            return f"Clarification requested for {subtask_label}: {question}"
        return f"Clarification requested for {subtask_label}."
    if event_type in {"ask_user_answered", "ask_user_timeout", "ask_user_cancelled"}:
        answer = event_data.get("answer", {})
        if not isinstance(answer, dict):
            answer = {}
        answer_text = self._one_line(answer.get("custom_response", ""), 120)
        if not answer_text:
            selected_labels = answer.get("selected_labels", [])
            if isinstance(selected_labels, list):
                answer_text = self._one_line(
                    ", ".join(
                        str(item).strip()
                        for item in selected_labels
                        if str(item).strip()
                    ),
                    120,
                )
        if not answer_text:
            answer_text = self._one_line(answer.get("response_type", ""), 60)
        if event_type == "ask_user_answered":
            if answer_text:
                return f"Clarification answered for {subtask_label}: {answer_text}"
            return f"Clarification answered for {subtask_label}."
        if event_type == "ask_user_timeout":
            if answer_text:
                return f"Clarification timed out for {subtask_label}: {answer_text}"
            return f"Clarification timed out for {subtask_label}."
        if answer_text:
            return f"Clarification cancelled for {subtask_label}: {answer_text}"
        return f"Clarification cancelled for {subtask_label}."
    if event_type == "subtask_started":
        return f"Started {subtask_label}."
    if event_type == "subtask_retrying":
        attempt = event_data.get("attempt")
        tier = event_data.get("escalated_tier")
        reason = self._one_line(event_data.get("feedback", ""), 120)
        msg = f"Retrying {subtask_label}"
        if attempt:
            msg += f" (attempt {attempt})"
        if tier:
            msg += f", tier {tier}"
        if reason:
            msg += f": {reason}"
        return f"{msg}."
    if event_type == "subtask_completed":
        return f"Completed {subtask_label}."
    if event_type == "subtask_failed":
        reason = self._one_line(
            event_data.get("feedback")
            or event_data.get("reason")
            or event_data.get("error")
            or "",
            140,
        )
        if reason:
            return f"Failed {subtask_label}: {reason}"
        return f"Failed {subtask_label}."
    if event_type == "task_replanning":
        reason = self._one_line(event_data.get("reason", ""), 140)
        if reason:
            return f"Replanning task: {reason}"
        return "Replanning task..."
    if event_type == "task_plan_normalized":
        normalized = event_data.get("normalized_subtasks", [])
        if not isinstance(normalized, list):
            normalized = []
        changed_ids: list[str] = []
        for item in normalized:
            if not isinstance(item, dict):
                continue
            subtask_id = str(item.get("subtask_id", "")).strip()
            if subtask_id:
                changed_ids.append(subtask_id)
        context = str(event_data.get("context", "")).strip()
        if changed_ids:
            joined = ", ".join(changed_ids[:3])
            if len(changed_ids) > 3:
                joined += ", ..."
            if context:
                return (
                    f"Normalized plan ({context}): demoted non-terminal synthesis "
                    f"for {joined}."
                )
            return (
                "Normalized plan: demoted non-terminal synthesis for "
                f"{joined}."
            )
        return "Normalized plan topology."
    if event_type == "task_stalled":
        blocked = event_data.get("blocked_subtasks", [])
        if not isinstance(blocked, list):
            blocked = []
        attempt = event_data.get("attempt")
        attempt_text = ""
        try:
            attempt_num = int(attempt)
            if attempt_num > 0:
                attempt_text = f" (attempt {attempt_num})"
        except (TypeError, ValueError):
            attempt_text = ""
        if blocked and isinstance(blocked[0], dict):
            first = blocked[0]
            first_id = str(first.get("subtask_id", "")).strip() or "subtask"
            reasons = first.get("reasons", [])
            if isinstance(reasons, list):
                reason_text = self._one_line(
                    ", ".join(
                        str(reason).strip()
                        for reason in reasons
                        if str(reason).strip()
                    ),
                    120,
                )
            else:
                reason_text = self._one_line(reasons, 120)
            if reason_text:
                return (
                    f"Execution stalled{attempt_text}: "
                    f"{first_id} blocked ({reason_text})."
                )
            return f"Execution stalled{attempt_text}: {first_id} blocked."
        return f"Execution stalled{attempt_text}: no runnable subtasks."
    if event_type == "task_stalled_recovery_attempted":
        mode = str(event_data.get("recovery_mode", "")).strip().lower()
        success = event_data.get("recovery_success")
        attempt = event_data.get("attempt")
        attempt_suffix = ""
        try:
            attempt_num = int(attempt)
            if attempt_num > 0:
                attempt_suffix = f" (attempt {attempt_num})"
        except (TypeError, ValueError):
            attempt_suffix = ""
        mode_label = mode or "recovery"
        if success is True:
            return (
                f"Stall recovery via {mode_label} succeeded"
                f"{attempt_suffix}."
            )
        reason = self._one_line(event_data.get("reason", ""), 120)
        if reason:
            return (
                f"Stall recovery via {mode_label} failed"
                f"{attempt_suffix}: {reason}"
            )
        return f"Stall recovery via {mode_label} failed{attempt_suffix}."
    if event_type == "task_paused":
        if bool(event_data.get("requested", False)):
            return "Process run paused."
        reason = self._one_line(event_data.get("error", ""), 140)
        if reason:
            return f"Pause rejected: {reason}"
        return "Pause request rejected."
    if event_type == "task_resumed":
        if bool(event_data.get("requested", False)):
            return "Process run resumed."
        reason = self._one_line(event_data.get("error", ""), 140)
        if reason:
            return f"Resume rejected: {reason}"
        return "Resume request rejected."
    if event_type == "task_injected":
        if bool(event_data.get("requested", False)):
            chars = event_data.get("chars")
            try:
                inject_chars = int(chars)
            except (TypeError, ValueError):
                inject_chars = 0
            if inject_chars > 0:
                return f"Injected user instruction ({inject_chars} chars)."
            return "Injected user instruction."
        reason = self._one_line(event_data.get("error", ""), 140)
        if reason:
            return f"Inject rejected: {reason}"
        return "Inject request rejected."
    if event_type == "task_cancel_requested":
        return "Cancellation requested."
    if event_type == "task_cancel_ack":
        return "Cancellation acknowledged by orchestrator."
    if event_type == "task_cancel_timeout":
        return "Cancellation wait timed out."
    if event_type == "task_cancelled":
        if mode == "cowork_delegate":
            return "Delegated task cancelled."
        return "Process run cancelled."
    if event_type == "task_completed":
        if mode == "cowork_delegate":
            return "Delegated task completed."
        return "Process run completed."
    if event_type == "task_failed":
        reason = self._one_line(
            event_data.get("reason")
            or event_data.get("error")
            or "",
            140,
        )
        if mode == "cowork_delegate":
            if reason:
                return f"Delegated task failed: {reason}"
            return "Delegated task failed."
        if reason:
            return f"Process run failed: {reason}"
        return "Process run failed."
    detail = self._one_line(
        event_data.get("reason")
        or event_data.get("error")
        or event_data.get("outcome")
        or event_data.get("subtask_id")
        or event_data.get("check_name")
        or event_data.get("classification")
        or "",
        140,
    )
    label = event_type.replace("_", " ").strip()
    if detail:
        return f"{label}: {detail}"
    return label or None


def _now_str() -> str:
    """Return current time as HH:MM:SS string."""
    return datetime.now().strftime("%H:%M:%S")


def _event_data_dict(data: dict[str, Any]) -> dict[str, Any]:
    event_data = data.get("event_data", {})
    if isinstance(event_data, dict):
        return event_data
    return {}


def _apply_process_progress_workspace_side_effects(
    self,
    *,
    event_type: str,
    event_data: dict[str, Any],
) -> None:
    if event_type == "tool_call_completed":
        tool_name = str(event_data.get("tool", "")).strip()
        if self._is_mutating_tool(tool_name):
            self._request_workspace_refresh(f"process:{tool_name}")
        self._ingest_files_panel_from_paths(
            event_data.get(
                "files_changed_paths",
                event_data.get("files_changed", []),
            ),
            operation_hint="modify",
        )
    if event_type in _WORKSPACE_REFRESH_EVENT_TYPES:
        self._request_workspace_refresh(f"process:{event_type}")


def on_process_progress_event(
    self,
    data: dict,
    *,
    run_id: str | None = None,
) -> None:
    """Handle incremental delegate_task progress events in /run flows."""
    if not isinstance(data, dict):
        return
    run = self._process_runs.get(run_id) if run_id else None
    if run_id is not None and run is None:
        return

    if run is not None:
        if run.closed:
            return
        now = time.monotonic()
        run.launch_last_progress_at = now
        run.launch_last_heartbeat_at = 0.0
        run.launch_silent_warning_emitted = False

        event_type = str(data.get("event_type") or "")
        event_data = _event_data_dict(data)
        task_status = str(data.get("status", "") or "").strip().lower()
        if task_status == "cancelled":
            self._set_process_run_status(run, "cancelled")
            run.ended_at = run.ended_at or now
            run.launch_error = "Process run cancelled."
            self._process_run_pending_inject.pop(run.run_id, None)
        elif task_status == "paused":
            self._set_process_run_status(run, "paused")
            run.launch_error = ""
        elif task_status == "failed" and run.status == "cancel_requested":
            self._set_process_run_status(run, "cancel_failed")
            run.launch_error = "Cancellation timed out."
        elif task_status in {"executing", "planning"} and run.status == "paused":
            self._set_process_run_status(run, "running")
            run.launch_error = ""
        elif task_status == "running" and run.status == "cancel_failed":
            self._set_process_run_status(run, "running")
            run.launch_error = ""

        tasks = data.get("tasks", [])
        if isinstance(tasks, list):
            normalized = self._normalize_process_run_tasks(run, tasks)
            run.tasks = normalized

        task_id = str(data.get("task_id", "")).strip()
        if task_id:
            run.task_id = task_id
            self._persist_process_run_conversation_link(run)

        if event_type == "ask_user_requested":
            question_id = str(event_data.get("question_id", "")).strip()
            if question_id:
                seen = self._process_run_seen_questions.setdefault(run.run_id, set())
                if question_id not in seen:
                    seen.add(question_id)
                    self.run_worker(
                        self._prompt_process_run_question(
                            run_id=run.run_id,
                            question_payload=dict(event_data),
                        ),
                        name=f"process-run-question-{run.run_id}-{question_id[:8]}",
                        group=f"process-run-question-{run.run_id}",
                        exclusive=False,
                    )

        _apply_process_progress_workspace_side_effects(
            self,
            event_type=event_type,
            event_data=event_data,
        )
        refresh_due = event_type in _FORCE_REFRESH_EVENT_TYPES
        if not refresh_due:
            last_refresh = float(getattr(run, "progress_ui_last_refresh_at", 0.0) or 0.0)
            refresh_due = (
                (now - last_refresh) >= self._tui_run_progress_refresh_interval_seconds()
            )
        if refresh_due:
            self._refresh_process_run_progress(run)
            self._refresh_process_run_outputs(run)
            self._update_process_run_visuals(run)
            self._refresh_sidebar_progress_summary()
            run.progress_ui_last_refresh_at = now

        message = self._format_process_progress_event(data, run=run)
        if not message:
            return
        if (
            message == run.last_progress_message
            and (now - run.last_progress_at) < 2.0
        ):
            return
        run.last_progress_message = message
        run.last_progress_at = now
        self._append_process_run_activity(run, message)
        try:
            events_panel = self.query_one("#events-panel")
            add_event = getattr(events_panel, "add_event", None)
            if event_type != "token_streamed" and callable(add_event):
                add_event(
                    _now_str(),
                    "process",
                    f"{run.run_id}: {message[:132]}",
                )
        except Exception:
            pass
        return

    self._update_sidebar_tasks(data)
    event_type = str(data.get("event_type") or "")
    event_data = _event_data_dict(data)
    _apply_process_progress_workspace_side_effects(
        self,
        event_type=event_type,
        event_data=event_data,
    )
    message = self._format_process_progress_event(data)
    if not message:
        return
    try:
        chat = self.query_one("#chat-log")
        add_info = getattr(chat, "add_info", None)
        if callable(add_info):
            add_info(message)
    except Exception:
        pass
    try:
        events_panel = self.query_one("#events-panel")
        add_event = getattr(events_panel, "add_event", None)
        if event_type != "token_streamed" and callable(add_event):
            add_event(_now_str(), "process", message[:140])
    except Exception:
        pass
