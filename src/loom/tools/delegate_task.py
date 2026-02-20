"""Task delegation tool: submit complex work to the orchestrator.

Bridges cowork mode to task mode's plan-execute-verify pipeline.
The model calls this when work requires decomposition, verification,
or parallel execution — the same way Claude Code spawns Task subagents.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loom.state.task_state import TaskStatus
from loom.tools.registry import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from loom.engine.orchestrator import Orchestrator
    from loom.state.task_state import Task


DEFAULT_DELEGATE_TIMEOUT_SECONDS = 3600
logger = logging.getLogger(__name__)


def _normalize_timeout_seconds(value: object, *, default: int) -> int:
    """Normalize timeout input to a positive integer."""
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return default
    return normalized if normalized > 0 else default


def _delegate_timeout_seconds(default_timeout: int) -> int:
    """Resolve delegate timeout from env with safe fallback."""
    raw = os.environ.get("LOOM_DELEGATE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return default_timeout
    return _normalize_timeout_seconds(raw, default=default_timeout)


class DelegateTaskTool(Tool):
    """Delegate complex work to Loom's task orchestration engine.

    Use this when work requires:
    - Breaking down into multiple steps with dependencies
    - Verification of each step's output
    - Parallel execution of independent steps
    - Structured planning before execution

    For simple operations (read a file, run a command, edit code),
    use the direct tools instead.  Delegation adds overhead — only
    use it when the task is genuinely complex.
    """

    name = "delegate_task"
    description = (
        "Submit complex multi-step work to the task orchestrator. "
        "The orchestrator will plan, decompose into subtasks, execute "
        "with verification, and return results. Use for tasks that need "
        "decomposition, parallel execution, or step-by-step verification. "
        "Simple operations should use direct tools instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "What needs to be accomplished. Be specific — include "
                    "file paths, constraints, and acceptance criteria."
                ),
            },
            "context": {
                "type": "object",
                "description": (
                    "Additional context from the conversation: constraints, "
                    "decisions, preferences, files already discussed."
                ),
            },
            "wait": {
                "type": "boolean",
                "description": (
                    "If true (default), block until task completes and "
                    "return full results. If false, return task_id for "
                    "later status checks."
                ),
            },
        },
        "required": ["goal"],
    }

    def __init__(
        self,
        orchestrator_factory: Callable[..., Awaitable[Orchestrator]] | None = None,
        timeout_seconds: int | None = None,
    ):
        self._factory = orchestrator_factory
        self._configured_timeout_seconds = _normalize_timeout_seconds(
            (
                timeout_seconds
                if timeout_seconds is not None
                else DEFAULT_DELEGATE_TIMEOUT_SECONDS
            ),
            default=DEFAULT_DELEGATE_TIMEOUT_SECONDS,
        )

    @property
    def timeout_seconds(self) -> int:
        # Long-running orchestration can exceed simple tool time budgets.
        return _delegate_timeout_seconds(self._configured_timeout_seconds)

    def bind(
        self,
        orchestrator_factory: Callable[..., Awaitable[Orchestrator]],
    ) -> None:
        """Bind the orchestrator factory after construction."""
        self._factory = orchestrator_factory

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        if self._factory is None:
            return ToolResult.fail(
                "Task delegation is not available (no orchestrator configured)."
            )

        goal = args.get("goal", "")
        if not goal:
            return ToolResult.fail("'goal' parameter is required.")

        context = args.get("context", {})
        wait = args.get("wait", True)
        progress_callback = args.get("_progress_callback")
        process_override = args.get("_process_override")
        approval_mode = str(
            args.get("_approval_mode", "confidence_threshold"),
        ).strip() or "confidence_threshold"
        read_roots_raw = args.get("_read_roots", [])
        if isinstance(read_roots_raw, str):
            read_roots_raw = [read_roots_raw]
        read_roots: list[str] = []
        if isinstance(read_roots_raw, list):
            for item in read_roots_raw:
                text = str(item or "").strip()
                if text:
                    read_roots.append(text)
        workspace = str(ctx.workspace) if ctx.workspace else ""

        # Fresh orchestrator per call to isolate concurrent delegated runs.
        try:
            supports_override = False
            try:
                signature = inspect.signature(self._factory)
                supports_override = len(signature.parameters) > 0
            except (TypeError, ValueError):
                supports_override = False
            if supports_override:
                orchestrator = await self._factory(process_override)
            else:
                orchestrator = await self._factory()
        except Exception as e:
            return ToolResult.fail(f"Failed to initialize orchestrator: {e}")

        # Build task
        task = _create_task(
            goal,
            workspace,
            context,
            approval_mode=approval_mode,
        )
        if read_roots:
            task.metadata["read_roots"] = read_roots
        event_bus = getattr(orchestrator, "_events", None)
        event_log_handle = None
        event_log_path = ""
        event_log_sequence = 0
        event_log_broken = False

        if wait:
            event_log_file = _resolve_delegate_event_log_file(orchestrator, task.id)
            if event_log_file is not None:
                try:
                    event_log_file.parent.mkdir(parents=True, exist_ok=True)
                    event_log_handle = event_log_file.open(
                        "a", encoding="utf-8",
                    )
                    event_log_path = str(event_log_file)
                except Exception as e:
                    logger.warning(
                        "Failed to open delegate event log for %s: %s",
                        task.id,
                        e,
                    )

        def _log_event(
            kind: str,
            *,
            payload: dict | None = None,
            **metadata: object,
        ) -> None:
            nonlocal event_log_sequence, event_log_broken
            if event_log_handle is None or event_log_broken:
                return

            event_log_sequence += 1
            record: dict[str, object] = {
                "seq": event_log_sequence,
                "timestamp": datetime.now().isoformat(),
                "kind": kind,
                "task_id": task.id,
            }
            if payload is not None:
                record["payload"] = payload
            if metadata:
                record.update(metadata)
            try:
                event_log_handle.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
                event_log_handle.flush()
            except Exception as e:
                event_log_broken = True
                logger.warning(
                    "Failed writing delegate event log for %s: %s",
                    task.id,
                    e,
                )

        _log_event(
            "meta",
            payload={
                "goal": goal,
                "workspace": workspace,
                "wait": bool(wait),
                "has_progress_callback": bool(callable(progress_callback)),
                "process_override": bool(process_override),
                "approval_mode": approval_mode,
                "read_roots_count": len(read_roots),
            },
            event_log_path=event_log_path,
        )
        subscriptions: list[tuple[str, object]] = []
        token_burst_count = 0
        token_burst_subtask = ""
        last_token_emit = 0.0

        def _emit_progress(
            event_type: str | None = None,
            event_data: dict | None = None,
        ) -> None:
            payload = _progress_payload(task)
            if event_log_path:
                payload["event_log_path"] = event_log_path
            if event_type:
                payload["event_type"] = event_type
            if isinstance(event_data, dict) and event_data:
                payload["event_data"] = dict(event_data)
            _log_event("progress_snapshot", payload=payload)
            if not callable(progress_callback):
                return
            try:
                maybe = progress_callback(payload)
                if inspect.isawaitable(maybe):
                    asyncio.create_task(maybe)
            except Exception:
                pass

        def _flush_token_burst() -> None:
            nonlocal token_burst_count, token_burst_subtask, last_token_emit
            if token_burst_count <= 0:
                return
            payload = {"token_count": token_burst_count}
            if token_burst_subtask:
                payload["subtask_id"] = token_burst_subtask
            _emit_progress("token_streamed", payload)
            token_burst_count = 0
            token_burst_subtask = ""
            last_token_emit = time.monotonic()

        if event_bus is not None and (
            callable(progress_callback) or event_log_handle is not None
        ):
            from loom.events.types import (
                MODEL_INVOCATION,
                SUBTASK_COMPLETED,
                SUBTASK_FAILED,
                SUBTASK_RETRYING,
                SUBTASK_STARTED,
                TASK_COMPLETED,
                TASK_EXECUTING,
                TASK_FAILED,
                TASK_PLAN_READY,
                TASK_PLANNING,
                TASK_REPLANNING,
                TOKEN_STREAMED,
                TOOL_CALL_COMPLETED,
                TOOL_CALL_STARTED,
            )

            observed = (
                TASK_PLANNING,
                TASK_PLAN_READY,
                TASK_EXECUTING,
                MODEL_INVOCATION,
                SUBTASK_STARTED,
                SUBTASK_RETRYING,
                TOKEN_STREAMED,
                TOOL_CALL_STARTED,
                TOOL_CALL_COMPLETED,
                SUBTASK_COMPLETED,
                SUBTASK_FAILED,
                TASK_REPLANNING,
                TASK_COMPLETED,
                TASK_FAILED,
            )

            def _on_event(event) -> None:
                nonlocal token_burst_count, token_burst_subtask, last_token_emit
                if getattr(event, "task_id", "") != task.id:
                    return
                event_data = getattr(event, "data", None)
                _log_event(
                    "event_bus",
                    payload={
                        "event_type": str(getattr(event, "event_type", "")),
                        "event_timestamp": str(getattr(event, "timestamp", "")),
                        "event_data": (
                            dict(event_data) if isinstance(event_data, dict) else event_data
                        ),
                    },
                )
                if event.event_type == TOKEN_STREAMED:
                    token_burst_count += 1
                    if isinstance(event_data, dict):
                        subtask = str(event_data.get("subtask_id", "")).strip()
                        if subtask:
                            token_burst_subtask = subtask
                    now = time.monotonic()
                    if token_burst_count < 40 and (now - last_token_emit) < 2.0:
                        return
                    _flush_token_burst()
                    return
                if token_burst_count > 0:
                    _flush_token_burst()
                _emit_progress(event.event_type, event_data)

            for event_type in observed:
                try:
                    event_bus.subscribe(event_type, _on_event)
                    subscriptions.append((event_type, _on_event))
                except Exception:
                    pass

        if not wait:
            asyncio.create_task(orchestrator.execute_task(task))
            return ToolResult.ok(
                f"Task submitted (async): {task.id}\n"
                f"Goal: {goal}\n"
                f"Use task_tracker to monitor progress.",
                data={
                    "task_id": task.id,
                    "status": "submitted",
                    "tasks": [],
                },
            )

        # Synchronous: execute and wait
        try:
            _emit_progress()
            completed = await orchestrator.execute_task(task)
            _flush_token_burst()
            status_raw = (
                completed.status.value
                if hasattr(completed.status, "value")
                else str(completed.status)
            ).strip().lower()
            payload = _progress_payload(completed)
            if event_log_path:
                payload["event_log_path"] = event_log_path
            summary = _format_result(completed)
            if event_log_path:
                summary += f"\n\nEvent log: {event_log_path}"
            files_changed = _collect_files_changed(completed)
            if status_raw == TaskStatus.COMPLETED.value:
                _emit_progress("task_completed")
                return ToolResult.ok(
                    summary,
                    files_changed=files_changed,
                    data=payload,
                )

            if status_raw == TaskStatus.CANCELLED.value:
                _emit_progress("task_failed", {"reason": "Task cancelled"})
                return ToolResult(
                    success=False,
                    output=summary,
                    error="Task execution cancelled.",
                    files_changed=files_changed,
                    data=payload,
                )

            _emit_progress("task_failed", {"reason": "Task failed"})
            return ToolResult(
                success=False,
                output=summary,
                error="Task execution failed.",
                files_changed=files_changed,
                data=payload,
            )
        except Exception as e:
            _log_event(
                "delegate_exception",
                payload={
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )
            if event_log_path:
                return ToolResult.fail(
                    f"Task execution failed: {e} (event log: {event_log_path})"
                )
            return ToolResult.fail(f"Task execution failed: {e}")
        finally:
            if event_bus is not None and subscriptions:
                for event_type, handler in subscriptions:
                    try:
                        event_bus.unsubscribe(event_type, handler)
                    except Exception:
                        pass
            if event_log_handle is not None:
                try:
                    event_log_handle.close()
                except Exception:
                    pass


def _create_task(
    goal: str, workspace: str, context: dict,
    approval_mode: str = "confidence_threshold",
) -> Task:
    """Create a Task object for the orchestrator."""
    from loom.state.task_state import Task, TaskStatus

    task_id = f"cowork-{uuid.uuid4().hex[:8]}"
    return Task(
        id=task_id,
        goal=goal,
        workspace=workspace,
        context=context,
        status=TaskStatus.PENDING,
        approval_mode=approval_mode,
        created_at=datetime.now().isoformat(),
    )


def _resolve_delegate_event_log_file(
    orchestrator: object,
    task_id: str,
) -> Path | None:
    """Resolve per-task JSONL event log path from orchestrator config."""
    config = getattr(orchestrator, "_config", None)
    if config is None:
        return None

    raw_dir: object | None = None
    try:
        raw_dir = getattr(config, "log_path")
    except Exception:
        raw_dir = None

    if raw_dir is None:
        logging_cfg = getattr(config, "logging", None)
        if logging_cfg is not None:
            raw_dir = getattr(logging_cfg, "event_log_path", None)

    if raw_dir is None:
        return None

    # Ignore mock-derived path-like values to avoid creating literal
    # "MagicMock/..." folders during tests.
    if type(raw_dir).__module__.startswith("unittest.mock"):
        return None
    try:
        fspath_value = os.fspath(raw_dir)
    except TypeError:
        return None
    if isinstance(fspath_value, bytes):
        try:
            candidate = fspath_value.decode()
        except UnicodeDecodeError:
            return None
    elif isinstance(fspath_value, str):
        candidate = fspath_value
    else:
        return None
    candidate = candidate.strip()
    if not candidate or candidate.startswith("MagicMock/") or "<MagicMock" in candidate:
        return None

    log_dir = Path(candidate).expanduser()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"{stamp}-{task_id}.events.jsonl"


def _format_result(task: Task) -> str:
    """Format completed task as a readable summary for the conversation."""
    lines = []
    status = task.status.value if hasattr(task.status, "value") else str(task.status)
    lines.append(f'Task {status}: "{task.goal}"')
    lines.append("")

    # Subtask summary
    if task.plan.subtasks:
        lines.append("Subtasks:")
        for s in task.plan.subtasks:
            s_status = s.status.value if hasattr(s.status, "value") else str(s.status)
            icon = {"completed": "[x]", "failed": "[!]", "skipped": "[-]"}.get(
                s_status, "[ ]"
            )
            desc = s.summary or s.description
            lines.append(f"  {icon} {desc}")
        lines.append("")

    # Files changed
    changes = task.workspace_changes
    if changes.files_created or changes.files_modified or changes.files_deleted:
        lines.append("Files changed:")
        if changes.files_created:
            lines.append(f"  {changes.files_created} created")
        if changes.files_modified:
            lines.append(f"  {changes.files_modified} modified")
        if changes.files_deleted:
            lines.append(f"  {changes.files_deleted} deleted")
        lines.append("")

    # Decisions
    if task.decisions_log:
        lines.append("Decisions:")
        for d in task.decisions_log[-5:]:
            lines.append(f"  - {d}")
        lines.append("")

    # Errors
    if task.errors_encountered:
        lines.append("Errors encountered:")
        for e in task.errors_encountered:
            resolution = f" (resolved: {e.resolution})" if e.resolution else ""
            lines.append(f"  - {e.error}{resolution}")

    return "\n".join(lines)


def _collect_files_changed(task: Task) -> list[str]:
    """Report file change summary from task workspace changes."""
    changes = task.workspace_changes
    result: list[str] = []
    if changes.files_created:
        result.append(f"({changes.files_created} files created)")
    if changes.files_modified:
        result.append(f"({changes.files_modified} files modified)")
    if changes.files_deleted:
        result.append(f"({changes.files_deleted} files deleted)")
    return result


def _progress_payload(task: Task) -> dict:
    """Build a sidebar-friendly progress payload from orchestrator task state."""
    task_status = task.status.value if hasattr(task.status, "value") else str(task.status)
    progress_rows: list[dict] = []
    for subtask in task.plan.subtasks:
        raw_status = (
            subtask.status.value
            if hasattr(subtask.status, "value")
            else str(subtask.status)
        )
        if raw_status in {"running", "blocked"}:
            status = "in_progress"
        elif raw_status in {"completed", "failed", "skipped"}:
            status = raw_status
        else:
            status = "pending"
        content = (subtask.summary or subtask.description or subtask.id).strip()
        progress_rows.append({
            "id": subtask.id,
            "status": status,
            "content": content,
        })
    return {
        "task_id": task.id,
        "status": task_status,
        "tasks": progress_rows,
    }
