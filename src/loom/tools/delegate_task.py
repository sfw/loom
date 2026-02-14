"""Task delegation tool: submit complex work to the orchestrator.

Bridges cowork mode to task mode's plan-execute-verify pipeline.
The model calls this when work requires decomposition, verification,
or parallel execution — the same way Claude Code spawns Task subagents.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from loom.tools.registry import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from loom.engine.orchestrator import Orchestrator
    from loom.state.task_state import Task


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

    timeout_seconds = 600  # 10 minutes for complex tasks

    def __init__(
        self,
        orchestrator_factory: Callable[[], Awaitable[Orchestrator]] | None = None,
    ):
        self._factory = orchestrator_factory
        self._orchestrator: Orchestrator | None = None

    def bind(self, orchestrator_factory: Callable[[], Awaitable[Orchestrator]]) -> None:
        """Bind the orchestrator factory after construction."""
        self._factory = orchestrator_factory
        self._orchestrator = None

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
        workspace = str(ctx.workspace) if ctx.workspace else ""

        # Lazy-init orchestrator
        if self._orchestrator is None:
            try:
                self._orchestrator = await self._factory()
            except Exception as e:
                return ToolResult.fail(f"Failed to initialize orchestrator: {e}")

        # Build task
        task = _create_task(goal, workspace, context)

        if not wait:
            asyncio.create_task(self._orchestrator.execute_task(task))
            return ToolResult.ok(
                f"Task submitted (async): {task.id}\n"
                f"Goal: {goal}\n"
                f"Use task_tracker to monitor progress."
            )

        # Synchronous: execute and wait
        try:
            completed = await self._orchestrator.execute_task(task)
            return ToolResult.ok(
                _format_result(completed),
                files_changed=_collect_files_changed(completed),
            )
        except Exception as e:
            return ToolResult.fail(f"Task execution failed: {e}")


def _create_task(goal: str, workspace: str, context: dict, approval_mode: str = "confidence_threshold") -> Task:
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
    """Extract file paths from task changes for the ToolResult."""
    if task.workspace_changes.last_change:
        return [task.workspace_changes.last_change]
    return []
