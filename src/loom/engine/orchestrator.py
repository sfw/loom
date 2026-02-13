"""Core orchestrator loop.

Drives task execution: plan -> execute subtasks -> verify -> complete.
The model never decides to "continue" — the harness does.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loom.config import Config
from loom.engine.scheduler import Scheduler
from loom.events.bus import Event, EventBus
from loom.events.types import (
    SUBTASK_COMPLETED,
    SUBTASK_STARTED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLAN_READY,
    TASK_PLANNING,
)
from loom.models.base import ModelResponse
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.memory import MemoryManager
from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStateManager,
    TaskStatus,
)
from loom.tools.registry import ToolRegistry, ToolResult


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation during subtask execution."""

    tool: str
    args: dict
    result: ToolResult
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class SubtaskResult:
    """Result of a subtask execution."""

    status: str  # success, failed, blocked
    summary: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    duration_seconds: float = 0.0
    tokens_used: int = 0
    model_used: str = ""


class Orchestrator:
    """Core orchestrator loop."""

    def __init__(
        self,
        model_router: ModelRouter,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        prompt_assembler: PromptAssembler,
        state_manager: TaskStateManager,
        event_bus: EventBus,
        config: Config,
    ):
        self._router = model_router
        self._tools = tool_registry
        self._memory = memory_manager
        self._prompts = prompt_assembler
        self._state = state_manager
        self._events = event_bus
        self._config = config
        self._scheduler = Scheduler()
        self._validator = ResponseValidator()

    async def execute_task(self, task: Task) -> Task:
        """Main entry point. Drives the full task lifecycle."""
        try:
            # 1. Planning phase
            task.status = TaskStatus.PLANNING
            self._emit(TASK_PLANNING, task.id, {"goal": task.goal})

            plan = await self._plan_task(task)
            task.plan = plan
            task.status = TaskStatus.EXECUTING
            self._state.save(task)
            self._emit(TASK_PLAN_READY, task.id, {
                "subtask_count": len(plan.subtasks),
                "subtask_ids": [s.id for s in plan.subtasks],
            })

            # 2. Execution loop
            self._emit(TASK_EXECUTING, task.id, {})
            iteration = 0
            max_iterations = self._config.execution.max_loop_iterations

            while self._scheduler.has_pending(task.plan) and iteration < max_iterations:
                iteration += 1

                if task.status == TaskStatus.CANCELLED:
                    break

                subtask = self._scheduler.next_subtask(task.plan)
                if subtask is None:
                    # All remaining subtasks are blocked
                    break

                result = await self._execute_subtask(task, subtask)

                # Re-planning gate
                if result.status == "failed":
                    # Could trigger re-planning here (Phase 2)
                    pass

            # 3. Completion
            return self._finalize_task(task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.add_error("orchestrator", str(e))
            self._state.save(task)
            self._emit(TASK_FAILED, task.id, {"error": str(e)})
            return task

    async def _plan_task(self, task: Task) -> Plan:
        """Invoke the planner model to decompose the task into subtasks."""
        # Build workspace listing if available
        workspace_listing = ""
        if task.workspace:
            workspace_path = Path(task.workspace)
            if workspace_path.exists():
                listing_result = await self._tools.execute(
                    "list_directory", {}, workspace=workspace_path
                )
                if listing_result.success:
                    workspace_listing = listing_result.output

        prompt = self._prompts.build_planner_prompt(
            task=task,
            workspace_listing=workspace_listing,
        )

        model = self._router.select(tier=2, role="planner")
        response = await model.complete([{"role": "user", "content": prompt}])

        return self._parse_plan(response)

    def _parse_plan(self, response: ModelResponse) -> Plan:
        """Parse a plan from the model's JSON response."""
        validation = self._validator.validate_json_response(
            response, expected_keys=["subtasks"]
        )

        if not validation.valid or validation.parsed is None:
            # Fallback: create a single-step plan
            return Plan(
                subtasks=[Subtask(
                    id="execute-goal",
                    description="Execute the task goal directly",
                    model_tier=2,
                )],
                version=1,
            )

        subtasks = []
        for s in validation.parsed.get("subtasks", []):
            subtasks.append(Subtask(
                id=s.get("id", f"step-{len(subtasks) + 1}"),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
                model_tier=s.get("model_tier", 1),
                verification_tier=s.get("verification_tier", 1),
                is_critical_path=s.get("is_critical_path", False),
                acceptance_criteria=s.get("acceptance_criteria", ""),
                max_retries=self._config.execution.max_subtask_retries,
            ))

        return Plan(subtasks=subtasks, version=1)

    async def _execute_subtask(self, task: Task, subtask: Subtask) -> SubtaskResult:
        """Execute a single subtask with tool-calling loop."""
        subtask.status = SubtaskStatus.RUNNING
        self._state.save(task)
        self._emit(SUBTASK_STARTED, task.id, {"subtask_id": subtask.id})

        start_time = time.monotonic()
        workspace = Path(task.workspace) if task.workspace else None

        # Assemble prompt
        memory_entries = await self._memory.query_relevant(task.id, subtask.id)
        prompt = self._prompts.build_executor_prompt(
            task=task,
            subtask=subtask,
            state_manager=self._state,
            memory_entries=memory_entries,
            available_tools=self._tools.all_schemas(),
        )

        # Select model
        model = self._router.select(tier=subtask.model_tier, role="executor")

        # Inner tool-calling loop
        messages: list[dict] = [{"role": "user", "content": prompt}]
        tool_calls_record: list[ToolCallRecord] = []
        max_tool_iterations = 20
        total_tokens = 0

        for _ in range(max_tool_iterations):
            response = await model.complete(
                messages, tools=self._tools.all_schemas()
            )
            total_tokens += response.usage.total_tokens

            if response.has_tool_calls():
                # Process tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })

                for tc in response.tool_calls:
                    tool_result = await self._tools.execute(
                        tc.name, tc.arguments, workspace=workspace
                    )
                    tool_calls_record.append(ToolCallRecord(
                        tool=tc.name, args=tc.arguments, result=tool_result,
                    ))

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result.to_json(),
                    })

                # Anti-amnesia: inject TODO reminder
                messages.append({
                    "role": "system",
                    "content": self._build_todo_reminder(task, subtask),
                })
            else:
                # Text-only response — subtask complete
                break

        elapsed = time.monotonic() - start_time
        summary = response.text[:200] if response.text else "No output"

        result = SubtaskResult(
            status="success",
            summary=summary,
            tool_calls=tool_calls_record,
            duration_seconds=elapsed,
            tokens_used=total_tokens,
            model_used=model.name,
        )

        # Update state
        subtask.status = SubtaskStatus.COMPLETED
        subtask.summary = summary
        task.update_subtask(subtask.id, status=SubtaskStatus.COMPLETED, summary=summary)
        self._state.save(task)

        self._emit(SUBTASK_COMPLETED, task.id, {
            "subtask_id": subtask.id,
            "status": result.status,
            "summary": summary,
            "duration": elapsed,
        })

        return result

    def _finalize_task(self, task: Task) -> Task:
        """Finalize task: set status, emit events."""
        completed, total = task.progress
        all_done = completed == total and total > 0

        if task.status == TaskStatus.CANCELLED:
            # Skip remaining subtasks
            for s in task.plan.subtasks:
                if s.status == SubtaskStatus.PENDING:
                    s.status = SubtaskStatus.SKIPPED
            self._emit(TASK_CANCELLED, task.id, {"completed": completed, "total": total})
        elif all_done:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            self._emit(TASK_COMPLETED, task.id, {
                "completed": completed,
                "total": total,
            })
        else:
            task.status = TaskStatus.FAILED
            failed = [s for s in task.plan.subtasks if s.status == SubtaskStatus.FAILED]
            self._emit(TASK_FAILED, task.id, {
                "completed": completed,
                "total": total,
                "failed_subtasks": [s.id for s in failed],
            })

        self._state.save(task)
        return task

    @staticmethod
    def _build_todo_reminder(task: Task, subtask: Subtask) -> str:
        """Anti-amnesia: remind the model what it should be doing."""
        return (
            f"CURRENT TASK STATE:\n"
            f"Goal: {task.goal}\n"
            f"Current subtask: {subtask.id} — {subtask.description}\n\n"
            f"REMAINING WORK FOR THIS SUBTASK:\n"
            f"Continue working on: {subtask.description}\n"
            f"Do NOT move to the next subtask. Complete ONLY this one.\n"
            f"When finished, provide a summary of what you accomplished."
        )

    def _emit(self, event_type: str, task_id: str, data: dict) -> None:
        self._events.emit(Event(
            event_type=event_type, task_id=task_id, data=data
        ))

    def cancel_task(self, task: Task) -> None:
        """Mark a task for cancellation."""
        task.status = TaskStatus.CANCELLED
        self._state.save(task)


def create_task(
    goal: str,
    workspace: str = "",
    approval_mode: str = "auto",
    context: dict | None = None,
) -> Task:
    """Factory for creating new tasks with a generated ID."""
    return Task(
        id=uuid.uuid4().hex[:8],
        goal=goal,
        workspace=workspace,
        approval_mode=approval_mode,
        context=context or {},
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
