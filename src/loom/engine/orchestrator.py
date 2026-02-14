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
from loom.engine.verification import VerificationGates
from loom.events.bus import Event, EventBus
from loom.events.types import (
    SUBTASK_COMPLETED,
    SUBTASK_FAILED,
    SUBTASK_STARTED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLAN_READY,
    TASK_PLANNING,
    TASK_REPLANNING,
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
from loom.tools.workspace import ChangeLog


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
        self._verification = VerificationGates(
            model_router=model_router,
            prompt_assembler=prompt_assembler,
            config=config.verification,
        )

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

                # Retry + re-planning on failure
                if result.status == "failed":
                    if subtask.retry_count < subtask.max_retries:
                        subtask.retry_count += 1
                        subtask.status = SubtaskStatus.PENDING
                        task.update_subtask(
                            subtask.id,
                            status=SubtaskStatus.PENDING,
                            retry_count=subtask.retry_count,
                        )
                        self._state.save(task)
                    else:
                        # All retries exhausted — trigger re-planning
                        replanned = await self._replan_task(task)
                        if not replanned:
                            break  # Re-planning failed, stop

            # 3. Completion
            return self._finalize_task(task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.add_error("orchestrator", str(e))
            self._state.save(task)
            self._emit(TASK_FAILED, task.id, {"error": str(e)})
            return task

    async def _replan_task(self, task: Task) -> bool:
        """Re-plan the task after subtask failures.

        Returns True if re-planning succeeded and execution can continue.
        """
        self._emit(TASK_REPLANNING, task.id, {"reason": "subtask_failures"})

        # Gather discoveries and errors for the replanner
        discoveries = [d for d in task.decisions_log]
        errors = [
            f"{e.subtask}: {e.error}" for e in task.errors_encountered
        ]

        try:
            state_yaml = self._state.to_compact_yaml(task)
            prompt = self._prompts.build_replanner_prompt(
                goal=task.goal,
                current_state_yaml=state_yaml,
                discoveries=discoveries,
                errors=errors,
                original_plan=task.plan,
            )

            model = self._router.select(tier=2, role="planner")
            response = await model.complete([{"role": "user", "content": prompt}])
            new_plan = self._parse_plan(response)

            # Preserve completed subtask state
            completed_ids = {
                s.id for s in task.plan.subtasks
                if s.status == SubtaskStatus.COMPLETED
            }
            new_plan.version = task.plan.version + 1
            new_plan.last_replanned = datetime.now().isoformat()

            # Mark any subtasks that were already completed
            for s in new_plan.subtasks:
                if s.id in completed_ids:
                    s.status = SubtaskStatus.COMPLETED

            task.plan = new_plan
            self._state.save(task)

            self._emit(TASK_PLAN_READY, task.id, {
                "subtask_count": len(new_plan.subtasks),
                "version": new_plan.version,
                "replanned": True,
            })
            return True

        except Exception as e:
            task.add_error("replanner", str(e))
            self._state.save(task)
            return False

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

    def _get_changelog(self, task: Task) -> ChangeLog | None:
        """Get or create a ChangeLog for the task's workspace."""
        if not task.workspace:
            return None
        workspace = Path(task.workspace)
        data_dir = self._state._data_dir / "tasks" / task.id
        data_dir.mkdir(parents=True, exist_ok=True)
        return ChangeLog(task_id=task.id, workspace=workspace, data_dir=data_dir)

    async def _execute_subtask(self, task: Task, subtask: Subtask) -> SubtaskResult:
        """Execute a single subtask with tool-calling loop."""
        subtask.status = SubtaskStatus.RUNNING
        self._state.save(task)
        self._emit(SUBTASK_STARTED, task.id, {"subtask_id": subtask.id})

        start_time = time.monotonic()
        workspace = Path(task.workspace) if task.workspace else None
        changelog = self._get_changelog(task)

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
                # Validate tool calls before execution
                validation = self._validator.validate_tool_calls(
                    response, self._tools.all_schemas()
                )
                if not validation.valid:
                    # Invalid tool calls — inject error and retry
                    messages.append({
                        "role": "assistant",
                        "content": response.text or None,
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
                        tc.name, tc.arguments,
                        workspace=workspace,
                        changelog=changelog,
                        subtask_id=subtask.id,
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

        # Run verification gates
        verification = await self._verification.verify(
            subtask=subtask,
            result_summary=summary,
            tool_calls=tool_calls_record,
            workspace=workspace,
            tier=subtask.verification_tier,
        )

        if not verification.passed:
            result.status = "failed"
            subtask.status = SubtaskStatus.FAILED
            subtask.summary = verification.feedback or "Verification failed"
            task.update_subtask(
                subtask.id,
                status=SubtaskStatus.FAILED,
                summary=subtask.summary,
            )
            task.add_error(subtask.id, f"Verification failed (tier {verification.tier})")
            self._state.save(task)
            self._emit(SUBTASK_FAILED, task.id, {
                "subtask_id": subtask.id,
                "verification_tier": verification.tier,
                "feedback": verification.feedback,
            })
            return result

        # Update state
        subtask.status = SubtaskStatus.COMPLETED
        subtask.summary = summary
        task.update_subtask(subtask.id, status=SubtaskStatus.COMPLETED, summary=summary)

        # Update workspace_changes from changelog
        if changelog:
            change_summary = changelog.get_summary()
            task.workspace_changes.files_created = len(change_summary["created"])
            task.workspace_changes.files_modified = len(change_summary["modified"])
            task.workspace_changes.files_deleted = len(change_summary["deleted"])
            task.workspace_changes.last_change = datetime.now().isoformat()

        self._state.save(task)

        self._emit(SUBTASK_COMPLETED, task.id, {
            "subtask_id": subtask.id,
            "status": result.status,
            "summary": summary,
            "duration": elapsed,
        })

        # Extract structured memory entries from the subtask execution
        await self._extract_memory(task, subtask, result)

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

    async def _extract_memory(
        self, task: Task, subtask: Subtask, result: SubtaskResult
    ) -> None:
        """Extract structured memory entries from subtask execution.

        Uses a tier-1 extractor model to identify decisions, discoveries,
        errors, and artifacts from the tool calls and output.
        """
        try:
            model = self._router.select(tier=1, role="extractor")
        except Exception:
            # No extractor model configured — skip memory extraction
            return

        # Format tool calls for the prompt
        tool_lines = []
        for tc in result.tool_calls:
            status = "OK" if tc.result.success else f"FAILED: {tc.result.error}"
            tool_lines.append(f"- {tc.tool}({json.dumps(tc.args)}) → {status}")
        tool_calls_formatted = "\n".join(tool_lines) if tool_lines else "No tool calls."

        prompt = self._prompts.build_extractor_prompt(
            subtask_id=subtask.id,
            tool_calls_formatted=tool_calls_formatted,
            model_output=result.summary,
        )

        try:
            response = await model.complete([{"role": "user", "content": prompt}])
            entries = self._parse_memory_entries(response, task.id, subtask.id)
            if entries:
                await self._memory.store_many(entries)
        except Exception:
            pass  # Memory extraction is best-effort

    def _parse_memory_entries(
        self, response: ModelResponse, task_id: str, subtask_id: str
    ) -> list:
        """Parse extractor model response into MemoryEntry objects."""
        from loom.state.memory import MemoryEntry

        validation = self._validator.validate_json_response(
            response, expected_keys=["entries"]
        )
        if not validation.valid or validation.parsed is None:
            return []

        entries = []
        for e in validation.parsed.get("entries", []):
            entry_type = e.get("type", "discovery")
            if entry_type not in (
                "decision", "error", "tool_result", "discovery", "artifact", "context"
            ):
                entry_type = "discovery"
            entries.append(MemoryEntry(
                task_id=task_id,
                subtask_id=subtask_id,
                entry_type=entry_type,
                summary=str(e.get("summary", ""))[:150],
                detail=str(e.get("detail", "")),
                tags=str(e.get("tags", "")),
            ))
        return entries

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
