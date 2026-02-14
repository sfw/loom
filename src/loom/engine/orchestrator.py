"""Core orchestrator loop.

Drives task execution: plan -> execute subtasks -> verify -> complete.
The model never decides to "continue" — the harness does.

Subtask execution is delegated to SubtaskRunner.  Independent subtasks
(no unmet dependencies) are dispatched in parallel up to
``config.execution.max_parallel_subtasks``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from pathlib import Path

from loom.config import Config
from loom.engine.runner import SubtaskResult, SubtaskRunner, ToolCallRecord
from loom.engine.scheduler import Scheduler
from loom.engine.verification import VerificationGates, VerificationResult
from loom.events.bus import Event, EventBus
from loom.events.types import (
    SUBTASK_COMPLETED,
    SUBTASK_FAILED,
    SUBTASK_RETRYING,
    SUBTASK_STARTED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLAN_READY,
    TASK_PLANNING,
    TASK_REPLANNING,
)
from loom.learning.manager import LearningManager
from loom.models.base import ModelResponse
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalDecision, ApprovalManager, ApprovalRequest
from loom.recovery.confidence import ConfidenceScorer
from loom.recovery.retry import AttemptRecord, RetryManager
from loom.state.memory import MemoryManager
from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStateManager,
    TaskStatus,
)
from loom.tools.registry import ToolRegistry
from loom.tools.workspace import ChangeLog

# Re-export dataclasses that existing code imports from here
__all__ = [
    "Orchestrator",
    "SubtaskResult",
    "ToolCallRecord",
    "create_task",
]


class Orchestrator:
    """Core orchestrator loop.

    Responsibilities:
    - Task lifecycle (planning, execution, finalization)
    - Subtask scheduling and parallel dispatch
    - Retry / escalation / re-planning decisions
    - Approval gating (confidence-based)
    - Event emission and state persistence
    - Post-task learning

    Subtask execution (tool loop, verification, memory extraction)
    is delegated to SubtaskRunner.
    """

    def __init__(
        self,
        model_router: ModelRouter,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        prompt_assembler: PromptAssembler,
        state_manager: TaskStateManager,
        event_bus: EventBus,
        config: Config,
        approval_manager: ApprovalManager | None = None,
        learning_manager: LearningManager | None = None,
    ):
        self._router = model_router
        self._tools = tool_registry
        self._memory = memory_manager
        self._prompts = prompt_assembler
        self._state = state_manager
        self._events = event_bus
        self._config = config
        self._learning = learning_manager
        self._scheduler = Scheduler()
        self._validator = ResponseValidator()
        self._verification = VerificationGates(
            model_router=model_router,
            prompt_assembler=prompt_assembler,
            config=config.verification,
        )
        self._confidence = ConfidenceScorer()
        self._approval = approval_manager or ApprovalManager(event_bus)
        self._retry = RetryManager(
            max_retries=config.execution.max_subtask_retries,
        )
        self._state_lock = asyncio.Lock()

        # Runner handles the inner subtask execution
        self._runner = SubtaskRunner(
            model_router=model_router,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            prompt_assembler=prompt_assembler,
            state_manager=state_manager,
            verification=self._verification,
            config=config,
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

            # 2. Execution loop — parallel dispatch of independent subtasks
            self._emit(TASK_EXECUTING, task.id, {})
            iteration = 0
            max_iterations = self._config.execution.max_loop_iterations
            max_parallel = self._config.execution.max_parallel_subtasks
            attempts_by_subtask: dict[str, list[AttemptRecord]] = {}

            while self._scheduler.has_pending(task.plan) and iteration < max_iterations:
                if task.status == TaskStatus.CANCELLED:
                    break

                # Get all runnable subtasks (dependencies met)
                runnable = self._scheduler.runnable_subtasks(task.plan)
                if not runnable:
                    break

                # Cap to max_parallel_subtasks
                batch = runnable[:max_parallel]
                iteration += 1

                # Dispatch batch
                if len(batch) == 1:
                    # Single subtask — no gather overhead
                    outcomes = [await self._dispatch_subtask(
                        task, batch[0], attempts_by_subtask,
                    )]
                else:
                    # Parallel dispatch — use return_exceptions so one
                    # failure doesn't abort the entire batch.
                    raw_outcomes = await asyncio.gather(
                        *[
                            self._dispatch_subtask(
                                task, s, attempts_by_subtask,
                            )
                            for s in batch
                        ],
                        return_exceptions=True,
                    )
                    outcomes = []
                    for i, item in enumerate(raw_outcomes):
                        if isinstance(item, BaseException):
                            # Convert exception into a failed result
                            from loom.engine.runner import SubtaskResult
                            from loom.recovery.confidence import (
                                VerificationResult,
                            )

                            failed = SubtaskResult(
                                status="failed",
                                output="",
                                error=str(item),
                            )
                            no_verif = VerificationResult()
                            outcomes.append(
                                (batch[i], failed, no_verif),
                            )
                        else:
                            outcomes.append(item)

                # Process outcomes (retry / replan / approve)
                for subtask, result, verification in outcomes:
                    if result.status == "failed":
                        await self._handle_failure(
                            task, subtask, result, verification,
                            attempts_by_subtask,
                        )
                    else:
                        await self._handle_success(
                            task, subtask, result, verification,
                        )

            # 3. Completion
            result_task = self._finalize_task(task)

            # 4. Learn from execution (best-effort)
            await self._learn_from_task(result_task)

            return result_task

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.add_error("orchestrator", str(e))
            self._state.save(task)
            self._emit(TASK_FAILED, task.id, {"error": str(e)})
            await self._learn_from_task(task)
            return task

    # ------------------------------------------------------------------
    # Subtask dispatch
    # ------------------------------------------------------------------

    async def _dispatch_subtask(
        self,
        task: Task,
        subtask: Subtask,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
    ) -> tuple[Subtask, SubtaskResult, VerificationResult]:
        """Prepare and dispatch a subtask to the runner.

        Handles pre-dispatch bookkeeping (status, events, escalation)
        and returns (subtask, result, verification) for the orchestrator
        to process.
        """
        # Mark running and emit event (under lock for parallel safety)
        async with self._state_lock:
            subtask.status = SubtaskStatus.RUNNING
            self._state.save(task)
        self._emit(SUBTASK_STARTED, task.id, {"subtask_id": subtask.id})

        # Determine escalation tier
        attempts = attempts_by_subtask.get(subtask.id, [])
        escalated_tier = self._retry.get_escalation_tier(
            attempt=len(attempts),
            original_tier=subtask.model_tier,
        )
        retry_context = self._retry.build_retry_context(attempts)
        changelog = self._get_changelog(task)

        result, verification = await self._runner.run(
            task, subtask,
            model_tier=escalated_tier,
            retry_context=retry_context,
            changelog=changelog,
        )

        return subtask, result, verification

    # ------------------------------------------------------------------
    # Outcome handlers
    # ------------------------------------------------------------------

    async def _handle_failure(
        self,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
    ) -> None:
        """Process a failed subtask: record attempt, retry or replan."""
        attempt_list = attempts_by_subtask.setdefault(subtask.id, [])
        attempt_list.append(
            AttemptRecord(
                attempt=len(attempt_list) + 1,
                tier=self._retry.get_escalation_tier(
                    len(attempt_list), subtask.model_tier,
                ),
                feedback=verification.feedback if verification else None,
                error=result.summary,
            )
        )

        async with self._state_lock:
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

        if subtask.retry_count < subtask.max_retries:
            subtask.retry_count += 1
            async with self._state_lock:
                subtask.status = SubtaskStatus.PENDING
                task.update_subtask(
                    subtask.id,
                    status=SubtaskStatus.PENDING,
                    retry_count=subtask.retry_count,
                )
                self._state.save(task)

            self._emit(SUBTASK_RETRYING, task.id, {
                "subtask_id": subtask.id,
                "attempt": subtask.retry_count,
                "escalated_tier": self._retry.get_escalation_tier(
                    subtask.retry_count, subtask.model_tier,
                ),
                "feedback": verification.feedback if verification else None,
            })
        else:
            # All retries exhausted — trigger re-planning
            await self._replan_task(task)

    async def _handle_success(
        self,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
    ) -> None:
        """Process a successful subtask: update state, check approval."""
        summary = result.summary

        # Update state
        async with self._state_lock:
            subtask.status = SubtaskStatus.COMPLETED
            subtask.summary = summary
            task.update_subtask(subtask.id, status=SubtaskStatus.COMPLETED, summary=summary)

            # Update workspace_changes from changelog
            changelog = self._get_changelog(task)
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
            "duration": result.duration_seconds,
        })

        # Confidence scoring and approval check
        if verification:
            confidence = self._confidence.score(subtask, result, verification)
            decision = self._approval.check_approval(
                approval_mode=task.approval_mode,
                confidence=confidence.score,
                result=result,
                confidence_threshold=self._config.execution.auto_approve_confidence_threshold,
            )

            if decision in (
                ApprovalDecision.WAIT,
                ApprovalDecision.WAIT_WITH_TIMEOUT,
            ):
                timeout = 10 if decision == ApprovalDecision.WAIT_WITH_TIMEOUT else None
                async with self._state_lock:
                    task.status = TaskStatus.WAITING_APPROVAL
                    self._state.save(task)

                approved = await self._approval.request_approval(
                    ApprovalRequest(
                        task_id=task.id,
                        subtask_id=subtask.id,
                        reason=f"Confidence {confidence.band} ({confidence.score:.2f})",
                        proposed_action=result.summary,
                        risk_level=confidence.band,
                        details=confidence.components,
                        auto_approve_timeout=timeout,
                    )
                )

                async with self._state_lock:
                    task.status = TaskStatus.EXECUTING
                    self._state.save(task)

                if not approved:
                    async with self._state_lock:
                        subtask.status = SubtaskStatus.FAILED
                        task.update_subtask(
                            subtask.id,
                            status=SubtaskStatus.FAILED,
                            summary="Rejected by human reviewer",
                        )
                        self._state.save(task)

            elif decision == ApprovalDecision.ABORT:
                async with self._state_lock:
                    subtask.status = SubtaskStatus.FAILED
                    task.update_subtask(
                        subtask.id,
                        status=SubtaskStatus.FAILED,
                        summary="Aborted: confidence too low",
                    )
                    self._state.save(task)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def _plan_task(self, task: Task) -> Plan:
        """Invoke the planner model to decompose the task into subtasks."""
        workspace_listing = ""
        code_analysis = ""
        if task.workspace:
            workspace_path = Path(task.workspace)
            if workspace_path.exists():
                listing_result = await self._tools.execute(
                    "list_directory", {}, workspace=workspace_path
                )
                if listing_result.success:
                    workspace_listing = listing_result.output

                # Auto-analyze code structure for better planning
                code_analysis = await self._analyze_workspace(workspace_path)

        prompt = self._prompts.build_planner_prompt(
            task=task,
            workspace_listing=workspace_listing,
            code_analysis=code_analysis,
        )

        model = self._router.select(tier=2, role="planner")
        response = await model.complete([{"role": "user", "content": prompt}])

        return self._parse_plan(response, goal=task.goal)

    async def _analyze_workspace(self, workspace_path: Path) -> str:
        """Run code analysis on workspace files for better planning context.

        Returns a summary of code structure (classes, functions, imports)
        for key source files. Best-effort — returns empty string on failure.
        """
        try:
            from loom.tools.code_analysis import analyze_directory

            structures = analyze_directory(workspace_path, max_files=20)
            if not structures:
                return ""
            summaries = [s.to_summary() for s in structures]
            return "\n\n".join(summaries)
        except Exception:
            return ""

    async def _replan_task(self, task: Task) -> bool:
        """Re-plan the task after subtask failures.

        Returns True if re-planning succeeded and execution can continue.
        """
        self._emit(TASK_REPLANNING, task.id, {"reason": "subtask_failures"})

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
            new_plan = self._parse_plan(response, goal=task.goal)

            # Preserve completed subtask state
            completed_ids = {
                s.id for s in task.plan.subtasks
                if s.status == SubtaskStatus.COMPLETED
            }
            new_plan.version = task.plan.version + 1
            new_plan.last_replanned = datetime.now().isoformat()

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

    def _parse_plan(self, response: ModelResponse, goal: str = "") -> Plan:
        """Parse a plan from the model's JSON response."""
        validation = self._validator.validate_json_response(
            response, expected_keys=["subtasks"]
        )

        if not validation.valid or validation.parsed is None:
            return Plan(
                subtasks=[Subtask(
                    id="execute-goal",
                    description=goal or "Execute the task goal directly",
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_changelog(self, task: Task) -> ChangeLog | None:
        """Get or create a ChangeLog for the task's workspace."""
        if not task.workspace:
            return None
        workspace = Path(task.workspace)
        data_dir = self._state._data_dir / "tasks" / task.id
        data_dir.mkdir(parents=True, exist_ok=True)
        return ChangeLog(task_id=task.id, workspace=workspace, data_dir=data_dir)

    def _finalize_task(self, task: Task) -> Task:
        """Finalize task: set status, emit events."""
        completed, total = task.progress
        all_done = completed == total and total > 0

        if task.status == TaskStatus.CANCELLED:
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

    async def _learn_from_task(self, task: Task) -> None:
        """Run post-task learning extraction (best-effort)."""
        if self._learning is None:
            return
        try:
            await self._learning.learn_from_task(task)
        except Exception:
            pass

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
    callback_url: str = "",
    context: dict | None = None,
) -> Task:
    """Factory for creating new tasks with a generated ID."""
    return Task(
        id=uuid.uuid4().hex[:8],
        goal=goal,
        workspace=workspace,
        approval_mode=approval_mode,
        callback_url=callback_url,
        context=context or {},
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
