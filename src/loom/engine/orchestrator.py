"""Core orchestrator loop.

Drives task execution: plan -> execute subtasks -> verify -> complete.
The model never decides to "continue" — the harness does.

Subtask execution is delegated to SubtaskRunner.  Independent subtasks
(no unmet dependencies) are dispatched in parallel up to
``config.execution.max_parallel_subtasks``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from loom.auth.runtime import AuthResolutionError, build_run_auth_context
from loom.config import Config

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition
from loom.engine.runner import SubtaskResult, SubtaskResultStatus, SubtaskRunner, ToolCallRecord
from loom.engine.scheduler import Scheduler
from loom.engine.verification import VerificationGates, VerificationResult
from loom.events.bus import Event, EventBus
from loom.events.types import (
    MODEL_INVOCATION,
    REMEDIATION_ATTEMPT,
    REMEDIATION_EXPIRED,
    REMEDIATION_FAILED,
    REMEDIATION_QUEUED,
    REMEDIATION_RESOLVED,
    REMEDIATION_STARTED,
    REMEDIATION_TERMINAL,
    SUBTASK_COMPLETED,
    SUBTASK_FAILED,
    SUBTASK_OUTCOME_STALE,
    SUBTASK_RETRYING,
    SUBTASK_STARTED,
    TASK_BUDGET_EXHAUSTED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLAN_DEGRADED,
    TASK_PLAN_NORMALIZED,
    TASK_PLAN_READY,
    TASK_PLANNING,
    TASK_REPLAN_REJECTED,
    TASK_REPLANNING,
    TASK_RUN_ACQUIRED,
    TASK_STALLED,
    TASK_STALLED_RECOVERY_ATTEMPTED,
    TELEMETRY_RUN_SUMMARY,
    UNCONFIRMED_DATA_QUEUED,
)
from loom.learning.manager import LearningManager
from loom.models.base import ModelResponse
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalDecision, ApprovalManager, ApprovalRequest
from loom.recovery.confidence import ConfidenceScorer
from loom.recovery.errors import ErrorCategory, categorize_error
from loom.recovery.retry import AttemptRecord, RetryManager, RetryStrategy
from loom.state.evidence import merge_evidence_records
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
from loom.utils.concurrency import run_blocking_io

logger = logging.getLogger(__name__)

_REMEDIATION_TERMINAL_STATES = frozenset({"resolved", "failed", "expired"})
_VALID_CRITICAL_PATH_BEHAVIORS = frozenset({
    "block",
    "confirm_or_prune_then_queue",
    "queue_follow_up",
})
_FAILURE_RESOLUTION_METADATA_KEYS = (
    "remediation_required",
    "remediation_mode",
    "missing_targets",
    "unverified_claim_count",
    "verified_claim_count",
    "supporting_ratio",
    "coverage_sufficient",
    "coverage_insufficient_reason",
    "contradiction_detected",
    "contradiction_downgraded",
    "contradicted_reason_code",
    "parser_stage",
    "issues",
)

# Re-export dataclasses that existing code imports from here
__all__ = [
    "Orchestrator",
    "SubtaskResult",
    "ToolCallRecord",
    "create_task",
]


class _RunBudget:
    """Tracks task-level resource usage and limit enforcement."""

    def __init__(self, config: Config):
        execution = getattr(config, "execution", None)
        self.enabled = bool(getattr(execution, "enable_global_run_budget", False))
        self.wall_clock_seconds_limit = int(
            max(0, getattr(execution, "max_task_wall_clock_seconds", 0) or 0),
        )
        self.total_tokens_limit = int(
            max(0, getattr(execution, "max_task_total_tokens", 0) or 0),
        )
        self.model_invocations_limit = int(
            max(0, getattr(execution, "max_task_model_invocations", 0) or 0),
        )
        self.tool_calls_limit = int(
            max(0, getattr(execution, "max_task_tool_calls", 0) or 0),
        )
        self.mutating_tool_calls_limit = int(
            max(0, getattr(execution, "max_task_mutating_tool_calls", 0) or 0),
        )
        self.replans_limit = int(
            max(0, getattr(execution, "max_task_replans", 0) or 0),
        )
        self.remediation_attempts_limit = int(
            max(0, getattr(execution, "max_task_remediation_attempts", 0) or 0),
        )

        self.started_monotonic = time.monotonic()
        self.total_tokens = 0
        self.model_invocations = 0
        self.tool_calls = 0
        self.mutating_tool_calls = 0
        self.replans = 0
        self.remediation_attempts = 0

    def observe_result(self, result: SubtaskResult) -> None:
        self.total_tokens += max(0, int(getattr(result, "tokens_used", 0) or 0))
        counters = getattr(result, "telemetry_counters", None)
        if isinstance(counters, dict):
            self.model_invocations += max(0, int(counters.get("model_invocations", 0) or 0))
            self.tool_calls += max(0, int(counters.get("tool_calls", 0) or 0))
            self.mutating_tool_calls += max(
                0,
                int(counters.get("mutating_tool_calls", 0) or 0),
            )

    def observe_replan(self) -> None:
        self.replans += 1

    def observe_remediation_attempt(self) -> None:
        self.remediation_attempts += 1

    def snapshot(self) -> dict[str, int | float]:
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        return {
            "elapsed_seconds": round(elapsed, 3),
            "total_tokens": int(self.total_tokens),
            "model_invocations": int(self.model_invocations),
            "tool_calls": int(self.tool_calls),
            "mutating_tool_calls": int(self.mutating_tool_calls),
            "replans": int(self.replans),
            "remediation_attempts": int(self.remediation_attempts),
        }

    def exceeded(self) -> tuple[bool, str, int | float, int]:
        if not self.enabled:
            return False, "", 0, 0
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        checks: tuple[tuple[str, float, int], ...] = (
            ("max_task_wall_clock_seconds", elapsed, self.wall_clock_seconds_limit),
            ("max_task_total_tokens", float(self.total_tokens), self.total_tokens_limit),
            (
                "max_task_model_invocations",
                float(self.model_invocations),
                self.model_invocations_limit,
            ),
            ("max_task_tool_calls", float(self.tool_calls), self.tool_calls_limit),
            (
                "max_task_mutating_tool_calls",
                float(self.mutating_tool_calls),
                self.mutating_tool_calls_limit,
            ),
            ("max_task_replans", float(self.replans), self.replans_limit),
            (
                "max_task_remediation_attempts",
                float(self.remediation_attempts),
                self.remediation_attempts_limit,
            ),
        )
        for key, current, limit in checks:
            if limit > 0 and current > float(limit):
                return True, key, current, limit
        return False, "", 0, 0


class Orchestrator:
    """Core orchestrator loop.

    Responsibilities:
    - Task lifecycle (planning, execution, finalization)
    - Subtask scheduling and parallel dispatch
    - Retry / escalation / re-planning decisions
    - Approval gating (confidence-based)
    - Event emission and state persistence
    - Post-task learning

    When a process definition is provided, domain intelligence is injected
    into all prompts (persona, phases, tool guidance, verification rules,
    memory guidance) without changing the engine's control flow.

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
        process: ProcessDefinition | None = None,
    ):
        self._router = model_router
        self._tools = tool_registry
        self._memory = memory_manager
        self._prompts = prompt_assembler
        self._state = state_manager
        self._events = event_bus
        self._config = config
        self._learning = learning_manager
        self._process = process
        self._scheduler = Scheduler()
        self._validator = ResponseValidator()
        self._verification = VerificationGates(
            model_router=model_router,
            prompt_assembler=prompt_assembler,
            config=config.verification,
            limits=getattr(getattr(config, "limits", None), "verifier", None),
            compactor_limits=getattr(getattr(config, "limits", None), "compactor", None),
            evidence_context_text_max_chars=int(
                getattr(
                    getattr(config, "limits", None),
                    "evidence_context_text_max_chars",
                    4000,
                ),
            ),
            process=process,
            event_bus=event_bus,
        )
        self._confidence = ConfidenceScorer()
        self._approval = approval_manager or ApprovalManager(event_bus)
        self._retry = RetryManager(
            max_retries=config.execution.max_subtask_retries,
        )
        self._state_lock = asyncio.Lock()
        self._changelog_cache: dict[str, ChangeLog] = {}
        self._telemetry_rollup: dict[str, int] = self._new_telemetry_rollup()
        self._run_budget = _RunBudget(config)
        self._active_run_id = ""

        # Inject process into prompt assembler
        if process is not None:
            self._prompts.process = process

        # Apply process tool policy
        if process:
            excluded_tools = list(getattr(process.tools, "excluded", []) or [])
            for tool_name in excluded_tools:
                self._tools.exclude(tool_name)

            required_tools = list(getattr(process.tools, "required", []) or [])
            if required_tools:
                available = set(self._tools.list_tools())
                missing = sorted(
                    tool_name
                    for tool_name in required_tools
                    if tool_name not in available
                )
                if missing:
                    joined = ", ".join(missing)
                    raise ValueError(
                        f"Process '{process.name}' requires missing tool(s): {joined}"
                    )

        # Runner handles the inner subtask execution
        self._runner = SubtaskRunner(
            model_router=model_router,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            prompt_assembler=prompt_assembler,
            state_manager=state_manager,
            verification=self._verification,
            config=config,
            event_bus=event_bus,
        )

    async def execute_task(
        self,
        task: Task,
        *,
        reuse_existing_plan: bool = False,
    ) -> Task:
        """Main entry point. Drives the full task lifecycle."""
        try:
            self._telemetry_rollup = self._new_telemetry_rollup()
            self._run_budget = _RunBudget(self._config)
            run_id = self._initialize_task_run_id(task)
            self._emit(TASK_RUN_ACQUIRED, task.id, {
                "run_id": run_id,
            })
            if bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
                await self._hydrate_remediation_queue_from_db(task)
            plan: Plan
            if reuse_existing_plan and task.plan and task.plan.subtasks:
                plan = task.plan
                plan = self._prepare_plan_for_execution(
                    task=task,
                    plan=plan,
                    context="reused_plan",
                )
                task.plan = plan
                task.status = TaskStatus.EXECUTING
                self._state.save(task)
                self._emit(TASK_PLAN_READY, task.id, {
                    "subtask_count": len(plan.subtasks),
                    "subtask_ids": [s.id for s in plan.subtasks],
                    "reused": True,
                    "run_id": run_id,
                })
            else:
                # 1. Planning phase
                task.status = TaskStatus.PLANNING
                self._emit(TASK_PLANNING, task.id, {
                    "goal": task.goal,
                    "run_id": run_id,
                })

                plan = await self._plan_task_with_validation(task)
                task.plan = plan
                task.status = TaskStatus.EXECUTING
                self._state.save(task)
                self._emit(TASK_PLAN_READY, task.id, {
                    "subtask_count": len(plan.subtasks),
                    "subtask_ids": [s.id for s in plan.subtasks],
                    "run_id": run_id,
                })

            # 2. Execution loop — parallel dispatch of independent subtasks
            self._emit(TASK_EXECUTING, task.id, {"run_id": run_id})
            iteration = 0
            max_iterations = self._config.execution.max_loop_iterations
            max_parallel = self._config.execution.max_parallel_subtasks
            attempts_by_subtask: dict[str, list[AttemptRecord]] = {}
            pending_replan: dict[str, str | None] | None = None
            stall_recovery_attempts = 0
            max_stall_recovery_attempts = 2

            while self._scheduler.has_pending(task.plan) and iteration < max_iterations:
                if await self._enforce_global_budget(task):
                    break
                if task.status == TaskStatus.CANCELLED:
                    break

                # Get all runnable subtasks (dependencies met)
                runnable = self._scheduler.runnable_subtasks(task.plan)
                if not runnable:
                    blocked_subtasks = self._blocked_pending_subtasks(task.plan)
                    self._emit(TASK_STALLED, task.id, {
                        "pending_subtasks": [
                            item["subtask_id"] for item in blocked_subtasks
                        ],
                        "blocked_subtasks": blocked_subtasks,
                        "attempt": stall_recovery_attempts + 1,
                    })
                    recovered = False
                    if (
                        task.status == TaskStatus.EXECUTING
                        and stall_recovery_attempts < max_stall_recovery_attempts
                    ):
                        stall_recovery_attempts += 1
                        recovered = await self._attempt_stalled_recovery(
                            task=task,
                            blocked_subtasks=blocked_subtasks,
                            attempt=stall_recovery_attempts,
                        )
                    if recovered:
                        continue
                    task.metadata["blocked_subtasks"] = blocked_subtasks
                    break
                stall_recovery_attempts = 0
                task.metadata.pop("blocked_subtasks", None)

                # Cap to max_parallel_subtasks
                batch = runnable[:max_parallel]
                iteration += 1
                batch_plan_version = task.plan.version

                # Dispatch batch
                if len(batch) == 1:
                    # Single subtask — no gather overhead
                    try:
                        outcomes = [await self._dispatch_subtask(
                            task, batch[0], attempts_by_subtask,
                        )]
                    except BaseException as item:
                        outcomes = [
                            self._build_subtask_exception_outcome(batch[0], item),
                        ]
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
                            outcomes.append(
                                self._build_subtask_exception_outcome(
                                    batch[i],
                                    item,
                                ),
                            )
                        else:
                            outcomes.append(item)

                # Process outcomes (retry / replan / approve).
                # Replanning is deferred until the whole batch is processed.
                for subtask, result, verification in outcomes:
                    self._accumulate_subtask_telemetry(result)
                    self._run_budget.observe_result(result)
                    if batch_plan_version != task.plan.version:
                        self._record_stale_outcome(
                            task=task,
                            subtask=subtask,
                            outcome_plan_version=batch_plan_version,
                        )
                        continue
                    if result.status == "failed":
                        replan_request = await self._handle_failure(
                            task, subtask, result, verification,
                            attempts_by_subtask,
                        )
                        if replan_request is not None and pending_replan is None:
                            pending_replan = replan_request
                    else:
                        await self._handle_success(
                            task, subtask, result, verification,
                        )

                if pending_replan is not None and task.status == TaskStatus.EXECUTING:
                    self._run_budget.observe_replan()
                    await self._replan_task(
                        task,
                        reason=str(pending_replan.get("reason", "subtask_failures")),
                        failed_subtask_id=str(
                            pending_replan.get("failed_subtask_id", ""),
                        ),
                        verification_feedback=pending_replan.get(
                            "verification_feedback",
                        ),
                    )
                    pending_replan = None
                elif task.status != TaskStatus.EXECUTING:
                    pending_replan = None

                # Opportunistically execute queued remediation work between
                # scheduling batches so non-blocking follow-ups can converge.
                await self._process_remediation_queue(
                    task=task,
                    attempts_by_subtask=attempts_by_subtask,
                    finalizing=False,
                )
                if bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
                    await self._sync_remediation_queue_to_db(task)

            # 3. Completion
            await self._process_remediation_queue(
                task=task,
                attempts_by_subtask=attempts_by_subtask,
                finalizing=True,
            )
            if bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
                await self._sync_remediation_queue_to_db(task)
            result_task = self._finalize_task(task)
            self._export_evidence_ledger_csv(result_task)

            # 4. Learn from execution (best-effort)
            await self._learn_from_task(result_task)

            return result_task

        except Exception as e:
            logger.exception("Fatal error in task %s", task.id)
            task.status = TaskStatus.FAILED
            task.add_error("orchestrator", f"{type(e).__name__}: {e}")
            try:
                self._state.save(task)
            except Exception as save_err:
                logger.error("Failed to save after fatal: %s", save_err)
            self._emit(TASK_FAILED, task.id, {
                "error": str(e),
                "error_type": type(e).__name__,
            })
            self._export_evidence_ledger_csv(task)
            await self._learn_from_task(task)
            return task

    def _build_subtask_exception_outcome(
        self,
        subtask: Subtask,
        error: BaseException,
    ) -> tuple[Subtask, SubtaskResult, VerificationResult]:
        """Convert a dispatch exception into a normal failed subtask outcome."""
        logger.error(
            "Subtask %s raised exception: %s",
            subtask.id,
            error,
            exc_info=error,
        )
        failed = SubtaskResult(
            status=SubtaskResultStatus.FAILED,
            summary=f"{type(error).__name__}: {error}",
        )
        no_verif = VerificationResult(
            tier=0,
            passed=False,
            feedback=f"Exception during execution: {error}",
        )
        return subtask, failed, no_verif

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
        retry_strategy = (
            attempts[-1].retry_strategy
            if attempts
            else RetryStrategy.GENERIC
        )
        prior_successful_tool_calls: list[ToolCallRecord] = []
        prior_evidence_records = self._evidence_for_subtask(task.id, subtask.id)
        for attempt in attempts:
            raw_calls = getattr(attempt, "successful_tool_calls", [])
            if isinstance(raw_calls, list):
                for call in raw_calls:
                    if isinstance(call, ToolCallRecord):
                        prior_successful_tool_calls.append(call)
            raw_evidence = getattr(attempt, "evidence_records", [])
            if isinstance(raw_evidence, list):
                prior_evidence_records = merge_evidence_records(
                    prior_evidence_records,
                    [item for item in raw_evidence if isinstance(item, dict)],
                )
        escalated_tier = self._retry.get_escalation_tier(
            attempt=len(attempts),
            original_tier=subtask.model_tier,
        )
        expected_deliverables = self._expected_deliverables_for_subtask(subtask)
        retry_context = self._retry.build_retry_context(attempts)
        retry_context = self._augment_retry_context_for_outputs(
            subtask=subtask,
            attempts=attempts,
            strategy=retry_strategy,
            expected_deliverables=expected_deliverables,
            base_context=retry_context,
        )
        changelog = self._get_changelog(task)

        result, verification = await self._runner.run(
            task, subtask,
            model_tier=escalated_tier,
            retry_context=retry_context,
            changelog=changelog,
            prior_successful_tool_calls=prior_successful_tool_calls,
            prior_evidence_records=prior_evidence_records,
            expected_deliverables=expected_deliverables,
            enforce_deliverable_paths=bool(attempts and expected_deliverables),
            edit_existing_only=bool(attempts and expected_deliverables),
            retry_strategy=retry_strategy.value,
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
    ) -> dict[str, str | None] | None:
        """Process a failed subtask: record attempt, retry or replan."""
        self._persist_subtask_evidence(task.id, subtask.id, result.evidence_records)
        attempt_list = attempts_by_subtask.setdefault(subtask.id, [])
        strategy, missing_targets = self._retry.classify_failure(
            verification_feedback=verification.feedback,
            execution_error=result.summary,
            verification=verification,
        )
        combined_error = " | ".join(
            part for part in [verification.feedback, result.summary] if part
        )
        attempt_record = AttemptRecord(
            attempt=len(attempt_list) + 1,
            tier=self._retry.get_escalation_tier(
                len(attempt_list), subtask.model_tier,
            ),
            feedback=verification.feedback if verification else None,
            error=combined_error or None,
            successful_tool_calls=[
                call for call in result.tool_calls
                if getattr(getattr(call, "result", None), "success", False)
            ],
            evidence_records=[
                item for item in result.evidence_records
                if isinstance(item, dict)
            ],
            retry_strategy=strategy,
            missing_targets=missing_targets,
        )
        attempt_list.append(attempt_record)
        await self._persist_subtask_attempt_record(
            task=task,
            subtask_id=subtask.id,
            attempt_record=attempt_record,
            verification=verification,
        )

        # Parse failures in verifier output should retry verification only
        # instead of re-running full subtask execution.
        if (
            strategy == RetryStrategy.VERIFIER_PARSE
            and subtask.retry_count < subtask.max_retries
        ):
            verification_retry = await self._retry_verification_only(
                task=task,
                subtask=subtask,
                result=result,
                attempts=attempt_list,
            )
            if verification_retry.passed:
                await self._handle_success(task, subtask, result, verification_retry)
                return None
            verification = verification_retry
            attempt_record.feedback = verification_retry.feedback or attempt_record.feedback
            if verification_retry.feedback:
                attempt_record.error = " | ".join(
                    part for part in [attempt_record.error, verification_retry.feedback]
                    if part
                )

        resolution_plan = await self._plan_failure_resolution(
            task=task,
            subtask=subtask,
            result=result,
            verification=verification,
            strategy=strategy,
            missing_targets=missing_targets,
            prior_attempts=attempt_list[:-1],
        )
        if resolution_plan:
            attempt_record.resolution_plan = resolution_plan

        critical_path_behavior = self._critical_path_behavior()
        hard_invariant_failure = self._is_hard_invariant_failure(verification)
        if (
            strategy == RetryStrategy.UNCONFIRMED_DATA
            and not hard_invariant_failure
            and (
                not subtask.is_critical_path
                or critical_path_behavior == "queue_follow_up"
            )
        ):
            await self._queue_remediation_work_item(
                task=task,
                subtask=subtask,
                verification=verification,
                strategy=strategy,
                blocking=False,
            )
            if subtask.is_critical_path:
                note = (
                    "Remediation queued for follow-up "
                    "(critical path policy: queue_follow_up)."
                )
                default_reason = "unconfirmed_critical_queue_follow_up"
            else:
                note = "Remediation queued for follow-up (non-critical path)."
                default_reason = "unconfirmed_noncritical"
            self._apply_unconfirmed_follow_up_success(
                result=result,
                verification=verification,
                note=note,
                default_reason_code=default_reason,
            )
            await self._handle_success(task, subtask, result, verification)
            return None

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
            "verification_outcome": verification.outcome,
            "reason_code": verification.reason_code,
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
                "retry_strategy": strategy.value,
                "resolution_plan_generated": bool(resolution_plan),
            })
        else:
            # All retries exhausted.
            # Critical-path failures abort the remaining plan.
            if subtask.is_critical_path:
                if (
                    strategy == RetryStrategy.UNCONFIRMED_DATA
                    and not hard_invariant_failure
                ):
                    if critical_path_behavior == "confirm_or_prune_then_queue":
                        remediation_recovered, _ = (
                            await self._run_confirm_or_prune_remediation(
                                task=task,
                                subtask=subtask,
                                attempts=attempt_list,
                            )
                        )
                        if remediation_recovered:
                            return None
                        await self._queue_remediation_work_item(
                            task=task,
                            subtask=subtask,
                            verification=verification,
                            strategy=strategy,
                            blocking=True,
                        )
                        self._apply_unconfirmed_follow_up_success(
                            result=result,
                            verification=verification,
                            note=(
                                "Critical-path remediation queued as blocking "
                                "follow-up (policy: confirm_or_prune_then_queue)."
                            ),
                            default_reason_code="unconfirmed_critical_path",
                        )
                        await self._handle_success(task, subtask, result, verification)
                        return None
                    if critical_path_behavior == "queue_follow_up":
                        await self._queue_remediation_work_item(
                            task=task,
                            subtask=subtask,
                            verification=verification,
                            strategy=strategy,
                            blocking=False,
                        )
                        self._apply_unconfirmed_follow_up_success(
                            result=result,
                            verification=verification,
                            note=(
                                "Critical-path remediation queued as follow-up "
                                "(policy: queue_follow_up)."
                            ),
                            default_reason_code="unconfirmed_critical_path",
                        )
                        await self._handle_success(task, subtask, result, verification)
                        return None
                await self._abort_on_critical_path_failure(
                    task, subtask, verification,
                )
                return None

            # Non-critical failures request re-planning at batch boundary.
            verification_feedback = verification.feedback
            if resolution_plan:
                details = (
                    f"{verification_feedback}\n\n"
                    "MODEL-PLANNED RESOLUTION:\n"
                    f"{resolution_plan}"
                )
                verification_feedback = details.strip()
            return {
                "reason": self._build_replan_reason(subtask, verification),
                "failed_subtask_id": subtask.id,
                "verification_feedback": verification_feedback,
            }

        return None

    async def _abort_on_critical_path_failure(
        self,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
    ) -> None:
        """Abort remaining work when a critical-path subtask exhausts retries."""
        block_summary = (
            f"Skipped: blocked by critical-path failure in {subtask.id}"
        )
        async with self._state_lock:
            for candidate in task.plan.subtasks:
                if candidate.status == SubtaskStatus.PENDING:
                    candidate.status = SubtaskStatus.SKIPPED
                    candidate.summary = block_summary
                    task.update_subtask(
                        candidate.id,
                        status=SubtaskStatus.SKIPPED,
                        summary=candidate.summary,
                    )

            task.status = TaskStatus.FAILED
            task.add_error(
                subtask.id,
                (
                    "Critical-path subtask failed after retries: "
                    + (verification.feedback or subtask.summary or "unknown failure")
                ),
            )
            self._state.save(task)

    def _phase_mode(self) -> str:
        process = self._process
        if process is None:
            return "guided"
        value = str(getattr(process, "phase_mode", "guided") or "").strip().lower()
        if value in {"strict", "guided", "suggestive"}:
            return value
        return "guided"

    def _topology_retry_attempts(self) -> int:
        """Bounded retry budget for topology-invalid planner outputs."""
        return 2 if self._phase_mode() == "strict" else 1

    def _planner_degraded_mode(self) -> str:
        raw = str(
            getattr(self._config.execution, "planner_degraded_mode", "allow"),
        ).strip().lower()
        if raw in {"allow", "require_approval", "deny"}:
            return raw
        return "allow"

    async def _build_planner_degraded_plan(
        self,
        *,
        task: Task,
        reason_code: str,
        detail: str,
    ) -> Plan:
        mode = self._planner_degraded_mode()
        if mode == "deny":
            raise ValueError(
                f"Planner degraded mode denied ({reason_code}): {detail}",
            )
        if mode == "require_approval":
            async with self._state_lock:
                task.status = TaskStatus.WAITING_APPROVAL
                self._state.save(task)
            approved = await self._approval.request_approval(
                ApprovalRequest(
                    task_id=task.id,
                    subtask_id="planning",
                    reason=f"Planner degraded: {reason_code}",
                    proposed_action="Use fallback execute-goal plan",
                    risk_level="high",
                    details={
                        "reason_code": reason_code,
                        "detail": detail,
                    },
                    auto_approve_timeout=None,
                ),
            )
            async with self._state_lock:
                task.status = TaskStatus.PLANNING
                self._state.save(task)
            if not approved:
                raise ValueError(
                    f"Planner degraded fallback not approved ({reason_code})",
                )

        fallback = Plan(
            subtasks=[Subtask(
                id="execute-goal",
                description=task.goal or "Execute the task goal directly",
                model_tier=2,
                max_retries=self._config.execution.max_subtask_retries,
            )],
            version=1,
        )
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["planner_degraded"] = True
        metadata["planner_degraded_reason"] = reason_code
        metadata["planner_degraded_detail"] = detail
        task.metadata = metadata
        self._emit(TASK_PLAN_DEGRADED, task.id, {
            "run_id": self._task_run_id(task),
            "reason_code": reason_code,
            "detail": detail,
            "policy_mode": mode,
            "fallback_subtasks": ["execute-goal"],
        })
        return self._apply_process_phase_mode(fallback)

    async def _plan_task_with_validation(self, task: Task) -> Plan:
        """Plan task with bounded retries when topology validation fails."""
        max_attempts = self._topology_retry_attempts()
        planner_feedback = ""
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            plan = await self._plan_task(task, planner_feedback=planner_feedback)
            try:
                return self._prepare_plan_for_execution(
                    task=task,
                    plan=plan,
                    context="planner",
                )
            except ValueError as e:
                last_error = str(e).strip() or "unknown topology validation error"
                task.add_decision(
                    "Rejected planner output due to invalid topology "
                    f"(attempt {attempt}/{max_attempts}): {last_error}",
                )
                self._state.save(task)
                if attempt >= max_attempts:
                    break
                planner_feedback = (
                    "Previous planner output was rejected for invalid topology.\n"
                    f"Validation error: {last_error}\n"
                    "Return corrected JSON that satisfies all dependency and "
                    "synthesis-topology constraints."
                )

        raise ValueError(last_error or "Planner output failed topology validation.")

    def _prepare_plan_for_execution(
        self,
        *,
        task: Task,
        plan: Plan,
        context: str,
    ) -> Plan:
        """Normalize and validate planner output before execution."""
        working = deepcopy(plan)
        normalized_plan, normalized_subtasks = self._normalize_non_terminal_synthesis(
            working,
        )
        if normalized_subtasks:
            if self._phase_mode() == "strict":
                details = ", ".join(
                    str(item.get("subtask_id", "")).strip()
                    for item in normalized_subtasks
                    if str(item.get("subtask_id", "")).strip()
                )
                raise ValueError(
                    "Strict phase mode does not allow non-terminal synthesis subtasks "
                    f"in {context}: {details or 'unknown'}",
                )
            working = normalized_plan
            self._emit(TASK_PLAN_NORMALIZED, task.id, {
                "context": context,
                "normalized_subtasks": normalized_subtasks,
                "plan_version": int(working.version),
            })

        topology_issues = self._plan_topology_issues(working)
        if topology_issues:
            raise ValueError(
                f"Invalid plan topology from {context}: " + "; ".join(topology_issues),
            )
        return working

    @staticmethod
    def _normalize_non_terminal_synthesis(
        plan: Plan,
    ) -> tuple[Plan, list[dict[str, object]]]:
        """Demote synthesis flags on non-terminal subtasks."""
        normalized = deepcopy(plan)
        dependents: dict[str, list[str]] = {}
        for subtask in normalized.subtasks:
            for dep_id in subtask.depends_on:
                dependents.setdefault(dep_id, []).append(subtask.id)

        changes: list[dict[str, object]] = []
        for subtask in normalized.subtasks:
            if not subtask.is_synthesis:
                continue
            child_ids = sorted({
                child
                for child in dependents.get(subtask.id, [])
                if child != subtask.id
            })
            if not child_ids:
                continue
            subtask.is_synthesis = False
            changes.append({
                "subtask_id": subtask.id,
                "reason": "non_terminal_synthesis",
                "dependents": child_ids,
            })
        return normalized, changes

    @classmethod
    def _plan_topology_issues(cls, plan: Plan) -> list[str]:
        """Return deterministic topology issues for a plan graph."""
        issues: list[str] = []
        ids = [subtask.id for subtask in plan.subtasks]
        id_set = set(ids)

        if len(ids) != len(id_set):
            duplicates: list[str] = []
            seen: set[str] = set()
            for subtask_id in ids:
                if subtask_id in seen and subtask_id not in duplicates:
                    duplicates.append(subtask_id)
                seen.add(subtask_id)
            duplicates.sort()
            issues.append("duplicate subtask IDs: " + ", ".join(duplicates))

        unresolved_deps: list[str] = []
        for subtask in plan.subtasks:
            bad = sorted(dep for dep in subtask.depends_on if dep not in id_set)
            if bad:
                unresolved_deps.append(f"{subtask.id} -> {', '.join(bad)}")
        if unresolved_deps:
            issues.append("unresolved dependencies: " + "; ".join(unresolved_deps))

        adjacency: dict[str, list[str]] = {}
        for subtask in plan.subtasks:
            adjacency[subtask.id] = [
                dep for dep in subtask.depends_on
                if dep in id_set
            ]
        cycle = cls._detect_dependency_cycle(adjacency)
        if cycle:
            issues.append("dependency cycle detected: " + " -> ".join(cycle))

        dependents: dict[str, list[str]] = {}
        for subtask in plan.subtasks:
            for dep in subtask.depends_on:
                if dep in id_set:
                    dependents.setdefault(dep, []).append(subtask.id)

        synthesis_ids = {
            subtask.id for subtask in plan.subtasks if bool(subtask.is_synthesis)
        }
        for synthesis_id in sorted(synthesis_ids):
            child_ids = sorted({
                child
                for child in dependents.get(synthesis_id, [])
                if child != synthesis_id
            })
            if child_ids:
                issues.append(
                    "synthesis subtask has dependents: "
                    + f"{synthesis_id} -> {', '.join(child_ids)}",
                )

        for subtask in plan.subtasks:
            if subtask.is_synthesis:
                continue
            bad = sorted(dep for dep in subtask.depends_on if dep in synthesis_ids)
            if bad:
                issues.append(
                    "non-synthesis subtask depends on synthesis subtask: "
                    + f"{subtask.id} -> {', '.join(bad)}",
                )

        return issues

    @staticmethod
    def _detect_dependency_cycle(
        adjacency: dict[str, list[str]],
    ) -> list[str] | None:
        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []

        def _walk(node: str) -> list[str] | None:
            visiting.add(node)
            stack.append(node)
            for neighbor in adjacency.get(node, []):
                if neighbor in visited:
                    continue
                if neighbor in visiting:
                    start = stack.index(neighbor)
                    return stack[start:] + [neighbor]
                cycle = _walk(neighbor)
                if cycle:
                    return cycle
            stack.pop()
            visiting.remove(node)
            visited.add(node)
            return None

        for node in adjacency:
            if node in visited:
                continue
            cycle = _walk(node)
            if cycle:
                return cycle
        return None

    @staticmethod
    def _format_blocked_subtasks_feedback(blocked_subtasks: list[dict[str, object]]) -> str:
        if not blocked_subtasks:
            return "No blocked subtasks were identified."
        lines = ["Blocked subtasks:"]
        for item in blocked_subtasks:
            subtask_id = str(item.get("subtask_id", "")).strip() or "unknown"
            raw_reasons = item.get("reasons", [])
            if isinstance(raw_reasons, list):
                reasons = [
                    str(reason).strip()
                    for reason in raw_reasons
                    if str(reason).strip()
                ]
            else:
                reason_text = str(raw_reasons).strip()
                reasons = [reason_text] if reason_text else []
            lines.append(f"- {subtask_id}: {', '.join(reasons) if reasons else 'blocked'}")
        return "\n".join(lines)

    def _blocked_pending_subtasks(self, plan: Plan) -> list[dict[str, object]]:
        """Return blocked reasons for pending/running subtasks."""
        by_id = {subtask.id: subtask for subtask in plan.subtasks}
        blocked: list[dict[str, object]] = []

        for subtask in plan.subtasks:
            if subtask.status not in {
                SubtaskStatus.PENDING,
                SubtaskStatus.RUNNING,
            }:
                continue
            reasons: list[str] = []
            if subtask.status == SubtaskStatus.RUNNING:
                reasons.append("status=running")

            for dep_id in subtask.depends_on:
                dep = by_id.get(dep_id)
                if dep is None:
                    reasons.append(f"dependency_missing:{dep_id}")
                elif dep.status != SubtaskStatus.COMPLETED:
                    reasons.append(f"dependency_unmet:{dep_id}={dep.status.value}")

            if (
                not reasons
                and subtask.status == SubtaskStatus.PENDING
                and Scheduler._is_terminal_synthesis(plan, subtask)
            ):
                waiting: list[str] = []
                for candidate in plan.subtasks:
                    if candidate.id == subtask.id or candidate.is_synthesis:
                        continue
                    if candidate.status != SubtaskStatus.COMPLETED:
                        waiting.append(f"{candidate.id}={candidate.status.value}")
                if waiting:
                    reasons.append(
                        "synthesis_waiting_on_non_synthesis:" + ", ".join(waiting),
                    )

            if not reasons:
                reasons.append("not_runnable_unknown")

            blocked.append({
                "subtask_id": subtask.id,
                "reasons": reasons,
            })

        return blocked

    async def _attempt_stalled_recovery(
        self,
        *,
        task: Task,
        blocked_subtasks: list[dict[str, object]],
        attempt: int,
    ) -> bool:
        """Try bounded recovery when pending work is blocked."""
        normalized_plan, normalized_subtasks = self._normalize_non_terminal_synthesis(
            task.plan,
        )
        if normalized_subtasks and self._phase_mode() != "strict":
            task.plan = normalized_plan
            task.metadata.pop("blocked_subtasks", None)
            task.add_decision(
                "Recovered from scheduler stall by demoting non-terminal synthesis subtasks.",
            )
            async with self._state_lock:
                self._state.save(task)
            self._emit(TASK_PLAN_NORMALIZED, task.id, {
                "context": "stalled_recovery",
                "normalized_subtasks": normalized_subtasks,
                "plan_version": int(task.plan.version),
            })
            self._emit(TASK_STALLED_RECOVERY_ATTEMPTED, task.id, {
                "attempt": attempt,
                "recovery_mode": "normalize",
                "recovery_success": True,
                "normalized_subtasks": normalized_subtasks,
            })
            return True

        if normalized_subtasks and self._phase_mode() == "strict":
            self._emit(TASK_STALLED_RECOVERY_ATTEMPTED, task.id, {
                "attempt": attempt,
                "recovery_mode": "normalize",
                "recovery_success": False,
                "reason": "strict_phase_mode_disallows_normalization",
                "normalized_subtasks": normalized_subtasks,
            })

        feedback = self._format_blocked_subtasks_feedback(blocked_subtasks)
        self._run_budget.observe_replan()
        replanned = await self._replan_task(
            task,
            reason="scheduler_deadlock",
            failed_subtask_id="",
            verification_feedback=feedback,
        )
        self._emit(TASK_STALLED_RECOVERY_ATTEMPTED, task.id, {
            "attempt": attempt,
            "recovery_mode": "replan",
            "recovery_success": bool(replanned),
        })
        return bool(replanned)

    def _critical_path_behavior(self) -> str:
        process = self._process
        if process is None:
            return "block"
        getter = getattr(process, "remediation_critical_path_behavior", None)
        if callable(getter):
            value = str(getter() or "").strip().lower()
        else:
            remediation = getattr(process, "verification_remediation", None)
            value = str(
                getattr(remediation, "critical_path_behavior", "") or "",
            ).strip().lower()
        if value in _VALID_CRITICAL_PATH_BEHAVIORS:
            return value
        return "block"

    @staticmethod
    def _is_hard_invariant_failure(verification: VerificationResult | None) -> bool:
        if verification is None:
            return False
        reason_code = str(verification.reason_code or "").strip().lower()
        severity = str(verification.severity_class or "").strip().lower()
        return (
            reason_code == "hard_invariant_failed"
            or severity == "hard_invariant"
        )

    @staticmethod
    def _normalize_missing_targets(raw: object) -> list[str]:
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            if "," in text:
                candidates = [item.strip() for item in text.split(",")]
            else:
                candidates = [text]
            return [item for item in candidates if item]
        if isinstance(raw, list):
            deduped: list[str] = []
            for item in raw:
                text = str(item or "").strip()
                if text and text not in deduped:
                    deduped.append(text)
            return deduped
        return []

    @staticmethod
    def _to_int_or_none(value: object) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            try:
                return int(float(text))
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_ratio_or_none(value: object) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            numeric = float(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                numeric = float(text.rstrip("%"))
            except ValueError:
                return None
            if text.endswith("%"):
                numeric /= 100.0
        else:
            return None
        if numeric > 1.0 and numeric <= 100.0:
            numeric /= 100.0
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _to_float_or_none(value: object) -> float | None:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    @classmethod
    def _extract_unconfirmed_metadata(
        cls,
        verification: VerificationResult,
    ) -> dict[str, object]:
        metadata = verification.metadata if isinstance(verification.metadata, dict) else {}
        extracted: dict[str, object] = {}

        remediation_required = metadata.get("remediation_required", False)
        if isinstance(remediation_required, bool):
            extracted["remediation_required"] = remediation_required
        else:
            text = str(remediation_required or "").strip().lower()
            extracted["remediation_required"] = text in {"1", "true", "yes", "on"}

        remediation_mode = str(metadata.get("remediation_mode", "") or "").strip().lower()
        if remediation_mode:
            extracted["remediation_mode"] = remediation_mode

        missing_targets = cls._normalize_missing_targets(metadata.get("missing_targets"))
        if missing_targets:
            extracted["missing_targets"] = missing_targets

        unverified_claim_count = cls._to_int_or_none(
            metadata.get("unverified_claim_count"),
        )
        if unverified_claim_count is not None:
            extracted["unverified_claim_count"] = max(0, unverified_claim_count)

        verified_claim_count = cls._to_int_or_none(
            metadata.get("verified_claim_count"),
        )
        if verified_claim_count is not None:
            extracted["verified_claim_count"] = max(0, verified_claim_count)

        supporting_ratio = cls._to_ratio_or_none(metadata.get("supporting_ratio"))
        if supporting_ratio is not None:
            extracted["supporting_ratio"] = supporting_ratio

        return extracted

    def _remediation_queue_limits(self) -> tuple[int, float, float]:
        max_attempts = int(
            getattr(self._config.verification, "remediation_queue_max_attempts", 3) or 3,
        )
        process = self._process
        if process is not None:
            retry_budget = getattr(
                getattr(process, "verification_remediation", None),
                "retry_budget",
                {},
            )
            if isinstance(retry_budget, dict):
                raw_max_attempts = self._to_int_or_none(retry_budget.get("max_attempts"))
                if raw_max_attempts is not None and raw_max_attempts > 0:
                    max_attempts = raw_max_attempts
        if max_attempts <= 0:
            max_attempts = int(
                getattr(self._config.verification, "confirm_or_prune_max_attempts", 2) or 2,
            )
        max_attempts = max(1, max_attempts)

        base_backoff = float(
            getattr(self._config.verification, "remediation_queue_backoff_seconds", 2.0) or 0.0,
        )
        if base_backoff < 0:
            base_backoff = 0.0

        max_backoff = float(
            getattr(
                self._config.verification,
                "remediation_queue_max_backoff_seconds",
                30.0,
            ) or 0.0,
        )
        if max_backoff <= 0:
            max_backoff = max(base_backoff, 0.0)
        if max_backoff < base_backoff:
            max_backoff = base_backoff
        return max_attempts, base_backoff, max_backoff

    @staticmethod
    def _bounded_remediation_backoff_seconds(
        *,
        base_backoff_seconds: float,
        max_backoff_seconds: float,
        attempt_count: int,
    ) -> float:
        base = max(0.0, float(base_backoff_seconds or 0.0))
        if base <= 0:
            return 0.0
        ceiling = max(base, float(max_backoff_seconds or 0.0))
        exponent = max(0, int(attempt_count) - 1)
        computed = base * (2**exponent)
        return min(ceiling, computed)

    def _apply_unconfirmed_follow_up_success(
        self,
        *,
        result: SubtaskResult,
        verification: VerificationResult,
        note: str,
        default_reason_code: str,
    ) -> None:
        verification.passed = True
        if verification.outcome == "fail":
            verification.outcome = (
                "partial_verified"
                if self._config.verification.allow_partial_verified
                else "pass_with_warnings"
            )
        elif verification.outcome == "pass":
            verification.outcome = "pass_with_warnings"
        if not verification.reason_code:
            verification.reason_code = default_reason_code
        if verification.severity_class == "hard_invariant":
            verification.severity_class = "semantic"
        verification.feedback = (
            f"{verification.feedback}\n{note}" if verification.feedback else note
        )
        result.status = SubtaskResultStatus.SUCCESS

    async def _queue_remediation_work_item(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
        strategy: RetryStrategy,
        blocking: bool,
    ) -> None:
        queue = self._remediation_queue(task)
        reason_code = str(verification.reason_code or "").strip().lower()
        strategy_value = strategy.value
        now = datetime.now()
        max_attempts, base_backoff_seconds, max_backoff_seconds = (
            self._remediation_queue_limits()
        )
        uncertainty = self._extract_unconfirmed_metadata(verification)
        missing_targets = self._normalize_missing_targets(
            uncertainty.get("missing_targets"),
        )

        for item in queue:
            if not isinstance(item, dict):
                continue
            if str(item.get("subtask_id", "")).strip() != subtask.id:
                continue
            if str(item.get("strategy", "")).strip() != strategy_value:
                continue
            if str(item.get("reason_code", "")).strip().lower() != reason_code:
                continue
            state = str(item.get("state", "queued")).strip().lower()
            if state in _REMEDIATION_TERMINAL_STATES:
                continue

            if blocking and not bool(item.get("blocking", False)):
                item["blocking"] = True
            existing_targets = self._normalize_missing_targets(
                item.get("missing_targets"),
            )
            merged_targets = existing_targets + [
                target for target in missing_targets
                if target not in existing_targets
            ]
            if merged_targets:
                item["missing_targets"] = merged_targets
            for key, value in uncertainty.items():
                if key == "missing_targets":
                    continue
                item[key] = value
            item["feedback"] = verification.feedback or str(item.get("feedback", ""))
            item["verification_outcome"] = verification.outcome
            existing_max_attempts = self._to_int_or_none(item.get("max_attempts"))
            if existing_max_attempts is None:
                existing_max_attempts = max_attempts
            existing_base_backoff = self._to_float_or_none(
                item.get("base_backoff_seconds"),
            )
            if existing_base_backoff is None:
                try:
                    existing_base_backoff = float(
                        item.get("base_backoff_seconds", base_backoff_seconds) or 0.0,
                    )
                except (TypeError, ValueError):
                    existing_base_backoff = base_backoff_seconds
            existing_max_backoff = self._to_float_or_none(
                item.get("max_backoff_seconds"),
            )
            if existing_max_backoff is None:
                try:
                    existing_max_backoff = float(
                        item.get("max_backoff_seconds", max_backoff_seconds) or 0.0,
                    )
                except (TypeError, ValueError):
                    existing_max_backoff = max_backoff_seconds
            item["max_attempts"] = max(
                existing_max_attempts,
                max_attempts,
            )
            item["base_backoff_seconds"] = max(
                max(0.0, float(existing_base_backoff or 0.0)),
                base_backoff_seconds,
            )
            item["max_backoff_seconds"] = max(
                max(0.0, float(existing_max_backoff or 0.0)),
                max_backoff_seconds,
            )
            item["updated_at"] = now.isoformat()
            async with self._state_lock:
                self._state.save(task)
            self._emit(UNCONFIRMED_DATA_QUEUED, task.id, {
                "remediation_id": str(item.get("id", "")).strip(),
                "subtask_id": subtask.id,
                "strategy": strategy_value,
                "reason_code": reason_code,
                "blocking": bool(item.get("blocking", False)),
                "critical_path": bool(subtask.is_critical_path),
                "deduped": True,
            })
            return

        item = {
            "id": f"rem-{uuid.uuid4().hex[:10]}",
            "task_id": task.id,
            "subtask_id": subtask.id,
            "strategy": strategy_value,
            "reason_code": reason_code,
            "verification_outcome": verification.outcome,
            "feedback": verification.feedback or "",
            "blocking": bool(blocking),
            "state": "queued",
            "attempt_count": 0,
            "last_error": "",
            "terminal_reason": "",
            "critical_path": bool(subtask.is_critical_path),
            "max_attempts": max_attempts,
            "base_backoff_seconds": base_backoff_seconds,
            "max_backoff_seconds": max_backoff_seconds,
            "missing_targets": missing_targets,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "next_attempt_at": now.isoformat(),
            "ttl_at": (now + timedelta(hours=6)).isoformat(),
        }
        item.update({
            key: value
            for key, value in uncertainty.items()
            if key != "missing_targets"
        })
        queue.append(item)
        async with self._state_lock:
            task.add_decision(
                "Queued remediation for "
                f"{subtask.id} ({strategy.value}, blocking={blocking})."
            )
            self._state.save(task)
        self._emit(REMEDIATION_QUEUED, task.id, {
            "remediation_id": item["id"],
            "subtask_id": subtask.id,
            "strategy": strategy_value,
            "reason_code": reason_code,
            "blocking": bool(blocking),
        })
        self._emit(UNCONFIRMED_DATA_QUEUED, task.id, {
            "remediation_id": item["id"],
            "subtask_id": subtask.id,
            "strategy": strategy_value,
            "reason_code": reason_code,
            "blocking": bool(blocking),
            "critical_path": bool(subtask.is_critical_path),
            "deduped": False,
            "missing_targets": list(missing_targets),
        })

    @staticmethod
    def _resolution_plan_items(raw: object, *, max_items: int = 8) -> list[str]:
        if isinstance(raw, str):
            parts = [
                item.strip(" -\t")
                for item in raw.splitlines()
                if item.strip()
            ]
        elif isinstance(raw, list):
            parts = [str(item or "").strip() for item in raw]
        else:
            return []
        normalized: list[str] = []
        for item in parts:
            text = str(item or "").strip()
            if not text or text in normalized:
                continue
            normalized.append(text)
            if len(normalized) >= max_items:
                break
        return normalized

    @staticmethod
    def _compact_failure_resolution_metadata_value(
        value: object,
        *,
        depth: int = 0,
        max_depth: int = 3,
        max_list_items: int = 8,
        max_dict_items: int = 12,
        max_text_chars: int = 220,
    ) -> object:
        if depth >= max_depth:
            text = str(value or "").strip()
            if len(text) > max_text_chars:
                return text[: max_text_chars - 14] + "...[truncated]"
            return text

        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, str):
            text = value.strip()
            if len(text) > max_text_chars:
                return text[: max_text_chars - 14] + "...[truncated]"
            return text

        if isinstance(value, list):
            items: list[object] = []
            for item in value[:max_list_items]:
                items.append(
                    Orchestrator._compact_failure_resolution_metadata_value(
                        item,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_list_items=max_list_items,
                        max_dict_items=max_dict_items,
                        max_text_chars=max_text_chars,
                    ),
                )
            remainder = len(value) - len(items)
            if remainder > 0:
                items.append(f"...[{remainder} more items]")
            return items

        if isinstance(value, dict):
            compact: dict[str, object] = {}
            items = list(value.items())[:max_dict_items]
            for key, raw in items:
                key_text = str(key or "").strip()[:80]
                if not key_text:
                    continue
                compact[key_text] = Orchestrator._compact_failure_resolution_metadata_value(
                    raw,
                    depth=depth + 1,
                    max_depth=max_depth,
                    max_list_items=max_list_items,
                    max_dict_items=max_dict_items,
                    max_text_chars=max_text_chars,
                )
            remainder = len(value) - len(items)
            if remainder > 0:
                compact["_truncated_keys"] = remainder
            return compact

        text = str(value or "").strip()
        if len(text) > max_text_chars:
            return text[: max_text_chars - 14] + "...[truncated]"
        return text

    @classmethod
    def _summarize_failure_resolution_metadata(
        cls,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        if not isinstance(metadata, dict):
            return {}

        summary: dict[str, object] = {}
        for key in _FAILURE_RESOLUTION_METADATA_KEYS:
            if key not in metadata:
                continue
            summary[key] = cls._compact_failure_resolution_metadata_value(
                metadata.get(key),
            )

        scan = metadata.get("deterministic_placeholder_scan")
        if isinstance(scan, dict):
            prioritized_scan = {
                key: scan.get(key)
                for key in (
                    "scan_mode",
                    "scanned_file_count",
                    "matched_file_count",
                    "coverage_sufficient",
                    "coverage_insufficient_reason",
                    "cap_exhausted",
                    "cap_exhaustion_reason",
                    "candidate_source_counts",
                )
                if key in scan
            }
            if prioritized_scan:
                summary["deterministic_placeholder_scan"] = (
                    cls._compact_failure_resolution_metadata_value(
                        prioritized_scan,
                    )
                )

        if summary:
            return summary

        # Fallback for unknown schemas: include a small, compact preview.
        for key, raw in metadata.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            summary[key_text[:64]] = cls._compact_failure_resolution_metadata_value(raw)
            if len(summary) >= 6:
                break
        return summary

    def _failure_resolution_metadata_char_budget(self) -> int:
        raw = getattr(
            getattr(self._config, "limits", None),
            "evidence_context_text_max_chars",
            4000,
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 1400
        return max(600, min(value, 2200))

    def _format_failure_resolution_plan(self, response: ModelResponse) -> str:
        validation = self._validator.validate_json_response(
            response,
            expected_keys=["diagnosis", "actions"],
        )
        if validation.valid and isinstance(validation.parsed, dict):
            payload = validation.parsed
            diagnosis = str(payload.get("diagnosis", "") or "").strip()
            actions = self._resolution_plan_items(payload.get("actions"))
            guardrails = self._resolution_plan_items(payload.get("guardrails"))
            success = self._resolution_plan_items(
                payload.get("success_criteria"),
            )
            lines: list[str] = []
            if diagnosis:
                lines.append(f"Diagnosis: {diagnosis}")
            if actions:
                lines.append("Actions:")
                for index, action in enumerate(actions, start=1):
                    lines.append(f"{index}. {action}")
            if guardrails:
                lines.append("Guardrails:")
                for item in guardrails:
                    lines.append(f"- {item}")
            if success:
                lines.append("Success criteria:")
                for item in success:
                    lines.append(f"- {item}")
            rendered = "\n".join(lines).strip()
            if rendered:
                return rendered[:2400]
        fallback = str(response.text or "").strip()
        if not fallback:
            return ""
        return fallback[:2400]

    def _build_failure_resolution_prompt(
        self,
        *,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
        strategy: RetryStrategy,
        missing_targets: list[str],
        prior_attempts: list[AttemptRecord],
    ) -> str:
        raw_metadata = (
            verification.metadata
            if isinstance(verification.metadata, dict)
            else {}
        )
        metadata = self._summarize_failure_resolution_metadata(raw_metadata)
        metadata_json = json.dumps(metadata, indent=2, sort_keys=True, default=str)
        metadata_budget = self._failure_resolution_metadata_char_budget()
        if len(metadata_json) > metadata_budget:
            metadata_json = json.dumps(
                {
                    "_loom_meta": "metadata_truncated",
                    "total_chars": len(metadata_json),
                    "included_keys": list(metadata.keys())[:12],
                },
                indent=2,
                sort_keys=True,
            )
        expected_deliverables = self._expected_deliverables_for_subtask(subtask)
        changed_files = self._files_from_attempts(prior_attempts)
        current_changed = self._files_from_tool_calls(result.tool_calls)
        for path in current_changed:
            if path not in changed_files:
                changed_files.append(path)
        lines = [
            "You are planning a targeted remediation for a failed subtask.",
            "Return strict JSON with keys:",
            (
                '{"diagnosis": "...", "actions": ["..."], '
                '"guardrails": ["..."], "success_criteria": ["..."]}'
            ),
            "Keep the plan short and actionable.",
            "",
            "Rules:",
            "- Handle the observed failure generically; do not hardcode one-off logic.",
            "- Preserve validated outputs and evidence.",
            "- Prefer minimal edits over broad reruns.",
            (
                "- If outputs must align to canonical deliverables, explain how to "
                "reconcile file paths safely."
            ),
            "",
            f"Subtask ID: {subtask.id}",
            f"Subtask description: {subtask.description}",
            f"Acceptance criteria: {subtask.acceptance_criteria}",
            f"Retry strategy: {strategy.value}",
            f"Verification tier: {verification.tier}",
            f"Verification outcome: {verification.outcome}",
            f"Reason code: {verification.reason_code}",
            f"Severity class: {verification.severity_class}",
            f"Verification feedback: {verification.feedback or ''}",
            f"Execution summary: {result.summary or ''}",
        ]
        if missing_targets:
            lines.append(f"Missing targets: {', '.join(missing_targets)}")
        if expected_deliverables:
            lines.append(
                "Expected deliverables: " + ", ".join(expected_deliverables),
            )
        if changed_files:
            lines.append("Touched files: " + ", ".join(changed_files))
        if metadata_json != "{}":
            lines.extend([
                "Verification metadata:",
                metadata_json,
            ])
        return "\n".join(lines)

    async def _plan_failure_resolution(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
        strategy: RetryStrategy,
        missing_targets: list[str],
        prior_attempts: list[AttemptRecord],
    ) -> str:
        prompt = self._build_failure_resolution_prompt(
            subtask=subtask,
            result=result,
            verification=verification,
            strategy=strategy,
            missing_targets=missing_targets,
            prior_attempts=prior_attempts,
        )
        if not prompt.strip():
            return ""

        model = self._router.select(tier=2, role="planner")
        request_messages = [{"role": "user", "content": prompt}]
        policy = ModelRetryPolicy.from_execution_config(self._config.execution)
        invocation_attempt = 0
        request_diag = None
        max_tokens = self._planning_response_max_tokens()
        if max_tokens is None:
            max_tokens = 900
        max_tokens = max(256, min(max_tokens, 1500))

        async def _invoke_model():
            nonlocal invocation_attempt, request_diag
            invocation_attempt += 1
            request_diag = collect_request_diagnostics(
                messages=request_messages,
                origin="orchestrator.failure_resolution.complete",
            )
            self._emit(MODEL_INVOCATION, task.id, {
                "subtask_id": subtask.id,
                "model": model.name,
                "phase": "start",
                "operation": "failure_resolution_plan",
                "invocation_attempt": invocation_attempt,
                "invocation_max_attempts": policy.max_attempts,
                "retry_strategy": strategy.value,
                **request_diag.to_event_payload(),
            })
            return await model.complete(
                request_messages,
                max_tokens=max_tokens,
            )

        def _on_failure(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
        ) -> None:
            self._emit(MODEL_INVOCATION, task.id, {
                "subtask_id": subtask.id,
                "model": model.name,
                "phase": "done",
                "operation": "failure_resolution_plan",
                "invocation_attempt": attempt,
                "invocation_max_attempts": max_attempts,
                "retry_queue_remaining": remaining,
                "origin": request_diag.origin if request_diag else "",
                "error_type": type(error).__name__,
                "error": str(error),
                "retry_strategy": strategy.value,
            })

        try:
            response = await call_with_model_retry(
                _invoke_model,
                policy=policy,
                on_failure=_on_failure,
            )
        except Exception as e:
            logger.debug(
                "Failure-resolution planner call failed for %s/%s: %s",
                task.id,
                subtask.id,
                e,
                exc_info=True,
            )
            return ""

        self._emit(MODEL_INVOCATION, task.id, {
            "subtask_id": subtask.id,
            "model": model.name,
            "phase": "done",
            "operation": "failure_resolution_plan",
            "invocation_attempt": invocation_attempt,
            "invocation_max_attempts": policy.max_attempts,
            "origin": request_diag.origin if request_diag else "",
            "retry_strategy": strategy.value,
            **collect_response_diagnostics(response).to_event_payload(),
        })
        return self._format_failure_resolution_plan(response)

    def _build_remediation_retry_context(self, *, strategy: RetryStrategy) -> str:
        lines = [
            "TARGETED REMEDIATION:",
            "- Keep already validated work; avoid redoing solved sections.",
            "- Resolve only failing verification findings and missing evidence links.",
            "- Use explicit evidence to confirm uncertain claims; otherwise relabel "
            "or remove unsupported claims per process policy.",
            "- Make the smallest safe edits needed to satisfy acceptance criteria.",
        ]
        process = self._process
        if process is not None:
            instructions = process.prompt_remediation_instructions(strategy.value)
            if instructions:
                lines.append("PROCESS REMEDIATION INSTRUCTIONS:")
                lines.append(instructions)
        return "\n".join(lines)

    async def _run_confirm_or_prune_remediation(
        self,
        *,
        task: Task,
        subtask: Subtask,
        attempts: list[AttemptRecord],
        remediation_id: str | None = None,
    ) -> tuple[bool, str]:
        if not self._config.verification.auto_confirm_prune_critical_path:
            return False, "auto_confirm_prune_critical_path disabled"

        max_attempts = max(
            1,
            int(
                getattr(
                    self._config.verification,
                    "confirm_or_prune_max_attempts",
                    2,
                ) or 2
            ),
        )
        backoff_seconds = max(
            0.0,
            float(
                getattr(
                    self._config.verification,
                    "confirm_or_prune_backoff_seconds",
                    2.0,
                ) or 0.0
            ),
        )
        retry_on_transient = bool(
            getattr(
                self._config.verification,
                "confirm_or_prune_retry_on_transient",
                True,
            ),
        )
        last_failure = "process remediation failed"

        for attempt_number in range(1, max_attempts + 1):
            self._run_budget.observe_remediation_attempt()
            self._emit(SUBTASK_RETRYING, task.id, {
                "subtask_id": subtask.id,
                "mode": "confirm_or_prune",
                "reason": "critical_unconfirmed_data",
                "attempt": attempt_number,
                "max_attempts": max_attempts,
            })
            self._emit(REMEDIATION_ATTEMPT, task.id, {
                "remediation_id": remediation_id or "",
                "subtask_id": subtask.id,
                "attempt": attempt_number,
                "max_attempts": max_attempts,
                "phase": "start",
            })
            await self._persist_remediation_attempt(
                task=task,
                remediation_id=remediation_id or "",
                subtask_id=subtask.id,
                attempt=attempt_number,
                max_attempts=max_attempts,
                phase="start",
            )

            prior_successful_tool_calls: list[ToolCallRecord] = []
            prior_evidence_records = self._evidence_for_subtask(task.id, subtask.id)
            for attempt in attempts:
                raw_calls = getattr(attempt, "successful_tool_calls", [])
                if isinstance(raw_calls, list):
                    for call in raw_calls:
                        if isinstance(call, ToolCallRecord):
                            prior_successful_tool_calls.append(call)
                raw_evidence = getattr(attempt, "evidence_records", [])
                if isinstance(raw_evidence, list):
                    prior_evidence_records = merge_evidence_records(
                        prior_evidence_records,
                        [item for item in raw_evidence if isinstance(item, dict)],
                    )

            remediation_context = (
                self._build_remediation_retry_context(
                    strategy=RetryStrategy.UNCONFIRMED_DATA,
                )
            )
            expected_deliverables = self._expected_deliverables_for_subtask(subtask)
            remediation_context = self._augment_retry_context_for_outputs(
                subtask=subtask,
                attempts=attempts,
                strategy=RetryStrategy.UNCONFIRMED_DATA,
                expected_deliverables=expected_deliverables,
                base_context=remediation_context,
            )
            escalated_tier = self._retry.get_escalation_tier(
                attempt=len(attempts),
                original_tier=subtask.model_tier,
            )
            changelog = self._get_changelog(task)
            remediation_result, remediation_verification = await self._runner.run(
                task,
                subtask,
                model_tier=escalated_tier,
                retry_context=remediation_context,
                changelog=changelog,
                prior_successful_tool_calls=prior_successful_tool_calls,
                prior_evidence_records=prior_evidence_records,
                expected_deliverables=expected_deliverables,
                enforce_deliverable_paths=bool(expected_deliverables),
                edit_existing_only=bool(expected_deliverables),
                retry_strategy=RetryStrategy.UNCONFIRMED_DATA.value,
            )
            self._persist_subtask_evidence(
                task.id,
                subtask.id,
                remediation_result.evidence_records,
            )

            if remediation_verification.passed:
                await self._handle_success(
                    task,
                    subtask,
                    remediation_result,
                    remediation_verification,
                )
                self._emit(REMEDIATION_ATTEMPT, task.id, {
                    "remediation_id": remediation_id or "",
                    "subtask_id": subtask.id,
                    "attempt": attempt_number,
                    "max_attempts": max_attempts,
                    "phase": "done",
                    "outcome": "resolved",
                })
                await self._persist_remediation_attempt(
                    task=task,
                    remediation_id=remediation_id or "",
                    subtask_id=subtask.id,
                    attempt=attempt_number,
                    max_attempts=max_attempts,
                    phase="done",
                    outcome="resolved",
                    retry_strategy="resolved",
                    reason_code=str(remediation_verification.reason_code or ""),
                )
                self._record_confirm_or_prune_attempt(
                    task=task,
                    subtask_id=subtask.id,
                    status="resolved",
                    attempt=attempt_number,
                    max_attempts=max_attempts,
                    transient=False,
                    reason_code=remediation_verification.reason_code,
                    retry_strategy="resolved",
                    error="",
                )
                return True, ""

            remediation_strategy, missing_targets = self._retry.classify_failure(
                verification_feedback=remediation_verification.feedback,
                execution_error=remediation_result.summary,
                verification=remediation_verification,
            )
            combined_error = " | ".join(
                part
                for part in [
                    remediation_verification.feedback,
                    remediation_result.summary,
                ]
                if part
            ) or "process remediation failed"
            categorized = categorize_error(combined_error)
            transient = (
                remediation_strategy == RetryStrategy.RATE_LIMIT
                or categorized.category in {
                    ErrorCategory.MODEL_ERROR,
                    ErrorCategory.TIMEOUT,
                }
            )
            attempts.append(AttemptRecord(
                attempt=len(attempts) + 1,
                tier=escalated_tier,
                feedback=remediation_verification.feedback or None,
                error=combined_error,
                successful_tool_calls=[
                    call for call in remediation_result.tool_calls
                    if getattr(getattr(call, "result", None), "success", False)
                ],
                evidence_records=[
                    item for item in remediation_result.evidence_records
                    if isinstance(item, dict)
                ],
                retry_strategy=remediation_strategy,
                missing_targets=missing_targets,
                error_category=categorized.category,
            ))
            self._record_confirm_or_prune_attempt(
                task=task,
                subtask_id=subtask.id,
                status="failed",
                attempt=attempt_number,
                max_attempts=max_attempts,
                transient=transient,
                reason_code=remediation_verification.reason_code,
                retry_strategy=remediation_strategy.value,
                error=combined_error,
            )
            self._emit(REMEDIATION_ATTEMPT, task.id, {
                "remediation_id": remediation_id or "",
                "subtask_id": subtask.id,
                "attempt": attempt_number,
                "max_attempts": max_attempts,
                "phase": "done",
                "outcome": "failed",
                "retry_strategy": remediation_strategy.value,
                "transient": transient,
            })
            await self._persist_remediation_attempt(
                task=task,
                remediation_id=remediation_id or "",
                subtask_id=subtask.id,
                attempt=attempt_number,
                max_attempts=max_attempts,
                phase="done",
                outcome="failed",
                retry_strategy=remediation_strategy.value,
                transient=transient,
                reason_code=str(remediation_verification.reason_code or ""),
                error=combined_error,
            )
            last_failure = combined_error

            more_attempts_remain = attempt_number < max_attempts
            if remediation_strategy == RetryStrategy.RATE_LIMIT and more_attempts_remain:
                if backoff_seconds > 0:
                    await asyncio.sleep(backoff_seconds)
                continue
            if retry_on_transient and transient and more_attempts_remain:
                if backoff_seconds > 0:
                    await asyncio.sleep(backoff_seconds)
                continue
            break

        return False, last_failure

    def _record_confirm_or_prune_attempt(
        self,
        *,
        task: Task,
        subtask_id: str,
        status: str,
        attempt: int,
        max_attempts: int,
        transient: bool,
        reason_code: str,
        retry_strategy: str,
        error: str,
    ) -> None:
        attempts_map = task.metadata.get("confirm_or_prune_attempts")
        if not isinstance(attempts_map, dict):
            attempts_map = {}
            task.metadata["confirm_or_prune_attempts"] = attempts_map
        rows = attempts_map.get(subtask_id)
        if not isinstance(rows, list):
            rows = []
            attempts_map[subtask_id] = rows
        rows.append({
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "status": str(status),
            "transient": bool(transient),
            "reason_code": str(reason_code or "").strip().lower(),
            "retry_strategy": str(retry_strategy or "").strip().lower(),
            "error": str(error or ""),
            "at": datetime.now().isoformat(),
        })
        if len(rows) > 25:
            del rows[:-25]

    async def _persist_subtask_attempt_record(
        self,
        *,
        task: Task,
        subtask_id: str,
        attempt_record: AttemptRecord,
        verification: VerificationResult,
    ) -> None:
        if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
            return
        try:
            await self._memory.insert_subtask_attempt(
                task_id=task.id,
                run_id=self._task_run_id(task),
                subtask_id=subtask_id,
                attempt=int(getattr(attempt_record, "attempt", 0) or 0),
                tier=int(getattr(attempt_record, "tier", 1) or 1),
                retry_strategy=str(
                    getattr(getattr(attempt_record, "retry_strategy", None), "value", "")
                    or getattr(attempt_record, "retry_strategy", "")
                    or RetryStrategy.GENERIC.value
                ),
                reason_code=str(getattr(verification, "reason_code", "") or ""),
                feedback=str(getattr(attempt_record, "feedback", "") or ""),
                error=str(getattr(attempt_record, "error", "") or ""),
                missing_targets=list(getattr(attempt_record, "missing_targets", []) or []),
                error_category=str(
                    getattr(
                        getattr(attempt_record, "error_category", None),
                        "value",
                        "",
                    ) or ""
                ),
            )
        except Exception:
            logger.debug(
                "Failed persisting subtask attempt for %s/%s",
                task.id,
                subtask_id,
                exc_info=True,
            )

    async def _sync_remediation_queue_to_db(self, task: Task) -> None:
        if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
            return
        queue = task.metadata.get("remediation_queue")
        if not isinstance(queue, list):
            return
        for item in queue:
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            payload["task_id"] = task.id
            payload.setdefault("run_id", self._task_run_id(task))
            try:
                await self._memory.upsert_remediation_item(payload)
            except Exception:
                logger.debug(
                    "Failed syncing remediation item %s for task %s",
                    str(item.get("id", "")),
                    task.id,
                    exc_info=True,
                )

    async def _hydrate_remediation_queue_from_db(self, task: Task) -> None:
        if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
            return
        try:
            db_items = await self._memory.list_remediation_items(task_id=task.id)
        except Exception:
            logger.debug("Failed loading remediation queue from db for %s", task.id, exc_info=True)
            return
        if not db_items:
            return
        queue = self._remediation_queue(task)
        existing_ids = {
            str(item.get("id", "")).strip()
            for item in queue
            if isinstance(item, dict)
        }
        for item in db_items:
            item_id = str(item.get("id", "")).strip()
            if not item_id or item_id in existing_ids:
                continue
            queue.append(item)
        self._state.save(task)

    async def _persist_remediation_attempt(
        self,
        *,
        task: Task,
        remediation_id: str,
        subtask_id: str,
        attempt: int,
        max_attempts: int,
        phase: str,
        outcome: str = "",
        retry_strategy: str = "",
        transient: bool = False,
        reason_code: str = "",
        error: str = "",
    ) -> None:
        if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
            return
        try:
            await self._memory.insert_remediation_attempt(
                remediation_id=remediation_id,
                task_id=task.id,
                run_id=self._task_run_id(task),
                subtask_id=subtask_id,
                attempt=attempt,
                max_attempts=max_attempts,
                phase=phase,
                outcome=outcome,
                retry_strategy=retry_strategy,
                transient=transient,
                reason_code=reason_code,
                error=error,
            )
        except Exception:
            logger.debug(
                "Failed persisting remediation attempt for %s/%s",
                task.id,
                remediation_id,
                exc_info=True,
            )

    def _remediation_queue(self, task: Task) -> list[dict]:
        queue = task.metadata.get("remediation_queue")
        if not isinstance(queue, list):
            queue = []
            task.metadata["remediation_queue"] = queue
        queue[:] = [item for item in queue if isinstance(item, dict)]
        return queue

    @staticmethod
    def _parse_iso_datetime(raw: object) -> datetime | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _remediation_item_due(self, item: dict, now: datetime) -> bool:
        due_at = self._parse_iso_datetime(item.get("next_attempt_at"))
        if due_at is None:
            return True
        return now >= due_at

    def _remediation_item_expired(self, item: dict, now: datetime) -> bool:
        ttl_at = self._parse_iso_datetime(item.get("ttl_at"))
        if ttl_at is None:
            return False
        return now >= ttl_at

    async def _process_remediation_queue(
        self,
        *,
        task: Task,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
        finalizing: bool,
    ) -> None:
        queue = task.metadata.get("remediation_queue")
        if not isinstance(queue, list) or not queue:
            return

        default_max_attempts, default_base_backoff, default_max_backoff = (
            self._remediation_queue_limits()
        )
        changed = False
        now = datetime.now()

        for item in queue:
            if not isinstance(item, dict):
                continue
            state = str(item.get("state", "queued")).strip().lower()
            if state in _REMEDIATION_TERMINAL_STATES:
                continue

            remediation_id = str(item.get("id", "")).strip() or f"rem-{uuid.uuid4().hex[:10]}"
            item["id"] = remediation_id
            subtask_id = str(item.get("subtask_id", "")).strip()

            if self._remediation_item_expired(item, now):
                item["state"] = "expired"
                item["updated_at"] = now.isoformat()
                item["last_error"] = (
                    str(item.get("last_error", "")).strip()
                    or "remediation ttl exceeded"
                )
                item["terminal_reason"] = "ttl_expired"
                changed = True
                self._emit(REMEDIATION_EXPIRED, task.id, {
                    "remediation_id": remediation_id,
                    "subtask_id": subtask_id,
                    "strategy": str(item.get("strategy", "")),
                })
                self._emit(REMEDIATION_TERMINAL, task.id, {
                    "remediation_id": remediation_id,
                    "subtask_id": subtask_id,
                    "strategy": str(item.get("strategy", "")),
                    "state": "expired",
                    "reason": "ttl_expired",
                })
                continue

            if not self._remediation_item_due(item, now):
                continue

            item["state"] = "running"
            item["updated_at"] = now.isoformat()
            changed = True
            self._emit(REMEDIATION_STARTED, task.id, {
                "remediation_id": remediation_id,
                "subtask_id": subtask_id,
                "strategy": str(item.get("strategy", "")),
                "blocking": bool(item.get("blocking", False)),
            })

            resolved, error = await self._execute_remediation_item(
                task=task,
                item=item,
                attempts_by_subtask=attempts_by_subtask,
            )
            if resolved:
                item["state"] = "resolved"
                item["updated_at"] = datetime.now().isoformat()
                item["last_error"] = ""
                item["terminal_reason"] = "resolved"
                changed = True
                self._emit(REMEDIATION_RESOLVED, task.id, {
                    "remediation_id": remediation_id,
                    "subtask_id": subtask_id,
                    "strategy": str(item.get("strategy", "")),
                })
                self._emit(REMEDIATION_TERMINAL, task.id, {
                    "remediation_id": remediation_id,
                    "subtask_id": subtask_id,
                    "strategy": str(item.get("strategy", "")),
                    "state": "resolved",
                    "reason": "resolved",
                })
                continue

            attempt_count = int(item.get("attempt_count", 0) or 0) + 1
            item["attempt_count"] = attempt_count
            item["last_error"] = str(error or "").strip()
            item["updated_at"] = datetime.now().isoformat()
            parsed_max_attempts = self._to_int_or_none(item.get("max_attempts"))
            if parsed_max_attempts is None:
                parsed_max_attempts = default_max_attempts
            max_attempts = max(1, parsed_max_attempts)

            parsed_base_backoff = self._to_float_or_none(
                item.get("base_backoff_seconds"),
            )
            if parsed_base_backoff is None:
                parsed_base_backoff = default_base_backoff
            base_backoff_seconds = max(0.0, parsed_base_backoff)

            parsed_max_backoff = self._to_float_or_none(
                item.get("max_backoff_seconds"),
            )
            if parsed_max_backoff is None:
                parsed_max_backoff = default_max_backoff
            max_backoff_seconds = max(
                base_backoff_seconds,
                parsed_max_backoff,
            )
            exhausted = attempt_count >= max_attempts
            if exhausted or (finalizing and bool(item.get("blocking", False))):
                item["state"] = "failed"
                item["terminal_reason"] = (
                    "max_attempts_exhausted"
                    if exhausted else "blocking_unresolved_at_finalization"
                )
                self._emit(REMEDIATION_FAILED, task.id, {
                    "remediation_id": remediation_id,
                    "subtask_id": subtask_id,
                    "strategy": str(item.get("strategy", "")),
                    "attempt_count": attempt_count,
                    "error": item["last_error"],
                })
                self._emit(REMEDIATION_TERMINAL, task.id, {
                    "remediation_id": remediation_id,
                    "subtask_id": subtask_id,
                    "strategy": str(item.get("strategy", "")),
                    "state": "failed",
                    "reason": str(item.get("terminal_reason", "")),
                    "attempt_count": attempt_count,
                })
            else:
                item["state"] = "queued"
                delay_seconds = self._bounded_remediation_backoff_seconds(
                    base_backoff_seconds=base_backoff_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                    attempt_count=attempt_count,
                )
                item["next_attempt_at"] = (
                    datetime.now() + timedelta(seconds=delay_seconds)
                ).isoformat()
            changed = True

        if finalizing:
            for item in queue:
                if not isinstance(item, dict):
                    continue
                if not bool(item.get("blocking", False)):
                    continue
                state = str(item.get("state", "queued")).strip().lower()
                if state in _REMEDIATION_TERMINAL_STATES:
                    continue
                item["state"] = "failed"
                item["updated_at"] = datetime.now().isoformat()
                item["last_error"] = (
                    str(item.get("last_error", "")).strip()
                    or "blocking remediation unresolved at task finalization"
                )
                item["terminal_reason"] = "blocking_unresolved_at_finalization"
                changed = True
                self._emit(REMEDIATION_FAILED, task.id, {
                    "remediation_id": str(item.get("id", "")).strip(),
                    "subtask_id": str(item.get("subtask_id", "")).strip(),
                    "strategy": str(item.get("strategy", "")),
                    "attempt_count": int(item.get("attempt_count", 0) or 0),
                    "error": item["last_error"],
                })
                self._emit(REMEDIATION_TERMINAL, task.id, {
                    "remediation_id": str(item.get("id", "")).strip(),
                    "subtask_id": str(item.get("subtask_id", "")).strip(),
                    "strategy": str(item.get("strategy", "")),
                    "state": "failed",
                    "reason": "blocking_unresolved_at_finalization",
                    "attempt_count": int(item.get("attempt_count", 0) or 0),
                })

        if changed:
            async with self._state_lock:
                self._state.save(task)

    async def _execute_remediation_item(
        self,
        *,
        task: Task,
        item: dict,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
    ) -> tuple[bool, str]:
        strategy_value = str(item.get("strategy", "")).strip()
        subtask_id = str(item.get("subtask_id", "")).strip()
        subtask = task.get_subtask(subtask_id)
        if subtask is None:
            return False, f"subtask not found: {subtask_id}"

        if strategy_value != RetryStrategy.UNCONFIRMED_DATA.value:
            return False, f"unsupported remediation strategy: {strategy_value}"

        attempts = attempts_by_subtask.setdefault(subtask.id, [])
        return await self._run_confirm_or_prune_remediation(
            task=task,
            subtask=subtask,
            attempts=attempts,
            remediation_id=str(item.get("id", "")).strip() or None,
        )

    def _expected_deliverables_for_subtask(self, subtask: Subtask) -> list[str]:
        if self._process is None:
            return []
        deliverables = self._process.get_deliverables()
        if not deliverables:
            return []
        if subtask.id in deliverables:
            return [
                str(item).strip()
                for item in deliverables[subtask.id]
                if str(item).strip()
            ]
        if len(deliverables) == 1:
            return [
                str(item).strip()
                for item in next(iter(deliverables.values()))
                if str(item).strip()
            ]
        return []

    @staticmethod
    def _files_from_attempts(attempts: list[AttemptRecord], *, max_items: int = 24) -> list[str]:
        files: list[str] = []
        seen: set[str] = set()
        for attempt in attempts:
            raw_calls = getattr(attempt, "successful_tool_calls", [])
            if not isinstance(raw_calls, list):
                continue
            for call in raw_calls:
                result = getattr(call, "result", None)
                changed = getattr(result, "files_changed", [])
                if not isinstance(changed, list):
                    continue
                for item in changed:
                    text = str(item or "").strip()
                    if not text or text in seen:
                        continue
                    seen.add(text)
                    files.append(text)
                    if len(files) >= max_items:
                        return files
        return files

    @staticmethod
    def _files_from_tool_calls(tool_calls: list, *, max_items: int = 24) -> list[str]:
        files: list[str] = []
        seen: set[str] = set()
        if not isinstance(tool_calls, list):
            return files
        for call in tool_calls:
            result = getattr(call, "result", None)
            if not getattr(result, "success", False):
                continue
            changed = getattr(result, "files_changed", [])
            if not isinstance(changed, list):
                continue
            for item in changed:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                files.append(text)
                if len(files) >= max_items:
                    return files
        return files

    def _augment_retry_context_for_outputs(
        self,
        *,
        subtask: Subtask,
        attempts: list[AttemptRecord],
        strategy: RetryStrategy,
        expected_deliverables: list[str],
        base_context: str,
    ) -> str:
        del subtask  # Reserved for future per-subtask formatting.
        lines: list[str] = []
        existing_files = self._files_from_attempts(attempts)
        if existing_files:
            lines.append("EDIT-IN-PLACE FILES (do not fork or rename):")
            for path in existing_files:
                lines.append(f"- {path}")
            lines.append(
                "Do not create alternate copies such as *-v2.*, *_v2.*, *-copy.*, "
                "or similarly suffixed variants."
            )
        if expected_deliverables:
            lines.append("CANONICAL DELIVERABLE FILES FOR THIS SUBTASK:")
            for name in expected_deliverables:
                lines.append(f"- {name}")
            lines.append(
                "Write/update only these deliverable filenames for this phase. "
                "If fixing verification issues, patch these files in place."
            )
        if (
            strategy in {
                RetryStrategy.RATE_LIMIT,
                RetryStrategy.EVIDENCE_GAP,
                RetryStrategy.UNCONFIRMED_DATA,
            }
            and expected_deliverables
        ):
            lines.append(
                "Remediation scope: keep validated content and make only minimal edits "
                "needed to satisfy failed checks."
            )
        if not lines:
            return base_context
        block = "\n".join(lines)
        if base_context.strip():
            return f"{base_context}\n\n{block}"
        return block

    @staticmethod
    def _build_replan_reason(
        subtask: Subtask,
        verification: VerificationResult,
    ) -> str:
        """Build a concise, structured reason string for replanning prompts."""
        feedback = (verification.feedback or "").strip()
        reason = f"reason_code={verification.reason_code}" if verification.reason_code else ""
        outcome = (
            f"outcome={verification.outcome}"
            if verification.outcome else ""
        )
        suffix = ", ".join(part for part in [outcome, reason] if part)
        suffix_text = f" ({suffix})" if suffix else ""
        if feedback:
            return (
                f"Subtask '{subtask.id}' failed verification tier "
                f"{verification.tier}{suffix_text}: {feedback}"
            )
        return (
            f"Subtask '{subtask.id}' failed verification tier "
            f"{verification.tier}{suffix_text} with no feedback."
        )

    async def _retry_verification_only(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        attempts: list[AttemptRecord],
    ) -> VerificationResult:
        """Retry verifier path only (no executor/tool rerun)."""
        self._emit(SUBTASK_RETRYING, task.id, {
            "subtask_id": subtask.id,
            "mode": "verification_only",
            "reason": "verifier_parse_error",
        })
        prior_calls: list[ToolCallRecord] = []
        prior_evidence = self._evidence_for_subtask(task.id, subtask.id)
        for attempt in attempts:
            raw_calls = getattr(attempt, "successful_tool_calls", [])
            if isinstance(raw_calls, list):
                for call in raw_calls:
                    if isinstance(call, ToolCallRecord):
                        prior_calls.append(call)
            raw_evidence = getattr(attempt, "evidence_records", [])
            if isinstance(raw_evidence, list):
                prior_evidence = merge_evidence_records(
                    prior_evidence,
                    [item for item in raw_evidence if isinstance(item, dict)],
                )
        prior_evidence = merge_evidence_records(
            prior_evidence,
            [item for item in result.evidence_records if isinstance(item, dict)],
        )
        workspace = Path(task.workspace) if task.workspace else None
        return await self._verification.verify(
            subtask=subtask,
            result_summary=result.summary or "",
            tool_calls=result.tool_calls,
            evidence_tool_calls=prior_calls,
            evidence_records=prior_evidence,
            workspace=workspace,
            tier=max(2, subtask.verification_tier),
            task_id=task.id,
        )

    async def _handle_success(
        self,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
    ) -> None:
        """Process a successful subtask: update state, check approval."""
        self._persist_subtask_evidence(task.id, subtask.id, result.evidence_records)
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

        remediation_mode = ""
        remediation_required = False
        if verification and isinstance(verification.metadata, dict):
            remediation_mode = str(
                verification.metadata.get("remediation_mode", ""),
            ).strip().lower()
            remediation_required = bool(
                verification.metadata.get("remediation_required", False),
            )
        if verification and remediation_required and remediation_mode == "queue_follow_up":
            await self._queue_remediation_work_item(
                task=task,
                subtask=subtask,
                verification=verification,
                strategy=RetryStrategy.UNCONFIRMED_DATA,
                blocking=False,
            )

        self._emit(SUBTASK_COMPLETED, task.id, {
            "subtask_id": subtask.id,
            "status": result.status,
            "summary": summary,
            "duration": result.duration_seconds,
            "verification_outcome": verification.outcome if verification else "",
            "reason_code": verification.reason_code if verification else "",
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

    async def _plan_task(
        self,
        task: Task,
        *,
        planner_feedback: str = "",
    ) -> Plan:
        """Invoke the planner model to decompose the task into subtasks."""
        workspace_listing = ""
        code_analysis = ""
        workspace_analysis = ""
        read_roots = self._read_roots_for_task(task)
        auth_context = None
        try:
            metadata = task.metadata if isinstance(task.metadata, dict) else {}
            auth_context = build_run_auth_context(
                workspace=Path(task.workspace) if task.workspace else None,
                metadata=metadata,
                available_mcp_aliases=set(self._config.mcp.servers.keys()),
            )
        except AuthResolutionError as e:
            logger.warning("Auth context unavailable during planning for %s: %s", task.id, e)

        if task.workspace:
            workspace_path = Path(task.workspace)
            if workspace_path.exists():
                # Run listing and analysis in parallel
                async def _do_listing():
                    return await self._tools.execute(
                        "list_directory",
                        {},
                        workspace=workspace_path,
                        read_roots=read_roots,
                        auth_context=auth_context,
                    )

                async def _do_analysis():
                    analysis_path = read_roots[0] if read_roots else workspace_path
                    if self._process and self._process.workspace_scan:
                        result = await self._analyze_workspace_for_process(
                            analysis_path,
                        )
                        return ("workspace", result)
                    return ("code", await self._analyze_workspace(
                        analysis_path,
                    ))

                listing_result, analysis_result = await asyncio.gather(
                    _do_listing(), _do_analysis(),
                )
                if listing_result.success:
                    workspace_listing = listing_result.output
                analysis_type, analysis_text = analysis_result
                if analysis_type == "workspace":
                    workspace_analysis = analysis_text
                else:
                    code_analysis = analysis_text

        prompt = self._prompts.build_planner_prompt(
            task=task,
            workspace_listing=workspace_listing,
            code_analysis=code_analysis,
            workspace_analysis=workspace_analysis,
        )
        planner_feedback_text = str(planner_feedback or "").strip()
        if planner_feedback_text:
            prompt = (
                f"{prompt}\n\nPLANNER RETRY FEEDBACK:\n"
                f"{planner_feedback_text}\n"
                "Return corrected JSON only."
            )

        model = self._router.select(tier=2, role="planner")
        request_messages = [{"role": "user", "content": prompt}]
        policy = ModelRetryPolicy.from_execution_config(self._config.execution)
        invocation_attempt = 0
        request_diag = None

        async def _invoke_model():
            nonlocal invocation_attempt, request_diag
            invocation_attempt += 1
            request_diag = collect_request_diagnostics(
                messages=request_messages,
                origin="orchestrator.plan_task.complete",
            )
            self._emit(MODEL_INVOCATION, task.id, {
                "subtask_id": "planning",
                "model": model.name,
                "phase": "start",
                "operation": "complete",
                "invocation_attempt": invocation_attempt,
                "invocation_max_attempts": policy.max_attempts,
                **request_diag.to_event_payload(),
            })
            return await model.complete(
                request_messages,
                max_tokens=self._planning_response_max_tokens(),
            )

        def _on_failure(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
        ) -> None:
            self._emit(MODEL_INVOCATION, task.id, {
                "subtask_id": "planning",
                "model": model.name,
                "phase": "done",
                "operation": "complete",
                "invocation_attempt": attempt,
                "invocation_max_attempts": max_attempts,
                "retry_queue_remaining": remaining,
                "origin": request_diag.origin if request_diag else "",
                "error_type": type(error).__name__,
                "error": str(error),
            })

        try:
            response = await call_with_model_retry(
                _invoke_model,
                policy=policy,
                on_failure=_on_failure,
            )
        except Exception as e:
            logger.warning(
                "Planning model call failed after %s attempts for task %s; using fallback plan: %s",
                policy.max_attempts,
                task.id,
                e,
            )
            return await self._build_planner_degraded_plan(
                task=task,
                reason_code="planner_model_failure",
                detail=f"{type(e).__name__}: {e}",
            )
        self._emit(MODEL_INVOCATION, task.id, {
            "subtask_id": "planning",
            "model": model.name,
            "phase": "done",
            "operation": "complete",
            "invocation_attempt": invocation_attempt,
            "invocation_max_attempts": policy.max_attempts,
            "origin": request_diag.origin if request_diag else "",
            **collect_response_diagnostics(response).to_event_payload(),
        })

        try:
            plan = self._parse_plan(response, goal=task.goal)
        except ValueError as e:
            return await self._build_planner_degraded_plan(
                task=task,
                reason_code="planner_json_parse_failed",
                detail=str(e),
            )
        return self._apply_process_phase_mode(plan)

    @staticmethod
    def _read_roots_for_task(task: Task) -> list[Path]:
        """Resolve additional read roots from task metadata.

        Only parent roots of the task workspace are accepted.
        """
        workspace_text = str(task.workspace or "").strip()
        if not workspace_text:
            return []
        try:
            workspace = Path(workspace_text).resolve()
        except Exception:
            return []

        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        raw_roots = metadata.get("read_roots", [])
        if isinstance(raw_roots, str):
            raw_roots = [raw_roots]
        if not isinstance(raw_roots, list):
            return []

        roots: list[Path] = []
        seen: set[Path] = set()
        for raw in raw_roots:
            try:
                candidate = Path(str(raw)).expanduser().resolve()
            except Exception:
                continue
            if candidate == Path(candidate.anchor):
                continue
            try:
                workspace.relative_to(candidate)
            except ValueError:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            roots.append(candidate)
        return roots

    # Document extensions grouped by category for workspace scanning.
    _DOC_EXTENSIONS: dict[str, tuple[str, ...]] = {
        "Documents": (".md", ".rst", ".txt", ".pdf"),
        "Data": (".csv", ".json", ".yaml", ".yml", ".toml", ".xml"),
        "Images": (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"),
        "Office": (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"),
    }
    _ALL_DOC_EXTENSIONS: frozenset[str] = frozenset(
        ext for exts in _DOC_EXTENSIONS.values() for ext in exts
    )
    _EVIDENCE_LEDGER_CSV_NAME = "evidence-ledger.csv"
    _EVIDENCE_LEDGER_CSV_BASE_FIELDS: tuple[str, ...] = (
        "evidence_id",
        "task_id",
        "subtask_id",
        "tool",
        "evidence_kind",
        "tool_call_id",
        "source_url",
        "query",
        "quality",
        "created_at",
        "snippet",
        "context_text",
        "facets",
    )

    async def _analyze_workspace(self, workspace_path: Path) -> str:
        """Run code analysis *and* document scan for better planning context.

        Returns a summary combining code structure (classes, functions,
        imports) and an inventory of non-code documents found in the
        workspace.  Best-effort — returns empty string on failure.
        """
        parts: list[str] = []

        # --- Code analysis (existing behaviour) ---
        try:
            from loom.tools.code_analysis import analyze_directory

            structures = await run_blocking_io(
                analyze_directory,
                workspace_path,
                max_files=20,
            )
            if structures:
                summaries = [s.to_summary() for s in structures]
                parts.append("\n\n".join(summaries))
        except Exception as e:
            logger.warning("Code analysis failed for %s: %s", workspace_path, e)

        # --- Document / non-code file scan ---
        try:
            doc_summary = await run_blocking_io(
                self._scan_workspace_documents,
                workspace_path,
            )
            if doc_summary:
                parts.append(doc_summary)
        except Exception as e:
            logger.warning("Document scan failed for %s: %s", workspace_path, e)

        return "\n\n".join(parts)

    def _scan_workspace_documents(
        self,
        workspace_path: Path,
        max_per_category: int = 15,
    ) -> str:
        """Scan workspace for non-code documents grouped by category.

        Returns a concise inventory string, or empty string if nothing
        found.  Skips hidden directories and common noise directories.
        """
        skip_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
            "dist", "build", ".eggs",
        }

        found: dict[str, list[str]] = {}

        for path in sorted(workspace_path.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in self._ALL_DOC_EXTENSIONS:
                continue
            # Skip noisy directories
            rel_parts = path.relative_to(workspace_path).parts
            if any(p.startswith(".") or p in skip_dirs for p in rel_parts[:-1]):
                continue

            for category, extensions in self._DOC_EXTENSIONS.items():
                if suffix in extensions:
                    found.setdefault(category, [])
                    if len(found[category]) < max_per_category:
                        found[category].append(
                            str(path.relative_to(workspace_path)),
                        )
                    break

        if not found:
            return ""

        lines = ["Documents and non-code files:"]
        for category, files in found.items():
            lines.append(f"\n  {category}:")
            for f in files:
                lines.append(f"    - {f}")
        return "\n".join(lines)

    async def _analyze_workspace_for_process(
        self, workspace_path: Path,
    ) -> str:
        """Analyze workspace using process-specific scan guidance.

        Instead of code analysis, scans for file types specified
        in the process definition's workspace_analysis.scan_for.
        """
        return await run_blocking_io(self._analyze_workspace_for_process_sync, workspace_path)

    def _analyze_workspace_for_process_sync(self, workspace_path: Path) -> str:
        """Sync implementation for process-specific workspace scan."""
        try:
            found_files: dict[str, list[str]] = {}
            for pattern in self._process.workspace_scan:
                # Pattern format: "*.md — description"
                glob_pattern = pattern.split("—")[0].split(" — ")[0].strip()
                # Handle comma-separated patterns like "*.csv, *.xlsx"
                for sub_pattern in glob_pattern.split(","):
                    sub_pattern = sub_pattern.strip()
                    if sub_pattern:
                        matches = list(workspace_path.glob(sub_pattern))
                        if matches:
                            found_files[sub_pattern] = [
                                str(m.relative_to(workspace_path))
                                for m in matches[:20]
                            ]

            if not found_files:
                return "No relevant existing files found in workspace."

            lines = ["Existing workspace files:"]
            for pattern, files in found_files.items():
                lines.append(f"\n{pattern}:")
                for f in files:
                    lines.append(f"  - {f}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("Process workspace scan failed: %s", e)
            return ""

    async def _replan_task(
        self,
        task: Task,
        *,
        reason: str = "subtask_failures",
        failed_subtask_id: str = "",
        verification_feedback: str | None = None,
    ) -> bool:
        """Re-plan the task after subtask failures.

        Returns True if re-planning succeeded and execution can continue.
        """
        self._emit(TASK_REPLANNING, task.id, {
            "reason": reason,
            "failed_subtask_id": failed_subtask_id,
            "verification_feedback": verification_feedback or "",
        })

        discoveries = [d for d in task.decisions_log]
        errors = [
            f"{e.subtask}: {e.error}" for e in task.errors_encountered
        ]

        try:
            state_yaml = self._state.to_compact_yaml(task)
            model = self._router.select(tier=2, role="planner")
            policy = ModelRetryPolicy.from_execution_config(self._config.execution)
            max_structural_attempts = self._topology_retry_attempts()
            topology_feedback = ""

            for structural_attempt in range(1, max_structural_attempts + 1):
                prompt = self._prompts.build_replanner_prompt(
                    goal=task.goal,
                    current_state_yaml=state_yaml,
                    discoveries=discoveries,
                    errors=errors,
                    original_plan=task.plan,
                    replan_reason=reason,
                )
                feedback_parts: list[str] = []
                base_feedback = str(verification_feedback or "").strip()
                if base_feedback:
                    feedback_parts.append(base_feedback)
                if topology_feedback:
                    feedback_parts.append(topology_feedback)
                if feedback_parts:
                    prompt = (
                        f"{prompt}\n\nREPLANNER FEEDBACK:\n"
                        + "\n\n".join(feedback_parts)
                        + "\n\nReturn corrected JSON only."
                    )

                request_messages = [{"role": "user", "content": prompt}]
                invocation_attempt = 0
                request_diag = None

                async def _invoke_replanner():
                    nonlocal invocation_attempt, request_diag
                    invocation_attempt += 1
                    request_diag = collect_request_diagnostics(
                        messages=request_messages,
                        origin="orchestrator.replan_task.complete",
                    )
                    self._emit(MODEL_INVOCATION, task.id, {
                        "subtask_id": failed_subtask_id or "replanning",
                        "model": model.name,
                        "phase": "start",
                        "operation": "complete",
                        "invocation_attempt": invocation_attempt,
                        "invocation_max_attempts": policy.max_attempts,
                        "structural_attempt": structural_attempt,
                        "structural_max_attempts": max_structural_attempts,
                        **request_diag.to_event_payload(),
                    })
                    return await model.complete(
                        request_messages,
                        max_tokens=self._planning_response_max_tokens(),
                    )

                def _on_replanner_failure(
                    attempt: int,
                    max_attempts: int,
                    error: BaseException,
                    remaining: int,
                ) -> None:
                    self._emit(MODEL_INVOCATION, task.id, {
                        "subtask_id": failed_subtask_id or "replanning",
                        "model": model.name,
                        "phase": "done",
                        "operation": "complete",
                        "invocation_attempt": attempt,
                        "invocation_max_attempts": max_attempts,
                        "retry_queue_remaining": remaining,
                        "origin": request_diag.origin if request_diag else "",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "structural_attempt": structural_attempt,
                        "structural_max_attempts": max_structural_attempts,
                    })

                response = await call_with_model_retry(
                    _invoke_replanner,
                    policy=policy,
                    on_failure=_on_replanner_failure,
                )
                self._emit(MODEL_INVOCATION, task.id, {
                    "subtask_id": failed_subtask_id or "replanning",
                    "model": model.name,
                    "phase": "done",
                    "operation": "complete",
                    "invocation_attempt": invocation_attempt,
                    "invocation_max_attempts": policy.max_attempts,
                    "origin": request_diag.origin if request_diag else "",
                    "structural_attempt": structural_attempt,
                    "structural_max_attempts": max_structural_attempts,
                    **collect_response_diagnostics(response).to_event_payload(),
                })
                try:
                    parsed_plan = self._apply_process_phase_mode(
                        self._parse_plan(response, goal=task.goal),
                    )
                except ValueError as e:
                    parse_error = str(e).strip() or "invalid replanner JSON"
                    self._emit(TASK_REPLAN_REJECTED, task.id, {
                        "failed_subtask_id": failed_subtask_id,
                        "reason": reason,
                        "validation_error": parse_error,
                        "old_subtask_ids": [s.id for s in task.plan.subtasks],
                        "new_subtask_ids": [],
                        "attempt": structural_attempt,
                        "max_attempts": max_structural_attempts,
                    })
                    task.add_decision(
                        f"Rejected replanned plan: {parse_error}",
                    )
                    self._state.save(task)
                    if structural_attempt >= max_structural_attempts:
                        return False
                    topology_feedback = (
                        "Previous replanned output was not valid JSON.\n"
                        f"Validation error: {parse_error}\n"
                        "Return corrected JSON only."
                    )
                    continue
                new_plan = parsed_plan
                topology_error = ""
                try:
                    new_plan = self._prepare_plan_for_execution(
                        task=task,
                        plan=parsed_plan,
                        context="replanner",
                    )
                except ValueError as e:
                    topology_error = str(e).strip() or "invalid replanned topology"

                if topology_error:
                    self._emit(TASK_REPLAN_REJECTED, task.id, {
                        "failed_subtask_id": failed_subtask_id,
                        "reason": reason,
                        "validation_error": topology_error,
                        "old_subtask_ids": [s.id for s in task.plan.subtasks],
                        "new_subtask_ids": [s.id for s in parsed_plan.subtasks],
                        "attempt": structural_attempt,
                        "max_attempts": max_structural_attempts,
                    })
                    task.add_decision(f"Rejected replanned plan: {topology_error}")
                    self._state.save(task)
                    if structural_attempt >= max_structural_attempts:
                        return False
                    topology_feedback = (
                        "Previous replanned plan was rejected for invalid topology.\n"
                        f"Validation error: {topology_error}\n"
                        "Return corrected JSON that preserves existing IDs and "
                        "satisfies all dependency and synthesis-topology constraints."
                    )
                    continue

                validation_error = self._validate_replan_contract(
                    current_plan=task.plan,
                    replanned_plan=new_plan,
                )
                if validation_error:
                    self._emit(TASK_REPLAN_REJECTED, task.id, {
                        "failed_subtask_id": failed_subtask_id,
                        "reason": reason,
                        "validation_error": validation_error,
                        "old_subtask_ids": [s.id for s in task.plan.subtasks],
                        "new_subtask_ids": [s.id for s in new_plan.subtasks],
                        "attempt": structural_attempt,
                        "max_attempts": max_structural_attempts,
                    })
                    task.add_decision(
                        f"Rejected replanned plan: {validation_error}",
                    )
                    self._state.save(task)
                    if structural_attempt >= max_structural_attempts:
                        return False
                    topology_feedback = (
                        "Previous replanned plan violated contract constraints.\n"
                        f"Validation error: {validation_error}\n"
                        "Preserve all existing subtask IDs and provide valid dependencies."
                    )
                    continue

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
                    "subtask_ids": [s.id for s in new_plan.subtasks],
                })
                return True

            return False

        except Exception as e:
            task.add_error("replanner", str(e))
            self._state.save(task)
            return False

    @staticmethod
    def _validate_replan_contract(
        *,
        current_plan: Plan,
        replanned_plan: Plan,
    ) -> str | None:
        """Ensure replanning preserves prior subtask IDs exactly.

        Replanning may add new subtasks, but it must not drop or rename
        existing IDs. This keeps reconciliation deterministic and avoids
        any remapping logic.
        """
        current_ids = [s.id for s in current_plan.subtasks]
        new_ids = [s.id for s in replanned_plan.subtasks]
        new_id_set = set(new_ids)

        if len(new_ids) != len(new_id_set):
            duplicates: list[str] = []
            seen: set[str] = set()
            for subtask_id in new_ids:
                if subtask_id in seen and subtask_id not in duplicates:
                    duplicates.append(subtask_id)
                seen.add(subtask_id)
            duplicates.sort()
            return "duplicate subtask IDs in replanned plan: " + ", ".join(duplicates)

        missing_ids = sorted(
            subtask_id
            for subtask_id in current_ids
            if subtask_id not in new_id_set
        )
        if missing_ids:
            return "replanned plan dropped existing subtask IDs: " + ", ".join(missing_ids)

        unresolved_deps: list[str] = []
        for subtask in replanned_plan.subtasks:
            bad = sorted(dep for dep in subtask.depends_on if dep not in new_id_set)
            if bad:
                unresolved_deps.append(f"{subtask.id} -> {', '.join(bad)}")
        if unresolved_deps:
            return "replanned plan contains unresolved dependencies: " + "; ".join(unresolved_deps)

        topology_issues = Orchestrator._plan_topology_issues(replanned_plan)
        if topology_issues:
            return "replanned plan has invalid topology: " + "; ".join(topology_issues)

        return None

    def _parse_plan(self, response: ModelResponse, goal: str = "") -> Plan:
        """Parse a plan from the model's JSON response."""
        validation = self._validator.validate_json_response(
            response, expected_keys=["subtasks"]
        )

        if not validation.valid or validation.parsed is None:
            raise ValueError(validation.error or "planner output JSON parse failed")

        subtasks = []
        for s in validation.parsed.get("subtasks", []):
            subtasks.append(Subtask(
                id=s.get("id", f"step-{len(subtasks) + 1}"),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
                model_tier=s.get("model_tier", 1),
                verification_tier=s.get("verification_tier", 1),
                is_critical_path=s.get("is_critical_path", False),
                is_synthesis=s.get("is_synthesis", False),
                acceptance_criteria=s.get("acceptance_criteria", ""),
                max_retries=self._config.execution.max_subtask_retries,
            ))

        return Plan(subtasks=subtasks, version=1)

    def _apply_process_phase_mode(self, plan: Plan) -> Plan:
        """Apply process phase-mode constraints to the planner output."""
        if not self._process or not self._process.phases:
            return plan
        if self._process.phase_mode != "strict":
            return plan

        planner_subtasks = {s.id: s for s in plan.subtasks}
        strict_subtasks: list[Subtask] = []

        for phase in self._process.phases:
            existing = planner_subtasks.get(phase.id)
            if existing is None:
                strict_subtasks.append(
                    Subtask(
                        id=phase.id,
                        description=phase.description,
                        depends_on=list(phase.depends_on),
                        model_tier=phase.model_tier,
                        verification_tier=phase.verification_tier,
                        is_critical_path=phase.is_critical_path,
                        is_synthesis=phase.is_synthesis,
                        acceptance_criteria=phase.acceptance_criteria,
                        max_retries=self._config.execution.max_subtask_retries,
                    )
                )
                continue

            existing.description = phase.description or existing.description
            existing.depends_on = list(phase.depends_on)
            existing.model_tier = phase.model_tier
            existing.verification_tier = phase.verification_tier
            existing.is_critical_path = phase.is_critical_path
            existing.is_synthesis = phase.is_synthesis
            if phase.acceptance_criteria:
                existing.acceptance_criteria = phase.acceptance_criteria
            strict_subtasks.append(existing)

        return Plan(
            subtasks=strict_subtasks,
            version=plan.version,
            last_replanned=plan.last_replanned,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _planning_response_max_tokens(self) -> int | None:
        """Resolve token cap for planner responses.

        A non-positive configured value disables the explicit cap and lets
        providers choose their own default.
        """
        raw = getattr(
            getattr(self._config, "limits", None),
            "planning_response_max_tokens",
            0,
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        return value if value > 0 else None

    def _record_stale_outcome(
        self,
        *,
        task: Task,
        subtask: Subtask,
        outcome_plan_version: int,
    ) -> None:
        """Emit telemetry for outcomes produced by an outdated plan version."""
        self._emit(SUBTASK_OUTCOME_STALE, task.id, {
            "subtask_id": subtask.id,
            "outcome_plan_version": int(outcome_plan_version),
            "current_plan_version": int(task.plan.version),
        })

    def _evidence_for_subtask(self, task_id: str, subtask_id: str) -> list[dict]:
        """Load persisted evidence records scoped to one subtask."""
        try:
            records = self._state.load_evidence_records(task_id)
        except Exception as e:
            logger.warning("Failed loading evidence ledger for %s: %s", task_id, e)
            return []
        scoped: list[dict] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            if str(item.get("subtask_id", "")).strip() != subtask_id:
                continue
            scoped.append(item)
        return scoped

    def _persist_subtask_evidence(
        self,
        task_id: str,
        subtask_id: str,
        evidence_records: list[dict] | None,
    ) -> None:
        """Persist newly captured evidence records."""
        if not evidence_records:
            return
        scoped: list[dict] = []
        for item in evidence_records:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized["subtask_id"] = subtask_id
            normalized.setdefault("task_id", task_id)
            scoped.append(normalized)
        if not scoped:
            return
        try:
            self._state.append_evidence_records(task_id, scoped)
        except Exception as e:
            logger.warning("Failed persisting evidence ledger for %s: %s", task_id, e)

    def _get_changelog(self, task: Task) -> ChangeLog | None:
        """Get or create a ChangeLog for the task's workspace."""
        if not task.workspace:
            return None
        if task.id in self._changelog_cache:
            return self._changelog_cache[task.id]
        workspace = Path(task.workspace)
        data_dir = self._state._data_dir / "tasks" / task.id
        data_dir.mkdir(parents=True, exist_ok=True)
        changelog = ChangeLog(task_id=task.id, workspace=workspace, data_dir=data_dir)
        self._changelog_cache[task.id] = changelog
        return changelog

    @staticmethod
    def _stringify_evidence_csv_value(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)

    @classmethod
    def _evidence_csv_fieldnames(cls, rows: list[dict[str, str]]) -> list[str]:
        base = list(cls._EVIDENCE_LEDGER_CSV_BASE_FIELDS)
        extras: set[str] = set()
        for row in rows:
            for key in row:
                if key and key not in base:
                    extras.add(key)
        return base + sorted(extras)

    def _evidence_csv_rows(self, records: list[dict]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            row: dict[str, str] = {}
            for raw_key, value in record.items():
                key = str(raw_key or "").strip()
                if not key:
                    continue
                row[key] = self._stringify_evidence_csv_value(value)
            if row:
                rows.append(row)
        return rows

    def _export_evidence_ledger_csv(self, task: Task) -> None:
        """Best-effort evidence ledger export to the task workspace."""
        workspace_text = str(task.workspace or "").strip()
        if not workspace_text:
            return
        workspace = Path(workspace_text).expanduser()
        if not workspace.exists() or not workspace.is_dir():
            return
        try:
            records = self._state.load_evidence_records(task.id)
        except Exception as e:
            logger.warning("Failed loading evidence ledger for CSV export %s: %s", task.id, e)
            return
        rows = self._evidence_csv_rows(records)
        if not rows:
            return
        output_path = workspace / self._EVIDENCE_LEDGER_CSV_NAME
        fieldnames = self._evidence_csv_fieldnames(rows)
        try:
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            logger.warning(
                "Failed exporting evidence ledger CSV for %s to %s: %s",
                task.id,
                output_path,
                e,
            )

    def _finalize_task(self, task: Task) -> Task:
        """Finalize task: set status, emit events."""
        completed, total = task.progress
        blocking_remediation_failures: list[str] = []
        blocked_subtasks: list[dict[str, object]] = []
        raw_blocked_subtasks = task.metadata.get("blocked_subtasks")
        if isinstance(raw_blocked_subtasks, list):
            for item in raw_blocked_subtasks:
                if not isinstance(item, dict):
                    continue
                subtask_id = str(item.get("subtask_id", "")).strip()
                raw_reasons = item.get("reasons", [])
                reasons: list[str] = []
                if isinstance(raw_reasons, list):
                    for reason in raw_reasons:
                        text = str(reason).strip()
                        if text:
                            reasons.append(text)
                else:
                    text = str(raw_reasons).strip()
                    if text:
                        reasons.append(text)
                if not subtask_id:
                    continue
                blocked_subtasks.append({
                    "subtask_id": subtask_id,
                    "reasons": reasons,
                })
        queue = task.metadata.get("remediation_queue")
        if isinstance(queue, list):
            for item in queue:
                if not isinstance(item, dict):
                    continue
                if not bool(item.get("blocking", False)):
                    continue
                state = str(item.get("state", "queued")).strip().lower()
                if state != "resolved":
                    subtask_id = str(item.get("subtask_id", "")).strip()
                    terminal_reason = str(item.get("terminal_reason", "")).strip()
                    last_error = str(item.get("last_error", "")).strip()
                    label = subtask_id or str(item.get("id", "")).strip() or "unknown"
                    if terminal_reason:
                        label = f"{label} ({terminal_reason})"
                    elif last_error:
                        label = f"{label} ({last_error})"
                    blocking_remediation_failures.append(
                        label,
                    )

        all_done = (
            completed == total
            and total > 0
            and not blocking_remediation_failures
        )

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
            if blocking_remediation_failures:
                task.add_error(
                    "remediation",
                    "Blocking remediation unresolved for: "
                    + ", ".join(blocking_remediation_failures),
                )
            if blocked_subtasks:
                labels = ", ".join(
                    entry["subtask_id"] for entry in blocked_subtasks
                    if isinstance(entry, dict) and entry.get("subtask_id")
                )
                task.add_error(
                    "scheduler",
                    "Execution stalled with blocked pending subtasks: "
                    + (labels or "unknown"),
                )
            self._emit(TASK_FAILED, task.id, {
                "completed": completed,
                "total": total,
                "failed_subtasks": [s.id for s in failed],
                "blocking_remediation_failures": blocking_remediation_failures,
                "blocked_subtasks": blocked_subtasks,
            })

        self._emit_telemetry_run_summary(task)
        self._state.save(task)
        return task

    async def _learn_from_task(self, task: Task) -> None:
        """Run post-task learning extraction (best-effort)."""
        if self._learning is None:
            return
        try:
            await self._learning.learn_from_task(task)
        except Exception as e:
            logger.warning("Post-task learning failed for %s: %s", task.id, e)

    def _emit(self, event_type: str, task_id: str, data: dict) -> None:
        payload = dict(data or {})
        run_id = str(payload.get("run_id", "") or "").strip()
        if not run_id:
            run_id = str(getattr(self, "_active_run_id", "") or "").strip()
        if run_id and not str(payload.get("run_id", "")).strip():
            payload["run_id"] = run_id
        self._events.emit(Event(
            event_type=event_type,
            task_id=task_id,
            data=payload,
        ))

    @staticmethod
    def _task_run_id(task: Task) -> str:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        return str(metadata.get("run_id", "") or "").strip()

    def _initialize_task_run_id(self, task: Task) -> str:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        run_id = str(metadata.get("run_id", "") or "").strip()
        if not run_id:
            run_id = f"run-{uuid.uuid4().hex[:12]}"
            metadata["run_id"] = run_id
            task.metadata = metadata
            self._state.save(task)
        self._active_run_id = run_id
        return run_id

    def _apply_budget_metadata(
        self,
        task: Task,
        budget_name: str,
        observed: int | float,
        limit: int,
    ) -> None:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["budget_exhausted"] = {
            "name": budget_name,
            "observed": observed,
            "limit": limit,
            "snapshot": self._run_budget.snapshot(),
            "at": datetime.now().isoformat(),
            "run_id": self._task_run_id(task),
        }
        task.metadata = metadata

    async def _enforce_global_budget(self, task: Task) -> bool:
        exceeded, budget_name, observed, limit = self._run_budget.exceeded()
        if not exceeded:
            return False
        self._apply_budget_metadata(task, budget_name, observed, limit)
        for candidate in task.plan.subtasks:
            if candidate.status == SubtaskStatus.PENDING:
                candidate.status = SubtaskStatus.SKIPPED
                candidate.summary = (
                    "Skipped: global run budget exhausted "
                    f"({budget_name})."
                )
        task.status = TaskStatus.FAILED
        task.add_error(
            "budget",
            f"Global run budget exhausted: {budget_name} "
            f"(observed={observed}, limit={limit})",
        )
        self._state.save(task)
        self._emit(TASK_BUDGET_EXHAUSTED, task.id, {
            "run_id": self._task_run_id(task),
            "budget_name": budget_name,
            "observed": observed,
            "limit": limit,
            "snapshot": self._run_budget.snapshot(),
        })
        return True

    @staticmethod
    def _new_telemetry_rollup() -> dict[str, int]:
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

    def _accumulate_subtask_telemetry(self, result: SubtaskResult) -> None:
        counters = getattr(result, "telemetry_counters", None)
        if not isinstance(counters, dict):
            return
        rollup = getattr(self, "_telemetry_rollup", None)
        if not isinstance(rollup, dict):
            self._telemetry_rollup = self._new_telemetry_rollup()
            rollup = self._telemetry_rollup
        for key in (
            "model_invocations",
            "tool_calls",
            "mutating_tool_calls",
            "artifact_ingests",
            "artifact_reads",
            "artifact_retention_deletes",
            "compaction_policy_decisions",
            "overflow_fallback_count",
            "compactor_warning_count",
        ):
            try:
                increment = int(counters.get(key, 0))
            except (TypeError, ValueError):
                increment = 0
            if increment <= 0:
                continue
            rollup[key] = int(rollup.get(key, 0)) + increment

    def _emit_telemetry_run_summary(self, task: Task) -> None:
        runner_limits = getattr(getattr(self._config, "limits", None), "runner", None)
        if not bool(getattr(runner_limits, "enable_artifact_telemetry_events", False)):
            return
        rollup = getattr(self, "_telemetry_rollup", None)
        if not isinstance(rollup, dict):
            rollup = self._new_telemetry_rollup()
        self._emit(TELEMETRY_RUN_SUMMARY, task.id, {
            "run_id": self._task_run_id(task),
            "model_invocations": int(rollup.get("model_invocations", 0)),
            "tool_calls": int(rollup.get("tool_calls", 0)),
            "mutating_tool_calls": int(rollup.get("mutating_tool_calls", 0)),
            "artifact_ingests": int(rollup.get("artifact_ingests", 0)),
            "artifact_reads": int(rollup.get("artifact_reads", 0)),
            "artifact_retention_deletes": int(rollup.get("artifact_retention_deletes", 0)),
            "compaction_policy_decisions": int(rollup.get("compaction_policy_decisions", 0)),
            "overflow_fallback_count": int(rollup.get("overflow_fallback_count", 0)),
            "compactor_warning_count": int(rollup.get("compactor_warning_count", 0)),
            "budget_snapshot": self._run_budget.snapshot(),
        })

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
    metadata: dict | None = None,
) -> Task:
    """Factory for creating new tasks with a generated ID."""
    return Task(
        id=uuid.uuid4().hex[:8],
        goal=goal,
        workspace=workspace,
        approval_mode=approval_mode,
        callback_url=callback_url,
        context=context or {},
        metadata=metadata or {},
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )
