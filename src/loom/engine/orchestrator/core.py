"""Core orchestrator loop.

Drives task execution: plan -> execute subtasks -> verify -> complete.
The model never decides to "continue" — the harness does.

Subtask execution is delegated to SubtaskRunner.  Independent subtasks
(no unmet dependencies) are dispatched in parallel up to
``config.execution.max_parallel_subtasks``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loom.config import Config

if TYPE_CHECKING:
    from loom.processes.schema import IterationPolicy, ProcessDefinition
from loom.engine.iteration_gates import IterationEvaluation, IterationGateEvaluator
from loom.engine.runner import SubtaskResult, SubtaskRunner, ToolCallRecord
from loom.engine.scheduler import Scheduler
from loom.engine.verification import VerificationGates, VerificationResult
from loom.events.bus import EventBus
from loom.events.types import (
    SUBTASK_BLOCKED,
    SUBTASK_OUTCOME_STALE,
    SUBTASK_OUTPUT_CONFLICT_DEFERRED,
    SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING,
    SUBTASK_RETRYING,
    TASK_CANCEL_REQUESTED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PAUSED,
    TASK_PLAN_DEGRADED,
    TASK_PLAN_READY,
    TASK_PLANNING,
    TASK_RESUMED,
    TASK_RUN_ACQUIRED,
    TASK_STALLED,
    VERIFICATION_OUTCOME,
)
from loom.learning.manager import LearningManager
from loom.models.base import ModelResponse
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalManager, ApprovalRequest
from loom.recovery.confidence import ConfidenceScorer
from loom.recovery.questions import QuestionManager
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

from . import budget as orchestrator_budget
from . import dispatch as orchestrator_dispatch
from . import evidence as orchestrator_evidence
from . import output as orchestrator_output
from . import planning as orchestrator_planning
from . import profile as orchestrator_profile
from . import remediation as orchestrator_remediation
from . import runtime as orchestrator_runtime
from . import task_factory as orchestrator_task_factory
from . import telemetry as orchestrator_telemetry
from . import validity as orchestrator_validity
from .budget import _RunBudget

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
    "failure_class",
    "missing_targets",
    "placeholder_finding_count",
    "placeholder_findings",
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
_CLAIM_TERMINAL_UNRESOLVED = frozenset({
    "contradicted",
    "insufficient_evidence",
    "extracted",
    "stale",
})
_CLAIM_REASON_CODES = {
    "supported": "claim_supported",
    "contradicted": "claim_contradicted",
    "insufficient_evidence": "claim_insufficient_evidence",
    "stale": "claim_stale_source",
    "pruned": "claim_pruned",
}
_CLAIM_RECOVERABLE_FAILURE_CODES = frozenset({
    "recommendation_unconfirmed",
    "unconfirmed_noncritical",
    "unconfirmed_critical_path",
    "claim_insufficient_evidence",
    "claim_contradicted",
    "claim_stale_source",
    "coverage_below_threshold",
})
_SCOPE_ADAPTIVE_REPLAN_REASON_CODES = frozenset({
    "insufficient_sample_size",
    "insufficient_volume",
    "cardinality_mismatch",
    "incomplete_verification_pending_phase2",
})
_PLACEHOLDER_UNCONFIRMED_REASON_CODES = frozenset({
    "incomplete_deliverable_placeholder",
    "incomplete_deliverable_content",
})
_PLACEHOLDER_PREPASS_MODE = "deterministic_placeholder_prepass"

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

    When a process definition is provided, domain intelligence is injected
    into all prompts (persona, phases, tool guidance, verification rules,
    memory guidance) without changing the engine's control flow.

    Subtask execution (tool loop, verification, memory extraction)
    is delegated to SubtaskRunner.
    """
    _OUTPUT_CONFLICT_STARVATION_THRESHOLD = 3
    _OUTPUT_ROLE_WORKER = "worker"
    _OUTPUT_ROLE_PHASE_FINALIZER = "phase_finalizer"

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
        question_manager: QuestionManager | None = None,
        learning_manager: LearningManager | None = None,
        process: ProcessDefinition | None = None,
        snapshot_mirror_writer: Callable[[Task], Awaitable[None]] | None = None,
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
        self._snapshot_mirror_writer = snapshot_mirror_writer
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
        ask_user_runtime_enabled = bool(
            getattr(config.execution, "ask_user_runtime_blocking_enabled", False),
        )
        ask_user_durable_state_enabled = bool(
            getattr(config.execution, "ask_user_durable_state_enabled", False),
        )
        if question_manager is not None:
            self._question = question_manager
        elif ask_user_runtime_enabled and ask_user_durable_state_enabled:
            self._question = QuestionManager(event_bus, memory_manager)
        else:
            self._question = None
        self._retry = RetryManager(
            max_retries=config.execution.max_subtask_retries,
        )
        self._state_lock = asyncio.Lock()
        self._changelog_cache: dict[str, ChangeLog] = {}
        self._telemetry_rollup: dict[str, int] = self._new_telemetry_rollup()
        self._emitted_telemetry_summary_runs: set[str] = set()
        self._run_budget = _RunBudget(config)
        self._active_run_id = ""
        self._iteration_enabled = bool(
            getattr(self._config.execution, "enable_process_iteration_loops", False),
        )
        self._iteration_gates = IterationGateEvaluator(
            command_allowlisted_prefixes=list(
                getattr(
                    self._config.execution,
                    "iteration_command_exit_allowlisted_prefixes",
                    [],
                ) or [],
            ),
            enable_command_exit=bool(
                getattr(
                    self._config.execution,
                    "enable_iteration_command_exit_gate",
                    False,
                ),
            ),
        )

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
            question_manager=self._question,
            task_snapshot_writer=self._save_task_state,
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
            run_id = await self._initialize_task_run_id_async(task)
            self._emit(TASK_RUN_ACQUIRED, task.id, {
                "run_id": run_id,
            })
            if bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
                await self._hydrate_remediation_queue_from_db(task)
            if self._iteration_enabled:
                await self._reconcile_iteration_state(task)
            plan: Plan
            if reuse_existing_plan and task.plan and task.plan.subtasks:
                plan = task.plan
                plan = self._prepare_plan_for_execution(
                    task=task,
                    plan=plan,
                    context="reused_plan",
                )
                task.plan = plan
                reused_plan_replanned = False
                if await self._should_replan_resumed_existing_plan(task):
                    self._run_budget.observe_replan()
                    reused_plan_replanned = await self._replan_task(
                        task,
                        reason="resume_existing_plan_context_shift",
                        verification_feedback=self._build_resume_existing_plan_feedback(
                            task,
                        ),
                    )
                    if reused_plan_replanned and task.plan and task.plan.subtasks:
                        plan = task.plan
                task.status = TaskStatus.EXECUTING
                await self._save_task_state(task)
                if not reused_plan_replanned:
                    self._emit(TASK_PLAN_READY, task.id, {
                        "subtask_count": len(plan.subtasks),
                        "subtask_ids": [s.id for s in plan.subtasks],
                        "reused": True,
                        "run_id": run_id,
                    })
            else:
                # 1. Planning phase
                task.status = TaskStatus.PLANNING
                await self._save_task_state(task)
                self._emit(TASK_PLANNING, task.id, {
                    "goal": task.goal,
                    "run_id": run_id,
                })

                plan = await self._plan_task_with_validation(task)
                task.plan = plan
                task.status = TaskStatus.EXECUTING
                await self._save_task_state(task)
                self._emit(TASK_PLAN_READY, task.id, {
                    "subtask_count": len(plan.subtasks),
                    "subtask_ids": [s.id for s in plan.subtasks],
                    "run_id": run_id,
                })

            await self._reconcile_subtask_policy_state(task)

            # 2. Execution loop — parallel dispatch of independent subtasks
            self._emit(TASK_EXECUTING, task.id, {"run_id": run_id})
            iteration = 0
            max_iterations = self._config.execution.max_loop_iterations
            max_parallel = self._config.execution.max_parallel_subtasks
            attempts_by_subtask: dict[str, list[AttemptRecord]] = {}
            output_conflict_tracker: dict[str, dict[str, object]] = {}
            output_conflict_sequence = 1
            pending_replan: dict[str, str | None] | None = None
            stall_recovery_attempts = 0
            max_stall_recovery_attempts = 2

            while self._scheduler.has_pending(task.plan) and iteration < max_iterations:
                await self._sync_external_control_state(task)
                if await self._enforce_global_budget(task):
                    break
                if task.status == TaskStatus.PAUSED:
                    await asyncio.sleep(0.25)
                    continue
                if task.status == TaskStatus.CANCELLED:
                    break

                # Get all runnable subtasks (dependencies met)
                runnable = self._scheduler.runnable_subtasks(task.plan)
                if not runnable:
                    blocked_subtasks = self._blocked_pending_subtasks(task.plan)
                    for blocked in blocked_subtasks:
                        if not isinstance(blocked, dict):
                            continue
                        raw_reasons = blocked.get("reasons", [])
                        if isinstance(raw_reasons, list):
                            reasons = [str(reason) for reason in raw_reasons]
                        else:
                            text_reason = str(raw_reasons).strip()
                            reasons = [text_reason] if text_reason else []
                        self._emit(SUBTASK_BLOCKED, task.id, {
                            "subtask_id": str(blocked.get("subtask_id", "") or "").strip(),
                            "reasons": reasons,
                        })
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

                # Cap to max_parallel_subtasks.
                # Additional guard: do not execute subtasks in parallel when they
                # target overlapping canonical deliverable paths.
                conflict_deferrals: list[dict[str, object]] = []
                if self._output_enforce_single_writer():
                    batch, conflict_deferrals, output_conflict_sequence = (
                        self._select_conflict_safe_batch(
                            task=task,
                            runnable=runnable,
                            max_parallel=max_parallel,
                            conflict_tracker=output_conflict_tracker,
                            sequence_counter=output_conflict_sequence,
                        )
                    )
                    if not batch:
                        batch = runnable[:max_parallel]
                else:
                    batch = runnable[:max_parallel]
                    output_conflict_tracker.clear()
                    output_conflict_sequence = 1

                if conflict_deferrals and self._output_conflict_policy() == "fail_fast":
                    first = conflict_deferrals[0]
                    first_id = str(first.get("subtask_id", "") or "").strip()
                    conflicting_paths = list(first.get("conflicting_paths", []))
                    conflicting_with = list(first.get("conflicting_with", []))
                    raise RuntimeError(
                        "Output conflict policy fail_fast blocked dispatch for "
                        f"subtask '{first_id}' due to overlapping canonical paths "
                        f"{conflicting_paths} with {conflicting_with}.",
                    )
                for deferred in conflict_deferrals:
                    subtask_id = str(deferred.get("subtask_id", "")).strip()
                    if not subtask_id:
                        continue
                    self._emit(SUBTASK_OUTPUT_CONFLICT_DEFERRED, task.id, {
                        "subtask_id": subtask_id,
                        "phase_id": str(deferred.get("phase_id", "")).strip(),
                        "conflicting_paths": list(deferred.get("conflicting_paths", [])),
                        "conflicting_with": list(deferred.get("conflicting_with", [])),
                        "deferral_streak": int(deferred.get("deferral_streak", 0) or 0),
                        "deferral_count": int(deferred.get("deferral_count", 0) or 0),
                    })
                    if bool(deferred.get("starvation_warning", False)):
                        self._emit(SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING, task.id, {
                            "subtask_id": subtask_id,
                            "phase_id": str(deferred.get("phase_id", "")).strip(),
                            "deferral_streak": int(
                                deferred.get("deferral_streak", 0) or 0,
                            ),
                            "threshold": int(
                                deferred.get("starvation_threshold", 0) or 0,
                            ),
                            "conflicting_paths": list(
                                deferred.get("conflicting_paths", []),
                            ),
                            "conflicting_with": list(
                                deferred.get("conflicting_with", []),
                            ),
                        })
                iteration += 1
                batch_plan_version = task.plan.version

                # Dispatch batch
                if len(batch) == 1:
                    # Single subtask — no gather overhead
                    try:
                        outcomes = [await self._dispatch_subtask(
                            task, batch[0], attempts_by_subtask,
                        )]
                    except asyncio.CancelledError:
                        raise
                    except Exception as item:
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
                        if isinstance(item, asyncio.CancelledError):
                            raise item
                        if isinstance(item, Exception):
                            outcomes.append(
                                self._build_subtask_exception_outcome(
                                    batch[i],
                                    item,
                                ),
                            )
                        elif isinstance(item, BaseException):
                            raise item
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
                    self._observe_iteration_runner_invocation(subtask)
                    self._observe_iteration_runtime_usage(
                        task=task,
                        subtask=subtask,
                        result=result,
                    )
                    if result.status == "failed":
                        replan_request = await self._handle_failure(
                            task, subtask, result, verification,
                            attempts_by_subtask,
                        )
                        if replan_request is not None and pending_replan is None:
                            pending_replan = replan_request
                    else:
                        replan_request = await self._handle_iteration_after_success(
                            task=task,
                            subtask=subtask,
                            result=result,
                            verification=verification,
                        )
                        if replan_request is not None and pending_replan is None:
                            pending_replan = replan_request

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
            result_task = await self._finalize_task_async(task)
            self._export_evidence_ledger_csv(result_task)

            # 4. Learn from execution (best-effort)
            await self._learn_from_task(result_task)

            return result_task

        except Exception as e:
            logger.exception("Fatal error in task %s", task.id)
            task.status = TaskStatus.FAILED
            task.add_error("orchestrator", f"{type(e).__name__}: {e}")
            try:
                await self._save_task_state(task)
            except Exception as save_err:
                logger.error("Failed to save after fatal: %s", save_err)
            self._emit(TASK_FAILED, task.id, {
                "error": str(e),
                "error_type": type(e).__name__,
                "reason": "uncaught_exception",
                "outcome": "failed",
            })
            self._emit_telemetry_run_summary(task)
            self._export_evidence_ledger_csv(task)
            await self._learn_from_task(task)
            return task

    def _build_subtask_exception_outcome(
        self,
        subtask: Subtask,
        error: BaseException,
    ) -> tuple[Subtask, SubtaskResult, VerificationResult]:
        return orchestrator_dispatch._build_subtask_exception_outcome(self, subtask, error)

    def _validity_contract_for_subtask(self, subtask: Subtask) -> dict[str, object]:
        return orchestrator_validity._validity_contract_for_subtask(self, subtask)

    def _synthesis_verification_floor(self, subtask: Subtask) -> int:
        return orchestrator_validity._synthesis_verification_floor(self, subtask)

    @staticmethod
    def _tool_call_succeeded(call: ToolCallRecord) -> bool:
        return orchestrator_validity._tool_call_succeeded(call)

    def _fact_checker_used(self, tool_calls: list[ToolCallRecord]) -> bool:
        return orchestrator_validity._fact_checker_used(self, tool_calls)

    def _fact_checker_verdict_count(self, tool_calls: list[ToolCallRecord]) -> int:
        return orchestrator_validity._fact_checker_verdict_count(self, tool_calls)

    def _requires_fact_checker_for_subtask(self, subtask: Subtask) -> bool:
        return orchestrator_validity._requires_fact_checker_for_subtask(self, subtask)

    def _claim_graph_state(self, task: Task) -> dict[str, object]:
        return orchestrator_validity._claim_graph_state(self, task)

    def _update_claim_graph_from_verification(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
    ) -> None:
        return orchestrator_validity._update_claim_graph_from_verification(
            self,
            task=task,
            subtask=subtask,
            verification=verification,
        )

    @staticmethod
    def _claims_from_verification(verification: VerificationResult) -> list[dict[str, object]]:
        return orchestrator_validity._claims_from_verification(verification)

    @staticmethod
    def _normalize_claim_reason_code(status: str, reason_code: str) -> str:
        return orchestrator_validity._normalize_claim_reason_code(status, reason_code)

    @staticmethod
    def _claim_counts(claims: list[dict[str, object]]) -> dict[str, int]:
        return orchestrator_validity._claim_counts(claims)

    @staticmethod
    def _claim_ratios(counts: dict[str, int]) -> dict[str, float]:
        return orchestrator_validity._claim_ratios(counts)

    @staticmethod
    def _assertions_from_verification(
        verification: VerificationResult,
    ) -> list[orchestrator_validity.AssertionEnvelope]:
        return orchestrator_validity._assertions_from_verification(verification)

    @staticmethod
    def _runtime_assertions_from_tool_calls(
        *,
        subtask: Subtask,
        tool_calls: list[ToolCallRecord] | None,
    ) -> list[orchestrator_validity.AssertionEnvelope]:
        return orchestrator_validity._runtime_assertions_from_tool_calls(
            subtask=subtask,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _assertion_counts(
        assertions: list[orchestrator_validity.AssertionEnvelope],
    ) -> dict[str, int]:
        return orchestrator_validity._assertion_counts(assertions)

    def _attach_runtime_assertions(
        self,
        *,
        subtask: Subtask,
        verification: VerificationResult,
        tool_calls: list[ToolCallRecord] | None,
    ) -> VerificationResult:
        return orchestrator_validity._attach_runtime_assertions(
            self,
            subtask=subtask,
            verification=verification,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _verification_with_metadata(
        verification: VerificationResult,
        *,
        metadata: dict[str, object],
        passed: bool | None = None,
        outcome: str | None = None,
        reason_code: str | None = None,
        feedback: str | None = None,
        severity_class: str | None = None,
        confidence: float | None = None,
    ) -> VerificationResult:
        return orchestrator_validity._verification_with_metadata(
            verification,
            metadata=metadata,
            passed=passed,
            outcome=outcome,
            reason_code=reason_code,
            feedback=feedback,
            severity_class=severity_class,
            confidence=confidence,
        )

    def _apply_intermediate_claim_pruning(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
        contract: dict[str, object],
    ) -> VerificationResult:
        return orchestrator_validity._apply_intermediate_claim_pruning(
            self,
            task=task,
            subtask=subtask,
            result=result,
            verification=verification,
            contract=contract,
        )

    @staticmethod
    def _parse_temporal_date_token(value: object) -> datetime | None:
        return orchestrator_validity._parse_temporal_date_token(value)

    @classmethod
    def _extract_temporal_dates_from_text(cls, text: str) -> list[datetime]:
        return orchestrator_validity._extract_temporal_dates_from_text(cls, text)

    @classmethod
    def _claim_temporal_scope(cls, claim: dict[str, object]) -> dict[str, object]:
        return orchestrator_validity._claim_temporal_scope(cls, claim)

    @staticmethod
    def _temporal_claim_key(text: str) -> str:
        return orchestrator_validity._temporal_claim_key(text)

    def _enforce_temporal_consistency_gate(
        self,
        *,
        subtask: Subtask,
        verification: VerificationResult,
        contract: dict[str, object],
    ) -> VerificationResult:
        return orchestrator_validity._enforce_temporal_consistency_gate(
            self,
            subtask=subtask,
            verification=verification,
            contract=contract,
        )

    def _enforce_synthesis_claim_gate(
        self,
        *,
        subtask: Subtask,
        verification: VerificationResult,
        contract: dict[str, object],
    ) -> VerificationResult:
        return orchestrator_validity._enforce_synthesis_claim_gate(
            self,
            subtask=subtask,
            verification=verification,
            contract=contract,
        )

    @staticmethod
    def _artifact_provenance_evidence(
        *,
        task_id: str,
        subtask_id: str,
        tool_calls: list[ToolCallRecord] | None,
        existing_ids: set[str],
        workspace: Path | None = None,
    ) -> list[dict[str, object]]:
        return orchestrator_validity._artifact_provenance_evidence(
            task_id=task_id,
            subtask_id=subtask_id,
            tool_calls=tool_calls,
            existing_ids=existing_ids,
            workspace=workspace,
        )

    @staticmethod
    def _claim_evidence_links(
        *,
        claims: list[dict[str, object]],
        evidence_records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return orchestrator_validity._claim_evidence_links(
            claims=claims,
            evidence_records=evidence_records,
        )

    async def _persist_claim_validity_artifacts(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
        evidence_records: list[dict],
        tool_calls: list[ToolCallRecord] | None = None,
    ) -> None:
        return await orchestrator_validity._persist_claim_validity_artifacts(
            self,
            task=task,
            subtask=subtask,
            verification=verification,
            evidence_records=evidence_records,
            tool_calls=tool_calls,
        )

    def _verified_context_for_synthesis(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification_profile: str = "hybrid",
    ) -> tuple[bool, str, str]:
        return orchestrator_validity._verified_context_for_synthesis(
            self,
            task=task,
            subtask=subtask,
            verification_profile=verification_profile,
        )

    def _resolve_verification_profile(
        self,
        *,
        task: Task,
        subtask: Subtask,
        tool_calls: list | None = None,
    ) -> orchestrator_profile.VerificationProfileResolution:
        return orchestrator_profile.resolve_verification_profile(
            task=task,
            subtask=subtask,
            process=self._process,
            tool_calls=tool_calls,
        )

    def _enforce_required_fact_checker(
        self,
        *,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
    ) -> VerificationResult:
        return orchestrator_validity._enforce_required_fact_checker(
            self,
            subtask=subtask,
            result=result,
            verification=verification,
        )

    # ------------------------------------------------------------------
    # Subtask dispatch
    # ------------------------------------------------------------------

    async def _dispatch_subtask(
        self,
        task: Task,
        subtask: Subtask,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
    ) -> tuple[SubtaskResult, VerificationResult]:
        return await orchestrator_dispatch.dispatch_subtask(
            self,
            task,
            subtask,
            attempts_by_subtask,
        )

    # ------------------------------------------------------------------
    # Outcome handlers
    # ------------------------------------------------------------------

    def _phase_iteration_policy(self, subtask: Subtask) -> IterationPolicy | None:
        return orchestrator_dispatch._phase_iteration_policy(self, subtask)

    def _iteration_retry_mode(self, subtask: Subtask) -> tuple[bool, str]:
        return orchestrator_dispatch.iteration_retry_mode(self, subtask)

    def _observe_iteration_runner_invocation(self, subtask: Subtask) -> None:
        orchestrator_dispatch.observe_iteration_runner_invocation(self, subtask)

    def _observe_iteration_runtime_usage(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
    ) -> None:
        orchestrator_dispatch.observe_iteration_runtime_usage(
            self,
            task=task,
            subtask=subtask,
            result=result,
        )

    def _iteration_runtime_entry(self, task: Task, subtask_id: str) -> dict[str, object]:
        return orchestrator_dispatch._iteration_runtime_entry(self, task, subtask_id)

    def _update_iteration_runtime(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
    ) -> dict[str, object]:
        return orchestrator_dispatch._update_iteration_runtime(
            self,
            task=task,
            subtask=subtask,
            result=result,
        )

    async def _sync_external_control_state(self, task: Task) -> None:
        return await orchestrator_dispatch._sync_external_control_state(self, task)

    @staticmethod
    def _iteration_budget_snapshot(
        *,
        policy: IterationPolicy,
        runtime: dict[str, object],
    ) -> dict[str, object]:
        return orchestrator_dispatch._iteration_budget_snapshot(policy=policy, runtime=runtime)

    @staticmethod
    def _iteration_budget_exhausted_reason(
        *,
        policy: IterationPolicy,
        runtime: dict[str, object],
    ) -> str:
        return orchestrator_dispatch._iteration_budget_exhausted_reason(
            policy=policy,
            runtime=runtime,
        )

    @staticmethod
    def _format_iteration_gate_failures(
        failures: list[object],
    ) -> str:
        return orchestrator_dispatch._format_iteration_gate_failures(failures)

    def _iteration_replan_cap(self, policy: IterationPolicy) -> int:
        return orchestrator_dispatch._iteration_replan_cap(self, policy)

    @staticmethod
    def _iteration_exhaustion_fingerprint(
        *,
        subtask: Subtask,
        terminal_reason: str,
        gate_summary: str,
    ) -> str:
        return orchestrator_dispatch._iteration_exhaustion_fingerprint(
            subtask=subtask,
            terminal_reason=terminal_reason,
            gate_summary=gate_summary,
        )

    async def _request_iteration_replan(
        self,
        *,
        task: Task,
        subtask: Subtask,
        policy: IterationPolicy,
        terminal_reason: str,
        gate_summary: str,
    ) -> dict[str, str | None] | None:
        return await orchestrator_dispatch._request_iteration_replan(
            self,
            task=task,
            subtask=subtask,
            policy=policy,
            terminal_reason=terminal_reason,
            gate_summary=gate_summary,
        )

    async def _persist_iteration_evaluation(
        self,
        *,
        task: Task,
        subtask: Subtask,
        policy: IterationPolicy,
        evaluation: IterationEvaluation | None,
        attempt_index: int,
        status: str,
        gate_summary: str,
        budget_snapshot: dict[str, object],
        terminal_reason: str = "",
        exhaustion_fingerprint: str = "",
    ) -> None:
        return await orchestrator_dispatch._persist_iteration_evaluation(
            self,
            task=task,
            subtask=subtask,
            policy=policy,
            evaluation=evaluation,
            attempt_index=attempt_index,
            status=status,
            gate_summary=gate_summary,
            budget_snapshot=budget_snapshot,
            terminal_reason=terminal_reason,
            exhaustion_fingerprint=exhaustion_fingerprint,
        )

    async def _handle_iteration_after_success(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
    ) -> tuple[bool, str]:
        return await orchestrator_dispatch.handle_iteration_after_success(
            self,
            task=task,
            subtask=subtask,
            result=result,
            verification=verification,
        )

    async def _reconcile_iteration_state(self, task: Task) -> None:
        return await orchestrator_dispatch._reconcile_iteration_state(self, task)

    async def _reconcile_subtask_policy_state(self, task: Task) -> None:
        return await orchestrator_dispatch._reconcile_subtask_policy_state(self, task)

    async def _handle_failure(
        self,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
    ) -> dict[str, str | None] | None:
        return await orchestrator_dispatch.handle_failure(
            self,
            task,
            subtask,
            result,
            verification,
            attempts_by_subtask,
        )

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
            await self._save_task_state(task)

    def _phase_mode(self) -> str:
        return orchestrator_planning.phase_mode(self)

    def _topology_retry_attempts(self) -> int:
        return orchestrator_planning.topology_retry_attempts(self)

    def _planner_degraded_mode(self) -> str:
        return orchestrator_planning.planner_degraded_mode(self)

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
                await self._save_task_state(task)
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
                await self._save_task_state(task)
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
        return await orchestrator_planning.plan_task_with_validation(self, task)

    def _prepare_plan_for_execution(
        self,
        *,
        task: Task,
        plan: Plan,
        context: str,
    ) -> Plan:
        return orchestrator_planning._prepare_plan_for_execution(
            self,
            task=task,
            plan=plan,
            context=context,
        )

    def _annotate_subtask_phase_ids(self, *, task: Task, plan: Plan) -> None:
        return orchestrator_planning._annotate_subtask_phase_ids(self, task=task, plan=plan)

    def _phase_output_strategy(self, phase_id: str) -> str:
        return orchestrator_planning._phase_output_strategy(self, phase_id)

    def _phase_finalizer_id(self, phase_id: str) -> str:
        return orchestrator_planning._phase_finalizer_id(self, phase_id)

    def _subtask_output_role(self, subtask: Subtask) -> str:
        return orchestrator_planning._subtask_output_role(self, subtask)

    def _align_plan_output_coordination(
        self,
        *,
        plan: Plan,
    ) -> tuple[Plan, list[dict[str, object]]]:
        return orchestrator_planning._align_plan_output_coordination(self, plan=plan)

    @staticmethod
    def _hash_validity_contract(contract: dict[str, object]) -> str:
        return orchestrator_validity.hash_validity_contract(contract)

    @staticmethod
    def _to_bool(value: object, default: bool = False) -> bool:
        return orchestrator_validity.to_bool(value, default)

    @staticmethod
    def _to_ratio(value: object, default: float) -> float:
        return orchestrator_validity.to_ratio(value, default)

    @staticmethod
    def _to_non_negative_int(value: object, default: int) -> int:
        return orchestrator_validity.to_non_negative_int(value, default)

    @classmethod
    def _normalize_validity_contract(
        cls,
        contract: dict[str, object] | None,
    ) -> dict[str, object]:
        del cls  # retained for compatibility during extraction
        return orchestrator_validity.normalize_validity_contract(contract)

    def _default_validity_contract_for_subtask(self, subtask: Subtask) -> dict[str, object]:
        return self._normalize_validity_contract({
            "enabled": False,
            "claim_extraction": {"enabled": False},
            "critical_claim_types": ["numeric", "date", "entity_fact"],
            "min_supported_ratio": 0.75,
            "max_unverified_ratio": 0.25,
            "max_contradicted_count": 0,
            "prune_mode": "drop",
            "require_fact_checker_for_synthesis": False,
            "final_gate": {
                "enforce_verified_context_only": bool(subtask.is_synthesis),
                "synthesis_min_verification_tier": 2 if subtask.is_synthesis else 1,
                "critical_claim_support_ratio": 1.0,
                "temporal_consistency": {
                    "enabled": False,
                    "require_as_of_alignment": False,
                    "enforce_cross_claim_date_conflict_check": False,
                    "max_source_age_days": 0,
                    "as_of": "",
                },
            },
        })

    def _resolve_subtask_validity_contract(
        self,
        *,
        subtask: Subtask,
    ) -> dict[str, object]:
        process = self._process
        if process is None:
            return self._default_validity_contract_for_subtask(subtask)
        resolver = getattr(process, "resolve_validity_contract_for_phase", None)
        if callable(resolver):
            phase_hint = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
            try:
                resolved = resolver(phase_hint, is_synthesis=bool(subtask.is_synthesis))
            except TypeError:
                resolved = resolver(phase_hint)
            if isinstance(resolved, dict):
                return self._normalize_validity_contract(resolved)
        return self._default_validity_contract_for_subtask(subtask)

    def _ensure_subtask_validity_snapshot(self, *, subtask: Subtask) -> None:
        if (
            isinstance(subtask.validity_contract_snapshot, dict)
            and subtask.validity_contract_snapshot
        ):
            normalized = self._normalize_validity_contract(subtask.validity_contract_snapshot)
        else:
            normalized = self._resolve_subtask_validity_contract(subtask=subtask)
        subtask.validity_contract_snapshot = normalized
        subtask.validity_contract_hash = self._hash_validity_contract(normalized)

    def _apply_subtask_policy_from_process_phase(
        self,
        *,
        subtask: Subtask,
        phase: object | None,
    ) -> None:
        if phase is not None:
            subtask.model_tier = max(
                int(getattr(subtask, "model_tier", 1) or 1),
                int(getattr(phase, "model_tier", 1) or 1),
            )
            subtask.verification_tier = max(
                int(getattr(subtask, "verification_tier", 1) or 1),
                int(getattr(phase, "verification_tier", 1) or 1),
            )
            if not str(subtask.acceptance_criteria or "").strip():
                subtask.acceptance_criteria = str(
                    getattr(phase, "acceptance_criteria", "") or "",
                ).strip()
        resolved = self._resolve_subtask_validity_contract(subtask=subtask)
        subtask.validity_contract_snapshot = resolved
        subtask.validity_contract_hash = self._hash_validity_contract(resolved)

    @staticmethod
    def _normalize_non_terminal_synthesis(
        plan: Plan,
    ) -> tuple[Plan, list[dict[str, object]]]:
        return orchestrator_planning._normalize_non_terminal_synthesis(plan)

    @classmethod
    def _plan_topology_issues(cls, plan: Plan) -> list[str]:
        return orchestrator_planning._plan_topology_issues(cls, plan)

    @staticmethod
    def _detect_dependency_cycle(
        adjacency: dict[str, list[str]],
    ) -> list[str] | None:
        return orchestrator_planning._detect_dependency_cycle(adjacency)

    @staticmethod
    def _format_blocked_subtasks_feedback(blocked_subtasks: list[dict[str, object]]) -> str:
        return orchestrator_planning._format_blocked_subtasks_feedback(blocked_subtasks)

    def _blocked_pending_subtasks(self, plan: Plan) -> list[dict[str, object]]:
        return orchestrator_planning._blocked_pending_subtasks(self, plan)

    async def _attempt_stalled_recovery(
        self,
        *,
        task: Task,
        blocked_subtasks: list[dict[str, object]],
        attempt: int,
    ) -> bool:
        return await orchestrator_planning._attempt_stalled_recovery(
            self,
            task=task,
            blocked_subtasks=blocked_subtasks,
            attempt=attempt,
        )

    def _critical_path_behavior(self) -> str:
        return orchestrator_remediation._critical_path_behavior(self)

    @staticmethod
    def _is_hard_invariant_failure(verification: VerificationResult | None) -> bool:
        return orchestrator_remediation._is_hard_invariant_failure(verification)

    @staticmethod
    def _normalize_missing_targets(raw: object) -> list[str]:
        return orchestrator_validity.normalize_missing_targets(raw)

    @staticmethod
    def _to_int_or_none(value: object) -> int | None:
        return orchestrator_validity.to_int_or_none(value)

    @staticmethod
    def _to_ratio_or_none(value: object) -> float | None:
        return orchestrator_validity.to_ratio_or_none(value)

    @staticmethod
    def _to_float_or_none(value: object) -> float | None:
        return orchestrator_validity.to_float_or_none(value)

    @classmethod
    def _extract_unconfirmed_metadata(
        cls,
        verification: VerificationResult,
    ) -> dict[str, object]:
        return orchestrator_remediation._extract_unconfirmed_metadata(cls, verification)

    @staticmethod
    def _is_placeholder_unconfirmed_failure(
        *,
        verification: VerificationResult | None,
        placeholder_metadata: dict[str, object] | None = None,
    ) -> bool:
        return orchestrator_remediation._is_placeholder_unconfirmed_failure(
            verification=verification,
            placeholder_metadata=placeholder_metadata,
        )

    @classmethod
    def _normalize_placeholder_findings(
        cls,
        findings: object,
        *,
        max_items: int = 120,
    ) -> list[dict[str, object]]:
        del cls  # retained for compatibility during extraction
        return orchestrator_validity.normalize_placeholder_findings(
            findings,
            max_items=max_items,
        )

    @staticmethod
    def _normalize_workspace_relpath(workspace: Path, raw_path: str) -> str | None:
        return orchestrator_validity.normalize_workspace_relpath(workspace, raw_path)

    @staticmethod
    def _apply_deterministic_placeholder_prune_actions(
        *,
        workspace: Path,
        findings: list[dict[str, object]],
        replacement_token: str = "UNSUPPORTED_NO_EVIDENCE",
    ) -> dict[str, object]:
        return orchestrator_remediation._apply_deterministic_placeholder_prune_actions(
            workspace=workspace,
            findings=findings,
            replacement_token=replacement_token,
        )

    @staticmethod
    def _summarize_placeholder_resolution_state(
        *,
        workspace: Path,
        findings: list[dict[str, object]],
        replacement_token: str = "UNSUPPORTED_NO_EVIDENCE",
    ) -> dict[str, int]:
        return orchestrator_remediation._summarize_placeholder_resolution_state(
            workspace=workspace,
            findings=findings,
            replacement_token=replacement_token,
        )

    async def _run_deterministic_placeholder_prepass(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult | None = None,
        placeholder_metadata: dict[str, object] | None = None,
        origin: str = "unconfirmed_data_retry",
        attempt_number: int | None = None,
        max_attempts: int | None = None,
    ) -> tuple[bool, str, dict[str, object]]:
        return await orchestrator_remediation._run_deterministic_placeholder_prepass(
            self,
            task=task,
            subtask=subtask,
            verification=verification,
            placeholder_metadata=placeholder_metadata,
            origin=origin,
            attempt_number=attempt_number,
            max_attempts=max_attempts,
        )

    def _remediation_queue_limits(self) -> tuple[int, float, float]:
        return orchestrator_remediation.remediation_queue_limits(self)

    @staticmethod
    def _bounded_remediation_backoff_seconds(
        *,
        base_backoff_seconds: float,
        max_backoff_seconds: float,
        attempt_count: int,
    ) -> float:
        return orchestrator_remediation.bounded_remediation_backoff_seconds(
            base_backoff_seconds=base_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            attempt_count=attempt_count,
        )

    def _apply_unconfirmed_follow_up_success(
        self,
        *,
        result: SubtaskResult,
        verification: VerificationResult,
        note: str,
        default_reason_code: str,
    ) -> None:
        return orchestrator_remediation._apply_unconfirmed_follow_up_success(
            self,
            result=result,
            verification=verification,
            note=note,
            default_reason_code=default_reason_code,
        )

    async def _queue_remediation_work_item(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
        strategy: RetryStrategy,
        blocking: bool,
    ) -> None:
        await orchestrator_remediation.queue_remediation_work_item(
            self,
            task=task,
            subtask=subtask,
            verification=verification,
            strategy=strategy,
            blocking=blocking,
        )

    @staticmethod
    def _resolution_plan_items(raw: object, *, max_items: int = 8) -> list[str]:
        return orchestrator_remediation._resolution_plan_items(raw, max_items=max_items)

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
        return orchestrator_remediation._compact_failure_resolution_metadata_value(
            value,
            depth=depth,
            max_depth=max_depth,
            max_list_items=max_list_items,
            max_dict_items=max_dict_items,
            max_text_chars=max_text_chars,
        )

    @classmethod
    def _summarize_failure_resolution_metadata(
        cls,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        return orchestrator_remediation._summarize_failure_resolution_metadata(cls, metadata)

    def _failure_resolution_metadata_char_budget(self) -> int:
        return orchestrator_remediation._failure_resolution_metadata_char_budget(self)

    def _format_failure_resolution_plan(self, response: ModelResponse) -> str:
        return orchestrator_remediation._format_failure_resolution_plan(self, response)

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
        return orchestrator_remediation._build_failure_resolution_prompt(
            self,
            subtask=subtask,
            result=result,
            verification=verification,
            strategy=strategy,
            missing_targets=missing_targets,
            prior_attempts=prior_attempts,
        )

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
        return await orchestrator_remediation._plan_failure_resolution(
            self,
            task=task,
            subtask=subtask,
            result=result,
            verification=verification,
            strategy=strategy,
            missing_targets=missing_targets,
            prior_attempts=prior_attempts,
        )

    def _build_remediation_retry_context(
        self,
        *,
        strategy: RetryStrategy,
        reason_code: str = "",
    ) -> str:
        return orchestrator_remediation._build_remediation_retry_context(
            self,
            strategy=strategy,
            reason_code=reason_code,
        )

    async def _run_confirm_or_prune_remediation(
        self,
        *,
        task: Task,
        subtask: Subtask,
        attempts: list[AttemptRecord],
        remediation_id: str | None = None,
        verification: VerificationResult | None = None,
        placeholder_metadata: dict | None = None,
    ) -> tuple[bool, str]:
        return await orchestrator_remediation.run_confirm_or_prune_remediation(
            self,
            task=task,
            subtask=subtask,
            attempts=attempts,
            remediation_id=remediation_id,
            verification=verification,
            placeholder_metadata=placeholder_metadata,
        )

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
        return orchestrator_remediation._record_confirm_or_prune_attempt(
            self,
            task=task,
            subtask_id=subtask_id,
            status=status,
            attempt=attempt,
            max_attempts=max_attempts,
            transient=transient,
            reason_code=reason_code,
            retry_strategy=retry_strategy,
            error=error,
        )

    async def _persist_subtask_attempt_record(
        self,
        *,
        task: Task,
        subtask: Subtask,
        subtask_id: str,
        attempt_record: AttemptRecord,
        verification: VerificationResult,
    ) -> None:
        return await orchestrator_remediation._persist_subtask_attempt_record(
            self,
            task=task,
            subtask=subtask,
            subtask_id=subtask_id,
            attempt_record=attempt_record,
            verification=verification,
        )

    async def _sync_remediation_queue_to_db(self, task: Task) -> None:
        return await orchestrator_remediation._sync_remediation_queue_to_db(self, task)

    async def _hydrate_remediation_queue_from_db(self, task: Task) -> None:
        return await orchestrator_remediation._hydrate_remediation_queue_from_db(self, task)

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
        return await orchestrator_remediation._persist_remediation_attempt(
            self,
            task=task,
            remediation_id=remediation_id,
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

    def _remediation_queue(self, task: Task) -> list[dict]:
        return orchestrator_remediation._remediation_queue(self, task)

    @staticmethod
    def _parse_iso_datetime(raw: object) -> datetime | None:
        return orchestrator_remediation.parse_iso_datetime(raw)

    def _remediation_item_due(self, item: dict, now: datetime) -> bool:
        return orchestrator_remediation.remediation_item_due(item, now)

    def _remediation_item_expired(self, item: dict, now: datetime) -> bool:
        return orchestrator_remediation.remediation_item_expired(item, now)

    async def _process_remediation_queue(
        self,
        *,
        task: Task,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
        finalizing: bool,
    ) -> None:
        await orchestrator_remediation.process_remediation_queue(
            self,
            task=task,
            attempts_by_subtask=attempts_by_subtask,
            finalizing=finalizing,
        )

    async def _execute_remediation_item(
        self,
        *,
        task: Task,
        item: dict,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
    ) -> tuple[bool, str]:
        return await orchestrator_remediation.execute_remediation_item(
            self,
            task=task,
            item=item,
            attempts_by_subtask=attempts_by_subtask,
        )

    def _expected_deliverables_for_subtask(self, subtask: Subtask) -> list[str]:
        return orchestrator_output._expected_deliverables_for_subtask(self, subtask)

    def _output_intermediate_root(self) -> str:
        return orchestrator_output.output_intermediate_root(self._process)

    def _output_enforce_single_writer(self) -> bool:
        return orchestrator_output.output_enforce_single_writer(self._process)

    def _output_conflict_policy(self) -> str:
        return orchestrator_output.output_conflict_policy(self._process)

    def _output_publish_mode(self) -> str:
        return orchestrator_output.output_publish_mode(self._process)

    def _phase_finalizer_input_policy(self, phase_id: str) -> str:
        return orchestrator_output.phase_finalizer_input_policy(
            self._process,
            phase_id,
        )

    def _output_write_policy_for_subtask(
        self,
        *,
        subtask: Subtask,
    ) -> dict[str, object]:
        return orchestrator_output._output_write_policy_for_subtask(self, subtask=subtask)

    def _fan_in_worker_output_prefixes(
        self,
        *,
        task: Task,
        subtask: Subtask,
    ) -> list[str]:
        return orchestrator_output.fan_in_worker_output_prefixes(
            self,
            task=task,
            subtask=subtask,
        )

    def _phase_artifact_manifest_path(
        self,
        *,
        task: Task,
        phase_id: str,
    ) -> Path | None:
        return orchestrator_output.phase_artifact_manifest_path(
            self,
            task=task,
            phase_id=phase_id,
        )

    def _phase_worker_subtask_ids(
        self,
        *,
        task: Task,
        phase_id: str,
        finalizer_id: str,
    ) -> list[str]:
        return orchestrator_output.phase_worker_subtask_ids(
            self,
            task=task,
            phase_id=phase_id,
            finalizer_id=finalizer_id,
        )

    def _phase_worker_artifact_paths(
        self,
        *,
        task: Task,
        phase_id: str,
    ) -> dict[str, list[str]]:
        return orchestrator_output.phase_worker_artifact_paths(
            self,
            task=task,
            phase_id=phase_id,
        )

    def _evaluate_finalizer_manifest_requirements(
        self,
        *,
        task: Task,
        subtask: Subtask,
    ) -> dict[str, object]:
        return orchestrator_output.evaluate_finalizer_manifest_requirements(
            self,
            task=task,
            subtask=subtask,
        )

    def _record_fan_in_worker_artifacts(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
    ) -> None:
        orchestrator_output.record_fan_in_worker_artifacts(
            self,
            task=task,
            subtask=subtask,
            result=result,
        )

    def _latest_worker_artifacts_for_phase(
        self,
        *,
        task: Task,
        phase_id: str,
    ) -> dict[str, list[dict[str, object]]]:
        return orchestrator_output.latest_worker_artifacts_for_phase(
            self,
            task=task,
            phase_id=phase_id,
        )

    def _augment_retry_context_with_phase_artifacts(
        self,
        *,
        task: Task,
        subtask: Subtask,
        base_context: str,
    ) -> str:
        return orchestrator_output.augment_retry_context_with_phase_artifacts(
            self,
            task=task,
            subtask=subtask,
            base_context=base_context,
        )

    def _finalizer_stage_publish_plan(
        self,
        *,
        task: Task,
        subtask: Subtask,
        canonical_deliverables: list[str],
        attempt_index: int,
    ) -> dict[str, object]:
        return orchestrator_output.finalizer_stage_publish_plan(
            self,
            task=task,
            subtask=subtask,
            canonical_deliverables=canonical_deliverables,
            attempt_index=attempt_index,
        )

    @staticmethod
    def _merge_unique_paths(*groups: list[str]) -> list[str]:
        return orchestrator_output.merge_unique_paths(*groups)

    def _augment_retry_context_for_stage_publish(
        self,
        *,
        base_context: str,
        stage_plan: dict[str, object],
    ) -> str:
        del self  # retained for compatibility during extraction
        return orchestrator_output.augment_retry_context_for_stage_publish(
            base_context=base_context,
            stage_plan=stage_plan,
        )

    def _intermediate_phase_prefix(
        self,
        *,
        task: Task,
        phase_id: str,
    ) -> str:
        return orchestrator_output.intermediate_phase_prefix(
            self,
            task=task,
            phase_id=phase_id,
        )

    def _intermediate_read_paths_from_tool_calls(
        self,
        *,
        task: Task,
        tool_calls: list[ToolCallRecord],
    ) -> list[str]:
        return orchestrator_output.intermediate_read_paths_from_tool_calls(
            self,
            task=task,
            tool_calls=tool_calls,
        )

    def _manifest_only_input_violations(
        self,
        *,
        task: Task,
        subtask: Subtask,
        tool_calls: list[ToolCallRecord],
        allowed_manifest_paths: list[str],
        allowed_extra_prefixes: list[str],
    ) -> list[str]:
        return orchestrator_output.manifest_only_input_violations(
            self,
            task=task,
            subtask=subtask,
            tool_calls=tool_calls,
            allowed_manifest_paths=allowed_manifest_paths,
            allowed_extra_prefixes=allowed_extra_prefixes,
        )

    @staticmethod
    def _artifact_seals_snapshot(task: Task) -> dict[str, dict[str, object]]:
        return orchestrator_output.artifact_seals_snapshot(task=task)

    @staticmethod
    def _restore_artifact_seals_snapshot(
        *,
        task: Task,
        snapshot: dict[str, dict[str, object]],
    ) -> None:
        orchestrator_output.restore_artifact_seals_snapshot(
            task=task,
            snapshot=snapshot,
        )

    def _seal_paths_after_commit(
        self,
        *,
        task: Task,
        subtask_id: str,
        paths: list[str],
    ) -> None:
        orchestrator_output.seal_paths_after_commit(
            self,
            task=task,
            subtask_id=subtask_id,
            paths=paths,
        )

    def _commit_finalizer_stage_publish(
        self,
        *,
        task: Task,
        subtask: Subtask,
        stage_plan: dict[str, object],
    ) -> tuple[bool, str]:
        return orchestrator_output.commit_finalizer_stage_publish(
            self,
            task=task,
            subtask=subtask,
            stage_plan=stage_plan,
        )

    @staticmethod
    def _normalize_deliverable_paths_for_conflict(
        raw_paths: list[str],
        *,
        workspace: Path | None,
    ) -> list[str]:
        return orchestrator_output.normalize_deliverable_paths_for_conflict(
            raw_paths,
            workspace=workspace,
        )

    def _canonical_deliverable_paths_for_subtask(
        self,
        *,
        task: Task,
        subtask: Subtask,
    ) -> list[str]:
        return orchestrator_output.canonical_deliverable_paths_for_subtask(
            task=task,
            subtask=subtask,
            output_write_policy_for_subtask=(
                lambda target_subtask: self._output_write_policy_for_subtask(
                    subtask=target_subtask,
                )
            ),
        )

    def _prioritize_runnable_for_output_conflicts(
        self,
        *,
        runnable: list[Subtask],
        conflict_tracker: dict[str, dict[str, object]],
    ) -> list[Subtask]:
        threshold = int(
            getattr(
                self,
                "_output_conflict_starvation_threshold",
                self._OUTPUT_CONFLICT_STARVATION_THRESHOLD,
            ),
        )
        return orchestrator_output.prioritize_runnable_for_output_conflicts(
            runnable=runnable,
            conflict_tracker=conflict_tracker,
            starvation_threshold=threshold,
        )

    def _select_conflict_safe_batch(
        self,
        *,
        task: Task,
        runnable: list[Subtask],
        max_parallel: int,
        conflict_tracker: dict[str, dict[str, object]],
        sequence_counter: int,
    ) -> tuple[list[Subtask], list[dict[str, object]], int]:
        threshold = int(
            getattr(
                self,
                "_output_conflict_starvation_threshold",
                self._OUTPUT_CONFLICT_STARVATION_THRESHOLD,
            ),
        )
        active_pending = {
            subtask.id
            for subtask in task.plan.subtasks
            if subtask.status == SubtaskStatus.PENDING
        }
        return orchestrator_output.select_conflict_safe_batch(
            runnable=runnable,
            max_parallel=max_parallel,
            conflict_tracker=conflict_tracker,
            sequence_counter=sequence_counter,
            canonical_paths_for_subtask=(
                lambda subtask: self._canonical_deliverable_paths_for_subtask(
                    task=task,
                    subtask=subtask,
                )
            ),
            starvation_threshold=threshold,
            active_pending_ids=active_pending,
        )

    @staticmethod
    def _files_from_attempts(attempts: list[AttemptRecord], *, max_items: int = 24) -> list[str]:
        return orchestrator_output._files_from_attempts(attempts, max_items=max_items)

    @staticmethod
    def _files_from_tool_calls(tool_calls: list, *, max_items: int = 24) -> list[str]:
        return orchestrator_output._files_from_tool_calls(tool_calls, max_items=max_items)

    def _augment_retry_context_for_outputs(
        self,
        *,
        subtask: Subtask,
        attempts: list[AttemptRecord],
        strategy: RetryStrategy,
        expected_deliverables: list[str],
        forbidden_deliverables: list[str],
        base_context: str,
    ) -> str:
        return orchestrator_output._augment_retry_context_for_outputs(
            self,
            subtask=subtask,
            attempts=attempts,
            strategy=strategy,
            expected_deliverables=expected_deliverables,
            forbidden_deliverables=forbidden_deliverables,
            base_context=base_context,
        )

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

    async def _should_replan_resumed_existing_plan(self, task: Task) -> bool:
        """Replan once when resuming a failed task with pending work."""
        decisions = [str(item or "").strip() for item in task.decisions_log]
        if "Resumed execution from prior task state." not in decisions:
            return False
        if not task.errors_encountered:
            return False
        has_pending = any(
            subtask.status in {SubtaskStatus.PENDING, SubtaskStatus.RUNNING}
            for subtask in task.plan.subtasks
        )
        if not has_pending:
            return False
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        prior_count = int(metadata.get("resume_preflight_replan_count", 0) or 0)
        if prior_count >= 1:
            return False
        metadata["resume_preflight_replan_count"] = prior_count + 1
        task.metadata = metadata
        try:
            await self._save_task_state(task)
        except Exception:
            logger.debug(
                "Failed persisting resume preflight replan marker for %s",
                task.id,
                exc_info=True,
            )
        return True

    def _build_resume_existing_plan_feedback(self, task: Task) -> str:
        """Compose targeted feedback for preflight replanning on resumed tasks."""
        lines = [
            "Task resumed from prior state with failed verification history.",
            "Replan remaining subtasks to match clarified user intent and available data.",
            "Avoid stale cardinality assumptions when the full live dataset is unavailable.",
        ]
        clarification_notes = [
            str(item or "").strip()
            for item in task.decisions_log
            if "clarification (" in str(item or "").lower()
        ]
        if clarification_notes:
            lines.append("Recent clarifications:")
            for note in clarification_notes[-4:]:
                lines.append(f"- {note}")
        recent_errors = [
            f"{str(item.subtask or '').strip()}: {str(item.error or '').strip()}"
            for item in task.errors_encountered[-4:]
            if str(item.error or "").strip()
        ]
        if recent_errors:
            lines.append("Recent verification failures:")
            for item in recent_errors:
                lines.append(f"- {item}")
        return "\n".join(lines)

    async def _should_auto_replan_critical_path_scope_failure(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
        attempts: list[AttemptRecord],
    ) -> bool:
        """Allow one adaptive replan for critical-path scope/cardinality mismatches."""
        reason_codes = {
            str(getattr(verification, "reason_code", "") or "").strip().lower(),
        }
        reason_codes.update(
            str(getattr(item, "reason_code", "") or "").strip().lower()
            for item in attempts
            if isinstance(item, AttemptRecord)
        )
        if not (reason_codes & _SCOPE_ADAPTIVE_REPLAN_REASON_CODES):
            return False
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        tracker = metadata.get("critical_path_scope_replans", {})
        if not isinstance(tracker, dict):
            tracker = {}
        tracker_key = (
            f"{str(getattr(task.plan, 'version', 1) or 1)}:"
            f"{str(getattr(subtask, 'id', '') or '')}"
        )
        prior_count = int(tracker.get(tracker_key, 0) or 0)
        if prior_count >= 1:
            return False
        tracker[tracker_key] = prior_count + 1
        metadata["critical_path_scope_replans"] = tracker
        task.metadata = metadata
        try:
            await self._save_task_state(task)
        except Exception:
            logger.debug(
                "Failed persisting critical-path adaptive replan marker for %s/%s",
                task.id,
                subtask.id,
                exc_info=True,
            )
        return True

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
        prior_evidence = await self._evidence_for_subtask_async(task.id, subtask.id)
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
        contract = self._validity_contract_for_subtask(subtask)
        verification_tier = max(
            2,
            self._synthesis_verification_floor(subtask),
        )
        return await self._verification.verify(
            subtask=subtask,
            result_summary=result.summary or "",
            tool_calls=result.tool_calls,
            evidence_tool_calls=prior_calls,
            evidence_records=prior_evidence,
            validity_contract=contract,
            workspace=workspace,
            tier=verification_tier,
            task_id=task.id,
        )

    async def _handle_success(
        self,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
    ) -> None:
        await orchestrator_dispatch.handle_success(
            self,
            task,
            subtask,
            result,
            verification,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def _plan_task(
        self,
        task: Task,
        *,
        planner_feedback: str = "",
    ) -> Plan:
        return await orchestrator_planning.plan_task(
            self,
            task,
            planner_feedback=planner_feedback,
        )

    @staticmethod
    def _read_roots_for_task(task: Task) -> list[Path]:
        return orchestrator_planning._read_roots_for_task(task)

    @staticmethod
    def _read_path_map_for_task(task: Task) -> dict[str, Path]:
        return orchestrator_planning._read_path_map_for_task(task)

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
    _VALIDITY_SCORECARD_JSON_NAME = "validity-scorecard.json"
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
        return await orchestrator_planning._analyze_workspace(self, workspace_path)

    def _scan_workspace_documents(
        self,
        workspace_path: Path,
        max_per_category: int = 15,
    ) -> str:
        return orchestrator_planning._scan_workspace_documents(
            self,
            workspace_path,
            max_per_category,
        )

    async def _analyze_workspace_for_process(
        self, workspace_path: Path,
    ) -> str:
        return await orchestrator_planning._analyze_workspace_for_process(self, workspace_path)

    def _analyze_workspace_for_process_sync(self, workspace_path: Path) -> str:
        return orchestrator_planning._analyze_workspace_for_process_sync(self, workspace_path)

    async def _replan_task(
        self,
        task: Task,
        *,
        reason: str = "subtask_failures",
        failed_subtask_id: str = "",
        verification_feedback: str | None = None,
    ) -> bool:
        return await orchestrator_planning.replan_task(
            self,
            task,
            reason=reason,
            failed_subtask_id=failed_subtask_id,
            verification_feedback=verification_feedback,
        )

    @staticmethod
    def _validate_replan_contract(
        *,
        current_plan: Plan,
        replanned_plan: Plan,
    ) -> str | None:
        return orchestrator_planning._validate_replan_contract(
            current_plan=current_plan,
            replanned_plan=replanned_plan,
        )

    def _parse_plan(self, response: ModelResponse, goal: str = "") -> Plan:
        return orchestrator_planning._parse_plan(self, response, goal)

    def _apply_process_phase_mode(self, plan: Plan) -> Plan:
        return orchestrator_planning._apply_process_phase_mode(self, plan)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _planning_response_max_tokens(self) -> int | None:
        return orchestrator_planning._planning_response_max_tokens(self)

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

    async def _save_task_state(self, task: Task) -> None:
        await run_blocking_io(self._state.save, task)
        if self._snapshot_mirror_writer is None:
            return
        try:
            await self._snapshot_mirror_writer(task)
        except Exception:
            logger.debug(
                "Failed syncing mirrored task snapshot for %s",
                task.id,
                exc_info=True,
            )

    async def _sync_task_snapshot_projection(self, task: Task) -> None:
        if self._snapshot_mirror_writer is None:
            return
        try:
            await self._snapshot_mirror_writer(task)
        except Exception:
            logger.debug(
                "Failed syncing mirrored task snapshot for %s",
                task.id,
                exc_info=True,
            )

    def _save_task_state_sync(self, task: Task) -> None:
        self._state.save(task)
        if self._snapshot_mirror_writer is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self._sync_task_snapshot_projection(task))
            except Exception:
                logger.debug(
                    "Failed syncing mirrored task snapshot for %s",
                    task.id,
                    exc_info=True,
                )
            return
        loop.create_task(self._sync_task_snapshot_projection(task))

    def set_snapshot_mirror_writer(
        self,
        writer: Callable[[Task], Awaitable[None]] | None,
    ) -> None:
        self._snapshot_mirror_writer = writer

    async def _load_task_state(self, task_id: str) -> Task | None:
        try:
            task = await run_blocking_io(self._state.load, task_id)
        except Exception:
            return None
        return task if isinstance(task, Task) else None

    async def _task_state_exists(self, task_id: str) -> bool:
        return bool(await run_blocking_io(self._state.exists, task_id))

    async def _load_evidence_records(self, task_id: str) -> list[dict]:
        try:
            records = await run_blocking_io(self._state.load_evidence_records, task_id)
        except Exception:
            return []
        return [record for record in records if isinstance(record, dict)]

    async def _append_evidence_records(
        self,
        task_id: str,
        records: list[dict],
    ) -> list[dict]:
        merged = await run_blocking_io(
            self._state.append_evidence_records,
            task_id,
            records,
        )
        return [record for record in merged if isinstance(record, dict)]

    def _evidence_for_subtask(self, task_id: str, subtask_id: str) -> list[dict]:
        return orchestrator_evidence._evidence_for_subtask(
            self,
            task_id,
            subtask_id,
        )

    async def _evidence_for_subtask_async(self, task_id: str, subtask_id: str) -> list[dict]:
        return await orchestrator_evidence._evidence_for_subtask_async(
            self,
            task_id,
            subtask_id,
        )

    def _persist_subtask_evidence(
        self,
        task_id: str,
        subtask_id: str,
        evidence_records: list[dict] | None,
        *,
        tool_calls: list[ToolCallRecord] | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        return orchestrator_evidence._persist_subtask_evidence(
            self,
            task_id,
            subtask_id,
            evidence_records,
            tool_calls=tool_calls,
            workspace=workspace,
        )

    async def _persist_subtask_evidence_async(
        self,
        task_id: str,
        subtask_id: str,
        evidence_records: list[dict] | None,
        *,
        tool_calls: list[ToolCallRecord] | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        return await orchestrator_evidence._persist_subtask_evidence_async(
            self,
            task_id,
            subtask_id,
            evidence_records,
            tool_calls=tool_calls,
            workspace=workspace,
        )

    @staticmethod
    def _artifact_content_for_call(
        tool_name: str,
        args: dict[str, object],
        result_data: dict[str, object],
    ) -> str:
        return orchestrator_evidence._artifact_content_for_call(tool_name, args, result_data)

    def _artifact_seal_registry(self, task: Task) -> dict[str, dict[str, object]]:
        return orchestrator_evidence._artifact_seal_registry(self, task)

    def _record_artifact_seals(
        self,
        *,
        task: Task,
        subtask_id: str,
        tool_calls: list[ToolCallRecord] | None,
    ) -> int:
        return orchestrator_evidence._record_artifact_seals(
            self,
            task=task,
            subtask_id=subtask_id,
            tool_calls=tool_calls,
        )

    def _is_intermediate_artifact_path(self, *, task: Task, relpath: str) -> bool:
        return orchestrator_evidence._is_intermediate_artifact_path(
            self,
            task=task,
            relpath=relpath,
        )

    def _validate_artifact_seals(
        self,
        *,
        task: Task,
    ) -> tuple[bool, list[dict[str, object]], int]:
        return orchestrator_evidence._validate_artifact_seals(self, task=task)

    def _backfill_artifact_seals_from_evidence(self, task: Task) -> int:
        return orchestrator_evidence._backfill_artifact_seals_from_evidence(self, task)

    def _validity_scorecard_state(self, task: Task) -> dict[str, object]:
        return orchestrator_evidence._validity_scorecard_state(self, task)

    def _record_subtask_validity_metrics(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
    ) -> None:
        return orchestrator_evidence._record_subtask_validity_metrics(
            self,
            task=task,
            subtask=subtask,
            verification=verification,
        )

    def _scorecard_source_window(self, task: Task) -> dict[str, str]:
        return orchestrator_evidence._scorecard_source_window(self, task)

    def _build_run_validity_scorecard(self, task: Task) -> dict[str, object]:
        return orchestrator_evidence._build_run_validity_scorecard(self, task)

    def _refresh_run_validity_scorecard(self, task: Task) -> dict[str, object]:
        return orchestrator_evidence._refresh_run_validity_scorecard(self, task)

    def _export_validity_scorecard_json(self, task: Task) -> None:
        return orchestrator_evidence._export_validity_scorecard_json(self, task)

    def _emit_run_validity_scorecard(self, task: Task) -> None:
        return orchestrator_evidence._emit_run_validity_scorecard(self, task)

    def _append_synthesis_provenance_footer(
        self,
        *,
        task: Task,
        summary: str,
    ) -> str:
        return orchestrator_evidence._append_synthesis_provenance_footer(
            self,
            task=task,
            summary=summary,
        )

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
        return orchestrator_evidence.stringify_evidence_csv_value(value)

    @classmethod
    def _evidence_csv_fieldnames(cls, rows: list[dict[str, str]]) -> list[str]:
        return orchestrator_evidence.evidence_csv_fieldnames(
            base_fields=cls._EVIDENCE_LEDGER_CSV_BASE_FIELDS,
            rows=rows,
        )

    def _evidence_csv_rows(self, records: list[dict]) -> list[dict[str, str]]:
        del self  # retained for compatibility during extraction
        return orchestrator_evidence.evidence_csv_rows(records)

    def _export_evidence_ledger_csv(self, task: Task) -> None:
        return orchestrator_evidence._export_evidence_ledger_csv(self, task)

    def _finalize_task(self, task: Task) -> Task:
        return orchestrator_output._finalize_task(self, task)

    async def _finalize_task_async(self, task: Task) -> Task:
        return await orchestrator_output._finalize_task_async(self, task)

    async def _learn_from_task(self, task: Task) -> None:
        return await orchestrator_telemetry._learn_from_task(self, task)

    def _emit(self, event_type: str, task_id: str, data: dict) -> None:
        return orchestrator_runtime.emit_event(self, event_type, task_id, data)

    @staticmethod
    def _task_run_id(task: Task) -> str:
        return orchestrator_runtime.task_run_id(task)

    def _initialize_task_run_id(self, task: Task) -> str:
        return orchestrator_runtime.initialize_task_run_id(self, task)

    async def _initialize_task_run_id_async(self, task: Task) -> str:
        return await orchestrator_runtime.initialize_task_run_id_async(self, task)

    def _apply_budget_metadata(
        self,
        task: Task,
        budget_name: str,
        observed: int | float,
        limit: int,
    ) -> None:
        return orchestrator_budget._apply_budget_metadata(self, task, budget_name, observed, limit)

    async def _enforce_global_budget(self, task: Task) -> bool:
        return await orchestrator_budget._enforce_global_budget(self, task)

    @staticmethod
    def _new_telemetry_rollup() -> dict[str, int]:
        return orchestrator_telemetry.new_telemetry_rollup()

    def _accumulate_subtask_telemetry(self, result: SubtaskResult) -> None:
        orchestrator_telemetry.accumulate_subtask_telemetry(self, result)

    def _task_event_counts(self, task_id: str) -> dict[str, int]:
        return orchestrator_telemetry.task_event_counts(self._events, task_id)

    def _verification_reason_counts(self, task_id: str) -> dict[str, int]:
        return orchestrator_telemetry.verification_reason_counts(
            event_bus=self._events,
            task_id=task_id,
            verification_outcome_event_type=VERIFICATION_OUTCOME,
        )

    def _emit_telemetry_run_summary(self, task: Task) -> None:
        return orchestrator_telemetry._emit_telemetry_run_summary(self, task)

    def cancel_task(self, task: Task) -> None:
        """Mark a task for cancellation."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["cancel_reason"] = "cancel_requested"
        task.status = TaskStatus.CANCELLED
        self._emit(TASK_CANCEL_REQUESTED, task.id, {
            "requested": True,
            "path": "orchestrator",
        })

    def pause_task(self, task: Task) -> None:
        """Pause a running task at the next orchestration boundary."""
        if task.status not in {TaskStatus.EXECUTING, TaskStatus.PLANNING}:
            self._emit(TASK_PAUSED, task.id, {
                "requested": False,
                "error": f"invalid_status:{task.status.value}",
                "path": "orchestrator",
            })
            return
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["paused_from_status"] = task.status.value
        task.status = TaskStatus.PAUSED
        self._emit(TASK_PAUSED, task.id, {
            "requested": True,
            "status": task.status.value,
            "path": "orchestrator",
        })

    def resume_task(self, task: Task) -> None:
        """Resume a paused task."""
        if task.status != TaskStatus.PAUSED:
            self._emit(TASK_RESUMED, task.id, {
                "requested": False,
                "error": f"invalid_status:{task.status.value}",
                "path": "orchestrator",
            })
            return
        paused_from = ""
        if isinstance(task.metadata, dict):
            paused_from = str(task.metadata.pop("paused_from_status", "") or "").strip().lower()
        if paused_from == TaskStatus.PLANNING.value:
            task.status = TaskStatus.PLANNING
        else:
            task.status = TaskStatus.EXECUTING
        self._emit(TASK_RESUMED, task.id, {
            "requested": True,
            "status": task.status.value,
            "path": "orchestrator",
        })

    @property
    def question_manager(self) -> QuestionManager | None:
        """Shared ask-user question manager used by this orchestrator."""
        return self._question

    async def list_pending_questions(self, task_id: str) -> list[dict]:
        if self._question is None:
            return []
        return await self._question.list_pending_questions(task_id)

    async def answer_question(self, task_id: str, question_id: str, answer_payload: dict):
        if self._question is None:
            return None
        return await self._question.answer_question(
            task_id=task_id,
            question_id=question_id,
            answer_payload=answer_payload,
        )


def create_task(
    goal: str,
    workspace: str = "",
    approval_mode: str = "auto",
    callback_url: str = "",
    context: dict | None = None,
    metadata: dict | None = None,
) -> Task:
    return orchestrator_task_factory.create_task(
        goal, workspace, approval_mode, callback_url, context, metadata,
    )
