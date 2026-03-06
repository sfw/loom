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
import hashlib
import json
import logging
import re
import time
import uuid
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from loom.auth.runtime import AuthResolutionError, build_run_auth_context
from loom.config import Config

if TYPE_CHECKING:
    from loom.processes.schema import IterationPolicy, ProcessDefinition
from loom.engine.iteration_gates import IterationEvaluation, IterationGateEvaluator
from loom.engine.runner import SubtaskResult, SubtaskResultStatus, SubtaskRunner, ToolCallRecord
from loom.engine.scheduler import Scheduler
from loom.engine.verification import VerificationGates, VerificationResult
from loom.events.bus import Event, EventBus
from loom.events.types import (
    APPROVAL_RECEIVED,
    APPROVAL_REQUESTED,
    ARTIFACT_SEAL_VALIDATION,
    ASK_USER_ANSWERED,
    ASK_USER_CANCELLED,
    ASK_USER_REQUESTED,
    ASK_USER_TIMEOUT,
    CLAIMS_PRUNED,
    ITERATION_COMPLETED,
    ITERATION_GATE_FAILED,
    ITERATION_RETRYING,
    ITERATION_STARTED,
    ITERATION_STATE_RECONCILED,
    ITERATION_TERMINAL,
    MODEL_INVOCATION,
    PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED,
    PLACEHOLDER_FILLED,
    PLACEHOLDER_PRUNED,
    PLACEHOLDER_REMEDIATION_UNRESOLVED,
    REMEDIATION_ATTEMPT,
    REMEDIATION_EXPIRED,
    REMEDIATION_FAILED,
    REMEDIATION_QUEUED,
    REMEDIATION_RESOLVED,
    REMEDIATION_STARTED,
    REMEDIATION_TERMINAL,
    RUN_VALIDITY_SCORECARD,
    STEER_INSTRUCTION,
    SUBTASK_BLOCKED,
    SUBTASK_COMPLETED,
    SUBTASK_FAILED,
    SUBTASK_OUTCOME_STALE,
    SUBTASK_POLICY_RECONCILED,
    SUBTASK_RETRYING,
    SUBTASK_STARTED,
    SYNTHESIS_INPUT_GATE_DECISION,
    TASK_BUDGET_EXHAUSTED,
    TASK_CANCEL_ACK,
    TASK_CANCEL_REQUESTED,
    TASK_CANCEL_TIMEOUT,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_INJECTED,
    TASK_PAUSED,
    TASK_PLAN_DEGRADED,
    TASK_PLAN_NORMALIZED,
    TASK_PLAN_READY,
    TASK_PLANNING,
    TASK_REPLAN_REJECTED,
    TASK_REPLANNING,
    TASK_RESUMED,
    TASK_RUN_ACQUIRED,
    TASK_STALLED,
    TASK_STALLED_RECOVERY_ATTEMPTED,
    TELEMETRY_RUN_SUMMARY,
    UNCONFIRMED_DATA_QUEUED,
    VERIFICATION_FAILED,
    VERIFICATION_OUTCOME,
    VERIFICATION_PASSED,
    VERIFICATION_STARTED,
)
from loom.learning.manager import LearningManager
from loom.models.base import ModelResponse
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter, ResponseValidator
from loom.processes.phase_alignment import infer_phase_id_for_subtask
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalDecision, ApprovalManager, ApprovalRequest
from loom.recovery.confidence import ConfidenceScorer
from loom.recovery.errors import ErrorCategory, categorize_error
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
        question_manager: QuestionManager | None = None,
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

            await self._reconcile_subtask_policy_state(task)

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

    def _validity_contract_for_subtask(self, subtask: Subtask) -> dict[str, object]:
        contract = (
            dict(subtask.validity_contract_snapshot)
            if isinstance(subtask.validity_contract_snapshot, dict)
            and subtask.validity_contract_snapshot
            else self._resolve_subtask_validity_contract(subtask=subtask)
        )
        normalized = self._normalize_validity_contract(contract)
        subtask.validity_contract_snapshot = normalized
        subtask.validity_contract_hash = self._hash_validity_contract(normalized)
        return normalized

    def _synthesis_verification_floor(self, subtask: Subtask) -> int:
        if not subtask.is_synthesis:
            return max(1, int(subtask.verification_tier or 1))
        contract = self._validity_contract_for_subtask(subtask)
        final_gate = contract.get("final_gate", {})
        if isinstance(final_gate, dict):
            floor = max(
                1,
                self._to_non_negative_int(
                    final_gate.get("synthesis_min_verification_tier", 2),
                    2,
                ),
            )
        else:
            floor = 2
        return max(floor, int(subtask.verification_tier or 1))

    @staticmethod
    def _tool_call_succeeded(call: ToolCallRecord) -> bool:
        result = getattr(call, "result", None)
        return bool(result is not None and getattr(result, "success", False))

    def _fact_checker_used(self, tool_calls: list[ToolCallRecord]) -> bool:
        for call in tool_calls:
            if str(getattr(call, "tool", "") or "").strip().lower() != "fact_checker":
                continue
            if self._tool_call_succeeded(call):
                return True
        return False

    def _fact_checker_verdict_count(self, tool_calls: list[ToolCallRecord]) -> int:
        total = 0
        for call in tool_calls:
            if str(getattr(call, "tool", "") or "").strip().lower() != "fact_checker":
                continue
            result = getattr(call, "result", None)
            if result is None or not bool(getattr(result, "success", False)):
                continue
            data = getattr(result, "data", {})
            if not isinstance(data, dict):
                continue
            verdicts = data.get("verdicts", [])
            if not isinstance(verdicts, list):
                continue
            total += sum(
                1
                for verdict in verdicts
                if isinstance(verdict, dict)
                and str(verdict.get("claim", "") or "").strip()
            )
        return total

    def _requires_fact_checker_for_subtask(self, subtask: Subtask) -> bool:
        if not subtask.is_synthesis:
            return False
        contract = self._validity_contract_for_subtask(subtask)
        if not self._to_bool(contract.get("enabled", False), False):
            return False
        return self._to_bool(
            contract.get("require_fact_checker_for_synthesis", False),
            False,
        )

    def _claim_graph_state(self, task: Task) -> dict[str, object]:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        graph = metadata.get("claim_graph")
        if not isinstance(graph, dict):
            graph = {}
        supported = graph.get("supported_by_subtask")
        if not isinstance(supported, dict):
            supported = {}
        graph["supported_by_subtask"] = supported
        unresolved = graph.get("unresolved_by_subtask")
        if not isinstance(unresolved, dict):
            unresolved = {}
        graph["unresolved_by_subtask"] = unresolved
        metadata["claim_graph"] = graph
        task.metadata = metadata
        return graph

    def _update_claim_graph_from_verification(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
    ) -> None:
        metadata = verification.metadata if isinstance(verification.metadata, dict) else {}
        claims = metadata.get("claim_lifecycle", [])
        if not isinstance(claims, list):
            return
        graph = self._claim_graph_state(task)
        supported_by_subtask = graph.get("supported_by_subtask")
        unresolved_by_subtask = graph.get("unresolved_by_subtask")
        if (
            not isinstance(supported_by_subtask, dict)
            or not isinstance(unresolved_by_subtask, dict)
        ):
            return
        supported_claims: list[dict[str, object]] = []
        unresolved_claims: list[dict[str, object]] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            status = str(claim.get("status", "") or "").strip().lower()
            if status == "supported":
                supported_claims.append(dict(claim))
            elif status in _CLAIM_TERMINAL_UNRESOLVED:
                unresolved_claims.append(dict(claim))
        supported_by_subtask[subtask.id] = supported_claims
        unresolved_by_subtask[subtask.id] = unresolved_claims

    @staticmethod
    def _claims_from_verification(verification: VerificationResult) -> list[dict[str, object]]:
        metadata = verification.metadata if isinstance(verification.metadata, dict) else {}
        raw_claims = metadata.get("claim_lifecycle", [])
        if not isinstance(raw_claims, list):
            return []
        claims: list[dict[str, object]] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            claim = dict(item)
            status = str(claim.get("status", "extracted") or "extracted").strip().lower()
            if not status:
                status = "extracted"
            claim["status"] = status
            claim["claim_id"] = str(claim.get("claim_id", "") or "").strip()
            claim["text"] = str(claim.get("text", "") or "").strip()
            claim["claim_type"] = str(
                claim.get("claim_type", "qualitative") or "qualitative",
            ).strip().lower()
            claim["criticality"] = str(
                claim.get("criticality", "important") or "important",
            ).strip().lower()
            reason_code = str(claim.get("reason_code", "") or "").strip().lower()
            claim["reason_code"] = reason_code
            refs = claim.get("evidence_refs", [])
            if isinstance(refs, str):
                refs = [refs]
            if not isinstance(refs, list):
                refs = []
            normalized_refs = [
                str(ref or "").strip()
                for ref in refs
                if str(ref or "").strip()
            ]
            claim["evidence_refs"] = normalized_refs
            lifecycle = claim.get("lifecycle", [])
            if isinstance(lifecycle, str):
                lifecycle = [lifecycle]
            if not isinstance(lifecycle, list):
                lifecycle = []
            normalized_lifecycle = [
                str(step or "").strip().lower()
                for step in lifecycle
                if str(step or "").strip()
            ]
            if "extracted" not in normalized_lifecycle:
                normalized_lifecycle.insert(0, "extracted")
            if status not in normalized_lifecycle:
                normalized_lifecycle.append(status)
            claim["lifecycle"] = normalized_lifecycle
            claims.append(claim)
        return claims

    @staticmethod
    def _normalize_claim_reason_code(status: str, reason_code: str) -> str:
        normalized_reason = str(reason_code or "").strip().lower()
        if normalized_reason:
            return normalized_reason
        normalized_status = str(status or "").strip().lower()
        if normalized_status in _CLAIM_REASON_CODES:
            return _CLAIM_REASON_CODES[normalized_status]
        return "claim_insufficient_evidence"

    @staticmethod
    def _claim_counts(claims: list[dict[str, object]]) -> dict[str, int]:
        counts = {
            "extracted": len(claims),
            "supported": 0,
            "contradicted": 0,
            "insufficient_evidence": 0,
            "stale": 0,
            "pruned": 0,
            "unresolved": 0,
            "critical_total": 0,
            "critical_supported": 0,
            "critical_contradicted": 0,
        }
        for claim in claims:
            status = str(claim.get("status", "") or "").strip().lower()
            if status in counts:
                counts[status] += 1
            if status in _CLAIM_TERMINAL_UNRESOLVED:
                counts["unresolved"] += 1
            criticality = str(claim.get("criticality", "") or "").strip().lower()
            if criticality != "critical":
                continue
            counts["critical_total"] += 1
            if status == "supported":
                counts["critical_supported"] += 1
            elif status == "contradicted":
                counts["critical_contradicted"] += 1
        return counts

    @staticmethod
    def _claim_ratios(counts: dict[str, int]) -> dict[str, float]:
        extracted = max(0, int(counts.get("extracted", 0) or 0))
        supported = max(0, int(counts.get("supported", 0) or 0))
        unresolved = max(0, int(counts.get("unresolved", 0) or 0))
        critical_total = max(0, int(counts.get("critical_total", 0) or 0))
        critical_supported = max(0, int(counts.get("critical_supported", 0) or 0))
        return {
            "supported_ratio": (float(supported) / float(extracted)) if extracted > 0 else 1.0,
            "unverified_ratio": (float(unresolved) / float(extracted)) if extracted > 0 else 0.0,
            "critical_support_ratio": (
                float(critical_supported) / float(critical_total)
            ) if critical_total > 0 else 1.0,
        }

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
        return VerificationResult(
            tier=int(verification.tier),
            passed=verification.passed if passed is None else bool(passed),
            confidence=float(verification.confidence if confidence is None else confidence),
            checks=list(verification.checks or []),
            feedback=verification.feedback if feedback is None else feedback,
            outcome=str(verification.outcome if outcome is None else outcome),
            reason_code=str(verification.reason_code if reason_code is None else reason_code),
            severity_class=(
                str(verification.severity_class or "")
                if severity_class is None
                else str(severity_class)
            ),
            metadata=metadata,
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
        if subtask.is_synthesis:
            return verification
        claims = self._claims_from_verification(verification)
        if not claims:
            return verification
        unresolved = [
            claim for claim in claims
            if str(claim.get("status", "") or "").strip().lower() in _CLAIM_TERMINAL_UNRESOLVED
        ]
        if not unresolved:
            return verification

        prune_mode = str(contract.get("prune_mode", "drop") or "").strip().lower()
        if prune_mode not in {"drop", "rewrite_uncertainty"}:
            prune_mode = "drop"

        supported_claims: list[dict[str, object]] = []
        pruned_claims: list[dict[str, object]] = []
        uncertainty_notes: list[str] = []
        for claim in claims:
            normalized = dict(claim)
            status = str(normalized.get("status", "") or "").strip().lower()
            reason = self._normalize_claim_reason_code(
                status=status,
                reason_code=str(normalized.get("reason_code", "") or ""),
            )
            normalized["reason_code"] = reason
            lifecycle = normalized.get("lifecycle", [])
            if isinstance(lifecycle, str):
                lifecycle = [lifecycle]
            if not isinstance(lifecycle, list):
                lifecycle = []
            lifecycle_norm = [
                str(item or "").strip().lower()
                for item in lifecycle
                if str(item or "").strip()
            ]
            if "extracted" not in lifecycle_norm:
                lifecycle_norm.insert(0, "extracted")

            if status in _CLAIM_TERMINAL_UNRESOLVED:
                pruned = dict(normalized)
                pruned["status"] = "pruned"
                pruned["reason_code"] = _CLAIM_REASON_CODES["pruned"]
                if "pruned" not in lifecycle_norm:
                    lifecycle_norm.append("pruned")
                pruned["lifecycle"] = lifecycle_norm
                pruned_claims.append(pruned)
                if prune_mode == "rewrite_uncertainty":
                    text = str(pruned.get("text", "") or "").strip()
                    if text:
                        uncertainty_notes.append(
                            f"Uncertain claim excluded from synthesis: {text}",
                        )
                continue

            if status not in lifecycle_norm:
                lifecycle_norm.append(status)
            normalized["lifecycle"] = lifecycle_norm
            supported_claims.append(normalized)

        pruned_count = len(pruned_claims)
        updated_claims = supported_claims + pruned_claims
        counts = self._claim_counts(updated_claims)
        ratios = self._claim_ratios(counts)

        metadata = dict(verification.metadata) if isinstance(verification.metadata, dict) else {}
        metadata["claim_lifecycle_original"] = claims
        metadata["claim_lifecycle"] = updated_claims
        metadata["claim_pruned"] = bool(pruned_count > 0)
        metadata["claim_prune_mode"] = prune_mode
        metadata["claim_pruned_count"] = int(pruned_count)
        metadata["claim_status_counts"] = counts
        metadata["claim_reason_codes"] = sorted({
            self._normalize_claim_reason_code(
                status=str(item.get("status", "") or ""),
                reason_code=str(item.get("reason_code", "") or ""),
            )
            for item in updated_claims
        })
        metadata["supported_ratio"] = ratios["supported_ratio"]
        metadata["unverified_ratio"] = ratios["unverified_ratio"]
        min_supported_ratio = self._to_ratio(contract.get("min_supported_ratio", 0.75), 0.75)
        max_unverified_ratio = self._to_ratio(contract.get("max_unverified_ratio", 0.25), 0.25)
        max_contradicted = self._to_non_negative_int(
            contract.get("max_contradicted_count", 0),
            0,
        )
        metadata["claim_gate_thresholds"] = {
            "min_supported_ratio": min_supported_ratio,
            "max_unverified_ratio": max_unverified_ratio,
            "max_contradicted_count": max_contradicted,
        }
        post_prune_gate_passed = (
            int(counts.get("contradicted", 0) or 0) <= max_contradicted
            and float(ratios.get("supported_ratio", 0.0) or 0.0) >= min_supported_ratio
            and float(ratios.get("unverified_ratio", 0.0) or 0.0) <= max_unverified_ratio
        )
        metadata["post_prune_gate_passed"] = bool(post_prune_gate_passed)
        if uncertainty_notes:
            metadata["uncertainty_annotations"] = uncertainty_notes[:20]
            note = (
                "Unsupported or uncertain claims were rewritten as uncertainty "
                "annotations and excluded from downstream verified context."
            )
            result.summary = "\n".join(part for part in [result.summary, note] if part).strip()

        self._emit(CLAIMS_PRUNED, task.id, {
            "subtask_id": subtask.id,
            "phase_id": subtask.phase_id,
            "pruned_count": int(pruned_count),
            "supported_count": int(counts.get("supported", 0)),
            "contradicted_count": int(counts.get("contradicted", 0)),
            "insufficient_evidence_count": int(counts.get("insufficient_evidence", 0)),
            "prune_mode": prune_mode,
        })

        reason_code = str(verification.reason_code or "").strip().lower()
        if (
            not verification.passed
            and reason_code in _CLAIM_RECOVERABLE_FAILURE_CODES
            and post_prune_gate_passed
        ):
            note = (
                "Intermediate validity policy pruned unsupported claims and "
                "allowed execution to continue."
            )
            result.status = SubtaskResultStatus.SUCCESS
            return self._verification_with_metadata(
                verification,
                metadata=metadata,
                passed=True,
                outcome=(
                    "partial_verified"
                    if self._config.verification.allow_partial_verified
                    else "pass_with_warnings"
                ),
                reason_code="claim_pruned",
                feedback="\n".join(part for part in [verification.feedback or "", note] if part),
                severity_class="semantic",
                confidence=min(0.8, max(0.3, float(verification.confidence or 0.5))),
            )
        if not post_prune_gate_passed:
            gate_reason = "coverage_below_threshold"
            if int(counts.get("contradicted", 0) or 0) > max_contradicted:
                gate_reason = "claim_contradicted"
            elif float(ratios.get("unverified_ratio", 0.0) or 0.0) > max_unverified_ratio:
                gate_reason = "claim_insufficient_evidence"
            threshold_note = (
                "Intermediate validity policy pruned unsupported claims, but "
                "post-prune coverage did not satisfy contract thresholds."
            )
            return self._verification_with_metadata(
                verification,
                metadata=metadata,
                passed=False,
                outcome="fail",
                reason_code=gate_reason,
                feedback="\n".join(
                    part
                    for part in [verification.feedback or "", threshold_note]
                    if part
                ),
                severity_class="semantic",
                confidence=min(float(verification.confidence or 0.5), 0.45),
            )
        return self._verification_with_metadata(
            verification,
            metadata=metadata,
        )

    @staticmethod
    def _parse_temporal_date_token(value: object) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        if isinstance(value, datetime):
            return value
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed

        ymd_match = re.search(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b", text)
        if ymd_match:
            try:
                return datetime.strptime(ymd_match.group(0).replace("/", "-"), "%Y-%m-%d")
            except ValueError:
                pass
        ym_match = re.search(r"\b(19|20)\d{2}[-/](0[1-9]|1[0-2])\b", text)
        if ym_match:
            try:
                return datetime.strptime(ym_match.group(0).replace("/", "-"), "%Y-%m")
            except ValueError:
                pass
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        if year_match:
            try:
                return datetime.strptime(year_match.group(0), "%Y")
            except ValueError:
                return None
        return None

    @classmethod
    def _extract_temporal_dates_from_text(cls, text: str) -> list[datetime]:
        parsed: list[datetime] = []
        seen: set[str] = set()
        pattern = (
            r"\b(?:19|20)\d{2}"
            r"(?:[-/](?:0[1-9]|1[0-2])"
            r"(?:[-/](?:0[1-9]|[12]\d|3[01]))?)?\b"
        )
        for match in re.finditer(pattern, text):
            token = str(match.group(0) or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            parsed_date = cls._parse_temporal_date_token(token)
            if parsed_date is not None:
                parsed.append(parsed_date)
        return parsed

    @classmethod
    def _claim_temporal_scope(cls, claim: dict[str, object]) -> dict[str, object]:
        metadata = claim.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source_scope = claim.get("source_time_scope", {})
        if not isinstance(source_scope, dict):
            source_scope = {}

        def _lookup(*keys: str) -> str:
            for key in keys:
                for container in (claim, metadata, source_scope):
                    value = container.get(key)
                    text = str(value or "").strip()
                    if text:
                        return text
            return ""

        as_of_text = _lookup("as_of")
        source_as_of_text = _lookup("source_as_of", "evidence_as_of")
        period_start_text = _lookup("period_start")
        period_end_text = _lookup("period_end")
        text_dates = cls._extract_temporal_dates_from_text(str(claim.get("text", "") or ""))
        return {
            "as_of": cls._parse_temporal_date_token(as_of_text),
            "source_as_of": cls._parse_temporal_date_token(source_as_of_text),
            "period_start": cls._parse_temporal_date_token(period_start_text),
            "period_end": cls._parse_temporal_date_token(period_end_text),
            "text_dates": text_dates,
        }

    @staticmethod
    def _temporal_claim_key(text: str) -> str:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return ""
        cleaned = re.sub(
            r"\b(?:19|20)\d{2}(?:[-/](?:0[1-9]|1[0-2])(?:[-/](?:0[1-9]|[12]\d|3[01]))?)?\b",
            " ",
            lowered,
        )
        cleaned = re.sub(r"\b\d+(?:\.\d+)?\b", " ", cleaned)
        tokens = [
            token
            for token in re.split(r"[^a-z0-9]+", cleaned)
            if len(token) >= 3
        ]
        if not tokens:
            return ""
        return " ".join(tokens[:8])

    def _enforce_temporal_consistency_gate(
        self,
        *,
        subtask: Subtask,
        verification: VerificationResult,
        contract: dict[str, object],
    ) -> VerificationResult:
        if not subtask.is_synthesis:
            return verification
        claims = self._claims_from_verification(verification)
        if not claims:
            return verification

        final_gate = contract.get("final_gate", {})
        temporal: dict[str, object] = {}
        if isinstance(final_gate, dict):
            raw_temporal = final_gate.get("temporal_consistency", {})
            if isinstance(raw_temporal, dict):
                temporal = raw_temporal
        enabled = self._to_bool(temporal.get("enabled", False), False)
        if not enabled:
            return verification

        require_as_of_alignment = self._to_bool(
            temporal.get("require_as_of_alignment", False),
            False,
        )
        enforce_date_conflicts = self._to_bool(
            temporal.get("enforce_cross_claim_date_conflict_check", False),
            False,
        )
        max_source_age_days = self._to_non_negative_int(
            temporal.get("max_source_age_days", 0),
            0,
        )
        as_of_text = str(temporal.get("as_of", "") or "").strip()
        reference_dt = self._parse_temporal_date_token(as_of_text) or datetime.now()
        reference_date = reference_dt.date()

        as_of_values: set[str] = set()
        stale_claims: list[dict[str, object]] = []
        future_source_claims: list[str] = []
        conflict_index: dict[str, set[str]] = {}

        stale_ids: set[str] = set()
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            status = str(claim.get("status", "") or "").strip().lower()
            if status != "supported":
                continue
            scope = self._claim_temporal_scope(claim)
            as_of_dt = scope.get("as_of")
            source_as_of_dt = scope.get("source_as_of")
            period_end_dt = scope.get("period_end")
            text_dates = scope.get("text_dates", [])
            if not isinstance(text_dates, list):
                text_dates = []

            if require_as_of_alignment and isinstance(as_of_dt, datetime):
                as_of_values.add(as_of_dt.date().isoformat())

            source_dt = (
                source_as_of_dt
                if isinstance(source_as_of_dt, datetime)
                else (
                    as_of_dt
                    if isinstance(as_of_dt, datetime)
                    else (
                        period_end_dt
                        if isinstance(period_end_dt, datetime)
                        else (text_dates[0] if text_dates else None)
                    )
                )
            )
            if isinstance(source_dt, datetime) and max_source_age_days > 0:
                age_days = (reference_date - source_dt.date()).days
                if age_days > max_source_age_days:
                    claim_id = str(claim.get("claim_id", "") or "").strip()
                    stale_ids.add(claim_id)
                    stale_claims.append({
                        "claim_id": claim_id,
                        "text": str(claim.get("text", "") or "")[:200],
                        "age_days": int(age_days),
                        "source_date": source_dt.date().isoformat(),
                    })
                elif age_days < -1:
                    text = str(claim.get("text", "") or "").strip()
                    if text:
                        future_source_claims.append(text[:160])

            if enforce_date_conflicts:
                key = self._temporal_claim_key(str(claim.get("text", "") or ""))
                if not key:
                    continue
                anchor_dt = (
                    as_of_dt
                    if isinstance(as_of_dt, datetime)
                    else (
                        period_end_dt
                        if isinstance(period_end_dt, datetime)
                        else (text_dates[0] if text_dates else None)
                    )
                )
                if not isinstance(anchor_dt, datetime):
                    continue
                conflict_index.setdefault(key, set()).add(anchor_dt.date().isoformat())

        temporal_conflicts: list[dict[str, object]] = []
        if require_as_of_alignment and len(as_of_values) > 1:
            temporal_conflicts.append({
                "kind": "as_of_misalignment",
                "observed_as_of_values": sorted(as_of_values),
            })
        if future_source_claims:
            temporal_conflicts.append({
                "kind": "future_source_date",
                "claims": future_source_claims[:10],
            })
        if enforce_date_conflicts:
            for key, values in conflict_index.items():
                if len(values) <= 1:
                    continue
                temporal_conflicts.append({
                    "kind": "cross_claim_date_conflict",
                    "claim_key": key,
                    "observed_dates": sorted(values),
                })

        updated_claims = [dict(item) for item in claims]
        if stale_ids:
            for claim in updated_claims:
                claim_id = str(claim.get("claim_id", "") or "").strip()
                if claim_id not in stale_ids:
                    continue
                claim["status"] = "stale"
                claim["reason_code"] = "claim_stale_source"
                lifecycle = claim.get("lifecycle", [])
                if isinstance(lifecycle, str):
                    lifecycle = [lifecycle]
                if not isinstance(lifecycle, list):
                    lifecycle = []
                lifecycle_norm = [
                    str(step or "").strip().lower()
                    for step in lifecycle
                    if str(step or "").strip()
                ]
                if "stale" not in lifecycle_norm:
                    lifecycle_norm.append("stale")
                claim["lifecycle"] = lifecycle_norm

        metadata = dict(verification.metadata) if isinstance(verification.metadata, dict) else {}
        metadata["claim_lifecycle"] = updated_claims
        metadata["claim_status_counts"] = self._claim_counts(updated_claims)
        metadata["claim_reason_codes"] = sorted({
            self._normalize_claim_reason_code(
                status=str(item.get("status", "") or ""),
                reason_code=str(item.get("reason_code", "") or ""),
            )
            for item in updated_claims
        })
        metadata["temporal_consistency"] = {
            "enabled": True,
            "reference_as_of": reference_date.isoformat(),
            "require_as_of_alignment": require_as_of_alignment,
            "enforce_cross_claim_date_conflict_check": enforce_date_conflicts,
            "max_source_age_days": max_source_age_days,
            "stale_claim_count": len(stale_claims),
            "conflict_count": len(temporal_conflicts),
        }
        if stale_claims:
            metadata["stale_claims"] = stale_claims[:20]
        if temporal_conflicts:
            metadata["temporal_conflicts"] = temporal_conflicts[:20]

        if not stale_claims and not temporal_conflicts:
            return self._verification_with_metadata(
                verification,
                metadata=metadata,
            )

        fail_reason_code = "claim_stale_source" if stale_claims else "temporal_conflict"
        feedback_lines = [str(verification.feedback or "").strip()]
        if stale_claims:
            feedback_lines.append(
                "Temporal gate failed: stale source dates exceeded max_source_age_days.",
            )
        if temporal_conflicts:
            feedback_lines.append(
                "Temporal gate failed: as_of alignment or cross-claim date consistency violation.",
            )
        return self._verification_with_metadata(
            verification,
            metadata=metadata,
            passed=False,
            outcome="fail",
            reason_code=fail_reason_code,
            feedback="\n".join(line for line in feedback_lines if line),
            severity_class="semantic",
            confidence=min(float(verification.confidence or 0.5), 0.45),
        )

    def _enforce_synthesis_claim_gate(
        self,
        *,
        subtask: Subtask,
        verification: VerificationResult,
        contract: dict[str, object],
    ) -> VerificationResult:
        if not subtask.is_synthesis:
            return verification
        claims = self._claims_from_verification(verification)
        if not claims:
            return verification

        counts = self._claim_counts(claims)
        ratios = self._claim_ratios(counts)
        min_supported_ratio = self._to_ratio(contract.get("min_supported_ratio", 0.75), 0.75)
        max_unverified_ratio = self._to_ratio(contract.get("max_unverified_ratio", 0.25), 0.25)
        max_contradicted = self._to_non_negative_int(
            contract.get("max_contradicted_count", 0),
            0,
        )
        final_gate = contract.get("final_gate", {})
        critical_support_floor = 1.0
        if isinstance(final_gate, dict):
            critical_support_floor = self._to_ratio(
                final_gate.get("critical_claim_support_ratio", 1.0),
                1.0,
            )

        fail_reasons: list[str] = []
        reason_code = ""
        if counts["contradicted"] > max_contradicted:
            fail_reasons.append(
                "Synthesis claim gate failed: contradicted claims exceed contract threshold.",
            )
            reason_code = "claim_contradicted"
        if counts["critical_supported"] < counts["critical_total"]:
            if counts["critical_contradicted"] > 0:
                fail_reasons.append(
                    "Synthesis claim gate failed: contradicted critical claims detected.",
                )
                reason_code = "claim_contradicted"
            else:
                fail_reasons.append(
                    "Synthesis claim gate failed: unsupported critical claims remain.",
                )
                if not reason_code:
                    reason_code = "claim_insufficient_evidence"
        if ratios["critical_support_ratio"] < critical_support_floor:
            fail_reasons.append(
                "Synthesis claim gate failed: critical claim support ratio below threshold.",
            )
            if not reason_code:
                reason_code = "coverage_below_threshold"
        if ratios["supported_ratio"] < min_supported_ratio:
            fail_reasons.append(
                "Synthesis claim gate failed: supported-claim ratio below threshold.",
            )
            if not reason_code:
                reason_code = "coverage_below_threshold"
        if ratios["unverified_ratio"] > max_unverified_ratio:
            fail_reasons.append(
                "Synthesis claim gate failed: unresolved claim ratio above threshold.",
            )
            if not reason_code:
                reason_code = "claim_insufficient_evidence"

        orphan_critical_numeric_claims: list[str] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_type = str(claim.get("claim_type", "") or "").strip().lower()
            criticality = str(claim.get("criticality", "") or "").strip().lower()
            if claim_type != "numeric" or criticality != "critical":
                continue
            refs = claim.get("evidence_refs", [])
            if isinstance(refs, str):
                refs = [refs]
            if not isinstance(refs, list):
                refs = []
            normalized_refs = [
                str(ref or "").strip()
                for ref in refs
                if str(ref or "").strip()
            ]
            if normalized_refs:
                continue
            claim_text = str(claim.get("text", "") or "").strip()
            orphan_critical_numeric_claims.append(claim_text[:160] if claim_text else "")
        if orphan_critical_numeric_claims:
            fail_reasons.append(
                "Synthesis claim gate failed: critical numeric claim missing evidence lineage.",
            )
            if not reason_code:
                reason_code = "claim_insufficient_evidence"

        metadata = dict(verification.metadata) if isinstance(verification.metadata, dict) else {}
        metadata["claim_status_counts"] = counts
        metadata["supported_ratio"] = ratios["supported_ratio"]
        metadata["unverified_ratio"] = ratios["unverified_ratio"]
        metadata["critical_support_ratio"] = ratios["critical_support_ratio"]
        if orphan_critical_numeric_claims:
            metadata["orphan_critical_numeric_claims"] = orphan_critical_numeric_claims[:10]
        metadata["claim_gate_thresholds"] = {
            "min_supported_ratio": min_supported_ratio,
            "max_unverified_ratio": max_unverified_ratio,
            "max_contradicted_count": max_contradicted,
            "critical_claim_support_ratio": critical_support_floor,
        }

        if not fail_reasons:
            return self._verification_with_metadata(verification, metadata=metadata)
        return self._verification_with_metadata(
            verification,
            metadata=metadata,
            passed=False,
            outcome="fail",
            reason_code=reason_code or "coverage_below_threshold",
            feedback="\n".join([
                *(part for part in [verification.feedback or ""] if part),
                *fail_reasons,
            ]),
            severity_class="semantic",
            confidence=min(float(verification.confidence or 0.5), 0.45),
        )

    @staticmethod
    def _artifact_provenance_evidence(
        *,
        task_id: str,
        subtask_id: str,
        tool_calls: list[ToolCallRecord] | None,
        existing_ids: set[str],
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for call in tool_calls or []:
            tool = str(getattr(call, "tool", "") or "").strip().lower()
            if tool not in {"write_file", "document_write"}:
                continue
            result = getattr(call, "result", None)
            if result is None or not bool(getattr(result, "success", False)):
                continue
            args = getattr(call, "args", {})
            if not isinstance(args, dict):
                args = {}
            data = getattr(result, "data", {})
            if not isinstance(data, dict):
                data = {}
            relpath = str(
                args.get("path")
                or args.get("file_path")
                or data.get("path")
                or "",
            ).strip()
            if not relpath:
                continue
            content = ""
            if tool == "write_file":
                content = str(args.get("content", "") or "")
            else:
                parts: list[str] = []
                title = str(args.get("title", "") or "").strip()
                if title:
                    parts.append(title)
                body = str(args.get("content", "") or "")
                if body:
                    parts.append(body)
                sections = args.get("sections", [])
                if isinstance(sections, list):
                    for section in sections[:8]:
                        if not isinstance(section, dict):
                            continue
                        heading = str(section.get("heading", "") or "").strip()
                        if heading:
                            parts.append(heading)
                        section_body = str(section.get("body", "") or "")
                        if section_body:
                            parts.append(section_body)
                content = "\n\n".join(parts)
            payload = f"{tool}|{subtask_id}|{relpath}|{content[:200]}"
            evidence_id = "EV-WRITE-" + hashlib.sha1(
                payload.encode("utf-8", errors="replace"),
            ).hexdigest().upper()[:10]
            if evidence_id in existing_ids:
                continue
            existing_ids.add(evidence_id)
            sha256 = ""
            size_bytes = 0
            if content:
                encoded = content.encode("utf-8", errors="replace")
                size_bytes = len(encoded)
                sha256 = hashlib.sha256(encoded).hexdigest()
            records.append({
                "evidence_id": evidence_id,
                "task_id": task_id,
                "subtask_id": subtask_id,
                "tool": tool,
                "evidence_kind": "artifact",
                "tool_call_id": str(getattr(call, "call_id", "") or ""),
                "query": relpath,
                "source_url": "",
                "facets": {"artifact_path": relpath[:120]},
                "artifact_workspace_relpath": relpath,
                "artifact_sha256": sha256,
                "artifact_size_bytes": int(size_bytes),
                "snippet": f"{tool}:{relpath}",
                "context_text": f"{tool} wrote {relpath}",
                "quality": 1.0,
                "created_at": str(getattr(call, "timestamp", "") or datetime.now().isoformat()),
            })
        return records

    @staticmethod
    def _claim_evidence_links(
        *,
        claims: list[dict[str, object]],
        evidence_records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        evidence_ids: list[str] = []
        source_index: dict[str, str] = {}
        artifact_index: dict[str, str] = {}
        for record in evidence_records:
            if not isinstance(record, dict):
                continue
            evidence_id = str(record.get("evidence_id", "") or "").strip()
            if not evidence_id:
                continue
            evidence_ids.append(evidence_id)
            source_url = str(record.get("source_url", "") or "").strip()
            if source_url and source_url not in source_index:
                source_index[source_url] = evidence_id
            artifact_path = str(record.get("artifact_workspace_relpath", "") or "").strip()
            if artifact_path and artifact_path not in artifact_index:
                artifact_index[artifact_path] = evidence_id
            facets = record.get("facets", {})
            if isinstance(facets, dict):
                facet_path = str(facets.get("artifact_path", "") or "").strip()
                if facet_path and facet_path not in artifact_index:
                    artifact_index[facet_path] = evidence_id

        links: list[dict[str, object]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id", "") or "").strip()
            if not claim_id:
                continue
            refs = claim.get("evidence_refs", [])
            if isinstance(refs, str):
                refs = [refs]
            if not isinstance(refs, list):
                refs = []
            matched_ids: list[str] = []
            for ref in refs:
                ref_text = str(ref or "").strip()
                if not ref_text:
                    continue
                if ref_text in evidence_ids:
                    matched_ids.append(ref_text)
                    continue
                matched_source = False
                for source_url, evidence_id in source_index.items():
                    if ref_text == source_url or ref_text in source_url or source_url in ref_text:
                        matched_ids.append(evidence_id)
                        matched_source = True
                        break
                if matched_source:
                    continue
                for artifact_path, evidence_id in artifact_index.items():
                    if (
                        ref_text == artifact_path
                        or ref_text.endswith(artifact_path)
                        or artifact_path.endswith(ref_text)
                    ):
                        matched_ids.append(evidence_id)
                        break
            for evidence_id in matched_ids:
                key = (claim_id, evidence_id)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                links.append({
                    "claim_id": claim_id,
                    "evidence_id": evidence_id,
                    "link_type": "supporting",
                    "score": 1.0,
                    "metadata": {},
                })
        return links

    async def _persist_claim_validity_artifacts(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
        evidence_records: list[dict],
        tool_calls: list[ToolCallRecord] | None = None,
    ) -> None:
        claims = self._claims_from_verification(verification)
        if not claims:
            return
        normalized_evidence = [
            item for item in evidence_records if isinstance(item, dict)
        ]
        existing_ids = {
            str(item.get("evidence_id", "") or "").strip()
            for item in normalized_evidence
            if str(item.get("evidence_id", "") or "").strip()
        }
        provenance_records = self._artifact_provenance_evidence(
            task_id=task.id,
            subtask_id=subtask.id,
            tool_calls=tool_calls,
            existing_ids=existing_ids,
        )
        if provenance_records:
            normalized_evidence = merge_evidence_records(
                normalized_evidence,
                provenance_records,
            )
        counts = self._claim_counts(claims)
        ratios = self._claim_ratios(counts)
        run_id = self._task_run_id(task)
        phase_id = str(getattr(subtask, "phase_id", "") or "")
        links = self._claim_evidence_links(
            claims=claims,
            evidence_records=normalized_evidence,
        )
        claim_results = []
        for claim in claims:
            status = str(claim.get("status", "extracted") or "extracted").strip().lower()
            claim_results.append({
                "claim_id": str(claim.get("claim_id", "") or "").strip(),
                "status": status,
                "reason_code": self._normalize_claim_reason_code(
                    status=status,
                    reason_code=str(claim.get("reason_code", "") or ""),
                ),
                "verifier": "verification_gates",
                "confidence": float(verification.confidence or 0.0),
                "metadata": {
                    "claim_type": str(claim.get("claim_type", "qualitative") or "qualitative"),
                    "criticality": str(claim.get("criticality", "important") or "important"),
                },
            })
        try:
            await self._memory.insert_artifact_claims(
                task_id=task.id,
                run_id=run_id,
                subtask_id=subtask.id,
                phase_id=phase_id,
                claims=claims,
            )
            await self._memory.insert_claim_verification_results(
                task_id=task.id,
                run_id=run_id,
                subtask_id=subtask.id,
                phase_id=phase_id,
                results=claim_results,
            )
            if links:
                await self._memory.insert_claim_evidence_links(
                    task_id=task.id,
                    run_id=run_id,
                    subtask_id=subtask.id,
                    links=links,
                )
            await self._memory.insert_artifact_validity_summary(
                task_id=task.id,
                run_id=run_id,
                subtask_id=subtask.id,
                phase_id=phase_id,
                extracted_count=int(counts["extracted"]),
                supported_count=int(counts["supported"]),
                contradicted_count=int(counts["contradicted"]),
                insufficient_evidence_count=int(counts["insufficient_evidence"]),
                pruned_count=int(counts["pruned"]),
                supported_ratio=float(ratios["supported_ratio"]),
                gate_decision="pass" if verification.passed else "fail",
                reason_code=str(verification.reason_code or ""),
                metadata={
                    "critical_total": int(counts["critical_total"]),
                    "critical_supported": int(counts["critical_supported"]),
                    "critical_support_ratio": float(ratios["critical_support_ratio"]),
                    "validity_contract_hash": str(
                        getattr(subtask, "validity_contract_hash", "") or "",
                    ),
                },
            )
        except Exception:
            logger.debug(
                "Failed persisting claim validity artifacts for %s/%s",
                task.id,
                subtask.id,
                exc_info=True,
            )

    def _verified_context_for_synthesis(
        self,
        *,
        task: Task,
        subtask: Subtask,
    ) -> tuple[bool, str, str]:
        contract = self._validity_contract_for_subtask(subtask)
        claim_extraction = contract.get("claim_extraction", {})
        claim_extraction_enabled = isinstance(claim_extraction, dict) and self._to_bool(
            claim_extraction.get("enabled", False),
            False,
        )
        final_gate = contract.get("final_gate", {})
        enforce_verified_context = isinstance(final_gate, dict) and self._to_bool(
            final_gate.get("enforce_verified_context_only", False),
            False,
        )
        if not (subtask.is_synthesis and claim_extraction_enabled and enforce_verified_context):
            return True, "", ""

        graph = self._claim_graph_state(task)
        supported_by_subtask = graph.get("supported_by_subtask", {})
        if not isinstance(supported_by_subtask, dict):
            supported_by_subtask = {}
        unresolved_by_subtask = graph.get("unresolved_by_subtask", {})
        if not isinstance(unresolved_by_subtask, dict):
            unresolved_by_subtask = {}

        lines: list[str] = []
        total_supported = 0
        for subtask_id, claims in supported_by_subtask.items():
            if not isinstance(claims, list):
                continue
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                text = str(claim.get("text", "") or "").strip()
                if not text:
                    continue
                total_supported += 1
                lines.append(f"- [{subtask_id}] {text}")
        total_unresolved = sum(
            len(claims)
            for claims in unresolved_by_subtask.values()
            if isinstance(claims, list)
        )
        if total_supported <= 0 and total_unresolved <= 0:
            return True, "", ""
        if total_supported <= 0:
            return (
                False,
                "",
                "Synthesis gate blocked: no supported claims available in verified context bundle.",
            )
        bundle = "\n".join(lines[:80]).strip()
        return True, bundle, ""

    def _enforce_required_fact_checker(
        self,
        *,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
    ) -> VerificationResult:
        if not self._requires_fact_checker_for_subtask(subtask):
            return verification
        tool_calls = list(result.tool_calls or [])
        if not self._fact_checker_used(tool_calls):
            result.status = SubtaskResultStatus.FAILED
            metadata = (
                dict(verification.metadata)
                if isinstance(verification.metadata, dict)
                else {}
            )
            metadata["required_tool"] = "fact_checker"
            metadata["required_verifier_missing"] = True
            details = (
                "Synthesis requires fact grounding, but no successful "
                "`fact_checker` invocation was observed."
            )
            return VerificationResult(
                tier=max(verification.tier, int(subtask.verification_tier or 1)),
                passed=False,
                confidence=min(verification.confidence, 0.3),
                checks=list(verification.checks or []),
                feedback=details,
                outcome="fail",
                reason_code="required_verifier_missing",
                severity_class="semantic",
                metadata=metadata,
            )

        contract = self._validity_contract_for_subtask(subtask)
        claim_extraction = contract.get("claim_extraction", {})
        claim_extraction_enabled = isinstance(claim_extraction, dict) and self._to_bool(
            claim_extraction.get("enabled", False),
            False,
        )
        verdict_count = self._fact_checker_verdict_count(tool_calls)
        if claim_extraction_enabled and verdict_count <= 0:
            result.status = SubtaskResultStatus.FAILED
            metadata = (
                dict(verification.metadata)
                if isinstance(verification.metadata, dict)
                else {}
            )
            metadata["required_tool"] = "fact_checker"
            metadata["required_verifier_empty"] = True
            metadata["fact_checker_verdict_count"] = 0
            details = (
                "Synthesis requires claim-level fact grounding, but `fact_checker` "
                "returned no claim verdicts."
            )
            return VerificationResult(
                tier=max(verification.tier, int(subtask.verification_tier or 1)),
                passed=False,
                confidence=min(verification.confidence, 0.3),
                checks=list(verification.checks or []),
                feedback=details,
                outcome="fail",
                reason_code="required_verifier_empty",
                severity_class="semantic",
                metadata=metadata,
            )

        return verification

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
        contract = self._validity_contract_for_subtask(subtask)
        required_verification_tier = self._synthesis_verification_floor(subtask)

        # Mark running and emit event (under lock for parallel safety)
        async with self._state_lock:
            subtask.status = SubtaskStatus.RUNNING
            if subtask.verification_tier < required_verification_tier:
                subtask.verification_tier = required_verification_tier
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
        is_iteration_retry, iteration_strategy = self._iteration_retry_mode(subtask)
        targeted_iteration_retry = (
            is_iteration_retry and iteration_strategy == "targeted_remediation"
        )
        retry_context = self._retry.build_retry_context(attempts)
        retry_context = self._augment_retry_context_for_outputs(
            subtask=subtask,
            attempts=attempts,
            strategy=retry_strategy,
            expected_deliverables=expected_deliverables,
            base_context=retry_context,
        )
        prior_iteration_feedback = str(
            getattr(subtask, "iteration_last_gate_summary", "") or "",
        ).strip()
        if prior_iteration_feedback:
            if is_iteration_retry and iteration_strategy == "full_rerun":
                retry_context = (
                    f"{retry_context}\n\n"
                    "FAILED ITERATION GATES FROM PRIOR ATTEMPT:\n"
                    f"{prior_iteration_feedback}\n"
                    "You may rerun the full phase output to resolve these gates."
                ).strip()
            else:
                retry_context = (
                    f"{retry_context}\n\n"
                    "FAILED ITERATION GATES FROM PRIOR ATTEMPT:\n"
                    f"{prior_iteration_feedback}\n"
                    "Preserve already-correct content and fix only the listed gaps."
                ).strip()

        gate_passed = True
        verified_context_bundle = ""
        gate_error = ""
        if subtask.is_synthesis:
            seal_passed, seal_mismatches, validated_seals = self._validate_artifact_seals(
                task=task,
            )
            self._emit(ARTIFACT_SEAL_VALIDATION, task.id, {
                "subtask_id": subtask.id,
                "phase_id": subtask.phase_id,
                "passed": bool(seal_passed),
                "validated_seal_count": int(validated_seals),
                "mismatch_count": len(seal_mismatches),
            })
            if not seal_passed:
                first = seal_mismatches[0] if seal_mismatches else {}
                first_path = str(first.get("path", "") or "").strip()
                first_reason = str(first.get("reason", "") or "").strip()
                details = []
                if first_path:
                    details.append(first_path)
                if first_reason:
                    details.append(first_reason)
                detail_suffix = f" ({', '.join(details)})" if details else ""
                gate_error = (
                    "Synthesis gate blocked: artifact seal validation failed"
                    f"{detail_suffix}."
                )
                blocked = SubtaskResult(
                    status=SubtaskResultStatus.FAILED,
                    summary=gate_error,
                )
                blocked_verification = VerificationResult(
                    tier=max(2, int(subtask.verification_tier or required_verification_tier)),
                    passed=False,
                    confidence=0.0,
                    feedback=gate_error,
                    outcome="fail",
                    reason_code="artifact_seal_invalid",
                    severity_class="semantic",
                    metadata={
                        "artifact_seal_validation_failed": True,
                        "artifact_seal_mismatches": seal_mismatches[:10],
                    },
                )
                return subtask, blocked, blocked_verification
            gate_passed, verified_context_bundle, gate_error = self._verified_context_for_synthesis(
                task=task,
                subtask=subtask,
            )
            claim_graph = self._claim_graph_state(task)
            supported_total = 0
            unresolved_total = 0
            supported_by_subtask = claim_graph.get("supported_by_subtask", {})
            if isinstance(supported_by_subtask, dict):
                supported_total = sum(
                    len(value) for value in supported_by_subtask.values()
                    if isinstance(value, list)
                )
            unresolved_by_subtask = claim_graph.get("unresolved_by_subtask", {})
            if isinstance(unresolved_by_subtask, dict):
                unresolved_total = sum(
                    len(value) for value in unresolved_by_subtask.values()
                    if isinstance(value, list)
                )
            self._emit(SYNTHESIS_INPUT_GATE_DECISION, task.id, {
                "subtask_id": subtask.id,
                "phase_id": subtask.phase_id,
                "passed": bool(gate_passed),
                "supported_claim_count": int(supported_total),
                "unresolved_claim_count": int(unresolved_total),
                "reason": gate_error if not gate_passed else "verified_context_bundle_ready",
            })
            if not gate_passed:
                blocked = SubtaskResult(
                    status=SubtaskResultStatus.FAILED,
                    summary=gate_error,
                )
                blocked_verification = VerificationResult(
                    tier=max(2, int(subtask.verification_tier or required_verification_tier)),
                    passed=False,
                    confidence=0.0,
                    feedback=gate_error,
                    outcome="fail",
                    reason_code="coverage_below_threshold",
                    severity_class="semantic",
                    metadata={
                        "synthesis_input_gate_blocked": True,
                    },
                )
                return subtask, blocked, blocked_verification
            if verified_context_bundle:
                retry_context = (
                    f"{retry_context}\n\n"
                    "VERIFIED CONTEXT BUNDLE (SUPPORTED CLAIMS ONLY):\n"
                    f"{verified_context_bundle}\n"
                    "Use this verified bundle as the primary basis for final synthesis. "
                    "Do not reintroduce unresolved claims."
                ).strip()

        changelog = self._get_changelog(task)

        result, verification = await self._runner.run(
            task, subtask,
            model_tier=escalated_tier,
            retry_context=retry_context,
            changelog=changelog,
            prior_successful_tool_calls=prior_successful_tool_calls,
            prior_evidence_records=prior_evidence_records,
            expected_deliverables=expected_deliverables,
            enforce_deliverable_paths=bool(expected_deliverables) and bool(
                attempts or targeted_iteration_retry
            ),
            edit_existing_only=bool(expected_deliverables) and bool(
                attempts or targeted_iteration_retry
            ),
            retry_strategy=retry_strategy.value,
        )

        verification = self._enforce_required_fact_checker(
            subtask=subtask,
            result=result,
            verification=verification,
        )
        claim_extraction = contract.get("claim_extraction", {})
        claim_policy_enabled = self._to_bool(contract.get("enabled", False), False) and (
            isinstance(claim_extraction, dict)
            and self._to_bool(claim_extraction.get("enabled", False), False)
        )
        if claim_policy_enabled:
            verification = self._apply_intermediate_claim_pruning(
                task=task,
                subtask=subtask,
                result=result,
                verification=verification,
                contract=contract,
            )
            verification = self._enforce_temporal_consistency_gate(
                subtask=subtask,
                verification=verification,
                contract=contract,
            )
            if verification.passed:
                verification = self._enforce_synthesis_claim_gate(
                    subtask=subtask,
                    verification=verification,
                    contract=contract,
                )
            if not verification.passed:
                result.status = SubtaskResultStatus.FAILED
        self._update_claim_graph_from_verification(
            task=task,
            subtask=subtask,
            verification=verification,
        )
        await self._persist_claim_validity_artifacts(
            task=task,
            subtask=subtask,
            verification=verification,
            evidence_records=result.evidence_records,
            tool_calls=result.tool_calls,
        )

        return subtask, result, verification

    # ------------------------------------------------------------------
    # Outcome handlers
    # ------------------------------------------------------------------

    def _phase_iteration_policy(self, subtask: Subtask) -> IterationPolicy | None:
        if not self._iteration_enabled or self._process is None:
            return None
        phases = list(getattr(self._process, "phases", []) or [])
        if not phases:
            return None

        subtask_id = str(getattr(subtask, "id", "") or "").strip()
        phase_id = str(getattr(subtask, "phase_id", "") or "").strip()
        for phase in phases:
            phase_key = str(getattr(phase, "id", "") or "").strip()
            if not phase_key:
                continue
            if phase_key not in {subtask_id, phase_id}:
                continue
            policy = getattr(phase, "iteration", None)
            if policy is not None and bool(getattr(policy, "enabled", False)):
                return policy

        if len(phases) == 1:
            policy = getattr(phases[0], "iteration", None)
            if policy is not None and bool(getattr(policy, "enabled", False)):
                return policy
        return None

    def _iteration_retry_mode(self, subtask: Subtask) -> tuple[bool, str]:
        policy = self._phase_iteration_policy(subtask)
        if policy is None:
            return False, ""
        strategy = str(getattr(policy, "strategy", "") or "").strip().lower()
        if strategy not in {"targeted_remediation", "full_rerun"}:
            strategy = "targeted_remediation"
        prior_gate_feedback = str(
            getattr(subtask, "iteration_last_gate_summary", "") or "",
        ).strip()
        is_iteration_retry = bool(
            int(getattr(subtask, "iteration_attempt", 0) or 0) > 0
            and prior_gate_feedback,
        )
        return is_iteration_retry, strategy

    def _observe_iteration_runner_invocation(self, subtask: Subtask) -> None:
        policy = self._phase_iteration_policy(subtask)
        if policy is None:
            return
        subtask.iteration_runner_invocations = int(
            max(0, subtask.iteration_runner_invocations) + 1,
        )
        if subtask.iteration_max_attempts <= 0:
            subtask.iteration_max_attempts = int(max(1, policy.max_attempts))

    def _observe_iteration_runtime_usage(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
    ) -> None:
        if self._phase_iteration_policy(subtask) is None:
            return
        self._update_iteration_runtime(task=task, subtask=subtask, result=result)

    def _iteration_runtime_entry(self, task: Task, subtask_id: str) -> dict[str, object]:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        runtime = metadata.get("iteration_runtime")
        if not isinstance(runtime, dict):
            runtime = {}
            metadata["iteration_runtime"] = runtime
        entry = runtime.get(subtask_id)
        if not isinstance(entry, dict):
            entry = {}
            runtime[subtask_id] = entry
        task.metadata = metadata
        return entry

    def _update_iteration_runtime(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
    ) -> dict[str, object]:
        entry = self._iteration_runtime_entry(task, subtask.id)
        if "started_monotonic" not in entry:
            entry["started_monotonic"] = float(time.monotonic())
            entry["started_at"] = datetime.now().isoformat()
        entry["tokens_used"] = int(entry.get("tokens_used", 0) or 0) + int(
            max(0, getattr(result, "tokens_used", 0) or 0),
        )
        counters = getattr(result, "telemetry_counters", None)
        tool_calls_used = 0
        if isinstance(counters, dict):
            tool_calls_used = int(counters.get("tool_calls", 0) or 0)
        if tool_calls_used <= 0:
            tool_calls_used = len(getattr(result, "tool_calls", []) or [])
        entry["tool_calls"] = int(entry.get("tool_calls", 0) or 0) + max(
            0,
            int(tool_calls_used),
        )
        entry["updated_at"] = datetime.now().isoformat()
        return entry

    async def _sync_external_control_state(self, task: Task) -> None:
        """Apply pause/cancel/resume state changes persisted by control APIs."""
        try:
            loaded = self._state.load(task.id)
        except Exception:
            return
        if loaded.status == task.status:
            return
        if loaded.status in {
            TaskStatus.PAUSED,
            TaskStatus.CANCELLED,
            TaskStatus.EXECUTING,
            TaskStatus.PLANNING,
        }:
            task.status = loaded.status

    @staticmethod
    def _iteration_budget_snapshot(
        *,
        policy: IterationPolicy,
        runtime: dict[str, object],
    ) -> dict[str, object]:
        started = runtime.get("started_monotonic")
        elapsed = 0.0
        if isinstance(started, (int, float)) and started > 0:
            elapsed = max(0.0, float(time.monotonic()) - float(started))
        return {
            "used": {
                "elapsed_seconds": round(elapsed, 3),
                "tokens": int(runtime.get("tokens_used", 0) or 0),
                "tool_calls": int(runtime.get("tool_calls", 0) or 0),
            },
            "limits": {
                "max_wall_clock_seconds": int(policy.budget.max_wall_clock_seconds),
                "max_tokens": int(policy.budget.max_tokens),
                "max_tool_calls": int(policy.budget.max_tool_calls),
            },
        }

    @staticmethod
    def _iteration_budget_exhausted_reason(
        *,
        policy: IterationPolicy,
        runtime: dict[str, object],
    ) -> str:
        started = runtime.get("started_monotonic")
        elapsed = 0.0
        if isinstance(started, (int, float)) and started > 0:
            elapsed = max(0.0, float(time.monotonic()) - float(started))
        if (
            int(policy.budget.max_wall_clock_seconds) > 0
            and elapsed > float(policy.budget.max_wall_clock_seconds)
        ):
            return "iteration_budget_exhausted:wall_clock"
        tokens_used = int(runtime.get("tokens_used", 0) or 0)
        if int(policy.budget.max_tokens) > 0 and tokens_used > int(policy.budget.max_tokens):
            return "iteration_budget_exhausted:tokens"
        tool_calls = int(runtime.get("tool_calls", 0) or 0)
        if int(policy.budget.max_tool_calls) > 0 and tool_calls > int(policy.budget.max_tool_calls):
            return "iteration_budget_exhausted:tool_calls"
        return ""

    @staticmethod
    def _format_iteration_gate_failures(
        failures: list[object],
    ) -> str:
        lines = []
        for item in failures:
            gate_id = str(getattr(item, "gate_id", "") or "").strip() or "gate"
            reason = str(getattr(item, "reason_code", "") or "").strip() or "failed"
            detail = str(getattr(item, "detail", "") or "").strip()
            if detail:
                lines.append(f"- {gate_id}: {reason} ({detail})")
            else:
                lines.append(f"- {gate_id}: {reason}")
        return "\n".join(lines).strip()

    def _iteration_replan_cap(self, policy: IterationPolicy) -> int:
        process_cap = int(getattr(policy, "max_replans_after_exhaustion", 0) or 0)
        if process_cap > 0:
            return process_cap
        return int(
            max(
                0,
                getattr(
                    self._config.execution,
                    "max_iteration_replans_after_exhaustion",
                    2,
                ) or 0,
            ),
        )

    @staticmethod
    def _iteration_exhaustion_fingerprint(
        *,
        subtask: Subtask,
        terminal_reason: str,
        gate_summary: str,
    ) -> str:
        return "|".join([
            str(subtask.id or "").strip(),
            str(terminal_reason or "").strip().lower(),
            str(gate_summary or "").strip().lower(),
        ])

    async def _request_iteration_replan(
        self,
        *,
        task: Task,
        subtask: Subtask,
        policy: IterationPolicy,
        terminal_reason: str,
        gate_summary: str,
    ) -> dict[str, str | None] | None:
        if not bool(getattr(policy, "replan_on_exhaustion", True)):
            return None

        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        replan_counts = metadata.get("iteration_replan_counts")
        if not isinstance(replan_counts, dict):
            replan_counts = {}
            metadata["iteration_replan_counts"] = replan_counts
        seen_fingerprints = metadata.get("iteration_exhaustion_fingerprints")
        if not isinstance(seen_fingerprints, dict):
            seen_fingerprints = {}
            metadata["iteration_exhaustion_fingerprints"] = seen_fingerprints

        subtask_key = str(subtask.id or "").strip()
        fingerprint = self._iteration_exhaustion_fingerprint(
            subtask=subtask,
            terminal_reason=terminal_reason,
            gate_summary=gate_summary,
        )
        prior_fingerprints = seen_fingerprints.get(subtask_key)
        if not isinstance(prior_fingerprints, list):
            prior_fingerprints = []
        if fingerprint in prior_fingerprints:
            return None

        cap = self._iteration_replan_cap(policy)
        prior_count = int(replan_counts.get(subtask_key, 0) or 0)
        if cap > 0 and prior_count >= cap:
            return None

        replan_counts[subtask_key] = prior_count + 1
        subtask.iteration_replan_count = int(replan_counts[subtask_key])
        prior_fingerprints.append(fingerprint)
        if len(prior_fingerprints) > 8:
            prior_fingerprints = prior_fingerprints[-8:]
        seen_fingerprints[subtask_key] = prior_fingerprints
        task.metadata = metadata
        self._state.save(task)

        return {
            "reason": f"iteration_loop_exhausted:{terminal_reason}",
            "failed_subtask_id": subtask.id,
            "verification_feedback": gate_summary,
        }

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
        if not self._iteration_enabled:
            return
        loop_run_id = str(subtask.iteration_loop_run_id or "").strip()
        if not loop_run_id:
            return
        try:
            await self._memory.upsert_iteration_run(
                loop_run_id=loop_run_id,
                task_id=task.id,
                run_id=self._task_run_id(task),
                subtask_id=subtask.id,
                phase_id=str(getattr(subtask, "phase_id", "") or ""),
                policy_snapshot=asdict(policy),
                terminal_reason=terminal_reason,
                attempt_count=int(subtask.iteration_attempt),
                replan_count=int(subtask.iteration_replan_count),
                exhaustion_fingerprint=exhaustion_fingerprint,
                metadata={
                    "iteration_runner_invocations": int(
                        subtask.iteration_runner_invocations,
                    ),
                    "iteration_no_improvement_count": int(
                        subtask.iteration_no_improvement_count,
                    ),
                    "iteration_best_score": subtask.iteration_best_score,
                    "iteration_last_gate_summary": gate_summary,
                },
            )
            attempt_id = await self._memory.insert_iteration_attempt(
                loop_run_id=loop_run_id,
                task_id=task.id,
                run_id=self._task_run_id(task),
                subtask_id=subtask.id,
                phase_id=str(getattr(subtask, "phase_id", "") or ""),
                attempt_index=attempt_index,
                status=status,
                summary=gate_summary,
                gate_summary={
                    "blocking_failures": [
                        getattr(item, "gate_id", "")
                        for item in (evaluation.blocking_failures if evaluation else [])
                    ],
                    "advisory_failures": [
                        getattr(item, "gate_id", "")
                        for item in (evaluation.advisory_failures if evaluation else [])
                    ],
                },
                budget_snapshot=budget_snapshot,
            )
            if evaluation is None:
                return
            for gate in evaluation.results:
                await self._memory.insert_iteration_gate_result(
                    loop_run_id=loop_run_id,
                    attempt_id=attempt_id,
                    task_id=task.id,
                    run_id=self._task_run_id(task),
                    subtask_id=subtask.id,
                    phase_id=str(getattr(subtask, "phase_id", "") or ""),
                    attempt_index=attempt_index,
                    gate_id=str(getattr(gate, "gate_id", "") or ""),
                    gate_type=str(getattr(gate, "gate_type", "") or ""),
                    status=str(getattr(gate, "status", "") or ""),
                    blocking=bool(getattr(gate, "blocking", False)),
                    reason_code=str(getattr(gate, "reason_code", "") or ""),
                    measured_value=getattr(gate, "measured_value", None),
                    threshold_value=getattr(gate, "threshold_value", None),
                    detail=str(getattr(gate, "detail", "") or ""),
                )
        except Exception:
            logger.debug(
                "Failed persisting iteration evaluation for %s/%s",
                task.id,
                subtask.id,
                exc_info=True,
            )

    async def _handle_iteration_after_success(
        self,
        *,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
    ) -> dict[str, str | None] | None:
        policy = self._phase_iteration_policy(subtask)
        if policy is None:
            await self._handle_success(task, subtask, result, verification)
            return None

        if not subtask.iteration_loop_run_id:
            subtask.iteration_loop_run_id = f"iter-{uuid.uuid4().hex[:10]}"
            self._emit(ITERATION_STARTED, task.id, {
                "subtask_id": subtask.id,
                "phase_id": subtask.phase_id,
                "loop_run_id": subtask.iteration_loop_run_id,
                "max_attempts": int(policy.max_attempts),
                "max_runner_invocations": int(policy.max_total_runner_invocations),
            })

        runtime = self._iteration_runtime_entry(task, subtask.id)
        if "started_monotonic" not in runtime:
            runtime["started_monotonic"] = float(time.monotonic())
            runtime["started_at"] = datetime.now().isoformat()
        runtime["updated_at"] = datetime.now().isoformat()
        budget_snapshot = self._iteration_budget_snapshot(policy=policy, runtime=runtime)
        budget_reason = self._iteration_budget_exhausted_reason(policy=policy, runtime=runtime)

        evaluation: IterationEvaluation | None = None
        if not budget_reason:
            evaluation = await self._iteration_gates.evaluate(
                policy=policy,
                result=result,
                verification=verification,
                workspace=Path(task.workspace) if task.workspace else None,
                expected_deliverables=self._expected_deliverables_for_subtask(subtask),
            )

        attempt_index = int(max(0, subtask.iteration_attempt) + 1)
        subtask.iteration_attempt = attempt_index
        if subtask.iteration_max_attempts <= 0:
            subtask.iteration_max_attempts = int(max(1, policy.max_attempts))

        if evaluation and evaluation.score_hint is not None:
            score = float(evaluation.score_hint)
            best = subtask.iteration_best_score
            if best is None or score > best:
                subtask.iteration_best_score = score
                subtask.iteration_no_improvement_count = 0
            else:
                subtask.iteration_no_improvement_count = int(
                    max(0, subtask.iteration_no_improvement_count) + 1,
                )

        blocking_failures = list(evaluation.blocking_failures if evaluation else [])
        has_blocking_failures = bool(blocking_failures) or bool(budget_reason)
        if not has_blocking_failures:
            subtask.iteration_terminal_reason = "passed"
            subtask.iteration_last_gate_summary = ""
            await self._persist_iteration_evaluation(
                task=task,
                subtask=subtask,
                policy=policy,
                evaluation=evaluation,
                attempt_index=attempt_index,
                status="completed",
                gate_summary="all blocking iteration gates passed",
                budget_snapshot=budget_snapshot,
                terminal_reason="passed",
            )
            self._emit(ITERATION_COMPLETED, task.id, {
                "subtask_id": subtask.id,
                "phase_id": subtask.phase_id,
                "loop_run_id": subtask.iteration_loop_run_id,
                "attempt": attempt_index,
                "max_attempts": int(policy.max_attempts),
            })
            await self._handle_success(task, subtask, result, verification)
            return None

        gate_summary = (
            budget_reason
            if budget_reason
            else self._format_iteration_gate_failures(blocking_failures)
        )
        subtask.iteration_last_gate_summary = gate_summary

        attempts_exhausted = attempt_index >= int(max(1, policy.max_attempts))
        invocations_exhausted = (
            int(policy.max_total_runner_invocations) > 0
            and subtask.iteration_runner_invocations >= int(policy.max_total_runner_invocations)
        )
        no_improvement_exhausted = (
            int(policy.stop_on_no_improvement_attempts) > 0
            and subtask.iteration_no_improvement_count
            >= int(policy.stop_on_no_improvement_attempts)
        )

        terminal_reason = ""
        if budget_reason:
            terminal_reason = "iteration_budget_exhausted"
        elif no_improvement_exhausted:
            terminal_reason = "no_improvement"
        elif invocations_exhausted:
            terminal_reason = "max_runner_invocations_exhausted"
        elif attempts_exhausted:
            terminal_reason = "max_attempts_exhausted"

        self._emit(ITERATION_GATE_FAILED, task.id, {
            "subtask_id": subtask.id,
            "phase_id": subtask.phase_id,
            "loop_run_id": subtask.iteration_loop_run_id,
            "attempt": attempt_index,
            "max_attempts": int(policy.max_attempts),
            "terminal_reason": terminal_reason,
            "gate_summary": gate_summary,
        })

        if not terminal_reason:
            async with self._state_lock:
                subtask.status = SubtaskStatus.PENDING
                subtask.summary = gate_summary or "Iteration gate failed"
                subtask.active_issue = gate_summary
                task.update_subtask(
                    subtask.id,
                    status=SubtaskStatus.PENDING,
                    summary=subtask.summary,
                    active_issue=subtask.active_issue,
                    iteration_attempt=subtask.iteration_attempt,
                    iteration_best_score=subtask.iteration_best_score,
                    iteration_no_improvement_count=subtask.iteration_no_improvement_count,
                    iteration_last_gate_summary=subtask.iteration_last_gate_summary,
                )
                self._state.save(task)
            await self._persist_iteration_evaluation(
                task=task,
                subtask=subtask,
                policy=policy,
                evaluation=evaluation,
                attempt_index=attempt_index,
                status="retrying",
                gate_summary=gate_summary,
                budget_snapshot=budget_snapshot,
            )
            self._emit(ITERATION_RETRYING, task.id, {
                "subtask_id": subtask.id,
                "phase_id": subtask.phase_id,
                "loop_run_id": subtask.iteration_loop_run_id,
                "attempt": attempt_index,
                "next_attempt": attempt_index + 1,
                "max_attempts": int(policy.max_attempts),
                "gate_summary": gate_summary,
            })
            return None

        subtask.iteration_terminal_reason = terminal_reason
        exhaustion_fingerprint = self._iteration_exhaustion_fingerprint(
            subtask=subtask,
            terminal_reason=terminal_reason,
            gate_summary=gate_summary,
        )
        await self._persist_iteration_evaluation(
            task=task,
            subtask=subtask,
            policy=policy,
            evaluation=evaluation,
            attempt_index=attempt_index,
            status="terminal",
            gate_summary=gate_summary,
            budget_snapshot=budget_snapshot,
            terminal_reason=terminal_reason,
            exhaustion_fingerprint=exhaustion_fingerprint,
        )
        self._emit(ITERATION_TERMINAL, task.id, {
            "subtask_id": subtask.id,
            "phase_id": subtask.phase_id,
            "loop_run_id": subtask.iteration_loop_run_id,
            "attempt": attempt_index,
            "terminal_reason": terminal_reason,
            "gate_summary": gate_summary,
        })

        replan_request = await self._request_iteration_replan(
            task=task,
            subtask=subtask,
            policy=policy,
            terminal_reason=terminal_reason,
            gate_summary=gate_summary,
        )
        if replan_request is not None:
            async with self._state_lock:
                subtask.status = SubtaskStatus.FAILED
                subtask.summary = f"Iteration exhausted: {terminal_reason}"
                subtask.active_issue = gate_summary
                task.update_subtask(
                    subtask.id,
                    status=SubtaskStatus.FAILED,
                    summary=subtask.summary,
                    active_issue=subtask.active_issue,
                    iteration_terminal_reason=subtask.iteration_terminal_reason,
                    iteration_replan_count=subtask.iteration_replan_count,
                )
                self._state.save(task)
            return replan_request

        if subtask.is_critical_path:
            await self._abort_on_critical_path_failure(
                task,
                subtask,
                VerificationResult(
                    tier=max(1, int(subtask.verification_tier or 1)),
                    passed=False,
                    feedback=gate_summary,
                    outcome="fail",
                    reason_code="iteration_exhausted",
                    severity_class="semantic",
                ),
            )
            return None

        async with self._state_lock:
            subtask.status = SubtaskStatus.FAILED
            subtask.summary = f"Iteration exhausted: {terminal_reason}"
            subtask.active_issue = gate_summary
            task.update_subtask(
                subtask.id,
                status=SubtaskStatus.FAILED,
                summary=subtask.summary,
                active_issue=subtask.active_issue,
                iteration_terminal_reason=subtask.iteration_terminal_reason,
            )
            task.add_error(
                subtask.id,
                f"Iteration exhausted ({terminal_reason}): {gate_summary}",
            )
            self._state.save(task)
        return None

    async def _reconcile_iteration_state(self, task: Task) -> None:
        if not self._iteration_enabled:
            return
        try:
            runs = await self._memory.list_iteration_runs(task_id=task.id)
        except Exception:
            return
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        mirror = metadata.get("iteration_sqlite_mirror")
        if not isinstance(mirror, dict):
            mirror = {}
        prior_count = int(mirror.get("run_count", 0) or 0)
        current_count = len(runs)
        latest_by_subtask: dict[str, dict] = {}
        for row in runs:
            if not isinstance(row, dict):
                continue
            subtask_id = str(row.get("subtask_id", "") or "").strip()
            if not subtask_id:
                continue
            prior = latest_by_subtask.get(subtask_id)
            row_sort_key = str(row.get("updated_at", "") or row.get("created_at", ""))
            prior_sort_key = ""
            if isinstance(prior, dict):
                prior_sort_key = str(
                    prior.get("updated_at", "") or prior.get("created_at", ""),
                )
            if prior is None or row_sort_key >= prior_sort_key:
                latest_by_subtask[subtask_id] = row

        hydrated_subtask_ids: list[str] = []
        for subtask in task.plan.subtasks:
            row = latest_by_subtask.get(subtask.id)
            if not isinstance(row, dict):
                continue
            row_metadata = row.get("metadata")
            if not isinstance(row_metadata, dict):
                row_metadata = {}
            policy_snapshot = row.get("policy_snapshot")
            if not isinstance(policy_snapshot, dict):
                policy_snapshot = {}

            updates: dict[str, object] = {}
            loop_run_id = str(row.get("loop_run_id", "") or "").strip()
            if loop_run_id and subtask.iteration_loop_run_id != loop_run_id:
                updates["iteration_loop_run_id"] = loop_run_id

            try:
                attempt_count = max(0, int(row.get("attempt_count", 0) or 0))
            except (TypeError, ValueError):
                attempt_count = 0
            if subtask.iteration_attempt != attempt_count:
                updates["iteration_attempt"] = attempt_count

            try:
                replan_count = max(0, int(row.get("replan_count", 0) or 0))
            except (TypeError, ValueError):
                replan_count = 0
            if subtask.iteration_replan_count != replan_count:
                updates["iteration_replan_count"] = replan_count

            terminal_reason = str(row.get("terminal_reason", "") or "")
            if subtask.iteration_terminal_reason != terminal_reason:
                updates["iteration_terminal_reason"] = terminal_reason

            try:
                runner_invocations = max(
                    0,
                    int(row_metadata.get("iteration_runner_invocations", 0) or 0),
                )
            except (TypeError, ValueError):
                runner_invocations = 0
            if subtask.iteration_runner_invocations != runner_invocations:
                updates["iteration_runner_invocations"] = runner_invocations

            try:
                no_improvement_count = max(
                    0,
                    int(row_metadata.get("iteration_no_improvement_count", 0) or 0),
                )
            except (TypeError, ValueError):
                no_improvement_count = 0
            if subtask.iteration_no_improvement_count != no_improvement_count:
                updates["iteration_no_improvement_count"] = no_improvement_count

            best_score_raw = row_metadata.get("iteration_best_score", None)
            best_score: float | None
            if best_score_raw in (None, ""):
                best_score = None
            else:
                try:
                    best_score = float(best_score_raw)
                except (TypeError, ValueError):
                    best_score = subtask.iteration_best_score
            if subtask.iteration_best_score != best_score:
                updates["iteration_best_score"] = best_score

            gate_summary = str(
                row_metadata.get("iteration_last_gate_summary", "") or "",
            )
            if subtask.iteration_last_gate_summary != gate_summary:
                updates["iteration_last_gate_summary"] = gate_summary

            try:
                max_attempts = max(0, int(policy_snapshot.get("max_attempts", 0) or 0))
            except (TypeError, ValueError):
                max_attempts = 0
            if max_attempts > 0 and subtask.iteration_max_attempts != max_attempts:
                updates["iteration_max_attempts"] = max_attempts

            if updates:
                for field_name, field_value in updates.items():
                    setattr(subtask, field_name, field_value)
                hydrated_subtask_ids.append(subtask.id)

        mirror["run_count"] = current_count
        mirror["updated_at"] = datetime.now().isoformat()
        mirror["subtasks"] = {
            subtask_id: {
                "loop_run_id": str(row.get("loop_run_id", "") or ""),
                "attempt_count": int(row.get("attempt_count", 0) or 0),
                "replan_count": int(row.get("replan_count", 0) or 0),
                "terminal_reason": str(row.get("terminal_reason", "") or ""),
            }
            for subtask_id, row in latest_by_subtask.items()
            if isinstance(row, dict)
        }
        metadata["iteration_sqlite_mirror"] = mirror
        task.metadata = metadata

        if prior_count == current_count and not hydrated_subtask_ids:
            return

        self._state.save(task)
        self._emit(ITERATION_STATE_RECONCILED, task.id, {
            "run_id": self._task_run_id(task),
            "task_id": task.id,
            "previous_count": prior_count,
            "sqlite_count": current_count,
            "hydrated_subtask_ids": hydrated_subtask_ids,
        })

    async def _reconcile_subtask_policy_state(self, task: Task) -> None:
        process = self._process
        phase_by_id: dict[str, object] = {}
        if process is not None:
            for phase in list(getattr(process, "phases", []) or []):
                phase_id = str(getattr(phase, "id", "") or "").strip()
                if phase_id:
                    phase_by_id[phase_id] = phase

        reconciled: list[dict[str, object]] = []
        changed = False
        for subtask in task.plan.subtasks:
            phase_id = str(getattr(subtask, "phase_id", "") or "").strip()
            phase = phase_by_id.get(phase_id)

            before = {
                "model_tier": int(getattr(subtask, "model_tier", 1) or 1),
                "verification_tier": int(getattr(subtask, "verification_tier", 1) or 1),
                "acceptance_criteria": str(getattr(subtask, "acceptance_criteria", "") or ""),
                "validity_contract_hash": str(
                    getattr(subtask, "validity_contract_hash", "") or "",
                ),
            }

            self._apply_subtask_policy_from_process_phase(
                subtask=subtask,
                phase=phase,
            )

            contract_snapshot = (
                dict(subtask.validity_contract_snapshot)
                if isinstance(subtask.validity_contract_snapshot, dict)
                else self._default_validity_contract_for_subtask(subtask)
            )
            contract_final_gate = contract_snapshot.get("final_gate", {})
            if isinstance(contract_final_gate, dict):
                synthesis_floor = max(
                    1,
                    self._to_non_negative_int(
                        contract_final_gate.get("synthesis_min_verification_tier", 2),
                        2,
                    ),
                )
            else:
                synthesis_floor = 2
            if subtask.is_synthesis:
                subtask.verification_tier = max(
                    int(getattr(subtask, "verification_tier", 1) or 1),
                    synthesis_floor,
                )

            self._ensure_subtask_validity_snapshot(subtask=subtask)

            after = {
                "model_tier": int(getattr(subtask, "model_tier", 1) or 1),
                "verification_tier": int(getattr(subtask, "verification_tier", 1) or 1),
                "acceptance_criteria": str(getattr(subtask, "acceptance_criteria", "") or ""),
                "validity_contract_hash": str(
                    getattr(subtask, "validity_contract_hash", "") or "",
                ),
            }
            if after != before:
                changed = True
                reconciled.append({
                    "subtask_id": subtask.id,
                    "phase_id": phase_id,
                    "from": before,
                    "to": after,
                })

        if not changed:
            return
        self._state.save(task)
        self._emit(SUBTASK_POLICY_RECONCILED, task.id, {
            "run_id": self._task_run_id(task),
            "reconciled_subtasks": reconciled,
            "reconciled_count": len(reconciled),
        })

    async def _handle_failure(
        self,
        task: Task,
        subtask: Subtask,
        result: SubtaskResult,
        verification: VerificationResult,
        attempts_by_subtask: dict[str, list[AttemptRecord]],
    ) -> dict[str, str | None] | None:
        """Process a failed subtask: record attempt, retry or replan."""
        self._persist_subtask_evidence(
            task.id,
            subtask.id,
            result.evidence_records,
            tool_calls=result.tool_calls,
        )
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
            subtask=subtask,
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
            self._record_artifact_seals(
                task=task,
                subtask_id=subtask.id,
                tool_calls=result.tool_calls,
            )
            self._record_subtask_validity_metrics(
                task=task,
                subtask=subtask,
                verification=verification,
            )
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

        if (
            strategy == RetryStrategy.UNCONFIRMED_DATA
            and not hard_invariant_failure
            and self._is_placeholder_unconfirmed_failure(verification=verification)
            and subtask.retry_count < subtask.max_retries
        ):
            (
                _resolved_deterministically,
                deterministic_note,
                deterministic_details,
            ) = await self._run_deterministic_placeholder_prepass(
                task=task,
                subtask=subtask,
                verification=verification,
                origin="unconfirmed_data_retry",
            )
            if deterministic_note:
                verification.feedback = (
                    f"{verification.feedback}\n{deterministic_note}".strip()
                    if verification.feedback
                    else deterministic_note
                )
            if deterministic_details:
                metadata = (
                    dict(verification.metadata)
                    if isinstance(verification.metadata, dict)
                    else {}
                )
                metadata["deterministic_placeholder_prepass"] = deterministic_details
                verification.metadata = metadata

        runner_cap_exhausted = False
        iteration_policy = self._phase_iteration_policy(subtask)
        iteration_budget_reason = ""
        if (
            iteration_policy is not None
            and int(iteration_policy.max_total_runner_invocations) > 0
            and int(subtask.iteration_runner_invocations)
            >= int(iteration_policy.max_total_runner_invocations)
        ):
            runner_cap_exhausted = True
        if iteration_policy is not None:
            runtime = self._iteration_runtime_entry(task, subtask.id)
            iteration_budget_reason = self._iteration_budget_exhausted_reason(
                policy=iteration_policy,
                runtime=runtime,
            )

        if (
            not runner_cap_exhausted
            and not iteration_budget_reason
            and subtask.retry_count < subtask.max_retries
        ):
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
            if runner_cap_exhausted:
                extra = (
                    "Retry budget cut off by iteration.max_total_runner_invocations "
                    f"({subtask.iteration_runner_invocations}/"
                    f"{iteration_policy.max_total_runner_invocations})."
                )
                verification.feedback = (
                    f"{verification.feedback}\n{extra}".strip()
                    if verification.feedback
                    else extra
                )
            if iteration_budget_reason:
                verification.feedback = (
                    f"{verification.feedback}\n{iteration_budget_reason}".strip()
                    if verification.feedback
                    else iteration_budget_reason
                )
            if iteration_policy is not None and (runner_cap_exhausted or iteration_budget_reason):
                terminal_reason = (
                    "iteration_budget_exhausted"
                    if iteration_budget_reason
                    else "max_runner_invocations_exhausted"
                )
                gate_summary = (
                    iteration_budget_reason
                    if iteration_budget_reason
                    else str(verification.feedback or "").strip()
                )
                if not gate_summary:
                    gate_summary = "iteration_exhausted"
                subtask.iteration_terminal_reason = terminal_reason
                replan_request = await self._request_iteration_replan(
                    task=task,
                    subtask=subtask,
                    policy=iteration_policy,
                    terminal_reason=terminal_reason,
                    gate_summary=gate_summary,
                )
                if replan_request is not None:
                    async with self._state_lock:
                        subtask.status = SubtaskStatus.FAILED
                        subtask.summary = f"Iteration exhausted: {terminal_reason}"
                        subtask.active_issue = gate_summary
                        task.update_subtask(
                            subtask.id,
                            status=SubtaskStatus.FAILED,
                            summary=subtask.summary,
                            active_issue=subtask.active_issue,
                            iteration_terminal_reason=subtask.iteration_terminal_reason,
                            iteration_replan_count=subtask.iteration_replan_count,
                        )
                        self._state.save(task)
                    return replan_request

                if subtask.is_critical_path:
                    await self._abort_on_critical_path_failure(
                        task,
                        subtask,
                        VerificationResult(
                            tier=max(1, int(subtask.verification_tier or 1)),
                            passed=False,
                            feedback=gate_summary,
                            outcome="fail",
                            reason_code="iteration_exhausted",
                            severity_class="semantic",
                        ),
                    )
                    return None

                async with self._state_lock:
                    subtask.status = SubtaskStatus.FAILED
                    subtask.summary = f"Iteration exhausted: {terminal_reason}"
                    subtask.active_issue = gate_summary
                    task.update_subtask(
                        subtask.id,
                        status=SubtaskStatus.FAILED,
                        summary=subtask.summary,
                        active_issue=subtask.active_issue,
                        iteration_terminal_reason=subtask.iteration_terminal_reason,
                    )
                    task.add_error(
                        subtask.id,
                        f"Iteration exhausted ({terminal_reason}): {gate_summary}",
                    )
                    self._state.save(task)
                return None
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
                                verification=verification,
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
        self._annotate_subtask_phase_ids(task=task, plan=working)
        return working

    def _annotate_subtask_phase_ids(self, *, task: Task, plan: Plan) -> None:
        """Annotate each subtask with the closest matching process phase id."""
        process = self._process
        if process is None or not getattr(process, "phases", None):
            return

        phase_ids: list[str] = []
        phase_descriptions: dict[str, str] = {}
        phase_by_id: dict[str, object] = {}
        for phase in process.phases:
            phase_id = str(getattr(phase, "id", "")).strip()
            if not phase_id or phase_id in phase_descriptions:
                continue
            phase_ids.append(phase_id)
            phase_descriptions[phase_id] = str(
                getattr(phase, "description", ""),
            ).strip()
            phase_by_id[phase_id] = phase

        if not phase_ids:
            return
        phase_set = set(phase_ids)

        deliverables = process.get_deliverables()
        for phase_id in deliverables.keys():
            if phase_id in phase_set:
                continue
            phase_ids.append(phase_id)
            phase_set.add(phase_id)
            phase_descriptions.setdefault(phase_id, phase_id)

        prior_assignments: dict[str, str] = {}
        current_plan = getattr(task, "plan", None)
        if current_plan is not None:
            for prior in getattr(current_plan, "subtasks", []):
                prior_id = str(getattr(prior, "id", "")).strip()
                prior_phase_id = str(getattr(prior, "phase_id", "")).strip()
                if prior_id and prior_phase_id in phase_set:
                    prior_assignments[prior_id] = prior_phase_id

        for subtask in plan.subtasks:
            subtask_id = str(getattr(subtask, "id", "")).strip()
            assigned = ""
            if subtask_id in phase_set:
                assigned = subtask_id
            elif str(getattr(subtask, "phase_id", "")).strip() in phase_set:
                assigned = str(getattr(subtask, "phase_id", "")).strip()
            elif subtask_id and subtask_id in prior_assignments:
                assigned = prior_assignments[subtask_id]
            else:
                text = " ".join([
                    str(getattr(subtask, "description", "")).strip(),
                    str(getattr(subtask, "acceptance_criteria", "")).strip(),
                ]).strip()
                assigned = infer_phase_id_for_subtask(
                    subtask_id=subtask_id,
                    text=text,
                    phase_ids=phase_ids,
                    phase_descriptions=phase_descriptions,
                    phase_deliverables=deliverables,
                )
                if not assigned and len(phase_ids) == 1:
                    assigned = phase_ids[0]

            subtask.phase_id = assigned
            phase_obj = phase_by_id.get(assigned)
            if phase_obj is not None:
                policy = getattr(phase_obj, "iteration", None)
                if policy is not None and bool(getattr(policy, "enabled", False)):
                    subtask.iteration_max_attempts = int(
                        max(1, getattr(policy, "max_attempts", 1)),
                    )
                self._apply_subtask_policy_from_process_phase(
                    subtask=subtask,
                    phase=phase_obj,
                )
            else:
                self._ensure_subtask_validity_snapshot(subtask=subtask)

    @staticmethod
    def _hash_validity_contract(contract: dict[str, object]) -> str:
        serialized = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(
            serialized.encode("utf-8", errors="replace"),
        ).hexdigest()

    @staticmethod
    def _to_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        lowered = str(value or "").strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
        return bool(default)

    @staticmethod
    def _to_ratio(value: object, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(default)
        return max(0.0, min(1.0, parsed))

    @staticmethod
    def _to_non_negative_int(value: object, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        return max(0, parsed)

    @classmethod
    def _normalize_validity_contract(
        cls,
        contract: dict[str, object] | None,
    ) -> dict[str, object]:
        payload = dict(contract or {})
        claim_extraction_raw = payload.get("claim_extraction", {})
        if isinstance(claim_extraction_raw, bool):
            claim_extraction_raw = {"enabled": claim_extraction_raw}
        if not isinstance(claim_extraction_raw, dict):
            claim_extraction_raw = {}

        final_gate_raw = payload.get("final_gate", {})
        if not isinstance(final_gate_raw, dict):
            final_gate_raw = {}
        temporal_raw = final_gate_raw.get("temporal_consistency", {})
        if not isinstance(temporal_raw, dict):
            temporal_raw = {}

        critical_claim_types_raw = payload.get("critical_claim_types", [])
        if isinstance(critical_claim_types_raw, str):
            critical_claim_types_raw = [critical_claim_types_raw]
        if not isinstance(critical_claim_types_raw, list):
            critical_claim_types_raw = []

        prune_mode = str(payload.get("prune_mode", "drop") or "").strip().lower()
        if prune_mode not in {"drop", "rewrite_uncertainty"}:
            prune_mode = "drop"

        return {
            "enabled": cls._to_bool(payload.get("enabled", False), False),
            "claim_extraction": {
                "enabled": cls._to_bool(claim_extraction_raw.get("enabled", False), False),
            },
            "critical_claim_types": list(dict.fromkeys(
                str(item or "").strip().lower()
                for item in critical_claim_types_raw
                if str(item or "").strip()
            )),
            "min_supported_ratio": cls._to_ratio(payload.get("min_supported_ratio", 0.75), 0.75),
            "max_unverified_ratio": cls._to_ratio(payload.get("max_unverified_ratio", 0.25), 0.25),
            "max_contradicted_count": cls._to_non_negative_int(
                payload.get("max_contradicted_count", 0),
                0,
            ),
            "prune_mode": prune_mode,
            "require_fact_checker_for_synthesis": cls._to_bool(
                payload.get("require_fact_checker_for_synthesis", False),
                False,
            ),
            "final_gate": {
                "enforce_verified_context_only": cls._to_bool(
                    final_gate_raw.get("enforce_verified_context_only", True),
                    True,
                ),
                "synthesis_min_verification_tier": max(
                    1,
                    cls._to_non_negative_int(
                        final_gate_raw.get("synthesis_min_verification_tier", 2),
                        2,
                    ),
                ),
                "critical_claim_support_ratio": cls._to_ratio(
                    final_gate_raw.get("critical_claim_support_ratio", 1.0),
                    1.0,
                ),
                "temporal_consistency": {
                    "enabled": cls._to_bool(temporal_raw.get("enabled", False), False),
                    "require_as_of_alignment": cls._to_bool(
                        temporal_raw.get("require_as_of_alignment", False),
                        False,
                    ),
                    "enforce_cross_claim_date_conflict_check": cls._to_bool(
                        temporal_raw.get("enforce_cross_claim_date_conflict_check", False),
                        False,
                    ),
                    "max_source_age_days": cls._to_non_negative_int(
                        temporal_raw.get("max_source_age_days", 0),
                        0,
                    ),
                    "as_of": str(temporal_raw.get("as_of", "") or "").strip(),
                },
            },
        }

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

        failure_class = str(metadata.get("failure_class", "") or "").strip().lower()
        if failure_class:
            extracted["failure_class"] = failure_class

        missing_targets = cls._normalize_missing_targets(metadata.get("missing_targets"))
        if missing_targets:
            extracted["missing_targets"] = missing_targets

        placeholder_findings = metadata.get("placeholder_findings", [])
        if isinstance(placeholder_findings, list) and placeholder_findings:
            compacted: list[dict[str, object]] = []
            for raw in placeholder_findings:
                if not isinstance(raw, dict):
                    continue
                compacted.append({
                    "rule_name": str(raw.get("rule_name", "") or "").strip(),
                    "pattern": str(raw.get("pattern", "") or ""),
                    "source": str(raw.get("source", "") or ""),
                    "file_path": str(raw.get("file_path", "") or "").strip(),
                    "line": cls._to_int_or_none(raw.get("line")) or 0,
                    "column": cls._to_int_or_none(raw.get("column")) or 0,
                    "token": str(raw.get("token", "") or ""),
                    "context": str(raw.get("context", "") or ""),
                })
                if len(compacted) >= 120:
                    break
            if compacted:
                extracted["placeholder_findings"] = compacted
                extracted["placeholder_finding_count"] = len(compacted)
        else:
            finding_count = cls._to_int_or_none(metadata.get("placeholder_finding_count"))
            if finding_count is not None and finding_count > 0:
                extracted["placeholder_finding_count"] = int(finding_count)

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

    @staticmethod
    def _is_placeholder_unconfirmed_failure(
        *,
        verification: VerificationResult | None,
        placeholder_metadata: dict[str, object] | None = None,
    ) -> bool:
        reason_code = ""
        metadata: dict[str, object] = {}
        if verification is not None:
            reason_code = str(verification.reason_code or "").strip().lower()
            if isinstance(verification.metadata, dict):
                metadata = dict(verification.metadata)
        if isinstance(placeholder_metadata, dict):
            metadata = {**metadata, **placeholder_metadata}
        if reason_code in _PLACEHOLDER_UNCONFIRMED_REASON_CODES:
            return True
        failure_class = str(metadata.get("failure_class", "") or "").strip().lower()
        remediation_mode = str(metadata.get("remediation_mode", "") or "").strip().lower()
        findings = metadata.get("placeholder_findings")
        has_findings = isinstance(findings, list) and bool(findings)
        if not has_findings:
            try:
                has_findings = int(metadata.get("placeholder_finding_count", 0) or 0) > 0
            except (TypeError, ValueError):
                has_findings = False
        return (
            has_findings
            and (
                failure_class == "recoverable_placeholder"
                or remediation_mode == "confirm_or_prune"
            )
        )

    @classmethod
    def _normalize_placeholder_findings(
        cls,
        findings: object,
        *,
        max_items: int = 120,
    ) -> list[dict[str, object]]:
        if not isinstance(findings, list):
            return []
        normalized: list[dict[str, object]] = []
        seen: set[tuple[str, int, int, str, str]] = set()
        for raw in findings:
            if not isinstance(raw, dict):
                continue
            file_path = str(raw.get("file_path", "") or "").strip()
            token = str(raw.get("token", "") or "")
            rule_name = str(raw.get("rule_name", "") or "").strip()
            pattern = str(raw.get("pattern", "") or "")
            line = cls._to_int_or_none(raw.get("line")) or 0
            column = cls._to_int_or_none(raw.get("column")) or 0
            key = (file_path, max(0, line), max(0, column), token, pattern)
            if key in seen:
                continue
            seen.add(key)
            normalized.append({
                "rule_name": rule_name,
                "pattern": pattern,
                "source": str(raw.get("source", "") or ""),
                "file_path": file_path,
                "line": max(0, line),
                "column": max(0, column),
                "token": token,
                "context": str(raw.get("context", "") or ""),
            })
            if len(normalized) >= max_items:
                break
        return normalized

    @staticmethod
    def _normalize_workspace_relpath(workspace: Path, raw_path: str) -> str | None:
        text = str(raw_path or "").strip()
        if not text:
            return None
        root = workspace.resolve(strict=False)
        candidate = Path(text)
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
        else:
            resolved = (root / candidate).resolve(strict=False)
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            return None
        rel_text = rel.as_posix().strip()
        if not rel_text or rel_text == ".":
            return None
        return rel_text

    @staticmethod
    def _apply_deterministic_placeholder_prune_actions(
        *,
        workspace: Path,
        findings: list[dict[str, object]],
        replacement_token: str = "UNSUPPORTED_NO_EVIDENCE",
    ) -> dict[str, object]:
        root = workspace.resolve(strict=False)
        grouped: dict[str, list[dict[str, object]]] = {}
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            rel_path = Orchestrator._normalize_workspace_relpath(
                root,
                str(finding.get("file_path", "") or ""),
            )
            if not rel_path:
                continue
            grouped.setdefault(rel_path, []).append(finding)

        actions: list[dict[str, object]] = []
        files_modified: list[str] = []
        remaining_count = 0
        applied_count = 0

        for rel_path, rows in grouped.items():
            file_path = root / rel_path
            try:
                if not file_path.exists() or not file_path.is_file():
                    continue
                original = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = original.splitlines(keepends=True)
            changed = False

            for row in rows:
                token = str(row.get("token", "") or "")
                if not token:
                    continue
                line_no = 0
                try:
                    line_no = int(row.get("line", 0) or 0)
                except (TypeError, ValueError):
                    line_no = 0
                replaced = False
                if 1 <= line_no <= len(lines):
                    line_text = lines[line_no - 1]
                    if token in line_text:
                        lines[line_no - 1] = line_text.replace(
                            token,
                            replacement_token,
                            1,
                        )
                        replaced = True
                if not replaced:
                    current_text = "".join(lines)
                    if token in current_text:
                        lines = current_text.replace(token, replacement_token, 1).splitlines(
                            keepends=True,
                        )
                        replaced = True
                if replaced:
                    changed = True
                    applied_count += 1
                    actions.append({
                        "file_path": rel_path,
                        "line": max(0, line_no),
                        "token": token,
                        "action": "pruned_token",
                        "replacement": replacement_token,
                    })

            updated = "".join(lines)
            if changed and updated != original:
                try:
                    file_path.write_text(updated, encoding="utf-8")
                    files_modified.append(rel_path)
                except OSError:
                    continue

        for finding in findings:
            if not isinstance(finding, dict):
                continue
            token = str(finding.get("token", "") or "")
            if not token:
                continue
            rel_path = Orchestrator._normalize_workspace_relpath(
                root,
                str(finding.get("file_path", "") or ""),
            )
            if not rel_path:
                continue
            file_path = root / rel_path
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                remaining_count += 1
                continue
            if token in text:
                remaining_count += 1

        return {
            "applied_count": int(applied_count),
            "remaining_count": int(remaining_count),
            "files_modified": files_modified,
            "actions": actions,
        }

    @staticmethod
    def _summarize_placeholder_resolution_state(
        *,
        workspace: Path,
        findings: list[dict[str, object]],
        replacement_token: str = "UNSUPPORTED_NO_EVIDENCE",
    ) -> dict[str, int]:
        root = workspace.resolve(strict=False)
        tracked_findings = 0
        remaining_findings = 0
        replacement_token_occurrences = 0
        scanned_file_count = 0

        grouped_tokens: dict[str, set[str]] = {}
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            token = str(finding.get("token", "") or "")
            if not token:
                continue
            rel_path = Orchestrator._normalize_workspace_relpath(
                root,
                str(finding.get("file_path", "") or ""),
            )
            if not rel_path:
                continue
            tracked_findings += 1
            grouped_tokens.setdefault(rel_path, set()).add(token)

        for rel_path, tokens in grouped_tokens.items():
            file_path = root / rel_path
            try:
                if not file_path.exists() or not file_path.is_file():
                    remaining_findings += len(tokens)
                    continue
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                remaining_findings += len(tokens)
                continue
            scanned_file_count += 1
            replacement_token_occurrences += text.count(replacement_token)
            for token in tokens:
                if token in text:
                    remaining_findings += 1

        return {
            "tracked_findings": int(tracked_findings),
            "remaining_findings": int(remaining_findings),
            "replacement_token_occurrences": int(replacement_token_occurrences),
            "scanned_file_count": int(scanned_file_count),
        }

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
        metadata: dict[str, object] = {}
        if verification is not None and isinstance(verification.metadata, dict):
            metadata.update(verification.metadata)
        if isinstance(placeholder_metadata, dict):
            metadata.update(placeholder_metadata)
        findings = self._normalize_placeholder_findings(metadata.get("placeholder_findings"))
        reason_code = str(getattr(verification, "reason_code", "") or "").strip().lower()
        parent_mode = str(origin or "unconfirmed_data_retry").strip().lower()
        source_remediation_mode = str(metadata.get("remediation_mode", "") or "").strip().lower()
        failure_class = str(metadata.get("failure_class", "") or "").strip().lower()
        finding_count = len(findings)
        self._emit(PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED, task.id, {
            "subtask_id": subtask.id,
            "reason_code": reason_code,
            "mode": _PLACEHOLDER_PREPASS_MODE,
            "parent_mode": parent_mode,
            "attempt": int(attempt_number or 0),
            "max_attempts": int(max_attempts or 0),
            "remediation_mode": _PLACEHOLDER_PREPASS_MODE,
            "source_remediation_mode": source_remediation_mode,
            "failure_class": failure_class,
            "finding_count": finding_count,
        })

        workspace_text = str(getattr(task, "workspace", "") or "").strip()
        if not workspace_text:
            self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
                "subtask_id": subtask.id,
                "reason_code": reason_code,
                "mode": _PLACEHOLDER_PREPASS_MODE,
                "parent_mode": parent_mode,
                "attempt": int(attempt_number or 0),
                "max_attempts": int(max_attempts or 0),
                "stage": "deterministic_prepass",
                "outcome": "workspace_unavailable",
                "finding_count": finding_count,
            })
            return False, "deterministic placeholder remediation skipped: workspace unavailable", {}
        workspace = Path(workspace_text)
        if not workspace.exists() or not workspace.is_dir():
            self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
                "subtask_id": subtask.id,
                "reason_code": reason_code,
                "mode": _PLACEHOLDER_PREPASS_MODE,
                "parent_mode": parent_mode,
                "attempt": int(attempt_number or 0),
                "max_attempts": int(max_attempts or 0),
                "stage": "deterministic_prepass",
                "outcome": "workspace_path_missing",
                "finding_count": finding_count,
            })
            return (
                False,
                "deterministic placeholder remediation skipped: workspace path missing",
                {},
            )

        if not findings:
            self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
                "subtask_id": subtask.id,
                "reason_code": reason_code,
                "mode": _PLACEHOLDER_PREPASS_MODE,
                "parent_mode": parent_mode,
                "attempt": int(attempt_number or 0),
                "max_attempts": int(max_attempts or 0),
                "stage": "deterministic_prepass",
                "outcome": "no_structured_findings",
                "finding_count": finding_count,
            })
            return (
                False,
                "deterministic placeholder remediation skipped: no structured findings",
                {},
            )

        outcome = await run_blocking_io(
            self._apply_deterministic_placeholder_prune_actions,
            workspace=workspace,
            findings=findings,
        )
        if not isinstance(outcome, dict):
            self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
                "subtask_id": subtask.id,
                "reason_code": reason_code,
                "mode": _PLACEHOLDER_PREPASS_MODE,
                "parent_mode": parent_mode,
                "attempt": int(attempt_number or 0),
                "max_attempts": int(max_attempts or 0),
                "stage": "deterministic_prepass",
                "outcome": "invalid_outcome",
                "finding_count": finding_count,
            })
            return False, "deterministic placeholder remediation failed: invalid outcome", {}

        applied_count = int(outcome.get("applied_count", 0) or 0)
        remaining_count = int(outcome.get("remaining_count", 0) or 0)
        files_modified = outcome.get("files_modified", [])
        if not isinstance(files_modified, list):
            files_modified = []
        summary = (
            "Deterministic placeholder prepass "
            f"applied={applied_count} remaining={remaining_count} "
            f"files={len(files_modified)}."
        )
        details = {
            "mode": _PLACEHOLDER_PREPASS_MODE,
            "parent_mode": parent_mode,
            "applied_count": applied_count,
            "remaining_count": remaining_count,
            "files_modified": files_modified[:40],
            "actions": outcome.get("actions", [])[:120],
        }
        if applied_count > 0:
            self._emit(PLACEHOLDER_PRUNED, task.id, {
                "subtask_id": subtask.id,
                "reason_code": reason_code,
                "mode": _PLACEHOLDER_PREPASS_MODE,
                "parent_mode": parent_mode,
                "attempt": int(attempt_number or 0),
                "max_attempts": int(max_attempts or 0),
                "stage": "deterministic_prepass",
                "finding_count": finding_count,
                "applied_count": applied_count,
                "remaining_count": remaining_count,
                "files_modified_count": len(files_modified),
                "files_modified": files_modified[:40],
            })
        if remaining_count > 0 or applied_count <= 0:
            unresolved_outcome = (
                "placeholders_remaining"
                if remaining_count > 0
                else "no_mutations_applied"
            )
            self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
                "subtask_id": subtask.id,
                "reason_code": reason_code,
                "mode": _PLACEHOLDER_PREPASS_MODE,
                "parent_mode": parent_mode,
                "attempt": int(attempt_number or 0),
                "max_attempts": int(max_attempts or 0),
                "stage": "deterministic_prepass",
                "outcome": unresolved_outcome,
                "finding_count": finding_count,
                "applied_count": applied_count,
                "remaining_count": remaining_count,
                "files_modified_count": len(files_modified),
            })
        return (applied_count > 0 and remaining_count == 0), summary, details

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
        verification: VerificationResult | None = None,
        placeholder_metadata: dict[str, object] | None = None,
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
        placeholder_unconfirmed = self._is_placeholder_unconfirmed_failure(
            verification=verification,
            placeholder_metadata=placeholder_metadata,
        )
        deterministic_note = ""
        deterministic_details: dict[str, object] = {}
        placeholder_reason_code = (
            str(getattr(verification, "reason_code", "") or "").strip().lower()
        )
        placeholder_findings: list[dict[str, object]] = []
        if placeholder_unconfirmed:
            placeholder_context: dict[str, object] = {}
            if verification is not None and isinstance(verification.metadata, dict):
                placeholder_context.update(verification.metadata)
            if isinstance(placeholder_metadata, dict):
                placeholder_context.update(placeholder_metadata)
            placeholder_findings = self._normalize_placeholder_findings(
                placeholder_context.get("placeholder_findings"),
            )
        workspace_path: Path | None = None
        workspace_text = str(getattr(task, "workspace", "") or "").strip()
        if workspace_text:
            candidate_workspace = Path(workspace_text)
            if candidate_workspace.exists() and candidate_workspace.is_dir():
                workspace_path = candidate_workspace

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

            if attempt_number == 1 and placeholder_unconfirmed:
                (
                    _resolved_deterministically,
                    deterministic_note,
                    deterministic_details,
                ) = await self._run_deterministic_placeholder_prepass(
                    task=task,
                    subtask=subtask,
                    verification=verification,
                    placeholder_metadata=placeholder_metadata,
                    origin="confirm_or_prune_remediation",
                    attempt_number=attempt_number,
                    max_attempts=max_attempts,
                )
                if deterministic_note:
                    last_failure = deterministic_note

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
            if deterministic_note:
                remediation_context = (
                    f"{remediation_context}\n\n"
                    "DETERMINISTIC PLACEHOLDER REMEDIATION RESULT:\n"
                    f"- {deterministic_note}"
                ).strip()
            if deterministic_details:
                remediation_context = (
                    f"{remediation_context}\n"
                    "DETERMINISTIC PLACEHOLDER PREPASS DETAILS:\n"
                    f"{json.dumps(deterministic_details, ensure_ascii=True)}"
                ).strip()
            escalated_tier = self._retry.get_escalation_tier(
                attempt=len(attempts),
                original_tier=subtask.model_tier,
            )
            changelog = self._get_changelog(task)
            pre_resolution_state: dict[str, int] | None = None
            if placeholder_unconfirmed and placeholder_findings and workspace_path is not None:
                raw_pre_state = await run_blocking_io(
                    self._summarize_placeholder_resolution_state,
                    workspace=workspace_path,
                    findings=placeholder_findings,
                )
                if isinstance(raw_pre_state, dict):
                    pre_resolution_state = {
                        "tracked_findings": int(raw_pre_state.get("tracked_findings", 0) or 0),
                        "remaining_findings": int(raw_pre_state.get("remaining_findings", 0) or 0),
                        "replacement_token_occurrences": int(
                            raw_pre_state.get("replacement_token_occurrences", 0) or 0,
                        ),
                    }
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
                tool_calls=remediation_result.tool_calls,
            )

            if remediation_verification.passed:
                await self._handle_success(
                    task,
                    subtask,
                    remediation_result,
                    remediation_verification,
                )
                if (
                    placeholder_unconfirmed
                    and placeholder_findings
                    and workspace_path is not None
                    and pre_resolution_state is not None
                ):
                    raw_post_state = await run_blocking_io(
                        self._summarize_placeholder_resolution_state,
                        workspace=workspace_path,
                        findings=placeholder_findings,
                    )
                    if isinstance(raw_post_state, dict):
                        pre_remaining = int(
                            pre_resolution_state.get("remaining_findings", 0) or 0,
                        )
                        post_remaining = int(raw_post_state.get("remaining_findings", 0) or 0)
                        pre_replacement_count = int(
                            pre_resolution_state.get("replacement_token_occurrences", 0) or 0,
                        )
                        post_replacement_count = int(
                            raw_post_state.get("replacement_token_occurrences", 0) or 0,
                        )
                        resolved_count = max(0, pre_remaining - post_remaining)
                        replacement_delta = max(0, post_replacement_count - pre_replacement_count)
                        if post_remaining > 0:
                            self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
                                "subtask_id": subtask.id,
                                "reason_code": placeholder_reason_code,
                                "mode": "confirm_or_prune_remediation",
                                "attempt": attempt_number,
                                "max_attempts": max_attempts,
                                "stage": "post_model_success",
                                "outcome": "placeholders_remaining",
                                "finding_count": len(placeholder_findings),
                                "remaining_count": post_remaining,
                            })
                        elif resolved_count > 0:
                            if replacement_delta > 0:
                                self._emit(PLACEHOLDER_PRUNED, task.id, {
                                    "subtask_id": subtask.id,
                                    "reason_code": placeholder_reason_code,
                                    "mode": "confirm_or_prune_remediation",
                                    "attempt": attempt_number,
                                    "max_attempts": max_attempts,
                                    "stage": "model_retry",
                                    "finding_count": len(placeholder_findings),
                                    "resolved_count": resolved_count,
                                    "replacement_delta": replacement_delta,
                                })
                            else:
                                self._emit(PLACEHOLDER_FILLED, task.id, {
                                    "subtask_id": subtask.id,
                                    "reason_code": placeholder_reason_code,
                                    "mode": "confirm_or_prune_remediation",
                                    "attempt": attempt_number,
                                    "max_attempts": max_attempts,
                                    "stage": "model_retry",
                                    "finding_count": len(placeholder_findings),
                                    "filled_count": resolved_count,
                                })
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

        if placeholder_unconfirmed:
            self._emit(PLACEHOLDER_REMEDIATION_UNRESOLVED, task.id, {
                "subtask_id": subtask.id,
                "reason_code": placeholder_reason_code,
                "mode": "confirm_or_prune_remediation",
                "attempt": max_attempts,
                "max_attempts": max_attempts,
                "stage": "terminal",
                "outcome": "max_attempts_exhausted",
                "finding_count": len(placeholder_findings),
                "error": last_failure,
            })
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
        subtask: Subtask,
        subtask_id: str,
        attempt_record: AttemptRecord,
        verification: VerificationResult,
    ) -> None:
        if not bool(getattr(self._config.execution, "enable_sqlite_remediation_queue", False)):
            return
        verification_metadata = (
            dict(verification.metadata)
            if isinstance(verification.metadata, dict)
            else {}
        )
        attempt_metadata: dict[str, object] = {
            "model_tier": int(getattr(subtask, "model_tier", 1) or 1),
            "verification_tier": int(getattr(subtask, "verification_tier", 1) or 1),
            "acceptance_criteria": str(getattr(subtask, "acceptance_criteria", "") or ""),
            "validity_contract_hash": str(
                getattr(subtask, "validity_contract_hash", "") or "",
            ),
            "validity_contract_snapshot": (
                dict(getattr(subtask, "validity_contract_snapshot", {}))
                if isinstance(getattr(subtask, "validity_contract_snapshot", {}), dict)
                else {}
            ),
        }
        if "claim_status_counts" in verification_metadata:
            attempt_metadata["claim_status_counts"] = verification_metadata.get(
                "claim_status_counts",
            )
        if "claim_reason_codes" in verification_metadata:
            attempt_metadata["claim_reason_codes"] = verification_metadata.get("claim_reason_codes")
        if "supported_ratio" in verification_metadata:
            attempt_metadata["supported_ratio"] = verification_metadata.get("supported_ratio")
        if "unverified_ratio" in verification_metadata:
            attempt_metadata["unverified_ratio"] = verification_metadata.get("unverified_ratio")
        if "critical_support_ratio" in verification_metadata:
            attempt_metadata["critical_support_ratio"] = verification_metadata.get(
                "critical_support_ratio",
            )
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
                metadata=attempt_metadata,
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
            placeholder_metadata=item if isinstance(item, dict) else None,
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
        """Process a successful subtask: update state, check approval."""
        self._persist_subtask_evidence(
            task.id,
            subtask.id,
            result.evidence_records,
            tool_calls=result.tool_calls,
        )
        summary = result.summary

        # Update state
        async with self._state_lock:
            self._record_artifact_seals(
                task=task,
                subtask_id=subtask.id,
                tool_calls=result.tool_calls,
            )
            self._record_subtask_validity_metrics(
                task=task,
                subtask=subtask,
                verification=verification,
            )
            if subtask.is_synthesis:
                summary = self._append_synthesis_provenance_footer(
                    task=task,
                    summary=summary,
                )
                result.summary = summary
            subtask.status = SubtaskStatus.COMPLETED
            subtask.summary = summary
            subtask.active_issue = ""
            subtask.iteration_last_gate_summary = ""
            task.update_subtask(
                subtask.id,
                status=SubtaskStatus.COMPLETED,
                summary=summary,
                active_issue="",
                iteration_last_gate_summary="",
                iteration_terminal_reason=subtask.iteration_terminal_reason,
            )

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
                phase_id=s.get("phase_id", ""),
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
                iteration_max_attempts = 0
                if (
                    getattr(phase, "iteration", None) is not None
                    and bool(getattr(phase.iteration, "enabled", False))
                ):
                    iteration_max_attempts = int(
                        max(1, getattr(phase.iteration, "max_attempts", 1)),
                    )
                strict_subtasks.append(
                    Subtask(
                        id=phase.id,
                        description=phase.description,
                        depends_on=list(phase.depends_on),
                        phase_id=phase.id,
                        model_tier=phase.model_tier,
                        verification_tier=phase.verification_tier,
                        is_critical_path=phase.is_critical_path,
                        is_synthesis=phase.is_synthesis,
                        acceptance_criteria=phase.acceptance_criteria,
                        max_retries=self._config.execution.max_subtask_retries,
                        iteration_max_attempts=iteration_max_attempts,
                    )
                )
                continue

            existing.description = phase.description or existing.description
            existing.depends_on = list(phase.depends_on)
            existing.phase_id = phase.id
            existing.model_tier = phase.model_tier
            existing.verification_tier = phase.verification_tier
            existing.is_critical_path = phase.is_critical_path
            existing.is_synthesis = phase.is_synthesis
            if phase.acceptance_criteria:
                existing.acceptance_criteria = phase.acceptance_criteria
            if (
                getattr(phase, "iteration", None) is not None
                and bool(getattr(phase.iteration, "enabled", False))
            ):
                existing.iteration_max_attempts = int(
                    max(1, getattr(phase.iteration, "max_attempts", 1)),
                )
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
        *,
        tool_calls: list[ToolCallRecord] | None = None,
    ) -> None:
        """Persist newly captured evidence records."""
        scoped: list[dict] = []
        if evidence_records:
            for item in evidence_records:
                if not isinstance(item, dict):
                    continue
                normalized = dict(item)
                normalized["subtask_id"] = subtask_id
                normalized.setdefault("task_id", task_id)
                scoped.append(normalized)
        existing_ids: set[str] = {
            str(item.get("evidence_id", "") or "").strip()
            for item in scoped
            if isinstance(item, dict) and str(item.get("evidence_id", "") or "").strip()
        }
        provenance_records = self._artifact_provenance_evidence(
            task_id=task_id,
            subtask_id=subtask_id,
            tool_calls=tool_calls,
            existing_ids=existing_ids,
        )
        if provenance_records:
            scoped = merge_evidence_records(scoped, provenance_records)
        if not scoped:
            return
        try:
            self._state.append_evidence_records(task_id, scoped)
        except Exception as e:
            logger.warning("Failed persisting evidence ledger for %s: %s", task_id, e)

    @staticmethod
    def _artifact_content_for_call(
        tool_name: str,
        args: dict[str, object],
        result_data: dict[str, object],
    ) -> str:
        if tool_name == "write_file":
            return str(args.get("content", "") or "")
        if tool_name == "document_write":
            parts: list[str] = []
            title = str(args.get("title", "") or "").strip()
            if title:
                parts.append(title)
            content = str(args.get("content", "") or "")
            if content:
                parts.append(content)
            sections = args.get("sections", [])
            if isinstance(sections, list):
                for section in sections[:8]:
                    if not isinstance(section, dict):
                        continue
                    heading = str(section.get("heading", "") or "").strip()
                    body = str(section.get("body", "") or "")
                    if heading:
                        parts.append(heading)
                    if body:
                        parts.append(body)
            if parts:
                return "\n\n".join(parts)
            return str(result_data.get("content", "") or "")
        return ""

    def _artifact_seal_registry(self, task: Task) -> dict[str, dict[str, object]]:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        registry = metadata.get("artifact_seals")
        if not isinstance(registry, dict):
            registry = {}
            metadata["artifact_seals"] = registry
        task.metadata = metadata
        return registry

    def _record_artifact_seals(
        self,
        *,
        task: Task,
        subtask_id: str,
        tool_calls: list[ToolCallRecord] | None,
    ) -> int:
        if not tool_calls:
            return 0
        workspace_text = str(task.workspace or "").strip()
        if not workspace_text:
            return 0
        try:
            workspace = Path(workspace_text).expanduser().resolve()
        except Exception:
            return 0
        if not workspace.exists():
            return 0

        seals = self._artifact_seal_registry(task)
        updated = 0
        for call in tool_calls:
            tool_name = str(getattr(call, "tool", "") or "").strip().lower()
            if tool_name not in {"write_file", "document_write"}:
                continue
            result = getattr(call, "result", None)
            if result is None or not bool(getattr(result, "success", False)):
                continue
            args = getattr(call, "args", {})
            if not isinstance(args, dict):
                args = {}
            result_data = getattr(result, "data", {})
            if not isinstance(result_data, dict):
                result_data = {}
            raw_path = str(
                args.get("path")
                or args.get("file_path")
                or result_data.get("path")
                or "",
            ).strip()
            if not raw_path:
                continue

            candidate = Path(raw_path)
            if candidate.is_absolute():
                try:
                    resolved = candidate.expanduser().resolve()
                    relpath = resolved.relative_to(workspace).as_posix()
                except Exception:
                    continue
            else:
                relpath = candidate.as_posix()
                try:
                    resolved = (workspace / relpath).resolve()
                    resolved.relative_to(workspace)
                except Exception:
                    continue

            sha256 = ""
            size_bytes = 0
            if resolved.exists() and resolved.is_file():
                try:
                    payload = resolved.read_bytes()
                except Exception:
                    payload = b""
                if payload:
                    size_bytes = len(payload)
                    sha256 = hashlib.sha256(payload).hexdigest()
            if not sha256:
                content = self._artifact_content_for_call(tool_name, args, result_data)
                if content:
                    payload = content.encode("utf-8", errors="replace")
                    size_bytes = len(payload)
                    sha256 = hashlib.sha256(payload).hexdigest()
            if not sha256:
                continue

            seals[relpath] = {
                "path": relpath,
                "sha256": sha256,
                "size_bytes": int(size_bytes),
                "tool": tool_name,
                "tool_call_id": str(getattr(call, "call_id", "") or ""),
                "subtask_id": subtask_id,
                "run_id": self._task_run_id(task),
                "sealed_at": datetime.now().isoformat(),
            }
            updated += 1

        if updated > 0:
            task.metadata["artifact_seals"] = seals
        return updated

    def _validate_artifact_seals(
        self,
        *,
        task: Task,
    ) -> tuple[bool, list[dict[str, object]], int]:
        seals = self._artifact_seal_registry(task)
        if not seals:
            self._backfill_artifact_seals_from_evidence(task)
            seals = self._artifact_seal_registry(task)
        if not seals:
            return True, [], 0

        workspace_text = str(task.workspace or "").strip()
        if not workspace_text:
            return True, [], 0
        try:
            workspace = Path(workspace_text).expanduser().resolve()
        except Exception:
            return True, [], 0
        if not workspace.exists():
            return True, [], 0

        mismatches: list[dict[str, object]] = []
        validated = 0
        for relpath, seal in seals.items():
            if not isinstance(seal, dict):
                continue
            expected = str(seal.get("sha256", "") or "").strip()
            if not expected:
                continue
            try:
                artifact_path = (workspace / str(relpath)).resolve()
                artifact_path.relative_to(workspace)
            except Exception:
                mismatches.append({
                    "path": str(relpath),
                    "reason": "path_outside_workspace",
                })
                continue
            validated += 1
            if not artifact_path.exists() or not artifact_path.is_file():
                mismatches.append({
                    "path": str(relpath),
                    "reason": "artifact_missing",
                })
                continue
            try:
                observed = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            except Exception:
                mismatches.append({
                    "path": str(relpath),
                    "reason": "artifact_unreadable",
                })
                continue
            if observed != expected:
                mismatches.append({
                    "path": str(relpath),
                    "reason": "artifact_seal_mismatch",
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                })
        return len(mismatches) == 0, mismatches, validated

    def _backfill_artifact_seals_from_evidence(self, task: Task) -> int:
        seals = self._artifact_seal_registry(task)
        if seals:
            return 0
        try:
            records = self._state.load_evidence_records(task.id)
        except Exception:
            return 0
        latest_by_path: dict[str, dict[str, object]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            tool = str(record.get("tool", "") or "").strip().lower()
            if tool not in {"write_file", "document_write"}:
                continue
            relpath = str(record.get("artifact_workspace_relpath", "") or "").strip()
            sha256 = str(record.get("artifact_sha256", "") or "").strip()
            if not relpath or not sha256:
                continue
            current = latest_by_path.get(relpath)
            if current is None:
                latest_by_path[relpath] = record
                continue
            current_ts = str(current.get("created_at", "") or "")
            record_ts = str(record.get("created_at", "") or "")
            if record_ts >= current_ts:
                latest_by_path[relpath] = record

        if not latest_by_path:
            return 0

        for relpath, record in latest_by_path.items():
            seals[relpath] = {
                "path": relpath,
                "sha256": str(record.get("artifact_sha256", "") or ""),
                "size_bytes": int(record.get("artifact_size_bytes", 0) or 0),
                "tool": str(record.get("tool", "") or ""),
                "tool_call_id": str(record.get("tool_call_id", "") or ""),
                "subtask_id": str(record.get("subtask_id", "") or ""),
                "run_id": self._task_run_id(task),
                "sealed_at": str(record.get("created_at", "") or ""),
                "backfilled_from_evidence": True,
            }
        task.metadata["artifact_seals"] = seals
        return len(latest_by_path)

    def _validity_scorecard_state(self, task: Task) -> dict[str, object]:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        scorecard = metadata.get("validity_scorecard")
        if not isinstance(scorecard, dict):
            scorecard = {}
        per_subtask = scorecard.get("subtask_metrics")
        if not isinstance(per_subtask, dict):
            per_subtask = {}
        scorecard["subtask_metrics"] = per_subtask
        metadata["validity_scorecard"] = scorecard
        task.metadata = metadata
        return scorecard

    def _record_subtask_validity_metrics(
        self,
        *,
        task: Task,
        subtask: Subtask,
        verification: VerificationResult,
    ) -> None:
        if verification is None:
            return
        scorecard = self._validity_scorecard_state(task)
        per_subtask = scorecard.get("subtask_metrics", {})
        if not isinstance(per_subtask, dict):
            per_subtask = {}
            scorecard["subtask_metrics"] = per_subtask

        metadata = verification.metadata if isinstance(verification.metadata, dict) else {}
        counts = metadata.get("claim_status_counts")
        if not isinstance(counts, dict):
            counts = self._claim_counts(self._claims_from_verification(verification))
        counts = {
            "extracted": int(counts.get("extracted", 0) or 0),
            "supported": int(counts.get("supported", 0) or 0),
            "contradicted": int(counts.get("contradicted", 0) or 0),
            "insufficient_evidence": int(counts.get("insufficient_evidence", 0) or 0),
            "stale": int(counts.get("stale", 0) or 0),
            "pruned": int(counts.get("pruned", 0) or 0),
            "unresolved": int(
                counts.get(
                    "unresolved",
                    int(counts.get("contradicted", 0) or 0)
                    + int(counts.get("insufficient_evidence", 0) or 0)
                    + int(counts.get("stale", 0) or 0),
                ) or 0,
            ),
            "critical_total": int(counts.get("critical_total", 0) or 0),
            "critical_supported": int(counts.get("critical_supported", 0) or 0),
            "critical_contradicted": int(counts.get("critical_contradicted", 0) or 0),
        }
        ratios = self._claim_ratios(counts)
        reason_codes = metadata.get("claim_reason_codes")
        if not isinstance(reason_codes, list):
            reason_codes = []
        normalized_reason_codes = sorted({
            str(item or "").strip().lower()
            for item in reason_codes
            if str(item or "").strip()
        })
        per_subtask[subtask.id] = {
            "subtask_id": subtask.id,
            "phase_id": str(subtask.phase_id or ""),
            "is_synthesis": bool(subtask.is_synthesis),
            "verification_outcome": str(verification.outcome or ""),
            "reason_code": str(verification.reason_code or "").strip().lower(),
            "counts": counts,
            "ratios": {
                "supported_ratio": float(ratios.get("supported_ratio", 0.0)),
                "unverified_ratio": float(ratios.get("unverified_ratio", 0.0)),
                "critical_support_ratio": float(ratios.get("critical_support_ratio", 0.0)),
            },
            "reason_codes": normalized_reason_codes,
            "updated_at": datetime.now().isoformat(),
        }
        scorecard["run"] = self._build_run_validity_scorecard(task)

    def _scorecard_source_window(self, task: Task) -> dict[str, str]:
        try:
            records = self._state.load_evidence_records(task.id)
        except Exception:
            return {"min": "", "max": ""}
        timestamps = sorted({
            str(record.get("created_at", "") or "").strip()
            for record in records
            if isinstance(record, dict) and str(record.get("created_at", "") or "").strip()
        })
        if not timestamps:
            return {"min": "", "max": ""}
        return {"min": timestamps[0], "max": timestamps[-1]}

    def _build_run_validity_scorecard(self, task: Task) -> dict[str, object]:
        scorecard = self._validity_scorecard_state(task)
        per_subtask = scorecard.get("subtask_metrics", {})
        if not isinstance(per_subtask, dict):
            per_subtask = {}
        aggregate = {
            "extracted": 0,
            "supported": 0,
            "contradicted": 0,
            "insufficient_evidence": 0,
            "stale": 0,
            "pruned": 0,
            "unresolved": 0,
            "critical_total": 0,
            "critical_supported": 0,
            "critical_contradicted": 0,
        }
        reason_codes: set[str] = set()
        for entry in per_subtask.values():
            if not isinstance(entry, dict):
                continue
            counts = entry.get("counts", {})
            if not isinstance(counts, dict):
                continue
            for key in aggregate:
                aggregate[key] += int(counts.get(key, 0) or 0)
            raw_codes = entry.get("reason_codes", [])
            if isinstance(raw_codes, list):
                reason_codes.update(
                    str(item or "").strip().lower()
                    for item in raw_codes
                    if str(item or "").strip()
                )
            entry_reason = str(entry.get("reason_code", "") or "").strip().lower()
            if entry_reason:
                reason_codes.add(entry_reason)

        ratios = self._claim_ratios(aggregate)
        extracted = max(0, int(aggregate.get("extracted", 0) or 0))
        contradicted = max(0, int(aggregate.get("contradicted", 0) or 0))
        contradicted_ratio = (float(contradicted) / float(extracted)) if extracted > 0 else 0.0
        trust_score = max(
            0.0,
            min(
                1.0,
                float(ratios.get("supported_ratio", 0.0))
                - (0.6 * float(ratios.get("unverified_ratio", 0.0)))
                - (0.9 * contradicted_ratio),
            ),
        )
        source_window = self._scorecard_source_window(task)
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "source_time_window": source_window,
            "counts": aggregate,
            "supported_ratio": round(float(ratios.get("supported_ratio", 0.0)), 4),
            "unverified_ratio": round(float(ratios.get("unverified_ratio", 0.0)), 4),
            "critical_support_ratio": round(
                float(ratios.get("critical_support_ratio", 0.0)),
                4,
            ),
            "trust_score": round(trust_score, 4),
            "reason_codes": sorted(reason_codes),
            "verification_report_path": self._VALIDITY_SCORECARD_JSON_NAME,
        }

    def _refresh_run_validity_scorecard(self, task: Task) -> dict[str, object]:
        scorecard = self._validity_scorecard_state(task)
        run_summary = self._build_run_validity_scorecard(task)
        scorecard["run"] = run_summary
        task.metadata["validity_scorecard"] = scorecard
        return run_summary

    def _export_validity_scorecard_json(self, task: Task) -> None:
        workspace_text = str(task.workspace or "").strip()
        if not workspace_text:
            return
        workspace = Path(workspace_text).expanduser()
        if not workspace.exists() or not workspace.is_dir():
            return
        run_summary = self._refresh_run_validity_scorecard(task)
        output = workspace / self._VALIDITY_SCORECARD_JSON_NAME
        payload = {
            "task_id": task.id,
            "run_id": self._task_run_id(task),
            "status": str(
                task.status.value if isinstance(task.status, TaskStatus) else task.status,
            ),
            "summary": run_summary,
        }
        try:
            output.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed exporting validity scorecard for %s: %s", task.id, e)

    def _emit_run_validity_scorecard(self, task: Task) -> None:
        run_summary = self._refresh_run_validity_scorecard(task)
        self._emit(RUN_VALIDITY_SCORECARD, task.id, {
            "run_id": self._task_run_id(task),
            **run_summary,
        })

    def _append_synthesis_provenance_footer(
        self,
        *,
        task: Task,
        summary: str,
    ) -> str:
        base = str(summary or "").strip()
        marker = "VALIDITY_PROVENANCE_FOOTER:"
        if marker in base:
            return base
        run_summary = self._refresh_run_validity_scorecard(task)
        source_window = run_summary.get("source_time_window", {})
        if not isinstance(source_window, dict):
            source_window = {}
        footer_lines = [
            marker,
            f"analysis_timestamp={run_summary.get('analysis_timestamp', '')}",
            f"source_time_window={source_window.get('min', '')}..{source_window.get('max', '')}",
            f"supported_ratio={run_summary.get('supported_ratio', 0.0)}",
            f"critical_support_ratio={run_summary.get('critical_support_ratio', 0.0)}",
            f"trust_score={run_summary.get('trust_score', 0.0)}",
            "verification_report="
            + str(
                run_summary.get(
                    "verification_report_path",
                    self._VALIDITY_SCORECARD_JSON_NAME,
                ),
            ),
        ]
        footer = "\n".join(footer_lines).strip()
        if not base:
            return footer
        return f"{base}\n\n{footer}"

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
        run_validity_summary = self._refresh_run_validity_scorecard(task)
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
            cancel_reason = ""
            if isinstance(task.metadata, dict):
                cancel_reason = str(task.metadata.get("cancel_reason", "") or "").strip()
            self._emit(TASK_CANCELLED, task.id, {
                "completed": completed,
                "total": total,
                "reason": cancel_reason or "cancel_requested",
                "outcome": "cancelled",
            })
        elif all_done:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            self._emit(TASK_COMPLETED, task.id, {
                "completed": completed,
                "total": total,
                "reason": "all_subtasks_completed",
                "outcome": "completed",
                "validity_summary": run_validity_summary,
            })
        else:
            task.status = TaskStatus.FAILED
            failed = [s for s in task.plan.subtasks if s.status == SubtaskStatus.FAILED]
            failure_reason = "subtask_failure"
            if blocking_remediation_failures:
                task.add_error(
                    "remediation",
                    "Blocking remediation unresolved for: "
                    + ", ".join(blocking_remediation_failures),
                )
                failure_reason = "blocking_remediation_unresolved"
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
                failure_reason = "blocked_pending_subtasks"
            self._emit(TASK_FAILED, task.id, {
                "completed": completed,
                "total": total,
                "failed_subtasks": [s.id for s in failed],
                "reason": failure_reason,
                "outcome": "failed",
                "blocking_remediation_failures": blocking_remediation_failures,
                "blocked_subtasks": blocked_subtasks,
                "validity_summary": run_validity_summary,
            })

        self._emit_run_validity_scorecard(task)
        self._emit_telemetry_run_summary(task)
        self._export_validity_scorecard_json(task)
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
        payload.setdefault("source_component", "orchestrator")
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

    def _task_event_counts(self, task_id: str) -> dict[str, int]:
        history_limit = max(1000, int(getattr(self._events, "_max_history", 1000) or 1000))
        counters: dict[str, int] = {}
        for event in self._events.recent_events(limit=history_limit):
            if str(getattr(event, "task_id", "") or "") != task_id:
                continue
            event_type = str(getattr(event, "event_type", "") or "").strip()
            if not event_type:
                continue
            counters[event_type] = int(counters.get(event_type, 0)) + 1
        return counters

    def _verification_reason_counts(self, task_id: str) -> dict[str, int]:
        history_limit = max(1000, int(getattr(self._events, "_max_history", 1000) or 1000))
        reasons: dict[str, int] = {}
        for event in self._events.recent_events(limit=history_limit):
            if str(getattr(event, "task_id", "") or "") != task_id:
                continue
            if str(getattr(event, "event_type", "") or "").strip() != VERIFICATION_OUTCOME:
                continue
            payload = getattr(event, "data", None)
            if not isinstance(payload, dict):
                continue
            reason = str(payload.get("reason_code", "") or "").strip().lower()
            if not reason:
                reason = "unspecified"
            reasons[reason] = int(reasons.get(reason, 0)) + 1
        return reasons

    def _emit_telemetry_run_summary(self, task: Task) -> None:
        run_key = self._task_run_id(task) or task.id
        if run_key in self._emitted_telemetry_summary_runs:
            return
        rollup = getattr(self, "_telemetry_rollup", None)
        if not isinstance(rollup, dict):
            rollup = self._new_telemetry_rollup()
        validity_summary = {}
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if isinstance(metadata, dict):
            scorecard = metadata.get("validity_scorecard", {})
            if isinstance(scorecard, dict):
                run_summary = scorecard.get("run", {})
                if isinstance(run_summary, dict):
                    validity_summary = run_summary
        event_counts = self._task_event_counts(task.id)
        verification_reason_counts = self._verification_reason_counts(task.id)
        verification_lifecycle_counts = {
            "started": int(event_counts.get(VERIFICATION_STARTED, 0)),
            "passed": int(event_counts.get(VERIFICATION_PASSED, 0)),
            "failed": int(event_counts.get(VERIFICATION_FAILED, 0)),
            "outcome": int(event_counts.get(VERIFICATION_OUTCOME, 0)),
        }
        remediation_lifecycle_counts = {
            "queued": int(event_counts.get(REMEDIATION_QUEUED, 0)),
            "started": int(event_counts.get(REMEDIATION_STARTED, 0)),
            "attempt": int(event_counts.get(REMEDIATION_ATTEMPT, 0)),
            "resolved": int(event_counts.get(REMEDIATION_RESOLVED, 0)),
            "failed": int(event_counts.get(REMEDIATION_FAILED, 0)),
            "expired": int(event_counts.get(REMEDIATION_EXPIRED, 0)),
            "terminal": int(event_counts.get(REMEDIATION_TERMINAL, 0)),
        }
        human_loop_counts = {
            "approval_requested": int(event_counts.get(APPROVAL_REQUESTED, 0)),
            "approval_received": int(event_counts.get(APPROVAL_RECEIVED, 0)),
            "ask_user_requested": int(event_counts.get(ASK_USER_REQUESTED, 0)),
            "ask_user_answered": int(event_counts.get(ASK_USER_ANSWERED, 0)),
            "ask_user_timeout": int(event_counts.get(ASK_USER_TIMEOUT, 0)),
            "ask_user_cancelled": int(event_counts.get(ASK_USER_CANCELLED, 0)),
            "steer_instruction": int(event_counts.get(STEER_INSTRUCTION, 0)),
        }
        control_plane_counts = {
            "paused": int(event_counts.get(TASK_PAUSED, 0)),
            "resumed": int(event_counts.get(TASK_RESUMED, 0)),
            "injected": int(event_counts.get(TASK_INJECTED, 0)),
            "cancel_requested": int(event_counts.get(TASK_CANCEL_REQUESTED, 0)),
            "cancel_ack": int(event_counts.get(TASK_CANCEL_ACK, 0)),
            "cancel_timeout": int(event_counts.get(TASK_CANCEL_TIMEOUT, 0)),
        }
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
            "verification_lifecycle_counts": verification_lifecycle_counts,
            "verification_reason_counts": verification_reason_counts,
            "remediation_lifecycle_counts": remediation_lifecycle_counts,
            "human_loop_counts": human_loop_counts,
            "control_plane_counts": control_plane_counts,
            "blocked_indicator": bool(event_counts.get(SUBTASK_BLOCKED, 0) > 0),
            "degraded_indicator": bool(event_counts.get(TASK_PLAN_DEGRADED, 0) > 0),
            "replanned_count": int(event_counts.get(TASK_REPLANNING, 0)),
            "stalled_count": int(event_counts.get(TASK_STALLED, 0)),
            "budget_snapshot": self._run_budget.snapshot(),
            "validity_summary": validity_summary,
        })
        self._emitted_telemetry_summary_runs.add(run_key)

    def cancel_task(self, task: Task) -> None:
        """Mark a task for cancellation."""
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["cancel_reason"] = "cancel_requested"
        task.status = TaskStatus.CANCELLED
        self._state.save(task)
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
        self._state.save(task)
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
        self._state.save(task)
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

    async def answer_question(
        self,
        task_id: str,
        question_id: str,
        answer_payload: dict,
    ) -> dict | None:
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
