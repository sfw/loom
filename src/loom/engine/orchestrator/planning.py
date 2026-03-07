"""Planning helpers for orchestrator extraction."""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from loom.auth.runtime import AuthResolutionError, build_run_auth_context
from loom.engine.scheduler import Scheduler
from loom.events.types import (
    MODEL_INVOCATION,
    TASK_PLAN_NORMALIZED,
    TASK_PLAN_READY,
    TASK_REPLAN_REJECTED,
    TASK_REPLANNING,
    TASK_STALLED_RECOVERY_ATTEMPTED,
)
from loom.models.base import ModelResponse
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.processes.phase_alignment import infer_phase_id_for_subtask
from loom.state.task_state import Plan, Subtask, SubtaskStatus, Task
from loom.utils.concurrency import run_blocking_io

logger = logging.getLogger(__name__)


def phase_mode(orchestrator) -> str:
    """Resolve process phase mode with bounded valid values."""
    process = orchestrator._process
    if process is None:
        return "guided"
    value = str(getattr(process, "phase_mode", "guided") or "").strip().lower()
    if value in {"strict", "guided", "suggestive"}:
        return value
    return "guided"


def topology_retry_attempts(orchestrator) -> int:
    """Bounded retry budget for topology-invalid planner outputs."""
    return 2 if phase_mode(orchestrator) == "strict" else 1


def planner_degraded_mode(orchestrator) -> str:
    """Resolve planner degraded-mode behavior with bounded values."""
    raw = str(
        getattr(orchestrator._config.execution, "planner_degraded_mode", "allow"),
    ).strip().lower()
    if raw in {"allow", "require_approval", "deny"}:
        return raw
    return "allow"

async def plan_task_with_validation(orchestrator, task: Task) -> Plan:
    """Plan task with bounded retries when topology validation fails."""
    max_attempts = orchestrator._topology_retry_attempts()
    planner_feedback = ""
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        plan = await orchestrator._plan_task(task, planner_feedback=planner_feedback)
        try:
            return orchestrator._prepare_plan_for_execution(
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
            orchestrator._state.save(task)
            if attempt >= max_attempts:
                break
            planner_feedback = (
                "Previous planner output was rejected for invalid topology.\n"
                f"Validation error: {last_error}\n"
                "Return corrected JSON that satisfies all dependency and "
                "synthesis-topology constraints."
            )

    raise ValueError(last_error or "Planner output failed topology validation.")

async def plan_task(
    orchestrator,
    task: Task,
    *,
    planner_feedback: str = "",
) -> Plan:
    """Invoke the planner model to decompose the task into subtasks."""
    workspace_listing = ""
    code_analysis = ""
    workspace_analysis = ""
    read_roots = orchestrator._read_roots_for_task(task)
    auth_context = None
    try:
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        auth_context = build_run_auth_context(
            workspace=Path(task.workspace) if task.workspace else None,
            metadata=metadata,
            available_mcp_aliases=set(orchestrator._config.mcp.servers.keys()),
        )
    except AuthResolutionError as e:
        logger.warning("Auth context unavailable during planning for %s: %s", task.id, e)

    if task.workspace:
        workspace_path = Path(task.workspace)
        if workspace_path.exists():
            # Run listing and analysis in parallel
            async def _do_listing():
                return await orchestrator._tools.execute(
                    "list_directory",
                    {},
                    workspace=workspace_path,
                    read_roots=read_roots,
                    auth_context=auth_context,
                )

            async def _do_analysis():
                analysis_path = read_roots[0] if read_roots else workspace_path
                if orchestrator._process and orchestrator._process.workspace_scan:
                    result = await orchestrator._analyze_workspace_for_process(
                        analysis_path,
                    )
                    return ("workspace", result)
                return ("code", await orchestrator._analyze_workspace(
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

    prompt = orchestrator._prompts.build_planner_prompt(
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

    model = orchestrator._router.select(tier=2, role="planner")
    request_messages = [{"role": "user", "content": prompt}]
    policy = ModelRetryPolicy.from_execution_config(orchestrator._config.execution)
    invocation_attempt = 0
    request_diag = None

    async def _invoke_model():
        nonlocal invocation_attempt, request_diag
        invocation_attempt += 1
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin="orchestrator.plan_task.complete",
        )
        orchestrator._emit(MODEL_INVOCATION, task.id, {
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
            max_tokens=orchestrator._planning_response_max_tokens(),
        )

    def _on_failure(
        attempt: int,
        max_attempts: int,
        error: BaseException,
        remaining: int,
    ) -> None:
        orchestrator._emit(MODEL_INVOCATION, task.id, {
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
        return await orchestrator._build_planner_degraded_plan(
            task=task,
            reason_code="planner_model_failure",
            detail=f"{type(e).__name__}: {e}",
        )
    orchestrator._emit(MODEL_INVOCATION, task.id, {
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
        plan = orchestrator._parse_plan(response, goal=task.goal)
    except ValueError as e:
        return await orchestrator._build_planner_degraded_plan(
            task=task,
            reason_code="planner_json_parse_failed",
            detail=str(e),
        )
    return orchestrator._apply_process_phase_mode(plan)

async def replan_task(
    orchestrator,
    task: Task,
    *,
    reason: str = "subtask_failures",
    failed_subtask_id: str = "",
    verification_feedback: str | None = None,
) -> bool:
    """Re-plan the task after subtask failures.

    Returns True if re-planning succeeded and execution can continue.
    """
    orchestrator._emit(TASK_REPLANNING, task.id, {
        "reason": reason,
        "failed_subtask_id": failed_subtask_id,
        "verification_feedback": verification_feedback or "",
    })

    discoveries = [d for d in task.decisions_log]
    errors = [
        f"{e.subtask}: {e.error}" for e in task.errors_encountered
    ]

    try:
        state_yaml = orchestrator._state.to_compact_yaml(task)
        model = orchestrator._router.select(tier=2, role="planner")
        policy = ModelRetryPolicy.from_execution_config(orchestrator._config.execution)
        max_structural_attempts = orchestrator._topology_retry_attempts()
        topology_feedback = ""

        for structural_attempt in range(1, max_structural_attempts + 1):
            prompt = orchestrator._prompts.build_replanner_prompt(
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
                orchestrator._emit(MODEL_INVOCATION, task.id, {
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
                    max_tokens=orchestrator._planning_response_max_tokens(),
                )

            def _on_replanner_failure(
                attempt: int,
                max_attempts: int,
                error: BaseException,
                remaining: int,
            ) -> None:
                orchestrator._emit(MODEL_INVOCATION, task.id, {
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
            orchestrator._emit(MODEL_INVOCATION, task.id, {
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
                parsed_plan = orchestrator._apply_process_phase_mode(
                    orchestrator._parse_plan(response, goal=task.goal),
                )
            except ValueError as e:
                parse_error = str(e).strip() or "invalid replanner JSON"
                orchestrator._emit(TASK_REPLAN_REJECTED, task.id, {
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
                orchestrator._state.save(task)
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
                new_plan = orchestrator._prepare_plan_for_execution(
                    task=task,
                    plan=parsed_plan,
                    context="replanner",
                )
            except ValueError as e:
                topology_error = str(e).strip() or "invalid replanned topology"

            if topology_error:
                orchestrator._emit(TASK_REPLAN_REJECTED, task.id, {
                    "failed_subtask_id": failed_subtask_id,
                    "reason": reason,
                    "validation_error": topology_error,
                    "old_subtask_ids": [s.id for s in task.plan.subtasks],
                    "new_subtask_ids": [s.id for s in parsed_plan.subtasks],
                    "attempt": structural_attempt,
                    "max_attempts": max_structural_attempts,
                })
                task.add_decision(f"Rejected replanned plan: {topology_error}")
                orchestrator._state.save(task)
                if structural_attempt >= max_structural_attempts:
                    return False
                topology_feedback = (
                    "Previous replanned plan was rejected for invalid topology.\n"
                    f"Validation error: {topology_error}\n"
                    "Return corrected JSON that preserves existing IDs and "
                    "satisfies all dependency and synthesis-topology constraints."
                )
                continue

            validation_error = orchestrator._validate_replan_contract(
                current_plan=task.plan,
                replanned_plan=new_plan,
            )
            if validation_error:
                orchestrator._emit(TASK_REPLAN_REJECTED, task.id, {
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
                orchestrator._state.save(task)
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
            orchestrator._state.save(task)

            orchestrator._emit(TASK_PLAN_READY, task.id, {
                "subtask_count": len(new_plan.subtasks),
                "version": new_plan.version,
                "replanned": True,
                "subtask_ids": [s.id for s in new_plan.subtasks],
            })
            return True

        return False

    except Exception as e:
        task.add_error("replanner", str(e))
        orchestrator._state.save(task)
        return False


# Extracted planning/replan orchestration helpers

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
    working, output_alignment = self._align_plan_output_coordination(
        plan=working,
    )
    if output_alignment:
        self._emit(TASK_PLAN_NORMALIZED, task.id, {
            "context": context,
            "normalized_subtasks": output_alignment,
            "plan_version": int(working.version),
        })
        post_alignment_issues = self._plan_topology_issues(working)
        if post_alignment_issues:
            raise ValueError(
                "Output-coordination normalization introduced invalid topology from "
                f"{context}: " + "; ".join(post_alignment_issues),
            )
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

def _phase_output_strategy(self, phase_id: str) -> str:
    process = self._process
    if process is None:
        return "direct"
    resolver = getattr(process, "phase_output_strategy", None)
    if callable(resolver):
        try:
            resolved = str(resolver(phase_id)).strip().lower()
        except Exception:
            resolved = ""
        if resolved in {"direct", "fan_in"}:
            return resolved
    return "direct"

def _phase_finalizer_id(self, phase_id: str) -> str:
    process = self._process
    if process is None:
        return ""
    resolver = getattr(process, "phase_finalizer_id", None)
    if callable(resolver):
        try:
            return str(resolver(phase_id) or "").strip()
        except Exception:
            return ""
    normalized_phase_id = str(phase_id or "").strip()
    if not normalized_phase_id:
        return ""
    return f"{normalized_phase_id}__finalize_output"

def _subtask_output_role(self, subtask: Subtask) -> str:
    phase_id = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
    if self._phase_output_strategy(phase_id) != "fan_in":
        return self._OUTPUT_ROLE_WORKER
    explicit = str(getattr(subtask, "output_role", "") or "").strip().lower()
    if explicit in {self._OUTPUT_ROLE_WORKER, self._OUTPUT_ROLE_PHASE_FINALIZER}:
        return explicit
    finalizer_id = self._phase_finalizer_id(phase_id)
    if finalizer_id and str(getattr(subtask, "id", "")).strip() == finalizer_id:
        return self._OUTPUT_ROLE_PHASE_FINALIZER
    return self._OUTPUT_ROLE_WORKER

def _align_plan_output_coordination(
    self,
    *,
    plan: Plan,
) -> tuple[Plan, list[dict[str, object]]]:
    process = self._process
    if process is None:
        return plan, []

    deliverables_by_phase = process.get_deliverables()
    if not isinstance(deliverables_by_phase, dict):
        deliverables_by_phase = {}

    phase_by_id: dict[str, object] = {}
    for phase in list(getattr(process, "phases", []) or []):
        phase_id = str(getattr(phase, "id", "") or "").strip()
        if phase_id:
            phase_by_id[phase_id] = phase

    changed: list[dict[str, object]] = []
    finalizer_id_by_phase: dict[str, str] = {}
    subtask_by_id = {subtask.id: subtask for subtask in plan.subtasks}

    for phase_id, deliverables in deliverables_by_phase.items():
        normalized_phase_id = str(phase_id or "").strip()
        if (
            not normalized_phase_id
            or not isinstance(deliverables, list)
            or not deliverables
            or self._phase_output_strategy(normalized_phase_id) != "fan_in"
        ):
            continue
        finalizer_id = self._phase_finalizer_id(normalized_phase_id)
        if not finalizer_id:
            continue
        phase_subtasks = [
            subtask
            for subtask in plan.subtasks
            if str(getattr(subtask, "phase_id", "") or "").strip() == normalized_phase_id
        ]
        if not phase_subtasks:
            continue

        worker_subtasks = [subtask for subtask in phase_subtasks if subtask.id != finalizer_id]
        finalizer = subtask_by_id.get(finalizer_id)
        if (
            finalizer is not None
            and str(getattr(finalizer, "phase_id", "") or "").strip() != normalized_phase_id
        ):
            finalizer = None
        if finalizer is None and worker_subtasks:
            phase = phase_by_id.get(normalized_phase_id)
            worker_ids = sorted({
                worker.id
                for worker in worker_subtasks
                if worker.id and worker.id != finalizer_id
            })
            model_tier = int(getattr(phase, "model_tier", 1) or 1) if phase else 1
            verification_tier = (
                int(getattr(phase, "verification_tier", 1) or 1) if phase else 1
            )
            if worker_subtasks:
                model_tier = max(
                    model_tier,
                    max(
                        int(getattr(worker, "model_tier", 1) or 1)
                        for worker in worker_subtasks
                    ),
                )
                verification_tier = max(
                    verification_tier,
                    max(
                        int(getattr(worker, "verification_tier", 1) or 1)
                        for worker in worker_subtasks
                    ),
                )
            finalizer = Subtask(
                id=finalizer_id,
                description=(
                    f"Finalize canonical deliverables for phase '{normalized_phase_id}' "
                    "from worker artifacts."
                ),
                depends_on=worker_ids,
                phase_id=normalized_phase_id,
                output_role=self._OUTPUT_ROLE_PHASE_FINALIZER,
                output_strategy="fan_in",
                model_tier=model_tier,
                verification_tier=verification_tier,
                is_critical_path=bool(getattr(phase, "is_critical_path", False)),
                acceptance_criteria=str(
                    getattr(phase, "acceptance_criteria", "") or "",
                ).strip(),
                max_retries=self._config.execution.max_subtask_retries,
            )
            plan.subtasks.append(finalizer)
            subtask_by_id[finalizer_id] = finalizer
            changed.append({
                "subtask_id": finalizer_id,
                "phase_id": normalized_phase_id,
                "reason": "fan_in_finalizer_injected",
                "depends_on": list(worker_ids),
            })

        if finalizer is not None:
            finalizer_id_by_phase[normalized_phase_id] = finalizer.id
            finalizer.output_role = self._OUTPUT_ROLE_PHASE_FINALIZER
            finalizer.output_strategy = "fan_in"
            required_deps = sorted({
                worker.id
                for worker in worker_subtasks
                if worker.id and worker.id != finalizer.id
            })
            merged_deps = sorted({
                dep_id
                for dep_id in [*list(getattr(finalizer, "depends_on", [])), *required_deps]
                if dep_id and dep_id != finalizer.id
            })
            if merged_deps != list(getattr(finalizer, "depends_on", [])):
                finalizer.depends_on = merged_deps
                changed.append({
                    "subtask_id": finalizer.id,
                    "phase_id": normalized_phase_id,
                    "reason": "fan_in_finalizer_dependencies_updated",
                    "depends_on": list(merged_deps),
                })

        if not worker_subtasks:
            continue
        for worker in worker_subtasks:
            worker.output_role = self._OUTPUT_ROLE_WORKER
            worker.output_strategy = "fan_in"
            worker_dependencies = list(getattr(worker, "depends_on", []))
            if finalizer is not None and finalizer.id in worker_dependencies:
                worker.depends_on = [
                    dep_id
                    for dep_id in worker_dependencies
                    if dep_id != finalizer.id
                ]
                changed.append({
                    "subtask_id": worker.id,
                    "phase_id": normalized_phase_id,
                    "reason": "fan_in_worker_cycle_dependency_removed",
                    "depends_on": list(worker.depends_on),
                })

    subtask_phase_by_id = {
        subtask.id: str(getattr(subtask, "phase_id", "") or "").strip()
        for subtask in plan.subtasks
        if str(getattr(subtask, "id", "") or "").strip()
    }
    for subtask in plan.subtasks:
        current_phase = str(getattr(subtask, "phase_id", "") or "").strip()
        remapped_dependencies: list[str] = []
        for dep in list(getattr(subtask, "depends_on", [])):
            dep_id = str(dep or "").strip()
            if not dep_id:
                continue
            dep_phase = subtask_phase_by_id.get(dep_id, "")
            replacement = dep_id
            if dep_phase and dep_phase in finalizer_id_by_phase:
                finalizer_id = finalizer_id_by_phase[dep_phase]
                if current_phase != dep_phase and dep_id != finalizer_id:
                    replacement = finalizer_id
            if replacement and replacement not in remapped_dependencies:
                remapped_dependencies.append(replacement)
        if remapped_dependencies != list(getattr(subtask, "depends_on", [])):
            subtask.depends_on = remapped_dependencies
            changed.append({
                "subtask_id": subtask.id,
                "phase_id": current_phase,
                "reason": "fan_in_dependency_remapped_to_finalizer",
                "depends_on": list(remapped_dependencies),
            })

    for subtask in plan.subtasks:
        phase_id = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
        strategy = self._phase_output_strategy(phase_id)
        role = self._subtask_output_role(subtask)
        before_role = str(getattr(subtask, "output_role", "") or "")
        before_strategy = str(getattr(subtask, "output_strategy", "") or "")
        subtask.output_role = role
        subtask.output_strategy = strategy
        if before_role != subtask.output_role or before_strategy != subtask.output_strategy:
            changed.append({
                "subtask_id": subtask.id,
                "phase_id": str(getattr(subtask, "phase_id", "") or "").strip(),
                "reason": "subtask_output_policy_aligned",
                "output_role": subtask.output_role,
                "output_strategy": subtask.output_strategy,
            })

    deduped: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in changed:
        key = (
            str(item.get("subtask_id", "")).strip(),
            str(item.get("reason", "")).strip(),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return plan, deduped

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

def _plan_topology_issues(cls, plan: Plan) -> list[str]:
    """Return deterministic topology issues for a plan graph."""
    del cls  # retained for compatibility with classmethod wrapper
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
    cycle = _detect_dependency_cycle(adjacency)
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

    topology_issues = _plan_topology_issues(None, replanned_plan)
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
            output_role=str(s.get("output_role", "") or ""),
            output_strategy=str(s.get("output_strategy", "") or ""),
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
                    output_role=self._OUTPUT_ROLE_WORKER,
                    output_strategy=self._phase_output_strategy(phase.id),
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
        existing.output_strategy = self._phase_output_strategy(phase.id)
        existing.output_role = self._OUTPUT_ROLE_WORKER
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
