"""Dispatch and iteration runtime helpers for orchestrator extraction."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loom.engine.iteration_gates import IterationEvaluation
from loom.engine.runner import SubtaskResult, SubtaskResultStatus, ToolCallRecord
from loom.engine.verification import VerificationResult
from loom.engine.verification.development import optional_failure_capability_for_reason
from loom.engine.verification.policy import normalize_profile, resolve_policy_decision
from loom.events.types import (
    ARTIFACT_SEAL_VALIDATION,
    ITERATION_COMPLETED,
    ITERATION_GATE_FAILED,
    ITERATION_RETRYING,
    ITERATION_STARTED,
    ITERATION_STATE_RECONCILED,
    ITERATION_TERMINAL,
    SUBTASK_COMPLETED,
    SUBTASK_FAILED,
    SUBTASK_POLICY_RECONCILED,
    SUBTASK_RETRYING,
    SUBTASK_STARTED,
    SYNTHESIS_INPUT_GATE_DECISION,
)
from loom.processes.schema import IterationPolicy
from loom.recovery.approval import ApprovalDecision, ApprovalRequest
from loom.recovery.retry import AttemptRecord, RetryStrategy
from loom.state.evidence import merge_evidence_records
from loom.state.task_state import Subtask, SubtaskStatus, Task, TaskStatus

logger = logging.getLogger(__name__)


def iteration_retry_mode(orchestrator, subtask) -> tuple[bool, str]:
    """Resolve whether a subtask run is an iteration retry and selected strategy."""
    policy = orchestrator._phase_iteration_policy(subtask)
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


def observe_iteration_runner_invocation(orchestrator, subtask) -> None:
    """Record runner invocation counts for iteration-enabled phases."""
    policy = orchestrator._phase_iteration_policy(subtask)
    if policy is None:
        return
    subtask.iteration_runner_invocations = int(
        max(0, subtask.iteration_runner_invocations) + 1,
    )
    if subtask.iteration_max_attempts <= 0:
        subtask.iteration_max_attempts = int(max(1, policy.max_attempts))


def observe_iteration_runtime_usage(orchestrator, *, task, subtask, result) -> None:
    """Accumulate iteration runtime usage after runner invocation."""
    if orchestrator._phase_iteration_policy(subtask) is None:
        return
    orchestrator._update_iteration_runtime(task=task, subtask=subtask, result=result)


def _process_outcome_policy(orchestrator) -> dict[str, object]:
    process = getattr(orchestrator, "_process", None)
    if process is None:
        return {}
    resolver = getattr(process, "verifier_outcome_policy", None)
    if callable(resolver):
        outcome_policy = resolver()
        if isinstance(outcome_policy, dict):
            normalized = dict(outcome_policy)
            optional_capabilities_getter = getattr(
                process,
                "verifier_optional_capabilities",
                None,
            )
            if callable(optional_capabilities_getter):
                optional_capabilities = optional_capabilities_getter()
                if isinstance(optional_capabilities, list) and optional_capabilities:
                    normalized["optional_capabilities"] = list(optional_capabilities)
            treat_infra_getter = getattr(
                process,
                "verifier_treat_infra_as_warning",
                None,
            )
            if callable(treat_infra_getter):
                normalized["treat_verifier_infra_as_warning"] = bool(
                    treat_infra_getter(),
                )
            return normalized
    verification_policy = getattr(process, "verification_policy", None)
    outcome_policy = getattr(verification_policy, "outcome_policy", {})
    if isinstance(outcome_policy, dict):
        return dict(outcome_policy)
    return {}


def _should_complete_with_warning_success(
    orchestrator,
    *,
    subtask: Subtask,
    verification: VerificationResult,
    policy_action: str,
) -> bool:
    if policy_action != "pass_with_warnings":
        return False
    if subtask.is_synthesis:
        return False
    outcome_policy = _process_outcome_policy(orchestrator)
    if not outcome_policy:
        return False
    if not bool(outcome_policy.get("treat_verifier_infra_as_warning", False)):
        return False
    metadata = dict(verification.metadata) if isinstance(verification.metadata, dict) else {}
    capability = ""
    dev_summary = metadata.get("dev_verification_summary", {})
    if isinstance(dev_summary, dict):
        reason_code = str(getattr(verification, "reason_code", "") or "").strip().lower()
        capability = optional_failure_capability_for_reason(
            dev_summary,
            reason_code=reason_code,
        )
    if not capability:
        return False
    optional_capabilities = outcome_policy.get("optional_capabilities", [])
    if not isinstance(optional_capabilities, list):
        optional_capabilities = []
    normalized_optional = {
        str(item or "").strip().lower()
        for item in optional_capabilities
        if str(item or "").strip()
    }
    return capability in normalized_optional


def _apply_warning_success(
    *,
    result: SubtaskResult,
    verification: VerificationResult,
    note: str,
) -> VerificationResult:
    result.status = SubtaskResultStatus.SUCCESS
    result.summary = "\n".join(
        part
        for part in [result.summary or "", note]
        if part
    ).strip()
    metadata = dict(verification.metadata) if isinstance(verification.metadata, dict) else {}
    metadata["warning_success"] = True
    metadata["warning_success_note"] = note
    verification.metadata = metadata
    verification.passed = True
    verification.outcome = "pass_with_warnings"
    verification.feedback = "\n".join(
        part for part in [verification.feedback or "", note] if part
    ).strip()
    verification.confidence = min(0.75, max(0.35, float(verification.confidence or 0.5)))
    if str(verification.severity_class or "").strip().lower() not in {"infra", "inconclusive"}:
        verification.severity_class = "semantic"
    return verification

async def dispatch_subtask(
    orchestrator,
    task: Task,
    subtask: Subtask,
    attempts_by_subtask: dict[str, list[AttemptRecord]],
) -> tuple[Subtask, SubtaskResult, VerificationResult]:
    """Prepare and dispatch a subtask to the runner.

    Handles pre-dispatch bookkeeping (status, events, escalation)
    and returns (subtask, result, verification) for the orchestrator
    to process.
    """
    contract = orchestrator._validity_contract_for_subtask(subtask)
    required_verification_tier = orchestrator._synthesis_verification_floor(subtask)
    profile_resolution = orchestrator._resolve_verification_profile(
        task=task,
        subtask=subtask,
        tool_calls=None,
    )
    profile_confidence_threshold = float(
        getattr(
            orchestrator._config.verification,
            "resilience_profile_confidence_threshold",
            0.65,
        ) or 0.65,
    )
    effective_profile = normalize_profile(
        profile_resolution.profile
        if profile_resolution.confidence >= profile_confidence_threshold
        else profile_resolution.fallback_profile,
    )
    policy_mode = str(
        getattr(orchestrator._config.verification, "resilience_policy_mode", "enforce"),
    ).strip().lower()

    # Mark running and emit event (under lock for parallel safety)
    async with orchestrator._state_lock:
        subtask.status = SubtaskStatus.RUNNING
        if subtask.verification_tier < required_verification_tier:
            subtask.verification_tier = required_verification_tier
        await orchestrator._save_task_state(task)
    orchestrator._emit(SUBTASK_STARTED, task.id, {"subtask_id": subtask.id})

    # Determine escalation tier
    attempts = attempts_by_subtask.get(subtask.id, [])
    retry_strategy = (
        attempts[-1].retry_strategy
        if attempts
        else RetryStrategy.GENERIC
    )
    prior_successful_tool_calls: list[ToolCallRecord] = []
    prior_evidence_records = await orchestrator._evidence_for_subtask_async(
        task.id,
        subtask.id,
    )
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
    escalated_tier = orchestrator._retry.get_escalation_tier(
        attempt=len(attempts),
        original_tier=subtask.model_tier,
    )
    output_policy = orchestrator._output_write_policy_for_subtask(subtask=subtask)
    expected_deliverables = list(output_policy.get("expected_deliverables", []))
    forbidden_deliverables = list(output_policy.get("forbidden_deliverables", []))
    allowed_output_prefixes = orchestrator._fan_in_worker_output_prefixes(
        task=task,
        subtask=subtask,
    )
    stage_plan = orchestrator._finalizer_stage_publish_plan(
        task=task,
        subtask=subtask,
        canonical_deliverables=expected_deliverables,
        attempt_index=len(attempts) + 1,
    )
    runner_expected_deliverables = list(expected_deliverables)
    runner_forbidden_deliverables = list(forbidden_deliverables)
    if bool(stage_plan.get("enabled", False)):
        runner_expected_deliverables = list(stage_plan.get("stage_deliverables", []))
        runner_forbidden_deliverables = orchestrator._merge_unique_paths(
            runner_forbidden_deliverables,
            expected_deliverables,
        )
    subtask.output_role = str(
        output_policy.get("output_role", "") or orchestrator._OUTPUT_ROLE_WORKER,
    )
    subtask.output_strategy = str(output_policy.get("output_strategy", "") or "direct")
    manifest_requirements = orchestrator._evaluate_finalizer_manifest_requirements(
        task=task,
        subtask=subtask,
    )
    is_iteration_retry, iteration_strategy = orchestrator._iteration_retry_mode(subtask)
    targeted_iteration_retry = (
        is_iteration_retry and iteration_strategy == "targeted_remediation"
    )
    retry_context = orchestrator._retry.build_retry_context(attempts)
    retry_context = orchestrator._augment_retry_context_for_evidence_recovery(
        base_context=retry_context,
        reason_code=(
            str(attempts[-1].reason_code or "").strip().lower()
            if attempts
            else ""
        ),
        prior_evidence_records=prior_evidence_records,
    )
    retry_context = orchestrator._augment_retry_context_for_outputs(
        subtask=subtask,
        attempts=attempts,
        strategy=retry_strategy,
        expected_deliverables=expected_deliverables,
        forbidden_deliverables=runner_forbidden_deliverables,
        base_context=retry_context,
    )
    retry_context = orchestrator._augment_retry_context_for_stage_publish(
        base_context=retry_context,
        stage_plan=stage_plan,
    )
    retry_context = orchestrator._augment_retry_context_with_phase_artifacts(
        task=task,
        subtask=subtask,
        base_context=retry_context,
    )
    if bool(manifest_requirements.get("enabled", False)):
        policy = str(manifest_requirements.get("policy", "") or "").strip().lower()
        missing_worker_ids = list(manifest_requirements.get("missing_worker_ids", []))
        if missing_worker_ids:
            retry_context = (
                f"{retry_context}\n\n"
                "FINALIZER INPUT POLICY STATUS:\n"
                f"- policy: {policy}\n"
                "- workers missing manifest artifacts: "
                f"{', '.join(missing_worker_ids)}"
            ).strip()
        if missing_worker_ids and policy == "require_all_workers":
            message = (
                "Finalizer blocked: missing worker artifacts for phase policy "
                f"'require_all_workers': {', '.join(missing_worker_ids)}"
            )
            blocked = SubtaskResult(
                status=SubtaskResultStatus.FAILED,
                summary=message,
            )
            blocked_verification = VerificationResult(
                tier=max(1, int(subtask.verification_tier or required_verification_tier)),
                passed=False,
                confidence=0.0,
                feedback=message,
                outcome="fail",
                reason_code="finalizer_missing_worker_artifacts",
                severity_class="semantic",
                metadata={
                    "finalizer_input_policy": policy,
                    "missing_worker_ids": missing_worker_ids,
                },
            )
            return subtask, blocked, blocked_verification
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
        seal_passed, seal_mismatches, validated_seals = orchestrator._validate_artifact_seals(
            task=task,
        )
        orchestrator._emit(ARTIFACT_SEAL_VALIDATION, task.id, {
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
        gate_passed, verified_context_bundle, gate_error = (
            orchestrator._verified_context_for_synthesis(
                task=task,
                subtask=subtask,
                verification_profile=effective_profile,
            )
        )
        claim_graph = orchestrator._claim_graph_state(task)
        supported_total = 0
        unresolved_total = 0
        unresolved_hard_total = 0
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
            for values in unresolved_by_subtask.values():
                if not isinstance(values, list):
                    continue
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    status = str(item.get("status", "") or "").strip().lower()
                    if status in {"contradicted", "stale"}:
                        unresolved_hard_total += 1
        gate_decision = resolve_policy_decision(
            severity_class="semantic",
            reason_code=(
                "claim_contradicted"
                if unresolved_hard_total > 0
                else "coverage_below_threshold"
            ),
            profile=effective_profile,
            mode=policy_mode,
            contradiction_detected=bool(unresolved_hard_total > 0),
            profile_confidence=profile_resolution.confidence,
        )
        if not gate_passed and gate_decision.action == "pass_with_warnings":
            gate_passed = True
            gate_error = (
                "Synthesis gate warning: proceeding under profile-aware policy "
                "despite inconclusive claim coverage."
            )
        orchestrator._emit(SYNTHESIS_INPUT_GATE_DECISION, task.id, {
            "subtask_id": subtask.id,
            "phase_id": subtask.phase_id,
            "passed": bool(gate_passed),
            "supported_claim_count": int(supported_total),
            "unresolved_claim_count": int(unresolved_total),
            "unresolved_hard_count": int(unresolved_hard_total),
            "verification_profile": effective_profile,
            "verification_profile_confidence": float(profile_resolution.confidence),
            "policy_action": gate_decision.action,
            "policy_mode": gate_decision.mode,
            "policy_shadow_diff": gate_decision.shadow_diff,
            "reason": (
                gate_error
                if gate_error
                else (
                    "verified_context_bundle_ready"
                    if verified_context_bundle
                    else "verified_context_bundle_unavailable"
                )
            ),
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
                    "verification_profile": effective_profile,
                    "verification_profile_confidence": float(
                        profile_resolution.confidence,
                    ),
                    "policy_action": gate_decision.action,
                    "policy_mode": gate_decision.mode,
                    "policy_shadow_diff": gate_decision.shadow_diff,
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
        elif gate_error:
            retry_context = (
                f"{retry_context}\n\n"
                f"{gate_error}\n"
                "If claims remain uncertain, keep uncertainty explicit and avoid "
                "stating unverified assertions as facts."
            ).strip()

    changelog = orchestrator._get_changelog(task)

    result, verification = await orchestrator._runner.run(
        task, subtask,
        model_tier=escalated_tier,
        retry_context=retry_context,
        changelog=changelog,
        prior_successful_tool_calls=prior_successful_tool_calls,
        prior_evidence_records=prior_evidence_records,
        expected_deliverables=runner_expected_deliverables,
        forbidden_deliverables=runner_forbidden_deliverables,
        allowed_output_prefixes=allowed_output_prefixes,
        enforce_deliverable_paths=bool(runner_expected_deliverables) and bool(
            attempts or targeted_iteration_retry or bool(stage_plan.get("enabled", False))
        ),
        edit_existing_only=bool(runner_expected_deliverables) and bool(
            attempts or targeted_iteration_retry or bool(stage_plan.get("enabled", False))
        ),
        retry_strategy=retry_strategy.value,
    )

    if bool(manifest_requirements.get("enabled", False)):
        allowed_manifest_paths = list(
            manifest_requirements.get("allowed_manifest_paths", []),
        )
        allowed_stage_prefixes = list(stage_plan.get("stage_prefixes", []))
        violations = orchestrator._manifest_only_input_violations(
            task=task,
            subtask=subtask,
            tool_calls=result.tool_calls,
            allowed_manifest_paths=allowed_manifest_paths,
            allowed_extra_prefixes=allowed_stage_prefixes,
        )
        if violations:
            message = (
                "Finalizer input policy violation: read access to intermediate "
                "artifacts outside latest worker manifest entries: "
                + ", ".join(violations)
            )
            result.status = SubtaskResultStatus.FAILED
            verification = VerificationResult(
                tier=max(1, int(subtask.verification_tier or required_verification_tier)),
                passed=False,
                confidence=0.0,
                feedback=message,
                outcome="fail",
                reason_code="manifest_input_policy_violation",
                severity_class="semantic",
                metadata={
                    "violating_paths": violations,
                    "finalizer_input_policy": str(
                        manifest_requirements.get("policy", ""),
                    ),
                },
            )

    profile_resolution = orchestrator._resolve_verification_profile(
        task=task,
        subtask=subtask,
        tool_calls=result.tool_calls,
    )
    effective_profile = normalize_profile(
        profile_resolution.profile
        if profile_resolution.confidence >= profile_confidence_threshold
        else profile_resolution.fallback_profile,
    )
    verification_metadata = (
        dict(verification.metadata)
        if isinstance(verification.metadata, dict)
        else {}
    )
    verification_metadata["verification_profile"] = effective_profile
    verification_metadata["verification_profile_confidence"] = float(
        profile_resolution.confidence,
    )
    verification_metadata["verification_profile_reasons"] = list(
        profile_resolution.reason_codes,
    )
    verification = orchestrator._verification_with_metadata(
        verification,
        metadata=verification_metadata,
    )
    verification = orchestrator._attach_runtime_assertions(
        subtask=subtask,
        verification=verification,
        tool_calls=result.tool_calls,
    )

    verification = orchestrator._enforce_required_fact_checker(
        subtask=subtask,
        result=result,
        verification=verification,
    )
    claim_extraction = contract.get("claim_extraction", {})
    claim_policy_enabled = orchestrator._to_bool(contract.get("enabled", False), False) and (
        isinstance(claim_extraction, dict)
        and orchestrator._to_bool(claim_extraction.get("enabled", False), False)
    )
    if claim_policy_enabled:
        verification = orchestrator._apply_intermediate_claim_pruning(
            task=task,
            subtask=subtask,
            result=result,
            verification=verification,
            contract=contract,
        )
        verification = orchestrator._enforce_temporal_consistency_gate(
            subtask=subtask,
            verification=verification,
            contract=contract,
        )
        if verification.passed:
            verification = orchestrator._enforce_synthesis_claim_gate(
                subtask=subtask,
                verification=verification,
                contract=contract,
            )
        if not verification.passed:
            metadata = (
                dict(verification.metadata)
                if isinstance(verification.metadata, dict)
                else {}
            )
            policy_decision = resolve_policy_decision(
                severity_class=str(verification.severity_class or ""),
                reason_code=str(verification.reason_code or ""),
                profile=effective_profile,
                mode=policy_mode,
                contradiction_detected=bool(
                    metadata.get("contradiction_detected", False),
                ),
                profile_confidence=profile_resolution.confidence,
            )
            metadata["policy_action"] = policy_decision.action
            metadata["policy_mode"] = policy_decision.mode
            metadata["policy_shadow_diff"] = policy_decision.shadow_diff
            verification = orchestrator._verification_with_metadata(
                verification,
                metadata=metadata,
            )
            if policy_decision.action == "pass_with_warnings" and subtask.is_synthesis:
                result.status = SubtaskResultStatus.SUCCESS
                warning_note = (
                    "Verification remained inconclusive, but profile-aware policy "
                    "allowed completion with warnings."
                )
                verification = orchestrator._verification_with_metadata(
                    verification,
                    metadata=metadata,
                    passed=True,
                    outcome=(
                        "partial_verified"
                        if orchestrator._config.verification.allow_partial_verified
                        else "pass_with_warnings"
                    ),
                    feedback="\n".join(
                        part
                        for part in [verification.feedback or "", warning_note]
                        if part
                    ),
                    severity_class=(
                        "inconclusive"
                        if str(verification.severity_class or "").strip().lower()
                        in {"inconclusive", "infra"}
                        else "semantic"
                    ),
                    confidence=min(0.7, max(0.35, float(verification.confidence or 0.5))),
                )
            else:
                result.status = SubtaskResultStatus.FAILED
    orchestrator._update_claim_graph_from_verification(
        task=task,
        subtask=subtask,
        verification=verification,
    )
    await orchestrator._persist_claim_validity_artifacts(
        task=task,
        subtask=subtask,
        verification=verification,
        evidence_records=result.evidence_records,
        tool_calls=result.tool_calls,
    )
    if (
        bool(stage_plan.get("enabled", False))
        and result.status != SubtaskResultStatus.FAILED
        and verification.passed
    ):
        commit_ok, commit_error = orchestrator._commit_finalizer_stage_publish(
            task=task,
            subtask=subtask,
            stage_plan=stage_plan,
        )
        if not commit_ok:
            result.status = SubtaskResultStatus.FAILED
            message = commit_error or "Transactional stage+commit publish failed."
            verification = VerificationResult(
                tier=max(1, int(subtask.verification_tier or required_verification_tier)),
                passed=False,
                confidence=0.0,
                feedback=message,
                outcome="fail",
                reason_code="output_publish_commit_failed",
                severity_class="semantic",
            )
            result.summary = (
                f"{result.summary}\n{message}".strip()
                if result.summary
                else message
            )

    return subtask, result, verification

async def handle_failure(
    orchestrator,
    task: Task,
    subtask: Subtask,
    result: SubtaskResult,
    verification: VerificationResult,
    attempts_by_subtask: dict[str, list[AttemptRecord]],
) -> dict[str, str | None] | None:
    """Process a failed subtask: record attempt, retry or replan."""
    await orchestrator._persist_subtask_evidence_async(
        task.id,
        subtask.id,
        result.evidence_records,
        tool_calls=result.tool_calls,
        workspace=task.workspace,
    )
    attempt_list = attempts_by_subtask.setdefault(subtask.id, [])
    verification_metadata = (
        dict(verification.metadata)
        if isinstance(verification.metadata, dict)
        else {}
    )
    profile = normalize_profile(
        verification_metadata.get("verification_profile", "hybrid"),
    )
    try:
        profile_confidence = float(
            verification_metadata.get("verification_profile_confidence", 0.0),
        )
    except (TypeError, ValueError):
        profile_confidence = 0.0
    policy_mode = str(
        getattr(orchestrator._config.verification, "resilience_policy_mode", "enforce"),
    ).strip().lower()
    policy_decision = resolve_policy_decision(
        severity_class=str(verification.severity_class or ""),
        reason_code=str(verification.reason_code or ""),
        profile=profile,
        mode=policy_mode,
        contradiction_detected=bool(
            verification_metadata.get("contradiction_detected", False),
        ),
        profile_confidence=profile_confidence,
    )
    strategy, missing_targets = orchestrator._retry.classify_failure(
        verification_feedback=verification.feedback,
        execution_error=result.summary,
        verification=verification,
        profile=profile,
        policy_mode=policy_mode,
        profile_confidence=profile_confidence,
    )
    combined_error = " | ".join(
        part for part in [verification.feedback, result.summary] if part
    )
    progress_signature = orchestrator._retry.progress_signature(
        verification_feedback=verification.feedback,
        execution_error=result.summary,
        reason_code=str(verification.reason_code or ""),
        strategy=strategy,
        missing_targets=missing_targets,
    )
    attempt_record = AttemptRecord(
        attempt=len(attempt_list) + 1,
        tier=orchestrator._retry.get_escalation_tier(
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
        reason_code=str(verification.reason_code or "").strip().lower(),
        severity_class=str(verification.severity_class or "").strip().lower(),
        policy_action=policy_decision.action,
        progress_signature=progress_signature,
    )
    attempt_list.append(attempt_record)
    await orchestrator._persist_subtask_attempt_record(
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
        verification_retry = await orchestrator._retry_verification_only(
            task=task,
            subtask=subtask,
            result=result,
            attempts=attempt_list,
        )
        if verification_retry.passed:
            await orchestrator._handle_success(task, subtask, result, verification_retry)
            return None
        verification = verification_retry
        attempt_record.feedback = verification_retry.feedback or attempt_record.feedback
        if verification_retry.feedback:
            attempt_record.error = " | ".join(
                part for part in [attempt_record.error, verification_retry.feedback]
                if part
            )

    resolution_plan = await orchestrator._plan_failure_resolution(
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

    if _should_complete_with_warning_success(
        orchestrator,
        subtask=subtask,
        verification=verification,
        policy_action=policy_decision.action,
    ):
        note = (
            "Verification completed with warnings because only optional "
            "development verification capabilities failed in this environment."
        )
        verification = _apply_warning_success(
            result=result,
            verification=verification,
            note=note,
        )
        attempt_record.feedback = verification.feedback or attempt_record.feedback
        await orchestrator._handle_success(task, subtask, result, verification)
        return None

    critical_path_behavior = orchestrator._critical_path_behavior()
    hard_invariant_failure = orchestrator._is_hard_invariant_failure(verification)
    if (
        strategy == RetryStrategy.UNCONFIRMED_DATA
        and not hard_invariant_failure
        and (
            not subtask.is_critical_path
            or critical_path_behavior == "queue_follow_up"
        )
    ):
        await orchestrator._queue_remediation_work_item(
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
        orchestrator._apply_unconfirmed_follow_up_success(
            result=result,
            verification=verification,
            note=note,
            default_reason_code=default_reason,
        )
        await orchestrator._handle_success(task, subtask, result, verification)
        return None

    async with orchestrator._state_lock:
        orchestrator._record_artifact_seals(
            task=task,
            subtask_id=subtask.id,
            tool_calls=result.tool_calls,
        )
        orchestrator._record_subtask_validity_metrics(
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
        await orchestrator._save_task_state(task)

    orchestrator._emit(SUBTASK_FAILED, task.id, {
        "subtask_id": subtask.id,
        "verification_tier": verification.tier,
        "feedback": verification.feedback,
        "verification_outcome": verification.outcome,
        "reason_code": verification.reason_code,
    })

    if (
        strategy == RetryStrategy.UNCONFIRMED_DATA
        and not hard_invariant_failure
        and orchestrator._is_placeholder_unconfirmed_failure(verification=verification)
        and subtask.retry_count < subtask.max_retries
    ):
        (
            _resolved_deterministically,
            deterministic_note,
            deterministic_details,
        ) = await orchestrator._run_deterministic_placeholder_prepass(
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

    no_progress_exhausted = orchestrator._retry.should_stop_for_no_progress(
        attempt_list,
        max_stalled_attempts=int(
            getattr(
                orchestrator._config.verification,
                "resilience_no_progress_attempts",
                2,
            ) or 2,
        ),
    )
    if no_progress_exhausted:
        no_progress_note = (
            "Retry policy stopped additional attempts due to no-progress "
            "signatures across consecutive retries."
        )
        verification.feedback = (
            f"{verification.feedback}\n{no_progress_note}".strip()
            if verification.feedback
            else no_progress_note
        )

    runner_cap_exhausted = False
    iteration_policy = orchestrator._phase_iteration_policy(subtask)
    iteration_budget_reason = ""
    if (
        iteration_policy is not None
        and int(iteration_policy.max_total_runner_invocations) > 0
        and int(subtask.iteration_runner_invocations)
        >= int(iteration_policy.max_total_runner_invocations)
    ):
        runner_cap_exhausted = True
    if iteration_policy is not None:
        runtime = orchestrator._iteration_runtime_entry(task, subtask.id)
        iteration_budget_reason = orchestrator._iteration_budget_exhausted_reason(
            policy=iteration_policy,
            runtime=runtime,
        )

    if (
        not no_progress_exhausted
        and not runner_cap_exhausted
        and not iteration_budget_reason
        and subtask.retry_count < subtask.max_retries
    ):
        subtask.retry_count += 1
        async with orchestrator._state_lock:
            subtask.status = SubtaskStatus.PENDING
            task.update_subtask(
                subtask.id,
                status=SubtaskStatus.PENDING,
                retry_count=subtask.retry_count,
            )
            await orchestrator._save_task_state(task)

        orchestrator._emit(SUBTASK_RETRYING, task.id, {
            "subtask_id": subtask.id,
            "attempt": subtask.retry_count,
            "escalated_tier": orchestrator._retry.get_escalation_tier(
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
            replan_request = await orchestrator._request_iteration_replan(
                task=task,
                subtask=subtask,
                policy=iteration_policy,
                terminal_reason=terminal_reason,
                gate_summary=gate_summary,
            )
            if replan_request is not None:
                async with orchestrator._state_lock:
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
                    await orchestrator._save_task_state(task)
                return replan_request

            if subtask.is_critical_path:
                await orchestrator._abort_on_critical_path_failure(
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

            async with orchestrator._state_lock:
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
                await orchestrator._save_task_state(task)
            return None
        # All retries exhausted.
        # Critical-path failures abort the remaining plan.
        if subtask.is_critical_path:
            if await orchestrator._should_auto_replan_critical_path_scope_failure(
                task=task,
                subtask=subtask,
                verification=verification,
                attempts=attempt_list,
            ):
                verification_feedback = verification.feedback
                if resolution_plan:
                    details = (
                        f"{verification_feedback}\n\n"
                        "MODEL-PLANNED RESOLUTION:\n"
                        f"{resolution_plan}"
                    )
                    verification_feedback = details.strip()
                return {
                    "reason": "critical_path_scope_adaptation: "
                    + orchestrator._build_replan_reason(subtask, verification),
                    "failed_subtask_id": subtask.id,
                    "verification_feedback": verification_feedback,
                }
            if (
                strategy == RetryStrategy.UNCONFIRMED_DATA
                and not hard_invariant_failure
            ):
                if critical_path_behavior == "confirm_or_prune_then_queue":
                    remediation_recovered, _ = (
                        await orchestrator._run_confirm_or_prune_remediation(
                            task=task,
                            subtask=subtask,
                            attempts=attempt_list,
                            verification=verification,
                        )
                    )
                    if remediation_recovered:
                        return None
                    await orchestrator._queue_remediation_work_item(
                        task=task,
                        subtask=subtask,
                        verification=verification,
                        strategy=strategy,
                        blocking=True,
                    )
                    orchestrator._apply_unconfirmed_follow_up_success(
                        result=result,
                        verification=verification,
                        note=(
                            "Critical-path remediation queued as blocking "
                            "follow-up (policy: confirm_or_prune_then_queue)."
                        ),
                        default_reason_code="unconfirmed_critical_path",
                    )
                    await orchestrator._handle_success(task, subtask, result, verification)
                    return None
                if critical_path_behavior == "queue_follow_up":
                    await orchestrator._queue_remediation_work_item(
                        task=task,
                        subtask=subtask,
                        verification=verification,
                        strategy=strategy,
                        blocking=False,
                    )
                    orchestrator._apply_unconfirmed_follow_up_success(
                        result=result,
                        verification=verification,
                        note=(
                            "Critical-path remediation queued as follow-up "
                            "(policy: queue_follow_up)."
                        ),
                        default_reason_code="unconfirmed_critical_path",
                    )
                    await orchestrator._handle_success(task, subtask, result, verification)
                    return None
            await orchestrator._abort_on_critical_path_failure(
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
            "reason": orchestrator._build_replan_reason(subtask, verification),
            "failed_subtask_id": subtask.id,
            "verification_feedback": verification_feedback,
        }

    return None

async def handle_success(
    orchestrator,
    task: Task,
    subtask: Subtask,
    result: SubtaskResult,
    verification: VerificationResult,
) -> None:
    """Process a successful subtask: update state, check approval."""
    await orchestrator._persist_subtask_evidence_async(
        task.id,
        subtask.id,
        result.evidence_records,
        tool_calls=result.tool_calls,
        workspace=task.workspace,
    )
    try:
        orchestrator._record_fan_in_worker_artifacts(
            task=task,
            subtask=subtask,
            result=result,
        )
    except Exception:
        logger.debug(
            "Failed recording fan-in worker artifact manifest for %s/%s",
            task.id,
            subtask.id,
            exc_info=True,
        )
    summary = result.summary

    # Update state
    async with orchestrator._state_lock:
        orchestrator._record_artifact_seals(
            task=task,
            subtask_id=subtask.id,
            tool_calls=result.tool_calls,
        )
        orchestrator._record_subtask_validity_metrics(
            task=task,
            subtask=subtask,
            verification=verification,
        )
        if subtask.is_synthesis:
            summary = orchestrator._append_synthesis_provenance_footer(
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
        changelog = orchestrator._get_changelog(task)
        if changelog:
            change_summary = changelog.get_summary()
            task.workspace_changes.files_created = len(change_summary["created"])
            task.workspace_changes.files_modified = len(change_summary["modified"])
            task.workspace_changes.files_deleted = len(change_summary["deleted"])
            task.workspace_changes.last_change = datetime.now().isoformat()

        await orchestrator._save_task_state(task)

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
        await orchestrator._queue_remediation_work_item(
            task=task,
            subtask=subtask,
            verification=verification,
            strategy=RetryStrategy.UNCONFIRMED_DATA,
            blocking=False,
        )

    orchestrator._emit(SUBTASK_COMPLETED, task.id, {
        "subtask_id": subtask.id,
        "status": result.status,
        "summary": summary,
        "duration": result.duration_seconds,
        "verification_outcome": verification.outcome if verification else "",
        "reason_code": verification.reason_code if verification else "",
    })

    # Confidence scoring and approval check
    if verification:
        confidence = orchestrator._confidence.score(subtask, result, verification)
        decision = orchestrator._approval.check_approval(
            approval_mode=task.approval_mode,
            confidence=confidence.score,
            result=result,
            confidence_threshold=orchestrator._config.execution.auto_approve_confidence_threshold,
        )

        if decision in (
            ApprovalDecision.WAIT,
            ApprovalDecision.WAIT_WITH_TIMEOUT,
        ):
            async with orchestrator._state_lock:
                task.status = TaskStatus.WAITING_APPROVAL
                await orchestrator._save_task_state(task)

            approved = await orchestrator._approval.request_approval(
                ApprovalRequest(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    reason=f"Confidence {confidence.band} ({confidence.score:.2f})",
                    proposed_action=result.summary,
                    risk_level=confidence.band,
                    details=confidence.components,
                    # Keep the run paused until the user explicitly decides.
                    auto_approve_timeout=None,
                )
            )

            async with orchestrator._state_lock:
                task.status = TaskStatus.EXECUTING
                await orchestrator._save_task_state(task)

            if not approved:
                async with orchestrator._state_lock:
                    subtask.status = SubtaskStatus.FAILED
                    task.update_subtask(
                        subtask.id,
                        status=SubtaskStatus.FAILED,
                        summary="Rejected by human reviewer",
                    )
                    await orchestrator._save_task_state(task)

        elif decision == ApprovalDecision.ABORT:
            async with orchestrator._state_lock:
                subtask.status = SubtaskStatus.FAILED
                task.update_subtask(
                    subtask.id,
                    status=SubtaskStatus.FAILED,
                    summary="Aborted: confidence too low",
                )
                await orchestrator._save_task_state(task)

async def handle_iteration_after_success(
    orchestrator,
    *,
    task: Task,
    subtask: Subtask,
    result: SubtaskResult,
    verification: VerificationResult,
) -> dict[str, str | None] | None:
    policy = orchestrator._phase_iteration_policy(subtask)
    if policy is None:
        await orchestrator._handle_success(task, subtask, result, verification)
        return None

    if not subtask.iteration_loop_run_id:
        subtask.iteration_loop_run_id = f"iter-{uuid.uuid4().hex[:10]}"
        orchestrator._emit(ITERATION_STARTED, task.id, {
            "subtask_id": subtask.id,
            "phase_id": subtask.phase_id,
            "loop_run_id": subtask.iteration_loop_run_id,
            "max_attempts": int(policy.max_attempts),
            "max_runner_invocations": int(policy.max_total_runner_invocations),
        })

    runtime = orchestrator._iteration_runtime_entry(task, subtask.id)
    if "started_monotonic" not in runtime:
        runtime["started_monotonic"] = float(time.monotonic())
        runtime["started_at"] = datetime.now().isoformat()
    runtime["updated_at"] = datetime.now().isoformat()
    budget_snapshot = orchestrator._iteration_budget_snapshot(policy=policy, runtime=runtime)
    budget_reason = orchestrator._iteration_budget_exhausted_reason(policy=policy, runtime=runtime)

    evaluation: IterationEvaluation | None = None
    if not budget_reason:
        output_policy = orchestrator._output_write_policy_for_subtask(subtask=subtask)
        expected_deliverables = list(output_policy.get("expected_deliverables", []))
        evaluation = await orchestrator._iteration_gates.evaluate(
            policy=policy,
            result=result,
            verification=verification,
            workspace=Path(task.workspace) if task.workspace else None,
            expected_deliverables=expected_deliverables,
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
        await orchestrator._persist_iteration_evaluation(
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
        orchestrator._emit(ITERATION_COMPLETED, task.id, {
            "subtask_id": subtask.id,
            "phase_id": subtask.phase_id,
            "loop_run_id": subtask.iteration_loop_run_id,
            "attempt": attempt_index,
            "max_attempts": int(policy.max_attempts),
        })
        await orchestrator._handle_success(task, subtask, result, verification)
        return None

    gate_summary = (
        budget_reason
        if budget_reason
        else orchestrator._format_iteration_gate_failures(blocking_failures)
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

    orchestrator._emit(ITERATION_GATE_FAILED, task.id, {
        "subtask_id": subtask.id,
        "phase_id": subtask.phase_id,
        "loop_run_id": subtask.iteration_loop_run_id,
        "attempt": attempt_index,
        "max_attempts": int(policy.max_attempts),
        "terminal_reason": terminal_reason,
        "gate_summary": gate_summary,
    })

    if not terminal_reason:
        async with orchestrator._state_lock:
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
            await orchestrator._save_task_state(task)
        await orchestrator._persist_iteration_evaluation(
            task=task,
            subtask=subtask,
            policy=policy,
            evaluation=evaluation,
            attempt_index=attempt_index,
            status="retrying",
            gate_summary=gate_summary,
            budget_snapshot=budget_snapshot,
        )
        orchestrator._emit(ITERATION_RETRYING, task.id, {
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
    exhaustion_fingerprint = orchestrator._iteration_exhaustion_fingerprint(
        subtask=subtask,
        terminal_reason=terminal_reason,
        gate_summary=gate_summary,
    )
    await orchestrator._persist_iteration_evaluation(
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
    orchestrator._emit(ITERATION_TERMINAL, task.id, {
        "subtask_id": subtask.id,
        "phase_id": subtask.phase_id,
        "loop_run_id": subtask.iteration_loop_run_id,
        "attempt": attempt_index,
        "terminal_reason": terminal_reason,
        "gate_summary": gate_summary,
    })

    replan_request = await orchestrator._request_iteration_replan(
        task=task,
        subtask=subtask,
        policy=policy,
        terminal_reason=terminal_reason,
        gate_summary=gate_summary,
    )
    if replan_request is not None:
        async with orchestrator._state_lock:
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
            await orchestrator._save_task_state(task)
        return replan_request

    if subtask.is_critical_path:
        await orchestrator._abort_on_critical_path_failure(
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

    async with orchestrator._state_lock:
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
        await orchestrator._save_task_state(task)
    return None


# Extracted iteration runtime + reconciliation orchestration helpers

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
    await self._save_task_state(task)

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

    await self._save_task_state(task)
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
            "output_role": str(getattr(subtask, "output_role", "") or ""),
            "output_strategy": str(getattr(subtask, "output_strategy", "") or ""),
            "validity_contract_hash": str(
                getattr(subtask, "validity_contract_hash", "") or "",
            ),
        }

        self._apply_subtask_policy_from_process_phase(
            subtask=subtask,
            phase=phase,
        )
        phase_hint = phase_id or subtask.id
        phase_strategy = self._phase_output_strategy(phase_hint)
        subtask.output_strategy = phase_strategy
        finalizer_id = self._phase_finalizer_id(phase_hint)
        if phase_strategy == "fan_in" and finalizer_id and subtask.id == finalizer_id:
            subtask.output_role = self._OUTPUT_ROLE_PHASE_FINALIZER
        else:
            subtask.output_role = self._OUTPUT_ROLE_WORKER

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
            "output_role": str(getattr(subtask, "output_role", "") or ""),
            "output_strategy": str(getattr(subtask, "output_strategy", "") or ""),
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
    await self._save_task_state(task)
    self._emit(SUBTASK_POLICY_RECONCILED, task.id, {
        "run_id": self._task_run_id(task),
        "reconciled_subtasks": reconciled,
        "reconciled_count": len(reconciled),
    })


# Extracted dispatch exception normalizer

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
