"""Deterministic blocker classification and repair routing."""

from __future__ import annotations

from loom.engine.correction.types import (
    Blocker,
    CorrectionHandler,
    Repairability,
    RepairAction,
)

_TERMINAL_INTEGRITY_CODES = frozenset({
    "artifact_seal_invalid",
    "manifest_input_policy_violation",
    "output_publish_commit_failed",
    "path_policy_violation",
    "safety_policy_violation",
    "forbidden_canonical_write",
})
_HUMAN_CODES = frozenset({
    "approval_required",
    "authentication_required",
    "destructive_action_requires_approval",
})
_VERIFIER_CODES = frozenset({
    "parse_inconclusive",
    "required_verifier_empty",
    "required_verifier_missing",
    "infra_verifier_error",
})
_TOOL_CODES = frozenset({
    "tool_capability_unavailable",
    "tool_method_failed",
    "tool_runtime_capability_unavailable",
    "tool_runtime_retryable",
    "tool_transient_failure",
    "tool_upstream_unavailable",
    "tool_write_retryable",
})
_EVIDENCE_CODES = frozenset({
    "claim_insufficient_evidence",
    "coverage_below_threshold",
    "recommendation_unconfirmed",
    "unconfirmed_critical_path",
    "unconfirmed_noncritical",
})
_PLACEHOLDER_CODES = frozenset({
    "incomplete_deliverable_content",
    "incomplete_deliverable_placeholder",
})
_REASON_CODE_ALIASES = {
    "iteration_budget_exceeded": "runner_tool_budget_exhausted",
    "tool_budget_exhausted": "runner_tool_budget_exhausted",
}


def _metadata_list(metadata: dict[str, object], key: str) -> tuple[str, ...]:
    value = metadata.get(key, [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def classify_blockers(verification) -> tuple[Blocker, ...]:
    """Convert a verification result into stable, typed blockers."""
    metadata = (
        dict(verification.metadata)
        if isinstance(getattr(verification, "metadata", None), dict)
        else {}
    )
    code = str(getattr(verification, "reason_code", "") or "").strip().lower()
    code = _REASON_CODE_ALIASES.get(code, code)
    severity = str(getattr(verification, "severity_class", "") or "").strip().lower()
    feedback = str(getattr(verification, "feedback", "") or "").strip()
    targets = _metadata_list(metadata, "missing_targets")
    failed_checks = [
        check for check in (getattr(verification, "checks", None) or [])
        if not bool(getattr(check, "passed", False))
    ]

    repairability = Repairability.CONDITIONAL
    if code in _TERMINAL_INTEGRITY_CODES:
        repairability = Repairability.TERMINAL
    elif code in _HUMAN_CODES:
        repairability = Repairability.HUMAN_REQUIRED
    elif code == "claim_contradicted":
        remediation_mode = str(metadata.get("remediation_mode", "") or "").strip().lower()
        repairability = (
            Repairability.CONDITIONAL
            if remediation_mode in {"confirm_or_prune", "confirm_or_prune_then_queue"}
            or bool(metadata.get("prune_authorized", False))
            else Repairability.HUMAN_REQUIRED
        )
    elif code in _VERIFIER_CODES or code in _TOOL_CODES or code in _PLACEHOLDER_CODES:
        repairability = Repairability.AUTOMATIC
    elif code == "artifact_confirmation_required":
        repairability = Repairability.AUTOMATIC
    elif code in _EVIDENCE_CODES:
        repairability = Repairability.CONDITIONAL
    elif code == "iteration_gate_failed":
        repairability = Repairability.AUTOMATIC
    elif code in {
        "iteration_budget_exhausted",
        "iteration_exhausted",
        "max_attempts_exhausted",
        "max_runner_invocations_exhausted",
        "no_improvement",
    }:
        repairability = Repairability.CONDITIONAL
    elif code == "hard_invariant_failed" or severity == "hard_invariant":
        check_text = " ".join(
            f"{getattr(check, 'name', '')} {getattr(check, 'detail', '')}"
            for check in failed_checks
        ).lower()
        if any(token in check_text for token in ("safety", "seal", "canonical write", "path")):
            repairability = Repairability.TERMINAL
        elif any(token in check_text for token in ("placeholder", "deliverable", "missing")):
            repairability = Repairability.AUTOMATIC

    if failed_checks:
        blockers = []
        for check in failed_checks:
            check_name = str(getattr(check, "name", "") or "verification_check")
            detail = str(getattr(check, "detail", "") or feedback or check_name)
            blockers.append(
                Blocker(
                    code=code or check_name.lower().replace(" ", "_"),
                    message=detail,
                    blocking=True,
                    repairability=repairability,
                    source=f"verification:{check_name}",
                    targets=targets,
                )
            )
        return tuple(blockers)
    return (
        Blocker(
            code=code or "verification_failed",
            message=feedback or "Verification failed without a structured explanation.",
            blocking=True,
            repairability=repairability,
            targets=targets,
        ),
    )


def select_handler(blockers: tuple[Blocker, ...]) -> CorrectionHandler:
    codes = {blocker.code for blocker in blockers}
    repairabilities = {blocker.repairability for blocker in blockers}
    if Repairability.TERMINAL in repairabilities:
        return CorrectionHandler.NONE
    if Repairability.HUMAN_REQUIRED in repairabilities:
        return CorrectionHandler.HUMAN_REVIEW
    if codes & _VERIFIER_CODES:
        return CorrectionHandler.RETRY_VERIFICATION
    if "runner_tool_budget_exhausted" in codes:
        return CorrectionHandler.CHECKPOINT_CONTINUE
    if codes & _PLACEHOLDER_CODES:
        return CorrectionHandler.PLACEHOLDER_PREPASS
    if "artifact_confirmation_required" in codes:
        return CorrectionHandler.CONTEXT_REFRESH
    if codes & _TOOL_CODES:
        return CorrectionHandler.SOURCE_FALLBACK
    if codes & _EVIDENCE_CODES:
        return CorrectionHandler.CONFIRM_OR_PRUNE
    if codes & {
        "iteration_budget_exhausted",
        "iteration_exhausted",
        "max_attempts_exhausted",
        "max_runner_invocations_exhausted",
        "no_improvement",
    }:
        return CorrectionHandler.REPLAN
    return CorrectionHandler.RETRY_EXECUTION


def build_actions(
    handler: CorrectionHandler,
    blockers: tuple[Blocker, ...],
) -> tuple[RepairAction, ...]:
    targets = sorted({target for blocker in blockers for target in blocker.targets})
    action_type = {
        CorrectionHandler.RETRY_VERIFICATION: "rerun_verifier",
        CorrectionHandler.CHECKPOINT_CONTINUE: "continue_from_partial_checkpoint",
        CorrectionHandler.PLACEHOLDER_PREPASS: "confirm_or_prune_placeholders",
        CorrectionHandler.CONTEXT_REFRESH: "refresh_artifact_context",
        CorrectionHandler.SOURCE_FALLBACK: "change_tool_or_source_method",
        CorrectionHandler.CONFIRM_OR_PRUNE: "confirm_or_prune_evidence",
        CorrectionHandler.REPLAN: "request_structural_replan",
        CorrectionHandler.HUMAN_REVIEW: "request_human_decision",
        CorrectionHandler.RETRY_EXECUTION: "retry_with_targeted_feedback",
        CorrectionHandler.NONE: "stop",
    }[handler]
    return (
        RepairAction(
            action_type=action_type,
            handler=handler,
            arguments={
                "reason_codes": sorted({blocker.code for blocker in blockers}),
                "targets": targets,
                "guardrails": (
                    [
                        "reuse existing deliverables and validated evidence",
                        "do not repeat broad research",
                        "repair only named missing targets",
                        "finish immediately after acceptance criteria are satisfied",
                    ]
                    if handler == CorrectionHandler.CHECKPOINT_CONTINUE
                    else []
                ),
            },
        ),
    )
