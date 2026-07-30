"""Deterministic blocker classification and repair routing."""

from __future__ import annotations

import re

from loom.engine.correction.types import (
    Blocker,
    BlockerClass,
    CorrectionHandler,
    Repairability,
    RepairAction,
)

_TERMINAL_INTEGRITY_CODES = frozenset({
    "artifact_seal_invalid",
    "path_policy_violation",
    "safety_policy_violation",
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
_SCHEMA_CODES = frozenset({
    "csv_schema_mismatch",
})
_CONTRACT_CODES = frozenset({
    "aggregate_scan_not_zonal",
    "incomplete_pestle_coverage_unverified_assumptions_remain",
    "missing_leading_indicators_and_geographic_granularity",
    "missing_market_specific_recommendations",
    "missing_required_contract_field",
    "missing_structured_leading_indicators",
    "structured_output_contract_failed",
    "unverified_primary_deliverables",
})
_REASON_CODE_ALIASES = {
    "budget_exhausted_incomplete": "runner_tool_budget_exhausted",
    "iteration_budget_exceeded": "runner_tool_budget_exhausted",
    "tool_budget_exceeded": "runner_tool_budget_exhausted",
    "tool_budget_exhausted": "runner_tool_budget_exhausted",
}

_BUDGET_REASON_RE = re.compile(
    r"(?:tool|iteration|runner).*(?:budget|limit).*(?:exhaust|exceed|incomplete)"
    r"|(?:budget|limit).*(?:exhaust|exceed|incomplete)",
)
_EVIDENCE_REASON_RE = re.compile(
    r"(?:unsupported|unverified|insufficient|missing|invalid).*(?:evidence|citation|source)"
    r"|(?:evidence|citation|source).*(?:unsupported|unverified|insufficient|missing|invalid)",
)
_MISSING_ARTIFACT_RE = re.compile(
    r"(?:missing|required|incomplete|unmet).*(?:deliverable|artifact|field|register|output)"
    r"|(?:deliverable|artifact|field|register|output).*(?:missing|required|incomplete|unmet)",
)


def canonicalize_reason_code(
    reason_code: str,
    *,
    feedback: str = "",
    check_text: str = "",
) -> tuple[str, BlockerClass]:
    """Map open-ended verifier labels into a closed recovery taxonomy."""
    raw = str(reason_code or "").strip().lower()
    text = " ".join([raw, str(feedback or ""), str(check_text or "")]).lower()
    aliased = _REASON_CODE_ALIASES.get(raw, raw)

    if any(
        token in text
        for token in (
            "outside workspace",
            "path traversal",
            "workspace escape",
            "symlink escape",
        )
    ):
        return "path_policy_violation", BlockerClass.INTEGRITY
    if aliased in {
        "forbidden_canonical_write",
        "forbidden_output_path",
        "manifest_input_policy_violation",
        "output_path_policy_violation",
        "output_publish_commit_failed",
    } or (
        "output" in text
        and any(token in text for token in ("path", "canonical", "staging", "publish"))
        and any(token in text for token in ("forbidden", "policy", "failed", "reserved"))
    ):
        return "forbidden_output_path", BlockerClass.ARTIFACT_WRITE_POLICY
    if aliased in _TERMINAL_INTEGRITY_CODES:
        blocker_class = (
            BlockerClass.SAFETY
            if any(token in aliased for token in ("safety", "path", "forbidden"))
            else BlockerClass.INTEGRITY
        )
        return aliased, blocker_class
    if aliased == "authentication_required" or "auth_preflight" in aliased:
        return "authentication_required", BlockerClass.AUTHENTICATION
    if aliased in {"approval_required", "destructive_action_requires_approval"}:
        return aliased, BlockerClass.AUTHORIZATION
    if aliased == "csv_schema_mismatch" or (
        "schema" in text and any(token in text for token in ("csv", "column", "row"))
    ):
        return "csv_schema_mismatch", BlockerClass.ARTIFACT_SCHEMA
    if (
        "verification" in text
        and any(token in text for token in ("budget", "tool method", "tool_method"))
        and any(token in text for token in ("exhaust", "failed", "failure"))
    ):
        return (
            aliased or "infra_verifier_error",
            BlockerClass.VERIFIER_FAILURE,
        )
    if aliased in _CONTRACT_CODES or (
        any(token in text for token in ("structured", "required", "primary"))
        and any(token in text for token in ("field", "indicator", "deliverable", "contract"))
        and any(token in text for token in ("missing", "unverified", "invalid", "failed"))
    ):
        return aliased or "structured_output_contract_failed", BlockerClass.ARTIFACT_CONTRACT
    if aliased == "runner_tool_budget_exhausted" or _BUDGET_REASON_RE.search(raw):
        return "runner_tool_budget_exhausted", BlockerClass.RESOURCE_EXHAUSTION
    if aliased in _VERIFIER_CODES or "verifier" in aliased and (
        "parse" in aliased or "infra" in aliased or "missing" in aliased
    ):
        return aliased or "infra_verifier_error", BlockerClass.VERIFIER_FAILURE
    if aliased in _TOOL_CODES:
        blocker_class = (
            BlockerClass.SOURCE_UNAVAILABLE
            if any(token in text for token in ("source", "fetch", "http", "upstream"))
            else BlockerClass.TOOL_FAILURE
        )
        return aliased, blocker_class
    if any(token in text for token in ("http 403", "http 404", "access denied", "anti-bot")):
        return "tool_method_failed", BlockerClass.SOURCE_UNAVAILABLE
    if aliased in _PLACEHOLDER_CODES or "placeholder" in text:
        return "incomplete_deliverable_content", BlockerClass.ARTIFACT_MISSING
    if _MISSING_ARTIFACT_RE.search(raw):
        return "incomplete_deliverable_content", BlockerClass.ARTIFACT_MISSING
    if aliased == "claim_contradicted" or any(
        token in raw for token in ("contradict", "inconsisten", "conflict")
    ):
        return "claim_contradicted", BlockerClass.CONTRADICTION
    if aliased in _EVIDENCE_CODES or _EVIDENCE_REASON_RE.search(raw):
        return "claim_insufficient_evidence", BlockerClass.EVIDENCE_GAP
    if aliased in {
        "iteration_budget_exhausted",
        "iteration_exhausted",
        "max_attempts_exhausted",
        "max_runner_invocations_exhausted",
        "no_improvement",
    }:
        return aliased, BlockerClass.RESOURCE_EXHAUSTION
    if aliased == "hard_invariant_failed":
        if any(token in text for token in ("safety", "seal", "integrity", "canonical write")):
            return aliased, BlockerClass.INTEGRITY
        if any(token in text for token in ("missing", "deliverable", "field", "placeholder")):
            return "incomplete_deliverable_content", BlockerClass.ARTIFACT_MISSING
    if aliased.startswith("tool_"):
        return aliased, BlockerClass.TOOL_FAILURE
    if aliased.startswith("infra_") or "infrastructure" in text:
        return aliased, BlockerClass.INFRASTRUCTURE
    return aliased or "verification_failed", BlockerClass.SEMANTIC_GAP


def _metadata_list(metadata: dict[str, object], key: str) -> tuple[str, ...]:
    value = metadata.get(key, [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _schema_diagnostics(detail: str) -> dict[str, object]:
    """Extract stable CSV diagnostics from deterministic verifier feedback."""
    match = re.search(
        r"CSV row\s+(?P<row>\d+)\s+has\s+(?P<actual>\d+)\s+columns?\s+"
        r"\(expected\s+(?P<expected>\d+)\)",
        str(detail or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return {}
    return {
        "row_number": int(match.group("row")),
        "actual_columns": int(match.group("actual")),
        "expected_columns": int(match.group("expected")),
    }


def _repairability_for(
    *,
    code: str,
    blocker_class: BlockerClass,
    metadata: dict[str, object],
    severity: str,
    check_text: str,
) -> Repairability:
    """Resolve repairability for one normalized blocker."""
    if blocker_class in {BlockerClass.SAFETY, BlockerClass.INTEGRITY}:
        return Repairability.TERMINAL
    if blocker_class in {BlockerClass.AUTHENTICATION, BlockerClass.AUTHORIZATION}:
        return Repairability.HUMAN_REQUIRED
    if code == "claim_contradicted":
        remediation_mode = str(metadata.get("remediation_mode", "") or "").strip().lower()
        if (
            remediation_mode in {"confirm_or_prune", "confirm_or_prune_then_queue"}
            or bool(metadata.get("prune_authorized", False))
        ):
            return Repairability.CONDITIONAL
        return Repairability.HUMAN_REQUIRED
    if blocker_class in {
        BlockerClass.ARTIFACT_MISSING,
        BlockerClass.ARTIFACT_SCHEMA,
        BlockerClass.ARTIFACT_CONTRACT,
        BlockerClass.ARTIFACT_WRITE_POLICY,
        BlockerClass.INFRASTRUCTURE,
        BlockerClass.SOURCE_UNAVAILABLE,
        BlockerClass.TOOL_FAILURE,
        BlockerClass.VERIFIER_FAILURE,
    } or code in {"artifact_confirmation_required", "iteration_gate_failed"}:
        return Repairability.AUTOMATIC
    if code in _EVIDENCE_CODES or code in {
        "iteration_budget_exhausted",
        "iteration_exhausted",
        "max_attempts_exhausted",
        "max_runner_invocations_exhausted",
        "no_improvement",
    }:
        return Repairability.CONDITIONAL
    if code == "hard_invariant_failed" or severity == "hard_invariant":
        check_text_lower = check_text.lower()
        if any(
            token in check_text_lower
            for token in ("safety", "seal", "canonical write", "integrity")
        ):
            return Repairability.TERMINAL
        if any(
            token in check_text_lower
            for token in ("placeholder", "deliverable", "missing", "field")
        ):
            return Repairability.AUTOMATIC
    return Repairability.CONDITIONAL


def _check_reason_code(check_name: str, detail: str, fallback: str) -> str:
    """Prefer a verifier check's own reason code when it supplied one."""
    match = re.search(
        r"\breason_code\s*[=:]\s*([a-z0-9_-]+)",
        str(detail or ""),
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()
    normalized_name = str(check_name or "").strip().lower()
    if normalized_name in (
        _SCHEMA_CODES
        | _CONTRACT_CODES
        | _VERIFIER_CODES
        | _TOOL_CODES
        | _EVIDENCE_CODES
    ):
        return normalized_name
    return fallback


def classify_blockers(verification) -> tuple[Blocker, ...]:
    """Convert a verification result into stable, typed blockers."""
    metadata = (
        dict(verification.metadata)
        if isinstance(getattr(verification, "metadata", None), dict)
        else {}
    )
    raw_code = str(getattr(verification, "reason_code", "") or "").strip().lower()
    severity = str(getattr(verification, "severity_class", "") or "").strip().lower()
    feedback = str(getattr(verification, "feedback", "") or "").strip()
    targets = _metadata_list(metadata, "missing_targets")
    failed_checks = [
        check for check in (getattr(verification, "checks", None) or [])
        if not bool(getattr(check, "passed", False))
    ]
    check_text = " ".join(
        f"{getattr(check, 'name', '')} {getattr(check, 'detail', '')}"
        for check in failed_checks
    )
    code, blocker_class = canonicalize_reason_code(
        raw_code,
        feedback=feedback,
        check_text=check_text,
    )

    repairability = _repairability_for(
        code=code,
        blocker_class=blocker_class,
        metadata=metadata,
        severity=severity,
        check_text=check_text,
    )

    if failed_checks:
        blockers = []
        for check in failed_checks:
            check_name = str(getattr(check, "name", "") or "verification_check")
            detail = str(getattr(check, "detail", "") or feedback or check_name)
            check_code, check_class = canonicalize_reason_code(
                _check_reason_code(check_name, detail, raw_code),
                feedback=feedback,
                check_text=f"{check_name} {detail}",
            )
            check_repairability = _repairability_for(
                code=check_code,
                blocker_class=check_class,
                metadata=metadata,
                severity=severity,
                check_text=f"{check_name} {detail}",
            )
            blocker_targets = list(targets)
            if check_code in _SCHEMA_CODES and check_name.startswith("syntax_"):
                path = check_name.removeprefix("syntax_").strip()
                if path and path not in blocker_targets:
                    blocker_targets.append(path)
            blockers.append(
                Blocker(
                    code=check_code or check_name.lower().replace(" ", "_"),
                    message=detail,
                    blocking=True,
                    repairability=check_repairability,
                    blocker_class=check_class,
                    source=f"verification:{check_name}",
                    targets=tuple(blocker_targets),
                    metadata={
                        "original_reason_code": raw_code,
                        **(
                            _schema_diagnostics(detail)
                            if check_code in _SCHEMA_CODES
                            else {}
                        ),
                    },
                )
            )
        return tuple(blockers)
    return (
        Blocker(
            code=code or "verification_failed",
            message=feedback or "Verification failed without a structured explanation.",
            blocking=True,
            repairability=repairability,
            blocker_class=blocker_class,
            targets=targets,
            metadata={"original_reason_code": raw_code},
        ),
    )


def select_handler(blockers: tuple[Blocker, ...]) -> CorrectionHandler:
    codes = {blocker.code for blocker in blockers}
    classes = {blocker.blocker_class for blocker in blockers}
    repairabilities = {blocker.repairability for blocker in blockers}
    if Repairability.TERMINAL in repairabilities:
        return CorrectionHandler.NONE
    if Repairability.HUMAN_REQUIRED in repairabilities:
        return CorrectionHandler.HUMAN_REVIEW
    if BlockerClass.VERIFIER_FAILURE in classes:
        return CorrectionHandler.RETRY_VERIFICATION
    if BlockerClass.ARTIFACT_SCHEMA in classes:
        return CorrectionHandler.SCHEMA_REPAIR
    if BlockerClass.ARTIFACT_CONTRACT in classes:
        return CorrectionHandler.CONTRACT_REPAIR
    if BlockerClass.RESOURCE_EXHAUSTION in classes:
        return CorrectionHandler.CHECKPOINT_CONTINUE
    if BlockerClass.ARTIFACT_WRITE_POLICY in classes:
        return CorrectionHandler.OUTPUT_REROUTE
    if BlockerClass.ARTIFACT_MISSING in classes:
        return CorrectionHandler.PLACEHOLDER_PREPASS
    if "artifact_confirmation_required" in codes:
        return CorrectionHandler.CONTEXT_REFRESH
    if classes & {BlockerClass.SOURCE_UNAVAILABLE, BlockerClass.TOOL_FAILURE}:
        return CorrectionHandler.SOURCE_FALLBACK
    if classes & {BlockerClass.CONTRADICTION, BlockerClass.EVIDENCE_GAP}:
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
    diagnostics = [
        {
            "target": target,
            **{
                key: value
                for key, value in blocker.metadata.items()
                if key in {"row_number", "actual_columns", "expected_columns"}
            },
        }
        for blocker in blockers
        for target in (blocker.targets or ("",))
        if any(
            key in blocker.metadata
            for key in {"row_number", "actual_columns", "expected_columns"}
        )
    ]
    action_type = {
        CorrectionHandler.RETRY_VERIFICATION: "rerun_verifier",
        CorrectionHandler.SCHEMA_REPAIR: "repair_structured_output_schema",
        CorrectionHandler.CONTRACT_REPAIR: "repair_structured_output_contract",
        CorrectionHandler.OUTPUT_REROUTE: "reroute_output_to_allowed_path",
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
                "diagnostics": diagnostics,
                "guardrails": (
                    [
                        "edit the existing structured output in place",
                        "preserve the declared header and valid row values",
                        "repair delimiter, quoting, escaping, or field-count errors only",
                        "do not repeat research or create alternate output files",
                        "rerun deterministic schema verification after editing",
                    ]
                    if handler == CorrectionHandler.SCHEMA_REPAIR
                    else [
                        "edit only verifier-named fields, rows, or sections",
                        "preserve validated content and evidence",
                        "do not repeat broad discovery or research",
                        "rerun only the failed contract checks",
                    ]
                    if handler == CorrectionHandler.CONTRACT_REPAIR
                    else [
                        "use only verifier-declared canonical or staging paths",
                        "patch existing deliverables instead of creating variants",
                        "preserve validated content and artifact seals",
                        "finish immediately after the required output is accepted",
                    ]
                    if handler == CorrectionHandler.OUTPUT_REROUTE
                    else [
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
