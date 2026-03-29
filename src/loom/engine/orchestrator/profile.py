"""Verification profile resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass

from loom.engine.verification.policy import VerificationProfile, normalize_profile
from loom.state.task_state import Subtask, Task

_RESEARCH_HINTS = (
    "analysis",
    "citation",
    "compliance",
    "fact",
    "finance",
    "financial",
    "legal",
    "medical",
    "research",
    "report",
    "source",
    "summary",
)
_CODING_HINTS = (
    "build",
    "code",
    "compile",
    "fix",
    "lint",
    "refactor",
    "test",
)
_DATA_OPS_HINTS = (
    "csv",
    "data",
    "dataset",
    "etl",
    "migration",
    "schema",
    "sql",
    "table",
)


@dataclass(frozen=True)
class VerificationProfileResolution:
    """Resolved verification profile and confidence metadata."""

    profile: VerificationProfile
    confidence: float
    fallback_profile: VerificationProfile
    reason_codes: tuple[str, ...]


def _score_hints(text: str, hints: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(1 for hint in hints if hint in lowered)


def _tool_usage_text(tool_calls: list | None) -> str:
    if not isinstance(tool_calls, list):
        return ""
    names: list[str] = []
    for call in tool_calls:
        name = str(getattr(call, "tool", "") or "").strip().lower()
        if name:
            names.append(name)
    return " ".join(names)


def _process_policy_signals(process: object | None) -> tuple[int, int, list[str]]:
    """Return coding/data-ops score boosts derived from process policy."""
    if process is None:
        return 0, 0, []

    coding_boost = 0
    data_ops_boost = 0
    reasons: list[str] = []

    mode_resolver = getattr(process, "verifier_mode", None)
    mode = (
        str(mode_resolver() or "").strip().lower()
        if callable(mode_resolver)
        else str(
            getattr(getattr(process, "verification_policy", None), "mode", "") or "",
        )
        .strip()
        .lower()
    )
    if mode == "static_first":
        coding_boost += 1
        reasons.append("policy:static_first")

    tool_success_policy_resolver = getattr(process, "verifier_tool_success_policy", None)
    tool_success_policy = (
        str(tool_success_policy_resolver() or "").strip().lower()
        if callable(tool_success_policy_resolver)
        else ""
    )
    if tool_success_policy == "development_balanced":
        coding_boost += 2
        reasons.append("policy:development_balanced")

    optional_capabilities_resolver = getattr(process, "verifier_optional_capabilities", None)
    optional_capabilities = (
        optional_capabilities_resolver()
        if callable(optional_capabilities_resolver)
        else []
    )
    if isinstance(optional_capabilities, list):
        normalized = {
            str(item or "").strip().lower()
            for item in optional_capabilities
            if str(item or "").strip()
        }
        if normalized.intersection({"browser_runtime", "service_runtime"}):
            coding_boost += 1
            reasons.append("policy:dev_optional_capabilities")

    required_capabilities_resolver = getattr(process, "verifier_required_capabilities", None)
    required_capabilities = (
        required_capabilities_resolver()
        if callable(required_capabilities_resolver)
        else []
    )
    if isinstance(required_capabilities, list):
        normalized_required = {
            str(item or "").strip().lower()
            for item in required_capabilities
            if str(item or "").strip()
        }
        if normalized_required.intersection({"browser_runtime", "service_runtime"}):
            coding_boost += 1
            reasons.append("policy:dev_required_capabilities")

    tags = getattr(process, "tags", [])
    if isinstance(tags, list):
        normalized_tags = {
            str(tag or "").strip().lower()
            for tag in tags
            if str(tag or "").strip()
        }
        if normalized_tags.intersection({"data", "data_ops", "migration", "sql"}):
            data_ops_boost += 1
            reasons.append("tags:data_ops")

    return coding_boost, data_ops_boost, reasons


def resolve_verification_profile(
    *,
    task: Task | None,
    subtask: Subtask,
    process: object | None = None,
    tool_calls: list | None = None,
) -> VerificationProfileResolution:
    """Infer verification profile from process/task/subtask context."""
    source_bits: list[str] = [
        str(getattr(subtask, "description", "") or ""),
        str(getattr(subtask, "acceptance_criteria", "") or ""),
    ]
    if task is not None:
        source_bits.append(str(getattr(task, "goal", "") or ""))
    if hasattr(subtask, "phase_id"):
        source_bits.append(str(getattr(subtask, "phase_id", "") or ""))
    tags = []
    if process is not None:
        raw_tags = getattr(process, "tags", [])
        if isinstance(raw_tags, list):
            tags = [str(tag or "").strip().lower() for tag in raw_tags if str(tag or "").strip()]
    metadata_tags: list[str] = []
    if task is not None and isinstance(getattr(task, "metadata", None), dict):
        raw = task.metadata.get("tags", [])
        if isinstance(raw, list):
            metadata_tags = [
                str(item or "").strip().lower()
                for item in raw
                if str(item or "").strip()
            ]
    source_bits.extend(tags)
    source_bits.extend(metadata_tags)
    source_bits.append(_tool_usage_text(tool_calls))
    source = " ".join(bit for bit in source_bits if bit).strip().lower()

    research_score = _score_hints(source, _RESEARCH_HINTS)
    coding_score = _score_hints(source, _CODING_HINTS)
    data_ops_score = _score_hints(source, _DATA_OPS_HINTS)
    if subtask.is_synthesis:
        research_score += 1
    if "fact_checker" in source:
        research_score += 2
    if any(token in source for token in ("pytest", "ruff", "mypy", "unit test")):
        coding_score += 2
    if any(token in source for token in ("sql", "table", "query", "warehouse")):
        data_ops_score += 2
    policy_coding_boost, policy_data_ops_boost, policy_reasons = _process_policy_signals(
        process,
    )
    coding_score += policy_coding_boost
    data_ops_score += policy_data_ops_boost

    scores = {
        "research": research_score,
        "coding": coding_score,
        "data_ops": data_ops_score,
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ordered[0]
    second_score = ordered[1][1]
    reason_codes = tuple(
        f"{name}:{score}"
        for name, score in scores.items()
        if score > 0
    ) or ("profile_sparse_signal",)
    if policy_reasons:
        reason_codes = tuple(list(reason_codes) + policy_reasons)

    if top_score <= 0:
        return VerificationProfileResolution(
            profile="hybrid",
            confidence=0.0,
            fallback_profile="hybrid",
            reason_codes=reason_codes,
        )

    margin = max(0, top_score - second_score)
    confidence = max(0.0, min(1.0, 0.45 + (0.15 * margin)))
    if margin <= 0 or confidence < 0.6:
        return VerificationProfileResolution(
            profile="hybrid",
            confidence=confidence,
            fallback_profile="hybrid",
            reason_codes=reason_codes,
        )
    return VerificationProfileResolution(
        profile=normalize_profile(top_name),
        confidence=confidence,
        fallback_profile="hybrid",
        reason_codes=reason_codes,
    )
