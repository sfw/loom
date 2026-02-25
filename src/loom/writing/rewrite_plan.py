"""Rewrite guidance synthesis from deterministic writing issues."""

from __future__ import annotations

from typing import Any

from loom.writing.models import HumanizationIssue

_ISSUE_ACTIONS: dict[str, str] = {
    "repetition_ngram": (
        "Replace repeated phrase patterns with specific alternatives and vary "
        "sentence constructions across adjacent lines."
    ),
    "duplicate_sentences": (
        "Merge duplicate points into one stronger sentence, then use the freed "
        "space for a concrete detail or example."
    ),
    "monotone_openings": (
        "Vary sentence openings: alternate between subject-led, action-led, and "
        "context-led constructions."
    ),
    "low_lexical_diversity": (
        "Swap generic repeated words for more precise synonyms that match the "
        "intended audience and domain."
    ),
    "flat_sentence_rhythm": (
        "Introduce rhythm contrast by mixing short, medium, and long sentences "
        "within each paragraph."
    ),
    "long_sentences": (
        "Split long compound sentences into clearer units and keep one core idea "
        "per sentence."
    ),
    "filler_phrases": (
        "Remove filler lead-ins and replace them with direct claims supported by "
        "specific facts."
    ),
    "low_specificity": (
        "Add concrete details (numbers, named entities, dates, or measurable "
        "outcomes) to ground abstract claims."
    ),
    "passive_voice": (
        "Convert passive constructions to active voice where possible to improve "
        "clarity and ownership."
    ),
    "weak_transitions": (
        "Use varied transition phrases to connect ideas and signal progression "
        "between paragraphs."
    ),
    "topic_drift": (
        "Reorder or trim paragraphs so each one advances a single thread tied to "
        "the main argument."
    ),
    "hedging": (
        "Reduce hedging words unless uncertainty is intentional and evidence-based."
    ),
}


def build_rewrite_actions(
    *,
    issues: list[HumanizationIssue],
    constraints: dict[str, Any] | None = None,
    max_actions: int = 8,
) -> tuple[list[str], list[str]]:
    """Build a concise prioritized rewrite plan."""
    normalized_max = max(1, min(20, int(max_actions)))
    actions: list[str] = []
    applied_constraints: list[str] = []

    for issue in sorted(issues, key=lambda item: (-item.severity, item.code)):
        action = _ISSUE_ACTIONS.get(issue.code)
        if not action:
            continue
        if action not in actions:
            actions.append(action)
        if len(actions) >= normalized_max:
            break

    constraints_dict = constraints if isinstance(constraints, dict) else {}

    preserve_terms = _as_string_list(constraints_dict.get("preserve_terms"))
    if preserve_terms:
        actions.append("Preserve these exact terms: " + ", ".join(preserve_terms[:8]) + ".")
        applied_constraints.append("preserve_terms")

    banned_phrases = _as_string_list(constraints_dict.get("banned_phrases"))
    if banned_phrases:
        actions.append("Avoid these phrases: " + ", ".join(banned_phrases[:8]) + ".")
        applied_constraints.append("banned_phrases")

    max_sentence_length = _to_int(constraints_dict.get("max_sentence_length"))
    if max_sentence_length is not None and max_sentence_length > 0:
        actions.append(f"Keep sentence length under {max_sentence_length} words.")
        applied_constraints.append("max_sentence_length")

    reading_level = str(constraints_dict.get("reading_level", "")).strip()
    if reading_level:
        actions.append(f"Target a {reading_level} reading level while keeping precision.")
        applied_constraints.append("reading_level")

    deduped: list[str] = []
    for action in actions:
        if action not in deduped:
            deduped.append(action)
    return deduped[:normalized_max], applied_constraints


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def _to_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
