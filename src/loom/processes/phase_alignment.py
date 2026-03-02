"""Helpers for mapping planner subtask labels to declared process phases."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = frozenset({
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "phase",
    "step",
    "subtask",
    "task",
    "the",
    "to",
    "with",
})


def _slug(value: object) -> str:
    """Normalize text into a lowercase alnum-hyphen slug."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return _NON_ALNUM_RE.sub("-", text).strip("-")


def _tokenize(value: object) -> set[str]:
    """Extract lightweight semantic tokens from free-form text."""
    text = str(value or "").lower()
    if not text:
        return set()

    tokens: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).strip()
        if len(token) < 2 or token in _STOPWORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if token and token not in _STOPWORDS:
            tokens.add(token)
    return tokens


def _ratio_overlap(lhs: set[str], rhs: set[str]) -> float:
    if not lhs or not rhs:
        return 0.0
    return float(len(lhs & rhs)) / float(len(lhs))


def _jaccard(lhs: set[str], rhs: set[str]) -> float:
    if not lhs and not rhs:
        return 0.0
    union = lhs | rhs
    if not union:
        return 0.0
    return float(len(lhs & rhs)) / float(len(union))


def _deliverable_tokens(
    phase_id: str,
    phase_deliverables: Mapping[str, Sequence[str]] | None,
) -> set[str]:
    if not isinstance(phase_deliverables, Mapping):
        return set()
    raw = phase_deliverables.get(phase_id, [])
    if not isinstance(raw, Sequence):
        return set()
    tokens: set[str] = set()
    for item in raw:
        path_text = str(item or "").strip()
        if not path_text:
            continue
        tokens |= _tokenize(path_text)
        stem = Path(path_text).stem
        if stem:
            tokens |= _tokenize(stem)
    return tokens


def match_phase_id_for_subtask(
    *,
    subtask_id: str,
    text: str,
    phase_ids: Sequence[str],
    phase_descriptions: Mapping[str, str] | None = None,
    phase_deliverables: Mapping[str, Sequence[str]] | None = None,
    min_score: float = 0.4,
    ambiguity_margin: float = 0.08,
) -> tuple[str, float]:
    """Return the best matching phase_id and confidence-like score."""
    ordered_phase_ids: list[str] = []
    seen: set[str] = set()
    for raw in phase_ids:
        phase_id = str(raw or "").strip()
        if not phase_id or phase_id in seen:
            continue
        seen.add(phase_id)
        ordered_phase_ids.append(phase_id)
    if not ordered_phase_ids:
        return "", 0.0

    clean_subtask_id = str(subtask_id or "").strip()
    if clean_subtask_id in seen:
        return clean_subtask_id, 1.0

    if len(ordered_phase_ids) == 1:
        return ordered_phase_ids[0], 0.45

    subtask_slug = _slug(clean_subtask_id)
    text_slug = _slug(text)
    subtask_tokens = _tokenize(clean_subtask_id)
    subtask_tokens |= _tokenize(text)

    scored: list[tuple[str, float]] = []
    for phase_id in ordered_phase_ids:
        phase_slug = _slug(phase_id)
        phase_id_tokens = _tokenize(phase_id)
        phase_desc_tokens = _tokenize(
            (phase_descriptions or {}).get(phase_id, "") if phase_descriptions else "",
        )
        phase_deliv_tokens = _deliverable_tokens(phase_id, phase_deliverables)
        phase_tokens = phase_id_tokens | phase_desc_tokens | phase_deliv_tokens

        id_bonus = 0.0
        if subtask_slug and phase_slug:
            if subtask_slug == phase_slug:
                id_bonus = 1.0
            elif subtask_slug.startswith(phase_slug) or phase_slug.startswith(subtask_slug):
                id_bonus = 0.72 if min(len(subtask_slug), len(phase_slug)) >= 4 else 0.45
            elif phase_slug in subtask_slug and len(phase_slug) >= 4:
                id_bonus = 0.62

        mention_bonus = 0.0
        if phase_slug and text_slug and phase_slug in text_slug:
            mention_bonus = 0.38

        primary_overlap = _ratio_overlap(phase_id_tokens, subtask_tokens)
        phase_overlap = _ratio_overlap(phase_tokens, subtask_tokens)
        jaccard = _jaccard(phase_tokens, subtask_tokens)
        score = (
            id_bonus
            + mention_bonus
            + (0.45 * primary_overlap)
            + (0.35 * phase_overlap)
            + (0.2 * jaccard)
        )
        scored.append((phase_id, score))

    if not scored:
        return "", 0.0

    scored.sort(key=lambda item: item[1], reverse=True)
    best_id, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0

    if best_score < float(min_score):
        return "", best_score
    if best_score < 0.75 and (best_score - second_score) < float(ambiguity_margin):
        return "", best_score
    return best_id, best_score


def infer_phase_id_for_subtask(
    *,
    subtask_id: str,
    text: str,
    phase_ids: Sequence[str],
    phase_descriptions: Mapping[str, str] | None = None,
    phase_deliverables: Mapping[str, Sequence[str]] | None = None,
    min_score: float = 0.4,
    ambiguity_margin: float = 0.08,
) -> str:
    """Return only the best matching phase_id when match confidence is sufficient."""
    phase_id, _score = match_phase_id_for_subtask(
        subtask_id=subtask_id,
        text=text,
        phase_ids=phase_ids,
        phase_descriptions=phase_descriptions,
        phase_deliverables=phase_deliverables,
        min_score=min_score,
        ambiguity_margin=ambiguity_margin,
    )
    return phase_id
