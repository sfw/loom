"""Deterministic writing metrics used by the humanization tool."""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean
from typing import Any

from loom.writing.models import HumanizationIssue, HumanizationReport, WritingMetricSet
from loom.writing.style_profiles import mode_weights, normalize_mode, style_targets

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+[a-z]+(?:ed|en)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
}

_FILLER_PHRASES = (
    "it is important to note",
    "in conclusion",
    "in summary",
    "very important",
    "needless to say",
    "as previously mentioned",
    "at the end of the day",
    "it should be noted",
)

_HEDGE_WORDS = {
    "maybe",
    "perhaps",
    "possibly",
    "generally",
    "usually",
    "somewhat",
    "relatively",
    "arguably",
    "appears",
    "seems",
    "likely",
    "might",
    "could",
}

_TRANSITION_MARKERS = (
    "however",
    "meanwhile",
    "therefore",
    "for example",
    "for instance",
    "in contrast",
    "on the other hand",
    "next",
    "finally",
    "because",
    "so",
    "then",
)


def analyze_text(
    content: str,
    *,
    mode: str = "custom",
    audience: str = "",
    voice_profile: dict[str, Any] | None = None,
    target_score: float | None = None,
) -> HumanizationReport:
    """Return a deterministic report for a writing sample."""
    normalized_mode = normalize_mode(mode)
    targets = style_targets(normalized_mode, voice_profile=voice_profile)

    text = str(content or "").strip()
    paragraphs = _split_paragraphs(text)
    sentences = _split_sentences(text)
    words = _tokenize(text)

    word_count = len(words)
    sentence_count = max(1, len(sentences))
    paragraph_count = max(1, len(paragraphs))

    sentence_lengths = [len(_tokenize(sentence)) for sentence in sentences] or [word_count]
    avg_sentence_length = float(mean(sentence_lengths)) if sentence_lengths else 0.0
    sentence_stddev = _stddev(sentence_lengths)

    repeated_ngram_ratio = _repeated_ngram_ratio(words, n=3)
    near_duplicate_sentence_ratio = _near_duplicate_sentence_ratio(sentences)
    opening_similarity_ratio = _opening_similarity_ratio(sentences)
    lexical_diversity = _lexical_diversity(words)
    repeated_word_concentration = _repeated_word_concentration(words)
    punctuation_variety = _punctuation_variety(text)
    filler_phrase_density = _filler_density(text, word_count=word_count)
    detail_density = _detail_density(text, words)
    passive_voice_ratio = _passive_voice_ratio(text, sentence_count=sentence_count)
    hedge_density = _hedge_density(words)
    transition_variety = _transition_variety(text, paragraph_count=paragraph_count)
    paragraph_drift = _paragraph_drift(paragraphs)

    metrics = WritingMetricSet(
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        avg_sentence_length=avg_sentence_length,
        sentence_length_stddev=sentence_stddev,
        repeated_ngram_ratio=repeated_ngram_ratio,
        near_duplicate_sentence_ratio=near_duplicate_sentence_ratio,
        opening_similarity_ratio=opening_similarity_ratio,
        lexical_diversity=lexical_diversity,
        repeated_word_concentration=repeated_word_concentration,
        punctuation_variety=punctuation_variety,
        filler_phrase_density=filler_phrase_density,
        detail_density=detail_density,
        passive_voice_ratio=passive_voice_ratio,
        hedge_density=hedge_density,
        transition_variety=transition_variety,
        paragraph_drift=paragraph_drift,
    )

    sub_scores = _compute_sub_scores(metrics, targets=targets)
    weights = mode_weights(normalized_mode)
    humanization_score = _clamp(
        sum(sub_scores[key] * weights.get(key, 0.0) for key in sub_scores.keys()),
        0.0,
        100.0,
    )
    issues = _derive_issues(metrics)
    passes_target = (
        humanization_score >= target_score if target_score is not None else None
    )

    return HumanizationReport(
        mode=normalized_mode,
        audience=str(audience or "").strip(),
        humanization_score=round(humanization_score, 2),
        sub_scores={key: round(value, 2) for key, value in sub_scores.items()},
        metrics=metrics,
        issues=issues,
        target_score=target_score,
        passes_target=passes_target,
    )


def sub_score_delta(
    current: HumanizationReport,
    baseline: HumanizationReport,
) -> dict[str, float]:
    """Return sub-score deltas between current and baseline reports."""
    keys = sorted(set(current.sub_scores) | set(baseline.sub_scores))
    return {
        key: round(
            float(current.sub_scores.get(key, 0.0))
            - float(baseline.sub_scores.get(key, 0.0)),
            2,
        )
        for key in keys
    }


def _split_paragraphs(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text or "") if chunk.strip()]
    if not chunks and text.strip():
        return [text.strip()]
    return chunks


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    sentences = [part.strip() for part in _SENTENCE_RE.split(normalized) if part.strip()]
    if not sentences:
        return [normalized]
    return sentences


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def _stddev(values: list[int]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = float(mean(values))
    variance = sum((float(value) - avg) ** 2 for value in values) / float(len(values))
    return math.sqrt(max(variance, 0.0))


def _repeated_ngram_ratio(tokens: list[str], *, n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [" ".join(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeats = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeats) / float(len(grams))


def _near_duplicate_sentence_ratio(sentences: list[str]) -> float:
    if len(sentences) <= 1:
        return 0.0
    token_sets = [set(_tokenize(sentence)) for sentence in sentences]
    flagged: set[int] = set()
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            if _jaccard(token_sets[i], token_sets[j]) >= 0.84:
                flagged.add(i)
                flagged.add(j)
    return float(len(flagged)) / float(len(sentences))


def _opening_similarity_ratio(sentences: list[str]) -> float:
    if len(sentences) <= 1:
        return 0.0
    starts: list[str] = []
    for sentence in sentences:
        words = _tokenize(sentence)
        if not words:
            continue
        starts.append(" ".join(words[:2]))
    if not starts:
        return 0.0
    counts = Counter(starts)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return float(repeated) / float(len(starts))


def _lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return float(len(set(tokens))) / float(len(tokens))


def _repeated_word_concentration(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    meaningful = [token for token in tokens if token not in _STOPWORDS and len(token) > 2]
    sample = meaningful or tokens
    counts = Counter(sample)
    top_count = max(counts.values())
    return float(top_count) / float(len(sample))


def _punctuation_variety(text: str) -> float:
    symbols = {char for char in text if char in {",", ";", ":", "!", "?", "-", "(", ")"}}
    return min(1.0, float(len(symbols)) / 8.0)


def _filler_density(text: str, *, word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    lowered = (text or "").lower()
    count = sum(lowered.count(phrase) for phrase in _FILLER_PHRASES)
    return float(count) / float(word_count)


def _detail_density(text: str, tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    numeric = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    long_words = sum(1 for token in tokens if len(token) >= 8)
    proper_nouns = len(_PROPER_NOUN_RE.findall(text or ""))
    points = (numeric * 1.2) + (proper_nouns * 0.9) + (long_words * 0.2)
    return min(1.0, points / float(max(len(tokens), 1)))


def _passive_voice_ratio(text: str, *, sentence_count: int) -> float:
    if sentence_count <= 0:
        return 0.0
    matches = len(_PASSIVE_RE.findall(text or ""))
    return float(matches) / float(sentence_count)


def _hedge_density(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    hedges = sum(1 for token in tokens if token in _HEDGE_WORDS)
    return float(hedges) / float(len(tokens))


def _transition_variety(text: str, *, paragraph_count: int) -> float:
    lowered = (text or "").lower()
    used = {marker for marker in _TRANSITION_MARKERS if marker in lowered}
    denominator = min(8, max(2, paragraph_count + 1))
    return min(1.0, float(len(used)) / float(denominator))


def _paragraph_drift(paragraphs: list[str]) -> float:
    if len(paragraphs) <= 1:
        return 0.0
    token_sets = [
        {
            token
            for token in _tokenize(paragraph)
            if token not in _STOPWORDS and len(token) > 2
        }
        for paragraph in paragraphs
    ]
    similarities: list[float] = []
    for idx in range(len(token_sets) - 1):
        similarities.append(_jaccard(token_sets[idx], token_sets[idx + 1]))
    if not similarities:
        return 0.0
    return 1.0 - float(mean(similarities))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    if union <= 0:
        return 0.0
    return float(len(a & b)) / float(union)


def _compute_sub_scores(
    metrics: WritingMetricSet,
    targets: dict[str, float],
) -> dict[str, float]:
    repetition_score = _clamp(
        100.0
        - (metrics.repeated_ngram_ratio * 280.0)
        - (metrics.near_duplicate_sentence_ratio * 260.0)
        - (metrics.opening_similarity_ratio * 220.0)
        - (max(0.0, metrics.repeated_word_concentration - 0.18) * 220.0),
        0.0,
        100.0,
    )

    target_avg = targets.get("target_avg_sentence_length", 16.0)
    target_std = targets.get("target_sentence_stddev", 6.8)
    rhythm_score = _clamp(
        80.0
        - (abs(metrics.avg_sentence_length - target_avg) * 2.6)
        - (abs(metrics.sentence_length_stddev - target_std) * 2.1)
        - (max(0.0, 5.0 - metrics.sentence_length_stddev) * 3.0)
        + (metrics.punctuation_variety * 18.0),
        0.0,
        100.0,
    )

    lexical_target = targets.get("target_lexical_diversity", 0.45)
    lexical_score = _clamp(
        90.0
        - (abs(metrics.lexical_diversity - lexical_target) * 170.0)
        - (max(0.0, metrics.repeated_word_concentration - 0.16) * 260.0),
        0.0,
        100.0,
    )

    concreteness_score = _clamp(
        56.0
        + min(34.0, metrics.detail_density * 240.0)
        - (metrics.filler_phrase_density * 500.0),
        0.0,
        100.0,
    )

    clarity_score = _clamp(
        95.0
        - (max(0.0, metrics.avg_sentence_length - 24.0) * 3.2)
        - (max(0.0, 9.0 - metrics.avg_sentence_length) * 3.2)
        - (metrics.passive_voice_ratio * 120.0)
        - (metrics.hedge_density * 420.0),
        0.0,
        100.0,
    )

    coherence_base = 72.0 + (metrics.transition_variety * 30.0)
    if metrics.paragraph_count > 1:
        coherence_base -= max(0.0, metrics.paragraph_drift - 0.45) * 120.0
    coherence_score = _clamp(coherence_base, 0.0, 100.0)

    return {
        "repetition": repetition_score,
        "rhythm": rhythm_score,
        "lexical": lexical_score,
        "concreteness": concreteness_score,
        "clarity": clarity_score,
        "coherence": coherence_score,
    }


def _derive_issues(metrics: WritingMetricSet) -> list[HumanizationIssue]:
    issues: list[HumanizationIssue] = []

    if metrics.repeated_ngram_ratio >= 0.06:
        severity = 5 if metrics.repeated_ngram_ratio >= 0.12 else 4
        issues.append(
            HumanizationIssue(
                code="repetition_ngram",
                title="High phrase repetition",
                detail="Repeated 3-gram patterns make the draft sound templated.",
                severity=severity,
                evidence=f"repeated_ngram_ratio={metrics.repeated_ngram_ratio:.3f}",
            )
        )

    if metrics.near_duplicate_sentence_ratio >= 0.10:
        issues.append(
            HumanizationIssue(
                code="duplicate_sentences",
                title="Near-duplicate sentences",
                detail="Multiple sentences communicate the same idea with minimal change.",
                severity=4,
                evidence=(
                    "near_duplicate_sentence_ratio="
                    f"{metrics.near_duplicate_sentence_ratio:.3f}"
                ),
            )
        )

    if metrics.opening_similarity_ratio >= 0.28:
        issues.append(
            HumanizationIssue(
                code="monotone_openings",
                title="Monotone sentence openings",
                detail="Sentence starts repeat a similar pattern, flattening cadence.",
                severity=3,
                evidence=f"opening_similarity_ratio={metrics.opening_similarity_ratio:.3f}",
            )
        )

    if metrics.lexical_diversity <= 0.30:
        issues.append(
            HumanizationIssue(
                code="low_lexical_diversity",
                title="Low lexical diversity",
                detail="Vocabulary range is narrow, reducing natural voice variation.",
                severity=3,
                evidence=f"lexical_diversity={metrics.lexical_diversity:.3f}",
            )
        )

    if metrics.sentence_length_stddev <= 4.2 and metrics.sentence_count >= 4:
        issues.append(
            HumanizationIssue(
                code="flat_sentence_rhythm",
                title="Flat sentence rhythm",
                detail="Sentence lengths are too uniform; cadence sounds mechanical.",
                severity=3,
                evidence=f"sentence_length_stddev={metrics.sentence_length_stddev:.2f}",
            )
        )

    if metrics.avg_sentence_length >= 26.0:
        issues.append(
            HumanizationIssue(
                code="long_sentences",
                title="Overlong sentence structure",
                detail="Average sentence length is high and likely hard to scan.",
                severity=3,
                evidence=f"avg_sentence_length={metrics.avg_sentence_length:.2f}",
            )
        )

    if metrics.filler_phrase_density >= 0.012:
        issues.append(
            HumanizationIssue(
                code="filler_phrases",
                title="Filler phrase overuse",
                detail="Generic filler phrases reduce specificity and authenticity.",
                severity=3,
                evidence=f"filler_phrase_density={metrics.filler_phrase_density:.3f}",
            )
        )

    if metrics.detail_density <= 0.05:
        issues.append(
            HumanizationIssue(
                code="low_specificity",
                title="Low specificity",
                detail="The draft lacks concrete details (numbers, entities, precise terms).",
                severity=2,
                evidence=f"detail_density={metrics.detail_density:.3f}",
            )
        )

    if metrics.passive_voice_ratio >= 0.18:
        issues.append(
            HumanizationIssue(
                code="passive_voice",
                title="Frequent passive voice",
                detail="Passive constructions weaken directness and natural flow.",
                severity=2,
                evidence=f"passive_voice_ratio={metrics.passive_voice_ratio:.3f}",
            )
        )

    if metrics.paragraph_count > 2 and metrics.transition_variety <= 0.15:
        issues.append(
            HumanizationIssue(
                code="weak_transitions",
                title="Weak transitions",
                detail="Paragraph and idea transitions are sparse or repetitive.",
                severity=2,
                evidence=f"transition_variety={metrics.transition_variety:.3f}",
            )
        )

    if metrics.paragraph_count > 2 and metrics.paragraph_drift >= 0.75:
        issues.append(
            HumanizationIssue(
                code="topic_drift",
                title="Paragraph topic drift",
                detail="Adjacent paragraphs have low topical overlap, hurting coherence.",
                severity=2,
                evidence=f"paragraph_drift={metrics.paragraph_drift:.3f}",
            )
        )

    if metrics.hedge_density >= 0.02:
        issues.append(
            HumanizationIssue(
                code="hedging",
                title="Hedging language density",
                detail="Excessive hedging can make the writing uncertain or indirect.",
                severity=1,
                evidence=f"hedge_density={metrics.hedge_density:.3f}",
            )
        )

    issues.sort(key=lambda issue: (-issue.severity, issue.code))
    return issues


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
