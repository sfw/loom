"""Structured models for humanized writing analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


@dataclass
class HumanizationIssue:
    """One actionable writing issue identified by analysis."""

    code: str
    title: str
    detail: str
    severity: int
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "severity": int(self.severity),
            "evidence": self.evidence,
        }


@dataclass
class WritingMetricSet:
    """Deterministic feature set used by the humanization scorer."""

    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    sentence_length_stddev: float
    repeated_ngram_ratio: float
    near_duplicate_sentence_ratio: float
    opening_similarity_ratio: float
    lexical_diversity: float
    repeated_word_concentration: float
    punctuation_variety: float
    filler_phrase_density: float
    detail_density: float
    passive_voice_ratio: float
    hedge_density: float
    transition_variety: float
    paragraph_drift: float

    def to_dict(self) -> dict:
        return {
            "word_count": int(self.word_count),
            "sentence_count": int(self.sentence_count),
            "paragraph_count": int(self.paragraph_count),
            "avg_sentence_length": _round_float(self.avg_sentence_length),
            "sentence_length_stddev": _round_float(self.sentence_length_stddev),
            "repeated_ngram_ratio": _round_float(self.repeated_ngram_ratio),
            "near_duplicate_sentence_ratio": _round_float(
                self.near_duplicate_sentence_ratio
            ),
            "opening_similarity_ratio": _round_float(self.opening_similarity_ratio),
            "lexical_diversity": _round_float(self.lexical_diversity),
            "repeated_word_concentration": _round_float(
                self.repeated_word_concentration
            ),
            "punctuation_variety": _round_float(self.punctuation_variety),
            "filler_phrase_density": _round_float(self.filler_phrase_density),
            "detail_density": _round_float(self.detail_density),
            "passive_voice_ratio": _round_float(self.passive_voice_ratio),
            "hedge_density": _round_float(self.hedge_density),
            "transition_variety": _round_float(self.transition_variety),
            "paragraph_drift": _round_float(self.paragraph_drift),
        }


@dataclass
class HumanizationReport:
    """Complete analysis report for one writing sample."""

    mode: str
    audience: str
    humanization_score: float
    sub_scores: dict[str, float]
    metrics: WritingMetricSet
    issues: list[HumanizationIssue] = field(default_factory=list)
    recommended_edits: list[str] = field(default_factory=list)
    constraints_applied: list[str] = field(default_factory=list)
    target_score: float | None = None
    passes_target: bool | None = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "audience": self.audience,
            "humanization_score": _round_float(self.humanization_score, 2),
            "sub_scores": {
                key: _round_float(value, 2) for key, value in self.sub_scores.items()
            },
            "metrics": self.metrics.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "recommended_edits": list(self.recommended_edits),
            "constraints_applied": list(self.constraints_applied),
            "target_score": (
                _round_float(self.target_score, 2)
                if self.target_score is not None
                else None
            ),
            "passes_target": self.passes_target,
        }
