"""Style profiles and scoring weights for writing modes."""

from __future__ import annotations

from typing import Any

_VALID_MODES = {
    "creative_copy",
    "blog_post",
    "email",
    "report",
    "social_post",
    "custom",
}

_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "creative_copy": {
        "repetition": 0.20,
        "rhythm": 0.24,
        "lexical": 0.20,
        "concreteness": 0.14,
        "clarity": 0.12,
        "coherence": 0.10,
    },
    "blog_post": {
        "repetition": 0.18,
        "rhythm": 0.18,
        "lexical": 0.18,
        "concreteness": 0.16,
        "clarity": 0.18,
        "coherence": 0.12,
    },
    "email": {
        "repetition": 0.15,
        "rhythm": 0.12,
        "lexical": 0.12,
        "concreteness": 0.18,
        "clarity": 0.25,
        "coherence": 0.18,
    },
    "report": {
        "repetition": 0.20,
        "rhythm": 0.10,
        "lexical": 0.15,
        "concreteness": 0.15,
        "clarity": 0.25,
        "coherence": 0.15,
    },
    "social_post": {
        "repetition": 0.20,
        "rhythm": 0.22,
        "lexical": 0.18,
        "concreteness": 0.14,
        "clarity": 0.16,
        "coherence": 0.10,
    },
    "custom": {
        "repetition": 1 / 6,
        "rhythm": 1 / 6,
        "lexical": 1 / 6,
        "concreteness": 1 / 6,
        "clarity": 1 / 6,
        "coherence": 1 / 6,
    },
}

_MODE_TARGETS: dict[str, dict[str, float]] = {
    "creative_copy": {
        "target_avg_sentence_length": 14.5,
        "target_sentence_stddev": 8.0,
        "target_lexical_diversity": 0.50,
    },
    "blog_post": {
        "target_avg_sentence_length": 16.0,
        "target_sentence_stddev": 7.0,
        "target_lexical_diversity": 0.47,
    },
    "email": {
        "target_avg_sentence_length": 14.0,
        "target_sentence_stddev": 6.0,
        "target_lexical_diversity": 0.43,
    },
    "report": {
        "target_avg_sentence_length": 18.0,
        "target_sentence_stddev": 6.2,
        "target_lexical_diversity": 0.44,
    },
    "social_post": {
        "target_avg_sentence_length": 11.5,
        "target_sentence_stddev": 7.5,
        "target_lexical_diversity": 0.46,
    },
    "custom": {
        "target_avg_sentence_length": 16.0,
        "target_sentence_stddev": 6.8,
        "target_lexical_diversity": 0.45,
    },
}


def normalize_mode(mode: str) -> str:
    """Normalize and validate requested mode."""
    candidate = str(mode or "").strip().lower()
    if candidate not in _VALID_MODES:
        return "custom"
    return candidate


def mode_weights(mode: str) -> dict[str, float]:
    """Return scoring weights for a mode."""
    return dict(_MODE_WEIGHTS[normalize_mode(mode)])


def style_targets(
    mode: str,
    voice_profile: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return style targets for metric normalization."""
    targets = dict(_MODE_TARGETS[normalize_mode(mode)])
    profile = voice_profile if isinstance(voice_profile, dict) else {}

    rhythm = str(profile.get("sentence_rhythm", "")).strip().lower()
    if rhythm == "steady":
        targets["target_sentence_stddev"] = max(
            4.0,
            targets["target_sentence_stddev"] - 2.0,
        )
    elif rhythm in {"varied", "dynamic"}:
        targets["target_sentence_stddev"] += 1.8

    lexical_complexity = str(profile.get("lexical_complexity", "")).strip().lower()
    if lexical_complexity == "simple":
        targets["target_lexical_diversity"] = max(
            0.30,
            targets["target_lexical_diversity"] - 0.08,
        )
    elif lexical_complexity in {"rich", "advanced"}:
        targets["target_lexical_diversity"] = min(
            0.62,
            targets["target_lexical_diversity"] + 0.08,
        )

    formality = str(profile.get("formality", "")).strip().lower()
    if formality in {"high", "formal"}:
        targets["target_avg_sentence_length"] += 1.5
    elif formality in {"low", "casual"}:
        targets["target_avg_sentence_length"] = max(
            10.0,
            targets["target_avg_sentence_length"] - 1.5,
        )

    return targets
