"""Deterministic earnings-surprise model helpers."""

from __future__ import annotations

import math
from typing import Any

from loom.research.earnings_features import build_earnings_features
from loom.research.finance import clamp, safe_float


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / float(len(values)) for _ in values]
    return [v / total for v in exps]


def predict_earnings_surprise(feature_payload: dict[str, Any]) -> dict[str, Any]:
    """Predict EPS surprise using a transparent weighted-factor baseline model."""
    features = feature_payload.get("model_inputs", {})
    if not isinstance(features, dict):
        raise ValueError("feature_payload.model_inputs must be an object")

    latest_eps = safe_float(feature_payload.get("latest_eps"))
    if latest_eps is None:
        latest_eps = 1.0

    eps_trend = safe_float(features.get("eps_trend")) or 0.0
    margin = safe_float(features.get("margin")) or 0.0
    momentum = safe_float(features.get("momentum_20d")) or 0.0
    volatility = safe_float(features.get("volatility_20d")) or 0.0
    sentiment = safe_float(features.get("sentiment_score")) or 0.0
    options_flow = safe_float(features.get("options_flow_score")) or 0.0
    short_squeeze = safe_float(features.get("short_squeeze_potential")) or 0.0
    insider = safe_float(features.get("insider_score")) or 0.0
    guidance = safe_float(features.get("guidance_delta")) or 0.0
    revenue_scale = safe_float(features.get("revenue_scale")) or 0.0

    expected_surprise_pct = (
        0.01
        + 0.05 * eps_trend
        + 0.03 * margin
        + 0.02 * momentum
        - 0.015 * volatility
        + 0.03 * sentiment
        + 0.025 * options_flow
        + 0.015 * short_squeeze
        + 0.02 * insider
        + 0.03 * guidance
        + 0.004 * revenue_scale
    )
    expected_surprise_pct = clamp(expected_surprise_pct, -0.5, 0.5)

    predicted_eps = latest_eps * (1.0 + expected_surprise_pct)
    coverage = clamp(safe_float(feature_payload.get("feature_coverage")) or 0.5, 0.0, 1.0)

    uncertainty = clamp(0.05 + 0.09 * (1.0 - coverage) + 0.08 * volatility, 0.03, 0.3)
    low_eps = latest_eps * (1.0 + expected_surprise_pct - uncertainty)
    high_eps = latest_eps * (1.0 + expected_surprise_pct + uncertainty)
    if low_eps > high_eps:
        low_eps, high_eps = high_eps, low_eps

    beat_score = expected_surprise_pct / 0.03
    meet_score = -abs(expected_surprise_pct) / 0.025
    miss_score = -beat_score
    beat_prob, meet_prob, miss_prob = _softmax([beat_score, meet_score, miss_score])

    if uncertainty >= 0.14:
        vol_bucket = "high"
    elif uncertainty >= 0.08:
        vol_bucket = "medium"
    else:
        vol_bucket = "low"

    confidence = clamp(0.32 + 0.55 * coverage - 0.2 * volatility, 0.2, 0.92)
    return {
        "expected_surprise_pct": expected_surprise_pct,
        "predicted_eps": predicted_eps,
        "predicted_eps_range": [low_eps, high_eps],
        "beat_prob": beat_prob,
        "meet_prob": meet_prob,
        "miss_prob": miss_prob,
        "expected_post_earnings_volatility": vol_bucket,
        "uncertainty_band": uncertainty,
        "confidence": confidence,
    }


def compare_prediction_to_consensus(
    *,
    prediction: dict[str, Any],
    consensus_eps: object,
    actual_eps: object | None = None,
) -> dict[str, Any]:
    predicted_eps = safe_float(prediction.get("predicted_eps"))
    if predicted_eps is None:
        raise ValueError("prediction.predicted_eps is required")

    consensus = safe_float(consensus_eps)
    if consensus is None or consensus == 0:
        raise ValueError("consensus_eps must be a non-zero number")

    implied_surprise_vs_consensus = (predicted_eps / consensus) - 1.0
    out = {
        "consensus_eps": consensus,
        "model_predicted_eps": predicted_eps,
        "model_minus_consensus": predicted_eps - consensus,
        "implied_surprise_vs_consensus": implied_surprise_vs_consensus,
    }

    realized = safe_float(actual_eps)
    if realized is not None:
        model_error = abs(predicted_eps - realized)
        consensus_error = abs(consensus - realized)
        out.update(
            {
                "actual_eps": realized,
                "model_error": model_error,
                "consensus_error": consensus_error,
                "winner": "model" if model_error < consensus_error else "consensus",
            }
        )

    return out


def backtest_earnings_model(history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not history_rows:
        raise ValueError("history rows are required")

    samples: list[dict[str, float]] = []
    for row in history_rows:
        if not isinstance(row, dict):
            continue
        actual = safe_float(row.get("actual_surprise_pct"))
        if actual is None:
            continue

        if isinstance(row.get("model_inputs"), dict):
            feature_payload = {
                "latest_eps": safe_float(row.get("latest_eps")) or 1.0,
                "model_inputs": row["model_inputs"],
                "feature_coverage": safe_float(row.get("feature_coverage")) or 0.8,
            }
        else:
            feature_payload = build_earnings_features(row)

        pred = predict_earnings_surprise(feature_payload)
        predicted = safe_float(pred.get("expected_surprise_pct"))
        if predicted is None:
            continue
        samples.append({"predicted": predicted, "actual": actual})

    if not samples:
        raise ValueError("history rows missing usable actual_surprise_pct data")

    abs_errors = [abs(s["predicted"] - s["actual"]) for s in samples]
    sq_errors = [(s["predicted"] - s["actual"]) ** 2 for s in samples]
    directional_hits = [
        1.0
        if (
            (s["predicted"] >= 0 and s["actual"] >= 0)
            or (s["predicted"] < 0 and s["actual"] < 0)
        )
        else 0.0
        for s in samples
    ]

    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))
    hit_rate = sum(directional_hits) / len(directional_hits)

    return {
        "sample_count": len(samples),
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": hit_rate,
        "confidence": clamp(0.3 + 0.5 * hit_rate - 0.8 * mae, 0.1, 0.9),
    }


__all__ = [
    "backtest_earnings_model",
    "compare_prediction_to_consensus",
    "predict_earnings_surprise",
]
