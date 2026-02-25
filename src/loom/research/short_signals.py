"""Signal analytics for short-interest and short-volume data."""

from __future__ import annotations

from typing import Any

from loom.research.finance import clamp, safe_float


def compute_short_pressure(
    *,
    short_interest_shares: object,
    float_shares: object | None = None,
    average_daily_volume: object | None = None,
    short_volume: object | None = None,
    total_volume: object | None = None,
) -> dict[str, Any]:
    """Estimate short-pressure conditions using available inputs.

    The score is 0..100, where higher means greater squeeze/short-pressure risk.
    """
    warnings: list[str] = []

    si = safe_float(short_interest_shares)
    if si is None or si < 0:
        raise ValueError("short_interest_shares must be a non-negative number")

    flt = safe_float(float_shares)
    adv = safe_float(average_daily_volume)
    sv = safe_float(short_volume)
    tv = safe_float(total_volume)

    si_ratio = (si / flt) if flt and flt > 0 else None
    if si_ratio is None:
        warnings.append("float_shares missing; short-interest ratio unavailable.")

    dtc = (si / adv) if adv and adv > 0 else None
    if dtc is None:
        warnings.append("average_daily_volume missing; days-to-cover unavailable.")

    short_volume_ratio = (sv / tv) if sv is not None and tv and tv > 0 else None
    if short_volume_ratio is None:
        warnings.append("short_volume/total_volume missing; daily short-flow ratio unavailable.")

    component_weights: list[tuple[float, float]] = []
    if si_ratio is not None:
        component_weights.append((clamp(si_ratio / 0.25, 0.0, 1.0), 0.45))
    if dtc is not None:
        component_weights.append((clamp(dtc / 8.0, 0.0, 1.0), 0.35))
    if short_volume_ratio is not None:
        component_weights.append((clamp((short_volume_ratio - 0.35) / 0.35, 0.0, 1.0), 0.2))

    if component_weights:
        numerator = sum(score * w for score, w in component_weights)
        denom = sum(w for _score, w in component_weights)
        score = 100.0 * (numerator / denom)
    else:
        score = 0.0

    score = clamp(score, 0.0, 100.0)
    if score >= 75:
        label = "elevated"
    elif score >= 45:
        label = "moderate"
    else:
        label = "low"

    confidence = clamp(0.2 + 0.2 * len(component_weights), 0.2, 0.9)
    return {
        "short_pressure_score": score,
        "pressure_label": label,
        "short_interest_ratio": si_ratio,
        "days_to_cover": dtc,
        "daily_short_volume_ratio": short_volume_ratio,
        "confidence": confidence,
        "warnings": warnings,
    }


def detect_squeeze_setup(
    *,
    short_pressure_score: object,
    price_momentum_20d: object | None = None,
    threshold: float = 70.0,
) -> dict[str, Any]:
    score = safe_float(short_pressure_score)
    if score is None:
        raise ValueError("short_pressure_score is required")
    score = clamp(score, 0.0, 100.0)

    threshold = clamp(float(threshold), 0.0, 100.0)
    momentum = safe_float(price_momentum_20d)
    warnings: list[str] = []

    pressure_hit = score >= threshold
    momentum_hit = momentum is not None and momentum > 0.0
    if momentum is None:
        warnings.append("price_momentum_20d not provided; setup confidence reduced.")

    setup = pressure_hit and (momentum_hit or momentum is None)
    confidence = 0.75 if pressure_hit and momentum_hit else (0.58 if pressure_hit else 0.35)
    return {
        "squeeze_setup": setup,
        "short_pressure_score": score,
        "threshold": threshold,
        "price_momentum_20d": momentum,
        "pressure_condition_met": pressure_hit,
        "momentum_condition_met": momentum_hit,
        "confidence": confidence,
        "warnings": warnings,
    }


__all__ = ["compute_short_pressure", "detect_squeeze_setup"]
