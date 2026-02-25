"""Signal analytics for options flow and put/call history."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from loom.research.finance import clamp, safe_float


def _to_ratio(row: dict[str, Any]) -> float | None:
    ratio = safe_float(row.get("put_call_ratio"))
    if ratio is not None and ratio > 0:
        return ratio
    put_v = safe_float(row.get("put_volume"))
    call_v = safe_float(row.get("call_volume"))
    if put_v is None or call_v in (None, 0):
        return None
    return put_v / call_v


def _to_total_volume(row: dict[str, Any]) -> float | None:
    total = safe_float(row.get("total_volume"))
    if total is not None and total >= 0:
        return total
    put_v = safe_float(row.get("put_volume"))
    call_v = safe_float(row.get("call_volume"))
    if put_v is None and call_v is None:
        return None
    return max(0.0, float(put_v or 0.0) + float(call_v or 0.0))


def _label(score: float) -> str:
    if score >= 0.3:
        return "bullish"
    if score <= -0.3:
        return "bearish"
    return "neutral"


def score_options_flow(rows: list[dict[str, Any]], *, lookback: int = 20) -> dict[str, Any]:
    """Convert options flow rows into a normalized directional score.

    Positive score means relatively bullish flow (low put/call vs baseline);
    negative score means relatively bearish flow (high put/call vs baseline).
    """
    lookback = max(5, min(252, int(lookback)))
    ratios = [r for r in (_to_ratio(row) for row in rows) if r is not None]
    if len(ratios) < 2:
        return {
            "flow_score": 0.0,
            "sentiment_label": "neutral",
            "latest_put_call_ratio": None,
            "z_score": 0.0,
            "confidence": 0.2,
            "warnings": ["Insufficient put/call history for robust scoring."],
        }

    latest = ratios[-1]
    baseline = ratios[-(lookback + 1) : -1] if len(ratios) > 2 else ratios[:-1]
    if not baseline:
        baseline = ratios[:-1]

    baseline_mean = mean(baseline)
    baseline_std = pstdev(baseline) if len(baseline) > 1 else 0.0
    if baseline_std > 1e-8:
        z_score = (latest - baseline_mean) / baseline_std
    else:
        z_score = (latest - baseline_mean) / max(1e-8, abs(baseline_mean))

    flow_score = clamp(-(z_score / 3.0), -1.0, 1.0)
    confidence = clamp(0.3 + 0.02 * min(len(ratios), 30), 0.25, 0.9)
    warnings: list[str] = []
    if len(ratios) < lookback:
        warnings.append("History shorter than lookback window; score is less stable.")

    return {
        "flow_score": flow_score,
        "sentiment_label": _label(flow_score),
        "latest_put_call_ratio": latest,
        "z_score": z_score,
        "baseline_ratio_mean": baseline_mean,
        "baseline_ratio_std": baseline_std,
        "history_count": len(ratios),
        "lookback": lookback,
        "confidence": confidence,
        "warnings": warnings,
    }


def detect_unusual_options_flow(
    rows: list[dict[str, Any]],
    *,
    lookback: int = 20,
    z_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """Flag unusually large put/call or total-volume moves."""
    lookback = max(5, min(252, int(lookback)))
    z_threshold = max(0.5, min(10.0, float(z_threshold)))
    if len(rows) < lookback + 2:
        return []

    events: list[dict[str, Any]] = []
    ratio_series = [_to_ratio(row) for row in rows]
    volume_series = [_to_total_volume(row) for row in rows]

    for i in range(lookback, len(rows)):
        row = rows[i]
        date_text = str(row.get("date", "")).strip()
        symbol = str(row.get("symbol", "")).strip().upper() or None

        ratio_value = ratio_series[i]
        if ratio_value is not None:
            baseline = [
                r for r in ratio_series[max(0, i - lookback) : i] if r is not None
            ]
            if len(baseline) >= 3:
                mu = mean(baseline)
                sigma = pstdev(baseline)
                if sigma > 1e-8:
                    z = (ratio_value - mu) / sigma
                    if abs(z) >= z_threshold:
                        events.append(
                            {
                                "date": date_text,
                                "symbol": symbol,
                                "signal": "put_call_ratio",
                                "value": ratio_value,
                                "baseline_mean": mu,
                                "z_score": z,
                                "direction": "bearish" if z > 0 else "bullish",
                            }
                        )

        volume_value = volume_series[i]
        if volume_value is not None:
            baseline_vol = [
                v for v in volume_series[max(0, i - lookback) : i] if v is not None
            ]
            if len(baseline_vol) >= 3:
                mu_v = mean(baseline_vol)
                sigma_v = pstdev(baseline_vol)
                if sigma_v > 1e-8:
                    z_v = (volume_value - mu_v) / sigma_v
                    if z_v >= z_threshold:
                        events.append(
                            {
                                "date": date_text,
                                "symbol": symbol,
                                "signal": "total_volume",
                                "value": volume_value,
                                "baseline_mean": mu_v,
                                "z_score": z_v,
                                "direction": "high_activity",
                            }
                        )

    return events


__all__ = ["detect_unusual_options_flow", "score_options_flow"]
