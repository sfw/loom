"""Feature engineering helpers for earnings surprise prediction."""

from __future__ import annotations

from statistics import pstdev
from typing import Any

from loom.research.finance import clamp, cumulative_growth, safe_float


def _coerce_float_list(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for item in values:
        value = safe_float(item)
        if value is not None:
            out.append(float(value))
    return out


def _eps_trend(prior_eps: list[float]) -> float:
    if len(prior_eps) < 2:
        return 0.0
    base = prior_eps[0]
    latest = prior_eps[-1]
    if abs(base) < 1e-8:
        return 0.0
    return clamp((latest - base) / abs(base), -2.0, 2.0)


def build_earnings_features(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a model-ready feature vector from heterogeneous earnings inputs."""
    warnings: list[str] = []
    prior_eps = _coerce_float_list(payload.get("prior_eps"))
    price_returns = _coerce_float_list(payload.get("price_returns"))

    latest_eps = safe_float(payload.get("latest_eps"))
    if latest_eps is None and prior_eps:
        latest_eps = prior_eps[-1]
    if latest_eps is None:
        latest_eps = 1.0
        warnings.append("latest_eps missing; default baseline EPS=1.0 used.")

    revenue_ttm = safe_float(payload.get("revenue_ttm"))
    net_margin_ttm = safe_float(payload.get("net_margin_ttm"))

    sentiment_score = safe_float(payload.get("sentiment_score"))
    options_flow_score = safe_float(payload.get("options_flow_score"))
    short_pressure_score = safe_float(payload.get("short_pressure_score"))
    insider_score = safe_float(payload.get("insider_score"))
    guidance_delta = safe_float(payload.get("guidance_delta"))

    if sentiment_score is None:
        sentiment_score = 0.0
        warnings.append("sentiment_score missing; defaulted to 0.")
    if options_flow_score is None:
        options_flow_score = 0.0
        warnings.append("options_flow_score missing; defaulted to 0.")
    if short_pressure_score is None:
        short_pressure_score = 50.0
        warnings.append("short_pressure_score missing; defaulted to neutral 50.")
    if insider_score is None:
        insider_score = 0.0
        warnings.append("insider_score missing; defaulted to 0.")
    if guidance_delta is None:
        guidance_delta = 0.0

    eps_trend = _eps_trend(prior_eps)
    momentum_20d = cumulative_growth(price_returns[-20:]) - 1.0 if price_returns else 0.0
    volatility_20d = pstdev(price_returns[-20:]) if len(price_returns) >= 2 else 0.0

    revenue_log_scale = 0.0
    if revenue_ttm is not None and revenue_ttm > 0:
        revenue_log_scale = clamp((revenue_ttm / 1_000_000_000.0), 0.0, 500.0)

    margin = clamp(net_margin_ttm or 0.0, -1.0, 1.0)
    short_squeeze_potential = clamp((short_pressure_score - 50.0) / 50.0, -1.0, 1.0)

    model_inputs = {
        "eps_trend": eps_trend,
        "margin": margin,
        "momentum_20d": clamp(momentum_20d, -1.0, 1.0),
        "volatility_20d": clamp(volatility_20d, 0.0, 1.0),
        "sentiment_score": clamp(sentiment_score, -1.0, 1.0),
        "options_flow_score": clamp(options_flow_score, -1.0, 1.0),
        "short_squeeze_potential": short_squeeze_potential,
        "insider_score": clamp(insider_score, -1.0, 1.0),
        "guidance_delta": clamp(guidance_delta, -1.0, 1.0),
        "revenue_scale": clamp(revenue_log_scale / 100.0, 0.0, 5.0),
    }

    coverage_fields = [
        "prior_eps",
        "latest_eps",
        "revenue_ttm",
        "net_margin_ttm",
        "price_returns",
        "sentiment_score",
        "options_flow_score",
        "short_pressure_score",
        "insider_score",
        "guidance_delta",
    ]
    present = 0
    for field in coverage_fields:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        present += 1
    feature_coverage = present / float(len(coverage_fields))

    return {
        "latest_eps": latest_eps,
        "prior_eps": prior_eps,
        "price_returns": price_returns,
        "raw": {
            "revenue_ttm": revenue_ttm,
            "net_margin_ttm": net_margin_ttm,
            "sentiment_score": sentiment_score,
            "options_flow_score": options_flow_score,
            "short_pressure_score": short_pressure_score,
            "insider_score": insider_score,
            "guidance_delta": guidance_delta,
            "momentum_20d": momentum_20d,
            "volatility_20d": volatility_20d,
        },
        "model_inputs": model_inputs,
        "feature_coverage": feature_coverage,
        "warnings": warnings,
    }


__all__ = ["build_earnings_features"]
