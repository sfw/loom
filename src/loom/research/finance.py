"""Shared finance analytics helpers (stdlib-only)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from statistics import mean, pstdev


def safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            return float(text)
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def align_series_tail(series_by_key: dict[str, list[float]]) -> dict[str, list[float]]:
    cleaned: dict[str, list[float]] = {}
    min_len: int | None = None
    for key, values in series_by_key.items():
        nums = [float(v) for v in values if safe_float(v) is not None]
        if len(nums) < 2:
            continue
        cleaned[key] = nums
        min_len = len(nums) if min_len is None else min(min_len, len(nums))
    if not cleaned or min_len is None or min_len < 2:
        return {}
    return {key: values[-min_len:] for key, values in cleaned.items()}


def pct_returns(prices: list[float]) -> list[float]:
    if len(prices) < 2:
        return []
    returns: list[float] = []
    prev = prices[0]
    for price in prices[1:]:
        if prev == 0:
            prev = price
            continue
        returns.append((price / prev) - 1.0)
        prev = price
    return returns


def cumulative_growth(returns: list[float]) -> float:
    growth = 1.0
    for r in returns:
        growth *= 1.0 + float(r)
    return growth


def annualized_return(returns: list[float], *, periods_per_year: int = 252) -> float:
    if not returns:
        return 0.0
    growth = cumulative_growth(returns)
    years = max(len(returns) / float(periods_per_year), 1.0 / float(periods_per_year))
    if growth <= 0:
        return -1.0
    return (growth ** (1.0 / years)) - 1.0


def annualized_volatility(returns: list[float], *, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) * math.sqrt(float(periods_per_year))


def sharpe_ratio(
    returns: list[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if not returns:
        return 0.0
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    if ann_vol <= 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def downside_deviation(
    returns: list[float],
    *,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if not returns:
        return 0.0
    target_period = target_return / float(periods_per_year)
    downside = [min(0.0, r - target_period) for r in returns]
    squared = [d * d for d in downside]
    if not squared:
        return 0.0
    return math.sqrt(sum(squared) / len(squared)) * math.sqrt(float(periods_per_year))


def sortino_ratio(
    returns: list[float],
    *,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if not returns:
        return 0.0
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    dd = downside_deviation(
        returns,
        target_return=target_return,
        periods_per_year=periods_per_year,
    )
    if dd <= 0:
        return 0.0
    return (ann_ret - target_return) / dd


def max_drawdown(prices: list[float]) -> tuple[float, int, int]:
    if not prices:
        return 0.0, -1, -1
    peak = prices[0]
    peak_idx = 0
    trough_idx = 0
    best_dd = 0.0
    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            peak_idx = i
        if peak > 0:
            dd = (price / peak) - 1.0
            if dd < best_dd:
                best_dd = dd
                trough_idx = i
    return best_dd, peak_idx, trough_idx


def mean_return(returns: list[float]) -> float:
    if not returns:
        return 0.0
    return mean(returns)


def covariance(x: list[float], y: list[float]) -> float:
    if not x or not y:
        return 0.0
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x_vals = x[-n:]
    y_vals = y[-n:]
    mx = mean(x_vals)
    my = mean(y_vals)
    acc = 0.0
    for i in range(n):
        acc += (x_vals[i] - mx) * (y_vals[i] - my)
    return acc / float(n - 1)


def correlation(x: list[float], y: list[float]) -> float:
    cov = covariance(x, y)
    sx = pstdev(x) if len(x) > 1 else 0.0
    sy = pstdev(y) if len(y) > 1 else 0.0
    if sx <= 0 or sy <= 0:
        return 0.0
    return cov / (sx * sy)


def covariance_matrix(series_by_key: dict[str, list[float]]) -> tuple[list[str], list[list[float]]]:
    aligned = align_series_tail(series_by_key)
    keys = sorted(aligned.keys())
    matrix: list[list[float]] = []
    for a in keys:
        row: list[float] = []
        for b in keys:
            row.append(covariance(aligned[a], aligned[b]))
        matrix.append(row)
    return keys, matrix


def correlation_matrix(
    series_by_key: dict[str, list[float]],
) -> tuple[list[str], list[list[float]]]:
    aligned = align_series_tail(series_by_key)
    keys = sorted(aligned.keys())
    matrix: list[list[float]] = []
    for a in keys:
        row: list[float] = []
        for b in keys:
            row.append(correlation(aligned[a], aligned[b]))
        matrix.append(row)
    return keys, matrix


def portfolio_return(weights: dict[str, float], expected_returns: dict[str, float]) -> float:
    out = 0.0
    for key, weight in weights.items():
        out += float(weight) * float(expected_returns.get(key, 0.0))
    return out


def portfolio_variance(
    weights: dict[str, float],
    order: list[str],
    covariance_values: list[list[float]],
) -> float:
    idx = {name: i for i, name in enumerate(order)}
    var = 0.0
    for a, wa in weights.items():
        ia = idx.get(a)
        if ia is None:
            continue
        for b, wb in weights.items():
            ib = idx.get(b)
            if ib is None:
                continue
            var += wa * wb * covariance_values[ia][ib]
    return max(0.0, var)


def normalize_weights_long_only(raw_weights: dict[str, float]) -> dict[str, float]:
    clipped = {k: max(0.0, float(v)) for k, v in raw_weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        count = len(clipped)
        if count == 0:
            return {}
        equal = 1.0 / float(count)
        return {k: equal for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def weights_turnover(old: dict[str, float], new: dict[str, float]) -> float:
    keys = set(old.keys()) | set(new.keys())
    return 0.5 * sum(abs(float(old.get(k, 0.0)) - float(new.get(k, 0.0))) for k in keys)


def generate_weight_grid(
    assets: list[str],
    *,
    step: float = 0.1,
    max_portfolios: int = 50_000,
) -> Iterable[dict[str, float]]:
    if not assets:
        return []
    if len(assets) == 1:
        return [{assets[0]: 1.0}]

    step = clamp(step, 0.01, 1.0)
    buckets = max(1, int(round(1.0 / step)))
    out: list[dict[str, float]] = []

    # Enumerate integer partitions of `buckets` over N assets.
    ranges = [range(0, buckets + 1) for _ in assets]
    for combo in itertools.product(*ranges):
        if sum(combo) != buckets:
            continue
        weights = {assets[i]: combo[i] / float(buckets) for i in range(len(assets))}
        out.append(weights)
        if len(out) >= max_portfolios:
            break
    return out


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    p = clamp(p, 0.0, 1.0)
    ordered = sorted(float(v) for v in values)
    idx = int(round((len(ordered) - 1) * p))
    return ordered[idx]


__all__ = [
    "align_series_tail",
    "annualized_return",
    "annualized_volatility",
    "clamp",
    "correlation",
    "correlation_matrix",
    "covariance",
    "covariance_matrix",
    "cumulative_growth",
    "downside_deviation",
    "generate_weight_grid",
    "max_drawdown",
    "mean_return",
    "normalize_weights_long_only",
    "pct_returns",
    "percentile",
    "portfolio_return",
    "portfolio_variance",
    "safe_float",
    "sharpe_ratio",
    "sortino_ratio",
    "weights_turnover",
]
