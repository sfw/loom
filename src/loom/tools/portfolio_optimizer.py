"""Portfolio optimization tool (long-only heuristics)."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

from loom.research.finance import (
    clamp,
    covariance_matrix,
    generate_weight_grid,
    normalize_weights_long_only,
    percentile,
    portfolio_return,
    portfolio_variance,
    weights_turnover,
)
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"optimize_mvo", "optimize_risk_parity", "optimize_cvar"}


class PortfolioOptimizerTool(Tool):
    """Optimize long-only portfolios under practical constraints."""

    @property
    def name(self) -> str:
        return "portfolio_optimizer"

    @property
    def description(self) -> str:
        return (
            "Optimize long-only portfolios using MVO/risk-parity/CVaR heuristics "
            "with constraint support."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["optimize_mvo", "optimize_risk_parity", "optimize_cvar"],
                },
                "expected_returns": {
                    "type": "object",
                    "description": "Asset expected returns mapping.",
                },
                "asset_returns": {
                    "type": "object",
                    "description": "Asset historical returns mapping for covariance/CVaR.",
                },
                "current_weights": {
                    "type": "object",
                    "description": "Current portfolio weights for turnover penalties.",
                },
                "constraints": {
                    "type": "object",
                    "description": (
                        "Constraint config: min_weight,max_weight,max_turnover,"
                        "grid_step,risk_aversion."
                    ),
                },
                "output_path": {"type": "string"},
            },
            "required": ["operation"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 60

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be optimize_mvo/optimize_risk_parity/optimize_cvar"
            )

        try:
            if operation == "optimize_mvo":
                payload = await asyncio.to_thread(_optimize_mvo, args)
            elif operation == "optimize_risk_parity":
                payload = await asyncio.to_thread(_optimize_risk_parity, args)
            else:
                payload = await asyncio.to_thread(_optimize_cvar, args)
        except Exception as e:
            return ToolResult.fail(str(e))

        files_changed: list[str] = []
        lines = [f"Computed {operation}."]
        output_path = str(args.get("output_path", "")).strip()
        if output_path:
            if ctx.workspace is None:
                return ToolResult.fail("No workspace set for output_path")
            path = self._resolve_path(output_path, ctx.workspace)
            path.parent.mkdir(parents=True, exist_ok=True)
            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(path), subtask_id=ctx.subtask_id)
            path.write_text(_render_report(payload), encoding="utf-8")
            rel = str(path.relative_to(ctx.workspace))
            files_changed.append(rel)
            lines.append(f"Report written: {rel}")

        return ToolResult.ok("\n".join(lines), data=payload, files_changed=files_changed)


def _as_float_map(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _as_series_map(raw: object) -> dict[str, list[float]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[float]] = {}
    for key, values in raw.items():
        if not isinstance(values, list):
            continue
        nums: list[float] = []
        for item in values:
            try:
                nums.append(float(item))
            except (TypeError, ValueError):
                continue
        if nums:
            out[str(key)] = nums
    return out


def _constraints(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        raw = {}
    min_w = _to_float(raw.get("min_weight"), 0.0)
    max_w = _to_float(raw.get("max_weight"), 1.0)
    max_turnover = _to_float(raw.get("max_turnover"), 10.0)
    grid_step = _to_float(raw.get("grid_step"), 0.1)
    risk_aversion = _to_float(raw.get("risk_aversion"), 3.0)
    cvar_alpha = _to_float(raw.get("cvar_alpha"), 0.05)
    return {
        "min_weight": clamp(min_w, 0.0, 1.0),
        "max_weight": clamp(max_w, 0.01, 1.0),
        "max_turnover": max(0.0, max_turnover),
        "grid_step": clamp(grid_step, 0.01, 0.5),
        "risk_aversion": max(0.0, risk_aversion),
        "cvar_alpha": clamp(cvar_alpha, 0.01, 0.2),
    }


def _to_float(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _expected_and_cov(
    args: dict[str, Any],
) -> tuple[dict[str, float], list[str], list[list[float]]]:
    expected = _as_float_map(args.get("expected_returns"))
    if not expected:
        raise ValueError("expected_returns is required")
    series = _as_series_map(args.get("asset_returns"))
    if series:
        order, cov = covariance_matrix(series)
        filtered = {k: v for k, v in expected.items() if k in order}
        if filtered:
            expected = filtered
            order = [asset for asset in order if asset in expected]
            idx = {name: i for i, name in enumerate(order)}
            cov = [[cov[idx[a]][idx[b]] for b in order] for a in order]
            return expected, order, cov

    # Fallback diagonal covariance when return series unavailable.
    order = sorted(expected.keys())
    cov = [[0.0 for _ in order] for _ in order]
    for i in range(len(order)):
        cov[i][i] = 0.04  # 20% annual vol squared fallback.
    return expected, order, cov


def _valid_weights(
    weights: dict[str, float],
    *,
    constraints: dict[str, float],
    current_weights: dict[str, float],
) -> bool:
    min_w = constraints["min_weight"]
    max_w = constraints["max_weight"]
    for w in weights.values():
        if w < min_w - 1e-9 or w > max_w + 1e-9:
            return False
    turnover = weights_turnover(current_weights, weights)
    return turnover <= constraints["max_turnover"] + 1e-9


def _optimize_mvo(args: dict[str, Any]) -> dict[str, Any]:
    expected, order, cov = _expected_and_cov(args)
    current = normalize_weights_long_only(_as_float_map(args.get("current_weights")))
    c = _constraints(args.get("constraints"))

    best_weights: dict[str, float] | None = None
    best_score = -1e18
    best_stats: dict[str, float] = {}
    for weights in generate_weight_grid(order, step=c["grid_step"], max_portfolios=100_000):
        if not _valid_weights(weights, constraints=c, current_weights=current):
            continue
        exp_ret = portfolio_return(weights, expected)
        var = portfolio_variance(weights, order, cov)
        vol = math.sqrt(max(0.0, var))
        turnover = weights_turnover(current, weights)
        score = exp_ret - c["risk_aversion"] * vol - 0.02 * turnover
        if score > best_score:
            best_score = score
            best_weights = weights
            best_stats = {
                "expected_return": exp_ret,
                "expected_volatility": vol,
                "objective_score": score,
                "turnover": turnover,
            }

    if best_weights is None:
        raise ValueError("No feasible portfolio under constraints")
    return {
        "operation": "optimize_mvo",
        "weights": best_weights,
        "stats": best_stats,
        "constraints": c,
        "keyless": True,
        "confidence": 0.68,
    }


def _optimize_risk_parity(args: dict[str, Any]) -> dict[str, Any]:
    expected, order, cov = _expected_and_cov(args)
    c = _constraints(args.get("constraints"))
    current = normalize_weights_long_only(_as_float_map(args.get("current_weights")))

    inv_vol: dict[str, float] = {}
    for idx, asset in enumerate(order):
        var = max(1e-8, cov[idx][idx])
        inv_vol[asset] = 1.0 / math.sqrt(var)
    weights = normalize_weights_long_only(inv_vol)

    # Clamp and renormalize according to min/max bounds.
    bounded = {}
    for asset, weight in weights.items():
        bounded[asset] = clamp(weight, c["min_weight"], c["max_weight"])
    weights = normalize_weights_long_only(bounded)
    if not _valid_weights(weights, constraints=c, current_weights=current):
        raise ValueError("Risk parity solution violates constraints")

    exp_ret = portfolio_return(weights, expected)
    vol = math.sqrt(max(0.0, portfolio_variance(weights, order, cov)))
    return {
        "operation": "optimize_risk_parity",
        "weights": weights,
        "stats": {
            "expected_return": exp_ret,
            "expected_volatility": vol,
            "turnover": weights_turnover(current, weights),
        },
        "constraints": c,
        "keyless": True,
        "confidence": 0.63,
    }


def _optimize_cvar(args: dict[str, Any]) -> dict[str, Any]:
    series = _as_series_map(args.get("asset_returns"))
    if not series:
        raise ValueError("asset_returns is required for optimize_cvar")
    expected = _as_float_map(args.get("expected_returns")) or {
        asset: sum(vals) / len(vals) for asset, vals in series.items() if vals
    }
    assets = sorted(set(series.keys()) & set(expected.keys()))
    if not assets:
        raise ValueError("No overlap between asset_returns and expected_returns")

    c = _constraints(args.get("constraints"))
    current = normalize_weights_long_only(_as_float_map(args.get("current_weights")))
    # Align return lengths.
    n = min(len(series[a]) for a in assets)
    if n < 5:
        raise ValueError("Need at least 5 return points per asset for CVaR optimization")

    best_weights: dict[str, float] | None = None
    best_obj = 1e18
    best_stats: dict[str, float] = {}
    for weights in generate_weight_grid(assets, step=c["grid_step"], max_portfolios=100_000):
        if not _valid_weights(weights, constraints=c, current_weights=current):
            continue
        pnl: list[float] = []
        for i in range(n):
            period = 0.0
            for asset in assets:
                period += weights[asset] * series[asset][-n + i]
            pnl.append(period)
        var_level = percentile(pnl, c["cvar_alpha"])
        tail = [x for x in pnl if x <= var_level]
        cvar = sum(tail) / len(tail) if tail else var_level
        exp_ret = portfolio_return(weights, expected)
        turnover = weights_turnover(current, weights)
        obj = -exp_ret + abs(cvar) + 0.01 * turnover
        if obj < best_obj:
            best_obj = obj
            best_weights = weights
            best_stats = {
                "expected_return": exp_ret,
                "var_alpha": c["cvar_alpha"],
                "value_at_risk": var_level,
                "conditional_value_at_risk": cvar,
                "turnover": turnover,
                "objective_score": obj,
            }

    if best_weights is None:
        raise ValueError("No feasible CVaR portfolio under constraints")
    return {
        "operation": "optimize_cvar",
        "weights": best_weights,
        "stats": best_stats,
        "constraints": c,
        "keyless": True,
        "confidence": 0.6,
    }


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Portfolio Optimizer Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["PortfolioOptimizerTool"]
