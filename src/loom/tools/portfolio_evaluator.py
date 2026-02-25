"""Portfolio evaluation and attribution tool."""

from __future__ import annotations

import json
import math
from statistics import mean, pstdev
from typing import Any

from loom.research.finance import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    percentile,
    sharpe_ratio,
    sortino_ratio,
)
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"performance_stats", "risk_stats", "drawdown", "benchmark_attribution"}


class PortfolioEvaluatorTool(Tool):
    """Evaluate portfolio performance, risk, and benchmark-relative outcomes."""

    @property
    def name(self) -> str:
        return "portfolio_evaluator"

    @property
    def description(self) -> str:
        return "Compute portfolio performance/risk stats, drawdowns, and benchmark attribution."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "performance_stats",
                        "risk_stats",
                        "drawdown",
                        "benchmark_attribution",
                    ],
                },
                "portfolio_returns": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "benchmark_returns": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "portfolio_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional value path for drawdown operation.",
                },
                "risk_free_rate": {"type": "number"},
                "periods_per_year": {"type": "integer"},
                "output_path": {"type": "string"},
            },
            "required": ["operation"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be performance_stats/risk_stats/drawdown/benchmark_attribution"
            )

        periods = _to_int(args.get("periods_per_year"), 252)
        risk_free = _to_float(args.get("risk_free_rate"), 0.0)

        try:
            if operation == "drawdown":
                payload = _drawdown_payload(args)
            else:
                returns = _to_float_list(args.get("portfolio_returns"))
                if len(returns) < 2:
                    raise ValueError("portfolio_returns with at least 2 values required")
                if operation == "performance_stats":
                    payload = _performance_payload(returns, periods=periods, risk_free=risk_free)
                elif operation == "risk_stats":
                    payload = _risk_payload(returns, periods=periods, risk_free=risk_free)
                else:
                    benchmark = _to_float_list(args.get("benchmark_returns"))
                    payload = _benchmark_payload(
                        returns,
                        benchmark,
                        periods=periods,
                        risk_free=risk_free,
                    )
        except Exception as e:
            return ToolResult.fail(str(e))

        lines = [f"Computed {operation}."]
        files_changed: list[str] = []
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


def _to_float(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float_list(raw: object) -> list[float]:
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _performance_payload(returns: list[float], *, periods: int, risk_free: float) -> dict[str, Any]:
    growth = 1.0
    for r in returns:
        growth *= 1.0 + r
    win_rate = sum(1 for r in returns if r > 0) / len(returns)
    return {
        "operation": "performance_stats",
        "period_count": len(returns),
        "total_return": growth - 1.0,
        "annualized_return": annualized_return(returns, periods_per_year=periods),
        "annualized_volatility": annualized_volatility(returns, periods_per_year=periods),
        "sharpe": sharpe_ratio(returns, risk_free_rate=risk_free, periods_per_year=periods),
        "sortino": sortino_ratio(returns, target_return=risk_free, periods_per_year=periods),
        "win_rate": win_rate,
        "keyless": True,
    }


def _risk_payload(returns: list[float], *, periods: int, risk_free: float) -> dict[str, Any]:
    vol = annualized_volatility(returns, periods_per_year=periods)
    var_95 = percentile(returns, 0.05)
    tail = [r for r in returns if r <= var_95]
    cvar_95 = sum(tail) / len(tail) if tail else var_95
    downside = [r for r in returns if r < 0]
    downside_dev = pstdev(downside) if len(downside) >= 2 else 0.0
    return {
        "operation": "risk_stats",
        "period_count": len(returns),
        "annualized_volatility": vol,
        "value_at_risk_95": var_95,
        "conditional_var_95": cvar_95,
        "downside_deviation": downside_dev,
        "sharpe": sharpe_ratio(returns, risk_free_rate=risk_free, periods_per_year=periods),
        "keyless": True,
    }


def _drawdown_payload(args: dict[str, Any]) -> dict[str, Any]:
    values = _to_float_list(args.get("portfolio_values"))
    if len(values) < 2:
        returns = _to_float_list(args.get("portfolio_returns"))
        if len(returns) < 2:
            raise ValueError("Provide portfolio_values or portfolio_returns for drawdown")
        values = [1.0]
        for r in returns:
            values.append(values[-1] * (1.0 + r))
    mdd, peak_idx, trough_idx = max_drawdown(values)
    return {
        "operation": "drawdown",
        "max_drawdown": mdd,
        "peak_index": peak_idx,
        "trough_index": trough_idx,
        "series_length": len(values),
        "keyless": True,
    }


def _benchmark_payload(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    *,
    periods: int,
    risk_free: float,
) -> dict[str, Any]:
    if len(benchmark_returns) < 2:
        raise ValueError("benchmark_returns with at least 2 values required")
    n = min(len(portfolio_returns), len(benchmark_returns))
    p = portfolio_returns[-n:]
    b = benchmark_returns[-n:]
    active = [p[i] - b[i] for i in range(n)]
    tracking_error = pstdev(active) * math.sqrt(periods) if len(active) >= 2 else 0.0
    active_return = annualized_return(p, periods_per_year=periods) - annualized_return(
        b, periods_per_year=periods
    )
    info_ratio = active_return / tracking_error if tracking_error > 0 else 0.0

    var_b = pstdev(b) ** 2 if len(b) >= 2 else 0.0
    cov_pb = 0.0
    if len(p) >= 2 and len(b) >= 2:
        mp = mean(p)
        mb = mean(b)
        cov_pb = sum((p[i] - mp) * (b[i] - mb) for i in range(n)) / (n - 1)
    beta = cov_pb / var_b if var_b > 1e-12 else 0.0

    up_capture = _capture_ratio(p, b, positive=True)
    down_capture = _capture_ratio(p, b, positive=False)
    return {
        "operation": "benchmark_attribution",
        "period_count": n,
        "active_return_annualized": active_return,
        "tracking_error": tracking_error,
        "information_ratio": info_ratio,
        "beta": beta,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "portfolio_sharpe": sharpe_ratio(p, risk_free_rate=risk_free, periods_per_year=periods),
        "benchmark_sharpe": sharpe_ratio(b, risk_free_rate=risk_free, periods_per_year=periods),
        "keyless": True,
    }


def _capture_ratio(portfolio: list[float], benchmark: list[float], *, positive: bool) -> float:
    idx = [i for i, r in enumerate(benchmark) if (r > 0 if positive else r < 0)]
    if not idx:
        return 0.0
    p = sum(portfolio[i] for i in idx)
    b = sum(benchmark[i] for i in idx)
    if b == 0:
        return 0.0
    return p / b


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Portfolio Evaluator Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["PortfolioEvaluatorTool"]
