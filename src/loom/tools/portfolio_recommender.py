"""Portfolio recommendation and monitoring tool."""

from __future__ import annotations

import json
from typing import Any

from loom.research.finance import clamp, normalize_weights_long_only, weights_turnover
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"recommend_portfolio", "propose_rebalance", "monitor_alerts"}


class PortfolioRecommenderTool(Tool):
    """Generate portfolio recommendations, rebalances, and monitoring alerts."""

    @property
    def name(self) -> str:
        return "portfolio_recommender"

    @property
    def description(self) -> str:
        return (
            "Recommend target portfolios, generate rebalance trade lists, "
            "and monitor thesis/risk alerts."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["recommend_portfolio", "propose_rebalance", "monitor_alerts"],
                },
                "candidates": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Candidate rows (symbol, score, conviction, max_weight, etc).",
                },
                "target_weights": {
                    "type": "object",
                    "description": "Target weights for propose_rebalance/monitor_alerts.",
                },
                "current_weights": {
                    "type": "object",
                    "description": "Current weights for rebalance comparisons.",
                },
                "risk_profile": {
                    "type": "string",
                    "enum": ["conservative", "balanced", "aggressive"],
                },
                "max_positions": {"type": "integer"},
                "max_weight": {"type": "number"},
                "turnover_budget": {"type": "number"},
                "estimated_cost_bps": {"type": "number"},
                "rules": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Alert rules: {name,metric,value,threshold,direction}.",
                },
                "output_path": {"type": "string"},
            },
            "required": ["operation"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be recommend_portfolio/propose_rebalance/monitor_alerts"
            )

        try:
            if operation == "recommend_portfolio":
                payload = _recommend(args)
            elif operation == "propose_rebalance":
                payload = _rebalance(args)
            else:
                payload = _alerts(args)
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


def _risk_scaler(profile: str) -> float:
    if profile == "conservative":
        return 0.75
    if profile == "aggressive":
        return 1.2
    return 1.0


def _recommend(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("candidates", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("candidates list is required")
    profile = str(args.get("risk_profile", "balanced")).strip().lower() or "balanced"
    max_positions = _to_int(args.get("max_positions"), 12)
    max_positions = max(1, min(100, max_positions))
    max_weight = clamp(_to_float(args.get("max_weight"), 0.2), 0.02, 1.0)
    scale = _risk_scaler(profile)

    scored: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        score = _to_float(item.get("score"), 0.0)
        conviction = clamp(_to_float(item.get("conviction"), 0.5), 0.0, 1.0)
        risk = max(1e-8, _to_float(item.get("risk"), 0.2))
        adjusted = (score * conviction * scale) / risk
        scored.append(
            {
                "symbol": symbol,
                "score": score,
                "conviction": conviction,
                "risk": risk,
                "adjusted_score": adjusted,
                "thesis": str(item.get("thesis", "")).strip(),
                "thesis_breakers": item.get("thesis_breakers", []),
            }
        )
    scored.sort(key=lambda row: row["adjusted_score"], reverse=True)
    selected = scored[:max_positions]
    if not selected:
        raise ValueError("No valid candidates provided")

    raw_weights = {row["symbol"]: max(0.0, row["adjusted_score"]) for row in selected}
    weights = normalize_weights_long_only(raw_weights)
    capped = {sym: min(w, max_weight) for sym, w in weights.items()}
    weights = normalize_weights_long_only(capped)

    rationale = [
        {
            "symbol": row["symbol"],
            "why": (
                f"Adjusted score {row['adjusted_score']:.3f} "
                f"(score={row['score']:.3f}, conviction={row['conviction']:.2f}, "
                f"risk={row['risk']:.2f})."
            ),
            "thesis": row["thesis"],
            "thesis_breakers": row["thesis_breakers"],
        }
        for row in selected
    ]
    return {
        "operation": "recommend_portfolio",
        "risk_profile": profile,
        "target_weights": weights,
        "position_count": len(weights),
        "rationale": rationale,
        "confidence": clamp(0.45 + 0.03 * len(weights), 0.4, 0.9),
        "keyless": True,
    }


def _rebalance(args: dict[str, Any]) -> dict[str, Any]:
    target = normalize_weights_long_only(_as_float_map(args.get("target_weights")))
    current = normalize_weights_long_only(_as_float_map(args.get("current_weights")))
    if not target:
        # Allow generating target from candidates inline.
        auto = _recommend(args)
        target = auto["target_weights"]
    turnover_budget = max(0.0, _to_float(args.get("turnover_budget"), 1.0))
    cost_bps = max(0.0, _to_float(args.get("estimated_cost_bps"), 5.0))

    keys = sorted(set(target.keys()) | set(current.keys()))
    trades: list[dict[str, Any]] = []
    gross_notional = 0.0
    for symbol in keys:
        cur = float(current.get(symbol, 0.0))
        tgt = float(target.get(symbol, 0.0))
        delta = tgt - cur
        if abs(delta) < 1e-6:
            continue
        side = "buy" if delta > 0 else "sell"
        trades.append(
            {
                "symbol": symbol,
                "current_weight": cur,
                "target_weight": tgt,
                "delta_weight": delta,
                "side": side,
            }
        )
        gross_notional += abs(delta)

    turnover = weights_turnover(current, target)
    est_cost = gross_notional * (cost_bps / 10_000.0)
    within_budget = turnover <= turnover_budget
    return {
        "operation": "propose_rebalance",
        "current_weights": current,
        "target_weights": target,
        "trades": trades,
        "turnover": turnover,
        "turnover_budget": turnover_budget,
        "within_turnover_budget": within_budget,
        "estimated_transaction_cost_pct": est_cost,
        "confidence": 0.7 if within_budget else 0.55,
        "keyless": True,
    }


def _alerts(args: dict[str, Any]) -> dict[str, Any]:
    rules = args.get("rules", [])
    if not isinstance(rules, list):
        rules = []
    triggered: list[dict[str, Any]] = []
    for item in rules:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "rule")).strip()
        metric = str(item.get("metric", "metric")).strip()
        value = _to_float(item.get("value"), 0.0)
        threshold = _to_float(item.get("threshold"), 0.0)
        direction = str(item.get("direction", "gt")).strip().lower()
        if direction == "lt":
            hit = value < threshold
        elif direction == "abs_gt":
            hit = abs(value) > abs(threshold)
        else:
            hit = value > threshold
        if hit:
            triggered.append(
                {
                    "name": name,
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "direction": direction,
                }
            )

    return {
        "operation": "monitor_alerts",
        "triggered_count": len(triggered),
        "alerts": triggered,
        "status": "action_required" if triggered else "normal",
        "keyless": True,
        "confidence": 0.75 if rules else 0.3,
    }


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


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Portfolio Recommender Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["PortfolioRecommenderTool"]
