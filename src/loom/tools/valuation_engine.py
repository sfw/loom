"""Valuation toolkit (DCF, implied growth, scenarios)."""

from __future__ import annotations

import json
from typing import Any

from loom.research.finance import clamp
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"intrinsic_value_range", "implied_growth", "scenario_valuation"}


class ValuationEngineTool(Tool):
    """Compute valuation ranges and implied expectations."""

    @property
    def name(self) -> str:
        return "valuation_engine"

    @property
    def description(self) -> str:
        return (
            "Run lightweight valuation models (DCF range, implied growth, "
            "probability-weighted scenarios)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["intrinsic_value_range", "implied_growth", "scenario_valuation"],
                },
                "free_cash_flow": {"type": "number"},
                "discount_rate": {"type": "number"},
                "terminal_growth": {"type": "number"},
                "years": {"type": "integer"},
                "shares_outstanding": {"type": "number"},
                "net_debt": {"type": "number"},
                "growth_base": {"type": "number"},
                "growth_bull": {"type": "number"},
                "growth_bear": {"type": "number"},
                "market_price": {"type": "number"},
                "scenarios": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Scenario rows: {name, probability, free_cash_flow, growth, multiple, "
                        "value_per_share}."
                    ),
                },
                "output_path": {"type": "string"},
            },
            "required": ["operation"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be intrinsic_value_range/implied_growth/scenario_valuation"
            )

        try:
            if operation == "intrinsic_value_range":
                payload = _intrinsic_value_range(args)
            elif operation == "implied_growth":
                payload = _implied_growth(args)
            else:
                payload = _scenario_valuation(args)
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


def _to_float(value: object, default: float | None = None) -> float | None:
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


def _dcf_value(
    *,
    free_cash_flow: float,
    growth: float,
    discount_rate: float,
    terminal_growth: float,
    years: int,
) -> float:
    if years < 1:
        years = 1
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must be greater than terminal_growth")

    pv = 0.0
    fcf = free_cash_flow
    for year in range(1, years + 1):
        fcf *= 1.0 + growth
        pv += fcf / ((1.0 + discount_rate) ** year)
    terminal = (fcf * (1.0 + terminal_growth)) / (discount_rate - terminal_growth)
    pv += terminal / ((1.0 + discount_rate) ** years)
    return pv


def _intrinsic_value_range(args: dict[str, Any]) -> dict[str, Any]:
    fcf = _to_float(args.get("free_cash_flow"))
    shares = _to_float(args.get("shares_outstanding"))
    if fcf is None or shares in (None, 0):
        raise ValueError("free_cash_flow and shares_outstanding are required")
    discount = _to_float(args.get("discount_rate"), 0.1) or 0.1
    terminal = _to_float(args.get("terminal_growth"), 0.02) or 0.02
    years = _to_int(args.get("years"), 5)
    net_debt = _to_float(args.get("net_debt"), 0.0) or 0.0
    growth_base = _to_float(args.get("growth_base"), 0.06) or 0.06
    growth_bull = _to_float(args.get("growth_bull"), growth_base + 0.03) or (growth_base + 0.03)
    growth_bear = _to_float(args.get("growth_bear"), growth_base - 0.03) or (growth_base - 0.03)

    enterprise = {
        "bear": _dcf_value(
            free_cash_flow=fcf,
            growth=growth_bear,
            discount_rate=discount,
            terminal_growth=terminal,
            years=years,
        ),
        "base": _dcf_value(
            free_cash_flow=fcf,
            growth=growth_base,
            discount_rate=discount,
            terminal_growth=terminal,
            years=years,
        ),
        "bull": _dcf_value(
            free_cash_flow=fcf,
            growth=growth_bull,
            discount_rate=discount,
            terminal_growth=terminal,
            years=years,
        ),
    }
    equity = {name: val - net_debt for name, val in enterprise.items()}
    per_share = {name: val / shares for name, val in equity.items()}
    base_ps = per_share["base"]
    return {
        "operation": "intrinsic_value_range",
        "assumptions": {
            "free_cash_flow": fcf,
            "discount_rate": discount,
            "terminal_growth": terminal,
            "years": years,
            "shares_outstanding": shares,
            "net_debt": net_debt,
            "growth_bear": growth_bear,
            "growth_base": growth_base,
            "growth_bull": growth_bull,
        },
        "enterprise_value": enterprise,
        "equity_value": equity,
        "per_share_value": per_share,
        "confidence": 0.62,
        "keyless": True,
        "warnings": [
            "Simplified DCF; validate with detailed statement and capital structure modeling."
        ],
        "recommended_anchor": base_ps,
    }


def _implied_growth(args: dict[str, Any]) -> dict[str, Any]:
    price = _to_float(args.get("market_price"))
    shares = _to_float(args.get("shares_outstanding"))
    fcf = _to_float(args.get("free_cash_flow"))
    if price is None or shares in (None, 0) or fcf is None:
        raise ValueError("market_price, shares_outstanding, and free_cash_flow are required")
    discount = _to_float(args.get("discount_rate"), 0.1) or 0.1
    terminal = _to_float(args.get("terminal_growth"), 0.02) or 0.02
    years = _to_int(args.get("years"), 5)
    net_debt = _to_float(args.get("net_debt"), 0.0) or 0.0
    target_equity = price * shares
    target_enterprise = target_equity + net_debt

    lo = -0.25
    hi = 0.45
    for _ in range(60):
        mid = (lo + hi) / 2.0
        val = _dcf_value(
            free_cash_flow=fcf,
            growth=mid,
            discount_rate=discount,
            terminal_growth=terminal,
            years=years,
        )
        if val > target_enterprise:
            hi = mid
        else:
            lo = mid
    implied = (lo + hi) / 2.0
    return {
        "operation": "implied_growth",
        "market_price": price,
        "shares_outstanding": shares,
        "target_enterprise_value": target_enterprise,
        "implied_growth_rate": implied,
        "assumptions": {
            "free_cash_flow": fcf,
            "discount_rate": discount,
            "terminal_growth": terminal,
            "years": years,
            "net_debt": net_debt,
        },
        "confidence": 0.55,
        "keyless": True,
    }


def _scenario_valuation(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("scenarios", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("scenarios is required")
    shares = _to_float(args.get("shares_outstanding"))
    fcf = _to_float(args.get("free_cash_flow"))

    scenarios: list[dict[str, Any]] = []
    total_prob = 0.0
    expected = 0.0
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"scenario-{len(scenarios) + 1}")).strip()
        prob = _to_float(item.get("probability"), 0.0) or 0.0
        prob = max(0.0, prob)
        total_prob += prob

        value_per_share = _to_float(item.get("value_per_share"))
        if value_per_share is None:
            local_fcf = _to_float(item.get("free_cash_flow"), fcf)
            growth = _to_float(item.get("growth"), 0.05) or 0.05
            multiple = _to_float(item.get("multiple"))
            if multiple is not None and local_fcf is not None and shares not in (None, 0):
                value_per_share = (local_fcf * multiple) / shares
            elif local_fcf is not None and shares not in (None, 0):
                discount = _to_float(item.get("discount_rate"), 0.1) or 0.1
                terminal = _to_float(item.get("terminal_growth"), 0.02) or 0.02
                years = _to_int(item.get("years"), 5)
                ev = _dcf_value(
                    free_cash_flow=local_fcf,
                    growth=growth,
                    discount_rate=discount,
                    terminal_growth=terminal,
                    years=years,
                )
                net_debt = _to_float(item.get("net_debt"), 0.0) or 0.0
                value_per_share = (ev - net_debt) / shares
            else:
                continue

        expected += prob * value_per_share
        scenarios.append(
            {
                "name": name,
                "probability": prob,
                "value_per_share": value_per_share,
            }
        )

    if not scenarios:
        raise ValueError("No valid scenarios parsed")
    if total_prob <= 0:
        raise ValueError("Scenario probabilities must sum to > 0")
    normalized_expected = expected / total_prob
    for row in scenarios:
        row["normalized_probability"] = row["probability"] / total_prob

    sorted_vals = sorted(float(row["value_per_share"]) for row in scenarios)
    low = sorted_vals[0]
    high = sorted_vals[-1]

    return {
        "operation": "scenario_valuation",
        "scenarios": scenarios,
        "expected_value_per_share": normalized_expected,
        "value_range": {"low": low, "high": high},
        "confidence": clamp(0.4 + 0.07 * len(scenarios), 0.35, 0.9),
        "keyless": True,
    }


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Valuation Engine Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["ValuationEngineTool"]
