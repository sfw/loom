"""Macro regime and headwind/tailwind scoring tool."""

from __future__ import annotations

import json
from typing import Any

from loom.research.finance import clamp
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"classify_regime", "score_headwinds_tailwinds"}


class MacroRegimeEngineTool(Tool):
    """Classify macro regimes and map impacts to exposures."""

    @property
    def name(self) -> str:
        return "macro_regime_engine"

    @property
    def description(self) -> str:
        return (
            "Classify macro regime and compute headwind/tailwind scores for sector/style exposures."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["classify_regime", "score_headwinds_tailwinds"],
                },
                "indicators": {
                    "type": "object",
                    "description": (
                        "Macro indicators: inflation_yoy, gdp_growth, policy_rate, "
                        "yield_curve_spread, unemployment_rate, credit_spread."
                    ),
                },
                "exposures": {
                    "type": "object",
                    "description": (
                        "Exposure map with sector/style weights "
                        "(for example technology, financials, defensives, duration_sensitive)."
                    ),
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown output path.",
                },
            },
            "required": ["operation", "indicators"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail("operation must be classify_regime/score_headwinds_tailwinds")

        indicators = args.get("indicators", {})
        if not isinstance(indicators, dict):
            return ToolResult.fail("indicators must be an object")

        regime_payload = _classify(indicators)
        if operation == "classify_regime":
            payload = {
                "operation": operation,
                "classification": regime_payload,
                "keyless": True,
            }
        else:
            exposures = args.get("exposures", {})
            if not isinstance(exposures, dict) or not exposures:
                return ToolResult.fail("exposures object is required for score_headwinds_tailwinds")
            payload = _score(regime_payload, exposures)

        files_changed: list[str] = []
        output_path = str(args.get("output_path", "")).strip()
        lines = [f"Computed {operation}."]
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


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _classify(indicators: dict[str, Any]) -> dict[str, Any]:
    inflation = _to_float(indicators.get("inflation_yoy"), 2.5)
    growth = _to_float(indicators.get("gdp_growth"), 2.0)
    policy = _to_float(indicators.get("policy_rate"), 2.0)
    curve = _to_float(indicators.get("yield_curve_spread"), 1.0)
    unemployment = _to_float(indicators.get("unemployment_rate"), 4.5)
    credit = _to_float(indicators.get("credit_spread"), 1.5)

    if growth < 0.0:
        regime = "contraction"
    elif inflation >= 3.5 and growth <= 1.0:
        regime = "stagflation"
    elif growth >= 2.3 and inflation <= 3.0 and curve > 0:
        regime = "expansion"
    elif inflation < 2.3 and policy >= 4.0:
        regime = "disinflation_tightening"
    else:
        regime = "late_cycle"

    risk_score = clamp(
        (max(0.0, inflation - 2.0) * 0.18)
        + (max(0.0, 5.0 - growth) * 0.15)
        + (max(0.0, -curve) * 0.2)
        + (max(0.0, unemployment - 4.0) * 0.12)
        + (max(0.0, credit - 1.0) * 0.15),
        0.0,
        1.0,
    )

    return {
        "regime": regime,
        "risk_score": risk_score,
        "inputs": {
            "inflation_yoy": inflation,
            "gdp_growth": growth,
            "policy_rate": policy,
            "yield_curve_spread": curve,
            "unemployment_rate": unemployment,
            "credit_spread": credit,
        },
        "confidence": clamp(0.45 + 0.4 * (1.0 - abs(0.5 - risk_score)), 0.2, 0.9),
    }


_REGIME_IMPACT = {
    "expansion": {
        "technology": 0.25,
        "industrials": 0.2,
        "financials": 0.18,
        "defensives": -0.08,
        "duration_sensitive": -0.05,
        "cyclicals": 0.2,
        "value": 0.12,
        "growth": 0.1,
    },
    "late_cycle": {
        "technology": -0.06,
        "industrials": -0.03,
        "financials": 0.05,
        "defensives": 0.12,
        "duration_sensitive": -0.1,
        "cyclicals": -0.08,
        "value": 0.08,
        "growth": -0.06,
    },
    "contraction": {
        "technology": -0.2,
        "industrials": -0.22,
        "financials": -0.18,
        "defensives": 0.28,
        "duration_sensitive": 0.18,
        "cyclicals": -0.25,
        "value": -0.06,
        "growth": -0.1,
    },
    "stagflation": {
        "technology": -0.16,
        "industrials": -0.12,
        "financials": -0.08,
        "defensives": 0.14,
        "duration_sensitive": -0.2,
        "cyclicals": -0.18,
        "value": 0.05,
        "growth": -0.15,
    },
    "disinflation_tightening": {
        "technology": 0.05,
        "industrials": -0.02,
        "financials": 0.04,
        "defensives": 0.03,
        "duration_sensitive": -0.12,
        "cyclicals": -0.03,
        "value": 0.02,
        "growth": 0.04,
    },
}


def _score(classification: dict[str, Any], exposures: dict[str, Any]) -> dict[str, Any]:
    regime = str(classification.get("regime", "late_cycle"))
    impacts = _REGIME_IMPACT.get(regime, _REGIME_IMPACT["late_cycle"])

    details: list[dict[str, Any]] = []
    score = 0.0
    weight_sum = 0.0
    for key, raw_weight in exposures.items():
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        impact = impacts.get(str(key), 0.0)
        contribution = weight * impact
        score += contribution
        weight_sum += abs(weight)
        details.append(
            {
                "exposure": key,
                "weight": weight,
                "regime_impact": impact,
                "contribution": contribution,
            }
        )

    normalized = score / weight_sum if weight_sum > 0 else 0.0
    normalized = clamp(normalized, -1.0, 1.0)
    return {
        "operation": "score_headwinds_tailwinds",
        "classification": classification,
        "exposures": exposures,
        "contributions": details,
        "tailwind_score": normalized,
        "tailwind_label": "tailwind"
        if normalized > 0.1
        else ("headwind" if normalized < -0.1 else "neutral"),
        "confidence": clamp(float(classification.get("confidence", 0.5)) * 0.95, 0.1, 0.95),
        "keyless": True,
    }


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Macro Regime Engine Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["MacroRegimeEngineTool"]
