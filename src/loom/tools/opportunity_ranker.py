"""Opportunity ranking and thesis-break assessment tool."""

from __future__ import annotations

import json
from typing import Any

from loom.research.finance import clamp
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"rank_candidates", "explain_rank", "thesis_breakers"}


class OpportunityRankerTool(Tool):
    """Rank opportunities and surface investability rationale."""

    @property
    def name(self) -> str:
        return "opportunity_ranker"

    @property
    def description(self) -> str:
        return (
            "Rank investment candidates by expected return/risk/confidence, "
            "generate ranking explanations, and evaluate thesis-breakers."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["rank_candidates", "explain_rank", "thesis_breakers"],
                },
                "candidates": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Candidate rows with fields such as symbol, expected_return, risk, "
                        "confidence, valuation_upside, liquidity_score, downside_risk."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Top rows to return for rank_candidates/explain_rank.",
                },
                "candidate": {
                    "type": "object",
                    "description": "Single candidate for thesis_breakers.",
                },
                "break_conditions": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": (
                        "Break rows: {name,current,threshold,direction('lt'|'gt'|'abs_gt')}"
                    ),
                },
                "output_path": {"type": "string"},
            },
            "required": ["operation"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail("operation must be rank_candidates/explain_rank/thesis_breakers")

        try:
            if operation == "rank_candidates":
                payload = _rank(args)
            elif operation == "explain_rank":
                payload = _explain(args)
            else:
                payload = _thesis_breakers(args)
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


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_candidate(row: dict[str, Any]) -> tuple[float, dict[str, float]]:
    expected_return = _to_float(row.get("expected_return"))
    risk = max(1e-8, _to_float(row.get("risk"), 0.2))
    confidence = clamp(_to_float(row.get("confidence"), 0.5), 0.0, 1.0)
    valuation_upside = _to_float(row.get("valuation_upside"))
    liquidity = clamp(_to_float(row.get("liquidity_score"), 0.5), 0.0, 1.0)
    downside = _to_float(row.get("downside_risk"), 0.1)
    macro = _to_float(row.get("macro_tailwind_score"), 0.0)

    parts = {
        "risk_adjusted_return": expected_return / risk,
        "valuation_component": valuation_upside,
        "confidence_component": confidence,
        "liquidity_component": liquidity,
        "downside_penalty": -abs(downside),
        "macro_component": macro,
    }
    score = (
        0.42 * parts["risk_adjusted_return"]
        + 0.2 * parts["valuation_component"]
        + 0.14 * parts["confidence_component"]
        + 0.08 * parts["liquidity_component"]
        + 0.08 * parts["macro_component"]
        + 0.08 * parts["downside_penalty"]
    )
    return score, parts


def _rank(args: dict[str, Any]) -> dict[str, Any]:
    raw = args.get("candidates", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("candidates list is required")
    top_k = _to_top_k(args.get("top_k"), default=10)

    ranked: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper() or str(
            item.get("name", f"candidate-{len(ranked) + 1}")
        )
        score, parts = _score_candidate(item)
        ranked.append(
            {
                "symbol": symbol,
                "score": score,
                "score_parts": parts,
                "expected_return": _to_float(item.get("expected_return")),
                "risk": _to_float(item.get("risk"), 0.2),
                "confidence": clamp(_to_float(item.get("confidence"), 0.5), 0.0, 1.0),
            }
        )
    ranked.sort(key=lambda row: row["score"], reverse=True)
    out = ranked[:top_k]
    return {
        "operation": "rank_candidates",
        "count": len(out),
        "ranked": out,
        "keyless": True,
        "confidence": clamp(0.45 + 0.03 * len(out), 0.4, 0.85),
    }


def _explain(args: dict[str, Any]) -> dict[str, Any]:
    ranked = _rank(args)
    explanations: list[dict[str, Any]] = []
    for row in ranked["ranked"]:
        parts = row.get("score_parts", {})
        drivers = sorted(
            ((k, v) for k, v in parts.items() if isinstance(v, (int, float))),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:3]
        driver_text = ", ".join(f"{name}={value:+.3f}" for name, value in drivers)
        explanations.append(
            {
                "symbol": row["symbol"],
                "score": row["score"],
                "explanation": (
                    f"Top-ranked due to {driver_text}; "
                    f"expected_return={row['expected_return']:+.2%}, risk={row['risk']:.2f}."
                ),
            }
        )
    return {
        "operation": "explain_rank",
        "ranked": ranked["ranked"],
        "explanations": explanations,
        "keyless": True,
        "confidence": ranked.get("confidence", 0.6),
    }


def _thesis_breakers(args: dict[str, Any]) -> dict[str, Any]:
    candidate = args.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}
    symbol = str(candidate.get("symbol", "candidate")).strip().upper()

    raw_breaks = args.get("break_conditions", candidate.get("break_conditions", []))
    if not isinstance(raw_breaks, list) or not raw_breaks:
        raise ValueError("break_conditions list is required")

    checks: list[dict[str, Any]] = []
    triggered: list[str] = []
    for item in raw_breaks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "condition")).strip()
        current = _to_float(item.get("current"))
        threshold = _to_float(item.get("threshold"))
        direction = str(item.get("direction", "lt")).strip().lower()
        if direction == "gt":
            hit = current > threshold
        elif direction == "abs_gt":
            hit = abs(current) > abs(threshold)
        else:
            hit = current < threshold
        checks.append(
            {
                "name": name,
                "direction": direction,
                "current": current,
                "threshold": threshold,
                "triggered": hit,
            }
        )
        if hit:
            triggered.append(name)

    return {
        "operation": "thesis_breakers",
        "symbol": symbol,
        "checks": checks,
        "triggered_count": len(triggered),
        "triggered": triggered,
        "thesis_status": "broken" if triggered else "intact",
        "confidence": 0.7 if checks else 0.25,
        "keyless": True,
    }


def _to_top_k(value: object, *, default: int) -> int:
    try:
        if value is None:
            return default
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(200, parsed))


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Opportunity Ranker Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["OpportunityRankerTool"]
