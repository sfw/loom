"""Google Analytics metrics calculation tool.

Provides GA4-specific calculations: conversion rate lifts, funnel
drop-off analysis, statistical significance testing, and attribution
model comparisons.
"""

from __future__ import annotations

import math
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult


def conversion_rate(conversions: int, sessions: int) -> float:
    """Calculate conversion rate as a percentage."""
    if sessions <= 0:
        raise ValueError("Sessions must be positive")
    return (conversions / sessions) * 100


def conversion_lift(
    baseline_rate: float, variant_rate: float,
) -> float:
    """Calculate relative lift between two conversion rates."""
    if baseline_rate <= 0:
        raise ValueError("Baseline rate must be positive")
    return ((variant_rate - baseline_rate) / baseline_rate) * 100


def funnel_dropoff(
    step_visitors: list[int],
) -> list[dict[str, Any]]:
    """Analyze funnel drop-off between steps.

    Returns a list of dicts with step number, visitors, drop-off count,
    drop-off rate, and cumulative conversion from step 1.
    """
    if not step_visitors or len(step_visitors) < 2:
        raise ValueError("Need at least 2 funnel steps")
    if any(v < 0 for v in step_visitors):
        raise ValueError("Visitor counts cannot be negative")
    if step_visitors[0] <= 0:
        raise ValueError("First step must have positive visitors")

    results = []
    for i, visitors in enumerate(step_visitors):
        if i == 0:
            results.append({
                "step": i + 1,
                "visitors": visitors,
                "dropoff": 0,
                "dropoff_rate": 0.0,
                "cumulative_rate": 100.0,
            })
        else:
            prev = step_visitors[i - 1]
            dropoff = prev - visitors
            dropoff_rate = (dropoff / prev * 100) if prev > 0 else 0.0
            cumulative = (visitors / step_visitors[0]) * 100
            results.append({
                "step": i + 1,
                "visitors": visitors,
                "dropoff": dropoff,
                "dropoff_rate": round(dropoff_rate, 2),
                "cumulative_rate": round(cumulative, 2),
            })
    return results


def z_test_significance(
    control_conversions: int,
    control_sessions: int,
    variant_conversions: int,
    variant_sessions: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Two-proportion z-test for conversion rate significance.

    Returns whether the difference is statistically significant at the
    given confidence level.
    """
    if control_sessions <= 0 or variant_sessions <= 0:
        raise ValueError("Session counts must be positive")

    p1 = control_conversions / control_sessions
    p2 = variant_conversions / variant_sessions
    p_pool = (
        (control_conversions + variant_conversions)
        / (control_sessions + variant_sessions)
    )

    se = math.sqrt(
        p_pool * (1 - p_pool)
        * (1 / control_sessions + 1 / variant_sessions),
    )

    if se == 0:
        return {
            "z_score": 0.0,
            "p_value": 1.0,
            "significant": False,
            "confidence": confidence,
            "control_rate": round(p1 * 100, 4),
            "variant_rate": round(p2 * 100, 4),
            "lift_pct": 0.0,
        }

    z = (p2 - p1) / se

    # Approximate two-tailed p-value using the error function
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    # Z critical values for common confidence levels
    z_critical = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    threshold = z_critical.get(confidence, 1.96)

    lift = conversion_lift(p1 * 100, p2 * 100) if p1 > 0 else 0.0

    return {
        "z_score": round(z, 4),
        "p_value": round(p_value, 6),
        "significant": abs(z) > threshold,
        "confidence": confidence,
        "control_rate": round(p1 * 100, 4),
        "variant_rate": round(p2 * 100, 4),
        "lift_pct": round(lift, 2),
    }


def attribution_compare(
    channel: str,
    last_click: float,
    data_driven: float,
    linear: float | None = None,
) -> dict[str, Any]:
    """Compare attribution model credits for a single channel.

    Returns credit amounts and shifts relative to last-click baseline.
    """
    result: dict[str, Any] = {
        "channel": channel,
        "last_click": last_click,
        "data_driven": data_driven,
        "dd_shift_pct": (
            round((data_driven - last_click) / last_click * 100, 2)
            if last_click > 0 else 0.0
        ),
    }
    if linear is not None:
        result["linear"] = linear
        result["linear_shift_pct"] = (
            round((linear - last_click) / last_click * 100, 2)
            if last_click > 0 else 0.0
        )
    return result


class GAMetricsTool(Tool):
    """Google Analytics metrics calculations."""

    @property
    def name(self) -> str:
        return "ga_metrics"

    @property
    def description(self) -> str:
        return (
            "GA4-specific calculations: conversion_rate(conversions, sessions), "
            "conversion_lift(baseline_rate, variant_rate), "
            "funnel_dropoff([step1_visitors, step2_visitors, ...]), "
            "z_test(control_conv, control_sessions, variant_conv, variant_sessions), "
            "attribution_compare(channel, last_click, data_driven, linear)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "conversion_rate",
                        "conversion_lift",
                        "funnel_dropoff",
                        "z_test",
                        "attribution_compare",
                    ],
                    "description": "The calculation to perform.",
                },
                "args": {
                    "type": "object",
                    "description": "Arguments for the calculation.",
                },
            },
            "required": ["operation", "args"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = args.get("operation", "")
        op_args = args.get("args", {})

        try:
            if operation == "conversion_rate":
                rate = conversion_rate(
                    int(op_args["conversions"]),
                    int(op_args["sessions"]),
                )
                return ToolResult.ok(
                    f"Conversion rate: {rate:.2f}%",
                    data={"rate": rate},
                )

            elif operation == "conversion_lift":
                lift = conversion_lift(
                    float(op_args["baseline_rate"]),
                    float(op_args["variant_rate"]),
                )
                return ToolResult.ok(
                    f"Relative lift: {lift:+.2f}%",
                    data={"lift_pct": lift},
                )

            elif operation == "funnel_dropoff":
                steps = [int(v) for v in op_args["step_visitors"]]
                analysis = funnel_dropoff(steps)
                lines = ["Funnel Analysis:"]
                for s in analysis:
                    lines.append(
                        f"  Step {s['step']}: {s['visitors']} visitors "
                        f"(drop-off: {s['dropoff_rate']}%, "
                        f"cumulative: {s['cumulative_rate']}%)",
                    )
                return ToolResult.ok(
                    "\n".join(lines),
                    data={"steps": analysis},
                )

            elif operation == "z_test":
                result = z_test_significance(
                    int(op_args["control_conversions"]),
                    int(op_args["control_sessions"]),
                    int(op_args["variant_conversions"]),
                    int(op_args["variant_sessions"]),
                    float(op_args.get("confidence", 0.95)),
                )
                sig = "YES" if result["significant"] else "NO"
                return ToolResult.ok(
                    f"Significance test: {sig} "
                    f"(z={result['z_score']}, p={result['p_value']}, "
                    f"lift={result['lift_pct']}%)",
                    data=result,
                )

            elif operation == "attribution_compare":
                result = attribution_compare(
                    str(op_args["channel"]),
                    float(op_args["last_click"]),
                    float(op_args["data_driven"]),
                    float(op_args["linear"])
                    if "linear" in op_args else None,
                )
                return ToolResult.ok(
                    f"{result['channel']}: DD shift "
                    f"{result['dd_shift_pct']:+.1f}% vs last-click",
                    data=result,
                )

            else:
                return ToolResult.fail(
                    f"Unknown operation: {operation}. "
                    f"Use: conversion_rate, conversion_lift, "
                    f"funnel_dropoff, z_test, attribution_compare",
                )

        except (KeyError, TypeError) as e:
            return ToolResult.fail(f"Missing or invalid argument: {e}")
        except (ValueError, ZeroDivisionError) as e:
            return ToolResult.fail(f"Calculation error: {e}")
