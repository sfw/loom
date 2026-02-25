"""Factor exposure and attribution tool."""

from __future__ import annotations

import json
from typing import Any

from loom.research.finance import (
    align_series_tail,
    clamp,
    correlation_matrix,
    covariance,
    mean_return,
)
from loom.tools.registry import Tool, ToolContext, ToolResult

_OPERATIONS = {"estimate_betas", "factor_contribution", "factor_correlation"}


class FactorExposureEngineTool(Tool):
    """Estimate factor betas and portfolio factor contributions."""

    @property
    def name(self) -> str:
        return "factor_exposure_engine"

    @property
    def description(self) -> str:
        return (
            "Estimate asset/portfolio factor exposures and factor correlation from return series."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["estimate_betas", "factor_contribution", "factor_correlation"],
                },
                "asset_returns": {
                    "type": "object",
                    "description": "Mapping asset -> list[period returns].",
                },
                "factor_returns": {
                    "type": "object",
                    "description": "Mapping factor -> list[period returns].",
                },
                "weights": {
                    "type": "object",
                    "description": "Portfolio weights mapping asset -> weight.",
                },
                "factor_premia": {
                    "type": "object",
                    "description": "Expected factor premia mapping factor -> annual value.",
                },
                "betas": {
                    "type": "object",
                    "description": "Optional precomputed betas mapping asset -> factor -> beta.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown output path.",
                },
            },
            "required": ["operation"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "")).strip().lower()
        if operation not in _OPERATIONS:
            return ToolResult.fail(
                "operation must be estimate_betas/factor_contribution/factor_correlation"
            )

        try:
            if operation == "estimate_betas":
                payload = _estimate_betas(args)
            elif operation == "factor_contribution":
                payload = _factor_contribution(args)
            else:
                payload = _factor_correlation(args)
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


def _estimate_betas(args: dict[str, Any]) -> dict[str, Any]:
    assets = _as_series_map(args.get("asset_returns"))
    factors = _as_series_map(args.get("factor_returns"))
    if not assets or not factors:
        raise ValueError("asset_returns and factor_returns are required")

    aligned_factors = align_series_tail(factors)
    if not aligned_factors:
        raise ValueError("factor_returns must contain at least one factor with >=2 values")

    betas: dict[str, dict[str, float]] = {}
    r2_proxy: dict[str, float] = {}
    for asset, asset_series in assets.items():
        betas[asset] = {}
        aligned_asset = align_series_tail({"asset": asset_series})
        asset_vals = aligned_asset.get("asset", [])
        if len(asset_vals) < 2:
            continue

        beta_abs_sum = 0.0
        for factor, factor_series in aligned_factors.items():
            n = min(len(asset_vals), len(factor_series))
            if n < 2:
                beta = 0.0
            else:
                x = asset_vals[-n:]
                f = factor_series[-n:]
                var_f = covariance(f, f)
                beta = covariance(x, f) / var_f if var_f > 1e-12 else 0.0
            betas[asset][factor] = beta
            beta_abs_sum += abs(beta)
        r2_proxy[asset] = clamp(beta_abs_sum / max(1.0, len(aligned_factors)), 0.0, 1.0)

    return {
        "operation": "estimate_betas",
        "betas": betas,
        "r2_proxy": r2_proxy,
        "asset_count": len(betas),
        "factor_count": len(aligned_factors),
        "confidence": clamp(0.35 + 0.1 * len(aligned_factors), 0.25, 0.9),
        "keyless": True,
    }


def _factor_contribution(args: dict[str, Any]) -> dict[str, Any]:
    betas = args.get("betas")
    if not isinstance(betas, dict) or not betas:
        betas = _estimate_betas(args).get("betas", {})
    weights_raw = args.get("weights", {})
    if not isinstance(weights_raw, dict) or not weights_raw:
        raise ValueError("weights are required")
    premia_raw = args.get("factor_premia", {})
    if not isinstance(premia_raw, dict) or not premia_raw:
        raise ValueError("factor_premia are required")

    weights: dict[str, float] = {}
    total_abs = 0.0
    for asset, value in weights_raw.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            continue
        weights[str(asset)] = weight
        total_abs += abs(weight)
    if total_abs <= 0:
        raise ValueError("weights cannot be all zero")

    factor_premia: dict[str, float] = {}
    for factor, value in premia_raw.items():
        try:
            factor_premia[str(factor)] = float(value)
        except (TypeError, ValueError):
            continue
    if not factor_premia:
        raise ValueError("factor_premia contains no numeric values")

    exposures: dict[str, float] = {factor: 0.0 for factor in factor_premia}
    for asset, weight in weights.items():
        asset_betas = betas.get(asset, {})
        if not isinstance(asset_betas, dict):
            continue
        for factor in exposures:
            exposures[factor] += weight * float(asset_betas.get(factor, 0.0))

    contributions = {factor: exposures[factor] * factor_premia[factor] for factor in exposures}
    explained_return = sum(contributions.values())
    return {
        "operation": "factor_contribution",
        "portfolio_factor_exposures": exposures,
        "factor_contributions": contributions,
        "explained_expected_return": explained_return,
        "weights": weights,
        "factor_premia": factor_premia,
        "keyless": True,
    }


def _factor_correlation(args: dict[str, Any]) -> dict[str, Any]:
    factors = _as_series_map(args.get("factor_returns"))
    if not factors:
        raise ValueError("factor_returns are required")
    order, matrix = correlation_matrix(factors)
    return {
        "operation": "factor_correlation",
        "factors": order,
        "correlation_matrix": matrix,
        "means": {factor: mean_return(series) for factor, series in factors.items()},
        "keyless": True,
    }


def _render_report(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Factor Exposure Engine Report",
            "",
            f"- **Operation**: {payload.get('operation', '')}",
            "",
            "```json",
            json.dumps(payload, indent=2),
            "```",
            "",
        ]
    )


__all__ = ["FactorExposureEngineTool"]
