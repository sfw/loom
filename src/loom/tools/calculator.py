"""Calculator tool — safe arithmetic and financial formula evaluation.

Evaluates mathematical expressions in a restricted sandbox.
Supports basic arithmetic, common math functions, and financial
formulas (NPV, IRR approximation, CAGR, WACC, etc.).
"""

from __future__ import annotations

import math
import operator
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

# Allowed operators and functions for safe evaluation
_SAFE_OPS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    # Math functions
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "pow": pow,
    "ceil": math.ceil,
    "floor": math.floor,
    # Constants
    "pi": math.pi,
    "e": math.e,
    # Operators (used by AST evaluator)
    "true": True,
    "false": False,
}


def _safe_eval(expr: str) -> float | int:
    """Evaluate a mathematical expression safely using AST."""
    import ast

    # Parse the expression
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}") from e

    return _eval_node(tree.body)


def _eval_node(node: Any) -> Any:
    """Recursively evaluate an AST node."""
    import ast

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.Name):
        if node.id in _SAFE_OPS:
            return _SAFE_OPS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
        op_func = ops.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        return op_func(operand)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        op_func = ops.get(type(node.op))
        if op_func is None:
            raise ValueError(
                f"Unsupported binary operator: {type(node.op)}",
            )
        return op_func(left, right)

    if isinstance(node, ast.Call):
        func = _eval_node(node.func)
        if not callable(func):
            raise ValueError(f"Not callable: {func}")
        args = [_eval_node(a) for a in node.args]
        return func(*args)

    if isinstance(node, ast.List):
        return [_eval_node(el) for el in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(el) for el in node.elts)

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def npv(rate: float, cashflows: list[float]) -> float:
    """Net Present Value."""
    return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))


def cagr(
    beginning_value: float, ending_value: float, years: float,
) -> float:
    """Compound Annual Growth Rate."""
    if beginning_value <= 0 or years <= 0:
        raise ValueError("Beginning value and years must be positive")
    return (ending_value / beginning_value) ** (1 / years) - 1


def wacc(
    equity: float, debt: float,
    cost_of_equity: float, cost_of_debt: float,
    tax_rate: float,
) -> float:
    """Weighted Average Cost of Capital."""
    total = equity + debt
    if total <= 0:
        raise ValueError("Total capital must be positive")
    return (
        (equity / total) * cost_of_equity
        + (debt / total) * cost_of_debt * (1 - tax_rate)
    )


def pmt(rate: float, nper: int, pv: float) -> float:
    """Payment for a loan with constant payments and interest rate."""
    if rate == 0:
        return -pv / nper
    return -pv * rate * (1 + rate) ** nper / ((1 + rate) ** nper - 1)


# Add financial functions to safe ops
_SAFE_OPS.update({
    "npv": npv,
    "cagr": cagr,
    "wacc": wacc,
    "pmt": pmt,
})


class CalculatorTool(Tool):
    """Evaluate mathematical and financial expressions safely."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate mathematical expressions and financial formulas. "
            "Supports: arithmetic (+, -, *, /, **, //), math functions "
            "(sqrt, log, exp, ceil, floor, abs, round, min, max, sum), "
            "and financial functions: npv(rate, [cashflows]), "
            "cagr(start, end, years), "
            "wacc(equity, debt, cost_equity, cost_debt, tax_rate), "
            "pmt(rate, nper, pv). "
            "Examples: '1000 * 1.05 ** 10', 'npv(0.1, [-1000, 300, 400, 500])', "
            "'cagr(100, 200, 5)'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Mathematical expression to evaluate. "
                        "Use Python syntax."
                    ),
                },
            },
            "required": ["expression"],
        }

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        expression = args.get("expression", "")
        if not expression:
            return ToolResult.fail("No expression provided")

        # Safety: reject overly long expressions
        if len(expression) > 1000:
            return ToolResult.fail("Expression too long (max 1000 chars)")

        try:
            result = _safe_eval(expression)
            # Format nicely
            if isinstance(result, float):
                # Avoid floating point noise
                if result == int(result) and abs(result) < 1e15:
                    formatted = str(int(result))
                else:
                    formatted = f"{result:.10g}"
            else:
                formatted = str(result)
            return ToolResult.ok(
                f"{expression} = {formatted}",
                data={"result": result, "expression": expression},
            )
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return ToolResult.fail(f"Evaluation error: {e}")
        except Exception as e:
            return ToolResult.fail(f"Calculator error: {type(e).__name__}: {e}")
