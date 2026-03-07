"""Focused tests for extracted orchestrator run-budget helper."""

from __future__ import annotations

from types import SimpleNamespace

from loom.engine.orchestrator.budget import _RunBudget


def test_run_budget_disabled_never_exceeds() -> None:
    config = SimpleNamespace(
        execution=SimpleNamespace(enable_global_run_budget=False),
    )
    budget = _RunBudget(config)  # type: ignore[arg-type]

    exceeded, _, _, _ = budget.exceeded()

    assert exceeded is False


def test_run_budget_observe_result_tracks_counters() -> None:
    config = SimpleNamespace(
        execution=SimpleNamespace(
            enable_global_run_budget=True,
            max_task_wall_clock_seconds=0,
            max_task_total_tokens=10,
            max_task_model_invocations=1,
            max_task_tool_calls=1,
            max_task_mutating_tool_calls=1,
            max_task_replans=0,
            max_task_remediation_attempts=0,
        ),
    )
    budget = _RunBudget(config)  # type: ignore[arg-type]
    result = SimpleNamespace(
        tokens_used=11,
        telemetry_counters={
            "model_invocations": 2,
            "tool_calls": 2,
            "mutating_tool_calls": 2,
        },
    )

    budget.observe_result(result)  # type: ignore[arg-type]
    exceeded, budget_name, observed, limit = budget.exceeded()

    assert exceeded is True
    assert budget_name in {
        "max_task_total_tokens",
        "max_task_model_invocations",
        "max_task_tool_calls",
        "max_task_mutating_tool_calls",
    }
    assert observed > limit
