"""Orchestrator iteration loop tests."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock

import pytest

from loom.config import Config, ExecutionConfig
from loom.engine.orchestrator import Orchestrator, SubtaskResult, ToolCallRecord
from loom.engine.verification import VerificationResult
from loom.events.types import ITERATION_COMPLETED, ITERATION_RETRYING
from loom.processes.schema import (
    IterationBudget,
    IterationGate,
    IterationPolicy,
    PhaseTemplate,
    ProcessDefinition,
)
from loom.state.task_state import Plan, Subtask, TaskStatus
from loom.tools.registry import ToolResult
from tests.orchestrator.conftest import (
    _make_event_bus,
    _make_mock_memory,
    _make_mock_prompts,
    _make_mock_router,
    _make_mock_tools,
    _make_state_manager,
    _make_task,
)


class TestIterationLoops:
    def _make_iteration_process(
        self,
        *,
        max_attempts: int = 4,
        max_replans_after_exhaustion: int = 2,
        strategy: str = "targeted_remediation",
        deliverables: list[str] | None = None,
    ) -> ProcessDefinition:
        return ProcessDefinition(
            name="iter-process",
            phase_mode="strict",
            phases=[
                PhaseTemplate(
                    id="rewrite",
                    description="Rewrite the draft",
                    deliverables=list(deliverables or []),
                    iteration=IterationPolicy(
                        enabled=True,
                        max_attempts=max_attempts,
                        strategy=strategy,
                        max_total_runner_invocations=6,
                        max_replans_after_exhaustion=max_replans_after_exhaustion,
                        stop_on_no_improvement_attempts=2,
                        budget=IterationBudget(
                            max_wall_clock_seconds=600,
                            max_tokens=100000,
                            max_tool_calls=20,
                        ),
                        gates=[
                            IterationGate(
                                id="score",
                                type="tool_metric",
                                blocking=True,
                                tool="humanize_writing",
                                metric_path="report.humanization_score",
                                operator="gte",
                                value=80,
                            ),
                        ],
                    ),
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_iteration_gate_failure_retries_until_pass(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {"id": "rewrite", "description": "Rewrite draft"},
            ],
        })
        events = []
        bus = _make_event_bus()
        bus.subscribe_all(lambda e: events.append(e))

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=Config(
                execution=ExecutionConfig(enable_process_iteration_loops=True),
            ),
            process=self._make_iteration_process(),
        )

        first_result = SubtaskResult(
            status="success",
            summary="score too low",
            tool_calls=[
                ToolCallRecord(
                    tool="humanize_writing",
                    args={"operation": "evaluate"},
                    result=ToolResult.ok(
                        "ok",
                        data={"report": {"humanization_score": 72}},
                    ),
                ),
            ],
            telemetry_counters={"tool_calls": 1},
        )
        second_result = SubtaskResult(
            status="success",
            summary="score good",
            tool_calls=[
                ToolCallRecord(
                    tool="humanize_writing",
                    args={"operation": "evaluate"},
                    result=ToolResult.ok(
                        "ok",
                        data={"report": {"humanization_score": 86}},
                    ),
                ),
            ],
            telemetry_counters={"tool_calls": 1},
        )
        verification_ok = VerificationResult(
            tier=2,
            passed=True,
            outcome="pass",
            feedback="ok",
        )
        orch._runner.run = AsyncMock(
            side_effect=[
                (first_result, verification_ok),
                (second_result, verification_ok),
            ],
        )

        task = _make_task()
        result = await orch.execute_task(task)
        subtask = result.get_subtask("rewrite")

        assert result.status == TaskStatus.COMPLETED
        assert subtask is not None
        assert subtask.iteration_attempt == 2
        assert subtask.iteration_terminal_reason == "passed"
        assert subtask.iteration_runner_invocations == 2
        assert orch._runner.run.await_count == 2

        event_types = [event.event_type for event in events]
        assert ITERATION_RETRYING in event_types
        assert ITERATION_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_iteration_replan_fingerprint_blocks_repeat(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=Config(
                execution=ExecutionConfig(enable_process_iteration_loops=True),
            ),
            process=self._make_iteration_process(
                max_attempts=1,
                max_replans_after_exhaustion=1,
            ),
        )

        task = _make_task()
        task.status = TaskStatus.EXECUTING
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="rewrite",
                    description="Rewrite draft",
                    phase_id="rewrite",
                    max_retries=1,
                ),
            ],
        )
        subtask = task.plan.subtasks[0]
        result = SubtaskResult(
            status="success",
            summary="score too low",
            tool_calls=[
                ToolCallRecord(
                    tool="humanize_writing",
                    args={"operation": "evaluate"},
                    result=ToolResult.ok(
                        "ok",
                        data={"report": {"humanization_score": 60}},
                    ),
                ),
            ],
            telemetry_counters={"tool_calls": 1},
        )
        verification_ok = VerificationResult(tier=2, passed=True, outcome="pass")

        first = await orch._handle_iteration_after_success(
            task=task,
            subtask=subtask,
            result=result,
            verification=verification_ok,
        )
        second = await orch._handle_iteration_after_success(
            task=task,
            subtask=subtask,
            result=result,
            verification=verification_ok,
        )

        assert first is not None
        assert first["reason"] == "iteration_loop_exhausted:max_attempts_exhausted"
        assert second is None

    @pytest.mark.asyncio
    async def test_iteration_strategy_targeted_enforces_edit_existing_only(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(enable_process_iteration_loops=True)),
            process=self._make_iteration_process(deliverables=["draft.md"]),
        )
        task = _make_task(workspace=str(tmp_path))
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="rewrite",
                    description="Rewrite",
                    phase_id="rewrite",
                ),
            ],
        )
        subtask = task.plan.subtasks[0]
        subtask.iteration_attempt = 1
        subtask.iteration_last_gate_summary = "score too low"
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))

        await orch._dispatch_subtask(task, subtask, {})
        kwargs = orch._runner.run.await_args.kwargs
        assert kwargs["enforce_deliverable_paths"] is True
        assert kwargs["edit_existing_only"] is True

    @pytest.mark.asyncio
    async def test_iteration_strategy_full_rerun_allows_non_targeted_editing(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(enable_process_iteration_loops=True)),
            process=self._make_iteration_process(
                strategy="full_rerun",
                deliverables=["draft.md"],
            ),
        )
        task = _make_task(workspace=str(tmp_path))
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="rewrite",
                    description="Rewrite",
                    phase_id="rewrite",
                ),
            ],
        )
        subtask = task.plan.subtasks[0]
        subtask.iteration_attempt = 1
        subtask.iteration_last_gate_summary = "score too low"
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))

        await orch._dispatch_subtask(task, subtask, {})
        kwargs = orch._runner.run.await_args.kwargs
        assert kwargs["enforce_deliverable_paths"] is False
        assert kwargs["edit_existing_only"] is False

    @pytest.mark.asyncio
    async def test_iteration_failure_budget_exhaustion_uses_iteration_replan_path(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(enable_process_iteration_loops=True)),
            process=self._make_iteration_process(),
        )
        task = _make_task(workspace=str(tmp_path))
        task.status = TaskStatus.EXECUTING
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="rewrite",
                    description="Rewrite",
                    phase_id="rewrite",
                    max_retries=3,
                ),
            ],
        )
        subtask = task.plan.subtasks[0]
        runtime = orch._iteration_runtime_entry(task, subtask.id)
        runtime["started_monotonic"] = float(time.monotonic()) - 1
        runtime["tokens_used"] = 200_000
        runtime["tool_calls"] = 25

        result = SubtaskResult(status="failed", summary="execution failed", tokens_used=0)
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            feedback="failed",
            reason_code="verification_failed",
        )
        replan = await orch._handle_failure(task, subtask, result, verification, {})

        assert replan is not None
        assert replan["reason"] == "iteration_loop_exhausted:iteration_budget_exhausted"
        assert subtask.iteration_terminal_reason == "iteration_budget_exhausted"

    @pytest.mark.asyncio
    async def test_reconcile_iteration_state_hydrates_subtask_fields(self, tmp_path):
        memory = _make_mock_memory()
        memory.list_iteration_runs = AsyncMock(return_value=[
            {
                "loop_run_id": "iter-abc123",
                "task_id": "t1",
                "subtask_id": "rewrite",
                "phase_id": "rewrite",
                "policy_snapshot": {"max_attempts": 4},
                "terminal_reason": "max_attempts_exhausted",
                "attempt_count": 3,
                "replan_count": 1,
                "metadata": {
                    "iteration_runner_invocations": 4,
                    "iteration_no_improvement_count": 2,
                    "iteration_best_score": 82.0,
                    "iteration_last_gate_summary": "score gate failed",
                },
                "created_at": "2026-03-01T00:00:00",
                "updated_at": "2026-03-01T00:05:00",
            },
        ])
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=memory,
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(enable_process_iteration_loops=True)),
            process=self._make_iteration_process(),
        )
        task = _make_task(goal="Hydrate")
        task.plan = Plan(
            subtasks=[Subtask(id="rewrite", description="Rewrite", phase_id="rewrite")],
        )
        task.metadata["iteration_sqlite_mirror"] = {"run_count": 1}

        await orch._reconcile_iteration_state(task)
        subtask = task.get_subtask("rewrite")

        assert subtask is not None
        assert subtask.iteration_loop_run_id == "iter-abc123"
        assert subtask.iteration_attempt == 3
        assert subtask.iteration_replan_count == 1
        assert subtask.iteration_terminal_reason == "max_attempts_exhausted"
        assert subtask.iteration_runner_invocations == 4
        assert subtask.iteration_no_improvement_count == 2
        assert subtask.iteration_best_score == 82.0
        assert subtask.iteration_last_gate_summary == "score gate failed"
