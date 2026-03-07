"""Orchestrator critical path behavior tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config, ExecutionConfig
from loom.engine.orchestrator import Orchestrator, SubtaskResult
from loom.engine.verification import VerificationResult
from loom.events.types import TASK_REPLAN_REJECTED
from loom.models.base import ModelResponse, TokenUsage
from loom.models.router import ModelRouter
from loom.state.task_state import Plan, Subtask, SubtaskStatus, TaskStatus
from tests.orchestrator.conftest import (
    _make_config,
    _make_event_bus,
    _make_mock_memory,
    _make_mock_prompts,
    _make_mock_router,
    _make_mock_tools,
    _make_state_manager,
    _make_task,
)


class TestOrchestratorCriticalPathBehavior:
    @pytest.mark.asyncio
    async def test_critical_path_failure_exhausted_retries_aborts_without_replan(
        self, tmp_path
    ):
        plan_json = json.dumps({
            "subtasks": [
                {
                    "id": "critical-step",
                    "description": "Must succeed",
                    "is_critical_path": True,
                },
                {
                    "id": "later-step",
                    "description": "Depends on critical",
                    "depends_on": ["critical-step"],
                },
            ]
        })
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=50,
            max_parallel_subtasks=3,
            auto_approve_confidence_threshold=0.8,
            enable_streaming=False,
        ))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=cfg,
        )
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="failed", summary="failed"),
            VerificationResult(tier=1, passed=False, feedback="failed checks"),
        ))
        orch._replan_task = AsyncMock(return_value=True)

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.FAILED
        assert result.get_subtask("critical-step").status == SubtaskStatus.FAILED
        assert result.get_subtask("later-step").status == SubtaskStatus.SKIPPED
        orch._replan_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_critical_failure_still_triggers_replan(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {"id": "normal-step", "description": "Can replan"},
            ]
        })
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=50,
            max_parallel_subtasks=3,
            auto_approve_confidence_threshold=0.8,
            enable_streaming=False,
        ))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=cfg,
        )
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="failed", summary="failed"),
            VerificationResult(tier=1, passed=False, feedback="failed checks"),
        ))
        orch._replan_task = AsyncMock(return_value=False)

        task = _make_task()
        await orch.execute_task(task)

        orch._replan_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_replan_is_deferred_until_batch_outcomes_are_applied(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {"id": "s1", "description": "Fails"},
                {"id": "s2", "description": "Succeeds"},
            ]
        })
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=50,
            max_parallel_subtasks=2,
            auto_approve_confidence_threshold=0.8,
            enable_streaming=False,
        ))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=cfg,
        )

        outcomes_by_id = {
            "s1": (
                SubtaskResult(status="failed", summary="failed"),
                VerificationResult(tier=1, passed=False, feedback="failed checks"),
            ),
            "s2": (
                SubtaskResult(status="success", summary="ok"),
                VerificationResult(tier=1, passed=True),
            ),
        }

        async def dispatch_side_effect(_task, subtask, _attempts_by_subtask):
            result, verification = outcomes_by_id[subtask.id]
            return subtask, result, verification

        orch._dispatch_subtask = AsyncMock(side_effect=dispatch_side_effect)

        call_order: list[str] = []
        original_handle_success = orch._handle_success

        async def success_side_effect(*args, **kwargs):
            call_order.append("success")
            return await original_handle_success(*args, **kwargs)

        async def replan_side_effect(*_args, **_kwargs):
            call_order.append("replan")
            assert "success" in call_order
            return False

        orch._handle_success = AsyncMock(side_effect=success_side_effect)
        orch._replan_task = AsyncMock(side_effect=replan_side_effect)

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.FAILED
        assert call_order == ["success", "replan"]
        orch._replan_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_replan_task_rejects_id_churn_without_mutating_plan(self, tmp_path):
        replanned_json = json.dumps({
            "subtasks": [
                {"id": "kept", "description": "Still here"},
                {"id": "renamed-id", "description": "Dropped previous id"},
            ]
        })
        router = MagicMock(spec=ModelRouter)
        planner_model = AsyncMock()
        planner_model.name = "mock-planner"
        planner_model.complete = AsyncMock(return_value=ModelResponse(
            text=replanned_json,
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        ))
        router.select = MagicMock(return_value=planner_model)

        prompts = _make_mock_prompts()
        prompts.build_replanner_prompt = MagicMock(return_value="Replan this task")

        bus = _make_event_bus()
        events_received = []
        bus.subscribe_all(lambda e: events_received.append(e))

        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=router,
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=prompts,
            state_manager=state_manager,
            event_bus=bus,
            config=_make_config(),
        )

        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(id="kept", description="Done", status=SubtaskStatus.COMPLETED),
                Subtask(id="must-stay", description="Failed"),
            ],
            version=3,
        )
        state_manager.create(task)

        replanned = await orch._replan_task(
            task,
            reason="test_replan_contract",
            failed_subtask_id="must-stay",
        )

        assert replanned is False
        assert [s.id for s in task.plan.subtasks] == ["kept", "must-stay"]
        assert task.plan.version == 3
        reject_events = [
            e for e in events_received
            if e.event_type == TASK_REPLAN_REJECTED
        ]
        assert len(reject_events) == 1
        assert "must-stay" in reject_events[0].data.get("validation_error", "")
