"""Orchestrator planning tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config, LimitsConfig, RunnerLimitsConfig
from loom.engine.orchestrator import Orchestrator, SubtaskResult
from loom.engine.verification import VerificationResult
from loom.events.types import (
    ASK_USER_ANSWERED,
    ASK_USER_REQUESTED,
    REMEDIATION_QUEUED,
    REMEDIATION_RESOLVED,
    SUBTASK_BLOCKED,
    TASK_CANCELLED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PAUSED,
    TASK_PLAN_DEGRADED,
    TASK_PLAN_READY,
    TASK_PLANNING,
    TASK_REPLANNING,
    TASK_RESUMED,
    TASK_STALLED,
    TELEMETRY_RUN_SUMMARY,
    VERIFICATION_FAILED,
    VERIFICATION_OUTCOME,
    VERIFICATION_PASSED,
    VERIFICATION_STARTED,
)
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


class TestOrchestratorPlan:
    """Tests for the planning phase."""

    @pytest.mark.asyncio
    async def test_plan_with_valid_json(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {"id": "step-1", "description": "Create file", "model_tier": 1},
                {"id": "step-2", "description": "Verify file", "depends_on": ["step-1"]},
            ]
        })

        bus = _make_event_bus()
        events_received = []
        bus.subscribe_all(lambda e: events_received.append(e))

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert len(result.plan.subtasks) == 2
        assert result.plan.subtasks[0].id == "step-1"

        # Check events
        event_types = [e.event_type for e in events_received]
        assert TASK_PLANNING in event_types
        assert TASK_PLAN_READY in event_types
        assert TASK_EXECUTING in event_types
        assert TASK_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_plan_uses_planning_response_token_limit(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {"id": "step-1", "description": "Create file"},
            ]
        })
        planner_max_tokens: list[int | None] = []

        planner_model = AsyncMock()
        planner_model.name = "mock-planner"
        planner_model.complete = AsyncMock(side_effect=lambda _messages, **kwargs: (
            planner_max_tokens.append(kwargs.get("max_tokens")),
            ModelResponse(
                text=plan_json,
                usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            ),
        )[1])

        executor_model = AsyncMock()
        executor_model.name = "mock-executor"
        executor_model.complete = AsyncMock(return_value=ModelResponse(
            text="done",
            usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        ))

        router = MagicMock(spec=ModelRouter)

        def select_fn(tier=1, role="executor"):
            del tier
            return planner_model if role == "planner" else executor_model

        router.select = MagicMock(side_effect=select_fn)

        orch = Orchestrator(
            model_router=router,
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=Config(limits=LimitsConfig(planning_response_max_tokens=16384)),
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert planner_max_tokens
        assert planner_max_tokens[0] == 16384

    @pytest.mark.asyncio
    async def test_emits_telemetry_run_summary_when_enabled(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Emit telemetry"}]
        })
        bus = _make_event_bus()
        events_received = []
        bus.subscribe_all(lambda e: events_received.append(e))

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=Config(
                limits=LimitsConfig(
                    runner=RunnerLimitsConfig(
                        enable_artifact_telemetry_events=True,
                    ),
                ),
            ),
        )
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="success",
                summary="ok",
                telemetry_counters={
                    "artifact_ingests": 2,
                    "artifact_reads": 1,
                    "artifact_retention_deletes": 3,
                    "compaction_policy_decisions": 4,
                    "overflow_fallback_count": 1,
                    "compactor_warning_count": 2,
                },
            ),
            VerificationResult(tier=1, passed=True),
        ))

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        summary_event = next(
            event for event in events_received
            if event.event_type == TELEMETRY_RUN_SUMMARY
        )
        assert summary_event.data["artifact_ingests"] == 2
        assert summary_event.data["artifact_reads"] == 1
        assert summary_event.data["artifact_retention_deletes"] == 3
        assert summary_event.data["compaction_policy_decisions"] == 4
        assert summary_event.data["overflow_fallback_count"] == 1
        assert summary_event.data["compactor_warning_count"] == 2
        assert "output_conflict_counts" in summary_event.data
        assert summary_event.data["output_conflict_counts"][
            "forbidden_canonical_write_blocked"
        ] == 0
        assert "run_id" in summary_event.data
        assert "budget_snapshot" in summary_event.data

    @pytest.mark.asyncio
    async def test_telemetry_run_summary_reconciles_event_counters(self, tmp_path):
        bus = _make_event_bus()
        events_received = []
        bus.subscribe_all(lambda e: events_received.append(e))

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )
        task = _make_task()
        run_id = orch._initialize_task_run_id(task)

        orch._emit(VERIFICATION_STARTED, task.id, {
            "run_id": run_id,
            "subtask_id": "s1",
            "target_tier": 1,
        })
        orch._emit(VERIFICATION_OUTCOME, task.id, {
            "run_id": run_id,
            "subtask_id": "s1",
            "tier": 1,
            "passed": False,
            "outcome": "fail",
            "reason_code": "hard_invariant_failed",
            "severity_class": "hard_invariant",
            "confidence": 0.0,
        })
        orch._emit(VERIFICATION_FAILED, task.id, {
            "run_id": run_id,
            "subtask_id": "s1",
            "tier": 1,
            "outcome": "fail",
            "reason_code": "hard_invariant_failed",
        })
        orch._emit(REMEDIATION_QUEUED, task.id, {
            "run_id": run_id,
            "subtask_id": "s1",
        })
        orch._emit(REMEDIATION_RESOLVED, task.id, {
            "run_id": run_id,
            "subtask_id": "s1",
        })
        orch._emit(ASK_USER_REQUESTED, task.id, {"run_id": run_id})
        orch._emit(ASK_USER_ANSWERED, task.id, {"run_id": run_id})
        orch._emit(TASK_PAUSED, task.id, {"run_id": run_id, "requested": True})
        orch._emit(TASK_RESUMED, task.id, {"run_id": run_id, "requested": True})
        orch._emit(SUBTASK_BLOCKED, task.id, {
            "run_id": run_id,
            "subtask_id": "s1",
            "reasons": ["dependency_unmet"],
        })
        orch._emit(TASK_PLAN_DEGRADED, task.id, {"run_id": run_id})
        orch._emit(TASK_REPLANNING, task.id, {"run_id": run_id, "reason": "retriable"})
        orch._emit(TASK_STALLED, task.id, {"run_id": run_id})

        orch._emit_telemetry_run_summary(task)
        orch._emit_telemetry_run_summary(task)

        summary_events = [
            event for event in events_received
            if event.event_type == TELEMETRY_RUN_SUMMARY
        ]
        assert len(summary_events) == 1
        summary = summary_events[0].data

        assert summary["verification_lifecycle_counts"] == {
            "started": 1,
            "passed": 0,
            "failed": 1,
            "outcome": 1,
        }
        assert summary["verification_reason_counts"] == {"hard_invariant_failed": 1}
        assert summary["remediation_lifecycle_counts"]["queued"] == 1
        assert summary["remediation_lifecycle_counts"]["resolved"] == 1
        assert summary["human_loop_counts"]["ask_user_requested"] == 1
        assert summary["human_loop_counts"]["ask_user_answered"] == 1
        assert summary["control_plane_counts"]["paused"] == 1
        assert summary["control_plane_counts"]["resumed"] == 1
        assert summary["blocked_indicator"] is True
        assert summary["degraded_indicator"] is True
        assert summary["replanned_count"] == 1
        assert summary["stalled_count"] == 1

    @pytest.mark.asyncio
    async def test_lifecycle_events_follow_partial_order(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Order assertions"}]
        })
        bus = _make_event_bus()
        events_received = []
        bus.subscribe_all(lambda e: events_received.append(e))

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )

        task = _make_task()
        result = await orch.execute_task(task)
        assert result.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

        event_types = [event.event_type for event in events_received]
        planning_ix = event_types.index(TASK_PLANNING)
        ready_ix = event_types.index(TASK_PLAN_READY)
        executing_ix = event_types.index(TASK_EXECUTING)
        assert planning_ix < ready_ix < executing_ix

        terminal_task_ix = min(
            event_types.index(event_type)
            for event_type in (TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED)
            if event_type in event_types
        )
        assert executing_ix < terminal_task_ix

        if VERIFICATION_STARTED in event_types:
            verification_start_ix = event_types.index(VERIFICATION_STARTED)
            assert executing_ix < verification_start_ix < terminal_task_ix
            verification_terminal_candidates = [
                event_types.index(event_type)
                for event_type in (VERIFICATION_PASSED, VERIFICATION_FAILED)
                if event_type in event_types
            ]
            if verification_terminal_candidates:
                verification_terminal_ix = min(verification_terminal_candidates)
                assert verification_start_ix < verification_terminal_ix < terminal_task_ix

    @pytest.mark.asyncio
    async def test_execute_task_reuses_existing_plan_when_requested(self, tmp_path):
        bus = _make_event_bus()
        events_received = []
        bus.subscribe_all(lambda e: events_received.append(e))

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text="unused"),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )
        orch._plan_task = AsyncMock(side_effect=AssertionError("planner should not run"))

        task = _make_task()
        task.status = TaskStatus.FAILED
        task.plan = Plan(subtasks=[
            Subtask(
                id="done",
                description="Already complete",
                status=SubtaskStatus.COMPLETED,
            ),
        ])

        result = await orch.execute_task(task, reuse_existing_plan=True)

        assert result.status == TaskStatus.COMPLETED
        orch._plan_task.assert_not_awaited()
        event_types = [e.event_type for e in events_received]
        assert TASK_PLAN_READY in event_types
        assert TASK_PLANNING not in event_types
        ready_event = next(e for e in events_received if e.event_type == TASK_PLAN_READY)
        assert ready_event.data.get("reused") is True

    @pytest.mark.asyncio
    async def test_plan_fallback_on_invalid_json(self, tmp_path):
        """When the planner returns invalid JSON, a single-step fallback plan is used."""
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text="This is not JSON"),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

        task = _make_task()
        result = await orch.execute_task(task)

        # Should still complete with a fallback plan
        assert result.status == TaskStatus.COMPLETED
        assert len(result.plan.subtasks) == 1
        assert result.plan.subtasks[0].id == "execute-goal"
