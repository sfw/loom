"""Orchestrator process phase mode tests."""

from __future__ import annotations

import json

import pytest

from loom.engine.orchestrator import Orchestrator
from loom.events.types import TASK_PLAN_NORMALIZED
from loom.models.base import ModelResponse, TokenUsage
from loom.processes.schema import PhaseTemplate, ProcessDefinition
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


class TestOrchestratorProcessPhaseMode:
    @pytest.mark.asyncio
    async def test_strict_phase_mode_overrides_planner_structure(self, tmp_path):
        """Strict process mode should force plan shape to declared phases."""
        plan_json = json.dumps({
            "subtasks": [
                {"id": "unexpected-step", "description": "Wrong output"},
                {"id": "implement", "description": "Partial match"},
            ]
        })
        process = ProcessDefinition(
            name="strict-proc",
            phase_mode="strict",
            phases=[
                PhaseTemplate(
                    id="research",
                    description="Research phase",
                    depends_on=[],
                    model_tier=1,
                    verification_tier=1,
                    acceptance_criteria="Gather context",
                ),
                PhaseTemplate(
                    id="implement",
                    description="Implementation phase",
                    depends_on=["research"],
                    model_tier=2,
                    verification_tier=1,
                    is_critical_path=True,
                    acceptance_criteria="Ship changes",
                ),
            ],
        )

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert [s.id for s in result.plan.subtasks] == ["research", "implement"]
        assert [s.phase_id for s in result.plan.subtasks] == ["research", "implement"]
        assert result.plan.subtasks[1].depends_on == ["research"]
        assert result.plan.subtasks[1].is_critical_path is True
        assert result.plan.subtasks[1].acceptance_criteria == "Ship changes"

    @pytest.mark.asyncio
    async def test_guided_phase_mode_keeps_planner_structure(self, tmp_path):
        """Guided mode should preserve planner-created subtask graph."""
        plan_json = json.dumps({
            "subtasks": [
                {"id": "model-step", "description": "Planner owns structure"},
            ]
        })
        process = ProcessDefinition(
            name="guided-proc",
            phase_mode="guided",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Phase A",
                ),
            ],
        )

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert [s.id for s in result.plan.subtasks] == ["model-step"]
        assert result.plan.subtasks[0].phase_id == "phase-a"

    @pytest.mark.asyncio
    async def test_guided_phase_mode_infers_phase_id_from_subtask_description(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {
                    "id": "build-market-sizing-model",
                    "description": "Estimate market sizing assumptions and TAM ranges.",
                },
            ]
        })
        process = ProcessDefinition(
            name="guided-proc",
            phase_mode="guided",
            phases=[
                PhaseTemplate(
                    id="market-sizing",
                    description="Estimate market sizing assumptions and TAM ranges.",
                    deliverables=["market-sizing.md"],
                ),
                PhaseTemplate(
                    id="risk-map",
                    description="Map risks and mitigations.",
                    deliverables=["risk-map.md"],
                ),
            ],
        )

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert [s.id for s in result.plan.subtasks] == ["build-market-sizing-model"]
        assert result.plan.subtasks[0].phase_id == "market-sizing"

    @pytest.mark.asyncio
    async def test_strict_phase_mode_propagates_synthesis_flag(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {"id": "custom-step", "description": "Planner output"},
            ]
        })
        process = ProcessDefinition(
            name="strict-synth",
            phase_mode="strict",
            phases=[
                PhaseTemplate(
                    id="analyze",
                    description="Analyze inputs",
                ),
                PhaseTemplate(
                    id="synthesize",
                    description="Synthesize final output",
                    depends_on=["analyze"],
                    is_synthesis=True,
                ),
            ],
        )

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert [s.id for s in result.plan.subtasks] == ["analyze", "synthesize"]
        assert result.plan.subtasks[1].is_synthesis is True

    @pytest.mark.asyncio
    async def test_guided_plan_normalizes_non_terminal_synthesis(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {"id": "gather-data", "description": "Gather"},
                {
                    "id": "intermediate-synth",
                    "description": "Intermediate synthesis",
                    "depends_on": ["gather-data"],
                    "is_synthesis": True,
                },
                {
                    "id": "post-step",
                    "description": "Downstream step",
                    "depends_on": ["intermediate-synth"],
                },
            ],
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
        assert result.get_subtask("intermediate-synth") is not None
        assert result.get_subtask("intermediate-synth").is_synthesis is False
        assert TASK_PLAN_NORMALIZED in [e.event_type for e in events_received]

    @pytest.mark.asyncio
    async def test_strict_mode_retries_planner_on_topology_rejection(self, tmp_path):
        invalid_plan_json = json.dumps({
            "subtasks": [
                {"id": "prep", "description": "Prepare data"},
                {
                    "id": "synth",
                    "description": "Intermediate synth",
                    "depends_on": ["prep"],
                    "is_synthesis": True,
                },
                {"id": "post", "description": "Post", "depends_on": ["synth"]},
            ],
        })
        valid_plan_json = json.dumps({
            "subtasks": [
                {"id": "prep", "description": "Prepare data"},
                {
                    "id": "synth",
                    "description": "Final synth",
                    "depends_on": ["prep"],
                    "is_synthesis": True,
                },
            ],
        })
        planner_responses = [
            ModelResponse(
                text=invalid_plan_json,
                usage=TokenUsage(input_tokens=100, output_tokens=60, total_tokens=160),
            ),
            ModelResponse(
                text=valid_plan_json,
                usage=TokenUsage(input_tokens=100, output_tokens=60, total_tokens=160),
            ),
        ]
        process = ProcessDefinition(name="strict-topology", phase_mode="strict")
        router = _make_mock_router(
            planner_responses=planner_responses,
        )

        orch = Orchestrator(
            model_router=router,
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )

        result = await orch.execute_task(_make_task())

        assert result.status == TaskStatus.COMPLETED
        planner_model = router.select(tier=2, role="planner")
        assert planner_model.complete.await_count == 2

    @pytest.mark.asyncio
    async def test_strict_mode_retries_replanner_on_topology_rejection(self, tmp_path):
        invalid_replan_json = json.dumps({
            "subtasks": [
                {"id": "prep", "description": "Prepare data"},
                {
                    "id": "synth",
                    "description": "Intermediate synth",
                    "depends_on": ["prep"],
                    "is_synthesis": True,
                },
                {"id": "post", "description": "Downstream", "depends_on": ["synth"]},
            ],
        })
        valid_replan_json = json.dumps({
            "subtasks": [
                {"id": "prep", "description": "Prepare data"},
                {
                    "id": "synth",
                    "description": "Final synth",
                    "depends_on": ["prep"],
                    "is_synthesis": True,
                },
            ],
        })
        planner_responses = [
            ModelResponse(
                text=invalid_replan_json,
                usage=TokenUsage(input_tokens=100, output_tokens=60, total_tokens=160),
            ),
            ModelResponse(
                text=valid_replan_json,
                usage=TokenUsage(input_tokens=100, output_tokens=60, total_tokens=160),
            ),
        ]
        process = ProcessDefinition(name="strict-topology", phase_mode="strict")
        router = _make_mock_router(
            planner_responses=planner_responses,
        )
        orch = Orchestrator(
            model_router=router,
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )

        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="prep",
                    description="Prepare",
                    status=SubtaskStatus.COMPLETED,
                ),
                Subtask(
                    id="synth",
                    description="Synthesize",
                    status=SubtaskStatus.PENDING,
                    depends_on=["prep"],
                    is_synthesis=True,
                ),
            ],
            version=1,
        )

        replanned = await orch._replan_task(
            task,
            reason="scheduler_deadlock",
            verification_feedback="Blocked subtasks: synth",
        )

        assert replanned is True
        assert task.plan.version == 2
        assert task.get_subtask("prep") is not None
        assert task.get_subtask("prep").status == SubtaskStatus.COMPLETED
        planner_model = router.select(tier=2, role="planner")
        assert planner_model.complete.await_count == 2
