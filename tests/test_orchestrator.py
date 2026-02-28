"""Tests for the orchestrator loop."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from loom.config import (
    Config,
    ExecutionConfig,
    LimitsConfig,
    RunnerLimitsConfig,
    VerificationConfig,
)
from loom.engine.orchestrator import Orchestrator, SubtaskResult, ToolCallRecord, create_task
from loom.engine.verification import VerificationResult
from loom.events.bus import EventBus
from loom.events.types import (
    ARTIFACT_CONFINEMENT_VIOLATION,
    ARTIFACT_INGEST_CLASSIFIED,
    ARTIFACT_INGEST_COMPLETED,
    ARTIFACT_READ_COMPLETED,
    ARTIFACT_RETENTION_PRUNED,
    COMPACTION_POLICY_DECISION,
    OVERFLOW_FALLBACK_APPLIED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLAN_NORMALIZED,
    TASK_PLAN_READY,
    TASK_PLANNING,
    TASK_REPLAN_REJECTED,
    TASK_STALLED,
    TELEMETRY_RUN_SUMMARY,
    TOOL_CALL_COMPLETED,
)
from loom.models.base import ModelConnectionError, ModelResponse, TokenUsage, ToolCall
from loom.models.router import ModelRouter
from loom.processes.schema import (
    PhaseTemplate,
    ProcessDefinition,
    VerificationRemediationContract,
)
from loom.prompts.assembler import PromptAssembler
from loom.recovery.retry import AttemptRecord, RetryStrategy
from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStateManager,
    TaskStatus,
)
from loom.tools.registry import ToolRegistry, ToolResult

# --- Fixtures ---


def _make_config() -> Config:
    return Config()


def _make_event_bus() -> EventBus:
    return EventBus()


def _make_state_manager(tmp_path) -> TaskStateManager:
    return TaskStateManager(data_dir=tmp_path)


def _make_mock_router(
    plan_response_text: str = "",
    executor_responses=None,
    planner_responses=None,
):
    """Create a mock router that returns prescribed responses."""
    router = MagicMock(spec=ModelRouter)

    planner_model = AsyncMock()
    planner_model.name = "mock-planner"
    if planner_responses is None:
        planner_model.complete = AsyncMock(return_value=ModelResponse(
            text=plan_response_text,
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        ))
    else:
        planner_model.complete = AsyncMock(side_effect=planner_responses)

    executor_model = AsyncMock()
    executor_model.name = "mock-executor"

    if executor_responses is None:
        # Default: single text-only response (no tool calls)
        executor_model.complete = AsyncMock(return_value=ModelResponse(
            text="Subtask completed successfully.",
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))
    else:
        executor_model.complete = AsyncMock(side_effect=executor_responses)

    def select_fn(tier=1, role="executor"):
        if role == "planner":
            return planner_model
        return executor_model

    router.select = MagicMock(side_effect=select_fn)
    return router


def _make_mock_tools():
    tools = MagicMock(spec=ToolRegistry)
    tools.execute = AsyncMock(return_value=ToolResult.ok("success"))
    tools.list_tools = MagicMock(return_value=["read_file", "write_file"])
    tools.all_schemas = MagicMock(return_value=[
        {"name": "read_file", "description": "Read a file"},
        {"name": "write_file", "description": "Write a file"},
    ])
    return tools


def _make_mock_memory():
    memory = AsyncMock()
    memory.query_relevant = AsyncMock(return_value=[])
    return memory


def _make_mock_prompts():
    prompts = MagicMock(spec=PromptAssembler)
    prompts.build_planner_prompt = MagicMock(return_value="Plan this task")
    prompts.build_executor_prompt = MagicMock(return_value="Execute this subtask")
    return prompts


def _make_task(goal="Write hello world", workspace="") -> Task:
    return create_task(goal=goal, workspace=workspace)


def _make_process_with_critical_behavior(behavior: str) -> ProcessDefinition:
    return ProcessDefinition(
        name=f"proc-{behavior}",
        schema_version=2,
        phases=[
            PhaseTemplate(
                id="s1",
                description="Critical phase",
                is_critical_path=True,
            ),
        ],
        verification_remediation=VerificationRemediationContract(
            critical_path_behavior=behavior,
        ),
    )


# --- create_task ---


class TestCreateTask:
    def test_creates_with_unique_id(self):
        t1 = create_task("goal 1")
        t2 = create_task("goal 2")
        assert t1.id != t2.id

    def test_has_required_fields(self):
        t = create_task("my goal", workspace="/tmp/w", approval_mode="confirm")
        assert t.goal == "my goal"
        assert t.workspace == "/tmp/w"
        assert t.approval_mode == "confirm"
        assert t.created_at != ""

    def test_defaults(self):
        t = create_task("g")
        assert t.workspace == ""
        assert t.approval_mode == "auto"
        assert t.context == {}


# --- ToolCallRecord ---


class TestToolCallRecord:
    def test_auto_timestamp(self):
        r = ToolCallRecord(tool="read", args={}, result=ToolResult.ok("ok"))
        assert r.timestamp != ""

    def test_explicit_timestamp(self):
        r = ToolCallRecord(
            tool="read", args={}, result=ToolResult.ok("ok"),
            timestamp="2025-01-01",
        )
        assert r.timestamp == "2025-01-01"


# --- SubtaskResult ---


class TestSubtaskResult:
    def test_defaults(self):
        r = SubtaskResult(status="success", summary="done")
        assert r.tool_calls == []
        assert r.duration_seconds == 0.0
        assert r.tokens_used == 0


# --- Orchestrator ---


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
        assert "run_id" in summary_event.data
        assert "budget_snapshot" in summary_event.data

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


class TestOrchestratorExecution:
    """Tests for the subtask execution phase."""

    @pytest.mark.asyncio
    async def test_dispatch_subtask_carries_prior_successful_tool_calls(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "First"}]
        })
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First")
        task.plan.subtasks = [subtask]

        successful_call = ToolCallRecord(
            tool="web_search",
            args={"query": "Arizona water market evidence"},
            result=ToolResult.ok("Arizona data found"),
        )
        attempts = {
            "s1": [
                AttemptRecord(
                    attempt=1,
                    tier=1,
                    feedback="retry",
                    error="prior verification failed",
                    successful_tool_calls=[successful_call, "ignore-non-tool-call"],
                )
            ]
        }

        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=1, passed=True),
        ))

        await orch._dispatch_subtask(task, subtask, attempts)

        _, kwargs = orch._runner.run.await_args
        assert kwargs["prior_successful_tool_calls"] == [successful_call]

    @pytest.mark.asyncio
    async def test_dispatch_subtask_retry_enforces_canonical_deliverables(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "phase-a", "description": "Analyze"}]
        })
        process = ProcessDefinition(
            name="test-process",
            description="Test",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task()
        subtask = Subtask(id="phase-a", description="Analyze")
        task.plan.subtasks = [subtask]

        successful_call = ToolCallRecord(
            tool="write_file",
            args={"path": "analysis.md"},
            result=ToolResult.ok("ok", files_changed=["analysis.md"]),
        )
        attempts = {
            "phase-a": [
                AttemptRecord(
                    attempt=1,
                    tier=1,
                    feedback="Retry with fixes",
                    retry_strategy=RetryStrategy.UNCONFIRMED_DATA,
                    successful_tool_calls=[successful_call],
                ),
            ],
        }
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=1, passed=True),
        ))

        await orch._dispatch_subtask(task, subtask, attempts)

        _, kwargs = orch._runner.run.await_args
        assert kwargs["expected_deliverables"] == ["analysis.md"]
        assert kwargs["enforce_deliverable_paths"] is True
        assert kwargs["edit_existing_only"] is True
        assert kwargs["retry_strategy"] == RetryStrategy.UNCONFIRMED_DATA.value
        assert "CANONICAL DELIVERABLE FILES FOR THIS SUBTASK" in kwargs["retry_context"]

    @pytest.mark.asyncio
    async def test_stalled_plan_emits_blocked_subtasks_on_failure(self, tmp_path):
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda e: events.append(e))

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text="{}"),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )
        orch._replan_task = AsyncMock(return_value=False)

        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(
                    id="failed-upstream",
                    description="Failed upstream work",
                    status=SubtaskStatus.FAILED,
                ),
                Subtask(
                    id="downstream-pending",
                    description="Blocked by failed step",
                    status=SubtaskStatus.PENDING,
                    depends_on=["failed-upstream"],
                ),
            ],
            version=1,
        )

        result = await orch.execute_task(task, reuse_existing_plan=True)

        assert result.status == TaskStatus.FAILED
        assert TASK_STALLED in [e.event_type for e in events]
        failed_events = [e for e in events if e.event_type == TASK_FAILED]
        assert failed_events
        blocked_subtasks = failed_events[-1].data.get("blocked_subtasks")
        assert isinstance(blocked_subtasks, list)
        assert blocked_subtasks[0]["subtask_id"] == "downstream-pending"
        orch._replan_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_single_subtask_exception_retries_instead_of_fatal_abort(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "First"}]
        })
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=1,
            max_loop_iterations=50,
            max_parallel_subtasks=3,
            auto_approve_confidence_threshold=0.5,
            enable_streaming=False,
        ))
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda e: events.append(e))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=cfg,
        )
        orch._runner.run = AsyncMock(side_effect=[
            ModelConnectionError("Model server returned HTTP 522: upstream timeout"),
            (
                SubtaskResult(status="success", summary="ok"),
                VerificationResult(tier=1, passed=True),
            ),
        ])

        task = _make_task()
        result = await orch.execute_task(task)

        assert orch._runner.run.await_count == 2
        fatal_task_failed = [
            e for e in events
            if e.event_type == TASK_FAILED and isinstance(e.data, dict) and "error_type" in e.data
        ]
        assert not fatal_task_failed
        assert all(err.subtask != "orchestrator" for err in result.errors_encountered)

    @pytest.mark.asyncio
    async def test_runner_retries_model_connection_errors_within_single_attempt(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "First"}]
        })
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=50,
            max_parallel_subtasks=3,
            auto_approve_confidence_threshold=0.5,
            enable_streaming=False,
            model_call_max_attempts=3,
            model_call_retry_base_delay_seconds=0.0,
            model_call_retry_max_delay_seconds=0.0,
            model_call_retry_jitter_seconds=0.0,
        ))
        router = MagicMock(spec=ModelRouter)
        planner_model = AsyncMock()
        planner_model.name = "mock-planner"
        planner_model.complete = AsyncMock(return_value=ModelResponse(
            text=plan_json,
            usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
        ))
        executor_model = AsyncMock()
        executor_model.name = "mock-executor"
        executor_model.complete = AsyncMock(side_effect=[
            ModelConnectionError("Model server returned HTTP 522: upstream timeout"),
            ModelResponse(
                text="Subtask completed successfully.",
                usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
            ),
        ])

        def select_fn(tier=1, role="executor"):
            if role == "planner":
                return planner_model
            return executor_model

        router.select = MagicMock(side_effect=select_fn)
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda e: events.append(e))
        orch = Orchestrator(
            model_router=router,
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=cfg,
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert executor_model.complete.await_count == 2
        event_types = [e.event_type for e in events]
        assert TASK_FAILED not in event_types

    @pytest.mark.asyncio
    async def test_dispatch_subtask_carries_prior_evidence_records(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "First"}]
        })
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First")
        task.plan.subtasks = [subtask]
        state_manager.create(task)
        state_manager.append_evidence_records(task.id, [{
            "evidence_id": "EV-PERSISTED-1",
            "task_id": task.id,
            "subtask_id": "s1",
            "market": "Arizona Water/Wastewater",
            "source_url": "https://example.com/az",
        }])

        attempts = {
            "s1": [
                AttemptRecord(
                    attempt=1,
                    tier=1,
                    feedback="retry",
                    error="prior verification failed",
                    evidence_records=[{
                        "evidence_id": "EV-ATTEMPT-1",
                        "task_id": task.id,
                        "subtask_id": "s1",
                        "market": "Alberta Retail Energy",
                        "source_url": "https://example.com/ab",
                    }],
                )
            ]
        }

        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=1, passed=True),
        ))

        await orch._dispatch_subtask(task, subtask, attempts)

        _, kwargs = orch._runner.run.await_args
        evidence_ids = {
            str(item.get("evidence_id", ""))
            for item in kwargs["prior_evidence_records"]
            if isinstance(item, dict)
        }
        assert "EV-PERSISTED-1" in evidence_ids
        assert "EV-ATTEMPT-1" in evidence_ids

    @pytest.mark.asyncio
    async def test_handle_failure_uses_verification_only_retry_for_parse_errors(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "First"}]
        })
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First")
        task.plan.subtasks = [subtask]
        state_manager.create(task)

        result = SubtaskResult(
            status="failed",
            summary="Execution summary",
            evidence_records=[{
                "evidence_id": "EV-1",
                "task_id": task.id,
                "subtask_id": "s1",
            }],
        )
        verification = VerificationResult(
            tier=2,
            passed=False,
            feedback="Verification inconclusive: could not parse verifier output.",
        )
        attempts_by_subtask: dict[str, list[AttemptRecord]] = {}

        orch._retry_verification_only = AsyncMock(return_value=VerificationResult(
            tier=2,
            passed=True,
            confidence=0.8,
        ))
        orch._handle_success = AsyncMock()

        await orch._handle_failure(
            task,
            subtask,
            result,
            verification,
            attempts_by_subtask,
        )

        orch._retry_verification_only.assert_awaited_once()
        orch._handle_success.assert_awaited_once()
        attempts = attempts_by_subtask.get("s1", [])
        assert len(attempts) == 1
        assert attempts[0].error is not None

    @pytest.mark.asyncio
    async def test_handle_failure_generates_model_resolution_plan_for_replan(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(max_subtask_retries=0)),
        )
        task = _make_task()
        subtask = Subtask(
            id="s1",
            description="First",
            max_retries=0,
            is_critical_path=False,
        )
        task.plan.subtasks = [subtask]
        state_manager.create(task)

        result = SubtaskResult(status="failed", summary="Wrong output file")
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="llm_semantic_failed",
            feedback="Deliverable path mismatch",
        )
        attempts_by_subtask: dict[str, list[AttemptRecord]] = {}
        orch._plan_failure_resolution = AsyncMock(return_value=(
            "Diagnosis: Output path diverged from canonical deliverable.\n"
            "Actions:\n"
            "1. Update canonical filename in place."
        ))

        replan_request = await orch._handle_failure(
            task,
            subtask,
            result,
            verification,
            attempts_by_subtask,
        )

        orch._plan_failure_resolution.assert_awaited_once()
        attempts = attempts_by_subtask.get("s1", [])
        assert len(attempts) == 1
        assert "canonical deliverable" in attempts[0].resolution_plan
        assert isinstance(replan_request, dict)
        assert "MODEL-PLANNED RESOLUTION" in str(
            replan_request.get("verification_feedback", ""),
        )

    @pytest.mark.asyncio
    async def test_dispatch_subtask_retry_context_includes_model_resolution_plan(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First")
        task.plan.subtasks = [subtask]
        state_manager.create(task)

        attempts = {
            "s1": [
                AttemptRecord(
                    attempt=1,
                    tier=1,
                    feedback="retry",
                    error="verification failed",
                    resolution_plan=(
                        "Diagnosis: Wrong file path.\n"
                        "Actions:\n"
                        "1. Patch canonical file."
                    ),
                ),
            ],
        }
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=1, passed=True),
        ))

        await orch._dispatch_subtask(task, subtask, attempts)

        _, kwargs = orch._runner.run.await_args
        assert "Model-planned remediation" in kwargs["retry_context"]
        assert "Patch canonical file." in kwargs["retry_context"]

    @pytest.mark.asyncio
    async def test_failure_resolution_prompt_compacts_large_metadata(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First")
        task.plan.subtasks = [subtask]
        state_manager.create(task)

        large_metadata = {
            "issues": [f"issue-{i}-" + ("x" * 80) for i in range(40)],
            "deterministic_placeholder_scan": {
                "scan_mode": "targeted_plus_fallback",
                "scanned_file_count": 240,
                "matched_file_count": 0,
                "scanned_files": [f"file-{i}.md" for i in range(200)],
                "candidate_source_counts": {"canonical": 1, "fallback": 199},
            },
            "raw_blob": "Y" * 20_000,
        }
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="llm_semantic_failed",
            feedback="verification failed",
            metadata=large_metadata,
        )
        result = SubtaskResult(status="failed", summary="Execution summary")

        prompt = orch._build_failure_resolution_prompt(
            subtask=subtask,
            result=result,
            verification=verification,
            strategy=RetryStrategy.GENERIC,
            missing_targets=[],
            prior_attempts=[],
        )

        assert "Verification metadata:" in prompt
        assert "raw_blob" not in prompt
        assert "file-199.md" not in prompt
        assert len(prompt) < 6000

    @pytest.mark.asyncio
    async def test_plan_failure_resolution_uses_fallback_when_planner_output_not_json(
        self, tmp_path
    ):
        state_manager = _make_state_manager(tmp_path)
        config = Config(
            execution=ExecutionConfig(
                model_call_max_attempts=1,
                model_call_retry_base_delay_seconds=0.0,
                model_call_retry_max_delay_seconds=0.0,
                model_call_retry_jitter_seconds=0.0,
            ),
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=config,
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First")
        task.plan.subtasks = [subtask]
        state_manager.create(task)
        result = SubtaskResult(status="failed", summary="Execution summary")
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="llm_semantic_failed",
            feedback="verification failed",
        )

        planner_model = AsyncMock()
        planner_model.name = "mock-planner"
        planner_model.complete = AsyncMock(return_value=ModelResponse(
            text="Use canonical deliverable and patch only the failing section.",
            usage=TokenUsage(input_tokens=20, output_tokens=20, total_tokens=40),
        ))
        orch._router.select = MagicMock(return_value=planner_model)

        plan = await orch._plan_failure_resolution(
            task=task,
            subtask=subtask,
            result=result,
            verification=verification,
            strategy=RetryStrategy.GENERIC,
            missing_targets=[],
            prior_attempts=[],
        )

        assert "Use canonical deliverable" in plan
        planner_model.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plan_failure_resolution_returns_empty_on_planner_call_error(
        self, tmp_path
    ):
        state_manager = _make_state_manager(tmp_path)
        config = Config(
            execution=ExecutionConfig(
                model_call_max_attempts=1,
                model_call_retry_base_delay_seconds=0.0,
                model_call_retry_max_delay_seconds=0.0,
                model_call_retry_jitter_seconds=0.0,
            ),
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=config,
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First")
        task.plan.subtasks = [subtask]
        state_manager.create(task)
        result = SubtaskResult(status="failed", summary="Execution summary")
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="llm_semantic_failed",
            feedback="verification failed",
        )

        planner_model = AsyncMock()
        planner_model.name = "mock-planner"
        planner_model.complete = AsyncMock(side_effect=RuntimeError("planner offline"))
        orch._router.select = MagicMock(return_value=planner_model)

        plan = await orch._plan_failure_resolution(
            task=task,
            subtask=subtask,
            result=result,
            verification=verification,
            strategy=RetryStrategy.GENERIC,
            missing_targets=[],
            prior_attempts=[],
        )

        assert plan == ""
        planner_model.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handle_failure_queues_noncritical_unconfirmed_for_follow_up(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "First"}]
        })
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First", is_critical_path=False)
        task.plan.subtasks = [subtask]
        state_manager.create(task)

        result = SubtaskResult(
            status="failed",
            summary="Execution summary",
            evidence_records=[],
        )
        verification = VerificationResult(
            tier=2,
            passed=False,
            feedback=(
                "Recommendations include unconfirmed claims; "
                "confirm-or-prune remediation is required."
            ),
            outcome="fail",
            reason_code="recommendation_unconfirmed",
        )
        attempts_by_subtask: dict[str, list[AttemptRecord]] = {}

        orch._handle_success = AsyncMock()

        await orch._handle_failure(
            task,
            subtask,
            result,
            verification,
            attempts_by_subtask,
        )

        orch._handle_success.assert_awaited_once()
        queue = task.metadata.get("remediation_queue", [])
        assert isinstance(queue, list)
        assert queue
        assert queue[0]["subtask_id"] == "s1"
        assert queue[0]["blocking"] is False
        attempts = attempts_by_subtask.get("s1", [])
        assert attempts
        assert attempts[0].retry_strategy.value == "unconfirmed_data"

    @pytest.mark.asyncio
    async def test_confirm_or_prune_then_queue_policy_attempts_remediation_first(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Critical"}]
        })
        state_manager = _make_state_manager(tmp_path)
        process = _make_process_with_critical_behavior("confirm_or_prune_then_queue")
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
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=cfg,
            process=process,
        )
        task = _make_task()
        subtask = Subtask(
            id="s1",
            description="Critical",
            is_critical_path=True,
            max_retries=0,
        )
        task.plan.subtasks = [subtask]
        state_manager.create(task)

        result = SubtaskResult(
            status="failed",
            summary="Execution summary",
            evidence_records=[],
        )
        verification = VerificationResult(
            tier=2,
            passed=False,
            feedback=(
                "Recommendations include unconfirmed claims; "
                "confirm-or-prune remediation is required."
            ),
            outcome="fail",
            reason_code="recommendation_unconfirmed",
        )
        attempts_by_subtask: dict[str, list[AttemptRecord]] = {}

        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="remediated"),
            VerificationResult(tier=2, passed=True, outcome="pass"),
        ))
        orch._handle_success = AsyncMock()
        orch._abort_on_critical_path_failure = AsyncMock()

        await orch._handle_failure(
            task,
            subtask,
            result,
            verification,
            attempts_by_subtask,
        )

        orch._runner.run.assert_awaited_once()
        _, kwargs = orch._runner.run.await_args
        assert "TARGETED REMEDIATION:" in kwargs["retry_context"]
        orch._handle_success.assert_awaited_once()
        orch._abort_on_critical_path_failure.assert_not_awaited()
        attempted = task.metadata.get("confirm_or_prune_attempts", {})
        assert isinstance(attempted, dict)
        assert "s1" in attempted
        assert isinstance(attempted["s1"], list)
        assert attempted["s1"]

    @pytest.mark.asyncio
    async def test_confirm_or_prune_retries_transient_failures_before_abort(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Critical"}]
        })
        state_manager = _make_state_manager(tmp_path)
        process = _make_process_with_critical_behavior("confirm_or_prune_then_queue")
        cfg = Config(
            execution=ExecutionConfig(
                max_subtask_retries=0,
                max_loop_iterations=50,
                max_parallel_subtasks=3,
                auto_approve_confidence_threshold=0.8,
                enable_streaming=False,
            ),
            verification=VerificationConfig(
                confirm_or_prune_max_attempts=3,
                confirm_or_prune_backoff_seconds=0.0,
                confirm_or_prune_retry_on_transient=True,
            ),
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=cfg,
            process=process,
        )
        task = _make_task()
        subtask = Subtask(
            id="s1",
            description="Critical",
            is_critical_path=True,
            max_retries=0,
        )
        task.plan.subtasks = [subtask]
        state_manager.create(task)

        result = SubtaskResult(
            status="failed",
            summary="Execution summary",
            evidence_records=[],
        )
        verification = VerificationResult(
            tier=2,
            passed=False,
            feedback=(
                "Recommendations include unconfirmed claims; "
                "confirm-or-prune remediation is required."
            ),
            outcome="fail",
            reason_code="recommendation_unconfirmed",
        )
        attempts_by_subtask: dict[str, list[AttemptRecord]] = {}

        orch._runner.run = AsyncMock(side_effect=[
            (
                SubtaskResult(status="failed", summary="HTTP 429 from upstream"),
                VerificationResult(
                    tier=2,
                    passed=False,
                    outcome="fail",
                    feedback="rate limited while confirming evidence",
                    reason_code="infra_verifier_error",
                ),
            ),
            (
                SubtaskResult(status="success", summary="remediated"),
                VerificationResult(tier=2, passed=True, outcome="pass"),
            ),
        ])
        orch._handle_success = AsyncMock()
        orch._abort_on_critical_path_failure = AsyncMock()

        await orch._handle_failure(
            task,
            subtask,
            result,
            verification,
            attempts_by_subtask,
        )

        assert orch._runner.run.await_count == 2
        orch._handle_success.assert_awaited_once()
        orch._abort_on_critical_path_failure.assert_not_awaited()

        attempted = task.metadata.get("confirm_or_prune_attempts", {})
        assert isinstance(attempted, dict)
        rows = attempted.get("s1", [])
        assert isinstance(rows, list)
        assert len(rows) >= 2
        assert rows[0]["status"] == "failed"
        assert rows[-1]["status"] == "resolved"

    @pytest.mark.asyncio
    async def test_critical_unconfirmed_block_policy_aborts(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        process = _make_process_with_critical_behavior("block")
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(max_subtask_retries=0)),
            process=process,
        )
        task = _make_task()
        subtask = Subtask(
            id="s1",
            description="Critical",
            is_critical_path=True,
            max_retries=0,
        )
        task.plan.subtasks = [subtask]
        state_manager.create(task)
        result = SubtaskResult(status="failed", summary="Execution summary")
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="recommendation_unconfirmed",
            feedback="confirm-or-prune remediation required",
        )

        orch._run_confirm_or_prune_remediation = AsyncMock(return_value=(True, ""))
        orch._abort_on_critical_path_failure = AsyncMock()

        await orch._handle_failure(
            task,
            subtask,
            result,
            verification,
            attempts_by_subtask={},
        )

        orch._run_confirm_or_prune_remediation.assert_not_awaited()
        orch._abort_on_critical_path_failure.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_critical_unconfirmed_queue_follow_up_policy_continues(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        process = _make_process_with_critical_behavior("queue_follow_up")
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(max_subtask_retries=0)),
            process=process,
        )
        task = _make_task()
        subtask = Subtask(
            id="s1",
            description="Critical",
            is_critical_path=True,
            max_retries=0,
        )
        task.plan.subtasks = [subtask]
        state_manager.create(task)
        result = SubtaskResult(status="failed", summary="Execution summary")
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="recommendation_unconfirmed",
            feedback="confirm-or-prune remediation required",
        )
        orch._handle_success = AsyncMock()
        orch._abort_on_critical_path_failure = AsyncMock()

        await orch._handle_failure(
            task,
            subtask,
            result,
            verification,
            attempts_by_subtask={},
        )

        orch._handle_success.assert_awaited_once()
        orch._abort_on_critical_path_failure.assert_not_awaited()
        queue = task.metadata.get("remediation_queue", [])
        assert isinstance(queue, list)
        assert queue
        assert queue[0]["blocking"] is False

    @pytest.mark.asyncio
    async def test_remediation_queue_dedupes_and_promotes_blocking(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        subtask = Subtask(id="s1", description="First")
        task.plan.subtasks = [subtask]
        state_manager.create(task)
        verification = VerificationResult(
            tier=2,
            passed=False,
            outcome="fail",
            reason_code="recommendation_unconfirmed",
            feedback="confirm-or-prune needed",
        )

        await orch._queue_remediation_work_item(
            task=task,
            subtask=subtask,
            verification=verification,
            strategy=RetryStrategy.UNCONFIRMED_DATA,
            blocking=False,
        )
        await orch._queue_remediation_work_item(
            task=task,
            subtask=subtask,
            verification=verification,
            strategy=RetryStrategy.UNCONFIRMED_DATA,
            blocking=True,
        )

        queue = task.metadata.get("remediation_queue", [])
        assert isinstance(queue, list)
        assert len(queue) == 1
        assert queue[0]["blocking"] is True

    @pytest.mark.asyncio
    async def test_finalizing_remediation_queue_fails_unresolved_blocking_items(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        task.plan.subtasks = [Subtask(id="s1", description="First")]
        task.metadata["remediation_queue"] = [{
            "id": "rem-1",
            "task_id": task.id,
            "subtask_id": "s1",
            "strategy": "unsupported",
            "reason_code": "x",
            "blocking": True,
            "state": "queued",
            "attempt_count": 0,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "next_attempt_at": "2026-01-01T00:00:00",
        }]
        state_manager.create(task)

        await orch._process_remediation_queue(
            task=task,
            attempts_by_subtask={},
            finalizing=True,
        )

        queue = task.metadata.get("remediation_queue", [])
        assert isinstance(queue, list)
        assert queue[0]["state"] == "failed"

    @pytest.mark.asyncio
    async def test_remediation_queue_exhaustion_sets_terminal_reason(self, tmp_path):
        state_manager = _make_state_manager(tmp_path)
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        task.plan.subtasks = [Subtask(id="s1", description="First")]
        task.metadata["remediation_queue"] = [{
            "id": "rem-1",
            "task_id": task.id,
            "subtask_id": "s1",
            "strategy": "unsupported",
            "reason_code": "x",
            "blocking": False,
            "state": "queued",
            "attempt_count": 0,
            "max_attempts": 1,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "next_attempt_at": "2026-01-01T00:00:00",
        }]
        state_manager.create(task)

        await orch._process_remediation_queue(
            task=task,
            attempts_by_subtask={},
            finalizing=False,
        )

        queue = task.metadata.get("remediation_queue", [])
        assert isinstance(queue, list)
        assert queue[0]["state"] == "failed"
        assert queue[0]["terminal_reason"] == "max_attempts_exhausted"

    @pytest.mark.asyncio
    async def test_executes_subtasks_in_order(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [
                {"id": "s1", "description": "First"},
                {"id": "s2", "description": "Second", "depends_on": ["s1"]},
            ]
        })

        executed = []
        prompts = _make_mock_prompts()

        def tracking_build(**kwargs):
            executed.append(kwargs.get("subtask").id if kwargs.get("subtask") else "?")
            return "Execute"

        prompts.build_executor_prompt = MagicMock(side_effect=tracking_build)

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=prompts,
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert executed == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_tool_calling_loop(self, tmp_path):
        """Test that tool calls are processed in a loop until text-only response."""
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Use tools"}]
        })

        # First response has tool calls, second is text-only
        tool_response = ModelResponse(
            text="",
            tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "/tmp/x"})],
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        )
        final_response = ModelResponse(
            text="Done with tools.",
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        )

        tools = _make_mock_tools()
        tools.execute = AsyncMock(return_value=ToolResult.ok("file content here"))

        orch = Orchestrator(
            model_router=_make_mock_router(
                plan_response_text=plan_json,
                executor_responses=[tool_response, final_response],
            ),
            tool_registry=tools,
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        # Tool was executed
        tools.execute.assert_called_once_with(
            "read_file", {"path": "/tmp/x"},
            workspace=None,
            read_roots=[],
            scratch_dir=ANY,
            changelog=None,
            subtask_id="s1",
            auth_context=ANY,
        )

    @pytest.mark.asyncio
    async def test_subtask_summary_captured(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Do something"}]
        })

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.plan.subtasks[0].status == SubtaskStatus.COMPLETED
        assert result.plan.subtasks[0].summary != ""


class TestOrchestratorFinalize:
    """Tests for the finalization phase."""

    @pytest.mark.asyncio
    async def test_completed_when_all_done(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Only step"}]
        })

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

        task = _make_task()
        result = await orch.execute_task(task)
        assert result.status == TaskStatus.COMPLETED
        assert result.completed_at != ""

    @pytest.mark.asyncio
    async def test_wrap_up_exports_evidence_ledger_csv_to_workspace(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Only step"}]
        })
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path / "state"),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="success",
                summary="captured evidence",
                evidence_records=[{
                    "evidence_id": "EV-1",
                    "tool": "web_fetch",
                    "source_url": "https://example.com/report",
                    "quality": 0.9,
                    "facets": {"market": "water"},
                }],
            ),
            VerificationResult(tier=1, passed=True),
        ))

        task = _make_task(workspace=str(workspace))
        result = await orch.execute_task(task)
        assert result.status == TaskStatus.COMPLETED

        ledger_csv = workspace / "evidence-ledger.csv"
        assert ledger_csv.exists()
        with ledger_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["evidence_id"] == "EV-1"
        assert rows[0]["task_id"] == task.id
        assert rows[0]["subtask_id"] == "s1"
        assert rows[0]["facets"] == '{"market": "water"}'

    @pytest.mark.asyncio
    async def test_wrap_up_skips_evidence_csv_when_no_ledger(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Only step"}]
        })
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path / "state"),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

        task = _make_task(workspace=str(workspace))
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert not (workspace / "evidence-ledger.csv").exists()

    @pytest.mark.asyncio
    async def test_wrap_up_exports_evidence_csv_for_failed_task(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{
                "id": "s1",
                "description": "Only step",
                "is_critical_path": True,
            }]
        })
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path / "state"),
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(max_subtask_retries=0)),
        )
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status="failed",
                summary="verification failed",
                evidence_records=[{
                    "evidence_id": "EV-FAIL-1",
                    "tool": "web_search",
                    "query": "utility market",
                }],
            ),
            VerificationResult(
                tier=2,
                passed=False,
                feedback="Verification failed",
            ),
        ))

        task = _make_task(workspace=str(workspace))
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.FAILED
        ledger_csv = workspace / "evidence-ledger.csv"
        assert ledger_csv.exists()
        with ledger_csv.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["evidence_id"] == "EV-FAIL-1"

    @pytest.mark.asyncio
    async def test_cancel_task(self, tmp_path):
        """Cancel during execution: after s1 completes, cancel before s2 runs."""
        plan_json = json.dumps({
            "subtasks": [
                {"id": "s1", "description": "First"},
                {"id": "s2", "description": "Second", "depends_on": ["s1"]},
            ]
        })

        executor_model = AsyncMock()
        executor_model.name = "mock-exec"
        cancel_target = {"task": None, "orch": None}

        async def executor_complete(messages, tools=None):
            # Cancel after first subtask executes
            if cancel_target["task"] and cancel_target["orch"]:
                cancel_target["orch"].cancel_task(cancel_target["task"])
            return ModelResponse(
                text="Done",
                usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
            )

        executor_model.complete = executor_complete

        router = MagicMock(spec=ModelRouter)
        planner_model = AsyncMock()
        planner_model.name = "mock-plan"
        planner_model.complete = AsyncMock(return_value=ModelResponse(
            text=plan_json,
            usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        ))

        def select_fn(tier=1, role="executor"):
            if role == "planner":
                return planner_model
            return executor_model

        router.select = MagicMock(side_effect=select_fn)

        orch = Orchestrator(
            model_router=router,
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

        task = _make_task()
        cancel_target["task"] = task
        cancel_target["orch"] = orch

        result = await orch.execute_task(task)

        assert result.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_failed_on_exception(self, tmp_path):
        """If an exception occurs during execution, task should be FAILED."""
        router = _make_mock_router(plan_response_text="bad")
        # Force planner to raise
        planner_model = AsyncMock()
        planner_model.complete = AsyncMock(side_effect=RuntimeError("Model crashed"))

        router.select = MagicMock(return_value=planner_model)

        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda e: events.append(e))

        orch = Orchestrator(
            model_router=router,
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.FAILED
        assert len(result.errors_encountered) > 0
        event_types = [e.event_type for e in events]
        assert TASK_FAILED in event_types

    @pytest.mark.asyncio
    async def test_planning_model_connection_errors_fallback_to_safe_plan(self, tmp_path):
        router = MagicMock(spec=ModelRouter)
        planner_model = AsyncMock()
        planner_model.name = "mock-planner"
        planner_model.complete = AsyncMock(side_effect=[
            ModelConnectionError("Model server returned HTTP 522: upstream timeout"),
            ModelConnectionError("Model server returned HTTP 522: upstream timeout"),
            ModelConnectionError("Model server returned HTTP 522: upstream timeout"),
        ])
        executor_model = AsyncMock()
        executor_model.name = "mock-executor"
        executor_model.complete = AsyncMock(return_value=ModelResponse(
            text="Subtask completed successfully.",
            usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
        ))

        def select_fn(tier=1, role="executor"):
            if role == "planner":
                return planner_model
            return executor_model

        router.select = MagicMock(side_effect=select_fn)

        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=50,
            max_parallel_subtasks=3,
            auto_approve_confidence_threshold=0.5,
            enable_streaming=False,
            model_call_max_attempts=3,
            model_call_retry_base_delay_seconds=0.0,
            model_call_retry_max_delay_seconds=0.0,
            model_call_retry_jitter_seconds=0.0,
        ))
        orch = Orchestrator(
            model_router=router,
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=cfg,
        )

        task = _make_task()
        result = await orch.execute_task(task)

        assert planner_model.complete.await_count == 3
        assert result.plan.subtasks[0].id == "execute-goal"
        assert result.status == TaskStatus.COMPLETED


class TestWorkspaceDocumentScan:
    """Tests for the expanded workspace analysis that includes non-code files."""

    def _make_orchestrator(self, tmp_path):
        return Orchestrator(
            model_router=_make_mock_router(),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

    def test_scan_finds_documents_by_category(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / "README.md").write_text("# Hello")
        (workspace / "spec.pdf").write_bytes(b"%PDF-fake")
        (workspace / "data.csv").write_text("a,b\n1,2")
        (workspace / "logo.png").write_bytes(b"\x89PNG")
        (workspace / "slides.pptx").write_bytes(b"PK-fake")

        orch = self._make_orchestrator(tmp_path)
        result = orch._scan_workspace_documents(workspace)

        assert "Documents and non-code files:" in result
        assert "README.md" in result
        assert "spec.pdf" in result
        assert "data.csv" in result
        assert "logo.png" in result
        assert "slides.pptx" in result

    def test_scan_skips_hidden_and_noise_dirs(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / ".git").mkdir()
        (workspace / ".git" / "notes.md").write_text("internal")
        (workspace / "node_modules").mkdir()
        (workspace / "node_modules" / "pkg.json").write_text("{}")
        # This one should be found
        (workspace / "docs").mkdir()
        (workspace / "docs" / "guide.md").write_text("# Guide")

        orch = self._make_orchestrator(tmp_path)
        result = orch._scan_workspace_documents(workspace)

        assert "guide.md" in result
        assert "notes.md" not in result
        assert "pkg.json" not in result

    def test_scan_returns_empty_for_code_only_workspace(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / "main.py").write_text("print('hello')")
        (workspace / "utils.go").write_text("package main")

        orch = self._make_orchestrator(tmp_path)
        result = orch._scan_workspace_documents(workspace)

        assert result == ""

    def test_scan_respects_max_per_category(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        for i in range(20):
            (workspace / f"doc_{i:02d}.md").write_text(f"# Doc {i}")

        orch = self._make_orchestrator(tmp_path)
        result = orch._scan_workspace_documents(workspace, max_per_category=5)

        # Should only list 5
        md_lines = [line for line in result.splitlines() if "doc_" in line]
        assert len(md_lines) == 5

    @pytest.mark.asyncio
    async def test_analyze_workspace_includes_documents(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        (workspace / "app.py").write_text("class App:\n    pass\n")
        (workspace / "README.md").write_text("# My App")
        (workspace / "data.csv").write_text("x,y\n1,2")

        orch = self._make_orchestrator(tmp_path)
        result = await orch._analyze_workspace(workspace)

        # Code analysis should find the Python file
        assert "App" in result
        # Document scan should find the non-code files
        assert "README.md" in result
        assert "data.csv" in result


class TestOrchestratorTodoReminder:
    def test_build_todo_reminder(self):
        task = _make_task(goal="Build a CLI tool")
        subtask = Subtask(id="step-1", description="Create main.py")

        from loom.engine.runner import SubtaskRunner
        reminder = SubtaskRunner._build_todo_reminder(task, subtask)

        assert "Build a CLI tool" in reminder
        assert "step-1" in reminder
        assert "Create main.py" in reminder
        assert "Do NOT move to the next subtask" in reminder


class TestSubtaskRunnerContextBudget:
    class _FakeCompactor:
        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            value = str(text or "")
            if len(value) <= max_chars:
                return value
            words = value.split()
            if not words:
                return value
            if len(words) == 1 and len(words[0]) > max_chars:
                return f"[compacted {len(value)} chars]"
            compacted = ""
            for word in words:
                candidate = f"{compacted} {word}".strip()
                if compacted and len(candidate) > max_chars:
                    break
                compacted = candidate
            return compacted or value

    class _RecordingCompactor:
        def __init__(self):
            self.calls: list[tuple[str, int, int]] = []

        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            value = str(text or "")
            self.calls.append((str(label), int(max_chars), len(value)))
            if len(value) <= max_chars:
                return value
            words = value.split()
            if not words:
                return value
            compacted = ""
            for word in words:
                candidate = f"{compacted} {word}".strip()
                if compacted and len(candidate) > max_chars:
                    break
                compacted = candidate
            return compacted or value

    @staticmethod
    def _make_runner_for_compaction():
        from loom.engine.runner import SubtaskRunner

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._compactor = TestSubtaskRunnerContextBudget._FakeCompactor()
        runner._runner_compaction_policy_mode = "legacy"
        runner._max_model_context_tokens = SubtaskRunner.MAX_MODEL_CONTEXT_TOKENS
        return runner

    @staticmethod
    def _make_runner_for_tiered_compaction(*, context_budget: int = 2500):
        from loom.engine.runner import SubtaskRunner

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._compactor = TestSubtaskRunnerContextBudget._RecordingCompactor()
        runner._runner_compaction_policy_mode = "tiered"
        runner._max_model_context_tokens = context_budget
        runner._compaction_pressure_ratio_soft = 0.70
        runner._compaction_pressure_ratio_hard = 0.92
        runner._preserve_recent_critical_messages = 4
        runner._compact_tool_call_argument_chars = 160
        runner._compact_tool_result_output_chars = 180
        runner._compact_text_output_chars = 220
        runner._minimal_text_output_chars = 120
        runner._compaction_timeout_guard_seconds = 25
        runner._compaction_no_gain_min_delta_chars = 4
        runner._compaction_no_gain_attempt_limit = 2
        runner._compaction_churn_warning_calls = 100
        runner._extractor_tool_args_max_chars = 180
        runner._extractor_tool_trace_max_chars = 1800
        runner._extractor_prompt_max_chars = 2600
        runner._enable_model_overflow_fallback = True
        runner._overflow_fallback_tool_message_min_chars = 500
        runner._overflow_fallback_tool_output_excerpt_chars = 220
        return runner

    @staticmethod
    def _make_runner_for_telemetry():
        from loom.engine.runner import SubtaskRunner

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._event_bus = EventBus()
        runner._enable_artifact_telemetry_events = True
        runner._artifact_telemetry_max_metadata_chars = 120
        runner._runner_compaction_policy_mode = "tiered"
        runner._max_model_context_tokens = 2400
        runner._last_compaction_diagnostics = {
            "compaction_policy_mode": "tiered",
            "compaction_pressure_ratio": 0.91,
            "compaction_stage": "stage_2_tool_outputs",
            "compaction_skipped_reason": "",
        }
        runner._active_subtask_telemetry_counters = (
            SubtaskRunner._new_subtask_telemetry_counters()
        )
        return runner

    class _NoopCompactor:
        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            return str(text or "")

    def test_detects_model_overflow_error_markers(self):
        from loom.engine.runner import SubtaskRunner

        assert SubtaskRunner._is_model_request_overflow_error(
            "Invalid request: total message size 123 exceeds limit 99",
        )
        assert SubtaskRunner._is_model_request_overflow_error(
            "Invalid request: Your request exceeded model token limit: 262144",
        )
        assert not SubtaskRunner._is_model_request_overflow_error(
            "Model server returned HTTP 522: upstream timeout",
        )

    def test_overflow_fallback_rewrites_older_tool_messages_and_preserves_latest(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=1200)
        tool_payload = json.dumps({
            "success": True,
            "output": "A" * 12_000,
            "error": None,
            "files_changed": [],
            "data": {
                "content_kind": "pdf",
                "artifact_ref": "af_123",
                "size_bytes": 2_000_000,
                "url": "https://example.com/report.pdf",
            },
        })
        latest_payload = json.dumps({
            "success": True,
            "output": "latest short output",
            "error": None,
            "files_changed": [],
            "data": {"content_kind": "pdf", "artifact_ref": "af_latest"},
        })
        messages = [
            {"role": "user", "content": "Goal: analyze report."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://example.com/a.pdf\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_old", "content": tool_payload},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_latest",
                    "type": "function",
                    "function": {"name": "web_fetch", "arguments": "{\"url\":\"https://example.com/b.pdf\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_latest", "content": latest_payload},
        ]

        rewritten, report = runner._apply_model_overflow_fallback(messages)

        assert report["overflow_fallback_applied"] is True
        assert report["overflow_fallback_rewritten_messages"] == 1
        assert report["overflow_fallback_chars_reduced"] > 0
        rewritten_old = json.loads(rewritten[2]["content"])
        assert "overflow fallback applied" in rewritten_old["output"]
        # Latest tool result is preserved verbatim.
        assert rewritten[4]["content"] == latest_payload

    @pytest.mark.asyncio
    async def test_serialize_tool_result_for_model_compacts_output_and_data(self):
        runner = self._make_runner_for_compaction()

        result = ToolResult.ok(
            "x" * 20_000,
            data={
                "url": "https://example.com/really/long/path",
                "nested": {"a": 1, "b": 2},
                "results": [1, 2, 3],
            },
            files_changed=["report.md"],
        )

        payload = await runner._serialize_tool_result_for_model("web_fetch", result)
        parsed = json.loads(payload)

        assert parsed["success"] is True
        assert len(parsed["output"]) < len(result.output)
        assert parsed["files_changed"] == ["report.md"]
        assert parsed["data"]["url"].startswith("https://example.com/")
        assert "a" in parsed["data"]["nested"]
        assert "1" in parsed["data"]["results"]

    @pytest.mark.asyncio
    async def test_compact_messages_for_model_keeps_structure_and_reduces_tokens(self):
        runner = self._make_runner_for_compaction()

        messages = [{"role": "user", "content": "Goal: perform market research"}]
        for idx in range(24):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": "web_fetch",
                        "arguments": "{\"url\": \"https://example.com\"}",
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{idx}",
                "content": json.dumps({
                    "success": True,
                    "output": "A" * 8_000,
                    "error": None,
                    "files_changed": [],
                }),
            })
            messages.append({
                "role": "user",
                "content": (
                    "CURRENT TASK STATE:\n"
                    "Goal: market research\n"
                    "Current subtask: analyze\n"
                    "Do NOT move to the next subtask."
                ),
            })

        before = runner._estimate_message_tokens(messages)
        compacted = await runner._compact_messages_for_model(messages)
        after = runner._estimate_message_tokens(compacted)

        assert after < before
        assert after <= runner.MAX_MODEL_CONTEXT_TOKENS
        assert compacted[0]["role"] == "user"
        assert any(
            isinstance(msg, dict) and msg.get("role") == "tool"
            and msg.get("tool_call_id") == "call_23"
            for msg in compacted
        )

    @pytest.mark.asyncio
    async def test_summarize_model_output_uses_semantic_compaction(self):
        runner = self._make_runner_for_compaction()

        text = (
            "First sentence is complete. "
            "Second sentence is also complete. "
            "Third sentence should be cut near boundary."
        )
        summary = await runner._summarize_model_output(
            text,
            max_chars=60,
            label="test summary",
        )

        assert "First sentence is complete." in summary
        assert len(summary) <= 60

    @pytest.mark.asyncio
    async def test_compact_text_keeps_oversize_compactor_output(self):
        runner = self._make_runner_for_compaction()
        runner._compactor = self._NoopCompactor()

        value = "A" * 5000
        compacted = await runner._compact_text(
            value,
            max_chars=120,
            label="oversize guard",
        )

        assert compacted == value

    @pytest.mark.asyncio
    async def test_compacts_recent_assistant_tool_call_arguments(self):
        runner = self._make_runner_for_compaction()
        huge_args = json.dumps({
            "path": "report.md",
            "content": "A" * 400_000,
        })
        messages = [
            {"role": "user", "content": "Goal: write report"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_recent",
                    "type": "function",
                    "function": {
                        "name": "document_write",
                        "arguments": huge_args,
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_recent",
                "content": json.dumps({
                    "success": True,
                    "output": "ok",
                    "error": None,
                    "files_changed": ["report.md"],
                }),
            },
            {
                "role": "user",
                "content": (
                    "CURRENT TASK STATE:\nGoal: report\nCurrent subtask: s1\n"
                    "Do NOT move to next subtask"
                ),
            },
        ]

        compacted = await runner._compact_messages_for_model(messages)
        assistant = next(
            msg for msg in compacted
            if isinstance(msg, dict) and msg.get("role") == "assistant"
        )
        args_text = (
            assistant.get("tool_calls", [{}])[0]
            .get("function", {})
            .get("arguments", "")
        )
        assert len(args_text) <= 500
        assert "A" * 200 not in args_text

    @pytest.mark.asyncio
    async def test_no_compaction_when_under_budget(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=12_000)
        messages = [
            {"role": "user", "content": "Goal: summarize file structure."},
            {"role": "assistant", "content": "I will inspect the workspace."},
            {"role": "user", "content": "Focus on src and tests only."},
        ]

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=240)

        assert compacted == messages
        assert runner._compactor.calls == []
        assert runner._last_compaction_diagnostics["compaction_skipped_reason"] == "no_pressure"

    @pytest.mark.asyncio
    async def test_compaction_policy_mode_off_disables_runner_compaction(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=1200)
        runner._runner_compaction_policy_mode = "off"

        result = ToolResult.ok("X" * 5000)
        payload = await runner._serialize_tool_result_for_model(
            "web_fetch",
            result,
            max_output_chars=120,
        )
        parsed = json.loads(payload)

        messages = [
            {"role": "user", "content": "Goal: draft deliverable."},
            {"role": "assistant", "content": "Y " * 1500},
        ]
        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=120)

        assert parsed["output"] == result.output
        assert compacted == messages
        assert runner._compactor.calls == []
        assert runner._last_compaction_diagnostics["compaction_policy_mode"] == "off"
        assert runner._last_compaction_diagnostics["compaction_skipped_reason"] == "policy_disabled"

    @pytest.mark.asyncio
    async def test_compaction_order_tool_trace_before_critical_context(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=1200)
        huge_args_older = json.dumps({"path": "report-old.md", "content": "A" * 14000})
        huge_args_latest = json.dumps({"path": "report.md", "content": "D" * 9000})
        messages = [
            {"role": "user", "content": "Goal: build report."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_older",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args_older},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_older",
                "content": json.dumps({
                    "success": True,
                    "output": "B" * 10000,
                    "error": None,
                    "files_changed": ["report-old.md"],
                }),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_latest",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args_latest},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_latest",
                "content": json.dumps({
                    "success": True,
                    "output": "short ok",
                    "error": None,
                    "files_changed": ["report.md"],
                }),
            },
            {"role": "user", "content": "Historical context " + ("C " * 1400)},
            {
                "role": "assistant",
                "content": "LATEST CRITICAL: keep acceptance criteria unchanged.",
            },
            {"role": "user", "content": "LATEST USER STEER: preserve bullet ordering exactly."},
        ]

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=240)
        labels = [label for label, _max, _size in runner._compactor.calls]
        arg_idx = next(i for i, label in enumerate(labels) if "assistant tool-call args" in label)
        context_indices = [
            idx for idx, label in enumerate(labels)
            if label.endswith("context")
        ]
        if context_indices:
            assert arg_idx < min(context_indices)
        assert any(
            msg.get("content") == "LATEST USER STEER: preserve bullet ordering exactly."
            for msg in compacted
            if isinstance(msg, dict)
        )

    @pytest.mark.asyncio
    async def test_preserve_latest_critical_turns_under_pressure(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=1500)
        messages = [{"role": "user", "content": "Goal: analyze telemetry."}]
        for idx in range(10):
            messages.append({"role": "user", "content": f"Old context {idx}: " + ("x " * 600)})
            messages.append({
                "role": "assistant",
                "content": f"Old assistant note {idx}: " + ("y " * 400),
            })
        latest_assistant = "LATEST CRITICAL ASSISTANT: keep file names exact."
        latest_user = "LATEST CRITICAL USER: do not drop failed-subtask IDs."
        messages.extend([
            {"role": "assistant", "content": latest_assistant},
            {"role": "user", "content": latest_user},
        ])

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=240)
        contents = [
            msg.get("content", "")
            for msg in compacted
            if isinstance(msg, dict) and isinstance(msg.get("content"), str)
        ]
        assert latest_assistant in contents
        assert latest_user in contents

    @pytest.mark.asyncio
    async def test_critical_tier_old_context_merge_without_latest_instruction_loss(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=600)
        huge_args = json.dumps({"path": "output.md", "content": "Z" * 18000})
        messages = [{"role": "user", "content": "Goal: execute and verify all subtasks."}]
        for idx in range(6):
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{idx}",
                "content": json.dumps({
                    "success": True,
                    "output": "Q" * 8000,
                    "error": None,
                    "files_changed": ["output.md"],
                }),
            })
            messages.append({"role": "user", "content": f"Older narrative {idx}: " + ("r " * 500)})

        latest_instruction = "LATEST INSTRUCTION: preserve the rubric schema exactly."
        messages.extend([
            {"role": "assistant", "content": latest_instruction},
            {"role": "user", "content": "LATEST USER: keep unresolved evidence IDs in output."},
        ])

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=240)
        contents = [
            msg.get("content", "")
            for msg in compacted
            if isinstance(msg, dict) and isinstance(msg.get("content"), str)
        ]
        assert any(content.startswith("Prior compacted context:\n") for content in contents)
        assert latest_instruction in contents
        assert runner._last_compaction_diagnostics["compaction_pressure_tier"] == "critical"

    @pytest.mark.asyncio
    async def test_memory_extractor_compacts_large_tool_args(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=2000)
        runner._config = Config()
        runner._subtask_deadline_monotonic = time.monotonic() + 120.0
        runner._memory = AsyncMock()
        runner._memory.store_many = AsyncMock()
        runner._event_bus = None
        runner._validator = MagicMock()
        runner._validator.validate_json_response = MagicMock(
            return_value=MagicMock(valid=False, parsed=None),
        )

        class _ExtractorModel:
            name = "mock-extractor"
            roles = ["extractor"]

            def __init__(self):
                self.messages = []

            async def complete(self, messages, **kwargs):
                del kwargs
                self.messages.append(messages)
                return ModelResponse(
                    text="[]",
                    usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
                )

        extractor_model = _ExtractorModel()
        runner._router = MagicMock()
        runner._router.select = MagicMock(return_value=extractor_model)
        runner._prompts = MagicMock()
        runner._prompts.build_extractor_prompt = MagicMock(
            side_effect=lambda subtask_id, tool_calls_formatted, model_output: (
                f"SUBTASK {subtask_id}\nTOOLS\n{tool_calls_formatted}\nOUTPUT\n{model_output}"
            ),
        )

        result = SubtaskResult(
            summary="Execution completed with output artifacts.",
            tool_calls=[
                ToolCallRecord(
                    tool="document_write",
                    args={"path": "report.md", "content": "A" * 200_000},
                    result=ToolResult.ok("ok"),
                ),
            ],
        )

        await runner._extract_memory("task-1", "subtask-1", result)

        assert extractor_model.messages
        prompt = extractor_model.messages[0][0]["content"]
        assert "document_write(" in prompt
        assert "A" * 500 not in prompt
        runner._memory.store_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_timeout_near_skips_nonessential_compaction(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=800)
        huge_args_old = json.dumps({"path": "report-old.md", "content": "A" * 12000})
        huge_args_latest = json.dumps({"path": "report.md", "content": "D" * 8000})
        messages = [
            {"role": "user", "content": "Goal: finish report."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_old",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args_old},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": json.dumps({
                    "success": True,
                    "output": "B" * 9000,
                    "error": None,
                    "files_changed": ["report-old.md"],
                }),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_latest",
                    "type": "function",
                    "function": {"name": "document_write", "arguments": huge_args_latest},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_latest",
                "content": json.dumps({
                    "success": True,
                    "output": "done",
                    "error": None,
                    "files_changed": ["report.md"],
                }),
            },
            {"role": "user", "content": "Old narrative " + ("x " * 800)},
            {"role": "assistant", "content": "LATEST CRITICAL: keep final file name unchanged."},
        ]

        await runner._compact_messages_for_model(messages, remaining_seconds=5)
        labels = [label for label, _max, _size in runner._compactor.calls]
        assert any("assistant tool-call args" in label for label in labels)
        assert all("tool message output" not in label for label in labels)
        assert runner._last_compaction_diagnostics["compaction_skipped_reason"] == "timeout_guard"

    @pytest.mark.asyncio
    async def test_no_hard_truncation_marker_inserted_by_runner_compaction_path(self):
        runner = self._make_runner_for_tiered_compaction(context_budget=700)
        messages = [{"role": "user", "content": "Goal: reduce context safely."}]
        for idx in range(8):
            messages.append({"role": "user", "content": f"Context {idx}: " + ("data " * 1000)})
        messages.append({"role": "assistant", "content": "LATEST CRITICAL: keep all file paths."})

        compacted = await runner._compact_messages_for_model(messages, remaining_seconds=180)
        serialized = json.dumps(compacted, ensure_ascii=False, default=str)
        assert "...[truncated]..." not in serialized

    def test_emit_tool_event_emits_artifact_confinement_violation(self):
        from loom.engine.runner import SubtaskRunner

        bus = EventBus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._event_bus = bus

        runner._emit_tool_event(
            TOOL_CALL_COMPLETED,
            "task-1",
            "subtask-1",
            "write_file",
            {"path": "../outside.md"},
            result=ToolResult.fail(
                "Safety violation: Path '/tmp/outside.md' escapes workspace '/tmp/run'."
            ),
            workspace=Path("/tmp/run"),
        )

        event_types = [event.event_type for event in events]
        assert TOOL_CALL_COMPLETED in event_types
        assert ARTIFACT_CONFINEMENT_VIOLATION in event_types
        violation = next(
            event for event in events
            if event.event_type == ARTIFACT_CONFINEMENT_VIOLATION
        )
        assert violation.data["attempted_path"] == "../outside.md"

    def test_tool_call_completed_payload_contract_additive(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        runner._emit_tool_event(
            TOOL_CALL_COMPLETED,
            "task-1",
            "subtask-1",
            "web_fetch",
            {"url": "https://example.com/report.pdf"},
            result=ToolResult.ok("ok"),
        )

        tool_event = next(event for event in events if event.event_type == TOOL_CALL_COMPLETED)
        assert tool_event.data == {
            "subtask_id": "subtask-1",
            "tool": "web_fetch",
            "args": {"url": "https://example.com/report.pdf"},
            "success": True,
            "error": "",
            "files_changed": [],
            "files_changed_paths": [],
        }

    def test_artifact_ingest_telemetry_required_fields_and_redaction(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        result = ToolResult.ok(
            "Fetched PDF artifact",
            data={
                "url": "https://example.com/report.pdf?token=secret#fragment",
                "content_kind": "pdf",
                "content_type": "application/pdf",
                "artifact_ref": "af_1234abcd",
                "artifact_workspace_relpath": ".loom_artifacts/fetched/s1/af_1234abcd.pdf",
                "size_bytes": 4096,
                "declared_size_bytes": 5000,
                "handler": "pdf_handler",
                "extracted_chars": 1800,
                "extraction_truncated": True,
                "handler_metadata": {"details": "x" * 800},
            },
        )

        runner._emit_artifact_ingest_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="web_fetch",
            tool_args={"url": "https://example.com/report.pdf?token=secret#fragment"},
            result=result,
        )

        classified = next(
            event for event in events
            if event.event_type == ARTIFACT_INGEST_CLASSIFIED
        )
        completed = next(
            event for event in events
            if event.event_type == ARTIFACT_INGEST_COMPLETED
        )
        for event in (classified, completed):
            payload = event.data
            assert payload["subtask_id"] == "subtask-1"
            assert payload["tool"] == "web_fetch"
            assert payload["status"] == "ok"
            assert payload["url"] == "https://example.com/report.pdf"
            assert "token=secret" not in payload["url"]
            assert payload["content_kind"] == "pdf"
            assert payload["content_type"] == "application/pdf"
            assert payload["artifact_ref"] == "af_1234abcd"
            assert payload["artifact_workspace_relpath"].startswith(".loom_artifacts/")
            assert payload["size_bytes"] == 4096
            assert payload["declared_size_bytes"] == 5000
            assert payload["handler"] == "pdf_handler"
            assert payload["extracted_chars"] == 1800
            assert payload["extraction_truncated"] is True
            metadata_payload = payload.get("handler_metadata")
            assert isinstance(metadata_payload, dict)
            assert metadata_payload.get("_loom_meta") == "metadata_omitted"
            assert metadata_payload.get("original_type") == "dict"
            assert isinstance(metadata_payload.get("sha1"), str)
            assert metadata_payload["sha1"]
            assert "truncated" not in json.dumps(metadata_payload, ensure_ascii=False)
        assert runner._active_subtask_telemetry_counters["artifact_ingests"] == 1

    def test_artifact_retention_event_emitted_only_when_deletions_occur(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        no_delete = ToolResult.ok(
            "ok",
            data={
                "url": "https://example.com/report.pdf",
                "content_kind": "pdf",
                "content_type": "application/pdf",
                "artifact_ref": "af_no_delete",
                "artifact_workspace_relpath": ".loom_artifacts/fetched/s1/af_no_delete.pdf",
                "artifact_retention": {
                    "scopes_scanned": 1,
                    "files_deleted": 0,
                    "bytes_deleted": 0,
                },
            },
        )
        runner._emit_artifact_ingest_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="web_fetch",
            tool_args={"url": "https://example.com/report.pdf"},
            result=no_delete,
        )
        assert ARTIFACT_RETENTION_PRUNED not in [event.event_type for event in events]

        with_delete = ToolResult.ok(
            "ok",
            data={
                "url": "https://example.com/report.pdf",
                "content_kind": "pdf",
                "content_type": "application/pdf",
                "artifact_ref": "af_deleted",
                "artifact_workspace_relpath": ".loom_artifacts/fetched/s1/af_deleted.pdf",
                "artifact_retention": {
                    "scopes_scanned": 2,
                    "files_deleted": 3,
                    "bytes_deleted": 9000,
                },
            },
        )
        runner._emit_artifact_ingest_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="web_fetch",
            tool_args={"url": "https://example.com/report.pdf"},
            result=with_delete,
        )
        retention = next(
            event for event in events
            if event.event_type == ARTIFACT_RETENTION_PRUNED
        )
        assert retention.data["files_deleted"] == 3
        assert retention.data["bytes_deleted"] == 9000
        assert runner._active_subtask_telemetry_counters["artifact_retention_deletes"] == 3

    def test_artifact_read_completed_emits_success_and_failure(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        success_result = ToolResult.ok(
            "ok",
            data={
                "source_url": "https://example.com/report.pdf?sig=hidden",
                "content_kind": "pdf",
                "content_type": "application/pdf",
                "artifact_ref": "af_read_ok",
                "artifact_workspace_relpath": ".loom_artifacts/fetched/s1/af_read_ok.pdf",
                "handler": "pdf_handler",
                "extracted_chars": 1200,
                "extraction_truncated": False,
            },
        )
        runner._emit_artifact_read_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="read_artifact",
            tool_args={"artifact_ref": "af_read_ok"},
            result=success_result,
        )

        failed_result = ToolResult.fail("Artifact not found")
        runner._emit_artifact_read_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            tool_name="read_artifact",
            tool_args={"artifact_ref": "af_missing"},
            result=failed_result,
        )

        read_events = [
            event for event in events
            if event.event_type == ARTIFACT_READ_COMPLETED
        ]
        assert len(read_events) == 2
        assert read_events[0].data["status"] == "ok"
        assert read_events[0].data["url"] == "https://example.com/report.pdf"
        assert read_events[1].data["status"] == "error"
        assert read_events[1].data["artifact_ref"] == "af_missing"
        assert runner._active_subtask_telemetry_counters["artifact_reads"] == 2

    def test_compaction_and_overflow_telemetry_events(self):
        runner = self._make_runner_for_telemetry()
        events = []
        runner._event_bus.subscribe_all(lambda event: events.append(event))

        runner._emit_compaction_policy_decision_from_diagnostics(
            task_id="task-1",
            subtask_id="subtask-1",
        )
        decision_event = next(
            event for event in events
            if event.event_type == COMPACTION_POLICY_DECISION
        )
        assert decision_event.data["decision"] == "compact_tool"
        assert decision_event.data["reason"] == "tool_output_compacted"

        runner._emit_overflow_fallback_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            report={
                "overflow_fallback_applied": True,
                "overflow_fallback_rewritten_messages": 2,
                "overflow_fallback_chars_reduced": 6400,
                "overflow_fallback_preserved_recent_messages": 1,
            },
        )
        overflow_events = [
            event for event in events
            if event.event_type == OVERFLOW_FALLBACK_APPLIED
        ]
        assert len(overflow_events) == 1
        overflow_payload = overflow_events[0].data
        assert overflow_payload["decision"] == "fallback_rewrite"
        assert overflow_payload["rewritten_messages"] == 2
        assert overflow_payload["chars_reduced"] == 6400
        assert overflow_payload["preserved_recent_messages"] == 1

        runner._emit_overflow_fallback_telemetry(
            task_id="task-1",
            subtask_id="subtask-1",
            report={"overflow_fallback_applied": False},
        )
        overflow_events = [
            event for event in events
            if event.event_type == OVERFLOW_FALLBACK_APPLIED
        ]
        assert len(overflow_events) == 1
        assert runner._active_subtask_telemetry_counters["overflow_fallback_count"] == 1

    def test_tool_iteration_budget_uses_global_limit(self):
        from loom.engine.runner import SubtaskRunner

        research_subtask = Subtask(
            id="collect-evidence",
            description="Research and collect supporting evidence.",
        )
        verify_subtask = Subtask(
            id="verify-findings",
            description="Run verification checks on outputs.",
        )
        final_subtask = Subtask(
            id="evaluate-select-twelve",
            description=(
                "Apply selection rubric to longlist to select exactly 12 final cases."
            ),
        )
        research_budget = SubtaskRunner._tool_iteration_budget(
            subtask=research_subtask,
            retry_strategy="",
            has_expected_deliverables=False,
        )
        verify_budget = SubtaskRunner._tool_iteration_budget(
            subtask=verify_subtask,
            retry_strategy="",
            has_expected_deliverables=False,
        )
        remediation_budget = SubtaskRunner._tool_iteration_budget(
            subtask=research_subtask,
            retry_strategy="evidence_gap",
            has_expected_deliverables=True,
        )
        final_budget = SubtaskRunner._tool_iteration_budget(
            subtask=final_subtask,
            retry_strategy="",
            has_expected_deliverables=False,
        )
        custom_budget = SubtaskRunner._tool_iteration_budget(
            subtask=final_subtask,
            retry_strategy="rate_limit",
            has_expected_deliverables=True,
            base_budget=37,
        )

        assert research_budget == SubtaskRunner.MAX_TOOL_ITERATIONS
        assert verify_budget == SubtaskRunner.MAX_TOOL_ITERATIONS
        assert final_budget == SubtaskRunner.MAX_TOOL_ITERATIONS
        assert remediation_budget == SubtaskRunner.MAX_TOOL_ITERATIONS
        assert custom_budget == 37

    def test_deliverable_policy_blocks_variant_and_noncanonical_retry_paths(self, tmp_path):
        from loom.engine.runner import SubtaskRunner

        variant_error = SubtaskRunner._validate_deliverable_write_policy(
            tool_name="write_file",
            tool_args={"path": "analysis-v2.md"},
            workspace=tmp_path,
            expected_deliverables=["analysis.md"],
            enforce_deliverable_paths=False,
            edit_existing_only=False,
        )
        assert variant_error is not None
        assert "analysis.md" in variant_error

        noncanonical_error = SubtaskRunner._validate_deliverable_write_policy(
            tool_name="write_file",
            tool_args={"path": "scratch-notes.md"},
            workspace=tmp_path,
            expected_deliverables=["analysis.md"],
            enforce_deliverable_paths=True,
            edit_existing_only=True,
        )
        assert noncanonical_error is not None
        assert "Unexpected target(s)" in noncanonical_error
