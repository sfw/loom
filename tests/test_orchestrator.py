"""Tests for the orchestrator loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config, ExecutionConfig, VerificationConfig
from loom.engine.orchestrator import Orchestrator, SubtaskResult, ToolCallRecord, create_task
from loom.engine.verification import VerificationResult
from loom.events.bus import EventBus
from loom.events.types import (
    ARTIFACT_CONFINEMENT_VIOLATION,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLAN_READY,
    TASK_PLANNING,
    TOOL_CALL_COMPLETED,
)
from loom.models.base import ModelConnectionError, ModelResponse, TokenUsage, ToolCall
from loom.models.router import ModelRouter
from loom.processes.schema import PhaseTemplate, ProcessDefinition
from loom.prompts.assembler import PromptAssembler
from loom.recovery.retry import AttemptRecord, RetryStrategy
from loom.state.task_state import (
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


def _make_mock_router(plan_response_text: str = "", executor_responses=None):
    """Create a mock router that returns prescribed responses."""
    router = MagicMock(spec=ModelRouter)

    planner_model = AsyncMock()
    planner_model.name = "mock-planner"
    planner_model.complete = AsyncMock(return_value=ModelResponse(
        text=plan_response_text,
        usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
    ))

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
    async def test_critical_unconfirmed_triggers_confirm_or_prune_before_abort(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Critical"}]
        })
        state_manager = _make_state_manager(tmp_path)
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
            changelog=None,
            subtask_id="s1",
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

    @staticmethod
    def _make_runner_for_compaction():
        from loom.engine.runner import SubtaskRunner

        runner = SubtaskRunner.__new__(SubtaskRunner)
        runner._compactor = TestSubtaskRunnerContextBudget._FakeCompactor()
        return runner

    class _NoopCompactor:
        async def compact(self, text: str, *, max_chars: int, label: str = "") -> str:
            return str(text or "")

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
    async def test_compact_text_hard_caps_when_compactor_returns_oversize(self):
        runner = self._make_runner_for_compaction()
        runner._compactor = self._NoopCompactor()

        value = "A" * 5000
        compacted = await runner._compact_text(
            value,
            max_chars=120,
            label="oversize guard",
        )

        assert len(compacted) <= 120
        assert compacted.startswith("A")
        assert compacted.endswith("A")

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
