"""Orchestrator execution loop tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from loom.config import Config, ExecutionConfig, VerificationConfig
from loom.engine.orchestrator import (
    Orchestrator,
    SubtaskResult,
    SubtaskResultStatus,
    ToolCallRecord,
)
from loom.engine.verification import VerificationResult
from loom.events.types import (
    FORBIDDEN_CANONICAL_WRITE_BLOCKED,
    PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED,
    PLACEHOLDER_PRUNED,
    PLACEHOLDER_REMEDIATION_UNRESOLVED,
    SEALED_RESEAL_APPLIED,
    SUBTASK_BLOCKED,
    SUBTASK_OUTPUT_CONFLICT_DEFERRED,
    SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING,
    TASK_FAILED,
    TASK_STALLED,
)
from loom.models.base import ModelConnectionError, ModelResponse, TokenUsage, ToolCall
from loom.models.router import ModelRouter
from loom.processes.schema import OutputCoordination, PhaseTemplate, ProcessDefinition
from loom.prompts.assembler import PromptAssembler
from loom.recovery.retry import AttemptRecord, RetryStrategy
from loom.state.task_state import Plan, Subtask, SubtaskStatus, TaskStatus
from loom.tools.registry import ToolResult
from tests.orchestrator.conftest import (
    _make_config,
    _make_event_bus,
    _make_mock_memory,
    _make_mock_prompts,
    _make_mock_router,
    _make_mock_tools,
    _make_process_with_critical_behavior,
    _make_state_manager,
    _make_task,
)


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
        assert kwargs["forbidden_deliverables"] == []
        assert kwargs["enforce_deliverable_paths"] is True
        assert kwargs["edit_existing_only"] is True
        assert kwargs["retry_strategy"] == RetryStrategy.UNCONFIRMED_DATA.value
        assert "CANONICAL DELIVERABLE FILES FOR THIS SUBTASK" in kwargs["retry_context"]

    @pytest.mark.asyncio
    async def test_dispatch_subtask_retry_maps_deliverables_from_phase_hint(self, tmp_path):
        plan_json = json.dumps({
            "subtasks": [{"id": "phase-a", "description": "Analyze"}]
        })
        process = ProcessDefinition(
            name="test-process",
            description="Test",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze telecom funding programs",
                    deliverables=["analysis.md"],
                ),
                PhaseTemplate(
                    id="phase-b",
                    description="Synthesize report",
                    deliverables=["report.md"],
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
        subtask = Subtask(
            id="research-bell-media-crave-funding",
            description="Research Bell and Crave funding pathways",
            phase_id="phase-a",
        )
        task.plan.subtasks = [subtask]

        successful_call = ToolCallRecord(
            tool="write_file",
            args={"path": "analysis.md"},
            result=ToolResult.ok("ok", files_changed=["analysis.md"]),
        )
        attempts = {
            "research-bell-media-crave-funding": [
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
        assert kwargs["forbidden_deliverables"] == []
        assert kwargs["enforce_deliverable_paths"] is True
        assert kwargs["edit_existing_only"] is True
        assert "CANONICAL DELIVERABLE FILES FOR THIS SUBTASK" in kwargs["retry_context"]

    @pytest.mark.asyncio
    async def test_dispatch_subtask_fan_in_worker_blocks_canonical_outputs(self, tmp_path):
        process = ProcessDefinition(
            name="fan-in-process",
            description="Test",
            output_coordination=OutputCoordination(strategy="fan_in"),
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task()
        subtask = Subtask(
            id="phase-a-worker",
            description="Collect artifacts",
            phase_id="phase-a",
        )
        task.plan.subtasks = [subtask]
        attempts = {
            subtask.id: [
                AttemptRecord(
                    attempt=1,
                    tier=1,
                    feedback="Retry with fixes",
                    retry_strategy=RetryStrategy.UNCONFIRMED_DATA,
                ),
            ],
        }
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status="success", summary="ok"),
            VerificationResult(tier=1, passed=True),
        ))

        await orch._dispatch_subtask(task, subtask, attempts)

        _, kwargs = orch._runner.run.await_args
        assert kwargs["expected_deliverables"] == []
        assert kwargs["forbidden_deliverables"] == ["analysis.md"]
        assert kwargs["allowed_output_prefixes"] == [
            "loom/phase-artifacts/phase-a/phase-a-worker",
        ]
        assert kwargs["enforce_deliverable_paths"] is False
        assert kwargs["edit_existing_only"] is False
        assert "OUTPUT COORDINATION MODE: FAN-IN WORKER" in kwargs["retry_context"]

    @pytest.mark.asyncio
    async def test_dispatch_subtask_fan_in_finalizer_owns_canonical_outputs(self, tmp_path):
        process = ProcessDefinition(
            name="fan-in-process",
            description="Test",
            output_coordination=OutputCoordination(strategy="fan_in"),
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task()
        subtask = Subtask(
            id=ProcessDefinition.phase_finalizer_id("phase-a"),
            description="Finalize outputs",
            phase_id="phase-a",
            output_role="phase_finalizer",
            output_strategy="fan_in",
        )
        task.plan.subtasks = [subtask]
        attempts = {
            subtask.id: [
                AttemptRecord(
                    attempt=1,
                    tier=1,
                    feedback="Retry with fixes",
                    retry_strategy=RetryStrategy.UNCONFIRMED_DATA,
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
        assert kwargs["forbidden_deliverables"] == []
        assert kwargs["allowed_output_prefixes"] == []
        assert kwargs["enforce_deliverable_paths"] is True
        assert kwargs["edit_existing_only"] is True
        assert "OUTPUT COORDINATION MODE: FAN-IN PHASE FINALIZER" in kwargs["retry_context"]

    def test_prepare_plan_injects_fan_in_finalizer_and_remaps_downstream_dependencies(
        self,
        tmp_path,
    ):
        process = ProcessDefinition(
            name="fan-in-plan-alignment",
            description="Test",
            output_coordination=OutputCoordination(strategy="fan_in"),
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    depends_on=[],
                    deliverables=["analysis.md"],
                ),
                PhaseTemplate(
                    id="phase-b",
                    description="Synthesize",
                    depends_on=["phase-a"],
                    deliverables=["report.md"],
                ),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task()
        plan = Plan(
            subtasks=[
                Subtask(id="phase-a", description="Analyze", depends_on=[], phase_id="phase-a"),
                Subtask(
                    id="phase-b",
                    description="Synthesize",
                    depends_on=["phase-a"],
                    phase_id="phase-b",
                ),
            ],
            version=1,
        )

        prepared = orch._prepare_plan_for_execution(
            task=task,
            plan=plan,
            context="planner",
        )

        finalizer_id = ProcessDefinition.phase_finalizer_id("phase-a")
        finalizer = next((s for s in prepared.subtasks if s.id == finalizer_id), None)
        assert finalizer is not None
        assert finalizer.phase_id == "phase-a"
        assert finalizer.output_role == "phase_finalizer"
        assert finalizer.output_strategy == "fan_in"
        assert "phase-a" in finalizer.depends_on
        phase_b = next((s for s in prepared.subtasks if s.id == "phase-b"), None)
        assert phase_b is not None
        assert phase_b.depends_on == [finalizer_id]

    @pytest.mark.asyncio
    async def test_output_conflict_guard_serializes_same_deliverable_writers(self, tmp_path):
        process = ProcessDefinition(
            name="output-conflict-process",
            description="Test",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=20,
            max_parallel_subtasks=3,
            auto_approve_confidence_threshold=0.8,
            enable_streaming=False,
        ))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text="{}"),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=cfg,
            process=process,
        )
        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(id="s1", description="worker 1", phase_id="phase-a"),
                Subtask(id="s2", description="worker 2", phase_id="phase-a"),
                Subtask(id="s3", description="worker 3", phase_id="phase-a"),
            ],
            version=1,
        )

        active = 0
        max_active = 0

        async def dispatch_side_effect(_task, subtask, _attempts):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return (
                subtask,
                SubtaskResult(status="success", summary="ok"),
                VerificationResult(tier=1, passed=True),
            )

        orch._dispatch_subtask = AsyncMock(side_effect=dispatch_side_effect)

        result = await orch.execute_task(task, reuse_existing_plan=True)

        assert result.status == TaskStatus.COMPLETED
        assert max_active == 1

    @pytest.mark.asyncio
    async def test_output_conflict_guard_preserves_parallelism_for_disjoint_deliverables(
        self,
        tmp_path,
    ):
        process = ProcessDefinition(
            name="disjoint-deliverables-process",
            description="Test",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
                PhaseTemplate(
                    id="phase-b",
                    description="Synthesize",
                    deliverables=["report.md"],
                ),
            ],
        )
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=20,
            max_parallel_subtasks=2,
            auto_approve_confidence_threshold=0.8,
            enable_streaming=False,
        ))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text="{}"),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=cfg,
            process=process,
        )
        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(id="s1", description="worker 1", phase_id="phase-a"),
                Subtask(id="s2", description="worker 2", phase_id="phase-b"),
            ],
            version=1,
        )

        active = 0
        max_active = 0

        async def dispatch_side_effect(_task, subtask, _attempts):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return (
                subtask,
                SubtaskResult(status="success", summary="ok"),
                VerificationResult(tier=1, passed=True),
            )

        orch._dispatch_subtask = AsyncMock(side_effect=dispatch_side_effect)

        result = await orch.execute_task(task, reuse_existing_plan=True)

        assert result.status == TaskStatus.COMPLETED
        assert max_active == 2

    @pytest.mark.asyncio
    async def test_output_conflict_deferral_emits_deferred_and_starvation_events(
        self,
        tmp_path,
    ):
        process = ProcessDefinition(
            name="output-conflict-events-process",
            description="Test",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=20,
            max_parallel_subtasks=2,
            auto_approve_confidence_threshold=0.8,
            enable_streaming=False,
        ))
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda e: events.append(e))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text="{}"),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=cfg,
            process=process,
        )
        orch._output_conflict_starvation_threshold = 1
        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(id="s1", description="worker 1", phase_id="phase-a"),
                Subtask(id="s2", description="worker 2", phase_id="phase-a"),
            ],
            version=1,
        )

        async def dispatch_side_effect(_task, subtask, _attempts):
            await asyncio.sleep(0.01)
            return (
                subtask,
                SubtaskResult(status="success", summary="ok"),
                VerificationResult(tier=1, passed=True),
            )

        orch._dispatch_subtask = AsyncMock(side_effect=dispatch_side_effect)

        result = await orch.execute_task(task, reuse_existing_plan=True)

        assert result.status == TaskStatus.COMPLETED
        assert any(e.event_type == SUBTASK_OUTPUT_CONFLICT_DEFERRED for e in events)
        assert any(
            e.event_type == SUBTASK_OUTPUT_CONFLICT_STARVATION_WARNING
            for e in events
        )

    @pytest.mark.asyncio
    async def test_output_conflict_guard_can_be_disabled(self, tmp_path):
        process = ProcessDefinition(
            name="output-conflict-disabled-process",
            description="Test",
            output_coordination=OutputCoordination(enforce_single_writer=False),
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=20,
            max_parallel_subtasks=2,
            auto_approve_confidence_threshold=0.8,
            enable_streaming=False,
        ))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text="{}"),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=cfg,
            process=process,
        )
        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(id="s1", description="worker 1", phase_id="phase-a"),
                Subtask(id="s2", description="worker 2", phase_id="phase-a"),
            ],
            version=1,
        )
        active = 0
        max_active = 0

        async def dispatch_side_effect(_task, subtask, _attempts):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return (
                subtask,
                SubtaskResult(status="success", summary="ok"),
                VerificationResult(tier=1, passed=True),
            )

        orch._dispatch_subtask = AsyncMock(side_effect=dispatch_side_effect)
        result = await orch.execute_task(task, reuse_existing_plan=True)

        assert result.status == TaskStatus.COMPLETED
        assert max_active == 2

    @pytest.mark.asyncio
    async def test_output_conflict_fail_fast_policy_aborts_run(self, tmp_path):
        process = ProcessDefinition(
            name="output-conflict-fail-fast-process",
            description="Test",
            output_coordination=OutputCoordination(conflict_policy="fail_fast"),
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        cfg = Config(execution=ExecutionConfig(
            max_subtask_retries=0,
            max_loop_iterations=20,
            max_parallel_subtasks=2,
            auto_approve_confidence_threshold=0.8,
            enable_streaming=False,
        ))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text="{}"),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=cfg,
            process=process,
        )
        task = _make_task()
        task.plan = Plan(
            subtasks=[
                Subtask(id="s1", description="worker 1", phase_id="phase-a"),
                Subtask(id="s2", description="worker 2", phase_id="phase-a"),
            ],
            version=1,
        )
        orch._dispatch_subtask = AsyncMock()

        result = await orch.execute_task(task, reuse_existing_plan=True)

        assert result.status == TaskStatus.FAILED
        assert result.errors_encountered
        assert "fail_fast" in result.errors_encountered[-1].error
        orch._dispatch_subtask.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_subtask_transactional_finalizer_commit_missing_stage_fails_safe(
        self,
        tmp_path,
    ):
        process = ProcessDefinition(
            name="fan-in-transactional-rollback",
            description="Test",
            output_coordination=OutputCoordination(strategy="fan_in", publish_mode="transactional"),
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task(workspace=str(tmp_path))
        task.metadata["run_id"] = "run-test"
        (tmp_path / "analysis.md").write_text("before rollback", encoding="utf-8")
        subtask = Subtask(
            id=ProcessDefinition.phase_finalizer_id("phase-a"),
            description="Finalize outputs",
            phase_id="phase-a",
            output_role="phase_finalizer",
            output_strategy="fan_in",
        )
        task.plan.subtasks = [subtask]

        async def run_side_effect(*_args, **_kwargs):
            return (
                SubtaskResult(status=SubtaskResultStatus.SUCCESS, summary="prepared stage"),
                VerificationResult(tier=1, passed=True, feedback="ok"),
            )

        orch._runner.run = AsyncMock(side_effect=run_side_effect)

        _, result, verification = await orch._dispatch_subtask(
            task,
            subtask,
            attempts_by_subtask={},
        )

        assert result.status == SubtaskResultStatus.FAILED
        assert not verification.passed
        assert verification.reason_code == "output_publish_commit_failed"
        assert (tmp_path / "analysis.md").read_text(encoding="utf-8") == "before rollback"
        assert "Transactional publish failed: missing staged output" in result.summary

    @pytest.mark.asyncio
    async def test_dispatch_subtask_transactional_finalizer_stage_commit_success_updates_seals(
        self,
        tmp_path,
    ):
        process = ProcessDefinition(
            name="fan-in-transactional-commit",
            description="Test",
            output_coordination=OutputCoordination(strategy="fan_in", publish_mode="transactional"),
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task(workspace=str(tmp_path))
        task.metadata["run_id"] = "run-test"
        (tmp_path / "analysis.md").write_text("before", encoding="utf-8")
        subtask = Subtask(
            id=ProcessDefinition.phase_finalizer_id("phase-a"),
            description="Finalize outputs",
            phase_id="phase-a",
            output_role="phase_finalizer",
            output_strategy="fan_in",
        )
        task.plan.subtasks = [subtask]

        async def run_side_effect(*_args, **kwargs):
            stage_targets = list(kwargs.get("expected_deliverables", []))
            assert stage_targets
            for stage_target in stage_targets:
                target_path = tmp_path / stage_target
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text("after", encoding="utf-8")
            return (
                SubtaskResult(status=SubtaskResultStatus.SUCCESS, summary="ready"),
                VerificationResult(tier=1, passed=True, feedback="ok"),
            )

        orch._runner.run = AsyncMock(side_effect=run_side_effect)

        _, result, verification = await orch._dispatch_subtask(
            task,
            subtask,
            attempts_by_subtask={},
        )

        assert result.status == SubtaskResultStatus.SUCCESS
        assert verification.passed
        assert (tmp_path / "analysis.md").read_text(encoding="utf-8") == "after"
        seals = task.metadata.get("artifact_seals", {})
        assert "analysis.md" in seals
        assert seals["analysis.md"]["tool"] == "fan_in_commit"
        assert not any(
            path.startswith(".loom/phase-artifacts/")
            or path.startswith("loom/phase-artifacts/")
            for path in seals
        )

    @pytest.mark.asyncio
    async def test_dispatch_subtask_finalizer_require_all_workers_blocks_when_missing_manifest(
        self,
        tmp_path,
    ):
        process = ProcessDefinition(
            name="fan-in-finalizer-require-all",
            description="Test",
            output_coordination=OutputCoordination(strategy="fan_in"),
            phases=[
                PhaseTemplate(id="phase-a", description="Collect", deliverables=["analysis.md"]),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task(workspace=str(tmp_path))
        task.metadata["run_id"] = "run-test"
        worker = Subtask(
            id="phase-a-worker",
            description="Worker",
            phase_id="phase-a",
            output_role="worker",
            output_strategy="fan_in",
        )
        finalizer = Subtask(
            id=ProcessDefinition.phase_finalizer_id("phase-a"),
            description="Finalize",
            phase_id="phase-a",
            output_role="phase_finalizer",
            output_strategy="fan_in",
        )
        task.plan.subtasks = [worker, finalizer]
        orch._runner.run = AsyncMock()

        _, result, verification = await orch._dispatch_subtask(
            task,
            finalizer,
            attempts_by_subtask={},
        )

        assert result.status == SubtaskResultStatus.FAILED
        assert not verification.passed
        assert verification.reason_code == "finalizer_missing_worker_artifacts"
        orch._runner.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_subtask_finalizer_allow_partial_does_not_block_missing_manifest(
        self,
        tmp_path,
    ):
        process = ProcessDefinition(
            name="fan-in-finalizer-allow-partial",
            description="Test",
            output_coordination=OutputCoordination(
                strategy="fan_in",
                publish_mode="best_effort",
                finalizer_input_policy="allow_partial",
            ),
            phases=[
                PhaseTemplate(id="phase-a", description="Collect", deliverables=["analysis.md"]),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task(workspace=str(tmp_path))
        task.metadata["run_id"] = "run-test"
        worker = Subtask(
            id="phase-a-worker",
            description="Worker",
            phase_id="phase-a",
            output_role="worker",
            output_strategy="fan_in",
        )
        finalizer = Subtask(
            id=ProcessDefinition.phase_finalizer_id("phase-a"),
            description="Finalize",
            phase_id="phase-a",
            output_role="phase_finalizer",
            output_strategy="fan_in",
        )
        task.plan.subtasks = [worker, finalizer]
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(status=SubtaskResultStatus.SUCCESS, summary="ok"),
            VerificationResult(tier=1, passed=True, feedback="ok"),
        ))

        _, result, verification = await orch._dispatch_subtask(
            task,
            finalizer,
            attempts_by_subtask={},
        )

        assert result.status == SubtaskResultStatus.SUCCESS
        assert verification.passed
        orch._runner.run.assert_awaited()

    @pytest.mark.asyncio
    async def test_dispatch_subtask_finalizer_manifest_only_read_violation_fails(
        self,
        tmp_path,
    ):
        process = ProcessDefinition(
            name="fan-in-manifest-enforcement",
            description="Test",
            output_coordination=OutputCoordination(
                strategy="fan_in",
                publish_mode="best_effort",
                finalizer_input_policy="allow_partial",
            ),
            phases=[
                PhaseTemplate(id="phase-a", description="Collect", deliverables=["analysis.md"]),
            ],
        )
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
            process=process,
        )
        task = _make_task(workspace=str(tmp_path))
        task.metadata["run_id"] = "run-test"
        worker = Subtask(
            id="phase-a-worker",
            description="Worker",
            phase_id="phase-a",
            output_role="worker",
            output_strategy="fan_in",
        )
        finalizer = Subtask(
            id=ProcessDefinition.phase_finalizer_id("phase-a"),
            description="Finalize",
            phase_id="phase-a",
            output_role="phase_finalizer",
            output_strategy="fan_in",
        )
        task.plan.subtasks = [worker, finalizer]

        allowed_path = (
            ".loom/phase-artifacts/run-test/phase-a/phase-a-worker/notes.md"
        )
        disallowed_path = (
            ".loom/phase-artifacts/run-test/phase-a/unexpected-worker/notes.md"
        )
        allowed_file = tmp_path / allowed_path
        disallowed_file = tmp_path / disallowed_path
        allowed_file.parent.mkdir(parents=True, exist_ok=True)
        disallowed_file.parent.mkdir(parents=True, exist_ok=True)
        allowed_file.write_text("allowed", encoding="utf-8")
        disallowed_file.write_text("blocked", encoding="utf-8")

        manifest_path = (
            tmp_path
            / ".loom/phase-artifacts/run-test/phase-a/manifest.jsonl"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_entry = {
            "schema_version": 1,
            "task_id": task.id,
            "run_id": "run-test",
            "phase_id": "phase-a",
            "subtask_id": "phase-a-worker",
            "attempt": 1,
            "generated_at": "2026-03-06T12:00:00",
            "output_role": "worker",
            "output_strategy": "fan_in",
            "artifact_path": allowed_path,
            "content_hash": "",
        }
        manifest_path.write_text(json.dumps(manifest_entry) + "\n", encoding="utf-8")

        read_call = ToolCallRecord(
            tool="read_file",
            args={"path": disallowed_path},
            result=ToolResult.ok("ok"),
        )
        orch._runner.run = AsyncMock(return_value=(
            SubtaskResult(
                status=SubtaskResultStatus.SUCCESS,
                summary="ok",
                tool_calls=[read_call],
            ),
            VerificationResult(tier=1, passed=True, feedback="ok"),
        ))

        _, result, verification = await orch._dispatch_subtask(
            task,
            finalizer,
            attempts_by_subtask={},
        )

        assert result.status == SubtaskResultStatus.FAILED
        assert not verification.passed
        assert verification.reason_code == "manifest_input_policy_violation"
        assert "outside latest worker manifest entries" in (verification.feedback or "")

    @pytest.mark.asyncio
    async def test_dispatch_subtask_emits_forbidden_canonical_write_blocked_event(
        self,
        tmp_path,
    ):
        plan_json = json.dumps({
            "subtasks": [{"id": "phase-a", "description": "Analyze"}]
        })
        process = ProcessDefinition(
            name="forbidden-write-process",
            description="Test",
            phases=[
                PhaseTemplate(
                    id="phase-a",
                    description="Analyze",
                    deliverables=["analysis.md"],
                ),
            ],
        )
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda e: events.append(e))
        tool_call = ToolCall(
            id="call-1",
            name="write_file",
            arguments={"path": "scratch-notes.md", "content": "hello"},
        )
        executor_responses = [
            ModelResponse(
                text="",
                tool_calls=[tool_call],
                usage=TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
            ModelResponse(
                text=json.dumps({
                    "status": "failed",
                    "deliverables_touched": [],
                    "verification_notes": "policy blocked write",
                }),
                usage=TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        ]
        orch = Orchestrator(
            model_router=_make_mock_router(
                plan_response_text=plan_json,
                executor_responses=executor_responses,
            ),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=PromptAssembler(process=process),
            state_manager=_make_state_manager(tmp_path),
            event_bus=bus,
            config=_make_config(),
            process=process,
        )
        task = _make_task(workspace=str(tmp_path))
        subtask = Subtask(id="phase-a", description="Analyze", phase_id="phase-a")
        task.plan.subtasks = [subtask]
        attempts = {
            "phase-a": [
                AttemptRecord(
                    attempt=1,
                    tier=1,
                    feedback="Retry with fixes",
                    retry_strategy=RetryStrategy.UNCONFIRMED_DATA,
                ),
            ],
        }

        _, result, verification = await orch._dispatch_subtask(task, subtask, attempts)

        assert result.status == SubtaskResultStatus.FAILED
        assert verification.reason_code in {"forbidden_output_path", "hard_invariant_failed"}
        blocked = [
            e for e in events if e.event_type == FORBIDDEN_CANONICAL_WRITE_BLOCKED
        ]
        assert blocked
        payload = blocked[-1].data
        assert payload.get("subtask_id") == "phase-a"
        assert payload.get("tool") == "write_file"
        assert "scratch-notes.md" in list(payload.get("attempted_paths", []))

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
        assert SUBTASK_BLOCKED in [e.event_type for e in events]
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
    async def test_confirm_or_prune_runs_deterministic_placeholder_prune_before_model_retry(
        self,
        tmp_path,
    ):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Critical"}]
        })
        state_manager = _make_state_manager(tmp_path)
        process = _make_process_with_critical_behavior("confirm_or_prune_then_queue")
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=bus,
            config=_make_config(),
            process=process,
        )
        task = _make_task(workspace=str(tmp_path))
        subtask = Subtask(
            id="s1",
            description="Critical",
            is_critical_path=True,
            max_retries=0,
        )
        task.plan.subtasks = [subtask]
        state_manager.create(task)
        initial_report = "Evidence status: N/A\n"
        (tmp_path / "report.md").write_text(initial_report, encoding="utf-8")
        initial_sha = hashlib.sha256(initial_report.encode("utf-8")).hexdigest()
        task.metadata["artifact_seals"] = {
            "report.md": {
                "path": "report.md",
                "sha256": initial_sha,
                "sealed_at": "2026-03-01T12:00:00",
                "subtask_id": "seed",
            },
        }

        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="incomplete_deliverable_placeholder",
            severity_class="semantic",
            metadata={
                "failure_class": "recoverable_placeholder",
                "remediation_mode": "confirm_or_prune",
                "placeholder_findings": [{
                    "rule_name": "no-placeholders",
                    "pattern": r"\bN/A\b",
                    "source": "deliverable",
                    "file_path": "report.md",
                    "line": 1,
                    "column": 18,
                    "token": "N/A",
                    "context": "Evidence status: N/A",
                }],
            },
        )

        async def _runner_side_effect(*_args, **_kwargs):
            assert "UNSUPPORTED_NO_EVIDENCE" in (tmp_path / "report.md").read_text(
                encoding="utf-8",
            )
            return (
                SubtaskResult(status="success", summary="runner executed"),
                VerificationResult(tier=1, passed=True, outcome="pass"),
            )

        orch._runner.run = AsyncMock(side_effect=_runner_side_effect)
        orch._handle_success = AsyncMock()

        recovered, _ = await orch._run_confirm_or_prune_remediation(
            task=task,
            subtask=subtask,
            attempts=[],
            verification=verification,
        )

        assert recovered is True
        assert orch._runner.run.await_count == 1
        orch._handle_success.assert_awaited_once()
        assert "UNSUPPORTED_NO_EVIDENCE" in (tmp_path / "report.md").read_text(
            encoding="utf-8",
        )
        event_types = [event.event_type for event in events]
        assert PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED in event_types
        assert PLACEHOLDER_PRUNED in event_types
        assert PLACEHOLDER_REMEDIATION_UNRESOLVED not in event_types
        started_events = [
            event for event in events
            if event.event_type == PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED
        ]
        assert started_events
        assert started_events[-1].data.get("mode") == "deterministic_placeholder_prepass"
        assert started_events[-1].data.get("parent_mode") == "confirm_or_prune_remediation"
        pruned_events = [event for event in events if event.event_type == PLACEHOLDER_PRUNED]
        assert pruned_events
        assert int(pruned_events[-1].data.get("applied_count", 0) or 0) >= 1
        assert pruned_events[-1].data.get("mode") == "deterministic_placeholder_prepass"
        reseal_events = [event for event in events if event.event_type == SEALED_RESEAL_APPLIED]
        assert reseal_events
        assert int(reseal_events[-1].data.get("path_count", 0) or 0) >= 1
        final_report = (tmp_path / "report.md").read_text(encoding="utf-8")
        final_sha = hashlib.sha256(final_report.encode("utf-8")).hexdigest()
        seal = task.metadata.get("artifact_seals", {}).get("report.md", {})
        assert seal.get("sha256") == final_sha
        assert seal.get("previous_sha256") == initial_sha
        assert seal.get("tool") == "deterministic_placeholder_prepass"

    @pytest.mark.asyncio
    async def test_deterministic_placeholder_prune_emits_unresolved_when_no_mutation(
        self,
        tmp_path,
    ):
        plan_json = json.dumps({
            "subtasks": [{"id": "s1", "description": "Critical"}]
        })
        state_manager = _make_state_manager(tmp_path)
        process = _make_process_with_critical_behavior("confirm_or_prune_then_queue")
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text=plan_json),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=state_manager,
            event_bus=bus,
            config=_make_config(),
            process=process,
        )
        task = _make_task(workspace=str(tmp_path))
        subtask = Subtask(
            id="s1",
            description="Critical",
            is_critical_path=True,
            max_retries=0,
        )
        task.plan.subtasks = [subtask]
        state_manager.create(task)
        (tmp_path / "report.md").write_text("Evidence status: complete\n", encoding="utf-8")

        verification = VerificationResult(
            tier=1,
            passed=False,
            outcome="fail",
            reason_code="incomplete_deliverable_placeholder",
            severity_class="semantic",
            metadata={
                "failure_class": "recoverable_placeholder",
                "remediation_mode": "confirm_or_prune",
                "placeholder_findings": [{
                    "rule_name": "no-placeholders",
                    "pattern": r"\bN/A\b",
                    "source": "deliverable",
                    "file_path": "report.md",
                    "line": 1,
                    "column": 18,
                    "token": "N/A",
                    "context": "Evidence status: N/A",
                }],
            },
        )

        resolved, _, _ = await orch._run_deterministic_placeholder_prepass(
            task=task,
            subtask=subtask,
            verification=verification,
            origin="confirm_or_prune_remediation",
            attempt_number=1,
            max_attempts=2,
        )

        assert resolved is False
        event_types = [event.event_type for event in events]
        assert PLACEHOLDER_CONFIRM_OR_PRUNE_STARTED in event_types
        assert PLACEHOLDER_REMEDIATION_UNRESOLVED in event_types
        assert PLACEHOLDER_PRUNED not in event_types
        unresolved_events = [
            event for event in events
            if event.event_type == PLACEHOLDER_REMEDIATION_UNRESOLVED
        ]
        assert unresolved_events
        assert unresolved_events[-1].data.get("outcome") == "no_mutations_applied"
        assert unresolved_events[-1].data.get("mode") == "deterministic_placeholder_prepass"
        assert unresolved_events[-1].data.get("parent_mode") == "confirm_or_prune_remediation"

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
            execution_surface="api",
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

    @pytest.mark.asyncio
    async def test_execute_task_reraises_cancelled_error_from_single_dispatch(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        task.plan = Plan(subtasks=[Subtask(id="s1", description="only")])

        orch._dispatch_subtask = AsyncMock(side_effect=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await orch.execute_task(task, reuse_existing_plan=True)

    @pytest.mark.asyncio
    async def test_execute_task_reraises_cancelled_error_from_parallel_gather(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=Config(execution=ExecutionConfig(max_parallel_subtasks=2)),
        )
        task = _make_task()
        task.plan = Plan(subtasks=[
            Subtask(id="s1", description="first"),
            Subtask(id="s2", description="second"),
        ])

        async def _dispatch(_task, subtask, _attempts):
            if subtask.id == "s1":
                raise asyncio.CancelledError()
            return (
                subtask,
                SubtaskResult(status="success", summary="ok"),
                VerificationResult(tier=1, passed=True),
            )

        orch._dispatch_subtask = AsyncMock(side_effect=_dispatch)

        with pytest.raises(asyncio.CancelledError):
            await orch.execute_task(task, reuse_existing_plan=True)
