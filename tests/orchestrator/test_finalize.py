"""Orchestrator finalization tests."""

from __future__ import annotations

import asyncio
import csv
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config, ExecutionConfig
from loom.engine.orchestrator import Orchestrator, SubtaskResult
from loom.engine.verification import VerificationResult
from loom.events.types import TASK_CANCEL_REQUESTED, TASK_FAILED, TASK_PAUSED, TASK_RESUMED
from loom.models.base import ModelConnectionError, ModelResponse, TokenUsage, ToolCall
from loom.models.router import ModelRouter
from loom.state.task_state import Plan, Subtask, TaskStatus
from loom.tools.registry import ToolResult
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

        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
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
        cancel_target["task"] = task
        cancel_target["orch"] = orch

        result = await orch.execute_task(task)

        assert result.status == TaskStatus.CANCELLED
        assert TASK_CANCEL_REQUESTED in [event.event_type for event in events]

    def test_pause_and_resume_task(self, tmp_path):
        bus = _make_event_bus()
        events = []
        bus.subscribe_all(lambda event: events.append(event))
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
        task.status = TaskStatus.EXECUTING

        orch.pause_task(task)
        assert task.status == TaskStatus.PAUSED
        assert task.metadata.get("paused_from_status") == TaskStatus.EXECUTING.value
        assert TASK_PAUSED in [event.event_type for event in events]

        orch.resume_task(task)
        assert task.status == TaskStatus.EXECUTING
        assert "paused_from_status" not in task.metadata
        assert TASK_RESUMED in [event.event_type for event in events]

    def test_pause_and_resume_preserves_planning_status(self, tmp_path):
        orch = Orchestrator(
            model_router=_make_mock_router(plan_response_text='{"subtasks": []}'),
            tool_registry=_make_mock_tools(),
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )
        task = _make_task()
        task.status = TaskStatus.PLANNING

        orch.pause_task(task)
        assert task.status == TaskStatus.PAUSED
        assert task.metadata.get("paused_from_status") == TaskStatus.PLANNING.value

        orch.resume_task(task)
        assert task.status == TaskStatus.PLANNING

    @pytest.mark.asyncio
    async def test_pause_blocks_subtask_runner_until_resume(self, tmp_path):
        executor_model = AsyncMock()
        executor_model.name = "mock-exec"
        executor_model.complete = AsyncMock(side_effect=[
            ModelResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc-1",
                        name="read_file",
                        arguments={"path": "notes.md"},
                    ),
                ],
                usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
            ),
            ModelResponse(
                text="Done",
                usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
            ),
        ])
        planner_model = AsyncMock()
        planner_model.name = "mock-plan"
        planner_model.complete = AsyncMock(return_value=ModelResponse(
            text=json.dumps({
                "subtasks": [{"id": "s1", "description": "Only step"}],
            }),
            usage=TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
        ))

        router = MagicMock(spec=ModelRouter)

        def select_fn(tier=1, role="executor"):
            if role == "planner":
                return planner_model
            return executor_model

        router.select = MagicMock(side_effect=select_fn)

        tools = _make_mock_tools()
        first_tool_done = asyncio.Event()
        release_first_tool = asyncio.Event()

        async def _execute_tool(name, args, **kwargs):
            first_tool_done.set()
            await release_first_tool.wait()
            return ToolResult.ok("ok")

        tools.execute = AsyncMock(side_effect=_execute_tool)

        orch = Orchestrator(
            model_router=router,
            tool_registry=tools,
            memory_manager=_make_mock_memory(),
            prompt_assembler=_make_mock_prompts(),
            state_manager=_make_state_manager(tmp_path),
            event_bus=_make_event_bus(),
            config=_make_config(),
        )

        task = _make_task()
        task.plan = Plan(subtasks=[Subtask(id="s1", description="Only step")])
        task.status = TaskStatus.EXECUTING

        run_task = asyncio.create_task(
            orch.execute_task(task, reuse_existing_plan=True),
        )
        await asyncio.wait_for(first_tool_done.wait(), timeout=2.0)

        orch.pause_task(task)
        assert task.status == TaskStatus.PAUSED
        release_first_tool.set()

        await asyncio.sleep(0.3)
        assert executor_model.complete.await_count == 1
        assert run_task.done() is False

        orch.resume_task(task)
        result = await asyncio.wait_for(run_task, timeout=2.0)
        assert result.status == TaskStatus.COMPLETED
        assert executor_model.complete.await_count >= 2

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
