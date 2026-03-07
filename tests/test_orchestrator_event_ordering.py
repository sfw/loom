"""Lifecycle event-ordering parity tests for orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config
from loom.engine.orchestrator import Orchestrator, SubtaskResult, SubtaskResultStatus, create_task
from loom.engine.verification import VerificationResult
from loom.events.bus import EventBus
from loom.events.types import (
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLAN_READY,
    TASK_PLANNING,
)
from loom.models.base import ModelResponse, TokenUsage
from loom.models.router import ModelRouter
from loom.state.task_state import TaskStateManager
from loom.tools.registry import ToolRegistry, ToolResult


def _make_router(plan_response_text: str) -> MagicMock:
    router = MagicMock(spec=ModelRouter)

    planner_model = AsyncMock()
    planner_model.name = "mock-planner"
    planner_model.complete = AsyncMock(return_value=ModelResponse(
        text=plan_response_text,
        usage=TokenUsage(total_tokens=10),
    ))

    executor_model = AsyncMock()
    executor_model.name = "mock-executor"
    executor_model.complete = AsyncMock(return_value=ModelResponse(
        text="ok",
        usage=TokenUsage(total_tokens=8),
    ))

    def _select(*, tier: int = 1, role: str = "executor"):
        del tier
        return planner_model if role == "planner" else executor_model

    router.select = MagicMock(side_effect=_select)
    return router


def _make_tools() -> MagicMock:
    tools = MagicMock(spec=ToolRegistry)
    tools.execute = AsyncMock(return_value=ToolResult.ok("ok"))
    tools.list_tools = MagicMock(return_value=["read_file", "write_file"])
    tools.all_schemas = MagicMock(return_value=[])
    tools.has = MagicMock(return_value=False)
    tools.exclude = MagicMock()
    return tools


def _make_orchestrator(
    *,
    tmp_path: Path,
    events: list,
    plan_response_text: str,
) -> tuple[Orchestrator, TaskStateManager]:
    event_bus = EventBus()
    event_bus.subscribe_all(lambda event: events.append(event))
    state = TaskStateManager(data_dir=tmp_path)
    prompts = MagicMock()
    prompts.build_planner_prompt = MagicMock(return_value="plan")
    prompts.build_executor_prompt = MagicMock(return_value="execute")
    memory = MagicMock()
    memory.query_relevant = AsyncMock(return_value=[])
    orchestrator = Orchestrator(
        model_router=_make_router(plan_response_text=plan_response_text),
        tool_registry=_make_tools(),
        memory_manager=memory,
        prompt_assembler=prompts,
        state_manager=state,
        event_bus=event_bus,
        config=Config(),
    )
    return orchestrator, state


@pytest.mark.asyncio
async def test_execute_task_success_event_sequence(
    tmp_path: Path,
) -> None:
    events: list = []
    orch, state = _make_orchestrator(
        tmp_path=tmp_path,
        events=events,
        plan_response_text='{"subtasks":[{"id":"s1","description":"one"}]}',
    )
    orch._runner.run = AsyncMock(return_value=(
        SubtaskResult(status=SubtaskResultStatus.SUCCESS, summary="ok"),
        VerificationResult(
            tier=1,
            passed=True,
            outcome="pass",
            reason_code="",
        ),
    ))
    task = create_task(goal="Verify ordering success", workspace=str(tmp_path))
    state.create(task)

    await orch.execute_task(task)

    event_types = [event.event_type for event in events]
    planning_ix = event_types.index(TASK_PLANNING)
    ready_ix = event_types.index(TASK_PLAN_READY)
    executing_ix = event_types.index(TASK_EXECUTING)
    completed_ix = event_types.index(TASK_COMPLETED)

    assert planning_ix < ready_ix < executing_ix < completed_ix


@pytest.mark.asyncio
async def test_execute_task_failure_emits_terminal_after_execution_started(
    tmp_path: Path,
) -> None:
    events: list = []
    orch, state = _make_orchestrator(
        tmp_path=tmp_path,
        events=events,
        plan_response_text='{"subtasks":[{"id":"s1","description":"one"}]}',
    )
    orch._dispatch_subtask = AsyncMock(side_effect=RuntimeError("dispatch exploded"))
    task = create_task(goal="Verify ordering failure", workspace=str(tmp_path))
    state.create(task)

    await orch.execute_task(task)

    event_types = [event.event_type for event in events]
    executing_ix = event_types.index(TASK_EXECUTING)
    failed_ix = event_types.index(TASK_FAILED)
    assert executing_ix < failed_ix
