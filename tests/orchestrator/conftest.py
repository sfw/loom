"""Shared orchestrator test helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from loom.config import Config
from loom.engine.orchestrator import create_task
from loom.events.bus import EventBus
from loom.models.base import ModelResponse, TokenUsage
from loom.models.router import ModelRouter
from loom.processes.schema import (
    PhaseTemplate,
    ProcessDefinition,
    VerificationRemediationContract,
)
from loom.prompts.assembler import PromptAssembler
from loom.state.task_state import Task, TaskStateManager
from loom.tools.registry import ToolRegistry, ToolResult


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
        planner_model.complete = AsyncMock(
            return_value=ModelResponse(
                text=plan_response_text,
                usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150),
            )
        )
    else:
        planner_model.complete = AsyncMock(side_effect=planner_responses)

    executor_model = AsyncMock()
    executor_model.name = "mock-executor"

    if executor_responses is None:
        executor_model.complete = AsyncMock(
            return_value=ModelResponse(
                text="Subtask completed successfully.",
                usage=TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80),
            )
        )
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
    tools.all_schemas = MagicMock(
        return_value=[
            {"name": "read_file", "description": "Read a file"},
            {"name": "write_file", "description": "Write a file"},
        ]
    )
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
