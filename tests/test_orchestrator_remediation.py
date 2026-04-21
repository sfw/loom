"""Focused remediation queue tests for orchestrator extraction parity."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from loom.config import Config, ExecutionConfig
from loom.engine.orchestrator import Orchestrator, create_task
from loom.events.bus import EventBus
from loom.events.types import REMEDIATION_EXPIRED, REMEDIATION_TERMINAL
from loom.models.base import ModelResponse, TokenUsage
from loom.models.router import ModelRouter
from loom.recovery.retry import RetryStrategy
from loom.state.task_state import Subtask, TaskStateManager
from loom.tools.registry import ToolRegistry, ToolResult


def _make_router(plan_response_text: str = '{"subtasks": []}') -> MagicMock:
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
) -> tuple[Orchestrator, TaskStateManager]:
    event_bus = EventBus()
    event_bus.subscribe_all(lambda event: events.append(event))
    state = TaskStateManager(data_dir=tmp_path)
    prompts = MagicMock()
    prompts.build_planner_prompt = MagicMock(return_value="plan")
    prompts.build_executor_prompt = MagicMock(return_value="execute")
    memory = MagicMock()
    memory.query_relevant = AsyncMock(return_value=[])
    orch = Orchestrator(
        model_router=_make_router(),
        tool_registry=_make_tools(),
        memory_manager=memory,
        prompt_assembler=prompts,
        state_manager=state,
        event_bus=event_bus,
        config=Config(execution=ExecutionConfig(enable_streaming=False)),
    )
    return orch, state


def test_bounded_remediation_backoff_caps_growth() -> None:
    assert Orchestrator._bounded_remediation_backoff_seconds(
        base_backoff_seconds=2.0,
        max_backoff_seconds=5.0,
        attempt_count=1,
    ) == 2.0
    assert Orchestrator._bounded_remediation_backoff_seconds(
        base_backoff_seconds=2.0,
        max_backoff_seconds=5.0,
        attempt_count=2,
    ) == 4.0
    assert Orchestrator._bounded_remediation_backoff_seconds(
        base_backoff_seconds=2.0,
        max_backoff_seconds=5.0,
        attempt_count=4,
    ) == 5.0


def test_build_remediation_retry_context_includes_reason_specific_guidance(tmp_path: Path) -> None:
    events: list = []
    orch, _ = _make_orchestrator(tmp_path=tmp_path, events=events)
    text = orch._build_remediation_retry_context(
        strategy=RetryStrategy.UNCONFIRMED_DATA,
        reason_code="missing_precedent_transactions",
    )
    assert "structured precedent transaction evidence" in text


def test_build_remediation_retry_context_prefers_reason_specific_process_instructions(
    tmp_path: Path,
) -> None:
    events: list = []
    orch, _ = _make_orchestrator(tmp_path=tmp_path, events=events)
    process = MagicMock()
    process.prompt_remediation_instructions = MagicMock(return_value="Fix precedent rows first.")
    orch._process = process

    text = orch._build_remediation_retry_context(
        strategy=RetryStrategy.UNCONFIRMED_DATA,
        reason_code="missing_precedent_transactions",
    )
    assert "Fix precedent rows first." in text
    process.prompt_remediation_instructions.assert_has_calls([
        call("missing_precedent_transactions"),
    ])


def test_augment_retry_context_for_evidence_recovery_surfaces_unfetched_urls(
    tmp_path: Path,
) -> None:
    events: list = []
    orch, _ = _make_orchestrator(tmp_path=tmp_path, events=events)
    text = orch._augment_retry_context_for_evidence_recovery(
        base_context="PREVIOUS ATTEMPTS",
        reason_code="claim_insufficient_evidence",
        prior_evidence_records=[
            {
                "tool": "web_search",
                "query": "example channel subscribers",
                "source_url": "https://example.com/channel",
            },
            {
                "tool": "web_search",
                "query": "example secondary source",
                "source_url": "https://example.com/secondary",
            },
            {
                "tool": "web_fetch",
                "source_url": "https://example.com/secondary",
            },
        ],
    )
    assert "EVIDENCE RECOVERY GUIDANCE" in text
    assert "https://example.com/channel" in text
    assert "https://example.com/secondary" not in text
    assert "Do not treat web_search snippets as final support" in text


@pytest.mark.asyncio
async def test_process_remediation_queue_marks_ttl_expired_items_terminal(
    tmp_path: Path,
) -> None:
    events: list = []
    orch, state = _make_orchestrator(tmp_path=tmp_path, events=events)
    task = create_task(goal="Test remediation expiry", workspace=str(tmp_path))
    task.plan.subtasks = [Subtask(id="s1", description="one")]
    task.metadata["remediation_queue"] = [{
        "id": "rem-expired",
        "task_id": task.id,
        "subtask_id": "s1",
        "strategy": RetryStrategy.UNCONFIRMED_DATA.value,
        "reason_code": "recommendation_unconfirmed",
        "state": "queued",
        "attempt_count": 0,
        "next_attempt_at": (datetime.now() - timedelta(seconds=10)).isoformat(),
        "ttl_at": (datetime.now() - timedelta(seconds=1)).isoformat(),
    }]
    state.create(task)

    await orch._process_remediation_queue(
        task=task,
        attempts_by_subtask={},
        finalizing=False,
    )

    queue = task.metadata.get("remediation_queue", [])
    assert isinstance(queue, list)
    assert queue[0]["state"] == "expired"
    assert queue[0]["terminal_reason"] == "ttl_expired"
    event_types = [event.event_type for event in events]
    assert REMEDIATION_EXPIRED in event_types
    assert REMEDIATION_TERMINAL in event_types


@pytest.mark.asyncio
async def test_process_remediation_queue_requeues_with_capped_backoff(
    tmp_path: Path,
) -> None:
    events: list = []
    orch, state = _make_orchestrator(tmp_path=tmp_path, events=events)
    task = create_task(goal="Test remediation backoff", workspace=str(tmp_path))
    task.plan.subtasks = [Subtask(id="s1", description="one")]
    task.metadata["remediation_queue"] = [{
        "id": "rem-retry",
        "task_id": task.id,
        "subtask_id": "s1",
        "strategy": "unsupported",  # Forces remediation execution failure.
        "reason_code": "x",
        "state": "queued",
        "attempt_count": 1,
        "max_attempts": 4,
        "base_backoff_seconds": 1.0,
        "max_backoff_seconds": 1.5,
        "next_attempt_at": (datetime.now() - timedelta(seconds=1)).isoformat(),
        "ttl_at": (datetime.now() + timedelta(hours=1)).isoformat(),
    }]
    state.create(task)

    before = datetime.now()
    await orch._process_remediation_queue(
        task=task,
        attempts_by_subtask={},
        finalizing=False,
    )
    after = datetime.now()

    queue = task.metadata.get("remediation_queue", [])
    assert isinstance(queue, list)
    assert queue[0]["state"] == "queued"
    assert int(queue[0]["attempt_count"]) == 2
    next_attempt = datetime.fromisoformat(str(queue[0]["next_attempt_at"]))
    lower_bound = before + timedelta(seconds=1.0)
    upper_bound = after + timedelta(seconds=2.5)
    assert lower_bound <= next_attempt <= upper_bound
