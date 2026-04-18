"""Focused execution-loop tests for runner session cleanup invariants."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import loom.engine.runner as runner_module
from loom.auth.runtime import AuthResolutionError
from loom.config import Config, ExecutionConfig
from loom.engine.runner import SubtaskResultStatus, SubtaskRunner
from loom.engine.verification import VerificationResult
from loom.models.base import ModelResponse, TokenUsage, ToolCall
from loom.state.task_state import Subtask, Task
from loom.tools.registry import ToolResult


def _make_task(tmp_path: Path) -> tuple[Task, Subtask]:
    task = Task(
        id="task-1",
        goal="Ship runner refactor safely.",
        workspace=str(tmp_path),
    )
    subtask = Subtask(id="subtask-1", description="Execute and verify.")
    return task, subtask


def _make_runner(
    *,
    memory_query: AsyncMock,
    model_complete: AsyncMock | None = None,
    config: Config | None = None,
) -> SubtaskRunner:
    resolved_config = config or Config()
    router = MagicMock()
    executor_model = MagicMock()
    executor_model.name = "mock-executor"
    executor_model.complete = model_complete or AsyncMock(
        return_value=ModelResponse(
            text="Completed.",
            tool_calls=None,
            usage=TokenUsage(total_tokens=12),
        ),
    )
    router.select = MagicMock(return_value=executor_model)

    tool_registry = MagicMock()
    tool_registry.all_schemas = MagicMock(return_value=[])
    tool_registry.has = MagicMock(return_value=False)

    memory_manager = MagicMock()
    memory_manager.query_relevant = memory_query

    prompt_assembler = MagicMock()
    prompt_assembler.build_executor_prompt = MagicMock(return_value="Do the work.")

    verification = MagicMock()
    verification.verify = AsyncMock(return_value=VerificationResult(tier=1, passed=True))

    runner = SubtaskRunner(
        model_router=router,
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        prompt_assembler=prompt_assembler,
        state_manager=MagicMock(),
        verification=verification,
        config=resolved_config,
    )
    runner._spawn_memory_extraction = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_run_cleans_session_state_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "build_run_auth_context",
        lambda **kwargs: {},
    )
    runner = _make_runner(memory_query=AsyncMock(return_value=[]))
    task, subtask = _make_task(tmp_path)

    result, verification = await runner.run(task, subtask)

    assert result.status == SubtaskResultStatus.SUCCESS
    assert verification.passed is True
    assert runner._subtask_deadline_monotonic is None
    assert runner._active_subtask_telemetry_counters is None


@pytest.mark.asyncio
async def test_run_cleans_session_state_on_auth_preflight_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _raise_auth_error(**kwargs):
        del kwargs
        raise AuthResolutionError("missing auth binding")

    monkeypatch.setattr(runner_module, "build_run_auth_context", _raise_auth_error)
    runner = _make_runner(memory_query=AsyncMock(return_value=[]))
    task, subtask = _make_task(tmp_path)

    result, verification = await runner.run(task, subtask)

    assert result.status == SubtaskResultStatus.FAILED
    assert verification.passed is False
    assert "Auth preflight failed" in result.summary
    assert runner._subtask_deadline_monotonic is None
    assert runner._active_subtask_telemetry_counters is None


@pytest.mark.asyncio
async def test_run_cleans_session_state_on_unhandled_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "build_run_auth_context",
        lambda **kwargs: {},
    )
    runner = _make_runner(memory_query=AsyncMock(side_effect=RuntimeError("boom")))
    task, subtask = _make_task(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        await runner.run(task, subtask)

    assert runner._subtask_deadline_monotonic is None
    assert runner._active_subtask_telemetry_counters is None


def test_runner_resolves_post_call_guard_mode_from_config() -> None:
    runner = _make_runner(
        memory_query=AsyncMock(return_value=[]),
        config=Config(
            execution=ExecutionConfig(sealed_artifact_post_call_guard="warn"),
        ),
    )
    assert runner._sealed_artifact_post_call_guard_mode() == "warn"


@pytest.mark.asyncio
async def test_run_locks_to_text_completion_after_writing_expected_deliverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "build_run_auth_context",
        lambda **kwargs: {},
    )
    model_complete = AsyncMock(side_effect=[
        ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="write_file",
                    arguments={"path": "report.md", "content": "draft"},
                ),
            ],
            usage=TokenUsage(total_tokens=18),
        ),
        ModelResponse(
            text="Completed.",
            tool_calls=None,
            usage=TokenUsage(total_tokens=8),
        ),
    ])
    runner = _make_runner(
        memory_query=AsyncMock(return_value=[]),
        model_complete=model_complete,
        config=Config(execution=ExecutionConfig(enable_streaming=False)),
    )
    runner._tools.all_schemas = MagicMock(return_value=[
        {
            "name": "write_file",
            "parameters": {"type": "object", "required": ["path", "content"]},
        },
    ])
    tool_obj = MagicMock()
    tool_obj.is_mutating = True
    tool_obj.mutation_target_arg_keys = ()
    runner._tools.get = MagicMock(return_value=tool_obj)
    runner._tools.execute = AsyncMock(
        return_value=ToolResult.ok("ok", files_changed=["report.md"]),
    )

    task, subtask = _make_task(tmp_path)

    result, verification = await runner.run(
        task,
        subtask,
        expected_deliverables=["report.md"],
    )

    assert result.status == SubtaskResultStatus.SUCCESS
    assert verification.passed is True
    assert runner._tools.execute.await_count == 1
    assert len(model_complete.await_args_list) == 2
    assert model_complete.await_args_list[0].kwargs["tools"] != []
    assert model_complete.await_args_list[1].kwargs["tools"] == []
    second_call = model_complete.await_args_list[1]
    second_messages = second_call.kwargs.get("messages", second_call.args[0])
    assert any(
        "CANONICAL DELIVERABLE WRITE COMPLETE" in str(message.get("content", ""))
        for message in second_messages
    )


@pytest.mark.asyncio
async def test_run_allows_same_response_non_mutating_tools_after_canonical_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "build_run_auth_context",
        lambda **kwargs: {},
    )
    model_complete = AsyncMock(side_effect=[
        ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="write_file",
                    arguments={"path": "report.md", "content": "draft"},
                ),
                ToolCall(
                    id="call-2",
                    name="fact_checker",
                    arguments={
                        "claims": ["draft"],
                        "sources": ["report.md"],
                    },
                ),
            ],
            usage=TokenUsage(total_tokens=22),
        ),
        ModelResponse(
            text="Completed.",
            tool_calls=None,
            usage=TokenUsage(total_tokens=8),
        ),
    ])
    runner = _make_runner(
        memory_query=AsyncMock(return_value=[]),
        model_complete=model_complete,
        config=Config(execution=ExecutionConfig(enable_streaming=False)),
    )
    runner._tools.all_schemas = MagicMock(return_value=[
        {
            "name": "write_file",
            "parameters": {"type": "object", "required": ["path", "content"]},
        },
        {
            "name": "fact_checker",
            "parameters": {"type": "object", "required": ["claims"]},
        },
    ])
    write_tool = MagicMock()
    write_tool.is_mutating = True
    write_tool.mutation_target_arg_keys = ()
    fact_checker_tool = MagicMock()
    fact_checker_tool.is_mutating = True
    fact_checker_tool.mutation_target_arg_keys = ()
    runner._tools.get = MagicMock(side_effect=lambda name: {
        "write_file": write_tool,
        "fact_checker": fact_checker_tool,
    }[name])
    runner._tools.execute = AsyncMock(side_effect=[
        ToolResult.ok("ok", files_changed=["report.md"]),
        ToolResult.ok("supported", data={"verdicts": [{"claim": "draft"}]}),
    ])

    task, subtask = _make_task(tmp_path)

    result, verification = await runner.run(
        task,
        subtask,
        expected_deliverables=["report.md"],
    )

    assert result.status == SubtaskResultStatus.SUCCESS
    assert verification.passed is True
    assert runner._tools.execute.await_count == 2
    assert runner._tools.execute.await_args_list[0].args[0] == "write_file"
    assert runner._tools.execute.await_args_list[1].args[0] == "fact_checker"
    assert len(model_complete.await_args_list) == 2
    assert model_complete.await_args_list[1].kwargs["tools"] == []


@pytest.mark.asyncio
async def test_run_blocks_same_response_mutation_after_canonical_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner_module,
        "build_run_auth_context",
        lambda **kwargs: {},
    )
    model_complete = AsyncMock(side_effect=[
        ModelResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="write_file",
                    arguments={"path": "report.md", "content": "draft"},
                ),
                ToolCall(
                    id="call-2",
                    name="write_file",
                    arguments={"path": "notes.md", "content": "extra"},
                ),
            ],
            usage=TokenUsage(total_tokens=20),
        ),
        ModelResponse(
            text="Completed.",
            tool_calls=None,
            usage=TokenUsage(total_tokens=8),
        ),
    ])
    runner = _make_runner(
        memory_query=AsyncMock(return_value=[]),
        model_complete=model_complete,
        config=Config(execution=ExecutionConfig(enable_streaming=False)),
    )
    runner._tools.all_schemas = MagicMock(return_value=[
        {
            "name": "write_file",
            "parameters": {"type": "object", "required": ["path", "content"]},
        },
    ])
    tool_obj = MagicMock()
    tool_obj.is_mutating = True
    tool_obj.mutation_target_arg_keys = ()
    runner._tools.get = MagicMock(return_value=tool_obj)
    runner._tools.execute = AsyncMock(
        return_value=ToolResult.ok("ok", files_changed=["report.md"]),
    )

    task, subtask = _make_task(tmp_path)

    result, verification = await runner.run(
        task,
        subtask,
        expected_deliverables=["report.md"],
    )

    assert result.status == SubtaskResultStatus.SUCCESS
    assert verification.passed is True
    assert runner._tools.execute.await_count == 1
    assert len(model_complete.await_args_list) == 2
    assert model_complete.await_args_list[1].kwargs["tools"] == []
