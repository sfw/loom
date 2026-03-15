"""Tests for ask_user QuestionManager lifecycle and policies."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config, ExecutionConfig
from loom.engine.runner import SubtaskResultStatus, SubtaskRunner
from loom.engine.verification import VerificationResult
from loom.events.bus import EventBus
from loom.events.types import (
    ASK_USER_ANSWERED,
    ASK_USER_CANCELLED,
    ASK_USER_REQUESTED,
    ASK_USER_TIMEOUT,
)
from loom.models.base import ModelResponse, TokenUsage, ToolCall
from loom.prompts.assembler import PromptAssembler
from loom.recovery.questions import QuestionManager, QuestionRequest, QuestionStatus
from loom.state.memory import Database, MemoryManager
from loom.state.task_state import Plan, Subtask, SubtaskStatus, Task, TaskStateManager, TaskStatus
from loom.tools import create_default_registry


@pytest.fixture
async def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "questions.db")
    await db.initialize()
    await db.insert_task(task_id="t1", goal="Question manager test")
    return db


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
async def memory_manager(database: Database) -> MemoryManager:
    return MemoryManager(database)


@pytest.fixture
async def question_manager(event_bus: EventBus, memory_manager: MemoryManager) -> QuestionManager:
    return QuestionManager(
        event_bus=event_bus,
        memory_manager=memory_manager,
        poll_interval_seconds=0.01,
    )


class TestQuestionManager:
    @pytest.mark.asyncio
    async def test_request_question_roundtrip_answer(
        self,
        question_manager: QuestionManager,
        event_bus: EventBus,
    ):
        events = []
        event_bus.subscribe_all(lambda event: events.append(event))

        request = QuestionRequest.from_ask_user_args(
            {
                "question": "Choose a language",
                "options": ["Python", "Rust"],
            },
            timeout_policy="block",
            tool_call_id="call-1",
        )

        pending_wait = asyncio.create_task(
            question_manager.request_question(
                task_id="t1",
                subtask_id="s1",
                request=request,
            ),
        )

        question_id = ""
        for _ in range(100):
            pending = await question_manager.list_pending_questions("t1")
            if pending:
                question_id = str(pending[0].get("question_id", "") or "").strip()
                break
            await asyncio.sleep(0.01)
        assert question_id

        resolved = await question_manager.answer_question(
            task_id="t1",
            question_id=question_id,
            answer_payload={
                "selected_option_ids": ["python"],
                "answered_by": "tester",
                "client_id": "ui",
                "source": "tui",
            },
        )
        assert isinstance(resolved, dict)
        assert resolved["status"] == "answered"

        answer = await asyncio.wait_for(pending_wait, timeout=1.0)
        assert answer.status == QuestionStatus.ANSWERED
        assert answer.selected_option_ids == ["python"]
        assert answer.selected_labels == ["Python"]
        assert answer.answered_by == "tester"
        assert answer.client_id == "ui"

        event_types = [event.event_type for event in events]
        assert ASK_USER_REQUESTED in event_types
        assert ASK_USER_ANSWERED in event_types

    @pytest.mark.asyncio
    async def test_replayed_deterministic_request_returns_existing_answer(
        self,
        question_manager: QuestionManager,
    ):
        request = QuestionRequest.from_ask_user_args(
            {
                "question": "Choose a language",
                "options": ["Python", "Rust"],
            },
            timeout_policy="block",
            tool_call_id="call-2",
        )

        pending_wait = asyncio.create_task(
            question_manager.request_question(
                task_id="t1",
                subtask_id="s1",
                request=request,
            ),
        )

        question_id = ""
        for _ in range(100):
            pending = await question_manager.list_pending_questions("t1")
            if pending:
                question_id = str(pending[0].get("question_id", "") or "").strip()
                break
            await asyncio.sleep(0.01)
        assert question_id

        await question_manager.answer_question(
            task_id="t1",
            question_id=question_id,
            answer_payload={
                "selected_option_ids": ["rust"],
                "source": "api",
            },
        )
        first_answer = await asyncio.wait_for(pending_wait, timeout=1.0)
        assert first_answer.status == QuestionStatus.ANSWERED

        replay = await question_manager.request_question(
            task_id="t1",
            subtask_id="s1",
            request=request,
        )
        assert replay.status == QuestionStatus.ANSWERED
        assert replay.question_id == first_answer.question_id
        assert replay.selected_option_ids == ["rust"]

    @pytest.mark.asyncio
    async def test_request_without_tool_call_id_uses_deterministic_fallback(
        self,
        question_manager: QuestionManager,
    ):
        request = QuestionRequest.from_ask_user_args(
            {
                "question": "Choose deployment target",
                "question_type": "single_choice",
                "options": ["Staging", "Production"],
            },
            timeout_policy="block",
        )

        first_wait = asyncio.create_task(
            question_manager.request_question(
                task_id="t1",
                subtask_id="s-fallback",
                request=request,
            ),
        )

        question_id = ""
        for _ in range(100):
            pending = await question_manager.list_pending_questions("t1")
            if pending:
                question_id = str(pending[0].get("question_id", "") or "").strip()
                break
            await asyncio.sleep(0.01)
        assert question_id.startswith("q_")

        await question_manager.answer_question(
            task_id="t1",
            question_id=question_id,
            answer_payload={"selected_option_ids": ["staging"], "source": "tui"},
        )
        first_answer = await asyncio.wait_for(first_wait, timeout=1.0)
        assert first_answer.question_id == question_id

        replay = await question_manager.request_question(
            task_id="t1",
            subtask_id="s-fallback",
            request=request,
        )
        assert replay.question_id == question_id
        assert replay.status == QuestionStatus.ANSWERED

    @pytest.mark.asyncio
    async def test_timeout_default_policy_resolves_with_default_option(
        self,
        question_manager: QuestionManager,
        event_bus: EventBus,
    ):
        events = []
        event_bus.subscribe_all(lambda event: events.append(event))

        request = QuestionRequest.from_ask_user_args(
            {
                "question": "Pick a runtime",
                "question_type": "single_choice",
                "options": ["Python", "Rust"],
                "default_option_id": "python",
            },
            timeout_policy="timeout_default",
            timeout_seconds=1,
            tool_call_id="call-timeout-default",
        )
        answer = await asyncio.wait_for(
            question_manager.request_question(
                task_id="t1",
                subtask_id="s2",
                request=request,
            ),
            timeout=2.0,
        )
        assert answer.status == QuestionStatus.TIMEOUT
        assert answer.selected_option_ids == ["python"]
        assert answer.selected_labels == ["Python"]
        assert ASK_USER_TIMEOUT in [event.event_type for event in events]

    @pytest.mark.asyncio
    async def test_fail_closed_timeout_resolves_without_user_answer(
        self,
        question_manager: QuestionManager,
    ):
        request = QuestionRequest.from_ask_user_args(
            {
                "question": "Provide deployment details",
                "question_type": "free_text",
            },
            timeout_policy="fail_closed",
            timeout_seconds=1,
            tool_call_id="call-timeout-fail-closed",
        )
        answer = await asyncio.wait_for(
            question_manager.request_question(
                task_id="t1",
                subtask_id="s3",
                request=request,
            ),
            timeout=2.0,
        )
        assert answer.status == QuestionStatus.TIMEOUT
        assert answer.response_type == "timeout"
        assert answer.custom_response == ""

    @pytest.mark.asyncio
    async def test_cancellation_preempts_wait(
        self,
        question_manager: QuestionManager,
        event_bus: EventBus,
    ):
        events = []
        event_bus.subscribe_all(lambda event: events.append(event))

        request = QuestionRequest.from_ask_user_args(
            {"question": "Need confirmation"},
            timeout_policy="block",
            tool_call_id="call-cancel",
        )
        answer = await question_manager.request_question(
            task_id="t1",
            subtask_id="s4",
            request=request,
            check_task_control=lambda: "cancelled",
        )
        assert answer.status == QuestionStatus.CANCELLED
        assert answer.response_type == "cancelled"
        assert ASK_USER_CANCELLED in [event.event_type for event in events]

    @pytest.mark.asyncio
    async def test_pending_question_recovers_across_manager_instances(
        self,
        question_manager: QuestionManager,
        memory_manager: MemoryManager,
        event_bus: EventBus,
    ):
        request = QuestionRequest.from_ask_user_args(
            {
                "question": "Pick language",
                "question_type": "single_choice",
                "options": ["Python", "Rust"],
            },
            timeout_policy="block",
            tool_call_id="call-recover",
        )
        pending_wait = asyncio.create_task(
            question_manager.request_question(
                task_id="t1",
                subtask_id="s-recover",
                request=request,
            ),
        )
        question_id = ""
        for _ in range(100):
            pending = await question_manager.list_pending_questions("t1")
            if pending:
                question_id = str(pending[0].get("question_id", "") or "").strip()
                break
            await asyncio.sleep(0.01)
        assert question_id

        recovered_manager = QuestionManager(
            event_bus=event_bus,
            memory_manager=memory_manager,
            poll_interval_seconds=0.01,
        )
        await recovered_manager.answer_question(
            task_id="t1",
            question_id=question_id,
            answer_payload={"selected_option_ids": ["rust"], "source": "api"},
        )
        answer = await asyncio.wait_for(pending_wait, timeout=1.0)
        assert answer.status == QuestionStatus.ANSWERED
        assert answer.selected_option_ids == ["rust"]

    @pytest.mark.asyncio
    async def test_duplicate_answer_submission_returns_current_resolved_row(
        self,
        question_manager: QuestionManager,
        memory_manager: MemoryManager,
        event_bus: EventBus,
    ):
        request = QuestionRequest.from_ask_user_args(
            {
                "question": "Choose language",
                "question_type": "single_choice",
                "options": ["Python", "Rust"],
            },
            timeout_policy="block",
            tool_call_id="call-dup-answer",
        )
        pending_wait = asyncio.create_task(
            question_manager.request_question(
                task_id="t1",
                subtask_id="s5",
                request=request,
            ),
        )

        question_id = ""
        for _ in range(100):
            pending = await question_manager.list_pending_questions("t1")
            if pending:
                question_id = str(pending[0].get("question_id", "") or "").strip()
                break
            await asyncio.sleep(0.01)
        assert question_id

        first = await question_manager.answer_question(
            task_id="t1",
            question_id=question_id,
            answer_payload={"selected_option_ids": ["python"], "source": "api"},
        )
        assert isinstance(first, dict)
        assert first["status"] == "answered"

        second = await question_manager.answer_question(
            task_id="t1",
            question_id=question_id,
            answer_payload={"selected_option_ids": ["rust"], "source": "api"},
        )
        assert isinstance(second, dict)
        assert second["status"] == "answered"
        assert second["answer_payload"]["selected_option_ids"] == ["python"]

        await asyncio.wait_for(pending_wait, timeout=1.0)

        recovered_manager = QuestionManager(
            event_bus=event_bus,
            memory_manager=memory_manager,
            poll_interval_seconds=0.01,
        )
        listed = await recovered_manager.list_pending_questions("t1")
        assert listed == []


class TestSubtaskRunnerAskUser:
    @staticmethod
    def _make_task(tmp_path: Path) -> tuple[Task, Subtask]:
        subtask = Subtask(
            id="s1",
            description="Need clarification",
            status=SubtaskStatus.RUNNING,
            verification_tier=1,
        )
        task = Task(
            id="task-ask-user",
            goal="Ask user integration test",
            workspace=str(tmp_path),
            status=TaskStatus.EXECUTING,
            plan=Plan(subtasks=[subtask], version=1),
            metadata={"execution_surface": "tui"},
        )
        return task, subtask

    @pytest.mark.asyncio
    async def test_runner_blocks_then_resumes_after_answer(
        self,
        tmp_path: Path,
        database: Database,
        memory_manager: MemoryManager,
        event_bus: EventBus,
    ):
        state_manager = TaskStateManager(tmp_path / "state")
        task, subtask = self._make_task(tmp_path)
        state_manager.create(task)
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path=task.workspace,
            status=task.status.value,
        )

        config = Config(
            execution=ExecutionConfig(
                ask_user_v2_enabled=True,
                ask_user_runtime_blocking_enabled=True,
                ask_user_durable_state_enabled=True,
                ask_user_policy="block",
                ask_user_max_questions_per_subtask=3,
                ask_user_min_seconds_between_questions=0,
            ),
        )
        question_manager = QuestionManager(event_bus, memory_manager, poll_interval_seconds=0.01)

        executor_model = AsyncMock()
        executor_model.name = "mock-executor"
        executor_model.complete = AsyncMock(side_effect=[
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="ask-1",
                    name="ask_user",
                    arguments={
                        "question": "Pick stack",
                        "options": ["Python", "Rust"],
                    },
                )],
                usage=TokenUsage(total_tokens=40),
            ),
            ModelResponse(
                text="Completed after clarification.",
                tool_calls=None,
                usage=TokenUsage(total_tokens=20),
            ),
        ])
        router = MagicMock()
        router.select = MagicMock(return_value=executor_model)

        verification = MagicMock()
        verification.verify = AsyncMock(return_value=VerificationResult(tier=1, passed=True))

        runner = SubtaskRunner(
            model_router=router,
            tool_registry=create_default_registry(config),
            memory_manager=memory_manager,
            prompt_assembler=PromptAssembler(),
            state_manager=state_manager,
            verification=verification,
            config=config,
            event_bus=event_bus,
            question_manager=question_manager,
        )
        runner._spawn_memory_extraction = MagicMock()

        run_task = asyncio.create_task(runner.run(task, subtask))
        question_id = ""
        for _ in range(100):
            pending = await question_manager.list_pending_questions(task.id)
            if pending:
                question_id = str(pending[0].get("question_id", "") or "").strip()
                break
            await asyncio.sleep(0.01)
        assert question_id
        pending_task = state_manager.load(task.id)
        marker = (
            pending_task.metadata.get("awaiting_user_input")
            if isinstance(pending_task.metadata, dict)
            else {}
        )
        assert isinstance(marker, dict)
        assert marker.get("question_id") == question_id
        assert marker.get("subtask_id") == subtask.id

        await question_manager.answer_question(
            task_id=task.id,
            question_id=question_id,
            answer_payload={"selected_option_ids": ["python"], "source": "tui"},
        )

        result, verification_result = await asyncio.wait_for(run_task, timeout=2.0)
        assert result.status == SubtaskResultStatus.SUCCESS
        assert verification_result.passed is True
        assert any(call.tool == "ask_user" for call in result.tool_calls)
        ask_record = next(call for call in result.tool_calls if call.tool == "ask_user")
        assert ask_record.result.success is True
        assert ask_record.result.output == "Python"
        completed_task = state_manager.load(task.id)
        if isinstance(completed_task.metadata, dict):
            assert "awaiting_user_input" not in completed_task.metadata
            history = completed_task.metadata.get("clarification_history", [])
            assert isinstance(history, list)
            assert history
            assert "pick stack" in str(history[-1].get("question", "")).lower()
            assert "python" in str(history[-1].get("answer", "")).lower()
        assert any(
            "clarification (" in str(item).lower()
            and "pick stack" in str(item).lower()
            and "python" in str(item).lower()
            for item in completed_task.decisions_log
        )

        instructions = await memory_manager.query(task.id, entry_type="user_instruction")
        decisions = await memory_manager.query(task.id, entry_type="decision")
        assert any("Pick stack" in entry.detail for entry in instructions)
        assert any("python" in entry.detail.lower() for entry in decisions)

    @pytest.mark.asyncio
    async def test_runner_enforces_question_cap_per_subtask(
        self,
        tmp_path: Path,
        database: Database,
        memory_manager: MemoryManager,
        event_bus: EventBus,
    ):
        state_manager = TaskStateManager(tmp_path / "state")
        task, subtask = self._make_task(tmp_path)
        state_manager.create(task)
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path=task.workspace,
            status=task.status.value,
        )

        config = Config(
            execution=ExecutionConfig(
                ask_user_v2_enabled=True,
                ask_user_runtime_blocking_enabled=True,
                ask_user_durable_state_enabled=True,
                ask_user_policy="block",
                ask_user_max_questions_per_subtask=1,
                ask_user_min_seconds_between_questions=0,
            ),
        )
        question_manager = QuestionManager(event_bus, memory_manager, poll_interval_seconds=0.01)

        executor_model = AsyncMock()
        executor_model.name = "mock-executor"
        executor_model.complete = AsyncMock(side_effect=[
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="ask-1",
                    name="ask_user",
                    arguments={"question": "First?", "options": ["Yes", "No"]},
                )],
                usage=TokenUsage(total_tokens=30),
            ),
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="ask-2",
                    name="ask_user",
                    arguments={"question": "Second?", "options": ["Yes", "No"]},
                )],
                usage=TokenUsage(total_tokens=30),
            ),
            ModelResponse(
                text="Done.",
                tool_calls=None,
                usage=TokenUsage(total_tokens=10),
            ),
        ])
        router = MagicMock()
        router.select = MagicMock(return_value=executor_model)

        verification = MagicMock()
        verification.verify = AsyncMock(return_value=VerificationResult(tier=1, passed=True))

        runner = SubtaskRunner(
            model_router=router,
            tool_registry=create_default_registry(config),
            memory_manager=memory_manager,
            prompt_assembler=PromptAssembler(),
            state_manager=state_manager,
            verification=verification,
            config=config,
            event_bus=event_bus,
            question_manager=question_manager,
        )
        runner._spawn_memory_extraction = MagicMock()

        async def _answer_first_question() -> None:
            for _ in range(100):
                pending = await question_manager.list_pending_questions(task.id)
                if pending:
                    question_id = str(pending[0].get("question_id", "") or "").strip()
                    if question_id:
                        await question_manager.answer_question(
                            task_id=task.id,
                            question_id=question_id,
                            answer_payload={"selected_option_ids": ["yes"], "source": "api"},
                        )
                        return
                await asyncio.sleep(0.01)

        answer_task = asyncio.create_task(_answer_first_question())
        result, verification_result = await asyncio.wait_for(runner.run(task, subtask), timeout=3.0)
        await asyncio.wait_for(answer_task, timeout=1.0)

        assert result.status == SubtaskResultStatus.SUCCESS
        assert verification_result.passed is True
        ask_calls = [call for call in result.tool_calls if call.tool == "ask_user"]
        assert len(ask_calls) == 2
        assert ask_calls[0].result.success is True
        assert ask_calls[1].result.success is False
        assert "question cap reached" in str(ask_calls[1].result.error or "").lower()

    @pytest.mark.asyncio
    async def test_runner_blocks_ask_user_on_cli_surface(
        self,
        tmp_path: Path,
        database: Database,
        memory_manager: MemoryManager,
        event_bus: EventBus,
    ):
        state_manager = TaskStateManager(tmp_path / "state")
        task, subtask = self._make_task(tmp_path)
        task.metadata["execution_surface"] = "cli"
        state_manager.create(task)
        await database.insert_task(
            task_id=task.id,
            goal=task.goal,
            workspace_path=task.workspace,
            status=task.status.value,
        )

        config = Config(
            execution=ExecutionConfig(
                ask_user_v2_enabled=True,
                ask_user_runtime_blocking_enabled=True,
                ask_user_durable_state_enabled=True,
                ask_user_policy="block",
                ask_user_max_questions_per_subtask=3,
                ask_user_min_seconds_between_questions=0,
            ),
        )
        question_manager = QuestionManager(event_bus, memory_manager, poll_interval_seconds=0.01)

        executor_model = AsyncMock()
        executor_model.name = "mock-executor"
        executor_model.complete = AsyncMock(side_effect=[
            ModelResponse(
                text="",
                tool_calls=[ToolCall(
                    id="ask-1",
                    name="ask_user",
                    arguments={"question": "Need clarification"},
                )],
                usage=TokenUsage(total_tokens=20),
            ),
            ModelResponse(
                text="Done with assumptions.",
                tool_calls=None,
                usage=TokenUsage(total_tokens=10),
            ),
        ])
        router = MagicMock()
        router.select = MagicMock(return_value=executor_model)

        verification = MagicMock()
        verification.verify = AsyncMock(return_value=VerificationResult(tier=1, passed=True))

        runner = SubtaskRunner(
            model_router=router,
            tool_registry=create_default_registry(config),
            memory_manager=memory_manager,
            prompt_assembler=PromptAssembler(),
            state_manager=state_manager,
            verification=verification,
            config=config,
            event_bus=event_bus,
            question_manager=question_manager,
        )
        runner._spawn_memory_extraction = MagicMock()

        result, verification_result = await asyncio.wait_for(runner.run(task, subtask), timeout=2.0)
        assert result.status == SubtaskResultStatus.SUCCESS
        assert verification_result.passed is True
        assert [call for call in result.tool_calls if call.tool == "ask_user"] == []
        assert executor_model.complete.await_count == 2
        pending = await question_manager.list_pending_questions(task.id)
        assert pending == []
