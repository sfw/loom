"""Tests for the orchestrator loop."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.config import Config
from loom.engine.orchestrator import Orchestrator, SubtaskResult, ToolCallRecord, create_task
from loom.events.bus import EventBus
from loom.events.types import (
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_FAILED,
    TASK_PLAN_READY,
    TASK_PLANNING,
)
from loom.models.base import ModelResponse, TokenUsage, ToolCall
from loom.models.router import ModelRouter
from loom.processes.schema import PhaseTemplate, ProcessDefinition
from loom.prompts.assembler import PromptAssembler
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


class TestOrchestratorExecution:
    """Tests for the subtask execution phase."""

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
