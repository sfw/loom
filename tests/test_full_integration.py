"""Full end-to-end integration test for the Loom orchestration engine.

Exercises the entire spec: task creation → planning → subtask execution
→ tool calls → verification gates → confidence scoring → approval check
→ memory extraction → event emission → learning → completion.

Uses a deterministic FakeModelProvider so no real LLM is needed.
Everything else is real: Database, TaskStateManager, EventBus, EventPersister,
ToolRegistry (with real file tools), VerificationGates, ConfidenceScorer,
ApprovalManager, RetryManager, LearningManager, PromptAssembler, Orchestrator.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from loom.config import Config, ExecutionConfig, MemoryConfig, VerificationConfig
from loom.engine.orchestrator import Orchestrator, create_task
from loom.events.bus import Event, EventBus, EventPersister
from loom.events.types import (
    SUBTASK_COMPLETED,
    SUBTASK_FAILED,
    SUBTASK_RETRYING,
    SUBTASK_STARTED,
    TASK_COMPLETED,
    TASK_EXECUTING,
    TASK_PLAN_READY,
    TASK_PLANNING,
)
from loom.learning.manager import LearningManager
from loom.models.base import ModelProvider, ModelResponse, TokenUsage, ToolCall
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
from loom.recovery.approval import ApprovalManager
from loom.state.memory import Database, MemoryManager
from loom.state.task_state import SubtaskStatus, TaskStateManager, TaskStatus
from loom.tools import create_default_registry

# ---------------------------------------------------------------------------
# Deterministic fake model provider
# ---------------------------------------------------------------------------

class FakeModelProvider(ModelProvider):
    """Deterministic model provider for integration testing.

    Accepts a list of response factories keyed by role.  Each call to
    ``complete()`` pops the next canned response for the role that
    constructed this provider.  This lets us control exactly what the
    planner, executor, verifier, and extractor return without any LLM.
    """

    def __init__(
        self,
        name: str,
        tier: int,
        roles: list[str],
        responses: list[ModelResponse] | None = None,
    ):
        self._name = name
        self._tier = tier
        self._roles = roles
        self._responses: list[ModelResponse] = list(responses or [])
        self._call_count = 0
        self.call_log: list[dict] = []

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        self.call_log.append({
            "messages_count": len(messages),
            "has_tools": tools is not None,
        })
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            # Default: text-only "Done." so the orchestrator stops looping.
            resp = ModelResponse(
                text="Done.",
                usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        self._call_count += 1
        return resp

    async def health_check(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> int:
        return self._tier

    @property
    def roles(self) -> list[str]:
        return self._roles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=50, output_tokens=30, total_tokens=80)


def _plan_response(*subtasks: dict) -> ModelResponse:
    """Create a planner response with the given subtask dicts."""
    return ModelResponse(
        text=json.dumps({"subtasks": list(subtasks)}),
        usage=_usage(),
    )


def _tool_call_response(tool_name: str, args: dict, text: str = "") -> ModelResponse:
    """Create a response that invokes a single tool."""
    return ModelResponse(
        text=text,
        tool_calls=[ToolCall(id="tc1", name=tool_name, arguments=args)],
        usage=_usage(),
    )


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(text=text, usage=_usage())


def _verification_response(passed: bool, confidence: float = 0.9) -> ModelResponse:
    return ModelResponse(
        text=json.dumps({
            "passed": passed,
            "confidence": confidence,
            "issues": [] if passed else ["test failure"],
            "suggestion": None if passed else "Fix the failing test",
        }),
        usage=_usage(),
    )


def _extractor_response() -> ModelResponse:
    return ModelResponse(
        text=json.dumps({
            "entries": [
                {
                    "type": "discovery",
                    "summary": "Created hello.py with a greeting function",
                    "detail": "Wrote a Python file containing a hello() function.",
                    "tags": "python,file-creation",
                },
            ],
        }),
        usage=_usage(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(str(tmp_path / "integration.db"))
    await database.initialize()
    return database


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def event_collector() -> dict:
    """Returns a dict with a list of captured events and a handler function."""
    captured: list[Event] = []

    def handler(event: Event) -> None:
        captured.append(event)

    return {"events": captured, "handler": handler}


# ---------------------------------------------------------------------------
# Test: Happy-path full lifecycle
# ---------------------------------------------------------------------------

class TestFullLifecycleHappyPath:
    """Submit a task → plan → execute subtask with real tool → verify → complete.

    Covers specs: 02 (orchestrator loop), 03 (state + memory), 04 (router),
    05 (tools), 06 (verification), 08 (events), 12 (prompts), 13 (retry),
    14 (approval/confidence), 15 (learning).
    """

    @pytest.mark.asyncio
    async def test_single_subtask_writes_file_and_completes(
        self, tmp_path: Path, db: Database, workspace: Path, event_collector: dict,
    ):
        # -- Arrange ----------------------------------------------------------

        # Planner returns a single subtask
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Create hello.py"},
                ),
            ],
        )

        # Executor: 1st call → write_file tool call, 2nd call → text "Done."
        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[
                _tool_call_response("write_file", {
                    "path": "hello.py",
                    "content": "def hello():\n    return 'Hello, world!'\n",
                }),
                _text_response("Created hello.py with a greeting function."),
            ],
        )

        # Verifier (tier-2 LLM verification)
        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[_verification_response(passed=True, confidence=0.95)],
        )

        # Extractor (memory extraction)
        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response()],
        )

        router = ModelRouter(providers={
            "planner": planner,
            "executor": executor,
            "verifier": verifier,
            "extractor": extractor,
        })

        tools = create_default_registry()
        memory = MemoryManager(db)
        prompts = PromptAssembler()
        state_mgr = TaskStateManager(data_dir=tmp_path / "state")
        event_bus = EventBus()

        # Wire event persistence
        persister = EventPersister(db)
        persister.attach(event_bus)

        # Wire event collector for assertions
        event_bus.subscribe_all(event_collector["handler"])

        approval = ApprovalManager(event_bus)
        learning = LearningManager(db)

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(
                max_subtask_retries=2,
                max_loop_iterations=10,
            ),
            verification=VerificationConfig(
                tier1_enabled=True,
                tier2_enabled=True,
                tier3_enabled=False,
            ),
        )

        orch = Orchestrator(
            model_router=router,
            tool_registry=tools,
            memory_manager=memory,
            prompt_assembler=prompts,
            state_manager=state_mgr,
            event_bus=event_bus,
            config=config,
            approval_manager=approval,
            learning_manager=learning,
        )

        # -- Act --------------------------------------------------------------
        task = create_task(
            goal="Create a hello.py file with a greeting function",
            workspace=str(workspace),
            approval_mode="auto",
        )
        result = await orch.execute_task(task)

        # Give async event handlers and fire-and-forget memory extraction time to flush
        await asyncio.sleep(0.4)

        # -- Assert: Task completed -------------------------------------------
        assert result.status == TaskStatus.COMPLETED
        assert result.completed_at != ""

        # -- Assert: Plan was created with 1 subtask --------------------------
        assert len(result.plan.subtasks) == 1
        assert result.plan.subtasks[0].id == "s1"
        assert result.plan.subtasks[0].status == SubtaskStatus.COMPLETED

        # -- Assert: File was actually written (real tool execution) -----------
        hello_file = workspace / "hello.py"
        assert hello_file.exists()
        content = hello_file.read_text()
        assert "def hello():" in content
        assert "'Hello, world!'" in content

        # -- Assert: Workspace changes tracked --------------------------------
        assert result.workspace_changes.files_created >= 1

        # -- Assert: State persisted to disk ----------------------------------
        loaded = state_mgr.load(result.id)
        assert loaded.status == TaskStatus.COMPLETED
        assert loaded.goal == result.goal

        # -- Assert: Events emitted in correct order --------------------------
        events = event_collector["events"]
        event_types = [e.event_type for e in events]

        assert TASK_PLANNING in event_types
        assert TASK_PLAN_READY in event_types
        assert TASK_EXECUTING in event_types
        assert SUBTASK_STARTED in event_types
        assert SUBTASK_COMPLETED in event_types
        assert TASK_COMPLETED in event_types

        # Lifecycle order: planning → plan_ready → executing → subtask → task
        planning_idx = event_types.index(TASK_PLANNING)
        plan_ready_idx = event_types.index(TASK_PLAN_READY)
        executing_idx = event_types.index(TASK_EXECUTING)
        subtask_start_idx = event_types.index(SUBTASK_STARTED)
        subtask_done_idx = event_types.index(SUBTASK_COMPLETED)
        task_done_idx = event_types.index(TASK_COMPLETED)

        assert planning_idx < plan_ready_idx < executing_idx
        assert executing_idx < subtask_start_idx < subtask_done_idx < task_done_idx

        # -- Assert: Events persisted to database -----------------------------
        db_events = await db.query_events(result.id)
        db_event_types = {e["event_type"] for e in db_events}
        assert TASK_COMPLETED in db_event_types
        assert SUBTASK_COMPLETED in db_event_types

        # -- Assert: Memory extraction stored entries -------------------------
        mem_entries = await db.query_memory(result.id)
        assert len(mem_entries) >= 1
        assert any(e.entry_type == "discovery" for e in mem_entries)

        # -- Assert: Learning patterns extracted ------------------------------
        patterns = await learning.query_patterns(pattern_type="subtask_success")
        assert len(patterns) >= 1

        task_templates = await learning.query_patterns(pattern_type="task_template")
        assert len(task_templates) >= 1

        # -- Assert: Models were called the right number of times -------------
        assert len(planner.call_log) == 1     # 1 planning call
        assert len(executor.call_log) == 2    # tool call + text response
        assert len(extractor.call_log) >= 1   # memory extraction


# ---------------------------------------------------------------------------
# Test: Multi-subtask with dependencies
# ---------------------------------------------------------------------------

class TestMultiSubtaskWithDependencies:
    """Two subtasks where s2 depends on s1."""

    @pytest.mark.asyncio
    async def test_dependency_chain_executes_in_order(
        self, tmp_path: Path, db: Database, workspace: Path, event_collector: dict,
    ):
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Create config.json"},
                    {"id": "s2", "description": "Create reader.py that uses config.json",
                     "depends_on": ["s1"]},
                ),
            ],
        )

        # Executor gets 4 calls: tool+text for s1, tool+text for s2
        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[
                # s1: write config.json
                _tool_call_response("write_file", {
                    "path": "config.json",
                    "content": '{"key": "value"}',
                }),
                _text_response("Created config.json."),
                # s2: write reader.py
                _tool_call_response("write_file", {
                    "path": "reader.py",
                    "content": (
                        "import json\n"
                        "with open('config.json') as f:\n"
                        "    data = json.load(f)\n"
                    ),
                }),
                _text_response("Created reader.py that reads config.json."),
            ],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[
                _verification_response(passed=True),
                _verification_response(passed=True),
            ],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response(), _extractor_response()],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(max_subtask_retries=2, max_loop_iterations=20),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=True),
        )

        event_bus = EventBus()
        event_bus.subscribe_all(event_collector["handler"])
        persister = EventPersister(db)
        persister.attach(event_bus)

        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
            approval_manager=ApprovalManager(event_bus),
            learning_manager=LearningManager(db),
        )

        task = create_task(
            goal="Create config and reader",
            workspace=str(workspace),
        )
        result = await orch.execute_task(task)
        await asyncio.sleep(0.2)

        # Both subtasks completed
        assert result.status == TaskStatus.COMPLETED
        assert all(s.status == SubtaskStatus.COMPLETED for s in result.plan.subtasks)

        # Both files exist
        assert (workspace / "config.json").exists()
        assert (workspace / "reader.py").exists()

        # Verify config.json is valid JSON
        config_data = json.loads((workspace / "config.json").read_text())
        assert config_data == {"key": "value"}

        # s1 was started before s2
        events = event_collector["events"]
        started = [e for e in events if e.event_type == SUBTASK_STARTED]
        assert len(started) == 2
        assert started[0].data["subtask_id"] == "s1"
        assert started[1].data["subtask_id"] == "s2"


# ---------------------------------------------------------------------------
# Test: Verification failure → retry with escalation
# ---------------------------------------------------------------------------

class TestVerificationFailureRetry:
    """Subtask fails tier-1 verification → retried → succeeds on retry."""

    @pytest.mark.asyncio
    async def test_retry_after_verification_failure(
        self, tmp_path: Path, db: Database, workspace: Path, event_collector: dict,
    ):
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Create valid Python file"},
                ),
            ],
        )

        # Executor attempt 1: writes invalid Python (syntax error)
        # Executor attempt 2 (retry): writes valid Python
        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[
                # Attempt 1: invalid Python
                _tool_call_response("write_file", {
                    "path": "app.py",
                    "content": "def broken(\n",  # SyntaxError
                }),
                _text_response("Created app.py."),
                # Attempt 2 (retry): valid Python
                _tool_call_response("write_file", {
                    "path": "app.py",
                    "content": "def hello():\n    return 42\n",
                }),
                _text_response("Fixed app.py with correct syntax."),
            ],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[
                # After retry, tier-2 verification passes
                _verification_response(passed=True),
            ],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response(), _extractor_response()],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(max_subtask_retries=2, max_loop_iterations=20),
            verification=VerificationConfig(
                tier1_enabled=True,
                tier2_enabled=True,
            ),
        )

        event_bus = EventBus()
        event_bus.subscribe_all(event_collector["handler"])

        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
            approval_manager=ApprovalManager(event_bus),
            learning_manager=LearningManager(db),
        )

        task = create_task(
            goal="Create a valid Python file",
            workspace=str(workspace),
        )
        result = await orch.execute_task(task)
        await asyncio.sleep(0.2)

        # Task ultimately completes
        assert result.status == TaskStatus.COMPLETED

        # File ends up with valid content
        content = (workspace / "app.py").read_text()
        assert "def hello():" in content

        # Events show the retry happened
        events = event_collector["events"]
        event_types = [e.event_type for e in events]

        assert SUBTASK_FAILED in event_types
        assert SUBTASK_RETRYING in event_types
        assert SUBTASK_COMPLETED in event_types

        # Retry event happened before final completion
        retry_idx = event_types.index(SUBTASK_RETRYING)
        complete_idx = event_types.index(SUBTASK_COMPLETED)
        assert retry_idx < complete_idx

        # Error was recorded on the task
        assert len(result.errors_encountered) >= 1

        # Learning captured the retry pattern
        patterns = await LearningManager(db).query_patterns(
            pattern_type="retry_pattern",
        )
        assert len(patterns) >= 1


# ---------------------------------------------------------------------------
# Test: All retries exhausted → task fails
# ---------------------------------------------------------------------------

class TestAllRetriesExhausted:
    """Subtask keeps failing → all retries exhausted → task fails."""

    @pytest.mark.asyncio
    async def test_task_fails_when_retries_exhausted(
        self, tmp_path: Path, db: Database, workspace: Path,
    ):
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Create valid file"},
                ),
                # Re-plan response (will also fail)
                _plan_response(
                    {"id": "s1-v2", "description": "Try again differently"},
                ),
            ],
        )

        # Every executor attempt writes broken Python
        bad_responses = []
        for _ in range(10):
            bad_responses.append(_tool_call_response("write_file", {
                "path": "broken.py",
                "content": "def broken(\n",
            }))
            bad_responses.append(_text_response("Created broken.py."))

        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=bad_responses,
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response() for _ in range(10)],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[_verification_response(passed=True) for _ in range(10)],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        # max_loop_iterations=4: 2 attempts for s1 (attempt+retry), replan,
        # 2 attempts for s1-v2 (attempt+retry), replan, then loop exits.
        # The executor never runs out of bad responses before the loop ends.
        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(max_subtask_retries=1, max_loop_iterations=4),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=False),
        )

        event_bus = EventBus()
        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
            approval_manager=ApprovalManager(event_bus),
            learning_manager=LearningManager(db),
        )

        task = create_task(
            goal="Create a valid file (will fail)",
            workspace=str(workspace),
        )
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.FAILED
        assert len(result.errors_encountered) >= 1

        # Learning still captured patterns from the failure
        patterns = await LearningManager(db).query_patterns(
            pattern_type="model_failure",
        )
        assert len(patterns) >= 1


# ---------------------------------------------------------------------------
# Test: No tool calls (text-only subtask)
# ---------------------------------------------------------------------------

class TestTextOnlySubtask:
    """Subtask where the model responds with text only, no tool calls."""

    @pytest.mark.asyncio
    async def test_text_only_subtask_completes(
        self, tmp_path: Path, db: Database,
    ):
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Explain the architecture"},
                ),
            ],
        )

        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[
                _text_response(
                    "The architecture uses a layered approach with "
                    "orchestrator, tools, and verification."
                ),
            ],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[_verification_response(passed=True)],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response()],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(max_subtask_retries=1, max_loop_iterations=10),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=True),
        )

        event_bus = EventBus()
        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
            approval_manager=ApprovalManager(event_bus),
            learning_manager=LearningManager(db),
        )

        task = create_task(goal="Explain the architecture")
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert result.plan.subtasks[0].summary != ""


# ---------------------------------------------------------------------------
# Test: Invalid tool name → model retries with valid response
# ---------------------------------------------------------------------------

class TestInvalidToolHandling:
    """Model hallucinates a tool name; orchestrator injects error and retries."""

    @pytest.mark.asyncio
    async def test_hallucinated_tool_triggers_error_injection(
        self, tmp_path: Path, db: Database,
    ):
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Do something"},
                ),
            ],
        )

        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[
                # Call 1: hallucinated tool name
                _tool_call_response("nonexistent_tool", {"x": 1}),
                # Call 2: text-only fallback after error injection
                _text_response("Completed without tools."),
            ],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[_verification_response(passed=True)],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response()],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(max_subtask_retries=1, max_loop_iterations=10),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=True),
        )

        event_bus = EventBus()
        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
        )

        task = create_task(goal="Do something")
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        # Executor was called twice: once with invalid tool, once with text
        assert len(executor.call_log) == 2


# ---------------------------------------------------------------------------
# Test: Edit file tool with changelog tracking
# ---------------------------------------------------------------------------

class TestEditFileWithChangelog:
    """Write a file, then edit it — verify changelog tracks both operations."""

    @pytest.mark.asyncio
    async def test_write_then_edit_tracked_in_changelog(
        self, tmp_path: Path, db: Database, workspace: Path,
    ):
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Create and modify app.py"},
                ),
            ],
        )

        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[
                # Write initial file
                _tool_call_response("write_file", {
                    "path": "app.py",
                    "content": "x = 1\ny = 2\n",
                }),
                # Edit file
                _tool_call_response("edit_file", {
                    "path": "app.py",
                    "old_str": "x = 1",
                    "new_str": "x = 42",
                }),
                _text_response("Created and modified app.py."),
            ],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[_verification_response(passed=True)],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response()],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(max_subtask_retries=1, max_loop_iterations=10),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=True),
        )

        event_bus = EventBus()
        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
        )

        task = create_task(
            goal="Create and modify app.py",
            workspace=str(workspace),
        )
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED

        # File has the edited content
        content = (workspace / "app.py").read_text()
        assert "x = 42" in content
        assert "y = 2" in content

        # Workspace changes tracked
        changes = result.workspace_changes
        assert changes.files_created >= 1 or changes.files_modified >= 1


# ---------------------------------------------------------------------------
# Test: Event persistence roundtrip
# ---------------------------------------------------------------------------

class TestEventPersistenceRoundtrip:
    """Events emitted during orchestration are persisted to SQLite and queryable."""

    @pytest.mark.asyncio
    async def test_all_lifecycle_events_persisted(
        self, tmp_path: Path, db: Database,
    ):
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response({"id": "s1", "description": "Think"}),
            ],
        )

        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[_text_response("Done thinking.")],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[_verification_response(passed=True)],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response()],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(max_subtask_retries=1, max_loop_iterations=10),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=True),
        )

        event_bus = EventBus()
        persister = EventPersister(db)
        persister.attach(event_bus)

        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
        )

        task = create_task(goal="Think about things")
        result = await orch.execute_task(task)
        await asyncio.sleep(0.3)

        assert result.status == TaskStatus.COMPLETED

        # Query persisted events from the database
        rows = await db.query_events(result.id)
        persisted_types = {r["event_type"] for r in rows}

        # All lifecycle events should be there
        assert TASK_PLANNING in persisted_types
        assert TASK_PLAN_READY in persisted_types
        assert TASK_EXECUTING in persisted_types
        assert SUBTASK_STARTED in persisted_types
        assert SUBTASK_COMPLETED in persisted_types
        assert TASK_COMPLETED in persisted_types


# ---------------------------------------------------------------------------
# Test: Memory extraction and retrieval
# ---------------------------------------------------------------------------

class TestMemoryRoundtrip:
    """Memory entries extracted during execution are stored and queryable."""

    @pytest.mark.asyncio
    async def test_memory_entries_stored_and_queryable(
        self, tmp_path: Path, db: Database,
    ):
        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response({"id": "s1", "description": "Discover something"}),
            ],
        )

        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[_text_response("Found an important pattern.")],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[_verification_response(passed=True)],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[
                ModelResponse(
                    text=json.dumps({
                        "entries": [
                            {
                                "type": "discovery",
                                "summary": "Found auth uses bcrypt for hashing",
                                "detail": "The auth module hashes passwords with bcrypt.",
                                "tags": "auth,security",
                            },
                            {
                                "type": "decision",
                                "summary": "Chose to keep bcrypt for compatibility",
                                "detail": "Switching to argon2 would break existing hashes.",
                                "tags": "auth,decision",
                            },
                        ],
                    }),
                    usage=_usage(),
                ),
            ],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(max_subtask_retries=1, max_loop_iterations=10),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=True),
        )

        event_bus = EventBus()
        memory = MemoryManager(db)

        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=memory,
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
        )

        task = create_task(goal="Discover something")
        result = await orch.execute_task(task)

        # Memory extraction is fire-and-forget — give it time to flush
        await asyncio.sleep(0.3)

        assert result.status == TaskStatus.COMPLETED

        # Query memory entries back
        entries = await memory.query(result.id, entry_type="discovery")
        assert len(entries) >= 1
        assert any("bcrypt" in e.summary for e in entries)

        decisions = await memory.query(result.id, entry_type="decision")
        assert len(decisions) >= 1

        # Search works
        search_results = await memory.search(result.id, "auth")
        assert len(search_results) >= 1

        # Relevant query for the subtask
        relevant = await memory.query_relevant(result.id, "s1")
        assert len(relevant) >= 1


# ---------------------------------------------------------------------------
# Test: Parallel subtask execution
# ---------------------------------------------------------------------------

class TestParallelSubtaskExecution:
    """Independent subtasks run concurrently when max_parallel_subtasks > 1."""

    @pytest.mark.asyncio
    async def test_independent_subtasks_run_in_parallel(
        self, tmp_path: Path, db: Database, workspace: Path, event_collector: dict,
    ):
        """Three independent subtasks (no depends_on) should all be dispatched
        in a single batch and complete without sequential ordering."""

        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Create file_a.txt"},
                    {"id": "s2", "description": "Create file_b.txt"},
                    {"id": "s3", "description": "Create file_c.txt"},
                ),
            ],
        )

        # Each subtask writes one file then responds with text
        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[
                _tool_call_response("write_file", {
                    "path": "file_a.txt", "content": "aaa",
                }),
                _text_response("Created file_a.txt."),
                _tool_call_response("write_file", {
                    "path": "file_b.txt", "content": "bbb",
                }),
                _text_response("Created file_b.txt."),
                _tool_call_response("write_file", {
                    "path": "file_c.txt", "content": "ccc",
                }),
                _text_response("Created file_c.txt."),
            ],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[
                _verification_response(passed=True),
                _verification_response(passed=True),
                _verification_response(passed=True),
            ],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response() for _ in range(3)],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(
                max_subtask_retries=1,
                max_loop_iterations=10,
                max_parallel_subtasks=3,  # Allow all 3 to run concurrently
            ),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=True),
        )

        event_bus = EventBus()
        event_bus.subscribe_all(event_collector["handler"])

        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
        )

        task = create_task(
            goal="Create three independent files",
            workspace=str(workspace),
        )
        result = await orch.execute_task(task)
        await asyncio.sleep(0.3)

        # All subtasks completed
        assert result.status == TaskStatus.COMPLETED
        assert all(s.status == SubtaskStatus.COMPLETED for s in result.plan.subtasks)

        # All files written
        assert (workspace / "file_a.txt").exists()
        assert (workspace / "file_b.txt").exists()
        assert (workspace / "file_c.txt").exists()

        # All three SUBTASK_STARTED events fired
        events = event_collector["events"]
        started = [e for e in events if e.event_type == SUBTASK_STARTED]
        assert len(started) == 3

        # All three SUBTASK_COMPLETED events fired
        completed = [e for e in events if e.event_type == SUBTASK_COMPLETED]
        assert len(completed) == 3

    @pytest.mark.asyncio
    async def test_max_parallel_subtasks_caps_batch_size(
        self, tmp_path: Path, db: Database, workspace: Path, event_collector: dict,
    ):
        """With max_parallel_subtasks=1, independent subtasks still run sequentially."""

        planner = FakeModelProvider(
            name="fake-planner", tier=2,
            roles=["planner"],
            responses=[
                _plan_response(
                    {"id": "s1", "description": "Create x.txt"},
                    {"id": "s2", "description": "Create y.txt"},
                ),
            ],
        )

        executor = FakeModelProvider(
            name="fake-executor", tier=1,
            roles=["executor"],
            responses=[
                _tool_call_response("write_file", {"path": "x.txt", "content": "x"}),
                _text_response("Done s1."),
                _tool_call_response("write_file", {"path": "y.txt", "content": "y"}),
                _text_response("Done s2."),
            ],
        )

        verifier = FakeModelProvider(
            name="fake-verifier", tier=1,
            roles=["verifier"],
            responses=[
                _verification_response(passed=True),
                _verification_response(passed=True),
            ],
        )

        extractor = FakeModelProvider(
            name="fake-extractor", tier=1,
            roles=["extractor"],
            responses=[_extractor_response(), _extractor_response()],
        )

        router = ModelRouter(providers={
            "planner": planner, "executor": executor,
            "verifier": verifier, "extractor": extractor,
        })

        config = Config(
            memory=MemoryConfig(database_path=str(tmp_path / "integration.db")),
            execution=ExecutionConfig(
                max_subtask_retries=1,
                max_loop_iterations=10,
                max_parallel_subtasks=1,  # Force sequential
            ),
            verification=VerificationConfig(tier1_enabled=True, tier2_enabled=True),
        )

        event_bus = EventBus()
        event_bus.subscribe_all(event_collector["handler"])

        orch = Orchestrator(
            model_router=router,
            tool_registry=create_default_registry(),
            memory_manager=MemoryManager(db),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_path / "state"),
            event_bus=event_bus,
            config=config,
        )

        task = create_task(
            goal="Create two files sequentially",
            workspace=str(workspace),
        )
        result = await orch.execute_task(task)

        assert result.status == TaskStatus.COMPLETED
        assert (workspace / "x.txt").exists()
        assert (workspace / "y.txt").exists()

        # With max_parallel=1, s1 should complete before s2 starts
        events = event_collector["events"]
        # s1 completed before s2 started
        s1_complete_idx = next(
            i for i, e in enumerate(events)
            if e.event_type == SUBTASK_COMPLETED and e.data["subtask_id"] == "s1"
        )
        s2_start_idx = next(
            i for i, e in enumerate(events)
            if e.event_type == SUBTASK_STARTED and e.data["subtask_id"] == "s2"
        )
        assert s1_complete_idx < s2_start_idx
