"""Process-level testing helpers for deterministic and live validation."""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from loom.config import Config, ExecutionConfig, MemoryConfig, VerificationConfig
from loom.engine.orchestrator import Orchestrator, create_task
from loom.events.bus import EventBus
from loom.models.base import ModelProvider, ModelResponse, TokenUsage, ToolCall
from loom.models.router import ModelRouter
from loom.prompts.assembler import PromptAssembler
from loom.state.memory import Database, MemoryManager
from loom.state.task_state import SubtaskStatus, TaskStateManager, TaskStatus
from loom.tools import create_default_registry

if TYPE_CHECKING:
    from loom.processes.schema import (
        ProcessDefinition,
        ProcessTestAcceptance,
        ProcessTestCase,
    )


@dataclass
class ProcessCaseResult:
    """Outcome of running one process test case."""

    case_id: str
    mode: str
    passed: bool
    duration_seconds: float
    message: str = ""
    details: list[str] = field(default_factory=list)
    task_status: str = ""
    event_log_path: str = ""


class _ScriptedModelProvider(ModelProvider):
    """Deterministic scripted model provider for process contract tests."""

    def __init__(
        self,
        *,
        name: str,
        tier: int,
        roles: list[str],
        responses: list[ModelResponse],
    ):
        self._name = name
        self._tier = tier
        self._roles = list(roles)
        self._responses = list(responses)
        self._index = 0

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        if self._index < len(self._responses):
            response = self._responses[self._index]
            self._index += 1
            return response
        return ModelResponse(
            text="Done.",
            usage=TokenUsage(input_tokens=8, output_tokens=4, total_tokens=12),
        )

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
        return list(self._roles)


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=24, output_tokens=16, total_tokens=40)


def _flatten_deliverables(process: ProcessDefinition) -> list[str]:
    deliverables = process.get_deliverables()
    flattened: list[str] = []
    for phase_files in deliverables.values():
        flattened.extend(phase_files)
    return flattened


def _default_acceptance(
    process: ProcessDefinition,
) -> ProcessTestAcceptance:
    from loom.processes.schema import ProcessTestAcceptance

    return ProcessTestAcceptance(
        phases_must_include=[phase.id for phase in process.phases],
        deliverables_must_exist=_flatten_deliverables(process),
        verification_forbidden_patterns=[],
    )


def _default_case(process: ProcessDefinition) -> ProcessTestCase:
    from loom.processes.schema import ProcessTestCase

    return ProcessTestCase(
        id="smoke",
        mode="deterministic",
        goal=f"Run the {process.name} workflow and produce required deliverables.",
        timeout_seconds=900,
        requires_network=False,
        requires_tools=[],
        acceptance=_default_acceptance(process),
    )


def select_process_test_cases(
    process: ProcessDefinition,
    *,
    include_live: bool = False,
    case_id: str | None = None,
) -> list[ProcessTestCase]:
    """Select process test cases for execution."""
    cases = list(process.tests) if process.tests else [_default_case(process)]

    if case_id:
        selected = [case for case in cases if case.id == case_id]
        if not selected:
            raise ValueError(
                f"Process test case not found: {case_id!r}. "
                f"Available: {', '.join(sorted(c.id for c in cases))}"
            )
        cases = selected

    if not include_live:
        cases = [case for case in cases if case.mode != "live"]

    return cases


def _planner_response(process: ProcessDefinition) -> ModelResponse:
    if process.phases:
        subtasks = [
            {
                "id": phase.id,
                "description": phase.description,
                "depends_on": list(phase.depends_on),
                "model_tier": phase.model_tier,
                "verification_tier": phase.verification_tier,
                "is_critical_path": phase.is_critical_path,
                "is_synthesis": phase.is_synthesis,
                "acceptance_criteria": phase.acceptance_criteria,
            }
            for phase in process.phases
        ]
    else:
        subtasks = [{
            "id": "execute-goal",
            "description": "Execute the task goal directly",
            "depends_on": [],
        }]

    return ModelResponse(
        text=json.dumps({"subtasks": subtasks}),
        usage=_usage(),
    )


def _content_for_deliverable(path: str, phase_id: str) -> str:
    if path.endswith(".csv"):
        return "metric,value\nsample,1\n"
    if path.endswith(".json"):
        return json.dumps({"phase": phase_id, "status": "ok"}, indent=2) + "\n"
    return (
        f"# {phase_id}\n\n"
        "Generated by Loom process contract testing.\n"
    )


def _executor_responses(process: ProcessDefinition) -> list[ModelResponse]:
    responses: list[ModelResponse] = []
    deliverables = process.get_deliverables()

    if not process.phases:
        responses.append(ModelResponse(text="Completed execute-goal.", usage=_usage()))
        return responses

    for phase in process.phases:
        files = deliverables.get(phase.id, [])
        if files:
            tool_calls: list[ToolCall] = []
            for index, path in enumerate(files, start=1):
                tool_calls.append(ToolCall(
                    id=f"{phase.id}-write-{index}",
                    name="write_file",
                    arguments={
                        "path": path,
                        "content": _content_for_deliverable(path, phase.id),
                    },
                ))
            responses.append(ModelResponse(
                text=f"Writing deliverables for {phase.id}.",
                tool_calls=tool_calls,
                usage=_usage(),
            ))
        completion_text = f"Completed {phase.id}."
        if phase.is_synthesis:
            upstream_inputs: list[str] = []
            for dep_id in phase.depends_on:
                dep_name = str(dep_id).strip()
                if not dep_name:
                    continue
                dep_files = [
                    str(path).strip()
                    for path in deliverables.get(dep_name, [])
                    if str(path).strip()
                ]
                if dep_files:
                    input_names = ", ".join(Path(path).name for path in dep_files)
                    upstream_inputs.append(f"{dep_name} ({input_names})")
                else:
                    upstream_inputs.append(dep_name)
            if upstream_inputs:
                completion_text = (
                    f"Completed {phase.id} using upstream inputs: "
                    + "; ".join(upstream_inputs)
                    + "."
                )
        responses.append(ModelResponse(
            text=completion_text,
            usage=_usage(),
        ))

    return responses


def _run_config_for_case(
    *,
    base: Config | None,
    database_path: Path,
    max_iterations: int,
    deterministic: bool,
) -> Config:
    if base is None:
        return Config(
            execution=ExecutionConfig(
                max_subtask_retries=1,
                max_loop_iterations=max_iterations,
                max_parallel_subtasks=1,
                enable_streaming=False,
            ),
            verification=VerificationConfig(
                tier1_enabled=True,
                tier2_enabled=not deterministic,
                tier3_enabled=False,
            ),
            memory=MemoryConfig(database_path=str(database_path)),
        )

    execution = replace(
        base.execution,
        max_parallel_subtasks=1,
        max_loop_iterations=max(base.execution.max_loop_iterations, max_iterations),
    )
    memory = replace(base.memory, database_path=str(database_path))
    verification = base.verification
    if deterministic:
        verification = replace(
            base.verification,
            tier2_enabled=False,
            tier3_enabled=False,
        )
    return replace(
        base,
        execution=execution,
        memory=memory,
        verification=verification,
    )


def _collect_failure_text(task) -> str:
    chunks: list[str] = []
    chunks.extend(e.error for e in task.errors_encountered if e.error)
    for subtask in task.plan.subtasks:
        if subtask.active_issue:
            chunks.append(subtask.active_issue)
        if subtask.summary:
            chunks.append(subtask.summary)
    return "\n".join(chunks)


def _missing_required_tools(
    *,
    case: ProcessTestCase,
    available_tools: list[str],
) -> list[str]:
    """Return required tools declared by the case but absent at runtime."""
    available = set(available_tools)
    return sorted(
        tool_name
        for tool_name in case.requires_tools
        if tool_name not in available
    )


def _evaluate_acceptance(
    *,
    process: ProcessDefinition,
    case: ProcessTestCase,
    workspace: Path,
    task,
) -> tuple[bool, list[str]]:
    details: list[str] = []

    if task.status != TaskStatus.COMPLETED:
        details.append(f"Task status is {task.status.value}, expected completed.")

    incomplete = [
        subtask.id
        for subtask in task.plan.subtasks
        if subtask.status != SubtaskStatus.COMPLETED
    ]
    if incomplete:
        details.append(
            "Subtasks not completed: " + ", ".join(incomplete)
        )

    plan_subtask_ids = {subtask.id for subtask in task.plan.subtasks}
    required_phases = (
        case.acceptance.phases_must_include
        if case.acceptance.phases_must_include
        else [phase.id for phase in process.phases]
    )
    missing_phases = sorted(
        phase_id for phase_id in required_phases if phase_id not in plan_subtask_ids
    )
    if missing_phases:
        details.append(
            "Required phases missing from plan: " + ", ".join(missing_phases)
        )

    required_deliverables = (
        case.acceptance.deliverables_must_exist
        if case.acceptance.deliverables_must_exist
        else _flatten_deliverables(process)
    )
    missing_deliverables = [
        path for path in required_deliverables if not (workspace / path).exists()
    ]
    if missing_deliverables:
        details.append(
            "Missing deliverables: " + ", ".join(sorted(missing_deliverables))
        )

    haystack = _collect_failure_text(task)
    for pattern in case.acceptance.verification_forbidden_patterns:
        try:
            if re.search(pattern, haystack):
                details.append(
                    f"Forbidden verification pattern matched: {pattern!r}"
                )
        except re.error as e:
            details.append(
                f"Invalid forbidden pattern {pattern!r}: {e}"
            )

    return (not details), details


def _write_event_log(
    *,
    event_bus: EventBus,
    path: Path,
) -> str:
    """Persist captured event stream to JSONL and return file path."""
    events = event_bus.recent_events(limit=10_000)
    if not events:
        return ""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for event in events:
            payload = {
                "event_type": event.event_type,
                "task_id": event.task_id,
                "timestamp": event.timestamp,
                "data": event.data,
            }
            f.write(json.dumps(payload) + "\n")
    return str(path)


async def run_process_case_deterministic(
    process: ProcessDefinition,
    case: ProcessTestCase,
    *,
    workspace: Path | None = None,
) -> ProcessCaseResult:
    """Run one deterministic process test case using scripted model responses."""
    start = time.monotonic()
    tmp_root = Path(
        tempfile.mkdtemp(prefix=f"loom-proc-test-{process.name}-")
    )
    ws = workspace.resolve() if workspace else (tmp_root / "workspace")
    ws.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / "process-test.db"

    run_config = _run_config_for_case(
        base=None,
        database_path=db_path,
        max_iterations=max(20, len(process.phases) * 4),
        deterministic=True,
    )
    tool_registry = create_default_registry(run_config)
    missing_tools = _missing_required_tools(
        case=case,
        available_tools=tool_registry.list_tools(),
    )
    if missing_tools:
        return ProcessCaseResult(
            case_id=case.id,
            mode=case.mode,
            passed=False,
            duration_seconds=0.0,
            message="Required tools missing for process test case.",
            details=[f"Missing tools: {', '.join(missing_tools)}"],
            task_status="tooling_error",
        )

    database = Database(str(db_path))
    await database.initialize()
    event_bus = EventBus()
    try:
        planner = _ScriptedModelProvider(
            name="process-test-planner",
            tier=2,
            roles=["planner"],
            responses=[_planner_response(process)],
        )
        executor = _ScriptedModelProvider(
            name="process-test-executor",
            tier=2,
            roles=["executor"],
            responses=_executor_responses(process),
        )
        router = ModelRouter(providers={
            "planner": planner,
            "executor": executor,
        })
        orchestrator = Orchestrator(
            model_router=router,
            tool_registry=tool_registry,
            memory_manager=MemoryManager(database),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_root / "state"),
            event_bus=event_bus,
            config=run_config,
            process=process,
        )
        task = create_task(
            goal=case.goal or _default_case(process).goal,
            workspace=str(ws),
            approval_mode="auto",
        )
        task_result = await asyncio.wait_for(
            orchestrator.execute_task(task),
            timeout=case.timeout_seconds,
        )
    except TimeoutError:
        elapsed = time.monotonic() - start
        return ProcessCaseResult(
            case_id=case.id,
            mode=case.mode,
            passed=False,
            duration_seconds=elapsed,
            message=f"Timed out after {case.timeout_seconds}s.",
            details=[],
            task_status="timeout",
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        return ProcessCaseResult(
            case_id=case.id,
            mode=case.mode,
            passed=False,
            duration_seconds=elapsed,
            message=f"Execution error: {type(e).__name__}: {e}",
            details=[],
            task_status="error",
        )
    finally:
        await database.close()

    passed, details = _evaluate_acceptance(
        process=process,
        case=case,
        workspace=ws,
        task=task_result,
    )
    event_log_path = _write_event_log(
        event_bus=event_bus,
        path=tmp_root / f"{case.id}-events.jsonl",
    )
    elapsed = time.monotonic() - start
    return ProcessCaseResult(
        case_id=case.id,
        mode=case.mode,
        passed=passed,
        duration_seconds=elapsed,
        message="Passed" if passed else "Acceptance checks failed",
        details=details,
        task_status=task_result.status.value,
        event_log_path=event_log_path,
    )


async def run_process_case_live(
    process: ProcessDefinition,
    case: ProcessTestCase,
    *,
    config: Config,
    workspace: Path | None = None,
) -> ProcessCaseResult:
    """Run one live process test case using configured real providers."""
    start = time.monotonic()
    if not config.models:
        return ProcessCaseResult(
            case_id=case.id,
            mode=case.mode,
            passed=False,
            duration_seconds=0.0,
            message="No models configured for live process test.",
            details=[],
            task_status="unconfigured",
        )

    tmp_root = Path(
        tempfile.mkdtemp(prefix=f"loom-proc-live-{process.name}-")
    )
    ws = workspace.resolve() if workspace else (tmp_root / "workspace")
    ws.mkdir(parents=True, exist_ok=True)
    db_path = tmp_root / "process-live.db"
    run_config = _run_config_for_case(
        base=config,
        database_path=db_path,
        max_iterations=max(config.execution.max_loop_iterations, len(process.phases) * 4),
        deterministic=False,
    )
    tool_registry = create_default_registry(run_config)
    missing_tools = _missing_required_tools(
        case=case,
        available_tools=tool_registry.list_tools(),
    )
    if missing_tools:
        return ProcessCaseResult(
            case_id=case.id,
            mode=case.mode,
            passed=False,
            duration_seconds=0.0,
            message="Required tools missing for process test case.",
            details=[f"Missing tools: {', '.join(missing_tools)}"],
            task_status="tooling_error",
        )

    database = Database(str(db_path))
    await database.initialize()
    event_bus = EventBus()
    try:
        router = ModelRouter.from_config(run_config)
        orchestrator = Orchestrator(
            model_router=router,
            tool_registry=tool_registry,
            memory_manager=MemoryManager(database),
            prompt_assembler=PromptAssembler(),
            state_manager=TaskStateManager(data_dir=tmp_root / "state"),
            event_bus=event_bus,
            config=run_config,
            process=process,
        )
        task = create_task(
            goal=case.goal or _default_case(process).goal,
            workspace=str(ws),
            approval_mode="auto",
        )
        task_result = await asyncio.wait_for(
            orchestrator.execute_task(task),
            timeout=case.timeout_seconds,
        )
    except TimeoutError:
        elapsed = time.monotonic() - start
        return ProcessCaseResult(
            case_id=case.id,
            mode=case.mode,
            passed=False,
            duration_seconds=elapsed,
            message=f"Timed out after {case.timeout_seconds}s.",
            details=[],
            task_status="timeout",
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        return ProcessCaseResult(
            case_id=case.id,
            mode=case.mode,
            passed=False,
            duration_seconds=elapsed,
            message=f"Execution error: {type(e).__name__}: {e}",
            details=[],
            task_status="error",
        )
    finally:
        await database.close()

    passed, details = _evaluate_acceptance(
        process=process,
        case=case,
        workspace=ws,
        task=task_result,
    )
    event_log_path = _write_event_log(
        event_bus=event_bus,
        path=tmp_root / f"{case.id}-events.jsonl",
    )
    elapsed = time.monotonic() - start
    return ProcessCaseResult(
        case_id=case.id,
        mode=case.mode,
        passed=passed,
        duration_seconds=elapsed,
        message="Passed" if passed else "Acceptance checks failed",
        details=details,
        task_status=task_result.status.value,
        event_log_path=event_log_path,
    )


async def run_process_tests(
    process: ProcessDefinition,
    *,
    config: Config,
    workspace: Path | None = None,
    include_live: bool = False,
    case_id: str | None = None,
) -> list[ProcessCaseResult]:
    """Run selected test cases for a process definition."""
    cases = select_process_test_cases(
        process,
        include_live=include_live,
        case_id=case_id,
    )

    results: list[ProcessCaseResult] = []
    for case in cases:
        if case.mode == "live":
            results.append(await run_process_case_live(
                process,
                case,
                config=config,
                workspace=workspace,
            ))
            continue
        results.append(await run_process_case_deterministic(
            process,
            case,
            workspace=workspace,
        ))

    return results
