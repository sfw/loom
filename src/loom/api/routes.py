"""API route handlers for Loom."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from loom.api.engine import Engine
from loom.api.schemas import (
    ApprovalRequest,
    ConversationMessageRequest,
    FeedbackRequest,
    HealthResponse,
    ModelCapabilitiesResponse,
    ModelInfo,
    PlanResponse,
    ProgressResponse,
    SubtaskSummaryResponse,
    TaskCreateRequest,
    TaskCreateResponse,
    TaskListItem,
    TaskResponse,
    TaskSteerRequest,
    ToolInfo,
)
from loom.engine.orchestrator import create_task
from loom.events.bus import Event
from loom.state.memory import MemoryEntry
from loom.state.task_state import SubtaskStatus, TaskStatus
from loom.tools.workspace import validate_workspace

router = APIRouter()


def _get_engine(request: Request) -> Engine:
    """Get the engine from the app state."""
    return request.app.state.engine


# --- Task Lifecycle ---


@router.post("/tasks", response_model=TaskCreateResponse, status_code=201)
async def create_new_task(request: Request, body: TaskCreateRequest):
    """Create and start a new task."""
    engine = _get_engine(request)

    # Validate workspace if provided
    if body.workspace:
        valid, msg = validate_workspace(body.workspace)
        if not valid:
            raise HTTPException(status_code=400, detail=msg)

    # Resolve process definition if specified
    process_def = None
    if body.process:
        from loom.processes.schema import ProcessLoader, ProcessNotFoundError

        extra = [Path(p) for p in engine.config.process.search_paths]
        loader = ProcessLoader(
            workspace=Path(body.workspace) if body.workspace else None,
            extra_search_paths=extra,
        )
        try:
            process_def = loader.load(body.process)
        except ProcessNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Validate required tools early so clients get immediate feedback.
    if process_def is not None:
        tools_cfg = getattr(process_def, "tools", None)
        required = list(getattr(tools_cfg, "required", []) or [])
        excluded = set(getattr(tools_cfg, "excluded", []) or [])
        if required:
            from loom.tools import create_default_registry

            available = set(create_default_registry().list_tools()) - excluded
            missing = sorted(name for name in required if name not in available)
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Process '{getattr(process_def, 'name', body.process)}' "
                        f"requires missing tool(s): {', '.join(missing)}"
                    ),
                )

    task = create_task(
        goal=body.goal,
        workspace=body.workspace or "",
        approval_mode=body.approval_mode,
        callback_url=body.callback_url or "",
        context=body.context,
    )
    task.metadata["process"] = body.process or ""

    engine.state_manager.save(task)

    # Persist task metadata to SQLite (source of truth for /tasks list)
    await engine.database.insert_task(
        task_id=task.id,
        goal=task.goal,
        workspace_path=task.workspace,
        status=task.status.value,
        approval_mode=task.approval_mode,
        context=task.context or None,
        callback_url=task.callback_url or None,
        metadata=body.metadata or None,
    )

    # Register webhook if callback_url provided
    if task.callback_url:
        engine.webhook_delivery.register(task.id, task.callback_url)

    # Launch execution in background
    asyncio.create_task(_execute_in_background(engine, task, process_def))

    return TaskCreateResponse(
        task_id=task.id,
        status=task.status.value,
        message="Task created and execution started.",
    )


async def _execute_in_background(engine: Engine, task, process_def=None) -> None:
    """Run task execution without blocking the API response."""
    import logging
    _bg_logger = logging.getLogger(__name__)
    try:
        # Process-aware runs use an isolated orchestrator to avoid
        # cross-task state leakage on shared engine.orchestrator.
        orchestrator = engine.orchestrator
        if process_def is not None:
            orchestrator = engine.create_task_orchestrator(process_def)

        result = await orchestrator.execute_task(task)
        # Sync final status to SQLite
        try:
            await engine.database.update_task_status(result.id, result.status.value)
        except Exception:
            _bg_logger.warning("Failed to sync task %s status to DB", result.id)
    except Exception as e:
        _bg_logger.exception("Task %s failed with uncaught exception: %s", task.id, e)
        # Ensure task is marked failed even if orchestrator didn't catch it
        try:
            task.status = TaskStatus.FAILED
            engine.state_manager.save(task)
            await engine.database.update_task_status(task.id, TaskStatus.FAILED.value)
        except Exception:
            _bg_logger.exception("Failed to save error state for task %s", task.id)


@router.get("/tasks", response_model=list[TaskListItem])
async def list_tasks(request: Request, status: str | None = None):
    """List all tasks, optionally filtered by status."""
    engine = _get_engine(request)
    tasks = await engine.database.list_tasks(status=status)
    return [
        TaskListItem(
            task_id=t["id"],
            goal=t["goal"],
            status=t["status"],
            created_at=t["created_at"],
        )
        for t in tasks
    ]


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(request: Request, task_id: str):
    """Get full task state."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = engine.state_manager.load(task_id)

    # Build plan response
    plan_response = None
    if task.plan and task.plan.subtasks:
        plan_response = PlanResponse(
            version=task.plan.version,
            subtasks=[
                SubtaskSummaryResponse(
                    id=s.id,
                    description=s.description,
                    status=s.status.value,
                    depends_on=s.depends_on,
                    retry_count=s.retry_count,
                    summary=s.summary or "",
                )
                for s in task.plan.subtasks
            ],
        )

    # Build progress
    completed, total = task.progress
    failed = sum(
        1 for s in task.plan.subtasks if s.status == SubtaskStatus.FAILED
    ) if task.plan else 0
    pending = sum(
        1 for s in task.plan.subtasks if s.status == SubtaskStatus.PENDING
    ) if task.plan else 0
    running = sum(
        1 for s in task.plan.subtasks if s.status == SubtaskStatus.RUNNING
    ) if task.plan else 0

    progress = ProgressResponse(
        total_subtasks=total,
        completed=completed,
        failed=failed,
        pending=pending,
        running=running,
        percent_complete=(completed / total * 100) if total > 0 else 0,
    )

    return TaskResponse(
        task_id=task.id,
        goal=task.goal,
        status=task.status.value,
        workspace=task.workspace or None,
        plan=plan_response,
        created_at=task.created_at,
        updated_at=task.updated_at,
        approval_mode=task.approval_mode,
        progress=progress,
    )


@router.get("/tasks/{task_id}/stream")
async def stream_task_events(request: Request, task_id: str):
    """SSE stream of real-time task events."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    async def event_generator():
        queue: asyncio.Queue[Event] = asyncio.Queue()

        def handler(event: Event):
            if event.task_id == task_id:
                queue.put_nowait(event)

        engine.event_bus.subscribe_all(handler)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        "event": event.event_type,
                        "data": json.dumps({
                            "task_id": event.task_id,
                            **event.data,
                            "timestamp": event.timestamp,
                        }),
                    }
                    # Stop streaming when task is terminal
                    if event.event_type in (
                        "task_completed", "task_failed", "task_cancelled",
                    ):
                        return
                except TimeoutError:
                    # Send keepalive comment
                    yield {"comment": "keepalive"}
        except asyncio.CancelledError:
            return
        finally:
            engine.event_bus.unsubscribe_all(handler)

    return EventSourceResponse(event_generator())


@router.get("/tasks/{task_id}/tokens")
async def stream_task_tokens(request: Request, task_id: str):
    """SSE stream of raw model tokens for the active subtask."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    async def token_generator():
        queue: asyncio.Queue[Event] = asyncio.Queue()

        def handler(event: Event):
            if event.task_id == task_id and event.event_type == "token_streamed":
                queue.put_nowait(event)

        # Also listen for terminal events to know when to stop
        terminal_queue: asyncio.Queue[Event] = asyncio.Queue()

        def terminal_handler(event: Event):
            if event.task_id == task_id and event.event_type in (
                "task_completed", "task_failed", "task_cancelled",
            ):
                terminal_queue.put_nowait(event)

        engine.event_bus.subscribe("token_streamed", handler)
        engine.event_bus.subscribe_all(terminal_handler)

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield {
                        "event": "token",
                        "data": json.dumps({
                            "token": event.data.get("token", ""),
                            "subtask_id": event.data.get("subtask_id", ""),
                            "model": event.data.get("model", ""),
                        }),
                    }
                except TimeoutError:
                    # Check if task is done
                    if not terminal_queue.empty():
                        return
                    yield {"comment": "keepalive"}
        except asyncio.CancelledError:
            return
        finally:
            engine.event_bus.unsubscribe("token_streamed", handler)
            engine.event_bus.unsubscribe_all(terminal_handler)

    return EventSourceResponse(token_generator())


@router.patch("/tasks/{task_id}")
async def steer_task(request: Request, task_id: str, body: TaskSteerRequest):
    """Inject instructions into a running task."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = engine.state_manager.load(task_id)
    if task.status not in (TaskStatus.EXECUTING, TaskStatus.PLANNING):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot steer task in status: {task.status.value}",
        )

    # Store as user instruction in memory
    await engine.memory_manager.store(MemoryEntry(
        task_id=task_id,
        entry_type="user_instruction",
        summary=body.instruction[:150],
        detail=body.instruction,
        tags="steer",
    ))

    return {"status": "ok", "message": "Instruction injected."}


@router.delete("/tasks/{task_id}")
async def cancel_task(request: Request, task_id: str):
    """Cancel a running task."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = engine.state_manager.load(task_id)
    engine.orchestrator.cancel_task(task)

    return {"status": "ok", "message": f"Task {task_id} cancelled."}


@router.post("/tasks/{task_id}/approve")
async def approve_task(request: Request, task_id: str, body: ApprovalRequest):
    """Approve or reject a gated step."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    # Resolve pending approval via the approval manager
    resolved = engine.approval_manager.resolve_approval(
        task_id=task_id,
        subtask_id=body.subtask_id,
        approved=body.approved,
    )

    # Store approval decision in memory
    content = f"{'Approved' if body.approved else 'Rejected'}: {body.reason or 'No reason given'}"
    await engine.memory_manager.store(MemoryEntry(
        task_id=task_id,
        subtask_id=body.subtask_id,
        entry_type="decision",
        summary=content[:150],
        detail=content,
        tags="approval",
    ))

    return {
        "status": "ok",
        "approved": body.approved,
        "subtask_id": body.subtask_id,
        "resolved_pending": resolved,
    }


@router.post("/tasks/{task_id}/feedback")
async def submit_feedback(request: Request, task_id: str, body: FeedbackRequest):
    """Provide mid-task feedback."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    await engine.memory_manager.store(MemoryEntry(
        task_id=task_id,
        subtask_id=body.subtask_id or "",
        entry_type="user_instruction",
        summary=body.feedback[:150],
        detail=body.feedback,
        tags="feedback",
    ))

    return {"status": "ok", "message": "Feedback recorded."}


@router.post("/tasks/{task_id}/message")
async def send_conversation_message(
    request: Request, task_id: str, body: ConversationMessageRequest,
):
    """Send a conversational message to a running task.

    Messages are injected into the executor's context as memory entries,
    enabling back-and-forth clarification during execution.
    """
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = engine.state_manager.load(task_id)
    if task.status not in (TaskStatus.EXECUTING, TaskStatus.PLANNING):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot send message to task in status: {task.status.value}",
        )

    # Store as conversation turn in memory
    await engine.memory_manager.store(MemoryEntry(
        task_id=task_id,
        entry_type="user_instruction",
        summary=body.message[:150],
        detail=body.message,
        tags="conversation",
    ))

    # Emit conversation event
    engine.event_bus.emit(Event(
        event_type="conversation_message",
        task_id=task_id,
        data={
            "role": body.role,
            "message": body.message,
        },
    ))

    return {
        "status": "ok",
        "message": "Message delivered.",
        "task_id": task_id,
    }


@router.get("/tasks/{task_id}/conversation")
async def get_conversation_history(request: Request, task_id: str):
    """Retrieve conversation history for a task."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    entries = await engine.memory_manager.query(
        task_id=task_id,
        entry_type="user_instruction",
    )
    return [
        {
            "id": e.id,
            "message": e.detail,
            "summary": e.summary,
            "tags": e.tags,
            "timestamp": e.timestamp,
        }
        for e in entries
    ]


# --- Subtask Level ---


@router.get("/tasks/{task_id}/subtasks")
async def list_subtasks(request: Request, task_id: str):
    """List all subtasks with status."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = engine.state_manager.load(task_id)
    if not task.plan:
        return []

    return [
        SubtaskSummaryResponse(
            id=s.id,
            description=s.description,
            status=s.status.value,
            depends_on=s.depends_on,
            retry_count=s.retry_count,
            summary=s.summary or "",
        )
        for s in task.plan.subtasks
    ]


@router.get("/tasks/{task_id}/subtasks/{subtask_id}")
async def get_subtask(request: Request, task_id: str, subtask_id: str):
    """Get subtask detail."""
    engine = _get_engine(request)

    if not engine.state_manager.exists(task_id):
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = engine.state_manager.load(task_id)
    subtask = task.get_subtask(subtask_id)
    if subtask is None:
        raise HTTPException(status_code=404, detail=f"Subtask not found: {subtask_id}")

    return SubtaskSummaryResponse(
        id=subtask.id,
        description=subtask.description,
        status=subtask.status.value,
        depends_on=subtask.depends_on,
        retry_count=subtask.retry_count,
        summary=subtask.summary or "",
    )


# --- Memory ---


@router.get("/tasks/{task_id}/memory")
async def query_task_memory(request: Request, task_id: str, entry_type: str | None = None):
    """Query memory entries for a task."""
    engine = _get_engine(request)

    entries = await engine.memory_manager.query(
        task_id=task_id,
        entry_type=entry_type,
    )
    return [
        {
            "id": e.id,
            "task_id": e.task_id,
            "subtask_id": e.subtask_id,
            "entry_type": e.entry_type,
            "summary": e.summary,
            "detail": e.detail,
            "tags": e.tags,
            "timestamp": e.timestamp,
        }
        for e in entries
    ]


@router.get("/memory/search")
async def search_memory(request: Request, q: str, task_id: str | None = None):
    """Search across task memory."""
    engine = _get_engine(request)
    if not task_id:
        return []
    entries = await engine.memory_manager.search(task_id=task_id, query=q)
    return [
        {
            "id": e.id,
            "task_id": e.task_id,
            "entry_type": e.entry_type,
            "summary": e.summary,
            "detail": e.detail,
            "tags": e.tags,
        }
        for e in entries
    ]


# --- System ---


@router.get("/models", response_model=list[ModelInfo])
async def list_models(request: Request):
    """List available models and their health status."""
    engine = _get_engine(request)
    providers = engine.model_router.list_providers()
    result = []
    for p in providers:
        caps = p.get("capabilities")
        caps_response = None
        if caps:
            caps_response = ModelCapabilitiesResponse(
                vision=caps.get("vision", False),
                native_pdf=caps.get("native_pdf", False),
                thinking=caps.get("thinking", False),
                citations=caps.get("citations", False),
                audio_input=caps.get("audio_input", False),
                audio_output=caps.get("audio_output", False),
            )
        result.append(ModelInfo(
            name=p["name"],
            model=p["model"],
            tier=p["tier"],
            roles=p["roles"],
            capabilities=caps_response,
        ))
    return result


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools(request: Request):
    """List available tools and schemas."""
    engine = _get_engine(request)
    schemas = engine.tool_registry.all_schemas()
    return [
        ToolInfo(name=s["name"], description=s.get("description", ""))
        for s in schemas
    ]


@router.get("/health", response_model=HealthResponse)
async def health():
    """System health check."""
    from loom import __version__

    return HealthResponse(status="ok", version=__version__)


@router.get("/config")
async def get_config(request: Request):
    """Current configuration (redacted)."""
    engine = _get_engine(request)
    config = engine.config

    def _caps_dict(m):
        caps = m.resolved_capabilities
        return {
            "vision": caps.vision,
            "native_pdf": caps.native_pdf,
            "thinking": caps.thinking,
        }

    return {
        "server": {"host": config.server.host, "port": config.server.port},
        "models": {
            name: {
                "provider": m.provider,
                "model": m.model,
                "roles": m.roles,
                "max_tokens": m.max_tokens,
                "capabilities": _caps_dict(m),
            }
            for name, m in config.models.items()
        },
        "execution": {
            "max_loop_iterations": config.execution.max_loop_iterations,
            "max_subtask_retries": config.execution.max_subtask_retries,
        },
    }
