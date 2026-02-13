"""API route handlers for Loom."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from loom.api.engine import Engine
from loom.api.schemas import (
    ApprovalRequest,
    FeedbackRequest,
    HealthResponse,
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

    task = create_task(
        goal=body.goal,
        workspace=body.workspace or "",
        approval_mode=body.approval_mode,
        context=body.context,
    )

    engine.state_manager.save(task)

    # Launch execution in background
    asyncio.create_task(_execute_in_background(engine, task))

    return TaskCreateResponse(
        task_id=task.id,
        status=task.status.value,
        message="Task created and execution started.",
    )


async def _execute_in_background(engine: Engine, task) -> None:
    """Run task execution without blocking the API response."""
    try:
        await engine.orchestrator.execute_task(task)
    except Exception:
        pass  # Errors are captured in the task state by the orchestrator


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

    return EventSourceResponse(event_generator())


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
    return [
        ModelInfo(
            name=p["name"],
            model=p["model"],
            tier=p["tier"],
            roles=p["roles"],
        )
        for p in providers
    ]


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
    return {
        "server": {"host": config.server.host, "port": config.server.port},
        "models": {
            name: {
                "provider": m.provider,
                "model": m.model,
                "roles": m.roles,
                "max_tokens": m.max_tokens,
            }
            for name, m in config.models.items()
        },
        "execution": {
            "max_loop_iterations": config.execution.max_loop_iterations,
            "max_subtask_retries": config.execution.max_subtask_retries,
        },
    }
