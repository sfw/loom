# Spec 07: API Server

## Overview

The FastAPI server is Loom's primary interface. Everything — the TUI, web dashboard, external agents, MCP clients — is a consumer of this API. The API is the product.

## Server Setup

```python
# api/server.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize engine, database, model connections
    engine = await create_engine(app.state.config)
    app.state.engine = engine
    yield
    # Shutdown: graceful cleanup
    await engine.shutdown()

app = FastAPI(
    title="Loom",
    description="Local LLM Agentic Task Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)
```

## Endpoints

### Task Lifecycle

```
POST   /tasks                         Create and start a new task
GET    /tasks                         List all tasks (with filters)
GET    /tasks/{task_id}               Get full task state
GET    /tasks/{task_id}/stream        SSE stream of real-time events
PATCH  /tasks/{task_id}               Steer a running task (inject instructions)
DELETE /tasks/{task_id}               Cancel a running task
POST   /tasks/{task_id}/approve       Approve a gated step
POST   /tasks/{task_id}/reject        Reject a gated step (with reason)
POST   /tasks/{task_id}/feedback      Provide correction or context mid-task
```

### Subtask Level

```
GET    /tasks/{task_id}/subtasks                  List all subtasks with status
GET    /tasks/{task_id}/subtasks/{subtask_id}     Get subtask detail + result
GET    /tasks/{task_id}/subtasks/{subtask_id}/trace   Full execution trace (prompts, responses, tool calls)
```

### Memory

```
GET    /tasks/{task_id}/memory        Query memory entries for a task
GET    /memory/search?q=...           Search across all task memory
```

### System

```
GET    /models                        Available models and health status
GET    /tools                         Available tools and schemas
GET    /health                        System health check
GET    /config                        Current configuration (redacted)
```

## Request/Response Schemas

### POST /tasks

```python
class TaskCreateRequest(BaseModel):
    goal: str                                      # Natural language goal
    workspace: str | None = None                   # Path to working directory
    context: dict = {}                             # Additional context
    approval_mode: Literal["auto", "manual", "confidence_threshold"] = "auto"
    callback_url: str | None = None                # Webhook for completion
    metadata: dict = {}                            # Arbitrary metadata

class TaskCreateResponse(BaseModel):
    task_id: str
    status: str
    message: str
```

### GET /tasks/{task_id}

```python
class TaskResponse(BaseModel):
    id: str
    goal: str
    status: str
    workspace: str | None
    plan: PlanResponse | None
    created_at: str
    updated_at: str
    approval_mode: str
    progress: ProgressResponse

class PlanResponse(BaseModel):
    version: int
    subtasks: list[SubtaskSummary]
    replanned_at: str | None

class SubtaskSummary(BaseModel):
    id: str
    description: str
    status: str
    depends_on: list[str]
    retry_count: int

class ProgressResponse(BaseModel):
    total_subtasks: int
    completed: int
    failed: int
    pending: int
    running: int
    percent_complete: float
```

### GET /tasks/{task_id}/stream (SSE)

Server-Sent Events stream. Each event is a JSON-encoded TaskEvent:

```
event: subtask_started
data: {"task_id": "abc", "subtask_id": "install-deps", "timestamp": "..."}

event: subtask_progress
data: {"task_id": "abc", "subtask_id": "install-deps", "detail": "Running npm install..."}

event: subtask_completed
data: {"task_id": "abc", "subtask_id": "install-deps", "status": "success", "summary": "..."}

event: approval_requested
data: {"task_id": "abc", "subtask_id": "drop-table", "reason": "Destructive operation", "details": {...}}

event: task_completed
data: {"task_id": "abc", "status": "completed", "summary": "...", "artifacts": [...]}
```

Implementation using sse-starlette:

```python
from sse_starlette.sse import EventSourceResponse

@router.get("/tasks/{task_id}/stream")
async def stream_task_events(task_id: str):
    async def event_generator():
        async for event in engine.event_bus.subscribe(task_id):
            yield {
                "event": event.event_type,
                "data": json.dumps(event.data),
                "id": event.correlation_id,
            }
    return EventSourceResponse(event_generator())
```

### PATCH /tasks/{task_id}

```python
class TaskSteerRequest(BaseModel):
    instruction: str                               # Mid-task instruction injection
    # e.g., "Use PostgreSQL instead of MySQL for the migration"
```

Injects a `user_instruction` memory entry and modifies the current subtask context on next iteration.

### POST /tasks/{task_id}/approve

```python
class ApprovalRequest(BaseModel):
    subtask_id: str
    approved: bool
    reason: str | None = None
```

### POST /tasks/{task_id}/feedback

```python
class FeedbackRequest(BaseModel):
    feedback: str
    subtask_id: str | None = None                  # Specific subtask, or general
```

Stored as a `user_instruction` memory entry and injected into subsequent prompts.

## Error Responses

All errors follow a consistent format:

```python
class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    task_id: str | None = None

# HTTP status codes:
# 400 — Invalid request (bad goal, invalid workspace path)
# 404 — Task or subtask not found
# 409 — Conflict (e.g., approving already-completed subtask)
# 422 — Validation error (Pydantic)
# 500 — Internal engine error
# 503 — No models available
```

## CORS and Security

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*"],           # Local only for V1
    allow_methods=["*"],
    allow_headers=["*"],
)
```

V1 is local-only. No authentication. Binds to 127.0.0.1 by default.

## Server Configuration

```toml
[server]
host = "127.0.0.1"
port = 9000
workers = 1                # Single worker — engine is single-threaded
```

## Acceptance Criteria

- [ ] `loom serve` starts FastAPI on configured host:port
- [ ] POST /tasks creates a task and returns task_id immediately
- [ ] GET /tasks/{id} returns full task state including plan and progress
- [ ] GET /tasks/{id}/stream delivers SSE events in real-time
- [ ] PATCH /tasks/{id} injects instructions into running task
- [ ] DELETE /tasks/{id} cancels a running task gracefully
- [ ] POST /tasks/{id}/approve unblocks a gated subtask
- [ ] GET /models returns model health status
- [ ] GET /health returns system status
- [ ] All endpoints return consistent error format
- [ ] OpenAPI docs available at /docs
- [ ] SSE stream reconnects cleanly (client can resume with last event ID)
- [ ] Workspace path is validated on task creation (must exist, must be directory)
