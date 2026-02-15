"""Pydantic request/response schemas for the Loom API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Request Schemas ---


class TaskCreateRequest(BaseModel):
    goal: str
    workspace: str | None = None
    context: dict = Field(default_factory=dict)
    approval_mode: str = "auto"
    callback_url: str | None = None
    metadata: dict = Field(default_factory=dict)


class TaskSteerRequest(BaseModel):
    instruction: str


class ApprovalRequest(BaseModel):
    subtask_id: str
    approved: bool
    reason: str | None = None


class FeedbackRequest(BaseModel):
    feedback: str
    subtask_id: str | None = None


class ConversationMessageRequest(BaseModel):
    message: str
    role: str = "user"


# --- Response Schemas ---


class TaskCreateResponse(BaseModel):
    task_id: str
    status: str
    message: str


class SubtaskSummaryResponse(BaseModel):
    id: str
    description: str
    status: str
    depends_on: list[str] = Field(default_factory=list)
    retry_count: int = 0
    summary: str = ""


class PlanResponse(BaseModel):
    version: int
    subtasks: list[SubtaskSummaryResponse]


class ProgressResponse(BaseModel):
    total_subtasks: int
    completed: int
    failed: int
    pending: int
    running: int
    percent_complete: float


class TaskResponse(BaseModel):
    task_id: str
    goal: str
    status: str
    workspace: str | None
    plan: PlanResponse | None
    created_at: str
    updated_at: str
    approval_mode: str
    progress: ProgressResponse


class TaskListItem(BaseModel):
    task_id: str
    goal: str
    status: str
    created_at: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    task_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ModelCapabilitiesResponse(BaseModel):
    vision: bool = False
    native_pdf: bool = False
    thinking: bool = False
    citations: bool = False
    audio_input: bool = False
    audio_output: bool = False


class ModelInfo(BaseModel):
    name: str
    model: str
    tier: int
    roles: list[str]
    capabilities: ModelCapabilitiesResponse | None = None


class ToolInfo(BaseModel):
    name: str
    description: str


class ContentBlockResponse(BaseModel):
    """A multimodal content block from a tool result."""

    type: str
    source_path: str = ""
    media_type: str = ""
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    page_count: int = 0
    page_range: list[int] | None = None
    text_fallback: str = ""
