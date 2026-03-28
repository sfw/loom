"""Pydantic request/response schemas for the Loom API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# --- Request Schemas ---


class TaskCreateRequest(BaseModel):
    goal: str
    workspace: str | None = None
    context: dict = Field(default_factory=dict)
    approval_mode: str = "auto"
    callback_url: str | None = None
    metadata: dict = Field(default_factory=dict)
    process: str | None = None
    auto_subfolder: bool = False


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


class ConversationApprovalDecisionRequest(BaseModel):
    decision: str = "approve"


class ConversationInjectRequest(BaseModel):
    instruction: str


class ConversationCreateRequest(BaseModel):
    model_name: str = ""
    system_prompt: str = ""


class ConversationPatchRequest(BaseModel):
    title: str | None = None


class TaskQuestionAnswerRequest(BaseModel):
    response_type: str | None = None
    selected_option_ids: list[str] = Field(default_factory=list)
    selected_labels: list[str] = Field(default_factory=list)
    custom_response: str = ""
    source: str = "api"
    answered_by: str | None = None
    client_id: str | None = None


class TelemetrySettingsPatchRequest(BaseModel):
    mode: str
    persist: bool = False


class WorkspaceCreateRequest(BaseModel):
    path: str
    display_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspacePatchRequest(BaseModel):
    display_name: str | None = None
    sort_order: int | None = None
    archived: bool | None = None
    last_opened_at: str | None = None
    metadata: dict[str, Any] | None = None


class SettingsPatchRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    clear_paths: list[str] = Field(default_factory=list)
    persist: bool = False


class WorkspaceSettingsPatchRequest(BaseModel):
    overrides: dict[str, Any] = Field(default_factory=dict)


class ApprovalReplyRequest(BaseModel):
    decision: str = "approve"
    reason: str | None = None
    response_type: str | None = None
    selected_option_ids: list[str] = Field(default_factory=list)
    selected_labels: list[str] = Field(default_factory=list)
    custom_response: str = ""
    source: str = "api"
    answered_by: str | None = None
    client_id: str | None = None


# --- Response Schemas ---


class TaskCreateResponse(BaseModel):
    task_id: str
    status: str
    message: str
    run_id: str = ""


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
    run_id: str = ""
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
    ready: bool = True
    runtime_role: str = ""


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
    model_id: str = ""
    tier: int
    roles: list[str]
    capabilities: ModelCapabilitiesResponse | None = None


class ToolInfo(BaseModel):
    name: str
    description: str
    auth_mode: str = "no_auth"
    auth_required: bool = False
    auth_requirements: list[dict[str, Any]] = Field(default_factory=list)
    execution_surfaces: list[str] = Field(default_factory=list)


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
    extracted_text: str = ""
    thinking: str = ""
    signature: str = ""


class TaskQuestionResponse(BaseModel):
    question_id: str
    task_id: str
    subtask_id: str
    status: str
    request_payload: dict[str, Any] = Field(default_factory=dict)
    answer_payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    resolved_at: str = ""
    timeout_at: str = ""


class TelemetrySettingsResponse(BaseModel):
    configured_mode: str
    runtime_override_mode: str = ""
    effective_mode: str
    scope: str
    updated_at: str


class RuntimeStatusResponse(BaseModel):
    status: str
    ready: bool
    version: str
    runtime_role: str
    started_at: str
    config_path: str = ""
    database_path: str = ""
    scratch_dir: str = ""
    host: str = ""
    port: int = 0
    workspace_default_path: str = ""


class WorkspaceSummaryResponse(BaseModel):
    id: str
    canonical_path: str
    display_name: str
    workspace_type: str = "local"
    is_archived: bool = False
    sort_order: int = 0
    last_opened_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    exists_on_disk: bool = False
    conversation_count: int = 0
    run_count: int = 0
    active_run_count: int = 0
    last_activity_at: str = ""


class ConversationSummaryResponse(BaseModel):
    id: str
    workspace_id: str
    workspace_path: str
    model_name: str
    title: str
    turn_count: int = 0
    total_tokens: int = 0
    last_active_at: str = ""
    started_at: str = ""
    is_active: bool = False
    linked_run_ids: list[str] = Field(default_factory=list)


class RunSummaryResponse(BaseModel):
    id: str
    workspace_id: str
    workspace_path: str
    goal: str
    status: str
    created_at: str = ""
    updated_at: str = ""
    execution_run_id: str = ""
    process_name: str = ""
    linked_conversation_ids: list[str] = Field(default_factory=list)
    changed_files_count: int = 0


class RunArtifactResponse(BaseModel):
    path: str
    category: str = ""
    source: str = ""
    sha256: str = ""
    size_bytes: int = 0
    exists_on_disk: bool = False
    is_intermediate: bool = False
    created_at: str = ""
    tool_name: str = ""
    subtask_ids: list[str] = Field(default_factory=list)
    phase_ids: list[str] = Field(default_factory=list)
    facets: dict[str, Any] = Field(default_factory=dict)


class WorkspaceArtifactResponse(RunArtifactResponse):
    latest_run_id: str = ""
    run_ids: list[str] = Field(default_factory=list)
    run_count: int = 0


class WorkspaceSearchItemResponse(BaseModel):
    kind: str
    item_id: str = ""
    title: str
    subtitle: str = ""
    snippet: str = ""
    badges: list[str] = Field(default_factory=list)
    conversation_id: str = ""
    run_id: str = ""
    approval_item_id: str = ""
    path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSearchResponse(BaseModel):
    workspace: WorkspaceSummaryResponse
    query: str
    total_results: int = 0
    conversations: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    runs: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    approvals: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    artifacts: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    files: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    processes: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    mcp_servers: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    tools: list[WorkspaceSearchItemResponse] = Field(default_factory=list)


class WorkspaceOverviewResponse(BaseModel):
    workspace: WorkspaceSummaryResponse
    recent_conversations: list[ConversationSummaryResponse] = Field(default_factory=list)
    recent_runs: list[RunSummaryResponse] = Field(default_factory=list)
    pending_approvals_count: int = 0
    counts: dict[str, int] = Field(default_factory=dict)


class ProcessInfoResponse(BaseModel):
    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    path: str = ""


class MCPServerInfoResponse(BaseModel):
    alias: str
    type: str
    enabled: bool = True
    source: str = ""
    command: str = ""
    url: str = ""
    cwd: str = ""
    timeout_seconds: int = 0
    oauth_enabled: bool = False


class WorkspaceInventoryResponse(BaseModel):
    workspace: WorkspaceSummaryResponse
    processes: list[ProcessInfoResponse] = Field(default_factory=list)
    mcp_servers: list[MCPServerInfoResponse] = Field(default_factory=list)
    tools: list[ToolInfo] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class WorkspaceFileEntryResponse(BaseModel):
    path: str
    name: str
    is_dir: bool = False
    size_bytes: int = 0
    modified_at: str = ""
    extension: str = ""


class WorkspaceFilePreviewTableResponse(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    truncated: bool = False


class WorkspaceFilePreviewResponse(BaseModel):
    path: str
    name: str
    extension: str = ""
    size_bytes: int = 0
    modified_at: str = ""
    preview_kind: str = "text"
    language: str = ""
    text_content: str = ""
    table: WorkspaceFilePreviewTableResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False
    error: str = ""


class ApprovalFeedItemResponse(BaseModel):
    id: str
    kind: str
    status: str = "pending"
    created_at: str = ""
    title: str = ""
    summary: str = ""
    workspace_id: str = ""
    workspace_path: str = ""
    workspace_display_name: str = ""
    task_id: str = ""
    run_id: str = ""
    conversation_id: str = ""
    subtask_id: str = ""
    question_id: str = ""
    approval_id: str = ""
    tool_name: str = ""
    risk_level: str = ""
    request_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
