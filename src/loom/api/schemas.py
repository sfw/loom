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
    message: str = ""
    role: str = "user"
    workspace_paths: list[str] = Field(default_factory=list)
    workspace_files: list[str] = Field(default_factory=list)
    workspace_directories: list[str] = Field(default_factory=list)
    content_blocks: list[dict[str, Any]] = Field(default_factory=list)


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


class ModelPatchRequest(BaseModel):
    max_tokens: int | None = None
    temperature: float | None = None


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


class SetupDiscoverModelsRequest(BaseModel):
    provider: str
    base_url: str
    api_key: str = ""


class SetupModelDraftRequest(BaseModel):
    name: str
    provider: str
    base_url: str = ""
    model: str
    api_key: str = ""
    roles: list[str] = Field(default_factory=list)
    max_tokens: int = 8192
    temperature: float = 0.1


class SetupCompleteRequest(BaseModel):
    models: list[SetupModelDraftRequest] = Field(default_factory=list)


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
    provider: str = ""
    base_url: str = ""
    model: str
    model_id: str = ""
    tier: int
    roles: list[str]
    max_tokens: int = 0
    temperature: float = 0.0
    capabilities: ModelCapabilitiesResponse | None = None


class ToolInfo(BaseModel):
    name: str
    description: str
    auth_mode: str = "no_auth"
    auth_required: bool = False
    auth_requirements: list[dict[str, Any]] = Field(default_factory=list)
    execution_surfaces: list[str] = Field(default_factory=list)
    availability_state: str = "available"
    runnable: bool = True
    availability_checked_at: str = ""
    availability_reasons: list[dict[str, Any]] = Field(default_factory=list)


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
    tool_availability: list[dict[str, Any]] = Field(default_factory=list)


class SetupProviderResponse(BaseModel):
    display_name: str
    provider_key: str
    needs_api_key: bool = False
    default_base_url: str = ""


class SetupStatusResponse(BaseModel):
    needs_setup: bool
    config_path: str
    providers: list[SetupProviderResponse] = Field(default_factory=list)
    role_presets: dict[str, list[str]] = Field(default_factory=dict)


class SetupDiscoverModelsResponse(BaseModel):
    models: list[str] = Field(default_factory=list)


class SetupCompleteResponse(BaseModel):
    status: str
    config_path: str


class ActivitySummaryResponse(BaseModel):
    status: str
    active: bool
    active_conversation_count: int = 0
    active_run_count: int = 0
    updated_at: str = ""


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


class RunFailureRemediationResponse(BaseModel):
    attempted: bool = False
    queued: bool = False
    resolved: bool = False
    failed: bool = False
    expired: bool = False
    why_not_remedied: str = ""


class RunFailureAnalysisResponse(BaseModel):
    headline: str = ""
    summary: str = ""
    failing_subtask_id: str = ""
    failing_subtask_label: str = ""
    primary_reason_code: str = ""
    reason_family: str = ""
    technical_detail: str = ""
    evidence: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    remediation: RunFailureRemediationResponse = Field(
        default_factory=RunFailureRemediationResponse,
    )


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
    failure_analysis: RunFailureAnalysisResponse | None = None


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
    workspace_id: str = ""
    workspace_display_name: str = ""
    workspace_path: str = ""
    conversation_id: str = ""
    run_id: str = ""
    approval_item_id: str = ""
    path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSearchResponse(BaseModel):
    workspace: WorkspaceSummaryResponse | None = None
    query: str
    total_results: int = 0
    workspaces: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    conversations: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    runs: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    approvals: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    artifacts: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    files: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    processes: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
    accounts: list[WorkspaceSearchItemResponse] = Field(default_factory=list)
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


class IntegrationAuthStateResponse(BaseModel):
    state: str
    label: str = ""
    reason: str = ""
    storage: str = ""
    has_token: bool = False
    expired: bool = False
    expires_at: int | None = None
    token_type: str | None = None
    scopes: list[str] = Field(default_factory=list)
    profile_id: str = ""
    account_label: str = ""
    mode: str = ""


class IntegrationEffectiveAccountResponse(BaseModel):
    profile_id: str
    provider: str
    account_label: str = ""
    mode: str
    status: str = "ready"
    source: str = ""
    source_path: str = ""
    routing_reason: str = ""
    auth_state: IntegrationAuthStateResponse


class MCPServerManagementResponse(BaseModel):
    alias: str
    type: str
    enabled: bool = True
    source: str = ""
    source_path: str = ""
    source_label: str = ""
    command: str = ""
    args: list[str] = Field(default_factory=list)
    url: str = ""
    fallback_sse_url: str = ""
    cwd: str = ""
    timeout_seconds: int = 0
    oauth_enabled: bool = False
    oauth_scopes: list[str] = Field(default_factory=list)
    allow_insecure_http: bool = False
    allow_private_network: bool = False
    trust_state: str = ""
    trust_summary: str = ""
    approval_required: bool = False
    approval_state: str = "not_required"
    runtime_state: str = ""
    resource_id: str = ""
    auth_provider: str = ""
    auth_state: IntegrationAuthStateResponse
    effective_account: IntegrationEffectiveAccountResponse | None = None
    bound_profile_ids: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class AccountInfoResponse(BaseModel):
    profile_id: str
    provider: str
    account_label: str = ""
    mode: str
    status: str = "ready"
    source: str = ""
    source_path: str = ""
    mcp_server: str = ""
    token_ref: str = ""
    secret_ref: str = ""
    writable_storage_kind: str = ""
    auth_state: IntegrationAuthStateResponse
    default_selectors: list[str] = Field(default_factory=list)
    bound_resource_refs: list[str] = Field(default_factory=list)
    used_by_mcp_servers: list[str] = Field(default_factory=list)
    effective_for_mcp_servers: list[str] = Field(default_factory=list)
    remediation: list[str] = Field(default_factory=list)


class AccountCreateRequest(BaseModel):
    profile_id: str
    provider: str
    mode: str
    account_label: str = ""
    mcp_server: str = ""
    secret_ref: str = ""
    token_ref: str = ""
    scopes: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    command: str = ""
    auth_check: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    status: str = "draft"


class AccountUpdateRequest(BaseModel):
    account_label: str | None = None
    mcp_server: str | None = None
    clear_mcp_server: bool = False
    secret_ref: str | None = None
    token_ref: str | None = None
    scopes: list[str] | None = None
    env: dict[str, str] | None = None
    command: str | None = None
    auth_check: list[str] | None = None
    metadata: dict[str, str] | None = None


class WorkspaceIntegrationsResponse(BaseModel):
    workspace: WorkspaceSummaryResponse
    mcp_servers: list[MCPServerManagementResponse] = Field(default_factory=list)
    accounts: list[AccountInfoResponse] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class MCPServerCreateRequest(BaseModel):
    alias: str
    type: str
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    fallback_sse_url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    oauth_enabled: bool = False
    oauth_scopes: list[str] = Field(default_factory=list)
    allow_insecure_http: bool = False
    allow_private_network: bool = False
    cwd: str = ""
    timeout_seconds: int = 30
    enabled: bool = True


class MCPServerUpdateRequest(BaseModel):
    type: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    fallback_sse_url: str | None = None
    headers: dict[str, str] | None = None
    oauth_enabled: bool | None = None
    oauth_scopes: list[str] | None = None
    allow_insecure_http: bool | None = None
    allow_private_network: bool | None = None
    cwd: str | None = None
    timeout_seconds: int | None = None
    enabled: bool | None = None


class MCPServerActionResponse(BaseModel):
    alias: str
    status: str = ""
    message: str = ""
    tool_count: int = 0
    tool_names: list[str] = Field(default_factory=list)


class IntegrationOAuthStartResponse(BaseModel):
    flow_id: str
    authorization_url: str
    redirect_uri: str = ""
    callback_mode: str = ""
    expires_at_unix: int = 0
    browser_warning: str = ""


class IntegrationOAuthCompleteRequest(BaseModel):
    flow_id: str
    callback_input: str = ""


class IntegrationOAuthCompleteResponse(BaseModel):
    status: str
    message: str = ""
    account: AccountInfoResponse | None = None
    expires_at: int | None = None
    scopes: list[str] = Field(default_factory=list)


class AuthDraftSyncResponse(BaseModel):
    created_drafts: int = 0
    created_bindings: int = 0
    updated_defaults: int = 0
    warnings: list[str] = Field(default_factory=list)
    integrations: WorkspaceIntegrationsResponse


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
