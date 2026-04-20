export interface RuntimeStatus {
  status: string;
  ready: boolean;
  version: string;
  runtime_role: string;
  started_at: string;
  config_path: string;
  database_path: string;
  scratch_dir: string;
  host: string;
  port: number;
  workspace_default_path: string;
}

export interface SetupProviderInfo {
  display_name: string;
  provider_key: string;
  needs_api_key: boolean;
  default_base_url: string;
}

export interface SetupStatus {
  needs_setup: boolean;
  config_path: string;
  providers: SetupProviderInfo[];
  role_presets: Record<string, string[]>;
}

export interface SetupDiscoverModelsRequest {
  provider: string;
  base_url: string;
  api_key?: string;
}

export interface SetupDiscoverModelsResponse {
  models: string[];
}

export interface SetupModelDraft {
  name: string;
  provider: string;
  base_url?: string;
  model: string;
  api_key?: string;
  roles: string[];
  max_tokens?: number;
  temperature?: number;
}

export interface SetupCompleteRequest {
  models: SetupModelDraft[];
}

export interface SetupCompleteResponse {
  status: string;
  config_path: string;
}

export interface ActivitySummary {
  status: string;
  active: boolean;
  active_conversation_count: number;
  active_run_count: number;
  updated_at: string;
}

export interface ModelInfo {
  name: string;
  provider?: string;
  base_url?: string;
  model: string;
  model_id: string;
  tier: number;
  roles: string[];
  max_tokens?: number;
  temperature?: number;
}

export interface ModelPatchRequest {
  max_tokens?: number;
  temperature?: number;
}

export interface ToolInfo {
  name: string;
  description: string;
  auth_mode: string;
  auth_required: boolean;
  auth_requirements: Array<Record<string, unknown>>;
  execution_surfaces: string[];
}

export interface WorkspaceSummary {
  id: string;
  canonical_path: string;
  display_name: string;
  workspace_type: string;
  is_archived: boolean;
  sort_order: number;
  last_opened_at: string;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
  exists_on_disk: boolean;
  conversation_count: number;
  run_count: number;
  active_run_count: number;
  last_activity_at: string;
}

export interface ConversationSummary {
  id: string;
  workspace_id: string;
  workspace_path: string;
  model_name: string;
  title: string;
  turn_count: number;
  total_tokens: number;
  last_active_at: string;
  started_at: string;
  is_active: boolean;
  linked_run_ids: string[];
}

export interface RunSummary {
  id: string;
  workspace_id: string;
  workspace_path: string;
  goal: string;
  status: string;
  created_at: string;
  updated_at: string;
  execution_run_id: string;
  process_name: string;
  linked_conversation_ids: string[];
  changed_files_count: number;
  failure_analysis?: RunFailureAnalysis | null;
}

export interface RunFailureRemediation {
  attempted: boolean;
  queued: boolean;
  resolved: boolean;
  failed: boolean;
  expired: boolean;
  why_not_remedied: string;
}

export interface RunFailureAnalysis {
  headline: string;
  summary: string;
  failing_subtask_id: string;
  failing_subtask_label: string;
  primary_reason_code: string;
  reason_family: string;
  technical_detail: string;
  evidence: string[];
  next_actions: string[];
  remediation: RunFailureRemediation;
}

export interface WorkspaceOverview {
  workspace: WorkspaceSummary;
  recent_conversations: ConversationSummary[];
  recent_runs: RunSummary[];
  pending_approvals_count: number;
  counts: Record<string, number>;
}

export interface ProcessInfo {
  name: string;
  version: string;
  description: string;
  author: string;
  path: string;
}

export interface MCPServerInfo {
  alias: string;
  type: string;
  enabled: boolean;
  source: string;
  command: string;
  url: string;
  cwd: string;
  timeout_seconds: number;
  oauth_enabled: boolean;
}

export interface WorkspaceInventory {
  workspace: WorkspaceSummary;
  processes: ProcessInfo[];
  mcp_servers: MCPServerInfo[];
  tools: ToolInfo[];
  counts: Record<string, number>;
}

export interface IntegrationAuthState {
  state: string;
  label: string;
  reason: string;
  storage: string;
  has_token: boolean;
  expired: boolean;
  expires_at: number | null;
  token_type: string | null;
  scopes: string[];
  profile_id: string;
  account_label: string;
  mode: string;
}

export interface IntegrationEffectiveAccount {
  profile_id: string;
  provider: string;
  account_label: string;
  mode: string;
  status: string;
  source: string;
  source_path: string;
  routing_reason: string;
  auth_state: IntegrationAuthState;
}

export interface MCPServerManagementInfo {
  alias: string;
  type: string;
  enabled: boolean;
  source: string;
  source_path: string;
  source_label: string;
  command: string;
  args: string[];
  url: string;
  fallback_sse_url: string;
  cwd: string;
  timeout_seconds: number;
  oauth_enabled: boolean;
  oauth_scopes: string[];
  allow_insecure_http: boolean;
  allow_private_network: boolean;
  trust_state: string;
  trust_summary: string;
  approval_required: boolean;
  approval_state: string;
  runtime_state: string;
  resource_id: string;
  auth_provider: string;
  auth_state: IntegrationAuthState;
  effective_account: IntegrationEffectiveAccount | null;
  bound_profile_ids: string[];
  remediation: string[];
  flags: string[];
}

export interface AccountInfo {
  profile_id: string;
  provider: string;
  account_label: string;
  mode: string;
  status: string;
  source: string;
  source_path: string;
  mcp_server: string;
  token_ref: string;
  secret_ref: string;
  writable_storage_kind: string;
  auth_state: IntegrationAuthState;
  default_selectors: string[];
  bound_resource_refs: string[];
  used_by_mcp_servers: string[];
  effective_for_mcp_servers: string[];
  remediation: string[];
}

export interface AccountCreateRequest {
  profile_id: string;
  provider: string;
  mode: string;
  account_label?: string;
  mcp_server?: string;
  secret_ref?: string;
  token_ref?: string;
  scopes?: string[];
  env?: Record<string, string>;
  command?: string;
  auth_check?: string[];
  metadata?: Record<string, string>;
  status?: string;
}

export interface AccountUpdateRequest {
  account_label?: string;
  mcp_server?: string;
  clear_mcp_server?: boolean;
  secret_ref?: string;
  token_ref?: string;
  scopes?: string[];
  env?: Record<string, string>;
  command?: string;
  auth_check?: string[];
  metadata?: Record<string, string>;
}

export interface WorkspaceIntegrations {
  workspace: WorkspaceSummary;
  mcp_servers: MCPServerManagementInfo[];
  accounts: AccountInfo[];
  counts: Record<string, number>;
}

export interface MCPServerActionResult {
  alias: string;
  status: string;
  message: string;
  tool_count: number;
  tool_names: string[];
}

export interface MCPServerCreateRequest {
  alias: string;
  type: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  fallback_sse_url?: string;
  headers?: Record<string, string>;
  oauth_enabled?: boolean;
  oauth_scopes?: string[];
  allow_insecure_http?: boolean;
  allow_private_network?: boolean;
  cwd?: string;
  timeout_seconds?: number;
  enabled?: boolean;
}

export interface MCPServerUpdateRequest {
  type?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  url?: string;
  fallback_sse_url?: string;
  headers?: Record<string, string>;
  oauth_enabled?: boolean;
  oauth_scopes?: string[];
  allow_insecure_http?: boolean;
  allow_private_network?: boolean;
  cwd?: string;
  timeout_seconds?: number;
  enabled?: boolean;
}

export interface IntegrationOAuthStart {
  flow_id: string;
  authorization_url: string;
  redirect_uri: string;
  callback_mode: string;
  expires_at_unix: number;
  browser_warning: string;
}

export interface IntegrationOAuthCompleteRequest {
  flow_id: string;
  callback_input?: string;
}

export interface IntegrationOAuthCompleteResult {
  status: string;
  message: string;
  account: AccountInfo | null;
  expires_at: number | null;
  scopes: string[];
}

export interface AuthDraftSyncResult {
  created_drafts: number;
  created_bindings: number;
  updated_defaults: number;
  warnings: string[];
  integrations: WorkspaceIntegrations;
}

export interface WorkspaceFileEntry {
  path: string;
  name: string;
  is_dir: boolean;
  size_bytes: number;
  modified_at: string;
  extension: string;
}

export interface WorkspaceFilePreviewTable {
  columns: string[];
  rows: string[][];
  truncated: boolean;
}

export interface WorkspaceFilePreview {
  path: string;
  name: string;
  extension: string;
  size_bytes: number;
  modified_at: string;
  preview_kind: string;
  language: string;
  text_content: string;
  table: WorkspaceFilePreviewTable | null;
  metadata: Record<string, unknown>;
  truncated: boolean;
  error: string;
}

export interface WorkspaceArtifact extends RunArtifact {
  latest_run_id: string;
  run_ids: string[];
  run_count: number;
}

export interface RunConversationEntry {
  id: string;
  message: string;
  summary: string;
  tags: string;
  timestamp: string;
}

export interface WorkspaceSearchItem {
  kind: string;
  item_id: string;
  title: string;
  subtitle: string;
  snippet: string;
  badges: string[];
  workspace_id: string;
  workspace_display_name: string;
  workspace_path: string;
  conversation_id: string;
  run_id: string;
  approval_item_id: string;
  path: string;
  metadata: Record<string, unknown>;
}

export interface WorkspaceSearchResponse {
  workspace: WorkspaceSummary | null;
  query: string;
  total_results: number;
  workspaces: WorkspaceSearchItem[];
  conversations: WorkspaceSearchItem[];
  runs: WorkspaceSearchItem[];
  approvals: WorkspaceSearchItem[];
  artifacts: WorkspaceSearchItem[];
  files: WorkspaceSearchItem[];
  processes: WorkspaceSearchItem[];
  accounts: WorkspaceSearchItem[];
  mcp_servers: WorkspaceSearchItem[];
  tools: WorkspaceSearchItem[];
}

export interface ApprovalFeedItem {
  id: string;
  kind: "task_approval" | "task_question" | "conversation_approval" | string;
  status: string;
  created_at: string;
  title: string;
  summary: string;
  workspace_id: string;
  workspace_path: string;
  workspace_display_name: string;
  task_id: string;
  run_id: string;
  conversation_id: string;
  subtask_id: string;
  question_id: string;
  approval_id: string;
  tool_name: string;
  risk_level: string;
  request_payload: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface ApprovalReplyRequest {
  decision: string;
  reason?: string;
  response_type?: string;
  selected_option_ids?: string[];
  selected_labels?: string[];
  custom_response?: string;
  source?: string;
  answered_by?: string;
  client_id?: string;
}

export interface NotificationEvent {
  id: string;
  stream_id?: number | null;
  event_type: string;
  created_at: string;
  workspace_id: string;
  workspace_path: string;
  workspace_display_name: string;
  task_id: string;
  conversation_id: string;
  approval_id: string;
  kind: string;
  title: string;
  summary: string;
  payload: Record<string, unknown>;
}

export interface ConversationDetail extends ConversationSummary {
  system_prompt: string;
  session_state: Record<string, unknown>;
  workspace: WorkspaceSummary;
}

export interface ConversationContentBlock {
  type: string;
  text?: string;
  source_path?: string;
  media_type?: string;
  width?: number;
  height?: number;
  size_bytes?: number;
  page_count?: number;
  page_range?: number[];
  extracted_text?: string;
  text_fallback?: string;
  thinking?: string;
  signature?: string;
}

export interface ConversationMessageAttachments {
  workspace_paths?: string[];
  workspace_files?: string[];
  workspace_directories?: string[];
  content_blocks?: ConversationContentBlock[];
}

export interface ConversationMessage {
  id: number;
  session_id: string;
  turn_number: number;
  role: string;
  content: string | null;
  metadata?: Record<string, unknown>;
  tool_calls: Array<Record<string, unknown>>;
  tool_call_id: string | null;
  tool_name: string | null;
  token_count: number;
  created_at: string;
}

export interface RunTimelineEvent {
  id: number;
  task_id: string;
  run_id: string;
  correlation_id: string;
  event_id: string;
  sequence: number;
  timestamp: string;
  event_type: string;
  source_component: string;
  schema_version: number;
  data: Record<string, unknown>;
}

export interface PlanSubtask {
  id: string;
  description: string;
  status: string;
  depends_on: string[];
  phase_id: string;
  summary: string;
  is_critical_path: boolean;
  is_synthesis: boolean;
}

export interface RunDetail extends RunSummary {
  task: Record<string, unknown>;
  task_run: Record<string, unknown>;
  events_count: number;
  workspace: WorkspaceSummary;
  plan_subtasks: PlanSubtask[];
}

export interface RunArtifact {
  path: string;
  category: string;
  source: string;
  sha256: string;
  size_bytes: number;
  exists_on_disk: boolean;
  is_intermediate: boolean;
  created_at: string;
  tool_name: string;
  subtask_ids: string[];
  phase_ids: string[];
  facets: Record<string, unknown>;
}

export interface SettingsEntry {
  path: string;
  section: string;
  field: string;
  description: string;
  supports_runtime: boolean;
  supports_persist: boolean;
  application_class: string;
  requires_restart: boolean;
  exposure_level: "basic" | "advanced";
  configured: unknown;
  configured_display: string;
  runtime_override: unknown;
  runtime_display: string;
  effective: unknown;
  effective_display: string;
  updated_at: string;
  source_path: string;
}

export interface SettingsPayload {
  basic: SettingsEntry[];
  advanced: SettingsEntry[];
  updated_at: string;
}

export interface WorkspaceSettingsPayload {
  workspace: WorkspaceSummary;
  workspace_id: string;
  overrides: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface WorkspaceCreateRequest {
  path: string;
  display_name?: string;
  metadata?: Record<string, unknown>;
}

export interface WorkspacePatchRequest {
  display_name?: string;
  sort_order?: number;
  archived?: boolean;
  last_opened_at?: string;
  metadata?: Record<string, unknown>;
}

export interface ConversationCreateRequest {
  model_name?: string;
  system_prompt?: string;
}

export interface ConversationPatchRequest {
  title?: string;
}

export interface TaskCreateRequest {
  goal: string;
  workspace?: string;
  approval_mode?: string;
  process?: string;
  context?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  auto_subfolder?: boolean;
}

export interface TaskCreateResponse {
  task_id: string;
  status: string;
  message: string;
  run_id: string;
}

export interface RunActionResponse {
  status: string;
  message: string;
  task_status?: string;
  task_id?: string;
}

export interface ConversationActionResponse {
  status: string;
  message: string;
  conversation_id?: string;
  pending_inject_count?: number;
}

export interface ConversationPromptOption {
  id: string;
  label: string;
  description?: string;
}

export interface ConversationPrompt {
  question: string;
  question_type: string;
  options: ConversationPromptOption[];
  allow_custom_response: boolean;
  min_selections: number;
  max_selections: number;
  context_note: string;
  urgency: string;
  default_option_id: string;
  tool_call_id?: string;
}

export interface ConversationApproval {
  approval_id: string;
  conversation_id: string;
  tool_name: string;
  args: Record<string, unknown>;
  risk_info: Record<string, unknown> | null;
  created_at: string;
}

export interface ConversationContextStatus {
  estimated_tokens: number;
  max_tokens: number;
  percent_used: number;
  pressure_state: string;
  compaction_enabled?: boolean;
  compaction_policy_mode?: string;
  compacted: boolean;
  compacted_message_count: number;
  compacted_tool_message_count: number;
  recall_index_used: boolean;
  memory_index_degraded: boolean;
  likely_compaction_next_turn: boolean;
  last_compaction_at: string;
  updated_at: string;
}

export interface ConversationStatus {
  conversation_id: string;
  processing: boolean;
  stop_requested: boolean;
  pending_inject_count: number;
  awaiting_approval: boolean;
  pending_approval: ConversationApproval | null;
  awaiting_user_input: boolean;
  pending_prompt: ConversationPrompt | null;
  context_status?: ConversationContextStatus | null;
}

export interface SettingsPatchRequest {
  values: Record<string, unknown>;
  clear_paths?: string[];
  persist?: boolean;
}

interface DesktopBootstrapResponse {
  base_url: string;
  managed_by_desktop: boolean;
}

export interface DesktopRuntimeMetadata {
  mode: string;
  enabled_extras: string[];
  loom_version: string | null;
  python_version: string | null;
  python_request: string | null;
  uv_version: string | null;
  entry_module: string | null;
  python_home: string | null;
  python_executable: string | null;
  environment_root: string | null;
  site_packages: string | null;
  repo_root: string | null;
}

export interface DesktopSidecarStatus {
  running: boolean;
  managed_by_desktop: boolean;
  base_url: string;
  pid: number | null;
  database_path: string;
  scratch_dir: string;
  workspace_default_path: string;
  log_path: string;
  runtime: DesktopRuntimeMetadata | null;
  runtime_error: string | null;
}

export interface ConversationStreamEvent {
  id: number;
  session_id: string;
  seq: number;
  event_type: string;
  payload: Record<string, unknown>;
  payload_parse_error: boolean;
  created_at: string;
  turn_number?: number;
  _optimistic?: boolean;
  _client_id?: string;
  _delivery_state?: "queued" | "sending" | "accepted" | "failed";
}

export interface RunStreamEvent {
  id?: number;
  event_type: string;
  task_id: string;
  run_id?: string;
  correlation_id?: string;
  event_id?: string;
  sequence?: number;
  timestamp: string;
  source_component?: string;
  schema_version?: number;
  data?: Record<string, unknown>;
  status?: string;
  terminal?: boolean;
  streaming?: boolean;
  [key: string]: unknown;
}

let runtimeBaseUrl = (() => {
  // In dev mode, Vite proxies API requests to loomd so we use same-origin
  // (empty string = relative URLs). In production/Tauri, fall back to explicit URL.
  const raw =
    import.meta.env.VITE_LOOMD_URL
    || (import.meta.env.DEV ? "" : "http://127.0.0.1:9000");
  return raw.replace(/\/+$/, "");
})();

const DEFAULT_REQUEST_TIMEOUT_MS = 15000;
const HEAVY_REQUEST_TIMEOUT_MS = 20000;
const RUN_LAUNCH_REQUEST_TIMEOUT_MS = 60000;
const RUN_TIMELINE_REQUEST_LIMIT = 1000;

type RequestJsonInit = RequestInit & {
  timeoutMs?: number;
};

async function tryInvokeDesktopCommand<T>(command: string): Promise<T | null> {
  try {
    const mod = await import("@tauri-apps/api/core");
    return (await mod.invoke(command)) as T;
  } catch {
    return null;
  }
}

async function requestJson<T>(
  path: string,
  init?: RequestJsonInit,
): Promise<T> {
  const {
    timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
    signal: sourceSignal,
    ...fetchInit
  } = init ?? {};
  const controller = new AbortController();
  const timeoutId = globalThis.setTimeout(() => {
    controller.abort(new Error(`Request timed out after ${timeoutMs}ms`));
  }, timeoutMs);
  const abortFromSource = () => {
    controller.abort(sourceSignal?.reason);
  };

  if (sourceSignal) {
    if (sourceSignal.aborted) {
      abortFromSource();
    } else {
      sourceSignal.addEventListener("abort", abortFromSource, { once: true });
    }
  }

  try {
    const response = await fetch(`${runtimeBaseUrl}${path}`, {
      ...fetchInit,
      signal: controller.signal,
    });
    if (!response.ok) {
      let detail = "";
      try {
        const responseText = await response.text();
        if (responseText) {
          try {
            const parsed = JSON.parse(responseText) as { detail?: unknown; message?: unknown };
            if (typeof parsed.detail === "string" && parsed.detail.trim()) {
              detail = parsed.detail.trim();
            } else if (typeof parsed.message === "string" && parsed.message.trim()) {
              detail = parsed.message.trim();
            } else {
              detail = responseText.trim();
            }
          } catch {
            detail = responseText.trim();
          }
        }
      } catch {
        // Ignore response-body parsing failures and fall back to HTTP status.
      }
      throw new Error(detail || `${response.status} ${response.statusText}`);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (controller.signal.aborted && !sourceSignal?.aborted) {
      throw new Error(`Request timed out after ${timeoutMs}ms`);
    }
    throw error;
  } finally {
    globalThis.clearTimeout(timeoutId);
    sourceSignal?.removeEventListener("abort", abortFromSource);
  }
}

export function getRuntimeBaseUrl(): string {
  return runtimeBaseUrl;
}

export async function bootstrapDesktopRuntime(): Promise<boolean> {
  const payload = await tryInvokeDesktopCommand<DesktopBootstrapResponse>(
    "desktop_bootstrap",
  );
  if (payload?.base_url) {
    runtimeBaseUrl = payload.base_url.replace(/\/+$/, "");
    return Boolean(payload.managed_by_desktop);
  }

  // Sidecar unavailable (browser mode or Tauri sidecar failed).
  // Try to discover a running loomd by probing common ports.
  if (!runtimeBaseUrl || runtimeBaseUrl === "") {
    runtimeBaseUrl = "http://127.0.0.1:9000";
  }
  for (const port of [9000, 9001, 9002, 9003]) {
    const candidate = `http://127.0.0.1:${port}`;
    try {
      const response = await fetch(`${candidate}/runtime`, {
        signal: AbortSignal.timeout(1500),
      });
      if (response.ok) {
        runtimeBaseUrl = candidate;
        return false;
      }
    } catch {
      // Port not available, try next
    }
  }
  return false;
}

export async function fetchDesktopSidecarStatus(): Promise<DesktopSidecarStatus | null> {
  return await tryInvokeDesktopCommand<DesktopSidecarStatus>("desktop_sidecar_status");
}

export function fetchRuntimeStatus(): Promise<RuntimeStatus> {
  return requestJson<RuntimeStatus>("/runtime");
}

export function fetchSetupStatus(): Promise<SetupStatus> {
  return requestJson<SetupStatus>("/setup/status");
}

export function discoverSetupModels(
  payload: SetupDiscoverModelsRequest,
): Promise<SetupDiscoverModelsResponse> {
  return requestJson<SetupDiscoverModelsResponse>("/setup/discover-models", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: {
      "content-type": "application/json",
    },
  });
}

export function completeInitialSetup(
  payload: SetupCompleteRequest,
): Promise<SetupCompleteResponse> {
  return requestJson<SetupCompleteResponse>("/setup/complete", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: {
      "content-type": "application/json",
    },
  });
}

export function fetchActivitySummary(): Promise<ActivitySummary> {
  return requestJson<ActivitySummary>("/activity");
}

export function fetchModels(): Promise<ModelInfo[]> {
  return requestJson<ModelInfo[]>("/models");
}

export function patchModel(
  modelName: string,
  body: ModelPatchRequest,
): Promise<ModelInfo> {
  return requestJson<ModelInfo>(`/models/${encodeURIComponent(modelName)}`, {
    method: "PATCH",
    body: JSON.stringify(body),
    headers: {
      "content-type": "application/json",
    },
  });
}

export function fetchWorkspaces(includeArchived = false): Promise<WorkspaceSummary[]> {
  const suffix = includeArchived ? "?include_archived=true" : "";
  return requestJson<WorkspaceSummary[]>(`/workspaces${suffix}`);
}

export function createWorkspace(
  body: WorkspaceCreateRequest,
): Promise<WorkspaceSummary> {
  return requestJson<WorkspaceSummary>("/workspaces", {
    method: "POST",
    body: JSON.stringify(body),
    headers: {
      "content-type": "application/json",
    },
  });
}

export function patchWorkspace(
  workspaceId: string,
  body: WorkspacePatchRequest,
): Promise<WorkspaceSummary> {
  return requestJson<WorkspaceSummary>(
    `/workspaces/${encodeURIComponent(workspaceId)}`,
    {
      method: "PATCH",
      body: JSON.stringify(body),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function fetchWorkspaceOverview(
  workspaceId: string,
): Promise<WorkspaceOverview> {
  return requestJson<WorkspaceOverview>(
    `/workspaces/${encodeURIComponent(workspaceId)}/overview`,
  );
}

export function fetchWorkspaceInventory(
  workspaceId: string,
): Promise<WorkspaceInventory> {
  return requestJson<WorkspaceInventory>(
    `/workspaces/${encodeURIComponent(workspaceId)}/inventory`,
  );
}

export function fetchWorkspaceIntegrations(
  workspaceId: string,
): Promise<WorkspaceIntegrations> {
  return requestJson<WorkspaceIntegrations>(
    `/workspaces/${encodeURIComponent(workspaceId)}/integrations`,
  );
}

export function createWorkspaceMcpServer(
  workspaceId: string,
  payload: MCPServerCreateRequest,
): Promise<MCPServerManagementInfo> {
  return requestJson<MCPServerManagementInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp`,
    {
      method: "POST",
      body: JSON.stringify(payload),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function updateWorkspaceMcpServer(
  workspaceId: string,
  alias: string,
  payload: MCPServerUpdateRequest,
): Promise<MCPServerManagementInfo> {
  return requestJson<MCPServerManagementInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp/${encodeURIComponent(alias)}`,
    {
      method: "PATCH",
      body: JSON.stringify(payload),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function deleteWorkspaceMcpServer(
  workspaceId: string,
  alias: string,
): Promise<MCPServerActionResult> {
  return requestJson<MCPServerActionResult>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp/${encodeURIComponent(alias)}`,
    {
      method: "DELETE",
    },
  );
}

export function approveWorkspaceMcpServer(
  workspaceId: string,
  alias: string,
): Promise<MCPServerManagementInfo> {
  return requestJson<MCPServerManagementInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp/${encodeURIComponent(alias)}/approve`,
    {
      method: "POST",
    },
  );
}

export function rejectWorkspaceMcpServer(
  workspaceId: string,
  alias: string,
): Promise<MCPServerManagementInfo> {
  return requestJson<MCPServerManagementInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp/${encodeURIComponent(alias)}/reject`,
    {
      method: "POST",
    },
  );
}

export function testWorkspaceMcpServer(
  workspaceId: string,
  alias: string,
): Promise<MCPServerActionResult> {
  return requestJson<MCPServerActionResult>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp/${encodeURIComponent(alias)}/test`,
    {
      method: "POST",
    },
  );
}

export function reconnectWorkspaceMcpServer(
  workspaceId: string,
  alias: string,
): Promise<MCPServerActionResult> {
  return requestJson<MCPServerActionResult>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp/${encodeURIComponent(alias)}/reconnect`,
    {
      method: "POST",
    },
  );
}

export function setWorkspaceMcpServerEnabled(
  workspaceId: string,
  alias: string,
  enabled: boolean,
): Promise<MCPServerManagementInfo> {
  return requestJson<MCPServerManagementInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp/${encodeURIComponent(alias)}/${enabled ? "enable" : "disable"}`,
    {
      method: "POST",
    },
  );
}

export function selectWorkspaceMcpAccount(
  workspaceId: string,
  alias: string,
  profileId: string,
): Promise<MCPServerActionResult> {
  return requestJson<MCPServerActionResult>(
    `/workspaces/${encodeURIComponent(workspaceId)}/mcp/${encodeURIComponent(alias)}/accounts/${encodeURIComponent(profileId)}/select`,
    {
      method: "POST",
    },
  );
}

export function syncWorkspaceAuthDrafts(
  workspaceId: string,
): Promise<AuthDraftSyncResult> {
  return requestJson<AuthDraftSyncResult>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts/sync-drafts`,
    {
      method: "POST",
    },
  );
}

export function createWorkspaceAuthAccount(
  workspaceId: string,
  payload: AccountCreateRequest,
): Promise<AccountInfo> {
  return requestJson<AccountInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts`,
    {
      method: "POST",
      body: JSON.stringify(payload),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function updateWorkspaceAuthAccount(
  workspaceId: string,
  profileId: string,
  payload: AccountUpdateRequest,
): Promise<AccountInfo> {
  return requestJson<AccountInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts/${encodeURIComponent(profileId)}`,
    {
      method: "PATCH",
      body: JSON.stringify(payload),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function archiveWorkspaceAuthAccount(
  workspaceId: string,
  profileId: string,
): Promise<AccountInfo> {
  return requestJson<AccountInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts/${encodeURIComponent(profileId)}/archive`,
    {
      method: "POST",
    },
  );
}

export function restoreWorkspaceAuthAccount(
  workspaceId: string,
  profileId: string,
): Promise<AccountInfo> {
  return requestJson<AccountInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts/${encodeURIComponent(profileId)}/restore`,
    {
      method: "POST",
    },
  );
}

export function startWorkspaceAuthAccountLogin(
  workspaceId: string,
  profileId: string,
): Promise<IntegrationOAuthStart> {
  return requestJson<IntegrationOAuthStart>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts/${encodeURIComponent(profileId)}/login/start`,
    {
      method: "POST",
    },
  );
}

export function completeWorkspaceAuthAccountLogin(
  workspaceId: string,
  profileId: string,
  body: IntegrationOAuthCompleteRequest,
): Promise<IntegrationOAuthCompleteResult> {
  return requestJson<IntegrationOAuthCompleteResult>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts/${encodeURIComponent(profileId)}/login/complete`,
    {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function refreshWorkspaceAuthAccount(
  workspaceId: string,
  profileId: string,
): Promise<AccountInfo> {
  return requestJson<AccountInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts/${encodeURIComponent(profileId)}/refresh`,
    {
      method: "POST",
    },
  );
}

export function logoutWorkspaceAuthAccount(
  workspaceId: string,
  profileId: string,
): Promise<AccountInfo> {
  return requestJson<AccountInfo>(
    `/workspaces/${encodeURIComponent(workspaceId)}/auth/accounts/${encodeURIComponent(profileId)}/logout`,
    {
      method: "POST",
    },
  );
}

export function fetchWorkspaceArtifacts(
  workspaceId: string,
): Promise<WorkspaceArtifact[]> {
  return requestJson<WorkspaceArtifact[]>(
    `/workspaces/${encodeURIComponent(workspaceId)}/artifacts`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function fetchWorkspaceFiles(
  workspaceId: string,
  directory = "",
): Promise<WorkspaceFileEntry[]> {
  const params = new URLSearchParams();
  if (directory.trim()) {
    params.set("directory", directory);
  }
  const suffix = params.size > 0 ? `?${params.toString()}` : "";
  return requestJson<WorkspaceFileEntry[]>(
    `/workspaces/${encodeURIComponent(workspaceId)}/files${suffix}`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function fetchWorkspaceFilePreview(
  workspaceId: string,
  path: string,
): Promise<WorkspaceFilePreview> {
  const params = new URLSearchParams({ path });
  return requestJson<WorkspaceFilePreview>(
    `/workspaces/${encodeURIComponent(workspaceId)}/files/preview?${params.toString()}`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function fetchWorkspaceSearch(
  workspaceId: string,
  query: string,
  limitPerGroup = 5,
): Promise<WorkspaceSearchResponse> {
  const params = new URLSearchParams({
    q: query,
    limit_per_group: String(limitPerGroup),
  });
  return requestJson<WorkspaceSearchResponse>(
    `/workspaces/${encodeURIComponent(workspaceId)}/search?${params.toString()}`,
  );
}

export function fetchWorkspacePathSuggestions(
  workspaceId: string,
  query: string,
  limit = 20,
): Promise<WorkspaceFileEntry[]> {
  const params = new URLSearchParams({
    q: query,
    limit: String(limit),
  });
  return requestJson<WorkspaceFileEntry[]>(
    `/workspaces/${encodeURIComponent(workspaceId)}/paths/search?${params.toString()}`,
  );
}

export function fetchGlobalSearch(
  query: string,
  limitPerGroup = 5,
): Promise<WorkspaceSearchResponse> {
  const params = new URLSearchParams({
    q: query,
    limit_per_group: String(limitPerGroup),
  });
  return requestJson<WorkspaceSearchResponse>(
    `/search?${params.toString()}`,
  );
}

export function fetchApprovals(workspaceId?: string): Promise<ApprovalFeedItem[]> {
  const suffix = workspaceId
    ? `?workspace_id=${encodeURIComponent(workspaceId)}`
    : "";
  return requestJson<ApprovalFeedItem[]>(`/approvals${suffix}`);
}

export function replyApproval(
  approvalItemId: string,
  body: ApprovalReplyRequest,
): Promise<Record<string, unknown>> {
  return requestJson<Record<string, unknown>>(
    `/approvals/${encodeURIComponent(approvalItemId)}/reply`,
    {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function fetchConversationDetail(
  conversationId: string,
): Promise<ConversationDetail> {
  return requestJson<ConversationDetail>(
    `/conversations/${encodeURIComponent(conversationId)}`,
  );
}

export function deleteConversation(
  conversationId: string,
): Promise<{ status: string; message: string }> {
  return requestJson<{ status: string; message: string }>(
    `/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" },
  );
}

export function patchConversation(
  conversationId: string,
  body: ConversationPatchRequest,
): Promise<ConversationDetail> {
  return requestJson<ConversationDetail>(
    `/conversations/${encodeURIComponent(conversationId)}`,
    {
      method: "PATCH",
      body: JSON.stringify(body),
      headers: { "content-type": "application/json" },
    },
  );
}

export function createConversation(
  workspaceId: string,
  body: ConversationCreateRequest,
): Promise<ConversationSummary> {
  return requestJson<ConversationSummary>(
    `/workspaces/${encodeURIComponent(workspaceId)}/conversations`,
    {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function fetchConversationMessages(
  conversationId: string,
  options?: { offset?: number; limit?: number; beforeTurn?: number; latest?: boolean },
): Promise<ConversationMessage[]> {
  const params = new URLSearchParams();
  if (options?.offset != null) params.set("offset", String(options.offset));
  if (options?.limit != null) params.set("limit", String(options.limit));
  if (options?.beforeTurn != null) params.set("before_turn", String(options.beforeTurn));
  if (options?.latest) params.set("latest", "true");
  const qs = params.toString();
  return requestJson<ConversationMessage[]>(
    `/conversations/${encodeURIComponent(conversationId)}/messages${qs ? `?${qs}` : ""}`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function fetchConversationEvents(
  conversationId: string,
  options?: { beforeSeq?: number; beforeTurn?: number; afterSeq?: number; limit?: number },
): Promise<ConversationStreamEvent[]> {
  const params = new URLSearchParams();
  if (options?.beforeSeq != null) params.set("before_seq", String(options.beforeSeq));
  if (options?.beforeTurn != null) params.set("before_turn", String(options.beforeTurn));
  if (options?.afterSeq != null) params.set("after_seq", String(options.afterSeq));
  if (options?.limit != null) params.set("limit", String(options.limit));
  const qs = params.toString();
  return requestJson<ConversationStreamEvent[]>(
    `/conversations/${encodeURIComponent(conversationId)}/events${qs ? `?${qs}` : ""}`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function fetchConversationStatus(
  conversationId: string,
): Promise<ConversationStatus> {
  return requestJson<ConversationStatus>(
    `/conversations/${encodeURIComponent(conversationId)}/status`,
  );
}

export function sendConversationMessage(
  conversationId: string,
  message: string,
  role = "user",
  attachments?: ConversationMessageAttachments,
): Promise<ConversationActionResponse> {
  const body = {
    message,
    role,
    workspace_paths: attachments?.workspace_paths || [],
    workspace_files: attachments?.workspace_files || [],
    workspace_directories: attachments?.workspace_directories || [],
    content_blocks: attachments?.content_blocks || [],
  };
  return requestJson<ConversationActionResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      body: JSON.stringify(body),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function stopConversationTurn(
  conversationId: string,
): Promise<ConversationActionResponse> {
  return requestJson<ConversationActionResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/stop`,
    {
      method: "POST",
    },
  );
}

export function injectConversationInstruction(
  conversationId: string,
  instruction: string,
): Promise<ConversationActionResponse> {
  return requestJson<ConversationActionResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/inject`,
    {
      method: "POST",
      body: JSON.stringify({ instruction }),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function resolveConversationApproval(
  conversationId: string,
  approvalId: string,
  decision: "approve" | "approve_all" | "deny",
): Promise<ConversationActionResponse> {
  return requestJson<ConversationActionResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/approvals/${encodeURIComponent(approvalId)}`,
    {
      method: "POST",
      body: JSON.stringify({ decision }),
      headers: {
        "content-type": "application/json",
      },
    },
  );
}

export function fetchRunDetail(runId: string): Promise<RunDetail> {
  return requestJson<RunDetail>(
    `/runs/${encodeURIComponent(runId)}`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function fetchRunTimeline(
  runId: string,
  options?: { limit?: number; includeNoise?: boolean },
): Promise<RunTimelineEvent[]> {
  const params = new URLSearchParams({
    limit: String(options?.limit ?? RUN_TIMELINE_REQUEST_LIMIT),
  });
  if (options?.includeNoise === false) {
    params.set("include_noise", "false");
  }
  return requestJson<RunTimelineEvent[]>(
    `/runs/${encodeURIComponent(runId)}/timeline?${params.toString()}`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function fetchRunArtifacts(runId: string): Promise<RunArtifact[]> {
  return requestJson<RunArtifact[]>(
    `/runs/${encodeURIComponent(runId)}/artifacts`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function createTask(body: TaskCreateRequest): Promise<TaskCreateResponse> {
  return requestJson<TaskCreateResponse>("/tasks", {
    method: "POST",
    body: JSON.stringify(body),
    headers: {
      "content-type": "application/json",
    },
    timeoutMs: RUN_LAUNCH_REQUEST_TIMEOUT_MS,
  });
}

export function pauseRun(runId: string): Promise<RunActionResponse> {
  return requestJson<RunActionResponse>(`/runs/${encodeURIComponent(runId)}/pause`, {
    method: "POST",
  });
}

export function resumeRun(runId: string): Promise<RunActionResponse> {
  return requestJson<RunActionResponse>(`/runs/${encodeURIComponent(runId)}/resume`, {
    method: "POST",
  });
}

export function cancelRun(runId: string): Promise<RunActionResponse> {
  return requestJson<RunActionResponse>(`/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
  });
}

export function deleteRun(runId: string): Promise<RunActionResponse> {
  return requestJson<RunActionResponse>(`/runs/${encodeURIComponent(runId)}`, {
    method: "DELETE",
  });
}

export function restartRun(runId: string): Promise<TaskCreateResponse> {
  return requestJson<TaskCreateResponse>(`/runs/${encodeURIComponent(runId)}/restart`, {
    method: "POST",
    timeoutMs: RUN_LAUNCH_REQUEST_TIMEOUT_MS,
  });
}

export function fetchRunConversationHistory(
  runId: string,
): Promise<RunConversationEntry[]> {
  return requestJson<RunConversationEntry[]>(
    `/tasks/${encodeURIComponent(runId)}/conversation`,
    { timeoutMs: HEAVY_REQUEST_TIMEOUT_MS },
  );
}

export function sendRunMessage(
  runId: string,
  message: string,
  role = "user",
): Promise<RunActionResponse> {
  return requestJson<RunActionResponse>(`/runs/${encodeURIComponent(runId)}/message`, {
    method: "POST",
    body: JSON.stringify({ message, role }),
    headers: {
      "content-type": "application/json",
    },
  });
}

function parseStreamPayload<T>(event: MessageEvent<string>): T | null {
  try {
    return JSON.parse(event.data) as T;
  } catch {
    return null;
  }
}

export function subscribeConversationStream(
  conversationId: string,
  onEvent: (event: ConversationStreamEvent) => void,
  onError?: () => void,
  options?: { afterSeq?: number },
): () => void {
  const params = new URLSearchParams();
  if (options?.afterSeq != null && options.afterSeq > 0) {
    params.set("after_seq", String(options.afterSeq));
  }
  const qs = params.toString();
  const source = new EventSource(
    `${runtimeBaseUrl}/conversations/${encodeURIComponent(conversationId)}/stream${qs ? `?${qs}` : ""}`,
  );
  const handleEvent = (event: MessageEvent<string>) => {
    const payload = parseStreamPayload<ConversationStreamEvent>(event);
    if (payload) {
      onEvent(payload);
    }
  };
  source.addEventListener("chat_event", handleEvent as EventListener);
  source.onerror = () => {
    onError?.();
  };
  return () => {
    source.removeEventListener("chat_event", handleEvent as EventListener);
    source.close();
  };
}

export function subscribeRunStream(
  runId: string,
  onEvent: (event: RunStreamEvent) => void,
  onError?: () => void,
  options?: { afterSequence?: number; includeNoise?: boolean },
): () => void {
  const params = new URLSearchParams();
  if (options?.afterSequence != null && options.afterSequence > 0) {
    params.set("after_sequence", String(options.afterSequence));
  }
  if (options?.includeNoise === false) {
    params.set("include_noise", "false");
  }
  const qs = params.toString();
  const source = new EventSource(
    `${runtimeBaseUrl}/runs/${encodeURIComponent(runId)}/stream${qs ? `?${qs}` : ""}`,
  );
  const handleEvent = (event: MessageEvent<string>) => {
    const payload = parseStreamPayload<RunStreamEvent>(event);
    if (payload) {
      onEvent(payload);
      if (payload.terminal || payload.streaming === false) {
        source.close();
      }
    }
  };
  source.addEventListener("run_event", handleEvent as EventListener);
  source.onerror = () => {
    onError?.();
  };
  return () => {
    source.removeEventListener("run_event", handleEvent as EventListener);
    source.close();
  };
}

export function subscribeNotificationsStream(
  workspaceId: string,
  onEvent: (event: NotificationEvent) => void,
  onError?: () => void,
  options?: { afterId?: number },
): () => void {
  const params = new URLSearchParams();
  params.set("workspace_id", workspaceId);
  if (options?.afterId != null && options.afterId > 0) {
    params.set("after_id", String(options.afterId));
  }
  const source = new EventSource(
    `${runtimeBaseUrl}/notifications/stream?${params.toString()}`,
  );
  const handleEvent = (event: MessageEvent<string>) => {
    const payload = parseStreamPayload<NotificationEvent>(event);
    if (payload) {
      onEvent(payload);
    }
  };
  source.addEventListener("notification", handleEvent as EventListener);
  source.onerror = () => {
    onError?.();
  };
  return () => {
    source.removeEventListener("notification", handleEvent as EventListener);
    source.close();
  };
}

export function fetchSettings(): Promise<SettingsPayload> {
  return requestJson<SettingsPayload>("/settings");
}

export function fetchWorkspaceSettings(
  workspaceId: string,
): Promise<WorkspaceSettingsPayload> {
  return requestJson<WorkspaceSettingsPayload>(
    `/workspaces/${encodeURIComponent(workspaceId)}/settings`,
  );
}

export function patchSettings(
  body: SettingsPatchRequest,
): Promise<SettingsPayload> {
  return requestJson<SettingsPayload>("/settings", {
    method: "PATCH",
    body: JSON.stringify(body),
    headers: {
      "content-type": "application/json",
    },
  });
}

export async function createWorkspaceDirectory(path: string): Promise<string> {
  const mod = await import("@tauri-apps/api/core");
  return await mod.invoke<string>("desktop_create_workspace_directory", {
    path,
  });
}

export async function createWorkspaceFile(
  workspacePath: string,
  relativePath: string,
  content: string,
  overwrite = false,
): Promise<string> {
  const mod = await import("@tauri-apps/api/core");
  return await mod.invoke<string>("desktop_create_workspace_file", {
    workspacePath,
    relativePath,
    content,
    overwrite,
  });
}

export async function importWorkspaceFiles(
  workspacePath: string,
  files: Array<{ relativePath: string; bytes: number[]; overwrite?: boolean }>,
): Promise<string[]> {
  const mod = await import("@tauri-apps/api/core");
  return await mod.invoke<string[]>("desktop_import_workspace_files", {
    workspacePath,
    files,
  });
}

export async function openWorkspaceFile(
  workspacePath: string,
  relativePath: string,
): Promise<void> {
  const mod = await import("@tauri-apps/api/core");
  await mod.invoke("desktop_open_workspace_file", {
    workspacePath,
    relativePath,
  });
}

export async function revealWorkspaceFile(
  workspacePath: string,
  relativePath: string,
): Promise<void> {
  const mod = await import("@tauri-apps/api/core");
  await mod.invoke("desktop_reveal_workspace_file", {
    workspacePath,
    relativePath,
  });
}

export async function writeScratchFile(
  scratchDir: string,
  suggestedName: string,
  bytes: Uint8Array,
): Promise<string> {
  const mod = await import("@tauri-apps/api/core");
  return await mod.invoke<string>("desktop_write_scratch_file", {
    scratchDir,
    suggestedName,
    bytes: Array.from(bytes),
  });
}
