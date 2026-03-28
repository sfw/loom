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

export interface ModelInfo {
  name: string;
  model: string;
  model_id: string;
  tier: number;
  roles: string[];
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

export interface ConversationMessage {
  id: number;
  session_id: string;
  turn_number: number;
  role: string;
  content: string | null;
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

export interface ConversationStatus {
  conversation_id: string;
  processing: boolean;
  stop_requested: boolean;
  pending_inject_count: number;
  awaiting_approval: boolean;
  pending_approval: ConversationApproval | null;
  awaiting_user_input: boolean;
  pending_prompt: ConversationPrompt | null;
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
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${runtimeBaseUrl}${path}`, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
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

export function fetchRuntimeStatus(): Promise<RuntimeStatus> {
  return requestJson<RuntimeStatus>("/runtime");
}

export function fetchModels(): Promise<ModelInfo[]> {
  return requestJson<ModelInfo[]>("/models");
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

export function fetchWorkspaceArtifacts(
  workspaceId: string,
): Promise<WorkspaceArtifact[]> {
  return requestJson<WorkspaceArtifact[]>(
    `/workspaces/${encodeURIComponent(workspaceId)}/artifacts`,
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
  );
}

export function fetchWorkspaceFilePreview(
  workspaceId: string,
  path: string,
): Promise<WorkspaceFilePreview> {
  const params = new URLSearchParams({ path });
  return requestJson<WorkspaceFilePreview>(
    `/workspaces/${encodeURIComponent(workspaceId)}/files/preview?${params.toString()}`,
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
  );
}

export function fetchConversationEvents(
  conversationId: string,
  options?: { beforeSeq?: number; afterSeq?: number; limit?: number },
): Promise<ConversationStreamEvent[]> {
  const params = new URLSearchParams();
  if (options?.beforeSeq != null) params.set("before_seq", String(options.beforeSeq));
  if (options?.afterSeq != null) params.set("after_seq", String(options.afterSeq));
  if (options?.limit != null) params.set("limit", String(options.limit));
  const qs = params.toString();
  return requestJson<ConversationStreamEvent[]>(
    `/conversations/${encodeURIComponent(conversationId)}/events${qs ? `?${qs}` : ""}`,
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
): Promise<ConversationActionResponse> {
  return requestJson<ConversationActionResponse>(
    `/conversations/${encodeURIComponent(conversationId)}/messages`,
    {
      method: "POST",
      body: JSON.stringify({ message, role }),
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
  return requestJson<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
}

export function fetchRunTimeline(runId: string): Promise<RunTimelineEvent[]> {
  return requestJson<RunTimelineEvent[]>(
    `/runs/${encodeURIComponent(runId)}/timeline`,
  );
}

export function fetchRunArtifacts(runId: string): Promise<RunArtifact[]> {
  return requestJson<RunArtifact[]>(
    `/runs/${encodeURIComponent(runId)}/artifacts`,
  );
}

export function createTask(body: TaskCreateRequest): Promise<TaskCreateResponse> {
  return requestJson<TaskCreateResponse>("/tasks", {
    method: "POST",
    body: JSON.stringify(body),
    headers: {
      "content-type": "application/json",
    },
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
  });
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
  options?: { afterId?: number },
): () => void {
  const params = new URLSearchParams();
  if (options?.afterId != null && options.afterId > 0) {
    params.set("after_id", String(options.afterId));
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
