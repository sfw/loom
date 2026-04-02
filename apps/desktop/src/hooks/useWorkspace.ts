import {
  startTransition,
  type FormEvent,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

import {
  createWorkspace,
  createWorkspaceDirectory,
  fetchApprovals,
  fetchWorkspaceArtifacts,
  fetchWorkspaceInventory,
  fetchWorkspaceOverview,
  fetchWorkspaceSearch,
  fetchWorkspaceSettings,
  fetchWorkspaces,
  patchWorkspace,
  subscribeNotificationsStream,
  type ApprovalFeedItem,
  type ConversationDetail,
  type ConversationSummary,
  type NotificationEvent,
  type RunDetail,
  type RunSummary,
  type RuntimeStatus,
  type WorkspaceArtifact,
  type WorkspaceInventory,
  type WorkspaceOverview,
  type WorkspaceSearchResponse,
  type WorkspaceSettingsPayload,
  type WorkspaceSummary,
} from "../api";
import { matchesWorkspaceSearch } from "../history";
import { isRunActiveStatus, normalizeRunStatus } from "../runStatus";
import {
  defaultWorkspaceName,
  isTransientRequestError,
  joinWorkspacePath,
  mergeWorkspaceSummary,
  parseWorkspaceTags,
  workspaceNoteFromMetadata,
  workspaceTagsFromMetadata,
  type ViewTab,
} from "../utils";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface WorkspaceState {
  selectedWorkspaceId: string;
  showArchivedWorkspaces: boolean;
  workspaceNameDraft: string;
  workspaceTagsDraft: string;
  workspaceNoteDraft: string;
  workspaceFileTreeMode: "all" | "active" | "recent";
  workspaceImportFolderDraft: string;
  importPath: string;
  importDisplayName: string;
  createParentPath: string;
  createFolderName: string;
  createDisplayName: string;
  importingWorkspace: boolean;
  creatingWorkspace: boolean;
  savingWorkspaceMeta: boolean;
  workspaceSearchQuery: string;
  overview: WorkspaceOverview | null;
  inventory: WorkspaceInventory | null;
  loadingOverview: boolean;
  approvalInbox: ApprovalFeedItem[];
  notifications: NotificationEvent[];
  workspaceSettings: WorkspaceSettingsPayload | null;
  workspaceArtifacts: WorkspaceArtifact[];
  workspaceSearchResults: WorkspaceSearchResponse | null;
  searchingWorkspace: boolean;

  // Computed
  selectedWorkspaceSummary: WorkspaceSummary | null;
  selectedWorkspaceArchived: boolean;
  selectedWorkspacePinned: boolean;
  selectedWorkspaceTags: string[];
  selectedWorkspaceNote: string;
  noWorkspacesRegistered: boolean;
  selectedWorkspaceIsEmpty: boolean;
  workspaceConversationRows: Array<{ id: string; workspace_id: string; workspace_path: string; model_name: string; title: string; turn_count: number; total_tokens: number; last_active_at: string; started_at: string; is_active: boolean; linked_run_ids: string[] }>;
  workspaceRunRows: Array<{ id: string; workspace_id: string; workspace_path: string; goal: string; status: string; created_at: string; updated_at: string; execution_run_id: string; process_name: string; linked_conversation_ids: string[]; changed_files_count: number }>;
  filteredConversations: Array<{ id: string; title: string; model_name: string; last_active_at: string; started_at: string; linked_run_ids: string[] }>;
  filteredRuns: Array<{ id: string; goal: string; status: string; created_at: string; updated_at: string; process_name: string; linked_conversation_ids: string[] }>;
  filteredApprovalItems: ApprovalFeedItem[];
  filteredProcesses: Array<{ name: string; version: string; description: string; author: string; path: string }>;
  filteredMcpServers: Array<{ alias: string; type: string; enabled: boolean; source: string; command: string; url: string; cwd: string; timeout_seconds: number; oauth_enabled: boolean }>;
  filteredTools: Array<{ name: string; description: string; auth_mode: string; auth_required: boolean; execution_surfaces: string[] }>;
  filteredWorkspaceArtifacts: WorkspaceArtifact[];
  recentWorkspaceArtifacts: WorkspaceArtifact[];
  recentNotifications: NotificationEvent[];
  searchGroups: Array<{ label: string; rows: Array<{ kind: string; item_id: string; title: string; subtitle: string; snippet: string; badges: string[]; conversation_id: string; run_id: string; approval_item_id: string; path: string; metadata: Record<string, unknown> }> }>;

  // Refs
  searchInputRef: React.RefObject<HTMLInputElement | null>;
  inboxSectionRef: React.RefObject<HTMLElement | null>;
}

export interface WorkspaceActions {
  setSelectedWorkspaceId: React.Dispatch<React.SetStateAction<string>>;
  setShowArchivedWorkspaces: React.Dispatch<React.SetStateAction<boolean>>;
  setWorkspaceNameDraft: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceTagsDraft: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceNoteDraft: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceFileTreeMode: React.Dispatch<React.SetStateAction<"all" | "active" | "recent">>;
  setWorkspaceImportFolderDraft: React.Dispatch<React.SetStateAction<string>>;
  setImportPath: React.Dispatch<React.SetStateAction<string>>;
  setImportDisplayName: React.Dispatch<React.SetStateAction<string>>;
  setCreateParentPath: React.Dispatch<React.SetStateAction<string>>;
  setCreateFolderName: React.Dispatch<React.SetStateAction<string>>;
  setCreateDisplayName: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceSearchQuery: React.Dispatch<React.SetStateAction<string>>;
  handleImportWorkspace: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleCreateWorkspace: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleSaveWorkspaceDetails: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleArchiveWorkspace: (nextArchived: boolean) => Promise<void>;
  handlePinWorkspace: (nextPinned: boolean) => Promise<void>;
  handlePrefillStarterWorkspace: () => void;
  focusSearch: () => void;
  refreshWorkspaceList: (preferredWorkspaceId?: string) => Promise<void>;
  refreshWorkspaceSurface: (workspaceId: string) => Promise<void>;
  refreshWorkspaceArtifacts: (
    workspaceId: string,
    options?: {
      force?: boolean;
    },
  ) => Promise<void>;
  refreshApprovalInbox: (workspaceId: string) => Promise<void>;
  syncConversationSummary: (
    detail: ConversationSummary | ConversationDetail,
    options?: {
      incrementCount?: boolean;
      processing?: boolean;
      workspaceId?: string;
    },
  ) => void;
  setConversationProcessing: (
    conversationId: string,
    processing: boolean,
    options?: {
      lastActiveAt?: string;
      workspaceId?: string;
    },
  ) => void;
  removeConversationSummary: (conversationId: string, workspaceId?: string) => void;
  applyApprovalItem: (
    item: ApprovalFeedItem,
    options?: {
      incrementCount?: boolean;
    },
  ) => void;
  removeApprovalItem: (itemId: string, workspaceId?: string) => void;
  syncRunDetail: (detail: RunSummary | RunDetail) => void;
  removeRunSummary: (runId: string, workspaceId?: string) => void;
}

type WorkspaceConversationRow = WorkspaceState["workspaceConversationRows"][number];
type WorkspaceRunRow = WorkspaceState["workspaceRunRows"][number];

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useWorkspace(deps: {
  selectedWorkspaceId: string;
  selectedConversationId: string;
  selectedRunId: string;
  connectionState?: "connecting" | "connected" | "failed";
  setSelectedWorkspaceId: React.Dispatch<React.SetStateAction<string>>;
  showArchivedWorkspaces: boolean;
  setShowArchivedWorkspaces: React.Dispatch<React.SetStateAction<boolean>>;
  createParentPath: string;
  setCreateParentPath: React.Dispatch<React.SetStateAction<string>>;
  workspaces: WorkspaceSummary[];
  setWorkspaces: React.Dispatch<React.SetStateAction<WorkspaceSummary[]>>;
  runtime: RuntimeStatus | null;
  setError: React.Dispatch<React.SetStateAction<string>>;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
  activeTab: ViewTab;
  setActiveTab: React.Dispatch<React.SetStateAction<ViewTab>>;
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
  setSelectedRunId: React.Dispatch<React.SetStateAction<string>>;
}): WorkspaceState & WorkspaceActions {
  const {
    selectedWorkspaceId,
    selectedConversationId,
    selectedRunId,
    connectionState = "connected",
    setSelectedWorkspaceId,
    showArchivedWorkspaces,
    setShowArchivedWorkspaces,
    createParentPath,
    setCreateParentPath,
    workspaces,
    setWorkspaces,
    runtime,
    setError,
    setNotice,
    activeTab,
    setSelectedConversationId,
    setSelectedRunId,
  } = deps;

  // State (items NOT hoisted to orchestrator)
  const [workspaceNameDraft, setWorkspaceNameDraft] = useState("");
  const [workspaceTagsDraft, setWorkspaceTagsDraft] = useState("");
  const [workspaceNoteDraft, setWorkspaceNoteDraft] = useState("");
  const [workspaceFileTreeMode, setWorkspaceFileTreeMode] = useState<"all" | "active" | "recent">("all");
  const [workspaceImportFolderDraft, setWorkspaceImportFolderDraft] = useState("");
  const [importPath, setImportPath] = useState("");
  const [importDisplayName, setImportDisplayName] = useState("");
  const [createFolderName, setCreateFolderName] = useState("");
  const [createDisplayName, setCreateDisplayName] = useState("");
  const [importingWorkspace, setImportingWorkspace] = useState(false);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [savingWorkspaceMeta, setSavingWorkspaceMeta] = useState(false);
  const [workspaceSearchQuery, setWorkspaceSearchQuery] = useState("");

  const [overview, setOverview] = useState<WorkspaceOverview | null>(null);
  const [inventory, setInventory] = useState<WorkspaceInventory | null>(null);
  const [loadingOverview, setLoadingOverview] = useState(false);
  const [approvalInbox, setApprovalInbox] = useState<ApprovalFeedItem[]>([]);
  const [notifications, setNotifications] = useState<NotificationEvent[]>([]);
  const [workspaceSettings, setWorkspaceSettings] = useState<WorkspaceSettingsPayload | null>(null);
  const [workspaceArtifacts, setWorkspaceArtifacts] = useState<WorkspaceArtifact[]>([]);
  const [workspaceSearchResults, setWorkspaceSearchResults] = useState<WorkspaceSearchResponse | null>(null);
  const [searchingWorkspace, setSearchingWorkspace] = useState(false);

  // Refs
  const approvalInboxRef = useRef<ApprovalFeedItem[]>([]);
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const inboxSectionRef = useRef<HTMLElement | null>(null);
  const workspaceSearchTimerRef = useRef<number | null>(null);
  const notificationRefreshTimerRef = useRef<number | null>(null);
  const notificationRepairRef = useRef<{
    approvals: boolean;
    overview: boolean;
    workspaceId: string;
  }>({
    approvals: false,
    overview: false,
    workspaceId: "",
  });
  const lastSeenNotificationStreamIdRef = useRef(0);
  const previousWorkspaceIdRef = useRef("");
  const previousWorkspaceDraftSourceRef = useRef<{
    id: string;
    name: string;
    note: string;
    tags: string;
  }>({
    id: "",
    name: "",
    note: "",
    tags: "",
  });
  const selectedWorkspaceIdRef = useRef(selectedWorkspaceId);
  const previousSelectedConversationIdRef = useRef(selectedConversationId);
  const previousSelectedRunIdRef = useRef(selectedRunId);
  const overviewRef = useRef<WorkspaceOverview | null>(null);
  const loadedWorkspaceArtifactsIdRef = useRef("");
  const loadedWorkspaceInventoryIdRef = useRef("");
  const loadedWorkspaceSettingsIdRef = useRef("");
  const loadingWorkspaceArtifactsIdRef = useRef("");
  const loadingWorkspaceInventoryIdRef = useRef("");
  const loadingWorkspaceSettingsIdRef = useRef("");

  // ---------------------------------------------------------------------------
  // useEffectEvent handlers
  // ---------------------------------------------------------------------------

  const refreshWorkspaceList = useEffectEvent(async (preferredWorkspaceId = "") => {
    const workspaceRows = await fetchWorkspaces(showArchivedWorkspaces);
    setWorkspaces(workspaceRows);
    const fallbackWorkspaceId = workspaceRows[0]?.id || "";
    const nextWorkspaceId =
      (preferredWorkspaceId &&
      workspaceRows.some((workspace) => workspace.id === preferredWorkspaceId))
        ? preferredWorkspaceId
        : (selectedWorkspaceId &&
          workspaceRows.some((workspace) => workspace.id === selectedWorkspaceId))
          ? selectedWorkspaceId
          : fallbackWorkspaceId;
    startTransition(() => {
      setSelectedWorkspaceId(nextWorkspaceId);
    });
  });

  const applyWorkspaceOverview = useEffectEvent((workspaceId: string, overviewPayload: WorkspaceOverview) => {
    const isCurrentWorkspace = selectedWorkspaceIdRef.current === workspaceId;
    if (!isCurrentWorkspace) {
      setWorkspaces((current) => mergeWorkspaceSummary(current, overviewPayload.workspace));
      return;
    }

    setOverview(overviewPayload);
    setWorkspaces((current) => mergeWorkspaceSummary(current, overviewPayload.workspace));

    const firstConversationId = overviewPayload.recent_conversations[0]?.id || "";
    const firstRunId = overviewPayload.recent_runs[0]?.id || "";

    startTransition(() => {
      setSelectedConversationId((current) => {
        if (!current) return firstConversationId;
        return current;
      });
      setSelectedRunId((current) => {
        if (!current) return firstRunId;
        return current;
      });
    });
  });

  function latestTimestamp(...values: Array<string | null | undefined>): string {
    let latest = "";
    for (const value of values) {
      const nextValue = String(value || "").trim();
      if (nextValue && nextValue > latest) {
        latest = nextValue;
      }
    }
    return latest;
  }

  function sortConversationRows(rows: WorkspaceConversationRow[]): WorkspaceConversationRow[] {
    return rows
      .slice()
      .sort((left, right) =>
        latestTimestamp(right.last_active_at, right.started_at)
          .localeCompare(latestTimestamp(left.last_active_at, left.started_at)),
      );
  }

  function sortRunRows(rows: WorkspaceRunRow[]): WorkspaceRunRow[] {
    return rows
      .slice()
      .sort((left, right) =>
        latestTimestamp(right.updated_at, right.created_at)
          .localeCompare(latestTimestamp(left.updated_at, left.created_at)),
      );
  }

  function sortWorkspaceSummaries(rows: WorkspaceSummary[]): WorkspaceSummary[] {
    return rows
      .slice()
      .sort((left, right) => {
        if (left.sort_order !== right.sort_order) {
          return left.sort_order - right.sort_order;
        }
        const activityCompare = latestTimestamp(
          right.last_activity_at,
          right.updated_at,
          right.created_at,
        ).localeCompare(latestTimestamp(
          left.last_activity_at,
          left.updated_at,
          left.created_at,
        ));
        if (activityCompare !== 0) {
          return activityCompare;
        }
        return String(left.display_name || "").localeCompare(String(right.display_name || ""));
      });
  }

  function buildConversationRow(
    detail: ConversationSummary | ConversationDetail,
    options?: {
      processing?: boolean;
      workspaceId?: string;
    },
  ): WorkspaceConversationRow {
    return {
      id: detail.id,
      workspace_id: String(options?.workspaceId || detail.workspace_id || "").trim(),
      workspace_path: String(detail.workspace_path || "").trim(),
      model_name: String(detail.model_name || "").trim(),
      title: String(detail.title || "").trim(),
      turn_count: Number(detail.turn_count || 0),
      total_tokens: Number(detail.total_tokens || 0),
      last_active_at: String(detail.last_active_at || "").trim(),
      started_at: String(detail.started_at || "").trim(),
      is_active:
        typeof options?.processing === "boolean"
          ? options.processing
          : Boolean("is_active" in detail && detail.is_active),
      linked_run_ids: Array.isArray(detail.linked_run_ids)
        ? detail.linked_run_ids.filter(Boolean)
        : [],
    };
  }

  function buildRunRow(detail: RunSummary | RunDetail, normalizedStatus: string): WorkspaceRunRow {
    const detailWorkspace =
      "workspace" in detail && detail.workspace && typeof detail.workspace === "object"
        ? detail.workspace
        : null;
    return {
      id: String(detail.id || "").trim(),
      workspace_id: String(detail.workspace_id || detailWorkspace?.id || "").trim(),
      workspace_path: String(detail.workspace_path || "").trim(),
      goal: String(detail.goal || "").trim(),
      status: normalizedStatus,
      created_at: String(detail.created_at || "").trim(),
      updated_at: String(detail.updated_at || "").trim(),
      execution_run_id: String(detail.execution_run_id || "").trim(),
      process_name: String(detail.process_name || "").trim(),
      linked_conversation_ids: Array.isArray(detail.linked_conversation_ids)
        ? detail.linked_conversation_ids.filter(Boolean)
        : [],
      changed_files_count: Number(detail.changed_files_count || 0),
    };
  }

  function notificationPayloadValue(
    payload: Record<string, unknown>,
    ...keys: string[]
  ): string {
    for (const key of keys) {
      const value = String(payload[key] || "").trim();
      if (value) {
        return value;
      }
    }
    return "";
  }

  function approvalItemFromNotification(event: NotificationEvent): ApprovalFeedItem | null {
    const payload = event.payload && typeof event.payload === "object"
      ? event.payload as Record<string, unknown>
      : {};
    const workspaceId = String(event.workspace_id || "").trim();
    if (!workspaceId) {
      return null;
    }

    if (event.kind === "conversation_approval") {
      const approvalId = String(event.approval_id || payload.approval_id || "").trim();
      const conversationId = String(event.conversation_id || payload.conversation_id || "").trim();
      if (!approvalId || !conversationId) {
        return null;
      }
      const argsPayload = payload.args && typeof payload.args === "object"
        ? payload.args as Record<string, unknown>
        : payload;
      const toolName = String(payload.tool_name || "").trim();
      return {
        id: `conversation:${conversationId}:${approvalId}`,
        kind: "conversation_approval",
        status: "pending",
        created_at: String(event.created_at || "").trim(),
        title: toolName ? `${toolName} approval` : String(event.title || "Tool approval").trim(),
        summary:
          notificationPayloadValue(argsPayload, "command", "path", "question", "text")
          || String(event.summary || "").trim(),
        workspace_id: workspaceId,
        workspace_path: String(event.workspace_path || "").trim(),
        workspace_display_name: String(event.workspace_display_name || "").trim(),
        task_id: "",
        run_id: "",
        conversation_id: conversationId,
        subtask_id: "",
        question_id: "",
        approval_id: approvalId,
        tool_name: toolName,
        risk_level: notificationPayloadValue(payload, "risk_level"),
        request_payload: argsPayload,
        metadata: {},
      };
    }

    if (event.kind === "task_question") {
      const questionId = notificationPayloadValue(payload, "question_id");
      const taskId = String(event.task_id || "").trim();
      if (!taskId || !questionId) {
        return null;
      }
      return {
        id: `question:${taskId}:${questionId}`,
        kind: "task_question",
        status: "pending",
        created_at: String(event.created_at || "").trim(),
        title: notificationPayloadValue(payload, "question") || String(event.title || "").trim(),
        summary:
          notificationPayloadValue(payload, "context_note")
          || String(event.summary || "").trim(),
        workspace_id: workspaceId,
        workspace_path: String(event.workspace_path || "").trim(),
        workspace_display_name: String(event.workspace_display_name || "").trim(),
        task_id: taskId,
        run_id: taskId,
        conversation_id: "",
        subtask_id: notificationPayloadValue(payload, "subtask_id"),
        question_id: questionId,
        approval_id: "",
        tool_name: "",
        risk_level: "",
        request_payload: payload,
        metadata: {},
      };
    }

    if (event.kind === "task_approval") {
      const taskId = String(event.task_id || "").trim();
      if (!taskId) {
        return null;
      }
      const subtaskId = notificationPayloadValue(payload, "subtask_id");
      return {
        id: subtaskId ? `task:${taskId}:${subtaskId}` : `task:${taskId}`,
        kind: "task_approval",
        status: "pending",
        created_at: String(event.created_at || "").trim(),
        title:
          notificationPayloadValue(payload, "proposed_action", "reason")
          || String(event.title || "").trim(),
        summary:
          notificationPayloadValue(payload, "reason", "proposed_action")
          || String(event.summary || "").trim(),
        workspace_id: workspaceId,
        workspace_path: String(event.workspace_path || "").trim(),
        workspace_display_name: String(event.workspace_display_name || "").trim(),
        task_id: taskId,
        run_id: taskId,
        conversation_id: "",
        subtask_id: subtaskId,
        question_id: "",
        approval_id: "",
        tool_name: notificationPayloadValue(payload, "tool_name"),
        risk_level: notificationPayloadValue(payload, "risk_level"),
        request_payload: {
          reason: notificationPayloadValue(payload, "reason"),
          proposed_action: notificationPayloadValue(payload, "proposed_action"),
          details:
            payload.details && typeof payload.details === "object"
              ? payload.details as Record<string, unknown>
              : {},
          auto_approve_timeout: payload.auto_approve_timeout,
        },
        metadata: {},
      };
    }

    return null;
  }

  const refreshWorkspaceOverviewState = useEffectEvent(async (workspaceId: string) => {
    const overviewPayload = await fetchWorkspaceOverview(workspaceId);
    applyWorkspaceOverview(workspaceId, overviewPayload);
  });

  const refreshWorkspaceArtifactsState = useEffectEvent(async (
    workspaceId: string,
    options?: {
      force?: boolean;
    },
  ) => {
    const force = options?.force === true;
    if (
      (!force && loadedWorkspaceArtifactsIdRef.current === workspaceId)
      || loadingWorkspaceArtifactsIdRef.current === workspaceId
    ) {
      return;
    }
    loadingWorkspaceArtifactsIdRef.current = workspaceId;
    try {
      const artifactPayload = await fetchWorkspaceArtifacts(workspaceId);
      if (selectedWorkspaceIdRef.current !== workspaceId) {
        return;
      }
      setWorkspaceArtifacts(artifactPayload);
      loadedWorkspaceArtifactsIdRef.current = workspaceId;
    } finally {
      if (loadingWorkspaceArtifactsIdRef.current === workspaceId) {
        loadingWorkspaceArtifactsIdRef.current = "";
      }
    }
  });

  const refreshWorkspaceInventoryState = useEffectEvent(async (workspaceId: string) => {
    if (
      loadedWorkspaceInventoryIdRef.current === workspaceId
      || loadingWorkspaceInventoryIdRef.current === workspaceId
    ) {
      return;
    }
    loadingWorkspaceInventoryIdRef.current = workspaceId;
    try {
      const inventoryPayload = await fetchWorkspaceInventory(workspaceId);
      if (selectedWorkspaceIdRef.current !== workspaceId) {
        return;
      }
      setInventory(inventoryPayload);
      loadedWorkspaceInventoryIdRef.current = workspaceId;
    } finally {
      if (loadingWorkspaceInventoryIdRef.current === workspaceId) {
        loadingWorkspaceInventoryIdRef.current = "";
      }
    }
  });

  const refreshWorkspaceSettingsState = useEffectEvent(async (workspaceId: string) => {
    if (
      loadedWorkspaceSettingsIdRef.current === workspaceId
      || loadingWorkspaceSettingsIdRef.current === workspaceId
    ) {
      return;
    }
    loadingWorkspaceSettingsIdRef.current = workspaceId;
    try {
      const workspaceSettingsPayload = await fetchWorkspaceSettings(workspaceId);
      if (selectedWorkspaceIdRef.current !== workspaceId) {
        return;
      }
      setWorkspaceSettings(workspaceSettingsPayload);
      loadedWorkspaceSettingsIdRef.current = workspaceId;
    } finally {
      if (loadingWorkspaceSettingsIdRef.current === workspaceId) {
        loadingWorkspaceSettingsIdRef.current = "";
      }
    }
  });

  const refreshVisibleWorkspaceTabState = useEffectEvent(async (workspaceId: string) => {
    if (activeTab === "overview") {
      await refreshWorkspaceArtifactsState(workspaceId);
      return;
    }
    if (activeTab === "runs") {
      await refreshWorkspaceInventoryState(workspaceId);
      return;
    }
    if (activeTab === "settings") {
      await refreshWorkspaceSettingsState(workspaceId);
    }
  });

  // Heavy recovery path: use for bootstrap, reconnect/manual refresh, or
  // explicit stale-repair cases when local patches cannot reliably derive
  // the affected workspace state.
  const refreshWorkspaceSurface = useEffectEvent(async (workspaceId: string) => {
    const [
      overviewPayload,
      approvalPayload,
    ] = await Promise.all([
      fetchWorkspaceOverview(workspaceId),
      fetchApprovals(workspaceId),
    ]);
    const isCurrentWorkspace = selectedWorkspaceIdRef.current === workspaceId;
    applyWorkspaceOverview(workspaceId, overviewPayload);
    if (!isCurrentWorkspace) {
      return;
    }
    setApprovalInbox(approvalPayload);
    await refreshVisibleWorkspaceTabState(workspaceId);
  });

  const refreshWorkspaceArtifacts = useEffectEvent(async (
    workspaceId: string,
    options?: {
      force?: boolean;
    },
  ) => {
    await refreshWorkspaceArtifactsState(workspaceId, options);
  });

  const refreshApprovalInbox = useEffectEvent(async (workspaceId: string) => {
    setApprovalInbox(await fetchApprovals(workspaceId));
  });

  const patchPendingApprovalCount = useEffectEvent((workspaceId: string, delta: number) => {
    if (!workspaceId || delta === 0) {
      return;
    }
    if (overviewRef.current?.workspace?.id !== workspaceId) {
      return;
    }
    setOverview((current) => {
      if (!current || current.workspace.id !== workspaceId) {
        return current;
      }
      const nextOverview = {
        ...current,
        pending_approvals_count: Math.max(
          0,
          Number(current.pending_approvals_count || 0) + delta,
        ),
      };
      overviewRef.current = nextOverview;
      return nextOverview;
    });
  });

  const syncConversationSummary = useEffectEvent((
    detail: ConversationSummary | ConversationDetail,
    options?: {
      incrementCount?: boolean;
      processing?: boolean;
      workspaceId?: string;
    },
  ) => {
    const workspaceId = String(options?.workspaceId || detail.workspace_id || "").trim();
    if (!workspaceId) {
      return;
    }

    const nextRow = buildConversationRow(detail, {
      processing: options?.processing,
      workspaceId,
    });
    const currentOverview = overviewRef.current;
    const isCurrentWorkspace = currentOverview?.workspace?.id === workspaceId;
    const existingConversation = isCurrentWorkspace
      ? currentOverview.recent_conversations.find((conversation) => conversation.id === nextRow.id)
      : null;
    const conversationCountDelta =
      existingConversation || options?.incrementCount === false ? 0 : 1;
    const nextLastActivityAt = latestTimestamp(
      nextRow.last_active_at,
      nextRow.started_at,
      currentOverview?.workspace.last_activity_at,
    );

    if (isCurrentWorkspace && currentOverview) {
      const nextRows = sortConversationRows([
        nextRow,
        ...currentOverview.recent_conversations.filter((conversation) => conversation.id !== nextRow.id),
      ]);
      const nextOverview = {
        ...currentOverview,
        workspace: {
          ...currentOverview.workspace,
          conversation_count: Math.max(
            nextRows.length,
            Number(currentOverview.workspace.conversation_count || 0) + conversationCountDelta,
          ),
          last_activity_at: nextLastActivityAt,
        },
        recent_conversations: nextRows,
      };
      overviewRef.current = nextOverview;
      setOverview(nextOverview);
    }

    setWorkspaces((current) => current.map((workspace) => {
      if (workspace.id !== workspaceId) {
        return workspace;
      }
      return {
        ...workspace,
        conversation_count: Math.max(
          0,
          Number(workspace.conversation_count || 0) + conversationCountDelta,
        ),
        last_activity_at: latestTimestamp(
          workspace.last_activity_at,
          nextRow.last_active_at,
          nextRow.started_at,
        ),
      };
    }));
  });

  const setConversationProcessing = useEffectEvent((
    conversationId: string,
    processing: boolean,
    options?: {
      lastActiveAt?: string;
      workspaceId?: string;
    },
  ) => {
    const workspaceId = String(options?.workspaceId || selectedWorkspaceIdRef.current || "").trim();
    if (!workspaceId || !conversationId) {
      return;
    }

    const currentOverview = overviewRef.current;
    if (currentOverview?.workspace?.id !== workspaceId) {
      return;
    }

    const existingConversation = currentOverview.recent_conversations.find(
      (conversation) => conversation.id === conversationId,
    );
    if (!existingConversation) {
      return;
    }

    const nextRow = {
      ...existingConversation,
      is_active: processing,
      last_active_at: latestTimestamp(
        existingConversation.last_active_at,
        options?.lastActiveAt,
      ),
    };
    const nextOverview = {
      ...currentOverview,
      workspace: {
        ...currentOverview.workspace,
        last_activity_at: latestTimestamp(
          currentOverview.workspace.last_activity_at,
          nextRow.last_active_at,
        ),
      },
      recent_conversations: sortConversationRows([
        nextRow,
        ...currentOverview.recent_conversations.filter((conversation) => conversation.id !== conversationId),
      ]),
    };
    overviewRef.current = nextOverview;
    setOverview(nextOverview);
    setWorkspaces((current) => current.map((workspace) => (
      workspace.id === workspaceId
        ? {
            ...workspace,
            last_activity_at: latestTimestamp(
              workspace.last_activity_at,
              nextRow.last_active_at,
            ),
          }
        : workspace
    )));
  });

  const removeConversationSummary = useEffectEvent((conversationId: string, workspaceId?: string) => {
    const nextWorkspaceId = String(workspaceId || selectedWorkspaceIdRef.current || "").trim();
    if (!nextWorkspaceId || !conversationId) {
      return;
    }

    const currentOverview = overviewRef.current;
    const isCurrentWorkspace = currentOverview?.workspace?.id === nextWorkspaceId;
    const existingConversation = isCurrentWorkspace
      ? currentOverview.recent_conversations.find((conversation) => conversation.id === conversationId)
      : null;
    const conversationCountDelta = existingConversation ? -1 : 0;

    if (isCurrentWorkspace && currentOverview && existingConversation) {
      const nextRows = currentOverview.recent_conversations.filter(
        (conversation) => conversation.id !== conversationId,
      );
      const nextOverview = {
        ...currentOverview,
        workspace: {
          ...currentOverview.workspace,
          conversation_count: Math.max(
            nextRows.length,
            Number(currentOverview.workspace.conversation_count || 0) - 1,
          ),
        },
        recent_conversations: nextRows,
      };
      overviewRef.current = nextOverview;
      setOverview(nextOverview);
    }

    setWorkspaces((current) => current.map((workspace) => (
      workspace.id === nextWorkspaceId
        ? {
            ...workspace,
            conversation_count: Math.max(
              0,
              Number(workspace.conversation_count || 0) + conversationCountDelta,
            ),
          }
        : workspace
    )));
  });

  const applyApprovalItem = useEffectEvent((
    item: ApprovalFeedItem,
    options?: {
      incrementCount?: boolean;
    },
  ) => {
    const workspaceId = String(item.workspace_id || "").trim();
    if (!workspaceId || workspaceId !== selectedWorkspaceIdRef.current) {
      return;
    }
    const exists = approvalInboxRef.current.some((currentItem) => currentItem.id === item.id);
    setApprovalInbox((current) => (
      [
        item,
        ...current.filter((currentItem) => currentItem.id !== item.id),
      ].sort((left, right) => String(right.created_at || "").localeCompare(String(left.created_at || "")))
    ));
    if (!exists && options?.incrementCount !== false) {
      patchPendingApprovalCount(workspaceId, 1);
    }
  });

  const removeApprovalItem = useEffectEvent((itemId: string, workspaceId?: string) => {
    const nextWorkspaceId = String(workspaceId || selectedWorkspaceIdRef.current || "").trim();
    if (!itemId || nextWorkspaceId !== selectedWorkspaceIdRef.current) {
      return;
    }
    const exists = approvalInboxRef.current.some((currentItem) => currentItem.id === itemId);
    if (!exists) {
      return;
    }
    setApprovalInbox((current) => current.filter((currentItem) => currentItem.id !== itemId));
    patchPendingApprovalCount(nextWorkspaceId, -1);
  });

  const syncRunDetail = useEffectEvent((detail: RunSummary | RunDetail) => {
    const currentOverview = overviewRef.current;
    const currentWorkspaceRun = currentOverview?.recent_runs.find((run) => run.id === detail.id);
    const detailWorkspace =
      "workspace" in detail && detail.workspace && typeof detail.workspace === "object"
        ? detail.workspace
        : null;
    const workspaceId = String(
      detail.workspace_id
      || detailWorkspace?.id
      || (
        currentOverview?.workspace?.id
        && currentWorkspaceRun
          ? currentOverview.workspace.id
          : ""
      ),
    ).trim();
    const normalizedStatus = normalizeRunStatus(detail.status);
    if (!workspaceId || !normalizedStatus) {
      return;
    }
    const nextRow = buildRunRow(detail, normalizedStatus);
    const existingRun = currentOverview?.workspace?.id === workspaceId
      ? (currentWorkspaceRun || null)
      : null;
    const previousActive = existingRun ? isRunActiveStatus(existingRun.status) : false;
    const nextActive = isRunActiveStatus(normalizedStatus);
    const activeRunCountDelta = nextActive === previousActive ? 0 : nextActive ? 1 : -1;
    const runCountDelta = existingRun ? 0 : 1;
    let nextOverviewWorkspace: WorkspaceOverview["workspace"] | null = null;

    if (currentOverview?.workspace?.id === workspaceId) {
      const nextRows = sortRunRows([
        nextRow,
        ...currentOverview.recent_runs.filter((run) => run.id !== detail.id),
      ]);
      const nextOverview = {
        ...currentOverview,
        workspace: {
          ...currentOverview.workspace,
          active_run_count: Math.max(
            0,
            Number(currentOverview.workspace.active_run_count || 0) + activeRunCountDelta,
          ),
          run_count: Math.max(
            nextRows.length,
            Number(currentOverview.workspace.run_count || 0) + runCountDelta,
          ),
          last_activity_at: latestTimestamp(
            currentOverview.workspace.last_activity_at,
            nextRow.updated_at,
            nextRow.created_at,
          ),
        },
        recent_runs: nextRows,
      };
      nextOverviewWorkspace = nextOverview.workspace;
      overviewRef.current = nextOverview;
      setOverview(nextOverview);
    }

    setWorkspaces((current) => current.map((workspace) => (
      workspace.id === workspaceId
        ? {
            ...workspace,
            active_run_count: nextOverviewWorkspace
              ? Number(nextOverviewWorkspace.active_run_count || 0)
              : Math.max(
                0,
                Number(workspace.active_run_count || 0) + activeRunCountDelta,
              ),
            run_count: nextOverviewWorkspace
              ? Number(nextOverviewWorkspace.run_count || 0)
              : Math.max(
                0,
                Number(workspace.run_count || 0) + runCountDelta,
              ),
            last_activity_at: nextOverviewWorkspace
              ? String(nextOverviewWorkspace.last_activity_at || "")
              : latestTimestamp(
                workspace.last_activity_at,
                nextRow.updated_at,
                nextRow.created_at,
              ),
          }
        : workspace
    )));
  });

  const removeRunSummary = useEffectEvent((runId: string, workspaceId?: string) => {
    const nextWorkspaceId = String(workspaceId || selectedWorkspaceIdRef.current || "").trim();
    if (!nextWorkspaceId || !runId) {
      return;
    }

    const currentOverview = overviewRef.current;
    const existingRun = currentOverview?.workspace?.id === nextWorkspaceId
      ? currentOverview.recent_runs.find((run) => run.id === runId)
      : null;
    const activeRunCountDelta = existingRun && isRunActiveStatus(existingRun.status) ? -1 : 0;
    const runCountDelta = existingRun ? -1 : 0;
    let nextOverviewWorkspace: WorkspaceOverview["workspace"] | null = null;

    if (currentOverview?.workspace?.id === nextWorkspaceId && existingRun) {
      const nextRows = currentOverview.recent_runs.filter((run) => run.id !== runId);
      const nextOverview = {
        ...currentOverview,
        workspace: {
          ...currentOverview.workspace,
          active_run_count: Math.max(
            0,
            Number(currentOverview.workspace.active_run_count || 0) + activeRunCountDelta,
          ),
          run_count: Math.max(
            nextRows.length,
            Number(currentOverview.workspace.run_count || 0) - 1,
          ),
        },
        recent_runs: nextRows,
      };
      nextOverviewWorkspace = nextOverview.workspace;
      overviewRef.current = nextOverview;
      setOverview(nextOverview);
    }

    setWorkspaces((current) => current.map((workspace) => (
      workspace.id === nextWorkspaceId
        ? {
            ...workspace,
            active_run_count: nextOverviewWorkspace
              ? Number(nextOverviewWorkspace.active_run_count || 0)
              : Math.max(
                0,
                Number(workspace.active_run_count || 0) + activeRunCountDelta,
              ),
            run_count: nextOverviewWorkspace
              ? Number(nextOverviewWorkspace.run_count || 0)
              : Math.max(
                0,
                Number(workspace.run_count || 0) + runCountDelta,
              ),
          }
        : workspace
    )));
  });

  const appendNotification = useEffectEvent((event: NotificationEvent) => {
    setNotifications((current) => {
      const deduped = current.filter((item) => item.id !== event.id);
      return [event, ...deduped].slice(0, 8);
    });
  });

  const scheduleNotificationRepair = useEffectEvent((workspaceId: string, repair: {
    approvals?: boolean;
    overview?: boolean;
  }) => {
    if (!workspaceId) {
      return;
    }
    const pending = notificationRepairRef.current;
    if (pending.workspaceId && pending.workspaceId !== workspaceId) {
      pending.workspaceId = workspaceId;
      pending.approvals = false;
      pending.overview = false;
    }
    pending.workspaceId = workspaceId;
    pending.approvals ||= Boolean(repair.approvals);
    pending.overview ||= Boolean(repair.overview);
    if (notificationRefreshTimerRef.current !== null) {
      return;
    }
    notificationRefreshTimerRef.current = window.setTimeout(() => {
      notificationRefreshTimerRef.current = null;
      const nextRepair = notificationRepairRef.current;
      notificationRepairRef.current = {
        approvals: false,
        overview: false,
        workspaceId: "",
      };
      void Promise.all([
        nextRepair.approvals && nextRepair.workspaceId
          ? refreshApprovalInbox(nextRepair.workspaceId)
          : Promise.resolve(),
        nextRepair.overview && nextRepair.workspaceId
          ? refreshWorkspaceOverviewState(nextRepair.workspaceId)
          : Promise.resolve(),
      ]).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh inbox.");
        }
      });
    }, 150);
  });

  const applyNotificationEvent = useEffectEvent((event: NotificationEvent) => {
    appendNotification(event);
    const workspaceId = String(event.workspace_id || "").trim();
    if (!workspaceId) {
      return;
    }

    if (event.event_type === "approval_requested" || event.event_type === "ask_user_requested") {
      const item = approvalItemFromNotification(event);
      if (item) {
        applyApprovalItem(item);
      } else if (event.event_type === "approval_requested") {
        patchPendingApprovalCount(workspaceId, 1);
        scheduleNotificationRepair(workspaceId, { approvals: true });
      } else {
        scheduleNotificationRepair(workspaceId, { approvals: true });
      }
      return;
    }

    if (
      event.event_type === "approval_received"
      || event.event_type === "approval_rejected"
      || event.event_type === "approval_timed_out"
      || event.event_type === "ask_user_answered"
      || event.event_type === "ask_user_cancelled"
      || event.event_type === "ask_user_timeout"
    ) {
      let itemId = "";
      if (event.kind === "conversation_approval" && event.conversation_id && event.approval_id) {
        itemId = `conversation:${event.conversation_id}:${event.approval_id}`;
      } else if (event.kind === "task_question") {
        const questionId = notificationPayloadValue(
          event.payload as Record<string, unknown>,
          "question_id",
        );
        if (event.task_id && questionId) {
          itemId = `question:${event.task_id}:${questionId}`;
        }
      } else if (event.kind === "task_approval" && event.task_id) {
        const subtaskId = notificationPayloadValue(
          event.payload as Record<string, unknown>,
          "subtask_id",
        );
        itemId = subtaskId ? `task:${event.task_id}:${subtaskId}` : `task:${event.task_id}`;
      }

      if (!itemId) {
        const payload = event.payload && typeof event.payload === "object"
          ? event.payload as Record<string, unknown>
          : {};
        itemId = approvalInboxRef.current.find((item) => (
          item.workspace_id === workspaceId
          && (
            (event.approval_id && item.approval_id === event.approval_id)
            || (
              notificationPayloadValue(payload, "question_id")
              && item.question_id === notificationPayloadValue(payload, "question_id")
            )
            || (event.task_id && item.task_id === event.task_id && item.kind === event.kind)
          )
        ))?.id || "";
      }

      if (itemId) {
        removeApprovalItem(itemId, workspaceId);
      } else if (event.event_type.startsWith("approval_")) {
        patchPendingApprovalCount(workspaceId, -1);
        scheduleNotificationRepair(workspaceId, { approvals: true });
      } else {
        scheduleNotificationRepair(workspaceId, { approvals: true });
      }
    }
  });

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (workspaceSearchTimerRef.current !== null) {
        window.clearTimeout(workspaceSearchTimerRef.current);
      }
      if (notificationRefreshTimerRef.current !== null) {
        window.clearTimeout(notificationRefreshTimerRef.current);
      }
    };
  }, []);

  // Sync workspace drafts from the selected summary without clobbering dirty local edits.
  useEffect(() => {
    const selectedWorkspace = workspaces.find((workspace) => workspace.id === selectedWorkspaceId);
    const nextSource = {
      id: selectedWorkspaceId,
      name: selectedWorkspace?.display_name || "",
      tags: workspaceTagsFromMetadata(selectedWorkspace?.metadata).join(", "),
      note: workspaceNoteFromMetadata(selectedWorkspace?.metadata),
    };
    const previousSource = previousWorkspaceDraftSourceRef.current;
    const selectedWorkspaceChanged = previousSource.id !== nextSource.id;

    setWorkspaceNameDraft((current) => (
      selectedWorkspaceChanged || current === previousSource.name
        ? nextSource.name
        : current
    ));
    setWorkspaceTagsDraft((current) => (
      selectedWorkspaceChanged || current === previousSource.tags
        ? nextSource.tags
        : current
    ));
    setWorkspaceNoteDraft((current) => (
      selectedWorkspaceChanged || current === previousSource.note
        ? nextSource.note
        : current
    ));

    if (selectedWorkspaceChanged) {
      setWorkspaceFileTreeMode("all");
      setWorkspaceImportFolderDraft("");
    }

    previousWorkspaceDraftSourceRef.current = nextSource;
  }, [selectedWorkspaceId, workspaces]);

  // Load workspace overview on selection change
  useEffect(() => {
    selectedWorkspaceIdRef.current = selectedWorkspaceId;
  }, [selectedWorkspaceId]);

  useEffect(() => {
    overviewRef.current = overview;
  }, [overview]);

  useEffect(() => {
    approvalInboxRef.current = approvalInbox;
  }, [approvalInbox]);

  useEffect(() => {
    if (!selectedWorkspaceId) {
      setOverview(null);
      setApprovalInbox([]);
      setNotifications([]);
      setInventory(null);
      setWorkspaceArtifacts([]);
      setWorkspaceSearchResults(null);
      setWorkspaceSettings(null);
      loadedWorkspaceArtifactsIdRef.current = "";
      loadedWorkspaceInventoryIdRef.current = "";
      loadedWorkspaceSettingsIdRef.current = "";
      loadingWorkspaceArtifactsIdRef.current = "";
      loadingWorkspaceInventoryIdRef.current = "";
      loadingWorkspaceSettingsIdRef.current = "";
      lastSeenNotificationStreamIdRef.current = 0;
      previousWorkspaceIdRef.current = "";
      return;
    }
    const hasMatchingOverview =
      overviewRef.current?.workspace?.id === selectedWorkspaceId;
    if (connectionState !== "connected") {
      loadedWorkspaceArtifactsIdRef.current = "";
      loadedWorkspaceInventoryIdRef.current = "";
      loadedWorkspaceSettingsIdRef.current = "";
      loadingWorkspaceArtifactsIdRef.current = "";
      loadingWorkspaceInventoryIdRef.current = "";
      loadingWorkspaceSettingsIdRef.current = "";
      if (!hasMatchingOverview) {
        setOverview(null);
        setApprovalInbox([]);
        setNotifications([]);
        setInventory(null);
        setWorkspaceArtifacts([]);
        setWorkspaceSearchResults(null);
        setWorkspaceSettings(null);
        loadedWorkspaceArtifactsIdRef.current = "";
        loadedWorkspaceInventoryIdRef.current = "";
        loadedWorkspaceSettingsIdRef.current = "";
        loadingWorkspaceArtifactsIdRef.current = "";
        loadingWorkspaceInventoryIdRef.current = "";
        loadingWorkspaceSettingsIdRef.current = "";
        lastSeenNotificationStreamIdRef.current = 0;
      }
      setLoadingOverview(!hasMatchingOverview);
      return;
    }
    let cancelled = false;
    setLoadingOverview(true);
    setError("");
    if (!hasMatchingOverview) {
      setOverview(null);
      setApprovalInbox([]);
      setNotifications([]);
      setInventory(null);
      setWorkspaceArtifacts([]);
      setWorkspaceSearchResults(null);
      setWorkspaceSettings(null);
      loadedWorkspaceArtifactsIdRef.current = "";
      loadedWorkspaceInventoryIdRef.current = "";
      loadedWorkspaceSettingsIdRef.current = "";
      loadingWorkspaceArtifactsIdRef.current = "";
      loadingWorkspaceInventoryIdRef.current = "";
      loadingWorkspaceSettingsIdRef.current = "";
      lastSeenNotificationStreamIdRef.current = 0;
    }
    if (
      previousWorkspaceIdRef.current
      && previousWorkspaceIdRef.current !== selectedWorkspaceId
    ) {
      const shouldClearConversation =
        !selectedConversationId
        || selectedConversationId === previousSelectedConversationIdRef.current;
      const shouldClearRun =
        !selectedRunId
        || selectedRunId === previousSelectedRunIdRef.current;
      startTransition(() => {
        if (shouldClearConversation) {
          setSelectedConversationId("");
        }
        if (shouldClearRun) {
          setSelectedRunId("");
        }
      });
    }
    previousWorkspaceIdRef.current = selectedWorkspaceId;

    void (async () => {
      try {
        const [
          overviewPayload,
          approvalPayload,
        ] = await Promise.all([
          fetchWorkspaceOverview(selectedWorkspaceId),
          fetchApprovals(selectedWorkspaceId),
        ]);
        if (cancelled) {
          return;
        }
        setApprovalInbox(approvalPayload);
        setWorkspaceSearchResults(null);
        applyWorkspaceOverview(selectedWorkspaceId, overviewPayload);
        void refreshVisibleWorkspaceTabState(selectedWorkspaceId).catch((err) => {
          if (!isTransientRequestError(err)) {
            setError(err instanceof Error ? err.message : "Failed to load workspace details.");
          }
        });
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load overview.");
          if (!hasMatchingOverview) {
            setOverview(null);
            setWorkspaceSettings(null);
            setInventory(null);
            setWorkspaceArtifacts([]);
          }
        }
      } finally {
        if (!cancelled) {
          setLoadingOverview(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [connectionState, selectedWorkspaceId]);

  useEffect(() => {
    if (connectionState !== "connected" || !selectedWorkspaceId) {
      return;
    }
    if (activeTab === "overview") {
      if (loadedWorkspaceArtifactsIdRef.current === selectedWorkspaceId) {
        return;
      }
      void refreshWorkspaceArtifactsState(selectedWorkspaceId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to load workspace artifacts.");
        }
      });
      return;
    }
    if (activeTab === "runs") {
      if (loadedWorkspaceInventoryIdRef.current === selectedWorkspaceId) {
        return;
      }
      void refreshWorkspaceInventoryState(selectedWorkspaceId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to load workspace inventory.");
        }
      });
      return;
    }
    if (
      activeTab === "settings"
      && loadedWorkspaceSettingsIdRef.current !== selectedWorkspaceId
    ) {
      void refreshWorkspaceSettingsState(selectedWorkspaceId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to load workspace settings.");
        }
      });
    }
  }, [
    activeTab,
    connectionState,
    selectedWorkspaceId,
    setError,
  ]);

  useEffect(() => {
    previousSelectedConversationIdRef.current = selectedConversationId;
  }, [selectedConversationId, selectedWorkspaceId]);

  useEffect(() => {
    previousSelectedRunIdRef.current = selectedRunId;
  }, [selectedRunId, selectedWorkspaceId]);

  // Debounced workspace search
  useEffect(() => {
    const query = workspaceSearchQuery.trim();
    if (workspaceSearchTimerRef.current !== null) {
      window.clearTimeout(workspaceSearchTimerRef.current);
      workspaceSearchTimerRef.current = null;
    }
    if (!selectedWorkspaceId || !query) {
      setSearchingWorkspace(false);
      setWorkspaceSearchResults(null);
      return;
    }
    setSearchingWorkspace(true);
    let cancelled = false;
    workspaceSearchTimerRef.current = window.setTimeout(() => {
      workspaceSearchTimerRef.current = null;
      void (async () => {
        try {
          const results = await fetchWorkspaceSearch(selectedWorkspaceId, query, 4);
          if (!cancelled) {
            setWorkspaceSearchResults(results);
          }
        } catch (err) {
          if (!cancelled) {
            setWorkspaceSearchResults(null);
            setError(err instanceof Error ? err.message : "Failed to search workspace.");
          }
        } finally {
          if (!cancelled) {
            setSearchingWorkspace(false);
          }
        }
      })();
    }, 180);
    return () => {
      cancelled = true;
      if (workspaceSearchTimerRef.current !== null) {
        window.clearTimeout(workspaceSearchTimerRef.current);
        workspaceSearchTimerRef.current = null;
      }
    };
  }, [selectedWorkspaceId, workspaceSearchQuery]);

  // Notification stream subscription
  useEffect(() => {
    if (!selectedWorkspaceId) {
      setNotifications([]);
      notificationRepairRef.current = {
        approvals: false,
        overview: false,
        workspaceId: "",
      };
      return;
    }
    if (notificationRefreshTimerRef.current !== null) {
      window.clearTimeout(notificationRefreshTimerRef.current);
      notificationRefreshTimerRef.current = null;
    }
    const cleanup = subscribeNotificationsStream(
      selectedWorkspaceId,
      (event) => {
        if (typeof event.stream_id === "number" && event.stream_id > 0) {
          lastSeenNotificationStreamIdRef.current = Math.max(
            lastSeenNotificationStreamIdRef.current,
            event.stream_id,
          );
        }
        applyNotificationEvent(event);
      },
      () => {
        scheduleNotificationRepair(selectedWorkspaceId, {
          approvals: true,
          overview: true,
        });
      },
      {
        afterId: lastSeenNotificationStreamIdRef.current,
      },
    );
    return () => {
      if (notificationRefreshTimerRef.current !== null) {
        window.clearTimeout(notificationRefreshTimerRef.current);
        notificationRefreshTimerRef.current = null;
      }
      notificationRepairRef.current = {
        approvals: false,
        overview: false,
        workspaceId: "",
      };
      cleanup();
    };
  }, [selectedWorkspaceId]);

  // ---------------------------------------------------------------------------
  // Computed values
  // ---------------------------------------------------------------------------

  const selectedWorkspaceSummary = workspaces.find(
    (workspace) => workspace.id === selectedWorkspaceId,
  ) || null;
  const selectedWorkspaceArchived = Boolean(selectedWorkspaceSummary?.is_archived);
  const selectedWorkspacePinned = Boolean(
    selectedWorkspaceSummary && selectedWorkspaceSummary.sort_order < 0,
  );
  const selectedWorkspaceTags = workspaceTagsFromMetadata(selectedWorkspaceSummary?.metadata);
  const selectedWorkspaceNote = workspaceNoteFromMetadata(selectedWorkspaceSummary?.metadata);
  const noWorkspacesRegistered = workspaces.length === 0;
  const selectedWorkspaceIsEmpty = Boolean(
    selectedWorkspaceId
    && overview
    && overview.recent_conversations.length === 0
    && overview.recent_runs.length === 0,
  );
  const workspaceConversationRows = overview?.recent_conversations || [];
  const workspaceRunRows = overview?.recent_runs || [];
  const filteredConversations = workspaceConversationRows.filter((conversation) =>
    matchesWorkspaceSearch(
      workspaceSearchQuery,
      conversation.title,
      conversation.model_name,
      conversation.last_active_at,
      conversation.linked_run_ids.join(" "),
    ),
  );
  const filteredRuns = workspaceRunRows.filter((run) =>
    matchesWorkspaceSearch(
      workspaceSearchQuery,
      run.goal,
      run.process_name,
      run.status,
      run.linked_conversation_ids.join(" "),
    ),
  );
  const filteredApprovalItems = approvalInbox.filter((item) =>
    matchesWorkspaceSearch(
      workspaceSearchQuery,
      item.title,
      item.summary,
      item.tool_name,
      item.kind,
      item.request_payload,
    ),
  );
  const filteredProcesses = (inventory?.processes || []).filter((process) =>
    matchesWorkspaceSearch(
      workspaceSearchQuery,
      process.name,
      process.description,
      process.author,
      process.path,
    ),
  );
  const filteredMcpServers = (inventory?.mcp_servers || []).filter((server) =>
    matchesWorkspaceSearch(
      workspaceSearchQuery,
      server.alias,
      server.type,
      server.command,
      server.url,
      server.cwd,
    ),
  );
  const filteredTools = (inventory?.tools || []).filter((tool) =>
    matchesWorkspaceSearch(
      workspaceSearchQuery,
      tool.name,
      tool.description,
      tool.auth_mode,
      tool.execution_surfaces.join(" "),
    ),
  );
  const filteredWorkspaceArtifacts = workspaceArtifacts.filter((artifact) =>
    matchesWorkspaceSearch(
      workspaceSearchQuery,
      artifact.path,
      artifact.category,
      artifact.source,
      artifact.tool_name,
      artifact.latest_run_id,
      artifact.run_ids.join(" "),
      artifact.phase_ids.join(" "),
      artifact.subtask_ids.join(" "),
      artifact.facets,
    ),
  );
  const recentWorkspaceArtifacts = [...filteredWorkspaceArtifacts]
    .sort((left, right) => {
      const leftTime = Date.parse(left.created_at || "");
      const rightTime = Date.parse(right.created_at || "");
      if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) {
        return rightTime - leftTime;
      }
      if (left.latest_run_id && !right.latest_run_id) {
        return -1;
      }
      if (!left.latest_run_id && right.latest_run_id) {
        return 1;
      }
      return left.path.localeCompare(right.path);
    })
    .slice(0, 24);
  const recentNotifications = notifications.slice(0, 4);
  const searchGroups = workspaceSearchResults
    ? [
        { label: "Threads", rows: workspaceSearchResults.conversations },
        { label: "Runs", rows: workspaceSearchResults.runs },
        { label: "Approvals", rows: workspaceSearchResults.approvals },
        { label: "Artifacts", rows: workspaceSearchResults.artifacts },
        { label: "Files", rows: workspaceSearchResults.files },
        { label: "Processes", rows: workspaceSearchResults.processes },
        { label: "MCP servers", rows: workspaceSearchResults.mcp_servers },
        { label: "Tools", rows: workspaceSearchResults.tools },
      ]
    : [];

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  async function handleImportWorkspace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const path = importPath.trim();
    if (!path) {
      setError("Workspace path is required.");
      return;
    }
    setImportingWorkspace(true);
    setError("");
    setNotice("");
    try {
      const created = await createWorkspace({
        path,
        display_name: importDisplayName.trim() || defaultWorkspaceName(path),
      });
      await refreshWorkspaceList(created.id);
      setImportPath("");
      setImportDisplayName("");
      setNotice(`Imported workspace ${created.display_name}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to import workspace.");
    } finally {
      setImportingWorkspace(false);
    }
  }

  async function handleCreateWorkspace(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const workspacePath = joinWorkspacePath(createParentPath, createFolderName);
    if (!workspacePath.trim()) {
      setError("Workspace parent path and folder name are required.");
      return;
    }
    setCreatingWorkspace(true);
    setError("");
    setNotice("");
    try {
      const createdPath = await createWorkspaceDirectory(workspacePath);
      const created = await createWorkspace({
        path: createdPath,
        display_name: createDisplayName.trim() || defaultWorkspaceName(createdPath),
      });
      await refreshWorkspaceList(created.id);
      setCreateFolderName("");
      setCreateDisplayName("");
      setNotice(`Created workspace ${created.display_name}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create workspace.");
    } finally {
      setCreatingWorkspace(false);
    }
  }

  async function handleSaveWorkspaceDetails(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedWorkspaceId) {
      setError("Select a workspace before updating it.");
      return;
    }
    const nextName = workspaceNameDraft.trim();
    if (!nextName) {
      setError("Workspace name is required.");
      return;
    }
    setSavingWorkspaceMeta(true);
    setError("");
    setNotice("");
    try {
      const nextTags = parseWorkspaceTags(workspaceTagsDraft);
      const updated = await patchWorkspace(selectedWorkspaceId, {
        display_name: nextName,
        metadata: {
          note: workspaceNoteDraft.trim(),
          tags: nextTags,
        },
      });
      setWorkspaces((current) => mergeWorkspaceSummary(current, updated));
      if (overview?.workspace.id === updated.id) {
        setOverview((current) => (
          current
            ? {
                ...current,
                workspace: updated,
              }
            : current
        ));
      }
      setWorkspaceNameDraft(updated.display_name);
      setWorkspaceTagsDraft(workspaceTagsFromMetadata(updated.metadata).join(", "));
      setWorkspaceNoteDraft(workspaceNoteFromMetadata(updated.metadata));
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save workspace details.");
    } finally {
      setSavingWorkspaceMeta(false);
    }
  }

  async function handleArchiveWorkspace(nextArchived: boolean) {
    if (!selectedWorkspaceId) {
      setError("Select a workspace before updating it.");
      return;
    }
    setSavingWorkspaceMeta(true);
    setError("");
    setNotice("");
    try {
      const updated = await patchWorkspace(selectedWorkspaceId, {
        archived: nextArchived,
      });
      const nextVisibleWorkspaces = sortWorkspaceSummaries(
        mergeWorkspaceSummary(workspaces, updated).filter((workspace) =>
          showArchivedWorkspaces || !workspace.is_archived,
        ),
      );
      setWorkspaces(nextVisibleWorkspaces);
      if (overview?.workspace.id === updated.id) {
        setOverview((current) => (
          current
            ? {
                ...current,
                workspace: updated,
              }
            : current
        ));
      }
      if (nextArchived && !showArchivedWorkspaces) {
        startTransition(() => {
          setSelectedWorkspaceId(nextVisibleWorkspaces.find((workspace) => workspace.id !== updated.id)?.id || "");
        });
      }
      setNotice(
        nextArchived
          ? `Archived workspace ${updated.display_name}.`
          : `Restored workspace ${updated.display_name}.`,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : `Failed to ${nextArchived ? "archive" : "restore"} workspace.`,
      );
    } finally {
      setSavingWorkspaceMeta(false);
    }
  }

  async function handlePinWorkspace(nextPinned: boolean) {
    if (!selectedWorkspaceId) {
      setError("Select a workspace before updating its order.");
      return;
    }
    setSavingWorkspaceMeta(true);
    setError("");
    setNotice("");
    try {
      const nextSortOrder = nextPinned
        ? Math.min(
            workspaces.reduce((min, workspace) => Math.min(min, workspace.sort_order), 0) - 1,
            -1,
          )
        : 0;
      const updated = await patchWorkspace(selectedWorkspaceId, {
        sort_order: nextSortOrder,
      });
      setWorkspaces((current) => sortWorkspaceSummaries(mergeWorkspaceSummary(current, updated)));
      if (overview?.workspace.id === updated.id) {
        setOverview((current) => (
          current
            ? {
                ...current,
                workspace: updated,
              }
            : current
        ));
      }
      setNotice(
        nextPinned
          ? `Pinned workspace ${updated.display_name} to the top.`
          : `Workspace ${updated.display_name} now follows recent ordering.`,
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : `Failed to ${nextPinned ? "pin" : "unpin"} workspace.`,
      );
    } finally {
      setSavingWorkspaceMeta(false);
    }
  }

  function handlePrefillStarterWorkspace() {
    const defaultParent = runtime?.workspace_default_path || createParentPath || "";
    setCreateParentPath(defaultParent);
    setCreateFolderName((current) => current || "loom-starter");
    setCreateDisplayName((current) => current || "Starter Workspace");
    
    setError("");
  }

  function focusSearch() {
    searchInputRef.current?.focus();
    searchInputRef.current?.select();
  }

  return {
    // State
    selectedWorkspaceId,
    showArchivedWorkspaces,
    workspaceNameDraft,
    workspaceTagsDraft,
    workspaceNoteDraft,
    workspaceFileTreeMode,
    workspaceImportFolderDraft,
    importPath,
    importDisplayName,
    createParentPath,
    createFolderName,
    createDisplayName,
    importingWorkspace,
    creatingWorkspace,
    savingWorkspaceMeta,
    workspaceSearchQuery,
    overview,
    inventory,
    loadingOverview,
    approvalInbox,
    notifications,
    workspaceSettings,
    workspaceArtifacts,
    workspaceSearchResults,
    searchingWorkspace,

    // Computed
    selectedWorkspaceSummary,
    selectedWorkspaceArchived,
    selectedWorkspacePinned,
    selectedWorkspaceTags,
    selectedWorkspaceNote,
    noWorkspacesRegistered,
    selectedWorkspaceIsEmpty,
    workspaceConversationRows,
    workspaceRunRows,
    filteredConversations,
    filteredRuns,
    filteredApprovalItems,
    filteredProcesses,
    filteredMcpServers,
    filteredTools,
    filteredWorkspaceArtifacts,
    recentWorkspaceArtifacts,
    recentNotifications,
    searchGroups,

    // Refs
    searchInputRef,
    inboxSectionRef,

    // Setters
    setSelectedWorkspaceId,
    setShowArchivedWorkspaces,
    setWorkspaceNameDraft,
    setWorkspaceTagsDraft,
    setWorkspaceNoteDraft,
    setWorkspaceFileTreeMode,
    setWorkspaceImportFolderDraft,
    setImportPath,
    setImportDisplayName,
    setCreateParentPath,
    setCreateFolderName,
    setCreateDisplayName,
    setWorkspaceSearchQuery,

    // Handlers
    handleImportWorkspace,
    handleCreateWorkspace,
    handleSaveWorkspaceDetails,
    handleArchiveWorkspace,
    handlePinWorkspace,
    handlePrefillStarterWorkspace,
    focusSearch,
    refreshWorkspaceList,
    refreshWorkspaceSurface,
    refreshWorkspaceArtifacts,
    refreshApprovalInbox,
    syncConversationSummary,
    setConversationProcessing,
    removeConversationSummary,
    applyApprovalItem,
    removeApprovalItem,
    syncRunDetail,
    removeRunSummary,
  };
}
