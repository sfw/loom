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
  type NotificationEvent,
  type RuntimeStatus,
  type WorkspaceArtifact,
  type WorkspaceInventory,
  type WorkspaceOverview,
  type WorkspaceSearchResponse,
  type WorkspaceSettingsPayload,
  type WorkspaceSummary,
} from "../api";
import { matchesWorkspaceSearch } from "../history";
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
  refreshApprovalInbox: (workspaceId: string) => Promise<void>;
}

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
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const inboxSectionRef = useRef<HTMLElement | null>(null);
  const workspaceSearchTimerRef = useRef<number | null>(null);
  const notificationRefreshTimerRef = useRef<number | null>(null);
  const lastSeenNotificationStreamIdRef = useRef(0);
  const previousWorkspaceIdRef = useRef("");
  const selectedWorkspaceIdRef = useRef(selectedWorkspaceId);
  const previousSelectedConversationIdRef = useRef(selectedConversationId);
  const previousSelectedRunIdRef = useRef(selectedRunId);
  const overviewRef = useRef<WorkspaceOverview | null>(null);

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

  const refreshWorkspaceSurface = useEffectEvent(async (workspaceId: string) => {
    const [
      overviewPayload,
      workspaceSettingsPayload,
      approvalPayload,
      inventoryPayload,
      artifactPayload,
    ] = await Promise.all([
      fetchWorkspaceOverview(workspaceId),
      fetchWorkspaceSettings(workspaceId),
      fetchApprovals(workspaceId),
      fetchWorkspaceInventory(workspaceId),
      fetchWorkspaceArtifacts(workspaceId),
    ]);
    const isCurrentWorkspace = selectedWorkspaceIdRef.current === workspaceId;
    if (!isCurrentWorkspace) {
      setWorkspaces((current) => mergeWorkspaceSummary(current, overviewPayload.workspace));
      return;
    }
    setOverview(overviewPayload);
    setWorkspaceSettings(workspaceSettingsPayload);
    setApprovalInbox(approvalPayload);
    setInventory(inventoryPayload);
    setWorkspaceArtifacts(artifactPayload);
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

  const refreshApprovalInbox = useEffectEvent(async (workspaceId: string) => {
    setApprovalInbox(await fetchApprovals(workspaceId));
  });

  const scheduleNotificationRefresh = useEffectEvent((workspaceId: string) => {
    if (!workspaceId || notificationRefreshTimerRef.current !== null) {
      return;
    }
    notificationRefreshTimerRef.current = window.setTimeout(() => {
      notificationRefreshTimerRef.current = null;
      void refreshWorkspaceSurface(workspaceId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh inbox.");
        }
      });
    }, 150);
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

  // Sync workspace name/tags/note drafts when selection or workspaces change
  useEffect(() => {
    const selectedWorkspace = workspaces.find((workspace) => workspace.id === selectedWorkspaceId);
    setWorkspaceNameDraft(selectedWorkspace?.display_name || "");
    setWorkspaceTagsDraft(workspaceTagsFromMetadata(selectedWorkspace?.metadata).join(", "));
    setWorkspaceNoteDraft(workspaceNoteFromMetadata(selectedWorkspace?.metadata));
    setWorkspaceFileTreeMode("all");
    setWorkspaceImportFolderDraft("");
  }, [selectedWorkspaceId, workspaces]);

  // Load workspace overview on selection change
  useEffect(() => {
    selectedWorkspaceIdRef.current = selectedWorkspaceId;
  }, [selectedWorkspaceId]);

  useEffect(() => {
    overviewRef.current = overview;
  }, [overview]);

  useEffect(() => {
    if (!selectedWorkspaceId) {
      setOverview(null);
      setApprovalInbox([]);
      setNotifications([]);
      setInventory(null);
      setWorkspaceArtifacts([]);
      setWorkspaceSearchResults(null);
      setWorkspaceSettings(null);
      lastSeenNotificationStreamIdRef.current = 0;
      previousWorkspaceIdRef.current = "";
      return;
    }
    const hasMatchingOverview =
      overviewRef.current?.workspace?.id === selectedWorkspaceId;
    if (connectionState !== "connected") {
      if (!hasMatchingOverview) {
        setOverview(null);
        setApprovalInbox([]);
        setNotifications([]);
        setInventory(null);
        setWorkspaceArtifacts([]);
        setWorkspaceSearchResults(null);
        setWorkspaceSettings(null);
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
          workspaceSettingsPayload,
          approvalPayload,
          inventoryPayload,
          artifactPayload,
        ] = await Promise.all([
          fetchWorkspaceOverview(selectedWorkspaceId),
          fetchWorkspaceSettings(selectedWorkspaceId),
          fetchApprovals(selectedWorkspaceId),
          fetchWorkspaceInventory(selectedWorkspaceId),
          fetchWorkspaceArtifacts(selectedWorkspaceId),
        ]);
        if (cancelled) {
          return;
        }
        setOverview(overviewPayload);
        setApprovalInbox(approvalPayload);
        setInventory(inventoryPayload);
        setWorkspaceArtifacts(artifactPayload);
        setWorkspaceSearchResults(null);
        setWorkspaceSettings(workspaceSettingsPayload);
        setWorkspaces((current) => mergeWorkspaceSummary(current, overviewPayload.workspace));
        const firstConversationId = overviewPayload.recent_conversations[0]?.id || "";
        const firstRunId = overviewPayload.recent_runs[0]?.id || "";

        startTransition(() => {
          setSelectedConversationId((current) => {
            return current || firstConversationId;
          });
          setSelectedRunId((current) => {
            return current || firstRunId;
          });
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
        setNotifications((current) => {
          const deduped = current.filter((item) => item.id !== event.id);
          return [event, ...deduped].slice(0, 8);
        });
        scheduleNotificationRefresh(selectedWorkspaceId);
      },
      () => {
        scheduleNotificationRefresh(selectedWorkspaceId);
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
      cleanup();
    };
  }, [scheduleNotificationRefresh, selectedWorkspaceId]);

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
      await refreshWorkspaceList(nextArchived && !showArchivedWorkspaces ? "" : updated.id);
      if (!nextArchived || showArchivedWorkspaces) {
        await refreshWorkspaceSurface(updated.id);
      } else {
        setOverview(null);
        setWorkspaceSettings(null);
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
      await refreshWorkspaceList(updated.id);
      await refreshWorkspaceSurface(updated.id);
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
    refreshApprovalInbox,
  };
}
