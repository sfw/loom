import { useState } from "react";

import type {
  ApprovalFeedItem,
  ConversationApproval,
  ConversationDetail,
  ConversationMessage,
  ConversationPrompt,
  ConversationStatus,
  ConversationStreamEvent,
  ModelInfo,
  NotificationEvent,
  RunArtifact,
  RunConversationEntry,
  RunDetail,
  RunTimelineEvent,
  RuntimeStatus,
  SettingsPayload,
  WorkspaceArtifact,
  WorkspaceFileEntry,
  WorkspaceFilePreview,
  WorkspaceInventory,
  WorkspaceOverview,
  WorkspaceSearchResponse,
  WorkspaceSettingsPayload,
  WorkspaceSummary,
} from "../api";
import type { CommandOption, PaletteEntry } from "../shell";
import type { ViewTab } from "../utils";
import { useConnection } from "./useConnection";
import { useWorkspace } from "./useWorkspace";
import { useConversation } from "./useConversation";
import { useRuns } from "./useRuns";
import { useFiles } from "./useFiles";
import { useInbox } from "./useInbox";
import { useCommandPalette } from "./useCommandPalette";
import { useKeyboardShortcuts } from "./useKeyboardShortcuts";
import { useDesktopActivity, type DesktopActivityState } from "./useDesktopActivity";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface AppState {
  // Runtime & global
  runtime: RuntimeStatus | null;
  models: ModelInfo[];
  workspaces: WorkspaceSummary[];
  settings: SettingsPayload | null;
  runtimeManaged: boolean;
  connectionState: "connecting" | "connected" | "failed";
  desktopActivity: DesktopActivityState;

  // Workspace selection
  selectedWorkspaceId: string;
  showArchivedWorkspaces: boolean;
  workspaceNameDraft: string;
  workspaceTagsDraft: string;
  workspaceNoteDraft: string;
  workspaceFileTreeMode: "all" | "active" | "recent";
  workspaceImportFolderDraft: string;
  importingWorkspaceFiles: boolean;

  // Import / Create workspace
  importPath: string;
  importDisplayName: string;
  createParentPath: string;
  createFolderName: string;
  createDisplayName: string;
  importingWorkspace: boolean;
  creatingWorkspace: boolean;
  savingWorkspaceMeta: boolean;

  // Conversation selection
  selectedConversationId: string;
  selectedRunId: string;

  // Workspace search
  workspaceSearchQuery: string;
  workspaceFileFilterQuery: string;

  // Command palette
  commandDraft: string;
  commandPaletteOpen: boolean;
  activeCommandIndex: number;
  commandSearchResults: WorkspaceSearchResponse | null;
  searchingCommandPalette: boolean;
  recentPaletteEntries: PaletteEntry[];

  // Overview & inventory
  overview: WorkspaceOverview | null;
  inventory: WorkspaceInventory | null;
  loadingOverview: boolean;

  // File tree
  workspaceFilesByDirectory: Record<string, WorkspaceFileEntry[]>;
  expandedWorkspaceDirectories: string[];
  loadingWorkspaceDirectory: string;
  selectedWorkspaceFilePath: string;
  workspaceFilePreview: WorkspaceFilePreview | null;
  loadingWorkspaceFilePreview: boolean;
  workspaceFileEditorPath: string;
  workspaceFileEditorDraft: string;
  workspaceFileEditorDirty: boolean;
  savingWorkspaceFile: boolean;
  workspaceArtifacts: WorkspaceArtifact[];
  workspaceSearchResults: WorkspaceSearchResponse | null;
  searchingWorkspace: boolean;

  // Inbox
  approvalInbox: ApprovalFeedItem[];
  notifications: NotificationEvent[];
  approvalReplyDrafts: Record<string, string>;
  workspaceSettings: WorkspaceSettingsPayload | null;

  // Conversation detail
  conversationDetail: ConversationDetail | null;
  conversationMessages: ConversationMessage[];
  conversationEvents: ConversationStreamEvent[];
  conversationStatus: ConversationStatus | null;
  conversationStreaming: boolean;

  // Live streaming state
  streamingText: string;
  streamingThinking: string;
  streamingToolCalls: Array<{
    id: string;
    tool_name: string;
    started_at: string;
    completed: boolean;
    success?: boolean;
    args_preview?: string;
    output_preview?: string;
    elapsed_ms?: number;
  }>;
  lastTurnStats: { tokens: number; tool_count: number; visible: boolean } | null;

  // Message queue visualization
  queuedMessages: Array<{
    id: string;
    text: string;
    queuedAt: number;
    type: "inject" | "redirect" | "next";
  }>;

  // Run detail
  runDetail: RunDetail | null;
  runTimeline: RunTimelineEvent[];
  runArtifacts: RunArtifact[];
  runInstructionHistory: RunConversationEntry[];
  runStreaming: boolean;
  loadingRunDetail: boolean;
  runLoadError: string;

  // Messaging
  error: string;
  notice: string;

  // Conversation composer
  newConversationModel: string;
  newConversationPrompt: string;
  creatingConversation: boolean;
  conversationComposerMessage: string;
  sendingConversationMessage: boolean;
  conversationInjectMessage: string;
  sendingConversationInject: boolean;
  conversationTurnPending: boolean;
  conversationHistoryQuery: string;
  activeConversationMatchIndex: number;

  // Run composer
  runGoal: string;
  runProcess: string;
  runApprovalMode: string;
  launchingRun: boolean;
  runOperatorMessage: string;
  runActionPending: string;
  sendingRunMessage: boolean;
  runHistoryQuery: string;
  activeRunMatchIndex: number;

  // Approval reply
  replyingApprovalId: string;

  // NEW state: tab & new-workspace panel
  activeTab: ViewTab;
  showNewWorkspace: boolean;

  // Computed values
  normalizedRunStatus: string;
  runIsTerminal: boolean;
  runCanPause: boolean;
  runCanResume: boolean;
  runCanMessage: boolean;
  selectedWorkspaceSummary: WorkspaceSummary | null;
  selectedWorkspaceArchived: boolean;
  selectedWorkspacePinned: boolean;
  selectedWorkspaceTags: string[];
  selectedWorkspaceNote: string;
  conversationIsProcessing: boolean;
  pendingConversationApproval: ConversationApproval | null;
  conversationAwaitingApproval: boolean;
  pendingConversationPrompt: ConversationPrompt | null;
  conversationAwaitingInput: boolean;
  noWorkspacesRegistered: boolean;
  selectedWorkspaceIsEmpty: boolean;
  conversationPhaseLabel: string;
  quickReplyOptions: Array<{ id: string; label: string; description?: string }>;
  workspaceConversationRows: Array<{ id: string; workspace_id: string; workspace_path: string; model_name: string; title: string; turn_count: number; total_tokens: number; last_active_at: string; started_at: string; is_active: boolean; linked_run_ids: string[] }>;
  workspaceRunRows: Array<{ id: string; workspace_id: string; workspace_path: string; goal: string; status: string; created_at: string; updated_at: string; execution_run_id: string; process_name: string; linked_conversation_ids: string[]; changed_files_count: number }>;
  selectedConversationSummary: ConversationDetail | { id: string; title: string; model_name: string; last_active_at: string; started_at: string; linked_run_ids: string[] } | null;
  selectedRunSummary: RunDetail | { id: string; goal: string; status: string; created_at: string; updated_at: string; process_name: string; linked_conversation_ids: string[] } | null;
  settingsPreview: Array<{ path: string; section: string; field: string; description: string; configured_display: string; effective_display: string }>;
  filteredConversations: Array<{ id: string; title: string; model_name: string; last_active_at: string; started_at: string; linked_run_ids: string[] }>;
  filteredRuns: Array<{ id: string; goal: string; status: string; created_at: string; updated_at: string; process_name: string; linked_conversation_ids: string[] }>;
  filteredApprovalItems: ApprovalFeedItem[];
  filteredProcesses: Array<{ name: string; version: string; description: string; author: string; path: string }>;
  filteredMcpServers: Array<{ alias: string; type: string; enabled: boolean; source: string; command: string; url: string; cwd: string; timeout_seconds: number; oauth_enabled: boolean }>;
  filteredTools: Array<{ name: string; description: string; auth_mode: string; auth_required: boolean; execution_surfaces: string[] }>;
  filteredWorkspaceArtifacts: WorkspaceArtifact[];
  recentWorkspaceArtifacts: WorkspaceArtifact[];
  recentNotifications: NotificationEvent[];
  filteredConversationEvents: ConversationStreamEvent[];
  filteredConversationMessages: ConversationMessage[];
  visibleConversationEvents: ConversationStreamEvent[];
  visibleConversationMessages: ConversationMessage[];
  filteredRunArtifacts: RunArtifact[];
  filteredRunTimeline: RunTimelineEvent[];
  visibleRunArtifacts: RunArtifact[];
  visibleRunTimeline: RunTimelineEvent[];
  totalConversationMatches: number;
  totalRunMatches: number;
  trimmedCommandDraft: string;
  normalizedCommandDraft: string;
  commandSearchTerm: string;
  commandOptions: CommandOption[];
  pinnedPaletteEntries: PaletteEntry[];
  commandPaletteEntries: PaletteEntry[];
  resultPaletteEntries: PaletteEntry[];
  paletteEntries: PaletteEntry[];
  paletteSections: Array<{ label: string; entries: PaletteEntry[] }>;
  searchGroups: Array<{ label: string; rows: Array<{ kind: string; item_id: string; title: string; subtitle: string; snippet: string; badges: string[]; conversation_id: string; run_id: string; approval_item_id: string; path: string; metadata: Record<string, unknown> }> }>;
  normalizedWorkspaceFileFilterQuery: string;
  loadedWorkspaceFileEntries: WorkspaceFileEntry[];
  matchedWorkspaceDirectories: string[];
  locallyVisibleWorkspaceFilePaths: Set<string>;
  rootWorkspaceFiles: WorkspaceFileEntry[];
  selectedWorkspaceFileEntry: WorkspaceFileEntry | null;
  selectedWorkspaceFileIsEditable: boolean;
  selectedWorkspaceFileEditorHasChanges: boolean;
  selectedWorkspaceFileEditHint: string;
  selectedFileArtifactHistory: WorkspaceArtifact[];
  selectedFileLatestArtifact: WorkspaceArtifact | null;
  selectedFileRunIds: string[];
  selectedFileRelatedConversations: Array<{ id: string; title: string; linked_run_ids: string[] }>;
  selectedConversationRunIds: string[];
  contextualArtifacts: WorkspaceArtifact[];
  contextualFilePaths: Set<string>;
  contextualDirectoryCounts: Map<string, number>;
  recentFilePaths: Set<string>;
  recentDirectoryCounts: Map<string, number>;
  activeFilesLabel: string;
  visibleRootWorkspaceFiles: WorkspaceFileEntry[];

  // Refs
  searchInputRef: React.RefObject<HTMLInputElement | null>;
  commandInputRef: React.RefObject<HTMLInputElement | null>;
  inboxSectionRef: React.RefObject<HTMLElement | null>;
  conversationComposerRef: React.RefObject<HTMLElement | null>;
  runComposerRef: React.RefObject<HTMLElement | null>;
  workspaceFileInputRef: React.RefObject<HTMLInputElement | null>;
  conversationMatchRefs: React.MutableRefObject<Array<HTMLDivElement | null>>;
  runMatchRefs: React.MutableRefObject<Array<HTMLDivElement | null>>;
}

export interface AppActions {
  // Setters
  setSelectedWorkspaceId: React.Dispatch<React.SetStateAction<string>>;
  setShowArchivedWorkspaces: React.Dispatch<React.SetStateAction<boolean>>;
  setWorkspaceNameDraft: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceTagsDraft: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceNoteDraft: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceFileTreeMode: React.Dispatch<React.SetStateAction<"all" | "active" | "recent">>;
  setWorkspaceImportFolderDraft: React.Dispatch<React.SetStateAction<string>>;
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
  setSelectedRunId: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceSearchQuery: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceFileFilterQuery: React.Dispatch<React.SetStateAction<string>>;
  setCommandDraft: React.Dispatch<React.SetStateAction<string>>;
  setCommandPaletteOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setActiveCommandIndex: React.Dispatch<React.SetStateAction<number>>;
  setImportPath: React.Dispatch<React.SetStateAction<string>>;
  setImportDisplayName: React.Dispatch<React.SetStateAction<string>>;
  setCreateParentPath: React.Dispatch<React.SetStateAction<string>>;
  setCreateFolderName: React.Dispatch<React.SetStateAction<string>>;
  setCreateDisplayName: React.Dispatch<React.SetStateAction<string>>;
  setNewConversationModel: React.Dispatch<React.SetStateAction<string>>;
  setNewConversationPrompt: React.Dispatch<React.SetStateAction<string>>;
  setConversationComposerMessage: React.Dispatch<React.SetStateAction<string>>;
  setConversationInjectMessage: React.Dispatch<React.SetStateAction<string>>;
  setQueuedMessages: React.Dispatch<React.SetStateAction<Array<{
    id: string;
    text: string;
    queuedAt: number;
    type: "inject" | "redirect" | "next";
  }>>>;
  editQueuedMessage: (queueId: string) => void;
  cancelQueuedMessage: (queueId: string) => void;
  setConversationHistoryQuery: React.Dispatch<React.SetStateAction<string>>;
  setActiveConversationMatchIndex: React.Dispatch<React.SetStateAction<number>>;
  setRunGoal: React.Dispatch<React.SetStateAction<string>>;
  setRunProcess: React.Dispatch<React.SetStateAction<string>>;
  setRunApprovalMode: React.Dispatch<React.SetStateAction<string>>;
  setRunOperatorMessage: React.Dispatch<React.SetStateAction<string>>;
  setRunHistoryQuery: React.Dispatch<React.SetStateAction<string>>;
  setActiveRunMatchIndex: React.Dispatch<React.SetStateAction<number>>;
  setApprovalReplyDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  setError: React.Dispatch<React.SetStateAction<string>>;
  retryConnection: () => void;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
  setSelectedWorkspaceFilePath: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceFileEditorDraft: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceFileEditorDirty: React.Dispatch<React.SetStateAction<boolean>>;
  setActiveTab: React.Dispatch<React.SetStateAction<ViewTab>>;
  setShowNewWorkspace: React.Dispatch<React.SetStateAction<boolean>>;

  // Handlers
  handleImportWorkspace: (event: import("react").FormEvent<HTMLFormElement>) => Promise<void>;
  handleCreateWorkspace: (event: import("react").FormEvent<HTMLFormElement>) => Promise<void>;
  handleSaveWorkspaceDetails: (event: import("react").FormEvent<HTMLFormElement>) => Promise<void>;
  handleArchiveWorkspace: (nextArchived: boolean) => Promise<void>;
  handlePinWorkspace: (nextPinned: boolean) => Promise<void>;
  handleImportWorkspaceFiles: (files: FileList | null) => Promise<void>;
  handlePrefillStarterWorkspace: () => void;
  handlePrefillStarterConversation: () => void;
  handlePrefillStarterRun: () => void;
  handleCreateConversation: (event: import("react").FormEvent<HTMLFormElement>) => Promise<void>;
  handleLaunchRun: (
    event: import("react").FormEvent<HTMLFormElement>,
    extraContext?: Record<string, unknown>,
  ) => Promise<void>;
  handleRunControl: (action: "pause" | "resume" | "cancel") => Promise<void>;
  handleDeleteRun: () => Promise<void>;
  handleRestartRun: () => Promise<void>;
  handleSendRunMessage: (event: import("react").FormEvent<HTMLFormElement>) => Promise<void>;
  handleSendConversationMessage: (event: import("react").FormEvent<HTMLFormElement>) => Promise<void>;
  submitConversationMessage: (rawMessage: string) => Promise<void>;
  handleQuickConversationReply: (optionLabel: string) => Promise<void>;
  handleInjectConversationInstruction: (event: import("react").FormEvent<HTMLFormElement>) => Promise<void>;
  handleResolveConversationApproval: (decision: "approve" | "approve_all" | "deny") => Promise<void>;
  handleReplyApproval: (
    item: ApprovalFeedItem,
    body: {
      decision: string;
      reason?: string;
      response_type?: string;
      selected_option_ids?: string[];
      selected_labels?: string[];
      custom_response?: string;
    },
  ) => Promise<void>;
  handleSelectApprovalContext: (item: ApprovalFeedItem) => void;
  handleSearchResultSelection: (result: {
    conversation_id?: string;
    run_id?: string;
    approval_item_id?: string;
    path?: string;
    kind?: string;
  }) => void;
  handleOpenWorkspaceFile: (path: string) => Promise<void>;
  handleOpenWorkspaceFileExternally: () => Promise<void>;
  handleRevealWorkspaceFile: () => Promise<void>;
  handleSaveWorkspaceFile: () => Promise<void>;
  handleResetWorkspaceFileEditor: () => void;
  handleExpandActiveWorkspaceFiles: () => Promise<void>;
  handleExpandRecentWorkspaceFiles: () => Promise<void>;
  toggleWorkspaceDirectory: (path: string) => void;
  handleWorkspaceFileSelection: (entry: WorkspaceFileEntry) => void;
  handleStopConversationTurn: () => Promise<void>;
  focusSearch: () => void;
  focusCommandBar: () => void;
  focusConversationComposer: () => void;
  focusRunComposer: () => void;
  handleCommandAction: (rawCommand: string) => void;
  executeCommandOption: (option: CommandOption) => void;
  executePaletteEntry: (entry: PaletteEntry) => void;
  handleCommandInputKeyDown: (event: React.KeyboardEvent<HTMLInputElement>) => void;
  handleCommandSubmit: (event: import("react").FormEvent<HTMLFormElement>) => void;
  rememberPaletteEntry: (entry: PaletteEntry) => void;
  scrollConversationMatchIntoView: (index: number) => void;
  scrollRunMatchIntoView: (index: number) => void;
  stepConversationMatch: (delta: number) => void;
  stepRunMatch: (delta: number) => void;
  refreshWorkspaceList: (preferredWorkspaceId?: string) => Promise<void>;
  refreshWorkspaceSurface: (workspaceId: string) => Promise<void>;
  refreshApprovalInbox: (workspaceId: string) => Promise<void>;
  refreshConversation: (conversationId: string) => Promise<void>;
  loadOlderMessages: () => Promise<void>;
  hasOlderMessages: boolean;
  loadingOlderMessages: boolean;
  refreshRun: (runId: string) => Promise<void>;
  loadWorkspaceDirectory: (workspaceId: string, directory?: string) => Promise<void>;
}

// ---------------------------------------------------------------------------
// The hook — thin orchestrator
// ---------------------------------------------------------------------------

export function useAppState(): AppState & AppActions {
  // -----------------------------------------------------------------------
  // Hoisted cross-cutting state (shared across multiple sub-hooks)
  // -----------------------------------------------------------------------
  const [activeTab, setActiveTab] = useState<ViewTab>("overview");
  const [showNewWorkspace, setShowNewWorkspace] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState("");
  const [selectedConversationId, setSelectedConversationId] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [showArchivedWorkspaces, setShowArchivedWorkspaces] = useState(false);
  const [createParentPath, setCreateParentPath] = useState("");

  // -----------------------------------------------------------------------
  // 1. Connection (standalone — bootstrap, runtime, models, settings)
  // -----------------------------------------------------------------------
  const connection = useConnection({
    setError,
    showArchivedWorkspaces,
    selectedWorkspaceId,
    setSelectedWorkspaceId,
    setCreateParentPath,
  });

  // -----------------------------------------------------------------------
  // 2. Workspace (overview, inventory, search, CRUD, notifications)
  // -----------------------------------------------------------------------
  const workspace = useWorkspace({
    selectedWorkspaceId,
    selectedConversationId,
    selectedRunId,
    setSelectedWorkspaceId,
    showArchivedWorkspaces,
    setShowArchivedWorkspaces,
    createParentPath,
    setCreateParentPath,
    workspaces: connection.workspaces,
    setWorkspaces: connection.setWorkspaces,
    runtime: connection.runtime,
    setError,
    setNotice,
    activeTab,
    setActiveTab,
    setSelectedConversationId,
    setSelectedRunId,
  });

  // -----------------------------------------------------------------------
  // 3. Conversation (detail, messages, events, SSE, streaming, composer)
  // -----------------------------------------------------------------------
  const conversation = useConversation({
    selectedConversationId,
    setSelectedConversationId,
    selectedWorkspaceId,
    overview: workspace.overview,
    models: connection.models,
    setError,
    setNotice,
    setActiveTab,
    refreshWorkspaceSurface: workspace.refreshWorkspaceSurface,
  });

  // -----------------------------------------------------------------------
  // 4. Runs (detail, timeline, artifacts, SSE, controls, launcher)
  // -----------------------------------------------------------------------
  const runs = useRuns({
    selectedRunId,
    setSelectedRunId,
    selectedWorkspaceId,
    overview: workspace.overview,
    setError,
    setNotice,
    setActiveTab,
    refreshWorkspaceSurface: workspace.refreshWorkspaceSurface,
  });

  const desktopActivity = useDesktopActivity({
    connectionState: connection.connectionState,
    conversationTurnPending: conversation.conversationTurnPending,
    conversationStreaming: conversation.conversationStreaming,
    streamingToolCalls: conversation.streamingToolCalls,
    runStreaming: runs.runStreaming,
  });

  // -----------------------------------------------------------------------
  // 5. Files (file tree, preview, editor, import)
  // -----------------------------------------------------------------------
  const files = useFiles({
    selectedWorkspaceId,
    selectedWorkspaceSummary: workspace.selectedWorkspaceSummary,
    workspaceArtifacts: workspace.workspaceArtifacts,
    recentWorkspaceArtifacts: workspace.recentWorkspaceArtifacts,
    workspaceConversationRows: workspace.workspaceConversationRows,
    selectedConversationRunIds: conversation.selectedConversationRunIds,
    selectedRunId,
    runArtifacts: runs.runArtifacts,
    selectedConversationSummary: conversation.selectedConversationSummary,
    workspaceFileTreeMode: workspace.workspaceFileTreeMode,
    workspaceImportFolderDraft: workspace.workspaceImportFolderDraft,
    setError,
    setNotice,
  });

  // -----------------------------------------------------------------------
  // 6. Inbox (approval items, reply drafts, handlers)
  // -----------------------------------------------------------------------
  const inbox = useInbox({
    selectedWorkspaceId,
    selectedConversationId,
    selectedRunId,
    setSelectedWorkspaceId,
    setSelectedConversationId,
    setSelectedRunId,
    setActiveTab,
    setRunProcess: runs.setRunProcess,
    setError,
    setNotice,
    refreshWorkspaceSurface: workspace.refreshWorkspaceSurface,
    refreshApprovalInbox: workspace.refreshApprovalInbox,
    refreshConversation: conversation.refreshConversation,
    refreshRun: runs.refreshRun,
    queueWorkspaceFileOpen: files.queueWorkspaceFileOpen,
    focusRunComposer: runs.focusRunComposer,
  });

  // -----------------------------------------------------------------------
  // 7. Command palette (command draft, entries, search, navigation)
  // -----------------------------------------------------------------------
  const palette = useCommandPalette({
    selectedWorkspaceId,
    overview: workspace.overview,
    setSelectedConversationId,
    setSelectedRunId,
    setWorkspaceSearchQuery: workspace.setWorkspaceSearchQuery,
    setActiveTab,
    setError,
    setNotice,
    focusSearch: workspace.focusSearch,
    focusConversationComposer: conversation.focusConversationComposer,
    focusRunComposer: runs.focusRunComposer,
    handlePrefillStarterConversation: conversation.handlePrefillStarterConversation,
    handlePrefillStarterRun: runs.handlePrefillStarterRun,
    handleSearchResultSelection: inbox.handleSearchResultSelection,
  });

  // -----------------------------------------------------------------------
  // 8. Keyboard shortcuts (global keyboard handler)
  // -----------------------------------------------------------------------
  useKeyboardShortcuts({
    overview: workspace.overview,
    workspaces: connection.workspaces,
    selectedWorkspaceId,
    selectedConversationId,
    selectedRunId,
    workspaceFilePreview: files.workspaceFilePreview,
    workspaceFileEditorDraft: files.workspaceFileEditorDraft,
    setSelectedWorkspaceId,
    setSelectedConversationId,
    setSelectedRunId,
    setActiveTab,
    focusSearch: workspace.focusSearch,
    focusCommandBar: palette.focusCommandBar,
    handleSaveWorkspaceFile: files.handleSaveWorkspaceFile,
  });

  // -----------------------------------------------------------------------
  // Return — composed from all hooks
  // -----------------------------------------------------------------------
  return {
    // Hoisted orchestrator state
    activeTab,
    setActiveTab,
    showNewWorkspace,
    setShowNewWorkspace,
    error,
    setError,
    notice,
    setNotice,
    selectedWorkspaceId,
    setSelectedWorkspaceId,
    selectedConversationId,
    setSelectedConversationId,
    selectedRunId,
    setSelectedRunId,
    showArchivedWorkspaces,
    setShowArchivedWorkspaces,
    createParentPath,
    setCreateParentPath,

    // Connection
    runtime: connection.runtime,
    models: connection.models,
    workspaces: connection.workspaces,
    settings: connection.settings,
    runtimeManaged: connection.runtimeManaged,
    connectionState: connection.connectionState,
    desktopActivity,
    settingsPreview: connection.settingsPreview,
    retryConnection: connection.retryConnection,

    // Workspace
    workspaceNameDraft: workspace.workspaceNameDraft,
    setWorkspaceNameDraft: workspace.setWorkspaceNameDraft,
    workspaceTagsDraft: workspace.workspaceTagsDraft,
    setWorkspaceTagsDraft: workspace.setWorkspaceTagsDraft,
    workspaceNoteDraft: workspace.workspaceNoteDraft,
    setWorkspaceNoteDraft: workspace.setWorkspaceNoteDraft,
    workspaceFileTreeMode: workspace.workspaceFileTreeMode,
    setWorkspaceFileTreeMode: workspace.setWorkspaceFileTreeMode,
    workspaceImportFolderDraft: workspace.workspaceImportFolderDraft,
    setWorkspaceImportFolderDraft: workspace.setWorkspaceImportFolderDraft,
    importPath: workspace.importPath,
    setImportPath: workspace.setImportPath,
    importDisplayName: workspace.importDisplayName,
    setImportDisplayName: workspace.setImportDisplayName,
    createFolderName: workspace.createFolderName,
    setCreateFolderName: workspace.setCreateFolderName,
    createDisplayName: workspace.createDisplayName,
    setCreateDisplayName: workspace.setCreateDisplayName,
    importingWorkspace: workspace.importingWorkspace,
    creatingWorkspace: workspace.creatingWorkspace,
    savingWorkspaceMeta: workspace.savingWorkspaceMeta,
    workspaceSearchQuery: workspace.workspaceSearchQuery,
    setWorkspaceSearchQuery: workspace.setWorkspaceSearchQuery,
    overview: workspace.overview,
    inventory: workspace.inventory,
    loadingOverview: workspace.loadingOverview,
    approvalInbox: workspace.approvalInbox,
    notifications: workspace.notifications,
    workspaceSettings: workspace.workspaceSettings,
    workspaceArtifacts: workspace.workspaceArtifacts,
    workspaceSearchResults: workspace.workspaceSearchResults,
    searchingWorkspace: workspace.searchingWorkspace,
    selectedWorkspaceSummary: workspace.selectedWorkspaceSummary,
    selectedWorkspaceArchived: workspace.selectedWorkspaceArchived,
    selectedWorkspacePinned: workspace.selectedWorkspacePinned,
    selectedWorkspaceTags: workspace.selectedWorkspaceTags,
    selectedWorkspaceNote: workspace.selectedWorkspaceNote,
    noWorkspacesRegistered: workspace.noWorkspacesRegistered,
    selectedWorkspaceIsEmpty: workspace.selectedWorkspaceIsEmpty,
    workspaceConversationRows: workspace.workspaceConversationRows,
    workspaceRunRows: workspace.workspaceRunRows,
    filteredConversations: workspace.filteredConversations,
    filteredRuns: workspace.filteredRuns,
    filteredApprovalItems: workspace.filteredApprovalItems,
    filteredProcesses: workspace.filteredProcesses,
    filteredMcpServers: workspace.filteredMcpServers,
    filteredTools: workspace.filteredTools,
    filteredWorkspaceArtifacts: workspace.filteredWorkspaceArtifacts,
    recentWorkspaceArtifacts: workspace.recentWorkspaceArtifacts,
    recentNotifications: workspace.recentNotifications,
    searchGroups: workspace.searchGroups,
    searchInputRef: workspace.searchInputRef,
    inboxSectionRef: workspace.inboxSectionRef,
    handleImportWorkspace: workspace.handleImportWorkspace,
    handleCreateWorkspace: workspace.handleCreateWorkspace,
    handleSaveWorkspaceDetails: workspace.handleSaveWorkspaceDetails,
    handleArchiveWorkspace: workspace.handleArchiveWorkspace,
    handlePinWorkspace: workspace.handlePinWorkspace,
    handlePrefillStarterWorkspace: workspace.handlePrefillStarterWorkspace,
    focusSearch: workspace.focusSearch,
    refreshWorkspaceList: workspace.refreshWorkspaceList,
    refreshWorkspaceSurface: workspace.refreshWorkspaceSurface,
    refreshApprovalInbox: workspace.refreshApprovalInbox,

    // Conversation
    conversationDetail: conversation.conversationDetail,
    conversationMessages: conversation.conversationMessages,
    conversationEvents: conversation.conversationEvents,
    conversationStatus: conversation.conversationStatus,
    conversationStreaming: conversation.conversationStreaming,
    streamingText: conversation.streamingText,
    streamingThinking: conversation.streamingThinking,
    streamingToolCalls: conversation.streamingToolCalls,
    lastTurnStats: conversation.lastTurnStats,
    queuedMessages: conversation.queuedMessages,
    setQueuedMessages: conversation.setQueuedMessages,
    editQueuedMessage: conversation.editQueuedMessage,
    cancelQueuedMessage: conversation.cancelQueuedMessage,
    newConversationModel: conversation.newConversationModel,
    setNewConversationModel: conversation.setNewConversationModel,
    newConversationPrompt: conversation.newConversationPrompt,
    setNewConversationPrompt: conversation.setNewConversationPrompt,
    creatingConversation: conversation.creatingConversation,
    conversationComposerMessage: conversation.conversationComposerMessage,
    setConversationComposerMessage: conversation.setConversationComposerMessage,
    sendingConversationMessage: conversation.sendingConversationMessage,
    conversationInjectMessage: conversation.conversationInjectMessage,
    setConversationInjectMessage: conversation.setConversationInjectMessage,
    sendingConversationInject: conversation.sendingConversationInject,
    conversationTurnPending: conversation.conversationTurnPending,
    conversationHistoryQuery: conversation.conversationHistoryQuery,
    setConversationHistoryQuery: conversation.setConversationHistoryQuery,
    activeConversationMatchIndex: conversation.activeConversationMatchIndex,
    setActiveConversationMatchIndex: conversation.setActiveConversationMatchIndex,
    conversationIsProcessing: conversation.conversationIsProcessing,
    pendingConversationApproval: conversation.pendingConversationApproval,
    conversationAwaitingApproval: conversation.conversationAwaitingApproval,
    pendingConversationPrompt: conversation.pendingConversationPrompt,
    conversationAwaitingInput: conversation.conversationAwaitingInput,
    conversationPhaseLabel: conversation.conversationPhaseLabel,
    quickReplyOptions: conversation.quickReplyOptions,
    selectedConversationSummary: conversation.selectedConversationSummary,
    selectedConversationRunIds: conversation.selectedConversationRunIds,
    filteredConversationEvents: conversation.filteredConversationEvents,
    filteredConversationMessages: conversation.filteredConversationMessages,
    visibleConversationEvents: conversation.visibleConversationEvents,
    visibleConversationMessages: conversation.visibleConversationMessages,
    totalConversationMatches: conversation.totalConversationMatches,
    conversationComposerRef: conversation.conversationComposerRef,
    conversationMatchRefs: conversation.conversationMatchRefs,
    handleCreateConversation: conversation.handleCreateConversation,
    handleSendConversationMessage: conversation.handleSendConversationMessage,
    submitConversationMessage: conversation.submitConversationMessage,
    handleQuickConversationReply: conversation.handleQuickConversationReply,
    handleInjectConversationInstruction: conversation.handleInjectConversationInstruction,
    handleResolveConversationApproval: conversation.handleResolveConversationApproval,
    handleStopConversationTurn: conversation.handleStopConversationTurn,
    handlePrefillStarterConversation: conversation.handlePrefillStarterConversation,
    focusConversationComposer: conversation.focusConversationComposer,
    refreshConversation: conversation.refreshConversation,
    loadOlderMessages: conversation.loadOlderMessages,
    hasOlderMessages: conversation.hasOlderMessages,
    loadingOlderMessages: conversation.loadingOlderMessages,
    scrollConversationMatchIntoView: conversation.scrollConversationMatchIntoView,
    stepConversationMatch: conversation.stepConversationMatch,

    // Runs
    runDetail: runs.runDetail,
    runTimeline: runs.runTimeline,
    runArtifacts: runs.runArtifacts,
    runInstructionHistory: runs.runInstructionHistory,
    runStreaming: runs.runStreaming,
    loadingRunDetail: runs.loadingRunDetail,
    runLoadError: runs.runLoadError,
    runGoal: runs.runGoal,
    setRunGoal: runs.setRunGoal,
    runProcess: runs.runProcess,
    setRunProcess: runs.setRunProcess,
    runApprovalMode: runs.runApprovalMode,
    setRunApprovalMode: runs.setRunApprovalMode,
    launchingRun: runs.launchingRun,
    runOperatorMessage: runs.runOperatorMessage,
    setRunOperatorMessage: runs.setRunOperatorMessage,
    runActionPending: runs.runActionPending,
    sendingRunMessage: runs.sendingRunMessage,
    runHistoryQuery: runs.runHistoryQuery,
    setRunHistoryQuery: runs.setRunHistoryQuery,
    activeRunMatchIndex: runs.activeRunMatchIndex,
    setActiveRunMatchIndex: runs.setActiveRunMatchIndex,
    normalizedRunStatus: runs.normalizedRunStatus,
    runIsTerminal: runs.runIsTerminal,
    runCanPause: runs.runCanPause,
    runCanResume: runs.runCanResume,
    runCanMessage: runs.runCanMessage,
    selectedRunSummary: runs.selectedRunSummary,
    filteredRunArtifacts: runs.filteredRunArtifacts,
    filteredRunTimeline: runs.filteredRunTimeline,
    visibleRunArtifacts: runs.visibleRunArtifacts,
    visibleRunTimeline: runs.visibleRunTimeline,
    totalRunMatches: runs.totalRunMatches,
    runComposerRef: runs.runComposerRef,
    runMatchRefs: runs.runMatchRefs,
    handleLaunchRun: runs.handleLaunchRun,
    handleRunControl: runs.handleRunControl,
    handleDeleteRun: runs.handleDeleteRun,
    handleRestartRun: runs.handleRestartRun,
    handleSendRunMessage: runs.handleSendRunMessage,
    handlePrefillStarterRun: runs.handlePrefillStarterRun,
    focusRunComposer: runs.focusRunComposer,
    refreshRun: runs.refreshRun,
    scrollRunMatchIntoView: runs.scrollRunMatchIntoView,
    stepRunMatch: runs.stepRunMatch,

    // Files
    workspaceFilesByDirectory: files.workspaceFilesByDirectory,
    expandedWorkspaceDirectories: files.expandedWorkspaceDirectories,
    loadingWorkspaceDirectory: files.loadingWorkspaceDirectory,
    selectedWorkspaceFilePath: files.selectedWorkspaceFilePath,
    setSelectedWorkspaceFilePath: files.setSelectedWorkspaceFilePath,
    workspaceFilePreview: files.workspaceFilePreview,
    loadingWorkspaceFilePreview: files.loadingWorkspaceFilePreview,
    workspaceFileEditorPath: files.workspaceFileEditorPath,
    workspaceFileEditorDraft: files.workspaceFileEditorDraft,
    setWorkspaceFileEditorDraft: files.setWorkspaceFileEditorDraft,
    workspaceFileEditorDirty: files.workspaceFileEditorDirty,
    setWorkspaceFileEditorDirty: files.setWorkspaceFileEditorDirty,
    savingWorkspaceFile: files.savingWorkspaceFile,
    workspaceFileFilterQuery: files.workspaceFileFilterQuery,
    setWorkspaceFileFilterQuery: files.setWorkspaceFileFilterQuery,
    importingWorkspaceFiles: files.importingWorkspaceFiles,
    normalizedWorkspaceFileFilterQuery: files.normalizedWorkspaceFileFilterQuery,
    loadedWorkspaceFileEntries: files.loadedWorkspaceFileEntries,
    matchedWorkspaceDirectories: files.matchedWorkspaceDirectories,
    locallyVisibleWorkspaceFilePaths: files.locallyVisibleWorkspaceFilePaths,
    rootWorkspaceFiles: files.rootWorkspaceFiles,
    selectedWorkspaceFileEntry: files.selectedWorkspaceFileEntry,
    selectedWorkspaceFileIsEditable: files.selectedWorkspaceFileIsEditable,
    selectedWorkspaceFileEditorHasChanges: files.selectedWorkspaceFileEditorHasChanges,
    selectedWorkspaceFileEditHint: files.selectedWorkspaceFileEditHint,
    selectedFileArtifactHistory: files.selectedFileArtifactHistory,
    selectedFileLatestArtifact: files.selectedFileLatestArtifact,
    selectedFileRunIds: files.selectedFileRunIds,
    selectedFileRelatedConversations: files.selectedFileRelatedConversations,
    contextualArtifacts: files.contextualArtifacts,
    contextualFilePaths: files.contextualFilePaths,
    contextualDirectoryCounts: files.contextualDirectoryCounts,
    recentFilePaths: files.recentFilePaths,
    recentDirectoryCounts: files.recentDirectoryCounts,
    activeFilesLabel: files.activeFilesLabel,
    visibleRootWorkspaceFiles: files.visibleRootWorkspaceFiles,
    workspaceFileInputRef: files.workspaceFileInputRef,
    handleOpenWorkspaceFile: files.handleOpenWorkspaceFile,
    handleOpenWorkspaceFileExternally: files.handleOpenWorkspaceFileExternally,
    handleRevealWorkspaceFile: files.handleRevealWorkspaceFile,
    handleSaveWorkspaceFile: files.handleSaveWorkspaceFile,
    handleResetWorkspaceFileEditor: files.handleResetWorkspaceFileEditor,
    handleExpandActiveWorkspaceFiles: files.handleExpandActiveWorkspaceFiles,
    handleExpandRecentWorkspaceFiles: files.handleExpandRecentWorkspaceFiles,
    toggleWorkspaceDirectory: files.toggleWorkspaceDirectory,
    handleWorkspaceFileSelection: files.handleWorkspaceFileSelection,
    handleImportWorkspaceFiles: files.handleImportWorkspaceFiles,
    loadWorkspaceDirectory: files.loadWorkspaceDirectory,

    // Inbox
    approvalReplyDrafts: inbox.approvalReplyDrafts,
    setApprovalReplyDrafts: inbox.setApprovalReplyDrafts,
    replyingApprovalId: inbox.replyingApprovalId,
    handleReplyApproval: inbox.handleReplyApproval,
    handleSelectApprovalContext: inbox.handleSelectApprovalContext,
    handleSearchResultSelection: inbox.handleSearchResultSelection,

    // Command palette
    commandDraft: palette.commandDraft,
    setCommandDraft: palette.setCommandDraft,
    commandPaletteOpen: palette.commandPaletteOpen,
    setCommandPaletteOpen: palette.setCommandPaletteOpen,
    activeCommandIndex: palette.activeCommandIndex,
    setActiveCommandIndex: palette.setActiveCommandIndex,
    commandSearchResults: palette.commandSearchResults,
    searchingCommandPalette: palette.searchingCommandPalette,
    recentPaletteEntries: palette.recentPaletteEntries,
    trimmedCommandDraft: palette.trimmedCommandDraft,
    normalizedCommandDraft: palette.normalizedCommandDraft,
    commandSearchTerm: palette.commandSearchTerm,
    commandOptions: palette.commandOptions,
    pinnedPaletteEntries: palette.pinnedPaletteEntries,
    commandPaletteEntries: palette.commandPaletteEntries,
    resultPaletteEntries: palette.resultPaletteEntries,
    paletteEntries: palette.paletteEntries,
    paletteSections: palette.paletteSections,
    commandInputRef: palette.commandInputRef,
    focusCommandBar: palette.focusCommandBar,
    handleCommandAction: palette.handleCommandAction,
    executeCommandOption: palette.executeCommandOption,
    executePaletteEntry: palette.executePaletteEntry,
    handleCommandInputKeyDown: palette.handleCommandInputKeyDown,
    handleCommandSubmit: palette.handleCommandSubmit,
    rememberPaletteEntry: palette.rememberPaletteEntry,
  };
}
