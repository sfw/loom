import {
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

import {
  createWorkspaceFile,
  fetchWorkspaceFilePreview,
  fetchWorkspaceFiles,
  importWorkspaceFiles,
  openWorkspaceFile,
  revealWorkspaceFile,
  type RunArtifact,
  type WorkspaceArtifact,
  type WorkspaceFileEntry,
  type WorkspaceFilePreview,
  type WorkspaceSummary,
} from "../api";
import { matchesWorkspaceSearch } from "../history";
import {
  ancestorDirectories,
  fileSortKey,
} from "../utils";

function isNotFoundRequestError(error: unknown): boolean {
  const message = String(error instanceof Error ? error.message : error || "")
    .trim()
    .toLowerCase();
  if (!message) {
    return false;
  }
  return message === "404"
    || message.startsWith("404 ")
    || message.includes(" not found");
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface FilesState {
  workspaceFilesByDirectory: Record<string, WorkspaceFileEntry[]>;
  expandedWorkspaceDirectories: string[];
  loadingWorkspaceDirectory: string;
  refreshingWorkspaceFiles: boolean;
  selectedWorkspaceFilePath: string;
  workspaceFilePreview: WorkspaceFilePreview | null;
  loadingWorkspaceFilePreview: boolean;
  workspaceFileEditorPath: string;
  workspaceFileEditorDraft: string;
  workspaceFileEditorDirty: boolean;
  savingWorkspaceFile: boolean;
  workspaceFileFilterQuery: string;
  importingWorkspaceFiles: boolean;

  // Computed
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
  contextualArtifacts: WorkspaceArtifact[];
  contextualFilePaths: Set<string>;
  contextualDirectoryCounts: Map<string, number>;
  recentFilePaths: Set<string>;
  recentDirectoryCounts: Map<string, number>;
  activeFilesLabel: string;
  visibleRootWorkspaceFiles: WorkspaceFileEntry[];

  // Refs
  workspaceFileInputRef: React.RefObject<HTMLInputElement | null>;
}

export interface FilesActions {
  setSelectedWorkspaceFilePath: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceFileEditorDraft: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceFileEditorDirty: React.Dispatch<React.SetStateAction<boolean>>;
  setWorkspaceFileFilterQuery: React.Dispatch<React.SetStateAction<string>>;
  handleOpenWorkspaceFile: (path: string) => Promise<void>;
  queueWorkspaceFileOpen: (workspaceId: string, path: string) => void;
  handleOpenWorkspaceFileExternally: () => Promise<void>;
  handleRevealWorkspaceFile: () => Promise<void>;
  handleSaveWorkspaceFile: () => Promise<void>;
  handleResetWorkspaceFileEditor: () => void;
  handleRefreshWorkspaceFiles: () => Promise<void>;
  handleExpandActiveWorkspaceFiles: () => Promise<void>;
  handleExpandRecentWorkspaceFiles: () => Promise<void>;
  toggleWorkspaceDirectory: (path: string) => void;
  handleWorkspaceFileSelection: (entry: WorkspaceFileEntry) => void;
  handleImportWorkspaceFiles: (files: FileList | null) => Promise<void>;
  loadWorkspaceDirectory: (workspaceId: string, directory?: string) => Promise<void>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useFiles(deps: {
  selectedWorkspaceId: string;
  selectedWorkspaceSummary: WorkspaceSummary | null;
  workspaceArtifacts: WorkspaceArtifact[];
  recentWorkspaceArtifacts: WorkspaceArtifact[];
  workspaceConversationRows: Array<{ id: string; title: string; linked_run_ids: string[] }>;
  selectedConversationRunIds: string[];
  selectedRunId: string;
  runArtifacts: RunArtifact[];
  selectedConversationSummary: { linked_run_ids?: string[] } | null;
  workspaceFileTreeMode: "all" | "active" | "recent";
  workspaceImportFolderDraft: string;
  setError: React.Dispatch<React.SetStateAction<string>>;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
}): FilesState & FilesActions {
  const {
    selectedWorkspaceId,
    selectedWorkspaceSummary,
    workspaceArtifacts,
    recentWorkspaceArtifacts,
    workspaceConversationRows,
    selectedConversationRunIds,
    selectedRunId,
    runArtifacts,
    selectedConversationSummary,
    workspaceFileTreeMode,
    workspaceImportFolderDraft,
    setError,
    setNotice,
  } = deps;

  // State
  const [workspaceFilesByDirectory, setWorkspaceFilesByDirectory] =
    useState<Record<string, WorkspaceFileEntry[]>>({});
  const [expandedWorkspaceDirectories, setExpandedWorkspaceDirectories] = useState<string[]>([""]);
  const [loadingWorkspaceDirectory, setLoadingWorkspaceDirectory] = useState("");
  const [refreshingWorkspaceFiles, setRefreshingWorkspaceFiles] = useState(false);
  const [selectedWorkspaceFilePath, setSelectedWorkspaceFilePath] = useState("");
  const [workspaceFilePreview, setWorkspaceFilePreview] =
    useState<WorkspaceFilePreview | null>(null);
  const [loadingWorkspaceFilePreview, setLoadingWorkspaceFilePreview] = useState(false);
  const [workspaceFileEditorPath, setWorkspaceFileEditorPath] = useState("");
  const [workspaceFileEditorDraft, setWorkspaceFileEditorDraft] = useState("");
  const [workspaceFileEditorDirty, setWorkspaceFileEditorDirty] = useState(false);
  const [savingWorkspaceFile, setSavingWorkspaceFile] = useState(false);
  const [workspaceFileFilterQuery, setWorkspaceFileFilterQuery] = useState("");
  const [importingWorkspaceFiles, setImportingWorkspaceFiles] = useState(false);
  const [pendingWorkspaceFileOpen, setPendingWorkspaceFileOpen] = useState<{
    workspaceId: string;
    path: string;
  } | null>(null);

  // Refs
  const workspaceFileInputRef = useRef<HTMLInputElement | null>(null);

  // ---------------------------------------------------------------------------
  // useEffectEvent handlers
  // ---------------------------------------------------------------------------

  async function readWorkspaceDirectory(workspaceId: string, directory = "") {
    const rows = await fetchWorkspaceFiles(workspaceId, directory);
    return rows
      .slice()
      .sort((left, right) => fileSortKey(left).localeCompare(fileSortKey(right)));
  }

  const loadWorkspaceDirectory = useEffectEvent(async (workspaceId: string, directory = "") => {
    setLoadingWorkspaceDirectory(directory);
    try {
      const rows = await readWorkspaceDirectory(workspaceId, directory);
      setWorkspaceFilesByDirectory((current) => ({
        ...current,
        [directory]: rows,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load workspace files.");
    } finally {
      setLoadingWorkspaceDirectory((current) => (current === directory ? "" : current));
    }
  });

  // ---------------------------------------------------------------------------
  // Computed values
  // ---------------------------------------------------------------------------

  const normalizedWorkspaceFileFilterQuery = workspaceFileFilterQuery.trim();
  const loadedWorkspaceFileEntries = Object.values(workspaceFilesByDirectory).flat();
  const matchedWorkspaceDirectories = normalizedWorkspaceFileFilterQuery
    ? loadedWorkspaceFileEntries
        .filter((entry) =>
          entry.is_dir
          && matchesWorkspaceSearch(
            workspaceFileFilterQuery,
            entry.path,
            entry.name,
            entry.extension,
          ),
        )
        .map((entry) => entry.path)
    : [];
  const locallyVisibleWorkspaceFilePaths = new Set<string>();
  if (normalizedWorkspaceFileFilterQuery) {
    for (const entry of loadedWorkspaceFileEntries) {
      const directMatch = matchesWorkspaceSearch(
        workspaceFileFilterQuery,
        entry.path,
        entry.name,
        entry.extension,
      );
      const insideMatchedDirectory = matchedWorkspaceDirectories.some((directoryPath) =>
        entry.path === directoryPath || entry.path.startsWith(`${directoryPath}/`),
      );
      if (!directMatch && !insideMatchedDirectory) {
        continue;
      }
      locallyVisibleWorkspaceFilePaths.add(entry.path);
      for (const directory of ancestorDirectories(entry.path)) {
        locallyVisibleWorkspaceFilePaths.add(directory);
      }
    }
  }
  const rootWorkspaceFiles = workspaceFilesByDirectory[""] || [];
  const selectedWorkspaceFileEntry = Object.values(workspaceFilesByDirectory)
    .flat()
    .find((entry) => entry.path === selectedWorkspaceFilePath) || null;
  const selectedWorkspaceFileIsEditable = Boolean(
    selectedWorkspaceFileEntry
      && !selectedWorkspaceFileEntry.is_dir
      && workspaceFilePreview?.preview_kind === "text"
      && !workspaceFilePreview.error
      && !workspaceFilePreview.truncated,
  );
  const selectedWorkspaceFileEditorHasChanges = selectedWorkspaceFileIsEditable
    && workspaceFileEditorDraft !== (workspaceFilePreview?.text_content || "");
  const selectedWorkspaceFileEditHint = !selectedWorkspaceFileEntry || selectedWorkspaceFileEntry.is_dir
    ? ""
    : workspaceFilePreview?.preview_kind === "document"
      ? "Document previews are extracted text and stay read-only in the desktop app."
      : workspaceFilePreview?.preview_kind === "image"
        ? "Image previews are metadata-only for now."
        : workspaceFilePreview?.preview_kind === "table"
          ? "Structured table previews stay read-only for now."
          : workspaceFilePreview?.preview_kind === "unsupported"
            ? "Binary files should be opened externally."
            : workspaceFilePreview?.truncated
              ? "Large text previews stay read-only to avoid accidental partial saves."
              : "";
  const selectedFileArtifactHistory = selectedWorkspaceFilePath
    ? workspaceArtifacts
        .filter((artifact) => artifact.path === selectedWorkspaceFilePath)
        .sort((left, right) => {
          const leftTime = Date.parse(left.created_at || "");
          const rightTime = Date.parse(right.created_at || "");
          if (Number.isFinite(leftTime) && Number.isFinite(rightTime) && leftTime !== rightTime) {
            return rightTime - leftTime;
          }
          return left.path.localeCompare(right.path);
        })
    : [];
  const selectedFileLatestArtifact = selectedFileArtifactHistory[0] || null;
  const selectedFileRunIds = Array.from(new Set(
    selectedFileArtifactHistory.flatMap((artifact) => artifact.run_ids || []).filter(Boolean),
  ));
  const selectedFileRelatedConversations = selectedFileRunIds.length > 0
    ? workspaceConversationRows.filter((conversation) =>
        conversation.linked_run_ids.some((runId) => selectedFileRunIds.includes(runId)),
      )
    : [];
  const contextualArtifacts = selectedRunId
    ? workspaceArtifacts.filter((artifact) =>
        artifact.latest_run_id === selectedRunId || artifact.run_ids.includes(selectedRunId),
      )
    : selectedConversationRunIds.length > 0
      ? workspaceArtifacts.filter((artifact) =>
          artifact.run_ids.some((runId) => selectedConversationRunIds.includes(runId)),
        )
      : [];
  const contextualFilePaths = new Set(
    [
      ...contextualArtifacts.map((artifact) => artifact.path),
      ...(selectedRunId ? runArtifacts.map((artifact) => artifact.path) : []),
    ].filter(Boolean),
  );
  const contextualDirectoryCounts = new Map<string, number>();
  for (const path of contextualFilePaths) {
    for (const directory of ancestorDirectories(path)) {
      contextualDirectoryCounts.set(directory, (contextualDirectoryCounts.get(directory) || 0) + 1);
    }
  }
  const recentFilePaths = new Set(
    recentWorkspaceArtifacts.map((artifact) => artifact.path).filter(Boolean),
  );
  const recentDirectoryCounts = new Map<string, number>();
  for (const path of recentFilePaths) {
    for (const directory of ancestorDirectories(path)) {
      recentDirectoryCounts.set(directory, (recentDirectoryCounts.get(directory) || 0) + 1);
    }
  }
  const activeFilesLabel = selectedRunId
    ? "Selected run"
    : selectedConversationSummary
      ? "Selected thread"
      : "";
  const visibleRootWorkspaceFiles = rootWorkspaceFiles.filter((entry) => {
    if (
      normalizedWorkspaceFileFilterQuery
      && !locallyVisibleWorkspaceFilePaths.has(entry.path)
    ) {
      return false;
    }
    if (workspaceFileTreeMode === "all") {
      return true;
    }
    if (workspaceFileTreeMode === "active") {
      return entry.is_dir
        ? (contextualDirectoryCounts.get(entry.path) || 0) > 0
        : contextualFilePaths.has(entry.path);
    }
    return entry.is_dir
      ? (recentDirectoryCounts.get(entry.path) || 0) > 0
      : recentFilePaths.has(entry.path);
  });

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  // Reset file tree state and reload root when workspace changes
  useEffect(() => {
    setWorkspaceFilesByDirectory({});
    setExpandedWorkspaceDirectories([""]);
    setSelectedWorkspaceFilePath("");
    setWorkspaceFilePreview(null);
    if (selectedWorkspaceId) {
      void loadWorkspaceDirectory(selectedWorkspaceId, "");
    }
  }, [selectedWorkspaceId]);

  useEffect(() => {
    if (!pendingWorkspaceFileOpen) {
      return;
    }
    if (pendingWorkspaceFileOpen.workspaceId !== selectedWorkspaceId) {
      return;
    }
    void handleOpenWorkspaceFile(pendingWorkspaceFileOpen.path).finally(() => {
      setPendingWorkspaceFileOpen((current) =>
        current?.workspaceId === pendingWorkspaceFileOpen.workspaceId
        && current?.path === pendingWorkspaceFileOpen.path
          ? null
          : current,
      );
    });
  }, [pendingWorkspaceFileOpen, selectedWorkspaceId]);

  // Load file preview when selection changes
  useEffect(() => {
    if (!selectedWorkspaceId || !selectedWorkspaceFilePath) {
      setWorkspaceFilePreview(null);
      setLoadingWorkspaceFilePreview(false);
      return;
    }
    let cancelled = false;
    setLoadingWorkspaceFilePreview(true);
    void (async () => {
      try {
        const preview = await fetchWorkspaceFilePreview(
          selectedWorkspaceId,
          selectedWorkspaceFilePath,
        );
        if (!cancelled) {
          setWorkspaceFilePreview(preview);
        }
      } catch (err) {
        if (!cancelled) {
          setWorkspaceFilePreview(null);
          setError(err instanceof Error ? err.message : "Failed to load file preview.");
        }
      } finally {
        if (!cancelled) {
          setLoadingWorkspaceFilePreview(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selectedWorkspaceFilePath, selectedWorkspaceId]);

  // Sync file editor state with preview
  useEffect(() => {
    const editablePreview = workspaceFilePreview?.preview_kind === "text"
      && !workspaceFilePreview.error
      && !workspaceFilePreview.truncated;
    if (!selectedWorkspaceFilePath || !editablePreview || !workspaceFilePreview) {
      if (workspaceFileEditorPath || workspaceFileEditorDraft || workspaceFileEditorDirty) {
        setWorkspaceFileEditorPath("");
        setWorkspaceFileEditorDraft("");
        setWorkspaceFileEditorDirty(false);
      }
      return;
    }
    if (
      workspaceFileEditorPath !== selectedWorkspaceFilePath
      || !workspaceFileEditorDirty
    ) {
      setWorkspaceFileEditorPath(selectedWorkspaceFilePath);
      setWorkspaceFileEditorDraft(workspaceFilePreview.text_content || "");
      setWorkspaceFileEditorDirty(false);
    }
  }, [
    selectedWorkspaceFilePath,
    workspaceFileEditorDirty,
    workspaceFileEditorDraft,
    workspaceFileEditorPath,
    workspaceFilePreview,
  ]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  async function handleOpenWorkspaceFile(path: string) {
    if (!selectedWorkspaceId || !path.trim()) {
      return;
    }
    if (
      selectedWorkspaceFileEditorHasChanges
      && path !== selectedWorkspaceFilePath
      && !window.confirm(`Discard unsaved changes to ${selectedWorkspaceFilePath}?`)
    ) {
      return;
    }
    const ancestors = ancestorDirectories(path);
    setExpandedWorkspaceDirectories((current) => {
      const merged = new Set(current);
      merged.add("");
      for (const item of ancestors) {
        merged.add(item);
      }
      return Array.from(merged);
    });
    for (const directory of ancestors) {
      if (workspaceFilesByDirectory[directory] === undefined) {
        // eslint-disable-next-line no-await-in-loop
        await loadWorkspaceDirectory(selectedWorkspaceId, directory);
      }
    }
    setSelectedWorkspaceFilePath(path);
  }

  function queueWorkspaceFileOpen(workspaceId: string, path: string) {
    const cleanWorkspaceId = String(workspaceId || "").trim();
    const cleanPath = String(path || "").trim();
    if (!cleanWorkspaceId || !cleanPath) {
      return;
    }
    if (cleanWorkspaceId === selectedWorkspaceId) {
      void handleOpenWorkspaceFile(cleanPath);
      return;
    }
    setPendingWorkspaceFileOpen({
      workspaceId: cleanWorkspaceId,
      path: cleanPath,
    });
  }

  async function handleOpenWorkspaceFileExternally() {
    const workspacePath = selectedWorkspaceSummary?.canonical_path || "";
    const relativePath = selectedWorkspaceFilePath.trim();
    if (!workspacePath || !relativePath) {
      setError("Select a workspace file before opening it externally.");
      return;
    }
    setError("");
    setNotice("");
    try {
      await openWorkspaceFile(workspacePath, relativePath);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to open workspace file.");
    }
  }

  async function handleRevealWorkspaceFile() {
    const workspacePath = selectedWorkspaceSummary?.canonical_path || "";
    const relativePath = selectedWorkspaceFilePath.trim();
    if (!workspacePath || !relativePath) {
      setError("Select a workspace file before revealing it.");
      return;
    }
    setError("");
    setNotice("");
    try {
      await revealWorkspaceFile(workspacePath, relativePath);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reveal workspace file.");
    }
  }

  async function handleSaveWorkspaceFile() {
    const workspacePath = selectedWorkspaceSummary?.canonical_path || "";
    const relativePath = selectedWorkspaceFilePath.trim();
    const editablePreview = workspaceFilePreview?.preview_kind === "text"
      && !workspaceFilePreview.error
      && !workspaceFilePreview.truncated;
    if (!workspacePath || !relativePath) {
      setError("Select a workspace text file before saving.");
      return;
    }
    if (!editablePreview) {
      setError("Only non-truncated text previews can be edited in the desktop app.");
      return;
    }
    setSavingWorkspaceFile(true);
    setError("");
    setNotice("");
    try {
      await createWorkspaceFile(
        workspacePath,
        relativePath,
        workspaceFileEditorDraft,
        true,
      );
      const refreshedPreview = selectedWorkspaceId
        ? await fetchWorkspaceFilePreview(selectedWorkspaceId, relativePath)
        : null;
      if (refreshedPreview) {
        setWorkspaceFilePreview(refreshedPreview);
      }
      setWorkspaceFileEditorPath(relativePath);
      setWorkspaceFileEditorDirty(false);
      setNotice(`Saved ${relativePath}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save workspace file.");
    } finally {
      setSavingWorkspaceFile(false);
    }
  }

  function handleResetWorkspaceFileEditor() {
    if (!workspaceFilePreview || workspaceFilePreview.preview_kind !== "text") {
      return;
    }
    setWorkspaceFileEditorDraft(workspaceFilePreview.text_content || "");
    setWorkspaceFileEditorDirty(false);
    
    setError("");
  }

  async function handleRefreshWorkspaceFiles() {
    if (!selectedWorkspaceId) {
      return;
    }
    setRefreshingWorkspaceFiles(true);
    setError("");
    setNotice("");
    const directories = Array.from(
      new Set(["", ...Object.keys(workspaceFilesByDirectory), ...expandedWorkspaceDirectories]),
    ).sort((left, right) => {
      const depthDifference = left.split("/").length - right.split("/").length;
      if (depthDifference !== 0) {
        return depthDifference;
      }
      return left.localeCompare(right);
    });
    try {
      const nextWorkspaceFilesByDirectory: Record<string, WorkspaceFileEntry[]> = {};
      const missingDirectories = new Set<string>();
      for (const directory of directories) {
        try {
          // eslint-disable-next-line no-await-in-loop
          const rows = await readWorkspaceDirectory(selectedWorkspaceId, directory);
          nextWorkspaceFilesByDirectory[directory] = rows;
        } catch (err) {
          if (!isNotFoundRequestError(err)) {
            throw err;
          }
          missingDirectories.add(directory);
        }
      }
      setWorkspaceFilesByDirectory(nextWorkspaceFilesByDirectory);
      if (missingDirectories.size > 0) {
        setExpandedWorkspaceDirectories((current) =>
          current.filter((directory) => !missingDirectories.has(directory)),
        );
      }
      if (selectedWorkspaceFilePath) {
        setLoadingWorkspaceFilePreview(true);
        try {
          const preview = await fetchWorkspaceFilePreview(
            selectedWorkspaceId,
            selectedWorkspaceFilePath,
          );
          setWorkspaceFilePreview(preview);
        } catch (err) {
          if (isNotFoundRequestError(err)) {
            setSelectedWorkspaceFilePath("");
            setWorkspaceFilePreview(null);
          } else {
            setWorkspaceFilePreview(null);
            setError(err instanceof Error ? err.message : "Failed to load file preview.");
          }
        } finally {
          setLoadingWorkspaceFilePreview(false);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reload workspace files.");
    } finally {
      setRefreshingWorkspaceFiles(false);
    }
  }

  async function handleExpandActiveWorkspaceFiles() {
    if (!selectedWorkspaceId || contextualFilePaths.size === 0) {
      return;
    }
    const directorySet = new Set<string>([""]);
    for (const path of contextualFilePaths) {
      for (const directory of ancestorDirectories(path)) {
        directorySet.add(directory);
      }
    }
    const directories = Array.from(directorySet).sort((left, right) => left.localeCompare(right));
    setExpandedWorkspaceDirectories(directories);
    for (const directory of directories) {
      if (workspaceFilesByDirectory[directory] === undefined) {
        // eslint-disable-next-line no-await-in-loop
        await loadWorkspaceDirectory(selectedWorkspaceId, directory);
      }
    }
  }

  async function handleExpandRecentWorkspaceFiles() {
    if (!selectedWorkspaceId || recentFilePaths.size === 0) {
      return;
    }
    const directorySet = new Set<string>([""]);
    for (const path of recentFilePaths) {
      for (const directory of ancestorDirectories(path)) {
        directorySet.add(directory);
      }
    }
    const directories = Array.from(directorySet).sort((left, right) => left.localeCompare(right));
    setExpandedWorkspaceDirectories(directories);
    for (const directory of directories) {
      if (workspaceFilesByDirectory[directory] === undefined) {
        // eslint-disable-next-line no-await-in-loop
        await loadWorkspaceDirectory(selectedWorkspaceId, directory);
      }
    }
  }

  function toggleWorkspaceDirectory(path: string) {
    const isExpanded = expandedWorkspaceDirectories.includes(path);
    if (isExpanded) {
      setExpandedWorkspaceDirectories((current) => current.filter((item) => item !== path));
      return;
    }
    setExpandedWorkspaceDirectories((current) => [...current, path]);
    if (selectedWorkspaceId && workspaceFilesByDirectory[path] === undefined) {
      void loadWorkspaceDirectory(selectedWorkspaceId, path);
    }
  }

  function handleWorkspaceFileSelection(entry: WorkspaceFileEntry) {
    if (entry.is_dir) {
      toggleWorkspaceDirectory(entry.path);
      return;
    }
    if (
      selectedWorkspaceFileEditorHasChanges
      && entry.path !== selectedWorkspaceFilePath
      && !window.confirm(`Discard unsaved changes to ${selectedWorkspaceFilePath}?`)
    ) {
      return;
    }
    setSelectedWorkspaceFilePath(entry.path);
    
    setError("");
  }

  async function handleImportWorkspaceFiles(files: FileList | null) {
    const workspacePath = selectedWorkspaceSummary?.canonical_path || "";
    if (!workspacePath) {
      setError("Select a workspace before importing files.");
      return;
    }
    if (!files || files.length === 0) {
      return;
    }
    const destinationFolder = workspaceImportFolderDraft.trim().replace(/^[\\/]+|[\\/]+$/g, "");
    const importRows = await Promise.all(
      Array.from(files).map(async (file) => ({
        relativePath: destinationFolder
          ? `${destinationFolder}/${file.name}`
          : file.name,
        bytes: Array.from(new Uint8Array(await file.arrayBuffer())),
      })),
    );
    if (importRows.length === 0) {
      return;
    }
    setImportingWorkspaceFiles(true);
    setError("");
    setNotice("");
    try {
      const createdPaths = await importWorkspaceFiles(
        workspacePath,
        importRows,
      );
      setNotice(
        createdPaths.length === 1
          ? `Imported ${createdPaths[0]} into ${selectedWorkspaceSummary?.display_name || "workspace"}.`
          : `Imported ${createdPaths.length} files into ${selectedWorkspaceSummary?.display_name || "workspace"}.`,
      );
      setWorkspaceFilesByDirectory({});
      setExpandedWorkspaceDirectories([""]);
      setSelectedWorkspaceFilePath(createdPaths[0] || "");
      void loadWorkspaceDirectory(selectedWorkspaceId, "");
      if (workspaceFileInputRef.current) {
        workspaceFileInputRef.current.value = "";
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to import workspace files.");
    } finally {
      setImportingWorkspaceFiles(false);
    }
  }

  return {
    // State
    workspaceFilesByDirectory,
    expandedWorkspaceDirectories,
    loadingWorkspaceDirectory,
    refreshingWorkspaceFiles,
    selectedWorkspaceFilePath,
    workspaceFilePreview,
    loadingWorkspaceFilePreview,
    workspaceFileEditorPath,
    workspaceFileEditorDraft,
    workspaceFileEditorDirty,
    savingWorkspaceFile,
    workspaceFileFilterQuery,
    importingWorkspaceFiles,

    // Computed
    normalizedWorkspaceFileFilterQuery,
    loadedWorkspaceFileEntries,
    matchedWorkspaceDirectories,
    locallyVisibleWorkspaceFilePaths,
    rootWorkspaceFiles,
    selectedWorkspaceFileEntry,
    selectedWorkspaceFileIsEditable,
    selectedWorkspaceFileEditorHasChanges,
    selectedWorkspaceFileEditHint,
    selectedFileArtifactHistory,
    selectedFileLatestArtifact,
    selectedFileRunIds,
    selectedFileRelatedConversations,
    contextualArtifacts,
    contextualFilePaths,
    contextualDirectoryCounts,
    recentFilePaths,
    recentDirectoryCounts,
    activeFilesLabel,
    visibleRootWorkspaceFiles,

    // Refs
    workspaceFileInputRef,

    // Actions
    setSelectedWorkspaceFilePath,
    setWorkspaceFileEditorDraft,
    setWorkspaceFileEditorDirty,
    setWorkspaceFileFilterQuery,
    handleOpenWorkspaceFile,
    queueWorkspaceFileOpen,
    handleOpenWorkspaceFileExternally,
    handleRevealWorkspaceFile,
    handleSaveWorkspaceFile,
    handleResetWorkspaceFileEditor,
    handleRefreshWorkspaceFiles,
    handleExpandActiveWorkspaceFiles,
    handleExpandRecentWorkspaceFiles,
    toggleWorkspaceDirectory,
    handleWorkspaceFileSelection,
    handleImportWorkspaceFiles,
    loadWorkspaceDirectory,
  };
}
