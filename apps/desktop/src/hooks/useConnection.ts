import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

import {
  bootstrapDesktopRuntime,
  fetchModels,
  fetchRuntimeStatus,
  fetchSettings,
  fetchWorkspaces,
  patchSettings,
  type ModelInfo,
  type RuntimeStatus,
  type SettingsPayload,
  type WorkspaceSummary,
} from "../api";
import { mergeWorkspaceSummary } from "../utils";

// ---------------------------------------------------------------------------
// Shell snapshot loaders
// ---------------------------------------------------------------------------

async function loadShellSnapshotWithArchived(includeArchived: boolean): Promise<{
  runtimeSnapshot: RuntimeStatus;
  modelRows: ModelInfo[];
  workspaceRows: WorkspaceSummary[];
  settingsPayload: SettingsPayload;
}> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const [runtimeSnapshot, modelRows, workspaceRows, settingsPayload] = await Promise.all([
        fetchRuntimeStatus(),
        fetchModels(),
        fetchWorkspaces(includeArchived),
        fetchSettings(),
      ]);
      return { runtimeSnapshot, modelRows, workspaceRows, settingsPayload };
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => window.setTimeout(resolve, 300));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Failed to load Loomd.");
}

const HEALTH_CHECK_INTERVAL_MS = 15000;
const AUTO_RECONNECT_DELAY_MS = 1500;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface ConnectionState {
  runtime: RuntimeStatus | null;
  models: ModelInfo[];
  workspaces: WorkspaceSummary[];
  settings: SettingsPayload | null;
  runtimeManaged: boolean;
  connectionState: "connecting" | "connected" | "failed";
  settingsPreview: Array<{ path: string; section: string; field: string; description: string; configured_display: string; effective_display: string }>;
}

export interface ConnectionActions {
  retryConnection: () => void;
  setWorkspaces: React.Dispatch<React.SetStateAction<WorkspaceSummary[]>>;
  setSettings: React.Dispatch<React.SetStateAction<SettingsPayload | null>>;
  setRuntime: React.Dispatch<React.SetStateAction<RuntimeStatus | null>>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useConnection(deps: {
  setError: React.Dispatch<React.SetStateAction<string>>;
  showArchivedWorkspaces: boolean;
  selectedWorkspaceId: string;
  setSelectedWorkspaceId: React.Dispatch<React.SetStateAction<string>>;
  setCreateParentPath: React.Dispatch<React.SetStateAction<string>>;
}): ConnectionState & ConnectionActions {
  const { setError, showArchivedWorkspaces, setSelectedWorkspaceId, setCreateParentPath } = deps;

  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [settings, setSettings] = useState<SettingsPayload | null>(null);
  const [connectionState, setConnectionState] = useState<"connecting" | "connected" | "failed">("connecting");
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const [runtimeManaged, setRuntimeManaged] = useState(false);
  const loadInFlightRef = useRef(false);
  const healthCheckInFlightRef = useRef(false);

  // Bootstrap runtime and load initial shell data
  useEffect(() => {
    let cancelled = false;
    loadInFlightRef.current = true;

    async function loadShell() {
      setConnectionState("connecting");
      try {
        const managed = await bootstrapDesktopRuntime();
        const {
          runtimeSnapshot,
          modelRows,
          workspaceRows,
          settingsPayload,
        } = await loadShellSnapshotWithArchived(showArchivedWorkspaces);
        if (cancelled) {
          return;
        }
        setRuntimeManaged(managed);
        setRuntime(runtimeSnapshot);
        setModels(modelRows);
        setWorkspaces(workspaceRows);
        setSettings(settingsPayload);
        setError("");
        setConnectionState("connected");
        setCreateParentPath((current) => current || runtimeSnapshot.workspace_default_path || "");
        if (workspaceRows.length > 0) {
          startTransition(() => {
            setSelectedWorkspaceId((currentId) => {
              const stillValid = workspaceRows.some((workspace) => workspace.id === currentId);
              return stillValid ? currentId : workspaceRows[0].id;
            });
          });
        }
      } catch (err) {
        if (!cancelled) {
          setConnectionState("failed");
          setError(err instanceof Error ? err.message : "Failed to connect to Loomd.");
        }
      } finally {
        if (!cancelled) {
          loadInFlightRef.current = false;
        }
      }
    }

    void loadShell();
    return () => {
      cancelled = true;
      loadInFlightRef.current = false;
    };
  }, [showArchivedWorkspaces, connectionAttempt]);

  useEffect(() => {
    if (connectionState !== "failed" || loadInFlightRef.current) {
      return;
    }

    const timerId = window.setTimeout(() => {
      if (!loadInFlightRef.current) {
        setConnectionAttempt((current) => current + 1);
      }
    }, AUTO_RECONNECT_DELAY_MS);

    return () => {
      window.clearTimeout(timerId);
    };
  }, [connectionState]);

  const refreshConnectionHealth = useEffectEvent(async () => {
    if (connectionState !== "connected" || loadInFlightRef.current || healthCheckInFlightRef.current) {
      return;
    }

    healthCheckInFlightRef.current = true;
    try {
      const runtimeSnapshot = await fetchRuntimeStatus();
      setRuntime(runtimeSnapshot);
    } catch {
      setConnectionState("failed");
      setError("Lost connection to Loomd. Reconnecting...");
    } finally {
      healthCheckInFlightRef.current = false;
    }
  });

  useEffect(() => {
    if (connectionState !== "connected") {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshConnectionHealth();
    }, HEALTH_CHECK_INTERVAL_MS);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [connectionState, refreshConnectionHealth]);

  function retryConnection() {
    setConnectionState("connecting");
    setError("");
    setConnectionAttempt((c) => c + 1);
  }

  // Computed
  const settingsPreview = (settings?.basic || []).slice(0, 4);

  return {
    runtime,
    models,
    workspaces,
    settings,
    runtimeManaged,
    connectionState,
    settingsPreview,
    retryConnection,
    setWorkspaces,
    setSettings,
    setRuntime,
  };
}
