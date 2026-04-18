import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

import {
  bootstrapDesktopRuntime,
  completeInitialSetup as completeInitialSetupRequest,
  discoverSetupModels as discoverSetupModelsRequest,
  fetchDesktopSidecarStatus,
  fetchModels,
  patchModel as patchModelRequest,
  fetchRuntimeStatus,
  fetchSettings,
  fetchSetupStatus,
  fetchWorkspaces,
  type ModelPatchRequest,
  type ModelInfo,
  type RuntimeStatus,
  type SetupCompleteRequest,
  type SetupStatus,
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
  setupStatus: SetupStatus;
}> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const [runtimeSnapshot, modelRows, workspaceRows, settingsPayload, setupStatus] = await Promise.all([
        fetchRuntimeStatus(),
        fetchModels(),
        fetchWorkspaces(includeArchived),
        fetchSettings(),
        fetchSetupStatus(),
      ]);
      return {
        runtimeSnapshot,
        modelRows,
        workspaceRows,
        settingsPayload,
        setupStatus,
      };
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => window.setTimeout(resolve, 300));
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Failed to load Loomd.");
}

const HEALTH_CHECK_INTERVAL_MS = 15000;
const AUTO_RECONNECT_DELAY_MS = 1500;
const HEALTH_CHECK_FAILURE_THRESHOLD = 3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface ConnectionState {
  runtime: RuntimeStatus | null;
  models: ModelInfo[];
  workspaces: WorkspaceSummary[];
  settings: SettingsPayload | null;
  setupStatus: SetupStatus | null;
  runtimeManaged: boolean;
  connectionState: "connecting" | "connected" | "failed";
  settingsPreview: Array<{ path: string; section: string; field: string; description: string; configured_display: string; effective_display: string }>;
}

export interface ConnectionActions {
  retryConnection: () => void;
  discoverSetupModels: (provider: string, baseUrl: string, apiKey?: string) => Promise<string[]>;
  completeInitialSetup: (payload: SetupCompleteRequest) => Promise<void>;
  updateModelSettings: (modelName: string, payload: ModelPatchRequest) => Promise<void>;
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
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null);
  const [connectionState, setConnectionState] = useState<"connecting" | "connected" | "failed">("connecting");
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const [runtimeManaged, setRuntimeManaged] = useState(false);
  const loadInFlightRef = useRef(false);
  const healthCheckInFlightRef = useRef(false);
  const lastArchivedVisibilityRef = useRef(showArchivedWorkspaces);
  const consecutiveHealthFailuresRef = useRef(0);

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
          setupStatus: setupStatusPayload,
        } = await loadShellSnapshotWithArchived(showArchivedWorkspaces);
        if (cancelled) {
          return;
        }
        setRuntimeManaged(managed);
        setRuntime(runtimeSnapshot);
        setModels(modelRows);
        setWorkspaces(workspaceRows);
        setSettings(settingsPayload);
        setSetupStatus(setupStatusPayload);
        consecutiveHealthFailuresRef.current = 0;
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
  }, [connectionAttempt]);

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
      consecutiveHealthFailuresRef.current = 0;
      setRuntime(runtimeSnapshot);
    } catch {
      consecutiveHealthFailuresRef.current += 1;

      const sidecarStatus = runtimeManaged
        ? await fetchDesktopSidecarStatus()
        : null;
      const sidecarStillRunning = Boolean(
        runtimeManaged
        && sidecarStatus?.managed_by_desktop
        && sidecarStatus.running,
      );

      if (sidecarStillRunning) {
        return;
      }

      if (consecutiveHealthFailuresRef.current < HEALTH_CHECK_FAILURE_THRESHOLD) {
        return;
      }

      setConnectionState("failed");
      setError("Lost connection to Loomd. Reconnecting...");
    } finally {
      healthCheckInFlightRef.current = false;
    }
  });

  const refreshWorkspaceSummaries = useEffectEvent(async () => {
    if (connectionState !== "connected" || loadInFlightRef.current) {
      return;
    }

    const workspaceRows = await fetchWorkspaces(showArchivedWorkspaces);
    setWorkspaces(workspaceRows);
    startTransition(() => {
      setSelectedWorkspaceId((currentId) => {
        const stillValid = workspaceRows.some((workspace) => workspace.id === currentId);
        return stillValid ? currentId : (workspaceRows[0]?.id || "");
      });
    });
  });

  useEffect(() => {
    if (connectionState !== "connected") {
      lastArchivedVisibilityRef.current = showArchivedWorkspaces;
      return;
    }
    if (lastArchivedVisibilityRef.current === showArchivedWorkspaces) {
      return;
    }
    lastArchivedVisibilityRef.current = showArchivedWorkspaces;
    void refreshWorkspaceSummaries().catch((err) => {
      setError(err instanceof Error ? err.message : "Failed to refresh workspaces.");
    });
  }, [connectionState, setError, showArchivedWorkspaces]);

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
  }, [connectionState]);

  function retryConnection() {
    consecutiveHealthFailuresRef.current = 0;
    setConnectionState("connecting");
    setError("");
    setConnectionAttempt((c) => c + 1);
  }

  // Computed
  const settingsPreview = (settings?.basic || []).slice(0, 4);

  async function discoverSetupModels(
    provider: string,
    baseUrl: string,
    apiKey = "",
  ): Promise<string[]> {
    const response = await discoverSetupModelsRequest({
      provider,
      base_url: baseUrl,
      api_key: apiKey,
    });
    return response.models;
  }

  async function completeInitialSetup(payload: SetupCompleteRequest): Promise<void> {
    setError("");
    await completeInitialSetupRequest(payload);
    const {
      runtimeSnapshot,
      modelRows,
      workspaceRows,
      settingsPayload,
      setupStatus: setupStatusPayload,
    } = await loadShellSnapshotWithArchived(showArchivedWorkspaces);
    setRuntime(runtimeSnapshot);
    setModels(modelRows);
    setWorkspaces(workspaceRows);
    setSettings(settingsPayload);
    setSetupStatus(setupStatusPayload);
    setConnectionState("connected");
    setCreateParentPath((current) => current || runtimeSnapshot.workspace_default_path || "");
    startTransition(() => {
      setSelectedWorkspaceId((currentId) => {
        const stillValid = workspaceRows.some((workspace) => workspace.id === currentId);
        return stillValid ? currentId : (workspaceRows[0]?.id || "");
      });
    });
  }

  async function updateModelSettings(
    modelName: string,
    payload: ModelPatchRequest,
  ): Promise<void> {
    setError("");
    const updated = await patchModelRequest(modelName, payload);
    setModels((current) => current.map((model) => (
      model.name === updated.name ? updated : model
    )));
  }

  return {
    runtime,
    models,
    workspaces,
    settings,
    setupStatus,
    runtimeManaged,
    connectionState,
    settingsPreview,
    retryConnection,
    discoverSetupModels,
    completeInitialSetup,
    updateModelSettings,
    setWorkspaces,
    setSettings,
    setRuntime,
  };
}
