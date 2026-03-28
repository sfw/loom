import {
  startTransition,
  type FormEvent,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

import {
  cancelRun,
  createTask,
  deleteRun,
  fetchRunArtifacts,
  fetchRunDetail,
  fetchRunTimeline,
  pauseRun,
  restartRun,
  resumeRun,
  sendRunMessage,
  subscribeRunStream,
  type RunArtifact,
  type RunDetail,
  type RunTimelineEvent,
  type WorkspaceOverview,
} from "../api";
import {
  isRunTimelineNoise,
  matchesWorkspaceSearch,
  runTimelineDetail,
  runTimelinePills,
  runTimelineTitle,
} from "../history";
import { isTransientRequestError, type ViewTab } from "../utils";

function mergeRunTimelineEvents(
  existing: RunTimelineEvent[],
  incoming: RunTimelineEvent[],
): RunTimelineEvent[] {
  if (incoming.length === 0) {
    return existing;
  }
  const byId = new Map<number, RunTimelineEvent>();
  for (const item of existing) {
    byId.set(item.id, item);
  }
  for (const item of incoming) {
    byId.set(item.id, item);
  }
  return Array.from(byId.values()).sort((left, right) => {
    if (left.id !== right.id) {
      return left.id - right.id;
    }
    if (left.sequence !== right.sequence) {
      return left.sequence - right.sequence;
    }
    return left.timestamp.localeCompare(right.timestamp);
  });
}

function streamEventToTimelineEvent(event: {
  id?: number;
  task_id: string;
  run_id?: string;
  correlation_id?: string;
  event_id?: string;
  sequence?: number;
  timestamp: string;
  event_type: string;
  source_component?: string;
  schema_version?: number;
  data?: Record<string, unknown>;
}): RunTimelineEvent | null {
  if (typeof event.id !== "number" || event.id <= 0) {
    return null;
  }
  return {
    id: event.id,
    task_id: event.task_id,
    run_id: typeof event.run_id === "string" ? event.run_id : "",
    correlation_id: typeof event.correlation_id === "string" ? event.correlation_id : "",
    event_id: typeof event.event_id === "string" ? event.event_id : "",
    sequence: typeof event.sequence === "number" ? event.sequence : 0,
    timestamp: event.timestamp,
    event_type: event.event_type,
    source_component:
      typeof event.source_component === "string" ? event.source_component : "",
    schema_version:
      typeof event.schema_version === "number" ? event.schema_version : 1,
    data: event.data && typeof event.data === "object" ? event.data : {},
  };
}

function deriveRunStatusFromStreamEvent(
  eventType: string,
  nextStatus: string,
): string {
  if (nextStatus) {
    return nextStatus;
  }
  switch (eventType) {
    case "task_paused":
      return "paused";
    case "task_resumed":
    case "task_executing":
      return "executing";
    case "task_planning":
      return "planning";
    case "task_completed":
      return "completed";
    case "task_failed":
      return "failed";
    case "task_cancelled":
    case "task_cancel_requested":
      return "cancelled";
    default:
      return "";
  }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface RunsState {
  selectedRunId: string;
  runDetail: RunDetail | null;
  runTimeline: RunTimelineEvent[];
  runArtifacts: RunArtifact[];
  runStreaming: boolean;
  loadingRunDetail: boolean;
  runLoadError: string;
  runGoal: string;
  runProcess: string;
  runApprovalMode: string;
  launchingRun: boolean;
  runOperatorMessage: string;
  runActionPending: string;
  sendingRunMessage: boolean;
  runHistoryQuery: string;
  activeRunMatchIndex: number;

  // Computed
  normalizedRunStatus: string;
  runIsTerminal: boolean;
  runCanPause: boolean;
  runCanResume: boolean;
  runCanMessage: boolean;
  selectedRunSummary: RunDetail | { id: string; goal: string; status: string; created_at: string; updated_at: string; process_name: string; linked_conversation_ids: string[] } | null;
  filteredRunArtifacts: RunArtifact[];
  filteredRunTimeline: RunTimelineEvent[];
  visibleRunArtifacts: RunArtifact[];
  visibleRunTimeline: RunTimelineEvent[];
  totalRunMatches: number;

  // Refs
  runComposerRef: React.RefObject<HTMLElement | null>;
  runMatchRefs: React.MutableRefObject<Array<HTMLDivElement | null>>;
}

export interface RunsActions {
  setSelectedRunId: React.Dispatch<React.SetStateAction<string>>;
  setRunGoal: React.Dispatch<React.SetStateAction<string>>;
  setRunProcess: React.Dispatch<React.SetStateAction<string>>;
  setRunApprovalMode: React.Dispatch<React.SetStateAction<string>>;
  setRunOperatorMessage: React.Dispatch<React.SetStateAction<string>>;
  setRunHistoryQuery: React.Dispatch<React.SetStateAction<string>>;
  setActiveRunMatchIndex: React.Dispatch<React.SetStateAction<number>>;
  handleLaunchRun: (
    event: FormEvent<HTMLFormElement>,
    extraContext?: Record<string, unknown>,
  ) => Promise<void>;
  handleRunControl: (action: "pause" | "resume" | "cancel") => Promise<void>;
  handleDeleteRun: () => Promise<void>;
  handleRestartRun: () => Promise<void>;
  handleSendRunMessage: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handlePrefillStarterRun: () => void;
  focusRunComposer: () => void;
  refreshRun: (runId: string) => Promise<void>;
  scrollRunMatchIntoView: (index: number) => void;
  stepRunMatch: (delta: number) => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useRuns(deps: {
  selectedRunId: string;
  setSelectedRunId: React.Dispatch<React.SetStateAction<string>>;
  selectedWorkspaceId: string;
  overview: WorkspaceOverview | null;
  setError: React.Dispatch<React.SetStateAction<string>>;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
  setActiveTab: React.Dispatch<React.SetStateAction<ViewTab>>;
  refreshWorkspaceSurface: (workspaceId: string) => Promise<void>;
}): RunsState & RunsActions {
  const {
    selectedRunId,
    setSelectedRunId,
    selectedWorkspaceId,
    overview,
    setError,
    setNotice,
    setActiveTab,
    refreshWorkspaceSurface,
  } = deps;
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [runTimeline, setRunTimeline] = useState<RunTimelineEvent[]>([]);
  const [runArtifacts, setRunArtifacts] = useState<RunArtifact[]>([]);
  const [runStreaming, setRunStreaming] = useState(false);
  const [loadingRunDetail, setLoadingRunDetail] = useState(false);
  const [runLoadError, setRunLoadError] = useState("");
  const [runGoal, setRunGoal] = useState("");
  const [runProcess, setRunProcess] = useState("");
  const [runApprovalMode, setRunApprovalMode] = useState("auto");
  const [launchingRun, setLaunchingRun] = useState(false);
  const [runOperatorMessage, setRunOperatorMessage] = useState("");
  const [runActionPending, setRunActionPending] = useState("");
  const [sendingRunMessage, setSendingRunMessage] = useState(false);
  const [runHistoryQuery, setRunHistoryQuery] = useState("");
  const [activeRunMatchIndex, setActiveRunMatchIndex] = useState(0);

  // Refs
  const runComposerRef = useRef<HTMLElement | null>(null);
  const runRefreshTimerRef = useRef<number | null>(null);
  const runMatchRefs = useRef<Array<HTMLDivElement | null>>([]);
  const lastSeenRunEventIdRef = useRef(0);
  const lastRunStreamActivityAtRef = useRef(0);

  // ---------------------------------------------------------------------------
  // Computed values
  // ---------------------------------------------------------------------------

  const normalizedRunStatus = String(runDetail?.status || "").trim().toLowerCase();
  const runIsActivelyExecuting = ["executing", "planning", "running"].includes(normalizedRunStatus);
  const runIsTerminal = ["completed", "failed", "cancelled"].includes(normalizedRunStatus);
  const runCanPause = ["executing", "planning"].includes(normalizedRunStatus);
  const runCanResume = normalizedRunStatus === "paused";
  const runCanMessage = ["executing", "planning"].includes(normalizedRunStatus);
  const workspaceRunRows = overview?.recent_runs || [];
  const selectedRunSummary =
    runDetail
    || workspaceRunRows.find((run) => run.id === selectedRunId)
    || null;
  const filteredRunArtifacts = runArtifacts
    .filter((artifact) => {
      // Filter out meaningless entries like ".", "./", or "subfolder/."
      const p = artifact.path.trim().replace(/\\/g, "/");
      if (!p) return false;
      const parts = p.split("/").filter(Boolean);
      if (parts.length === 0) return false;
      if (parts[parts.length - 1] === ".") return false;
      return true;
    })
    .filter((artifact) =>
      matchesWorkspaceSearch(
        runHistoryQuery,
        artifact.path,
        artifact.category,
        artifact.source,
        artifact.tool_name,
        artifact.phase_ids.join(" "),
        artifact.subtask_ids.join(" "),
        artifact.facets,
      ),
    );
  // Filter out noise events that add no value, then apply search
  const meaningfulRunTimeline = runTimeline.filter((event) => !isRunTimelineNoise(event));
  const filteredRunTimeline = meaningfulRunTimeline.filter((event) =>
    matchesWorkspaceSearch(
      runHistoryQuery,
      event.event_type,
      runTimelineTitle(event),
      runTimelineDetail(event),
      event.data,
      runTimelinePills(event).join(" "),
    ),
  );
  const visibleRunArtifacts = filteredRunArtifacts;
  const visibleRunTimeline = runHistoryQuery.trim()
    ? filteredRunTimeline
    : meaningfulRunTimeline;
  const totalRunMatches = filteredRunArtifacts.length + filteredRunTimeline.length;

  async function loadRunSnapshot(runId: string): Promise<[RunDetail, RunTimelineEvent[], RunArtifact[]]> {
    let lastError: unknown = null;
    for (let attempt = 0; attempt < 6; attempt += 1) {
      try {
        return await Promise.all([
          fetchRunDetail(runId),
          fetchRunTimeline(runId),
          fetchRunArtifacts(runId),
        ]);
      } catch (error) {
        lastError = error;
        if (!isTransientRequestError(error) || attempt === 5) {
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 150 * (attempt + 1)));
      }
    }
    throw lastError instanceof Error ? lastError : new Error("Failed to load run.");
  }

  // ---------------------------------------------------------------------------
  // useEffectEvent handlers
  // ---------------------------------------------------------------------------

  const refreshRun = useEffectEvent(async (runId: string) => {
    const [detail, timeline, artifacts] = await loadRunSnapshot(runId);
    setRunLoadError("");
    setRunDetail(detail);
    setRunTimeline((current) => mergeRunTimelineEvents(current, timeline));
    setRunArtifacts(artifacts);
    lastSeenRunEventIdRef.current = Math.max(
      lastSeenRunEventIdRef.current,
      timeline.reduce(
        (maxId, row) => Math.max(maxId, Number(row.id || 0)),
        0,
      ),
    );
  });

  const scheduleRunRefresh = useEffectEvent(() => {
    if (runRefreshTimerRef.current !== null || !selectedRunId) {
      return;
    }
    runRefreshTimerRef.current = window.setTimeout(() => {
      runRefreshTimerRef.current = null;
      void Promise.all([
        refreshRun(selectedRunId),
        selectedWorkspaceId ? refreshWorkspaceSurface(selectedWorkspaceId) : Promise.resolve(),
      ]).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh run.");
        }
      });
    }, 200);
  });

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (runRefreshTimerRef.current !== null) {
        window.clearTimeout(runRefreshTimerRef.current);
      }
    };
  }, []);

  // Load run detail
  useEffect(() => {
    if (!selectedRunId) {
      setRunDetail(null);
      setRunTimeline([]);
      setRunArtifacts([]);
      setLoadingRunDetail(false);
      setRunLoadError("");
      setRunHistoryQuery("");
      setActiveRunMatchIndex(0);
      lastSeenRunEventIdRef.current = 0;
      lastRunStreamActivityAtRef.current = 0;
      return;
    }
    let cancelled = false;
    setLoadingRunDetail(true);
    setRunLoadError("");
    lastSeenRunEventIdRef.current = 0;
    lastRunStreamActivityAtRef.current = 0;

    void (async () => {
      try {
        const [detail, timeline, artifacts] = await loadRunSnapshot(selectedRunId);
        if (!cancelled) {
          setRunDetail(detail);
          setRunTimeline(timeline);
          setRunArtifacts(artifacts);
          lastSeenRunEventIdRef.current = timeline.reduce(
            (maxId, row) => Math.max(maxId, Number(row.id || 0)),
            0,
          );
        }
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Failed to load run.";
          setRunLoadError(message);
          setError(message);
          setRunDetail(null);
          setRunTimeline([]);
          setRunArtifacts([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingRunDetail(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedRunId]);

  // Run stream subscription
  useEffect(() => {
    if (!selectedRunId) {
      setRunStreaming(false);
      return;
    }
    if (loadingRunDetail) {
      return;
    }
    const cleanup = subscribeRunStream(
      selectedRunId,
      (event) => {
        lastRunStreamActivityAtRef.current = Date.now();
        const timelineEvent = streamEventToTimelineEvent(event);
        if (timelineEvent) {
          lastSeenRunEventIdRef.current = Math.max(
            lastSeenRunEventIdRef.current,
            timelineEvent.id,
          );
          setRunTimeline((current) => mergeRunTimelineEvents(current, [timelineEvent]));
        }
        const eventType = String(event.event_type || "").trim().toLowerCase();
        const nextStatus = String(event.status || "").trim().toLowerCase();
        const derivedStatus = deriveRunStatusFromStreamEvent(eventType, nextStatus);
        if (derivedStatus) {
          setRunDetail((current) =>
            current
              ? {
                  ...current,
                  status: derivedStatus,
                }
              : current,
          );
        }
        const shouldStopStreaming =
          event.terminal
          || event.streaming === false
          || eventType === "task_paused"
          || nextStatus === "paused";
        const shouldStartStreaming =
          eventType === "task_resumed"
          || eventType === "task_executing"
          || eventType === "task_planning"
          || nextStatus === "executing"
          || nextStatus === "planning"
          || nextStatus === "running";
        if (shouldStopStreaming) {
          setRunStreaming(false);
        } else if (shouldStartStreaming) {
          setRunStreaming(true);
        }
        scheduleRunRefresh();
      },
      () => {
        lastRunStreamActivityAtRef.current = 0;
        setRunStreaming(false);
        scheduleRunRefresh();
      },
      {
        afterId: lastSeenRunEventIdRef.current,
      },
    );
    return () => {
      setRunStreaming(false);
      cleanup();
    };
  }, [loadingRunDetail, scheduleRunRefresh, selectedRunId]);

  useEffect(() => {
    setRunStreaming(runIsActivelyExecuting);
  }, [runIsActivelyExecuting]);

  useEffect(() => {
    if (!selectedRunId || runIsTerminal) {
      return;
    }
    const timer = window.setInterval(() => {
      const staleForMs = Date.now() - lastRunStreamActivityAtRef.current;
      if (lastRunStreamActivityAtRef.current > 0 && staleForMs < 8000) {
        return;
      }
      void refreshRun(selectedRunId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh run.");
        }
      });
    }, 5000);
    return () => {
      window.clearInterval(timer);
    };
  }, [refreshRun, runIsTerminal, selectedRunId, setError]);

  // Run match scroll tracking
  useEffect(() => {
    runMatchRefs.current = [];
    if (!runHistoryQuery.trim() || totalRunMatches === 0) {
      setActiveRunMatchIndex(0);
      return;
    }
    setActiveRunMatchIndex(0);
    window.setTimeout(() => {
      scrollRunMatchIntoView(0);
    }, 0);
  }, [runHistoryQuery, totalRunMatches]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  async function handleLaunchRun(
    event: FormEvent<HTMLFormElement>,
    extraContext?: Record<string, unknown>,
  ) {
    event.preventDefault();
    if (!overview?.workspace.canonical_path) {
      setError("Select a workspace before launching a run.");
      return;
    }
    const goal = runGoal.trim();
    if (!goal) {
      setError("Run goal is required.");
      return;
    }
    setLaunchingRun(true);
    setError("");
    setNotice("");
    try {
      const created = await createTask({
        goal,
        workspace: overview.workspace.canonical_path,
        process: runProcess.trim() || undefined,
        approval_mode: runApprovalMode || "auto",
        context: extraContext,
        auto_subfolder: true,
      });
      const nextRunId = created.task_id || created.run_id;
      startTransition(() => {
        setSelectedRunId(nextRunId);
        setActiveTab("runs");
      });
      setRunGoal("");
      setRunProcess("");
      setRunApprovalMode("auto");
      setNotice(`Launched run ${nextRunId}.`);
      void refreshWorkspaceSurface(selectedWorkspaceId).catch(() => {});
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to launch run.");
    } finally {
      setLaunchingRun(false);
    }
  }

  async function handleRunControl(action: "pause" | "resume" | "cancel") {
    if (!selectedRunId) {
      setError("Select a run before using controls.");
      return;
    }
    setRunActionPending(action);
    setError("");
    setNotice("");
    try {
      const response =
        action === "pause"
          ? await pauseRun(selectedRunId)
          : action === "resume"
            ? await resumeRun(selectedRunId)
            : await cancelRun(selectedRunId);
      // Optimistically update local state so UI reflects the action immediately
      if (action === "cancel" && runDetail) {
        setRunDetail({ ...runDetail, status: "cancelled" });
        setRunStreaming(false);
      } else if (action === "pause" && runDetail) {
        setRunDetail({ ...runDetail, status: "paused" });
        setRunStreaming(false);
      } else if (action === "resume" && runDetail) {
        setRunDetail({ ...runDetail, status: "executing" });
        setRunStreaming(true);
      }
      await Promise.all([
        refreshRun(selectedRunId),
        selectedWorkspaceId ? refreshWorkspaceSurface(selectedWorkspaceId) : Promise.resolve(),
      ]);
      setNotice(response.message || `Run ${action} requested.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} run.`);
    } finally {
      setRunActionPending("");
    }
  }

  async function handleDeleteRun() {
    if (!selectedRunId) return;
    setRunActionPending("delete");
    setError("");
    setNotice("");
    try {
      const response = await deleteRun(selectedRunId);
      setSelectedRunId("");
      if (selectedWorkspaceId) {
        await refreshWorkspaceSurface(selectedWorkspaceId);
      }
      setNotice(response.message || "Run deleted.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete run.");
    } finally {
      setRunActionPending("");
    }
  }

  async function handleRestartRun() {
    if (!selectedRunId) return;
    setRunActionPending("restart");
    setError("");
    setNotice("");
    try {
      const created = await restartRun(selectedRunId);
      const nextRunId = created.task_id || created.run_id;
      if (selectedWorkspaceId) {
        void refreshWorkspaceSurface(selectedWorkspaceId).catch(() => {});
      }
      startTransition(() => {
        setSelectedRunId(nextRunId);
        setActiveTab("runs");
      });
      setNotice(`Relaunched as ${nextRunId}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to restart run.");
    } finally {
      setRunActionPending("");
    }
  }

  async function handleSendRunMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedRunId) {
      setError("Select a run before sending a message.");
      return;
    }
    const message = runOperatorMessage.trim();
    if (!message) {
      setError("Message is required.");
      return;
    }
    setSendingRunMessage(true);
    setError("");
    setNotice("");
    try {
      const response = await sendRunMessage(selectedRunId, message);
      await refreshRun(selectedRunId);
      setRunOperatorMessage("");
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send run message.");
    } finally {
      setSendingRunMessage(false);
    }
  }

  function handlePrefillStarterRun() {
    setRunGoal(
      "Inspect this workspace, summarize what exists, and recommend the highest-leverage first task.",
    );
    setRunProcess("");
    
    setError("");
  }

  function focusRunComposer() {
    setActiveTab("runs");
    // scrollToSection inlined
    runComposerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    window.setTimeout(() => {
      const target = runComposerRef.current?.querySelector("textarea");
      if (target instanceof HTMLTextAreaElement) {
        target.focus();
      }
    }, 140);
  }

  function scrollRunMatchIntoView(index: number) {
    const target = runMatchRefs.current[index];
    target?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function stepRunMatch(delta: number) {
    if (!runHistoryQuery.trim() || totalRunMatches === 0) {
      return;
    }
    const nextIndex = (activeRunMatchIndex + delta + totalRunMatches) % totalRunMatches;
    setActiveRunMatchIndex(nextIndex);
    window.setTimeout(() => {
      scrollRunMatchIntoView(nextIndex);
    }, 0);
  }

  return {
    // State
    selectedRunId,
    runDetail,
    runTimeline,
    runArtifacts,
    runStreaming,
    loadingRunDetail,
    runLoadError,
    runGoal,
    runProcess,
    runApprovalMode,
    launchingRun,
    runOperatorMessage,
    runActionPending,
    sendingRunMessage,
    runHistoryQuery,
    activeRunMatchIndex,

    // Computed
    normalizedRunStatus,
    runIsTerminal,
    runCanPause,
    runCanResume,
    runCanMessage,
    selectedRunSummary,
    filteredRunArtifacts,
    filteredRunTimeline,
    visibleRunArtifacts,
    visibleRunTimeline,
    totalRunMatches,

    // Refs
    runComposerRef,
    runMatchRefs,

    // Actions
    setSelectedRunId,
    setRunGoal,
    setRunProcess,
    setRunApprovalMode,
    setRunOperatorMessage,
    setRunHistoryQuery,
    setActiveRunMatchIndex,
    handleLaunchRun,
    handleRunControl,
    handleDeleteRun,
    handleRestartRun,
    handleSendRunMessage,
    handlePrefillStarterRun,
    focusRunComposer,
    refreshRun,
    scrollRunMatchIntoView,
    stepRunMatch,
  };
}
