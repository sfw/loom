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
  fetchRunConversationHistory,
  fetchRunDetail,
  fetchRunTimeline,
  pauseRun,
  restartRun,
  resumeRun,
  sendRunMessage,
  subscribeRunStream,
  type RunArtifact,
  type RunConversationEntry,
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
import {
  canMessageRunStatus,
  canPauseRunStatus,
  canResumeRunStatus,
  isRunActiveStatus,
  isRunTerminalStatus,
  normalizeRunStatus,
} from "../runStatus";
import { isTransientRequestError, type ViewTab } from "../utils";

function isPersistedRunTimelineEvent(event: Pick<RunTimelineEvent, "id">): boolean {
  return typeof event.id === "number" && event.id > 0;
}

function runTimelineEventIdentity(
  event: Pick<
    RunTimelineEvent,
    "id" | "task_id" | "run_id" | "correlation_id" | "event_id" | "sequence" | "timestamp" | "event_type"
  >,
): string {
  const eventId = String(event.event_id || "").trim();
  if (eventId) {
    return `event:${eventId}`;
  }
  const sequence = typeof event.sequence === "number" ? event.sequence : 0;
  if (sequence > 0) {
    return [
      "sequence",
      event.task_id,
      String(event.run_id || "").trim(),
      String(event.correlation_id || "").trim(),
      String(event.event_type || "").trim(),
      String(sequence),
    ].join(":");
  }
  if (isPersistedRunTimelineEvent(event)) {
    return `id:${event.id}`;
  }
  return [
    "fallback",
    event.task_id,
    String(event.run_id || "").trim(),
    String(event.timestamp || "").trim(),
    String(event.event_type || "").trim(),
  ].join(":");
}

function compareRunTimelineEvents(left: RunTimelineEvent, right: RunTimelineEvent): number {
  const leftSequence = typeof left.sequence === "number" ? left.sequence : 0;
  const rightSequence = typeof right.sequence === "number" ? right.sequence : 0;
  if (leftSequence > 0 && rightSequence > 0 && leftSequence !== rightSequence) {
    return leftSequence - rightSequence;
  }
  const timestampCompare = left.timestamp.localeCompare(right.timestamp);
  if (timestampCompare !== 0) {
    return timestampCompare;
  }
  const leftPersisted = isPersistedRunTimelineEvent(left);
  const rightPersisted = isPersistedRunTimelineEvent(right);
  if (leftPersisted !== rightPersisted) {
    return leftPersisted ? 1 : -1;
  }
  const leftId = typeof left.id === "number" ? left.id : 0;
  const rightId = typeof right.id === "number" ? right.id : 0;
  return leftId - rightId;
}

function mergeRunTimelineEvents(
  existing: RunTimelineEvent[],
  incoming: RunTimelineEvent[],
): RunTimelineEvent[] {
  if (incoming.length === 0) {
    return existing;
  }
  const byIdentity = new Map<string, RunTimelineEvent>();
  for (const item of existing) {
    byIdentity.set(runTimelineEventIdentity(item), item);
  }
  for (const item of incoming) {
    const identity = runTimelineEventIdentity(item);
    const previous = byIdentity.get(identity);
    if (previous && isPersistedRunTimelineEvent(previous) && !isPersistedRunTimelineEvent(item)) {
      continue;
    }
    byIdentity.set(identity, previous ? {
      ...previous,
      ...item,
      data:
        item.data && Object.keys(item.data).length > 0
          ? item.data
          : previous.data,
    } : item);
  }
  return Array.from(byIdentity.values()).sort(compareRunTimelineEvents);
}

function canonicalizeRunDetail(detail: RunDetail): RunDetail {
  const normalizedStatus = normalizeRunStatus(detail.status);
  return normalizedStatus && normalizedStatus !== detail.status
    ? { ...detail, status: normalizedStatus }
    : detail;
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
}, syntheticId?: number): RunTimelineEvent | null {
  const eventId =
    typeof event.id === "number" && event.id > 0
      ? event.id
      : (typeof syntheticId === "number" ? syntheticId : null);
  if (eventId === null) {
    return null;
  }
  return {
    id: eventId,
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
    return normalizeRunStatus(nextStatus);
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
  runInstructionHistory: RunConversationEntry[];
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
  connectionState?: "connecting" | "connected" | "failed";
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
    connectionState = "connected",
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
  const [runInstructionHistory, setRunInstructionHistory] = useState<RunConversationEntry[]>([]);
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
  const lastSeenRunSequenceRef = useRef(0);
  const lastRunStreamActivityAtRef = useRef(0);
  const nextSyntheticRunEventIdRef = useRef(-1);
  const runDetailRef = useRef<RunDetail | null>(null);
  const pendingWorkspaceRefreshRef = useRef<{
    runId: string;
    workspaceId: string;
  } | null>(null);

  // ---------------------------------------------------------------------------
  // Computed values
  // ---------------------------------------------------------------------------

  const normalizedRunStatus = normalizeRunStatus(runDetail?.status);
  const runIsActivelyExecuting = isRunActiveStatus(normalizedRunStatus);
  const runIsTerminal = isRunTerminalStatus(normalizedRunStatus);
  const runCanPause = canPauseRunStatus(normalizedRunStatus);
  const runCanResume = canResumeRunStatus(normalizedRunStatus);
  const runCanMessage = canMessageRunStatus(normalizedRunStatus);
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

  const maybeRefreshWorkspaceSurfaceAfterRunLoad = useEffectEvent((runId: string) => {
    const pending = pendingWorkspaceRefreshRef.current;
    if (!pending || pending.runId !== runId || !pending.workspaceId) {
      return;
    }
    pendingWorkspaceRefreshRef.current = null;
    void refreshWorkspaceSurface(pending.workspaceId).catch(() => {});
  });

  const refreshRun = useEffectEvent(async (runId: string) => {
    const [detail, timeline, artifacts] = await loadRunSnapshot(runId);
    setRunLoadError("");
    setRunDetail(canonicalizeRunDetail(detail));
    setRunTimeline((current) => mergeRunTimelineEvents(current, timeline));
    setRunArtifacts(artifacts);
    lastSeenRunSequenceRef.current = Math.max(
      lastSeenRunSequenceRef.current,
      timeline.reduce(
        (maxSequence, row) => Math.max(maxSequence, Number(row.sequence || 0)),
        0,
      ),
    );
    maybeRefreshWorkspaceSurfaceAfterRunLoad(runId);
  });

  const refreshRunInstructionHistory = useEffectEvent(async (runId: string) => {
    try {
      const entries = await fetchRunConversationHistory(runId);
      setRunInstructionHistory(entries);
    } catch {
      setRunInstructionHistory([]);
    }
  });

  const scheduleRunRefresh = useEffectEvent((options?: { includeWorkspaceSurface?: boolean }) => {
    if (runRefreshTimerRef.current !== null || !selectedRunId) {
      return;
    }
    runRefreshTimerRef.current = window.setTimeout(() => {
      runRefreshTimerRef.current = null;
      const includeWorkspaceSurface = Boolean(options?.includeWorkspaceSurface);
      void Promise.all([
        refreshRun(selectedRunId),
        includeWorkspaceSurface && selectedWorkspaceId
          ? refreshWorkspaceSurface(selectedWorkspaceId)
          : Promise.resolve(),
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
    runDetailRef.current = runDetail;
  }, [runDetail]);

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
      setRunInstructionHistory([]);
      setLoadingRunDetail(false);
      setRunLoadError("");
      setRunHistoryQuery("");
      setActiveRunMatchIndex(0);
      lastSeenRunSequenceRef.current = 0;
      lastRunStreamActivityAtRef.current = 0;
      nextSyntheticRunEventIdRef.current = -1;
      pendingWorkspaceRefreshRef.current = null;
      return;
    }
    const hasMatchingRunDetail = runDetailRef.current?.id === selectedRunId;
    if (connectionState !== "connected") {
      if (!hasMatchingRunDetail) {
        setRunDetail(null);
        setRunTimeline([]);
        setRunArtifacts([]);
        setRunInstructionHistory([]);
        setRunLoadError("");
        setRunHistoryQuery("");
        setActiveRunMatchIndex(0);
        lastSeenRunSequenceRef.current = 0;
        lastRunStreamActivityAtRef.current = 0;
        nextSyntheticRunEventIdRef.current = -1;
      }
      setLoadingRunDetail(!hasMatchingRunDetail);
      return;
    }
    let cancelled = false;
    setLoadingRunDetail(true);
    setRunLoadError("");
    if (!hasMatchingRunDetail) {
      setRunDetail(null);
      setRunTimeline([]);
      setRunArtifacts([]);
      setRunInstructionHistory([]);
      setRunHistoryQuery("");
      setActiveRunMatchIndex(0);
      lastSeenRunSequenceRef.current = 0;
      lastRunStreamActivityAtRef.current = 0;
      nextSyntheticRunEventIdRef.current = -1;
    }

    void (async () => {
      try {
        const [snapshot, instructionHistory] = await Promise.all([
          loadRunSnapshot(selectedRunId),
          fetchRunConversationHistory(selectedRunId).catch(() => [] as RunConversationEntry[]),
        ]);
        const [detail, timeline, artifacts] = snapshot;
        if (!cancelled) {
          setRunDetail(canonicalizeRunDetail(detail));
          setRunTimeline(timeline);
          setRunArtifacts(artifacts);
          setRunInstructionHistory(instructionHistory);
          lastSeenRunSequenceRef.current = timeline.reduce(
            (maxSequence, row) => Math.max(maxSequence, Number(row.sequence || 0)),
            0,
          );
          maybeRefreshWorkspaceSurfaceAfterRunLoad(selectedRunId);
        }
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Failed to load run.";
          setRunLoadError(message);
          setError(message);
          if (!hasMatchingRunDetail) {
            setRunDetail(null);
            setRunTimeline([]);
            setRunArtifacts([]);
            setRunInstructionHistory([]);
          }
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
  }, [connectionState, selectedRunId]);

  // Run stream subscription
  useEffect(() => {
    if (connectionState !== "connected") {
      setRunStreaming(false);
      return;
    }
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
        const timelineEvent = streamEventToTimelineEvent(
          event,
          typeof event.id === "number" && event.id > 0
            ? undefined
            : nextSyntheticRunEventIdRef.current--,
        );
        if (timelineEvent) {
          if (Number(timelineEvent.sequence || 0) > 0) {
            lastSeenRunSequenceRef.current = Math.max(
              lastSeenRunSequenceRef.current,
              Number(timelineEvent.sequence || 0),
            );
          }
          setRunTimeline((current) => mergeRunTimelineEvents(current, [timelineEvent]));
        }
        const eventType = String(event.event_type || "").trim().toLowerCase();
        const nextStatus = normalizeRunStatus(String(event.status || ""));
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
        if (event.terminal) {
          scheduleRunRefresh({ includeWorkspaceSurface: true });
        }
      },
      () => {
        lastRunStreamActivityAtRef.current = 0;
        setRunStreaming(false);
        scheduleRunRefresh();
      },
      {
        afterSequence: lastSeenRunSequenceRef.current,
      },
    );
    return () => {
      setRunStreaming(false);
      cleanup();
    };
  }, [connectionState, loadingRunDetail, scheduleRunRefresh, selectedRunId]);

  useEffect(() => {
    setRunStreaming(runIsActivelyExecuting);
  }, [runIsActivelyExecuting]);

  useEffect(() => {
    if (!selectedRunId || runIsTerminal) {
      return;
    }
    const timer = window.setInterval(() => {
      const staleForMs = Date.now() - lastRunStreamActivityAtRef.current;
      if (lastRunStreamActivityAtRef.current > 0 && staleForMs < 15000) {
        return;
      }
      void refreshRun(selectedRunId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh run.");
        }
      });
    }, 10000);
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
      pendingWorkspaceRefreshRef.current = selectedWorkspaceId
        ? { runId: nextRunId, workspaceId: selectedWorkspaceId }
        : null;
      startTransition(() => {
        setSelectedRunId(nextRunId);
        setActiveTab("runs");
      });
      setRunGoal("");
      setRunProcess("");
      setRunApprovalMode("auto");
      setNotice(`Launched run ${nextRunId}.`);
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
        setRunDetail(canonicalizeRunDetail({ ...runDetail, status: "cancelled" }));
        setRunStreaming(false);
      } else if (action === "pause" && runDetail) {
        setRunDetail(canonicalizeRunDetail({ ...runDetail, status: "paused" }));
        setRunStreaming(false);
      } else if (action === "resume" && runDetail) {
        setRunDetail(canonicalizeRunDetail({ ...runDetail, status: "executing" }));
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
      pendingWorkspaceRefreshRef.current = selectedWorkspaceId
        ? { runId: nextRunId, workspaceId: selectedWorkspaceId }
        : null;
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
      await Promise.all([
        refreshRun(selectedRunId),
        refreshRunInstructionHistory(selectedRunId),
      ]);
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
    runInstructionHistory,
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
