import {
  startTransition,
  type FormEvent,
  useEffect,
  useEffectEvent,
  useMemo,
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
  type RunSummary,
  type RunStreamEvent,
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
    if (isRunTimelineNoise(item)) {
      continue;
    }
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

function maxRunTimelineSequence(events: Array<Pick<RunTimelineEvent, "sequence">>): number {
  return events.reduce(
    (maxSequence, row) => Math.max(maxSequence, Number(row.sequence || 0)),
    0,
  );
}

function canonicalizeRunDetail(detail: RunDetail): RunDetail {
  const normalizedStatus = normalizeRunStatus(detail.status);
  return normalizedStatus && normalizedStatus !== detail.status
    ? { ...detail, status: normalizedStatus }
    : detail;
}

function buildOptimisticRunSummary(params: {
  runId: string;
  executionRunId?: string;
  workspaceId: string;
  workspacePath: string;
  goal: string;
  processName: string;
  status: string;
}): RunSummary {
  const timestamp = new Date().toISOString();
  return {
    id: params.runId,
    workspace_id: params.workspaceId,
    workspace_path: params.workspacePath,
    goal: params.goal,
    status: params.status,
    created_at: timestamp,
    updated_at: timestamp,
    execution_run_id: String(params.executionRunId || "").trim(),
    process_name: params.processName,
    linked_conversation_ids: [],
    changed_files_count: 0,
  };
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
  const normalizedNextStatus = normalizeRunStatus(nextStatus);
  const canTrustStreamStatus =
    eventType === "run_snapshot"
    || eventType.startsWith("task_");
  if (
    canTrustStreamStatus
    && normalizedNextStatus
    && [
      "pending",
      "planning",
      "executing",
      "running",
      "paused",
      "completed",
      "failed",
      "cancelled",
      "waiting_approval",
    ].includes(normalizedNextStatus)
  ) {
    return normalizedNextStatus;
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
  selectedRunSummary: RunDetail | RunSummary | null;
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
  refreshWorkspaceSurface?: (workspaceId: string) => Promise<void>;
  syncRunDetail?: (detail: RunSummary | RunDetail) => void;
  removeRunSummary?: (runId: string, workspaceId?: string) => void;
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
    syncRunDetail,
    removeRunSummary,
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
  const runStreamFrameRef = useRef<number | null>(null);
  const runMatchRefs = useRef<Array<HTMLDivElement | null>>([]);
  const lastSeenRunSequenceRef = useRef(0);
  const lastRunStreamActivityAtRef = useRef(0);
  const nextSyntheticRunEventIdRef = useRef(-1);
  const runDetailRef = useRef<RunDetail | null>(null);
  const runTimelineRef = useRef<RunTimelineEvent[]>([]);
  const runStreamingRef = useRef(false);
  const pendingRunStreamEventsRef = useRef<RunStreamEvent[]>([]);

  // ---------------------------------------------------------------------------
  // Computed values
  // ---------------------------------------------------------------------------

  const normalizedRunStatus = useMemo(
    () => normalizeRunStatus(runDetail?.status),
    [runDetail?.status],
  );
  const runIsActivelyExecuting = useMemo(
    () => isRunActiveStatus(normalizedRunStatus),
    [normalizedRunStatus],
  );
  const runIsTerminal = useMemo(
    () => isRunTerminalStatus(normalizedRunStatus),
    [normalizedRunStatus],
  );
  const runCanPause = useMemo(
    () => canPauseRunStatus(normalizedRunStatus),
    [normalizedRunStatus],
  );
  const runCanResume = useMemo(
    () => canResumeRunStatus(normalizedRunStatus),
    [normalizedRunStatus],
  );
  const runCanMessage = useMemo(
    () => canMessageRunStatus(normalizedRunStatus),
    [normalizedRunStatus],
  );
  const workspaceRunRows = useMemo(
    () => overview?.recent_runs || [],
    [overview?.recent_runs],
  );
  const selectedRunSummary = useMemo(() => (
    runDetail
    || workspaceRunRows.find((run) => run.id === selectedRunId)
    || null
  ), [runDetail, selectedRunId, workspaceRunRows]);
  const filteredRunArtifacts = useMemo(() => runArtifacts
    .filter((artifact) => {
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
    ), [runArtifacts, runHistoryQuery]);
  const filteredRunTimeline = useMemo(() => runTimeline.filter((event) =>
    matchesWorkspaceSearch(
      runHistoryQuery,
      event.event_type,
      runTimelineTitle(event),
      runTimelineDetail(event),
      event.data,
      runTimelinePills(event).join(" "),
    ),
  ), [runHistoryQuery, runTimeline]);
  const visibleRunArtifacts = useMemo(
    () => filteredRunArtifacts,
    [filteredRunArtifacts],
  );
  const visibleRunTimeline = useMemo(() => (
    runHistoryQuery.trim()
      ? filteredRunTimeline
      : runTimeline
  ), [filteredRunTimeline, runHistoryQuery, runTimeline]);
  const totalRunMatches = useMemo(
    () => filteredRunArtifacts.length + filteredRunTimeline.length,
    [filteredRunArtifacts.length, filteredRunTimeline.length],
  );

  async function loadRunSnapshot(runId: string): Promise<[RunDetail, RunTimelineEvent[], RunArtifact[]]> {
    let lastError: unknown = null;
    for (let attempt = 0; attempt < 6; attempt += 1) {
      try {
        return await Promise.all([
          fetchRunDetail(runId),
          fetchRunTimeline(runId, { includeNoise: false }),
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
    const nextDetail = canonicalizeRunDetail(detail);
    const nextTimeline = mergeRunTimelineEvents(runTimelineRef.current, timeline);
    setRunLoadError("");
    runDetailRef.current = nextDetail;
    setRunDetail(nextDetail);
    pushRunDetailToWorkspaceState(nextDetail);
    runTimelineRef.current = nextTimeline;
    setRunTimeline(nextTimeline);
    setRunArtifacts(artifacts);
    lastSeenRunSequenceRef.current = Math.max(
      lastSeenRunSequenceRef.current,
      maxRunTimelineSequence(timeline),
    );
  });

  const refreshRunInstructionHistory = useEffectEvent(async (runId: string) => {
    try {
      const entries = await fetchRunConversationHistory(runId);
      setRunInstructionHistory(entries);
    } catch {
      setRunInstructionHistory([]);
    }
  });

  const pushRunDetailToWorkspaceState = useEffectEvent((detail: RunSummary | RunDetail | null) => {
    if (!detail) {
      return;
    }
    syncRunDetail?.(detail);
  });

  const scheduleRunRefresh = useEffectEvent(() => {
    if (runRefreshTimerRef.current !== null || !selectedRunId) {
      return;
    }
    runRefreshTimerRef.current = window.setTimeout(() => {
      runRefreshTimerRef.current = null;
      void refreshRun(selectedRunId).catch((err) => {
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
    runTimelineRef.current = runTimeline;
  }, [runTimeline]);

  useEffect(() => {
    runStreamingRef.current = runStreaming;
  }, [runStreaming]);

  useEffect(() => {
    return () => {
      if (runRefreshTimerRef.current !== null) {
        window.clearTimeout(runRefreshTimerRef.current);
      }
      if (runStreamFrameRef.current !== null) {
        window.cancelAnimationFrame(runStreamFrameRef.current);
        runStreamFrameRef.current = null;
      }
      pendingRunStreamEventsRef.current = [];
    };
  }, []);

  // Load run detail
  useEffect(() => {
    if (!selectedRunId) {
      runDetailRef.current = null;
      runTimelineRef.current = [];
      runStreamingRef.current = false;
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
      pendingRunStreamEventsRef.current = [];
      if (runStreamFrameRef.current !== null) {
        window.cancelAnimationFrame(runStreamFrameRef.current);
        runStreamFrameRef.current = null;
      }
      return;
    }
    const hasMatchingRunDetail = runDetailRef.current?.id === selectedRunId;
    if (connectionState !== "connected") {
      if (!hasMatchingRunDetail) {
        runDetailRef.current = null;
        runTimelineRef.current = [];
        runStreamingRef.current = false;
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
        pendingRunStreamEventsRef.current = [];
        if (runStreamFrameRef.current !== null) {
          window.cancelAnimationFrame(runStreamFrameRef.current);
          runStreamFrameRef.current = null;
        }
      }
      setLoadingRunDetail(!hasMatchingRunDetail);
      return;
    }
    let cancelled = false;
    setLoadingRunDetail(true);
    setRunLoadError("");
    if (!hasMatchingRunDetail) {
      runDetailRef.current = null;
      runTimelineRef.current = [];
      runStreamingRef.current = false;
      setRunDetail(null);
      setRunTimeline([]);
      setRunArtifacts([]);
      setRunInstructionHistory([]);
      setRunHistoryQuery("");
      setActiveRunMatchIndex(0);
      lastSeenRunSequenceRef.current = 0;
      lastRunStreamActivityAtRef.current = 0;
      nextSyntheticRunEventIdRef.current = -1;
      pendingRunStreamEventsRef.current = [];
      if (runStreamFrameRef.current !== null) {
        window.cancelAnimationFrame(runStreamFrameRef.current);
        runStreamFrameRef.current = null;
      }
    }

    void (async () => {
      try {
        const [snapshot, instructionHistory] = await Promise.all([
          loadRunSnapshot(selectedRunId),
          fetchRunConversationHistory(selectedRunId).catch(() => [] as RunConversationEntry[]),
        ]);
        const [detail, timeline, artifacts] = snapshot;
        if (!cancelled) {
          const nextDetail = canonicalizeRunDetail(detail);
          const nextTimeline = mergeRunTimelineEvents([], timeline);
          runDetailRef.current = nextDetail;
          runTimelineRef.current = nextTimeline;
          setRunDetail(nextDetail);
          pushRunDetailToWorkspaceState(nextDetail);
          setRunTimeline(nextTimeline);
          setRunArtifacts(artifacts);
          setRunInstructionHistory(instructionHistory);
          lastSeenRunSequenceRef.current = maxRunTimelineSequence(timeline);
        }
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Failed to load run.";
          setRunLoadError(message);
          setError(message);
          if (!hasMatchingRunDetail) {
            runDetailRef.current = null;
            runTimelineRef.current = [];
            runStreamingRef.current = false;
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

  const flushRunStreamBatch = useEffectEvent(() => {
    if (runStreamFrameRef.current !== null) {
      window.cancelAnimationFrame(runStreamFrameRef.current);
      runStreamFrameRef.current = null;
    }

    const batch = pendingRunStreamEventsRef.current;
    if (batch.length === 0) {
      return;
    }
    pendingRunStreamEventsRef.current = [];

    let nextTimeline = runTimelineRef.current;
    let nextDetail = runDetailRef.current;
    let nextStreaming = runStreamingRef.current;
    const timelineUpdates: RunTimelineEvent[] = [];
    let shouldRefreshRun = false;

    for (const event of batch) {
      lastRunStreamActivityAtRef.current = Date.now();
      const sequence = Number(event.sequence || 0);
      if (sequence > 0) {
        lastSeenRunSequenceRef.current = Math.max(lastSeenRunSequenceRef.current, sequence);
      }

      const timelineEvent = streamEventToTimelineEvent(
        event,
        typeof event.id === "number" && event.id > 0
          ? undefined
          : nextSyntheticRunEventIdRef.current--,
      );
      if (timelineEvent) {
        timelineUpdates.push(timelineEvent);
      }

      const eventType = String(event.event_type || "").trim().toLowerCase();
      const nextStatus = normalizeRunStatus(String(event.status || ""));
      const derivedStatus = deriveRunStatusFromStreamEvent(eventType, nextStatus);
      if (derivedStatus && nextDetail) {
        nextDetail = {
          ...nextDetail,
          status: derivedStatus,
        };
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
        nextStreaming = false;
      } else if (shouldStartStreaming) {
        nextStreaming = true;
      }

      if (event.terminal) {
        shouldRefreshRun = true;
      }
    }

    if (timelineUpdates.length > 0) {
      nextTimeline = mergeRunTimelineEvents(nextTimeline, timelineUpdates);
      if (nextTimeline !== runTimelineRef.current) {
        runTimelineRef.current = nextTimeline;
        setRunTimeline(nextTimeline);
      }
    }
    if (nextDetail !== runDetailRef.current) {
      runDetailRef.current = nextDetail;
      setRunDetail(nextDetail);
      pushRunDetailToWorkspaceState(nextDetail);
    }
    if (nextStreaming !== runStreamingRef.current) {
      runStreamingRef.current = nextStreaming;
      setRunStreaming(nextStreaming);
    }
    if (shouldRefreshRun) {
      scheduleRunRefresh();
    }
  });

  const queueRunStreamEvent = useEffectEvent((event: RunStreamEvent) => {
    pendingRunStreamEventsRef.current.push(event);
    if (runStreamFrameRef.current !== null) {
      return;
    }
    runStreamFrameRef.current = window.requestAnimationFrame(() => {
      flushRunStreamBatch();
    });
  });

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
    if (runDetailRef.current?.id !== selectedRunId) {
      return;
    }
    const cleanup = subscribeRunStream(
      selectedRunId,
      (event) => {
        queueRunStreamEvent(event);
      },
      () => {
        lastRunStreamActivityAtRef.current = 0;
        runStreamingRef.current = false;
        setRunStreaming(false);
        scheduleRunRefresh();
      },
      {
        afterSequence: lastSeenRunSequenceRef.current,
        includeNoise: false,
      },
    );
    return () => {
      if (runStreamFrameRef.current !== null) {
        window.cancelAnimationFrame(runStreamFrameRef.current);
        runStreamFrameRef.current = null;
      }
      pendingRunStreamEventsRef.current = [];
      runStreamingRef.current = false;
      setRunStreaming(false);
      cleanup();
    };
  }, [
    connectionState,
    loadingRunDetail,
    selectedRunId,
  ]);

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
  }, [runIsTerminal, selectedRunId, setError]);

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
      const processName = runProcess.trim();
      const created = await createTask({
        goal,
        workspace: overview.workspace.canonical_path,
        process: processName || undefined,
        approval_mode: runApprovalMode || "auto",
        context: extraContext,
        auto_subfolder: true,
      });
      const nextRunId = created.task_id || created.run_id;
      const optimisticStatus = normalizeRunStatus(created.status) || "planning";
      if (nextRunId && selectedWorkspaceId) {
        pushRunDetailToWorkspaceState(buildOptimisticRunSummary({
          runId: nextRunId,
          executionRunId: created.run_id,
          workspaceId: selectedWorkspaceId,
          workspacePath: overview.workspace.canonical_path,
          goal,
          processName,
          status: optimisticStatus,
        }));
      }
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
        const nextDetail = canonicalizeRunDetail({ ...runDetail, status: "cancelled" });
        runDetailRef.current = nextDetail;
        setRunDetail(nextDetail);
        pushRunDetailToWorkspaceState(nextDetail);
        setRunStreaming(false);
      } else if (action === "pause" && runDetail) {
        const nextDetail = canonicalizeRunDetail({ ...runDetail, status: "paused" });
        runDetailRef.current = nextDetail;
        setRunDetail(nextDetail);
        pushRunDetailToWorkspaceState(nextDetail);
        setRunStreaming(false);
      } else if (action === "resume" && runDetail) {
        const nextDetail = canonicalizeRunDetail({ ...runDetail, status: "executing" });
        runDetailRef.current = nextDetail;
        setRunDetail(nextDetail);
        pushRunDetailToWorkspaceState(nextDetail);
        setRunStreaming(true);
      }
      await refreshRun(selectedRunId);
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
      removeRunSummary?.(selectedRunId, selectedWorkspaceId);
      setSelectedRunId("");
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
      const optimisticStatus = normalizeRunStatus(created.status) || "planning";
      const fallbackProcessName =
        runDetail?.process_name
        || selectedRunSummary?.process_name
        || "";
      const fallbackGoal = runDetail?.goal || selectedRunSummary?.goal || "";
      if (nextRunId && selectedWorkspaceId && overview?.workspace.canonical_path) {
        pushRunDetailToWorkspaceState(buildOptimisticRunSummary({
          runId: nextRunId,
          executionRunId: created.run_id,
          workspaceId: selectedWorkspaceId,
          workspacePath: overview.workspace.canonical_path,
          goal: fallbackGoal,
          processName: fallbackProcessName,
          status: optimisticStatus,
        }));
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
    runComposerRef.current?.scrollIntoView({ behavior: "auto", block: "start" });
    window.setTimeout(() => {
      const target = runComposerRef.current?.querySelector("textarea");
      if (target instanceof HTMLTextAreaElement) {
        target.focus();
      }
    }, 140);
  }

  function scrollRunMatchIntoView(index: number) {
    const target = runMatchRefs.current[index];
    target?.scrollIntoView({ behavior: "auto", block: "center" });
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
