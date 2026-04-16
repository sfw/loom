import { useState, useMemo, useRef, useEffect, useEffectEvent, useCallback, useDeferredValue, lazy, Suspense, type FormEvent } from "react";

const Markdown = lazy(() => import("react-markdown"));
import {
  Search,
  Pause,
  Play,
  XCircle,
  Send,
  ChevronUp,
  ChevronDown,
  Zap,
  FileText,
  ArrowLeft,
  Shield,
  Loader2,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Wrench,
  FolderOpen,
  Trash2,
  RotateCcw,
  RefreshCw,
  Sparkles,
  Package,
  FolderTree,
} from "lucide-react";
import {
  shallowEqual,
  useAppActions,
  useApp,
  useAppSelector,
} from "@/context/AppContext";
import type { RunTimelineEvent } from "@/api";
import {
  approvalQuestionContext,
  approvalQuestionOptions,
  approvalQuestionType,
  formatDate,
  formatBytes,
  highlightText,
} from "@/utils";
import {
  runTimelineTitle,
  runTimelineDetail,
  runTimelinePills,
  runTimelineToolArgs,
  runTimelineToolName,
} from "../history";
import { displayRunStatus, normalizeRunStatus } from "../runStatus";
import { cn } from "@/lib/utils";
import {
  buildWorkspaceAttachmentOptions,
  rankWorkspaceAttachmentSuggestions,
  workspaceAttachmentName,
} from "@/workspacePathAttachments";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SAGE = {
  button: "#6b7a5e",
  buttonHover: "#8a9a7b",
  border: "#8a9a7b",
  text: "#a3b396",
  textLight: "#bec8b4",
} as const;

const APPROVAL_MODES = [
  { value: "auto", label: "Auto", desc: "Gate destructive ops" },
  { value: "manual", label: "Manual", desc: "Gate every step" },
  { value: "disabled", label: "Disabled", desc: "No gating" },
] as const;

const MAX_RENDERED_ACTIVITY_EVENTS = 250;
const MAX_TOOL_ARGS_PREVIEW_CHARS = 12_000;
const MAX_ACTIVITY_DETAIL_RENDER_CHARS = 8_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function eventTypeColor(eventType: string): string {
  if (eventType === "tool_call_completed") return "bg-[#6b7a5e]/20 text-[#a3b396]";
  if (eventType === "tool_call_started") return "bg-[#6b7a5e]/10 text-[#a3b396]/70";
  if (eventType === "subtask_completed") return "bg-emerald-500/15 text-emerald-400";
  if (eventType === "subtask_started") return "bg-sky-500/15 text-sky-400";
  if (eventType === "subtask_failed" || eventType === "task_failed") return "bg-red-500/15 text-red-400";
  if (eventType === "task_restarted") return "bg-sky-500/15 text-sky-400";
  if (eventType === "approval_rejected" || eventType === "approval_timed_out") return "bg-red-500/15 text-red-400";
  if (eventType === "verification_passed") return "bg-emerald-500/10 text-emerald-400/80";
  if (eventType === "verification_failed") return "bg-red-500/15 text-red-400";
  if (eventType.startsWith("verification_")) return "bg-amber-500/10 text-amber-400/80";
  if (eventType.includes("fail") || eventType.includes("error") || eventType.includes("stalled")) return "bg-red-500/15 text-red-400";
  if (eventType === "task_completed") return "bg-emerald-500/15 text-emerald-300";
  if (eventType.startsWith("approval_") || eventType.startsWith("ask_user_")) return "bg-violet-500/15 text-violet-400";
  if (eventType === "model_invocation") return "bg-zinc-800/60 text-zinc-500";
  if (eventType.startsWith("task_")) return "bg-zinc-700/50 text-zinc-400";
  return "bg-zinc-800 text-zinc-500";
}

/** Extract structured highlights from event data — files, duration, key metrics. */
function extractEventHighlights(event: { event_type: string; data: Record<string, unknown> }): {
  files: string[];
  duration: string;
  metrics: string[];
} {
  const d = event.data;
  const args = (typeof d.args === "object" && d.args !== null ? d.args : {}) as Record<string, unknown>;

  // Files
  const rawFiles = d.files_changed ?? d.files_changed_paths;
  const files: string[] = Array.isArray(rawFiles)
    ? rawFiles.filter((f): f is string => typeof f === "string" && f.trim() !== "")
    : [];
  // Also extract **filename.ext** and `filename.ext` patterns from markdown text
  if (files.length === 0) {
    const summary = String(d.summary || d.message || "");
    // Match **file.ext** and `file.ext` patterns
    const mdBold = summary.match(/\*\*([^*]+\.\w{1,6})\*\*/g);
    const mdCode = summary.match(/`([^`]+\.\w{1,6})`/g);
    const seen = new Set<string>();
    for (const m of mdBold ?? []) {
      const name = m.replace(/\*\*/g, "").trim();
      if (name && !seen.has(name)) { seen.add(name); files.push(name); }
    }
    for (const m of mdCode ?? []) {
      const name = m.replace(/`/g, "").trim();
      if (name && !seen.has(name)) { seen.add(name); files.push(name); }
    }
  }
  // Path from tool args or artifact path
  if (files.length === 0) {
    const p = String(d.path || d.artifact_path || args.path || "").trim();
    if (p) files.push(p);
  }

  // Duration
  const elapsed = typeof d.elapsed_ms === "number" ? d.elapsed_ms : (typeof d.duration === "number" ? d.duration * 1000 : 0);
  const duration = elapsed >= 1000 ? `${(elapsed / 1000).toFixed(1)}s` : elapsed > 0 ? `${Math.round(elapsed)}ms` : "";

  // Key metrics
  const metrics: string[] = [];
  if (typeof d.confidence === "number") metrics.push(`confidence ${Math.round(Number(d.confidence) * 100)}%`);
  if (typeof d.tier === "number") metrics.push(`tier ${d.tier}`);
  if (typeof d.extracted === "number") metrics.push(`${d.extracted} claims`);
  if (typeof d.supported === "number") metrics.push(`${d.supported} supported`);
  if (typeof d.contradicted === "number" && Number(d.contradicted) > 0) metrics.push(`${d.contradicted} contradicted`);
  if (typeof d.score === "number") metrics.push(`score ${d.score}`);
  const model = typeof d.model === "string" ? d.model.trim() : "";
  if (model) metrics.push(model);
  const tokens = typeof d.request_est_tokens === "number" ? `~${d.request_est_tokens} tok` : "";
  if (tokens) metrics.push(tokens);

  return { files, duration, metrics };
}

/** True if the detail text is long enough to warrant collapsing. */
function isLongDetail(detail: string): boolean {
  return detail.length > 120;
}

function eventTypeBadgeLabel(eventType: string): string {
  switch (eventType) {
    case "tool_call_started": return "TOOL";
    case "tool_call_completed": return "TOOL";
    case "subtask_started": return "STARTED";
    case "subtask_completed": return "DONE";
    case "subtask_failed": return "FAILED";
    case "subtask_blocked": return "BLOCKED";
    case "subtask_retrying": return "RETRY";
    case "verification_started": return "VERIFY";
    case "verification_passed": return "PASS";
    case "verification_failed": return "FAIL";
    case "verification_outcome": return "VERIFY";
    case "verification_rule_applied": return "RULE";
    case "verification_rule_skipped": return "SKIP";
    case "model_invocation": return "MODEL";
    case "task_completed": return "DONE";
    case "task_failed": return "FAILED";
    case "task_cancelled": return "CANCEL";
    case "task_restarted": return "RESTART";
    case "task_plan_ready": return "PLAN";
    case "approval_requested": return "APPROVE";
    case "approval_received": return "APPROVED";
    case "approval_rejected": return "REJECTED";
    case "approval_timed_out": return "TIMEOUT";
    case "ask_user_requested": return "INPUT";
    case "run_validity_scorecard": return "SCORE";
    default: {
      const parts = eventType.replace(/_/g, " ").split(" ");
      return parts.slice(0, 2).join(" ").toUpperCase().slice(0, 12);
    }
  }
}

function subtaskStatus(events: Array<{ event_type: string; data: Record<string, unknown> }>, subtaskId: string): "pending" | "running" | "completed" | "failed" {
  const matching = events.filter(
    (e) =>
      (e.event_type.startsWith("subtask_") || e.event_type.startsWith("phase_")) &&
      (e.data.subtask_id === subtaskId || e.data.phase_id === subtaskId),
  );
  if (matching.some((e) => e.event_type === "subtask_completed" || e.data.status === "completed")) return "completed";
  if (matching.some((e) => e.event_type === "subtask_failed" || e.data.status === "failed")) return "failed";
  if (matching.some((e) => e.event_type === "subtask_started" || e.data.status === "running")) return "running";
  return "pending";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Scroll button with accelerating hold-to-scroll. */
function ScrollButton({ direction, scrollRef }: { direction: "up" | "down"; scrollRef: React.RefObject<HTMLDivElement | null> }) {
  const intervalRef = useRef<number | null>(null);
  const speedRef = useRef(60);

  const startScroll = useCallback(() => {
    speedRef.current = 60;
    const tick = () => {
      const el = scrollRef.current;
      if (!el) return;
      el.scrollBy({
        top: direction === "up" ? -speedRef.current : speedRef.current,
        behavior: "auto",
      });
      speedRef.current = Math.min(speedRef.current + 20, 400);
      intervalRef.current = window.setTimeout(tick, 80);
    };
    tick();
  }, [direction, scrollRef]);

  const stopScroll = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearTimeout(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => stopScroll, [stopScroll]);

  return (
    <button
      type="button"
      onMouseDown={startScroll}
      onMouseUp={stopScroll}
      onMouseLeave={stopScroll}
      className="rounded p-0.5 hover:bg-zinc-800 text-zinc-600 hover:text-zinc-400 transition-colors"
    >
      {direction === "up" ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
    </button>
  );
}

function hasToolArgs(event: RunTimelineEvent): boolean {
  const args = event.data.args;
  return Boolean(args && typeof args === "object" && Object.keys(args).length > 0);
}

function truncatedPreview(text: string, maxChars: number): { text: string; truncated: boolean } {
  if (text.length <= maxChars) {
    return { text, truncated: false };
  }
  return {
    text: `${text.slice(0, maxChars)}\n\n...\n\n[truncated in desktop preview]`,
    truncated: true,
  };
}

function ActivityDetailMarkdown({ detail }: { detail: string }) {
  const [open, setOpen] = useState(false);
  const preview = useMemo(
    () => (open ? truncatedPreview(detail, MAX_ACTIVITY_DETAIL_RENDER_CHARS) : null),
    [detail, open],
  );

  return (
    <details
      className="mt-1.5"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="text-[10px] text-zinc-600 cursor-pointer hover:text-zinc-400 transition-colors select-none">
        Show details
      </summary>
      {open && preview && (
        <div className="mt-1 border-l-2 border-zinc-800 pl-2.5 text-[11px] text-zinc-400 leading-relaxed prose-invert prose-xs [&_strong]:text-zinc-300 [&_em]:text-zinc-400 [&_code]:text-[#a3b396] [&_code]:bg-zinc-800/60 [&_code]:px-1 [&_code]:py-px [&_code]:rounded [&_code]:text-[10px] [&_ul]:list-disc [&_ul]:pl-4 [&_ul]:my-1 [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:my-1 [&_li]:my-0.5 [&_p]:my-1 [&_h1]:text-xs [&_h1]:font-semibold [&_h1]:text-zinc-300 [&_h2]:text-xs [&_h2]:font-semibold [&_h2]:text-zinc-300 [&_h3]:text-[11px] [&_h3]:font-semibold [&_h3]:text-zinc-300">
          <Suspense fallback={<p className="text-zinc-600">Loading...</p>}>
            <Markdown>{preview.text}</Markdown>
          </Suspense>
          {preview.truncated && (
            <p className="mt-2 text-[10px] text-zinc-600">
              Detail was truncated to keep the desktop responsive.
            </p>
          )}
        </div>
      )}
    </details>
  );
}

function ActivityToolArgs({ event }: { event: RunTimelineEvent }) {
  const [open, setOpen] = useState(false);
  const preview = useMemo(() => {
    if (!open) {
      return null;
    }
    return truncatedPreview(runTimelineToolArgs(event), MAX_TOOL_ARGS_PREVIEW_CHARS);
  }, [event, open]);

  if (!hasToolArgs(event)) {
    return null;
  }

  return (
    <details
      className="mt-1.5"
      onToggle={(toggleEvent) => setOpen(toggleEvent.currentTarget.open)}
    >
      <summary className="text-[10px] text-zinc-600 cursor-pointer hover:text-zinc-400 transition-colors select-none">
        Show tool call
      </summary>
      {open && preview && (
        <div>
          <pre className="mt-1 border-l-2 border-zinc-800 pl-2.5 text-[10px] text-zinc-500 font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap break-all max-h-60 overflow-y-auto">
            {preview.text}
          </pre>
          {preview.truncated && (
            <p className="mt-2 text-[10px] text-zinc-600">
              Tool-call args were truncated to keep the desktop responsive.
            </p>
          )}
        </div>
      )}
    </details>
  );
}

function RunDot({ status }: { status: string }) {
  const s = normalizeRunStatus(status);
  const active = s === "executing" || s === "planning" || s === "running";
  return (
    <span
      className={cn(
        "h-2 w-2 shrink-0 rounded-full",
        active
          ? "bg-sky-400 animate-pulse"
          : s === "failed"
            ? "bg-red-400"
            : s === "completed"
              ? "bg-emerald-400"
              : s === "paused"
                ? "bg-amber-400"
                : "bg-zinc-700",
      )}
    />
  );
}

function StatusBadge({ status }: { status: string }) {
  const s = normalizeRunStatus(status);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
        s === "executing" || s === "planning" || s === "running"
          ? "bg-sky-500/15 text-sky-400"
          : s === "completed"
            ? "bg-emerald-500/15 text-emerald-300"
            : s === "failed"
              ? "bg-red-500/15 text-red-400"
              : s === "paused"
                ? "bg-amber-500/15 text-amber-400"
                : s === "cancelled"
                  ? "bg-zinc-800 text-zinc-600"
                  : "bg-zinc-800 text-zinc-500",
      )}
    >
      <RunDot status={status} />
      {displayRunStatus(status)}
    </span>
  );
}

function SubtaskStatusIcon({
  status,
  animated = true,
}: {
  status: "pending" | "running" | "completed" | "failed";
  animated?: boolean;
}) {
  switch (status) {
    case "completed":
      return <CheckCircle2 size={14} className="text-emerald-400" />;
    case "failed":
      return <AlertTriangle size={14} className="text-red-400" />;
    case "running":
      return <Loader2 size={14} className={cn("text-sky-400", animated && "animate-spin")} />;
    default:
      return <Clock size={14} className="text-zinc-600" />;
  }
}

function normalizeProcessQuery(value: string): string {
  return String(value || "").trim().toLowerCase();
}

function processPromptHint(process: {
  name: string;
  description?: string;
} | null): string {
  if (!process) {
    return "Give me a challenge";
  }

  const name = normalizeProcessQuery(process.name);
  const description = normalizeProcessQuery(process.description || "");
  const haystack = `${name} ${description}`;

  if (
    haystack.includes("seo")
    || haystack.includes("geo")
    || haystack.includes("website")
    || haystack.includes("url")
    || haystack.includes("accessibility")
  ) {
    return "Paste the site URL and what you want reviewed";
  }
  if (
    haystack.includes("investment")
    || haystack.includes("portfolio")
    || haystack.includes("equities")
    || haystack.includes("valuation")
  ) {
    return "Describe the company, thesis, portfolio, or investor constraint to analyze";
  }
  if (
    haystack.includes("competitive")
    || haystack.includes("competitor")
    || haystack.includes("market-gap")
  ) {
    return "Name the company, product, or market you want benchmarked";
  }
  if (
    haystack.includes("research")
    || haystack.includes("report")
    || haystack.includes("consulting")
  ) {
    return "Describe the question, topic, or business problem you want investigated";
  }
  if (
    haystack.includes("marketing")
    || haystack.includes("campaign")
    || haystack.includes("audience")
    || haystack.includes("youtube")
  ) {
    return "Share the product, audience, and outcome you want this process to produce";
  }
  if (
    haystack.includes("prd")
    || haystack.includes("software")
    || haystack.includes("design")
  ) {
    return "Describe the product idea, user problem, or feature you want designed";
  }

  return "Tell this process what to work on";
}

type ProcessBucket = "custom" | "installed" | "builtin";

function normalizePath(value: string): string {
  return String(value || "").trim().replace(/\\/g, "/").replace(/\/+$/, "");
}

function pathBasename(path: string): string {
  const normalized = normalizePath(path);
  if (!normalized) return "";
  const parts = normalized.split("/").filter(Boolean);
  return parts[parts.length - 1] || "";
}

function inferProcessBucket(
  process: { path?: string },
  workspacePath: string,
): ProcessBucket {
  const processPath = normalizePath(process.path || "");
  const normalizedWorkspace = normalizePath(workspacePath);
  if (
    normalizedWorkspace
    && (
      processPath.startsWith(`${normalizedWorkspace}/loom-processes/`)
      || processPath.startsWith(`${normalizedWorkspace}/.loom/processes/`)
      || processPath === `${normalizedWorkspace}/loom-processes`
      || processPath === `${normalizedWorkspace}/.loom/processes`
    )
  ) {
    return "custom";
  }
  if (processPath.includes("/.loom/processes/") || processPath.endsWith("/.loom/processes")) {
    return "installed";
  }
  return "builtin";
}

function processBucketLabel(bucket: ProcessBucket): string {
  switch (bucket) {
    case "custom":
      return "Custom";
    case "installed":
      return "Installed";
    default:
      return "Built-in";
  }
}

function processBucketDescription(bucket: ProcessBucket): string {
  switch (bucket) {
    case "custom":
      return "Workspace-local processes that shape this project.";
    case "installed":
      return "Processes installed into your personal Loom environment.";
    default:
      return "Processes that ship with Loom by default.";
  }
}

function pathOptionParent(path: string): string {
  const normalized = normalizePath(path);
  const parts = normalized.split("/").filter(Boolean);
  return parts.length > 1 ? parts.slice(0, -1).join("/") : "";
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function RunsTab() {
  const {
    activeRunMatchIndex,
    approvalInbox,
    approvalReplyDrafts,
    filteredRuns,
    inventory,
    launchingRun,
    loadedWorkspaceFileEntries,
    loadingRunDetail,
    overview,
    replyingApprovalId,
    runActionPending,
    runApprovalMode,
    runArtifacts,
    runCanMessage,
    runCanPause,
    runCanResume,
    runDetail,
    runGoal,
    runHistoryQuery,
    runInstructionHistory,
    runIsTerminal,
    runLoadError,
    runMatchRefs,
    runOperatorMessage,
    runProcess,
    runStreaming,
    runTimeline,
    selectedRunId,
    selectedRunSummary,
    selectedWorkspaceId,
    sendingRunMessage,
    totalRunMatches,
    visibleRunArtifacts,
    visibleRunTimeline,
    workspaceArtifacts,
    workspaceSearchQuery,
  } = useAppSelector((state) => ({
    activeRunMatchIndex: state.activeRunMatchIndex,
    approvalInbox: state.approvalInbox,
    approvalReplyDrafts: state.approvalReplyDrafts,
    filteredRuns: state.filteredRuns,
    inventory: state.inventory,
    launchingRun: state.launchingRun,
    loadedWorkspaceFileEntries: state.loadedWorkspaceFileEntries,
    loadingRunDetail: state.loadingRunDetail,
    overview: state.overview,
    replyingApprovalId: state.replyingApprovalId,
    runActionPending: state.runActionPending,
    runApprovalMode: state.runApprovalMode,
    runArtifacts: state.runArtifacts,
    runCanMessage: state.runCanMessage,
    runCanPause: state.runCanPause,
    runCanResume: state.runCanResume,
    runDetail: state.runDetail,
    runGoal: state.runGoal,
    runHistoryQuery: state.runHistoryQuery,
    runInstructionHistory: state.runInstructionHistory,
    runIsTerminal: state.runIsTerminal,
    runLoadError: state.runLoadError,
    runMatchRefs: state.runMatchRefs,
    runOperatorMessage: state.runOperatorMessage,
    runProcess: state.runProcess,
    runStreaming: state.runStreaming,
    runTimeline: state.runTimeline,
    selectedRunId: state.selectedRunId,
    selectedRunSummary: state.selectedRunSummary,
    selectedWorkspaceId: state.selectedWorkspaceId,
    sendingRunMessage: state.sendingRunMessage,
    totalRunMatches: state.totalRunMatches,
    visibleRunArtifacts: state.visibleRunArtifacts,
    visibleRunTimeline: state.visibleRunTimeline,
    workspaceArtifacts: state.workspaceArtifacts,
    workspaceSearchQuery: state.workspaceSearchQuery,
  }), shallowEqual);
  const {
    handleDeleteRun,
    handleLaunchRun,
    handleOpenWorkspaceFile,
    handleReplyApproval,
    handleRestartRun,
    handleRefreshWorkspaceFiles,
    handleRunControl,
    handleSendRunMessage,
    refreshWorkspaceArtifacts,
    refreshRun,
    setActiveTab,
    setApprovalReplyDrafts,
    setRunApprovalMode,
    setRunGoal,
    setRunHistoryQuery,
    setRunOperatorMessage,
    setRunProcess,
    setSelectedRunId,
    stepRunMatch,
  } = useAppActions();

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activitySearch, setActivitySearch] = useState("");
  const [processQuery, setProcessQuery] = useState("");
  const [attachQuery, setAttachQuery] = useState("");
  const [attachedPaths, setAttachedPaths] = useState<string[]>([]);
  const lastLauncherContextRefreshAtRef = useRef(0);

  const recentWorkspaceArtifacts = useMemo(
    () => [...workspaceArtifacts]
      .filter((artifact) => artifact.exists_on_disk)
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
      .slice(0, 24),
    [workspaceArtifacts],
  );

  const highlightQuery = runHistoryQuery || workspaceSearchQuery || "";
  const processes = inventory?.processes || [];
  const disabled = !selectedWorkspaceId || launchingRun;
  const workspacePath = overview?.workspace?.canonical_path || "";
  const selectedProcessInfo = useMemo(
    () => processes.find((process) => process.name === runProcess) || null,
    [processes, runProcess],
  );
  const normalizedProcessQuery = normalizeProcessQuery(processQuery);
  const groupedProcesses = useMemo(() => {
    const scored = processes
      .map((process, index) => {
        const name = normalizeProcessQuery(process.name);
        const description = normalizeProcessQuery(process.description || "");
        const author = normalizeProcessQuery(process.author || "");
        const bucket = inferProcessBucket(process, workspacePath);
        const text = `${name} ${description} ${author} ${bucket}`;
        const selected = process.name === runProcess;
        const matches = !normalizedProcessQuery || text.includes(normalizedProcessQuery);
        return { process, index, selected, matches, bucket };
      })
      .filter((entry) => entry.matches)
      .sort((left, right) => {
        if (left.selected !== right.selected) {
          return left.selected ? -1 : 1;
        }
        return left.index - right.index;
      });
    const grouped = new Map<ProcessBucket, typeof scored>();
    grouped.set("custom", []);
    grouped.set("installed", []);
    grouped.set("builtin", []);
    for (const entry of scored) {
      grouped.get(entry.bucket)!.push(entry);
    }
    return grouped;
  }, [normalizedProcessQuery, processes, runProcess]);
  const launchPlaceholder = processPromptHint(selectedProcessInfo);
  const attachablePathOptions = useMemo(() => {
    return buildWorkspaceAttachmentOptions({
      workspaceEntries: loadedWorkspaceFileEntries,
      recentArtifacts: recentWorkspaceArtifacts,
    });
  }, [loadedWorkspaceFileEntries, recentWorkspaceArtifacts]);
  const attachablePathLookup = useMemo(
    () => new Map(attachablePathOptions.map((option) => [option.path, option])),
    [attachablePathOptions],
  );
  const launchContext = useMemo(() => {
    if (attachedPaths.length === 0) return undefined;
    const directories = attachedPaths.filter((path) => attachablePathLookup.get(path)?.isDir);
    const files = attachedPaths.filter((path) => !attachablePathLookup.get(path)?.isDir);
    return {
      workspace_paths: attachedPaths,
      workspace_files: files,
      workspace_directories: directories,
    };
  }, [attachablePathLookup, attachedPaths]);
  const visibleAttachSuggestions = useMemo(() => {
    return rankWorkspaceAttachmentSuggestions({
      options: attachablePathOptions,
      query: attachQuery,
      selectedPaths: attachedPaths,
      limit: attachQuery.trim() ? 24 : 18,
    });
  }, [attachQuery, attachedPaths, attachablePathOptions]);

  const refreshLauncherContext = useEffectEvent((force = false) => {
    if (selectedRunId || !selectedWorkspaceId) {
      return;
    }
    const now = Date.now();
    if (!force && (now - lastLauncherContextRefreshAtRef.current) < 1000) {
      return;
    }
    lastLauncherContextRefreshAtRef.current = now;
    void Promise.allSettled([
      handleRefreshWorkspaceFiles({ silent: true }),
      refreshWorkspaceArtifacts(selectedWorkspaceId, { force: true }),
    ]);
  });

  useEffect(() => {
    if (selectedRunId || !selectedWorkspaceId) {
      return undefined;
    }
    refreshLauncherContext(true);

    function handleWindowFocus() {
      refreshLauncherContext();
    }

    function handleVisibilityChange() {
      if (document.visibilityState === "visible") {
        refreshLauncherContext();
      }
    }

    window.addEventListener("focus", handleWindowFocus);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("focus", handleWindowFocus);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [refreshLauncherContext, selectedRunId, selectedWorkspaceId]);

  useEffect(() => {
    setAttachedPaths([]);
    setAttachQuery("");
  }, [selectedWorkspaceId]);

  useEffect(() => {
    setAttachedPaths((current) => {
      const next = current.filter((path) => attachablePathLookup.has(path));
      return next.length === current.length ? current : next;
    });
  }, [attachablePathLookup]);

  function toggleAttachedPath(path: string) {
    setAttachedPaths((current) =>
      current.includes(path)
        ? current.filter((item) => item !== path)
        : [...current, path],
    );
    setAttachQuery("");
  }

  // --- Mode routing ---
  if (selectedRunId) {
    if (runDetail) {
      return (
        <RunDetailView
          runDetail={runDetail}
          runTimeline={runTimeline}
          runArtifacts={runArtifacts}
          runInstructionHistory={runInstructionHistory}
          runStreaming={runStreaming}
          runIsTerminal={runIsTerminal}
          runCanPause={runCanPause}
          runCanResume={runCanResume}
          runCanMessage={runCanMessage}
          approvalInbox={approvalInbox}
          approvalReplyDrafts={approvalReplyDrafts}
          setApprovalReplyDrafts={setApprovalReplyDrafts}
          replyingApprovalId={replyingApprovalId}
          handleReplyApproval={handleReplyApproval}
          visibleRunTimeline={visibleRunTimeline}
          visibleRunArtifacts={visibleRunArtifacts}
          runHistoryQuery={runHistoryQuery}
          setRunHistoryQuery={setRunHistoryQuery}
          activeRunMatchIndex={activeRunMatchIndex}
          totalRunMatches={totalRunMatches}
          runOperatorMessage={runOperatorMessage}
          setRunOperatorMessage={setRunOperatorMessage}
          sendingRunMessage={sendingRunMessage}
          runActionPending={runActionPending}
          runMatchRefs={runMatchRefs}
          handleRunControl={handleRunControl}
          handleDeleteRun={handleDeleteRun}
          handleRestartRun={handleRestartRun}
          handleSendRunMessage={handleSendRunMessage}
          stepRunMatch={stepRunMatch}
          setSelectedRunId={setSelectedRunId}
          handleOpenWorkspaceFile={handleOpenWorkspaceFile}
          setActiveTab={setActiveTab}
          highlightQuery={highlightQuery}
          activitySearch={activitySearch}
          setActivitySearch={setActivitySearch}
        />
      );
    }

    return (
      <RunLoadingView
        runId={selectedRunId}
        runGoal={selectedRunSummary?.goal || ""}
        processName={selectedRunSummary?.process_name || ""}
        loading={Boolean(loadingRunDetail) || !runLoadError}
        error={runLoadError || ""}
        onBack={() => setSelectedRunId("")}
        onRetry={() => { void refreshRun(selectedRunId); }}
      />
    );
  }

  // =========================================================================
  // Mode 1: Full-width launcher
  // =========================================================================
  return (
    <div className="flex h-full flex-col overflow-y-auto bg-[#0f0f12]">
      <div className="flex-1 flex flex-col items-center px-6 py-8 max-w-5xl mx-auto w-full">
        <div className="w-full rounded-2xl border border-zinc-800 bg-zinc-900/50 p-5 mb-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-3 mb-2">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#6b7a5e]/15">
                  <Play className="h-5 w-5 text-[#a3b396]" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-zinc-100">Launch a Run</h2>
                  <p className="mt-1 text-sm text-zinc-500">
                    Launch ad-hoc with just a goal, or choose a process below to structure the run.
                  </p>
                </div>
              </div>
              {workspacePath && (
                <p className="flex items-center gap-1.5 text-xs text-zinc-600">
                  <FolderOpen size={12} />
                  <span className="font-mono truncate max-w-xl">{workspacePath}</span>
                </p>
              )}
            </div>

            <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-500">
              <div className="flex items-center gap-2">
                <Sparkles size={12} className="text-[#a3b396]" />
                <span>
                  {runProcess
                    ? `Using ${runProcess}`
                    : "No process selected. This will run in ad-hoc mode."}
                </span>
              </div>
            </div>
          </div>

          <form
            onSubmit={(e: FormEvent<HTMLFormElement>) => handleLaunchRun(e, launchContext)}
            className="mt-5 space-y-4"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
                Mode
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-[#6b7a5e]/15 px-2.5 py-1 text-xs font-medium text-[#bec8b4]">
                {runProcess ? <Package size={11} /> : <Sparkles size={11} />}
                {runProcess || "Ad-hoc"}
              </span>
              {runProcess && (
                <button
                  type="button"
                  onClick={() => setRunProcess("")}
                  className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  Clear process
                </button>
              )}
            </div>

            <textarea
              value={runGoal}
              onChange={(e) => setRunGoal(e.target.value)}
              placeholder={launchPlaceholder}
              rows={3}
              disabled={disabled}
              className="w-full rounded-xl border border-zinc-700 bg-zinc-900/80 px-4 py-3 text-sm text-zinc-200 placeholder:text-zinc-600 resize-none focus:outline-none focus:ring-1 focus:ring-[#8a9a7b]/50 focus:border-[#8a9a7b]/50 disabled:opacity-40"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
            />

            {selectedProcessInfo?.description ? (
              <p className="text-[11px] leading-relaxed text-zinc-500">
                {selectedProcessInfo.description}
              </p>
            ) : (
              <p className="text-[11px] leading-relaxed text-zinc-600">
                Start with a plain-English goal and Loom will launch an ad-hoc run, or pick a process below to use a predefined workflow.
              </p>
            )}

            <div className="rounded-xl border border-zinc-800 bg-zinc-950/50 p-3">
              <div className="flex items-center gap-2 mb-2">
                <FolderTree size={13} className="text-[#a3b396]" />
                <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Workspace context
                </p>
              </div>
              <p className="mb-3 text-xs text-zinc-600">
                Attach files or folders from this workspace so downstream processes can use them as explicit context.
              </p>
              <div className="relative">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-600" />
                <input
                  type="text"
                  value={attachQuery}
                  onChange={(e) => setAttachQuery(e.target.value)}
                  placeholder="Attach files or folders (optional)"
                  className="w-full rounded-lg border border-zinc-800 bg-zinc-900/70 py-2 pl-8 pr-3 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-[#8a9a7b]/50"
                />
              </div>
              {attachedPaths.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {attachedPaths.map((path) => (
                    <button
                      key={path}
                      type="button"
                      onClick={() => toggleAttachedPath(path)}
                      className="inline-flex items-center gap-1 rounded-full bg-[#6b7a5e]/15 px-2.5 py-1 text-xs text-[#bec8b4] hover:bg-[#6b7a5e]/25"
                    >
                      <span className="truncate max-w-[22rem]">{path}</span>
                      <XCircle size={11} />
                    </button>
                  ))}
                </div>
              )}
              {visibleAttachSuggestions.length > 0 && (
                <div className="mt-3">
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-600">
                    Suggested context
                  </p>
                  <div className="max-h-64 overflow-y-auto pr-1">
                  <div className="flex flex-wrap gap-2">
                  {visibleAttachSuggestions.map((option) => (
                    <button
                      key={option.path}
                      type="button"
                      onClick={() => toggleAttachedPath(option.path)}
                      className="inline-flex items-center gap-1 rounded-full border border-zinc-800 bg-zinc-900 px-2.5 py-1 text-xs text-zinc-400 hover:border-zinc-700 hover:text-zinc-200"
                    >
                      {option.isDir ? <FolderOpen size={11} /> : <FileText size={11} />}
                      <span>{workspaceAttachmentName(option.path)}</span>
                      {option.source === "artifact" && (
                        <span className="rounded-full bg-[#6b7a5e]/15 px-1.5 py-0.5 text-[9px] font-medium text-[#bec8b4]">
                          recent output
                        </span>
                      )}
                      {pathOptionParent(option.path) && (
                        <span className="max-w-[18rem] truncate text-zinc-600">
                          {pathOptionParent(option.path)}
                        </span>
                      )}
                    </button>
                  ))}
                </div>
                </div>
                </div>
              )}
            </div>

            <div>
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-1 text-[11px] text-zinc-600 hover:text-zinc-400 transition-colors"
              >
                {showAdvanced ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
                Advanced options
              </button>
              {showAdvanced && (
                <div className="mt-2 rounded-lg border border-zinc-800 bg-zinc-950/50 p-3">
                  <label className="flex items-center gap-1.5 text-[11px] font-medium text-zinc-500 mb-2">
                    <Shield className="h-3 w-3" />
                    Approval mode
                  </label>
                  <div className="flex gap-1.5">
                    {APPROVAL_MODES.map((mode) => (
                      <button
                        key={mode.value}
                        type="button"
                        onClick={() => setRunApprovalMode(mode.value)}
                        disabled={disabled}
                        title={mode.desc}
                        className={cn(
                          "flex-1 rounded-lg px-2.5 py-2 text-[11px] font-medium transition-colors border",
                          runApprovalMode === mode.value
                            ? "bg-[#6b7a5e]/15 text-[#bec8b4] border-[#8a9a7b]/30"
                            : "text-zinc-400 border-zinc-700 hover:bg-zinc-800",
                          "disabled:opacity-40",
                        )}
                      >
                        {mode.label}
                        <span className="block text-[9px] text-zinc-600 font-normal mt-0.5">{mode.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <button
                type="submit"
                disabled={disabled || (!runGoal.trim() && !runProcess)}
                className={cn(
                  "flex w-full items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold transition-colors sm:w-auto sm:min-w-[16rem]",
                  "bg-[#6b7a5e] text-white hover:bg-[#8a9a7b]",
                  "disabled:opacity-40 disabled:cursor-not-allowed",
                )}
              >
                {launchingRun ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Launching...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    {runProcess ? `Launch ${runProcess}` : "Launch ad-hoc run"}
                  </>
                )}
              </button>
              <p className="text-[10px] text-zinc-700">
                {typeof navigator !== "undefined" && navigator.platform?.includes("Mac") ? "\u2318" : "Ctrl"}+Enter to launch
              </p>
            </div>
          </form>
        </div>

        {processes.length > 0 && (
          <div className="w-full mb-6">
            <div className="mb-3 flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-600">
                  Process library ({processes.length})
                </p>
                <p className="mt-1 text-xs text-zinc-600">
                  Pick a process to structure the run, or leave it unset and launch ad-hoc.
                </p>
              </div>
              <div className="relative w-full lg:max-w-sm">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-zinc-600" />
                <input
                  type="text"
                  value={processQuery}
                  onChange={(e) => setProcessQuery(e.target.value)}
                  placeholder="Search processes..."
                  className="w-full rounded-lg border border-zinc-800 bg-zinc-900/70 py-2 pl-8 pr-3 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-[#8a9a7b]/50"
                />
              </div>
            </div>

            {(["custom", "installed", "builtin"] as ProcessBucket[]).map((bucket) => {
              const rows = groupedProcesses.get(bucket) || [];
              if (rows.length === 0) return null;
              return (
                <section key={bucket} className="mb-5">
                  <div className="mb-2 flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-zinc-200">
                        {processBucketLabel(bucket)}
                      </p>
                      <p className="text-xs text-zinc-600">
                        {processBucketDescription(bucket)}
                      </p>
                    </div>
                    <span className="rounded-full bg-zinc-900 px-2 py-0.5 text-[10px] text-zinc-500">
                      {rows.length}
                    </span>
                  </div>
                  <div className="grid grid-cols-1 gap-2 lg:grid-cols-2 2xl:grid-cols-3">
                    {rows.map(({ process: p }) => {
                      const isSelected = runProcess === p.name;
                      return (
                        <button
                          key={p.name}
                          type="button"
                          onClick={() => {
                            setRunProcess(isSelected ? "" : p.name);
                          }}
                          className={cn(
                            "rounded-xl px-4 py-3 text-left transition-all border",
                            isSelected
                              ? "bg-[#6b7a5e]/15 border-[#8a9a7b]/50 ring-1 ring-[#8a9a7b]/25"
                              : "bg-zinc-900/60 border-zinc-800 hover:border-zinc-700 hover:bg-zinc-800/50",
                          )}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="flex items-center gap-2">
                                <span
                                  className={cn(
                                    "text-sm font-semibold truncate",
                                    isSelected ? "text-[#bec8b4]" : "text-zinc-200",
                                  )}
                                >
                                  {p.name}
                                </span>
                                <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-500">
                                  {processBucketLabel(bucket)}
                                </span>
                              </div>
                              {p.description && (
                                <p className="mt-1 text-[11px] text-zinc-500 line-clamp-2 leading-relaxed">
                                  {p.description}
                                </p>
                              )}
                              <div className="mt-2 flex items-center gap-2 text-[10px] text-zinc-600">
                                {p.author && <span>by {p.author}</span>}
                                {p.version && <span className="font-mono">v{p.version}</span>}
                              </div>
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </section>
              );
            })}

            {Array.from(groupedProcesses.values()).every((rows) => rows.length === 0) && (
              <div className="rounded-xl border border-dashed border-zinc-800 px-4 py-6 text-center text-sm text-zinc-600">
                No processes match that search yet.
              </div>
            )}
          </div>
        )}

        {/* --- Recent runs --- */}
        {filteredRuns.length > 0 && (
          <div className="w-full mt-10">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-zinc-600 mb-2">
              Recent runs ({filteredRuns.length})
            </p>
            <div className="space-y-1">
              {filteredRuns.map((run) => (
                <button
                  key={run.id}
                  type="button"
                  onClick={() => setSelectedRunId(run.id)}
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/60 group"
                >
                  <RunDot status={run.status} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-zinc-200 truncate group-hover:text-zinc-100">
                      {highlightText(run.goal || "Untitled run", highlightQuery)}
                    </p>
                    <div className="flex items-center gap-2 mt-0.5">
                      <StatusBadge status={run.status} />
                      {run.process_name && (
                        <span className="text-[10px] text-zinc-600 truncate">{run.process_name}</span>
                      )}
                    </div>
                  </div>
                  <span className="text-[10px] text-zinc-700 shrink-0 tabular-nums">
                    {formatDate(run.updated_at)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* --- Empty state --- */}
        {filteredRuns.length === 0 && processes.length === 0 && (
          <div className="w-full mt-10 text-center py-8 border border-dashed border-zinc-800 rounded-xl">
            <Zap className="h-8 w-8 text-zinc-700 mx-auto mb-2" />
            <p className="text-sm text-zinc-500">No processes or runs yet</p>
            <p className="text-xs text-zinc-700 mt-1">
              Describe a goal above to launch an ad-hoc run
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function RunLoadingView({
  runId,
  runGoal,
  processName,
  loading,
  error,
  onBack,
  onRetry,
}: {
  runId: string;
  runGoal: string;
  processName: string;
  loading: boolean;
  error: string;
  onBack: () => void;
  onRetry: () => void;
}) {
  const title = runGoal || "Opening run";
  return (
    <div className="flex h-full flex-col bg-[#0f0f12]">
      <div className="border-b border-zinc-800 px-5 py-4">
        <button
          type="button"
          onClick={onBack}
          className="mb-3 inline-flex items-center gap-1 text-xs text-zinc-500 transition-colors hover:text-zinc-300"
        >
          <ArrowLeft size={12} />
          Back to launcher
        </button>
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#6b7a5e]/15">
            {loading ? <Loader2 className="h-5 w-5 animate-spin text-[#a3b396]" /> : <AlertTriangle className="h-5 w-5 text-amber-400" />}
          </div>
          <div className="min-w-0 flex-1">
            <h2 className="text-lg font-semibold leading-snug text-zinc-100 whitespace-pre-wrap break-words">{title}</h2>
            <p className="mt-1 text-xs text-zinc-500 break-words">
              {processName ? `${processName} • ` : ""}{runId}
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-1 items-center justify-center px-6">
        <div className="w-full max-w-lg rounded-2xl border border-zinc-800 bg-zinc-900/60 p-6 text-center">
          {error ? (
            <>
              <AlertTriangle className="mx-auto mb-3 h-8 w-8 text-amber-400" />
              <h3 className="mb-2 text-base font-semibold text-zinc-100">Run created, but the detail view did not finish loading</h3>
              <p className="mb-4 text-sm text-zinc-500">{error}</p>
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex items-center gap-2 rounded-lg bg-[#6b7a5e] px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#8a9a7b]"
              >
                <RefreshCw size={14} />
                Retry load
              </button>
            </>
          ) : (
            <>
              <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-[#a3b396]" />
              <h3 className="mb-2 text-base font-semibold text-zinc-100">Opening run...</h3>
              <p className="text-sm text-zinc-500">
                We created the run and are loading its timeline now.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ===========================================================================
// Mode 2: Run detail view
// ===========================================================================

interface RunDetailViewProps {
  runDetail: NonNullable<ReturnType<typeof useApp>["runDetail"]>;
  runTimeline: ReturnType<typeof useApp>["runTimeline"];
  runArtifacts: ReturnType<typeof useApp>["runArtifacts"];
  runInstructionHistory: ReturnType<typeof useApp>["runInstructionHistory"];
  runStreaming: boolean;
  runIsTerminal: boolean;
  runCanPause: boolean;
  runCanResume: boolean;
  runCanMessage: boolean;
  approvalInbox: ReturnType<typeof useApp>["approvalInbox"];
  approvalReplyDrafts: ReturnType<typeof useApp>["approvalReplyDrafts"];
  setApprovalReplyDrafts: ReturnType<typeof useApp>["setApprovalReplyDrafts"];
  replyingApprovalId: string;
  handleReplyApproval: ReturnType<typeof useApp>["handleReplyApproval"];
  visibleRunTimeline: ReturnType<typeof useApp>["visibleRunTimeline"];
  visibleRunArtifacts: ReturnType<typeof useApp>["visibleRunArtifacts"];
  runHistoryQuery: string;
  setRunHistoryQuery: (q: string) => void;
  activeRunMatchIndex: number;
  totalRunMatches: number;
  runOperatorMessage: string;
  setRunOperatorMessage: (m: string) => void;
  sendingRunMessage: boolean;
  runActionPending: string;
  runMatchRefs: ReturnType<typeof useApp>["runMatchRefs"];
  handleRunControl: (action: "pause" | "resume" | "cancel") => void;
  handleDeleteRun: () => void;
  handleRestartRun: () => void;
  handleSendRunMessage: (e: FormEvent<HTMLFormElement>) => void;
  stepRunMatch: (delta: number) => void;
  setSelectedRunId: (id: string) => void;
  handleOpenWorkspaceFile: ReturnType<typeof useApp>["handleOpenWorkspaceFile"];
  setActiveTab: ReturnType<typeof useApp>["setActiveTab"];
  highlightQuery: string;
  activitySearch: string;
  setActivitySearch: (q: string) => void;
}

function RunDetailView({
  runDetail,
  runTimeline,
  runArtifacts,
  runInstructionHistory,
  runStreaming,
  runIsTerminal,
  runCanPause,
  runCanResume,
  runCanMessage,
  approvalInbox,
  approvalReplyDrafts,
  setApprovalReplyDrafts,
  replyingApprovalId,
  handleReplyApproval,
  visibleRunTimeline,
  visibleRunArtifacts,
  runHistoryQuery,
  setRunHistoryQuery,
  activeRunMatchIndex,
  totalRunMatches,
  runOperatorMessage,
  setRunOperatorMessage,
  sendingRunMessage,
  runActionPending,
  runMatchRefs,
  handleRunControl,
  handleDeleteRun,
  handleRestartRun,
  handleSendRunMessage,
  stepRunMatch,
  setSelectedRunId,
  handleOpenWorkspaceFile,
  setActiveTab,
  highlightQuery,
  activitySearch,
  setActivitySearch,
}: RunDetailViewProps) {
  const runStatus = String(runDetail.status || "").trim().toLowerCase();
  const failureAnalysis = runDetail.failure_analysis ?? null;
  const liveAnimationsEnabled = runStatus === "executing" || runStatus === "planning" || runStatus === "running";
  const displayableRunArtifacts = useMemo(
    () => visibleRunArtifacts.filter((artifact) => isDisplayableArtifactPath(artifact.path)),
    [visibleRunArtifacts],
  );
  const runWorkspaceRelative = useMemo(() => {
    const workspaceRoot = normalizePath(runDetail.workspace?.canonical_path || "");
    const runWorkspace = normalizePath(runDetail.workspace_path || "");
    if (!workspaceRoot || !runWorkspace) return "";
    if (runWorkspace === workspaceRoot) return "";
    if (!runWorkspace.startsWith(`${workspaceRoot}/`)) return "";
    return normalizePath(runWorkspace.slice(workspaceRoot.length + 1));
  }, [runDetail.workspace?.canonical_path, runDetail.workspace_path]);
  const resolveRunFilePath = useCallback((rawPath: string) => {
    const cleanPath = normalizePath(rawPath);
    if (!cleanPath) return "";

    const exactArtifact = displayableRunArtifacts.find((artifact) => normalizePath(artifact.path) === cleanPath);
    if (exactArtifact) {
      return exactArtifact.path;
    }

    if (runWorkspaceRelative && !cleanPath.startsWith(`${runWorkspaceRelative}/`) && cleanPath !== runWorkspaceRelative) {
      const prefixed = normalizePath(`${runWorkspaceRelative}/${cleanPath}`);
      const prefixedArtifact = displayableRunArtifacts.find(
        (artifact) => normalizePath(artifact.path) === prefixed,
      );
      if (prefixedArtifact) {
        return prefixedArtifact.path;
      }
    }

    const basename = pathBasename(cleanPath);
    if (basename) {
      const basenameMatches = displayableRunArtifacts.filter(
        (artifact) => pathBasename(artifact.path) === basename,
      );
      if (basenameMatches.length === 1) {
        return basenameMatches[0].path;
      }
    }

    if (runWorkspaceRelative && !cleanPath.startsWith(`${runWorkspaceRelative}/`) && cleanPath !== runWorkspaceRelative) {
      return normalizePath(`${runWorkspaceRelative}/${cleanPath}`);
    }
    return cleanPath;
  }, [displayableRunArtifacts, runWorkspaceRelative]);
  const pendingRunApprovals = useMemo(
    () =>
      approvalInbox
        .filter(
          (item) =>
            item.task_id === runDetail.id
            && item.status === "pending"
            && (item.kind === "task_approval" || item.kind === "task_question"),
        )
        .sort((left, right) => right.created_at.localeCompare(left.created_at)),
    [approvalInbox, runDetail.id],
  );
  const instructionHistory = useMemo(
    () =>
      runInstructionHistory
        .filter((entry) => {
          const tags = String(entry.tags || "")
            .split(",")
            .map((tag) => tag.trim().toLowerCase())
            .filter(Boolean);
          return tags.includes("conversation");
        })
        .sort((left, right) => right.timestamp.localeCompare(left.timestamp)),
    [runInstructionHistory],
  );
  const deferredRunHistoryQuery = useDeferredValue(runHistoryQuery);
  const deferredActivitySearch = useDeferredValue(activitySearch);
  const [showFullActivity, setShowFullActivity] = useState(false);
  const [timelineNowMs, setTimelineNowMs] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setTimelineNowMs(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  // Build plan graph: prefer authoritative plan_subtasks from API, fall back to timeline extraction
  const planNodes = useMemo(() => {
    const apiPlan = runDetail.plan_subtasks;
    if (Array.isArray(apiPlan) && apiPlan.length > 0) {
      return apiPlan.map((s, i) => ({
        id: s.id,
        goal: s.description || s.id,
        status: s.status as "pending" | "running" | "completed" | "failed",
        depends_on: s.depends_on ?? [],
        phase_id: s.phase_id || "",
        is_critical_path: s.is_critical_path,
        is_synthesis: s.is_synthesis,
        order: i,
      }));
    }
    // Fallback: extract from timeline events
    const ids = new Map<string, { id: string; goal: string; order: number; depends_on: string[]; phase_id: string; is_critical_path: boolean; is_synthesis: boolean }>();
    for (const event of runTimeline) {
      const sid = event.data.subtask_id;
      const pid = event.data.phase_id;
      const key = typeof sid === "string" && sid.trim() ? sid.trim() : typeof pid === "string" && pid.trim() ? pid.trim() : null;
      if (key && !ids.has(key)) {
        const goal = typeof event.data.goal === "string" ? event.data.goal : typeof event.data.message === "string" ? event.data.message : key;
        ids.set(key, { id: key, goal, order: ids.size, depends_on: [], phase_id: "", is_critical_path: false, is_synthesis: false });
      }
    }
    return Array.from(ids.values()).sort((a, b) => a.order - b.order).map((n) => ({
      ...n,
      status: subtaskStatus(runTimeline, n.id),
    }));
  }, [runDetail, runTimeline]);

  // --- Activity category filters ---
  type ActivityCategory = "tool" | "subtask" | "verify" | "model" | "task" | "other";
  const [hiddenCategories, setHiddenCategories] = useState<Set<ActivityCategory>>(
    () => new Set<ActivityCategory>(),
  );

  function eventCategory(eventType: string): ActivityCategory {
    if (eventType.startsWith("tool_call")) return "tool";
    if (eventType.startsWith("subtask_") || eventType.startsWith("phase_")) return "subtask";
    if (eventType.startsWith("verification_") || eventType === "claim_verification_summary" || eventType === "artifact_seal_validation") return "verify";
    if (eventType === "model_invocation") return "model";
    if (eventType.startsWith("task_") || eventType === "run_validity_scorecard") return "task";
    return "other";
  }

  const categoryCounts = useMemo(() => {
    const counts: Record<ActivityCategory, number> = { tool: 0, subtask: 0, verify: 0, model: 0, task: 0, other: 0 };
    for (const e of visibleRunTimeline) {
      counts[eventCategory(e.event_type)]++;
    }
    return counts;
  }, [visibleRunTimeline]);

  const toggleCategory = (cat: ActivityCategory) => {
    setHiddenCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  // Filter activity by category toggles + local search
  const filteredActivity = useMemo(() => {
    let events = visibleRunTimeline.filter(
      (e) => !hiddenCategories.has(eventCategory(e.event_type)),
    );
    if (deferredActivitySearch.trim()) {
      const needle = deferredActivitySearch.toLowerCase();
      events = events.filter((e) => {
        const title = runTimelineTitle(e).toLowerCase();
        const detail = runTimelineDetail(e, timelineNowMs).toLowerCase();
        return title.includes(needle) || detail.includes(needle) || e.event_type.includes(needle);
      });
    }
    return events;
  }, [visibleRunTimeline, hiddenCategories, deferredActivitySearch, timelineNowMs]);
  const hasActivityFilter = Boolean(
    deferredRunHistoryQuery.trim() || deferredActivitySearch.trim(),
  );
  const displayedActivity = useMemo(() => {
    if (showFullActivity || hasActivityFilter) {
      return filteredActivity;
    }
    return filteredActivity.slice(-MAX_RENDERED_ACTIVITY_EVENTS);
  }, [filteredActivity, hasActivityFilter, showFullActivity]);
  const hiddenActivityCount = Math.max(0, filteredActivity.length - displayedActivity.length);
  const activityIndexOffset = Math.max(0, filteredActivity.length - displayedActivity.length);

  // --- Activity scroll: pin to bottom during streaming, unpin on user scroll up ---
  const activityScrollRef = useRef<HTMLDivElement>(null);
  const isPinnedRef = useRef(true);
  const prevActivityLenRef = useRef(displayedActivity.length);

  useEffect(() => {
    setShowFullActivity(false);
  }, [runDetail.id]);

  const handleActivityScroll = useCallback(() => {
    const el = activityScrollRef.current;
    if (!el) return;
    // Pinned if scrolled within 40px of the bottom
    isPinnedRef.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
  }, []);

  useEffect(() => {
    const el = activityScrollRef.current;
    if (!el) return;
    // Auto-scroll to bottom when new events arrive and pinned
    if (isPinnedRef.current && displayedActivity.length !== prevActivityLenRef.current) {
      el.scrollTop = el.scrollHeight;
    }
    prevActivityLenRef.current = displayedActivity.length;
  }, [displayedActivity.length]);

  // Initial pin on mount
  useEffect(() => {
    const el = activityScrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, []);

  // Group artifacts by category
  const artifactsByCategory = useMemo(() => {
    const groups = new Map<string, typeof displayableRunArtifacts>();
    for (const artifact of displayableRunArtifacts) {
      const cat = artifact.category || "uncategorized";
      if (!groups.has(cat)) groups.set(cat, []);
      groups.get(cat)!.push(artifact);
    }
    return groups;
  }, [displayableRunArtifacts]);

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[#0f0f12]">
      {/* --- Header bar --- */}
      <header className="flex items-start gap-3 border-b border-zinc-800/60 px-5 py-3 shrink-0 bg-zinc-900/50">
        <button
          type="button"
          onClick={() => setSelectedRunId("")}
          className="flex h-7 w-7 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
          title="Back to runs"
        >
          <ArrowLeft size={16} />
        </button>

        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold leading-relaxed text-zinc-100 whitespace-pre-wrap break-words">
            {runDetail.goal || "Untitled run"}
          </p>
          <div className="flex items-center gap-2 mt-0.5">
            {runDetail.process_name && (
              <span className="text-[11px] text-zinc-500 font-medium">{runDetail.process_name}</span>
            )}
            {!runDetail.process_name && (
              <span className="text-[11px] text-zinc-600 italic">ad-hoc</span>
            )}
            <StatusBadge status={runDetail.status} />
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {runStreaming && liveAnimationsEnabled && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-sky-500/15 px-2.5 py-0.5 text-[10px] font-semibold text-sky-400">
              <span className={cn("h-1.5 w-1.5 rounded-full bg-sky-400", liveAnimationsEnabled && "animate-pulse")} />
              Streaming
            </span>
          )}
          {runCanPause && (
            <button
              type="button"
              onClick={() => handleRunControl("pause")}
              disabled={runActionPending !== ""}
              className="flex items-center gap-1.5 rounded-lg bg-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-700 disabled:opacity-40 transition-colors border border-zinc-700"
            >
              <Pause size={12} /> Pause
            </button>
          )}
          {runCanResume && (
            <button
              type="button"
              onClick={() => handleRunControl("resume")}
              disabled={runActionPending !== ""}
              className="flex items-center gap-1.5 rounded-lg bg-[#6b7a5e] px-3 py-1.5 text-xs font-medium text-white hover:bg-[#8a9a7b] disabled:opacity-40 transition-colors"
            >
              <Play size={12} /> Resume
            </button>
          )}
          {!runIsTerminal && (
            <button
              type="button"
              onClick={() => handleRunControl("cancel")}
              disabled={runActionPending !== ""}
              className="flex items-center gap-1.5 rounded-lg border border-red-500/20 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/10 disabled:opacity-40 transition-colors"
            >
              <XCircle size={12} /> Cancel
            </button>
          )}
          {runIsTerminal && normalizeRunStatus(runDetail.status) !== "completed" && (
            <button
              type="button"
              onClick={() => {
                handleRestartRun();
              }}
              disabled={runActionPending !== ""}
              className="flex items-center gap-1.5 rounded-lg bg-[#6b7a5e] px-3 py-1.5 text-xs font-medium text-white hover:bg-[#8a9a7b] disabled:opacity-40 transition-colors"
            >
              <RotateCcw size={12} /> Restart
            </button>
          )}
          <button
            type="button"
            onClick={async () => {
              const msg = runIsTerminal
                ? "Delete this run? This cannot be undone."
                : "Force-delete this run? It may still be executing. This cannot be undone.";
              try {
                const { confirm } = await import("@tauri-apps/plugin-dialog");
                const ok = await confirm(msg, { title: "Loom Desktop", kind: "warning" });
                if (!ok) return;
              } catch {
                // Fallback for non-Tauri environments
                if (!window.confirm(msg)) return;
              }
              handleDeleteRun();
            }}
            disabled={runActionPending !== ""}
            className="flex items-center gap-1.5 rounded-lg border border-red-500/20 px-3 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/10 disabled:opacity-40 transition-colors"
          >
            <Trash2 size={12} /> Delete
          </button>
        </div>
      </header>

      {/* --- Scrollable body --- */}
      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-8">
        {runStatus === "failed" && failureAnalysis && (
          <section className="rounded-2xl border border-red-500/20 bg-red-500/[0.06] p-4">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 rounded-xl bg-red-500/10 p-2 text-red-300">
                <AlertTriangle size={16} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-sm font-semibold text-red-100">Why This Failed</h3>
                  {failureAnalysis.primary_reason_code && (
                    <span className="rounded-full border border-red-400/20 bg-red-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-red-200">
                      {failureAnalysis.primary_reason_code}
                    </span>
                  )}
                  {failureAnalysis.remediation.attempted && (
                    <span className="rounded-full border border-amber-400/20 bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-200">
                      remediation attempted
                    </span>
                  )}
                </div>
                {failureAnalysis.headline && (
                  <p className="mt-2 text-sm leading-6 text-zinc-100">
                    {failureAnalysis.headline}
                  </p>
                )}
                {failureAnalysis.summary && (
                  <p className="mt-2 text-xs leading-5 text-zinc-300">
                    {failureAnalysis.summary}
                  </p>
                )}
                {failureAnalysis.remediation.why_not_remedied && (
                  <p className="mt-2 text-xs leading-5 text-zinc-400">
                    {failureAnalysis.remediation.why_not_remedied}
                  </p>
                )}
                {failureAnalysis.evidence.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {failureAnalysis.evidence.map((item) => (
                      <span
                        key={item}
                        className="rounded-full border border-zinc-700/80 bg-zinc-900/70 px-2.5 py-1 text-[11px] text-zinc-300"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                )}
                {failureAnalysis.next_actions.length > 0 && (
                  <div className="mt-3">
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
                      Suggested Next Steps
                    </p>
                    <ul className="mt-2 space-y-1.5 text-xs leading-5 text-zinc-300">
                      {failureAnalysis.next_actions.map((item) => (
                        <li key={item} className="flex items-start gap-2">
                          <span className="mt-[7px] h-1 w-1 rounded-full bg-zinc-500" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </section>
        )}
        {/* ============================================================= */}
        {/* Section 1: Plan / Subtasks                                     */}
        {/* ============================================================= */}
        {planNodes.length > 0 && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Zap size={14} className="text-[#a3b396]" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Plan ({planNodes.length} subtasks)
              </h3>
            </div>
            {/* Connected graph layout */}
            <div className="relative flex flex-col gap-0">
              {planNodes.map((node, idx) => {
                const status = runDetail.plan_subtasks
                  ? node.status
                  : subtaskStatus(runTimeline, node.id);
                const hasDeps = node.depends_on.length > 0;
                const isLast = idx === planNodes.length - 1;
                const statusColor =
                  status === "completed"
                    ? "border-emerald-500/40 bg-emerald-500/5"
                    : status === "failed"
                      ? "border-red-500/40 bg-red-500/5"
                      : status === "running"
                        ? "border-sky-500/40 bg-sky-500/5 ring-1 ring-sky-500/20"
                        : "border-zinc-800 bg-zinc-900/40";
                const lineColor =
                  status === "completed"
                    ? "bg-emerald-500/40"
                    : status === "failed"
                      ? "bg-red-500/30"
                      : "bg-zinc-700/50";
                return (
                  <div key={node.id} className="relative flex items-stretch">
                    {/* Vertical connector rail */}
                    <div className="relative flex flex-col items-center w-8 shrink-0">
                      {/* Line from previous node */}
                      {idx > 0 && (
                        <div className={cn("w-px flex-1 min-h-2", lineColor)} />
                      )}
                      {/* Node dot */}
                      <div className="shrink-0 my-1">
                        <SubtaskStatusIcon
                          status={status as "pending" | "running" | "completed" | "failed"}
                          animated={liveAnimationsEnabled}
                        />
                      </div>
                      {/* Line to next node */}
                      {!isLast && (
                        <div className="w-px flex-1 min-h-2 bg-zinc-700/50" />
                      )}
                    </div>

                    {/* Node card */}
                    <div
                      className={cn(
                        "flex-1 rounded-lg border px-3 py-2 my-1 transition-colors min-w-0",
                        statusColor,
                      )}
                    >
                      <div className="flex items-center gap-2">
                        <p className="text-xs font-semibold text-zinc-200 truncate">{node.id}</p>
                        {node.is_critical_path && (
                          <span className="rounded bg-amber-500/15 px-1 py-px text-[8px] font-semibold text-amber-400 shrink-0">
                            CRITICAL
                          </span>
                        )}
                        {node.is_synthesis && (
                          <span className="rounded bg-violet-500/15 px-1 py-px text-[8px] font-semibold text-violet-400 shrink-0">
                            SYNTHESIS
                          </span>
                        )}
                        {node.phase_id && node.phase_id !== node.id && (
                          <span className="rounded bg-zinc-800 px-1 py-px text-[8px] text-zinc-500 shrink-0">
                            {node.phase_id}
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-zinc-400 line-clamp-2 mt-0.5">{node.goal}</p>
                      {hasDeps && (
                        <div className="flex items-center gap-1 mt-1.5">
                          <span className="text-[9px] text-zinc-600">depends on:</span>
                          {node.depends_on.map((dep) => (
                            <span key={dep} className="rounded bg-zinc-800 px-1 py-px text-[9px] text-zinc-500 font-mono">
                              {dep}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* ============================================================= */}
        {/* Section 2: Instructions                                        */}
        {/* ============================================================= */}
        {(runCanMessage || instructionHistory.length > 0) && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Sparkles size={14} className="text-[#a3b396]" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Instructions
              </h3>
            </div>
            <div className="rounded-xl border border-zinc-800/70 bg-zinc-900/35 p-4">
              <p className="text-sm text-zinc-400">
                Inject a live instruction into this run. Loom will pick it up on the next planning or execution step.
              </p>

              {runCanMessage ? (
                <form
                  onSubmit={(e: FormEvent<HTMLFormElement>) => handleSendRunMessage(e)}
                  className="mt-4 flex items-center gap-2"
                >
                  <input
                    type="text"
                    value={runOperatorMessage}
                    onChange={(e) => setRunOperatorMessage(e.target.value)}
                    placeholder="Add an instruction to this run..."
                    disabled={sendingRunMessage}
                    className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-[#8a9a7b] disabled:opacity-40"
                  />
                  <button
                    type="submit"
                    disabled={sendingRunMessage || !runOperatorMessage.trim()}
                    className="rounded-lg bg-[#6b7a5e] p-2 text-white hover:bg-[#8a9a7b] disabled:opacity-40 transition-colors"
                    title="Send instruction"
                  >
                    {sendingRunMessage ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                  </button>
                </form>
              ) : (
                <div className="mt-4 rounded-lg border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-xs text-zinc-500">
                  This run is not currently accepting new instructions.
                </div>
              )}

              {instructionHistory.length > 0 ? (
                <div className="mt-4 space-y-2">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600">
                    Instruction history
                  </p>
                  <div className="space-y-2">
                    {instructionHistory.map((entry) => (
                      <div
                        key={entry.id || `${entry.timestamp}-${entry.message}`}
                        className="rounded-lg border border-zinc-800/70 bg-zinc-950/40 px-3 py-2"
                      >
                        <div className="mb-1 flex items-center gap-2">
                          <span className="rounded-full bg-sky-500/15 px-2 py-0.5 text-[10px] font-semibold text-sky-400">
                            Instruction
                          </span>
                          <span className="text-[10px] text-zinc-600">
                            {formatDate(entry.timestamp)}
                          </span>
                        </div>
                        <p className="text-sm text-zinc-300">{entry.message}</p>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="mt-4 text-xs text-zinc-600">
                  No instructions have been sent to this run yet.
                </p>
              )}
            </div>
          </section>
        )}

        {/* ============================================================= */}
        {/* Section 3: Pending approvals                                    */}
        {/* ============================================================= */}
        {pendingRunApprovals.length > 0 && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <Shield size={14} className="text-[#a3b396]" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Approvals ({pendingRunApprovals.length})
              </h3>
            </div>
            <div className="space-y-3">
              {pendingRunApprovals.map((item) => {
                const isQuestion = item.kind === "task_question";
                const options = isQuestion ? approvalQuestionOptions(item) : [];
                const questionType = isQuestion ? approvalQuestionType(item) : "";
                const contextNote = isQuestion ? approvalQuestionContext(item) : "";
                const draftText = approvalReplyDrafts[item.id] || "";
                const isReplying = replyingApprovalId === item.id;
                return (
                  <div
                    key={item.id}
                    className="rounded-xl border border-zinc-800/70 bg-zinc-900/35 p-4"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <span
                          className={cn(
                            "rounded-full px-2 py-0.5 text-[10px] font-semibold",
                            isQuestion
                              ? "bg-sky-500/15 text-sky-400"
                              : "bg-violet-500/15 text-violet-400",
                          )}
                        >
                          {isQuestion ? "Question" : "Approval needed"}
                        </span>
                        {item.risk_level && (
                          <span
                            className={cn(
                              "rounded-full px-2 py-0.5 text-[10px] font-semibold capitalize",
                              item.risk_level === "high"
                                ? "bg-red-500/15 text-red-400"
                                : item.risk_level === "medium"
                                  ? "bg-amber-500/15 text-amber-400"
                                  : "bg-zinc-800 text-zinc-400",
                            )}
                          >
                            {item.risk_level}
                          </span>
                        )}
                      </div>
                      <span className="text-[10px] text-zinc-600">
                        {formatDate(item.created_at)}
                      </span>
                    </div>

                    <p className="mt-3 text-sm font-medium text-zinc-100">
                      {item.title || (isQuestion ? "Run question" : "Run approval")}
                    </p>
                    {item.summary && (
                      <p className="mt-1 text-sm text-zinc-400 whitespace-pre-wrap break-words">
                        {item.summary}
                      </p>
                    )}
                    {contextNote && (
                      <p className="mt-2 text-xs italic text-zinc-500">
                        {contextNote}
                      </p>
                    )}

                    {isQuestion ? (
                      <div className="mt-4 space-y-3">
                        {options.length > 0 && (
                          <div className="flex flex-wrap gap-2">
                            {options.map((option) => (
                              <button
                                key={option.id}
                                type="button"
                                disabled={isReplying}
                                onClick={() =>
                                  handleReplyApproval(item, {
                                    decision: "answer",
                                    response_type: "answered",
                                    selected_option_ids: [option.id],
                                    selected_labels: [option.label],
                                    custom_response: option.label,
                                  })
                                }
                                className="rounded-lg border border-zinc-700/60 bg-zinc-800/60 px-3 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:border-[#8a9a7b]/40 hover:bg-[#6b7a5e]/10 hover:text-[#bec8b4] disabled:opacity-50"
                              >
                                {option.label}
                              </button>
                            ))}
                          </div>
                        )}
                        <div className="flex gap-2">
                          <textarea
                            value={draftText}
                            onChange={(e) =>
                              setApprovalReplyDrafts((prev) => ({
                                ...prev,
                                [item.id]: e.target.value,
                              }))
                            }
                            placeholder={questionType ? `Reply to ${questionType}...` : "Type a custom reply..."}
                            rows={2}
                            className="flex-1 rounded-lg border border-zinc-700/60 bg-zinc-800/40 px-3 py-2 text-xs text-zinc-200 placeholder-zinc-600 outline-none transition-colors focus:border-[#8a9a7b]/50 focus:ring-1 focus:ring-[#8a9a7b]/20 resize-none"
                          />
                          <button
                            type="button"
                            disabled={!draftText.trim() || isReplying}
                            onClick={() =>
                              handleReplyApproval(item, {
                                decision: "answer",
                                response_type: "answered",
                                custom_response: draftText.trim(),
                              })
                            }
                            className="self-end rounded-lg bg-[#6b7a5e] px-4 py-2 text-xs font-semibold text-white transition-colors hover:bg-[#8a9a7b] disabled:opacity-50"
                          >
                            Send
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="mt-4 flex flex-wrap gap-2">
                        <button
                          type="button"
                          disabled={isReplying}
                          onClick={() => handleReplyApproval(item, { decision: "approve" })}
                          className="rounded-lg bg-emerald-600/80 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-emerald-500 disabled:opacity-50"
                        >
                          Approve
                        </button>
                        <button
                          type="button"
                          disabled={isReplying}
                          onClick={() => handleReplyApproval(item, { decision: "deny" })}
                          className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-1.5 text-xs font-semibold text-red-400 transition-colors hover:bg-red-500/20 disabled:opacity-50"
                        >
                          Deny
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* ============================================================= */}
        {/* Section 4: Files & Artifacts                                   */}
        {/* ============================================================= */}
        {displayableRunArtifacts.length > 0 && (
          <section>
            <div className="flex items-center gap-2 mb-3">
              <FileText size={14} className="text-[#a3b396]" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Files ({displayableRunArtifacts.length})
              </h3>
            </div>

            {artifactsByCategory.size > 1 ? (
              // Grouped by category
              Array.from(artifactsByCategory.entries()).map(([category, artifacts]) => (
                <div key={category} className="mb-4">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-700 mb-1.5 pl-1">
                    {category}
                  </p>
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {artifacts.map((artifact) => {
                      const globalIndex = displayableRunArtifacts.indexOf(artifact);
                      return (
                        <ArtifactCard
                          key={artifact.path}
                          artifact={artifact}
                          index={globalIndex}
                          highlightQuery={highlightQuery}
                          activeMatchIndex={activeRunMatchIndex}
                          runHistoryQuery={runHistoryQuery}
                          runMatchRefs={runMatchRefs}
                          onNavigate={() => {
                            void handleOpenWorkspaceFile(artifact.path);
                            setActiveTab("files");
                          }}
                        />
                      );
                    })}
                  </div>
                </div>
              ))
            ) : (
              // Flat grid
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {displayableRunArtifacts.map((artifact, index) => (
                  <ArtifactCard
                    key={artifact.path}
                    artifact={artifact}
                    index={index}
                    highlightQuery={highlightQuery}
                    activeMatchIndex={activeRunMatchIndex}
                    runHistoryQuery={runHistoryQuery}
                    runMatchRefs={runMatchRefs}
                    onNavigate={() => {
                      void handleOpenWorkspaceFile(artifact.path);
                      setActiveTab("files");
                    }}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {/* ============================================================= */}
        {/* Section 5: Live Activity                                       */}
        {/* ============================================================= */}
        <section>
          <div className="flex items-center gap-2 mb-3">
            <Wrench size={14} className="text-[#a3b396]" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
              Live Activity ({filteredActivity.length})
            </h3>
          </div>

          {/* Category filter chips */}
          <div className="flex flex-wrap items-center gap-1.5 mb-3">
            {(
              [
                { key: "tool" as ActivityCategory, label: "Tools", color: "bg-[#6b7a5e]/20 text-[#a3b396]" },
                { key: "subtask" as ActivityCategory, label: "Subtasks", color: "bg-sky-500/15 text-sky-400" },
                { key: "verify" as ActivityCategory, label: "Verification", color: "bg-amber-500/15 text-amber-400" },
                { key: "task" as ActivityCategory, label: "Task", color: "bg-zinc-700/50 text-zinc-400" },
                { key: "other" as ActivityCategory, label: "Other", color: "bg-zinc-800 text-zinc-500" },
              ] as const
            )
              .filter((cat) => categoryCounts[cat.key] > 0)
              .map((cat) => {
                const active = !hiddenCategories.has(cat.key);
                return (
                  <button
                    key={cat.key}
                    type="button"
                    onClick={() => toggleCategory(cat.key)}
                    className={cn(
                      "rounded-full px-2.5 py-1 text-[10px] font-semibold transition-all border",
                      active
                        ? cn(cat.color, "border-transparent")
                        : "bg-transparent text-zinc-700 border-zinc-800 line-through",
                    )}
                  >
                    {cat.label} ({categoryCounts[cat.key]})
                  </button>
                );
              })}
          </div>

          {/* Search + scroll controls */}
          <div className="flex items-center gap-2 mb-3">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-600" />
              <input
                type="text"
                value={runHistoryQuery}
                onChange={(e) => setRunHistoryQuery(e.target.value)}
                placeholder="Search timeline and artifacts..."
                className="w-full rounded-lg border border-zinc-800 bg-zinc-900/50 py-2 pl-8 pr-3 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-[#8a9a7b]"
              />
            </div>
            <div className="flex items-center gap-1 text-[10px] text-zinc-500 shrink-0">
              <span className="tabular-nums">{displayedActivity.length}</span>
              <ScrollButton direction="up" scrollRef={activityScrollRef} />
              <ScrollButton direction="down" scrollRef={activityScrollRef} />
            </div>
          </div>

          {!hasActivityFilter && filteredActivity.length > MAX_RENDERED_ACTIVITY_EVENTS && (
            <div className="mb-3 flex items-center justify-between gap-3 rounded-lg border border-zinc-800/70 bg-zinc-900/40 px-3 py-2 text-[11px] text-zinc-500">
              <p className="min-w-0 flex-1">
                {showFullActivity
                  ? `Showing all ${filteredActivity.length} events.`
                  : `Showing the latest ${displayedActivity.length} of ${filteredActivity.length} events to keep the desktop responsive.`}
              </p>
              <button
                type="button"
                onClick={() => setShowFullActivity((current) => !current)}
                className="shrink-0 rounded-md border border-zinc-700/80 px-2 py-1 font-medium text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-zinc-100"
              >
                {showFullActivity ? "Show latest" : `Show all (${hiddenActivityCount} older)`}
              </button>
            </div>
          )}

          {/* Timeline events — scrollable container pinned to bottom during streaming */}
          {displayedActivity.length === 0 ? (
            <div className="text-center py-8 border border-dashed border-zinc-800 rounded-xl">
              <Clock className="h-6 w-6 text-zinc-700 mx-auto mb-2" />
              <p className="text-xs text-zinc-600">No timeline events yet</p>
            </div>
          ) : (
            <div
              ref={activityScrollRef}
              onScroll={handleActivityScroll}
              className="space-y-1.5 max-h-[60vh] overflow-y-auto rounded-lg"
            >
              {displayedActivity.map((event, index) => {
                const title = runTimelineTitle(event);
                const detail = runTimelineDetail(event, timelineNowMs);
                const { files, duration, metrics } = extractEventHighlights(event);
                const refIndex = displayableRunArtifacts.length + activityIndexOffset + index;
                const hasStructured = files.length > 0 || duration || metrics.length > 0;
                const longContent = detail && detail !== title && isLongDetail(detail);
                const shortDetail = detail && detail !== title && !longContent;
                return (
                  <div
                    key={event.id}
                    ref={(node) => {
                      runMatchRefs.current[refIndex] = node;
                    }}
                    className={cn(
                      "flex items-start gap-3 rounded-lg border border-zinc-800/60 bg-zinc-900/40 px-4 py-2.5 transition-colors hover:bg-zinc-800/30",
                      runHistoryQuery.trim() && activeRunMatchIndex === refIndex && "ring-1 ring-[#8a9a7b] border-[#8a9a7b]/30",
                    )}
                  >
                    {/* Type badge */}
                    <span
                      className={cn(
                        "shrink-0 mt-0.5 rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide",
                        eventTypeColor(event.event_type),
                      )}
                    >
                      {eventTypeBadgeLabel(event.event_type)}
                    </span>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="text-xs font-medium text-zinc-300">
                          {highlightText(title, highlightQuery)}
                        </p>
                        {runTimelineToolName(event) && (
                          <span className="shrink-0 rounded bg-zinc-800 px-1.5 py-px text-[9px] font-mono text-zinc-400">
                            {runTimelineToolName(event)}
                          </span>
                        )}
                      </div>

                      {/* Short detail (shown inline when not long) */}
                      {shortDetail && (
                        <p className="text-[11px] text-zinc-500 mt-0.5">
                          {highlightText(detail, highlightQuery)}
                        </p>
                      )}

                      {/* Structured highlights: files, duration, metrics */}
                      {hasStructured && (
                        <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                          {files.map((f) => (
                            <button
                              key={f}
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                void handleOpenWorkspaceFile(resolveRunFilePath(f));
                                setActiveTab("files");
                              }}
                              className="inline-flex items-center gap-1 rounded bg-[#6b7a5e]/15 px-1.5 py-px text-[10px] font-mono text-[#a3b396] hover:bg-[#6b7a5e]/30 hover:text-[#bec8b4] transition-colors cursor-pointer"
                            >
                              <FileText size={9} className="shrink-0" />
                              {f}
                            </button>
                          ))}
                          {duration && (
                            <span className="rounded bg-zinc-800 px-1.5 py-px text-[9px] text-zinc-500 tabular-nums">
                              {duration}
                            </span>
                          )}
                          {metrics.map((m) => (
                            <span key={m} className="rounded bg-zinc-800 px-1.5 py-px text-[9px] text-zinc-500">
                              {m}
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Long content: collapsible with rendered markdown */}
                      {longContent && <ActivityDetailMarkdown detail={detail} />}

                      {/* Tool call args: collapsible */}
                      <ActivityToolArgs event={event} />
                    </div>

                    {/* Timestamp */}
                    <span className="text-[10px] text-zinc-700 shrink-0 tabular-nums whitespace-nowrap">
                      {formatDate(event.timestamp)}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Artifact card sub-component
// ---------------------------------------------------------------------------

interface ArtifactCardProps {
  artifact: {
    path: string;
    category: string;
    size_bytes: number;
    exists_on_disk: boolean;
  };
  index: number;
  highlightQuery: string;
  activeMatchIndex: number;
  runHistoryQuery: string;
  runMatchRefs: React.MutableRefObject<Array<HTMLElement | null>>;
  onNavigate: () => void;
}

function artifactDisplayName(path: string): string {
  const normalized = String(path || "").trim().replace(/\\/g, "/");
  if (!normalized || normalized === ".") {
    return normalized || "(untitled)";
  }
  const parts = normalized.split("/").filter(Boolean);
  return parts[parts.length - 1] || normalized;
}

function artifactDisplayDirectory(path: string): string {
  const normalized = String(path || "").trim().replace(/\\/g, "/");
  if (!normalized || normalized === ".") {
    return "";
  }
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length <= 1) {
    return "";
  }
  return parts.slice(0, -1).join("/");
}

function isDisplayableArtifactPath(path: string): boolean {
  const normalized = String(path || "").trim().replace(/\\/g, "/");
  if (!normalized) return false;
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length === 0) return false;
  return parts[parts.length - 1] !== ".";
}

function ArtifactCard({
  artifact,
  index,
  highlightQuery,
  activeMatchIndex,
  runHistoryQuery,
  runMatchRefs,
  onNavigate,
}: ArtifactCardProps) {
  const displayName = artifactDisplayName(artifact.path);
  const displayDirectory = artifactDisplayDirectory(artifact.path);
  return (
    <button
      type="button"
      onClick={onNavigate}
      ref={(node) => {
        runMatchRefs.current[index] = node;
      }}
      className={cn(
        "rounded-lg border bg-zinc-900/50 px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/50 hover:border-zinc-700",
        runHistoryQuery.trim() && activeMatchIndex === index
          ? "ring-1 ring-[#8a9a7b] border-[#8a9a7b]/30"
          : "border-zinc-800",
      )}
    >
      <p className="text-xs font-medium text-zinc-300 truncate font-mono">
        {highlightText(displayName, highlightQuery)}
      </p>
      {displayDirectory && (
        <p className="mt-1 truncate text-[10px] text-zinc-600 font-mono">
          {highlightText(displayDirectory, highlightQuery)}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
        <span
          className={cn(
            "rounded px-1.5 py-px text-[9px] font-medium",
            artifact.exists_on_disk
              ? "bg-emerald-500/15 text-emerald-400"
              : "bg-red-500/15 text-red-400",
          )}
        >
          {artifact.exists_on_disk ? "on disk" : "missing"}
        </span>
        <span className="rounded bg-zinc-800 px-1.5 py-px text-[9px] text-zinc-500">
          {artifact.category}
        </span>
        {artifact.size_bytes > 0 && (
          <span className="text-[9px] text-zinc-600">{formatBytes(artifact.size_bytes)}</span>
        )}
      </div>
    </button>
  );
}
