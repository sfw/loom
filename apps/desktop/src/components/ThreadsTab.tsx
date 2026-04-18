import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
  useDeferredValue,
  type FormEvent,
} from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  shallowEqual,
  useAppActions,
  useAppSelector,
} from "@/context/AppContext";
import {
  fetchWorkspacePathSuggestions,
  deleteConversation,
  writeScratchFile,
  type ConversationMessageAttachments,
  type ConversationApproval,
  type ConversationContentBlock,
  type ConversationDetail,
  type ConversationMessage,
  type ConversationPrompt,
  type ConversationStreamEvent,
  type WorkspaceFileEntry,
} from "@/api";
import { cn } from "@/lib/utils";
import { formatDate } from "@/utils";
import {
  buildWorkspaceAttachmentOptions,
  isHiddenWorkspaceAttachmentPath,
  rankWorkspaceAttachmentSuggestions,
  workspaceAttachmentName,
  type WorkspaceAttachmentOption,
} from "@/workspacePathAttachments";
import {
  appendConversationTimelineItems,
  buildHistoricalConversationTimelineItems,
  buildConversationMessageFallbackItems,
  buildConversationTimelineItems,
  buildConversationTimelineWindow,
  canUseDeferredConversationTranscript,
  historicalConversationTimelineCoversLatestLiveAssistant,
  historicalConversationTimelineCoversLiveTail,
  estimateConversationTimelineItemHeight,
  shouldDeferConversationTranscript,
  type ConversationTimelineItem,
} from "@/conversationTimeline";
import {
  conversationTurnSeparatorParts,
  conversationApprovalPreview,
  conversationEventDetail,
  conversationEventPills,
  conversationEventTitle,
  normalizeConversationTurnSeparatorPayload,
  summarizeMessage,
} from "../history";
import {
  Send,
  Square,
  Check,
  X,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Loader2,
  Wrench,
  CheckCircle2,
  XCircle,
  Zap,
  CornerDownLeft,
  Pencil,
  Trash2,
  Clock,
  Copy,
  MessageSquare,
  FileText,
  FolderOpen,
  ImageIcon,
} from "lucide-react";
import { useVirtualizedList } from "@/hooks/useVirtualizedList";

/** Lightweight live thinking indicator for the active turn only. */
function ThinkingIndicator({
  live = true,
  label = "Thinking...",
}: {
  live?: boolean;
  label?: string;
}) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!live) return;
    setElapsed(0);
    const interval = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => window.clearInterval(interval);
  }, [live]);
  const formatElapsed = (s: number) => {
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${m}m ${rem}s`;
  };
  return (
    <div className="mr-12 py-1">
      <div className="flex items-center gap-2 px-1 text-zinc-600">
        <Loader2 size={11} className={cn("shrink-0 text-zinc-600/80", live && "animate-spin")} />
        <span
          className={cn(
            "flex-1 text-[11px] font-medium tracking-[0.01em]",
            live ? "thinking-shimmer" : "text-zinc-500",
          )}
        >
          {label}
        </span>
        {live && (
          <span className="text-[10px] text-zinc-700/80 tabular-nums">{formatElapsed(elapsed)}</span>
        )}
      </div>
    </div>
  );
}

function LiveFeedbackPanel({
  text,
  draftText,
  markdownComponents,
}: {
  text: string;
  draftText?: string;
  markdownComponents: Parameters<typeof Markdown>[0]["components"];
}) {
  const hasThinking = Boolean(text.trim());
  const hasDraft = Boolean((draftText || "").trim());

  return (
    <div className="group relative pr-8">
      <div className="rounded-xl border border-[#8a9a7b]/10 bg-[#8a9a7b]/[0.05] px-4 py-3">
        <span className="float-right ml-3 mt-0.5 inline-flex items-center gap-1.5 rounded-full bg-[#8a9a7b]/15 px-1.5 py-px text-[9px] font-medium text-[#a3b396]">
          <span className="h-1.5 w-1.5 rounded-full bg-[#a3b396] animate-pulse" />
          Live
        </span>
        <div className="max-h-52 overflow-y-auto pr-1">
        {hasThinking && (
          <div className={cn(
            "prose prose-invert prose-sm max-w-none",
            "prose-p:my-1.5 prose-p:leading-relaxed",
            "prose-headings:mb-1.5 prose-headings:mt-3 prose-headings:font-semibold prose-headings:text-zinc-200",
            "prose-h1:text-lg prose-h2:text-base prose-h3:text-sm",
            "prose-code:rounded prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:text-xs prose-code:text-[#bec8b4] prose-code:before:content-none prose-code:after:content-none",
            "prose-pre:rounded-lg prose-pre:border prose-pre:border-zinc-800 prose-pre:bg-zinc-900 prose-pre:text-xs",
            "prose-a:text-[#a3b396] prose-a:no-underline hover:prose-a:underline",
            "prose-strong:text-zinc-200 prose-em:text-zinc-300",
            "prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5",
            "prose-blockquote:border-[#8a9a7b]/30 prose-blockquote:text-zinc-400",
            "prose-hr:border-zinc-800",
            "prose-th:text-zinc-300 prose-td:text-zinc-400",
            "text-zinc-300",
          )}>
            <Markdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{text}</Markdown>
          </div>
        )}
        {hasDraft && (
          <div className={cn(
            hasThinking && "mt-3 border-t border-[#8a9a7b]/15 pt-3",
            "text-zinc-300",
          )}>
            <div className={cn(
              "prose prose-invert prose-sm max-w-none",
              "prose-p:my-1.5 prose-p:leading-relaxed",
              "prose-headings:mb-1.5 prose-headings:mt-3 prose-headings:font-semibold prose-headings:text-zinc-200",
              "prose-h1:text-lg prose-h2:text-base prose-h3:text-sm",
              "prose-code:rounded prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:text-xs prose-code:text-[#bec8b4] prose-code:before:content-none prose-code:after:content-none",
              "prose-pre:rounded-lg prose-pre:border prose-pre:border-zinc-800 prose-pre:bg-zinc-900 prose-pre:text-xs",
              "prose-a:text-[#a3b396] prose-a:no-underline hover:prose-a:underline",
              "prose-strong:text-zinc-200 prose-em:text-zinc-300",
              "prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5",
              "prose-blockquote:border-[#8a9a7b]/30 prose-blockquote:text-zinc-400",
              "prose-hr:border-zinc-800",
              "prose-th:text-zinc-300 prose-td:text-zinc-400",
              "text-zinc-300",
            )}>
              <Markdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{draftText || ""}</Markdown>
              <span className="ml-0.5 inline-block h-4 w-1 rounded-full bg-[#a3b396]/70 align-[-0.15em] animate-pulse" />
            </div>
          </div>
        )}
        </div>
      </div>
    </div>
  );
}

function formatToolDuration(elapsedMs: number): string {
  const safeElapsedMs = Math.max(0, Math.round(elapsedMs));
  if (safeElapsedMs < 1000) {
    return `${safeElapsedMs}ms`;
  }
  if (safeElapsedMs < 60_000) {
    const seconds = safeElapsedMs / 1000;
    return seconds >= 10 ? `${Math.round(seconds)}s` : `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(safeElapsedMs / 60_000);
  const seconds = Math.floor((safeElapsedMs % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function toolCardArgText(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function compactToolCardUrl(rawUrl: string): string {
  const trimmed = rawUrl.trim();
  if (!trimmed) {
    return "";
  }
  try {
    const parsed = new URL(trimmed);
    const suffix = `${parsed.pathname}${parsed.search}${parsed.hash}`.replace(/\/$/, "");
    return `${parsed.host}${suffix}`;
  } catch {
    return trimmed.replace(/^https?:\/\//i, "").replace(/\/$/, "");
  }
}

function toolCardArgsPreview(
  toolName: string,
  argsPayload: Record<string, unknown> | null,
): string {
  if (!argsPayload) {
    return "";
  }

  const normalizedToolName = toolName.trim().toLowerCase();
  if (normalizedToolName === "web_search") {
    return toolCardArgText(argsPayload.query);
  }

  if (normalizedToolName === "web_fetch" || normalizedToolName === "web_fetch_html") {
    const url = compactToolCardUrl(toolCardArgText(argsPayload.url));
    const query = toolCardArgText(argsPayload.query);
    return [url, query].filter(Boolean).join(" · ");
  }

  return "";
}

type ComposerAction = "send" | "inject" | "redirect" | "stop";
type AttachedWorkspacePath = WorkspaceFileEntry;
type PastedImageAttachment = {
  id: string;
  dataUrl: string;
  name: string;
  sourcePath: string;
  mediaType: string;
  sizeBytes: number;
  width: number;
  height: number;
};

function detectPathMention(
  text: string,
  caret: number,
): { start: number; end: number; query: string } | null {
  const safeCaret = Math.max(0, Math.min(caret, text.length));
  const before = text.slice(0, safeCaret);
  const match = before.match(/(^|[\s([{"'])@([^\s`]*)$/);
  if (!match) {
    return null;
  }
  const query = match[2] ?? "";
  const start = safeCaret - query.length - 1;
  if (start < 0) {
    return null;
  }
  return { start, end: safeCaret, query };
}

function attachmentHasPayload(attachments: ConversationMessageAttachments): boolean {
  return Boolean(
    attachments.workspace_paths?.length
    || attachments.workspace_files?.length
    || attachments.workspace_directories?.length
    || attachments.content_blocks?.length,
  );
}

async function readFileAsDataUrl(file: File): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read pasted image."));
    reader.onload = () => resolve(String(reader.result || ""));
    reader.readAsDataURL(file);
  });
}

async function measureImageDimensions(dataUrl: string): Promise<{ width: number; height: number }> {
  return await new Promise((resolve, reject) => {
    const image = new Image();
    image.onerror = () => reject(new Error("Failed to decode pasted image."));
    image.onload = () => {
      resolve({
        width: image.naturalWidth || 0,
        height: image.naturalHeight || 0,
      });
    };
    image.src = dataUrl;
  });
}

function pastedImageFilename(file: File): string {
  const provided = file.name?.trim();
  if (provided) {
    return provided;
  }
  const extension = file.type === "image/jpeg"
    ? "jpg"
    : file.type === "image/gif"
      ? "gif"
      : file.type === "image/webp"
        ? "webp"
        : "png";
  return `pasted-image.${extension}`;
}

function conversationInputHistory(
  detail: ConversationDetail | null,
  messages: ConversationMessage[],
): string[] {
  const uiState = (detail?.session_state as { ui_state?: { input_history?: unknown } } | undefined)?.ui_state;
  const payload = uiState?.input_history;
  const sessionItems = Array.isArray(payload)
    ? payload
    : payload && typeof payload === "object" && Array.isArray((payload as { items?: unknown }).items)
      ? (payload as { items: unknown[] }).items
      : null;

  if (sessionItems) {
    return sessionItems
      .map((item) => String(item || "").trim())
      .filter((item) => item.length > 0)
      .slice(-50);
  }

  return messages
    .filter((message) => String(message.role || "").toLowerCase() === "user")
    .map((message) => String(message.content || "").trim())
    .filter((message) => message.length > 0)
    .slice(-50);
}

export default function ThreadsTab() {
  const {
    conversationAwaitingApproval,
    conversationAwaitingInput,
    conversationComposerMessage,
    conversationDetail,
    conversationHistoryQuery,
    conversationInjectMessage,
    conversationIsProcessing,
    conversationLoadError,
    conversationPhaseLabel,
    conversationStatus,
    conversationStreaming,
    hasOlderMessages,
    lastTurnStats,
    loadingConversationDetail,
    loadingOlderMessages,
    overview,
    pendingConversationApproval,
    pendingConversationPrompt,
    queuedMessages,
    quickReplyOptions,
    selectedConversationId,
    selectedWorkspaceId,
    sendingConversationInject,
    sendingConversationMessage,
    streamingText,
    streamingThinking,
    streamingToolCalls,
    runtime,
    visibleConversationEvents,
    visibleConversationMessages,
    workspaceArtifacts,
    workspaceFilesByDirectory,
  } = useAppSelector((state) => ({
    conversationAwaitingApproval: state.conversationAwaitingApproval,
    conversationAwaitingInput: state.conversationAwaitingInput,
    conversationComposerMessage: state.conversationComposerMessage,
    conversationDetail: state.conversationDetail,
    conversationHistoryQuery: state.conversationHistoryQuery,
    conversationInjectMessage: state.conversationInjectMessage,
    conversationIsProcessing: state.conversationIsProcessing,
    conversationLoadError: state.conversationLoadError,
    conversationPhaseLabel: state.conversationPhaseLabel,
    conversationStatus: state.conversationStatus,
    conversationStreaming: state.conversationStreaming,
    hasOlderMessages: state.hasOlderMessages,
    lastTurnStats: state.lastTurnStats,
    loadingConversationDetail: state.loadingConversationDetail,
    loadingOlderMessages: state.loadingOlderMessages,
    overview: state.overview,
    pendingConversationApproval: state.pendingConversationApproval,
    pendingConversationPrompt: state.pendingConversationPrompt,
    queuedMessages: state.queuedMessages,
    quickReplyOptions: state.quickReplyOptions,
    selectedConversationId: state.selectedConversationId,
    selectedWorkspaceId: state.selectedWorkspaceId,
    sendingConversationInject: state.sendingConversationInject,
    sendingConversationMessage: state.sendingConversationMessage,
    streamingText: state.streamingText,
    streamingThinking: state.streamingThinking,
    streamingToolCalls: state.streamingToolCalls,
    runtime: state.runtime,
    visibleConversationEvents: state.visibleConversationEvents,
    visibleConversationMessages: state.visibleConversationMessages,
    workspaceArtifacts: state.workspaceArtifacts,
    workspaceFilesByDirectory: state.workspaceFilesByDirectory,
  }), shallowEqual);
  const {
    cancelQueuedMessage,
    editQueuedMessage,
    handleInjectConversationInstruction,
    handleQuickConversationReply,
    handleResolveConversationApproval,
    handleStopConversationTurn,
    loadOlderMessages,
    removeConversationSummary,
    retryConversationLoad,
    setActiveTab,
    setConversationComposerMessage,
    setConversationHistoryQuery,
    setConversationInjectMessage,
    setError,
    setNotice,
    setSelectedConversationId,
    setSelectedWorkspaceFilePath,
    submitConversationMessage,
  } = useAppActions();

  const scrollRef = useRef<HTMLDivElement>(null);
  const timelineContainerRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const [composerDropdownOpen, setComposerDropdownOpen] = useState(false);
  const [composerAction, setComposerAction] = useState<ComposerAction>("send");
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [attachedWorkspacePaths, setAttachedWorkspacePaths] = useState<AttachedWorkspacePath[]>([]);
  const [pastedImages, setPastedImages] = useState<PastedImageAttachment[]>([]);
  const [activeMention, setActiveMention] = useState<{
    start: number;
    end: number;
    query: string;
  } | null>(null);
  const [workspacePathSuggestions, setWorkspacePathSuggestions] = useState<WorkspaceFileEntry[]>([]);
  const [workspacePathSuggestionIndex, setWorkspacePathSuggestionIndex] = useState(0);
  const [loadingWorkspacePathSuggestions, setLoadingWorkspacePathSuggestions] = useState(false);
  const [archivedTimelineVisibleCount, setArchivedTimelineVisibleCount] = useState(0);
  const [showJumpToLatest, setShowJumpToLatest] = useState(false);
  const workspacePathSearchTokenRef = useRef(0);

  const workspacePathOptions = useMemo<WorkspaceAttachmentOption[]>(() => {
    const fetchedEntries = workspacePathSuggestions.filter((entry) => (
      !isHiddenWorkspaceAttachmentPath(entry.path)
    ));
    return buildWorkspaceAttachmentOptions({
      workspaceEntries: fetchedEntries,
      recentArtifacts: workspaceArtifacts,
    });
  }, [workspaceArtifacts, workspacePathSuggestions]);

  const visibleWorkspacePathSuggestions = useMemo(() => rankWorkspaceAttachmentSuggestions({
    options: workspacePathOptions,
    query: activeMention?.query || "",
    selectedPaths: attachedWorkspacePaths.map((entry) => entry.path),
    limit: (activeMention?.query || "").trim() ? 24 : 18,
  }), [activeMention?.query, attachedWorkspacePaths, workspacePathOptions]);

  const conversationAttachments = useMemo<ConversationMessageAttachments>(() => {
    const workspacePaths = attachedWorkspacePaths.map((entry) => entry.path);
    const workspaceFiles = attachedWorkspacePaths
      .filter((entry) => !entry.is_dir)
      .map((entry) => entry.path);
    const workspaceDirectories = attachedWorkspacePaths
      .filter((entry) => entry.is_dir)
      .map((entry) => entry.path);
    const contentBlocks: ConversationContentBlock[] = pastedImages.map((image) => ({
      type: "image",
      source_path: image.sourcePath,
      media_type: image.mediaType,
      width: image.width,
      height: image.height,
      size_bytes: image.sizeBytes,
      text_fallback: `Attached image: ${image.name}`,
    }));
    return {
      workspace_paths: workspacePaths,
      workspace_files: workspaceFiles,
      workspace_directories: workspaceDirectories,
      content_blocks: contentBlocks,
    };
  }, [attachedWorkspacePaths, pastedImages]);

  const openWorkspaceAttachment = useCallback((path: string) => {
    setSelectedWorkspaceFilePath(path);
    setActiveTab("files");
  }, [setActiveTab, setSelectedWorkspaceFilePath]);

  // Build set of known workspace file paths for linkifying code elements
  const knownFilePaths = useMemo(() => {
    const paths = new Set<string>();
    for (const entries of Object.values(workspaceFilesByDirectory)) {
      for (const entry of entries) {
        paths.add(entry.path);
        // Also add just the filename for matching
        const name = entry.path.split("/").pop();
        if (name) paths.add(name);
      }
    }
    for (const entry of attachedWorkspacePaths) {
      paths.add(entry.path);
      const name = entry.path.split("/").pop();
      if (name) paths.add(name);
    }
    for (const event of visibleConversationEvents) {
      const payload = event.payload as Record<string, unknown>;
      const workspacePaths = Array.isArray(payload.workspace_paths) ? payload.workspace_paths : [];
      for (const item of workspacePaths) {
        const path = String(item || "").trim();
        if (!path) continue;
        paths.add(path);
        const name = path.split("/").pop();
        if (name) paths.add(name);
      }
    }
    for (const message of visibleConversationMessages) {
      const metadata = message.metadata && typeof message.metadata === "object"
        ? message.metadata as Record<string, unknown>
        : {};
      const workspacePaths = Array.isArray(metadata.workspace_paths) ? metadata.workspace_paths : [];
      for (const item of workspacePaths) {
        const path = String(item || "").trim();
        if (!path) continue;
        paths.add(path);
        const name = path.split("/").pop();
        if (name) paths.add(name);
      }
    }
    return paths;
  }, [attachedWorkspacePaths, visibleConversationEvents, visibleConversationMessages, workspaceFilesByDirectory]);

  useEffect(() => {
    if (!selectedConversationId) {
      setAttachedWorkspacePaths([]);
      setPastedImages([]);
      setActiveMention(null);
      setWorkspacePathSuggestions([]);
      setWorkspacePathSuggestionIndex(0);
    }
  }, [selectedConversationId]);

  useEffect(() => {
    if (composerAction !== "send" || !activeMention || !selectedWorkspaceId) {
      setWorkspacePathSuggestions([]);
      setWorkspacePathSuggestionIndex(0);
      setLoadingWorkspacePathSuggestions(false);
      return;
    }

    const requestToken = workspacePathSearchTokenRef.current + 1;
    workspacePathSearchTokenRef.current = requestToken;
    setLoadingWorkspacePathSuggestions(true);

    const timer = window.setTimeout(() => {
      void fetchWorkspacePathSuggestions(
        selectedWorkspaceId,
        activeMention.query,
        24,
      ).then((results) => {
        if (workspacePathSearchTokenRef.current !== requestToken) {
          return;
        }
        setWorkspacePathSuggestions(results);
        setWorkspacePathSuggestionIndex(0);
      }).catch(() => {
        if (workspacePathSearchTokenRef.current !== requestToken) {
          return;
        }
        setWorkspacePathSuggestions([]);
      }).finally(() => {
        if (workspacePathSearchTokenRef.current === requestToken) {
          setLoadingWorkspacePathSuggestions(false);
        }
      });
    }, activeMention.query ? 80 : 0);

    return () => {
      window.clearTimeout(timer);
    };
  }, [activeMention, composerAction, selectedWorkspaceId]);

  const markdownComponents = useMemo(() => ({
    code: ({ children, className }: { children?: React.ReactNode; className?: string }) => {
      // Only linkify inline code (no className means not a fenced code block)
      if (className) {
        return <code className={className}>{children}</code>;
      }
      const text = String(children || "").trim();
      if (text && knownFilePaths.has(text)) {
        return (
          <button
            type="button"
            onClick={() => openWorkspaceAttachment(text)}
            className="inline-flex items-center gap-0.5 rounded bg-[#6b7a5e]/20 px-1.5 py-px text-[#a3b396] hover:bg-[#6b7a5e]/35 hover:text-[#bec8b4] transition-colors cursor-pointer font-mono text-[inherit]"
          >
            {text}
          </button>
        );
      }
      return <code>{children}</code>;
    },
  }), [knownFilePaths, openWorkspaceAttachment]);

  // Injected messages shown as user bubbles in the chat
  const [injectedMessages, setInjectedMessages] = useState<Array<{ id: string; text: string; type: "inject" | "redirect"; timestamp: string }>>([]);
  const [transcriptHydratedConversationId, setTranscriptHydratedConversationId] = useState("");

  // Message history for up/down arrow cycling
  const messageHistoryRef = useRef<string[]>([]);
  const historyIndexRef = useRef(-1);
  const draftRef = useRef("");
  const hydratedInputHistory = useMemo(
    () => conversationInputHistory(conversationDetail, visibleConversationMessages),
    [conversationDetail, visibleConversationMessages],
  );

  // Auto-scroll with pin/unpin — pinned to bottom unless user scrolls up
  const isPinnedRef = useRef(true);
  const userScrollingRef = useRef(false);
  const scrollTimeoutRef = useRef<number | null>(null);

  const handleChatScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 80;
    isPinnedRef.current = atBottom;
    setShowJumpToLatest(!atBottom);
    // Mark user as actively scrolling — suppress auto-scroll briefly
    if (!atBottom) {
      userScrollingRef.current = true;
      if (scrollTimeoutRef.current) window.clearTimeout(scrollTimeoutRef.current);
      scrollTimeoutRef.current = window.setTimeout(() => {
        userScrollingRef.current = false;
      }, 1000);
    }
  }, []);

  const messageCount = visibleConversationMessages.length;
  const streamingLen = streamingText.length + streamingThinking.length;
  const liveTranscriptHasContent = visibleConversationEvents.length > 0 || visibleConversationMessages.length > 0;
  const deferredConversationEvents = useDeferredValue(visibleConversationEvents);
  const deferredConversationMessages = useDeferredValue(visibleConversationMessages);
  const shouldDeferTranscript = shouldDeferConversationTranscript({
    isProcessing: conversationIsProcessing,
    eventCount: visibleConversationEvents.length,
    messageCount: visibleConversationMessages.length,
    searchActive: conversationHistoryQuery.trim().length > 0,
    selectionHydrated: transcriptHydratedConversationId === selectedConversationId,
  });
  const useDeferredTranscript = canUseDeferredConversationTranscript({
    shouldDefer: shouldDeferTranscript,
    selectedConversationId,
    liveHasContent: liveTranscriptHasContent,
    deferredEvents: deferredConversationEvents,
    deferredMessages: deferredConversationMessages,
  });
  const transcriptSourceEvents = useDeferredTranscript
    ? deferredConversationEvents
    : visibleConversationEvents;
  const transcriptSourceMessages = useDeferredTranscript
    ? deferredConversationMessages
    : visibleConversationMessages;
  const timelineCacheRef = useRef<{
    events: ConversationStreamEvent[];
    items: ConversationTimelineItem[];
  }>({
    events: [],
    items: [],
  });
  const timelineItems = useMemo(
    () => {
      const cached = timelineCacheRef.current;
      if (cached.events === transcriptSourceEvents) {
        return cached.items;
      }

      let nextItems: ConversationTimelineItem[];
      if (
        cached.events.length > 0
        && transcriptSourceEvents.length >= cached.events.length
        && cached.events.every((event, index) => transcriptSourceEvents[index] === event)
      ) {
        const appendedEvents = transcriptSourceEvents.slice(cached.events.length);
        nextItems = appendedEvents.length > 0
          ? appendConversationTimelineItems(cached.items, appendedEvents)
          : cached.items;
      } else {
        nextItems = buildConversationTimelineItems(transcriptSourceEvents);
      }

      timelineCacheRef.current = {
        events: transcriptSourceEvents,
        items: nextItems,
      };
      return nextItems;
    },
    [transcriptSourceEvents],
  );
  const fallbackTimelineItems = useMemo(
    () => buildConversationMessageFallbackItems(transcriptSourceMessages),
    [transcriptSourceMessages],
  );
  const historicalTimelineItems = useMemo(
    () => buildHistoricalConversationTimelineItems(
      transcriptSourceMessages,
      transcriptSourceEvents,
    ),
    [transcriptSourceEvents, transcriptSourceMessages],
  );
  const historicalTimelineReady = useMemo(() => {
    if (historicalTimelineItems.length === 0) {
      return false;
    }
    return historicalConversationTimelineCoversLiveTail(
      timelineItems,
      historicalTimelineItems,
    );
  }, [historicalTimelineItems, timelineItems]);
  const historicalAssistantReady = useMemo(() => (
    historicalConversationTimelineCoversLatestLiveAssistant(
      timelineItems,
      historicalTimelineItems,
    )
  ), [historicalTimelineItems, timelineItems]);
  const prefersHistoricalTranscript = !conversationIsProcessing
    && !conversationStreaming
    && (historicalTimelineReady || historicalAssistantReady)
    && transcriptSourceMessages.length > 0;
  const transcriptTimelineItems = prefersHistoricalTranscript
    ? (historicalTimelineItems.length > 0 ? historicalTimelineItems : timelineItems)
    : (timelineItems.length > 0 ? timelineItems : fallbackTimelineItems);
  const archiveDisabled = conversationHistoryQuery.trim().length > 0;
  const {
    archivedCount: archivedTimelineCount,
    nextRevealCount: nextArchivedRevealCount,
    renderedItems: renderedTimelineItems,
  } = useMemo(
    () => buildConversationTimelineWindow(transcriptTimelineItems, {
      archivedVisibleCount: archivedTimelineVisibleCount,
      disableArchive: archiveDisabled,
    }),
    [archiveDisabled, archivedTimelineVisibleCount, transcriptTimelineItems],
  );
  const latestRenderedTimelineItem = renderedTimelineItems[renderedTimelineItems.length - 1];
  const showLiveFeedback = Boolean(
    conversationIsProcessing
    && streamingThinking.trim(),
  );
  const showLiveAssistantDraft = Boolean(
    conversationIsProcessing
    && streamingText.trim()
    && !(
      latestRenderedTimelineItem?.kind === "text"
      && latestRenderedTimelineItem.role === "assistant"
      && latestRenderedTimelineItem.text === streamingText
    ),
  );
  const { totalHeight: virtualTimelineHeight, virtualItems, reportSize: reportTimelineRowSize } = useVirtualizedList({
    items: renderedTimelineItems,
    containerRef: scrollRef,
    listRef: timelineContainerRef,
    estimateSize: estimateConversationTimelineItemHeight,
  });
  const activeAskUserToolCallId = pendingConversationPrompt?.tool_call_id || "";
  const activeApprovalToolTimelineId = useMemo(() => {
    if (!conversationAwaitingApproval || !pendingConversationApproval) return "";
    const pendingToolName = String(pendingConversationApproval.tool_name || "");
    for (let index = renderedTimelineItems.length - 1; index >= 0; index -= 1) {
      const item = renderedTimelineItems[index];
      if (item?.kind !== "tool") continue;
      const toolName = String(item.completedPayload?.tool_name || item.startedPayload?.tool_name || "");
      if (toolName !== pendingToolName) continue;
      if (item.completedPayload != null) continue;
      return item.id;
    }
    return "";
  }, [conversationAwaitingApproval, pendingConversationApproval, renderedTimelineItems]);
  const activeAskUserTimelineId = useMemo(() => {
    if (!conversationAwaitingInput) return "";
    for (let index = renderedTimelineItems.length - 1; index >= 0; index -= 1) {
      const item = renderedTimelineItems[index];
      if (item?.kind !== "tool") continue;
      const toolName = String(item.completedPayload?.tool_name || item.startedPayload?.tool_name || "");
      if (toolName !== "ask_user") continue;
      const toolCallId = String(item.completedPayload?.tool_call_id || item.startedPayload?.tool_call_id || "");
      if (!activeAskUserToolCallId || toolCallId === activeAskUserToolCallId) {
        return item.id;
      }
    }
    return "";
  }, [conversationAwaitingInput, activeAskUserToolCallId, renderedTimelineItems]);
  const showDetachedApprovalCard = Boolean(
    conversationAwaitingApproval
    && pendingConversationApproval
    && !activeApprovalToolTimelineId,
  );
  const showDetachedAskUserCard = Boolean(
    conversationAwaitingInput
    && pendingConversationPrompt
    && !activeAskUserTimelineId,
  );
  const composerContextHint = conversationAwaitingApproval && pendingConversationApproval
    ? `Use Approve, Deny, or Always allow to resolve ${pendingConversationApproval.tool_name}. Sending a message only queues it for later.`
    : conversationAwaitingInput && pendingConversationPrompt
      ? "Choose a reply option below to continue, or send a message if you want to answer in free text."
      : "";
  // Scroll to bottom when content changes and pinned (debounced)
  useEffect(() => {
    if (!isPinnedRef.current || userScrollingRef.current) return;
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messageCount, streamingLen, streamingToolCalls.length, renderedTimelineItems.length, virtualTimelineHeight]);

  useEffect(() => {
    setTranscriptHydratedConversationId("");
    timelineCacheRef.current = {
      events: [],
      items: [],
    };
  }, [selectedConversationId]);

  useEffect(() => {
    if (!selectedConversationId || !liveTranscriptHasContent) {
      return;
    }
    setTranscriptHydratedConversationId((current) => (
      current === selectedConversationId ? current : selectedConversationId
    ));
  }, [liveTranscriptHasContent, selectedConversationId]);

  // Force pin + clear injected messages on conversation switch
  useEffect(() => {
    isPinnedRef.current = true;
    userScrollingRef.current = false;
    setShowJumpToLatest(false);
    setArchivedTimelineVisibleCount(0);
    setInjectedMessages([]);
    messageHistoryRef.current = hydratedInputHistory;
    historyIndexRef.current = -1;
    draftRef.current = "";
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [hydratedInputHistory, selectedConversationId]);

  useEffect(() => {
    if (historyIndexRef.current !== -1) return;
    messageHistoryRef.current = hydratedInputHistory;
  }, [hydratedInputHistory]);

  useEffect(() => {
    if (!archiveDisabled) {
      return;
    }
    setArchivedTimelineVisibleCount(0);
  }, [archiveDisabled, conversationHistoryQuery]);

  // Reset composer action and clear inject bubbles when processing ends
  useEffect(() => {
    if (!conversationIsProcessing) {
      setComposerAction("send");
      setInjectedMessages([]);
    }
  }, [conversationIsProcessing]);

  const workspaceConversations = overview?.recent_conversations ?? [];
  const selectedConversationBelongsToWorkspace = !selectedConversationId
    ? true
    : loadingConversationDetail
      || workspaceConversations.some((conversation) => conversation.id === selectedConversationId)
      || (
        conversationDetail != null
        && String((conversationDetail as { workspace_id?: string }).workspace_id || "") === selectedWorkspaceId
      );

  useEffect(() => {
    if (
      selectedConversationId
      && selectedWorkspaceId
      && !selectedConversationBelongsToWorkspace
    ) {
      setSelectedConversationId("");
    }
  }, [
    conversationDetail,
    selectedConversationBelongsToWorkspace,
    selectedConversationId,
    selectedWorkspaceId,
    setSelectedConversationId,
  ]);

  useEffect(() => {
    if (!selectedConversationId && workspaceConversations.length === 1) {
      setSelectedConversationId(workspaceConversations[0]!.id);
    }
  }, [selectedConversationId, setSelectedConversationId, workspaceConversations]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setComposerDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  /* ---- Empty / chooser state ---- */
  if (!selectedConversationId || !selectedConversationBelongsToWorkspace) {
    if (workspaceConversations.length === 1) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-zinc-500 select-none">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-8 py-10 text-center max-w-sm">
            <Loader2 size={18} className="mx-auto mb-3 animate-spin text-[#a3b396]" />
            <p className="text-sm font-medium text-zinc-300">Opening thread...</p>
          </div>
        </div>
      );
    }

    if (workspaceConversations.length > 1) {
      return (
        <div className="h-full overflow-y-auto px-6 py-6">
          <div className="mx-auto max-w-3xl">
            <div className="mb-5">
              <p className="text-lg font-semibold text-zinc-100">Threads</p>
              <p className="mt-1 text-sm text-zinc-500">
                Pick a thread to continue in this workspace.
              </p>
            </div>
            <div className="space-y-2">
              {workspaceConversations.map((conversation) => (
                <button
                  key={conversation.id}
                  type="button"
                  onClick={() => setSelectedConversationId(conversation.id)}
                  className="flex w-full items-start gap-3 rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-left transition-colors hover:border-zinc-700 hover:bg-zinc-800/40"
                >
                  <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-zinc-800/80 text-zinc-500">
                    <MessageSquare size={14} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-zinc-200">
                      {conversation.title || "Untitled thread"}
                    </p>
                    <p className="mt-1 text-xs text-zinc-500">
                      {conversation.model_name} · {conversation.turn_count} turn{conversation.turn_count === 1 ? "" : "s"}
                    </p>
                  </div>
                  <span className="shrink-0 text-[11px] text-zinc-600">
                    {formatDate(conversation.last_active_at)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 select-none">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-8 py-10 text-center max-w-sm">
          <p className="text-sm font-medium text-zinc-400">No threads yet</p>
          <p className="text-xs mt-2 text-zinc-600 leading-relaxed">
            Start a new thread from the sidebar to begin working in this workspace.
          </p>
        </div>
      </div>
    );
  }

  if (loadingConversationDetail) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 select-none">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-8 py-10 text-center max-w-sm">
          <Loader2 size={18} className="mx-auto mb-3 animate-spin text-[#a3b396]" />
          <p className="text-sm font-medium text-zinc-300">Opening thread...</p>
        </div>
      </div>
    );
  }

  if (!conversationDetail && conversationLoadError) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <div className="w-full max-w-md rounded-2xl border border-red-500/20 bg-zinc-900/40 px-6 py-6 text-center">
          <p className="text-sm font-semibold text-zinc-100">Couldn&apos;t open this thread</p>
          <p className="mt-2 text-sm leading-relaxed text-zinc-400">
            {conversationLoadError}
          </p>
          <div className="mt-5 flex items-center justify-center gap-2">
            <button
              type="button"
              onClick={() => void retryConversationLoad()}
              className="rounded-lg bg-[#6b7a5e] px-3 py-2 text-xs font-medium text-white hover:bg-[#8a9a7b] transition-colors"
            >
              Retry
            </button>
            <button
              type="button"
              onClick={() => setSelectedConversationId("")}
              className="rounded-lg border border-zinc-700 px-3 py-2 text-xs font-medium text-zinc-300 hover:bg-zinc-800 transition-colors"
            >
              Back to Threads
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!conversationDetail) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-zinc-500 select-none">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-8 py-10 text-center max-w-sm">
          <Loader2 size={18} className="mx-auto mb-3 animate-spin text-[#a3b396]" />
          <p className="text-sm font-medium text-zinc-300">Opening thread...</p>
        </div>
      </div>
    );
  }

  /* ---- Status badge ---- */
  function statusBadge() {
    if (!conversationStatus) return null;
    if (conversationIsProcessing && conversationStreaming) {
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-sky-500/20 px-2 py-0.5 text-xs font-medium text-sky-400">
          <span className="h-1.5 w-1.5 rounded-full bg-sky-400 animate-pulse" />
          Streaming
        </span>
      );
    }
    if (conversationIsProcessing) {
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-sky-500/20 px-2 py-0.5 text-xs font-medium text-sky-400">
          <span className="h-1.5 w-1.5 rounded-full bg-sky-400 animate-pulse" />
          Processing
        </span>
      );
    }
    if (conversationAwaitingApproval) {
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-yellow-500/20 px-2 py-0.5 text-xs font-medium text-yellow-400">
          Awaiting approval
        </span>
      );
    }
    if (conversationAwaitingInput) {
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/20 px-2 py-0.5 text-xs font-medium text-amber-400">
          Awaiting input
        </span>
      );
    }
    return (
      <span className="inline-flex items-center rounded-full bg-zinc-700/50 px-2 py-0.5 text-xs font-medium text-zinc-400">
        Idle
      </span>
    );
  }

  function closePathMention() {
    setActiveMention(null);
    setWorkspacePathSuggestions([]);
    setWorkspacePathSuggestionIndex(0);
    setLoadingWorkspacePathSuggestions(false);
  }

  function syncActiveMention(text: string, caret: number) {
    if (composerAction !== "send") {
      closePathMention();
      return;
    }
    const mention = detectPathMention(text, caret);
    if (!mention) {
      closePathMention();
      return;
    }
    setActiveMention(mention);
  }

  function insertWorkspacePathSuggestion(entry: WorkspaceAttachmentOption) {
    const mention = activeMention;
    if (!mention) {
      return;
    }
    const prefix = conversationComposerMessage.slice(0, mention.start);
    const suffix = conversationComposerMessage.slice(mention.end);
    const replacement = `\`${entry.path}\` `;
    const nextValue = `${prefix}${replacement}${suffix}`;
    const nextCaret = prefix.length + replacement.length;
    setConversationComposerMessage(nextValue);
    setAttachedWorkspacePaths((current) => (
      current.some((item) => item.path === entry.path)
        ? current
        : [...current, {
          path: entry.path,
          name: workspaceAttachmentName(entry.path),
          is_dir: entry.isDir,
          size_bytes: 0,
          modified_at: "",
          extension: entry.isDir ? "" : "",
        }]
    ));
    closePathMention();
    requestAnimationFrame(() => {
      composerRef.current?.focus();
      composerRef.current?.setSelectionRange(nextCaret, nextCaret);
    });
  }

  function removeAttachedWorkspacePath(path: string) {
    setAttachedWorkspacePaths((current) => current.filter((entry) => entry.path !== path));
  }

  /* ---- Composer submit handler ---- */
  async function handleComposerSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    // Always pin to bottom when user sends a message and clear old inject bubbles
    isPinnedRef.current = true;
    userScrollingRef.current = false;
    setInjectedMessages([]);
    const el = scrollRef.current;
    if (el) requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
    if (composerAction === "stop") {
      await handleStopConversationTurn();
      return;
    }
    if (composerAction === "inject" || composerAction === "redirect") {
      if (attachmentHasPayload(conversationAttachments)) {
        setError("Attachments are only supported when sending a new turn.");
        return;
      }
      // Use the inject pathway for both inject and redirect
      setConversationInjectMessage(conversationComposerMessage);
      // Build a synthetic form event for the inject handler
      await handleInjectConversationInstruction(e);
      return;
    }
    const sent = await submitConversationMessage(
      conversationComposerMessage,
      attachmentHasPayload(conversationAttachments) ? conversationAttachments : undefined,
    );
    if (sent) {
      setAttachedWorkspacePaths([]);
      setPastedImages([]);
      closePathMention();
    }
  }

  function handleComposerKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    const textarea = e.currentTarget;
    const atStart = textarea.selectionStart === 0 && textarea.selectionEnd === 0;
    const atEnd = textarea.selectionStart === textarea.value.length;
    const isSingleLine = !textarea.value.includes("\n");
    const pathSuggestionCount = visibleWorkspacePathSuggestions.length;

    if (activeMention && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      if (pathSuggestionCount === 0) {
        return;
      }
      setWorkspacePathSuggestionIndex((current) => {
        if (e.key === "ArrowDown") {
          return (current + 1) % pathSuggestionCount;
        }
        return (current - 1 + pathSuggestionCount) % pathSuggestionCount;
      });
      return;
    }

    if (activeMention && (e.key === "Enter" || e.key === "Tab")) {
      if (pathSuggestionCount > 0) {
        e.preventDefault();
        insertWorkspacePathSuggestion(
          visibleWorkspacePathSuggestions[Math.min(workspacePathSuggestionIndex, pathSuggestionCount - 1)],
        );
        return;
      }
    }

    if (activeMention && e.key === "Escape") {
      e.preventDefault();
      closePathMention();
      return;
    }

    // ArrowUp at start of input = cycle back through history
    if (e.key === "ArrowUp" && atStart && isSingleLine) {
      const history = messageHistoryRef.current;
      if (history.length === 0) return;
      e.preventDefault();
      if (historyIndexRef.current === -1) {
        // Save current draft
        draftRef.current = conversationComposerMessage;
      }
      const nextIndex = Math.min(historyIndexRef.current + 1, history.length - 1);
      historyIndexRef.current = nextIndex;
      setConversationComposerMessage(history[history.length - 1 - nextIndex]);
      return;
    }

    // ArrowDown at end of input = cycle forward through history
    if (e.key === "ArrowDown" && atEnd && isSingleLine) {
      if (historyIndexRef.current === -1) return;
      e.preventDefault();
      const nextIndex = historyIndexRef.current - 1;
      historyIndexRef.current = nextIndex;
      if (nextIndex < 0) {
        // Restore draft
        setConversationComposerMessage(draftRef.current);
      } else {
        const history = messageHistoryRef.current;
        setConversationComposerMessage(history[history.length - 1 - nextIndex]);
      }
      return;
    }

    if (e.key === "Enter" && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      // Enter = send message (default action)
      e.preventDefault();
      // Add to history
      const msg = conversationComposerMessage.trim();
      if (msg) {
        const history = messageHistoryRef.current;
        // Deduplicate consecutive
        if (history[history.length - 1] !== msg) {
          history.push(msg);
          // Cap at 50
          if (history.length > 50) history.shift();
        }
        historyIndexRef.current = -1;
        draftRef.current = "";
      }
      setComposerAction("send");
      e.currentTarget.form?.requestSubmit();
    } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
      // ⌘+Enter = inject instruction
      e.preventDefault();
      if (conversationIsProcessing) {
        setComposerAction("inject");
        e.currentTarget.form?.requestSubmit();
      }
    } else if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && e.shiftKey) {
      // ⇧+⌘+Enter = redirect
      e.preventDefault();
      if (conversationIsProcessing) {
        setComposerAction("redirect");
        e.currentTarget.form?.requestSubmit();
      }
    }
  }

  function selectAction(action: ComposerAction) {
    setComposerAction(action);
    setComposerDropdownOpen(false);
    if (action !== "send") {
      closePathMention();
    }
    composerRef.current?.focus();
  }

  async function handlePaste(e: React.ClipboardEvent) {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of Array.from(items)) {
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        const file = item.getAsFile();
        if (!file) continue;
        if (!runtime?.scratch_dir) {
          setError("Runtime scratch directory is unavailable for pasted images.");
          continue;
        }
        void (async () => {
          try {
            const name = pastedImageFilename(file);
            const dataUrlPromise = readFileAsDataUrl(file);
            const [dataUrl, dimensions, buffer] = await Promise.all([
              dataUrlPromise,
              dataUrlPromise.then((url) => measureImageDimensions(url)),
              file.arrayBuffer(),
            ]);
            const sourcePath = await writeScratchFile(
              runtime.scratch_dir,
              name,
              new Uint8Array(buffer),
            );
            setPastedImages((current) => [
              ...current,
              {
                id: `img-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
                dataUrl,
                name,
                sourcePath,
                mediaType: file.type || "image/png",
                sizeBytes: file.size || buffer.byteLength,
                width: dimensions.width,
                height: dimensions.height,
              },
            ]);
          } catch (error) {
            setError(
              error instanceof Error
                ? error.message
                : "Failed to attach pasted image.",
            );
          }
        })();
      }
    }
  }

  function removePastedImage(id: string) {
    setPastedImages((current) => current.filter((img) => img.id !== id));
  }

  const composerPlaceholder =
    composerAction === "inject"
      ? "Inject a steering instruction..."
      : composerAction === "redirect"
        ? "Redirect the conversation..."
        : composerAction === "stop"
          ? "Press send to stop the current turn"
          : "Send a message...";

  const composerDisabled =
    composerAction === "stop"
      ? false
      : composerAction === "send"
        ? sendingConversationMessage || (!conversationComposerMessage.trim() && !attachmentHasPayload(conversationAttachments))
        : sendingConversationInject || !conversationComposerMessage.trim();

  const actionLabel =
    composerAction === "inject"
      ? "Inject"
      : composerAction === "redirect"
        ? "Redirect"
        : composerAction === "stop"
          ? "Stop"
          : "Send";

  function handleRevealOlderTranscript() {
    setArchivedTimelineVisibleCount((current) => current + Math.max(1, nextArchivedRevealCount));
  }

  function handleRevealFullTranscript() {
    setArchivedTimelineVisibleCount(Number.MAX_SAFE_INTEGER);
  }

  function handleJumpToLatest() {
    const el = scrollRef.current;
    if (!el) return;
    isPinnedRef.current = true;
    userScrollingRef.current = false;
    setShowJumpToLatest(false);
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }

  /* ---- Render ---- */
  return (
    <div className="flex flex-col h-full min-h-0">
      {/* ===== Header ===== */}
      <header className="flex items-center justify-between gap-3 px-5 py-3 border-b border-border shrink-0">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-zinc-100 truncate">
            {conversationDetail.title || "Untitled"}
          </h2>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[11px] text-zinc-500">
              {conversationDetail.model_name}
            </span>
            <span className="text-[11px] text-zinc-600">
              Started {formatDate(conversationDetail.started_at)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {statusBadge()}
          <button
            type="button"
            onClick={async () => {
              try {
                const { confirm } = await import("@tauri-apps/plugin-dialog");
                const ok = await confirm("Delete this thread and all its messages? This cannot be undone.", { title: "Loom Desktop", kind: "warning" });
                if (!ok) return;
              } catch {
                if (!window.confirm("Delete this thread and all its messages? This cannot be undone.")) return;
              }
              try {
                await deleteConversation(selectedConversationId);
                removeConversationSummary(selectedConversationId, selectedWorkspaceId);
                setSelectedConversationId("");
                setNotice("Thread deleted.");
              } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to delete thread.");
              }
            }}
            className="flex items-center gap-1.5 rounded-lg border border-red-500/20 px-2.5 py-1.5 text-xs font-medium text-red-400 hover:bg-red-500/10 transition-colors"
          >
            <Trash2 size={12} /> Delete
          </button>
        </div>
      </header>

      {/* ===== Message stream ===== */}
      <div ref={scrollRef} onScroll={handleChatScroll} className="flex-1 min-h-0 overflow-y-auto px-5 py-4 space-y-3">
        {/* Transcript hydration indicator */}
        {loadingOlderMessages && (
          <div className="flex items-center justify-center py-3">
            <Loader2 size={14} className="text-zinc-600 animate-spin" />
            <span className="text-[11px] text-zinc-600 ml-2">Loading transcript...</span>
          </div>
        )}
        {hasOlderMessages && !loadingOlderMessages && (
          <button
            type="button"
            onClick={() => void loadOlderMessages()}
            className="flex items-center justify-center w-full py-2 text-[11px] text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            Retry loading transcript
          </button>
        )}

        {archivedTimelineCount > 0 && (
          <div className="flex flex-col items-center gap-2 py-1">
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-center">
              <p className="text-[11px] font-medium text-zinc-300">
                {archivedTimelineCount.toLocaleString()} older transcript row{archivedTimelineCount === 1 ? "" : "s"} archived
              </p>
              <p className="mt-1 text-[11px] leading-relaxed text-zinc-500">
                The full replay is loaded. Older rows stay collapsed until you ask for them so long threads stay fast.
              </p>
              <div className="mt-3 flex items-center justify-center gap-2">
                <button
                  type="button"
                  onClick={handleRevealOlderTranscript}
                  className="rounded-lg border border-zinc-700 px-3 py-1.5 text-[11px] font-medium text-zinc-300 transition-colors hover:bg-zinc-800"
                >
                  Show {nextArchivedRevealCount.toLocaleString()} older rows
                </button>
                <button
                  type="button"
                  onClick={handleRevealFullTranscript}
                  className="rounded-lg px-3 py-1.5 text-[11px] font-medium text-[#a3b396] transition-colors hover:bg-[#6b7a5e]/10"
                >
                  Show full transcript
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Empty thread */}
        {renderedTimelineItems.length === 0 &&
          !streamingText &&
          streamingToolCalls.length === 0 && (
            <p className="py-16 text-center text-xs text-zinc-600">
              No messages yet. Send a message to get started.
            </p>
          )}

        {/* Transcript timeline */}
        {renderedTimelineItems.length > 0 && (
          <div ref={timelineContainerRef} className="relative" style={{ height: virtualTimelineHeight }}>
            {virtualItems.map((virtualItem) => (
              <TimelineVirtualRow
                key={virtualItem.item.id}
                item={virtualItem.item}
                top={virtualItem.start}
                reportSize={reportTimelineRowSize}
                markdownComponents={markdownComponents}
                onOpenWorkspaceAttachment={openWorkspaceAttachment}
                conversationAwaitingApproval={conversationAwaitingApproval}
                pendingConversationApproval={pendingConversationApproval}
                activeApprovalToolTimelineId={activeApprovalToolTimelineId}
                handleResolveConversationApproval={handleResolveConversationApproval}
                conversationAwaitingInput={conversationAwaitingInput}
                pendingConversationPrompt={pendingConversationPrompt}
                activeAskUserTimelineId={activeAskUserTimelineId}
                quickReplyOptions={quickReplyOptions}
                handleQuickConversationReply={handleQuickConversationReply}
              />
            ))}
          </div>
        )}

        {showDetachedApprovalCard && pendingConversationApproval && (
          <ApprovalActionCard
            approval={pendingConversationApproval}
            onResolve={handleResolveConversationApproval}
          />
        )}

        {showDetachedAskUserCard && pendingConversationPrompt && (
          <AskUserCard
            prompt={pendingConversationPrompt}
            options={quickReplyOptions}
            onReply={handleQuickConversationReply}
          />
        )}

        {/* ===== Injected messages (shown as user bubbles) ===== */}
        {injectedMessages.map((im) => (
          <div key={im.id} className="flex justify-end">
            <div className="max-w-[85%] rounded-xl px-4 py-2.5 bg-violet-500/10 border border-violet-500/20">
              <div className="flex items-center gap-2 mb-1">
                <span className={cn(
                  "rounded-full px-1.5 py-px text-[9px] font-medium",
                  im.type === "redirect" ? "bg-red-500/15 text-red-400" : "bg-violet-500/15 text-violet-400",
                )}>
                  {im.type}
                </span>
                <span className="text-[9px] text-zinc-600">{formatDate(im.timestamp)}</span>
              </div>
              <p className="text-sm text-zinc-200 leading-relaxed whitespace-pre-wrap break-words">{im.text}</p>
            </div>
          </div>
        ))}

        {/* ===== Live thinking placeholder ===== */}
        {showLiveFeedback && (
          <LiveFeedbackPanel
            text={streamingThinking}
            draftText={showLiveAssistantDraft ? streamingText : ""}
            markdownComponents={markdownComponents}
          />
        )}

        {conversationIsProcessing && !showLiveFeedback && !showLiveAssistantDraft && (
          <div className="space-y-2">
            <ThinkingIndicator />
          </div>
        )}

        {showLiveAssistantDraft && !showLiveFeedback && (
          <div className="group relative pr-8">
            <div className="rounded-xl border border-[#8a9a7b]/10 bg-[#8a9a7b]/[0.05] px-4 py-3">
              <span className="float-right ml-3 mt-0.5 inline-flex items-center gap-1.5 rounded-full bg-[#8a9a7b]/15 px-1.5 py-px text-[9px] font-medium text-[#a3b396]">
                <span className="h-1.5 w-1.5 rounded-full bg-[#a3b396] animate-pulse" />
                Live
              </span>
              <div className={cn(
                "prose prose-invert prose-sm max-w-none",
                "prose-p:my-1.5 prose-p:leading-relaxed",
                "prose-headings:mb-1.5 prose-headings:mt-3 prose-headings:font-semibold prose-headings:text-zinc-200",
                "prose-h1:text-lg prose-h2:text-base prose-h3:text-sm",
                "prose-code:rounded prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:text-xs prose-code:text-[#bec8b4] prose-code:before:content-none prose-code:after:content-none",
                "prose-pre:rounded-lg prose-pre:border prose-pre:border-zinc-800 prose-pre:bg-zinc-900 prose-pre:text-xs",
                "prose-a:text-[#a3b396] prose-a:no-underline hover:prose-a:underline",
                "prose-strong:text-zinc-200 prose-em:text-zinc-300",
                "prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5",
                "prose-blockquote:border-[#8a9a7b]/30 prose-blockquote:text-zinc-400",
                "prose-hr:border-zinc-800",
                "prose-th:text-zinc-300 prose-td:text-zinc-400",
                "text-zinc-300",
              )}>
                <Markdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{streamingText}</Markdown>
                <span className="ml-0.5 inline-block h-4 w-1 rounded-full bg-[#a3b396]/70 align-[-0.15em] animate-pulse" />
              </div>
            </div>
          </div>
        )}

        {showJumpToLatest && (
          <div className="sticky bottom-4 flex justify-end pr-1">
            <button
              type="button"
              onClick={handleJumpToLatest}
              className="inline-flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-950/95 px-3 py-2 text-[11px] font-medium text-zinc-200 shadow-lg backdrop-blur transition-colors hover:border-[#8a9a7b]/40 hover:text-[#bec8b4]"
            >
              <ChevronDown size={12} />
              Jump to latest
            </button>
          </div>
        )}

      </div>

      {/* ===== Queued messages ===== */}
      {queuedMessages.length > 0 && (
        <div className="border-t border-amber-600/20 bg-amber-500/[0.03] shrink-0 px-5 py-2 space-y-1.5">
          <div className="flex items-center gap-1.5">
            <Clock size={11} className="text-amber-400/70" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-amber-400/70">
              Queued ({queuedMessages.length})
            </span>
          </div>
          {queuedMessages.map((qm) => (
            <QueuedMessageCard
              key={qm.id}
              item={qm}
              onEdit={() => editQueuedMessage(qm.id)}
              onCancel={() => cancelQueuedMessage(qm.id)}
              onInject={async () => {
                try {
                  const { injectConversationInstruction } = await import("@/api");
                  await injectConversationInstruction(selectedConversationId, qm.text);
                  setInjectedMessages((prev) => [...prev, { id: qm.id, text: qm.text, type: "inject", timestamp: new Date().toISOString() }]);
                  cancelQueuedMessage(qm.id);
                } catch {}
              }}
              onRedirect={async () => {
                try {
                  const { injectConversationInstruction } = await import("@/api");
                  await injectConversationInstruction(selectedConversationId, qm.text);
                  setInjectedMessages((prev) => [...prev, { id: qm.id, text: qm.text, type: "redirect", timestamp: new Date().toISOString() }]);
                  cancelQueuedMessage(qm.id);
                } catch {}
              }}
            />
          ))}
        </div>
      )}

      {/* ===== Composer (pinned at bottom) ===== */}
      <div className="border-t border-border shrink-0 px-5 py-3">
        {/* Processing indicator above composer */}
        {(conversationIsProcessing || conversationAwaitingApproval || conversationAwaitingInput) && (
          <div className="flex items-center gap-2 mb-2">
            <span
              className={cn(
                "h-1.5 w-1.5 rounded-full",
                conversationAwaitingApproval
                  ? "bg-yellow-400 animate-pulse"
                  : conversationAwaitingInput
                    ? "bg-amber-400 animate-pulse"
                    : "bg-sky-400 animate-pulse",
              )}
            />
            <span
              className={cn(
                "text-[11px]",
                conversationAwaitingApproval
                  ? "text-yellow-400"
                  : conversationAwaitingInput
                    ? "text-amber-400"
                    : "text-zinc-500",
              )}
            >
              {conversationPhaseLabel || "Processing..."}
            </span>
          </div>
        )}
        {composerContextHint && (
          <p className="mb-2 rounded-lg border border-zinc-800 bg-zinc-950/70 px-3 py-2 text-[11px] leading-relaxed text-zinc-400">
            {composerContextHint}
          </p>
        )}

        {attachedWorkspacePaths.length > 0 && (
          <div className="mb-2 flex flex-wrap items-center gap-2">
            {attachedWorkspacePaths.map((entry) => (
              <button
                key={entry.path}
                type="button"
                onClick={() => removeAttachedWorkspacePath(entry.path)}
                className="inline-flex items-center gap-1 rounded-full border border-[#8a9a7b]/30 bg-[#8a9a7b]/10 px-2.5 py-1 text-[11px] text-[#bec8b4] transition-colors hover:border-red-400/40 hover:text-red-300"
              >
                <span>{entry.is_dir ? "Dir" : "File"}</span>
                <span className="max-w-[18rem] truncate">{entry.path}</span>
                <span aria-hidden="true">×</span>
              </button>
            ))}
          </div>
        )}

        {pastedImages.length > 0 && (
          <div className="mb-2 flex items-center gap-2 flex-wrap">
            {pastedImages.map((img) => (
              <div key={img.id} className="relative group">
                <img
                  src={img.dataUrl}
                  alt={img.name}
                  className="h-16 w-16 rounded-lg border border-zinc-700 object-cover"
                />
                <button
                  type="button"
                  onClick={() => removePastedImage(img.id)}
                  className="absolute -top-1.5 -right-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-zinc-800 border border-zinc-600 text-zinc-400 hover:text-red-400 hover:bg-zinc-700 text-[10px] opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  ×
                </button>
              </div>
            ))}
            <span className="text-[10px] text-zinc-500">
              {pastedImages.length} image{pastedImages.length !== 1 ? "s" : ""} attached
            </span>
          </div>
        )}

        <form onSubmit={handleComposerSubmit} className="flex items-center gap-2">
          <div className="relative flex-1">
            {activeMention && (
              <div className="absolute bottom-full left-0 right-0 z-50 mb-2 overflow-hidden rounded-xl border border-zinc-700 bg-zinc-950 shadow-2xl">
                <div className="border-b border-zinc-800 px-3 py-2 text-[10px] uppercase tracking-[0.16em] text-zinc-500">
                  Attach Workspace Path
                </div>
                <div className="max-h-64 overflow-y-auto py-1">
                  {visibleWorkspacePathSuggestions.map((entry, index) => (
                    <button
                      key={`${entry.path}:${entry.isDir ? "dir" : "file"}`}
                      type="button"
                      onMouseDown={(event) => {
                        event.preventDefault();
                        insertWorkspacePathSuggestion(entry);
                      }}
                      className={cn(
                        "flex w-full items-center justify-between gap-3 px-3 py-2 text-left transition-colors",
                        index === workspacePathSuggestionIndex
                          ? "bg-[#8a9a7b]/15 text-zinc-100"
                          : "text-zinc-300 hover:bg-zinc-900",
                      )}
                    >
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm">{workspaceAttachmentName(entry.path)}</div>
                        <div className="truncate text-[11px] text-zinc-500">
                          {entry.path}
                        </div>
                      </div>
                      <span className="shrink-0 text-[10px] uppercase tracking-[0.16em] text-zinc-500">
                        {entry.isDir ? "Dir" : "File"}
                      </span>
                    </button>
                  ))}
                  {!loadingWorkspacePathSuggestions && visibleWorkspacePathSuggestions.length === 0 && (
                    <div className="px-3 py-3 text-sm text-zinc-500">
                      No matching workspace paths.
                    </div>
                  )}
                  {loadingWorkspacePathSuggestions && (
                    <div className="px-3 py-3 text-sm text-zinc-500">
                      Searching workspace paths...
                    </div>
                  )}
                </div>
              </div>
            )}
            <textarea
              ref={composerRef}
              value={conversationComposerMessage}
              onChange={(e) => {
                setConversationComposerMessage(e.target.value);
                syncActiveMention(e.target.value, e.target.selectionStart);
              }}
              onClick={(e) => {
                syncActiveMention(e.currentTarget.value, e.currentTarget.selectionStart);
              }}
              onKeyUp={(e) => {
                syncActiveMention(e.currentTarget.value, e.currentTarget.selectionStart);
              }}
              onPaste={handlePaste}
              placeholder={composerPlaceholder}
              rows={2}
              onKeyDown={handleComposerKeyDown}
              className="w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 resize-none focus:outline-none focus:ring-1 focus:ring-[#8a9a7b]"
            />
          </div>

          {/* Composite action button */}
          <div className="relative flex" ref={dropdownRef}>
            {/* Main action */}
            <button
              type="submit"
              disabled={composerDisabled}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-l-lg px-3 py-2 text-xs font-medium transition-colors",
                composerAction === "stop"
                  ? "bg-red-600 text-white hover:bg-red-500"
                  : "bg-[#6b7a5e] text-white hover:bg-[#8a9a7b]",
                "disabled:opacity-40 disabled:cursor-not-allowed",
              )}
            >
              {composerAction === "stop" ? (
                <Square className="h-3.5 w-3.5" />
              ) : composerAction === "inject" || composerAction === "redirect" ? (
                <Zap className="h-3.5 w-3.5" />
              ) : (
                <Send className="h-3.5 w-3.5" />
              )}
              {actionLabel}
            </button>

            {/* Dropdown toggle */}
            <button
              type="button"
              onClick={() => setComposerDropdownOpen(!composerDropdownOpen)}
              className={cn(
                "inline-flex items-center rounded-r-lg border-l px-1.5 py-2 transition-colors",
                composerAction === "stop"
                  ? "bg-red-600 border-red-500 text-white hover:bg-red-500"
                  : "bg-[#6b7a5e] border-[#8a9a7b] text-white hover:bg-[#8a9a7b]",
              )}
            >
              <ChevronUp className="h-3 w-3" />
            </button>

            {/* Dropdown menu */}
            {composerDropdownOpen && (
              <div className="absolute bottom-full right-0 mb-1 w-44 rounded-lg border border-zinc-700 bg-zinc-900 shadow-xl py-1 z-50">
                <button
                  type="button"
                  onClick={() => selectAction("send")}
                  className={cn(
                    "flex w-full items-center gap-2 px-3 py-1.5 text-xs text-left transition-colors hover:bg-zinc-800",
                    composerAction === "send" ? "text-[#a3b396]" : "text-zinc-300",
                  )}
                >
                  <Send size={12} />
                  Send message
                  <kbd className="ml-auto text-[9px] text-zinc-600">Enter</kbd>
                </button>
                <button
                  type="button"
                  onClick={() => selectAction("inject")}
                  disabled={!conversationIsProcessing}
                  className={cn(
                    "flex w-full items-center gap-2 px-3 py-1.5 text-xs text-left transition-colors hover:bg-zinc-800",
                    "disabled:opacity-30 disabled:cursor-not-allowed",
                    composerAction === "inject" ? "text-[#a3b396]" : "text-zinc-300",
                  )}
                >
                  <Zap size={12} />
                  Inject instruction
                  <kbd className="ml-auto text-[9px] text-zinc-600">⌘↵</kbd>
                </button>
                <button
                  type="button"
                  onClick={() => selectAction("redirect")}
                  disabled={!conversationIsProcessing}
                  className={cn(
                    "flex w-full items-center gap-2 px-3 py-1.5 text-xs text-left transition-colors hover:bg-zinc-800",
                    "disabled:opacity-30 disabled:cursor-not-allowed",
                    composerAction === "redirect" ? "text-[#a3b396]" : "text-zinc-300",
                  )}
                >
                  <CornerDownLeft size={12} />
                  Redirect
                  <kbd className="ml-auto text-[9px] text-zinc-600">⇧⌘↵</kbd>
                </button>
                <div className="my-1 h-px bg-zinc-800" />
                <button
                  type="button"
                  onClick={() => selectAction("stop")}
                  disabled={!conversationIsProcessing}
                  className={cn(
                    "flex w-full items-center gap-2 px-3 py-1.5 text-xs text-left transition-colors hover:bg-zinc-800",
                    "disabled:opacity-30 disabled:cursor-not-allowed",
                    composerAction === "stop" ? "text-red-400" : "text-zinc-300",
                  )}
                >
                  <Square size={12} />
                  Stop turn
                </button>
              </div>
            )}
          </div>
        </form>

        <p className="text-[10px] text-zinc-600 mt-1.5 text-right">
          Enter to send · {navigator.platform?.includes("Mac") ? "⌘" : "Ctrl"}+Enter inject · ⇧+{navigator.platform?.includes("Mac") ? "⌘" : "Ctrl"}+Enter redirect
        </p>
      </div>
    </div>
  );
}

const TimelineVirtualRow = React.memo(function TimelineVirtualRow({
  item,
  top,
  reportSize,
  markdownComponents,
  onOpenWorkspaceAttachment,
  conversationAwaitingApproval,
  pendingConversationApproval,
  activeApprovalToolTimelineId,
  handleResolveConversationApproval,
  conversationAwaitingInput,
  pendingConversationPrompt,
  activeAskUserTimelineId,
  quickReplyOptions,
  handleQuickConversationReply,
}: {
  item: ConversationTimelineItem;
  top: number;
  reportSize: (id: string, size: number) => void;
  markdownComponents: React.ComponentProps<typeof Markdown>["components"];
  onOpenWorkspaceAttachment: (path: string) => void;
  conversationAwaitingApproval: boolean;
  pendingConversationApproval: ConversationApproval | null;
  activeApprovalToolTimelineId: string;
  handleResolveConversationApproval: (decision: "approve" | "approve_all" | "deny") => void;
  conversationAwaitingInput: boolean;
  pendingConversationPrompt: ConversationPrompt | null;
  activeAskUserTimelineId: string;
  quickReplyOptions: Array<{ id: string; label: string; description?: string }>;
  handleQuickConversationReply: (optionLabel: string) => Promise<void>;
}) {
  const rowRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = rowRef.current;
    if (!node) {
      return;
    }

    const measure = () => {
      reportSize(item.id, node.getBoundingClientRect().height);
    };

    measure();
    if (typeof ResizeObserver !== "function") {
      return;
    }

    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [item, reportSize]);

  return (
    <div className="absolute inset-x-0" style={{ top }}>
      <div ref={rowRef} className="pb-3">
        <TimelineRowContent
          item={item}
          markdownComponents={markdownComponents}
          onOpenWorkspaceAttachment={onOpenWorkspaceAttachment}
          conversationAwaitingApproval={conversationAwaitingApproval}
          pendingConversationApproval={pendingConversationApproval}
          activeApprovalToolTimelineId={activeApprovalToolTimelineId}
          handleResolveConversationApproval={handleResolveConversationApproval}
          conversationAwaitingInput={conversationAwaitingInput}
          pendingConversationPrompt={pendingConversationPrompt}
          activeAskUserTimelineId={activeAskUserTimelineId}
          quickReplyOptions={quickReplyOptions}
          handleQuickConversationReply={handleQuickConversationReply}
        />
      </div>
    </div>
  );
});

const TimelineRowContent = React.memo(function TimelineRowContent({
  item,
  markdownComponents,
  onOpenWorkspaceAttachment,
  conversationAwaitingApproval,
  pendingConversationApproval,
  activeApprovalToolTimelineId,
  handleResolveConversationApproval,
  conversationAwaitingInput,
  pendingConversationPrompt,
  activeAskUserTimelineId,
  quickReplyOptions,
  handleQuickConversationReply,
}: {
  item: ConversationTimelineItem;
  markdownComponents: React.ComponentProps<typeof Markdown>["components"];
  onOpenWorkspaceAttachment: (path: string) => void;
  conversationAwaitingApproval: boolean;
  pendingConversationApproval: ConversationApproval | null;
  activeApprovalToolTimelineId: string;
  handleResolveConversationApproval: (decision: "approve" | "approve_all" | "deny") => void;
  conversationAwaitingInput: boolean;
  pendingConversationPrompt: ConversationPrompt | null;
  activeAskUserTimelineId: string;
  quickReplyOptions: Array<{ id: string; label: string; description?: string }>;
  handleQuickConversationReply: (optionLabel: string) => Promise<void>;
}) {
  if (item.kind === "text") {
    const isUser = item.role === "user";
    const isSending = item.deliveryState === "sending";
    const isTimedOut = item.deliveryState === "failed";

    return (
      <div
        className={cn(
          "group relative",
          isUser && "pl-16",
          !isUser && "pr-8",
        )}
      >
        <div
          className={cn(
            "rounded-xl px-4 py-3",
            isUser && "border border-[#8a9a7b]/10 bg-[#8a9a7b]/8",
            isUser && isSending && "border-[#a3b396]/30 bg-[#8a9a7b]/12",
            isUser && isTimedOut && "border-red-400/30 bg-red-500/8",
            !isUser && "bg-transparent",
          )}
        >
          <span className={cn(
            "float-right ml-3 mt-0.5 flex items-center gap-1.5 transition-opacity",
            (isSending || isTimedOut) ? "opacity-100" : "opacity-0 group-hover:opacity-100",
          )}>
            {isSending && (
              <span className="inline-flex items-center gap-1 rounded-full bg-[#8a9a7b]/15 px-1.5 py-px text-[9px] font-medium text-[#a3b396]">
                <span className="h-1.5 w-1.5 rounded-full bg-[#a3b396] animate-pulse" />
                Sending...
              </span>
            )}
            {isTimedOut && (
              <span className="inline-flex items-center gap-1 rounded-full bg-red-500/15 px-1.5 py-px text-[9px] font-medium text-red-300">
                <span className="h-1.5 w-1.5 rounded-full bg-red-300" />
                Timed out
              </span>
            )}
            <CopyButton text={item.text} />
            <span className="text-[10px] text-zinc-700">
              {formatDate(item.createdAt)}
            </span>
          </span>
          <div className={cn(
            "prose prose-invert prose-sm max-w-none",
            "prose-p:my-1.5 prose-p:leading-relaxed",
            "prose-headings:mb-1.5 prose-headings:mt-3 prose-headings:font-semibold prose-headings:text-zinc-200",
            "prose-h1:text-lg prose-h2:text-base prose-h3:text-sm",
            "prose-code:rounded prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:text-xs prose-code:text-[#bec8b4] prose-code:before:content-none prose-code:after:content-none",
            "prose-pre:rounded-lg prose-pre:border prose-pre:border-zinc-800 prose-pre:bg-zinc-900 prose-pre:text-xs",
            "prose-a:text-[#a3b396] prose-a:no-underline hover:prose-a:underline",
            "prose-strong:text-zinc-200 prose-em:text-zinc-300",
            "prose-ul:my-1.5 prose-ol:my-1.5 prose-li:my-0.5",
            "prose-blockquote:border-[#8a9a7b]/30 prose-blockquote:text-zinc-400",
            "prose-hr:border-zinc-800",
            "prose-th:text-zinc-300 prose-td:text-zinc-400",
            isUser && "text-zinc-200",
            !isUser && "text-zinc-300",
          )}>
            <Markdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{item.text}</Markdown>
          </div>
        </div>
      </div>
    );
  }

  if (item.kind === "tool") {
    const toolName = String(item.completedPayload?.tool_name || item.startedPayload?.tool_name || "");
    const showInlineApproval =
      conversationAwaitingApproval
      && pendingConversationApproval
      && item.id === activeApprovalToolTimelineId;
    const showInlinePrompt =
      conversationAwaitingInput
      && pendingConversationPrompt
      && toolName === "ask_user"
      && item.id === activeAskUserTimelineId;

    return (
      <div className="space-y-3">
        <ToolEventCard
          startedPayload={item.startedPayload}
          completedPayload={item.completedPayload}
          createdAt={item.createdAt}
        />
        {showInlineApproval && (
          <ApprovalActionCard
            approval={pendingConversationApproval}
            onResolve={handleResolveConversationApproval}
          />
        )}
        {showInlinePrompt && pendingConversationPrompt && (
          <AskUserCard
            prompt={pendingConversationPrompt}
            options={quickReplyOptions}
            onReply={handleQuickConversationReply}
          />
        )}
      </div>
    );
  }

  if (item.kind === "attachment") {
    const isSending = item.deliveryState === "sending";
    const isTimedOut = item.deliveryState === "failed";
    const directorySet = new Set(item.workspaceDirectories);
    const imageCount = item.contentBlocks.filter((block) => String(block.type || "") === "image").length;
    const documentCount = item.contentBlocks.filter((block) => String(block.type || "") === "document").length;

    return (
      <div className="pl-16">
        <div
          className={cn(
            "rounded-xl border px-4 py-3",
            "border-[#8a9a7b]/15 bg-[#8a9a7b]/5",
            isSending && "border-[#a3b396]/30 bg-[#8a9a7b]/10",
            isTimedOut && "border-red-400/30 bg-red-500/8",
          )}
        >
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-[11px] font-medium text-zinc-400">
              <span>Explicit context</span>
              {imageCount > 0 && (
                <span className="inline-flex items-center gap-1 rounded-full bg-zinc-950 px-2 py-0.5 text-[10px] text-zinc-400">
                  <ImageIcon size={11} />
                  {imageCount} image{imageCount !== 1 ? "s" : ""}
                </span>
              )}
              {documentCount > 0 && (
                <span className="inline-flex items-center gap-1 rounded-full bg-zinc-950 px-2 py-0.5 text-[10px] text-zinc-400">
                  <FileText size={11} />
                  {documentCount} document{documentCount !== 1 ? "s" : ""}
                </span>
              )}
            </div>
            <span className="text-[10px] text-zinc-700">
              {formatDate(item.createdAt)}
            </span>
          </div>
          {item.workspacePaths.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {item.workspacePaths.map((path) => {
                const isDirectory = directorySet.has(path);
                return (
                  <button
                    key={path}
                    type="button"
                    onClick={() => onOpenWorkspaceAttachment(path)}
                    aria-label={`Open ${isDirectory ? "folder" : "file"} ${path}`}
                    className="inline-flex max-w-full items-center gap-1 rounded-full border border-[#8a9a7b]/25 bg-zinc-950/80 px-2.5 py-1 text-[11px] text-[#bec8b4] transition-colors hover:border-[#a3b396]/50 hover:text-zinc-100"
                  >
                    {isDirectory ? <FolderOpen size={11} /> : <FileText size={11} />}
                    <span>{isDirectory ? "Dir" : "File"}</span>
                    <span className="max-w-[22rem] truncate">{path}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <ConversationEventCard event={item.event} />
    </div>
  );
});

function ConversationEventCard({ event }: { event: ConversationStreamEvent }) {
  if (event.event_type === "turn_separator") {
    const stats = normalizeConversationTurnSeparatorPayload(event.payload);
    const parts = conversationTurnSeparatorParts(stats);
    return (
      <div className="flex items-center justify-center gap-3 py-2">
        <div className="h-px flex-1 bg-zinc-800" />
        <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 text-[10px] text-zinc-600">
          <Zap size={10} className="shrink-0" />
          {parts.map((part, index) => (
            <React.Fragment key={`${part}-${index}`}>
              {index > 0 && <span aria-hidden="true">·</span>}
              <span className="tabular-nums">{part}</span>
            </React.Fragment>
          ))}
        </div>
        <div className="h-px flex-1 bg-zinc-800" />
      </div>
    );
  }

  const title = conversationEventTitle(event);
  const detail = conversationEventDetail(event);
  const pills = conversationEventPills(event);

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 px-4 py-3">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-zinc-300">{title}</span>
        <span className="text-[10px] text-zinc-600 ml-auto">
          {formatDate(event.created_at)}
        </span>
      </div>
      {detail && (
        <p className="mt-1 text-xs text-zinc-400 whitespace-pre-wrap break-words">
          {detail}
        </p>
      )}
      {pills.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {pills.map((pill) => (
            <span
              key={`${event.seq}-${pill}`}
              className="rounded-full border border-zinc-800 bg-zinc-950 px-2 py-0.5 text-[10px] text-zinc-500"
            >
              {pill}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function ToolEventCard({
  startedPayload,
  completedPayload,
  createdAt,
}: {
  startedPayload?: Record<string, unknown>;
  completedPayload?: Record<string, unknown>;
  createdAt: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const toolName = String(
    completedPayload?.tool_name || startedPayload?.tool_name || "tool",
  );
  const toolCallId = String(
    completedPayload?.tool_call_id || startedPayload?.tool_call_id || "",
  );
  const completed = Boolean(completedPayload);
  const startedAtMs = Date.parse(createdAt);
  const [liveElapsedMs, setLiveElapsedMs] = useState(() =>
    Number.isFinite(startedAtMs) ? Math.max(0, Date.now() - startedAtMs) : 0,
  );
  useEffect(() => {
    if (completed || !Number.isFinite(startedAtMs)) {
      return;
    }
    const tick = () => {
      setLiveElapsedMs(Math.max(0, Date.now() - startedAtMs));
    };
    tick();
    const timer = window.setInterval(tick, 250);
    return () => {
      window.clearInterval(timer);
    };
  }, [completed, startedAtMs]);
  const success = completedPayload?.success !== false;
  const argsPayload = (() => {
    const rawArgs = startedPayload?.args || completedPayload?.args;
    return rawArgs && typeof rawArgs === "object"
      ? rawArgs as Record<string, unknown>
      : null;
  })();
  const elapsedMs = Number(completedPayload?.elapsed_ms || 0);
  const elapsedLabel = completed && elapsedMs > 0
    ? formatToolDuration(elapsedMs)
    : !completed && liveElapsedMs > 0
      ? formatToolDuration(liveElapsedMs)
      : formatDate(createdAt);
  const argCount = argsPayload ? Object.keys(argsPayload).length : 0;
  const argsPreview = toolCardArgsPreview(toolName, argsPayload);
  const statusLabel = !completed ? "Running" : success ? "Done" : "Failed";
  const statusClass = !completed
    ? "bg-[#8a9a7b]/15 text-[#bec8b4]"
    : success
      ? "bg-emerald-500/15 text-emerald-300"
      : "bg-red-500/15 text-red-300";

  return (
    <div className="mx-8">
      <button
        type="button"
        onClick={() => setExpanded((current) => !current)}
        className={cn(
          "flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs transition-colors",
          completed
            ? success
              ? "border-emerald-500/20 bg-emerald-500/5 hover:bg-emerald-500/10"
              : "border-red-500/20 bg-red-500/5 hover:bg-red-500/10"
            : "border-[#8a9a7b]/20 bg-[#8a9a7b]/5 hover:bg-[#8a9a7b]/10",
          expanded && "rounded-b-none",
        )}
      >
        {expanded ? (
          <ChevronDown size={12} className="shrink-0 text-zinc-600" />
        ) : (
          <ChevronRight size={12} className="shrink-0 text-zinc-600" />
        )}
        {!completed ? (
          <Loader2 size={12} className="text-[#a3b396] animate-spin shrink-0" />
        ) : success ? (
          <CheckCircle2 size={12} className="text-emerald-400 shrink-0" />
        ) : (
          <XCircle size={12} className="text-red-400 shrink-0" />
        )}
        <span
          className={cn(
            "font-semibold font-mono",
            !completed ? "text-[#bec8b4]" : success ? "text-emerald-300" : "text-red-300",
          )}
        >
          {toolName}
        </span>
        <span className={cn("rounded-full px-1.5 py-px text-[9px] font-medium", statusClass)}>
          {statusLabel}
        </span>
        {argCount > 0 && (
          <span className="text-[10px] text-zinc-600">
            {argCount} arg{argCount === 1 ? "" : "s"}
          </span>
        )}
        {argsPreview && (
          <span
            className="min-w-0 flex-1 truncate text-[10px] text-zinc-500"
            title={argsPreview}
          >
            {argsPreview}
          </span>
        )}
        {toolCallId && (
          <span className="hidden shrink-0 text-[10px] text-zinc-600 md:inline">
            {toolCallId}
          </span>
        )}
        <span className="ml-auto shrink-0 text-[10px] text-zinc-600">
          {elapsedLabel}
        </span>
      </button>
      {expanded && (
        <div
          className={cn(
            "rounded-b-lg border border-t-0 px-4 py-3",
            completed
              ? success
                ? "border-emerald-500/20 bg-zinc-950/60"
                : "border-red-500/20 bg-zinc-950/60"
              : "border-[#8a9a7b]/20 bg-zinc-950/60",
          )}
        >
          {toolCallId && (
            <div className="mb-3">
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-600">
                Call ID
              </p>
              <code className="text-[11px] text-zinc-400">{toolCallId}</code>
            </div>
          )}
          <div>
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-600">
              Tool Call Spec
            </p>
            <pre className="text-[11px] text-zinc-400 whitespace-pre-wrap break-words font-mono leading-relaxed">
              {argsPayload ? JSON.stringify(argsPayload, null, 2) : "{}"}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

function ApprovalActionCard({
  approval,
  onResolve,
}: {
  approval: ConversationApproval;
  onResolve: (decision: "approve" | "approve_all" | "deny") => void;
}) {
  return (
    <div className="rounded-xl border-2 border-yellow-600/40 bg-yellow-500/5 px-4 py-4 space-y-3">
      <div className="flex items-center gap-2">
        <span className="inline-block h-2 w-2 rounded-full bg-yellow-400 animate-pulse" />
        <span className="text-xs font-semibold text-yellow-300">
          Approval Required
        </span>
      </div>
      <p className="text-xs text-zinc-300">
        Tool:{" "}
        <span className="font-medium text-zinc-100 font-mono">
          {approval.tool_name}
        </span>
      </p>
      {conversationApprovalPreview(approval) && (
        <pre className="rounded-md bg-zinc-900/80 p-2.5 text-[11px] text-zinc-400 whitespace-pre-wrap break-words max-h-40 overflow-y-auto font-mono">
          {conversationApprovalPreview(approval)}
        </pre>
      )}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onResolve("approve")}
          className="inline-flex items-center gap-1 rounded-md bg-green-600/80 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-500 transition-colors"
        >
          <Check className="h-3.5 w-3.5" />
          Approve
        </button>
        <button
          type="button"
          onClick={() => onResolve("deny")}
          className="inline-flex items-center gap-1 rounded-md bg-red-600/80 px-3 py-1.5 text-xs font-medium text-white hover:bg-red-500 transition-colors"
        >
          <X className="h-3.5 w-3.5" />
          Deny
        </button>
        <button
          type="button"
          onClick={() => onResolve("approve_all")}
          className="inline-flex items-center gap-1 rounded-md bg-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-200 hover:bg-zinc-600 transition-colors"
        >
          Always allow
        </button>
      </div>
    </div>
  );
}

/* ---------- Collapsible tool call card ---------- */

function ToolCallCard({ msg, summary }: { msg: { id: number; tool_name: string | null; content: string | null; tool_calls: Array<Record<string, unknown>>; created_at: string }; summary: string }) {
  const [expanded, setExpanded] = useState(false);
  const toolName = msg.tool_name || "tool";

  // Try to parse tool content as JSON for structured display
  let parsedContent: Record<string, unknown> | null = null;
  try {
    if (typeof msg.content === "string" && msg.content.trim().startsWith("{")) {
      parsedContent = JSON.parse(msg.content);
    }
  } catch { /* ignore parse errors */ }

  // Extract key fields for the summary line
  const success = parsedContent ? (parsedContent as { success?: boolean }).success : undefined;
  const output = parsedContent
    ? String((parsedContent as { output?: string }).output || "").slice(0, 200)
    : typeof msg.content === "string" ? msg.content.slice(0, 200) : "";

  return (
    <div className="mx-4">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left transition-colors",
          "border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800/60",
          expanded && "rounded-b-none border-b-0",
        )}
      >
        {expanded ? (
          <ChevronDown size={12} className="shrink-0 text-zinc-600" />
        ) : (
          <ChevronRight size={12} className="shrink-0 text-zinc-600" />
        )}
        {success === false ? (
          <XCircle size={13} className="shrink-0 text-red-400" />
        ) : success === true ? (
          <CheckCircle2 size={13} className="shrink-0 text-emerald-400" />
        ) : (
          <Wrench size={13} className="shrink-0 text-zinc-500" />
        )}
        <span className="font-mono text-xs font-semibold text-zinc-300">{toolName}</span>
        {success === false && (
          <span className="rounded bg-red-500/15 px-1.5 py-px text-[10px] font-medium text-red-400">failed</span>
        )}
        <span className="ml-auto text-[10px] text-zinc-700">{formatDate(msg.created_at)}</span>
      </button>

      {expanded && (
        <div className="rounded-b-lg border border-t-0 border-zinc-800 bg-zinc-950/60 max-h-64 overflow-y-auto">
          {/* Output preview */}
          {output && (
            <div className="border-b border-zinc-800/50 px-4 py-3">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600 mb-1">Output</p>
              <pre className="text-xs text-zinc-400 whitespace-pre-wrap break-words font-mono leading-relaxed">{output}</pre>
            </div>
          )}

          {/* Structured data */}
          {parsedContent && (
            <div className="px-4 py-3">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600 mb-1">Data</p>
              <pre className="text-[11px] text-zinc-500 whitespace-pre-wrap break-words font-mono leading-relaxed">
                {JSON.stringify(parsedContent, null, 2)}
              </pre>
            </div>
          )}

          {/* Raw content fallback */}
          {!parsedContent && msg.content && (
            <div className="px-4 py-3">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600 mb-1">Content</p>
              <pre className="text-[11px] text-zinc-500 whitespace-pre-wrap break-words font-mono leading-relaxed">
                {msg.content}
              </pre>
            </div>
          )}

          {/* Tool calls (if this message contains outgoing tool calls) */}
          {msg.tool_calls && msg.tool_calls.length > 0 && (
            <div className="border-t border-zinc-800/50 px-4 py-3">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600 mb-1">Tool calls</p>
              <pre className="text-[11px] text-zinc-500 whitespace-pre-wrap break-words font-mono leading-relaxed">
                {JSON.stringify(msg.tool_calls, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ---------- Queued message card ---------- */

function QueuedMessageCard({
  item,
  onEdit,
  onCancel,
  onInject,
  onRedirect,
}: {
  item: { id: string; text: string; queuedAt: number; type: "inject" | "redirect" | "next" };
  onEdit: () => void;
  onCancel: () => void;
  onInject: () => void;
  onRedirect: () => void;
}) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 5000);
    return () => clearInterval(timer);
  }, []);

  const secondsAgo = Math.max(0, Math.floor((now - item.queuedAt) / 1000));
  const agoLabel =
    secondsAgo < 5
      ? "just now"
      : secondsAgo < 60
        ? `${secondsAgo}s ago`
        : `${Math.floor(secondsAgo / 60)}m ago`;

  const typeLabel = item.type === "next" ? "queued" : item.type;
  const typeColor = item.type === "inject"
    ? "bg-violet-500/15 text-violet-400"
    : item.type === "redirect"
      ? "bg-red-500/15 text-red-400"
      : "bg-amber-500/15 text-amber-400";

  return (
    <div className="flex items-start gap-2 rounded-lg border border-amber-600/20 bg-amber-500/[0.04] px-3 py-2">
      <div className="mt-0.5 shrink-0">
        <Zap size={12} className="text-amber-400/60" />
      </div>

      <div className="min-w-0 flex-1">
        <p className="text-xs text-zinc-300 leading-relaxed break-words line-clamp-3">
          {item.text}
        </p>
        <div className="flex items-center gap-2 mt-1.5 flex-wrap">
          <span className={cn("inline-flex items-center rounded-full px-1.5 py-px text-[9px] font-medium", typeColor)}>
            {typeLabel}
          </span>
          <span className="text-[9px] text-zinc-600 tabular-nums">{agoLabel}</span>

          {/* Action buttons — always visible */}
          <div className="flex items-center gap-1 ml-auto">
            {item.type === "next" && (
              <>
                <button
                  type="button"
                  onClick={onInject}
                  title="Inject now — send as steering instruction"
                  className="rounded px-2 py-0.5 text-[9px] font-medium text-violet-400 bg-violet-500/10 hover:bg-violet-500/20 transition-colors"
                >
                  Inject
                </button>
                <button
                  type="button"
                  onClick={onRedirect}
                  title="Redirect — stop current work and pivot"
                  className="rounded px-2 py-0.5 text-[9px] font-medium text-red-400 bg-red-500/10 hover:bg-red-500/20 transition-colors"
                >
                  Redirect
                </button>
              </>
            )}
            <button
              type="button"
              onClick={onEdit}
              title="Edit — move back to composer"
              className="rounded p-0.5 text-zinc-500 hover:text-[#a3b396] hover:bg-zinc-800 transition-colors"
            >
              <Pencil size={11} />
            </button>
            <button
              type="button"
              onClick={onCancel}
              title="Remove"
              className="rounded p-0.5 text-zinc-500 hover:text-red-400 hover:bg-zinc-800 transition-colors"
            >
              <X size={11} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- Ask user card with free text input ---------- */

function AskUserCard({
  prompt,
  options,
  onReply,
}: {
  prompt: { question: string; context_note: string };
  options: Array<{ id: string; label: string }>;
  onReply: (text: string) => void;
}) {
  const [freeText, setFreeText] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div className="rounded-xl border-2 border-amber-600/40 bg-amber-500/5 px-4 py-4 space-y-3">
      <div className="flex items-center gap-2">
        <span className="inline-block h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
        <span className="text-xs font-semibold text-amber-300">Input Requested</span>
      </div>
      <p className="text-sm text-zinc-200">{prompt.question}</p>
      {prompt.context_note && (
        <p className="text-[11px] text-zinc-500">{prompt.context_note}</p>
      )}
      {options.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {options.map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => onReply(opt.label)}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-2.5 py-1 text-xs text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100 transition-colors"
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          const text = freeText.trim();
          if (text) {
            onReply(text);
            setFreeText("");
          }
        }}
        className="flex items-center gap-2"
      >
        <input
          ref={inputRef}
          type="text"
          value={freeText}
          onChange={(e) => setFreeText(e.target.value)}
          placeholder="Type a response..."
          className="flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-amber-500/50 focus:border-amber-500/50"
        />
        <button
          type="submit"
          disabled={!freeText.trim()}
          className="rounded-lg bg-amber-600 px-3 py-2 text-xs font-medium text-white hover:bg-amber-500 disabled:opacity-40 transition-colors"
        >
          Reply
        </button>
      </form>
    </div>
  );
}

/* ---------- Copy to clipboard button ---------- */

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }).catch(() => {
      // Fallback for environments without clipboard API
    });
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="flex h-5 w-5 items-center justify-center rounded text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800 transition-colors"
      title="Copy to clipboard"
    >
      {copied ? <Check size={11} className="text-[#a3b396]" /> : <Copy size={11} />}
    </button>
  );
}
