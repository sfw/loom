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
  createConversation,
  patchConversation,
  fetchConversationDetail,
  fetchConversationEvents,
  fetchConversationMessages,
  fetchConversationStatus,
  injectConversationInstruction,
  resolveConversationApproval,
  sendConversationMessage,
  stopConversationTurn,
  subscribeConversationStream,
  type ConversationApproval,
  type ConversationDetail,
  type ConversationMessageAttachments,
  type ConversationMessage,
  type ConversationPrompt,
  type ConversationSummary,
  type ConversationStatus,
  type ConversationStreamEvent,
  type ModelInfo,
  type WorkspaceOverview,
} from "../api";
import {
  conversationEventDetail,
  conversationEventPills,
  conversationEventTitle,
  matchesWorkspaceSearch,
  normalizeConversationTurnSeparatorPayload,
  normalizeConversationPrompt,
  summarizeMessage,
} from "../history";
import { isTransientRequestError } from "../utils";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface ConversationState {
  selectedConversationId: string;
  conversationDetail: ConversationDetail | null;
  loadingConversationDetail: boolean;
  conversationLoadError: string;
  conversationMessages: ConversationMessage[];
  conversationEvents: ConversationStreamEvent[];
  conversationStatus: ConversationStatus | null;
  conversationStreaming: boolean;
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
  queuedMessages: Array<{
    id: string;
    text: string;
    queuedAt: number;
    type: "inject" | "redirect" | "next";
  }>;
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

  // Pagination
  hasOlderMessages: boolean;
  loadingOlderMessages: boolean;

  // Computed
  conversationIsProcessing: boolean;
  pendingConversationApproval: ConversationApproval | null;
  conversationAwaitingApproval: boolean;
  pendingConversationPrompt: ConversationPrompt | null;
  conversationAwaitingInput: boolean;
  conversationPhaseLabel: string;
  quickReplyOptions: Array<{ id: string; label: string; description?: string }>;
  selectedConversationSummary: ConversationDetail | { id: string; title: string; model_name: string; last_active_at: string; started_at: string; linked_run_ids: string[] } | null;
  filteredConversationEvents: ConversationStreamEvent[];
  filteredConversationMessages: ConversationMessage[];
  visibleConversationEvents: ConversationStreamEvent[];
  visibleConversationMessages: ConversationMessage[];
  totalConversationMatches: number;
  selectedConversationRunIds: string[];

  // Refs
  conversationComposerRef: React.RefObject<HTMLElement | null>;
  conversationMatchRefs: React.MutableRefObject<Array<HTMLDivElement | null>>;
}

export interface ConversationActions {
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
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
  handleCreateConversation: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleSendConversationMessage: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  submitConversationMessage: (
    rawMessage: string,
    attachments?: ConversationMessageAttachments,
  ) => Promise<boolean>;
  handleQuickConversationReply: (optionLabel: string) => Promise<void>;
  handleInjectConversationInstruction: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleResolveConversationApproval: (decision: "approve" | "approve_all" | "deny") => Promise<void>;
  handleStopConversationTurn: () => Promise<void>;
  handlePrefillStarterConversation: () => void;
  focusConversationComposer: () => void;
  refreshConversation: (conversationId: string) => Promise<void>;
  retryConversationLoad: () => Promise<void>;
  loadOlderMessages: () => Promise<void>;
  scrollConversationMatchIntoView: (index: number) => void;
  stepConversationMatch: (delta: number) => void;
}

// ---------------------------------------------------------------------------
// Replay helpers
// ---------------------------------------------------------------------------

export function conversationEventKey(event: ConversationStreamEvent): string {
  if (event._client_id) {
    return `optimistic:${event._client_id}`;
  }
  if (!event._optimistic && event.turn_number == null && Number(event.seq || 0) > 0) {
    return `event:${event.seq}`;
  }
  if (event.turn_number != null) {
    return `synthetic:${event.turn_number}:${event.event_type}:${event.seq}`;
  }
  return `event:${event.seq}:${event.event_type}:${event.created_at}`;
}

function hasConversationAttachments(
  attachments: ConversationMessageAttachments | undefined,
): boolean {
  if (!attachments) {
    return false;
  }
  return Boolean(
    attachments.workspace_paths?.length
    || attachments.workspace_files?.length
    || attachments.workspace_directories?.length
    || attachments.content_blocks?.length,
  );
}

function conversationAttachmentPayload(
  attachments: ConversationMessageAttachments | undefined,
): Record<string, unknown> {
  if (!attachments) {
    return {};
  }
  const payload: Record<string, unknown> = {};
  if (attachments.workspace_paths?.length) payload.workspace_paths = attachments.workspace_paths;
  if (attachments.workspace_files?.length) payload.workspace_files = attachments.workspace_files;
  if (attachments.workspace_directories?.length) {
    payload.workspace_directories = attachments.workspace_directories;
  }
  if (attachments.content_blocks?.length) payload.content_blocks = attachments.content_blocks;
  return payload;
}

export function conversationMessageKey(message: ConversationMessage): string {
  if (Number(message.id || 0) > 0) {
    return `message:${message.id}`;
  }
  return `message:${message.turn_number}:${message.role}:${message.created_at}`;
}

export function isDurableConversationEvent(
  event: ConversationStreamEvent | undefined,
): boolean {
  return Boolean(event && !event._optimistic && Number(event.id || 0) > 0);
}

export function mergeConversationMessages(
  current: ConversationMessage[],
  incoming: ConversationMessage[],
  mode: "prepend" | "append",
): ConversationMessage[] {
  if (incoming.length === 0) {
    return current;
  }
  const seen = new Set(current.map((message) => conversationMessageKey(message)));
  const unique = incoming.filter((message) => !seen.has(conversationMessageKey(message)));
  if (unique.length === 0) {
    return current;
  }
  return mode === "prepend" ? [...unique, ...current] : [...current, ...unique];
}

export function mergeConversationEvents(
  current: ConversationStreamEvent[],
  incoming: ConversationStreamEvent[],
  mode: "prepend" | "append",
): ConversationStreamEvent[] {
  if (incoming.length === 0) {
    return current;
  }
  const seen = new Set(current.map((event) => conversationEventKey(event)));
  const unique = incoming.filter((event) => !seen.has(conversationEventKey(event)));
  if (unique.length === 0) {
    return current;
  }
  return mode === "prepend" ? [...unique, ...current] : [...current, ...unique];
}

export function hasOlderConversationHistory(
  messages: ConversationMessage[],
  events: ConversationStreamEvent[],
): boolean {
  const oldestTurn = Number(messages[0]?.turn_number || 0);
  if (oldestTurn > 1) {
    return true;
  }
  const oldestEvent = events[0];
  return isDurableConversationEvent(oldestEvent) && Number(oldestEvent?.seq || 0) > 1;
}

export function durableConversationSeq(events: ConversationStreamEvent[]): number {
  return Math.max(
    0,
    ...events
      .filter((event) => !event._optimistic && event.turn_number == null && Number(event.seq || 0) > 0)
      .map((event) => event.seq),
  );
}

function settledValue<T>(
  result: PromiseSettledResult<T>,
  fallback: T,
): T {
  return result.status === "fulfilled" ? result.value : fallback;
}

function firstSettledError(
  results: PromiseSettledResult<unknown>[],
): Error | null {
  for (const result of results) {
    if (result.status !== "rejected") {
      continue;
    }
    if (result.reason instanceof Error) {
      return result.reason;
    }
    return new Error(String(result.reason || "Request failed"));
  }
  return null;
}

export function shouldContinuouslySyncConversation(options: {
  localProcessing: boolean;
  turnPending: boolean;
  streaming: boolean;
  serverReportedActive: boolean;
}): boolean {
  return Boolean(
    options.localProcessing
    || options.turnPending
    || options.streaming
    || options.serverReportedActive,
  );
}

export function appendStreamingThinkingChunk(
  current: string,
  chunk: string,
): string {
  const normalizedChunk = String(chunk || "").replace(/\r\n?/g, "\n");
  if (!normalizedChunk) {
    return current;
  }
  if (!current) {
    return normalizedChunk;
  }

  const currentTail = current.slice(-1);
  const chunkHead = normalizedChunk[0] || "";
  if (!chunkHead) {
    return current;
  }

  if (/\s/.test(currentTail) || /\s/.test(chunkHead)) {
    return `${current}${normalizedChunk}`;
  }

  if (/^[,.;:!?)}\]]/.test(normalizedChunk)) {
    return `${current}${normalizedChunk}`;
  }

  if (/[a-z0-9]/.test(currentTail) && /[a-z]/.test(chunkHead)) {
    return `${current}${normalizedChunk}`;
  }

  return `${current}\n\n${normalizedChunk}`;
}

export function isConversationStreamHealthy(
  lastActivityAt: number,
  options: {
    now?: number;
    healthWindowMs?: number;
  } = {},
): boolean {
  const {
    now = Date.now(),
    healthWindowMs = CONVERSATION_STREAM_HEALTH_WINDOW_MS,
  } = options;
  if (lastActivityAt <= 0) {
    return false;
  }
  return (now - lastActivityAt) < healthWindowMs;
}

const INITIAL_TURN_PROGRESS_EVENT_TYPES = new Set([
  "assistant_text",
  "assistant_thinking",
  "tool_call_started",
  "tool_call_completed",
  "turn_separator",
  "turn_interrupted",
]);

export function isInitialTurnProgressEvent(eventType: string): boolean {
  return INITIAL_TURN_PROGRESS_EVENT_TYPES.has(eventType);
}

export function defaultConversationTitle(title: string | null | undefined): boolean {
  return /^Conversation [a-f0-9]{6,}$/.test(String(title || ""));
}

const CONVERSATION_STREAM_HEALTH_WINDOW_MS = 6000;
const CONVERSATION_SYNC_INTERVAL_MS = 5000;

export interface PendingConversationTitle {
  conversationId: string;
  title: string;
}

export function reconcilePendingConversationTitle<T extends { id: string; title: string }>(
  detail: T,
  pendingTitle: PendingConversationTitle | null | undefined,
): { detail: T; keepPending: boolean } {
  if (!pendingTitle || pendingTitle.conversationId !== detail.id) {
    return { detail, keepPending: false };
  }

  const incomingTitle = String(detail.title || "").trim();
  if (!incomingTitle || defaultConversationTitle(incomingTitle)) {
    return {
      detail: {
        ...detail,
        title: pendingTitle.title,
      },
      keepPending: true,
    };
  }

  if (incomingTitle === pendingTitle.title) {
    return { detail, keepPending: false };
  }

  return { detail, keepPending: false };
}

function optimisticUserText(event: ConversationStreamEvent): string {
  return String(event.payload.text || "");
}

function optimisticUserEventMatchIndex(
  events: ConversationStreamEvent[],
  text: string,
  options?: { preferNonQueued?: boolean },
): number {
  const matchEvent = (event: ConversationStreamEvent, allowQueued: boolean) => (
    event._optimistic
    && event.event_type === "user_message"
    && optimisticUserText(event) === text
    && (allowQueued || event._delivery_state !== "queued")
  );

  if (options?.preferNonQueued !== false) {
    const nonQueuedIndex = events.findIndex((event) => matchEvent(event, false));
    if (nonQueuedIndex >= 0) {
      return nonQueuedIndex;
    }
  }

  return events.findIndex((event) => matchEvent(event, true));
}

export function reconcileOptimisticConversationEvents(
  current: ConversationStreamEvent[],
  acknowledgedEvents: ConversationStreamEvent[],
): ConversationStreamEvent[] {
  const acknowledgedUserTexts = acknowledgedEvents
    .filter((event) => event.event_type === "user_message")
    .map((event) => optimisticUserText(event))
    .filter((text) => text.length > 0);
  if (acknowledgedUserTexts.length === 0) {
    return current;
  }

  let next = current;
  for (const text of acknowledgedUserTexts) {
    const matchIndex = optimisticUserEventMatchIndex(next, text);
    if (matchIndex >= 0) {
      const matchedEvent = next[matchIndex];
      const matchedClientId = String(matchedEvent?._client_id || "").trim();
      next = next.filter((event, index) => (
        index !== matchIndex
        && (
          !matchedClientId
          || event._client_id !== `${matchedClientId}:attachments`
        )
      ));
    }
  }
  return next;
}

export async function hydrateConversationReplayPages(args: {
  seedMessages: ConversationMessage[];
  seedEvents: ConversationStreamEvent[];
  fetchMessagesPage: (beforeTurn: number) => Promise<ConversationMessage[]>;
  fetchEventsPage: (options: { beforeSeq?: number; beforeTurn?: number }) => Promise<ConversationStreamEvent[]>;
}): Promise<{
  messages: ConversationMessage[];
  events: ConversationStreamEvent[];
  hasOlder: boolean;
}> {
  const {
    seedMessages,
    seedEvents,
    fetchMessagesPage,
    fetchEventsPage,
  } = args;
  let replayMessages = seedMessages;
  let replayEvents = seedEvents;
  let keepLoading = hasOlderConversationHistory(replayMessages, replayEvents);

  while (keepLoading) {
    const oldestTurn = replayMessages[0]?.turn_number;
    const oldestEvent = replayEvents[0];
    const oldestSeq = isDurableConversationEvent(oldestEvent) ? oldestEvent?.seq : undefined;
    const [olderMessages, olderEvents] = await Promise.all([
      oldestTurn != null
        ? fetchMessagesPage(oldestTurn)
        : Promise.resolve([]),
      (oldestSeq != null || oldestTurn != null)
        ? fetchEventsPage({
            beforeSeq: oldestSeq,
            beforeTurn: oldestTurn,
          })
        : Promise.resolve([]),
    ]);

    if (olderMessages.length === 0 && olderEvents.length === 0) {
      keepLoading = false;
      break;
    }

    replayMessages = mergeConversationMessages(replayMessages, olderMessages, "prepend");
    replayEvents = mergeConversationEvents(replayEvents, olderEvents, "prepend");
    keepLoading = hasOlderConversationHistory(replayMessages, replayEvents);
  }

  return {
    messages: replayMessages,
    events: replayEvents,
    hasOlder: keepLoading,
  };
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useConversation(deps: {
  selectedConversationId: string;
  connectionState?: "connecting" | "connected" | "failed";
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
  selectedWorkspaceId: string;
  overview: WorkspaceOverview | null;
  models: ModelInfo[];
  setError: React.Dispatch<React.SetStateAction<string>>;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
  setActiveTab: React.Dispatch<React.SetStateAction<import("../utils").ViewTab>>;
  refreshWorkspaceSurface: (workspaceId: string) => Promise<void>;
  syncConversationSummary: (
    detail: ConversationSummary | ConversationDetail,
    options?: {
      incrementCount?: boolean;
      processing?: boolean;
      workspaceId?: string;
    },
  ) => void;
  setConversationProcessing: (
    conversationId: string,
    processing: boolean,
    options?: {
      lastActiveAt?: string;
      workspaceId?: string;
    },
  ) => void;
  removeApprovalItem: (itemId: string, workspaceId?: string) => void;
}): ConversationState & ConversationActions {
  const {
    selectedConversationId,
    connectionState = "connected",
    setSelectedConversationId,
    selectedWorkspaceId,
    overview,
    setError,
    setNotice,
    setActiveTab,
    refreshWorkspaceSurface,
    syncConversationSummary,
    setConversationProcessing,
    removeApprovalItem,
  } = deps;
  const [conversationDetail, setConversationDetail] = useState<ConversationDetail | null>(null);
  const [loadingConversationDetail, setLoadingConversationDetail] = useState(false);
  const [conversationLoadError, setConversationLoadError] = useState("");
  const [conversationMessages, setConversationMessages] = useState<ConversationMessage[]>([]);
  const [conversationEvents, setConversationEvents] = useState<ConversationStreamEvent[]>([]);
  const [optimisticConversationEvents, setOptimisticConversationEvents] = useState<ConversationStreamEvent[]>([]);
  const [conversationStatus, setConversationStatus] = useState<ConversationStatus | null>(null);
  const [conversationStreaming, setConversationStreaming] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [streamingThinking, setStreamingThinking] = useState("");
  const [streamingToolCalls, setStreamingToolCalls] = useState<Array<{
    id: string;
    tool_name: string;
    started_at: string;
    completed: boolean;
    success?: boolean;
    args_preview?: string;
    output_preview?: string;
    elapsed_ms?: number;
  }>>([]);
  const [lastTurnStats, setLastTurnStats] = useState<{
    tokens: number;
    tool_count: number;
    visible: boolean;
  } | null>(null);
  const [queuedMessages, setQueuedMessages] = useState<Array<{
    id: string;
    text: string;
    queuedAt: number;
    type: "inject" | "redirect" | "next";
  }>>([]);
  const [newConversationModel, setNewConversationModel] = useState("");
  const [newConversationPrompt, setNewConversationPrompt] = useState("");
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [conversationComposerMessage, setConversationComposerMessage] = useState("");
  const [sendingConversationMessage, setSendingConversationMessage] = useState(false);
  const [conversationInjectMessage, setConversationInjectMessage] = useState("");
  const [sendingConversationInject, setSendingConversationInject] = useState(false);
  const [conversationTurnPending, setConversationTurnPending] = useState(false);
  const [conversationHistoryQuery, setConversationHistoryQuery] = useState("");
  const [activeConversationMatchIndex, setActiveConversationMatchIndex] = useState(0);

  // Refs
  const conversationComposerRef = useRef<HTMLElement | null>(null);
  const conversationRefreshTimerRef = useRef<number | null>(null);
  const conversationMatchRefs = useRef<Array<HTMLDivElement | null>>([]);
  const turnTimeoutCleanupRef = useRef<(() => void) | null>(null);
  const autoTitledRef = useRef(false);
  const conversationReplayHydrationTokenRef = useRef(0);
  const conversationMessagesRef = useRef<ConversationMessage[]>([]);
  const conversationEventsRef = useRef<ConversationStreamEvent[]>([]);
  const optimisticConversationEventsRef = useRef<ConversationStreamEvent[]>([]);
  const conversationDetailRef = useRef<ConversationDetail | null>(null);
  const conversationStatusRef = useRef<ConversationStatus | null>(null);
  const conversationStreamingRef = useRef(false);
  const streamingTextRef = useRef("");
  const streamingThinkingRef = useRef("");
  const streamingToolCallsRef = useRef<Array<{
    id: string;
    tool_name: string;
    started_at: string;
    completed: boolean;
    success?: boolean;
    args_preview?: string;
    output_preview?: string;
    elapsed_ms?: number;
  }>>([]);
  const lastTurnStatsRef = useRef<{
    tokens: number;
    tool_count: number;
    visible: boolean;
  } | null>(null);
  const queuedMessagesRef = useRef<Array<{
    id: string;
    text: string;
    queuedAt: number;
    type: "inject" | "redirect" | "next";
  }>>([]);
  const conversationTurnPendingRef = useRef(false);
  const conversationStreamAfterSeqRef = useRef(0);
  const conversationStreamMaxSeenSeqRef = useRef(0);
  const conversationStreamActivityAtRef = useRef(0);
  const pendingConversationStreamEventsRef = useRef<ConversationStreamEvent[]>([]);
  const conversationStreamFrameRef = useRef<number | null>(null);
  const conversationSyncInFlightRef = useRef(false);
  const pendingConversationTitleRef = useRef<PendingConversationTitle | null>(null);
  const [conversationStreamReady, setConversationStreamReady] = useState(false);

  function trimmedConversationTitleFromText(text: string): string {
    const raw = String(text || "").trim();
    return raw.length > 60 ? `${raw.slice(0, 57)}...` : raw;
  }

  function applyIncomingConversationDetail(detail: ConversationDetail) {
    const reconciled = reconcilePendingConversationTitle(
      detail,
      pendingConversationTitleRef.current,
    );
    pendingConversationTitleRef.current = reconciled.keepPending
      ? pendingConversationTitleRef.current
      : null;
    setConversationDetail(reconciled.detail);
    syncConversationSummary(reconciled.detail, {
      processing: Boolean(conversationStatusRef.current?.processing ?? reconciled.detail.is_active),
      workspaceId: selectedWorkspaceId,
    });
  }

  function maybeApplyOptimisticConversationTitle(rawText: string) {
    const detail = conversationDetailRef.current;
    if (!detail || autoTitledRef.current || !defaultConversationTitle(detail.title || "")) {
      return;
    }
    const nextTitle = trimmedConversationTitleFromText(rawText);
    if (!nextTitle) return;
    const conversationId = detail.id;
    const workspaceId = selectedWorkspaceId;
    autoTitledRef.current = true;
    pendingConversationTitleRef.current = {
      conversationId,
      title: nextTitle,
    };
    setConversationDetail((current) => current ? { ...current, title: nextTitle } : current);
    void patchConversation(conversationId, { title: nextTitle }).then((updatedDetail) => {
      if (selectedConversationId === conversationId) {
        applyIncomingConversationDetail(updatedDetail);
      }
      syncConversationSummary(updatedDetail, {
        processing: Boolean(conversationStatusRef.current?.processing ?? updatedDetail.is_active),
        workspaceId,
      });
    }).catch(() => {
      autoTitledRef.current = false;
      pendingConversationTitleRef.current = null;
    });
  }

  // Pagination
  const [hasOlderMessages, setHasOlderMessages] = useState(false);
  const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
  const MESSAGE_PAGE_SIZE = 250;
  const EVENT_PAGE_SIZE = 250;

  // ---------------------------------------------------------------------------
  // Computed values
  // ---------------------------------------------------------------------------

  const conversationIsProcessing = useMemo(
    () => Boolean(conversationStatus?.processing || conversationTurnPending),
    [conversationStatus?.processing, conversationTurnPending],
  );
  const pendingConversationApproval = useMemo(
    () => conversationStatus?.pending_approval || null,
    [conversationStatus?.pending_approval],
  );
  const conversationAwaitingApproval = useMemo(() => Boolean(
    conversationStatus?.awaiting_approval && pendingConversationApproval,
  ), [conversationStatus?.awaiting_approval, pendingConversationApproval]);
  const pendingConversationPrompt = useMemo(
    () => conversationStatus?.pending_prompt || null,
    [conversationStatus?.pending_prompt],
  );
  const conversationAwaitingInput = useMemo(() => Boolean(
    conversationStatus?.awaiting_user_input && pendingConversationPrompt,
  ), [conversationStatus?.awaiting_user_input, pendingConversationPrompt]);
  const conversationPhaseLabel = useMemo(() => (
    conversationAwaitingApproval
      ? "Awaiting approval"
      : conversationIsProcessing
        ? "Running"
        : conversationAwaitingInput
          ? "Awaiting input"
          : conversationStreaming
            ? "Live"
            : "Idle"
  ), [
    conversationAwaitingApproval,
    conversationAwaitingInput,
    conversationIsProcessing,
    conversationStreaming,
  ]);
  const quickReplyOptions = useMemo(() => (
    conversationAwaitingInput
    && pendingConversationPrompt
    && pendingConversationPrompt.question_type === "single_choice"
      ? pendingConversationPrompt.options
      : []
  ), [conversationAwaitingInput, pendingConversationPrompt]);
  const workspaceConversationRows = useMemo(
    () => overview?.recent_conversations || [],
    [overview?.recent_conversations],
  );
  const selectedConversationSummary = useMemo(() => (
    conversationDetail
    || workspaceConversationRows.find((conversation) => conversation.id === selectedConversationId)
    || null
  ), [conversationDetail, selectedConversationId, workspaceConversationRows]);
  const selectedConversationServerActive = useMemo(() => Boolean(
    selectedConversationSummary
    && "is_active" in selectedConversationSummary
    && selectedConversationSummary.is_active,
  ), [selectedConversationSummary]);
  const selectedConversationRunIds = useMemo(() => (
    selectedConversationSummary
      ? ("linked_run_ids" in selectedConversationSummary ? selectedConversationSummary.linked_run_ids : [])
      : []
  ), [selectedConversationSummary]);
  const allConversationEvents = useMemo(() => (
    optimisticConversationEvents.length > 0
      ? [...conversationEvents, ...optimisticConversationEvents]
      : conversationEvents
  ), [conversationEvents, optimisticConversationEvents]);
  const filteredConversationEvents = useMemo(() => allConversationEvents.filter((event) =>
    matchesWorkspaceSearch(
      conversationHistoryQuery,
      event.event_type,
      conversationEventTitle(event),
      conversationEventDetail(event),
      event.payload,
      conversationEventPills(event).join(" "),
    ),
  ), [allConversationEvents, conversationHistoryQuery]);
  const filteredConversationMessages = useMemo(() => conversationMessages.filter((message) =>
    matchesWorkspaceSearch(
      conversationHistoryQuery,
      message.role,
      message.content,
      message.tool_name,
      message.tool_call_id,
      summarizeMessage(message),
    ),
  ), [conversationHistoryQuery, conversationMessages]);
  const visibleConversationEvents = useMemo(() => (
    conversationHistoryQuery.trim()
      ? filteredConversationEvents
      : allConversationEvents
  ), [allConversationEvents, conversationHistoryQuery, filteredConversationEvents]);
  const visibleConversationMessages = useMemo(() => (
    conversationHistoryQuery.trim()
      ? filteredConversationMessages
      : conversationMessages
  ), [conversationHistoryQuery, conversationMessages, filteredConversationMessages]);
  const totalConversationMatches = useMemo(
    () => filteredConversationEvents.length + filteredConversationMessages.length,
    [filteredConversationEvents.length, filteredConversationMessages.length],
  );

  useEffect(() => {
    conversationMessagesRef.current = conversationMessages;
  }, [conversationMessages]);

  useEffect(() => {
    conversationEventsRef.current = conversationEvents;
  }, [conversationEvents]);

  useEffect(() => {
    optimisticConversationEventsRef.current = optimisticConversationEvents;
  }, [optimisticConversationEvents]);

  useEffect(() => {
    conversationDetailRef.current = conversationDetail;
  }, [conversationDetail]);

  useEffect(() => {
    conversationStatusRef.current = conversationStatus;
  }, [conversationStatus]);

  useEffect(() => {
    conversationStreamingRef.current = conversationStreaming;
  }, [conversationStreaming]);

  useEffect(() => {
    streamingTextRef.current = streamingText;
  }, [streamingText]);

  useEffect(() => {
    streamingThinkingRef.current = streamingThinking;
  }, [streamingThinking]);

  useEffect(() => {
    streamingToolCallsRef.current = streamingToolCalls;
  }, [streamingToolCalls]);

  useEffect(() => {
    lastTurnStatsRef.current = lastTurnStats;
  }, [lastTurnStats]);

  useEffect(() => {
    queuedMessagesRef.current = queuedMessages;
  }, [queuedMessages]);

  useEffect(() => {
    conversationTurnPendingRef.current = conversationTurnPending;
  }, [conversationTurnPending]);

  // ---------------------------------------------------------------------------
  // useEffectEvent handlers
  // ---------------------------------------------------------------------------

  const refreshConversation = useEffectEvent(async (conversationId: string) => {
    const latestSeq = durableConversationSeq(conversationEventsRef.current);
    const [detailResult, eventsResult, statusResult, messagesResult] = await Promise.allSettled([
      fetchConversationDetail(conversationId),
      fetchConversationEvents(conversationId, { afterSeq: latestSeq, limit: EVENT_PAGE_SIZE }),
      fetchConversationStatus(conversationId),
      fetchConversationMessages(conversationId, {
        latest: true,
        limit: MESSAGE_PAGE_SIZE,
      }),
    ]);
    const detail = settledValue(detailResult, null);
    const events = settledValue(eventsResult, [] as ConversationStreamEvent[]);
    const status = settledValue(statusResult, null);
    const messages = settledValue(messagesResult, [] as ConversationMessage[]);
    if (detail == null && events.length === 0 && status == null && messages.length === 0) {
      throw firstSettledError([detailResult, eventsResult, statusResult, messagesResult])
        || new Error("Failed to load conversation.");
    }
    if (detail != null) {
      applyIncomingConversationDetail(detail);
    } else if (status != null) {
      setConversationProcessing(conversationId, Boolean(status.processing), {
        workspaceId: selectedWorkspaceId,
      });
    }
    setConversationLoadError("");
    setConversationStreamReady(true);
    if (messages.length > 0) {
      setConversationMessages((current) =>
        mergeConversationMessages(current, messages, "append"),
      );
    }
    if (events.length > 0) {
      setConversationEvents((current) => {
        const seen = new Set(current.map((row) => conversationEventKey(row)));
        const appended = events.filter((row) => !seen.has(conversationEventKey(row)));
        return appended.length > 0 ? [...current, ...appended] : current;
      });
      setOptimisticConversationEvents((current) =>
        reconcileOptimisticConversationEvents(current, events),
      );
      const maxFetchedSeq = durableConversationSeq(events);
      conversationStreamAfterSeqRef.current = Math.max(
        conversationStreamAfterSeqRef.current,
        maxFetchedSeq,
      );
      conversationStreamMaxSeenSeqRef.current = Math.max(
        conversationStreamMaxSeenSeqRef.current,
        ...events.map((row) => row.seq),
      );
    }
    if (status != null) {
      setConversationStatus(status);
      setConversationProcessing(conversationId, Boolean(status.processing), {
        workspaceId: selectedWorkspaceId,
      });
    }
    // Sync turn pending state from server — clears stale "Processing" indicators
    setLoadingConversationDetail(false);
    if (status != null && !status.processing) {
      setConversationTurnPending(false);
      setConversationStreaming(false);
      setStreamingText(""); setStreamingThinking("");
      setStreamingToolCalls([]);
    }
  });

  const retryConversationLoad = useEffectEvent(async () => {
    if (!selectedConversationId) {
      return;
    }
    setLoadingConversationDetail(true);
    setConversationLoadError("");
    try {
      await refreshConversation(selectedConversationId);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load conversation.";
      setConversationLoadError(message);
      setError(message);
      setLoadingConversationDetail(false);
    }
  });

  const hydrateConversationReplay = useEffectEvent(async (options?: {
    conversationId?: string;
    seedMessages?: ConversationMessage[];
    seedEvents?: ConversationStreamEvent[];
  }) => {
    const conversationId = options?.conversationId || selectedConversationId;
    const seededHydration = Array.isArray(options?.seedMessages) || Array.isArray(options?.seedEvents);
    if (!conversationId || loadingOlderMessages || (!seededHydration && !hasOlderMessages)) {
      return;
    }
    const token = ++conversationReplayHydrationTokenRef.current;
    setLoadingOlderMessages(true);
    try {
      const replay = await hydrateConversationReplayPages({
        seedMessages: options?.seedMessages || conversationMessagesRef.current,
        seedEvents: options?.seedEvents || conversationEventsRef.current,
        fetchMessagesPage: (beforeTurn) =>
          fetchConversationMessages(conversationId, {
            beforeTurn,
            limit: MESSAGE_PAGE_SIZE,
          }),
        fetchEventsPage: ({ beforeSeq, beforeTurn }) =>
          fetchConversationEvents(conversationId, {
            beforeSeq,
            beforeTurn,
            limit: EVENT_PAGE_SIZE,
          }),
      });

      if (token === conversationReplayHydrationTokenRef.current) {
        setConversationMessages((current) =>
          mergeConversationMessages(current, replay.messages, "prepend"),
        );
        setConversationEvents((current) =>
          mergeConversationEvents(current, replay.events, "prepend"),
        );
        setHasOlderMessages(replay.hasOlder);
      }
    } catch {
      if (token === conversationReplayHydrationTokenRef.current) {
        setHasOlderMessages(true);
      }
    } finally {
      if (token === conversationReplayHydrationTokenRef.current) {
        setLoadingOlderMessages(false);
      }
    }
  });

  async function loadOlderMessages() {
    await hydrateConversationReplay();
  }

  const scheduleConversationRefresh = useEffectEvent(() => {
    if (conversationRefreshTimerRef.current !== null || !selectedConversationId) {
      return;
    }
    conversationRefreshTimerRef.current = window.setTimeout(() => {
      conversationRefreshTimerRef.current = null;
      void refreshConversation(selectedConversationId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh conversation.");
        }
      });
    }, 200);
  });

  const syncConversationLiveState = useEffectEvent(async (conversationId: string) => {
    if (!conversationId || conversationSyncInFlightRef.current) {
      return;
    }
    conversationSyncInFlightRef.current = true;
    try {
      const latestSeq = durableConversationSeq(conversationEventsRef.current);
      const [eventsResult, statusResult] = await Promise.allSettled([
        fetchConversationEvents(conversationId, {
          afterSeq: latestSeq,
          limit: EVENT_PAGE_SIZE,
        }),
        fetchConversationStatus(conversationId),
      ]);
      const events = settledValue(eventsResult, [] as ConversationStreamEvent[]);
      const status = settledValue(statusResult, null);
      const messages = status != null && !status.processing
        ? await fetchConversationMessages(conversationId, {
            latest: true,
            limit: MESSAGE_PAGE_SIZE,
          }).catch(() => [] as ConversationMessage[])
        : [];
      if (events.length === 0 && status == null) {
        throw firstSettledError([eventsResult, statusResult])
          || new Error("Failed to refresh conversation.");
      }

      if (messages.length > 0) {
        setConversationMessages((current) =>
          mergeConversationMessages(current, messages, "append"),
        );
      }
      if (events.length > 0) {
        setConversationEvents((current) => {
          const seen = new Set(current.map((row) => conversationEventKey(row)));
          const appended = events.filter((row) => !seen.has(conversationEventKey(row)));
          return appended.length > 0 ? [...current, ...appended] : current;
        });
        setOptimisticConversationEvents((current) =>
          reconcileOptimisticConversationEvents(current, events),
        );
        const maxFetchedSeq = durableConversationSeq(events);
        conversationStreamAfterSeqRef.current = Math.max(
          conversationStreamAfterSeqRef.current,
          maxFetchedSeq,
        );
        conversationStreamMaxSeenSeqRef.current = Math.max(
          conversationStreamMaxSeenSeqRef.current,
          ...events.map((row) => row.seq),
        );
        conversationStreamActivityAtRef.current = Date.now();
      }

      if (status != null) {
        setConversationStatus(status);
        setConversationProcessing(conversationId, Boolean(status.processing), {
          workspaceId: selectedWorkspaceId,
        });
      }
      setConversationStreamReady(true);

      if (status != null && !status.processing) {
        setConversationTurnPending(false);
        setConversationStreaming(false);
        setStreamingText("");
        setStreamingThinking("");
        setStreamingToolCalls([]);
      }
    } finally {
      conversationSyncInFlightRef.current = false;
    }
  });

  const flushConversationStreamBatch = useEffectEvent(() => {
    if (conversationStreamFrameRef.current !== null) {
      window.cancelAnimationFrame(conversationStreamFrameRef.current);
      conversationStreamFrameRef.current = null;
    }

    const batch = pendingConversationStreamEventsRef.current;
    if (batch.length === 0) {
      return;
    }
    pendingConversationStreamEventsRef.current = [];

    let nextEvents = conversationEventsRef.current;
    let nextOptimisticEvents = optimisticConversationEventsRef.current;
    let nextConversationStreaming = conversationStreamingRef.current;
    let nextStreamingText = streamingTextRef.current;
    let nextStreamingThinking = streamingThinkingRef.current;
    let nextStreamingToolCalls = streamingToolCallsRef.current;
    let nextLastTurnStats = lastTurnStatsRef.current;
    let nextQueuedMessages = queuedMessagesRef.current;
    let nextStatus = conversationStatusRef.current;
    let nextTurnPending = conversationTurnPendingRef.current;
    let queuedNextMessage: { id: string; text: string } | null = null;
    let shouldRefreshConversation = false;
    let autoTitleCandidate = "";

    const seenEventKeys = new Set(nextEvents.map((event) => conversationEventKey(event)));

    for (const event of batch) {
      conversationStreamActivityAtRef.current = Date.now();

      const eventKey = conversationEventKey(event);
      if (!seenEventKeys.has(eventKey)) {
        seenEventKeys.add(eventKey);
        nextEvents = [...nextEvents, event];
      }

      if (event.event_type === "user_message") {
        nextOptimisticEvents = reconcileOptimisticConversationEvents(nextOptimisticEvents, [event]);
      }

      const isNewEvent = event.seq > conversationStreamMaxSeenSeqRef.current;
      conversationStreamMaxSeenSeqRef.current = Math.max(
        conversationStreamMaxSeenSeqRef.current,
        event.seq,
      );
      if (!event._optimistic && event.turn_number == null && Number(event.seq || 0) > 0) {
        conversationStreamAfterSeqRef.current = Math.max(
          conversationStreamAfterSeqRef.current,
          event.seq,
        );
      }

      if (!isNewEvent) {
        continue;
      }

      if (event.event_type === "assistant_thinking") {
        nextConversationStreaming = true;
        const text = String(event.payload.text || "");
        if (text) {
          nextStreamingThinking = appendStreamingThinkingChunk(
            nextStreamingThinking,
            text,
          );
        }
      }

      if (event.event_type === "assistant_text") {
        nextConversationStreaming = true;
        const text = String(event.payload.text || "");
        if (text) {
          nextStreamingText += text;
        }
      }

      if (event.event_type === "tool_call_started") {
        const toolName = String(event.payload.tool_name || "tool");
        const callId = String(event.payload.tool_call_id || event.payload.id || `tc-${Date.now()}`);
        const argsPreview = typeof event.payload.args === "object" && event.payload.args
          ? Object.entries(event.payload.args as Record<string, unknown>)
              .slice(0, 3)
              .map(([key, value]) => `${key}: ${typeof value === "string" ? value.slice(0, 60) : JSON.stringify(value).slice(0, 40)}`)
              .join(", ")
          : "";
        nextStreamingToolCalls = [
          ...nextStreamingToolCalls,
          {
            id: callId,
            tool_name: toolName,
            started_at: event.created_at,
            completed: false,
            args_preview: argsPreview,
          },
        ];
      }

      if (event.event_type === "tool_call_completed") {
        const callId = String(event.payload.tool_call_id || event.payload.id || "");
        const toolName = String(event.payload.tool_name || "tool");
        const success = event.payload.success !== false;
        const elapsed = typeof event.payload.elapsed_ms === "number"
          ? event.payload.elapsed_ms
          : undefined;
        const outputPreview = typeof event.payload.output === "string"
          ? event.payload.output.slice(0, 120)
          : typeof event.payload.error === "string"
            ? event.payload.error.slice(0, 120)
            : "";
        const match = nextStreamingToolCalls.find((toolCall) =>
          toolCall.id === callId
          || (!callId && toolCall.tool_name === toolName && !toolCall.completed)
        );
        if (match) {
          nextStreamingToolCalls = nextStreamingToolCalls.map((toolCall) =>
            toolCall === match
              ? {
                  ...toolCall,
                  completed: true,
                  success,
                  elapsed_ms: elapsed,
                  output_preview: outputPreview,
                }
              : toolCall,
          );
        } else {
          nextStreamingToolCalls = [
            ...nextStreamingToolCalls,
            {
              id: callId || `tc-${Date.now()}`,
              tool_name: toolName,
              started_at: event.created_at,
              completed: true,
              success,
              elapsed_ms: elapsed,
              output_preview: outputPreview,
            },
          ];
        }
      }

      if (event.event_type === "turn_separator") {
        const stats = normalizeConversationTurnSeparatorPayload(event.payload);
        nextLastTurnStats = {
          tokens: stats.tokens,
          tool_count: stats.tool_count,
          visible: true,
        };
        nextConversationStreaming = false;
        nextStreamingText = "";
        nextStreamingThinking = "";
        nextStreamingToolCalls = [];
        const nextMessage = nextQueuedMessages.find((message) => message.type === "next");
        if (nextMessage) {
          queuedNextMessage = {
            id: nextMessage.id,
            text: nextMessage.text,
          };
          nextQueuedMessages = nextQueuedMessages.filter((message) => message.id !== nextMessage.id);
        }
        shouldRefreshConversation = true;
        setConversationProcessing(selectedConversationId, false, {
          lastActiveAt: event.created_at,
          workspaceId: selectedWorkspaceId,
        });

        const detail = conversationDetailRef.current;
        if (
          detail
          && defaultConversationTitle(detail.title || "")
          && !autoTitledRef.current
        ) {
          const firstOptimisticUser = nextOptimisticEvents.find((candidate) =>
            candidate.event_type === "user_message"
          );
          const firstUserMessage = conversationMessagesRef.current.find((message) => message.role === "user");
          const rawTitle = String(
            firstUserMessage?.content
            || firstOptimisticUser?.payload.text
            || "",
          ).trim();
          if (rawTitle) {
            autoTitleCandidate = rawTitle;
          }
        }
      }

      if (event.event_type === "turn_interrupted") {
        nextConversationStreaming = false;
        nextStreamingText = "";
        nextStreamingThinking = "";
        nextStreamingToolCalls = [];
        nextLastTurnStats = null;
        nextQueuedMessages = [];
        shouldRefreshConversation = true;
        setConversationProcessing(selectedConversationId, false, {
          lastActiveAt: event.created_at,
          workspaceId: selectedWorkspaceId,
        });
      }

      if (event.event_type === "user_message") {
        nextConversationStreaming = false;
        nextStreamingText = "";
        nextStreamingThinking = "";
        nextStreamingToolCalls = [];
        nextLastTurnStats = null;
      }

      if (isInitialTurnProgressEvent(event.event_type)) {
        nextTurnPending = false;
        turnTimeoutCleanupRef.current?.();
        turnTimeoutCleanupRef.current = null;
      }

      if (event.event_type === "user_message") {
        nextStatus = {
          conversation_id: nextStatus?.conversation_id || selectedConversationId,
          processing: true,
          stop_requested: nextStatus?.stop_requested || false,
          pending_inject_count: 0,
          awaiting_approval: false,
          pending_approval: null,
          awaiting_user_input: false,
          pending_prompt: null,
        };
        setConversationProcessing(selectedConversationId, true, {
          lastActiveAt: event.created_at,
          workspaceId: selectedWorkspaceId,
        });
      }

      if (event.event_type === "steering_instruction") {
        const pendingCount = Number(event.payload.pending_inject_count || 0);
        nextStatus = {
          conversation_id: nextStatus?.conversation_id || selectedConversationId,
          processing: nextStatus?.processing ?? true,
          stop_requested: nextStatus?.stop_requested || false,
          pending_inject_count: pendingCount,
          awaiting_approval: nextStatus?.awaiting_approval || false,
          pending_approval: nextStatus?.pending_approval || null,
          awaiting_user_input: nextStatus?.awaiting_user_input || false,
          pending_prompt: nextStatus?.pending_prompt || null,
        };
      }

      if (event.event_type === "approval_requested") {
        const pendingApproval = event.payload as unknown as ConversationApproval;
        nextTurnPending = false;
        nextConversationStreaming = false;
        nextStatus = {
          conversation_id: nextStatus?.conversation_id || selectedConversationId,
          processing: true,
          stop_requested: nextStatus?.stop_requested || false,
          pending_inject_count: nextStatus?.pending_inject_count || 0,
          awaiting_approval: true,
          pending_approval: pendingApproval,
          awaiting_user_input: false,
          pending_prompt: null,
        };
      }

      if (event.event_type === "approval_resolved") {
        nextStatus = {
          conversation_id: nextStatus?.conversation_id || selectedConversationId,
          processing: true,
          stop_requested: nextStatus?.stop_requested || false,
          pending_inject_count: nextStatus?.pending_inject_count || 0,
          awaiting_approval: false,
          pending_approval: null,
          awaiting_user_input: false,
          pending_prompt: null,
        };
      }

      if (event.event_type === "tool_call_completed" && event.payload.tool_name === "ask_user") {
        const pendingPrompt = normalizeConversationPrompt(event.payload.question_payload);
        if (pendingPrompt) {
          nextTurnPending = false;
          nextConversationStreaming = false;
          nextStatus = {
            conversation_id: nextStatus?.conversation_id || selectedConversationId,
            processing: false,
            stop_requested: false,
            pending_inject_count: 0,
            awaiting_approval: false,
            pending_approval: null,
            awaiting_user_input: true,
            pending_prompt: pendingPrompt,
          };
          setConversationProcessing(selectedConversationId, false, {
            lastActiveAt: event.created_at,
            workspaceId: selectedWorkspaceId,
          });
        }
      }

      if (event.event_type === "turn_separator" || event.event_type === "turn_interrupted") {
        nextStatus = {
          conversation_id: nextStatus?.conversation_id || selectedConversationId,
          processing: false,
          stop_requested: false,
          pending_inject_count: 0,
          awaiting_approval: false,
          pending_approval: null,
          awaiting_user_input: false,
          pending_prompt: null,
        };
      }
    }

    if (nextEvents !== conversationEventsRef.current) {
      conversationEventsRef.current = nextEvents;
      setConversationEvents(nextEvents);
    }
    if (nextOptimisticEvents !== optimisticConversationEventsRef.current) {
      optimisticConversationEventsRef.current = nextOptimisticEvents;
      setOptimisticConversationEvents(nextOptimisticEvents);
    }
    if (nextConversationStreaming !== conversationStreamingRef.current) {
      conversationStreamingRef.current = nextConversationStreaming;
      setConversationStreaming(nextConversationStreaming);
    }
    if (nextStreamingText !== streamingTextRef.current) {
      streamingTextRef.current = nextStreamingText;
      setStreamingText(nextStreamingText);
    }
    if (nextStreamingThinking !== streamingThinkingRef.current) {
      streamingThinkingRef.current = nextStreamingThinking;
      setStreamingThinking(nextStreamingThinking);
    }
    if (nextStreamingToolCalls !== streamingToolCallsRef.current) {
      streamingToolCallsRef.current = nextStreamingToolCalls;
      setStreamingToolCalls(nextStreamingToolCalls);
    }
    if (nextLastTurnStats !== lastTurnStatsRef.current) {
      lastTurnStatsRef.current = nextLastTurnStats;
      setLastTurnStats(nextLastTurnStats);
    }
    if (nextQueuedMessages !== queuedMessagesRef.current) {
      queuedMessagesRef.current = nextQueuedMessages;
      setQueuedMessages(nextQueuedMessages);
    }
    if (nextStatus !== conversationStatusRef.current) {
      conversationStatusRef.current = nextStatus;
      setConversationStatus(nextStatus);
    }
    if (nextTurnPending !== conversationTurnPendingRef.current) {
      conversationTurnPendingRef.current = nextTurnPending;
      setConversationTurnPending(nextTurnPending);
    }

    if (autoTitleCandidate) {
      maybeApplyOptimisticConversationTitle(autoTitleCandidate);
    }
    if (queuedNextMessage) {
      void submitConversationMessage(queuedNextMessage.text);
    }
    if (shouldRefreshConversation) {
      scheduleConversationRefresh();
    }
  });

  const queueConversationStreamEvent = useEffectEvent((event: ConversationStreamEvent) => {
    pendingConversationStreamEventsRef.current = [
      ...pendingConversationStreamEventsRef.current,
      event,
    ];
    if (conversationStreamFrameRef.current !== null) {
      return;
    }
    conversationStreamFrameRef.current = window.requestAnimationFrame(() => {
      flushConversationStreamBatch();
    });
  });

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (conversationRefreshTimerRef.current !== null) {
        window.clearTimeout(conversationRefreshTimerRef.current);
      }
      if (conversationStreamFrameRef.current !== null) {
        window.cancelAnimationFrame(conversationStreamFrameRef.current);
        conversationStreamFrameRef.current = null;
      }
      pendingConversationStreamEventsRef.current = [];
      turnTimeoutCleanupRef.current?.();
      turnTimeoutCleanupRef.current = null;
    };
  }, []);

  // Load conversation detail
  useEffect(() => {
    if (!selectedConversationId) {
      setConversationDetail(null);
      setLoadingConversationDetail(false);
      setConversationLoadError("");
      setConversationMessages([]);
      setConversationEvents([]);
      setOptimisticConversationEvents([]);
      setConversationStatus(null);
      conversationReplayHydrationTokenRef.current += 1;
      conversationStreamAfterSeqRef.current = 0;
      conversationStreamMaxSeenSeqRef.current = 0;
      conversationStreamActivityAtRef.current = 0;
      pendingConversationStreamEventsRef.current = [];
      if (conversationStreamFrameRef.current !== null) {
        window.cancelAnimationFrame(conversationStreamFrameRef.current);
        conversationStreamFrameRef.current = null;
      }
      setConversationStreamReady(false);
      setConversationTurnPending(false);
      setConversationStreaming(false);
      setHasOlderMessages(false);
      setLoadingOlderMessages(false);
      setConversationHistoryQuery("");
      setActiveConversationMatchIndex(0);
      autoTitledRef.current = false;
      return;
    }
    autoTitledRef.current = false;
    pendingConversationTitleRef.current = null;
    const hasMatchingConversationDetail =
      conversationDetailRef.current?.id === selectedConversationId;
    if (connectionState !== "connected") {
      if (!hasMatchingConversationDetail) {
        setConversationDetail(null);
        setConversationMessages([]);
        setConversationEvents([]);
        setOptimisticConversationEvents([]);
        setConversationStatus(null);
        setLoadingOlderMessages(false);
        setConversationHistoryQuery("");
        setActiveConversationMatchIndex(0);
      }
      setLoadingConversationDetail(!hasMatchingConversationDetail);
      conversationReplayHydrationTokenRef.current += 1;
      conversationStreamAfterSeqRef.current = 0;
      conversationStreamMaxSeenSeqRef.current = 0;
      conversationStreamActivityAtRef.current = 0;
      pendingConversationStreamEventsRef.current = [];
      if (conversationStreamFrameRef.current !== null) {
        window.cancelAnimationFrame(conversationStreamFrameRef.current);
        conversationStreamFrameRef.current = null;
      }
      setConversationStreamReady(false);
      setConversationStreaming(false);
      return;
    }
    conversationReplayHydrationTokenRef.current += 1;
    setLoadingConversationDetail(true);
    setConversationLoadError("");
    setOptimisticConversationEvents((current) =>
      current.filter((event) => event.session_id === selectedConversationId),
    );
    conversationStreamAfterSeqRef.current = 0;
    conversationStreamMaxSeenSeqRef.current = 0;
    conversationStreamActivityAtRef.current = 0;
    pendingConversationStreamEventsRef.current = [];
    if (conversationStreamFrameRef.current !== null) {
      window.cancelAnimationFrame(conversationStreamFrameRef.current);
      conversationStreamFrameRef.current = null;
    }
    setConversationStreamReady(false);
    if (!hasMatchingConversationDetail) {
      setConversationDetail(null);
      setConversationMessages([]);
      setConversationEvents([]);
      setConversationStatus(null);
      setLoadingOlderMessages(false);
      setConversationHistoryQuery("");
      setActiveConversationMatchIndex(0);
    }
    let cancelled = false;

    void (async () => {
      try {
        const [detailResult, statusResult] = await Promise.allSettled([
          fetchConversationDetail(selectedConversationId),
          fetchConversationStatus(selectedConversationId),
        ]);
        const [messagesResult, eventsResult] = await Promise.allSettled([
          fetchConversationMessages(selectedConversationId, {
            latest: true,
            limit: MESSAGE_PAGE_SIZE,
          }),
          fetchConversationEvents(selectedConversationId, {
            limit: EVENT_PAGE_SIZE,
          }),
        ]);
        const detail = settledValue(detailResult, null);
        const status = settledValue(statusResult, null);
        const messages = settledValue(messagesResult, [] as ConversationMessage[]);
        const events = settledValue(eventsResult, [] as ConversationStreamEvent[]);
        if (detail == null && status == null && messages.length === 0 && events.length === 0) {
          throw firstSettledError([detailResult, statusResult, messagesResult, eventsResult])
            || new Error("Failed to load conversation.");
        }
        if (!cancelled) {
          if (detail != null) {
            applyIncomingConversationDetail(detail);
          }
          setLoadingConversationDetail(false);
          setConversationLoadError("");
          setConversationMessages(messages);
          setConversationEvents(events);
          setOptimisticConversationEvents((current) =>
            reconcileOptimisticConversationEvents(
              current.filter((event) => event.session_id === selectedConversationId),
              events,
            ),
          );
          if (status != null) {
            setConversationStatus(status);
          }
          conversationStreamAfterSeqRef.current = durableConversationSeq(events);
          conversationStreamMaxSeenSeqRef.current = Math.max(
            conversationStreamAfterSeqRef.current,
            ...events.map((event) => event.seq),
          );
          conversationStreamActivityAtRef.current = Date.now();
          setConversationStreamReady(true);
          const hasOlder = hasOlderConversationHistory(messages, events);
          setHasOlderMessages(hasOlder);
          const isActive = Boolean(
            status?.processing
            && !status.awaiting_user_input
            && !status.awaiting_approval,
          );
          if (detail != null) {
            syncConversationSummary(detail, {
              processing: isActive,
              workspaceId: selectedWorkspaceId,
            });
          } else {
            setConversationProcessing(selectedConversationId, isActive, {
              workspaceId: selectedWorkspaceId,
            });
          }
          setConversationTurnPending(isActive);
          setConversationStreaming(false);
          if (hasOlder) {
            void hydrateConversationReplay({
              conversationId: selectedConversationId,
              seedMessages: messages,
              seedEvents: events,
            });
          }
        }
      } catch (err) {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : "Failed to load conversation.";
          setError(message);
          setLoadingConversationDetail(false);
          setConversationLoadError(message);
          if (!hasMatchingConversationDetail) {
            setConversationDetail(null);
            setConversationMessages([]);
            setConversationEvents([]);
            setOptimisticConversationEvents([]);
            setConversationStatus(null);
          }
          setConversationStreamReady(false);
          setLoadingOlderMessages(false);
          if (
            selectedWorkspaceId
            && /404|conversation not found/i.test(message)
          ) {
            void refreshWorkspaceSurface(selectedWorkspaceId).catch(() => {});
          }
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [connectionState, selectedConversationId, selectedWorkspaceId]);

  // Conversation stream subscription
  useEffect(() => {
    if (connectionState !== "connected") {
      setConversationStreaming(false);
      setStreamingText("");
      setStreamingThinking("");
      setStreamingToolCalls([]);
      setLastTurnStats(null);
      return;
    }
    if (!selectedConversationId || !conversationStreamReady) {
      setConversationStreaming(false);
      setStreamingText(""); setStreamingThinking("");
      setStreamingToolCalls([]);
      setLastTurnStats(null);
      return;
    }
    // Don't assume streaming — let events determine the actual state
    setStreamingText(""); setStreamingThinking("");
    setStreamingToolCalls([]);
    setLastTurnStats(null);
    streamingTextRef.current = "";
    streamingThinkingRef.current = "";
    streamingToolCallsRef.current = [];
    lastTurnStatsRef.current = null;
    conversationStreamMaxSeenSeqRef.current = Math.max(
      conversationStreamMaxSeenSeqRef.current,
      conversationStreamAfterSeqRef.current,
    );

    const cleanup = subscribeConversationStream(
      selectedConversationId,
      (event) => {
        queueConversationStreamEvent(event);
      },
      () => {
        // SSE errors are transient — EventSource auto-reconnects.
        // Don't clear streaming state; just schedule a refresh to sync.
        conversationStreamActivityAtRef.current = 0;
        scheduleConversationRefresh();
      },
      { afterSeq: conversationStreamAfterSeqRef.current },
    );
    return () => {
      if (conversationStreamFrameRef.current !== null) {
        window.cancelAnimationFrame(conversationStreamFrameRef.current);
        conversationStreamFrameRef.current = null;
      }
      pendingConversationStreamEventsRef.current = [];
      setConversationStreaming(false);
      cleanup();
    };
  }, [
    connectionState,
    conversationStreamReady,
    selectedConversationId,
  ]);

  useEffect(() => {
    if (connectionState !== "connected") {
      return;
    }
    if (!selectedConversationId || !conversationStreamReady) {
      return;
    }
    const shouldSync = shouldContinuouslySyncConversation({
      localProcessing: conversationIsProcessing,
      turnPending: conversationTurnPending,
      streaming: conversationStreaming,
      serverReportedActive: selectedConversationServerActive,
    });
    if (!shouldSync) {
      return;
    }

    const maybeSync = () => {
      if (isConversationStreamHealthy(conversationStreamActivityAtRef.current)) {
        return;
      }
      void syncConversationLiveState(selectedConversationId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh conversation.");
        }
      });
    };

    maybeSync();

    const timer = window.setInterval(() => {
      maybeSync();
    }, CONVERSATION_SYNC_INTERVAL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, [
    connectionState,
    conversationIsProcessing,
    conversationStreaming,
    conversationStreamReady,
    conversationTurnPending,
    selectedConversationId,
    selectedConversationServerActive,
    setError,
  ]);

  // Conversation match scroll tracking
  useEffect(() => {
    conversationMatchRefs.current = [];
    if (!conversationHistoryQuery.trim() || totalConversationMatches === 0) {
      setActiveConversationMatchIndex(0);
      return;
    }
    setActiveConversationMatchIndex(0);
    window.setTimeout(() => {
      scrollConversationMatchIntoView(0);
    }, 0);
  }, [conversationHistoryQuery, totalConversationMatches]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  async function handleCreateConversation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedWorkspaceId) {
      setError("Select a workspace before starting a conversation.");
      return;
    }
    setCreatingConversation(true);
    setError("");
    setNotice("");
    try {
      const created = await createConversation(selectedWorkspaceId, {
        model_name: newConversationModel.trim(),
        system_prompt: newConversationPrompt.trim(),
      });
      syncConversationSummary(created, {
        incrementCount: true,
        processing: false,
        workspaceId: selectedWorkspaceId,
      });
      startTransition(() => {
        setSelectedConversationId(created.id);
      });
      setNewConversationPrompt("");
      setNotice(`Started conversation ${created.title}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create conversation.");
    } finally {
      setCreatingConversation(false);
    }
  }

  async function handleSendConversationMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await submitConversationMessage(conversationComposerMessage);
  }

  async function submitConversationMessage(
    rawMessage: string,
    attachments?: ConversationMessageAttachments,
  ): Promise<boolean> {
    if (!selectedConversationId) {
      setError("Select a conversation before sending a message.");
      return false;
    }
    const message = rawMessage.trim();
    const hasAttachments = hasConversationAttachments(attachments);
    if (!message && !hasAttachments) {
      setError("Message or attachment is required.");
      return false;
    }
    // Check the live status from the server to avoid stale local state
    // causing messages to go to the inject path when they shouldn't.
    let isProcessing = Boolean(
      conversationStatus?.processing || conversationTurnPending,
    );
    if (isProcessing) {
      try {
        const freshStatus = await fetchConversationStatus(selectedConversationId);
        isProcessing = Boolean(freshStatus.processing);
        if (!isProcessing) {
          // Local state was stale — fix it
          setConversationTurnPending(false);
          setConversationStatus(freshStatus);
        }
      } catch {
        // If status check fails, trust local state
      }
    }
    if (isProcessing && hasAttachments) {
      setError("Attachments can only be sent when the current turn is idle.");
      return false;
    }
    setSendingConversationMessage(true);
    setError("");
    setNotice("");
    let optimisticClientId = "";
    try {
      if (isProcessing) {
        // Conversation is active — queue for next turn (user can inject/redirect manually)
        const queueId = `q-${Date.now()}`;
        setQueuedMessages((current) => [...current, { id: queueId, text: message, queuedAt: Date.now(), type: "next" as const }]);
        setConversationComposerMessage("");
        setNotice("Message queued — will be sent when current turn completes.");
      } else {
        // Conversation is idle — send as a new turn
        // Optimistically add the user message to the visible list immediately
        const createdAt = new Date().toISOString();
        const clientId = `local-user-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        optimisticClientId = clientId;
        const attachmentPayload = conversationAttachmentPayload(attachments);
        setOptimisticConversationEvents((current) => [
          ...current,
          {
            id: -Date.now(),
            session_id: selectedConversationId,
            seq: -(current.length + 1),
            event_type: "user_message",
            payload: { text: message, ...attachmentPayload },
            payload_parse_error: false,
            created_at: createdAt,
            _optimistic: true,
            _client_id: clientId,
            _delivery_state: "sending",
          },
          ...(hasAttachments
            ? [{
                id: -(Date.now() + 1),
                session_id: selectedConversationId,
                seq: -(current.length + 2),
                event_type: "content_indicator" as const,
                payload: attachmentPayload,
                payload_parse_error: false,
                created_at: createdAt,
                _optimistic: true,
                _client_id: `${clientId}:attachments`,
              }]
            : []),
        ]);
        setConversationComposerMessage("");
        setConversationTurnPending(true);
        setConversationStatus((current) => ({
          conversation_id: current?.conversation_id || selectedConversationId,
          processing: true,
          stop_requested: false,
          pending_inject_count: 0,
          awaiting_approval: false,
          pending_approval: null,
          awaiting_user_input: false,
          pending_prompt: null,
        }));
        setConversationProcessing(selectedConversationId, true, {
          lastActiveAt: createdAt,
          workspaceId: selectedWorkspaceId,
        });
        await sendConversationMessage(
          selectedConversationId,
          message,
          "user",
          attachments,
        );
        setOptimisticConversationEvents((current) => current.map((event) => (
          event._client_id === optimisticClientId || event._client_id === `${optimisticClientId}:attachments`
            ? { ...event, _delivery_state: "accepted" }
            : event
        )));
        if (message) {
          maybeApplyOptimisticConversationTitle(message);
        }

        // Safety timeout: if no assistant-side progress arrives within 120s,
        // reset the pending state so the UI doesn't freeze forever.
        const turnTimeoutId = window.setTimeout(() => {
          setConversationTurnPending((current) => {
            if (current) {
              setError("Thread timed out waiting for a response. Try sending again.");
              setOptimisticConversationEvents((events) => events.map((event) => (
                event._client_id === optimisticClientId
                  ? { ...event, _delivery_state: "failed" }
                  : event
              )));
              setConversationStatus((s) => s ? { ...s, processing: false } : s);
            }
            return false;
          });
        }, 120_000);

        // Clear the timeout once the turn starts producing assistant-side progress.
        const clearTurnTimeout = () => window.clearTimeout(turnTimeoutId);
        // Progress events set conversationTurnPending to false, which makes this
        // timeout a no-op even if the cleanup is not called first.
        // But we store the cleanup so it can be cleared on unmount.
        turnTimeoutCleanupRef.current = clearTurnTimeout;
      }
      return true;
    } catch (err) {
      if (!isProcessing) {
        setOptimisticConversationEvents((current) =>
          current.filter((event) =>
            event._client_id !== optimisticClientId
            && event._client_id !== `${optimisticClientId}:attachments`,
          ),
        );
        setConversationTurnPending(false);
        setConversationStatus((current) => ({
          conversation_id: current?.conversation_id || selectedConversationId,
          processing: false,
          stop_requested: current?.stop_requested || false,
          pending_inject_count: current?.pending_inject_count || 0,
          awaiting_approval: current?.awaiting_approval || false,
          pending_approval: current?.pending_approval || null,
          awaiting_user_input: current?.awaiting_user_input || false,
          pending_prompt: current?.pending_prompt || null,
        }));
        setConversationProcessing(selectedConversationId, false, {
          workspaceId: selectedWorkspaceId,
        });
      }
      setError(
        err instanceof Error
          ? err.message
          : "Failed to send message.",
      );
      return false;
    } finally {
      setSendingConversationMessage(false);
    }
  }

  async function handleQuickConversationReply(optionLabel: string) {
    await submitConversationMessage(optionLabel);
  }

  async function handleInjectConversationInstruction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedConversationId) {
      setError("Select a conversation before injecting instructions.");
      return;
    }
    const instruction = conversationInjectMessage.trim();
    if (!instruction) {
      setError("Instruction is required.");
      return;
    }
    setSendingConversationInject(true);
    setError("");
    setNotice("");
    try {
      const queueId = `q-${Date.now()}`;
      setQueuedMessages((current) => [...current, { id: queueId, text: instruction, queuedAt: Date.now(), type: "inject" }]);
      const response = await injectConversationInstruction(
        selectedConversationId,
        instruction,
      );
      setConversationInjectMessage("");
      setConversationStatus((current) => ({
        conversation_id: current?.conversation_id || selectedConversationId,
        processing: current?.processing ?? true,
        stop_requested: current?.stop_requested || false,
        pending_inject_count: Number(
          (response as { pending_inject_count?: unknown }).pending_inject_count
            || current?.pending_inject_count
            || 0,
        ),
        awaiting_approval: current?.awaiting_approval || false,
        pending_approval: current?.pending_approval || null,
        awaiting_user_input: current?.awaiting_user_input || false,
        pending_prompt: current?.pending_prompt || null,
      }));
      
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to inject conversation instruction.",
      );
    } finally {
      setSendingConversationInject(false);
    }
  }

  function editQueuedMessage(queueId: string) {
    setQueuedMessages((current) => {
      const item = current.find((m) => m.id === queueId);
      if (item) {
        setConversationComposerMessage(item.text);
      }
      return current.filter((m) => m.id !== queueId);
    });
  }

  function cancelQueuedMessage(queueId: string) {
    setQueuedMessages((current) => current.filter((m) => m.id !== queueId));
  }

  async function handleResolveConversationApproval(
    decision: "approve" | "approve_all" | "deny",
  ) {
    const pendingApproval = conversationStatus?.pending_approval;
    if (!selectedConversationId || !pendingApproval?.approval_id) {
      setError("No pending approval is available.");
      return;
    }
    setError("");
    setNotice("");
    try {
      const response = await resolveConversationApproval(
        selectedConversationId,
        pendingApproval.approval_id,
        decision,
      );
      setConversationStatus((current) => ({
        conversation_id: current?.conversation_id || selectedConversationId,
        processing: true,
        stop_requested: current?.stop_requested || false,
        pending_inject_count: current?.pending_inject_count || 0,
        awaiting_approval: false,
        pending_approval: null,
        awaiting_user_input: false,
        pending_prompt: null,
      }));
      setConversationTurnPending(true);
      removeApprovalItem(
        `conversation:${selectedConversationId}:${pendingApproval.approval_id}`,
        selectedWorkspaceId,
      );
      setConversationProcessing(selectedConversationId, true, {
        workspaceId: selectedWorkspaceId,
      });
      
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to resolve conversation approval.",
      );
    }
  }

  async function handleStopConversationTurn() {
    if (!selectedConversationId) {
      setError("Select a conversation before stopping it.");
      return;
    }
    setError("");
    setNotice("");
    try {
      const response = await stopConversationTurn(selectedConversationId);
      setConversationStatus((current) => ({
        conversation_id: current?.conversation_id || selectedConversationId,
        processing: true,
        stop_requested: true,
        pending_inject_count: current?.pending_inject_count || 0,
        awaiting_approval: false,
        pending_approval: null,
        awaiting_user_input: false,
        pending_prompt: null,
      }));
      
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to stop conversation.");
    }
  }

  function handlePrefillStarterConversation() {
    setNewConversationPrompt(
      "Help me understand this workspace, summarize the architecture, and suggest the best first improvement.",
    );
    
    setError("");
  }

  function focusConversationComposer() {
    setActiveTab("threads");
    const ref = conversationComposerRef;
    // scrollToSection inlined here to avoid circular import
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    window.setTimeout(() => {
      const target = ref.current?.querySelector("textarea");
      if (target instanceof HTMLTextAreaElement) {
        target.focus();
      }
    }, 140);
  }

  function scrollConversationMatchIntoView(index: number) {
    const target = conversationMatchRefs.current[index];
    target?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function stepConversationMatch(delta: number) {
    if (!conversationHistoryQuery.trim() || totalConversationMatches === 0) {
      return;
    }
    const nextIndex = (activeConversationMatchIndex + delta + totalConversationMatches) % totalConversationMatches;
    setActiveConversationMatchIndex(nextIndex);
    window.setTimeout(() => {
      scrollConversationMatchIntoView(nextIndex);
    }, 0);
  }

  return {
    // State
    selectedConversationId,
    conversationDetail,
    loadingConversationDetail,
    conversationLoadError,
    conversationMessages,
    conversationEvents,
    conversationStatus,
    conversationStreaming,
    streamingText,
    streamingThinking,
    streamingToolCalls,
    lastTurnStats,
    queuedMessages,
    newConversationModel,
    newConversationPrompt,
    creatingConversation,
    conversationComposerMessage,
    sendingConversationMessage,
    conversationInjectMessage,
    sendingConversationInject,
    conversationTurnPending,
    conversationHistoryQuery,
    activeConversationMatchIndex,

    // Computed
    conversationIsProcessing,
    pendingConversationApproval,
    conversationAwaitingApproval,
    pendingConversationPrompt,
    conversationAwaitingInput,
    conversationPhaseLabel,
    quickReplyOptions,
    selectedConversationSummary,
    filteredConversationEvents,
    filteredConversationMessages,
    visibleConversationEvents,
    visibleConversationMessages,
    totalConversationMatches,
    selectedConversationRunIds,
    hasOlderMessages,
    loadingOlderMessages,

    // Refs
    conversationComposerRef,
    conversationMatchRefs,

    // Actions
    setSelectedConversationId,
    setNewConversationModel,
    setNewConversationPrompt,
    setConversationComposerMessage,
    setConversationInjectMessage,
    setQueuedMessages,
    editQueuedMessage,
    cancelQueuedMessage,
    setConversationHistoryQuery,
    setActiveConversationMatchIndex,
    handleCreateConversation,
    handleSendConversationMessage,
    submitConversationMessage,
    handleQuickConversationReply,
    handleInjectConversationInstruction,
    handleResolveConversationApproval,
    handleStopConversationTurn,
    handlePrefillStarterConversation,
    focusConversationComposer,
    refreshConversation,
    retryConversationLoad,
    loadOlderMessages,
    scrollConversationMatchIntoView,
    stepConversationMatch,
  };
}
