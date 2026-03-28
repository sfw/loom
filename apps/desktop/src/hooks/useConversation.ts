import {
  startTransition,
  type FormEvent,
  useEffect,
  useEffectEvent,
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
  type ConversationMessage,
  type ConversationPrompt,
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
  submitConversationMessage: (rawMessage: string) => Promise<void>;
  handleQuickConversationReply: (optionLabel: string) => Promise<void>;
  handleInjectConversationInstruction: (event: FormEvent<HTMLFormElement>) => Promise<void>;
  handleResolveConversationApproval: (decision: "approve" | "approve_all" | "deny") => Promise<void>;
  handleStopConversationTurn: () => Promise<void>;
  handlePrefillStarterConversation: () => void;
  focusConversationComposer: () => void;
  refreshConversation: (conversationId: string) => Promise<void>;
  loadOlderMessages: () => Promise<void>;
  scrollConversationMatchIntoView: (index: number) => void;
  stepConversationMatch: (delta: number) => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useConversation(deps: {
  selectedConversationId: string;
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
  selectedWorkspaceId: string;
  overview: WorkspaceOverview | null;
  models: ModelInfo[];
  setError: React.Dispatch<React.SetStateAction<string>>;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
  setActiveTab: React.Dispatch<React.SetStateAction<import("../utils").ViewTab>>;
  refreshWorkspaceSurface: (workspaceId: string) => Promise<void>;
}): ConversationState & ConversationActions {
  const {
    selectedConversationId,
    setSelectedConversationId,
    selectedWorkspaceId,
    overview,
    setError,
    setNotice,
    setActiveTab,
    refreshWorkspaceSurface,
  } = deps;
  const [conversationDetail, setConversationDetail] = useState<ConversationDetail | null>(null);
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
  const conversationMessagesRef = useRef<ConversationMessage[]>([]);
  const conversationEventsRef = useRef<ConversationStreamEvent[]>([]);
  const optimisticConversationEventsRef = useRef<ConversationStreamEvent[]>([]);
  const conversationDetailRef = useRef<ConversationDetail | null>(null);
  const conversationStreamAfterSeqRef = useRef(0);
  const conversationStreamActivityAtRef = useRef(0);
  const [conversationStreamReady, setConversationStreamReady] = useState(false);

  function defaultConversationTitle(title: string | null | undefined): boolean {
    return /^Conversation [a-f0-9]{6,}$/.test(String(title || ""));
  }

  function trimmedConversationTitleFromText(text: string): string {
    const raw = String(text || "").trim();
    return raw.length > 60 ? `${raw.slice(0, 57)}...` : raw;
  }

  function maybeApplyOptimisticConversationTitle(rawText: string) {
    const detail = conversationDetailRef.current;
    if (!detail || autoTitledRef.current || !defaultConversationTitle(detail.title || "")) {
      return;
    }
    const nextTitle = trimmedConversationTitleFromText(rawText);
    if (!nextTitle) return;
    autoTitledRef.current = true;
    setConversationDetail((current) => current ? { ...current, title: nextTitle } : current);
    void patchConversation(selectedConversationId, { title: nextTitle }).then(() => {
      if (selectedWorkspaceId) {
        return refreshWorkspaceSurface(selectedWorkspaceId);
      }
      return Promise.resolve();
    }).catch(() => {
      autoTitledRef.current = false;
    });
  }

  // Pagination
  const [hasOlderMessages, setHasOlderMessages] = useState(false);
  const [loadingOlderMessages, setLoadingOlderMessages] = useState(false);
  const MESSAGE_PAGE_SIZE = 100;
  const EVENT_PAGE_SIZE = 200;

  // ---------------------------------------------------------------------------
  // Computed values
  // ---------------------------------------------------------------------------

  const conversationIsProcessing = Boolean(
    conversationStatus?.processing || conversationTurnPending,
  );
  const pendingConversationApproval = conversationStatus?.pending_approval || null;
  const conversationAwaitingApproval = Boolean(
    conversationStatus?.awaiting_approval && pendingConversationApproval,
  );
  const pendingConversationPrompt = conversationStatus?.pending_prompt || null;
  const conversationAwaitingInput = Boolean(
    conversationStatus?.awaiting_user_input && pendingConversationPrompt,
  );
  const conversationPhaseLabel = conversationAwaitingApproval
    ? "Awaiting approval"
    : conversationIsProcessing
      ? "Running"
      : conversationAwaitingInput
        ? "Awaiting input"
        : conversationStreaming
          ? "Live"
          : "Idle";
  const quickReplyOptions =
    conversationAwaitingInput
    && pendingConversationPrompt
    && pendingConversationPrompt.question_type === "single_choice"
      ? pendingConversationPrompt.options
      : [];
  const workspaceConversationRows = overview?.recent_conversations || [];
  const selectedConversationSummary =
    conversationDetail
    || workspaceConversationRows.find((conversation) => conversation.id === selectedConversationId)
    || null;
  const selectedConversationRunIds = selectedConversationSummary
    ? ("linked_run_ids" in selectedConversationSummary ? selectedConversationSummary.linked_run_ids : [])
    : [];
  const allConversationEvents = optimisticConversationEvents.length > 0
    ? [...conversationEvents, ...optimisticConversationEvents]
    : conversationEvents;
  const filteredConversationEvents = allConversationEvents.filter((event) =>
    matchesWorkspaceSearch(
      conversationHistoryQuery,
      event.event_type,
      conversationEventTitle(event),
      conversationEventDetail(event),
      event.payload,
      conversationEventPills(event).join(" "),
    ),
  );
  const filteredConversationMessages = conversationMessages.filter((message) =>
    matchesWorkspaceSearch(
      conversationHistoryQuery,
      message.role,
      message.content,
      message.tool_name,
      message.tool_call_id,
      summarizeMessage(message),
    ),
  );
  const visibleConversationEvents = conversationHistoryQuery.trim()
    ? filteredConversationEvents
    : allConversationEvents;
  const visibleConversationMessages = conversationHistoryQuery.trim()
    ? filteredConversationMessages
    : conversationMessages;
  const totalConversationMatches = filteredConversationEvents.length + filteredConversationMessages.length;

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

  // ---------------------------------------------------------------------------
  // useEffectEvent handlers
  // ---------------------------------------------------------------------------

  const refreshConversation = useEffectEvent(async (conversationId: string) => {
    const latestSeq = Math.max(0, ...conversationEventsRef.current.map((event) => event.seq));
    const [detail, events, status] = await Promise.all([
      fetchConversationDetail(conversationId),
      fetchConversationEvents(conversationId, { afterSeq: latestSeq, limit: EVENT_PAGE_SIZE }),
      fetchConversationStatus(conversationId),
    ]);
    setConversationDetail(detail);
    if (events.length > 0) {
      setConversationEvents((current) => {
        const seen = new Set(current.map((row) => row.seq));
        const appended = events.filter((row) => !seen.has(row.seq));
        return appended.length > 0 ? [...current, ...appended] : current;
      });
      const acknowledgedUserTexts = events
        .filter((event) => event.event_type === "user_message")
        .map((event) => String(event.payload.text || ""))
        .filter((text) => text.length > 0);
      if (acknowledgedUserTexts.length > 0) {
        setOptimisticConversationEvents((current) => {
          let next = current;
          for (const text of acknowledgedUserTexts) {
            const matchIndex = next.findIndex((event) =>
              event._optimistic
              && event.event_type === "user_message"
              && String(event.payload.text || "") === text
            );
            if (matchIndex >= 0) {
              next = [...next.slice(0, matchIndex), ...next.slice(matchIndex + 1)];
            }
          }
          return next;
        });
      }
      const maxFetchedSeq = Math.max(0, ...events.map((event) => event.seq));
      conversationStreamAfterSeqRef.current = Math.max(
        conversationStreamAfterSeqRef.current,
        maxFetchedSeq,
      );
    }
    setConversationStatus(status);
    // Sync turn pending state from server — clears stale "Processing" indicators
    setConversationStreaming(Boolean(status.processing));
    if (!status.processing) {
      setConversationTurnPending(false);
      setStreamingText(""); setStreamingThinking("");
      setStreamingToolCalls([]);
    }
  });

  async function loadOlderMessages() {
    if (!selectedConversationId || loadingOlderMessages || !hasOlderMessages) return;
    setLoadingOlderMessages(true);
    try {
      const oldestTurn = conversationMessages[0]?.turn_number;
      const oldestSeq = conversationEvents[0]?.seq;
      const [olderMessages, olderEvents] = await Promise.all([
        oldestTurn != null
          ? fetchConversationMessages(selectedConversationId, {
              beforeTurn: oldestTurn,
              limit: MESSAGE_PAGE_SIZE,
            })
          : Promise.resolve([]),
        oldestSeq != null
          ? fetchConversationEvents(selectedConversationId, {
              beforeSeq: oldestSeq,
              limit: EVENT_PAGE_SIZE,
            })
          : Promise.resolve([]),
      ]);

      if (olderMessages.length > 0) {
        setConversationMessages((current) => [...olderMessages, ...current]);
      }
      if (olderEvents.length > 0) {
        setConversationEvents((current) => {
          const seen = new Set(current.map((row) => row.seq));
          const prepended = olderEvents.filter((row) => !seen.has(row.seq));
          return prepended.length > 0 ? [...prepended, ...current] : current;
        });
      }

      const hasOlderTurns = olderMessages.length > 0 && Number(olderMessages[0]?.turn_number || 0) > 1;
      const hasOlderEvents = olderEvents.length > 0 && Number(olderEvents[0]?.seq || 0) > 1;
      setHasOlderMessages(hasOlderTurns || hasOlderEvents);
    } catch {
      // Silently fail — user can try scrolling up again
    } finally {
      setLoadingOlderMessages(false);
    }
  }

  const scheduleConversationRefresh = useEffectEvent(() => {
    if (conversationRefreshTimerRef.current !== null || !selectedConversationId) {
      return;
    }
    conversationRefreshTimerRef.current = window.setTimeout(() => {
      conversationRefreshTimerRef.current = null;
      void Promise.all([
        refreshConversation(selectedConversationId),
        selectedWorkspaceId ? refreshWorkspaceSurface(selectedWorkspaceId) : Promise.resolve(),
      ]).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh conversation.");
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
      if (conversationRefreshTimerRef.current !== null) {
        window.clearTimeout(conversationRefreshTimerRef.current);
      }
      turnTimeoutCleanupRef.current?.();
      turnTimeoutCleanupRef.current = null;
    };
  }, []);

  // Load conversation detail
  useEffect(() => {
    if (!selectedConversationId) {
      setConversationDetail(null);
      setConversationMessages([]);
      setConversationEvents([]);
      setOptimisticConversationEvents([]);
      setConversationStatus(null);
      conversationStreamAfterSeqRef.current = 0;
      conversationStreamActivityAtRef.current = 0;
      setConversationStreamReady(false);
      setConversationTurnPending(false);
      setConversationStreaming(false);
      setHasOlderMessages(false);
      setConversationHistoryQuery("");
      setActiveConversationMatchIndex(0);
      autoTitledRef.current = false;
      return;
    }
    autoTitledRef.current = false;
    conversationStreamAfterSeqRef.current = 0;
    conversationStreamActivityAtRef.current = 0;
    setConversationStreamReady(false);
    let cancelled = false;

    void (async () => {
      try {
        const [detail, status] = await Promise.all([
          fetchConversationDetail(selectedConversationId),
          fetchConversationStatus(selectedConversationId),
        ]);
        const [messages, events] = await Promise.all([
          fetchConversationMessages(selectedConversationId, {
            latest: true,
            limit: MESSAGE_PAGE_SIZE,
          }),
          fetchConversationEvents(selectedConversationId, {
            limit: EVENT_PAGE_SIZE,
          }),
        ]);
        if (!cancelled) {
          setConversationDetail(detail);
          setConversationMessages(messages);
          setConversationEvents(events);
          setOptimisticConversationEvents([]);
          setConversationStatus(status);
          conversationStreamAfterSeqRef.current = Math.max(
            0,
            ...events.map((event) => event.seq),
          );
          conversationStreamActivityAtRef.current = Date.now();
          setConversationStreamReady(true);
          const hasOlderTurns = messages.length > 0 && Number(messages[0]?.turn_number || 0) > 1;
          const hasOlderEvents = events.length > 0 && Number(events[0]?.seq || 0) > 1;
          setHasOlderMessages(hasOlderTurns || hasOlderEvents);
          const isActive = status.processing && !status.awaiting_user_input && !status.awaiting_approval;
          setConversationTurnPending(isActive);
          setConversationStreaming(isActive);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load conversation.");
          setConversationDetail(null);
          setConversationMessages([]);
          setConversationEvents([]);
          setOptimisticConversationEvents([]);
          setConversationStatus(null);
          setConversationStreamReady(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedConversationId]);

  // Conversation stream subscription
  useEffect(() => {
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
    // Track the highest seq seen so far to distinguish new events from replays
    let maxSeenSeq = Math.max(0, conversationStreamAfterSeqRef.current);

    const cleanup = subscribeConversationStream(
      selectedConversationId,
      (event) => {
        conversationStreamActivityAtRef.current = Date.now();
        setConversationStreaming(event.event_type !== "turn_separator" && event.event_type !== "turn_interrupted");

        const isNewEvent = event.seq > maxSeenSeq;
        maxSeenSeq = Math.max(maxSeenSeq, event.seq);
        conversationStreamAfterSeqRef.current = Math.max(
          conversationStreamAfterSeqRef.current,
          event.seq,
        );

        setConversationEvents((current) => {
          if (current.some((row) => row.seq === event.seq)) {
            return current;
          }
          return [...current, event];
        });
        if (event.event_type === "user_message") {
          const eventText = String(event.payload.text || "");
          if (eventText) {
            setOptimisticConversationEvents((current) => {
              const matchIndex = current.findIndex((candidate) =>
                candidate._optimistic
                && candidate.event_type === "user_message"
                && String(candidate.payload.text || "") === eventText
              );
              return matchIndex >= 0
                ? [...current.slice(0, matchIndex), ...current.slice(matchIndex + 1)]
                : current;
            });
          }
        }

        if (!isNewEvent) return;

        // --- Live streaming: accumulate text, thinking, and track tool calls ---
        if (event.event_type === "assistant_thinking") {
          setConversationStreaming(true);
          const text = String(event.payload.text || "");
          if (text) {
            setStreamingThinking((current) => current + text);
          }
        }
        if (event.event_type === "assistant_text") {
          setConversationStreaming(true);
          const text = String(event.payload.text || "");
          if (text) {
            setStreamingText((current) => current + text);
          }
        }
        if (event.event_type === "tool_call_started") {
          const toolName = String(event.payload.tool_name || "tool");
          const callId = String(event.payload.tool_call_id || event.payload.id || `tc-${Date.now()}`);
          const argsPreview = typeof event.payload.args === "object" && event.payload.args
            ? Object.entries(event.payload.args as Record<string, unknown>)
                .slice(0, 3)
                .map(([k, v]) => `${k}: ${typeof v === "string" ? v.slice(0, 60) : JSON.stringify(v).slice(0, 40)}`)
                .join(", ")
            : "";
          setStreamingToolCalls((current) => [
            ...current,
            { id: callId, tool_name: toolName, started_at: event.created_at, completed: false, args_preview: argsPreview },
          ]);
        }
        if (event.event_type === "tool_call_completed") {
          const callId = String(event.payload.tool_call_id || event.payload.id || "");
          const toolName = String(event.payload.tool_name || "tool");
          const success = event.payload.success !== false;
          const elapsed = typeof event.payload.elapsed_ms === "number" ? event.payload.elapsed_ms : undefined;
          const outputPreview = typeof event.payload.output === "string"
            ? event.payload.output.slice(0, 120)
            : typeof event.payload.error === "string"
              ? event.payload.error.slice(0, 120)
              : "";
          setStreamingToolCalls((current) => {
            const match = current.find((tc) => tc.id === callId || (!callId && tc.tool_name === toolName && !tc.completed));
            if (match) {
              return current.map((tc) =>
                tc === match ? { ...tc, completed: true, success, elapsed_ms: elapsed, output_preview: outputPreview } : tc
              );
            }
            // If we didn't find a matching started call, add a completed one
            return [...current, { id: callId || `tc-${Date.now()}`, tool_name: toolName, started_at: event.created_at, completed: true, success, elapsed_ms: elapsed, output_preview: outputPreview }];
          });
        }
        if (event.event_type === "turn_separator") {
          const tokens = Number(event.payload.tokens || 0);
          const toolCount = Number(event.payload.tool_count || 0);
          setLastTurnStats({ tokens, tool_count: toolCount, visible: true });
          // Clear streaming buffer on turn completion
          setConversationStreaming(false);
          setStreamingText(""); setStreamingThinking("");
          setStreamingToolCalls([]);
          // Auto-send the first queued "next" message, clear the rest
          setQueuedMessages((current) => {
            const nextMsg = current.find((m) => m.type === "next");
            if (nextMsg) {
              // Fire and forget — send as a new turn
              void submitConversationMessage(nextMsg.text);
            }
            return current.filter((m) => m !== nextMsg && m.type !== "next");
          });
        }
        if (event.event_type === "turn_interrupted") {
          setConversationStreaming(false);
          setStreamingText(""); setStreamingThinking("");
          setStreamingToolCalls([]);
          setLastTurnStats(null);
          setQueuedMessages([]);
        }
        if (event.event_type === "user_message") {
          // Clear streaming buffer for new turn — but preserve queued injects
          setStreamingText(""); setStreamingThinking("");
          setStreamingToolCalls([]);
          setLastTurnStats(null);
        }
        // Auto-title: only on the very first turn (turn_count was 0 before this turn)
        if (event.event_type === "turn_separator" && isNewEvent) {
          const detail = conversationDetailRef.current;
          const turnCount = Number(event.payload.turn_count ?? event.payload.tool_count ?? -1);
          // Only auto-title if title still matches default AND this is the first turn
          if (
            detail
            && defaultConversationTitle(detail.title || "")
            && !autoTitledRef.current
          ) {
            const firstOptimisticUser = optimisticConversationEventsRef.current.find((evt) => evt.event_type === "user_message");
            const firstUserMsg = conversationMessagesRef.current.find((m) => m.role === "user");
            const raw = String(
              firstUserMsg?.content
              || firstOptimisticUser?.payload.text
              || "",
            ).trim();
            if (raw) {
              maybeApplyOptimisticConversationTitle(raw);
            }
          }
        }
        // --- End live streaming ---

        if (
          event.event_type === "assistant_text"
          || event.event_type === "turn_separator"
          || event.event_type === "turn_interrupted"
        ) {
          setConversationTurnPending(false);
          // Clear the safety timeout — the turn is alive
          turnTimeoutCleanupRef.current?.();
          turnTimeoutCleanupRef.current = null;
        }
        if (event.event_type === "user_message") {
          setConversationStatus((current) => ({
            conversation_id: current?.conversation_id || selectedConversationId,
            processing: true,
            stop_requested: current?.stop_requested || false,
            pending_inject_count: 0,
            awaiting_approval: false,
            pending_approval: null,
            awaiting_user_input: false,
            pending_prompt: null,
          }));
        }
        if (event.event_type === "steering_instruction") {
          const pendingCount = Number(event.payload.pending_inject_count || 0);
          setConversationStatus((current) => ({
            conversation_id: current?.conversation_id || selectedConversationId,
            processing: current?.processing ?? true,
            stop_requested: current?.stop_requested || false,
            pending_inject_count: pendingCount,
            awaiting_approval: current?.awaiting_approval || false,
            pending_approval: current?.pending_approval || null,
            awaiting_user_input: current?.awaiting_user_input || false,
            pending_prompt: current?.pending_prompt || null,
          }));
          // Don't remove from visual queue here — the SSE event arrives
          // almost instantly after inject POST, causing the UI to flash.
          // Queue is cleared on turn_separator when injects are consumed.
        }
        if (event.event_type === "approval_requested") {
          const pendingApproval = event.payload as unknown as ConversationApproval;
          setConversationTurnPending(false);
          setConversationStatus((current) => ({
            conversation_id: current?.conversation_id || selectedConversationId,
            processing: true,
            stop_requested: current?.stop_requested || false,
            pending_inject_count: current?.pending_inject_count || 0,
            awaiting_approval: true,
            pending_approval: pendingApproval,
            awaiting_user_input: false,
            pending_prompt: null,
          }));
        }
        if (event.event_type === "approval_resolved") {
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
        }
        if (event.event_type === "tool_call_completed" && event.payload.tool_name === "ask_user") {
          const pendingPrompt = normalizeConversationPrompt(event.payload.question_payload);
          if (pendingPrompt) {
            setConversationTurnPending(false);
            setConversationStatus((current) => ({
              conversation_id: current?.conversation_id || selectedConversationId,
              processing: false,
              stop_requested: false,
              pending_inject_count: 0,
              awaiting_approval: false,
              pending_approval: null,
              awaiting_user_input: true,
              pending_prompt: pendingPrompt,
            }));
          }
        }
        if (event.event_type === "turn_separator" || event.event_type === "turn_interrupted") {
          setConversationStatus((current) => ({
            conversation_id: current?.conversation_id || selectedConversationId,
            processing: false,
            stop_requested: false,
            pending_inject_count: 0,
            awaiting_approval: false,
            pending_approval: null,
            awaiting_user_input: false,
            pending_prompt: null,
          }));
        }
        scheduleConversationRefresh();
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
      setConversationStreaming(false);
      cleanup();
    };
  }, [conversationStreamReady, scheduleConversationRefresh, selectedConversationId]);

  useEffect(() => {
    if (!selectedConversationId || !conversationIsProcessing) {
      return;
    }
    const timer = window.setInterval(() => {
      const staleForMs = Date.now() - conversationStreamActivityAtRef.current;
      if (conversationStreamActivityAtRef.current > 0 && staleForMs < 8000) {
        return;
      }
      void refreshConversation(selectedConversationId).catch((err) => {
        if (!isTransientRequestError(err)) {
          setError(err instanceof Error ? err.message : "Failed to refresh conversation.");
        }
      });
    }, 5000);
    return () => {
      window.clearInterval(timer);
    };
  }, [
    conversationIsProcessing,
    refreshConversation,
    selectedConversationId,
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
      await refreshWorkspaceSurface(selectedWorkspaceId);
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

  async function submitConversationMessage(rawMessage: string) {
    if (!selectedConversationId) {
      setError("Select a conversation before sending a message.");
      return;
    }
    const message = rawMessage.trim();
    if (!message) {
      setError("Message is required.");
      return;
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
    setSendingConversationMessage(true);
    setError("");
    setNotice("");
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
        setOptimisticConversationEvents((current) => [
          ...current,
          {
            id: -Date.now(),
            session_id: selectedConversationId,
            seq: -(current.length + 1),
            event_type: "user_message",
            payload: { text: message },
            payload_parse_error: false,
            created_at: createdAt,
            _optimistic: true,
            _client_id: clientId,
          },
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
        await sendConversationMessage(
          selectedConversationId,
          message,
        );
        maybeApplyOptimisticConversationTitle(message);

        // Safety timeout: if no streaming event arrives within 30s,
        // reset the pending state so the UI doesn't freeze forever.
        const turnTimeoutId = window.setTimeout(() => {
          setConversationTurnPending((current) => {
            if (current) {
              setError("Thread timed out waiting for a response. Try sending again.");
              setConversationStatus((s) => s ? { ...s, processing: false } : s);
            }
            return false;
          });
        }, 120_000);

        // Clear the timeout when streaming events start arriving
        const clearTurnTimeout = () => window.clearTimeout(turnTimeoutId);
        // The turn_separator or assistant_text event handler already sets
        // conversationTurnPending to false, which will make this timeout a no-op.
        // But we store the cleanup so it can be cleared on unmount.
        turnTimeoutCleanupRef.current = clearTurnTimeout;
      }
    } catch (err) {
      if (!isProcessing) {
        setOptimisticConversationEvents((current) =>
          current.filter((event) => !(event._optimistic && String(event.payload.text || "") === message)),
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
      }
      setError(
        err instanceof Error
          ? err.message
          : "Failed to send message.",
      );
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
    loadOlderMessages,
    scrollConversationMatchIntoView,
    stepConversationMatch,
  };
}
