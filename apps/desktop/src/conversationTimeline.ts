import type { ConversationMessage, ConversationStreamEvent } from "./api";

const TOOL_CALL_CONTEXT_PLACEHOLDERS = new Set([
  "tool call context omitted.",
  "tool call required to continue.",
]);

function stripConversationToolCallPlaceholders(text: string): string {
  const normalized = String(text || "").replace(/\r\n?/g, "\n");
  if (normalized.length === 0) {
    return "";
  }
  const stripped = normalized
    .split("\n")
    .filter((line) => !TOOL_CALL_CONTEXT_PLACEHOLDERS.has(line.trim().toLowerCase()))
    .join("\n");
  return stripped;
}

export type ConversationTimelineItem =
  | {
      kind: "text";
      id: string;
      seq: number;
      role: "user" | "assistant";
      text: string;
      createdAt: string;
      deliveryState: "queued" | "sending" | "accepted" | "failed" | null;
    }
  | {
      kind: "tool";
      id: string;
      seq: number;
      createdAt: string;
      startedPayload?: Record<string, unknown>;
      completedPayload?: Record<string, unknown>;
    }
  | {
      kind: "event";
      id: string;
      seq: number;
      event: ConversationStreamEvent;
    };

function parseMessageToolStartPayload(
  rawCall: Record<string, unknown>,
): Record<string, unknown> | null {
  const functionPayload = rawCall.function;
  if (!functionPayload || typeof functionPayload !== "object") {
    return null;
  }

  const toolName = String((functionPayload as { name?: unknown }).name || "").trim();
  if (!toolName) {
    return null;
  }

  const toolCallId = String(rawCall.id || "").trim();
  const rawArguments = (functionPayload as { arguments?: unknown }).arguments;
  let args: Record<string, unknown> = {};

  if (typeof rawArguments === "string" && rawArguments.trim()) {
    try {
      const parsed = JSON.parse(rawArguments);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        args = parsed as Record<string, unknown>;
      } else {
        args = { raw: rawArguments };
      }
    } catch {
      args = { raw: rawArguments };
    }
  } else if (rawArguments && typeof rawArguments === "object" && !Array.isArray(rawArguments)) {
    args = rawArguments as Record<string, unknown>;
  }

  return {
    tool_name: toolName,
    tool_call_id: toolCallId,
    args,
  };
}

function parseMessageToolCompletionPayload(
  message: ConversationMessage,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    tool_name: String(message.tool_name || "").trim(),
    tool_call_id: String(message.tool_call_id || "").trim(),
    success: true,
    elapsed_ms: 0,
  };

  const rawContent = String(message.content || "");
  if (!rawContent.trim()) {
    return payload;
  }

  try {
    const parsed = JSON.parse(rawContent);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const data = parsed as Record<string, unknown>;
      payload.success = data.success !== false;
      if (typeof data.output === "string") {
        payload.output = data.output;
      }
      if (typeof data.error === "string") {
        payload.error = data.error;
      }
      if (data.data && typeof data.data === "object" && !Array.isArray(data.data)) {
        payload.data = data.data as Record<string, unknown>;
      }
      if (
        data.question_payload
        && typeof data.question_payload === "object"
        && !Array.isArray(data.question_payload)
      ) {
        payload.question_payload = data.question_payload as Record<string, unknown>;
      }
      return payload;
    }
  } catch {
    // Fall back to showing the raw content as the tool output preview.
  }

  payload.output = rawContent;
  return payload;
}

export function buildConversationMessageTimelineItems(
  messages: ConversationMessage[],
): ConversationTimelineItem[] {
  const items: ConversationTimelineItem[] = [];

  for (const message of messages) {
    const role = String(message.role || "").trim().toLowerCase();
    const seqBase = Number(message.turn_number || 0) * 100;
    const createdAt = String(message.created_at || "");
    let seqOffset = 0;

    const appendTextItem = (
      itemRole: "user" | "assistant",
      rawText: string,
    ) => {
      const text = itemRole === "assistant"
        ? stripConversationToolCallPlaceholders(rawText)
        : rawText;
      if (!text || (itemRole === "assistant" && !text.trim())) {
        return;
      }
      items.push({
        kind: "text",
        id: `message-${Number(message.id || 0) > 0 ? message.id : `${message.turn_number}-${itemRole}-${createdAt}`}`,
        seq: seqBase + seqOffset,
        role: itemRole,
        text,
        createdAt,
        deliveryState: null,
      });
      seqOffset += 1;
    };

    if (role === "user" || role === "assistant") {
      appendTextItem(role, String(message.content || ""));
    }

    if (role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      for (const rawCall of message.tool_calls) {
        if (!rawCall || typeof rawCall !== "object" || Array.isArray(rawCall)) {
          continue;
        }
        const startedPayload = parseMessageToolStartPayload(rawCall as Record<string, unknown>);
        if (!startedPayload) {
          continue;
        }
        items.push({
          kind: "tool",
          id: `message-tool-start-${Number(message.id || 0) > 0 ? message.id : `${message.turn_number}`}-${seqOffset}`,
          seq: seqBase + seqOffset,
          createdAt,
          startedPayload,
        });
        seqOffset += 1;
      }
      continue;
    }

    if (role !== "tool") {
      continue;
    }

    const completedPayload = parseMessageToolCompletionPayload(message);
    const toolCallId = String(completedPayload.tool_call_id || "").trim();
    const toolName = String(completedPayload.tool_name || "").trim();
    const matchIndex = [...items]
      .reverse()
      .findIndex((candidate) => (
        candidate.kind === "tool"
        && candidate.completedPayload == null
        && (
          (toolCallId && String(candidate.startedPayload?.tool_call_id || "").trim() === toolCallId)
          || (
            !toolCallId
            && toolName
            && String(candidate.startedPayload?.tool_name || "").trim() === toolName
          )
        )
      ));

    if (matchIndex >= 0) {
      const targetIndex = items.length - 1 - matchIndex;
      const existing = items[targetIndex];
      if (existing?.kind === "tool") {
        existing.completedPayload = completedPayload;
        existing.createdAt = createdAt;
        existing.seq = seqBase + seqOffset;
        continue;
      }
    }

    items.push({
      kind: "tool",
      id: `message-tool-complete-${Number(message.id || 0) > 0 ? message.id : `${message.turn_number}`}-${seqOffset}`,
      seq: seqBase + seqOffset,
      createdAt,
      completedPayload,
    });
  }

  return items;
}

export function conversationEventRowKey(event: ConversationStreamEvent): string {
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

export function buildConversationTimelineItems(
  events: ConversationStreamEvent[],
): ConversationTimelineItem[] {
  return appendConversationTimelineItems([], events);
}

function appendConversationTimelineItem(
  items: ConversationTimelineItem[],
  event: ConversationStreamEvent,
): void {
  const eventKey = conversationEventRowKey(event);
  if (event.event_type === "user_message" || event.event_type === "assistant_text") {
    const role = event.event_type === "user_message" ? "user" : "assistant";
    const rawText = String(event.payload.text || "");
    const text = role === "assistant"
      ? stripConversationToolCallPlaceholders(rawText)
      : rawText;
    const deliveryState =
      role === "user"
        ? event._delivery_state || (event._optimistic ? "sending" : null)
        : null;
    if (!text) return;
    const previous = items[items.length - 1];
    if (
      previous?.kind === "text"
      && previous.role === role
      && previous.deliveryState === deliveryState
    ) {
      previous.text += text;
      previous.createdAt = event.created_at;
      previous.seq = event.seq;
    } else {
      items.push({
        kind: "text",
        id: `text-${eventKey}`,
        seq: event.seq,
        role,
        text,
        createdAt: event.created_at,
        deliveryState,
      });
    }
    return;
  }

  if (event.event_type === "assistant_thinking") {
    return;
  }

  if (
    event.event_type === "approval_requested"
    || event.event_type === "approval_resolved"
  ) {
    return;
  }

  if (event.event_type === "tool_call_started") {
    items.push({
      kind: "tool",
      id: `tool-${eventKey}`,
      seq: event.seq,
      createdAt: event.created_at,
      startedPayload: event.payload,
    });
    return;
  }

  if (event.event_type === "tool_call_completed") {
    const toolCallId = String(event.payload.tool_call_id || "");
    const toolName = String(event.payload.tool_name || "");
    const matchIndex = [...items]
      .reverse()
      .findIndex((candidate) => (
        candidate.kind === "tool"
        && candidate.completedPayload == null
        && (
          (toolCallId && String(candidate.startedPayload?.tool_call_id || "") === toolCallId)
          || (
            !toolCallId
            && toolName
            && String(candidate.startedPayload?.tool_name || "") === toolName
          )
        )
      ));
    if (matchIndex >= 0) {
      const targetIndex = items.length - 1 - matchIndex;
      const existing = items[targetIndex];
      if (existing?.kind === "tool") {
        existing.completedPayload = event.payload;
        existing.createdAt = event.created_at;
        existing.seq = event.seq;
        return;
      }
    }
    items.push({
      kind: "tool",
      id: `tool-${eventKey}`,
      seq: event.seq,
      createdAt: event.created_at,
      completedPayload: event.payload,
    });
    return;
  }

  items.push({
    kind: "event",
    id: `event-${eventKey}`,
    seq: event.seq,
    event,
  });
}

export function appendConversationTimelineItems(
  existingItems: ConversationTimelineItem[],
  events: ConversationStreamEvent[],
): ConversationTimelineItem[] {
  if (events.length === 0) {
    return existingItems;
  }
  const items = existingItems.slice();
  for (const event of events) {
    appendConversationTimelineItem(items, event);
  }
  return items;
}

export function buildConversationMessageFallbackItems(
  messages: ConversationMessage[],
): ConversationTimelineItem[] {
  const items: ConversationTimelineItem[] = [];

  for (const message of messages) {
    const role = String(message.role || "").trim().toLowerCase();
    if (role !== "user" && role !== "assistant") {
      continue;
    }
    const rawText = String(message.content || "");
    const text = role === "assistant"
      ? stripConversationToolCallPlaceholders(rawText)
      : rawText;
    if (!text || (role === "assistant" && !text.trim())) {
      continue;
    }
    items.push({
      kind: "text",
      id: `message-${Number(message.id || 0) > 0 ? message.id : `${message.turn_number}-${role}-${message.created_at}`}`,
      seq: Number(message.turn_number || 0) * 100,
      role,
      text,
      createdAt: String(message.created_at || ""),
      deliveryState: null,
    });
  }

  return items;
}

function buildConversationSupplementalEventItems(
  events: ConversationStreamEvent[],
): ConversationTimelineItem[] {
  const supplementalEvents = events.filter((event) => {
    if (
      event.event_type === "user_message"
      || event.event_type === "assistant_text"
      || event.event_type === "assistant_thinking"
      || event.event_type === "tool_call_started"
      || event.event_type === "tool_call_completed"
      || event.event_type === "approval_requested"
      || event.event_type === "approval_resolved"
    ) {
      return false;
    }
    return true;
  });
  return buildConversationTimelineItems(supplementalEvents);
}

function isConversationTurnBoundaryItem(item: ConversationTimelineItem): boolean {
  return item.kind === "event" && (
    item.event.event_type === "turn_separator"
    || item.event.event_type === "turn_interrupted"
  );
}

export function buildHistoricalConversationTimelineItems(
  messages: ConversationMessage[],
  events: ConversationStreamEvent[],
): ConversationTimelineItem[] {
  const messageItems = buildConversationMessageTimelineItems(messages);
  const supplementalEventItems = buildConversationSupplementalEventItems(events);
  if (messageItems.length === 0) {
    return supplementalEventItems;
  }
  if (supplementalEventItems.length === 0) {
    return messageItems;
  }

  const anchorIndices: number[] = [];
  for (let index = 0; index < messageItems.length; index += 1) {
    const item = messageItems[index];
    if (item.kind === "text" && item.role === "user" && index > 0) {
      anchorIndices.push(index - 1);
    }
  }
  if (messageItems.length > 0) {
    anchorIndices.push(messageItems.length - 1);
  }
  const candidateAnchorIndices = anchorIndices.length > 0
    ? anchorIndices
    : messageItems.map((_, index) => index);

  if (candidateAnchorIndices.length === 0) {
    return [...messageItems, ...supplementalEventItems];
  }

  const segments: ConversationTimelineItem[][] = [];
  let currentSegment: ConversationTimelineItem[] = [];
  for (const item of supplementalEventItems) {
    currentSegment.push(item);
    if (isConversationTurnBoundaryItem(item)) {
      segments.push(currentSegment);
      currentSegment = [];
    }
  }
  if (currentSegment.length > 0) {
    segments.push(currentSegment);
  }

  const supplementalItemsByAnchor = new Map<number, ConversationTimelineItem[]>();
  const anchorOffset = Math.max(0, candidateAnchorIndices.length - segments.length);
  segments.forEach((segment, index) => {
    const anchorIndex = candidateAnchorIndices[
      Math.min(anchorOffset + index, candidateAnchorIndices.length - 1)
    ];
    if (anchorIndex == null) {
      return;
    }
    const anchoredItems = supplementalItemsByAnchor.get(anchorIndex) ?? [];
    anchoredItems.push(...segment);
    supplementalItemsByAnchor.set(anchorIndex, anchoredItems);
  });

  const mergedItems: ConversationTimelineItem[] = [];
  messageItems.forEach((item, index) => {
    mergedItems.push(item);
    const anchoredItems = supplementalItemsByAnchor.get(index);
    if (anchoredItems) {
      mergedItems.push(...anchoredItems);
    }
  });

  return mergedItems;
}

type ConversationContentTailItem =
  | {
      kind: "text";
      role: "user" | "assistant";
      text: string;
    }
  | {
      kind: "tool";
      toolName: string;
      toolCallId: string;
      status: "started" | "completed" | "failed";
    };

function normalizeConversationText(text: string): string {
  return String(text || "").replace(/\s+/g, " ").trim();
}

function contentTailItem(
  item: ConversationTimelineItem,
): ConversationContentTailItem | null {
  if (item.kind === "text") {
    return {
      kind: "text",
      role: item.role,
      text: normalizeConversationText(item.text),
    };
  }
  if (item.kind === "tool") {
    return {
      kind: "tool",
      toolName: String(item.completedPayload?.tool_name || item.startedPayload?.tool_name || "").trim(),
      toolCallId: String(item.completedPayload?.tool_call_id || item.startedPayload?.tool_call_id || "").trim(),
      status: item.completedPayload == null
        ? "started"
        : item.completedPayload.success === false
          ? "failed"
          : "completed",
    };
  }
  return null;
}

export function conversationTimelineTailSignature(
  items: ConversationTimelineItem[],
  limit = 3,
): ConversationContentTailItem[] {
  const signatures: ConversationContentTailItem[] = [];

  for (let index = items.length - 1; index >= 0; index -= 1) {
    const signature = contentTailItem(items[index]!);
    if (!signature) {
      continue;
    }
    signatures.unshift(signature);
    if (signatures.length >= limit) {
      break;
    }
  }

  return signatures;
}

function contentTailItemsMatch(
  historical: ConversationContentTailItem,
  live: ConversationContentTailItem,
): boolean {
  if (historical.kind !== live.kind) {
    return false;
  }

  if (historical.kind === "text" && live.kind === "text") {
    return historical.role === live.role
      && historical.text.length > 0
      && historical.text.includes(live.text);
  }

  if (historical.kind === "tool" && live.kind === "tool") {
    const toolCallIdsMatch = historical.toolCallId && live.toolCallId
      ? historical.toolCallId === live.toolCallId
      : true;
    return historical.toolName === live.toolName
      && historical.status === live.status
      && toolCallIdsMatch;
  }

  return false;
}

export function historicalConversationTimelineCoversLiveTail(
  liveItems: ConversationTimelineItem[],
  historicalItems: ConversationTimelineItem[],
  limit = 3,
): boolean {
  const liveTail = conversationTimelineTailSignature(liveItems, limit);
  if (liveTail.length === 0) {
    return historicalItems.length > 0;
  }

  const historicalTail = conversationTimelineTailSignature(
    historicalItems,
    Math.max(limit * 2, liveTail.length),
  );
  if (historicalTail.length === 0) {
    return false;
  }

  let historicalIndex = historicalTail.length - 1;
  for (let liveIndex = liveTail.length - 1; liveIndex >= 0; liveIndex -= 1) {
    const liveItem = liveTail[liveIndex]!;
    let matched = false;
    while (historicalIndex >= 0) {
      const historicalItem = historicalTail[historicalIndex]!;
      historicalIndex -= 1;
      if (contentTailItemsMatch(historicalItem, liveItem)) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }

  return true;
}

export function estimateConversationTimelineItemHeight(
  item: ConversationTimelineItem,
): number {
  if (item.kind === "tool") {
    return item.completedPayload ? 88 : 72;
  }
  if (item.kind === "event") {
    return item.event.event_type === "turn_separator" ? 44 : 84;
  }
  const textLength = Math.max(item.text.length, 1);
  const lineEstimate = Math.ceil(textLength / 78);
  const baseHeight = item.role === "user" ? 96 : 84;
  return Math.min(520, baseHeight + lineEstimate * 24);
}

export function buildConversationTimelineWindow(
  items: ConversationTimelineItem[],
  options?: {
    archivedVisibleCount?: number;
    initialVisibleCount?: number;
    archiveChunkSize?: number;
    disableArchive?: boolean;
  },
): {
  archivedCount: number;
  nextRevealCount: number;
  renderedItems: ConversationTimelineItem[];
} {
  const initialVisibleCount = Math.max(1, options?.initialVisibleCount ?? 220);
  const archiveChunkSize = Math.max(1, options?.archiveChunkSize ?? 200);
  const archivedVisibleCount = Math.max(0, options?.archivedVisibleCount ?? 0);
  if (options?.disableArchive || items.length <= initialVisibleCount + archivedVisibleCount) {
    return {
      archivedCount: 0,
      nextRevealCount: 0,
      renderedItems: items,
    };
  }

  const visibleCount = Math.min(items.length, initialVisibleCount + archivedVisibleCount);
  const archivedCount = Math.max(0, items.length - visibleCount);

  return {
    archivedCount,
    nextRevealCount: Math.min(archiveChunkSize, archivedCount),
    renderedItems: items.slice(items.length - visibleCount),
  };
}

export function shouldDeferConversationTranscript(options: {
  isProcessing: boolean;
  eventCount: number;
  messageCount: number;
  searchActive?: boolean;
  selectionHydrated?: boolean;
}): boolean {
  if (options.isProcessing || options.searchActive || options.selectionHydrated === false) {
    return false;
  }
  return options.eventCount > 400 || options.messageCount > 400;
}

export function canUseDeferredConversationTranscript(options: {
  shouldDefer: boolean;
  selectedConversationId: string;
  liveHasContent: boolean;
  deferredEvents: ConversationStreamEvent[];
  deferredMessages: ConversationMessage[];
}): boolean {
  if (!options.shouldDefer) {
    return false;
  }

  const deferredHasContent = options.deferredEvents.length > 0 || options.deferredMessages.length > 0;
  if (options.liveHasContent && !deferredHasContent) {
    return false;
  }

  const eventsMatchConversation = options.deferredEvents.every((event) => event.session_id === options.selectedConversationId);
  const messagesMatchConversation = options.deferredMessages.every((message) => message.session_id === options.selectedConversationId);
  return eventsMatchConversation && messagesMatchConversation;
}
