import { describe, expect, it, vi } from "vitest";

import {
  conversationEventKey,
  defaultConversationTitle,
  durableConversationSeq,
  hasOlderConversationHistory,
  hydrateConversationReplayPages,
  isConversationStreamHealthy,
  isInitialTurnProgressEvent,
  mergeConversationEvents,
  reconcilePendingConversationTitle,
  reconcileOptimisticConversationEvents,
  shouldContinuouslySyncConversation,
} from "./useConversation";

function makeMessage(turnNumber: number, role: string, content: string) {
  return {
    id: turnNumber,
    session_id: "conversation-1",
    turn_number: turnNumber,
    role,
    content,
    tool_calls: [],
    tool_call_id: null,
    tool_name: null,
    token_count: 0,
    created_at: `2026-03-29T00:00:0${turnNumber}Z`,
  };
}

function makeLegacyEvent(turnNumber: number, eventType: string, text: string) {
  return {
    id: 0,
    session_id: "conversation-1",
    seq: turnNumber * 100,
    turn_number: turnNumber,
    event_type: eventType,
    payload: { text },
    payload_parse_error: false,
    created_at: `2026-03-29T00:00:0${turnNumber}Z`,
  };
}

function makeDurableEvent(seq: number, eventType: string, text: string) {
  return {
    id: seq,
    session_id: "conversation-1",
    seq,
    event_type: eventType,
    payload: { text },
    payload_parse_error: false,
    created_at: `2026-03-29T00:00:0${seq}Z`,
  };
}

function makeLiveReplayEvent(seq: number, eventType: string, text: string) {
  return {
    id: 0,
    session_id: "conversation-1",
    seq,
    event_type: eventType,
    payload: { text },
    payload_parse_error: false,
    created_at: `2026-03-29T00:00:${String(seq).padStart(2, "0")}Z`,
  };
}

function makeOptimisticUserEvent(
  clientId: string,
  text: string,
  deliveryState: "queued" | "sending" | "accepted" | "failed" = "sending",
) {
  return {
    id: -1,
    session_id: "conversation-1",
    seq: -1,
    event_type: "user_message",
    payload: { text },
    payload_parse_error: false,
    created_at: "2026-03-29T00:00:09Z",
    _optimistic: true,
    _client_id: clientId,
    _delivery_state: deliveryState,
  };
}

describe("conversation replay helpers", () => {
  it("treats synthesized and durable rows as distinct event identities", () => {
    const synthetic = makeLegacyEvent(1, "user_message", "hello");
    const durable = makeDurableEvent(1, "user_message", "hello");

    expect(conversationEventKey(synthetic)).not.toBe(conversationEventKey(durable));
    expect(mergeConversationEvents([durable], [synthetic], "prepend")).toHaveLength(2);
  });

  it("treats live replay rows and fetched rows with the same seq as one event", () => {
    const live = makeLiveReplayEvent(10, "assistant_text", "hello");
    const durable = makeDurableEvent(10, "assistant_text", "hello");

    expect(conversationEventKey(live)).toBe(conversationEventKey(durable));
    expect(mergeConversationEvents([live], [durable], "append")).toHaveLength(1);
  });

  it("uses stable replay rows when computing the live stream cursor", () => {
    expect(durableConversationSeq([
      makeLegacyEvent(4, "assistant_text", "synthetic latest"),
      makeLiveReplayEvent(5, "assistant_text", "stream latest"),
      makeDurableEvent(3, "assistant_text", "durable latest"),
    ])).toBe(5);
  });

  it("treats assistant thinking and tool progress as a live turn response", () => {
    expect(isInitialTurnProgressEvent("assistant_thinking")).toBe(true);
    expect(isInitialTurnProgressEvent("tool_call_started")).toBe(true);
    expect(isInitialTurnProgressEvent("tool_call_completed")).toBe(true);
    expect(isInitialTurnProgressEvent("user_message")).toBe(false);
  });

  it("detects older history from either turns or durable event pages", () => {
    expect(hasOlderConversationHistory(
      [makeMessage(3, "user", "question"), makeMessage(4, "assistant", "answer")],
      [makeLegacyEvent(3, "user_message", "question")],
    )).toBe(true);

    expect(hasOlderConversationHistory(
      [makeMessage(1, "user", "hello"), makeMessage(2, "assistant", "hi")],
      [makeDurableEvent(2, "assistant_text", "hi")],
    )).toBe(true);

    expect(hasOlderConversationHistory(
      [makeMessage(1, "user", "hello"), makeMessage(2, "assistant", "hi")],
      [makeLegacyEvent(1, "user_message", "hello")],
    )).toBe(false);
  });

  it("hydrates a legacy transcript without requiring user paging", async () => {
    const fetchMessagesPage = vi.fn(async (beforeTurn: number) => {
      if (beforeTurn === 3) {
        return [
          makeMessage(1, "user", "hello"),
          makeMessage(2, "assistant", "hi"),
        ];
      }
      return [];
    });
    const fetchEventsPage = vi.fn(async (options: { beforeTurn?: number }) => {
      if (options.beforeTurn === 3) {
        return [
          makeLegacyEvent(1, "user_message", "hello"),
          makeLegacyEvent(2, "assistant_text", "hi"),
        ];
      }
      return [];
    });

    const replay = await hydrateConversationReplayPages({
      seedMessages: [
        makeMessage(3, "user", "question"),
        makeMessage(4, "assistant", "answer"),
      ],
      seedEvents: [
        makeLegacyEvent(3, "user_message", "question"),
        makeLegacyEvent(4, "assistant_text", "answer"),
      ],
      fetchMessagesPage,
      fetchEventsPage,
    });

    expect(fetchMessagesPage).toHaveBeenCalledWith(3);
    expect(fetchEventsPage).toHaveBeenCalledWith({ beforeSeq: undefined, beforeTurn: 3 });
    expect(replay.hasOlder).toBe(false);
    expect(replay.messages).toHaveLength(4);
    expect(replay.events).toHaveLength(4);
  });

  it("bridges from durable latest rows into older legacy transcript pages", async () => {
    const fetchMessagesPage = vi.fn(async (beforeTurn: number) => {
      if (beforeTurn === 3) {
        return [
          makeMessage(1, "user", "hello"),
          makeMessage(2, "assistant", "hi"),
        ];
      }
      return [];
    });
    const fetchEventsPage = vi.fn(async (options: { beforeSeq?: number; beforeTurn?: number }) => {
      if (options.beforeSeq === 1 && options.beforeTurn === 3) {
        return [
          makeLegacyEvent(1, "user_message", "hello"),
          makeLegacyEvent(2, "assistant_text", "hi"),
        ];
      }
      return [];
    });

    const replay = await hydrateConversationReplayPages({
      seedMessages: [
        makeMessage(3, "user", "question"),
        makeMessage(4, "assistant", "answer"),
      ],
      seedEvents: [
        makeDurableEvent(1, "user_message", "question"),
        makeDurableEvent(2, "assistant_text", "answer"),
      ],
      fetchMessagesPage,
      fetchEventsPage,
    });

    expect(fetchEventsPage).toHaveBeenCalledWith({ beforeSeq: 1, beforeTurn: 3 });
    expect(replay.hasOlder).toBe(false);
    expect(replay.events.map((event) => event.event_type)).toEqual([
      "user_message",
      "assistant_text",
      "user_message",
      "assistant_text",
    ]);
  });

  it("reconciles acknowledged optimistic sends without dropping queued duplicates", () => {
    const reconciled = reconcileOptimisticConversationEvents(
      [
        makeOptimisticUserEvent("queued-1", "please continue", "queued"),
        makeOptimisticUserEvent("sending-1", "please continue", "sending"),
      ],
      [makeDurableEvent(10, "user_message", "please continue")],
    );

    expect(reconciled).toEqual([
      expect.objectContaining({
        _client_id: "queued-1",
        _delivery_state: "queued",
      }),
    ]);
  });

  it("keeps live sync running when the server still reports an active thread", () => {
    expect(shouldContinuouslySyncConversation({
      localProcessing: false,
      turnPending: false,
      streaming: false,
      serverReportedActive: true,
    })).toBe(true);
  });

  it("treats recent stream activity as healthy and skips the polling fallback window", () => {
    expect(isConversationStreamHealthy(5_000, { now: 10_000 })).toBe(true);
    expect(isConversationStreamHealthy(5_000, { now: 11_500 })).toBe(false);
    expect(isConversationStreamHealthy(0, { now: 10_000 })).toBe(false);
  });

  it("preserves a pending auto-title when a stale default detail refresh arrives", () => {
    const reconciled = reconcilePendingConversationTitle(
      {
        id: "conversation-1",
        title: "Conversation abc123",
      },
      {
        conversationId: "conversation-1",
        title: "Countries with active wealth taxes",
      },
    );

    expect(defaultConversationTitle("Conversation abc123")).toBe(true);
    expect(reconciled).toEqual({
      detail: {
        id: "conversation-1",
        title: "Countries with active wealth taxes",
      },
      keepPending: true,
    });
  });
});
