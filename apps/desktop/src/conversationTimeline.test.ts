import { describe, expect, it } from "vitest";

import {
  appendConversationTimelineItems,
  buildConversationMessageTimelineItems,
  buildHistoricalConversationTimelineItems,
  buildConversationMessageFallbackItems,
  buildConversationTimelineItems,
  historicalConversationTimelineCoversLiveTail,
  buildConversationTimelineWindow,
  canUseDeferredConversationTranscript,
  shouldDeferConversationTranscript,
} from "./conversationTimeline";

function makeEvent(seq: number, eventType: string, payload: Record<string, unknown>) {
  return {
    id: seq,
    session_id: "conversation-1",
    seq,
    event_type: eventType,
    payload,
    payload_parse_error: false,
    created_at: `2026-03-29T00:${String(seq).padStart(2, "0")}:00Z`,
  };
}

function makeMessage(
  turnNumber: number,
  role: "user" | "assistant",
  content: string,
  createdAt: string,
) {
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
    created_at: createdAt,
  };
}

describe("conversationTimeline", () => {
  it("collapses contiguous transcript text and pairs tool lifecycle rows", () => {
    const items = buildConversationTimelineItems([
      makeEvent(1, "user_message", { text: "hello " }),
      makeEvent(2, "user_message", { text: "there" }),
      makeEvent(3, "assistant_thinking", { text: "hidden" }),
      makeEvent(4, "assistant_text", { text: "hi " }),
      makeEvent(5, "assistant_text", { text: "friend" }),
      makeEvent(6, "tool_call_started", {
        tool_name: "search",
        tool_call_id: "tool-1",
      }),
      makeEvent(7, "tool_call_completed", {
        tool_name: "search",
        tool_call_id: "tool-1",
        success: true,
      }),
    ]);

    expect(items).toHaveLength(3);
    expect(items[0]).toMatchObject({ kind: "text", text: "hello there" });
    expect(items[1]).toMatchObject({ kind: "text", text: "hi friend" });
    expect(items[2]).toMatchObject({
      kind: "tool",
      startedPayload: { tool_name: "search", tool_call_id: "tool-1" },
      completedPayload: { tool_name: "search", tool_call_id: "tool-1", success: true },
    });
  });

  it("can incrementally append new events without rebuilding prior timeline rows", () => {
    const baseEvents = [
      makeEvent(1, "user_message", { text: "hello " }),
      makeEvent(2, "user_message", { text: "there" }),
      makeEvent(3, "assistant_text", { text: "hi " }),
      makeEvent(4, "assistant_text", { text: "friend" }),
      makeEvent(5, "tool_call_started", {
        tool_name: "search",
        tool_call_id: "tool-1",
      }),
    ];
    const appendedEvents = [
      makeEvent(6, "tool_call_completed", {
        tool_name: "search",
        tool_call_id: "tool-1",
        success: true,
      }),
      makeEvent(7, "assistant_text", { text: " done" }),
    ];

    const baseItems = buildConversationTimelineItems(baseEvents);
    const incrementalItems = appendConversationTimelineItems(baseItems, appendedEvents);
    const rebuiltItems = buildConversationTimelineItems([...baseEvents, ...appendedEvents]);

    expect(incrementalItems).toEqual(rebuiltItems);
    expect(incrementalItems[0]).toBe(baseItems[0]);
    expect(incrementalItems[1]).toBe(baseItems[1]);
    expect(incrementalItems[2]).toBe(baseItems[2]);
  });

  it("archives older timeline rows while keeping the latest window visible", () => {
    const items = Array.from({ length: 300 }, (_, index) => ({
      kind: "event" as const,
      id: `event-${index + 1}`,
      seq: index + 1,
      event: makeEvent(index + 1, "turn_separator", { tokens: index + 1, tool_count: 0 }),
    }));

    const initialWindow = buildConversationTimelineWindow(items);
    expect(initialWindow.archivedCount).toBe(80);
    expect(initialWindow.nextRevealCount).toBe(80);
    expect(initialWindow.renderedItems).toHaveLength(220);
    expect(initialWindow.renderedItems[0]?.id).toBe("event-81");

    const expandedWindow = buildConversationTimelineWindow(items, {
      archivedVisibleCount: 80,
    });
    expect(expandedWindow.archivedCount).toBe(0);
    expect(expandedWindow.renderedItems).toHaveLength(300);
  });

  it("builds a visible fallback transcript from persisted messages when event rows are not usable yet", () => {
    const items = buildConversationMessageFallbackItems([
      {
        id: 1,
        session_id: "conversation-1",
        turn_number: 1,
        role: "user",
        content: "hello",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 1,
        created_at: "2026-03-29T00:00:01Z",
      },
      {
        id: 2,
        session_id: "conversation-1",
        turn_number: 2,
        role: "assistant",
        content: "hi there",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 2,
        created_at: "2026-03-29T00:00:02Z",
      },
      {
        id: 3,
        session_id: "conversation-1",
        turn_number: 3,
        role: "tool",
        content: "{\"success\":true}",
        tool_calls: [],
        tool_call_id: "tool-1",
        tool_name: "search",
        token_count: 1,
        created_at: "2026-03-29T00:00:03Z",
      },
    ]);

    expect(items).toHaveLength(2);
    expect(items[0]).toMatchObject({ kind: "text", role: "user", text: "hello" });
    expect(items[1]).toMatchObject({ kind: "text", role: "assistant", text: "hi there" });
  });

  it("filters internal tool-call placeholder text from assistant transcript rows", () => {
    const timelineItems = buildConversationTimelineItems([
      makeEvent(1, "assistant_text", { text: "Tool call context omitted." }),
      makeEvent(2, "assistant_text", { text: "Real answer" }),
    ]);
    const fallbackItems = buildConversationMessageFallbackItems([
      {
        id: 1,
        session_id: "conversation-1",
        turn_number: 1,
        role: "assistant",
        content: "Tool call context omitted.",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 1,
        created_at: "2026-03-29T00:00:01Z",
      },
      {
        id: 2,
        session_id: "conversation-1",
        turn_number: 2,
        role: "assistant",
        content: "Actual reply",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 2,
        created_at: "2026-03-29T00:00:02Z",
      },
    ]);

    expect(timelineItems).toHaveLength(1);
    expect(timelineItems[0]).toMatchObject({ kind: "text", text: "Real answer" });
    expect(fallbackItems).toHaveLength(1);
    expect(fallbackItems[0]).toMatchObject({ kind: "text", text: "Actual reply" });
  });

  it("rebuilds persisted tool activity inline from message order instead of appending it to the end", () => {
    const items = buildHistoricalConversationTimelineItems(
      [
        makeMessage(1, "user", "Inspect the workspace", "2026-03-29T12:00:00Z"),
        {
          id: 2,
          session_id: "conversation-1",
          turn_number: 2,
          role: "assistant",
          content: "Checking the files now.",
          tool_calls: [
            {
              id: "tool-1",
              type: "function",
              function: {
                name: "list_directory",
                arguments: "{\"path\":\".\"}",
              },
            },
          ],
          tool_call_id: null,
          tool_name: null,
          token_count: 2,
          created_at: "2026-03-29T12:00:01Z",
        },
        {
          id: 3,
          session_id: "conversation-1",
          turn_number: 3,
          role: "tool",
          content: "{\"success\":true,\"output\":\"src\\nREADME.md\"}",
          tool_calls: [],
          tool_call_id: "tool-1",
          tool_name: "list_directory",
          token_count: 4,
          created_at: "2026-03-29T12:00:02Z",
        },
        makeMessage(4, "assistant", "Here is what I found.", "2026-03-29T12:00:03Z"),
      ],
      [
        makeEvent(1, "turn_separator", { tokens: 25, tool_count: 1 }),
      ],
    );

    expect(items).toHaveLength(5);
    expect(items[0]).toMatchObject({ kind: "text", text: "Inspect the workspace" });
    expect(items[1]).toMatchObject({ kind: "text", text: "Checking the files now." });
    expect(items[2]).toMatchObject({
      kind: "tool",
      startedPayload: expect.objectContaining({
        tool_name: "list_directory",
        tool_call_id: "tool-1",
      }),
      completedPayload: expect.objectContaining({
        tool_name: "list_directory",
        tool_call_id: "tool-1",
        success: true,
      }),
    });
    expect(items[3]).toMatchObject({ kind: "text", text: "Here is what I found." });
    expect(items[4]).toMatchObject({
      kind: "event",
      event: expect.objectContaining({ event_type: "turn_separator" }),
    });
  });

  it("anchors settled turn separators under their assistant replies instead of sorting them to the top", () => {
    const items = buildHistoricalConversationTimelineItems(
      [
        makeMessage(1, "user", "First prompt", "2026-03-29T12:00:00Z"),
        makeMessage(2, "assistant", "First reply", "2026-03-29T12:00:01Z"),
        makeMessage(3, "user", "Second prompt", "2026-03-29T12:00:02Z"),
        makeMessage(4, "assistant", "Second reply", "2026-03-29T12:00:03Z"),
      ],
      [
        makeEvent(1, "turn_separator", { tokens: 9, tool_count: 0 }),
        makeEvent(2, "turn_separator", { tokens: 99, tool_count: 0 }),
      ],
    );

    expect(items).toHaveLength(6);
    expect(items[0]).toMatchObject({ kind: "text", text: "First prompt" });
    expect(items[1]).toMatchObject({ kind: "text", text: "First reply" });
    expect(items[2]).toMatchObject({
      kind: "event",
      event: expect.objectContaining({
        event_type: "turn_separator",
        payload: expect.objectContaining({ tokens: 9 }),
      }),
    });
    expect(items[3]).toMatchObject({ kind: "text", text: "Second prompt" });
    expect(items[4]).toMatchObject({ kind: "text", text: "Second reply" });
    expect(items[5]).toMatchObject({
      kind: "event",
      event: expect.objectContaining({
        event_type: "turn_separator",
        payload: expect.objectContaining({ tokens: 99 }),
      }),
    });
  });

  it("anchors replay-only supplemental rows to the trailing settled assistant turns", () => {
    const items = buildHistoricalConversationTimelineItems(
      [
        makeMessage(1, "user", "Turn one", "2026-03-29T12:00:00Z"),
        makeMessage(2, "assistant", "Answer one", "2026-03-29T12:00:01Z"),
        makeMessage(3, "user", "Turn two", "2026-03-29T12:00:02Z"),
        makeMessage(4, "assistant", "Answer two", "2026-03-29T12:00:03Z"),
        makeMessage(5, "user", "Turn three", "2026-03-29T12:00:04Z"),
        makeMessage(6, "assistant", "Answer three", "2026-03-29T12:00:05Z"),
      ],
      [
        makeEvent(1, "turn_separator", { tokens: 42, tool_count: 0 }),
      ],
    );

    expect(items.slice(-2)).toEqual([
      expect.objectContaining({ kind: "text", text: "Answer three" }),
      expect.objectContaining({
        kind: "event",
        event: expect.objectContaining({
          event_type: "turn_separator",
          payload: expect.objectContaining({ tokens: 42 }),
        }),
      }),
    ]);
  });

  it("detects when the historical transcript covers the visible live tail", () => {
    const liveItems = buildConversationTimelineItems([
      makeEvent(1, "user_message", { text: "hello" }),
      makeEvent(2, "assistant_text", { text: "Checking..." }),
      makeEvent(3, "tool_call_started", {
        tool_name: "list_directory",
        tool_call_id: "tool-1",
      }),
      makeEvent(4, "tool_call_completed", {
        tool_name: "list_directory",
        tool_call_id: "tool-1",
        success: true,
      }),
      makeEvent(5, "assistant_text", { text: "Done." }),
    ]);
    const staleHistoricalItems = buildConversationMessageTimelineItems([
      makeMessage(1, "user", "hello", "2026-03-29T12:00:00Z"),
      makeMessage(2, "assistant", "Old answer", "2026-03-29T12:00:01Z"),
    ]);
    const freshHistoricalItems = buildConversationMessageTimelineItems([
      makeMessage(1, "user", "hello", "2026-03-29T12:00:00Z"),
      {
        id: 2,
        session_id: "conversation-1",
        turn_number: 2,
        role: "assistant",
        content: "Checking...",
        tool_calls: [
          {
            id: "tool-1",
            type: "function",
            function: {
              name: "list_directory",
              arguments: "{}",
            },
          },
        ],
        tool_call_id: null,
        tool_name: null,
        token_count: 2,
        created_at: "2026-03-29T12:00:01Z",
      },
      {
        id: 3,
        session_id: "conversation-1",
        turn_number: 3,
        role: "tool",
        content: "{\"success\":true}",
        tool_calls: [],
        tool_call_id: "tool-1",
        tool_name: "list_directory",
        token_count: 1,
        created_at: "2026-03-29T12:00:02Z",
      },
      makeMessage(4, "assistant", "Done.", "2026-03-29T12:00:03Z"),
    ]);

    expect(historicalConversationTimelineCoversLiveTail(
      liveItems,
      staleHistoricalItems,
    )).toBe(false);
    expect(historicalConversationTimelineCoversLiveTail(
      liveItems,
      freshHistoricalItems,
    )).toBe(true);
  });

  it("preserves whitespace-only assistant chunks so streamed markdown stays valid", () => {
    const items = buildConversationTimelineItems([
      makeEvent(1, "assistant_text", { text: "# Heading 1" }),
      makeEvent(2, "assistant_text", { text: "\n" }),
      makeEvent(3, "assistant_text", { text: "## Heading 2" }),
      makeEvent(4, "assistant_text", { text: "\n\n" }),
      makeEvent(5, "assistant_text", { text: "- Item 1" }),
      makeEvent(6, "assistant_text", { text: "\n" }),
      makeEvent(7, "assistant_text", { text: "  " }),
      makeEvent(8, "assistant_text", { text: "- Nested item" }),
      makeEvent(9, "assistant_text", { text: "\n" }),
      makeEvent(10, "assistant_text", { text: "> Quote line" }),
      makeEvent(11, "assistant_text", { text: "\n" }),
      makeEvent(12, "assistant_text", { text: "```js" }),
      makeEvent(13, "assistant_text", { text: "\n" }),
      makeEvent(14, "assistant_text", { text: "console.log('hi');" }),
      makeEvent(15, "assistant_text", { text: "\n" }),
      makeEvent(16, "assistant_text", { text: "```" }),
    ]);

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({
      kind: "text",
      text: "# Heading 1\n## Heading 2\n\n- Item 1\n  - Nested item\n> Quote line\n```js\nconsole.log('hi');\n```",
    });
  });

  it("does not defer the transcript source while a conversation is actively processing", () => {
    expect(shouldDeferConversationTranscript({
      isProcessing: true,
      eventCount: 1200,
      messageCount: 1200,
    })).toBe(false);
  });

  it("defers only large idle transcripts", () => {
    expect(shouldDeferConversationTranscript({
      isProcessing: false,
      eventCount: 401,
      messageCount: 0,
    })).toBe(true);

    expect(shouldDeferConversationTranscript({
      isProcessing: false,
      eventCount: 1200,
      messageCount: 1200,
      searchActive: true,
    })).toBe(false);
  });

  it("does not defer the first paint of a newly selected large transcript", () => {
    expect(shouldDeferConversationTranscript({
      isProcessing: false,
      eventCount: 1200,
      messageCount: 1200,
      selectionHydrated: false,
    })).toBe(false);

    expect(shouldDeferConversationTranscript({
      isProcessing: false,
      eventCount: 1200,
      messageCount: 1200,
      selectionHydrated: true,
    })).toBe(true);
  });

  it("does not swap a live transcript back to an empty deferred snapshot", () => {
    expect(canUseDeferredConversationTranscript({
      shouldDefer: true,
      selectedConversationId: "conversation-1",
      liveHasContent: true,
      deferredEvents: [],
      deferredMessages: [],
    })).toBe(false);
  });

  it("does not use deferred transcript data from a different conversation", () => {
    expect(canUseDeferredConversationTranscript({
      shouldDefer: true,
      selectedConversationId: "conversation-1",
      liveHasContent: true,
      deferredEvents: [
        makeEvent(1, "user_message", { text: "stale" }),
        {
          ...makeEvent(2, "assistant_text", { text: "content" }),
          session_id: "conversation-2",
        },
      ],
      deferredMessages: [],
    })).toBe(false);
  });

  it("allows deferred transcript rendering once the deferred snapshot belongs to the selected conversation", () => {
    expect(canUseDeferredConversationTranscript({
      shouldDefer: true,
      selectedConversationId: "conversation-1",
      liveHasContent: true,
      deferredEvents: [
        makeEvent(1, "user_message", { text: "hello" }),
      ],
      deferredMessages: [],
    })).toBe(true);
  });
});
