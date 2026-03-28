import { describe, expect, it, vi, beforeEach } from "vitest";

import {
  subscribeConversationStream,
  subscribeNotificationsStream,
  subscribeRunStream,
} from "./api";

describe("conversation stream api", () => {
  const addEventListener = vi.fn();
  const removeEventListener = vi.fn();
  const close = vi.fn();
  const EventSourceMock = vi.fn(() => ({
    addEventListener,
    removeEventListener,
    close,
  }));

  beforeEach(() => {
    addEventListener.mockReset();
    removeEventListener.mockReset();
    close.mockReset();
    EventSourceMock.mockClear();
    vi.stubGlobal("EventSource", EventSourceMock);
  });

  it("includes the last seen sequence when subscribing", () => {
    const cleanup = subscribeConversationStream(
      "conversation-1",
      vi.fn(),
      vi.fn(),
      { afterSeq: 42 },
    );

    expect(EventSourceMock).toHaveBeenCalledTimes(1);
    const firstCall = (EventSourceMock.mock.calls as unknown as Array<Array<unknown>>)[0];
    expect(String(firstCall?.[0] || "")).toContain("after_seq=42");

    cleanup();
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("includes the last seen run event id when subscribing", () => {
    const cleanup = subscribeRunStream(
      "run-1",
      vi.fn(),
      vi.fn(),
      { afterId: 17 },
    );

    expect(EventSourceMock).toHaveBeenCalledTimes(1);
    const firstCall = (EventSourceMock.mock.calls as unknown as Array<Array<unknown>>)[0];
    expect(String(firstCall?.[0] || "")).toContain("after_id=17");

    cleanup();
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("includes the last seen notification event id when subscribing", () => {
    const cleanup = subscribeNotificationsStream(
      "workspace-1",
      vi.fn(),
      vi.fn(),
      { afterId: 9 },
    );

    expect(EventSourceMock).toHaveBeenCalledTimes(1);
    const firstCall = (EventSourceMock.mock.calls as unknown as Array<Array<unknown>>)[0];
    expect(String(firstCall?.[0] || "")).toContain("workspace_id=workspace-1");
    expect(String(firstCall?.[0] || "")).toContain("after_id=9");

    cleanup();
    expect(close).toHaveBeenCalledTimes(1);
  });
});
