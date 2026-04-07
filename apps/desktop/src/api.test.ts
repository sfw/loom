import { describe, expect, it, vi, beforeEach } from "vitest";

import {
  createTask,
  fetchConversationEvents,
  fetchConversationMessages,
  fetchRunDetail,
  fetchRuntimeStatus,
  fetchRunTimeline,
  fetchWorkspaceFiles,
  restartRun,
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

  it("includes the last seen run sequence when subscribing", () => {
    const cleanup = subscribeRunStream(
      "run-1",
      vi.fn(),
      vi.fn(),
      { afterSequence: 17, includeNoise: false },
    );

    expect(EventSourceMock).toHaveBeenCalledTimes(1);
    const firstCall = (EventSourceMock.mock.calls as unknown as Array<Array<unknown>>)[0];
    expect(String(firstCall?.[0] || "")).toContain("after_sequence=17");
    expect(String(firstCall?.[0] || "")).toContain("include_noise=false");

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

describe("request timeouts", () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  it("keeps the default 15s timeout for lightweight runtime requests", async () => {
    vi.useFakeTimers();
    let abortMessage = "";
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => new Promise((_resolve, reject) => {
      const signal = init?.signal as AbortSignal | undefined;
      signal?.addEventListener("abort", () => {
        const reason = signal.reason;
        abortMessage = String(reason instanceof Error ? reason.message : reason || "");
        reject(reason);
      }, { once: true });
    }));
    vi.stubGlobal("fetch", fetchMock);

    const request = fetchRuntimeStatus();
    const assertion = expect(request).rejects.toThrow("Request timed out after 15000ms");

    await vi.advanceTimersByTimeAsync(15000);

    await assertion;
    expect(abortMessage).toContain("15000ms");
  });

  it("uses the extended timeout for workspace file listing", async () => {
    vi.useFakeTimers();
    let abortMessage = "";
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => new Promise((_resolve, reject) => {
      const signal = init?.signal as AbortSignal | undefined;
      signal?.addEventListener("abort", () => {
        const reason = signal.reason;
        abortMessage = String(reason instanceof Error ? reason.message : reason || "");
        reject(reason);
      }, { once: true });
    }));
    vi.stubGlobal("fetch", fetchMock);

    const request = fetchWorkspaceFiles("workspace-1", "nested/path");
    const assertion = expect(request).rejects.toThrow("Request timed out after 20000ms");

    await vi.advanceTimersByTimeAsync(5000);
    expect(abortMessage).toBe("");

    await vi.advanceTimersByTimeAsync(15000);

    await assertion;
    expect(abortMessage).toContain("20000ms");
  });

  it("uses the extended timeout for run detail requests", async () => {
    vi.useFakeTimers();
    let abortMessage = "";
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => new Promise((_resolve, reject) => {
      const signal = init?.signal as AbortSignal | undefined;
      signal?.addEventListener("abort", () => {
        const reason = signal.reason;
        abortMessage = String(reason instanceof Error ? reason.message : reason || "");
        reject(reason);
      }, { once: true });
    }));
    vi.stubGlobal("fetch", fetchMock);

    const request = fetchRunDetail("run-1");
    const assertion = expect(request).rejects.toThrow("Request timed out after 20000ms");

    await vi.advanceTimersByTimeAsync(5000);
    expect(abortMessage).toBe("");

    await vi.advanceTimersByTimeAsync(15000);

    await assertion;
    expect(abortMessage).toContain("20000ms");
  });

  it("uses the extended timeout for conversation history requests", async () => {
    vi.useFakeTimers();
    let abortMessages: string[] = [];
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => new Promise((_resolve, reject) => {
      const signal = init?.signal as AbortSignal | undefined;
      signal?.addEventListener("abort", () => {
        const reason = signal.reason;
        abortMessages = [
          ...abortMessages,
          String(reason instanceof Error ? reason.message : reason || ""),
        ];
        reject(reason);
      }, { once: true });
    }));
    vi.stubGlobal("fetch", fetchMock);

    const messagesRequest = fetchConversationMessages("conversation-1", { latest: true, limit: 250 });
    const eventsRequest = fetchConversationEvents("conversation-1", { limit: 250 });
    const messagesAssertion = expect(messagesRequest).rejects.toThrow("Request timed out after 20000ms");
    const eventsAssertion = expect(eventsRequest).rejects.toThrow("Request timed out after 20000ms");

    await vi.advanceTimersByTimeAsync(5000);
    expect(abortMessages).toEqual([]);

    await vi.advanceTimersByTimeAsync(15000);

    await messagesAssertion;
    await eventsAssertion;
    expect(abortMessages).toHaveLength(2);
    expect(abortMessages.every((message) => message.includes("20000ms"))).toBe(true);
  });

  it("uses the launch timeout budget for creating runs", async () => {
    vi.useFakeTimers();
    let abortMessage = "";
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => new Promise((_resolve, reject) => {
      const signal = init?.signal as AbortSignal | undefined;
      signal?.addEventListener("abort", () => {
        const reason = signal.reason;
        abortMessage = String(reason instanceof Error ? reason.message : reason || "");
        reject(reason);
      }, { once: true });
    }));
    vi.stubGlobal("fetch", fetchMock);

    const request = createTask({
      goal: "Launch a run",
      workspace: "/tmp/workspace",
    });
    const assertion = expect(request).rejects.toThrow("Request timed out after 60000ms");

    await vi.advanceTimersByTimeAsync(20000);
    expect(abortMessage).toBe("");

    await vi.advanceTimersByTimeAsync(40000);

    await assertion;
    expect(abortMessage).toContain("60000ms");
  });

  it("uses the launch timeout budget for restarting runs", async () => {
    vi.useFakeTimers();
    let abortMessage = "";
    const fetchMock = vi.fn((_url: string, init?: RequestInit) => new Promise((_resolve, reject) => {
      const signal = init?.signal as AbortSignal | undefined;
      signal?.addEventListener("abort", () => {
        const reason = signal.reason;
        abortMessage = String(reason instanceof Error ? reason.message : reason || "");
        reject(reason);
      }, { once: true });
    }));
    vi.stubGlobal("fetch", fetchMock);

    const request = restartRun("run-1");
    const assertion = expect(request).rejects.toThrow("Request timed out after 60000ms");

    await vi.advanceTimersByTimeAsync(20000);
    expect(abortMessage).toBe("");

    await vi.advanceTimersByTimeAsync(40000);

    await assertion;
    expect(abortMessage).toContain("60000ms");
  });

  it("requests a capped run timeline payload", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => [],
    }));
    vi.stubGlobal("fetch", fetchMock);

    await fetchRunTimeline("run-1", { includeNoise: false });

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/runs/run-1/timeline?limit=1000&include_noise=false"),
      expect.objectContaining({
        signal: expect.any(AbortSignal),
      }),
    );
  });
});
