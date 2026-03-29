import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useDesktopActivity } from "./useDesktopActivity";

const apiMocks = vi.hoisted(() => ({
  fetchActivitySummary: vi.fn(),
}));

vi.mock("../api", () => ({
  fetchActivitySummary: apiMocks.fetchActivitySummary,
}));

vi.mock("../utils", () => ({
  isTransientRequestError: () => false,
}));

describe("useDesktopActivity", () => {
  beforeEach(() => {
    apiMocks.fetchActivitySummary.mockReset();
  });

  it("falls back to local thread activity while backend summary is idle", async () => {
    apiMocks.fetchActivitySummary.mockResolvedValue({
      status: "ok",
      active: false,
      active_conversation_count: 0,
      active_run_count: 0,
      updated_at: "2026-03-29T12:00:00Z",
    });

    const { result } = renderHook(() =>
      useDesktopActivity({
        connectionState: "connected",
        conversationIsProcessing: true,
        conversationStreaming: false,
        streamingToolCalls: [],
        runStreaming: false,
      }),
    );

    await waitFor(() => {
      expect(apiMocks.fetchActivitySummary).toHaveBeenCalled();
    });

    expect(result.current.active).toBe(true);
    expect(result.current.mode).toBe("thread");
    expect(result.current.activeConversationCount).toBe(1);
    expect(result.current.activeRunCount).toBe(0);
    expect(result.current.label).toBe("1 active thread");
  });

  it("uses the backend summary for global mixed activity", async () => {
    apiMocks.fetchActivitySummary.mockResolvedValue({
      status: "ok",
      active: true,
      active_conversation_count: 1,
      active_run_count: 2,
      updated_at: "2026-03-29T12:05:00Z",
    });

    const { result } = renderHook(() =>
      useDesktopActivity({
        connectionState: "connected",
        conversationIsProcessing: false,
        conversationStreaming: false,
        streamingToolCalls: [],
        runStreaming: false,
      }),
    );

    await waitFor(() => {
      expect(result.current.backendConnected).toBe(true);
    });

    expect(result.current.active).toBe(true);
    expect(result.current.mode).toBe("mixed");
    expect(result.current.activeConversationCount).toBe(1);
    expect(result.current.activeRunCount).toBe(2);
    expect(result.current.label).toBe("1 active thread · 2 active runs");
  });
});
