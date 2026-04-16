import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useCommandPalette } from "./useCommandPalette";

vi.mock("../api", () => ({
  fetchGlobalSearch: vi.fn(async () => null),
}));

describe("useCommandPalette", () => {
  it("opens integrations shortcuts in the selected workspace", () => {
    const setActiveTab = vi.fn();
    const setIntegrationIntent = vi.fn();

    const { result } = renderHook(() =>
      useCommandPalette({
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {} as any,
          recent_conversations: [],
          recent_runs: [],
          pending_approvals_count: 0,
          counts: {},
        },
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
        setWorkspaceSearchQuery: vi.fn(),
        setActiveTab,
        setError: vi.fn(),
        setNotice: vi.fn(),
        setIntegrationIntent,
        focusSearch: vi.fn(),
        focusConversationComposer: vi.fn(),
        focusRunComposer: vi.fn(),
        handlePrefillStarterConversation: vi.fn(),
        handlePrefillStarterRun: vi.fn(),
        handleSearchResultSelection: vi.fn(),
      }),
    );

    act(() => {
      result.current.handleCommandAction("connect account");
    });

    expect(setActiveTab).toHaveBeenCalledWith("integrations");
    expect(setIntegrationIntent).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "create_account",
      }),
    );
  });

  it("opens the latest thread by recency and clears stale run selection", () => {
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();
    const setActiveTab = vi.fn();

    const { result } = renderHook(() =>
      useCommandPalette({
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {} as any,
          recent_conversations: [
            {
              id: "conversation-old",
              last_active_at: "2026-03-27T00:00:00Z",
            },
            {
              id: "conversation-new",
              last_active_at: "2026-03-28T00:00:00Z",
            },
          ] as any,
          recent_runs: [],
          pending_approvals_count: 0,
          counts: {},
        },
        setSelectedConversationId,
        setSelectedRunId,
        setWorkspaceSearchQuery: vi.fn(),
        setActiveTab,
        setError: vi.fn(),
        setNotice: vi.fn(),
        setIntegrationIntent: vi.fn(),
        focusSearch: vi.fn(),
        focusConversationComposer: vi.fn(),
        focusRunComposer: vi.fn(),
        handlePrefillStarterConversation: vi.fn(),
        handlePrefillStarterRun: vi.fn(),
        handleSearchResultSelection: vi.fn(),
      }),
    );

    act(() => {
      result.current.handleCommandAction("latest thread");
    });

    expect(setSelectedRunId).toHaveBeenCalledWith("");
    expect(setSelectedConversationId).toHaveBeenCalledWith("conversation-new");
    expect(setActiveTab).toHaveBeenCalledWith("threads");
  });

  it("opens the latest run by recency and clears stale thread selection", () => {
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();
    const setActiveTab = vi.fn();

    const { result } = renderHook(() =>
      useCommandPalette({
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {} as any,
          recent_conversations: [],
          recent_runs: [
            {
              id: "run-old",
              updated_at: "2026-03-27T00:00:00Z",
            },
            {
              id: "run-new",
              updated_at: "2026-03-28T00:00:00Z",
            },
          ] as any,
          pending_approvals_count: 0,
          counts: {},
        },
        setSelectedConversationId,
        setSelectedRunId,
        setWorkspaceSearchQuery: vi.fn(),
        setActiveTab,
        setError: vi.fn(),
        setNotice: vi.fn(),
        setIntegrationIntent: vi.fn(),
        focusSearch: vi.fn(),
        focusConversationComposer: vi.fn(),
        focusRunComposer: vi.fn(),
        handlePrefillStarterConversation: vi.fn(),
        handlePrefillStarterRun: vi.fn(),
        handleSearchResultSelection: vi.fn(),
      }),
    );

    act(() => {
      result.current.handleCommandAction("latest run");
    });

    expect(setSelectedConversationId).toHaveBeenCalledWith("");
    expect(setSelectedRunId).toHaveBeenCalledWith("run-new");
    expect(setActiveTab).toHaveBeenCalledWith("runs");
  });
});
