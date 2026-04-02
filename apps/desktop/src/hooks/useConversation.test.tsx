import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useConversation } from "./useConversation";

const apiMocks = vi.hoisted(() => ({
  createConversation: vi.fn(),
  patchConversation: vi.fn(),
  fetchConversationDetail: vi.fn(),
  fetchConversationEvents: vi.fn(),
  fetchConversationMessages: vi.fn(),
  fetchConversationStatus: vi.fn(),
  injectConversationInstruction: vi.fn(),
  resolveConversationApproval: vi.fn(),
  sendConversationMessage: vi.fn(),
  stopConversationTurn: vi.fn(),
  subscribeConversationStream: vi.fn(() => () => {}),
}));

vi.mock("../api", () => ({
  createConversation: apiMocks.createConversation,
  patchConversation: apiMocks.patchConversation,
  fetchConversationDetail: apiMocks.fetchConversationDetail,
  fetchConversationEvents: apiMocks.fetchConversationEvents,
  fetchConversationMessages: apiMocks.fetchConversationMessages,
  fetchConversationStatus: apiMocks.fetchConversationStatus,
  injectConversationInstruction: apiMocks.injectConversationInstruction,
  resolveConversationApproval: apiMocks.resolveConversationApproval,
  sendConversationMessage: apiMocks.sendConversationMessage,
  stopConversationTurn: apiMocks.stopConversationTurn,
  subscribeConversationStream: apiMocks.subscribeConversationStream,
}));

vi.mock("../history", () => ({
  conversationEventDetail: () => "",
  conversationEventPills: () => [],
  conversationEventTitle: () => "",
  matchesWorkspaceSearch: () => true,
  normalizeConversationPrompt: () => null,
  summarizeMessage: () => "",
}));

vi.mock("../utils", () => ({
  isTransientRequestError: () => false,
}));

describe("useConversation", () => {
  beforeEach(() => {
    apiMocks.createConversation.mockReset();
    apiMocks.patchConversation.mockReset();
    apiMocks.fetchConversationDetail.mockReset();
    apiMocks.fetchConversationEvents.mockReset();
    apiMocks.fetchConversationMessages.mockReset();
    apiMocks.fetchConversationStatus.mockReset();
    apiMocks.injectConversationInstruction.mockReset();
    apiMocks.resolveConversationApproval.mockReset();
    apiMocks.sendConversationMessage.mockReset();
    apiMocks.stopConversationTurn.mockReset();
    apiMocks.subscribeConversationStream.mockClear();

    apiMocks.fetchConversationDetail.mockResolvedValue({
      id: "conv-1",
      workspace_id: "workspace-1",
      workspace_path: "/tmp/workspace",
      model_name: "gpt-5.4",
      title: "Conversation conv-1",
      turn_count: 1,
      total_tokens: 0,
      started_at: "2026-04-01T00:00:00Z",
      last_active_at: "2026-04-01T00:00:00Z",
      linked_run_ids: [],
      is_active: false,
      session_state: {},
    });
    apiMocks.fetchConversationStatus.mockResolvedValue({
      conversation_id: "conv-1",
      processing: false,
      stop_requested: false,
      pending_inject_count: 0,
      awaiting_approval: false,
      pending_approval: null,
      awaiting_user_input: false,
      pending_prompt: null,
    });
    apiMocks.fetchConversationMessages.mockResolvedValue([
      {
        id: 1,
        turn_number: 1,
        role: "user",
        content: "Hello",
        created_at: "2026-04-01T00:00:00Z",
      },
    ]);
    apiMocks.fetchConversationEvents.mockResolvedValue([]);
  });

  it("subscribes to the conversation stream once after boot load settles", async () => {
    const { rerender } = renderHook(() =>
      useConversation({
        selectedConversationId: "conv-1",
        connectionState: "connected",
        setSelectedConversationId: vi.fn(),
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {
            id: "workspace-1",
            canonical_path: "/tmp/workspace",
            display_name: "Workspace 1",
          },
          recent_conversations: [],
          recent_runs: [],
          pending_approvals_count: 0,
        } as any,
        models: [],
        setError: vi.fn(),
        setNotice: vi.fn(),
        setActiveTab: vi.fn(),
        refreshWorkspaceSurface: vi.fn(async () => {}),
        syncConversationSummary: vi.fn(),
        setConversationProcessing: vi.fn(),
        removeApprovalItem: vi.fn(),
      }),
    );

    await waitFor(() => {
      expect(apiMocks.subscribeConversationStream).toHaveBeenCalledTimes(1);
    });

    rerender();

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(apiMocks.subscribeConversationStream).toHaveBeenCalledTimes(1);
  });
});
