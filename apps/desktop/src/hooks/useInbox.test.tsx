import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { replyApproval } from "../api";
import { useInbox } from "./useInbox";

vi.mock("../api", () => ({
  replyApproval: vi.fn(),
}));

describe("useInbox", () => {
  it("opens a run search result and clears stale thread selection", () => {
    const setSelectedWorkspaceId = vi.fn();
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();
    const setWorkspaceSearchQuery = vi.fn();
    const setActiveTab = vi.fn();

    const { result } = renderHook(() =>
      useInbox({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "conversation-stale",
        selectedRunId: "",
        setSelectedWorkspaceId,
        setSelectedConversationId,
        setSelectedRunId,
        setWorkspaceSearchQuery,
        setActiveTab,
        setRunProcess: vi.fn(),
        setError: vi.fn(),
        setNotice: vi.fn(),
        removeApprovalItem: vi.fn(),
        refreshConversation: vi.fn(async () => {}),
        refreshRun: vi.fn(async () => {}),
        queueWorkspaceFileOpen: vi.fn(),
        focusRunComposer: vi.fn(),
      }),
    );

    act(() => {
      result.current.handleSearchResultSelection({
        kind: "run",
        workspace_id: "workspace-2",
        run_id: "run-123",
      });
    });

    expect(setSelectedWorkspaceId).toHaveBeenCalledWith("workspace-2");
    expect(setSelectedConversationId).toHaveBeenCalledWith("");
    expect(setSelectedRunId).toHaveBeenCalledWith("run-123");
    expect(setActiveTab).toHaveBeenCalledWith("runs");
  });

  it("opens a thread search result and clears stale run selection", () => {
    const setSelectedWorkspaceId = vi.fn();
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();
    const setWorkspaceSearchQuery = vi.fn();
    const setActiveTab = vi.fn();

    const { result } = renderHook(() =>
      useInbox({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "run-stale",
        setSelectedWorkspaceId,
        setSelectedConversationId,
        setSelectedRunId,
        setWorkspaceSearchQuery,
        setActiveTab,
        setRunProcess: vi.fn(),
        setError: vi.fn(),
        setNotice: vi.fn(),
        removeApprovalItem: vi.fn(),
        refreshConversation: vi.fn(async () => {}),
        refreshRun: vi.fn(async () => {}),
        queueWorkspaceFileOpen: vi.fn(),
        focusRunComposer: vi.fn(),
      }),
    );

    act(() => {
      result.current.handleSearchResultSelection({
        kind: "conversation",
        workspace_id: "workspace-2",
        conversation_id: "conversation-123",
      });
    });

    expect(setSelectedWorkspaceId).toHaveBeenCalledWith("workspace-2");
    expect(setSelectedRunId).toHaveBeenCalledWith("");
    expect(setSelectedConversationId).toHaveBeenCalledWith("conversation-123");
    expect(setActiveTab).toHaveBeenCalledWith("threads");
  });

  it("opens an artifact result in Files and queues the file path", () => {
    const setSelectedWorkspaceId = vi.fn();
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();
    const setWorkspaceSearchQuery = vi.fn();
    const setActiveTab = vi.fn();
    const queueWorkspaceFileOpen = vi.fn();
    const setNotice = vi.fn();

    const { result } = renderHook(() =>
      useInbox({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "conversation-stale",
        selectedRunId: "",
        setSelectedWorkspaceId,
        setSelectedConversationId,
        setSelectedRunId,
        setWorkspaceSearchQuery,
        setActiveTab,
        setRunProcess: vi.fn(),
        setError: vi.fn(),
        setNotice,
        removeApprovalItem: vi.fn(),
        refreshConversation: vi.fn(async () => {}),
        refreshRun: vi.fn(async () => {}),
        queueWorkspaceFileOpen,
        focusRunComposer: vi.fn(),
      }),
    );

    act(() => {
      result.current.handleSearchResultSelection({
        kind: "artifact",
        workspace_id: "workspace-2",
        run_id: "run-123",
        path: "reports/auth-report.md",
      });
    });

    expect(setSelectedWorkspaceId).toHaveBeenCalledWith("workspace-2");
    expect(queueWorkspaceFileOpen).toHaveBeenCalledWith("workspace-2", "reports/auth-report.md");
    expect(setSelectedConversationId).toHaveBeenCalledWith("");
    expect(setSelectedRunId).toHaveBeenCalledWith("run-123");
    expect(setActiveTab).toHaveBeenCalledWith("files");
    expect(setNotice).toHaveBeenCalledWith("Opened context for artifact reports/auth-report.md.");
  });

  it("opens integrations for account search results", () => {
    const setSelectedWorkspaceId = vi.fn();
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();
    const setWorkspaceSearchQuery = vi.fn();
    const setActiveTab = vi.fn();
    const setNotice = vi.fn();

    const { result } = renderHook(() =>
      useInbox({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "conversation-stale",
        selectedRunId: "run-stale",
        setSelectedWorkspaceId,
        setSelectedConversationId,
        setSelectedRunId,
        setWorkspaceSearchQuery,
        setActiveTab,
        setRunProcess: vi.fn(),
        setError: vi.fn(),
        setNotice,
        removeApprovalItem: vi.fn(),
        refreshConversation: vi.fn(async () => {}),
        refreshRun: vi.fn(async () => {}),
        queueWorkspaceFileOpen: vi.fn(),
        focusRunComposer: vi.fn(),
      }),
    );

    act(() => {
      result.current.handleSearchResultSelection({
        kind: "account",
        workspace_id: "workspace-2",
        item_id: "auth_search_profile",
        title: "Auth Search Account",
      });
    });

    expect(setSelectedWorkspaceId).toHaveBeenCalledWith("workspace-2");
    expect(setSelectedConversationId).toHaveBeenCalledWith("");
    expect(setSelectedRunId).toHaveBeenCalledWith("");
    expect(setWorkspaceSearchQuery).toHaveBeenCalledWith("Auth Search Account");
    expect(setActiveTab).toHaveBeenCalledWith("integrations");
    expect(setNotice).toHaveBeenCalledWith("Opened integrations for account Auth Search Account.");
  });

  it("removes the approval locally and only refreshes the active detail pane", async () => {
    vi.mocked(replyApproval).mockResolvedValue({ ok: true } as any);

    const removeApprovalItem = vi.fn();
    const refreshConversation = vi.fn(async () => {});
    const refreshRun = vi.fn(async () => {});
    const setNotice = vi.fn();

    const { result } = renderHook(() =>
      useInbox({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "conversation-1",
        selectedRunId: "run-stale",
        setSelectedWorkspaceId: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
        setWorkspaceSearchQuery: vi.fn(),
        setActiveTab: vi.fn(),
        setRunProcess: vi.fn(),
        setError: vi.fn(),
        setNotice,
        removeApprovalItem,
        refreshConversation,
        refreshRun,
        queueWorkspaceFileOpen: vi.fn(),
        focusRunComposer: vi.fn(),
      }),
    );

    await act(async () => {
      await result.current.handleReplyApproval({
        id: "approval-item-1",
        kind: "conversation_approval",
        status: "pending",
        created_at: "2026-03-31T18:00:00Z",
        title: "Thread approval",
        summary: "Approve tool use",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace 1",
        task_id: "",
        run_id: "",
        conversation_id: "conversation-1",
        subtask_id: "",
        question_id: "",
        approval_id: "approval-1",
        tool_name: "shell",
        risk_level: "medium",
        request_payload: {},
        metadata: {},
      }, {
        decision: "approve",
      });
    });

    expect(removeApprovalItem).toHaveBeenCalledWith("approval-item-1", "workspace-1");
    expect(refreshConversation).toHaveBeenCalledWith("conversation-1");
    expect(refreshRun).not.toHaveBeenCalled();
    expect(setNotice).toHaveBeenCalledWith("Thread approval updated.");
  });

  it("removes stale approvals locally when the server says they are gone", async () => {
    vi.mocked(replyApproval).mockRejectedValue(new Error("404 Not Found"));

    const removeApprovalItem = vi.fn();
    const refreshRun = vi.fn(async () => {});
    const setError = vi.fn();
    const setNotice = vi.fn();

    const { result } = renderHook(() =>
      useInbox({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "run-1",
        setSelectedWorkspaceId: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
        setWorkspaceSearchQuery: vi.fn(),
        setActiveTab: vi.fn(),
        setRunProcess: vi.fn(),
        setError,
        setNotice,
        removeApprovalItem,
        refreshConversation: vi.fn(async () => {}),
        refreshRun,
        queueWorkspaceFileOpen: vi.fn(),
        focusRunComposer: vi.fn(),
      }),
    );

    await act(async () => {
      await result.current.handleReplyApproval({
        id: "task:run-1:subtask-1",
        kind: "task_approval",
        status: "pending",
        created_at: "2026-03-31T18:00:00Z",
        title: "Run approval",
        summary: "Approve run step",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace 1",
        task_id: "run-1",
        run_id: "run-1",
        conversation_id: "",
        subtask_id: "subtask-1",
        question_id: "",
        approval_id: "",
        tool_name: "",
        risk_level: "medium",
        request_payload: {},
        metadata: {},
      }, {
        decision: "approve",
      });
    });

    expect(removeApprovalItem).toHaveBeenCalledWith("task:run-1:subtask-1", "workspace-1");
    expect(refreshRun).toHaveBeenCalledWith("run-1");
    expect(setNotice).toHaveBeenCalledWith("Run approval was already resolved.");
    expect(setError).not.toHaveBeenCalledWith("404 Not Found");
  });
});
