import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useInbox } from "./useInbox";

vi.mock("../api", () => ({
  replyApproval: vi.fn(),
}));

describe("useInbox", () => {
  it("opens a run search result and clears stale thread selection", () => {
    const setSelectedWorkspaceId = vi.fn();
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();
    const setActiveTab = vi.fn();

    const { result } = renderHook(() =>
      useInbox({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "conversation-stale",
        selectedRunId: "",
        setSelectedWorkspaceId,
        setSelectedConversationId,
        setSelectedRunId,
        setActiveTab,
        setRunProcess: vi.fn(),
        setError: vi.fn(),
        setNotice: vi.fn(),
        refreshWorkspaceSurface: vi.fn(async () => {}),
        refreshApprovalInbox: vi.fn(async () => {}),
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
    const setActiveTab = vi.fn();

    const { result } = renderHook(() =>
      useInbox({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "run-stale",
        setSelectedWorkspaceId,
        setSelectedConversationId,
        setSelectedRunId,
        setActiveTab,
        setRunProcess: vi.fn(),
        setError: vi.fn(),
        setNotice: vi.fn(),
        refreshWorkspaceSurface: vi.fn(async () => {}),
        refreshApprovalInbox: vi.fn(async () => {}),
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
        setActiveTab,
        setRunProcess: vi.fn(),
        setError: vi.fn(),
        setNotice,
        refreshWorkspaceSurface: vi.fn(async () => {}),
        refreshApprovalInbox: vi.fn(async () => {}),
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
});
