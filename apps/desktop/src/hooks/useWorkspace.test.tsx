import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useWorkspace } from "./useWorkspace";

const apiMocks = vi.hoisted(() => ({
  createWorkspace: vi.fn(),
  createWorkspaceDirectory: vi.fn(),
  fetchApprovals: vi.fn(),
  fetchWorkspaceArtifacts: vi.fn(),
  fetchWorkspaceInventory: vi.fn(),
  fetchWorkspaceOverview: vi.fn(),
  fetchWorkspaceSearch: vi.fn(),
  fetchWorkspaceSettings: vi.fn(),
  fetchWorkspaces: vi.fn(),
  patchWorkspace: vi.fn(),
  subscribeNotificationsStream: vi.fn(() => () => {}),
}));

vi.mock("../api", () => ({
  createWorkspace: apiMocks.createWorkspace,
  createWorkspaceDirectory: apiMocks.createWorkspaceDirectory,
  fetchApprovals: apiMocks.fetchApprovals,
  fetchWorkspaceArtifacts: apiMocks.fetchWorkspaceArtifacts,
  fetchWorkspaceInventory: apiMocks.fetchWorkspaceInventory,
  fetchWorkspaceOverview: apiMocks.fetchWorkspaceOverview,
  fetchWorkspaceSearch: apiMocks.fetchWorkspaceSearch,
  fetchWorkspaceSettings: apiMocks.fetchWorkspaceSettings,
  fetchWorkspaces: apiMocks.fetchWorkspaces,
  patchWorkspace: apiMocks.patchWorkspace,
  subscribeNotificationsStream: apiMocks.subscribeNotificationsStream,
}));

vi.mock("../history", () => ({
  matchesWorkspaceSearch: () => true,
}));

describe("useWorkspace", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  beforeEach(() => {
    apiMocks.createWorkspace.mockReset();
    apiMocks.createWorkspaceDirectory.mockReset();
    apiMocks.fetchApprovals.mockReset();
    apiMocks.fetchWorkspaceArtifacts.mockReset();
    apiMocks.fetchWorkspaceInventory.mockReset();
    apiMocks.fetchWorkspaceOverview.mockReset();
    apiMocks.fetchWorkspaceSearch.mockReset();
    apiMocks.fetchWorkspaceSettings.mockReset();
    apiMocks.fetchWorkspaces.mockReset();
    apiMocks.patchWorkspace.mockReset();
    apiMocks.subscribeNotificationsStream.mockClear();

    apiMocks.fetchWorkspaceOverview.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      recent_conversations: [],
      recent_runs: [],
    });
    apiMocks.fetchWorkspaceSettings.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspace_id: "workspace-1",
      overrides: {},
      created_at: "",
      updated_at: "",
    });
    apiMocks.fetchApprovals.mockResolvedValue([]);
    apiMocks.fetchWorkspaceInventory.mockResolvedValue({
      processes: [],
      tools: [],
      mcp_servers: [],
    });
    apiMocks.fetchWorkspaceArtifacts.mockResolvedValue([]);
    apiMocks.fetchWorkspaceSearch.mockResolvedValue(null);
  });

  it("coalesces notification-triggered refreshes into one workspace refresh burst", async () => {
    vi.useFakeTimers();
    let notificationEvent: ((event: any) => void) | undefined;
    let notificationError: (() => void) | undefined;
    apiMocks.subscribeNotificationsStream.mockImplementation(((
      _workspaceId: string,
      onEvent: (event: unknown) => void,
      onError?: () => void,
    ) => {
      notificationEvent = onEvent as (event: any) => void;
      notificationError = onError;
      return () => {};
    }) as any);

    renderHook(() =>
      useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces: [{
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
          metadata: {},
          is_archived: false,
          sort_order: 0,
        }] as any,
        setWorkspaces: vi.fn(),
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "overview",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      }),
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchWorkspaceOverview.mockClear();
    apiMocks.fetchWorkspaceSettings.mockClear();
    apiMocks.fetchApprovals.mockClear();
    apiMocks.fetchWorkspaceInventory.mockClear();
    apiMocks.fetchWorkspaceArtifacts.mockClear();

    act(() => {
      notificationEvent?.({
        id: "evt-1",
        stream_id: 4,
        event_type: "approval_requested",
        created_at: "2026-03-27T00:00:00Z",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace 1",
        task_id: "task-1",
        conversation_id: "",
        approval_id: "",
        kind: "task_approval",
        title: "Task approval",
        summary: "Need approval",
        payload: {},
      });
      notificationEvent?.({
        id: "evt-2",
        stream_id: 5,
        event_type: "approval_received",
        created_at: "2026-03-27T00:00:01Z",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace 1",
        task_id: "task-1",
        conversation_id: "",
        approval_id: "",
        kind: "task_approval",
        title: "Task approval",
        summary: "Approved",
        payload: {},
      });
      notificationError?.();
      vi.advanceTimersByTime(200);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.fetchApprovals).toHaveBeenCalledTimes(1);
    expect(apiMocks.fetchWorkspaceOverview).toHaveBeenCalledTimes(1);
    expect(apiMocks.fetchWorkspaceSettings).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceInventory).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceArtifacts).not.toHaveBeenCalled();
  });

  it("preserves the loaded workspace surface during a disconnect and refreshes on reconnect", async () => {
    const initialProps: { connectionState: "connected" | "failed" } = {
      connectionState: "connected",
    };
    const { result, rerender } = renderHook(
      ({ connectionState }: { connectionState: "connected" | "failed" }) =>
        useWorkspace({
          selectedWorkspaceId: "workspace-1",
          selectedConversationId: "",
          selectedRunId: "",
          connectionState,
          setSelectedWorkspaceId: vi.fn(),
          showArchivedWorkspaces: false,
          setShowArchivedWorkspaces: vi.fn(),
          createParentPath: "/tmp",
          setCreateParentPath: vi.fn(),
          workspaces: [{
            id: "workspace-1",
            canonical_path: "/tmp/workspace",
            display_name: "Workspace 1",
            metadata: {},
            is_archived: false,
            sort_order: 0,
          }] as any,
          setWorkspaces: vi.fn(),
          runtime: null,
          setError: vi.fn(),
          setNotice: vi.fn(),
          activeTab: "overview",
          setActiveTab: vi.fn(),
          setSelectedConversationId: vi.fn(),
          setSelectedRunId: vi.fn(),
        }),
      {
        initialProps,
      },
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(result.current.overview?.workspace.id).toBe("workspace-1");

    apiMocks.fetchWorkspaceOverview.mockClear();
    apiMocks.fetchWorkspaceSettings.mockClear();
    apiMocks.fetchApprovals.mockClear();
    apiMocks.fetchWorkspaceInventory.mockClear();
    apiMocks.fetchWorkspaceArtifacts.mockClear();

    rerender({ connectionState: "failed" });

    expect(result.current.overview?.workspace.id).toBe("workspace-1");
    expect(result.current.loadingOverview).toBe(false);

    rerender({ connectionState: "connected" });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(apiMocks.fetchWorkspaceOverview).toHaveBeenCalledWith("workspace-1");
    expect(apiMocks.fetchApprovals).toHaveBeenCalledWith("workspace-1");
    expect(apiMocks.fetchWorkspaceArtifacts).toHaveBeenCalledWith("workspace-1");
    expect(apiMocks.fetchWorkspaceSettings).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceInventory).not.toHaveBeenCalled();
  });

  it("loads only the visible workspace surface and fetches inventory lazily for the runs tab", async () => {
    const { rerender } = renderHook(
      ({ activeTab }: { activeTab: "threads" | "runs" }) =>
        useWorkspace({
          selectedWorkspaceId: "workspace-1",
          selectedConversationId: "",
          selectedRunId: "",
          setSelectedWorkspaceId: vi.fn(),
          showArchivedWorkspaces: false,
          setShowArchivedWorkspaces: vi.fn(),
          createParentPath: "/tmp",
          setCreateParentPath: vi.fn(),
          workspaces: [{
            id: "workspace-1",
            canonical_path: "/tmp/workspace",
            display_name: "Workspace 1",
            metadata: {},
            is_archived: false,
            sort_order: 0,
          }] as any,
          setWorkspaces: vi.fn(),
          runtime: null,
          setError: vi.fn(),
          setNotice: vi.fn(),
          activeTab,
          setActiveTab: vi.fn(),
          setSelectedConversationId: vi.fn(),
          setSelectedRunId: vi.fn(),
        }),
      {
        initialProps: {
          activeTab: "threads",
        },
      },
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(apiMocks.fetchWorkspaceOverview).toHaveBeenCalledTimes(1);
    expect(apiMocks.fetchApprovals).toHaveBeenCalledTimes(1);
    expect(apiMocks.fetchWorkspaceArtifacts).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceInventory).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceSettings).not.toHaveBeenCalled();

    apiMocks.fetchWorkspaceOverview.mockClear();
    apiMocks.fetchApprovals.mockClear();
    apiMocks.fetchWorkspaceArtifacts.mockClear();
    apiMocks.fetchWorkspaceInventory.mockClear();
    apiMocks.fetchWorkspaceSettings.mockClear();

    rerender({ activeTab: "runs" });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(apiMocks.fetchWorkspaceInventory).toHaveBeenCalledWith("workspace-1");
    expect(apiMocks.fetchWorkspaceOverview).not.toHaveBeenCalled();
    expect(apiMocks.fetchApprovals).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceArtifacts).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceSettings).not.toHaveBeenCalled();
  });

  it("clears stale thread and run selection immediately when switching workspaces", async () => {
    vi.useFakeTimers();
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();

    let resolveOverview!: (value: any) => void;
    apiMocks.fetchWorkspaceOverview.mockImplementationOnce(
      () => new Promise((resolve) => { resolveOverview = resolve; }),
    );

    const { rerender } = renderHook(
      ({
        selectedWorkspaceId,
        workspaces,
      }: {
        selectedWorkspaceId: string;
        workspaces: any[];
      }) =>
        useWorkspace({
          selectedWorkspaceId,
          selectedConversationId: "",
          selectedRunId: "",
          setSelectedWorkspaceId: vi.fn(),
          showArchivedWorkspaces: false,
          setShowArchivedWorkspaces: vi.fn(),
          createParentPath: "/tmp",
          setCreateParentPath: vi.fn(),
          workspaces,
          setWorkspaces: vi.fn(),
          runtime: null,
          setError: vi.fn(),
          setNotice: vi.fn(),
          activeTab: "threads",
          setActiveTab: vi.fn(),
          setSelectedConversationId,
          setSelectedRunId,
        }),
      {
        initialProps: {
          selectedWorkspaceId: "workspace-1",
          workspaces: [{
            id: "workspace-1",
            canonical_path: "/tmp/workspace-1",
            display_name: "Workspace 1",
            metadata: {},
            is_archived: false,
            sort_order: 0,
          }],
        },
      },
    );

    await act(async () => {
      resolveOverview({
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace-1",
          display_name: "Workspace 1",
        },
        recent_conversations: [{
          id: "conversation-1",
          title: "Conversation 1",
          model_name: "kimi-k2.5",
          last_active_at: "",
          started_at: "",
          linked_run_ids: [],
        }],
        recent_runs: [{
          id: "run-1",
          goal: "Run 1",
          status: "completed",
          created_at: "",
          updated_at: "",
          process_name: "",
          linked_conversation_ids: [],
        }],
      });
      await Promise.resolve();
      await Promise.resolve();
    });

    setSelectedConversationId.mockClear();
    setSelectedRunId.mockClear();

    rerender({
      selectedWorkspaceId: "workspace-2",
      workspaces: [
        {
          id: "workspace-1",
          canonical_path: "/tmp/workspace-1",
          display_name: "Workspace 1",
          metadata: {},
          is_archived: false,
          sort_order: 0,
        },
        {
          id: "workspace-2",
          canonical_path: "/tmp/workspace-2",
          display_name: "Workspace 2",
          metadata: {},
          is_archived: false,
          sort_order: 1,
        },
      ],
    });

    expect(setSelectedConversationId).toHaveBeenCalledWith("");
    expect(setSelectedRunId).toHaveBeenCalledWith("");
  });

  it("preserves an explicitly targeted thread and run during cross-workspace navigation", async () => {
    vi.useFakeTimers();
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();

    apiMocks.fetchWorkspaceOverview.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace-1",
        display_name: "Workspace 1",
      },
      recent_conversations: [{
        id: "conversation-1",
        title: "Conversation 1",
        model_name: "kimi-k2.5",
        last_active_at: "",
        started_at: "",
        linked_run_ids: [],
      }],
      recent_runs: [{
        id: "run-1",
        goal: "Run 1",
        status: "completed",
        created_at: "",
        updated_at: "",
        process_name: "",
        linked_conversation_ids: [],
      }],
    });

    const { rerender } = renderHook(
      ({
        selectedWorkspaceId,
        selectedConversationId,
        selectedRunId,
        workspaces,
      }: {
        selectedWorkspaceId: string;
        selectedConversationId: string;
        selectedRunId: string;
        workspaces: any[];
      }) =>
        useWorkspace({
          selectedWorkspaceId,
          selectedConversationId,
          selectedRunId,
          setSelectedWorkspaceId: vi.fn(),
          showArchivedWorkspaces: false,
          setShowArchivedWorkspaces: vi.fn(),
          createParentPath: "/tmp",
          setCreateParentPath: vi.fn(),
          workspaces,
          setWorkspaces: vi.fn(),
          runtime: null,
          setError: vi.fn(),
          setNotice: vi.fn(),
          activeTab: "threads",
          setActiveTab: vi.fn(),
          setSelectedConversationId,
          setSelectedRunId,
        }),
      {
        initialProps: {
          selectedWorkspaceId: "workspace-1",
          selectedConversationId: "conversation-1",
          selectedRunId: "run-1",
          workspaces: [
            {
              id: "workspace-1",
              canonical_path: "/tmp/workspace-1",
              display_name: "Workspace 1",
              metadata: {},
              is_archived: false,
              sort_order: 0,
            },
          ],
        },
      },
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    setSelectedConversationId.mockClear();
    setSelectedRunId.mockClear();

    rerender({
      selectedWorkspaceId: "workspace-2",
      selectedConversationId: "conversation-2",
      selectedRunId: "run-2",
      workspaces: [
        {
          id: "workspace-1",
          canonical_path: "/tmp/workspace-1",
          display_name: "Workspace 1",
          metadata: {},
          is_archived: false,
          sort_order: 0,
        },
        {
          id: "workspace-2",
          canonical_path: "/tmp/workspace-2",
          display_name: "Workspace 2",
          metadata: {},
          is_archived: false,
          sort_order: 1,
        },
      ],
    });

    expect(setSelectedConversationId).not.toHaveBeenCalledWith("");
    expect(setSelectedRunId).not.toHaveBeenCalledWith("");
  });

  it("does not clear a selected run just because it is not in the recent-runs overview slice", async () => {
    vi.useFakeTimers();
    const setSelectedConversationId = vi.fn();
    const setSelectedRunId = vi.fn();

    apiMocks.fetchWorkspaceOverview.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace-1",
        display_name: "Workspace 1",
      },
      recent_conversations: [],
      recent_runs: [{
        id: "run-1",
        goal: "Run 1",
        status: "completed",
        created_at: "",
        updated_at: "",
        process_name: "",
        linked_conversation_ids: [],
      }],
    });

    renderHook(() =>
      useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "run-newly-created",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces: [{
          id: "workspace-1",
          canonical_path: "/tmp/workspace-1",
          display_name: "Workspace 1",
          metadata: {},
          is_archived: false,
          sort_order: 0,
        }] as any,
        setWorkspaces: vi.fn(),
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "runs",
        setActiveTab: vi.fn(),
        setSelectedConversationId,
        setSelectedRunId,
      }),
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(setSelectedRunId).not.toHaveBeenCalledWith("run-1");
    expect(setSelectedRunId).not.toHaveBeenCalledWith("");
  });

  it("does not refetch the full workspace surface when only the selected thread or run changes", async () => {
    const { rerender } = renderHook(
      ({
        selectedConversationId,
        selectedRunId,
      }: {
        selectedConversationId: string;
        selectedRunId: string;
      }) =>
        useWorkspace({
          selectedWorkspaceId: "workspace-1",
          selectedConversationId,
          selectedRunId,
          setSelectedWorkspaceId: vi.fn(),
          showArchivedWorkspaces: false,
          setShowArchivedWorkspaces: vi.fn(),
          createParentPath: "/tmp",
          setCreateParentPath: vi.fn(),
          workspaces: [{
            id: "workspace-1",
            canonical_path: "/tmp/workspace",
            display_name: "Workspace 1",
            metadata: {},
            is_archived: false,
            sort_order: 0,
          }] as any,
          setWorkspaces: vi.fn(),
          runtime: null,
          setError: vi.fn(),
          setNotice: vi.fn(),
          activeTab: "threads",
          setActiveTab: vi.fn(),
          setSelectedConversationId: vi.fn(),
          setSelectedRunId: vi.fn(),
        }),
      {
        initialProps: {
          selectedConversationId: "",
          selectedRunId: "",
        },
      },
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchWorkspaceOverview.mockClear();
    apiMocks.fetchWorkspaceSettings.mockClear();
    apiMocks.fetchApprovals.mockClear();
    apiMocks.fetchWorkspaceInventory.mockClear();
    apiMocks.fetchWorkspaceArtifacts.mockClear();

    rerender({
      selectedConversationId: "conversation-1",
      selectedRunId: "",
    });

    await act(async () => {
      await Promise.resolve();
    });

    rerender({
      selectedConversationId: "conversation-1",
      selectedRunId: "run-1",
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.fetchWorkspaceOverview).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceSettings).not.toHaveBeenCalled();
    expect(apiMocks.fetchApprovals).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceInventory).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceArtifacts).not.toHaveBeenCalled();
  });
});
