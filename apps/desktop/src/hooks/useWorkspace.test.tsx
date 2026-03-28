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
    expect(apiMocks.fetchWorkspaceSettings).toHaveBeenCalledTimes(1);
    expect(apiMocks.fetchWorkspaceInventory).toHaveBeenCalledTimes(1);
    expect(apiMocks.fetchWorkspaceArtifacts).toHaveBeenCalledTimes(1);
  });
});
