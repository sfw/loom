import { useState } from "react";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useWorkspace } from "./useWorkspace";

const apiMocks = vi.hoisted(() => ({
  approveWorkspaceMcpServer: vi.fn(),
  archiveWorkspaceAuthAccount: vi.fn(),
  createWorkspaceAuthAccount: vi.fn(),
  completeWorkspaceAuthAccountLogin: vi.fn(),
  createWorkspaceMcpServer: vi.fn(),
  createWorkspace: vi.fn(),
  createWorkspaceDirectory: vi.fn(),
  deleteWorkspaceMcpServer: vi.fn(),
  fetchApprovals: vi.fn(),
  fetchWorkspaceArtifacts: vi.fn(),
  fetchWorkspaceInventory: vi.fn(),
  fetchWorkspaceIntegrations: vi.fn(),
  fetchWorkspaceOverview: vi.fn(),
  fetchWorkspaceSearch: vi.fn(),
  fetchWorkspaceSettings: vi.fn(),
  fetchWorkspaces: vi.fn(),
  logoutWorkspaceAuthAccount: vi.fn(),
  patchWorkspace: vi.fn(),
  selectWorkspaceMcpAccount: vi.fn(),
  reconnectWorkspaceMcpServer: vi.fn(),
  refreshWorkspaceAuthAccount: vi.fn(),
  rejectWorkspaceMcpServer: vi.fn(),
  restoreWorkspaceAuthAccount: vi.fn(),
  setWorkspaceMcpServerEnabled: vi.fn(),
  startWorkspaceAuthAccountLogin: vi.fn(),
  subscribeNotificationsStream: vi.fn(() => () => {}),
  syncWorkspaceAuthDrafts: vi.fn(),
  testWorkspaceMcpServer: vi.fn(),
  updateWorkspaceAuthAccount: vi.fn(),
  updateWorkspaceMcpServer: vi.fn(),
}));

vi.mock("../api", () => ({
  approveWorkspaceMcpServer: apiMocks.approveWorkspaceMcpServer,
  archiveWorkspaceAuthAccount: apiMocks.archiveWorkspaceAuthAccount,
  createWorkspaceAuthAccount: apiMocks.createWorkspaceAuthAccount,
  completeWorkspaceAuthAccountLogin: apiMocks.completeWorkspaceAuthAccountLogin,
  createWorkspaceMcpServer: apiMocks.createWorkspaceMcpServer,
  createWorkspace: apiMocks.createWorkspace,
  createWorkspaceDirectory: apiMocks.createWorkspaceDirectory,
  deleteWorkspaceMcpServer: apiMocks.deleteWorkspaceMcpServer,
  fetchApprovals: apiMocks.fetchApprovals,
  fetchWorkspaceArtifacts: apiMocks.fetchWorkspaceArtifacts,
  fetchWorkspaceInventory: apiMocks.fetchWorkspaceInventory,
  fetchWorkspaceIntegrations: apiMocks.fetchWorkspaceIntegrations,
  fetchWorkspaceOverview: apiMocks.fetchWorkspaceOverview,
  fetchWorkspaceSearch: apiMocks.fetchWorkspaceSearch,
  fetchWorkspaceSettings: apiMocks.fetchWorkspaceSettings,
  fetchWorkspaces: apiMocks.fetchWorkspaces,
  logoutWorkspaceAuthAccount: apiMocks.logoutWorkspaceAuthAccount,
  patchWorkspace: apiMocks.patchWorkspace,
  selectWorkspaceMcpAccount: apiMocks.selectWorkspaceMcpAccount,
  reconnectWorkspaceMcpServer: apiMocks.reconnectWorkspaceMcpServer,
  refreshWorkspaceAuthAccount: apiMocks.refreshWorkspaceAuthAccount,
  rejectWorkspaceMcpServer: apiMocks.rejectWorkspaceMcpServer,
  restoreWorkspaceAuthAccount: apiMocks.restoreWorkspaceAuthAccount,
  setWorkspaceMcpServerEnabled: apiMocks.setWorkspaceMcpServerEnabled,
  startWorkspaceAuthAccountLogin: apiMocks.startWorkspaceAuthAccountLogin,
  subscribeNotificationsStream: apiMocks.subscribeNotificationsStream,
  syncWorkspaceAuthDrafts: apiMocks.syncWorkspaceAuthDrafts,
  testWorkspaceMcpServer: apiMocks.testWorkspaceMcpServer,
  updateWorkspaceAuthAccount: apiMocks.updateWorkspaceAuthAccount,
  updateWorkspaceMcpServer: apiMocks.updateWorkspaceMcpServer,
}));

vi.mock("../history", () => ({
  matchesWorkspaceSearch: () => true,
}));

describe("useWorkspace", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  beforeEach(() => {
    apiMocks.createWorkspaceMcpServer.mockReset();
    apiMocks.createWorkspaceAuthAccount.mockReset();
    apiMocks.createWorkspace.mockReset();
    apiMocks.createWorkspaceDirectory.mockReset();
    apiMocks.approveWorkspaceMcpServer.mockReset();
    apiMocks.archiveWorkspaceAuthAccount.mockReset();
    apiMocks.completeWorkspaceAuthAccountLogin.mockReset();
    apiMocks.deleteWorkspaceMcpServer.mockReset();
    apiMocks.fetchApprovals.mockReset();
    apiMocks.fetchWorkspaceArtifacts.mockReset();
    apiMocks.fetchWorkspaceInventory.mockReset();
    apiMocks.fetchWorkspaceIntegrations.mockReset();
    apiMocks.fetchWorkspaceOverview.mockReset();
    apiMocks.fetchWorkspaceSearch.mockReset();
    apiMocks.fetchWorkspaceSettings.mockReset();
    apiMocks.fetchWorkspaces.mockReset();
    apiMocks.logoutWorkspaceAuthAccount.mockReset();
    apiMocks.patchWorkspace.mockReset();
    apiMocks.selectWorkspaceMcpAccount.mockReset();
    apiMocks.reconnectWorkspaceMcpServer.mockReset();
    apiMocks.refreshWorkspaceAuthAccount.mockReset();
    apiMocks.rejectWorkspaceMcpServer.mockReset();
    apiMocks.restoreWorkspaceAuthAccount.mockReset();
    apiMocks.setWorkspaceMcpServerEnabled.mockReset();
    apiMocks.startWorkspaceAuthAccountLogin.mockReset();
    apiMocks.subscribeNotificationsStream.mockClear();
    apiMocks.syncWorkspaceAuthDrafts.mockReset();
    apiMocks.testWorkspaceMcpServer.mockReset();
    apiMocks.updateWorkspaceAuthAccount.mockReset();
    apiMocks.updateWorkspaceMcpServer.mockReset();

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
    apiMocks.fetchWorkspaceIntegrations.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      mcp_servers: [],
      accounts: [],
      counts: {},
    });
    apiMocks.fetchWorkspaceArtifacts.mockResolvedValue([]);
    apiMocks.fetchWorkspaceSearch.mockResolvedValue(null);
    apiMocks.syncWorkspaceAuthDrafts.mockResolvedValue({
      created_drafts: 0,
      created_bindings: 0,
      updated_defaults: 0,
      warnings: [],
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [],
        accounts: [],
        counts: {},
      },
    });
    apiMocks.testWorkspaceMcpServer.mockResolvedValue({
      alias: "demo",
      status: "ok",
      message: "ok",
      tool_count: 0,
      tool_names: [],
    });
    apiMocks.reconnectWorkspaceMcpServer.mockResolvedValue({
      alias: "demo",
      status: "ready",
      message: "ready",
      tool_count: 0,
      tool_names: [],
    });
    apiMocks.createWorkspaceMcpServer.mockResolvedValue({
      alias: "demo",
      type: "remote",
      enabled: true,
      source: "workspace",
      source_path: "/tmp/workspace/.loom/mcp.toml",
      source_label: "Workspace config",
      command: "",
      url: "https://mcp.demo.example",
      cwd: "",
      timeout_seconds: 30,
      oauth_enabled: true,
      trust_state: "review_recommended",
      trust_summary: "Workspace-defined remote server. Review provenance before relying on it.",
      approval_required: true,
      approval_state: "pending",
      runtime_state: "pending_approval",
      resource_id: "resource-mcp-demo",
      auth_provider: "demo",
      auth_state: {
        state: "missing",
        label: "Not connected",
        reason: "",
        storage: "none",
        has_token: false,
        expired: false,
        expires_at: null,
        token_type: null,
        scopes: [],
        profile_id: "",
        account_label: "",
        mode: "",
      },
      effective_account: null,
      bound_profile_ids: [],
      remediation: [],
      flags: [],
    });
    apiMocks.updateWorkspaceMcpServer.mockResolvedValue({
      alias: "demo",
      type: "remote",
      enabled: true,
      source: "workspace",
      source_path: "/tmp/workspace/.loom/mcp.toml",
      source_label: "Workspace config",
      command: "",
      url: "https://mcp.demo.example",
      cwd: "",
      timeout_seconds: 30,
      oauth_enabled: true,
      trust_state: "review_recommended",
      trust_summary: "Workspace-defined remote server. Review provenance before relying on it.",
      approval_required: true,
      approval_state: "pending",
      runtime_state: "pending_approval",
      resource_id: "resource-mcp-demo",
      auth_provider: "demo",
      auth_state: {
        state: "missing",
        label: "Not connected",
        reason: "",
        storage: "none",
        has_token: false,
        expired: false,
        expires_at: null,
        token_type: null,
        scopes: [],
        profile_id: "",
        account_label: "",
        mode: "",
      },
      effective_account: null,
      bound_profile_ids: [],
      remediation: [],
      flags: [],
    });
    apiMocks.createWorkspaceAuthAccount.mockResolvedValue({
      profile_id: "notion_personal",
      provider: "notion",
      account_label: "Notion Personal",
      mode: "oauth2_pkce",
      status: "draft",
      source: "user",
      source_path: "/tmp/.loom/auth.toml",
      mcp_server: "notion",
      token_ref: "keychain://loom/notion/notion_personal/tokens",
      secret_ref: "",
      writable_storage_kind: "keychain",
      auth_state: {
        state: "draft",
        label: "Draft",
        reason: "Complete this draft account before Loom can use it.",
        storage: "profile",
        has_token: false,
        expired: false,
        expires_at: null,
        token_type: null,
        scopes: [],
        profile_id: "notion_personal",
        account_label: "Notion Personal",
        mode: "oauth2_pkce",
      },
      default_selectors: [],
      bound_resource_refs: [],
      used_by_mcp_servers: [],
      effective_for_mcp_servers: [],
      remediation: [],
    });
    apiMocks.updateWorkspaceAuthAccount.mockResolvedValue({
      profile_id: "notion_personal",
      provider: "notion",
      account_label: "Notion Personal",
      mode: "oauth2_pkce",
      status: "draft",
      source: "user",
      source_path: "/tmp/.loom/auth.toml",
      mcp_server: "notion",
      token_ref: "keychain://loom/notion/notion_personal/tokens",
      secret_ref: "",
      writable_storage_kind: "keychain",
      auth_state: {
        state: "draft",
        label: "Draft",
        reason: "Complete this draft account before Loom can use it.",
        storage: "profile",
        has_token: false,
        expired: false,
        expires_at: null,
        token_type: null,
        scopes: [],
        profile_id: "notion_personal",
        account_label: "Notion Personal",
        mode: "oauth2_pkce",
      },
      default_selectors: [],
      bound_resource_refs: [],
      used_by_mcp_servers: [],
      effective_for_mcp_servers: [],
      remediation: [],
    });
    apiMocks.archiveWorkspaceAuthAccount.mockResolvedValue({
      profile_id: "notion_personal",
      provider: "notion",
      account_label: "Notion Personal",
      mode: "oauth2_pkce",
      status: "archived",
      source: "user",
      source_path: "/tmp/.loom/auth.toml",
      mcp_server: "notion",
      token_ref: "keychain://loom/notion/notion_personal/tokens",
      secret_ref: "",
      writable_storage_kind: "keychain",
      auth_state: {
        state: "archived",
        label: "Archived",
        reason: "This account is archived and will not be selected.",
        storage: "profile",
        has_token: false,
        expired: false,
        expires_at: null,
        token_type: null,
        scopes: [],
        profile_id: "notion_personal",
        account_label: "Notion Personal",
        mode: "oauth2_pkce",
      },
      default_selectors: [],
      bound_resource_refs: [],
      used_by_mcp_servers: [],
      effective_for_mcp_servers: [],
      remediation: [],
    });
    apiMocks.restoreWorkspaceAuthAccount.mockResolvedValue({
      profile_id: "notion_personal",
      provider: "notion",
      account_label: "Notion Personal",
      mode: "oauth2_pkce",
      status: "draft",
      source: "user",
      source_path: "/tmp/.loom/auth.toml",
      mcp_server: "notion",
      token_ref: "keychain://loom/notion/notion_personal/tokens",
      secret_ref: "",
      writable_storage_kind: "keychain",
      auth_state: {
        state: "draft",
        label: "Draft",
        reason: "Complete this draft account before Loom can use it.",
        storage: "profile",
        has_token: false,
        expired: false,
        expires_at: null,
        token_type: null,
        scopes: [],
        profile_id: "notion_personal",
        account_label: "Notion Personal",
        mode: "oauth2_pkce",
      },
      default_selectors: [],
      bound_resource_refs: [],
      used_by_mcp_servers: [],
      effective_for_mcp_servers: [],
      remediation: [],
    });
    apiMocks.deleteWorkspaceMcpServer.mockResolvedValue({
      alias: "demo",
      status: "ok",
      message: "deleted",
      tool_count: 0,
      tool_names: [],
    });
    apiMocks.setWorkspaceMcpServerEnabled.mockResolvedValue({
      alias: "demo",
      type: "remote",
      enabled: false,
      source: "workspace",
      source_path: "/tmp/workspace/.loom/mcp.toml",
      source_label: "Workspace config",
      command: "",
      url: "https://mcp.demo.example",
      cwd: "",
      timeout_seconds: 30,
      oauth_enabled: true,
      trust_state: "review_recommended",
      trust_summary: "Workspace-defined remote server. Review provenance before relying on it.",
      approval_required: true,
      approval_state: "pending",
      runtime_state: "disabled",
      resource_id: "resource-mcp-demo",
      auth_provider: "demo",
      auth_state: {
        state: "missing",
        label: "Not connected",
        reason: "",
        storage: "none",
        has_token: false,
        expired: false,
        expires_at: null,
        token_type: null,
        scopes: [],
        profile_id: "",
        account_label: "",
        mode: "",
      },
      effective_account: null,
      bound_profile_ids: [],
      remediation: [],
      flags: [],
    });
    apiMocks.selectWorkspaceMcpAccount.mockResolvedValue({
      alias: "demo",
      status: "ok",
      message: "selected",
      tool_count: 0,
      tool_names: [],
    });
    apiMocks.startWorkspaceAuthAccountLogin.mockResolvedValue({
      flow_id: "flow-1",
      authorization_url: "https://example.com/oauth",
      redirect_uri: "http://127.0.0.1:8765/oauth/callback",
      callback_mode: "loopback",
      expires_at_unix: 123,
      browser_warning: "",
    });
    apiMocks.completeWorkspaceAuthAccountLogin.mockResolvedValue({
      status: "completed",
      message: "done",
      account: null,
      expires_at: null,
      scopes: [],
    });
    apiMocks.refreshWorkspaceAuthAccount.mockResolvedValue({});
    apiMocks.logoutWorkspaceAuthAccount.mockResolvedValue({});
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

  it("patches approval notifications into inbox state before any repair refresh", async () => {
    vi.useFakeTimers();
    let notificationEvent: ((event: any) => void) | undefined;
    apiMocks.subscribeNotificationsStream.mockImplementation(((
      _workspaceId: string,
      onEvent: (event: unknown) => void,
    ) => {
      notificationEvent = onEvent as (event: any) => void;
      return () => {};
    }) as any);

    const { result } = renderHook(() => {
      const [workspaces, setWorkspaces] = useState([{
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        metadata: {},
        is_archived: false,
        sort_order: 0,
        conversation_count: 0,
        run_count: 0,
        active_run_count: 0,
      }] as any);
      return useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces,
        setWorkspaces,
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "overview",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      });
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchApprovals.mockClear();
    apiMocks.fetchWorkspaceOverview.mockClear();

    act(() => {
      notificationEvent?.({
        id: "evt-approval-1",
        stream_id: 11,
        event_type: "approval_requested",
        created_at: "2026-03-27T00:00:00Z",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace 1",
        task_id: "task-1",
        conversation_id: "",
        approval_id: "",
        kind: "task_approval",
        title: "Deploy approval",
        summary: "Need approval",
        payload: {
          proposed_action: "Deploy release",
          reason: "Production deployment",
          tool_name: "deploy",
          risk_level: "high",
          subtask_id: "subtask-1",
        },
      });
    });

    expect(result.current.approvalInbox).toEqual([
      expect.objectContaining({
        id: "task:task-1:subtask-1",
        title: "Deploy release",
        summary: "Production deployment",
        tool_name: "deploy",
      }),
    ]);
    expect(result.current.overview?.pending_approvals_count).toBe(1);

    act(() => {
      vi.advanceTimersByTime(200);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.fetchApprovals).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceOverview).not.toHaveBeenCalled();
  });

  it("reconciles stale pending approval counts from a repair refresh", async () => {
    vi.useFakeTimers();
    let notificationEvent: ((event: any) => void) | undefined;
    apiMocks.subscribeNotificationsStream.mockImplementation(((
      _workspaceId: string,
      onEvent: (event: unknown) => void,
    ) => {
      notificationEvent = onEvent as (event: any) => void;
      return () => {};
    }) as any);

    const { result } = renderHook(() => {
      const [workspaces, setWorkspaces] = useState([{
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        metadata: {},
        is_archived: false,
        sort_order: 0,
        conversation_count: 0,
        run_count: 0,
        active_run_count: 0,
      }] as any);
      return useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces,
        setWorkspaces,
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "overview",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      });
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchApprovals.mockClear();
    apiMocks.fetchWorkspaceOverview.mockClear();
    apiMocks.fetchApprovals.mockResolvedValueOnce([]);

    act(() => {
      notificationEvent?.({
        id: "evt-approval-1",
        stream_id: 11,
        event_type: "approval_requested",
        created_at: "2026-03-27T00:00:00Z",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace 1",
        task_id: "",
        conversation_id: "",
        approval_id: "",
        kind: "task_approval",
        title: "Task approval",
        summary: "Pending approval",
        payload: {},
      });
    });

    expect(result.current.overview?.pending_approvals_count).toBe(1);
    expect(result.current.approvalInbox).toEqual([]);

    act(() => {
      vi.advanceTimersByTime(200);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.fetchApprovals).toHaveBeenCalledWith("workspace-1");
    expect(result.current.overview?.pending_approvals_count).toBe(0);
    expect(result.current.approvalInbox).toEqual([]);
  });

  it("removes back-to-back conversation approval notifications without drifting the count", async () => {
    vi.useFakeTimers();
    let notificationEvent: ((event: any) => void) | undefined;
    apiMocks.subscribeNotificationsStream.mockImplementation(((
      _workspaceId: string,
      onEvent: (event: unknown) => void,
    ) => {
      notificationEvent = onEvent as (event: any) => void;
      return () => {};
    }) as any);

    const { result } = renderHook(() => {
      const [workspaces, setWorkspaces] = useState([{
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        metadata: {},
        is_archived: false,
        sort_order: 0,
        conversation_count: 1,
        run_count: 0,
        active_run_count: 0,
      }] as any);
      return useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces,
        setWorkspaces,
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "overview",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      });
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchApprovals.mockClear();

    act(() => {
      notificationEvent?.({
        id: "evt-conversation-req",
        stream_id: 21,
        event_type: "approval_requested",
        created_at: "2026-04-18T00:00:00Z",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace 1",
        task_id: "",
        conversation_id: "conversation-1",
        approval_id: "approval-1",
        kind: "conversation_approval",
        title: "approval requested",
        summary: "mcp.notion.notion-search",
        payload: {
          tool_name: "mcp.notion.notion-search",
        },
      });
      notificationEvent?.({
        id: "evt-conversation-ok",
        stream_id: 22,
        event_type: "approval_received",
        created_at: "2026-04-18T00:00:01Z",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace 1",
        task_id: "",
        conversation_id: "conversation-1",
        approval_id: "approval-1",
        kind: "conversation_approval",
        title: "approval received",
        summary: "mcp.notion.notion-search",
        payload: {
          tool_name: "mcp.notion.notion-search",
          decision: "approve",
        },
      });
    });

    expect(result.current.overview?.pending_approvals_count).toBe(0);
    expect(result.current.approvalInbox).toEqual([]);

    act(() => {
      vi.advanceTimersByTime(200);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.fetchApprovals).not.toHaveBeenCalled();
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

  it("pushes live run status into overview and workspace counters without a workspace refetch", async () => {
    apiMocks.fetchWorkspaceOverview.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        active_run_count: 1,
        last_activity_at: "2026-03-27T00:01:00Z",
      },
      recent_conversations: [],
      recent_runs: [{
        id: "run-1",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        goal: "Run 1",
        status: "planning",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:01:00Z",
        execution_run_id: "exec-1",
        process_name: "ad-hoc",
        linked_conversation_ids: [],
        changed_files_count: 0,
      }],
      pending_approvals_count: 0,
      counts: {},
    });

    const { result } = renderHook(() => {
      const [workspaces, setWorkspaces] = useState([{
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        metadata: {},
        is_archived: false,
        sort_order: 0,
        active_run_count: 1,
        conversation_count: 0,
        run_count: 1,
        last_activity_at: "2026-03-27T00:01:00Z",
      }] as any);
      return useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "run-1",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces,
        setWorkspaces,
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "overview",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      });
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchWorkspaceOverview.mockClear();

    act(() => {
      result.current.syncRunDetail({
        id: "run-1",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        goal: "Run 1",
        status: "completed",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:05:00Z",
        execution_run_id: "exec-1",
        process_name: "ad-hoc",
        linked_conversation_ids: [],
        changed_files_count: 2,
        task: {},
        task_run: {},
        events_count: 0,
        plan_subtasks: [],
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
      } as any);
    });

    expect(result.current.overview?.recent_runs[0]?.status).toBe("completed");
    expect(result.current.overview?.recent_runs[0]?.changed_files_count).toBe(2);
    expect(result.current.overview?.workspace.active_run_count).toBe(0);
    expect(result.current.selectedWorkspaceSummary?.active_run_count).toBe(0);
    expect(apiMocks.fetchWorkspaceOverview).not.toHaveBeenCalled();
  });

  it("reconciles an existing run row when detail arrives without workspace identity", async () => {
    apiMocks.fetchWorkspaceOverview.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        active_run_count: 1,
        last_activity_at: "2026-03-27T00:01:00Z",
      },
      recent_conversations: [],
      recent_runs: [{
        id: "run-1",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace/scoped-run",
        goal: "Run 1",
        status: "executing",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:01:00Z",
        execution_run_id: "exec-1",
        process_name: "ad-hoc",
        linked_conversation_ids: [],
        changed_files_count: 0,
      }],
      pending_approvals_count: 0,
      counts: {},
    });

    const { result } = renderHook(() => {
      const [workspaces, setWorkspaces] = useState([{
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        metadata: {},
        is_archived: false,
        sort_order: 0,
        active_run_count: 1,
        conversation_count: 0,
        run_count: 1,
        last_activity_at: "2026-03-27T00:01:00Z",
      }] as any);
      return useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "run-1",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces,
        setWorkspaces,
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "overview",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      });
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchWorkspaceOverview.mockClear();

    act(() => {
      result.current.syncRunDetail({
        id: "run-1",
        workspace_id: "",
        workspace_path: "/tmp/workspace/scoped-run",
        goal: "Run 1",
        status: "paused",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:06:00Z",
        execution_run_id: "exec-1",
        process_name: "ad-hoc",
        linked_conversation_ids: [],
        changed_files_count: 1,
        task: {},
        task_run: {},
        events_count: 0,
        plan_subtasks: [],
      } as any);
    });

    expect(result.current.overview?.recent_runs[0]?.status).toBe("paused");
    expect(result.current.overview?.recent_runs[0]?.changed_files_count).toBe(1);
    expect(result.current.overview?.workspace.active_run_count).toBe(0);
    expect(result.current.selectedWorkspaceSummary?.active_run_count).toBe(0);
    expect(apiMocks.fetchWorkspaceOverview).not.toHaveBeenCalled();
  });

  it("resyncs a drifted selected workspace active run count from the updated overview", async () => {
    apiMocks.fetchWorkspaceOverview.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        active_run_count: 1,
        run_count: 1,
        last_activity_at: "2026-03-27T00:01:00Z",
      },
      recent_conversations: [],
      recent_runs: [{
        id: "run-1",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        goal: "Run 1",
        status: "executing",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:01:00Z",
        execution_run_id: "exec-1",
        process_name: "ad-hoc",
        linked_conversation_ids: [],
        changed_files_count: 0,
      }],
      pending_approvals_count: 0,
      counts: {},
    });

    const { result } = renderHook(() => {
      const [workspaces, setWorkspaces] = useState([{
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        metadata: {},
        is_archived: false,
        sort_order: 0,
        active_run_count: 2,
        conversation_count: 0,
        run_count: 1,
        last_activity_at: "2026-03-27T00:01:00Z",
      }] as any);
      return useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "run-1",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces,
        setWorkspaces,
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "overview",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      });
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    act(() => {
      result.current.syncRunDetail({
        id: "run-1",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        goal: "Run 1",
        status: "planning",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:02:00Z",
        execution_run_id: "exec-1",
        process_name: "ad-hoc",
        linked_conversation_ids: [],
        changed_files_count: 0,
        task: {},
        task_run: {},
        events_count: 0,
        plan_subtasks: [],
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
      } as any);
    });

    expect(result.current.overview?.workspace.active_run_count).toBe(1);
    expect(result.current.selectedWorkspaceSummary?.active_run_count).toBe(1);
  });

  it("upserts conversation summaries locally without a workspace refetch", async () => {
    apiMocks.fetchWorkspaceOverview.mockResolvedValue({
      workspace: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        conversation_count: 0,
        last_activity_at: "2026-03-27T00:00:00Z",
      },
      recent_conversations: [],
      recent_runs: [],
      pending_approvals_count: 0,
      counts: {},
    });

    const { result } = renderHook(() => {
      const [workspaces, setWorkspaces] = useState([{
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        metadata: {},
        is_archived: false,
        sort_order: 0,
        conversation_count: 0,
        run_count: 0,
        active_run_count: 0,
        last_activity_at: "2026-03-27T00:00:00Z",
      }] as any);
      return useWorkspace({
        selectedWorkspaceId: "workspace-1",
        selectedConversationId: "",
        selectedRunId: "",
        setSelectedWorkspaceId: vi.fn(),
        showArchivedWorkspaces: false,
        setShowArchivedWorkspaces: vi.fn(),
        createParentPath: "/tmp",
        setCreateParentPath: vi.fn(),
        workspaces,
        setWorkspaces,
        runtime: null,
        setError: vi.fn(),
        setNotice: vi.fn(),
        activeTab: "threads",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      });
    });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchWorkspaceOverview.mockClear();

    act(() => {
      result.current.syncConversationSummary({
        id: "conversation-1",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        model_name: "gpt-5.4-mini",
        title: "Fresh thread",
        turn_count: 1,
        total_tokens: 42,
        last_active_at: "2026-03-27T00:05:00Z",
        started_at: "2026-03-27T00:05:00Z",
        is_active: true,
        linked_run_ids: [],
      }, {
        incrementCount: true,
        processing: true,
      });
    });

    expect(result.current.overview?.recent_conversations[0]).toEqual(expect.objectContaining({
      id: "conversation-1",
      title: "Fresh thread",
      is_active: true,
    }));
    expect(result.current.overview?.workspace.conversation_count).toBe(1);
    expect(result.current.selectedWorkspaceSummary?.conversation_count).toBe(1);
    expect(apiMocks.fetchWorkspaceOverview).not.toHaveBeenCalled();
  });

  it("preserves dirty workspace drafts and file-tree mode across unrelated summary churn", async () => {
    const { result, rerender } = renderHook(
      ({ workspaces }: { workspaces: any[] }) =>
        useWorkspace({
          selectedWorkspaceId: "workspace-1",
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
          activeTab: "files",
          setActiveTab: vi.fn(),
          setSelectedConversationId: vi.fn(),
          setSelectedRunId: vi.fn(),
        }),
      {
        initialProps: {
          workspaces: [{
            id: "workspace-1",
            canonical_path: "/tmp/workspace",
            display_name: "Workspace 1",
            metadata: {
              note: "Original note",
              tags: ["alpha"],
            },
            is_archived: false,
            sort_order: 0,
            conversation_count: 0,
            run_count: 0,
            active_run_count: 0,
            last_activity_at: "2026-03-27T00:00:00Z",
          }],
        },
      },
    );

    act(() => {
      result.current.setWorkspaceNameDraft("Locally edited name");
      result.current.setWorkspaceNoteDraft("Locally edited note");
      result.current.setWorkspaceTagsDraft("alpha, beta");
      result.current.setWorkspaceFileTreeMode("recent");
    });

    rerender({
      workspaces: [{
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
        metadata: {
          note: "Original note",
          tags: ["alpha"],
        },
        is_archived: false,
        sort_order: 0,
        conversation_count: 1,
        run_count: 0,
        active_run_count: 0,
        last_activity_at: "2026-03-27T00:05:00Z",
      }],
    });

    expect(result.current.workspaceNameDraft).toBe("Locally edited name");
    expect(result.current.workspaceNoteDraft).toBe("Locally edited note");
    expect(result.current.workspaceTagsDraft).toBe("alpha, beta");
    expect(result.current.workspaceFileTreeMode).toBe("recent");
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

  it("fetches integrations lazily only when the integrations tab becomes visible", async () => {
    const { rerender } = renderHook(
      ({ activeTab }: { activeTab: "threads" | "integrations" }) =>
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

    expect(apiMocks.fetchWorkspaceIntegrations).not.toHaveBeenCalled();

    rerender({ activeTab: "integrations" });

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(apiMocks.fetchWorkspaceIntegrations).toHaveBeenCalledWith("workspace-1");
    expect(apiMocks.fetchWorkspaceInventory).not.toHaveBeenCalled();
    expect(apiMocks.fetchWorkspaceSettings).not.toHaveBeenCalled();
  });

  it("allows launcher flows to force-refresh workspace artifacts without reloading the full surface", async () => {
    const { result } = renderHook(() =>
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

    expect(apiMocks.fetchWorkspaceArtifacts).toHaveBeenCalledTimes(1);

    apiMocks.fetchWorkspaceArtifacts.mockClear();

    await act(async () => {
      await result.current.refreshWorkspaceArtifacts("workspace-1");
    });

    expect(apiMocks.fetchWorkspaceArtifacts).not.toHaveBeenCalled();

    await act(async () => {
      await result.current.refreshWorkspaceArtifacts("workspace-1", { force: true });
    });

    expect(apiMocks.fetchWorkspaceArtifacts).toHaveBeenCalledTimes(1);
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

  it("selects an existing account for an MCP server and refreshes integrations", async () => {
    const setError = vi.fn();
    const setNotice = vi.fn();
    const { result } = renderHook(() =>
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
        setError,
        setNotice,
        activeTab: "integrations",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      }),
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchWorkspaceIntegrations.mockClear();

    await act(async () => {
      await result.current.handleSelectIntegrationAccountForServer(
        "notion",
        "notion_marketing",
      );
    });

    expect(apiMocks.selectWorkspaceMcpAccount).toHaveBeenCalledWith(
      "workspace-1",
      "notion",
      "notion_marketing",
    );
    expect(apiMocks.fetchWorkspaceIntegrations).toHaveBeenCalledWith("workspace-1");
    expect(setNotice).toHaveBeenCalledWith("selected");
    expect(setError).toHaveBeenCalledWith("");
  });

  it("creates an MCP server and refreshes integrations", async () => {
    const setError = vi.fn();
    const setNotice = vi.fn();
    const { result } = renderHook(() =>
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
        setError,
        setNotice,
        activeTab: "integrations",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      }),
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchWorkspaceIntegrations.mockClear();

    await act(async () => {
      await result.current.handleCreateIntegrationServer({
        alias: "demo",
        type: "remote",
        url: "https://mcp.demo.example",
        oauth_enabled: true,
        oauth_scopes: ["read"],
      });
    });

    expect(apiMocks.createWorkspaceMcpServer).toHaveBeenCalledWith(
      "workspace-1",
      expect.objectContaining({
        alias: "demo",
        type: "remote",
        url: "https://mcp.demo.example",
      }),
    );
    expect(apiMocks.fetchWorkspaceIntegrations).toHaveBeenCalledWith("workspace-1");
    expect(setNotice).toHaveBeenCalledWith("Added MCP server demo.");
    expect(setError).toHaveBeenCalledWith("");
  });

  it("creates an auth account and refreshes integrations", async () => {
    const setError = vi.fn();
    const setNotice = vi.fn();
    const { result } = renderHook(() =>
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
        setError,
        setNotice,
        activeTab: "integrations",
        setActiveTab: vi.fn(),
        setSelectedConversationId: vi.fn(),
        setSelectedRunId: vi.fn(),
      }),
    );

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });

    apiMocks.fetchWorkspaceIntegrations.mockClear();

    await act(async () => {
      await result.current.handleCreateIntegrationAccount({
        profile_id: "notion_personal",
        provider: "notion",
        mode: "oauth2_pkce",
        account_label: "Notion Personal",
        mcp_server: "notion",
      });
    });

    expect(apiMocks.createWorkspaceAuthAccount).toHaveBeenCalledWith(
      "workspace-1",
      expect.objectContaining({
        profile_id: "notion_personal",
        provider: "notion",
      }),
    );
    expect(apiMocks.fetchWorkspaceIntegrations).toHaveBeenCalledWith("workspace-1");
    expect(setNotice).toHaveBeenCalledWith("Added account Notion Personal.");
    expect(setError).toHaveBeenCalledWith("");
  });
});
