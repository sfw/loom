import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import IntegrationsTab from "./IntegrationsTab";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  shallowEqual: (left: unknown, right: unknown) => left === right,
  useAppActions: () => mockApp,
  useAppSelector: (selector: (state: any) => unknown) => selector(mockApp),
}));

describe("IntegrationsTab", () => {
  it("does not crash when integrations load after the first render", () => {
    mockApp = {
      filteredIntegrationServers: [],
      filteredAccounts: [],
      integrations: null,
      integrationIntent: null,
      loadingOverview: true,
      clearIntegrationIntent: vi.fn(),
      handleCompleteIntegrationAccountLogin: vi.fn(),
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount: vi.fn(),
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer: vi.fn(),
      handleStartIntegrationAccountLogin: vi.fn(),
      handleSetIntegrationApproval: vi.fn(),
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    const view = render(<IntegrationsTab />);
    expect(screen.getByText("Loading MCP servers and connected accounts...")).toBeInTheDocument();

    mockApp = {
      ...mockApp,
      loadingOverview: false,
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [],
        accounts: [],
        counts: {
          mcp_servers: 0,
          accounts: 0,
        },
      },
    };

    view.rerender(<IntegrationsTab />);
    expect(screen.getByText("Integrations")).toBeInTheDocument();
  });

  it("renders management-grade MCP and account state", () => {
    mockApp = {
      filteredIntegrationServers: [
        {
          alias: "notion",
          type: "remote",
          enabled: true,
          source: "workspace",
          source_path: "/tmp/workspace/.loom/mcp.toml",
          source_label: "Workspace config",
          command: "",
          args: [],
          url: "https://mcp.notion.example",
          fallback_sse_url: "",
          cwd: "",
          timeout_seconds: 30,
          oauth_enabled: true,
          oauth_scopes: ["read"],
          allow_insecure_http: false,
          allow_private_network: false,
          trust_state: "review_recommended",
          trust_summary: "Workspace-defined remote server. Review provenance before relying on it.",
          approval_required: true,
          approval_state: "pending",
          runtime_state: "pending_approval",
          resource_id: "resource-mcp-notion",
          auth_provider: "notion",
          auth_state: {
            state: "ready",
            label: "Connected",
            reason: "",
            storage: "profile_token_ref",
            has_token: true,
            expired: false,
            expires_at: null,
            token_type: "Bearer",
            scopes: [],
            profile_id: "notion_marketing",
            account_label: "Notion Marketing",
            mode: "oauth2_pkce",
          },
          effective_account: {
            profile_id: "notion_marketing",
            provider: "notion",
            account_label: "Notion Marketing",
            mode: "oauth2_pkce",
            status: "ready",
            source: "user",
            source_path: "/tmp/.loom/auth.toml",
            routing_reason: "selected_resource_default",
            auth_state: {
              state: "ready",
              label: "Connected",
              reason: "",
              storage: "profile_token_ref",
              has_token: true,
              expired: false,
              expires_at: null,
              token_type: "Bearer",
              scopes: [],
              profile_id: "notion_marketing",
              account_label: "Notion Marketing",
              mode: "oauth2_pkce",
            },
          },
          bound_profile_ids: ["notion_marketing"],
          remediation: ["Review this workspace-defined remote server before connecting an account."],
          flags: ["workspace_remote"],
        },
      ],
      filteredAccounts: [
        {
          profile_id: "notion_marketing",
          provider: "notion",
          account_label: "Notion Marketing",
          mode: "oauth2_pkce",
          status: "ready",
          source: "user",
          source_path: "/tmp/.loom/auth.toml",
          mcp_server: "notion",
          token_ref: "${LOOM_TEST_NOTION_TOKEN}",
          secret_ref: "",
          writable_storage_kind: "keychain",
          auth_state: {
            state: "ready",
            label: "Connected",
            reason: "",
            storage: "profile_token_ref",
            has_token: true,
            expired: false,
            expires_at: null,
            token_type: "Bearer",
            scopes: [],
            profile_id: "notion_marketing",
            account_label: "Notion Marketing",
            mode: "oauth2_pkce",
          },
          default_selectors: ["resource-mcp-notion"],
          bound_resource_refs: ["mcp:notion"],
          used_by_mcp_servers: ["notion"],
          effective_for_mcp_servers: ["notion"],
          remediation: [],
        },
      ],
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [],
        accounts: [],
        counts: {
          mcp_servers: 1,
          accounts: 1,
          connected_mcp_servers: 1,
          attention_mcp_servers: 0,
        },
      },
      integrationIntent: null,
      loadingOverview: false,
      clearIntegrationIntent: vi.fn(),
      handleCompleteIntegrationAccountLogin: vi.fn(),
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount: vi.fn(),
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer: vi.fn(),
      handleStartIntegrationAccountLogin: vi.fn(),
      handleSetIntegrationApproval: vi.fn(),
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    render(<IntegrationsTab />);

    expect(screen.getByText("Integrations")).toBeInTheDocument();
    expect(screen.getByText("Issues")).toBeInTheDocument();
    expect(screen.getByText("Add local server")).toBeInTheDocument();
    expect(screen.getByText("Add account")).toBeInTheDocument();
    expect(screen.getAllByText("Notion Marketing").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("Workspace config")).toBeInTheDocument();
    expect(screen.getByText("Effective for notion")).toBeInTheDocument();
    expect(screen.getByText("Default Account")).toBeInTheDocument();
    expect(screen.getByText("Create another account")).toBeInTheDocument();
    expect(screen.getByText("Verify connection")).toBeInTheDocument();
    expect(screen.getByText("Approve server")).toBeInTheDocument();
    expect(screen.getByText("Default for:")).toBeInTheDocument();
  });

  it("responds to palette integration intents", () => {
    const clearIntegrationIntent = vi.fn();
    mockApp = {
      filteredIntegrationServers: [],
      filteredAccounts: [],
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
      integrationIntent: {
        kind: "add_remote_server",
        requestedAt: Date.now(),
      },
      loadingOverview: false,
      clearIntegrationIntent,
      handleCompleteIntegrationAccountLogin: vi.fn(),
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount: vi.fn(),
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer: vi.fn(),
      handleStartIntegrationAccountLogin: vi.fn(),
      handleSetIntegrationApproval: vi.fn(),
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    render(<IntegrationsTab />);

    expect(screen.getByText("Add MCP Server")).toBeInTheDocument();
    expect(clearIntegrationIntent).toHaveBeenCalled();
  });

  it("offers inline remediation actions in the issues panel", async () => {
    const user = userEvent.setup();
    const handleSetIntegrationApproval = vi.fn();
    mockApp = {
      filteredIntegrationServers: [
        {
          alias: "notion",
          type: "remote",
          enabled: true,
          source: "workspace",
          source_path: "/tmp/workspace/.loom/mcp.toml",
          source_label: "Workspace config",
          command: "",
          args: [],
          url: "https://mcp.notion.example",
          fallback_sse_url: "",
          cwd: "",
          timeout_seconds: 30,
          oauth_enabled: true,
          oauth_scopes: ["read"],
          allow_insecure_http: false,
          allow_private_network: false,
          trust_state: "review_recommended",
          trust_summary: "Workspace-defined remote server. Review provenance before relying on it.",
          approval_required: true,
          approval_state: "pending",
          runtime_state: "pending_approval",
          resource_id: "resource-mcp-notion",
          auth_provider: "notion",
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
          remediation: ["Review this workspace-defined remote server before connecting an account."],
          flags: ["workspace_remote"],
        },
      ],
      filteredAccounts: [],
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [],
        accounts: [],
        counts: {
          attention_mcp_servers: 1,
          pending_approval_mcp_servers: 1,
        },
      },
      integrationIntent: null,
      loadingOverview: false,
      clearIntegrationIntent: vi.fn(),
      handleCompleteIntegrationAccountLogin: vi.fn(),
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount: vi.fn(),
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer: vi.fn(),
      handleStartIntegrationAccountLogin: vi.fn(),
      handleSetIntegrationApproval,
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    render(<IntegrationsTab />);

    await user.click(screen.getByText("Approve now"));

    expect(handleSetIntegrationApproval).toHaveBeenCalledWith("notion", "approved");
  });

  it("labels reconnect as auth refresh for expired legacy MCP tokens", () => {
    mockApp = {
      filteredIntegrationServers: [
        {
          alias: "notion",
          type: "remote",
          enabled: true,
          source: "user",
          source_path: "/tmp/.loom/mcp.toml",
          source_label: "User config",
          command: "",
          args: [],
          url: "https://mcp.notion.com/mcp",
          fallback_sse_url: "",
          cwd: "",
          timeout_seconds: 30,
          oauth_enabled: true,
          oauth_scopes: ["read"],
          allow_insecure_http: false,
          allow_private_network: false,
          trust_state: "trusted",
          trust_summary: "User-defined server.",
          approval_required: false,
          approval_state: "not_required",
          runtime_state: "needs_refresh",
          resource_id: "resource-mcp-notion",
          auth_provider: "notion",
          auth_state: {
            state: "expired",
            label: "Expired",
            reason: "",
            storage: "legacy_alias_store",
            has_token: true,
            expired: true,
            expires_at: null,
            token_type: "Bearer",
            scopes: ["read"],
            profile_id: "",
            account_label: "",
            mode: "",
          },
          effective_account: null,
          bound_profile_ids: [],
          remediation: ["Reconnect or refresh this account."],
          flags: ["legacy_auth_storage"],
        },
      ],
      filteredAccounts: [],
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [],
        accounts: [],
        counts: {
          mcp_servers: 1,
          accounts: 0,
          attention_mcp_servers: 1,
        },
      },
      integrationIntent: null,
      loadingOverview: false,
      clearIntegrationIntent: vi.fn(),
      handleCompleteIntegrationAccountLogin: vi.fn(),
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount: vi.fn(),
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer: vi.fn(),
      handleStartIntegrationAccountLogin: vi.fn(),
      handleSetIntegrationApproval: vi.fn(),
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    render(<IntegrationsTab />);

    expect(screen.getByText("Refresh auth")).toBeInTheDocument();
  });

  it("lets account creation pick a linked server and set it as the default", async () => {
    const user = userEvent.setup();
    const handleCreateIntegrationAccount = vi.fn(async () => true);
    const handleSelectIntegrationAccountForServer = vi.fn(async () => ({
      alias: "notion",
      status: "ok",
      message: "Using account Notion Personal for MCP server notion.",
      tool_count: 0,
      tool_names: [],
    }));

    const notionServer = {
      alias: "notion",
      type: "remote",
      enabled: true,
      source: "user",
      source_path: "/tmp/.loom/mcp.toml",
      source_label: "User config",
      command: "",
      args: [],
      url: "https://mcp.notion.example",
      fallback_sse_url: "",
      cwd: "",
      timeout_seconds: 30,
      oauth_enabled: true,
      oauth_scopes: ["read"],
      allow_insecure_http: false,
      allow_private_network: false,
      trust_state: "trusted",
      trust_summary: "User-defined server.",
      approval_required: false,
      approval_state: "not_required",
      runtime_state: "needs_auth",
      resource_id: "resource-mcp-notion",
      auth_provider: "notion",
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
      remediation: ["Connect an account for this server."],
      flags: [],
    };

    mockApp = {
      filteredIntegrationServers: [notionServer],
      filteredAccounts: [],
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [notionServer],
        accounts: [],
        counts: {
          mcp_servers: 1,
          accounts: 0,
        },
      },
      integrationIntent: null,
      loadingOverview: false,
      clearIntegrationIntent: vi.fn(),
      handleCompleteIntegrationAccountLogin: vi.fn(),
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount,
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer,
      handleStartIntegrationAccountLogin: vi.fn(),
      handleSetIntegrationApproval: vi.fn(),
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    render(<IntegrationsTab />);

    await user.click(screen.getByText("Add account"));
    await user.type(screen.getByLabelText("Profile id"), "notion_personal");
    await user.type(screen.getByLabelText("Account label"), "Notion Personal");
    await user.type(screen.getByLabelText("Provider"), "notion");
    await user.selectOptions(screen.getByLabelText("Linked server (optional)"), "notion");
    await user.selectOptions(screen.getByLabelText("Default for server (optional)"), "notion");
    await user.click(screen.getByText("Save account"));

    expect(handleCreateIntegrationAccount).toHaveBeenCalledWith(expect.objectContaining({
      profile_id: "notion_personal",
      provider: "notion",
      mcp_server: "notion",
    }));
    expect(handleSelectIntegrationAccountForServer).toHaveBeenCalledWith(
      "notion",
      "notion_personal",
    );
  });

  it("hides archived accounts by default and reveals them on demand", async () => {
    const user = userEvent.setup();
    mockApp = {
      filteredIntegrationServers: [],
      filteredAccounts: [
        {
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
            reason: "",
            storage: "profile_token_ref",
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
          remediation: ["Complete this draft account before using it."],
        },
        {
          profile_id: "legacy_archived",
          provider: "notion",
          account_label: "Legacy Archived",
          mode: "oauth2_pkce",
          status: "archived",
          source: "user",
          source_path: "/tmp/.loom/auth.toml",
          mcp_server: "notion",
          token_ref: "keychain://loom/notion/legacy_archived/tokens",
          secret_ref: "",
          writable_storage_kind: "keychain",
          auth_state: {
            state: "archived",
            label: "Archived",
            reason: "",
            storage: "profile_token_ref",
            has_token: false,
            expired: false,
            expires_at: null,
            token_type: null,
            scopes: [],
            profile_id: "legacy_archived",
            account_label: "Legacy Archived",
            mode: "oauth2_pkce",
          },
          default_selectors: [],
          bound_resource_refs: [],
          used_by_mcp_servers: [],
          effective_for_mcp_servers: [],
          remediation: ["Restore or replace this archived account."],
        },
      ],
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [],
        accounts: [],
        counts: {
          accounts: 2,
        },
      },
      integrationIntent: null,
      loadingOverview: false,
      clearIntegrationIntent: vi.fn(),
      handleCompleteIntegrationAccountLogin: vi.fn(),
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount: vi.fn(),
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer: vi.fn(),
      handleStartIntegrationAccountLogin: vi.fn(),
      handleSetIntegrationApproval: vi.fn(),
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    render(<IntegrationsTab />);

    expect(screen.getAllByText("Notion Personal").length).toBeGreaterThan(0);
    expect(screen.queryByText("Legacy Archived")).not.toBeInTheDocument();
    expect(screen.getByText("Show archived (1)")).toBeInTheDocument();

    await user.click(screen.getByText("Show archived (1)"));

    expect(screen.getAllByText("Legacy Archived").length).toBeGreaterThan(0);
    expect(screen.getByText("Hide archived")).toBeInTheDocument();
  });

  it("auto-waits for loopback account logins instead of asking for manual callback input", async () => {
    const user = userEvent.setup();
    const handleStartIntegrationAccountLogin = vi.fn(async () => ({
      flow_id: "flow-1",
      authorization_url: "https://auth.example/authorize",
      redirect_uri: "http://127.0.0.1:8765/oauth/callback",
      callback_mode: "loopback",
      expires_at_unix: 1_900_000_000,
      browser_warning: "",
    }));
    const handleCompleteIntegrationAccountLogin = vi.fn()
      .mockResolvedValueOnce({
        status: "pending",
        message: "Waiting for OAuth callback.",
      })
      .mockResolvedValueOnce({
        status: "completed",
        message: "Connected account notion_personal.",
      });

    mockApp = {
      filteredIntegrationServers: [],
      filteredAccounts: [
        {
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
            reason: "",
            storage: "profile_token_ref",
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
          effective_for_mcp_servers: ["notion"],
          remediation: ["Complete this draft account before using it."],
        },
      ],
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [],
        accounts: [],
        counts: {
          accounts: 1,
        },
      },
      integrationIntent: null,
      loadingOverview: false,
      clearIntegrationIntent: vi.fn(),
      handleCompleteIntegrationAccountLogin,
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount: vi.fn(),
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer: vi.fn(),
      handleStartIntegrationAccountLogin,
      handleSetIntegrationApproval: vi.fn(),
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    render(<IntegrationsTab />);

    await user.click(screen.getByText("Connect account"));

    expect(handleStartIntegrationAccountLogin).toHaveBeenCalledWith("notion_personal");
    expect(screen.getByText("Waiting for Browser Callback")).toBeInTheDocument();
    expect(screen.getByText("Enter callback manually")).toBeInTheDocument();
    expect(screen.queryByText("Paste callback URL or authorization code")).not.toBeInTheDocument();

    await waitFor(() => {
      expect(handleCompleteIntegrationAccountLogin).toHaveBeenCalledTimes(1);
    }, { timeout: 2500 });

    await waitFor(() => {
      expect(handleCompleteIntegrationAccountLogin).toHaveBeenCalledTimes(2);
    }, { timeout: 4500 });
    await waitFor(() => {
      expect(screen.queryByText("Waiting for Browser Callback")).not.toBeInTheDocument();
    }, { timeout: 4500 });
  });

  it("keeps manual callback entry for non-loopback account logins", async () => {
    const user = userEvent.setup();
    const handleStartIntegrationAccountLogin = vi.fn(async () => ({
      flow_id: "flow-manual",
      authorization_url: "https://auth.example/authorize",
      redirect_uri: "urn:ietf:wg:oauth:2.0:oob",
      callback_mode: "manual",
      expires_at_unix: 1_900_000_000,
      browser_warning: "",
    }));
    const handleCompleteIntegrationAccountLogin = vi.fn();

    mockApp = {
      filteredIntegrationServers: [],
      filteredAccounts: [
        {
          profile_id: "notion_manual",
          provider: "notion",
          account_label: "Manual Notion",
          mode: "oauth2_pkce",
          status: "draft",
          source: "user",
          source_path: "/tmp/.loom/auth.toml",
          mcp_server: "notion",
          token_ref: "",
          secret_ref: "",
          writable_storage_kind: "keychain",
          auth_state: {
            state: "draft",
            label: "Draft",
            reason: "",
            storage: "profile_token_ref",
            has_token: false,
            expired: false,
            expires_at: null,
            token_type: null,
            scopes: [],
            profile_id: "notion_manual",
            account_label: "Manual Notion",
            mode: "oauth2_pkce",
          },
          default_selectors: [],
          bound_resource_refs: [],
          used_by_mcp_servers: [],
          effective_for_mcp_servers: [],
          remediation: ["Complete this draft account before using it."],
        },
      ],
      integrations: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace 1",
        },
        mcp_servers: [],
        accounts: [],
        counts: {
          accounts: 1,
        },
      },
      integrationIntent: null,
      loadingOverview: false,
      clearIntegrationIntent: vi.fn(),
      handleCompleteIntegrationAccountLogin,
      handleCreateIntegrationServer: vi.fn(),
      handleCreateIntegrationAccount: vi.fn(),
      handleDeleteIntegrationServer: vi.fn(),
      handleLogoutIntegrationAccount: vi.fn(),
      handleReconnectIntegrationServer: vi.fn(),
      handleRefreshIntegrationAccount: vi.fn(),
      handleArchiveIntegrationAccount: vi.fn(),
      handleRestoreIntegrationAccount: vi.fn(),
      handleSetIntegrationEnabled: vi.fn(),
      handleSelectIntegrationAccountForServer: vi.fn(),
      handleStartIntegrationAccountLogin,
      handleSetIntegrationApproval: vi.fn(),
      handleSyncIntegrationDrafts: vi.fn(),
      handleTestIntegrationServer: vi.fn(),
      handleUpdateIntegrationAccount: vi.fn(),
      handleUpdateIntegrationServer: vi.fn(),
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace 1",
      },
      workspaceSearchQuery: "",
    };

    render(<IntegrationsTab />);

    await user.click(screen.getByText("Connect account"));

    expect(screen.getByText("Finish Account Connection")).toBeInTheDocument();
    expect(screen.getByText("Paste callback URL or authorization code")).toBeInTheDocument();
    expect(handleCompleteIntegrationAccountLogin).not.toHaveBeenCalled();
  });
});
