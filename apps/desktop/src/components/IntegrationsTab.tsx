import { useEffect, useState } from "react";

import {
  Cable,
  CheckCircle2,
  KeyRound,
  LockKeyhole,
  Archive,
  PencilLine,
  Plus,
  Power,
  Server,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  TriangleAlert,
  UserCircle2,
} from "lucide-react";

import {
  shallowEqual,
  useAppActions,
  useAppSelector,
} from "@/context/AppContext";
import { cn } from "@/lib/utils";

function runtimeTone(runtimeState: string): string {
  switch (runtimeState) {
    case "ready":
      return "bg-emerald-500/10 text-emerald-300 border-emerald-500/20";
    case "needs_refresh":
      return "bg-amber-500/10 text-amber-300 border-amber-500/20";
    case "draft":
    case "needs_auth":
      return "bg-amber-500/10 text-amber-300 border-amber-500/20";
    case "disabled":
      return "bg-zinc-800 text-zinc-400 border-zinc-700";
    default:
      return "bg-zinc-800 text-zinc-400 border-zinc-700";
  }
}

function trustTone(trustState: string): string {
  switch (trustState) {
    case "trusted":
      return "text-emerald-300";
    case "review_recommended":
      return "text-amber-300";
    case "legacy":
      return "text-zinc-400";
    default:
      return "text-zinc-300";
  }
}

function approvalTone(approvalState: string): string {
  switch (approvalState) {
    case "approved":
      return "bg-emerald-500/10 text-emerald-300 border-emerald-500/20";
    case "rejected":
      return "bg-rose-500/10 text-rose-300 border-rose-500/20";
    case "pending":
      return "bg-amber-500/10 text-amber-300 border-amber-500/20";
    default:
      return "bg-zinc-800 text-zinc-400 border-zinc-700";
  }
}

function formatExpiry(epochSeconds: number | null): string {
  if (!epochSeconds) {
    return "No expiry reported";
  }
  return new Date(epochSeconds * 1000).toLocaleString();
}

function humanizeState(value: string): string {
  return value.replace(/_/g, " ");
}

function serverReconnectLabel(server: {
  auth_state: { storage: string; expired: boolean };
  runtime_state: string;
}): string {
  if (server.auth_state.storage === "legacy_alias_store" && server.auth_state.expired) {
    return "Refresh auth";
  }
  if (server.runtime_state === "needs_refresh") {
    return "Refresh and reconnect";
  }
  return "Reconnect";
}

function serverTestLabel(server: {
  type: string;
}): string {
  return server.type === "remote" ? "Verify connection" : "Test server";
}

type ServerEditorState = {
  mode: "create" | "edit";
  originalAlias: string;
  alias: string;
  type: "local" | "remote";
  command: string;
  argsText: string;
  url: string;
  fallbackSseUrl: string;
  oauthEnabled: boolean;
  oauthScopesText: string;
  allowInsecureHttp: boolean;
  allowPrivateNetwork: boolean;
  cwd: string;
  timeoutSeconds: string;
  enabled: boolean;
};

type AccountEditorState = {
  mode: "create" | "edit";
  originalProfileId: string;
  profileId: string;
  provider: string;
  modeValue: string;
  accountLabel: string;
  mcpServer: string;
  scopesText: string;
  startLoginAfterSave: boolean;
  testServerAlias: string;
};

function parseLineList(value: string): string[] {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseScopeList(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export default function IntegrationsTab() {
  const {
    filteredAccounts,
    filteredIntegrationServers,
    integrationIntent,
    integrations,
    loadingOverview,
    selectedWorkspaceSummary,
    workspaceSearchQuery,
  } = useAppSelector((state) => ({
    filteredAccounts: state.filteredAccounts,
    filteredIntegrationServers: state.filteredIntegrationServers,
    integrationIntent: state.integrationIntent,
    integrations: state.integrations,
    loadingOverview: state.loadingOverview,
    selectedWorkspaceSummary: state.selectedWorkspaceSummary,
    workspaceSearchQuery: state.workspaceSearchQuery,
  }), shallowEqual);
  const {
    handleCreateIntegrationServer,
    handleCreateIntegrationAccount,
    clearIntegrationIntent,
    handleDeleteIntegrationServer,
    handleCompleteIntegrationAccountLogin,
    handleArchiveIntegrationAccount,
    handleLogoutIntegrationAccount,
    handleReconnectIntegrationServer,
    handleRefreshIntegrationAccount,
    handleRestoreIntegrationAccount,
    handleSelectIntegrationAccountForServer,
    handleSetIntegrationEnabled,
    handleSetIntegrationApproval,
    handleStartIntegrationAccountLogin,
    handleSyncIntegrationDrafts,
    handleTestIntegrationServer,
    handleUpdateIntegrationAccount,
    handleUpdateIntegrationServer,
  } = useAppActions();
  const [pendingLogin, setPendingLogin] = useState<{
    authorizationUrl: string;
    callbackInput: string;
    callbackMode: string;
    expiresAt: number;
    flowId: string;
    profileId: string;
    redirectUri: string;
    submitting: boolean;
    testServerAlias: string;
  } | null>(null);
  const [serverSelections, setServerSelections] = useState<Record<string, string>>({});
  const [serverEditor, setServerEditor] = useState<ServerEditorState | null>(null);
  const [accountEditor, setAccountEditor] = useState<AccountEditorState | null>(null);
  const [showIssuesOnly, setShowIssuesOnly] = useState(false);

  useEffect(() => {
    if (!integrationIntent) {
      return;
    }
    if (integrationIntent.kind === "add_local_server") {
      openCreateServerEditor("local");
      setShowIssuesOnly(false);
    } else if (integrationIntent.kind === "add_remote_server") {
      openCreateServerEditor("remote");
      setShowIssuesOnly(false);
    } else if (integrationIntent.kind === "create_account") {
      openCreateAccountEditor({
        mode: "oauth2_pkce",
        startLoginAfterSave: true,
      });
      setShowIssuesOnly(false);
    } else if (integrationIntent.kind === "focus_issues") {
      setShowIssuesOnly(true);
    }
    clearIntegrationIntent();
  }, [clearIntegrationIntent, integrationIntent]);

  if (!selectedWorkspaceSummary) {
    return (
      <div className="flex h-full items-center justify-center bg-[#09090b] text-zinc-400">
        Select a workspace to inspect integrations.
      </div>
    );
  }

  if (!integrations) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 bg-[#09090b] px-6 text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[#8a9a7b]/10 text-[#a3b396]">
          <Cable size={22} />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-zinc-100">Integrations</h1>
          <p className="mt-1 text-sm text-zinc-500">
            {loadingOverview
              ? "Loading MCP servers and connected accounts..."
              : "Open this workspace's integration inventory to see source, trust, and effective account state."}
          </p>
        </div>
      </div>
    );
  }

  const counts = integrations.counts || {};
  const showingFiltered = workspaceSearchQuery.trim().length > 0;
  const visibleIntegrationServers = showIssuesOnly
    ? filteredIntegrationServers.filter((server) => server.remediation.length > 0)
    : filteredIntegrationServers;
  const visibleAccounts = showIssuesOnly
    ? filteredAccounts.filter((account) => account.remediation.length > 0)
    : filteredAccounts;

  async function beginAccountLogin(
    profileId: string,
    options?: {
      testServerAlias?: string;
    },
  ) {
    const started = await handleStartIntegrationAccountLogin(profileId);
    if (!started) {
      return;
    }
    setPendingLogin({
      authorizationUrl: started.authorization_url,
      callbackInput: "",
      callbackMode: started.callback_mode,
      expiresAt: started.expires_at_unix,
      flowId: started.flow_id,
      profileId,
      redirectUri: started.redirect_uri,
      submitting: false,
      testServerAlias: options?.testServerAlias || "",
    });
  }

  async function completePendingLogin() {
    if (!pendingLogin || pendingLogin.submitting) {
      return;
    }
    setPendingLogin((current) => (
      current
        ? {
            ...current,
            submitting: true,
          }
        : current
    ));
    const result = await handleCompleteIntegrationAccountLogin(
      pendingLogin.profileId,
      pendingLogin.flowId,
      pendingLogin.callbackInput,
    );
    if (result?.status === "completed") {
      const serverAliasToTest = pendingLogin.testServerAlias;
      setPendingLogin(null);
      if (serverAliasToTest) {
        await handleTestIntegrationServer(serverAliasToTest);
      }
      return;
    }
    setPendingLogin((current) => (
      current
        ? {
            ...current,
            submitting: false,
          }
        : current
    ));
  }

  function openCreateServerEditor(type: "local" | "remote") {
    setServerEditor({
      mode: "create",
      originalAlias: "",
      alias: "",
      type,
      command: "",
      argsText: "",
      url: "",
      fallbackSseUrl: "",
      oauthEnabled: type === "remote",
      oauthScopesText: "",
      allowInsecureHttp: false,
      allowPrivateNetwork: false,
      cwd: "",
      timeoutSeconds: "30",
      enabled: true,
    });
  }

  function openEditServerEditor(server: typeof filteredIntegrationServers[number]) {
    setServerEditor({
      mode: "edit",
      originalAlias: server.alias,
      alias: server.alias,
      type: server.type === "remote" ? "remote" : "local",
      command: server.command,
      argsText: server.args.join("\n"),
      url: server.url,
      fallbackSseUrl: server.fallback_sse_url,
      oauthEnabled: server.oauth_enabled,
      oauthScopesText: server.oauth_scopes.join(", "),
      allowInsecureHttp: server.allow_insecure_http,
      allowPrivateNetwork: server.allow_private_network,
      cwd: server.cwd,
      timeoutSeconds: String(server.timeout_seconds || 30),
      enabled: server.enabled,
    });
  }

  async function submitServerEditor(testAfterSave: boolean) {
    if (!serverEditor) {
      return;
    }
    const timeoutSeconds = Number.parseInt(serverEditor.timeoutSeconds, 10);
    const payload = {
      type: serverEditor.type,
      command: serverEditor.command,
      args: parseLineList(serverEditor.argsText),
      url: serverEditor.url,
      fallback_sse_url: serverEditor.fallbackSseUrl,
      oauth_enabled: serverEditor.oauthEnabled,
      oauth_scopes: parseScopeList(serverEditor.oauthScopesText),
      allow_insecure_http: serverEditor.allowInsecureHttp,
      allow_private_network: serverEditor.allowPrivateNetwork,
      cwd: serverEditor.cwd,
      timeout_seconds: Number.isFinite(timeoutSeconds) ? timeoutSeconds : 30,
      enabled: serverEditor.enabled,
    };
    if (serverEditor.mode === "create") {
      const saved = await handleCreateIntegrationServer(
        {
          alias: serverEditor.alias,
          ...payload,
        },
        { testAfterSave },
      );
      if (saved) {
        setServerEditor(null);
      }
    } else {
      const saved = await handleUpdateIntegrationServer(
        serverEditor.originalAlias,
        payload,
        { testAfterSave },
      );
      if (saved) {
        setServerEditor(null);
      }
    }
  }

  function openCreateAccountEditor(prefill?: {
    provider?: string;
    mode?: string;
    mcpServer?: string;
    accountLabel?: string;
    startLoginAfterSave?: boolean;
    testServerAlias?: string;
  }) {
    setAccountEditor({
      mode: "create",
      originalProfileId: "",
      profileId: "",
      provider: prefill?.provider || "",
      modeValue: prefill?.mode || "oauth2_pkce",
      accountLabel: prefill?.accountLabel || "",
      mcpServer: prefill?.mcpServer || "",
      scopesText: "",
      startLoginAfterSave: Boolean(prefill?.startLoginAfterSave),
      testServerAlias: prefill?.testServerAlias || "",
    });
  }

  function openEditAccountEditor(account: typeof filteredAccounts[number]) {
    setAccountEditor({
      mode: "edit",
      originalProfileId: account.profile_id,
      profileId: account.profile_id,
      provider: account.provider,
      modeValue: account.mode,
      accountLabel: account.account_label,
      mcpServer: account.mcp_server,
      scopesText: account.auth_state.scopes.join(", "),
      startLoginAfterSave: false,
      testServerAlias: "",
    });
  }

  async function submitAccountEditor(connectAfterSave: boolean) {
    if (!accountEditor) {
      return;
    }
    const profileId = (
      accountEditor.mode === "create"
        ? accountEditor.profileId
        : accountEditor.originalProfileId
    ).trim();
    const provider = accountEditor.provider.trim();
    const linkedServer = accountEditor.mcpServer.trim();
    const shouldStartLogin = (
      (connectAfterSave || accountEditor.startLoginAfterSave)
      && accountEditor.modeValue.startsWith("oauth2")
      && Boolean(profileId)
    );
    const testServerAlias = accountEditor.testServerAlias || linkedServer;

    if (accountEditor.mode === "create") {
      const saved = await handleCreateIntegrationAccount({
        profile_id: profileId,
        provider,
        mode: accountEditor.modeValue,
        account_label: accountEditor.accountLabel,
        mcp_server: linkedServer,
        scopes: parseScopeList(accountEditor.scopesText),
        status: "draft",
      });
      if (saved) {
        setAccountEditor(null);
        if (shouldStartLogin) {
          await beginAccountLogin(profileId, {
            testServerAlias,
          });
        }
      }
      return;
    }
    const saved = await handleUpdateIntegrationAccount(
      profileId,
      {
        account_label: accountEditor.accountLabel,
        mcp_server: linkedServer,
        clear_mcp_server: !linkedServer,
        scopes: parseScopeList(accountEditor.scopesText),
      },
    );
    if (saved) {
      setAccountEditor(null);
      if (shouldStartLogin) {
        await beginAccountLogin(profileId, {
          testServerAlias,
        });
      }
    }
  }

  const issueItems = [
    ...filteredIntegrationServers
      .filter((server) => server.remediation.length > 0)
      .map((server) => {
        const compatibleAccounts = filteredAccounts.filter((account) => (
          server.bound_profile_ids.includes(account.profile_id)
          || account.used_by_mcp_servers.includes(server.alias)
          || account.effective_for_mcp_servers.includes(server.alias)
          || account.mcp_server === server.alias
          || account.provider === server.auth_provider
        ));
        const firstCompatibleAccount = compatibleAccounts[0];
        let actionLabel = "";
        let onAction: (() => void | Promise<void>) | null = null;

        if (server.approval_required && server.approval_state !== "approved") {
          actionLabel = "Approve now";
          onAction = async () => {
            await handleSetIntegrationApproval(server.alias, "approved");
          };
        } else if (!server.enabled || server.runtime_state === "disabled") {
          actionLabel = "Enable now";
          onAction = async () => {
            await handleSetIntegrationEnabled(server.alias, true);
          };
        } else if (
          firstCompatibleAccount
          && (server.runtime_state === "needs_auth" || server.runtime_state === "draft" || !server.effective_account)
        ) {
          actionLabel = "Use first account";
          onAction = async () => {
            await handleSelectIntegrationAccountForServer(
              server.alias,
              firstCompatibleAccount.profile_id,
            );
          };
        } else if (server.oauth_enabled && !server.effective_account) {
          actionLabel = "Create account";
          onAction = () => openCreateAccountEditor({
            provider: server.auth_provider || server.alias,
            mode: "oauth2_pkce",
            mcpServer: server.alias,
            accountLabel: `${server.alias} account`,
            startLoginAfterSave: true,
            testServerAlias: server.alias,
          });
        } else if (
          server.effective_account?.profile_id
          && (server.runtime_state === "needs_refresh" || server.auth_state.expired)
        ) {
          actionLabel = "Refresh now";
          onAction = async () => {
            await handleRefreshIntegrationAccount(server.effective_account!.profile_id);
          };
        } else if (server.runtime_state !== "ready") {
          actionLabel = serverReconnectLabel(server);
          onAction = async () => {
            await handleReconnectIntegrationServer(server.alias);
          };
        }

        return {
          id: `server:${server.alias}`,
          title: server.alias,
          kind: "server",
          detail: server.remediation[0] || "Needs attention.",
          actionLabel,
          onAction,
        };
      }),
    ...filteredAccounts
      .filter((account) => account.remediation.length > 0)
      .map((account) => {
        let actionLabel = "";
        let onAction: (() => void | Promise<void>) | null = null;

        if (account.status === "archived") {
          actionLabel = "Restore now";
          onAction = () => handleRestoreIntegrationAccount(account.profile_id);
        } else if (account.auth_state.expired) {
          actionLabel = "Refresh now";
          onAction = () => handleRefreshIntegrationAccount(account.profile_id);
        } else if (account.mode.startsWith("oauth2") && !account.auth_state.has_token) {
          actionLabel = "Connect now";
          onAction = () => beginAccountLogin(account.profile_id, {
            testServerAlias: account.mcp_server,
          });
        } else if (account.mode.startsWith("oauth2")) {
          actionLabel = "Reconnect now";
          onAction = () => beginAccountLogin(account.profile_id, {
            testServerAlias: account.mcp_server,
          });
        }

        return {
          id: `account:${account.profile_id}`,
          title: account.account_label || account.profile_id,
          kind: "account",
          detail: account.remediation[0] || "Needs attention.",
          actionLabel,
          onAction,
        };
      }),
  ];

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[#09090b]">
      <div className="border-b border-zinc-800/60 bg-[#0c0c0e] px-6 py-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-[#8a9a7b]/12 text-[#a3b396]">
              <Cable size={18} />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-zinc-100">Integrations</h1>
              <p className="text-xs text-zinc-500">
                MCP servers, linked accounts, provenance, and effective routing.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2 text-[11px]">
            <button
              type="button"
              onClick={() => openCreateServerEditor("local")}
              className="inline-flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
            >
              <Plus size={13} />
              Add local server
            </button>
            <button
              type="button"
              onClick={() => openCreateServerEditor("remote")}
              className="inline-flex items-center gap-1 rounded-full border border-[#8a9a7b]/30 bg-[#8a9a7b]/14 px-3 py-1.5 text-xs font-medium text-[#d8e5cd] transition hover:bg-[#8a9a7b]/20"
            >
              <Plus size={13} />
              Add remote server
            </button>
            <span className="rounded-full border border-zinc-800 bg-zinc-900/70 px-3 py-1 text-zinc-400">
              {counts.mcp_servers ?? integrations.mcp_servers.length} servers
            </span>
            <span className="rounded-full border border-zinc-800 bg-zinc-900/70 px-3 py-1 text-zinc-400">
              {counts.accounts ?? integrations.accounts.length} accounts
            </span>
            <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-emerald-300">
              {counts.connected_mcp_servers ?? 0} ready
            </span>
            <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-3 py-1 text-amber-300">
              {counts.attention_mcp_servers ?? 0} need attention
            </span>
            <span className="rounded-full border border-zinc-800 bg-zinc-900/70 px-3 py-1 text-zinc-400">
              {counts.pending_approval_mcp_servers ?? 0} pending approval
            </span>
            <span className="rounded-full border border-zinc-800 bg-zinc-900/70 px-3 py-1 text-zinc-400">
              {counts.draft_accounts ?? 0} draft accounts
            </span>
            <button
              type="button"
              onClick={() => setShowIssuesOnly((current) => !current)}
              className={cn(
                "rounded-full border px-3 py-1 transition",
                showIssuesOnly
                  ? "border-amber-500/20 bg-amber-500/10 text-amber-300"
                  : "border-zinc-800 bg-zinc-900/70 text-zinc-400 hover:border-zinc-700 hover:text-zinc-300",
              )}
            >
              {showIssuesOnly ? "Showing issues only" : "Issues only"}
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6">
        <div className="grid gap-6 xl:grid-cols-[1.25fr_0.95fr]">
          <section className="space-y-4">
            {serverEditor && (
              <article className="rounded-2xl border border-[#8a9a7b]/25 bg-[#8a9a7b]/7 p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-semibold text-zinc-100">
                      {serverEditor.mode === "create" ? "Add MCP Server" : "Edit MCP Server"}
                    </h2>
                    <p className="text-xs text-zinc-500">
                      Start with transport details first. Auth and advanced routing come after.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setServerEditor(null)}
                    className="rounded-full border border-zinc-700 px-3 py-1 text-[11px] text-zinc-300 transition hover:border-zinc-600 hover:bg-zinc-900"
                  >
                    Close
                  </button>
                </div>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <label className="block text-xs text-zinc-400">
                    Alias
                    <input
                      value={serverEditor.alias}
                      disabled={serverEditor.mode === "edit"}
                      onChange={(event) => setServerEditor((current) => (
                        current ? { ...current, alias: event.target.value } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60 disabled:cursor-not-allowed disabled:opacity-60"
                      placeholder="notion"
                    />
                  </label>
                  <label className="block text-xs text-zinc-400">
                    Type
                    <select
                      value={serverEditor.type}
                      onChange={(event) => setServerEditor((current) => (
                        current ? {
                          ...current,
                          type: event.target.value === "remote" ? "remote" : "local",
                          oauthEnabled: event.target.value === "remote" ? current.oauthEnabled : false,
                        } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                    >
                      <option value="local">Local</option>
                      <option value="remote">Remote</option>
                    </select>
                  </label>
                  {serverEditor.type === "local" ? (
                    <>
                      <label className="block text-xs text-zinc-400 md:col-span-2">
                        Command
                        <input
                          value={serverEditor.command}
                          onChange={(event) => setServerEditor((current) => (
                            current ? { ...current, command: event.target.value } : current
                          ))}
                          className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                          placeholder="npx"
                        />
                      </label>
                      <label className="block text-xs text-zinc-400 md:col-span-2">
                        Arguments (one per line)
                        <textarea
                          value={serverEditor.argsText}
                          onChange={(event) => setServerEditor((current) => (
                            current ? { ...current, argsText: event.target.value } : current
                          ))}
                          rows={4}
                          className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                          placeholder={"-y\n@modelcontextprotocol/server-notion"}
                        />
                      </label>
                      <label className="block text-xs text-zinc-400 md:col-span-2">
                        Working directory (optional)
                        <input
                          value={serverEditor.cwd}
                          onChange={(event) => setServerEditor((current) => (
                            current ? { ...current, cwd: event.target.value } : current
                          ))}
                          className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                          placeholder="/path/to/workspace"
                        />
                      </label>
                    </>
                  ) : (
                    <>
                      <label className="block text-xs text-zinc-400 md:col-span-2">
                        URL
                        <input
                          value={serverEditor.url}
                          onChange={(event) => setServerEditor((current) => (
                            current ? { ...current, url: event.target.value } : current
                          ))}
                          className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                          placeholder="https://mcp.example.com"
                        />
                      </label>
                      <label className="block text-xs text-zinc-400">
                        OAuth scopes
                        <input
                          value={serverEditor.oauthScopesText}
                          onChange={(event) => setServerEditor((current) => (
                            current ? { ...current, oauthScopesText: event.target.value } : current
                          ))}
                          className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                          placeholder="read, search"
                        />
                      </label>
                      <label className="block text-xs text-zinc-400">
                        Fallback SSE URL (optional)
                        <input
                          value={serverEditor.fallbackSseUrl}
                          onChange={(event) => setServerEditor((current) => (
                            current ? { ...current, fallbackSseUrl: event.target.value } : current
                          ))}
                          className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                          placeholder="https://mcp.example.com/sse"
                        />
                      </label>
                    </>
                  )}
                  <label className="block text-xs text-zinc-400">
                    Timeout (seconds)
                    <input
                      value={serverEditor.timeoutSeconds}
                      onChange={(event) => setServerEditor((current) => (
                        current ? { ...current, timeoutSeconds: event.target.value } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                      inputMode="numeric"
                    />
                  </label>
                  <div className="flex flex-wrap items-center gap-4 pt-6 text-xs text-zinc-300">
                    <label className="inline-flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={serverEditor.enabled}
                        onChange={(event) => setServerEditor((current) => (
                          current ? { ...current, enabled: event.target.checked } : current
                        ))}
                      />
                      Enabled
                    </label>
                    {serverEditor.type === "remote" && (
                      <>
                        <label className="inline-flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={serverEditor.oauthEnabled}
                            onChange={(event) => setServerEditor((current) => (
                              current ? { ...current, oauthEnabled: event.target.checked } : current
                            ))}
                          />
                          OAuth enabled
                        </label>
                        <label className="inline-flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={serverEditor.allowInsecureHttp}
                            onChange={(event) => setServerEditor((current) => (
                              current ? { ...current, allowInsecureHttp: event.target.checked } : current
                            ))}
                          />
                          Allow HTTP
                        </label>
                        <label className="inline-flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={serverEditor.allowPrivateNetwork}
                            onChange={(event) => setServerEditor((current) => (
                              current ? { ...current, allowPrivateNetwork: event.target.checked } : current
                            ))}
                          />
                          Allow private network
                        </label>
                      </>
                    )}
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void submitServerEditor(false)}
                    className="rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    onClick={() => void submitServerEditor(true)}
                    className="rounded-full border border-[#8a9a7b]/30 bg-[#8a9a7b]/15 px-3 py-1.5 text-xs font-medium text-[#d8e5cd] transition hover:bg-[#8a9a7b]/20"
                  >
                    Save and test
                  </button>
                </div>
              </article>
            )}

            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-zinc-100">MCP Servers</h2>
                <p className="text-xs text-zinc-500">
                  Source, trust, connection posture, and the account Loom would use.
                </p>
              </div>
              {(showingFiltered || showIssuesOnly) && (
                <span className="text-[11px] text-zinc-500">
                  {visibleIntegrationServers.length} matching
                </span>
              )}
            </div>

            {visibleIntegrationServers.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-zinc-800 bg-zinc-950/40 px-5 py-6 text-sm text-zinc-500">
                {showIssuesOnly
                  ? "No MCP servers currently need repair in this workspace."
                  : showingFiltered
                  ? "No MCP servers match the current workspace search."
                  : "No MCP servers are configured for this workspace yet."}
              </div>
            ) : (
              visibleIntegrationServers.map((server) => {
                const compatibleAccounts = visibleAccounts.filter((account) => (
                  server.bound_profile_ids.includes(account.profile_id)
                  || account.used_by_mcp_servers.includes(server.alias)
                  || account.effective_for_mcp_servers.includes(server.alias)
                  || account.mcp_server === server.alias
                  || account.provider === server.auth_provider
                ));
                const selectedProfileId = (
                  serverSelections[server.alias]
                  || server.effective_account?.profile_id
                  || compatibleAccounts[0]?.profile_id
                  || ""
                );

                return (
                <article
                  key={server.alias}
                  className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-5 shadow-[0_12px_30px_rgba(0,0,0,0.18)]"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-sm font-semibold text-zinc-100">{server.alias}</h3>
                        <span className="rounded-full border border-zinc-800 bg-zinc-900/80 px-2 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
                          {server.type}
                        </span>
                        <span className={cn(
                          "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                          runtimeTone(server.runtime_state),
                        )}>
                          {humanizeState(server.runtime_state)}
                        </span>
                        {server.approval_required && (
                          <span className={cn(
                            "rounded-full border px-2 py-0.5 text-[10px] font-medium",
                            approvalTone(server.approval_state),
                          )}>
                            {humanizeState(server.approval_state)}
                          </span>
                        )}
                      </div>
                      <p className="mt-1 text-xs text-zinc-500">
                        {server.url || server.command || "No transport details available"}
                      </p>
                    </div>
                    <div className={cn("flex items-center gap-1 text-xs", trustTone(server.trust_state))}>
                      {server.trust_state === "trusted" ? <ShieldCheck size={14} /> : <ShieldAlert size={14} />}
                      <span>{server.source_label}</span>
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3 sm:grid-cols-2">
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/45 p-3">
                      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-zinc-500">
                        <Server size={12} />
                        Source
                      </div>
                      <p className="mt-2 text-sm text-zinc-200">{server.trust_summary}</p>
                      {server.approval_required && (
                        <p className="mt-2 text-xs text-zinc-400">
                          Approval: <span className="text-zinc-200">{humanizeState(server.approval_state)}</span>
                        </p>
                      )}
                      <p className="mt-1 break-all font-mono text-[11px] text-zinc-500">
                        {server.source_path || "Path unavailable"}
                      </p>
                    </div>

                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/45 p-3">
                      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-zinc-500">
                        <LockKeyhole size={12} />
                        Auth
                      </div>
                      <p className="mt-2 text-sm text-zinc-200">{server.auth_state.label}</p>
                      <p className="mt-1 text-xs text-zinc-500">
                        {server.auth_state.reason || "No auth issues reported."}
                      </p>
                      {server.auth_state.storage && (
                        <p className="mt-2 text-[11px] text-zinc-500">
                          Storage: <span className="text-zinc-300">{server.auth_state.storage}</span>
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="mt-4 rounded-xl border border-zinc-800 bg-[#101114] p-4">
                    <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-zinc-500">
                      <UserCircle2 size={12} />
                      Effective Account
                    </div>
                    {server.effective_account ? (
                      <div className="mt-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <p className="text-sm font-medium text-zinc-100">
                            {server.effective_account.account_label || server.effective_account.profile_id}
                          </p>
                          <span className="rounded-full border border-zinc-800 bg-zinc-900/80 px-2 py-0.5 text-[10px] text-zinc-400">
                            {server.effective_account.mode}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-zinc-500">
                          Provider {server.effective_account.provider} via {server.effective_account.source}
                        </p>
                      </div>
                    ) : (
                      <p className="mt-2 text-sm text-zinc-400">No Loom account is selected for this server yet.</p>
                    )}

                    {server.remediation.length > 0 && (
                      <div className="mt-4 rounded-xl border border-amber-500/15 bg-amber-500/6 p-3">
                        <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-amber-200/80">
                          <TriangleAlert size={12} />
                          Next Action
                        </div>
                        <ul className="mt-2 space-y-1 text-sm text-amber-100/90">
                          {server.remediation.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                      {server.oauth_enabled && compatibleAccounts.length > 0 && (
                        <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-900/45 p-3">
                        <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-zinc-500">
                          <UserCircle2 size={12} />
                          Use Existing Account
                        </div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <select
                            value={selectedProfileId}
                            onChange={(event) => setServerSelections((current) => ({
                              ...current,
                              [server.alias]: event.target.value,
                            }))}
                            className="min-w-[220px] rounded-full border border-zinc-700 bg-zinc-950/80 px-3 py-1.5 text-xs text-zinc-200 outline-none transition focus:border-[#8a9a7b]/60"
                          >
                            {compatibleAccounts.map((account) => (
                              <option key={account.profile_id} value={account.profile_id}>
                                {account.account_label || account.profile_id}
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            disabled={!selectedProfileId}
                            onClick={() => void handleSelectIntegrationAccountForServer(
                              server.alias,
                              selectedProfileId,
                            )}
                            className="rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60"
                          >
                            Use account
                          </button>
                        </div>
                      </div>
                    )}
                    <div className="mt-4 flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => openEditServerEditor(server)}
                        className="inline-flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                      >
                        <PencilLine size={12} />
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleSetIntegrationEnabled(server.alias, !server.enabled)}
                        className="inline-flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                      >
                        <Power size={12} />
                        {server.enabled ? "Disable" : "Enable"}
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleTestIntegrationServer(server.alias)}
                        className="rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                      >
                        {serverTestLabel(server)}
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleReconnectIntegrationServer(server.alias)}
                        className="rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                      >
                        {serverReconnectLabel(server)}
                      </button>
                      {server.oauth_enabled && (
                        <button
                          type="button"
                          onClick={() => openCreateAccountEditor({
                            provider: server.auth_provider || server.alias,
                            mode: "oauth2_pkce",
                            mcpServer: server.alias,
                            accountLabel: `${server.alias} account`,
                            startLoginAfterSave: true,
                            testServerAlias: server.alias,
                          })}
                          className={cn(
                            "rounded-full px-3 py-1.5 text-xs font-medium transition",
                            server.effective_account
                              ? "border border-zinc-700 bg-zinc-900/80 text-zinc-200 hover:border-zinc-600 hover:bg-zinc-800"
                              : "border border-[#8a9a7b]/30 bg-[#8a9a7b]/15 text-[#d8e5cd] hover:bg-[#8a9a7b]/20",
                          )}
                        >
                          {compatibleAccounts.length > 0 || server.effective_account
                            ? "Create another account"
                            : "Create and connect account"}
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void handleDeleteIntegrationServer(server.alias)}
                        className="inline-flex items-center gap-1 rounded-full border border-rose-500/20 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-200 transition hover:bg-rose-500/15"
                      >
                        <Trash2 size={12} />
                        Delete
                      </button>
                    </div>
                    {server.approval_required && (
                      <div className="mt-4 flex flex-wrap gap-2">
                        <button
                          type="button"
                          onClick={() => void handleSetIntegrationApproval(server.alias, "approved")}
                          className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-200 transition hover:bg-emerald-500/15"
                        >
                          Approve server
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleSetIntegrationApproval(server.alias, "rejected")}
                          className="rounded-full border border-rose-500/20 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-200 transition hover:bg-rose-500/15"
                        >
                          Reject server
                        </button>
                      </div>
                    )}
                  </div>
                </article>
                );
              })
            )}
          </section>

          <section className="space-y-4">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-sm font-semibold text-zinc-100">Issues</h2>
                  <p className="text-xs text-zinc-500">
                    The safest next actions across trust, auth, and routing.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => void handleSyncIntegrationDrafts()}
                  className="rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                >
                  Create missing drafts
                </button>
                {showIssuesOnly && (
                  <button
                    type="button"
                    onClick={() => setShowIssuesOnly(false)}
                    className="rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                  >
                    Show all integrations
                  </button>
                )}
              </div>
              {issueItems.length === 0 ? (
                <p className="mt-3 text-sm text-zinc-500">
                  No integration issues are currently surfaced for this workspace.
                </p>
              ) : (
                <div className="mt-3 space-y-2">
                  {issueItems.slice(0, 6).map((item) => (
                    <div
                      key={item.id}
                      className="rounded-xl border border-zinc-800 bg-zinc-900/45 px-3 py-2"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <p className="text-sm font-medium text-zinc-100">{item.title}</p>
                        <span className="text-[10px] uppercase tracking-wide text-zinc-500">
                          {item.kind}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-zinc-400">{item.detail}</p>
                      {item.actionLabel && item.onAction && (
                        <button
                          type="button"
                          onClick={() => void item.onAction?.()}
                          className="mt-3 rounded-full border border-[#8a9a7b]/30 bg-[#8a9a7b]/15 px-3 py-1.5 text-xs font-medium text-[#d8e5cd] transition hover:bg-[#8a9a7b]/20"
                        >
                          {item.actionLabel}
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-zinc-100">Accounts</h2>
                <p className="text-xs text-zinc-500">
                  Credential homes, account state, and where each account is routed.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => openCreateAccountEditor()}
                  className="inline-flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                >
                  <Plus size={13} />
                  Add account
                </button>
                {(showingFiltered || showIssuesOnly) && (
                  <span className="text-[11px] text-zinc-500">
                    {visibleAccounts.length} matching
                  </span>
                )}
              </div>
            </div>

            {accountEditor && (
              <article className="rounded-2xl border border-[#8a9a7b]/25 bg-[#8a9a7b]/7 p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-100">
                      {accountEditor.mode === "create" ? "Add Account" : "Edit Account"}
                    </h3>
                    <p className="text-xs text-zinc-500">
                      Keep account identity simple first. Advanced token plumbing can stay hidden.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setAccountEditor(null)}
                    className="rounded-full border border-zinc-700 px-3 py-1 text-[11px] text-zinc-300 transition hover:border-zinc-600 hover:bg-zinc-900"
                  >
                    Close
                  </button>
                </div>
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  <label className="block text-xs text-zinc-400">
                    Profile id
                    <input
                      value={accountEditor.profileId}
                      disabled={accountEditor.mode === "edit"}
                      onChange={(event) => setAccountEditor((current) => (
                        current ? { ...current, profileId: event.target.value } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60 disabled:cursor-not-allowed disabled:opacity-60"
                      placeholder="notion_personal"
                    />
                  </label>
                  <label className="block text-xs text-zinc-400">
                    Account label
                    <input
                      value={accountEditor.accountLabel}
                      onChange={(event) => setAccountEditor((current) => (
                        current ? { ...current, accountLabel: event.target.value } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                      placeholder="Notion Personal"
                    />
                  </label>
                  <label className="block text-xs text-zinc-400">
                    Provider
                    <input
                      value={accountEditor.provider}
                      disabled={accountEditor.mode === "edit"}
                      onChange={(event) => setAccountEditor((current) => (
                        current ? { ...current, provider: event.target.value } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60 disabled:cursor-not-allowed disabled:opacity-60"
                      placeholder="notion"
                    />
                  </label>
                  <label className="block text-xs text-zinc-400">
                    Mode
                    <select
                      value={accountEditor.modeValue}
                      disabled={accountEditor.mode === "edit"}
                      onChange={(event) => setAccountEditor((current) => (
                        current ? { ...current, modeValue: event.target.value } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      <option value="oauth2_pkce">oauth2_pkce</option>
                      <option value="oauth2_device">oauth2_device</option>
                      <option value="api_key">api_key</option>
                      <option value="env_passthrough">env_passthrough</option>
                      <option value="cli_passthrough">cli_passthrough</option>
                    </select>
                  </label>
                  <label className="block text-xs text-zinc-400">
                    Linked server (optional)
                    <input
                      value={accountEditor.mcpServer}
                      onChange={(event) => setAccountEditor((current) => (
                        current ? { ...current, mcpServer: event.target.value } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                      placeholder="notion"
                    />
                  </label>
                  <label className="block text-xs text-zinc-400">
                    Scopes (optional)
                    <input
                      value={accountEditor.scopesText}
                      onChange={(event) => setAccountEditor((current) => (
                        current ? { ...current, scopesText: event.target.value } : current
                      ))}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                      placeholder="read, search"
                    />
                  </label>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void submitAccountEditor(false)}
                    className="rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                  >
                    Save account
                  </button>
                  {accountEditor.modeValue.startsWith("oauth2") && (
                    <button
                      type="button"
                      onClick={() => void submitAccountEditor(true)}
                      className="rounded-full border border-[#8a9a7b]/30 bg-[#8a9a7b]/15 px-3 py-1.5 text-xs font-medium text-[#d8e5cd] transition hover:bg-[#8a9a7b]/20"
                    >
                      Save and connect
                    </button>
                  )}
                </div>
              </article>
            )}

            {visibleAccounts.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-zinc-800 bg-zinc-950/40 px-5 py-6 text-sm text-zinc-500">
                {showIssuesOnly
                  ? "No accounts currently need repair in this workspace."
                  : showingFiltered
                  ? "No accounts match the current workspace search."
                  : "No Loom accounts are tracked for this workspace yet."}
              </div>
            ) : (
              visibleAccounts.map((account) => (
                <article
                  key={account.profile_id}
                  className="rounded-2xl border border-zinc-800 bg-zinc-950/40 p-5"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-sm font-semibold text-zinc-100">
                          {account.account_label || account.profile_id}
                        </h3>
                        <span className="rounded-full border border-zinc-800 bg-zinc-900/80 px-2 py-0.5 text-[10px] text-zinc-400">
                          {account.mode}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-zinc-500">
                        {account.provider} · {account.profile_id}
                      </p>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-zinc-400">
                      <KeyRound size={13} />
                      <span>{account.auth_state.label}</span>
                    </div>
                  </div>

                  <div className="mt-4 grid gap-3">
                    <div className="rounded-xl border border-zinc-800 bg-zinc-900/45 p-3">
                      <div className="flex items-center gap-2 text-[11px] uppercase tracking-wide text-zinc-500">
                        <CheckCircle2 size={12} />
                        Routing
                      </div>
                      <p className="mt-2 text-sm text-zinc-200">
                        {account.effective_for_mcp_servers.length > 0
                          ? `Effective for ${account.effective_for_mcp_servers.join(", ")}`
                          : account.used_by_mcp_servers.length > 0
                            ? `Bound to ${account.used_by_mcp_servers.join(", ")}`
                            : "Not routed to any MCP server yet"}
                      </p>
                      {account.default_selectors.length > 0 && (
                        <p className="mt-2 text-[11px] text-zinc-500">
                          Defaults: <span className="text-zinc-300">{account.default_selectors.join(", ")}</span>
                        </p>
                      )}
                      {account.writable_storage_kind && account.writable_storage_kind !== "none" && (
                        <p className="mt-2 text-[11px] text-zinc-500">
                          Writable storage: <span className="text-zinc-300">{account.writable_storage_kind}</span>
                        </p>
                      )}
                    </div>

                    {account.auth_state.expires_at && (
                      <div className="rounded-xl border border-zinc-800 bg-zinc-900/45 p-3 text-xs text-zinc-400">
                        OAuth expiry: <span className="text-zinc-200">{formatExpiry(account.auth_state.expires_at)}</span>
                      </div>
                    )}

                    {account.remediation.length > 0 && (
                      <div className="rounded-xl border border-amber-500/15 bg-amber-500/6 p-3 text-sm text-amber-100/90">
                        {account.remediation.join(" ")}
                      </div>
                    )}

                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => openEditAccountEditor(account)}
                        className="inline-flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                      >
                        <PencilLine size={12} />
                        Edit
                      </button>
                      {account.mode.startsWith("oauth2") && (
                        <button
                          type="button"
                          onClick={() => void beginAccountLogin(account.profile_id)}
                          className="rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                        >
                          {account.auth_state.has_token ? "Reconnect account" : "Connect account"}
                        </button>
                      )}
                      {account.auth_state.expired && (
                        <button
                          type="button"
                          onClick={() => void handleRefreshIntegrationAccount(account.profile_id)}
                          className="rounded-full border border-amber-500/20 bg-amber-500/10 px-3 py-1.5 text-xs font-medium text-amber-200 transition hover:bg-amber-500/15"
                        >
                          Refresh token
                        </button>
                      )}
                      {account.auth_state.has_token && (
                        <button
                          type="button"
                          onClick={() => void handleLogoutIntegrationAccount(account.profile_id)}
                          className="rounded-full border border-rose-500/20 bg-rose-500/10 px-3 py-1.5 text-xs font-medium text-rose-200 transition hover:bg-rose-500/15"
                        >
                          Disconnect
                        </button>
                      )}
                      {account.status === "archived" ? (
                        <button
                          type="button"
                          onClick={() => void handleRestoreIntegrationAccount(account.profile_id)}
                          className="inline-flex items-center gap-1 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-200 transition hover:bg-emerald-500/15"
                        >
                          <Archive size={12} />
                          Restore
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => void handleArchiveIntegrationAccount(account.profile_id)}
                          className="inline-flex items-center gap-1 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1.5 text-xs font-medium text-zinc-200 transition hover:border-zinc-600 hover:bg-zinc-800"
                        >
                          <Archive size={12} />
                          Archive
                        </button>
                      )}
                    </div>

                    <p className="break-all font-mono text-[11px] text-zinc-600">
                      {account.source_path || "Source path unavailable"}
                    </p>
                  </div>
                </article>
              ))
            )}

            {pendingLogin && (
              <article className="rounded-2xl border border-[#8a9a7b]/25 bg-[#8a9a7b]/7 p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-100">
                      Finish Account Connection
                    </h3>
                    <p className="text-xs text-zinc-500">
                      Complete OAuth for {pendingLogin.profileId} and confirm the callback.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setPendingLogin(null)}
                    className="rounded-full border border-zinc-700 px-3 py-1 text-[11px] text-zinc-300 transition hover:border-zinc-600 hover:bg-zinc-900"
                  >
                    Close
                  </button>
                </div>
                <div className="mt-4 space-y-3">
                  <a
                    href={pendingLogin.authorizationUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="block break-all rounded-xl border border-zinc-800 bg-zinc-950/50 px-3 py-3 text-xs text-[#c7d5bb] underline-offset-4 hover:underline"
                  >
                    {pendingLogin.authorizationUrl}
                  </a>
                  <p className="text-xs text-zinc-500">
                    Callback mode: {pendingLogin.callbackMode} · Expires{" "}
                    {formatExpiry(pendingLogin.expiresAt)}
                  </p>
                  <p className="break-all text-[11px] text-zinc-600">
                    Redirect URI: {pendingLogin.redirectUri}
                  </p>
                  <label className="block text-xs text-zinc-400">
                    Paste callback URL or authorization code
                    <textarea
                      value={pendingLogin.callbackInput}
                      onChange={(event) => setPendingLogin((current) => (
                        current
                          ? {
                              ...current,
                              callbackInput: event.target.value,
                            }
                          : current
                      ))}
                      rows={3}
                      className="mt-2 w-full rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-[#8a9a7b]/60"
                      placeholder="Callback URL or code"
                    />
                  </label>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => void completePendingLogin()}
                      disabled={pendingLogin.submitting}
                      className="rounded-full border border-[#8a9a7b]/30 bg-[#8a9a7b]/15 px-3 py-1.5 text-xs font-medium text-[#d8e5cd] transition hover:bg-[#8a9a7b]/20 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {pendingLogin.submitting ? "Finishing..." : "Finish connection"}
                    </button>
                  </div>
                </div>
              </article>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
