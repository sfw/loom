import { useEffect, useRef } from "react";
import {
  shallowEqual,
  useAppActions,
  useAppSelector,
} from "@/context/AppContext";
import { highlightText } from "@/utils";
import { getRuntimeBaseUrl } from "@/api";
import {
  Search,
  Circle,
  MessageSquare,
  Play,
  AlertTriangle,
  Wifi,
  WifiOff,
  RefreshCw,
  Loader2,
  Terminal,
} from "lucide-react";
import CommandPalette from "../CommandPalette";
import Sidebar from "./Sidebar";
import OverviewTab from "./OverviewTab";
import ThreadsTab from "./ThreadsTab";
import RunsTab from "./RunsTab";
import FilesTab from "./FilesTab";
import SettingsPanel from "./SettingsPanel";
import WorkspaceModal from "./WorkspaceModal";
import { cn } from "@/lib/utils";

function CommandPaletteOverlay() {
  const {
    activeCommandIndex,
    commandDraft,
    commandInputRef,
    commandPaletteOpen,
    paletteEntries,
    paletteSections,
    searchingCommandPalette,
  } = useAppSelector((state) => ({
    activeCommandIndex: state.activeCommandIndex,
    commandDraft: state.commandDraft,
    commandInputRef: state.commandInputRef,
    commandPaletteOpen: state.commandPaletteOpen,
    paletteEntries: state.paletteEntries,
    paletteSections: state.paletteSections,
    searchingCommandPalette: state.searchingCommandPalette,
  }), shallowEqual);
  const {
    executePaletteEntry,
    handleCommandInputKeyDown,
    handleCommandSubmit,
    setCommandDraft,
    setCommandPaletteOpen,
  } = useAppActions();

  if (!commandPaletteOpen) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[18vh] bg-black/55 backdrop-blur-sm"
      onClick={() => setCommandPaletteOpen(false)}
    >
      <div
        className="w-[540px] max-h-[400px] bg-[#111114] border border-zinc-800 rounded-xl shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <form onSubmit={handleCommandSubmit} className="border-b border-zinc-800">
          <input
            ref={commandInputRef}
            autoFocus
            type="text"
            value={commandDraft}
            onChange={(e) => setCommandDraft(e.target.value)}
            onKeyDown={handleCommandInputKeyDown}
            placeholder="Search workspaces, threads, runs, files..."
            className="w-full bg-transparent px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 outline-none"
          />
        </form>
        <CommandPalette
          open={commandPaletteOpen}
          commandDraft={commandDraft}
          searching={searchingCommandPalette}
          activeIndex={activeCommandIndex}
          paletteEntries={paletteEntries}
          paletteSections={paletteSections}
          onSelect={executePaletteEntry}
          renderHighlight={highlightText}
        />
      </div>
    </div>
  );
}

export default function AppShell() {
  const {
    activeTab,
    approvalInbox,
    connectionState,
    error,
    notice,
    overview,
    runtime,
    selectedWorkspaceId,
    selectedWorkspaceSummary,
    showNewWorkspace,
    workspaces,
  } = useAppSelector((state) => ({
    activeTab: state.activeTab,
    approvalInbox: state.approvalInbox,
    connectionState: state.connectionState,
    error: state.error,
    notice: state.notice,
    overview: state.overview,
    runtime: state.runtime,
    selectedWorkspaceId: state.selectedWorkspaceId,
    selectedWorkspaceSummary: state.selectedWorkspaceSummary,
    showNewWorkspace: state.showNewWorkspace,
    workspaces: state.workspaces,
  }), shallowEqual);
  const {
    focusCommandBar,
    retryConnection,
    setActiveTab,
    setCommandPaletteOpen,
    setError,
    setNotice,
  } = useAppActions();

  const overviewWorkspace = (
    overview?.workspace
    && selectedWorkspaceId
    && overview.workspace.id === selectedWorkspaceId
  )
    ? overview.workspace
    : null;
  const ws = overviewWorkspace ?? selectedWorkspaceSummary;
  const activeRuns = ws?.active_run_count ?? 0;
  const totalActiveRuns = workspaces.reduce((sum, workspace) => (
    sum + (
      overviewWorkspace && workspace.id === overviewWorkspace.id
        ? (overviewWorkspace.active_run_count ?? 0)
        : (workspace.active_run_count ?? 0)
    )
  ), 0);
  const pendingApprovals = approvalInbox.length;
  const conversationCount = overview?.recent_conversations?.length ?? 0;
  const runCount = overview?.recent_runs?.length ?? 0;
  const closeConfirmedRef = useRef(false);
  const hasShellSnapshot = Boolean(runtime) || workspaces.length > 0;
  const showBlockingConnectionScreen =
    connectionState !== "connected" && !hasShellSnapshot;

  // Auto-dismiss toasts after 4 seconds
  useEffect(() => {
    if (!notice) return;
    const t = window.setTimeout(() => setNotice(""), 4000);
    return () => window.clearTimeout(t);
  }, [notice, setNotice]);

  useEffect(() => {
    if (!error) return;
    const t = window.setTimeout(() => setError(""), 8000);
    return () => window.clearTimeout(t);
  }, [error, setError]);

  useEffect(() => {
    if ((activeTab as string) === "inbox") {
      setActiveTab("overview");
    }
  }, [activeTab, setActiveTab]);

  useEffect(() => {
    let disposed = false;
    let removeCloseListener: (() => void) | undefined;

    const bindCloseConfirm = async () => {
      try {
        const [{ getCurrentWindow }, dialog] = await Promise.all([
          import("@tauri-apps/api/window"),
          import("@tauri-apps/plugin-dialog").catch(() => null),
        ]);
        if (disposed) return;
        const appWindow = getCurrentWindow();
        removeCloseListener = await appWindow.onCloseRequested(async (event) => {
          if (closeConfirmedRef.current || totalActiveRuns <= 0) return;

          event.preventDefault();
          const runLabel = totalActiveRuns === 1 ? "1 active run" : `${totalActiveRuns} active runs`;
          const message = `Closing Loom Desktop will pause ${runLabel} so you can resume later. Close anyway?`;

          let confirmed = false;
          try {
            if (dialog?.confirm) {
              confirmed = await dialog.confirm(message, { title: "Close Loom Desktop?", kind: "warning" });
            } else {
              confirmed = window.confirm(message);
            }
          } catch {
            confirmed = window.confirm(message);
          }

          if (!confirmed) return;
          closeConfirmedRef.current = true;
          await appWindow.close();
        });
      } catch {
        // Non-Tauri/test environments do not expose window lifecycle APIs.
      }
    };

    void bindCloseConfirm();

    return () => {
      disposed = true;
      removeCloseListener?.();
    };
  }, [totalActiveRuns]);

  // Show connection screen when not connected
  if (showBlockingConnectionScreen) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#09090b] text-zinc-100">
        <div className="max-w-sm text-center">
          {connectionState === "connecting" ? (
            <>
              <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#6b7a5e]/15">
                <Loader2 className="h-7 w-7 text-[#a3b396] animate-spin" />
              </div>
              <h2 className="text-lg font-bold text-zinc-100 mb-2">Connecting to Loomd</h2>
              <p className="text-sm text-zinc-500 mb-1">
                Trying <span className="font-mono text-zinc-400">{getRuntimeBaseUrl()}</span>
              </p>
              <p className="text-xs text-zinc-600">
                This may take a moment if the runtime is starting up...
              </p>
            </>
          ) : (
            <>
              <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-red-500/15">
                <WifiOff className="h-7 w-7 text-red-400" />
              </div>
              <h2 className="text-lg font-bold text-zinc-100 mb-2">Cannot reach Loomd</h2>
              <p className="text-sm text-zinc-500 mb-4 leading-relaxed">
                Could not connect to <span className="font-mono text-zinc-400">{getRuntimeBaseUrl()}</span>.
                Make sure the runtime is running.
              </p>

              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-3 mb-5 text-left">
                <p className="text-xs font-medium text-zinc-400 mb-2">Start the runtime with:</p>
                <div className="flex items-center gap-2 rounded-md bg-zinc-950 px-3 py-2">
                  <Terminal size={13} className="shrink-0 text-zinc-600" />
                  <code className="text-xs text-zinc-300 font-mono">uv run loomd</code>
                </div>
                <p className="text-[10.5px] text-zinc-600 mt-2">
                  Or set <span className="font-mono">VITE_LOOMD_URL</span> if the runtime is on a different host/port.
                </p>
              </div>

              <button
                type="button"
                onClick={retryConnection}
                className="inline-flex items-center gap-2 rounded-lg bg-[#6b7a5e] px-5 py-2.5 text-sm font-semibold text-white hover:bg-[#8a9a7b] transition-colors"
              >
                <RefreshCw size={14} />
                Retry connection
              </button>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-[256px_1fr] h-screen overflow-hidden bg-[#09090b] text-zinc-100">
      <Sidebar />

      <main className="flex flex-col h-full overflow-hidden">
        {/* Header */}
        <header className="flex items-center justify-between gap-4 border-b border-zinc-800/60 bg-[#0c0c0e] px-5 py-2.5">
          {/* Left: workspace context */}
          <div className="flex items-center gap-3 min-w-0">
            {(selectedWorkspaceId || ws) ? (
              <>
                <div className="flex flex-col min-w-0">
                  <h1 className="truncate text-sm font-semibold text-zinc-100">
                    {ws?.display_name || overview?.workspace?.display_name || "Loading workspace..."}
                  </h1>
                  <p className="truncate text-[10.5px] text-zinc-600 font-mono">
                    {ws?.canonical_path || overview?.workspace?.canonical_path || ""}
                  </p>
                </div>

                {/* Workspace stats pills */}
                <div className="hidden md:flex items-center gap-1.5">
                  {activeRuns > 0 && (
                    <span className="inline-flex items-center gap-1 rounded-md bg-sky-500/10 px-2 py-0.5 text-[10px] font-semibold text-sky-400">
                      <Circle size={5} className="fill-current animate-pulse" />
                      {activeRuns} running
                    </span>
                  )}
                  {pendingApprovals > 0 && (
                    <span className="inline-flex items-center gap-1 rounded-md bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-400">
                      <AlertTriangle size={9} />
                      {pendingApprovals}
                    </span>
                  )}
                  {conversationCount > 0 && (
                    <span className="inline-flex items-center gap-1 rounded-md bg-zinc-800/80 px-2 py-0.5 text-[10px] font-medium text-zinc-500">
                      <MessageSquare size={9} />
                      {conversationCount}
                    </span>
                  )}
                  {runCount > 0 && (
                    <span className="inline-flex items-center gap-1 rounded-md bg-zinc-800/80 px-2 py-0.5 text-[10px] font-medium text-zinc-500">
                      <Play size={9} />
                      {runCount}
                    </span>
                  )}
                </div>
              </>
            ) : (
              <span className="text-sm text-zinc-500">Select a workspace to get started</span>
            )}
          </div>

          {/* Right: search trigger */}
          <button
            type="button"
            onClick={() => {
              setCommandPaletteOpen(true);
              focusCommandBar();
            }}
            className="flex items-center gap-2 rounded-lg border border-zinc-800/60 bg-zinc-900/50 px-3 py-1.5 text-xs text-zinc-500 transition-colors hover:border-zinc-700 hover:text-zinc-300 shrink-0"
          >
            <Search size={13} />
            <span className="hidden sm:inline">Search...</span>
            <kbd className="hidden sm:inline-flex h-[18px] items-center rounded border border-zinc-700/60 bg-zinc-800/60 px-1.5 text-[10px] font-medium text-zinc-600">
              ⌘K
            </kbd>
          </button>
        </header>

        {connectionState !== "connected" && (
          <div
            className={cn(
              "flex items-center justify-between gap-3 border-b px-5 py-2 text-xs",
              connectionState === "connecting"
                ? "border-sky-500/20 bg-sky-500/8 text-sky-200"
                : "border-amber-500/20 bg-amber-500/8 text-amber-200",
            )}
          >
            <div className="flex min-w-0 items-center gap-2">
              {connectionState === "connecting" ? (
                <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
              ) : (
                <WifiOff className="h-3.5 w-3.5 shrink-0" />
              )}
              <span className="truncate">
                {connectionState === "connecting"
                  ? "Reconnecting to Loomd..."
                  : "Connection to Loomd dropped. Retrying automatically..."}
              </span>
            </div>
            <button
              type="button"
              onClick={retryConnection}
              className="shrink-0 rounded-md border border-current/20 px-2 py-1 font-medium text-inherit transition-colors hover:bg-white/5"
            >
              Retry now
            </button>
          </div>
        )}

        {/* Tab content */}
        <div className="flex-1 overflow-hidden min-h-0">
          {activeTab === "overview" && <OverviewTab />}
          {activeTab === "threads" && <ThreadsTab />}
          {activeTab === "runs" && <RunsTab />}
          {activeTab === "files" && <FilesTab />}
          {activeTab === "settings" && <SettingsPanel />}
        </div>
      </main>

      <CommandPaletteOverlay />

      {/* Workspace modal */}
      {showNewWorkspace && <WorkspaceModal />}

      {/* Toasts */}
      {(error || notice) && (
        <div className="fixed top-14 right-4 z-40 flex flex-col gap-2 max-w-sm pointer-events-none">
          {error && (
            <div className="pointer-events-auto flex items-start gap-2 rounded-lg border border-red-500/25 bg-red-950/80 px-4 py-3 text-sm text-red-300 shadow-lg backdrop-blur">
              <span className="flex-1 break-words">{error}</span>
              <button
                type="button"
                onClick={() => setError("")}
                className="shrink-0 text-red-500/60 hover:text-red-300"
              >
                ×
              </button>
            </div>
          )}
          {notice && (
            <div className="pointer-events-auto flex items-start gap-2 rounded-lg border border-[#8a9a7b]/25 bg-[#1a2016]/80 px-4 py-3 text-sm text-[#bec8b4] shadow-lg backdrop-blur">
              <span className="flex-1 break-words">{notice}</span>
              <button
                type="button"
                onClick={() => setNotice("")}
                className="shrink-0 text-[#8a9a7b]/60 hover:text-[#bec8b4]"
              >
                ×
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
