import {
  MessageSquare,
  Play,
  Zap,
  FolderOpen,
  Clock,
  Activity,
  Plus,
  Circle,
  ArrowRight,
} from "lucide-react";
import { useApp } from "@/context/AppContext";
import { formatDate } from "@/utils";
import { notificationSummary } from "@/history";
import { cn } from "@/lib/utils";

export default function OverviewTab() {
  const {
    overview,
    loadingOverview,
    selectedWorkspaceSummary,
    noWorkspacesRegistered,
    selectedWorkspaceIsEmpty,
    selectedWorkspaceTags,
    selectedWorkspaceNote,
    recentNotifications,
    recentWorkspaceArtifacts,
    approvalInbox,
    setActiveTab,
    focusConversationComposer,
    focusRunComposer,
    handlePrefillStarterWorkspace,
    setSelectedConversationId,
    setSelectedRunId,
    setShowNewWorkspace,
  } = useApp();

  // ---------------------------------------------------------------------------
  // Loading
  // ---------------------------------------------------------------------------

  if (loadingOverview && !overview) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="flex flex-col items-center gap-3 text-zinc-500">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-700 border-t-[#8a9a7b]" />
          <span className="text-sm">Loading workspace...</span>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // No workspaces — welcome screen
  // ---------------------------------------------------------------------------

  if (noWorkspacesRegistered) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="max-w-md text-center">
          <div className="mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-[#6b7a5e]/15">
            <FolderOpen className="h-7 w-7 text-[#a3b396]" />
          </div>
          <h2 className="text-xl font-bold text-zinc-100 mb-2">Welcome to Loom</h2>
          <p className="text-sm text-zinc-400 mb-8 leading-relaxed">
            Get started by connecting a workspace — a directory where Loom will
            manage threads, runs, and artifacts.
          </p>
          <button
            type="button"
            onClick={() => setShowNewWorkspace(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-[#6b7a5e] px-5 py-2.5 text-sm font-semibold text-white hover:bg-[#8a9a7b] transition-colors"
          >
            <Plus size={16} />
            Add workspace
          </button>
          <p className="mt-4 text-xs text-zinc-600">
            Or press <kbd className="rounded border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-[10px]">⌘K</kbd> and
            type "starter workspace" to quick-start.
          </p>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Workspace selected but empty
  // ---------------------------------------------------------------------------

  if (selectedWorkspaceIsEmpty) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <div className="max-w-lg text-center">
          <div className="mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/15">
            <Zap className="h-7 w-7 text-emerald-400" />
          </div>
          <h2 className="text-xl font-bold text-zinc-100 mb-2">
            {selectedWorkspaceSummary?.display_name || "Workspace"} is ready
          </h2>
          <p className="text-sm text-zinc-400 mb-8 leading-relaxed">
            This workspace has no activity yet. Start a thread for interactive
            work or launch a run for autonomous execution.
          </p>
          <div className="flex items-center justify-center gap-3">
            <button
              type="button"
              onClick={() => { setActiveTab("threads"); focusConversationComposer(); }}
              className="inline-flex items-center gap-2 rounded-lg bg-zinc-800 px-5 py-2.5 text-sm font-medium text-zinc-200 hover:bg-zinc-700 transition-colors"
            >
              <MessageSquare size={15} />
              New thread
            </button>
            <button
              type="button"
              onClick={() => { setActiveTab("runs"); focusRunComposer(); }}
              className="inline-flex items-center gap-2 rounded-lg bg-[#6b7a5e] px-5 py-2.5 text-sm font-semibold text-white hover:bg-[#8a9a7b] transition-colors"
            >
              <Play size={15} />
              Launch run
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Main overview
  // ---------------------------------------------------------------------------

  const workspace = overview?.workspace ?? selectedWorkspaceSummary;
  const activeRuns = workspace?.active_run_count ?? 0;
  const pendingApprovals = overview?.pending_approvals_count ?? 0;
  const recentConversations = overview?.recent_conversations ?? [];
  const recentRuns = overview?.recent_runs ?? [];

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-6 space-y-6">

        {/* Hero strip — key metrics */}
        <div className="flex items-center gap-4 flex-wrap">
          <Metric
            icon={<Activity size={14} />}
            label="Active runs"
            value={activeRuns}
            accent={activeRuns > 0}
          />
          <Metric
            icon={<Clock size={14} />}
            label="Pending"
            value={pendingApprovals}
            warn={pendingApprovals > 0}
          />
          <Metric
            icon={<MessageSquare size={14} />}
            label="Threads"
            value={recentConversations.length}
          />
          <Metric
            icon={<Play size={14} />}
            label="Runs"
            value={recentRuns.length}
          />
          {selectedWorkspaceTags.length > 0 && (
            <div className="flex items-center gap-1.5 ml-auto">
              {selectedWorkspaceTags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-md bg-zinc-800/80 px-2 py-0.5 text-[10.5px] text-zinc-500"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>

        {selectedWorkspaceNote && (
          <p className="text-[13px] text-zinc-500 italic border-l-2 border-zinc-800 pl-3">
            {selectedWorkspaceNote}
          </p>
        )}

        {/* Quick actions */}
        <div className="flex items-center gap-2">
          <ActionBtn
            icon={<MessageSquare size={14} />}
            label="New thread"
            onClick={() => { setActiveTab("threads"); focusConversationComposer(); }}
          />
          <ActionBtn
            icon={<Play size={14} />}
            label="Launch run"
            onClick={() => { setActiveTab("runs"); focusRunComposer(); }}
            primary
          />
        </div>

        {/* Two-column layout: conversations + runs side by side */}
        <div className="grid gap-4 lg:grid-cols-2">
          {/* Recent conversations */}
          <section className="space-y-2">
            <SectionHeader
              icon={<MessageSquare size={13} />}
              title="Recent threads"
              action={{ label: "View all", onClick: () => setActiveTab("threads") }}
            />
            {recentConversations.length === 0 ? (
              <EmptyCard message="No threads yet" />
            ) : (
              <div className="space-y-1">
                {recentConversations.slice(0, 5).map((conv) => (
                  <button
                    key={conv.id}
                    type="button"
                    className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/50"
                    onClick={() => {
                      setSelectedConversationId(conv.id);
                      setActiveTab("threads");
                    }}
                  >
                    <MessageSquare size={13} className="shrink-0 text-zinc-600" />
                    <div className="flex-1 min-w-0">
                      <p className="text-[13px] font-medium text-zinc-200 truncate">
                        {conv.title || "Untitled"}
                      </p>
                      <p className="text-[11px] text-zinc-600">
                        {conv.model_name} · {conv.turn_count} turn{conv.turn_count !== 1 ? "s" : ""}
                      </p>
                    </div>
                    <span className="shrink-0 text-[10px] text-zinc-700">
                      {formatDate(conv.last_active_at)}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </section>

          {/* Recent runs */}
          <section className="space-y-2">
            <SectionHeader
              icon={<Play size={13} />}
              title="Recent runs"
              action={{ label: "View all", onClick: () => setActiveTab("runs") }}
            />
            {recentRuns.length === 0 ? (
              <EmptyCard message="No runs yet" />
            ) : (
              <div className="space-y-1">
                {recentRuns.slice(0, 5).map((run) => (
                  <button
                    key={run.id}
                    type="button"
                    className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-zinc-800/50"
                    onClick={() => {
                      setSelectedRunId(run.id);
                      setActiveTab("runs");
                    }}
                  >
                    <RunDot status={run.status} />
                    <div className="flex-1 min-w-0">
                      <p className="text-[13px] font-medium text-zinc-200 truncate">
                        {run.goal || "Untitled"}
                      </p>
                      <div className="flex items-center gap-2 text-[11px] text-zinc-600">
                        <StatusPill status={run.status} />
                        {run.process_name && <span>{run.process_name}</span>}
                      </div>
                    </div>
                    <span className="shrink-0 text-[10px] text-zinc-700">
                      {formatDate(run.updated_at)}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </section>
        </div>

        {/* Notifications */}
        {recentNotifications.length > 0 && (
          <section className="space-y-2">
            <SectionHeader icon={<Zap size={13} />} title="Live activity" />
            <div className="space-y-1">
              {recentNotifications.slice(0, 4).map((event) => (
                <div
                  key={event.id}
                  className="flex items-start gap-2.5 rounded-lg px-3 py-2 bg-zinc-900/40"
                >
                  <Zap size={12} className="shrink-0 text-amber-500 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <p className="text-[12.5px] text-zinc-300 truncate">
                      {notificationSummary(event)}
                    </p>
                    <p className="text-[10px] text-zinc-600">{formatDate(event.created_at)}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Recent artifacts */}
        {recentWorkspaceArtifacts.length > 0 && (
          <section className="space-y-2">
            <SectionHeader
              icon={<FolderOpen size={13} />}
              title="Recent files"
              action={{ label: "Files tab", onClick: () => setActiveTab("files") }}
            />
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {recentWorkspaceArtifacts.slice(0, 6).map((a) => (
                <div
                  key={`${a.path}-${a.sha256}`}
                  className="rounded-lg border border-zinc-800/60 bg-zinc-900/30 px-3 py-2.5"
                >
                  <p className="text-[12px] text-zinc-300 truncate font-mono">{a.path}</p>
                  <p className="text-[10.5px] text-zinc-600 mt-0.5">
                    {a.category}
                    {a.run_count > 0 && ` · ${a.run_count} run${a.run_count !== 1 ? "s" : ""}`}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Metric({
  icon,
  label,
  value,
  accent,
  warn,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
  accent?: boolean;
  warn?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 rounded-lg border px-3 py-2",
        warn
          ? "border-amber-500/20 bg-amber-500/5"
          : accent
            ? "border-emerald-500/20 bg-emerald-500/5"
            : "border-zinc-800/60 bg-zinc-900/30",
      )}
    >
      <span
        className={cn(
          "text-zinc-500",
          warn && "text-amber-500",
          accent && "text-emerald-500",
        )}
      >
        {icon}
      </span>
      <span
        className={cn(
          "text-sm font-bold tabular-nums",
          warn ? "text-amber-400" : accent ? "text-emerald-400" : "text-zinc-200",
        )}
      >
        {value}
      </span>
      <span className="text-[10.5px] text-zinc-600">{label}</span>
    </div>
  );
}

function ActionBtn({
  icon,
  label,
  onClick,
  primary,
  warn,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  primary?: boolean;
  warn?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-lg px-3.5 py-2 text-[13px] font-medium transition-colors",
        primary
          ? "bg-[#6b7a5e] text-white hover:bg-[#8a9a7b]"
          : warn
            ? "bg-amber-500/10 text-amber-400 hover:bg-amber-500/20"
            : "bg-zinc-800/60 text-zinc-300 hover:bg-zinc-800",
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function SectionHeader({
  icon,
  title,
  action,
}: {
  icon: React.ReactNode;
  title: string;
  action?: { label: string; onClick: () => void };
}) {
  return (
    <div className="flex items-center justify-between">
      <h3 className="flex items-center gap-1.5 text-[11.5px] font-semibold uppercase tracking-wider text-zinc-500">
        {icon}
        {title}
      </h3>
      {action && (
        <button
          type="button"
          onClick={action.onClick}
          className="flex items-center gap-0.5 text-[11px] text-zinc-600 hover:text-zinc-400 transition-colors"
        >
          {action.label}
          <ArrowRight size={10} />
        </button>
      )}
    </div>
  );
}

function EmptyCard({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-dashed border-zinc-800 px-4 py-6 text-center text-[12.5px] text-zinc-600">
      {message}
    </div>
  );
}

function RunDot({ status }: { status: string }) {
  const s = status.toLowerCase();
  return (
    <Circle
      size={8}
      className={cn(
        "shrink-0",
        s === "executing" || s === "planning" || s === "running"
          ? "fill-emerald-400 text-emerald-400 animate-pulse"
          : s === "failed"
            ? "fill-red-400 text-red-400"
            : s === "completed"
              ? "fill-emerald-400 text-emerald-400"
              : s === "paused"
                ? "fill-amber-400 text-amber-400"
                : s === "cancelled"
                  ? "fill-zinc-600 text-zinc-600"
                  : "fill-zinc-700 text-zinc-700",
      )}
    />
  );
}

function StatusPill({ status }: { status: string }) {
  const s = status.toLowerCase();
  return (
    <span
      className={cn(
        "rounded px-1.5 py-px text-[10px] font-medium",
        s === "executing" || s === "planning" || s === "running"
          ? "bg-emerald-500/15 text-emerald-400"
          : s === "completed"
            ? "bg-emerald-500/15 text-emerald-300"
            : s === "failed"
              ? "bg-red-500/15 text-red-400"
              : s === "paused"
                ? "bg-amber-500/15 text-amber-400"
                : s === "cancelled"
                  ? "bg-zinc-800 text-zinc-600"
                  : "bg-zinc-800 text-zinc-500",
      )}
    >
      {status}
    </span>
  );
}
