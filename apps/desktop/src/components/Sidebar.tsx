import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useApp } from "@/context/AppContext";
import { createConversation, patchConversation } from "@/api";
import { cn } from "@/lib/utils";
import { formatDate, workspaceTagsFromMetadata, type ViewTab } from "@/utils";
import {
  LayoutDashboard,
  MessageSquare,
  Play,
  FolderOpen,
  Plus,
  ChevronDown,
  ChevronRight,
  Circle,
  Settings,
  FileText,
} from "lucide-react";

const NAV_ITEMS: Array<{ tab: ViewTab; label: string; icon: typeof LayoutDashboard }> = [
  { tab: "overview", label: "Overview", icon: LayoutDashboard },
  { tab: "threads", label: "Threads", icon: MessageSquare },
  { tab: "runs", label: "Runs", icon: Play },
  { tab: "files", label: "Files", icon: FolderOpen },
];

type SectionKey = "conversations" | "runs";

function SectionChevron({ expanded }: { expanded: boolean }) {
  return (
    <span className="flex h-4 w-4 items-center justify-center shrink-0">
      {expanded ? (
        <ChevronDown size={12} className="text-zinc-600" />
      ) : (
        <ChevronRight size={12} className="text-zinc-600" />
      )}
    </span>
  );
}

function normalizeRunStatus(status: string): string {
  return String(status || "").trim().toLowerCase();
}

function runStatusLabel(status: string): string {
  const normalized = normalizeRunStatus(status);
  switch (normalized) {
    case "executing":
      return "Executing";
    case "planning":
      return "Planning";
    case "running":
      return "Running";
    case "paused":
      return "Paused";
    case "completed":
      return "Done";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    default:
      return normalized ? normalized[0]!.toUpperCase() + normalized.slice(1) : "Idle";
  }
}

function runStatusBadgeClass(status: string): string {
  const normalized = normalizeRunStatus(status);
  if (normalized === "executing") {
    return "bg-emerald-500/15 text-emerald-400";
  }
  if (normalized === "planning") {
    return "bg-sky-500/15 text-sky-400";
  }
  if (normalized === "running") {
    return "bg-cyan-500/15 text-cyan-400";
  }
  if (normalized === "paused") {
    return "bg-amber-500/15 text-amber-400";
  }
  if (normalized === "failed") {
    return "bg-red-500/15 text-red-400";
  }
  if (normalized === "completed") {
    return "bg-blue-500/15 text-blue-300";
  }
  if (normalized === "cancelled") {
    return "bg-zinc-800 text-zinc-500";
  }
  return "bg-zinc-800 text-zinc-500";
}

function runStatusDotClass(status: string): string {
  const normalized = normalizeRunStatus(status);
  if (normalized === "executing") {
    return "fill-emerald-400 text-emerald-400 animate-pulse";
  }
  if (normalized === "planning") {
    return "fill-sky-400 text-sky-400 animate-pulse";
  }
  if (normalized === "running") {
    return "fill-cyan-400 text-cyan-400 animate-pulse";
  }
  if (normalized === "paused") {
    return "fill-amber-400 text-amber-400";
  }
  if (normalized === "failed") {
    return "fill-red-400 text-red-400";
  }
  if (normalized === "completed") {
    return "fill-blue-300 text-blue-300";
  }
  return "fill-zinc-500 text-zinc-500";
}

export default function Sidebar() {
  const {
    activeTab,
    setActiveTab,
    workspaces,
    selectedWorkspaceId,
    selectedConversationId,
    setSelectedWorkspaceId,
    setSelectedConversationId,
    setSelectedRunId,
    showArchivedWorkspaces,
    setShowArchivedWorkspaces,
    setShowNewWorkspace,
    runtime,
    overview,
    conversationDetail,
    conversationIsProcessing,
    models,
    refreshWorkspaceSurface,
    refreshConversation,
    handleArchiveWorkspace,
    setError,
    setNotice,
  } = useApp();

  const [wsContextMenu, setWsContextMenu] = useState<{ wsId: string; x: number; y: number } | null>(null);

  const [creatingConversation, setCreatingConversation] = useState(false);
  const [showNewConvPopover, setShowNewConvPopover] = useState(false);
  const [newConvModel, setNewConvModel] = useState("");
  const [newConvPrompt, setNewConvPrompt] = useState("");
  const longPressTimer = useRef<number | null>(null);
  const convPlusRef = useRef<HTMLButtonElement>(null);
  const [popoverPos, setPopoverPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });

  const [renamingConvId, setRenamingConvId] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");

  const visibleWorkspaces = showArchivedWorkspaces
    ? workspaces
    : workspaces.filter((ws) => !ws.is_archived);

  // --- Local tree state ---------------------------------------------------

  const [expandedWorkspaces, setExpandedWorkspaces] = useState<Set<string>>(
    () => new Set(selectedWorkspaceId ? [selectedWorkspaceId] : []),
  );

  const [expandedSections, setExpandedSections] = useState<
    Record<string, Set<SectionKey>>
  >({});

  // Auto-expand when selected workspace changes
  useEffect(() => {
    if (selectedWorkspaceId) {
      setExpandedWorkspaces((prev) => {
        if (prev.has(selectedWorkspaceId)) return prev;
        const next = new Set(prev);
        next.add(selectedWorkspaceId);
        return next;
      });
    }
  }, [selectedWorkspaceId]);

  const toggleWorkspace = useCallback((id: string) => {
    setExpandedWorkspaces((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleSection = useCallback((wsId: string, section: SectionKey) => {
    setExpandedSections((prev) => {
      const current = prev[wsId] ?? new Set<SectionKey>();
      const next = new Set(current);
      if (next.has(section)) next.delete(section);
      else next.add(section);
      return { ...prev, [wsId]: next };
    });
  }, []);

  const isSectionExpanded = useCallback(
    (wsId: string, section: SectionKey) =>
      expandedSections[wsId]?.has(section) ?? false,
    [expandedSections],
  );

  // Conversations / runs for the currently-selected workspace from overview
  const wsConversations = useMemo(
    () => {
      const rows = overview?.recent_conversations ?? [];
      if (!selectedWorkspaceId || !conversationDetail || conversationDetail.workspace_id !== selectedWorkspaceId) {
        return rows;
      }
      const overlay = {
        id: conversationDetail.id,
        workspace_id: conversationDetail.workspace_id,
        workspace_path: conversationDetail.workspace_path,
        model_name: conversationDetail.model_name,
        title: conversationDetail.title,
        turn_count: conversationDetail.turn_count,
        total_tokens: conversationDetail.total_tokens,
        last_active_at: new Date().toISOString(),
        started_at: conversationDetail.started_at,
        is_active: conversationIsProcessing || conversationDetail.is_active,
        linked_run_ids: conversationDetail.linked_run_ids,
      };
      const deduped = rows.filter((conversation) => conversation.id !== overlay.id);
      return [overlay, ...deduped];
    },
    [overview, selectedWorkspaceId, conversationDetail, conversationIsProcessing],
  );
  const wsRuns = useMemo(() => overview?.recent_runs ?? [], [overview]);
  // --- Handlers -----------------------------------------------------------

  const handleSelectWorkspace = useCallback(
    (id: string) => {
      setSelectedWorkspaceId(id);
    },
    [setSelectedWorkspaceId],
  );

  // Quick-create conversation with defaults (click)
  // Long-press shows popover with model/prompt options
  const handleQuickCreateConversation = useCallback(async () => {
    if (!selectedWorkspaceId || creatingConversation) return;
    setCreatingConversation(true);
    setError("");
    try {
      const created = await createConversation(selectedWorkspaceId, {});
      await refreshWorkspaceSurface(selectedWorkspaceId);
      setSelectedConversationId(created.id);
      setActiveTab("threads");
      setNotice(`Started conversation ${created.title || created.id.slice(0, 8)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create conversation.");
    } finally {
      setCreatingConversation(false);
    }
  }, [selectedWorkspaceId, creatingConversation, setActiveTab, setSelectedConversationId, refreshWorkspaceSurface, setError, setNotice]);

  const handleAdvancedCreateConversation = useCallback(async () => {
    if (!selectedWorkspaceId || creatingConversation) return;
    setCreatingConversation(true);
    setError("");
    try {
      const created = await createConversation(selectedWorkspaceId, {
        model_name: newConvModel.trim() || undefined,
        system_prompt: newConvPrompt.trim() || undefined,
      });
      await refreshWorkspaceSurface(selectedWorkspaceId);
      setSelectedConversationId(created.id);
      setActiveTab("threads");
      setShowNewConvPopover(false);
      setNewConvModel("");
      setNewConvPrompt("");
      setNotice(`Started conversation ${created.title || created.id.slice(0, 8)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create conversation.");
    } finally {
      setCreatingConversation(false);
    }
  }, [selectedWorkspaceId, creatingConversation, newConvModel, newConvPrompt, setActiveTab, setSelectedConversationId, refreshWorkspaceSurface, setError, setNotice]);

  const handleRevealWorkspace = useCallback((_path: string) => {
    // This is a convenience action — in Tauri it would open the folder
    // For now, just show a notice
    setNotice(`Workspace path: ${_path}`);
  }, [setNotice]);

  const handleConvPlusMouseDown = useCallback(() => {
    longPressTimer.current = window.setTimeout(() => {
      longPressTimer.current = null;
      if (convPlusRef.current) {
        const rect = convPlusRef.current.getBoundingClientRect();
        setPopoverPos({ top: rect.bottom + 4, left: rect.right + 8 });
      }
      setShowNewConvPopover(true);
    }, 300);
  }, []);

  const handleConvPlusMouseUp = useCallback(() => {
    if (longPressTimer.current !== null) {
      window.clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
      void handleQuickCreateConversation();
    }
  }, [handleQuickCreateConversation]);

  const handleConvPlusMouseLeave = useCallback(() => {
    if (longPressTimer.current !== null) {
      window.clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
    }
  }, []);

  const handleConvPlusRightClick = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    if (convPlusRef.current) {
      const rect = convPlusRef.current.getBoundingClientRect();
      setPopoverPos({ top: rect.bottom + 4, left: rect.right + 8 });
    }
    setShowNewConvPopover(true);
  }, []);

  const handleSelectConversation = useCallback(
    (convId: string) => {
      setSelectedConversationId(convId);
      setActiveTab("threads");
    },
    [setSelectedConversationId, setActiveTab],
  );

  const handleSelectRun = useCallback(
    (runId: string) => {
      setSelectedRunId(runId);
      setActiveTab("runs");
    },
    [setSelectedRunId, setActiveTab],
  );

  return (
    <aside className="flex h-full flex-col bg-[#0f0f12] border-r border-zinc-800/60">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-4 pt-4 pb-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#6b7a5e]">
          <span className="text-xs font-bold text-white leading-none">L</span>
        </div>
        <span className="text-[15px] font-semibold tracking-tight text-zinc-100">Loom</span>
      </div>

      {/* Divider */}
      <div className="mx-3 my-2 h-px bg-zinc-800/60" />

      {/* Nav tabs */}
      <nav className="flex flex-col gap-0.5 px-2 pb-1">
        {NAV_ITEMS.map(({ tab, label, icon: Icon }) => {
          const isActive = activeTab === tab;
          const count =
            tab === "threads"
                ? (overview?.recent_conversations.length ?? 0)
                : tab === "runs"
                  ? (overview?.recent_runs.length ?? 0)
                  : 0;
          return (
            <button
              key={tab}
              type="button"
              onClick={() => setActiveTab(tab)}
              className={cn(
                "group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors",
                isActive
                  ? "bg-[#8a9a7b]/15 text-[#a3b396]"
                  : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200",
              )}
            >
              <Icon
                size={15}
                className={cn(
                  "shrink-0",
                  isActive ? "text-[#a3b396]" : "text-zinc-500 group-hover:text-zinc-300",
                )}
              />
              <span className="truncate">{label}</span>
              {count > 0 && (
                <span
                  className="ml-auto flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-zinc-800 px-1 text-[10px] font-semibold tabular-nums text-zinc-500"
                >
                  {count}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Divider */}
      <div className="mx-3 my-2 h-px bg-zinc-800/60" />

      {/* Workspaces tree */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex items-center justify-between px-4 pb-1.5">
          <span className="text-[10.5px] font-semibold uppercase tracking-widest text-zinc-600">
            Workspaces
          </span>
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={() => setShowArchivedWorkspaces(!showArchivedWorkspaces)}
              className={cn(
                "flex h-5 items-center gap-0.5 rounded px-1 text-[10px] font-medium transition-colors",
                showArchivedWorkspaces ? "text-[#a3b396]" : "text-zinc-600 hover:text-zinc-400",
              )}
            >
              <ChevronDown
                size={10}
                className={cn("transition-transform", showArchivedWorkspaces && "rotate-180")}
              />
              {showArchivedWorkspaces ? "All" : "Active"}
            </button>
            <button
              type="button"
              onClick={() => setShowNewWorkspace(true)}
              className="flex h-5 w-5 items-center justify-center rounded text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
              title="Add workspace"
            >
              <Plus size={12} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-2 pb-2">
          {visibleWorkspaces.length === 0 && (
            <button
              type="button"
              onClick={() => setShowNewWorkspace(true)}
              className="mx-1 mt-2 flex w-full flex-col items-center gap-2 rounded-lg border border-dashed border-zinc-800 px-3 py-5 text-center transition-colors hover:border-zinc-700 hover:bg-zinc-900/40"
            >
              <Plus size={16} className="text-zinc-600" />
              <span className="text-xs text-zinc-500">Add your first workspace</span>
            </button>
          )}

          <div className="flex flex-col gap-0.5">
            {visibleWorkspaces.map((ws) => {
              const isSelected = ws.id === selectedWorkspaceId;
              const isExpanded = expandedWorkspaces.has(ws.id);
              const wsName =
                ws.display_name || ws.canonical_path.split("/").pop() || "Untitled";

              // Only show nested detail for the selected workspace (we only
              // have overview data for the selected workspace).
              const showNested = isExpanded && isSelected;

              const convCount = showNested ? wsConversations.length : ws.conversation_count;
              const runCount = showNested ? wsRuns.length : ws.run_count;

              return (
                <div
                  key={ws.id}
                  className={cn(
                    "flex flex-col rounded-lg transition-all",
                    isSelected
                      ? "bg-[#8a9a7b]/10 ring-1 ring-[#8a9a7b]/25"
                      : "",
                  )}
                >
                  {/* Workspace row */}
                  <button
                    type="button"
                    onClick={() => {
                      handleSelectWorkspace(ws.id);
                      toggleWorkspace(ws.id);
                    }}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      handleSelectWorkspace(ws.id);
                      setWsContextMenu({ wsId: ws.id, x: e.clientX, y: e.clientY });
                    }}
                    className={cn(
                      "group flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left transition-all",
                      isSelected
                        ? ""
                        : "hover:bg-zinc-800/40",
                    )}
                  >
                    {isExpanded ? (
                      <ChevronDown size={12} className="shrink-0 text-zinc-600" />
                    ) : (
                      <ChevronRight size={12} className="shrink-0 text-zinc-600" />
                    )}
                    {ws.active_run_count > 0 ? (
                      <Circle size={7} className="shrink-0 fill-emerald-400 text-emerald-400" />
                    ) : (
                      <Circle
                        size={7}
                        className={cn(
                          "shrink-0",
                          isSelected
                            ? "fill-[#8a9a7b] text-[#8a9a7b]"
                            : "fill-zinc-700 text-zinc-700",
                        )}
                      />
                    )}
                    <span
                      className={cn(
                        "truncate text-[13px] font-medium",
                        isSelected ? "text-zinc-100" : "text-zinc-300 group-hover:text-zinc-100",
                      )}
                    >
                      {wsName}
                    </span>
                    {ws.is_archived && (
                      <span className="shrink-0 rounded bg-zinc-800 px-1 py-px text-[9px] text-zinc-600">
                        archived
                      </span>
                    )}
                  </button>

                  {/* Nested sections */}
                  {showNested && (
                    <div className="flex flex-col gap-px pl-4 pt-0.5">
                      {/* ---- Conversations section ---- */}
                      <div className="flex flex-col">
                        <div className="flex items-center">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              const wasExpanded = isSectionExpanded(ws.id, "conversations");
                              toggleSection(ws.id, "conversations");
                              if (!wasExpanded) {
                                setActiveTab("threads");
                              }
                            }}
                            className="group flex flex-1 items-center gap-1.5 rounded px-2 py-1 text-[11.5px] font-medium text-zinc-500 hover:text-zinc-300 transition-colors cursor-pointer"
                          >
                            <SectionChevron
                              expanded={isSectionExpanded(ws.id, "conversations")}
                            />
                            <MessageSquare size={12} className="shrink-0" />
                            <span>Threads</span>
                            <span className="ml-auto text-[10px] tabular-nums text-zinc-600">
                              {convCount}
                            </span>
                          </button>
                          <div className="relative mr-1">
                            <button
                              ref={convPlusRef}
                              type="button"
                              onMouseDown={handleConvPlusMouseDown}
                              onMouseUp={handleConvPlusMouseUp}
                              onMouseLeave={handleConvPlusMouseLeave}
                              onContextMenu={handleConvPlusRightClick}
                              disabled={creatingConversation || !selectedWorkspaceId}
                              className="flex h-5 w-5 items-center justify-center rounded text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300 transition-colors disabled:opacity-40"
                              title="Click: new thread · Hold or right-click: options"
                            >
                              {creatingConversation ? (
                                <span className="h-2.5 w-2.5 rounded-full border border-zinc-500 border-t-[#a3b396] animate-spin" />
                              ) : (
                                <Plus size={11} />
                              )}
                            </button>
                            {/* Long-press popover — rendered fixed to escape sidebar overflow */}
                            {showNewConvPopover && (
                              <>
                                <div className="fixed inset-0 z-[100]" onClick={() => setShowNewConvPopover(false)} />
                                <div className="fixed z-[101] w-56 rounded-lg border border-zinc-700/60 bg-[#111114] shadow-xl p-3 space-y-2" style={{ top: popoverPos.top, left: popoverPos.left }}>
                                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">New thread</p>
                                  <div>
                                    <label className="text-[10px] text-zinc-500 block mb-0.5">Model</label>
                                    <select
                                      value={newConvModel}
                                      onChange={(e) => setNewConvModel(e.target.value)}
                                      className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-200 outline-none focus:border-[#8a9a7b]/50"
                                    >
                                      <option value="">Default ({models[0]?.model_id || models[0]?.name || "primary"})</option>
                                      {models.map((m) => (
                                        <option key={m.name} value={m.name}>
                                          {m.model_id || m.name} ({m.name})
                                        </option>
                                      ))}
                                    </select>
                                  </div>
                                  <div>
                                    <label className="text-[10px] text-zinc-500 block mb-0.5">System prompt</label>
                                    <textarea
                                      value={newConvPrompt}
                                      onChange={(e) => setNewConvPrompt(e.target.value)}
                                      placeholder="Optional override..."
                                      rows={2}
                                      className="w-full rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-200 placeholder:text-zinc-600 resize-none outline-none focus:border-[#8a9a7b]/50"
                                    />
                                  </div>
                                  <button
                                    type="button"
                                    onClick={() => void handleAdvancedCreateConversation()}
                                    disabled={creatingConversation}
                                    className="w-full rounded bg-[#6b7a5e] px-2 py-1 text-[11px] font-medium text-white hover:bg-[#8a9a7b] disabled:opacity-50 transition-colors"
                                  >
                                    {creatingConversation ? "Creating..." : "Create"}
                                  </button>
                                </div>
                              </>
                            )}
                          </div>
                        </div>
                        {isSectionExpanded(ws.id, "conversations") && (
                          <div className="flex flex-col gap-px pl-6 pb-1">
                            {wsConversations.length === 0 && (
                              <span className="px-2 py-1 text-[11px] text-zinc-700 italic">
                                No threads
                              </span>
                            )}
                            {wsConversations.map((conv) => (
                              <button
                                key={conv.id}
                                type="button"
                                onClick={() => handleSelectConversation(conv.id)}
                                onDoubleClick={(e) => {
                                  e.stopPropagation();
                                  setRenamingConvId(conv.id);
                                  setRenameText(conv.title || `conv-${conv.id.slice(0, 6)}`);
                                }}
                                className="flex items-center gap-1.5 rounded px-2 py-1 text-[11.5px] text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200 transition-colors text-left"
                              >
                                <FileText size={11} className="shrink-0 text-zinc-600" />
                                {renamingConvId === conv.id ? (
                                  <input
                                    type="text"
                                    value={renameText}
                                    onChange={(e) => setRenameText(e.target.value)}
                                    onKeyDown={async (e) => {
                                      if (e.key === "Enter") {
                                        e.preventDefault();
                                        const trimmed = renameText.trim();
                                        if (trimmed && trimmed !== conv.title) {
                                          try {
                                            await patchConversation(conv.id, { title: trimmed });
                                            await refreshConversation(conv.id);
                                            if (selectedWorkspaceId) {
                                              await refreshWorkspaceSurface(selectedWorkspaceId);
                                            }
                                          } catch (err) {
                                            setError(err instanceof Error ? err.message : "Failed to rename thread.");
                                          }
                                        }
                                        setRenamingConvId(null);
                                        setRenameText("");
                                      } else if (e.key === "Escape") {
                                        setRenamingConvId(null);
                                        setRenameText("");
                                      }
                                    }}
                                    onBlur={async () => {
                                      const trimmed = renameText.trim();
                                      if (trimmed && trimmed !== conv.title) {
                                        try {
                                          await patchConversation(conv.id, { title: trimmed });
                                          await refreshConversation(conv.id);
                                          if (selectedWorkspaceId) {
                                            await refreshWorkspaceSurface(selectedWorkspaceId);
                                          }
                                        } catch (err) {
                                          setError(err instanceof Error ? err.message : "Failed to rename thread.");
                                        }
                                      }
                                      setRenamingConvId(null);
                                      setRenameText("");
                                    }}
                                    onClick={(e) => e.stopPropagation()}
                                    onDoubleClick={(e) => e.stopPropagation()}
                                    autoFocus
                                    className="min-w-0 flex-1 truncate rounded bg-zinc-900/80 px-1 py-0 text-[11.5px] text-zinc-200 outline-none border border-transparent focus:border-[#8a9a7b]/50 placeholder:text-zinc-600"
                                  />
                                ) : (
                                  <span className={cn(
                                    "truncate",
                                    conv.id === selectedConversationId && "text-zinc-200",
                                  )}>
                                    {conv.title || `conv-${conv.id.slice(0, 6)}`}
                                  </span>
                                )}
                                {conv.id === selectedConversationId && conversationIsProcessing ? (
                                  <Circle
                                    size={5}
                                    className="shrink-0 fill-[#a3b396] text-[#a3b396] animate-pulse ml-auto"
                                  />
                                ) : conv.is_active && (
                                  <Circle
                                    size={5}
                                    className="shrink-0 fill-sky-400 text-sky-400 ml-auto"
                                  />
                                )}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* ---- Runs section ---- */}
                      <div className="flex flex-col">
                        <div className="flex items-center">
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              const wasExpanded = isSectionExpanded(ws.id, "runs");
                              toggleSection(ws.id, "runs");
                              if (!wasExpanded) {
                                setActiveTab("runs");
                              }
                            }}
                            className="group flex flex-1 items-center gap-1.5 rounded px-2 py-1 text-[11.5px] font-medium text-zinc-500 hover:text-zinc-300 transition-colors cursor-pointer"
                          >
                            <SectionChevron expanded={isSectionExpanded(ws.id, "runs")} />
                            <Play size={12} className="shrink-0" />
                            <span>Runs</span>
                            <span className="ml-auto text-[10px] tabular-nums text-zinc-600">
                              {runCount}
                            </span>
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setSelectedRunId("");
                              setActiveTab("runs");
                            }}
                            className="mr-1 flex h-5 w-5 items-center justify-center rounded text-zinc-600 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
                            title="New run"
                          >
                            <Plus size={11} />
                          </button>
                        </div>
                        {isSectionExpanded(ws.id, "runs") && (
                          <div className="flex flex-col gap-px pl-6 pb-1">
                            {wsRuns.length === 0 && (
                              <span className="px-2 py-1 text-[11px] text-zinc-700 italic">
                                No runs
                              </span>
                            )}
                            {wsRuns.map((run) => (
                              <button
                                key={run.id}
                                type="button"
                                onClick={() => handleSelectRun(run.id)}
                                className="flex items-center gap-1.5 rounded px-2 py-1 text-[11.5px] text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200 transition-colors text-left"
                              >
                                <Play size={11} className="shrink-0 text-zinc-600" />
                                <span className="truncate">
                                  {run.goal || `run-${run.id.slice(0, 6)}`}
                                </span>
                                <span
                                  className={cn(
                                    "ml-auto inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[9px] font-medium tabular-nums",
                                    runStatusBadgeClass(run.status),
                                  )}
                                  title={`Run status: ${runStatusLabel(run.status)}`}
                                >
                                  <Circle
                                    size={5}
                                    className={cn(
                                      "shrink-0",
                                      runStatusDotClass(run.status),
                                    )}
                                  />
                                  <span className="truncate">{runStatusLabel(run.status)}</span>
                                </span>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* ---- Files link ---- */}
                      <button
                        type="button"
                        onClick={() => setActiveTab("files")}
                        className="group flex items-center gap-1.5 rounded px-2 py-1 text-[11.5px] font-medium text-zinc-500 hover:text-zinc-300 transition-colors"
                      >
                        <FolderOpen size={12} className="shrink-0" />
                        <span>Files</span>
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="mx-3 h-px bg-zinc-800/60" />

      {/* Settings */}
      <div className="px-2 py-1.5">
        <button
          type="button"
          onClick={() => setActiveTab("settings")}
          className={cn(
            "group flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] font-medium transition-colors",
            activeTab === "settings"
              ? "bg-[#8a9a7b]/15 text-[#a3b396]"
              : "text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-300",
          )}
        >
          <Settings
            size={15}
            className={cn(
              "shrink-0",
              activeTab === "settings"
                ? "text-[#a3b396]"
                : "text-zinc-600 group-hover:text-zinc-300",
            )}
          />
          <span>Settings</span>
        </button>
      </div>

      {/* Divider */}
      <div className="mx-3 h-px bg-zinc-800/60" />

      {/* Footer -- Runtime */}
      <div className="px-3 py-2.5">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "h-[6px] w-[6px] shrink-0 rounded-full",
              runtime?.ready
                ? "bg-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.35)]"
                : "bg-yellow-500 shadow-[0_0_5px_rgba(234,179,8,0.35)]",
            )}
          />
          <span className="truncate text-[11px] text-zinc-500">
            {runtime?.ready ? "Connected" : "Starting..."}
            {runtime?.version ? ` \u00b7 ${runtime.version}` : ""}
          </span>
        </div>
      </div>

      {/* Workspace context menu */}
      {wsContextMenu && (
        <>
          <div className="fixed inset-0 z-[100]" onClick={() => setWsContextMenu(null)} />
          <div
            className="fixed z-[101] w-40 rounded-lg border border-zinc-700/60 bg-[#111114] shadow-xl py-1"
            style={{ top: wsContextMenu.y, left: wsContextMenu.x }}
          >
            {(() => {
              const targetWs = workspaces.find((w) => w.id === wsContextMenu.wsId);
              if (!targetWs) return null;
              return (
                <>
                  <button
                    type="button"
                    onClick={() => {
                      setWsContextMenu(null);
                      void handleArchiveWorkspace(!targetWs.is_archived);
                    }}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-left text-zinc-300 hover:bg-zinc-800 transition-colors"
                  >
                    {targetWs.is_archived ? "Restore workspace" : "Archive workspace"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setWsContextMenu(null);
                      handleRevealWorkspace(targetWs.canonical_path);
                    }}
                    className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-left text-zinc-300 hover:bg-zinc-800 transition-colors"
                  >
                    Reveal in Finder
                  </button>
                </>
              );
            })()}
          </div>
        </>
      )}
    </aside>
  );
}
