import { useState } from "react";
import {
  shallowEqual,
  useAppActions,
  useAppSelector,
} from "@/context/AppContext";
import { cn } from "@/lib/utils";
import { defaultWorkspaceName } from "@/utils";
import { createWorkspace } from "@/api";
import { X, FolderOpen, Loader2 } from "lucide-react";

async function pickFolder(): Promise<string | null> {
  try {
    const mod = await import("@tauri-apps/plugin-dialog");
    const selected = await mod.open({ directory: true, multiple: false });
    return typeof selected === "string" ? selected : null;
  } catch {
    // Not in Tauri — fall back to prompt
    const path = window.prompt("Enter folder path:");
    return path?.trim() || null;
  }
}

export default function WorkspaceModal() {
  const { showNewWorkspace } = useAppSelector((state) => ({
    showNewWorkspace: state.showNewWorkspace,
  }), shallowEqual);
  const {
    setShowNewWorkspace,
    refreshWorkspaceList,
    setError,
    setNotice,
  } = useAppActions();

  const [importing, setImporting] = useState(false);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState("");

  if (!showNewWorkspace) return null;

  async function handlePickFolder() {
    const path = await pickFolder();
    if (path) {
      setSelectedPath(path);
      setDisplayName(defaultWorkspaceName(path));
    }
  }

  async function handleAdd() {
    if (!selectedPath) return;
    setImporting(true);
    setError("");
    setNotice("");
    try {
      const created = await createWorkspace({
        path: selectedPath,
        display_name: displayName.trim() || defaultWorkspaceName(selectedPath),
      });
      await refreshWorkspaceList(created.id);
      setNotice(`Added workspace ${created.display_name}`);
      setShowNewWorkspace(false);
      setSelectedPath(null);
      setDisplayName("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add workspace.");
    } finally {
      setImporting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={() => setShowNewWorkspace(false)}
    >
      <div
        className="w-full max-w-md rounded-xl border border-zinc-800/60 bg-[#0f0f12] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Add workspace"
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800/60 px-6 py-4">
          <h2 className="text-base font-semibold text-zinc-100">Add Workspace</h2>
          <button
            type="button"
            onClick={() => setShowNewWorkspace(false)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
            aria-label="Close"
          >
            <X size={16} />
          </button>
        </div>

        <div className="px-6 py-6 space-y-5">
          {!selectedPath ? (
            /* Step 1: Pick a folder */
            <>
              <p className="text-sm text-zinc-400 text-center leading-relaxed">
                Choose a folder to use as a workspace. Loom will track
                threads, runs, and artifacts here.
              </p>

              <button
                type="button"
                onClick={handlePickFolder}
                className="flex w-full flex-col items-center gap-3 rounded-xl border-2 border-dashed border-zinc-700/60 px-6 py-8 text-center transition-colors hover:border-[#8a9a7b]/40 hover:bg-[#8a9a7b]/5"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#6b7a5e]/15">
                  <FolderOpen className="h-6 w-6 text-[#a3b396]" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-zinc-200">Choose folder</p>
                  <p className="text-xs text-zinc-500 mt-0.5">
                    Pick any directory — existing projects or new empty folders
                  </p>
                </div>
              </button>
            </>
          ) : (
            /* Step 2: Confirm and name */
            <>
              <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 px-4 py-3">
                <p className="text-[10.5px] font-medium uppercase tracking-wider text-zinc-600 mb-1">
                  Selected folder
                </p>
                <p className="text-sm text-zinc-200 font-mono break-all">{selectedPath}</p>
              </div>

              <label className="block">
                <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-500 mb-1.5 block">
                  Display name
                </span>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder={defaultWorkspaceName(selectedPath)}
                  autoFocus
                  className="w-full rounded-lg border border-zinc-700/60 bg-zinc-800/40 px-3 py-2.5 text-sm text-zinc-200 placeholder-zinc-600 outline-none focus:border-[#8a9a7b]/50 focus:ring-1 focus:ring-[#8a9a7b]/20"
                />
              </label>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => { setSelectedPath(null); setDisplayName(""); }}
                  className="flex-1 rounded-lg border border-zinc-700/60 bg-zinc-800/40 px-4 py-2.5 text-sm font-medium text-zinc-300 hover:bg-zinc-800 transition-colors"
                >
                  Change folder
                </button>
                <button
                  type="button"
                  onClick={handleAdd}
                  disabled={importing}
                  className={cn(
                    "flex-1 inline-flex items-center justify-center gap-2 rounded-lg bg-[#6b7a5e] px-4 py-2.5 text-sm font-semibold text-white transition-colors",
                    "hover:bg-[#8a9a7b] disabled:opacity-50 disabled:cursor-not-allowed",
                  )}
                >
                  {importing ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      Adding...
                    </>
                  ) : (
                    "Add workspace"
                  )}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
