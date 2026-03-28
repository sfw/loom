import { useState, useCallback } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useApp } from "@/context/AppContext";
import { createWorkspaceDirectory } from "@/api";
import { cn } from "@/lib/utils";
import { formatDate, formatBytes, highlightText, fileDepth } from "@/utils";
import type { WorkspaceFileEntry } from "@/api";
import {
  File,
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  ExternalLink,
  Eye,
  Save,
  RotateCcw,
  Upload,
  Filter,
  Code,
  BookOpen,
} from "lucide-react";

export default function FilesTab() {
  const {
    visibleRootWorkspaceFiles,
    workspaceFilesByDirectory,
    expandedWorkspaceDirectories,
    loadingWorkspaceDirectory,
    selectedWorkspaceFilePath,
    selectedWorkspaceFileEntry,
    workspaceFilePreview,
    loadingWorkspaceFilePreview,
    selectedWorkspaceFileIsEditable,
    selectedWorkspaceFileEditorHasChanges,
    selectedWorkspaceFileEditHint,
    workspaceFileEditorDraft,
    setWorkspaceFileEditorDraft,
    workspaceFileEditorDirty,
    setWorkspaceFileEditorDirty,
    savingWorkspaceFile,
    workspaceFileFilterQuery,
    setWorkspaceFileFilterQuery,
    workspaceFileTreeMode,
    setWorkspaceFileTreeMode,
    normalizedWorkspaceFileFilterQuery,
    locallyVisibleWorkspaceFilePaths,
    contextualFilePaths,
    contextualDirectoryCounts,
    recentFilePaths,
    recentDirectoryCounts,
    importingWorkspaceFiles,
    workspaceImportFolderDraft,
    setWorkspaceImportFolderDraft,
    workspaceFileInputRef,
    handleWorkspaceFileSelection,
    handleOpenWorkspaceFileExternally,
    handleRevealWorkspaceFile,
    handleSaveWorkspaceFile,
    handleResetWorkspaceFileEditor,
    handleExpandActiveWorkspaceFiles,
    handleExpandRecentWorkspaceFiles,
    handleImportWorkspaceFiles,
    selectedWorkspaceId,
    selectedWorkspaceSummary,
    loadWorkspaceDirectory,
    setError,
    setNotice,
  } = useApp();

  const [mdRendered, setMdRendered] = useState(true);
  const [newFolderName, setNewFolderName] = useState("");
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [creatingFolder, setCreatingFolder] = useState(false);

  const isMarkdownFile = selectedWorkspaceFilePath.endsWith(".md") || selectedWorkspaceFilePath.endsWith(".mdx") || workspaceFilePreview?.language === "markdown";

  const handleCreateFolder = useCallback(async () => {
    if (!newFolderName.trim() || !selectedWorkspaceSummary?.canonical_path) return;
    setCreatingFolder(true);
    try {
      // Create in the selected directory or workspace root
      const parentDir = selectedWorkspaceFileEntry?.is_dir
        ? selectedWorkspaceFilePath
        : selectedWorkspaceFilePath
          ? selectedWorkspaceFilePath.split("/").slice(0, -1).join("/")
          : "";
      const fullPath = `${selectedWorkspaceSummary.canonical_path}/${parentDir ? parentDir + "/" : ""}${newFolderName.trim()}`;
      await createWorkspaceDirectory(fullPath);
      setNewFolderName("");
      setShowNewFolder(false);
      setNotice(`Created folder ${newFolderName.trim()}`);
      // Refresh the file tree
      if (selectedWorkspaceId) {
        await loadWorkspaceDirectory(selectedWorkspaceId, parentDir);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create folder.");
    } finally {
      setCreatingFolder(false);
    }
  }, [newFolderName, selectedWorkspaceSummary, selectedWorkspaceFileEntry, selectedWorkspaceFilePath, selectedWorkspaceId, loadWorkspaceDirectory, setError, setNotice]);

  /* ------------------------------------------------------------------ */
  /*  Recursive file tree renderer                                      */
  /* ------------------------------------------------------------------ */

  function renderFileRows(directory: string) {
    const entries = directory
      ? workspaceFilesByDirectory[directory]
      : visibleRootWorkspaceFiles;
    if (!entries || entries.length === 0) return null;

    return entries.map((entry: WorkspaceFileEntry) => {
      // When a filter is active, skip entries that aren't locally visible
      if (
        normalizedWorkspaceFileFilterQuery &&
        !locallyVisibleWorkspaceFilePaths.has(entry.path)
      ) {
        return null;
      }

      const depth = fileDepth(entry.path);
      const isSelected = selectedWorkspaceFilePath === entry.path;
      const isExpanded = expandedWorkspaceDirectories.includes(entry.path);
      const isLoading = loadingWorkspaceDirectory === entry.path;

      const isContextual = contextualFilePaths.has(entry.path);
      const contextualCount = entry.is_dir
        ? contextualDirectoryCounts.get(entry.path) ?? 0
        : 0;
      const isRecent = recentFilePaths.has(entry.path);
      const recentCount = entry.is_dir
        ? recentDirectoryCounts.get(entry.path) ?? 0
        : 0;

      return (
        <div key={entry.path}>
          <button
            type="button"
            onClick={() => handleWorkspaceFileSelection(entry)}
            className={cn(
              "group flex w-full items-center gap-1.5 rounded-md py-1.5 pr-2 text-left text-[13px] transition-colors",
              isSelected
                ? "bg-[#6b7a5e]/15 text-[#bec8b4]"
                : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200",
              isContextual && "border-l-2 border-[#8a9a7b]",
              !isContextual && isRecent && "border-l-2 border-yellow-500",
              !isContextual && !isRecent && "border-l-2 border-transparent"
            )}
            style={{ paddingLeft: `${depth * 16 + 8}px` }}
            title={entry.path}
          >
            {/* Expand/collapse or dot indicator */}
            {entry.is_dir ? (
              <span className="flex h-4 w-4 shrink-0 items-center justify-center text-zinc-500">
                {isLoading ? (
                  <span className="h-3 w-3 animate-spin rounded-full border-2 border-zinc-600 border-t-zinc-300" />
                ) : isExpanded ? (
                  <ChevronDown size={14} />
                ) : (
                  <ChevronRight size={14} />
                )}
              </span>
            ) : (
              <span className="flex h-4 w-4 shrink-0 items-center justify-center text-zinc-600">
                <span className="h-1 w-1 rounded-full bg-current" />
              </span>
            )}

            {/* Icon */}
            {entry.is_dir ? (
              isExpanded ? (
                <FolderOpen size={14} className="shrink-0 text-[#a3b396]/70" />
              ) : (
                <Folder size={14} className="shrink-0 text-zinc-500" />
              )
            ) : (
              <File size={14} className="shrink-0 text-zinc-500" />
            )}

            {/* Name */}
            <span className="min-w-0 flex-1 truncate font-mono text-[12px]">
              {normalizedWorkspaceFileFilterQuery
                ? highlightText(entry.name, normalizedWorkspaceFileFilterQuery)
                : entry.name}
            </span>

            {/* Metadata badges */}
            {entry.is_dir ? (
              <>
                {contextualCount > 0 && (
                  <span className="shrink-0 rounded bg-[#6b7a5e]/20 px-1 py-0.5 text-[9px] font-medium tabular-nums text-[#a3b396]">
                    {contextualCount}
                  </span>
                )}
                {recentCount > 0 && (
                  <span className="shrink-0 rounded bg-yellow-600/20 px-1 py-0.5 text-[9px] font-medium tabular-nums text-yellow-400">
                    {recentCount}
                  </span>
                )}
              </>
            ) : (
              <span className="shrink-0 text-[10px] tabular-nums text-zinc-600 opacity-0 transition-opacity group-hover:opacity-100">
                {formatBytes(entry.size_bytes)}
              </span>
            )}
          </button>

          {/* Recurse into expanded directory */}
          {entry.is_dir && isExpanded && renderFileRows(entry.path)}
        </div>
      );
    });
  }

  /* ------------------------------------------------------------------ */
  /*  File preview body                                                 */
  /* ------------------------------------------------------------------ */

  function renderPreviewBody() {
    if (!workspaceFilePreview) return null;
    const preview = workspaceFilePreview;

    if (preview.error) {
      return (
        <p className="px-4 py-6 text-sm text-red-400">{preview.error}</p>
      );
    }

    // Text preview
    if (preview.preview_kind === "text") {
      // Markdown rendered view
      if (isMarkdownFile && mdRendered) {
        return (
          <div className="p-6 prose prose-invert prose-sm max-w-none prose-p:my-2 prose-p:leading-relaxed prose-headings:text-zinc-200 prose-headings:font-semibold prose-h1:text-xl prose-h2:text-lg prose-h3:text-base prose-code:text-[#bec8b4] prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-xs prose-code:before:content-none prose-code:after:content-none prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-800 prose-pre:rounded-lg prose-pre:text-xs prose-a:text-[#a3b396] prose-strong:text-zinc-200 prose-em:text-zinc-300 prose-blockquote:border-zinc-700 prose-blockquote:text-zinc-400 prose-hr:border-zinc-800 prose-th:text-zinc-300 prose-td:text-zinc-400 prose-table:text-xs">
            <Markdown remarkPlugins={[remarkGfm]}>{preview.text_content || ""}</Markdown>
          </div>
        );
      }
      if (selectedWorkspaceFileIsEditable) {
        return (
          <textarea
            className="h-full min-h-[320px] w-full resize-none bg-transparent p-4 font-mono text-[12px] leading-relaxed text-zinc-300 outline-none placeholder:text-zinc-600"
            value={workspaceFileEditorDirty ? workspaceFileEditorDraft : preview.text_content}
            onChange={(e) => {
              setWorkspaceFileEditorDraft(e.target.value);
              setWorkspaceFileEditorDirty(true);
            }}
            spellCheck={false}
          />
        );
      }
      return (
        <pre className="whitespace-pre-wrap break-words p-4 font-mono text-[12px] leading-relaxed text-zinc-300">
          {preview.text_content}
        </pre>
      );
    }

    // Table preview
    if (preview.preview_kind === "table" && preview.table) {
      return (
        <div className="overflow-auto p-4">
          <table className="w-full border-collapse text-[12px]">
            <thead>
              <tr>
                {preview.table.columns.map((col) => (
                  <th
                    key={col}
                    className="border border-zinc-800 bg-zinc-900 px-2 py-1 text-left font-semibold text-zinc-400"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.table.rows.map((row, rowIdx) => (
                <tr key={rowIdx} className="hover:bg-zinc-800/40">
                  {row.map((cell, cellIdx) => (
                    <td
                      key={cellIdx}
                      className="border border-zinc-800 px-2 py-1 text-zinc-300"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {preview.table.truncated && (
            <p className="mt-2 text-[11px] text-zinc-500">
              Table truncated — showing partial data.
            </p>
          )}
        </div>
      );
    }

    // Image preview
    if (preview.preview_kind === "image") {
      return (
        <div className="flex flex-col gap-2 p-4">
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Eye size={14} />
            <span>Image file</span>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 text-[12px] text-zinc-500">
            <p>
              <span className="text-zinc-400">{preview.name}</span>{" "}
              &mdash; {formatBytes(preview.size_bytes)}
            </p>
            {preview.metadata &&
              Object.entries(preview.metadata).map(([key, value]) => (
                <p key={key} className="mt-1">
                  <span className="text-zinc-500">{key}:</span>{" "}
                  <span className="text-zinc-400">{String(value)}</span>
                </p>
              ))}
          </div>
        </div>
      );
    }

    // Unsupported
    return (
      <div className="flex flex-col items-center justify-center gap-2 p-8 text-zinc-500">
        <File size={24} />
        <p className="text-sm">Preview not available for this file type.</p>
        {preview.truncated && (
          <p className="text-[11px] text-zinc-600">File content was truncated.</p>
        )}
      </div>
    );
  }

  /* ------------------------------------------------------------------ */
  /*  Render                                                            */
  /* ------------------------------------------------------------------ */

  const hasContextualFiles = contextualFilePaths.size > 0;
  const hasRecentFiles = recentFilePaths.size > 0;

  return (
    <div className="flex h-full">
      {/* ============================================================ */}
      {/*  Left panel — file tree                                      */}
      {/* ============================================================ */}
      <div className="flex w-80 shrink-0 flex-col border-r border-zinc-800/60 bg-[#0f0f12]">
        {/* Filter input */}
        <div className="flex items-center gap-2 border-b border-zinc-800/60 px-3 py-2">
          <Filter size={14} className="shrink-0 text-zinc-500" />
          <input
            type="text"
            placeholder="Filter files..."
            value={workspaceFileFilterQuery}
            onChange={(e) => setWorkspaceFileFilterQuery(e.target.value)}
            className="min-w-0 flex-1 bg-transparent text-[13px] text-zinc-200 outline-none placeholder:text-zinc-600"
          />
          {workspaceFileFilterQuery && (
            <button
              type="button"
              onClick={() => setWorkspaceFileFilterQuery("")}
              className="text-[11px] text-zinc-500 hover:text-zinc-300"
            >
              Clear
            </button>
          )}
        </div>

        {/* Mode toggle row */}
        <div className="flex items-center gap-1 border-b border-zinc-800/60 px-3 py-1.5">
          {(["all", "active", "recent"] as const).map((mode) => {
            // Hide active/recent toggle buttons when no data for them
            if (mode === "active" && !hasContextualFiles) return null;
            if (mode === "recent" && !hasRecentFiles) return null;

            const isActive = workspaceFileTreeMode === mode;
            return (
              <button
                key={mode}
                type="button"
                onClick={() => setWorkspaceFileTreeMode(mode)}
                className={cn(
                  "rounded px-2 py-0.5 text-[11px] font-medium capitalize transition-colors",
                  isActive
                    ? "bg-[#6b7a5e]/20 text-[#a3b396]"
                    : "text-zinc-500 hover:text-zinc-300"
                )}
              >
                {mode}
              </button>
            );
          })}

          {/* Expand helpers */}
          <div className="ml-auto flex items-center gap-1">
            {hasContextualFiles && workspaceFileTreeMode === "active" && (
              <button
                type="button"
                onClick={handleExpandActiveWorkspaceFiles}
                className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
                title="Expand all active file directories"
              >
                Expand
              </button>
            )}
            {hasRecentFiles && workspaceFileTreeMode === "recent" && (
              <button
                type="button"
                onClick={handleExpandRecentWorkspaceFiles}
                className="rounded px-1.5 py-0.5 text-[10px] text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
                title="Expand all recent file directories"
              >
                Expand
              </button>
            )}
          </div>
        </div>

        {/* Scrollable file tree */}
        <div className="flex-1 overflow-y-auto px-1 py-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-800">
          {!selectedWorkspaceId ? (
            <p className="px-3 py-6 text-center text-xs text-zinc-600">
              Select a workspace to browse files.
            </p>
          ) : visibleRootWorkspaceFiles.length === 0 ? (
            <p className="px-3 py-6 text-center text-xs text-zinc-600">
              No files found.
            </p>
          ) : (
            <div className="flex flex-col gap-px">{renderFileRows("")}</div>
          )}
        </div>

        {/* File actions toolbar */}
        <div className="border-t border-zinc-800/60 px-2 py-2 flex items-center gap-1">
          {/* Import files — auto-imports to selected folder */}
          <input
            ref={workspaceFileInputRef}
            type="file"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) {
                // Auto-set destination to the currently selected directory
                const selectedDir = selectedWorkspaceFileEntry?.is_dir
                  ? selectedWorkspaceFilePath
                  : selectedWorkspaceFilePath
                    ? selectedWorkspaceFilePath.split("/").slice(0, -1).join("/")
                    : "";
                if (selectedDir) {
                  setWorkspaceImportFolderDraft(selectedDir);
                }
                handleImportWorkspaceFiles(e.target.files);
              }
            }}
            disabled={importingWorkspaceFiles || !selectedWorkspaceId}
          />
          <button
            type="button"
            onClick={() => workspaceFileInputRef.current?.click()}
            disabled={importingWorkspaceFiles || !selectedWorkspaceId}
            className="flex items-center gap-1.5 rounded px-2 py-1 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors disabled:opacity-40"
            title={selectedWorkspaceFilePath ? `Import to ${selectedWorkspaceFilePath.split("/").slice(0, -1).join("/") || "root"}` : "Import files"}
          >
            <Upload size={12} />
            {importingWorkspaceFiles ? "Importing..." : "Import"}
          </button>

          {/* New folder */}
          <button
            type="button"
            onClick={() => setShowNewFolder(!showNewFolder)}
            disabled={!selectedWorkspaceId}
            className="flex items-center gap-1.5 rounded px-2 py-1 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors disabled:opacity-40"
            title="New folder"
          >
            <FolderOpen size={12} />
            New folder
          </button>
        </div>

        {/* New folder inline form */}
        {showNewFolder && (
          <div className="border-t border-zinc-800/40 px-3 py-2 flex items-center gap-1.5">
            <input
              type="text"
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              placeholder="Folder name..."
              autoFocus
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void handleCreateFolder();
                } else if (e.key === "Escape") {
                  setShowNewFolder(false);
                  setNewFolderName("");
                }
              }}
              className="min-w-0 flex-1 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-[11px] text-zinc-300 outline-none placeholder:text-zinc-600 focus:border-[#8a9a7b]/50"
            />
            <button
              type="button"
              onClick={() => void handleCreateFolder()}
              disabled={!newFolderName.trim() || creatingFolder}
              className="rounded bg-[#6b7a5e] px-2 py-1 text-[11px] font-medium text-white hover:bg-[#8a9a7b] disabled:opacity-40 transition-colors"
            >
              {creatingFolder ? "..." : "Create"}
            </button>
            <button
              type="button"
              onClick={() => { setShowNewFolder(false); setNewFolderName(""); }}
              className="rounded px-1.5 py-1 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
            >
              ✕
            </button>
          </div>
        )}
      </div>

      {/* ============================================================ */}
      {/*  Right panel — file preview / editor                         */}
      {/* ============================================================ */}
      <div className="flex min-w-0 flex-1 flex-col bg-[#111114]">
        {!selectedWorkspaceFileEntry ? (
          /* Empty state */
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-zinc-600">
            <File size={32} className="text-zinc-700" />
            <p className="text-sm">Select a file to preview</p>
          </div>
        ) : (
          <>
            {/* File actions bar */}
            <div className="flex items-center gap-2 border-b border-zinc-800/60 px-4 py-2">
              <span className="min-w-0 flex-1 truncate font-mono text-[13px] text-zinc-300">
                {selectedWorkspaceFileEntry.path}
              </span>

              {isMarkdownFile && (
                <button
                  type="button"
                  onClick={() => setMdRendered(!mdRendered)}
                  className={cn(
                    "flex items-center gap-1 rounded px-2 py-1 text-[11px] transition-colors",
                    mdRendered
                      ? "bg-[#8a9a7b]/15 text-[#a3b396]"
                      : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200",
                  )}
                  title={mdRendered ? "Show raw source" : "Render markdown"}
                >
                  {mdRendered ? <Code size={12} /> : <BookOpen size={12} />}
                  <span>{mdRendered ? "Source" : "Preview"}</span>
                </button>
              )}

              <button
                type="button"
                onClick={handleOpenWorkspaceFileExternally}
                className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                title="Open in default application"
              >
                <ExternalLink size={12} />
                <span>Open externally</span>
              </button>

              <button
                type="button"
                onClick={handleRevealWorkspaceFile}
                className="flex items-center gap-1 rounded px-2 py-1 text-[11px] text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
                title="Reveal in file manager"
              >
                <FolderOpen size={12} />
                <span>Reveal in folder</span>
              </button>

              {selectedWorkspaceFileIsEditable && (
                <>
                  <button
                    type="button"
                    onClick={handleResetWorkspaceFileEditor}
                    disabled={!selectedWorkspaceFileEditorHasChanges}
                    className={cn(
                      "flex items-center gap-1 rounded px-2 py-1 text-[11px] transition-colors",
                      selectedWorkspaceFileEditorHasChanges
                        ? "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
                        : "cursor-not-allowed text-zinc-600"
                    )}
                    title="Discard edits"
                  >
                    <RotateCcw size={12} />
                    <span>Reset edits</span>
                  </button>

                  <button
                    type="button"
                    onClick={handleSaveWorkspaceFile}
                    disabled={
                      !selectedWorkspaceFileEditorHasChanges || savingWorkspaceFile
                    }
                    className={cn(
                      "flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors",
                      selectedWorkspaceFileEditorHasChanges && !savingWorkspaceFile
                        ? "bg-[#6b7a5e] text-white hover:bg-[#8a9a7b]"
                        : "cursor-not-allowed bg-zinc-800 text-zinc-500"
                    )}
                    title="Save changes"
                  >
                    <Save size={12} />
                    <span>{savingWorkspaceFile ? "Saving..." : "Save"}</span>
                  </button>
                </>
              )}
            </div>

            {/* Edit hint */}
            {selectedWorkspaceFileEditHint && (
              <div className="border-b border-zinc-800/60 px-4 py-1.5 text-[11px] text-zinc-500">
                {selectedWorkspaceFileEditHint}
              </div>
            )}

            {/* File metadata strip */}
            <div className="flex items-center gap-4 border-b border-zinc-800/60 px-4 py-1.5 text-[10px] text-zinc-500">
              <span>{formatBytes(selectedWorkspaceFileEntry.size_bytes)}</span>
              <span>Modified {formatDate(selectedWorkspaceFileEntry.modified_at)}</span>
              {selectedWorkspaceFileEntry.extension && (
                <span className="rounded bg-zinc-800/80 px-1.5 py-0.5 font-mono text-zinc-400">
                  .{selectedWorkspaceFileEntry.extension}
                </span>
              )}
              {workspaceFilePreview?.language && (
                <span className="rounded bg-zinc-800/80 px-1.5 py-0.5 text-zinc-400">
                  {workspaceFilePreview.language}
                </span>
              )}
              {workspaceFilePreview?.truncated && (
                <span className="text-yellow-500">Truncated</span>
              )}
            </div>

            {/* Preview body */}
            <div className="flex-1 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-800">
              {loadingWorkspaceFilePreview ? (
                <div className="flex items-center justify-center py-12">
                  <span className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-700 border-t-zinc-400" />
                </div>
              ) : (
                renderPreviewBody()
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
