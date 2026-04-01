import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useFiles } from "./useFiles";

const apiMocks = vi.hoisted(() => ({
  createWorkspaceFile: vi.fn(),
  fetchWorkspaceFilePreview: vi.fn(),
  fetchWorkspaceFiles: vi.fn(),
  importWorkspaceFiles: vi.fn(),
  openWorkspaceFile: vi.fn(),
  revealWorkspaceFile: vi.fn(),
}));

vi.mock("../api", () => ({
  createWorkspaceFile: apiMocks.createWorkspaceFile,
  fetchWorkspaceFilePreview: apiMocks.fetchWorkspaceFilePreview,
  fetchWorkspaceFiles: apiMocks.fetchWorkspaceFiles,
  importWorkspaceFiles: apiMocks.importWorkspaceFiles,
  openWorkspaceFile: apiMocks.openWorkspaceFile,
  revealWorkspaceFile: apiMocks.revealWorkspaceFile,
}));

vi.mock("../history", () => ({
  matchesWorkspaceSearch: () => true,
}));

const rootFile = {
  path: "notes.md",
  name: "notes.md",
  is_dir: false,
  size_bytes: 128,
  modified_at: "2026-03-31T12:00:00Z",
  extension: ".md",
};

function buildDeps(overrides: Record<string, unknown> = {}) {
  return {
    selectedWorkspaceId: "workspace-1",
    selectedWorkspaceSummary: {
      id: "workspace-1",
      canonical_path: "/tmp/workspace-1",
      display_name: "Workspace 1",
      workspace_type: "directory",
      is_archived: false,
      sort_order: 0,
      last_opened_at: "2026-03-31T12:00:00Z",
      created_at: "2026-03-31T12:00:00Z",
      updated_at: "2026-03-31T12:00:00Z",
      metadata: {},
      exists_on_disk: true,
      conversation_count: 0,
      run_count: 0,
      active_run_count: 0,
      last_activity_at: "2026-03-31T12:00:00Z",
    },
    workspaceArtifacts: [],
    recentWorkspaceArtifacts: [],
    workspaceConversationRows: [],
    selectedConversationRunIds: [],
    selectedRunId: "",
    runArtifacts: [],
    selectedConversationSummary: null,
    workspaceFileTreeMode: "all" as const,
    workspaceImportFolderDraft: "",
    setError: vi.fn(),
    setNotice: vi.fn(),
    ...overrides,
  };
}

describe("useFiles", () => {
  beforeEach(() => {
    apiMocks.createWorkspaceFile.mockReset();
    apiMocks.fetchWorkspaceFilePreview.mockReset();
    apiMocks.fetchWorkspaceFiles.mockReset();
    apiMocks.importWorkspaceFiles.mockReset();
    apiMocks.openWorkspaceFile.mockReset();
    apiMocks.revealWorkspaceFile.mockReset();
  });

  it("drops missing directories during refresh instead of surfacing a 404", async () => {
    const setError = vi.fn();
    let reportsDirectoryExists = true;

    apiMocks.fetchWorkspaceFiles.mockImplementation(async (_workspaceId: string, directory = "") => {
      if (directory === "") {
        return [
          {
            path: "reports",
            name: "reports",
            is_dir: true,
            size_bytes: 0,
            modified_at: "2026-03-31T12:00:00Z",
            extension: "",
          },
          rootFile,
        ];
      }
      if (directory === "reports") {
        if (!reportsDirectoryExists) {
          throw new Error("404 Not Found");
        }
        return [{
          path: "reports/summary.md",
          name: "summary.md",
          is_dir: false,
          size_bytes: 256,
          modified_at: "2026-03-31T12:01:00Z",
          extension: ".md",
        }];
      }
      return [];
    });

    const { result } = renderHook(() => useFiles(buildDeps({ setError })));

    await waitFor(() => {
      expect(result.current.workspaceFilesByDirectory[""]).toHaveLength(2);
    });

    await act(async () => {
      await result.current.loadWorkspaceDirectory("workspace-1", "reports");
    });

    act(() => {
      result.current.toggleWorkspaceDirectory("reports");
    });

    expect(result.current.workspaceFilesByDirectory.reports).toHaveLength(1);
    expect(result.current.expandedWorkspaceDirectories).toContain("reports");

    reportsDirectoryExists = false;

    await act(async () => {
      await result.current.handleRefreshWorkspaceFiles();
    });

    expect(setError).not.toHaveBeenCalledWith("404 Not Found");
    expect(result.current.workspaceFilesByDirectory.reports).toBeUndefined();
    expect(result.current.expandedWorkspaceDirectories).not.toContain("reports");
    expect(result.current.workspaceFilesByDirectory[""]).toHaveLength(2);
  });

  it("clears a deleted selected file during refresh instead of surfacing a 404", async () => {
    const setError = vi.fn();
    let fileExists = true;

    apiMocks.fetchWorkspaceFiles.mockImplementation(async (_workspaceId: string, directory = "") => {
      if (directory !== "") {
        return [];
      }
      return fileExists ? [rootFile] : [];
    });
    apiMocks.fetchWorkspaceFilePreview.mockImplementation(async (_workspaceId: string, path: string) => {
      if (fileExists && path === "notes.md") {
        return {
          path: "notes.md",
          name: "notes.md",
          extension: ".md",
          size_bytes: 128,
          modified_at: "2026-03-31T12:00:00Z",
          preview_kind: "text",
          language: "markdown",
          text_content: "# Notes\n",
          table: null,
          metadata: {},
          truncated: false,
          error: "",
        };
      }
      throw new Error("404 Not Found");
    });

    const { result } = renderHook(() => useFiles(buildDeps({ setError })));

    await waitFor(() => {
      expect(result.current.workspaceFilesByDirectory[""]).toHaveLength(1);
    });

    await act(async () => {
      result.current.setSelectedWorkspaceFilePath("notes.md");
    });

    await waitFor(() => {
      expect(result.current.workspaceFilePreview?.path).toBe("notes.md");
    });

    fileExists = false;

    await act(async () => {
      await result.current.handleRefreshWorkspaceFiles();
    });

    expect(setError).not.toHaveBeenCalledWith("404 Not Found");
    expect(result.current.selectedWorkspaceFilePath).toBe("");
    expect(result.current.workspaceFilePreview).toBeNull();
    expect(result.current.workspaceFilesByDirectory[""]).toHaveLength(0);
  });
});
