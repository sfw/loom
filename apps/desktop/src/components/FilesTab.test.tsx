import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import FilesTab from "./FilesTab";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  shallowEqual: (left: unknown, right: unknown) => left === right,
  useApp: () => mockApp,
  useAppActions: () => mockApp,
  useAppSelector: (selector: (state: any) => unknown) => selector(mockApp),
}));

describe("FilesTab", () => {
  beforeEach(() => {
    mockApp = {
      visibleRootWorkspaceFiles: [
        {
          path: "notes.md",
          name: "notes.md",
          is_dir: false,
          size_bytes: 180_000,
          modified_at: "2026-03-30T20:00:00Z",
          extension: ".md",
        },
      ],
      workspaceFilesByDirectory: {
        "": [
          {
            path: "notes.md",
            name: "notes.md",
            is_dir: false,
            size_bytes: 180_000,
            modified_at: "2026-03-30T20:00:00Z",
            extension: ".md",
          },
        ],
      },
      expandedWorkspaceDirectories: [""],
      loadingWorkspaceDirectory: "",
      refreshingWorkspaceFiles: false,
      selectedWorkspaceFilePath: "notes.md",
      selectedWorkspaceFileEntry: {
        path: "notes.md",
        name: "notes.md",
        is_dir: false,
        size_bytes: 180_000,
        modified_at: "2026-03-30T20:00:00Z",
        extension: ".md",
      },
      workspaceFilePreview: {
        path: "notes.md",
        name: "notes.md",
        extension: ".md",
        size_bytes: 180_000,
        modified_at: "2026-03-30T20:00:00Z",
        preview_kind: "text",
        language: "markdown",
        text_content: `# Large doc\n\n${"content ".repeat(15_000)}`,
        table: null,
        metadata: {},
        truncated: false,
        error: "",
      },
      loadingWorkspaceFilePreview: false,
      selectedWorkspaceFileIsEditable: false,
      selectedWorkspaceFileEditorHasChanges: false,
      selectedWorkspaceFileEditHint: "",
      workspaceFileEditorDraft: "",
      setWorkspaceFileEditorDraft: vi.fn(),
      workspaceFileEditorDirty: false,
      setWorkspaceFileEditorDirty: vi.fn(),
      savingWorkspaceFile: false,
      workspaceFileFilterQuery: "",
      setWorkspaceFileFilterQuery: vi.fn(),
      workspaceFileTreeMode: "all",
      setWorkspaceFileTreeMode: vi.fn(),
      normalizedWorkspaceFileFilterQuery: "",
      locallyVisibleWorkspaceFilePaths: new Set<string>(),
      contextualFilePaths: new Set<string>(),
      contextualDirectoryCounts: new Map<string, number>(),
      recentFilePaths: new Set<string>(),
      recentDirectoryCounts: new Map<string, number>(),
      importingWorkspaceFiles: false,
      workspaceImportFolderDraft: "",
      setWorkspaceImportFolderDraft: vi.fn(),
      workspaceFileInputRef: { current: null },
      handleWorkspaceFileSelection: vi.fn(),
      handleOpenWorkspaceFileExternally: vi.fn(async () => {}),
      handleRevealWorkspaceFile: vi.fn(async () => {}),
      handleSaveWorkspaceFile: vi.fn(async () => {}),
      handleResetWorkspaceFileEditor: vi.fn(),
      handleRefreshWorkspaceFiles: vi.fn(async () => {}),
      handleExpandActiveWorkspaceFiles: vi.fn(async () => {}),
      handleExpandRecentWorkspaceFiles: vi.fn(async () => {}),
      handleImportWorkspaceFiles: vi.fn(async () => {}),
      selectedWorkspaceId: "workspace-1",
      selectedWorkspaceSummary: {
        canonical_path: "/tmp/workspace",
      },
      loadWorkspaceDirectory: vi.fn(async () => {}),
      setError: vi.fn(),
      setNotice: vi.fn(),
    };
  });

  it("guards large markdown previews until the user opts in", async () => {
    const user = userEvent.setup();

    render(<FilesTab />);

    expect(screen.getByText("Large Markdown preview paused")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Render anyway" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Render anyway" }));

    expect(screen.getByRole("heading", { name: "Large doc" })).toBeInTheDocument();
  });

  it("renders updated markdown content as soon as the preview payload arrives", () => {
    mockApp.workspaceFilePreview = {
      ...mockApp.workspaceFilePreview,
      size_bytes: 10_500,
      text_content: "",
    };
    mockApp.selectedWorkspaceFileEntry = {
      ...mockApp.selectedWorkspaceFileEntry,
      size_bytes: 10_500,
    };
    mockApp.visibleRootWorkspaceFiles = [
      {
        ...mockApp.visibleRootWorkspaceFiles[0],
        size_bytes: 10_500,
      },
    ];
    mockApp.workspaceFilesByDirectory = {
      "": [
        {
          ...mockApp.workspaceFilesByDirectory[""][0],
          size_bytes: 10_500,
        },
      ],
    };

    const { rerender } = render(<FilesTab />);

    expect(screen.queryByRole("heading", { name: "Build Summary" })).not.toBeInTheDocument();

    mockApp.workspaceFilePreview = {
      ...mockApp.workspaceFilePreview,
      text_content: "# Build Summary\n\nRendered preview body.\n",
    };

    rerender(<FilesTab />);

    expect(screen.getByRole("heading", { name: "Build Summary" })).toBeInTheDocument();
    expect(screen.getByText("Rendered preview body.")).toBeInTheDocument();
  });

  it("reloads the file panel when the refresh button is clicked", async () => {
    const user = userEvent.setup();

    render(<FilesTab />);

    await user.click(screen.getByRole("button", { name: "Reload files" }));

    expect(mockApp.handleRefreshWorkspaceFiles).toHaveBeenCalledTimes(1);
  });
});
