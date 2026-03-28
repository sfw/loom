import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import OverviewTab from "./OverviewTab";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  useApp: () => mockApp,
}));

describe("OverviewTab", () => {
  beforeEach(() => {
    mockApp = {
      overview: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace",
          active_run_count: 0,
        },
        recent_conversations: [],
        recent_runs: [],
        pending_approvals_count: 0,
      },
      loadingOverview: false,
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace",
        display_name: "Workspace",
      },
      noWorkspacesRegistered: false,
      selectedWorkspaceIsEmpty: false,
      selectedWorkspaceTags: [],
      selectedWorkspaceNote: "",
      recentNotifications: [],
      recentWorkspaceArtifacts: Array.from({ length: 10 }, (_, index) => ({
        path: `reports/file-${index + 1}.md`,
        sha256: `sha-${index + 1}`,
        category: "document",
        run_count: 1,
      })),
      approvalInbox: [],
      setActiveTab: vi.fn(),
      handleOpenWorkspaceFile: vi.fn(async () => {}),
      focusConversationComposer: vi.fn(),
      focusRunComposer: vi.fn(),
      handlePrefillStarterWorkspace: vi.fn(),
      setSelectedConversationId: vi.fn(),
      setSelectedRunId: vi.fn(),
      setShowNewWorkspace: vi.fn(),
    };
  });

  it("opens a recent file directly in the Files tab", async () => {
    const user = userEvent.setup();
    render(<OverviewTab />);

    await user.click(screen.getByRole("button", { name: /reports\/file-1\.md/i }));

    expect(mockApp.handleOpenWorkspaceFile).toHaveBeenCalledWith("reports/file-1.md");
    expect(mockApp.setActiveTab).toHaveBeenCalledWith("files");
  });

  it("shows more recent files on the overview grid", () => {
    render(<OverviewTab />);

    expect(screen.getByRole("button", { name: /reports\/file-9\.md/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /reports\/file-10\.md/i })).not.toBeInTheDocument();
  });

  it("opens the run launcher instead of leaving a stale selected run open", async () => {
    const user = userEvent.setup();
    render(<OverviewTab />);

    await user.click(screen.getByRole("button", { name: /launch run/i }));

    expect(mockApp.setSelectedRunId).toHaveBeenCalledWith("");
    expect(mockApp.setActiveTab).toHaveBeenCalledWith("runs");
    expect(mockApp.focusRunComposer).toHaveBeenCalled();
  });
});
