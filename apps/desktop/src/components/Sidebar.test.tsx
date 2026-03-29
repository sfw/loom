import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import Sidebar from "./Sidebar";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  useApp: () => mockApp,
}));

describe("Sidebar", () => {
  beforeEach(() => {
    mockApp = {
      activeTab: "threads",
      setActiveTab: vi.fn(),
      workspaces: [
        {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace",
          workspace_type: "local",
          is_archived: false,
          sort_order: 0,
          last_opened_at: "2026-03-27T00:00:00Z",
          created_at: "2026-03-27T00:00:00Z",
          updated_at: "2026-03-27T00:00:00Z",
          metadata: {},
          exists_on_disk: true,
          conversation_count: 1,
          run_count: 0,
          active_run_count: 0,
          last_activity_at: "2026-03-27T00:00:00Z",
        },
      ],
      selectedWorkspaceId: "workspace-1",
      selectedConversationId: "conversation-1",
      setSelectedWorkspaceId: vi.fn(),
      setSelectedConversationId: vi.fn(),
      setSelectedRunId: vi.fn(),
      showArchivedWorkspaces: false,
      setShowArchivedWorkspaces: vi.fn(),
      setShowNewWorkspace: vi.fn(),
      runtime: { ready: true, version: "0.2.2" },
      approvalInbox: [],
      overview: {
        workspace: {
          id: "workspace-1",
          canonical_path: "/tmp/workspace",
          display_name: "Workspace",
          workspace_type: "local",
          is_archived: false,
          sort_order: 0,
          last_opened_at: "2026-03-27T00:00:00Z",
          created_at: "2026-03-27T00:00:00Z",
          updated_at: "2026-03-27T00:00:00Z",
          metadata: {},
          exists_on_disk: true,
          conversation_count: 1,
          run_count: 0,
          active_run_count: 0,
          last_activity_at: "2026-03-27T00:00:00Z",
        },
        recent_conversations: [
          {
            id: "conversation-1",
            workspace_id: "workspace-1",
            workspace_path: "/tmp/workspace",
            model_name: "kimi-k2.5",
            title: "Conversation 5bfafa42",
            turn_count: 1,
            total_tokens: 0,
            last_active_at: "2026-03-27T00:00:00Z",
            started_at: "2026-03-27T00:00:00Z",
            is_active: false,
            linked_run_ids: [],
          },
        ],
        recent_runs: [],
        pending_approvals_count: 0,
        counts: {},
      },
      conversationDetail: {
        id: "conversation-1",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        model_name: "kimi-k2.5",
        title: "Hi!",
        turn_count: 2,
        total_tokens: 42,
        last_active_at: "2026-03-27T00:01:00Z",
        started_at: "2026-03-27T00:00:00Z",
        is_active: true,
        linked_run_ids: [],
      },
      conversationIsProcessing: true,
      models: [],
      refreshWorkspaceSurface: vi.fn(async () => {}),
      refreshConversation: vi.fn(async () => {}),
      handleArchiveWorkspace: vi.fn(async () => {}),
      setError: vi.fn(),
      setNotice: vi.fn(),
    };
  });

  it("shows the selected conversation's live title instead of stale overview data", async () => {
    const user = userEvent.setup();
    render(<Sidebar />);

    const threadButtons = screen.getAllByRole("button", { name: /Threads/i });
    await user.click(threadButtons[threadButtons.length - 1]!);

    expect(screen.getByText("Hi!")).toBeInTheDocument();
    expect(screen.queryByText("Conversation 5bfafa42")).not.toBeInTheDocument();
  });

  it("shows run status badges in the workspace run list", async () => {
    const user = userEvent.setup();
    mockApp.overview.recent_runs = [
      {
        id: "run-1",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        goal: "Active SEO run",
        status: "executing",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:01:00Z",
        execution_run_id: "run-abc",
        process_name: "ad-hoc",
        linked_conversation_ids: [],
        changed_files_count: 0,
      },
      {
        id: "run-2",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        goal: "Paused research run",
        status: "paused",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:01:00Z",
        execution_run_id: "run-def",
        process_name: "ad-hoc",
        linked_conversation_ids: [],
        changed_files_count: 0,
      },
    ];

    render(<Sidebar />);

    const runButtons = screen.getAllByRole("button", { name: /Runs/i });
    await user.click(runButtons[runButtons.length - 1]!);

    expect(screen.getByText("Active SEO run")).toBeInTheDocument();
    expect(screen.getByText("Executing")).toBeInTheDocument();
    expect(screen.getByText("Paused")).toBeInTheDocument();
  });

  it("uses a blue workspace activity dot when runs are active", () => {
    mockApp.workspaces[0].active_run_count = 1;
    mockApp.overview.workspace.active_run_count = 1;

    const { container } = render(<Sidebar />);

    const workspaceButton = screen.getAllByRole("button", { name: /Workspace/i }).find((button) =>
      button.getAttribute("aria-expanded") !== null,
    );
    expect(workspaceButton).toBeDefined();
    const workspaceHeader = workspaceButton!.closest("div");
    expect(workspaceHeader).not.toBeNull();

    const activeDot = (workspaceHeader as HTMLElement).querySelector("svg.lucide-circle");
    expect(activeDot).not.toBeNull();
    expect(activeDot).toHaveClass("fill-sky-400", "text-sky-400");
    expect(container.querySelector(".fill-emerald-400.text-emerald-400")).toBeNull();
  });

  it("updates workspace chevrons when the selected workspace changes", () => {
    mockApp.workspaces = [
      {
        ...mockApp.workspaces[0],
        id: "workspace-1",
        display_name: "Workspace One",
      },
      {
        ...mockApp.workspaces[0],
        id: "workspace-2",
        display_name: "Workspace Two",
      },
    ];

    const { rerender } = render(<Sidebar />);

    expect(
      screen.getByRole("button", { name: /Workspace One/i }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByRole("button", { name: /Workspace Two/i }),
    ).toHaveAttribute("aria-expanded", "false");

    mockApp.selectedWorkspaceId = "workspace-2";
    mockApp.overview = {
      ...mockApp.overview,
      workspace: {
        ...mockApp.overview.workspace,
        id: "workspace-2",
        display_name: "Workspace Two",
      },
      recent_conversations: [],
      recent_runs: [],
    };

    rerender(<Sidebar />);

    expect(
      screen.getByRole("button", { name: /Workspace One/i }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.getByRole("button", { name: /Workspace Two/i }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("renders archived workspaces in a separate section", () => {
    mockApp.showArchivedWorkspaces = true;
    mockApp.workspaces = [
      {
        ...mockApp.workspaces[0],
        id: "workspace-1",
        display_name: "Active Workspace",
        is_archived: false,
      },
      {
        ...mockApp.workspaces[0],
        id: "workspace-2",
        display_name: "Archived Workspace",
        is_archived: true,
      },
    ];

    render(<Sidebar />);

    expect(screen.getByText("Archived Workspaces")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Active Workspace/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Archived Workspace/i })).toBeInTheDocument();
    expect(screen.getAllByText("archived")[0]).toBeInTheDocument();
  });
});
