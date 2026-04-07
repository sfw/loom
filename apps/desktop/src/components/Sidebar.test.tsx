import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import Sidebar from "./Sidebar";
import { createConversation, patchConversation } from "@/api";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  shallowEqual: (left: unknown, right: unknown) => left === right,
  useApp: () => mockApp,
  useAppActions: () => mockApp,
  useAppSelector: (selector: (state: any) => unknown) => selector(mockApp),
}));

vi.mock("@/api", () => ({
  createConversation: vi.fn(),
  patchConversation: vi.fn(),
}));

describe("Sidebar", () => {
  beforeEach(() => {
    vi.mocked(createConversation).mockResolvedValue({
      id: "conversation-created",
      workspace_id: "workspace-1",
      workspace_path: "/tmp/workspace",
      model_name: "kimi-k2.5",
      title: "Conversation created",
      turn_count: 0,
      total_tokens: 0,
      last_active_at: "2026-03-27T00:02:00Z",
      started_at: "2026-03-27T00:02:00Z",
      is_active: false,
      linked_run_ids: [],
    });
    vi.mocked(patchConversation).mockResolvedValue({
      id: "conversation-1",
      workspace_id: "workspace-1",
      workspace_path: "/tmp/workspace",
      model_name: "kimi-k2.5",
      title: "Renamed",
      turn_count: 2,
      total_tokens: 42,
      last_active_at: "2026-03-27T00:01:00Z",
      started_at: "2026-03-27T00:00:00Z",
      is_active: true,
      linked_run_ids: [],
      system_prompt: "",
      session_state: {},
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
    });
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
      runtime: { ready: true, version: "0.3.0" },
      connectionState: "connected",
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
      desktopActivity: {
        active: false,
        mode: "idle",
        activeConversationCount: 0,
        activeRunCount: 0,
        sourceCount: 0,
        label: "Idle",
        updatedAt: "",
        backendConnected: false,
      },
      models: [],
      syncConversationSummary: vi.fn(),
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

  it("renders the brand activity bar from shared desktop activity state", () => {
    mockApp.desktopActivity = {
      active: true,
      mode: "mixed",
      activeConversationCount: 1,
      activeRunCount: 1,
      sourceCount: 2,
      label: "1 active thread · 1 active run",
      updatedAt: "2026-03-29T12:00:00Z",
      backendConnected: true,
    };

    render(<Sidebar />);

    expect(screen.getByAltText("Loom logo")).toBeInTheDocument();

    const activityBar = screen.getByTestId("desktop-activity-bar");
    expect(activityBar).toHaveAttribute("data-active", "true");
    expect(activityBar).toHaveAttribute("data-mode", "mixed");
    expect(activityBar).toHaveAttribute("title", "1 active thread · 1 active run");
  });

  it("shows the runtime footer from live connection state", () => {
    mockApp.connectionState = "failed";

    render(<Sidebar />);

    expect(screen.getByText("Disconnected · 0.3.0")).toBeInTheDocument();
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

  it("selects a newly created thread immediately without waiting for workspace refresh", async () => {
    const user = userEvent.setup();
    mockApp.selectedConversationId = "";
    mockApp.conversationDetail = null;
    mockApp.overview.recent_conversations = [];
    mockApp.syncConversationSummary = vi.fn();

    render(<Sidebar />);

    const plusButtons = screen.getAllByTitle(/Click: new thread/i);
    await user.click(plusButtons[plusButtons.length - 1]!);

    expect(createConversation).toHaveBeenCalledWith("workspace-1", {});
    expect(mockApp.setSelectedConversationId).toHaveBeenCalledWith("conversation-created");
    expect(mockApp.setActiveTab).toHaveBeenCalledWith("threads");
    expect(mockApp.syncConversationSummary).toHaveBeenCalledWith(
      expect.objectContaining({ id: "conversation-created" }),
      expect.objectContaining({ incrementCount: true, workspaceId: "workspace-1" }),
    );
    expect(mockApp.refreshWorkspaceSurface).not.toHaveBeenCalled();
  });
});
