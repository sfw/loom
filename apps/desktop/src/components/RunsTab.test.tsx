import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import RunsTab from "./RunsTab";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  shallowEqual: (left: unknown, right: unknown) => left === right,
  useApp: () => mockApp,
  useAppActions: () => mockApp,
  useAppSelector: (selector: (state: any) => unknown) => selector(mockApp),
}));

describe("RunsTab", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  beforeEach(() => {
    mockApp = {
      filteredRuns: [],
      selectedRunId: "",
      setSelectedRunId: vi.fn(),
      selectedRunSummary: null,
      runDetail: null,
      runTimeline: [],
      runArtifacts: [],
      runInstructionHistory: [],
      runStreaming: false,
      loadingRunDetail: false,
      runLoadError: "",
      runIsTerminal: false,
      runCanPause: false,
      runCanResume: false,
      runCanMessage: false,
      visibleRunTimeline: [],
      visibleRunArtifacts: [],
      runHistoryQuery: "",
      setRunHistoryQuery: vi.fn(),
      activeRunMatchIndex: 0,
      totalRunMatches: 0,
      runGoal: "",
      setRunGoal: vi.fn(),
      runProcess: "",
      setRunProcess: vi.fn(),
      runApprovalMode: "auto",
      setRunApprovalMode: vi.fn(),
      launchingRun: false,
      runOperatorMessage: "",
      setRunOperatorMessage: vi.fn(),
      sendingRunMessage: false,
      runActionPending: "",
      runMatchRefs: { current: [] },
      handleLaunchRun: vi.fn(async () => {}),
      handleRunControl: vi.fn(async () => {}),
      handleDeleteRun: vi.fn(async () => {}),
      handleRestartRun: vi.fn(async () => {}),
      handleRefreshWorkspaceFiles: vi.fn(async () => {}),
      handleSendRunMessage: vi.fn(async () => {}),
      handleOpenWorkspaceFile: vi.fn(async () => {}),
      refreshRun: vi.fn(async () => {}),
      refreshWorkspaceArtifacts: vi.fn(async () => {}),
      stepRunMatch: vi.fn(),
      selectedWorkspaceId: "workspace-1",
      workspaceSearchQuery: "",
      overview: {
        workspace: { canonical_path: "/tmp/workspace" },
      },
      inventory: { processes: [] },
      approvalInbox: [],
      approvalReplyDrafts: {},
      setApprovalReplyDrafts: vi.fn(),
      replyingApprovalId: "",
      handleReplyApproval: vi.fn(async () => {}),
      loadedWorkspaceFileEntries: [],
      workspaceArtifacts: [],
      recentWorkspaceArtifacts: [],
      setActiveTab: vi.fn(),
    };
  });

  it("shows an opening state when a run is selected before detail arrives", () => {
    mockApp.selectedRunId = "run-abc";
    mockApp.selectedRunSummary = {
      id: "run-abc",
      goal: "Review the site",
      process_name: "seo-geo-review",
    };
    mockApp.loadingRunDetail = true;

    render(<RunsTab />);

    expect(screen.getByText("Opening run...")).toBeInTheDocument();
    expect(screen.getByText("Review the site")).toBeInTheDocument();
    expect(screen.queryByText("Launch a Run")).not.toBeInTheDocument();
  });

  it("wraps long run goals in the selected run header instead of truncating them", () => {
    const goal = "We are a film and TV production house that will be attending Banff World Media Festival in 2026 and need the entire prompt visible in the run header";
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal,
      status: "failed",
      process_name: "ad-hoc",
      plan_subtasks: [],
    };

    render(<RunsTab />);

    expect(screen.getByText(goal)).toHaveClass("whitespace-pre-wrap", "break-words");
    expect(screen.getByText(goal)).not.toHaveClass("truncate");
  });

  it("does not mount tool-call payloads until the row is expanded", async () => {
    const user = userEvent.setup();
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "executing",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.visibleRunTimeline = [
      {
        id: 1,
        task_id: "run-abc",
        run_id: "run-abc",
        correlation_id: "corr-1",
        event_id: "evt-1",
        sequence: 1,
        timestamp: "2026-03-30T20:00:00Z",
        event_type: "tool_call_started",
        source_component: "test",
        schema_version: 1,
        data: {
          tool_name: "write_file",
          args: {
            path: "report.md",
            content: "very secret payload",
          },
        },
      },
    ];

    render(<RunsTab />);

    expect(screen.queryByText("very secret payload")).not.toBeInTheDocument();

    await user.click(screen.getByText("Show tool call"));

    expect(screen.getByText(/very secret payload/i)).toBeInTheDocument();
  });

  it("offers a retry action when a selected run fails to load", async () => {
    const user = userEvent.setup();
    mockApp.selectedRunId = "run-abc";
    mockApp.selectedRunSummary = {
      id: "run-abc",
      goal: "Review the site",
      process_name: "seo-geo-review",
    };
    mockApp.runLoadError = "Load failed";

    render(<RunsTab />);

    await user.click(screen.getByRole("button", { name: "Retry load" }));

    expect(mockApp.refreshRun).toHaveBeenCalledWith("run-abc");
    expect(screen.getByText("Run created, but the detail view did not finish loading")).toBeInTheDocument();
  });

  it("suppresses live streaming animations when a run is paused", () => {
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "paused",
      process_name: "",
      plan_subtasks: [
        {
          id: "fetch-site",
          description: "Fetch the site",
          status: "running",
          depends_on: [],
          phase_id: "",
          is_critical_path: false,
          is_synthesis: false,
        },
      ],
    };
    mockApp.runStreaming = true;
    mockApp.runCanResume = true;
    mockApp.visibleRunTimeline = [];

    const { container } = render(<RunsTab />);

    expect(screen.queryByText("Streaming")).not.toBeInTheDocument();
    expect(container.querySelector(".animate-spin")).toBeNull();
  });

  it("uses 'Give me a challenge' as the default launcher placeholder", () => {
    render(<RunsTab />);

    expect(screen.getByPlaceholderText("Give me a challenge")).toBeInTheDocument();
  });

  it("does not auto-fill the goal when selecting a process", async () => {
    const user = userEvent.setup();
    mockApp.inventory = {
      processes: [
        {
          name: "investment-analysis",
          version: "1.0",
          description: "Decision-grade investment workflow for public equities and private companies.",
          author: "Loom Team",
          path: "/tmp/processes/investment-analysis.yaml",
        },
      ],
    };

    render(<RunsTab />);

    await user.click(screen.getByRole("button", { name: /investment-analysis/i }));

    expect(mockApp.setRunProcess).toHaveBeenCalledWith("investment-analysis");
    expect(mockApp.setRunGoal).not.toHaveBeenCalled();
  });

  it("shows process-specific guidance instead of a generic process description in the goal placeholder", () => {
    mockApp.runProcess = "investment-analysis";
    mockApp.inventory = {
      processes: [
        {
          name: "investment-analysis",
          version: "1.0",
          description: "Decision-grade investment workflow for public equities and private companies.",
          author: "Loom Team",
          path: "/tmp/processes/investment-analysis.yaml",
        },
      ],
    };

    render(<RunsTab />);

    expect(
      screen.getByPlaceholderText(
        "Describe the company, thesis, portfolio, or investor constraint to analyze",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("Decision-grade investment workflow for public equities and private companies.").length,
    ).toBeGreaterThan(0);
  });

  it("refreshes launcher context when the runs tab mounts and when the window regains focus", () => {
    vi.useFakeTimers();

    render(<RunsTab />);

    expect(mockApp.handleRefreshWorkspaceFiles.mock.calls.length).toBeGreaterThan(0);
    expect(mockApp.refreshWorkspaceArtifacts).toHaveBeenLastCalledWith(
      "workspace-1",
      { force: true },
    );

    mockApp.handleRefreshWorkspaceFiles.mockClear();
    mockApp.refreshWorkspaceArtifacts.mockClear();

    vi.advanceTimersByTime(1001);
    window.dispatchEvent(new Event("focus"));

    expect(mockApp.handleRefreshWorkspaceFiles).toHaveBeenCalledTimes(1);
    expect(mockApp.refreshWorkspaceArtifacts).toHaveBeenCalledWith(
      "workspace-1",
      { force: true },
    );
  });

  it("shows an Instructions section with timestamped instruction history", () => {
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "executing",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.runCanMessage = true;
    mockApp.runInstructionHistory = [
      {
        id: "msg-1",
        message: "Focus on Alberta only.",
        summary: "Focus on Alberta only.",
        tags: "conversation",
        timestamp: "2026-03-28T16:45:00Z",
      },
      {
        id: "msg-2",
        message: "Ignore this feedback entry",
        summary: "Ignore this feedback entry",
        tags: "feedback",
        timestamp: "2026-03-28T16:40:00Z",
      },
    ];

    render(<RunsTab />);

    expect(screen.getByText("Instructions")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Add an instruction to this run...")).toBeInTheDocument();
    expect(screen.getByText("Instruction history")).toBeInTheDocument();
    expect(screen.getByText("Focus on Alberta only.")).toBeInTheDocument();
    expect(screen.queryByText("Ignore this feedback entry")).not.toBeInTheDocument();
  });

  it("surfaces pending run approvals inline with approve and deny actions", async () => {
    const user = userEvent.setup();
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "executing",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.approvalInbox = [
      {
        id: "task:run-abc:extract-structured-content",
        kind: "task_approval",
        status: "pending",
        created_at: "2026-03-28T16:45:00Z",
        title: "Extracted content looks complete",
        summary: "Approve use of extracted-content.json",
        workspace_id: "workspace-1",
        workspace_path: "/tmp/workspace",
        workspace_display_name: "Workspace",
        task_id: "run-abc",
        run_id: "run-abc",
        conversation_id: "",
        subtask_id: "extract-structured-content",
        question_id: "",
        approval_id: "",
        tool_name: "",
        risk_level: "medium",
        request_payload: {},
        metadata: {},
      },
    ];

    render(<RunsTab />);

    expect(screen.getByText("Approvals (1)")).toBeInTheDocument();
    expect(screen.getByText("Extracted content looks complete")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Approve" }));
    expect(mockApp.handleReplyApproval).toHaveBeenCalledWith(
      mockApp.approvalInbox[0],
      { decision: "approve" },
    );
  });

  it("caps rendered live activity by default to keep noisy runs responsive", async () => {
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "executing",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.visibleRunTimeline = Array.from({ length: 300 }, (_value, index) => ({
      id: index + 1,
      task_id: "run-abc",
      run_id: "exec-run-1",
      correlation_id: `corr-${index + 1}`,
      event_id: `evt-${index + 1}`,
      sequence: index + 1,
      timestamp: `2026-03-28T16:${String(index % 60).padStart(2, "0")}:00Z`,
      event_type: "tool_call_started",
      source_component: "tests",
      schema_version: 1,
      data: {
        tool_name: "list_directory",
        args: {
          path: `path-${index + 1}.txt`,
        },
      },
    }));

    render(<RunsTab />);

    expect(
      screen.getByText("Showing the latest 250 of 300 events to keep the desktop responsive."),
    ).toBeInTheDocument();
    expect(screen.queryByText("list_directory → path-1.txt")).not.toBeInTheDocument();
    expect(screen.getByText("list_directory → path-300.txt")).toBeInTheDocument();
  });

  it("keeps the run activity timeline on instant scrolling to avoid desktop scroll animation crashes", () => {
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "executing",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.visibleRunTimeline = [
      {
        id: 1,
        task_id: "run-abc",
        run_id: "exec-run-1",
        correlation_id: "corr-1",
        event_id: "evt-1",
        sequence: 1,
        timestamp: "2026-03-28T16:00:00Z",
        event_type: "tool_call_started",
        source_component: "tests",
        schema_version: 1,
        data: {
          tool_name: "list_directory",
          args: {
            path: "path-1.txt",
          },
        },
      },
    ];

    const { container } = render(<RunsTab />);

    expect(container.querySelector(".scroll-smooth")).toBeNull();
  });

  it("groups processes into custom, installed, and built-in sections", () => {
    mockApp.overview = {
      workspace: { canonical_path: "/tmp/workspace" },
    };
    mockApp.inventory = {
      processes: [
        {
          name: "custom-process",
          version: "1.0",
          description: "Workspace-local process.",
          author: "Workspace",
          path: "/tmp/workspace/loom-processes/custom-process.yaml",
        },
        {
          name: "installed-process",
          version: "1.0",
          description: "Installed process.",
          author: "User",
          path: "/Users/test/.loom/processes/installed-process.yaml",
        },
        {
          name: "builtin-process",
          version: "1.0",
          description: "Built-in process.",
          author: "Loom",
          path: "/Applications/Loom.app/processes/builtin-process.yaml",
        },
      ],
    };

    render(<RunsTab />);

    expect(screen.getAllByText("Custom").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Installed").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Built-in").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /custom-process/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /installed-process/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /builtin-process/i })).toBeInTheDocument();
  });

  it("hides hidden files from workspace context suggestions", async () => {
    const user = userEvent.setup();
    mockApp.loadedWorkspaceFileEntries = [
      {
        path: ".DS_Store",
        name: ".DS_Store",
        is_dir: false,
        size_bytes: 128,
        modified_at: "",
        extension: "",
      },
      {
        path: "notes/brief.md",
        name: "brief.md",
        is_dir: false,
        size_bytes: 512,
        modified_at: "",
        extension: "md",
      },
    ];
    mockApp.workspaceArtifacts = [
      {
        path: ".hidden/report.md",
        category: "document",
        source: "seal",
        sha256: "",
        size_bytes: 1024,
        exists_on_disk: true,
        is_intermediate: false,
        created_at: "",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    render(<RunsTab />);

    await user.type(
      screen.getByPlaceholderText("Attach files or folders (optional)"),
      "brief",
    );

    expect(screen.queryByText(".DS_Store")).not.toBeInTheDocument();
    expect(screen.queryByText(".hidden/report.md")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /brief\.md/i })).toBeInTheDocument();
  });

  it("prioritizes recent output suggestions ahead of general workspace files", () => {
    mockApp.loadedWorkspaceFileEntries = [
      {
        path: "research",
        name: "research",
        is_dir: true,
        size_bytes: 0,
        modified_at: "",
        extension: "",
      },
      {
        path: "alpha-notes.md",
        name: "alpha-notes.md",
        is_dir: false,
        size_bytes: 256,
        modified_at: "",
        extension: "md",
      },
    ];
    mockApp.workspaceArtifacts = [
      {
        path: "research/final-gap-analysis.md",
        category: "document",
        source: "seal",
        sha256: "",
        size_bytes: 2048,
        exists_on_disk: true,
        is_intermediate: false,
        created_at: "2026-03-28T10:00:00Z",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    render(<RunsTab />);

    const suggestions = within(
      screen.getByText("Suggested context").closest("div") as HTMLElement,
    ).getAllByRole("button");

    expect(suggestions[0]).not.toHaveTextContent("alpha-notes.md");
    expect(suggestions[0]).toHaveTextContent("recent output");
    expect(suggestions[0]).toHaveTextContent("research");
  });

  it("filters out stale artifact suggestions when the artifact is no longer on disk", () => {
    mockApp.workspaceArtifacts = [
      {
        path: "renamed-folder/report.md",
        category: "document",
        source: "seal",
        sha256: "",
        size_bytes: 2048,
        exists_on_disk: false,
        is_intermediate: false,
        created_at: "2026-03-28T10:00:00Z",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    render(<RunsTab />);

    expect(screen.queryByText("report.md")).not.toBeInTheDocument();
    expect(screen.queryByText("recent output")).not.toBeInTheDocument();
  });

  it("prioritizes folders ahead of files in workspace context suggestions", async () => {
    const user = userEvent.setup();
    mockApp.loadedWorkspaceFileEntries = [
      {
        path: "research",
        name: "research",
        is_dir: true,
        size_bytes: 0,
        modified_at: "",
        extension: "",
      },
      {
        path: "research-summary.md",
        name: "research-summary.md",
        is_dir: false,
        size_bytes: 512,
        modified_at: "",
        extension: "md",
      },
    ];

    render(<RunsTab />);

    await user.type(
      screen.getByPlaceholderText("Attach files or folders (optional)"),
      "research",
    );

    const suggestions = within(
      screen.getByText("Suggested context").closest("div") as HTMLElement,
    ).getAllByRole("button");

    expect(suggestions[0]).toHaveTextContent("research");
    expect(suggestions[1]).toHaveTextContent("research-summary.md");
  });

  it("drops attached paths that disappear after a background refresh", async () => {
    const user = userEvent.setup();
    mockApp.workspaceArtifacts = [
      {
        path: "old-name/report.md",
        category: "document",
        source: "seal",
        sha256: "",
        size_bytes: 2048,
        exists_on_disk: true,
        is_intermediate: false,
        created_at: "2026-03-28T10:00:00Z",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    const { rerender } = render(<RunsTab />);

    await user.click(screen.getByRole("button", { name: /report\.md/i }));
    expect(screen.getByText("old-name/report.md")).toBeInTheDocument();

    mockApp.workspaceArtifacts = [
      {
        path: "old-name/report.md",
        category: "document",
        source: "seal",
        sha256: "",
        size_bytes: 2048,
        exists_on_disk: false,
        is_intermediate: false,
        created_at: "2026-03-28T10:00:00Z",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    rerender(<RunsTab />);

    expect(screen.queryByText("old-name/report.md")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /report\.md/i })).not.toBeInTheDocument();
  });

  it("shows more suggested context items by default inside a scrollable pane", () => {
    mockApp.loadedWorkspaceFileEntries = Array.from({ length: 20 }, (_, index) => ({
      path: `notes/file-${index}.md`,
      name: `file-${index}.md`,
      is_dir: false,
      size_bytes: 256,
      modified_at: "",
      extension: "md",
    }));

    render(<RunsTab />);

    const suggestionsPanel = screen.getByText("Suggested context").closest("div")?.parentElement;
    expect(suggestionsPanel?.querySelector(".max-h-64.overflow-y-auto")).not.toBeNull();
    expect(within(suggestionsPanel as HTMLElement).getAllByRole("button").length).toBe(18);
  });

  it("shows artifact chips with filename-first titles instead of the full folder path", () => {
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "completed",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.visibleRunArtifacts = [
      {
        path: "seo-geo-review-https-www-albertadentalassociation/template-matches.csv",
        category: "structured_data",
        source: "seal",
        sha256: "",
        size_bytes: 3600,
        exists_on_disk: true,
        is_intermediate: false,
        created_at: "",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    render(<RunsTab />);

    expect(screen.getByText("template-matches.csv")).toBeInTheDocument();
    expect(
      screen.getByText("seo-geo-review-https-www-albertadentalassociation"),
    ).toBeInTheDocument();
  });

  it("opens the selected artifact in the Files tab when a file chip is clicked", async () => {
    const user = userEvent.setup();
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "completed",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.visibleRunArtifacts = [
      {
        path: "seo-geo-review-https-www-albertadentalassociation/template-matches.csv",
        category: "structured_data",
        source: "seal",
        sha256: "",
        size_bytes: 3600,
        exists_on_disk: true,
        is_intermediate: false,
        created_at: "",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    render(<RunsTab />);

    await user.click(screen.getByRole("button", { name: /template-matches\.csv/i }));

    expect(mockApp.handleOpenWorkspaceFile).toHaveBeenCalledWith(
      "seo-geo-review-https-www-albertadentalassociation/template-matches.csv",
    );
    expect(mockApp.setActiveTab).toHaveBeenCalledWith("files");
  });

  it("opens timeline file pills through the Files hook so ancestor folders load first", async () => {
    const user = userEvent.setup();
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "completed",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.visibleRunTimeline = [
      {
        id: "evt-1",
        run_id: "run-abc",
        event_type: "tool_call_completed",
        sequence: 1,
        timestamp: "2026-03-28T01:00:00Z",
        data: {
          path: "seo-geo-review-https-www-albertadentalassociation/audit-scorecard.md",
          tool_name: "document_write",
          summary: "Wrote audit scorecard",
        },
      },
    ];

    render(<RunsTab />);

    await user.click(screen.getByRole("button", { name: /audit-scorecard\.md/i }));

    expect(mockApp.handleOpenWorkspaceFile).toHaveBeenCalledWith(
      "seo-geo-review-https-www-albertadentalassociation/audit-scorecard.md",
    );
    expect(mockApp.setActiveTab).toHaveBeenCalledWith("files");
  });

  it("resolves basename-only timeline file pills against the run workspace artifacts", async () => {
    const user = userEvent.setup();
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "completed",
      process_name: "",
      workspace_path: "/tmp/workspace/can-you-convert-all-this-data",
      workspace: {
        canonical_path: "/tmp/workspace",
      },
      plan_subtasks: [],
    };
    mockApp.visibleRunArtifacts = [
      {
        path: "can-you-convert-all-this-data/ui-integration-validation-report.md",
        category: "document",
        source: "seal",
        sha256: "",
        size_bytes: 3600,
        exists_on_disk: true,
        is_intermediate: false,
        created_at: "",
        tool_name: "document_write",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];
    mockApp.visibleRunTimeline = [
      {
        id: "evt-2",
        run_id: "run-abc",
        event_type: "tool_call_completed",
        sequence: 2,
        timestamp: "2026-03-28T01:00:00Z",
        data: {
          path: "ui-integration-validation-report.md",
          tool_name: "document_write",
          summary: "Wrote validation report",
        },
      },
    ];

    render(<RunsTab />);

    const buttons = screen.getAllByRole("button", { name: /ui-integration-validation-report\.md/i });
    await user.click(buttons[buttons.length - 1]);

    expect(mockApp.handleOpenWorkspaceFile).toHaveBeenCalledWith(
      "can-you-convert-all-this-data/ui-integration-validation-report.md",
    );
    expect(mockApp.setActiveTab).toHaveBeenCalledWith("files");
  });

  it("filters out the workspace root artifact entry", () => {
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "completed",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.visibleRunArtifacts = [
      {
        path: ".",
        category: "workspace_file",
        source: "seal",
        sha256: "",
        size_bytes: 0,
        exists_on_disk: true,
        is_intermediate: false,
        created_at: "",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    render(<RunsTab />);

    expect(screen.queryByText(/^Files \(1\)$/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^\.$/ })).not.toBeInTheDocument();
  });

  it("filters out run-subfolder root artifact entries that end in dot", () => {
    mockApp.selectedRunId = "run-abc";
    mockApp.runDetail = {
      id: "run-abc",
      goal: "Review the site",
      status: "completed",
      process_name: "",
      plan_subtasks: [],
    };
    mockApp.visibleRunArtifacts = [
      {
        path: "seo-geo-review-https-www-albertadentalassociation/.",
        category: "workspace_file",
        source: "seal",
        sha256: "",
        size_bytes: 0,
        exists_on_disk: true,
        is_intermediate: false,
        created_at: "",
        tool_name: "",
        subtask_ids: [],
        phase_ids: [],
        facets: {},
      },
    ];

    render(<RunsTab />);

    expect(screen.queryByRole("button", { name: /^\.$/ })).not.toBeInTheDocument();
    expect(
      screen.queryByText("seo-geo-review-https-www-albertadentalassociation"),
    ).not.toBeInTheDocument();
  });
});
