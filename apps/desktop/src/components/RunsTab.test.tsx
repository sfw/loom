import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import RunsTab from "./RunsTab";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  useApp: () => mockApp,
}));

describe("RunsTab", () => {
  beforeEach(() => {
    mockApp = {
      filteredRuns: [],
      selectedRunId: "",
      setSelectedRunId: vi.fn(),
      selectedRunSummary: null,
      runDetail: null,
      runTimeline: [],
      runArtifacts: [],
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
      handleSendRunMessage: vi.fn(async () => {}),
      handleOpenWorkspaceFile: vi.fn(async () => {}),
      refreshRun: vi.fn(async () => {}),
      stepRunMatch: vi.fn(),
      selectedWorkspaceId: "workspace-1",
      workspaceSearchQuery: "",
      overview: {
        workspace: { canonical_path: "/tmp/workspace" },
      },
      inventory: { processes: [] },
      loadedWorkspaceFileEntries: [],
      loadWorkspaceDirectory: vi.fn(async () => {}),
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
    mockApp.recentWorkspaceArtifacts = [
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
    mockApp.recentWorkspaceArtifacts = [
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
