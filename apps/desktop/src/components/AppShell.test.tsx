import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AppShell from "./AppShell";

let mockApp: any;
let closeHandler: ((event: { preventDefault: () => void }) => Promise<void> | void) | null;
const mockClose = vi.fn(async () => {});
const mockUnlisten = vi.fn();
const mockConfirm = vi.fn<(message: string, options?: { title?: string; kind?: string }) => Promise<boolean>>(async () => false);

vi.mock("@/context/AppContext", () => ({
  shallowEqual: (left: unknown, right: unknown) => left === right,
  useApp: () => mockApp,
  useAppActions: () => mockApp,
  useAppSelector: (selector: (state: any) => unknown) => selector(mockApp),
}));

vi.mock("./Sidebar", () => ({ default: () => <div>Sidebar</div> }));
vi.mock("./OverviewTab", () => ({ default: () => <div>Overview</div> }));
vi.mock("./ThreadsTab", () => ({ default: () => <div>Threads</div> }));
vi.mock("./RunsTab", () => ({ default: () => <div>Runs</div> }));
vi.mock("./FilesTab", () => ({ default: () => <div>Files</div> }));
vi.mock("./IntegrationsTab", () => ({ default: () => <div>Integrations</div> }));
vi.mock("./DesktopSetupWizard", () => ({ default: () => <div>Setup Wizard</div> }));
vi.mock("./SettingsPanel", () => ({ default: () => <div>Settings</div> }));
vi.mock("./WorkspaceModal", () => ({ default: () => <div>Workspace Modal</div> }));
vi.mock("../CommandPalette", () => ({ default: () => <div>Command Palette</div> }));

vi.mock("@tauri-apps/api/window", () => ({
  getCurrentWindow: () => ({
    onCloseRequested: vi.fn(async (handler: (event: { preventDefault: () => void }) => Promise<void> | void) => {
      closeHandler = handler;
      return mockUnlisten;
    }),
    close: mockClose,
  }),
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  confirm: async (message: string, options?: { title?: string; kind?: string }) => mockConfirm(message, options),
}));

describe("AppShell close confirmation", () => {
  beforeEach(() => {
    closeHandler = null;
    mockClose.mockClear();
    mockConfirm.mockReset();
    mockConfirm.mockResolvedValue(false);
    mockUnlisten.mockClear();

    mockApp = {
      activeTab: "overview",
      setActiveTab: vi.fn(),
      workspaces: [
        {
          id: "workspace-1",
          canonical_path: "/tmp/workspace-1",
          display_name: "Workspace One",
          workspace_type: "local",
          is_archived: false,
          sort_order: 0,
          last_opened_at: "2026-03-27T00:00:00Z",
          created_at: "2026-03-27T00:00:00Z",
          updated_at: "2026-03-27T00:00:00Z",
          metadata: {},
          exists_on_disk: true,
          conversation_count: 0,
          run_count: 1,
          active_run_count: 0,
          last_activity_at: "2026-03-27T00:00:00Z",
        },
      ],
      selectedWorkspaceSummary: {
        id: "workspace-1",
        canonical_path: "/tmp/workspace-1",
        display_name: "Workspace One",
        workspace_type: "local",
        is_archived: false,
        sort_order: 0,
        last_opened_at: "2026-03-27T00:00:00Z",
        created_at: "2026-03-27T00:00:00Z",
        updated_at: "2026-03-27T00:00:00Z",
        metadata: {},
        exists_on_disk: true,
        conversation_count: 0,
        run_count: 1,
        active_run_count: 0,
        last_activity_at: "2026-03-27T00:00:00Z",
      },
      selectedWorkspaceId: "workspace-1",
      commandPaletteOpen: false,
      setCommandPaletteOpen: vi.fn(),
      discoverSetupModels: vi.fn(),
      completeInitialSetup: vi.fn(),
      commandDraft: "",
      setCommandDraft: vi.fn(),
      commandInputRef: { current: null },
      handleCommandInputKeyDown: vi.fn(),
      handleCommandSubmit: vi.fn(),
      paletteEntries: [],
      paletteSections: [],
      activeCommandIndex: 0,
      executePaletteEntry: vi.fn(),
      searchingCommandPalette: false,
      focusCommandBar: vi.fn(),
      showNewWorkspace: false,
      error: "",
      notice: "",
      setError: vi.fn(),
      setNotice: vi.fn(),
      runtime: { ready: true, version: "0.3.0" },
      setupStatus: { needs_setup: false, config_path: "/tmp/.loom/loom.toml", providers: [], role_presets: {} },
      overview: {
        workspace: null,
        recent_conversations: [],
        recent_runs: [],
        pending_approvals_count: 0,
        counts: {},
      },
      approvalInbox: [],
      connectionState: "connected",
      retryConnection: vi.fn(),
    };
  });

  afterEach(() => {
    closeHandler = null;
  });

  it("registers and cleans up the window close listener", async () => {
    const view = render(<AppShell />);

    await waitFor(() => expect(closeHandler).not.toBeNull());
    view.unmount();

    expect(mockUnlisten).toHaveBeenCalledTimes(1);
  });

  it("allows closing without confirmation when there are no active runs", async () => {
    render(<AppShell />);
    await waitFor(() => expect(closeHandler).not.toBeNull());

    const event = { preventDefault: vi.fn() };
    await closeHandler?.(event);

    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(mockConfirm).not.toHaveBeenCalled();
    expect(mockClose).not.toHaveBeenCalled();
  });

  it("confirms close when any workspace has active runs and cancels cleanly", async () => {
    mockApp.workspaces[0].active_run_count = 2;
    mockApp.selectedWorkspaceSummary.active_run_count = 0;
    render(<AppShell />);
    await waitFor(() => expect(closeHandler).not.toBeNull());

    const event = { preventDefault: vi.fn() };
    await closeHandler?.(event);

    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(mockConfirm).toHaveBeenCalledWith(
      expect.stringContaining("pause 2 active runs"),
      expect.objectContaining({ title: "Close Loom Desktop?", kind: "warning" }),
    );
    expect(mockClose).not.toHaveBeenCalled();
  });

  it("closes after confirmation when active runs are in flight", async () => {
    mockApp.workspaces[0].active_run_count = 1;
    mockApp.selectedWorkspaceSummary.active_run_count = 1;
    mockConfirm.mockResolvedValue(true);
    render(<AppShell />);
    await waitFor(() => expect(closeHandler).not.toBeNull());

    const event = { preventDefault: vi.fn() };
    await closeHandler?.(event);

    expect(event.preventDefault).toHaveBeenCalledTimes(1);
    expect(mockClose).toHaveBeenCalledTimes(1);
  });

  it("redirects the retired inbox tab back to overview", async () => {
    mockApp.activeTab = "inbox";
    mockApp.setActiveTab = vi.fn();

    render(<AppShell />);

    await waitFor(() => expect(mockApp.setActiveTab).toHaveBeenCalledWith("overview"));
  });

  it("uses the selected workspace overview count for the running chip when the list summary drifts", () => {
    mockApp.selectedWorkspaceSummary.active_run_count = 2;
    mockApp.overview = {
      workspace: {
        ...mockApp.selectedWorkspaceSummary,
        active_run_count: 1,
      },
      recent_conversations: [],
      recent_runs: [],
      pending_approvals_count: 0,
      counts: {},
    };

    render(<AppShell />);

    expect(screen.getByText("1 running")).toBeInTheDocument();
    expect(screen.queryByText("2 running")).not.toBeInTheDocument();
  });

  it("uses the selected workspace overview count for the pending chip when the inbox drifts", () => {
    mockApp.approvalInbox = Array.from({ length: 4 }, (_, index) => ({ id: `approval-${index}` }));
    mockApp.overview = {
      workspace: {
        ...mockApp.selectedWorkspaceSummary,
      },
      recent_conversations: [],
      recent_runs: [],
      pending_approvals_count: 0,
      counts: {},
    };

    render(<AppShell />);

    expect(screen.queryByText("4")).not.toBeInTheDocument();
  });

  it("keeps the shell visible and shows a reconnect banner when cached data exists", () => {
    mockApp.connectionState = "failed";

    render(<AppShell />);

    expect(screen.getByText("Sidebar")).toBeInTheDocument();
    expect(
      screen.getByText("Connection to Loomd dropped. Retrying automatically..."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Cannot reach Loomd")).not.toBeInTheDocument();
  });

  it("renders the integrations tab as a first-class top-level surface", () => {
    mockApp.activeTab = "integrations";

    render(<AppShell />);

    expect(screen.getByText("Integrations")).toBeInTheDocument();
    expect(screen.queryByText("Settings")).not.toBeInTheDocument();
  });

  it("blocks on the first-run setup wizard when models are not configured yet", () => {
    mockApp.setupStatus = {
      needs_setup: true,
      config_path: "/tmp/.loom/loom.toml",
      providers: [],
      role_presets: {},
    };

    render(<AppShell />);

    expect(screen.getByText("Setup Wizard")).toBeInTheDocument();
    expect(screen.queryByText("Sidebar")).not.toBeInTheDocument();
  });
});
