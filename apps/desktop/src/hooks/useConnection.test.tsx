import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useConnection } from "./useConnection";

const apiMocks = vi.hoisted(() => ({
  bootstrapDesktopRuntime: vi.fn(),
  completeInitialSetup: vi.fn(),
  discoverSetupModels: vi.fn(),
  fetchDesktopSidecarStatus: vi.fn(),
  fetchModels: vi.fn(),
  fetchRuntimeStatus: vi.fn(),
  fetchSettings: vi.fn(),
  fetchSetupStatus: vi.fn(),
  fetchWorkspaces: vi.fn(),
  patchSettings: vi.fn(),
}));

vi.mock("../api", () => ({
  bootstrapDesktopRuntime: apiMocks.bootstrapDesktopRuntime,
  completeInitialSetup: apiMocks.completeInitialSetup,
  discoverSetupModels: apiMocks.discoverSetupModels,
  fetchDesktopSidecarStatus: apiMocks.fetchDesktopSidecarStatus,
  fetchModels: apiMocks.fetchModels,
  fetchRuntimeStatus: apiMocks.fetchRuntimeStatus,
  fetchSettings: apiMocks.fetchSettings,
  fetchSetupStatus: apiMocks.fetchSetupStatus,
  fetchWorkspaces: apiMocks.fetchWorkspaces,
  patchSettings: apiMocks.patchSettings,
}));

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}

describe("useConnection", () => {
  beforeEach(() => {
    apiMocks.bootstrapDesktopRuntime.mockReset();
    apiMocks.completeInitialSetup.mockReset();
    apiMocks.discoverSetupModels.mockReset();
    apiMocks.fetchDesktopSidecarStatus.mockReset();
    apiMocks.fetchModels.mockReset();
    apiMocks.fetchRuntimeStatus.mockReset();
    apiMocks.fetchSettings.mockReset();
    apiMocks.fetchSetupStatus.mockReset();
    apiMocks.fetchWorkspaces.mockReset();
    apiMocks.patchSettings.mockReset();

    apiMocks.bootstrapDesktopRuntime.mockResolvedValue(false);
    apiMocks.completeInitialSetup.mockResolvedValue({
      status: "ok",
      config_path: "/tmp/.loom/loom.toml",
    });
    apiMocks.discoverSetupModels.mockResolvedValue({ models: [] });
    apiMocks.fetchDesktopSidecarStatus.mockResolvedValue(null);
    apiMocks.fetchModels.mockResolvedValue([]);
    apiMocks.fetchSettings.mockResolvedValue({ basic: [] });
    apiMocks.fetchSetupStatus.mockResolvedValue({
      needs_setup: false,
      config_path: "/tmp/.loom/loom.toml",
      providers: [],
      role_presets: {},
    });
    apiMocks.fetchWorkspaces.mockResolvedValue([]);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("stays connected when the desktop sidecar is still running", async () => {
    vi.useFakeTimers();
    apiMocks.fetchRuntimeStatus
      .mockResolvedValueOnce({
        ready: true,
        version: "0.3.0",
        workspace_default_path: "/tmp/workspaces",
      })
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue({
        ready: true,
        version: "0.3.0",
        workspace_default_path: "/tmp/workspaces",
      });
    apiMocks.bootstrapDesktopRuntime.mockResolvedValue(true);
    apiMocks.fetchDesktopSidecarStatus.mockResolvedValue({
      running: true,
      managed_by_desktop: true,
      base_url: "http://127.0.0.1:9000",
      pid: 42,
      database_path: "/tmp/loomd.db",
      scratch_dir: "/tmp/scratch",
      workspace_default_path: "/tmp/workspaces",
      log_path: "/tmp/loomd.log",
      runtime: null,
      runtime_error: null,
    });

    const setError = vi.fn();

    const { result } = renderHook(() =>
      useConnection({
        setError,
        showArchivedWorkspaces: false,
        selectedWorkspaceId: "",
        setSelectedWorkspaceId: vi.fn(),
        setCreateParentPath: vi.fn(),
      }),
    );

    await act(async () => {
      await flushPromises();
    });
    expect(result.current.connectionState).toBe("connected");

    await act(async () => {
      vi.advanceTimersByTime(15000);
      await flushPromises();
    });

    expect(result.current.connectionState).toBe("connected");
    expect(setError).not.toHaveBeenCalledWith("Lost connection to Loomd. Reconnecting...");
  });

  it("loads setup status alongside the shell snapshot", async () => {
    apiMocks.fetchRuntimeStatus.mockResolvedValue({
      ready: true,
      version: "0.3.0",
      workspace_default_path: "/tmp/workspaces",
    });
    apiMocks.fetchSetupStatus.mockResolvedValue({
      needs_setup: true,
      config_path: "/tmp/.loom/loom.toml",
      providers: [
        {
          display_name: "Ollama",
          provider_key: "ollama",
          needs_api_key: false,
          default_base_url: "http://localhost:11434",
        },
      ],
      role_presets: {
        all: ["planner"],
      },
    });

    const { result } = renderHook(() =>
      useConnection({
        setError: vi.fn(),
        showArchivedWorkspaces: false,
        selectedWorkspaceId: "",
        setSelectedWorkspaceId: vi.fn(),
        setCreateParentPath: vi.fn(),
      }),
    );

    await act(async () => {
      await flushPromises();
    });

    expect(result.current.setupStatus?.needs_setup).toBe(true);
    expect(result.current.setupStatus?.config_path).toBe("/tmp/.loom/loom.toml");
  });

  it("auto-reconnects after repeated health check failures", async () => {
    vi.useFakeTimers();
    apiMocks.fetchRuntimeStatus
      .mockResolvedValueOnce({
        ready: true,
        version: "0.3.0",
        workspace_default_path: "/tmp/workspaces",
      })
      .mockRejectedValueOnce(new Error("offline"))
      .mockRejectedValueOnce(new Error("offline"))
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue({
        ready: true,
        version: "0.3.0",
        workspace_default_path: "/tmp/workspaces",
      });

    const setError = vi.fn();

    const { result } = renderHook(() =>
      useConnection({
        setError,
        showArchivedWorkspaces: false,
        selectedWorkspaceId: "",
        setSelectedWorkspaceId: vi.fn(),
        setCreateParentPath: vi.fn(),
      }),
    );

    await act(async () => {
      await flushPromises();
    });
    expect(result.current.connectionState).toBe("connected");

    for (let index = 0; index < 3; index += 1) {
      await act(async () => {
        vi.advanceTimersByTime(15000);
        await flushPromises();
      });
    }

    expect(result.current.connectionState).toBe("failed");
    expect(setError).toHaveBeenCalledWith("Lost connection to Loomd. Reconnecting...");

    await act(async () => {
      vi.advanceTimersByTime(1500);
      await flushPromises();
      await flushPromises();
    });

    expect(result.current.connectionState).toBe("connected");
    expect(apiMocks.bootstrapDesktopRuntime).toHaveBeenCalledTimes(2);
  });
});
