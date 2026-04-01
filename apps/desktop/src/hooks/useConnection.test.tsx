import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useConnection } from "./useConnection";

const apiMocks = vi.hoisted(() => ({
  bootstrapDesktopRuntime: vi.fn(),
  fetchModels: vi.fn(),
  fetchRuntimeStatus: vi.fn(),
  fetchSettings: vi.fn(),
  fetchWorkspaces: vi.fn(),
  patchSettings: vi.fn(),
}));

vi.mock("../api", () => ({
  bootstrapDesktopRuntime: apiMocks.bootstrapDesktopRuntime,
  fetchModels: apiMocks.fetchModels,
  fetchRuntimeStatus: apiMocks.fetchRuntimeStatus,
  fetchSettings: apiMocks.fetchSettings,
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
    apiMocks.fetchModels.mockReset();
    apiMocks.fetchRuntimeStatus.mockReset();
    apiMocks.fetchSettings.mockReset();
    apiMocks.fetchWorkspaces.mockReset();
    apiMocks.patchSettings.mockReset();

    apiMocks.bootstrapDesktopRuntime.mockResolvedValue(false);
    apiMocks.fetchModels.mockResolvedValue([]);
    apiMocks.fetchSettings.mockResolvedValue({ basic: [] });
    apiMocks.fetchWorkspaces.mockResolvedValue([]);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("auto-reconnects after the runtime health check fails", async () => {
    vi.useFakeTimers();
    apiMocks.fetchRuntimeStatus
      .mockResolvedValueOnce({
        ready: true,
        version: "0.2.2",
        workspace_default_path: "/tmp/workspaces",
      })
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValue({
        ready: true,
        version: "0.2.2",
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

    await act(async () => {
      vi.advanceTimersByTime(5000);
      await flushPromises();
    });

    expect(result.current.connectionState).toBe("failed");
    expect(setError).toHaveBeenCalledWith("Lost connection to Loomd. Reconnecting...");

    await act(async () => {
      vi.advanceTimersByTime(1500);
      await flushPromises();
    });

    expect(result.current.connectionState).toBe("connected");
    expect(apiMocks.bootstrapDesktopRuntime).toHaveBeenCalledTimes(2);
  });
});
