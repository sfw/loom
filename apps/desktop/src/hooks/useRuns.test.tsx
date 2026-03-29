import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useRuns } from "./useRuns";

const apiMocks = vi.hoisted(() => ({
  createTask: vi.fn(),
  restartRun: vi.fn(),
  fetchRunDetail: vi.fn(),
  fetchRunTimeline: vi.fn(),
  fetchRunArtifacts: vi.fn(),
  fetchRunConversationHistory: vi.fn(),
  subscribeRunStream: vi.fn(() => () => {}),
  cancelRun: vi.fn(),
  deleteRun: vi.fn(),
  pauseRun: vi.fn(),
  resumeRun: vi.fn(),
  sendRunMessage: vi.fn(),
}));

vi.mock("../api", () => ({
  cancelRun: apiMocks.cancelRun,
  createTask: apiMocks.createTask,
  deleteRun: apiMocks.deleteRun,
  fetchRunArtifacts: apiMocks.fetchRunArtifacts,
  fetchRunConversationHistory: apiMocks.fetchRunConversationHistory,
  fetchRunDetail: apiMocks.fetchRunDetail,
  fetchRunTimeline: apiMocks.fetchRunTimeline,
  pauseRun: apiMocks.pauseRun,
  restartRun: apiMocks.restartRun,
  resumeRun: apiMocks.resumeRun,
  sendRunMessage: apiMocks.sendRunMessage,
  subscribeRunStream: apiMocks.subscribeRunStream,
}));

vi.mock("../history", () => ({
  isRunTimelineNoise: () => false,
  matchesWorkspaceSearch: () => true,
  runTimelineDetail: () => "",
  runTimelinePills: () => [],
  runTimelineTitle: () => "",
}));

describe("useRuns", () => {
  beforeEach(() => {
    apiMocks.createTask.mockReset();
    apiMocks.restartRun.mockReset();
    apiMocks.fetchRunDetail.mockReset();
    apiMocks.fetchRunTimeline.mockReset();
    apiMocks.fetchRunArtifacts.mockReset();
    apiMocks.fetchRunConversationHistory.mockReset();
    apiMocks.subscribeRunStream.mockClear();
    apiMocks.cancelRun.mockReset();
    apiMocks.deleteRun.mockReset();
    apiMocks.pauseRun.mockReset();
    apiMocks.resumeRun.mockReset();
    apiMocks.sendRunMessage.mockReset();
    apiMocks.fetchRunConversationHistory.mockResolvedValue([]);
  });

  it("opens the newly created run by task id and does not block launch on workspace refresh", async () => {
    apiMocks.createTask.mockResolvedValue({
      task_id: "task-123",
      run_id: "run-abc",
      status: "executing",
      message: "Task created and execution started.",
    });

    const setSelectedRunId = vi.fn();
    const setError = vi.fn();
    const setNotice = vi.fn();
    const setActiveTab = vi.fn();
    const refreshWorkspaceSurface = vi.fn(async () => {
      throw new Error("Load failed");
    });

    const { result } = renderHook(() =>
      useRuns({
        selectedRunId: "",
        setSelectedRunId,
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {
            canonical_path: "/tmp/workspace",
          },
          recent_runs: [],
        } as any,
        setError,
        setNotice,
        setActiveTab,
        refreshWorkspaceSurface,
      }),
    );

    act(() => {
      result.current.setRunGoal("Review the site");
      result.current.setRunProcess("seo-geo-review");
    });

    await act(async () => {
      await result.current.handleLaunchRun({
        preventDefault() {},
      } as any);
    });

    expect(apiMocks.createTask).toHaveBeenCalledWith({
      goal: "Review the site",
      workspace: "/tmp/workspace",
      process: "seo-geo-review",
      approval_mode: "auto",
      context: undefined,
      auto_subfolder: true,
    });
    expect(setSelectedRunId).toHaveBeenCalledWith("task-123");
    expect(setActiveTab).toHaveBeenCalledWith("runs");
    expect(setNotice).toHaveBeenCalledWith("Launched run task-123.");
    expect(setError).not.toHaveBeenCalledWith("Load failed");
  });

  it("restarts in place by keeping the stable task id selected", async () => {
    apiMocks.restartRun.mockResolvedValue({
      task_id: "task-123",
      run_id: "run-retry-2",
      status: "pending",
      message: "Restarted run task-123.",
    });

    const setSelectedRunId = vi.fn();
    const setError = vi.fn();
    const setNotice = vi.fn();
    const setActiveTab = vi.fn();
    const refreshWorkspaceSurface = vi.fn(async () => {});

    const { result } = renderHook(() =>
      useRuns({
        selectedRunId: "task-123",
        setSelectedRunId,
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {
            canonical_path: "/tmp/workspace",
          },
          recent_runs: [],
        } as any,
        setError,
        setNotice,
        setActiveTab,
        refreshWorkspaceSurface,
      }),
    );

    await act(async () => {
      await result.current.handleRestartRun();
    });

    expect(apiMocks.restartRun).toHaveBeenCalledWith("task-123");
    expect(setSelectedRunId).toHaveBeenCalledWith("task-123");
    expect(setActiveTab).toHaveBeenCalledWith("runs");
    expect(setNotice).toHaveBeenCalledWith("Relaunched as task-123.");
  });

  it("refreshes the selected run when the live stream errors", async () => {
    vi.useFakeTimers();
    let streamError: (() => void) | undefined;
    apiMocks.fetchRunDetail.mockResolvedValue({
      id: "run-1",
      goal: "Review the site",
      status: "executing",
      process_name: "seo-geo-review",
      plan_subtasks: [],
    });
    apiMocks.fetchRunTimeline.mockResolvedValue([]);
    apiMocks.fetchRunArtifacts.mockResolvedValue([]);
    apiMocks.subscribeRunStream.mockImplementation(((
      _runId: string,
      _onEvent: (event: unknown) => void,
      onError?: () => void,
    ) => {
      streamError = onError;
      return () => {};
    }) as any);

    renderHook(() =>
      useRuns({
        selectedRunId: "run-1",
        setSelectedRunId: vi.fn(),
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {
            canonical_path: "/tmp/workspace",
          },
          recent_runs: [],
        } as any,
        setError: vi.fn(),
        setNotice: vi.fn(),
        setActiveTab: vi.fn(),
        refreshWorkspaceSurface: vi.fn(async () => {}),
      }),
    );

    await act(async () => {
      await Promise.resolve();
    });
    apiMocks.fetchRunDetail.mockClear();
    apiMocks.fetchRunTimeline.mockClear();
    apiMocks.fetchRunArtifacts.mockClear();

    act(() => {
      streamError?.();
      vi.advanceTimersByTime(250);
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.fetchRunDetail).toHaveBeenCalledWith("run-1");
    expect(apiMocks.fetchRunTimeline).toHaveBeenCalledWith("run-1");
    expect(apiMocks.fetchRunArtifacts).toHaveBeenCalledWith("run-1");
    vi.useRealTimers();
  });

  it("subscribes after the latest loaded timeline row and appends streamed rows", async () => {
    let streamEvent: ((event: any) => void) | undefined;
    apiMocks.fetchRunDetail.mockResolvedValue({
      id: "run-1",
      goal: "Review the site",
      status: "executing",
      process_name: "seo-geo-review",
      plan_subtasks: [],
    });
    apiMocks.fetchRunTimeline.mockResolvedValue([
      {
        id: 3,
        task_id: "run-1",
        run_id: "exec-run-1",
        correlation_id: "corr-1",
        event_id: "evt-3",
        sequence: 3,
        timestamp: "2026-03-27T00:00:03Z",
        event_type: "task_executing",
        source_component: "tests",
        schema_version: 1,
        data: { status: "executing" },
      },
    ]);
    apiMocks.fetchRunArtifacts.mockResolvedValue([]);
    apiMocks.subscribeRunStream.mockImplementation(((
      _runId: string,
      onEvent: (event: unknown) => void,
    ) => {
      streamEvent = onEvent as (event: any) => void;
      return () => {};
    }) as any);

    const { result } = renderHook(() =>
      useRuns({
        selectedRunId: "run-1",
        setSelectedRunId: vi.fn(),
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {
            canonical_path: "/tmp/workspace",
          },
          recent_runs: [],
        } as any,
        setError: vi.fn(),
        setNotice: vi.fn(),
        setActiveTab: vi.fn(),
        refreshWorkspaceSurface: vi.fn(async () => {}),
      }),
    );

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMocks.subscribeRunStream).toHaveBeenCalledWith(
      "run-1",
      expect.any(Function),
      expect.any(Function),
      { afterId: 3 },
    );

    act(() => {
      streamEvent?.({
        id: 4,
        task_id: "run-1",
        run_id: "exec-run-1",
        correlation_id: "corr-1",
        event_id: "evt-4",
        sequence: 4,
        timestamp: "2026-03-27T00:00:04Z",
        event_type: "task_paused",
        source_component: "tests",
        schema_version: 1,
        data: { status: "paused", message: "paused" },
        status: "paused",
        streaming: false,
      });
    });

    expect(result.current.runTimeline).toHaveLength(2);
    expect(result.current.runTimeline[1]?.id).toBe(4);
    expect(result.current.runDetail?.status).toBe("paused");
    expect(result.current.runStreaming).toBe(false);
  });

  it("does not drop newer streamed timeline rows during a stale refresh", async () => {
    let streamEvent: ((event: any) => void) | undefined;
    apiMocks.fetchRunDetail.mockResolvedValue({
      id: "run-1",
      goal: "Review the site",
      status: "executing",
      process_name: "seo-geo-review",
      plan_subtasks: [],
    });
    apiMocks.fetchRunTimeline.mockResolvedValueOnce([
      {
        id: 238,
        task_id: "run-1",
        run_id: "exec-run-1",
        correlation_id: "corr-1",
        event_id: "evt-238",
        sequence: 238,
        timestamp: "2026-03-27T00:03:58Z",
        event_type: "tool_call_completed",
        source_component: "tests",
        schema_version: 1,
        data: {},
      },
    ]);
    apiMocks.fetchRunTimeline.mockResolvedValueOnce([
      {
        id: 238,
        task_id: "run-1",
        run_id: "exec-run-1",
        correlation_id: "corr-1",
        event_id: "evt-238",
        sequence: 238,
        timestamp: "2026-03-27T00:03:58Z",
        event_type: "tool_call_completed",
        source_component: "tests",
        schema_version: 1,
        data: {},
      },
    ]);
    apiMocks.fetchRunArtifacts.mockResolvedValue([]);
    apiMocks.subscribeRunStream.mockImplementation(((
      _runId: string,
      onEvent: (event: unknown) => void,
    ) => {
      streamEvent = onEvent as (event: any) => void;
      return () => {};
    }) as any);

    const { result } = renderHook(() =>
      useRuns({
        selectedRunId: "run-1",
        setSelectedRunId: vi.fn(),
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {
            canonical_path: "/tmp/workspace",
          },
          recent_runs: [],
        } as any,
        setError: vi.fn(),
        setNotice: vi.fn(),
        setActiveTab: vi.fn(),
        refreshWorkspaceSurface: vi.fn(async () => {}),
      }),
    );

    await act(async () => {
      await Promise.resolve();
    });

    act(() => {
      streamEvent?.({
        id: 239,
        task_id: "run-1",
        run_id: "exec-run-1",
        correlation_id: "corr-1",
        event_id: "evt-239",
        sequence: 239,
        timestamp: "2026-03-27T00:03:59Z",
        event_type: "tool_call_completed",
        source_component: "tests",
        schema_version: 1,
        data: {},
      });
      streamEvent?.({
        id: 240,
        task_id: "run-1",
        run_id: "exec-run-1",
        correlation_id: "corr-1",
        event_id: "evt-240",
        sequence: 240,
        timestamp: "2026-03-27T00:04:00Z",
        event_type: "verification_started",
        source_component: "tests",
        schema_version: 1,
        data: {},
      });
    });

    expect(result.current.runTimeline.map((row) => row.id)).toEqual([238, 239, 240]);

    await act(async () => {
      await result.current.refreshRun("run-1");
    });

    expect(result.current.runTimeline.map((row) => row.id)).toEqual([238, 239, 240]);
  });

  it("forwards attached workspace context when launching a run", async () => {
    apiMocks.createTask.mockResolvedValue({
      task_id: "task-456",
      run_id: "run-def",
      status: "executing",
      message: "Task created and execution started.",
    });

    const { result } = renderHook(() =>
      useRuns({
        selectedRunId: "",
        setSelectedRunId: vi.fn(),
        selectedWorkspaceId: "workspace-1",
        overview: {
          workspace: {
            canonical_path: "/tmp/workspace",
          },
          recent_runs: [],
        } as any,
        setError: vi.fn(),
        setNotice: vi.fn(),
        setActiveTab: vi.fn(),
        refreshWorkspaceSurface: vi.fn(async () => {}),
      }),
    );

    act(() => {
      result.current.setRunGoal("Synthesize the existing research");
      result.current.setRunProcess("gap-analysis");
    });

    await act(async () => {
      await result.current.handleLaunchRun(
        { preventDefault() {} } as any,
        {
          workspace_paths: ["research/output.md", "research"],
          workspace_files: ["research/output.md"],
          workspace_directories: ["research"],
        },
      );
    });

    expect(apiMocks.createTask).toHaveBeenCalledWith({
      goal: "Synthesize the existing research",
      workspace: "/tmp/workspace",
      process: "gap-analysis",
      approval_mode: "auto",
      context: {
        workspace_paths: ["research/output.md", "research"],
        workspace_files: ["research/output.md"],
        workspace_directories: ["research"],
      },
      auto_subfolder: true,
    });
  });
});
