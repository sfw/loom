import {
  useEffect,
  useEffectEvent,
  useState,
} from "react";

import {
  fetchActivitySummary,
  type ActivitySummary,
} from "../api";
import { isTransientRequestError } from "../utils";

const ACTIVE_ACTIVITY_POLL_MS = 3000;
const IDLE_ACTIVITY_POLL_MS = 10000;

export interface DesktopActivityState {
  active: boolean;
  mode: "idle" | "thread" | "run" | "mixed";
  activeConversationCount: number;
  activeRunCount: number;
  sourceCount: number;
  label: string;
  updatedAt: string;
  backendConnected: boolean;
}

export function useDesktopActivity(deps: {
  connectionState: "connecting" | "connected" | "failed";
  conversationTurnPending: boolean;
  conversationStreaming: boolean;
  streamingToolCalls: Array<{ completed: boolean }>;
  runStreaming: boolean;
}): DesktopActivityState {
  const {
    connectionState,
    conversationTurnPending,
    conversationStreaming,
    streamingToolCalls,
    runStreaming,
  } = deps;

  const [backendSummary, setBackendSummary] = useState<ActivitySummary | null>(null);
  const [backendConnected, setBackendConnected] = useState(false);

  const localConversationActive = Boolean(
    conversationTurnPending
    || conversationStreaming
    || streamingToolCalls.some((toolCall) => !toolCall.completed),
  );
  const localRunActive = Boolean(runStreaming);
  const localActivityActive = localConversationActive || localRunActive;

  const refreshActivitySummary = useEffectEvent(async () => {
    try {
      const summary = await fetchActivitySummary();
      setBackendSummary(summary);
      setBackendConnected(true);
    } catch (error) {
      if (!isTransientRequestError(error)) {
        setBackendConnected(false);
      }
    }
  });

  useEffect(() => {
    if (connectionState !== "connected") {
      setBackendSummary(null);
      setBackendConnected(false);
      return;
    }

    void refreshActivitySummary();
  }, [connectionState, refreshActivitySummary]);

  useEffect(() => {
    if (connectionState !== "connected") {
      return;
    }

    const intervalMs = (backendSummary?.active || localActivityActive)
      ? ACTIVE_ACTIVITY_POLL_MS
      : IDLE_ACTIVITY_POLL_MS;
    const intervalId = window.setInterval(() => {
      void refreshActivitySummary();
    }, intervalMs);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [
    backendSummary?.active,
    connectionState,
    localActivityActive,
    refreshActivitySummary,
  ]);

  const activeConversationCount = Math.max(
    backendSummary?.active_conversation_count || 0,
    localConversationActive ? 1 : 0,
  );
  const activeRunCount = Math.max(
    backendSummary?.active_run_count || 0,
    localRunActive ? 1 : 0,
  );
  const sourceCount = activeConversationCount + activeRunCount;
  const active = sourceCount > 0;

  let mode: DesktopActivityState["mode"] = "idle";
  if (activeConversationCount > 0 && activeRunCount > 0) {
    mode = "mixed";
  } else if (activeConversationCount > 0) {
    mode = "thread";
  } else if (activeRunCount > 0) {
    mode = "run";
  }

  const labelParts: string[] = [];
  if (activeConversationCount > 0) {
    labelParts.push(
      activeConversationCount === 1
        ? "1 active thread"
        : `${activeConversationCount} active threads`,
    );
  }
  if (activeRunCount > 0) {
    labelParts.push(
      activeRunCount === 1
        ? "1 active run"
        : `${activeRunCount} active runs`,
    );
  }

  return {
    active,
    mode,
    activeConversationCount,
    activeRunCount,
    sourceCount,
    label: labelParts.length > 0 ? labelParts.join(" · ") : "Idle",
    updatedAt: backendSummary?.updated_at || "",
    backendConnected,
  };
}
