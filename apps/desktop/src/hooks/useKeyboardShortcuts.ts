import {
  startTransition,
  useEffect,
} from "react";

import type { WorkspaceFilePreview, WorkspaceOverview, WorkspaceSummary } from "../api";
import {
  adjacentItemId,
  isTypingIntoField,
  type ViewTab,
} from "../utils";

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useKeyboardShortcuts(deps: {
  overview: WorkspaceOverview | null;
  workspaces: WorkspaceSummary[];
  selectedWorkspaceId: string;
  selectedConversationId: string;
  selectedRunId: string;
  workspaceFilePreview: WorkspaceFilePreview | null;
  workspaceFileEditorDraft: string;
  setSelectedWorkspaceId: React.Dispatch<React.SetStateAction<string>>;
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
  setSelectedRunId: React.Dispatch<React.SetStateAction<string>>;
  setActiveTab: React.Dispatch<React.SetStateAction<ViewTab>>;
  focusSearch: () => void;
  focusCommandBar: () => void;
  handleSaveWorkspaceFile: () => Promise<void>;
}): void {
  const {
    overview,
    workspaces,
    selectedWorkspaceId,
    selectedConversationId,
    selectedRunId,
    workspaceFilePreview,
    workspaceFileEditorDraft,
    setSelectedWorkspaceId,
    setSelectedConversationId,
    setSelectedRunId,
    setActiveTab,
    focusSearch,
    focusCommandBar,
    handleSaveWorkspaceFile,
  } = deps;

  useEffect(() => {
    function handleKeydown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        focusCommandBar();
        return;
      }
      if (
        (event.metaKey || event.ctrlKey)
        && event.key.toLowerCase() === "s"
        && workspaceFilePreview?.preview_kind === "text"
        && !workspaceFilePreview.error
        && !workspaceFilePreview.truncated
        && workspaceFileEditorDraft !== (workspaceFilePreview.text_content || "")
      ) {
        event.preventDefault();
        void handleSaveWorkspaceFile();
        return;
      }
      if (!event.metaKey && !event.ctrlKey && !event.altKey && event.key === "/") {
        if (isTypingIntoField(event.target)) {
          return;
        }
        event.preventDefault();
        focusSearch();
        return;
      }
      if (isTypingIntoField(event.target)) {
        return;
      }
      if (!event.altKey) {
        return;
      }

      const conversations = overview?.recent_conversations || [];
      const runs = overview?.recent_runs || [];

      if (event.shiftKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
        if (conversations.length === 0) {
          return;
        }
        event.preventDefault();
        const delta = event.key === "ArrowUp" ? -1 : 1;
        const nextConversationId = adjacentItemId(
          conversations,
          selectedConversationId,
          delta,
        );
        startTransition(() => {
          setSelectedConversationId(nextConversationId);
          setActiveTab("threads");
        });
        return;
      }

      if (!event.shiftKey && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
        if (workspaces.length === 0) {
          return;
        }
        event.preventDefault();
        const delta = event.key === "ArrowUp" ? -1 : 1;
        const nextWorkspaceId = adjacentItemId(workspaces, selectedWorkspaceId, delta);
        startTransition(() => {
          setSelectedWorkspaceId(nextWorkspaceId);
        });
        return;
      }

      if (!event.shiftKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
        if (runs.length === 0) {
          return;
        }
        event.preventDefault();
        const delta = event.key === "ArrowLeft" ? -1 : 1;
        const nextRunId = adjacentItemId(runs, selectedRunId, delta);
        startTransition(() => {
          setSelectedRunId(nextRunId);
          setActiveTab("runs");
        });
      }
    }

    window.addEventListener("keydown", handleKeydown);
    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  }, [
    overview,
    selectedConversationId,
    selectedWorkspaceId,
    selectedRunId,
    workspaceFileEditorDraft,
    workspaceFilePreview,
    workspaces,
  ]);
}
