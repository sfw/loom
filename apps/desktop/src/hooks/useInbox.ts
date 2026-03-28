import {
  startTransition,
  useState,
} from "react";

import {
  replyApproval,
  type ApprovalFeedItem,
} from "../api";
import {
  type ViewTab,
} from "../utils";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface InboxState {
  approvalReplyDrafts: Record<string, string>;
  replyingApprovalId: string;
}

export interface InboxActions {
  setApprovalReplyDrafts: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  handleReplyApproval: (
    item: ApprovalFeedItem,
    body: {
      decision: string;
      reason?: string;
      response_type?: string;
      selected_option_ids?: string[];
      selected_labels?: string[];
      custom_response?: string;
    },
  ) => Promise<void>;
  handleSelectApprovalContext: (item: ApprovalFeedItem) => void;
  handleSearchResultSelection: (result: {
    conversation_id?: string;
    run_id?: string;
    approval_item_id?: string;
    path?: string;
    kind?: string;
  }) => void;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useInbox(deps: {
  selectedWorkspaceId: string;
  selectedConversationId: string;
  selectedRunId: string;
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
  setSelectedRunId: React.Dispatch<React.SetStateAction<string>>;
  setActiveTab: React.Dispatch<React.SetStateAction<ViewTab>>;
  setError: React.Dispatch<React.SetStateAction<string>>;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
  refreshWorkspaceSurface: (workspaceId: string) => Promise<void>;
  refreshApprovalInbox: (workspaceId: string) => Promise<void>;
  refreshConversation: (conversationId: string) => Promise<void>;
  refreshRun: (runId: string) => Promise<void>;
  handleOpenWorkspaceFile: (path: string) => Promise<void>;
}): InboxState & InboxActions {
  const {
    selectedWorkspaceId,
    selectedConversationId,
    selectedRunId,
    setSelectedConversationId,
    setSelectedRunId,
    setActiveTab,
    setError,
    setNotice,
    refreshWorkspaceSurface,
    refreshApprovalInbox,
    refreshConversation,
    refreshRun,
    handleOpenWorkspaceFile,
  } = deps;

  // State
  const [approvalReplyDrafts, setApprovalReplyDrafts] = useState<Record<string, string>>({});
  const [replyingApprovalId, setReplyingApprovalId] = useState("");

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  async function handleReplyApproval(
    item: ApprovalFeedItem,
    body: {
      decision: string;
      reason?: string;
      response_type?: string;
      selected_option_ids?: string[];
      selected_labels?: string[];
      custom_response?: string;
    },
  ) {
    setReplyingApprovalId(item.id);
    setError("");
    setNotice("");
    try {
      await replyApproval(item.id, {
        ...body,
        source: "desktop",
      });
      setApprovalReplyDrafts((current) => ({
        ...current,
        [item.id]: "",
      }));
      await Promise.all([
        selectedWorkspaceId ? refreshWorkspaceSurface(selectedWorkspaceId) : Promise.resolve(),
        selectedWorkspaceId ? refreshApprovalInbox(selectedWorkspaceId) : Promise.resolve(),
        item.conversation_id && item.conversation_id === selectedConversationId
          ? refreshConversation(item.conversation_id)
          : Promise.resolve(),
        item.task_id && item.task_id === selectedRunId
          ? refreshRun(item.task_id)
          : Promise.resolve(),
      ]);
      setNotice(`${item.title} updated.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reply to approval.");
    } finally {
      setReplyingApprovalId("");
    }
  }

  function handleSelectApprovalContext(item: ApprovalFeedItem) {
    if (item.conversation_id) {
      startTransition(() => {
        setSelectedConversationId(item.conversation_id);
        setActiveTab("threads");
      });
    }
    if (item.task_id) {
      startTransition(() => {
        setSelectedRunId(item.task_id);
        setActiveTab("runs");
      });
    }
  }

  function handleSearchResultSelection(result: {
    conversation_id?: string;
    run_id?: string;
    approval_item_id?: string;
    path?: string;
    kind?: string;
  }) {
    if (result.conversation_id) {
      startTransition(() => {
        setSelectedConversationId(result.conversation_id || "");
        setActiveTab("threads");
      });
    }
    if (result.run_id) {
      startTransition(() => {
        setSelectedRunId(result.run_id || "");
        setActiveTab("runs");
      });
    }
    if ((result.kind === "artifact" || result.kind === "file") && result.path) {
      if (result.kind === "file") {
        setActiveTab("files");
      }
      void handleOpenWorkspaceFile(result.path);
      setNotice(
        result.kind === "file"
          ? `Opened file ${result.path}.`
          : `Opened context for artifact ${result.path}.`,
      );
    }
  }

  return {
    approvalReplyDrafts,
    replyingApprovalId,
    setApprovalReplyDrafts,
    handleReplyApproval,
    handleSelectApprovalContext,
    handleSearchResultSelection,
  };
}
