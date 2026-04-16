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
    workspace_id?: string;
    conversation_id?: string;
    run_id?: string;
    approval_item_id?: string;
    item_id?: string;
    title?: string;
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
  setSelectedWorkspaceId: React.Dispatch<React.SetStateAction<string>>;
  setSelectedConversationId: React.Dispatch<React.SetStateAction<string>>;
  setSelectedRunId: React.Dispatch<React.SetStateAction<string>>;
  setWorkspaceSearchQuery: React.Dispatch<React.SetStateAction<string>>;
  setActiveTab: React.Dispatch<React.SetStateAction<ViewTab>>;
  setRunProcess: React.Dispatch<React.SetStateAction<string>>;
  setError: React.Dispatch<React.SetStateAction<string>>;
  setNotice: React.Dispatch<React.SetStateAction<string>>;
  removeApprovalItem: (itemId: string, workspaceId?: string) => void;
  refreshConversation: (conversationId: string) => Promise<void>;
  refreshRun: (runId: string) => Promise<void>;
  queueWorkspaceFileOpen: (workspaceId: string, path: string) => void;
  focusRunComposer: () => void;
}): InboxState & InboxActions {
  const {
    selectedWorkspaceId,
    selectedConversationId,
    selectedRunId,
    setSelectedWorkspaceId,
    setSelectedConversationId,
    setSelectedRunId,
    setWorkspaceSearchQuery,
    setActiveTab,
    setRunProcess,
    setError,
    setNotice,
    removeApprovalItem,
    refreshConversation,
    refreshRun,
    queueWorkspaceFileOpen,
    focusRunComposer,
  } = deps;

  // State
  const [approvalReplyDrafts, setApprovalReplyDrafts] = useState<Record<string, string>>({});
  const [replyingApprovalId, setReplyingApprovalId] = useState("");

  function isApprovalReplyNotFound(error: unknown): boolean {
    return error instanceof Error && /^\s*404\b/.test(error.message);
  }

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
    const refreshCurrentContext = async () => {
      await Promise.all([
        item.conversation_id && item.conversation_id === selectedConversationId
          ? refreshConversation(item.conversation_id)
          : Promise.resolve(),
        item.task_id && item.task_id === selectedRunId
          ? refreshRun(item.task_id)
          : Promise.resolve(),
      ]);
    };
    try {
      await replyApproval(item.id, {
        ...body,
        source: "desktop",
      });
      removeApprovalItem(item.id, item.workspace_id || selectedWorkspaceId);
      setApprovalReplyDrafts((current) => ({
        ...current,
        [item.id]: "",
      }));
      await refreshCurrentContext();
      setNotice(`${item.title} updated.`);
    } catch (err) {
      if (isApprovalReplyNotFound(err)) {
        removeApprovalItem(item.id, item.workspace_id || selectedWorkspaceId);
        setApprovalReplyDrafts((current) => ({
          ...current,
          [item.id]: "",
        }));
        await refreshCurrentContext();
        setNotice(`${item.title} was already resolved.`);
        return;
      }
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
    workspace_id?: string;
    conversation_id?: string;
    run_id?: string;
    approval_item_id?: string;
    item_id?: string;
    title?: string;
    path?: string;
    kind?: string;
  }) {
    const targetWorkspaceId = String(result.workspace_id || "").trim();
    if (result.kind === "workspace" && targetWorkspaceId) {
      startTransition(() => {
        setSelectedWorkspaceId(targetWorkspaceId);
        setSelectedConversationId("");
        setSelectedRunId("");
        setActiveTab("overview");
      });
      return;
    }
    if (result.kind === "process") {
      startTransition(() => {
        if (targetWorkspaceId) {
          setSelectedWorkspaceId(targetWorkspaceId);
        }
        setSelectedConversationId("");
        setSelectedRunId("");
        setActiveTab("runs");
        setRunProcess(String(result.item_id || result.title || "").trim());
      });
      focusRunComposer();
      return;
    }
    if (result.kind === "account" || result.kind === "mcp_server") {
      const searchTerm = String(result.title || result.item_id || "").trim();
      startTransition(() => {
        if (targetWorkspaceId) {
          setSelectedWorkspaceId(targetWorkspaceId);
        }
        setSelectedConversationId("");
        setSelectedRunId("");
        setWorkspaceSearchQuery(searchTerm);
        setActiveTab("integrations");
      });
      setNotice(
        result.kind === "account"
          ? `Opened integrations for account ${String(result.title || result.item_id || "").trim()}.`
          : `Opened integrations for server ${String(result.title || result.item_id || "").trim()}.`,
      );
      return;
    }
    if ((result.kind === "artifact" || result.kind === "file") && result.path) {
      startTransition(() => {
        if (targetWorkspaceId) {
          setSelectedWorkspaceId(targetWorkspaceId);
        }
        setSelectedConversationId("");
        if (result.run_id) {
          setSelectedRunId(result.run_id);
        } else {
          setSelectedRunId("");
        }
        setActiveTab("files");
      });
      if (targetWorkspaceId) {
        queueWorkspaceFileOpen(targetWorkspaceId, result.path);
      }
      setNotice(
        result.kind === "file"
          ? `Opened file ${result.path}.`
          : `Opened context for artifact ${result.path}.`,
      );
      return;
    }
    if (result.conversation_id) {
      startTransition(() => {
        if (targetWorkspaceId) {
          setSelectedWorkspaceId(targetWorkspaceId);
        }
        setSelectedRunId("");
        setSelectedConversationId(result.conversation_id || "");
        setActiveTab("threads");
      });
      return;
    }
    if (result.run_id) {
      startTransition(() => {
        if (targetWorkspaceId) {
          setSelectedWorkspaceId(targetWorkspaceId);
        }
        setSelectedConversationId("");
        setSelectedRunId(result.run_id || "");
        setActiveTab("runs");
      });
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
