import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ThreadsTab from "./ThreadsTab";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  useApp: () => mockApp,
}));

function makeEvent(seq: number, eventType: string, payload: Record<string, unknown>) {
  return {
    id: seq,
    session_id: "conversation-1",
    seq,
    event_type: eventType,
    payload,
    payload_parse_error: false,
    created_at: `2026-03-27T00:${String(seq).padStart(2, "0")}:00Z`,
  };
}

describe("ThreadsTab", () => {
  beforeEach(() => {
    mockApp = {
      selectedConversationId: "conversation-1",
      overview: {
        recent_conversations: [
          {
            id: "conversation-1",
            title: "Thread Title",
            model_name: "kimi-k2.5",
            turn_count: 2,
            last_active_at: "2026-03-27T00:00:00Z",
            started_at: "2026-03-27T00:00:00Z",
            linked_run_ids: [],
          },
        ],
      },
      conversationDetail: {
        id: "conversation-1",
        title: "Thread Title",
        model_name: "kimi-k2.5",
        started_at: "2026-03-27T00:00:00Z",
      },
      loadingConversationDetail: false,
      conversationLoadError: "",
      conversationStatus: {
        conversation_id: "conversation-1",
        processing: false,
      },
      conversationStreaming: false,
      conversationIsProcessing: false,
      conversationAwaitingApproval: false,
      conversationAwaitingInput: false,
      pendingConversationApproval: null,
      pendingConversationPrompt: null,
      quickReplyOptions: [],
      conversationPhaseLabel: "Idle",
      visibleConversationMessages: [],
      visibleConversationEvents: [
        makeEvent(1, "user_message", { text: "hello" }),
        makeEvent(2, "assistant_thinking", { text: "step one", streaming: false }),
        makeEvent(3, "assistant_thinking", { text: " and step two", streaming: false }),
        makeEvent(4, "assistant_text", { text: "hi " }),
        makeEvent(5, "assistant_text", { text: "there" }),
        makeEvent(6, "tool_call_started", {
          tool_name: "ripgrep_search",
          tool_call_id: "call-1",
          args: { pattern: "TODO" },
        }),
        makeEvent(7, "tool_call_completed", {
          tool_name: "ripgrep_search",
          tool_call_id: "call-1",
          success: true,
          output: "2 matches",
          elapsed_ms: 18,
        }),
        makeEvent(8, "turn_separator", { tokens: 25, tool_count: 1 }),
      ],
      conversationComposerMessage: "",
      setConversationComposerMessage: vi.fn(),
      sendingConversationMessage: false,
      conversationInjectMessage: "",
      setConversationInjectMessage: vi.fn(),
      sendingConversationInject: false,
      handleSendConversationMessage: vi.fn(async () => {}),
      handleInjectConversationInstruction: vi.fn(async () => {}),
      handleResolveConversationApproval: vi.fn(async () => {}),
      handleQuickConversationReply: vi.fn(async () => {}),
      handleStopConversationTurn: vi.fn(async () => {}),
      streamingText: "",
      streamingThinking: "",
      streamingToolCalls: [],
      lastTurnStats: null,
      conversationHistoryQuery: "",
      setConversationHistoryQuery: vi.fn(),
      queuedMessages: [],
      editQueuedMessage: vi.fn(),
      cancelQueuedMessage: vi.fn(),
      setSelectedConversationId: vi.fn(),
      selectedWorkspaceId: "workspace-1",
      refreshWorkspaceSurface: vi.fn(async () => {}),
      setSelectedWorkspaceFilePath: vi.fn(),
      setActiveTab: vi.fn(),
      workspaceFilesByDirectory: {},
      hasOlderMessages: false,
      loadingOlderMessages: false,
      loadOlderMessages: vi.fn(async () => {}),
      retryConversationLoad: vi.fn(async () => {}),
      setError: vi.fn(),
      setNotice: vi.fn(),
    };
  });

  it("auto-opens the only thread in a workspace", async () => {
    mockApp.selectedConversationId = "";
    mockApp.conversationDetail = null;

    render(<ThreadsTab />);

    await waitFor(() => {
      expect(mockApp.setSelectedConversationId).toHaveBeenCalledWith("conversation-1");
    });
    expect(screen.getByText("Opening thread...")).toBeInTheDocument();
  });

  it("shows a thread chooser when the workspace has multiple threads", async () => {
    const user = userEvent.setup();
    mockApp.selectedConversationId = "";
    mockApp.conversationDetail = null;
    mockApp.overview = {
      recent_conversations: [
        {
          id: "conversation-1",
          title: "Tennis plan",
          model_name: "kimi-k2.5",
          turn_count: 3,
          last_active_at: "2026-03-27T00:00:00Z",
          started_at: "2026-03-27T00:00:00Z",
          linked_run_ids: [],
        },
        {
          id: "conversation-2",
          title: "Coach outreach",
          model_name: "kimi-k2.5",
          turn_count: 7,
          last_active_at: "2026-03-27T01:00:00Z",
          started_at: "2026-03-27T01:00:00Z",
          linked_run_ids: [],
        },
      ],
    };

    render(<ThreadsTab />);

    expect(screen.getByText("Pick a thread to continue in this workspace.")).toBeInTheDocument();
    expect(screen.getByText("Tennis plan")).toBeInTheDocument();
    expect(screen.getByText("Coach outreach")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Coach outreach/i }));

    expect(mockApp.setSelectedConversationId).toHaveBeenCalledWith("conversation-2");
  });

  it("clears and hides a stale thread when it does not belong to the active workspace", async () => {
    mockApp.selectedWorkspaceId = "workspace-2";
    mockApp.selectedConversationId = "conversation-1";
    mockApp.overview = {
      recent_conversations: [],
    };
    mockApp.conversationDetail = {
      id: "conversation-1",
      workspace_id: "workspace-1",
      title: "Thread Title",
      model_name: "kimi-k2.5",
      started_at: "2026-03-27T00:00:00Z",
    };

    render(<ThreadsTab />);

    await waitFor(() => {
      expect(mockApp.setSelectedConversationId).toHaveBeenCalledWith("");
    });
    expect(screen.getByText("No threads yet")).toBeInTheDocument();
    expect(screen.queryByText("Thread Title")).not.toBeInTheDocument();
  });

  it("renders replay events as the canonical thread transcript, including persisted reasoning", async () => {
    const user = userEvent.setup();
    mockApp.visibleConversationEvents = [
      makeEvent(1, "user_message", { text: "hello" }),
      makeEvent(2, "assistant_thinking", { text: "step one", streaming: true }),
      makeEvent(3, "assistant_thinking", { text: " and step two", streaming: true }),
      makeEvent(4, "assistant_text", { text: "hi " }),
      makeEvent(5, "assistant_text", { text: "there" }),
      makeEvent(6, "tool_call_started", {
        tool_name: "ripgrep_search",
        tool_call_id: "call-1",
        args: { pattern: "TODO" },
      }),
      makeEvent(7, "tool_call_completed", {
        tool_name: "ripgrep_search",
        tool_call_id: "call-1",
        success: true,
        output: "2 matches",
        elapsed_ms: 18,
      }),
      makeEvent(8, "turn_separator", { tokens: 25, tool_count: 1 }),
    ];
    render(<ThreadsTab />);

    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("hi there")).toBeInTheDocument();
    expect(screen.getByText("ripgrep_search")).toBeInTheDocument();
    expect(screen.queryByText("2 matches")).not.toBeInTheDocument();
    expect(screen.getByText("25 tokens")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /ripgrep_search/i }));
    expect(screen.getByText("Tool Call Spec")).toBeInTheDocument();
    expect(screen.getByText(/"pattern": "TODO"/)).toBeInTheDocument();
    expect(screen.queryByText("2 matches")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Reasoning/i }));

    expect(screen.getByText("step one and step two")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Thinking\.\.\./i })).not.toBeInTheDocument();
  });

  it("marks only the trailing thinking block as live for an active turn", () => {
    mockApp.conversationStatus = {
      conversation_id: "conversation-1",
      processing: true,
    };
    mockApp.conversationStreaming = true;
    mockApp.conversationIsProcessing = true;
    mockApp.conversationPhaseLabel = "Running";
    mockApp.visibleConversationEvents = [
      makeEvent(1, "user_message", { text: "hello" }),
      makeEvent(2, "assistant_thinking", { text: "earlier reasoning", streaming: true }),
      makeEvent(3, "assistant_text", { text: "partial answer" }),
      makeEvent(4, "assistant_thinking", { text: "latest reasoning", streaming: true }),
    ];

    render(<ThreadsTab />);

    expect(screen.getByRole("button", { name: /^Reasoning$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Thinking\.\.\./i })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /^Thinking\.\.\./i })).toHaveLength(1);
    expect(screen.queryByText("Processing...")).not.toBeInTheDocument();
  });

  it("renders optimistic outgoing user bubbles immediately with a sending state", () => {
    mockApp.visibleConversationEvents = [
      {
        ...makeEvent(1, "user_message", { text: "my toss is a big problem" }),
        _optimistic: true,
      },
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("my toss is a big problem")).toBeInTheDocument();
    expect(screen.getByText("Sending...")).toBeInTheDocument();
  });

  it("anchors pending approval and ask-user actions inline in the transcript", () => {
    mockApp.conversationStatus = {
      conversation_id: "conversation-1",
      processing: false,
      awaiting_approval: true,
      pending_approval: {
        approval_id: "approval-1",
        tool_name: "write_file",
        args: { path: "notes.md" },
        risk_info: { impact_preview: "Will modify notes.md" },
      },
      awaiting_user_input: true,
      pending_prompt: {
        question: "Which direction should I take?",
        question_type: "single_choice",
        options: [{ id: "a", label: "Option A" }],
        context_note: "Choose one to continue",
        allow_custom_response: true,
        min_selections: 0,
        max_selections: 1,
        urgency: "normal",
        default_option_id: "",
        tool_call_id: "ask-1",
      },
    };
    mockApp.conversationAwaitingApproval = true;
    mockApp.pendingConversationApproval = mockApp.conversationStatus.pending_approval;
    mockApp.conversationAwaitingInput = true;
    mockApp.pendingConversationPrompt = mockApp.conversationStatus.pending_prompt;
    mockApp.quickReplyOptions = [{ id: "a", label: "Option A" }];
    mockApp.visibleConversationEvents = [
      makeEvent(1, "approval_requested", {
        approval_id: "approval-1",
        tool_name: "write_file",
        risk_info: { impact_preview: "Will modify notes.md" },
      }),
      makeEvent(2, "tool_call_completed", {
        tool_name: "ask_user",
        tool_call_id: "ask-1",
        success: true,
        question_payload: {
          question: "Which direction should I take?",
          question_type: "single_choice",
          options: [{ id: "a", label: "Option A" }],
          context_note: "Choose one to continue",
        },
      }),
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("Approval Required")).toBeInTheDocument();
    expect(screen.getAllByText("Will modify notes.md")).toHaveLength(2);
    expect(screen.getByText("Input Requested")).toBeInTheDocument();
    expect(screen.getByText("Which direction should I take?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Option A" })).toBeInTheDocument();
  });

  it("shows a recoverable error state when thread detail fails to load", async () => {
    const user = userEvent.setup();
    mockApp.conversationDetail = null;
    mockApp.loadingConversationDetail = false;
    mockApp.conversationLoadError = "404 Not Found";

    render(<ThreadsTab />);

    expect(screen.getByText("Couldn't open this thread")).toBeInTheDocument();
    expect(screen.getByText("404 Not Found")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(mockApp.retryConversationLoad).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "Back to Threads" }));
    expect(mockApp.setSelectedConversationId).toHaveBeenCalledWith("");
  });

  it("hydrates arrow-key prompt history from persisted thread input history", async () => {
    const user = userEvent.setup();
    mockApp.conversationDetail = {
      ...mockApp.conversationDetail,
      session_state: {
        ui_state: {
          input_history: {
            items: ["first prompt", "second prompt", "third prompt"],
          },
        },
      },
    };

    render(<ThreadsTab />);

    const composer = screen.getByPlaceholderText("Send a message...");
    await user.click(composer);
    await user.keyboard("{ArrowUp}");

    expect(mockApp.setConversationComposerMessage).toHaveBeenCalledWith("third prompt");
  });

  it("does not auto-backfill older history when the initial page is short", async () => {
    const loadOlderMessages = vi.fn(async () => {});
    mockApp.hasOlderMessages = true;
    mockApp.loadOlderMessages = loadOlderMessages;

    render(<ThreadsTab />);

    await waitFor(() => {
      expect(loadOlderMessages).not.toHaveBeenCalled();
    });
  });
});
