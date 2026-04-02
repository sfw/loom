import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ThreadsTab from "./ThreadsTab";
import { deleteConversation } from "@/api";

let mockApp: any;

vi.mock("@/context/AppContext", () => ({
  shallowEqual: (left: unknown, right: unknown) => left === right,
  useApp: () => mockApp,
  useAppActions: () => mockApp,
  useAppSelector: (selector: (state: any) => unknown) => selector(mockApp),
}));

vi.mock("@/api", () => ({
  deleteConversation: vi.fn(),
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
    vi.mocked(deleteConversation).mockResolvedValue(undefined as never);
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
      removeConversationSummary: vi.fn(),
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

  it("keeps a newly selected thread open while its detail is still loading", async () => {
    mockApp.selectedConversationId = "conversation-1";
    mockApp.overview = {
      recent_conversations: [],
    };
    mockApp.conversationDetail = null;
    mockApp.loadingConversationDetail = true;

    render(<ThreadsTab />);

    expect(screen.getByText("Opening thread...")).toBeInTheDocument();
    expect(mockApp.setSelectedConversationId).not.toHaveBeenCalledWith("");
  });

  it("deletes the selected thread by patching local workspace state instead of refreshing the workspace", async () => {
    const user = userEvent.setup();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<ThreadsTab />);

    await user.click(screen.getByRole("button", { name: /delete/i }));

    await waitFor(() => {
      expect(deleteConversation).toHaveBeenCalledWith("conversation-1");
    });
    expect(mockApp.removeConversationSummary).toHaveBeenCalledWith("conversation-1", "workspace-1");
    expect(mockApp.setSelectedConversationId).toHaveBeenCalledWith("");
    expect(mockApp.setNotice).toHaveBeenCalledWith("Thread deleted.");
    expect(mockApp.refreshWorkspaceSurface).not.toHaveBeenCalled();

    confirmSpy.mockRestore();
  });

  it("renders replay events as the canonical thread transcript without persisted reasoning rows", async () => {
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
    expect(screen.queryByRole("button", { name: /Reasoning/i })).not.toBeInTheDocument();
    expect(screen.queryByText("step one and step two")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /ripgrep_search/i }));
    expect(screen.getByText("Tool Call Spec")).toBeInTheDocument();
    expect(screen.getByText(/"pattern": "TODO"/)).toBeInTheDocument();
    expect(screen.queryByText("2 matches")).not.toBeInTheDocument();
  });

  it("renders detailed turn-separator stats without surfacing NaN values", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(1, "assistant_text", { text: "All set." }),
      makeEvent(2, "turn_separator", {
        tokens: "not-a-number",
        tool_count: "1",
        tokens_per_second: "12.34",
        latency_ms: "1500",
        total_time_ms: "2600",
        context_tokens: "64000",
        context_messages: "22",
        omitted_messages: "3",
        recall_index_used: "true",
        model: "gpt-5.4",
      }),
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("0 tokens")).toBeInTheDocument();
    expect(screen.getByText("1 tool")).toBeInTheDocument();
    expect(screen.getByText("12.3 tok/s")).toBeInTheDocument();
    expect(screen.getByText("1.5s latency")).toBeInTheDocument();
    expect(screen.getByText("2.6s total")).toBeInTheDocument();
    expect(screen.getByText("ctx 64,000 tok")).toBeInTheDocument();
    expect(screen.getByText("22 ctx msg")).toBeInTheDocument();
    expect(screen.getByText("3 archived")).toBeInTheDocument();
    expect(screen.getByText("recall-index")).toBeInTheDocument();
    expect(screen.getByText("gpt-5.4")).toBeInTheDocument();
    expect(screen.queryByText(/NaN tokens/i)).not.toBeInTheDocument();
  });

  it("renders markdown correctly when assistant text arrives in tiny streamed chunks", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(1, "assistant_text", { text: "# Heading 1" }),
      makeEvent(2, "assistant_text", { text: "\n" }),
      makeEvent(3, "assistant_text", { text: "## Heading 2" }),
      makeEvent(4, "assistant_text", { text: "\n\n" }),
      makeEvent(5, "assistant_text", { text: "- Item 1" }),
      makeEvent(6, "assistant_text", { text: "\n" }),
      makeEvent(7, "assistant_text", { text: "  " }),
      makeEvent(8, "assistant_text", { text: "- Nested item" }),
      makeEvent(9, "assistant_text", { text: "\n\n" }),
      makeEvent(10, "assistant_text", { text: "```js" }),
      makeEvent(11, "assistant_text", { text: "\n" }),
      makeEvent(12, "assistant_text", { text: "console.log('hi');" }),
      makeEvent(13, "assistant_text", { text: "\n" }),
      makeEvent(14, "assistant_text", { text: "```" }),
      makeEvent(15, "assistant_text", { text: "\n\n" }),
      makeEvent(16, "assistant_text", { text: "| Left | Right |" }),
      makeEvent(17, "assistant_text", { text: "\n" }),
      makeEvent(18, "assistant_text", { text: "| --- | --- |" }),
      makeEvent(19, "assistant_text", { text: "\n" }),
      makeEvent(20, "assistant_text", { text: "| A | B |" }),
    ];

    render(<ThreadsTab />);

    expect(screen.getByRole("heading", { level: 1, name: "Heading 1" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Heading 2" })).toBeInTheDocument();
    expect(screen.getByText("Item 1")).toBeInTheDocument();
    expect(screen.getByText("Nested item")).toBeInTheDocument();
    expect(screen.getByText("console.log('hi');")).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("falls back to persisted messages when the initial event slice has no visible transcript rows", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(1, "assistant_thinking", { text: "hidden", streaming: true }),
      makeEvent(2, "assistant_thinking", { text: "still hidden", streaming: true }),
    ];
    mockApp.visibleConversationMessages = [
      {
        id: 1,
        session_id: "conversation-1",
        turn_number: 1,
        role: "user",
        content: "hello",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 1,
        created_at: "2026-03-27T00:00:00Z",
      },
      {
        id: 2,
        session_id: "conversation-1",
        turn_number: 2,
        role: "assistant",
        content: "hi there",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 2,
        created_at: "2026-03-27T00:01:00Z",
      },
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("hi there")).toBeInTheDocument();
    expect(screen.queryByText("No messages yet. Send a message to get started.")).not.toBeInTheDocument();
  });

  it("prefers persisted settled transcript text when the replay slice starts mid-answer", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(2, "tool_call_started", {
        tool_name: "ripgrep_search",
        tool_call_id: "call-1",
        args: { pattern: "TODO" },
      }),
      makeEvent(3, "tool_call_completed", {
        tool_name: "ripgrep_search",
        tool_call_id: "call-1",
        success: true,
        output: "2 matches",
        elapsed_ms: 18,
      }),
      makeEvent(4, "assistant_text", { text: "- Nested item" }),
      makeEvent(5, "turn_separator", { tokens: 25, tool_count: 1 }),
    ];
    mockApp.visibleConversationMessages = [
      {
        id: 1,
        session_id: "conversation-1",
        turn_number: 1,
        role: "user",
        content: "show me the full answer",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 4,
        created_at: "2026-03-27T00:00:00Z",
      },
      {
        id: 2,
        session_id: "conversation-1",
        turn_number: 2,
        role: "assistant",
        content: "Let me inspect the workspace first.",
        tool_calls: [
          {
            id: "call-1",
            type: "function",
            function: {
              name: "ripgrep_search",
              arguments: "{\"pattern\":\"TODO\"}",
            },
          },
        ],
        tool_call_id: null,
        tool_name: null,
        token_count: 5,
        created_at: "2026-03-27T00:03:00Z",
      },
      {
        id: 3,
        session_id: "conversation-1",
        turn_number: 3,
        role: "tool",
        content: "{\"success\":true,\"output\":\"2 matches\"}",
        tool_calls: [],
        tool_call_id: "call-1",
        tool_name: "ripgrep_search",
        token_count: 4,
        created_at: "2026-03-27T00:04:00Z",
      },
      {
        id: 4,
        session_id: "conversation-1",
        turn_number: 4,
        role: "assistant",
        content: "# Full answer\n\n- Item 1\n  - Nested item\n\nClosing note.",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 12,
        created_at: "2026-03-27T00:05:00Z",
      },
    ];

    render(<ThreadsTab />);

    expect(screen.getByRole("heading", { level: 1, name: "Full answer" })).toBeInTheDocument();
    expect(screen.getByText("Item 1")).toBeInTheDocument();
    expect(screen.getByText("Nested item")).toBeInTheDocument();
    expect(screen.getByText("Closing note.")).toBeInTheDocument();
    expect(screen.getByText("ripgrep_search")).toBeInTheDocument();
  });

  it("keeps showing the live settled turn until persisted messages catch up", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(1, "user_message", { text: "hello" }),
      makeEvent(2, "assistant_text", { text: "Earlier reply" }),
      makeEvent(3, "user_message", { text: "tell me about loom processes" }),
      makeEvent(4, "assistant_text", { text: "Checking your workspace...Here is how Loom works." }),
      makeEvent(5, "tool_call_started", {
        tool_name: "list_directory",
        tool_call_id: "tool-1",
        args: { path: "." },
      }),
      makeEvent(6, "tool_call_completed", {
        tool_name: "list_directory",
        tool_call_id: "tool-1",
        success: true,
      }),
      makeEvent(7, "assistant_text", { text: "Here is how Loom works." }),
      makeEvent(8, "turn_separator", { tokens: 42, tool_count: 1 }),
    ];
    mockApp.visibleConversationMessages = [
      {
        id: 1,
        session_id: "conversation-1",
        turn_number: 1,
        role: "user",
        content: "hello",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 1,
        created_at: "2026-03-27T00:00:00Z",
      },
      {
        id: 2,
        session_id: "conversation-1",
        turn_number: 2,
        role: "assistant",
        content: "Earlier reply",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 2,
        created_at: "2026-03-27T00:01:00Z",
      },
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("Checking your workspace...Here is how Loom works.")).toBeInTheDocument();
    expect(screen.getByText("list_directory")).toBeInTheDocument();
  });

  it("keeps settled token separators under the matching transcript turns", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(1, "turn_separator", { tokens: 42, tool_count: 0 }),
    ];
    mockApp.visibleConversationMessages = [
      {
        id: 1,
        session_id: "conversation-1",
        turn_number: 1,
        role: "user",
        content: "First prompt",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 1,
        created_at: "2026-03-27T12:00:00Z",
      },
      {
        id: 2,
        session_id: "conversation-1",
        turn_number: 2,
        role: "assistant",
        content: "First reply",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 2,
        created_at: "2026-03-27T12:00:01Z",
      },
      {
        id: 3,
        session_id: "conversation-1",
        turn_number: 3,
        role: "user",
        content: "Second prompt",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 1,
        created_at: "2026-03-27T12:00:02Z",
      },
      {
        id: 4,
        session_id: "conversation-1",
        turn_number: 4,
        role: "assistant",
        content: "Second reply",
        tool_calls: [],
        tool_call_id: null,
        tool_name: null,
        token_count: 2,
        created_at: "2026-03-27T12:00:03Z",
      },
    ];

    const { container } = render(<ThreadsTab />);
    const transcriptText = container.textContent ?? "";

    expect(transcriptText.indexOf("42 tokens")).toBeGreaterThan(transcriptText.indexOf("First prompt"));
    expect(transcriptText.indexOf("42 tokens")).toBeGreaterThan(transcriptText.indexOf("Second reply"));
  });

  it("shows only a lightweight live thinking indicator during an active turn", () => {
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

    expect(screen.getByText("Thinking...")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Reasoning/i })).not.toBeInTheDocument();
    expect(screen.queryByText("earlier reasoning")).not.toBeInTheDocument();
    expect(screen.queryByText("latest reasoning")).not.toBeInTheDocument();
  });

  it("renders a live assistant draft when stream text is ahead of the transcript", () => {
    mockApp.conversationStatus = {
      conversation_id: "conversation-1",
      processing: true,
    };
    mockApp.conversationStreaming = true;
    mockApp.conversationIsProcessing = true;
    mockApp.conversationPhaseLabel = "Running";
    mockApp.visibleConversationEvents = [
      makeEvent(1, "user_message", { text: "hello" }),
    ];
    mockApp.streamingText = "Working through the numbers now";

    render(<ThreadsTab />);

    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(screen.getByText("Working through the numbers now")).toBeInTheDocument();
    expect(screen.queryByText("Thinking...")).not.toBeInTheDocument();
  });

  it("renders optimistic outgoing user bubbles immediately with a sending state", () => {
    mockApp.visibleConversationEvents = [
      {
        ...makeEvent(1, "user_message", { text: "my toss is a big problem" }),
        _optimistic: true,
        _delivery_state: "sending",
      },
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("my toss is a big problem")).toBeInTheDocument();
    expect(screen.getByText("Sending...")).toBeInTheDocument();
  });

  it("hides the sending badge once an optimistic message has been accepted", () => {
    mockApp.conversationIsProcessing = true;
    mockApp.conversationStatus = {
      conversation_id: "conversation-1",
      processing: true,
    };
    mockApp.visibleConversationEvents = [
      {
        ...makeEvent(1, "user_message", { text: "please continue" }),
        _optimistic: true,
        _delivery_state: "accepted",
      },
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("please continue")).toBeInTheDocument();
    expect(screen.queryByText("Sending...")).not.toBeInTheDocument();
  });

  it("marks timed out optimistic messages explicitly", () => {
    mockApp.visibleConversationEvents = [
      {
        ...makeEvent(1, "user_message", { text: "are you there?" }),
        _optimistic: true,
        _delivery_state: "failed",
      },
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("are you there?")).toBeInTheDocument();
    expect(screen.getByText("Timed out")).toBeInTheDocument();
  });

  it("never renders assistant thinking rows inside the transcript", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(1, "assistant_thinking", { text: "step one", streaming: false }),
      makeEvent(2, "assistant_thinking", { text: "step two", streaming: false }),
    ];

    render(<ThreadsTab />);

    expect(screen.queryByText(/^assistant thinking$/i)).not.toBeInTheDocument();
    expect(screen.queryByText("step one")).not.toBeInTheDocument();
    expect(screen.queryByText("step two")).not.toBeInTheDocument();
  });

  it("shows web tool args in collapsed tool cards", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(1, "tool_call_started", {
        tool_name: "web_search",
        tool_call_id: "web-search-1",
        args: { query: "python docs" },
      }),
      makeEvent(2, "tool_call_completed", {
        tool_name: "web_search",
        tool_call_id: "web-search-1",
        success: true,
        elapsed_ms: 100,
      }),
      makeEvent(3, "tool_call_started", {
        tool_name: "web_fetch",
        tool_call_id: "web-fetch-1",
        args: {
          url: "https://example.com/docs",
          query: "auth token refresh",
        },
      }),
      makeEvent(4, "tool_call_completed", {
        tool_name: "web_fetch",
        tool_call_id: "web-fetch-1",
        success: true,
        elapsed_ms: 200,
      }),
    ];

    render(<ThreadsTab />);

    expect(screen.getByText("python docs")).toBeInTheDocument();
    expect(screen.getByText("example.com/docs · auth token refresh")).toBeInTheDocument();
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
      makeEvent(1, "tool_call_started", {
        tool_name: "write_file",
        tool_call_id: "write-1",
        args: { path: "notes.md" },
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
    expect(screen.getByText("Will modify notes.md")).toBeInTheDocument();
    expect(screen.getByText("Input Requested")).toBeInTheDocument();
    expect(screen.getByText("Which direction should I take?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Option A" })).toBeInTheDocument();
  });

  it("does not leave historical approval-needed cards in the transcript after approval is resolved", () => {
    mockApp.visibleConversationEvents = [
      makeEvent(1, "tool_call_started", {
        tool_name: "shell_execute",
        tool_call_id: "shell-1",
        args: { command: "curl https://example.com" },
      }),
      makeEvent(2, "approval_requested", {
        approval_id: "approval-1",
        tool_name: "shell_execute",
      }),
      makeEvent(3, "approval_resolved", {
        approval_id: "approval-1",
        tool_name: "shell_execute",
        decision: "approve",
      }),
      makeEvent(4, "tool_call_completed", {
        tool_name: "shell_execute",
        tool_call_id: "shell-1",
        success: true,
      }),
    ];

    render(<ThreadsTab />);

    expect(screen.queryByText(/Approval needed for shell_execute/i)).not.toBeInTheDocument();
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

  it("archives older transcript rows behind an explicit reveal control", async () => {
    const user = userEvent.setup();
    mockApp.visibleConversationEvents = Array.from({ length: 260 }, (_, index) =>
      makeEvent(index + 1, "turn_separator", {
        tokens: index + 1,
        tool_count: 0,
      })
    );

    render(<ThreadsTab />);

    expect(screen.getByText(/older transcript rows archived/i)).toBeInTheDocument();
    const revealButton = screen.getByRole("button", { name: /Show 40 older rows/i });
    expect(revealButton).toBeInTheDocument();

    await user.click(revealButton);

    await waitFor(() => {
      expect(screen.queryByText(/older transcript rows archived/i)).not.toBeInTheDocument();
    });
  });

  it("keeps very large transcripts bounded to an archived render window", () => {
    mockApp.visibleConversationEvents = Array.from({ length: 12000 }, (_, index) =>
      makeEvent(index + 1, "turn_separator", {
        tokens: index + 1,
        tool_count: 0,
      })
    );

    render(<ThreadsTab />);

    expect(screen.getByText("11,780 older transcript rows archived")).toBeInTheDocument();
    expect(screen.getByText("11,781 tokens")).toBeInTheDocument();
    expect(screen.queryByText("1 tokens")).not.toBeInTheDocument();
  });
});
