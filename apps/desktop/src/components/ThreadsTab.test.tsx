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
      conversationDetail: {
        id: "conversation-1",
        title: "Thread Title",
        model_name: "kimi-k2.5",
        started_at: "2026-03-27T00:00:00Z",
      },
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
      setError: vi.fn(),
      setNotice: vi.fn(),
    };
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
    expect(screen.getByText("2 matches")).toBeInTheDocument();
    expect(screen.getByText("25 tokens")).toBeInTheDocument();

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
    expect(screen.getAllByText("Which direction should I take?")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Option A" })).toBeInTheDocument();
  });

  it("backfills older history when the initial page does not overflow the transcript", async () => {
    const loadOlderMessages = vi.fn(async () => {});
    mockApp.hasOlderMessages = true;
    mockApp.loadOlderMessages = loadOlderMessages;

    const clientHeight = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight");
    const scrollHeight = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollHeight");
    const raf = vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    const caf = vi.spyOn(window, "cancelAnimationFrame").mockImplementation(() => {});

    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      get: () => 600,
    });
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      get: () => 420,
    });

    try {
      render(<ThreadsTab />);

      await waitFor(() => {
        expect(loadOlderMessages).toHaveBeenCalledTimes(1);
      });
    } finally {
      raf.mockRestore();
      caf.mockRestore();
      if (clientHeight) {
        Object.defineProperty(HTMLElement.prototype, "clientHeight", clientHeight);
      } else {
        delete (HTMLElement.prototype as any).clientHeight;
      }
      if (scrollHeight) {
        Object.defineProperty(HTMLElement.prototype, "scrollHeight", scrollHeight);
      } else {
        delete (HTMLElement.prototype as any).scrollHeight;
      }
    }
  });
});
