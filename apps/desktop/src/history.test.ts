import { describe, expect, it } from "vitest";

import type {
  ConversationMessage,
  ConversationStreamEvent,
  NotificationEvent,
  RunTimelineEvent,
} from "./api";
import {
  conversationEventDetail,
  conversationEventPills,
  conversationEventTitle,
  firstMeaningfulString,
  isRunTimelineNoise,
  matchesWorkspaceSearch,
  normalizeConversationPrompt,
  notificationSummary,
  runTimelineDetail,
  runTimelinePills,
  runTimelineTitle,
  summarizeMessage,
} from "./history";

const runEvent: RunTimelineEvent = {
  id: 1,
  task_id: "task-1",
  run_id: "run-1",
  correlation_id: "corr-1",
  event_id: "evt-1",
  sequence: 3,
  timestamp: "2026-03-24T10:00:00",
  event_type: "tool_call_completed",
  source_component: "executor",
  schema_version: 1,
  data: {
    tool_name: "apply_patch",
    success: false,
    elapsed_ms: 42,
    sequence: 3,
    message: "Patch failed cleanly",
  },
};

const askUserEvent: ConversationStreamEvent = {
  id: 1,
  session_id: "session-1",
  seq: 4,
  event_type: "tool_call_completed",
  payload: {
    tool_name: "ask_user",
    success: true,
    question_payload: {
      question: "Which path should we take?",
      question_type: "single_choice",
      options: [
        { id: "a", label: "Safe path", description: "Lower risk" },
        { id: "b", label: "Fast path", description: "Higher risk" },
      ],
      allow_custom_response: true,
      min_selections: 1,
      max_selections: 1,
      context_note: "Need a decision before continuing",
      urgency: "high",
      default_option_id: "a",
      tool_call_id: "tool-1",
    },
  },
  payload_parse_error: false,
  created_at: "2026-03-24T10:00:00",
};

describe("history helpers", () => {
  it("finds the first meaningful nested string", () => {
    expect(
      firstMeaningfulString({
        meta: { empty: "" },
        payload: [{}, { error: "Patch conflict" }],
      }),
    ).toBe("Patch conflict");
  });

  it("formats run timeline titles, details, and pills", () => {
    expect(runTimelineTitle(runEvent)).toBe("apply_patch failed");
    expect(runTimelineDetail(runEvent)).toBe("Patch failed cleanly");
    expect(runTimelinePills(runEvent)).toEqual([
      "apply_patch",
      "seq 3",
      "42 ms",
      "failed",
    ]);
  });

  it("formats ask-user conversation events with prompt details", () => {
    expect(conversationEventTitle(askUserEvent)).toBe("Asked for guidance");
    expect(conversationEventDetail(askUserEvent)).toBe("Which path should we take?");
    expect(conversationEventPills(askUserEvent)).toEqual([
      "ask_user",
      "success",
      "2 options",
    ]);
  });

  it("sanitizes and expands turn-separator conversation stats", () => {
    const turnSeparatorEvent: ConversationStreamEvent = {
      ...askUserEvent,
      event_type: "turn_separator",
      payload: {
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
      },
    };

    expect(conversationEventTitle(turnSeparatorEvent)).toBe("Turn completed");
    expect(conversationEventDetail(turnSeparatorEvent)).toBe(
      "1 tool · 0 tokens · 12.3 tok/s · 1.5s latency · 2.6s total · ctx 64,000 tok · 22 ctx msg · 3 archived · recall-index · gpt-5.4",
    );
    expect(conversationEventPills(turnSeparatorEvent)).toEqual([
      "0 tokens",
      "1 tool",
    ]);
  });

  it("treats empty claim verification summaries as noise", () => {
    const claimSummary: RunTimelineEvent = {
      ...runEvent,
      event_type: "claim_verification_summary",
      data: {
        extracted: 0,
        supported: 0,
        partially_supported: 0,
        contradicted: 0,
        insufficient_evidence: 0,
        pruned: 0,
      },
    };

    expect(isRunTimelineNoise(claimSummary)).toBe(true);
  });

  it("shows retry-scheduled model events instead of hiding them as noise", () => {
    const retryEvent: RunTimelineEvent = {
      ...runEvent,
      event_type: "model_invocation",
      data: {
        subtask_id: "planning",
        model: "gpt-test",
        phase: "done",
        retry_scheduled: true,
        retry_resume_at_ms: 10000,
        retry_delay_seconds: 8,
        http_status: 429,
        model_error_code: "engine_overloaded_error",
      },
    };

    expect(isRunTimelineNoise(retryEvent)).toBe(false);
    expect(runTimelineTitle(retryEvent)).toBe("Retry scheduled for planning");
    expect(runTimelineDetail(retryEvent, 2000)).toBe(
      "gpt-test · retry in 8s · HTTP 429 · engine_overloaded_error",
    );
  });

  it("normalizes conversation prompts and drops invalid ones", () => {
    expect(normalizeConversationPrompt(null)).toBeNull();
    expect(
      normalizeConversationPrompt({
        question: "Proceed?",
        options: [{ id: "yes", label: "Yes" }, { id: "no", label: "No" }],
      }),
    ).toEqual(
      expect.objectContaining({
        question: "Proceed?",
        options: [
          { id: "yes", label: "Yes", description: "" },
          { id: "no", label: "No", description: "" },
        ],
      }),
    );
  });

  it("matches workspace search across mixed values", () => {
    expect(matchesWorkspaceSearch("auth", "API", { summary: "Auth findings" })).toBe(true);
    expect(matchesWorkspaceSearch("missing", "API", { summary: "Auth findings" })).toBe(false);
    expect(matchesWorkspaceSearch("", "anything")).toBe(true);
  });

  it("summarizes conversation messages and notifications", () => {
    const toolMessage: ConversationMessage = {
      id: 1,
      session_id: "session-1",
      turn_number: 1,
      role: "assistant",
      content: "",
      tool_calls: [],
      created_at: "2026-03-24T10:00:00",
      tool_name: "exec_command",
      tool_call_id: "tool-2",
      token_count: 0,
    };
    const notification: NotificationEvent = {
      id: "note-1",
      event_type: "approval_requested",
      workspace_id: "ws-1",
      workspace_path: "/tmp/workspace",
      workspace_display_name: "Workspace One",
      task_id: "",
      conversation_id: "",
      approval_id: "approval-1",
      kind: "conversation_approval",
      title: "Approval needed",
      summary: "",
      created_at: "2026-03-24T10:00:00",
      payload: {},
    };

    expect(summarizeMessage(toolMessage)).toBe("Tool call: exec_command");
    expect(notificationSummary(notification)).toBe("Approval Requested");
  });
});
