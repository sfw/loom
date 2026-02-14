"""Cowork session: conversation-first interactive execution.

The core interaction loop:
1. User sends a message
2. Model responds with text + optional tool calls
3. Tool calls are executed, results fed back
4. Repeat until model produces text-only response
5. Display response to user, wait for next input

The full conversation history is the context — no separate memory system,
no subtask decomposition, no planning phase.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path

from loom.cowork.approval import ApprovalDecision, ToolApprover
from loom.config import Config
from loom.models.base import ModelProvider, ModelResponse, StreamChunk, ToolCall
from loom.models.router import ModelRouter
from loom.tools.registry import ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ToolCallEvent:
    """Emitted when a tool call starts or completes."""

    name: str
    args: dict
    result: ToolResult | None = None
    elapsed_ms: int = 0


@dataclass
class CoworkTurn:
    """One full turn in the conversation (model response + tool calls)."""

    text: str = ""
    tool_calls: list[ToolCallEvent] = field(default_factory=list)
    tokens_used: int = 0
    model: str = ""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

# Tools that signal "ask the user" — the tool loop should pause and
# return to the user instead of continuing.
_USER_INTERACTION_TOOLS = frozenset({"ask_user"})

MAX_TOOL_ITERATIONS = 40


class CoworkSession:
    """Interactive cowork session.

    Maintains the full conversation history as a list[dict] in the
    OpenAI message format.  Each user message, assistant response, and
    tool result is preserved across the entire session.

    The session does NOT own the I/O — it yields events that the CLI
    or TUI renders.  This keeps the session testable and transport-agnostic.
    """

    def __init__(
        self,
        model: ModelProvider,
        tools: ToolRegistry,
        workspace: Path | None = None,
        system_prompt: str = "",
        max_context_messages: int = 200,
        approver: ToolApprover | None = None,
    ):
        self._model = model
        self._tools = tools
        self._workspace = workspace
        self._max_context = max_context_messages
        self._approver = approver

        self._messages: list[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    @property
    def messages(self) -> list[dict]:
        """Read-only view of conversation history."""
        return list(self._messages)

    @property
    def workspace(self) -> Path | None:
        return self._workspace

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(self, user_message: str) -> AsyncGenerator[CoworkTurn | ToolCallEvent | str, None]:
        """Send a user message and yield events as the model responds.

        Yields:
        - str: streamed text tokens (partial)
        - ToolCallEvent: tool call start/completion
        - CoworkTurn: final turn summary when the model finishes

        The caller should display streamed text inline, show tool calls
        as they happen, and use the final CoworkTurn for bookkeeping.
        """
        self._messages.append({"role": "user", "content": user_message})

        total_tokens = 0
        all_tool_events: list[ToolCallEvent] = []
        final_text = ""

        for _ in range(MAX_TOOL_ITERATIONS):
            # Call the model
            response = await self._model.complete(
                self._context_window(),
                tools=self._tools.all_schemas() or None,
            )
            total_tokens += response.usage.total_tokens

            # Accumulate text
            if response.text:
                final_text = response.text

            if response.has_tool_calls():
                # Append assistant message with tool calls
                self._messages.append({
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                })

                # Execute each tool call
                ask_user_pending = False
                for tc in response.tool_calls:
                    event = ToolCallEvent(name=tc.name, args=tc.arguments)
                    yield event  # signal: tool call starting

                    # Check approval before executing
                    if self._approver is not None:
                        decision = await self._approver.check(tc.name, tc.arguments)
                        if decision == ApprovalDecision.DENY:
                            result = ToolResult.fail(
                                f"Tool call '{tc.name}' denied by user."
                            )
                            event.result = result
                            event.elapsed_ms = 0
                            all_tool_events.append(event)
                            yield event
                            self._messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result.to_json(),
                            })
                            continue

                    start = time.monotonic()
                    result = await self._tools.execute(
                        tc.name, tc.arguments,
                        workspace=self._workspace,
                    )
                    event.result = result
                    event.elapsed_ms = int((time.monotonic() - start) * 1000)
                    all_tool_events.append(event)
                    yield event  # signal: tool call completed (now has result)

                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.to_json(),
                    })

                    if tc.name in _USER_INTERACTION_TOOLS:
                        ask_user_pending = True

                # If ask_user was called, break the tool loop and return
                # to the user for their answer
                if ask_user_pending:
                    break
            else:
                # Text-only response — model is done with this turn
                self._messages.append({
                    "role": "assistant",
                    "content": response.text or "",
                })
                break

        # Trim conversation if it's getting too long
        self._maybe_trim()

        yield CoworkTurn(
            text=final_text,
            tool_calls=all_tool_events,
            tokens_used=total_tokens,
            model=self._model.name,
        )

    async def send_streaming(
        self, user_message: str,
    ) -> AsyncGenerator[CoworkTurn | ToolCallEvent | str, None]:
        """Like send() but streams text tokens as they arrive.

        Yields str chunks for incremental display, ToolCallEvents for
        tool calls, and a final CoworkTurn.
        """
        self._messages.append({"role": "user", "content": user_message})

        total_tokens = 0
        all_tool_events: list[ToolCallEvent] = []
        final_text = ""

        for _ in range(MAX_TOOL_ITERATIONS):
            # Stream from the model
            text_parts: list[str] = []
            final_tool_calls: list[ToolCall] | None = None
            final_usage = None

            async for chunk in self._model.stream(
                self._context_window(),
                tools=self._tools.all_schemas() or None,
            ):
                if chunk.text:
                    text_parts.append(chunk.text)
                    yield chunk.text  # stream to display
                if chunk.tool_calls is not None:
                    final_tool_calls = chunk.tool_calls
                if chunk.usage is not None:
                    final_usage = chunk.usage

            from loom.models.base import TokenUsage
            response_text = "".join(text_parts)
            total_tokens += (final_usage or TokenUsage()).total_tokens

            if response_text:
                final_text = response_text

            if final_tool_calls:
                self._messages.append({
                    "role": "assistant",
                    "content": response_text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in final_tool_calls
                    ],
                })

                ask_user_pending = False
                for tc in final_tool_calls:
                    event = ToolCallEvent(name=tc.name, args=tc.arguments)
                    yield event

                    # Check approval before executing
                    if self._approver is not None:
                        decision = await self._approver.check(tc.name, tc.arguments)
                        if decision == ApprovalDecision.DENY:
                            result = ToolResult.fail(
                                f"Tool call '{tc.name}' denied by user."
                            )
                            event.result = result
                            event.elapsed_ms = 0
                            all_tool_events.append(event)
                            yield event
                            self._messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result.to_json(),
                            })
                            continue

                    start = time.monotonic()
                    result = await self._tools.execute(
                        tc.name, tc.arguments,
                        workspace=self._workspace,
                    )
                    event.result = result
                    event.elapsed_ms = int((time.monotonic() - start) * 1000)
                    all_tool_events.append(event)
                    yield event

                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.to_json(),
                    })

                    if tc.name in _USER_INTERACTION_TOOLS:
                        ask_user_pending = True

                if ask_user_pending:
                    break
            else:
                self._messages.append({
                    "role": "assistant",
                    "content": response_text or "",
                })
                break

        self._maybe_trim()

        yield CoworkTurn(
            text=final_text,
            tool_calls=all_tool_events,
            tokens_used=total_tokens,
            model=self._model.name,
        )

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def _context_window(self) -> list[dict]:
        """Return the messages to send to the model.

        Keeps system prompt + most recent messages within budget.
        """
        if len(self._messages) <= self._max_context:
            return list(self._messages)

        # Always keep system prompt if present
        system = []
        rest = self._messages
        if self._messages and self._messages[0]["role"] == "system":
            system = [self._messages[0]]
            rest = self._messages[1:]

        # Keep the most recent messages
        trimmed = rest[-(self._max_context - len(system)):]
        return system + trimmed

    def _maybe_trim(self) -> None:
        """Trim old messages if conversation exceeds 2x the context budget.

        This is a soft trim — we keep the system prompt and recent history.
        The _context_window() method handles the hard trim per-call.
        """
        limit = self._max_context * 2
        if len(self._messages) <= limit:
            return

        system = []
        if self._messages and self._messages[0]["role"] == "system":
            system = [self._messages[0]]

        keep = self._messages[-(self._max_context):]
        self._messages = system + keep


def build_cowork_system_prompt(workspace: Path | None = None) -> str:
    """Build the system prompt for cowork mode."""
    workspace_info = f"Workspace: {workspace}" if workspace else "No workspace set."

    return f"""\
You are a collaborative coding assistant working interactively with a developer.
You have access to tools for reading, writing, editing, and searching files,
running shell commands, executing git operations, and fetching web content.

{workspace_info}

GUIDELINES:
- Read files before editing them. Understand existing code before modifying it.
- Use targeted edits (edit_file) rather than rewriting entire files.
- Keep changes minimal and focused on what was requested.
- If something is unclear, use the ask_user tool to ask for clarification.
- Show your work: explain what you're doing and why.
- When you encounter errors, investigate and fix them rather than giving up.
- Do NOT modify files outside the workspace directory.
- Do NOT fabricate file contents. Always read first.

TOOL USAGE:
- Use glob_find to discover files by pattern (fast).
- Use ripgrep_search for content search (much faster than search_files).
- Use web_search to find documentation, solutions, or package information online.
- Use web_fetch to read a specific URL's content.
- Use shell_execute for running tests, builds, linters, etc.
- Use git_command for version control operations (including push).
- Use task_tracker to organize multi-step work and show progress.
- Use ask_user when you need the developer's input or decision."""
