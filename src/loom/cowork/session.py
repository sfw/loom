"""Cowork session: conversation-first interactive execution.

The core interaction loop:
1. User sends a message
2. Model responds with text + optional tool calls
3. Tool calls are executed, results fed back
4. Repeat until model produces text-only response
5. Display response to user, wait for next input

All conversation turns are persisted to SQLite (write-through).
Session state is maintained as always-in-context structured metadata.
The model can recall past conversation via the conversation_recall tool.
Complex work can be delegated to the task orchestrator via delegate_task.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loom.cowork.approval import ApprovalDecision, ToolApprover
from loom.cowork.session_state import SessionState, extract_state_from_tool_events
from loom.models.base import ModelProvider, ToolCall
from loom.tools.registry import ToolRegistry, ToolResult

if TYPE_CHECKING:
    from loom.state.conversation_store import ConversationStore

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

# Phrases that suggest the user is referencing earlier context.
_DANGLING_REF_INDICATORS = [
    "like we discussed", "as before", "remember when",
    "go back to", "that file", "that error", "that function",
    "what we did", "earlier", "the previous", "as I said",
    "like I mentioned", "that thing", "we already",
]

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token per 4 characters."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_message_tokens(msg: dict) -> int:
    """Estimate tokens for a full message dict."""
    total = _estimate_tokens(msg.get("content") or "")
    if msg.get("tool_calls"):
        total += _estimate_tokens(json.dumps(msg["tool_calls"]))
    return max(1, total)


class CoworkSession:
    """Interactive cowork session.

    Maintains the full conversation history as a list[dict] in the
    OpenAI message format.  Each user message, assistant response, and
    tool result is preserved across the entire session.

    All messages are persisted to SQLite via write-through.  The in-memory
    list is a hot cache; the ConversationStore holds everything.

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
        store: ConversationStore | None = None,
        session_id: str = "",
        session_state: SessionState | None = None,
        max_context_tokens: int = 180_000,
    ):
        self._model = model
        self._tools = tools
        self._workspace = workspace
        self._max_context = max_context_messages
        self._max_context_tokens = max_context_tokens
        self._approver = approver
        self._total_tokens = 0
        self._turn_counter = 0
        self._message_counter = 0  # monotonic per-message counter for DB persistence

        # Persistence
        self._store = store
        self._session_id = session_id

        # Session state (Layer 1 for cowork)
        self._session_state = session_state or SessionState(
            workspace=str(workspace) if workspace else "",
            model_name=model.name,
            session_id=session_id,
        )
        self._static_system_prompt = system_prompt

        self._messages: list[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": self._build_system_content()})

    @property
    def messages(self) -> list[dict]:
        """Read-only view of conversation history."""
        return list(self._messages)

    @property
    def workspace(self) -> Path | None:
        return self._workspace

    @property
    def total_tokens(self) -> int:
        """Cumulative token count across all turns in this session."""
        return self._total_tokens

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_state(self) -> SessionState:
        return self._session_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self, user_message: str,
    ) -> AsyncGenerator[CoworkTurn | ToolCallEvent | str, None]:
        """Send a user message and yield events as the model responds.

        Yields:
        - str: streamed text tokens (partial)
        - ToolCallEvent: tool call start/completion
        - CoworkTurn: final turn summary when the model finishes

        The caller should display streamed text inline, show tool calls
        as they happen, and use the final CoworkTurn for bookkeeping.
        """
        self._turn_counter += 1
        self._session_state.set_focus(user_message.split("\n")[0])

        # Possibly inject a recall hint
        hint = self._maybe_recall_hint(user_message)

        self._messages.append({"role": "user", "content": user_message})
        await self._persist_turn("user", content=user_message)

        if hint:
            self._messages.append({"role": "system", "content": hint})
            await self._persist_turn("system", content=hint)

        total_tokens = 0
        all_tool_events: list[ToolCallEvent] = []
        text_parts: list[str] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            # Call the model
            response = await self._model.complete(
                self._context_window(),
                tools=self._tools.all_schemas() or None,
            )
            total_tokens += response.usage.total_tokens

            # Accumulate text across iterations
            if response.text:
                text_parts.append(response.text)

            if response.has_tool_calls():
                tc_dicts = self._tool_calls_to_dicts(response.tool_calls)
                self._messages.append({
                    "role": "assistant",
                    "content": response.text or None,
                    "tool_calls": tc_dicts,
                })
                await self._persist_turn(
                    "assistant",
                    content=response.text,
                    tool_calls=tc_dicts,
                )

                ask_user_pending = False
                for tc in response.tool_calls:
                    event = ToolCallEvent(name=tc.name, args=tc.arguments)
                    yield event  # signal: tool call starting

                    # Check approval
                    if self._approver is not None:
                        decision = await self._approver.check(tc.name, tc.arguments)
                        if decision == ApprovalDecision.DENY:
                            result = ToolResult.fail(f"Tool call '{tc.name}' denied by user.")
                            event.result = result
                            event.elapsed_ms = 0
                            all_tool_events.append(event)
                            yield event
                            await self._append_tool_result(tc.id, tc.name, result)
                            continue

                    start = time.monotonic()
                    result = await self._tools.execute(
                        tc.name, tc.arguments, workspace=self._workspace,
                    )
                    event.result = result
                    event.elapsed_ms = int((time.monotonic() - start) * 1000)
                    all_tool_events.append(event)
                    yield event

                    await self._append_tool_result(tc.id, tc.name, result)

                    if tc.name in _USER_INTERACTION_TOOLS:
                        ask_user_pending = True

                if ask_user_pending:
                    break
            else:
                self._messages.append({"role": "assistant", "content": response.text or ""})
                await self._persist_turn("assistant", content=response.text or "")
                break

        # Post-turn bookkeeping
        self._maybe_trim()
        self._total_tokens += total_tokens
        self._session_state.turn_count = self._turn_counter
        self._session_state.total_tokens = self._total_tokens
        self._update_system_message()

        # Extract session state from tool events
        extract_state_from_tool_events(
            self._session_state, self._turn_counter, all_tool_events,
        )

        # Persist session metadata
        await self._persist_session_metadata()

        yield CoworkTurn(
            text="\n\n".join(text_parts),
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
        self._turn_counter += 1
        self._session_state.set_focus(user_message.split("\n")[0])

        hint = self._maybe_recall_hint(user_message)

        self._messages.append({"role": "user", "content": user_message})
        await self._persist_turn("user", content=user_message)

        if hint:
            self._messages.append({"role": "system", "content": hint})
            await self._persist_turn("system", content=hint)

        total_tokens = 0
        all_tool_events: list[ToolCallEvent] = []
        all_text_parts: list[str] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            iter_text_parts: list[str] = []
            final_tool_calls: list[ToolCall] | None = None
            final_usage = None

            async for chunk in self._model.stream(
                self._context_window(),
                tools=self._tools.all_schemas() or None,
            ):
                if chunk.text:
                    iter_text_parts.append(chunk.text)
                    yield chunk.text
                if chunk.tool_calls is not None:
                    final_tool_calls = chunk.tool_calls
                if chunk.usage is not None:
                    final_usage = chunk.usage

            from loom.models.base import TokenUsage
            response_text = "".join(iter_text_parts)
            total_tokens += (final_usage or TokenUsage()).total_tokens

            if response_text:
                all_text_parts.append(response_text)

            if final_tool_calls:
                tc_dicts = self._tool_calls_to_dicts(final_tool_calls)
                self._messages.append({
                    "role": "assistant",
                    "content": response_text or None,
                    "tool_calls": tc_dicts,
                })
                await self._persist_turn(
                    "assistant",
                    content=response_text,
                    tool_calls=tc_dicts,
                )

                ask_user_pending = False
                for tc in final_tool_calls:
                    event = ToolCallEvent(name=tc.name, args=tc.arguments)
                    yield event

                    if self._approver is not None:
                        decision = await self._approver.check(tc.name, tc.arguments)
                        if decision == ApprovalDecision.DENY:
                            result = ToolResult.fail(f"Tool call '{tc.name}' denied by user.")
                            event.result = result
                            event.elapsed_ms = 0
                            all_tool_events.append(event)
                            yield event
                            await self._append_tool_result(tc.id, tc.name, result)
                            continue

                    start = time.monotonic()
                    result = await self._tools.execute(
                        tc.name, tc.arguments, workspace=self._workspace,
                    )
                    event.result = result
                    event.elapsed_ms = int((time.monotonic() - start) * 1000)
                    all_tool_events.append(event)
                    yield event

                    await self._append_tool_result(tc.id, tc.name, result)

                    if tc.name in _USER_INTERACTION_TOOLS:
                        ask_user_pending = True

                if ask_user_pending:
                    break
            else:
                self._messages.append({"role": "assistant", "content": response_text or ""})
                await self._persist_turn("assistant", content=response_text or "")
                break

        self._maybe_trim()
        self._total_tokens += total_tokens
        self._session_state.turn_count = self._turn_counter
        self._session_state.total_tokens = self._total_tokens
        self._update_system_message()

        extract_state_from_tool_events(
            self._session_state, self._turn_counter, all_tool_events,
        )

        await self._persist_session_metadata()

        yield CoworkTurn(
            text="\n\n".join(all_text_parts),
            tool_calls=all_tool_events,
            tokens_used=total_tokens,
            model=self._model.name,
        )

    # ------------------------------------------------------------------
    # Session resumption
    # ------------------------------------------------------------------

    async def resume(self, session_id: str) -> None:
        """Resume a previous session from the archive."""
        if self._store is None:
            raise RuntimeError("Cannot resume without a ConversationStore.")

        session = await self._store.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        self._session_id = session_id
        self._total_tokens = session.get("total_tokens", 0)
        self._turn_counter = session.get("turn_count", 0)

        # Restore message counter from DB to avoid collisions
        self._message_counter = await self._store.get_turn_count(session_id)

        # Restore session state
        self._session_state = SessionState.from_json(session.get("session_state"))
        self._session_state.session_id = session_id

        # Load recent turns into in-memory cache
        recent = await self._store.resume_session(session_id)
        self._messages = [{"role": "system", "content": self._build_system_content()}] + recent

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def _context_window(self) -> list[dict]:
        """Return the messages to send to the model.

        Token-aware: walks backward from most recent messages, adding
        turns until the token budget is exhausted.  System prompt is
        always included.  Ensures we never cut mid assistant→tool sequence.
        """
        if not self._messages:
            return []

        # Always include system prompt
        system = []
        rest = self._messages
        if self._messages[0]["role"] == "system":
            system = [self._messages[0]]
            rest = self._messages[1:]

        system_tokens = sum(_estimate_message_tokens(m) for m in system)
        budget = self._max_context_tokens - system_tokens - 4000  # reserve for output

        # Walk backward, adding messages until budget is exhausted
        selected: list[dict] = []
        used = 0
        for msg in reversed(rest):
            msg_tokens = _estimate_message_tokens(msg)
            if used + msg_tokens > budget:
                break
            selected.insert(0, msg)
            used += msg_tokens

        # Ensure we don't start on an orphaned tool result
        while selected and selected[0].get("role") == "tool":
            selected.pop(0)

        return system + selected

    def _maybe_trim(self) -> None:
        """Trim in-memory message cache if very large.

        This does NOT lose data — all messages are persisted in the
        conversation store.  This just keeps RAM usage bounded.
        Ensures trim doesn't break message sequences.
        """
        max_cache = self._max_context * 3
        if len(self._messages) <= max_cache:
            return

        system = []
        if self._messages and self._messages[0]["role"] == "system":
            system = [self._messages[0]]

        kept = self._messages[-(self._max_context * 2):]
        # Don't start on an orphaned tool result
        while kept and kept[0].get("role") == "tool":
            kept.pop(0)

        self._messages = system + kept

    # ------------------------------------------------------------------
    # Dangling reference detection
    # ------------------------------------------------------------------

    def _maybe_recall_hint(self, user_message: str) -> str | None:
        """If the user references earlier context not in the window, nudge the model."""
        lower = user_message.lower()
        if any(phrase in lower for phrase in _DANGLING_REF_INDICATORS):
            return (
                "[System: The user may be referencing earlier conversation "
                "that is no longer in your context window. Use the "
                "conversation_recall tool to search for relevant context "
                "before proceeding.]"
            )
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_content(self) -> str:
        """Build the full system prompt with session state."""
        parts = [self._static_system_prompt]
        if self._session_state and self._session_state.turn_count > 0:
            parts.append(f"\n## Session State\n{self._session_state.to_yaml()}")
        return "\n".join(parts)

    def _update_system_message(self) -> None:
        """Update the system message in-place with current session state."""
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = self._build_system_content()

    async def _append_tool_result(
        self, tool_call_id: str, tool_name: str, result: ToolResult,
    ) -> None:
        """Append a tool result message and persist it."""
        content = result.to_json()
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        await self._persist_turn(
            "tool", content=content,
            tool_call_id=tool_call_id, tool_name=tool_name,
        )

    @staticmethod
    def _tool_calls_to_dicts(tool_calls: list[ToolCall]) -> list[dict]:
        """Convert ToolCall objects to OpenAI message format dicts."""
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ]

    async def _persist_turn(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Write-through: persist a turn to the conversation store."""
        if self._store is None or not self._session_id:
            return
        try:
            self._message_counter += 1
            await self._store.append_turn(
                session_id=self._session_id,
                turn_number=self._message_counter,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Persist turn failed: %s", e)

    async def _persist_session_metadata(self) -> None:
        """Persist session state and token counts."""
        if self._store is None or not self._session_id:
            return
        try:
            await self._store.update_session(
                session_id=self._session_id,
                total_tokens=self._total_tokens,
                turn_count=self._turn_counter,
                session_state=self._session_state.to_dict(),
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Persist metadata failed: %s", e)


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
- Use ask_user when you need the developer's input or decision.

CONVERSATION HISTORY:
- Your context window contains only recent turns. The full session history
  is preserved in the archive and is never lost.
- Use conversation_recall to search for earlier context when:
  - The user references something discussed earlier
  - You need information from a previous tool call
  - You're unsure about a prior decision or constraint
- When your context doesn't contain enough information to act confidently,
  search the archive before guessing or asking the user to repeat themselves.

TASK DELEGATION:
- Use delegate_task for complex multi-step work that benefits from planning,
  decomposition, verification, or parallel execution.
- delegate_task submits work to the orchestration engine which will plan,
  execute subtasks, verify results, and return a summary.
- Use delegate_task when: multi-file refactoring, migrations, new features
  with tests, or any work that would benefit from structured decomposition.
- Use direct tools when: reading/editing a single file, running a command,
  quick fixes, exploratory work, or anything simple and straightforward."""
