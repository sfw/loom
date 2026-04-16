from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from loom.cowork.session import CoworkTurn
from loom.tui.app.chat import history, session, steering, turns


def test_session_id_and_coercions() -> None:
    assert session.active_session_id(SimpleNamespace(session_id="abc123")) == "abc123"
    assert session.chat_event_cursor_turn({"turn_number": "4"}) == 4
    assert session.coerce_int("5") == 5
    assert session.coerce_float("3.5") == 3.5
    assert session.coerce_bool("yes") is True


def test_history_render_window_advances_in_steps_for_live_appends() -> None:
    rerender, start, follow_latest, hidden_older, hidden_newer = history.update_chat_render_window(
        total_rows=3,
        max_rows=2,
        current_start=1,
        follow_latest=True,
        mode="append",
    )
    assert rerender is False
    assert start == 1
    assert follow_latest is True
    assert hidden_older == 1
    assert hidden_newer == 0

    rerender, start, follow_latest, hidden_older, hidden_newer = history.update_chat_render_window(
        total_rows=4,
        max_rows=2,
        current_start=1,
        follow_latest=True,
        mode="append",
    )
    assert rerender is False
    assert start == 1
    assert follow_latest is True
    assert hidden_older == 1
    assert hidden_newer == 0

    rerender, start, follow_latest, hidden_older, hidden_newer = history.update_chat_render_window(
        total_rows=5,
        max_rows=2,
        current_start=1,
        follow_latest=True,
        mode="append",
    )
    assert rerender is True
    assert start == 3
    assert follow_latest is True
    assert hidden_older == 3
    assert hidden_newer == 0


def test_history_render_window_preserves_loaded_older_rows() -> None:
    rerender, start, follow_latest, hidden_older, hidden_newer = history.update_chat_render_window(
        total_rows=6,
        max_rows=2,
        current_start=2,
        follow_latest=True,
        mode="prepend_older",
    )
    assert rerender is True
    assert start == 2
    assert follow_latest is False
    assert hidden_older == 2
    assert hidden_newer == 2

    visible, hidden_older, hidden_newer = history.visible_chat_replay_events(
        replay_events=[{"seq": idx} for idx in range(1, 7)],
        start=start,
        max_rows=2,
        follow_latest=follow_latest,
    )
    assert [row["seq"] for row in visible] == [3, 4]
    assert hidden_older == 2
    assert hidden_newer == 2


def test_build_chat_render_events_collapses_lookup_noise_and_merges_thinking() -> None:
    events = [
        {
            "event_type": "assistant_thinking",
            "payload": {"text": "First pass."},
        },
        {
            "event_type": "assistant_thinking",
            "payload": {"text": "Second pass."},
        },
        {
            "event_type": "tool_call_started",
            "payload": {
                "tool_name": "web_search",
                "args": {"query": "loom transcript rendering"},
            },
        },
        {
            "event_type": "tool_call_completed",
            "payload": {
                "tool_name": "web_search",
                "args": {"query": "loom transcript rendering"},
                "output": "2 results",
            },
        },
        {
            "event_type": "content_indicator",
            "payload": {"content_blocks": []},
        },
        {
            "event_type": "tool_call_started",
            "payload": {
                "tool_name": "read_file",
                "args": {"path": "src/loom/tui/app/chat/history.py"},
            },
        },
        {
            "event_type": "tool_call_completed",
            "payload": {
                "tool_name": "read_file",
                "args": {"path": "src/loom/tui/app/chat/history.py"},
                "output": "history helpers",
            },
        },
        {
            "event_type": "assistant_text",
            "payload": {"text": "Summary ready."},
        },
    ]

    render_events = history.build_chat_render_events(
        events,
        transcript_mode=True,
        show_thinking=True,
    )

    assert [event["event_type"] for event in render_events] == [
        "assistant_thinking",
        "info",
        "assistant_text",
    ]
    assert render_events[0]["payload"]["text"] == "First pass.\n\nSecond pass."
    assert "Collapsed 2 lookup tool calls." in render_events[1]["payload"]["text"]
    assert render_events[2]["payload"]["text"] == "Summary ready."


def test_steering_and_turn_helpers() -> None:
    directive = steering.new_steering_directive(
        kind="inject",
        text="focus tests",
        source="slash",
        id_factory=lambda: "abc123",
    )
    assert directive.id == "abc123"
    assert turns.delegate_target_for_tool_call("run_tool", {"name": "read_file"}) == "read_file"
    assert turns.delegate_progress_title("run_tool") == "Delegated progress (run_tool)"


@pytest.mark.asyncio
async def test_run_interaction_suppresses_live_feedback_chunks() -> None:
    class FakeSession:
        persisted_turn_count = 0

        async def send_streaming(self, _message: str):
            yield ("thinking", "First pass.")
            yield ("thinking", "Second pass.")
            yield CoworkTurn(
                text="Final answer.",
                tool_calls=[],
                tokens_used=12,
                model="test-model",
            )

    chat = MagicMock()
    status = SimpleNamespace(state="", total_tokens=0)
    events_panel = MagicMock()

    def _query_one(selector: str, *_args, **_kwargs):
        if selector == "#chat-log":
            return chat
        if selector == "#status-bar":
            return status
        if selector == "#events-panel":
            return events_panel
        raise AssertionError(f"Unexpected selector: {selector}")

    app = SimpleNamespace(
        _session=FakeSession(),
        _total_tokens=0,
        query_one=_query_one,
        _sync_pending_inject_apply_state=AsyncMock(),
        _append_chat_replay_event=AsyncMock(),
        _update_files_panel=MagicMock(),
    )

    await turns.run_interaction(app, "hello")

    chat.add_model_text.assert_called_once_with("Final answer.")
    chat.add_live_feedback.assert_not_called()
    replay_event_types = [call.args[0] for call in app._append_chat_replay_event.await_args_list]
    assert replay_event_types == ["assistant_thinking", "assistant_text", "turn_separator"]
    assert app._append_chat_replay_event.await_args_list[0].args[1]["text"] == (
        "First pass.\n\nSecond pass."
    )
