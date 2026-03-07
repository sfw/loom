from __future__ import annotations

from types import SimpleNamespace

from loom.tui.app.chat import history, session, steering, turns


def test_session_id_and_coercions() -> None:
    assert session.active_session_id(SimpleNamespace(session_id="abc123")) == "abc123"
    assert session.chat_event_cursor_turn({"turn_number": "4"}) == 4
    assert session.coerce_int("5") == 5
    assert session.coerce_float("3.5") == 3.5
    assert session.coerce_bool("yes") is True


def test_history_render_cap_trims_and_tracks_cursor() -> None:
    events = [{"seq": 1}, {"seq": 2}, {"seq": 3}]
    trimmed, replay_events, trim_total, oldest_seq, oldest_turn = history.apply_chat_render_cap(
        replay_events=events,
        max_rows=2,
        trim_total=0,
        history_source="journal",
        oldest_seq=None,
        oldest_turn=None,
        active_session_id="session-1",
        event_cursor_turn=lambda _event: None,
        logger=__import__("logging").getLogger(__name__),
    )
    assert trimmed is True
    assert len(replay_events) == 2
    assert trim_total == 1
    assert oldest_seq == 2
    assert oldest_turn is None


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
