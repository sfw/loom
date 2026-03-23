"""TUI core panel and foundational widget tests."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestTheme:
    def test_loom_dark_theme(self):
        from loom.tui.theme import LOOM_DARK
        assert LOOM_DARK.name == "loom-dark"
        assert LOOM_DARK.dark is True
        assert LOOM_DARK.primary == "#7dcfff"

    def test_markdown_rich_theme_overrides_magenta_defaults(self):
        from loom.tui.theme import LOOM_MARKDOWN_RICH_THEME

        styles = LOOM_MARKDOWN_RICH_THEME.styles
        assert "markdown.h2" in styles
        assert str(styles["markdown.h2"]) == "underline #7dcfff"
        assert str(styles["markdown.block_quote"]) == "#9aa5ce"

    def test_color_constants(self):
        from loom.tui.theme import (
            ACCENT_CYAN,
            TOOL_ERR,
            TOOL_OK,
            USER_MSG,
        )
        assert USER_MSG == "#73daca"
        assert TOOL_OK == "#9ece6a"
        assert TOOL_ERR == "#f7768e"
        assert ACCENT_CYAN == "#7dcfff"

class TestStatusBar:
    def test_default_state(self):
        from loom.tui.widgets.status_bar import StatusBar
        bar = StatusBar()
        assert bar.state == "Ready"
        assert bar.total_tokens == 0

    def test_render_includes_process_name(self):
        from loom.tui.widgets.status_bar import StatusBar

        bar = StatusBar()
        bar.workspace_name = "loom"
        bar.model_name = "primary"
        bar.process_name = "marketing-strategy"
        bar.total_tokens = 12

        rendered = bar.render()
        assert "process:marketing-strategy" in rendered
        assert "12 tokens" in rendered

class TestActivityIndicator:
    def test_idle_render_uses_dim_static_dots(self):
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        indicator = ActivityIndicator(dot_count=8, idle_hold_ms=0)
        rendered = indicator.render()
        assert rendered.count(indicator._GLYPH_IDLE) == 8
        assert indicator._GLYPH_HEAD not in rendered

    def test_active_frame_progression_ping_pongs(self):
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        indicator = ActivityIndicator(dot_count=4, idle_hold_ms=0)
        indicator.set_active(True)
        sequence = [indicator._frame_index]
        for _ in range(7):
            indicator._advance_frame()
            sequence.append(indicator._frame_index)
        assert sequence == [0, 1, 2, 3, 2, 1, 0, 1]

    def test_inactive_resets_to_dim_strip(self):
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        indicator = ActivityIndicator(dot_count=4, idle_hold_ms=0)
        indicator.set_active(True)
        indicator._advance_frame()
        indicator.set_active(False)
        rendered = indicator.render()
        assert indicator._frame_index == 0
        assert rendered.count(indicator._GLYPH_IDLE) == 4
        assert indicator._GLYPH_HEAD not in rendered

    def test_repeated_idle_sync_does_not_restart_hold_window(self):
        from loom.tui.widgets.activity_indicator import ActivityIndicator

        indicator = ActivityIndicator(dot_count=8, idle_hold_ms=300)
        indicator.set_active(True)
        indicator.set_active(False)
        first_hold_until = indicator._hold_until
        indicator.set_active(False)
        assert indicator._hold_until == first_hold_until

class TestTaskProgressPanel:
    def test_render_empty(self):
        from rich.text import Text

        from loom.tui.widgets.sidebar import TaskProgressPanel
        panel = TaskProgressPanel()
        rendered = panel.render()
        assert isinstance(rendered, Text)
        assert "No tasks tracked" in rendered.plain

    def test_render_with_tasks(self):
        from rich.console import Console

        from loom.tui.widgets.sidebar import TaskProgressPanel
        panel = TaskProgressPanel()
        panel.tasks = [
            {"content": "Read file", "status": "completed"},
            {
                "content": "crypto-externalities-article-adhoc #2f3f27 Running 29:46",
                "status": "in_progress",
            },
            {"content": "Run tests", "status": "pending"},
            {"content": "Handle failure", "status": "failed"},
            {"content": "Skip optional step", "status": "skipped"},
        ]
        rendered = panel.render()
        console = Console(width=34, record=True)
        console.print(rendered)
        plain = console.export_text(styles=False)

        assert "Read file" in plain
        assert "crypto-externalities-article-adh" in plain
        assert "oc #2f3f27 Running 29:46" in plain
        assert "Run tests" in plain
        assert "Handle failure" in plain
        assert "Skip optional step" in plain
        assert "\n◉\n" not in plain

    def test_task_update_triggers_scroll_hook(self):
        from loom.tui.widgets.sidebar import TaskProgressPanel

        panel = TaskProgressPanel(auto_follow=True)
        panel._scroll_to_latest = MagicMock()
        panel.tasks = [{"content": "Read file", "status": "completed"}]
        assert panel._scroll_to_latest.call_count >= 1

    def test_empty_message_update_triggers_scroll_hook(self):
        from loom.tui.widgets.sidebar import TaskProgressPanel

        panel = TaskProgressPanel(auto_follow=True)
        panel._scroll_to_latest = MagicMock()
        panel.empty_message = "No outputs yet"
        assert panel._scroll_to_latest.call_count >= 1

class TestProcessRunPane:
    def test_process_run_panels_enable_auto_follow(self):
        from loom.tui.app import ProcessRunPane

        pane = ProcessRunPane(
            run_id="abc123",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
        )
        assert pane._progress._auto_follow is True
        assert pane._progress._follow_mode == "active"
        assert pane._outputs._auto_follow is True
        assert pane._outputs._follow_mode == "active"

    def test_process_run_controls_and_restart_visibility(self):
        from loom.tui.app import ProcessRunPane

        pane = ProcessRunPane(
            run_id="abc123",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
        )
        pane.set_status_header(
            status="queued",
            elapsed="0:00",
            task_id="",
            working_folder="(workspace root)",
        )
        meta_text = str(getattr(pane._meta, "_Static__content", ""))
        assert "Working folder:" in meta_text
        assert "(workspace root)" in meta_text
        assert pane._actions.display is True
        assert pane._toggle_pause_button.display is False
        assert pane._stop_button.display is True
        assert pane._stop_button.disabled is False

        pane.set_status_header(status="running", elapsed="0:01", task_id="cowork-1")
        assert pane._actions.display is True
        assert pane._toggle_pause_button.display is True
        assert pane._toggle_pause_button.disabled is False
        assert pane._toggle_pause_button.label == "\u23f8"
        assert pane._stop_button.display is True
        assert pane._stop_button.disabled is False
        assert pane._restart_button.display is False
        assert pane._restart_button.disabled is True

        pane.set_status_header(status="paused", elapsed="0:11", task_id="cowork-1")
        assert pane._toggle_pause_button.disabled is False
        assert pane._toggle_pause_button.label == "\u25b6"

        pane.set_status_header(status="failed", elapsed="0:42", task_id="")
        assert pane._actions.display is True
        assert pane._toggle_pause_button.display is False
        assert pane._restart_button.display is True
        assert pane._restart_button.disabled is False

        pane.set_status_header(status="failed", elapsed="0:42", task_id="cowork-1")
        assert pane._actions.display is True
        assert pane._toggle_pause_button.display is False
        assert pane._restart_button.display is True
        assert pane._restart_button.disabled is False

    def test_elapsed_seconds_freezes_while_paused(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            started_at=10.0,
            ended_at=None,
            paused_started_at=30.0,
            paused_accumulated_seconds=4.0,
        )

        with patch("loom.tui.app.time.monotonic", return_value=40.0):
            first = app._elapsed_seconds_for_run(run)
        with patch("loom.tui.app.time.monotonic", return_value=120.0):
            second = app._elapsed_seconds_for_run(run)

        assert first == second
        assert first == 16.0

    def test_elapsed_seconds_freezes_while_waiting_for_user_input(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._update_process_run_visuals = MagicMock()
        app._refresh_sidebar_progress_summary = MagicMock()
        run = SimpleNamespace(
            run_id="abc123",
            started_at=10.0,
            ended_at=None,
            paused_started_at=0.0,
            paused_accumulated_seconds=0.0,
            user_input_pause_started_at=0.0,
            user_input_paused_accumulated_seconds=0.0,
        )
        app._process_runs = {"abc123": run}

        with patch("loom.tui.app.process_runs.ui_state.time.monotonic", return_value=40.0):
            app._begin_process_run_user_input_pause("abc123")
            first = app._elapsed_seconds_for_run(run)
        with patch("loom.tui.app.process_runs.ui_state.time.monotonic", return_value=120.0):
            second = app._elapsed_seconds_for_run(run)

        assert first == second
        assert first == 30.0

    def test_has_active_process_runs_ignores_paused_status(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._process_runs = {
            "p1": SimpleNamespace(status="paused"),
            "p2": SimpleNamespace(status="completed"),
        }
        assert app._has_active_process_runs() is False

        app._process_runs["p3"] = SimpleNamespace(status="running")
        assert app._has_active_process_runs() is True

    def test_process_run_working_folder_label_formats_root_and_relative(self, tmp_path):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=tmp_path,
        )
        run = SimpleNamespace(run_workspace=tmp_path)
        assert app._process_run_working_folder_label(run) == "(workspace root)"

        run.run_workspace = tmp_path / "runs" / "research-report"
        assert app._process_run_working_folder_label(run) == "runs/research-report"

    def test_process_run_rows_render_with_fold_overflow(self):
        from rich.text import Text

        from loom.tui.app import ProcessRunList

        panel = ProcessRunList(empty_message="No progress yet")
        panel._rows = [{
            "status": "completed",
            "content": (
                "Generate a high-volume longlist of slogan/tagline options across all "
                "territories and devices before filtering."
            ),
        }]

        rendered = panel._render_rows()
        assert isinstance(rendered, Text)
        assert rendered.no_wrap is False
        assert rendered.overflow == "fold"
        assert "longlist of slogan/tagline options" in rendered.plain

    def test_process_run_rows_escape_markup_like_content(self):
        from loom.tui.app import ProcessRunList

        panel = ProcessRunList(empty_message="No progress yet")
        panel._rows = [{
            "status": "completed",
            "content": "Verifier note ...[truncated]... follow-up",
        }]

        rendered = panel._render_rows()
        assert "[truncated]" in rendered.plain

    def test_process_run_result_coerces_rich_text_to_plain(self):
        from rich.text import Text

        from loom.tui.app import ProcessRunPane

        pane = ProcessRunPane(
            run_id="abc123",
            process_name="campaign-slogans",
            goal="Generate campaign slogans",
        )
        pane._log = MagicMock()

        pane.add_result(Text.from_markup("[dim]ok[/dim]"), success=True)

        assert pane._pending_results == [("ok", True)]

    def test_process_run_active_follow_waits_for_started_rows(self):
        from unittest.mock import PropertyMock, patch

        from loom.tui.app import ProcessRunList

        panel = ProcessRunList(
            empty_message="No outputs yet",
            auto_follow=True,
            follow_mode="active",
        )
        panel._rows = [
            {"status": "pending", "content": "slogan-longlist.csv"},
            {"status": "pending", "content": "shortlist-scorecard.csv"},
        ]
        panel.call_after_refresh = MagicMock()

        with patch.object(
            ProcessRunList,
            "is_attached",
            new_callable=PropertyMock,
            return_value=True,
        ):
            panel._scroll_to_latest()

        panel.call_after_refresh.assert_not_called()

    def test_process_run_active_follow_targets_current_output(self):
        from unittest.mock import PropertyMock, patch

        from loom.tui.app import ProcessRunList

        panel = ProcessRunList(
            empty_message="No outputs yet",
            auto_follow=True,
            follow_mode="active",
        )
        panel._rows = [
            {"status": "pending", "content": "brief-normalized.md"},
            {"status": "completed", "content": "brief-assumptions.md"},
            {"status": "completed", "content": "tension-map.csv"},
            {"status": "in_progress", "content": "insight-angles.md"},
            {"status": "pending", "content": "signal-board.md"},
        ]
        panel.scroll_to = MagicMock()

        def _run_immediately(callback, *_args, **_kwargs):
            callback()

        panel.call_after_refresh = MagicMock(side_effect=_run_immediately)

        with patch.object(
            ProcessRunList,
            "is_attached",
            new_callable=PropertyMock,
            return_value=True,
        ):
            panel._scroll_to_latest()

        panel.scroll_to.assert_called_once_with(y=1, animate=False, force=True)

    def test_process_run_stage_rows_render_when_tasks_empty(self):
        from loom.tui.app import LoomApp, ProcessRunState

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = ProcessRunState(
            run_id="abc123",
            process_name="market-research",
            goal="Analyze EPCOR",
            run_workspace=Path("/tmp"),
            process_defn=None,
            pane_id="tab-run-abc123",
            pane=pane,
            status="queued",
            launch_stage="auth_preflight",
            tasks=[],
        )

        rows = app._process_run_stage_rows(run)

        assert rows
        assert any(row["content"] == "Auth preflight" for row in rows)
        assert any(row["status"] == "in_progress" for row in rows)

    def test_process_run_heartbeat_emits_liveness_line(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            closed=False,
            status="queued",
            launch_stage="auth_preflight",
            launch_last_progress_at=0.0,
            launch_last_heartbeat_at=0.0,
            launch_silent_warning_emitted=False,
            launch_stage_heartbeat_stage="",
            launch_stage_heartbeat_dots=0,
            activity_log=[],
            launch_stage_activity_indices={},
            pane=MagicMock(),
        )

        app._maybe_emit_process_run_heartbeat(run)

        assert run.activity_log
        assert run.activity_log[-1].startswith("Auth preflight.")
        assert "Still working" not in run.activity_log[-1]
        assert run.launch_last_heartbeat_at > 0.0

    def test_process_run_heartbeat_updates_same_stage_line_with_more_dots(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            closed=False,
            status="queued",
            launch_stage="resolving_process",
            launch_last_progress_at=0.0,
            launch_last_heartbeat_at=0.0,
            launch_silent_warning_emitted=False,
            launch_stage_heartbeat_stage="",
            launch_stage_heartbeat_dots=0,
            activity_log=[],
            launch_stage_activity_indices={},
            pane=MagicMock(),
        )

        app._maybe_emit_process_run_heartbeat(run)
        first = run.activity_log[-1]
        run.launch_last_progress_at -= 7.0
        run.launch_last_heartbeat_at -= 7.0
        app._maybe_emit_process_run_heartbeat(run)
        second = run.activity_log[-1]

        assert len(run.activity_log) == 1
        assert first == "Resolving process."
        assert second == "Resolving process.."

    def test_process_run_heartbeat_updates_running_stage_line(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        run = SimpleNamespace(
            closed=False,
            status="running",
            launch_stage="running",
            launch_last_progress_at=0.0,
            launch_last_heartbeat_at=0.0,
            launch_silent_warning_emitted=False,
            launch_stage_heartbeat_stage="",
            launch_stage_heartbeat_dots=0,
            activity_log=[],
            launch_stage_activity_indices={},
            pane=MagicMock(),
        )

        app._maybe_emit_process_run_heartbeat(run)
        first = run.activity_log[-1]
        run.launch_last_progress_at -= 7.0
        run.launch_last_heartbeat_at -= 7.0
        app._maybe_emit_process_run_heartbeat(run)
        second = run.activity_log[-1]

        assert len(run.activity_log) == 1
        assert first == "Running."
        assert second == "Running.."
        run.pane.upsert_activity.assert_called()

    def test_set_process_run_launch_stage_finalizes_previous_phase_with_elapsed(self):
        from loom.tui.app import LoomApp, ProcessRunState

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = ProcessRunState(
            run_id="abc777",
            process_name="market-research",
            goal="Analyze EPCOR",
            run_workspace=Path("/tmp"),
            process_defn=None,
            pane_id="tab-run-abc777",
            pane=pane,
            status="queued",
            launch_stage="resolving_process",
            launch_stage_started_at=time.monotonic() - 12.0,
        )

        app._set_process_run_launch_stage(run, "provisioning_workspace", note="")

        assert any("Resolving process." in line and "00:12" in line for line in run.activity_log)
        assert any(line.startswith("Provisioning workspace.") for line in run.activity_log)

    def test_process_run_progress_keeps_stage_summary_when_tasks_ready(self):
        from loom.tui.app import LoomApp, ProcessRunState

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        pane = MagicMock()
        run = ProcessRunState(
            run_id="abc125",
            process_name="market-research",
            goal="Analyze EPCOR",
            run_workspace=Path("/tmp"),
            process_defn=None,
            pane_id="tab-run-abc125",
            pane=pane,
            status="running",
            launch_stage="auth_preflight",
            tasks=[
                {
                    "id": "scope-companies",
                    "status": "in_progress",
                    "content": "Scope requested companies",
                },
            ],
        )

        app._refresh_process_run_progress(run)

        rows = pane.set_tasks.call_args.args[0]
        assert rows[0]["id"] == "stage:summary"
        assert rows[0]["status"] == "in_progress"
        assert "Auth preflight" in rows[0]["content"]
        assert rows[1]["id"] == "scope-companies"

class TestSidebarWidget:
    def test_refresh_workspace_tree_calls_reload(self):
        from loom.tui.widgets.sidebar import Sidebar

        sidebar = Sidebar(workspace=Path("/tmp"))
        tree = MagicMock()
        sidebar.query_one = MagicMock(return_value=tree)

        sidebar.refresh_workspace_tree()

        tree.reload.assert_called_once()

class TestChatLogStreaming:
    def test_flush_stream_buffer_uses_internal_text(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        widget = MagicMock()
        log._stream_widget = widget
        log._stream_text = "hello"
        log._stream_buffer = [" ", "world"]

        log._flush_stream_buffer()

        widget.update.assert_called_once_with("hello world")
        assert log._stream_text == "hello world"
        assert log._stream_buffer == []

    def test_flush_and_reset_stream_clears_state(self):
        from rich.markdown import Markdown as RichMarkdown

        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        widget = MagicMock()
        log._stream_widget = widget
        log._stream_text = "chunk"
        log._stream_buffer = ["!"]

        log._flush_and_reset_stream()

        assert widget.update.call_count == 2
        assert widget.update.call_args_list[0].args[0] == "chunk!"
        assert isinstance(widget.update.call_args_list[1].args[0], RichMarkdown)
        assert log._stream_widget is None
        assert log._stream_text == ""
        assert log._stream_buffer == []

    def test_static_messages_expand_to_available_width(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_user_message("hello")
        log.add_model_text("world")
        log.add_info("info")
        log.add_turn_separator(tool_count=1, tokens=42, model="test-model")

        assert mounted
        assert all(getattr(widget, "expand", False) for widget in mounted)

    def test_turn_separator_renders_latency_and_throughput(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_turn_separator(
            tool_count=2,
            tokens=42,
            model="test-model",
            tokens_per_second=21.0,
            latency_ms=450,
            total_time_ms=2200,
        )

        rendered = str(mounted[-1].render())
        assert "2 tools" in rendered
        assert "42 tokens" in rendered
        assert "21.0 tok/s" in rendered
        assert "450ms latency" in rendered
        assert "2.2s total" in rendered
        assert "test-model" in rendered

    def test_turn_separator_renders_context_telemetry(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_turn_separator(
            tool_count=0,
            tokens=12,
            model="test-model",
            context_tokens=19221,
            context_messages=45,
            omitted_messages=57,
            recall_index_used=True,
        )

        rendered = str(mounted[-1].render())
        assert "ctx 19,221 tok" in rendered
        assert "45 ctx msg" in rendered
        assert "57 archived" in rendered
        assert "recall-index" in rendered

    def test_model_text_uses_markdown_renderer(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_model_text("## Heading\n\n- a\n- b\n\n`code`")

        assert mounted
        rendered = mounted[-1].render()
        assert type(rendered).__name__ == "RichVisual"

    def test_model_text_markup_mode_keeps_rich_markup(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_model_text("[bold]Error[/]", markup=True)

        assert mounted
        rendered = mounted[-1].render()
        assert str(rendered) == "Error"

    def test_streaming_widget_expands_to_available_width(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_streaming_text("hello")

        assert log._stream_widget is not None
        assert log._stream_widget.expand is True
        assert mounted == [log._stream_widget]

    def test_streaming_scrolls_on_mount_and_flush_only(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        log.mount = lambda *_args, **_kwargs: None
        log._schedule_stream_flush = lambda: None
        log._scroll_to_end = MagicMock()

        # First chunk mounts stream widget, so one scroll.
        log.add_streaming_text("a")
        # Next three chunks are buffered only, no additional scroll.
        log.add_streaming_text("b")
        log.add_streaming_text("c")
        log.add_streaming_text("d")
        # Fifth buffered chunk flushes, so one more scroll.
        log.add_streaming_text("e")

        assert log._scroll_to_end.call_count == 2

    def test_scroll_to_end_is_coalesced_per_refresh(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        log._auto_scroll = True
        log._scroll_end_pending = False
        scheduled: list = []
        log.call_after_refresh = lambda callback, *_a, **_k: scheduled.append(callback)
        log.scroll_end = MagicMock()

        log._scroll_to_end()
        log._scroll_to_end()

        assert len(scheduled) == 1
        scheduled[0]()
        log.scroll_end.assert_called_once_with(animate=False, immediate=True)
        assert log._scroll_end_pending is False

    def test_watch_scroll_y_toggles_auto_follow_by_position(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        calls: list[tuple[float, float]] = []

        # Ensure parent watch handler is still invoked.
        log.show_vertical_scrollbar = False
        log._refresh_scroll = lambda: calls.append((1.0, 2.0))

        log._auto_scroll = True
        log._is_near_bottom = lambda **_kwargs: False
        log.watch_scroll_y(1.0, 2.0)
        assert log._auto_scroll is False

        log._is_near_bottom = lambda **_kwargs: True
        log.watch_scroll_y(2.0, 3.0)
        assert log._auto_scroll is True
        assert calls

    def test_link_aware_widget_opens_url_on_click(self):
        from types import SimpleNamespace

        from rich.style import Style

        from loom.tui.widgets.chat_log import LinkAwareStatic

        opened: list[str] = []
        stopped: list[bool] = []
        widget = LinkAwareStatic("")
        widget._open_link = lambda href: opened.append(href)  # type: ignore[method-assign]

        widget.on_click(
            SimpleNamespace(
                style=Style(link="https://example.com"),
                stop=lambda: stopped.append(True),
            )
        )

        assert opened == ["https://example.com"]
        assert stopped == [True]

    def test_link_aware_widget_sets_tooltip_from_hovered_link(self):
        from types import SimpleNamespace

        from rich.style import Style

        from loom.tui.widgets.chat_log import LinkAwareStatic

        widget = LinkAwareStatic("")
        widget.on_mouse_move(SimpleNamespace(style=Style(link="https://example.com")))
        assert widget.tooltip == "https://example.com"

        widget.on_mouse_move(SimpleNamespace(style=Style(link="#heading")))
        assert widget.tooltip is None

    def test_delegate_progress_section_lifecycle(self):
        from loom.tui.widgets.chat_log import ChatLog

        log = ChatLog()
        mounted: list = []
        log.mount = lambda widget, *_args, **_kwargs: mounted.append(widget)
        log._scroll_to_end = lambda: None

        log.add_delegate_progress_section("call_1", title="Delegated progress")
        assert log.has_delegate_progress_section("call_1") is True
        assert log.append_delegate_progress_line("call_1", "Started subtask.") is True
        assert log.finalize_delegate_progress_section(
            "call_1",
            success=True,
            elapsed_ms=1250,
        ) is True

        log.reset_runtime_state()
        assert log.has_delegate_progress_section("call_1") is False

class TestCoworkSessionTokens:
    def test_initial_total_tokens(self):
        from unittest.mock import MagicMock

        from loom.cowork.session import CoworkSession

        model = MagicMock()
        model.name = "test-model"
        tools = MagicMock()
        tools.all_schemas.return_value = []

        session = CoworkSession(model=model, tools=tools)
        assert session.total_tokens == 0

class TestQuitConfirmation:
    @pytest.mark.asyncio
    async def test_action_quit_confirmed_persists_and_exits(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._confirm_exit = AsyncMock(return_value=True)
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._session = SimpleNamespace(session_id="sess-123")
        app.exit = MagicMock()

        await app.action_quit()

        app._confirm_exit.assert_awaited_once()
        app._store.update_session.assert_awaited_once_with(
            "sess-123", is_active=False,
        )
        app.exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_action_quit_cancelled_does_not_exit(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._confirm_exit = AsyncMock(return_value=False)
        app._store = MagicMock()
        app._store.update_session = AsyncMock()
        app._session = SimpleNamespace(session_id="sess-123")
        app.exit = MagicMock()

        await app.action_quit()

        app._confirm_exit.assert_awaited_once()
        app._store.update_session.assert_not_called()
        app.exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_slash_quit_routes_through_request_exit(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app.action_request_quit = MagicMock()
        app.query_one = MagicMock(return_value=MagicMock())

        handled = await app._handle_slash_command("/quit")

        assert handled is True
        app.action_request_quit.assert_called_once()

    def test_action_request_quit_starts_worker(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        captured: dict = {}

        def fake_run_worker(coro, **kwargs):
            captured["kwargs"] = kwargs
            coro.close()
            return MagicMock()

        app.run_worker = fake_run_worker
        app.action_request_quit()

        assert captured["kwargs"]["group"] == "exit-flow"
        assert captured["kwargs"]["exclusive"] is True

    @pytest.mark.asyncio
    async def test_ctrl_c_modal_accepts_y_and_exits(self):
        from loom.tui.app import LoomApp
        from loom.tui.screens.confirm_exit import ExitConfirmScreen

        app = LoomApp(
            model=SimpleNamespace(name="test-model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._initialize_session = AsyncMock()
        app.exit = MagicMock()

        async with app.run_test() as pilot:
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert isinstance(app.screen_stack[-1], ExitConfirmScreen)

            await pilot.press("y")
            await pilot.pause()
            assert app.exit.called

    @pytest.mark.asyncio
    async def test_confirm_exit_reentrant_uses_single_modal(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )

        callbacks = []
        app.push_screen = lambda _screen, callback: callbacks.append(callback)

        first = asyncio.create_task(app._confirm_exit())
        await asyncio.sleep(0)
        second = asyncio.create_task(app._confirm_exit())
        await asyncio.sleep(0)

        assert len(callbacks) == 1
        callbacks[0](True)

        assert await first is True
        assert await second is True
