"""TUI slash help and file viewer tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestSlashHelp:
    def test_help_lines_include_resume_and_aliases(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        rendered = "\n".join(app._help_lines())
        assert "/resume <session-id-prefix>" in rendered
        assert "/history" in rendered
        assert "/run <goal|close" in rendered
        assert "resume <run-id-prefix|current>" in rendered
        assert "run-id-prefix" in rendered
        assert "/quit (aliases: /exit, /q)" in rendered
        assert "/setup" in rendered
        assert "ctrl + r reload workspace" in rendered
        assert "ctrl + w close tab" in rendered
        assert "ctrl + p commands" in rendered
        assert "ctrl + a auth" in rendered
        assert "ctrl + m" in rendered
        assert "mcp" in rendered

    @pytest.mark.asyncio
    async def test_resume_without_arg_shows_usage(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/resume")

        assert handled is True
        chat.add_info.assert_called_once()
        message = chat.add_info.call_args.args[0]
        assert "Usage" in message
        assert "/resume" in message
        assert "<session-id-prefix>" in message

    @pytest.mark.asyncio
    async def test_resume_while_busy_is_blocked(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._chat_busy = True
        app._store = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/resume abc")

        assert handled is True
        message = chat.add_info.call_args.args[0]
        assert "Cannot create/switch sessions" in message

    @pytest.mark.asyncio
    async def test_new_while_busy_is_blocked(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._chat_busy = True
        app._store = MagicMock()
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/new")

        assert handled is True
        message = chat.add_info.call_args.args[0]
        assert "Cannot create/switch sessions" in message

    @pytest.mark.asyncio
    async def test_history_older_reports_result(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._load_older_chat_history = AsyncMock(return_value=True)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/history older")

        assert handled is True
        app._load_older_chat_history.assert_awaited_once()
        chat.add_info.assert_called_once_with("Loaded older chat history.")

    @pytest.mark.asyncio
    async def test_history_search_routes_to_transcript_search(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._search_chat_history = MagicMock(return_value=3)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/history search render cap")

        assert handled is True
        app._search_chat_history.assert_called_once_with("render cap")
        chat.add_info.assert_called_once_with(
            "Found 3 transcript match(es) for 'render cap'."
        )

    @pytest.mark.asyncio
    async def test_history_next_reports_missing_search(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._step_chat_history_search = MagicMock(return_value=(False, 0, 0))
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled = await app._handle_slash_command("/history next")

        assert handled is True
        app._step_chat_history_search.assert_called_once_with(1)
        chat.add_info.assert_called_once_with("No active transcript search.")

    @pytest.mark.asyncio
    async def test_history_transcript_and_thinking_controls_toggle_flags(self):
        from loom.tui.app import LoomApp

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=Path("/tmp"),
        )
        app._set_chat_transcript_mode = MagicMock(return_value=True)
        app._set_chat_transcript_show_thinking = MagicMock(return_value=True)
        chat = MagicMock()
        app.query_one = MagicMock(return_value=chat)

        handled_transcript = await app._handle_slash_command("/history transcript on")
        handled_thinking = await app._handle_slash_command("/history thinking show")

        assert handled_transcript is True
        assert handled_thinking is True
        app._set_chat_transcript_mode.assert_called_once_with(True)
        app._set_chat_transcript_show_thinking.assert_called_once_with(True)
        assert chat.add_info.call_args_list[0].args[0] == "Transcript mode enabled."
        assert chat.add_info.call_args_list[1].args[0] == "Transcript thinking shown."

class TestFileViewer:
    def test_renderer_registry_supports_common_types(self):
        from loom.tui.screens.file_viewer import resolve_file_renderer

        assert resolve_file_renderer(Path("README.md")) is not None
        assert resolve_file_renderer(Path("README.markdown")) is not None
        assert resolve_file_renderer(Path("src/main.ts")) is not None
        assert resolve_file_renderer(Path("styles/site.css")) is not None
        assert resolve_file_renderer(Path("data.json")) is not None
        assert resolve_file_renderer(Path("report.csv")) is not None
        assert resolve_file_renderer(Path("slides.pptx")) is not None
        assert resolve_file_renderer(Path("paper.pdf")) is not None
        assert resolve_file_renderer(Path("image.png")) is not None
        assert resolve_file_renderer(Path("Dockerfile")) is not None
        assert resolve_file_renderer(Path("README.foobar")) is None

    def test_file_viewer_loads_markdown_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "notes.md"
        doc.write_text("# Hello\n\nThis is a markdown preview.\n", encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_file_viewer_loads_json_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "data.json"
        doc.write_text('{"b":2,"a":1}', encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_file_viewer_loads_csv_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "table.csv"
        doc.write_text("name,value\nfoo,1\nbar,2\n", encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_file_viewer_loads_html_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "index.html"
        doc.write_text(
            "<html><body><h1>Title</h1><p>Hello world</p></body></html>",
            encoding="utf-8",
        )

        screen = FileViewerScreen(doc, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_render_pdf_missing_dependency_shows_sync_hint(self, tmp_path, monkeypatch):
        import builtins

        from loom.tui.screens import file_viewer

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pypdf":
                raise ImportError("No module named 'pypdf'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        widget = file_viewer._render_pdf(pdf, None)
        rendered = str(widget.render())
        assert "PDF preview unavailable" in rendered
        assert "uv sync" in rendered

    def test_file_viewer_image_metadata_preview(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        image = workspace / "pixel.png"
        image.write_bytes(
            bytes.fromhex(
                "89504E470D0A1A0A"
                "0000000D49484452"
                "0000000100000001"
                "08060000001F15C489"
                "0000000A49444154"
                "789C6360000000020001E221BC33"
                "0000000049454E44AE426082"
            ),
        )

        screen = FileViewerScreen(image, workspace)

        assert screen._error is None
        assert screen._viewer is not None

    def test_file_viewer_unsupported_extension_sets_error(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        unknown = workspace / "data.foobar"
        unknown.write_text("hello", encoding="utf-8")

        screen = FileViewerScreen(unknown, workspace)

        assert screen._viewer is None
        assert screen._error is not None
        assert "No viewer renderer registered" in screen._error

    def test_file_viewer_click_outside_dismisses(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "notes.md"
        doc.write_text("# Hello\n", encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)
        screen.dismiss = MagicMock()
        dialog = MagicMock()
        dialog.region.contains.return_value = False
        screen.query_one = MagicMock(return_value=dialog)

        event = MagicMock()
        event.screen_x = 0
        event.screen_y = 0

        screen.on_mouse_down(event)

        screen.dismiss.assert_called_once_with(None)
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()

    def test_file_viewer_click_inside_does_not_dismiss(self, tmp_path):
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "notes.md"
        doc.write_text("# Hello\n", encoding="utf-8")

        screen = FileViewerScreen(doc, workspace)
        screen.dismiss = MagicMock()
        dialog = MagicMock()
        dialog.region.contains.return_value = True
        screen.query_one = MagicMock(return_value=dialog)

        event = MagicMock()
        event.screen_x = 1
        event.screen_y = 1

        screen.on_mouse_down(event)

        screen.dismiss.assert_not_called()
        event.stop.assert_not_called()
        event.prevent_default.assert_not_called()

    def test_workspace_file_selected_opens_viewer_modal(self, tmp_path):
        from textual.widgets import DirectoryTree

        from loom.tui.app import LoomApp
        from loom.tui.screens.file_viewer import FileViewerScreen

        workspace = tmp_path / "ws"
        workspace.mkdir()
        doc = workspace / "notes.md"
        doc.write_text("# Hello\n", encoding="utf-8")

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=workspace,
        )
        app.push_screen = MagicMock()
        app.notify = MagicMock()

        event = DirectoryTree.FileSelected(MagicMock(), doc)
        app.on_workspace_file_selected(event)

        app.push_screen.assert_called_once()
        screen = app.push_screen.call_args.args[0]
        assert isinstance(screen, FileViewerScreen)
        app.notify.assert_not_called()

    def test_workspace_file_selected_rejects_paths_outside_workspace(self, tmp_path):
        from textual.widgets import DirectoryTree

        from loom.tui.app import LoomApp

        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("# Outside\n", encoding="utf-8")

        app = LoomApp(
            model=MagicMock(name="model"),
            tools=MagicMock(),
            workspace=workspace,
        )
        app.push_screen = MagicMock()
        app.notify = MagicMock()

        event = DirectoryTree.FileSelected(MagicMock(), outside)
        app.on_workspace_file_selected(event)

        app.push_screen.assert_not_called()
        app.notify.assert_called_once_with(
            "Cannot open files outside the workspace.",
            severity="error",
            timeout=4,
        )
