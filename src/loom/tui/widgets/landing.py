"""Landing surface widget for startup-first task entry."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static

_LOOM_LOGO = "\n".join((
    "[#c0caf5] _        ____   ____  __  __ [/]",
    "[#a9b9f3]| |      / __ \\ / __ \\|  \\/  |[/]",
    "[#8da4ee]| |     / / / // / / /| \\  / |[/]",
    "[#7dcfff]| |____/ /_/ // /_/ / | |\\/| |[/]",
    "[#c0caf5]|_____/\\____/ \\____/  |_|  |_|[/]",
  #  "[dim]       \\\\____/ \\\\____/           [/]",
))


def _workspace_display(path: Path) -> str:
    """Format workspace path with home-dir shorthand for readability."""
    text = str(path)
    try:
        home = str(Path.home())
    except Exception:
        home = ""
    if home and text.startswith(home):
        return "~" + text[len(home):]
    return text


def _escape_markup(value: object | None) -> str:
    """Escape Rich markup brackets in dynamic labels."""
    return str(value or "").replace("[", "\\[")


class LandingSurface(Vertical):
    """Branded startup composer shown before entering chat workspace."""

    DEFAULT_CSS = """
    LandingSurface {
        height: 1fr;
        display: none;
        background: $background;
        padding: 1 2;
    }
    LandingSurface.active {
        display: block;
    }
    LandingSurface #landing-center {
        height: 1fr;
        width: 1fr;
    }
    LandingSurface #landing-center-middle {
        width: 1fr;
        height: auto;
    }
    LandingSurface #landing-header {
        dock: top;
        width: 1fr;
        height: 3;
    }
    LandingSurface #landing-header-spacer {
        width: 1fr;
    }
    LandingSurface .landing-close-btn {
        width: 5;
        min-width: 5;
        height: 3;
        margin: 0 0 0 1;
        padding: 0;
        content-align: center middle;
        border: solid $primary-darken-1;
        background: $surface;
        color: $text-muted;
        text-style: bold;
    }
    LandingSurface .landing-close-btn:hover {
        border: solid $primary;
        background: $surface;
        color: $primary;
    }
    LandingSurface .landing-close-btn:focus {
        border: solid $primary;
        background: $surface;
        color: $primary;
        text-style: bold;
    }
    LandingSurface #landing-center-top-spacer,
    LandingSurface #landing-center-bottom-spacer {
        width: 1fr;
        height: 1fr;
    }
    LandingSurface #landing-center-left-spacer,
    LandingSurface #landing-center-right-spacer {
        width: 1fr;
    }
    LandingSurface #landing-stack {
        width: 84;
        max-width: 92%;
        min-width: 52;
        height: auto;
    }
    LandingSurface #landing-logo {
        width: 1fr;
        height: auto;
        text-align: center;
        text-style: bold;
        margin: 0 0 1 0;
    }
    LandingSurface #landing-card {
        width: 1fr;
        height: auto;
        background: transparent;
        border: none;
        padding: 0;
    }
    LandingSurface #landing-input-top-rule {
        height: 1;
        border: none;
        border-top: solid $primary-darken-1;
        border-left: solid $primary-darken-2;
        border-right: solid $primary-darken-2;
        background: $panel;
    }
    LandingSurface #landing-input-row {
        width: 100%;
        height: 2;
    }
    LandingSurface #landing-input {
        width: 1fr;
        height: 2;
        margin: 0;
        padding: 0 1;
        border: none;
        border-left: solid $primary-darken-2;
        border-right: solid $primary-darken-2;
        border-bottom: solid $primary-darken-1;
        background: $panel;
        color: $text;
    }
    LandingSurface #landing-input:focus {
        border: none;
        border-left: solid $primary-darken-1;
        border-right: solid $primary-darken-1;
        border-bottom: solid $primary;
        background: $surface;
    }
    LandingSurface #landing-shortcuts {
        width: 1fr;
        height: auto;
        margin: 1 0 0 0;
    }
    LandingSurface #landing-slash-hint {
        width: 1fr;
        height: auto;
        max-height: 10;
        margin: 0 0 1 0;
        padding: 0;
        border: none;
        border-left: solid $primary-darken-2;
        border-right: solid $primary-darken-2;
        border-bottom: solid $primary-darken-1;
        background: $panel;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-size: 1 1;
        display: none;
    }
    LandingSurface #landing-slash-hint-body {
        width: 1fr;
        min-height: 1;
        padding: 0 1;
        color: $text-muted;
        display: none;
    }
    LandingSurface .landing-shortcut {
        color: $text-muted;
        width: auto;
        margin: 0 2 0 0;
    }
    LandingSurface #landing-meta {
        dock: bottom;
        width: 1fr;
        height: 1;
        padding: 0 1;
    }
    LandingSurface #landing-workspace-path {
        width: 1fr;
        color: $text-muted;
    }
    LandingSurface #landing-model-name {
        width: auto;
        color: $text-muted;
    }
    """

    def __init__(self, workspace: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self._workspace = workspace

    def compose(self) -> ComposeResult:
        with Horizontal(id="landing-header"):
            yield Static("", id="landing-header-spacer")
            yield Static(
                "x",
                id="landing-close-btn",
                classes="landing-close-btn",
            )
        with Vertical(id="landing-center"):
            yield Static("", id="landing-center-top-spacer")
            with Horizontal(id="landing-center-middle"):
                yield Static("", id="landing-center-left-spacer")
                with Vertical(id="landing-stack"):
                    yield Static(_LOOM_LOGO, id="landing-logo")
                    with Vertical(id="landing-card"):
                        yield Static("", id="landing-input-top-rule")
                        with Horizontal(id="landing-input-row"):
                            yield Input(
                                placeholder="Give me a challenge",
                                id="landing-input",
                            )
                    with VerticalScroll(id="landing-slash-hint"):
                        yield Static("", id="landing-slash-hint-body")
                    with Horizontal(id="landing-shortcuts"):
                        yield Static(
                            "[#7dcfff]ctrl + p[/] commands",
                            classes="landing-shortcut",
                        )
                        yield Static(
                            "[#7dcfff]ctrl + c[/] exit",
                            classes="landing-shortcut",
                        )
                        yield Static(
                            "[#7dcfff]esc[/] goto main window",
                            classes="landing-shortcut",
                        )
                yield Static("", id="landing-center-right-spacer")
            yield Static("", id="landing-center-bottom-spacer")
        with Horizontal(id="landing-meta"):
            yield Static("", id="landing-workspace-path")
            yield Static("", id="landing-model-name")

    def set_context(
        self,
        *,
        model_name: str,
    ) -> None:
        """Update landing metadata labels (workspace + model)."""
        model_text = _escape_markup(str(model_name or "").strip() or "unconfigured")
        workspace_text = _escape_markup(_workspace_display(self._workspace))
        self.query_one("#landing-workspace-path", Static).update(
            f"[#7dcfff]workspace:[/] [dim]{workspace_text}[/]"
        )
        self.query_one("#landing-model-name", Static).update(
            f"[#7dcfff]model:[/] [dim]{model_text}[/]"
        )

    def focus_input(self) -> None:
        """Move focus to landing composer input."""
        self.query_one("#landing-input", Input).focus()

    def set_input_text(self, text: str) -> None:
        """Seed landing input text and move cursor to end."""
        input_widget = self.query_one("#landing-input", Input)
        input_widget.value = text
        input_widget.cursor_position = len(text)
