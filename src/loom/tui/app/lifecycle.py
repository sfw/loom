"""App lifecycle and startup interaction helpers."""

from __future__ import annotations

import asyncio
import logging
import time

from textual import events
from textual.app import ComposeResult, ScreenStackError
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Static, TabbedContent, TabPane

from loom.tui.screens import SetupScreen, ToolApprovalScreen
from loom.tui.theme import LOOM_DARK, LOOM_MARKDOWN_RICH_THEME
from loom.tui.widgets import (
    ChatLog,
    EventPanel,
    FilesChangedPanel,
    LandingSurface,
    Sidebar,
    StatusBar,
)
from loom.utils.latency import diagnostics_enabled, log_latency_event

from .constants import (
    _EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS,
    _EVENT_LOOP_LAG_WARN_THRESHOLD_SECONDS,
)

logger = logging.getLogger("loom.tui.app.core")


def landing_model_display_name(self) -> str:
    """Resolve startup landing model label from config when possible."""
    if self._model is None:
        return "unconfigured"
    model_alias = str(getattr(self._model, "name", "") or "").strip()
    models_cfg = getattr(self._config, "models", None)
    if isinstance(models_cfg, dict):
        model_cfg = models_cfg.get(model_alias)
        configured_model = str(getattr(model_cfg, "model", "") or "").strip()
        if configured_model:
            return configured_model
        if len(models_cfg) == 1:
            first_cfg = next(iter(models_cfg.values()), None)
            configured_model = str(getattr(first_cfg, "model", "") or "").strip()
            if configured_model:
                return configured_model
    runtime_model = str(getattr(self._model, "model", "") or "").strip()
    if runtime_model:
        return runtime_model
    if model_alias:
        return model_alias
    return "unconfigured"


def sync_landing_surface(self) -> None:
    """Refresh landing metadata labels (workspace + model)."""
    try:
        landing = self.query_one("#landing-surface", LandingSurface)
    except Exception:
        return
    landing.set_context(
        model_name=self._landing_model_display_name(),
    )


def set_startup_surface(self, *, show_landing: bool) -> None:
    """Toggle between startup landing surface and workspace layout."""
    self._startup_landing_active = bool(show_landing)
    try:
        landing = self.query_one("#landing-surface", LandingSurface)
    except Exception:
        landing = None
    try:
        main_layout = self.query_one("#main-layout", Horizontal)
    except Exception:
        main_layout = None
    try:
        bottom_stack = self.query_one("#bottom-stack", Vertical)
    except Exception:
        bottom_stack = None

    if landing is not None:
        if show_landing:
            landing.add_class("active")
        else:
            landing.remove_class("active")
    if main_layout is not None:
        if show_landing:
            main_layout.add_class("hidden")
        else:
            main_layout.remove_class("hidden")
    if bottom_stack is not None:
        if show_landing:
            bottom_stack.add_class("landing")
        else:
            bottom_stack.remove_class("landing")
    if show_landing:
        self._set_slash_hint("")
        self._sync_landing_surface()
    self._sync_chat_stop_control()


async def enter_workspace_surface(self, *, ensure_session: bool) -> None:
    """Exit landing surface and optionally initialize session first."""
    if ensure_session and self._session is None and self._model is not None:
        await self._initialize_session(
            allow_auto_resume=False,
            emit_info_messages=False,
        )
    self._set_startup_surface(show_landing=False)
    try:
        self.query_one("#user-input", Input).focus()
    except Exception:
        pass


def compose(self) -> ComposeResult:
    yield Header(show_clock=True, id="app-header")
    with Vertical(id="content-stack"):
        yield LandingSurface(self._workspace, id="landing-surface")
        with Horizontal(id="main-layout"):
            yield Sidebar(
                self._workspace,
                progress_auto_follow=self._tui_progress_auto_follow(),
                id="sidebar",
            )
            with Vertical(id="main-area"):
                with TabbedContent(id="tabs"):
                    with TabPane("Chat", id="tab-chat"):
                        yield ChatLog(id="chat-log")
                    with TabPane("Files", id="tab-files"):
                        yield FilesChangedPanel(id="files-panel")
                    with TabPane("Events", id="tab-events"):
                        yield EventPanel(id="events-panel")
    with VerticalScroll(id="slash-hint"):
        with Grid(id="steer-queue-grid"):
            yield Static("", id="slash-hint-body")
            yield Vertical(id="steer-queue-list")
    yield StatusBar(id="status-bar")
    with Vertical(id="bottom-stack"):
        yield Static("", id="input-top-rule")
        with Horizontal(id="input-row"):
            yield Input(
                placeholder="Type a message... (Enter to send)",
                id="user-input",
                classes="no-chat-controls",
            )
            yield Button("⤓", id="chat-inject-btn")
            yield Button("⤴", id="chat-redirect-btn")
            yield Button("■", id="chat-stop-btn")
        with Horizontal(id="footer-row"):
            yield Footer(id="app-footer")
            with Horizontal(id="footer-shortcuts"):
                yield Static("|", classes="footer-shortcut-divider")
                yield Button(
                    "[#ff9e64]ctrl + a[/] auth",
                    id="footer-auth-shortcut",
                    classes="footer-shortcut-btn",
                )
                yield Static("|", classes="footer-shortcut-divider")
                yield Button(
                    "[#ff9e64]ctrl + m[/] mcp",
                    id="footer-mcp-shortcut",
                    classes="footer-shortcut-btn",
                )


async def on_mount(self) -> None:
    # Register and activate theme
    self.register_theme(LOOM_DARK)
    self.theme = "loom-dark"
    self.console.push_theme(LOOM_MARKDOWN_RICH_THEME, inherit=True)
    self.error_console.push_theme(LOOM_MARKDOWN_RICH_THEME, inherit=True)
    try:
        self.query_one("#chat-stop-btn", Button).tooltip = "Stop active chat turn"
    except Exception:
        pass
    try:
        self.query_one("#chat-inject-btn", Button).tooltip = (
            "Queue steering for next safe boundary"
        )
        self.query_one("#chat-redirect-btn", Button).tooltip = (
            "Interrupt and redirect cowork objective now"
        )
    except Exception:
        pass
    self._mount_header_activity_indicator()
    self._sync_activity_indicator()
    if diagnostics_enabled():
        self.run_worker(
            self._monitor_event_loop_lag(),
            group="tui-event-loop-lag",
            exclusive=True,
        )
    self._process_elapsed_timer = self.set_interval(
        1.0,
        self._tick_process_run_elapsed,
    )

    if self._model is None:
        # No model configured - launch the setup wizard
        self.push_screen(
            SetupScreen(), callback=self._on_setup_complete,
        )
        return

    resume_target, auto_resume = await self._resolve_startup_resume_target()
    show_landing = self._should_show_startup_landing(resume_target=resume_target)
    self._set_startup_surface(show_landing=show_landing)
    if show_landing:
        self._start_workspace_watch()
        try:
            self.query_one("#landing-surface", LandingSurface).focus_input()
        except Exception:
            pass
        return

    await self._initialize_session(
        startup_resume=(resume_target, auto_resume),
        allow_auto_resume=False,
        emit_info_messages=True,
    )
    self._start_workspace_watch()
    # Keep input focus deterministic even when _initialize_session is mocked
    # in tests or returns early in partial startup paths.
    try:
        self.query_one("#user-input", Input).focus()
    except Exception:
        pass
    self._sync_chat_stop_control()
    self._refresh_hint_panel()


async def monitor_event_loop_lag(self) -> None:
    """Emit periodic event-loop lag diagnostics when enabled."""
    expected = time.monotonic() + _EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS
    while True:
        await asyncio.sleep(_EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS)
        now = time.monotonic()
        lag = max(0.0, now - expected)
        if lag >= _EVENT_LOOP_LAG_WARN_THRESHOLD_SECONDS:
            log_latency_event(
                logger,
                event="tui_event_loop_lag",
                duration_seconds=lag,
                fields={"threshold_ms": int(_EVENT_LOOP_LAG_WARN_THRESHOLD_SECONDS * 1000)},
            )
        expected = now + _EVENT_LOOP_LAG_PROBE_INTERVAL_SECONDS


def on_setup_complete(self, result: list[dict] | None) -> None:
    """Handle setup wizard dismissal."""
    if result is None:
        self.exit()
        return
    self._finalize_setup()


async def finalize_setup(self) -> None:
    """Reload config and initialize after setup wizard completes."""
    from loom.config import Config, load_config
    from loom.mcp.config import apply_mcp_overrides
    from loom.models.router import ModelRouter

    loaded = load_config()
    if isinstance(loaded, Config):
        self._config = apply_mcp_overrides(
            loaded,
            workspace=self._workspace,
        )
    else:
        # Defensive fallback for mocked/non-standard config objects.
        self._config = loaded
    self._config_runtime_store.set_config(
        self._config,
        source_path=self._config_source_path,
    )
    self._sync_effective_runtime_config()
    router = ModelRouter.from_config(self._config)
    try:
        self._model = router.select(role="executor")
    except Exception as e:
        chat = self.query_one("#chat-log", ChatLog)
        chat.add_info(
            f"[bold #f7768e]Setup error: {e}[/]\n"
            f"Edit ~/.loom/loom.toml or run /setup to try again."
        )
        return

    # P1-5: If re-running /setup during an active session, invalidate
    # the old session so _initialize_session creates a fresh one with
    # the new model and system prompt.
    if self._session is not None:
        if self._store and self._session.session_id:
            await self._store.update_session(
                self._session.session_id, is_active=False,
            )
        self._session = None

    self._set_startup_surface(show_landing=False)
    await self._initialize_session()
    self._start_workspace_watch()


def on_key(self, event: events.Key) -> None:
    """Handle user-input key captures (autocomplete + close-run shortcut)."""
    active_screen = None
    try:
        active_screen = self.screen
    except ScreenStackError:
        # Some unit tests invoke on_key() before mounting any screens.
        active_screen = None
    if isinstance(active_screen, ToolApprovalScreen):
        key = event.key.lower()
        if key == "y":
            active_screen.dismiss("approve")
            event.stop()
            event.prevent_default()
            return
        if key == "a":
            active_screen.dismiss("approve_all")
            event.stop()
            event.prevent_default()
            return
        if key in {"n", "escape", "ctrl+c", "ctrl+z"}:
            active_screen.dismiss("deny")
            event.stop()
            event.prevent_default()
            return

    if event.key == "escape":
        try:
            if len(self.screen_stack) > 1:
                return
        except Exception:
            pass

    if event.key == "escape" and self._startup_landing_active:
        event.stop()
        event.prevent_default()
        self.run_worker(
            self._enter_workspace_surface(ensure_session=True),
            name="landing-open-chat",
            group="landing-open-chat",
            exclusive=True,
        )
        return

    if event.key in {"ctrl+w", "ctrl+a", "ctrl+m"}:
        try:
            if len(self.screen_stack) > 1:
                return
        except Exception:
            pass
        focused = self.focused
        if isinstance(focused, Input) and focused.id == "user-input":
            event.stop()
            event.prevent_default()
            if event.key == "ctrl+w":
                self.action_close_process_tab()
            elif event.key == "ctrl+a":
                self.action_open_auth_tab()
            elif event.key == "ctrl+m":
                self.action_open_mcp_tab()
            return

    if event.key in ("up", "down"):
        focused = self.focused
        if isinstance(focused, Input) and focused.id == "user-input":
            if self._apply_input_history_navigation(older=event.key == "up"):
                event.stop()
                event.prevent_default()
        return

    if event.key not in ("tab", "shift+tab"):
        return
    focused = self.focused
    if not isinstance(focused, Input) or focused.id not in {"user-input", "landing-input"}:
        return
    if self._apply_slash_tab_completion(
        reverse=event.key == "shift+tab",
        input_widget=focused,
    ):
        event.stop()
        event.prevent_default()
