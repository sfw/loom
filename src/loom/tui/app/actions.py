"""Action helpers for key-driven TUI behavior."""

from __future__ import annotations

import asyncio

from textual.app import ScreenStackError
from textual.widgets import TabbedContent

from loom.tui.screens import ExitConfirmScreen, LoomCommandPaletteScreen
from loom.tui.widgets import ChatLog, Sidebar


def activate_tab(tabs: object, tab_id: str) -> None:
    """Set the active tab id on a TabbedContent-like object."""
    setattr(tabs, "active", str(tab_id))


def command_palette_active(screen: object, *, palette_type: type[object]) -> bool:
    """Return True when the current screen is the command palette."""
    return isinstance(screen, palette_type)


async def request_manager_tab_close(self, pane_id: str) -> bool:
    """Ask active manager widget to run its guarded close flow when available."""
    selector = ""
    if pane_id == self._MCP_MANAGER_TAB_ID:
        selector = f"#{pane_id} MCPManagerScreen"
    elif pane_id == self._AUTH_MANAGER_TAB_ID:
        selector = f"#{pane_id} AuthManagerScreen"
    if not selector:
        return False
    try:
        manager_widget = self.query_one(selector)
    except Exception:
        return False
    close_action = getattr(manager_widget, "action_request_close", None)
    if not callable(close_action):
        return False
    result = close_action()
    if asyncio.iscoroutine(result):
        await result
    return True


def command_palette_is_active(self) -> bool:
    try:
        return command_palette_active(
            self.screen,
            palette_type=LoomCommandPaletteScreen,
        )
    except ScreenStackError:
        return False


def action_close_process_tab(self) -> None:
    """Close current tab (process run tab or manager tab)."""
    if self._command_palette_active():
        return
    current = self._current_process_run()
    active_tab = ""
    try:
        tabs = self.query_one("#tabs", TabbedContent)
        active_tab = str(getattr(tabs, "active", "") or "")
    except Exception:
        active_tab = ""
    close_key = current.run_id if current is not None else (active_tab or "current")
    if close_key in self._close_process_tab_inflight:
        return
    self._close_process_tab_inflight.add(close_key)

    async def _close_current_tab() -> None:
        try:
            if current is not None:
                await self._close_process_run(current)
            elif active_tab in {self._MCP_MANAGER_TAB_ID, self._AUTH_MANAGER_TAB_ID}:
                requested = await self._request_manager_tab_close(active_tab)
                if not requested:
                    await self._remove_tab_if_present(active_tab)
            elif not self._process_runs:
                try:
                    chat = self.query_one("#chat-log", ChatLog)
                    chat.add_info(
                        "No closable tabs are open. ctrl + w closes run, MCP, or Auth tabs."
                    )
                except Exception:
                    pass
            else:
                await self._close_process_run_from_target("current")
        finally:
            self._close_process_tab_inflight.discard(close_key)

    try:
        self.run_worker(
            _close_current_tab(),
            group="close-process-tab",
            exclusive=False,
        )
    except Exception:
        self._close_process_tab_inflight.discard(close_key)
        raise


def action_toggle_sidebar(self) -> None:
    self.query_one("#sidebar", Sidebar).toggle()


def action_clear_chat(self) -> None:
    self._clear_chat_widgets()


def action_reload_workspace(self) -> None:
    """Reload sidebar workspace tree to show external file changes."""
    self._request_workspace_refresh("manual", immediate=True)
    self._announce_user_feedback(
        "Workspace reloaded.",
        chat_line=True,
        timeout=2,
    )


def action_tab_chat(self) -> None:
    tabs = self.query_one("#tabs", TabbedContent)
    activate_tab(tabs, "tab-chat")


def action_tab_files(self) -> None:
    tabs = self.query_one("#tabs", TabbedContent)
    activate_tab(tabs, "tab-files")


def action_tab_events(self) -> None:
    tabs = self.query_one("#tabs", TabbedContent)
    activate_tab(tabs, "tab-events")


def action_command_palette(self) -> None:
    """Open Loom's custom command palette surface."""
    self.push_screen(LoomCommandPaletteScreen())


def action_open_auth_tab(self) -> None:
    if self._command_palette_active():
        return
    self._open_auth_manager_screen()


def action_open_mcp_tab(self) -> None:
    if self._command_palette_active():
        return
    self._open_mcp_manager_screen()


async def action_quit(self) -> None:
    """Compatibility action that runs the exit flow inline."""
    await self._request_exit()


def action_request_quit(self) -> None:
    """Start the exit flow without blocking key/event dispatch."""
    self.run_worker(
        self._request_exit(),
        group="exit-flow",
        exclusive=True,
    )


async def confirm_exit(self) -> bool:
    """Show exit confirmation modal and return True when confirmed."""
    if self._confirm_exit_waiter is not None:
        return await self._confirm_exit_waiter

    result_waiter: asyncio.Future[bool] = asyncio.Future()
    self._confirm_exit_waiter = result_waiter

    def handle_result(confirmed: bool) -> None:
        if not result_waiter.done():
            result_waiter.set_result(bool(confirmed))

    self.push_screen(ExitConfirmScreen(), callback=handle_result)
    try:
        return await result_waiter
    finally:
        self._confirm_exit_waiter = None


async def request_exit(self) -> None:
    """Prompt for exit confirmation, then persist and quit when approved."""
    if not await self._confirm_exit():
        return
    await self._persist_process_run_ui_state(is_active=False)
    self.exit()


async def action_loom_command(self, command: str) -> None:
    """Dispatch command palette actions."""
    if command.startswith("process_run_prompt:"):
        process_name = command.partition(":")[2].strip()
        if process_name:
            self._prefill_user_input(f"/{process_name} ")
        return
    if command == "quit":
        self.action_request_quit()
        return
    if command == "setup":
        await self._handle_slash_command("/setup")
        return
    if command == "session_info":
        await self._handle_slash_command("/session")
        return
    if command == "new_session":
        await self._handle_slash_command("/new")
        return
    if command == "sessions_list":
        await self._handle_slash_command("/sessions")
        return
    if command == "mcp_list":
        await self._handle_slash_command("/mcp list")
        return
    if command == "mcp_manage":
        await self._handle_slash_command("/mcp manage")
        return
    if command == "mcp_add_prompt":
        self._prefill_user_input(
            "/mcp add <alias> --command <cmd> --arg <value> "
        )
        return
    if command == "auth_list":
        await self._handle_slash_command("/auth manage")
        return
    if command == "auth_manage":
        await self._handle_slash_command("/auth manage")
        return
    if command == "auth_add_prompt":
        await self._handle_slash_command("/auth manage")
        return
    if command == "learned_patterns":
        await self._handle_slash_command("/learned")
        return
    if command == "run_prompt":
        self._prefill_user_input("/run ")
        return
    if command == "pause_chat":
        await self._handle_slash_command("/pause")
        return
    if command == "resume_chat":
        await self._handle_slash_command("/steer resume")
        return
    if command == "inject_prompt":
        self._prefill_user_input("/inject ")
        return
    if command == "redirect_prompt":
        self._prefill_user_input("/redirect ")
        return
    if command == "steer_queue":
        await self._handle_slash_command("/steer queue")
        return
    if command == "steer_clear":
        await self._handle_slash_command("/steer clear")
        return
    if command == "stop_chat":
        self.action_stop_chat()
        return
    if command == "resume_prompt":
        self._prefill_user_input("/resume ")
        return
    if command == "close_process_tab":
        self.action_close_process_tab()
        return
    actions = {
        "clear_chat": self.action_clear_chat,
        "toggle_sidebar": self.action_toggle_sidebar,
        "reload_workspace": self.action_reload_workspace,
        "tab_chat": self.action_tab_chat,
        "tab_files": self.action_tab_files,
        "tab_events": self.action_tab_events,
        "open_auth_tab": self.action_open_auth_tab,
        "open_mcp_tab": self.action_open_mcp_tab,
        "list_tools": self._show_tools,
        "model_info": self._show_model_info,
        "models_info": self._show_models_info,
        "process_info": self._show_process_info,
        "process_list": self._show_process_list,
        "token_info": self._show_token_info,
        "help": self._show_help,
    }
    action_fn = actions.get(command)
    if action_fn:
        action_fn()
