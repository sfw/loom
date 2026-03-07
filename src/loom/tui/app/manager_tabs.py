"""Auth/MCP manager tab helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from textual.widgets import TabbedContent, TabPane

from loom.processes.schema import ProcessDefinition

logger = logging.getLogger("loom.tui.app.core")


def tab_exists(self, tabs: TabbedContent, pane_id: str) -> bool:
    """Return True when TabbedContent already has pane_id registered."""
    try:
        tabs.get_tab(pane_id)
    except Exception:
        return False
    return True


async def remove_tab_if_present(
    self,
    pane_id: str,
    *,
    fallback_active: str = "tab-chat",
) -> None:
    """Remove pane if present and restore fallback active tab when needed."""
    try:
        tabs = self.query_one("#tabs", TabbedContent)
    except Exception:
        return
    if not self._tab_exists(tabs, pane_id):
        if not str(getattr(tabs, "active", "") or ""):
            tabs.active = fallback_active
        return
    was_active = str(getattr(tabs, "active", "") or "") == pane_id
    try:
        await tabs.remove_pane(pane_id)
    except Exception:
        return
    if was_active or not str(getattr(tabs, "active", "") or ""):
        tabs.active = fallback_active


def announce_user_feedback(
    self,
    message: str,
    *,
    chat_line: bool = False,
    severity: str = "information",
    timeout: int = 3,
) -> None:
    """Emit consistent short feedback via notification and optional chat line."""
    text = str(message or "").strip()
    if not text:
        return
    try:
        self.notify(text, severity=severity, timeout=timeout)
    except Exception:
        pass
    if not chat_line:
        return
    try:
        self.query_one("#chat-log").add_info(text)
    except Exception:
        return


def handle_mcp_manager_tab_close(self, result: dict[str, object]) -> None:
    """Handle embedded MCP manager close callback."""
    if bool(result.get("changed")):
        self.run_worker(
            self._reload_mcp_runtime(),
            group="mcp-manager-refresh",
            exclusive=True,
        )
        self._announce_user_feedback(
            "MCP configuration updated.",
            chat_line=True,
        )
    self.run_worker(
        self._remove_tab_if_present(self._MCP_MANAGER_TAB_ID),
        group="mcp-manager-tab-close",
        exclusive=True,
    )


async def open_mcp_manager_tab(
    self,
    *,
    mcp_manager_screen_cls: type | None = None,
) -> None:
    """Open MCP manager as a first-class tab."""
    if mcp_manager_screen_cls is None:
        from loom.tui.screens import MCPManagerScreen

        mcp_manager_screen_cls = MCPManagerScreen

    tabs = self.query_one("#tabs", TabbedContent)
    if not self._tab_exists(tabs, self._MCP_MANAGER_TAB_ID):
        manager = self._mcp_manager()
        await tabs.add_pane(
            TabPane(
                "MCP",
                mcp_manager_screen_cls(
                    manager,
                    explicit_auth_path=self._explicit_auth_path,
                    oauth_browser_login_enabled=bool(
                        getattr(
                            getattr(self._config, "mcp", None),
                            "oauth_browser_login",
                            True,
                        )
                    ),
                    embedded=True,
                    on_close=self._handle_mcp_manager_tab_close,
                ),
                id=self._MCP_MANAGER_TAB_ID,
            ),
            after="tab-events",
        )
    tabs.active = self._MCP_MANAGER_TAB_ID


def open_mcp_manager_screen(self) -> None:
    """Open MCP manager tab."""
    self.run_worker(
        self._open_mcp_manager_tab(),
        group="mcp-manager-tab-open",
        exclusive=True,
    )


def handle_auth_manager_tab_close(self, result: dict[str, object]) -> None:
    """Handle embedded auth manager close callback."""
    if bool(result.get("changed")):
        self._announce_user_feedback(
            "Auth configuration updated.",
            chat_line=True,
        )
    self.run_worker(
        self._remove_tab_if_present(self._AUTH_MANAGER_TAB_ID),
        group="auth-manager-tab-close",
        exclusive=True,
    )


async def open_auth_manager_tab(
    self,
    *,
    auth_manager_screen_cls: type | None = None,
) -> None:
    """Open auth manager as a first-class tab."""
    if auth_manager_screen_cls is None:
        from loom.tui.screens import AuthManagerScreen

        auth_manager_screen_cls = AuthManagerScreen

    tabs = self.query_one("#tabs", TabbedContent)
    if not self._tab_exists(tabs, self._AUTH_MANAGER_TAB_ID):
        process_defs = self._auth_discovery_process_defs()
        if process_defs:
            # Ensure registry sees bundled tools loaded by process contracts.
            self._refresh_tool_registry()
        await tabs.add_pane(
            TabPane(
                "Auth",
                auth_manager_screen_cls(
                    workspace=self._workspace,
                    explicit_auth_path=self._explicit_auth_path,
                    mcp_manager=self._mcp_manager(),
                    process_def=self._process_defn,
                    process_defs=process_defs,
                    tool_registry=self._tools,
                    embedded=True,
                    on_close=self._handle_auth_manager_tab_close,
                ),
                id=self._AUTH_MANAGER_TAB_ID,
            ),
            after="tab-events",
        )
    tabs.active = self._AUTH_MANAGER_TAB_ID


def open_auth_manager_screen(self) -> None:
    """Open auth manager tab."""
    self.run_worker(
        self._open_auth_manager_tab(),
        group="auth-manager-tab-open",
        exclusive=True,
    )


def auth_discovery_process_defs(self) -> list[ProcessDefinition]:
    """Load all discoverable process definitions for workspace-wide auth sync."""
    loader = self._create_process_loader()
    process_defs: list[ProcessDefinition] = []
    seen_names: set[str] = set()
    try:
        available = loader.list_available()
    except Exception:
        logger.exception("Failed listing process definitions for auth discovery")
        available = []
    for item in available:
        name = str(item.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        try:
            loaded = loader.load(name)
        except Exception:
            logger.exception("Failed loading process %s for auth discovery", name)
            continue
        loaded_name = str(getattr(loaded, "name", "")).strip() or name
        if loaded_name in seen_names:
            continue
        process_defs.append(loaded)
        seen_names.add(loaded_name)
    if self._process_defn is not None:
        active_name = str(getattr(self._process_defn, "name", "")).strip()
        if active_name and active_name not in seen_names:
            process_defs.append(self._process_defn)
    return process_defs


def auth_defaults_path(self) -> Path:
    from loom.auth.config import default_workspace_auth_defaults_path

    return default_workspace_auth_defaults_path(self._workspace)
