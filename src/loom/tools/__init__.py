"""Tool system: registration, dispatch, and built-in tools.

Tools are auto-discovered via ``Tool.__init_subclass__``.  Any concrete
``Tool`` subclass defined in a module under ``loom.tools`` is collected
automatically when ``discover_tools()`` scans the package.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
from typing import Any, Literal

import loom.tools as _pkg
from loom.config import Config
from loom.tools.registry import Tool as Tool
from loom.tools.registry import ToolContext as ToolContext
from loom.tools.registry import ToolRegistry
from loom.tools.registry import ToolResult as ToolResult
from loom.tools.registry import ToolSafetyError as ToolSafetyError

# Modules that contain utilities, not tools — skip during discovery.
_SKIP_MODULES = frozenset({"registry", "workspace"})
logger = logging.getLogger(__name__)
_RUN_TOOL_BLOCKED_TARGETS = frozenset({"ask_user"})
_DISCOVER_TOOLS_LOCK = threading.Lock()
_DISCOVER_TOOLS_IMPORTED = False
_DISCOVER_TOOLS_CACHE: list[type[Tool]] = []


def _tool_catalog_category(name: str) -> str:
    """Infer a compact category label for list_tools fallback output."""
    if name.startswith("mcp."):
        return "mcp"
    if name in {"list_tools", "run_tool", "ask_user", "task_tracker", "delegate_task"}:
        return "core"
    if name in {
        "read_file",
        "write_file",
        "edit_file",
        "move_file",
        "delete_file",
        "search_files",
        "list_directory",
        "shell_execute",
        "git_command",
        "analyze_code",
    }:
        return "coding"
    if name.startswith("web_"):
        return "web"
    if name in {"spreadsheet", "calculator"}:
        return "analysis"
    return "other"


def _bind_hybrid_fallback_tools(registry: ToolRegistry) -> None:
    """Bind `list_tools` and `run_tool` against this registry by default.

    Cowork sessions can re-bind these tools with session-specific policies, but
    delegated/task-mode runs also need a working baseline binding.
    """
    list_tools_tool = registry.get("list_tools")
    if list_tools_tool is not None and hasattr(list_tools_tool, "bind"):
        def _catalog_provider(auth_context: Any | None = None) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            for schema in registry.all_schemas(auth_context=auth_context):
                name = str(schema.get("name", "")).strip()
                if not name:
                    continue
                description = str(schema.get("description", "") or "").strip()
                parameters = schema.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}
                tool = registry.get(name)
                rows.append({
                    "name": name,
                    "summary": " ".join(description.split()),
                    "description": description,
                    "parameters": parameters,
                    "mutates": bool(getattr(tool, "is_mutating", False)),
                    "auth_required": bool(getattr(tool, "auth_requirements", [])),
                    "category": _tool_catalog_category(name),
                })
            rows.sort(key=lambda item: str(item.get("name", "")))
            return rows

        try:
            list_tools_tool.bind(_catalog_provider)
        except Exception as e:
            logger.debug("Failed binding list_tools fallback provider: %s", e)

    run_tool_tool = registry.get("run_tool")
    if run_tool_tool is not None and hasattr(run_tool_tool, "bind"):
        async def _dispatch(tool_name: str, arguments: dict, ctx: ToolContext) -> ToolResult:
            target = str(tool_name or "").strip()
            if not target:
                return ToolResult.fail("run_tool requires a non-empty tool name.")
            if target in _RUN_TOOL_BLOCKED_TARGETS:
                return ToolResult.fail(
                    (
                        f"Tool '{target}' must be called directly and cannot "
                        "be delegated via run_tool."
                    ),
                )
            if not isinstance(arguments, dict):
                return ToolResult.fail("run_tool 'arguments' must be an object.")

            auth_context = getattr(ctx, "auth_context", None)
            if not registry.has(target, auth_context=auth_context):
                return ToolResult.fail(f"Unknown tool: {target}")

            return await registry.execute(
                target,
                arguments,
                workspace=ctx.workspace,
                read_roots=ctx.read_roots,
                scratch_dir=ctx.scratch_dir,
                changelog=ctx.changelog,
                subtask_id=ctx.subtask_id,
                auth_context=auth_context,
            )

        try:
            run_tool_tool.bind(_dispatch)
        except Exception as e:
            logger.debug("Failed binding run_tool fallback dispatcher: %s", e)


def discover_tools() -> list[type[Tool]]:
    """Import all tool modules in the package and return discovered classes.

    Each module in ``loom.tools`` (except those in ``_SKIP_MODULES``) is
    imported, which triggers ``Tool.__init_subclass__`` for every concrete
    subclass defined there.  The collected classes are returned sorted by
    name for deterministic ordering.
    """
    global _DISCOVER_TOOLS_IMPORTED
    global _DISCOVER_TOOLS_CACHE
    with _DISCOVER_TOOLS_LOCK:
        if not _DISCOVER_TOOLS_IMPORTED:
            for _finder, module_name, _is_pkg in pkgutil.walk_packages(
                _pkg.__path__, prefix=_pkg.__name__ + "."
            ):
                short_name = module_name.rsplit(".", 1)[-1]
                if short_name in _SKIP_MODULES:
                    continue
                importlib.import_module(module_name)
            _DISCOVER_TOOLS_IMPORTED = True

        # Runtime-loaded bundled process tools can mutate Tool._registered_classes.
        # Rebuild cache when class set changes, without rescanning tool modules.
        current = sorted(Tool._registered_classes, key=lambda cls: cls.__name__)
        if _DISCOVER_TOOLS_CACHE != current:
            _DISCOVER_TOOLS_CACHE = list(current)
        return list(_DISCOVER_TOOLS_CACHE)


def create_default_registry(
    config: Config | None = None,
    *,
    mcp_startup_mode: Literal["sync", "background"] = "sync",
) -> ToolRegistry:
    """Create a registry with all discovered built-in tools.

    This is the main entry-point used by ``loom.api.engine`` and tests.
    The public API is unchanged — callers still get a fully-populated
    ``ToolRegistry`` — but tools are now found automatically instead of
    being listed by hand.
    """
    registry = ToolRegistry()
    for tool_cls in discover_tools():
        if (
            config is not None
            and getattr(tool_cls, "name", "") == "delegate_task"
        ):
            timeout_seconds = int(
                getattr(
                    config.execution,
                    "delegate_task_timeout_seconds",
                    3600,
                ) or 3600
            )
            timeout_seconds = max(1, timeout_seconds)
            registry.register(tool_cls(timeout_seconds=timeout_seconds))
            continue
        registry.register(tool_cls())

    if config and config.mcp.servers:
        try:
            from loom.integrations.mcp_tools import register_mcp_tools

            startup_mode = str(mcp_startup_mode or "sync").strip().lower()
            if startup_mode not in {"sync", "background"}:
                startup_mode = "sync"
            register_mcp_tools(
                registry,
                mcp_config=config.mcp,
                startup_mode=startup_mode,
            )
        except Exception as e:
            logger.warning("Failed to register MCP tools: %s", e)

    _bind_hybrid_fallback_tools(registry)
    return registry
