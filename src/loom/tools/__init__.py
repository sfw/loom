"""Tool system: registration, dispatch, and built-in tools.

Tools are auto-discovered via ``Tool.__init_subclass__``.  Any concrete
``Tool`` subclass defined in a module under ``loom.tools`` is collected
automatically when ``discover_tools()`` scans the package.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

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


def discover_tools() -> list[type[Tool]]:
    """Import all tool modules in the package and return discovered classes.

    Each module in ``loom.tools`` (except those in ``_SKIP_MODULES``) is
    imported, which triggers ``Tool.__init_subclass__`` for every concrete
    subclass defined there.  The collected classes are returned sorted by
    name for deterministic ordering.
    """
    for _finder, module_name, _is_pkg in pkgutil.walk_packages(
        _pkg.__path__, prefix=_pkg.__name__ + "."
    ):
        short_name = module_name.rsplit(".", 1)[-1]
        if short_name in _SKIP_MODULES:
            continue
        importlib.import_module(module_name)
    return sorted(Tool._registered_classes, key=lambda cls: cls.__name__)


def create_default_registry(config: Config | None = None) -> ToolRegistry:
    """Create a registry with all discovered built-in tools.

    This is the main entry-point used by ``loom.api.engine`` and tests.
    The public API is unchanged — callers still get a fully-populated
    ``ToolRegistry`` — but tools are now found automatically instead of
    being listed by hand.
    """
    registry = ToolRegistry()
    for tool_cls in discover_tools():
        registry.register(tool_cls())

    if config and config.mcp.servers:
        try:
            from loom.integrations.mcp_tools import register_mcp_tools

            register_mcp_tools(registry, mcp_config=config.mcp)
        except Exception as e:
            logger.warning("Failed to register MCP tools: %s", e)

    return registry
