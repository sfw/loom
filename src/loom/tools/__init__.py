"""Tool system: registration, dispatch, and built-in tools."""

from loom.tools.file_ops import EditFileTool, ReadFileTool, WriteFileTool
from loom.tools.registry import Tool as Tool
from loom.tools.registry import ToolContext as ToolContext
from loom.tools.registry import ToolRegistry
from loom.tools.registry import ToolResult as ToolResult
from loom.tools.search import ListDirectoryTool, SearchFilesTool
from loom.tools.shell import ShellExecuteTool


def create_default_registry() -> ToolRegistry:
    """Create a registry with all built-in tools registered."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ShellExecuteTool())
    registry.register(SearchFilesTool())
    registry.register(ListDirectoryTool())
    return registry
