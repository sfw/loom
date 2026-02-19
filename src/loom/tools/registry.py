"""Tool registry and dispatch system.

Provides registration, argument validation, execution with timeout,
and schema generation for model consumption.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    content_blocks: list | None = None  # list[ContentBlock] — rich content
    data: dict | None = None
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> str:
        from loom.content import serialize_block

        payload: dict = {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "files_changed": self.files_changed,
            "data": self.data,
        }
        if self.content_blocks:
            payload["content_blocks"] = [
                serialize_block(b) for b in self.content_blocks
            ]
        return json.dumps(payload)

    @classmethod
    def from_json(cls, data: str) -> ToolResult:
        """Reconstruct a ToolResult from its JSON representation."""
        from loom.content import deserialize_block

        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return cls(success=False, output="", error="Invalid JSON")

        if not isinstance(parsed, dict):
            return cls(success=False, output=str(parsed), error="Invalid JSON structure")

        blocks = None
        raw_blocks = parsed.get("content_blocks")
        if raw_blocks and isinstance(raw_blocks, list):
            blocks = [deserialize_block(b) for b in raw_blocks if isinstance(b, dict)]

        return cls(
            success=parsed.get("success", False),
            output=parsed.get("output", ""),
            error=parsed.get("error"),
            files_changed=parsed.get("files_changed", []),
            data=parsed.get("data"),
            content_blocks=blocks or None,
        )

    @classmethod
    def ok(cls, output: str, **kwargs) -> ToolResult:
        return cls(success=True, output=output, **kwargs)

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(success=False, output="", error=error)

    @classmethod
    def multimodal(
        cls, output: str, blocks: list, **kwargs,
    ) -> ToolResult:
        """Create a result with both text and content blocks."""
        return cls(success=True, output=output, content_blocks=blocks, **kwargs)


@dataclass
class ToolContext:
    """Context passed to tool execution."""

    workspace: Path | None
    scratch_dir: Path | None = None
    changelog: Any | None = None  # ChangeLog instance for tracking file modifications
    subtask_id: str = ""


class ToolSafetyError(Exception):
    """Raised when a tool call violates safety constraints."""


class Tool(ABC):
    """Abstract base class for all tools.

    Concrete subclasses are auto-collected via ``__init_subclass__``.
    Call ``discover_tools()`` (from ``loom.tools``) to import all tool
    modules and retrieve the collected classes.
    """

    _registered_classes: ClassVar[set[type[Tool]]] = set()
    __loom_register__: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only collect concrete classes (no remaining abstract methods)
        if (
            getattr(cls, "__loom_register__", True)
            and not getattr(cls, "__abstractmethods__", None)
        ):
            Tool._registered_classes.add(cls)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for parameters."""
        ...

    @property
    def timeout_seconds(self) -> int:
        return 30

    @abstractmethod
    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        ...

    def schema(self) -> dict:
        """Return OpenAI-format tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def _resolve_path(self, raw_path: str, workspace: Path) -> Path:
        """Resolve a path relative to workspace with safety check."""
        path = Path(raw_path)
        if not path.is_absolute():
            path = workspace / path
        resolved = path.resolve()
        self._verify_within_workspace(resolved, workspace)
        return resolved

    @staticmethod
    def _verify_within_workspace(path: Path, workspace: Path) -> None:
        """Ensure the resolved path is within the workspace."""
        try:
            path.relative_to(workspace.resolve())
        except ValueError:
            raise ToolSafetyError(f"Path '{path}' escapes workspace '{workspace}'")


class ToolRegistry:
    """Registry for tool registration and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._mcp_refresh_hook: Any = None
        self._mcp_refresh_interval_seconds: float = 30.0
        self._mcp_last_refresh_at: float = 0.0
        self._mcp_refresh_running = False

    def set_mcp_refresh_hook(
        self,
        hook: Any,
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        """Register a best-effort MCP refresh hook for dynamic tool sets."""
        self._mcp_refresh_hook = hook
        self._mcp_refresh_interval_seconds = max(1.0, float(interval_seconds))
        self._mcp_last_refresh_at = 0.0

    def _maybe_refresh_mcp(self, *, force: bool = False) -> None:
        if self._mcp_refresh_hook is None:
            return
        if self._mcp_refresh_running:
            return

        now = time.monotonic()
        if not force and (
            now - self._mcp_last_refresh_at
        ) < self._mcp_refresh_interval_seconds:
            return

        self._mcp_refresh_running = True
        try:
            self._mcp_refresh_hook(force=force)
            self._mcp_last_refresh_at = time.monotonic()
        except Exception as e:
            logging.getLogger(__name__).warning(
                "MCP refresh hook failed: %s",
                e,
            )
        finally:
            self._mcp_refresh_running = False

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if name conflicts."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def exclude(self, name: str) -> bool:
        """Remove a tool. Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    def has(self, name: str) -> bool:
        """Check if tool is registered."""
        if name.startswith("mcp."):
            self._maybe_refresh_mcp()
        return name in self._tools

    async def execute(
        self,
        name: str,
        arguments: dict,
        workspace: Path | None = None,
        scratch_dir: Path | None = None,
        changelog: Any = None,
        subtask_id: str = "",
    ) -> ToolResult:
        """Execute a tool by name with timeout and context."""
        self._maybe_refresh_mcp()
        tool = self._tools.get(name)
        if tool is None and name.startswith("mcp."):
            self._maybe_refresh_mcp(force=True)
            tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(f"Unknown tool: {name}")

        ctx = ToolContext(
            workspace=workspace,
            scratch_dir=scratch_dir,
            changelog=changelog,
            subtask_id=subtask_id,
        )

        try:
            result = await asyncio.wait_for(
                tool.execute(arguments, ctx),
                timeout=tool.timeout_seconds,
            )
            return result
        except TimeoutError:
            return ToolResult.fail(
                f"Tool '{name}' timed out after {tool.timeout_seconds}s"
            )
        except ToolSafetyError as e:
            return ToolResult.fail(f"Safety violation: {e}")
        except Exception as e:
            return ToolResult.fail(f"Tool error: {type(e).__name__}: {e}")

    def all_schemas(self) -> list[dict]:
        """Return all tool schemas for model consumption."""
        self._maybe_refresh_mcp()
        return [tool.schema() for tool in self._tools.values()]

    def list_tools(self) -> list[str]:
        """Return registered tool names."""
        self._maybe_refresh_mcp()
        return list(self._tools.keys())
