"""Tool registry and dispatch system.

Provides registration, argument validation, execution with timeout,
and schema generation for model consumption.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    data: dict | None = None
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None

    MAX_OUTPUT_SIZE = 30720  # 30KB

    def to_json(self) -> str:
        return json.dumps({
            "success": self.success,
            "output": self.output[:self.MAX_OUTPUT_SIZE],
            "error": self.error,
            "files_changed": self.files_changed,
        })

    @classmethod
    def ok(cls, output: str, **kwargs) -> ToolResult:
        return cls(success=True, output=output, **kwargs)

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(success=False, output="", error=error)


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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only collect concrete classes (no remaining abstract methods)
        if not getattr(cls, "__abstractmethods__", None):
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

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if name conflicts."""
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

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
        return [tool.schema() for tool in self._tools.values()]

    def list_tools(self) -> list[str]:
        """Return registered tool names."""
        return list(self._tools.keys())
