"""Abstract model interface.

All model providers implement this interface, providing a unified
API for completions, tool calling, and health checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A parsed tool call from a model response."""

    id: str
    name: str
    arguments: dict


@dataclass
class TokenUsage:
    """Token usage statistics for a model response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ModelResponse:
    """Structured response from a model completion."""

    text: str
    tool_calls: list[ToolCall] | None = None
    raw: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    latency_ms: int = 0

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ModelProvider(ABC):
    """Abstract base class for all model providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ModelResponse:
        """Send a completion request and return structured response."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model is available and responding."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...

    @property
    @abstractmethod
    def tier(self) -> int:
        """Model capability tier (1=fast/small, 2=medium, 3=large)."""
        ...

    @property
    @abstractmethod
    def roles(self) -> list[str]:
        """Roles this provider can fulfill."""
        ...


class ModelNotAvailableError(Exception):
    """Raised when no suitable model is available for a request."""
