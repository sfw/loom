"""Loom exception hierarchy.

Provides a structured exception tree so catch blocks can be specific
and callers can distinguish between different failure modes.
"""

from __future__ import annotations


class LoomError(Exception):
    """Base for all Loom exceptions."""


class EngineError(LoomError):
    """Orchestrator, runner, scheduler failures."""


class ModelError(LoomError):
    """Provider connection, timeout, parse failures."""


class ToolError(LoomError):
    """Tool execution failures."""


class StateError(LoomError):
    """Persistence, memory, session state failures."""
