"""Tool registry and dispatch system.

Provides registration, argument validation, execution with timeout,
and schema generation for model consumption.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from loom.utils.latency import log_latency_event

_MAX_MCP_AUTH_VIEW_CACHE_ENTRIES = 64


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
    read_roots: list[Path] = field(default_factory=list)
    scratch_dir: Path | None = None
    changelog: Any | None = None  # ChangeLog instance for tracking file modifications
    subtask_id: str = ""
    auth_context: Any | None = None


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

    @property
    def auth_requirements(self) -> list[dict[str, Any]]:
        """Optional auth requirements consumed during run preflight."""
        return []

    @property
    def is_mutating(self) -> bool:
        """Whether this tool mutates local/external state."""
        return False

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
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self._strip_redundant_workspace_prefix(path, workspace)
            path = workspace / path
        resolved = path.resolve()
        self._verify_within_workspace(resolved, workspace)
        return resolved

    def _resolve_read_path(
        self,
        raw_path: str,
        workspace: Path,
        read_roots: list[Path] | None = None,
    ) -> Path:
        """Resolve a read path allowing optional parent/root read scopes.

        Relative read paths prefer the task workspace, then fall back to each
        allowed read root so callers don't need to guess ``../`` prefixes.
        """
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            resolved = path.resolve()
            self._verify_within_allowed_roots(resolved, workspace, read_roots)
            return resolved

        candidates: list[Path] = [workspace / path]
        normalized = self._strip_redundant_workspace_prefix(path, workspace)
        if normalized != path:
            candidates.append(workspace / normalized)
        seen_roots: set[Path] = set()
        for root in read_roots or []:
            try:
                root_path = Path(root).expanduser().resolve()
            except Exception:
                continue
            if root_path in seen_roots:
                continue
            seen_roots.add(root_path)
            candidates.append(root_path / path)
            if normalized != path:
                candidates.append(root_path / normalized)

        first_valid: Path | None = None
        first_error: ToolSafetyError | None = None
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                self._verify_within_allowed_roots(resolved, workspace, read_roots)
            except ToolSafetyError as e:
                if first_error is None:
                    first_error = e
                continue
            if first_valid is None:
                first_valid = resolved
            if resolved.exists():
                return resolved

        if first_valid is not None:
            return first_valid
        if first_error is not None:
            raise first_error

        resolved = (workspace / path).resolve()
        self._verify_within_allowed_roots(resolved, workspace, read_roots)
        return resolved

    @staticmethod
    def _strip_redundant_workspace_prefix(path: Path, workspace: Path) -> Path:
        """Collapse leading ``<workspace_name>/`` segments in relative paths.

        Models sometimes prepend the workspace folder name to workspace-relative
        paths (for example ``run-abc/output.md``). Without normalization that
        creates a nested duplicate workspace directory.
        """
        if path.is_absolute():
            return path
        workspace_name = str(workspace.name or "").strip()
        if not workspace_name:
            return path

        parts = [part for part in path.parts if part not in {"", "."}]
        if not parts:
            return path

        # Strip repeated workspace-name prefixes while preserving non-trivial
        # relative targets (leave plain "workspace_name" untouched).
        removed = 0
        while len(parts) > 1 and parts[0] == workspace_name:
            parts = parts[1:]
            removed += 1
        if removed == 0:
            return path
        return Path(*parts)

    @staticmethod
    def _verify_within_workspace(path: Path, workspace: Path) -> None:
        """Ensure the resolved path is within the workspace."""
        Tool._verify_within_allowed_roots(path, workspace, None)

    @staticmethod
    def _verify_within_allowed_roots(
        path: Path,
        workspace: Path,
        extra_roots: list[Path] | None,
    ) -> None:
        """Ensure path stays inside workspace or one of allowed read roots."""
        roots: list[Path] = [workspace.resolve()]
        if extra_roots:
            for root in extra_roots:
                try:
                    roots.append(Path(root).expanduser().resolve())
                except Exception:
                    continue

        for root in roots:
            try:
                path.relative_to(root)
                return
            except ValueError:
                continue
        roots_text = ", ".join(str(root) for root in roots)
        try:
            workspace_text = str(workspace.resolve())
        except Exception:
            workspace_text = str(workspace)
        raise ToolSafetyError(
            f"Path '{path}' escapes workspace '{workspace_text}'. "
            f"Allowed roots: {roots_text}"
        )


class ToolRegistry:
    """Registry for tool registration and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._tools_lock = threading.RLock()
        self._mcp_refresh_hook: Any = None
        self._mcp_refresh_hook_supports_auth = False
        self._mcp_refresh_interval_seconds: float = 30.0
        self._mcp_last_refresh_at: float = 0.0
        self._mcp_refresh_running = False
        self._mcp_refresh_state_lock = threading.Lock()
        self._mcp_refresh_state_cond = threading.Condition(self._mcp_refresh_state_lock)
        self._mcp_discovery_hook: Any = None
        self._mcp_discovery_hook_supports_auth = False
        self._mcp_discovery_interval_seconds: float = 30.0
        self._mcp_auth_view_cache: dict[str, tuple[float, dict[str, Tool]]] = {}
        self._mcp_discovery_state_lock = threading.Lock()
        self._mcp_discovery_refresh_inflight: set[str] = set()

    def _has_registered_mcp_tools(self) -> bool:
        """Return True when at least one MCP tool is currently registered."""
        with self._tools_lock:
            return any(name.startswith("mcp.") for name in self._tools)

    def set_mcp_refresh_hook(
        self,
        hook: Any,
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        """Register a best-effort MCP refresh hook for dynamic tool sets."""
        self._mcp_refresh_hook = hook
        supports_auth = False
        try:
            params = inspect.signature(hook).parameters
            if "auth_context" in params:
                supports_auth = True
            else:
                supports_auth = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in params.values()
                )
        except (TypeError, ValueError):
            supports_auth = False
        self._mcp_refresh_hook_supports_auth = supports_auth
        self._mcp_refresh_interval_seconds = max(1.0, float(interval_seconds))
        self._mcp_last_refresh_at = 0.0

    def set_mcp_discovery_hook(
        self,
        hook: Any,
        *,
        interval_seconds: float = 30.0,
    ) -> None:
        """Register auth-scoped MCP discovery hook returning tool-name->tool map."""
        self._mcp_discovery_hook = hook
        supports_auth = False
        try:
            params = inspect.signature(hook).parameters
            if "auth_context" in params:
                supports_auth = True
            else:
                supports_auth = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in params.values()
                )
        except (TypeError, ValueError):
            supports_auth = False
        self._mcp_discovery_hook_supports_auth = supports_auth
        self._mcp_discovery_interval_seconds = max(1.0, float(interval_seconds))
        with self._mcp_discovery_state_lock:
            self._mcp_auth_view_cache.clear()
            self._mcp_discovery_refresh_inflight.clear()

    @staticmethod
    def _auth_context_fingerprint(auth_context: Any) -> str:
        if auth_context is None:
            return ""

        fingerprint_fn = getattr(auth_context, "mcp_discovery_fingerprint", None)
        if callable(fingerprint_fn):
            try:
                fingerprint = str(fingerprint_fn() or "").strip()
            except Exception:
                fingerprint = ""
            if fingerprint:
                return fingerprint

        mapping = getattr(auth_context, "selected_by_mcp_alias", None)
        if isinstance(mapping, dict):
            parts: list[str] = []
            for alias, profile in sorted(mapping.items(), key=lambda item: str(item[0])):
                clean_alias = str(alias or "").strip()
                profile_id = str(getattr(profile, "profile_id", "") or "").strip()
                if clean_alias and profile_id:
                    parts.append(f"{clean_alias}:{profile_id}")
            if parts:
                return "|".join(parts)

        return f"context:{id(auth_context)}"

    def _refresh_mcp_discovery_view_sync(
        self,
        *,
        auth_context: Any,
        fingerprint: str,
    ) -> dict[str, Tool]:
        started = time.monotonic()
        with self._mcp_discovery_state_lock:
            cached = self._mcp_auth_view_cache.get(fingerprint)

        try:
            if self._mcp_discovery_hook_supports_auth:
                raw_discovered = self._mcp_discovery_hook(auth_context=auth_context)
            else:
                raw_discovered = self._mcp_discovery_hook()
        except Exception as e:
            logging.getLogger(__name__).warning(
                "MCP discovery hook failed: %s",
                e,
            )
            if cached is not None:
                log_latency_event(
                    logging.getLogger(__name__),
                    event="mcp_discovery_sync",
                    duration_seconds=time.monotonic() - started,
                    fields={"outcome": "error_fallback"},
                )
                return cached[1]
            log_latency_event(
                logging.getLogger(__name__),
                event="mcp_discovery_sync",
                duration_seconds=time.monotonic() - started,
                fields={"outcome": "error_empty"},
            )
            return {}

        discovered: dict[str, Tool] = {}
        if isinstance(raw_discovered, dict):
            for name, tool in raw_discovered.items():
                clean_name = str(name or "").strip()
                if not clean_name.startswith("mcp."):
                    continue
                if not isinstance(tool, Tool):
                    continue
                discovered[clean_name] = tool
        elif isinstance(raw_discovered, list):
            for name in raw_discovered:
                clean_name = str(name or "").strip()
                with self._tools_lock:
                    tool = self._tools.get(clean_name)
                if clean_name.startswith("mcp.") and isinstance(tool, Tool):
                    discovered[clean_name] = tool

        with self._mcp_discovery_state_lock:
            self._mcp_auth_view_cache[fingerprint] = (time.monotonic(), discovered)
            if len(self._mcp_auth_view_cache) > _MAX_MCP_AUTH_VIEW_CACHE_ENTRIES:
                oldest_key = min(
                    self._mcp_auth_view_cache.items(),
                    key=lambda item: float(item[1][0]),
                )[0]
                self._mcp_auth_view_cache.pop(oldest_key, None)
        log_latency_event(
            logging.getLogger(__name__),
            event="mcp_discovery_sync",
            duration_seconds=time.monotonic() - started,
            fields={"outcome": "ok", "tools": len(discovered)},
        )
        return discovered

    def _schedule_mcp_discovery_refresh(
        self,
        *,
        auth_context: Any,
        fingerprint: str,
    ) -> None:
        with self._mcp_discovery_state_lock:
            if fingerprint in self._mcp_discovery_refresh_inflight:
                return
            self._mcp_discovery_refresh_inflight.add(fingerprint)

        def _worker() -> None:
            try:
                self._refresh_mcp_discovery_view_sync(
                    auth_context=auth_context,
                    fingerprint=fingerprint,
                )
            finally:
                with self._mcp_discovery_state_lock:
                    self._mcp_discovery_refresh_inflight.discard(fingerprint)

        thread = threading.Thread(
            target=_worker,
            name=f"loom-mcp-discovery-{fingerprint[:8]}",
            daemon=True,
        )
        thread.start()

    def _discover_mcp_view(
        self,
        *,
        auth_context: Any = None,
        force: bool = False,
        background_on_expiry: bool = False,
    ) -> dict[str, Tool]:
        if self._mcp_discovery_hook is None or auth_context is None:
            return {}

        fingerprint = self._auth_context_fingerprint(auth_context)
        if not fingerprint:
            return {}

        now = time.monotonic()
        with self._mcp_discovery_state_lock:
            cached = self._mcp_auth_view_cache.get(fingerprint)
        if (
            not force
            and cached is not None
            and (now - cached[0]) < self._mcp_discovery_interval_seconds
        ):
            return cached[1]

        if not force and background_on_expiry:
            self._schedule_mcp_discovery_refresh(
                auth_context=auth_context,
                fingerprint=fingerprint,
            )
            if cached is not None:
                return cached[1]
            return {}

        return self._refresh_mcp_discovery_view_sync(
            auth_context=auth_context,
            fingerprint=fingerprint,
        )

    def _run_mcp_refresh(self, *, force: bool, auth_context: Any = None) -> None:
        """Execute MCP refresh hook once and release running state."""
        started = time.monotonic()
        outcome = "ok"
        try:
            if self._mcp_refresh_hook_supports_auth:
                self._mcp_refresh_hook(force=force, auth_context=auth_context)
            else:
                self._mcp_refresh_hook(force=force)
            with self._mcp_refresh_state_lock:
                self._mcp_last_refresh_at = time.monotonic()
        except Exception as e:
            outcome = "error"
            logging.getLogger(__name__).warning(
                "MCP refresh hook failed: %s",
                e,
            )
        finally:
            with self._mcp_refresh_state_cond:
                self._mcp_refresh_running = False
                self._mcp_refresh_state_cond.notify_all()
        log_latency_event(
            logging.getLogger(__name__),
            event="mcp_refresh",
            duration_seconds=time.monotonic() - started,
            fields={"outcome": outcome, "force": force},
        )

    def _maybe_refresh_mcp(
        self,
        *,
        force: bool = False,
        auth_context: Any = None,
        background: bool = False,
        wait_if_running: bool = False,
        wait_timeout_seconds: float = 5.0,
    ) -> None:
        hook = self._mcp_refresh_hook
        if hook is None:
            return

        with self._mcp_refresh_state_cond:
            if self._mcp_refresh_running:
                if not wait_if_running:
                    return
                deadline = time.monotonic() + max(0.0, float(wait_timeout_seconds))
                while self._mcp_refresh_running:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    self._mcp_refresh_state_cond.wait(timeout=remaining)
            now = time.monotonic()
            if not force and (
                now - self._mcp_last_refresh_at
            ) < self._mcp_refresh_interval_seconds:
                return
            self._mcp_refresh_running = True

        if background:
            thread = threading.Thread(
                target=self._run_mcp_refresh,
                kwargs={"force": force, "auth_context": auth_context},
                name="loom-mcp-refresh",
                daemon=True,
            )
            thread.start()
            return

        self._run_mcp_refresh(force=force, auth_context=auth_context)

    def register(self, tool: Tool) -> None:
        """Register a tool. Raises if name conflicts."""
        with self._tools_lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        with self._tools_lock:
            return self._tools.get(name)

    def exclude(self, name: str) -> bool:
        """Remove a tool. Returns True if it existed."""
        with self._tools_lock:
            return self._tools.pop(name, None) is not None

    def has(self, name: str, *, auth_context: Any = None) -> bool:
        """Check if tool is registered."""
        if name.startswith("mcp."):
            if auth_context is not None and self._mcp_discovery_hook is not None:
                discovered = self._discover_mcp_view(
                    auth_context=auth_context,
                    background_on_expiry=True,
                )
                if name in discovered:
                    return True
                discovered = self._discover_mcp_view(
                    auth_context=auth_context,
                    force=True,
                )
                return name in discovered
            with self._tools_lock:
                if name in self._tools:
                    return True
            # Only force-refresh when the requested MCP tool is missing.
            self._maybe_refresh_mcp(force=True, wait_if_running=True)
        with self._tools_lock:
            return name in self._tools

    async def execute(
        self,
        name: str,
        arguments: dict,
        workspace: Path | None = None,
        read_roots: list[Path] | None = None,
        scratch_dir: Path | None = None,
        changelog: Any = None,
        subtask_id: str = "",
        auth_context: Any = None,
    ) -> ToolResult:
        """Execute a tool by name with timeout and context."""
        tool: Tool | None = None
        if name.startswith("mcp.") and auth_context is not None and self._mcp_discovery_hook:
            discovered = self._discover_mcp_view(
                auth_context=auth_context,
                background_on_expiry=True,
            )
            tool = discovered.get(name)
            if tool is None:
                discovered = self._discover_mcp_view(
                    auth_context=auth_context,
                    force=True,
                )
                tool = discovered.get(name)
            if tool is None:
                return ToolResult.fail(f"Unknown tool: {name}")
        if tool is None:
            if name.startswith("mcp."):
                with self._tools_lock:
                    tool = self._tools.get(name)
                if tool is None:
                    # Avoid periodic refresh on every MCP execution. Only refresh
                    # when this specific tool is missing.
                    self._maybe_refresh_mcp(force=True, wait_if_running=True)
                    with self._tools_lock:
                        tool = self._tools.get(name)
            else:
                with self._tools_lock:
                    tool = self._tools.get(name)
            if tool is None and name.startswith("mcp."):
                self._maybe_refresh_mcp(force=True, wait_if_running=True)
                with self._tools_lock:
                    tool = self._tools.get(name)
        if tool is None:
            return ToolResult.fail(f"Unknown tool: {name}")

        normalized_read_roots: list[Path] = []
        for raw in read_roots or []:
            try:
                normalized = Path(raw).expanduser().resolve()
            except Exception:
                continue
            normalized_read_roots.append(normalized)

        ctx = ToolContext(
            workspace=workspace,
            read_roots=normalized_read_roots,
            scratch_dir=scratch_dir,
            changelog=changelog,
            subtask_id=subtask_id,
            auth_context=auth_context,
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

    def all_schemas(self, *, auth_context: Any = None) -> list[dict]:
        """Return all tool schemas for model consumption."""
        if auth_context is not None and self._mcp_discovery_hook is not None:
            with self._tools_lock:
                non_mcp_schemas = [
                    tool.schema()
                    for name, tool in self._tools.items()
                    if not name.startswith("mcp.")
                ]
            discovered = self._discover_mcp_view(
                auth_context=auth_context,
                background_on_expiry=True,
            )
            mcp_schemas = [
                discovered[name].schema()
                for name in sorted(discovered.keys())
            ]
            return [*non_mcp_schemas, *mcp_schemas]
        # Keep reads responsive: only trigger MCP refresh when no MCP tools
        # have been loaded yet.
        if not self._has_registered_mcp_tools():
            self._maybe_refresh_mcp(background=True)
        with self._tools_lock:
            tools = list(self._tools.values())
        return [tool.schema() for tool in tools]

    def list_tools(self, *, auth_context: Any = None) -> list[str]:
        """Return registered tool names."""
        if auth_context is not None and self._mcp_discovery_hook is not None:
            with self._tools_lock:
                non_mcp_tools = [
                    name for name in self._tools.keys()
                    if not name.startswith("mcp.")
                ]
            discovered = self._discover_mcp_view(
                auth_context=auth_context,
                background_on_expiry=True,
            )
            return [*non_mcp_tools, *sorted(discovered.keys())]
        # Keep reads responsive: only trigger MCP refresh when no MCP tools
        # have been loaded yet.
        if not self._has_registered_mcp_tools():
            self._maybe_refresh_mcp(background=True)
        with self._tools_lock:
            return list(self._tools.keys())
