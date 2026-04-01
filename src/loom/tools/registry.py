"""Tool registry and dispatch system.

Provides registration, argument validation, execution with timeout,
and schema generation for model consumption.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from loom.read_scope import normalize_workspace_relative_read_path
from loom.utils.latency import log_latency_event

_MAX_MCP_AUTH_VIEW_CACHE_ENTRIES = 64
_INTERNAL_TOOL_ARG_PREFIX = "_loom_"
ToolAuthMode = Literal["no_auth", "optional_auth", "required_auth"]
_VALID_TOOL_AUTH_MODES = frozenset({"no_auth", "optional_auth", "required_auth"})
ToolExecutionSurface = Literal["tui", "api", "cli"]
_VALID_TOOL_EXECUTION_SURFACES = frozenset({"tui", "api", "cli"})
_DEFAULT_TOOL_EXECUTION_SURFACE: ToolExecutionSurface = "tui"
_DEFAULT_TOOL_EXECUTION_SURFACES: tuple[ToolExecutionSurface, ...] = (
    "tui",
    "api",
    "cli",
)


def normalize_tool_auth_mode(value: object) -> ToolAuthMode:
    """Normalize auth posture to one of the supported mode literals."""
    mode = str(value or "").strip().lower()
    if mode in _VALID_TOOL_AUTH_MODES:
        return mode  # type: ignore[return-value]
    return "no_auth"


def normalize_tool_execution_surface(
    value: object,
    *,
    default: ToolExecutionSurface = _DEFAULT_TOOL_EXECUTION_SURFACE,
) -> ToolExecutionSurface:
    """Normalize runtime execution surface to a supported literal."""
    surface = str(value or "").strip().lower()
    if surface in _VALID_TOOL_EXECUTION_SURFACES:
        return surface  # type: ignore[return-value]
    return default


def normalize_tool_execution_surfaces(value: object) -> tuple[ToolExecutionSurface, ...]:
    """Normalize declared tool execution surfaces with safe defaults."""
    if isinstance(value, str):
        cleaned = normalize_tool_execution_surface(
            value,
            default=_DEFAULT_TOOL_EXECUTION_SURFACE,
        )
        return (cleaned,)
    if isinstance(value, (tuple, list, set)):
        normalized: list[ToolExecutionSurface] = []
        for item in value:
            clean = normalize_tool_execution_surface(
                item,
                default=_DEFAULT_TOOL_EXECUTION_SURFACE,
            )
            if clean not in normalized:
                normalized.append(clean)
        if normalized:
            return tuple(normalized)
    return _DEFAULT_TOOL_EXECUTION_SURFACES


def tool_supports_execution_surface(
    tool: object,
    execution_surface: object,
) -> bool:
    """Return whether a tool can run on the requested execution surface."""
    clean_surface = normalize_tool_execution_surface(
        execution_surface,
        default=_DEFAULT_TOOL_EXECUTION_SURFACE,
    )
    declared = normalize_tool_execution_surfaces(
        getattr(tool, "supported_execution_surfaces", _DEFAULT_TOOL_EXECUTION_SURFACES),
    )
    return clean_surface in declared


def tool_auth_required(tool: object) -> bool:
    """Return whether tool requires or can require credentials."""
    mode = normalize_tool_auth_mode(getattr(tool, "auth_mode", "no_auth"))
    if mode != "no_auth":
        return True
    return bool(getattr(tool, "auth_requirements", []))


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
    read_path_map: dict[str, Path] = field(default_factory=dict)
    scratch_dir: Path | None = None
    changelog: Any | None = None  # ChangeLog instance for tracking file modifications
    subtask_id: str = ""
    auth_context: Any | None = None
    execution_surface: ToolExecutionSurface = _DEFAULT_TOOL_EXECUTION_SURFACE


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
    _AUTO_REGISTER_MODULE_PREFIXES: ClassVar[tuple[str, ...]] = (
        "loom.tools",
        "loom.processes._bundled",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only collect concrete classes (no remaining abstract methods)
        if (
            getattr(cls, "__loom_register__", True)
            and not getattr(cls, "__abstractmethods__", None)
            and cls._should_auto_register()
        ):
            Tool._registered_classes.add(cls)

    @classmethod
    def _should_auto_register(cls) -> bool:
        module_name = str(getattr(cls, "__module__", "") or "").strip()
        if not module_name:
            return False
        return module_name.startswith(cls._AUTO_REGISTER_MODULE_PREFIXES)

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
    def supports_live_timeout_updates(self) -> bool:
        """Whether timeout should be recomputed while a call is in flight."""
        return False

    @property
    def auth_requirements(self) -> list[dict[str, Any]]:
        """Optional auth requirements consumed during run preflight."""
        return []

    @property
    def auth_mode(self) -> ToolAuthMode:
        """Auth posture declaration for this tool."""
        declared = self.auth_requirements
        if declared:
            return "required_auth"
        return "no_auth"

    @property
    def is_mutating(self) -> bool:
        """Whether this tool mutates local/external state."""
        return False

    @property
    def supported_execution_surfaces(self) -> tuple[ToolExecutionSurface, ...]:
        """Declared execution surfaces where this tool is valid."""
        return _DEFAULT_TOOL_EXECUTION_SURFACES

    @abstractmethod
    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        ...

    def schema(self) -> dict:
        """Return OpenAI-format tool definition."""
        surfaces = normalize_tool_execution_surfaces(
            getattr(self, "supported_execution_surfaces", _DEFAULT_TOOL_EXECUTION_SURFACES),
        )
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "x_supported_execution_surfaces": list(surfaces),
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
        read_path_map: dict[str, Path] | None = None,
    ) -> Path:
        """Resolve a read path allowing optional parent/root read scopes.

        Relative read paths prefer the task workspace, then fall back to each
        allowed read root so callers don't need to guess ``../`` prefixes.
        """
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            resolved = path.resolve()
            self._verify_within_allowed_roots(
                resolved,
                workspace,
                read_roots,
                read_path_map=read_path_map,
            )
            return resolved

        candidates: list[Path] = [workspace / path]
        normalized = self._strip_redundant_workspace_prefix(path, workspace)
        if normalized != path:
            candidates.append(workspace / normalized)
        normalized_texts = [
            text
            for text in {
                normalize_workspace_relative_read_path(path),
                normalize_workspace_relative_read_path(normalized),
            }
            if text
        ]
        for key, target in (read_path_map or {}).items():
            try:
                target_path = Path(target).expanduser().resolve()
            except Exception:
                continue
            for text in normalized_texts:
                if text == key:
                    candidates.append(target_path)
                    continue
                if not target_path.is_dir():
                    continue
                prefix = f"{key}/"
                if text.startswith(prefix):
                    suffix = text[len(prefix):]
                    if suffix:
                        candidates.append(target_path / suffix)
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
                self._verify_within_allowed_roots(
                    resolved,
                    workspace,
                    read_roots,
                    read_path_map=read_path_map,
                )
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
        self._verify_within_allowed_roots(
            resolved,
            workspace,
            read_roots,
            read_path_map=read_path_map,
        )
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
        *,
        read_path_map: dict[str, Path] | None = None,
    ) -> None:
        """Ensure path stays inside workspace or one of allowed read roots."""
        roots: list[Path] = [workspace.resolve()]
        exact_paths: list[Path] = []
        if extra_roots:
            for root in extra_roots:
                try:
                    resolved_root = Path(root).expanduser().resolve()
                except Exception:
                    continue
                if resolved_root.is_dir():
                    roots.append(resolved_root)
                else:
                    exact_paths.append(resolved_root)
        if read_path_map:
            for target in read_path_map.values():
                try:
                    resolved_target = Path(target).expanduser().resolve()
                except Exception:
                    continue
                if resolved_target.is_dir():
                    roots.append(resolved_target)
                else:
                    exact_paths.append(resolved_target)

        for root in roots:
            try:
                path.relative_to(root)
                return
            except ValueError:
                continue
        for exact in exact_paths:
            if path == exact:
                return
        roots_text = ", ".join(str(root) for root in roots)
        exact_text = ", ".join(str(path_item) for path_item in exact_paths)
        try:
            workspace_text = str(workspace.resolve())
        except Exception:
            workspace_text = str(workspace)
        raise ToolSafetyError(
            f"Path '{path}' escapes workspace '{workspace_text}'. "
            f"Allowed roots: {roots_text}"
            + (f"; allowed exact paths: {exact_text}" if exact_text else "")
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
        self._validate_tool_auth_contract(tool)
        with self._tools_lock:
            if tool.name in self._tools:
                raise ValueError(f"Tool already registered: {tool.name}")
            self._tools[tool.name] = tool

    @staticmethod
    def _validate_tool_auth_contract(tool: Tool) -> None:
        """Validate declared auth posture against requirements payload."""
        mode_raw = getattr(tool, "auth_mode", "no_auth")
        mode = str(mode_raw or "").strip().lower()
        if mode not in _VALID_TOOL_AUTH_MODES:
            raise ValueError(
                f"Tool '{tool.name}' has invalid auth_mode={mode_raw!r}. "
                "Expected one of: no_auth, optional_auth, required_auth.",
            )
        declared = getattr(tool, "auth_requirements", [])
        if declared is None:
            declared = []
        if not isinstance(declared, list):
            raise ValueError(
                f"Tool '{tool.name}' auth_requirements must be a list, got "
                f"{type(declared).__name__}.",
            )
        has_requirements = bool(declared)
        if mode == "no_auth" and has_requirements:
            raise ValueError(
                f"Tool '{tool.name}' is auth_mode='no_auth' but declares "
                "auth_requirements.",
            )
        if mode in {"optional_auth", "required_auth"} and not has_requirements:
            raise ValueError(
                f"Tool '{tool.name}' is auth_mode='{mode}' but does not declare "
                "auth_requirements.",
            )

    def get(self, name: str) -> Tool | None:
        with self._tools_lock:
            return self._tools.get(name)

    def exclude(self, name: str) -> bool:
        """Remove a tool. Returns True if it existed."""
        with self._tools_lock:
            return self._tools.pop(name, None) is not None

    @staticmethod
    def _requested_execution_surface(value: object) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        return normalize_tool_execution_surface(text)

    def has(
        self,
        name: str,
        *,
        auth_context: Any = None,
        execution_surface: str | None = None,
    ) -> bool:
        """Check if tool is registered."""
        requested_surface = self._requested_execution_surface(execution_surface)
        if name.startswith("mcp."):
            if auth_context is not None and self._mcp_discovery_hook is not None:
                discovered = self._discover_mcp_view(
                    auth_context=auth_context,
                    background_on_expiry=True,
                )
                tool = discovered.get(name)
                if tool is not None:
                    return (
                        not requested_surface
                        or tool_supports_execution_surface(tool, requested_surface)
                    )
                if name in discovered:
                    return True
                discovered = self._discover_mcp_view(
                    auth_context=auth_context,
                    force=True,
                )
                tool = discovered.get(name)
                if tool is None:
                    return False
                if requested_surface and not tool_supports_execution_surface(
                    tool,
                    requested_surface,
                ):
                    return False
                return True
            with self._tools_lock:
                tool = self._tools.get(name)
                if tool is not None:
                    if requested_surface and not tool_supports_execution_surface(
                        tool,
                        requested_surface,
                    ):
                        return False
                    return True
            # Only force-refresh when the requested MCP tool is missing.
            self._maybe_refresh_mcp(force=True, wait_if_running=True)
        with self._tools_lock:
            tool = self._tools.get(name)
        if tool is None:
            return False
        if requested_surface and not tool_supports_execution_surface(
            tool,
            requested_surface,
        ):
            return False
        return True

    async def _execute_with_live_timeout_monitor(
        self,
        *,
        tool: Tool,
        name: str,
        normalized_arguments: dict,
        ctx: ToolContext,
    ) -> ToolResult:
        """Execute one tool while recomputing timeout against elapsed time."""
        started_at = time.monotonic()
        task = asyncio.create_task(tool.execute(normalized_arguments, ctx))
        poll_interval_seconds = 0.25
        try:
            while True:
                timeout_seconds = max(1.0, float(tool.timeout_seconds))
                elapsed = max(0.0, time.monotonic() - started_at)
                if elapsed >= timeout_seconds:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    raise TimeoutError
                remaining = timeout_seconds - elapsed
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=max(0.05, min(poll_interval_seconds, remaining)),
                    )
                except TimeoutError:
                    continue
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def execute(
        self,
        name: str,
        arguments: dict,
        workspace: Path | None = None,
        read_roots: list[Path] | None = None,
        read_path_map: dict[str, Path] | dict[str, str] | None = None,
        scratch_dir: Path | None = None,
        changelog: Any = None,
        subtask_id: str = "",
        auth_context: Any = None,
        allow_internal_args: bool = False,
        execution_surface: str | None = None,
    ) -> ToolResult:
        """Execute a tool by name with timeout and context."""
        tool: Tool | None = None
        requested_surface = self._requested_execution_surface(execution_surface)
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
        if requested_surface and not tool_supports_execution_surface(
            tool,
            requested_surface,
        ):
            return ToolResult.fail(
                (
                    f"Tool '{name}' is not available on execution surface "
                    f"'{requested_surface}'."
                ),
            )

        normalized_read_roots: list[Path] = []
        for raw in read_roots or []:
            try:
                normalized = Path(raw).expanduser().resolve()
            except Exception:
                continue
            normalized_read_roots.append(normalized)
        normalized_read_path_map: dict[str, Path] = {}
        if isinstance(read_path_map, dict):
            for raw_key, raw_value in read_path_map.items():
                key = normalize_workspace_relative_read_path(raw_key)
                if not key:
                    continue
                try:
                    normalized_value = Path(raw_value).expanduser().resolve()
                except Exception:
                    continue
                normalized_read_path_map[key] = normalized_value

        normalized_arguments: dict
        if isinstance(arguments, dict):
            normalized_arguments = dict(arguments)
        else:
            normalized_arguments = {}
        if not allow_internal_args:
            normalized_arguments = self._sanitize_public_arguments(
                name,
                normalized_arguments,
            )

        ctx = ToolContext(
            workspace=workspace,
            read_roots=normalized_read_roots,
            read_path_map=normalized_read_path_map,
            scratch_dir=scratch_dir,
            changelog=changelog,
            subtask_id=subtask_id,
            auth_context=auth_context,
            execution_surface=(
                normalize_tool_execution_surface(requested_surface)
                if requested_surface
                else _DEFAULT_TOOL_EXECUTION_SURFACE
            ),
        )

        try:
            if getattr(tool, "supports_live_timeout_updates", False):
                result = await self._execute_with_live_timeout_monitor(
                    tool=tool,
                    name=name,
                    normalized_arguments=normalized_arguments,
                    ctx=ctx,
                )
            else:
                result = await asyncio.wait_for(
                    tool.execute(normalized_arguments, ctx),
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

    @staticmethod
    def _sanitize_public_arguments(tool_name: str, args: dict) -> dict:
        """Strip reserved internal runtime controls from external tool args."""
        cleaned: dict = {}
        for key, value in args.items():
            if str(key or "").startswith(_INTERNAL_TOOL_ARG_PREFIX):
                continue
            cleaned[key] = value

        # wp high-risk confirmation is host-owned and not model/public controlled.
        if str(tool_name or "").strip() == "wp_cli":
            cleaned.pop("confirm_high_risk", None)

        return cleaned

    def all_schemas(
        self,
        *,
        auth_context: Any = None,
        execution_surface: str | None = None,
    ) -> list[dict]:
        """Return all tool schemas for model consumption."""
        requested_surface = self._requested_execution_surface(execution_surface)
        if auth_context is not None and self._mcp_discovery_hook is not None:
            with self._tools_lock:
                non_mcp_schemas = [
                    tool.schema()
                    for name, tool in self._tools.items()
                    if not name.startswith("mcp.")
                    and (
                        not requested_surface
                        or tool_supports_execution_surface(tool, requested_surface)
                    )
                ]
            discovered = self._discover_mcp_view(
                auth_context=auth_context,
                background_on_expiry=True,
            )
            mcp_schemas = [
                discovered[name].schema()
                for name in sorted(discovered.keys())
                if (
                    not requested_surface
                    or tool_supports_execution_surface(
                        discovered[name],
                        requested_surface,
                    )
                )
            ]
            return [*non_mcp_schemas, *mcp_schemas]
        # Keep reads responsive: only trigger MCP refresh when no MCP tools
        # have been loaded yet.
        if not self._has_registered_mcp_tools():
            self._maybe_refresh_mcp(background=True)
        with self._tools_lock:
            tools = list(self._tools.values())
        return [
            tool.schema()
            for tool in tools
            if (
                not requested_surface
                or tool_supports_execution_surface(tool, requested_surface)
            )
        ]

    def list_tools(
        self,
        *,
        auth_context: Any = None,
        execution_surface: str | None = None,
    ) -> list[str]:
        """Return registered tool names."""
        requested_surface = self._requested_execution_surface(execution_surface)
        if auth_context is not None and self._mcp_discovery_hook is not None:
            with self._tools_lock:
                non_mcp_tools = [
                    name
                    for name, tool in self._tools.items()
                    if not name.startswith("mcp.")
                    and (
                        not requested_surface
                        or tool_supports_execution_surface(tool, requested_surface)
                    )
                ]
            discovered = self._discover_mcp_view(
                auth_context=auth_context,
                background_on_expiry=True,
            )
            return [
                *non_mcp_tools,
                *[
                    name
                    for name in sorted(discovered.keys())
                    if (
                        not requested_surface
                        or tool_supports_execution_surface(
                            discovered[name],
                            requested_surface,
                        )
                    )
                ],
            ]
        # Keep reads responsive: only trigger MCP refresh when no MCP tools
        # have been loaded yet.
        if not self._has_registered_mcp_tools():
            self._maybe_refresh_mcp(background=True)
        with self._tools_lock:
            return [
                name
                for name, tool in self._tools.items()
                if (
                    not requested_surface
                    or tool_supports_execution_surface(tool, requested_surface)
                )
            ]
