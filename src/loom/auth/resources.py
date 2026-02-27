"""Resource-first auth registry, bindings, and draft sync helpers."""

from __future__ import annotations

import os
import re
import shutil
import tomllib
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loom.auth.config import (
    AuthProfile,
    default_workspace_auth_defaults_path,
    load_auth_file,
    load_merged_auth_config,
    remove_auth_profile,
    resolve_auth_write_path,
    upsert_auth_profile,
    write_auth_file,
)
from loom.mcp.config import MCPConfigManager

if TYPE_CHECKING:
    from loom.tools.registry import ToolRegistry


class AuthResourceError(Exception):
    """Raised when auth resource registry parsing/mutation fails."""


@dataclass(frozen=True)
class AuthResource:
    """One auth-requiring resource known to the workspace."""

    resource_id: str
    resource_kind: str  # mcp | tool | api_integration
    resource_key: str
    display_name: str
    provider: str
    source: str = "api"  # api | mcp
    modes: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    required_env_keys: tuple[str, ...] = ()
    status: str = "active"  # active | deleted
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""

    @property
    def resource_ref(self) -> str:
        return f"{self.resource_kind}:{self.resource_key}"


@dataclass(frozen=True)
class AuthBinding:
    """One link between a resource and an auth profile/account."""

    binding_id: str
    resource_id: str
    profile_id: str
    is_default_workspace: bool = False
    priority: int = 0
    generated_from: str = ""
    status: str = "active"  # active | deleted
    created_at: str = ""
    updated_at: str = ""
    deleted_at: str = ""


@dataclass(frozen=True)
class AuthResourcesStore:
    """Workspace-scoped auth resource registry store."""

    schema_version: int = 1
    resources: dict[str, AuthResource] = field(default_factory=dict)
    bindings: dict[str, AuthBinding] = field(default_factory=dict)
    workspace_defaults: dict[str, str] = field(default_factory=dict)  # resource_id -> profile_id
    tombstones: dict[str, dict[str, str]] = field(default_factory=dict)
    pending_operations: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscoveredAuthResource:
    """Discovered auth contract for one resource candidate."""

    resource_kind: str
    resource_key: str
    display_name: str
    provider: str
    source: str
    modes: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    required_env_keys: tuple[str, ...] = ()
    preferred_resource_id: str = ""

    @property
    def discovery_key(self) -> str:
        return f"{self.resource_kind}:{self.resource_key}"


@dataclass(frozen=True)
class AuthDraftSyncResult:
    """Result summary for draft sync operations."""

    created_resources: int = 0
    updated_resources: int = 0
    created_drafts: int = 0
    created_bindings: int = 0
    updated_defaults: int = 0
    warnings: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return (
            self.created_resources > 0
            or self.updated_resources > 0
            or self.created_drafts > 0
            or self.created_bindings > 0
            or self.updated_defaults > 0
        )


@dataclass(frozen=True)
class AuthMigrationResult:
    """Summary of legacy->resource migration."""

    snapshot_path: Path
    created_resources: int = 0
    created_bindings: int = 0
    created_workspace_defaults: int = 0
    created_user_resource_defaults: int = 0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthAuditReport:
    """Auth resource lifecycle audit report."""

    orphaned_profiles: tuple[str, ...] = ()
    orphaned_bindings: tuple[str, ...] = ()
    deleted_resource_bindings: tuple[str, ...] = ()
    legacy_provider_defaults: tuple[str, ...] = ()
    dangling_workspace_resource_defaults: tuple[str, ...] = ()
    dangling_user_resource_defaults: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceDeleteImpact:
    """Impact summary shown before deleting a resource."""

    resource_id: str = ""
    active_binding_ids: tuple[str, ...] = ()
    active_profile_ids: tuple[str, ...] = ()
    workspace_default_profile_id: str = ""
    referencing_processes: tuple[str, ...] = ()


_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SAFE_ID_RE = re.compile(r"[^a-z0-9_]+")
_SUPPORTED_MODES = (
    "api_key",
    "oauth2_pkce",
    "oauth2_device",
    "env_passthrough",
    "cli_passthrough",
)


def _toml_escape(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _toml_key(value: str) -> str:
    text = str(value)
    if _TOML_BARE_KEY_RE.fullmatch(text):
        return text
    return _toml_escape(text)


def _toml_array(values: tuple[str, ...] | list[str]) -> str:
    return "[" + ", ".join(_toml_escape(str(v)) for v in values) + "]"


def _iso_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_str_tuple(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple):
        return ()
    values: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text:
            values.append(text)
    return tuple(values)


def _normalize_mode_tuple(raw: object) -> tuple[str, ...]:
    normalized = []
    for mode in _normalize_str_tuple(raw):
        lower = mode.lower()
        if lower in _SUPPORTED_MODES:
            normalized.append(lower)
    return tuple(dict.fromkeys(normalized))


def _parse_bool(raw: object, *, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(raw: object, *, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return default


def _resource_sort_key(resource: AuthResource) -> tuple[str, str, str]:
    return (
        str(resource.resource_kind).strip(),
        str(resource.resource_key).strip(),
        str(resource.resource_id).strip(),
    )


def _binding_sort_key(binding: AuthBinding) -> tuple[int, str]:
    return (_parse_int(binding.priority), str(binding.profile_id).strip())


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        try:
            import fcntl  # POSIX only

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            # Best-effort on platforms without fcntl.
            pass
        try:
            yield
        finally:
            try:
                import fcntl  # POSIX only

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def default_workspace_auth_resources_path(workspace: Path) -> Path:
    """Default workspace auth resources path."""
    return workspace / ".loom" / "auth.resources.toml"


def _parse_resource(resource_id: str, raw: object, *, path: Path) -> AuthResource:
    if not isinstance(raw, dict):
        raise AuthResourceError(
            f"Invalid resource {resource_id!r} in {path}: expected table/dict."
        )
    resource_kind = str(raw.get("kind", "")).strip().lower() or "api_integration"
    if resource_kind not in {"mcp", "tool", "api_integration"}:
        raise AuthResourceError(
            f"Invalid resource kind {resource_kind!r} for {resource_id!r} in {path}."
        )
    status = str(raw.get("status", "active")).strip().lower() or "active"
    if status not in {"active", "deleted"}:
        raise AuthResourceError(
            f"Invalid resource status {status!r} for {resource_id!r} in {path}."
        )
    return AuthResource(
        resource_id=resource_id,
        resource_kind=resource_kind,
        resource_key=str(raw.get("key", "")).strip(),
        display_name=str(raw.get("display_name", "")).strip(),
        provider=str(raw.get("provider", "")).strip(),
        source=str(raw.get("source", "api")).strip().lower() or "api",
        modes=_normalize_mode_tuple(raw.get("modes", [])),
        scopes=_normalize_str_tuple(raw.get("scopes", [])),
        required_env_keys=_normalize_str_tuple(raw.get("required_env_keys", [])),
        status=status,
        created_at=str(raw.get("created_at", "")).strip(),
        updated_at=str(raw.get("updated_at", "")).strip(),
        deleted_at=str(raw.get("deleted_at", "")).strip(),
    )


def _parse_binding(binding_id: str, raw: object, *, path: Path) -> AuthBinding:
    if not isinstance(raw, dict):
        raise AuthResourceError(
            f"Invalid binding {binding_id!r} in {path}: expected table/dict."
        )
    status = str(raw.get("status", "active")).strip().lower() or "active"
    if status not in {"active", "deleted"}:
        raise AuthResourceError(
            f"Invalid binding status {status!r} for {binding_id!r} in {path}."
        )
    return AuthBinding(
        binding_id=binding_id,
        resource_id=str(raw.get("resource_id", "")).strip(),
        profile_id=str(raw.get("profile_id", "")).strip(),
        is_default_workspace=_parse_bool(raw.get("is_default_workspace"), default=False),
        priority=_parse_int(raw.get("priority", 0), default=0),
        generated_from=str(raw.get("generated_from", "")).strip(),
        status=status,
        created_at=str(raw.get("created_at", "")).strip(),
        updated_at=str(raw.get("updated_at", "")).strip(),
        deleted_at=str(raw.get("deleted_at", "")).strip(),
    )


def load_workspace_auth_resources(path: Path) -> AuthResourcesStore:
    """Load workspace auth resource registry."""
    if not path.exists():
        return AuthResourcesStore()

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise AuthResourceError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise AuthResourceError(f"Cannot read auth resources {path}: {e}") from e

    if not isinstance(raw, dict):
        raise AuthResourceError(f"Invalid auth resources file {path}: expected table.")

    schema_version = _parse_int(raw.get("schema_version", 1), default=1)
    if schema_version <= 0:
        schema_version = 1

    resources_raw = raw.get("resources", {})
    if resources_raw is None:
        resources_raw = {}
    if not isinstance(resources_raw, dict):
        raise AuthResourceError(f"Invalid [resources] in {path}: expected table.")

    bindings_raw = raw.get("bindings", {})
    if bindings_raw is None:
        bindings_raw = {}
    if not isinstance(bindings_raw, dict):
        raise AuthResourceError(f"Invalid [bindings] in {path}: expected table.")

    defaults_raw = raw.get("defaults", {})
    if defaults_raw is None:
        defaults_raw = {}
    if not isinstance(defaults_raw, dict):
        defaults_raw = {}
    workspace_defaults_raw = defaults_raw.get("workspace", {})
    if not isinstance(workspace_defaults_raw, dict):
        workspace_defaults_raw = {}

    tombstones_raw = raw.get("tombstones", {})
    if tombstones_raw is None:
        tombstones_raw = {}
    if not isinstance(tombstones_raw, dict):
        tombstones_raw = {}
    pending_raw = raw.get("pending", {})
    if pending_raw is None:
        pending_raw = {}
    if not isinstance(pending_raw, dict):
        pending_raw = {}

    resources: dict[str, AuthResource] = {}
    for resource_id_raw, entry in resources_raw.items():
        resource_id = str(resource_id_raw or "").strip()
        if not resource_id:
            continue
        resource = _parse_resource(resource_id, entry, path=path)
        if not resource.resource_key:
            continue
        resources[resource_id] = resource

    bindings: dict[str, AuthBinding] = {}
    for binding_id_raw, entry in bindings_raw.items():
        binding_id = str(binding_id_raw or "").strip()
        if not binding_id:
            continue
        binding = _parse_binding(binding_id, entry, path=path)
        if not binding.resource_id or not binding.profile_id:
            continue
        bindings[binding_id] = binding

    workspace_defaults: dict[str, str] = {}
    for resource_id_raw, profile_id_raw in workspace_defaults_raw.items():
        resource_id = str(resource_id_raw or "").strip()
        profile_id = str(profile_id_raw or "").strip()
        if resource_id and profile_id:
            workspace_defaults[resource_id] = profile_id

    tombstones: dict[str, dict[str, str]] = {}
    for resource_id_raw, entry in tombstones_raw.items():
        resource_id = str(resource_id_raw or "").strip()
        if not resource_id:
            continue
        if not isinstance(entry, dict):
            continue
        payload = {
            str(key): str(value)
            for key, value in entry.items()
            if isinstance(key, str)
        }
        if payload:
            tombstones[resource_id] = payload

    pending_operations: dict[str, dict[str, str]] = {}
    for op_id_raw, entry in pending_raw.items():
        op_id = str(op_id_raw or "").strip()
        if not op_id:
            continue
        if not isinstance(entry, dict):
            continue
        payload = {
            str(key): str(value)
            for key, value in entry.items()
            if isinstance(key, str)
        }
        if payload:
            pending_operations[op_id] = payload

    return AuthResourcesStore(
        schema_version=schema_version,
        resources=resources,
        bindings=bindings,
        workspace_defaults=workspace_defaults,
        tombstones=tombstones,
        pending_operations=pending_operations,
    )


def render_workspace_auth_resources(store: AuthResourcesStore) -> str:
    """Render auth.resources.toml content."""
    lines: list[str] = [
        "# Loom workspace auth resource registry",
        "# Resource-first links between auth profiles and MCP/API/tool requirements.",
        "",
        f"schema_version = {_parse_int(store.schema_version, default=1)}",
        "",
    ]

    for resource in sorted(store.resources.values(), key=_resource_sort_key):
        lines.append(f"[resources.{_toml_key(resource.resource_id)}]")
        lines.append(f"kind = {_toml_escape(resource.resource_kind)}")
        lines.append(f"key = {_toml_escape(resource.resource_key)}")
        lines.append(f"display_name = {_toml_escape(resource.display_name)}")
        lines.append(f"provider = {_toml_escape(resource.provider)}")
        lines.append(f"source = {_toml_escape(resource.source)}")
        if resource.modes:
            lines.append(f"modes = {_toml_array(resource.modes)}")
        if resource.scopes:
            lines.append(f"scopes = {_toml_array(resource.scopes)}")
        if resource.required_env_keys:
            lines.append(
                f"required_env_keys = {_toml_array(resource.required_env_keys)}"
            )
        lines.append(f"status = {_toml_escape(resource.status)}")
        if resource.created_at:
            lines.append(f"created_at = {_toml_escape(resource.created_at)}")
        if resource.updated_at:
            lines.append(f"updated_at = {_toml_escape(resource.updated_at)}")
        if resource.deleted_at:
            lines.append(f"deleted_at = {_toml_escape(resource.deleted_at)}")
        lines.append("")

    for binding in sorted(store.bindings.values(), key=lambda b: str(b.binding_id)):
        lines.append(f"[bindings.{_toml_key(binding.binding_id)}]")
        lines.append(f"resource_id = {_toml_escape(binding.resource_id)}")
        lines.append(f"profile_id = {_toml_escape(binding.profile_id)}")
        if binding.priority:
            lines.append(f"priority = {binding.priority}")
        if binding.is_default_workspace:
            lines.append("is_default_workspace = true")
        if binding.generated_from:
            lines.append(f"generated_from = {_toml_escape(binding.generated_from)}")
        lines.append(f"status = {_toml_escape(binding.status)}")
        if binding.created_at:
            lines.append(f"created_at = {_toml_escape(binding.created_at)}")
        if binding.updated_at:
            lines.append(f"updated_at = {_toml_escape(binding.updated_at)}")
        if binding.deleted_at:
            lines.append(f"deleted_at = {_toml_escape(binding.deleted_at)}")
        lines.append("")

    lines.append("[defaults.workspace]")
    for resource_id in sorted(store.workspace_defaults):
        profile_id = str(store.workspace_defaults.get(resource_id, "")).strip()
        if profile_id:
            lines.append(f"{_toml_key(resource_id)} = {_toml_escape(profile_id)}")
    lines.append("")

    for resource_id in sorted(store.tombstones):
        payload = store.tombstones[resource_id]
        lines.append(f"[tombstones.{_toml_key(resource_id)}]")
        for key in sorted(payload):
            value = str(payload.get(key, "")).strip()
            if value:
                lines.append(f"{_toml_key(key)} = {_toml_escape(value)}")
        lines.append("")

    for op_id in sorted(store.pending_operations):
        payload = store.pending_operations[op_id]
        lines.append(f"[pending.{_toml_key(op_id)}]")
        for key in sorted(payload):
            value = str(payload.get(key, "")).strip()
            if value:
                lines.append(f"{_toml_key(key)} = {_toml_escape(value)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_workspace_auth_resources(path: Path, store: AuthResourcesStore) -> None:
    """Atomically write workspace auth resource registry."""
    _atomic_write_text(path, render_workspace_auth_resources(store))


def mutate_workspace_auth_resources(
    path: Path,
    mutator,
) -> AuthResourcesStore:
    """Mutate auth resources under a file lock."""
    lock_path = path.with_name(f".{path.name}.lock")
    with _file_lock(lock_path):
        current = load_workspace_auth_resources(path)
        updated = mutator(current)
        if updated != current:
            write_workspace_auth_resources(path, updated)
        return updated


def _parse_resource_ref(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    if ":" not in text:
        return "", ""
    kind, key = text.split(":", 1)
    return kind.strip().lower(), key.strip()


def resolve_resource(
    store: AuthResourcesStore,
    *,
    resource_id: str = "",
    resource_ref: str = "",
    source: str = "",
    provider: str = "",
    mcp_server: str = "",
) -> AuthResource | None:
    """Resolve one active resource by id/ref/fallback selectors."""
    clean_resource_id = str(resource_id or "").strip()
    if clean_resource_id:
        resource = store.resources.get(clean_resource_id)
        if resource is not None and resource.status == "active":
            return resource

    kind, key = _parse_resource_ref(resource_ref)
    if kind and key:
        for resource in store.resources.values():
            if resource.status != "active":
                continue
            if resource.resource_kind == kind and resource.resource_key == key:
                return resource

    clean_source = str(source or "").strip().lower()
    clean_mcp_server = str(mcp_server or "").strip()
    if clean_source == "mcp" and clean_mcp_server:
        for resource in store.resources.values():
            if resource.status != "active":
                continue
            if resource.resource_kind == "mcp" and resource.resource_key == clean_mcp_server:
                return resource

    clean_provider = str(provider or "").strip()
    if clean_provider:
        matches = [
            resource
            for resource in store.resources.values()
            if resource.status == "active" and resource.provider == clean_provider
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def active_bindings_for_resource(
    store: AuthResourcesStore,
    resource_id: str,
) -> list[AuthBinding]:
    """Return active bindings for a resource ordered by priority then profile id."""
    bindings = [
        binding
        for binding in store.bindings.values()
        if binding.status == "active"
        and binding.resource_id == resource_id
        and str(binding.profile_id).strip()
    ]
    return sorted(bindings, key=_binding_sort_key)


def has_active_binding(
    store: AuthResourcesStore,
    *,
    resource_id: str,
    profile_id: str,
) -> bool:
    """Whether an active binding exists for resource/profile."""
    clean_resource_id = str(resource_id or "").strip()
    clean_profile_id = str(profile_id or "").strip()
    if not clean_resource_id or not clean_profile_id:
        return False
    for binding in store.bindings.values():
        if binding.status != "active":
            continue
        if binding.resource_id == clean_resource_id and binding.profile_id == clean_profile_id:
            return True
    return False


def _coerce_requirement(raw: object) -> dict[str, object] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        provider = str(raw.get("provider", "")).strip()
        source = str(raw.get("source", "api")).strip().lower() or "api"
        modes = _normalize_mode_tuple(raw.get("modes", []))
        scopes = _normalize_str_tuple(raw.get("scopes", []))
        env_keys = _normalize_str_tuple(raw.get("required_env_keys", []))
        mcp_server = str(raw.get("mcp_server", "")).strip()
        resource_ref = str(raw.get("resource_ref", "")).strip()
        resource_id = str(raw.get("resource_id", "")).strip()
        return {
            "provider": provider,
            "source": source,
            "modes": modes,
            "scopes": scopes,
            "required_env_keys": env_keys,
            "mcp_server": mcp_server,
            "resource_ref": resource_ref,
            "resource_id": resource_id,
        }

    provider = str(getattr(raw, "provider", "")).strip()
    source = str(getattr(raw, "source", "api")).strip().lower() or "api"
    modes = _normalize_mode_tuple(getattr(raw, "modes", []))
    scopes = _normalize_str_tuple(getattr(raw, "scopes", []))
    env_keys = _normalize_str_tuple(getattr(raw, "required_env_keys", []))
    mcp_server = str(getattr(raw, "mcp_server", "")).strip()
    resource_ref = str(getattr(raw, "resource_ref", "")).strip()
    resource_id = str(getattr(raw, "resource_id", "")).strip()
    return {
        "provider": provider,
        "source": source,
        "modes": modes,
        "scopes": scopes,
        "required_env_keys": env_keys,
        "mcp_server": mcp_server,
        "resource_ref": resource_ref,
        "resource_id": resource_id,
    }


def _make_discovered_resource(
    requirement: dict[str, object],
    *,
    generated_from: str = "",
) -> DiscoveredAuthResource | None:
    provider = str(requirement.get("provider", "")).strip()
    source = str(requirement.get("source", "api")).strip().lower() or "api"
    resource_ref = str(requirement.get("resource_ref", "")).strip()
    preferred_resource_id = str(requirement.get("resource_id", "")).strip()
    mcp_server = str(requirement.get("mcp_server", "")).strip()
    modes = tuple(requirement.get("modes", ()) or ())
    scopes = tuple(requirement.get("scopes", ()) or ())
    env_keys = tuple(requirement.get("required_env_keys", ()) or ())

    kind = ""
    key = ""
    if resource_ref:
        parsed_kind, parsed_key = _parse_resource_ref(resource_ref)
        kind, key = parsed_kind, parsed_key
    if not kind or not key:
        if source == "mcp":
            kind = "mcp"
            key = mcp_server or provider
        else:
            kind = "api_integration"
            key = provider

    if not kind or not key:
        return None
    if kind not in {"mcp", "tool", "api_integration"}:
        return None
    resolved_provider = provider or key
    if kind == "mcp":
        display_name = f"MCP: {key}"
    elif kind == "tool":
        display_name = f"Tool: {key}"
    else:
        display_name = f"API: {resolved_provider}"

    if generated_from:
        display_name = display_name
    return DiscoveredAuthResource(
        resource_kind=kind,
        resource_key=key,
        display_name=display_name,
        provider=resolved_provider,
        source="mcp" if kind == "mcp" else "api",
        modes=tuple(modes),
        scopes=tuple(scopes),
        required_env_keys=tuple(env_keys),
        preferred_resource_id=preferred_resource_id,
    )


def discover_auth_resources(
    *,
    workspace: Path | None = None,
    process_def: object | None = None,
    tool_registry: ToolRegistry | None = None,
    mcp_manager: MCPConfigManager | None = None,
    scope: str = "active",
) -> list[DiscoveredAuthResource]:
    """Discover auth resources from process requirements, tools, and MCP aliases."""
    raw_requirements: list[dict[str, object]] = []

    if process_def is not None:
        auth_block = getattr(process_def, "auth", None)
        required = getattr(auth_block, "required", [])
        if isinstance(required, list):
            for item in required:
                parsed = _coerce_requirement(item)
                if parsed is not None:
                    raw_requirements.append(parsed)

    if tool_registry is not None:
        list_tools = getattr(tool_registry, "list_tools", None)
        get_tool = getattr(tool_registry, "get", None)
        if callable(list_tools) and callable(get_tool):
            tool_names: set[str] = set()
            if str(scope or "active").strip().lower() == "full":
                tool_names = {
                    str(name).strip()
                    for name in list_tools()
                    if str(name).strip()
                }
            else:
                if process_def is not None:
                    tools_cfg = getattr(process_def, "tools", None)
                    excluded = {
                        str(item).strip()
                        for item in (getattr(tools_cfg, "excluded", []) or [])
                        if str(item).strip()
                    }
                    tool_names = {
                        str(name).strip()
                        for name in list_tools()
                        if str(name).strip() and str(name).strip() not in excluded
                    }
            for tool_name in sorted(
                tool_names
            ):
                tool = get_tool(tool_name)
                if tool is None:
                    continue
                declared = getattr(tool, "auth_requirements", [])
                if not isinstance(declared, list):
                    continue
                for item in declared:
                    parsed = _coerce_requirement(item)
                    if parsed is None:
                        continue
                    if not str(parsed.get("provider", "")).strip():
                        # Tool name can serve as fallback key for tool-scoped refs.
                        parsed["provider"] = tool_name
                    raw_requirements.append(parsed)

    discovered_by_key: dict[str, DiscoveredAuthResource] = {}

    def _merge(
        existing: DiscoveredAuthResource,
        new: DiscoveredAuthResource,
    ) -> DiscoveredAuthResource:
        modes = tuple(dict.fromkeys([*existing.modes, *new.modes]))
        scopes = tuple(dict.fromkeys([*existing.scopes, *new.scopes]))
        env_keys = tuple(
            dict.fromkeys([*existing.required_env_keys, *new.required_env_keys])
        )
        preferred_resource_id = (
            existing.preferred_resource_id or new.preferred_resource_id
        )
        provider = existing.provider or new.provider
        display_name = existing.display_name or new.display_name
        source = existing.source or new.source
        return replace(
            existing,
            provider=provider,
            display_name=display_name,
            source=source,
            modes=modes,
            scopes=scopes,
            required_env_keys=env_keys,
            preferred_resource_id=preferred_resource_id,
        )

    for requirement in raw_requirements:
        discovered = _make_discovered_resource(requirement)
        if discovered is None:
            continue
        existing = discovered_by_key.get(discovered.discovery_key)
        if existing is None:
            discovered_by_key[discovered.discovery_key] = discovered
        else:
            discovered_by_key[discovered.discovery_key] = _merge(existing, discovered)

    if mcp_manager is not None and scope in {"active", "full"}:
        try:
            for view in mcp_manager.list_views():
                alias = str(getattr(view, "alias", "")).strip()
                if not alias:
                    continue
                resource = DiscoveredAuthResource(
                    resource_kind="mcp",
                    resource_key=alias,
                    display_name=f"MCP: {alias}",
                    provider=alias,
                    source="mcp",
                )
                existing = discovered_by_key.get(resource.discovery_key)
                if existing is None:
                    discovered_by_key[resource.discovery_key] = resource
                else:
                    discovered_by_key[resource.discovery_key] = _merge(existing, resource)
        except Exception:
            pass

    return sorted(
        discovered_by_key.values(),
        key=lambda item: (item.resource_kind, item.resource_key),
    )


def _find_resource_by_discovery_key(
    store: AuthResourcesStore,
    *,
    resource_kind: str,
    resource_key: str,
) -> AuthResource | None:
    for resource in store.resources.values():
        if resource.resource_kind != resource_kind:
            continue
        if resource.resource_key != resource_key:
            continue
        if resource.status != "active":
            continue
        return resource
    return None


def _find_deleted_resource_by_discovery_key(
    store: AuthResourcesStore,
    *,
    resource_kind: str,
    resource_key: str,
) -> AuthResource | None:
    for resource in store.resources.values():
        if resource.resource_kind != resource_kind:
            continue
        if resource.resource_key != resource_key:
            continue
        if resource.status != "deleted":
            continue
        return resource
    return None


def _with_upserted_resource(
    store: AuthResourcesStore,
    discovered: DiscoveredAuthResource,
) -> tuple[AuthResourcesStore, AuthResource, bool, bool]:
    now = _iso_now()
    existing = _find_resource_by_discovery_key(
        store,
        resource_kind=discovered.resource_kind,
        resource_key=discovered.resource_key,
    )
    created = False
    updated = False

    if existing is None:
        deleted = _find_deleted_resource_by_discovery_key(
            store,
            resource_kind=discovered.resource_kind,
            resource_key=discovered.resource_key,
        )
        if deleted is not None:
            restored = replace(
                deleted,
                display_name=discovered.display_name or deleted.display_name,
                provider=discovered.provider or deleted.provider,
                source=discovered.source or deleted.source,
                modes=tuple(
                    dict.fromkeys([*deleted.modes, *tuple(discovered.modes)])
                ),
                scopes=tuple(
                    dict.fromkeys([*deleted.scopes, *tuple(discovered.scopes)])
                ),
                required_env_keys=tuple(
                    dict.fromkeys(
                        [*deleted.required_env_keys, *tuple(discovered.required_env_keys)]
                    )
                ),
                status="active",
                deleted_at="",
                updated_at=now,
            )
            resources = dict(store.resources)
            resources[restored.resource_id] = restored
            tombstones = dict(store.tombstones)
            tombstones.pop(restored.resource_id, None)
            store = replace(store, resources=resources, tombstones=tombstones)
            updated = True
            return store, restored, created, updated

        resource_id = str(discovered.preferred_resource_id or "").strip()
        if not resource_id or resource_id in store.resources:
            resource_id = str(uuid.uuid4())
        resource = AuthResource(
            resource_id=resource_id,
            resource_kind=discovered.resource_kind,
            resource_key=discovered.resource_key,
            display_name=discovered.display_name,
            provider=discovered.provider,
            source=discovered.source,
            modes=tuple(discovered.modes),
            scopes=tuple(discovered.scopes),
            required_env_keys=tuple(discovered.required_env_keys),
            status="active",
            created_at=now,
            updated_at=now,
        )
        resources = dict(store.resources)
        resources[resource.resource_id] = resource
        store = replace(store, resources=resources)
        created = True
        return store, resource, created, updated

    next_resource = replace(
        existing,
        display_name=discovered.display_name or existing.display_name,
        provider=discovered.provider or existing.provider,
        source=discovered.source or existing.source,
        modes=tuple(
            dict.fromkeys([*existing.modes, *tuple(discovered.modes)])
        ),
        scopes=tuple(
            dict.fromkeys([*existing.scopes, *tuple(discovered.scopes)])
        ),
        required_env_keys=tuple(
            dict.fromkeys(
                [*existing.required_env_keys, *tuple(discovered.required_env_keys)]
            )
        ),
        status="active",
        deleted_at="",
        updated_at=now,
    )
    if next_resource != existing:
        resources = dict(store.resources)
        resources[next_resource.resource_id] = next_resource
        store = replace(store, resources=resources)
        updated = True
    return store, next_resource, created, updated


def _sanitize_profile_fragment(value: str) -> str:
    text = str(value or "").strip().lower()
    text = _SAFE_ID_RE.sub("_", text)
    text = text.strip("_")
    return text or "resource"


def _build_draft_profile(
    *,
    resource: AuthResource,
    existing_profile_ids: set[str],
) -> AuthProfile:
    kind_part = _sanitize_profile_fragment(resource.resource_kind)
    key_part = _sanitize_profile_fragment(resource.resource_key)
    base = f"draft_{kind_part}_{key_part}"
    profile_id = base
    counter = 2
    while profile_id in existing_profile_ids:
        profile_id = f"{base}_{counter}"
        counter += 1

    mode = ""
    for candidate in resource.modes:
        if candidate in _SUPPORTED_MODES:
            mode = candidate
            break
    if not mode:
        mode = "api_key"

    env: dict[str, str] = {}
    for key in resource.required_env_keys:
        env[str(key)] = f"${{{key}}}"

    secret_ref = ""
    token_ref = ""
    command = ""
    if mode == "api_key":
        secret_ref = f"env://TODO_{_sanitize_profile_fragment(resource.provider).upper()}_API_KEY"
    elif mode in {"oauth2_pkce", "oauth2_device"}:
        token_ref = f"env://TODO_{_sanitize_profile_fragment(resource.provider).upper()}_TOKEN"
    elif mode == "env_passthrough" and not env:
        env["API_KEY"] = "${API_KEY}"
    elif mode == "cli_passthrough":
        command = "TODO_AUTH_COMMAND"

    metadata = {
        "generated": "true",
        "generated_from": resource.resource_ref,
        "resource_id": resource.resource_id,
    }

    mcp_server = resource.resource_key if resource.resource_kind == "mcp" else ""
    return AuthProfile(
        profile_id=profile_id,
        provider=resource.provider or resource.resource_key,
        mode=mode,
        account_label=f"{resource.display_name} (Draft)",
        mcp_server=mcp_server,
        secret_ref=secret_ref,
        token_ref=token_ref,
        scopes=list(resource.scopes),
        env=env,
        command=command,
        auth_check=[],
        metadata=metadata,
        status="draft",
    )


def _binding_for_resource_profile(
    store: AuthResourcesStore,
    *,
    resource_id: str,
    profile_id: str,
) -> AuthBinding | None:
    for binding in store.bindings.values():
        if binding.resource_id != resource_id:
            continue
        if binding.profile_id != profile_id:
            continue
        if binding.status != "active":
            continue
        return binding
    return None


def bind_resource_to_profile(
    path: Path,
    *,
    resource_id: str,
    profile_id: str,
    generated_from: str = "",
    priority: int = 0,
) -> AuthResourcesStore:
    """Create (or reactivate) one resource -> profile binding."""
    clean_resource_id = str(resource_id or "").strip()
    clean_profile_id = str(profile_id or "").strip()
    if not clean_resource_id or not clean_profile_id:
        raise AuthResourceError("resource_id and profile_id are required for binding.")

    def _mutate(store: AuthResourcesStore) -> AuthResourcesStore:
        existing = _binding_for_resource_profile(
            store,
            resource_id=clean_resource_id,
            profile_id=clean_profile_id,
        )
        now = _iso_now()
        bindings = dict(store.bindings)
        if existing is not None:
            updated = replace(
                existing,
                status="active",
                deleted_at="",
                priority=_parse_int(priority, default=existing.priority),
                generated_from=generated_from or existing.generated_from,
                updated_at=now,
            )
            bindings[updated.binding_id] = updated
            return replace(store, bindings=bindings)

        binding = AuthBinding(
            binding_id=str(uuid.uuid4()),
            resource_id=clean_resource_id,
            profile_id=clean_profile_id,
            priority=_parse_int(priority, default=0),
            generated_from=str(generated_from or "").strip(),
            status="active",
            created_at=now,
            updated_at=now,
        )
        bindings[binding.binding_id] = binding
        return replace(store, bindings=bindings)

    return mutate_workspace_auth_resources(path, _mutate)


def set_workspace_resource_default(
    path: Path,
    *,
    resource_id: str,
    profile_id: str | None,
) -> AuthResourcesStore:
    """Set or clear one workspace resource default."""
    clean_resource_id = str(resource_id or "").strip()
    if not clean_resource_id:
        raise AuthResourceError("resource_id cannot be empty for resource default.")

    def _mutate(store: AuthResourcesStore) -> AuthResourcesStore:
        defaults = dict(store.workspace_defaults)
        if profile_id is None:
            defaults.pop(clean_resource_id, None)
        else:
            clean_profile_id = str(profile_id or "").strip()
            if not clean_profile_id:
                raise AuthResourceError("profile_id cannot be empty for resource default.")
            defaults[clean_resource_id] = clean_profile_id
        return replace(store, workspace_defaults=defaults)

    return mutate_workspace_auth_resources(path, _mutate)


def _profile_is_usable(profile: AuthProfile) -> bool:
    status = str(getattr(profile, "status", "ready") or "ready").strip().lower()
    return status != "archived"


def sync_missing_drafts(
    *,
    workspace: Path,
    explicit_auth_path: Path | None = None,
    process_def: object | None = None,
    tool_registry: ToolRegistry | None = None,
    mcp_manager: MCPConfigManager | None = None,
    scope: str = "active",
) -> AuthDraftSyncResult:
    """Sync missing discovered resources and auto-create draft auth profiles.

    This operation is additive-only:
    - creates/updates resource registry entries
    - creates missing draft profiles + bindings
    - updates workspace defaults when exactly one usable binding exists
    It never deletes existing profiles/bindings/defaults.
    """
    ws = workspace.resolve()
    resources_path = default_workspace_auth_resources_path(ws)
    discovered = discover_auth_resources(
        workspace=ws,
        process_def=process_def,
        tool_registry=tool_registry,
        mcp_manager=mcp_manager,
        scope=scope,
    )

    initial_store = load_workspace_auth_resources(resources_path)
    store = initial_store
    created_resources = 0
    updated_resources = 0
    warnings: list[str] = []
    discovered_resources: list[AuthResource] = []
    for item in discovered:
        store, resource, created, updated = _with_upserted_resource(store, item)
        if created:
            created_resources += 1
        if updated:
            updated_resources += 1
        discovered_resources.append(resource)

    # Persist resource registry changes before cross-file draft/profile writes.
    if store != initial_store:
        write_workspace_auth_resources(resources_path, store)

    auth_write_path = resolve_auth_write_path(explicit_path=explicit_auth_path)
    merged = load_merged_auth_config(
        workspace=ws,
        explicit_path=explicit_auth_path,
    )
    profiles = dict(merged.config.profiles)
    existing_profile_ids = set(profiles.keys())

    created_drafts = 0
    created_bindings = 0
    updated_defaults = 0
    bindings = dict(store.bindings)
    defaults = dict(store.workspace_defaults)
    now = _iso_now()
    completed_pending_ids: set[str] = set()

    # Retry completion of previously pending cross-file ops.
    for pending_id, payload in sorted(store.pending_operations.items()):
        resource_id = str(payload.get("resource_id", "")).strip()
        profile_id = str(payload.get("profile_id", "")).strip()
        if not resource_id or not profile_id:
            completed_pending_ids.add(pending_id)
            continue
        profile = profiles.get(profile_id)
        if profile is None:
            continue
        if _binding_for_resource_profile(
            store,
            resource_id=resource_id,
            profile_id=profile_id,
        ) is None:
            binding_id = str(uuid.uuid4())
            bindings[binding_id] = AuthBinding(
                binding_id=binding_id,
                resource_id=resource_id,
                profile_id=profile_id,
                priority=0,
                generated_from=f"pending:{pending_id}",
                status="active",
                created_at=now,
                updated_at=now,
            )
            created_bindings += 1
        if defaults.get(resource_id) != profile_id:
            defaults[resource_id] = profile_id
            updated_defaults += 1
        completed_pending_ids.add(pending_id)

    store = replace(store, bindings=bindings, workspace_defaults=defaults)

    for resource in discovered_resources:
        active_bindings = [
            binding
            for binding in active_bindings_for_resource(store, resource.resource_id)
            if _profile_is_usable(profiles.get(binding.profile_id))
        ]
        if not active_bindings:
            pending_id = str(uuid.uuid4())
            pending_payload = {
                "op": "draft_sync_bind",
                "resource_id": resource.resource_id,
                "profile_id": "",
                "created_at": now,
            }
            store = mutate_workspace_auth_resources(
                resources_path,
                lambda current: replace(
                    current,
                    pending_operations={
                        **current.pending_operations,
                        pending_id: dict(pending_payload),
                    },
                ),
            )
            draft_profile = _build_draft_profile(
                resource=resource,
                existing_profile_ids=existing_profile_ids,
            )
            pending_payload["profile_id"] = draft_profile.profile_id
            store = mutate_workspace_auth_resources(
                resources_path,
                lambda current: replace(
                    current,
                    pending_operations={
                        **{
                            key: value
                            for key, value in current.pending_operations.items()
                            if key != pending_id
                        },
                        pending_id: dict(pending_payload),
                    },
                ),
            )
            try:
                upsert_auth_profile(auth_write_path, draft_profile, must_exist=False)
            except Exception as e:
                warnings.append(
                    f"Failed to create draft profile for {resource.resource_ref}: {e}"
                )
                continue
            profiles[draft_profile.profile_id] = draft_profile
            existing_profile_ids.add(draft_profile.profile_id)
            created_drafts += 1

            binding_id = str(uuid.uuid4())
            binding = AuthBinding(
                binding_id=binding_id,
                resource_id=resource.resource_id,
                profile_id=draft_profile.profile_id,
                priority=0,
                generated_from=f"auto:{resource.resource_ref}",
                status="active",
                created_at=now,
                updated_at=now,
            )
            bindings[binding.binding_id] = binding
            created_bindings += 1
            active_bindings = [binding]
            completed_pending_ids.add(pending_id)

        if len(active_bindings) == 1:
            only_profile_id = active_bindings[0].profile_id
            if defaults.get(resource.resource_id) != only_profile_id:
                defaults[resource.resource_id] = only_profile_id
                updated_defaults += 1

    latest_store = load_workspace_auth_resources(resources_path)
    next_store = replace(
        latest_store,
        bindings=bindings,
        workspace_defaults=defaults,
        pending_operations={
            pending_id: payload
            for pending_id, payload in latest_store.pending_operations.items()
            if pending_id not in completed_pending_ids
        },
    )
    if next_store != latest_store:
        try:
            write_workspace_auth_resources(resources_path, next_store)
        except Exception as e:
            warnings.append(
                f"Failed to finalize resource bindings/defaults; pending ops retained: {e}"
            )

    return AuthDraftSyncResult(
        created_resources=created_resources,
        updated_resources=updated_resources,
        created_drafts=created_drafts,
        created_bindings=created_bindings,
        updated_defaults=updated_defaults,
        warnings=tuple(warnings),
    )


def _resource_id_for_kind_key(
    store: AuthResourcesStore,
    *,
    resource_kind: str,
    resource_key: str,
) -> str:
    clean_kind = str(resource_kind or "").strip().lower()
    clean_key = str(resource_key or "").strip()
    if not clean_kind or not clean_key:
        return ""
    resource = _find_resource_by_discovery_key(
        store,
        resource_kind=clean_kind,
        resource_key=clean_key,
    )
    if resource is None:
        return ""
    return resource.resource_id


def rename_resource_key(
    *,
    workspace: Path,
    resource_kind: str,
    old_key: str,
    new_key: str,
) -> bool:
    """Rename resource key while preserving resource_id and bindings."""
    ws = workspace.resolve()
    path = default_workspace_auth_resources_path(ws)
    changed = False

    def _mutate(store: AuthResourcesStore) -> AuthResourcesStore:
        nonlocal changed
        resource = _find_resource_by_discovery_key(
            store,
            resource_kind=str(resource_kind or "").strip().lower(),
            resource_key=str(old_key or "").strip(),
        )
        if resource is None:
            return store
        clean_new_key = str(new_key or "").strip()
        if not clean_new_key or clean_new_key == resource.resource_key:
            return store
        updated = replace(
            resource,
            resource_key=clean_new_key,
            display_name=(
                f"MCP: {clean_new_key}"
                if resource.resource_kind == "mcp"
                else resource.display_name
            ),
            updated_at=_iso_now(),
        )
        resources = dict(store.resources)
        resources[updated.resource_id] = updated
        changed = True
        return replace(store, resources=resources)

    mutate_workspace_auth_resources(path, _mutate)
    return changed


def cleanup_deleted_resource(
    *,
    workspace: Path,
    explicit_auth_path: Path | None,
    resource_kind: str,
    resource_key: str,
) -> bool:
    """Delete/tombstone one resource and clean bindings/defaults safely."""
    ws = workspace.resolve()
    path = default_workspace_auth_resources_path(ws)
    store = load_workspace_auth_resources(path)
    resource = _find_resource_by_discovery_key(
        store,
        resource_kind=str(resource_kind or "").strip().lower(),
        resource_key=str(resource_key or "").strip(),
    )
    if resource is None:
        return False

    now = _iso_now()
    resources = dict(store.resources)
    resources[resource.resource_id] = replace(
        resource,
        status="deleted",
        deleted_at=now,
        updated_at=now,
    )
    tombstones = dict(store.tombstones)
    tombstones[resource.resource_id] = {
        "kind": resource.resource_kind,
        "key": resource.resource_key,
        "display_name": resource.display_name,
        "provider": resource.provider,
        "deleted_at": now,
    }

    bindings = dict(store.bindings)
    affected_profile_ids: set[str] = set()
    for binding in store.bindings.values():
        if binding.resource_id != resource.resource_id:
            continue
        if binding.status != "active":
            continue
        affected_profile_ids.add(binding.profile_id)
        bindings[binding.binding_id] = replace(
            binding,
            status="deleted",
            deleted_at=now,
            updated_at=now,
        )

    defaults = {
        rid: pid
        for rid, pid in store.workspace_defaults.items()
        if rid != resource.resource_id
    }
    next_store = replace(
        store,
        resources=resources,
        bindings=bindings,
        workspace_defaults=defaults,
        tombstones=tombstones,
    )
    write_workspace_auth_resources(path, next_store)

    merged = load_merged_auth_config(workspace=ws, explicit_path=explicit_auth_path)
    auth_write_path = resolve_auth_write_path(explicit_path=explicit_auth_path)

    for profile_id in sorted(affected_profile_ids):
        profile = merged.config.profiles.get(profile_id)
        if profile is None:
            continue
        remaining = [
            binding
            for binding in next_store.bindings.values()
            if binding.status == "active"
            and binding.profile_id == profile_id
        ]
        if remaining:
            continue
        status = str(getattr(profile, "status", "ready") or "ready").strip().lower()
        if status == "draft":
            try:
                remove_auth_profile(auth_write_path, profile_id)
            except Exception:
                continue
            continue
        archived = replace(profile, status="archived")
        try:
            upsert_auth_profile(auth_write_path, archived, must_exist=True)
        except Exception:
            continue
    return True


def resource_id_for_kind_key(
    store: AuthResourcesStore,
    *,
    resource_kind: str,
    resource_key: str,
) -> str:
    """Lookup helper used by UI/runtime."""
    return _resource_id_for_kind_key(
        store,
        resource_kind=resource_kind,
        resource_key=resource_key,
    )


def profile_bindings_map(store: AuthResourcesStore) -> dict[str, str]:
    """Return profile_id -> resource_id for active bindings (highest priority wins)."""
    by_profile: dict[str, tuple[int, str]] = {}
    for binding in store.bindings.values():
        if binding.status != "active":
            continue
        profile_id = str(binding.profile_id or "").strip()
        resource_id = str(binding.resource_id or "").strip()
        if not profile_id or not resource_id:
            continue
        candidate = (_parse_int(binding.priority, default=0), resource_id)
        current = by_profile.get(profile_id)
        if current is None or candidate < current:
            by_profile[profile_id] = candidate
    return {profile_id: resource_id for profile_id, (_priority, resource_id) in by_profile.items()}


def remove_profile_from_resource_store(
    path: Path,
    *,
    profile_id: str,
) -> AuthResourcesStore:
    """Detach a removed profile from resource bindings/defaults."""
    clean_profile_id = str(profile_id or "").strip()
    if not clean_profile_id:
        raise AuthResourceError("profile_id cannot be empty.")

    def _mutate(store: AuthResourcesStore) -> AuthResourcesStore:
        now = _iso_now()
        bindings = dict(store.bindings)
        changed = False
        for binding in store.bindings.values():
            if binding.status != "active":
                continue
            if binding.profile_id != clean_profile_id:
                continue
            bindings[binding.binding_id] = replace(
                binding,
                status="deleted",
                deleted_at=now,
                updated_at=now,
            )
            changed = True
        defaults = {
            resource_id: mapped_profile_id
            for resource_id, mapped_profile_id in store.workspace_defaults.items()
            if mapped_profile_id != clean_profile_id
        }
        if not changed and defaults == store.workspace_defaults:
            return store
        return replace(store, bindings=bindings, workspace_defaults=defaults)

    return mutate_workspace_auth_resources(path, _mutate)


def resource_delete_impact(
    *,
    workspace: Path,
    resource_kind: str,
    resource_key: str,
) -> ResourceDeleteImpact:
    """Compute active binding/default impact for one resource delete."""
    store = load_workspace_auth_resources(
        default_workspace_auth_resources_path(workspace.resolve())
    )
    resource = _find_resource_by_discovery_key(
        store,
        resource_kind=str(resource_kind or "").strip().lower(),
        resource_key=str(resource_key or "").strip(),
    )
    if resource is None:
        return ResourceDeleteImpact()
    bindings = [
        binding
        for binding in store.bindings.values()
        if binding.status == "active" and binding.resource_id == resource.resource_id
    ]
    binding_ids = tuple(sorted(binding.binding_id for binding in bindings))
    profile_ids = tuple(sorted({binding.profile_id for binding in bindings}))
    referencing_processes = _processes_referencing_resource(
        workspace.resolve(),
        resource,
    )
    return ResourceDeleteImpact(
        resource_id=resource.resource_id,
        active_binding_ids=binding_ids,
        active_profile_ids=profile_ids,
        workspace_default_profile_id=store.workspace_defaults.get(resource.resource_id, ""),
        referencing_processes=referencing_processes,
    )


def _processes_referencing_resource(
    workspace: Path,
    resource: AuthResource,
) -> tuple[str, ...]:
    """Best-effort process reference scan used by delete impact dialogs."""
    try:
        from loom.processes.schema import ProcessLoader

        loader = ProcessLoader(workspace=workspace)
        discovered = loader.discover()
    except Exception:
        return ()

    referenced: set[str] = set()
    for process_name in sorted(discovered):
        try:
            process_def = loader.load(process_name)
        except Exception:
            continue
        auth_block = getattr(process_def, "auth", None)
        required = getattr(auth_block, "required", []) or []
        for raw in required:
            requirement = _coerce_requirement(raw)
            if requirement is None:
                continue
            discovered_resource = _make_discovered_resource(requirement)
            if discovered_resource is None:
                continue
            if (
                discovered_resource.resource_kind == resource.resource_kind
                and discovered_resource.resource_key == resource.resource_key
            ):
                referenced.add(str(getattr(process_def, "name", "") or process_name))
                break
    return tuple(sorted(referenced))


def restore_deleted_resource(
    *,
    workspace: Path,
    resource_kind: str,
    resource_key: str,
) -> bool:
    """Restore a tombstoned/deleted resource deterministically."""
    ws = workspace.resolve()
    path = default_workspace_auth_resources_path(ws)
    store = load_workspace_auth_resources(path)
    target = _find_deleted_resource_by_discovery_key(
        store,
        resource_kind=str(resource_kind or "").strip().lower(),
        resource_key=str(resource_key or "").strip(),
    )
    if target is None:
        return False
    resources = dict(store.resources)
    resources[target.resource_id] = replace(
        target,
        status="active",
        deleted_at="",
        updated_at=_iso_now(),
    )
    tombstones = dict(store.tombstones)
    tombstones.pop(target.resource_id, None)
    write_workspace_auth_resources(
        path,
        replace(store, resources=resources, tombstones=tombstones),
    )
    return True


def _snapshot_dir(workspace: Path) -> Path:
    return workspace / ".loom" / "snapshots"


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def create_auth_snapshot(
    *,
    workspace: Path,
    explicit_auth_path: Path | None,
    label: str = "migrate",
) -> Path:
    """Create snapshot of auth config files used by migration/rollback."""
    ws = workspace.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = _SAFE_ID_RE.sub("_", str(label or "snapshot").strip().lower()).strip("_")
    safe_label = safe_label or "snapshot"
    snapshot_path = _snapshot_dir(ws) / f"auth-{stamp}-{safe_label}"
    snapshot_path.mkdir(parents=True, exist_ok=True)

    auth_path = resolve_auth_write_path(explicit_path=explicit_auth_path)
    defaults_path = default_workspace_auth_defaults_path(ws)
    resources_path = default_workspace_auth_resources_path(ws)

    copied = {
        "auth_toml": _copy_if_exists(auth_path, snapshot_path / "auth.toml"),
        "auth_defaults_toml": _copy_if_exists(
            defaults_path,
            snapshot_path / "auth.defaults.toml",
        ),
        "auth_resources_toml": _copy_if_exists(
            resources_path,
            snapshot_path / "auth.resources.toml",
        ),
    }
    metadata_lines = [
        "# Auth snapshot metadata",
        f"created_at = {_toml_escape(_iso_now())}",
        f"workspace = {_toml_escape(str(ws))}",
        f"auth_path = {_toml_escape(str(auth_path))}",
        f"defaults_path = {_toml_escape(str(defaults_path))}",
        f"resources_path = {_toml_escape(str(resources_path))}",
        "",
        "[copied]",
    ]
    for key in sorted(copied):
        metadata_lines.append(f"{_toml_key(key)} = {'true' if copied[key] else 'false'}")
    metadata_lines.append("")
    _atomic_write_text(snapshot_path / "metadata.toml", "\n".join(metadata_lines))
    return snapshot_path


def restore_auth_snapshot(
    *,
    workspace: Path,
    explicit_auth_path: Path | None,
    snapshot_path: Path,
) -> None:
    """Restore auth files from a previously created snapshot directory."""
    ws = workspace.resolve()
    snapshot = snapshot_path.expanduser().resolve()
    if not snapshot.exists() or not snapshot.is_dir():
        raise AuthResourceError(f"Snapshot not found: {snapshot}")

    auth_path = resolve_auth_write_path(explicit_path=explicit_auth_path)
    defaults_path = default_workspace_auth_defaults_path(ws)
    resources_path = default_workspace_auth_resources_path(ws)

    for source_name, target in (
        ("auth.toml", auth_path),
        ("auth.defaults.toml", defaults_path),
        ("auth.resources.toml", resources_path),
    ):
        source = snapshot / source_name
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def migrate_legacy_auth(
    *,
    workspace: Path,
    explicit_auth_path: Path | None,
) -> AuthMigrationResult:
    """Infer resource bindings/defaults from legacy provider and mcp_server config."""
    ws = workspace.resolve()
    snapshot = create_auth_snapshot(
        workspace=ws,
        explicit_auth_path=explicit_auth_path,
        label="migrate",
    )
    merged = load_merged_auth_config(workspace=ws, explicit_path=explicit_auth_path)
    store_path = default_workspace_auth_resources_path(ws)
    store = load_workspace_auth_resources(store_path)
    now = _iso_now()

    created_resources = 0
    created_bindings = 0
    created_workspace_defaults = 0
    created_user_resource_defaults = 0
    warnings: list[str] = []

    resources = dict(store.resources)
    bindings = dict(store.bindings)
    workspace_defaults = dict(store.workspace_defaults)

    def _ensure_resource(
        *,
        kind: str,
        key: str,
        provider: str,
        source: str,
        modes: tuple[str, ...] = (),
        required_env_keys: tuple[str, ...] = (),
    ) -> AuthResource:
        nonlocal created_resources
        existing = _find_resource_by_discovery_key(
            AuthResourcesStore(resources=resources),
            resource_kind=kind,
            resource_key=key,
        )
        if existing is not None:
            return existing
        deleted = _find_deleted_resource_by_discovery_key(
            AuthResourcesStore(resources=resources),
            resource_kind=kind,
            resource_key=key,
        )
        if deleted is not None:
            restored = replace(
                deleted,
                status="active",
                deleted_at="",
                provider=provider or deleted.provider,
                source=source or deleted.source,
                updated_at=now,
            )
            resources[restored.resource_id] = restored
            return restored
        resource = AuthResource(
            resource_id=str(uuid.uuid4()),
            resource_kind=kind,
            resource_key=key,
            display_name=(
                f"MCP: {key}"
                if kind == "mcp"
                else (f"Tool: {key}" if kind == "tool" else f"API: {provider or key}")
            ),
            provider=provider or key,
            source=source,
            modes=modes,
            required_env_keys=required_env_keys,
            status="active",
            created_at=now,
            updated_at=now,
        )
        resources[resource.resource_id] = resource
        created_resources += 1
        return resource

    for profile in merged.config.profiles.values():
        if str(profile.status or "").strip().lower() == "archived":
            continue
        if profile.mcp_server:
            resource = _ensure_resource(
                kind="mcp",
                key=profile.mcp_server,
                provider=profile.provider,
                source="mcp",
                modes=(profile.mode,),
                required_env_keys=tuple(sorted(profile.env.keys())),
            )
        else:
            resource = _ensure_resource(
                kind="api_integration",
                key=profile.provider,
                provider=profile.provider,
                source="api",
                modes=(profile.mode,),
                required_env_keys=tuple(sorted(profile.env.keys())),
            )
        if _binding_for_resource_profile(
            AuthResourcesStore(bindings=bindings),
            resource_id=resource.resource_id,
            profile_id=profile.profile_id,
        ) is None:
            binding_id = str(uuid.uuid4())
            bindings[binding_id] = AuthBinding(
                binding_id=binding_id,
                resource_id=resource.resource_id,
                profile_id=profile.profile_id,
                priority=0,
                generated_from="migrate:legacy",
                status="active",
                created_at=now,
                updated_at=now,
            )
            created_bindings += 1

    for selector, profile_id in merged.workspace_defaults.items():
        profile = merged.config.profiles.get(profile_id)
        if profile is None:
            warnings.append(
                f"workspace default {selector!r} references unknown profile {profile_id!r}"
            )
            continue
        if selector != profile.provider:
            warnings.append(
                "workspace default selector "
                f"{selector!r} does not match provider {profile.provider!r}"
            )
            continue
        resource = _find_resource_by_discovery_key(
            AuthResourcesStore(resources=resources),
            resource_kind=(
                "mcp" if profile.mcp_server else "api_integration"
            ),
            resource_key=(profile.mcp_server or profile.provider),
        )
        if resource is None:
            continue
        if workspace_defaults.get(resource.resource_id) != profile.profile_id:
            workspace_defaults[resource.resource_id] = profile.profile_id
            created_workspace_defaults += 1

    next_store = replace(
        store,
        resources=resources,
        bindings=bindings,
        workspace_defaults=workspace_defaults,
    )
    if next_store != store:
        write_workspace_auth_resources(store_path, next_store)

    auth_path = resolve_auth_write_path(explicit_path=explicit_auth_path)
    auth_cfg = load_auth_file(auth_path)
    user_resource_defaults = dict(auth_cfg.resource_defaults)
    for selector, profile_id in merged.config.defaults.items():
        profile = merged.config.profiles.get(profile_id)
        if profile is None:
            continue
        if selector != profile.provider:
            continue
        kind = "mcp" if profile.mcp_server else "api_integration"
        key = profile.mcp_server or profile.provider
        resource = _find_resource_by_discovery_key(
            AuthResourcesStore(resources=resources),
            resource_kind=kind,
            resource_key=key,
        )
        if resource is None:
            continue
        if user_resource_defaults.get(resource.resource_id) != profile.profile_id:
            user_resource_defaults[resource.resource_id] = profile.profile_id
            created_user_resource_defaults += 1
    if user_resource_defaults != auth_cfg.resource_defaults:
        write_auth_file(
            auth_path,
            replace(auth_cfg, resource_defaults=user_resource_defaults),
        )

    return AuthMigrationResult(
        snapshot_path=snapshot,
        created_resources=created_resources,
        created_bindings=created_bindings,
        created_workspace_defaults=created_workspace_defaults,
        created_user_resource_defaults=created_user_resource_defaults,
        warnings=tuple(warnings),
    )


def audit_auth_state(
    *,
    workspace: Path,
    explicit_auth_path: Path | None,
) -> AuthAuditReport:
    """Collect audit findings for auth migration and lifecycle cleanup."""
    ws = workspace.resolve()
    merged = load_merged_auth_config(workspace=ws, explicit_path=explicit_auth_path)
    store = load_workspace_auth_resources(default_workspace_auth_resources_path(ws))

    orphaned_profiles: list[str] = []
    for profile_id, profile in merged.config.profiles.items():
        if str(profile.status or "").strip().lower() == "archived":
            continue
        has_binding = any(
            binding.status == "active" and binding.profile_id == profile_id
            for binding in store.bindings.values()
        )
        if not has_binding:
            orphaned_profiles.append(profile_id)

    orphaned_bindings: list[str] = []
    deleted_resource_bindings: list[str] = []
    for binding_id, binding in store.bindings.items():
        resource = store.resources.get(binding.resource_id)
        if resource is None:
            orphaned_bindings.append(binding_id)
            continue
        if resource.status == "deleted" and binding.status == "active":
            deleted_resource_bindings.append(binding_id)
        if binding.profile_id not in merged.config.profiles:
            orphaned_bindings.append(binding_id)

    legacy_provider_defaults = sorted(
        f"{selector}->{profile_id}"
        for selector, profile_id in merged.config.defaults.items()
        if selector and profile_id
    )
    legacy_provider_defaults.extend(
        sorted(
            f"workspace:{selector}->{profile_id}"
            for selector, profile_id in merged.workspace_defaults.items()
            if selector and profile_id
        )
    )

    dangling_workspace_resource_defaults = sorted(
        resource_id
        for resource_id, profile_id in store.workspace_defaults.items()
        if resource_id not in store.resources
        or store.resources[resource_id].status != "active"
        or profile_id not in merged.config.profiles
    )
    dangling_user_resource_defaults = sorted(
        resource_id
        for resource_id, profile_id in merged.config.resource_defaults.items()
        if resource_id not in store.resources
        or store.resources[resource_id].status != "active"
        or profile_id not in merged.config.profiles
    )

    return AuthAuditReport(
        orphaned_profiles=tuple(sorted(orphaned_profiles)),
        orphaned_bindings=tuple(sorted(orphaned_bindings)),
        deleted_resource_bindings=tuple(sorted(deleted_resource_bindings)),
        legacy_provider_defaults=tuple(sorted(set(legacy_provider_defaults))),
        dangling_workspace_resource_defaults=tuple(dangling_workspace_resource_defaults),
        dangling_user_resource_defaults=tuple(dangling_user_resource_defaults),
    )
