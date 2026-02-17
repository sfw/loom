"""Shared MCP config loader/manager used by CLI and TUI."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from loom.config import Config, ConfigError, MCPConfig, MCPServerConfig


class MCPConfigManagerError(Exception):
    """Raised for MCP config parsing/mutation failures."""


@dataclass(frozen=True)
class MCPServerView:
    """Resolved server plus source metadata."""

    alias: str
    server: MCPServerConfig
    source: str
    source_path: Path | None


@dataclass(frozen=True)
class MergedMCPConfig:
    """Merged MCP config with alias source metadata."""

    config: MCPConfig
    sources: dict[str, MCPServerView]
    explicit_path: Path | None
    workspace_path: Path
    user_path: Path
    legacy_config_path: Path | None

    def get(self, alias: str) -> MCPServerView | None:
        return self.sources.get(alias)

    def as_views(self) -> list[MCPServerView]:
        return sorted(self.sources.values(), key=lambda v: v.alias)


def default_user_mcp_path() -> Path:
    """Default user-scoped MCP config path."""
    return Path.home() / ".loom" / "mcp.toml"


def default_workspace_mcp_path(workspace: Path) -> Path:
    """Default workspace-scoped MCP config path."""
    return workspace / ".loom" / "mcp.toml"


def _parse_timeout(value: object) -> int:
    try:
        timeout = int(value)
    except (TypeError, ValueError):
        timeout = 30
    return timeout if timeout > 0 else 30


def _parse_servers(raw: object) -> dict[str, MCPServerConfig]:
    servers: dict[str, MCPServerConfig] = {}
    if not isinstance(raw, dict):
        return servers
    for alias, server_raw in raw.items():
        if not isinstance(alias, str) or not isinstance(server_raw, dict):
            continue

        raw_args = server_raw.get("args", [])
        args = [str(v) for v in raw_args] if isinstance(raw_args, list) else []

        raw_env = server_raw.get("env", {})
        env: dict[str, str] = {}
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                if isinstance(key, str):
                    env[key] = str(value)

        servers[alias] = MCPServerConfig(
            command=str(server_raw.get("command", "")),
            args=args,
            env=env,
            cwd=str(server_raw.get("cwd", "")),
            timeout_seconds=_parse_timeout(server_raw.get("timeout_seconds", 30)),
            enabled=bool(server_raw.get("enabled", True)),
        )
    return servers


def _parse_mcp_section(raw: object) -> MCPConfig:
    if not isinstance(raw, dict):
        return MCPConfig()
    servers_raw = raw.get("servers", {})
    return MCPConfig(servers=_parse_servers(servers_raw))


def load_mcp_file(path: Path) -> MCPConfig:
    """Load MCP config from mcp.toml path."""
    if not path.exists():
        return MCPConfig()
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise MCPConfigManagerError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise MCPConfigManagerError(f"Cannot read MCP config {path}: {e}") from e
    return _parse_mcp_section(raw.get("mcp"))


def _merge_layers(
    *,
    legacy: MCPConfig,
    user: MCPConfig,
    workspace: MCPConfig,
    explicit: MCPConfig,
    user_path: Path,
    workspace_path: Path,
    explicit_path: Path | None,
) -> tuple[MCPConfig, dict[str, MCPServerView]]:
    merged: dict[str, MCPServerConfig] = {}
    sources: dict[str, MCPServerView] = {}
    layers = [
        ("legacy", None, legacy),
        ("user", user_path, user),
        ("workspace", workspace_path, workspace),
        ("explicit", explicit_path, explicit),
    ]
    for source_name, source_path, cfg in layers:
        for alias, server in cfg.servers.items():
            merged[alias] = server
            sources[alias] = MCPServerView(
                alias=alias,
                server=server,
                source=source_name,
                source_path=source_path,
            )
    return MCPConfig(servers=merged), sources


def _detect_legacy_path(
    config_path: Path | None,
    *,
    workspace: Path,
) -> Path | None:
    if config_path is not None and config_path.exists():
        return config_path
    for candidate in (workspace / "loom.toml", Path.home() / ".loom" / "loom.toml"):
        if candidate.exists():
            return candidate
    return None


def load_merged_mcp_config(
    *,
    config: Config | None,
    workspace: Path | None = None,
    explicit_path: Path | None = None,
    user_path: Path | None = None,
    legacy_config_path: Path | None = None,
) -> MergedMCPConfig:
    """Load merged MCP config with precedence metadata.

    Merge precedence (highest wins):
      explicit > workspace/.loom/mcp.toml > ~/.loom/mcp.toml > legacy loom.toml
    """
    ws = (workspace or Path.cwd()).resolve()
    user_cfg_path = (user_path or default_user_mcp_path()).expanduser()
    workspace_cfg_path = default_workspace_mcp_path(ws)
    explicit_cfg_path = explicit_path.expanduser().resolve() if explicit_path else None

    legacy = config.mcp if config is not None else MCPConfig()
    user_cfg = load_mcp_file(user_cfg_path)
    workspace_cfg = load_mcp_file(workspace_cfg_path)
    explicit_cfg = (
        load_mcp_file(explicit_cfg_path)
        if explicit_cfg_path is not None
        else MCPConfig()
    )

    merged, sources = _merge_layers(
        legacy=legacy,
        user=user_cfg,
        workspace=workspace_cfg,
        explicit=explicit_cfg,
        user_path=user_cfg_path,
        workspace_path=workspace_cfg_path,
        explicit_path=explicit_cfg_path,
    )
    legacy_path = _detect_legacy_path(legacy_config_path, workspace=ws)
    return MergedMCPConfig(
        config=merged,
        sources=sources,
        explicit_path=explicit_cfg_path,
        workspace_path=workspace_cfg_path,
        user_path=user_cfg_path,
        legacy_config_path=legacy_path,
    )


def apply_mcp_overrides(
    config: Config,
    *,
    workspace: Path | None = None,
    explicit_path: Path | None = None,
    user_path: Path | None = None,
    legacy_config_path: Path | None = None,
) -> Config:
    """Return Config with merged MCP layers applied."""
    merged = load_merged_mcp_config(
        config=config,
        workspace=workspace,
        explicit_path=explicit_path,
        user_path=user_path,
        legacy_config_path=legacy_config_path,
    )
    return replace(config, mcp=merged.config)


def is_env_reference(value: str) -> bool:
    """Whether value is an environment indirection like `${TOKEN}`."""
    return value.startswith("${") and value.endswith("}") and len(value) >= 4


def redact_secret(value: str) -> str:
    """Redact secrets while preserving env references."""
    if is_env_reference(value):
        return value
    return "<redacted>"


def redact_server_env(server: MCPServerConfig) -> dict[str, str]:
    """Return redacted env map for display."""
    return {key: redact_secret(value) for key, value in sorted(server.env.items())}


def _toml_escape(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _render_mcp_toml(servers: dict[str, MCPServerConfig]) -> str:
    lines: list[str] = [
        "# Loom MCP configuration",
        "# Managed by `loom mcp ...`",
        "",
    ]
    for alias in sorted(servers):
        server = servers[alias]
        lines.append(f"[mcp.servers.{alias}]")
        lines.append(f"command = {_toml_escape(server.command)}")
        args_rendered = ", ".join(_toml_escape(arg) for arg in server.args)
        lines.append(f"args = [{args_rendered}]")
        lines.append(f"cwd = {_toml_escape(server.cwd)}")
        lines.append(f"timeout_seconds = {server.timeout_seconds}")
        lines.append(f"enabled = {'true' if server.enabled else 'false'}")
        if server.env:
            lines.append("")
            lines.append(f"[mcp.servers.{alias}.env]")
            for env_key in sorted(server.env):
                lines.append(f"{env_key} = {_toml_escape(server.env[env_key])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_mcp_file(path: Path, servers: dict[str, MCPServerConfig]) -> None:
    """Write mcp.toml atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _render_mcp_toml(servers)
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
            # Best effort on platforms/filesystems that don't support dir fsync.
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


_MCP_HEADER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")


def remove_mcp_sections_from_loom_toml(path: Path) -> bool:
    """Remove `[mcp...]` sections from loom.toml and rewrite atomically.

    Returns True when any lines were removed.
    """
    if not path.exists():
        return False
    try:
        original = path.read_text(encoding="utf-8")
    except OSError as e:
        raise MCPConfigManagerError(f"Cannot read {path}: {e}") from e

    kept: list[str] = []
    removing = False
    removed_any = False
    for line in original.splitlines(keepends=True):
        match = _MCP_HEADER_RE.match(line)
        if match:
            section = match.group(1).strip()
            removing = section == "mcp" or section.startswith("mcp.")
            if removing:
                removed_any = True
                continue
        if removing:
            removed_any = True
            continue
        kept.append(line)

    if not removed_any:
        return False

    write_mcp_file_like(path, "".join(kept).rstrip() + "\n")
    return True


def write_mcp_file_like(path: Path, content: str) -> None:
    """Atomic generic text-file write used for migration rewrites."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


class MCPConfigManager:
    """Load/mutate MCP config while honoring source precedence."""

    def __init__(
        self,
        *,
        config: Config | None,
        workspace: Path | None = None,
        explicit_path: Path | None = None,
        user_path: Path | None = None,
        legacy_config_path: Path | None = None,
    ) -> None:
        self._config = config
        self._workspace = workspace
        self._explicit_path = explicit_path
        self._user_path = user_path
        self._legacy_config_path = legacy_config_path

    def load(self) -> MergedMCPConfig:
        return load_merged_mcp_config(
            config=self._config,
            workspace=self._workspace,
            explicit_path=self._explicit_path,
            user_path=self._user_path,
            legacy_config_path=self._legacy_config_path,
        )

    def resolve_write_path(self) -> Path:
        merged = self.load()
        if merged.explicit_path is not None:
            return merged.explicit_path
        if merged.workspace_path.exists():
            return merged.workspace_path
        return merged.user_path

    def _load_servers_from_path(self, path: Path) -> dict[str, MCPServerConfig]:
        return dict(load_mcp_file(path).servers)

    def list_views(self) -> list[MCPServerView]:
        return self.load().as_views()

    def get_view(self, alias: str) -> MCPServerView | None:
        return self.load().get(alias)

    def add_server(self, alias: str, server: MCPServerConfig) -> Path:
        if self.get_view(alias) is not None:
            raise MCPConfigManagerError(f"MCP server alias already exists: {alias}")
        target = self.resolve_write_path()
        servers = self._load_servers_from_path(target)
        servers[alias] = server
        write_mcp_file(target, servers)
        return target

    def edit_server(
        self,
        alias: str,
        mutator: Callable[[MCPServerConfig], MCPServerConfig],
    ) -> tuple[Path, MCPServerConfig]:
        view = self.get_view(alias)
        if view is None:
            raise MCPConfigManagerError(f"MCP server alias not found: {alias}")

        if view.source == "legacy":
            target = self.resolve_write_path()
            target_servers = self._load_servers_from_path(target)
            base = target_servers.get(alias, view.server)
            updated = mutator(base)
            target_servers[alias] = updated
            write_mcp_file(target, target_servers)
            return target, updated

        source_path = view.source_path
        if source_path is None:
            raise MCPConfigManagerError(
                f"Cannot edit alias {alias}: source path unknown ({view.source})"
            )
        servers = self._load_servers_from_path(source_path)
        current = servers.get(alias, view.server)
        updated = mutator(current)
        servers[alias] = updated
        write_mcp_file(source_path, servers)
        return source_path, updated

    def remove_server(self, alias: str) -> Path:
        view = self.get_view(alias)
        if view is None:
            raise MCPConfigManagerError(f"MCP server alias not found: {alias}")
        if view.source == "legacy":
            raise MCPConfigManagerError(
                "Alias is defined in legacy loom.toml config. "
                "Run `loom mcp migrate` first."
            )
        source_path = view.source_path
        if source_path is None:
            raise MCPConfigManagerError(
                f"Cannot remove alias {alias}: source path unknown ({view.source})"
            )
        servers = self._load_servers_from_path(source_path)
        if alias not in servers:
            raise MCPConfigManagerError(
                f"Alias {alias} is not writable in source {view.source}."
            )
        del servers[alias]
        write_mcp_file(source_path, servers)
        return source_path

    def migrate_legacy(self) -> tuple[Path, int, bool]:
        merged = self.load()
        legacy_aliases = [
            view.alias for view in merged.as_views() if view.source == "legacy"
        ]
        if not legacy_aliases:
            return self.resolve_write_path(), 0, False

        target = self.resolve_write_path()
        target_servers = self._load_servers_from_path(target)
        copied = 0
        for alias in legacy_aliases:
            if alias in target_servers:
                continue
            legacy_view = merged.get(alias)
            if legacy_view is None:
                continue
            target_servers[alias] = legacy_view.server
            copied += 1
        write_mcp_file(target, target_servers)

        legacy_path = merged.legacy_config_path
        removed = False
        if legacy_path is not None and legacy_path.exists():
            removed = remove_mcp_sections_from_loom_toml(legacy_path)
        return target, copied, removed

    def probe_server(self, alias: str) -> tuple[MCPServerView, list[dict]]:
        view = self.get_view(alias)
        if view is None:
            raise MCPConfigManagerError(f"MCP server alias not found: {alias}")
        return view, probe_mcp_tools(view)


def ensure_valid_alias(alias: str) -> str:
    clean = alias.strip()
    if not clean:
        raise MCPConfigManagerError("Alias cannot be empty.")
    if any(ch.isspace() for ch in clean):
        raise MCPConfigManagerError("Alias cannot contain whitespace.")
    return clean


def ensure_valid_env_key(key: str) -> str:
    cleaned = key.strip()
    if not cleaned:
        raise MCPConfigManagerError("Environment variable key cannot be empty.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cleaned):
        raise MCPConfigManagerError(
            f"Invalid environment variable key: {key!r}"
        )
    return cleaned


def parse_env_pairs(pairs: tuple[str, ...], refs: tuple[str, ...]) -> dict[str, str]:
    """Parse `KEY=VALUE` and `KEY=ENV` pairs into env map updates."""
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise MCPConfigManagerError(
                f"Invalid --env value {pair!r}; expected KEY=VALUE."
            )
        key, value = pair.split("=", 1)
        env[ensure_valid_env_key(key)] = value
    for pair in refs:
        if "=" not in pair:
            raise MCPConfigManagerError(
                f"Invalid --env-ref value {pair!r}; expected KEY=ENV_VAR."
            )
        key, env_var = pair.split("=", 1)
        env_key = ensure_valid_env_key(key)
        ref_name = ensure_valid_env_key(env_var)
        env[env_key] = f"${{{ref_name}}}"
    return env


def parse_mcp_server_from_flags(
    *,
    command: str,
    args: tuple[str, ...],
    env_pairs: tuple[str, ...],
    env_refs: tuple[str, ...],
    cwd: str,
    timeout: int,
    disabled: bool,
) -> MCPServerConfig:
    if not command.strip():
        raise MCPConfigManagerError("--command is required.")
    env = parse_env_pairs(env_pairs, env_refs)
    return MCPServerConfig(
        command=command.strip(),
        args=[arg for arg in args],
        env=env,
        cwd=cwd.strip(),
        timeout_seconds=_parse_timeout(timeout),
        enabled=not disabled,
    )


def merge_server_edits(
    *,
    current: MCPServerConfig,
    command: str | None,
    args: tuple[str, ...],
    env_pairs: tuple[str, ...],
    env_refs: tuple[str, ...],
    cwd: str | None,
    timeout: int | None,
    disabled: bool,
) -> MCPServerConfig:
    env = dict(current.env)
    if env_pairs or env_refs:
        env.update(parse_env_pairs(env_pairs, env_refs))
    new_command = current.command
    if command is not None and command.strip():
        new_command = command.strip()
    new_args = list(current.args) if not args else [arg for arg in args]
    new_cwd = current.cwd if cwd is None else cwd.strip()
    new_timeout = current.timeout_seconds if timeout is None else _parse_timeout(timeout)
    enabled = current.enabled
    if disabled:
        enabled = False
    return MCPServerConfig(
        command=new_command,
        args=new_args,
        env=env,
        cwd=new_cwd,
        timeout_seconds=new_timeout,
        enabled=enabled,
    )


def validate_legacy_toml(path: Path | None) -> None:
    """Raise when legacy config cannot be parsed."""
    if path is None or not path.exists():
        return
    try:
        with open(path, "rb") as f:
            tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e


def probe_mcp_tools(view: MCPServerView) -> list[dict]:
    """Probe a configured MCP server by issuing a tools/list request."""
    from loom.integrations.mcp_tools import _MCPStdioClient  # noqa: PLC2701

    client = _MCPStdioClient(alias=view.alias, server=view.server)
    return client.list_tools()
