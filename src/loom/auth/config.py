"""Auth profile config loader and merge helpers."""

from __future__ import annotations

import os
import re
import tomllib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


class AuthConfigError(Exception):
    """Raised when auth profile config cannot be parsed or validated."""


@dataclass(frozen=True)
class AuthProfile:
    """One named, non-secret auth profile."""

    profile_id: str
    provider: str
    mode: str
    account_label: str = ""
    mcp_server: str = ""
    secret_ref: str = ""
    token_ref: str = ""
    scopes: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    command: str = ""
    auth_check: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    status: str = "ready"


@dataclass(frozen=True)
class AuthConfig:
    """Merged auth profile data."""

    profiles: dict[str, AuthProfile] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)
    resource_defaults: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MergedAuthConfig:
    """Auth config plus source metadata."""

    config: AuthConfig
    user_path: Path
    explicit_path: Path | None
    workspace_defaults: dict[str, str] = field(default_factory=dict)
    workspace_defaults_path: Path | None = None


_TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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


def _toml_array(values: list[str]) -> str:
    return "[" + ", ".join(_toml_escape(v) for v in values) + "]"


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
            # Best effort on non-POSIX platforms.
            pass
        try:
            yield
        finally:
            try:
                import fcntl  # POSIX only

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def default_user_auth_path() -> Path:
    """Default user auth profile path."""
    return Path.home() / ".loom" / "auth.toml"


def default_workspace_auth_defaults_path(workspace: Path) -> Path:
    """Default workspace auth defaults path."""
    return workspace / ".loom" / "auth.defaults.toml"


def _parse_string_map(raw: object, *, field_name: str, path: Path) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AuthConfigError(
            f"Invalid {field_name} in {path}: expected table/dict."
        )
    result: dict[str, str] = {}

    def _walk(node: dict, *, prefix: str = "") -> None:
        for key, value in node.items():
            if not isinstance(key, str):
                raise AuthConfigError(
                    f"Invalid {field_name} key in {path}: expected string key."
                )
            clean_key = key.strip()
            if not clean_key:
                continue
            dotted_key = f"{prefix}.{clean_key}" if prefix else clean_key
            if isinstance(value, dict):
                _walk(value, prefix=dotted_key)
                continue
            text = str(value).strip()
            if not text:
                continue
            result[dotted_key] = text

    _walk(raw)
    return result


def _parse_profile(profile_id: str, raw: object, *, path: Path) -> AuthProfile:
    if not isinstance(raw, dict):
        raise AuthConfigError(
            f"Invalid auth profile '{profile_id}' in {path}: expected table/dict."
        )

    provider = str(raw.get("provider", "")).strip()
    mode = str(raw.get("mode", "")).strip()
    if not provider:
        raise AuthConfigError(
            f"Auth profile '{profile_id}' in {path} is missing required 'provider'."
        )
    if not mode:
        raise AuthConfigError(
            f"Auth profile '{profile_id}' in {path} is missing required 'mode'."
        )

    scopes_raw = raw.get("scopes", [])
    scopes: list[str] = []
    if isinstance(scopes_raw, list):
        scopes = [str(item).strip() for item in scopes_raw if str(item).strip()]
    elif scopes_raw not in (None, ""):
        raise AuthConfigError(
            f"Invalid scopes in auth profile '{profile_id}' ({path}); expected list."
        )

    env_raw = raw.get("env", {})
    env = _parse_string_map(env_raw, field_name="profile.env", path=path)

    auth_check_raw = raw.get("auth_check", [])
    auth_check: list[str] = []
    if isinstance(auth_check_raw, list):
        auth_check = [
            str(item).strip()
            for item in auth_check_raw
            if str(item).strip()
        ]
    elif auth_check_raw not in (None, ""):
        raise AuthConfigError(
            f"Invalid auth_check in auth profile '{profile_id}' ({path}); expected list."
        )

    known_fields = {
        "provider",
        "mode",
        "account_label",
        "mcp_server",
        "secret_ref",
        "token_ref",
        "scopes",
        "env",
        "command",
        "auth_check",
        "status",
    }
    metadata: dict[str, str] = {}
    for key, value in raw.items():
        if key in known_fields:
            continue
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = str(value)

    profile = AuthProfile(
        profile_id=profile_id,
        provider=provider,
        mode=mode,
        account_label=str(raw.get("account_label", "")).strip(),
        mcp_server=str(raw.get("mcp_server", "")).strip(),
        secret_ref=str(raw.get("secret_ref", "")).strip(),
        token_ref=str(raw.get("token_ref", "")).strip(),
        scopes=scopes,
        env=env,
        command=str(raw.get("command", "")).strip(),
        auth_check=auth_check,
        metadata=metadata,
        status=str(raw.get("status", "ready")).strip().lower() or "ready",
    )
    _validate_profile(profile, path=path)
    return profile


def _validate_profile(profile: AuthProfile, *, path: Path | None = None) -> None:
    """Validate one auth profile shape and mode-specific requirements."""
    location = f" in {path}" if path is not None else ""
    profile_id = str(profile.profile_id or "").strip() or "<unknown>"
    provider = str(profile.provider or "").strip()
    mode = str(profile.mode or "").strip().lower()
    status = str(profile.status or "").strip().lower() or "ready"

    if status not in {"draft", "ready", "archived"}:
        raise AuthConfigError(
            f"Auth profile '{profile_id}'{location} has invalid status {status!r}."
        )

    if not provider:
        raise AuthConfigError(
            f"Auth profile '{profile_id}'{location} is missing required 'provider'."
        )
    if not mode:
        raise AuthConfigError(
            f"Auth profile '{profile_id}'{location} is missing required 'mode'."
        )
    mcp_server = str(profile.mcp_server or "").strip()
    if mcp_server and any(ch.isspace() for ch in mcp_server):
        raise AuthConfigError(
            f"Auth profile '{profile_id}'{location} has invalid mcp_server "
            f"{mcp_server!r}: whitespace is not allowed."
        )

    has_env = bool(profile.env)
    has_secret_ref = bool(str(profile.secret_ref or "").strip())
    has_token_ref = bool(str(profile.token_ref or "").strip())
    has_command = bool(str(profile.command or "").strip())

    if status == "draft":
        return

    if mode == "api_key" and not (has_secret_ref or has_env):
        raise AuthConfigError(
            f"Auth profile '{profile_id}'{location} mode='api_key' requires "
            "'secret_ref' or at least one 'env' mapping."
        )
    if mode in {"oauth2_pkce", "oauth2_device"} and not has_token_ref:
        raise AuthConfigError(
            f"Auth profile '{profile_id}'{location} mode={mode!r} requires 'token_ref'."
        )
    if mode == "cli_passthrough" and not has_command:
        raise AuthConfigError(
            f"Auth profile '{profile_id}'{location} mode='cli_passthrough' requires 'command'."
        )
    if mode == "env_passthrough" and not has_env:
        raise AuthConfigError(
            f"Auth profile '{profile_id}'{location} mode='env_passthrough' requires "
            "at least one 'env' mapping."
        )


def _parse_auth_section(raw: object, *, path: Path) -> AuthConfig:
    if raw is None:
        return AuthConfig()
    if not isinstance(raw, dict):
        raise AuthConfigError(
            f"Invalid [auth] section in {path}: expected table/dict."
        )
    if "mcp_alias_profiles" in raw:
        warnings.warn(
            (
                f"{path}: [auth.mcp_alias_profiles] is deprecated and ignored. "
                "Use [auth.profiles.<id>].mcp_server plus provider defaults "
                "or run-time selection."
            ),
            FutureWarning,
            stacklevel=3,
        )

    defaults = _parse_string_map(
        raw.get("defaults", {}),
        field_name="auth.defaults",
        path=path,
    )
    resource_defaults = _parse_string_map(
        raw.get("resource_defaults", {}),
        field_name="auth.resource_defaults",
        path=path,
    )
    profiles_raw = raw.get("profiles", {})
    if profiles_raw is None:
        profiles_raw = {}
    if not isinstance(profiles_raw, dict):
        raise AuthConfigError(
            f"Invalid auth.profiles in {path}: expected table/dict."
        )
    profiles: dict[str, AuthProfile] = {}
    for profile_id_raw, profile_raw in profiles_raw.items():
        profile_id = str(profile_id_raw).strip()
        if not profile_id:
            raise AuthConfigError(f"Auth profile id cannot be empty in {path}.")
        profiles[profile_id] = _parse_profile(profile_id, profile_raw, path=path)

    return AuthConfig(
        profiles=profiles,
        defaults=defaults,
        resource_defaults=resource_defaults,
    )


def load_auth_file(path: Path) -> AuthConfig:
    """Load one auth TOML file."""
    if not path.exists():
        return AuthConfig()
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise AuthConfigError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise AuthConfigError(f"Cannot read auth config {path}: {e}") from e
    return _parse_auth_section(raw.get("auth"), path=path)


def load_workspace_auth_defaults(path: Path) -> dict[str, str]:
    """Load workspace provider->profile defaults from auth.defaults.toml."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise AuthConfigError(f"Invalid TOML in {path}: {e}") from e
    except OSError as e:
        raise AuthConfigError(f"Cannot read auth defaults {path}: {e}") from e

    if isinstance(raw.get("defaults"), dict):
        return _parse_string_map(
            raw.get("defaults"),
            field_name="defaults",
            path=path,
        )
    auth = raw.get("auth")
    if isinstance(auth, dict):
        return _parse_string_map(
            auth.get("defaults", {}),
            field_name="auth.defaults",
            path=path,
        )
    return {}


def merge_auth_config(base: AuthConfig, overlay: AuthConfig) -> AuthConfig:
    """Overlay profile/default maps."""
    merged_profiles = dict(base.profiles)
    merged_profiles.update(overlay.profiles)

    merged_defaults = dict(base.defaults)
    merged_defaults.update(overlay.defaults)

    merged_resource_defaults = dict(base.resource_defaults)
    merged_resource_defaults.update(overlay.resource_defaults)

    return AuthConfig(
        profiles=merged_profiles,
        defaults=merged_defaults,
        resource_defaults=merged_resource_defaults,
    )


def render_auth_toml(config: AuthConfig) -> str:
    """Render auth.toml content."""
    lines: list[str] = [
        "# Loom auth profile configuration",
        "# Stores profile metadata and secret references only.",
        "",
    ]

    if config.defaults:
        lines.append("[auth.defaults]")
        for selector in sorted(config.defaults):
            profile_id = str(config.defaults.get(selector, "")).strip()
            if not profile_id:
                continue
            lines.append(f"{_toml_key(selector)} = {_toml_escape(profile_id)}")
        lines.append("")

    if config.resource_defaults:
        lines.append("[auth.resource_defaults]")
        for resource_id in sorted(config.resource_defaults):
            profile_id = str(config.resource_defaults.get(resource_id, "")).strip()
            if not profile_id:
                continue
            lines.append(f"{_toml_key(resource_id)} = {_toml_escape(profile_id)}")
        lines.append("")

    for profile_id in sorted(config.profiles):
        profile = config.profiles[profile_id]
        profile_key = _toml_key(profile_id)
        lines.append(f"[auth.profiles.{profile_key}]")
        lines.append(f"provider = {_toml_escape(profile.provider)}")
        lines.append(f"mode = {_toml_escape(profile.mode)}")
        if profile.account_label:
            lines.append(f"account_label = {_toml_escape(profile.account_label)}")
        if profile.mcp_server:
            lines.append(f"mcp_server = {_toml_escape(profile.mcp_server)}")
        if profile.secret_ref:
            lines.append(f"secret_ref = {_toml_escape(profile.secret_ref)}")
        if profile.token_ref:
            lines.append(f"token_ref = {_toml_escape(profile.token_ref)}")
        if profile.scopes:
            lines.append(f"scopes = {_toml_array(list(profile.scopes))}")
        if profile.command:
            lines.append(f"command = {_toml_escape(profile.command)}")
        if profile.auth_check:
            lines.append(f"auth_check = {_toml_array(list(profile.auth_check))}")
        if str(profile.status or "").strip().lower() not in {"", "ready"}:
            lines.append(
                f"status = {_toml_escape(str(profile.status).strip().lower())}"
            )
        for key in sorted(profile.metadata):
            value = str(profile.metadata.get(key, "")).strip()
            if not value:
                continue
            lines.append(f"{_toml_key(key)} = {_toml_escape(value)}")
        if profile.env:
            lines.append("")
            lines.append(f"[auth.profiles.{profile_key}.env]")
            for env_key in sorted(profile.env):
                lines.append(f"{_toml_key(env_key)} = {_toml_escape(profile.env[env_key])}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_auth_file(path: Path, config: AuthConfig) -> None:
    """Atomically write auth.toml."""
    lock_path = path.with_name(f".{path.name}.lock")
    with _file_lock(lock_path):
        _atomic_write_text(path, render_auth_toml(config))


def _mutate_auth_file(path: Path, mutator) -> AuthConfig:
    lock_path = path.with_name(f".{path.name}.lock")
    with _file_lock(lock_path):
        current = load_auth_file(path)
        updated = mutator(current)
        if updated != current:
            _atomic_write_text(path, render_auth_toml(updated))
        return updated


def resolve_auth_write_path(
    *,
    explicit_path: Path | None,
    user_path: Path | None = None,
) -> Path:
    """Resolve writable auth.toml path."""
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    return (user_path or default_user_auth_path()).expanduser()


def upsert_auth_profile(
    path: Path,
    profile: AuthProfile,
    *,
    must_exist: bool | None = None,
) -> AuthConfig:
    """Add or replace one auth profile in auth.toml."""
    _validate_profile(profile)

    def _mutate(current: AuthConfig) -> AuthConfig:
        exists = profile.profile_id in current.profiles
        if must_exist is True and not exists:
            raise AuthConfigError(f"Auth profile not found: {profile.profile_id}")
        if must_exist is False and exists:
            raise AuthConfigError(f"Auth profile already exists: {profile.profile_id}")
        profiles = dict(current.profiles)
        profiles[profile.profile_id] = profile
        return AuthConfig(
            profiles=profiles,
            defaults=dict(current.defaults),
            resource_defaults=dict(current.resource_defaults),
        )

    return _mutate_auth_file(path, _mutate)


def remove_auth_profile(path: Path, profile_id: str) -> AuthConfig:
    """Remove one auth profile and clean dangling references."""
    clean_profile_id = str(profile_id or "").strip()
    if not clean_profile_id:
        raise AuthConfigError("Auth profile id cannot be empty.")
    def _mutate(current: AuthConfig) -> AuthConfig:
        if clean_profile_id not in current.profiles:
            raise AuthConfigError(f"Auth profile not found: {clean_profile_id}")
        profiles = dict(current.profiles)
        del profiles[clean_profile_id]

        defaults = {
            selector: mapped
            for selector, mapped in current.defaults.items()
            if mapped != clean_profile_id
        }
        return AuthConfig(
            profiles=profiles,
            defaults=defaults,
            resource_defaults={
                selector: mapped
                for selector, mapped in current.resource_defaults.items()
                if mapped != clean_profile_id
            },
        )

    return _mutate_auth_file(path, _mutate)


def set_auth_default(
    path: Path,
    *,
    selector: str,
    profile_id: str | None,
) -> AuthConfig:
    """Set or clear selector default mapping in auth.toml."""
    clean_selector = str(selector or "").strip()
    if not clean_selector:
        raise AuthConfigError("Auth default selector cannot be empty.")

    def _mutate(current: AuthConfig) -> AuthConfig:
        defaults = dict(current.defaults)
        if profile_id is None:
            defaults.pop(clean_selector, None)
        else:
            clean_profile_id = str(profile_id or "").strip()
            if not clean_profile_id:
                raise AuthConfigError("Auth default profile id cannot be empty.")
            defaults[clean_selector] = clean_profile_id
        return AuthConfig(
            profiles=dict(current.profiles),
            defaults=defaults,
            resource_defaults=dict(current.resource_defaults),
        )

    return _mutate_auth_file(path, _mutate)


def set_auth_resource_default(
    path: Path,
    *,
    resource_id: str,
    profile_id: str | None,
) -> AuthConfig:
    """Set or clear user-scoped resource default mapping in auth.toml."""
    clean_resource_id = str(resource_id or "").strip()
    if not clean_resource_id:
        raise AuthConfigError("Auth resource_id selector cannot be empty.")
    def _mutate(current: AuthConfig) -> AuthConfig:
        defaults = dict(current.resource_defaults)
        if profile_id is None:
            defaults.pop(clean_resource_id, None)
        else:
            clean_profile_id = str(profile_id or "").strip()
            if not clean_profile_id:
                raise AuthConfigError("Auth default profile id cannot be empty.")
            defaults[clean_resource_id] = clean_profile_id
        return AuthConfig(
            profiles=dict(current.profiles),
            defaults=dict(current.defaults),
            resource_defaults=defaults,
        )

    return _mutate_auth_file(path, _mutate)


def render_workspace_auth_defaults(defaults: dict[str, str]) -> str:
    """Render .loom/auth.defaults.toml content."""
    lines: list[str] = [
        "# Loom workspace auth defaults",
        "# Maps provider/selector -> profile id (no secrets).",
        "",
        "[auth.defaults]",
    ]
    for selector in sorted(defaults):
        profile_id = str(defaults.get(selector, "")).strip()
        if not profile_id:
            continue
        lines.append(f"{_toml_key(selector)} = {_toml_escape(profile_id)}")
    lines.append("")
    return "\n".join(lines)


def write_workspace_auth_defaults(path: Path, defaults: dict[str, str]) -> None:
    """Atomically write workspace auth defaults file."""
    lock_path = path.with_name(f".{path.name}.lock")
    with _file_lock(lock_path):
        _atomic_write_text(path, render_workspace_auth_defaults(defaults))


def _mutate_workspace_auth_defaults(path: Path, mutator) -> dict[str, str]:
    lock_path = path.with_name(f".{path.name}.lock")
    with _file_lock(lock_path):
        current = load_workspace_auth_defaults(path)
        updated = mutator(dict(current))
        if updated != current:
            _atomic_write_text(path, render_workspace_auth_defaults(updated))
        return updated


def set_workspace_auth_default(
    path: Path,
    *,
    selector: str,
    profile_id: str | None,
) -> dict[str, str]:
    """Set or clear one workspace auth default and write the file."""
    clean_selector = str(selector or "").strip()
    if not clean_selector:
        raise AuthConfigError("Auth default selector cannot be empty.")

    def _mutate(current: dict[str, str]) -> dict[str, str]:
        if profile_id is None:
            current.pop(clean_selector, None)
        else:
            clean_profile = str(profile_id or "").strip()
            if not clean_profile:
                raise AuthConfigError("Auth default profile id cannot be empty.")
            current[clean_selector] = clean_profile
        return current

    return _mutate_workspace_auth_defaults(path, _mutate)


def load_merged_auth_config(
    *,
    workspace: Path | None = None,
    explicit_path: Path | None = None,
    user_path: Path | None = None,
) -> MergedAuthConfig:
    """Load merged auth config (explicit overlay over user config)."""
    user_cfg_path = (user_path or default_user_auth_path()).expanduser()
    explicit_cfg_path = explicit_path.expanduser().resolve() if explicit_path else None

    user_cfg = load_auth_file(user_cfg_path)
    explicit_cfg = (
        load_auth_file(explicit_cfg_path)
        if explicit_cfg_path is not None
        else AuthConfig()
    )
    merged = merge_auth_config(user_cfg, explicit_cfg)

    ws_defaults: dict[str, str] = {}
    ws_defaults_path: Path | None = None
    if workspace is not None:
        ws_defaults_path = default_workspace_auth_defaults_path(workspace.resolve())
        ws_defaults = load_workspace_auth_defaults(ws_defaults_path)

    return MergedAuthConfig(
        config=merged,
        user_path=user_cfg_path,
        explicit_path=explicit_cfg_path,
        workspace_defaults=ws_defaults,
        workspace_defaults_path=ws_defaults_path,
    )
