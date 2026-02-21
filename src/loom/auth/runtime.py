"""Runtime auth profile selection and credential resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loom.auth.config import (
    AuthConfigError,
    AuthProfile,
    load_merged_auth_config,
)
from loom.auth.secrets import SecretResolutionError, SecretResolver


class AuthResolutionError(Exception):
    """Raised when auth profile selection or resolution fails."""


def parse_auth_profile_overrides(pairs: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated `selector=profile_id` values."""
    result: dict[str, str] = {}
    for pair in pairs:
        raw = str(pair or "").strip()
        if not raw:
            continue
        if "=" not in raw:
            raise AuthResolutionError(
                f"Invalid --auth-profile value {pair!r}; expected selector=profile_id."
            )
        selector, profile_id = raw.split("=", 1)
        selector = selector.strip()
        profile_id = profile_id.strip()
        if not selector:
            raise AuthResolutionError(
                f"Invalid --auth-profile value {pair!r}; selector is empty."
            )
        if not profile_id:
            raise AuthResolutionError(
                f"Invalid --auth-profile value {pair!r}; profile id is empty."
            )
        result[selector] = profile_id
    return result


def _coerce_overrides(raw: object) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AuthResolutionError(
            "Task metadata field 'auth_profile_overrides' must be a dict."
        )
    result: dict[str, str] = {}
    for key, value in raw.items():
        selector = str(key or "").strip()
        profile_id = str(value or "").strip()
        if not selector or not profile_id:
            continue
        result[selector] = profile_id
    return result


def _resolve_env_value(value: str, *, resolver: SecretResolver) -> str:
    try:
        return resolver.resolve_maybe(value)
    except SecretResolutionError as e:
        raise AuthResolutionError(str(e)) from e


@dataclass(frozen=True)
class RunAuthContext:
    """Auth profiles selected for one run."""

    profiles: dict[str, AuthProfile] = field(default_factory=dict)
    selected_by_selector: dict[str, AuthProfile] = field(default_factory=dict)
    selected_by_provider: dict[str, AuthProfile] = field(default_factory=dict)
    source_user_path: Path | None = None
    source_explicit_path: Path | None = None
    source_workspace_defaults_path: Path | None = None
    secret_resolver: SecretResolver = field(default_factory=SecretResolver)

    def profile_for_provider(self, provider: str) -> AuthProfile | None:
        key = str(provider or "").strip()
        if not key:
            return None
        return self.selected_by_provider.get(key)

    def profile_for_selector(self, selector: str) -> AuthProfile | None:
        key = str(selector or "").strip()
        if not key:
            return None
        return self.selected_by_selector.get(key)

    def env_for_mcp_alias(self, alias: str) -> dict[str, str]:
        """Resolve run-scoped env overrides for one MCP alias.

        MCP auth is intentionally managed by MCP connection aliases (e.g. mcp-remote
        OAuth session state), not by `/auth` selector routing.
        """
        del alias
        return {}

    def resolve_secret_ref(self, secret_ref: str) -> str:
        """Resolve one secret reference in this run's auth scope."""
        try:
            return self.secret_resolver.resolve(secret_ref)
        except SecretResolutionError as e:
            raise AuthResolutionError(str(e)) from e


def build_run_auth_context(
    *,
    workspace: Path | None,
    metadata: dict[str, Any] | None,
    explicit_auth_path: Path | None = None,
) -> RunAuthContext:
    """Build the selected auth context for one run."""
    task_metadata = metadata if isinstance(metadata, dict) else {}
    raw_auth_path = str(task_metadata.get("auth_config_path", "")).strip()

    metadata_auth_path: Path | None = None
    if raw_auth_path:
        metadata_auth_path = Path(raw_auth_path).expanduser().resolve()

    chosen_explicit = explicit_auth_path or metadata_auth_path
    try:
        merged = load_merged_auth_config(
            workspace=workspace,
            explicit_path=chosen_explicit,
        )
    except AuthConfigError as e:
        raise AuthResolutionError(str(e)) from e

    selections: dict[str, str] = {}
    # Base: user defaults from auth.toml.
    selections.update(merged.config.defaults)
    # Workspace defaults override user defaults.
    selections.update(merged.workspace_defaults)
    # Explicit run metadata overrides everything.
    explicit_overrides = _coerce_overrides(task_metadata.get("auth_profile_overrides"))
    selections.update(explicit_overrides)
    explicit_selectors = set(explicit_overrides.keys())
    secret_resolver = SecretResolver()

    selected_by_selector: dict[str, AuthProfile] = {}
    selected_by_provider: dict[str, AuthProfile] = {}
    for selector, profile_id in selections.items():
        profile = merged.config.profiles.get(profile_id)
        if profile is None:
            if selector in explicit_selectors:
                raise AuthResolutionError(
                    f"Auth selector {selector!r} references unknown profile {profile_id!r}."
                )
            continue
        if selector.startswith("mcp."):
            if selector in explicit_selectors:
                raise AuthResolutionError(
                    f"Auth selector {selector!r} is no longer supported. "
                    "Use separate MCP aliases for per-account MCP auth."
                )
            continue
        selected_by_selector[selector] = profile
        if selector != profile.provider:
            if selector in explicit_selectors:
                raise AuthResolutionError(
                    f"Auth selector {selector!r} maps to profile {profile_id!r} "
                    f"for provider {profile.provider!r}; selector must match provider."
                )
            continue
        selected_by_provider[selector] = profile
        # Convenience lookup by profile id for direct references.
        selected_by_selector[profile_id] = profile

    # Auto-select single unambiguous provider profiles.
    profiles_by_provider: dict[str, list[AuthProfile]] = {}
    for profile in merged.config.profiles.values():
        profiles_by_provider.setdefault(profile.provider, []).append(profile)
    for provider, provider_profiles in profiles_by_provider.items():
        if provider in selected_by_provider:
            continue
        if len(provider_profiles) != 1:
            continue
        profile = provider_profiles[0]
        selected_by_provider[provider] = profile
        selected_by_selector.setdefault(provider, profile)
        selected_by_selector.setdefault(profile.profile_id, profile)

    return RunAuthContext(
        profiles=dict(merged.config.profiles),
        selected_by_selector=selected_by_selector,
        selected_by_provider=selected_by_provider,
        source_user_path=merged.user_path,
        source_explicit_path=merged.explicit_path,
        source_workspace_defaults_path=merged.workspace_defaults_path,
        secret_resolver=secret_resolver,
    )
