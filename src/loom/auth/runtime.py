"""Runtime auth profile selection and credential resolution helpers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loom.auth.config import (
    AuthConfigError,
    AuthProfile,
    load_merged_auth_config,
)
from loom.auth.resources import (
    active_bindings_for_resource,
    default_workspace_auth_resources_path,
    has_active_binding,
    load_workspace_auth_resources,
    resolve_resource,
)
from loom.auth.secrets import SecretResolutionError, SecretResolver


class AuthResolutionError(Exception):
    """Raised when auth profile selection or resolution fails."""


@dataclass(frozen=True)
class AuthResourceRequirement:
    """One required auth resource for run preflight."""

    provider: str
    source: str = "api"
    modes: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    required_env_keys: tuple[str, ...] = ()
    mcp_server: str = ""
    resource_ref: str = ""
    resource_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "source": self.source,
        }
        if self.modes:
            payload["modes"] = list(self.modes)
        if self.scopes:
            payload["scopes"] = list(self.scopes)
        if self.required_env_keys:
            payload["required_env_keys"] = list(self.required_env_keys)
        if self.mcp_server:
            payload["mcp_server"] = self.mcp_server
        if self.resource_ref:
            payload["resource_ref"] = self.resource_ref
        if self.resource_id:
            payload["resource_id"] = self.resource_id
        return payload


@dataclass(frozen=True)
class UnresolvedAuthResource:
    """Structured unresolved auth requirement detail."""

    provider: str
    source: str
    reason: str
    message: str
    candidates: tuple[str, ...] = ()
    modes: tuple[str, ...] = ()
    required_env_keys: tuple[str, ...] = ()
    mcp_server: str = ""
    resource_ref: str = ""
    resource_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "source": self.source,
            "reason": self.reason,
            "message": self.message,
            "candidates": list(self.candidates),
        }
        if self.modes:
            payload["modes"] = list(self.modes)
        if self.required_env_keys:
            payload["required_env_keys"] = list(self.required_env_keys)
        if self.mcp_server:
            payload["mcp_server"] = self.mcp_server
        if self.resource_ref:
            payload["resource_ref"] = self.resource_ref
        if self.resource_id:
            payload["resource_id"] = self.resource_id
        return payload


class UnresolvedAuthResourcesError(AuthResolutionError):
    """Raised when required auth resources cannot be resolved deterministically."""

    def __init__(
        self,
        message: str,
        *,
        unresolved: list[UnresolvedAuthResource],
        defaults_user: dict[str, str],
        defaults_workspace: dict[str, str],
        explicit_overrides: dict[str, str],
        required_resources: list[AuthResourceRequirement],
    ) -> None:
        super().__init__(message)
        self.unresolved = list(unresolved)
        self.defaults_user = dict(defaults_user)
        self.defaults_workspace = dict(defaults_workspace)
        self.explicit_overrides = dict(explicit_overrides)
        self.required_resources = list(required_resources)

    def to_payload(self) -> dict[str, Any]:
        remediation: list[str] = []
        for item in self.unresolved:
            selector = (
                str(item.resource_id or "").strip()
                or str(item.resource_ref or "").strip()
                or str(item.provider or "").strip()
            )
            if not selector:
                selector = "<selector>"
            if item.candidates:
                profile_id = item.candidates[0]
                remediation.append(f"loom auth select {selector} {profile_id}")
                remediation.append(
                    "loom run <goal> "
                    f"--auth-profile {selector}={profile_id}"
                )
            else:
                remediation.append(
                    "loom auth profile add <profile-id> "
                    f"--provider {item.provider or '<provider>'} --mode <mode>"
                )
        remediation = list(dict.fromkeys(remediation))

        return {
            "code": "auth_unresolved",
            "message": str(self),
            "unresolved": [item.as_dict() for item in self.unresolved],
            "defaults": {
                "user": dict(self.defaults_user),
                "workspace": dict(self.defaults_workspace),
                "effective": {
                    **self.defaults_user,
                    **self.defaults_workspace,
                },
            },
            "explicit_overrides": dict(self.explicit_overrides),
            "required_resources": [
                req.as_dict() for req in self.required_resources
            ],
            "remediation": {
                "commands": remediation,
                "api_metadata_example": {
                    "auth_profile_overrides": {
                        (
                            str(item.resource_id or "").strip()
                            or str(item.resource_ref or "").strip()
                            or str(item.provider or "").strip()
                            or "<selector>"
                        ): (
                            item.candidates[0] if item.candidates else "<profile-id>"
                        )
                        for item in self.unresolved
                    }
                },
            },
        }


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


def _normalize_source(raw: object) -> str:
    source = str(raw or "api").strip().lower()
    if source in {"mcp", "api"}:
        return source
    return "api"


def _normalize_str_tuple(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list | tuple):
        return ()
    values: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text:
            values.append(text)
    return tuple(values)


def _coerce_requirement(raw: object) -> AuthResourceRequirement | None:
    if raw is None:
        return None
    if isinstance(raw, AuthResourceRequirement):
        return raw

    provider = ""
    source_raw: object = "api"
    modes_raw: object = []
    scopes_raw: object = []
    env_keys_raw: object = []
    mcp_server_raw: object = ""
    resource_ref_raw: object = ""
    resource_id_raw: object = ""

    if isinstance(raw, dict):
        provider = str(raw.get("provider", "")).strip()
        source_raw = raw.get("source", "api")
        modes_raw = raw.get("modes", [])
        scopes_raw = raw.get("scopes", [])
        env_keys_raw = raw.get("required_env_keys", [])
        mcp_server_raw = raw.get("mcp_server", "")
        resource_ref_raw = raw.get("resource_ref", "")
        resource_id_raw = raw.get("resource_id", "")
    else:
        provider = str(getattr(raw, "provider", "")).strip()
        source_raw = getattr(raw, "source", "api")
        modes_raw = getattr(raw, "modes", [])
        scopes_raw = getattr(raw, "scopes", [])
        env_keys_raw = getattr(raw, "required_env_keys", [])
        mcp_server_raw = getattr(raw, "mcp_server", "")
        resource_ref_raw = getattr(raw, "resource_ref", "")
        resource_id_raw = getattr(raw, "resource_id", "")
    resource_ref = str(resource_ref_raw or "").strip()
    resource_id = str(resource_id_raw or "").strip()
    if not provider and not resource_ref and not resource_id:
        return None
    return AuthResourceRequirement(
        provider=provider,
        source=_normalize_source(source_raw),
        modes=_normalize_str_tuple(modes_raw),
        scopes=_normalize_str_tuple(scopes_raw),
        required_env_keys=_normalize_str_tuple(env_keys_raw),
        mcp_server=str(mcp_server_raw or "").strip(),
        resource_ref=resource_ref,
        resource_id=resource_id,
    )


def coerce_auth_requirements(raw: object) -> list[AuthResourceRequirement]:
    """Normalize auth requirement declarations from process/tool/metadata shapes."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        # Support {"required": [...]} style blocks.
        if isinstance(raw.get("required"), list):
            raw = raw.get("required")
        else:
            raw = [raw]
    if not isinstance(raw, list | tuple):
        return []

    result: list[AuthResourceRequirement] = []
    for item in raw:
        parsed = _coerce_requirement(item)
        if parsed is None:
            continue
        result.append(parsed)

    deduped: list[AuthResourceRequirement] = []
    seen: set[
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            str,
            str,
            str,
        ]
    ] = set()
    for req in result:
        key = (
            req.provider,
            req.source,
            tuple(req.modes),
            tuple(req.scopes),
            tuple(req.required_env_keys),
            req.mcp_server,
            req.resource_ref,
            req.resource_id,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(req)
    return deduped


def serialize_auth_requirements(
    requirements: list[AuthResourceRequirement],
) -> list[dict[str, Any]]:
    """Serialize normalized requirement objects for task metadata persistence."""
    return [item.as_dict() for item in requirements]


def _profile_matches_requirement(
    profile: AuthProfile,
    requirement: AuthResourceRequirement,
) -> bool:
    required_provider = str(requirement.provider or "").strip()
    if required_provider and profile.provider != required_provider:
        return False

    if requirement.modes and profile.mode not in requirement.modes:
        return False
    if requirement.scopes:
        profile_scopes = {str(item).strip() for item in profile.scopes if str(item).strip()}
        if not set(requirement.scopes).issubset(profile_scopes):
            return False
    if requirement.required_env_keys:
        profile_env_keys = set(profile.env.keys())
        if not set(requirement.required_env_keys).issubset(profile_env_keys):
            return False
    if requirement.source == "mcp":
        mcp_server = str(profile.mcp_server or "").strip()
        if not mcp_server:
            return False
        if requirement.mcp_server and mcp_server != requirement.mcp_server:
            return False

    return True


def _resolve_env_value(value: str, *, resolver: SecretResolver) -> str:
    try:
        return resolver.resolve_maybe(value)
    except SecretResolutionError as e:
        raise AuthResolutionError(str(e)) from e


def _parse_expiry_epoch(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    else:
        text = str(raw).strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            normalized = text
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(normalized)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.timestamp()
    if value > 10_000_000_000:
        value = value / 1000.0
    if value <= 0:
        return None
    return value


def _extract_oauth_expiry_epoch(
    *,
    profile: AuthProfile,
    token_payload: dict[str, Any] | None,
) -> float | None:
    candidates: list[object] = []
    if isinstance(token_payload, dict):
        for key in ("expires_at", "expires_at_epoch", "expires_on"):
            if key in token_payload:
                candidates.append(token_payload.get(key))
    for key, value in profile.metadata.items():
        lowered = str(key or "").strip().lower()
        if lowered in {"expires_at", "expires_at_epoch", "expires_on"}:
            candidates.append(value)
    for candidate in candidates:
        parsed = _parse_expiry_epoch(candidate)
        if parsed is not None:
            return parsed
    return None


def _validate_oauth_requirement(
    *,
    profile: AuthProfile,
    requirement: AuthResourceRequirement,
    resolver: SecretResolver,
    candidate_ids: tuple[str, ...],
) -> UnresolvedAuthResource | None:
    mode = str(profile.mode or "").strip().lower()
    if mode not in {"oauth2_pkce", "oauth2_device"}:
        return None

    token_ref = str(profile.token_ref or "").strip()
    if not token_ref:
        return UnresolvedAuthResource(
            provider=requirement.provider,
            source=requirement.source,
            reason="auth_invalid",
            message=(
                f"Profile {profile.profile_id!r} uses {mode!r} but has no token_ref."
            ),
            candidates=candidate_ids,
            modes=requirement.modes,
            required_env_keys=requirement.required_env_keys,
            mcp_server=requirement.mcp_server,
        )

    try:
        token_value = resolver.resolve(token_ref).strip()
    except SecretResolutionError as e:
        return UnresolvedAuthResource(
            provider=requirement.provider,
            source=requirement.source,
            reason="auth_missing",
            message=(
                f"Profile {profile.profile_id!r} token_ref could not be resolved: {e}"
            ),
            candidates=candidate_ids,
            modes=requirement.modes,
            required_env_keys=requirement.required_env_keys,
            mcp_server=requirement.mcp_server,
        )

    if not token_value:
        return UnresolvedAuthResource(
            provider=requirement.provider,
            source=requirement.source,
            reason="auth_missing",
            message=(
                f"Profile {profile.profile_id!r} token_ref resolved to an empty value."
            ),
            candidates=candidate_ids,
            modes=requirement.modes,
            required_env_keys=requirement.required_env_keys,
            mcp_server=requirement.mcp_server,
        )

    token_payload: dict[str, Any] | None = None
    try:
        parsed = json.loads(token_value)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        if not isinstance(parsed, dict):
            return UnresolvedAuthResource(
                provider=requirement.provider,
                source=requirement.source,
                reason="auth_invalid",
                message=(
                    f"Profile {profile.profile_id!r} token payload must be an object "
                    "when JSON-encoded."
                ),
                candidates=candidate_ids,
                modes=requirement.modes,
                required_env_keys=requirement.required_env_keys,
                mcp_server=requirement.mcp_server,
            )
        token_payload = parsed
        access_token = str(token_payload.get("access_token", "")).strip()
        if not access_token:
            return UnresolvedAuthResource(
                provider=requirement.provider,
                source=requirement.source,
                reason="auth_invalid",
                message=(
                    f"Profile {profile.profile_id!r} token payload is missing "
                    "'access_token'."
                ),
                candidates=candidate_ids,
                modes=requirement.modes,
                required_env_keys=requirement.required_env_keys,
                mcp_server=requirement.mcp_server,
            )

    expires_at = _extract_oauth_expiry_epoch(
        profile=profile,
        token_payload=token_payload,
    )
    if expires_at is not None and expires_at <= time.time():
        return UnresolvedAuthResource(
            provider=requirement.provider,
            source=requirement.source,
            reason="auth_expired",
            message=(
                f"Profile {profile.profile_id!r} OAuth credentials are expired "
                f"(expires_at={int(expires_at)})."
            ),
            candidates=candidate_ids,
            modes=requirement.modes,
            required_env_keys=requirement.required_env_keys,
            mcp_server=requirement.mcp_server,
        )
    return None


@dataclass(frozen=True)
class RunAuthContext:
    """Auth profiles selected for one run."""

    profiles: dict[str, AuthProfile] = field(default_factory=dict)
    selected_by_selector: dict[str, AuthProfile] = field(default_factory=dict)
    selected_by_provider: dict[str, AuthProfile] = field(default_factory=dict)
    selected_by_mcp_alias: dict[str, AuthProfile] = field(default_factory=dict)
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

    def profile_for_mcp_alias(self, alias: str) -> AuthProfile | None:
        key = str(alias or "").strip()
        if not key:
            return None
        return self.selected_by_mcp_alias.get(key)

    def env_for_mcp_alias(self, alias: str) -> dict[str, str]:
        """Resolve run-scoped env overrides for one MCP alias.
        """
        profile = self.profile_for_mcp_alias(alias)
        if profile is None:
            return {}
        resolved: dict[str, str] = {}
        for key, value in profile.env.items():
            resolved[key] = _resolve_env_value(value, resolver=self.secret_resolver)
        return resolved

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
    required_resources: list[AuthResourceRequirement] | None = None,
    available_mcp_aliases: set[str] | None = None,
) -> RunAuthContext:
    """Build the selected auth context for one run."""
    task_metadata = metadata if isinstance(metadata, dict) else {}
    raw_auth_path = str(task_metadata.get("auth_config_path", "")).strip()
    raw_workspace_override = str(task_metadata.get("auth_workspace", "")).strip()

    metadata_workspace: Path | None = None
    if raw_workspace_override:
        metadata_workspace = Path(raw_workspace_override).expanduser().resolve()

    metadata_auth_path: Path | None = None
    if raw_auth_path:
        metadata_auth_path = Path(raw_auth_path).expanduser().resolve()

    chosen_explicit = explicit_auth_path or metadata_auth_path
    chosen_workspace = metadata_workspace or workspace
    try:
        merged = load_merged_auth_config(
            workspace=chosen_workspace,
            explicit_path=chosen_explicit,
        )
    except AuthConfigError as e:
        raise AuthResolutionError(str(e)) from e

    resource_store = load_workspace_auth_resources(
        default_workspace_auth_resources_path(chosen_workspace.resolve())
    ) if chosen_workspace is not None else load_workspace_auth_resources(
        default_workspace_auth_resources_path(Path.cwd())
    )
    active_resources_by_id = {
        resource_id: resource
        for resource_id, resource in resource_store.resources.items()
        if str(getattr(resource, "status", "")).strip().lower() == "active"
    }
    active_resource_refs = {
        resource.resource_ref: resource.resource_id
        for resource in active_resources_by_id.values()
    }

    selections: dict[str, str] = {}
    # Base: user defaults from auth.toml.
    selections.update(merged.config.defaults)
    # Workspace defaults override user defaults.
    selections.update(merged.workspace_defaults)
    # User resource defaults override provider defaults.
    selections.update(merged.config.resource_defaults)
    # Workspace resource defaults override provider defaults.
    selections.update(resource_store.workspace_defaults)
    # Explicit run metadata overrides everything.
    explicit_overrides = _coerce_overrides(task_metadata.get("auth_profile_overrides"))
    selections.update(explicit_overrides)
    explicit_selectors = set(explicit_overrides.keys())
    secret_resolver = SecretResolver()

    selected_by_selector: dict[str, AuthProfile] = {}
    selected_by_provider: dict[str, AuthProfile] = {}

    def _select_profile(
        profile: AuthProfile,
        *,
        selector: str | None = None,
        propagate_provider: bool = True,
    ) -> None:
        if selector:
            selected_by_selector[selector] = profile
        if propagate_provider:
            selected_by_provider[profile.provider] = profile
            selected_by_selector[profile.provider] = profile
        selected_by_selector[profile.profile_id] = profile

    def _profile_status(profile: AuthProfile) -> str:
        return str(getattr(profile, "status", "ready") or "ready").strip().lower()

    def _is_profile_archived(profile: AuthProfile) -> bool:
        return _profile_status(profile) == "archived"

    def _is_profile_draft(profile: AuthProfile) -> bool:
        return _profile_status(profile) == "draft"

    def _is_profile_candidate(profile: AuthProfile) -> bool:
        return not _is_profile_archived(profile) and not _is_profile_draft(profile)

    def _is_resource_selector(selector: str) -> bool:
        clean = str(selector or "").strip()
        if not clean:
            return False
        if clean in active_resources_by_id:
            return True
        if clean in active_resource_refs:
            return True
        resolved = resolve_resource(
            resource_store,
            resource_id=clean,
            resource_ref=clean,
        )
        return resolved is not None

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
        if selector != profile.provider and not _is_resource_selector(selector):
            if selector in explicit_selectors:
                raise AuthResolutionError(
                    f"Auth selector {selector!r} maps to profile {profile_id!r} "
                    f"for provider {profile.provider!r}; selector must match provider "
                    "or a known resource selector."
                )
            continue
        _select_profile(
            profile,
            selector=selector,
            propagate_provider=(selector == profile.provider),
        )

    # Auto-select single unambiguous provider profiles.
    profiles_by_provider: dict[str, list[AuthProfile]] = {}
    for profile in merged.config.profiles.values():
        if not _is_profile_candidate(profile):
            continue
        profiles_by_provider.setdefault(profile.provider, []).append(profile)
    for provider, provider_profiles in profiles_by_provider.items():
        if provider in selected_by_provider:
            continue
        if len(provider_profiles) != 1:
            continue
        profile = provider_profiles[0]
        _select_profile(profile, selector=provider)

    selected_by_mcp_alias: dict[str, AuthProfile] = {}
    for profile in selected_by_provider.values():
        alias = str(profile.mcp_server or "").strip()
        if not alias:
            continue
        existing = selected_by_mcp_alias.get(alias)
        if existing is not None and existing.profile_id != profile.profile_id:
            raise AuthResolutionError(
                f"Multiple selected profiles target MCP alias {alias!r}: "
                f"{existing.profile_id!r}, {profile.profile_id!r}."
            )
        selected_by_mcp_alias[alias] = profile

    all_requirements = coerce_auth_requirements(required_resources)
    all_requirements.extend(
        coerce_auth_requirements(task_metadata.get("auth_required_resources"))
    )
    all_requirements = coerce_auth_requirements(all_requirements)
    available_aliases = {
        str(alias).strip()
        for alias in (available_mcp_aliases or set())
        if str(alias).strip()
    }

    unresolved: list[UnresolvedAuthResource] = []
    for requirement in all_requirements:
        resolved_resource = resolve_resource(
            resource_store,
            resource_id=requirement.resource_id,
            resource_ref=requirement.resource_ref,
            source=requirement.source,
            provider=requirement.provider,
            mcp_server=requirement.mcp_server,
        )

        effective_requirement = requirement
        if not requirement.provider and resolved_resource is not None:
            effective_requirement = AuthResourceRequirement(
                provider=resolved_resource.provider,
                source=requirement.source,
                modes=requirement.modes,
                scopes=requirement.scopes,
                required_env_keys=requirement.required_env_keys,
                mcp_server=requirement.mcp_server,
                resource_ref=requirement.resource_ref,
                resource_id=requirement.resource_id,
            )

        resource_scoped = bool(
            str(requirement.resource_id or "").strip()
            or str(requirement.resource_ref or "").strip()
            or str(requirement.mcp_server or "").strip()
        )
        if resource_scoped and resolved_resource is None:
            unresolved.append(
                UnresolvedAuthResource(
                    provider=effective_requirement.provider,
                    source=effective_requirement.source,
                    reason="blocked_missing_resource",
                    message=(
                        "Auth requirement references a resource that is not registered "
                        "in this workspace."
                    ),
                    candidates=(),
                    modes=effective_requirement.modes,
                    required_env_keys=effective_requirement.required_env_keys,
                    mcp_server=effective_requirement.mcp_server,
                    resource_ref=effective_requirement.resource_ref,
                    resource_id=effective_requirement.resource_id,
                )
            )
            continue

        selected: AuthProfile | None = None
        selector_candidates = [
            str(effective_requirement.resource_id or "").strip(),
            str(effective_requirement.resource_ref or "").strip(),
        ]
        if resolved_resource is not None:
            selector_candidates.extend(
                [resolved_resource.resource_id, resolved_resource.resource_ref]
            )
        selector_candidates.append(str(effective_requirement.provider or "").strip())
        for selector in selector_candidates:
            if not selector:
                continue
            selected = selected_by_selector.get(selector)
            if selected is not None:
                break
        if selected is None and effective_requirement.provider:
            selected = selected_by_provider.get(effective_requirement.provider)
        if selected is not None and _is_profile_archived(selected):
            selected = None
        if selected is not None and not _profile_matches_requirement(
            selected,
            effective_requirement,
        ):
            selected = None

        if resolved_resource is not None and selected is not None:
            if not has_active_binding(
                resource_store,
                resource_id=resolved_resource.resource_id,
                profile_id=selected.profile_id,
            ):
                unresolved.append(
                    UnresolvedAuthResource(
                        provider=effective_requirement.provider,
                        source=effective_requirement.source,
                        reason="needs_rebind",
                        message=(
                            f"Profile {selected.profile_id!r} is selected for "
                            f"{resolved_resource.resource_ref} but is not bound to it."
                        ),
                        candidates=tuple(
                            binding.profile_id
                            for binding in active_bindings_for_resource(
                                resource_store,
                                resolved_resource.resource_id,
                            )
                            if binding.profile_id in merged.config.profiles
                        ),
                        modes=effective_requirement.modes,
                        required_env_keys=effective_requirement.required_env_keys,
                        mcp_server=effective_requirement.mcp_server,
                        resource_ref=resolved_resource.resource_ref,
                        resource_id=resolved_resource.resource_id,
                    )
                )
                continue

        if resolved_resource is not None:
            candidates = sorted(
                (
                    merged.config.profiles.get(binding.profile_id)
                    for binding in active_bindings_for_resource(
                        resource_store,
                        resolved_resource.resource_id,
                    )
                ),
                key=lambda item: item.profile_id if item is not None else "",
            )
            candidates = [
                profile
                for profile in candidates
                if profile is not None
                and _is_profile_candidate(profile)
                and _profile_matches_requirement(profile, effective_requirement)
            ]
        else:
            candidates = sorted(
                (
                    profile
                    for profile in merged.config.profiles.values()
                    if _is_profile_candidate(profile)
                    and _profile_matches_requirement(profile, effective_requirement)
                ),
                key=lambda item: item.profile_id,
            )
        candidate_ids = tuple(item.profile_id for item in candidates)

        if selected is None:
            if len(candidates) == 1:
                selected = candidates[0]
                _select_profile(selected, selector=selected.provider)
                if resolved_resource is not None:
                    selected_by_selector[resolved_resource.resource_id] = selected
                    selected_by_selector[resolved_resource.resource_ref] = selected
            elif len(candidates) > 1:
                unresolved.append(
                    UnresolvedAuthResource(
                        provider=effective_requirement.provider,
                        source=effective_requirement.source,
                        reason=(
                            "blocked_ambiguous_binding"
                            if resolved_resource is not None
                            else "ambiguous"
                        ),
                        message=(
                            "Multiple auth profiles match this requirement."
                            if resolved_resource is None
                            else (
                                "Multiple bound profiles match "
                                f"{resolved_resource.resource_ref!r}."
                            )
                        ),
                        candidates=candidate_ids,
                        modes=effective_requirement.modes,
                        required_env_keys=effective_requirement.required_env_keys,
                        mcp_server=effective_requirement.mcp_server,
                        resource_ref=(
                            resolved_resource.resource_ref
                            if resolved_resource is not None
                            else effective_requirement.resource_ref
                        ),
                        resource_id=(
                            resolved_resource.resource_id
                            if resolved_resource is not None
                            else effective_requirement.resource_id
                        ),
                    )
                )
                continue
            else:
                provider_candidates = sorted(
                    (
                        profile
                        for profile in merged.config.profiles.values()
                        if _is_profile_candidate(profile)
                        and _profile_matches_requirement(profile, effective_requirement)
                    ),
                    key=lambda item: item.profile_id,
                )
                provider_candidate_ids = tuple(
                    profile.profile_id for profile in provider_candidates
                )
                if (
                    resolved_resource is not None
                    and provider_candidate_ids
                ):
                    unresolved.append(
                        UnresolvedAuthResource(
                            provider=effective_requirement.provider,
                            source=effective_requirement.source,
                            reason="needs_rebind",
                            message=(
                                "Matching profiles exist for this provider, but none are "
                                f"bound to {resolved_resource.resource_ref!r}."
                            ),
                            candidates=provider_candidate_ids,
                            modes=effective_requirement.modes,
                            required_env_keys=effective_requirement.required_env_keys,
                            mcp_server=effective_requirement.mcp_server,
                            resource_ref=resolved_resource.resource_ref,
                            resource_id=resolved_resource.resource_id,
                        )
                    )
                else:
                    unresolved.append(
                        UnresolvedAuthResource(
                            provider=effective_requirement.provider,
                            source=effective_requirement.source,
                            reason="missing",
                            message=(
                                f"No auth profile matches provider "
                                f"{effective_requirement.provider!r} for "
                                f"source {effective_requirement.source!r}."
                            ),
                            candidates=(),
                            modes=effective_requirement.modes,
                            required_env_keys=effective_requirement.required_env_keys,
                            mcp_server=effective_requirement.mcp_server,
                            resource_ref=effective_requirement.resource_ref,
                            resource_id=effective_requirement.resource_id,
                        )
                    )
                continue

        selected_status = _profile_status(selected)
        if selected_status == "draft":
            unresolved.append(
                UnresolvedAuthResource(
                    provider=effective_requirement.provider,
                    source=effective_requirement.source,
                    reason="draft_incomplete",
                    message=(
                        f"Profile {selected.profile_id!r} is a draft and must be "
                        "completed before this run."
                    ),
                    candidates=candidate_ids,
                    modes=effective_requirement.modes,
                    required_env_keys=effective_requirement.required_env_keys,
                    mcp_server=effective_requirement.mcp_server,
                    resource_ref=effective_requirement.resource_ref,
                    resource_id=effective_requirement.resource_id,
                )
            )
            continue
        if selected_status not in {"ready", "draft", "archived"}:
            unresolved.append(
                UnresolvedAuthResource(
                    provider=effective_requirement.provider,
                    source=effective_requirement.source,
                    reason="draft_invalid",
                    message=(
                        f"Profile {selected.profile_id!r} has unsupported status "
                        f"{selected_status!r}."
                    ),
                    candidates=candidate_ids,
                    modes=effective_requirement.modes,
                    required_env_keys=effective_requirement.required_env_keys,
                    mcp_server=effective_requirement.mcp_server,
                    resource_ref=effective_requirement.resource_ref,
                    resource_id=effective_requirement.resource_id,
                )
            )
            continue

        oauth_unresolved = _validate_oauth_requirement(
            profile=selected,
            requirement=effective_requirement,
            resolver=secret_resolver,
            candidate_ids=candidate_ids,
        )
        if oauth_unresolved is not None:
            unresolved.append(oauth_unresolved)
            continue

        if effective_requirement.source == "mcp":
            alias = str(selected.mcp_server or "").strip()
            expected_alias = str(effective_requirement.mcp_server or "").strip()
            if not expected_alias and resolved_resource is not None:
                if resolved_resource.resource_kind == "mcp":
                    expected_alias = resolved_resource.resource_key
            if not alias:
                unresolved.append(
                    UnresolvedAuthResource(
                        provider=effective_requirement.provider,
                        source=effective_requirement.source,
                        reason="missing_mcp_binding",
                        message=(
                            f"Profile {selected.profile_id!r} for provider "
                            f"{effective_requirement.provider!r} is not bound to an MCP server."
                        ),
                        candidates=candidate_ids,
                        modes=effective_requirement.modes,
                        required_env_keys=effective_requirement.required_env_keys,
                        mcp_server=effective_requirement.mcp_server,
                        resource_ref=effective_requirement.resource_ref,
                        resource_id=effective_requirement.resource_id,
                    )
                )
                continue
            if expected_alias and alias != expected_alias:
                unresolved.append(
                    UnresolvedAuthResource(
                        provider=effective_requirement.provider,
                        source=effective_requirement.source,
                        reason="mcp_binding_mismatch",
                        message=(
                            f"Profile {selected.profile_id!r} is bound to MCP server "
                            f"{alias!r}, but requirement expects {expected_alias!r}."
                        ),
                        candidates=candidate_ids,
                        modes=effective_requirement.modes,
                        required_env_keys=effective_requirement.required_env_keys,
                        mcp_server=effective_requirement.mcp_server,
                        resource_ref=effective_requirement.resource_ref,
                        resource_id=effective_requirement.resource_id,
                    )
                )
                continue
            if available_aliases and alias not in available_aliases:
                unresolved.append(
                    UnresolvedAuthResource(
                        provider=effective_requirement.provider,
                        source=effective_requirement.source,
                        reason="unknown_mcp_server",
                        message=(
                            f"Profile {selected.profile_id!r} references MCP server "
                            f"{alias!r}, but that alias is not configured."
                        ),
                        candidates=candidate_ids,
                        modes=effective_requirement.modes,
                        required_env_keys=effective_requirement.required_env_keys,
                        mcp_server=effective_requirement.mcp_server or alias,
                        resource_ref=effective_requirement.resource_ref,
                        resource_id=effective_requirement.resource_id,
                    )
                )
                continue
            existing = selected_by_mcp_alias.get(alias)
            if existing is not None and existing.profile_id != selected.profile_id:
                unresolved.append(
                    UnresolvedAuthResource(
                        provider=effective_requirement.provider,
                        source=effective_requirement.source,
                        reason="mcp_alias_conflict",
                        message=(
                            f"Conflicting profile selection for MCP server {alias!r}: "
                            f"{existing.profile_id!r} vs {selected.profile_id!r}."
                        ),
                        candidates=candidate_ids,
                        modes=effective_requirement.modes,
                        required_env_keys=effective_requirement.required_env_keys,
                        mcp_server=alias,
                        resource_ref=effective_requirement.resource_ref,
                        resource_id=effective_requirement.resource_id,
                    )
                )
                continue
            selected_by_mcp_alias[alias] = selected

    if unresolved:
        raise UnresolvedAuthResourcesError(
            "Auth preflight failed: unresolved required auth resources.",
            unresolved=unresolved,
            defaults_user=merged.config.defaults,
            defaults_workspace=merged.workspace_defaults,
            explicit_overrides=explicit_overrides,
            required_resources=all_requirements,
        )

    return RunAuthContext(
        profiles=dict(merged.config.profiles),
        selected_by_selector=selected_by_selector,
        selected_by_provider=selected_by_provider,
        selected_by_mcp_alias=selected_by_mcp_alias,
        source_user_path=merged.user_path,
        source_explicit_path=merged.explicit_path,
        source_workspace_defaults_path=merged.workspace_defaults_path,
        secret_resolver=secret_resolver,
    )
