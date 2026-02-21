"""Tests for auth profile config loading and runtime selection."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from loom.auth.config import (
    AuthProfile,
    load_auth_file,
    load_merged_auth_config,
    load_workspace_auth_defaults,
    remove_auth_profile,
    resolve_auth_write_path,
    set_auth_default,
    set_workspace_auth_default,
    upsert_auth_profile,
)
from loom.auth.runtime import (
    AuthResolutionError,
    build_run_auth_context,
    parse_auth_profile_overrides,
)
from loom.auth.secrets import SecretResolutionError, SecretResolver


def test_load_auth_file_parses_profiles_and_defaults(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.defaults]
notion = "notion_marketing"

[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
account_label = "Marketing"
scopes = ["read:content", "write:content"]
token_ref = "keychain://loom/notion/notion_marketing/tokens"

[auth.profiles.ga_acme_prod]
provider = "google_analytics"
mode = "env_passthrough"

[auth.profiles.ga_acme_prod.env]
GA_TOKEN = "${REAL_GA_TOKEN}"
"""
    )

    cfg = load_auth_file(auth_path)
    assert cfg.defaults == {"notion": "notion_marketing"}
    assert set(cfg.profiles.keys()) == {"notion_marketing", "ga_acme_prod"}
    assert cfg.profiles["notion_marketing"].provider == "notion"
    assert cfg.profiles["ga_acme_prod"].env["GA_TOKEN"] == "${REAL_GA_TOKEN}"


def test_load_auth_file_ignores_legacy_mcp_alias_profiles_section(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.defaults]
notion = "notion_marketing"

[auth.mcp_alias_profiles]
notion = "notion_marketing"

[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
"""
    )

    cfg = load_auth_file(auth_path)
    assert cfg.defaults == {"notion": "notion_marketing"}
    assert cfg.profiles["notion_marketing"].provider == "notion"


def test_load_merged_auth_config_applies_workspace_defaults(tmp_path: Path):
    user_auth = tmp_path / "user-auth.toml"
    user_auth.write_text(
        """
[auth.defaults]
notion = "notion_marketing"

[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
"""
    )
    explicit_auth = tmp_path / "explicit-auth.toml"
    explicit_auth.write_text(
        """
[auth.profiles.notion_ops]
provider = "notion"
mode = "oauth2_pkce"
"""
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ws_defaults = workspace / ".loom" / "auth.defaults.toml"
    ws_defaults.parent.mkdir(parents=True)
    ws_defaults.write_text(
        """
[auth.defaults]
notion = "notion_ops"
"""
    )

    merged = load_merged_auth_config(
        workspace=workspace,
        explicit_path=explicit_auth,
        user_path=user_auth,
    )
    assert merged.config.defaults["notion"] == "notion_marketing"
    assert merged.workspace_defaults["notion"] == "notion_ops"
    assert "notion_ops" in merged.config.profiles


def test_parse_auth_profile_overrides_validates_shape():
    parsed = parse_auth_profile_overrides(
        ("notion=notion_marketing", "google_analytics=ga_acme_prod")
    )
    assert parsed == {
        "notion": "notion_marketing",
        "google_analytics": "ga_acme_prod",
    }

    with pytest.raises(AuthResolutionError):
        parse_auth_profile_overrides(("badpair",))


def test_build_run_auth_context_resolves_defaults_and_overrides(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("REAL_GA_TOKEN", "tok-123")
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.defaults]
google_analytics = "ga_default"

[auth.profiles.ga_default]
provider = "google_analytics"
mode = "env_passthrough"

[auth.profiles.ga_default.env]
GA_TOKEN = "${REAL_GA_TOKEN}"

[auth.profiles.ga_alt]
provider = "google_analytics"
mode = "env_passthrough"

[auth.profiles.ga_alt.env]
GA_TOKEN = "literal-token"
"""
    )

    ctx = build_run_auth_context(
        workspace=tmp_path,
        metadata={
            "auth_config_path": str(auth_path),
            "auth_profile_overrides": {"google_analytics": "ga_alt"},
        },
    )
    selected = ctx.profile_for_provider("google_analytics")
    assert selected is not None
    assert selected.profile_id == "ga_alt"
    assert ctx.env_for_mcp_alias("ga_local") == {}


def test_build_run_auth_context_rejects_unknown_explicit_override(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.defaults]
notion = "missing_profile"
"""
    )
    # Unknown defaults are ignored at runtime unless explicitly requested.
    context = build_run_auth_context(
        workspace=tmp_path,
        metadata={"auth_config_path": str(auth_path)},
    )
    assert context.profile_for_provider("notion") is None

    with pytest.raises(AuthResolutionError):
        build_run_auth_context(
            workspace=tmp_path,
            metadata={
                "auth_config_path": str(auth_path),
                "auth_profile_overrides": {"notion": "missing_profile"},
            },
        )


def test_build_run_auth_context_rejects_explicit_mcp_selector(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.notion_marketing]
provider = "notion"
mode = "oauth2_pkce"
"""
    )

    with pytest.raises(AuthResolutionError, match="no longer supported"):
        build_run_auth_context(
            workspace=tmp_path,
            metadata={
                "auth_config_path": str(auth_path),
                "auth_profile_overrides": {"mcp.notion": "notion_marketing"},
            },
        )


def test_workspace_auth_default_write_and_unset(tmp_path: Path):
    defaults_path = tmp_path / ".loom" / "auth.defaults.toml"
    updated = set_workspace_auth_default(
        defaults_path,
        selector="notion",
        profile_id="notion_marketing",
    )
    assert updated["notion"] == "notion_marketing"
    loaded = load_workspace_auth_defaults(defaults_path)
    assert loaded == {"notion": "notion_marketing"}

    updated = set_workspace_auth_default(
        defaults_path,
        selector="notion",
        profile_id=None,
    )
    assert updated == {}
    loaded = load_workspace_auth_defaults(defaults_path)
    assert loaded == {}


def test_build_run_auth_context_auto_selects_single_provider(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.ga_default]
provider = "google_analytics"
mode = "env_passthrough"

[auth.profiles.ga_default.env]
GA_TOKEN = "literal"
"""
    )
    context = build_run_auth_context(
        workspace=tmp_path,
        metadata={"auth_config_path": str(auth_path)},
    )
    selected = context.profile_for_provider("google_analytics")
    assert selected is not None
    assert selected.profile_id == "ga_default"


def test_secret_resolver_env_refs(monkeypatch):
    monkeypatch.setenv("REAL_TOKEN", "abc123")
    resolver = SecretResolver()
    assert resolver.resolve("${REAL_TOKEN}") == "abc123"
    assert resolver.resolve("env://REAL_TOKEN") == "abc123"
    assert resolver.resolve_maybe("literal-value") == "literal-value"

    monkeypatch.delenv("REAL_TOKEN", raising=False)
    with pytest.raises(SecretResolutionError):
        resolver.resolve("env://REAL_TOKEN")


def test_secret_resolver_keychain_refs(monkeypatch):
    module = types.SimpleNamespace()
    module.get_password = lambda service, account: (
        "token-xyz"
        if service == "loom" and account == "notion/notion_marketing/tokens"
        else None
    )
    monkeypatch.setitem(sys.modules, "keyring", module)
    resolver = SecretResolver()
    assert (
        resolver.resolve("keychain://loom/notion/notion_marketing/tokens")
        == "token-xyz"
    )
    with pytest.raises(SecretResolutionError):
        resolver.resolve("keychain://loom/notion/missing")


def test_auth_profile_upsert_remove_and_defaults_cleanup(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    profile = AuthProfile(
        profile_id="notion_marketing",
        provider="notion",
        mode="oauth2_pkce",
        token_ref="keychain://loom/notion/notion_marketing/tokens",
    )
    upserted = upsert_auth_profile(auth_path, profile, must_exist=False)
    assert "notion_marketing" in upserted.profiles
    loaded = load_auth_file(auth_path)
    assert loaded.profiles["notion_marketing"].provider == "notion"

    set_auth_default(
        auth_path,
        selector="notion",
        profile_id="notion_marketing",
    )
    loaded = load_auth_file(auth_path)
    assert loaded.defaults["notion"] == "notion_marketing"

    remove_auth_profile(auth_path, "notion_marketing")
    loaded = load_auth_file(auth_path)
    assert loaded.profiles == {}
    assert loaded.defaults == {}


def test_resolve_auth_write_path_prefers_explicit(tmp_path: Path):
    explicit = tmp_path / "explicit-auth.toml"
    resolved = resolve_auth_write_path(explicit_path=explicit)
    assert resolved == explicit.resolve()
