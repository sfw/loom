"""Tests for auth profile config loading and runtime selection."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
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
from loom.auth.resources import (
    AuthBinding,
    AuthResource,
    AuthResourcesStore,
    cleanup_deleted_resource,
    default_workspace_auth_resources_path,
    load_workspace_auth_resources,
    resource_delete_impact,
    restore_deleted_resource,
    sync_missing_drafts,
    write_workspace_auth_resources,
)
from loom.auth.runtime import (
    AuthResolutionError,
    UnresolvedAuthResourcesError,
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
token_ref = "keychain://loom/notion/notion_marketing/tokens"
"""
    )

    with pytest.warns(FutureWarning, match="mcp_alias_profiles"):
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
token_ref = "keychain://loom/notion/notion_marketing/tokens"
"""
    )
    explicit_auth = tmp_path / "explicit-auth.toml"
    explicit_auth.write_text(
        """
[auth.profiles.notion_ops]
provider = "notion"
mode = "oauth2_pkce"
token_ref = "keychain://loom/notion/notion_ops/tokens"
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
token_ref = "keychain://loom/notion/notion_marketing/tokens"
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


def test_build_run_auth_context_resolves_mcp_binding_env(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.notion_dev]
provider = "notion"
mode = "env_passthrough"
mcp_server = "demo"

[auth.profiles.notion_dev.env]
MCP_TOKEN = "run-token"
"""
    )

    context = build_run_auth_context(
        workspace=tmp_path,
        metadata={"auth_config_path": str(auth_path)},
        required_resources=[
            {"provider": "notion", "source": "mcp"},
        ],
        available_mcp_aliases={"demo"},
    )
    assert context.profile_for_mcp_alias("demo") is not None
    assert context.env_for_mcp_alias("demo") == {"MCP_TOKEN": "run-token"}


def test_build_run_auth_context_reports_structured_unresolved_requirements(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.notion_dev]
provider = "notion"
mode = "env_passthrough"
mcp_server = "demo"

[auth.profiles.notion_dev.env]
MCP_TOKEN = "dev-token"

[auth.profiles.notion_prod]
provider = "notion"
mode = "env_passthrough"
mcp_server = "demo"

[auth.profiles.notion_prod.env]
MCP_TOKEN = "prod-token"
"""
    )

    with pytest.raises(UnresolvedAuthResourcesError) as exc:
        build_run_auth_context(
            workspace=tmp_path,
            metadata={"auth_config_path": str(auth_path)},
            required_resources=[{"provider": "notion", "source": "mcp"}],
            available_mcp_aliases={"demo"},
        )
    unresolved = exc.value.unresolved
    assert len(unresolved) == 1
    assert unresolved[0].reason == "ambiguous"
    assert set(unresolved[0].candidates) == {"notion_dev", "notion_prod"}


def test_build_run_auth_context_reports_auth_missing_for_oauth_tokens(
    tmp_path: Path,
):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.notion_oauth]
provider = "notion"
mode = "oauth2_pkce"
token_ref = "env://NOTION_MISSING_TOKEN"
"""
    )

    with pytest.raises(UnresolvedAuthResourcesError) as exc:
        build_run_auth_context(
            workspace=tmp_path,
            metadata={"auth_config_path": str(auth_path)},
            required_resources=[{"provider": "notion", "source": "api"}],
        )
    unresolved = exc.value.unresolved
    assert len(unresolved) == 1
    assert unresolved[0].reason == "auth_missing"
    assert unresolved[0].provider == "notion"


def test_build_run_auth_context_reports_auth_invalid_for_oauth_tokens(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv(
        "NOTION_BROKEN_TOKEN",
        '{"refresh_token": "only-refresh-token"}',
    )
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.notion_oauth]
provider = "notion"
mode = "oauth2_pkce"
token_ref = "env://NOTION_BROKEN_TOKEN"
"""
    )

    with pytest.raises(UnresolvedAuthResourcesError) as exc:
        build_run_auth_context(
            workspace=tmp_path,
            metadata={"auth_config_path": str(auth_path)},
            required_resources=[{"provider": "notion", "source": "api"}],
        )
    unresolved = exc.value.unresolved
    assert len(unresolved) == 1
    assert unresolved[0].reason == "auth_invalid"
    assert unresolved[0].provider == "notion"


def test_build_run_auth_context_reports_auth_expired_for_oauth_tokens(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv(
        "NOTION_EXPIRED_TOKEN",
        '{"access_token": "abc", "expires_at": 1}',
    )
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.notion_oauth]
provider = "notion"
mode = "oauth2_pkce"
token_ref = "env://NOTION_EXPIRED_TOKEN"
"""
    )

    with pytest.raises(UnresolvedAuthResourcesError) as exc:
        build_run_auth_context(
            workspace=tmp_path,
            metadata={"auth_config_path": str(auth_path)},
            required_resources=[{"provider": "notion", "source": "api"}],
        )
    unresolved = exc.value.unresolved
    assert len(unresolved) == 1
    assert unresolved[0].reason == "auth_expired"
    assert unresolved[0].provider == "notion"


def test_sync_missing_drafts_creates_resource_binding_and_default(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    auth_path = tmp_path / "auth.toml"

    class _FakeMCPManager:
        def list_views(self):
            return [types.SimpleNamespace(alias="notion")]

    result = sync_missing_drafts(
        workspace=workspace,
        explicit_auth_path=auth_path,
        mcp_manager=_FakeMCPManager(),
    )
    assert result.created_drafts >= 1
    assert result.created_bindings >= 1

    auth_cfg = load_auth_file(auth_path)
    assert auth_cfg.profiles
    profile = next(iter(auth_cfg.profiles.values()))
    assert profile.status == "draft"

    store = load_workspace_auth_resources(
        default_workspace_auth_resources_path(workspace),
    )
    resources = [resource for resource in store.resources.values() if resource.status == "active"]
    assert resources
    resource_ids = {resource.resource_id for resource in resources}
    assert any(binding.resource_id in resource_ids for binding in store.bindings.values())
    assert any(
        mapped_profile_id == profile.profile_id
        for mapped_profile_id in store.workspace_defaults.values()
    )


def test_build_run_auth_context_uses_resource_default_selection(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.notion_dev]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_dev.env]
NOTION_TOKEN = "dev-token"

[auth.profiles.notion_prod]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_prod.env]
NOTION_TOKEN = "prod-token"
"""
    )

    resource = AuthResource(
        resource_id="res-notion",
        resource_kind="api_integration",
        resource_key="notion",
        display_name="API: notion",
        provider="notion",
        source="api",
        status="active",
    )
    binding = AuthBinding(
        binding_id="bind-notion-dev",
        resource_id="res-notion",
        profile_id="notion_dev",
        status="active",
    )
    write_workspace_auth_resources(
        default_workspace_auth_resources_path(tmp_path),
        AuthResourcesStore(
            resources={"res-notion": resource},
            bindings={"bind-notion-dev": binding},
            workspace_defaults={"res-notion": "notion_dev"},
        ),
    )

    context = build_run_auth_context(
        workspace=tmp_path,
        metadata={"auth_config_path": str(auth_path)},
        required_resources=[{"resource_id": "res-notion", "source": "api"}],
    )
    selected = context.profile_for_selector("res-notion")
    assert selected is not None
    assert selected.profile_id == "notion_dev"


def test_build_run_auth_context_prefers_workspace_resource_default_over_user_default(
    tmp_path: Path,
):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.resource_defaults]
res-notion = "notion_prod"

[auth.profiles.notion_dev]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_dev.env]
NOTION_TOKEN = "dev-token"

[auth.profiles.notion_prod]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_prod.env]
NOTION_TOKEN = "prod-token"
"""
    )

    resource = AuthResource(
        resource_id="res-notion",
        resource_kind="api_integration",
        resource_key="notion",
        display_name="API: notion",
        provider="notion",
        source="api",
        status="active",
    )
    write_workspace_auth_resources(
        default_workspace_auth_resources_path(tmp_path),
        AuthResourcesStore(
            resources={"res-notion": resource},
            bindings={
                "bind-notion-dev": AuthBinding(
                    binding_id="bind-notion-dev",
                    resource_id="res-notion",
                    profile_id="notion_dev",
                    status="active",
                ),
                "bind-notion-prod": AuthBinding(
                    binding_id="bind-notion-prod",
                    resource_id="res-notion",
                    profile_id="notion_prod",
                    status="active",
                ),
            },
            workspace_defaults={"res-notion": "notion_dev"},
        ),
    )

    context = build_run_auth_context(
        workspace=tmp_path,
        metadata={"auth_config_path": str(auth_path)},
        required_resources=[{"resource_id": "res-notion", "source": "api"}],
    )
    selected = context.profile_for_selector("res-notion")
    assert selected is not None
    assert selected.profile_id == "notion_dev"


def test_build_run_auth_context_provider_fallback_requires_resource_binding(tmp_path: Path):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.defaults]
notion = "notion_prod"

[auth.profiles.notion_dev]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_dev.env]
NOTION_TOKEN = "dev-token"

[auth.profiles.notion_prod]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_prod.env]
NOTION_TOKEN = "prod-token"
"""
    )

    resource = AuthResource(
        resource_id="res-notion",
        resource_kind="api_integration",
        resource_key="notion",
        display_name="API: notion",
        provider="notion",
        source="api",
        status="active",
    )
    binding = AuthBinding(
        binding_id="bind-notion-dev",
        resource_id="res-notion",
        profile_id="notion_dev",
        status="active",
    )
    write_workspace_auth_resources(
        default_workspace_auth_resources_path(tmp_path),
        AuthResourcesStore(
            resources={"res-notion": resource},
            bindings={"bind-notion-dev": binding},
        ),
    )

    with pytest.raises(UnresolvedAuthResourcesError) as exc:
        build_run_auth_context(
            workspace=tmp_path,
            metadata={"auth_config_path": str(auth_path)},
            required_resources=[{"resource_id": "res-notion", "source": "api"}],
        )
    unresolved = exc.value.unresolved
    assert unresolved
    assert unresolved[0].reason == "needs_rebind"


def test_build_run_auth_context_resource_defaults_do_not_imply_provider_selection(
    tmp_path: Path,
):
    auth_path = tmp_path / "auth.toml"
    auth_path.write_text(
        """
[auth.resource_defaults]
res-notion-a = "notion_dev"
res-notion-b = "notion_prod"

[auth.profiles.notion_dev]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_dev.env]
NOTION_TOKEN = "dev-token"

[auth.profiles.notion_prod]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_prod.env]
NOTION_TOKEN = "prod-token"
"""
    )

    write_workspace_auth_resources(
        default_workspace_auth_resources_path(tmp_path),
        AuthResourcesStore(
            resources={
                "res-notion-a": AuthResource(
                    resource_id="res-notion-a",
                    resource_kind="api_integration",
                    resource_key="notion_a",
                    display_name="API: notion_a",
                    provider="notion",
                    source="api",
                    status="active",
                ),
                "res-notion-b": AuthResource(
                    resource_id="res-notion-b",
                    resource_kind="api_integration",
                    resource_key="notion_b",
                    display_name="API: notion_b",
                    provider="notion",
                    source="api",
                    status="active",
                ),
            },
            bindings={
                "bind-notion-a": AuthBinding(
                    binding_id="bind-notion-a",
                    resource_id="res-notion-a",
                    profile_id="notion_dev",
                    status="active",
                ),
                "bind-notion-b": AuthBinding(
                    binding_id="bind-notion-b",
                    resource_id="res-notion-b",
                    profile_id="notion_prod",
                    status="active",
                ),
            },
            workspace_defaults={
                "res-notion-a": "notion_dev",
                "res-notion-b": "notion_prod",
            },
        ),
    )

    with pytest.raises(UnresolvedAuthResourcesError) as exc:
        build_run_auth_context(
            workspace=tmp_path,
            metadata={"auth_config_path": str(auth_path)},
            required_resources=[{"provider": "notion", "source": "api"}],
        )
    unresolved = exc.value.unresolved
    assert unresolved
    assert unresolved[0].reason == "ambiguous"


def test_auth_file_mutations_use_file_lock(tmp_path: Path, monkeypatch):
    import loom.auth.config as auth_cfg_mod

    calls: list[Path] = []

    @contextmanager
    def _fake_lock(path: Path):
        calls.append(path)
        yield

    monkeypatch.setattr(auth_cfg_mod, "_file_lock", _fake_lock)

    auth_path = tmp_path / "auth.toml"
    profile = AuthProfile(
        profile_id="notion_marketing",
        provider="notion",
        mode="oauth2_pkce",
        token_ref="keychain://loom/notion/notion_marketing/tokens",
    )
    upsert_auth_profile(auth_path, profile, must_exist=False)
    set_auth_default(
        auth_path,
        selector="notion",
        profile_id="notion_marketing",
    )
    remove_auth_profile(auth_path, "notion_marketing")

    assert calls
    assert all(path.name == ".auth.toml.lock" for path in calls)


def test_workspace_auth_defaults_mutation_uses_file_lock(tmp_path: Path, monkeypatch):
    import loom.auth.config as auth_cfg_mod

    calls: list[Path] = []

    @contextmanager
    def _fake_lock(path: Path):
        calls.append(path)
        yield

    monkeypatch.setattr(auth_cfg_mod, "_file_lock", _fake_lock)

    defaults_path = tmp_path / ".loom" / "auth.defaults.toml"
    set_workspace_auth_default(
        defaults_path,
        selector="notion",
        profile_id="notion_marketing",
    )
    set_workspace_auth_default(
        defaults_path,
        selector="notion",
        profile_id=None,
    )

    assert calls
    assert all(path.name == ".auth.defaults.toml.lock" for path in calls)


def test_resource_delete_impact_reports_referencing_processes(tmp_path: Path):
    workspace = tmp_path
    process_dir = workspace / ".loom" / "processes"
    process_dir.mkdir(parents=True)
    process_dir.joinpath("demo.yaml").write_text(
        """
name: demo-auth
version: "1.0"
auth:
  required:
    - provider: notion
      source: mcp
      resource_ref: mcp:notion
      mcp_server: notion
"""
    )
    write_workspace_auth_resources(
        default_workspace_auth_resources_path(workspace),
        AuthResourcesStore(
            resources={
                "res-mcp-notion": AuthResource(
                    resource_id="res-mcp-notion",
                    resource_kind="mcp",
                    resource_key="notion",
                    display_name="MCP: notion",
                    provider="notion",
                    source="mcp",
                    status="active",
                ),
            },
        ),
    )

    impact = resource_delete_impact(
        workspace=workspace,
        resource_kind="mcp",
        resource_key="notion",
    )
    assert "demo-auth" in impact.referencing_processes


def test_cleanup_deleted_resource_and_restore_flow(tmp_path: Path):
    workspace = tmp_path
    auth_path = workspace / "auth.toml"
    auth_path.write_text(
        """
[auth.profiles.notion_draft]
provider = "notion"
mode = "api_key"
status = "draft"

[auth.profiles.notion_ready]
provider = "notion"
mode = "env_passthrough"

[auth.profiles.notion_ready.env]
NOTION_TOKEN = "ready-token"
"""
    )
    write_workspace_auth_resources(
        default_workspace_auth_resources_path(workspace),
        AuthResourcesStore(
            resources={
                "res-mcp-notion": AuthResource(
                    resource_id="res-mcp-notion",
                    resource_kind="mcp",
                    resource_key="notion",
                    display_name="MCP: notion",
                    provider="notion",
                    source="mcp",
                    status="active",
                ),
            },
            bindings={
                "bind-draft": AuthBinding(
                    binding_id="bind-draft",
                    resource_id="res-mcp-notion",
                    profile_id="notion_draft",
                    status="active",
                ),
                "bind-ready": AuthBinding(
                    binding_id="bind-ready",
                    resource_id="res-mcp-notion",
                    profile_id="notion_ready",
                    status="active",
                ),
            },
            workspace_defaults={"res-mcp-notion": "notion_ready"},
        ),
    )

    assert cleanup_deleted_resource(
        workspace=workspace,
        explicit_auth_path=auth_path,
        resource_kind="mcp",
        resource_key="notion",
    )

    store_after_delete = load_workspace_auth_resources(
        default_workspace_auth_resources_path(workspace)
    )
    assert store_after_delete.resources["res-mcp-notion"].status == "deleted"
    assert "res-mcp-notion" not in store_after_delete.workspace_defaults
    assert (
        store_after_delete.bindings["bind-draft"].status == "deleted"
    )
    assert (
        store_after_delete.bindings["bind-ready"].status == "deleted"
    )

    auth_after_delete = load_auth_file(auth_path)
    assert "notion_draft" not in auth_after_delete.profiles
    assert auth_after_delete.profiles["notion_ready"].status == "archived"

    assert restore_deleted_resource(
        workspace=workspace,
        resource_kind="mcp",
        resource_key="notion",
    )
    store_after_restore = load_workspace_auth_resources(
        default_workspace_auth_resources_path(workspace)
    )
    assert store_after_restore.resources["res-mcp-notion"].status == "active"
