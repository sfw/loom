"""`loom auth ...` CLI commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from loom.auth.config import (
    AuthProfile,
    default_workspace_auth_defaults_path,
    set_workspace_auth_default,
)
from loom.auth.oauth_profiles import oauth_state_for_profile
from loom.auth.resources import (
    audit_auth_state,
    bind_resource_to_profile,
    create_auth_snapshot,
    default_workspace_auth_resources_path,
    has_active_binding,
    load_workspace_auth_resources,
    migrate_legacy_auth,
    profile_bindings_map,
    repair_auth_state,
    resolve_resource,
    restore_auth_snapshot,
    set_workspace_resource_default,
    sync_missing_drafts,
)
from loom.cli.commands.auth_profile import attach_auth_profile_commands
from loom.cli.context import (
    _effective_config,
    _mcp_manager,
    _merged_auth_config,
)
from loom.tools import create_default_registry


@click.group()
def auth() -> None:
    """Manage auth profile configuration."""


def _auth_profile_state_summary(profile: AuthProfile) -> tuple[str, str]:
    status = str(getattr(profile, "status", "ready") or "ready").strip().lower()
    if status == "archived":
        return "archived", "This account is archived and will not be selected."
    if status == "draft":
        return "draft", "Complete this draft account before Loom can use it."

    mode = str(getattr(profile, "mode", "") or "").strip().lower()
    if mode in {"oauth2_pkce", "oauth2_device"}:
        oauth_state = oauth_state_for_profile(profile)
        return oauth_state.state, oauth_state.reason
    if mode == "api_key":
        return (
            ("configured", "") if str(getattr(profile, "secret_ref", "") or "").strip()
            else ("missing", "secret_ref is missing.")
        )
    if mode == "env_passthrough":
        return (
            ("configured", "") if bool(getattr(profile, "env", {}) or {})
            else ("missing", "No env passthrough keys are configured.")
        )
    if mode == "cli_passthrough":
        return (
            ("configured", "") if str(getattr(profile, "command", "") or "").strip()
            else ("missing", "No auth command is configured.")
        )
    return "unsupported", f"Unsupported auth mode {profile.mode!r}."


def _profile_linked_resources(
    ctx: click.Context,
    *,
    profile_id: str,
) -> tuple[list[str], list[str], list[str]]:
    workspace = ctx.obj.get("workspace")
    resources_path = default_workspace_auth_resources_path(workspace.resolve())
    try:
        resource_store = load_workspace_auth_resources(resources_path)
    except Exception:
        return [], [], []

    linked_resource_refs: list[str] = []
    linked_mcp_aliases: list[str] = []
    by_profile = profile_bindings_map(resource_store)
    resource_id = by_profile.get(profile_id, "")
    if resource_id:
        resource = resource_store.resources.get(resource_id)
        if resource is not None:
            ref = str(getattr(resource, "resource_ref", "") or "").strip()
            if ref:
                linked_resource_refs.append(ref)
            if str(getattr(resource, "resource_kind", "") or "").strip() == "mcp":
                alias = str(getattr(resource, "resource_key", "") or "").strip()
                if alias:
                    linked_mcp_aliases.append(alias)

    merged = _merged_auth_config(ctx)
    default_selectors = sorted({
        selector
        for selector, selected_profile_id in {
            **merged.config.defaults,
            **merged.workspace_defaults,
            **merged.config.resource_defaults,
        }.items()
        if selected_profile_id == profile_id
    })
    return linked_resource_refs, linked_mcp_aliases, default_selectors


@auth.command(name="list")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Include all profile metadata fields.",
)
@click.pass_context
def auth_list(ctx: click.Context, as_json: bool, verbose: bool) -> None:
    """List merged auth profiles and defaults."""
    merged = _merged_auth_config(ctx)
    profiles = merged.config.profiles
    explicit_profile_ids = set(getattr(merged, "explicit_profile_ids", ()))
    if merged.explicit_path is not None:
        profiles = {
            profile_id: profile
            for profile_id, profile in profiles.items()
            if profile_id in explicit_profile_ids
        }

    def _profile_sort_key(profile: AuthProfile) -> tuple[int, int, str]:
        source_rank = 0 if profile.profile_id in explicit_profile_ids else 1
        status = str(profile.status or "ready").strip().lower()
        status_rank = {"ready": 0, "draft": 1, "archived": 2}.get(status, 3)
        return (source_rank, status_rank, profile.profile_id)

    provider_defaults = {
        selector: profile_id
        for selector, profile_id in {
            **merged.config.defaults,
            **merged.workspace_defaults,
        }.items()
        if not selector.startswith("mcp.")
    }
    workspace = ctx.obj.get("workspace")
    resources_path = default_workspace_auth_resources_path(workspace.resolve())
    try:
        resource_store = load_workspace_auth_resources(resources_path)
        workspace_resource_defaults = dict(resource_store.workspace_defaults)
    except Exception:
        workspace_resource_defaults = {}
    user_resource_defaults = dict(merged.config.resource_defaults)
    effective_resource_defaults = dict(user_resource_defaults)
    effective_resource_defaults.update(workspace_resource_defaults)

    payload = {
        "sources": {
            "user_path": str(merged.user_path),
            "explicit_path": (
                str(merged.explicit_path)
                if merged.explicit_path is not None
                else None
            ),
            "workspace_defaults_path": (
                str(merged.workspace_defaults_path)
                if merged.workspace_defaults_path is not None
                else None
            ),
            "workspace_resources_path": str(resources_path),
        },
        "defaults": provider_defaults,
        "resource_defaults": {
            "user": user_resource_defaults,
            "workspace": workspace_resource_defaults,
            "effective": effective_resource_defaults,
        },
        "profiles": [],
    }
    for profile in sorted(profiles.values(), key=_profile_sort_key):
        item = {
            "id": profile.profile_id,
            "provider": profile.provider,
            "mode": profile.mode,
            "account_label": profile.account_label,
            "mcp_server": profile.mcp_server,
        }
        if verbose:
            item.update(
                {
                    "secret_ref": profile.secret_ref,
                    "token_ref": profile.token_ref,
                    "scopes": list(profile.scopes),
                    "env_keys": sorted(profile.env.keys()),
                    "command": profile.command,
                    "auth_check": list(profile.auth_check),
                    "metadata": dict(profile.metadata),
                }
            )
        payload["profiles"].append(item)

    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("Auth profiles:")
    click.echo(f"  user:     {merged.user_path}")
    click.echo(f"  explicit: {merged.explicit_path or '-'}")
    click.echo(f"  workspace defaults: {merged.workspace_defaults_path or '-'}")
    if not profiles:
        click.echo("  (none)")
    else:
        for item in payload["profiles"]:
            click.echo(
                f"  {item['id']:24} provider={item['provider']} mode={item['mode']}"
            )
            if item.get("account_label"):
                click.echo(f"    label: {item['account_label']}")
            if item.get("mcp_server"):
                click.echo(f"    mcp_server: {item['mcp_server']}")
            if verbose:
                env_keys = ", ".join(item.get("env_keys", [])) or "-"
                click.echo(f"    env_keys: {env_keys}")
                if item.get("secret_ref"):
                    click.echo(f"    secret_ref: {item['secret_ref']}")
                if item.get("token_ref"):
                    click.echo(f"    token_ref: {item['token_ref']}")
                if item.get("command"):
                    click.echo(f"    command: {item['command']}")

    if provider_defaults:
        click.echo("Defaults:")
        for selector, profile_id in sorted(provider_defaults.items()):
            click.echo(f"  {selector} -> {profile_id}")
    if effective_resource_defaults:
        click.echo("Resource defaults:")
        for resource_id, profile_id in sorted(effective_resource_defaults.items()):
            click.echo(f"  {resource_id} -> {profile_id}")


@auth.command(name="show")
@click.argument("profile_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def auth_show(ctx: click.Context, profile_id: str, as_json: bool) -> None:
    """Show one auth profile."""
    merged = _merged_auth_config(ctx)
    profile = merged.config.profiles.get(profile_id)
    if profile is None:
        click.echo(f"Auth profile not found: {profile_id}", err=True)
        sys.exit(1)

    payload = {
        "id": profile.profile_id,
        "provider": profile.provider,
        "mode": profile.mode,
        "account_label": profile.account_label,
        "mcp_server": profile.mcp_server,
        "secret_ref": profile.secret_ref,
        "token_ref": profile.token_ref,
        "scopes": list(profile.scopes),
        "env": dict(profile.env),
        "command": profile.command,
        "auth_check": list(profile.auth_check),
        "metadata": dict(profile.metadata),
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    state, state_reason = _auth_profile_state_summary(profile)
    linked_resource_refs, linked_mcp_aliases, default_selectors = _profile_linked_resources(
        ctx,
        profile_id=profile_id,
    )
    click.echo(f"Profile: {payload['id']}")
    click.echo(f"Status: {state}")
    click.echo(f"Provider: {payload['provider']}")
    click.echo(f"Mode: {payload['mode']}")
    click.echo(f"Label: {payload['account_label'] or '-'}")
    click.echo(f"MCP server: {payload['mcp_server'] or '-'}")
    click.echo(
        "Linked MCP aliases: "
        f"{', '.join(linked_mcp_aliases) if linked_mcp_aliases else '-'}"
    )
    click.echo(
        "Linked resources: "
        f"{', '.join(linked_resource_refs) if linked_resource_refs else '-'}"
    )
    click.echo(
        "Default selectors: "
        f"{', '.join(default_selectors) if default_selectors else '-'}"
    )
    click.echo(f"Secret ref: {payload['secret_ref'] or '-'}")
    click.echo(f"Token ref: {payload['token_ref'] or '-'}")
    click.echo(f"Scopes: {', '.join(payload['scopes']) or '-'}")
    click.echo(f"Command: {payload['command'] or '-'}")
    if state_reason:
        click.echo(f"Reason: {state_reason}")
    if payload["env"]:
        click.echo("Env keys:")
        for key in sorted(payload["env"]):
            click.echo(f"  - {key}")


@auth.command(name="explain")
@click.argument("profile_id")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def auth_explain(ctx: click.Context, profile_id: str, as_json: bool) -> None:
    """Explain where an account is used, its status, and the next action."""
    merged = _merged_auth_config(ctx)
    profile = merged.config.profiles.get(profile_id)
    if profile is None:
        click.echo(f"Auth profile not found: {profile_id}", err=True)
        sys.exit(1)

    state, state_reason = _auth_profile_state_summary(profile)
    linked_resource_refs, linked_mcp_aliases, default_selectors = _profile_linked_resources(
        ctx,
        profile_id=profile_id,
    )
    next_actions: list[str] = []
    if state == "draft":
        next_actions.append("Connect or complete this draft account before using it.")
    elif state == "missing":
        next_actions.append(f"Run `loom auth profile login {profile_id}` to connect it.")
    elif state == "expired":
        next_actions.append(f"Run `loom auth profile refresh {profile_id}` to refresh it.")
    elif state == "archived":
        next_actions.append("Restore or replace this archived account.")
    elif state == "invalid":
        next_actions.append("Repair this account's stored credentials.")
    if not linked_resource_refs and not linked_mcp_aliases:
        next_actions.append("Bind this account to a resource or select it as a default.")

    payload = {
        "profile_id": profile.profile_id,
        "provider": profile.provider,
        "mode": profile.mode,
        "account_label": profile.account_label,
        "status": state,
        "reason": state_reason,
        "linked_mcp_aliases": linked_mcp_aliases,
        "linked_resource_refs": linked_resource_refs,
        "default_selectors": default_selectors,
        "next_actions": next_actions,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo(f"Profile: {profile.profile_id}")
    click.echo(f"Status: {state}")
    click.echo(f"Why: {state_reason or 'This account is ready for use.'}")
    click.echo(f"Provider: {profile.provider}")
    click.echo(f"Mode: {profile.mode}")
    click.echo(
        "Linked MCP aliases: "
        f"{', '.join(linked_mcp_aliases) if linked_mcp_aliases else '-'}"
    )
    click.echo(
        "Linked resources: "
        f"{', '.join(linked_resource_refs) if linked_resource_refs else '-'}"
    )
    click.echo(
        "Default selectors: "
        f"{', '.join(default_selectors) if default_selectors else '-'}"
    )
    if next_actions:
        click.echo("Next actions:")
        for item in next_actions:
            click.echo(f"  - {item}")


@auth.command(name="check")
@click.pass_context
def auth_check(ctx: click.Context) -> None:
    """Validate auth profile references and defaults."""
    merged = _merged_auth_config(ctx)
    profiles = dict(merged.config.profiles)
    explicit_profile_ids = set(getattr(merged, "explicit_profile_ids", ()))
    explicit_only = merged.explicit_path is not None
    if explicit_only:
        profiles = {
            profile_id: profile
            for profile_id, profile in profiles.items()
            if profile_id in explicit_profile_ids
        }

    merged_defaults = {
        selector: profile_id
        for selector, profile_id in {
            **merged.config.defaults,
            **merged.workspace_defaults,
        }.items()
        if not selector.startswith("mcp.")
    }
    if explicit_only:
        effective_defaults = {
            selector: profile_id
            for selector, profile_id in merged_defaults.items()
            if profile_id in profiles
        }
    else:
        effective_defaults = merged_defaults

    errors: list[str] = []
    mcp_aliases = set(_effective_config(ctx).mcp.servers.keys())
    for selector, profile_id in sorted(effective_defaults.items()):
        if profile_id not in profiles:
            errors.append(
                f"default selector {selector!r} references unknown profile {profile_id!r}"
            )
            continue
        profile = profiles[profile_id]
        if selector != profile.provider:
            errors.append(
                f"default selector {selector!r} must match profile provider {profile.provider!r}"
            )
    for profile in profiles.values():
        mcp_server = str(profile.mcp_server or "").strip()
        if not mcp_server:
            continue
        if mcp_server not in mcp_aliases:
            errors.append(
                f"profile {profile.profile_id!r} references unknown mcp_server {mcp_server!r}"
            )

    workspace = ctx.obj.get("workspace")
    resources_path = default_workspace_auth_resources_path(workspace.resolve())
    try:
        resource_store = load_workspace_auth_resources(resources_path)
    except Exception as e:
        errors.append(f"failed to read auth resources store: {e}")
        resource_store = None

    if resource_store is not None:
        for resource_id, profile_id in sorted(resource_store.workspace_defaults.items()):
            if explicit_only and profile_id not in profiles:
                continue
            resource = resource_store.resources.get(resource_id)
            if resource is None or str(resource.status).strip().lower() != "active":
                errors.append(
                    "workspace resource default "
                    f"{resource_id!r} references missing/deleted resource"
                )
                continue
            if profile_id not in profiles:
                errors.append(
                    "workspace resource default "
                    f"{resource_id!r} references unknown profile {profile_id!r}"
                )
                continue
            if not has_active_binding(
                resource_store,
                resource_id=resource_id,
                profile_id=profile_id,
            ):
                errors.append(
                    f"workspace resource default {resource_id!r} -> {profile_id!r} "
                    "has no active binding"
                )

        for resource_id, profile_id in sorted(merged.config.resource_defaults.items()):
            if explicit_only and profile_id not in profiles:
                continue
            resource = resource_store.resources.get(resource_id)
            if resource is None or str(resource.status).strip().lower() != "active":
                errors.append(
                    f"user resource default {resource_id!r} references missing/deleted resource"
                )
                continue
            if profile_id not in profiles:
                errors.append(
                    "user resource default "
                    f"{resource_id!r} references unknown profile {profile_id!r}"
                )
                continue
            if not has_active_binding(
                resource_store,
                resource_id=resource_id,
                profile_id=profile_id,
            ):
                errors.append(
                    f"user resource default {resource_id!r} -> {profile_id!r} "
                    "has no active binding"
                )

    if errors:
        click.echo("Auth config validation failed:", err=True)
        for error in errors:
            click.echo(f"  - {error}", err=True)
        sys.exit(1)

    click.echo("Auth config is valid.")
    click.echo(f"Profiles: {len(profiles)}")
    click.echo(f"Defaults: {len(effective_defaults)}")
    defaults_path = (
        merged.workspace_defaults_path
        or default_workspace_auth_defaults_path(ctx.obj.get("workspace"))
    )
    click.echo(f"Workspace defaults file: {defaults_path}")
    click.echo(f"Workspace resources file: {resources_path}")


@auth.command(name="sync")
@click.option(
    "--scope",
    type=click.Choice(["active", "full"], case_sensitive=False),
    default="active",
    show_default=True,
    help="Discovery scope for draft profile generation.",
)
@click.pass_context
def auth_sync(ctx: click.Context, scope: str) -> None:
    """Discover required auth resources and auto-create missing drafts."""
    workspace = ctx.obj.get("workspace")
    manager = _mcp_manager(ctx, workspace=workspace)
    config = _effective_config(ctx, workspace=workspace)
    process_def = None
    process_defs: list[object] = []
    clean_scope = str(scope or "active").strip().lower() or "active"
    if clean_scope == "active":
        try:
            from loom.processes.schema import ProcessLoader

            loader = ProcessLoader(
                workspace=workspace.resolve(),
                extra_search_paths=[Path(p) for p in config.process.search_paths],
                require_rule_scope_metadata=bool(
                    getattr(config.process, "require_rule_scope_metadata", False),
                ),
                require_v2_contract=bool(
                    getattr(config.process, "require_v2_contract", False),
                ),
            )
            seen_names: set[str] = set()
            for item in loader.list_available():
                process_name = str(item.get("name", "")).strip()
                if not process_name or process_name in seen_names:
                    continue
                try:
                    loaded = loader.load(process_name)
                except Exception as e:
                    click.echo(
                        (
                            "Auth sync warning: failed to load process "
                            f"{process_name!r}: {e}"
                        ),
                        err=True,
                    )
                    continue
                loaded_name = str(getattr(loaded, "name", "") or "").strip() or process_name
                if loaded_name in seen_names:
                    continue
                process_defs.append(loaded)
                seen_names.add(loaded_name)
        except Exception as e:
            click.echo(f"Auth sync warning: failed to list processes: {e}", err=True)
    try:
        registry = create_default_registry(config)
    except Exception:
        registry = create_default_registry()

    try:
        result = sync_missing_drafts(
            workspace=workspace,
            explicit_auth_path=ctx.obj.get("explicit_auth_path"),
            process_def=process_def,
            process_defs=process_defs,
            tool_registry=registry,
            mcp_manager=manager,
            scope=clean_scope,
        )
    except Exception as e:
        click.echo(f"Auth sync failed: {e}", err=True)
        sys.exit(1)

    click.echo("Auth sync complete:")
    click.echo(f"  created resources: {result.created_resources}")
    click.echo(f"  updated resources: {result.updated_resources}")
    click.echo(f"  created drafts: {result.created_drafts}")
    click.echo(f"  created bindings: {result.created_bindings}")
    click.echo(f"  updated defaults: {result.updated_defaults}")
    for warning in result.warnings:
        click.echo(f"  warning: {warning}")


@auth.command(name="audit")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def auth_audit(ctx: click.Context, as_json: bool) -> None:
    """Audit auth resources, bindings, and legacy defaults."""
    workspace = ctx.obj.get("workspace")
    try:
        report = audit_auth_state(
            workspace=workspace,
            explicit_auth_path=ctx.obj.get("explicit_auth_path"),
        )
    except Exception as e:
        click.echo(f"Auth audit failed: {e}", err=True)
        sys.exit(1)

    payload = {
        "orphaned_profiles": list(report.orphaned_profiles),
        "orphaned_bindings": list(report.orphaned_bindings),
        "historical_deleted_bindings": list(report.historical_deleted_bindings),
        "deleted_resource_bindings": list(report.deleted_resource_bindings),
        "duplicate_generated_draft_groups": list(
            report.duplicate_generated_draft_groups
        ),
        "stale_generated_profiles": list(report.stale_generated_profiles),
        "legacy_provider_defaults": list(report.legacy_provider_defaults),
        "dangling_workspace_resource_defaults": list(
            report.dangling_workspace_resource_defaults
        ),
        "dangling_user_resource_defaults": list(
            report.dangling_user_resource_defaults
        ),
    }
    actionable_keys = (
        "orphaned_profiles",
        "orphaned_bindings",
        "deleted_resource_bindings",
        "duplicate_generated_draft_groups",
        "stale_generated_profiles",
        "legacy_provider_defaults",
        "dangling_workspace_resource_defaults",
        "dangling_user_resource_defaults",
    )
    finding_count = sum(len(payload[key]) for key in actionable_keys)

    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        click.echo("Auth audit report:")
        for key, items in payload.items():
            if not items:
                click.echo(f"  {key}: 0")
                continue
            click.echo(f"  {key}: {len(items)}")
            for item in items:
                click.echo(f"    - {item}")
        if finding_count == 0:
            click.echo("No issues found.")

    if finding_count > 0:
        sys.exit(1)


@auth.command(name="repair")
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Apply the repair plan (default is dry-run plan only).",
)
@click.option(
    "--prune-deleted-history",
    is_flag=True,
    default=False,
    help="Prune deleted binding history noise when applying/planning.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def auth_repair(
    ctx: click.Context,
    apply_changes: bool,
    prune_deleted_history: bool,
    as_json: bool,
) -> None:
    """Plan/apply deterministic auth dedupe and rebind repairs."""
    workspace = ctx.obj.get("workspace")
    try:
        result = repair_auth_state(
            workspace=workspace,
            explicit_auth_path=ctx.obj.get("explicit_auth_path"),
            apply=apply_changes,
            prune_deleted_history=prune_deleted_history,
        )
    except Exception as e:
        click.echo(f"Auth repair failed: {e}", err=True)
        sys.exit(1)

    payload = {
        "applied": result.applied,
        "changed": result.changed,
        "snapshot_path": str(result.snapshot_path) if result.snapshot_path else None,
        "duplicate_generated_draft_groups": list(result.duplicate_generated_draft_groups),
        "deduped_profiles": list(result.deduped_profiles),
        "rebound_bindings": result.rebound_bindings,
        "updated_workspace_defaults": result.updated_workspace_defaults,
        "updated_user_resource_defaults": result.updated_user_resource_defaults,
        "pruned_deleted_bindings": result.pruned_deleted_bindings,
        "warnings": list(result.warnings),
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("Auth repair " f"{'apply' if apply_changes else 'plan'}:")
    click.echo(
        "  duplicate generated groups: "
        f"{len(result.duplicate_generated_draft_groups)}"
    )
    click.echo(f"  deduped profiles: {len(result.deduped_profiles)}")
    click.echo(f"  rebound bindings: {result.rebound_bindings}")
    click.echo(f"  updated workspace defaults: {result.updated_workspace_defaults}")
    click.echo(f"  updated user defaults: {result.updated_user_resource_defaults}")
    click.echo(f"  pruned deleted bindings: {result.pruned_deleted_bindings}")
    if result.snapshot_path is not None:
        click.echo(f"  snapshot: {result.snapshot_path}")
    for warning in result.warnings:
        click.echo(f"  warning: {warning}")
    if not result.changed:
        click.echo("No repair changes required.")
    elif not apply_changes:
        click.echo("Dry-run only. Re-run with --apply to write changes.")


@auth.command(name="migrate")
@click.option(
    "--rollback",
    type=click.Path(path_type=Path),
    default=None,
    help="Restore auth files from a snapshot directory.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit JSON.")
@click.pass_context
def auth_migrate(
    ctx: click.Context,
    rollback: Path | None,
    as_json: bool,
) -> None:
    """Migrate legacy provider defaults to resource bindings/defaults."""
    workspace = ctx.obj.get("workspace")
    explicit_auth_path = ctx.obj.get("explicit_auth_path")

    if rollback is not None:
        rollback_path = rollback.expanduser().resolve()
        try:
            restore_auth_snapshot(
                workspace=workspace,
                explicit_auth_path=explicit_auth_path,
                snapshot_path=rollback_path,
            )
        except Exception as e:
            click.echo(f"Auth rollback failed: {e}", err=True)
            sys.exit(1)
        payload = {
            "rolled_back": True,
            "snapshot_path": str(rollback_path),
        }
        if as_json:
            click.echo(json.dumps(payload, indent=2, sort_keys=True))
        else:
            click.echo(f"Restored auth files from snapshot: {rollback_path}")
        return

    pre_snapshot = create_auth_snapshot(
        workspace=workspace,
        explicit_auth_path=explicit_auth_path,
        label="migrate-preflight",
    )
    try:
        result = migrate_legacy_auth(
            workspace=workspace,
            explicit_auth_path=explicit_auth_path,
        )
    except Exception as e:
        try:
            restore_auth_snapshot(
                workspace=workspace,
                explicit_auth_path=explicit_auth_path,
                snapshot_path=pre_snapshot,
            )
            click.echo(
                (
                    "Auth migrate failed and changes were rolled back from "
                    f"{pre_snapshot}: {e}"
                ),
                err=True,
            )
        except Exception as rollback_error:
            click.echo(
                (
                    "Auth migrate failed and rollback failed. "
                    f"preflight snapshot={pre_snapshot}; "
                    f"migration_error={e}; rollback_error={rollback_error}"
                ),
                err=True,
            )
        sys.exit(1)

    payload = {
        "snapshot_path": str(result.snapshot_path),
        "created_resources": result.created_resources,
        "created_bindings": result.created_bindings,
        "created_workspace_defaults": result.created_workspace_defaults,
        "created_user_resource_defaults": result.created_user_resource_defaults,
        "warnings": list(result.warnings),
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    click.echo("Auth migration complete:")
    click.echo(f"  snapshot: {result.snapshot_path}")
    click.echo(f"  created resources: {result.created_resources}")
    click.echo(f"  created bindings: {result.created_bindings}")
    click.echo(f"  created workspace defaults: {result.created_workspace_defaults}")
    click.echo(
        "  created user resource defaults: "
        f"{result.created_user_resource_defaults}"
    )
    if result.warnings:
        click.echo("Warnings:")
        for warning in result.warnings:
            click.echo(f"  - {warning}")


@auth.command(name="select")
@click.argument("selector")
@click.argument("profile_id", required=False)
@click.option(
    "--unset",
    is_flag=True,
    default=False,
    help="Remove selector mapping from workspace defaults.",
)
@click.pass_context
def auth_select(
    ctx: click.Context,
    selector: str,
    profile_id: str | None,
    unset: bool,
) -> None:
    """Set/clear workspace auth default selector mapping."""
    merged = _merged_auth_config(ctx)
    clean_selector = str(selector or "").strip()
    if not clean_selector:
        click.echo("Selector cannot be empty.", err=True)
        sys.exit(1)
    if clean_selector.startswith("mcp."):
        click.echo(
            "MCP selectors are no longer supported in `loom auth select`. "
            "Use separate MCP aliases in `loom mcp` for multi-account MCP auth.",
            err=True,
        )
        sys.exit(1)

    workspace = ctx.obj.get("workspace")
    defaults_path = default_workspace_auth_defaults_path(workspace)
    resources_path = default_workspace_auth_resources_path(workspace.resolve())
    try:
        resource_store = load_workspace_auth_resources(resources_path)
    except Exception as e:
        click.echo(f"Auth select failed: {e}", err=True)
        sys.exit(1)

    selected_resource = None
    if ":" in clean_selector:
        selected_resource = resolve_resource(
            resource_store,
            resource_ref=clean_selector,
        )
    elif clean_selector in resource_store.resources:
        selected_resource = resolve_resource(
            resource_store,
            resource_id=clean_selector,
        )

    if unset:
        if profile_id:
            click.echo(
                "Do not pass profile_id when using --unset.",
                err=True,
            )
            sys.exit(1)
        if selected_resource is not None:
            try:
                set_workspace_resource_default(
                    resources_path,
                    resource_id=selected_resource.resource_id,
                    profile_id=None,
                )
            except Exception as e:
                click.echo(f"Auth select failed: {e}", err=True)
                sys.exit(1)
            click.echo(
                "Removed resource default mapping: "
                f"{selected_resource.resource_ref}"
            )
            click.echo(f"Workspace resources file: {resources_path}")
            return
        try:
            updated = set_workspace_auth_default(
                defaults_path,
                selector=clean_selector,
                profile_id=None,
            )
        except Exception as e:
            click.echo(f"Auth select failed: {e}", err=True)
            sys.exit(1)
        click.echo(f"Removed default mapping: {clean_selector}")
        click.echo(f"Workspace defaults file: {defaults_path}")
        click.echo(f"Remaining defaults: {len(updated)}")
        return

    clean_profile_id = str(profile_id or "").strip()
    if not clean_profile_id:
        click.echo("Missing profile_id. Usage: loom auth select <selector> <profile_id>", err=True)
        sys.exit(1)
    profile = merged.config.profiles.get(clean_profile_id)
    if profile is None:
        click.echo(f"Unknown auth profile: {clean_profile_id}", err=True)
        sys.exit(1)

    if selected_resource is not None:
        if profile.provider != selected_resource.provider:
            click.echo(
                (
                    f"Profile {clean_profile_id!r} provider {profile.provider!r} "
                    f"does not match resource provider {selected_resource.provider!r}."
                ),
                err=True,
            )
            sys.exit(1)
        try:
            if not has_active_binding(
                resource_store,
                resource_id=selected_resource.resource_id,
                profile_id=clean_profile_id,
            ):
                bind_resource_to_profile(
                    resources_path,
                    resource_id=selected_resource.resource_id,
                    profile_id=clean_profile_id,
                    generated_from=f"cli:auth-select:{clean_selector}",
                    priority=0,
                )
            set_workspace_resource_default(
                resources_path,
                resource_id=selected_resource.resource_id,
                profile_id=clean_profile_id,
            )
        except Exception as e:
            click.echo(f"Auth select failed: {e}", err=True)
            sys.exit(1)
        click.echo(
            "Set workspace resource default: "
            f"{selected_resource.resource_ref} -> {clean_profile_id}"
        )
        click.echo(f"Workspace resources file: {resources_path}")
        return

    if clean_selector != profile.provider:
        click.echo(
            (
                f"Selector {clean_selector!r} must match profile provider "
                f"{profile.provider!r}, or be a resource_id/resource_ref."
            ),
            err=True,
        )
        sys.exit(1)

    try:
        set_workspace_auth_default(
            defaults_path,
            selector=clean_selector,
            profile_id=clean_profile_id,
        )
    except Exception as e:
        click.echo(f"Auth select failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Set workspace default: {clean_selector} -> {clean_profile_id}")
    click.echo(f"Workspace defaults file: {defaults_path}")


attach_auth_profile_commands(
    auth,
    auth_list_callback=auth_list,
    auth_show_callback=auth_show,
)


def register_auth_commands(cli_group: click.Group) -> None:
    cli_group.add_command(auth)
