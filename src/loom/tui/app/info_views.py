"""Read-only info and catalog rendering helpers."""

from __future__ import annotations

from typing import Any

from loom.models.base import ModelProvider


def resolve_active_model_alias(self) -> tuple[str | None, list[str]]:
    """Resolve active model alias from config with ambiguity awareness."""
    configured = self._configured_models()
    if self._model is None or not configured:
        return None, []

    runtime_name = str(getattr(self._model, "name", "") or "").strip()
    if runtime_name and runtime_name in configured:
        return runtime_name, []

    runtime_provider = self._runtime_model_provider(self._model)
    runtime_model_id = self._runtime_model_id(self._model).lower()
    runtime_endpoint = self._runtime_model_endpoint(self._model)
    candidates: list[tuple[str, int]] = []

    for alias, cfg in configured.items():
        cfg_provider = self._normalize_provider_name(getattr(cfg, "provider", ""))
        if runtime_provider and cfg_provider and cfg_provider != runtime_provider:
            continue

        score = 0
        cfg_model_id = str(getattr(cfg, "model", "") or "").strip().lower()
        if runtime_model_id and cfg_model_id and runtime_model_id == cfg_model_id:
            score += 2

        cfg_endpoint = self._endpoint_for_config(
            cfg_provider,
            getattr(cfg, "base_url", ""),
        )
        if (
            runtime_endpoint not in {"-", "(invalid-configured-url)"}
            and cfg_endpoint not in {"-", "(invalid-configured-url)"}
            and runtime_endpoint == cfg_endpoint
        ):
            score += 1

        if score > 0:
            candidates.append((alias, score))

    if not candidates:
        return None, []

    candidates.sort(key=lambda item: (-item[1], item[0].casefold()))
    top_score = candidates[0][1]
    top_aliases = [alias for alias, score in candidates if score == top_score]
    if len(top_aliases) == 1:
        return top_aliases[0], []
    return None, sorted(top_aliases, key=str.casefold)


def render_model_block(
    self,
    *,
    alias: str,
    active: bool,
    provider: str,
    endpoint: str,
    model_id: str,
    roles: object | None,
    tier_label: str,
    temperature: object | None,
    max_tokens: object | None,
    reasoning_effort: object | None,
    capabilities: object | None,
) -> str:
    """Render one model detail block."""
    provider_label = provider or "-"
    reasoning = str(reasoning_effort or "").strip() or "-"
    lines = [
        f"[bold]{self._escape_markup(alias)}[/bold]",
        f"  [bold]active:[/] {'yes' if active else 'no'}",
        f"  [bold]provider:[/] {self._escape_markup(provider_label)}",
        "  [bold]protocol:[/] "
        f"{self._escape_markup(self._protocol_for_provider(provider_label))}",
        f"  [bold]endpoint:[/] {self._escape_markup(endpoint)}",
        f"  [bold]model_id:[/] {self._escape_markup(model_id or '-')}",
        f"  [bold]roles:[/] {self._escape_markup(self._format_model_roles(roles))}",
        f"  [bold]tier:[/] {self._escape_markup(tier_label)}",
        f"  [bold]temperature:[/] {self._escape_markup(self._format_temperature(temperature))}",
        f"  [bold]max_tokens:[/] {self._escape_markup(self._format_max_tokens(max_tokens))}",
        f"  [bold]reasoning_effort:[/] {self._escape_markup(reasoning)}",
        "  [bold]capabilities:[/] "
        f"{self._escape_markup(self._format_capabilities(capabilities))}",
    ]
    return "\n".join(lines)


def render_configured_model_block(
    self,
    alias: str,
    cfg: Any,
    *,
    active: bool,
    runtime_model: ModelProvider | None = None,
) -> str:
    """Render a detail block for one configured model alias."""
    provider = self._normalize_provider_name(getattr(cfg, "provider", ""))
    model_id = str(getattr(cfg, "model", "") or "").strip()
    inferred_tier = (
        self._runtime_model_tier(runtime_model)
        if active and runtime_model is not None
        else self._infer_tier_from_config(provider, model_id)
    )
    return self._render_model_block(
        alias=alias,
        active=active,
        provider=provider or "-",
        endpoint=self._endpoint_for_config(provider, getattr(cfg, "base_url", "")),
        model_id=model_id or "-",
        roles=getattr(cfg, "roles", []),
        tier_label=self._format_tier_label(getattr(cfg, "tier", 0), inferred_tier),
        temperature=getattr(cfg, "temperature", None),
        max_tokens=getattr(cfg, "max_tokens", None),
        reasoning_effort=getattr(cfg, "reasoning_effort", ""),
        capabilities=getattr(cfg, "resolved_capabilities", None),
    )


def render_runtime_model_block(
    self,
    model: ModelProvider,
    *,
    alias_override: str,
    active: bool,
) -> str:
    """Render a detail block for a runtime model with no config alias."""
    provider = self._runtime_model_provider(model) or "-"
    tier = self._runtime_model_tier(model)
    return self._render_model_block(
        alias=alias_override,
        active=active,
        provider=provider,
        endpoint=self._runtime_model_endpoint(model),
        model_id=self._runtime_model_id(model) or "-",
        roles=self._runtime_model_roles(model),
        tier_label=str(tier) if isinstance(tier, int) and tier > 0 else "auto",
        temperature=getattr(model, "configured_temperature", None),
        max_tokens=getattr(model, "configured_max_tokens", None),
        reasoning_effort=getattr(getattr(model, "_config", None), "reasoning_effort", ""),
        capabilities=getattr(model, "_capabilities", None),
    )


def render_active_model_info(self) -> str:
    """Render rich `/model` output for the active runtime model."""
    configured = self._configured_models()
    active_alias, ambiguous_aliases = self._resolve_active_model_alias()
    lines = ["[bold #7dcfff]Active Model[/bold #7dcfff]"]

    if self._model is None:
        lines.append("  [dim]No active model configured.[/dim]")
        if configured:
            lines.append("  [dim]Use /models to inspect configured aliases.[/dim]")
        return "\n".join(lines)

    if active_alias is not None:
        lines.append(
            self._render_configured_model_block(
                active_alias,
                configured[active_alias],
                active=True,
                runtime_model=self._model,
            )
        )
    else:
        lines.append(
            self._render_runtime_model_block(
                self._model,
                alias_override="(runtime-only)",
                active=True,
            )
        )

    if ambiguous_aliases:
        lines.append("  [bold]active_alias:[/] ambiguous")
        lines.append(
            "  [bold]candidates:[/] "
            + ", ".join(self._escape_markup(alias) for alias in ambiguous_aliases),
        )
    return "\n".join(lines)


def render_models_catalog(self) -> str:
    """Render rich `/models` output for all configured aliases."""
    configured = self._configured_models()
    active_alias, ambiguous_aliases = self._resolve_active_model_alias()
    lines = ["[bold #7dcfff]Configured Models[/bold #7dcfff]"]

    if not configured:
        lines.append("  [dim]No configured models.[/dim]")
        if self._model is not None:
            lines.append("")
            lines.append("[bold #7dcfff]Active Runtime Model[/bold #7dcfff]")
            lines.append(
                self._render_runtime_model_block(
                    self._model,
                    alias_override="(runtime-only)",
                    active=True,
                )
            )
        return "\n".join(lines)

    aliases = sorted(configured.keys(), key=str.casefold)
    if active_alias and active_alias in configured:
        aliases = [active_alias, *[alias for alias in aliases if alias != active_alias]]

    for idx, alias in enumerate(aliases):
        if idx > 0:
            lines.append("")
        lines.append(
            self._render_configured_model_block(
                alias,
                configured[alias],
                active=alias == active_alias,
                runtime_model=self._model if alias == active_alias else None,
            )
        )

    if ambiguous_aliases:
        lines.append("")
        lines.append("[bold #e0af68]Active alias is ambiguous.[/bold #e0af68]")
        lines.append(
            "  [bold]candidates:[/] "
            + ", ".join(self._escape_markup(alias) for alias in ambiguous_aliases),
        )
    elif self._model is not None and active_alias is None:
        lines.append("")
        lines.append(
            "[bold #e0af68]Active runtime model is not part of configured "
            "aliases.[/bold #e0af68]"
        )
        lines.append(
            self._render_runtime_model_block(
                self._model,
                alias_override="(runtime-only)",
                active=True,
            )
        )
    return "\n".join(lines)


def render_session_info(self, state) -> str:
    """Render `/session` output with compact sections."""
    session_id = self._escape_markup(self._session.session_id if self._session else "?")
    workspace = self._escape_markup(self._session.workspace if self._session else "?")
    process_name = self._escape_markup(self._active_process_name())
    focus = self._escape_markup(state.current_focus or "(none)")

    lines = [
        "[bold #7dcfff]Current Session[/bold #7dcfff]",
        f"  [bold]ID:[/] [dim]{session_id}[/dim]",
        f"  [bold]Workspace:[/] [dim]{workspace}[/dim]",
        f"  [bold]Process:[/] {process_name}",
        f"  [bold]Turns:[/] {state.turn_count}",
        f"  [bold]Tokens:[/] {state.total_tokens:,}",
        f"  [bold]Focus:[/] {focus}",
    ]
    if state.key_decisions:
        lines.append("  [bold]Recent Decisions:[/]")
        for decision in state.key_decisions[-5:]:
            lines.append(
                self._wrap_info_text(
                    self._escape_markup(decision),
                    initial_indent="    - ",
                    subsequent_indent="      ",
                )
            )
    return "\n".join(lines)


def render_sessions_list(self, sessions: list[dict]) -> str:
    """Render `/sessions` output with readable per-session rows."""
    lines = [
        "[bold #7dcfff]Recent Sessions[/bold #7dcfff]",
        "[dim]Use /resume <session-id-prefix> to switch old cowork sessions[/dim]",
    ]
    for row in sessions[:10]:
        sid = self._escape_markup(str(row.get("id", "")))
        turns = int(row.get("turn_count", 0) or 0)
        started = self._escape_markup(str(row.get("started_at", "?"))[:16])
        ws = self._escape_markup(row.get("workspace_path", "?"))

        badges: list[str] = []
        if row.get("is_active"):
            badges.append("[green]active[/green]")
        if self._session and sid == self._session.session_id:
            badges.append("[cyan]current[/cyan]")
        badge_text = f" [dim]({' | '.join(badges)})[/dim]" if badges else ""

        lines.append(
            f"  [bold]{sid[:12]}...[/bold]  [dim]{started}[/dim]  "
            f"{turns} turns{badge_text}"
        )
        lines.append(
            self._wrap_info_text(
                ws,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )
    return "\n".join(lines)


def render_startup_summary(self, *, tool_count: int, persisted: str) -> str:
    """Render startup summary block with workspace/session details."""
    workspace = self._escape_markup(self._workspace)
    process_name = self._escape_markup(self._active_process_name())
    lines = [
        f"[bold #7dcfff]Loom[/bold #7dcfff]  [dim]({self._model.name})[/dim]",
        f"  [bold]Workspace:[/] [dim]{workspace}[/dim]",
        f"  [bold]Tools:[/] {tool_count}",
        f"  [bold]Session Mode:[/] {persisted}",
        f"  [bold]Process:[/] {process_name}",
    ]
    if self._session and self._session.session_id:
        lines.append(
            f"  [bold]Session ID:[/] [dim]{self._escape_markup(self._session.session_id)}[/dim]"
        )
    return "\n".join(lines)


def render_process_usage() -> str:
    """Render process execution mode guidance."""
    return "\n".join([
        "[bold #7dcfff]Process Modes[/bold #7dcfff]",
        "  [bold]Ad hoc:[/] /run <goal>",
        "  [bold]Explicit:[/] /<process-name> <goal>",
        "  [bold]Catalog:[/] /processes",
    ])


def render_tools_catalog(self) -> str:
    """Render `/tools` output as a wrapped catalog."""
    tools = self._tools.list_tools()
    lines = [f"[bold #7dcfff]Tools[/bold #7dcfff] [dim]({len(tools)})[/dim]"]
    if not tools:
        lines.append("  [dim](none)[/dim]")
        return "\n".join(lines)
    joined = ", ".join(self._escape_markup(tool) for tool in tools)
    lines.append(
        self._wrap_info_text(
            joined,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )
    return "\n".join(lines)


def render_process_catalog(self) -> str:
    """Build a human-readable process list."""
    self._refresh_process_command_index()
    available = self._cached_process_catalog
    if not available:
        if self._blocked_process_commands:
            blocked = ", ".join(
                self._escape_markup(name) for name in self._blocked_process_commands
            )
            return (
                "[bold #7dcfff]Available Processes[/bold #7dcfff]\n"
                "  [dim]No selectable process definitions found.[/dim]\n"
                f"  [#f7768e]Blocked (name collisions): {blocked}[/]"
            )
        return (
            "[bold #7dcfff]Available Processes[/bold #7dcfff]\n"
            "  [dim]No process definitions found.[/dim]"
        )

    lines = [
        "[bold #7dcfff]Available Processes[/bold #7dcfff]",
        "[dim]Use /run <goal> for ad hoc or /<process-name> <goal> for explicit runs[/dim]",
    ]
    for proc in available:
        name = str(proc.get("name", "")).strip()
        ver = str(proc.get("version", "")).strip()
        desc = str(proc.get("description", "")).strip().split("\n")[0]
        safe_name = self._escape_markup(name)
        safe_ver = self._escape_markup(ver)
        lines.append(f"  [bold]{safe_name}[/bold] [dim]v{safe_ver}[/dim]")
        if desc:
            lines.append(
                self._wrap_info_text(
                    self._escape_markup(desc),
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )
    if self._blocked_process_commands:
        blocked = ", ".join(
            f"/{self._escape_markup(name)}" for name in self._blocked_process_commands
        )
        lines.append(f"  [#f7768e]Blocked (name collisions): {blocked}[/]")
    return "\n".join(lines)


def render_mcp_list(views: list) -> str:
    """Build a readable MCP server list."""
    if not views:
        return "No MCP servers configured."
    lines = ["[bold]MCP servers:[/bold]"]
    for view in views:
        status = "enabled" if view.server.enabled else "disabled"
        server_type = str(getattr(view.server, "type", "local"))
        lines.append(
            f"  {view.alias:16s} {status:8s} type={server_type:6s} "
            f"[dim]source={view.source}[/dim]"
        )
    return "\n".join(lines)


def render_mcp_view(view) -> str:
    """Build a readable MCP server details block."""
    from loom.mcp.config import redact_server_env, redact_server_headers

    env = redact_server_env(view.server)
    server_type = str(getattr(view.server, "type", "local"))
    lines = [
        f"[bold]{view.alias}[/bold]",
        f"  source: {view.source}",
        f"  source_path: {view.source_path or '-'}",
        f"  type: {server_type}",
        f"  enabled: {view.server.enabled}",
        f"  timeout_seconds: {view.server.timeout_seconds}",
        "  env:",
    ]
    if server_type == "remote":
        oauth_cfg = getattr(view.server, "oauth", None)
        oauth_enabled = bool(getattr(oauth_cfg, "enabled", False))
        oauth_scopes = list(getattr(oauth_cfg, "scopes", []) or [])
        allow_insecure_http = bool(
            getattr(view.server, "allow_insecure_http", False)
        )
        allow_private_network = bool(
            getattr(view.server, "allow_private_network", False)
        )
        lines.extend(
            [
                f"  url: {getattr(view.server, 'url', '') or '-'}",
                "  fallback_sse_url: "
                f"{getattr(view.server, 'fallback_sse_url', '') or '-'}",
                "  oauth: "
                f"{'enabled' if oauth_enabled else 'disabled'}",
                "  oauth_scopes: "
                f"{', '.join(oauth_scopes) if oauth_scopes else '-'}",
                "  allow_insecure_http: "
                f"{'true' if allow_insecure_http else 'false'}",
                "  allow_private_network: "
                f"{'true' if allow_private_network else 'false'}",
                "  headers:",
            ]
        )
        headers = redact_server_headers(view.server)
        if headers:
            for key, value in headers.items():
                lines.append(f"    {key}: {value}")
        else:
            lines.append("    (none)")
    else:
        lines.extend(
            [
                f"  command: {view.server.command}",
                f"  args: {' '.join(view.server.args) if view.server.args else '-'}",
                f"  cwd: {view.server.cwd or '-'}",
            ]
        )
    if env:
        for key, value in env.items():
            lines.append(f"    {key}={value}")
    else:
        lines.append("    (none)")
    return "\n".join(lines)
