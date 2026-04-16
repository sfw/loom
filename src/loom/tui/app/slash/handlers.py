"""Slash command dispatch entrypoint and routing implementation."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from typing import Any

from loom.tui.screens import SetupScreen
from loom.tui.widgets import ChatLog

from ..constants import _RUN_GOAL_FILE_CONTENT_MAX_CHARS
from . import config_command as slash_config_command


async def handle_slash_command(app, text: str) -> bool:
    """Dispatch slash command handling through the slash package implementation."""
    return await handle_slash_command_core(app, text)


async def handle_slash_command_core(self, text: str) -> bool:
    """Handle slash commands. Returns True if handled."""
    raw = text.strip()
    parts = raw.split(None, 1)
    token = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""
    chat = self.query_one("#chat-log", ChatLog)
    self._refresh_process_command_index()

    if token in ("/quit", "/exit", "/q"):
        self.action_request_quit()
        return True
    if token == "/clear":
        self.action_clear_chat()
        return True
    if token == "/help":
        self._show_help()
        return True
    if token == "/model":
        if arg:
            chat.add_info(
                self._render_slash_command_usage("/model", "(no arguments)")
                + "\n[dim]Runtime model switching is not supported yet.[/dim]"
            )
            return True
        chat.add_info(self._render_active_model_info())
        return True
    if token == "/models":
        if arg:
            chat.add_info(
                self._render_slash_command_usage("/models", "(no arguments)")
            )
            return True
        chat.add_info(self._render_models_catalog())
        return True
    if token == "/mcp":
        from loom.mcp.config import (
            MCPConfigManagerError,
            ensure_valid_alias,
            merge_server_edits,
            parse_mcp_server_from_flags,
        )

        manager = self._mcp_manager()
        if not arg:
            self._open_mcp_manager_screen()
            return True

        subparts = arg.split(None, 1)
        subcmd = subparts[0].lower()
        rest = subparts[1].strip() if len(subparts) > 1 else ""

        if subcmd == "manage":
            self._open_mcp_manager_screen()
            return True

        if subcmd == "list":
            try:
                merged = await asyncio.to_thread(manager.load)
                views = merged.as_views()
                output = self._render_mcp_list(views)
                if any(view.source == "legacy" for view in views):
                    output += (
                        "\n[dim]Legacy MCP config detected in loom.toml. "
                        "Run `loom mcp migrate` from CLI.[/dim]"
                    )
                chat.add_info(output)
            except Exception as e:
                chat.add_info(f"[bold #f7768e]MCP list failed: {e}[/]")
            return True

        if subcmd == "show":
            if not rest:
                chat.add_info(self._render_slash_command_usage("/mcp show", "<alias>"))
                return True
            try:
                alias = ensure_valid_alias(rest)
                view = await asyncio.to_thread(manager.get_view, alias)
            except MCPConfigManagerError as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
                return True
            if view is None:
                chat.add_info(
                    f"[bold #f7768e]MCP server not found: {alias}[/]"
                )
                return True
            output = self._render_mcp_view(view)
            if view.source == "legacy":
                output += (
                    "\n[dim]This alias comes from legacy loom.toml. "
                    "Run `loom mcp migrate` from CLI to move it.[/dim]"
                )
            chat.add_info(output)
            return True

        if subcmd == "test":
            if not rest:
                chat.add_info(self._render_slash_command_usage("/mcp test", "<alias>"))
                return True
            try:
                alias = ensure_valid_alias(rest)
                view, tools = await asyncio.to_thread(
                    manager.probe_server,
                    alias,
                )
            except Exception as e:
                chat.add_info(
                    f"[bold #f7768e]MCP probe failed for '{rest}': {e}[/]"
                )
                return True
            names = [str(tool.get("name", "")) for tool in tools]
            lines = [
                f"MCP probe succeeded for [bold]{view.alias}[/bold].",
                f"Tools discovered: {len(names)}",
            ]
            for name in names:
                lines.append(f"  - {name}")
            chat.add_info("\n".join(lines))
            return True

        if subcmd == "add":
            if not rest:
                chat.add_info(
                    self._render_slash_command_usage(
                        "/mcp add",
                        (
                            "<alias> --command <cmd> [--arg <value>] "
                            "[--env KEY=VALUE] [--env-ref KEY=ENV] "
                            "[--cwd <path>] [--timeout <seconds>] [--disabled]"
                        ),
                    )
                )
                return True
            try:
                tokens = self._split_slash_args(rest)
            except ValueError as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
                return True
            if not tokens:
                chat.add_info(
                    self._render_slash_command_usage(
                        "/mcp add",
                        "<alias> --command <cmd> [...]",
                    )
                )
                return True
            alias_token = tokens[0]
            args_tokens = tokens[1:]
            command = ""
            cmd_args: list[str] = []
            env_pairs: list[str] = []
            env_refs: list[str] = []
            cwd = ""
            timeout = 30
            disabled = False
            index = 0
            while index < len(args_tokens):
                item = args_tokens[index]
                if item == "--disabled":
                    disabled = True
                    index += 1
                    continue
                if item in {
                    "--command",
                    "--arg",
                    "--env",
                    "--env-ref",
                    "--cwd",
                    "--timeout",
                }:
                    if index + 1 >= len(args_tokens):
                        chat.add_info(
                            f"[bold #f7768e]Missing value for {item}.[/]"
                        )
                        return True
                    value = args_tokens[index + 1]
                    if item == "--command":
                        command = value
                    elif item == "--arg":
                        cmd_args.append(value)
                    elif item == "--env":
                        env_pairs.append(value)
                    elif item == "--env-ref":
                        env_refs.append(value)
                    elif item == "--cwd":
                        cwd = value
                    elif item == "--timeout":
                        try:
                            timeout = int(value)
                        except ValueError:
                            chat.add_info(
                                "[bold #f7768e]--timeout must be an integer.[/]"
                            )
                            return True
                    index += 2
                    continue
                chat.add_info(
                    f"[bold #f7768e]Unknown /mcp add option: {item}[/]"
                )
                return True
            try:
                alias = ensure_valid_alias(alias_token)
                server = parse_mcp_server_from_flags(
                    command=command,
                    args=tuple(cmd_args),
                    env_pairs=tuple(env_pairs),
                    env_refs=tuple(env_refs),
                    cwd=cwd,
                    timeout=timeout,
                    disabled=disabled,
                )
                await asyncio.to_thread(manager.add_server, alias, server)
                await self._reload_mcp_runtime()
                chat.add_info(f"MCP server '{alias}' added.")
            except Exception as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
            return True

        if subcmd == "edit":
            if not rest:
                chat.add_info(
                    self._render_slash_command_usage(
                        "/mcp edit",
                        (
                            "<alias> [--command <cmd>] [--arg <value>] "
                            "[--env KEY=VALUE] [--env-ref KEY=ENV] "
                            "[--cwd <path>] [--timeout <seconds>] "
                            "[--enable|--disabled]"
                        ),
                    )
                )
                return True
            try:
                tokens = self._split_slash_args(rest)
            except ValueError as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
                return True
            if not tokens:
                chat.add_info(
                    self._render_slash_command_usage("/mcp edit", "<alias> [...]")
                )
                return True
            alias_token = tokens[0]
            args_tokens = tokens[1:]
            command: str | None = None
            cmd_args: list[str] = []
            env_pairs: list[str] = []
            env_refs: list[str] = []
            cwd: str | None = None
            timeout: int | None = None
            enabled_toggle: bool | None = None
            index = 0
            while index < len(args_tokens):
                item = args_tokens[index]
                if item == "--enable":
                    enabled_toggle = True
                    index += 1
                    continue
                if item == "--disabled":
                    enabled_toggle = False
                    index += 1
                    continue
                if item in {
                    "--command",
                    "--arg",
                    "--env",
                    "--env-ref",
                    "--cwd",
                    "--timeout",
                }:
                    if index + 1 >= len(args_tokens):
                        chat.add_info(
                            f"[bold #f7768e]Missing value for {item}.[/]"
                        )
                        return True
                    value = args_tokens[index + 1]
                    if item == "--command":
                        command = value
                    elif item == "--arg":
                        cmd_args.append(value)
                    elif item == "--env":
                        env_pairs.append(value)
                    elif item == "--env-ref":
                        env_refs.append(value)
                    elif item == "--cwd":
                        cwd = value
                    elif item == "--timeout":
                        try:
                            timeout = int(value)
                        except ValueError:
                            chat.add_info(
                                "[bold #f7768e]--timeout must be an integer.[/]"
                            )
                            return True
                    index += 2
                    continue
                chat.add_info(
                    f"[bold #f7768e]Unknown /mcp edit option: {item}[/]"
                )
                return True

            if (
                command is None
                and not cmd_args
                and not env_pairs
                and not env_refs
                and cwd is None
                and timeout is None
                and enabled_toggle is None
            ):
                chat.add_info(
                    "[bold #f7768e]/mcp edit requires at least one change flag.[/]"
                )
                return True

            try:
                alias = ensure_valid_alias(alias_token)

                def _mutate(current):
                    merged = merge_server_edits(
                        current=current,
                        command=command,
                        args=tuple(cmd_args),
                        env_pairs=tuple(env_pairs),
                        env_refs=tuple(env_refs),
                        cwd=cwd,
                        timeout=timeout,
                        disabled=(enabled_toggle is False),
                    )
                    if enabled_toggle is True:
                        return replace(merged, enabled=True)
                    return merged

                await asyncio.to_thread(manager.edit_server, alias, _mutate)
                await self._reload_mcp_runtime()
                chat.add_info(f"MCP server '{alias}' updated.")
            except Exception as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
            return True

        if subcmd in {"enable", "disable"}:
            if not rest:
                chat.add_info(self._render_slash_command_usage(f"/mcp {subcmd}", "<alias>"))
                return True
            try:
                alias = ensure_valid_alias(rest)
                enabled = subcmd == "enable"
                await asyncio.to_thread(
                    manager.edit_server,
                    alias,
                    lambda current: replace(current, enabled=enabled),
                )
                await self._reload_mcp_runtime()
                chat.add_info(
                    f"MCP server '{alias}' "
                    f"{'enabled' if enabled else 'disabled'}."
                )
            except Exception as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
            return True

        if subcmd == "remove":
            if not rest:
                chat.add_info(self._render_slash_command_usage("/mcp remove", "<alias>"))
                return True
            try:
                alias = ensure_valid_alias(rest)
                from loom.auth.resources import (
                    cleanup_deleted_resource,
                    resource_delete_impact,
                )

                impact = await asyncio.to_thread(
                    resource_delete_impact,
                    workspace=self._workspace,
                    resource_kind="mcp",
                    resource_key=alias,
                )
                await asyncio.to_thread(manager.remove_server, alias)
                await asyncio.to_thread(
                    cleanup_deleted_resource,
                    workspace=self._workspace,
                    explicit_auth_path=self._explicit_auth_path,
                    resource_kind="mcp",
                    resource_key=alias,
                )
                await self._reload_mcp_runtime()
                impact_text = ""
                if impact.resource_id:
                    impact_text = (
                        " "
                        f"(auth cleanup: {len(impact.active_profile_ids)} profile(s), "
                        f"{len(impact.active_binding_ids)} binding(s), "
                        f"default={'yes' if impact.workspace_default_profile_id else 'no'})"
                    )
                chat.add_info(f"MCP server '{alias}' removed.{impact_text}")
            except Exception as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
            return True

        chat.add_info(
            self._render_slash_command_usage(
                "/mcp",
                (
                    "[manage|list|show <alias>|test <alias>|add <alias> ...|"
                    "edit <alias> ...|enable <alias>|disable <alias>|remove <alias>]"
                ),
            )
        )
        return True
    if token == "/auth":
        if not arg:
            self._open_auth_manager_screen()
            return True

        subparts = arg.split(None, 1)
        subcmd = subparts[0].lower()
        if subcmd == "manage":
            self._open_auth_manager_screen()
            return True

        chat.add_info(
            "[bold #7dcfff]TUI /auth is manager-first.[/]\n"
            "Use [bold]/auth[/bold] or [bold]/auth manage[/bold] to open "
            "the auth manager.\n"
            "Use CLI for scriptable auth actions: "
            "[bold]loom auth list|show|check|select|unset|"
            "profile add|profile edit|profile remove[/bold]."
        )
        return True
    if token == "/setup":
        self.push_screen(
            SetupScreen(), callback=self._on_setup_complete,
        )
        return True
    if token == "/tool":
        raw_tool_name, raw_json_args = self._split_tool_slash_args(arg)
        tool_name = self._strip_wrapping_quotes(raw_tool_name)
        if not tool_name:
            chat.add_info(
                self._render_slash_command_usage(
                    "/tool",
                    "<tool-name> [key=value ... | json-object-args]",
                )
            )
            return True

        tool_args, parse_error = self._parse_tool_slash_arguments(raw_json_args)
        if tool_args is None:
            chat.add_info(
                "[bold #f7768e]"
                f"{self._escape_markup(parse_error)}[/]\n"
                + self._render_slash_command_usage(
                    "/tool",
                    "<tool-name> [key=value ... | json-object-args]",
                )
            )
            return True

        available_tools = self._tool_name_inventory()
        resolved_tool_name = next(
            (name for name in available_tools if name.lower() == tool_name.lower()),
            "",
        )
        if not resolved_tool_name:
            suggestions = [
                name for name in available_tools
                if name.lower().startswith(tool_name.lower())
            ]
            message = (
                f"[bold #f7768e]Unknown tool:[/] {self._escape_markup(tool_name)}"
            )
            if suggestions:
                rendered = ", ".join(
                    self._escape_markup(name)
                    for name in suggestions
                )
                message += f"\n[dim]Matches: {rendered}[/dim]"
            chat.add_info(message)
            return True

        tool_args = dict(tool_args)
        if self.is_running:
            self.run_worker(
                self._execute_slash_tool_command(resolved_tool_name, tool_args),
                group=f"slash-tool-command-{uuid.uuid4().hex[:8]}",
                exclusive=False,
            )
        else:
            await self._execute_slash_tool_command(resolved_tool_name, tool_args)
        return True
    if token == "/tools":
        chat.add_info(self._render_tools_catalog())
        return True
    if token == "/tokens":
        chat.add_info(f"Session tokens: {self._total_tokens:,}")
        return True
    if token == "/config":
        chat.add_info(slash_config_command.handle_config_command(self, arg))
        return True
    if token == "/telemetry":
        if not arg or arg.lower() in {"status", "show"}:
            chat.add_info(self._render_telemetry_mode_status())
            return True
        try:
            tokens = self._split_slash_args(arg)
        except ValueError as e:
            chat.add_info(f"[bold #f7768e]{e}[/]")
            return True
        if len(tokens) != 1:
            chat.add_info(
                self._render_slash_command_usage(
                    "/telemetry",
                    "[status|off|active|all_typed|debug|internal_only]",
                ),
            )
            return True
        entry, parsed, _snapshot = self._set_runtime_config_value(
            path="telemetry.mode",
            raw_value=tokens[0],
        )
        self._refresh_runtime_config_bindings()
        lines = [
            "Telemetry mode updated.",
            f"effective mode: [bold]{self._escape_markup(parsed.display_value)}[/bold]",
        ]
        if parsed.warning_code:
            lines.append(
                "[dim]Input normalized to canonical mode "
                f"({self._escape_markup(parsed.warning_code)}).[/dim]"
            )
        lines.append(self._render_telemetry_mode_status())
        chat.add_info("\n".join(lines))
        return True
    if token == "/processes":
        self._refresh_process_command_index()
        if arg:
            chat.add_info(self._render_slash_command_usage("/processes", ""))
            return True
        chat.add_info(self._render_process_catalog())
        return True

    if token == "/run":
        if arg:
            try:
                tokens = self._split_slash_args(arg)
            except ValueError as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
                return True
            if tokens:
                subcmd = tokens[0].lower()
                if subcmd == "close":
                    target = " ".join(tokens[1:]).strip()
                    await self._close_process_run_from_target(target)
                    return True
                if subcmd == "pause":
                    target = tokens[1] if len(tokens) >= 2 else "current"
                    await self._pause_process_run_from_target(target)
                    return True
                if subcmd == "play":
                    target = tokens[1] if len(tokens) >= 2 else "current"
                    await self._play_process_run_from_target(target)
                    return True
                if subcmd == "stop":
                    target = tokens[1] if len(tokens) >= 2 else "current"
                    await self._stop_process_run_from_target(target)
                    return True
                if subcmd == "inject":
                    if len(tokens) < 2:
                        chat.add_info(
                            self._render_slash_command_usage(
                                "/run inject",
                                "<run-id-prefix|current> <text>",
                            )
                        )
                        return True
                    target = "current"
                    inject_tokens = tokens[1:]
                    if len(tokens) >= 3:
                        target_candidate = str(tokens[1] or "").strip()
                        explicit_target = (
                            target_candidate.startswith("#")
                            or target_candidate.lower() in {"current", "this"}
                        )
                        if explicit_target:
                            target = target_candidate
                            inject_tokens = tokens[2:]
                        else:
                            run_candidate, _ = self._resolve_process_run_target(
                                target_candidate,
                            )
                            if run_candidate is not None:
                                target = target_candidate
                                inject_tokens = tokens[2:]
                    inject_text = " ".join(
                        str(item or "").strip()
                        for item in inject_tokens
                        if str(item or "").strip()
                    ).strip()
                    if not inject_text:
                        chat.add_info(
                            self._render_slash_command_usage(
                                "/run inject",
                                "<run-id-prefix|current> <text>",
                            )
                        )
                        return True
                    await self._inject_process_run_from_target(
                        target,
                        inject_text,
                        source="slash",
                    )
                    return True
                if subcmd == "resume":
                    if len(tokens) < 2:
                        chat.add_info(
                            self._render_slash_command_usage(
                                "/run resume",
                                "<run-id-prefix|current>",
                            )
                        )
                        return True
                    target = tokens[1]
                    await self._resume_process_run_from_target(target)
                    return True
                if subcmd == "save":
                    if len(tokens) < 3:
                        chat.add_info(
                            self._render_slash_command_usage(
                                "/run save",
                                "<run-id-prefix|current> <name>",
                            )
                        )
                        return True
                    target = tokens[1]
                    package_name = tokens[2]
                    run, error = self._resolve_process_run_target(target)
                    if run is None:
                        chat.add_info(error or "Run not found.")
                        return True
                    if run.process_defn is None or not bool(
                        getattr(run, "is_adhoc", False),
                    ):
                        chat.add_info(
                            "[bold #f7768e]Only ad hoc /run processes can be saved.[/]"
                        )
                        return True
                    try:
                        saved_dir = await asyncio.to_thread(
                            self._save_adhoc_process_package,
                            process_defn=run.process_defn,
                            package_name=package_name,
                            recommended_tools=run.recommended_tools,
                        )
                    except Exception as e:
                        chat.add_info(
                            f"[bold #f7768e]Failed to save process package: {e}[/]"
                        )
                        return True
                    self._refresh_process_command_index()
                    safe_name = self._sanitize_kebab_token(
                        package_name,
                        fallback="adhoc-process",
                        max_len=40,
                    )
                    chat.add_info(
                        "Saved ad hoc process package:\n"
                        f"  [bold]{safe_name}[/bold]\n"
                        f"  [dim]{saved_dir}[/dim]\n"
                        f"Run it with [bold]/{safe_name} <goal>[/bold]."
                    )
                    return True

        goal = ""
        goal_tokens: list[str] = []
        force_fresh = False
        if arg:
            try:
                tokens = self._split_slash_args(arg)
            except ValueError as e:
                chat.add_info(f"[bold #f7768e]{e}[/]")
                return True
            idx = 0
            while idx < len(tokens):
                item = str(tokens[idx] or "").strip()
                if item in {"--fresh", "-f"}:
                    force_fresh = True
                    idx += 1
                    continue
                if item in {"--process", "-p"}:
                    chat.add_info(
                        "[bold #f7768e]/run --process is not supported in TUI.[/]\n"
                        "Use [bold]/<process-name> <goal>[/bold] for explicit runs."
                    )
                    return True
                if item.startswith("-"):
                    chat.add_info(
                        self._render_slash_command_usage(
                            "/run",
                            "[--fresh] <goal>",
                        )
                    )
                    return True
                break
            goal_tokens = [
                str(item or "").strip()
                for item in tokens[idx:]
                if str(item or "").strip()
            ]
            goal = " ".join(goal_tokens).strip()
        else:
            goal = ""
        if not goal:
            chat.add_info(self._render_slash_command_usage("/run", "<goal>"))
            return True

        execution_goal = goal
        synthesis_goal = goal
        goal_context_overrides: dict[str, Any] = {}
        if goal_tokens:
            (
                execution_goal,
                synthesis_goal,
                goal_context_overrides,
                file_goal_error,
            ) = self._expand_run_goal_file_input(goal_tokens)
            if file_goal_error:
                chat.add_info(
                    f"[bold #f7768e]{self._escape_markup(file_goal_error)}[/]"
                )
                return True
            file_context = goal_context_overrides.get("run_goal_file_input", {})
            if isinstance(file_context, dict) and file_context:
                file_label = self._escape_markup(str(file_context.get("path", "")).strip())
                truncated = bool(file_context.get("truncated", False))
                max_chars = int(
                    file_context.get("max_chars", _RUN_GOAL_FILE_CONTENT_MAX_CHARS),
                )
                trunc_note = (
                    f" (truncated to {max_chars:,} chars)"
                    if truncated
                    else ""
                )
                chat.add_info(
                    f"Loaded /run goal file [bold]{file_label}[/bold]{trunc_note}."
                )

        process_defn = None
        start_kwargs: dict[str, Any] = {
            "process_defn": process_defn,
            "process_name_override": None,
            "is_adhoc": True,
            "synthesis_goal": synthesis_goal,
            "force_fresh": force_fresh,
        }
        if goal_context_overrides:
            start_kwargs["goal_context_overrides"] = goal_context_overrides
        await self._start_process_run(execution_goal, **start_kwargs)
        return True

    if token == "/pause":
        if arg:
            chat.add_info(self._render_slash_command_usage("/pause", "(no arguments)"))
            return True
        await self._request_chat_pause(source="slash")
        return True

    if token == "/inject":
        inject_text = self._strip_wrapping_quotes(arg)
        if not inject_text:
            chat.add_info(self._render_slash_command_usage("/inject", "<text>"))
            return True
        await self._queue_chat_inject_instruction(inject_text, source="slash")
        return True

    if token == "/redirect":
        redirect_text = self._strip_wrapping_quotes(arg)
        if not redirect_text:
            chat.add_info(self._render_slash_command_usage("/redirect", "<text>"))
            return True
        if self._chat_redirect_inflight:
            chat.add_info("Redirect already in progress.")
            return True
        self._chat_redirect_inflight = True
        self._sync_chat_stop_control()
        try:
            await self._request_chat_redirect(redirect_text, source="slash")
        finally:
            self._chat_redirect_inflight = False
            self._sync_chat_stop_control()
        return True

    if token == "/steer":
        if not arg:
            chat.add_info(
                self._render_slash_command_usage(
                    "/steer",
                    "pause|resume|queue|clear",
                )
            )
            return True
        subparts = arg.split(None, 1)
        subcmd = str(subparts[0] or "").strip().lower()
        rest = subparts[1].strip() if len(subparts) > 1 else ""
        if rest:
            chat.add_info(
                self._render_slash_command_usage(
                    "/steer",
                    "pause|resume|queue|clear",
                )
            )
            return True
        if subcmd == "pause":
            await self._request_chat_pause(source="slash")
            return True
        if subcmd == "resume":
            await self._request_chat_resume(source="slash")
            return True
        if subcmd == "queue":
            chat.add_info(self._render_steer_queue_status())
            return True
        if subcmd == "clear":
            await self._clear_chat_steering(source="slash")
            return True
        chat.add_info(
            self._render_slash_command_usage(
                "/steer",
                "pause|resume|queue|clear",
            )
        )
        return True

    if token == "/stop":
        if arg:
            chat.add_info(self._render_slash_command_usage("/stop", "(no arguments)"))
            return True
        self.action_stop_chat()
        return True

    process_name = self._process_command_map.get(token)
    if process_name:
        goal = self._strip_wrapping_quotes(arg)
        if not goal:
            chat.add_info(
                self._render_slash_command_usage(f"/{process_name}", "<goal>")
            )
            return True
        await self._start_process_run(
            goal,
            process_name_override=process_name,
            command_prefix=f"/{process_name}",
        )
        return True

    # Persistence-dependent commands
    if token == "/session":
        if not self._session:
            chat.add_info("No active session.")
            return True
        state = self._session.session_state
        chat.add_info(self._render_session_info(state))
        return True

    if token == "/new":
        if self._chat_busy:
            chat.add_info(
                "[bold #f7768e]Cannot create/switch sessions while a turn is running.[/]"
            )
            return True
        if self._store:
            await self._new_session()
        else:
            chat.add_info("No database — sessions are ephemeral.")
        return True

    if token == "/sessions":
        if not self._store:
            chat.add_info("No database — sessions are ephemeral.")
            return True
        all_sessions = await self._store.list_sessions()
        if not all_sessions:
            chat.add_info("No previous sessions.")
            return True
        chat.add_info(self._render_sessions_list(all_sessions))
        return True

    if token == "/resume":
        if self._chat_busy:
            chat.add_info(
                "[bold #f7768e]Cannot create/switch sessions while a turn is running.[/]"
            )
            return True
        if not arg:
            chat.add_info(
                self._render_slash_command_usage("/resume", "<session-id-prefix>")
            )
            return True
        prefix = arg.lower()
        if not self._store:
            chat.add_info("No database — sessions are ephemeral.")
            return True
        all_sessions = await self._store.list_sessions()
        match = None
        for s in all_sessions:
            if s["id"].lower().startswith(prefix):
                match = s
                break
        if match:
            try:
                await self._switch_to_session(match["id"])
            except Exception as e:
                chat.add_info(f"[bold #f7768e]Resume failed: {e}[/]")
        else:
            chat.add_info(f"No session found matching '{prefix}'.")
        return True

    if token == "/history":
        if not arg:
            loaded = await self._load_older_chat_history()
            chat.add_info(
                "Loaded older chat history."
                if loaded
                else "No older chat history available."
            )
            return True

        history_parts = arg.split(None, 1)
        history_cmd = history_parts[0].lower()
        history_rest = history_parts[1].strip() if len(history_parts) > 1 else ""

        if history_cmd in {"older", "more"}:
            loaded = await self._load_older_chat_history()
            chat.add_info(
                "Loaded older chat history."
                if loaded
                else "No older chat history available."
            )
            return True

        if history_cmd == "latest":
            self._jump_chat_history_latest()
            chat.add_info("Jumped to the latest chat rows.")
            return True

        if history_cmd == "transcript":
            if not history_rest:
                state = "on" if bool(getattr(self, "_chat_transcript_mode", False)) else "off"
                chat.add_info(f"Transcript mode is {state}.")
                return True
            state_token = history_rest.lower()
            if state_token not in {"on", "off"}:
                chat.add_info(
                    self._render_slash_command_usage("/history transcript", "[on|off]")
                )
                return True
            enabled = state_token == "on"
            self._set_chat_transcript_mode(enabled)
            chat.add_info(
                "Transcript mode enabled."
                if enabled
                else "Transcript mode disabled."
            )
            return True

        if history_cmd == "thinking":
            if not history_rest:
                state = (
                    "shown"
                    if bool(getattr(self, "_chat_transcript_show_thinking", False))
                    else "hidden"
                )
                chat.add_info(f"Transcript thinking is {state}.")
                return True
            state_token = history_rest.lower()
            if state_token not in {"show", "hide"}:
                chat.add_info(
                    self._render_slash_command_usage("/history thinking", "[show|hide]")
                )
                return True
            enabled = state_token == "show"
            self._set_chat_transcript_show_thinking(enabled)
            chat.add_info(
                "Transcript thinking shown."
                if enabled
                else "Transcript thinking hidden."
            )
            return True

        if history_cmd == "search":
            if not history_rest:
                chat.add_info(
                    self._render_slash_command_usage("/history search", "<query>")
                )
                return True
            matches = self._search_chat_history(history_rest)
            escaped_query = self._escape_markup(history_rest)
            if matches:
                chat.add_info(
                    f"Found {matches} transcript match(es) for '{escaped_query}'."
                )
            else:
                chat.add_info(
                    f"No transcript matches for '{escaped_query}'."
                )
            return True

        if history_cmd in {"next", "prev"}:
            moved, current, total = self._step_chat_history_search(
                1 if history_cmd == "next" else -1
            )
            if moved:
                chat.add_info(f"Moved to transcript match {current}/{total}.")
            else:
                chat.add_info("No active transcript search.")
            return True

        if history_cmd == "clear-search":
            cleared = self._clear_chat_search()
            chat.add_info(
                "Cleared transcript search."
                if cleared
                else "No active transcript search."
            )
            return True

        chat.add_info(
            self._render_slash_command_usage(
                "/history",
                (
                    "[older|latest|transcript [on|off]|thinking [show|hide]|"
                    "search <query>|next|prev|clear-search]"
                ),
            )
        )
        return True

    if token == "/learned":
        if not self._db:
            chat.add_info("No database — learned patterns unavailable.")
            return True
        await self._show_learned_patterns()
        return True

    return False
