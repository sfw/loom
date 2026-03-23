"""Process-run auth preflight and selection helpers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loom.tui.screens import AskUserScreen, AuthManagerModalScreen
from loom.tui.widgets import ChatLog

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition


def format_auth_profile_option(profile: Any) -> str:
    """Render one concise auth profile choice label."""
    profile_id = str(getattr(profile, "profile_id", "") or "").strip()
    label = str(getattr(profile, "account_label", "") or "").strip()
    mcp_server = str(getattr(profile, "mcp_server", "") or "").strip()
    parts = [profile_id]
    if label:
        parts.append(f"label={label}")
    if mcp_server:
        parts.append(f"mcp_server={mcp_server}")
    return " | ".join(parts)


async def prompt_auth_choice(
    self,
    question: str,
    options: list[str],
    *,
    run_id: str = "",
) -> str:
    """Prompt for one auth selection via modal and return chosen option text."""
    answer_event = asyncio.Event()
    selected: list[str] = []

    def _handle(answer: str) -> None:
        selected.append(str(answer or "").strip())
        answer_event.set()

    self._begin_process_run_user_input_pause(run_id)
    self.push_screen(AskUserScreen(question, options), callback=_handle)
    try:
        await answer_event.wait()
    finally:
        self._end_process_run_user_input_pause(run_id)
    return selected[0] if selected else ""


async def open_auth_manager_for_run_start(
    self,
    *,
    process_def: ProcessDefinition | None = None,
    run_id: str = "",
) -> bool:
    """Open auth manager during run-start flow and wait for completion."""
    done = asyncio.Event()
    changed = {"value": False}

    def _handle(result: dict[str, object] | None) -> None:
        changed["value"] = bool(
            isinstance(result, dict) and result.get("changed")
        )
        done.set()

    process_defs = self._auth_discovery_process_defs()
    if process_def is not None:
        name = str(getattr(process_def, "name", "")).strip()
        if not name or all(
            str(getattr(item, "name", "")).strip() != name for item in process_defs
        ):
            process_defs.append(process_def)
    if process_defs:
        self._refresh_tool_registry()
    self._begin_process_run_user_input_pause(run_id)
    self.push_screen(
        AuthManagerModalScreen(
            workspace=self._workspace,
            explicit_auth_path=self._explicit_auth_path,
            mcp_manager=self._mcp_manager(),
            process_def=process_def or self._process_defn,
            process_defs=process_defs,
            tool_registry=self._tools,
        ),
        callback=_handle,
    )
    try:
        await done.wait()
    finally:
        self._end_process_run_user_input_pause(run_id)
    return bool(changed["value"])


def collect_required_auth_resources_for_process(
    self,
    process_defn: ProcessDefinition | None,
) -> list[dict[str, Any]]:
    """Collect process + allowed-tool auth requirements in metadata shape."""
    if process_defn is None:
        return []

    from loom.auth.runtime import (
        coerce_auth_requirements,
        serialize_auth_requirements,
    )

    raw_items: list[object] = []
    auth_block = getattr(process_defn, "auth", None)
    process_required = getattr(auth_block, "required", [])
    if isinstance(process_required, list):
        raw_items.extend(process_required)

    tools_cfg = getattr(process_defn, "tools", None)
    excluded = {
        str(item).strip()
        for item in (getattr(tools_cfg, "excluded", []) or [])
        if str(item).strip()
    }
    required = {
        str(item).strip()
        for item in (getattr(tools_cfg, "required", []) or [])
        if str(item).strip()
    }
    if required:
        # If a process explicitly declares required tools, preflight auth
        # against that set only to avoid unrelated auth prompts.
        candidate_tool_names = sorted(required - excluded)
    else:
        candidate_tool_names = sorted(
            {
                str(name).strip()
                for name in self._tools.list_tools()
                if str(name).strip() and str(name).strip() not in excluded
            }
        )

    for tool_name in candidate_tool_names:
        tool = self._tools.get(tool_name)
        if tool is None:
            continue
        declared = getattr(tool, "auth_requirements", [])
        if isinstance(declared, list):
            raw_items.extend(declared)

    return serialize_auth_requirements(coerce_auth_requirements(raw_items))


async def resolve_auth_overrides_for_run_start(
    self,
    *,
    process_defn: ProcessDefinition | None,
    base_overrides: dict[str, str],
    run_id: str = "",
) -> tuple[dict[str, str] | None, list[dict[str, Any]]]:
    """Resolve run-start auth, prompting for ambiguous resources when needed."""
    from loom.auth.config import (
        load_merged_auth_config,
        set_workspace_auth_default,
    )
    from loom.auth.runtime import (
        UnresolvedAuthResourcesError,
        build_run_auth_context,
        coerce_auth_requirements,
    )

    required_resources = self._collect_required_auth_resources_for_process(process_defn)
    if not required_resources:
        return dict(base_overrides), required_resources

    overrides = dict(base_overrides)
    while True:
        metadata: dict[str, Any] = {
            "auth_workspace": str(self._workspace.resolve()),
            "auth_required_resources": required_resources,
            "auth_profile_overrides": dict(overrides),
        }
        if self._explicit_auth_path is not None:
            metadata["auth_config_path"] = str(self._explicit_auth_path.resolve())

        try:
            auth_context = await asyncio.to_thread(
                build_run_auth_context,
                workspace=self._workspace,
                metadata=metadata,
                required_resources=coerce_auth_requirements(required_resources),
                available_mcp_aliases=set(self._config.mcp.servers.keys()),
            )
        except UnresolvedAuthResourcesError as e:
            unresolved = list(e.unresolved)
            blocking = [
                item
                for item in unresolved
                if str(getattr(item, "reason", "")).strip()
                not in {"ambiguous", "blocked_ambiguous_binding"}
            ]
            if blocking:
                chat = self.query_one("#chat-log", ChatLog)
                lines = [
                    "[bold #f7768e]Run auth preflight failed.[/]",
                ]
                for item in blocking:
                    lines.append(
                        f"  - provider={item.provider} source={item.source} "
                        f"reason={item.reason}"
                    )
                    message = str(item.message or "").strip()
                    if message:
                        lines.append(f"    {message}")
                chat.add_info("\n".join(lines))
                choice = await self._prompt_auth_choice(
                    "Open Auth Manager to fix auth and retry this run?",
                    ["Open Auth Manager", "Cancel run"],
                    run_id=run_id,
                )
                if choice != "Open Auth Manager":
                    chat.add_info("Run cancelled: unresolved auth requirements.")
                    return None, required_resources
                changed = await self._open_auth_manager_for_run_start(
                    process_def=process_defn,
                    run_id=run_id,
                )
                if changed:
                    chat.add_info("Auth configuration updated. Retrying preflight.")
                else:
                    chat.add_info(
                        "Auth Manager closed without changes. Retrying preflight."
                    )
                continue

            made_selection = False
            for item in unresolved:
                candidates = list(item.candidates)
                if not candidates:
                    continue
                merged = await asyncio.to_thread(
                    load_merged_auth_config,
                    workspace=self._workspace,
                    explicit_path=self._explicit_auth_path,
                )
                options: list[str] = []
                option_to_profile: dict[str, str] = {}
                for candidate_id in candidates:
                    profile = merged.config.profiles.get(candidate_id)
                    if profile is None:
                        label = str(candidate_id)
                    else:
                        label = self._format_auth_profile_option(profile)
                    option_to_profile[label] = candidate_id
                    options.append(label)
                question = (
                    "Select auth profile for "
                    f"provider={item.provider} source={item.source}"
                )
                answer = await self._prompt_auth_choice(
                    question,
                    options,
                    run_id=run_id,
                )
                profile_id = option_to_profile.get(answer)
                if not profile_id:
                    chat = self.query_one("#chat-log", ChatLog)
                    chat.add_info("Run cancelled: auth selection was not completed.")
                    return None, required_resources
                selector = (
                    str(getattr(item, "resource_id", "")).strip()
                    or str(getattr(item, "resource_ref", "")).strip()
                    or str(getattr(item, "provider", "")).strip()
                )
                if not selector:
                    chat = self.query_one("#chat-log", ChatLog)
                    chat.add_info(
                        "Run cancelled: unresolved auth item has no selector."
                    )
                    return None, required_resources
                overrides[selector] = profile_id
                save_default = await self._prompt_auth_choice(
                    (
                        "Save this as workspace default for "
                        f"{selector}?"
                    ),
                    ["No", "Yes"],
                    run_id=run_id,
                )
                if save_default == "Yes":
                    try:
                        if str(getattr(item, "resource_id", "")).strip():
                            from loom.auth.resources import (
                                default_workspace_auth_resources_path,
                                set_workspace_resource_default,
                            )

                            await asyncio.to_thread(
                                set_workspace_resource_default,
                                default_workspace_auth_resources_path(
                                    self._workspace.resolve()
                                ),
                                resource_id=str(item.resource_id).strip(),
                                profile_id=profile_id,
                            )
                        else:
                            await asyncio.to_thread(
                                set_workspace_auth_default,
                                self._auth_defaults_path(),
                                selector=selector,
                                profile_id=profile_id,
                            )
                    except Exception as e:
                        chat = self.query_one("#chat-log", ChatLog)
                        chat.add_info(
                            f"[bold #f7768e]Failed to save workspace auth default: {e}[/]"
                        )
                made_selection = True
            if not made_selection:
                chat = self.query_one("#chat-log", ChatLog)
                chat.add_info("Run cancelled: auth selection was not completed.")
                return None, required_resources
            continue
        except Exception as e:
            chat = self.query_one("#chat-log", ChatLog)
            chat.add_info(f"[bold #f7768e]Auth preflight failed: {e}[/]")
            return None, required_resources

        for req in coerce_auth_requirements(required_resources):
            selector = (
                str(getattr(req, "resource_id", "")).strip()
                or str(getattr(req, "resource_ref", "")).strip()
                or str(getattr(req, "provider", "")).strip()
            )
            profile_for_selector = getattr(
                auth_context,
                "profile_for_selector",
                None,
            )
            profile = None
            if callable(profile_for_selector):
                profile = profile_for_selector(selector)
            if profile is None:
                profile = auth_context.profile_for_provider(req.provider)
            if profile is None:
                continue
            if selector:
                overrides[selector] = profile.profile_id
        return overrides, required_resources
