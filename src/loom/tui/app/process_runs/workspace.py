"""Process-run workspace selection and provisioning helpers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from loom.tui.screens import ProcessRunWorkspaceScreen
from loom.tui.widgets import ChatLog

from . import state as process_run_state

logger = logging.getLogger(__name__)


def build_process_run_context(self, goal: str, *, workspace: Path) -> dict:
    """Build compact cowork context to pass into delegated process runs."""
    if self._session is None:
        return {
            "requested_goal": goal,
            "workspace": str(workspace),
            "source_workspace_root": str(self._workspace.resolve()),
        }
    state = self._session.session_state
    context: dict = {
        "requested_goal": goal,
        "workspace": str(workspace),
        "source_workspace_root": str(self._workspace.resolve()),
        "cowork": {
            "turn_count": state.turn_count,
            "current_focus": state.current_focus,
            "key_decisions": state.key_decisions[-8:],
        },
    }
    recent_messages: list[dict[str, str]] = []
    for message in reversed(self._session.messages):
        role = str(message.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = self._one_line(message.get("content", ""), 500)
        if not content:
            continue
        recent_messages.append({"role": role, "content": content})
        if len(recent_messages) >= 6:
            break
    if recent_messages:
        recent_messages.reverse()
        context["cowork"]["recent_messages"] = recent_messages
    return context


async def llm_process_run_folder_name(self, process_name: str, goal: str) -> str:
    process_cfg = getattr(self._config, "process", None)
    if not bool(getattr(process_cfg, "llm_run_folder_naming_enabled", True)):
        return ""
    if not self._has_configured_role_model("extractor"):
        return ""
    goal_seed = self._run_goal_for_folder_name(goal)
    prompt = (
        "Return exactly one kebab-case folder name for this process run.\n"
        "Requirements:\n"
        "- Output ONLY the slug (single line, no quotes, no backticks, no explanation).\n"
        "- 2-6 words, lowercase letters/numbers/hyphens only.\n"
        "- Use concrete topic words from the goal.\n"
        "- Do NOT echo prompt scaffolding or meta wording such as "
        "\"the user wants\", \"i need you to\", \"folder\", \"name\", or \"for a pr\".\n"
        f"Process label: {process_name}\n"
        f"Goal: {goal_seed or goal}\n"
        "Slug:"
    )
    try:
        response, _, _, _ = await self._invoke_helper_role_completion(
            role="extractor",
            tier=1,
            prompt=prompt,
            max_tokens=20,
            temperature=0.2,
        )
    except Exception as e:
        logger.warning("LLM run-folder naming failed: %s", e)
        return ""
    text = str(getattr(response, "text", "") or "")
    slug = self._extract_run_folder_slug(text)
    if self._is_low_quality_run_folder_slug(slug):
        logger.debug(
            "Discarding low-quality LLM run-folder name '%s' for goal '%s'",
            slug,
            goal_seed or goal,
        )
        return ""
    return slug


async def prepare_process_run_workspace(
    self,
    process_name: str,
    goal: str,
) -> Path:
    process_cfg = getattr(self._config, "process", None)
    if not bool(getattr(process_cfg, "tui_run_scoped_workspace_enabled", True)):
        return self._workspace

    root = self._workspace.resolve()
    slug = await self._llm_process_run_folder_name(process_name, goal)
    if not slug:
        slug = self._fallback_process_run_folder_name(process_name, goal)

    for suffix in range(1, 1000):
        candidate_name = slug if suffix == 1 else f"{slug}-{suffix}"
        candidate = root / candidate_name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
        except OSError as e:
            logger.warning("Failed to create run workspace %s: %s", candidate, e)
            break

    return self._workspace


def next_available_process_run_folder_name(self, base_slug: str) -> str:
    """Return the first non-existing run folder name for a base slug."""
    root = self._workspace.resolve()
    slug = self._slugify_process_run_folder(base_slug) or "process-run"
    for suffix in range(1, 1000):
        candidate_name = slug if suffix == 1 else f"{slug}-{suffix}"
        candidate = root / candidate_name
        if not candidate.exists():
            return candidate_name
    return slug


def materialize_process_run_workspace_selection(self, relative_path: str) -> Path:
    """Create or reuse selected run workspace path under the active workspace root."""
    root = self._workspace.resolve()
    clean = str(relative_path or "").strip()
    if not clean:
        return root

    candidate = (root / clean).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError("Folder must stay inside workspace root.") from e

    if candidate.exists():
        if candidate.is_dir():
            return candidate
        raise ValueError("Selected folder path exists but is not a directory.")

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


async def prompt_process_run_workspace_choice(
    self,
    *,
    run_id: str = "",
    process_name: str,
    suggested_folder: str,
) -> str | None:
    """Prompt for run working folder choice.

    Returns:
    - ``None`` when user cancels
    - ``""`` when user selects workspace root
    - ``<relative folder>`` when user selects folder mode
    """
    done = asyncio.Event()
    selected: list[str | None] = []

    def _handle(value: str | None) -> None:
        if value is None:
            selected.append(None)
        else:
            selected.append(str(value))
        done.set()

    run = self._process_runs.get(run_id) if run_id else None
    if run is not None and not run.closed:
        process_run_state.begin_process_run_user_input_pause(run)
    self.push_screen(
        ProcessRunWorkspaceScreen(
            process_name=process_name,
            workspace_root=str(self._workspace.resolve()),
            suggested_folder=suggested_folder,
        ),
        callback=_handle,
    )
    try:
        await done.wait()
    finally:
        if run is not None:
            process_run_state.end_process_run_user_input_pause(run)
    return selected[0] if selected else None


async def choose_process_run_workspace(
    self,
    run_id: str,
    process_name: str,
    goal: str,
) -> Path | None:
    """Resolve run workspace via provisioning modal selection."""
    process_cfg = getattr(self._config, "process", None)
    if not bool(getattr(process_cfg, "tui_run_scoped_workspace_enabled", True)):
        return self._workspace

    suggested = await self._llm_process_run_folder_name(process_name, goal)
    if not suggested:
        suggested = self._fallback_process_run_folder_name(process_name, goal)
    suggested = self._next_available_process_run_folder_name(suggested)

    chat = self.query_one("#chat-log", ChatLog)
    while True:
        selection = await self._prompt_process_run_workspace_choice(
            run_id=run_id,
            process_name=process_name,
            suggested_folder=suggested,
        )
        if selection is None:
            return None

        try:
            normalized = self._normalize_process_run_workspace_selection(selection)
            return self._materialize_process_run_workspace_selection(normalized)
        except ValueError as e:
            chat.add_info(
                f"[bold #f7768e]Invalid working folder:[/] {self._one_line(str(e), 220)}",
            )
            suggested = str(selection or "").strip() or suggested
            continue
        except OSError as e:
            chat.add_info(
                "[bold #f7768e]Failed to prepare working folder:[/] "
                f"{self._one_line(str(e), 220)}",
            )
            suggested = str(selection or "").strip() or suggested
            continue
