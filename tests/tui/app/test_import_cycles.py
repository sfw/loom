from __future__ import annotations

import importlib
from pathlib import Path


def test_internal_modules_do_not_import_facade() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "loom" / "tui" / "app"
    for path in root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "from loom.tui.app import" not in text, path
        assert "import loom.tui.app" not in text, path


def test_key_submodules_import_without_cycles() -> None:
    modules = [
        "loom.tui.app.core",
        "loom.tui.app.constants",
        "loom.tui.app.models",
        "loom.tui.app.widgets",
        "loom.tui.app.rendering",
        "loom.tui.app.actions",
        "loom.tui.app.command_palette",
        "loom.tui.app.info_models",
        "loom.tui.app.info_views",
        "loom.tui.app.lifecycle",
        "loom.tui.app.manager_tabs",
        "loom.tui.app.model_roles",
        "loom.tui.app.runtime_config",
        "loom.tui.app.state_init",
        "loom.tui.app.tool_binding",
        "loom.tui.app.ui_events",
        "loom.tui.app.workspace_watch",
        "loom.tui.app.files_panel",
        "loom.tui.app.chat.session",
        "loom.tui.app.chat.history",
        "loom.tui.app.chat.approval",
        "loom.tui.app.chat.learned",
        "loom.tui.app.chat.input_submission",
        "loom.tui.app.chat.turns",
        "loom.tui.app.chat.steering",
        "loom.tui.app.chat.delegate_progress",
        "loom.tui.app.process_runs.state",
        "loom.tui.app.process_runs.launch",
        "loom.tui.app.process_runs.auth",
        "loom.tui.app.process_runs.definition",
        "loom.tui.app.process_runs.workspace",
        "loom.tui.app.process_runs.controls",
        "loom.tui.app.process_runs.ui_state",
        "loom.tui.app.process_runs.rendering",
        "loom.tui.app.process_runs.questions",
        "loom.tui.app.process_runs.lifecycle",
        "loom.tui.app.process_runs.events",
        "loom.tui.app.slash.parsing",
        "loom.tui.app.slash.registry",
        "loom.tui.app.slash.hints",
        "loom.tui.app.slash.completion",
        "loom.tui.app.slash.input_history",
        "loom.tui.app.slash.process_catalog",
        "loom.tui.app.slash.tooling",
        "loom.tui.app.slash.handlers",
    ]
    for module in modules:
        assert importlib.import_module(module)
