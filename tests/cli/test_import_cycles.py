from __future__ import annotations

import importlib
from pathlib import Path

MODULE_NAMES = [
    "loom.cli.context",
    "loom.cli.persistence",
    "loom.cli.http_tasks",
    "loom.cli.commands.root",
    "loom.cli.commands.auth",
    "loom.cli.commands.auth_profile",
    "loom.cli.commands.mcp",
    "loom.cli.commands.mcp_auth",
    "loom.cli.commands.process",
    "loom.cli.commands.db",
    "loom.cli.commands.maintenance",
]


def test_cli_modules_import_cleanly() -> None:
    for module_name in MODULE_NAMES:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_internal_modules_do_not_import_via_facade() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "loom" / "cli"
    internal_files = [
        root / "context.py",
        root / "persistence.py",
        root / "http_tasks.py",
        root / "commands" / "root.py",
        root / "commands" / "auth.py",
        root / "commands" / "auth_profile.py",
        root / "commands" / "mcp.py",
        root / "commands" / "mcp_auth.py",
        root / "commands" / "process.py",
        root / "commands" / "db.py",
        root / "commands" / "maintenance.py",
    ]
    for path in internal_files:
        text = path.read_text(encoding="utf-8")
        assert "from loom.cli import" not in text
        assert "import loom.cli" not in text
