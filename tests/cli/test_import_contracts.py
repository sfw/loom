from __future__ import annotations

import importlib

import click

from loom.cli import (
    PersistenceInitError,
    _cancel_task,
    _check_status,
    _init_persistence,
    cli,
    main,
)


def test_cli_public_exports() -> None:
    assert isinstance(cli, click.core.Group)
    assert callable(main)
    assert PersistenceInitError.__name__ == "PersistenceInitError"
    assert callable(_init_persistence)
    assert callable(_check_status)
    assert callable(_cancel_task)


def test_cli_root_command_registration_order() -> None:
    assert list(cli.commands.keys()) == [
        "cowork",
        "serve",
        "run",
        "status",
        "cancel",
        "models",
        "mcp-serve",
        "auth",
        "mcp",
        "processes",
        "process",
        "install",
        "uninstall",
        "db",
        "learned",
        "reset-learning",
        "setup",
    ]


def test_cli_command_module_import_smoke() -> None:
    module_names = [
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
    for module_name in module_names:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name
