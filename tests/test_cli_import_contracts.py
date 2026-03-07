"""Compatibility checks for `loom.__main__` CLI exports."""

from __future__ import annotations

import loom.__main__ as main_mod


def test_main_module_reexports_required_cli_contracts() -> None:
    assert callable(main_mod.cli)
    assert callable(main_mod.main)
    assert callable(main_mod._init_persistence)
    assert callable(main_mod._check_status)
    assert callable(main_mod._cancel_task)
    assert main_mod.PersistenceInitError.__name__ == "PersistenceInitError"


def test_main_cli_reexport_points_to_cli_package() -> None:
    from loom.cli import cli as package_cli

    assert main_mod.cli is package_cli
