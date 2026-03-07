from __future__ import annotations

import click

from loom.cli.commands import root


def test_cli_callback_populates_context_object(tmp_path) -> None:
    cfg_path = tmp_path / "loom.toml"
    cfg_path.write_text("[server]\nport = 9000\n", encoding="utf-8")

    ctx = click.Context(root.cli, info_name="loom")
    ctx.obj = {}
    ctx.invoked_subcommand = "models"

    with ctx.scope(cleanup=False):
        root.cli.callback(  # type: ignore[misc]
            config_path=cfg_path,
            workspace=tmp_path,
            mcp_config_path=None,
            auth_config_path=None,
            model=None,
            resume_session=None,
            allow_ephemeral=False,
        )

    assert "base_config" in ctx.obj
    assert "config" in ctx.obj
    assert ctx.obj["workspace"] == tmp_path.resolve()
    assert ctx.obj["config_path"] == cfg_path
    assert ctx.obj["explicit_mcp_path"] is None
    assert ctx.obj["explicit_auth_path"] is None
    assert ctx.obj["allow_ephemeral"] is False


def test_setup_subcommand_tolerates_config_error(tmp_path) -> None:
    broken_cfg = tmp_path / "loom.toml"
    broken_cfg.write_text("[server\nport = 9000\n", encoding="utf-8")

    ctx = click.Context(root.cli, info_name="loom")
    ctx.obj = {}
    ctx.invoked_subcommand = "setup"

    with ctx.scope(cleanup=False):
        root.cli.callback(  # type: ignore[misc]
            config_path=broken_cfg,
            workspace=tmp_path,
            mcp_config_path=None,
            auth_config_path=None,
            model=None,
            resume_session=None,
            allow_ephemeral=True,
        )

    assert ctx.obj["workspace"] == tmp_path.resolve()
    assert ctx.obj["allow_ephemeral"] is True
