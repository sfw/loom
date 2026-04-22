from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_build_macos_desktop_dmg_creates_missing_output_directory(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_macos_desktop_dmg.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    app_bundle = tmp_path / "release" / "bundle" / "macos" / "Loom Desktop.app"
    app_bundle.mkdir(parents=True)
    (app_bundle / "Contents.txt").write_text("placeholder", encoding="utf-8")

    output_path = tmp_path / "release" / "bundle" / "dmg" / "Loom Desktop.dmg"
    hdiutil_log = tmp_path / "hdiutil.log"

    _write_executable(
        fake_bin / "hdiutil",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "{hdiutil_log}"
output="${{@: -1}}"
mkdir -p "$(dirname "$output")"
printf 'fake dmg' > "$output"
""",
    )
    _write_executable(
        fake_bin / "open",
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            str(script_path),
            "--app-bundle",
            os.path.relpath(app_bundle, tmp_path),
            "--output",
            os.path.relpath(output_path, tmp_path),
            "--volume-name",
            "Loom Desktop",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == "fake dmg"
    assert output_path.parent.is_dir()
    assert str(output_path) in hdiutil_log.read_text(encoding="utf-8")
