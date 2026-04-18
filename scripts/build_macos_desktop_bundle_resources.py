#!/usr/bin/env python3
"""Assemble bundled Python resources for the macOS desktop app."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

DEFAULT_PYTHON_VERSION = "3.11.14"
EXPECTED_UV_VERSION = "0.10.12"
MANIFEST_NAME = "loom-desktop-bundle.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the macOS desktop app's bundled Python runtime resources."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will become the Tauri bundle resource payload.",
    )
    parser.add_argument(
        "--python-version",
        default=DEFAULT_PYTHON_VERSION,
        help="Managed CPython version to bundle (default: %(default)s).",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=None,
        help="Use an explicit Python executable instead of resolving a uv-managed build.",
    )
    parser.add_argument(
        "--uv-cache-dir",
        type=Path,
        default=None,
        help="Workspace-local uv cache directory to use while assembling the bundle.",
    )
    parser.add_argument(
        "--allow-non-macos",
        action="store_true",
        help="Skip the platform guard. Intended only for tests and dry runs.",
    )
    return parser.parse_args()


def run(
    args: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        text=True,
        check=False,
        capture_output=capture_output,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip() if capture_output else ""
        suffix = f": {details}" if details else ""
        raise SystemExit(f"command failed ({completed.returncode}): {' '.join(args)}{suffix}")
    return completed


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_repo_root(script_path: Path) -> Path:
    return script_path.resolve().parent.parent


def read_loom_version(repo_root: Path) -> str:
    pyproject_path = repo_root / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    version = data.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        raise SystemExit(f"missing [project].version in {pyproject_path}")
    return version.strip()


def read_packaged_extras(repo_root: Path) -> list[str]:
    pyproject_path = repo_root / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    optional = data.get("project", {}).get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise SystemExit(f"invalid [project.optional-dependencies] in {pyproject_path}")
    extras = []
    for name in optional:
        if not isinstance(name, str) or not name.strip():
            continue
        if name.strip() == "dev":
            continue
        extras.append(name.strip())
    return sorted(set(extras))


def uv_env(repo_root: Path, cache_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_NATIVE_TLS", "1")
    env["UV_CACHE_DIR"] = str((cache_dir or repo_root / ".uv-cache").resolve())
    return env


def resolve_uv_version(repo_root: Path, env: dict[str, str]) -> str:
    completed = run(["uv", "--version"], cwd=repo_root, env=env, capture_output=True)
    version = completed.stdout.strip()
    if not version:
        raise SystemExit("failed to resolve uv version")
    return version


def resolve_managed_python(repo_root: Path, env: dict[str, str], version: str) -> Path:
    managed_env = env | {"UV_MANAGED_PYTHON": "1"}
    install = subprocess.run(
        ["uv", "python", "install", version, "--no-progress"],
        cwd=repo_root,
        env=managed_env,
        text=True,
        check=False,
        capture_output=True,
    )
    if install.returncode != 0:
        message = install.stderr.strip() or install.stdout.strip()
        raise SystemExit(
            "failed to install bundled Python runtime "
            f"{version} via uv-managed Python: {message}"
        )

    found = run(
        ["uv", "python", "find", version],
        cwd=repo_root,
        env=managed_env,
        capture_output=True,
    )
    python_path = Path(found.stdout.strip()).resolve()
    if not python_path.exists():
        raise SystemExit(f"uv reported a missing Python executable: {python_path}")
    return python_path


def resolve_python_executable(
    repo_root: Path,
    env: dict[str, str],
    explicit_python: Path | None,
    version: str,
) -> Path:
    if explicit_python is not None:
        python_path = explicit_python.expanduser().resolve()
        if not python_path.exists():
            raise SystemExit(f"explicit Python executable does not exist: {python_path}")
        return python_path
    return resolve_managed_python(repo_root, env, version)


def python_prefix_from_executable(python_executable: Path) -> Path:
    if python_executable.parent.name != "bin":
        raise SystemExit(
            f"expected bundled Python executable under a bin directory: {python_executable}"
        )
    return python_executable.parent.parent.resolve()


def bundled_python_details(prefix: Path, python_executable: Path) -> tuple[str, Path]:
    code = (
        "import json, sysconfig, sys\n"
        "prefix = sys.argv[1]\n"
        "paths = {\n"
        "  'python_version': '.'.join(map(str, sys.version_info[:3])),\n"
        "  'purelib': sysconfig.get_path('purelib', vars={'base': prefix, 'platbase': prefix}),\n"
        "}\n"
        "print(json.dumps(paths))\n"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", code, str(prefix)],
        text=True,
        check=True,
        capture_output=True,
    )
    data = json.loads(completed.stdout)
    return data["python_version"], Path(data["purelib"])


def build_dependency_payload(
    *,
    repo_root: Path,
    env: dict[str, str],
    extras: Sequence[str],
    python_executable: Path,
    env_root: Path,
    work_dir: Path,
) -> None:
    requirements_path = work_dir / "desktop-requirements.txt"
    dist_dir = work_dir / "dist"
    clean_dir(dist_dir)

    export_args = [
        "uv",
        "export",
        "--format",
        "requirements.txt",
        "--locked",
        "--no-dev",
        "--no-emit-project",
        "--python",
        str(python_executable),
        "--output-file",
        str(requirements_path),
    ]
    for extra in extras:
        export_args.extend(["--extra", extra])
    run(export_args, cwd=repo_root, env=env)

    run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(python_executable),
            "--prefix",
            str(env_root),
            "--compile-bytecode",
            str(requirements_path),
        ],
        cwd=repo_root,
        env=env,
    )

    run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=repo_root,
        env=env,
    )

    wheels = sorted(dist_dir.glob("loom-*.whl"))
    if not wheels:
        raise SystemExit(f"uv build did not produce a Loom wheel in {dist_dir}")

    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python_executable),
            "--prefix",
            str(env_root),
            "--compile-bytecode",
            "--no-deps",
            str(wheels[-1]),
        ],
        cwd=repo_root,
        env=env,
    )


def write_manifest(
    *,
    output_dir: Path,
    enabled_extras: Sequence[str],
    python_root: Path,
    python_executable: Path,
    env_root: Path,
    loom_version: str,
    python_version: str,
    python_request: str,
    site_packages_path: Path,
    uv_version: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "enabled_extras": list(enabled_extras),
        "loom_version": loom_version,
        "python_version": python_version,
        "python_request": python_request,
        "python_home_relative_path": str(python_root.relative_to(output_dir)),
        "python_executable_relative_path": str(python_executable.relative_to(output_dir)),
        "environment_root_relative_path": str(env_root.relative_to(output_dir)),
        "site_packages_relative_path": str(site_packages_path.relative_to(output_dir)),
        "entry_module": "loom.daemon.cli",
        "uv_version": uv_version,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path.write_text(f"{manifest_text}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if sys.platform != "darwin" and not args.allow_non_macos:
        raise SystemExit("the bundled desktop runtime builder is supported on macOS only")

    repo_root = resolve_repo_root(Path(__file__))
    output_dir = args.output_dir.expanduser().resolve()
    work_dir = output_dir.parent / ".desktop-python-build"
    env = uv_env(repo_root, args.uv_cache_dir)
    env["UV_PYTHON_INSTALL_DIR"] = str((work_dir / "uv-python").resolve())
    packaged_extras = read_packaged_extras(repo_root)
    loom_version = read_loom_version(repo_root)
    uv_version = resolve_uv_version(repo_root, env)

    clean_dir(output_dir)
    clean_dir(work_dir)

    python_executable = resolve_python_executable(
        repo_root,
        env,
        args.python_executable,
        args.python_version,
    )
    python_runtime_source = python_prefix_from_executable(python_executable)

    python_root = output_dir / "python"
    env_root = output_dir / "loom-env"

    shutil.copytree(python_runtime_source, python_root, symlinks=True, dirs_exist_ok=False)
    build_dependency_payload(
        repo_root=repo_root,
        env=env,
        extras=packaged_extras,
        python_executable=python_executable,
        env_root=env_root,
        work_dir=work_dir,
    )

    bundled_python_executable = python_root / "bin" / python_executable.name
    if not bundled_python_executable.exists():
        raise SystemExit(
            f"bundled Python executable missing after copy: {bundled_python_executable}"
        )

    python_version, site_packages_path = bundled_python_details(env_root, bundled_python_executable)
    if not site_packages_path.exists():
        raise SystemExit(f"bundled site-packages directory missing: {site_packages_path}")

    write_manifest(
        output_dir=output_dir,
        enabled_extras=packaged_extras,
        python_root=python_root,
        python_executable=bundled_python_executable,
        env_root=env_root,
        loom_version=loom_version,
        python_version=python_version,
        python_request=args.python_version,
        site_packages_path=site_packages_path,
        uv_version=uv_version,
    )

    print(f"Bundled desktop Python resources written to {output_dir}")
    print(f"  Loom version: {loom_version}")
    print(f"  Python runtime: {python_root}")
    print(f"  Python request: {args.python_version}")
    print(f"  Enabled extras: {', '.join(packaged_extras) if packaged_extras else '(none)'}")
    print(f"  uv version: {uv_version} (expected {EXPECTED_UV_VERSION})")
    print(f"  Loom environment: {env_root}")
    print(f"  Manifest: {output_dir / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
