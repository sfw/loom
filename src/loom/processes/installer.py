"""Process package installer.

Handles installing process packages from GitHub repositories, local paths,
or archive URLs. Validates structure, installs Python dependencies, and
places the package into the appropriate discovery directory.

Expected repository/package structure::

    my-process/
    ├── process.yaml          # Required — process definition
    ├── tools/                # Optional — bundled tool modules
    │   └── my_tool.py
    └── README.md             # Optional

The ``dependencies`` list in ``process.yaml`` declares pip packages that
are automatically installed during ``loom install``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InstallError(Exception):
    """Raised when a process package installation fails."""


class UninstallError(Exception):
    """Raised when a process package removal fails."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUILTIN_DIR = Path(__file__).parent / "builtin"

# Files/directories that are copied from the source into the install target
_PACKAGE_FILES = {"process.yaml", "tools", "README.md", "readme.md", "LICENSE"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_process(
    source: str,
    *,
    target_dir: Path | None = None,
    skip_deps: bool = False,
    python_cmd: str = "python3",
) -> Path:
    """Install a process package from *source*.

    Parameters
    ----------
    source:
        One of:
        - A GitHub URL (``https://github.com/user/repo``)
        - A GitHub shorthand (``user/repo``)
        - A local directory path
    target_dir:
        Where to install.  Defaults to ``~/.loom/processes/``.
    skip_deps:
        If True, skip installing Python dependencies.
    python_cmd:
        Python executable for ``pip install``.  Defaults to ``python3``.

    Returns
    -------
    Path
        The installed package directory.
    """
    if target_dir is None:
        target_dir = Path.home() / ".loom" / "processes"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the source to a local directory
    cleanup_tmp = False
    try:
        src_dir, cleanup_tmp = _resolve_source(source)
        name = _validate_package(src_dir)

        # Check not overwriting a built-in
        if (BUILTIN_DIR / f"{name}.yaml").exists():
            raise InstallError(
                f"Cannot overwrite built-in process {name!r}. "
                f"Choose a different name in process.yaml."
            )

        # Install dependencies
        if not skip_deps:
            _install_dependencies(src_dir, python_cmd)

        # Copy to target
        dest = target_dir / name
        if dest.exists():
            logger.info("Replacing existing installation at %s", dest)
            shutil.rmtree(dest)

        _copy_package(src_dir, dest)
        logger.info("Installed process %r to %s", name, dest)
        return dest

    finally:
        if cleanup_tmp and "src_dir" in dir():
            shutil.rmtree(src_dir, ignore_errors=True)


def uninstall_process(
    name: str,
    *,
    search_dirs: list[Path] | None = None,
) -> Path:
    """Remove an installed process package by name.

    Parameters
    ----------
    name:
        The process name (as declared in process.yaml).
    search_dirs:
        Directories to search for the installed package.
        Defaults to ``[~/.loom/processes/]``.

    Returns
    -------
    Path
        The removed directory.
    """
    if search_dirs is None:
        search_dirs = [Path.home() / ".loom" / "processes"]

    # Check it's not a built-in
    if (BUILTIN_DIR / f"{name}.yaml").exists():
        raise UninstallError(
            f"Cannot uninstall built-in process {name!r}."
        )

    for search_dir in search_dirs:
        # Check for directory package
        pkg_dir = search_dir / name
        if pkg_dir.is_dir() and (pkg_dir / "process.yaml").exists():
            shutil.rmtree(pkg_dir)
            logger.info("Removed process %r from %s", name, pkg_dir)
            return pkg_dir

        # Check for single YAML file
        yaml_file = search_dir / f"{name}.yaml"
        if yaml_file.is_file():
            yaml_file.unlink()
            logger.info("Removed process %r (%s)", name, yaml_file)
            return yaml_file

    raise UninstallError(
        f"Process {name!r} not found in: "
        + ", ".join(str(d) for d in search_dirs)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_source(source: str) -> tuple[Path, bool]:
    """Resolve *source* to a local directory.

    Returns ``(directory, should_cleanup)`` where *should_cleanup* is True
    when a temporary clone was created.
    """
    local = Path(source)
    if local.is_dir():
        return local.resolve(), False

    # GitHub shorthand: user/repo
    url = source
    if _is_github_shorthand(source):
        url = f"https://github.com/{source}.git"
    elif source.startswith("https://github.com/") and not source.endswith(".git"):
        url = source.rstrip("/") + ".git"

    if url.startswith("https://") or url.startswith("git@"):
        return _clone_repo(url), True

    # Last resort: maybe it's a path that doesn't exist yet
    raise InstallError(
        f"Cannot resolve source: {source!r}. "
        f"Provide a GitHub URL (https://github.com/user/repo), "
        f"shorthand (user/repo), or a local directory path."
    )


def _is_github_shorthand(s: str) -> bool:
    """Check if *s* looks like ``user/repo`` (no scheme, exactly one slash)."""
    if "/" not in s:
        return False
    if s.startswith(("https://", "http://", "git@", "/", ".")):
        return False
    parts = s.split("/")
    return len(parts) == 2 and all(parts)


def _clone_repo(url: str) -> Path:
    """Clone a git repository to a temporary directory. Returns the clone path."""
    tmp = Path(tempfile.mkdtemp(prefix="loom-install-"))
    clone_dir = tmp / "repo"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise InstallError("git is not installed. Install git to clone repositories.")
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise InstallError(f"git clone failed: {e.stderr.strip()}")
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp, ignore_errors=True)
        raise InstallError("git clone timed out (120s).")

    # If the repo has a single subdirectory with process.yaml, use that
    # (supports repos where the process is in a subdirectory)
    candidates = [clone_dir]
    for child in clone_dir.iterdir():
        if child.is_dir() and (child / "process.yaml").exists():
            candidates.insert(0, child)
            break

    for candidate in candidates:
        if (candidate / "process.yaml").exists():
            return candidate

    # No process.yaml found anywhere
    shutil.rmtree(tmp, ignore_errors=True)
    raise InstallError(
        f"Repository at {url} does not contain a process.yaml file "
        f"at the root or in any immediate subdirectory."
    )


def _validate_package(src_dir: Path) -> str:
    """Validate the package structure and return the process name."""
    manifest = src_dir / "process.yaml"
    if not manifest.exists():
        raise InstallError(f"Missing process.yaml in {src_dir}")

    try:
        with open(manifest) as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise InstallError(f"Invalid YAML in process.yaml: {e}")

    if not raw or not isinstance(raw, dict):
        raise InstallError("process.yaml is empty or invalid.")

    name = raw.get("name", "")
    if not name:
        raise InstallError("process.yaml is missing the 'name' field.")

    # Basic name validation (matches schema.py)
    import re
    if not re.match(r"^[a-z0-9][a-z0-9-]*$", name):
        raise InstallError(
            f"Invalid process name {name!r}: must be lowercase "
            f"alphanumeric with hyphens."
        )

    return name


def _install_dependencies(src_dir: Path, python_cmd: str) -> None:
    """Install Python dependencies declared in process.yaml."""
    manifest = src_dir / "process.yaml"
    with open(manifest) as f:
        raw = yaml.safe_load(f)

    deps = raw.get("dependencies", [])
    if not deps or not isinstance(deps, list):
        return

    # Filter to strings only
    deps = [d for d in deps if isinstance(d, str) and d.strip()]
    if not deps:
        return

    logger.info("Installing %d dependencies: %s", len(deps), ", ".join(deps))

    # Try uv first (faster), fall back to pip
    installed = False
    for pip_cmd in [
        ["uv", "pip", "install"],
        [python_cmd, "-m", "pip", "install"],
    ]:
        try:
            result = subprocess.run(
                [*pip_cmd, *deps],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                installed = True
                break
            logger.debug(
                "%s failed (rc=%d): %s",
                pip_cmd[0], result.returncode, result.stderr[:500],
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            raise InstallError(
                f"Dependency installation timed out (300s). "
                f"Packages: {', '.join(deps)}"
            )

    if not installed:
        raise InstallError(
            f"Failed to install dependencies: {', '.join(deps)}. "
            f"Ensure pip or uv is available."
        )


def _copy_package(src_dir: Path, dest: Path) -> None:
    """Copy the process package from *src_dir* to *dest*."""
    dest.mkdir(parents=True, exist_ok=True)

    # Always copy process.yaml
    shutil.copy2(src_dir / "process.yaml", dest / "process.yaml")

    # Copy tools/ directory if present
    tools_src = src_dir / "tools"
    if tools_src.is_dir():
        tools_dest = dest / "tools"
        if tools_dest.exists():
            shutil.rmtree(tools_dest)
        shutil.copytree(tools_src, tools_dest)

    # Copy optional files
    for name in ("README.md", "readme.md", "LICENSE"):
        src_file = src_dir / name
        if src_file.is_file():
            shutil.copy2(src_file, dest / name)
