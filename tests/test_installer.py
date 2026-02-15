"""Tests for process package installer.

Covers:
- Source resolution (local path, GitHub URL, shorthand)
- Package validation (process.yaml, name format)
- Dependency installation (pip/uv)
- Package copying (process.yaml, tools/, README)
- Uninstall (directory package, single YAML, built-in protection)
- Error handling (missing files, bad YAML, clone failures)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from loom.processes.installer import (
    InstallError,
    UninstallError,
    _clone_repo,
    _copy_package,
    _install_dependencies,
    _is_github_shorthand,
    _resolve_source,
    _validate_package,
    install_process,
    uninstall_process,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = "name: test-process\nversion: '1.0'\ndescription: A test\n"

YAML_WITH_DEPS = """\
name: dep-process
version: '1.0'
description: Process with dependencies
dependencies:
  - requests>=2.28.0
  - pandas>=2.0
"""

YAML_BAD_NAME = "name: Invalid Name!\nversion: '1.0'\n"

YAML_MISSING_NAME = "version: '1.0'\ndescription: No name\n"


def _make_package(tmp_path: Path, yaml_content: str = MINIMAL_YAML) -> Path:
    """Create a minimal process package directory."""
    pkg = tmp_path / "my-process"
    pkg.mkdir()
    (pkg / "process.yaml").write_text(yaml_content)
    return pkg


def _make_package_with_tools(tmp_path: Path) -> Path:
    """Create a process package with tools/ and README."""
    pkg = _make_package(tmp_path)
    tools = pkg / "tools"
    tools.mkdir()
    (tools / "my_tool.py").write_text("# A tool\nLOADED = True\n")
    (tools / "_private.py").write_text("# Private\n")
    (pkg / "README.md").write_text("# My Process\n")
    (pkg / "LICENSE").write_text("MIT\n")
    return pkg


# ===================================================================
# _is_github_shorthand
# ===================================================================


class TestIsGithubShorthand:
    def test_valid_shorthand(self):
        assert _is_github_shorthand("user/repo") is True
        assert _is_github_shorthand("acme/loom-analytics") is True

    def test_url_not_shorthand(self):
        assert _is_github_shorthand("https://github.com/user/repo") is False
        assert _is_github_shorthand("http://github.com/user/repo") is False
        assert _is_github_shorthand("git@github.com:user/repo") is False

    def test_local_path_not_shorthand(self):
        assert _is_github_shorthand("/home/user/repo") is False
        assert _is_github_shorthand("./my-process") is False

    def test_no_slash_not_shorthand(self):
        assert _is_github_shorthand("just-a-name") is False

    def test_too_many_slashes(self):
        assert _is_github_shorthand("a/b/c") is False

    def test_empty_parts(self):
        assert _is_github_shorthand("/repo") is False
        assert _is_github_shorthand("user/") is False


# ===================================================================
# _resolve_source
# ===================================================================


class TestResolveSource:
    def test_local_directory(self, tmp_path):
        pkg = _make_package(tmp_path)
        resolved, cleanup = _resolve_source(str(pkg))
        assert resolved == pkg.resolve()
        assert cleanup is False

    def test_nonexistent_path_raises(self):
        with pytest.raises(InstallError, match="Cannot resolve source"):
            _resolve_source("/nonexistent/path")

    @patch("loom.processes.installer._clone_repo")
    def test_github_url(self, mock_clone, tmp_path):
        pkg = _make_package(tmp_path)
        mock_clone.return_value = pkg
        resolved, cleanup = _resolve_source("https://github.com/user/repo")
        assert resolved == pkg
        assert cleanup is True
        mock_clone.assert_called_once_with("https://github.com/user/repo.git")

    @patch("loom.processes.installer._clone_repo")
    def test_github_url_with_dot_git(self, mock_clone, tmp_path):
        pkg = _make_package(tmp_path)
        mock_clone.return_value = pkg
        _resolve_source("https://github.com/user/repo.git")
        mock_clone.assert_called_once_with("https://github.com/user/repo.git")

    @patch("loom.processes.installer._clone_repo")
    def test_github_shorthand(self, mock_clone, tmp_path):
        pkg = _make_package(tmp_path)
        mock_clone.return_value = pkg
        resolved, cleanup = _resolve_source("user/repo")
        mock_clone.assert_called_once_with("https://github.com/user/repo.git")
        assert cleanup is True


# ===================================================================
# _validate_package
# ===================================================================


class TestValidatePackage:
    def test_valid_package(self, tmp_path):
        pkg = _make_package(tmp_path)
        name = _validate_package(pkg)
        assert name == "test-process"

    def test_missing_process_yaml(self, tmp_path):
        empty = tmp_path / "empty-pkg"
        empty.mkdir()
        with pytest.raises(InstallError, match="Missing process.yaml"):
            _validate_package(empty)

    def test_invalid_yaml(self, tmp_path):
        pkg = tmp_path / "bad-pkg"
        pkg.mkdir()
        (pkg / "process.yaml").write_text(":\n  [broken")
        with pytest.raises(InstallError, match="Invalid YAML"):
            _validate_package(pkg)

    def test_empty_yaml(self, tmp_path):
        pkg = tmp_path / "empty-yaml"
        pkg.mkdir()
        (pkg / "process.yaml").write_text("")
        with pytest.raises(InstallError, match="empty or invalid"):
            _validate_package(pkg)

    def test_missing_name(self, tmp_path):
        pkg = _make_package(tmp_path, YAML_MISSING_NAME)
        with pytest.raises(InstallError, match="missing the 'name' field"):
            _validate_package(pkg)

    def test_invalid_name_format(self, tmp_path):
        pkg = _make_package(tmp_path, YAML_BAD_NAME)
        with pytest.raises(InstallError, match="Invalid process name"):
            _validate_package(pkg)


# ===================================================================
# _install_dependencies
# ===================================================================


class TestInstallDependencies:
    def test_no_deps_does_nothing(self, tmp_path):
        pkg = _make_package(tmp_path)
        # Should not raise or call subprocess
        _install_dependencies(pkg, "python3")

    @patch("subprocess.run")
    def test_installs_deps_with_uv(self, mock_run, tmp_path):
        pkg = _make_package(tmp_path, YAML_WITH_DEPS)
        mock_run.return_value = MagicMock(returncode=0)
        _install_dependencies(pkg, "python3")
        # Should try uv first
        call_args = mock_run.call_args_list[0]
        assert "uv" in call_args[0][0]
        assert "requests>=2.28.0" in call_args[0][0]
        assert "pandas>=2.0" in call_args[0][0]

    @patch("subprocess.run")
    def test_falls_back_to_pip(self, mock_run, tmp_path):
        pkg = _make_package(tmp_path, YAML_WITH_DEPS)
        # uv fails (not found), pip succeeds
        mock_run.side_effect = [
            FileNotFoundError(),  # uv not found
            MagicMock(returncode=0),  # pip succeeds
        ]
        _install_dependencies(pkg, "python3")
        assert mock_run.call_count == 2
        pip_args = mock_run.call_args_list[1][0][0]
        assert "pip" in pip_args

    @patch("subprocess.run")
    def test_all_installers_fail_raises(self, mock_run, tmp_path):
        pkg = _make_package(tmp_path, YAML_WITH_DEPS)
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(InstallError, match="Failed to install dependencies"):
            _install_dependencies(pkg, "python3")

    @patch("subprocess.run")
    def test_timeout_raises(self, mock_run, tmp_path):
        pkg = _make_package(tmp_path, YAML_WITH_DEPS)
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pip", timeout=300)
        with pytest.raises(InstallError, match="timed out"):
            _install_dependencies(pkg, "python3")

    def test_non_list_deps_ignored(self, tmp_path):
        """dependencies: 'not-a-list' should be silently ignored."""
        pkg = _make_package(
            tmp_path,
            "name: test-process\nversion: '1.0'\ndependencies: not-a-list\n",
        )
        # Should not raise
        _install_dependencies(pkg, "python3")

    def test_empty_string_deps_ignored(self, tmp_path):
        """Empty strings in deps list should be filtered out."""
        yaml_content = (
            "name: test-process\nversion: '1.0'\n"
            "dependencies:\n  - ''\n  - '  '\n"
        )
        pkg = _make_package(tmp_path, yaml_content)
        # Should not raise or call pip (no valid deps)
        _install_dependencies(pkg, "python3")


# ===================================================================
# _copy_package
# ===================================================================


class TestCopyPackage:
    def test_copies_process_yaml(self, tmp_path):
        src = _make_package(tmp_path)
        dest = tmp_path / "dest"
        _copy_package(src, dest)
        assert (dest / "process.yaml").exists()
        assert (dest / "process.yaml").read_text() == MINIMAL_YAML

    def test_copies_tools_directory(self, tmp_path):
        src = _make_package_with_tools(tmp_path)
        dest = tmp_path / "dest"
        _copy_package(src, dest)
        assert (dest / "tools" / "my_tool.py").exists()
        assert (dest / "tools" / "_private.py").exists()

    def test_copies_readme_and_license(self, tmp_path):
        src = _make_package_with_tools(tmp_path)
        dest = tmp_path / "dest"
        _copy_package(src, dest)
        assert (dest / "README.md").exists()
        assert (dest / "LICENSE").exists()

    def test_creates_dest_directory(self, tmp_path):
        src = _make_package(tmp_path)
        dest = tmp_path / "nested" / "dest"
        _copy_package(src, dest)
        assert dest.exists()
        assert (dest / "process.yaml").exists()

    def test_overwrites_existing_tools(self, tmp_path):
        """Tools dir should be replaced, not merged."""
        src = _make_package_with_tools(tmp_path)
        dest = tmp_path / "dest"
        dest.mkdir()
        tools_dest = dest / "tools"
        tools_dest.mkdir()
        (tools_dest / "old_tool.py").write_text("# Old\n")

        _copy_package(src, dest)
        assert not (tools_dest / "old_tool.py").exists()
        assert (tools_dest / "my_tool.py").exists()


# ===================================================================
# install_process (integration)
# ===================================================================


class TestInstallProcess:
    def test_install_from_local_dir(self, tmp_path):
        src = _make_package(tmp_path)
        target = tmp_path / "installed"
        dest = install_process(str(src), target_dir=target, skip_deps=True)
        assert dest == target / "test-process"
        assert (dest / "process.yaml").exists()

    def test_install_with_tools(self, tmp_path):
        src = _make_package_with_tools(tmp_path)
        target = tmp_path / "installed"
        dest = install_process(str(src), target_dir=target, skip_deps=True)
        assert (dest / "tools" / "my_tool.py").exists()
        assert (dest / "README.md").exists()

    def test_install_replaces_existing(self, tmp_path):
        src = _make_package(tmp_path)
        target = tmp_path / "installed"

        # First install
        install_process(str(src), target_dir=target, skip_deps=True)

        # Modify source
        yaml_v2 = "name: test-process\nversion: '2.0'\ndescription: Updated\n"
        (src / "process.yaml").write_text(yaml_v2)

        # Reinstall
        dest = install_process(str(src), target_dir=target, skip_deps=True)
        content = (dest / "process.yaml").read_text()
        assert "2.0" in content

    def test_install_creates_target_dir(self, tmp_path):
        src = _make_package(tmp_path)
        target = tmp_path / "deep" / "nested" / "dir"
        dest = install_process(str(src), target_dir=target, skip_deps=True)
        assert dest.exists()

    @patch("loom.processes.installer._install_dependencies")
    def test_install_calls_dependency_installer(self, mock_deps, tmp_path):
        src = _make_package(tmp_path, YAML_WITH_DEPS)
        target = tmp_path / "installed"
        install_process(str(src), target_dir=target)
        mock_deps.assert_called_once()

    @patch("loom.processes.installer._install_dependencies")
    def test_install_skip_deps(self, mock_deps, tmp_path):
        src = _make_package(tmp_path, YAML_WITH_DEPS)
        target = tmp_path / "installed"
        install_process(str(src), target_dir=target, skip_deps=True)
        mock_deps.assert_not_called()

    def test_install_rejects_builtin_name(self, tmp_path):
        """Cannot install a process that shadows a built-in."""
        # research-report is a built-in
        yaml_content = "name: research-report\nversion: '1.0'\n"
        src = _make_package(tmp_path, yaml_content)
        target = tmp_path / "installed"
        with pytest.raises(InstallError, match="built-in"):
            install_process(str(src), target_dir=target, skip_deps=True)

    def test_install_bad_source(self, tmp_path):
        with pytest.raises(InstallError, match="Cannot resolve source"):
            install_process("/nonexistent/path", target_dir=tmp_path)


# ===================================================================
# uninstall_process
# ===================================================================


class TestUninstallProcess:
    def test_uninstall_directory_package(self, tmp_path):
        # Set up an installed package
        proc_dir = tmp_path / "processes"
        proc_dir.mkdir()
        pkg = proc_dir / "test-process"
        pkg.mkdir()
        (pkg / "process.yaml").write_text(MINIMAL_YAML)

        removed = uninstall_process("test-process", search_dirs=[proc_dir])
        assert removed == pkg
        assert not pkg.exists()

    def test_uninstall_yaml_file(self, tmp_path):
        proc_dir = tmp_path / "processes"
        proc_dir.mkdir()
        yaml_file = proc_dir / "test-process.yaml"
        yaml_file.write_text(MINIMAL_YAML)

        removed = uninstall_process("test-process", search_dirs=[proc_dir])
        assert removed == yaml_file
        assert not yaml_file.exists()

    def test_uninstall_not_found_raises(self, tmp_path):
        proc_dir = tmp_path / "processes"
        proc_dir.mkdir()
        with pytest.raises(UninstallError, match="not found"):
            uninstall_process("nonexistent", search_dirs=[proc_dir])

    def test_uninstall_builtin_raises(self):
        with pytest.raises(UninstallError, match="built-in"):
            uninstall_process("research-report")

    def test_uninstall_searches_multiple_dirs(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        pkg = dir2 / "test-process"
        pkg.mkdir()
        (pkg / "process.yaml").write_text(MINIMAL_YAML)

        removed = uninstall_process("test-process", search_dirs=[dir1, dir2])
        assert removed == pkg
        assert not pkg.exists()


# ===================================================================
# _clone_repo (mocked)
# ===================================================================


class TestCloneRepo:
    @patch("subprocess.run")
    def test_git_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(InstallError, match="git is not installed"):
            _clone_repo("https://github.com/user/repo.git")

    @patch("subprocess.run")
    def test_clone_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            128, "git", stderr="fatal: repo not found",
        )
        with pytest.raises(InstallError, match="git clone failed"):
            _clone_repo("https://github.com/user/repo.git")

    @patch("subprocess.run")
    def test_clone_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=120)
        with pytest.raises(InstallError, match="timed out"):
            _clone_repo("https://github.com/user/repo.git")


# ===================================================================
# Schema: dependencies field
# ===================================================================


class TestDependenciesField:
    """Test that ProcessDefinition.dependencies is properly parsed from YAML."""

    def test_dependencies_parsed(self, tmp_path):
        from loom.processes.schema import ProcessLoader

        f = tmp_path / "dep-proc.yaml"
        f.write_text(YAML_WITH_DEPS)
        loader = ProcessLoader()
        defn = loader.load(str(f))
        assert defn.dependencies == ["requests>=2.28.0", "pandas>=2.0"]

    def test_no_dependencies_defaults_empty(self, tmp_path):
        from loom.processes.schema import ProcessLoader

        f = tmp_path / "minimal.yaml"
        f.write_text(MINIMAL_YAML)
        loader = ProcessLoader()
        defn = loader.load(str(f))
        assert defn.dependencies == []

    def test_dependencies_non_list_ignored(self, tmp_path):
        from loom.processes.schema import ProcessLoader

        f = tmp_path / "bad-deps.yaml"
        f.write_text(
            "name: test-process\nversion: '1.0'\ndependencies: not-a-list\n"
        )
        loader = ProcessLoader()
        defn = loader.load(str(f))
        assert defn.dependencies == []

    def test_default_dependencies_empty(self):
        from loom.processes.schema import ProcessDefinition

        defn = ProcessDefinition(name="test")
        assert defn.dependencies == []
