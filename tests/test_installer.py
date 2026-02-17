"""Tests for process package installer.

Covers:
- Source resolution (local path, GitHub URL, shorthand)
- Package validation (process.yaml, name format)
- Security review (PackageReview, risk levels, formatting)
- Review callback gate (approve, reject, cancellation)
- Dependency installation (pip/uv)
- Package copying (process.yaml, tools/, README)
- Uninstall (directory package, single YAML, built-in protection)
- Error handling (missing files, bad YAML, clone failures)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from loom.processes.installer import (
    InstallError,
    PackageReview,
    UninstallError,
    _clone_repo,
    _copy_package,
    _install_dependencies,
    _is_github_shorthand,
    _resolve_source,
    _validate_package,
    format_review_for_terminal,
    install_process,
    review_package,
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

YAML_WITH_DEPS_AND_TOOLS = """\
name: full-process
version: '2.0'
description: Full process with deps and tools
author: Test Author
dependencies:
  - requests>=2.28.0
  - pandas
  - numpy
"""

YAML_BAD_NAME = "name: Invalid Name!\nversion: '1.0'\n"

YAML_MISSING_NAME = "version: '1.0'\ndescription: No name\n"


def _make_package(tmp_path: Path, yaml_content: str = MINIMAL_YAML) -> Path:
    """Create a minimal process package directory."""
    pkg = tmp_path / "my-process"
    pkg.mkdir()
    (pkg / "process.yaml").write_text(yaml_content)
    return pkg


def _make_package_with_tools(tmp_path: Path, yaml_content: str = MINIMAL_YAML) -> Path:
    """Create a process package with tools/ and README."""
    pkg = _make_package(tmp_path, yaml_content)
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
# PackageReview
# ===================================================================


class TestPackageReview:
    """Tests for PackageReview dataclass and risk classification."""

    def test_risk_level_low_yaml_only(self):
        review = PackageReview(
            name="simple", version="1.0", description="",
            author="", source="./simple",
        )
        assert review.risk_level == "LOW"
        assert not review.has_dependencies
        assert not review.has_bundled_code

    def test_risk_level_medium_deps_only(self):
        review = PackageReview(
            name="deps", version="1.0", description="",
            author="", source="./deps",
            dependencies=["requests>=2.28.0"],
        )
        assert review.risk_level == "MEDIUM"
        assert review.has_dependencies
        assert not review.has_bundled_code

    def test_risk_level_medium_code_only(self):
        review = PackageReview(
            name="code", version="1.0", description="",
            author="", source="./code",
            bundled_tool_files=["my_tool.py"],
        )
        assert review.risk_level == "MEDIUM"
        assert not review.has_dependencies
        assert review.has_bundled_code

    def test_risk_level_high_deps_and_code(self):
        review = PackageReview(
            name="full", version="1.0", description="",
            author="", source="./full",
            dependencies=["requests"],
            bundled_tool_files=["my_tool.py"],
        )
        assert review.risk_level == "HIGH"
        assert review.has_dependencies
        assert review.has_bundled_code


# ===================================================================
# review_package
# ===================================================================


class TestReviewPackage:
    """Tests for review_package() — pre-install security analysis."""

    def test_review_minimal_yaml_only(self, tmp_path):
        pkg = _make_package(tmp_path)
        target = tmp_path / "target"
        target.mkdir()
        review = review_package(pkg, "./my-process", target)
        assert review.name == "test-process"
        assert review.version == "1.0"
        assert review.dependencies == []
        assert review.bundled_tool_files == []
        assert review.risk_level == "LOW"
        assert not review.will_overwrite

    def test_review_with_deps(self, tmp_path):
        pkg = _make_package(tmp_path, YAML_WITH_DEPS)
        target = tmp_path / "target"
        target.mkdir()
        review = review_package(pkg, "user/repo", target)
        assert review.name == "dep-process"
        assert review.dependencies == ["requests>=2.28.0", "pandas>=2.0"]
        assert review.unpinned_deps == []  # both have version specifiers
        assert review.risk_level == "MEDIUM"

    def test_review_detects_unpinned_deps(self, tmp_path):
        pkg = _make_package(tmp_path, YAML_WITH_DEPS_AND_TOOLS)
        target = tmp_path / "target"
        target.mkdir()
        review = review_package(pkg, "user/repo", target)
        # "pandas" and "numpy" have no version specifier
        assert "pandas" in review.unpinned_deps
        assert "numpy" in review.unpinned_deps
        assert "requests>=2.28.0" not in review.unpinned_deps

    def test_review_with_tools(self, tmp_path):
        pkg = _make_package_with_tools(tmp_path)
        target = tmp_path / "target"
        target.mkdir()
        review = review_package(pkg, "./pkg", target)
        assert review.bundled_tool_files == ["my_tool.py"]
        # _private.py should be excluded (starts with _)
        assert "_private.py" not in review.bundled_tool_files

    def test_review_with_deps_and_tools(self, tmp_path):
        pkg = _make_package_with_tools(tmp_path, YAML_WITH_DEPS_AND_TOOLS)
        target = tmp_path / "target"
        target.mkdir()
        review = review_package(pkg, "user/repo", target)
        assert review.risk_level == "HIGH"
        assert review.has_dependencies
        assert review.has_bundled_code

    def test_review_detects_overwrite(self, tmp_path):
        pkg = _make_package(tmp_path)
        target = tmp_path / "target"
        target.mkdir()
        # Pre-create the install destination
        (target / "test-process").mkdir()
        review = review_package(pkg, "./pkg", target)
        assert review.will_overwrite is True

    def test_review_preserves_source_string(self, tmp_path):
        pkg = _make_package(tmp_path)
        target = tmp_path / "target"
        target.mkdir()
        review = review_package(pkg, "https://github.com/acme/loom-foo", target)
        assert review.source == "https://github.com/acme/loom-foo"


# ===================================================================
# format_review_for_terminal
# ===================================================================


class TestFormatReviewForTerminal:
    """Tests for the terminal display formatter."""

    def test_low_risk_output(self):
        review = PackageReview(
            name="simple", version="1.0",
            description="A simple process", author="Test",
            source="./simple", target_dir="/tmp/target",
        )
        output = format_review_for_terminal(review)
        assert "PACKAGE REVIEW" in output
        assert "simple v1.0" in output
        assert "DEPENDENCIES: None" in output
        assert "BUNDLED CODE: None" in output
        assert "Risk level: LOW" in output
        assert "YAML-only" in output

    def test_medium_risk_deps(self):
        review = PackageReview(
            name="deps", version="1.0",
            description="With deps", author="",
            source="user/repo", target_dir="/tmp/target",
            dependencies=["requests>=2.28.0", "pandas"],
            unpinned_deps=["pandas"],
        )
        output = format_review_for_terminal(review)
        assert "DEPENDENCIES (2 packages)" in output
        assert "requests>=2.28.0" in output
        assert "[!]" in output  # unpinned warning marker
        assert "WARNING" in output
        assert "no version" in output
        assert "Risk level: MEDIUM" in output

    def test_high_risk_deps_and_code(self):
        review = PackageReview(
            name="full", version="1.0",
            description="Full", author="",
            source="user/repo", target_dir="/tmp/target",
            dependencies=["requests"],
            bundled_tool_files=["ga_connect.py", "ga_query.py"],
        )
        output = format_review_for_terminal(review)
        assert "BUNDLED CODE (2 Python files)" in output
        assert "tools/ga_connect.py" in output
        assert "tools/ga_query.py" in output
        assert "automatically" in output  # auto-executed warning
        assert "Risk level: HIGH" in output
        assert "trusted sources" in output

    def test_overwrite_warning(self):
        review = PackageReview(
            name="existing", version="2.0",
            description="", author="",
            source="./pkg", target_dir="/tmp/target",
            will_overwrite=True,
        )
        output = format_review_for_terminal(review)
        assert "OVERWRITE" in output


# ===================================================================
# _install_dependencies
# ===================================================================


class TestInstallDependencies:
    @patch("subprocess.run")
    def test_no_deps_does_nothing(self, mock_run, tmp_path):
        pkg = _make_package(tmp_path)
        _install_dependencies(pkg, "python3")
        mock_run.assert_not_called()

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

    @patch("subprocess.run")
    def test_non_list_deps_ignored(self, mock_run, tmp_path):
        """dependencies: 'not-a-list' should be silently ignored."""
        pkg = _make_package(
            tmp_path,
            "name: test-process\nversion: '1.0'\ndependencies: not-a-list\n",
        )
        _install_dependencies(pkg, "python3")
        mock_run.assert_not_called()

    @patch("subprocess.run")
    def test_empty_string_deps_ignored(self, mock_run, tmp_path):
        """Empty strings in deps list should be filtered out."""
        yaml_content = (
            "name: test-process\nversion: '1.0'\n"
            "dependencies:\n  - ''\n  - '  '\n"
        )
        pkg = _make_package(tmp_path, yaml_content)
        _install_dependencies(pkg, "python3")
        mock_run.assert_not_called()


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

    def test_copies_additional_package_assets(self, tmp_path):
        """Installer should preserve non-tool package assets."""
        src = _make_package_with_tools(tmp_path)
        templates = src / "templates"
        templates.mkdir()
        (templates / "planner-snippet.txt").write_text("example")
        examples = src / "examples"
        examples.mkdir()
        (examples / "sample.yaml").write_text("name: sample\n")
        (src / ".gitignore").write_text("*.tmp\n")

        dest = tmp_path / "dest"
        _copy_package(src, dest)

        assert (dest / "templates" / "planner-snippet.txt").exists()
        assert (dest / "examples" / "sample.yaml").exists()
        assert (dest / ".gitignore").exists()

    def test_skips_vcs_cache_and_bytecode_artifacts(self, tmp_path):
        """Unsafe/noisy artifacts should be excluded from copied package."""
        src = _make_package_with_tools(tmp_path)
        git_dir = src / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git-config")
        pycache = src / "__pycache__"
        pycache.mkdir()
        (pycache / "tool.cpython-312.pyc").write_bytes(b"abc")
        (src / "artifact.pyc").write_bytes(b"compiled")
        (src / ".DS_Store").write_text("noise")

        dest = tmp_path / "dest"
        _copy_package(src, dest)

        assert not (dest / ".git").exists()
        assert not (dest / "__pycache__").exists()
        assert not (dest / "artifact.pyc").exists()
        assert not (dest / ".DS_Store").exists()
        # Core manifest still copied.
        assert (dest / "process.yaml").exists()


# ===================================================================
# install_process — review callback gate
# ===================================================================


class TestInstallProcessReviewGate:
    """Tests for the review_callback security gate in install_process."""

    def test_review_callback_approved_proceeds(self, tmp_path):
        """When callback returns True, installation proceeds."""
        src = _make_package(tmp_path)
        target = tmp_path / "installed"
        reviews_seen: list[PackageReview] = []

        def approve(review: PackageReview) -> bool:
            reviews_seen.append(review)
            return True

        dest = install_process(
            str(src), target_dir=target, skip_deps=True,
            review_callback=approve,
        )
        assert dest.exists()
        assert len(reviews_seen) == 1
        assert reviews_seen[0].name == "test-process"

    def test_review_callback_rejected_aborts(self, tmp_path):
        """When callback returns False, installation is cancelled."""
        src = _make_package(tmp_path)
        target = tmp_path / "installed"

        def reject(review: PackageReview) -> bool:
            return False

        with pytest.raises(InstallError, match="cancelled by user"):
            install_process(
                str(src), target_dir=target, skip_deps=True,
                review_callback=reject,
            )
        # Nothing should have been installed
        assert not (target / "test-process").exists()

    def test_no_callback_proceeds_without_review(self, tmp_path):
        """When review_callback is None, installation proceeds (for tests)."""
        src = _make_package(tmp_path)
        target = tmp_path / "installed"
        dest = install_process(
            str(src), target_dir=target, skip_deps=True,
            review_callback=None,
        )
        assert dest.exists()

    def test_review_callback_sees_deps(self, tmp_path):
        """Callback receives correct dependency information."""
        src = _make_package(tmp_path, YAML_WITH_DEPS)
        target = tmp_path / "installed"
        reviews_seen: list[PackageReview] = []

        def approve(review: PackageReview) -> bool:
            reviews_seen.append(review)
            return True

        install_process(
            str(src), target_dir=target, skip_deps=True,
            review_callback=approve,
        )
        assert reviews_seen[0].dependencies == ["requests>=2.28.0", "pandas>=2.0"]

    def test_review_callback_sees_tools(self, tmp_path):
        """Callback receives bundled tool file information."""
        src = _make_package_with_tools(tmp_path)
        target = tmp_path / "installed"
        reviews_seen: list[PackageReview] = []

        def approve(review: PackageReview) -> bool:
            reviews_seen.append(review)
            return True

        install_process(
            str(src), target_dir=target, skip_deps=True,
            review_callback=approve,
        )
        assert reviews_seen[0].bundled_tool_files == ["my_tool.py"]

    @patch("loom.processes.installer._install_dependencies")
    def test_deps_not_installed_before_review(self, mock_deps, tmp_path):
        """Dependencies must NOT be installed before review approval."""
        src = _make_package(tmp_path, YAML_WITH_DEPS)
        target = tmp_path / "installed"
        call_order: list[str] = []

        def reject(review: PackageReview) -> bool:
            call_order.append("review")
            return False

        mock_deps.side_effect = lambda *a, **kw: call_order.append("deps")

        with pytest.raises(InstallError, match="cancelled"):
            install_process(
                str(src), target_dir=target,
                review_callback=reject,
            )
        # Review was called, but deps were never touched
        assert call_order == ["review"]
        mock_deps.assert_not_called()


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

    @patch("subprocess.run")
    def test_install_with_isolated_deps(self, mock_run, tmp_path):
        src = _make_package(tmp_path, YAML_WITH_DEPS)
        target = tmp_path / "installed"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        dest = install_process(
            str(src),
            target_dir=target,
            isolated_deps=True,
            skip_deps=False,
        )

        assert dest == target / "dep-process"
        assert (dest / "process.yaml").exists()
        assert mock_run.call_count == 2
        create_cmd = mock_run.call_args_list[0][0][0]
        pip_cmd = mock_run.call_args_list[1][0][0]
        assert create_cmd[:3] == ["python3", "-m", "venv"]
        assert str(target / ".deps" / "dep-process") in create_cmd
        expected_python = target / ".deps" / "dep-process"
        if sys.platform.startswith("win"):
            expected_python = expected_python / "Scripts" / "python.exe"
        else:
            expected_python = expected_python / "bin" / "python"
        assert pip_cmd[:3] == [
            str(expected_python),
            "-m",
            "pip",
        ]
        assert "requests>=2.28.0" in pip_cmd

    @patch("loom.processes.installer._install_dependencies_isolated")
    @patch("loom.processes.installer._install_dependencies")
    def test_skip_deps_disables_isolated_dep_install(
        self,
        mock_install,
        mock_isolated,
        tmp_path,
    ):
        src = _make_package(tmp_path, YAML_WITH_DEPS)
        target = tmp_path / "installed"
        install_process(
            str(src),
            target_dir=target,
            skip_deps=True,
            isolated_deps=True,
        )
        mock_install.assert_not_called()
        mock_isolated.assert_not_called()


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

    def test_uninstall_removes_isolated_dependency_env(self, tmp_path):
        proc_dir = tmp_path / "processes"
        proc_dir.mkdir()
        pkg = proc_dir / "test-process"
        pkg.mkdir()
        (pkg / "process.yaml").write_text(MINIMAL_YAML)

        deps_dir = proc_dir / ".deps" / "test-process"
        deps_dir.mkdir(parents=True)
        (deps_dir / "marker.txt").write_text("x")

        removed = uninstall_process("test-process", search_dirs=[proc_dir])
        assert removed == pkg
        assert not pkg.exists()
        assert not deps_dir.exists()


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
