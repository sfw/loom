"""Tests for process definition schema, loading, validation, and prompt integration.

Covers:
- ProcessDefinition dataclass methods
- ProcessLoader discovery, loading, and validation
- PromptAssembler integration with process definitions
- ProcessConfig defaults and TOML parsing
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.config import Config, ProcessConfig, load_config
from loom.processes.schema import (
    MemoryType,
    PhaseTemplate,
    PlannerExample,
    ProcessDefinition,
    ProcessLoader,
    ProcessNotFoundError,
    ProcessValidationError,
    ToolRequirements,
    VerificationRule,
)
from loom.prompts.assembler import PromptAssembler
from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStateManager,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
name: test-process
version: '1.0'
description: A test process
persona: You are a test assistant.
"""

FULL_YAML = """\
name: full-process
version: '2.0'
description: A fully-loaded process definition
author: Test Author
tags:
  - test
  - full

persona: You are a domain-expert assistant for testing.

phase_mode: strict

tool_guidance: Always prefer read_file before editing.

tools:
  guidance: Always prefer read_file before editing.
  required:
    - read_file
    - write_file
  excluded:
    - shell_execute

phases:
  - id: research
    description: Research the codebase
    depends_on: []
    model_tier: 1
    verification_tier: 1
    is_critical_path: true
    acceptance_criteria: Codebase is understood
    deliverables:
      - research-notes.md \u2014 Summary of findings
  - id: implement
    description: Implement the changes
    depends_on:
      - research
    model_tier: 2
    verification_tier: 2
    is_critical_path: true
    acceptance_criteria: All changes compile
    deliverables:
      - main.py \u2014 Updated main file
  - id: verify
    description: Verify the implementation
    depends_on:
      - implement
    model_tier: 1
    verification_tier: 2
    is_synthesis: true
    acceptance_criteria: All tests pass
    deliverables:
      - test-report.md \u2014 Test execution report

verification:
  rules:
    - name: no-todos
      description: Ensure no TODO comments remain
      check: "TODO|FIXME|HACK"
      severity: warning
      type: regex
      target: deliverables
    - name: coverage-check
      description: Ensure test coverage is adequate
      check: Verify that test coverage exceeds 80%
      severity: error
      type: llm
      target: output

memory:
  extract_types:
    - type: architecture_decision
      description: Key architectural decisions made during analysis
    - type: risk_factor
      description: Identified risks and mitigations
  extraction_guidance: Focus on decisions that affect multiple phases.

workspace_analysis:
  scan_for:
    - "*.py"
    - "*.ts"
  guidance: Look for test files and configuration.

planner_examples:
  - goal: Refactor module X
    subtasks:
      - id: analyze-x
        description: Analyze module X structure
        depends_on: []
      - id: refactor-x
        description: Refactor module X
        depends_on:
          - analyze-x

replanning:
  triggers: When tests fail unexpectedly
  guidance: Re-examine the test setup before changing implementation.
"""

CYCLE_YAML = """\
name: cycle-process
version: '1.0'
phases:
  - id: a
    description: Phase A
    depends_on:
      - b
  - id: b
    description: Phase B
    depends_on:
      - a
"""

INVALID_NAME_YAML = """\
name: Invalid Name!
version: '1.0'
"""

MISSING_NAME_YAML = """\
version: '1.0'
description: No name
"""

DUPLICATE_PHASE_YAML = """\
name: dup-phase
version: '1.0'
phases:
  - id: alpha
    description: First
  - id: alpha
    description: Second
"""

EMPTY_PHASE_DESC_YAML = """\
name: empty-desc
version: '1.0'
phases:
  - id: alpha
    description: ""
"""

UNKNOWN_DEP_YAML = """\
name: unknown-dep
version: '1.0'
phases:
  - id: alpha
    description: Phase alpha
    depends_on:
      - nonexistent
"""

DUPLICATE_DELIVERABLE_YAML = """\
name: dup-deliv
version: '1.0'
phases:
  - id: alpha
    description: Phase alpha
    deliverables:
      - report.md \u2014 First report
  - id: beta
    description: Phase beta
    deliverables:
      - report.md \u2014 Second report
"""

INVALID_SEVERITY_YAML = """\
name: bad-severity
version: '1.0'
verification:
  rules:
    - name: rule1
      description: Bad severity
      check: something
      severity: critical
"""

INVALID_REGEX_YAML = """\
name: bad-regex
version: '1.0'
verification:
  rules:
    - name: rule1
      description: Bad regex
      check: "[invalid"
      type: regex
"""

INVALID_PHASE_MODE_YAML = """\
name: bad-mode
version: '1.0'
phase_mode: mandatory
"""


@pytest.fixture
def process_dir(tmp_path):
    """Create a temporary directory with process YAML files."""
    d = tmp_path / "processes"
    d.mkdir()
    (d / "test-process.yaml").write_text(MINIMAL_YAML)
    (d / "full-process.yaml").write_text(FULL_YAML)
    return d


@pytest.fixture
def loader(process_dir):
    """ProcessLoader with extra search path pointing at our temp dir."""
    return ProcessLoader(extra_search_paths=[process_dir])


@pytest.fixture
def full_process_defn():
    """Return a fully-populated ProcessDefinition for unit tests."""
    return ProcessDefinition(
        name="full-process",
        version="2.0",
        description="A fully-loaded process definition",
        author="Test Author",
        tags=["test", "full"],
        persona="You are a domain-expert assistant.",
        tool_guidance="Always prefer read_file before editing.",
        tools=ToolRequirements(
            guidance="Always prefer read_file before editing.",
            required=["read_file", "write_file"],
            excluded=["shell_execute"],
        ),
        phase_mode="strict",
        phases=[
            PhaseTemplate(
                id="research",
                description="Research the codebase",
                depends_on=[],
                model_tier=1,
                verification_tier=1,
                is_critical_path=True,
                deliverables=["research-notes.md \u2014 Summary of findings"],
            ),
            PhaseTemplate(
                id="implement",
                description="Implement the changes",
                depends_on=["research"],
                model_tier=2,
                verification_tier=2,
                is_critical_path=True,
                deliverables=["main.py \u2014 Updated main file"],
            ),
            PhaseTemplate(
                id="verify",
                description="Verify the implementation",
                depends_on=["implement"],
                model_tier=1,
                verification_tier=2,
                is_synthesis=True,
                deliverables=["test-report.md \u2014 Test execution report"],
            ),
        ],
        verification_rules=[
            VerificationRule(
                name="no-todos",
                description="Ensure no TODO comments remain",
                check="TODO|FIXME|HACK",
                severity="warning",
                type="regex",
                target="deliverables",
            ),
            VerificationRule(
                name="coverage-check",
                description="Ensure test coverage is adequate",
                check="Verify that test coverage exceeds 80%",
                severity="error",
                type="llm",
                target="output",
            ),
        ],
        memory_types=[
            MemoryType(type="architecture_decision", description="Key architectural decisions"),
            MemoryType(type="risk_factor", description="Identified risks"),
        ],
        extraction_guidance="Focus on decisions that affect multiple phases.",
        planner_examples=[
            PlannerExample(
                goal="Refactor module X",
                subtasks=[
                    {"id": "analyze-x", "description": "Analyze module X", "depends_on": []},
                    {"id": "refactor-x", "description": "Refactor X", "depends_on": ["analyze-x"]},
                ],
            ),
        ],
        replanning_triggers="When tests fail unexpectedly",
        replanning_guidance="Re-examine the test setup before changing implementation.",
    )


@pytest.fixture
def sample_task():
    return Task(
        id="task-1",
        goal="Build a REST API",
        status=TaskStatus.EXECUTING,
        workspace="/tmp/workspace",
        plan=Plan(
            subtasks=[
                Subtask(
                    id="setup",
                    description="Set up project structure",
                    status=SubtaskStatus.COMPLETED,
                    summary="Created project files",
                ),
                Subtask(
                    id="implement-api",
                    description="Implement API endpoints",
                    status=SubtaskStatus.RUNNING,
                    acceptance_criteria="All endpoints return 200",
                ),
                Subtask(
                    id="test-api",
                    description="Write and run tests",
                    status=SubtaskStatus.PENDING,
                    depends_on=["implement-api"],
                ),
            ],
            version=1,
        ),
    )


@pytest.fixture
def state_manager(tmp_path):
    return TaskStateManager(tmp_path)


# ===================================================================
# ProcessDefinition Tests
# ===================================================================


class TestProcessDefinition:
    """Tests for the ProcessDefinition dataclass and its methods."""

    def test_create_with_all_fields(self, full_process_defn):
        defn = full_process_defn
        assert defn.name == "full-process"
        assert defn.version == "2.0"
        assert defn.description == "A fully-loaded process definition"
        assert defn.author == "Test Author"
        assert defn.tags == ["test", "full"]
        assert defn.persona == "You are a domain-expert assistant."
        assert defn.phase_mode == "strict"
        assert len(defn.phases) == 3
        assert len(defn.verification_rules) == 2
        assert len(defn.memory_types) == 2
        assert len(defn.planner_examples) == 1
        assert defn.tools.required == ["read_file", "write_file"]
        assert defn.tools.excluded == ["shell_execute"]

    def test_has_phases_true(self, full_process_defn):
        assert full_process_defn.has_phases() is True

    def test_has_phases_false(self):
        defn = ProcessDefinition(name="empty")
        assert defn.has_phases() is False

    def test_has_verification_rules_true(self, full_process_defn):
        assert full_process_defn.has_verification_rules() is True

    def test_has_verification_rules_false(self):
        defn = ProcessDefinition(name="empty")
        assert defn.has_verification_rules() is False

    def test_regex_rules(self, full_process_defn):
        regex = full_process_defn.regex_rules()
        assert len(regex) == 1
        assert regex[0].name == "no-todos"
        assert regex[0].type == "regex"

    def test_llm_rules(self, full_process_defn):
        llm = full_process_defn.llm_rules()
        assert len(llm) == 1
        assert llm[0].name == "coverage-check"
        assert llm[0].type == "llm"

    def test_regex_rules_empty_when_no_rules(self):
        defn = ProcessDefinition(name="empty")
        assert defn.regex_rules() == []

    def test_llm_rules_empty_when_no_rules(self):
        defn = ProcessDefinition(name="empty")
        assert defn.llm_rules() == []

    def test_get_deliverables(self, full_process_defn):
        deliverables = full_process_defn.get_deliverables()
        assert "research" in deliverables
        assert "implement" in deliverables
        assert "verify" in deliverables
        assert deliverables["research"] == ["research-notes.md"]
        assert deliverables["implement"] == ["main.py"]
        assert deliverables["verify"] == ["test-report.md"]

    def test_get_deliverables_empty_phases(self):
        defn = ProcessDefinition(name="empty")
        assert defn.get_deliverables() == {}

    def test_get_deliverables_no_deliverables_on_phase(self):
        defn = ProcessDefinition(
            name="no-deliv",
            phases=[PhaseTemplate(id="a", description="Phase A")],
        )
        assert defn.get_deliverables() == {}

    def test_get_deliverables_dash_separator(self):
        """Test deliverable parsing with em-dash separator."""
        defn = ProcessDefinition(
            name="dash-test",
            phases=[
                PhaseTemplate(
                    id="p1",
                    description="Phase",
                    deliverables=["report.md\u2014Description here"],
                ),
            ],
        )
        deliverables = defn.get_deliverables()
        assert deliverables["p1"] == ["report.md"]

    def test_default_field_values(self):
        defn = ProcessDefinition(name="minimal")
        assert defn.version == "1.0"
        assert defn.description == ""
        assert defn.author == ""
        assert defn.tags == []
        assert defn.persona == ""
        assert defn.tool_guidance == ""
        assert defn.phase_mode == "guided"
        assert defn.phases == []
        assert defn.verification_rules == []
        assert defn.memory_types == []
        assert defn.extraction_guidance == ""
        assert defn.planner_examples == []
        assert defn.source_path is None
        assert defn.package_dir is None


# ===================================================================
# ProcessLoader Tests
# ===================================================================


class TestQuickParseName:
    """Tests for ProcessLoader._quick_parse_name."""

    def test_extracts_name_from_first_line(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("name: my-process\nversion: '1.0'\n")
        assert ProcessLoader._quick_parse_name(f) == "my-process"

    def test_extracts_quoted_name(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("name: 'quoted-name'\nversion: '1.0'\n")
        assert ProcessLoader._quick_parse_name(f) == "quoted-name"

    def test_extracts_double_quoted_name(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text('name: "double-quoted"\nversion: \'1.0\'\n')
        assert ProcessLoader._quick_parse_name(f) == "double-quoted"

    def test_skips_comments_and_blank_lines(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("# This is a comment\n\nname: after-comment\n")
        assert ProcessLoader._quick_parse_name(f) == "after-comment"

    def test_fallback_to_full_parse(self, tmp_path):
        """When name is not the first non-comment field, falls back to full parse."""
        f = tmp_path / "test.yaml"
        f.write_text("version: '1.0'\nname: fallback-name\n")
        assert ProcessLoader._quick_parse_name(f) == "fallback-name"

    def test_returns_none_for_empty_file(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        assert ProcessLoader._quick_parse_name(f) is None

    def test_returns_none_for_invalid_yaml(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text(":\n  - not valid\n  [broken")
        assert ProcessLoader._quick_parse_name(f) is None

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        f = tmp_path / "nonexistent.yaml"
        assert ProcessLoader._quick_parse_name(f) is None


class TestProcessLoaderDiscover:
    """Tests for ProcessLoader.discover()."""

    def test_discovers_yaml_files(self, process_dir):
        loader = ProcessLoader(extra_search_paths=[process_dir])
        available = loader.discover()
        assert "test-process" in available
        assert "full-process" in available

    def test_discover_returns_paths(self, process_dir):
        loader = ProcessLoader(extra_search_paths=[process_dir])
        available = loader.discover()
        assert available["test-process"].exists()
        assert available["test-process"].suffix == ".yaml"

    def test_discover_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        loader = ProcessLoader(extra_search_paths=[empty])
        available = loader.discover()
        # May contain builtin processes but not our custom ones
        assert "test-process" not in available

    def test_discover_nonexistent_directory(self, tmp_path):
        loader = ProcessLoader(extra_search_paths=[tmp_path / "nope"])
        # Should not raise, just return whatever built-ins exist
        available = loader.discover()
        assert isinstance(available, dict)

    def test_later_paths_override_earlier(self, tmp_path):
        """Later search paths take precedence over earlier ones."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        (dir1 / "test-process.yaml").write_text(
            "name: test-process\nversion: '1.0'\ndescription: from dir1\n"
        )
        (dir2 / "test-process.yaml").write_text(
            "name: test-process\nversion: '2.0'\ndescription: from dir2\n"
        )
        loader = ProcessLoader(extra_search_paths=[dir1, dir2])
        available = loader.discover()
        # dir2 should win (later in search paths)
        assert str(dir2) in str(available["test-process"])

    def test_discover_directory_package(self, tmp_path):
        """Test discovering a directory-based process package."""
        proc_dir = tmp_path / "processes"
        proc_dir.mkdir()
        pkg = proc_dir / "my-package"
        pkg.mkdir()
        (pkg / "process.yaml").write_text("name: my-package\nversion: '1.0'\n")
        loader = ProcessLoader(extra_search_paths=[proc_dir])
        available = loader.discover()
        assert "my-package" in available
        # Should point to the directory, not the YAML
        assert available["my-package"] == pkg


class TestProcessLoaderLoad:
    """Tests for ProcessLoader.load()."""

    def test_load_by_name(self, loader):
        defn = loader.load("test-process")
        assert defn.name == "test-process"
        assert defn.persona == "You are a test assistant."

    def test_load_full_process(self, loader):
        defn = loader.load("full-process")
        assert defn.name == "full-process"
        assert defn.version == "2.0"
        assert len(defn.phases) == 3
        assert len(defn.verification_rules) == 2
        assert defn.phase_mode == "strict"

    def test_load_by_path(self, process_dir):
        loader = ProcessLoader()
        path = str(process_dir / "test-process.yaml")
        defn = loader.load(path)
        assert defn.name == "test-process"

    def test_load_sets_source_path(self, loader, process_dir):
        defn = loader.load("test-process")
        assert defn.source_path is not None
        assert defn.source_path.exists()

    def test_load_not_found_raises(self, loader):
        with pytest.raises(ProcessNotFoundError) as exc_info:
            loader.load("nonexistent-process")
        assert "nonexistent-process" in str(exc_info.value)
        assert exc_info.value.name == "nonexistent-process"

    def test_load_not_found_includes_available(self, loader):
        with pytest.raises(ProcessNotFoundError) as exc_info:
            loader.load("nonexistent-process")
        # Should list available processes
        assert exc_info.value.available is not None
        assert "test-process" in exc_info.value.available

    def test_load_directory_package(self, tmp_path):
        """Test loading from a directory-based process package."""
        proc_dir = tmp_path / "processes"
        proc_dir.mkdir()
        pkg = proc_dir / "dir-process"
        pkg.mkdir()
        (pkg / "process.yaml").write_text(
            "name: dir-process\nversion: '1.0'\ndescription: Directory package\n"
        )
        loader = ProcessLoader(extra_search_paths=[proc_dir])
        defn = loader.load("dir-process")
        assert defn.name == "dir-process"
        assert defn.package_dir == pkg

    def test_load_directory_package_by_path(self, tmp_path):
        """Loading by path to a directory should find process.yaml inside."""
        pkg = tmp_path / "my-pkg"
        pkg.mkdir()
        (pkg / "process.yaml").write_text(
            "name: my-pkg\nversion: '1.0'\ndescription: Direct path load\n"
        )
        loader = ProcessLoader()
        defn = loader.load(str(pkg))
        assert defn.name == "my-pkg"
        assert defn.package_dir == pkg

    def test_load_directory_missing_process_yaml(self, tmp_path):
        """Loading from a directory without process.yaml should raise."""
        pkg = tmp_path / "empty-pkg"
        pkg.mkdir()
        loader = ProcessLoader()
        with pytest.raises(ProcessValidationError, match="process.yaml"):
            loader.load(str(pkg))


class TestProcessLoaderValidation:
    """Tests for ProcessLoader._validate and related validation."""

    def _load_yaml_str(self, tmp_path, content):
        """Helper: write YAML content to a file and load it."""
        f = tmp_path / "test.yaml"
        f.write_text(content)
        loader = ProcessLoader()
        return loader.load(str(f))

    def test_valid_process_passes(self, tmp_path):
        defn = self._load_yaml_str(tmp_path, MINIMAL_YAML)
        assert defn.name == "test-process"

    def test_missing_name_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, MISSING_NAME_YAML)
        assert any("Missing required field: name" in e for e in exc_info.value.errors)

    def test_invalid_name_format_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, INVALID_NAME_YAML)
        assert any("Invalid name" in e for e in exc_info.value.errors)

    def test_duplicate_phase_ids_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, DUPLICATE_PHASE_YAML)
        assert any("Duplicate phase id" in e for e in exc_info.value.errors)

    def test_empty_phase_description_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, EMPTY_PHASE_DESC_YAML)
        assert any("empty description" in e for e in exc_info.value.errors)

    def test_unknown_dependency_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, UNKNOWN_DEP_YAML)
        assert any("unknown phase" in e for e in exc_info.value.errors)

    def test_dependency_cycle_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, CYCLE_YAML)
        assert any("cycle" in e.lower() for e in exc_info.value.errors)

    def test_duplicate_deliverables_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, DUPLICATE_DELIVERABLE_YAML)
        assert any("Duplicate deliverable" in e for e in exc_info.value.errors)

    def test_invalid_severity_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, INVALID_SEVERITY_YAML)
        assert any("invalid severity" in e for e in exc_info.value.errors)

    def test_invalid_regex_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, INVALID_REGEX_YAML)
        assert any("invalid regex" in e for e in exc_info.value.errors)

    def test_invalid_phase_mode_raises(self, tmp_path):
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, INVALID_PHASE_MODE_YAML)
        assert any("Invalid phase_mode" in e for e in exc_info.value.errors)

    def test_empty_yaml_raises(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        loader = ProcessLoader()
        with pytest.raises(ProcessValidationError, match="Empty or invalid"):
            loader.load(str(f))

    def test_validation_error_includes_path(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text(INVALID_NAME_YAML)
        loader = ProcessLoader()
        with pytest.raises(ProcessValidationError) as exc_info:
            loader.load(str(f))
        assert exc_info.value.path == f


class TestDetectCycles:
    """Tests for ProcessLoader._detect_cycles."""

    def test_no_cycle_in_linear_chain(self):
        phases = [
            PhaseTemplate(id="a", description="A"),
            PhaseTemplate(id="b", description="B", depends_on=["a"]),
            PhaseTemplate(id="c", description="C", depends_on=["b"]),
        ]
        loader = ProcessLoader()
        assert loader._detect_cycles(phases) is None

    def test_no_cycle_in_dag(self):
        """Diamond DAG: A -> B, A -> C, B -> D, C -> D."""
        phases = [
            PhaseTemplate(id="a", description="A"),
            PhaseTemplate(id="b", description="B", depends_on=["a"]),
            PhaseTemplate(id="c", description="C", depends_on=["a"]),
            PhaseTemplate(id="d", description="D", depends_on=["b", "c"]),
        ]
        loader = ProcessLoader()
        assert loader._detect_cycles(phases) is None

    def test_simple_cycle(self):
        """A -> B -> A."""
        phases = [
            PhaseTemplate(id="a", description="A", depends_on=["b"]),
            PhaseTemplate(id="b", description="B", depends_on=["a"]),
        ]
        loader = ProcessLoader()
        cycle = loader._detect_cycles(phases)
        assert cycle is not None
        assert "a" in cycle
        assert "b" in cycle

    def test_three_node_cycle(self):
        """A -> B -> C -> A."""
        phases = [
            PhaseTemplate(id="a", description="A", depends_on=["c"]),
            PhaseTemplate(id="b", description="B", depends_on=["a"]),
            PhaseTemplate(id="c", description="C", depends_on=["b"]),
        ]
        loader = ProcessLoader()
        cycle = loader._detect_cycles(phases)
        assert cycle is not None

    def test_self_cycle(self):
        """A -> A."""
        phases = [
            PhaseTemplate(id="a", description="A", depends_on=["a"]),
        ]
        loader = ProcessLoader()
        cycle = loader._detect_cycles(phases)
        assert cycle is not None
        assert "a" in cycle

    def test_no_phases_no_cycle(self):
        loader = ProcessLoader()
        assert loader._detect_cycles([]) is None

    def test_disconnected_components_no_cycle(self):
        phases = [
            PhaseTemplate(id="a", description="A"),
            PhaseTemplate(id="b", description="B"),
            PhaseTemplate(id="c", description="C"),
        ]
        loader = ProcessLoader()
        assert loader._detect_cycles(phases) is None


class TestProcessLoaderListAvailable:
    """Tests for ProcessLoader.list_available()."""

    def test_list_available_returns_metadata(self, loader):
        items = loader.list_available()
        names = [i["name"] for i in items]
        assert "test-process" in names
        assert "full-process" in names

    def test_list_available_item_fields(self, loader):
        items = loader.list_available()
        for item in items:
            assert "name" in item
            assert "version" in item
            assert "description" in item
            assert "path" in item


class TestProcessLoaderSearchPaths:
    """Tests for search path configuration."""

    def test_search_paths_include_builtin(self):
        loader = ProcessLoader()
        paths = loader.search_paths
        assert any("builtin" in str(p) for p in paths)

    def test_search_paths_include_home(self):
        loader = ProcessLoader()
        paths = loader.search_paths
        assert any(".loom/processes" in str(p) for p in paths)

    def test_search_paths_include_workspace(self, tmp_path):
        loader = ProcessLoader(workspace=tmp_path)
        paths = loader.search_paths
        assert tmp_path / ".loom" / "processes" in paths
        assert tmp_path / "loom-processes" in paths

    def test_search_paths_include_extra(self, tmp_path):
        extra = tmp_path / "extra"
        loader = ProcessLoader(extra_search_paths=[extra])
        paths = loader.search_paths
        assert extra in paths


# ===================================================================
# PromptAssembler Integration Tests
# ===================================================================


class TestPlannerPromptWithProcess:
    """Test planner prompt assembly with a process definition loaded."""

    def test_persona_injected(self, sample_task, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        prompt = assembler.build_planner_prompt(sample_task)
        assert "domain-expert assistant" in prompt

    def test_phase_blueprint_injected(self, sample_task, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        prompt = assembler.build_planner_prompt(sample_task)
        assert "research" in prompt
        assert "implement" in prompt
        assert "verify" in prompt
        assert "strict" in prompt

    def test_planner_examples_injected(self, sample_task, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        prompt = assembler.build_planner_prompt(sample_task)
        assert "Refactor module X" in prompt
        assert "analyze-x" in prompt

    def test_workspace_guidance_injected(self, sample_task):
        defn = ProcessDefinition(
            name="ws-test",
            workspace_guidance="Look for configuration files first.",
        )
        assembler = PromptAssembler(process=defn)
        prompt = assembler.build_planner_prompt(sample_task)
        assert "configuration files first" in prompt

    def test_workspace_guidance_not_overridden_when_explicit(self, sample_task):
        """When workspace_analysis is explicitly provided, process guidance is not used."""
        defn = ProcessDefinition(
            name="ws-test",
            workspace_guidance="Process guidance here.",
        )
        assembler = PromptAssembler(process=defn)
        prompt = assembler.build_planner_prompt(
            sample_task, workspace_analysis="Explicit analysis provided."
        )
        assert "Explicit analysis provided" in prompt


class TestExecutorPromptWithProcess:
    """Test executor prompt assembly with a process definition loaded."""

    def test_persona_injected(self, sample_task, state_manager, full_process_defn):
        state_manager.create(sample_task)
        assembler = PromptAssembler(process=full_process_defn)
        subtask = sample_task.get_subtask("implement-api")
        prompt = assembler.build_executor_prompt(
            task=sample_task, subtask=subtask, state_manager=state_manager,
        )
        assert "domain-expert assistant" in prompt

    def test_tool_guidance_injected(self, sample_task, state_manager, full_process_defn):
        state_manager.create(sample_task)
        assembler = PromptAssembler(process=full_process_defn)
        subtask = sample_task.get_subtask("implement-api")
        prompt = assembler.build_executor_prompt(
            task=sample_task, subtask=subtask, state_manager=state_manager,
        )
        assert "DOMAIN-SPECIFIC TOOL GUIDANCE" in prompt
        assert "read_file before editing" in prompt


class TestVerifierPromptWithProcess:
    """Test verifier prompt assembly with process verification rules."""

    def test_llm_rules_injected(self, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        subtask = Subtask(id="verify-task", description="Verify the output")
        prompt = assembler.build_verifier_prompt(
            subtask=subtask,
            result_summary="Implementation complete.",
            tool_calls_formatted="No tool calls.",
        )
        assert "ADDITIONAL DOMAIN-SPECIFIC CHECKS" in prompt
        assert "coverage-check" in prompt
        assert "test coverage exceeds 80%" in prompt
        assert "ERROR" in prompt  # severity uppercase

    def test_regex_rules_not_injected_into_verifier(self, full_process_defn):
        """Regex rules are checked separately, not injected into LLM verifier prompt."""
        assembler = PromptAssembler(process=full_process_defn)
        subtask = Subtask(id="v", description="Verify")
        prompt = assembler.build_verifier_prompt(subtask, "Done.", "")
        # The regex rule "no-todos" should not appear as a domain check
        # (only llm rules are injected)
        assert "no-todos" not in prompt

    def test_no_rules_no_injection(self):
        defn = ProcessDefinition(name="no-rules")
        assembler = PromptAssembler(process=defn)
        subtask = Subtask(id="v", description="Verify")
        prompt = assembler.build_verifier_prompt(subtask, "Done.", "")
        assert "DOMAIN-SPECIFIC CHECKS" not in prompt


class TestExtractorPromptWithProcess:
    """Test extractor prompt assembly with process memory configuration."""

    def test_memory_types_injected(self, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        prompt = assembler.build_extractor_prompt(
            subtask_id="s1",
            tool_calls_formatted="No calls.",
            model_output="Done.",
        )
        assert "DOMAIN-SPECIFIC MEMORY TYPES" in prompt
        assert "architecture_decision" in prompt
        assert "risk_factor" in prompt

    def test_extraction_guidance_injected(self, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        prompt = assembler.build_extractor_prompt("s1", "", "")
        assert "EXTRACTION GUIDANCE" in prompt
        assert "decisions that affect multiple phases" in prompt

    def test_no_process_no_extraction_extras(self):
        assembler = PromptAssembler()
        prompt = assembler.build_extractor_prompt("s1", "", "")
        assert "DOMAIN-SPECIFIC MEMORY TYPES" not in prompt
        assert "EXTRACTION GUIDANCE" not in prompt


class TestReplannerPromptWithProcess:
    """Test replanner prompt assembly with process definition."""

    def test_replanning_guidance_injected(self, sample_task, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        prompt = assembler.build_replanner_prompt(
            goal=sample_task.goal,
            current_state_yaml="status: executing",
            discoveries=["Found legacy code"],
            errors=["Test failure"],
            original_plan=sample_task.plan,
        )
        assert "DOMAIN-SPECIFIC REPLANNING GUIDANCE" in prompt
        assert "test setup before changing" in prompt

    def test_persona_used_for_replanner(self, sample_task, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        prompt = assembler.build_replanner_prompt(
            goal=sample_task.goal,
            current_state_yaml="",
            discoveries=[],
            errors=[],
            original_plan=sample_task.plan,
        )
        assert "domain-expert assistant" in prompt

    def test_no_replanning_guidance_no_injection(self, sample_task):
        defn = ProcessDefinition(name="no-replan")
        assembler = PromptAssembler(process=defn)
        prompt = assembler.build_replanner_prompt(
            goal=sample_task.goal,
            current_state_yaml="",
            discoveries=[],
            errors=[],
            original_plan=sample_task.plan,
        )
        assert "DOMAIN-SPECIFIC REPLANNING GUIDANCE" not in prompt


class TestBackwardCompatibility:
    """Ensure that with no process loaded, everything works exactly as before."""

    def test_planner_no_process(self, sample_task):
        assembler = PromptAssembler()
        assert assembler.process is None
        prompt = assembler.build_planner_prompt(sample_task)
        assert "Build a REST API" in prompt
        # No process-specific content
        assert "DOMAIN-SPECIFIC" not in prompt
        assert "Phase mode" not in prompt

    def test_executor_no_process(self, sample_task, state_manager):
        state_manager.create(sample_task)
        assembler = PromptAssembler()
        subtask = sample_task.get_subtask("implement-api")
        prompt = assembler.build_executor_prompt(
            task=sample_task, subtask=subtask, state_manager=state_manager,
        )
        assert "implement-api" in prompt
        assert "DOMAIN-SPECIFIC TOOL GUIDANCE" not in prompt

    def test_verifier_no_process(self):
        assembler = PromptAssembler()
        subtask = Subtask(id="v1", description="Verify")
        prompt = assembler.build_verifier_prompt(subtask, "OK.", "")
        assert "DOMAIN-SPECIFIC CHECKS" not in prompt

    def test_extractor_no_process(self):
        assembler = PromptAssembler()
        prompt = assembler.build_extractor_prompt("s1", "", "")
        assert "DOMAIN-SPECIFIC MEMORY TYPES" not in prompt

    def test_replanner_no_process(self, sample_task):
        assembler = PromptAssembler()
        prompt = assembler.build_replanner_prompt(
            goal="Test", current_state_yaml="",
            discoveries=[], errors=[],
            original_plan=sample_task.plan,
        )
        assert "DOMAIN-SPECIFIC REPLANNING GUIDANCE" not in prompt


class TestProcessProperty:
    """Test PromptAssembler.process property getter/setter."""

    def test_set_process(self, full_process_defn):
        assembler = PromptAssembler()
        assert assembler.process is None
        assembler.process = full_process_defn
        assert assembler.process is not None
        assert assembler.process.name == "full-process"

    def test_clear_process(self, full_process_defn):
        assembler = PromptAssembler(process=full_process_defn)
        assembler.process = None
        assert assembler.process is None


# ===================================================================
# ProcessConfig Tests
# ===================================================================


class TestProcessConfig:
    """Tests for ProcessConfig defaults and TOML parsing."""

    def test_default_values(self):
        cfg = ProcessConfig()
        assert cfg.default == ""
        assert cfg.search_paths == []

    def test_config_includes_process(self):
        cfg = Config()
        assert isinstance(cfg.process, ProcessConfig)
        assert cfg.process.default == ""

    def test_load_config_with_process_section(self, tmp_path):
        toml_content = """\
[process]
default = "my-process"
search_paths = ["/home/user/processes", "/opt/loom/processes"]
"""
        toml_path = tmp_path / "loom.toml"
        toml_path.write_text(toml_content)
        cfg = load_config(toml_path)
        assert cfg.process.default == "my-process"
        assert cfg.process.search_paths == [
            "/home/user/processes",
            "/opt/loom/processes",
        ]

    def test_load_config_without_process_section(self, tmp_path):
        """Config without [process] section should use defaults."""
        toml_content = """\
[server]
port = 8080
"""
        toml_path = tmp_path / "loom.toml"
        toml_path.write_text(toml_content)
        cfg = load_config(toml_path)
        assert cfg.process.default == ""
        assert cfg.process.search_paths == []

    def test_process_config_is_frozen(self):
        cfg = ProcessConfig()
        with pytest.raises(AttributeError):
            cfg.default = "changed"


# ===================================================================
# Exception Tests
# ===================================================================


class TestExceptions:
    """Tests for process-specific exceptions."""

    def test_process_not_found_error_message(self):
        err = ProcessNotFoundError("missing-proc", available=["a", "b"])
        assert "missing-proc" in str(err)
        assert "a" in str(err)
        assert "b" in str(err)

    def test_process_not_found_error_no_available(self):
        err = ProcessNotFoundError("missing-proc")
        assert "missing-proc" in str(err)
        assert err.available == []

    def test_process_validation_error_with_path(self):
        err = ProcessValidationError(
            ["error 1", "error 2"],
            path=Path("/some/file.yaml"),
        )
        assert "2 validation error" in str(err)
        assert "/some/file.yaml" in str(err)
        assert "error 1" in str(err)

    def test_process_validation_error_without_path(self):
        err = ProcessValidationError(["single error"])
        assert "1 validation error" in str(err)
        assert "single error" in str(err)


# ===================================================================
# Bundled Tools Tests
# ===================================================================


class TestRegisterBundledTools:
    """Tests for _register_bundled_tools."""

    def test_skips_missing_tools_dir(self, tmp_path):
        """No error when package has no tools/ directory."""
        import sys

        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "process.yaml").write_text("name: no-tools\nversion: '1.0'\n")
        before = {
            name for name in sys.modules
            if name.startswith("loom.processes._bundled.pkg.")
        }
        ProcessLoader._register_bundled_tools(pkg)
        after = {
            name for name in sys.modules
            if name.startswith("loom.processes._bundled.pkg.")
        }
        assert after == before

    def test_skips_underscore_files(self, tmp_path):
        """Files starting with _ are skipped."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        tools_dir = pkg / "tools"
        tools_dir.mkdir()
        (tools_dir / "_private.py").write_text("LOADED = True\n")
        ProcessLoader._register_bundled_tools(pkg)
        assert "loom.processes._bundled._private" not in __import__("sys").modules

    def test_logs_warning_on_bad_tool(self, tmp_path, caplog):
        """Bad tool files should log a warning, not crash."""
        import logging
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        tools_dir = pkg / "tools"
        tools_dir.mkdir()
        (tools_dir / "bad_tool.py").write_text("raise RuntimeError('boom')\n")
        with caplog.at_level(logging.WARNING):
            ProcessLoader._register_bundled_tools(pkg)
        assert "Failed to load bundled tool" in caplog.text


# ===================================================================
# ProcessDefinition Edge Cases
# ===================================================================


class TestProcessDefinitionEdgeCases:
    """Additional edge case tests for ProcessDefinition."""

    def test_get_deliverables_no_separator(self):
        """Deliverable with no em-dash separator returns full string."""
        defn = ProcessDefinition(
            name="test",
            phases=[
                PhaseTemplate(
                    id="p1", description="Phase 1",
                    deliverables=["report.md"],
                ),
            ],
        )
        deliverables = defn.get_deliverables()
        assert deliverables["p1"] == ["report.md"]

    def test_get_deliverables_whitespace_handling(self):
        """Whitespace around filename is stripped."""
        defn = ProcessDefinition(
            name="test",
            phases=[
                PhaseTemplate(
                    id="p1", description="Phase 1",
                    deliverables=["  report.md  \u2014  Description  "],
                ),
            ],
        )
        deliverables = defn.get_deliverables()
        assert deliverables["p1"] == ["report.md"]

    def test_verification_rule_type_defaults(self):
        """Verify default values for VerificationRule."""
        rule = VerificationRule(
            name="test", description="A test rule", check="pattern",
        )
        assert rule.severity == "warning"
        assert rule.type == "llm"
        assert rule.target == "output"

    def test_phase_template_defaults(self):
        """Verify default values for PhaseTemplate."""
        phase = PhaseTemplate(id="p1", description="Phase 1")
        assert phase.depends_on == []
        assert phase.model_tier == 2
        assert phase.verification_tier == 1
        assert phase.is_critical_path is False
        assert phase.is_synthesis is False
        assert phase.acceptance_criteria == ""
        assert phase.deliverables == []


class TestToolOverlapValidation:
    """Tests for tool required/excluded overlap detection."""

    def _load_yaml_str(self, tmp_path, content):
        """Helper: write YAML and load."""
        f = tmp_path / "test.yaml"
        f.write_text(content)
        loader = ProcessLoader()
        return loader.load(str(f))

    def test_overlap_raises(self, tmp_path):
        """Tools in both required and excluded should fail validation."""
        yaml_content = """\
name: overlap-test
version: '1.0'
tools:
  required:
    - read_file
    - shell_execute
  excluded:
    - shell_execute
    - dangerous_tool
"""
        with pytest.raises(ProcessValidationError) as exc_info:
            self._load_yaml_str(tmp_path, yaml_content)
        assert any(
            "both required and excluded" in e
            for e in exc_info.value.errors
        )
        assert any("shell_execute" in e for e in exc_info.value.errors)

    def test_no_overlap_passes(self, tmp_path):
        """Disjoint required/excluded sets should pass validation."""
        yaml_content = """\
name: no-overlap
version: '1.0'
tools:
  required:
    - read_file
    - write_file
  excluded:
    - shell_execute
"""
        defn = self._load_yaml_str(tmp_path, yaml_content)
        assert defn.tools.required == ["read_file", "write_file"]
        assert defn.tools.excluded == ["shell_execute"]

    def test_only_required_passes(self, tmp_path):
        """Having only required (no excluded) should pass."""
        yaml_content = """\
name: only-required
version: '1.0'
tools:
  required:
    - read_file
"""
        defn = self._load_yaml_str(tmp_path, yaml_content)
        assert defn.tools.required == ["read_file"]
        assert defn.tools.excluded == []

    def test_only_excluded_passes(self, tmp_path):
        """Having only excluded (no required) should pass."""
        yaml_content = """\
name: only-excluded
version: '1.0'
tools:
  excluded:
    - shell_execute
"""
        defn = self._load_yaml_str(tmp_path, yaml_content)
        assert defn.tools.required == []
        assert defn.tools.excluded == ["shell_execute"]
