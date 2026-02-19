"""Process definition schema, loading, and validation.

A process definition is the unit of domain specialization. It can be
a single YAML file or a directory (package) containing a ``process.yaml``
plus optional bundled tools, templates, and examples.

The engine remains fully generic — the process definition injects domain
intelligence into existing prompt templates via the PromptAssembler.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ProcessNotFoundError(Exception):
    """Raised when a process definition cannot be found."""

    def __init__(self, name: str, available: list[str] | None = None):
        available = available or []
        msg = f"Process not found: {name!r}"
        if available:
            msg += f". Available: {', '.join(sorted(available))}"
        super().__init__(msg)
        self.name = name
        self.available = available


class ProcessValidationError(Exception):
    """Raised when a process definition fails validation."""

    def __init__(self, errors: list[str], path: Path | None = None):
        prefix = f"{path}: " if path else ""
        msg = f"{prefix}{len(errors)} validation error(s):\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)
        self.errors = errors
        self.path = path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PhaseTemplate:
    """A blueprint phase in the process definition."""

    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    model_tier: int = 2
    verification_tier: int = 1
    is_critical_path: bool = False
    is_synthesis: bool = False
    acceptance_criteria: str = ""
    deliverables: list[str] = field(default_factory=list)


@dataclass
class VerificationRule:
    """A domain-specific verification check."""

    name: str
    description: str
    check: str
    severity: str = "warning"  # "warning" | "error"
    type: str = "llm"  # "llm" | "regex"
    target: str = "output"  # "output" | "deliverables"
    enforcement: str = ""  # "" | "hard" | "advisory"
    scope: str = ""  # "" | "phase" | "global"
    applies_to_phases: list[str] = field(default_factory=list)  # [] | ["phase-a"] | ["*"]
    requires_exact_cardinality: bool = False
    min_count: int | None = None


@dataclass
class MemoryType:
    """A domain-specific memory extraction type."""

    type: str
    description: str


@dataclass
class PlannerExample:
    """A few-shot example for the planner."""

    goal: str
    subtasks: list[dict] = field(default_factory=list)


@dataclass
class ProcessTestAcceptance:
    """Acceptance contract for a process test case."""

    phases_must_include: list[str] = field(default_factory=list)
    deliverables_must_exist: list[str] = field(default_factory=list)
    verification_forbidden_patterns: list[str] = field(default_factory=list)


@dataclass
class ProcessTestCase:
    """Declarative test case embedded in process.yaml."""

    id: str
    mode: str = "deterministic"  # "deterministic" | "live"
    goal: str = ""
    timeout_seconds: int = 900
    requires_network: bool = False
    requires_tools: list[str] = field(default_factory=list)
    acceptance: ProcessTestAcceptance = field(
        default_factory=ProcessTestAcceptance,
    )


@dataclass
class ToolRequirements:
    """Tool availability and restriction configuration."""

    guidance: str = ""
    required: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)


@dataclass
class ProcessDefinition:
    """Complete process definition — the unit of domain specialization.

    Loaded from YAML and injected into the PromptAssembler to specialize
    the engine for a particular domain without any Python code changes.
    """

    # Metadata
    name: str
    version: str = "1.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)

    # Behavior
    persona: str = ""
    tool_guidance: str = ""
    tools: ToolRequirements = field(default_factory=ToolRequirements)
    phase_mode: str = "guided"  # "strict" | "guided" | "suggestive"
    phases: list[PhaseTemplate] = field(default_factory=list)
    verification_rules: list[VerificationRule] = field(
        default_factory=list,
    )
    memory_types: list[MemoryType] = field(default_factory=list)
    extraction_guidance: str = ""
    workspace_scan: list[str] = field(default_factory=list)
    workspace_guidance: str = ""
    planner_examples: list[PlannerExample] = field(default_factory=list)
    tests: list[ProcessTestCase] = field(default_factory=list)
    replanning_triggers: str = ""
    replanning_guidance: str = ""

    # Package metadata
    dependencies: list[str] = field(default_factory=list)

    # Resolved paths (set by loader)
    source_path: Path | None = None
    package_dir: Path | None = None

    def has_phases(self) -> bool:
        return bool(self.phases)

    def has_verification_rules(self) -> bool:
        return bool(self.verification_rules)

    def regex_rules(self) -> list[VerificationRule]:
        """Return only regex-type verification rules."""
        return [r for r in self.verification_rules if r.type == "regex"]

    def llm_rules(self) -> list[VerificationRule]:
        """Return only LLM-type verification rules."""
        return [r for r in self.verification_rules if r.type == "llm"]

    @staticmethod
    def _rule_applies_to_subtask(
        rule: VerificationRule,
        subtask_id: str,
        *,
        phase_scope_default: str = "global",
    ) -> bool:
        """Return True when rule should apply to current subtask."""
        normalized_scope_default = str(phase_scope_default or "global").strip().lower()
        if normalized_scope_default not in {"global", "current_phase"}:
            normalized_scope_default = "global"

        normalized_subtask = str(subtask_id or "").strip()
        if not normalized_subtask:
            return False

        applies = [str(p or "").strip() for p in (rule.applies_to_phases or [])]
        applies = [p for p in applies if p]
        if applies:
            if "*" in applies:
                return True
            return normalized_subtask in applies

        scope = str(rule.scope or "").strip().lower()
        if scope == "global":
            return True
        if scope == "phase":
            # Rule is explicitly phase-scoped but no explicit phase list exists.
            # Interpret as "apply to the currently executing phase/subtask".
            return bool(normalized_subtask)

        # Legacy rule (no scope metadata)
        if normalized_scope_default == "global":
            return True
        # Migration mode: treat legacy rules as current-phase checks.
        return bool(normalized_subtask)

    def regex_rules_for_subtask(
        self,
        subtask_id: str,
        *,
        phase_scope_default: str = "global",
    ) -> list[VerificationRule]:
        """Return regex rules applicable to the current subtask."""
        return [
            r for r in self.regex_rules()
            if self._rule_applies_to_subtask(
                r,
                subtask_id,
                phase_scope_default=phase_scope_default,
            )
        ]

    def llm_rules_for_subtask(
        self,
        subtask_id: str,
        *,
        phase_scope_default: str = "global",
    ) -> list[VerificationRule]:
        """Return llm rules applicable to the current subtask."""
        return [
            r for r in self.llm_rules()
            if self._rule_applies_to_subtask(
                r,
                subtask_id,
                phase_scope_default=phase_scope_default,
            )
        ]

    def get_deliverables(self) -> dict[str, list[str]]:
        """Return {phase_id: [deliverable_filenames]} for all phases."""
        result: dict[str, list[str]] = {}
        for phase in self.phases:
            if phase.deliverables:
                # Extract just the filename from "filename — description"
                filenames = []
                for d in phase.deliverables:
                    fname = d.split("—")[0].split(" — ")[0].strip()
                    filenames.append(fname)
                result[phase.id] = filenames
        return result


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class ProcessLoader:
    """Discovers and loads process definition files.

    Search order (later entries take precedence):
    1. Built-in processes shipped with Loom
    2. User-global (~/.loom/processes/)
    3. Workspace-local (./loom-processes/ or ./.loom/processes/)

    Supports both single YAML files and directory-based packages.
    """

    BUILTIN_DIR = Path(__file__).parent / "builtin"

    def __init__(
        self,
        workspace: Path | None = None,
        extra_search_paths: list[Path] | None = None,
    ):
        self._workspace = workspace
        self._extra = extra_search_paths or []

    @property
    def search_paths(self) -> list[Path]:
        """Return search paths in precedence order (last wins)."""
        paths = [self.BUILTIN_DIR]
        paths.append(Path.home() / ".loom" / "processes")
        for p in self._extra:
            paths.append(Path(p).expanduser())
        if self._workspace:
            paths.append(self._workspace / ".loom" / "processes")
            paths.append(self._workspace / "loom-processes")
        return paths

    def discover(self) -> dict[str, Path]:
        """Find all available process definitions.

        Returns ``{name: path}`` where path is the YAML file or the
        package directory.  Later search paths override earlier ones.
        """
        processes: dict[str, Path] = {}
        for search_dir in self.search_paths:
            if not search_dir.exists():
                continue
            # Single YAML files
            for yaml_file in search_dir.glob("*.yaml"):
                name = self._quick_parse_name(yaml_file)
                if name:
                    processes[name] = yaml_file
            # Directory-based packages (contain process.yaml)
            for subdir in search_dir.iterdir():
                if subdir.is_dir():
                    manifest = subdir / "process.yaml"
                    if manifest.exists():
                        name = self._quick_parse_name(manifest)
                        if name:
                            processes[name] = subdir
        return processes

    def load(self, name_or_path: str) -> ProcessDefinition:
        """Load a process definition by name or file/directory path."""
        path = Path(name_or_path)
        if path.exists():
            return self._load_from_path(path)

        # Search by name
        available = self.discover()
        if name_or_path not in available:
            raise ProcessNotFoundError(name_or_path, list(available.keys()))
        return self._load_from_path(available[name_or_path])

    def list_available(self) -> list[dict[str, str]]:
        """List all available processes with metadata.

        Only parses YAML for metadata — does NOT import bundled tools.
        """
        result = []
        for name, path in sorted(self.discover().items()):
            try:
                defn = self._load_metadata_only(path)
                result.append({
                    "name": defn.name,
                    "version": defn.version,
                    "description": defn.description,
                    "author": defn.author,
                    "path": str(path),
                })
            except Exception as e:
                logger.warning("Failed to load process %s: %s", name, e)
                result.append({
                    "name": name,
                    "version": "?",
                    "description": "(failed to load)",
                    "author": "",
                    "path": str(path),
                })
        return result

    # --- Internal ---

    def _load_metadata_only(self, path: Path) -> ProcessDefinition:
        """Parse a process definition for metadata only (no bundled tool imports)."""
        if path.is_dir():
            yaml_path = path / "process.yaml"
            if not yaml_path.exists():
                raise ProcessValidationError(
                    ["Package directory missing process.yaml"],
                    path,
                )
            defn = self._parse(yaml_path)
            defn.package_dir = path
            # Deliberately skip _register_bundled_tools
            return defn
        else:
            return self._parse(path)

    def _load_from_path(self, path: Path) -> ProcessDefinition:
        """Load from a YAML file or package directory."""
        if path.is_dir():
            yaml_path = path / "process.yaml"
            if not yaml_path.exists():
                raise ProcessValidationError(
                    ["Package directory missing process.yaml"],
                    path,
                )
            defn = self._parse(yaml_path)
            defn.package_dir = path
            # Activate per-package isolated dependencies if present.
            self._activate_isolated_dependencies(path)
            # Register bundled tools if present
            self._register_bundled_tools(path)
            return defn
        else:
            return self._parse(path)

    def _parse(self, path: Path) -> ProcessDefinition:
        """Parse a YAML file into a ProcessDefinition."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not raw or not isinstance(raw, dict):
            raise ProcessValidationError(
                ["Empty or invalid YAML file"], path,
            )

        defn = self._build_definition(raw)
        defn.source_path = path

        errors = self._validate(defn)
        if errors:
            raise ProcessValidationError(errors, path)
        return defn

    def _build_definition(self, raw: dict[str, Any]) -> ProcessDefinition:
        """Construct a ProcessDefinition from parsed YAML."""
        # Phases
        phases = []
        for p in raw.get("phases", []):
            phases.append(PhaseTemplate(
                id=p.get("id", ""),
                description=p.get("description", ""),
                depends_on=p.get("depends_on", []),
                model_tier=p.get("model_tier", 2),
                verification_tier=p.get("verification_tier", 1),
                is_critical_path=p.get("is_critical_path", False),
                is_synthesis=p.get("is_synthesis", False),
                acceptance_criteria=p.get("acceptance_criteria", ""),
                deliverables=p.get("deliverables", []),
            ))

        # Verification rules
        rules = []
        verif = raw.get("verification", {})
        for r in verif.get("rules", []) if isinstance(verif, dict) else []:
            applies_to = r.get("applies_to_phases", [])
            if isinstance(applies_to, str):
                applies_to = [applies_to]
            if not isinstance(applies_to, list):
                applies_to = []
            rules.append(VerificationRule(
                name=r.get("name", ""),
                description=r.get("description", ""),
                check=r.get("check", ""),
                severity=r.get("severity", "warning"),
                type=r.get("type", "llm"),
                target=r.get("target", "output"),
                enforcement=r.get("enforcement", ""),
                scope=r.get("scope", ""),
                applies_to_phases=[str(p).strip() for p in applies_to if str(p).strip()],
                requires_exact_cardinality=bool(
                    r.get("requires_exact_cardinality", False)
                ),
                min_count=r.get("min_count"),
            ))

        # Memory types
        memory_raw = raw.get("memory", {})
        memory_types = []
        for m in memory_raw.get("extract_types", []) if isinstance(
            memory_raw, dict,
        ) else []:
            memory_types.append(MemoryType(
                type=m.get("type", ""),
                description=m.get("description", ""),
            ))

        # Planner examples
        examples = []
        for ex in raw.get("planner_examples", []):
            examples.append(PlannerExample(
                goal=ex.get("goal", ""),
                subtasks=ex.get("subtasks", []),
            ))

        # Process tests
        tests: list[ProcessTestCase] = []
        tests_raw = raw.get("tests", [])
        if isinstance(tests_raw, list):
            for t in tests_raw:
                if not isinstance(t, dict):
                    continue
                timeout_raw = t.get("timeout_seconds", 900)
                try:
                    timeout_seconds = int(timeout_raw)
                except (TypeError, ValueError):
                    timeout_seconds = -1
                acceptance_raw = t.get("acceptance", {})
                if not isinstance(acceptance_raw, dict):
                    acceptance_raw = {}

                phases_raw = acceptance_raw.get("phases", {})
                if not isinstance(phases_raw, dict):
                    phases_raw = {}
                deliverables_raw = acceptance_raw.get("deliverables", {})
                if not isinstance(deliverables_raw, dict):
                    deliverables_raw = {}
                verification_raw = acceptance_raw.get("verification", {})
                if not isinstance(verification_raw, dict):
                    verification_raw = {}

                tests.append(ProcessTestCase(
                    id=str(t.get("id", "")).strip(),
                    mode=str(t.get("mode", "deterministic")).strip(),
                    goal=str(t.get("goal", "")).strip(),
                    timeout_seconds=timeout_seconds,
                    requires_network=bool(t.get("requires_network", False)),
                    requires_tools=[
                        item
                        for item in t.get("requires_tools", [])
                        if isinstance(item, str) and item.strip()
                    ] if isinstance(t.get("requires_tools", []), list) else [],
                    acceptance=ProcessTestAcceptance(
                        phases_must_include=[
                            item
                            for item in phases_raw.get("must_include", [])
                            if isinstance(item, str) and item.strip()
                        ] if isinstance(phases_raw.get("must_include", []), list) else [],
                        deliverables_must_exist=[
                            item
                            for item in deliverables_raw.get("must_exist", [])
                            if isinstance(item, str) and item.strip()
                        ] if isinstance(deliverables_raw.get("must_exist", []), list) else [],
                        verification_forbidden_patterns=[
                            item
                            for item in verification_raw.get("forbidden_patterns", [])
                            if isinstance(item, str) and item.strip()
                        ] if isinstance(
                            verification_raw.get("forbidden_patterns", []), list
                        ) else [],
                    ),
                ))

        # Tool requirements
        tool_raw = raw.get("tools", {})
        if isinstance(tool_raw, dict):
            tools = ToolRequirements(
                guidance=tool_raw.get("guidance", ""),
                required=tool_raw.get("required", []),
                excluded=tool_raw.get("excluded", []),
            )
        else:
            tools = ToolRequirements()

        # Workspace analysis
        ws_raw = raw.get("workspace_analysis", {})
        ws_scan = ws_raw.get("scan_for", []) if isinstance(
            ws_raw, dict,
        ) else []
        ws_guidance = ws_raw.get("guidance", "") if isinstance(
            ws_raw, dict,
        ) else ""

        # Replanning
        replan_raw = raw.get("replanning", {})
        replan_triggers = replan_raw.get("triggers", "") if isinstance(
            replan_raw, dict,
        ) else ""
        replan_guidance = replan_raw.get("guidance", "") if isinstance(
            replan_raw, dict,
        ) else ""

        # Top-level tool_guidance (legacy compat — also check tools.guidance)
        tool_guidance = raw.get("tool_guidance", "") or tools.guidance

        # Dependencies
        deps_raw = raw.get("dependencies", [])
        if not isinstance(deps_raw, list):
            deps_raw = []

        return ProcessDefinition(
            name=raw.get("name", ""),
            version=str(raw.get("version", "1.0")),
            description=raw.get("description", ""),
            author=raw.get("author", ""),
            tags=raw.get("tags", []),
            persona=raw.get("persona", ""),
            tool_guidance=tool_guidance,
            tools=tools,
            phase_mode=raw.get("phase_mode", "guided"),
            phases=phases,
            verification_rules=rules,
            memory_types=memory_types,
            extraction_guidance=memory_raw.get(
                "extraction_guidance", "",
            ) if isinstance(memory_raw, dict) else "",
            workspace_scan=ws_scan,
            workspace_guidance=ws_guidance,
            planner_examples=examples,
            tests=tests,
            replanning_triggers=replan_triggers,
            replanning_guidance=replan_guidance,
            dependencies=deps_raw,
        )

    def _validate(self, defn: ProcessDefinition) -> list[str]:
        """Validate a ProcessDefinition beyond schema structure."""
        errors: list[str] = []

        # Required fields
        if not defn.name:
            errors.append("Missing required field: name")

        # Name format
        if defn.name and not re.match(r"^[a-z0-9][a-z0-9-]*$", defn.name):
            errors.append(
                f"Invalid name {defn.name!r}: must be lowercase "
                f"alphanumeric with hyphens",
            )

        # Phase mode
        valid_modes = {"strict", "guided", "suggestive"}
        if defn.phase_mode not in valid_modes:
            errors.append(
                f"Invalid phase_mode {defn.phase_mode!r}: "
                f"must be one of {valid_modes}",
            )

        # Phase validation
        phase_ids = set()
        for phase in defn.phases:
            if not phase.id:
                errors.append("Phase missing required field: id")
                continue
            if phase.id in phase_ids:
                errors.append(f"Duplicate phase id: {phase.id!r}")
            phase_ids.add(phase.id)
            if not phase.description:
                errors.append(f"Phase {phase.id!r} has empty description")

        # Dependency graph validation
        for phase in defn.phases:
            for dep in phase.depends_on:
                if dep not in phase_ids:
                    errors.append(
                        f"Phase {phase.id!r} depends on unknown phase "
                        f"{dep!r}",
                    )

        # Cycle detection (topological sort)
        cycle = self._detect_cycles(defn.phases)
        if cycle:
            errors.append(
                f"Dependency cycle detected: {' -> '.join(cycle)}",
            )

        # Deliverable uniqueness
        seen_deliverables: dict[str, str] = {}
        for phase in defn.phases:
            for d in phase.deliverables:
                fname = d.split("—")[0].split(" — ")[0].strip()
                if fname in seen_deliverables:
                    errors.append(
                        f"Duplicate deliverable {fname!r} in phases "
                        f"{seen_deliverables[fname]!r} and {phase.id!r}",
                    )
                seen_deliverables[fname] = phase.id

        # Tool requirements: required and excluded must not overlap
        if defn.tools.required and defn.tools.excluded:
            overlap = set(defn.tools.required) & set(defn.tools.excluded)
            if overlap:
                errors.append(
                    f"Tool(s) both required and excluded: "
                    f"{', '.join(sorted(overlap))}",
                )

        # Process test validation
        seen_test_ids: set[str] = set()
        valid_test_modes = {"deterministic", "live"}
        for test_case in defn.tests:
            if not test_case.id:
                errors.append("Process test case missing required field: id")
            elif test_case.id in seen_test_ids:
                errors.append(
                    f"Duplicate process test id: {test_case.id!r}",
                )
            else:
                seen_test_ids.add(test_case.id)

            if test_case.mode not in valid_test_modes:
                errors.append(
                    f"Process test {test_case.id!r}: invalid mode "
                    f"{test_case.mode!r} (expected one of {valid_test_modes})",
                )

            if not test_case.goal:
                errors.append(
                    f"Process test {test_case.id!r} missing required field: goal",
                )

            if test_case.timeout_seconds <= 0:
                errors.append(
                    f"Process test {test_case.id!r}: timeout_seconds must be > 0",
                )

        # Verification rule validation
        for rule in defn.verification_rules:
            if not rule.name:
                errors.append("Verification rule missing name")
            if not rule.check:
                errors.append(
                    f"Verification rule {rule.name!r} missing check",
                )
            if rule.severity not in ("warning", "error"):
                errors.append(
                    f"Rule {rule.name!r}: invalid severity "
                    f"{rule.severity!r}",
                )
            if rule.type not in ("llm", "regex"):
                errors.append(
                    f"Rule {rule.name!r}: invalid type {rule.type!r}",
                )
            if rule.enforcement and rule.enforcement not in ("hard", "advisory"):
                errors.append(
                    f"Rule {rule.name!r}: invalid enforcement {rule.enforcement!r}",
                )
            if rule.scope and rule.scope not in ("phase", "global"):
                errors.append(
                    f"Rule {rule.name!r}: invalid scope {rule.scope!r}",
                )
            for phase_id in rule.applies_to_phases:
                if phase_id == "*":
                    continue
                if phase_id not in phase_ids:
                    errors.append(
                        f"Rule {rule.name!r}: applies_to_phases contains unknown phase "
                        f"{phase_id!r}",
                    )
            if rule.min_count is not None:
                try:
                    min_count = int(rule.min_count)
                except (TypeError, ValueError):
                    errors.append(
                        f"Rule {rule.name!r}: min_count must be an integer",
                    )
                else:
                    if min_count < 0:
                        errors.append(
                            f"Rule {rule.name!r}: min_count must be >= 0",
                        )
            # Validate regex rules compile
            if rule.type == "regex":
                try:
                    re.compile(rule.check)
                except re.error as e:
                    errors.append(
                        f"Rule {rule.name!r}: invalid regex: {e}",
                    )

        return errors

    def _detect_cycles(
        self, phases: list[PhaseTemplate],
    ) -> list[str] | None:
        """Detect dependency cycles using DFS. Returns cycle path or None."""
        adj: dict[str, list[str]] = {p.id: p.depends_on for p in phases}
        _white, _gray, _black = 0, 1, 2
        color: dict[str, int] = {pid: _white for pid in adj}
        parent: dict[str, str | None] = {pid: None for pid in adj}

        def dfs(node: str) -> list[str] | None:
            color[node] = _gray
            for neighbor in adj.get(node, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == _gray:
                    # Found a cycle — reconstruct path
                    cycle = [neighbor, node]
                    cur: str | None = node
                    while cur and parent.get(cur) and parent[cur] != neighbor:
                        cur = parent[cur]
                        if cur:
                            cycle.append(cur)
                    cycle.append(neighbor)
                    return list(reversed(cycle))
                if color[neighbor] == _white:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result:
                        return result
            color[node] = _black
            return None

        for node in adj:
            if color[node] == _white:
                result = dfs(node)
                if result:
                    return result
        return None

    @staticmethod
    def _quick_parse_name(yaml_file: Path) -> str | None:
        """Extract just the name field from a YAML file (fast)."""
        try:
            with open(yaml_file) as f:
                # Read just enough to find the name field
                for line in f:
                    line = line.strip()
                    if line.startswith("name:"):
                        val = line.split(":", 1)[1].strip()
                        # Strip quotes
                        return val.strip("'\"") or None
                    if line.startswith("#") or not line:
                        continue
                    # If we hit non-name content, fall back to full parse
                    break
            # Full parse fallback
            with open(yaml_file) as f:
                raw = yaml.safe_load(f)
            if raw and isinstance(raw, dict):
                return raw.get("name")
            return None
        except Exception:
            return None

    @staticmethod
    def _activate_isolated_dependencies(package_dir: Path) -> None:
        """Add isolated package dependency paths to sys.path if available."""
        deps_root = package_dir.parent / ".deps" / package_dir.name
        if not deps_root.exists():
            return

        candidates: list[Path] = []
        if sys.platform.startswith("win"):
            candidates.append(deps_root / "Lib" / "site-packages")
        else:
            lib_root = deps_root / "lib"
            for site_dir in sorted(lib_root.glob("python*/site-packages")):
                candidates.append(site_dir)

        for candidate in candidates:
            if candidate.exists():
                path_str = str(candidate)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                    logger.info(
                        "Activated isolated process dependencies: %s",
                        candidate,
                    )

    @staticmethod
    def _tool_name_from_class(tool_cls: type) -> str | None:
        """Best-effort extraction of a Tool name from a Tool subclass."""
        try:
            tool = tool_cls()
            name = getattr(tool, "name", "")
            if isinstance(name, str) and name.strip():
                return name.strip()
        except Exception:
            return None
        return None

    @staticmethod
    def _registered_tool_name_map() -> dict[str, type]:
        """Return {tool_name: class} for currently registered tool classes."""
        from loom.tools.registry import Tool

        name_map: dict[str, type] = {}
        for cls in Tool._registered_classes:
            name = ProcessLoader._tool_name_from_class(cls)
            if name:
                name_map[name] = cls
        return name_map

    @staticmethod
    def _register_bundled_tools(package_dir: Path) -> None:
        """Import and register tools from a package's tools/ directory."""
        from loom.tools.registry import Tool

        tools_dir = package_dir / "tools"
        if not tools_dir.exists() or not tools_dir.is_dir():
            return
        for py_file in tools_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            before_classes = set(Tool._registered_classes)
            before_name_map = ProcessLoader._registered_tool_name_map()
            # Include package name to avoid collisions between processes
            safe_pkg = package_dir.name.replace("-", "_")
            module_name = f"loom.processes._bundled.{safe_pkg}.{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, py_file,
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)

                    # Detect and suppress bundled tools that collide on name
                    # with already-registered tools.
                    new_classes = [
                        cls
                        for cls in Tool._registered_classes
                        if cls not in before_classes
                    ]
                    name_owner: dict[str, type] = dict(before_name_map)
                    for cls in new_classes:
                        tool_name = ProcessLoader._tool_name_from_class(cls)
                        if not tool_name:
                            continue
                        existing = name_owner.get(tool_name)
                        if existing is None:
                            name_owner[tool_name] = cls
                            continue
                        Tool._registered_classes.discard(cls)
                        logger.warning(
                            "Bundled tool '%s' from %s conflicts with "
                            "existing tool class %s.%s; skipping bundled tool. "
                            "Rename bundled tools to unique names.",
                            tool_name,
                            py_file,
                            existing.__module__,
                            existing.__name__,
                        )
            except Exception as e:
                logger.warning(
                    "Failed to load bundled tool %s: %s", py_file, e,
                )
