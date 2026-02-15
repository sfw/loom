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
    replanning_triggers: str = ""
    replanning_guidance: str = ""

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
        """List all available processes with metadata."""
        result = []
        for name, path in sorted(self.discover().items()):
            try:
                defn = self._load_from_path(path)
                result.append({
                    "name": defn.name,
                    "version": defn.version,
                    "description": defn.description,
                    "author": defn.author,
                    "path": str(path),
                })
            except Exception:
                result.append({
                    "name": name,
                    "version": "?",
                    "description": "(failed to load)",
                    "path": str(path),
                })
        return result

    # --- Internal ---

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
            rules.append(VerificationRule(
                name=r.get("name", ""),
                description=r.get("description", ""),
                check=r.get("check", ""),
                severity=r.get("severity", "warning"),
                type=r.get("type", "llm"),
                target=r.get("target", "output"),
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
            replanning_triggers=replan_triggers,
            replanning_guidance=replan_guidance,
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
                    cur = node
                    while parent.get(cur) and parent[cur] != neighbor:
                        cur = parent[cur]  # type: ignore[assignment]
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
    def _register_bundled_tools(package_dir: Path) -> None:
        """Import and register tools from a package's tools/ directory."""
        tools_dir = package_dir / "tools"
        if not tools_dir.exists() or not tools_dir.is_dir():
            return
        for py_file in tools_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = f"loom.processes._bundled.{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, py_file,
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            except Exception as e:
                logger.warning(
                    "Failed to load bundled tool %s: %s", py_file, e,
                )
