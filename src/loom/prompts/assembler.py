"""Prompt assembly engine.

Builds purpose-built prompts for each model invocation with exactly
the information needed, in the correct order. Loads templates from
YAML files for editability.

When a ProcessDefinition is provided, domain-specific intelligence
(persona, phase blueprints, tool guidance, verification rules,
memory guidance) is injected into the generic templates without
changing their structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from loom.state.memory import MemoryEntry
from loom.state.task_state import Plan, Subtask, Task, TaskStateManager

if TYPE_CHECKING:
    from loom.processes.schema import ProcessDefinition


class PromptAssembler:
    """Assembles prompts from YAML templates and runtime data.

    Context assembly order (spec 12):
    1. ROLE DEFINITION
    2. LEARNED BEHAVIORS (from ALM reflection)
    3. TASK STATE
    4. CURRENT SUBTASK
    5. RETRIEVED CONTEXT (memory)
    6. AVAILABLE TOOLS
    7. OUTPUT FORMAT
    8. CONSTRAINTS

    When a process definition is loaded, domain intelligence is injected
    into these same sections — the template structure doesn't change.

    When learned behavioral patterns exist, they are injected after the
    role definition so the model sees them before any task-specific context.
    """

    SECTION_SEPARATOR = "\n\n---\n\n"

    def __init__(
        self,
        templates_dir: Path | None = None,
        process: ProcessDefinition | None = None,
        behaviors_section: str = "",
    ):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        self._templates_dir = templates_dir
        self._templates: dict[str, dict] = {}
        self._process = process
        self._behaviors_section = behaviors_section
        self._load_templates()

    @property
    def process(self) -> ProcessDefinition | None:
        return self._process

    @process.setter
    def process(self, value: ProcessDefinition | None) -> None:
        self._process = value

    @property
    def behaviors_section(self) -> str:
        return self._behaviors_section

    @behaviors_section.setter
    def behaviors_section(self, value: str) -> None:
        self._behaviors_section = value

    def _load_templates(self) -> None:
        """Load all YAML templates from the templates directory."""
        if not self._templates_dir.exists():
            return
        for yaml_file in self._templates_dir.glob("*.yaml"):
            name = yaml_file.stem
            with open(yaml_file) as f:
                self._templates[name] = yaml.safe_load(f) or {}

    def get_template(self, name: str) -> dict:
        """Get a loaded template by name."""
        if name not in self._templates:
            raise KeyError(f"Template not found: {name}")
        return self._templates[name]

    # --- Prompt Builders ---

    def build_planner_prompt(
        self,
        task: Task,
        workspace_listing: str = "",
        code_analysis: str = "",
        workspace_analysis: str = "",
    ) -> str:
        """Assemble prompt for task decomposition."""
        template = self.get_template("planner")

        role = template.get("role", "").strip()

        # PROCESS: persona overrides generic role
        if self._process and self._process.persona:
            role = self._process.persona.strip()

        # PROCESS: build phase blueprint
        phase_blueprint = ""
        if self._process and self._process.has_phases():
            phase_blueprint = self._format_phase_blueprint()

        # PROCESS: build planner examples
        planner_examples = ""
        if self._process and self._process.planner_examples:
            planner_examples = self._format_planner_examples()

        # PROCESS: workspace analysis from process guidance
        if (
            self._process
            and self._process.workspace_guidance
            and not workspace_analysis
        ):
            workspace_analysis = self._process.workspace_guidance

        # Use manual replacement to prevent KeyError from user braces
        raw = template.get("instructions", "")
        replacements = {
            "goal": task.goal,
            "workspace_path": task.workspace,
            "user_context": (
                json.dumps(task.context) if task.context
                else "None provided."
            ),
            "workspace_listing": workspace_listing or "Not yet inspected.",
            "code_analysis": code_analysis or "Not analyzed.",
            "workspace_analysis": workspace_analysis or "",
            "phase_blueprint": phase_blueprint,
            "planner_examples": planner_examples,
        }
        instructions = raw
        for key, value in replacements.items():
            instructions = instructions.replace(
                "{" + key + "}", str(value),
            )
        instructions = instructions.strip()
        constraints = template.get("constraints", "").strip()

        sections = [role, self._behaviors_section, instructions, constraints]
        return self.SECTION_SEPARATOR.join(s for s in sections if s)

    def build_executor_prompt(
        self,
        task: Task,
        subtask: Subtask,
        state_manager: TaskStateManager,
        memory_entries: list[MemoryEntry] | None = None,
        available_tools: list[dict] | None = None,
    ) -> str:
        """Assemble the full prompt for subtask execution.

        Follows the 7-section context assembly order.
        """
        template = self.get_template("executor")

        # 1. ROLE
        role = template.get("role", "").strip()
        # PROCESS: persona overrides generic role
        if self._process and self._process.persona:
            role = self._process.persona.strip()

        # 2. TASK STATE
        task_state_yaml = state_manager.to_compact_yaml(task)
        task_state = template.get(
            "task_state",
            "CURRENT TASK STATE:\n{task_state_yaml}",
        ).format(
            task_state_yaml=task_state_yaml,
        ).strip()

        # 3. CURRENT SUBTASK
        subtask_section = template.get("subtask", "").format(
            subtask_id=subtask.id,
            subtask_description=subtask.description,
            acceptance_criteria=(
                subtask.acceptance_criteria
                or "Complete the described task."
            ),
        ).strip()
        # PROCESS: enforce exact deliverable filenames for active phase/subtask.
        if self._process:
            deliverables = self._expected_deliverables_for_subtask(subtask.id)
            if deliverables:
                names = "\n".join(f"- {name}" for name in deliverables)
                subtask_section += (
                    "\n\nREQUIRED OUTPUT FILES (EXACT FILENAMES):\n"
                    f"{names}\n"
                    "You MUST create/update these exact filenames before "
                    "finishing this subtask. Do not rename them."
                )

        # 4. RETRIEVED CONTEXT (memory)
        if memory_entries:
            memory_text = self._format_memory(memory_entries)
        else:
            memory_text = "No relevant prior context."
        memory_section = template.get(
            "memory",
            "RELEVANT CONTEXT FROM PRIOR WORK:\n{memory_entries_formatted}",
        ).format(memory_entries_formatted=memory_text).strip()

        # 5. AVAILABLE TOOLS
        tools_section = ""
        if available_tools:
            tools_section = self._format_tools(available_tools)

        # 6. OUTPUT FORMAT
        output_format = template.get("output_format", "").strip()

        # 7. CONSTRAINTS
        constraints = template.get("constraints", "").strip()

        # PROCESS: inject tool guidance into constraints
        if self._process and self._process.tool_guidance:
            tool_guidance = (
                "\n\nDOMAIN-SPECIFIC TOOL GUIDANCE:\n"
                + self._process.tool_guidance.strip()
            )
            constraints = constraints + tool_guidance

        sections = [
            role,
            self._behaviors_section,  # Learned behaviors (from ALM reflection)
            task_state,
            subtask_section,
            memory_section,
            tools_section,
            output_format,
            constraints,
        ]
        prompt = self.SECTION_SEPARATOR.join(s for s in sections if s)

        return self._trim_to_budget(prompt)

    def _expected_deliverables_for_subtask(self, subtask_id: str) -> list[str]:
        """Return expected deliverable filenames for the given subtask id."""
        if not self._process:
            return []
        deliverables = self._process.get_deliverables()
        if not deliverables:
            return []
        if subtask_id in deliverables:
            return deliverables[subtask_id]
        if len(deliverables) == 1:
            return next(iter(deliverables.values()))
        return []

    def build_replanner_prompt(
        self,
        goal: str,
        current_state_yaml: str,
        discoveries: list[str],
        errors: list[str],
        original_plan: Plan,
        replan_reason: str = "",
    ) -> str:
        """Assemble prompt for re-planning."""
        template = self.get_template("replanner")

        role = template.get("role", "").strip()
        # PROCESS: persona overrides replanner role too
        if self._process and self._process.persona:
            role = self._process.persona.strip()

        plan_lines = []
        for s in original_plan.subtasks:
            plan_lines.append(
                f"- [{s.status.value}] {s.id}: {s.description}",
            )
        original_plan_formatted = (
            "\n".join(plan_lines) if plan_lines else "No prior plan."
        )
        process_replanning_triggers = "None provided."
        if self._process and self._process.replanning_triggers.strip():
            process_replanning_triggers = self._process.replanning_triggers.strip()

        instructions = template.get("instructions", "").format(
            goal=goal,
            current_state_yaml=current_state_yaml,
            replan_reason=replan_reason or "subtask failures",
            process_replanning_triggers=process_replanning_triggers,
            discoveries_formatted=(
                "\n".join(f"- {d}" for d in discoveries) if discoveries
                else "None."
            ),
            errors_formatted=(
                "\n".join(f"- {e}" for e in errors) if errors
                else "None."
            ),
            original_plan_formatted=original_plan_formatted,
        ).strip()

        constraints = template.get("constraints", "").strip()

        # PROCESS: inject replanning guidance
        if self._process and self._process.replanning_guidance:
            replan_extra = (
                "\n\nDOMAIN-SPECIFIC REPLANNING GUIDANCE:\n"
                + self._process.replanning_guidance.strip()
            )
            constraints = constraints + replan_extra

        sections = [role, instructions, constraints]
        return self.SECTION_SEPARATOR.join(s for s in sections if s)

    def build_extractor_prompt(
        self,
        subtask_id: str,
        tool_calls_formatted: str,
        model_output: str,
    ) -> str:
        """Assemble prompt for memory extraction."""
        template = self.get_template("extractor")

        role = template.get("role", "").strip()
        instructions = template.get("instructions", "").format(
            subtask_id=subtask_id,
            tool_calls_formatted=tool_calls_formatted,
            model_output=model_output,
        ).strip()
        constraints = template.get("constraints", "").strip()

        # PROCESS: inject extraction guidance and memory types
        if self._process:
            extra_parts = []
            if self._process.memory_types:
                types_text = ", ".join(
                    f"{m.type} ({m.description})"
                    for m in self._process.memory_types
                )
                extra_parts.append(
                    f"DOMAIN-SPECIFIC MEMORY TYPES: {types_text}"
                )
            if self._process.extraction_guidance:
                extra_parts.append(
                    "EXTRACTION GUIDANCE:\n"
                    + self._process.extraction_guidance.strip()
                )
            if extra_parts:
                constraints = (
                    constraints + "\n\n" + "\n\n".join(extra_parts)
                )

        sections = [role, instructions, constraints]
        return self.SECTION_SEPARATOR.join(s for s in sections if s)

    def build_verifier_prompt(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls_formatted: str,
    ) -> str:
        """Assemble prompt for Tier 2 LLM verification."""
        template = self.get_template("verifier")

        role = template.get("role", "").strip()
        instructions = template.get("instructions", "").format(
            subtask_id=subtask.id,
            subtask_description=subtask.description,
            acceptance_criteria=(
                subtask.acceptance_criteria
                or "Complete the described task."
            ),
            result_summary=result_summary,
            tool_calls_formatted=tool_calls_formatted,
        ).strip()
        constraints = template.get("constraints", "").strip()

        # PROCESS: inject LLM verification rules
        if self._process and self._process.has_verification_rules():
            llm_rules = self._process.llm_rules()
            if llm_rules:
                rules_text = "\n".join(
                    f"- [{r.severity.upper()}] {r.name}: {r.check}"
                    for r in llm_rules
                )
                instructions += (
                    "\n\nADDITIONAL DOMAIN-SPECIFIC CHECKS:\n"
                    + rules_text
                )

        sections = [role, instructions, constraints]
        return self.SECTION_SEPARATOR.join(s for s in sections if s)

    # --- Process formatting helpers ---

    def _format_phase_blueprint(self) -> str:
        """Format process phases as a planner blueprint."""
        if not self._process or not self._process.phases:
            return ""

        mode_desc = {
            "strict": (
                "You MUST follow this phase structure exactly. "
                "Do not add, remove, or reorder phases."
            ),
            "guided": (
                "Use this as your starting blueprint. You may adapt "
                "it (add, remove, modify phases) based on the specific "
                "goal, but explain any deviations."
            ),
            "suggestive": (
                "These phases are suggestions only. Decompose the goal "
                "however you see fit."
            ),
        }

        lines = [
            f"Phase mode: {self._process.phase_mode} — "
            + mode_desc.get(self._process.phase_mode, ""),
            "",
        ]

        for phase in self._process.phases:
            deps = (
                ", ".join(phase.depends_on) if phase.depends_on
                else "none"
            )
            flags = []
            if phase.is_critical_path:
                flags.append("critical")
            if phase.is_synthesis:
                flags.append("synthesis")
            flags_text = f", flags: {', '.join(flags)}" if flags else ""
            lines.append(f"- {phase.id} (depends: {deps}, "
                         f"tier: {phase.model_tier}{flags_text})")
            lines.append(f"  {phase.description.strip()}")
            if phase.acceptance_criteria:
                lines.append(
                    f"  Acceptance: {phase.acceptance_criteria.strip()}",
                )
            if phase.deliverables:
                lines.append(
                    f"  Deliverables: {', '.join(phase.deliverables)}",
                )
            lines.append("")

        return "\n".join(lines)

    def _format_planner_examples(self) -> str:
        """Format few-shot planner examples."""
        if not self._process or not self._process.planner_examples:
            return ""

        lines = ["EXAMPLE DECOMPOSITIONS:", ""]
        for ex in self._process.planner_examples:
            lines.append(f"Goal: {ex.goal}")
            lines.append("Subtasks:")
            for s in ex.subtasks:
                sid = s.get("id", "?")
                desc = s.get("description", "?")
                deps = s.get("depends_on", [])
                deps_str = ", ".join(deps) if deps else "none"
                lines.append(
                    f"  - {sid} (depends: {deps_str}): {desc}",
                )
            lines.append("")

        return "\n".join(lines)

    # --- Helpers ---

    def _format_memory(self, entries: list[MemoryEntry]) -> str:
        """Default memory formatting for prompt injection."""
        if not entries:
            return "No relevant prior context."
        lines = []
        for entry in entries:
            prefix = f"[{entry.entry_type}]"
            if entry.subtask_id:
                prefix += f" (from {entry.subtask_id})"
            lines.append(f"{prefix} {entry.summary}")
            if entry.detail and len(entry.detail) <= 200:
                lines.append(f"  Detail: {entry.detail}")
        return "\n".join(lines)

    def _format_tools(self, tools: list[dict]) -> str:
        """Format tool schemas for prompt injection."""
        lines = ["AVAILABLE TOOLS:"]
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            lines.append(f"\n{name}: {desc}")
            params = tool.get("parameters", {})
            if params:
                lines.append(
                    f"  Parameters: {json.dumps(params, indent=2)}",
                )
        return "\n".join(lines)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        from loom.utils.tokens import estimate_tokens
        return estimate_tokens(text)

    def _trim_to_budget(
        self, prompt: str, max_tokens: int = 8000,
    ) -> str:
        """Return prompt as-is.

        Model-facing compaction is handled semantically at runtime by the
        engine (LLM-based compactor). Avoid hard truncation during assembly.
        """
        return prompt
