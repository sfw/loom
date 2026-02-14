"""Prompt assembly engine.

Builds purpose-built prompts for each model invocation with exactly
the information needed, in the correct order. Loads templates from
YAML files for editability.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from loom.state.memory import MemoryEntry
from loom.state.task_state import Plan, Subtask, Task, TaskStateManager


class PromptAssembler:
    """Assembles prompts from YAML templates and runtime data.

    Context assembly order (spec 12):
    1. ROLE DEFINITION
    2. TASK STATE
    3. CURRENT SUBTASK
    4. RETRIEVED CONTEXT (memory)
    5. AVAILABLE TOOLS
    6. OUTPUT FORMAT
    7. CONSTRAINTS
    """

    SECTION_SEPARATOR = "\n\n---\n\n"

    def __init__(self, templates_dir: Path | None = None):
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        self._templates_dir = templates_dir
        self._templates: dict[str, dict] = {}
        self._load_templates()

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
    ) -> str:
        """Assemble prompt for task decomposition."""
        template = self.get_template("planner")

        role = template.get("role", "").strip()
        # Use safe_substitute to prevent KeyError from user-supplied {braces}
        from string import Template as _StrTemplate
        _tmpl = _StrTemplate(template.get("instructions", "").replace("{", "${"))
        instructions = _tmpl.safe_substitute(
            goal=task.goal,
            workspace_path=task.workspace,
            user_context=json.dumps(task.context) if task.context else "None provided.",
            workspace_listing=workspace_listing or "Not yet inspected.",
            code_analysis=code_analysis or "Not analyzed.",
        ).strip()
        constraints = template.get("constraints", "").strip()

        sections = [role, instructions, constraints]
        return self.SECTION_SEPARATOR.join(s for s in sections if s)

    def build_executor_prompt(
        self,
        task: Task,
        subtask: Subtask,
        state_manager: TaskStateManager,
        memory_entries: list[MemoryEntry] | None = None,
        available_tools: list[dict] | None = None,
        memory_formatter=None,
    ) -> str:
        """Assemble the full prompt for subtask execution.

        Follows the 7-section context assembly order.
        """
        template = self.get_template("executor")

        # 1. ROLE
        role = template.get("role", "").strip()

        # 2. TASK STATE
        task_state_yaml = state_manager.to_compact_yaml(task)
        task_state = template.get("task_state", "CURRENT TASK STATE:\n{task_state_yaml}").format(
            task_state_yaml=task_state_yaml,
        ).strip()

        # 3. CURRENT SUBTASK
        subtask_section = template.get("subtask", "").format(
            subtask_id=subtask.id,
            subtask_description=subtask.description,
            acceptance_criteria=subtask.acceptance_criteria or "Complete the described task.",
        ).strip()

        # 4. RETRIEVED CONTEXT (memory)
        if memory_entries and memory_formatter:
            memory_text = memory_formatter(memory_entries)
        elif memory_entries:
            memory_text = self._format_memory(memory_entries)
        else:
            memory_text = "No relevant prior context."
        memory_section = template.get(
            "memory", "RELEVANT CONTEXT FROM PRIOR WORK:\n{memory_entries_formatted}"
        ).format(memory_entries_formatted=memory_text).strip()

        # 5. AVAILABLE TOOLS
        tools_section = ""
        if available_tools:
            tools_section = self._format_tools(available_tools)

        # 6. OUTPUT FORMAT
        output_format = template.get("output_format", "").strip()

        # 7. CONSTRAINTS
        constraints = template.get("constraints", "").strip()

        sections = [
            role,
            task_state,
            subtask_section,
            memory_section,
            tools_section,
            output_format,
            constraints,
        ]
        prompt = self.SECTION_SEPARATOR.join(s for s in sections if s)

        return self._trim_to_budget(prompt)

    def build_replanner_prompt(
        self,
        goal: str,
        current_state_yaml: str,
        discoveries: list[str],
        errors: list[str],
        original_plan: Plan,
    ) -> str:
        """Assemble prompt for re-planning."""
        template = self.get_template("replanner")

        role = template.get("role", "").strip()

        plan_lines = []
        for s in original_plan.subtasks:
            plan_lines.append(f"- [{s.status.value}] {s.id}: {s.description}")
        original_plan_formatted = "\n".join(plan_lines) if plan_lines else "No prior plan."

        instructions = template.get("instructions", "").format(
            goal=goal,
            current_state_yaml=current_state_yaml,
            discoveries_formatted="\n".join(f"- {d}" for d in discoveries) if discoveries
            else "None.",
            errors_formatted="\n".join(f"- {e}" for e in errors) if errors else "None.",
            original_plan_formatted=original_plan_formatted,
        ).strip()

        constraints = template.get("constraints", "").strip()

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
            acceptance_criteria=subtask.acceptance_criteria or "Complete the described task.",
            result_summary=result_summary,
            tool_calls_formatted=tool_calls_formatted,
        ).strip()
        constraints = template.get("constraints", "").strip()

        sections = [role, instructions, constraints]
        return self.SECTION_SEPARATOR.join(s for s in sections if s)

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
                lines.append(f"  Parameters: {json.dumps(params, indent=2)}")
        return "\n".join(lines)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return len(text) // 4

    def _trim_to_budget(self, prompt: str, max_tokens: int = 8000) -> str:
        """Trim prompt if it exceeds the token budget.

        Priority (highest to lowest): role, task_state, subtask, constraints,
        output_format, tools, memory.
        Currently does a simple truncation. Future: section-aware trimming.
        """
        estimated = self.estimate_tokens(prompt)
        if estimated <= max_tokens:
            return prompt

        # Simple truncation to fit budget
        max_chars = max_tokens * 4
        if len(prompt) > max_chars:
            return prompt[:max_chars] + "\n\n[... prompt trimmed to fit context window ...]"
        return prompt
