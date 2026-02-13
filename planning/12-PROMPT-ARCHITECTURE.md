# Spec 12: Prompt Architecture

## Overview

Prompt assembly is where most local model agents fail quietly. Loom uses a disciplined, ordered context assembly system. Each model invocation receives a purpose-built prompt with exactly the information it needs — no more, no less. Local models lack the RLHF refinement of Claude/GPT-4, so explicit constraints are critical.

## Context Assembly Order

Every prompt is assembled in this exact order. Order matters — local models weight information by position, with system prompts and early content receiving the most attention.

```
1. ROLE DEFINITION       — What this model instance is doing
2. TASK STATE            — Always-in-context YAML state (Layer 1)
3. CURRENT SUBTASK       — Specific instructions for this step
4. RETRIEVED CONTEXT     — From memory Layer 2 (only if relevant)
5. AVAILABLE TOOLS       — Only tools relevant to this subtask
6. OUTPUT FORMAT         — Exact schema expected
7. CONSTRAINTS           — What NOT to do (critical for local models)
```

## Prompt Assembler

```python
class PromptAssembler:
    def __init__(self, templates_dir: Path):
        self._templates = self._load_templates(templates_dir)

    def build_executor_prompt(
        self,
        task: Task,
        subtask: Subtask,
        memory_entries: list[MemoryEntry],
        available_tools: list[dict],
    ) -> str:
        """Assemble the full prompt for subtask execution."""
        sections = [
            self._role_section("executor"),
            self._task_state_section(task),
            self._subtask_section(subtask),
            self._memory_section(memory_entries),
            self._tools_section(available_tools),
            self._output_format_section("executor"),
            self._constraints_section("executor"),
        ]
        return "\n\n---\n\n".join(s for s in sections if s)

    def build_planner_prompt(self, task: Task) -> str:
        """Assemble prompt for task decomposition."""
        ...

    def build_replanner_prompt(
        self, goal: str, current_state: str,
        discoveries: list, errors: list, original_plan: Plan,
    ) -> str:
        """Assemble prompt for re-planning."""
        ...

    def build_extractor_prompt(
        self, subtask_id: str, tool_calls: list, model_output: str,
    ) -> str:
        """Assemble prompt for memory extraction."""
        ...

    def build_verifier_prompt(
        self, subtask: Subtask, result: str, tool_calls: list,
    ) -> str:
        """Assemble prompt for Tier 2 verification."""
        ...
```

## Prompt Templates

### Planner Prompt

```yaml
# prompts/templates/planner.yaml
role: |
  You are a task planning assistant. Your job is to decompose a complex goal
  into a sequence of concrete, independently executable subtasks.

  Each subtask must be:
  - Atomic: accomplishes one specific thing
  - Verifiable: has clear success criteria
  - Scoped: can be completed by a model with access to file tools and shell

instructions: |
  GOAL:
  {goal}

  WORKSPACE:
  {workspace_path}

  CONTEXT PROVIDED BY USER:
  {user_context}

  WORKSPACE CONTENTS:
  {workspace_listing}

  Decompose this goal into subtasks. Respond with ONLY a JSON object:
  {{
    "subtasks": [
      {{
        "id": "short-kebab-id",
        "description": "Clear description of what to do",
        "depends_on": ["ids-of-prerequisites"],
        "model_tier": 1,
        "verification_tier": 1,
        "is_critical_path": true,
        "acceptance_criteria": "How to verify this is done correctly"
      }}
    ]
  }}

constraints: |
  - Create 3-15 subtasks. Fewer is better if the task is simple.
  - The first subtask should always read/inspect the current state before making changes.
  - Do NOT create subtasks that depend on information you don't have — inspect first, plan second.
  - Each subtask ID must be unique and descriptive (e.g., "install-deps", "add-types", "run-tests").
  - Set model_tier: 1 for simple file ops, 2 for code generation, 3 for complex reasoning.
  - Set verification_tier: 1 for file checks, 2 for LLM review of complex output.
  - Mark is_critical_path: true if failure blocks all downstream work.
```

### Executor Prompt

```yaml
# prompts/templates/executor.yaml
role: |
  You are a task execution assistant. You complete ONE specific subtask using
  the available tools. You do NOT plan ahead or think about other subtasks.
  Focus entirely on the current subtask.

task_state: |
  CURRENT TASK STATE:
  {task_state_yaml}

subtask: |
  YOUR CURRENT SUBTASK:
  ID: {subtask_id}
  Description: {subtask_description}
  Acceptance Criteria: {acceptance_criteria}

memory: |
  RELEVANT CONTEXT FROM PRIOR WORK:
  {memory_entries_formatted}

output_format: |
  Complete the subtask using tool calls. When finished, provide a brief
  summary of what you accomplished (2-3 sentences max).

constraints: |
  CRITICAL RULES:
  - Complete ONLY the current subtask. Do NOT proceed to the next one.
  - Do NOT explain your reasoning unless asked. Just act.
  - Do NOT modify files outside the workspace.
  - If you cannot complete the subtask, respond with:
    {{"status": "blocked", "reason": "explanation of what's wrong"}}
  - If you discover something unexpected, mention it in your summary.
  - Do NOT fabricate file contents or data. Read actual files before editing.
  - Do NOT substitute or guess when information is missing. Report the gap.
  - Keep tool calls minimal. Read before writing. Don't rewrite entire files
    when a targeted edit suffices.
```

### Replanner Prompt

```yaml
# prompts/templates/replanner.yaml
role: |
  You are a task re-planning assistant. The original plan has encountered
  issues or new information has been discovered. Your job is to revise the
  remaining subtasks while preserving completed work.

instructions: |
  ORIGINAL GOAL:
  {goal}

  CURRENT STATE:
  {current_state_yaml}

  NEW DISCOVERIES:
  {discoveries_formatted}

  ERRORS ENCOUNTERED:
  {errors_formatted}

  ORIGINAL PLAN (for reference):
  {original_plan_formatted}

  Revise the plan. Keep completed subtasks as-is. Modify pending subtasks
  based on what has been learned. You may add, remove, or reorder pending subtasks.

  Respond with ONLY the updated JSON plan (same format as planner output).

constraints: |
  - Do NOT re-do completed subtasks.
  - Address discovered issues in the revised plan.
  - If errors suggest the approach is fundamentally wrong, propose a new approach.
  - Keep it concise. Fewer subtasks is better.
```

### Extractor Prompt

```yaml
# prompts/templates/extractor.yaml
role: |
  You are a memory extraction assistant. Extract structured information
  from completed subtask execution that would be useful for future subtasks.

instructions: |
  SUBTASK: {subtask_id}

  TOOL CALLS AND RESULTS:
  {tool_calls_formatted}

  MODEL OUTPUT:
  {model_output}

  Extract useful information as a JSON array:
  [
    {{
      "type": "decision|error|tool_result|discovery|artifact|context",
      "summary": "1-2 sentence summary (max 150 chars)",
      "detail": "full relevant content",
      "tags": "comma,separated,keywords",
      "relevance_to": "subtask-ids,this,matters,for"
    }}
  ]

constraints: |
  - Only extract entries useful for FUTURE subtasks.
  - Do NOT extract trivial information (e.g., "read a file successfully").
  - Respond with ONLY the JSON array. No explanation.
  - If nothing worth extracting, respond with: []
```

## Constraint Library

Local models need explicit negative constraints that RLHF'd models handle implicitly. These are common constraints applied across roles:

```python
# prompts/constraints.py

COMMON_CONSTRAINTS = """
- Do NOT hallucinate file contents. Always read files before referencing them.
- Do NOT assume the state of the workspace. Inspect before acting.
- Do NOT produce markdown explanations when tool calls are expected.
- If you're unsure, use a tool to check rather than guessing.
"""

EXECUTOR_CONSTRAINTS = """
- Complete ONLY the current subtask. Do NOT move to the next one.
- Do NOT explain reasoning unless explicitly asked.
- If data is missing or unclear, report the gap. Do NOT substitute or fabricate.
- Keep edits minimal and targeted. Do not rewrite files unnecessarily.
"""

SAFETY_CONSTRAINTS = """
- Do NOT modify files outside the workspace directory.
- Do NOT execute commands that delete data without explicit instruction.
- Do NOT install system-level packages.
- Do NOT access the network unless the task explicitly requires it.
"""
```

## Token Budget Management

The assembler tracks approximate token counts per section to stay within model context limits:

```python
def _estimate_tokens(self, text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4

def _trim_to_budget(self, sections: list[str], max_tokens: int) -> list[str]:
    """
    If total exceeds budget, trim from lowest-priority sections first.
    Priority (highest to lowest): role, task_state, subtask, constraints,
    output_format, tools, memory.
    Memory is trimmed first (reduce entries), then tools (remove least relevant).
    """
    ...
```

## Acceptance Criteria

- [ ] Executor prompt includes all 7 sections in correct order
- [ ] Planner produces valid subtask decomposition JSON
- [ ] Replanner preserves completed subtasks
- [ ] Extractor produces structured memory entries
- [ ] Constraints section is present in every prompt
- [ ] Memory entries are formatted concisely for context injection
- [ ] Token budget is respected (prompts don't exceed model context)
- [ ] Templates are loaded from YAML files (editable without code changes)
- [ ] Prompt assembly is testable with mock data
- [ ] Tool schemas are included only when tools are relevant
