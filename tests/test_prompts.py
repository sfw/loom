"""Tests for prompt assembly and templates."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.prompts.assembler import PromptAssembler
from loom.prompts.constraints import (
    COMMON_CONSTRAINTS,
    EXECUTOR_CONSTRAINTS,
    SAFETY_CONSTRAINTS,
    get_constraints_for_role,
)
from loom.state.memory import MemoryEntry
from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStateManager,
    TaskStatus,
)


@pytest.fixture
def assembler() -> PromptAssembler:
    return PromptAssembler()


@pytest.fixture
def sample_task() -> Task:
    return Task(
        id="test-1",
        goal="Migrate Express app to TypeScript",
        status=TaskStatus.EXECUTING,
        workspace="/tmp/myapp",
        plan=Plan(
            subtasks=[
                Subtask(
                    id="install-deps",
                    description="Install TypeScript dependencies",
                    status=SubtaskStatus.COMPLETED,
                    summary="Installed typescript, ts-node, @types/node",
                ),
                Subtask(
                    id="add-tsconfig",
                    description="Create tsconfig.json with proper settings",
                    status=SubtaskStatus.RUNNING,
                    acceptance_criteria="tsconfig.json exists and is valid JSON",
                ),
                Subtask(
                    id="rename-files",
                    description="Rename .js files to .ts",
                    status=SubtaskStatus.PENDING,
                    depends_on=["add-tsconfig"],
                ),
            ],
            version=1,
        ),
    )


@pytest.fixture
def state_manager(tmp_path: Path) -> TaskStateManager:
    return TaskStateManager(tmp_path)


class TestTemplateLoading:
    """Test that YAML templates load correctly."""

    def test_loads_all_templates(self, assembler: PromptAssembler):
        assert "planner" in assembler._templates
        assert "executor" in assembler._templates
        assert "replanner" in assembler._templates
        assert "extractor" in assembler._templates
        assert "verifier" in assembler._templates

    def test_templates_have_required_fields(self, assembler: PromptAssembler):
        for name in ["planner", "executor", "replanner", "extractor", "verifier"]:
            template = assembler.get_template(name)
            assert "role" in template, f"{name} missing 'role'"
            assert "constraints" in template, f"{name} missing 'constraints'"

    def test_get_template_missing_raises(self, assembler: PromptAssembler):
        with pytest.raises(KeyError, match="Template not found"):
            assembler.get_template("nonexistent")

    def test_custom_templates_dir(self, tmp_path: Path):
        # Create a custom template
        (tmp_path / "custom.yaml").write_text("role: Custom role\nconstraints: None\n")
        asm = PromptAssembler(templates_dir=tmp_path)
        template = asm.get_template("custom")
        assert template["role"] == "Custom role"


class TestPlannerPrompt:
    """Test planner prompt assembly."""

    def test_planner_includes_goal(self, assembler: PromptAssembler, sample_task: Task):
        prompt = assembler.build_planner_prompt(sample_task)
        assert "Migrate Express app to TypeScript" in prompt

    def test_planner_includes_workspace(self, assembler: PromptAssembler, sample_task: Task):
        prompt = assembler.build_planner_prompt(sample_task)
        assert "/tmp/myapp" in prompt

    def test_planner_includes_constraints(self, assembler: PromptAssembler, sample_task: Task):
        prompt = assembler.build_planner_prompt(sample_task)
        assert "subtask ID must be unique" in prompt

    def test_planner_includes_workspace_listing(
        self, assembler: PromptAssembler, sample_task: Task
    ):
        listing = "src/\npackage.json\nREADME.md"
        prompt = assembler.build_planner_prompt(sample_task, workspace_listing=listing)
        assert "package.json" in prompt

    def test_planner_no_listing(self, assembler: PromptAssembler, sample_task: Task):
        prompt = assembler.build_planner_prompt(sample_task)
        assert "Not yet inspected" in prompt

    def test_planner_has_json_format(self, assembler: PromptAssembler, sample_task: Task):
        prompt = assembler.build_planner_prompt(sample_task)
        assert '"subtasks"' in prompt
        assert '"id"' in prompt


class TestExecutorPrompt:
    """Test executor prompt assembly."""

    def test_executor_has_all_sections(
        self,
        assembler: PromptAssembler,
        sample_task: Task,
        state_manager: TaskStateManager,
    ):
        state_manager.create(sample_task)
        subtask = sample_task.get_subtask("add-tsconfig")

        prompt = assembler.build_executor_prompt(
            task=sample_task,
            subtask=subtask,
            state_manager=state_manager,
        )

        # Check all 7 sections present
        assert "task execution assistant" in prompt  # role
        assert "CURRENT TASK STATE" in prompt  # task state
        assert "add-tsconfig" in prompt  # subtask
        assert "No relevant prior context" in prompt  # memory (empty)
        assert "CRITICAL RULES" in prompt  # constraints
        assert "summary of what you accomplished" in prompt  # output format

    def test_executor_includes_memory(
        self,
        assembler: PromptAssembler,
        sample_task: Task,
        state_manager: TaskStateManager,
    ):
        state_manager.create(sample_task)
        subtask = sample_task.get_subtask("add-tsconfig")

        memory = [
            MemoryEntry(
                task_id="test-1",
                subtask_id="install-deps",
                entry_type="decision",
                summary="Using strict TypeScript settings",
            ),
        ]

        prompt = assembler.build_executor_prompt(
            task=sample_task,
            subtask=subtask,
            state_manager=state_manager,
            memory_entries=memory,
        )

        assert "strict TypeScript" in prompt

    def test_executor_includes_tools(
        self,
        assembler: PromptAssembler,
        sample_task: Task,
        state_manager: TaskStateManager,
    ):
        state_manager.create(sample_task)
        subtask = sample_task.get_subtask("add-tsconfig")

        tools = [
            {"name": "write_file", "description": "Write content to a file"},
            {"name": "read_file", "description": "Read a file's contents"},
        ]

        prompt = assembler.build_executor_prompt(
            task=sample_task,
            subtask=subtask,
            state_manager=state_manager,
            available_tools=tools,
        )

        assert "write_file" in prompt
        assert "read_file" in prompt

    def test_executor_section_order(
        self,
        assembler: PromptAssembler,
        sample_task: Task,
        state_manager: TaskStateManager,
    ):
        state_manager.create(sample_task)
        subtask = sample_task.get_subtask("add-tsconfig")

        prompt = assembler.build_executor_prompt(
            task=sample_task,
            subtask=subtask,
            state_manager=state_manager,
        )

        # Role should come before task state
        role_pos = prompt.find("task execution assistant")
        state_pos = prompt.find("CURRENT TASK STATE")
        subtask_pos = prompt.find("YOUR CURRENT SUBTASK")
        assert role_pos < state_pos < subtask_pos

    def test_executor_includes_evidence_contract_and_ledger_snapshot(
        self,
        assembler: PromptAssembler,
        state_manager: TaskStateManager,
    ):
        subtask = Subtask(
            id="environmental-scan",
            description="Perform environmental scans for each market and synthesize risk.",
            acceptance_criteria="Deliver evidence-backed implications with source citations.",
            status=SubtaskStatus.PENDING,
        )
        task = Task(
            id="task-evidence",
            goal="Assess market risks",
            status=TaskStatus.EXECUTING,
            workspace="/tmp/market",
            plan=Plan(subtasks=[subtask], version=1),
        )
        state_manager.create(task)

        prompt = assembler.build_executor_prompt(
            task=task,
            subtask=subtask,
            state_manager=state_manager,
            evidence_ledger_summary=(
                "- EV-1234 | market=Alberta Retail Energy | "
                "dimension=economic | quality=0.85 | source=https://example.com/ab"
            ),
        )

        assert "EVIDENCE CONTRACT" in prompt
        assert "UNSUPPORTED_NO_EVIDENCE" in prompt
        assert "EVIDENCE LEDGER SNAPSHOT (PERSISTED)" in prompt
        assert "EV-1234" in prompt


class TestReplannerPrompt:
    """Test replanner prompt assembly."""

    def test_replanner_includes_goal(self, assembler: PromptAssembler, sample_task: Task):
        prompt = assembler.build_replanner_prompt(
            goal=sample_task.goal,
            current_state_yaml="status: executing",
            discoveries=["Found legacy code in src/"],
            errors=["tsconfig.json parse error"],
            original_plan=sample_task.plan,
        )
        assert "Migrate Express app to TypeScript" in prompt

    def test_replanner_includes_discoveries(self, assembler: PromptAssembler, sample_task: Task):
        prompt = assembler.build_replanner_prompt(
            goal=sample_task.goal,
            current_state_yaml="status: executing",
            discoveries=["Database uses MongoDB"],
            errors=[],
            original_plan=sample_task.plan,
        )
        assert "MongoDB" in prompt

    def test_replanner_includes_original_plan(
        self, assembler: PromptAssembler, sample_task: Task
    ):
        prompt = assembler.build_replanner_prompt(
            goal=sample_task.goal,
            current_state_yaml="status: executing",
            discoveries=[],
            errors=[],
            original_plan=sample_task.plan,
        )
        assert "install-deps" in prompt
        assert "completed" in prompt


class TestExtractorPrompt:
    """Test extractor prompt assembly."""

    def test_extractor_includes_subtask(self, assembler: PromptAssembler):
        prompt = assembler.build_extractor_prompt(
            subtask_id="install-deps",
            tool_calls_formatted="Tool: shell_execute\n  command: npm install typescript",
            model_output="Installed TypeScript successfully.",
        )
        assert "install-deps" in prompt

    def test_extractor_includes_tool_calls(self, assembler: PromptAssembler):
        prompt = assembler.build_extractor_prompt(
            subtask_id="s1",
            tool_calls_formatted="Tool: write_file\n  path: tsconfig.json",
            model_output="Created config.",
        )
        assert "write_file" in prompt
        assert "tsconfig.json" in prompt

    def test_extractor_requests_json(self, assembler: PromptAssembler):
        prompt = assembler.build_extractor_prompt("s1", "", "output")
        assert "JSON array" in prompt


class TestVerifierPrompt:
    """Test verifier prompt assembly."""

    def test_verifier_includes_criteria(self, assembler: PromptAssembler):
        subtask = Subtask(
            id="add-tsconfig",
            description="Create tsconfig.json",
            acceptance_criteria="File exists and is valid JSON",
        )
        prompt = assembler.build_verifier_prompt(
            subtask=subtask,
            result_summary="Created tsconfig.json with strict settings.",
            tool_calls_formatted="Tool: write_file\n  path: tsconfig.json",
        )
        assert "valid JSON" in prompt

    def test_verifier_requests_json_response(self, assembler: PromptAssembler):
        subtask = Subtask(id="s1", description="Test")
        prompt = assembler.build_verifier_prompt(subtask, "Done.", "")
        assert '"passed"' in prompt
        assert '"confidence"' in prompt

    def test_verifier_handles_plural_wording_without_forced_minimum(
        self, assembler: PromptAssembler,
    ):
        subtask = Subtask(id="scope-companies", description="Interpret company name(s)")
        prompt = assembler.build_verifier_prompt(subtask, "Done.", "")
        assert "cardinality-agnostic" in prompt
        assert "company name(s)" in prompt


class TestConstraints:
    """Test constraint library."""

    def test_common_constraints_exist(self):
        assert "hallucinate" in COMMON_CONSTRAINTS
        assert "workspace" in COMMON_CONSTRAINTS

    def test_executor_constraints_exist(self):
        assert "current subtask" in EXECUTOR_CONSTRAINTS

    def test_safety_constraints_exist(self):
        assert "outside the workspace" in SAFETY_CONSTRAINTS

    def test_get_constraints_executor(self):
        text = get_constraints_for_role("executor")
        assert "hallucinate" in text  # common
        assert "workspace directory" in text  # safety
        assert "current subtask" in text  # executor-specific

    def test_get_constraints_unknown_role(self):
        text = get_constraints_for_role("unknown")
        assert "hallucinate" in text  # common still included
        assert "workspace directory" in text  # safety still included


class TestTokenBudget:
    """Test token estimation and budget handling."""

    def test_estimate_tokens(self, assembler: PromptAssembler):
        text = "Hello world"  # 11 chars ~= 2-3 tokens
        tokens = assembler.estimate_tokens(text)
        assert tokens == 2  # 11 // 4

    def test_trim_within_budget(self, assembler: PromptAssembler):
        short_text = "Short prompt"
        result = assembler._trim_to_budget(short_text, max_tokens=100)
        assert result == short_text

    def test_trim_exceeds_budget(self, assembler: PromptAssembler):
        long_text = "x" * 40000  # ~10000 tokens
        result = assembler._trim_to_budget(long_text, max_tokens=1000)
        # Prompt assembly no longer hard-trims; semantic compaction is handled
        # in the runtime engine before model invocation.
        assert result == long_text
