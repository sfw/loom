# Spec 15: Learning System

## Overview

Loom improves over time by tracking which subtask patterns succeed or fail, with which models, with which prompts. This Phase 2 feature transforms execution history into a persistent knowledge base that informs future task planning, model selection, and prompt construction.

## What Gets Learned

| Pattern Type | Example | How It's Used |
|--------------|---------|---------------|
| `subtask_success` | "SQL migration subtasks succeed 92% with Qwen3 14B" | Model routing preference |
| `model_failure` | "Qwen3 8B fails at regex generation >40% of the time" | Escalation shortcut |
| `prompt_effective` | "Including 'read before writing' constraint reduced file errors by 60%" | Prompt template improvement |
| `task_template` | "Express→TypeScript migration: known-good 7-step plan" | Cold start templates |
| `user_correction` | "User always prefers YAML over JSON for configs" | Prompt personalization |
| `retry_pattern` | "Schema conversion subtasks need tier 2 model on first try" | Skip failed tier |

## Learned Patterns Database

```sql
-- Already defined in Spec 03 schema.sql
CREATE TABLE learned_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,
    pattern_key TEXT NOT NULL,          -- Searchable identifier
    data TEXT NOT NULL,                  -- JSON with details
    frequency INTEGER NOT NULL DEFAULT 1,
    last_seen TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

## Pattern Extraction

After every task completes (or fails), extract patterns from the execution history:

```python
class LearningManager:
    async def learn_from_task(self, task: Task) -> list[LearnedPattern]:
        """
        Analyze a completed task and extract reusable patterns.
        Runs after task completion, not during execution.
        """
        patterns = []

        # 1. Model success rates per subtask type
        for subtask in task.plan.subtasks:
            if subtask.result:
                patterns.append(LearnedPattern(
                    pattern_type="subtask_success",
                    pattern_key=self._subtask_type_key(subtask),
                    data={
                        "model": subtask.result.model_used,
                        "success": subtask.result.status == "success",
                        "retries": subtask.retry_count,
                        "duration": subtask.result.duration_seconds,
                        "tokens": subtask.result.tokens_used,
                    },
                ))

        # 2. Retry patterns — which subtasks needed escalation?
        escalated = [s for s in task.plan.subtasks if s.retry_count > 0]
        for subtask in escalated:
            patterns.append(LearnedPattern(
                pattern_type="retry_pattern",
                pattern_key=self._subtask_type_key(subtask),
                data={
                    "retries_needed": subtask.retry_count,
                    "final_model_tier": subtask.result.model_used if subtask.result else None,
                    "failure_reasons": self._extract_failure_reasons(subtask),
                },
            ))

        # 3. User corrections — what did the human steer?
        corrections = await self._memory.query(task.id, entry_type="user_instruction")
        for correction in corrections:
            patterns.append(LearnedPattern(
                pattern_type="user_correction",
                pattern_key=correction.tags,
                data={"instruction": correction.detail},
            ))

        # 4. Successful task plans as templates
        if task.status == "completed":
            patterns.append(LearnedPattern(
                pattern_type="task_template",
                pattern_key=self._goal_type_key(task.goal),
                data={
                    "goal": task.goal,
                    "plan": [{"id": s.id, "description": s.description} for s in task.plan.subtasks],
                    "total_duration": sum(s.result.duration_seconds for s in task.plan.subtasks if s.result),
                },
            ))

        # Store patterns (merge with existing)
        for pattern in patterns:
            await self._store_or_update(pattern)

        return patterns

    def _subtask_type_key(self, subtask: Subtask) -> str:
        """
        Generate a type key for a subtask based on its description.
        Groups similar subtasks together.
        E.g., "install-dependencies", "run-tests", "add-type-annotations"
        """
        # Use the subtask ID as the key (already descriptive kebab-case)
        return subtask.id

    def _goal_type_key(self, goal: str) -> str:
        """
        Generate a type key for a task goal.
        Groups similar goals together for template matching.
        """
        # Simple keyword extraction. Phase 3 could use embeddings.
        keywords = sorted(set(goal.lower().split()))[:5]
        return "-".join(keywords)
```

## Using Learned Patterns

### Informed Model Selection

```python
class ModelRouter:
    async def select_informed(self, subtask: Subtask, role: str) -> ModelProvider:
        """
        Select model considering learned patterns.
        If we know this subtask type fails with tier 1, start at tier 2.
        """
        patterns = await self._learning.query(
            pattern_type="retry_pattern",
            pattern_key=subtask.id,
        )

        min_tier = subtask.model_tier
        if patterns:
            avg_retries = sum(p.data["retries_needed"] for p in patterns) / len(patterns)
            if avg_retries > 1.5:
                min_tier = max(min_tier, 2)  # Skip tier 1, it usually fails

        return self.select(tier=min_tier, role=role)
```

### Template Matching for Planning

```python
class Planner:
    async def plan_with_templates(self, task: Task) -> Plan:
        """
        Before asking the model to plan from scratch, check if we have
        a known-good template for a similar task.
        """
        templates = await self._learning.query(
            pattern_type="task_template",
            pattern_key=self._goal_type_key(task.goal),
        )

        if templates:
            # Provide the best template as a reference in the planning prompt
            best = max(templates, key=lambda t: t.frequency)
            return await self._plan_with_reference(task, best.data["plan"])
        else:
            return await self._plan_from_scratch(task)
```

## Cold Start

When Loom has no execution history, everything runs at baseline reliability. Cold start mitigations:

### Built-in Templates

Ship with pre-built task templates in `templates/`:

```yaml
# templates/code_refactor.yaml
name: "Code Refactoring"
keywords: ["refactor", "restructure", "reorganize", "clean up"]
plan:
  - id: inspect-codebase
    description: "Read and understand current code structure"
    model_tier: 1
    verification_tier: 1
  - id: identify-targets
    description: "List specific files and functions to refactor"
    model_tier: 2
    depends_on: [inspect-codebase]
  - id: run-tests-before
    description: "Run existing tests to establish baseline"
    model_tier: 1
    depends_on: [inspect-codebase]
  - id: refactor
    description: "Apply refactoring changes"
    model_tier: 2
    depends_on: [identify-targets, run-tests-before]
  - id: run-tests-after
    description: "Run tests to verify no regressions"
    model_tier: 1
    depends_on: [refactor]
```

### Conservative Defaults

On cold start:
- Higher verification tiers (tier 2 by default)
- Lower auto-approval thresholds (confidence ≥ 0.9 to auto-proceed)
- Smaller subtask granularity (more steps, each simpler)
- More frequent re-planning gates

As the learning database grows, these can be relaxed.

## Data Retention

- Patterns are aggregated, not raw traces. Storage is compact.
- Patterns older than 90 days with frequency=1 are pruned.
- High-frequency patterns are kept indefinitely.
- Users can clear the learning database: `loom reset-learning`
- Users can review and selectively delete patterns: `loom learned` (CLI) or `/learned` (TUI)

## Acceptance Criteria

- [x] Patterns are extracted after every task completion
- [x] Model success rates are tracked per subtask type
- [x] Retry patterns inform future model selection (skip known-bad tiers)
- [x] User corrections are captured and available for prompt personalization
- [x] Successful task plans are stored as reusable templates
- [x] Template matching works for similar goals
- [ ] Built-in templates are loaded at startup
- [ ] Cold start uses conservative defaults
- [x] Learning database is persistent across restarts
- [x] `loom reset-learning` clears all learned patterns
- [x] `loom learned` lists, filters, and deletes individual patterns (CLI)
- [x] `/learned` opens an interactive review/delete modal (TUI)
- [x] Pattern frequency updates correctly on repeat observations
- [x] Stale low-frequency patterns are pruned
