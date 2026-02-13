# Spec 02: Orchestrator Loop

## Overview

The orchestrator loop is the core brain of Loom. It drives all work by repeatedly invoking models with scoped prompts, processing tool calls, and managing task progression. The model never decides to "continue" — the harness does.

## Critical Design Principle

**The loop continues as long as there are incomplete subtasks. The model's only job is to execute the current subtask. The orchestrator decides what happens next.**

This is the fundamental difference from chat-based LLM interactions. In chat, the model decides when it's done. Here, the harness drives execution based on structured task state.

## Orchestrator Lifecycle

```
User/Agent submits task via API
         │
         ▼
┌─── PLANNING PHASE ────────────────────────────┐
│ 1. Load task goal + context                    │
│ 2. Invoke planner model with decomposition     │
│    prompt                                      │
│ 3. Parse plan into structured subtasks         │
│ 4. Validate plan (are subtasks actionable?)    │
│ 5. Store plan in task state                    │
│ 6. Emit: task_plan_ready event                 │
└───────────────────┬───────────────────────────┘
                    │
         ▼ (for each subtask in dependency order)
┌─── EXECUTION PHASE ───────────────────────────┐
│                                                │
│  ┌── SUBTASK LOOP ──────────────────────────┐  │
│  │ 1. Select next subtask (scheduler)       │  │
│  │ 2. Assemble prompt (Spec 12)             │  │
│  │ 3. Select model (Spec 04)                │  │
│  │ 4. Invoke model                          │  │
│  │ 5. Parse response                        │  │
│  │    ├─ Tool call? → Execute tool → feed   │  │
│  │    │   result back → goto 4              │  │
│  │    ├─ Text only? → Subtask complete      │  │
│  │    └─ Error? → Retry/escalate (Spec 13)  │  │
│  │ 6. Verify output (Spec 06)               │  │
│  │ 7. Extract memory entries (Spec 03)      │  │
│  │ 8. Update task state                     │  │
│  │ 9. Emit: subtask_completed event         │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  ┌── RE-PLANNING GATE ─────────────────────┐   │
│  │ After every N subtasks OR any failure:   │   │
│  │ - Check if plan still makes sense        │   │
│  │ - Feed original goal + current state +   │   │
│  │   new discoveries to planner             │   │
│  │ - Modify remaining subtasks if needed    │   │
│  │ - Emit: task_replanning event            │   │
│  └──────────────────────────────────────────┘   │
│                                                │
└───────────────────┬───────────────────────────┘
                    │
         ▼
┌─── COMPLETION PHASE ──────────────────────────┐
│ 1. All subtasks complete + verified            │
│ 2. Run final validation (optional Tier 3)      │
│ 3. Generate summary                            │
│ 4. Emit: task_completed event                  │
│ 5. Notify callback URL if registered           │
└────────────────────────────────────────────────┘
```

## Core Data Structures

### Task

```python
@dataclass
class Task:
    id: str                          # UUID
    goal: str                        # Natural language goal
    context: dict                    # User-provided context (repo path, constraints, etc.)
    workspace_path: Path | None      # Mounted working directory
    status: TaskStatus               # planning | executing | verifying | completed | failed | cancelled
    plan: Plan | None                # Decomposed plan
    created_at: datetime
    updated_at: datetime
    approval_mode: str               # "auto" | "manual" | "confidence_threshold"
    callback_url: str | None         # Optional webhook for completion
    metadata: dict                   # Arbitrary metadata from submitter
```

### Plan

```python
@dataclass
class Plan:
    version: int                     # Increments on re-plan
    subtasks: list[Subtask]
    created_at: datetime
    replanned_at: datetime | None
```

### Subtask

```python
@dataclass
class Subtask:
    id: str                          # Short descriptive ID: "install-deps", "rename-files"
    description: str                 # What this subtask should accomplish
    status: SubtaskStatus            # pending | running | completed | failed | skipped | blocked
    depends_on: list[str]            # IDs of subtasks that must complete first
    model_tier: int                  # 1=fast, 2=medium, 3=large (hint for router)
    verification_tier: int           # 1=deterministic, 2=llm, 3=voting
    is_critical_path: bool           # If true, failure blocks all downstream
    result: SubtaskResult | None     # Populated on completion
    retry_count: int                 # How many times this has been retried
    max_retries: int                 # Override per-subtask (default from config)
```

### SubtaskResult

```python
@dataclass
class SubtaskResult:
    status: str                      # "success" | "failed" | "partial"
    summary: str                     # 1-2 sentence summary of what was done
    artifacts: list[str]             # Paths to output files
    tool_calls: list[ToolCallRecord] # Full record of tools invoked
    verification: VerificationResult # Outcome of verification gate
    duration_seconds: float
    tokens_used: int
    model_used: str
```

## Orchestrator Implementation

### orchestrator.py

```python
class Orchestrator:
    """
    Core orchestrator loop. Drives task execution by repeatedly
    invoking models, processing tool calls, and managing state.
    """

    def __init__(
        self,
        model_router: ModelRouter,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        verification_gates: VerificationGates,
        event_bus: EventBus,
        config: Config,
    ):
        # All dependencies injected — no global state

    async def execute_task(self, task: Task) -> TaskResult:
        """
        Main entry point. Drives the full task lifecycle:
        plan → execute subtasks → verify → complete.
        """
        # 1. Planning phase
        plan = await self._plan_task(task)
        task.plan = plan
        task.status = TaskStatus.EXECUTING

        # 2. Execution loop
        while self._has_pending_subtasks(task):
            # Check for cancellation
            if task.status == TaskStatus.CANCELLED:
                break

            # Select next runnable subtask (respects dependencies)
            subtask = self._scheduler.next_subtask(task.plan)
            if subtask is None:
                # All remaining subtasks are blocked
                break

            # Execute the subtask
            result = await self._execute_subtask(task, subtask)

            # Re-planning gate
            if self._should_replan(task, subtask, result):
                await self._replan(task)

        # 3. Completion
        return await self._finalize_task(task)
```

### Subtask Execution (Inner Loop)

```python
async def _execute_subtask(self, task: Task, subtask: Subtask) -> SubtaskResult:
    """
    Execute a single subtask. This is the inner loop that handles
    model invocation, tool calls, and verification for one subtask.
    """
    subtask.status = SubtaskStatus.RUNNING
    self._event_bus.emit(SubtaskStarted(task.id, subtask.id))

    # Assemble the prompt with fresh context (no prior subtask history)
    prompt = self._prompt_assembler.build_executor_prompt(
        task=task,
        subtask=subtask,
        memory_entries=await self._memory.query_relevant(task.id, subtask.id),
        available_tools=self._tool_registry.tools_for_subtask(subtask),
    )

    # Select model based on subtask tier
    model = self._model_router.select(subtask.model_tier, role="executor")

    # Inner tool-calling loop
    messages = [{"role": "user", "content": prompt}]
    tool_calls_record = []
    iteration = 0
    max_iterations = 20  # Safety cap per subtask

    while iteration < max_iterations:
        iteration += 1
        response = await model.complete(messages)

        if response.has_tool_calls():
            for tool_call in response.tool_calls:
                # Execute the tool
                tool_result = await self._tool_registry.execute(
                    tool_call.name,
                    tool_call.arguments,
                    workspace=task.workspace_path,
                )
                tool_calls_record.append(ToolCallRecord(
                    tool=tool_call.name,
                    args=tool_call.arguments,
                    result=tool_result,
                    timestamp=datetime.now(),
                ))
                # Feed result back to model
                messages.append({"role": "assistant", "content": response.raw})
                messages.append({"role": "tool", "content": tool_result.to_json()})

            # Inject TODO reminder after tool use (anti-amnesia)
            messages.append({
                "role": "system",
                "content": self._build_todo_reminder(task, subtask),
            })
        else:
            # Model produced text without tool calls — subtask complete
            break

    # Verify the result
    verification = await self._verification_gates.verify(
        subtask=subtask,
        result=response.text,
        tool_calls=tool_calls_record,
        workspace=task.workspace_path,
        tier=subtask.verification_tier,
    )

    # Extract memory entries (write-time extraction)
    await self._memory.extract_and_store(
        task_id=task.id,
        subtask_id=subtask.id,
        tool_calls=tool_calls_record,
        model_output=response.text,
    )

    # Build result
    result = SubtaskResult(
        status="success" if verification.passed else "failed",
        summary=response.text[:200],
        artifacts=self._collect_artifacts(tool_calls_record),
        tool_calls=tool_calls_record,
        verification=verification,
        duration_seconds=elapsed,
        tokens_used=response.usage.total_tokens,
        model_used=model.name,
    )

    # Update state
    subtask.result = result
    subtask.status = SubtaskStatus.COMPLETED if verification.passed else SubtaskStatus.FAILED
    self._task_state.save(task)
    self._event_bus.emit(SubtaskCompleted(task.id, subtask.id, result))

    return result
```

### TODO Reminder Injection (Anti-Amnesia)

After every tool call, inject the current task state as a system message. This is the single most important technique for preventing the "one subtask and done" problem.

```python
def _build_todo_reminder(self, task: Task, current_subtask: Subtask) -> str:
    """
    Build a concise reminder of what the model should be doing.
    Injected after every tool call to prevent drift.
    """
    return f"""CURRENT TASK STATE:
Goal: {task.goal}
Current subtask: {current_subtask.id} — {current_subtask.description}
Status: {current_subtask.status}

REMAINING WORK FOR THIS SUBTASK:
Continue working on: {current_subtask.description}
Do NOT move to the next subtask. Complete ONLY this one.
When finished, provide a summary of what you accomplished."""
```

### Re-Planning Logic

```python
def _should_replan(self, task: Task, subtask: Subtask, result: SubtaskResult) -> bool:
    """Decide whether to trigger re-planning."""
    # Always replan after a failure
    if result.status == "failed":
        return True

    # Replan every N completed subtasks
    completed = sum(1 for s in task.plan.subtasks if s.status == SubtaskStatus.COMPLETED)
    if completed > 0 and completed % self._config.replan_interval == 0:
        return True

    # Replan if the subtask discovered something unexpected
    # (detected by memory entries tagged as "discovery")
    discoveries = await self._memory.query(
        task_id=task.id,
        subtask_id=subtask.id,
        entry_type="discovery",
    )
    if discoveries:
        return True

    return False

async def _replan(self, task: Task):
    """
    Re-plan with fresh context. Only the structured state is provided,
    not the full conversation history. This prevents sunk-cost reasoning.
    """
    self._event_bus.emit(TaskReplanning(task.id))

    # Build re-planning prompt with ONLY:
    # - Original goal
    # - Current state (what's done, what's pending)
    # - Discoveries and errors encountered
    # - Original plan for reference
    prompt = self._prompt_assembler.build_replanner_prompt(
        goal=task.goal,
        current_state=self._task_state.to_yaml(task),
        discoveries=await self._memory.query(task.id, entry_type="discovery"),
        errors=await self._memory.query(task.id, entry_type="error"),
        original_plan=task.plan,
    )

    model = self._model_router.select(tier=3, role="planner")
    response = await model.complete([{"role": "user", "content": prompt}])

    # Parse updated plan
    updated_plan = self._parse_plan(response.text)
    updated_plan.version = task.plan.version + 1
    updated_plan.replanned_at = datetime.now()

    # Preserve completed subtask results
    for new_subtask in updated_plan.subtasks:
        old = self._find_subtask(task.plan, new_subtask.id)
        if old and old.status == SubtaskStatus.COMPLETED:
            new_subtask.status = old.status
            new_subtask.result = old.result

    task.plan = updated_plan
    self._task_state.save(task)
```

### Scheduler (Dependency Resolution)

```python
class Scheduler:
    """
    Determines which subtask to execute next based on dependency graph.
    """

    def next_subtask(self, plan: Plan) -> Subtask | None:
        """
        Return the next runnable subtask, or None if all are
        blocked/completed.

        A subtask is runnable if:
        1. Status is 'pending'
        2. All depends_on subtasks are 'completed'
        """
        for subtask in plan.subtasks:
            if subtask.status != SubtaskStatus.PENDING:
                continue
            deps_met = all(
                self._get_subtask(plan, dep_id).status == SubtaskStatus.COMPLETED
                for dep_id in subtask.depends_on
            )
            if deps_met:
                return subtask
        return None

    def runnable_subtasks(self, plan: Plan) -> list[Subtask]:
        """
        Return ALL runnable subtasks (for potential parallel execution).
        """
        # Same logic as next_subtask but collects all matches
        ...
```

## Concurrency Model

For V1, execute subtasks **sequentially**. The orchestrator processes one subtask at a time to keep things debuggable.

For V2, support **parallel execution** of independent subtasks:
- `scheduler.runnable_subtasks()` returns all subtasks whose dependencies are met
- Use `asyncio.gather()` to execute independent subtasks concurrently
- Shared blackboard (key-value store) allows cross-subtask discovery sharing

## Cancellation

The orchestrator checks for cancellation at the top of every loop iteration. When cancelled:
1. Mark all pending subtasks as `skipped`
2. Wait for currently running subtask to complete (don't kill mid-execution)
3. Emit `task_cancelled` event
4. Save final state

## Safety Limits

- `max_loop_iterations`: Hard cap on total orchestrator loop iterations (default: 50)
- `max_subtask_iterations`: Hard cap on tool-call loop per subtask (default: 20)
- `max_subtask_retries`: Per-subtask retry limit before escalation (default: 3)
- `max_plan_versions`: Maximum re-plans before flagging for human review (default: 5)
- `max_total_tokens`: Token budget per task (optional, for cost control)

## Acceptance Criteria

- [ ] Orchestrator accepts a task with a goal and produces a plan with subtasks
- [ ] Subtasks execute in dependency order
- [ ] Tool calls within a subtask loop correctly: invoke → result → re-invoke
- [ ] TODO reminders are injected after every tool call
- [ ] Model text-only response terminates the subtask inner loop
- [ ] Re-planning triggers after failures and on schedule
- [ ] Re-planning preserves completed subtask results
- [ ] Cancellation gracefully stops execution
- [ ] All safety limits are enforced
- [ ] Events are emitted at every state transition
- [ ] Full execution is replayable from event log
