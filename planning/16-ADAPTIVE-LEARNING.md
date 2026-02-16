# Spec 16: Adaptive Learning Memory (ALM) — Behavioral Adaptation

## Problem

Users hate repeating themselves. Every time a user corrects an AI's behavior — "be more concise", "use YAML not JSON", "our project uses PostgreSQL" — that correction should stick permanently. Current LLM systems have no memory of steering direction across conversations. Users end up training the model over and over, which is tedious and erodes trust.

But explicit corrections are the easy case. Users could just put those in a CLAUDE.md equivalent. The hard case — the one that makes ALM genuinely valuable — is learning from **implicit behavioral gaps**: the difference between what the model delivered and what the user actually wanted.

When a user writes code and the model considers the task done, but then the user says "test and lint it" — that's not a correction. It's a task instruction. The learning signal isn't in the words the user used, it's in the **trajectory**: the model thought it was done, the user needed more. That gap is where the real learning lives.

Regex can't see this. Scanning individual messages for keywords like "stop" or "instead" catches noise ("I need to stop for lunch") and misses the patterns that actually matter. An LLM scanning a single message for "behavioral steering signals" can't see it either — "test and lint it" is a task instruction in isolation, not a steering signal. The signal only becomes apparent when you look at the conversation structure: model delivered → user pushed back → gap existed.

## Vision

ALM learns from the **gaps between what the model delivered and what the user actually wanted**. Instead of scanning individual messages for correction keywords, it detects task completion points and analyzes follow-ups to extract general behavioral rules.

This produces patterns like:
- "After writing or modifying code, always run the test suite and linter before considering the task complete"
- "Include error handling in implementations without being asked"
- "Consider edge cases proactively in analysis"

These are injected into every future prompt via frequency-based reinforcement. The model adapts without being asked.

This is not coding-specific. ALM works across any domain: writing, analysis, research, planning, consulting. The behavioral patterns it captures are universal to human-AI interaction.

## What Changed from Spec 15

Spec 15 defined post-task operational learning (model success rates, retry hints, plan templates). Spec 16 adds a second learning mode:

| Aspect | Spec 15 (Operational) | Spec 16 (Behavioral) |
|--------|----------------------|---------------------|
| **Trigger** | After task completion | At task completion gaps |
| **Scope** | Autonomous tasks only | All interactions (cowork + autonomous) |
| **What's learned** | Model success, retry patterns, plan templates | Behavioral rules from implicit gaps |
| **How it's used** | Model routing, plan templates | Prompt personalization across all modes |
| **Domain** | Primarily coding | Domain-agnostic |
| **Extraction** | Mechanical (from execution history) | LLM gap analysis at task boundaries |

Both modes coexist. Operational patterns inform execution strategy. Behavioral patterns inform prompt construction.

## Architecture: Task Completion Gap Analysis

The fundamental shift: stop scanning individual messages for correction keywords. Instead, detect the gap between what the model delivered and what the user actually wanted.

### Step 1: Task Boundary Detection

When the model gives a "final-sounding" response, mark it as a **task completion point**. This is a heuristic trigger — it doesn't need to be perfect.

Signals that a response is a completion point:
- Model wraps up with summary language ("Here's the implementation", "That should do it", "Let me know if you need anything else")
- Model is not asking a follow-up question
- Model has produced a concrete deliverable (code, analysis, plan)
- In autonomous mode: task executor reports subtask complete

This is a lightweight classifier — not a gate. A false positive just means one extra LLM call that finds no gap.

### Step 2: Follow-up Classification

When the user responds after a completion point, classify the follow-up into one of three categories:

| Category | Signal | Action |
|----------|--------|--------|
| **New task** | User moved to a different topic | No gap, no learning |
| **Continuation** | User adds to, extends, or asks for more on the completed work | **Gap signal** — trigger extraction |
| **Explicit correction** | User says "no, that's wrong" or directly contradicts | **Direct correction** — still captured |

The key insight: **continuations** are where implicit learning lives. The user didn't say the model was wrong — they said it wasn't done. The model's idea of "done" didn't match the user's.

### Step 3: Gap Extraction via LLM

For continuations and corrections, ask the LLM:

```
The assistant considered the following work complete:
[summary of what the model delivered]

The user then said:
[the follow-up message]

If this follow-up represents something the assistant should have done
proactively — without being asked — describe the general behavioral rule
the assistant should learn. Frame it as a reusable instruction, not specific
to this task.

If the follow-up is genuinely a new request or a reasonable extension that
couldn't have been anticipated, output nothing.
```

This produces behavioral rules, not raw text snippets:

| User follow-up | Extracted rule |
|---------------|----------------|
| "test and lint it" | "After writing or modifying code, run the test suite and linter before considering the task complete" |
| "what about edge cases?" | "Consider edge cases proactively in analysis" |
| "can you add error handling?" | "Include error handling in implementations without being asked" |
| "now push it" | *(nothing — pushing is a reasonable separate step)* |
| "actually let's talk about something else" | *(nothing — new task)* |

### Step 4: No Regex Phase

The regex phase is dropped entirely. The LLM is strictly better at understanding intent. Regex adds complexity, false positives ("I need to stop for lunch" matching `\bstop\b`), and zero coverage of the cases that actually matter (implicit gaps).

The cost argument for regex — "zero latency fast path" — doesn't hold because the trigger is different. We're not scanning every message. We only run the LLM when a gap is detected at a task boundary, and gaps are infrequent.

### Why This Works: "test and lint it"

1. Model writes code, gives final response → **task completion point**
2. User says "test and lint it" → classified as **continuation** (same topic, not new)
3. Gap analysis: "Model finished coding but user had to explicitly ask for testing and linting"
4. Learned pattern: "After writing or modifying code, run the test suite and linter before considering the task complete"

### Why This Works: "what about edge cases?"

1. Model delivers an analysis → **task completion point**
2. User says "what about edge cases?" → **continuation**
3. Gap analysis: "Model completed analysis but didn't consider edge cases"
4. Learned pattern: "Consider edge cases proactively in analysis"

### Why This Works: Explicit Corrections Too

1. Model generates YAML config → task completion point
2. User says "No, use JSON not YAML" → classified as **explicit correction**
3. Gap analysis: "Model used YAML when user wanted JSON"
4. Learned pattern: "Use JSON instead of YAML for configuration files"

The same mechanism handles both implicit gaps and explicit corrections. One clean path instead of two.

## Pattern Types

| Pattern Type | Example | Source |
|-------------|---------|--------|
| `behavioral_gap` | "Run tests after writing code" | Implicit — user had to ask for something the model should have done |
| `behavioral_correction` | "Use JSON not YAML" | Explicit — user contradicted the model's output |
| `behavioral_preference` | "Always show the diff before committing" | Explicit or implicit — user consistently asks for the same thing |

These extend the existing `learned_patterns` table — no schema changes needed. Pattern types are prefixed with `behavioral_` to distinguish from operational patterns.

## Frequency-Based Reinforcement

Same mechanism as Spec 15: when the same behavioral pattern is observed again, its frequency increments. Higher-frequency patterns are more reliable and appear first in prompt injection.

The gap analysis is particularly powerful here. If a user says "test and lint it" after three different coding tasks, the pattern "run tests after writing code" gets reinforced each time. The model learns the user's workflow expectations.

Patterns with frequency=1 and age >90 days are pruned (same policy as operational patterns). High-frequency patterns persist indefinitely.

## Prompt Injection

Learned behaviors are injected into every prompt via a dedicated section:

```
## Learned Behaviors

The following behavioral patterns have been learned from previous
interactions. Apply them without being asked:

- After writing or modifying code, run the test suite and linter (observed 5x)
- Keep responses concise, under 3 paragraphs (observed 3x)
- Project uses PostgreSQL 15 with pgvector extension (observed 2x)
- Consider edge cases proactively in analysis
```

**Injection points:**
- **Cowork system prompt**: After role definition, before session state
- **Executor prompt**: After role definition, before task state (section 2)
- **Planner prompt**: After role definition, before instructions

The section is capped at 15 patterns (sorted by frequency) to avoid prompt bloat.

## Integration Flow

```
User sends message
    │
    ▼
Model generates response
    │
    ▼
Response delivered to user
    │
    ▼
Gap analysis engine runs (non-blocking, best-effort)
    ├── Is this a task completion point? (heuristic check)
    │     No → skip, wait for next turn
    │     Yes → mark as completion point, store summary
    │
    ▼
User sends next message
    │
    ▼
Classify follow-up
    ├── New task → no gap, clear completion point
    ├── Continuation → GAP DETECTED
    │     └── LLM gap extraction
    │           └── Store/update pattern in learned_patterns
    │                 └── Refresh behaviors section in system prompt
    └── Explicit correction → same as continuation path
```

## Implementation

### New Module: `learning/reflection.py`

```python
class GapAnalysisEngine:
    """Detects behavioral gaps at task completion boundaries."""

    async def on_turn_complete(
        self,
        user_message: str,
        assistant_response: str,
        session_id: str,
    ) -> None:
        """Called after every turn. Manages completion point tracking
        and triggers gap analysis when appropriate."""

    def _is_completion_point(self, assistant_response: str) -> bool:
        """Heuristic: does this response look like the model considers
        the task done? Checks for wrap-up language, lack of questions,
        presence of deliverables."""

    async def _classify_followup(
        self,
        user_message: str,
        completed_work_summary: str,
    ) -> FollowupType:
        """Classify user follow-up as new_task, continuation, or correction."""

    async def _extract_gap_pattern(
        self,
        completed_work_summary: str,
        user_followup: str,
        followup_type: FollowupType,
    ) -> Optional[str]:
        """Ask LLM to extract a general behavioral rule from the gap.
        Returns None if the follow-up is a reasonable new request."""
```

### Modified: `cowork/session.py`

- `CoworkSession.__init__` accepts optional `GapAnalysisEngine`
- `send()` and `send_streaming()` call `engine.on_turn_complete()` after every turn
- `_build_system_content()` includes learned behaviors section
- Gap analysis is best-effort (exceptions are swallowed)

### Modified: `prompts/assembler.py`

- `PromptAssembler.__init__` accepts optional `behaviors_section`
- Behaviors injected into planner, executor, and replanner prompts
- Inserted after role definition, before task-specific context

### Modified: `learning/manager.py`

- `store_or_update()` made public (was `_store_or_update`)
- Added `query_behavioral()` for retrieving behavioral patterns
- Updated docstring to reflect dual learning modes

## Examples Across Domains

ALM is domain-agnostic. The gap analysis mechanism works everywhere:

**Coding:**
- Model writes code → user says "test and lint it" → *"Run tests and linter after writing code"*
- Model implements feature → user says "what about error handling?" → *"Include error handling in implementations"*
- Model commits code → user says "show me the diff first" → *"Show diff before committing"*

**Writing assistance:**
- Model writes draft → user says "use Oxford comma" → *"Use Oxford comma in all writing"*
- Model delivers analysis → user says "too much hedging, be more direct" → *"Be direct, avoid hedging language"*
- Model writes report → user says "what about the methodology section?" → *"Include methodology in reports"*

**Financial analysis:**
- Model presents conclusions → user says "show your assumptions" → *"Always show assumptions before conclusions"*
- Model does a DCF → user says "add a sensitivity table" → *"Include sensitivity tables with DCF analyses"*

**Research:**
- Model cites secondary sources → user says "use primary sources" → *"Cite primary sources, not secondary"*
- Model writes summary → user says "what about limitations?" → *"Include limitations in research summaries"*

## Trade-offs

**(+) Catches implicit patterns that regex can never detect.** "Test and lint it" has no correction keywords. No regex pattern will ever match it as a learning signal. But gap analysis sees it clearly: the model thought it was done, and it wasn't.

**(+) Simpler code.** No regex phase, no merge logic between rule-based and LLM phases, no four categories of markers with 52 patterns. One mechanism: detect gap, extract rule.

**(+) More actionable patterns.** The LLM produces behavioral rules ("run tests after writing code"), not raw text snippets. These are directly useful as prompt instructions.

**(+) Fewer false positives.** No "I need to stop for lunch" matching `\bstop\b`. No "the wrong approach might be right" matching a correction pattern. Gaps are structural, not lexical.

**(-) Requires an LLM call for each detected gap.** But gaps are infrequent — only at task boundaries, not every message. A typical conversation might have 2-4 task completion points. Most follow-ups will be classified as new tasks (no LLM call needed) or quick continuations (one LLM call).

**(-) Task boundary detection needs to be reasonably accurate.** But it doesn't need to be perfect. A false positive just means one wasted LLM call that finds no gap. A false negative means we miss a learning opportunity — which is fine, the same pattern will recur and get caught next time.

**(-) Follow-up classification adds a step.** But this can be lightweight — even heuristic-based (topic similarity, presence of continuation markers). Only the gap extraction step needs LLM quality.

## Open Question: Keep Regex as a Fast Path?

**Argument for:** Blatantly explicit corrections ("no, don't ever do X", "always use Y") don't need gap analysis. They're already explicit and the user is clearly stating a rule. A fast-path regex could catch these without waiting for a task boundary.

**Argument against:** The gap analysis mechanism handles these too — they'll be classified as "explicit correction" at the next task boundary. And removing regex simplifies the system to one clean mechanism. The latency difference is negligible for explicit corrections since the user is already engaged in a multi-turn conversation.

**Decision:** Drop it. One mechanism. If a user says "always use TypeScript", that will be captured the next time the model delivers something and the user follows up. The slight delay in capture (one turn instead of immediate) is worth the architectural simplicity. If real-world usage shows this matters, we can add a fast path later — but start without it.

## Data Retention

Same policy as Spec 15:
- Behavioral patterns are aggregated via frequency, not raw traces
- Patterns with frequency=1 and age >90 days are pruned
- High-frequency patterns persist indefinitely
- `loom reset-learning` clears all patterns (operational and behavioral)
- `loom learned` lists, filters, and deletes individual patterns (CLI)
- `/learned` opens an interactive review/delete modal (TUI)

## Privacy Considerations

- All patterns are stored locally in the user's SQLite database
- No patterns are transmitted externally
- Patterns contain behavioral rules, not raw conversation content
- `loom reset-learning` provides a complete erasure mechanism
- `loom learned --delete ID` provides surgical removal of individual patterns

## Acceptance Criteria

- [x] Task completion point detection works for wrap-up responses
- [x] Follow-up classification distinguishes new tasks from continuations from corrections
- [x] Gap extraction produces general behavioral rules via LLM
- [x] Implicit patterns are captured (e.g., "test and lint it" → "run tests after writing code")
- [x] Explicit corrections are captured (e.g., "no, use JSON" → "use JSON not YAML")
- [x] Behavioral patterns are stored in learned_patterns with frequency tracking
- [ ] Learned behaviors are injected into cowork system prompt
- [ ] Learned behaviors are injected into executor and planner prompts
- [x] Pattern deduplication prevents redundant storage
- [x] High-frequency patterns appear before low-frequency ones
- [x] Gap analysis is non-blocking (failures don't affect user experience)
- [x] Works for non-coding domains (writing, analysis, planning, etc.)
- [x] No regex in the detection pipeline
- [x] Stale behavioral patterns are pruned (same policy as operational)
- [x] `loom reset-learning` clears behavioral patterns too
- [x] `loom learned` reviews patterns with type, frequency, and description (CLI)
- [x] `/learned` opens interactive review/delete modal (TUI)
- [x] Individual patterns can be deleted by ID
