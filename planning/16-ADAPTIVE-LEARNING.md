# Spec 16: Adaptive Learning Memory (ALM) — Behavioral Adaptation

## Problem

Users hate repeating themselves. Every time a user corrects an AI's behavior — "be more concise", "use YAML not JSON", "our project uses PostgreSQL" — that correction should stick permanently. Current LLM systems have no memory of steering direction across conversations. Users end up training the model over and over, which is tedious and erodes trust.

## Vision

ALM makes behavioral adaptation automatic and seamless. After every user prompt, the system reflects on the exchange to detect steering signals — corrections, preferences, domain knowledge, communication style cues. These signals are stored as learned patterns that accumulate over time via frequency-based reinforcement. Learned behaviors are injected into every future prompt, so the model adapts without being asked.

This is not coding-specific. ALM works across any domain: writing, analysis, research, planning, consulting. The behavioral patterns it captures are universal to human-AI interaction.

## What Changed from Spec 15

Spec 15 defined post-task operational learning (model success rates, retry hints, plan templates). Spec 16 adds a second learning mode:

| Aspect | Spec 15 (Operational) | Spec 16 (Behavioral) |
|--------|----------------------|---------------------|
| **Trigger** | After task completion | After every user prompt |
| **Scope** | Autonomous tasks only | All interactions (cowork + autonomous) |
| **What's learned** | Model success, retry patterns, plan templates | Preferences, corrections, style, domain knowledge |
| **How it's used** | Model routing, plan templates | Prompt personalization across all modes |
| **Domain** | Primarily coding | Domain-agnostic |
| **Extraction** | Mechanical (from execution history) | Hybrid (rule-based + LLM-assisted) |

Both modes coexist. Operational patterns inform execution strategy. Behavioral patterns inform prompt construction.

## Architecture

### Two-Phase Extraction

Reflection runs after every user turn in two phases:

**Phase 1: Rule-based (zero latency)**
Fast regex scan for obvious steering markers in the user's message:
- **Correction markers**: "no", "don't", "instead", "actually", "I meant", "that's wrong"
- **Preference markers**: "always", "never", "prefer", "from now on", "by default"
- **Style markers**: "be more concise", "too verbose", "more detail", "get to the point"
- **Knowledge markers**: "we use", "our team", "our convention", "for context"

**Phase 2: LLM-assisted (when utility model available)**
Uses the small/utility model to extract nuanced signals the rules miss.
Prompt asks the model to identify behavioral patterns and output structured JSON.
Only adds ~100-200ms with a small local model.

### Pattern Types

| Pattern Type | Example | Storage Key |
|-------------|---------|-------------|
| `behavioral_correction` | "Don't add docstrings unless asked" | Normalized text |
| `behavioral_preference` | "Always use TypeScript, never JavaScript" | Normalized text |
| `behavioral_style` | "Keep responses under 3 paragraphs" | Normalized text |
| `behavioral_knowledge` | "Our API uses OAuth2 with JWT tokens" | Normalized text |

These extend the existing `learned_patterns` table — no schema changes needed. Pattern types are prefixed with `behavioral_` to distinguish from operational patterns.

### Frequency-Based Reinforcement

Same mechanism as Spec 15: when the same behavioral pattern is observed again, its frequency increments. Higher-frequency patterns are more reliable and appear first in prompt injection. A user correcting the same behavior multiple times makes that pattern "stick harder."

Patterns with frequency=1 and age >90 days are pruned (same policy as operational patterns). High-frequency patterns persist indefinitely.

### Prompt Injection

Learned behaviors are injected into every prompt via a dedicated section:

```
## Learned Preferences

The following behavioral patterns have been learned from previous
interactions. Apply them without being asked:

- Use YAML instead of JSON for configuration files (observed 5x)
- Keep responses concise, under 3 paragraphs (observed 3x)
- Project uses PostgreSQL 15 with pgvector extension (observed 2x)
- Prefer functional programming style over classes
```

**Injection points:**
- **Cowork system prompt**: After role definition, before session state
- **Executor prompt**: After role definition, before task state (section 2)
- **Planner prompt**: After role definition, before instructions

The section is capped at 15 patterns (sorted by frequency) to avoid prompt bloat.

### Integration Flow

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
Reflection engine runs (non-blocking, best-effort)
    ├── Phase 1: Rule-based signal detection
    ├── Phase 2: LLM-assisted extraction (if model available)
    ├── Store/update patterns in learned_patterns table
    └── Refresh behaviors section in system prompt
    │
    ▼
System prompt updated with any new learned behaviors
    │
    ▼
Next user message benefits from updated behaviors
```

## Implementation

### New Module: `learning/reflection.py`

```python
class ReflectionEngine:
    """Analyzes user-assistant exchanges after every turn."""

    async def reflect_on_turn(
        self,
        user_message: str,
        assistant_response: str,
        previous_assistant_message: str,
        session_id: str,
    ) -> ReflectionResult:
        """Detect steering signals and store behavioral patterns."""
```

### Modified: `cowork/session.py`

- `CoworkSession.__init__` accepts optional `ReflectionEngine`
- `send()` and `send_streaming()` call `_reflect()` after every turn
- `_build_system_content()` includes learned behaviors section
- `_reflect()` is best-effort (exceptions are swallowed)

### Modified: `prompts/assembler.py`

- `PromptAssembler.__init__` accepts optional `behaviors_section`
- Behaviors injected into planner, executor, and replanner prompts
- Inserted after role definition, before task-specific context

### Modified: `learning/manager.py`

- `store_or_update()` made public (was `_store_or_update`)
- Added `query_behavioral()` for retrieving behavioral patterns
- Updated docstring to reflect dual learning modes

## Signal Detection Examples

### Corrections
```
User: "No, don't add type hints everywhere. Only add them to public APIs."
→ behavioral_correction: "Only add type hints to public APIs, not everywhere"

User: "Actually, use tabs not spaces for this project."
→ behavioral_correction: "Use tabs instead of spaces"
```

### Preferences
```
User: "Always include a test when you add a new function."
→ behavioral_preference: "Always include tests with new functions"

User: "I prefer seeing the diff before you commit."
→ behavioral_preference: "Show diff before committing"
```

### Style
```
User: "Be more concise. I don't need the explanation, just the code."
→ behavioral_style: "Be concise, provide code without lengthy explanations"

User: "Can you use bullet points instead of paragraphs?"
→ behavioral_style: "Use bullet points instead of paragraphs"
```

### Domain Knowledge
```
User: "We use Kubernetes with Istio service mesh in production."
→ behavioral_knowledge: "Production uses Kubernetes with Istio service mesh"

User: "Our API versioning convention is /api/v{n}/ in the URL."
→ behavioral_knowledge: "API versioning uses /api/v{n}/ URL convention"
```

## Non-Coding Examples

ALM is domain-agnostic. Examples from other domains:

**Writing assistance:**
- "Use Oxford comma" → behavioral_preference
- "Our brand voice is professional but approachable" → behavioral_knowledge
- "Don't use passive voice" → behavioral_correction

**Financial analysis:**
- "Always show assumptions before conclusions" → behavioral_preference
- "Our fiscal year starts April 1" → behavioral_knowledge
- "Include sensitivity tables with every DCF" → behavioral_preference

**Research:**
- "Cite primary sources, not secondary" → behavioral_preference
- "We follow APA 7th edition" → behavioral_knowledge
- "Too much hedging language, be more direct" → behavioral_style

**Project management:**
- "Our sprint cadence is 2 weeks" → behavioral_knowledge
- "Always estimate in story points, not hours" → behavioral_preference
- "Be more specific with action items" → behavioral_style

## Data Retention

Same policy as Spec 15:
- Behavioral patterns are aggregated via frequency, not raw traces
- Patterns with frequency=1 and age >90 days are pruned
- High-frequency patterns persist indefinitely
- `loom reset-learning` clears all patterns (operational and behavioral)

## Privacy Considerations

- All patterns are stored locally in the user's SQLite database
- No patterns are transmitted externally
- Patterns contain the user's steering direction, not raw conversation content
- `loom reset-learning` provides a complete erasure mechanism

## Acceptance Criteria

- [ ] Reflection runs automatically after every user prompt in cowork mode
- [ ] Rule-based detection catches correction, preference, style, and knowledge signals
- [ ] LLM-assisted extraction works when utility model is available
- [ ] Behavioral patterns are stored in learned_patterns with frequency tracking
- [ ] Learned behaviors are injected into cowork system prompt
- [ ] Learned behaviors are injected into executor and planner prompts
- [ ] Pattern deduplication prevents redundant storage
- [ ] High-frequency patterns appear before low-frequency ones
- [ ] Reflection is non-blocking (failures don't affect user experience)
- [ ] Works for non-coding domains (writing, analysis, planning, etc.)
- [ ] Stale behavioral patterns are pruned (same policy as operational)
- [ ] `loom reset-learning` clears behavioral patterns too
