"""Task completion gap analysis: learn behavioral patterns from what users had to ask for.

Instead of scanning individual messages for correction keywords (regex), this
module detects the *gap* between what the model delivered and what the user
actually wanted.  The mechanism:

1. Task boundary detection -- when the model gives a "final-sounding" response,
   mark it as a completion point.
2. Follow-up classification -- when the user responds after a completion point,
   classify it as new_task, continuation, or correction.
3. Gap extraction -- for continuations and corrections, ask the LLM to extract
   a general behavioral rule the model should learn.

This catches implicit patterns that keyword scanning can never detect.
"test and lint it" has no correction keywords, but the gap is clear: the model
thought it was done, and the user needed more.

Design principles:
- No regex.  Gaps are structural, not lexical.
- LLM calls only at task boundaries (infrequent), not every message.
- Non-blocking: gap analysis runs after the response is delivered.
- Cumulative: pattern frequency increases with repeated observations.
- Injected: learned patterns influence future prompts via system prompt.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

from loom.engine.semantic_compactor import SemanticCompactor
from loom.learning.manager import LearnedPattern, LearningManager
from loom.models.retry import ModelRetryPolicy, call_with_model_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FollowupType(enum.Enum):
    """Classification of a user follow-up after a task completion point."""

    NEW_TASK = "new_task"
    CONTINUATION = "continuation"
    CORRECTION = "correction"


@dataclass
class GapSignal:
    """A behavioral gap detected between completion and follow-up."""

    followup_type: FollowupType
    rule: str  # The general behavioral rule extracted
    completed_summary: str = ""  # What the model considered complete
    user_followup: str = ""  # What the user said next
    confidence: float = 0.7


@dataclass
class GapResult:
    """Result of gap analysis for a single turn."""

    is_completion_point: bool = False
    followup_type: FollowupType | None = None
    signal: GapSignal | None = None
    pattern: LearnedPattern | None = None


# ---------------------------------------------------------------------------
# Completion detection heuristics
# ---------------------------------------------------------------------------

# Phrases that suggest the model considers its work done.
_COMPLETION_PHRASES = [
    "let me know if",
    "let me know when",
    "hope that helps",
    "should do it",
    "that should",
    "does that work",
    "anything else",
    "if you need",
    "feel free to",
    "here's the",
    "here is the",
    "i've completed",
    "i've finished",
    "i've updated",
    "i've implemented",
    "i've added",
    "i've created",
    "i've fixed",
    "i've made",
    "all done",
    "that's it",
    "you're all set",
    "should be good",
    "ready to go",
    "implementation is complete",
    "changes are in place",
    "everything looks good",
]

# Phrases that suggest the model is asking a question (not done).
_QUESTION_INDICATORS = [
    "would you like",
    "do you want",
    "shall i",
    "should i",
    "which one",
    "what do you",
    "how would you",
    "could you clarify",
    "can you tell me",
    "what should",
    "before i proceed",
    "before proceeding",
]


def _is_completion_point(assistant_response: str) -> bool:
    """Heuristic: does this response look like the model considers the task done?

    Checks for wrap-up language and absence of follow-up questions.
    Doesn't need to be perfect -- a false positive just means one extra
    LLM call that finds no gap.
    """
    if not assistant_response or len(assistant_response.strip()) < 20:
        return False

    lower = assistant_response.lower()

    # If the model is asking a question, it's not done
    if lower.rstrip().endswith("?"):
        for indicator in _QUESTION_INDICATORS:
            if indicator in lower:
                return False

    # Check for completion language
    for phrase in _COMPLETION_PHRASES:
        if phrase in lower:
            return True

    # Heuristic: if the response is substantial (>200 chars) and doesn't
    # end with a question, it's likely a completion point.  Most continuation
    # responses from the model are either questions or contain completion
    # phrases.  This catches the "here's your code" case without explicit
    # wrap-up language.
    if len(assistant_response.strip()) > 200 and not lower.rstrip().endswith("?"):
        return True

    return False


# ---------------------------------------------------------------------------
# Follow-up classification
# ---------------------------------------------------------------------------

# Strong indicators that the user is starting a new, unrelated topic.
_NEW_TASK_PHRASES = [
    "new topic",
    "different question",
    "unrelated",
    "switching to",
    "moving on",
    "by the way",
    "btw",
    "on another note",
    "separate question",
    "something else",
    "change of subject",
]

# Strong indicators of explicit correction.
_CORRECTION_PHRASES = [
    "no,",
    "no.",
    "that's wrong",
    "thats wrong",
    "that's not right",
    "thats not right",
    "that's incorrect",
    "thats incorrect",
    "not what i",
    "i said",
    "i meant",
    "i asked for",
    "wrong",
    "incorrect",
    "that's not what",
    "thats not what",
]


def _classify_followup_heuristic(
    user_message: str,
    completed_summary: str,
) -> FollowupType:
    """Fast heuristic classification of a follow-up message.

    This is a first pass.  The LLM gap extraction will further validate
    whether a continuation represents a real gap or a reasonable new request.
    """
    lower = user_message.lower().strip()

    # Very short messages after completion are usually continuations
    # ("test it", "push it", "lint it", "now deploy")
    if len(lower) < 5:
        return FollowupType.NEW_TASK  # Too short to analyze

    # Check for explicit correction language
    for phrase in _CORRECTION_PHRASES:
        if phrase in lower:
            return FollowupType.CORRECTION

    # Check for new-task indicators
    for phrase in _NEW_TASK_PHRASES:
        if phrase in lower:
            return FollowupType.NEW_TASK

    # Default: treat as continuation (the LLM will decide if it's a real gap)
    return FollowupType.CONTINUATION


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_GAP_EXTRACTION_PROMPT = """\
The assistant considered the following work complete:
{completed_summary}

The user then said:
{user_followup}

If this follow-up represents something the assistant should have done \
proactively -- without being asked -- describe the general behavioral rule \
the assistant should learn.  Frame it as a reusable instruction, not specific \
to this task.

If the follow-up is genuinely a new request, a reasonable extension that \
couldn't have been anticipated, or just a "thanks" / acknowledgment, output \
the single word NONE.

Output ONLY the behavioral rule (one sentence, imperative form) or NONE.  \
No explanation."""


_FOLLOWUP_CLASSIFICATION_PROMPT = """\
The assistant just completed some work and gave this summary:
{completed_summary}

The user responded with:
{user_followup}

Classify the user's response as one of:
- NEW_TASK: The user moved to a different topic or asked something unrelated
- CONTINUATION: The user is extending, adding to, or asking for more on the completed work
- CORRECTION: The user is saying the completed work was wrong or not what they wanted

Output exactly one word: NEW_TASK, CONTINUATION, or CORRECTION."""


# ---------------------------------------------------------------------------
# Gap analysis engine
# ---------------------------------------------------------------------------


class GapAnalysisEngine:
    """Detects behavioral gaps at task completion boundaries.

    Called after every turn in cowork mode.  Tracks when the model
    considers work complete and analyzes user follow-ups to extract
    general behavioral rules the model should learn.
    """

    def __init__(
        self,
        learning: LearningManager,
        model=None,  # ModelProvider, optional for LLM-assisted extraction
        model_retry_policy: ModelRetryPolicy | None = None,
    ):
        self._learning = learning
        self._model = model
        self._compactor = SemanticCompactor(model=model)
        self._model_retry_policy = model_retry_policy or ModelRetryPolicy()

        # State: the last completion point
        self._pending_completion: str | None = None  # summary of completed work

    @property
    def has_model(self) -> bool:
        """Whether LLM-assisted extraction is available."""
        return self._model is not None

    async def on_turn_complete(
        self,
        user_message: str,
        assistant_response: str,
        session_id: str = "",
    ) -> GapResult:
        """Called after every turn.  Manages completion tracking and gap analysis.

        Flow:
        1. If there's a pending completion point, analyze the user's message
           as a follow-up (gap analysis).
        2. Then check if the current assistant response is a new completion point.
        """
        result = GapResult()

        # --- Phase 1: Analyze follow-up to previous completion ---
        if self._pending_completion is not None:
            followup_type = await self._classify_followup(
                user_message, self._pending_completion,
            )
            result.followup_type = followup_type

            if followup_type in (FollowupType.CONTINUATION, FollowupType.CORRECTION):
                signal = await self._extract_gap(
                    completed_summary=self._pending_completion,
                    user_followup=user_message,
                    followup_type=followup_type,
                )
                if signal:
                    result.signal = signal
                    pattern = await self._store_gap_pattern(signal, session_id)
                    result.pattern = pattern

            # Clear the pending completion regardless of classification
            self._pending_completion = None

        # --- Phase 2: Check if current response is a completion point ---
        if _is_completion_point(assistant_response):
            result.is_completion_point = True
            # Store a summary of what was completed for gap analysis on next turn
            self._pending_completion = self._summarize_completion(assistant_response)

        return result

    async def _classify_followup(
        self,
        user_message: str,
        completed_summary: str,
    ) -> FollowupType:
        """Classify a user follow-up as new_task, continuation, or correction.

        Uses heuristics first.  If an LLM is available and the heuristic
        returns CONTINUATION (the ambiguous case), validates with the LLM.
        """
        heuristic = _classify_followup_heuristic(user_message, completed_summary)

        # For new_task and correction, the heuristics are strong enough
        if heuristic != FollowupType.CONTINUATION:
            return heuristic

        # For continuation (the default/ambiguous case), use LLM if available
        if self._model is not None:
            return await self._classify_followup_llm(user_message, completed_summary)

        return heuristic

    async def _classify_followup_llm(
        self,
        user_message: str,
        completed_summary: str,
    ) -> FollowupType:
        """Use LLM to classify the follow-up type."""
        compact_summary = await self._compact_for_prompt(
            completed_summary,
            max_chars=1_200,
            label="gap follow-up completed summary",
        )
        compact_followup = await self._compact_for_prompt(
            user_message,
            max_chars=1_200,
            label="gap follow-up user message",
        )
        prompt = _FOLLOWUP_CLASSIFICATION_PROMPT.format(
            completed_summary=compact_summary,
            user_followup=compact_followup,
        )
        try:
            response = await call_with_model_retry(
                lambda: self._model.complete(
                    [{"role": "user", "content": prompt}],
                ),
                policy=self._model_retry_policy,
            )
            text = (response.text or "").strip().upper()
            if "NEW_TASK" in text:
                return FollowupType.NEW_TASK
            if "CORRECTION" in text:
                return FollowupType.CORRECTION
            return FollowupType.CONTINUATION
        except Exception as e:
            logger.debug("LLM follow-up classification failed (non-fatal): %s", e)
            return FollowupType.CONTINUATION

    async def _extract_gap(
        self,
        completed_summary: str,
        user_followup: str,
        followup_type: FollowupType,
    ) -> GapSignal | None:
        """Extract a behavioral rule from the gap between completion and follow-up.

        If no LLM is available, falls back to a simple heuristic that
        captures the follow-up as-is.
        """
        if self._model is not None:
            return await self._extract_gap_llm(
                completed_summary, user_followup, followup_type,
            )

        # No LLM: store the follow-up directly as a rough pattern.
        # Less precise than LLM extraction but still captures the signal.
        normalized_completed = _normalize_text(completed_summary)
        normalized_followup = _normalize_text(user_followup)
        return GapSignal(
            followup_type=followup_type,
            rule=normalized_followup,
            completed_summary=normalized_completed,
            user_followup=normalized_followup,
            confidence=0.4,
        )

    async def _extract_gap_llm(
        self,
        completed_summary: str,
        user_followup: str,
        followup_type: FollowupType,
    ) -> GapSignal | None:
        """Use LLM to extract a general behavioral rule from the gap."""
        compact_summary = await self._compact_for_prompt(
            completed_summary,
            max_chars=1_200,
            label="gap extraction completed summary",
        )
        compact_followup = await self._compact_for_prompt(
            user_followup,
            max_chars=1_200,
            label="gap extraction follow-up",
        )
        prompt = _GAP_EXTRACTION_PROMPT.format(
            completed_summary=compact_summary,
            user_followup=compact_followup,
        )
        try:
            response = await call_with_model_retry(
                lambda: self._model.complete(
                    [{"role": "user", "content": prompt}],
                ),
                policy=self._model_retry_policy,
            )
            text = (response.text or "").strip()

            # LLM says this isn't a real gap
            if not text or text.upper() == "NONE":
                return None

            compact_rule = await self._compact_for_prompt(
                text,
                max_chars=500,
                label="gap extraction rule",
            )
            compact_completed = await self._compact_for_prompt(
                completed_summary,
                max_chars=900,
                label="gap extraction completed summary store",
            )
            compact_followup_store = await self._compact_for_prompt(
                user_followup,
                max_chars=900,
                label="gap extraction follow-up store",
            )

            return GapSignal(
                followup_type=followup_type,
                rule=compact_rule,
                completed_summary=compact_completed,
                user_followup=compact_followup_store,
                confidence=0.7,
            )
        except Exception as e:
            logger.debug("LLM gap extraction failed (non-fatal): %s", e)
            return None

    async def _store_gap_pattern(
        self,
        signal: GapSignal,
        session_id: str = "",
    ) -> LearnedPattern | None:
        """Convert a gap signal into a stored learned pattern."""
        if signal.followup_type == FollowupType.CORRECTION:
            pattern_type = "behavioral_correction"
        else:
            pattern_type = "behavioral_gap"

        pattern_key = _pattern_key(signal.rule)
        if not pattern_key:
            return None

        pattern = LearnedPattern(
            pattern_type=pattern_type,
            pattern_key=pattern_key,
            data={
                "description": signal.rule,
                "confidence": signal.confidence,
                "source": "gap_analysis",
                "session_id": session_id,
                "completed_summary": signal.completed_summary,
                "user_followup": signal.user_followup,
            },
        )

        try:
            await self._learning.store_or_update(pattern)
            return pattern
        except Exception as e:
            logger.warning("Failed to store gap pattern: %s", e)
            return None

    @staticmethod
    def _summarize_completion(assistant_response: str) -> str:
        """Create a concise summary of what the model delivered.

        Preserve full details and normalize whitespace only. Prompt-size
        control is handled later via semantic compaction.
        """
        return _normalize_text(assistant_response)

    async def _compact_for_prompt(
        self,
        text: str,
        *,
        max_chars: int,
        label: str,
    ) -> str:
        value = _normalize_text(text)
        if not value:
            return ""
        if len(value) <= max_chars:
            return value
        return await self._compactor.compact(
            value,
            max_chars=max_chars,
            label=label,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FILLER_WORDS = frozenset({
    "please", "just", "can", "you", "could", "would", "the", "a", "an",
})


def _pattern_key(text: str) -> str:
    """Generate a stable, deduplicated key from a behavioral rule."""
    text = text.strip().lower()
    if not text:
        return ""
    words = [w for w in text.split() if w not in _FILLER_WORDS]
    return "-".join(words)


def _normalize_text(text: str) -> str:
    """Normalize whitespace while preserving full content."""
    return " ".join(str(text or "").split())


# ---------------------------------------------------------------------------
# Convenience: query behavioral patterns for prompt injection
# ---------------------------------------------------------------------------


async def get_learned_behaviors(
    learning: LearningManager,
    limit: int = 15,
) -> list[LearnedPattern]:
    """Retrieve the most significant learned behavioral patterns.

    Returns patterns sorted by frequency (most reinforced first),
    suitable for injection into system prompts.
    """
    behavioral_types = [
        "behavioral_gap",
        "behavioral_correction",
        "behavioral_preference",
        "behavioral_style",
        "behavioral_knowledge",
        "user_correction",  # Legacy type from original spec
    ]

    all_patterns: list[LearnedPattern] = []
    for ptype in behavioral_types:
        patterns = await learning.query_patterns(
            pattern_type=ptype,
            limit=limit,
        )
        all_patterns.extend(patterns)

    # Sort by frequency (most reinforced = most important), then recency
    all_patterns.sort(key=lambda p: (p.frequency, p.last_seen), reverse=True)
    return all_patterns[:limit]


def format_behaviors_for_prompt(patterns: list[LearnedPattern]) -> str:
    """Format learned behavioral patterns for system prompt injection.

    Returns a concise section that can be appended to any system prompt.
    Domain-agnostic -- works for coding, writing, analysis, planning, etc.
    """
    if not patterns:
        return ""

    lines = ["## Learned Behaviors", ""]
    lines.append(
        "The following behavioral patterns have been learned from previous "
        "interactions. Apply them without being asked:"
    )
    lines.append("")

    for pattern in patterns:
        desc = pattern.data.get("description", pattern.pattern_key)
        freq = pattern.frequency
        if freq > 1:
            lines.append(f"- {desc} (observed {freq}x)")
        else:
            lines.append(f"- {desc}")

    return "\n".join(lines)
