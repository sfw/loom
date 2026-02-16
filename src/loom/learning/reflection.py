"""Automatic reflection engine: extract behavioral patterns from every interaction.

After every user prompt, the reflection engine analyzes the user-assistant exchange
to detect steering signals — corrections, preferences, domain knowledge, and
communication style cues. These are stored as learned patterns that accumulate
over time, allowing the system to adapt its behavior without users needing to
repeat themselves.

This is domain-agnostic: it works for coding, writing, analysis, planning, or
any other task type. The patterns it extracts are behavioral, not operational.

Design principles:
- Runs after every user turn (automatic, not opt-in)
- Non-blocking: reflection runs after the response is delivered
- Hybrid extraction: rule-based for obvious signals, LLM-assisted for nuance
- Cumulative: pattern frequency increases with repeated observations
- Injected: learned patterns influence future prompts via system prompt
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from loom.learning.manager import LearnedPattern, LearningManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal detection markers
# ---------------------------------------------------------------------------

# Phrases that indicate the user is correcting agent behavior.
_CORRECTION_MARKERS = [
    r"\bno[,.]?\s",
    r"\bdon'?t\b",
    r"\bstop\b",
    r"\binstead\b",
    r"\bactually[,]?\s",
    r"\bi said\b",
    r"\bi meant\b",
    r"\bnot like that\b",
    r"\bthat'?s (?:not |wrong)",
    r"\bi didn'?t (?:ask|want|mean)",
    r"\bplease don'?t\b",
    r"\bnot what i\b",
    r"\bwrong\b.*\bright\b",
]

# Phrases that indicate an explicit preference.
_PREFERENCE_MARKERS = [
    r"\balways\b",
    r"\bnever\b",
    r"\bprefer\b",
    r"\buse\b.+\bnot\b",
    r"\bi like\b",
    r"\bi want\b",
    r"\bi need\b.+\bto be\b",
    r"\bfrom now on\b",
    r"\bgoing forward\b",
    r"\bin the future\b",
    r"\bby default\b",
    r"\bmake sure\b.+\balways\b",
    r"\bremember\b.+\bto\b",
]

# Phrases that indicate communication style feedback.
_STYLE_MARKERS = [
    r"\bbe more\b",
    r"\bless\b.+\b(?:verbose|wordy|detailed)\b",
    r"\bmore\b.+\b(?:concise|brief|detail|specific)\b",
    r"\btoo\b.+\b(?:long|short|verbose|wordy|vague|detailed)\b",
    r"\bshorter\b",
    r"\blonger\b",
    r"\bsimpler\b",
    r"\bjust\b.+\b(?:tell|show|give)\b",
    r"\bskip\b.+\b(?:explanation|intro|preamble)\b",
    r"\bget to the point\b",
    r"\bdon'?t explain\b",
]

# Phrases that indicate domain knowledge the user is providing.
_KNOWLEDGE_MARKERS = [
    r"\bwe use\b",
    r"\bour\b.+\b(?:uses?|runs?|requires?|starts?|is|are)\b",
    r"\bour (?:team|company|org|project|codebase|stack|fiscal|api|database|workflow)\b",
    r"\bin our\b",
    r"\bfyi\b",
    r"\bjust so you know\b",
    r"\bfor context\b",
    r"\bfor reference\b",
    r"\bthe way we\b",
    r"\bour convention\b",
    r"\bour standard\b",
    r"\bkeep (?:that |this )?in mind\b",
    r"\bwe (?:follow|adopt|run|deploy)\b",
]


@dataclass
class SteeringSignal:
    """A detected steering signal from a user message."""

    signal_type: str  # correction, preference, style, knowledge
    content: str  # The relevant text from the user
    confidence: float = 0.5  # 0.0 to 1.0
    context: str = ""  # Surrounding context


@dataclass
class ReflectionResult:
    """Result of reflecting on a single user-assistant exchange."""

    signals: list[SteeringSignal] = field(default_factory=list)
    patterns: list[LearnedPattern] = field(default_factory=list)
    had_correction: bool = False
    had_preference: bool = False
    had_style_feedback: bool = False
    had_knowledge: bool = False


# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_REFLECTION_PROMPT = """\
You are analyzing a user-assistant exchange to detect behavioral steering signals.
The user may be correcting the assistant, expressing preferences, providing domain
knowledge, or giving feedback on communication style.

USER MESSAGE:
{user_message}

ASSISTANT'S PREVIOUS RESPONSE (if any):
{previous_response}

Analyze this exchange and extract any behavioral signals. For each signal found,
output a JSON object on its own line with these fields:
- "type": one of "correction", "preference", "style", "knowledge"
- "pattern": a concise, reusable description of the learned behavior (imperative form)
  Examples: "Use YAML instead of JSON for configs", "Keep responses under 3 paragraphs",
  "Project uses PostgreSQL 15", "Prefer functional style over classes"
- "confidence": 0.0-1.0 how confident you are this is a real signal (not just conversation)

Output ONLY the JSON lines, one per signal. If no signals are detected, output nothing.
Do not explain. Do not wrap in markdown. Just the JSON lines."""


class ReflectionEngine:
    """Analyzes user-assistant exchanges to extract behavioral patterns.

    Two-phase extraction:
    1. Rule-based: fast regex scan for obvious steering markers
    2. LLM-assisted: when a model is available, extract nuanced signals

    The engine is designed to be called after every user turn. It's
    intentionally lightweight — rule-based extraction adds zero latency,
    and LLM extraction uses the utility (small) model.
    """

    def __init__(
        self,
        learning: LearningManager,
        model=None,  # ModelProvider, optional for LLM-assisted extraction
    ):
        self._learning = learning
        self._model = model

        # Pre-compile regex patterns for performance
        self._correction_re = [re.compile(p, re.IGNORECASE) for p in _CORRECTION_MARKERS]
        self._preference_re = [re.compile(p, re.IGNORECASE) for p in _PREFERENCE_MARKERS]
        self._style_re = [re.compile(p, re.IGNORECASE) for p in _STYLE_MARKERS]
        self._knowledge_re = [re.compile(p, re.IGNORECASE) for p in _KNOWLEDGE_MARKERS]

    @property
    def has_model(self) -> bool:
        """Whether LLM-assisted extraction is available."""
        return self._model is not None

    async def reflect_on_turn(
        self,
        user_message: str,
        assistant_response: str = "",
        previous_assistant_message: str = "",
        session_id: str = "",
    ) -> ReflectionResult:
        """Analyze a user turn and extract behavioral patterns.

        Called after every user prompt. Runs both rule-based and (optionally)
        LLM-assisted extraction.

        Args:
            user_message: The user's current message.
            assistant_response: The assistant's response to this message (may
                be empty if reflection runs before response).
            previous_assistant_message: The assistant's prior response (the one
                the user might be reacting to).
            session_id: Current session ID for pattern scoping.

        Returns:
            ReflectionResult with detected signals and stored patterns.
        """
        result = ReflectionResult()

        # Phase 1: Rule-based signal detection
        rule_signals = self._detect_signals_rule_based(
            user_message, previous_assistant_message,
        )
        result.signals.extend(rule_signals)

        # Phase 2: LLM-assisted extraction (if model available and signals detected)
        if self._model is not None:
            llm_signals = await self._detect_signals_llm(
                user_message, previous_assistant_message,
            )
            # Merge, preferring LLM signals (higher quality) but keeping
            # rule-based signals that LLM might have missed
            result.signals = self._merge_signals(rule_signals, llm_signals)

        # Classify what we found
        for signal in result.signals:
            if signal.signal_type == "correction":
                result.had_correction = True
            elif signal.signal_type == "preference":
                result.had_preference = True
            elif signal.signal_type == "style":
                result.had_style_feedback = True
            elif signal.signal_type == "knowledge":
                result.had_knowledge = True

        # Store patterns from detected signals
        for signal in result.signals:
            if signal.confidence >= 0.3:  # Only store reasonably confident signals
                pattern = await self._signal_to_pattern(signal, session_id)
                if pattern:
                    result.patterns.append(pattern)

        return result

    # ------------------------------------------------------------------
    # Rule-based detection
    # ------------------------------------------------------------------

    def _detect_signals_rule_based(
        self,
        user_message: str,
        previous_response: str = "",
    ) -> list[SteeringSignal]:
        """Fast regex-based signal detection."""
        signals: list[SteeringSignal] = []
        msg_lower = user_message.lower()

        # Skip very short messages (likely just acknowledgments)
        if len(user_message.strip()) < 10:
            return signals

        # Check for corrections
        correction_score = self._scan_markers(msg_lower, self._correction_re)
        if correction_score > 0:
            signals.append(SteeringSignal(
                signal_type="correction",
                content=user_message[:200],
                confidence=min(0.7, 0.3 + correction_score * 0.2),
                context=previous_response[:200] if previous_response else "",
            ))

        # Check for preferences
        preference_score = self._scan_markers(msg_lower, self._preference_re)
        if preference_score > 0:
            signals.append(SteeringSignal(
                signal_type="preference",
                content=user_message[:200],
                confidence=min(0.8, 0.4 + preference_score * 0.2),
                context="",
            ))

        # Check for style feedback
        style_score = self._scan_markers(msg_lower, self._style_re)
        if style_score > 0:
            signals.append(SteeringSignal(
                signal_type="style",
                content=user_message[:200],
                confidence=min(0.8, 0.4 + style_score * 0.2),
                context="",
            ))

        # Check for domain knowledge
        knowledge_score = self._scan_markers(msg_lower, self._knowledge_re)
        if knowledge_score > 0:
            signals.append(SteeringSignal(
                signal_type="knowledge",
                content=user_message[:200],
                confidence=min(0.7, 0.3 + knowledge_score * 0.2),
                context="",
            ))

        return signals

    @staticmethod
    def _scan_markers(text: str, patterns: list[re.Pattern]) -> int:
        """Count how many marker patterns match in the text."""
        return sum(1 for p in patterns if p.search(text))

    # ------------------------------------------------------------------
    # LLM-assisted detection
    # ------------------------------------------------------------------

    async def _detect_signals_llm(
        self,
        user_message: str,
        previous_response: str = "",
    ) -> list[SteeringSignal]:
        """Use a small model to extract nuanced behavioral signals."""
        if self._model is None:
            return []

        prompt = _REFLECTION_PROMPT.format(
            user_message=user_message[:500],
            previous_response=previous_response[:500] if previous_response else "(none)",
        )

        try:
            response = await self._model.complete(
                [{"role": "user", "content": prompt}],
            )
            return self._parse_llm_signals(response.text or "")
        except Exception as e:
            logger.debug("LLM reflection failed (non-fatal): %s", e)
            return []

    @staticmethod
    def _parse_llm_signals(text: str) -> list[SteeringSignal]:
        """Parse JSON lines from LLM output into SteeringSignals."""
        signals: list[SteeringSignal] = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
                signal_type = data.get("type", "")
                if signal_type not in ("correction", "preference", "style", "knowledge"):
                    continue
                signals.append(SteeringSignal(
                    signal_type=signal_type,
                    content=data.get("pattern", "")[:200],
                    confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
                ))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return signals

    # ------------------------------------------------------------------
    # Signal merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_signals(
        rule_signals: list[SteeringSignal],
        llm_signals: list[SteeringSignal],
    ) -> list[SteeringSignal]:
        """Merge rule-based and LLM signals, preferring LLM quality.

        LLM signals have better pattern descriptions (concise, reusable).
        Rule signals catch things the LLM might miss.
        When both detect the same type, prefer the LLM version.
        """
        llm_types = {s.signal_type for s in llm_signals}
        merged = list(llm_signals)

        # Add rule-based signals for types the LLM didn't detect
        for signal in rule_signals:
            if signal.signal_type not in llm_types:
                merged.append(signal)

        return merged

    # ------------------------------------------------------------------
    # Pattern storage
    # ------------------------------------------------------------------

    async def _signal_to_pattern(
        self,
        signal: SteeringSignal,
        session_id: str = "",
    ) -> LearnedPattern | None:
        """Convert a steering signal into a stored learned pattern."""
        # Map signal types to pattern types
        pattern_type = f"behavioral_{signal.signal_type}"

        # Generate a stable key for deduplication
        pattern_key = self._signal_key(signal)
        if not pattern_key:
            return None

        pattern = LearnedPattern(
            pattern_type=pattern_type,
            pattern_key=pattern_key,
            data={
                "description": signal.content[:200],
                "confidence": signal.confidence,
                "source": "reflection",
                "session_id": session_id,
                "context": signal.context[:200] if signal.context else "",
            },
        )

        try:
            await self._learning.store_or_update(pattern)
            return pattern
        except Exception as e:
            logger.warning("Failed to store behavioral pattern: %s", e)
            return None

    @staticmethod
    def _signal_key(signal: SteeringSignal) -> str:
        """Generate a stable, deduplicated key for a signal.

        For LLM-extracted signals, the content IS the pattern description
        (concise, reusable). For rule-based signals, we normalize the
        user's raw text into a key.
        """
        text = signal.content.strip().lower()
        if not text:
            return ""

        # Remove common filler words for key generation
        text = re.sub(r"\b(please|just|can you|could you|would you)\b", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate to reasonable key length
        words = text.split()[:8]
        return "-".join(words)


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
    Domain-agnostic — works for coding, writing, analysis, planning, etc.
    """
    if not patterns:
        return ""

    lines = ["## Learned Preferences", ""]
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
