"""Tests for the reflection engine (behavioral pattern extraction)."""

from __future__ import annotations

import pytest

from loom.learning.manager import LearnedPattern, LearningManager
from loom.learning.reflection import (
    ReflectionEngine,
    SteeringSignal,
    format_behaviors_for_prompt,
    get_learned_behaviors,
)
from loom.state.memory import Database


# --- Helpers ---


async def _make_engine(tmp_path) -> tuple[ReflectionEngine, LearningManager]:
    """Create a ReflectionEngine with an in-memory DB (no LLM model)."""
    db = Database(str(tmp_path / "test.db"))
    await db.initialize()
    mgr = LearningManager(db)
    engine = ReflectionEngine(learning=mgr)
    return engine, mgr


# --- Rule-based signal detection ---


class TestRuleBasedDetection:
    @pytest.mark.asyncio
    async def test_detects_correction(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="No, don't add docstrings unless I ask for them.",
            assistant_response="",
            previous_assistant_message="I've added docstrings to all functions.",
        )
        assert result.had_correction
        assert any(s.signal_type == "correction" for s in result.signals)

    @pytest.mark.asyncio
    async def test_detects_preference(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="Always use TypeScript, never plain JavaScript.",
        )
        assert result.had_preference
        assert any(s.signal_type == "preference" for s in result.signals)

    @pytest.mark.asyncio
    async def test_detects_style_feedback(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="Be more concise. I don't need the full explanation.",
        )
        assert result.had_style_feedback
        assert any(s.signal_type == "style" for s in result.signals)

    @pytest.mark.asyncio
    async def test_detects_domain_knowledge(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="We use PostgreSQL with pgvector in production.",
        )
        assert result.had_knowledge
        assert any(s.signal_type == "knowledge" for s in result.signals)

    @pytest.mark.asyncio
    async def test_ignores_short_messages(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="ok",
        )
        assert len(result.signals) == 0

    @pytest.mark.asyncio
    async def test_ignores_neutral_messages(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="Can you read the file src/main.py and tell me what it does?",
        )
        # Neutral request — no steering signals
        assert not result.had_correction
        assert not result.had_style_feedback

    @pytest.mark.asyncio
    async def test_multiple_signals(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="No, don't use classes. I always prefer functional style. Be more concise.",
        )
        assert result.had_correction
        assert result.had_preference
        assert result.had_style_feedback
        assert len(result.signals) >= 3

    @pytest.mark.asyncio
    async def test_from_now_on(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="From now on, use snake_case for all variable names.",
        )
        assert result.had_preference

    @pytest.mark.asyncio
    async def test_our_convention(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)
        result = await engine.reflect_on_turn(
            user_message="Our convention is to put tests in a __tests__ directory.",
        )
        assert result.had_knowledge


# --- Pattern storage ---


class TestPatternStorage:
    @pytest.mark.asyncio
    async def test_stores_behavioral_pattern(self, tmp_path):
        engine, mgr = await _make_engine(tmp_path)
        await engine.reflect_on_turn(
            user_message="Always use YAML instead of JSON for config files.",
        )
        patterns = await mgr.query_behavioral()
        assert len(patterns) >= 1
        assert any("behavioral_" in p.pattern_type for p in patterns)

    @pytest.mark.asyncio
    async def test_frequency_increases_on_repeated_signal(self, tmp_path):
        engine, mgr = await _make_engine(tmp_path)

        # Same preference expressed twice
        await engine.reflect_on_turn(
            user_message="Always use YAML instead of JSON for config files.",
        )
        await engine.reflect_on_turn(
            user_message="Always use YAML instead of JSON for config files.",
        )

        patterns = await mgr.query_behavioral()
        # At least one pattern should have frequency > 1
        assert any(p.frequency >= 2 for p in patterns)

    @pytest.mark.asyncio
    async def test_stores_correction_pattern(self, tmp_path):
        engine, mgr = await _make_engine(tmp_path)
        await engine.reflect_on_turn(
            user_message="No, don't add comments to every line of code.",
            previous_assistant_message="I've added comments throughout.",
        )
        patterns = await mgr.query_behavioral()
        assert any(p.pattern_type == "behavioral_correction" for p in patterns)

    @pytest.mark.asyncio
    async def test_stores_style_pattern(self, tmp_path):
        engine, mgr = await _make_engine(tmp_path)
        await engine.reflect_on_turn(
            user_message="Be more concise. Skip the explanation and just show code.",
        )
        patterns = await mgr.query_behavioral()
        assert any(p.pattern_type == "behavioral_style" for p in patterns)


# --- LLM signal parsing ---


class TestLLMParsing:
    def test_parse_valid_json_lines(self):
        text = (
            '{"type": "preference", "pattern": "Use YAML for configs", "confidence": 0.8}\n'
            '{"type": "correction", "pattern": "No docstrings", "confidence": 0.6}\n'
        )
        signals = ReflectionEngine._parse_llm_signals(text)
        assert len(signals) == 2
        assert signals[0].signal_type == "preference"
        assert signals[1].signal_type == "correction"

    def test_parse_empty_output(self):
        signals = ReflectionEngine._parse_llm_signals("")
        assert len(signals) == 0

    def test_parse_invalid_json(self):
        text = "This is not JSON\n{broken json}\n"
        signals = ReflectionEngine._parse_llm_signals(text)
        assert len(signals) == 0

    def test_parse_unknown_type_ignored(self):
        text = '{"type": "unknown", "pattern": "something", "confidence": 0.5}\n'
        signals = ReflectionEngine._parse_llm_signals(text)
        assert len(signals) == 0

    def test_parse_clamps_confidence(self):
        text = '{"type": "preference", "pattern": "test", "confidence": 5.0}\n'
        signals = ReflectionEngine._parse_llm_signals(text)
        assert len(signals) == 1
        assert signals[0].confidence == 1.0


# --- Signal merging ---


class TestSignalMerging:
    def test_prefers_llm_signals(self):
        rule = [SteeringSignal(signal_type="correction", content="raw text")]
        llm = [SteeringSignal(signal_type="correction", content="Use YAML")]
        merged = ReflectionEngine._merge_signals(rule, llm)
        # LLM version should be preferred for correction type
        assert len(merged) == 1
        assert merged[0].content == "Use YAML"

    def test_adds_rule_signals_for_missing_types(self):
        rule = [
            SteeringSignal(signal_type="correction", content="raw"),
            SteeringSignal(signal_type="style", content="concise"),
        ]
        llm = [SteeringSignal(signal_type="correction", content="Use YAML")]
        merged = ReflectionEngine._merge_signals(rule, llm)
        assert len(merged) == 2
        types = {s.signal_type for s in merged}
        assert "correction" in types
        assert "style" in types


# --- Prompt formatting ---


class TestPromptFormatting:
    def test_format_empty(self):
        result = format_behaviors_for_prompt([])
        assert result == ""

    def test_format_single_pattern(self):
        patterns = [
            LearnedPattern(
                pattern_type="behavioral_preference",
                pattern_key="test",
                data={"description": "Use YAML for configs"},
                frequency=3,
            ),
        ]
        result = format_behaviors_for_prompt(patterns)
        assert "Learned Preferences" in result
        assert "Use YAML for configs" in result
        assert "3x" in result

    def test_format_frequency_1_no_count(self):
        patterns = [
            LearnedPattern(
                pattern_type="behavioral_preference",
                pattern_key="test",
                data={"description": "Use YAML for configs"},
                frequency=1,
            ),
        ]
        result = format_behaviors_for_prompt(patterns)
        assert "observed" not in result

    def test_format_multiple_patterns(self):
        patterns = [
            LearnedPattern(
                pattern_type="behavioral_preference",
                pattern_key="k1",
                data={"description": "Use YAML"},
                frequency=5,
            ),
            LearnedPattern(
                pattern_type="behavioral_correction",
                pattern_key="k2",
                data={"description": "No docstrings"},
                frequency=2,
            ),
        ]
        result = format_behaviors_for_prompt(patterns)
        assert "Use YAML" in result
        assert "No docstrings" in result


# --- Query behavioral patterns ---


class TestQueryBehavioral:
    @pytest.mark.asyncio
    async def test_query_returns_behavioral_only(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        # Store a behavioral pattern
        await mgr.store_or_update(LearnedPattern(
            pattern_type="behavioral_preference",
            pattern_key="yaml-preference",
            data={"description": "Use YAML"},
        ))

        # Store an operational pattern
        await mgr.store_or_update(LearnedPattern(
            pattern_type="subtask_success",
            pattern_key="install-deps",
            data={"success": True},
        ))

        behavioral = await mgr.query_behavioral()
        assert len(behavioral) == 1
        assert behavioral[0].pattern_type == "behavioral_preference"

    @pytest.mark.asyncio
    async def test_get_learned_behaviors(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        # Store multiple behavioral patterns
        for i in range(20):
            await mgr.store_or_update(LearnedPattern(
                pattern_type="behavioral_preference",
                pattern_key=f"pref-{i}",
                data={"description": f"Preference {i}"},
            ))

        patterns = await get_learned_behaviors(mgr, limit=10)
        assert len(patterns) <= 10

    @pytest.mark.asyncio
    async def test_includes_legacy_user_correction(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        await mgr.store_or_update(LearnedPattern(
            pattern_type="user_correction",
            pattern_key="legacy-correction",
            data={"description": "Prefer tabs over spaces"},
        ))

        behavioral = await mgr.query_behavioral()
        assert len(behavioral) == 1
        assert behavioral[0].pattern_type == "user_correction"


# --- Non-coding domain examples ---


class TestDomainAgnostic:
    """Verify ALM works for non-coding domains."""

    @pytest.mark.asyncio
    async def test_writing_preference(self, tmp_path):
        engine, mgr = await _make_engine(tmp_path)
        await engine.reflect_on_turn(
            user_message="Always use Oxford comma in lists, never skip it.",
        )
        patterns = await mgr.query_behavioral()
        assert len(patterns) >= 1

    @pytest.mark.asyncio
    async def test_financial_knowledge(self, tmp_path):
        engine, mgr = await _make_engine(tmp_path)
        await engine.reflect_on_turn(
            user_message="Our fiscal year starts April 1, keep that in mind for all reports.",
        )
        patterns = await mgr.query_behavioral()
        assert len(patterns) >= 1

    @pytest.mark.asyncio
    async def test_style_feedback_research(self, tmp_path):
        engine, mgr = await _make_engine(tmp_path)
        await engine.reflect_on_turn(
            user_message="Be more specific with citations. Too vague. Show the actual paper title.",
        )
        assert any(
            p.pattern_type == "behavioral_style"
            for p in await mgr.query_behavioral()
        )

    @pytest.mark.asyncio
    async def test_consulting_correction(self, tmp_path):
        engine, mgr = await _make_engine(tmp_path)
        await engine.reflect_on_turn(
            user_message="No, don't recommend waterfall. We use agile sprints exclusively.",
            previous_assistant_message="I recommend a waterfall approach for this project.",
        )
        result_patterns = await mgr.query_behavioral()
        assert any(p.pattern_type == "behavioral_correction" for p in result_patterns)
