"""Tests for the gap analysis engine (behavioral pattern extraction)."""

from __future__ import annotations

import pytest

from loom.learning.manager import LearnedPattern, LearningManager
from loom.learning.reflection import (
    FollowupType,
    GapAnalysisEngine,
    _classify_followup_heuristic,
    _is_completion_point,
    _pattern_key,
    format_behaviors_for_prompt,
    get_learned_behaviors,
)
from loom.state.memory import Database

# --- Helpers ---


async def _make_engine(tmp_path) -> tuple[GapAnalysisEngine, LearningManager]:
    """Create a GapAnalysisEngine with an in-memory DB (no LLM model)."""
    db = Database(str(tmp_path / "test.db"))
    await db.initialize()
    mgr = LearningManager(db)
    engine = GapAnalysisEngine(learning=mgr)
    return engine, mgr


# --- Completion point detection ---


class TestCompletionDetection:
    def test_completion_with_wrap_up_phrase(self):
        assert _is_completion_point(
            "I've implemented the feature. Let me know if you need anything else."
        )

    def test_completion_with_heres_the(self):
        assert _is_completion_point(
            "Here's the updated implementation with error handling added."
        )

    def test_completion_all_done(self):
        assert _is_completion_point("All done. The tests pass and linting is clean.")

    def test_completion_ive_fixed(self):
        assert _is_completion_point(
            "I've fixed the bug in the parser. The issue was an off-by-one error."
        )

    def test_not_completion_question(self):
        assert not _is_completion_point(
            "Would you like me to add error handling as well?"
        )

    def test_not_completion_short(self):
        assert not _is_completion_point("ok")

    def test_not_completion_empty(self):
        assert not _is_completion_point("")

    def test_completion_long_response_no_question(self):
        # Substantial response without explicit completion phrase but no question
        long_response = "def calculate(x, y):\n    return x + y\n\n" * 10
        assert _is_completion_point(long_response)

    def test_not_completion_question_with_indicator(self):
        assert not _is_completion_point(
            "I can implement this. Should I use async or sync approach?"
        )

    def test_completion_changes_in_place(self):
        assert _is_completion_point(
            "The changes are in place. The config now uses YAML format."
        )


# --- Follow-up classification ---


class TestFollowupClassification:
    def test_correction_no_comma(self):
        result = _classify_followup_heuristic("no, use JSON not YAML", "")
        assert result == FollowupType.CORRECTION

    def test_correction_thats_wrong(self):
        result = _classify_followup_heuristic("that's wrong, it should be async", "")
        assert result == FollowupType.CORRECTION

    def test_correction_i_said(self):
        result = _classify_followup_heuristic("i said use PostgreSQL not MySQL", "")
        assert result == FollowupType.CORRECTION

    def test_new_task_btw(self):
        result = _classify_followup_heuristic(
            "btw, can you help with something unrelated?", "",
        )
        assert result == FollowupType.NEW_TASK

    def test_new_task_different_question(self):
        result = _classify_followup_heuristic(
            "I have a different question about the API", "",
        )
        assert result == FollowupType.NEW_TASK

    def test_continuation_default(self):
        result = _classify_followup_heuristic("test and lint it", "")
        assert result == FollowupType.CONTINUATION

    def test_continuation_add_more(self):
        result = _classify_followup_heuristic("can you add error handling?", "")
        assert result == FollowupType.CONTINUATION

    def test_continuation_edge_cases(self):
        result = _classify_followup_heuristic("what about edge cases?", "")
        assert result == FollowupType.CONTINUATION

    def test_too_short_is_new_task(self):
        # Messages under 5 chars are too short to analyze
        result = _classify_followup_heuristic("ok", "")
        assert result == FollowupType.NEW_TASK

    def test_correction_i_meant(self):
        result = _classify_followup_heuristic(
            "i meant the other file, not this one", "",
        )
        assert result == FollowupType.CORRECTION


# --- Full gap analysis flow (no LLM) ---


class TestGapAnalysisNoLLM:
    @pytest.mark.asyncio
    async def test_detects_completion_point(self, tmp_path):
        engine, _ = await _make_engine(tmp_path)

        result = await engine.on_turn_complete(
            user_message="Add error handling to the parser",
            assistant_response=(
                "I've implemented the error handling."
                " Let me know if you need anything else."
            ),
        )
        assert result.is_completion_point

    @pytest.mark.asyncio
    async def test_no_gap_on_first_turn(self, tmp_path):
        """First turn has no pending completion, so no gap analysis."""
        engine, _ = await _make_engine(tmp_path)

        result = await engine.on_turn_complete(
            user_message="Add error handling",
            assistant_response="I've added error handling. Let me know if you need anything else.",
        )
        assert result.signal is None
        assert result.followup_type is None

    @pytest.mark.asyncio
    async def test_gap_detected_on_continuation(self, tmp_path):
        """When user follows up after completion, gap is detected."""
        engine, mgr = await _make_engine(tmp_path)

        # Turn 1: model completes work
        await engine.on_turn_complete(
            user_message="Write a function to parse CSV",
            assistant_response=(
                "Here's the CSV parser implementation."
                " Let me know if you need anything else."
            ),
        )

        # Turn 2: user asks for more (continuation)
        result = await engine.on_turn_complete(
            user_message="test and lint it",
            assistant_response="Running tests now...",
        )

        assert result.followup_type == FollowupType.CONTINUATION
        assert result.signal is not None
        # Without LLM, the rule is the raw follow-up text
        assert "test and lint it" in result.signal.rule

    @pytest.mark.asyncio
    async def test_gap_stored_as_pattern(self, tmp_path):
        """Detected gaps should be stored in the learning manager."""
        engine, mgr = await _make_engine(tmp_path)

        # Turn 1: completion
        await engine.on_turn_complete(
            user_message="Implement the feature",
            assistant_response="I've implemented the feature. All done.",
        )

        # Turn 2: continuation
        result = await engine.on_turn_complete(
            user_message="what about error handling?",
            assistant_response="Good point, adding error handling now.",
        )

        assert result.pattern is not None
        patterns = await mgr.query_behavioral()
        assert len(patterns) >= 1
        assert any(p.pattern_type == "behavioral_gap" for p in patterns)

    @pytest.mark.asyncio
    async def test_correction_stored_as_correction_type(self, tmp_path):
        """Explicit corrections should be stored as behavioral_correction."""
        engine, mgr = await _make_engine(tmp_path)

        # Turn 1: completion
        await engine.on_turn_complete(
            user_message="Write the config loader",
            assistant_response=(
                "Here's the config loader using YAML."
                " Let me know if you need changes."
            ),
        )

        # Turn 2: correction
        result = await engine.on_turn_complete(
            user_message="no, use JSON not YAML",
            assistant_response="Got it, switching to JSON.",
        )

        assert result.followup_type == FollowupType.CORRECTION
        assert result.pattern is not None
        assert result.pattern.pattern_type == "behavioral_correction"

    @pytest.mark.asyncio
    async def test_new_task_no_gap(self, tmp_path):
        """New tasks after completion should not trigger gap analysis."""
        engine, mgr = await _make_engine(tmp_path)

        # Turn 1: completion
        await engine.on_turn_complete(
            user_message="Fix the login bug",
            assistant_response="I've fixed the login bug. Let me know if you need anything else.",
        )

        # Turn 2: new task
        result = await engine.on_turn_complete(
            user_message="btw, can you help me with something else entirely?",
            assistant_response="Sure, what do you need?",
        )

        assert result.followup_type == FollowupType.NEW_TASK
        assert result.signal is None
        assert result.pattern is None

    @pytest.mark.asyncio
    async def test_no_completion_no_gap(self, tmp_path):
        """If the model didn't reach a completion point, no gap analysis."""
        engine, _ = await _make_engine(tmp_path)

        # Turn 1: model asks a question (not a completion point)
        await engine.on_turn_complete(
            user_message="Build the API",
            assistant_response="Should I use REST or GraphQL?",
        )

        # Turn 2: user answers (not a follow-up to completion)
        result = await engine.on_turn_complete(
            user_message="Use REST",
            assistant_response=(
                "Got it, implementing REST API."
                " Here's the implementation."
                " Let me know if you need anything else."
            ),
        )

        assert result.followup_type is None
        assert result.signal is None

    @pytest.mark.asyncio
    async def test_frequency_increases_on_repeated_gap(self, tmp_path):
        """Same gap pattern appearing multiple times increases frequency."""
        engine, mgr = await _make_engine(tmp_path)

        for i in range(3):
            # Completion
            await engine.on_turn_complete(
                user_message=f"Write function {i}",
                assistant_response=(
                    "I've implemented the function."
                    " Let me know if you need anything else."
                ),
            )
            # Same follow-up each time
            await engine.on_turn_complete(
                user_message="test and lint it",
                assistant_response="Running tests...",
            )

        patterns = await mgr.query_behavioral()
        assert any(p.frequency >= 3 for p in patterns)

    @pytest.mark.asyncio
    async def test_pending_completion_cleared_after_analysis(self, tmp_path):
        """Pending completion is cleared after follow-up analysis."""
        engine, _ = await _make_engine(tmp_path)

        # Turn 1: completion
        await engine.on_turn_complete(
            user_message="Do the thing",
            assistant_response="All done. Let me know if you need anything else.",
        )
        assert engine._pending_completion is not None

        # Turn 2: any follow-up clears it
        await engine.on_turn_complete(
            user_message="thanks",
            assistant_response="You're welcome!",
        )
        assert engine._pending_completion is None


# --- Completion summary ---


class TestCompletionSummary:
    def test_short_response_kept_as_is(self):
        text = "I've fixed the bug."
        assert GapAnalysisEngine._summarize_completion(text) == text

    def test_long_response_truncated(self):
        text = "A" * 500
        summary = GapAnalysisEngine._summarize_completion(text)
        assert len(summary) <= 304  # 300 + "..."

    def test_truncates_at_sentence_boundary(self):
        sentences = "First sentence. " * 25  # >300 chars
        summary = GapAnalysisEngine._summarize_completion(sentences)
        assert summary.endswith(".")
        assert len(summary) <= 300


# --- Pattern key ---


class TestPatternKey:
    def test_basic_key(self):
        key = _pattern_key("Run tests after writing code")
        assert key == "run-tests-after-writing-code"

    def test_removes_filler_words(self):
        key = _pattern_key("Can you please just run the tests")
        assert "please" not in key
        assert "just" not in key
        assert "can" not in key
        assert "you" not in key
        assert "the" not in key

    def test_truncates_to_8_words(self):
        key = _pattern_key("one two three four five six seven eight nine ten")
        assert len(key.split("-")) <= 8

    def test_empty_input(self):
        assert _pattern_key("") == ""

    def test_only_filler_words(self):
        assert _pattern_key("just the a") == ""


# --- Prompt formatting ---


class TestPromptFormatting:
    def test_format_empty(self):
        result = format_behaviors_for_prompt([])
        assert result == ""

    def test_format_single_pattern(self):
        patterns = [
            LearnedPattern(
                pattern_type="behavioral_gap",
                pattern_key="test",
                data={"description": "Run tests after writing code"},
                frequency=3,
            ),
        ]
        result = format_behaviors_for_prompt(patterns)
        assert "Learned Behaviors" in result
        assert "Run tests after writing code" in result
        assert "3x" in result

    def test_format_frequency_1_no_count(self):
        patterns = [
            LearnedPattern(
                pattern_type="behavioral_gap",
                pattern_key="test",
                data={"description": "Run tests after writing code"},
                frequency=1,
            ),
        ]
        result = format_behaviors_for_prompt(patterns)
        assert "observed" not in result

    def test_format_multiple_patterns(self):
        patterns = [
            LearnedPattern(
                pattern_type="behavioral_gap",
                pattern_key="k1",
                data={"description": "Run tests after code changes"},
                frequency=5,
            ),
            LearnedPattern(
                pattern_type="behavioral_correction",
                pattern_key="k2",
                data={"description": "Use JSON not YAML"},
                frequency=2,
            ),
        ]
        result = format_behaviors_for_prompt(patterns)
        assert "Run tests after code changes" in result
        assert "Use JSON not YAML" in result


# --- Query behavioral patterns ---


class TestQueryBehavioral:
    @pytest.mark.asyncio
    async def test_query_returns_behavioral_only(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        # Store a behavioral pattern
        await mgr.store_or_update(LearnedPattern(
            pattern_type="behavioral_gap",
            pattern_key="test-gap",
            data={"description": "Run tests after writing code"},
        ))

        # Store an operational pattern
        await mgr.store_or_update(LearnedPattern(
            pattern_type="subtask_success",
            pattern_key="install-deps",
            data={"success": True},
        ))

        behavioral = await mgr.query_behavioral()
        assert len(behavioral) == 1
        assert behavioral[0].pattern_type == "behavioral_gap"

    @pytest.mark.asyncio
    async def test_get_learned_behaviors(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        # Store multiple behavioral patterns
        for i in range(20):
            await mgr.store_or_update(LearnedPattern(
                pattern_type="behavioral_gap",
                pattern_key=f"gap-{i}",
                data={"description": f"Rule {i}"},
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


# --- Domain-agnostic examples ---


class TestDomainAgnostic:
    """Verify gap analysis works across domains."""

    @pytest.mark.asyncio
    async def test_coding_test_and_lint(self, tmp_path):
        """The canonical case: 'test and lint it' after code completion."""
        engine, mgr = await _make_engine(tmp_path)

        await engine.on_turn_complete(
            user_message="Implement the CSV parser",
            assistant_response="Here's the implementation. Let me know if you need anything else.",
        )
        result = await engine.on_turn_complete(
            user_message="test and lint it",
            assistant_response="Running tests and linter...",
        )

        assert result.followup_type == FollowupType.CONTINUATION
        assert result.signal is not None
        patterns = await mgr.query_behavioral()
        assert len(patterns) >= 1

    @pytest.mark.asyncio
    async def test_writing_edge_case(self, tmp_path):
        """Writing domain: user asks for something the model should have included."""
        engine, mgr = await _make_engine(tmp_path)

        await engine.on_turn_complete(
            user_message="Write an executive summary of the Q3 report",
            assistant_response=(
                "Here's the executive summary covering revenue"
                " and growth. Let me know if you need changes."
            ),
        )
        result = await engine.on_turn_complete(
            user_message="what about the methodology section?",
            assistant_response="Adding methodology now.",
        )

        assert result.followup_type == FollowupType.CONTINUATION
        assert result.signal is not None

    @pytest.mark.asyncio
    async def test_explicit_correction_across_domains(self, tmp_path):
        """Corrections work the same regardless of domain."""
        engine, mgr = await _make_engine(tmp_path)

        await engine.on_turn_complete(
            user_message="Recommend a deployment strategy",
            assistant_response=(
                "I recommend a waterfall approach."
                " Here's the plan."
                " Let me know if you need changes."
            ),
        )
        result = await engine.on_turn_complete(
            user_message="no, we use agile sprints exclusively",
            assistant_response="Got it, switching to agile.",
        )

        assert result.followup_type == FollowupType.CORRECTION
        patterns = await mgr.query_behavioral()
        assert any(p.pattern_type == "behavioral_correction" for p in patterns)

    @pytest.mark.asyncio
    async def test_financial_analysis(self, tmp_path):
        """Financial domain: model forgets sensitivity analysis."""
        engine, mgr = await _make_engine(tmp_path)

        await engine.on_turn_complete(
            user_message="Run a DCF analysis on the acquisition target",
            assistant_response=(
                "Here's the DCF analysis with projected cash"
                " flows. The NPV is $12M."
                " Let me know if you need anything else."
            ),
        )
        result = await engine.on_turn_complete(
            user_message="add a sensitivity table",
            assistant_response="Adding sensitivity table now.",
        )

        assert result.followup_type == FollowupType.CONTINUATION
        assert result.signal is not None
