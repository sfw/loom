"""Tests for the learning system."""

from __future__ import annotations

import pytest

from loom.learning.manager import LearnedPattern, LearningManager
from loom.state.memory import Database
from loom.state.task_state import (
    Plan,
    Subtask,
    SubtaskStatus,
    Task,
    TaskStatus,
)

# --- Helpers ---


def _make_task(
    task_id: str = "t1",
    goal: str = "Build a CLI tool",
    status: TaskStatus = TaskStatus.COMPLETED,
    subtasks: list[Subtask] | None = None,
) -> Task:
    default_subtasks = subtasks or [
        Subtask(
            id="s1",
            description="Create main.py",
            status=SubtaskStatus.COMPLETED,
            summary="Created main.py with CLI structure",
            model_tier=1,
        ),
        Subtask(
            id="s2",
            description="Add argument parser",
            status=SubtaskStatus.COMPLETED,
            summary="Added argparse configuration",
            model_tier=1,
            depends_on=["s1"],
        ),
    ]
    task = Task(
        id=task_id,
        goal=goal,
        status=status,
        plan=Plan(subtasks=default_subtasks, version=1),
        created_at="2025-01-01T00:00:00",
    )
    return task


def _make_failed_task() -> Task:
    task = _make_task(status=TaskStatus.FAILED)
    task.plan.subtasks[1].status = SubtaskStatus.FAILED
    task.plan.subtasks[1].retry_count = 2
    task.plan.subtasks[1].summary = "Failed to parse arguments"
    task.add_error("s2", "SyntaxError in generated code")
    return task


# --- Tests ---


class TestLearnedPattern:
    def test_auto_last_seen(self):
        p = LearnedPattern(pattern_type="test", pattern_key="k1")
        assert p.last_seen != ""

    def test_explicit_last_seen(self):
        p = LearnedPattern(
            pattern_type="test", pattern_key="k1",
            last_seen="2025-01-01",
        )
        assert p.last_seen == "2025-01-01"


class TestLearningManager:
    @pytest.mark.asyncio
    async def test_learn_from_completed_task(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        task = _make_task()
        patterns = await mgr.learn_from_task(task)

        # Should have: 2 subtask_success + 1 task_template = 3
        assert len(patterns) >= 3
        types = [p.pattern_type for p in patterns]
        assert "subtask_success" in types
        assert "task_template" in types

    @pytest.mark.asyncio
    async def test_learn_from_failed_task(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        task = _make_failed_task()
        patterns = await mgr.learn_from_task(task)

        types = [p.pattern_type for p in patterns]
        assert "retry_pattern" in types
        assert "model_failure" in types

    @pytest.mark.asyncio
    async def test_query_patterns_by_type(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        task = _make_task()
        await mgr.learn_from_task(task)

        patterns = await mgr.query_patterns(pattern_type="task_template")
        assert len(patterns) == 1
        assert patterns[0].pattern_type == "task_template"

    @pytest.mark.asyncio
    async def test_query_patterns_by_key(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        task = _make_task()
        await mgr.learn_from_task(task)

        patterns = await mgr.query_patterns(pattern_key="s1")
        assert len(patterns) >= 1
        assert patterns[0].pattern_key == "s1"

    @pytest.mark.asyncio
    async def test_frequency_updates(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        # Run learning twice with similar tasks
        task1 = _make_task(task_id="t1")
        task2 = _make_task(task_id="t2")
        await mgr.learn_from_task(task1)
        await mgr.learn_from_task(task2)

        patterns = await mgr.query_patterns(
            pattern_type="subtask_success", pattern_key="s1",
        )
        assert len(patterns) == 1
        assert patterns[0].frequency == 2

    @pytest.mark.asyncio
    async def test_get_task_template(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        task = _make_task()
        await mgr.learn_from_task(task)

        template = await mgr.get_task_template(task.goal)
        assert template is not None
        assert template.pattern_type == "task_template"
        assert "plan" in template.data

    @pytest.mark.asyncio
    async def test_get_task_template_not_found(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        template = await mgr.get_task_template("Something totally different")
        assert template is None

    @pytest.mark.asyncio
    async def test_get_retry_hint_no_data(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        hint = await mgr.get_retry_hint("unknown-subtask")
        assert hint is None

    @pytest.mark.asyncio
    async def test_get_retry_hint_with_data(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        # Create a task where s2 needed retries
        task = _make_failed_task()
        task.status = TaskStatus.COMPLETED  # But it eventually succeeded
        task.plan.subtasks[1].status = SubtaskStatus.COMPLETED
        await mgr.learn_from_task(task)

        # Query for retry hint
        hint = await mgr.get_retry_hint("s2")
        assert hint == 2  # Should recommend tier 2

    @pytest.mark.asyncio
    async def test_clear_all(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        task = _make_task()
        await mgr.learn_from_task(task)

        patterns_before = await mgr.query_patterns()
        assert len(patterns_before) > 0

        await mgr.clear_all()

        patterns_after = await mgr.query_patterns()
        assert len(patterns_after) == 0

    @pytest.mark.asyncio
    async def test_delete_pattern(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        await mgr.store_or_update(LearnedPattern(
            pattern_type="behavioral_gap",
            pattern_key="test-delete",
            data={"description": "Delete me"},
        ))
        patterns = await mgr.query_patterns(pattern_key="test-delete")
        assert len(patterns) == 1
        pid = patterns[0].id

        deleted = await mgr.delete_pattern(pid)
        assert deleted is True

        patterns = await mgr.query_patterns(pattern_key="test-delete")
        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_delete_pattern_not_found(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        deleted = await mgr.delete_pattern(9999)
        assert deleted is False

    @pytest.mark.asyncio
    async def test_query_all(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        await mgr.store_or_update(LearnedPattern(
            pattern_type="behavioral_gap",
            pattern_key="gap-1",
            data={"description": "Gap one"},
        ))
        await mgr.store_or_update(LearnedPattern(
            pattern_type="subtask_success",
            pattern_key="op-1",
            data={"success": True},
        ))

        all_patterns = await mgr.query_all()
        assert len(all_patterns) == 2
        types = {p.pattern_type for p in all_patterns}
        assert "behavioral_gap" in types
        assert "subtask_success" in types

    @pytest.mark.asyncio
    async def test_query_all_respects_limit(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        for i in range(10):
            await mgr.store_or_update(LearnedPattern(
                pattern_type="behavioral_gap",
                pattern_key=f"gap-{i}",
                data={"description": f"Gap {i}"},
            ))

        all_patterns = await mgr.query_all(limit=3)
        assert len(all_patterns) == 3

    @pytest.mark.asyncio
    async def test_subtask_type_key(self):
        assert LearningManager._subtask_type_key("install-deps") == "install-deps"

    @pytest.mark.asyncio
    async def test_goal_type_key(self):
        key = LearningManager._goal_type_key("Build a CLI tool")
        assert "a" in key
        assert "build" in key
        assert "cli" in key

    @pytest.mark.asyncio
    async def test_prune_stale(self, tmp_path):
        db = Database(str(tmp_path / "test.db"))
        await db.initialize()
        mgr = LearningManager(db)

        # Insert a pattern with old date
        await db.execute(
            """INSERT INTO learned_patterns (pattern_type, pattern_key, data, frequency, last_seen)
               VALUES (?, ?, ?, 1, ?)""",
            ("subtask_success", "old-key", '{}', "2020-01-01T00:00:00"),
        )

        # Insert a fresh pattern
        await db.execute(
            """INSERT INTO learned_patterns (pattern_type, pattern_key, data, frequency, last_seen)
               VALUES (?, ?, ?, 5, ?)""",
            ("subtask_success", "new-key", '{}', "2025-01-01T00:00:00"),
        )

        pruned = await mgr.prune_stale(max_age_days=90)
        assert pruned >= 1

        remaining = await mgr.query_patterns()
        keys = [p.pattern_key for p in remaining]
        assert "new-key" in keys
