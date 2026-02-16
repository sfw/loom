"""Learning system: extract patterns from execution and interaction.

Two learning modes:
1. Post-task: After autonomous task completion, extract operational patterns
   (model success, retry hints, plan templates). Informs model selection and planning.
2. Per-turn: After every user prompt in interactive sessions, extract behavioral
   patterns (corrections, preferences, style, domain knowledge). Informs prompt
   construction so users never repeat themselves.

Both modes store patterns in the same table with frequency-based reinforcement.
Behavioral patterns are domain-agnostic — they work for coding, writing,
analysis, or any other task type.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

from loom.state.memory import Database
from loom.state.task_state import Task, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class LearnedPattern:
    """A single learned pattern extracted from execution history."""

    pattern_type: str  # subtask_success, model_failure, retry_pattern, etc.
    pattern_key: str  # Searchable identifier
    data: dict = field(default_factory=dict)
    frequency: int = 1
    last_seen: str = ""
    id: int | None = None

    def __post_init__(self):
        if not self.last_seen:
            self.last_seen = datetime.now().isoformat()


class LearningManager:
    """Extracts and queries learned patterns from task execution."""

    def __init__(self, db: Database):
        self._db = db

    async def learn_from_task(self, task: Task) -> list[LearnedPattern]:
        """Analyze a completed/failed task and extract reusable patterns.

        Runs after task completion, not during execution.
        """
        patterns: list[LearnedPattern] = []

        # 1. Model success rates per subtask type
        for subtask in task.plan.subtasks:
            if subtask.summary:
                success = subtask.status.value == "completed"
                patterns.append(LearnedPattern(
                    pattern_type="subtask_success",
                    pattern_key=self._subtask_type_key(subtask.id),
                    data={
                        "success": success,
                        "retries": subtask.retry_count,
                        "model_tier": subtask.model_tier,
                        "description": subtask.description[:100],
                    },
                ))

        # 2. Retry patterns — which subtasks needed escalation?
        escalated = [s for s in task.plan.subtasks if s.retry_count > 0]
        for subtask in escalated:
            patterns.append(LearnedPattern(
                pattern_type="retry_pattern",
                pattern_key=self._subtask_type_key(subtask.id),
                data={
                    "retries_needed": subtask.retry_count,
                    "final_status": subtask.status.value,
                    "original_tier": subtask.model_tier,
                },
            ))

        # 3. Successful task plans as templates
        if task.status == TaskStatus.COMPLETED:
            patterns.append(LearnedPattern(
                pattern_type="task_template",
                pattern_key=self._goal_type_key(task.goal),
                data={
                    "goal": task.goal,
                    "plan": [
                        {"id": s.id, "description": s.description}
                        for s in task.plan.subtasks
                    ],
                    "subtask_count": len(task.plan.subtasks),
                },
            ))

        # 4. Error patterns from failures
        for error in task.errors_encountered:
            patterns.append(LearnedPattern(
                pattern_type="model_failure",
                pattern_key=f"error-{error.subtask}",
                data={
                    "subtask": error.subtask,
                    "error": error.error[:200],
                    "resolution": error.resolution or "",
                },
            ))

        # Store/update patterns
        stored = []
        for pattern in patterns:
            try:
                await self.store_or_update(pattern)
                stored.append(pattern)
            except Exception as e:
                logger.warning("Failed to store pattern: %s", e)

        return stored

    async def query_patterns(
        self,
        pattern_type: str | None = None,
        pattern_key: str | None = None,
        limit: int = 20,
    ) -> list[LearnedPattern]:
        """Query learned patterns with flexible filters."""
        conditions = []
        params: list = []

        if pattern_type:
            conditions.append("pattern_type = ?")
            params.append(pattern_type)

        if pattern_key:
            conditions.append("pattern_key = ?")
            params.append(pattern_key)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""SELECT * FROM learned_patterns
                  WHERE {where}
                  ORDER BY frequency DESC, last_seen DESC
                  LIMIT ?"""
        params.append(limit)

        rows = await self._db.query(sql, tuple(params))
        return [self._row_to_pattern(r) for r in rows]

    async def get_task_template(self, goal: str) -> LearnedPattern | None:
        """Find the best matching task template for a goal."""
        key = self._goal_type_key(goal)
        patterns = await self.query_patterns(
            pattern_type="task_template", pattern_key=key, limit=1,
        )
        return patterns[0] if patterns else None

    async def get_retry_hint(self, subtask_id: str) -> int | None:
        """Check if we have a learned min tier for a subtask type.

        Returns the recommended minimum tier, or None.
        """
        key = self._subtask_type_key(subtask_id)
        patterns = await self.query_patterns(
            pattern_type="retry_pattern", pattern_key=key,
        )
        if not patterns:
            return None

        # If average retries > 1.5, recommend starting at tier 2
        total_retries = sum(p.data.get("retries_needed", 0) for p in patterns)
        avg_retries = total_retries / len(patterns)
        if avg_retries > 1.5:
            return 2
        return None

    async def prune_stale(self, max_age_days: int = 90) -> int:
        """Remove low-frequency patterns older than max_age_days.

        Returns the number of patterns pruned.
        """
        cutoff = datetime.now().isoformat()[:10]  # Just date part
        result = await self._db.query(
            """SELECT COUNT(*) as cnt FROM learned_patterns
               WHERE frequency = 1
               AND date(last_seen) < date(?, '-' || ? || ' days')""",
            (cutoff, max_age_days),
        )
        count = result[0]["cnt"] if result else 0

        if count > 0:
            await self._db.execute(
                """DELETE FROM learned_patterns
                   WHERE frequency = 1
                   AND date(last_seen) < date(?, '-' || ? || ' days')""",
                (cutoff, max_age_days),
            )

        return count

    async def clear_all(self) -> None:
        """Clear all learned patterns (reset learning)."""
        await self._db.execute("DELETE FROM learned_patterns")

    async def query_behavioral(self, limit: int = 15) -> list[LearnedPattern]:
        """Query all behavioral patterns, sorted by frequency.

        Returns patterns from reflection (corrections, preferences, style,
        domain knowledge) — everything needed to personalize prompts.
        """
        sql = """SELECT * FROM learned_patterns
                 WHERE pattern_type LIKE 'behavioral_%'
                    OR pattern_type = 'user_correction'
                 ORDER BY frequency DESC, last_seen DESC
                 LIMIT ?"""
        rows = await self._db.query(sql, (limit,))
        return [self._row_to_pattern(r) for r in rows]

    async def store_or_update(self, pattern: LearnedPattern) -> None:
        """Store a new pattern or update frequency of an existing one.

        Public API — used by both post-task learning and per-turn reflection.
        """
        existing = await self._db.query_one(
            "SELECT id, frequency FROM learned_patterns WHERE pattern_type=? AND pattern_key=?",
            (pattern.pattern_type, pattern.pattern_key),
        )

        if existing:
            await self._db.execute(
                """UPDATE learned_patterns
                   SET frequency = frequency + 1,
                       last_seen = ?,
                       data = ?
                   WHERE id = ?""",
                (
                    datetime.now().isoformat(),
                    json.dumps(pattern.data),
                    existing["id"],
                ),
            )
        else:
            await self._db.execute(
                """INSERT INTO learned_patterns
                   (pattern_type, pattern_key, data, frequency, last_seen)
                   VALUES (?, ?, ?, 1, ?)""",
                (
                    pattern.pattern_type,
                    pattern.pattern_key,
                    json.dumps(pattern.data),
                    datetime.now().isoformat(),
                ),
            )

    @staticmethod
    def _subtask_type_key(subtask_id: str) -> str:
        """Generate a type key for a subtask based on its ID."""
        return subtask_id

    @staticmethod
    def _goal_type_key(goal: str) -> str:
        """Generate a type key for a task goal via keyword extraction."""
        keywords = sorted(set(goal.lower().split()))[:5]
        return "-".join(keywords)

    @staticmethod
    def _row_to_pattern(row: dict) -> LearnedPattern:
        data = row.get("data", "{}")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = {}
        return LearnedPattern(
            id=row.get("id"),
            pattern_type=row.get("pattern_type", ""),
            pattern_key=row.get("pattern_key", ""),
            data=data,
            frequency=row.get("frequency", 1),
            last_seen=row.get("last_seen", ""),
        )
