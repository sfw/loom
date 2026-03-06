"""Migration step registry."""

from __future__ import annotations

import hashlib

from loom.state.migrations.runner import MigrationStep
from loom.state.migrations.steps import events_v2, task_questions, validity_lineage


def _checksum(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


MIGRATIONS: tuple[MigrationStep, ...] = (
    MigrationStep(
        id="20260306_001_events_v2",
        description="Upgrade events table with run/event/sequence fields and indexes.",
        checksum=_checksum("20260306_001_events_v2/events_v2"),
        apply=events_v2.apply,
        verify=events_v2.verify,
    ),
    MigrationStep(
        id="20260306_002_task_questions",
        description="Add durable ask-user question lifecycle table and indexes.",
        checksum=_checksum("20260306_002_task_questions/task_questions"),
        apply=task_questions.apply,
        verify=task_questions.verify,
    ),
    MigrationStep(
        id="20260306_003_validity_lineage",
        description="Add claim/evidence validity lineage tables and indexes.",
        checksum=_checksum("20260306_003_validity_lineage/validity_lineage"),
        apply=validity_lineage.apply,
        verify=validity_lineage.verify,
    ),
)
