"""Ingest helpers for filetype-aware web/tool content handling."""

from __future__ import annotations

from loom.ingest.artifacts import (
    ArtifactRecord,
    cleanup_fetch_artifacts,
    persist_fetch_artifact,
    resolve_fetch_artifact,
)
from loom.ingest.handlers import ArtifactSummary, summarize_artifact
from loom.ingest.router import ContentKind, detect_content_kind

__all__ = [
    "ArtifactRecord",
    "ArtifactSummary",
    "ContentKind",
    "cleanup_fetch_artifacts",
    "detect_content_kind",
    "persist_fetch_artifact",
    "resolve_fetch_artifact",
    "summarize_artifact",
]
