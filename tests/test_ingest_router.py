"""Tests for ingest content-kind routing and artifact persistence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from loom.ingest.artifacts import (
    cleanup_fetch_artifacts,
    persist_fetch_artifact,
    resolve_fetch_artifact,
)
from loom.ingest.router import ContentKind, detect_content_kind


class TestIngestRouter:
    def test_detects_pdf_by_magic_even_when_mislabeled(self):
        kind = detect_content_kind(
            content_type="application/octet-stream",
            content_bytes=b"%PDF-1.7\n%....",
            url="https://example.com/file.bin",
        )
        assert kind == ContentKind.PDF

    def test_detects_office_doc_by_extension_with_zip_magic(self):
        kind = detect_content_kind(
            content_type="application/octet-stream",
            content_bytes=b"PK\x03\x04....",
            url="https://example.com/q4-report.pptx",
        )
        assert kind == ContentKind.OFFICE_DOC

    def test_detects_html_from_markup_sniff(self):
        kind = detect_content_kind(
            content_type="text/plain",
            content_bytes=b"<!DOCTYPE html><html><body>hello</body></html>",
            url="https://example.com",
        )
        assert kind == ContentKind.HTML


class TestArtifactStore:
    def test_persist_fetch_artifact_under_workspace(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        record = persist_fetch_artifact(
            content_bytes=b"%PDF-1.4\n",
            source_url="https://example.com/report.pdf",
            media_type="application/pdf",
            content_kind=ContentKind.PDF,
            workspace=workspace,
            subtask_id="analyze-market-trends",
        )

        assert record.path.exists()
        assert record.artifact_ref.startswith("af_")
        assert record.workspace_relpath.startswith(".loom_artifacts/")
        manifest = record.path.parent / "manifest.jsonl"
        assert manifest.exists()

    def test_resolve_fetch_artifact_by_ref(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        stored = persist_fetch_artifact(
            content_bytes=b"%PDF-1.4\n",
            source_url="https://example.com/report.pdf",
            media_type="application/pdf",
            content_kind=ContentKind.PDF,
            workspace=workspace,
            subtask_id="scope-a",
        )

        resolved = resolve_fetch_artifact(
            artifact_ref=stored.artifact_ref,
            workspace=workspace,
            subtask_id="scope-a",
        )
        assert resolved is not None
        assert resolved.artifact_ref == stored.artifact_ref
        assert resolved.path == stored.path
        assert resolved.content_kind == ContentKind.PDF

    def test_cleanup_fetch_artifacts_prunes_scope_by_count(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        subtask_id = "cleanup-scope"

        for idx in range(3):
            persist_fetch_artifact(
                content_bytes=f"payload-{idx}".encode(),
                source_url=f"https://example.com/file-{idx}.bin",
                media_type="application/octet-stream",
                content_kind=ContentKind.UNKNOWN_BINARY,
                workspace=workspace,
                subtask_id=subtask_id,
            )

        stats = cleanup_fetch_artifacts(
            workspace=workspace,
            subtask_id=subtask_id,
            max_age_days=0,
            max_files_per_scope=1,
            max_bytes_per_scope=10_000_000,
            scan_all_scopes=False,
        )
        assert stats["files_deleted"] >= 2

        scope_dir = workspace / ".loom_artifacts" / "fetched" / "cleanup-scope"
        kept_files = [
            path for path in scope_dir.iterdir()
            if path.is_file() and path.name != "manifest.jsonl"
        ]
        assert len(kept_files) <= 1

    def test_cleanup_fetch_artifacts_prunes_scope_by_age(self, tmp_path: Path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        subtask_id = "age-scope"

        record = persist_fetch_artifact(
            content_bytes=b"old",
            source_url="https://example.com/old.bin",
            media_type="application/octet-stream",
            content_kind=ContentKind.UNKNOWN_BINARY,
            workspace=workspace,
            subtask_id=subtask_id,
        )

        old_ts = (datetime.now(UTC) - timedelta(days=45)).timestamp()
        # Keep mtime stale so age pruning can remove this artifact.
        import os
        os.utime(record.path, (old_ts, old_ts))

        stats = cleanup_fetch_artifacts(
            workspace=workspace,
            subtask_id=subtask_id,
            max_age_days=7,
            max_files_per_scope=100,
            max_bytes_per_scope=10_000_000,
            scan_all_scopes=False,
        )
        assert stats["files_deleted"] >= 1
        assert not record.path.exists()
