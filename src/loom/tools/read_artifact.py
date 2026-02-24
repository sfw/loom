"""Tool for reading previously persisted fetched artifacts by reference."""

from __future__ import annotations

from loom.ingest.artifacts import resolve_fetch_artifact
from loom.ingest.handlers import summarize_artifact
from loom.ingest.router import ContentKind
from loom.tools.registry import Tool, ToolContext, ToolResult

DEFAULT_ARTIFACT_SUMMARY_MAX_CHARS = 12_000
MIN_ARTIFACT_SUMMARY_MAX_CHARS = 400
MAX_ARTIFACT_SUMMARY_MAX_CHARS = 120_000


class ReadArtifactTool(Tool):
    @property
    def name(self) -> str:
        return "read_artifact"

    @property
    def description(self) -> str:
        return (
            "Read a previously fetched artifact via artifact_ref. "
            "Returns extracted/summarized text without embedding raw binary payloads."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "artifact_ref": {
                    "type": "string",
                    "description": "Artifact reference id (e.g. af_1234abcd...)",
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Upper bound for returned extracted text summary. "
                        f"Default: {DEFAULT_ARTIFACT_SUMMARY_MAX_CHARS}."
                    ),
                },
            },
            "required": ["artifact_ref"],
        }

    @property
    def timeout_seconds(self) -> int:
        return 30

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        artifact_ref = str(args.get("artifact_ref", "")).strip()
        if not artifact_ref:
            return ToolResult.fail("artifact_ref is required")

        max_chars_raw = args.get("max_chars", DEFAULT_ARTIFACT_SUMMARY_MAX_CHARS)
        try:
            max_chars = int(max_chars_raw)
        except (TypeError, ValueError):
            max_chars = DEFAULT_ARTIFACT_SUMMARY_MAX_CHARS
        max_chars = max(
            MIN_ARTIFACT_SUMMARY_MAX_CHARS,
            min(MAX_ARTIFACT_SUMMARY_MAX_CHARS, max_chars),
        )

        record = resolve_fetch_artifact(
            artifact_ref=artifact_ref,
            workspace=ctx.workspace,
            scratch_dir=ctx.scratch_dir,
            subtask_id=ctx.subtask_id,
        )
        if record is None:
            return ToolResult.fail(
                f"Artifact not found: {artifact_ref}. "
                "Use artifact_ref values from web_fetch/web_fetch_html results.",
            )
        if not record.path.exists():
            return ToolResult.fail(f"Artifact missing on disk: {artifact_ref}")

        content_kind = str(record.content_kind or "").strip() or ContentKind.UNKNOWN_BINARY
        media_type = str(record.media_type or "").strip()

        try:
            summary = summarize_artifact(
                path=record.path,
                content_kind=content_kind,
                media_type=media_type,
                max_chars=max_chars,
            )
        except Exception as e:
            return ToolResult.fail(f"Failed reading artifact {artifact_ref}: {e}")

        data: dict[str, object] = {
            "artifact_ref": record.artifact_ref,
            "artifact_path": str(record.path),
            "content_kind": content_kind,
            "content_type": media_type,
            "url": record.source_url,
            "source_url": record.source_url,
            "size_bytes": record.size_bytes,
            "created_at": record.created_at,
            "handler": summary.handler,
            "extracted_chars": len(summary.extracted_text),
            "extraction_truncated": bool(summary.extraction_truncated),
        }
        if record.workspace_relpath:
            data["artifact_workspace_relpath"] = record.workspace_relpath
        if summary.metadata:
            data["handler_metadata"] = summary.metadata

        return ToolResult.ok(summary.summary_text, data=data)
