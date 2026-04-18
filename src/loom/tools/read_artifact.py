"""Tool for reading previously persisted fetched artifacts by reference."""

from __future__ import annotations

import re

from loom.ingest.artifacts import resolve_fetch_artifact
from loom.ingest.handlers import summarize_artifact
from loom.ingest.router import ContentKind
from loom.research.text import jaccard_similarity, token_overlap_ratio, tokenize
from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.utils.tokens import estimate_tokens

DEFAULT_ARTIFACT_SUMMARY_MAX_CHARS = 12_000
MIN_ARTIFACT_SUMMARY_MAX_CHARS = 400
MAX_ARTIFACT_SUMMARY_MAX_CHARS = 120_000
DEFAULT_QUERY_SNIPPETS = 4
MAX_QUERY_SNIPPETS = 8
QUERY_CHUNK_TOKENS = 500
DEFAULT_QUERY_SNIPPET_CHARS = 900


class ReadArtifactTool(Tool):
    @property
    def name(self) -> str:
        return "read_artifact"

    @property
    def description(self) -> str:
        return (
            "Read a previously fetched artifact via artifact_ref. "
            "Returns extracted/summarized text without embedding raw binary payloads. "
            "For text-like artifacts, optional query can return only the most relevant snippets."
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
                "query": {
                    "type": "string",
                    "description": (
                        "Optional focus query for returning only the most relevant "
                        "snippets from a text-like artifact."
                    ),
                },
                "max_snippets": {
                    "type": "integer",
                    "description": (
                        "When query is provided, maximum number of relevant snippets "
                        f"to return. Default: {DEFAULT_QUERY_SNIPPETS}."
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
        query = str(args.get("query", "") or "").strip()
        max_snippets_raw = args.get("max_snippets", DEFAULT_QUERY_SNIPPETS)
        try:
            max_snippets = int(max_snippets_raw)
        except (TypeError, ValueError):
            max_snippets = DEFAULT_QUERY_SNIPPETS
        max_snippets = max(1, min(MAX_QUERY_SNIPPETS, max_snippets))

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
            text_like = _artifact_is_text_like(content_kind=content_kind, media_type=media_type)
            if text_like:
                raw_text = record.path.read_text(encoding="utf-8", errors="replace")
                output, retrieval_strategy = _render_text_artifact_view(
                    raw_text,
                    query=query,
                    max_chars=max_chars,
                    max_snippets=max_snippets,
                )
                data: dict[str, object] = {
                    "artifact_ref": record.artifact_ref,
                    "artifact_path": str(record.path),
                    "content_kind": content_kind,
                    "content_type": media_type,
                    "url": record.source_url,
                    "source_url": record.source_url,
                    "size_bytes": record.size_bytes,
                    "created_at": record.created_at,
                    "handler": "text_artifact_reader",
                    "extracted_chars": len(raw_text),
                    "extraction_truncated": retrieval_strategy != "full_text",
                    "retrieval_strategy": retrieval_strategy,
                }
                if query:
                    data["query"] = query
                    data["max_snippets"] = max_snippets
                if record.workspace_relpath:
                    data["artifact_workspace_relpath"] = record.workspace_relpath
                return ToolResult.ok(output, data=data)

            summary = summarize_artifact(
                path=record.path,
                content_kind=content_kind,
                media_type=media_type,
                max_chars=max_chars,
            )
        except Exception as e:
            return ToolResult.fail(f"Failed reading artifact {artifact_ref}: {e}")

        output = summary.summary_text
        retrieval_strategy = "artifact_summary"
        if query and summary.extracted_text:
            output, retrieval_strategy = _render_text_artifact_view(
                summary.extracted_text,
                query=query,
                max_chars=max_chars,
                max_snippets=max_snippets,
            )

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
            "retrieval_strategy": retrieval_strategy,
        }
        if query:
            data["query"] = query
            data["max_snippets"] = max_snippets
        if record.workspace_relpath:
            data["artifact_workspace_relpath"] = record.workspace_relpath
        if summary.metadata:
            data["handler_metadata"] = summary.metadata

        return ToolResult.ok(output, data=data)


def _artifact_is_text_like(*, content_kind: str, media_type: str) -> bool:
    if content_kind in {ContentKind.TEXT, ContentKind.HTML}:
        return True
    media = str(media_type or "").lower()
    return media.startswith("text/") or media in {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/x-toml",
    }


def _render_text_artifact_view(
    text: str,
    *,
    query: str,
    max_chars: int,
    max_snippets: int,
) -> tuple[str, str]:
    clean_text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not clean_text:
        return "Artifact text is empty.", "empty"
    if query:
        return _query_text(clean_text, query=query, max_chars=max_chars, max_snippets=max_snippets)
    if len(clean_text) <= max_chars:
        return clean_text, "full_text"
    return _head_tail_excerpt(clean_text, max_chars=max_chars), "head_tail_excerpt"


def _query_text(
    text: str,
    *,
    query: str,
    max_chars: int,
    max_snippets: int,
) -> tuple[str, str]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return _head_tail_excerpt(text, max_chars=max_chars), "head_tail_excerpt"

    chunks = _chunk_text(text, target_tokens=QUERY_CHUNK_TOKENS)
    ranked: list[tuple[float, int, str]] = []
    for idx, chunk in enumerate(chunks):
        chunk_tokens = tokenize(chunk)
        score = token_overlap_ratio(query_tokens, chunk_tokens) * 3.0
        score += jaccard_similarity(query_tokens, chunk_tokens)
        score += max(0.0, 0.25 - idx * 0.02)
        if score <= 0:
            continue
        ranked.append((score, idx, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    if not ranked:
        output = (
            f"No strong lexical matches found for query: {query}\n\n"
            + _head_tail_excerpt(text, max_chars=max_chars)
        )
        return _fit_text(output, max_chars=max_chars), "query_no_match"

    snippet_chars = max(
        220,
        min(DEFAULT_QUERY_SNIPPET_CHARS, max_chars // max(1, max_snippets)),
    )
    lines = [
        f"Relevant snippets for query: {query}",
    ]
    for idx, (score, _chunk_idx, chunk) in enumerate(ranked[:max_snippets], 1):
        lines.append(
            f"{idx}. [score {score:.2f}] "
            + _best_chunk_excerpt(chunk, query_tokens=query_tokens, max_chars=snippet_chars),
        )
    return _fit_text("\n\n".join(lines).strip(), max_chars=max_chars), "query_snippets"


def _chunk_text(text: str, *, target_tokens: int) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    if not blocks:
        return [text] if text else []
    chunks: list[str] = []
    current: list[str] = []
    for block in blocks:
        tentative = "\n\n".join(current + [block]).strip()
        if current and estimate_tokens(tentative) > target_tokens:
            chunks.append("\n\n".join(current).strip())
            current = [block]
        else:
            current.append(block)
    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _best_chunk_excerpt(chunk: str, *, query_tokens: set[str], max_chars: int) -> str:
    best_text = chunk
    best_score = float("-inf")
    for block in [item.strip() for item in re.split(r"\n{2,}", chunk) if item.strip()]:
        block_tokens = tokenize(block)
        score = token_overlap_ratio(query_tokens, block_tokens) * 3.0
        score += jaccard_similarity(query_tokens, block_tokens)
        if score > best_score:
            best_score = score
            best_text = block
    return _excerpt(best_text, max_chars=max_chars, query_tokens=query_tokens)


def _excerpt(text: str, *, max_chars: int, query_tokens: set[str]) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    lower = value.lower()
    positions = [
        lower.find(token.lower())
        for token in query_tokens
        if token and lower.find(token.lower()) >= 0
    ]
    if positions:
        anchor = min(positions)
        start = max(0, anchor - (max_chars // 3))
        end = min(len(value), start + max_chars)
        excerpt = value[start:end].strip()
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(value):
            excerpt += "..."
        return excerpt
    head = max_chars - 18
    if head <= 0:
        return value[:max_chars]
    return value[:head].rstrip() + " ...[excerpt]"


def _head_tail_excerpt(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 80:
        return text[:max_chars]
    marker = "\n\n[... middle omitted ...]\n\n"
    remaining = max_chars - len(marker)
    if remaining <= 40:
        return text[:max_chars]
    head_len = max(20, int(remaining * 0.6))
    tail_len = max(20, remaining - head_len)
    return text[:head_len].rstrip() + marker + text[-tail_len:].lstrip()


def _fit_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return _head_tail_excerpt(text, max_chars=max_chars)
