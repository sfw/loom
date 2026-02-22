"""Claim verification tool with deterministic evidence scoring."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from loom.research.models import FactCheckVerdict
from loom.research.text import is_primary_domain, jaccard_similarity, normalize_text, tokenize
from loom.tools.registry import Tool, ToolContext, ToolResult

FETCH_TIMEOUT = 12.0
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SourceDoc:
    label: str
    text: str
    origin: str
    is_primary: bool


class FactCheckerTool(Tool):
    """Verify claims against provided source material."""

    @property
    def name(self) -> str:
        return "fact_checker"

    @property
    def description(self) -> str:
        return (
            "Check claims against source documents/URLs and return structured "
            "verdicts with confidence and rationale."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Inline claims to verify.",
                },
                "claims_path": {
                    "type": "string",
                    "description": "Path to claims file (txt/csv/json).",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Source URLs or file paths.",
                },
                "source_index_path": {
                    "type": "string",
                    "description": "Source index file with url/path entries.",
                },
                "strictness": {
                    "type": "string",
                    "enum": ["lenient", "standard", "strict"],
                },
                "require_primary_sources": {
                    "type": "boolean",
                    "description": "Require primary-ish domains for strong verdicts.",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown report path.",
                },
                "output_csv_path": {
                    "type": "string",
                    "description": "Optional csv report path.",
                },
            },
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        claims = _load_claims(args.get("claims"), args.get("claims_path"), self, ctx)
        if not claims:
            return ToolResult.fail("No claims provided")

        sources = await _load_sources(
            args.get("sources"),
            args.get("source_index_path"),
            self,
            ctx,
        )
        if not sources:
            return ToolResult.fail("No usable sources provided")

        strictness = str(args.get("strictness", "standard")).strip().lower() or "standard"
        if strictness not in {"lenient", "standard", "strict"}:
            return ToolResult.fail("strictness must be lenient, standard, or strict")

        require_primary = bool(args.get("require_primary_sources", False))
        thresholds = {
            "lenient": {"supported": 0.55, "partial": 0.35},
            "standard": {"supported": 0.70, "partial": 0.45},
            "strict": {"supported": 0.82, "partial": 0.55},
        }[strictness]

        verdicts: list[FactCheckVerdict] = []
        for claim in claims:
            verdicts.append(
                _check_one_claim(
                    claim,
                    sources,
                    support_threshold=thresholds["supported"],
                    partial_threshold=thresholds["partial"],
                    require_primary=require_primary,
                )
            )

        counts = {
            "supported": 0,
            "partially_supported": 0,
            "contradicted": 0,
            "unverifiable": 0,
        }
        for verdict in verdicts:
            counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1

        files_changed: list[str] = []
        if ctx.workspace is not None:
            md_raw = str(args.get("output_path", "fact-check-report.md")).strip()
            csv_raw = str(args.get("output_csv_path", "fact-check-report.csv")).strip()
            md_path = self._resolve_path(md_raw, ctx.workspace)
            csv_path = self._resolve_path(csv_raw, ctx.workspace)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(md_path), subtask_id=ctx.subtask_id)
                ctx.changelog.record_before_write(str(csv_path), subtask_id=ctx.subtask_id)

            md_path.write_text(_render_markdown(verdicts, counts), encoding="utf-8")
            _write_csv(csv_path, verdicts)

            files_changed.extend(
                [
                    str(md_path.relative_to(ctx.workspace)),
                    str(csv_path.relative_to(ctx.workspace)),
                ]
            )

        output = (
            "Fact-check completed: "
            f"supported={counts['supported']}, "
            f"partially_supported={counts['partially_supported']}, "
            f"contradicted={counts['contradicted']}, "
            f"unverifiable={counts['unverifiable']}"
        )
        if files_changed:
            output += "\nArtifacts: " + ", ".join(files_changed)

        return ToolResult.ok(
            output,
            files_changed=files_changed,
            data={
                "counts": counts,
                "claims": len(claims),
                "sources": len(sources),
                "strictness": strictness,
                "require_primary_sources": require_primary,
                "verdicts": [verdict.to_dict() for verdict in verdicts],
            },
        )


def _load_claims(
    claims_raw: object,
    claims_path_raw: object,
    tool: Tool,
    ctx: ToolContext,
) -> list[str]:
    claims: list[str] = []
    if isinstance(claims_raw, list):
        claims.extend(str(item).strip() for item in claims_raw if str(item).strip())

    claims_path = str(claims_path_raw or "").strip()
    if claims_path:
        if ctx.workspace is None:
            return []
        path = tool._resolve_read_path(claims_path, ctx.workspace, ctx.read_roots)
        if path.exists() and path.is_file():
            claims.extend(_read_claims_file(path))

    deduped: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        key = normalize_text(claim)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


def _read_claims_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("claims", [])
        else:
            items = payload
        if isinstance(items, list):
            return [str(item).strip() for item in items if str(item).strip()]
        return []

    if suffix == ".csv":
        out: list[str] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                claim = str(row.get("claim", "")).strip()
                if claim:
                    out.append(claim)
        return out

    text = path.read_text(encoding="utf-8")
    return [line.strip("- \t") for line in text.splitlines() if line.strip()]


async def _load_sources(
    sources_raw: object,
    source_index_path_raw: object,
    tool: Tool,
    ctx: ToolContext,
) -> list[SourceDoc]:
    refs: list[str] = []
    if isinstance(sources_raw, list):
        refs.extend(str(item).strip() for item in sources_raw if str(item).strip())

    index_path = str(source_index_path_raw or "").strip()
    if index_path:
        if ctx.workspace is None:
            return []
        path = tool._resolve_read_path(index_path, ctx.workspace, ctx.read_roots)
        if path.exists() and path.is_file():
            refs.extend(_read_source_index(path))

    refs = list(dict.fromkeys(refs))
    if not refs:
        return []

    local_paths: list[str] = []
    urls: list[str] = []
    for ref in refs:
        if ref.lower().startswith(("http://", "https://")):
            urls.append(ref)
        else:
            local_paths.append(ref)

    docs: list[SourceDoc] = []
    for ref in local_paths:
        if ctx.workspace is None:
            continue
        try:
            path = tool._resolve_read_path(ref, ctx.workspace, ctx.read_roots)
        except Exception:
            continue
        if not path.exists() or not path.is_file():
            continue
        text = _extract_local_text(path)
        if not text.strip():
            continue
        docs.append(
            SourceDoc(
                label=str(path.name),
                text=text,
                origin=str(path),
                is_primary=True,
            )
        )

    if urls:
        headers = {
            "User-Agent": os.environ.get(
                "LOOM_WEB_USER_AGENT",
                "Loom/1.0 (+https://github.com/sfw/loom)",
            ),
            "Accept": "text/html,application/json,text/plain,*/*",
        }
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(FETCH_TIMEOUT),
            headers=headers,
        ) as client:
            for url in urls:
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                except Exception:
                    continue
                body = response.text or ""
                text = _extract_text_from_web(body)
                if not text:
                    continue
                docs.append(
                    SourceDoc(
                        label=url,
                        text=text,
                        origin=url,
                        is_primary=is_primary_domain(url),
                    )
                )

    return docs


def _read_source_index(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        items: list[Any]
        if isinstance(payload, dict):
            items = payload.get("sources", [])
        elif isinstance(payload, list):
            items = payload
        else:
            items = []
        out: list[str] = []
        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
            elif isinstance(item, dict):
                text = str(item.get("url", item.get("path", ""))).strip()
                if text:
                    out.append(text)
        return out

    if suffix == ".csv":
        out: list[str] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                text = str(row.get("url", row.get("path", ""))).strip()
                if text:
                    out.append(text)
        return out

    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_local_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(payload)
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_text_from_web(body: str) -> str:
    text = _HTML_TAG_RE.sub(" ", body)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _contradiction_score(claim_norm: str, source_norm: str) -> float:
    if not claim_norm or not source_norm:
        return 0.0

    patterns = [
        f"not {claim_norm}",
        f"false {claim_norm}",
        f"incorrect {claim_norm}",
    ]
    for pattern in patterns:
        if pattern in source_norm:
            return 0.95

    if " is " in claim_norm:
        subject, predicate = claim_norm.split(" is ", 1)
        if subject and predicate and subject in source_norm and f"not {predicate}" in source_norm:
            return 0.8

    if " are " in claim_norm:
        subject, predicate = claim_norm.split(" are ", 1)
        if subject and predicate and subject in source_norm and f"not {predicate}" in source_norm:
            return 0.8

    return 0.0


def _best_excerpt(source_text: str, claim_norm: str) -> str:
    source_norm = normalize_text(source_text)
    index = source_norm.find(claim_norm)
    if index >= 0:
        start = max(0, index - 120)
        end = min(len(source_norm), index + len(claim_norm) + 120)
        return source_norm[start:end]
    return source_norm[:240]


def _check_one_claim(
    claim: str,
    sources: list[SourceDoc],
    *,
    support_threshold: float,
    partial_threshold: float,
    require_primary: bool,
) -> FactCheckVerdict:
    claim_norm = normalize_text(claim)
    claim_tokens = tokenize(claim)

    best_support = 0.0
    support_source: SourceDoc | None = None
    support_exact = False

    best_contradiction = 0.0
    contradiction_source: SourceDoc | None = None

    for source in sources:
        source_norm = normalize_text(source.text)
        source_tokens = tokenize(source.text)
        sim = jaccard_similarity(claim_tokens, source_tokens)
        exact = claim_norm in source_norm
        if exact:
            sim = max(sim, 0.98)

        if sim > best_support:
            best_support = sim
            support_source = source
            support_exact = exact

        c_score = _contradiction_score(claim_norm, source_norm)
        if c_score > best_contradiction:
            best_contradiction = c_score
            contradiction_source = source

    verdict = "unverifiable"
    confidence = min(0.99, max(0.05, best_support))
    rationale = "No source provided enough matching evidence."
    chosen_source = support_source

    if best_contradiction >= 0.75 and best_contradiction >= best_support + 0.05:
        verdict = "contradicted"
        confidence = min(0.99, 0.55 + 0.45 * best_contradiction)
        rationale = "Source text contains explicit contradiction patterns."
        chosen_source = contradiction_source
    elif support_exact or best_support >= support_threshold:
        verdict = "supported"
        confidence = min(0.99, 0.6 + 0.4 * best_support)
        rationale = "High lexical overlap and/or direct match to source text."
    elif best_support >= partial_threshold:
        verdict = "partially_supported"
        confidence = min(0.9, 0.45 + 0.35 * best_support)
        rationale = "Claim partially overlaps evidence but is not fully confirmed."

    if require_primary and chosen_source is not None and not chosen_source.is_primary:
        if verdict == "supported":
            verdict = "partially_supported"
            rationale += " Downgraded: strongest source is not primary."
            confidence = max(0.05, confidence - 0.2)
        elif verdict == "partially_supported":
            verdict = "unverifiable"
            rationale += " Downgraded: evidence is non-primary only."
            confidence = max(0.05, confidence - 0.2)

    excerpt = ""
    source_label = ""
    if chosen_source is not None:
        excerpt = _best_excerpt(chosen_source.text, claim_norm)
        source_label = chosen_source.label

    return FactCheckVerdict(
        claim=claim,
        verdict=verdict,
        confidence=round(confidence, 4),
        rationale=rationale,
        source=source_label,
        source_excerpt=excerpt,
    )


def _render_markdown(verdicts: list[FactCheckVerdict], counts: dict[str, int]) -> str:
    lines = [
        "# Fact Check Report",
        "",
        "## Summary",
        "",
        f"- Supported: {counts.get('supported', 0)}",
        f"- Partially Supported: {counts.get('partially_supported', 0)}",
        f"- Contradicted: {counts.get('contradicted', 0)}",
        f"- Unverifiable: {counts.get('unverifiable', 0)}",
        "",
        "## Claim Verdicts",
        "",
    ]

    for idx, verdict in enumerate(verdicts, start=1):
        lines.extend(
            [
                f"### {idx}. {verdict.claim}",
                f"- Verdict: **{verdict.verdict}**",
                f"- Confidence: {verdict.confidence:.4f}",
                f"- Rationale: {verdict.rationale}",
                f"- Source: {verdict.source or 'N/A'}",
                f"- Excerpt: {verdict.source_excerpt or 'N/A'}",
                "",
            ]
        )

    return "\n".join(lines)


def _write_csv(path: Path, verdicts: list[FactCheckVerdict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "claim",
                "verdict",
                "confidence",
                "rationale",
                "source",
                "source_excerpt",
            ],
        )
        writer.writeheader()
        for verdict in verdicts:
            writer.writerow(verdict.to_dict())
