"""Claim verification tool with hybrid deterministic + LLM evidence scoring."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from loom.config import Config
from loom.models.base import ModelProvider
from loom.models.retry import ModelRetryPolicy, call_with_model_retry
from loom.models.router import ModelRouter, ResponseValidator
from loom.research.models import FactCheckVerdict
from loom.research.text import (
    is_primary_domain,
    jaccard_similarity,
    normalize_text,
    token_overlap_ratio,
    tokenize,
)
from loom.tools.registry import Tool, ToolContext, ToolResult

FETCH_TIMEOUT = 12.0
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_DEFAULT_MAX_EVIDENCE_SNIPPETS = 6
_MAX_SNIPPET_CHARS = 460

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceDoc:
    label: str
    text: str
    origin: str
    is_primary: bool


@dataclass(frozen=True)
class EvidenceSnippet:
    snippet_id: str
    source_label: str
    source_origin: str
    is_primary: bool
    text: str
    score: float


class FactCheckerTool(Tool):
    """Verify claims against provided source material."""

    def __init__(self, config: Config | None = None):
        self._config = config
        self._router: ModelRouter | None = None
        self._response_validator = ResponseValidator()
        if config is not None:
            try:
                self._router = ModelRouter.from_config(config)
            except Exception:
                logger.debug("Failed to initialize model router for fact_checker", exc_info=True)

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
                "semantic_verification": {
                    "type": "string",
                    "enum": ["auto", "llm", "deterministic"],
                    "description": (
                        "Verification mode: auto uses router-backed verifier model "
                        "when available; llm requires model-based entailment when "
                        "available; deterministic skips model calls."
                    ),
                },
                "max_evidence_snippets": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "description": "Maximum retrieved evidence snippets per claim.",
                },
                "write_reports": {
                    "type": "boolean",
                    "description": (
                        "When true, write default markdown/csv artifacts unless "
                        "explicit output paths are provided."
                    ),
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

    @property
    def is_mutating(self) -> bool:
        return True

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

        semantic_mode = str(args.get("semantic_verification", "auto") or "auto").strip().lower()
        if semantic_mode not in {"auto", "llm", "deterministic"}:
            return ToolResult.fail("semantic_verification must be auto, llm, or deterministic")

        max_snippets = _coerce_int(
            args.get("max_evidence_snippets"),
            _DEFAULT_MAX_EVIDENCE_SNIPPETS,
        )
        max_snippets = max(1, min(12, max_snippets))

        require_primary = bool(args.get("require_primary_sources", False))
        thresholds = {
            "lenient": {"supported": 0.55, "partial": 0.35},
            "standard": {"supported": 0.70, "partial": 0.45},
            "strict": {"supported": 0.82, "partial": 0.55},
        }[strictness]

        verifier_model = self._select_verifier_model(mode=semantic_mode)
        effective_mode = "deterministic"
        if verifier_model is not None and semantic_mode in {"auto", "llm"}:
            effective_mode = "llm"

        verdicts: list[FactCheckVerdict] = []
        for claim in claims:
            verdicts.append(
                await _check_one_claim(
                    claim,
                    sources,
                    support_threshold=thresholds["supported"],
                    partial_threshold=thresholds["partial"],
                    require_primary=require_primary,
                    max_evidence_snippets=max_snippets,
                    verifier_model=verifier_model,
                    semantic_mode=semantic_mode,
                    config=self._config,
                    validator=self._response_validator,
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
        write_reports = bool(args.get("write_reports", False))
        md_raw = str(args.get("output_path", "")).strip()
        csv_raw = str(args.get("output_csv_path", "")).strip()
        if write_reports:
            if not md_raw:
                md_raw = "fact-check-report.md"
            if not csv_raw:
                csv_raw = "fact-check-report.csv"

        if md_raw or csv_raw:
            if ctx.workspace is None:
                return ToolResult.fail("No workspace set for output_path")
            files_changed = _write_optional_reports(
                tool=self,
                ctx=ctx,
                markdown_path=md_raw,
                csv_path=csv_raw,
                verdicts=verdicts,
                counts=counts,
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
                "semantic_verification_requested": semantic_mode,
                "semantic_verification_effective": effective_mode,
                "verifier_model": getattr(verifier_model, "name", ""),
                "verdicts": [verdict.to_dict() for verdict in verdicts],
            },
        )

    def _select_verifier_model(self, *, mode: str) -> ModelProvider | None:
        if mode == "deterministic" or self._router is None:
            return None
        candidates = [
            (2, "verifier"),
            (1, "verifier"),
            (2, "extractor"),
            (1, "extractor"),
            (2, "executor"),
        ]
        for tier, role in candidates:
            try:
                return self._router.select(tier=tier, role=role)
            except Exception:
                continue
        return None


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
        path = tool._resolve_read_path(
            claims_path,
            ctx.workspace,
            ctx.read_roots,
            ctx.read_path_map,
        )
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
        path = tool._resolve_read_path(
            index_path,
            ctx.workspace,
            ctx.read_roots,
            ctx.read_path_map,
        )
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
            path = tool._resolve_read_path(
                ref,
                ctx.workspace,
                ctx.read_roots,
                ctx.read_path_map,
            )
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


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _chunk_source_text(text: str, *, chunk_chars: int = 900, overlap_chars: int = 140) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    if not paragraphs:
        paragraphs = [raw]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_chars:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= chunk_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = paragraph
            continue

        pieces = [
            piece.strip()
            for piece in _SENTENCE_BOUNDARY_RE.split(paragraph)
            if piece.strip()
        ]
        if not pieces:
            pieces = [paragraph]
        for piece in pieces:
            if len(piece) > chunk_chars:
                for idx in range(0, len(piece), chunk_chars):
                    fragment = piece[idx : idx + chunk_chars].strip()
                    if fragment:
                        if current:
                            chunks.append(current)
                            current = ""
                        chunks.append(fragment)
                continue

            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= chunk_chars:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = piece

    if current:
        chunks.append(current)

    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    with_overlap: list[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            with_overlap.append(chunk)
            continue
        prefix = chunks[idx - 1][-overlap_chars:].strip()
        if prefix:
            with_overlap.append(f"{prefix} {chunk}".strip())
        else:
            with_overlap.append(chunk)
    return with_overlap


def _snippet_relevance_score(claim_norm: str, claim_tokens: set[str], snippet_text: str) -> float:
    snippet_norm = normalize_text(snippet_text)
    if not snippet_norm:
        return 0.0
    snippet_tokens = tokenize(snippet_norm)
    jaccard = jaccard_similarity(claim_tokens, snippet_tokens)
    token_cover = token_overlap_ratio(claim_tokens, snippet_tokens)
    phrase_boost = 1.0 if claim_norm and claim_norm in snippet_norm else 0.0
    # Weight token coverage higher than pure Jaccard so paraphrased claims
    # are less likely to collapse into fully unverifiable.
    score = (0.45 * jaccard) + (0.45 * token_cover) + (0.10 * phrase_boost)
    return max(0.0, min(1.0, score))


def _retrieve_evidence_snippets(
    claim: str,
    sources: list[SourceDoc],
    *,
    max_snippets: int,
    per_source_cap: int = 2,
) -> list[EvidenceSnippet]:
    claim_norm = normalize_text(claim)
    claim_tokens = tokenize(claim)
    candidates: list[EvidenceSnippet] = []
    counter = 1

    for source in sources:
        chunks = _chunk_source_text(source.text)
        for chunk in chunks:
            score = _snippet_relevance_score(claim_norm, claim_tokens, chunk)
            if score <= 0.0:
                continue
            candidates.append(
                EvidenceSnippet(
                    snippet_id=f"S{counter}",
                    source_label=source.label,
                    source_origin=source.origin,
                    is_primary=source.is_primary,
                    text=chunk,
                    score=round(score, 4),
                )
            )
            counter += 1

    if not candidates:
        return []

    candidates.sort(
        key=lambda item: (item.score, len(item.text)),
        reverse=True,
    )

    selected: list[EvidenceSnippet] = []
    source_counts: dict[str, int] = {}
    for item in candidates:
        count = source_counts.get(item.source_label, 0)
        if count >= max(1, per_source_cap):
            continue
        selected.append(item)
        source_counts[item.source_label] = count + 1
        if len(selected) >= max_snippets:
            break
    if selected:
        return selected

    return candidates[:max_snippets]


def _deterministic_verdict_from_snippets(
    *,
    claim: str,
    claim_norm: str,
    snippets: list[EvidenceSnippet],
    support_threshold: float,
    partial_threshold: float,
) -> dict[str, object]:
    if not snippets:
        return {
            "verdict": "unverifiable",
            "confidence": 0.05,
            "rationale": "No evidence snippets were retrieved for this claim.",
            "reason_code": "no_evidence_retrieved",
            "evidence_ids": [],
        }

    best_support = 0.0
    best_support_id = ""
    best_contradiction = 0.0
    best_contradiction_id = ""

    for snippet in snippets:
        support = float(snippet.score)
        if claim_norm and claim_norm in normalize_text(snippet.text):
            support = max(support, 0.98)
        if support > best_support:
            best_support = support
            best_support_id = snippet.snippet_id

        contradiction = _contradiction_score(claim_norm, normalize_text(snippet.text))
        if contradiction > best_contradiction:
            best_contradiction = contradiction
            best_contradiction_id = snippet.snippet_id

    if best_contradiction >= 0.75 and best_contradiction >= best_support + 0.05:
        confidence = min(0.99, 0.55 + 0.45 * best_contradiction)
        return {
            "verdict": "contradicted",
            "confidence": round(confidence, 4),
            "rationale": "Evidence snippet contains explicit contradiction cues.",
            "reason_code": "deterministic_contradicted",
            "evidence_ids": [best_contradiction_id] if best_contradiction_id else [],
        }

    if best_support >= support_threshold:
        confidence = min(0.99, 0.6 + 0.4 * best_support)
        return {
            "verdict": "supported",
            "confidence": round(confidence, 4),
            "rationale": "Top evidence snippet has strong lexical/semantic overlap.",
            "reason_code": "deterministic_supported",
            "evidence_ids": [best_support_id] if best_support_id else [],
        }

    if best_support >= partial_threshold:
        confidence = min(0.90, 0.45 + 0.35 * best_support)
        return {
            "verdict": "partially_supported",
            "confidence": round(confidence, 4),
            "rationale": "Evidence partially overlaps but does not fully entail the claim.",
            "reason_code": "deterministic_partial",
            "evidence_ids": [best_support_id] if best_support_id else [],
        }

    confidence = min(0.6, max(0.05, best_support))
    return {
        "verdict": "unverifiable",
        "confidence": round(confidence, 4),
        "rationale": "Evidence overlap is below support thresholds.",
        "reason_code": "deterministic_unverifiable",
        "evidence_ids": [best_support_id] if best_support_id else [],
    }


def _build_entailment_prompt(claim: str, snippets: list[EvidenceSnippet]) -> str:
    lines = [
        "You are a strict claim verifier.",
        "Classify claim support using only the provided evidence snippets.",
        "Return JSON only (no markdown) with keys:",
        "verdict, confidence, rationale, evidence.",
        "Allowed verdict values: supported, partially_supported, contradicted, unverifiable.",
        "evidence must be an array of objects with keys: snippet_id, stance, quote.",
        "Allowed stance values: supports, partial, contradicts.",
        "quote must be copied from the snippet and <= 220 characters.",
        "If evidence is mixed/insufficient, use partially_supported or unverifiable.",
        "",
        f"CLAIM: {claim}",
        "",
        "EVIDENCE SNIPPETS:",
    ]
    for snippet in snippets:
        snippet_text = _WHITESPACE_RE.sub(" ", snippet.text).strip()
        if len(snippet_text) > _MAX_SNIPPET_CHARS:
            snippet_text = snippet_text[: _MAX_SNIPPET_CHARS - 3] + "..."
        lines.append(
            f"- {snippet.snippet_id}"
            f" | source={snippet.source_label}"
            f" | primary={str(snippet.is_primary).lower()}"
            f" | relevance={snippet.score:.4f}"
            f"\n  {snippet_text}"
        )

    lines.extend(
        [
            "",
            "JSON response template:",
            (
                '{"verdict":"supported","confidence":0.0,'
                '"rationale":"...",'
                '"evidence":[{"snippet_id":"S1","stance":"supports","quote":"..."}]}'
            ),
        ]
    )
    return "\n".join(lines)


async def _classify_with_llm(
    *,
    claim: str,
    snippets: list[EvidenceSnippet],
    model: ModelProvider,
    validator: ResponseValidator,
    config: Config | None,
) -> dict[str, object] | None:
    prompt = _build_entailment_prompt(claim, snippets)
    request_messages = [{"role": "user", "content": prompt}]
    policy = (
        ModelRetryPolicy.from_execution_config(config.execution)
        if config is not None
        else ModelRetryPolicy()
    )

    async def _invoke_model():
        return await model.complete(request_messages)

    response = await call_with_model_retry(_invoke_model, policy=policy)
    validation = validator.validate_json_response(
        response,
        expected_keys=["verdict", "confidence", "rationale", "evidence"],
    )
    if not validation.valid:
        return None

    payload = validation.parsed
    if not isinstance(payload, dict):
        return None

    verdict = str(payload.get("verdict", "") or "").strip().lower()
    if verdict not in {"supported", "partially_supported", "contradicted", "unverifiable"}:
        return None

    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(payload.get("rationale", "") or "").strip()
    evidence_ids: list[str] = []
    evidence_items = payload.get("evidence", [])
    if isinstance(evidence_items, list):
        seen_ids: set[str] = set()
        for raw_item in evidence_items[:6]:
            if not isinstance(raw_item, dict):
                continue
            snippet_id = str(raw_item.get("snippet_id", "") or "").strip()
            if snippet_id and snippet_id not in seen_ids:
                seen_ids.add(snippet_id)
                evidence_ids.append(snippet_id)

    reason_code = f"llm_{verdict}"
    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "rationale": rationale,
        "reason_code": reason_code,
        "evidence_ids": evidence_ids,
    }


def _snippet_by_id(snippets: list[EvidenceSnippet]) -> dict[str, EvidenceSnippet]:
    return {item.snippet_id: item for item in snippets}


def _select_excerpt(snippet: EvidenceSnippet | None) -> str:
    if snippet is None:
        return ""
    text = _WHITESPACE_RE.sub(" ", snippet.text).strip()
    if len(text) > 240:
        return text[:237] + "..."
    return text


def _apply_primary_source_policy(
    *,
    selected: dict[str, object],
    snippets_by_id: dict[str, EvidenceSnippet],
    require_primary: bool,
) -> dict[str, object]:
    if not require_primary:
        return selected

    evidence_ids = selected.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = []

    has_primary = False
    for evidence_id in evidence_ids:
        snippet = snippets_by_id.get(str(evidence_id or ""))
        if snippet is not None and snippet.is_primary:
            has_primary = True
            break

    if has_primary:
        return selected

    updated = dict(selected)
    verdict = str(updated.get("verdict", "unverifiable") or "unverifiable")
    confidence = float(updated.get("confidence", 0.0) or 0.0)
    rationale = str(updated.get("rationale", "") or "").strip()

    if verdict == "supported":
        updated["verdict"] = "partially_supported"
        updated["confidence"] = round(max(0.05, confidence - 0.2), 4)
        updated["reason_code"] = "non_primary_downgrade"
        suffix = "Downgraded: strongest evidence is non-primary."
        updated["rationale"] = f"{rationale} {suffix}".strip()
        return updated

    if verdict == "partially_supported":
        updated["verdict"] = "unverifiable"
        updated["confidence"] = round(max(0.05, confidence - 0.2), 4)
        updated["reason_code"] = "non_primary_downgrade"
        suffix = "Downgraded: evidence is non-primary only."
        updated["rationale"] = f"{rationale} {suffix}".strip()
        return updated

    return updated


async def _check_one_claim(
    claim: str,
    sources: list[SourceDoc],
    *,
    support_threshold: float,
    partial_threshold: float,
    require_primary: bool,
    max_evidence_snippets: int,
    verifier_model: ModelProvider | None,
    semantic_mode: str,
    config: Config | None,
    validator: ResponseValidator,
) -> FactCheckVerdict:
    claim_norm = normalize_text(claim)
    snippets = _retrieve_evidence_snippets(
        claim,
        sources,
        max_snippets=max_evidence_snippets,
    )
    snippet_index = _snippet_by_id(snippets)

    deterministic = _deterministic_verdict_from_snippets(
        claim=claim,
        claim_norm=claim_norm,
        snippets=snippets,
        support_threshold=support_threshold,
        partial_threshold=partial_threshold,
    )

    selected = dict(deterministic)
    llm_attempted = verifier_model is not None and semantic_mode in {"auto", "llm"}
    llm_failed = False

    if llm_attempted:
        try:
            llm_verdict = await _classify_with_llm(
                claim=claim,
                snippets=snippets,
                model=verifier_model,
                validator=validator,
                config=config,
            )
            if llm_verdict is not None:
                selected = llm_verdict
            else:
                llm_failed = True
        except Exception:
            llm_failed = True
            logger.debug("LLM claim verification failed for claim", exc_info=True)

    if llm_failed and str(selected.get("verdict", "")).lower() in {
        "unverifiable",
        "partially_supported",
    }:
        selected["reason_code"] = "semantic_inconclusive"
        rationale = str(selected.get("rationale", "") or "").strip()
        selected["rationale"] = (
            f"{rationale} Semantic entailment classifier was inconclusive; "
            "falling back to deterministic retrieval score."
        ).strip()

    selected = _apply_primary_source_policy(
        selected=selected,
        snippets_by_id=snippet_index,
        require_primary=require_primary,
    )

    evidence_ids = selected.get("evidence_ids", [])
    if not isinstance(evidence_ids, list):
        evidence_ids = []
    evidence_snippets = [
        snippet_index[item]
        for item in evidence_ids
        if str(item or "") in snippet_index
    ]
    primary_snippet = (
        evidence_snippets[0]
        if evidence_snippets
        else (snippets[0] if snippets else None)
    )

    evidence_refs: list[str] = []
    seen_refs: set[str] = set()
    for snippet in evidence_snippets:
        ref = str(snippet.source_origin or snippet.source_label).strip()
        if ref and ref not in seen_refs:
            seen_refs.add(ref)
            evidence_refs.append(ref)
    if not evidence_refs and primary_snippet is not None:
        fallback_ref = str(
            primary_snippet.source_origin or primary_snippet.source_label,
        ).strip()
        if fallback_ref:
            evidence_refs.append(fallback_ref)

    verdict = str(selected.get("verdict", "unverifiable") or "unverifiable").strip().lower()
    if verdict not in {"supported", "partially_supported", "contradicted", "unverifiable"}:
        verdict = "unverifiable"

    confidence = float(selected.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    reason_code = str(selected.get("reason_code", "") or "").strip().lower()
    rationale = str(selected.get("rationale", "") or "").strip()

    return FactCheckVerdict(
        claim=claim,
        verdict=verdict,
        confidence=round(confidence, 4),
        rationale=rationale or "No rationale provided.",
        source=(primary_snippet.source_label if primary_snippet is not None else ""),
        source_excerpt=_select_excerpt(primary_snippet),
        reason_code=reason_code,
        evidence_refs=evidence_refs,
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
        refs = ", ".join(verdict.evidence_refs) if verdict.evidence_refs else "N/A"
        lines.extend(
            [
                f"### {idx}. {verdict.claim}",
                f"- Verdict: **{verdict.verdict}**",
                f"- Confidence: {verdict.confidence:.4f}",
                f"- Rationale: {verdict.rationale}",
                f"- Reason Code: {verdict.reason_code or 'N/A'}",
                f"- Source: {verdict.source or 'N/A'}",
                f"- Evidence Refs: {refs}",
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
                "reason_code",
                "source",
                "evidence_refs",
                "source_excerpt",
            ],
        )
        writer.writeheader()
        for verdict in verdicts:
            row = verdict.to_dict()
            refs = row.get("evidence_refs", [])
            row["evidence_refs"] = ";".join(
                str(item).strip() for item in refs if str(item).strip()
            )
            writer.writerow(row)


def _write_optional_reports(
    *,
    tool: Tool,
    ctx: ToolContext,
    markdown_path: str,
    csv_path: str,
    verdicts: list[FactCheckVerdict],
    counts: dict[str, int],
) -> list[str]:
    workspace = ctx.workspace
    if workspace is None:
        return []

    changed: list[str] = []

    if markdown_path:
        md = tool._resolve_path(markdown_path, workspace)
        md.parent.mkdir(parents=True, exist_ok=True)
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(str(md), subtask_id=ctx.subtask_id)
        md.write_text(_render_markdown(verdicts, counts), encoding="utf-8")
        changed.append(str(md.relative_to(workspace)))

    if csv_path:
        csv_out = tool._resolve_path(csv_path, workspace)
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        if ctx.changelog is not None:
            ctx.changelog.record_before_write(str(csv_out), subtask_id=ctx.subtask_id)
        _write_csv(csv_out, verdicts)
        changed.append(str(csv_out.relative_to(workspace)))

    return changed
