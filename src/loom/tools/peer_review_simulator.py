"""Peer review simulator for research drafts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult

_SENTENCE_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_CITATION_RE = re.compile(r"(\[[^\]]+\]|\([A-Za-z][^\)]*\d{4}[^\)]*\)|https?://\S+)")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


@dataclass(frozen=True)
class CriterionScore:
    name: str
    score: float
    rationale: str


_DEFAULT_RUBRICS: dict[str, list[dict[str, Any]]] = {
    "general": [
        {"name": "clarity", "weight": 0.25},
        {"name": "structure", "weight": 0.2},
        {"name": "evidence", "weight": 0.25},
        {"name": "limitations", "weight": 0.15},
        {"name": "actionability", "weight": 0.15},
    ],
    "historical_accuracy": [
        {"name": "chronology", "weight": 0.25},
        {"name": "evidence", "weight": 0.25},
        {"name": "source_quality", "weight": 0.2},
        {"name": "context", "weight": 0.15},
        {"name": "limitations", "weight": 0.15},
    ],
    "methodology": [
        {"name": "method_definition", "weight": 0.3},
        {"name": "reproducibility", "weight": 0.25},
        {"name": "evidence", "weight": 0.2},
        {"name": "limitations", "weight": 0.15},
        {"name": "clarity", "weight": 0.1},
    ],
    "argument_quality": [
        {"name": "thesis", "weight": 0.3},
        {"name": "evidence", "weight": 0.25},
        {"name": "counterarguments", "weight": 0.2},
        {"name": "coherence", "weight": 0.15},
        {"name": "clarity", "weight": 0.1},
    ],
    "citation_quality": [
        {"name": "citation_density", "weight": 0.3},
        {"name": "source_quality", "weight": 0.25},
        {"name": "traceability", "weight": 0.2},
        {"name": "consistency", "weight": 0.15},
        {"name": "recency_markers", "weight": 0.1},
    ],
}


class PeerReviewSimulatorTool(Tool):
    """Create structured, rubric-driven review outputs for drafts."""

    @property
    def name(self) -> str:
        return "peer_review_simulator"

    @property
    def description(self) -> str:
        return (
            "Simulate peer review using deterministic rubric scoring. Returns "
            "strengths, issues, and revision actions."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Document path to review."},
                "content": {
                    "type": "string",
                    "description": "Inline document content to review.",
                },
                "review_type": {
                    "type": "string",
                    "enum": [
                        "general",
                        "historical_accuracy",
                        "methodology",
                        "argument_quality",
                        "citation_quality",
                    ],
                },
                "rubric": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Optional custom rubric entries {name, weight}.",
                },
                "strictness": {
                    "type": "string",
                    "enum": ["light", "standard", "strict"],
                },
                "num_reviewers": {
                    "type": "integer",
                    "description": "Synthetic reviewer count (1-3).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown output path.",
                },
                "output_json_path": {
                    "type": "string",
                    "description": "Optional json output path.",
                },
            },
        }

    @property
    def timeout_seconds(self) -> int:
        return 30

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        content = str(args.get("content", "")).strip()
        path_text = str(args.get("path", "")).strip()

        if not content and path_text:
            if ctx.workspace is None:
                return ToolResult.fail("No workspace set")
            path = self._resolve_read_path(path_text, ctx.workspace, ctx.read_roots)
            if not path.exists() or not path.is_file():
                return ToolResult.fail("Review path not found")
            content = path.read_text(encoding="utf-8", errors="replace")

        if not content.strip():
            return ToolResult.fail("No review content provided")

        review_type = str(args.get("review_type", "general")).strip().lower() or "general"
        if review_type not in _DEFAULT_RUBRICS:
            return ToolResult.fail("Unsupported review_type")

        strictness = str(args.get("strictness", "standard")).strip().lower() or "standard"
        if strictness not in {"light", "standard", "strict"}:
            return ToolResult.fail("strictness must be light, standard, or strict")

        num_reviewers = _clamp_int(args.get("num_reviewers"), default=1, low=1, high=3)
        rubric = _normalize_rubric(args.get("rubric"), fallback=_DEFAULT_RUBRICS[review_type])

        metrics = _compute_metrics(content)
        base_scores = _score_rubric(rubric, metrics)
        reviewer_scores = _expand_reviewer_scores(base_scores, num_reviewers)
        aggregate_scores = _aggregate_scores(reviewer_scores)

        weighted_score = _weighted_score(aggregate_scores, rubric)
        strengths, major_issues, minor_issues, actions = _classify_feedback(
            aggregate_scores,
            strictness,
        )

        summary = (
            f"Peer review score: {weighted_score:.2f}/5.00 "
            f"(major={len(major_issues)}, minor={len(minor_issues)})"
        )

        payload = {
            "review_type": review_type,
            "strictness": strictness,
            "num_reviewers": num_reviewers,
            "weighted_score": weighted_score,
            "metrics": metrics,
            "scores": [
                {
                    "name": score.name,
                    "score": score.score,
                    "rationale": score.rationale,
                }
                for score in aggregate_scores
            ],
            "reviewers": reviewer_scores,
            "strengths": strengths,
            "major_issues": major_issues,
            "minor_issues": minor_issues,
            "revision_actions": actions,
        }

        files_changed: list[str] = []
        if ctx.workspace is not None:
            md_path = self._resolve_path(
                str(args.get("output_path", "peer-review.md")).strip() or "peer-review.md",
                ctx.workspace,
            )
            json_path = self._resolve_path(
                str(args.get("output_json_path", "peer-review.json")).strip() or "peer-review.json",
                ctx.workspace,
            )
            md_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.parent.mkdir(parents=True, exist_ok=True)

            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(md_path), subtask_id=ctx.subtask_id)
                ctx.changelog.record_before_write(str(json_path), subtask_id=ctx.subtask_id)

            md_path.write_text(_render_markdown(payload), encoding="utf-8")
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            files_changed.extend(
                [
                    str(md_path.relative_to(ctx.workspace)),
                    str(json_path.relative_to(ctx.workspace)),
                ]
            )
            summary += "\nArtifacts: " + ", ".join(files_changed)

        return ToolResult.ok(summary, files_changed=files_changed, data=payload)


def _clamp_int(value: object, *, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _normalize_rubric(raw: object, *, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        return fallback
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip().lower()
        if not name:
            continue
        weight = item.get("weight", 1.0)
        try:
            weight_f = max(0.0, float(weight))
        except (TypeError, ValueError):
            weight_f = 1.0
        out.append({"name": name, "weight": weight_f})
    return out or fallback


def _compute_metrics(content: str) -> dict[str, Any]:
    words = _WORD_RE.findall(content)
    word_count = len(words)
    sentence_parts = [part.strip() for part in _SENTENCE_RE.split(content) if part.strip()]
    sentence_count = max(1, len(sentence_parts))
    avg_sentence_len = word_count / sentence_count if sentence_count else 0.0

    headings = len(_HEADING_RE.findall(content))
    citations = len(_CITATION_RE.findall(content))
    year_markers = len(_YEAR_RE.findall(content))

    lowered = content.lower()
    keyword_flags = {
        "method": any(k in lowered for k in ["method", "approach", "dataset", "sample"]),
        "limitations": any(k in lowered for k in ["limitation", "uncertain", "caveat", "bias"]),
        "counterarguments": any(
            k in lowered for k in ["counterargument", "alternative", "however"]
        ),
        "thesis": any(k in lowered for k in ["thesis", "we argue", "we conclude", "this report"]),
        "timeline": any(k in lowered for k in ["timeline", "chronology", "sequence"]),
    }

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_len,
        "heading_count": headings,
        "citation_count": citations,
        "year_marker_count": year_markers,
        "keyword_flags": keyword_flags,
    }


def _score_rubric(rubric: list[dict[str, Any]], metrics: dict[str, Any]) -> list[CriterionScore]:
    scores: list[CriterionScore] = []
    for item in rubric:
        name = str(item["name"])
        score, rationale = _score_one(name, metrics)
        scores.append(CriterionScore(name=name, score=score, rationale=rationale))
    return scores


def _score_one(name: str, metrics: dict[str, Any]) -> tuple[float, str]:
    wc = float(metrics["word_count"])
    avg_len = float(metrics["avg_sentence_length"])
    headings = int(metrics["heading_count"])
    citations = int(metrics["citation_count"])
    years = int(metrics["year_marker_count"])
    flags = metrics["keyword_flags"]

    if name in {"clarity", "coherence"}:
        base = 5.0 - max(0.0, (avg_len - 22.0) / 9.0)
        if wc < 200:
            base -= 0.5
        return (_cap(base), f"Avg sentence length={avg_len:.1f}")

    if name in {"structure", "traceability"}:
        base = 2.0 + min(3.0, headings * 0.9)
        return (_cap(base), f"Heading count={headings}")

    if name in {"evidence", "citation_density", "source_quality"}:
        density = citations / max(1.0, wc / 350.0)
        base = min(5.0, 1.8 + density)
        return (_cap(base), f"Citation markers={citations}")

    if name in {"limitations", "context", "counterarguments"}:
        signal = flags.get("limitations") or flags.get("counterarguments")
        return (4.2 if signal else 2.3, "Keyword signal present" if signal else "Missing signal")

    if name in {"method_definition", "reproducibility"}:
        signal = flags.get("method")
        base = 4.3 if signal else 2.1
        if signal and citations > 0:
            base += 0.2
        return (_cap(base), "Method keywords detected" if signal else "Method details unclear")

    if name in {"chronology", "recency_markers"}:
        base = min(5.0, 2.0 + years / 3.0)
        if flags.get("timeline"):
            base += 0.4
        return (_cap(base), f"Year markers={years}")

    if name in {"actionability", "thesis"}:
        signal = flags.get("thesis")
        rationale = "Thesis/action framing present" if signal else "Thesis unclear"
        return (4.0 if signal else 2.4, rationale)

    return (3.0, "No specialized heuristic; neutral score")


def _cap(value: float) -> float:
    return max(1.0, min(5.0, value))


def _expand_reviewer_scores(
    base_scores: list[CriterionScore],
    num_reviewers: int,
) -> list[dict[str, Any]]:
    reviewers: list[dict[str, Any]] = []
    for i in range(num_reviewers):
        reviewer_name = f"reviewer_{i + 1}"
        scored: list[dict[str, Any]] = []
        for item in base_scores:
            delta = _reviewer_offset(reviewer_name, item.name)
            scored.append(
                {
                    "name": item.name,
                    "score": _cap(item.score + delta),
                    "rationale": item.rationale,
                }
            )
        reviewers.append({"reviewer": reviewer_name, "scores": scored})
    return reviewers


def _reviewer_offset(reviewer: str, criterion: str) -> float:
    digest = hashlib.sha1(f"{reviewer}:{criterion}".encode()).hexdigest()
    bucket = int(digest[:2], 16)
    return ((bucket / 255.0) - 0.5) * 0.6


def _aggregate_scores(reviewers: list[dict[str, Any]]) -> list[CriterionScore]:
    by_name: dict[str, list[float]] = {}
    rationale_by_name: dict[str, str] = {}
    for reviewer in reviewers:
        for row in reviewer.get("scores", []):
            name = str(row.get("name", "")).strip()
            if not name:
                continue
            score = float(row.get("score", 0.0))
            by_name.setdefault(name, []).append(score)
            rationale_by_name[name] = str(row.get("rationale", "")).strip()

    out: list[CriterionScore] = []
    for name in sorted(by_name):
        values = by_name[name]
        mean_score = sum(values) / max(1, len(values))
        out.append(
            CriterionScore(
                name=name,
                score=round(mean_score, 3),
                rationale=rationale_by_name.get(name, ""),
            )
        )
    return out


def _weighted_score(scores: list[CriterionScore], rubric: list[dict[str, Any]]) -> float:
    weights = {str(item["name"]): float(item.get("weight", 1.0)) for item in rubric}
    numerator = 0.0
    denominator = 0.0
    for score in scores:
        weight = max(0.0, weights.get(score.name, 1.0))
        numerator += score.score * weight
        denominator += weight
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def _classify_feedback(
    scores: list[CriterionScore],
    strictness: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    thresholds = {
        "light": {"major": 2.4, "minor": 3.2},
        "standard": {"major": 3.0, "minor": 3.6},
        "strict": {"major": 3.4, "minor": 4.0},
    }[strictness]

    strengths: list[str] = []
    major: list[str] = []
    minor: list[str] = []
    actions: list[str] = []

    for score in scores:
        if score.score >= 4.2:
            strengths.append(f"{score.name}: strong ({score.score:.2f})")
        elif score.score < thresholds["major"]:
            major.append(f"{score.name}: weak ({score.score:.2f})")
            actions.append(f"Improve {score.name}: {score.rationale}")
        elif score.score < thresholds["minor"]:
            minor.append(f"{score.name}: moderate ({score.score:.2f})")
            actions.append(f"Refine {score.name}: {score.rationale}")

    if not actions:
        actions.append("No urgent revisions detected; proceed to final polish.")

    return strengths, major, minor, actions


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Peer Review Simulation",
        "",
        f"- Review Type: {payload['review_type']}",
        f"- Strictness: {payload['strictness']}",
        f"- Synthetic Reviewers: {payload['num_reviewers']}",
        f"- Weighted Score: {payload['weighted_score']:.2f}/5.00",
        "",
        "## Criteria",
        "",
        "| Criterion | Score | Rationale |",
        "|---|---:|---|",
    ]

    for item in payload.get("scores", []):
        name = str(item.get("name", ""))
        score = float(item.get("score", 0.0))
        rationale = str(item.get("rationale", "")).replace("|", "\\|")
        lines.append(f"| {name} | {score:.2f} | {rationale} |")

    lines.extend(["", "## Strengths", ""])
    strengths = payload.get("strengths", [])
    if strengths:
        for row in strengths:
            lines.append(f"- {row}")
    else:
        lines.append("- None identified")

    lines.extend(["", "## Major Issues", ""])
    major = payload.get("major_issues", [])
    if major:
        for row in major:
            lines.append(f"- {row}")
    else:
        lines.append("- None")

    lines.extend(["", "## Minor Issues", ""])
    minor = payload.get("minor_issues", [])
    if minor:
        for row in minor:
            lines.append(f"- {row}")
    else:
        lines.append("- None")

    lines.extend(["", "## Revision Actions", ""])
    for action in payload.get("revision_actions", []):
        lines.append(f"- {action}")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(payload.get("metrics", {}), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
