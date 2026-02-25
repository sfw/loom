"""Report rendering for humanized writing analyses."""

from __future__ import annotations

from loom.writing.models import HumanizationReport


def render_markdown_report(
    *,
    operation: str,
    report: HumanizationReport,
    baseline_report: HumanizationReport | None = None,
    score_delta: float | None = None,
    sub_score_delta: dict[str, float] | None = None,
) -> str:
    """Render a markdown summary for analyze/evaluate/plan/compare outputs."""
    lines: list[str] = [
        "# Humanized Writing Report",
        "",
        f"- Operation: `{operation}`",
        f"- Mode: `{report.mode}`",
        f"- Humanization score: `{report.humanization_score:.2f}/100`",
    ]
    if report.target_score is not None:
        lines.append(f"- Target score: `{report.target_score:.2f}`")
        lines.append(f"- Pass target: `{bool(report.passes_target)}`")
    if report.audience:
        lines.append(f"- Audience: `{report.audience}`")

    if baseline_report is not None:
        lines.append(
            f"- Baseline score: `{baseline_report.humanization_score:.2f}/100`"
        )
    if score_delta is not None:
        lines.append(f"- Score delta: `{score_delta:+.2f}`")

    lines.extend(
        [
            "",
            "## Sub-scores",
            "",
        ]
    )
    for key in sorted(report.sub_scores.keys()):
        value = report.sub_scores[key]
        if sub_score_delta and key in sub_score_delta:
            lines.append(f"- {key}: `{value:.2f}` (`{sub_score_delta[key]:+.2f}`)")
        else:
            lines.append(f"- {key}: `{value:.2f}`")

    metrics = report.metrics.to_dict()
    lines.extend(
        [
            "",
            "## Core Metrics",
            "",
            f"- Words: `{metrics['word_count']}`",
            f"- Sentences: `{metrics['sentence_count']}`",
            f"- Paragraphs: `{metrics['paragraph_count']}`",
            f"- Avg sentence length: `{metrics['avg_sentence_length']}`",
            f"- Sentence length stddev: `{metrics['sentence_length_stddev']}`",
            f"- Repeated n-gram ratio: `{metrics['repeated_ngram_ratio']}`",
            "- Near-duplicate sentence ratio: "
            f"`{metrics['near_duplicate_sentence_ratio']}`",
            f"- Lexical diversity: `{metrics['lexical_diversity']}`",
            f"- Detail density: `{metrics['detail_density']}`",
            f"- Passive voice ratio: `{metrics['passive_voice_ratio']}`",
        ]
    )

    lines.extend(["", "## Issues", ""])
    if report.issues:
        for issue in report.issues:
            evidence_text = f" ({issue.evidence})" if issue.evidence else ""
            lines.append(
                f"- [S{issue.severity}] {issue.title}: {issue.detail}{evidence_text}"
            )
    else:
        lines.append("- No major issues detected.")

    lines.extend(["", "## Recommended Edits", ""])
    if report.recommended_edits:
        for edit in report.recommended_edits:
            lines.append(f"- {edit}")
    else:
        lines.append("- No rewrite edits suggested.")

    if report.constraints_applied:
        lines.extend(
            [
                "",
                "## Applied Constraints",
                "",
                "- " + ", ".join(report.constraints_applied),
            ]
        )

    return "\n".join(lines).rstrip() + "\n"
