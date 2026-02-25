"""Humanized writing tool for deterministic draft quality improvement."""

from __future__ import annotations

import json
from typing import Any

from loom.tools.registry import Tool, ToolContext, ToolResult
from loom.writing.metrics import analyze_text, sub_score_delta
from loom.writing.reporting import render_markdown_report
from loom.writing.rewrite_plan import build_rewrite_actions
from loom.writing.style_profiles import normalize_mode

_OPERATIONS = {"analyze", "plan_rewrite", "evaluate", "compare"}
_MODES = {"creative_copy", "blog_post", "email", "report", "social_post", "custom"}
_DEFAULT_MAX_CONTENT_CHARS = 80_000


class HumanizeWritingTool(Tool):
    """Analyze and improve human-like writing quality."""

    @property
    def name(self) -> str:
        return "humanize_writing"

    @property
    def description(self) -> str:
        return (
            "Analyze and improve writing naturalness with deterministic metrics. "
            "Supports analyze, plan_rewrite, evaluate, and compare operations."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": sorted(_OPERATIONS),
                    "description": "analyze | plan_rewrite | evaluate | compare",
                },
                "content": {
                    "type": "string",
                    "description": "Inline content to analyze.",
                },
                "path": {
                    "type": "string",
                    "description": "Workspace file path for content input.",
                },
                "baseline_content": {
                    "type": "string",
                    "description": "Baseline text used by compare operation.",
                },
                "baseline_path": {
                    "type": "string",
                    "description": "Baseline file path used by compare operation.",
                },
                "mode": {
                    "type": "string",
                    "enum": sorted(_MODES),
                    "description": "Writing mode controls sub-score weighting.",
                },
                "audience": {
                    "type": "string",
                    "description": "Optional audience descriptor.",
                },
                "voice_profile": {
                    "type": "object",
                    "description": (
                        "Optional style controls (formality, sentence_rhythm, "
                        "lexical_complexity, warmth, assertiveness, humor)."
                    ),
                },
                "constraints": {
                    "type": "object",
                    "description": (
                        "Rewrite constraints, e.g. preserve_terms, banned_phrases, "
                        "max_sentence_length, reading_level."
                    ),
                },
                "target_score": {
                    "type": "number",
                    "description": "Optional threshold (0-100) used by evaluate/compare.",
                },
                "max_recommendations": {
                    "type": "integer",
                    "description": "Max rewrite actions returned (default 8, max 20).",
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional markdown report output path.",
                },
                "output_json_path": {
                    "type": "string",
                    "description": "Optional JSON report output path.",
                },
            },
        }

    @property
    def timeout_seconds(self) -> int:
        return 45

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        operation = str(args.get("operation", "analyze")).strip().lower() or "analyze"
        if operation not in _OPERATIONS:
            return ToolResult.fail("operation must be analyze/plan_rewrite/evaluate/compare")

        mode = normalize_mode(str(args.get("mode", "custom")))
        audience = str(args.get("audience", "")).strip()
        voice_profile = args.get("voice_profile")
        if not isinstance(voice_profile, dict):
            voice_profile = {}
        constraints = args.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}
        target_score = _normalized_target_score(args.get("target_score"))
        max_recommendations = _clamp_int(args.get("max_recommendations"), default=8, lo=1, hi=20)

        content, load_error = self._load_content(
            args=args,
            ctx=ctx,
            content_key="content",
            path_key="path",
            required=True,
        )
        if load_error:
            return ToolResult.fail(load_error)
        if content is None:
            return ToolResult.fail("No content provided")
        if len(content) > _DEFAULT_MAX_CONTENT_CHARS:
            return ToolResult.fail(
                f"Content too large (max {_DEFAULT_MAX_CONTENT_CHARS} characters)"
            )

        report = analyze_text(
            content,
            mode=mode,
            audience=audience,
            voice_profile=voice_profile,
            target_score=target_score,
        )
        recommended_edits, constraints_applied = build_rewrite_actions(
            issues=report.issues,
            constraints=constraints,
            max_actions=max_recommendations,
        )
        report.recommended_edits = recommended_edits
        report.constraints_applied = constraints_applied

        baseline_report = None
        score_delta = None
        delta = None
        improved = None
        if operation == "compare":
            baseline_content, baseline_error = self._load_content(
                args=args,
                ctx=ctx,
                content_key="baseline_content",
                path_key="baseline_path",
                required=True,
            )
            if baseline_error:
                return ToolResult.fail(
                    "compare operation requires baseline_content or baseline_path: "
                    + baseline_error
                )
            if baseline_content is None:
                return ToolResult.fail(
                    "compare operation requires baseline_content or baseline_path",
                )
            if len(baseline_content) > _DEFAULT_MAX_CONTENT_CHARS:
                return ToolResult.fail(
                    "Baseline content too large "
                    f"(max {_DEFAULT_MAX_CONTENT_CHARS} characters)",
                )
            baseline_report = analyze_text(
                baseline_content,
                mode=mode,
                audience=audience,
                voice_profile=voice_profile,
                target_score=target_score,
            )
            score_delta = round(
                report.humanization_score - baseline_report.humanization_score,
                2,
            )
            delta = sub_score_delta(report, baseline_report)
            improved = score_delta > 0.0

        payload = {
            "operation": operation,
            "mode": mode,
            "audience": audience,
            "target_score": target_score,
            "report": report.to_dict(),
            "recommended_edits": list(report.recommended_edits),
            "constraints_applied": list(report.constraints_applied),
        }
        if baseline_report is not None:
            payload["baseline_report"] = baseline_report.to_dict()
            payload["score_delta"] = score_delta
            payload["sub_score_delta"] = delta or {}
            payload["improved"] = improved

        files_changed: list[str] = []
        output_path_raw = str(args.get("output_path", "")).strip()
        output_json_path_raw = str(args.get("output_json_path", "")).strip()
        if output_path_raw or output_json_path_raw:
            if ctx.workspace is None:
                return ToolResult.fail("No workspace set for output artifact write")
            try:
                files_changed = self._write_outputs(
                    operation=operation,
                    report=report,
                    payload=payload,
                    baseline_report=baseline_report,
                    score_delta=score_delta,
                    sub_score_delta_map=delta,
                    output_path_raw=output_path_raw,
                    output_json_path_raw=output_json_path_raw,
                    ctx=ctx,
                )
            except Exception as e:
                return ToolResult.fail(str(e))

        if operation == "compare":
            output_lines = [
                (
                    "Humanization compare complete: "
                    f"baseline={baseline_report.humanization_score:.2f}, "
                    f"current={report.humanization_score:.2f}, "
                    f"delta={score_delta:+.2f}."
                ),
                f"Improved: {bool(improved)}",
            ]
        else:
            output_lines = [
                (
                    f"Humanization {operation} complete: "
                    f"score={report.humanization_score:.2f}/100."
                ),
            ]
            if target_score is not None:
                output_lines.append(
                    f"Target={target_score:.2f}, pass={bool(report.passes_target)}."
                )
            output_lines.append(f"Issues={len(report.issues)}.")

        if report.issues:
            output_lines.append(f"Top issue: {report.issues[0].title}.")
        if files_changed:
            output_lines.append("Artifacts: " + ", ".join(files_changed))

        return ToolResult.ok(
            "\n".join(output_lines),
            data=payload,
            files_changed=files_changed,
        )

    def _load_content(
        self,
        *,
        args: dict,
        ctx: ToolContext,
        content_key: str,
        path_key: str,
        required: bool,
    ) -> tuple[str | None, str | None]:
        inline = args.get(content_key)
        if inline is not None:
            text = str(inline).strip()
            if text:
                return text, None

        path_raw = str(args.get(path_key, "")).strip()
        if not path_raw:
            if required:
                return None, f"missing {content_key} or {path_key}"
            return None, None
        if ctx.workspace is None:
            return None, "No workspace set"

        try:
            path = self._resolve_read_path(path_raw, ctx.workspace, ctx.read_roots)
        except Exception as e:
            return None, str(e)
        if not path.exists() or not path.is_file():
            return None, f"input path not found: {path_raw}"
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            return None, str(e)
        if not text and required:
            return None, f"input file is empty: {path_raw}"
        return text, None

    def _write_outputs(
        self,
        *,
        operation: str,
        report,
        payload: dict[str, Any],
        baseline_report,
        score_delta: float | None,
        sub_score_delta_map: dict[str, float] | None,
        output_path_raw: str,
        output_json_path_raw: str,
        ctx: ToolContext,
    ) -> list[str]:
        assert ctx.workspace is not None
        files_changed: list[str] = []

        if output_path_raw:
            md_path = self._resolve_path(output_path_raw, ctx.workspace)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(md_path), subtask_id=ctx.subtask_id)
            md_path.write_text(
                render_markdown_report(
                    operation=operation,
                    report=report,
                    baseline_report=baseline_report,
                    score_delta=score_delta,
                    sub_score_delta=sub_score_delta_map,
                ),
                encoding="utf-8",
            )
            files_changed.append(str(md_path.relative_to(ctx.workspace)))

        if output_json_path_raw:
            json_path = self._resolve_path(output_json_path_raw, ctx.workspace)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            if ctx.changelog is not None:
                ctx.changelog.record_before_write(str(json_path), subtask_id=ctx.subtask_id)
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            files_changed.append(str(json_path.relative_to(ctx.workspace)))

        return files_changed


def _normalized_target_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(100.0, parsed))


def _clamp_int(value: object, *, default: int, lo: int, hi: int) -> int:
    try:
        parsed = default if value is None else int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(lo, min(hi, parsed))
