"""Tier-2 LLM verification flow and parsing/repair handling."""

from __future__ import annotations

import contextvars
import csv
import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from loom.config import CompactorLimitsConfig, VerificationConfig, VerifierLimitsConfig
from loom.engine.semantic_compactor import SemanticCompactor
from loom.events.bus import EventBus
from loom.models.request_diagnostics import (
    collect_request_diagnostics,
    collect_response_diagnostics,
)
from loom.models.retry import (
    ModelRetryPolicy,
    build_model_retry_event_payload,
    call_with_model_retry,
)
from loom.models.router import ModelRouter, ResponseValidator
from loom.prompts.assembler import PromptAssembler
from loom.state.task_state import Subtask
from loom.utils.tokens import estimate_tokens

from . import events as verification_events
from . import parsing as verification_parsing
from . import prompting as verification_prompting
from .types import (
    _VALID_OUTCOMES,
    _VALID_SEVERITY_CLASSES,
    VerificationResult,
)

logger = logging.getLogger(__name__)

_COMPACTOR_EVENT_CONTEXT: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "verification_compactor_event_context", default=None
)
_HTTP_STATUS_PATTERN = re.compile(
    r"\b(?:http|status(?:\s*code)?)\s*[:=]?\s*([1-5]\d{2})\b",
    re.IGNORECASE,
)


def _extract_http_status(error_text: str) -> int | None:
    """Extract an HTTP status code from free-form error text when present."""
    match = _HTTP_STATUS_PATTERN.search(error_text or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


class LLMVerifier:
    """Tier 2: Independent LLM verification.

    Uses a DIFFERENT model instance with NO prior conversation history.
    Assesses whether the subtask output meets acceptance criteria.
    """

    def __init__(
        self,
        model_router: ModelRouter,
        prompt_assembler: PromptAssembler,
        validator: ResponseValidator,
        event_bus: EventBus | None = None,
        verification_config: VerificationConfig | None = None,
        limits: VerifierLimitsConfig | None = None,
        compactor_limits: CompactorLimitsConfig | None = None,
        evidence_context_text_max_chars: int = 4000,
    ):
        self._router = model_router
        self._prompts = prompt_assembler
        self._validator = validator
        resolved_limits = limits or VerifierLimitsConfig()
        self._max_tool_args_chars = int(
            getattr(resolved_limits, "max_tool_args_chars", self._MAX_TOOL_ARGS_CHARS),
        )
        self._max_tool_status_chars = int(
            getattr(
                resolved_limits,
                "max_tool_status_chars",
                self._MAX_TOOL_STATUS_CHARS,
            ),
        )
        self._max_tool_calls_tokens = int(
            getattr(
                resolved_limits,
                "max_tool_calls_tokens",
                self._MAX_TOOL_CALLS_TOKENS,
            ),
        )
        self._max_verifier_prompt_tokens = int(
            getattr(
                resolved_limits,
                "max_verifier_prompt_tokens",
                self._MAX_VERIFIER_PROMPT_TOKENS,
            ),
        )
        self._max_result_summary_chars = int(
            getattr(
                resolved_limits,
                "max_result_summary_chars",
                self._MAX_RESULT_SUMMARY_CHARS,
            ),
        )
        self._compact_result_summary_chars = int(
            getattr(
                resolved_limits,
                "compact_result_summary_chars",
                self._COMPACT_RESULT_SUMMARY_CHARS,
            ),
        )
        self._max_evidence_section_chars = int(
            getattr(
                resolved_limits,
                "max_evidence_section_chars",
                self._MAX_EVIDENCE_SECTION_CHARS,
            ),
        )
        self._max_evidence_section_compact_chars = int(
            getattr(
                resolved_limits,
                "max_evidence_section_compact_chars",
                self._MAX_EVIDENCE_SECTION_COMPACT_CHARS,
            ),
        )
        self._max_artifact_section_chars = int(
            getattr(
                resolved_limits,
                "max_artifact_section_chars",
                self._MAX_ARTIFACT_SECTION_CHARS,
            ),
        )
        self._max_artifact_section_compact_chars = int(
            getattr(
                resolved_limits,
                "max_artifact_section_compact_chars",
                self._MAX_ARTIFACT_SECTION_COMPACT_CHARS,
            ),
        )
        self._max_tool_output_excerpt_chars = int(
            getattr(
                resolved_limits,
                "max_tool_output_excerpt_chars",
                self._MAX_TOOL_OUTPUT_EXCERPT_CHARS,
            ),
        )
        self._max_artifact_file_excerpt_chars = int(
            getattr(
                resolved_limits,
                "max_artifact_file_excerpt_chars",
                self._MAX_ARTIFACT_FILE_EXCERPT_CHARS,
            ),
        )
        self._evidence_context_text_max_chars = max(
            120,
            int(evidence_context_text_max_chars or 0),
        )
        comp_limits = compactor_limits or CompactorLimitsConfig()
        self._compactor = SemanticCompactor(
            model_router,
            model_event_hook=self._emit_compactor_model_event,
            role="compactor",
            tier=1,
            allow_role_fallback=True,
            max_chunk_chars=int(
                getattr(comp_limits, "max_chunk_chars", SemanticCompactor._MAX_CHUNK_CHARS),
            ),
            max_chunks_per_round=int(
                getattr(
                    comp_limits,
                    "max_chunks_per_round",
                    SemanticCompactor._MAX_CHUNKS_PER_ROUND,
                ),
            ),
            max_reduction_rounds=int(
                getattr(
                    comp_limits,
                    "max_reduction_rounds",
                    SemanticCompactor._MAX_REDUCTION_ROUNDS,
                ),
            ),
            min_compact_target_chars=int(
                getattr(
                    comp_limits,
                    "min_compact_target_chars",
                    SemanticCompactor._MIN_COMPACT_TARGET_CHARS,
                ),
            ),
            response_tokens_floor=int(
                getattr(
                    comp_limits,
                    "response_tokens_floor",
                    SemanticCompactor._RESPONSE_TOKENS_FLOOR,
                ),
            ),
            response_tokens_ratio=float(
                getattr(
                    comp_limits,
                    "response_tokens_ratio",
                    SemanticCompactor._RESPONSE_TOKENS_RATIO,
                ),
            ),
            response_tokens_buffer=int(
                getattr(
                    comp_limits,
                    "response_tokens_buffer",
                    SemanticCompactor._RESPONSE_TOKENS_BUFFER,
                ),
            ),
            json_headroom_chars_floor=int(
                getattr(
                    comp_limits,
                    "json_headroom_chars_floor",
                    SemanticCompactor._JSON_HEADROOM_CHARS_FLOOR,
                ),
            ),
            json_headroom_chars_ratio=float(
                getattr(
                    comp_limits,
                    "json_headroom_chars_ratio",
                    SemanticCompactor._JSON_HEADROOM_CHARS_RATIO,
                ),
            ),
            json_headroom_chars_cap=int(
                getattr(
                    comp_limits,
                    "json_headroom_chars_cap",
                    SemanticCompactor._JSON_HEADROOM_CHARS_CAP,
                ),
            ),
            chars_per_token_estimate=float(
                getattr(
                    comp_limits,
                    "chars_per_token_estimate",
                    SemanticCompactor._CHARS_PER_TOKEN_ESTIMATE,
                ),
            ),
            token_headroom=int(
                getattr(
                    comp_limits,
                    "token_headroom",
                    SemanticCompactor._TOKEN_HEADROOM,
                ),
            ),
            target_chars_ratio=float(
                getattr(
                    comp_limits,
                    "target_chars_ratio",
                    SemanticCompactor._TARGET_CHARS_RATIO,
                ),
            ),
        )
        self._event_bus = event_bus
        self._config = verification_config or VerificationConfig()

    _MAX_TOOL_ARGS_CHARS = 240
    _MAX_TOOL_STATUS_CHARS = 220
    _MAX_TOOL_CALLS_TOKENS = 2500
    _MAX_VERIFIER_PROMPT_TOKENS = 8000
    _MAX_RESULT_SUMMARY_CHARS = 5000
    _COMPACT_RESULT_SUMMARY_CHARS = 1800
    _MAX_EVIDENCE_SECTION_CHARS = 2600
    _MAX_EVIDENCE_SECTION_COMPACT_CHARS = 1300
    _MAX_ARTIFACT_SECTION_CHARS = 2400
    _MAX_ARTIFACT_SECTION_COMPACT_CHARS = 1200
    _MAX_TOOL_OUTPUT_EXCERPT_CHARS = 700
    _MAX_ARTIFACT_FILE_EXCERPT_CHARS = 420
    _ADVISORY_TOOL_FAILURES = frozenset({"web_fetch", "web_fetch_html", "web_search"})
    _SOURCE_ATTRIBUTION_TOOLS = frozenset({"web_fetch", "web_fetch_html"})
    _TOOL_OUTPUT_EVIDENCE_TOOLS = frozenset(
        {
            "read_file",
            "spreadsheet",
            "document_write",
            "write_file",
        }
    )
    _ARTIFACT_PREVIEW_SUFFIXES = frozenset(
        {
            ".md",
            ".txt",
            ".csv",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".htm",
            ".rst",
        }
    )

    @staticmethod
    def _hard_cap_text(text: str, max_chars: int) -> str:
        """Deterministically bound text length when semantic compaction misses."""
        value = str(text or "")
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        if max_chars <= 40:
            return value[:max_chars]

        marker = "...[truncated]..."
        remaining = max_chars - len(marker)
        if remaining <= 0:
            return value[:max_chars]

        head = max(16, int(remaining * 0.65))
        tail = max(8, remaining - head)
        if head + tail > remaining:
            tail = max(0, remaining - head)
        compacted = f"{value[:head]}{marker}{value[-tail:] if tail else ''}"
        return compacted[:max_chars]

    async def _compact_text(self, text: str, *, max_chars: int, label: str) -> str:
        compacted = await self._compactor.compact(
            str(text or ""),
            max_chars=max_chars,
            label=label,
        )
        if len(compacted) > max_chars:
            logger.warning(
                "Verifier compaction exceeded budget for %s: got %d chars (limit %d)",
                label,
                len(compacted),
                max_chars,
            )
        return compacted

    async def _summarize_arg_value(
        self,
        value: object,
        *,
        max_chars: int = 80,
    ) -> object:
        if isinstance(value, str):
            return self._hard_cap_text(value, max_chars)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return self._hard_cap_text(
                json.dumps(value, ensure_ascii=False, default=str), max_chars
            )
        if isinstance(value, list):
            return self._hard_cap_text(
                json.dumps(value, ensure_ascii=False, default=str), max_chars
            )
        return self._hard_cap_text(str(value), max_chars)

    async def _summarize_tool_args(self, args: object, *, max_chars: int) -> str:
        if not isinstance(args, dict):
            return self._hard_cap_text(json.dumps(args, default=str), max_chars)

        preferred = (
            "path",
            "file_path",
            "url",
            "query",
            "pattern",
            "command",
            "operation",
            "source",
            "destination",
            "name",
            "column_name",
            "row_index",
            "action",
            "subtask_id",
        )
        summary: dict[str, object] = {}
        for key in preferred:
            if key in args:
                summary[key] = await self._summarize_arg_value(args.get(key))
        if not summary:
            for key, value in args.items():
                summary[str(key)] = await self._summarize_arg_value(value)

        text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        if len(text) <= max_chars:
            return text
        return self._hard_cap_text(text, max_chars)

    async def _format_tool_calls_compact(self, tool_calls: list) -> str:
        if not tool_calls:
            return "No tool calls."

        detailed = await self._format_tool_calls_for_prompt(tool_calls)
        return await self._compact_text(
            detailed,
            max_chars=2_400,
            label="verification compact tool-call history",
        )

    async def _format_tool_calls_for_prompt(self, tool_calls: list) -> str:
        if not tool_calls:
            return "No tool calls."

        lines: list[str] = []
        for tc in tool_calls:
            args_text = await self._summarize_tool_args(
                getattr(tc, "args", {}),
                max_chars=self._max_tool_args_chars,
            )
            result = getattr(tc, "result", None)
            if getattr(result, "success", False):
                status = "OK"
            else:
                error_text = str(getattr(result, "error", "") or "unknown error")
                compact_error = self._hard_cap_text(
                    error_text, self._max_tool_status_chars
                )
                if self._is_advisory_tool_failure(
                    str(getattr(tc, "tool", "") or ""),
                    error_text,
                ):
                    status = f"ADVISORY_FAILURE: {compact_error}"
                else:
                    status = f"FAILED: {compact_error}"
            tool_name = str(getattr(tc, "tool", "") or "unknown")
            line = f"- {tool_name}({args_text}) -> {status}"
            if getattr(result, "success", False):
                files_changed = [
                    str(item).strip()
                    for item in (getattr(result, "files_changed", []) or [])
                    if str(item).strip()
                ]
                if files_changed:
                    preview = ", ".join(files_changed[:4])
                    line += f"\n  files_changed: {preview}"
                excerpt = await self._tool_output_excerpt(
                    tool_name=tool_name,
                    args=getattr(tc, "args", {}),
                    result=result,
                )
                if excerpt:
                    indented_excerpt = self._indent_text_block(
                        excerpt,
                        prefix="    ",
                    )
                    line += f"\n  output_excerpt:\n{indented_excerpt}"
            lines.append(line)

        formatted = "\n".join(lines) if lines else "No tool calls."
        if estimate_tokens(formatted) > self._max_tool_calls_tokens:
            return await self._compact_text(
                formatted,
                max_chars=self._max_tool_calls_tokens * 4,
                label="verification tool-call history",
            )
        return formatted

    @staticmethod
    def _indent_text_block(text: str, *, prefix: str = "  ") -> str:
        if not text:
            return ""
        return "\n".join(f"{prefix}{line}" for line in str(text).splitlines())

    async def _tool_output_excerpt(
        self,
        *,
        tool_name: str,
        args: object,
        result: object,
    ) -> str:
        """Return bounded tool output excerpts that help semantic verification."""
        if tool_name not in self._TOOL_OUTPUT_EVIDENCE_TOOLS:
            return ""

        normalized_args = args if isinstance(args, dict) else {}
        if tool_name == "spreadsheet":
            operation = str(normalized_args.get("operation", "")).strip().lower()
            if operation not in {"read", "summary"}:
                return ""
        if tool_name == "document_write":
            parts: list[str] = []
            title = str(normalized_args.get("title", "")).strip()
            if title:
                parts.append(f"Title: {title}")
            raw_content = str(normalized_args.get("content", "")).strip()
            if raw_content:
                parts.append(raw_content)
            sections = normalized_args.get("sections", [])
            if isinstance(sections, list):
                for section in sections[:4]:
                    if not isinstance(section, dict):
                        continue
                    heading = str(section.get("heading", "")).strip()
                    body = str(section.get("body", "")).strip()
                    if heading:
                        parts.append(f"## {heading}")
                    if body:
                        parts.append(body)
            output = "\n\n".join(part for part in parts if part).strip()
            if not output:
                output = str(getattr(result, "output", "") or "").strip()
            if not output:
                return ""
            compacted = self._hard_cap_text(
                output, self._max_tool_output_excerpt_chars
            )
            return compacted.strip()

        if tool_name == "write_file":
            path = str(normalized_args.get("path", "")).strip()
            raw_content = str(normalized_args.get("content", "")).strip()
            if not raw_content:
                return ""
            prefix = f"Path: {path}\n\n" if path else ""
            compacted = self._hard_cap_text(
                f"{prefix}{raw_content}", self._max_tool_output_excerpt_chars
            )
            return compacted.strip()

        output = str(getattr(result, "output", "") or "").strip()
        if not output:
            return ""
        compacted = self._hard_cap_text(output, self._max_tool_output_excerpt_chars)
        return compacted.strip()

    @staticmethod
    def _claim_extraction_enabled(subtask: Subtask) -> bool:
        contract = getattr(subtask, "validity_contract_snapshot", {})
        if not isinstance(contract, dict) or not bool(contract.get("enabled", False)):
            return False
        extraction = contract.get("claim_extraction", {})
        return isinstance(extraction, dict) and bool(extraction.get("enabled", False))

    def _augment_verifier_prompt(
        self,
        prompt: str,
        *,
        subtask: Subtask,
        quality_contract: dict[str, object],
    ) -> str:
        """Inject inferred contracts that a process-owned assembler cannot know."""
        additions: list[str] = []
        if bool(quality_contract.get("enabled", False)) and bool(
            quality_contract.get("inferred", False)
        ):
            dimensions = [
                str(item).strip()
                for item in quality_contract.get("dimensions", [])
                if str(item).strip()
            ]
            additions.append(
                "INFERRED QUALITY AUDIT CONTRACT: Return metadata.quality with "
                "numeric 0..1 fields `overall` and `requirement_coverage`, a "
                "`dimensions` object containing every named dimension, and a "
                "`missing_targets` array of concrete repair targets. Required "
                f"dimensions: {', '.join(dimensions)}. Assess the actual artifacts; "
                "do not omit this object even when the verification passes."
            )
        if self._claim_extraction_enabled(subtask):
            additions.append(
                "CLAIM LIFECYCLE CONTRACT: Return metadata.claim_lifecycle as an "
                "array for material factual claims. Each item must contain "
                "claim_id, text, claim_type, criticality, status, evidence_refs, "
                "reason_code, and lifecycle. Mark a claim supported only when an "
                "explicit evidence reference supports it; otherwise use "
                "insufficient_evidence or contradicted. Do not invent references."
            )
        if not additions:
            return prompt
        return f"{prompt}\n\n" + "\n\n".join(additions)

    async def _build_artifact_content_section(
        self,
        *,
        subtask: Subtask,
        workspace: Path | None,
        tool_calls: list | None,
        max_chars: int,
        quality_contract: dict[str, object] | None = None,
    ) -> str:
        """Build bounded excerpts from changed artifacts for semantic review."""
        if workspace is None:
            return ""

        workspace_resolved = workspace.resolve()
        changed_files: list[tuple[str, Path]] = []
        seen: set[str] = set()

        quality_enabled = bool(
            isinstance(quality_contract, dict) and quality_contract.get("enabled", False)
        )
        process = getattr(self._prompts, "process", None)
        if quality_enabled and process is not None:
            for raw_path, _artifact_role in self._quality_artifact_paths(subtask):
                rel = str(raw_path or "").strip()
                if not rel or rel in seen:
                    continue
                candidate = (workspace / rel).resolve()
                try:
                    candidate.relative_to(workspace_resolved)
                except ValueError:
                    continue
                if not candidate.exists() or not candidate.is_file():
                    continue
                suffix = candidate.suffix.lower()
                if suffix and suffix not in self._ARTIFACT_PREVIEW_SUFFIXES:
                    continue
                seen.add(rel)
                changed_files.append((rel, candidate))

        for tc in tool_calls or []:
            result = getattr(tc, "result", None)
            if result is None or not getattr(result, "success", False):
                continue
            files_changed = getattr(result, "files_changed", []) or []
            for rel_path in files_changed:
                rel = str(rel_path or "").strip()
                if not rel or rel in seen:
                    continue
                seen.add(rel)
                candidate = (workspace / rel).resolve()
                try:
                    candidate.relative_to(workspace_resolved)
                except ValueError:
                    continue
                if not candidate.exists() or not candidate.is_file():
                    continue
                suffix = candidate.suffix.lower()
                if suffix and suffix not in self._ARTIFACT_PREVIEW_SUFFIXES:
                    continue
                changed_files.append((rel, candidate))

        if not changed_files:
            return ""

        lines = [
            "ARTIFACT CONTENT SNAPSHOT (for semantic review):",
            (
                "- Complete expected phase artifacts are included when the quality "
                "contract is enabled; otherwise this contains changed-file excerpts."
            ),
        ]
        selected_files = changed_files if quality_enabled else changed_files[:6]
        for rel, path in selected_files:
            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            if not content:
                continue
            lines.append(f"- {rel}:")
            lines.append(self._indent_text_block(content, prefix="    "))

        if len(lines) <= 2:
            return ""
        snapshot = "\n".join(lines)
        if len(snapshot) <= max_chars:
            return snapshot
        return await self._compact_text(
            snapshot,
            max_chars=max_chars,
            label="verification artifact snapshot batch",
        )

    def _quality_contract_for_subtask(self, subtask: Subtask) -> dict[str, object]:
        process = getattr(self._prompts, "process", None)
        resolver = getattr(process, "resolve_quality_contract_for_phase", None)
        if not callable(resolver):
            if not bool(getattr(subtask, "is_synthesis", False)):
                return {}
            return {
                "enabled": True,
                "dimensions": [
                    "completeness",
                    "evidence_traceability",
                    "analytical_depth",
                    "cross_artifact_synthesis",
                    "uncertainty_handling",
                    "internal_consistency",
                ],
                "minimum_overall_score": 0.72,
                "minimum_dimension_score": 0.60,
                "minimum_requirement_coverage": 0.80,
                "required_sections": [],
                "inferred": True,
            }
        phase_id = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
        resolved = resolver(phase_id)
        return dict(resolved) if isinstance(resolved, dict) else {}

    def _quality_artifact_paths(self, subtask: Subtask) -> list[tuple[str, str]]:
        """Return current and transitive upstream artifacts for a quality audit."""
        process = getattr(self._prompts, "process", None)
        if process is None:
            return []
        deliverables = process.get_deliverables()
        phase_id = str(getattr(subtask, "phase_id", "") or "").strip() or subtask.id
        phase_by_id = {
            str(phase.id or "").strip(): phase
            for phase in process.phases
            if str(phase.id or "").strip()
        }
        paths: list[tuple[str, str]] = []
        seen_paths: set[str] = set()

        def add_phase(artifact_phase_id: str, role: str) -> None:
            for raw_path in deliverables.get(artifact_phase_id, []):
                path = str(raw_path or "").strip()
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    paths.append((path, role))

        add_phase(phase_id, "current")
        phase = phase_by_id.get(phase_id)
        pending = [
            str(item or "").strip()
            for item in (
                list(phase.depends_on or [])
                if phase is not None
                else list(getattr(subtask, "depends_on", []) or [])
            )
            if str(item or "").strip()
        ]
        visited: set[str] = set()
        while pending:
            dependency_id = pending.pop(0)
            if dependency_id in visited:
                continue
            visited.add(dependency_id)
            add_phase(dependency_id, "upstream")
            dependency = phase_by_id.get(dependency_id)
            if dependency is not None:
                pending.extend(
                    str(item or "").strip()
                    for item in list(dependency.depends_on or [])
                    if str(item or "").strip()
                )
        return paths

    @staticmethod
    def _quality_score(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if score > 1.0 and score <= 100.0:
            score /= 100.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _normalize_quality_label(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    @classmethod
    def _quality_dimension_key(cls, value: object) -> str:
        return cls._normalize_quality_label(value).replace(" ", "_")

    @classmethod
    def _quality_heading_matches(cls, requirement: object, heading: object) -> bool:
        required = cls._normalize_quality_label(requirement)
        candidate = cls._normalize_quality_label(heading)
        if not required or not candidate:
            return False
        if required in candidate or candidate in required:
            return True

        def meaningful_tokens(value: str) -> set[str]:
            stopwords = {"a", "an", "and", "for", "of", "the", "to", "with"}
            return {
                token[:-1] if len(token) > 4 and token.endswith("s") else token
                for token in value.split()
                if token not in stopwords
            }

        required_tokens = meaningful_tokens(required)
        candidate_tokens = meaningful_tokens(candidate)
        if len(required_tokens) < 2:
            return required_tokens == candidate_tokens
        return (len(required_tokens.intersection(candidate_tokens)) / len(required_tokens)) >= 0.75

    @staticmethod
    def _normalize_source_url(value: object) -> str:
        """Normalize public source URLs for deterministic traceability matching."""
        raw = str(value or "").strip().rstrip(".,;:!?)]}>'\"")
        try:
            parsed = urlparse(raw)
        except ValueError:
            return ""
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return ""
        scheme = parsed.scheme.lower()
        try:
            hostname = (parsed.hostname or "").lower()
            parsed_port = parsed.port
        except ValueError:
            return ""
        if not hostname:
            return ""
        port = f":{parsed_port}" if parsed_port else ""
        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/") or "/"
        return parsed._replace(
            scheme=scheme,
            netloc=f"{hostname}{port}",
            path=path,
            fragment="",
        ).geturl()

    def _artifact_quality_diagnostics(
        self,
        *,
        subtask: Subtask,
        workspace: Path | None,
        quality_contract: dict[str, object],
    ) -> dict[str, object]:
        if workspace is None or not bool(quality_contract.get("enabled", False)):
            return {}
        process = getattr(self._prompts, "process", None)
        if process is None:
            return {}
        audit_paths = self._quality_artifact_paths(subtask)
        expected = [path for path, role in audit_paths if role == "current"]
        artifacts: list[dict[str, object]] = []
        current_evidence_refs: set[str] = set()
        upstream_evidence_refs: set[str] = set()
        current_source_urls: set[str] = set()
        upstream_source_urls: set[str] = set()
        headings: list[str] = []
        url_count = 0
        workspace_resolved = workspace.resolve()
        for rel, artifact_role in audit_paths:
            path = (workspace / rel).resolve()
            try:
                path.relative_to(workspace_resolved)
            except ValueError:
                artifacts.append(
                    {
                        "path": rel,
                        "role": artifact_role,
                        "exists": False,
                        "unsafe_path": True,
                    }
                )
                continue
            row: dict[str, object] = {
                "path": rel,
                "role": artifact_role,
                "exists": path.is_file(),
            }
            if not path.is_file():
                artifacts.append(row)
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                artifacts.append(row)
                continue
            row["chars"] = len(text)
            row["nonempty"] = bool(text.strip())
            discovered_urls = {
                normalized
                for match in re.findall(r"https?://[^\s<>\]]+", text)
                if (normalized := self._normalize_source_url(match))
            }
            url_count += len(discovered_urls)
            if path.name.lower() not in {"source-index.csv", "references.csv"}:
                refs = {
                    match.strip().upper()
                    for match in re.findall(
                        (
                            r"\b(?:EV|EVID|SRC|WEB)(?:[-_][A-Za-z0-9]"
                            r"[A-Za-z0-9_-]*|[0-9][A-Za-z0-9_-]*)\b"
                        ),
                        text,
                        flags=re.IGNORECASE,
                    )
                    if match.strip()
                }
                if artifact_role == "current":
                    current_evidence_refs.update(refs)
                    current_source_urls.update(discovered_urls)
                else:
                    upstream_evidence_refs.update(refs)
                    upstream_source_urls.update(discovered_urls)
            if path.suffix.lower() in {".md", ".markdown"}:
                artifact_headings = re.findall(r"^#{1,6}\s+(.+?)\s*$", text, re.MULTILINE)
                headings.extend(artifact_headings)
                row.update(
                    {
                        "kind": "markdown",
                        "word_count": len(re.findall(r"\b\w+\b", text)),
                        "heading_count": len(artifact_headings),
                    }
                )
            elif path.suffix.lower() in {".csv", ".tsv"}:
                delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
                try:
                    parsed = list(csv.reader(text.splitlines(), delimiter=delimiter))
                except csv.Error:
                    parsed = []
                header = parsed[0] if parsed else []
                body = parsed[1:] if len(parsed) > 1 else []
                cells = [str(cell).strip() for record in body for cell in record]
                row.update(
                    {
                        "kind": "table",
                        "row_count": len(body),
                        "column_count": len(header),
                        "empty_cell_ratio": (
                            round(sum(not cell for cell in cells) / len(cells), 4) if cells else 0.0
                        ),
                    }
                )
            artifacts.append(row)

        source_index_ids: set[str] = set()
        source_index_urls: set[str] = set()
        source_index_id_to_url: dict[str, str] = {}
        duplicate_source_index_ids: set[str] = set()
        source_index_missing_url_count = 0
        source_index = workspace / "source-index.csv"
        if source_index.is_file():
            try:
                with source_index.open(newline="", encoding="utf-8-sig") as handle:
                    for row in csv.DictReader(handle):
                        value = str(row.get("evidence_id", "") or "").strip()
                        if value:
                            normalized_value = value.upper()
                            if normalized_value in source_index_ids:
                                duplicate_source_index_ids.add(normalized_value)
                            source_index_ids.add(normalized_value)
                        source_url = next(
                            (
                                str(row.get(key, "") or "").strip()
                                for key in ("source_url", "url", "source", "citation")
                                if str(row.get(key, "") or "").strip()
                            ),
                            "",
                        )
                        if value and not source_url:
                            source_index_missing_url_count += 1
                        normalized_url = self._normalize_source_url(source_url)
                        if normalized_url:
                            source_index_urls.add(normalized_url)
                            if value:
                                source_index_id_to_url[normalized_value] = normalized_url
            except (OSError, csv.Error):
                source_index_ids = set()
                source_index_urls = set()
                source_index_id_to_url = {}

        def canonical_reference_atoms(
            evidence_ids: set[str],
            source_urls: set[str],
        ) -> set[str]:
            return {
                *source_urls,
                *(
                    source_index_id_to_url.get(evidence_id, f"id:{evidence_id}")
                    for evidence_id in evidence_ids
                ),
            }

        current_reference_atoms = canonical_reference_atoms(
            current_evidence_refs,
            current_source_urls,
        )
        upstream_reference_atoms = canonical_reference_atoms(
            upstream_evidence_refs,
            upstream_source_urls,
        )
        indexed_reference_atoms = {
            *source_index_urls,
            *(
                source_index_id_to_url.get(evidence_id, f"id:{evidence_id}")
                for evidence_id in source_index_ids
            ),
        }
        current_reference_count = len(current_reference_atoms)
        matched_current_reference_count = len(
            current_reference_atoms.intersection(indexed_reference_atoms),
        )
        traceability_ratio = (
            matched_current_reference_count / current_reference_count
            if current_reference_count
            else 0.0
        )
        upstream_reference_count = len(upstream_reference_atoms)
        reused_upstream_reference_count = len(
            current_reference_atoms.intersection(upstream_reference_atoms),
        )
        upstream_evidence_reuse_ratio = (
            reused_upstream_reference_count / upstream_reference_count
            if upstream_reference_count
            else 1.0
        )
        required_sections = [
            str(item or "").strip()
            for item in quality_contract.get("required_sections", [])
            if str(item or "").strip()
        ]
        covered_sections = 0
        missing_sections: list[str] = []
        for requirement in required_sections:
            matched = any(
                self._quality_heading_matches(requirement, heading) for heading in headings
            )
            if matched:
                covered_sections += 1
            else:
                missing_sections.append(requirement)
        return {
            "expected_artifact_count": len(expected),
            "present_artifact_count": sum(
                bool(item.get("exists")) for item in artifacts if item.get("role") == "current"
            ),
            "nonempty_artifact_count": sum(
                bool(item.get("nonempty")) for item in artifacts if item.get("role") == "current"
            ),
            "upstream_artifact_count": sum(item.get("role") == "upstream" for item in artifacts),
            "missing_upstream_artifact_count": sum(
                item.get("role") == "upstream"
                and (not bool(item.get("exists", False)) or not bool(item.get("nonempty", False)))
                for item in artifacts
            ),
            "artifact_metrics": artifacts,
            "evidence_reference_count": len(current_evidence_refs),
            "source_url_reference_count": len(current_source_urls),
            "upstream_evidence_reference_count": len(upstream_evidence_refs),
            "upstream_source_url_reference_count": len(upstream_source_urls),
            "source_index_evidence_count": len(source_index_ids),
            "source_index_url_count": len(source_index_urls),
            "matched_evidence_reference_count": len(
                current_evidence_refs.intersection(source_index_ids),
            ),
            "matched_source_url_reference_count": len(
                current_source_urls.intersection(source_index_urls),
            ),
            "traceability_ratio": round(traceability_ratio, 4),
            "upstream_evidence_reuse_ratio": round(
                upstream_evidence_reuse_ratio,
                4,
            ),
            "orphan_evidence_reference_count": len(
                current_reference_atoms.difference(indexed_reference_atoms),
            ),
            "duplicate_source_index_id_count": len(duplicate_source_index_ids),
            "source_index_missing_url_count": source_index_missing_url_count,
            "url_count": url_count,
            "required_section_count": len(required_sections),
            "covered_section_count": covered_sections,
            "missing_sections": missing_sections,
            "section_coverage_ratio": (
                round(covered_sections / len(required_sections), 4) if required_sections else 1.0
            ),
        }

    def _apply_quality_contract(
        self,
        *,
        result: VerificationResult,
        quality_contract: dict[str, object],
        diagnostics: dict[str, object],
    ) -> VerificationResult:
        if not bool(quality_contract.get("enabled", False)):
            return result
        metadata = dict(result.metadata) if isinstance(result.metadata, dict) else {}
        quality = metadata.get("quality", {})
        if not isinstance(quality, dict):
            quality = {}
        overall = self._quality_score(quality.get("overall"))
        dimensions_raw = quality.get("dimensions", {})
        dimensions: dict[str, float] = {}
        if isinstance(dimensions_raw, dict):
            for key, value in dimensions_raw.items():
                score = self._quality_score(value)
                if score is not None and str(key or "").strip():
                    dimensions[self._quality_dimension_key(key)] = score
        required_dimensions = [
            self._quality_dimension_key(item)
            for item in quality_contract.get("dimensions", [])
            if self._quality_dimension_key(item)
        ]
        missing_dimensions = [item for item in required_dimensions if item not in dimensions]
        minimum_overall = (
            self._quality_score(
                quality_contract.get("minimum_overall_score", 0.0),
            )
            or 0.0
        )
        minimum_dimension = (
            self._quality_score(
                quality_contract.get("minimum_dimension_score", 0.0),
            )
            or 0.0
        )
        minimum_requirement_coverage = (
            self._quality_score(
                quality_contract.get("minimum_requirement_coverage", 0.0),
            )
            or 0.0
        )
        minimum_traceability = (
            self._quality_score(
                quality_contract.get("minimum_traceability_ratio", 0.0),
            )
            or 0.0
        )
        minimum_upstream_reuse = (
            self._quality_score(
                quality_contract.get("minimum_upstream_evidence_reuse_ratio", 0.0),
            )
            or 0.0
        )
        traceability = self._quality_score(diagnostics.get("traceability_ratio")) or 0.0
        upstream_reuse = (
            self._quality_score(
                diagnostics.get("upstream_evidence_reuse_ratio"),
            )
            or 0.0
        )
        section_coverage = (
            self._quality_score(
                diagnostics.get("section_coverage_ratio"),
            )
            or 0.0
        )
        requirement_coverage = self._quality_score(
            quality.get("requirement_coverage"),
        )
        missing_targets = (
            [
                str(item or "").strip()
                for item in quality.get("missing_targets", [])
                if str(item or "").strip()
            ]
            if isinstance(quality.get("missing_targets", []), list)
            else []
        )
        missing_targets.extend(
            str(item or "").strip()
            for item in diagnostics.get("missing_sections", [])
            if str(item or "").strip()
        )
        artifact_metrics = diagnostics.get("artifact_metrics", [])
        if isinstance(artifact_metrics, list):
            missing_targets.extend(
                f"artifact content: {str(item.get('path', '')).strip()}"
                for item in artifact_metrics
                if isinstance(item, dict)
                and str(item.get("path", "")).strip()
                and (not bool(item.get("exists", False)) or not bool(item.get("nonempty", False)))
            )
        dimension_failures = [key for key, score in dimensions.items() if score < minimum_dimension]
        failure_reason = ""
        if overall is None or missing_dimensions or requirement_coverage is None:
            failure_reason = "quality_assessment_missing"
            missing_targets.extend(f"quality score: {item}" for item in missing_dimensions)
            if requirement_coverage is None:
                missing_targets.append("quality score: requirement_coverage")
        elif (
            overall < minimum_overall
            or dimension_failures
            or section_coverage < 1.0
            or (
                requirement_coverage is not None
                and requirement_coverage < minimum_requirement_coverage
            )
            or int(diagnostics.get("present_artifact_count", 0) or 0)
            < int(diagnostics.get("expected_artifact_count", 0) or 0)
            or int(diagnostics.get("nonempty_artifact_count", 0) or 0)
            < int(diagnostics.get("expected_artifact_count", 0) or 0)
            or int(diagnostics.get("missing_upstream_artifact_count", 0) or 0) > 0
        ):
            failure_reason = "quality_below_threshold"
            missing_targets.extend(f"quality dimension: {item}" for item in dimension_failures)
        elif (
            (minimum_traceability > 0 and traceability < minimum_traceability)
            or (
                minimum_upstream_reuse > 0
                and int(diagnostics.get("upstream_evidence_reference_count", 0) or 0) > 0
                and upstream_reuse < minimum_upstream_reuse
            )
            or (
                bool(quality_contract.get("require_source_index_urls", False))
                and int(diagnostics.get("source_index_missing_url_count", 0) or 0) > 0
            )
            or int(diagnostics.get("duplicate_source_index_id_count", 0) or 0) > 0
        ):
            failure_reason = "evidence_traceability_below_threshold"
            if minimum_upstream_reuse > 0 and upstream_reuse < minimum_upstream_reuse:
                missing_targets.append("upstream evidence carried into synthesis")
            if int(diagnostics.get("source_index_missing_url_count", 0) or 0) > 0:
                missing_targets.append("source-index rows with source URLs")
            if int(diagnostics.get("duplicate_source_index_id_count", 0) or 0) > 0:
                missing_targets.append("unique source-index evidence IDs")

        quality.update(
            {
                "overall": overall,
                "dimensions": dimensions,
                "minimum_overall_score": minimum_overall,
                "minimum_dimension_score": minimum_dimension,
                "minimum_requirement_coverage": minimum_requirement_coverage,
                "minimum_traceability_ratio": minimum_traceability,
                "minimum_upstream_evidence_reuse_ratio": minimum_upstream_reuse,
                "requirement_coverage": requirement_coverage,
                "diagnostics": diagnostics,
                "missing_targets": list(dict.fromkeys(missing_targets)),
                "meets_floor": not bool(failure_reason),
                "reason_code": failure_reason,
                "policy_mode": str(
                    getattr(self._config, "quality_policy_mode", "enforce") or "enforce"
                )
                .strip()
                .lower(),
            }
        )
        metadata["quality"] = quality
        metadata["missing_targets"] = list(
            dict.fromkeys(
                [
                    *(
                        metadata.get("missing_targets", [])
                        if isinstance(metadata.get("missing_targets", []), list)
                        else []
                    ),
                    *missing_targets,
                ],
            )
        )
        result.metadata = metadata
        policy_mode = str(quality.get("policy_mode", "enforce") or "enforce")
        if not failure_reason or not result.passed or policy_mode == "observe":
            return result
        result.passed = False
        result.outcome = "fail"
        result.reason_code = failure_reason
        result.severity_class = "semantic"
        result.confidence = min(float(result.confidence or 0.0), 0.8)
        detail = (
            "Output quality audit did not meet the process quality floor. "
            "Revise only the listed missing requirements or weak dimensions, "
            "preserving already-supported work."
        )
        result.feedback = f"{result.feedback}\n{detail}".strip() if result.feedback else detail
        return result

    def _is_advisory_tool_failure(
        self,
        tool_name: str,
        error: str | None,
    ) -> bool:
        if tool_name not in self._ADVISORY_TOOL_FAILURES or not error:
            return False
        text = error.lower()
        hard_fail_markers = (
            "blocked host:",
            "safety violation:",
            "only http:// and https:// urls are allowed",
            "no url provided",
            "no search query provided",
        )
        if any(marker in text for marker in hard_fail_markers):
            return False

        status_code = _extract_http_status(error)
        if status_code is not None and 400 <= status_code < 600:
            return True
        return any(
            marker in text
            for marker in (
                "timeout",
                "timed out",
                "connection failed",
                "temporarily unavailable",
                "not found",
                "rate limit",
                "rate-limited",
                "response too large",
            )
        )

    def _build_prompt(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls_formatted: str,
        *,
        llm_rules: list | None = None,
        phase_scope_default: str = "current_phase",
    ) -> str:
        return verification_prompting.build_verifier_prompt(
            self._prompts,
            subtask=subtask,
            result_summary=result_summary,
            tool_calls_formatted=tool_calls_formatted,
            llm_rules=llm_rules,
            phase_scope_default=phase_scope_default,
        )

    def _phase_scope_default(self) -> str:
        return verification_prompting.phase_scope_default(self._config)

    def _expected_verifier_response_keys(self) -> list[str]:
        return verification_prompting.expected_verifier_response_keys(self._prompts)

    def _emit_rule_scope_event(
        self,
        *,
        task_id: str,
        subtask_id: str,
        applied: bool,
        rule,
        reason: str,
    ) -> None:
        verification_events.emit_rule_scope_event(
            self._event_bus,
            task_id=task_id,
            subtask_id=subtask_id,
            applied=applied,
            rule=rule,
            reason=reason,
        )

    def _select_phase_scoped_rules(
        self,
        subtask: Subtask,
        *,
        task_id: str,
    ) -> list:
        process = getattr(self._prompts, "process", None)
        if process is None or not process.has_verification_rules():
            return []

        all_rules = list(process.llm_rules())
        phase_scope_default = self._phase_scope_default()
        if phase_scope_default == "global":
            for rule in all_rules:
                self._emit_rule_scope_event(
                    task_id=task_id,
                    subtask_id=subtask.id,
                    applied=True,
                    rule=rule,
                    reason="global_mode",
                )
            return all_rules

        scoped_rules = process.llm_rules_for_subtask(
            subtask.id,
            phase_scope_default=phase_scope_default,
        )
        scoped_ids = {id(rule) for rule in scoped_rules}
        for rule in all_rules:
            is_applied = id(rule) in scoped_ids
            self._emit_rule_scope_event(
                task_id=task_id,
                subtask_id=subtask.id,
                applied=is_applied,
                rule=rule,
                reason="phase_match" if is_applied else "phase_mismatch",
            )
        return scoped_rules

    @staticmethod
    def _normalized_text(value: object) -> str:
        return " ".join(str(value or "").strip().split())

    @staticmethod
    def _normalize_url(value: object) -> str:
        text = str(value or "").strip().strip("<>")
        if not text:
            return ""
        return text.rstrip("/")

    @classmethod
    def _domain_from_url(cls, value: object) -> str:
        text = cls._normalize_url(value)
        if not text:
            return ""
        try:
            return (urlparse(text).netloc or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _merge_evidence_records(
        base: list[dict] | None,
        extra: list[dict] | None,
    ) -> list[dict]:
        try:
            from loom.state.evidence import merge_evidence_records

            return merge_evidence_records(base or [], extra or [])
        except Exception:
            merged: list[dict] = []
            seen: set[str] = set()
            for bucket in (base or [], extra or []):
                for item in bucket:
                    if not isinstance(item, dict):
                        continue
                    evidence_id = str(item.get("evidence_id", "")).strip()
                    if evidence_id and evidence_id in seen:
                        continue
                    if evidence_id:
                        seen.add(evidence_id)
                    merged.append(dict(item))
            return merged

    def _extract_evidence_from_tool_calls(
        self,
        subtask_id: str,
        tool_calls: list | None,
    ) -> list[dict]:
        if not tool_calls:
            return []
        try:
            from loom.state.evidence import extract_evidence_records
        except Exception:
            return []
        try:
            return extract_evidence_records(
                task_id="",
                subtask_id=subtask_id,
                tool_calls=list(tool_calls),
                existing_ids=set(),
                context_text_max_chars=self._evidence_context_text_max_chars,
            )
        except Exception:
            return []

    @classmethod
    def _build_evidence_ledger_section(
        cls,
        records: list[dict],
        *,
        workspace: Path | None,
        tool_calls: list | None,
        max_chars: int,
    ) -> str:
        if not records:
            return ""

        facet_keys: set[str] = set()
        facet_examples: set[str] = set()
        domains: set[str] = set()
        tools: set[str] = set()

        for item in records:
            if not isinstance(item, dict):
                continue
            tool = cls._normalized_text(item.get("tool")).lower() or "unknown"
            tools.add(tool)

            facets = item.get("facets", {})
            if isinstance(facets, dict):
                for key, value in facets.items():
                    facet_key = cls._normalized_text(key).lower()
                    facet_val = cls._normalized_text(value)
                    if facet_key:
                        facet_keys.add(facet_key)
                    if facet_key and facet_val:
                        facet_examples.add(f"{facet_key}={facet_val}")

            source_url = cls._normalize_url(item.get("source_url"))
            if source_url:
                domain = cls._domain_from_url(source_url) or "unknown-domain"
                domains.add(domain)

        lines = ["EVIDENCE CONTEXT SNAPSHOT (advisory, non-binding):"]
        lines.append(
            "- Treat this as support context for LLM judgment, not a strict schema validator."
        )
        lines.append(
            "- Structured outputs may vary by process; infer schema from "
            "headers/sections and acceptance criteria."
        )
        if tools:
            lines.append("- observed_tools: " + ", ".join(sorted(tools)[:10]))
        if facet_keys:
            lines.append("- observed_facet_keys: " + ", ".join(sorted(facet_keys)[:12]))
        if facet_examples:
            lines.append("- observed_facet_examples: " + ", ".join(sorted(facet_examples)[:12]))
        if domains:
            lines.append("- observed_source_domain_examples: " + ", ".join(sorted(domains)[:10]))

        csv_lines = cls._csv_schema_lines(
            workspace=workspace,
            tool_calls=tool_calls,
        )
        if csv_lines:
            lines.append("- changed_csv_schema_hints:")
            lines.extend(csv_lines)

        lines.append("- sample_records:")
        for item in records[:10]:
            if not isinstance(item, dict):
                continue
            evidence_id = cls._normalized_text(item.get("evidence_id")) or "EV-UNKNOWN"
            facets = {}
            raw_facets = item.get("facets", {})
            if isinstance(raw_facets, dict):
                facets = {
                    cls._normalized_text(key): cls._normalized_text(value)
                    for key, value in raw_facets.items()
                    if cls._normalized_text(key) and cls._normalized_text(value)
                }
            facet_preview = "none"
            if facets:
                facet_preview = ", ".join(f"{key}={value}" for key, value in sorted(facets.items()))
                if len(facet_preview) > 96:
                    facet_preview = facet_preview[:93] + "..."
            source = (
                cls._normalize_url(item.get("source_url"))
                or cls._normalized_text(item.get("query"))
                or "n/a"
            )
            if len(source) > 96:
                source = source[:93] + "..."
            lines.append(f"  - {evidence_id} | facets={facet_preview} | source={source}")

        return cls._hard_cap_text("\n".join(lines), max_chars=max_chars)

    @staticmethod
    def _is_attribution_column(header: str) -> bool:
        h = str(header or "").strip().lower().replace(" ", "_")
        if not h:
            return False
        if "url" in h or "citation" in h or "reference" in h:
            return True
        if h in {"source", "source_link", "source_links"}:
            return True
        if h.startswith("source_") and any(
            marker in h for marker in ("url", "link", "ref", "citation")
        ):
            return True
        return False

    @classmethod
    def _csv_schema_lines(
        cls,
        *,
        workspace: Path | None,
        tool_calls: list | None,
    ) -> list[str]:
        if workspace is None or not tool_calls:
            return []

        changed_csv: list[Path] = []
        seen: set[str] = set()
        for tc in tool_calls:
            result = getattr(tc, "result", None)
            files_changed = getattr(result, "files_changed", []) if result else []
            for rel_path in files_changed:
                path = workspace / str(rel_path)
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                if path.suffix.lower() != ".csv" or not path.exists() or not path.is_file():
                    continue
                changed_csv.append(path)

        lines: list[str] = []
        for path in changed_csv[:8]:
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f)
                    headers = list(reader.fieldnames or [])
                    attrib_cols = [col for col in headers if cls._is_attribution_column(col)]
                    rows = list(reader)
            except Exception:
                continue

            header_preview = ", ".join(headers) if headers else "(none)"
            if len(header_preview) > 420:
                header_preview = header_preview[:417] + "..."
            lines.append(f"  - {path.name}: headers [{header_preview}]")

            if not attrib_cols or not rows:
                continue
            col_text = ", ".join(attrib_cols)
            lines.append(f"    attribution_columns: [{col_text}]")

        return lines

    def _emit_compactor_model_event(self, payload: dict) -> None:
        """Bridge semantic-compactor model events into verifier model_invocation events."""
        context = _COMPACTOR_EVENT_CONTEXT.get()
        if not context:
            return
        task_id, subtask_id = context
        model_name = str(payload.get("model", "")).strip() or "unknown"
        phase = str(payload.get("phase", "")).strip() or "done"
        details = {key: value for key, value in payload.items() if key not in {"model", "phase"}}
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=model_name,
            phase=phase,
            details=details,
        )

    def _emit_model_event(
        self,
        *,
        task_id: str,
        subtask_id: str,
        model_name: str,
        phase: str,
        details: dict | None = None,
    ) -> None:
        verification_events.emit_model_invocation_event(
            self._event_bus,
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=model_name,
            phase=phase,
            details=details,
        )

    @staticmethod
    def _to_bool(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "pass", "passed", "success", "successful"}:
                return True
            if lowered in {"false", "no", "fail", "failed", "failure", "unsuccessful"}:
                return False
        return None

    @staticmethod
    def _to_confidence(value: object) -> float | None:
        if isinstance(value, (int, float)):
            numeric = float(value)
            if numeric > 1.0 and numeric <= 100.0:
                numeric = numeric / 100.0
            return max(0.0, min(1.0, numeric))

        if isinstance(value, str):
            text = value.strip().lower()
            if not text:
                return None
            if text.endswith("%"):
                try:
                    numeric = float(text[:-1]) / 100.0
                    return max(0.0, min(1.0, numeric))
                except ValueError:
                    return None
            try:
                numeric = float(text)
            except ValueError:
                return None
            if numeric > 1.0 and numeric <= 100.0:
                numeric = numeric / 100.0
            return max(0.0, min(1.0, numeric))
        return None

    @staticmethod
    def _to_issues(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text or text.lower() in {"none", "no issues", "n/a"}:
                return []
            pieces = re.split(r"[;\n]+", text)
            return [piece.strip(" -\t") for piece in pieces if piece.strip(" -\t")]
        if isinstance(value, list):
            issues: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    issues.append(text)
            return issues
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _to_int(value: object) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            try:
                return int(float(text))
            except ValueError:
                return None
        return None

    @staticmethod
    def _to_string_list(value: object) -> list[str]:
        if isinstance(value, list):
            items = [str(item or "").strip() for item in value]
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            items = re.split(r"[,\n;]+", text)
        else:
            return []
        normalized: list[str] = []
        for item in items:
            text = str(item or "").strip()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    @classmethod
    def _normalize_verifier_metadata(cls, metadata: dict[str, object]) -> dict[str, object]:
        normalized: dict[str, object] = {
            str(key).strip(): value for key, value in metadata.items() if str(key).strip()
        }
        lookup: dict[str, str] = {key.lower(): key for key in normalized}

        def value_for(field_name: str) -> object | None:
            key = lookup.get(field_name)
            if key is None:
                return None
            return normalized.get(key)

        remediation_required_raw = value_for("remediation_required")
        if remediation_required_raw is not None:
            remediation_required = cls._to_bool(remediation_required_raw)
            if remediation_required is None:
                remediation_required = bool(remediation_required_raw)
            normalized["remediation_required"] = remediation_required

        remediation_mode_raw = value_for("remediation_mode")
        if remediation_mode_raw is not None:
            normalized["remediation_mode"] = (
                str(
                    remediation_mode_raw or "",
                )
                .strip()
                .lower()
            )

        missing_targets_raw = value_for("missing_targets")
        if missing_targets_raw is not None:
            normalized["missing_targets"] = cls._to_string_list(missing_targets_raw)

        unverified_claim_count_raw = value_for("unverified_claim_count")
        if unverified_claim_count_raw is not None:
            unverified_claim_count = cls._to_int(unverified_claim_count_raw)
            if unverified_claim_count is not None:
                normalized["unverified_claim_count"] = max(0, unverified_claim_count)

        verified_claim_count_raw = value_for("verified_claim_count")
        if verified_claim_count_raw is not None:
            verified_claim_count = cls._to_int(verified_claim_count_raw)
            if verified_claim_count is not None:
                normalized["verified_claim_count"] = max(0, verified_claim_count)

        supporting_ratio_raw = value_for("supporting_ratio")
        if supporting_ratio_raw is not None:
            supporting_ratio = cls._to_confidence(supporting_ratio_raw)
            if supporting_ratio is not None:
                normalized["supporting_ratio"] = supporting_ratio

        return normalized

    @staticmethod
    def _normalize_outcome(value: object, *, passed: bool) -> str:
        normalized = str(value or "").strip().lower()
        if normalized in _VALID_OUTCOMES:
            return normalized
        return "pass" if passed else "fail"

    @staticmethod
    def _extract_reason_code_from_text(text: str) -> str:
        match = re.search(
            r"\breason_code\b\s*[:=]\s*([a-z0-9_]+)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        return str(match.group(1)).strip().lower() if match else ""

    @staticmethod
    def _extract_severity_class_from_text(text: str) -> str:
        match = re.search(
            r"\b(?:severity_class|severity)\b\s*[:=]\s*([a-z_]+)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        severity = str(match.group(1)).strip().lower()
        if severity in _VALID_SEVERITY_CLASSES:
            return severity
        return ""

    @classmethod
    def _extract_outcome_from_text(cls, text: str, *, passed: bool) -> str:
        match = re.search(
            r"\boutcome\b\s*[:=]\s*([a-z_]+)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        if not match:
            return "pass" if passed else "fail"
        return cls._normalize_outcome(match.group(1), passed=passed)

    @classmethod
    def _coerce_assessment_mapping(cls, payload: dict) -> dict | None:
        return verification_parsing.coerce_assessment_mapping(
            payload,
            valid_outcomes=_VALID_OUTCOMES,
            valid_severity_classes=_VALID_SEVERITY_CLASSES,
        )

    @staticmethod
    def _strip_outer_fences(text: str) -> str:
        value = text.strip()
        if not value.startswith("```"):
            return value
        lines = value.splitlines()
        if not lines:
            return value
        body = "\n".join(lines[1:])
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
        return body.strip()

    @classmethod
    def _parse_yaml_like_assessment(cls, text: str) -> dict | None:
        try:
            import yaml
        except Exception:
            return None

        cleaned = cls._strip_outer_fences(str(text or ""))
        if not cleaned:
            return None

        candidates = [cleaned]
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            candidates.append(cleaned[first_brace : last_brace + 1])

        for candidate in candidates:
            try:
                parsed = yaml.safe_load(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                coerced = cls._coerce_assessment_mapping(parsed)
                if coerced is not None:
                    return coerced
        return None

    @staticmethod
    def _extract_passed_from_text(text: str) -> bool | None:
        lowered = str(text or "").lower()
        checks: list[tuple[str, bool]] = [
            (r"\bpassed?\s*[:=]\s*(true|yes|pass(?:ed)?|success(?:ful)?)\b", True),
            (r"\bpassed?\s*[:=]\s*(false|no|fail(?:ed)?|failure|unsuccessful)\b", False),
            (
                r"\b(verdict|result|outcome|assessment)\s*[:=]\s*"
                r"(pass(?:ed)?|success(?:ful)?)\b",
                True,
            ),
            (
                r"\b(verdict|result|outcome|assessment)\s*[:=]\s*"
                r"(fail(?:ed)?|failure|unsuccessful)\b",
                False,
            ),
            (r"\bsubtask\s+(passed|succeeded)\b", True),
            (r"\bsubtask\s+(failed|did not pass)\b", False),
            (r"\bacceptance criteria\s+(were|was)\s+met\b", True),
            (r"\bacceptance criteria\s+(were|was)\s+not met\b", False),
        ]
        for pattern, passed in checks:
            if re.search(pattern, lowered):
                return passed
        return None

    @classmethod
    def _extract_confidence_from_text(cls, text: str) -> float | None:
        lowered = str(text or "").lower()
        match = re.search(r"\bconfidence\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?%?)", lowered)
        if not match:
            return None
        return cls._to_confidence(match.group(1))

    @staticmethod
    def _extract_feedback_from_text(text: str) -> str | None:
        for label in ("feedback", "suggestion", "reason", "rationale"):
            match = re.search(
                rf"\b{label}\b\s*[:=]\s*(.+)",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                value = match.group(1).strip()
                if value:
                    return value
        return None

    @staticmethod
    def _extract_issues_from_text(text: str) -> list[str]:
        lines = str(text or "").splitlines()
        issues: list[str] = []
        capture = False
        for raw in lines:
            line = raw.strip()
            if not line:
                if capture:
                    break
                continue
            lowered = line.lower()
            if lowered.startswith("issues:"):
                capture = True
                trailing = line.split(":", 1)[1].strip()
                if trailing and trailing.lower() not in {"none", "no issues", "n/a"}:
                    issues.append(trailing)
                continue
            if capture:
                if line.startswith(("-", "*", "•")):
                    issues.append(line.lstrip("-*• ").strip())
                    continue
                if ":" in line and not line.startswith(("-", "*", "•")):
                    break
                issues.append(line.strip())
        return [item for item in issues if item]

    @classmethod
    def _coerce_assessment_from_text(cls, text: str) -> dict | None:
        return verification_parsing.coerce_assessment_from_text(
            text,
            valid_outcomes=_VALID_OUTCOMES,
            valid_severity_classes=_VALID_SEVERITY_CLASSES,
        )

    @staticmethod
    def _assessment_to_result(assessment: dict) -> VerificationResult:
        return verification_parsing.assessment_to_result(
            assessment,
            valid_outcomes=_VALID_OUTCOMES,
            valid_severity_classes=_VALID_SEVERITY_CLASSES,
        )

    def _inconclusive_result(self) -> VerificationResult:
        return VerificationResult(
            tier=2,
            passed=False,
            confidence=0.3,
            feedback="Verification inconclusive: could not parse verifier output.",
            outcome="fail",
            reason_code="parse_inconclusive",
            severity_class="inconclusive",
            metadata={
                "failure_class": "semantic_inconclusive",
                "remediation_mode": "remediate_and_retry",
                "missing_targets": [],
            },
        )

    def _parse_verifier_response(self, response) -> VerificationResult | None:
        return verification_parsing.parse_verifier_response(
            response=response,
            validator=self._validator,
            expected_keys=self._expected_verifier_response_keys(),
            valid_outcomes=_VALID_OUTCOMES,
            valid_severity_classes=_VALID_SEVERITY_CLASSES,
        )

    async def _repair_assessment_with_model(
        self,
        model,
        raw_text: str,
        *,
        task_id: str = "",
        subtask_id: str = "",
        origin: str = "verification.repair_assessment.complete",
    ) -> VerificationResult | None:
        expected_keys = self._expected_verifier_response_keys()
        repair_prompt = verification_prompting.build_repair_prompt(
            expected_keys=expected_keys,
            raw_text=raw_text,
            metadata_fields=verification_prompting.verifier_metadata_fields(
                self._prompts,
            ),
        )
        request_messages = [{"role": "user", "content": repair_prompt}]
        policy = ModelRetryPolicy()
        invocation_attempt = 0
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin=origin,
        )

        async def _invoke_model():
            nonlocal invocation_attempt
            invocation_attempt += 1
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="start",
                details={
                    "operation": "complete",
                    "invocation_attempt": invocation_attempt,
                    "invocation_max_attempts": policy.max_attempts,
                    **request_diag.to_event_payload(),
                },
            )
            return await model.complete(request_messages)

        def _on_failure(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
        ) -> None:
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="done",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": attempt,
                    "invocation_max_attempts": max_attempts,
                    "retry_queue_remaining": remaining,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

        def _on_retry_scheduled(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
            delay_seconds: float,
        ) -> None:
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="done",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": attempt,
                    "invocation_max_attempts": max_attempts,
                    "retry_queue_remaining": remaining,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    **build_model_retry_event_payload(
                        error,
                        delay_seconds=delay_seconds,
                    ),
                },
            )

        try:
            response = await call_with_model_retry(
                _invoke_model,
                policy=policy,
                on_failure=_on_failure,
                on_retry_scheduled=_on_retry_scheduled,
            )
        except Exception:
            return None
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=getattr(model, "name", "unknown"),
            phase="done",
            details={
                "operation": "complete",
                "origin": request_diag.origin,
                "invocation_attempt": invocation_attempt,
                "invocation_max_attempts": policy.max_attempts,
                **collect_response_diagnostics(response).to_event_payload(),
            },
        )
        return self._parse_verifier_response(response)

    async def _repair_assessment_with_alternate_model(
        self,
        raw_text: str,
        *,
        task_id: str = "",
        subtask_id: str = "",
        origin: str = "verification.repair_assessment.alternate_model.complete",
    ) -> VerificationResult | None:
        try:
            alternate = self._router.select(tier=2, role="verifier")
        except Exception:
            return None
        return await self._repair_assessment_with_model(
            alternate,
            raw_text,
            task_id=task_id,
            subtask_id=subtask_id,
            origin=origin,
        )

    async def _invoke_and_parse(
        self,
        model,
        prompt: str,
        *,
        task_id: str = "",
        subtask_id: str = "",
        origin: str = "verification.invoke_and_parse.complete",
    ) -> VerificationResult:
        request_messages = [{"role": "user", "content": prompt}]
        policy = ModelRetryPolicy()
        invocation_attempt = 0
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin=origin,
        )

        async def _invoke_model():
            nonlocal invocation_attempt
            invocation_attempt += 1
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="start",
                details={
                    "operation": "complete",
                    "invocation_attempt": invocation_attempt,
                    "invocation_max_attempts": policy.max_attempts,
                    **request_diag.to_event_payload(),
                },
            )
            return await model.complete(request_messages)

        def _on_failure(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
        ) -> None:
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="done",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": attempt,
                    "invocation_max_attempts": max_attempts,
                    "retry_queue_remaining": remaining,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

        def _on_retry_scheduled(
            attempt: int,
            max_attempts: int,
            error: BaseException,
            remaining: int,
            delay_seconds: float,
        ) -> None:
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask_id,
                model_name=getattr(model, "name", "unknown"),
                phase="done",
                details={
                    "operation": "complete",
                    "origin": request_diag.origin,
                    "invocation_attempt": attempt,
                    "invocation_max_attempts": max_attempts,
                    "retry_queue_remaining": remaining,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    **build_model_retry_event_payload(
                        error,
                        delay_seconds=delay_seconds,
                    ),
                },
            )

        response = await call_with_model_retry(
            _invoke_model,
            policy=policy,
            on_failure=_on_failure,
            on_retry_scheduled=_on_retry_scheduled,
        )
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask_id,
            model_name=getattr(model, "name", "unknown"),
            phase="done",
            details={
                "operation": "complete",
                "origin": request_diag.origin,
                "invocation_attempt": invocation_attempt,
                "invocation_max_attempts": policy.max_attempts,
                **collect_response_diagnostics(response).to_event_payload(),
            },
        )
        parsed = self._parse_verifier_response(response)
        if parsed is not None:
            return parsed

        if not bool(getattr(self._config, "strict_output_protocol", True)):
            return self._inconclusive_result()

        repaired = await self._repair_assessment_with_model(
            model,
            response.text or "",
            task_id=task_id,
            subtask_id=subtask_id,
            origin="verification.repair_assessment.same_model.complete",
        )
        if repaired is not None:
            repaired.metadata = dict(repaired.metadata)
            repaired.metadata["parser_stage"] = "repair_same_model"
            return repaired

        repaired_alt = await self._repair_assessment_with_alternate_model(
            response.text or "",
            task_id=task_id,
            subtask_id=subtask_id,
        )
        if repaired_alt is not None:
            repaired_alt.metadata = dict(repaired_alt.metadata)
            repaired_alt.metadata["parser_stage"] = "repair_alternate_model"
            return repaired_alt

        return self._inconclusive_result()

    @staticmethod
    def _exception_feedback(error: Exception) -> str:
        detail = str(error) or type(error).__name__
        detail = " ".join(detail.split())
        return f"Verification inconclusive: verifier raised an exception: {detail}"

    async def verify(
        self,
        subtask: Subtask,
        result_summary: str,
        tool_calls: list,
        workspace: Path | None,
        evidence_tool_calls: list | None = None,
        evidence_records: list[dict] | None = None,
        task_id: str = "",
    ) -> VerificationResult:
        compactor_context_token = _COMPACTOR_EVENT_CONTEXT.set((task_id, subtask.id))
        try:
            model = self._router.select(tier=1, role="verifier")
        except Exception as e:
            logger.warning("Verifier model not available: %s", e)
            _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
            return VerificationResult(
                tier=0,
                passed=True,
                confidence=0.5,
                feedback="Verification skipped: verifier model not configured",
                outcome="pass_with_warnings",
                reason_code="infra_verifier_error",
                severity_class="infra",
            )

        summary_for_prompt = await self._compact_text(
            result_summary,
            max_chars=self._max_result_summary_chars,
            label="verification result summary",
        )
        tool_calls_formatted = await self._format_tool_calls_for_prompt(tool_calls)
        derived_evidence = self._extract_evidence_from_tool_calls(
            subtask.id,
            evidence_tool_calls,
        )
        effective_evidence = self._merge_evidence_records(
            [item for item in (evidence_records or []) if isinstance(item, dict)],
            [item for item in derived_evidence if isinstance(item, dict)],
        )
        evidence_section = self._build_evidence_ledger_section(
            effective_evidence,
            workspace=workspace,
            tool_calls=tool_calls,
            max_chars=self._max_evidence_section_chars,
        )
        quality_contract = self._quality_contract_for_subtask(subtask)
        quality_enabled = bool(quality_contract.get("enabled", False))
        quality_diagnostics = self._artifact_quality_diagnostics(
            subtask=subtask,
            workspace=workspace,
            quality_contract=quality_contract,
        )
        artifact_max_chars = self._max_artifact_section_chars
        if quality_enabled:
            # Quality audits must see the complete expected artifact set whenever
            # it fits. Semantic compaction remains the overflow path; arbitrary
            # first-N-file clipping is not a valid quality assessment strategy.
            artifact_max_chars = max(
                artifact_max_chars,
                min(30_000, self._max_verifier_prompt_tokens * 3),
            )
        artifact_section = await self._build_artifact_content_section(
            subtask=subtask,
            workspace=workspace,
            tool_calls=tool_calls,
            max_chars=artifact_max_chars,
            quality_contract=quality_contract,
        )
        quality_diagnostics_section = ""
        if quality_enabled:
            quality_diagnostics_section = (
                "ARTIFACT QUALITY DIAGNOSTICS (deterministic inventory; use alongside "
                "semantic review):\n"
                + json.dumps(quality_diagnostics, ensure_ascii=True, sort_keys=True)
            )
        phase_scope_default = self._phase_scope_default()
        selected_rules = self._select_phase_scoped_rules(
            subtask,
            task_id=task_id,
        )
        prompt = self._build_prompt(
            subtask,
            summary_for_prompt,
            tool_calls_formatted,
            llm_rules=selected_rules,
            phase_scope_default=phase_scope_default,
        )
        prompt = self._augment_verifier_prompt(
            prompt,
            subtask=subtask,
            quality_contract=quality_contract,
        )
        if evidence_section:
            prompt = prompt + "\n\n" + evidence_section
        if artifact_section:
            prompt = prompt + "\n\n" + artifact_section
        if quality_diagnostics_section:
            prompt = prompt + "\n\n" + quality_diagnostics_section
        if estimate_tokens(prompt) > self._max_verifier_prompt_tokens:
            summary_for_prompt = await self._compact_text(
                summary_for_prompt,
                max_chars=self._compact_result_summary_chars,
                label="verification compact summary",
            )
            tool_calls_formatted = await self._format_tool_calls_compact(tool_calls)
            prompt = self._build_prompt(
                subtask,
                summary_for_prompt,
                tool_calls_formatted,
                llm_rules=selected_rules,
                phase_scope_default=phase_scope_default,
            )
            prompt = self._augment_verifier_prompt(
                prompt,
                subtask=subtask,
                quality_contract=quality_contract,
            )
            if evidence_section:
                prompt = (
                    prompt
                    + "\n\n"
                    + self._hard_cap_text(
                        evidence_section,
                        self._max_evidence_section_compact_chars,
                    )
                )
            if artifact_section:
                compact_artifact_chars = self._max_artifact_section_compact_chars
                if quality_enabled:
                    compact_artifact_chars = max(compact_artifact_chars, 12_000)
                prompt = (
                    prompt
                    + "\n\n"
                    + self._hard_cap_text(
                        artifact_section,
                        compact_artifact_chars,
                    )
                )
            if quality_diagnostics_section:
                prompt = prompt + "\n\n" + quality_diagnostics_section

        request_messages = [{"role": "user", "content": prompt}]
        request_diag = collect_request_diagnostics(
            messages=request_messages,
            origin="verification.tier2.complete",
        )
        self._emit_model_event(
            task_id=task_id,
            subtask_id=subtask.id,
            model_name=model.name,
            phase="start",
            details={
                "operation": "complete",
                "attempt": 1,
                "evidence_record_count": len(effective_evidence),
                **request_diag.to_event_payload(),
            },
        )
        try:
            first_result = await self._invoke_and_parse(
                model,
                prompt,
                task_id=task_id,
                subtask_id=subtask.id,
                origin="verification.tier2.invoke.complete",
            )
            if not first_result.reason_code and not first_result.passed:
                first_result.reason_code = "llm_semantic_failed"
            first_result = self._apply_quality_contract(
                result=first_result,
                quality_contract=quality_contract,
                diagnostics=quality_diagnostics,
            )
            if first_result.reason_code == "quality_assessment_missing":
                audit_retry_prompt = (
                    f"{prompt}\n\n"
                    "VERIFIER SELF-CORRECTION: Your previous response omitted part of "
                    "the required metadata.quality audit. Reassess the same supplied "
                    "artifacts without requesting new research. Return the complete "
                    "verifier JSON contract, including every required quality dimension, "
                    "overall, requirement_coverage, and concrete missing_targets."
                )
                try:
                    repaired_result = await self._invoke_and_parse(
                        model,
                        audit_retry_prompt,
                        task_id=task_id,
                        subtask_id=subtask.id,
                        origin="verification.tier2.quality_contract_retry.complete",
                    )
                    if not repaired_result.reason_code and not repaired_result.passed:
                        repaired_result.reason_code = "llm_semantic_failed"
                    first_result = self._apply_quality_contract(
                        result=repaired_result,
                        quality_contract=quality_contract,
                        diagnostics=quality_diagnostics,
                    )
                    first_result.metadata = dict(first_result.metadata)
                    first_result.metadata["quality_contract_retry"] = True
                except Exception as quality_retry_error:
                    logger.warning(
                        "Verifier quality-contract self-correction failed: %s",
                        quality_retry_error,
                    )
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask.id,
                model_name=model.name,
                phase="done",
                details={
                    "operation": "complete",
                    "attempt": 1,
                    "origin": request_diag.origin,
                    "verifier_passed": first_result.passed,
                    "verifier_confidence": first_result.confidence,
                    "verifier_outcome": first_result.outcome,
                    "reason_code": first_result.reason_code,
                },
            )
            _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
            return first_result
        except Exception as e:
            logger.warning("Verifier raised exception: %s", e)
            compact_tool_calls = await self._format_tool_calls_compact(tool_calls)
            compact_prompt = self._build_prompt(
                subtask,
                await self._compact_text(
                    summary_for_prompt,
                    max_chars=self._compact_result_summary_chars,
                    label="verification retry summary",
                ),
                compact_tool_calls,
                llm_rules=selected_rules,
                phase_scope_default=phase_scope_default,
            )
            compact_prompt = self._augment_verifier_prompt(
                compact_prompt,
                subtask=subtask,
                quality_contract=quality_contract,
            )
            if evidence_section:
                compact_prompt = (
                    compact_prompt
                    + "\n\n"
                    + self._hard_cap_text(
                        evidence_section,
                        self._max_evidence_section_compact_chars,
                    )
                )
            if artifact_section:
                compact_artifact_chars = self._max_artifact_section_compact_chars
                if quality_enabled:
                    compact_artifact_chars = max(compact_artifact_chars, 12_000)
                compact_prompt = (
                    compact_prompt
                    + "\n\n"
                    + self._hard_cap_text(
                        artifact_section,
                        compact_artifact_chars,
                    )
                )
            if quality_diagnostics_section:
                compact_prompt = compact_prompt + "\n\n" + quality_diagnostics_section
            retry_messages = [{"role": "user", "content": compact_prompt}]
            retry_diag = collect_request_diagnostics(
                messages=retry_messages,
                origin="verification.tier2.retry.complete",
            )
            self._emit_model_event(
                task_id=task_id,
                subtask_id=subtask.id,
                model_name=model.name,
                phase="start",
                details={
                    "operation": "complete",
                    "attempt": 2,
                    "evidence_record_count": len(effective_evidence),
                    **retry_diag.to_event_payload(),
                },
            )
            try:
                retry_result = await self._invoke_and_parse(
                    model,
                    compact_prompt,
                    task_id=task_id,
                    subtask_id=subtask.id,
                    origin="verification.tier2.retry.invoke.complete",
                )
                if not retry_result.reason_code and not retry_result.passed:
                    retry_result.reason_code = "llm_semantic_failed"
                retry_result = self._apply_quality_contract(
                    result=retry_result,
                    quality_contract=quality_contract,
                    diagnostics=quality_diagnostics,
                )
                self._emit_model_event(
                    task_id=task_id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "operation": "complete",
                        "attempt": 2,
                        "origin": retry_diag.origin,
                        "verifier_passed": retry_result.passed,
                        "verifier_confidence": retry_result.confidence,
                        "verifier_outcome": retry_result.outcome,
                        "reason_code": retry_result.reason_code,
                    },
                )
                _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
                return retry_result
            except Exception as retry_error:
                logger.warning(
                    "Verifier compact retry raised exception: %s",
                    retry_error,
                )
                self._emit_model_event(
                    task_id=task_id,
                    subtask_id=subtask.id,
                    model_name=model.name,
                    phase="done",
                    details={
                        "operation": "complete",
                        "attempt": 2,
                        "origin": retry_diag.origin,
                        "error": str(retry_error),
                    },
                )
                _COMPACTOR_EVENT_CONTEXT.reset(compactor_context_token)
                return VerificationResult(
                    tier=2,
                    passed=False,
                    confidence=0.3,
                    feedback=self._exception_feedback(retry_error),
                    outcome="fail",
                    reason_code="infra_verifier_error",
                    severity_class="infra",
                    metadata={
                        "failure_class": "infra_verifier_error",
                        "remediation_mode": "remediate_and_retry",
                        "missing_targets": [],
                    },
                )
