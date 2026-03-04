"""Background cowork memory indexing for long-horizon recall."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loom.cowork.session_state import SessionState
    from loom.models.base import ModelProvider
    from loom.state.conversation_store import ConversationStore

logger = logging.getLogger(__name__)

_MARKER_PATTERN = re.compile(
    r"(?im)^\s*(DECISION|PROPOSAL|RESEARCH|RATIONALE|OPEN[\s_-]?QUESTION|"
    r"CONSTRAINT|RISK|ACTION[\s_-]?ITEM)\s*:\s*(.+)$",
)
_MARKER_TYPE_MAP = {
    "decision": "decision",
    "proposal": "proposal",
    "research": "research",
    "rationale": "rationale",
    "openquestion": "open_question",
    "constraint": "constraint",
    "risk": "risk",
    "actionitem": "action_item",
}
_DEFAULT_SOURCE_ROLES = ("user", "assistant", "tool")


@dataclass
class MemoryEntryCandidate:
    """Normalized cowork memory entry candidate."""

    entry_type: str
    status: str
    summary: str
    rationale: str = ""
    topic: str = ""
    tags: list[str] | None = None
    entities: list[str] | None = None
    source_turn_start: int = 0
    source_turn_end: int = 0
    source_roles: list[str] | None = None
    evidence_excerpt: str = ""
    confidence: float = 0.0
    supersedes_entry_id: int | None = None
    fingerprint: str = ""


class CoworkMemoryIndexer:
    """Incremental background indexer for cowork conversation turns."""

    INDEX_VERSION = 1
    MAX_SUMMARY_CHARS = 220
    MAX_EVIDENCE_CHARS = 260
    MAX_LLM_ENTRIES_PER_BATCH = 64

    def __init__(
        self,
        *,
        store: ConversationStore,
        session_id: str,
        session_state: SessionState,
        model: ModelProvider | None = None,
        model_role: str = "",
        llm_extraction_enabled: bool = False,
        role_strict: bool = False,
        queue_max_batches: int = 32,
        section_limit: int = 4,
    ) -> None:
        self._store = store
        self._session_id = str(session_id or "").strip()
        self._state = session_state
        self._model = model
        self._model_role = str(model_role or "").strip().lower()
        self._llm_extraction_enabled = bool(llm_extraction_enabled)
        self._role_strict = bool(role_strict)
        self._queue_max_batches = max(1, int(queue_max_batches))
        self._section_limit = max(1, int(section_limit))

        self._pending_targets: list[int] = []
        self._queue_event = asyncio.Event()
        self._worker_task: asyncio.Task | None = None
        self._closed = False
        self._backpressure_degraded = False
        self._active = False
        self._processing = False

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        if self._closed:
            return
        if self._worker_task is not None and self._worker_task.done():
            self._worker_task = None
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._active = True

    async def close(self) -> None:
        self._closed = True
        self._queue_event.set()
        task = self._worker_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        self._worker_task = None
        self._active = False

    async def wait_idle(self, *, timeout_seconds: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.1, float(timeout_seconds))
        while time.monotonic() < deadline:
            if (
                not self._pending_targets
                and not self._processing
                and (
                    self._worker_task is None
                    or self._worker_task.done()
                )
            ):
                return True
            await asyncio.sleep(0.05)
        return False

    def enqueue_up_to_turn(self, turn_number: int) -> None:
        value = max(0, int(turn_number or 0))
        if value <= 0 or self._closed:
            return
        self.start()
        self._pending_targets.append(value)
        # Coalesce under pressure and mark index degraded until caught up.
        if len(self._pending_targets) > self._queue_max_batches:
            latest = max(self._pending_targets)
            self._pending_targets = [latest]
            self._backpressure_degraded = True
        self._queue_event.set()

    async def reindex_up_to_turn(self, turn_number: int) -> None:
        """Force an index pass up to the supplied turn."""
        target = max(0, int(turn_number or 0))
        if target <= 0:
            return
        self.enqueue_up_to_turn(target)
        await self.wait_idle(timeout_seconds=10.0)

    async def _worker_loop(self) -> None:
        try:
            while True:
                await self._queue_event.wait()
                self._queue_event.clear()
                if self._closed and not self._pending_targets:
                    break

                while self._pending_targets:
                    # Coalesce bursts: process only the furthest turn.
                    target = max(self._pending_targets)
                    self._pending_targets = []
                    if target <= 0:
                        continue
                    try:
                        self._processing = True
                        await self._index_to_turn(target)
                    except Exception as e:  # pragma: no cover - defensive
                        await self._record_failure(target_turn=target, error=e)
                    finally:
                        self._processing = False

                if self._closed:
                    break
                if self._pending_targets:
                    continue
                # Exit when drained to avoid idle background task leaks.
                if self._queue_event.is_set():
                    continue
                break
        finally:
            restart = (not self._closed) and bool(self._pending_targets)
            self._active = False
            self._worker_task = None
            if restart:
                self.start()
                self._queue_event.set()

    async def _index_to_turn(self, target_turn: int) -> None:
        started = time.monotonic()
        state = await self._store.get_cowork_memory_index_state(self._session_id)
        from_turn = int(state.get("last_indexed_turn", 0) or 0)
        if target_turn <= from_turn:
            return
        logger.info(
            "cowork_memory_index_started session=%s from_turn=%s to_turn=%s",
            self._session_id,
            from_turn + 1,
            target_turn,
        )

        rows = await self._store.get_turn_range(
            self._session_id,
            from_turn + 1,
            target_turn,
        )
        if not rows:
            await self._store.upsert_cowork_memory_index_state(
                self._session_id,
                last_indexed_turn=target_turn,
                index_degraded=bool(self._backpressure_degraded),
                failure_count=int(state.get("failure_count", 0) or 0),
                index_version=self.INDEX_VERSION,
            )
            self._apply_index_meta(
                last_indexed_turn=target_turn,
                degraded=bool(self._backpressure_degraded),
                failure_count=int(state.get("failure_count", 0) or 0),
                last_error="",
            )
            return

        deterministic = self._extract_entries_from_turns(rows)
        llm_entries: list[MemoryEntryCandidate] = []
        if not self._backpressure_degraded:
            llm_entries = await self._extract_entries_with_model(rows)
        merged = self._merge_candidates(deterministic, llm_entries)

        inserted = 0
        if merged:
            inserted = await self._store.upsert_cowork_memory_entries(
                self._session_id,
                [self._candidate_to_payload(item) for item in merged],
            )

        snapshot = await self._store.get_cowork_memory_active_snapshot(
            self._session_id,
            max_decisions=self._section_limit,
            max_proposals=self._section_limit,
            max_research=self._section_limit,
            max_questions=self._section_limit,
        )
        self._apply_snapshot(snapshot)

        max_turn = max(
            int(row.get("turn_number", 0) or 0)
            for row in rows
        )
        await self._store.upsert_cowork_memory_index_state(
            self._session_id,
            last_indexed_turn=max(max_turn, target_turn),
            index_degraded=bool(self._backpressure_degraded),
            last_error="",
            failure_count=0,
            index_version=self.INDEX_VERSION,
        )
        self._apply_index_meta(
            last_indexed_turn=max(max_turn, target_turn),
            degraded=bool(self._backpressure_degraded),
            failure_count=0,
            last_error="",
        )
        self._backpressure_degraded = False
        elapsed_ms = max(1, int((time.monotonic() - started) * 1000))
        logger.info(
            (
                "cowork_memory_index_completed session=%s from_turn=%s to_turn=%s "
                "rows=%s entries=%s model_role=%s strict=%s elapsed_ms=%s"
            ),
            self._session_id,
            from_turn + 1,
            target_turn,
            len(rows),
            inserted,
            self._model_role or "deterministic_only",
            self._role_strict,
            elapsed_ms,
        )

    async def _record_failure(self, *, target_turn: int, error: Exception) -> None:
        error_text = self._truncate(str(error or ""), 320)
        state = await self._store.get_cowork_memory_index_state(self._session_id)
        failure_count = int(state.get("failure_count", 0) or 0) + 1
        last_indexed_turn = int(state.get("last_indexed_turn", 0) or 0)
        self._backpressure_degraded = True
        await self._store.upsert_cowork_memory_index_state(
            self._session_id,
            last_indexed_turn=last_indexed_turn,
            index_degraded=True,
            last_error=error_text,
            failure_count=failure_count,
            index_version=self.INDEX_VERSION,
        )
        self._apply_index_meta(
            last_indexed_turn=last_indexed_turn,
            degraded=True,
            failure_count=failure_count,
            last_error=error_text,
        )
        logger.warning(
            "cowork_memory_index_failed session=%s target_turn=%s error=%s",
            self._session_id,
            target_turn,
            error_text,
        )

    async def _extract_entries_with_model(
        self,
        rows: list[dict],
    ) -> list[MemoryEntryCandidate]:
        if not self._llm_extraction_enabled:
            return []
        if self._model is None:
            return []
        if self._role_strict and self._model_role not in {"compactor", "extractor"}:
            return []

        rendered = []
        for row in rows:
            role = str(row.get("role", "") or "").strip().lower()
            if role not in _DEFAULT_SOURCE_ROLES:
                continue
            turn = int(row.get("turn_number", 0) or 0)
            content = str(row.get("content", "") or "").strip()
            if not content:
                continue
            snippet = self._truncate(content, self.MAX_EVIDENCE_CHARS)
            rendered.append(f"[turn {turn}][{role}] {snippet}")
        if not rendered:
            return []

        prompt = (
            "Extract typed cowork memory entries from conversation turns.\n"
            "Return JSON only with shape:\n"
            '{"entries":[{"entry_type":"decision|proposal|research|rationale|constraint|risk|open_question|action_item",'
            '"summary":"...",'
            '"rationale":"...",'
            '"topic":"...",'
            '"source_turn_start":0,'
            '"source_turn_end":0,'
            '"evidence_excerpt":"...",'
            '"confidence":0.0}]}\n'
            "Rules:\n"
            "- keep only materially useful entries\n"
            "- summaries must be concise\n"
            "- source turns must be within provided turns\n"
            "- no markdown fences\n\n"
            f"Turns:\n{chr(10).join(rendered)}"
        )
        try:
            response = await asyncio.wait_for(
                self._model.complete([{"role": "user", "content": prompt}], temperature=0),
                timeout=15.0,
            )
        except Exception as e:
            logger.debug(
                "cowork_memory_index_llm_extract_failed session=%s error=%s",
                self._session_id,
                e,
            )
            return []

        text = str(getattr(response, "text", "") or "").strip()
        payload = self._parse_json_payload(text)
        if payload is None:
            payload = await self._repair_json_payload(text)
        if not isinstance(payload, dict):
            return []
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            return []
        lo = min(int(row.get("turn_number", 0) or 0) for row in rows)
        hi = max(int(row.get("turn_number", 0) or 0) for row in rows)
        extracted: list[MemoryEntryCandidate] = []
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            entry_type = self._normalize_entry_type(item.get("entry_type"))
            summary = self._truncate(
                str(item.get("summary", "") or "").strip(),
                self.MAX_SUMMARY_CHARS,
            )
            if not summary:
                continue
            start = max(lo, min(hi, self._safe_int(item.get("source_turn_start"), lo)))
            end = max(start, min(hi, self._safe_int(item.get("source_turn_end"), start)))
            confidence = max(0.0, min(1.0, self._safe_float(item.get("confidence"), 0.5)))
            candidate = MemoryEntryCandidate(
                entry_type=entry_type,
                status="active",
                summary=summary,
                rationale=self._truncate(
                    str(item.get("rationale", "") or "").strip(),
                    self.MAX_EVIDENCE_CHARS,
                ),
                topic=self._normalize_topic(str(item.get("topic", "") or summary)),
                source_turn_start=start,
                source_turn_end=end,
                source_roles=[],
                evidence_excerpt=self._truncate(
                    str(item.get("evidence_excerpt", "") or summary),
                    self.MAX_EVIDENCE_CHARS,
                ),
                confidence=confidence,
            )
            candidate.fingerprint = self._fingerprint(candidate)
            extracted.append(candidate)
            if len(extracted) >= self.MAX_LLM_ENTRIES_PER_BATCH:
                break
        return extracted

    async def _repair_json_payload(self, malformed: str) -> dict | None:
        if self._model is None:
            return None
        cleaned = str(malformed or "").strip()
        if not cleaned:
            return None
        prompt = (
            "Repair this malformed JSON output and return JSON only.\n"
            "Required shape:\n"
            '{"entries":[{"entry_type":"decision|proposal|research|rationale|constraint|risk|open_question|action_item",'
            '"summary":"...",'
            '"rationale":"...",'
            '"topic":"...",'
            '"source_turn_start":0,'
            '"source_turn_end":0,'
            '"evidence_excerpt":"...",'
            '"confidence":0.0}]}\n'
            "Malformed:\n"
            f"{cleaned}"
        )
        try:
            response = await asyncio.wait_for(
                self._model.complete([{"role": "user", "content": prompt}], temperature=0),
                timeout=10.0,
            )
        except Exception:
            return None
        repaired = str(getattr(response, "text", "") or "").strip()
        return self._parse_json_payload(repaired)

    def _extract_entries_from_turns(self, rows: list[dict]) -> list[MemoryEntryCandidate]:
        entries: list[MemoryEntryCandidate] = []
        for row in rows:
            role = str(row.get("role", "") or "").strip().lower()
            if role not in _DEFAULT_SOURCE_ROLES:
                continue
            content = str(row.get("content", "") or "")
            if not content.strip():
                continue
            turn = max(0, int(row.get("turn_number", 0) or 0))

            marker_hits = list(_MARKER_PATTERN.finditer(content))
            if marker_hits:
                for match in marker_hits:
                    raw_marker = re.sub(r"[\s_-]+", "", str(match.group(1) or "").lower())
                    entry_type = _MARKER_TYPE_MAP.get(raw_marker, "research")
                    summary = self._truncate(
                        str(match.group(2) or "").strip(),
                        self.MAX_SUMMARY_CHARS,
                    )
                    if not summary:
                        continue
                    candidate = MemoryEntryCandidate(
                        entry_type=entry_type,
                        status="active",
                        summary=summary,
                        rationale="",
                        topic=self._normalize_topic(summary),
                        source_turn_start=turn,
                        source_turn_end=turn,
                        source_roles=[role],
                        evidence_excerpt=self._truncate(content, self.MAX_EVIDENCE_CHARS),
                        confidence=0.92,
                    )
                    candidate.fingerprint = self._fingerprint(candidate)
                    entries.append(candidate)
                continue

            heuristic = self._heuristic_candidate(role=role, turn=turn, content=content)
            if heuristic is not None:
                heuristic.fingerprint = self._fingerprint(heuristic)
                entries.append(heuristic)
        return entries

    def _heuristic_candidate(
        self,
        *,
        role: str,
        turn: int,
        content: str,
    ) -> MemoryEntryCandidate | None:
        normalized = " ".join(content.split())
        lowered = normalized.lower()
        if not normalized:
            return None

        if "?" in normalized and (
            "should we" in lowered
            or "can we" in lowered
            or "do we" in lowered
            or role == "user"
        ):
            return MemoryEntryCandidate(
                entry_type="open_question",
                status="active",
                summary=self._truncate(normalized, self.MAX_SUMMARY_CHARS),
                topic=self._normalize_topic(normalized),
                source_turn_start=turn,
                source_turn_end=turn,
                source_roles=[role],
                evidence_excerpt=self._truncate(normalized, self.MAX_EVIDENCE_CHARS),
                confidence=0.55,
            )

        if role == "assistant" and (
            "i recommend" in lowered
            or "we should" in lowered
            or lowered.startswith("let's ")
        ):
            return MemoryEntryCandidate(
                entry_type="proposal",
                status="active",
                summary=self._truncate(normalized, self.MAX_SUMMARY_CHARS),
                topic=self._normalize_topic(normalized),
                source_turn_start=turn,
                source_turn_end=turn,
                source_roles=[role],
                evidence_excerpt=self._truncate(normalized, self.MAX_EVIDENCE_CHARS),
                confidence=0.50,
            )

        if role == "tool":
            return MemoryEntryCandidate(
                entry_type="research",
                status="active",
                summary=self._truncate(
                    f"Tool evidence captured at turn {turn}.",
                    self.MAX_SUMMARY_CHARS,
                ),
                topic="tool-evidence",
                source_turn_start=turn,
                source_turn_end=turn,
                source_roles=[role],
                evidence_excerpt=self._truncate(normalized, self.MAX_EVIDENCE_CHARS),
                confidence=0.35,
            )
        return None

    def _merge_candidates(
        self,
        deterministic: list[MemoryEntryCandidate],
        llm_entries: list[MemoryEntryCandidate],
    ) -> list[MemoryEntryCandidate]:
        merged: dict[str, MemoryEntryCandidate] = {}
        # deterministic entries first; llm entries can override only on higher confidence
        for entry in deterministic:
            if not entry.fingerprint:
                entry.fingerprint = self._fingerprint(entry)
            merged[entry.fingerprint] = entry
        for entry in llm_entries:
            if not entry.fingerprint:
                entry.fingerprint = self._fingerprint(entry)
            existing = merged.get(entry.fingerprint)
            if existing is None or entry.confidence >= existing.confidence:
                merged[entry.fingerprint] = entry
        return list(merged.values())

    def _apply_snapshot(self, snapshot: dict[str, list[dict]]) -> None:
        if hasattr(self._state, "update_memory_snapshot"):
            try:
                self._state.update_memory_snapshot(snapshot)
                return
            except Exception:
                pass

        for field_name in (
            "active_decisions",
            "active_proposals",
            "recent_research",
            "open_questions",
        ):
            setattr(self._state, field_name, list(snapshot.get(field_name, [])))

    def _apply_index_meta(
        self,
        *,
        last_indexed_turn: int,
        degraded: bool,
        failure_count: int,
        last_error: str = "",
    ) -> None:
        if hasattr(self._state, "update_memory_index_meta"):
            try:
                self._state.update_memory_index_meta(
                    last_indexed_turn=last_indexed_turn,
                    degraded=degraded,
                    failure_count=failure_count,
                    last_error=last_error,
                )
                return
            except Exception:
                pass
        setattr(self._state, "memory_index_last_indexed_turn", max(0, int(last_indexed_turn)))
        setattr(self._state, "memory_index_degraded", bool(degraded))
        setattr(self._state, "memory_index_failure_count", max(0, int(failure_count)))
        setattr(self._state, "memory_index_last_error", self._truncate(last_error, 320))

    def _candidate_to_payload(self, entry: MemoryEntryCandidate) -> dict[str, Any]:
        return {
            "entry_type": self._normalize_entry_type(entry.entry_type),
            "status": str(entry.status or "active").strip().lower(),
            "summary": self._truncate(entry.summary, self.MAX_SUMMARY_CHARS),
            "rationale": self._truncate(entry.rationale, self.MAX_EVIDENCE_CHARS),
            "topic": self._normalize_topic(entry.topic or entry.summary),
            "tags": list(entry.tags or []),
            "entities": list(entry.entities or []),
            "source_turn_start": max(0, int(entry.source_turn_start or 0)),
            "source_turn_end": max(
                max(0, int(entry.source_turn_start or 0)),
                int(entry.source_turn_end or entry.source_turn_start or 0),
            ),
            "source_roles": list(entry.source_roles or []),
            "evidence_excerpt": self._truncate(entry.evidence_excerpt, self.MAX_EVIDENCE_CHARS),
            "supersedes_entry_id": entry.supersedes_entry_id,
            "confidence": max(0.0, min(1.0, float(entry.confidence or 0.0))),
            "fingerprint": entry.fingerprint or self._fingerprint(entry),
        }

    def _fingerprint(self, entry: MemoryEntryCandidate) -> str:
        base = "|".join(
            [
                self._session_id,
                self._normalize_entry_type(entry.entry_type),
                self._normalize_topic(entry.topic or entry.summary),
                self._truncate(entry.summary, self.MAX_SUMMARY_CHARS),
                str(max(0, int(entry.source_turn_start or 0))),
                str(max(0, int(entry.source_turn_end or entry.source_turn_start or 0))),
            ]
        )
        return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        value = " ".join(str(text or "").split())
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _normalize_topic(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9\s_-]+", " ", str(text or ""))
        tokens = [token for token in cleaned.lower().split() if token]
        if not tokens:
            return ""
        return "-".join(tokens[:6])

    @staticmethod
    def _normalize_entry_type(value: object) -> str:
        text = str(value or "").strip().lower()
        if text in {
            "decision",
            "proposal",
            "research",
            "rationale",
            "constraint",
            "risk",
            "open_question",
            "action_item",
        }:
            return text
        return "research"

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _parse_json_payload(cls, text: str) -> dict | None:
        value = str(text or "").strip()
        if not value:
            return None

        def _parse_blob(blob: str) -> dict | None:
            candidate = blob.strip()
            if candidate.startswith("```"):
                lines = candidate.splitlines()
                if lines:
                    lines = lines[1:]
                while lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                candidate = "\n".join(lines).strip()
            if not candidate:
                return None
            try:
                parsed = json.loads(candidate)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None

        parsed = _parse_blob(value)
        if parsed is not None:
            return parsed

        match = re.search(r"\{[\s\S]*\}", value)
        if not match:
            return None
        return _parse_blob(match.group(0))
