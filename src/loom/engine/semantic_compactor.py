"""Semantic text compaction for model-facing context windows.

This module avoids naive character slicing by using an LLM to compress text
while preserving concrete details (facts, identifiers, filenames, paths,
commands, URLs, errors, and numeric values).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from loom.models.base import ModelProvider
from loom.models.router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CacheKey:
    digest: str
    max_chars: int
    label: str


class SemanticCompactor:
    """LLM-backed semantic compactor with chunked map/reduce behavior."""

    _MAX_CHUNK_CHARS = 9_000
    _MAX_CHUNKS_PER_ROUND = 12
    _MAX_REDUCTION_ROUNDS = 4
    _MIN_COMPACT_TARGET_CHARS = 140

    def __init__(
        self,
        model_router: ModelRouter | None = None,
        *,
        model: ModelProvider | None = None,
        role: str = "extractor",
        tier: int = 1,
    ):
        self._router = model_router
        self._model = model
        self._role = role
        self._tier = tier
        self._cache: dict[_CacheKey, str] = {}

    async def compact(self, text: str, *, max_chars: int, label: str = "context") -> str:
        """Semantically compact text to approximately ``max_chars`` characters.

        This never performs hard character slicing. If no compactor model is
        available, it falls back to whitespace-only compaction (lossless).
        """
        value = str(text or "").strip()
        if max_chars <= 0:
            return ""
        if not value or len(value) <= max_chars:
            return value

        key = _CacheKey(
            digest=hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest(),
            max_chars=max_chars,
            label=label,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        model = self._select_model()
        if model is None:
            compacted = self._normalize_whitespace(value)
            self._cache[key] = compacted
            return compacted

        try:
            compacted = await self._compact_with_model(
                model=model,
                text=value,
                max_chars=max_chars,
                label=label,
            )
        except Exception as exc:
            logger.warning("Semantic compaction failed for %s: %s", label, exc)
            compacted = self._normalize_whitespace(value)

        self._cache[key] = compacted
        return compacted

    def _select_model(self) -> ModelProvider | None:
        """Resolve the configured compactor model, if available."""
        if self._model is not None:
            return self._model

        if self._router is None:
            return None

        candidate_roles = (self._role, "verifier", "executor")
        for role in candidate_roles:
            try:
                model = self._router.select(tier=self._tier, role=role)
            except Exception:
                continue

            roles = getattr(model, "roles", None)
            if isinstance(roles, list) and role not in roles:
                continue
            return model
        return None

    async def _compact_with_model(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
    ) -> str:
        current = text
        if len(current) > self._MAX_CHUNK_CHARS:
            current = await self._chunked_compaction(
                model=model,
                text=current,
                max_chars=max_chars,
                label=label,
            )

        rounds = 0
        while len(current) > max_chars and rounds < self._MAX_REDUCTION_ROUNDS:
            target = max(
                max_chars,
                int(len(current) * 0.72),
                self._MIN_COMPACT_TARGET_CHARS,
            )
            # Force the final rounds to aim directly at the requested budget.
            if rounds >= 2:
                target = max_chars
            candidate = await self._compact_once(
                model=model,
                text=current,
                max_chars=target,
                label=label,
                strict=(rounds >= 1),
            )
            if not candidate:
                break
            if len(candidate) >= len(current):
                # Retry with stricter instructions if we did not shrink.
                strict_candidate = await self._compact_once(
                    model=model,
                    text=current,
                    max_chars=max_chars,
                    label=label,
                    strict=True,
                )
                if strict_candidate and len(strict_candidate) < len(current):
                    candidate = strict_candidate
            if len(candidate) >= len(current):
                break
            current = candidate
            rounds += 1

        return current.strip() or text

    async def _chunked_compaction(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
    ) -> str:
        chunks = self._split_text(text, self._MAX_CHUNK_CHARS)
        if not chunks:
            return text

        # Hierarchical pre-merge when there are too many chunks.
        while len(chunks) > self._MAX_CHUNKS_PER_ROUND:
            merged: list[str] = []
            for idx in range(0, len(chunks), 3):
                block = "\n\n".join(chunks[idx:idx + 3])
                merged_text = await self._compact_once(
                    model=model,
                    text=block,
                    max_chars=max(max_chars * 3, 1_200),
                    label=f"{label} pre-merge",
                    strict=False,
                )
                merged.append(merged_text or block)
            chunks = merged

        map_target = max(
            self._MIN_COMPACT_TARGET_CHARS,
            min(max(max_chars * 2, 600), 2_400),
        )
        mapped: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            if len(chunk) <= map_target:
                mapped.append(chunk)
                continue
            mapped_text = await self._compact_once(
                model=model,
                text=chunk,
                max_chars=map_target,
                label=f"{label} chunk {idx}/{len(chunks)}",
                strict=False,
            )
            mapped.append(mapped_text or chunk)

        return "\n".join(part.strip() for part in mapped if part.strip()).strip()

    async def _compact_once(
        self,
        *,
        model: ModelProvider,
        text: str,
        max_chars: int,
        label: str,
        strict: bool,
    ) -> str:
        target = max(self._MIN_COMPACT_TARGET_CHARS, int(max_chars))
        strict_rule = (
            "You MUST keep the response at or below the character target."
            if strict
            else "Aim to stay at or below the character target."
        )
        system = (
            "You are a semantic context compactor for LLM pipelines. "
            "Preserve concrete details and required instructions while removing "
            "redundancy. Never invent facts."
        )
        user = (
            f"Compact the following {label}.\n"
            f"Character target: <= {target}.\n"
            f"{strict_rule}\n\n"
            "Preservation requirements:\n"
            "- Keep all facts, numbers, dates, file paths, URLs, IDs, commands, "
            "errors, acceptance criteria, and deliverable names.\n"
            "- Keep dependency/ordering constraints.\n"
            "- Keep meaning and intent intact.\n"
            "- Remove repetition and filler.\n"
            "- Return compact plain text only.\n\n"
            f"Source text:\n{text}"
        )
        response = await model.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max(256, target // 2 + 128),
        )
        return str(response.text or "").strip()

    @staticmethod
    def _split_text(text: str, chunk_chars: int) -> list[str]:
        chunks: list[str] = []
        remaining = text.strip()
        while remaining:
            if len(remaining) <= chunk_chars:
                chunks.append(remaining)
                break

            split_at = remaining.rfind("\n\n", 0, chunk_chars)
            if split_at < int(chunk_chars * 0.55):
                split_at = remaining.rfind("\n", 0, chunk_chars)
            if split_at < int(chunk_chars * 0.55):
                split_at = remaining.rfind(". ", 0, chunk_chars)
            if split_at < int(chunk_chars * 0.55):
                split_at = chunk_chars

            chunk = remaining[:split_at].strip()
            if not chunk:
                break
            chunks.append(chunk)
            remaining = remaining[split_at:].strip()
        return chunks

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Lossless fallback compaction: normalize repeated whitespace."""
        if not text:
            return text
        lines = [line.rstrip() for line in text.splitlines()]
        collapsed: list[str] = []
        blank = False
        for line in lines:
            if line:
                collapsed.append(" ".join(line.split()))
                blank = False
            elif not blank:
                collapsed.append("")
                blank = True
        return "\n".join(collapsed).strip()
