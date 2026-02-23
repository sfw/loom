"""Shared text helpers for research tools."""

from __future__ import annotations

import html
import re
from datetime import datetime
from urllib.parse import urlparse

_WORD_RE = re.compile(r"[a-z0-9]+")
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize free text for matching/scoring."""
    clean = html.unescape(text or "")
    clean = _TAG_RE.sub(" ", clean)
    clean = clean.lower()
    clean = _WHITESPACE_RE.sub(" ", clean)
    return clean.strip()


def tokenize(text: str) -> set[str]:
    """Tokenize normalized text into a lightweight set."""
    return set(_WORD_RE.findall(normalize_text(text)))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Return a stable Jaccard similarity score."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    if union <= 0:
        return 0.0
    return intersection / union


def coerce_int(value: object, default: int | None = None) -> int | None:
    """Best-effort parse integer value."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_year(value: object) -> int | None:
    """Extract a year value from an int/date/string input."""
    if value is None:
        return None
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return None

    if len(text) == 4 and text.isdigit():
        return int(text)

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).year
    except ValueError:
        pass

    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group(0))
    return None


def domain_of(url: str) -> str:
    """Return URL netloc in lowercase."""
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def is_primary_domain(url: str) -> bool:
    """Heuristic primary-source domain classifier."""
    domain = domain_of(url)
    if not domain:
        return False
    return domain.endswith((".gov", ".edu", ".ac.uk")) or any(
        key in domain
        for key in (
            "doi.org",
            "arxiv.org",
            "archive.org",
            "loc.gov",
            "nasa.gov",
            "un.org",
            "who.int",
        )
    )
