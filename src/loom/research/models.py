"""Normalized data models for research-oriented tools."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AcademicResult:
    """One normalized academic search hit."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    url: str = ""
    doi: str = ""
    abstract: str = ""
    source_db: str = ""
    source_type: str = ""
    citation_count: int | None = None
    confidence: float = 0.0

    def dedupe_key(self) -> str:
        """Return a stable key for cross-provider de-duplication."""
        if self.doi.strip():
            return f"doi:{self.doi.strip().lower()}"
        if self.url.strip():
            return f"url:{self.url.strip().lower()}"
        title = " ".join(self.title.lower().split())
        return f"title:{title}|{self.year or 0}"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "doi": self.doi,
            "abstract": self.abstract,
            "source_db": self.source_db,
            "source_type": self.source_type,
            "citation_count": self.citation_count,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ArchiveResult:
    """One normalized archive discovery result."""

    title: str
    creator: str = ""
    date: str = ""
    repository: str = ""
    record_url: str = ""
    access_url: str = ""
    rights: str = ""
    snippet: str = ""
    media_type: str = ""

    def dedupe_key(self) -> str:
        """Return a stable key for de-duplication."""
        if self.record_url.strip():
            return f"record:{self.record_url.strip().lower()}"
        if self.access_url.strip():
            return f"access:{self.access_url.strip().lower()}"
        title = " ".join(self.title.lower().split())
        date = self.date.strip().lower()
        return f"title:{title}|{date}|{self.repository.lower()}"

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "creator": self.creator,
            "date": self.date,
            "repository": self.repository,
            "record_url": self.record_url,
            "access_url": self.access_url,
            "rights": self.rights,
            "snippet": self.snippet,
            "media_type": self.media_type,
        }


@dataclass
class CitationRecord:
    """Canonical citation entry used by citation manager and fact checker."""

    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    url: str = ""
    doi: str = ""
    publisher: str = ""
    type: str = ""
    accessed_on: str = ""
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def dedupe_key(self) -> str:
        """Return canonical key for duplicate detection."""
        if self.doi.strip():
            return f"doi:{self.doi.strip().lower()}"
        if self.url.strip():
            return f"url:{self.url.strip().lower()}"
        title = " ".join(self.title.lower().split())
        return f"title:{title}|{self.year or 0}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "doi": self.doi,
            "publisher": self.publisher,
            "type": self.type,
            "accessed_on": self.accessed_on,
            "notes": self.notes,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class FactCheckVerdict:
    """Outcome for one checked claim."""

    claim: str
    verdict: str
    confidence: float
    rationale: str
    source: str = ""
    source_excerpt: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "source": self.source,
            "source_excerpt": self.source_excerpt,
        }
