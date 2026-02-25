"""SEC insider-filings adapters (keyless)."""

from __future__ import annotations

import asyncio
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import date
from typing import Any

import httpx

from loom.research.finance import safe_float

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_USER_AGENT = "Loom/1.0 (+https://github.com/sfw/loom)"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"
ALLOWED_INSIDER_FORMS = frozenset({"3", "3/A", "4", "4/A", "5", "5/A"})

# SEC guidance suggests <=10 requests/sec per client.
_SEC_RATE_LIMIT_HZ = 9.0
_SEC_MIN_INTERVAL_SECONDS = 1.0 / _SEC_RATE_LIMIT_HZ
_SEC_RATE_LOCK = asyncio.Lock()
_SEC_LAST_REQUEST_TS = 0.0


class SecInsiderDataError(RuntimeError):
    """Raised when SEC insider payloads are invalid."""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get("LOOM_WEB_USER_AGENT", "").strip() or DEFAULT_USER_AGENT,
        "Accept": "application/json,text/xml,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _normalize_cik(value: object) -> str:
    text = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not text:
        return ""
    return text.zfill(10)


def _parse_iso_day(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


async def _respect_sec_rate_limit() -> None:
    global _SEC_LAST_REQUEST_TS
    async with _SEC_RATE_LOCK:
        now = time.monotonic()
        wait = _SEC_MIN_INTERVAL_SECONDS - (now - _SEC_LAST_REQUEST_TS)
        if wait > 0:
            await asyncio.sleep(wait)
        _SEC_LAST_REQUEST_TS = time.monotonic()


async def _sec_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Mapping[str, object] | None = None,
) -> httpx.Response:
    await _respect_sec_rate_limit()
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response


def _local_name(tag: object) -> str:
    text = str(tag or "")
    if "}" in text:
        return text.rsplit("}", 1)[-1]
    return text


def _iter_by_local(node: ET.Element, name: str):
    for elem in node.iter():
        if _local_name(elem.tag) == name:
            yield elem


def _path_text(node: ET.Element, *segments: str) -> str:
    current = node
    for segment in segments:
        nxt = None
        for child in list(current):
            if _local_name(child.tag) == segment:
                nxt = child
                break
        if nxt is None:
            return ""
        current = nxt
    return str(current.text or "").strip()


def _safe_parse_xml(raw_text: str) -> ET.Element:
    text = raw_text.strip()
    if not text:
        raise SecInsiderDataError("SEC filing text payload is empty")
    try:
        return ET.fromstring(text)
    except ET.ParseError:
        start = text.find("<ownershipDocument")
        end = text.rfind("</ownershipDocument>")
        if start >= 0 and end > start:
            snippet = text[start : end + len("</ownershipDocument>")]
            try:
                return ET.fromstring(snippet)
            except ET.ParseError as e:  # pragma: no cover - defensive
                raise SecInsiderDataError("Unable to parse filing ownership XML") from e
        raise SecInsiderDataError("Unable to parse SEC filing XML")


def _owner_role_weight(owner: dict[str, Any]) -> float:
    weight = 1.0
    if owner.get("is_director"):
        weight += 0.15
    if owner.get("is_officer"):
        weight += 0.15
    if owner.get("is_ten_percent_owner"):
        weight += 0.1
    if owner.get("is_other"):
        weight += 0.05
    return weight


def _parse_reporting_owners(root: ET.Element) -> list[dict[str, Any]]:
    owners: list[dict[str, Any]] = []
    for owner_node in _iter_by_local(root, "reportingOwner"):
        owner_name = _path_text(owner_node, "reportingOwnerId", "rptOwnerName")
        if not owner_name:
            owner_name = _path_text(owner_node, "reportingOwnerId", "rptOwnerCik")
        relationship = _next_by_local(owner_node, "reportingOwnerRelationship")
        has_relationship = relationship is not None
        is_director = (
            _truthy(_path_text(relationship, "isDirector"))
            if has_relationship
            else False
        )
        is_officer = _truthy(_path_text(relationship, "isOfficer")) if has_relationship else False
        is_ten_percent_owner = (
            _truthy(_path_text(relationship, "isTenPercentOwner"))
            if has_relationship
            else False
        )
        is_other = _truthy(_path_text(relationship, "isOther")) if has_relationship else False
        owners.append(
            {
                "owner_name": owner_name,
                "is_director": is_director,
                "is_officer": is_officer,
                "is_ten_percent_owner": is_ten_percent_owner,
                "is_other": is_other,
                "officer_title": (
                    _path_text(relationship, "officerTitle")
                    if has_relationship
                    else ""
                ),
            }
        )
    return owners


def _next_by_local(node: ET.Element, name: str) -> ET.Element | None:
    for child in node.iter():
        if _local_name(child.tag) == name:
            return child
    return None


def _truthy(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "y", "yes"}


def parse_form345_transactions(xml_text: str) -> dict[str, Any]:
    """Parse Form 3/4/5 ownership XML into normalized transactions."""
    root = _safe_parse_xml(xml_text)
    owners = _parse_reporting_owners(root)
    owner_names = [o.get("owner_name", "") for o in owners if o.get("owner_name")]
    owner_primary = owners[0] if owners else {}

    issuer_cik = _path_text(root, "issuer", "issuerCik")
    issuer_name = _path_text(root, "issuer", "issuerName")
    issuer_trading_symbol = _path_text(root, "issuer", "issuerTradingSymbol").upper()
    period_of_report = _path_text(root, "periodOfReport")

    transactions: list[dict[str, Any]] = []
    for tag in ("nonDerivativeTransaction", "derivativeTransaction"):
        for txn in _iter_by_local(root, tag):
            shares = safe_float(
                _path_text(txn, "transactionAmounts", "transactionShares", "value")
            )
            price = safe_float(
                _path_text(txn, "transactionAmounts", "transactionPricePerShare", "value")
            )
            txn_date = _path_text(txn, "transactionDate", "value") or period_of_report
            code = _path_text(txn, "transactionCoding", "transactionCode").upper()
            acquired_disposed = _path_text(
                txn,
                "transactionAmounts",
                "transactionAcquiredDisposedCode",
                "value",
            ).upper()

            value = None
            if shares is not None and price is not None:
                value = shares * price

            row = {
                "instrument_type": "derivative" if tag == "derivativeTransaction" else "equity",
                "transaction_date": txn_date,
                "transaction_code": code,
                "acquired_disposed": acquired_disposed,
                "shares": shares,
                "price": price,
                "transaction_value": value,
                "security_title": _path_text(txn, "securityTitle", "value"),
                "ownership_type": _path_text(
                    txn,
                    "ownershipNature",
                    "directOrIndirectOwnership",
                    "value",
                ),
                "owner_name": owner_primary.get("owner_name", ""),
                "owner_names": owner_names,
                "owner_role_weight": _owner_role_weight(owner_primary) if owner_primary else 1.0,
                "owner_relationship": owner_primary,
                "period_of_report": period_of_report,
                "issuer_cik": issuer_cik,
                "issuer_name": issuer_name,
                "issuer_symbol": issuer_trading_symbol,
            }
            if (
                not row["transaction_code"]
                and row["shares"] is None
                and row["transaction_value"] is None
            ):
                continue
            transactions.append(row)

    return {
        "issuer": {
            "cik": issuer_cik,
            "name": issuer_name,
            "symbol": issuer_trading_symbol,
        },
        "period_of_report": period_of_report,
        "owner_names": owner_names,
        "owner_count": len(owner_names),
        "transactions": transactions,
    }


def build_sec_filing_url(*, cik: str, accession_number: str, primary_document: str) -> str:
    cik10 = _normalize_cik(cik)
    if not cik10:
        raise SecInsiderDataError("cik is required")
    accession = str(accession_number or "").strip()
    if not accession:
        raise SecInsiderDataError("accession_number is required")
    accession_nodash = accession.replace("-", "")
    doc = str(primary_document or "").strip()
    if not doc:
        doc = ""
        raise SecInsiderDataError("primary_document is required")
    return SEC_ARCHIVES_FILING_URL.format(
        cik_int=int(cik10),
        accession_nodash=accession_nodash,
        doc=doc,
    )


async def fetch_sec_submissions(
    *,
    cik: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    cik10 = _normalize_cik(cik)
    if not cik10:
        raise SecInsiderDataError("cik is required")

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )
    try:
        response = await _sec_get(client, SEC_SUBMISSIONS_URL.format(cik10=cik10))
        payload = response.json()
    finally:
        if owns_client:
            await client.aclose()

    if not isinstance(payload, dict):
        raise SecInsiderDataError("Unexpected SEC submissions payload")
    return payload


def extract_recent_form345_filings(
    submissions_payload: Mapping[str, Any],
    *,
    max_filings: int = 40,
    start_date: str = "",
    allowed_forms: set[str] | None = None,
) -> list[dict[str, Any]]:
    forms = set(allowed_forms or ALLOWED_INSIDER_FORMS)
    recent = submissions_payload.get("filings", {}).get("recent", {})
    if not isinstance(recent, Mapping):
        raise SecInsiderDataError("SEC submissions missing filings.recent")

    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    form_list = recent.get("form", [])
    primary_docs = recent.get("primaryDocument", [])

    start_day = _parse_iso_day(start_date) if start_date else None
    max_filings = max(1, min(500, int(max_filings)))

    rows: list[dict[str, Any]] = []
    n = min(
        len(accession_numbers),
        len(filing_dates),
        len(form_list),
        len(primary_docs),
        len(report_dates),
    )
    cik = _normalize_cik(submissions_payload.get("cik"))
    for i in range(n):
        form = str(form_list[i] or "").strip().upper()
        if form not in forms:
            continue
        filing_date = str(filing_dates[i] or "").strip()
        filing_day = _parse_iso_day(filing_date)
        if filing_day is None:
            continue
        if start_day and filing_day < start_day:
            continue

        accession = str(accession_numbers[i] or "").strip()
        primary_document = str(primary_docs[i] or "").strip()
        if not accession or not primary_document:
            continue
        filing_url = SEC_ARCHIVES_FILING_URL.format(
            cik_int=int(cik or "0"),
            accession_nodash=accession.replace("-", ""),
            doc=primary_document,
        )
        rows.append(
            {
                "cik": cik,
                "form": form,
                "accession_number": accession,
                "filing_date": filing_date,
                "report_date": str(report_dates[i] or "").strip(),
                "primary_document": primary_document,
                "filing_url": filing_url,
            }
        )
        if len(rows) >= max_filings:
            break

    rows.sort(key=lambda row: str(row.get("filing_date", "")), reverse=True)
    return rows


async def fetch_sec_filing_transactions(
    *,
    filing_url: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    url = filing_url.strip()
    if not url:
        raise SecInsiderDataError("filing_url is required")

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
            follow_redirects=True,
            headers=_headers(),
        )
    try:
        response = await _sec_get(client, url)
        parsed = parse_form345_transactions(response.text)
    finally:
        if owns_client:
            await client.aclose()

    parsed["source_url"] = str(response.url)
    return parsed


__all__ = [
    "ALLOWED_INSIDER_FORMS",
    "SEC_ARCHIVES_FILING_URL",
    "SEC_SUBMISSIONS_URL",
    "SecInsiderDataError",
    "build_sec_filing_url",
    "extract_recent_form345_filings",
    "fetch_sec_filing_transactions",
    "fetch_sec_submissions",
    "parse_form345_transactions",
]
