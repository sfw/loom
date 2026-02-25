"""Tests for market-signal provider adapters."""

from __future__ import annotations

import httpx

from loom.research.providers.finra_shorts import (
    fetch_finra_daily_short_volume,
    parse_finra_daily_short_volume,
    parse_finra_short_interest_csv,
)
from loom.research.providers.options_flow import (
    fetch_cboe_put_call_history,
    parse_options_flow_csv,
)
from loom.research.providers.sec_insiders import parse_form345_transactions


async def test_fetch_cboe_put_call_history_parses_csv():
    csv_body = "Date,Put,Call,PC\n2026-01-01,100,120,0.8333\n2026-01-02,130,100,1.3\n"

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert "totalpc.csv" in str(request.url)
        return httpx.Response(status_code=200, text=csv_body, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        payload = await fetch_cboe_put_call_history(client=client)

    assert payload["provider"] == "cboe"
    assert len(payload["rows"]) == 2
    assert payload["rows"][-1]["put_call_ratio"] == 1.3


def test_parse_options_flow_csv_handles_symbol_rows():
    csv_body = (
        "trade_date,symbol,put_volume,call_volume,total_volume\n"
        "2026-01-02,AAPL,200,400,600\n"
        "2026-01-03,AAPL,300,200,500\n"
    )
    rows = parse_options_flow_csv(csv_body, source="fixture")
    assert len(rows) == 2
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["put_call_ratio"] == 0.5


def test_parse_finra_short_interest_csv_parses_core_fields():
    csv_body = (
        "SettlementDate,Symbol,CurrentShortPositionQuantity,AverageDailyVolumeQuantity,DaysToCoverQuantity\n"
        "20260115,AAPL,1000000,250000,4\n"
    )
    rows = parse_finra_short_interest_csv(csv_body, source="fixture")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["short_interest"] == 1_000_000


def test_parse_finra_daily_short_volume_pipe_delimited():
    txt = (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20260116|AAPL|50000|1000|120000|NMS\n"
    )
    rows = parse_finra_daily_short_volume(txt, source="fixture")
    assert len(rows) == 1
    assert rows[0]["short_volume_ratio"] == 50000 / 120000


async def test_fetch_finra_daily_short_volume_builds_default_url():
    txt = (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20260116|MSFT|10|1|20|NMS\n"
    )

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert "CNMSshvol20260116.txt" in str(request.url)
        return httpx.Response(status_code=200, text=txt, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        payload = await fetch_finra_daily_short_volume(date_token="20260116", client=client)

    assert payload["dataset"] == "daily_short_volume"
    assert payload["rows"][0]["symbol"] == "MSFT"


def test_parse_form345_transactions_extracts_transaction_rows():
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<ownershipDocument>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>APPLE INC</issuerName>
    <issuerTradingSymbol>AAPL</issuerTradingSymbol>
  </issuer>
  <periodOfReport>2026-01-20</periodOfReport>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>0001214156</rptOwnerCik>
      <rptOwnerName>DOE JOHN</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>0</isTenPercentOwner>
      <isOther>0</isOther>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-01-20</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>200</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
      <ownershipNature>
        <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
      </ownershipNature>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""
    parsed = parse_form345_transactions(xml)
    assert parsed["issuer"]["symbol"] == "AAPL"
    assert len(parsed["transactions"]) == 1
    tx = parsed["transactions"][0]
    assert tx["transaction_code"] == "P"
    assert tx["transaction_value"] == 200000
