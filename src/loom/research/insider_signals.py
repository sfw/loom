"""Signal analytics for insider transactions."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Any

from loom.research.finance import clamp, safe_float


def _parse_iso_day(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _direction_and_signed_value(row: dict[str, Any]) -> tuple[str, float]:
    acquired_disposed = str(row.get("acquired_disposed", "")).strip().upper()
    code = str(row.get("transaction_code", "")).strip().upper()
    value = safe_float(row.get("transaction_value"))
    if value is None:
        shares = safe_float(row.get("shares")) or 0.0
        price = safe_float(row.get("price")) or 0.0
        value = shares * price

    is_buy = acquired_disposed == "A" or code in {"P", "M"}
    is_sell = acquired_disposed == "D" or code in {"S"}
    if is_buy and not is_sell:
        return "buy", abs(value)
    if is_sell and not is_buy:
        return "sell", -abs(value)
    return "unknown", 0.0


def summarize_insider_activity(transactions: list[dict[str, Any]]) -> dict[str, Any]:
    if not transactions:
        return {
            "transaction_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "net_value": 0.0,
            "role_weighted_score": 0.0,
            "signal_label": "neutral",
            "confidence": 0.2,
            "warnings": ["No insider transactions available."],
        }

    buy_value = 0.0
    sell_value = 0.0
    buy_count = 0
    sell_count = 0
    net_value = 0.0
    weighted_net = 0.0
    weight_sum = 0.0
    code_counter: Counter[str] = Counter()

    for row in transactions:
        direction, signed_value = _direction_and_signed_value(row)
        code = str(row.get("transaction_code", "")).strip().upper()
        if code:
            code_counter[code] += 1
        net_value += signed_value

        role_weight = safe_float(row.get("owner_role_weight"))
        if role_weight is None:
            role_weight = 1.0
        weight_sum += max(0.1, role_weight)
        weighted_net += signed_value * max(0.1, role_weight)

        if direction == "buy":
            buy_count += 1
            buy_value += abs(signed_value)
        elif direction == "sell":
            sell_count += 1
            sell_value += abs(signed_value)

    gross = buy_value + sell_value
    net_ratio = (net_value / gross) if gross > 0 else 0.0
    role_weighted_ratio = (weighted_net / gross) if gross > 0 else 0.0

    score = clamp(0.65 * net_ratio + 0.35 * role_weighted_ratio, -1.0, 1.0)
    if score >= 0.25:
        label = "bullish"
    elif score <= -0.25:
        label = "bearish"
    else:
        label = "neutral"

    confidence = clamp(0.3 + 0.03 * min(len(transactions), 20), 0.25, 0.9)
    return {
        "transaction_count": len(transactions),
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_value": buy_value,
        "sell_value": sell_value,
        "net_value": net_value,
        "net_value_ratio": net_ratio,
        "role_weighted_score": score,
        "signal_label": label,
        "transaction_codes": dict(code_counter),
        "confidence": confidence,
        "warnings": [],
    }


def detect_cluster_buys(
    transactions: list[dict[str, Any]],
    *,
    window_days: int = 30,
    min_insiders: int = 3,
) -> list[dict[str, Any]]:
    """Detect clusters of distinct insiders buying in a short window."""
    window_days = max(3, min(120, int(window_days)))
    min_insiders = max(2, min(20, int(min_insiders)))

    buys: list[dict[str, Any]] = []
    for row in transactions:
        direction, signed_value = _direction_and_signed_value(row)
        if direction != "buy":
            continue
        owner = str(row.get("owner_name", "")).strip()
        day = _parse_iso_day(row.get("transaction_date") or row.get("filing_date"))
        if not owner or day is None:
            continue
        buys.append(
            {
                "owner_name": owner,
                "date": day,
                "value": abs(signed_value),
            }
        )

    buys.sort(key=lambda item: item["date"])
    events: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for i, item in enumerate(buys):
        start = item["date"]
        end = start + timedelta(days=window_days)
        owners: set[str] = set()
        total_value = 0.0
        for row in buys[i:]:
            day = row["date"]
            if day > end:
                break
            owners.add(row["owner_name"])
            total_value += float(row["value"])

        if len(owners) < min_insiders:
            continue
        key = f"{start.isoformat()}::{len(owners)}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        events.append(
            {
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "distinct_buyers": len(owners),
                "buyers": sorted(owners),
                "total_buy_value": total_value,
            }
        )

    return events


__all__ = ["detect_cluster_buys", "summarize_insider_activity"]
