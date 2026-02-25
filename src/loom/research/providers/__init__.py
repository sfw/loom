"""Keyless research-data provider adapters."""

from .currency import (
    ECB_CSV_URL,
    convert_via_ecb_reference_rates,
    fetch_ecb_reference_rates,
)
from .economic import (
    SUPPORTED_ECONOMIC_PROVIDERS,
    EconomicProviderError,
    economic_get_observations,
    economic_get_series,
    economic_search,
)
from .markets import (
    SUPPORTED_MARKET_PROVIDERS,
    MarketDataProviderError,
    fetch_stooq_daily_prices,
    normalize_stooq_symbol,
)
from .sec_finance import (
    SEC_COMPANYFACTS_URL,
    SEC_TICKER_URL,
    SecDataError,
    extract_latest_value,
    extract_ttm_value,
    fetch_company_facts,
    fetch_sec_ticker_map,
    resolve_ticker_to_cik,
)

__all__ = [
    "ECB_CSV_URL",
    "SEC_COMPANYFACTS_URL",
    "SEC_TICKER_URL",
    "SUPPORTED_ECONOMIC_PROVIDERS",
    "SUPPORTED_MARKET_PROVIDERS",
    "EconomicProviderError",
    "MarketDataProviderError",
    "SecDataError",
    "convert_via_ecb_reference_rates",
    "economic_get_observations",
    "economic_get_series",
    "economic_search",
    "extract_latest_value",
    "extract_ttm_value",
    "fetch_company_facts",
    "fetch_sec_ticker_map",
    "fetch_ecb_reference_rates",
    "fetch_stooq_daily_prices",
    "normalize_stooq_symbol",
    "resolve_ticker_to_cik",
]
