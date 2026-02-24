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

__all__ = [
    "ECB_CSV_URL",
    "SUPPORTED_ECONOMIC_PROVIDERS",
    "EconomicProviderError",
    "convert_via_ecb_reference_rates",
    "economic_get_observations",
    "economic_get_series",
    "economic_search",
    "fetch_ecb_reference_rates",
]
