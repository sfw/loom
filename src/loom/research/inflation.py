"""Bundled inflation index data and helpers (API-key-free)."""

from __future__ import annotations

from dataclasses import dataclass

# CPI-U annual averages (US city average, all items). Values are rounded.
# Source lineage: historical BLS annual-average CPI-U table, bundled locally.
CPI_U_ANNUAL: dict[int, float] = {
    1913: 9.9,
    1914: 10.0,
    1915: 10.1,
    1916: 10.9,
    1917: 12.8,
    1918: 15.0,
    1919: 17.3,
    1920: 20.0,
    1921: 17.9,
    1922: 16.8,
    1923: 17.1,
    1924: 17.1,
    1925: 17.5,
    1926: 17.7,
    1927: 17.4,
    1928: 17.1,
    1929: 17.1,
    1930: 16.7,
    1931: 15.2,
    1932: 13.7,
    1933: 13.0,
    1934: 13.4,
    1935: 13.7,
    1936: 13.9,
    1937: 14.4,
    1938: 14.1,
    1939: 13.9,
    1940: 14.0,
    1941: 14.7,
    1942: 16.3,
    1943: 17.3,
    1944: 17.6,
    1945: 18.0,
    1946: 19.5,
    1947: 22.3,
    1948: 24.1,
    1949: 23.8,
    1950: 24.1,
    1951: 26.0,
    1952: 26.5,
    1953: 26.7,
    1954: 26.9,
    1955: 26.8,
    1956: 27.2,
    1957: 28.1,
    1958: 28.9,
    1959: 29.1,
    1960: 29.6,
    1961: 29.9,
    1962: 30.2,
    1963: 30.6,
    1964: 31.0,
    1965: 31.5,
    1966: 32.4,
    1967: 33.4,
    1968: 34.8,
    1969: 36.7,
    1970: 38.8,
    1971: 40.5,
    1972: 41.8,
    1973: 44.4,
    1974: 49.3,
    1975: 53.8,
    1976: 56.9,
    1977: 60.6,
    1978: 65.2,
    1979: 72.6,
    1980: 82.4,
    1981: 90.9,
    1982: 96.5,
    1983: 99.6,
    1984: 103.9,
    1985: 107.6,
    1986: 109.6,
    1987: 113.6,
    1988: 118.3,
    1989: 124.0,
    1990: 130.7,
    1991: 136.2,
    1992: 140.3,
    1993: 144.5,
    1994: 148.2,
    1995: 152.4,
    1996: 156.9,
    1997: 160.5,
    1998: 163.0,
    1999: 166.6,
    2000: 172.2,
    2001: 177.1,
    2002: 179.9,
    2003: 184.0,
    2004: 188.9,
    2005: 195.3,
    2006: 201.6,
    2007: 207.3,
    2008: 215.3,
    2009: 214.5,
    2010: 218.1,
    2011: 224.9,
    2012: 229.6,
    2013: 233.0,
    2014: 236.7,
    2015: 237.0,
    2016: 240.0,
    2017: 245.1,
    2018: 251.1,
    2019: 255.7,
    2020: 258.8,
    2021: 271.0,
    2022: 292.7,
    2023: 305.3,
    2024: 312.2,
    2025: 318.0,
}

SERIES_VERSION = "2026-02-21-bundled"
SERIES_PROVENANCE = "BLS CPI-U annual averages (bundled snapshot)"


@dataclass(frozen=True)
class InflationCalculation:
    amount: float
    from_year: int
    to_year: int
    adjusted_amount: float
    multiplier: float
    percent_change: float
    index: str
    from_index_value: float
    to_index_value: float
    note: str = ""


def available_years() -> tuple[int, int]:
    years = sorted(CPI_U_ANNUAL)
    return years[0], years[-1]


def cpi_series(index: str) -> tuple[dict[int, float], str]:
    """Return index series and optional note."""
    normalized = index.strip().lower() or "cpi_u"
    if normalized == "cpi_u":
        return CPI_U_ANNUAL, ""
    if normalized == "cpi_w":
        # Keep tool keyless/offline by deriving a CPI-W-like proxy series.
        # This is transparent in the returned note and metadata.
        proxy = {year: round(value * 0.973, 3) for year, value in CPI_U_ANNUAL.items()}
        return proxy, "CPI-W proxy derived from bundled CPI-U series"
    raise ValueError("Unsupported index. Use cpi_u or cpi_w")


def calculate_inflation(
    amount: float,
    from_year: int,
    to_year: int,
    *,
    index: str = "cpi_u",
) -> InflationCalculation:
    """Compute inflation-adjusted value from one year to another."""
    series, note = cpi_series(index)

    if from_year not in series:
        lo, hi = available_years()
        raise ValueError(f"from_year {from_year} out of range ({lo}-{hi})")
    if to_year not in series:
        lo, hi = available_years()
        raise ValueError(f"to_year {to_year} out of range ({lo}-{hi})")

    from_value = series[from_year]
    to_value = series[to_year]
    if from_value <= 0:
        raise ValueError("Invalid source index value")

    multiplier = to_value / from_value
    adjusted = amount * multiplier
    percent_change = (multiplier - 1.0) * 100.0

    return InflationCalculation(
        amount=amount,
        from_year=from_year,
        to_year=to_year,
        adjusted_amount=adjusted,
        multiplier=multiplier,
        percent_change=percent_change,
        index=index,
        from_index_value=from_value,
        to_index_value=to_value,
        note=note,
    )
