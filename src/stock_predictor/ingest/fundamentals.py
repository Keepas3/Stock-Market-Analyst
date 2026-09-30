"""Company fundamentals (P/E ratio) and company-profile fields (sector,
industry, description, website, ...).

**Finnhub is the primary source** (ingest/finnhub_client.py, `stock/profile2`
+ `stock/metric`) -- its free tier's 60-requests/minute cap with no daily
limit comfortably covers refreshing all watchlist symbols, unlike Alpha
Vantage's 25/day account-wide budget which this used to depend on alone
and regularly exhausted before P/E could be fetched.

**Alpha Vantage's OVERVIEW is still called every time, unconditionally, as
a fallback AND as the sole source of `description`/`address`** -- Finnhub's
`stock/profile2` has no equivalent field for either. Since Finnhub succeeds
far more often now (no daily cap), always deferring to it alone would
silently and permanently blank out the dashboard's "About" section
(dashboard/components.py::render_company_background is a no-op without a
description) -- so this module merges both sources every call rather than
picking one exclusively. Alpha Vantage's weekly cadence here (see
CACHE_TTL_SECONDS) still comfortably fits its shared 25/day budget
alongside ingest/sentiment.py and ingest/financials.py (see those modules'
own docstrings for the budget math).

**Real schema check, confirmed against Alpha Vantage's own documented
OVERVIEW response (not assumed): there is no employee-count field and no
structured "founded"/incorporation-year field** -- a founding year
sometimes appears in prose inside `Description`, but nothing structured.
This app never fabricates those two fields to match a richer competitor
UI; the company-background section just omits them (see
dashboard/components.py::render_company_background).

Every field on FundamentalsResult can independently be None -- a
loss-making company genuinely has no P/E; neither provider covers every
field for every symbol; and either provider can fail outright (missing
key, network error, rate limit, unknown ticker) without failing the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from stock_predictor.ingest import alpha_vantage_client, finnhub_client

# Weekly, not daily -- P/E moves on earnings, not day to day, and this
# cadence is what keeps Alpha Vantage's fallback/description call within
# its shared 25/day budget (see ingest/sentiment.py, ingest/financials.py).
CACHE_TTL_SECONDS = 7 * 24 * 3600

# Finnhub's documented field names for trailing/forward P/E inside
# stock/metric's `metric` dict vary by symbol/data availability -- read
# several candidates in preference order, take the first present.
_FINNHUB_PE_KEYS = ("peBasicExclExtraTTM", "peTTM", "peExclExtraTTM", "peNormalizedAnnual")
_FINNHUB_FORWARD_PE_KEYS = ("peForward",)

# Finnhub's stock/profile2 `marketCapitalization` is documented in millions
# of USD, unlike Alpha Vantage's raw-dollar MarketCapitalization.
_FINNHUB_MARKET_CAP_SCALE = 1_000_000


@dataclass
class FundamentalsResult:
    ticker: str
    pe_ratio: float | None
    forward_pe: float | None
    sector: str | None
    name: str | None
    description: str | None
    industry: str | None
    exchange: str | None
    country: str | None
    address: str | None
    official_site: str | None
    market_cap: float | None


def _parse_float(value: object) -> float | None:
    # Alpha Vantage's own sentinel for "not available" is the literal
    # string "None" -- float("None") raises ValueError, caught here.
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clean_str(value: object) -> str | None:
    # Same "None" sentinel as _parse_float, plus a genuinely empty string
    # -- both mean "not available," not a real value to display.
    if value in (None, "", "None"):
        return None
    return str(value)


def _first_present(metric: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _parse_float(metric.get(key))
        if value is not None:
            return value
    return None


def _fetch_from_alpha_vantage(ticker: str) -> FundamentalsResult | None:
    """None on any failure (missing key, network error, daily budget
    exhausted, unknown ticker, or a response with no real `Symbol` field)
    -- never raises, same contract as every other integration here.
    """
    try:
        data = alpha_vantage_client.get("OVERVIEW", {"symbol": ticker}, CACHE_TTL_SECONDS)
    except (alpha_vantage_client.MissingApiKey, requests.RequestException, RuntimeError, ValueError):
        return None

    if not data or "Symbol" not in data:
        return None

    return FundamentalsResult(
        ticker=ticker,
        pe_ratio=_parse_float(data.get("PERatio")),
        forward_pe=_parse_float(data.get("ForwardPE")),
        sector=_clean_str(data.get("Sector")),
        name=_clean_str(data.get("Name")),
        description=_clean_str(data.get("Description")),
        industry=_clean_str(data.get("Industry")),
        exchange=_clean_str(data.get("Exchange")),
        country=_clean_str(data.get("Country")),
        address=_clean_str(data.get("Address")),
        official_site=_clean_str(data.get("OfficialSite")),
        market_cap=_parse_float(data.get("MarketCapitalization")),
    )


def _fetch_from_finnhub(ticker: str) -> FundamentalsResult | None:
    """None on any failure (missing key, network error, rate limit
    exhausted, or a response with no real `metric`/profile data) -- never
    raises, same contract as every other integration here. Finnhub has no
    `description`/`address`/sector-equivalent fields -- those are always
    None from this source; fetch_overview fills them from Alpha Vantage.
    """
    try:
        profile = finnhub_client.get("stock/profile2", {"symbol": ticker}, CACHE_TTL_SECONDS)
        metric_response = finnhub_client.get("stock/metric", {"symbol": ticker, "metric": "all"}, CACHE_TTL_SECONDS)
    except (finnhub_client.MissingApiKey, requests.RequestException, RuntimeError, ValueError):
        return None

    if not profile and not metric_response:
        return None

    metric = metric_response.get("metric") or {} if isinstance(metric_response, dict) else {}
    market_cap = _parse_float(profile.get("marketCapitalization"))
    if market_cap is not None:
        market_cap *= _FINNHUB_MARKET_CAP_SCALE

    return FundamentalsResult(
        ticker=ticker,
        pe_ratio=_first_present(metric, _FINNHUB_PE_KEYS),
        forward_pe=_first_present(metric, _FINNHUB_FORWARD_PE_KEYS),
        sector=None,
        name=_clean_str(profile.get("name")),
        description=None,
        industry=_clean_str(profile.get("finnhubIndustry")),
        exchange=_clean_str(profile.get("exchange")),
        country=_clean_str(profile.get("country")),
        address=None,
        official_site=_clean_str(profile.get("weburl")),
        market_cap=market_cap,
    )


def fetch_overview(ticker: str) -> FundamentalsResult | None:
    """Merges Finnhub (primary, for pe_ratio/forward_pe/market_cap/profile
    fields) with Alpha Vantage (fallback for those same fields, and the
    sole source of description/address/sector). Both sources are always
    attempted -- Alpha Vantage's call is never skipped just because
    Finnhub succeeded, since it's the only source for description/address
    and dashboard/components.py::render_company_background hides the whole
    "About" section without one. Returns None only if both sources
    contribute nothing; any individual field can independently be None.
    """
    finnhub_result = _fetch_from_finnhub(ticker)
    av_result = _fetch_from_alpha_vantage(ticker)

    if finnhub_result is None and av_result is None:
        return None

    def prefer_finnhub(field: str) -> object:
        finnhub_value = getattr(finnhub_result, field, None) if finnhub_result else None
        if finnhub_value is not None:
            return finnhub_value
        return getattr(av_result, field, None) if av_result else None

    return FundamentalsResult(
        ticker=ticker,
        pe_ratio=prefer_finnhub("pe_ratio"),
        forward_pe=prefer_finnhub("forward_pe"),
        sector=av_result.sector if av_result else None,
        name=prefer_finnhub("name"),
        description=av_result.description if av_result else None,
        industry=prefer_finnhub("industry"),
        exchange=prefer_finnhub("exchange"),
        country=prefer_finnhub("country"),
        address=av_result.address if av_result else None,
        official_site=prefer_finnhub("official_site"),
        market_cap=prefer_finnhub("market_cap"),
    )
