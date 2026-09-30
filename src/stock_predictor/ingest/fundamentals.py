"""Company fundamentals (P/E ratio) and company-profile fields (sector,
industry, description, website, ...) via Alpha Vantage's OVERVIEW endpoint
-- one call covers both, no extra API cost for the profile fields.

Confirmed live (2026-09) with a real (non-demo) key: real PERatio/
ForwardPE/Sector for AAPL. Yahoo Finance's `quoteSummary`/`v7/finance/quote`
endpoints (the obvious free alternative) were checked and are now both
blocked behind a "crumb"/auth requirement -- confirmed dead ends, not
assumed -- so Alpha Vantage is the only working free source found.

**Real schema check, confirmed against Alpha Vantage's own documented
OVERVIEW response (not assumed): there is no employee-count field and no
structured "founded"/incorporation-year field** -- a founding year
sometimes appears in prose inside `Description`, but nothing structured.
This app never fabricates those two fields to match a richer competitor
UI; the company-background section just omits them (see
dashboard/components.py::render_company_background).

Shares ingest/alpha_vantage_client.py's disk cache + daily rate limiter
with ingest/sentiment.py's NEWS_SENTIMENT calls (same account, same
25-requests/day budget). Refreshed WEEKLY by scripts/refresh_live_data.py,
not daily like sentiment -- P/E moves on earnings, not day to day, and the
profile fields (sector/industry/description/website) change even less
often -- 16 watchlist symbols/7 days ~= 2.3 calls/day on top of the
existing ~16/day sentiment calls, comfortably under the shared cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from stock_predictor.ingest import alpha_vantage_client

# See module docstring -- weekly, not daily, to share the account-wide
# 25/day Alpha Vantage budget with ingest/sentiment.py's NEWS_SENTIMENT calls.
CACHE_TTL_SECONDS = 7 * 24 * 3600


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


def fetch_overview(ticker: str) -> FundamentalsResult | None:
    """None on any failure (missing key, network error, daily budget
    exhausted, unknown ticker, or a response with no real `Symbol` field)
    -- never raises, same contract as every other integration here.
    Every field can independently be None within a successful result (e.g.
    a loss-making company genuinely has no P/E; Alpha Vantage doesn't
    cover every field for every symbol).
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
