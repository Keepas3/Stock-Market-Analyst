"""Peer-comparison data for a watchlist symbol's real competitors -- CNN's
own "Competitors" panel inspiration. Competitor RELATIONSHIPS come from a
small hand-curated config file (config/competitors.yaml, see
config.py::load_competitor_map) -- Alpha Vantage's free tier has no
"list companies by industry" endpoint to derive this live (confirmed
against its documented API surface: neither OVERVIEW, SYMBOL_SEARCH, nor
LISTING_STATUS support sector/industry filtering).

Price/change/52-week-range come from the keyless, unlimited Yahoo Finance
endpoint (ingest/price_history.py) -- no quota concern. Market cap/P-E
come from Alpha Vantage's OVERVIEW (shares the account-wide 25/day budget
with news sentiment/fundamentals/financials -- see
ingest/alpha_vantage_client.py) via ingest/fundamentals.py's own 7-day
disk cache, so a competitor ticker that's ALSO a watchlist symbol (or was
already looked up as another symbol's competitor recently) costs nothing
extra here -- the cache is keyed by ticker, not by which symbol looked it
up.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from stock_predictor.ingest import price_history
from stock_predictor.ingest.fundamentals import fetch_overview

# Matches dashboard/components.py's own FIFTY_52_WEEK_DAYS window (same
# "trailing calendar year from the most recent bar" convention).
FIFTY_TWO_WEEK_DAYS = 365


@dataclass
class CompetitorSnapshot:
    ticker: str
    name: str | None
    price: float | None
    change: float | None
    pct_change: float | None
    fifty_two_week_low: float | None
    fifty_two_week_high: float | None
    market_cap: float | None
    pe_ratio: float | None


def fetch_competitor_snapshot(ticker: str) -> CompetitorSnapshot | None:
    """None if Yahoo has no price history at all for this ticker (a
    genuinely unknown/delisted symbol) -- name/market_cap/pe_ratio can
    independently be None if Alpha Vantage's OVERVIEW fails/has no data
    for it, same degrade-gracefully contract as every other integration
    here (the row still shows with just price data rather than
    disappearing entirely).
    """
    bars = price_history.fetch_full_history(ticker)
    if not bars:
        return None

    last = bars[-1]
    price = last.close
    change = None
    pct_change = None
    if len(bars) >= 2:
        previous_close = bars[-2].close
        change = price - previous_close
        pct_change = change / previous_close if previous_close else None

    cutoff = last.date - dt.timedelta(days=FIFTY_TWO_WEEK_DAYS)
    window = [b for b in bars if b.date >= cutoff]
    fifty_two_week_low = min(b.low for b in window)
    fifty_two_week_high = max(b.high for b in window)

    overview = fetch_overview(ticker)

    return CompetitorSnapshot(
        ticker=ticker,
        name=overview.name if overview else None,
        price=price,
        change=change,
        pct_change=pct_change,
        fifty_two_week_low=fifty_two_week_low,
        fifty_two_week_high=fifty_two_week_high,
        market_cap=overview.market_cap if overview else None,
        pe_ratio=overview.pe_ratio if overview else None,
    )
