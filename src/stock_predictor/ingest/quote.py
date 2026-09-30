"""Current price quote for one symbol -- Finnhub's /quote endpoint, the
cheapest possible per-symbol call on the free tier (no daily cap, and this
single call returns current/high/low/open/previous-close in one shot).

Distinct from ingest/fundamentals.py and ingest/price_history.py: those
feed the dashboard's daily OHLCV history and P/E, refreshed at most once a
day by scripts/refresh_live_data.py. This module exists ONLY for
scripts/check_price_alerts.py's near-real-time polling loop (every
~15 minutes during market hours, see
.github/workflows/check-price-alerts.yml) -- hence the short
CACHE_TTL_SECONDS below, the opposite of every other ingest module's
day(s)-long cache: a stale quote defeats the entire point of this feature.

Finnhub's own documented sentinel for "no data for this symbol" on /quote
is `c: 0` (or a missing `c` key) -- treated as no-data here, same as an
empty dict/list is treated as no-data by finnhub_client._is_non_data_response
for other endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from stock_predictor.ingest import finnhub_client

# Short on purpose -- see module docstring. Comfortably above the
# ~15 minute cron cadence this is actually called at, just enough to
# collapse an accidental double-invocation within the same minute.
CACHE_TTL_SECONDS = 60


@dataclass
class Quote:
    ticker: str
    current_price: float
    previous_close: float | None


def fetch_quote(ticker: str) -> Quote | None:
    """None on any failure (missing key, network error, rate limit
    exhausted, unknown ticker, or Finnhub's own `c: 0`/missing-`c`
    sentinel for "no data") -- never raises, same contract as every other
    ingest module here (see e.g. ingest/fundamentals.py's own module
    docstring).
    """
    try:
        data = finnhub_client.get("quote", {"symbol": ticker}, CACHE_TTL_SECONDS)
    except (finnhub_client.MissingApiKey, requests.RequestException, RuntimeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None
    current_price = data.get("c")
    if not current_price:  # None or 0 -- Finnhub's own "unknown symbol" sentinel
        return None

    previous_close = data.get("pc")
    return Quote(
        ticker=ticker,
        current_price=float(current_price),
        previous_close=float(previous_close) if previous_close else None,
    )
