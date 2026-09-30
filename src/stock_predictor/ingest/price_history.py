"""Daily OHLCV history from Yahoo Finance's public chart endpoint -- the
direct analog of soccer-predictor's historical_csv.py/understat_client.py,
just genuinely simpler: one call per symbol returns years of history in
one shot, no per-season file loop and no team-name-matching problem (a
ticker is already a clean, stable key).

**Real gotcha, confirmed live (2026-09), not assumed:** `range=max` does
NOT return genuine daily bars -- Yahoo silently downsamples a "max" request
to a sparse series (confirmed: AAPL's real "max" response is only 169
points covering 1984-2026, nowhere near one-per-trading-day). Bounded
ranges return real daily granularity (confirmed: `range=10y` -> 2512
points, ~251.2/year, matching real trading-day counts). `HISTORY_RANGE`
below is deliberately `"10y"`, not `"max"`, for this reason -- and it's a
good fit anyway: this app's return model time-decays with a ~90-day
half-life (model/time_weighting.py), so data older than a few years barely
influences a fit regardless of how far back it goes.

Undocumented and unofficial, same caveat as espn_client.py/understat_client.py
in the sibling soccer-predictor project: no published rate limit or
stability guarantee, requires a real browser-like User-Agent header
(confirmed live -- a bare default requests User-Agent gets a bot-challenge
page back, not JSON). Every parsing function here degrades to an empty
list on any failure rather than raising, same best-effort contract as the
rest of this app's integrations.

`close` below is the SPLIT/DIVIDEND-ADJUSTED close (Yahoo's own
`indicators.adjclose` series), not the raw close -- correct for
return-based modeling (a raw, unadjusted close would show a fake
crash/spike on every stock-split date in a symbol's history, e.g. NVDA's
and AAPL's real historical splits). `open`/`high`/`low` are left
unadjusted -- a documented simplification (only `close` feeds the return
model; OHLC together are only ever used for the dashboard's price chart,
where the small pre-split distortion doesn't matter).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from stock_predictor.config import DATA_DIR

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
CACHE_DIR = DATA_DIR / "cache" / "price_history"

# EOD daily data -- a day-old cache is fine; today's own bar isn't final
# until market close anyway, and refresh_live_data.py forces a re-fetch
# for "today" specifically (see fetch_full_history's `force` param).
CACHE_TTL_SECONDS = 24 * 3600

HEADERS = {"User-Agent": "Mozilla/5.0"}
HISTORY_RANGE = "10y"  # see module docstring -- NOT "max", which silently downsamples


@dataclass
class PriceBar:
    date: dt.date
    open: float
    high: float
    low: float
    close: float  # adjusted close -- see module docstring
    volume: int


def _cache_path(ticker: str, range_: str) -> Path:
    key = hashlib.sha256(f"{ticker}:{range_}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _fetch_chart(ticker: str, range_: str, force: bool = False) -> dict:
    cache_file = _cache_path(ticker, range_)
    if cache_file.exists() and not force:
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    response = requests.get(
        f"{BASE_URL}/{ticker}",
        params={"range": range_, "interval": "1d"},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def _parse_bars(data: dict) -> list[PriceBar]:
    result = data.get("chart", {}).get("result")
    if not result:
        return []
    result = result[0]

    timestamps = result.get("timestamp") or []
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or quote.get("close")

    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    volumes = quote.get("volume") or []

    bars = []
    for i, ts in enumerate(timestamps):
        try:
            o, h, l, c, v = opens[i], highs[i], lows[i], adjclose[i], volumes[i]
        except IndexError:
            continue
        # A halt/gap day has null entries in one or more arrays -- skip it
        # rather than crash, same "skip the bad row" discipline as
        # historical_csv.py's dropna.
        if None in (o, h, l, c, v):
            continue
        bars.append(
            PriceBar(
                date=dt.datetime.fromtimestamp(ts, tz=dt.UTC).date(),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=int(v),
            )
        )
    return bars


def fetch_full_history(ticker: str) -> list[PriceBar]:
    """This ticker's daily OHLCV history over HISTORY_RANGE (see module
    docstring for why that's a bounded range, not "max"). Empty list on
    any failure (network error, unknown ticker, unexpected response shape)
    -- degrades to "no data" rather than raising.
    """
    try:
        data = _fetch_chart(ticker, HISTORY_RANGE)
        return _parse_bars(data)
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return []


def fetch_recent_bars(ticker: str, days: int = 5) -> list[PriceBar]:
    """Just the last `days` (calendar, not trading) of bars -- used by
    refresh_live_data.py to pick up today's/yesterday's bar without
    re-downloading a symbol's whole history. `force=True`: today's own bar
    isn't final until market close, so this always bypasses the cache
    (a real, if small, exception to the 24h TTL every other fetch here
    respects -- this is the one place freshness matters more than
    quota/politeness, since it's a single cheap call per symbol per day).
    """
    try:
        data = _fetch_chart(ticker, f"{days}d", force=True)
        return _parse_bars(data)
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return []
