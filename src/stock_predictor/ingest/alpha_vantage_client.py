"""Shared HTTP client for every Alpha Vantage endpoint this app calls
(NEWS_SENTIMENT via ingest/sentiment.py, OVERVIEW via ingest/fundamentals.py,
INCOME_STATEMENT/EARNINGS via ingest/financials.py) -- one disk cache, one
DailyRateLimiter.

**Fixes a real bug**: each endpoint previously had its OWN independent
rate limiter/cache (ingest/sentiment.py's original DailyRateLimiter), which
would let two endpoints together exceed Alpha Vantage's real constraint --
the free tier's 25-requests/day cap is account-wide, not per-endpoint. A
shared module-level `_rate_limiter` here means every caller genuinely
respects one budget.

**Fixes a second real bug, confirmed live** -- Alpha Vantage's free tier
ALSO enforces an undocumented-in-our-code per-second burst limit
(confirmed via its own throttle response text: "consider spreading out
your free API requests more sparingly (1 request per second)"), separate
from the 25/day cap. A refresh run that fires several Alpha Vantage calls
back-to-back for one symbol (e.g. ingest/financials.py's two calls, then
ingest/fundamentals.py's OVERVIEW moments later) could collide with this
burst limit even while nowhere near the daily cap -- observed live as a
real OVERVIEW call for NVDA returning a throttle response 25 requests
into the day, not 25. Two fixes: `_pace_requests` enforces a minimum gap
between real HTTP calls, and `get` never caches a throttle/error response
(previously a transient burst collision would get baked into the disk
cache for the full `cache_ttl_seconds`, silently poisoning every call for
that ticker for up to a month).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from pathlib import Path

import requests

from stock_predictor.config import DATA_DIR, alpha_vantage_api_key

BASE_URL = "https://www.alphavantage.co/query"
CACHE_DIR = DATA_DIR / "cache" / "alpha_vantage"

MAX_REQUESTS_PER_DAY = 25

# Alpha Vantage's own stated burst limit is "1 request per second" -- a
# small margin above that (see module docstring for the real collision
# this fixes).
MIN_SECONDS_BETWEEN_REQUESTS = 1.2

# Alpha Vantage's own top-level keys for a throttle/error response instead
# of real data (confirmed live: {"Information": "...consider spreading
# out your free API requests..."} for a burst-limit hit; "Note" and
# "Error Message" are its other documented non-data response shapes).
_NON_DATA_RESPONSE_KEYS = ("Information", "Note", "Error Message")


class MissingApiKey(Exception):
    pass


class DailyRateLimiter:
    """Blocks just long enough to keep requests under N per rolling 24h --
    the daily-window analog of soccer-predictor's espn_client.py/
    api_client.py RateLimiter classes (both rolling-60s). A real
    blocking-sleep-until-tomorrow would be impractical for an interactive
    script, so this raises instead of sleeping once the daily cap is hit --
    callers treat that the same as any other failure: degrade to None,
    don't crash the whole refresh run.
    """

    def __init__(self, max_per_day: int = MAX_REQUESTS_PER_DAY):
        self.max_per_day = max_per_day
        self._timestamps: deque[float] = deque()

    def check(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > 24 * 3600:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_day:
            raise RuntimeError(f"Alpha Vantage daily request budget ({self.max_per_day}) exhausted")
        self._timestamps.append(now)


_rate_limiter = DailyRateLimiter()
_last_request_at: float | None = None


def _cache_path(function: str, params: dict) -> Path:
    key_input = function + ":" + json.dumps(params, sort_keys=True)
    key = hashlib.sha256(key_input.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _pace_requests() -> None:
    """Blocks just long enough to keep real HTTP calls at least
    MIN_SECONDS_BETWEEN_REQUESTS apart -- see module docstring for the
    real per-second burst-limit collision this prevents.
    """
    global _last_request_at
    now = time.monotonic()
    if _last_request_at is not None:
        elapsed = now - _last_request_at
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_at = time.monotonic()


def _is_non_data_response(data: dict) -> bool:
    return any(key in data for key in _NON_DATA_RESPONSE_KEYS)


def get(function: str, params: dict, cache_ttl_seconds: int) -> dict:
    """One Alpha Vantage call, sharing this module's disk cache + daily
    rate limiter + request pacing across every caller. `params` must NOT
    include `function` or `apikey` -- both are added here. Raises
    MissingApiKey/requests.RequestException/RuntimeError (budget
    exhausted) on failure; callers are expected to catch these and
    degrade to None, same contract every endpoint-specific client in this
    app already follows. A throttle/error response (see
    _NON_DATA_RESPONSE_KEYS) is returned as-is (callers already treat
    "no real data fields" as a normal empty result) but is deliberately
    NOT written to the disk cache, so the next call for the same ticker
    gets a genuine retry instead of replaying the failure for
    `cache_ttl_seconds`.
    """
    api_key = alpha_vantage_api_key()
    if not api_key:
        raise MissingApiKey(
            "ALPHA_VANTAGE_API_KEY not set -- copy .env.example to .env and fill it in "
            "(free registration at https://www.alphavantage.co/support/#api-key)"
        )

    cache_file = _cache_path(function, params)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < cache_ttl_seconds:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    _rate_limiter.check()
    _pace_requests()
    response = requests.get(
        BASE_URL,
        params={"function": function, "apikey": api_key, **params},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    if not _is_non_data_response(data):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data
