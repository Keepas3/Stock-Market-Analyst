"""Shared HTTP client for Finnhub's free-tier endpoints (stock/profile2 and
stock/metric via ingest/fundamentals.py) -- the analog of
ingest/alpha_vantage_client.py, but for a provider with a per-minute limit
instead of a per-day one.

Finnhub's free tier documents a 60-requests/minute cap and no hard daily
cap (unlike Alpha Vantage's 25/day), which is why it's used as the primary
source for fundamentals (see ingest/fundamentals.py) -- 16 watchlist
symbols x 2 calls (profile2 + metric) = 32 calls per full refresh run,
comfortably under budget even without spreading them out, but this module
paces requests anyway for the same politeness reason
alpha_vantage_client.py does.

Response shape differs from Alpha Vantage's: Finnhub returns `{}` for an
unknown symbol or a real HTTP 429 status when rate-limited, rather than a
documented "Information"/"Note"/"Error Message" key inside a 200 response.
Neither is cached, so a transient rate-limit collision gets a genuine
retry next time instead of being replayed for cache_ttl_seconds -- same
reasoning as alpha_vantage_client.py's own throttle-response handling.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from pathlib import Path

import requests

from stock_predictor.config import DATA_DIR, finnhub_api_key

BASE_URL = "https://finnhub.io/api/v1"
CACHE_DIR = DATA_DIR / "cache" / "finnhub"

# Finnhub's own documented free-tier cap is 60/minute -- a small margin
# below that, same idea as alpha_vantage_client.py's MIN_SECONDS_BETWEEN_REQUESTS.
MAX_REQUESTS_PER_MINUTE = 55
WINDOW_SECONDS = 60.0

MIN_SECONDS_BETWEEN_REQUESTS = WINDOW_SECONDS / MAX_REQUESTS_PER_MINUTE


class MissingApiKey(Exception):
    pass


class RollingWindowRateLimiter:
    """Blocks just long enough to keep requests under N per rolling window
    -- the short-window analog of alpha_vantage_client.py's DailyRateLimiter
    (24h window). Raises instead of sleeping-until-the-window-clears once
    the cap is hit, same contract: callers degrade to None, don't crash the
    whole refresh run.
    """

    def __init__(self, max_per_window: int = MAX_REQUESTS_PER_MINUTE, window_seconds: float = WINDOW_SECONDS):
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()

    def check(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] > self.window_seconds:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_window:
            raise RuntimeError(f"Finnhub rate limit ({self.max_per_window}/{self.window_seconds:.0f}s) exhausted")
        self._timestamps.append(now)


_rate_limiter = RollingWindowRateLimiter()
_last_request_at: float | None = None


def _cache_path(path: str, params: dict) -> Path:
    key_input = path + ":" + json.dumps(params, sort_keys=True)
    key = hashlib.sha256(key_input.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _pace_requests() -> None:
    global _last_request_at
    now = time.monotonic()
    if _last_request_at is not None:
        elapsed = now - _last_request_at
        if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_at = time.monotonic()


def _is_non_data_response(data: object, status_code: int) -> bool:
    return status_code == 429 or not isinstance(data, dict) or not data


def get(path: str, params: dict, cache_ttl_seconds: int) -> dict:
    """One Finnhub call, sharing this module's disk cache + rolling-window
    rate limiter + request pacing across every caller. `params` must NOT
    include `token` -- added here. Raises MissingApiKey/
    requests.RequestException/RuntimeError (rate limit exhausted) on
    failure; callers are expected to catch these and degrade to None, same
    contract as alpha_vantage_client.get. A non-data response (see
    _is_non_data_response) is returned as-is but deliberately NOT written
    to the disk cache.
    """
    api_key = finnhub_api_key()
    if not api_key:
        raise MissingApiKey(
            "FINNHUB_API_KEY not set -- copy .env.example to .env and fill it in "
            "(free registration at https://finnhub.io/register)"
        )

    cache_file = _cache_path(path, params)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < cache_ttl_seconds:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    _rate_limiter.check()
    _pace_requests()
    response = requests.get(
        f"{BASE_URL}/{path}",
        params={**params, "token": api_key},
        timeout=15,
    )
    if response.status_code != 429:
        response.raise_for_status()
    data = response.json() if response.content else {}

    if not _is_non_data_response(data, response.status_code):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data
