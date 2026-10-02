"""Social-media ("reputation") sentiment via StockTwits' public symbol
stream -- keyless, confirmed live (2026-09), no documented rate limit.

**Real gotcha, confirmed live:** a bare default `requests` User-Agent gets
a 403 Forbidden -- same bot-challenge behavior as Yahoo Finance's chart
endpoint (see ingest/price_history.py's own module docstring), fixed the
same way, with the same minimal `Mozilla/5.0` header.

Some posts already carry a user-self-tagged `entities.sentiment.basic`
field (`"Bullish"`/`"Bearish"`) -- used where present. Everything else is
scored locally with VADER (a free, rule-based sentiment lexicon
well-suited to short informal text), so every post contributes to the
average, not just the self-tagged minority. Kept as its own module/table,
separate from ingest/sentiment.py's news pipeline -- news and social are
two distinct signals worth showing distinctly, not one blended number.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stock_predictor.config import DATA_DIR

BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol"
CACHE_DIR = DATA_DIR / "cache" / "social_sentiment"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# StockTwits publishes no documented rate limit for this endpoint -- a
# short cache TTL is a self-imposed courtesy, same reasoning as
# soccer-predictor's espn_client.py RateLimiter for an undocumented source.
CACHE_TTL_SECONDS = 3600

# Sample size for fetch_recent_posts -- display-only, same "top N" idea as
# ingest/sentiment.py's RECENT_ARTICLES_LIMIT, independent of
# message_count (every fetched message still counts toward the aggregate
# score via fetch_recent_messages/score_messages).
RECENT_POSTS_LIMIT = 3

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class SocialMessage:
    body: str
    tagged_sentiment: str | None  # "Bullish" | "Bearish" | None (untagged)
    # Display-only fields, confirmed present in StockTwits' live raw
    # response (id, user.username, created_at) but not needed by
    # score_messages -- default to None so existing construction sites
    # (none, as of this addition, besides fetch_recent_messages itself)
    # don't need updating.
    external_id: int | None = None
    username: str | None = None
    created_at: dt.datetime | None = None


@dataclass
class SocialSentimentResult:
    ticker: str
    overall_sentiment_score: float  # -1..+1, averaged across message_count messages
    message_count: int


def _cache_path(ticker: str) -> Path:
    key = hashlib.sha256(ticker.encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _fetch_message_feed(ticker: str) -> list[dict]:
    cache_file = _cache_path(ticker)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    response = requests.get(f"{BASE_URL}/{ticker}.json", headers=HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()
    messages = data.get("messages", [])

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(messages), encoding="utf-8")
    return messages


def _parse_created_at(value: str | None) -> dt.datetime | None:
    """StockTwits timestamps look like "2026-09-30T07:22:37Z" -- parsed to
    a naive UTC datetime, same convention as every other `fetched_at` in
    this app (see storage/repository.py's `dt.datetime.now(dt.UTC).replace
    (tzinfo=None)` idiom). None on any unexpected shape.
    """
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def fetch_recent_messages(ticker: str) -> list[SocialMessage]:
    """Empty list on any failure (network error, unknown ticker, unexpected
    shape) -- same degrade-gracefully contract as every other ingest
    function in this app.
    """
    try:
        raw = _fetch_message_feed(ticker)
    except (requests.RequestException, ValueError, KeyError):
        return []

    parsed = []
    for entry in raw:
        body = entry.get("body")
        if not body:
            continue
        sentiment_block = (entry.get("entities") or {}).get("sentiment") or {}
        parsed.append(
            SocialMessage(
                body=body,
                tagged_sentiment=sentiment_block.get("basic"),
                external_id=entry.get("id"),
                username=(entry.get("user") or {}).get("username"),
                created_at=_parse_created_at(entry.get("created_at")),
            )
        )
    return parsed


def fetch_recent_posts(ticker: str, limit: int = RECENT_POSTS_LIMIT) -> list[SocialMessage]:
    """Newest-first sample of the same messages fetch_recent_messages
    scores, for display (see
    dashboard/components.py::render_recent_social_posts) -- same "let the
    user see the real data behind one aggregated number" purpose as
    ingest/sentiment.py::fetch_recent_articles. Messages with an
    unparseable/missing created_at sort last, same idiom as
    fetch_recent_articles' own published_at handling.
    """
    messages = fetch_recent_messages(ticker)
    messages.sort(key=lambda m: m.created_at or dt.datetime.min, reverse=True)
    return messages[:limit]


def score_messages(messages: list[SocialMessage]) -> tuple[float, int] | None:
    """(overall_sentiment_score, message_count) -- None for an empty list
    ("no reading available," same contract as
    ingest/sentiment.py::fetch_sentiment returning None for zero matching
    articles).
    """
    if not messages:
        return None
    scores = [
        1.0
        if message.tagged_sentiment == "Bullish"
        else -1.0
        if message.tagged_sentiment == "Bearish"
        else _analyzer.polarity_scores(message.body)["compound"]
        for message in messages
    ]
    return sum(scores) / len(scores), len(scores)


def fetch_social_sentiment(ticker: str) -> SocialSentimentResult | None:
    scored = score_messages(fetch_recent_messages(ticker))
    if scored is None:
        return None
    score, count = scored
    return SocialSentimentResult(ticker=ticker, overall_sentiment_score=score, message_count=count)
