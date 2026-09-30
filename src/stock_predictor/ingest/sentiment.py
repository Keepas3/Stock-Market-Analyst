"""News sentiment via Alpha Vantage's NEWS_SENTIMENT endpoint -- the
direct analog of soccer-predictor's Understat/ASA current-form clients,
but for "reputation"/news instead of on-pitch stats.

Confirmed live (2026-09, using Alpha Vantage's public `demo` key): real,
dated articles with genuine per-article (`overall_sentiment_score`) AND
per-ticker (`ticker_sentiment[].ticker_sentiment_score`,
`relevance_score`) fields, roughly on a -1 (very bearish) to +1 (very
bullish) scale.

HTTP/caching/rate-limiting is shared with ingest/fundamentals.py's
OVERVIEW calls via ingest/alpha_vantage_client.py, since both draw from
the same account-wide 25-requests/day free-tier budget -- see that
module's own docstring for why a per-endpoint rate limiter would be wrong.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import requests

from stock_predictor.ingest import alpha_vantage_client

# One reading per symbol per day is the whole point -- a re-run within the
# same day (e.g. local dev iterating) must never re-spend the 25/day
# budget, so this TTL is deliberately a full day, not a few hours like
# price_history.py's.
CACHE_TTL_SECONDS = 24 * 3600

RECENT_ARTICLES_LIMIT = 5


@dataclass
class SentimentResult:
    ticker: str
    overall_sentiment_score: float  # relevance-weighted average across today's matching articles
    article_count: int


@dataclass
class NewsArticle:
    title: str
    url: str
    source: str
    published_at: dt.datetime | None
    summary: str


def _fetch_news_feed(ticker: str) -> list[dict]:
    data = alpha_vantage_client.get("NEWS_SENTIMENT", {"tickers": ticker}, CACHE_TTL_SECONDS)
    return data.get("feed", [])


def fetch_sentiment(ticker: str) -> SentimentResult | None:
    """Today's aggregated sentiment reading for `ticker` -- a relevance-
    weighted average of `ticker_sentiment_score` across every article in
    the feed that actually mentions this ticker (a NEWS_SENTIMENT response
    can include articles about related-but-different tickers too).
    Returns None on any failure (missing key, network error, daily budget
    exhausted, or genuinely zero matching articles) -- never raises.
    """
    try:
        feed = _fetch_news_feed(ticker)
    except (alpha_vantage_client.MissingApiKey, requests.RequestException, RuntimeError, ValueError):
        return None

    weighted_sum = 0.0
    weight_total = 0.0
    article_count = 0
    for article in feed:
        for entry in article.get("ticker_sentiment", []):
            if entry.get("ticker") != ticker:
                continue
            try:
                relevance = float(entry["relevance_score"])
                score = float(entry["ticker_sentiment_score"])
            except (KeyError, TypeError, ValueError):
                continue
            weighted_sum += score * relevance
            weight_total += relevance
            article_count += 1

    if weight_total <= 0:
        return None

    return SentimentResult(
        ticker=ticker,
        overall_sentiment_score=weighted_sum / weight_total,
        article_count=article_count,
    )


def _parse_time_published(raw: str | None) -> dt.datetime | None:
    """Alpha Vantage's own format: "20260929T210404" (UTC, no separators).
    None on anything else -- a display nicety, not worth failing the whole
    article over.
    """
    if not raw:
        return None
    try:
        return dt.datetime.strptime(raw, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def fetch_recent_articles(ticker: str, limit: int = RECENT_ARTICLES_LIMIT) -> list[NewsArticle]:
    """The `limit` most recently published articles that actually mention
    `ticker` (same relevance filter as fetch_sentiment), newest first --
    so a user can read the real reporting behind the aggregated sentiment
    score themselves (a future AI-summarization pass over these is a
    natural next step, not built here).

    Reuses fetch_sentiment's own _fetch_news_feed, which is disk-cached --
    calling both for the same ticker on the same day (as
    scripts/refresh_live_data.py does) costs at most ONE real Alpha
    Vantage request, not two. Empty list on any failure, same
    degrade-gracefully contract as fetch_sentiment.
    """
    try:
        feed = _fetch_news_feed(ticker)
    except (alpha_vantage_client.MissingApiKey, requests.RequestException, RuntimeError, ValueError):
        return []

    articles = []
    for article in feed:
        if not any(entry.get("ticker") == ticker for entry in article.get("ticker_sentiment", [])):
            continue
        title = article.get("title")
        url = article.get("url")
        if not title or not url:
            continue
        articles.append(
            NewsArticle(
                title=title,
                url=url,
                source=article.get("source") or "Unknown",
                published_at=_parse_time_published(article.get("time_published")),
                summary=article.get("summary") or "",
            )
        )

    articles.sort(key=lambda a: a.published_at or dt.datetime.min, reverse=True)
    return articles[:limit]
