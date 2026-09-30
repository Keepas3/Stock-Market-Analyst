"""News sentiment via Finnhub's `company-news` endpoint + local VADER
scoring -- the same "score raw text ourselves" pattern
ingest/social_sentiment.py already uses for StockTwits, applied to real
news headlines/summaries instead.

Replaces Alpha Vantage's `NEWS_SENTIMENT`, which had a hard account-wide
25-requests/day cap that made growing the watchlist past ~16-20 symbols
impractical (see config/watchlist.yaml's own former header comment).
Finnhub's free tier (60 requests/minute, no daily cap -- see
ingest/finnhub_client.py) has no per-ticker sentiment SCORE on its free
tier (`/news-sentiment` is a $50/month Premium feature, confirmed against
its docs), but its raw `company-news` endpoint (headlines/summaries, North
American companies) is free and, being a per-symbol endpoint, needs no
relevance filtering the way Alpha Vantage's multi-ticker feed did -- every
article it returns for a symbol is already about that symbol.

HTTP/caching/rate-limiting is shared with ingest/fundamentals.py's Finnhub
calls via ingest/finnhub_client.py, since both draw from the same
60-requests/minute free-tier budget.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from stock_predictor.ingest import finnhub_client

# One reading per symbol per day is the whole point -- a re-run within the
# same day (e.g. local dev iterating) must not spend extra Finnhub budget
# for the same result. Matches ingest/social_sentiment.py's own daily-
# reading intent, just a longer TTL since news (unlike StockTwits posts)
# doesn't meaningfully change within a day.
CACHE_TTL_SECONDS = 24 * 3600

# A short trailing window, not just "today" -- catches a reading on quiet
# news days without going stale, since Finnhub's company-news is an exact
# date-range query, not an implicit "recent" feed like Alpha Vantage's was.
LOOKBACK_DAYS = 2

RECENT_ARTICLES_LIMIT = 5

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class SentimentResult:
    ticker: str
    overall_sentiment_score: float  # VADER compound average across today's matching articles
    article_count: int


@dataclass
class NewsArticle:
    title: str
    url: str
    source: str
    published_at: dt.datetime | None
    summary: str


def _fetch_news_feed(ticker: str) -> list[dict]:
    today = dt.date.today()
    params = {
        "symbol": ticker,
        "from": (today - dt.timedelta(days=LOOKBACK_DAYS)).isoformat(),
        "to": today.isoformat(),
    }
    data = finnhub_client.get("company-news", params, CACHE_TTL_SECONDS)
    return data if isinstance(data, list) else []


def _score_text(headline: str, summary: str) -> float:
    text = f"{headline}. {summary}".strip()
    return _analyzer.polarity_scores(text)["compound"]


def fetch_sentiment(ticker: str) -> SentimentResult | None:
    """Today's (well, `LOOKBACK_DAYS`-trailing) aggregated sentiment
    reading for `ticker` -- a plain average of VADER's compound score
    across every article Finnhub returns for this symbol. Returns None on
    any failure (missing key, network error, rate limit exhausted, or
    genuinely zero articles) -- never raises.
    """
    try:
        feed = _fetch_news_feed(ticker)
    except (finnhub_client.MissingApiKey, requests.RequestException, RuntimeError, ValueError):
        return None

    scores = [_score_text(article["headline"], article.get("summary") or "") for article in feed if article.get("headline")]

    if not scores:
        return None

    return SentimentResult(
        ticker=ticker,
        overall_sentiment_score=sum(scores) / len(scores),
        article_count=len(scores),
    )


def _parse_published_at(raw: object) -> dt.datetime | None:
    """Finnhub's own format: a UNIX timestamp in seconds. None on anything
    else -- a display nicety, not worth failing the whole article over.
    """
    try:
        return dt.datetime.fromtimestamp(float(raw), tz=dt.UTC).replace(tzinfo=None)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def fetch_recent_articles(ticker: str, limit: int = RECENT_ARTICLES_LIMIT) -> list[NewsArticle]:
    """The `limit` most recently published articles for `ticker`, newest
    first -- so a user can read the real reporting behind the aggregated
    sentiment score themselves.

    Reuses fetch_sentiment's own _fetch_news_feed, which is disk-cached --
    calling both for the same ticker on the same day (as
    scripts/refresh_live_data.py does) costs at most ONE real Finnhub
    request, not two. Empty list on any failure, same degrade-gracefully
    contract as fetch_sentiment.
    """
    try:
        feed = _fetch_news_feed(ticker)
    except (finnhub_client.MissingApiKey, requests.RequestException, RuntimeError, ValueError):
        return []

    articles = []
    for article in feed:
        title = article.get("headline")
        url = article.get("url")
        if not title or not url:
            continue
        articles.append(
            NewsArticle(
                title=title,
                url=url,
                source=article.get("source") or "Unknown",
                published_at=_parse_published_at(article.get("datetime")),
                summary=article.get("summary") or "",
            )
        )

    articles.sort(key=lambda a: a.published_at or dt.datetime.min, reverse=True)
    return articles[:limit]
