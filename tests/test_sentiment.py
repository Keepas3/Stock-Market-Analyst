from __future__ import annotations

import datetime as dt

import requests

from stock_predictor.ingest import alpha_vantage_client, sentiment


def _feed(entries):
    """`entries` is a list of (ticker, relevance, score) tuples -- wraps
    each into one article with a single ticker_sentiment entry, real
    Alpha Vantage shape (both fields are strings on the wire).
    """
    return [
        {
            "ticker_sentiment": [
                {"ticker": ticker, "relevance_score": str(relevance), "ticker_sentiment_score": str(score)}
            ]
        }
        for ticker, relevance, score in entries
    ]


def test_fetch_sentiment_relevance_weighted_average(monkeypatch):
    feed = _feed([("AAPL", 1.0, 0.5), ("AAPL", 0.5, -0.1)])
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    result = sentiment.fetch_sentiment("AAPL")

    assert result is not None
    expected = (0.5 * 1.0 + -0.1 * 0.5) / (1.0 + 0.5)
    assert result.overall_sentiment_score == expected
    assert result.article_count == 2


def test_fetch_sentiment_ignores_articles_about_other_tickers(monkeypatch):
    feed = _feed([("AAPL", 1.0, 0.5), ("MSFT", 1.0, -0.9)])
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    result = sentiment.fetch_sentiment("AAPL")

    assert result.overall_sentiment_score == 0.5
    assert result.article_count == 1


def test_fetch_sentiment_returns_none_when_no_matching_articles(monkeypatch):
    feed = _feed([("MSFT", 1.0, 0.5)])
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    assert sentiment.fetch_sentiment("AAPL") is None


def test_fetch_sentiment_returns_none_on_empty_feed(monkeypatch):
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: [])
    assert sentiment.fetch_sentiment("AAPL") is None


def test_fetch_sentiment_returns_none_on_missing_api_key(monkeypatch):
    def raise_missing_key(ticker):
        raise alpha_vantage_client.MissingApiKey("no key")

    monkeypatch.setattr(sentiment, "_fetch_news_feed", raise_missing_key)
    assert sentiment.fetch_sentiment("AAPL") is None


def test_fetch_sentiment_returns_none_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(sentiment, "_fetch_news_feed", raise_error)
    assert sentiment.fetch_sentiment("AAPL") is None


def test_fetch_sentiment_skips_malformed_entries(monkeypatch):
    feed = [{"ticker_sentiment": [{"ticker": "AAPL", "relevance_score": "not-a-number", "ticker_sentiment_score": "0.5"}]}]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    assert sentiment.fetch_sentiment("AAPL") is None


def _article(ticker, title="Title", url="https://example.com/a", source="Reuters", time_published=None, summary="Summary"):
    entry = {
        "title": title,
        "url": url,
        "source": source,
        "summary": summary,
        "ticker_sentiment": [{"ticker": ticker, "relevance_score": "1.0", "ticker_sentiment_score": "0.5"}],
    }
    if time_published is not None:
        entry["time_published"] = time_published
    return entry


def test_fetch_recent_articles_sorts_newest_first(monkeypatch):
    feed = [
        _article("AAPL", title="Older", time_published="20260101T090000"),
        _article("AAPL", title="Newest", time_published="20260301T090000"),
        _article("AAPL", title="Middle", time_published="20260201T090000"),
    ]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL")

    assert [a.title for a in articles] == ["Newest", "Middle", "Older"]


def test_fetch_recent_articles_respects_limit(monkeypatch):
    feed = [_article("AAPL", title=f"Article {i}", time_published=f"2026010{i}T090000") for i in range(1, 8)]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL", limit=5)

    assert len(articles) == 5


def test_fetch_recent_articles_ignores_articles_about_other_tickers(monkeypatch):
    feed = [_article("MSFT", title="Not AAPL", time_published="20260101T090000")]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    assert sentiment.fetch_recent_articles("AAPL") == []


def test_fetch_recent_articles_skips_entries_missing_title_or_url(monkeypatch):
    feed = [
        _article("AAPL", title="", time_published="20260101T090000"),
        _article("AAPL", title="Has both", time_published="20260102T090000"),
    ]
    feed[0]["url"] = ""
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL")

    assert len(articles) == 1
    assert articles[0].title == "Has both"


def test_fetch_recent_articles_parses_real_alpha_vantage_time_format(monkeypatch):
    feed = [_article("AAPL", time_published="20260929T210404")]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL")

    assert articles[0].published_at == dt.datetime(2026, 9, 29, 21, 4, 4)


def test_fetch_recent_articles_handles_missing_time_published(monkeypatch):
    feed = [_article("AAPL")]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL")

    assert len(articles) == 1
    assert articles[0].published_at is None


def test_fetch_recent_articles_returns_empty_list_on_missing_api_key(monkeypatch):
    def raise_missing_key(ticker):
        raise alpha_vantage_client.MissingApiKey("no key")

    monkeypatch.setattr(sentiment, "_fetch_news_feed", raise_missing_key)
    assert sentiment.fetch_recent_articles("AAPL") == []


def test_fetch_news_feed_delegates_to_shared_alpha_vantage_client(monkeypatch):
    captured = {}

    def fake_get(function, params, cache_ttl_seconds):
        captured["function"] = function
        captured["params"] = params
        captured["cache_ttl_seconds"] = cache_ttl_seconds
        return {"feed": _feed([("AAPL", 1.0, 0.5)])}

    monkeypatch.setattr(alpha_vantage_client, "get", fake_get)

    feed = sentiment._fetch_news_feed("AAPL")

    assert captured["function"] == "NEWS_SENTIMENT"
    assert captured["params"] == {"tickers": "AAPL"}
    assert captured["cache_ttl_seconds"] == sentiment.CACHE_TTL_SECONDS
    assert len(feed) == 1
