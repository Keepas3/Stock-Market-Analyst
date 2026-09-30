from __future__ import annotations

import datetime as dt

import requests

from stock_predictor.ingest import finnhub_client, sentiment


def _article(headline="Great news", summary="Very positive outlook", url="https://example.com/a", source="Reuters", timestamp=None):
    entry = {"headline": headline, "summary": summary, "url": url, "source": source}
    if timestamp is not None:
        entry["datetime"] = timestamp
    return entry


def test_fetch_sentiment_averages_vader_scores_across_articles(monkeypatch):
    feed = [
        _article(headline="This is wonderful, fantastic news", summary=""),
        _article(headline="This is terrible, awful news", summary=""),
    ]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    result = sentiment.fetch_sentiment("AAPL")

    assert result is not None
    assert result.article_count == 2
    # One clearly positive + one clearly negative article should roughly cancel out.
    assert -0.3 < result.overall_sentiment_score < 0.3


def test_fetch_sentiment_scores_unambiguously_positive_text_positively():
    """Real VADER scoring (not mocked), same reasoning as
    test_social_sentiment.py's own real-scoring tests.
    """
    feed = [_article(headline="Amazing, wonderful, fantastic quarter", summary="Stock surges on incredible results")]

    score = sentiment._score_text(feed[0]["headline"], feed[0]["summary"])

    assert score > 0.5


def test_fetch_sentiment_returns_none_on_empty_feed(monkeypatch):
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: [])
    assert sentiment.fetch_sentiment("AAPL") is None


def test_fetch_sentiment_skips_articles_with_no_headline(monkeypatch):
    feed = [{"summary": "no headline here", "url": "https://example.com/a"}]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    assert sentiment.fetch_sentiment("AAPL") is None


def test_fetch_sentiment_returns_none_on_missing_api_key(monkeypatch):
    def raise_missing_key(ticker):
        raise finnhub_client.MissingApiKey("no key")

    monkeypatch.setattr(sentiment, "_fetch_news_feed", raise_missing_key)
    assert sentiment.fetch_sentiment("AAPL") is None


def test_fetch_sentiment_returns_none_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(sentiment, "_fetch_news_feed", raise_error)
    assert sentiment.fetch_sentiment("AAPL") is None


def test_fetch_recent_articles_sorts_newest_first(monkeypatch):
    feed = [
        _article(headline="Older", timestamp=1_700_000_000),
        _article(headline="Newest", timestamp=1_750_000_000),
        _article(headline="Middle", timestamp=1_725_000_000),
    ]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL")

    assert [a.title for a in articles] == ["Newest", "Middle", "Older"]


def test_fetch_recent_articles_respects_limit(monkeypatch):
    feed = [_article(headline=f"Article {i}", timestamp=1_700_000_000 + i) for i in range(7)]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL", limit=5)

    assert len(articles) == 5


def test_fetch_recent_articles_skips_entries_missing_title_or_url(monkeypatch):
    feed = [
        _article(headline="", timestamp=1_700_000_000),
        _article(headline="Has both", timestamp=1_700_000_001),
    ]
    feed[0]["url"] = ""
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL")

    assert len(articles) == 1
    assert articles[0].title == "Has both"


def test_fetch_recent_articles_parses_unix_timestamp(monkeypatch):
    feed = [_article(headline="Real article", timestamp=1_727_654_644)]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL")

    assert articles[0].published_at == dt.datetime.fromtimestamp(1_727_654_644, tz=dt.UTC).replace(tzinfo=None)


def test_fetch_recent_articles_handles_missing_datetime(monkeypatch):
    feed = [_article(headline="No timestamp")]
    monkeypatch.setattr(sentiment, "_fetch_news_feed", lambda ticker: feed)

    articles = sentiment.fetch_recent_articles("AAPL")

    assert len(articles) == 1
    assert articles[0].published_at is None


def test_fetch_recent_articles_returns_empty_list_on_missing_api_key(monkeypatch):
    def raise_missing_key(ticker):
        raise finnhub_client.MissingApiKey("no key")

    monkeypatch.setattr(sentiment, "_fetch_news_feed", raise_missing_key)
    assert sentiment.fetch_recent_articles("AAPL") == []


def test_fetch_news_feed_delegates_to_shared_finnhub_client(monkeypatch):
    captured = {}

    def fake_get(path, params, cache_ttl_seconds):
        captured["path"] = path
        captured["params"] = params
        captured["cache_ttl_seconds"] = cache_ttl_seconds
        return [_article()]

    monkeypatch.setattr(finnhub_client, "get", fake_get)

    feed = sentiment._fetch_news_feed("AAPL")

    assert captured["path"] == "company-news"
    assert captured["params"]["symbol"] == "AAPL"
    assert "from" in captured["params"] and "to" in captured["params"]
    assert captured["cache_ttl_seconds"] == sentiment.CACHE_TTL_SECONDS
    assert len(feed) == 1


def test_fetch_news_feed_returns_empty_list_for_non_list_response(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: {})
    assert sentiment._fetch_news_feed("AAPL") == []
