from __future__ import annotations

import requests

from stock_predictor.ingest import social_sentiment


def _entry(body: str, tagged: str | None = None) -> dict:
    entry = {"body": body}
    if tagged is not None:
        entry["entities"] = {"sentiment": {"basic": tagged}}
    return entry


def test_fetch_recent_messages_parses_tagged_and_untagged(monkeypatch):
    raw = [_entry("Great quarter, stock is going to the moon!", "Bullish"), _entry("no tag here, just text")]
    monkeypatch.setattr(social_sentiment, "_fetch_message_feed", lambda ticker: raw)

    messages = social_sentiment.fetch_recent_messages("AAPL")

    assert len(messages) == 2
    assert messages[0].tagged_sentiment == "Bullish"
    assert messages[1].tagged_sentiment is None


def test_fetch_recent_messages_skips_entries_with_no_body(monkeypatch):
    raw = [{"entities": {}}, _entry("real message")]
    monkeypatch.setattr(social_sentiment, "_fetch_message_feed", lambda ticker: raw)

    messages = social_sentiment.fetch_recent_messages("AAPL")

    assert len(messages) == 1
    assert messages[0].body == "real message"


def test_fetch_recent_messages_returns_empty_list_on_request_exception(monkeypatch):
    def raise_error(ticker):
        raise requests.RequestException("network error")

    monkeypatch.setattr(social_sentiment, "_fetch_message_feed", raise_error)
    assert social_sentiment.fetch_recent_messages("AAPL") == []


def test_score_messages_returns_none_for_empty_list():
    assert social_sentiment.score_messages([]) is None


def test_score_messages_uses_tagged_sentiment_when_present():
    messages = [
        social_sentiment.SocialMessage(body="irrelevant text here", tagged_sentiment="Bullish"),
        social_sentiment.SocialMessage(body="irrelevant text here", tagged_sentiment="Bearish"),
    ]
    score, count = social_sentiment.score_messages(messages)

    assert score == 0.0  # +1 and -1 average to 0
    assert count == 2


def test_score_messages_falls_back_to_vader_for_untagged_posts():
    """Real VADER scoring (not mocked -- deterministic, fast, and this is
    exactly what this module claims to do: score raw, untagged text).
    """
    messages = [social_sentiment.SocialMessage(body="This is an amazing, wonderful, fantastic stock!", tagged_sentiment=None)]
    score, count = social_sentiment.score_messages(messages)

    assert count == 1
    assert score > 0.5  # VADER's compound score for unambiguously positive text


def test_score_messages_scores_negative_untagged_text_negatively():
    messages = [social_sentiment.SocialMessage(body="This is a terrible, awful, disastrous crash.", tagged_sentiment=None)]
    score, _ = social_sentiment.score_messages(messages)

    assert score < -0.5


def test_fetch_social_sentiment_returns_none_when_no_messages(monkeypatch):
    monkeypatch.setattr(social_sentiment, "fetch_recent_messages", lambda ticker: [])
    assert social_sentiment.fetch_social_sentiment("AAPL") is None


def test_fetch_social_sentiment_aggregates_into_result(monkeypatch):
    monkeypatch.setattr(
        social_sentiment,
        "fetch_recent_messages",
        lambda ticker: [social_sentiment.SocialMessage(body="x", tagged_sentiment="Bullish")],
    )

    result = social_sentiment.fetch_social_sentiment("AAPL")

    assert result is not None
    assert result.ticker == "AAPL"
    assert result.overall_sentiment_score == 1.0
    assert result.message_count == 1


def test_fetch_message_feed_serves_from_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(social_sentiment, "CACHE_DIR", tmp_path)
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return {"messages": [_entry("cached message")]}

    monkeypatch.setattr(social_sentiment.requests, "get", lambda *a, **k: FakeResponse())

    first = social_sentiment._fetch_message_feed("AAPL")
    second = social_sentiment._fetch_message_feed("AAPL")

    assert first == second
    assert len(calls) == 1  # second call served from disk cache
