from __future__ import annotations

import requests

from stock_predictor.ingest import finnhub_client, quote


def test_fetch_quote_parses_real_shaped_response(monkeypatch):
    monkeypatch.setattr(
        finnhub_client,
        "get",
        lambda *a, **k: {"c": 198.42, "h": 199.0, "l": 197.0, "o": 197.5, "pc": 195.0, "t": 123},
    )

    result = quote.fetch_quote("AAPL")

    assert result is not None
    assert result.ticker == "AAPL"
    assert result.current_price == 198.42
    assert result.previous_close == 195.0


def test_fetch_quote_returns_none_when_current_price_is_zero(monkeypatch):
    """Finnhub's own sentinel for an unknown symbol -- c: 0."""
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: {"c": 0, "pc": 0})
    assert quote.fetch_quote("NOPE") is None


def test_fetch_quote_returns_none_when_c_key_is_missing(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: {})
    assert quote.fetch_quote("NOPE") is None


def test_fetch_quote_handles_missing_previous_close(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: {"c": 100.0})
    result = quote.fetch_quote("AAPL")
    assert result is not None
    assert result.current_price == 100.0
    assert result.previous_close is None


def test_fetch_quote_returns_none_on_missing_api_key(monkeypatch):
    def raise_missing_key(*a, **k):
        raise finnhub_client.MissingApiKey("no key")

    monkeypatch.setattr(finnhub_client, "get", raise_missing_key)
    assert quote.fetch_quote("AAPL") is None


def test_fetch_quote_returns_none_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(finnhub_client, "get", raise_error)
    assert quote.fetch_quote("AAPL") is None


def test_fetch_quote_returns_none_on_rate_limit_exhausted(monkeypatch):
    def raise_error(*a, **k):
        raise RuntimeError("rate limit exhausted")

    monkeypatch.setattr(finnhub_client, "get", raise_error)
    assert quote.fetch_quote("AAPL") is None


def test_fetch_quote_returns_none_for_non_dict_response(monkeypatch):
    monkeypatch.setattr(finnhub_client, "get", lambda *a, **k: [])
    assert quote.fetch_quote("AAPL") is None


def test_fetch_quote_uses_short_cache_ttl(monkeypatch):
    captured = {}

    def fake_get(path, params, cache_ttl_seconds):
        captured["path"] = path
        captured["params"] = params
        captured["cache_ttl_seconds"] = cache_ttl_seconds
        return {"c": 100.0}

    monkeypatch.setattr(finnhub_client, "get", fake_get)

    quote.fetch_quote("AAPL")

    assert captured["path"] == "quote"
    assert captured["params"] == {"symbol": "AAPL"}
    assert captured["cache_ttl_seconds"] == quote.CACHE_TTL_SECONDS
    assert quote.CACHE_TTL_SECONDS == 60
