from __future__ import annotations

import requests

from stock_predictor.ingest import alpha_vantage_client, fundamentals


def test_fetch_overview_parses_real_shaped_response(monkeypatch):
    data = {
        "Symbol": "AAPL",
        "PERatio": "39.08",
        "ForwardPE": "35.21",
        "Sector": "TECHNOLOGY",
        "Name": "Apple Inc",
        "Description": "Apple designs, manufactures, and markets smartphones...",
        "Industry": "COMPUTER HARDWARE",
        "Exchange": "NASDAQ",
        "Country": "USA",
        "Address": "ONE APPLE PARK WAY, CUPERTINO, CA, US",
        "OfficialSite": "https://www.apple.com",
    }
    monkeypatch.setattr(alpha_vantage_client, "get", lambda *a, **k: data)

    result = fundamentals.fetch_overview("AAPL")

    assert result is not None
    assert result.ticker == "AAPL"
    assert result.pe_ratio == 39.08
    assert result.forward_pe == 35.21
    assert result.sector == "TECHNOLOGY"
    assert result.name == "Apple Inc"
    assert result.description.startswith("Apple designs")
    assert result.industry == "COMPUTER HARDWARE"
    assert result.exchange == "NASDAQ"
    assert result.country == "USA"
    assert result.address == "ONE APPLE PARK WAY, CUPERTINO, CA, US"
    assert result.official_site == "https://www.apple.com"


def test_fetch_overview_handles_none_sentinel_pe(monkeypatch):
    data = {
        "Symbol": "XYZ",
        "PERatio": "None",
        "ForwardPE": "None",
        "Sector": "None",
        "Name": "None",
        "Description": "None",
        "Industry": "None",
        "Exchange": "None",
        "Country": "None",
        "Address": "None",
        "OfficialSite": "None",
    }
    monkeypatch.setattr(alpha_vantage_client, "get", lambda *a, **k: data)

    result = fundamentals.fetch_overview("XYZ")

    assert result is not None
    assert result.pe_ratio is None
    assert result.forward_pe is None
    assert result.sector is None
    assert result.name is None
    assert result.description is None
    assert result.industry is None
    assert result.exchange is None
    assert result.country is None
    assert result.address is None
    assert result.official_site is None


def test_fetch_overview_missing_profile_fields_are_none(monkeypatch):
    data = {"Symbol": "AAPL", "PERatio": "10", "ForwardPE": "10"}
    monkeypatch.setattr(alpha_vantage_client, "get", lambda *a, **k: data)

    result = fundamentals.fetch_overview("AAPL")

    assert result.name is None
    assert result.description is None
    assert result.official_site is None


def test_fetch_overview_returns_none_when_no_symbol_field(monkeypatch):
    monkeypatch.setattr(alpha_vantage_client, "get", lambda *a, **k: {})
    assert fundamentals.fetch_overview("NOPE") is None


def test_fetch_overview_returns_none_on_missing_api_key(monkeypatch):
    def raise_missing_key(*a, **k):
        raise alpha_vantage_client.MissingApiKey("no key")

    monkeypatch.setattr(alpha_vantage_client, "get", raise_missing_key)
    assert fundamentals.fetch_overview("AAPL") is None


def test_fetch_overview_returns_none_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(alpha_vantage_client, "get", raise_error)
    assert fundamentals.fetch_overview("AAPL") is None


def test_fetch_overview_uses_weekly_cache_ttl(monkeypatch):
    captured = {}

    def fake_get(function, params, cache_ttl_seconds):
        captured["function"] = function
        captured["params"] = params
        captured["cache_ttl_seconds"] = cache_ttl_seconds
        return {"Symbol": "AAPL", "PERatio": "10", "ForwardPE": "10"}

    monkeypatch.setattr(alpha_vantage_client, "get", fake_get)

    fundamentals.fetch_overview("AAPL")

    assert captured["function"] == "OVERVIEW"
    assert captured["params"] == {"symbol": "AAPL"}
    assert captured["cache_ttl_seconds"] == fundamentals.CACHE_TTL_SECONDS
    assert fundamentals.CACHE_TTL_SECONDS == 7 * 24 * 3600
