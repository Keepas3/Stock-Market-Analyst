from __future__ import annotations

import pytest
import requests

from stock_predictor.ingest import alpha_vantage_client, finnhub_client, fundamentals


@pytest.fixture(autouse=True)
def _finnhub_unset_by_default(monkeypatch):
    """Every test below that doesn't override this stubs Finnhub as
    "no API key" -- the real state for anyone who hasn't set FINNHUB_API_KEY
    yet -- so the existing Alpha-Vantage-only tests keep exercising that
    fallback path in isolation, independent of whatever's in this machine's
    actual environment.
    """

    def raise_missing_key(*a, **k):
        raise finnhub_client.MissingApiKey("no key")

    monkeypatch.setattr(finnhub_client, "get", raise_missing_key)


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


def _stub_alpha_vantage(monkeypatch, **overrides):
    data = {
        "Symbol": "AAPL",
        "PERatio": "39.08",
        "ForwardPE": "35.21",
        "Sector": "TECHNOLOGY",
        "Name": "Apple Inc (AV)",
        "Description": "Apple designs, manufactures, and markets smartphones...",
        "Industry": "COMPUTER HARDWARE (AV)",
        "Exchange": "NASDAQ",
        "Country": "USA",
        "Address": "ONE APPLE PARK WAY, CUPERTINO, CA, US",
        "OfficialSite": "https://www.apple.com",
        "MarketCapitalization": "3200000000000",
    }
    data.update(overrides)
    monkeypatch.setattr(alpha_vantage_client, "get", lambda *a, **k: data)


def _stub_finnhub(monkeypatch, profile=None, metric=None):
    profile_data = {
        "name": "Apple Inc (Finnhub)",
        "finnhubIndustry": "Technology",
        "exchange": "NASDAQ NMS - GLOBAL MARKET",
        "country": "US",
        "weburl": "https://www.apple.com/finnhub",
        "marketCapitalization": 3_300_000.0,  # millions -> $3.3T
    }
    if profile is not None:
        profile_data = profile
    metric_data = {"peBasicExclExtraTTM": 40.5}
    if metric is not None:
        metric_data = metric

    def fake_get(path, params, cache_ttl_seconds):
        if path == "stock/profile2":
            return profile_data
        assert path == "stock/metric"
        return {"metric": metric_data}

    monkeypatch.setattr(finnhub_client, "get", fake_get)


def test_fetch_overview_prefers_finnhub_pe_when_available(monkeypatch):
    _stub_alpha_vantage(monkeypatch)
    _stub_finnhub(monkeypatch)

    result = fundamentals.fetch_overview("AAPL")

    assert result.pe_ratio == 40.5
    assert result.market_cap == 3_300_000.0 * fundamentals._FINNHUB_MARKET_CAP_SCALE
    assert result.name == "Apple Inc (Finnhub)"
    assert result.industry == "Technology"


def test_fetch_overview_falls_back_to_alpha_vantage_pe_when_finnhub_has_none(monkeypatch):
    _stub_alpha_vantage(monkeypatch)
    _stub_finnhub(monkeypatch, metric={})

    result = fundamentals.fetch_overview("AAPL")

    assert result.pe_ratio == 39.08
    assert result.forward_pe == 35.21


def test_fetch_overview_takes_description_from_alpha_vantage_even_when_finnhub_succeeds(monkeypatch):
    """Regression test: Finnhub's profile2 has no description field, and
    dashboard/components.py::render_company_background hides the entire
    "About" section without one. Finnhub succeeding for P/E must never
    blank out the description that Alpha Vantage still provides.
    """
    _stub_alpha_vantage(monkeypatch)
    _stub_finnhub(monkeypatch)

    result = fundamentals.fetch_overview("AAPL")

    assert result.description == "Apple designs, manufactures, and markets smartphones..."
    assert result.address == "ONE APPLE PARK WAY, CUPERTINO, CA, US"


def test_fetch_overview_returns_none_when_both_sources_fail(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(alpha_vantage_client, "get", raise_error)
    monkeypatch.setattr(finnhub_client, "get", raise_error)

    assert fundamentals.fetch_overview("AAPL") is None


def test_fetch_overview_works_with_finnhub_api_key_unset(monkeypatch):
    """The autouse _finnhub_unset_by_default fixture already covers this,
    but this test makes the backward-compatibility contract explicit: a
    user who hasn't added FINNHUB_API_KEY yet still gets a full result via
    Alpha Vantage alone, unchanged from pre-Finnhub behavior.
    """
    _stub_alpha_vantage(monkeypatch)

    result = fundamentals.fetch_overview("AAPL")

    assert result.pe_ratio == 39.08
    assert result.description == "Apple designs, manufactures, and markets smartphones..."
