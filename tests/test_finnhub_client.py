from __future__ import annotations

import pytest

from stock_predictor.ingest import finnhub_client


@pytest.fixture(autouse=True)
def _reset_pacing_state(monkeypatch):
    """Same reasoning as test_alpha_vantage_client.py's own fixture --
    without this, calls within a test would trigger a real ~1.1s sleep via
    _pace_requests.
    """
    monkeypatch.setattr(finnhub_client, "_last_request_at", None)
    monkeypatch.setattr(finnhub_client.time, "sleep", lambda seconds: None)


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.content = b"1" if payload is not None else b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise finnhub_client.requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self.payload


def test_rolling_window_rate_limiter_raises_once_exhausted():
    limiter = finnhub_client.RollingWindowRateLimiter(max_per_window=2, window_seconds=60)
    limiter.check()
    limiter.check()
    with pytest.raises(RuntimeError):
        limiter.check()


def test_rolling_window_rate_limiter_resets_after_window_rolls_off(monkeypatch):
    fake_times = [0.0, 61.0]
    monkeypatch.setattr(finnhub_client.time, "monotonic", lambda: fake_times.pop(0))

    limiter = finnhub_client.RollingWindowRateLimiter(max_per_window=1, window_seconds=60)
    limiter.check()
    # 61s later -- the first timestamp has rolled off the 60s window.
    limiter.check()


def test_get_raises_missing_api_key_when_unset(monkeypatch):
    monkeypatch.setattr(finnhub_client, "finnhub_api_key", lambda: None)
    with pytest.raises(finnhub_client.MissingApiKey):
        finnhub_client.get("stock/profile2", {"symbol": "AAPL"}, 3600)


def test_get_serves_from_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(finnhub_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finnhub_client, "finnhub_api_key", lambda: "fake-key")
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return FakeResponse({"name": "Apple Inc"})

    monkeypatch.setattr(finnhub_client.requests, "get", fake_get)

    first = finnhub_client.get("stock/profile2", {"symbol": "AAPL"}, 3600)
    second = finnhub_client.get("stock/profile2", {"symbol": "AAPL"}, 3600)

    assert first == second == {"name": "Apple Inc"}
    assert len(calls) == 1  # second call served from disk cache


def test_get_different_paths_or_params_do_not_share_a_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(finnhub_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finnhub_client, "finnhub_api_key", lambda: "fake-key")

    monkeypatch.setattr(
        finnhub_client.requests, "get", lambda url, params, timeout: FakeResponse({"symbol": params["symbol"]})
    )

    aapl = finnhub_client.get("stock/profile2", {"symbol": "AAPL"}, 3600)
    msft = finnhub_client.get("stock/profile2", {"symbol": "MSFT"}, 3600)

    assert aapl == {"symbol": "AAPL"}
    assert msft == {"symbol": "MSFT"}


def test_get_does_not_cache_an_empty_dict_response(tmp_path, monkeypatch):
    monkeypatch.setattr(finnhub_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finnhub_client, "finnhub_api_key", lambda: "fake-key")
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return FakeResponse({})

    monkeypatch.setattr(finnhub_client.requests, "get", fake_get)

    first = finnhub_client.get("stock/profile2", {"symbol": "NOPE"}, 3600)
    second = finnhub_client.get("stock/profile2", {"symbol": "NOPE"}, 3600)

    assert first == second == {}
    assert len(calls) == 2  # NOT served from cache on the second call


def test_get_does_not_cache_a_429_response(tmp_path, monkeypatch):
    monkeypatch.setattr(finnhub_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finnhub_client, "finnhub_api_key", lambda: "fake-key")
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return FakeResponse({"error": "limit exceeded"}, status_code=429)

    monkeypatch.setattr(finnhub_client.requests, "get", fake_get)

    finnhub_client.get("stock/metric", {"symbol": "AAPL", "metric": "all"}, 3600)
    finnhub_client.get("stock/metric", {"symbol": "AAPL", "metric": "all"}, 3600)

    assert len(calls) == 2  # NOT served from cache


def test_get_still_caches_a_genuine_response(tmp_path, monkeypatch):
    monkeypatch.setattr(finnhub_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(finnhub_client, "finnhub_api_key", lambda: "fake-key")
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return FakeResponse({"metric": {"peBasicExclExtraTTM": 39.08}})

    monkeypatch.setattr(finnhub_client.requests, "get", fake_get)

    finnhub_client.get("stock/metric", {"symbol": "AAPL", "metric": "all"}, 3600)
    finnhub_client.get("stock/metric", {"symbol": "AAPL", "metric": "all"}, 3600)

    assert len(calls) == 1  # second call served from disk cache
