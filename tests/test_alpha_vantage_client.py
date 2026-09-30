from __future__ import annotations

import pytest

from stock_predictor.ingest import alpha_vantage_client


@pytest.fixture(autouse=True)
def _reset_pacing_state(monkeypatch):
    """Resets the module-level pacing clock and no-ops real sleeping for
    every test in this file -- without this, two `get()` calls within one
    test (or across tests running close together in wall-clock time)
    would trigger a REAL ~1.2s sleep via _pace_requests, slowing the suite
    down for no reason. The dedicated pacing tests below re-monkeypatch
    time.sleep locally to capture calls instead of relying on this no-op.
    """
    monkeypatch.setattr(alpha_vantage_client, "_last_request_at", None)
    monkeypatch.setattr(alpha_vantage_client.time, "sleep", lambda seconds: None)


def test_daily_rate_limiter_raises_once_budget_exhausted():
    limiter = alpha_vantage_client.DailyRateLimiter(max_per_day=2)
    limiter.check()
    limiter.check()
    with pytest.raises(RuntimeError):
        limiter.check()


def test_get_raises_missing_api_key_when_unset(monkeypatch):
    monkeypatch.setattr(alpha_vantage_client, "alpha_vantage_api_key", lambda: None)
    with pytest.raises(alpha_vantage_client.MissingApiKey):
        alpha_vantage_client.get("OVERVIEW", {"symbol": "AAPL"}, 3600)


def test_get_serves_from_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(alpha_vantage_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(alpha_vantage_client, "alpha_vantage_api_key", lambda: "fake-key")
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return {"Symbol": "AAPL"}

    monkeypatch.setattr(alpha_vantage_client.requests, "get", lambda *a, **k: FakeResponse())

    first = alpha_vantage_client.get("OVERVIEW", {"symbol": "AAPL"}, 3600)
    second = alpha_vantage_client.get("OVERVIEW", {"symbol": "AAPL"}, 3600)

    assert first == second == {"Symbol": "AAPL"}
    assert len(calls) == 1  # second call served from disk cache


def test_get_different_functions_or_params_do_not_share_a_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(alpha_vantage_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(alpha_vantage_client, "alpha_vantage_api_key", lambda: "fake-key")

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    monkeypatch.setattr(
        alpha_vantage_client.requests, "get", lambda *a, **k: FakeResponse({"Symbol": k["params"]["symbol"]})
    )

    aapl = alpha_vantage_client.get("OVERVIEW", {"symbol": "AAPL"}, 3600)
    msft = alpha_vantage_client.get("OVERVIEW", {"symbol": "MSFT"}, 3600)

    assert aapl == {"Symbol": "AAPL"}
    assert msft == {"Symbol": "MSFT"}


def test_get_shares_one_rate_limiter_across_calls(tmp_path, monkeypatch):
    """The whole point of the refactor -- confirms a single module-level
    _rate_limiter is shared regardless of which `function` is requested,
    so two different Alpha Vantage endpoints genuinely draw from one
    account-wide budget, not two independent illusions of one.
    """
    monkeypatch.setattr(alpha_vantage_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(alpha_vantage_client, "alpha_vantage_api_key", lambda: "fake-key")
    monkeypatch.setattr(alpha_vantage_client, "_rate_limiter", alpha_vantage_client.DailyRateLimiter(max_per_day=1))

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    monkeypatch.setattr(alpha_vantage_client.requests, "get", lambda *a, **k: FakeResponse())

    alpha_vantage_client.get("NEWS_SENTIMENT", {"tickers": "AAPL"}, 3600)
    with pytest.raises(RuntimeError):
        alpha_vantage_client.get("OVERVIEW", {"symbol": "MSFT"}, 3600)


def test_get_does_not_cache_a_throttle_response(tmp_path, monkeypatch):
    """The real bug this fixes: a transient burst-limit collision was
    previously cached like real data, silently poisoning every call for
    that ticker for the full cache_ttl_seconds (confirmed live -- see
    module docstring). A throttle response must be retried, not replayed.
    """
    monkeypatch.setattr(alpha_vantage_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(alpha_vantage_client, "alpha_vantage_api_key", lambda: "fake-key")
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return {"Information": "...consider spreading out your free API requests...1 request per second"}

    monkeypatch.setattr(alpha_vantage_client.requests, "get", lambda *a, **k: FakeResponse())

    first = alpha_vantage_client.get("OVERVIEW", {"symbol": "AAPL"}, 3600)
    second = alpha_vantage_client.get("OVERVIEW", {"symbol": "AAPL"}, 3600)

    assert "Information" in first
    assert "Information" in second
    assert len(calls) == 2  # NOT served from cache on the second call


def test_get_still_caches_a_genuine_error_free_response(tmp_path, monkeypatch):
    monkeypatch.setattr(alpha_vantage_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(alpha_vantage_client, "alpha_vantage_api_key", lambda: "fake-key")
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return {"Symbol": "AAPL", "PERatio": "39.08"}

    monkeypatch.setattr(alpha_vantage_client.requests, "get", lambda *a, **k: FakeResponse())

    alpha_vantage_client.get("OVERVIEW", {"symbol": "AAPL"}, 3600)
    alpha_vantage_client.get("OVERVIEW", {"symbol": "AAPL"}, 3600)

    assert len(calls) == 1  # second call served from disk cache, as before


def test_pace_requests_does_not_sleep_on_the_first_call(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(alpha_vantage_client.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(alpha_vantage_client.time, "monotonic", lambda: 100.0)

    alpha_vantage_client._pace_requests()

    assert sleep_calls == []


def test_pace_requests_sleeps_when_calls_are_too_close(monkeypatch):
    fake_times = [100.0, 100.0, 100.05, 100.05]
    monkeypatch.setattr(alpha_vantage_client.time, "monotonic", lambda: fake_times.pop(0))
    sleep_calls = []
    monkeypatch.setattr(alpha_vantage_client.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    alpha_vantage_client._pace_requests()  # baseline call, no prior request to pace against
    alpha_vantage_client._pace_requests()  # 0.05s later -- must sleep the remainder

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(alpha_vantage_client.MIN_SECONDS_BETWEEN_REQUESTS - 0.05)


def test_pace_requests_does_not_sleep_when_calls_are_already_spaced_out(monkeypatch):
    fake_times = [100.0, 100.0, 105.0, 105.0]
    monkeypatch.setattr(alpha_vantage_client.time, "monotonic", lambda: fake_times.pop(0))
    sleep_calls = []
    monkeypatch.setattr(alpha_vantage_client.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    alpha_vantage_client._pace_requests()
    alpha_vantage_client._pace_requests()  # 5s later -- well past MIN_SECONDS_BETWEEN_REQUESTS

    assert sleep_calls == []
