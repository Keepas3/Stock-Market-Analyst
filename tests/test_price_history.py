from __future__ import annotations

import datetime as dt

import requests

from stock_predictor.ingest import price_history


def _chart_response(timestamps, opens, highs, lows, closes, volumes, adjcloses=None):
    return {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [{"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}],
                        "adjclose": [{"adjclose": adjcloses if adjcloses is not None else closes}],
                    },
                }
            ],
            "error": None,
        }
    }


def test_parse_bars_real_shaped_response():
    data = _chart_response(
        timestamps=[1758758400, 1758844800],
        opens=[100.0, 101.0],
        highs=[102.0, 103.0],
        lows=[99.0, 100.0],
        closes=[101.0, 102.0],
        volumes=[1000, 1200],
        adjcloses=[100.5, 101.5],
    )

    bars = price_history._parse_bars(data)

    assert len(bars) == 2
    assert bars[0].date == dt.date(2025, 9, 25)
    assert bars[0].open == 100.0
    assert bars[0].close == 100.5  # adjusted close, not raw close
    assert bars[0].volume == 1000


def test_parse_bars_skips_null_gap_day():
    data = _chart_response(
        timestamps=[1758758400, 1758844800],
        opens=[100.0, None],
        highs=[102.0, None],
        lows=[99.0, None],
        closes=[101.0, None],
        volumes=[1000, None],
        adjcloses=[100.5, None],
    )

    bars = price_history._parse_bars(data)

    assert len(bars) == 1
    assert bars[0].date == dt.date(2025, 9, 25)


def test_parse_bars_empty_on_missing_result():
    assert price_history._parse_bars({"chart": {"result": None, "error": {"code": "Not Found"}}}) == []


def test_parse_bars_falls_back_to_raw_close_when_no_adjclose():
    data = {
        "chart": {
            "result": [
                {
                    "timestamp": [1758758400],
                    "indicators": {
                        "quote": [{"open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0], "volume": [1000]}],
                        "adjclose": [{}],
                    },
                }
            ]
        }
    }
    bars = price_history._parse_bars(data)
    assert bars[0].close == 101.0


def test_fetch_full_history_returns_empty_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(price_history, "_fetch_chart", raise_error)
    assert price_history.fetch_full_history("AAPL") == []


def test_fetch_full_history_returns_empty_on_malformed_response(monkeypatch):
    monkeypatch.setattr(price_history, "_fetch_chart", lambda *a, **k: {"unexpected": "shape"})
    assert price_history.fetch_full_history("AAPL") == []


def test_fetch_chart_serves_from_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(price_history, "CACHE_DIR", tmp_path)
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return _chart_response([1758758400], [100.0], [102.0], [99.0], [101.0], [1000])

    monkeypatch.setattr(price_history.requests, "get", lambda *a, **k: FakeResponse())

    first = price_history._fetch_chart("AAPL", "max")
    second = price_history._fetch_chart("AAPL", "max")

    assert first == second
    assert len(calls) == 1  # second call served from disk cache


def test_fetch_chart_force_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(price_history, "CACHE_DIR", tmp_path)
    calls = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            calls.append(1)
            return _chart_response([1758758400], [100.0], [102.0], [99.0], [101.0], [1000])

    monkeypatch.setattr(price_history.requests, "get", lambda *a, **k: FakeResponse())

    price_history._fetch_chart("AAPL", "5d")
    price_history._fetch_chart("AAPL", "5d", force=True)

    assert len(calls) == 2  # force=True bypasses the cache
