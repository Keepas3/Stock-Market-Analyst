from __future__ import annotations

import datetime as dt

from stock_predictor.ingest import competitors
from stock_predictor.ingest.fundamentals import FundamentalsResult
from stock_predictor.ingest.price_history import PriceBar


def _bar(date, close, low=None, high=None, volume=1000):
    return PriceBar(
        date=date,
        open=close,
        high=high if high is not None else close + 1,
        low=low if low is not None else close - 1,
        close=close,
        volume=volume,
    )


def _overview(**overrides):
    defaults = dict(
        ticker="AMD",
        pe_ratio=156.0,
        forward_pe=140.0,
        sector="TECHNOLOGY",
        name="Advanced Micro Devices, Inc.",
        description="desc",
        industry="SEMICONDUCTORS",
        exchange="NASDAQ",
        country="USA",
        address="addr",
        official_site="https://www.amd.com",
        market_cap=992_330_000_000.0,
    )
    defaults.update(overrides)
    return FundamentalsResult(**defaults)


def test_fetch_competitor_snapshot_combines_price_and_overview(monkeypatch):
    bars = [
        _bar(dt.date(2026, 1, 1), 600.0, low=590.0, high=610.0),
        _bar(dt.date(2026, 1, 2), 607.57, low=600.0, high=615.0),
    ]
    monkeypatch.setattr(competitors.price_history, "fetch_full_history", lambda ticker: bars)
    monkeypatch.setattr(competitors, "fetch_overview", lambda ticker: _overview())

    snapshot = competitors.fetch_competitor_snapshot("AMD")

    assert snapshot is not None
    assert snapshot.ticker == "AMD"
    assert snapshot.name == "Advanced Micro Devices, Inc."
    assert snapshot.price == 607.57
    assert snapshot.change == 607.57 - 600.0
    assert snapshot.market_cap == 992_330_000_000.0
    assert snapshot.pe_ratio == 156.0


def test_fetch_competitor_snapshot_returns_none_when_no_price_history(monkeypatch):
    monkeypatch.setattr(competitors.price_history, "fetch_full_history", lambda ticker: [])
    snapshot = competitors.fetch_competitor_snapshot("NOPE")
    assert snapshot is None


def test_fetch_competitor_snapshot_degrades_gracefully_when_overview_fails(monkeypatch):
    bars = [_bar(dt.date(2026, 1, 1), 100.0)]
    monkeypatch.setattr(competitors.price_history, "fetch_full_history", lambda ticker: bars)
    monkeypatch.setattr(competitors, "fetch_overview", lambda ticker: None)

    snapshot = competitors.fetch_competitor_snapshot("AMD")

    assert snapshot is not None
    assert snapshot.price == 100.0
    assert snapshot.name is None
    assert snapshot.market_cap is None
    assert snapshot.pe_ratio is None


def test_fetch_competitor_snapshot_single_bar_has_no_change(monkeypatch):
    bars = [_bar(dt.date(2026, 1, 1), 100.0)]
    monkeypatch.setattr(competitors.price_history, "fetch_full_history", lambda ticker: bars)
    monkeypatch.setattr(competitors, "fetch_overview", lambda ticker: None)

    snapshot = competitors.fetch_competitor_snapshot("AMD")

    assert snapshot.change is None
    assert snapshot.pct_change is None


def test_fetch_competitor_snapshot_fifty_two_week_range_excludes_older_bars(monkeypatch):
    last_date = dt.date(2026, 6, 1)
    bars = [
        _bar(last_date - dt.timedelta(days=competitors.FIFTY_TWO_WEEK_DAYS + 30), 250.0, low=1.0, high=500.0),
        _bar(last_date - dt.timedelta(days=30), 110.0, low=90.0, high=120.0),
        _bar(last_date, 112.0, low=95.0, high=115.0),
    ]
    monkeypatch.setattr(competitors.price_history, "fetch_full_history", lambda ticker: bars)
    monkeypatch.setattr(competitors, "fetch_overview", lambda ticker: None)

    snapshot = competitors.fetch_competitor_snapshot("AMD")

    assert snapshot.fifty_two_week_low == 90.0
    assert snapshot.fifty_two_week_high == 120.0
