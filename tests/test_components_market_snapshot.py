from __future__ import annotations

import datetime as dt

import pandas as pd

from stock_predictor.dashboard.components import FIFTY_TWO_WEEK_DAYS, market_snapshot


def _bar(date, low, high, close, volume=1000):
    return {"date": date, "low": low, "high": high, "close": close, "volume": volume}


def testmarket_snapshot_empty_df_returns_all_none():
    snapshot = market_snapshot(pd.DataFrame(columns=["date", "low", "high", "close", "volume"]))
    assert snapshot == {"Price": None, "Change": None, "% Change": None, "Volume": None, "52W Low": None, "52W High": None}


def testmarket_snapshot_computes_price_change_and_volume():
    df = pd.DataFrame(
        [
            _bar(dt.date(2026, 1, 1), 95, 105, 100.0, volume=500),
            _bar(dt.date(2026, 1, 2), 100, 112, 110.0, volume=800),
        ]
    )
    snapshot = market_snapshot(df)

    assert snapshot["Price"] == 110.0
    assert snapshot["Change"] == 10.0
    assert snapshot["% Change"] == 0.1
    assert snapshot["Volume"] == 800


def testmarket_snapshot_single_bar_has_no_change():
    df = pd.DataFrame([_bar(dt.date(2026, 1, 1), 95, 105, 100.0)])
    snapshot = market_snapshot(df)

    assert snapshot["Price"] == 100.0
    assert snapshot["Change"] is None
    assert snapshot["% Change"] is None


def testmarket_snapshot_fifty_two_week_range_excludes_older_bars():
    last_date = dt.date(2026, 6, 1)
    df = pd.DataFrame(
        [
            _bar(last_date - dt.timedelta(days=FIFTY_TWO_WEEK_DAYS + 30), 1.0, 500.0, 250.0),  # outside the window
            _bar(last_date - dt.timedelta(days=30), 90.0, 120.0, 110.0),
            _bar(last_date, 95.0, 115.0, 112.0),
        ]
    )
    snapshot = market_snapshot(df)

    assert snapshot["52W Low"] == 90.0
    assert snapshot["52W High"] == 120.0
