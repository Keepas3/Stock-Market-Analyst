from __future__ import annotations

import datetime as dt

import pandas as pd

from stock_predictor.dashboard.components import PRICE_CHART_RANGES, price_chart


def _price_df(n: int) -> pd.DataFrame:
    base = dt.date(2026, 1, 1)
    return pd.DataFrame(
        {
            "date": [base + dt.timedelta(days=i) for i in range(n)],
            "close": [float(100 + i) for i in range(n)],
        }
    )


def test_price_chart_all_shows_every_bar():
    df = _price_df(50)
    fig = price_chart(df, "AAPL", "All")
    assert len(fig.data[0].x) == 50


def test_price_chart_5d_shows_only_last_five_bars():
    df = _price_df(50)
    fig = price_chart(df, "AAPL", "5D")
    assert len(fig.data[0].x) == 5
    assert fig.data[0].y[-1] == 149.0  # the most recent close


def test_price_chart_14d_shows_only_last_fourteen_bars():
    df = _price_df(50)
    fig = price_chart(df, "AAPL", "14D")
    assert len(fig.data[0].x) == 14


def test_price_chart_1m_shows_last_21_bars():
    df = _price_df(50)
    fig = price_chart(df, "AAPL", "1M")
    assert len(fig.data[0].x) == PRICE_CHART_RANGES["1M"]


def test_price_chart_has_no_1d_range():
    """See PRICE_CHART_RANGES's own comment -- this app is daily-bars-only,
    so a "1 day" range would always be exactly one point, and Plotly's
    default datetime axis auto-range renders that badly (confirmed live: a
    sub-millisecond-wide x-axis with fractional-second tick labels).
    """
    assert "1D" not in PRICE_CHART_RANGES


def test_price_chart_single_point_sets_an_explicit_axis_range():
    """A single-bar window (e.g. a freshly-added symbol with only one
    day of history so far, on any range) must not hit the same Plotly
    single-point auto-range bug "1D" used to.
    """
    df = _price_df(1)
    fig = price_chart(df, "AAPL", "All")

    assert len(fig.data[0].x) == 1
    assert fig.layout.xaxis.range is not None
    start, end = fig.layout.xaxis.range
    assert start < end


def test_price_chart_multi_point_leaves_axis_range_on_auto():
    df = _price_df(5)
    fig = price_chart(df, "AAPL", "5D")
    assert fig.layout.xaxis.range is None


def test_price_chart_range_narrower_than_history_still_works():
    df = _price_df(3)  # fewer bars than the 5D window
    fig = price_chart(df, "AAPL", "5D")
    assert len(fig.data[0].x) == 3


def test_price_chart_unknown_range_label_falls_back_to_all():
    df = _price_df(50)
    fig = price_chart(df, "AAPL", "not-a-real-range")
    assert len(fig.data[0].x) == 50


def test_price_chart_y_axis_has_dollar_prefix():
    fig = price_chart(_price_df(10), "AAPL", "All")
    assert fig.layout.yaxis.tickprefix == "$"


def test_price_chart_hover_includes_dollar_sign():
    fig = price_chart(_price_df(10), "AAPL", "All")
    assert "$" in fig.data[0].hovertemplate
