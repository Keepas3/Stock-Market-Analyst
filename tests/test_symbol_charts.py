from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_predictor.dashboard.symbol_charts import (
    DOWN_COLOR,
    NO_DATA,
    POSITION_COLUMN,
    UP_COLOR,
    available_sectors,
    filter_symbols,
    missing_count,
    symbols_chart,
    with_derived_metrics,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        [
            dict(symbol_id=1, Ticker="AAPL", Name="Apple Inc.", Sector="Technology", Price=150.0, Change=2.0, **{"% Change": 0.0135}, Volume=5e7, **{"52W Low": 100.0, "52W High": 200.0}, **{"P/E": 30.0, "News Sentiment": 0.3, "Social Sentiment": 0.2}, Recommendation="Buy"),
            dict(symbol_id=2, Ticker="MSFT", Name="Microsoft Corporation", Sector="Technology", Price=300.0, Change=-3.0, **{"% Change": -0.01}, Volume=2e7, **{"52W Low": 250.0, "52W High": 300.0}, **{"P/E": 35.0, "News Sentiment": -0.2, "Social Sentiment": 0.1}, Recommendation="Hold"),
            dict(symbol_id=3, Ticker="XOM", Name="Exxon Mobil", Sector="Energy", Price=100.0, Change=0.5, **{"% Change": 0.005}, Volume=1e7, **{"52W Low": 100.0, "52W High": 100.0}, **{"P/E": np.nan, "News Sentiment": 0.0, "Social Sentiment": np.nan}, Recommendation="Sell"),
            dict(symbol_id=4, Ticker="NEWCO", Name="New Co (Beta)", Sector=None, Price=10.0, Change=np.nan, **{"% Change": np.nan}, Volume=1e5, **{"52W Low": 5.0, "52W High": 20.0}, **{"P/E": 12.0, "News Sentiment": np.nan, "Social Sentiment": np.nan}, Recommendation=None),
        ]
    )


def test_available_sectors_sorted_and_without_blanks(df):
    assert available_sectors(df) == ["Energy", "Technology"]


def test_filter_search_matches_ticker_or_name_case_insensitively(df):
    assert filter_symbols(df, search="apple")["Ticker"].tolist() == ["AAPL"]
    assert filter_symbols(df, search="xo")["Ticker"].tolist() == ["XOM"]


def test_filter_search_is_not_a_regex(df):
    assert filter_symbols(df, search="(Beta)")["Ticker"].tolist() == ["NEWCO"]
    assert filter_symbols(df, search="(").shape[0] == 1


def test_filter_by_sector_and_recommendation_combine(df):
    assert filter_symbols(df, sectors=["Technology"])["Ticker"].tolist() == ["AAPL", "MSFT"]
    assert filter_symbols(df, sectors=["Technology"], recommendations=["Buy"])["Ticker"].tolist() == ["AAPL"]
    assert filter_symbols(df, recommendations=["Buy", "Sell"])["Ticker"].tolist() == ["AAPL", "XOM"]


def test_empty_filters_mean_no_filter(df):
    assert len(filter_symbols(df, "", [], [])) == len(df)


def test_52_week_position(df):
    out = with_derived_metrics(df)
    assert out[POSITION_COLUMN].iloc[0] == pytest.approx(0.5)
    assert out[POSITION_COLUMN].iloc[1] == pytest.approx(1.0)
    assert np.isnan(out[POSITION_COLUMN].iloc[2])  # zero-width range -> no position
    assert "Price" in df.columns and POSITION_COLUMN not in df.columns  # input untouched


def test_missing_count(df):
    assert missing_count(df, ["% Change"]) == 1
    assert missing_count(df, ["P/E", "% Change"]) == 2
    assert missing_count(df, ["Price"]) == 0


def test_diverging_bar_is_sorted_and_colored_by_sign(df):
    fig = symbols_chart(df, "Bar", metric="% Change")
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert list(trace.x) == ["AAPL", "XOM", "MSFT"]  # NEWCO has no % Change
    assert list(trace.marker.color) == [UP_COLOR, UP_COLOR, DOWN_COLOR]
    assert [c[0] for c in trace.customdata] == [1, 3, 2]
    assert list(fig.layout.xaxis.categoryarray) == ["AAPL", "XOM", "MSFT"]


def test_non_diverging_bar_has_one_trace_per_sector_in_global_order(df):
    fig = symbols_chart(df, "Bar", metric="P/E", color_by="Sector")
    assert {t.name for t in fig.data} == {"Technology", "n/a"}
    assert list(fig.layout.xaxis.categoryarray) == ["MSFT", "AAPL", "NEWCO"]  # sorted by P/E desc, XOM dropped


def test_bar_can_color_by_recommendation(df):
    fig = symbols_chart(df, "Bar", metric="Price", color_by="Recommendation")
    assert {t.name for t in fig.data} == {"Buy", "Hold", "Sell", NO_DATA}


def test_scatter_drops_rows_missing_either_axis_and_labels_points(df):
    fig = symbols_chart(df, "Scatter", x="P/E", y="% Change", color_by="Sector")
    tickers = sorted(t for trace in fig.data for t in trace.text)
    assert tickers == ["AAPL", "MSFT"]  # XOM has no P/E, NEWCO has no % Change
    assert fig.layout.xaxis.title.text == "P/E" and fig.layout.yaxis.title.text == "% Change"


def test_scatter_custom_data_carries_symbol_id_for_click_selection(df):
    fig = symbols_chart(df, "Scatter", x="Price", y="Volume", color_by="Recommendation")
    ids = sorted(point[0] for trace in fig.data for point in trace.customdata)
    assert ids == [1, 2, 3, 4]


def test_52w_position_is_chartable(df):
    fig = symbols_chart(df, "Bar", metric=POSITION_COLUMN)
    assert list(fig.layout.xaxis.categoryarray) == ["MSFT", "AAPL", "NEWCO"]


def test_screen_context_snapshot(df):
    from stock_predictor.dashboard.symbol_charts import screen_context

    filtered = filter_symbols(df, sectors=["Technology"])
    selected = filtered.iloc[0].to_dict()
    snap = screen_context(
        "Companies", "Chart", "", ["Technology"], [], filtered, len(df),
        {"kind": "Bar", "metric": "% Change", "color_by": None}, selected,
    )
    assert snap["page"] == "Companies" and snap["view"] == "Chart"
    assert snap["filters"] == {"search": None, "sectors": ["Technology"], "recommendations": []}
    assert snap["companies_visible"] == 2 and snap["companies_total"] == 4
    assert snap["visible_tickers"] == ["AAPL", "MSFT"]
    assert snap["selected_company"] == {"ticker": "AAPL", "name": "Apple Inc."}
    assert snap["chart"]["metric"] == "% Change"


def test_screen_context_hides_chart_settings_in_table_view(df):
    from stock_predictor.dashboard.symbol_charts import screen_context

    snap = screen_context("Watchlist", "Table", "app", [], ["Buy"], df.head(1), len(df), {"kind": "Bar"}, None)
    assert snap["chart"] is None and snap["selected_company"] is None
    assert snap["filters"]["search"] == "app" and snap["filters"]["recommendations"] == ["Buy"]
