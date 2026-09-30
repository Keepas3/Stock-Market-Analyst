from __future__ import annotations

import datetime as dt

from stock_predictor.dashboard.components import _format_money, _percent_change, financials_bar_chart


def _quarter(fiscal_date_ending, total_revenue, net_income):
    return {"fiscal_date_ending": fiscal_date_ending, "total_revenue": total_revenue, "net_income": net_income}


def test_format_money_billions():
    assert _format_money(96_220_000_000) == "$96.22B"


def test_format_money_millions():
    assert _format_money(21_400_000) == "$21.40M"


def test_format_money_small_value():
    assert _format_money(1234) == "$1,234"


def test_format_money_negative():
    assert _format_money(-1_500_000_000) == "-$1.50B"


def test_format_money_none_is_na():
    assert _format_money(None) == "n/a"


def test_percent_change_positive():
    assert _percent_change(120, 100) == 0.2


def test_percent_change_negative():
    assert _percent_change(80, 100) == -0.2


def test_percent_change_none_current_is_none():
    assert _percent_change(None, 100) is None


def test_percent_change_none_or_zero_previous_is_none():
    assert _percent_change(100, None) is None
    assert _percent_change(100, 0) is None


def test_financials_bar_chart_labels_by_month_year():
    quarters = [
        _quarter(dt.date(2025, 9, 30), 80.0, 40.0),
        _quarter(dt.date(2025, 12, 31), 88.0, 45.0),
    ]
    fig = financials_bar_chart(quarters, "NVDA")

    assert list(fig.data[0].x) == ["Sep 2025", "Dec 2025"]
    assert list(fig.data[0].y) == [80.0, 88.0]
    assert fig.data[0].name == "Total Revenue"
    assert fig.data[1].name == "Net Income"
