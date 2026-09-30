from __future__ import annotations

from stock_predictor.dashboard.components import COMPETITORS_DISPLAY_COLUMNS, _format_money, color_by_sign


def test_color_by_sign_positive_is_green():
    assert "16a34a" in color_by_sign(1.5)


def test_color_by_sign_negative_is_red():
    assert "dc2626" in color_by_sign(-1.5)


def test_color_by_sign_zero_is_blank():
    assert color_by_sign(0) == ""


def test_color_by_sign_none_is_blank():
    import pandas as pd

    assert color_by_sign(pd.NA) == ""


def test_format_money_trillions():
    assert _format_money(5_510_000_000_000) == "$5.51T"


def test_format_money_nan_is_na():
    """Regression test: a pandas column built from a mix of real numbers
    and None (e.g. render_competitors_table's Market Cap column when one
    row has no fundamentals data) upcasts None to float NaN, not None --
    confirmed live, this produced a literal "$nan" before the pd.isna
    check was added.
    """
    import pandas as pd

    assert _format_money(float("nan")) == "n/a"
    assert _format_money(pd.NA) == "n/a"


def test_competitors_display_columns_includes_market_cap_and_pe():
    assert "Market Cap" in COMPETITORS_DISPLAY_COLUMNS
    assert "P/E" in COMPETITORS_DISPLAY_COLUMNS
    assert COMPETITORS_DISPLAY_COLUMNS[0] == "Ticker"
