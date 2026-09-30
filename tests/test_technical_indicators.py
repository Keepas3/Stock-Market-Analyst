from __future__ import annotations

import pandas as pd

from stock_predictor.model.technical_indicators import (
    PE_EXPENSIVE_THRESHOLD,
    PE_VALUE_THRESHOLD,
    SMA_LONG_WINDOW,
    SMA_SHORT_WINDOW,
    compute_moving_averages,
    ma_signal,
    pe_signal,
)


def _price_df(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


def test_compute_moving_averages_matches_known_synthetic_series():
    # SMA_LONG_WINDOW bars of a constant 10.0, so both windows' mean is 10.0.
    df = _price_df([10.0] * SMA_LONG_WINDOW)
    sma_short, sma_long = compute_moving_averages(df)

    assert sma_short == 10.0
    assert sma_long == 10.0


def test_compute_moving_averages_none_when_not_enough_history():
    df = _price_df([10.0] * (SMA_SHORT_WINDOW - 1))
    sma_short, sma_long = compute_moving_averages(df)

    assert sma_short is None
    assert sma_long is None


def test_compute_moving_averages_short_available_long_not():
    df = _price_df([10.0] * SMA_SHORT_WINDOW)
    sma_short, sma_long = compute_moving_averages(df)

    assert sma_short == 10.0
    assert sma_long is None


def test_ma_signal_golden_cross_is_bullish():
    assert ma_signal(sma_short=110.0, sma_long=100.0) == 1


def test_ma_signal_death_cross_is_bearish():
    assert ma_signal(sma_short=90.0, sma_long=100.0) == -1


def test_ma_signal_equal_is_neutral():
    assert ma_signal(sma_short=100.0, sma_long=100.0) == 0


def test_ma_signal_none_is_neutral():
    assert ma_signal(sma_short=None, sma_long=100.0) == 0
    assert ma_signal(sma_short=100.0, sma_long=None) == 0


def test_pe_signal_below_value_threshold_is_bullish():
    assert pe_signal(PE_VALUE_THRESHOLD - 1) == 1


def test_pe_signal_above_expensive_threshold_is_bearish():
    assert pe_signal(PE_EXPENSIVE_THRESHOLD + 1) == -1


def test_pe_signal_between_thresholds_is_neutral():
    midpoint = (PE_VALUE_THRESHOLD + PE_EXPENSIVE_THRESHOLD) / 2
    assert pe_signal(midpoint) == 0


def test_pe_signal_none_is_neutral():
    assert pe_signal(None) == 0


def test_pe_signal_negative_or_zero_is_neutral():
    assert pe_signal(-5.0) == 0
    assert pe_signal(0.0) == 0
